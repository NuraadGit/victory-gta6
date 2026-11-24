//! Console wave64 compute programs on 32-lane hosts with two console lanes in each host lane
//! ([`super::lower::with_paired_lanes`]), run on the hardware device and on WARP and compared
//! with a reference: the same machine code run instruction by instruction over 64 lanes
//! ([`Reference`]).
//!
//! The programs are the cases the exact forms refused and the approximate one got wrong: a
//! decision of the whole wave inside a per-lane branch, in thread groups of several console
//! waves that decide differently; a lane mask walked bit by bit; reductions and an allocation
//! by lane 63; loops the waves of a group leave at different rounds. And random ones.

use super::host::{Dispatch, Host, Root, View};
use super::{HostShader, lower, validator};
use crate::gfx10::{Dpp, Fields, NULL, Operand};
use crate::ir::{self, Exit, Inst};
use crate::native::{ComputeIo, HwStage, NativeProgram, ResourceType, RootParam, RootType, StageIo, UsedResource};
use crate::{ShaderError, io, resource, structure};

const S_ENDPGM: u32 = 0xBF81_0000;
/// Dwords each thread stores at the end.
const OUT: u32 = 4;
/// Dwords of the UAV past the threads' own (a counter and slots).
const TAIL: u32 = 256;

// ---- a small assembler ----

const VCC: u32 = 106;
const EXEC: u32 = 126;
const ONES: u32 = 193;
/// An inline constant 0..64.
const fn k(n: u32) -> u32 {
    128 + n
}
/// A VGPR as a source.
const fn v(n: u32) -> u32 {
    256 + n
}

#[derive(Default)]
struct Asm {
    words: Vec<u32>,
    labels: Vec<Option<usize>>,
    /// (the branch's word, its label)
    fix: Vec<(usize, usize)>,
}

impl Asm {
    fn label(&mut self) -> usize {
        self.labels.push(None);
        self.labels.len() - 1
    }
    fn here(&mut self, l: usize) {
        self.labels[l] = Some(self.words.len());
    }
    fn sop1(&mut self, op: u32, d: u32, s: u32) {
        self.words.push(0xBE80_0000 | d << 16 | op << 8 | s);
    }
    fn sop2(&mut self, op: u32, d: u32, a: u32, b: u32) {
        self.words.push(0x8000_0000 | op << 23 | d << 16 | b << 8 | a);
    }
    fn sopc(&mut self, op: u32, a: u32, b: u32) {
        self.words.push(0xBF00_0000 | op << 16 | b << 8 | a);
    }
    fn branch(&mut self, op: u32, l: usize) {
        self.fix.push((self.words.len(), l));
        self.words.push(0xBF80_0000 | op << 16);
    }
    fn vop1(&mut self, op: u32, d: u32, a: u32) {
        self.words.push(0x7E00_0000 | d << 17 | op << 9 | a);
    }
    fn vop2(&mut self, op: u32, d: u32, a: u32, b: u32) {
        self.words.push(op << 25 | d << 17 | b << 9 | a);
    }
    fn vopc(&mut self, op: u32, a: u32, b: u32) {
        self.words.push(0x7C00_0000 | op << 17 | b << 9 | a);
    }
    fn vop3(&mut self, op: u32, d: u32, a: u32, b: u32, c: u32) {
        self.words.push(0xD400_0000 | op << 16 | d);
        self.words.push(a | b << 9 | c << 18);
    }
    /// buffer_store_dword v[d], v[a], s[8:11], 0 offen offset
    fn store(&mut self, d: u32, a: u32, offset: u32) {
        self.words.push(0xE000_0000 | 0x1C << 18 | 1 << 12 | offset);
        self.words.push(0x80 << 24 | 2 << 16 | d << 8 | a);
    }
    /// buffer_atomic_add v[d], s[8:11], offset glc
    fn atomic_add(&mut self, d: u32, offset: u32) {
        self.words.push(0xE000_0000 | 0x32 << 18 | 1 << 14 | offset);
        self.words.push(0x80 << 24 | 2 << 16 | d << 8);
    }
    // what the programs use most
    fn v_mov(&mut self, d: u32, a: u32) {
        self.vop1(1, d, a);
    }
    fn v_add(&mut self, d: u32, a: u32, b: u32) {
        self.vop2(0x25, d, a, b);
    }
    fn s_mov64(&mut self, d: u32, a: u32) {
        self.sop1(4, d, a);
    }
    /// s_and_saveexec_b64
    fn and_save(&mut self, d: u32, a: u32) {
        self.sop1(0x24, d, a);
    }
    /// The compiler's `WaveActiveBitOr(v7)` into s7 (v1, v2 and s[4:5] are its own).
    fn wave_or(&mut self) {
        self.words.extend_from_slice(&super::wave::WAVE_OR_V7_TO_S7);
    }
    /// `WaveActiveSum` of v7 into s7.
    fn wave_sum(&mut self) {
        // v_add_nc_u32 (0x25) for v_or_b32 (0x1C)
        self.words.extend(
            super::wave::WAVE_OR_V7_TO_S7
                .iter()
                .map(|&w| if w >> 25 == 0x1C { w & 0x01FF_FFFF | 0x25 << 25 } else { w }),
        );
    }
    /// The compiler's `WavePrefixSum(v7)` into v8: row shifts, the other row's last lane, then
    /// lane 31 for the upper half, less the lane's own term.
    fn wave_prefix(&mut self) {
        self.sop1(0x25, 4, ONES); // s_or_saveexec_b64 s[4:5], -1
        self.vop3(0x101, 1, k(0), v(7), 4); // v_cndmask_b32 v1, 0, v7, s[4:5]
        self.v_mov(2, v(1));
        for (ctrl, bound) in [(0x111u32, 1u32), (0x112, 0), (0x114, 0), (0x118, 0)] {
            self.vop2(0x25, 2, 250, 2);
            self.words.push(0xFF00_0000 | bound << 19 | ctrl << 8 | 2);
        }
        self.words.extend([0xBEFE_03FF, 0xFFFF_0000, 0xBEFF_03FF, 0xFFFF_0000]); // exec = odd rows
        self.words.extend([0xD778_4803, 0x0305_8302]); // v_permlanex16_b32 v3, v2, -1, -1 fi:1
        self.v_add(2, v(3), 2);
        self.words.extend([0xD760_0007, 0x0001_3F02]); // v_readlane_b32 s7, v2, 31
        self.words.extend([0xBEFE_0380, 0xBEFF_03C1]); // exec = lanes 32..63
        self.v_add(2, 7, 2);
        self.words.extend([0xBEFE_03C1, 0xBEFF_03C1]); // exec = -1
        self.vop2(0x26, 2, v(2), 1); // v_sub_nc_u32 v2, v2, v1
        self.s_mov64(EXEC, 4);
        self.v_mov(8, v(2));
    }
    fn finish(mut self) -> Vec<u32> {
        for (at, l) in self.fix {
            let to = self.labels[l].expect("a label placed") as i32;
            self.words[at] |= (to - at as i32 - 1) as u32 & 0xFFFF;
        }
        self.words
    }
}

/// The program: the UAV's descriptor, `body`, then v4..v7 of every thread stored at its own
/// dwords (the thread index is in v0).
fn native(threads: u32, body: impl FnOnce(&mut Asm)) -> NativeProgram {
    native_from(threads, 0, body)
}

/// The same with threads numbered from `first` on.
fn native_from(threads: u32, first: u32, body: impl FnOnce(&mut Asm)) -> NativeProgram {
    let mut a = Asm::default();
    a.words.extend([
        0xBE85_0380, // s_mov_b32 s5, 0
        0x3D << 26 | 0x02 << 18 | 8 << 6 | 2,
        125 << 25, // s_load_dwordx4 s[8:11], s[4:5], 0
    ]);
    if first != 0 {
        a.v_add(0, 255, 0);
        a.words.push(first);
    }
    body(&mut a);
    a.vop2(0x1A, 9, k(4), 0); // v_lshlrev_b32 v9, 4, v0
    for n in 0..OUT {
        a.store(4 + n, 9, 4 * n);
    }
    a.words.push(S_ENDPGM);
    let code = a.finish();
    NativeProgram {
        stage: HwStage::Cs,
        wave64: true,
        vgprs: 16,
        layout: vec![
            RootParam {
                root: RootType::Constants,
                table_offset_dwords: 0,
                sgprs: 4,
                base_register: 0,
                user_data: 0,
                slot: 0,
            },
            RootParam {
                root: RootType::Table,
                table_offset_dwords: 0,
                sgprs: 1,
                base_register: 0,
                user_data: 4,
                slot: 1,
            },
        ],
        used: vec![UsedResource {
            kind: ResourceType::Uav,
            root: RootType::Table,
            table_offset_dwords: 0,
            lower_bound: 0,
            stride: 0,
            dwords: 4,
            user_data: 4,
            slot: 1,
        }],
        io: StageIo::Cs(ComputeIo {
            rsrc2: 5 << 1,
            threads: [threads, 1, 1],
        }),
        code: code.iter().flat_map(|w| w.to_le_bytes()).collect(),
    }
}

// ---- the reference ----

/// The machine code of one console wave, run over its 64 lanes as the console runs it: scalars
/// and EXEC for the wave, vector instructions in the lanes EXEC holds.
struct Reference {
    v: Vec<[u32; 64]>,
    s: [u32; 128],
    scc: bool,
}

fn run_reference(n: &NativeProgram, threads: u32, constants: [u32; 4]) -> Vec<u32> {
    let p = ir::build(&n.code, true, n.vgprs).expect("ir::build");
    let mut mem = vec![0u32; (threads * OUT + TAIL) as usize];
    let StageIo::Cs(cs) = n.io else { unreachable!() };
    for wave in 0..cs.threads[0] / 64 {
        let mut r = Reference { v: vec![[0; 64]; 32], s: [0; 128], scc: false };
        r.s[..4].copy_from_slice(&constants);
        r.s[126] = u32::MAX;
        r.s[127] = u32::MAX;
        for l in 0..64 {
            r.v[0][l] = wave * 64 + l as u32;
        }
        let mut b = 0;
        for step in 0.. {
            assert!(step < 100_000, "the reference does not end");
            let mut taken = None;
            for i in &p.blocks[b].insts {
                taken = r.step(i, &mut mem).or(taken);
            }
            match p.blocks[b].exit {
                Exit::Jump(t) => b = t,
                Exit::Branch { taken: t, not_taken } => b = if taken.expect("a branch") { t } else { not_taken },
                Exit::End => break,
                Exit::Indirect => panic!("an indirect jump"),
            }
        }
    }
    mem
}

impl Reference {
    fn pair(&self, code: u32) -> u64 {
        u64::from(self.s[code as usize]) | u64::from(self.s[code as usize + 1]) << 32
    }
    fn set_pair(&mut self, code: u32, x: u64) {
        self.s[code as usize] = x as u32;
        self.s[code as usize + 1] = (x >> 32) as u32;
    }
    fn scalar(&self, o: &Operand) -> u32 {
        match *o {
            Operand::Scalar { code: NULL, .. } => 0,
            Operand::Scalar { code, .. } => self.s[usize::from(code)],
            Operand::Const(c) => resource::inline(c).expect("an inline constant"),
            Operand::Literal(x) => x,
            _ => panic!("{o:?}"),
        }
    }
    fn scalar64(&self, o: &Operand) -> u64 {
        match *o {
            Operand::Scalar { code, .. } => self.pair(code.into()),
            Operand::Const(_) => i64::from(self.scalar(o) as i32) as u64,
            _ => panic!("{o:?}"),
        }
    }
    fn src(&self, o: &Operand, lane: usize) -> u32 {
        match *o {
            Operand::Vector { reg, .. } => self.v[usize::from(reg)][lane],
            _ => self.scalar(o),
        }
    }

    /// One instruction; a conditional branch's decision.
    fn step(&mut self, i: &Inst, mem: &mut [u32]) -> Option<bool> {
        let name = i.name();
        let exec = self.pair(EXEC);
        match &i.fields {
            Fields::Sopp { .. } => {
                let vcc = self.pair(VCC);
                return Some(match name {
                    "s_cbranch_scc0" => !self.scc,
                    "s_cbranch_scc1" => self.scc,
                    "s_cbranch_vccz" => vcc == 0,
                    "s_cbranch_vccnz" => vcc != 0,
                    "s_cbranch_execz" => exec == 0,
                    "s_cbranch_execnz" => exec != 0,
                    _ => return None,
                });
            }
            Fields::Smem { .. } => {} // the descriptor
            Fields::Sop { dst, src, .. } => {
                let code = match dst {
                    Some(Operand::Scalar { code, .. }) => u32::from(*code),
                    _ => 125,
                };
                let a = |r: &Self, n: usize| r.scalar(&src[n]);
                let w = |r: &Self, n: usize| r.scalar64(&src[n]);
                let (mut one, mut two) = (None, None);
                match name {
                    "s_mov_b32" => one = Some(a(self, 0)),
                    "s_mov_b64" => two = Some(w(self, 0)),
                    "s_and_b64" | "s_or_b64" | "s_xor_b64" | "s_andn2_b64" => {
                        let (x, y) = (w(self, 0), w(self, 1));
                        let r = match name {
                            "s_and_b64" => x & y,
                            "s_or_b64" => x | y,
                            "s_xor_b64" => x ^ y,
                            _ => x & !y,
                        };
                        self.scc = r != 0;
                        two = Some(r);
                    }
                    "s_and_saveexec_b64" | "s_or_saveexec_b64" => {
                        let x = w(self, 0);
                        let r = if name.starts_with("s_and") { x & exec } else { x | exec };
                        self.set_pair(code, exec);
                        self.set_pair(EXEC, r);
                        self.scc = r != 0;
                    }
                    "s_xor_b32" | "s_and_b32" | "s_or_b32" => {
                        let (x, y) = (a(self, 0), a(self, 1));
                        let r = match name {
                            "s_xor_b32" => x ^ y,
                            "s_and_b32" => x & y,
                            _ => x | y,
                        };
                        self.scc = r != 0;
                        one = Some(r);
                    }
                    "s_add_u32" => {
                        let (r, c) = a(self, 0).overflowing_add(a(self, 1));
                        self.scc = c;
                        one = Some(r);
                    }
                    "s_lshl_b32" => {
                        let r = a(self, 0) << (a(self, 1) & 31);
                        self.scc = r != 0;
                        one = Some(r);
                    }
                    "s_bcnt1_i32_b64" => {
                        let r = w(self, 0).count_ones();
                        self.scc = r != 0;
                        one = Some(r);
                    }
                    "s_ff1_i32_b32" => {
                        let x = a(self, 0);
                        one = Some(if x == 0 { u32::MAX } else { x.trailing_zeros() });
                    }
                    "s_ff1_i32_b64" => {
                        let x = w(self, 0);
                        one = Some(if x == 0 { u32::MAX } else { x.trailing_zeros() });
                    }
                    "s_cselect_b32" => one = Some(if self.scc { a(self, 0) } else { a(self, 1) }),
                    "s_cselect_b64" => two = Some(if self.scc { w(self, 0) } else { w(self, 1) }),
                    "s_cmp_eq_u32" => self.scc = a(self, 0) == a(self, 1),
                    "s_cmp_lg_u32" => self.scc = a(self, 0) != a(self, 1),
                    "s_cmp_lt_u32" => self.scc = a(self, 0) < a(self, 1),
                    _ => panic!("the reference has no {name}"),
                }
                if let Some(x) = one {
                    self.s[code as usize] = x;
                }
                if let Some(x) = two {
                    self.set_pair(code, x);
                }
            }
            Fields::Valu(f) => self.valu(name, f, exec),
            Fields::Buffer { data, addr, offset, offen, cache, .. } => {
                let (Some(Operand::Vector { reg: d, .. }), Operand::Vector { reg: ad, .. }) = (data, addr) else {
                    panic!("a buffer access")
                };
                for l in (0..64).filter(|l| exec >> l & 1 != 0) {
                    let base = if *offen { self.v[usize::from(*ad)][l] } else { 0 };
                    let at = (base + u32::from(*offset)) as usize / 4;
                    let x = self.v[usize::from(*d)][l];
                    match name {
                        "buffer_store_dword" => mem[at] = x,
                        "buffer_atomic_add" => {
                            assert!(cache.glc);
                            self.v[usize::from(*d)][l] = mem[at];
                            mem[at] = mem[at].wrapping_add(x);
                        }
                        _ => panic!("the reference has no {name}"),
                    }
                }
            }
            f => panic!("the reference has no {name}: {f:?}"),
        }
        None
    }

    fn valu(&mut self, name: &str, f: &crate::gfx10::Valu, exec: u64) {
        let on = |l: usize| exec >> l & 1 != 0;
        if let Some(rest) = name.strip_prefix("v_cmp_").or(name.strip_prefix("v_cmpx_")) {
            let mut r = 0u64;
            for l in (0..64).filter(|&l| on(l)) {
                let (x, y) = (self.src(&f.src[0], l), self.src(&f.src[1], l));
                let hit = match rest {
                    "eq_u32" => x == y,
                    "ne_u32" => x != y,
                    "lt_u32" => x < y,
                    "le_u32" => x <= y,
                    "gt_u32" => x > y,
                    "ge_u32" => x >= y,
                    _ => panic!("the reference has no {name}"),
                };
                r |= u64::from(hit) << l;
            }
            if name.starts_with("v_cmpx") {
                self.set_pair(EXEC, r);
            } else if let Some(Operand::Scalar { code, .. }) = f.dst {
                self.set_pair(code.into(), r);
            }
            return;
        }
        match name {
            "v_readfirstlane_b32" => {
                let first = if exec == 0 { 0 } else { exec.trailing_zeros() as usize };
                let Some(Operand::Scalar { code, .. }) = f.dst else { panic!() };
                self.s[usize::from(code)] = self.src(&f.src[0], first);
                return;
            }
            "v_readlane_b32" => {
                let l = self.scalar(&f.src[1]) as usize & 63;
                let Some(Operand::Scalar { code, .. }) = f.dst else { panic!() };
                self.s[usize::from(code)] = self.src(&f.src[0], l);
                return;
            }
            _ => {}
        }
        let Some(Operand::Vector { reg, .. }) = f.dst else { panic!("{name}") };
        let d = usize::from(reg);
        if name == "v_writelane_b32" {
            let l = self.scalar(&f.src[1]) as usize & 63;
            self.v[d][l] = self.scalar(&f.src[0]);
            return;
        }
        let old = self.v.clone();
        for l in (0..64).filter(|&l| on(l)) {
            let mut write = true;
            let src0 = match f.dpp {
                None => self.src_of(&old, &f.src[0], l),
                Some(dpp) => {
                    let Operand::Vector { reg, .. } = f.src[0] else { panic!() };
                    let (from, bound) = match dpp {
                        Dpp::Lanes { sel, .. } => (Some(l & !7 | (sel >> (3 * (l & 7)) & 7) as usize), false),
                        Dpp::Row { ctrl, bound_ctrl, .. } => {
                            let (row, n) = (l & 15, usize::from(ctrl & 15));
                            let from = match ctrl {
                                0x111..=0x11F => (row >= n).then(|| l - n),
                                0x140 => Some(l ^ 15),
                                _ => panic!("DPP {ctrl:#x}"),
                            };
                            (from, bound_ctrl)
                        }
                    };
                    match from.filter(|&s| on(s)) {
                        Some(s) => old[usize::from(reg)][s],
                        None if bound => 0,
                        None => {
                            write = false;
                            0
                        }
                    }
                }
            };
            if !write {
                continue;
            }
            let b = |n: usize| self.src_of(&old, &f.src[n], l);
            self.v[d][l] = match name {
                "v_mov_b32" => src0,
                "v_add_nc_u32" => src0.wrapping_add(b(1)),
                "v_sub_nc_u32" => src0.wrapping_sub(b(1)),
                "v_and_b32" => src0 & b(1),
                "v_or_b32" => src0 | b(1),
                "v_xor_b32" => src0 ^ b(1),
                "v_lshlrev_b32" => b(1) << (src0 & 31),
                "v_lshrrev_b32" => b(1) >> (src0 & 31),
                "v_cndmask_b32" => {
                    let Operand::Scalar { code, .. } = f.src[2] else { panic!() };
                    if self.pair(code.into()) >> l & 1 != 0 { b(1) } else { src0 }
                }
                "v_mbcnt_lo_u32_b32" => {
                    let below = if l >= 32 { u32::MAX } else { (1u32 << l) - 1 };
                    (src0 & below).count_ones() + b(1)
                }
                "v_mbcnt_hi_u32_b32" => {
                    let below = if l < 32 { 0 } else { ((1u64 << (l - 32)) - 1) as u32 };
                    (src0 & below).count_ones() + b(1)
                }
                "v_permlanex16_b32" => {
                    // the selects are constants here: every lane takes lane 15 or 0
                    let at = if self.scalar(&f.src[1]) == 0 { 0 } else { 15 };
                    old[match f.src[0] {
                        Operand::Vector { reg, .. } => usize::from(reg),
                        _ => panic!(),
                    }][(l & !15) ^ 16 | at]
                }
                _ => panic!("the reference has no {name}"),
            };
        }
    }

    fn src_of(&self, v: &[[u32; 64]], o: &Operand, lane: usize) -> u32 {
        match *o {
            Operand::Vector { reg, .. } => v[usize::from(reg)][lane],
            _ => self.scalar(o),
        }
    }
}

// ---- lowering and running ----

fn lower_as(
    n: &NativeProgram,
    host_wave: u32,
    st: &structure::Structured,
    p: &ir::Program,
) -> Result<HostShader, ShaderError> {
    let res = resource::track(p, &n.layout, n.stage, Some(st)).expect("resource::track");
    let io = io::raise(p, n).expect("io::raise");
    let was = lower::TEST_HOST_WAVE.replace(Some(host_wave));
    let lowered = lower::lower(&lower::Parts {
        native: n,
        p,
        st,
        io: &io,
        res: &res,
        inputs: Vec::new(),
        outputs: Vec::new(),
        prims: Vec::new(),
        mesh_stride: 0,
    });
    lower::TEST_HOST_WAVE.set(was);
    let lowered = lowered?;
    Ok(HostShader {
        stage: HwStage::Cs,
        dxil: lowered.dxil,
        user_data: lowered.user_data,
        samplers: lowered.samplers,
        static_samplers: Vec::new(),
        root_cbvs: lowered.root_cbvs,
        root_srvs: lowered.root_srvs,
        root_uavs: lowered.root_uavs,
        root_indexed_views: res.root_indexed_views,
    })
}

/// `n` with two console lanes in each host lane.
fn pairs(n: &NativeProgram) -> HostShader {
    let p = ir::build(&n.code, true, n.vgprs).expect("ir::build");
    let st = structure::structure_console(&p).expect("structure_console");
    let res = resource::track(&p, &n.layout, n.stage, Some(&st)).expect("resource::track");
    let was = lower::TEST_HOST_WAVE.replace(Some(32));
    let can = lower::can_pair(n, &p, &st, &res);
    lower::TEST_HOST_WAVE.set(was);
    can.expect("can_pair");
    super::shader::TEST_BARRIERS.set((0, 0));
    let shader = lower::with_paired_lanes(|| lower_as(n, 32, &st, &p)).expect("lower");
    let (all, entry) = super::shader::TEST_BARRIERS.get();
    assert_eq!(all, entry, "a barrier past the entry block");
    if let Ok(mut v) = validator::Validator::load(validator::Validator::default_path().as_deref()) {
        let mut copy = shader.dxil.clone();
        v.validate(&mut copy).expect("DXIL validation failed");
    }
    shader
}

/// The forms there were without pairs: the exact one, and each half-wave on its own.
fn without_pairs(n: &NativeProgram) -> (Result<HostShader, ShaderError>, Result<HostShader, ShaderError>) {
    let p = ir::build(&n.code, true, n.vgprs).expect("ir::build");
    let st = structure::structure(&p).expect("structure");
    lower::take_needs();
    let exact = lower_as(n, 32, &st, &p);
    let local = lower::with_local_waves_any(|| lower_as(n, 32, &st, &p));
    (exact, local)
}

fn dispatch(host: &Host, shader: &HostShader, threads: u32, constants: [u32; 4]) -> Option<Vec<u32>> {
    let (root, pso) = match host.compute_pipeline(shader) {
        Ok(x) => x,
        Err(e) => {
            eprintln!("the device does not take this program (waves of 32?): {e}; not run");
            return None;
        }
    };
    let mut ud = vec![0u32; shader.user_data_dwords().max(1) as usize];
    for u in &shader.user_data {
        if let (RootType::Constants, 0, d @ 0..=3) = (u.root, u.slot, u.dword) {
            ud[u.index as usize] = constants[d as usize];
        }
    }
    let out = host
        .dispatch(
            &root,
            &pso,
            &Dispatch {
                roots: vec![(0, Root::Constants(ud))],
                views: &[View::RawUav(threads * OUT + TAIL)],
                samplers: &[],
                groups: [1, 1, 1],
            },
        )
        .unwrap_or_else(|e| panic!("dispatch failed: {e}"));
    Some(out.into_iter().next().expect("the UAV"))
}

/// The devices: the hardware one and WARP, where they run SM 6.6.
fn devices() -> Vec<(&'static str, Host)> {
    [("hardware", false), ("WARP", true)]
        .into_iter()
        // `PAIRED_DEVICES=hardware` or `WARP`: that one alone
        .filter(|(name, _)| std::env::var("PAIRED_DEVICES").map_or(true, |d| d == *name))
        .filter_map(|(name, warp)| Host::new(warp).ok().map(|h| (name, h)))
        .collect()
}

/// Runs `n` in pairs on every device for each set of constants and compares all it stores with
/// the reference. `wrong_before`: that the program is one the other forms do not get right
/// (refused, or another result) is checked too.
fn check(what: &str, n: &NativeProgram, threads: u32, constants: &[[u32; 4]], wrong_before: bool) {
    let shader = pairs(n);
    let _one = super::valu_tests::ONE_AT_A_TIME.lock().unwrap_or_else(|e| e.into_inner());
    let devices = devices();
    let mut ran = Vec::new();
    for (name, host) in &devices {
        let mut all = true;
        for &c in constants {
            let want = run_reference(n, threads, c);
            let Some(got) = dispatch(host, &shader, threads, c) else {
                all = false;
                break;
            };
            if let Some(at) = (0..want.len()).find(|&k| want[k] != got[k]) {
                panic!(
                    "{what} on {name}, constants {c:?}: thread {} dword {} is {:#x}, the reference's {:#x}",
                    at as u32 / OUT,
                    at as u32 % OUT,
                    got[at],
                    want[at]
                );
            }
        }
        if all {
            ran.push(*name);
        }
    }
    if wrong_before {
        let (exact, local) = without_pairs(n);
        assert!(exact.is_err(), "{what}: the exact form takes it");
        let wrong = match local {
            Err(_) => true,
            Ok(local) => devices.iter().all(|(_, host)| {
                constants.iter().any(|&c| {
                    dispatch(host, &local, threads, c).is_none_or(|got| got != run_reference(n, threads, c))
                })
            }),
        };
        assert!(wrong, "{what}: each half-wave on its own gives the same");
    }
    eprintln!("{what}: as the reference on {ran:?}");
}

// ---- the programs ----

/// `if (t & 3) { if (WaveActiveAnyTrue(t == s0)) r = 16 + count; r += 1 }`: the shadow
/// programs' skip (`v_cmp`, AND and XOR with EXEC, `s_bcnt1`, `s_cbranch_scc0`).
fn any_in_a_branch(a: &mut Asm) {
    a.v_mov(4, k(0));
    for r in 5..8 {
        a.v_mov(r, v(0));
    }
    let (end, skip) = (a.label(), a.label());
    a.vop2(0x1B, 3, k(3), 0); // v_and_b32 v3, 3, v0
    a.vopc(0xC5, k(0), 3); // v_cmp_ne_u32 vcc, 0, v3
    a.and_save(12, VCC);
    a.branch(8, end); // execz
    a.vopc(0xC5, 0, 0); // v_cmp_ne_u32 vcc, s0, v0
    a.sop2(15, 14, VCC, EXEC); // s_and_b64 s[14:15], vcc, exec
    a.sop2(19, 14, 14, EXEC); // s_xor_b64 s[14:15], s[14:15], exec
    a.sop1(16, 16, 14); // s_bcnt1_i32_b64 s16, s[14:15]
    a.branch(4, skip); // scc0
    a.v_add(4, 16, 4);
    a.v_add(4, k(16), 4);
    a.here(skip);
    a.v_add(4, k(1), 4);
    a.here(end);
    a.s_mov64(EXEC, 12);
}

#[test]
fn a_decision_of_the_wave_in_a_branch_is_each_console_waves() {
    for threads in [64, 128, 256] {
        // a lane of either half, of each wave; one outside the branch; none
        let constants = [5, 37, 63, 64 + 37, 128 + 2, 255, 100, 9999].map(|c| [c, 0, 0, 0]);
        check(&format!("any in a branch, {threads} threads"), &native(threads, any_in_a_branch), threads, &constants, true);
    }
}

/// A lane mask made inside a branch, walked bit by bit, its two dwords in turn (the cluster
/// lights program): every lane in the branch sums the lane numbers of the mask.
fn mask_walk(a: &mut Asm) {
    for r in 4..8 {
        a.v_mov(r, k(0));
    }
    let (end, outer, round, next) = (a.label(), a.label(), a.label(), a.label());
    a.vop2(0x1B, 3, k(3), 0);
    a.vopc(0xC5, k(0), 3);
    a.and_save(12, VCC);
    a.branch(8, end);
    a.vop2(0x1B, 3, k(7), 0); // v_and_b32 v3, 7, v0
    a.vop3(0xC2, 16, k(5), v(3), 0); // v_cmp_eq_u32 s[16:17], 5, v3
    a.vopc(0xC4, 0, 0); // v_cmp_gt_u32 vcc, s0, v0
    a.sop2(15, 16, 16, VCC);
    a.sop1(3, 18, k(0)); // s18: the dword
    a.sop1(3, 19, 16); // s19: its bits still to do
    a.here(outer);
    a.sopc(7, 19, k(0));
    a.branch(4, next);
    a.here(round);
    a.sop1(0x13, 20, 19); // s_ff1_i32_b32 s20, s19
    a.sop2(30, 21, k(1), 20); // s_lshl_b32 s21, 1, s20
    a.sop2(18, 19, 19, 21); // s_xor_b32 s19, s19, s21
    a.sop2(30, 22, 18, k(5)); // s_lshl_b32 s22, s18, 5
    a.sop2(0, 20, 20, 22); // s_add_u32 s20, s20, s22
    a.v_add(4, 20, 4);
    a.v_add(5, k(1), 5);
    a.sopc(7, 19, k(0));
    a.branch(5, round);
    a.here(next);
    a.sop2(0, 18, 18, k(1));
    a.sop1(3, 19, 17);
    a.sopc(10, 18, k(2));
    a.branch(5, outer);
    a.here(end);
    a.s_mov64(EXEC, 12);
}

#[test]
fn a_lane_mask_walked_in_a_branch_has_both_halves_bits() {
    for threads in [64, 128, 256] {
        let constants = [256, 100, 40, 10, 0, 170].map(|c| [c, 0, 0, 0]);
        check(&format!("mask walk, {threads} threads"), &native(threads, mask_walk), threads, &constants, true);
    }
}

/// `WaveActiveBitOr`, a sum and a prefix sum inside a branch, and the count of its lanes.
fn reductions(a: &mut Asm) {
    for r in 4..8 {
        a.v_mov(r, k(0));
    }
    let end = a.label();
    a.vop2(0x16, 10, k(1), 0); // v10 = t >> 1
    a.vop2(0x1A, 11, 1, 0); // v11 = t << (s1 & 31)
    a.vopc(0xC4, 0, 0); // v_cmp_gt_u32 vcc, s0, v0
    a.and_save(12, VCC);
    a.branch(8, end);
    a.v_mov(12, k(1));
    a.vop2(0x1A, 7, v(10), 12); // v7 = 1 << (t >> 1)
    a.wave_or();
    a.v_mov(4, 7);
    a.v_mov(7, v(11));
    a.wave_sum();
    a.v_mov(5, 7);
    a.wave_prefix();
    a.v_mov(6, v(8));
    a.sop1(16, 16, EXEC); // s_bcnt1_i32_b64 s16, exec
    a.v_mov(7, 16);
    a.here(end);
    a.s_mov64(EXEC, 12);
}

#[test]
fn reductions_and_prefixes_in_a_branch_are_the_console_waves() {
    for threads in [64, 128, 256] {
        let constants = [[256, 3, 0, 0], [200, 1, 0, 0], [100, 0, 0, 0], [33, 2, 0, 0], [7, 2, 0, 0], [0, 0, 0, 0]];
        check(&format!("reductions, {threads} threads"), &native(threads, reductions), threads, &constants, true);
    }
}

/// The shader-bin offsets program: lane 63 adds the wave's total to a counter, every lane reads
/// the base back from lane 63 and adds its prefix.
fn lane_63_allocates(a: &mut Asm) {
    for r in 4..8 {
        a.v_mov(r, k(0));
    }
    let end = a.label();
    a.vop2(0x1B, 10, 0, 0); // v10 = t & s0: the lane's count
    a.v_mov(7, v(10));
    a.wave_prefix();
    a.v_mov(6, v(8)); // the prefix
    a.vop3(0x365, 11, ONES, k(0), 0); // v_mbcnt_lo_u32_b32 v11, -1, 0
    a.vop3(0x366, 11, ONES, v(11), 0); // v_mbcnt_hi_u32_b32 v11, -1, v11
    a.v_mov(12, k(0));
    a.s_mov64(14, EXEC);
    a.vopc(0xD2, k(63), 11); // v_cmpx_eq_u32 exec, 63, v11
    a.branch(8, end);
    a.v_add(7, v(6), 10); // the wave's total
    a.wave_sum(); // (over lane 63 alone)
    a.s_mov64(16, EXEC);
    a.v_mov(12, 7);
    a.atomic_add(12, 4 * 64 * OUT);
    a.vop1(2, 18, v(12)); // v_readfirstlane_b32 s18, v12
    a.v_mov(12, 18);
    a.here(end);
    a.s_mov64(EXEC, 14);
    a.words.extend([0xD760_0012, k(63) << 9 | v(12)]); // v_readlane_b32 s18, v12, 63
    a.v_add(4, 18, 6);
    a.v_mov(5, v(11));
    a.v_mov(7, k(0));
}

#[test]
fn lane_63_allocates_for_its_whole_wave() {
    let constants = [u32::MAX, 3, 1, 0, 32].map(|c| [c, 0, 0, 0]);
    check("lane 63 allocates", &native(64, lane_63_allocates), 64, &constants, true);
}

/// A scalarised loop in a branch: the first lane's value, the lanes that hold it, the rest; the
/// waves of a group go round it different numbers of times (no barrier is involved).
fn first_lane_loop(a: &mut Asm) {
    for r in 4..8 {
        a.v_mov(r, k(0));
    }
    let (end, round) = (a.label(), a.label());
    a.vop2(0x1B, 3, k(3), 0);
    a.vopc(0xC5, k(0), 3);
    a.and_save(12, VCC);
    a.branch(8, end);
    a.vop2(0x16, 10, 1, 0); // v10 = t >> s1
    a.vop2(0x1B, 10, 0, 10); // v10 &= s0
    a.sop1(3, 20, k(0));
    a.s_mov64(14, EXEC);
    a.here(round);
    a.sop2(0, 20, 20, k(1)); // s20: the round
    a.vop1(2, 18, v(10)); // v_readfirstlane_b32 s18, v10
    a.vopc(0xC2, 18, 10); // v_cmp_eq_u32 vcc, s18, v10
    a.and_save(16, VCC);
    a.v_add(4, 18, 4);
    a.v_mov(5, 20);
    a.sop1(16, 22, EXEC); // the lanes that hold it
    a.v_mov(6, 22);
    a.sop2(21, EXEC, 16, EXEC); // s_andn2_b64 exec, s[16:17], exec
    a.branch(9, round); // execnz
    a.s_mov64(EXEC, 14);
    a.v_mov(7, 20);
    a.here(end);
    a.s_mov64(EXEC, 12);
}

#[test]
fn a_first_lane_loop_in_a_branch_takes_the_console_waves_first_lane() {
    for threads in [64, 128, 256] {
        let constants = [[7, 3, 0, 0], [63, 0, 0, 0], [15, 4, 0, 0], [3, 5, 0, 0], [0, 0, 0, 0]];
        check(&format!("first-lane loop, {threads} threads"), &native(threads, first_lane_loop), threads, &constants, true);
    }
}

/// `vccz` asks for VCC's bits in lanes outside EXEC too.
fn vcc_outside_exec(a: &mut Asm) {
    for r in 4..8 {
        a.v_mov(r, k(0));
    }
    let (end, skip) = (a.label(), a.label());
    a.vopc(0xC2, 0, 0); // v_cmp_eq_u32 vcc, s0, v0: one lane of the group
    a.s_mov64(14, VCC);
    a.vop2(0x1B, 3, k(7), 0);
    a.vopc(0xC2, k(3), 3); // the lanes with t & 7 == 3
    a.and_save(12, VCC);
    a.branch(8, end);
    a.s_mov64(VCC, 14);
    a.branch(6, skip); // vccz
    a.v_add(4, k(5), 4);
    a.here(skip);
    a.v_add(4, k(1), 4);
    a.here(end);
    a.s_mov64(EXEC, 12);
}

/// `s_bcnt1` of "EXEC under a scalar condition" where no branch skips an empty EXEC: a scalar
/// counts the waves that had a lane.
fn count_of_an_empty_exec(a: &mut Asm) {
    for r in 4..8 {
        a.v_mov(r, k(0));
    }
    let skip = a.label();
    a.sop1(3, 20, k(0));
    a.vopc(0xC2, 0, 0); // the lane t == s0
    a.and_save(12, VCC); // (no skip)
    a.sopc(6, 1, k(0)); // s_cmp_eq_u32 s1, 0
    a.sop2(11, 14, EXEC, k(0)); // s_cselect_b64 s[14:15], exec, 0
    a.sop1(16, 16, 14); // s_bcnt1_i32_b64 s16, s[14:15]
    a.branch(4, skip); // scc0
    a.sop2(0, 20, 20, k(7));
    a.here(skip);
    a.s_mov64(EXEC, 12);
    a.v_mov(4, 20);
}

#[test]
fn tests_of_whole_masks_in_the_console_flow_are_of_whole_masks() {
    for threads in [64, 128, 256] {
        let constants = [3, 4, 60, 64 + 3, 128 + 40, 9999].map(|c| [c, 0, 0, 0]);
        check(&format!("vcc outside EXEC, {threads} threads"), &native(threads, vcc_outside_exec), threads, &constants, false);
        check(
            &format!("a count of an empty EXEC, {threads} threads"),
            &native(threads, count_of_an_empty_exec),
            threads,
            &[[3, 0, 0, 0], [70, 0, 0, 0], [3, 1, 0, 0], [9999, 0, 0, 0]],
            false,
        );
    }
}

/// No barrier but the entry's, and none at all in a group of one console wave.
#[test]
fn paired_lanes_add_no_barrier_in_control_flow() {
    for (body, name) in [
        (any_in_a_branch as fn(&mut Asm), "any in a branch"),
        (mask_walk, "mask walk"),
        (reductions, "reductions"),
        (first_lane_loop, "first-lane loop"),
    ] {
        for (threads, want) in [(64, 0), (256, 1)] {
            let n = native(threads, body);
            let p = ir::build(&n.code, true, n.vgprs).expect("ir::build");
            let st = structure::structure_console(&p).expect("structure_console");
            super::shader::TEST_BARRIERS.set((0, 0));
            lower::with_paired_lanes(|| lower_as(&n, 32, &st, &p)).expect("lower");
            let (all, entry) = super::shader::TEST_BARRIERS.get();
            assert_eq!((all, entry), (want, want), "{name}, {threads} threads: barriers (all, in the entry block)");
        }
    }
}

// ---- random programs ----

struct Rng(u64);

impl Rng {
    fn next(&mut self, n: u32) -> u32 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        ((self.0 >> 20) % u64::from(n)) as u32
    }
}

/// A random program: v4..v7 worked on by vector code in nested lane branches, decisions of the
/// wave, first-lane and lane reads, reductions and loops.
struct Gen {
    r: Rng,
    a: Asm,
    /// The next SGPR free for a scalar or a mask.
    s: u32,
}

impl Gen {
    fn data(&mut self) -> u32 {
        4 + self.r.next(3)
    }
    fn scalars(&mut self, n: u32) -> u32 {
        let at = self.s.next_multiple_of(n);
        self.s = at + n;
        assert!(self.s <= 104, "out of scalars");
        at
    }
    fn vector_op(&mut self) {
        let (d, b) = (self.data(), self.data());
        let x = match self.r.next(5) {
            0 => k(self.r.next(40)),
            1 => self.r.next(4),
            2 => v(0),
            _ => v(self.data()),
        };
        match self.r.next(7) {
            0 => self.a.vop2(0x26, d, x, b),
            1 => self.a.vop2(0x1B, d, x, b),
            2 => self.a.vop2(0x1C, d, x, b),
            3 => self.a.vop2(0x1D, d, x, b),
            4 => self.a.vop2(0x16, d, k(1), b),
            _ => self.a.v_add(d, x, b),
        }
    }
    /// A compare of data into `to` (an SGPR pair or VCC).
    fn compare(&mut self, to: u32) {
        let op = [0xC1, 0xC2, 0xC3, 0xC4, 0xC5][self.r.next(5) as usize];
        self.a.vop2(0x1B, 10, k(15), 4); // v10 = v4 & 15
        let b = self.data();
        self.a.vop2(0x1B, 10, k([3, 7, 15, 63][self.r.next(4) as usize]), b);
        let x = match self.r.next(3) {
            0 => self.r.next(4),
            _ => k(self.r.next(12)),
        };
        if to == VCC {
            self.a.vopc(op, x, 10);
        } else {
            self.a.vop3(op, to, x, v(10), 0);
        }
    }
    fn block(&mut self, depth: u32) {
        for _ in 0..2 + self.r.next(3) {
            self.statement(depth);
        }
    }
    fn statement(&mut self, depth: u32) {
        let was = self.s;
        match self.r.next(if depth == 0 { 8 } else { 14 }) {
            0..=2 => self.vector_op(),
            3 => {
                // the first lane's value
                let (s, d) = (self.scalars(1), self.data());
                self.a.vop1(2, s, v(d));
                let d = self.data();
                self.a.v_add(d, s, d);
            }
            4 => {
                // a lane's value
                let (s, d, l) = (self.scalars(1), self.data(), self.r.next(64));
                self.a.words.extend([0xD760_0000 | s, k(l) << 9 | v(d)]);
                let d = self.data();
                self.a.vop2(0x1D, d, s, d);
            }
            5 => {
                // how many lanes of EXEC have it
                let (m, s) = (self.scalars(2), self.scalars(1));
                self.compare(m);
                self.a.sop2(15, m, m, EXEC);
                self.a.sop1(16, s, m);
                let d = self.data();
                self.a.v_add(d, s, d);
            }
            6 => {
                // a reduction or a prefix
                let d = self.data();
                self.a.v_mov(7, v(d));
                match self.r.next(3) {
                    0 => self.a.wave_or(),
                    1 => self.a.wave_sum(),
                    _ => {
                        self.a.wave_prefix();
                        let d = self.data();
                        self.a.v_add(d, v(8), d);
                        return;
                    }
                }
                let d = self.data();
                self.a.v_add(d, 7, d);
            }
            7 => {
                let m = self.scalars(2);
                self.compare(m);
                let (d, b) = (self.data(), self.data());
                self.a.vop3(0x101, d, v(d), v(b), m);
            }
            8 | 9 => {
                // a lane branch, its skip there or not
                let (save, end) = (self.scalars(2), self.a.label());
                self.compare(VCC);
                self.a.and_save(save, VCC);
                if self.r.next(3) != 0 {
                    self.a.branch(8, end);
                }
                self.block(depth - 1);
                self.a.here(end);
                self.a.s_mov64(EXEC, save);
            }
            10 | 11 => {
                // some lane, or every lane: the wave's decision
                let (m, s, skip) = (self.scalars(2), self.scalars(1), self.a.label());
                self.compare(m);
                self.a.sop2(15, m, m, EXEC);
                match self.r.next(3) {
                    0 => {
                        self.a.sop2(19, m, m, EXEC);
                        self.a.sop1(16, s, m);
                        self.a.branch(4, skip);
                    }
                    1 => {
                        self.a.sop2(15, VCC, m, EXEC);
                        self.a.branch(6, skip);
                    }
                    _ => {
                        self.a.sop1(16, s, m);
                        self.a.sopc(7, s, k(0));
                        self.a.branch(4, skip);
                    }
                }
                self.block(depth - 1);
                self.a.here(skip);
            }
            12 => {
                // a loop each lane leaves by its own count (v11..v13 by depth)
                let (save, count) = (self.scalars(2), 10 + depth);
                let (round, done) = (self.a.label(), self.a.label());
                let d = self.data();
                self.a.vop2(0x1B, count, k(3), d);
                self.a.s_mov64(save, EXEC);
                self.a.here(round);
                self.a.vopc(0xD5, k(0), count); // v_cmpx_ne_u32 exec, 0, count
                self.a.branch(8, done);
                self.block(depth - 1);
                self.a.vop2(0x26, count, v(count), 14); // less v14 (one)
                self.a.branch(2, round);
                self.a.here(done);
                self.a.s_mov64(EXEC, save);
            }
            _ => {
                // a first-lane loop over the values of two bits
                let (save, hold, s) = (self.scalars(2), self.scalars(2), self.scalars(1));
                let round = self.a.label();
                let (d, key) = (self.data(), 10 + depth);
                self.a.vop2(0x1B, key, k(3), d);
                self.a.s_mov64(save, EXEC);
                self.a.here(round);
                self.a.vop1(2, s, v(key));
                self.a.vopc(0xC2, s, key);
                self.a.and_save(hold, VCC);
                let d = self.data();
                self.a.v_add(d, s, d);
                self.vector_op();
                self.a.sop2(21, EXEC, hold, EXEC);
                self.a.branch(9, round);
                self.a.s_mov64(EXEC, save);
            }
        }
        self.s = was;
    }
}

fn random(seed: u64, threads: u32) -> NativeProgram {
    random_from(seed, threads, 0)
}

fn random_from(seed: u64, threads: u32, first: u32) -> NativeProgram {
    native_from(threads, first, |a| {
        let mut g = Gen { r: Rng(seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1), a: std::mem::take(a), s: 24 };
        for s in 24..104 {
            g.a.sop1(3, s, k(0));
        }
        g.a.v_mov(14, k(1));
        for r in [8, 10, 11, 12, 13] {
            g.a.v_mov(r, k(0));
        }
        g.a.v_mov(4, v(0));
        g.a.vop2(0x16, 5, k(2), 0);
        g.a.vop2(0x1D, 6, 0, 0);
        g.a.v_mov(7, k(0));
        for _ in 0..4 {
            g.statement(3);
            g.vector_op();
        }
        // (v7 is the reductions' own)
        g.a.v_mov(7, v(4));
        *a = g.a;
    })
}

#[test]
fn random_programs_are_as_the_reference() {
    let _one = super::valu_tests::ONE_AT_A_TIME.lock().unwrap_or_else(|e| e.into_inner());
    let devices = devices();
    let count: u64 = std::env::var("PAIRED_SEEDS").ok().and_then(|n| n.parse().ok()).unwrap_or(96);
    let (mut made, mut runs) = (0, 0);
    let mut instructions = 0;
    let mut wrong: Vec<String> = Vec::new();
    for seed in 1..=count {
        let threads = [64, 128, 256][(seed % 3) as usize];
        let n = random(seed, threads);
        let shader = pairs(&n);
        made += 1;
        instructions += n.code.len() / 4;
        // On WARP each console wave alone, as a group of its own with the same thread numbers:
        // in groups of several waves that go different ways WARP (1.0.13 to 1.0.22, each on
        // programs of its own) loses lanes, barrier or not.
        let alone: Vec<(NativeProgram, HostShader)> = (0..threads / 64)
            .map(|w| {
                let n = random_from(seed, 64, 64 * w);
                let shader = pairs(&n);
                (n, shader)
            })
            .collect();
        for (name, host) in &devices {
            let whole = [(n.clone(), shader.clone())];
            let each = if *name == "WARP" { &alone[..] } else { &whole[..] };
            for (n, shader) in each {
                for c in [[3, 9, 1, 200], [0, 0, 0, 0], [77, 2, 5, 130]] {
                    let want = run_reference(n, threads, c);
                    let Some(got) = dispatch(host, shader, threads, c) else { continue };
                    runs += 1;
                    if let Some(at) = (0..want.len()).find(|&k| want[k] != got[k]) {
                        wrong.push(format!(
                            "seed {seed} ({threads} threads, {} words) on {name}, constants {c:?}: thread {} v{} is {:#x}, the reference's {:#x}",
                            n.code.len() / 4,
                            at as u32 / OUT,
                            4 + at as u32 % OUT,
                            got[at],
                            want[at]
                        ));
                    }
                }
            }
        }
    }
    assert!(wrong.is_empty(), "{} of {runs} runs differ from the reference:
{}", wrong.len(), wrong.join("
"));
    eprintln!("{made} random programs ({instructions} words), {runs} runs as the reference");
}

/// A first-lane read where no lane is in EXEC gives lane 0's value, in the reference too.
#[test]
fn the_reference_reads_lane_0_of_an_empty_exec() {
    // EXEC 0 in a wave: the first lane read is lane 0's
    let n = native(64, |a| {
        for r in 4..8 {
            a.v_mov(r, v(0));
        }
        a.vopc(0xC4, k(0), 0); // v_cmp_gt_u32 vcc, 0, v0: none
        a.and_save(12, VCC);
        a.vop1(2, 18, v(0));
        a.s_mov64(EXEC, 12);
        a.v_mov(4, 18);
    });
    assert!(run_reference(&n, 64, [0; 4])[..256].chunks(4).all(|t| t[0] == 0));
}
