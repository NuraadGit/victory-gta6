//! Textures shaders write that the title's own code then reads on the CPU.
//! On the console the CPU and the GPU share memory, so what a compute pass
//! stores into a texture is in title memory for the CPU to read. Here it stays
//! in the texture's host copy. The title's ocean is one: its wave passes write
//! a 256x256 height map (R32_FLOAT) and a displacement map (R32G32_FLOAT), and
//! its CPU code samples both, which is what boats float on; it read memory
//! nothing had written (run rw1, `VICTORY_READ_WATCH=1`: about 30 places in
//! the title read those two kinds), and boats sank.
//!
//! With `VICTORY_TEXTURE_BACK=1`, a list in which one of the title's water
//! programs ([`WRITERS`], by shader hash: the traces in fork issue #863)
//! writes a texture copies it into a read-back buffer of ours when it
//! closes, and once the list has run the texels are tiled into the texture's
//! title memory in the console's layout, as the console's GPU would have
//! left them. By the writing program, not by the texture's shape: other
//! textures of the same sizes and formats are placed in memory the title
//! uses again within the frame, and writing those back puts texels over
//! whatever lies there by then.
//!
//! `VICTORY_TEXTURE_BACK=<width>x<height>f<format>[,...]`: by kind instead,
//! whatever program writes them, as the patch did at first.
//!
//! Only the GPU writes these textures' memory, so the title's bytes there
//! stay what we last landed. If they changed, the title has used the memory
//! for something else without telling us (its own heaps), and that texture
//! is landed no more: each copy carries a hash of the title's bytes when it
//! was recorded, and the bytes we land are hashed too ([`LANDED`]).
//!
//! The copy shows what the list's last write left; a title read in between
//! sees the frame before, as it would before the console's GPU has run.

use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex, OnceLock, Weak};

use windows::Win32::Graphics::Direct3D12::*;

use crate::object::This;

/// What is copied back.
enum Mode {
    Off,
    /// What the programs of [`WRITERS`] write.
    Programs,
    /// Textures of these kinds (width, height, format), whoever writes them.
    Kinds(Vec<(u64, u32, u32)>),
}

/// The compute programs whose output textures the title's CPU code reads:
/// `WaterSimMgr:CS_CompositeSimDataIntoOutputTexture*` (height and
/// displacement) and `water_compressed_simulation_rendertarget:CS_MinMax_*`
/// (the simulation's heights in blocks).
const WRITERS: [&str; 13] = [
    "ba423f650a4e8c2f",
    "5d3fa5f1ee755e44",
    "bf0e8d5ee5ae755c",
    "e4609cfad2176dac",
    "8e9fa5ede09d77c5",
    "f64bac83df2ea939",
    "6c4c4ff7283f90b3",
    "3ffb92133d372b30",
    "bdf438d5731d36cc",
    "253532a6de53babc",
    "a2f1004f8d6d7aad",
    "65b6e65c9994d495",
    "9809b8d965b04935",
];

fn parse(value: Option<&str>) -> Mode {
    match value.map(str::trim) {
        None | Some("" | "0") => Mode::Off,
        Some("1") => Mode::Programs,
        Some(spec) => {
            let kinds: Vec<(u64, u32, u32)> = spec
                .split(',')
                .filter_map(|k| {
                    let (size, format) = k.trim().split_once('f')?;
                    let (w, h) = size.split_once('x')?;
                    Some((w.parse().ok()?, h.parse().ok()?, format.parse().ok()?))
                })
                .collect();
            if kinds.is_empty() { Mode::Off } else { Mode::Kinds(kinds) }
        }
    }
}

/// Of the nine kinds the water programs write, the two copied back by
/// default: 64x64 R32G32_FLOAT and 128x128 R16G16_UINT (DXGI formats 16 and
/// 36; the list iGoldenNinja found on fork PR #999). With all nine landed
/// (the 256x256 and 512x512 height and displacement maps, the 2048x128 and
/// 4096x64 R8/R16 integer textures) the title's script read through a null
/// object ~13 s after "Follow the plane" began, or the card hung, 6 runs of
/// 6 on the Jason save; with these two the mission went on to "Collect the
/// package", the boat afloat (run k1, 572 s).
const KINDS: [(u64, u32, u32); 2] = [(64, 64, 16), (128, 128, 36)];

/// `VICTORY_TEXTURE_BACK_KINDS=<width>x<height>f<format>[,...]` with
/// `VICTORY_TEXTURE_BACK=1`: of what the water programs write, only these
/// (default [`KINDS`]; `all`: every kind they write).
fn kinds_filter() -> Option<&'static [(u64, u32, u32)]> {
    static FILTER: OnceLock<Option<Vec<(u64, u32, u32)>>> = OnceLock::new();
    FILTER
        .get_or_init(|| {
            let v = std::env::var("VICTORY_TEXTURE_BACK_KINDS").ok();
            if v.as_deref().map(str::trim) == Some("all") {
                return None;
            }
            match parse(v.as_deref()) {
                Mode::Kinds(k) => Some(k),
                _ => Some(KINDS.to_vec()),
            }
        })
        .as_deref()
}

fn mode() -> &'static Mode {
    static MODE: OnceLock<Mode> = OnceLock::new();
    MODE.get_or_init(|| {
        let mode = parse(std::env::var("VICTORY_TEXTURE_BACK").ok().as_deref());
        match &mode {
            Mode::Off => {}
            Mode::Programs => tracing::info!("the textures the title's water programs write are copied back into title memory (VICTORY_TEXTURE_BACK=1)"),
            Mode::Kinds(kinds) => tracing::info!(?kinds, "textures of these kinds that shaders write are copied back into title memory (VICTORY_TEXTURE_BACK)"),
        }
        mode
    })
}

thread_local! {
    /// Whether the dispatch this thread prepares is of a program of [`WRITERS`].
    static WRITING: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

fn writer(hash: &str) -> bool {
    WRITERS.iter().any(|w| w.eq_ignore_ascii_case(hash))
}

/// Before (and, with `None`, after) a shader's views are prepared: the
/// compute shader's hash.
pub(crate) fn preparing(hash: Option<&str>) {
    if matches!(mode(), Mode::Programs) {
        WRITING.set(hash.is_some_and(writer));
    }
}

/// Whether title texture `resource`, about to be written through unordered
/// access, is copied back.
pub(crate) fn wanted(resource: This) -> bool {
    match mode() {
        Mode::Off => false,
        Mode::Programs => {
            if !WRITING.get() {
                return false;
            }
            let Some(kind) = crate::resource::texture_kind(resource) else { return false };
            let take = kinds_filter().is_none_or(|f| f.contains(&kind));
            static SEEN: Mutex<Option<std::collections::HashSet<(u64, u32, u32, bool)>>> = Mutex::new(None);
            if SEEN.lock().unwrap_or_else(|e| e.into_inner()).get_or_insert_with(Default::default).insert((kind.0, kind.1, kind.2, take)) {
                tracing::info!(width = kind.0, height = kind.1, format = kind.2, take, "a water program writes a texture of this kind");
            }
            take
        }
        Mode::Kinds(kinds) => crate::resource::texture_kind(resource).is_some_and(|k| kinds.contains(&k)),
    }
}

/// An outstanding write to a specific allocation. Mapping and resource
/// changes invalidate only the writes that could touch the replaced memory.
struct Ticket {
    owner: usize,
    at: usize,
    end: usize,
    valid: AtomicBool,
}

static TICKETS: Mutex<Vec<Weak<Ticket>>> = Mutex::new(Vec::new());

/// By title address: the end and the hash of the bytes we last landed there.
static LANDED: Mutex<Option<std::collections::HashMap<usize, (usize, u64)>>> = Mutex::new(None);
/// Title textures (by resource) whose memory the title changed under us.
static DENIED: Mutex<Option<std::collections::HashSet<usize>>> = Mutex::new(None);
static REUSED: AtomicU64 = AtomicU64::new(0);

fn hash(bytes: &[u8]) -> u64 {
    use std::hash::Hasher;
    let mut h = std::collections::hash_map::DefaultHasher::new();
    h.write(bytes);
    h.finish()
}

/// The title's bytes at `[at, end)` changed under us: texture `owner` is
/// landed no more.
fn deny(owner: usize, at: usize, end: usize, why: &str) {
    DENIED.lock().unwrap_or_else(|e| e.into_inner()).get_or_insert_with(Default::default).insert(owner);
    if let Some(l) = LANDED.lock().unwrap_or_else(|e| e.into_inner()).as_mut() {
        l.remove(&at);
    }
    let n = REUSED.fetch_add(1, Ordering::Relaxed) + 1;
    if n.is_power_of_two() {
        tracing::info!(n, owner = format_args!("{owner:#x}"), at = format_args!("{at:#x}"), bytes = end - at, why, "texture readback: the title used this memory for something else; not landed again");
    }
}

fn denied(owner: usize) -> bool {
    DENIED.lock().unwrap_or_else(|e| e.into_inner()).as_ref().is_some_and(|d| d.contains(&owner))
}
static TRACKING: AtomicBool = AtomicBool::new(false);

fn track(owner: usize, at: usize, end: usize) -> Arc<Ticket> {
    let ticket = Arc::new(Ticket { owner, at, end, valid: AtomicBool::new(true) });
    let mut tickets = TICKETS.lock().unwrap_or_else(|e| e.into_inner());
    tickets.retain(|weak| weak.strong_count() != 0);
    tickets.push(Arc::downgrade(&ticket));
    TRACKING.store(true, Ordering::Relaxed);
    ticket
}

fn invalidate(matches: impl Fn(&Ticket) -> bool) {
    if !TRACKING.load(Ordering::Relaxed) {
        return;
    }
    TICKETS.lock().unwrap_or_else(|e| e.into_inner()).retain(|weak| {
        let Some(ticket) = weak.upgrade() else { return false };
        if matches(&ticket) {
            ticket.valid.store(false, Ordering::Release);
        }
        true
    });
}

pub(crate) fn released(owner: usize) {
    invalidate(|ticket| ticket.owner == owner);
    if let Some(d) = DENIED.lock().unwrap_or_else(|e| e.into_inner()).as_mut() {
        d.remove(&owner);
    }
}

pub(crate) fn replaced(at: usize) {
    invalidate(|ticket| ticket.at == at);
}

pub(crate) fn remapped(at: usize, len: usize) {
    let end = at.saturating_add(len);
    invalidate(|ticket| at < ticket.end && ticket.at < end);
}

pub(crate) struct Target {
    pub(crate) copy: Arc<crate::resource::HostTexture>,
    layout: victory_formats::texture::Layout,
    ticket: Arc<Ticket>,
    /// The hash of the title's bytes when the copy was recorded.
    before: u64,
}

impl Target {
    pub(crate) fn new(resource: This) -> Option<Self> {
        // Keep the host copy independently of the title's resource lifetime.
        let copy = crate::resource::host_copy(resource)?;
        let _mapping = crate::watch::title_memory_reading();
        let (layout, at, size) = crate::resource::title_texture(resource)?;
        let end = at.checked_add(size).filter(|&end| at != 0 && end > at)?;
        if denied(resource as usize) {
            return None;
        }
        let mut now = vec![0u8; end - at];
        if !crate::host::read_into(at as *const u8, &mut now) {
            return None;
        }
        let before = hash(&now);
        let last = LANDED.lock().unwrap_or_else(|e| e.into_inner()).as_ref().and_then(|l| l.get(&at).copied());
        if let Some((last_end, last)) = last
            && last_end == end
            && last != before
        {
            deny(resource as usize, at, end, "changed since we landed");
            return None;
        }
        let ticket = track(resource as usize, at, end);
        Some(Self { copy, layout, ticket, before })
    }

    pub(crate) fn owner(&self) -> usize { self.ticket.owner }
}

/// The readback resources and the allocation the pixels belong to.
pub(crate) struct Back {
    buffer: ID3D12Resource,
    footprints: Vec<D3D12_PLACED_SUBRESOURCE_FOOTPRINT>,
    rows: Vec<u32>,
    mips: u32,
    layout: victory_formats::texture::Layout,
    ticket: Arc<Ticket>,
    before: u64,
}

impl Back {
    pub(crate) fn range(&self) -> (usize, usize) {
        (self.ticket.at, self.ticket.end)
    }
}

/// Records a host copy from a texture already in COPY_SOURCE.
pub(crate) fn record(list: &ID3D12GraphicsCommandList, target: Target) -> Option<Back> {
    if !target.ticket.valid.load(Ordering::Acquire) {
        return None;
    }
    let host = crate::host::host()?;
    let texture = &target.copy.texture;
    // SAFETY: a live texture kept through the command list's state tracking.
    let desc = unsafe { texture.GetDesc() };
    let mips = u32::from(desc.MipLevels.max(1));
    let subresources = mips * u32::from(desc.DepthOrArraySize.max(1));
    let n = subresources as usize;
    let mut footprints = vec![D3D12_PLACED_SUBRESOURCE_FOOTPRINT::default(); n];
    let mut rows = vec![0u32; n];
    let mut total = 0u64;
    // SAFETY: out arrays of `n` entries.
    unsafe {
        host.device.GetCopyableFootprints(
            &desc,
            0,
            subresources,
            0,
            Some(footprints.as_mut_ptr()),
            Some(rows.as_mut_ptr()),
            None,
            Some(&mut total),
        )
    };
    // The last footprint's advertised extent can omit row-end padding.
    // The tiler reads a pitched slice, so the mapped buffer must include it.
    for (footprint, &rows) in footprints.iter().zip(&rows) {
        let format = footprint.Footprint;
        let pitched = u64::from(format.RowPitch)
            .checked_mul(u64::from(rows))?
            .checked_mul(u64::from(format.Depth))?;
        total = total.max(footprint.Offset.checked_add(pitched)?);
    }
    let buffer = host.read_back_buffer(total)?;
    for (i, footprint) in footprints.iter().enumerate() {
        let to = D3D12_TEXTURE_COPY_LOCATION {
            // SAFETY: a borrowed pointer the location never releases.
            pResource: unsafe { std::mem::transmute_copy(&buffer) },
            Type: D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT,
            Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 { PlacedFootprint: *footprint },
        };
        let from = D3D12_TEXTURE_COPY_LOCATION {
            // SAFETY: as above.
            pResource: unsafe { std::mem::transmute_copy(texture) },
            Type: D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX,
            Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 { SubresourceIndex: i as u32 },
        };
        // SAFETY: an open list; the texture is in the copy-source state.
        unsafe { list.CopyTextureRegion(&to, 0, 0, 0, &from, None) };
    }
    Some(Back { buffer, footprints, rows, mips, layout: target.layout, ticket: target.ticket, before: target.before })
}

/// Once the list has run, tile its texels back into the same title allocation.
/// `VICTORY_TEXTURE_BACK_GUARD=0`: land without checking the title's bytes.
fn guarded() -> bool {
    static GUARD: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *GUARD.get_or_init(|| std::env::var_os("VICTORY_TEXTURE_BACK_GUARD").is_none_or(|v| v != "0"))
}

fn quiet() -> bool {
    static QUIET: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *QUIET.get_or_init(|| std::env::var_os("VICTORY_TEXTURE_BACK_QUIET").is_some_and(|v| v == "1"))
}

fn dry() -> bool {
    static DRY: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *DRY.get_or_init(|| std::env::var_os("VICTORY_TEXTURE_BACK_DRY").is_some_and(|v| v == "1"))
}

pub(crate) fn land(back: &Back) {
    static COPIED: AtomicU64 = AtomicU64::new(0);
    static STALE: AtomicU64 = AtomicU64::new(0);
    let _mapping = crate::watch::title_memory_reading();
    let _ticket_guard = TICKETS.lock().unwrap_or_else(|e| e.into_inner());
    if !back.ticket.valid.load(Ordering::Acquire) {
        let skipped = STALE.fetch_add(1, Ordering::Relaxed) + 1;
        if skipped.is_power_of_two() {
            tracing::info!(skipped, "skipping texture readbacks whose title allocation changed");
        }
        return;
    }
    let (at, end) = back.range();
    let size = end - at;
    let mut tiled = vec![0u8; size];
    // Keep title bytes the subresources do not cover.
    if !crate::host::read_into(at as *const u8, &mut tiled) {
        return;
    }
    if guarded() {
        // What we recorded over, or what an earlier copy of the same
        // texture has landed since.
        let now = hash(&tiled);
        let landed = LANDED.lock().unwrap_or_else(|e| e.into_inner()).as_ref().and_then(|l| l.get(&at).copied());
        if now != back.before && landed.is_none_or(|(e, h)| e != end || h != now) {
            deny(back.ticket.owner, at, end, "changed while the copy ran");
            return;
        }
    }
    let mut data = std::ptr::null_mut();
    // SAFETY: our read-back buffer, written by the list that has run.
    unsafe {
        if back.buffer.Map(0, None, Some(&mut data)).is_err() || data.is_null() {
            return;
        }
    }
    for (i, (footprint, &rows)) in back.footprints.iter().zip(&back.rows).enumerate() {
        let (mip, slice) = (i as u32 % back.mips, i as u32 / back.mips);
        let f = footprint.Footprint;
        let pitch = f.RowPitch as usize;
        let len = pitch * rows as usize * f.Depth as usize;
        // SAFETY: the subresource's part of the mapped buffer.
        let linear = unsafe { std::slice::from_raw_parts(data.cast::<u8>().add(footprint.Offset as usize), len) };
        if let Err(e) = back.layout.tile(&mut tiled, mip, slice, linear, pitch, pitch * rows as usize) {
            static ONCE: std::sync::Once = std::sync::Once::new();
            ONCE.call_once(|| tracing::warn!(?e, mip, slice, "a texture copied back could not be tiled"));
            // SAFETY: mapped above.
            unsafe { back.buffer.Unmap(0, None) };
            return;
        }
    }
    // SAFETY: mapped above.
    unsafe { back.buffer.Unmap(0, None) };
    // VICTORY_TEXTURE_BACK_DRY=1: the copies run and are tiled, the title's
    // memory is left as it was (does the copy or the title's use of it hang?).
    if guarded() && (quiet() || !dry()) {
        LANDED.lock().unwrap_or_else(|e| e.into_inner()).get_or_insert_with(Default::default).insert(at, (end, hash(&tiled)));
    }
    if quiet() {
        crate::watch::writing_quiet(at, size);
        crate::host::copy_mapped(at as *mut u8, tiled.as_ptr(), size, false);
    } else if !dry() {
        crate::host::copy_mapped(at as *mut u8, tiled.as_ptr(), size, true);
    }
    let copied = COPIED.fetch_add(1, Ordering::Relaxed) + 1;
    if copied == 1 || copied.is_multiple_of(128) {
        tracing::info!(copied, bytes = size, "texture readback landed");
    }
}


#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn only_the_water_programs_textures_are_copied_back_by_default() {
        assert!(matches!(parse(None), Mode::Off) && matches!(parse(Some("0")), Mode::Off) && matches!(parse(Some(" ")), Mode::Off));
        assert!(matches!(parse(Some("1")), Mode::Programs));
        assert!(matches!(parse(Some("256x256f41, 64x64f16")), Mode::Kinds(k) if k == [(256, 256, 41), (64, 64, 16)]));
        assert!(matches!(parse(Some("nonsense")), Mode::Off));
        assert!(writer("bdf438d5731d36cc") && writer("BDF438D5731D36CC"));
        assert!(!writer("907ba309e720e216") && !writer(""));
        assert!(WRITERS.iter().all(|w| w.len() == 16), "whole hashes");
    }

    #[test]
    fn only_readbacks_for_changed_allocations_are_invalidated() {
        let first = track(91, 0x1000, 0x2000);
        let next = track(92, 0x2000, 0x3000);
        let far = track(93, 0x4000, 0x5000);

        remapped(0x0f00, 0x200);
        assert!(!first.valid.load(Ordering::Acquire));
        assert!(next.valid.load(Ordering::Acquire));
        replaced(0x2000);
        assert!(!next.valid.load(Ordering::Acquire));
        assert!(far.valid.load(Ordering::Acquire));

        released(93);
        assert!(!far.valid.load(Ordering::Acquire));
        assert!(track(94, 0x2000, 0x3000).valid.load(Ordering::Acquire));
    }
}
