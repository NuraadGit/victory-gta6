//! Title textures the GPU writes, placed in a title heap, share video
//! memory on the host as they share the heap on the console. Engines place
//! short-lived render targets and unordered access textures at
//! overlapping offsets of one heap, so passes reuse each other's memory and
//! the heap costs the largest set alive at once; a committed host texture
//! each costs the sum of them all.
//!
//! Each title heap gets a host heap of 9/8 its size, made with the first
//! such texture placed in it (two on resource heap tier 1, which keeps
//! render targets and depth textures apart from other textures). A texture
//! goes at 9/8 of its offset there ([`scaled`]) when the host's size and
//! alignment for it fit before 9/8 of the end of its console footprint.
//! The order of offsets is kept, so host textures overlap only where title
//! textures do; the eighth more room takes the host's own layouts, which
//! for full-HD and 1440p targets of 8 and 16 bytes a texel are about 5 %
//! larger than the console's (at the same offsets those did not fit).
//! Others (sizes that do not fit, textures whose contents come from title
//! memory, multisampled ones, depth textures also written through
//! unordered access) stay committed.
//!
//! Overlapping host textures are aliases. Before an alias is used after
//! another one that holds part of its memory, an aliasing barrier, and a
//! render target or depth texture is initialized: discarded. That holds
//! for a placed texture's first use too. The title's own barriers do not
//! say enough (a console aliasing barrier may name resources the host does
//! not have), so the lists track which texture last held each piece of a
//! host heap's memory: a use that takes memory another texture the list
//! used holds is recorded where it happens ([`ListAliases::use_texture`]);
//! a list's first use of memory is judged at submission, against what the
//! lists before it left ([`at_submission`]), and prepared by the list's
//! prelude. The title's aliasing barriers become host aliasing barriers
//! on any resource, its discards of such textures host discards.
//!
//! shortcut: a host heap is 9/8 of the title heap's full size, also when
//! few of its textures are placed there or they do not overlap; such title
//! heaps cost up to that much more than their textures did committed.
//! Upgrade (host heaps for windows of the title heap) if a title shows
//! that. A title texture that overlaps another one only in part and relies
//! on its other part surviving loses it: the host lays textures out its own
//! way, so which bytes overlap differs.

use std::sync::atomic::{AtomicU64, Ordering::Relaxed};
use std::sync::{Arc, Mutex, OnceLock};

use windows::Win32::Graphics::Direct3D12::*;
use windows::Win32::Graphics::Dxgi::Common::DXGI_FORMAT;
use windows::core::Interface;

use crate::resource::HostTexture;

/// Opt-in for this fork. Read once per process, before creating textures.
pub(crate) fn enabled() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_ALIAS_TARGETS").is_some_and(|v| v == "1"))
}

const RENDER_TARGET: u32 = 0x4;
const DEPTH_WRITE: u32 = 0x10;
const ALLOW_RENDER_TARGET: i32 = 0x1;
const ALLOW_DEPTH_STENCIL: i32 = 0x2;

/// Host heaps alive and their bytes; textures placed in them and the bytes
/// they would take committed; textures that did not fit their place.
static HEAPS: AtomicU64 = AtomicU64::new(0);
static HEAP_BYTES: AtomicU64 = AtomicU64::new(0);
static PLACED: AtomicU64 = AtomicU64::new(0);
static PLACED_BYTES: AtomicU64 = AtomicU64::new(0);
static UNFIT: AtomicU64 = AtomicU64::new(0);

/// Host heaps alive, their bytes, textures placed in them, the bytes those
/// would take committed, textures that did not fit their place (so far).
pub(crate) fn counts() -> (u64, u64, u64, u64, u64) {
    (
        HEAPS.load(Relaxed),
        HEAP_BYTES.load(Relaxed),
        PLACED.load(Relaxed),
        PLACED_BYTES.load(Relaxed),
        UNFIT.load(Relaxed),
    )
}

/// Pieces of a host heap's memory, `(start, end, texture)`: which texture
/// last held each. A texture is named by a number of its own (a freed
/// one's address may come back).
type Pieces = Vec<(u64, u64, u64)>;

/// What holds `start..end` for texture `id`: whether any piece overlaps,
/// whether another texture's does, and the bytes its own pieces cover.
fn look(pieces: &Pieces, start: u64, end: u64, id: u64) -> (bool, bool, u64) {
    let mut found = (false, false, 0);
    for &(s, e, t) in pieces {
        if s < end && start < e {
            found.0 = true;
            if t == id {
                found.2 += e.min(end) - s.max(start);
            } else {
                found.1 = true;
            }
        }
    }
    found
}

/// Texture `id` takes `start..end`: the pieces it overlaps keep what is left
/// of them outside it.
fn hold(pieces: &mut Pieces, start: u64, end: u64, id: u64) {
    let mut left = Vec::new();
    pieces.retain(|&(s, e, t)| {
        if s >= end || start >= e {
            return true;
        }
        if s < start {
            left.push((s, start, t));
        }
        if e > end {
            left.push((end, e, t));
        }
        false
    });
    pieces.extend(left);
    pieces.push((start, end, id));
}

/// Where an offset of a title heap lies in its host heap: 9/8 of it, on a
/// 64 KiB boundary. Order is kept, so textures apart in the title heap are
/// apart in the host heap too, and a texture has 1/8 more room than its
/// console footprint.
fn scaled(offset: u64) -> Option<u64> {
    let at = u64::try_from(u128::from(offset) * 9 / 8).ok()?;
    at.checked_next_multiple_of(D3D12_DEFAULT_RESOURCE_PLACEMENT_ALIGNMENT as u64)
}

/// A host heap standing in for a title heap.
pub(crate) struct HostHeap {
    heap: ID3D12Heap,
    size: u64,
    /// Which texture last held each piece, as of the lists submitted so far.
    held: Mutex<Pieces>,
}

impl Drop for HostHeap {
    fn drop(&mut self) {
        HEAPS.fetch_sub(1, Relaxed);
        HEAP_BYTES.fetch_sub(self.size, Relaxed);
        crate::resource::HOST_TEXTURES[1].fetch_sub(self.size, Relaxed);
    }
}

/// A title heap's host heaps, made on first use: one for every texture, or
/// on resource heap tier 1 one for render targets and depth textures and
/// one for the others.
#[derive(Default)]
pub(crate) struct HostHeaps(Mutex<[Option<Arc<HostHeap>>; 2]>);

impl HostHeaps {
    fn get(
        &self,
        host: &crate::host::Host,
        size: u64,
        tier1: bool,
        targets: bool,
    ) -> Option<Arc<HostHeap>> {
        let kind = usize::from(tier1 && !targets);
        let mut heaps = self.0.lock().unwrap_or_else(|e| e.into_inner());
        if let Some(h) = &heaps[kind] {
            return Some(h.clone());
        }
        let flags = match (tier1, targets) {
            (false, _) => D3D12_HEAP_FLAG_NONE,
            (true, true) => D3D12_HEAP_FLAG_ALLOW_ONLY_RT_DS_TEXTURES,
            (true, false) => D3D12_HEAP_FLAG_ALLOW_ONLY_NON_RT_DS_TEXTURES,
        };
        let size = size.next_multiple_of(D3D12_DEFAULT_RESOURCE_PLACEMENT_ALIGNMENT as u64);
        let desc = D3D12_HEAP_DESC {
            SizeInBytes: size,
            Properties: D3D12_HEAP_PROPERTIES {
                Type: D3D12_HEAP_TYPE_DEFAULT,
                ..Default::default()
            },
            Alignment: 0,
            Flags: flags,
        };
        let mut heap: Option<ID3D12Heap> = None;
        // SAFETY: a plain creation with a valid description.
        if let Err(e) = unsafe { host.device.CreateHeap(&desc, &mut heap) } {
            tracing::warn!(error = %e, size, "no host heap: its textures stay committed");
            return None;
        }
        let heap = heap?;
        crate::host::name(&heap, || format!("title heap of {size:#x} bytes"));
        keep_resident(&heap);
        HEAPS.fetch_add(1, Relaxed);
        HEAP_BYTES.fetch_add(size, Relaxed);
        crate::resource::HOST_TEXTURES[1].fetch_add(size, Relaxed);
        let h = Arc::new(HostHeap {
            heap,
            size,
            held: Mutex::default(),
        });
        heaps[kind] = Some(h.clone());
        Some(h)
    }
}

/// Gives host heap `heap` a high residency priority: its render targets
/// and textures shaders write are used every frame, so when video memory
/// runs short they are the last moved out of it. The priority is the
/// heap's: a texture placed in it takes none of its own (asked of a placed
/// texture, `SetResidencyPriority` removed the device at the next
/// submission).
fn keep_resident(heap: &ID3D12Heap) {
    let Some(host) = crate::host::host() else {
        return;
    };
    let (Ok(device), Ok(pageable)) = (
        host.device.cast::<ID3D12Device1>(),
        heap.cast::<ID3D12Pageable>(),
    ) else {
        return;
    };
    // SAFETY: one live pageable object and its priority.
    let set =
        unsafe { device.SetResidencyPriority(1, &Some(pageable), &D3D12_RESIDENCY_PRIORITY_HIGH) };
    if let Err(e) = set {
        static ONCE: OnceLock<()> = OnceLock::new();
        ONCE.get_or_init(|| tracing::debug!(error = %e, "no residency priority for host heaps"));
    }
}

/// Where a title texture lies in its title heap: the heap's host heaps and
/// size, the texture's offset and the bytes its console footprint takes
/// (all in the title heap: [`make`] scales them).
pub(crate) struct Spot<'a> {
    pub(crate) heaps: &'a HostHeaps,
    pub(crate) heap_size: u64,
    pub(crate) offset: u64,
    pub(crate) room: u64,
}

/// A host texture's place in a host heap: the heap, and where it lies there.
pub(crate) struct Placement {
    heap: Arc<HostHeap>,
    start: u64,
    end: u64,
    /// Its own number in [`Pieces`].
    id: u64,
    /// The state a discard needs (render targets and depth textures, whose
    /// contents must be initialized after another alias held their memory).
    discard: Option<u32>,
}

#[cfg(test)]
impl Placement {
    /// The host heap it lies in and its offset there.
    pub(crate) fn at(&self) -> (&Arc<HostHeap>, u64) {
        (&self.heap, self.start)
    }
}

impl Drop for Placement {
    fn drop(&mut self) {
        PLACED.fetch_sub(1, Relaxed);
        PLACED_BYTES.fetch_sub(self.end - self.start, Relaxed);
    }
}

#[cfg(test)]
impl HostHeap {
    /// The host heap itself.
    pub(crate) fn heap(&self) -> &ID3D12Heap {
        &self.heap
    }
}

/// The host's resource heap tier: 1 keeps render targets and depth
/// textures in heaps of their own.
fn tier1(host: &crate::host::Host) -> bool {
    static TIER1: OnceLock<bool> = OnceLock::new();
    *TIER1.get_or_init(|| {
        let mut options = D3D12_FEATURE_DATA_D3D12_OPTIONS::default();
        // SAFETY: the query writes its own structure of the given size.
        let known = unsafe {
            host.device.CheckFeatureSupport(
                D3D12_FEATURE_D3D12_OPTIONS,
                (&raw mut options).cast(),
                size_of_val(&options) as u32,
            )
        }
        .is_ok();
        let tier1 = !known || options.ResourceHeapTier.0 < D3D12_RESOURCE_HEAP_TIER_2.0;
        tracing::info!(tier = options.ResourceHeapTier.0, "host resource heap tier");
        tier1
    })
}

/// The host texture of host description `desc` (castable to `castable`)
/// at `spot`, if the host's size and alignment for it fit there: the
/// texture, starting in the common state, and its place. `None`: it goes
/// committed.
pub(crate) fn make(
    host: &crate::host::Host,
    spot: &Spot,
    desc: &D3D12_RESOURCE_DESC,
    clear: Option<*const D3D12_CLEAR_VALUE>,
    castable: &[DXGI_FORMAT],
) -> Option<(ID3D12Resource, Placement)> {
    if !enabled() || desc.SampleDesc.Count != 1 {
        return None;
    }
    let desc1 = D3D12_RESOURCE_DESC1 {
        Dimension: desc.Dimension,
        Alignment: desc.Alignment,
        Width: desc.Width,
        Height: desc.Height,
        DepthOrArraySize: desc.DepthOrArraySize,
        MipLevels: desc.MipLevels,
        Format: desc.Format,
        SampleDesc: desc.SampleDesc,
        Layout: desc.Layout,
        Flags: desc.Flags,
        SamplerFeedbackMipRegion: Default::default(),
    };
    // Castable formats may change the host's layout: sized with them,
    // through the interfaces that take them, or not placed.
    let device = if castable.is_empty() {
        None
    } else {
        Some(host.device.cast::<ID3D12Device12>().ok()?)
    };
    let info = match &device {
        Some(device) => {
            let (count, formats) = (castable.len() as u32, castable.as_ptr());
            // SAFETY: one description and its castable list, live through the call.
            unsafe {
                device.GetResourceAllocationInfo3(0, 1, &desc1, Some(&count), Some(&formats), None)
            }
        }
        // SAFETY: a plain query on a host description.
        None => unsafe { host.device.GetResourceAllocationInfo(0, &[*desc]) },
    };
    let title_end = spot.offset.checked_add(spot.room)?;
    if title_end > spot.heap_size {
        return None;
    }
    let (start, end) = (scaled(spot.offset)?, scaled(title_end)?);
    let fits = info.SizeInBytes != u64::MAX
        && info.Alignment.is_power_of_two()
        && start.is_multiple_of(info.Alignment)
        && start
            .checked_add(info.SizeInBytes)
            .is_some_and(|e| e <= end);
    if !fits {
        UNFIT.fetch_add(1, Relaxed);
        tracing::debug!(
            ?desc,
            offset = spot.offset,
            room = spot.room,
            size = info.SizeInBytes,
            alignment = info.Alignment,
            "a placed texture does not fit its place on the host"
        );
        return None;
    }
    let targets = desc.Flags.0 & (ALLOW_RENDER_TARGET | ALLOW_DEPTH_STENCIL) != 0;
    let heap = spot
        .heaps
        .get(host, scaled(spot.heap_size)?, tier1(host), targets)?;
    let mut texture: Option<ID3D12Resource> = None;
    // SAFETY: host descriptions, a live heap with room at the offset
    // (checked above), the clear value and castable list live through the
    // call; our out pointer.
    let made = unsafe {
        match &device {
            Some(device) => device.CreatePlacedResource2(
                &heap.heap,
                start,
                &desc1,
                D3D12_BARRIER_LAYOUT_COMMON,
                clear,
                Some(castable),
                &mut texture,
            ),
            None => host.device.CreatePlacedResource(
                &heap.heap,
                start,
                desc,
                D3D12_RESOURCE_STATE_COMMON,
                clear,
                &mut texture,
            ),
        }
    };
    if let Err(e) = made {
        tracing::warn!(error = %e, ?desc, offset = spot.offset, "no placed host texture: committed instead");
        return None;
    }
    static NEXT: AtomicU64 = AtomicU64::new(1);
    PLACED.fetch_add(1, Relaxed);
    PLACED_BYTES.fetch_add(info.SizeInBytes, Relaxed);
    let discard = if desc.Flags.0 & ALLOW_DEPTH_STENCIL != 0 {
        Some(DEPTH_WRITE)
    } else if desc.Flags.0 & ALLOW_RENDER_TARGET != 0 {
        Some(RENDER_TARGET)
    } else {
        None
    };
    let placement = Placement {
        heap,
        start,
        end: start + info.SizeInBytes,
        id: NEXT.fetch_add(1, Relaxed),
        discard,
    };
    Some((texture?, placement))
}

/// An aliasing barrier on any placed resource.
pub(crate) fn any_alias() -> D3D12_RESOURCE_BARRIER {
    D3D12_RESOURCE_BARRIER {
        Type: D3D12_RESOURCE_BARRIER_TYPE_ALIASING,
        Flags: D3D12_RESOURCE_BARRIER_FLAG_NONE,
        Anonymous: D3D12_RESOURCE_BARRIER_0 {
            Aliasing: std::mem::ManuallyDrop::new(D3D12_RESOURCE_ALIASING_BARRIER::default()),
        },
    }
}

/// The state `copy` is discarded in, if it is a placed render target or
/// depth texture (without a writable twin, whose copies keep their own
/// account of which side is newer).
pub(crate) fn discard_state(copy: &HostTexture) -> Option<u32> {
    let p = copy.placement.as_ref()?;
    p.discard
}

/// Records, for a placed texture taking memory another alias held: an
/// aliasing barrier if `barrier`, then its discard (render targets and
/// depth textures) from state `now`, which it is left in.
pub(crate) fn initialize(
    list: &ID3D12GraphicsCommandList,
    copy: &HostTexture,
    now: u32,
    barrier: bool,
) {
    let mut barriers = Vec::with_capacity(2);
    if barrier {
        barriers.push(any_alias());
    }
    let state = discard_state(copy);
    let to = |a: u32, b: u32| {
        crate::host::transition(
            &copy.texture,
            D3D12_RESOURCE_STATES(a as i32),
            D3D12_RESOURCE_STATES(b as i32),
        )
    };
    if let Some(s) = state
        && s != now
    {
        barriers.push(to(now, s));
    }
    // SAFETY: an open list and a live texture.
    unsafe {
        if !barriers.is_empty() {
            list.ResourceBarrier(&barriers);
        }
        if let Some(s) = state {
            list.DiscardResource(&copy.texture, None);
            if s != now {
                list.ResourceBarrier(&[to(s, now)]);
            }
        }
    }
    DISCARDS.fetch_add(u64::from(state.is_some()), Relaxed);
}

/// Discards [`initialize`] recorded (tests).
pub(crate) static DISCARDS: AtomicU64 = AtomicU64::new(0);

/// A list's uses of placed textures.
#[derive(Default)]
pub(crate) struct ListAliases {
    /// Each time a placed texture takes memory, in recorded order, and
    /// whether the list recorded its barrier and discard there (it took
    /// memory a texture the list used before held); the others are its
    /// first uses of that memory, judged at submission.
    events: Vec<(Arc<HostTexture>, bool)>,
    /// Which texture the list's uses left in each piece, by host heap.
    held: Vec<(usize, Pieces)>,
}

impl ListAliases {
    /// A use of placed texture `copy` at this point of the list: true if it
    /// takes memory another texture the list used holds (its barrier and
    /// discard are to be recorded here).
    pub(crate) fn use_texture(&mut self, copy: &Arc<HostTexture>) -> bool {
        let Some(p) = copy.placement.as_ref() else {
            return false;
        };
        let key = Arc::as_ptr(&p.heap) as usize;
        let at = match self.held.iter().position(|(k, _)| *k == key) {
            Some(at) => at,
            None => {
                self.held.push((key, Vec::new()));
                self.held.len() - 1
            }
        };
        let pieces = &mut self.held[at].1;
        let (any, other, _) = look(pieces, p.start, p.end, p.id);
        if any && !other {
            // Its own memory still: every piece over it is its own (only
            // other textures' uses take pieces away).
            return false;
        }
        hold(pieces, p.start, p.end, p.id);
        self.events.push((copy.clone(), other));
        other
    }

    pub(crate) fn clear(&mut self) {
        self.events.clear();
        self.held.clear();
    }
}

/// At a list's submission (in submission order): whether its prelude needs
/// an aliasing barrier, and the textures it discards, for the list's first
/// uses of memory that other textures held after the lists before it (or
/// that a new texture takes); the host heaps then hold what the list leaves.
pub(crate) fn at_submission(aliases: &ListAliases) -> (bool, Vec<Arc<HostTexture>>) {
    let mut barrier = false;
    let mut discards = Vec::new();
    for (copy, recorded) in &aliases.events {
        let Some(p) = copy.placement.as_ref() else {
            continue;
        };
        let mut held = p.heap.held.lock().unwrap_or_else(|e| e.into_inner());
        if !recorded {
            let (_, other, own) = look(&held, p.start, p.end, p.id);
            barrier |= other;
            if (other || own < p.end - p.start) && discard_state(copy).is_some() {
                discards.push(copy.clone());
            }
        }
        hold(&mut held, p.start, p.end, p.id);
    }
    (barrier, discards)
}

/// The frame-rate line's companion: host heaps and what they hold.
pub(crate) fn log() {
    let (heaps, bytes, placed, placed_bytes, unfit) = counts();
    if heaps > 0 || unfit > 0 {
        tracing::info!(
            "host heaps: {heaps} ({:.0} MB) holding {placed} title textures ({:.0} MB had they been committed); {unfit} did not fit their place",
            bytes as f64 / 1e6,
            placed_bytes as f64 / 1e6,
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn scaled_offsets_keep_disjoint_footprints_apart_and_reject_overflow() {
        assert_eq!(scaled(0), Some(0));
        assert_eq!(scaled(4 << 20), Some(9 << 19));
        assert_eq!(scaled(u64::MAX), None);
        for start in [0, 4096, 65536, 1234567] {
            let end = start + 123456;
            assert!(scaled(start).unwrap() <= scaled(end).unwrap());
        }
    }

    #[test]
    fn a_texture_takes_its_memory_and_others_keep_the_rest() {
        let mut pieces = Pieces::new();
        hold(&mut pieces, 0, 10, 1);
        assert_eq!(look(&pieces, 0, 10, 1), (true, false, 10));
        hold(&mut pieces, 4, 6, 2);
        assert_eq!(look(&pieces, 0, 10, 1), (true, true, 8));
        assert_eq!(
            look(&pieces, 0, 4, 1),
            (true, false, 4),
            "1 still holds its start"
        );
        assert_eq!(look(&pieces, 10, 20, 3), (false, false, 0));
        hold(&mut pieces, 0, 10, 1);
        assert_eq!(pieces, vec![(0, 10, 1)]);
    }
}
