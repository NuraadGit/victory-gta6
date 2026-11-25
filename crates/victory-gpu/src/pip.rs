//! Opt-in, synchronous present-time inspection of the previous frame's targets.
//! Nothing is allocated, polled or recorded when VICTORY_PIP is not exactly 1.

use std::collections::{BTreeMap, HashSet};
use std::sync::{Arc, Mutex, OnceLock, atomic::Ordering};

use windows::Win32::Foundation::RECT;
use windows::Win32::Graphics::Direct3D12::*;
use windows::Win32::Graphics::Dxgi::Common::*;
use windows::Win32::UI::Input::KeyboardAndMouse::{GetAsyncKeyState, VK_F6, VK_F7, VK_F8};
use windows::core::{Interface, s};

use crate::host::{Host, HostList, compile, legal_state, transition};
use crate::resource::HostTexture;

pub(crate) fn enabled() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_PIP").is_some_and(|v| v == "1"))
}

#[derive(Clone)]
pub(crate) struct Target {
    resource: usize,
    copy: Arc<HostTexture>,
}

struct Frames<T> {
    current: Vec<T>,
    previous: Vec<T>,
    seen: HashSet<usize>,
}

impl<T> Default for Frames<T> {
    fn default() -> Self {
        Self {
            current: Vec::new(),
            previous: Vec::new(),
            seen: HashSet::new(),
        }
    }
}

impl<T> Frames<T> {
    fn note(&mut self, key: usize, value: impl FnOnce() -> Option<T>) {
        if !self.seen.contains(&key)
            && let Some(value) = value()
        {
            self.seen.insert(key);
            self.current.push(value);
        }
    }

    fn advance(&mut self) {
        self.previous.clear();
        std::mem::swap(&mut self.previous, &mut self.current);
        self.seen.clear();
    }
}

struct Selection {
    index: usize,
    visible: bool,
    down: [bool; 3],
}

impl Default for Selection {
    fn default() -> Self {
        Self {
            index: 0,
            visible: true,
            down: [false; 3],
        }
    }
}

impl Selection {
    // Key order: toggle, previous, next. The high key-state bit is used;
    // the process-global "pressed since last query" bit is deliberately ignored.
    fn update(&mut self, count: usize, down: [bool; 3]) -> Option<usize> {
        let edge = std::array::from_fn::<_, 3, _>(|i| down[i] && !self.down[i]);
        self.down = down;
        if edge[0] {
            self.visible = !self.visible;
            tracing::info!(visible = self.visible, "PiP toggled");
        }
        if count == 0 {
            self.index = 0;
            return None;
        }
        self.index %= count;
        if edge[1] != edge[2] {
            self.index = if edge[2] {
                (self.index + 1) % count
            } else if self.index == 0 {
                count - 1
            } else {
                self.index - 1
            };
        }
        Some(self.index)
    }
}

#[derive(Default)]
struct Tracking {
    frames: Frames<Target>,
    selection: Selection,
    logged: Option<(usize, usize, usize, u64, u32, i32)>,
    /// The resource last shown: kept across frames by identity, not index
    /// (a frame writes some 300 targets, in an order that shifts).
    chosen: Option<usize>,
}

fn tracking() -> &'static Mutex<Tracking> {
    static TRACKING: OnceLock<Mutex<Tracking>> = OnceLock::new();
    TRACKING.get_or_init(Mutex::default)
}

/// Called at the same RT/DSV/UAV hooks as capture::note_target, including
/// in builds without diagnostics and frames that are not being dumped.
pub(crate) fn note_target(resource: crate::object::This) {
    if !enabled() || resource.is_null() {
        return;
    }
    tracking()
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .frames
        .note(resource as usize, || {
            crate::resource::host_copy(resource).map(|copy| Target {
                resource: resource as usize,
                copy,
            })
        });
}

/// Once per title present, even when the host cannot present its final
/// texture. The current/previous lists stay alive through the draw.
pub(crate) fn begin_present() -> Option<Target> {
    if !enabled() {
        return None;
    }
    // SAFETY: plain key-state queries, exactly once per key per present.
    let down = unsafe { [VK_F6, VK_F7, VK_F8].map(|key| GetAsyncKeyState(i32::from(key.0)) < 0) };
    let mut t = tracking().lock().unwrap_or_else(|e| e.into_inner());
    // Only the frame's full-size targets (at least half the widest one's
    // width): G-buffer, depth, lighting, post-processing; not the hundreds
    // of shadow tiles, probes and small intermediates.
    let full: Vec<Target> = {
        // SAFETY: the Arcs hold these resources.
        let sizes: Vec<(u64, u64)> = t
            .frames
            .previous
            .iter()
            .map(|x| unsafe { x.copy.texture.GetDesc() })
            .map(|d| (d.Width, u64::from(d.Height)))
            .collect();
        // The screen's size: the largest target shaped like a screen (width
        // 1.2 to 2.4 times its height), so wide atlases (15360x180) do not
        // set it.
        let screen = |&(w, h): &(u64, u64)| h > 0 && w * 10 >= h * 12 && w * 10 <= h * 24;
        let (sw, sh) = sizes.iter().copied().filter(screen).max_by_key(|&(w, h)| w * h).unwrap_or((0, 0));
        t.frames
            .previous
            .iter()
            .zip(sizes)
            .filter(|&(_, (w, h))| sw > 0 && w.saturating_mul(2) >= sw && h.saturating_mul(2) >= sh)
            .map(|(x, _)| x.clone())
            .collect()
    };
    let count = full.len();
    if let Some(at) = t.chosen.and_then(|r| full.iter().position(|x| x.resource == r)) {
        t.selection.index = at;
    }
    let selected = t.selection.update(count, down).map(|i| (i, full[i].clone()));
    if let Some((_, target)) = &selected {
        t.chosen = Some(target.resource);
    }
    if let Some((i, target)) = &selected {
        // SAFETY: the Arc holds this resource.
        let d = unsafe { target.copy.texture.GetDesc() };
        let key = (
            *i,
            target.resource,
            target.copy.texture.as_raw() as usize,
            d.Width,
            d.Height,
            d.Format.0,
        );
        if t.logged != Some(key) {
            tracing::info!(index = i, count, resource = format_args!("{:#x}", target.resource),
                texture = ?target.copy.texture.as_raw(), width = d.Width, height = d.Height,
                format = d.Format.0, "PiP selected previous-frame target");
            t.logged = Some(key);
        }
    } else {
        t.logged = None;
    }
    selected
        .filter(|_| t.selection.visible)
        .map(|(_, target)| target)
}

/// Release the old previous list only after the synchronous draw finishes.
pub(crate) fn end_present() {
    if enabled() {
        tracking()
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .frames
            .advance();
    }
}

fn skip(format: DXGI_FORMAT, reason: &'static str) {
    static WARNED: OnceLock<Mutex<HashSet<(i32, &'static str)>>> = OnceLock::new();
    if WARNED
        .get_or_init(Mutex::default)
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .insert((format.0, reason))
    {
        tracing::warn!(
            format = format.0,
            reason,
            "PiP skipped (logged once per format/reason)"
        );
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
struct ColourView {
    format: DXGI_FORMAT,
    grey: bool,
    tone_map: bool,
}

/// Only compatible float/unorm/snorm views. Typed integer and typed depth
/// resources cannot be reinterpreted; readable depth uses a typeless resource.
fn colour_view(format: DXGI_FORMAT, depth: bool) -> Option<ColourView> {
    let (format, grey, tone_map) = match format {
        DXGI_FORMAT_R32G32B32A32_TYPELESS | DXGI_FORMAT_R32G32B32A32_FLOAT => {
            (DXGI_FORMAT_R32G32B32A32_FLOAT, false, true)
        }
        DXGI_FORMAT_R32G32B32_TYPELESS | DXGI_FORMAT_R32G32B32_FLOAT => {
            (DXGI_FORMAT_R32G32B32_FLOAT, false, true)
        }
        DXGI_FORMAT_R16G16B16A16_TYPELESS | DXGI_FORMAT_R16G16B16A16_FLOAT => {
            (DXGI_FORMAT_R16G16B16A16_FLOAT, false, true)
        }
        DXGI_FORMAT_R32G32_TYPELESS | DXGI_FORMAT_R32G32_FLOAT => {
            (DXGI_FORMAT_R32G32_FLOAT, false, true)
        }
        DXGI_FORMAT_R32G8X24_TYPELESS => (DXGI_FORMAT_R32_FLOAT_X8X24_TYPELESS, true, false),
        DXGI_FORMAT_R10G10B10A2_TYPELESS => (DXGI_FORMAT_R10G10B10A2_UNORM, false, false),
        DXGI_FORMAT_R11G11B10_FLOAT | DXGI_FORMAT_R9G9B9E5_SHAREDEXP => (format, false, true),
        DXGI_FORMAT_R8G8B8A8_TYPELESS => (DXGI_FORMAT_R8G8B8A8_UNORM, false, false),
        DXGI_FORMAT_R16G16_TYPELESS | DXGI_FORMAT_R16G16_FLOAT => {
            (DXGI_FORMAT_R16G16_FLOAT, false, true)
        }
        DXGI_FORMAT_R32_TYPELESS => (DXGI_FORMAT_R32_FLOAT, true, !depth),
        DXGI_FORMAT_R32_FLOAT => (format, true, true),
        DXGI_FORMAT_R24G8_TYPELESS => (DXGI_FORMAT_R24_UNORM_X8_TYPELESS, true, false),
        DXGI_FORMAT_R8G8_TYPELESS => (DXGI_FORMAT_R8G8_UNORM, false, false),
        DXGI_FORMAT_R16_TYPELESS => (DXGI_FORMAT_R16_UNORM, true, false),
        DXGI_FORMAT_R16_FLOAT => (format, true, true),
        DXGI_FORMAT_R8_TYPELESS => (DXGI_FORMAT_R8_UNORM, true, false),
        DXGI_FORMAT_R8_UNORM
        | DXGI_FORMAT_R8_SNORM
        | DXGI_FORMAT_R16_UNORM
        | DXGI_FORMAT_R16_SNORM => (format, true, false),
        DXGI_FORMAT_B8G8R8A8_TYPELESS => (DXGI_FORMAT_B8G8R8A8_UNORM, false, false),
        DXGI_FORMAT_B8G8R8X8_TYPELESS => (DXGI_FORMAT_B8G8R8X8_UNORM, false, false),
        DXGI_FORMAT_R16G16B16A16_UNORM
        | DXGI_FORMAT_R16G16B16A16_SNORM
        | DXGI_FORMAT_R10G10B10A2_UNORM
        | DXGI_FORMAT_R8G8B8A8_UNORM
        | DXGI_FORMAT_R8G8B8A8_UNORM_SRGB
        | DXGI_FORMAT_R8G8B8A8_SNORM
        | DXGI_FORMAT_R16G16_UNORM
        | DXGI_FORMAT_R16G16_SNORM
        | DXGI_FORMAT_R8G8_UNORM
        | DXGI_FORMAT_R8G8_SNORM
        | DXGI_FORMAT_B8G8R8A8_UNORM
        | DXGI_FORMAT_B8G8R8A8_UNORM_SRGB
        | DXGI_FORMAT_B8G8R8X8_UNORM
        | DXGI_FORMAT_B8G8R8X8_UNORM_SRGB => (format, false, false),
        _ => return None,
    };
    Some(ColourView {
        format,
        grey,
        tone_map,
    })
}

const SHADER: &str = r#"
#define RS "DescriptorTable(SRV(t0),visibility=SHADER_VISIBILITY_PIXEL),RootConstants(num32BitConstants=6,b0,visibility=SHADER_VISIBILITY_PIXEL),StaticSampler(s0,filter=FILTER_MIN_MAG_MIP_LINEAR,addressU=TEXTURE_ADDRESS_CLAMP,addressV=TEXTURE_ADDRESS_CLAMP,addressW=TEXTURE_ADDRESS_CLAMP,visibility=SHADER_VISIBILITY_PIXEL)"
Texture2DArray<float4> image : register(t0);
SamplerState linear_clamp : register(s0);
cbuffer Options : register(b0) { float4 rectangle; uint grey; uint tone_map; };
float4 corner(uint i : SV_VertexID) : SV_Position {
    return float4(float(i & 1) * 4.0 - 1.0, float(i >> 1) * 4.0 - 1.0, 0.0, 1.0);
}
[RootSignature(RS)]
float4 picture(float4 p : SV_Position) : SV_Target {
    float2 local = p.xy - rectangle.xy;
    if (any(local < 2.0) || any(local >= rectangle.zw - 2.0))
        return float4(1.0, 1.0, 1.0, 1.0);
    float2 uv = (local - 2.0) / (rectangle.zw - 4.0);
    float3 colour = image.SampleLevel(linear_clamp, float3(uv, 0.0), 0.0).rgb;
    if (grey) colour = colour.rrr;
    if (tone_map) {
        colour = max(colour, 0.0);
        colour = colour / (1.0 + colour);
    }
    return float4(colour, 1.0);
}
"#;

/// Outer rectangle including a 2px border; image keeps its aspect ratio.
/// Very tall images shrink to fit the shown (dynamic-resolution) area.
fn rectangle(shown: (u32, u32), source: (u64, u32)) -> Option<[u32; 4]> {
    let (width, height) = shown;
    if width <= 4 || height <= 4 || source.0 == 0 || source.1 == 0 {
        return None;
    }
    let mut w = (width as f64 * 0.4 - 4.0).max(1.0);
    let mut h = w * source.1 as f64 / source.0 as f64;
    if h > (height - 4) as f64 {
        w *= (height - 4) as f64 / h;
        h = (height - 4) as f64;
    }
    let (w, h) = (
        (w.round() as u32 + 4).min(width),
        (h.round() as u32 + 4).min(height),
    );
    (w > 4 && h > 4).then_some([width - w, 0, w, h])
}

type Pipeline = (ID3D12RootSignature, ID3D12PipelineState);

/// One pair of descriptors, safe to overwrite because PiP presents wait
/// for their submissions. No heaps or HostTexture Arcs accumulate per frame.
pub(crate) struct Pass {
    views: ID3D12DescriptorHeap,
    rtvs: ID3D12DescriptorHeap,
    pipelines: BTreeMap<i32, Option<Pipeline>>,
    support: BTreeMap<i32, bool>,
    messages: Option<ID3D12InfoQueue>,
}

impl Pass {
    pub(crate) fn new(host: &Host) -> Option<Self> {
        let make = || -> windows::core::Result<Self> {
            // SAFETY: plain object creation on the host device.
            unsafe {
                Ok(Self {
                    views: host
                        .device
                        .CreateDescriptorHeap(&D3D12_DESCRIPTOR_HEAP_DESC {
                            Type: D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV,
                            NumDescriptors: 1,
                            Flags: D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE,
                            ..Default::default()
                        })?,
                    rtvs: host
                        .device
                        .CreateDescriptorHeap(&D3D12_DESCRIPTOR_HEAP_DESC {
                            Type: D3D12_DESCRIPTOR_HEAP_TYPE_RTV,
                            NumDescriptors: 1,
                            ..Default::default()
                        })?,
                    pipelines: BTreeMap::new(),
                    support: BTreeMap::new(),
                    messages: host.device.cast().ok(),
                })
            }
        };
        match make() {
            Ok(pass) => Some(pass),
            Err(e) => {
                tracing::warn!(error = %e, "PiP disabled: descriptor heap creation failed");
                None
            }
        }
    }

    fn make_pipeline(host: &Host, format: DXGI_FORMAT) -> windows::core::Result<Pipeline> {
        let vs = compile(SHADER, s!("corner"), s!("vs_5_1"))?;
        let ps = compile(SHADER, s!("picture"), s!("ps_5_1"))?;
        let code = |bytes: &[u8]| D3D12_SHADER_BYTECODE {
            pShaderBytecode: bytes.as_ptr().cast(),
            BytecodeLength: bytes.len(),
        };
        // SAFETY: live bytecode, an embedded root signature and a borrowed
        // root pointer (ManuallyDrop, as in the depth-plane pipeline).
        unsafe {
            let root: ID3D12RootSignature = host.device.CreateRootSignature(0, &ps)?;
            let stencil = D3D12_DEPTH_STENCILOP_DESC {
                StencilFailOp: D3D12_STENCIL_OP_KEEP,
                StencilDepthFailOp: D3D12_STENCIL_OP_KEEP,
                StencilPassOp: D3D12_STENCIL_OP_KEEP,
                StencilFunc: D3D12_COMPARISON_FUNC_ALWAYS,
            };
            let mut desc = D3D12_GRAPHICS_PIPELINE_STATE_DESC {
                pRootSignature: std::mem::transmute_copy(&root),
                VS: code(&vs),
                PS: code(&ps),
                SampleMask: u32::MAX,
                RasterizerState: D3D12_RASTERIZER_DESC {
                    FillMode: D3D12_FILL_MODE_SOLID,
                    CullMode: D3D12_CULL_MODE_NONE,
                    DepthClipEnable: true.into(),
                    ..Default::default()
                },
                DepthStencilState: D3D12_DEPTH_STENCIL_DESC {
                    DepthWriteMask: D3D12_DEPTH_WRITE_MASK_ZERO,
                    DepthFunc: D3D12_COMPARISON_FUNC_ALWAYS,
                    FrontFace: stencil,
                    BackFace: stencil,
                    ..Default::default()
                },
                PrimitiveTopologyType: D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE,
                NumRenderTargets: 1,
                SampleDesc: DXGI_SAMPLE_DESC {
                    Count: 1,
                    Quality: 0,
                },
                ..Default::default()
            };
            desc.RTVFormats[0] = format;
            desc.BlendState.RenderTarget[0] = D3D12_RENDER_TARGET_BLEND_DESC {
                SrcBlend: D3D12_BLEND_ONE,
                DestBlend: D3D12_BLEND_ZERO,
                BlendOp: D3D12_BLEND_OP_ADD,
                SrcBlendAlpha: D3D12_BLEND_ONE,
                DestBlendAlpha: D3D12_BLEND_ZERO,
                BlendOpAlpha: D3D12_BLEND_OP_ADD,
                LogicOp: D3D12_LOGIC_OP_NOOP,
                RenderTargetWriteMask: D3D12_COLOR_WRITE_ENABLE_ALL.0 as u8,
                ..Default::default()
            };
            let pipeline = host.device.CreateGraphicsPipelineState(&desc)?;
            Ok((root, pipeline))
        }
    }

    fn supported(&mut self, host: &Host, format: DXGI_FORMAT) -> bool {
        *self.support.entry(format.0).or_insert_with(|| {
            let mut support = D3D12_FEATURE_DATA_FORMAT_SUPPORT {
                Format: format,
                ..Default::default()
            };
            // SAFETY: properly sized feature-query output.
            let result = unsafe {
                host.device.CheckFeatureSupport(
                    D3D12_FEATURE_FORMAT_SUPPORT,
                    (&raw mut support).cast(),
                    std::mem::size_of_val(&support) as u32,
                )
            };
            let required =
                D3D12_FORMAT_SUPPORT1_TEXTURE2D.0 | D3D12_FORMAT_SUPPORT1_SHADER_SAMPLE.0;
            result.is_ok() && support.Support1.0 & required == required
        })
    }

    // D3D12 view creation returns void. Validate formats/flags ourselves;
    // when a debug info queue exists, also catch any view rejection it reports.
    fn view_errors_since(&self, start: u64) -> bool {
        let Some(messages) = &self.messages else {
            return false;
        };
        // SAFETY: queue-owned messages copied into aligned, sufficiently
        // large storage. No filter is changed and no messages are removed.
        unsafe {
            for i in start..messages.GetNumStoredMessagesAllowedByRetrievalFilter() {
                let mut size = 0;
                if messages.GetMessage(i, None, &mut size).is_err() {
                    continue;
                }
                let mut storage = vec![0usize; size.div_ceil(std::mem::size_of::<usize>())];
                let p = storage.as_mut_ptr().cast::<D3D12_MESSAGE>();
                if messages.GetMessage(i, Some(p), &mut size).is_ok()
                    && ((*p).Severity == D3D12_MESSAGE_SEVERITY_ERROR
                        || (*p).Severity == D3D12_MESSAGE_SEVERITY_CORRUPTION)
                {
                    return true;
                }
            }
        }
        false
    }

    /// Back buffer enters in COPY_DEST and leaves in RENDER_TARGET on
    /// success, otherwise unchanged. Source enters/leaves in its legal tracked
    /// state. All fallible work precedes
    /// the first barrier, so a rejected view leaves the copied image intact.
    pub(crate) fn record(
        &mut self,
        host: &Host,
        list: &HostList,
        target: &Target,
        back: &ID3D12Resource,
        shown: (u32, u32),
        final_source: (&ID3D12Resource, D3D12_RESOURCE_STATES),
    ) -> bool {
        let source = &target.copy.texture;
        // SAFETY: live resources.
        let (d, b) = unsafe { (source.GetDesc(), back.GetDesc()) };
        if d.Dimension != D3D12_RESOURCE_DIMENSION_TEXTURE2D
            || d.SampleDesc.Count != 1
            || d.Flags.0 & D3D12_RESOURCE_FLAG_DENY_SHADER_RESOURCE.0 != 0
            || source.as_raw() == back.as_raw()
        {
            skip(
                d.Format,
                "requires a shader-readable, single-sample 2D texture",
            );
            return false;
        }
        let Some(view) = colour_view(
            d.Format,
            d.Flags.0 & D3D12_RESOURCE_FLAG_ALLOW_DEPTH_STENCIL.0 != 0,
        ) else {
            skip(d.Format, "integer, typed depth, or unhandled format");
            return false;
        };
        if !self.supported(host, view.format) {
            skip(
                d.Format,
                "device does not support sampling this colour view",
            );
            return false;
        }
        let Some(rect) = rectangle(shown, (d.Width, d.Height)) else {
            return false;
        };
        let pipeline =
            self.pipelines.entry(b.Format.0).or_insert_with(|| {
                match Self::make_pipeline(host, b.Format) {
                    Ok(p) => Some(p),
                    Err(e) => {
                        tracing::warn!(error = %e, format = b.Format.0, "PiP pipeline unavailable");
                        None
                    }
                }
            });
        let Some((root, pipeline)) = pipeline else {
            return false;
        };
        let (root, pipeline) = (root.clone(), pipeline.clone());
        let srv = D3D12_SHADER_RESOURCE_VIEW_DESC {
            Format: view.format,
            ViewDimension: D3D12_SRV_DIMENSION_TEXTURE2DARRAY,
            Shader4ComponentMapping: D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING,
            Anonymous: D3D12_SHADER_RESOURCE_VIEW_DESC_0 {
                Texture2DArray: D3D12_TEX2D_ARRAY_SRV {
                    MipLevels: 1,
                    ArraySize: 1,
                    ..Default::default()
                },
            },
        };
        let rtv = D3D12_RENDER_TARGET_VIEW_DESC {
            Format: b.Format,
            ViewDimension: D3D12_RTV_DIMENSION_TEXTURE2D,
            ..Default::default()
        };
        // The final-image copy has already restored its source before this
        // method is called, including when the selected source is that image.
        // queue::present temporarily marks the final source COPY_SOURCE in
        // the tracker. Host::present has restored its actual entry state.
        let tracked = if source.as_raw() == final_source.0.as_raw() {
            final_source.1.0 as u32
        } else {
            target.copy.state.load(Ordering::Relaxed)
        };
        let state = D3D12_RESOURCE_STATES(legal_state(source, tracked) as i32);
        let readable = D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE;
        let constants = [
            (rect[0] as f32).to_bits(),
            (rect[1] as f32).to_bits(),
            (rect[2] as f32).to_bits(),
            (rect[3] as f32).to_bits(),
            u32::from(view.grey),
            u32::from(view.tone_map),
        ];
        // SAFETY: open host list; compatible resource/view formats checked
        // above. PiP presents are synchronous: heaps and source outlive GPU use.
        unsafe {
            let start = self
                .messages
                .as_ref()
                .map_or(0, |q| q.GetNumStoredMessagesAllowedByRetrievalFilter());
            host.device.CreateShaderResourceView(
                source,
                Some(&srv),
                self.views.GetCPUDescriptorHandleForHeapStart(),
            );
            let rtv_handle = self.rtvs.GetCPUDescriptorHandleForHeapStart();
            host.device
                .CreateRenderTargetView(back, Some(&rtv), rtv_handle);
            if self.view_errors_since(start) {
                skip(d.Format, "device debug layer rejected the view");
                return false;
            }
            let commands = &list.list;
            if state != readable {
                commands.ResourceBarrier(&[transition(source, state, readable)]);
            }
            commands.ResourceBarrier(&[transition(
                back,
                D3D12_RESOURCE_STATE_COPY_DEST,
                D3D12_RESOURCE_STATE_RENDER_TARGET,
            )]);
            commands.SetDescriptorHeaps(&[Some(self.views.clone())]);
            commands.SetGraphicsRootSignature(&root);
            commands.SetPipelineState(&pipeline);
            commands
                .SetGraphicsRootDescriptorTable(0, self.views.GetGPUDescriptorHandleForHeapStart());
            commands.SetGraphicsRoot32BitConstants(1, 6, constants.as_ptr().cast(), 0);
            commands.OMSetRenderTargets(1, Some(&rtv_handle), false, None);
            commands.RSSetViewports(&[D3D12_VIEWPORT {
                TopLeftX: rect[0] as f32,
                TopLeftY: rect[1] as f32,
                Width: rect[2] as f32,
                Height: rect[3] as f32,
                MinDepth: 0.0,
                MaxDepth: 1.0,
            }]);
            commands.RSSetScissorRects(&[RECT {
                left: rect[0] as i32,
                top: rect[1] as i32,
                right: (rect[0] + rect[2]) as i32,
                bottom: (rect[1] + rect[3]) as i32,
            }]);
            commands.IASetPrimitiveTopology(
                windows::Win32::Graphics::Direct3D::D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST,
            );
            commands.DrawInstanced(3, 1, 0, 0);
            if state != readable {
                commands.ResourceBarrier(&[transition(source, readable, state)]);
            }
        }
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn selection_edges_wrap_toggle_and_resize() {
        let mut s = Selection::default();
        assert_eq!(s.update(3, [false; 3]), Some(0));
        assert_eq!(s.update(3, [false, true, false]), Some(2));
        assert_eq!(s.update(3, [false, true, false]), Some(2)); // held
        s.update(3, [false; 3]);
        assert_eq!(s.update(3, [false, false, true]), Some(0));
        s.update(3, [true, false, false]);
        assert!(!s.visible);
        s.update(3, [true, false, false]);
        assert!(!s.visible);
        s.update(3, [false; 3]);
        s.update(3, [true, false, false]);
        assert!(s.visible);
        s.index = 7;
        assert_eq!(s.update(2, [false; 3]), Some(1));
        assert_eq!(s.update(2, [false, true, true]), Some(1));
        assert_eq!(s.update(0, [false, false, true]), None);
        assert_eq!(s.update(1, [false, false, true]), Some(0));
    }

    #[test]
    fn targets_keep_first_use_order_and_only_two_frames() {
        let mut f = Frames::default();
        f.note(4, || Some(Arc::new(4)));
        f.note(2, || Some(Arc::new(2)));
        f.note(4, || panic!("duplicate hook must not acquire another Arc"));
        assert!(f.previous.is_empty());
        f.advance();
        let old = Arc::downgrade(&f.previous[0]);
        assert_eq!(f.previous.iter().map(|v| **v).collect::<Vec<_>>(), [4, 2]);
        f.note(9, || Some(Arc::new(9)));
        assert_eq!(*f.previous[0], 4);
        f.advance();
        assert!(old.upgrade().is_none());
        assert_eq!(*f.previous[0], 9);
        f.advance();
        assert!(f.previous.is_empty());
    }

    #[test]
    fn colour_views_reject_integers_and_handle_depth() {
        for f in [
            DXGI_FORMAT_R8G8B8A8_UINT,
            DXGI_FORMAT_R32_UINT,
            DXGI_FORMAT_R16G16B16A16_SINT,
            DXGI_FORMAT_D24_UNORM_S8_UINT,
            DXGI_FORMAT_UNKNOWN,
        ] {
            assert!(colour_view(f, false).is_none());
        }
        let d = colour_view(DXGI_FORMAT_R24G8_TYPELESS, true).unwrap();
        assert_eq!(d.format, DXGI_FORMAT_R24_UNORM_X8_TYPELESS);
        assert!(d.grey && !d.tone_map);
        assert!(
            !colour_view(DXGI_FORMAT_R32_TYPELESS, true)
                .unwrap()
                .tone_map
        );
        assert!(colour_view(DXGI_FORMAT_R32_FLOAT, false).unwrap().tone_map);
        assert!(colour_view(DXGI_FORMAT_R8_UNORM, false).unwrap().grey);
    }

    #[test]
    fn rectangle_fits_shown_area_and_keeps_image_aspect() {
        assert_eq!(
            rectangle((1920, 1080), (1920, 1080)),
            Some([1152, 0, 768, 434])
        );
        let [x, y, w, h] = rectangle((640, 360), (100, 1000)).unwrap();
        assert_eq!((x + w, y + h), (640, 360));
        assert!(((w - 4) as f64 / (h - 4) as f64 - 0.1).abs() < 0.003);
        assert_eq!(rectangle((4, 4), (1, 1)), None);
    }

    #[test]
    fn pip_shaders_compile_without_a_device() {
        assert!(
            !compile(SHADER, s!("corner"), s!("vs_5_1"))
                .unwrap()
                .is_empty()
        );
        assert!(
            !compile(SHADER, s!("picture"), s!("ps_5_1"))
                .unwrap()
                .is_empty()
        );
    }
}
