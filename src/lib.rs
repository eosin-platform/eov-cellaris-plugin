//! Cellaris: predicted multiplex-IF markers overlaid on H&E.
//!
//! The model runs on 256px windows at a 128px stride, always at the size it was
//! trained on, and every 128px cell is the feather-weighted blend of the four
//! windows over it -- the method measured at a seam score of 1.01 in Python.
//! Windows are read at level 0 and predicted on a worker thread; the filter
//! only blends and paints.

mod model;
mod overlay;

use std::collections::{HashMap, HashSet};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::mpsc::{self, Sender};
use std::sync::{Arc, Mutex, OnceLock};
use std::time::{Duration, Instant};

use abi_stable::std_types::{ROption, RString, RVec};
use eov_plugin_api::ffi::{
    ActionResponseFFI, GpuFilterContextFFI, HostApiVTable, HudToolbarButtonFFI, PluginVTable,
    ToolbarButtonFFI, UiPropertyFFI, ViewportContextMenuItemFFI, ViewportFilterFFI,
    ViewportOverlayComponentRequestFFI, ViewportOverlayPointFFI, ViewportOverlayPolygonFFI,
    ViewportOverlayVertexFFI, ViewportSnapshotFFI,
};

const FILTER_ID: &str = "cellaris_markers";
const WINDOW: i64 = model::WINDOW as i64;
const STRIDE: i64 = WINDOW / 2;
const MAX_CELLS: usize = 600;
const MAX_WINDOWS: usize = 800;
const MAX_CELL_CACHE: usize = 2000;
const ALPHA: u16 = 170;
const REDRAW_INTERVAL: Duration = Duration::from_millis(150);

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
struct Key {
    file_id: i32,
    x: i64,
    y: i64,
}

type Planes = Arc<Vec<u8>>;

struct Shared {
    /// Scaled marker planes per window, `markers x WINDOW x WINDOW`.
    windows: Mutex<HashMap<Key, Planes>>,
    /// Blended RGB per cell, `STRIDE x STRIDE x 3`.
    cells: Mutex<HashMap<Key, Planes>>,
    markers: Mutex<Option<Vec<overlay::Marker>>>,
    requested: Mutex<HashSet<Key>>,
    /// Windows the current view needs; the worker skips anything else.
    visible: Mutex<HashSet<Key>>,
    host: Mutex<Option<HostApiVTable>>,
    failure: Mutex<Option<String>>,
}

static SHARED: OnceLock<Arc<Shared>> = OnceLock::new();
static ENABLED: AtomicBool = AtomicBool::new(false);
static SENDER: OnceLock<Mutex<Option<Sender<Key>>>> = OnceLock::new();
static PENDING: AtomicUsize = AtomicUsize::new(0);

fn shared() -> &'static Arc<Shared> {
    SHARED.get_or_init(|| {
        Arc::new(Shared {
            windows: Mutex::new(HashMap::new()),
            cells: Mutex::new(HashMap::new()),
            markers: Mutex::new(None),
            requested: Mutex::new(HashSet::new()),
            visible: Mutex::new(HashSet::new()),
            host: Mutex::new(None),
            failure: Mutex::new(None),
        })
    })
}

/// Raised-cosine taper over the overlap, as `inference.feather_window`. Two
/// windows a stride apart sum to one everywhere they overlap.
fn feather() -> &'static [f32; model::WINDOW as usize] {
    static RAMP: OnceLock<[f32; model::WINDOW as usize]> = OnceLock::new();
    RAMP.get_or_init(|| {
        let overlap = STRIDE as usize;
        let last = WINDOW as usize - 1;
        let mut ramp = [1f32; model::WINDOW as usize];
        for i in 0..overlap {
            let t = std::f32::consts::PI * (i as f32 + 0.5) / overlap as f32;
            let taper = (0.5 * (1.0 - t.cos())).max(1e-3);
            ramp[i] = taper;
            ramp[last - i] = taper;
        }
        ramp
    })
}

fn draw_frame(pixels: &mut [u8], width: usize, height: usize) {
    const EDGE: usize = 4;
    const AMBER: [u8; 3] = [255, 176, 0];
    if width <= 2 * EDGE || height <= 2 * EDGE || pixels.len() < width * height * 4 {
        return;
    }
    let mut paint = |x: usize, y: usize| {
        let i = (y * width + x) * 4;
        pixels[i..i + 3].copy_from_slice(&AMBER);
    };
    for y in (0..EDGE).chain(height - EDGE..height) {
        for x in 0..width {
            paint(x, y);
        }
    }
    for y in EDGE..height - EDGE {
        for x in (0..EDGE).chain(width - EDGE..width) {
            paint(x, y);
        }
    }
}

fn announce(enabled: bool) {
    let state = shared();
    if !enabled {
        state.visible.lock().unwrap().clear();
    }
    eprintln!("[cellaris] overlay {}", if enabled { "on" } else { "off" });
    let host = { *state.host.lock().unwrap() };
    if let Some(host) = host {
        let _ = (host.set_toolbar_button_active)(
            host.context,
            RString::from("toggle_cellaris"),
            enabled,
        );
    }
}

fn report(state: &Shared, error: String) {
    let mut slot = state.failure.lock().unwrap();
    if slot.as_deref() != Some(error.as_str()) {
        eprintln!("[cellaris] {error}");
    }
    *slot = Some(error);
}

// ---------------------------------------------------------------------------
// Worker
// ---------------------------------------------------------------------------

fn ensure_worker() {
    let slot = SENDER.get_or_init(|| Mutex::new(None));
    let mut guard = slot.lock().unwrap();
    if guard.is_some() {
        return;
    }

    let (sender, receiver) = mpsc::channel::<Key>();
    *guard = Some(sender);
    let state = Arc::clone(shared());

    std::thread::spawn(move || {
        let mut loaded = match model::Model::load() {
            Ok(model) => model,
            Err(error) => {
                report(&state, format!("model did not load: {error}"));
                return;
            }
        };
        let markers = match overlay::select(&loaded.channel_names, &overlay::DEFAULT_MARKERS) {
            Ok(markers) => markers,
            Err(error) => {
                report(&state, error);
                return;
            }
        };
        eprintln!(
            "[cellaris] loaded {} channels from {}, painting {}",
            loaded.channel_names.len(),
            loaded.path.display(),
            overlay::DEFAULT_MARKERS.map(|(name, _)| name).join(", ")
        );
        *state.markers.lock().unwrap() = Some(markers.clone());

        let plane = (WINDOW * WINDOW) as usize;
        let mut last_redraw = Instant::now();
        let mut computed = 0usize;
        let mut spent = Duration::ZERO;

        let finish = |host: Option<HostApiVTable>, last_redraw: &mut Instant| {
            let left = PENDING.fetch_sub(1, Ordering::Relaxed).saturating_sub(1);
            if let Some(host) = host {
                if left == 0 || last_redraw.elapsed() >= REDRAW_INTERVAL {
                    let _ = (host.request_render)(host.context);
                    *last_redraw = Instant::now();
                }
            }
        };

        while let Ok(key) = receiver.recv() {
            let host = { *state.host.lock().unwrap() };
            let Some(host) = host else {
                report(&state, "no host handle yet; window skipped".to_string());
                finish(None, &mut last_redraw);
                continue;
            };
            if !state.visible.lock().unwrap().contains(&key) {
                state.requested.lock().unwrap().remove(&key);
                finish(Some(host), &mut last_redraw);
                continue;
            }
            let started = Instant::now();

            let bytes = (host.read_region)(
                host.context,
                key.file_id,
                0,
                key.x,
                key.y,
                WINDOW as u32,
                WINDOW as u32,
            );
            let Ok(rgba) = bytes.into_result() else {
                report(&state, "the host refused a window read".to_string());
                finish(Some(host), &mut last_redraw);
                continue;
            };
            let predicted = match loaded.predict(rgba.as_slice()) {
                Ok(values) => values,
                Err(error) => {
                    report(&state, format!("inference failed: {error}"));
                    finish(Some(host), &mut last_redraw);
                    continue;
                }
            };

            let mut scaled = vec![0u8; markers.len() * plane];
            for (index, marker) in markers.iter().enumerate() {
                let source = &predicted[marker.channel * plane..(marker.channel + 1) * plane];
                let target = &mut scaled[index * plane..(index + 1) * plane];
                for (out, value) in target.iter_mut().zip(source) {
                    *out = overlay::scale(*value, marker, loaded.raw_intensity);
                }
            }
            {
                let mut windows = state.windows.lock().unwrap();
                windows.insert(key, Arc::new(scaled));
                if windows.len() > MAX_WINDOWS {
                    let visible = state.visible.lock().unwrap();
                    let mut requested = state.requested.lock().unwrap();
                    windows.retain(|candidate, _| {
                        let keep = visible.contains(candidate);
                        if !keep {
                            requested.remove(candidate);
                        }
                        keep
                    });
                }
            }

            computed += 1;
            spent += started.elapsed();
            if computed.is_multiple_of(50) {
                eprintln!(
                    "[cellaris] {computed} windows, {:.0} ms each, {} queued",
                    spent.as_secs_f64() * 1000.0 / 50.0,
                    PENDING.load(Ordering::Relaxed).saturating_sub(1)
                );
                spent = Duration::ZERO;
            }
            finish(Some(host), &mut last_redraw);
        }
    });
}

fn request(key: Key) {
    let state = shared();
    if state.windows.lock().unwrap().contains_key(&key) {
        return;
    }
    if !state.requested.lock().unwrap().insert(key) {
        return;
    }
    if let Some(slot) = SENDER.get() {
        if let Some(sender) = slot.lock().unwrap().as_ref() {
            if sender.send(key).is_ok() {
                PENDING.fetch_add(1, Ordering::Relaxed);
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Blending
// ---------------------------------------------------------------------------

fn covering(cell: Key) -> impl Iterator<Item = Key> {
    [(-STRIDE, -STRIDE), (0, -STRIDE), (-STRIDE, 0), (0, 0)]
        .into_iter()
        .map(move |(dx, dy)| Key {
            file_id: cell.file_id,
            x: cell.x + dx,
            y: cell.y + dy,
        })
        .filter(|key| key.x >= 0 && key.y >= 0)
}

/// The cell's RGB, or the windows still missing for it.
fn blend_cell(state: &Shared, cell: Key, markers: &[overlay::Marker]) -> Result<Vec<u8>, Vec<Key>> {
    let (found, missing): (Vec<_>, Vec<_>) = {
        let windows = state.windows.lock().unwrap();
        covering(cell)
            .map(|key| (key, windows.get(&key).cloned()))
            .partition(|(_, planes)| planes.is_some())
    };
    if !missing.is_empty() {
        return Err(missing.into_iter().map(|(key, _)| key).collect());
    }

    let ramp = feather();
    let side = STRIDE as usize;
    let span = WINDOW as usize;
    let plane = span * span;
    let mut rgb = vec![0u8; side * side * 3];
    let mut values = vec![0f32; markers.len()];

    for ly in 0..side {
        for lx in 0..side {
            values.fill(0.0);
            let mut total = 0f32;
            for (key, planes) in &found {
                let planes = planes.as_ref().unwrap();
                let wx = (cell.x + lx as i64 - key.x) as usize;
                let wy = (cell.y + ly as i64 - key.y) as usize;
                let weight = ramp[wx] * ramp[wy];
                total += weight;
                for (index, value) in values.iter_mut().enumerate() {
                    *value += weight * f32::from(planes[index * plane + wy * span + wx]);
                }
            }
            for value in values.iter_mut() {
                *value /= total;
            }
            let pixel = overlay::composite(&values, markers);
            let at = (ly * side + lx) * 3;
            rgb[at..at + 3].copy_from_slice(&pixel);
        }
    }
    Ok(rgb)
}

// ---------------------------------------------------------------------------
// Filter
// ---------------------------------------------------------------------------

extern "C" fn apply_filter_cpu_ffi(
    filter_id: RString,
    rgba_data: *mut u8,
    len: u32,
    width: u32,
    height: u32,
    viewport: *const ViewportSnapshotFFI,
) -> bool {
    if filter_id.as_str() != FILTER_ID
        || rgba_data.is_null()
        || viewport.is_null()
        || len == 0
        || width == 0
        || height == 0
        || !ENABLED.load(Ordering::Relaxed)
    {
        return false;
    }

    // SAFETY: the host guarantees `len` writable bytes of RGBA8 at
    // `rgba_data`, and a live snapshot at `viewport`, for this call only.
    let pixels = unsafe { std::slice::from_raw_parts_mut(rgba_data, len as usize) };
    let view = unsafe { &*viewport };

    let left = view.bounds_left;
    let top = view.bounds_top;
    let span_x = view.bounds_right - left;
    let span_y = view.bounds_bottom - top;
    if span_x <= 0.0 || span_y <= 0.0 {
        return false;
    }

    let first_x = ((left / STRIDE as f64).floor() as i64 * STRIDE).max(0);
    let first_y = ((top / STRIDE as f64).floor() as i64 * STRIDE).max(0);
    let last_x = (view.bounds_right / STRIDE as f64).ceil() as i64 * STRIDE;
    let last_y = (view.bounds_bottom / STRIDE as f64).ceil() as i64 * STRIDE;
    let across = ((last_x - first_x) / STRIDE).max(0) as usize;
    let down = ((last_y - first_y) / STRIDE).max(0) as usize;

    let state = shared();
    if across.saturating_mul(down) > MAX_CELLS {
        state.visible.lock().unwrap().clear();
        draw_frame(pixels, width as usize, height as usize);
        return true;
    }
    ensure_worker();

    let mut in_view = Vec::with_capacity(across * down);
    let mut y = first_y;
    while y < last_y {
        let mut x = first_x;
        while x < last_x {
            in_view.push(Key {
                file_id: view.file_id,
                x,
                y,
            });
            x += STRIDE;
        }
        y += STRIDE;
    }

    // Publish before requesting, or the worker can drop a fresh request.
    {
        let mut visible = state.visible.lock().unwrap();
        visible.clear();
        visible.extend(in_view.iter().flat_map(|cell| covering(*cell)));
    }

    let markers = { state.markers.lock().unwrap().clone() };
    let mut painted = false;

    for cell in &in_view {
        let cached = { state.cells.lock().unwrap().get(cell).cloned() };
        let rgb = match cached {
            Some(rgb) => rgb,
            None => {
                let Some(markers) = markers.as_deref() else {
                    covering(*cell).for_each(request);
                    continue;
                };
                match blend_cell(state, *cell, markers) {
                    Ok(rgb) => {
                        let rgb = Arc::new(rgb);
                        let mut cells = state.cells.lock().unwrap();
                        cells.insert(*cell, Arc::clone(&rgb));
                        if cells.len() > MAX_CELL_CACHE {
                            let keep: HashSet<Key> = in_view.iter().copied().collect();
                            cells.retain(|candidate, _| keep.contains(candidate));
                        }
                        rgb
                    }
                    Err(missing) => {
                        missing.into_iter().for_each(request);
                        continue;
                    }
                }
            }
        };

        let to_screen_x = |slide: f64| (slide - left) / span_x * f64::from(width);
        let to_screen_y = |slide: f64| (slide - top) / span_y * f64::from(height);
        let x0 = to_screen_x(cell.x as f64).floor().max(0.0) as i64;
        let x1 = (to_screen_x((cell.x + STRIDE) as f64).ceil() as i64).min(i64::from(width));
        let y0 = to_screen_y(cell.y as f64).floor().max(0.0) as i64;
        let y1 = (to_screen_y((cell.y + STRIDE) as f64).ceil() as i64).min(i64::from(height));

        for screen_y in y0..y1 {
            let slide_y = top + (screen_y as f64 + 0.5) / f64::from(height) * span_y;
            let local_y = (slide_y - cell.y as f64).floor() as i64;
            if !(0..STRIDE).contains(&local_y) {
                continue;
            }
            for screen_x in x0..x1 {
                let slide_x = left + (screen_x as f64 + 0.5) / f64::from(width) * span_x;
                let local_x = (slide_x - cell.x as f64).floor() as i64;
                if !(0..STRIDE).contains(&local_x) {
                    continue;
                }
                let source = ((local_y * STRIDE + local_x) * 3) as usize;
                let target = ((screen_y * i64::from(width) + screen_x) * 4) as usize;
                if target + 4 <= pixels.len() && source + 3 <= rgb.len() {
                    overlay::blend_pixel(
                        &mut pixels[target..target + 4],
                        &rgb[source..source + 3],
                        ALPHA,
                    );
                    painted = true;
                }
            }
        }
    }

    painted
}

// ---------------------------------------------------------------------------
// Registration
// ---------------------------------------------------------------------------

extern "C" fn set_host_api_ffi(host_api: HostApiVTable) {
    *shared().host.lock().unwrap() = Some(host_api);
}

extern "C" fn get_toolbar_buttons_ffi() -> RVec<ToolbarButtonFFI> {
    let mut buttons = RVec::new();
    buttons.push(ToolbarButtonFFI {
        button_id: RString::from("toggle_cellaris"),
        tooltip: RString::from("Cellaris marker overlay"),
        icon_svg: RString::from(
            "<svg xmlns=\"http://www.w3.org/2000/svg\" viewBox=\"0 0 24 24\" fill=\"none\" \
             stroke=\"currentColor\" stroke-width=\"1.8\"><circle cx=\"12\" cy=\"12\" r=\"8\"/>\
             <circle cx=\"12\" cy=\"12\" r=\"3\"/></svg>",
        ),
        action_id: RString::from("toggle_cellaris"),
        tool_mode: ROption::RNone,
        hotkey: ROption::RNone,
    });
    buttons
}

extern "C" fn on_action_ffi(action_id: RString) -> ActionResponseFFI {
    if action_id.as_str() == "toggle_cellaris" {
        let now = !ENABLED.load(Ordering::Relaxed);
        ENABLED.store(now, Ordering::Relaxed);
        if now {
            ensure_worker();
        }
        announce(now);
    }
    ActionResponseFFI { open_window: false }
}

extern "C" fn get_viewport_filters_ffi() -> RVec<ViewportFilterFFI> {
    let mut filters = RVec::new();
    filters.push(ViewportFilterFFI {
        filter_id: RString::from(FILTER_ID),
        name: RString::from("Cellaris markers"),
        supports_cpu: true,
        supports_gpu: false,
        enabled: ENABLED.load(Ordering::Relaxed),
    });
    filters
}

/// Called from inside the host's filter sync, which holds the locks a toolbar
/// update needs: record the state and never call back into the host.
extern "C" fn set_filter_enabled_ffi(filter_id: RString, enabled: bool) {
    if filter_id.as_str() == FILTER_ID {
        ENABLED.store(enabled, Ordering::Relaxed);
        if enabled {
            ensure_worker();
        } else {
            shared().visible.lock().unwrap().clear();
        }
    }
}

// -- everything a pure viewport filter does not need ------------------------

extern "C" fn get_hud_toolbar_buttons_ffi() -> RVec<HudToolbarButtonFFI> {
    RVec::new()
}
extern "C" fn on_hud_action_ffi(_: RString, _: ViewportSnapshotFFI) -> ActionResponseFFI {
    ActionResponseFFI { open_window: false }
}
extern "C" fn on_ui_callback_ffi(_: RString, _: RString) {}
extern "C" fn get_sidebar_properties_ffi() -> RVec<UiPropertyFFI> {
    RVec::new()
}
extern "C" fn get_viewport_context_menu_items_ffi(
    _: ViewportSnapshotFFI,
) -> RVec<ViewportContextMenuItemFFI> {
    RVec::new()
}
extern "C" fn on_viewport_context_menu_action_ffi(
    _: RString,
    _: ViewportSnapshotFFI,
) -> ActionResponseFFI {
    ActionResponseFFI { open_window: false }
}
extern "C" fn get_viewport_overlay_points_ffi(
    _: ViewportSnapshotFFI,
) -> RVec<ViewportOverlayPointFFI> {
    RVec::new()
}
extern "C" fn get_viewport_overlay_polygons_ffi(
    _: ViewportSnapshotFFI,
) -> RVec<ViewportOverlayPolygonFFI> {
    RVec::new()
}
extern "C" fn get_viewport_overlay_component_ffi() -> ROption<ViewportOverlayComponentRequestFFI> {
    ROption::RNone
}
extern "C" fn get_viewport_overlay_properties_ffi(_: ViewportSnapshotFFI) -> RVec<UiPropertyFFI> {
    RVec::new()
}
extern "C" fn on_point_annotation_placed_ffi(_: ViewportSnapshotFFI, _: f64, _: f64) {}
extern "C" fn on_polygon_annotation_placed_ffi(
    _: ViewportSnapshotFFI,
    _: RVec<ViewportOverlayVertexFFI>,
) {
}
extern "C" fn on_undo_ffi() -> ActionResponseFFI {
    ActionResponseFFI { open_window: false }
}
extern "C" fn on_redo_ffi() -> ActionResponseFFI {
    ActionResponseFFI { open_window: false }
}
extern "C" fn on_point_annotation_moved_ffi(_: ViewportSnapshotFFI, _: RString, _: f64, _: f64) {}
extern "C" fn on_polygon_annotation_moved_ffi(
    _: ViewportSnapshotFFI,
    _: RString,
    _: RVec<ViewportOverlayVertexFFI>,
) {
}
extern "C" fn on_viewport_annotation_selected_ffi(_: ViewportSnapshotFFI, _: RString) {}
extern "C" fn apply_filter_gpu_ffi(_: RString, _: *const GpuFilterContextFFI) -> bool {
    false
}

#[cfg_attr(feature = "export-vtable-symbol", unsafe(no_mangle))]
pub extern "C" fn eov_get_plugin_vtable() -> PluginVTable {
    PluginVTable {
        set_host_api: set_host_api_ffi,
        get_toolbar_buttons: get_toolbar_buttons_ffi,
        get_hud_toolbar_buttons: get_hud_toolbar_buttons_ffi,
        on_action: on_action_ffi,
        on_hud_action: on_hud_action_ffi,
        on_ui_callback: on_ui_callback_ffi,
        get_sidebar_properties: get_sidebar_properties_ffi,
        get_viewport_context_menu_items: get_viewport_context_menu_items_ffi,
        on_viewport_context_menu_action: on_viewport_context_menu_action_ffi,
        get_viewport_overlay_points: get_viewport_overlay_points_ffi,
        get_viewport_overlay_polygons: get_viewport_overlay_polygons_ffi,
        get_viewport_overlay_component: get_viewport_overlay_component_ffi,
        get_viewport_overlay_properties: get_viewport_overlay_properties_ffi,
        on_point_annotation_placed: on_point_annotation_placed_ffi,
        on_polygon_annotation_placed: on_polygon_annotation_placed_ffi,
        on_undo: on_undo_ffi,
        on_redo: on_redo_ffi,
        on_point_annotation_moved: on_point_annotation_moved_ffi,
        on_polygon_annotation_moved: on_polygon_annotation_moved_ffi,
        on_viewport_annotation_selected: on_viewport_annotation_selected_ffi,
        get_viewport_filters: get_viewport_filters_ffi,
        apply_filter_cpu: apply_filter_cpu_ffi,
        apply_filter_gpu: apply_filter_gpu_ffi,
        set_filter_enabled: set_filter_enabled_ffi,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn neighbouring_windows_sum_to_one() {
        let ramp = feather();
        for i in 0..STRIDE as usize {
            let total = ramp[i] + ramp[i + STRIDE as usize];
            assert!(
                (total - 1.0).abs() <= 1e-3,
                "weights sum to {total} at {i}, beyond the floor"
            );
        }
    }

    #[test]
    fn a_cell_is_covered_by_the_four_windows_over_it() {
        let cell = Key {
            file_id: 0,
            x: 256,
            y: 384,
        };
        let origins: HashSet<(i64, i64)> = covering(cell).map(|k| (k.x, k.y)).collect();
        let want: HashSet<(i64, i64)> = [(128, 256), (256, 256), (128, 384), (256, 384)].into();
        assert_eq!(origins, want);
        for (x, y) in origins {
            assert!(x <= cell.x && cell.x + STRIDE <= x + WINDOW);
            assert!(y <= cell.y && cell.y + STRIDE <= y + WINDOW);
        }
        assert_eq!(
            covering(Key {
                file_id: 0,
                x: 0,
                y: 0
            })
            .count(),
            1
        );
    }

    #[test]
    fn a_blend_of_identical_windows_is_that_window() {
        let state = Shared {
            windows: Mutex::new(HashMap::new()),
            cells: Mutex::new(HashMap::new()),
            markers: Mutex::new(None),
            requested: Mutex::new(HashSet::new()),
            visible: Mutex::new(HashSet::new()),
            host: Mutex::new(None),
            failure: Mutex::new(None),
        };
        let names: Vec<String> = ["Hoechst", "Pan-CK", "SMA"].map(String::from).to_vec();
        let markers = overlay::select(&names, &overlay::DEFAULT_MARKERS).unwrap();
        let plane = (WINDOW * WINDOW) as usize;
        let flat = Arc::new(vec![128u8; markers.len() * plane]);

        let cell = Key {
            file_id: 0,
            x: 128,
            y: 128,
        };
        assert_eq!(blend_cell(&state, cell, &markers).unwrap_err().len(), 4);

        for key in covering(cell) {
            state.windows.lock().unwrap().insert(key, Arc::clone(&flat));
        }
        let rgb = blend_cell(&state, cell, &markers).unwrap();
        let expected = overlay::composite(&[128.0; 3], &markers);
        assert!(
            rgb.chunks(3).all(|px| px == expected),
            "blending a constant changed it"
        );
    }
}
