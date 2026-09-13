//! Region readout: the prediction inside a drawn polygon, beside how much of it
//! optical density alone reproduces.
//!
//! The control is cellaris's tissue-density control, measured over 16px blocks
//! (about a cell) because the viewer has no segmentation.

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Mutex;

use abi_stable::std_types::{RString, RVec};
use eov_plugin_api::ffi::{
    HostApiVTable, UiPropertyFFI, ViewportOverlayPolygonFFI, ViewportOverlayVertexFFI,
};
use serde_json::json;

use crate::{model, overlay};

const WINDOW: i64 = model::WINDOW as i64;
const BLOCK: i64 = 16;
/// Block optical density above which it counts as tissue: the gap between
/// glass (~0.01) and tissue (from ~0.06) on CRC11. The 0.15 tile cutoff from
/// `baseline.py` dropped 73% of tissue blocks, and with them the pale end of
/// the density range, which pulled r² down.
const TISSUE: f64 = 0.04;
const MIN_BLOCKS: usize = 30;
const MAX_WINDOWS: usize = 400;
const OUTLINE: [u8; 3] = [0, 200, 255];
const SIDEBAR_WIDTH: u32 = 300;
pub const BUTTON_ID: &str = "cellaris_region";
const UI_PATH: &str = "ui/readout.slint";
const COMPONENT: &str = "CellarisReadout";

pub struct Polygon {
    vertices: Vec<(f64, f64)>,
}

impl Polygon {
    pub fn new(vertices: Vec<(f64, f64)>) -> Option<Self> {
        (vertices.len() >= 3).then_some(Self { vertices })
    }

    /// Even-odd rule, so a self-crossing outline still has an inside.
    pub fn contains(&self, x: f64, y: f64) -> bool {
        let mut inside = false;
        let mut previous = self.vertices[self.vertices.len() - 1];
        for &(xi, yi) in &self.vertices {
            let (xj, yj) = previous;
            if (yi > y) != (yj > y) && x < (xj - xi) * (y - yi) / (yj - yi) + xi {
                inside = !inside;
            }
            previous = (xi, yi);
        }
        inside
    }

    fn bounds(&self) -> (f64, f64, f64, f64) {
        self.vertices.iter().fold(
            (f64::MAX, f64::MAX, f64::MIN, f64::MIN),
            |(left, top, right, bottom), &(x, y)| {
                (left.min(x), top.min(y), right.max(x), bottom.max(y))
            },
        )
    }
}

/// Running least-squares of a marker on density.
#[derive(Clone, Copy, Debug, Default)]
pub struct Fit {
    n: f64,
    sx: f64,
    sy: f64,
    sxx: f64,
    syy: f64,
    sxy: f64,
}

impl Fit {
    pub fn add(&mut self, density: f64, value: f64) {
        self.n += 1.0;
        self.sx += density;
        self.sy += value;
        self.sxx += density * density;
        self.syy += value * value;
        self.sxy += density * value;
    }

    pub fn blocks(&self) -> usize {
        self.n as usize
    }

    pub fn mean(&self) -> f64 {
        if self.n == 0.0 {
            0.0
        } else {
            self.sy / self.n
        }
    }

    /// Share of the prediction's variance a straight line in density reproduces.
    /// Squared, so a control pointing either way counts, as `control_strength`.
    pub fn r2(&self) -> Option<f64> {
        if self.blocks() < MIN_BLOCKS {
            return None;
        }
        let vx = self.sxx - self.sx * self.sx / self.n;
        let vy = self.syy - self.sy * self.sy / self.n;
        let cov = self.sxy - self.sx * self.sy / self.n;
        (vx > 1e-12 && vy > 1e-12).then(|| (cov * cov / (vx * vy)).clamp(0.0, 1.0))
    }
}

/// Blocks of a window whose centre is inside the polygon and which lie wholly
/// on the slide.
fn blocks(
    origin: (i64, i64),
    polygon: &Polygon,
    image: (f64, f64),
) -> impl Iterator<Item = (usize, usize)> + '_ {
    let per = (WINDOW / BLOCK) as usize;
    (0..per)
        .flat_map(move |by| (0..per).map(move |bx| (bx, by)))
        .filter(move |&(bx, by)| {
            let x = origin.0 + bx as i64 * BLOCK;
            let y = origin.1 + by as i64 * BLOCK;
            (x + BLOCK) as f64 <= image.0
                && (y + BLOCK) as f64 <= image.1
                && polygon.contains((x + BLOCK / 2) as f64, (y + BLOCK / 2) as f64)
        })
}

/// Window origins on a 256px grid that hold at least one block of the region.
/// Stops one past `MAX_WINDOWS`, so a huge outline is refused without a full walk.
pub fn windows(polygon: &Polygon, image: (f64, f64)) -> Vec<(i64, i64)> {
    let (left, top, right, bottom) = polygon.bounds();
    let snap = |v: f64| (v.max(0.0) / WINDOW as f64).floor() as i64 * WINDOW;
    let mut found = Vec::new();
    let mut y = snap(top);
    while (y as f64) < bottom.min(image.1) {
        let mut x = snap(left);
        while (x as f64) < right.min(image.0) {
            if blocks((x, y), polygon, image).next().is_some() {
                found.push((x, y));
                if found.len() > MAX_WINDOWS {
                    return found;
                }
            }
            x += WINDOW;
        }
        y += WINDOW;
    }
    found
}

fn optical_density() -> &'static [f64; 256] {
    static TABLE: std::sync::OnceLock<[f64; 256]> = std::sync::OnceLock::new();
    TABLE.get_or_init(|| std::array::from_fn(|c| -((c as f64 + 1.0) / 256.0).log10()))
}

/// Adds one window's tissue blocks to `fits`, one fit per marker.
#[allow(clippy::too_many_arguments)]
pub fn measure(
    rgba: &[u8],
    predicted: &[f32],
    origin: (i64, i64),
    polygon: &Polygon,
    image: (f64, f64),
    markers: &[overlay::Marker],
    raw_intensity: bool,
    fits: &mut [Fit],
) {
    let side = WINDOW as usize;
    let plane = side * side;
    let block = BLOCK as usize;
    let pixels = (block * block) as f64;
    let channels = markers.iter().map(|m| m.channel + 1).max().unwrap_or(0);
    if rgba.len() < plane * 4 || predicted.len() < channels * plane {
        return;
    }
    let od = optical_density();
    let mut values = vec![0f64; markers.len()];

    for (bx, by) in blocks(origin, polygon, image) {
        let mut density = 0.0;
        values.fill(0.0);
        for py in by * block..(by + 1) * block {
            for px in bx * block..(bx + 1) * block {
                let i = py * side + px;
                let rgb = &rgba[i * 4..i * 4 + 3];
                density += (od[rgb[0] as usize] + od[rgb[1] as usize] + od[rgb[2] as usize]) / 3.0;
                for (value, marker) in values.iter_mut().zip(markers) {
                    let raw = predicted[marker.channel * plane + i];
                    *value += f64::from(overlay::intensity(raw, raw_intensity));
                }
            }
        }
        density /= pixels;
        if density < TISSUE {
            continue;
        }
        for (fit, value) in fits.iter_mut().zip(&values) {
            fit.add(density, value / pixels);
        }
    }
}

// ---------------------------------------------------------------------------
// State and worker
// ---------------------------------------------------------------------------

struct Readout {
    file_id: i32,
    vertices: Vec<(f64, f64)>,
    fits: Vec<Fit>,
    done: usize,
    total: usize,
    mpp: Option<f64>,
    message: Option<String>,
    running: bool,
}

static READOUT: Mutex<Option<Readout>> = Mutex::new(None);
static GENERATION: AtomicU64 = AtomicU64::new(0);

/// Applies `change` if no newer region has replaced this one, then redraws the panel.
fn update(generation: u64, host: &HostApiVTable, change: impl FnOnce(&mut Readout)) -> bool {
    {
        let mut slot = READOUT.lock().unwrap();
        match slot.as_mut() {
            Some(readout) if GENERATION.load(Ordering::SeqCst) == generation => change(readout),
            _ => return false,
        }
    }
    let _ = (host.refresh_sidebar)(host.context);
    true
}

fn fail(generation: u64, host: &HostApiVTable, message: String) {
    update(generation, host, |readout| {
        readout.message = Some(message);
        readout.running = false;
    });
}

pub fn open_sidebar(host: &HostApiVTable) {
    let snapshot = (host.get_snapshot)(host.context);
    let ours = snapshot
        .active_sidebar
        .as_ref()
        .into_option()
        .is_some_and(|sidebar| sidebar.component.as_str() == COMPONENT);
    if !ours {
        let _ = (host.show_sidebar)(
            host.context,
            RString::from(BUTTON_ID),
            SIDEBAR_WIDTH,
            RString::from(UI_PATH),
            RString::from(COMPONENT),
        );
    }
}

/// Called from the host's placement callback: record, then work elsewhere.
pub fn start(host: HostApiVTable, file_id: i32, image: (f64, f64), vertices: Vec<(f64, f64)>) {
    let Some(polygon) = Polygon::new(vertices.clone()) else {
        return;
    };
    let generation = GENERATION.fetch_add(1, Ordering::SeqCst) + 1;
    *READOUT.lock().unwrap() = Some(Readout {
        file_id,
        vertices,
        fits: vec![Fit::default(); overlay::DEFAULT_MARKERS.len()],
        done: 0,
        total: 0,
        mpp: None,
        message: None,
        running: true,
    });
    std::thread::spawn(move || run(host, generation, file_id, image, polygon));
}

fn run(host: HostApiVTable, generation: u64, file_id: i32, image: (f64, f64), polygon: Polygon) {
    open_sidebar(&host);
    let _ = (host.request_render)(host.context);

    let snapshot = (host.get_snapshot)(host.context);
    let mpp = snapshot
        .open_files
        .iter()
        .find(|file| file.file_id == file_id)
        .and_then(|file| file.mpp_x.as_ref().into_option().copied());

    let windows = windows(&polygon, image);
    if windows.is_empty() {
        return fail(
            generation,
            &host,
            "No part of that region is on the slide.".into(),
        );
    }
    if windows.len() > MAX_WINDOWS {
        return fail(
            generation,
            &host,
            format!(
                "That region needs more than {MAX_WINDOWS} windows, over eight minutes on a \
                 CPU. Draw a smaller one."
            ),
        );
    }

    let model = match model::shared() {
        Ok(model) => model,
        Err(error) => {
            return fail(
                generation,
                &host,
                format!("The model did not load: {error}"),
            )
        }
    };
    let (selected, raw_intensity) = {
        let loaded = model.lock().unwrap();
        (
            overlay::select(&loaded.channel_names, &overlay::DEFAULT_MARKERS),
            loaded.raw_intensity,
        )
    };
    let markers = match selected {
        Ok(markers) => markers,
        Err(error) => return fail(generation, &host, error),
    };

    let total = windows.len();
    update(generation, &host, |readout| {
        readout.total = total;
        readout.mpp = mpp;
    });

    let mut fits = vec![Fit::default(); markers.len()];
    for (done, &origin) in windows.iter().enumerate() {
        if GENERATION.load(Ordering::SeqCst) != generation {
            return;
        }
        let read = (host.read_region)(
            host.context,
            file_id,
            0,
            origin.0,
            origin.1,
            WINDOW as u32,
            WINDOW as u32,
        );
        let rgba = match read.into_result() {
            Ok(rgba) => rgba,
            Err(error) => {
                return fail(
                    generation,
                    &host,
                    format!("The host refused a window read: {error}"),
                )
            }
        };
        let predicted = match model.lock().unwrap().predict(rgba.as_slice()) {
            Ok(values) => values,
            Err(error) => return fail(generation, &host, format!("Inference failed: {error}")),
        };
        measure(
            rgba.as_slice(),
            &predicted,
            origin,
            &polygon,
            image,
            &markers,
            raw_intensity,
            &mut fits,
        );
        let so_far = fits.clone();
        update(generation, &host, |readout| {
            readout.fits = so_far;
            readout.done = done + 1;
        });
    }
    update(generation, &host, |readout| readout.running = false);
}

pub fn clear(host: Option<HostApiVTable>) {
    GENERATION.fetch_add(1, Ordering::SeqCst);
    *READOUT.lock().unwrap() = None;
    if let Some(host) = host {
        let _ = (host.request_render)(host.context);
        let _ = (host.refresh_sidebar)(host.context);
    }
}

// ---------------------------------------------------------------------------
// What the host asks for
// ---------------------------------------------------------------------------

pub fn polygons(file_id: i32) -> RVec<ViewportOverlayPolygonFFI> {
    let slot = READOUT.lock().unwrap();
    let Some(readout) = slot.as_ref().filter(|r| r.file_id == file_id) else {
        return RVec::new();
    };
    let mut out = RVec::new();
    out.push(ViewportOverlayPolygonFFI {
        annotation_id: RString::from("cellaris-region"),
        vertices: readout
            .vertices
            .iter()
            .map(|&(x_level0, y_level0)| ViewportOverlayVertexFFI { x_level0, y_level0 })
            .collect(),
        fill_red: OUTLINE[0],
        fill_green: OUTLINE[1],
        fill_blue: OUTLINE[2],
    });
    out
}

pub fn sidebar_properties() -> RVec<UiPropertyFFI> {
    let slot = READOUT.lock().unwrap();
    let (status, detail, progress, rows) = match slot.as_ref() {
        None => (
            "Pick the region tool and draw a polygon round some tissue.".to_string(),
            String::new(),
            0.0,
            Vec::new(),
        ),
        Some(readout) => describe(readout),
    };
    let property = |name: &str, value: serde_json::Value| UiPropertyFFI {
        name: name.into(),
        json_value: value.to_string().into(),
    };
    RVec::from(vec![
        property("status", json!(status)),
        property("detail", json!(detail)),
        property("progress", json!(progress)),
        property("rows", json!(rows)),
        property("has-region", json!(slot.is_some())),
    ])
}

fn describe(readout: &Readout) -> (String, String, f64, Vec<serde_json::Value>) {
    if let Some(message) = &readout.message {
        return (message.clone(), String::new(), 0.0, Vec::new());
    }
    let blocks = readout.fits.first().map_or(0, Fit::blocks);
    let progress = if readout.total == 0 {
        0.0
    } else {
        readout.done as f64 / readout.total as f64
    };

    let status = if readout.running {
        format!("Measuring: {} of {} windows", readout.done, readout.total)
    } else {
        match readout.mpp {
            Some(mpp) => {
                let side = BLOCK as f64 * mpp / 1000.0;
                format!(
                    "{blocks} tissue blocks, {:.3} mm²",
                    blocks as f64 * side * side
                )
            }
            None => format!("{blocks} tissue blocks"),
        }
    };
    let detail = if readout.running {
        "256px windows at level 0, about 1.2 s each on a CPU.".to_string()
    } else if blocks < MIN_BLOCKS {
        format!("Fewer than {MIN_BLOCKS} tissue blocks, too few for a density r².")
    } else {
        String::new()
    };

    let rows = if blocks == 0 {
        Vec::new()
    } else {
        overlay::DEFAULT_MARKERS
            .iter()
            .zip(&readout.fits)
            .map(|((name, colour), fit)| {
                let lane = |c: f32| (c * 255.0).round() as i32;
                json!({
                    "name": name,
                    "red": lane(colour[0]),
                    "green": lane(colour[1]),
                    "blue": lane(colour[2]),
                    "mean": format!("{:.2}", fit.mean()),
                    "density": fit.r2().map_or("—".to_string(), |r2| format!("{r2:.2}")),
                })
            })
            .collect()
    };
    (status, detail, progress, rows)
}

#[cfg(test)]
mod tests {
    use super::*;

    const SLIDE: (f64, f64) = (100_000.0, 100_000.0);

    fn square(left: f64, top: f64, side: f64) -> Polygon {
        Polygon::new(vec![
            (left, top),
            (left + side, top),
            (left + side, top + side),
            (left, top + side),
        ])
        .unwrap()
    }

    fn orion() -> Vec<String> {
        [
            "Hoechst",
            "AF1",
            "CD31",
            "CD45",
            "CD68",
            "Argo550",
            "CD4",
            "FOXP3",
            "CD8a",
            "CD45RO",
            "CD20",
            "PD-L1",
            "CD3e",
            "CD163",
            "E-cadherin",
            "PD-1",
            "Ki67",
            "Pan-CK",
            "SMA",
        ]
        .map(String::from)
        .to_vec()
    }

    #[test]
    fn a_concave_outline_excludes_its_notch() {
        let l_shape = Polygon::new(vec![
            (0.0, 0.0),
            (10.0, 0.0),
            (10.0, 4.0),
            (4.0, 4.0),
            (4.0, 10.0),
            (0.0, 10.0),
        ])
        .unwrap();
        assert!(l_shape.contains(2.0, 8.0));
        assert!(l_shape.contains(8.0, 2.0));
        assert!(!l_shape.contains(8.0, 8.0), "the notch counted as inside");
        assert!(Polygon::new(vec![(0.0, 0.0), (1.0, 1.0)]).is_none());
    }

    #[test]
    fn r2_is_one_on_a_line_and_absent_when_it_cannot_be_measured() {
        let mut line = Fit::default();
        for i in 0..50 {
            line.add(i as f64, 3.0 - 2.0 * i as f64);
        }
        assert!(
            (line.r2().unwrap() - 1.0).abs() < 1e-9,
            "a falling line is still density"
        );

        let mut few = Fit::default();
        for i in 0..MIN_BLOCKS - 1 {
            few.add(i as f64, i as f64);
        }
        assert!(few.r2().is_none(), "r² reported from too few blocks");

        let mut flat = Fit::default();
        for i in 0..50 {
            flat.add(i as f64, 5.0);
        }
        assert!(flat.r2().is_none(), "a constant prediction has no r²");
        assert!((flat.mean() - 5.0).abs() < 1e-12);
    }

    #[test]
    fn windows_follow_the_region_and_stay_on_the_slide() {
        assert_eq!(windows(&square(10.0, 10.0, 100.0), SLIDE), vec![(0, 0)]);
        assert_eq!(windows(&square(200.0, 200.0, 100.0), SLIDE).len(), 4);
        assert!(windows(&square(-500.0, -500.0, 100.0), SLIDE).is_empty());

        let edge = (300.0, 300.0);
        let found = windows(&square(0.0, 0.0, 1000.0), edge);
        assert_eq!(found, vec![(0, 0), (256, 0), (0, 256), (256, 256)]);
        assert_eq!(
            blocks((256, 256), &square(0.0, 0.0, 1000.0), edge).count(),
            4,
            "blocks hanging off the slide were kept"
        );

        let huge = windows(&square(0.0, 0.0, 99_000.0), SLIDE);
        assert_eq!(
            huge.len(),
            MAX_WINDOWS + 1,
            "a huge region was walked in full"
        );
    }

    /// A window whose grey level changes by column, with a white first column,
    /// and a prediction given the block's density and row.
    fn synthetic(prediction: impl Fn(f64, usize) -> f32) -> Vec<Fit> {
        let side = WINDOW as usize;
        let block = BLOCK as usize;
        let per = side / block;
        let mut rgba = vec![255u8; side * side * 4];
        let mut predicted = vec![0f32; 19 * side * side];
        let markers = overlay::select(&orion(), &overlay::DEFAULT_MARKERS).unwrap();
        let od = optical_density();

        for by in 0..per {
            for bx in 0..per {
                let grey: u8 = if bx == 0 { 255 } else { 40 + (bx * 7) as u8 };
                for py in by * block..(by + 1) * block {
                    for px in bx * block..(bx + 1) * block {
                        let i = py * side + px;
                        rgba[i * 4..i * 4 + 3].fill(grey);
                        for marker in &markers {
                            predicted[marker.channel * side * side + i] =
                                prediction(od[grey as usize], by);
                        }
                    }
                }
            }
        }
        let mut fits = vec![Fit::default(); markers.len()];
        measure(
            &rgba,
            &predicted,
            (0, 0),
            &square(0.0, 0.0, 256.0),
            SLIDE,
            &markers,
            true,
            &mut fits,
        );
        fits
    }

    #[test]
    fn a_prediction_that_is_density_scores_one_and_one_that_is_not_scores_low() {
        let as_raw = |target: f64| (5.0 * target.sinh()) as f32;

        let density_map = synthetic(|density, _| as_raw(4.0 + 2.0 * density));
        let blocks = density_map[0].blocks();
        assert_eq!(
            blocks,
            16 * 15,
            "white blocks were not left out as background"
        );
        assert!(density_map[0].r2().unwrap() > 0.999);

        // Every row meets every grey once, so this one shares nothing with density.
        let unrelated = synthetic(|_, row| as_raw(4.0 + row as f64 / 16.0));
        assert!(
            unrelated[0].r2().unwrap() < 1e-6,
            "an unrelated prediction scored {:.3}",
            unrelated[0].r2().unwrap()
        );
    }
}
