//! The whole plugin against eov's mock host, with a fake model.
//!
//! The fake reads its configuration from `fake.json` in the test's own data
//! directory (a fresh temporary directory per test, never the real home),
//! and predicts deterministic functions of its input:
//!
//! - Hoechst follows the input's optical density (a density map, r² ≈ 1);
//! - every other channel is flat, plus an optional bias at each window's
//!   border, imitating a model that sees no context there.

use super::*;
use crate::engine::{marks_layer, pred_layer, region_layer, values_layer};
use crate::model::{Batch, LoadError, LoadRequest, ModelInfo, Prediction, Predictor};
use crate::present::{MODEL_BADGE, NOTICE_BADGE, slide_badge};
use crate::resample::optical_density;
use eov_plugin_api::testing::{MockHost, MockState, TestHarness, test_init};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex, OnceLock};
use std::time::{Duration, Instant};

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(default)]
struct FakeConfig {
    fail_load: Option<String>,
    channel_names: Vec<String>,
    mpp: f64,
    window: usize,
    recon: bool,
    /// Added to flat channels within 4 px of a window's border.
    edge_bias: f32,
    batch: Option<usize>,
}

impl Default for FakeConfig {
    fn default() -> Self {
        Self {
            fail_load: None,
            channel_names: [
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
            .to_vec(),
            mpp: 0.325,
            window: 64,
            recon: true,
            edge_bias: 0.0,
            batch: None,
        }
    }
}

/// What the fake saw, per data directory (one per test).
#[derive(Default)]
struct Seen {
    loads: AtomicUsize,
    windows: AtomicUsize,
    batches: Mutex<Vec<usize>>,
}

fn seen(data_dir: &Path) -> Arc<Seen> {
    static SEEN: OnceLock<Mutex<HashMap<PathBuf, Arc<Seen>>>> = OnceLock::new();
    let map = SEEN.get_or_init(Default::default);
    Arc::clone(
        map.lock()
            .unwrap()
            .entry(data_dir.to_path_buf())
            .or_default(),
    )
}

#[derive(Default)]
struct FakeBackend;

struct FakePredictor {
    info: ModelInfo,
    config: FakeConfig,
    seen: Arc<Seen>,
}

impl Backend for FakeBackend {
    fn load(&self, request: &LoadRequest) -> Result<Box<dyn Predictor>, LoadError> {
        let config: FakeConfig = std::fs::read_to_string(request.data_dir.join("fake.json"))
            .ok()
            .and_then(|text| serde_json::from_str(&text).ok())
            .unwrap_or_default();
        let seen = seen(&request.data_dir);
        seen.loads.fetch_add(1, Ordering::SeqCst);
        if let Some(reason) = &config.fail_load {
            return Err(LoadError::new("No marker model", reason.clone()));
        }
        Ok(Box::new(FakePredictor {
            info: ModelInfo {
                path: request.data_dir.join("markers.onnx"),
                bytes: 3,
                sha256: "fa4e".repeat(16),
                channel_names: config.channel_names.clone(),
                raw_intensity: true,
                mpp: config.mpp,
                mpp_recorded: true,
                window: config.window,
                batch: config.batch.map_or(Batch::Dynamic, Batch::Fixed),
                recon: config.recon,
                recon_threshold: None,
                title: Some("fake".into()),
                version: None,
                description: None,
                producer: None,
                training_data: None,
                license: None,
                runtime: "none".into(),
                threads: 1,
                notes: Vec::new(),
            },
            config,
            seen,
        }))
    }
}

/// arcsinh(x / 5) = target, as a raw intensity.
fn raw(target: f32) -> f32 {
    5.0 * target.sinh()
}

impl Predictor for FakePredictor {
    fn info(&self) -> &ModelInfo {
        &self.info
    }

    fn predict(&mut self, input: &[f32], n: usize) -> Result<Prediction, String> {
        let side = self.info.window;
        let area = side * side;
        let channels = self.info.channel_names.len();
        self.seen.windows.fetch_add(n, Ordering::SeqCst);
        self.seen.batches.lock().unwrap().push(n);
        let mut markers = vec![0f32; n * channels * area];
        for k in 0..n {
            let window = &input[k * 3 * area..(k + 1) * 3 * area];
            for p in 0..area {
                let (x, y) = (p % side, p / side);
                let edge = x.min(y).min(side - 1 - x).min(side - 1 - y);
                let od = (0..3)
                    .map(|c| optical_density((window[c * area + p] + 1.0) / 2.0))
                    .sum::<f32>()
                    / 3.0;
                for c in 0..channels {
                    let target = if c == 0 {
                        4.0 + 4.0 * od
                    } else if edge < 4 {
                        5.0 + self.config.edge_bias
                    } else {
                        5.0
                    };
                    markers[(k * channels + c) * area + p] = raw(target);
                }
            }
        }
        let recon = self.config.recon.then(|| vec![0.05; n * area]);
        Ok(Prediction {
            n,
            channels,
            side,
            markers,
            recon,
        })
    }
}

type Plugin = Cellaris<FakeBackend>;

/// A synthetic slide: a disc of pink tissue on glass, transparent beyond
/// the scanned area, with exact pyramid levels.
#[derive(Clone)]
struct Slide {
    width: u64,
    height: u64,
    mpp: Option<f64>,
    downsamples: Vec<f64>,
    centre: (f64, f64),
    radius: f64,
}

impl Slide {
    fn new(mpp: Option<f64>) -> Self {
        Self {
            width: 20_000,
            height: 16_000,
            mpp,
            downsamples: vec![1.0, 4.0, 16.0],
            centre: (5_000.0, 5_000.0),
            radius: 3_000.0,
        }
    }

    fn info(&self, file_id: FileId) -> FileInfo {
        FileInfo {
            file_id,
            filename: format!("slide-{file_id}.svs"),
            width: self.width,
            height: self.height,
            mpp_x: self.mpp,
            mpp_y: self.mpp,
            identity: Some(format!("identity-{file_id}")),
            levels: self
                .downsamples
                .iter()
                .enumerate()
                .map(|(level, d)| LevelInfo {
                    level: level as u32,
                    width: (self.width as f64 / d) as u64,
                    height: (self.height as f64 / d) as u64,
                    downsample: *d,
                })
                .collect(),
            ..Default::default()
        }
    }

    fn pixel(&self, x: f64, y: f64) -> [u8; 4] {
        if x < 0.0 || y < 0.0 || x >= self.width as f64 || y >= self.height as f64 {
            return [0, 0, 0, 0];
        }
        let (dx, dy) = (x - self.centre.0, y - self.centre.1);
        if dx * dx + dy * dy <= self.radius * self.radius {
            // Darker towards the centre: density varies.
            let t = ((dx * dx + dy * dy).sqrt() / self.radius) as f32;
            let shade = |base: f32| (base + 60.0 * t).min(250.0) as u8;
            [shade(170.0), shade(90.0), shade(140.0), 255]
        } else {
            [246, 246, 246, 255]
        }
    }

    fn read(&self, request: &RegionRequest) -> Image {
        let d = self.downsamples[request.level as usize];
        let mut data = Vec::with_capacity((request.width * request.height * 4) as usize);
        for row in 0..request.height {
            for col in 0..request.width {
                let x = request.x as f64 + (col as f64 + 0.5) * d;
                let y = request.y as f64 + (row as f64 + 0.5) * d;
                data.extend_from_slice(&self.pixel(x, y));
            }
        }
        Image {
            width: request.width,
            height: request.height,
            data,
        }
    }
}

fn view(
    pane: PaneId,
    file_id: FileId,
    x: f64,
    y: f64,
    width: f64,
    height: f64,
    zoom: f64,
) -> ViewInfo {
    ViewInfo {
        pane,
        file_id,
        center_x: x + width / 2.0,
        center_y: y + height / 2.0,
        zoom,
        width: width * zoom,
        height: height * zoom,
        scale_factor: 1.0,
        visible: Rect {
            x,
            y,
            width,
            height,
        },
        ..Default::default()
    }
}

struct Rig {
    harness: TestHarness<Plugin>,
    data_dir: PathBuf,
    requests: Arc<Mutex<Vec<RegionRequest>>>,
}

impl Rig {
    fn new(slides: &[(FileId, Slide)], views: &[ViewInfo], fake: FakeConfig) -> Self {
        Self::build(slides, views, fake, None, |_| {})
    }

    /// The first `fail_first` reads fail.
    fn with_reader(
        slides: &[(FileId, Slide)],
        views: &[ViewInfo],
        fake: FakeConfig,
        fail_first: Option<usize>,
    ) -> Self {
        Self::build(slides, views, fake, fail_first, |_| {})
    }

    /// With the mock host set up by `setup` before the plugin, and its
    /// worker, start (e.g. to emulate an older host).
    fn with_host(
        slides: &[(FileId, Slide)],
        views: &[ViewInfo],
        fake: FakeConfig,
        setup: impl FnOnce(&mut MockState),
    ) -> Self {
        Self::build(slides, views, fake, None, setup)
    }

    fn build(
        slides: &[(FileId, Slide)],
        views: &[ViewInfo],
        fake: FakeConfig,
        fail_first: Option<usize>,
        setup: impl FnOnce(&mut MockState),
    ) -> Self {
        let host = MockHost::new();
        let requests: Arc<Mutex<Vec<RegionRequest>>> = Arc::default();
        {
            let mut state = host.state();
            state.snapshot.files = slides.iter().map(|(id, s)| s.info(*id)).collect();
            state.snapshot.panes = views
                .iter()
                .enumerate()
                .map(|(index, v)| PaneInfo {
                    pane: v.pane,
                    index: index as u32,
                    file_id: Some(v.file_id),
                    view: Some(*v),
                })
                .collect();
            state.snapshot.focused_pane = views.first().map(|v| v.pane);
            let slides: HashMap<FileId, Slide> = slides.iter().cloned().collect();
            let log = Arc::clone(&requests);
            let failures = AtomicUsize::new(fail_first.unwrap_or(0));
            state.region_reader = Some(Box::new(move |request: &RegionRequest| {
                log.lock().unwrap().push(*request);
                if failures
                    .fetch_update(Ordering::SeqCst, Ordering::SeqCst, |n| n.checked_sub(1))
                    .is_ok()
                {
                    return Err("the slide file went away".into());
                }
                let slide = slides.get(&request.file_id).ok_or("no such slide")?;
                Ok(slide.read(request))
            }));
            setup(&mut state);
        }
        let init = test_init("cellaris");
        let data_dir = PathBuf::from(&init.data_dir);
        assert!(
            data_dir.starts_with(std::env::temp_dir()),
            "tests must stay in the temporary directory"
        );
        std::fs::write(
            data_dir.join("fake.json"),
            serde_json::to_string(&fake).unwrap(),
        )
        .unwrap();
        let harness = TestHarness::with_host(host, init).unwrap();
        for (id, slide) in slides {
            harness
                .send(HostEvent::FileOpened {
                    file: slide.info(*id),
                })
                .unwrap();
        }
        Rig {
            harness,
            data_dir,
            requests,
        }
    }

    fn click(&self, button: &str) {
        self.harness
            .send(HostEvent::ToolbarButtonClicked {
                button_id: button.into(),
            })
            .unwrap();
    }

    fn rasters(&self, file_id: FileId) -> usize {
        self.harness
            .state()
            .layers
            .get(&pred_layer(file_id))
            .map_or(0, |layer| layer.rasters.len())
    }

    fn objects(&self, layer: &str) -> Vec<OverlayObject> {
        self.harness.state().objects(layer)
    }

    fn seen(&self) -> Arc<Seen> {
        seen(&self.data_dir)
    }

    /// A badge's slide and text.
    fn badge(&self, id: &str) -> Option<(Option<FileId>, String)> {
        self.harness.state().badges.get(id).cloned()
    }

    /// The text of a slide's own badge, or "".
    fn slide_line(&self, file_id: FileId) -> String {
        self.badge(&slide_badge(file_id))
            .map(|(_, text)| text)
            .unwrap_or_default()
    }

    fn sidebar(&self) -> Map<String, Value> {
        let state = self.harness.state();
        state
            .surfaces
            .values()
            .find(|s| s.open)
            .map(|s| s.properties.clone())
            .unwrap_or_default()
    }
}

/// Every cell in view is shown: the notice and legend are up, the slide's
/// badge has no progress, and nothing is veiled as pending.
fn view_done(rig: &Rig, file_id: FileId) -> bool {
    let notice = rig
        .badge(NOTICE_BADGE)
        .is_some_and(|(file, text)| file.is_none() && text.contains("research use only"));
    let legend = rig
        .badge(MODEL_BADGE)
        .is_some_and(|(_, text)| text.contains("Hoechst"));
    let line = rig.slide_line(file_id);
    notice
        && legend
        && !line.contains("Computing")
        && !rig
            .objects(&marks_layer(file_id))
            .iter()
            .any(|o| o.object_id.starts_with("pending"))
}

fn wait_for(what: &str, timeout: Duration, mut done: impl FnMut() -> bool) {
    let deadline = Instant::now() + timeout;
    while Instant::now() < deadline {
        if done() {
            return;
        }
        std::thread::sleep(Duration::from_millis(20));
    }
    panic!("timed out waiting for {what}");
}

const LONG: Duration = Duration::from_secs(30);

#[test]
fn registers_its_buttons_and_region_tool_without_loading_anything() {
    let rig = Rig::new(&[(1, Slide::new(Some(0.325)))], &[], FakeConfig::default());
    let state = rig.harness.state();
    assert!(state.toolbar_buttons.contains_key(OVERLAY_BUTTON));
    assert!(state.toolbar_buttons.contains_key(PANEL_BUTTON));
    let tool = &state.tools[REGION_TOOL];
    assert_eq!(tool.kind, ToolKind::Polygon);
    // Three layers for the open slide, ids carrying the file id.
    for layer in [pred_layer(1), marks_layer(1), region_layer(1)] {
        assert!(state.layers.contains_key(&layer), "no layer {layer}");
    }
    assert!(!state.layers[&pred_layer(1)].spec.visible);
    assert!(
        state.layers[&pred_layer(1)].spec.z < -50,
        "above the ROI dimming"
    );
    // No CPU filter: nothing forces CPU rendering (CEL-8).
    assert!(state.filters.is_empty());
    drop(state);
    std::thread::sleep(Duration::from_millis(100));
    assert_eq!(rig.seen().loads.load(Ordering::SeqCst), 0, "loaded eagerly");
}

#[test]
fn predictions_are_matched_to_the_model_pixel_size() {
    // A 20x slide (0.5 µm/px): model pixels are 0.65 slide pixels.
    let slide = Slide::new(Some(0.5));
    let v = view(1, 1, 4_000.0, 4_000.0, 300.0, 200.0, 2.0);
    let rig = Rig::new(&[(1, slide)], &[v], FakeConfig::default());
    rig.click(OVERLAY_BUTTON);
    wait_for("rasters", LONG, || rig.rasters(1) > 0);
    let state = rig.harness.state();
    let layer = &state.layers[&pred_layer(1)];
    assert!(layer.spec.visible);
    let raster = layer.rasters.values().next().unwrap();
    // A cell is 32 model px = 32 × 0.325 µm = 20.8 slide px at 0.5 µm/px.
    assert!(
        (raster.width - 20.8).abs() < 1e-9,
        "cell of {}",
        raster.width
    );
    assert_eq!((raster.pixels.width, raster.pixels.height), (32, 32));
    drop(state);
    // Windows are read from level 0 (no level is fine enough to downsample
    // from) over 41.6 slide px plus the filter margin.
    let requests = rig.requests.lock().unwrap();
    assert!(requests.iter().all(|r| r.level == 0));
    assert!(
        requests.iter().all(|r| (44..=50).contains(&r.width)),
        "{requests:?}"
    );
}

#[test]
fn a_fine_slide_is_read_from_the_matching_pyramid_level() {
    // 0.08125 µm/px: four slide px per model px, so the ×4 level serves it.
    let slide = Slide::new(Some(0.081_25));
    let v = view(1, 1, 4_000.0, 4_000.0, 600.0, 400.0, 1.0);
    let rig = Rig::new(&[(1, slide)], &[v], FakeConfig::default());
    rig.click(OVERLAY_BUTTON);
    wait_for("rasters", LONG, || rig.rasters(1) > 0);
    let requests = rig.requests.lock().unwrap();
    assert!(requests.iter().all(|r| r.level == 1), "{requests:?}");
    assert!(requests.iter().all(|r| (64..=72).contains(&r.width)));
}

#[test]
fn both_panes_of_a_split_view_are_computed() {
    let slide = Slide::new(Some(0.325));
    let left = view(1, 1, 3_000.0, 3_000.0, 200.0, 200.0, 2.0);
    let right = view(2, 1, 7_000.0, 6_000.0, 200.0, 200.0, 2.0);
    let rig = Rig::new(&[(1, slide)], &[left, right], FakeConfig::default());
    rig.click(OVERLAY_BUTTON);
    let in_rect = |r: &Rect, area: &Rect| {
        r.x < area.x + area.width
            && r.x + r.width > area.x
            && r.y < area.y + area.height
            && r.y + r.height > area.y
    };
    wait_for("both panes", LONG, || {
        let state = rig.harness.state();
        let Some(layer) = state.layers.get(&pred_layer(1)) else {
            return false;
        };
        let rects: Vec<Rect> = layer
            .rasters
            .values()
            .map(|r| Rect {
                x: r.x,
                y: r.y,
                width: r.width,
                height: r.height,
            })
            .collect();
        rects.iter().any(|r| in_rect(r, &left.visible))
            && rects.iter().any(|r| in_rect(r, &right.visible))
    });
    // The notice is fixed to every pane, not drawn on the slide (where it
    // would show in one pane's corner and drift during a pan).
    wait_for("the notice", LONG, || {
        rig.badge(NOTICE_BADGE)
            .is_some_and(|(file, text)| file.is_none() && text.contains("PREDICTED"))
    });
    assert!(
        rig.objects(&marks_layer(1))
            .iter()
            .all(|o| o.label.is_none()),
        "text drawn on the slide"
    );
}

#[test]
fn two_slides_side_by_side_each_carry_a_badge() {
    let left = view(1, 1, 4_000.0, 4_000.0, 100.0, 100.0, 2.0);
    let right = view(2, 2, 4_000.0, 4_000.0, 100.0, 100.0, 2.0);
    let rig = Rig::new(
        &[(1, Slide::new(Some(0.325))), (2, Slide::new(Some(0.5)))],
        &[left, right],
        FakeConfig::default(),
    );
    rig.click(OVERLAY_BUTTON);
    wait_for("both slides", LONG, || {
        rig.rasters(1) > 0 && rig.rasters(2) > 0
    });
    // Each slide's warnings show over its own pane only: the 20x one is
    // upsampled, the 40x one is not.
    wait_for("the upsampling note", LONG, || {
        rig.badge(&slide_badge(2))
            .is_some_and(|(file, text)| file == Some(2) && text.contains("upsampled"))
    });
    wait_for("both views", LONG, || view_done(&rig, 1));
    assert!(!rig.slide_line(1).contains("upsampled"));
}

#[test]
fn nothing_off_the_slide_is_read_and_glass_is_not_predicted() {
    // The view hangs over the slide's bottom-right corner, on glass.
    let slide = Slide::new(Some(0.325));
    let v = view(1, 1, 19_850.0, 15_850.0, 400.0, 400.0, 2.0);
    let rig = Rig::new(&[(1, slide)], &[v], FakeConfig::default());
    rig.click(OVERLAY_BUTTON);
    wait_for("the view to be read", LONG, || {
        rig.harness
            .state()
            .objects(&marks_layer(1))
            .iter()
            .all(|o| !o.object_id.starts_with("pending"))
            && !rig.requests.lock().unwrap().is_empty()
    });
    let requests = rig.requests.lock().unwrap().clone();
    for request in &requests {
        // Every window starts on the slide (only its far margin may hang off).
        assert!(request.x < 20_000 && request.y < 16_000, "{request:?}");
    }
    assert_eq!(
        rig.seen().windows.load(Ordering::SeqCst),
        0,
        "glass was predicted"
    );
    assert_eq!(rig.rasters(1), 0, "glass was painted");
}

#[test]
fn zoomed_out_views_say_so_instead_of_predicting() {
    let slide = Slide::new(Some(0.325));
    let v = view(1, 1, 0.0, 0.0, 20_000.0, 16_000.0, 0.05);
    let rig = Rig::new(&[(1, slide)], &[v], FakeConfig::default());
    rig.click(OVERLAY_BUTTON);
    wait_for("the zoom notice", LONG, || {
        rig.slide_line(1)
            .starts_with("Zoom in to predict: this view needs")
    });
    assert_eq!(rig.seen().windows.load(Ordering::SeqCst), 0);
}

#[test]
fn the_badges_name_the_predictions_and_stay_put_during_a_pan() {
    let slide = Slide::new(Some(0.325));
    let v = view(1, 1, 4_000.0, 4_000.0, 300.0, 300.0, 2.0);
    let rig = Rig::new(&[(1, slide)], &[v], FakeConfig::default());
    rig.click(OVERLAY_BUTTON);
    wait_for("the whole view", LONG, || view_done(&rig, 1));
    let (_, notice) = rig.badge(NOTICE_BADGE).unwrap();
    assert!(
        notice.contains("PREDICTED") && notice.contains("research use only"),
        "{notice}"
    );
    assert_eq!(
        rig.badge(MODEL_BADGE),
        Some((
            None,
            "Hoechst: yellow \u{b7} Pan-CK: magenta \u{b7} SMA: bluish green".to_string()
        ))
    );

    // Pan within what is computed: nothing is sent again for the badges
    // (they are fixed to the pane), and nothing moves on the slide.
    let sent = rig.harness.calls_to("badge.set").len();
    let marks = rig.objects(&marks_layer(1));
    let moved = view(1, 1, 4_050.0, 4_050.0, 200.0, 200.0, 3.0);
    rig.harness.state().snapshot.panes[0].view = Some(moved);
    rig.harness
        .send(HostEvent::ViewChanged { view: moved })
        .unwrap();
    wait_for("the panned view", LONG, || view_done(&rig, 1));
    assert_eq!(rig.harness.calls_to("badge.set").len(), sent);
    assert_eq!(rig.objects(&marks_layer(1)), marks);
    assert!(marks.iter().all(|o| o.label.is_none()));

    // Hiding a marker drops it from the legend.
    rig.harness
        .ui_callback("panel", "set-marker-visible", vec![json!(1), json!(false)])
        .unwrap();
    assert_eq!(
        rig.badge(MODEL_BADGE).unwrap().1,
        "Hoechst: yellow \u{b7} SMA: bluish green (1 hidden)"
    );
    assert_eq!(
        rig.harness.state().settings.get("hidden_markers"),
        Some(&json!(["Pan-CK"]))
    );
    // With the overlay off, every badge goes.
    rig.click(OVERLAY_BUTTON);
    assert!(rig.harness.state().badges.is_empty());
}

#[test]
fn an_older_host_without_passive_tooltips_gets_no_values_layer() {
    // Released 1.0 hosts have neither `layer.set_min_zoom` nor tooltips
    // on non-interactive layers.
    let slide = Slide::new(Some(0.325));
    let v = view(1, 1, 4_000.0, 4_000.0, 200.0, 200.0, 2.0);
    let rig = Rig::with_host(
        &[(1, slide.clone()), (2, slide.clone())],
        &[v],
        FakeConfig::default(),
        |state| state.unsupported = vec!["layer.set_min_zoom".into()],
    );
    rig.click(OVERLAY_BUTTON);
    wait_for("the whole view", LONG, || view_done(&rig, 1));
    assert!(rig.rasters(1) > 0);
    // Asked once; the layer it was asked for is gone, and no other slide
    // got one.
    assert_eq!(rig.harness.calls_to("layer.set_min_zoom").len(), 1);
    let state = rig.harness.state();
    assert!(!state.layers.contains_key(&values_layer(1)));
    assert!(!state.layers.contains_key(&values_layer(2)));
    assert!(state.layers.contains_key(&pred_layer(2)));
    let values_calls = state
        .calls
        .iter()
        .filter(|call| call.args["layer_id"] == json!(values_layer(1)))
        .count();
    // Created, asked, removed: nothing more.
    assert_eq!(values_calls, 2, "the values layer was still updated");
}

#[test]
fn an_older_host_without_badges_gets_the_notice_on_the_slide() {
    let slide = Slide::new(Some(0.325));
    let v = view(1, 1, 4_000.0, 4_000.0, 200.0, 200.0, 2.0);
    let rig = Rig::with_host(&[(1, slide)], &[v], FakeConfig::default(), |state| {
        state.unsupported = vec!["badge.set".into()];
    });
    rig.click(OVERLAY_BUTTON);
    let badge = |rig: &Rig| {
        rig.objects(&marks_layer(1))
            .into_iter()
            .find(|o| o.object_id == "badge-1")
    };
    wait_for("the badge on the slide", LONG, || {
        badge(&rig).is_some()
            && rig
                .objects(&marks_layer(1))
                .iter()
                .any(|o| o.object_id.starts_with("legend-1-"))
    });
    let label = badge(&rig).unwrap().label.unwrap();
    assert!(
        label.contains("PREDICTED") && label.contains("research use only"),
        "{label}"
    );
    let legend: Vec<String> = rig
        .objects(&marks_layer(1))
        .into_iter()
        .filter(|o| o.object_id.starts_with("legend-1-"))
        .filter_map(|o| o.label)
        .collect();
    assert_eq!(legend.len(), 3, "{legend:?}");
    assert!(legend[0].contains("Hoechst") && legend[0].contains("predicted"));
    // Asked once, then no more.
    std::thread::sleep(Duration::from_millis(300));
    assert_eq!(rig.harness.calls_to("badge.set").len(), 1);

    // There it follows the view, once the view settles.
    let moved = view(1, 1, 6_000.0, 6_000.0, 200.0, 200.0, 2.0);
    rig.harness.state().snapshot.panes[0].view = Some(moved);
    rig.harness
        .send(HostEvent::ViewChanged { view: moved })
        .unwrap();
    let Shape::Point { x, y } = badge(&rig).unwrap().shape else {
        panic!("the badge is not a point");
    };
    assert!((6_000.0..6_050.0).contains(&x) && (6_000.0..6_050.0).contains(&y));
}

#[test]
fn hovering_a_cell_shows_its_predicted_values() {
    let slide = Slide::new(Some(0.325));
    let v = view(1, 1, 4_000.0, 4_000.0, 200.0, 200.0, 2.0);
    let rig = Rig::new(&[(1, slide)], &[v], FakeConfig::default());
    rig.click(OVERLAY_BUTTON);
    wait_for("the whole view", LONG, || view_done(&rig, 1));
    let state = rig.harness.state();
    let layer = &state.layers[&values_layer(1)];
    // Not interactive: it takes no clicks or hovers from other tools.
    assert!(!layer.spec.interactive && !layer.spec.editable);
    assert!(layer.spec.visible);
    assert!(
        layer.spec.z > state.layers[&pred_layer(1)].spec.z && layer.spec.z < -50,
        "over the predictions, under the ROI dimming"
    );
    // Offered once a cell (32 slide px with the fake's 64 px windows) is 24
    // screen px wide.
    let min_zoom = layer.spec.min_zoom.unwrap();
    assert!((min_zoom - 24.0 / 32.0).abs() < 1e-9, "{min_zoom}");
    // A tooltip over each painted cell.
    let rasters: Vec<&String> = state.layers[&pred_layer(1)].rasters.keys().collect();
    let values: Vec<&String> = layer.objects.keys().collect();
    assert!(!values.is_empty());
    assert!(values.iter().all(|id| rasters.contains(id)), "{values:?}");
    let object = layer.objects.values().next().unwrap();
    let tooltip = object.tooltip.as_deref().unwrap();
    assert!(
        tooltip.starts_with("Predicted \u{2014} research use only"),
        "{tooltip}"
    );
    assert!(
        tooltip.contains("Hoechst ") && tooltip.contains("SMA 5.00"),
        "{tooltip}"
    );
    assert!(tooltip.contains("tissue blocks"), "{tooltip}");
    drop(state);
    // They go with the overlay.
    rig.click(OVERLAY_BUTTON);
    assert!(!rig.harness.state().layers[&values_layer(1)].spec.visible);
}

#[test]
fn slides_without_a_pixel_size_are_refused_unless_the_user_assumes_one() {
    let slide = Slide::new(None);
    let v = view(1, 1, 4_000.0, 4_000.0, 200.0, 200.0, 2.0);
    let rig = Rig::new(&[(1, slide)], &[v], FakeConfig::default());
    rig.click(OVERLAY_BUTTON);
    wait_for("the refusal", LONG, || {
        rig.slide_line(1)
            .starts_with("Not computed: pixel size unknown")
    });
    assert_eq!(rig.rasters(1), 0);
    rig.harness
        .ui_callback("panel", "set-assume-mpp", vec![json!(true)])
        .unwrap();
    wait_for("assumed-scale rasters", LONG, || rig.rasters(1) > 0);
    wait_for("the assumption to be shown", LONG, || {
        rig.slide_line(1)
            .contains("pixel size unknown \u{2014} assumed 0.325")
    });
    assert_eq!(
        rig.harness.state().settings.get("assume_mpp"),
        Some(&json!(true))
    );
}

#[test]
fn a_model_that_fails_to_load_is_reported_and_loads_on_retry() {
    let slide = Slide::new(Some(0.325));
    let v = view(1, 1, 4_000.0, 4_000.0, 200.0, 200.0, 2.0);
    let fake = FakeConfig {
        fail_load: Some("no markers.onnx in the data directory".into()),
        ..Default::default()
    };
    let rig = Rig::new(&[(1, slide)], &[v], fake);
    rig.click(PANEL_BUTTON);
    rig.click(OVERLAY_BUTTON);
    wait_for("the error", LONG, || {
        rig.harness
            .state()
            .notifications
            .iter()
            .any(|n| n.level == NotifyLevel::Error && n.message.contains("No marker model"))
    });
    wait_for("the panel to show it", LONG, || {
        let panel = rig.sidebar();
        panel.get("status-level") == Some(&json!(3)) && panel.get("can-retry") == Some(&json!(true))
    });
    let panel = rig.sidebar();
    assert!(
        panel["status-detail"]
            .as_str()
            .unwrap()
            .contains("no markers.onnx")
    );
    // Not retried in a loop.
    std::thread::sleep(Duration::from_millis(300));
    assert_eq!(rig.seen().loads.load(Ordering::SeqCst), 1);

    // The user fixes it and retries: no restart needed (CEL-6).
    std::fs::write(
        rig.data_dir.join("fake.json"),
        serde_json::to_string(&FakeConfig::default()).unwrap(),
    )
    .unwrap();
    rig.harness.ui_callback("panel", "retry", vec![]).unwrap();
    wait_for("rasters after the retry", LONG, || rig.rasters(1) > 0);
    assert_eq!(rig.seen().loads.load(Ordering::SeqCst), 2);
}

#[test]
fn failed_reads_are_shown_and_retried() {
    let slide = Slide::new(Some(0.325));
    let v = view(1, 1, 4_000.0, 4_000.0, 100.0, 100.0, 2.0);
    let rig = Rig::with_reader(&[(1, slide)], &[v], FakeConfig::default(), Some(6));
    rig.click(OVERLAY_BUTTON);
    wait_for("the failure outline", LONG, || {
        rig.objects(&marks_layer(1))
            .iter()
            .any(|o| o.object_id.starts_with("failed-"))
    });
    assert!(
        rig.harness
            .state()
            .notifications
            .iter()
            .any(|n| n.message.contains("could not compute")
                && n.detail.as_deref().is_some_and(|d| d.contains("went away")))
    );
    // After the back-off the same windows are read again and succeed.
    wait_for("recovery", LONG, || {
        rig.rasters(1) > 0
            && !rig
                .objects(&marks_layer(1))
                .iter()
                .any(|o| o.object_id.starts_with("failed-"))
    });
}

#[test]
fn inference_runs_in_batches() {
    let slide = Slide::new(Some(0.325));
    let v = view(1, 1, 4_500.0, 4_500.0, 300.0, 300.0, 2.0);
    let rig = Rig::new(&[(1, slide)], &[v], FakeConfig::default());
    rig.click(OVERLAY_BUTTON);
    wait_for("rasters", LONG, || rig.rasters(1) > 20);
    let batches = rig.seen().batches.lock().unwrap().clone();
    assert!(batches.iter().all(|n| (1..=4).contains(n)), "{batches:?}");
    assert!(batches.iter().any(|n| *n > 1), "{batches:?}");
}

fn readout_rows(rig: &Rig) -> Vec<Value> {
    rig.sidebar()
        .get("readout-rows")
        .and_then(Value::as_array)
        .cloned()
        .unwrap_or_default()
}

#[test]
fn a_region_is_measured_on_the_blended_cells_without_seams() {
    // Every window reads +3 within 8 px of its border on the flat channels.
    let slide = Slide::new(Some(0.325));
    let v = view(1, 1, 4_000.0, 4_000.0, 400.0, 400.0, 2.0);
    let fake = FakeConfig {
        edge_bias: 3.0,
        ..Default::default()
    };
    let rig = Rig::new(&[(1, slide)], &[v], fake);
    rig.click(PANEL_BUTTON);
    rig.harness
        .send(HostEvent::ShapeDrawn {
            tool_id: REGION_TOOL.into(),
            pane: 1,
            file_id: 1,
            shape: Shape::Polygon {
                exterior: vec![
                    [4_100.0, 4_100.0],
                    [4_400.0, 4_100.0],
                    [4_400.0, 4_400.0],
                    [4_100.0, 4_400.0],
                ],
                holes: vec![],
            },
            modifiers: Modifiers::default(),
        })
        .unwrap();
    wait_for("the readout", LONG, || {
        rig.sidebar().get("can-copy") == Some(&json!(true))
    });
    let rows = readout_rows(&rig);
    assert_eq!(rows.len(), 3, "{rows:?}");
    let value = |row: &Value, key: &str| {
        row[key]
            .as_str()
            .unwrap()
            .parse::<f64>()
            .unwrap_or(f64::NAN)
    };
    // Hoechst is a density map by construction.
    assert_eq!(rows[0]["name"], "Hoechst");
    assert!(value(&rows[0], "density") > 0.95, "{rows:?}");
    // Pan-CK is flat apart from the window-edge bias. Unblended 64-px
    // windows would put +3 on every block at a window border (half of
    // them); feathered, almost nothing is left of it.
    assert_eq!(rows[1]["name"], "Pan-CK");
    let mean = value(&rows[1], "mean");
    let spread = value(&rows[1], "spread");
    assert!(
        (mean - 5.0).abs() < 0.05,
        "the seams leaked into the mean: {mean}"
    );
    assert!(spread < 0.05, "the seams leaked into the spread: {spread}");

    // The job ran and finished; the region is drawn and editable.
    assert_eq!(rig.harness.calls_to("job.start").len(), 1);
    let region = rig.objects(&region_layer(1));
    assert_eq!(region.len(), 1);
    assert_eq!(region[0].editable, Some(true));

    // Copy carries provenance.
    rig.harness
        .ui_callback("panel", "copy-readout", vec![])
        .unwrap();
    let copied = rig.harness.state().clipboard.clone().unwrap();
    assert!(copied.contains("research use only") && copied.contains("slide-1.svs"));
}

#[test]
fn editing_the_region_measures_it_again() {
    let slide = Slide::new(Some(0.325));
    let v = view(1, 1, 4_000.0, 4_000.0, 400.0, 400.0, 2.0);
    let rig = Rig::new(&[(1, slide)], &[v], FakeConfig::default());
    rig.click(PANEL_BUTTON);
    let square = |side: f64| Shape::Polygon {
        exterior: vec![
            [4_100.0, 4_100.0],
            [4_100.0 + side, 4_100.0],
            [4_100.0 + side, 4_100.0 + side],
            [4_100.0, 4_100.0 + side],
        ],
        holes: vec![],
    };
    rig.harness
        .send(HostEvent::ShapeDrawn {
            tool_id: REGION_TOOL.into(),
            pane: 1,
            file_id: 1,
            shape: square(100.0),
            modifiers: Modifiers::default(),
        })
        .unwrap();
    wait_for("the first readout", LONG, || {
        rig.sidebar().get("can-copy") == Some(&json!(true))
    });
    let first = rig.sidebar()["readout-status"].clone();
    rig.harness
        .send(HostEvent::ObjectEdited {
            object: ObjectRef {
                layer_id: region_layer(1),
                object_id: "region".into(),
            },
            shape: square(250.0),
        })
        .unwrap();
    wait_for("the second readout", LONG, || {
        let panel = rig.sidebar();
        panel.get("can-copy") == Some(&json!(true)) && panel["readout-status"] != first
    });
    assert_eq!(rig.harness.calls_to("job.start").len(), 2);
}

#[test]
fn the_roi_can_be_measured_and_cleared() {
    let slide = Slide::new(Some(0.325));
    let v = view(1, 1, 4_000.0, 4_000.0, 400.0, 400.0, 2.0);
    let rig = Rig::new(&[(1, slide)], &[v], FakeConfig::default());
    rig.click(PANEL_BUTTON);
    // No ROI yet: the user is told, nothing starts.
    rig.harness
        .ui_callback("panel", "measure-roi", vec![])
        .unwrap();
    assert!(rig.harness.calls_to("job.start").is_empty());
    rig.harness.state().roi.insert(
        1,
        Rect {
            x: 4_100.0,
            y: 4_100.0,
            width: 150.0,
            height: 150.0,
        },
    );
    rig.harness
        .ui_callback("panel", "measure-roi", vec![])
        .unwrap();
    wait_for("the ROI readout", LONG, || {
        rig.sidebar().get("can-copy") == Some(&json!(true))
    });
    rig.harness
        .ui_callback("panel", "clear-region", vec![])
        .unwrap();
    assert!(rig.objects(&region_layer(1)).is_empty());
    assert_eq!(rig.sidebar()["has-region"], json!(false));

    // Selecting the region and pressing Delete clears it too.
    rig.harness
        .ui_callback("panel", "measure-roi", vec![])
        .unwrap();
    assert_eq!(rig.objects(&region_layer(1)).len(), 1);
    rig.harness
        .send(HostEvent::ObjectsDeleteRequested {
            objects: vec![ObjectRef {
                layer_id: region_layer(1),
                object_id: "region".into(),
            }],
        })
        .unwrap();
    assert!(rig.objects(&region_layer(1)).is_empty());
}

#[test]
fn a_readout_of_the_roi_follows_the_roi() {
    let slide = Slide::new(Some(0.325));
    let v = view(1, 1, 4_000.0, 4_000.0, 400.0, 400.0, 2.0);
    let rig = Rig::new(&[(1, slide)], &[v], FakeConfig::default());
    rig.click(PANEL_BUTTON);
    let square = |x: f64, side: f64| Rect {
        x,
        y: 4_100.0,
        width: side,
        height: side,
    };
    let shape = |r: Rect| Shape::Rectangle {
        x: r.x,
        y: r.y,
        width: r.width,
        height: r.height,
    };
    rig.harness.state().roi.insert(1, square(4_100.0, 100.0));
    rig.harness
        .ui_callback("panel", "measure-roi", vec![])
        .unwrap();
    wait_for("the ROI readout", LONG, || {
        rig.sidebar().get("can-copy") == Some(&json!(true))
    });
    let first = rig.sidebar()["readout-status"].clone();

    // The ROI is moved and enlarged: measured again, where it is now.
    let moved = square(4_200.0, 200.0);
    rig.harness.state().roi.insert(1, moved);
    rig.harness
        .send(HostEvent::RoiChanged {
            file_id: 1,
            roi: Some(moved),
        })
        .unwrap();
    assert_eq!(rig.objects(&region_layer(1))[0].shape, shape(moved));
    wait_for("the second readout", LONG, || {
        let panel = rig.sidebar();
        panel.get("can-copy") == Some(&json!(true)) && panel["readout-status"] != first
    });
    assert_eq!(rig.harness.calls_to("job.start").len(), 2);
    // The same ROI again starts nothing.
    rig.harness
        .send(HostEvent::RoiChanged {
            file_id: 1,
            roi: Some(moved),
        })
        .unwrap();
    assert_eq!(rig.harness.calls_to("job.start").len(), 2);
    // Removed, and the region goes with it.
    rig.harness
        .send(HostEvent::RoiChanged {
            file_id: 1,
            roi: None,
        })
        .unwrap();
    assert!(rig.objects(&region_layer(1)).is_empty());
    assert_eq!(rig.sidebar()["has-region"], json!(false));

    // A region drawn with the tool stays where it was drawn.
    let drawn = square(4_100.0, 120.0);
    rig.harness
        .send(HostEvent::ShapeDrawn {
            tool_id: REGION_TOOL.into(),
            pane: 1,
            file_id: 1,
            shape: shape(drawn),
            modifiers: Modifiers::default(),
        })
        .unwrap();
    rig.harness
        .send(HostEvent::RoiChanged {
            file_id: 1,
            roi: Some(square(4_300.0, 50.0)),
        })
        .unwrap();
    assert_eq!(rig.objects(&region_layer(1))[0].shape, shape(drawn));
}

#[test]
fn an_adjusted_display_is_noted_in_the_panel() {
    let slide = Slide::new(Some(0.325));
    let v = view(1, 1, 4_000.0, 4_000.0, 200.0, 200.0, 2.0);
    // Set before the worker first asks.
    let rig = Rig::with_host(&[(1, slide)], &[v], FakeConfig::default(), |state| {
        state.display = DisplayAdjustments {
            gamma: 1.4,
            modified: true,
            badge: "Gamma 1.4".into(),
            ..Default::default()
        };
    });
    rig.click(PANEL_BUTTON);
    let notes = |rig: &Rig| {
        rig.sidebar()["slide-notes"]
            .as_str()
            .unwrap_or_default()
            .to_string()
    };
    wait_for("the note", LONG, || {
        notes(&rig).contains("the display is adjusted (Gamma 1.4); predictions use the slide's")
    });
    // Each pane is asked once; changes come as events.
    let asked = rig.harness.calls_to("view.display");
    assert_eq!(asked.len(), 1);
    assert_eq!(asked[0]["pane"], json!(1));
    rig.harness
        .send(HostEvent::DisplayChanged {
            pane: 1,
            display: DisplayAdjustments::default(),
        })
        .unwrap();
    assert!(!notes(&rig).contains("adjusted"), "{}", notes(&rig));
    rig.click(OVERLAY_BUTTON);
    wait_for("the whole view", LONG, || view_done(&rig, 1));
    assert_eq!(rig.harness.calls_to("view.display").len(), 1);
}

#[test]
fn an_older_host_without_display_info_is_asked_once() {
    let slide = Slide::new(Some(0.325));
    let v = view(1, 1, 4_000.0, 4_000.0, 200.0, 200.0, 2.0);
    let rig = Rig::with_host(&[(1, slide)], &[v], FakeConfig::default(), |state| {
        state.unsupported = vec!["view.display".into()];
    });
    wait_for("the question", LONG, || {
        !rig.harness.calls_to("view.display").is_empty()
    });
    // A second pane appears: it is not asked about either.
    let other = view(2, 1, 6_000.0, 6_000.0, 200.0, 200.0, 2.0);
    rig.harness.state().snapshot.panes.push(PaneInfo {
        pane: 2,
        index: 1,
        file_id: Some(1),
        view: Some(other),
    });
    rig.click(OVERLAY_BUTTON);
    // Both views computed (the slide's badge counts both panes).
    wait_for("both views", LONG, || view_done(&rig, 1));
    assert_eq!(rig.harness.calls_to("view.display").len(), 1);
}

#[test]
fn the_overlay_has_a_key_and_a_refused_key_is_shown_in_the_panel() {
    let rig = Rig::new(&[(1, Slide::new(Some(0.325)))], &[], FakeConfig::default());
    assert_eq!(
        rig.harness.state().toolbar_buttons[OVERLAY_BUTTON].hotkey,
        Some(Hotkey::key("x"))
    );
    let reason = "Hotkey 'x' of cellaris:cellaris_overlay was ignored: it is the same key as \
                  other:tool.";
    rig.harness
        .send(HostEvent::RegistrationRefused {
            what: "hotkey".into(),
            id: OVERLAY_BUTTON.into(),
            reason: reason.into(),
        })
        .unwrap();
    rig.click(PANEL_BUTTON);
    let detail = rig.sidebar()["status-detail"]
        .as_str()
        .unwrap_or_default()
        .to_string();
    assert!(detail.contains(reason), "{detail}");
    assert!(detail.contains("set as overlay_key"), "{detail}");
}

#[test]
fn the_overlay_key_can_be_changed_or_turned_off_in_the_settings() {
    for (setting, hotkey) in [(json!("F7"), Some(Hotkey::key("F7"))), (json!(""), None)] {
        let mut init = test_init("cellaris");
        init.settings.insert("overlay_key".into(), setting);
        let harness = TestHarness::<Plugin>::with_host(MockHost::new(), init).unwrap();
        assert_eq!(
            harness.state().toolbar_buttons[OVERLAY_BUTTON].hotkey,
            hotkey
        );
    }
}

#[test]
fn a_long_readout_asks_first_and_can_be_declined() {
    let slide = Slide {
        radius: 9_000.0,
        ..Slide::new(Some(0.325))
    };
    let rig = Rig::new(&[(1, slide)], &[], FakeConfig::default());
    rig.click(PANEL_BUTTON);
    // ~6,000 × 6,000 px: far more than the confirmation threshold.
    rig.harness
        .send(HostEvent::ShapeDrawn {
            tool_id: REGION_TOOL.into(),
            pane: 1,
            file_id: 1,
            shape: Shape::Rectangle {
                x: 2_000.0,
                y: 2_000.0,
                width: 1_600.0,
                height: 1_600.0,
            },
            modifiers: Modifiers::default(),
        })
        .unwrap();
    wait_for("the question", LONG, || {
        !rig.harness.calls_to("dialog.confirm").is_empty()
    });
    assert!(rig.harness.calls_to("job.start").is_empty());
    assert_eq!(rig.harness.calls_to("dialog.confirm").len(), 1);
    // Decline. (The mock numbers its ids "request-<n>" from one counter.)
    for n in 1..20 {
        rig.harness
            .send(HostEvent::ConfirmResult {
                request_id: format!("request-{n}"),
                confirmed: false,
            })
            .unwrap();
    }
    std::thread::sleep(Duration::from_millis(300));
    assert!(rig.harness.calls_to("job.start").is_empty());
    assert!(
        rig.sidebar()["readout-status"]
            .as_str()
            .unwrap()
            .starts_with("Cancelled")
    );
}

#[test]
fn display_changes_redraw_without_predicting_again() {
    let slide = Slide::new(Some(0.325));
    let v = view(1, 1, 4_000.0, 4_000.0, 200.0, 200.0, 2.0);
    let rig = Rig::new(&[(1, slide)], &[v], FakeConfig::default());
    rig.click(OVERLAY_BUTTON);
    wait_for("the whole view", LONG, || view_done(&rig, 1));
    let predicted = rig.seen().windows.load(Ordering::SeqCst);
    let snapshot = |rig: &Rig| -> Vec<Vec<u8>> {
        rig.harness.state().layers[&pred_layer(1)]
            .rasters
            .values()
            .map(|r| r.pixels.data.to_vec())
            .collect()
    };
    let old = snapshot(&rig);
    rig.harness
        .ui_callback("panel", "set-fluorescence", vec![json!(true)])
        .unwrap();
    wait_for("redrawn rasters", LONG, || snapshot(&rig) != old);
    assert_eq!(rig.seen().windows.load(Ordering::SeqCst), predicted);

    // Opacity is the layer's, set at once.
    rig.harness
        .ui_callback("panel", "set-opacity", vec![json!(0.4)])
        .unwrap();
    let opacity = rig.harness.state().layers[&pred_layer(1)].spec.opacity;
    assert!((opacity - 0.4).abs() < 1e-6);
    assert_eq!(
        rig.harness.state().settings.get("opacity"),
        Some(&json!(0.4f32))
    );
}

#[test]
fn predictions_come_back_from_the_disk_cache() {
    let slide = Slide::new(Some(0.325));
    let v = view(1, 1, 4_000.0, 4_000.0, 150.0, 150.0, 2.0);
    let rig = Rig::new(&[(1, slide.clone())], &[v], FakeConfig::default());
    rig.click(OVERLAY_BUTTON);
    wait_for("the whole view", LONG, || view_done(&rig, 1));
    let first = rig.rasters(1);

    // The same slide in a second session, sharing the cache directory.
    let mut init = test_init("cellaris");
    let first_root = rig.data_dir.parent().unwrap().to_path_buf();
    init.cache_dir = first_root.join("cache").to_string_lossy().into_owned();
    std::fs::write(
        PathBuf::from(&init.data_dir).join("fake.json"),
        serde_json::to_string(&FakeConfig::default()).unwrap(),
    )
    .unwrap();
    let host = MockHost::new();
    {
        let mut state = host.state();
        state.snapshot.files = vec![slide.info(1)];
        state.snapshot.panes = vec![PaneInfo {
            pane: 1,
            index: 0,
            file_id: Some(1),
            view: Some(v),
        }];
        let reader_slide = slide.clone();
        state.region_reader = Some(Box::new(move |r: &RegionRequest| Ok(reader_slide.read(r))));
    }
    let second = TestHarness::<Plugin>::with_host(host, init.clone()).unwrap();
    second
        .send(HostEvent::FileOpened {
            file: slide.info(1),
        })
        .unwrap();
    second
        .send(HostEvent::ToolbarButtonClicked {
            button_id: OVERLAY_BUTTON.into(),
        })
        .unwrap();
    wait_for("cached rasters", LONG, || {
        second
            .state()
            .layers
            .get(&pred_layer(1))
            .is_some_and(|l| l.rasters.len() >= first)
    });
    assert_eq!(
        seen(Path::new(&init.data_dir))
            .windows
            .load(Ordering::SeqCst),
        0,
        "cached cells were predicted again"
    );
}

fn cached_cells(cache_dir: &Path) -> usize {
    std::fs::read_dir(cache_dir.join("cells"))
        .map(|shelves| {
            shelves
                .filter_map(Result::ok)
                .filter_map(|shelf| std::fs::read_dir(shelf.path()).ok())
                .map(|cells| cells.count())
                .sum()
        })
        .unwrap_or(0)
}

#[test]
fn the_disk_cache_starts_once_the_slide_identity_is_known() {
    // eov publishes a slide before it has computed its identity.
    let slide = Slide::new(Some(0.325));
    let v = view(1, 1, 4_000.0, 4_000.0, 100.0, 100.0, 2.0);
    let rig = Rig::new(&[(1, slide.clone())], &[v], FakeConfig::default());
    let cache_dir = rig.data_dir.parent().unwrap().join("cache");
    let anonymous = FileInfo {
        identity: None,
        ..slide.info(1)
    };
    rig.harness.state().snapshot.files = vec![anonymous.clone()];
    rig.harness
        .send(HostEvent::FileOpened { file: anonymous })
        .unwrap();
    rig.click(OVERLAY_BUTTON);
    wait_for("the first view", LONG, || view_done(&rig, 1));
    assert_eq!(cached_cells(&cache_dir), 0, "cached without an identity");

    // The identity arrives; cells computed from then on are kept.
    rig.harness.state().snapshot.files = vec![slide.info(1)];
    let moved = view(1, 1, 6_000.0, 6_000.0, 100.0, 100.0, 2.0);
    rig.harness.state().snapshot.panes[0].view = Some(moved);
    rig.harness
        .send(HostEvent::ViewChanged { view: moved })
        .unwrap();
    wait_for("cells on disk", LONG, || cached_cells(&cache_dir) > 0);
}

#[test]
fn closing_a_slide_ends_its_readout_and_shutdown_is_prompt() {
    let slide = Slide::new(Some(0.325));
    let v = view(1, 1, 4_000.0, 4_000.0, 400.0, 400.0, 2.0);
    let rig = Rig::new(&[(1, slide)], &[v], FakeConfig::default());
    rig.harness
        .send(HostEvent::ShapeDrawn {
            tool_id: REGION_TOOL.into(),
            pane: 1,
            file_id: 1,
            shape: Shape::Rectangle {
                x: 4_000.0,
                y: 4_000.0,
                width: 300.0,
                height: 300.0,
            },
            modifiers: Modifiers::default(),
        })
        .unwrap();
    rig.harness.state().snapshot.files.clear();
    rig.harness.state().snapshot.panes.clear();
    rig.harness
        .send(HostEvent::FileClosed { file_id: 1 })
        .unwrap();
    let started = Instant::now();
    drop(rig);
    assert!(started.elapsed() < Duration::from_secs(5));
}

#[test]
fn the_manifest_is_valid_for_plugin_api_1() {
    let manifest =
        eov_plugin_api::PluginManifest::from_toml(include_str!("../plugin.toml"), "cellaris")
            .unwrap();
    assert_eq!(manifest.id, "cellaris");
    assert!(manifest.declares_environment, "no [environment] (CEL-1)");
    let host = semver::Version::new(1, 0, 0);
    assert!(manifest.accepts_host(&host));
    assert!(manifest.accepts_api(&host));
    assert!(!manifest.accepts_api(&semver::Version::new(2, 0, 0)));
    assert_eq!(manifest.version, env!("CARGO_PKG_VERSION"));
    let description = manifest.description.unwrap();
    assert!(description.contains("Research use only"), "{description}");
    // The library is named after the id.
    assert_eq!(
        eov_plugin_api::abi::plugin_library_filename(&manifest.id),
        eov_plugin_api::abi::plugin_library_filename(env!("CARGO_PKG_NAME"))
    );
}

// ---------------------------------------------------------------------------
// With real ONNX Runtime (opt-in)
// ---------------------------------------------------------------------------

/// `CELLARIS_SMOKE_ORT` (a runtime library or a directory holding one) and
/// `CELLARIS_SMOKE_MODEL` (a markers.onnx; by default the synthetic one in
/// tests/data, written by tools/make_test_model.py).
fn smoke_paths() -> Option<(PathBuf, PathBuf)> {
    let runtime = std::env::var_os("CELLARIS_SMOKE_ORT")?.into();
    let model = std::env::var_os("CELLARIS_SMOKE_MODEL")
        .map(PathBuf::from)
        .unwrap_or_else(|| {
            Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/data/markers-synthetic.onnx")
        });
    Some((runtime, model))
}

#[test]
#[ignore = "needs ONNX Runtime: set CELLARIS_SMOKE_ORT"]
fn ort_smoke_backend() {
    let Some((runtime, model)) = smoke_paths() else {
        return;
    };
    let init = test_init("cellaris");
    let request = LoadRequest {
        model: Some(model),
        runtime: Some(runtime),
        plugin_root: init.plugin_root.clone().into(),
        data_dir: init.data_dir.clone().into(),
        threads: 2,
        ..Default::default()
    };
    let mut predictor = crate::model::OrtBackend.load(&request).unwrap();
    let info = predictor.info().clone();
    println!("{info:#?}");
    assert_eq!(info.channel_names.len(), 19);
    assert_eq!(info.window, 256);
    assert_eq!(info.batch, Batch::Dynamic);
    assert!(info.recon && info.mpp_recorded && info.runtime.starts_with("ONNX Runtime 1."));
    let side = 256;
    let input = vec![0.5f32; 2 * 3 * side * side];
    let out = predictor.predict(&input, 2).unwrap();
    assert_eq!((out.n, out.channels, out.side), (2, 19, side));
    assert_eq!(out.markers.len(), 2 * 19 * side * side);
    let recon = out.recon.unwrap();
    assert!(
        recon.iter().all(|r| (r - 0.5).abs() < 1e-5),
        "recon {}",
        recon[0]
    );
    // A wrong batch is an error, not a panic.
    assert!(predictor.predict(&input[..10], 2).is_err());
    // A stop asked for between runs stops the next one (shutdown must not
    // wait for a whole batch); the run after that works again.
    let stop = predictor.stopper().unwrap();
    stop();
    assert!(predictor.predict(&input, 2).is_err(), "the stop was lost");
    assert!(predictor.predict(&input, 2).is_ok());
}

#[test]
#[ignore = "needs ONNX Runtime: set CELLARIS_SMOKE_ORT"]
fn ort_smoke_plugin() {
    let Some((runtime, model)) = smoke_paths() else {
        return;
    };
    let slide = Slide::new(Some(0.25));
    let v = view(1, 1, 4_000.0, 4_000.0, 400.0, 300.0, 2.0);
    let host = MockHost::new();
    {
        let mut state = host.state();
        state.snapshot.files = vec![slide.info(1)];
        state.snapshot.panes = vec![PaneInfo {
            pane: 1,
            index: 0,
            file_id: Some(1),
            view: Some(v),
        }];
        state.snapshot.focused_pane = Some(1);
        let reader = slide.clone();
        state.region_reader = Some(Box::new(move |r: &RegionRequest| Ok(reader.read(r))));
    }
    let mut init = test_init("cellaris");
    init.settings
        .insert("model_path".into(), json!(model.to_string_lossy()));
    init.settings
        .insert("runtime_path".into(), json!(runtime.to_string_lossy()));
    let harness = TestHarness::<Cellaris<OrtBackend>>::with_host(host, init).unwrap();
    harness
        .send(HostEvent::FileOpened {
            file: slide.info(1),
        })
        .unwrap();
    harness
        .send(HostEvent::ToolbarButtonClicked {
            button_id: OVERLAY_BUTTON.into(),
        })
        .unwrap();
    let started = Instant::now();
    wait_for("real predictions", Duration::from_secs(120), || {
        harness
            .state()
            .layers
            .get(&pred_layer(1))
            .is_some_and(|l| l.rasters.len() >= 4)
    });
    println!("first rasters after {:?}", started.elapsed());
    let state = harness.state();
    let raster = state.layers[&pred_layer(1)]
        .rasters
        .values()
        .next()
        .unwrap();
    // 128 model px of 0.325 µm over a 0.25 µm/px slide.
    assert!((raster.width - 166.4).abs() < 1e-9, "{}", raster.width);
    assert!(
        state
            .notifications
            .iter()
            .all(|n| n.level != NotifyLevel::Error),
        "{:?}",
        state.notifications
    );
    // The cells carry their values, offered from 24 screen px a cell.
    let values = &state.layers[&values_layer(1)];
    assert!(!values.objects.is_empty());
    let min_zoom = values.spec.min_zoom.unwrap();
    assert!((min_zoom - 24.0 / 166.4).abs() < 1e-9, "{min_zoom}");
    let tooltip = values.objects.values().next().unwrap().tooltip.clone();
    println!("a cell's tooltip: {tooltip:?}");
    assert!(state.badges[NOTICE_BADGE].1.contains("research use only"));
    println!("badges: {:?}", state.badges);
}

/// `ort` 2.0.0-rc.13 keeps a failed load as if it had succeeded, so a
/// candidate that cannot be loaded must never reach it: the next one would
/// "load" without loading anything.
#[cfg(unix)]
#[test]
#[ignore = "needs ONNX Runtime: set CELLARIS_SMOKE_ORT; run alone (it loads once per process)"]
fn ort_smoke_a_candidate_that_fails_does_not_spoil_the_next() {
    let Some((runtime, _)) = smoke_paths() else {
        return;
    };
    if crate::runtime::tried() {
        println!("skipped: this process has already loaded ONNX Runtime; run the test alone");
        return;
    }
    let (os, name) = if cfg!(target_os = "macos") {
        ("macos", "libonnxruntime.dylib")
    } else {
        ("linux", "libonnxruntime.so")
    };
    // A library, or an extracted release (with it in lib/).
    let library = if runtime.is_dir() {
        [runtime.clone(), runtime.join("lib")]
            .iter()
            .filter_map(|dir| std::fs::read_dir(dir).ok())
            .flatten()
            .filter_map(Result::ok)
            .map(|entry| entry.path())
            .find(|path| {
                path.file_name()
                    .and_then(|n| n.to_str())
                    .is_some_and(|n| crate::runtime::is_runtime_name(n, os))
            })
            .expect("no runtime in CELLARIS_SMOKE_ORT")
    } else {
        runtime
    };
    let init = test_init("cellaris");
    let package = PathBuf::from(&init.plugin_root).join("package");
    let data = PathBuf::from(&init.data_dir);
    std::fs::create_dir_all(&package).unwrap();
    std::fs::write(package.join(name), b"not a library").unwrap();
    std::os::unix::fs::symlink(&library, data.join(name)).unwrap();
    let loaded = crate::runtime::load(&crate::runtime::Search {
        plugin_root: Some(package),
        data_dir: Some(data),
        ..Default::default()
    })
    .unwrap();
    assert_eq!(loaded.source, crate::runtime::Source::DataDir);
    assert!(loaded.version.is_some());
    // And ort can use it.
    assert!(std::panic::catch_unwind(ort::info).is_ok());
}

#[test]
#[ignore = "needs a library that is not ONNX Runtime: set CELLARIS_SMOKE_BAD_ORT; run alone"]
fn ort_smoke_wrong_library_is_reported() {
    let Some(bad) = std::env::var_os("CELLARIS_SMOKE_BAD_ORT") else {
        return;
    };
    let failure = crate::runtime::load(&crate::runtime::Search {
        setting: Some(bad.into()),
        ..Default::default()
    })
    .unwrap_err();
    println!("{failure:#?}");
    assert!(failure.detail.contains("OrtGetApiBase"), "{failure:?}");
}

#[test]
fn a_confirmed_long_readout_runs_as_a_job_and_can_be_cancelled() {
    let slide = Slide {
        radius: 9_000.0,
        ..Slide::new(Some(0.325))
    };
    let rig = Rig::new(&[(1, slide)], &[], FakeConfig::default());
    rig.click(PANEL_BUTTON);
    rig.harness
        .send(HostEvent::ShapeDrawn {
            tool_id: REGION_TOOL.into(),
            pane: 1,
            file_id: 1,
            shape: Shape::Rectangle {
                x: 2_000.0,
                y: 2_000.0,
                width: 1_600.0,
                height: 1_600.0,
            },
            modifiers: Modifiers::default(),
        })
        .unwrap();
    wait_for("the question", LONG, || {
        !rig.harness.calls_to("dialog.confirm").is_empty()
    });
    // The mock numbers its ids "<kind>-<n>" from one counter.
    for n in 1..20 {
        rig.harness
            .send(HostEvent::ConfirmResult {
                request_id: format!("request-{n}"),
                confirmed: true,
            })
            .unwrap();
    }
    assert_eq!(rig.harness.calls_to("job.start").len(), 1);
    wait_for("some progress", LONG, || {
        rig.sidebar()["readout-status"]
            .as_str()
            .is_some_and(|s| s.starts_with("Measuring"))
            && rig.seen().windows.load(Ordering::SeqCst) > 0
    });
    for n in 1..40 {
        rig.harness
            .send(HostEvent::JobCancelRequested {
                job_id: format!("job-{n}"),
            })
            .unwrap();
    }
    assert!(
        rig.sidebar()["readout-status"]
            .as_str()
            .unwrap()
            .starts_with("Cancelled"),
        "{:?}",
        rig.sidebar()["readout-status"]
    );
    assert_eq!(rig.sidebar()["readout-running"], json!(false));
    // Nothing more is computed for it.
    std::thread::sleep(Duration::from_millis(400));
    let after = rig.seen().windows.load(Ordering::SeqCst);
    std::thread::sleep(Duration::from_millis(400));
    assert_eq!(rig.seen().windows.load(Ordering::SeqCst), after);
}

#[test]
fn a_failure_nobody_needs_any_more_does_not_spin_the_worker() {
    // One read fails, then the overlay is turned off before the retry.
    let slide = Slide::new(Some(0.325));
    let v = view(1, 1, 4_000.0, 4_000.0, 60.0, 60.0, 2.0);
    let rig = Rig::with_reader(&[(1, slide)], &[v], FakeConfig::default(), Some(1));
    rig.click(OVERLAY_BUTTON);
    wait_for("the failure", LONG, || {
        rig.objects(&marks_layer(1))
            .iter()
            .any(|o| o.object_id.starts_with("failed-"))
    });
    rig.click(OVERLAY_BUTTON);
    // Past the 2 s back-off, nothing wants the window.
    std::thread::sleep(Duration::from_millis(2_500));
    let before = rig.harness.calls_to("snapshot").len();
    std::thread::sleep(Duration::from_secs(1));
    let passes = rig.harness.calls_to("snapshot").len() - before;
    assert!(
        passes <= 3,
        "the idle worker made {passes} passes in a second"
    );
}

#[test]
fn the_same_error_at_different_places_is_reported_once() {
    let host = MockHost::new();
    let sdk = Host::from_ffi(host.to_ffi());
    let mut st = crate::engine::State::new(Default::default(), Default::default());
    for (x, y) in [(0, 0), (512, 0), (1024, 2048)] {
        crate::engine::notify_once(
            &sdk,
            &mut st,
            "Cellaris could not compute part of the slide; retrying",
            &format!("corrupt region at level 0, ({x}, {y}), 256x256"),
        );
    }
    crate::engine::notify_once(&sdk, &mut st, "Another problem", "detail");
    assert_eq!(host.state().notifications.len(), 2);
}

#[test]
fn assuming_the_pixel_size_also_rescues_implausible_ones() {
    // 5,000 µm/px is "reported" but not a pixel size anyone scans at.
    let slide = Slide::new(Some(5_000.0));
    let v = view(1, 1, 4_000.0, 4_000.0, 200.0, 200.0, 2.0);
    let rig = Rig::new(&[(1, slide)], &[v], FakeConfig::default());
    rig.click(OVERLAY_BUTTON);
    wait_for("the refusal", LONG, || {
        rig.slide_line(1).contains("pixel size unknown")
    });
    rig.harness
        .ui_callback("panel", "set-assume-mpp", vec![json!(true)])
        .unwrap();
    wait_for("assumed-scale rasters", LONG, || rig.rasters(1) > 0);
}

#[test]
fn two_panes_on_the_same_view_do_not_double_the_counts() {
    let slide = Slide::new(Some(0.325));
    let a = view(1, 1, 4_000.0, 4_000.0, 64.0, 64.0, 2.0);
    let b = view(2, 1, 4_000.0, 4_000.0, 64.0, 64.0, 2.0);
    // Every read fails, so every cell in view is pending and failed.
    let rig = Rig::with_reader(&[(1, slide)], &[a, b], FakeConfig::default(), Some(10_000));
    rig.click(OVERLAY_BUTTON);
    // The text that matched, not a later read (which may be between
    // retries, with no failure in it).
    let mut text = String::new();
    wait_for("failures in the badge", LONG, || {
        text = rig.slide_line(1);
        text.contains("failed")
    });
    // "Computing 0 of N cells · M failed, retrying". The view is 2 × 2
    // cells of 32 px, so N is 4 (8 if the two panes were added up), and M
    // can be at most N.
    let numbers: Vec<usize> = text
        .split(|c: char| !c.is_ascii_digit())
        .filter_map(|n| n.parse().ok())
        .collect();
    assert!(numbers.len() >= 3, "{text}");
    assert_eq!(numbers[1], 4, "{text}");
    assert!(numbers[2] <= numbers[1], "{text}");
}

#[test]
fn a_slide_that_disappears_ends_its_readout_job() {
    let slide = Slide::new(Some(0.325));
    let rig = Rig::new(&[(1, slide)], &[], FakeConfig::default());
    rig.harness
        .send(HostEvent::ShapeDrawn {
            tool_id: REGION_TOOL.into(),
            pane: 1,
            file_id: 1,
            shape: Shape::Rectangle {
                x: 4_000.0,
                y: 4_000.0,
                width: 400.0,
                height: 400.0,
            },
            modifiers: Modifiers::default(),
        })
        .unwrap();
    wait_for("the job", LONG, || {
        !rig.harness.calls_to("job.start").is_empty()
    });
    // The host stops publishing the slide (its FileClosed is still to come).
    rig.harness.state().snapshot.files.clear();
    wait_for("the job to end as closed", LONG, || {
        rig.harness
            .calls_to("job.finish")
            .iter()
            .any(|args| args["error"] == json!("the slide was closed"))
    });
}

mod batch_runs;
