//! `eov batch run cellaris:predict` against the mock host: slides arrive
//! as `BatchRun` events in a plugin started without a window.

use super::*;
use crate::batch::{CELLS_CSV, PREDICT, SUMMARY_JSON};
use crate::config::RESEARCH_NOTICE;

/// A disc of tissue on glass, small enough to predict whole.
fn small_slide(mpp: Option<f64>) -> Slide {
    Slide {
        width: 1_600,
        height: 1_280,
        mpp,
        downsamples: vec![1.0, 4.0],
        centre: (700.0, 600.0),
        radius: 450.0,
    }
}

/// The plugin as `eov batch` starts it.
struct Headless {
    harness: TestHarness<Plugin>,
    out: PathBuf,
    /// eov numbers the slides of a run.
    runs: AtomicUsize,
}

impl Headless {
    fn new(slides: &[(FileId, Slide)], fake: FakeConfig) -> Self {
        Self::with(slides, fake, None, |_| {})
    }

    fn with(
        slides: &[(FileId, Slide)],
        fake: FakeConfig,
        fail_first: Option<usize>,
        setup: impl FnOnce(&mut MockState),
    ) -> Self {
        let host = MockHost::new();
        {
            let mut state = host.state();
            state.snapshot.files = slides.iter().map(|(id, s)| s.info(*id)).collect();
            let slides: HashMap<FileId, Slide> = slides.iter().cloned().collect();
            let failures = AtomicUsize::new(fail_first.unwrap_or(0));
            state.region_reader = Some(Box::new(move |request: &RegionRequest| {
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
        let mut init = test_init("cellaris");
        init.headless = true;
        let root = PathBuf::from(&init.plugin_root);
        assert!(root.starts_with(std::env::temp_dir()));
        std::fs::write(
            Path::new(&init.data_dir).join("fake.json"),
            serde_json::to_string(&fake).unwrap(),
        )
        .unwrap();
        Headless {
            harness: TestHarness::with_host(host, init).unwrap(),
            out: root.join("out"),
            runs: AtomicUsize::new(0),
        }
    }

    fn dir(&self, file_id: FileId) -> PathBuf {
        self.out.join(format!("slide-{file_id}"))
    }

    /// One slide, the way `eov batch run` sends it: opened, run with the
    /// command's defaults under `params`, closed once finished.
    fn run(&self, file: FileInfo, params: Value) -> BatchFinishArgs {
        let run_id = format!("{PREDICT}-{}", self.runs.fetch_add(1, Ordering::SeqCst) + 1);
        let dir = self.dir(file.file_id);
        std::fs::create_dir_all(&dir).unwrap();
        let mut all: Map<String, Value> = crate::batch::command()
            .params
            .into_iter()
            .filter(|p| !p.default.is_null())
            .map(|p| (p.name, p.default))
            .collect();
        if let Value::Object(given) = params {
            all.extend(given);
        }
        self.harness
            .send(HostEvent::FileOpened { file: file.clone() })
            .unwrap();
        self.harness
            .send(HostEvent::BatchRun {
                run_id: run_id.clone(),
                command_id: PREDICT.into(),
                file: file.clone(),
                output_dir: dir.to_string_lossy().into_owned(),
                params: all,
            })
            .unwrap();
        let mut finish = None;
        wait_for("the batch result", LONG, || {
            finish = self
                .harness
                .state()
                .batch_finished
                .iter()
                .find(|f| f.run_id == run_id)
                .cloned();
            finish.is_some()
        });
        self.harness
            .send(HostEvent::FileClosed {
                file_id: file.file_id,
            })
            .unwrap();
        finish.unwrap()
    }

    /// The table's comment lines, its header, and its rows.
    fn table(&self, file_id: FileId) -> (Vec<String>, Vec<String>, Vec<Vec<String>>) {
        let text = std::fs::read_to_string(self.dir(file_id).join(CELLS_CSV)).unwrap();
        let (comments, rest): (Vec<&str>, Vec<&str>) =
            text.lines().partition(|line| line.starts_with('#'));
        let header = rest[0].split(',').map(String::from).collect();
        let rows = rest[1..]
            .iter()
            .map(|line| line.split(',').map(String::from).collect())
            .collect();
        (
            comments.into_iter().map(String::from).collect(),
            header,
            rows,
        )
    }
}

fn number(row: &[String], header: &[String], column: &str) -> f64 {
    let index = header.iter().position(|h| h == column).unwrap();
    row[index].parse().unwrap()
}

fn region(x: f64, y: f64, width: f64, height: f64) -> Value {
    json!(format!("{x},{y},{width},{height}"))
}

#[test]
fn a_headless_plugin_offers_predict_and_sets_up_no_window() {
    let batch = Headless::new(&[(1, small_slide(Some(0.325)))], FakeConfig::default());
    batch
        .harness
        .send(HostEvent::FileOpened {
            file: small_slide(Some(0.325)).info(1),
        })
        .unwrap();
    std::thread::sleep(Duration::from_millis(200));
    let state = batch.harness.state();
    let command = &state.batch_commands[PREDICT];
    assert!(command.description.contains("Research use only"));
    let markers = command.params.iter().find(|p| p.name == "markers").unwrap();
    assert_eq!(markers.default, json!("Hoechst,Pan-CK,SMA"));
    // Nothing for a window: no buttons, tool, layers or worker polling.
    assert!(state.toolbar_buttons.is_empty() && state.tools.is_empty());
    assert!(state.layers.is_empty());
    assert!(state.calls_to("snapshot").is_empty());
}

#[test]
fn predict_writes_a_row_per_cell_with_tissue_and_a_summary() {
    let slide = small_slide(Some(0.325));
    let batch = Headless::new(&[(1, slide.clone())], FakeConfig::default());
    let finish = batch.run(slide.info(1), json!({}));
    assert!(finish.ok, "{:?}", finish.message);
    assert_eq!(finish.outputs, vec![CELLS_CSV, SUMMARY_JSON]);
    let summary = &finish.summary;
    let on_disk: Value =
        serde_json::from_str(&std::fs::read_to_string(batch.dir(1).join(SUMMARY_JSON)).unwrap())
            .unwrap();
    assert_eq!(*summary, on_disk);

    // The table: the notice first, then one row per cell with tissue.
    let (comments, header, rows) = batch.table(1);
    assert!(
        comments[0].starts_with(&format!("# {RESEARCH_NOTICE}")),
        "{comments:?}"
    );
    assert!(comments[0].contains(&format!("sha256:{}", "fa4e".repeat(16))));
    assert_eq!(
        header,
        [
            "x",
            "y",
            "width",
            "height",
            "size_um",
            "tissue_blocks",
            "dark_blocks",
            "od_mean",
            "recon_error",
            "Hoechst_mean",
            "Pan-CK_mean",
            "SMA_mean"
        ]
    );
    assert_eq!(
        rows.len() as u64,
        summary["cells"]["with_tissue"].as_u64().unwrap()
    );
    assert!(rows.len() > 100, "{} rows", rows.len());
    for row in &rows {
        // 32 model px of 0.325 µm on a 0.325 µm/px slide.
        assert_eq!(number(row, &header, "width"), 32.0);
        assert!((number(row, &header, "size_um") - 10.4).abs() < 1e-9);
        // Flat channels are flat: the windows' joins are feathered away.
        assert!(
            (number(row, &header, "SMA_mean") - 5.0).abs() < 1e-3,
            "{row:?}"
        );
        let blocks = number(row, &header, "tissue_blocks");
        assert!((1.0..=4.0).contains(&blocks), "{row:?}");
        // Every row is on the disc of tissue.
        let (cx, cy) = (
            number(row, &header, "x") + 16.0,
            number(row, &header, "y") + 16.0,
        );
        assert!((cx - 700.0).hypot(cy - 600.0) < 450.0 + 32.0, "{row:?}");
    }

    // The summary: what was predicted with what, and what it found.
    assert_eq!(summary["notice"], json!(RESEARCH_NOTICE));
    assert_eq!(summary["complete"], json!(true));
    assert_eq!(summary["model"]["sha256"], json!("fa4e".repeat(16)));
    assert_eq!(summary["runtime"]["device"], json!("cpu"));
    assert_eq!(summary["slide"]["pixel_size"], json!("reported"));
    assert_eq!(summary["region"]["source"], json!("the whole slide"));
    let cells = &summary["cells"];
    let sum: u64 = [
        "with_tissue",
        "glass",
        "dark",
        "no_measurable_tissue",
        "failed",
    ]
    .iter()
    .map(|key| cells[key].as_u64().unwrap())
    .sum();
    assert_eq!(sum, cells["in_region"].as_u64().unwrap(), "{cells}");
    assert_eq!(cells["in_region"], json!(50 * 40));
    // Tissue: about the disc's area (π · 450² px of 0.325 µm).
    let disc = std::f64::consts::PI * 450.0 * 450.0 * 0.325 * 0.325 / 1e6;
    let area = summary["tissue"]["area_mm2"].as_f64().unwrap();
    assert!((area / disc - 1.0).abs() < 0.2, "{area} mm² against {disc}");
    let markers = summary["markers"].as_array().unwrap();
    assert_eq!(markers.len(), 3);
    // Hoechst is a density map by construction; SMA is flat.
    assert!(
        markers[0]["density_r2"].as_f64().unwrap() > 0.95,
        "{}",
        markers[0]
    );
    assert!((markers[2]["mean"].as_f64().unwrap() - 5.0).abs() < 1e-3);
    for marker in markers {
        let p: Vec<f64> = ["p1", "p5", "p25", "p50", "p75", "p95", "p99"]
            .iter()
            .map(|k| marker["percentiles"][k].as_f64().unwrap())
            .collect();
        assert!(p.windows(2).all(|w| w[0] <= w[1]), "{marker}");
    }
    // To a 0.01 bin.
    assert!((markers[2]["percentiles"]["p50"].as_f64().unwrap() - 5.0).abs() < 0.011);
    // No local paths or names of the slide (runs may be de-identified).
    let text = summary.to_string();
    assert!(
        !text.contains("slide-1.svs") && !text.contains(&*std::env::temp_dir().to_string_lossy())
    );

    // Progress was reported as it went.
    let state = batch.harness.state();
    let last = state.batch_progress.last().unwrap();
    assert_eq!(last.run_id, finish.run_id);
    assert_eq!(last.fraction, Some(1.0));
}

#[test]
fn predict_gives_the_numbers_a_region_readout_gives() {
    let slide = small_slide(Some(0.325));
    let (x, y, side) = (500.0, 450.0, 300.0);

    // The readout, in the window.
    let rig = Rig::new(&[(1, slide.clone())], &[], FakeConfig::default());
    rig.click(PANEL_BUTTON);
    rig.harness
        .send(HostEvent::ShapeDrawn {
            tool_id: REGION_TOOL.into(),
            pane: 1,
            file_id: 1,
            shape: Shape::Rectangle {
                x,
                y,
                width: side,
                height: side,
            },
            modifiers: Modifiers::default(),
        })
        .unwrap();
    wait_for("the readout", LONG, || {
        rig.sidebar().get("can-copy") == Some(&json!(true))
    });
    rig.harness
        .ui_callback("panel", "copy-readout", vec![])
        .unwrap();
    let copied = rig.harness.state().clipboard.clone().unwrap();

    // The batch run, over the same rectangle.
    let batch = Headless::new(&[(1, slide.clone())], FakeConfig::default());
    let finish = batch.run(slide.info(1), json!({ "region": region(x, y, side, side) }));
    assert!(finish.ok, "{:?}", finish.message);
    assert_eq!(finish.summary["region"]["source"], json!("--param region"));

    let blocks = finish.summary["tissue"]["blocks"].as_u64().unwrap();
    assert!(
        copied.contains(&format!("# tissue\t{blocks} blocks")),
        "{blocks} blocks, readout:\n{copied}"
    );
    for marker in finish.summary["markers"].as_array().unwrap() {
        let name = marker["name"].as_str().unwrap();
        let line = copied
            .lines()
            .find(|line| line.starts_with(&format!("{name}\t")))
            .unwrap();
        let readout: f64 = line.split('\t').nth(1).unwrap().parse().unwrap();
        let mean = marker["mean"].as_f64().unwrap();
        assert!(
            (readout - mean).abs() < 1e-4,
            "{name}: {readout} against {mean}"
        );
    }
}

#[test]
fn predict_needs_a_pixel_size_or_assume_mpp() {
    let slide = small_slide(None);
    let batch = Headless::new(&[(1, slide.clone())], FakeConfig::default());
    let finish = batch.run(slide.info(1), json!({}));
    assert!(!finish.ok);
    let message = finish.message.unwrap();
    assert!(
        message.contains("does not report its pixel size")
            && message.contains("--param assume_mpp=true"),
        "{message}"
    );
    assert!(!batch.dir(1).join(CELLS_CSV).exists());

    let finish = batch.run(
        slide.info(1),
        json!({ "assume_mpp": true, "region": region(600.0, 500.0, 128.0, 128.0) }),
    );
    assert!(finish.ok, "{:?}", finish.message);
    assert_eq!(
        finish.summary["slide"]["pixel_size"],
        json!("assumed (assume_mpp)")
    );
    let warnings = finish.summary["warnings"].to_string();
    assert!(warnings.contains("assumed 0.325"), "{warnings}");
}

#[test]
fn predict_says_why_the_model_did_not_load() {
    let slide = small_slide(Some(0.325));
    let fake = FakeConfig {
        fail_load: Some("no markers.onnx in the data directory".into()),
        ..Default::default()
    };
    let batch = Headless::new(&[(1, slide.clone())], fake);
    let finish = batch.run(slide.info(1), json!({}));
    assert!(!finish.ok);
    let message = finish.message.unwrap();
    assert!(
        message.starts_with("No marker model: no markers.onnx")
            && message.contains("--param model=PATH"),
        "{message}"
    );
}

#[test]
fn predict_stops_at_max_cells_and_reports_any_channel() {
    let slide = small_slide(Some(0.325));
    let batch = Headless::new(&[(1, slide.clone())], FakeConfig::default());
    let finish = batch.run(
        slide.info(1),
        json!({ "markers": "all", "max_cells": 5, "region": region(400.0, 400.0, 400.0, 400.0) }),
    );
    assert!(finish.ok, "{:?}", finish.message);
    assert_eq!(finish.summary["complete"], json!(false));
    assert!(finish.message.unwrap().contains("max_cells"));
    let (_, header, rows) = batch.table(1);
    assert_eq!(rows.len(), 5);
    // All 19 of the fake model's channels, displayable or not.
    assert_eq!(header.len(), 9 + 19);
    assert!(header.contains(&"CD8a_mean".to_string()));

    let finish = batch.run(slide.info(1), json!({ "markers": "Hoechst,Nope" }));
    assert!(!finish.ok);
    assert!(
        finish
            .message
            .unwrap()
            .contains("the model has no channel named Nope")
    );
    let finish = batch.run(
        slide.info(1),
        json!({ "region": region(50_000.0, 0.0, 10.0, 10.0) }),
    );
    assert_eq!(
        finish.message.as_deref(),
        Some("the region lies outside the slide")
    );
}

#[test]
fn predict_leaves_out_what_it_cannot_read_and_says_so() {
    let slide = small_slide(Some(0.325));
    // Each window is tried twice: the first one fails, the rest are read.
    let batch = Headless::with(
        &[(1, slide.clone())],
        FakeConfig::default(),
        Some(2),
        |_| {},
    );
    let finish = batch.run(
        slide.info(1),
        json!({ "region": region(550.0, 450.0, 200.0, 200.0) }),
    );
    assert!(finish.ok, "{:?}", finish.message);
    assert_eq!(finish.summary["complete"], json!(false));
    assert_eq!(finish.summary["windows"]["failed"], json!(1));
    assert!(finish.summary["cells"]["failed"].as_u64().unwrap() >= 1);
    let message = finish.message.unwrap();
    assert!(
        message.contains("1 windows could not be read") && message.contains("went away"),
        "{message}"
    );
}

#[test]
fn predict_keeps_to_the_slides_region_of_interest() {
    let slide = small_slide(Some(0.325));
    // 5 × 3 cells of 32 px, all tissue.
    let roi = Rect {
        x: 608.0,
        y: 512.0,
        width: 160.0,
        height: 96.0,
    };
    let batch = Headless::with(
        &[(1, slide.clone())],
        FakeConfig::default(),
        None,
        |state| {
            state.roi.insert(1, roi);
        },
    );
    let finish = batch.run(slide.info(1), json!({}));
    assert!(finish.ok, "{:?}", finish.message);
    assert_eq!(
        finish.summary["region"]["source"],
        json!("the region of interest")
    );
    let (_, header, rows) = batch.table(1);
    assert_eq!(rows.len(), 5 * 3);
    for row in &rows {
        let (x, y) = (number(row, &header, "x"), number(row, &header, "y"));
        assert!(
            (608.0..768.0).contains(&x) && (512.0..608.0).contains(&y),
            "{row:?}"
        );
    }
}

#[test]
#[ignore = "needs ONNX Runtime: set CELLARIS_SMOKE_ORT"]
fn ort_smoke_batch() {
    let Some((runtime, model)) = smoke_paths() else {
        return;
    };
    let slide = small_slide(Some(0.25));
    let host = MockHost::new();
    {
        let mut state = host.state();
        state.snapshot.files = vec![slide.info(1)];
        let reader = slide.clone();
        state.region_reader = Some(Box::new(move |r: &RegionRequest| Ok(reader.read(r))));
    }
    let mut init = test_init("cellaris");
    init.headless = true;
    let out = PathBuf::from(&init.plugin_root).join("out");
    std::fs::create_dir_all(&out).unwrap();
    let harness = TestHarness::<Cellaris<OrtBackend>>::with_host(host, init).unwrap();
    let mut params: Map<String, Value> = crate::batch::command()
        .params
        .into_iter()
        .filter(|p| !p.default.is_null())
        .map(|p| (p.name, p.default))
        .collect();
    params.insert("model".into(), json!(model.to_string_lossy()));
    params.insert("runtime".into(), json!(runtime.to_string_lossy()));
    let started = Instant::now();
    harness
        .send(HostEvent::BatchRun {
            run_id: "predict-1".into(),
            command_id: PREDICT.into(),
            file: slide.info(1),
            output_dir: out.to_string_lossy().into_owned(),
            params,
        })
        .unwrap();
    wait_for("the batch result", Duration::from_secs(300), || {
        !harness.state().batch_finished.is_empty()
    });
    let finish = harness.state().batch_finished[0].clone();
    println!("predicted the slide in {:?}", started.elapsed());
    assert!(finish.ok, "{:?}", finish.message);
    let summary = &finish.summary;
    println!(
        "{}",
        serde_json::to_string_pretty(&summary["runtime"]).unwrap()
    );
    assert!(
        summary["runtime"]["library"]
            .as_str()
            .is_some_and(|name| name.contains("onnxruntime")),
        "{}",
        summary["runtime"]
    );
    assert_eq!(
        summary["model"]["file"],
        json!(model.file_name().unwrap().to_string_lossy())
    );
    // 128 model px of 0.325 µm over a 0.25 µm/px slide.
    assert!((summary["grid"]["cell_px"][0].as_f64().unwrap() - 166.4).abs() < 1e-9);
    assert!(summary["cells"]["with_tissue"].as_u64().unwrap() > 0);
    let table = std::fs::read_to_string(out.join(CELLS_CSV)).unwrap();
    assert!(table.starts_with(&format!("# {RESEARCH_NOTICE}")));
}
