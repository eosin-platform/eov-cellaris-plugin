//! Headless batch runs: `eov batch run cellaris:predict SLIDES --out DIR`.
//!
//! Each slide's tissue, or a region of it, is predicted at the model's own
//! pixel size the way the overlay does it: the same grid, windows and
//! feathered cells, and the same tissue blocks a region readout measures.
//! A slide's results are one CSV row per cell with tissue and a summary,
//! which eov also puts in the run's manifest (`batch.json`).
//!
//! Memory stays bounded whatever the slide's size: the slide is scanned in
//! strips [`STRIP`] cells wide, so only about two rows of a strip's cells
//! are being blended at any time. Rows go straight to the CSV, and the
//! summary keeps sums and fixed histograms.
//!
//! **Research use only**: the notice heads the CSV and the summary.

use crate::blend::{Accum, CellData, Channels, Ramp, expected_bits, finalize};
use crate::config::{Prefs, RESEARCH_NOTICE};
use crate::engine::{Paths, window_planes};
use crate::geometry::{BLOCK, Grid, Idx, Scale, ScaleProblem, ScaleSource};
use crate::markers::{self, DEFAULT_MARKERS, Marker};
use crate::model::{Backend, LoadError, LoadRequest, ModelInfo, Predictor, Stopper};
use crate::present::{self, WindowCounts};
use crate::region::{BlockUse, Fit, Region, block_use};
use crate::resample::{self, WindowImage, WindowKind};
use eov_plugin_api::prelude::*;
use std::collections::{HashMap, HashSet};
use std::fs::File;
use std::io::{BufWriter, Write};
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{Sender, channel};
use std::sync::{Arc, Mutex};
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

/// The command: `eov batch run cellaris:predict`.
pub const PREDICT: &str = "predict";
/// A slide's files, in its output folder.
pub const CELLS_CSV: &str = "cells.csv";
pub const SUMMARY_JSON: &str = "summary.json";
/// Cell columns per strip. Each strip reads again the column of windows it
/// shares with the one before (1/32 more windows), so that no cell waits
/// for a strip still to come.
pub const STRIP: i64 = 32;
/// A window that cannot be read is tried this many times, then its cells
/// are left out.
const READ_ATTEMPTS: usize = 2;
const PROGRESS_INTERVAL: Duration = Duration::from_millis(500);
/// Block values are binned over this arcsinh range for percentiles.
const HIST_LOW: f64 = -4.0;
const HIST_HIGH: f64 = 12.0;
/// 0.01 wide.
const HIST_BINS: usize = 1600;
const PERCENTILES: [u32; 7] = [1, 5, 25, 50, 75, 95, 99];

/// What `eov batch list` shows, and the parameters' defaults (which the
/// manifest records with each run).
pub fn command() -> BatchCommandSpec {
    let param = |name: &str, description: &str, default: Value| BatchParam {
        name: name.into(),
        description: description.into(),
        default,
    };
    BatchCommandSpec {
        command_id: PREDICT.into(),
        description: "Predicted IF markers over each slide's tissue: values per 41.6 \u{b5}m cell \
                      (cells.csv) and a summary. Research use only."
            .into(),
        params: vec![
            param(
                "markers",
                "Channels to report: names (\"Hoechst,SMA\") or \"all\" of the model's",
                json!(DEFAULT_MARKERS.join(",")),
            ),
            param(
                "assume_mpp",
                "Predict slides that report no pixel size as if they were at the model's",
                json!(false),
            ),
            param(
                "region",
                "Only this level-0 rectangle, \"x,y,width,height\" (default: the slide's region \
                 of interest, else the whole slide)",
                Value::Null,
            ),
            param(
                "max_cells",
                "Stop after this many cells with tissue (0: no limit)",
                json!(0),
            ),
            param(
                "model",
                "The marker model, .onnx (default: as in the Cellaris panel)",
                Value::Null,
            ),
            param(
                "runtime",
                "The ONNX Runtime library (default: as in the Cellaris panel)",
                Value::Null,
            ),
            param(
                "threads",
                "Inference threads (0: half the CPUs, at most 8)",
                json!(0),
            ),
            param("batch_size", "Windows per inference run", json!(4)),
        ],
    }
}

/// Which channels a run reports.
#[derive(Clone, Debug, PartialEq)]
pub enum Markers {
    All,
    Named(Vec<String>),
}

/// A run's parameters, over the plugin's settings for what they leave out.
#[derive(Clone, Debug, PartialEq)]
pub struct Params {
    pub markers: Markers,
    pub assume_mpp: bool,
    pub region: Option<Rect>,
    pub max_cells: u64,
    /// `--param model` and `--param runtime`; without them the model and
    /// runtime are found as in the panel.
    pub model: Option<PathBuf>,
    pub runtime: Option<PathBuf>,
    pub threads: usize,
    pub batch_size: usize,
}

fn bad(name: &str, value: &Value, expected: &str) -> String {
    format!("--param {name}={value}: expected {expected}")
}

fn flag(value: &Value) -> Option<bool> {
    match value {
        Value::Bool(flag) => Some(*flag),
        Value::Number(n) => n.as_u64().filter(|n| *n <= 1).map(|n| n == 1),
        Value::String(text) => match text.trim().to_ascii_lowercase().as_str() {
            "true" | "yes" | "on" | "1" => Some(true),
            "false" | "no" | "off" | "0" => Some(false),
            _ => None,
        },
        _ => None,
    }
}

fn path(value: &Value) -> Option<Option<PathBuf>> {
    match value {
        Value::Null => Some(None),
        Value::String(text) if !text.trim().is_empty() => Some(Some(PathBuf::from(text.trim()))),
        _ => None,
    }
}

/// `"x,y,width,height"`, `[x, y, width, height]` or `{x, y, width, height}`.
fn region(value: &Value) -> Option<Option<Rect>> {
    let numbers: Vec<f64> = match value {
        Value::Null => return Some(None),
        Value::String(text) => text
            .split(',')
            .map(|n| n.trim().parse::<f64>().ok())
            .collect::<Option<_>>()?,
        Value::Array(items) => items.iter().map(Value::as_f64).collect::<Option<_>>()?,
        Value::Object(map) => ["x", "y", "width", "height"]
            .iter()
            .map(|key| map.get(*key).and_then(Value::as_f64))
            .collect::<Option<_>>()?,
        _ => return None,
    };
    let [x, y, width, height] = numbers[..] else {
        return None;
    };
    (numbers.iter().all(|n| n.is_finite()) && width > 0.0 && height > 0.0).then_some(Some(Rect {
        x,
        y,
        width,
        height,
    }))
}

fn marker_names(value: &Value) -> Option<Markers> {
    let names: Vec<String> = match value {
        Value::String(text) if text.trim().eq_ignore_ascii_case("all") => {
            return Some(Markers::All);
        }
        Value::String(text) => text.split(',').map(|n| n.trim().to_string()).collect(),
        Value::Array(items) => items
            .iter()
            .map(|item| item.as_str().map(|n| n.trim().to_string()))
            .collect::<Option<_>>()?,
        _ => return None,
    };
    let names: Vec<String> = names.into_iter().filter(|n| !n.is_empty()).collect();
    (!names.is_empty()).then_some(Markers::Named(names))
}

impl Params {
    pub fn parse(params: &Map<String, Value>, prefs: &Prefs) -> Result<Params, String> {
        let mut out = Params {
            markers: Markers::Named(DEFAULT_MARKERS.map(String::from).to_vec()),
            assume_mpp: false,
            region: None,
            max_cells: 0,
            model: None,
            runtime: None,
            threads: prefs.threads,
            batch_size: prefs.batch_size,
        };
        // eov refuses names the command does not declare.
        for (name, value) in params {
            match name.as_str() {
                "markers" => {
                    out.markers = marker_names(value)
                        .ok_or_else(|| bad(name, value, "marker names or \"all\""))?;
                }
                "assume_mpp" => {
                    out.assume_mpp =
                        flag(value).ok_or_else(|| bad(name, value, "true or false"))?;
                }
                "region" => {
                    out.region = region(value)
                        .ok_or_else(|| bad(name, value, "x,y,width,height in level-0 pixels"))?;
                }
                "max_cells" => {
                    out.max_cells = value
                        .as_u64()
                        .ok_or_else(|| bad(name, value, "a whole number"))?;
                }
                "model" | "runtime" => {
                    let choice =
                        path(value).ok_or_else(|| bad(name, value, "the path of a file"))?;
                    if let Some(choice) = choice {
                        if name == "model" {
                            out.model = Some(choice);
                        } else {
                            out.runtime = Some(choice);
                        }
                    }
                }
                "threads" => {
                    out.threads = value
                        .as_u64()
                        .filter(|n| *n <= 64)
                        .ok_or_else(|| bad(name, value, "0 to 64"))?
                        as usize;
                }
                "batch_size" => {
                    out.batch_size = value
                        .as_u64()
                        .filter(|n| (1..=32).contains(n))
                        .ok_or_else(|| bad(name, value, "1 to 32"))?
                        as usize;
                }
                _ => {}
            }
        }
        Ok(out)
    }
}

/// Values binned to 0.01 for percentiles, in fixed memory.
#[derive(Clone, Debug)]
pub struct Histogram {
    counts: Vec<u64>,
    total: u64,
}

impl Default for Histogram {
    fn default() -> Self {
        Self {
            counts: vec![0; HIST_BINS],
            total: 0,
        }
    }
}

impl Histogram {
    pub fn add(&mut self, value: f64) {
        if !value.is_finite() {
            return;
        }
        let t = (value - HIST_LOW) / (HIST_HIGH - HIST_LOW) * HIST_BINS as f64;
        let bin = (t.floor() as i64).clamp(0, HIST_BINS as i64 - 1) as usize;
        self.counts[bin] += 1;
        self.total += 1;
    }

    /// The value at or below which `p` % of the values lie, to the middle of
    /// its 0.01 bin (values beyond the range count at its ends).
    pub fn percentile(&self, p: u32) -> Option<f64> {
        if self.total == 0 {
            return None;
        }
        let rank = ((f64::from(p) / 100.0) * self.total as f64).ceil().max(1.0) as u64;
        let width = (HIST_HIGH - HIST_LOW) / HIST_BINS as f64;
        let mut seen = 0;
        for (bin, count) in self.counts.iter().enumerate() {
            seen += count;
            if seen >= rank {
                return Some(HIST_LOW + (bin as f64 + 0.5) * width);
            }
        }
        Some(HIST_HIGH)
    }
}

/// A slide of a batch run.
pub struct Job {
    pub run_id: String,
    pub command_id: String,
    pub file: FileInfo,
    pub output_dir: PathBuf,
    pub params: Map<String, Value>,
}

/// Predicts batch slides one at a time, on a thread of its own, which
/// starts with the first slide.
#[derive(Default)]
pub struct BatchRunner {
    thread: Mutex<Option<(Sender<Job>, JoinHandle<()>)>>,
    stop: Arc<AtomicBool>,
    /// Interrupts a running prediction.
    stopper: Arc<Mutex<Option<Stopper>>>,
}

impl BatchRunner {
    /// Queues a slide. Its result is sent with `finish_batch` when done.
    pub fn submit<B: Backend>(&self, host: &Host, paths: &Paths, prefs: &Prefs, job: Job) {
        let mut thread = self.thread.lock().unwrap_or_else(|e| e.into_inner());
        if thread.is_none() {
            let (sender, receiver) = channel::<Job>();
            let mut slides = Slides::<B> {
                host: host.clone(),
                backend: B::default(),
                predictor: None,
                paths: paths.clone(),
                prefs: prefs.clone(),
                stop: Arc::clone(&self.stop),
                stopper: Arc::clone(&self.stopper),
            };
            let spawned = std::thread::Builder::new()
                .name("cellaris-batch".into())
                .spawn(move || {
                    for job in receiver {
                        let finish = slides.run(job);
                        let _ = slides.host.finish_batch(finish);
                    }
                });
            match spawned {
                Ok(handle) => *thread = Some((sender, handle)),
                Err(error) => {
                    let _ = host.finish_batch(failed(
                        job.run_id,
                        format!("Cellaris could not start its batch thread: {error}"),
                    ));
                    return;
                }
            }
        }
        if let Some((sender, _)) = thread.as_ref()
            && let Err(refused) = sender.send(job)
        {
            let _ = host.finish_batch(failed(
                refused.0.run_id,
                "the Cellaris batch thread has stopped".to_string(),
            ));
        }
    }

    /// Stops the slide being predicted (it finishes as failed).
    pub fn stop(&self) {
        self.stop.store(true, Ordering::SeqCst);
        if let Some(stop) = self
            .stopper
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .as_ref()
        {
            stop();
        }
    }

    /// Stops, and waits up to `wait` for the thread to end.
    pub fn shut_down(&self, wait: Duration) {
        self.stop();
        let thread = self.thread.lock().unwrap_or_else(|e| e.into_inner()).take();
        if let Some((sender, handle)) = thread {
            drop(sender);
            let deadline = Instant::now() + wait;
            while !handle.is_finished() && Instant::now() < deadline {
                std::thread::sleep(Duration::from_millis(10));
            }
            if handle.is_finished() {
                let _ = handle.join();
            }
        }
    }
}

fn failed(run_id: String, message: String) -> BatchFinishArgs {
    BatchFinishArgs {
        run_id,
        ok: false,
        message: Some(message),
        ..Default::default()
    }
}

/// The batch thread's state: the model, loaded with the first slide.
struct Slides<B: Backend> {
    host: Host,
    backend: B,
    predictor: Option<Box<dyn Predictor>>,
    paths: Paths,
    prefs: Prefs,
    stop: Arc<AtomicBool>,
    stopper: Arc<Mutex<Option<Stopper>>>,
}

/// A slide predicted: its summary, and a note if it is incomplete.
struct Done {
    summary: Value,
    message: Option<String>,
}

impl<B: Backend> Slides<B> {
    fn run(&mut self, job: Job) -> BatchFinishArgs {
        if job.command_id != PREDICT {
            return failed(
                job.run_id,
                format!("Cellaris has no batch command {}", job.command_id),
            );
        }
        let csv = job.output_dir.join(CELLS_CSV);
        let result =
            catch_unwind(AssertUnwindSafe(|| self.predict(&job))).unwrap_or_else(|payload| {
                Err(format!(
                    "Cellaris hit an internal error: {}",
                    panic_text(payload.as_ref())
                ))
            });
        match result {
            Ok(done) => BatchFinishArgs {
                run_id: job.run_id,
                ok: true,
                message: done.message,
                outputs: vec![CELLS_CSV.into(), SUMMARY_JSON.into()],
                summary: done.summary,
            },
            Err(message) => {
                // A partial table is not a result.
                let _ = std::fs::remove_file(&csv);
                failed(job.run_id, message)
            }
        }
    }

    /// The model, loaded once for the whole run.
    fn load(&mut self, params: &Params) -> Result<(), String> {
        if self.predictor.is_some() {
            return Ok(());
        }
        let request = LoadRequest {
            model: self.prefs.model_path.as_ref().map(PathBuf::from),
            runtime: self.prefs.runtime_path.as_ref().map(PathBuf::from),
            plugin_root: self.paths.plugin_root.clone(),
            data_dir: self.paths.data_dir.clone(),
            threads: if params.threads == 0 {
                crate::model::default_threads()
            } else {
                params.threads
            },
            param_model: params.model.clone(),
            param_runtime: params.runtime.clone(),
        };
        let loaded = catch_unwind(AssertUnwindSafe(|| self.backend.load(&request))).unwrap_or_else(
            |payload| {
                Err(LoadError::new(
                    "Loading the model crashed",
                    panic_text(payload.as_ref()),
                ))
            },
        );
        match loaded {
            Ok(predictor) => {
                *self.stopper.lock().unwrap_or_else(|e| e.into_inner()) = predictor.stopper();
                let info = predictor.info();
                self.host.info(format!(
                    "Cellaris batch: {} (sha256 {}) with {}",
                    info.file_name(),
                    info.short_hash(),
                    info.runtime
                ));
                self.predictor = Some(predictor);
                Ok(())
            }
            Err(error) => Err(format!(
                "{}: {} (a model and ONNX Runtime can be given with --param model=PATH and \
                 --param runtime=PATH)",
                error.summary, error.detail
            )),
        }
    }

    fn predict(&mut self, job: &Job) -> Result<Done, String> {
        let started = Instant::now();
        let params = Params::parse(&job.params, &self.prefs)?;
        self.load(&params)?;
        let Slides {
            host,
            predictor,
            stop,
            ..
        } = self;
        let predictor = predictor
            .as_deref_mut()
            .ok_or_else(|| "the model is not loaded".to_string())?;
        let info = predictor.info().clone();
        let names = match &params.markers {
            Markers::All => info.channel_names.clone(),
            Markers::Named(names) => names.clone(),
        };
        let markers = markers::resolve(&info.channel_names, &names)?;
        let scale = Scale::of(&job.file, info.mpp, params.assume_mpp)
            .map_err(|problem| scale_error(&problem, &info))?;
        let grid = Grid::new(&job.file, scale, info.mpp, info.window)
            .map_err(|problem| scale_error(&problem, &info))?;
        let (region, source) = match params.region {
            Some(rect) => (Some(rect), "--param region"),
            None => match host.roi(job.file.file_id) {
                Ok(Some(rect)) if rect.width > 0.0 && rect.height > 0.0 => {
                    (Some(rect), "the region of interest")
                }
                _ => (None, "the whole slide"),
            },
        };
        let range = grid.cell_range(&region.unwrap_or(grid.area));
        if range.0 == range.1 || range.2 == range.3 {
            return Err("the region lies outside the slide".into());
        }
        let path = job.output_dir.join(CELLS_CSV);
        let file =
            File::create(&path).map_err(|e| format!("could not write {}: {e}", path.display()))?;
        let channels = Channels {
            markers: markers.len(),
            recon: info.recon,
        };
        let mut scan = Scan {
            host,
            run_id: &job.run_id,
            grid: &grid,
            info: &info,
            markers: &markers,
            channels,
            ramp: Ramp::new(grid.window),
            region: region.map(Region::Rectangle),
            stop: stop.as_ref(),
            max_cells: params.max_cells,
            batch: info.batch.size(params.batch_size),
            csv: BufWriter::new(file),
            counts: Counts::default(),
            fits: vec![Fit::default(); markers.len()],
            hists: vec![Histogram::default(); markers.len()],
            first_error: None,
            started,
            last_progress: Instant::now() - PROGRESS_INTERVAL,
        };
        scan.header()
            .map_err(|e| format!("could not write {}: {e}", path.display()))?;
        let stopped_early = scan.run(predictor, range)?;
        scan.csv
            .flush()
            .map_err(|e| format!("could not write {}: {e}", path.display()))?;
        let (summary, message) = scan.summary(&job.file, region, source, stopped_early);
        let json = job.output_dir.join(SUMMARY_JSON);
        std::fs::write(
            &json,
            serde_json::to_string_pretty(&summary).unwrap_or_default(),
        )
        .map_err(|e| format!("could not write {}: {e}", json.display()))?;
        Ok(Done { summary, message })
    }
}

fn scale_error(problem: &ScaleProblem, info: &ModelInfo) -> String {
    match problem {
        ScaleProblem::Unknown => format!(
            "the slide does not report its pixel size, so it cannot be matched to the model's; \
             run with --param assume_mpp=true to assume the model's {:.3} \u{b5}m/px",
            info.mpp
        ),
        other => other.message(),
    }
}

/// What a slide's scan found.
#[derive(Clone, Debug, Default)]
struct Counts {
    /// Cells of the slide or region.
    cells: u64,
    /// Cells with tissue: the rows of the table.
    tissue_cells: u64,
    /// Glass or no image data.
    glass_cells: u64,
    /// Not bright-field H&E (mostly near-black).
    dark_cells: u64,
    /// Predicted, but no block wholly on the slide (and region) is tissue.
    unmeasured_cells: u64,
    /// Left out because a window could not be read.
    failed_cells: u64,
    windows_read: u64,
    windows_predicted: u64,
    windows_glass: u64,
    windows_dark: u64,
    windows_failed: u64,
    windows_unusual: u64,
    blocks: u64,
    dark_blocks: u64,
    od_sum: f64,
    recon_sum: f64,
    recon_blocks: u64,
    unusual_blocks: u64,
}

/// One slide's scan, strip by strip.
struct Scan<'a> {
    host: &'a Host,
    run_id: &'a str,
    grid: &'a Grid,
    info: &'a ModelInfo,
    markers: &'a [Marker],
    channels: Channels,
    ramp: Ramp,
    region: Option<Region>,
    stop: &'a AtomicBool,
    max_cells: u64,
    batch: usize,
    csv: BufWriter<File>,
    counts: Counts,
    /// Per marker: mean, spread and density r² over tissue blocks, as a
    /// readout gives them, and their distribution.
    fits: Vec<Fit>,
    hists: Vec<Histogram>,
    first_error: Option<String>,
    started: Instant,
    last_progress: Instant,
}

/// Quotes a CSV field if it needs it.
fn csv_field(text: &str) -> String {
    if text.contains([',', '"', '\n', '\r']) {
        format!("\"{}\"", text.replace('"', "\"\""))
    } else {
        text.to_string()
    }
}

/// `("1.30.0", Some("libonnxruntime.so.1.30.0"))` from "ONNX Runtime 1.30.0
/// from /path/libonnxruntime.so.1.30.0": provenance without local paths.
fn runtime_parts(runtime: &str) -> (String, Option<String>) {
    match runtime
        .strip_prefix("ONNX Runtime ")
        .and_then(|rest| rest.split_once(" from "))
    {
        Some((version, library)) => (
            version.to_string(),
            Path::new(library)
                .file_name()
                .map(|name| name.to_string_lossy().into_owned()),
        ),
        None => (runtime.to_string(), None),
    }
}

impl Scan<'_> {
    fn cell_um(&self) -> f64 {
        self.grid.stride as f64 * self.info.mpp
    }

    fn header(&mut self) -> std::io::Result<()> {
        let model = self
            .info
            .title
            .clone()
            .unwrap_or_else(|| self.info.file_name());
        writeln!(
            self.csv,
            "# {RESEARCH_NOTICE}. Cellaris {}, model {} sha256:{}.",
            env!("CARGO_PKG_VERSION"),
            model.replace(['\n', '\r'], " "),
            self.info.sha256
        )?;
        writeln!(
            self.csv,
            "# One row per {:.1} \u{b5}m cell with tissue: x, y, width, height in level-0 pixels; \
             marker columns are mean predicted arcsinh(x/5) intensities over the cell's tissue \
             blocks of {:.1} \u{b5}m, od_mean their optical density.",
            self.cell_um(),
            BLOCK as f64 * self.info.mpp
        )?;
        let mut columns: Vec<String> = [
            "x",
            "y",
            "width",
            "height",
            "size_um",
            "tissue_blocks",
            "dark_blocks",
            "od_mean",
            "recon_error",
        ]
        .map(String::from)
        .to_vec();
        columns.extend(
            self.markers
                .iter()
                .map(|m| csv_field(&format!("{}_mean", m.name))),
        );
        writeln!(self.csv, "{}", columns.join(","))
    }

    /// Scans the cells `(i0, i1, j0, j1)`; returns whether it stopped at
    /// `max_cells`.
    fn run(
        &mut self,
        predictor: &mut dyn Predictor,
        (i0, i1, j0, j1): (i64, i64, i64, i64),
    ) -> Result<bool, String> {
        self.counts.cells = ((i1 - i0) * (j1 - j0)) as u64;
        // Window (wi, wj) covers cells wi..=wi+1, wj..=wj+1.
        let window_rows = (j0 - 1).max(0)..j1;
        let strips: Vec<(i64, i64)> = (i0..i1)
            .step_by(STRIP as usize)
            .map(|start| (start, (start + STRIP).min(i1)))
            .collect();
        let total: u64 = strips
            .iter()
            .map(|(start, end)| {
                ((end - (start - 1).max(0)) * (window_rows.end - window_rows.start)) as u64
            })
            .sum();
        let mut done = 0u64;
        let (grid, markers, info, channels) = (self.grid, self.markers, self.info, self.channels);
        for (start, end) in strips {
            let windows: Vec<Idx> = window_rows
                .clone()
                .flat_map(|wj| ((start - 1).max(0)..end).map(move |wi| Idx::new(wi, wj)))
                .collect();
            let in_strip = |cell: Idx| (start..end).contains(&cell.i) && (j0..j1).contains(&cell.j);
            let mut accums: HashMap<Idx, Accum> = HashMap::new();
            let mut finished: HashSet<Idx> = HashSet::new();
            for chunk in windows.chunks(self.batch.max(1)) {
                if self.stop.load(Ordering::SeqCst) {
                    return Err("stopped: eov is shutting down".into());
                }
                let mut input: Vec<f32> = Vec::new();
                let mut tissue: Vec<(Idx, resample::Analysis)> = Vec::new();
                for &window in chunk {
                    let image = match self.read(window) {
                        Ok(image) => image,
                        Err(error) => {
                            self.counts.windows_failed += 1;
                            self.first_error.get_or_insert(error);
                            self.drop_cells(
                                window,
                                &in_strip,
                                &mut accums,
                                &mut finished,
                                Ending::Failed,
                            );
                            continue;
                        }
                    };
                    self.counts.windows_read += 1;
                    let analysis = resample::analyse(&image);
                    match analysis.kind {
                        WindowKind::Background => {
                            self.counts.windows_glass += 1;
                            self.drop_cells(
                                window,
                                &in_strip,
                                &mut accums,
                                &mut finished,
                                Ending::Glass,
                            );
                        }
                        WindowKind::Dark => {
                            self.counts.windows_dark += 1;
                            self.drop_cells(
                                window,
                                &in_strip,
                                &mut accums,
                                &mut finished,
                                Ending::Dark,
                            );
                        }
                        WindowKind::Tissue => {
                            let side = grid.window;
                            let offset = input.len();
                            input.resize(offset + 3 * side * side, 0.0);
                            resample::model_input(&image, &mut input[offset..]);
                            tissue.push((window, analysis));
                        }
                    }
                }
                if !tissue.is_empty() {
                    let n = tissue.len();
                    let prediction =
                        catch_unwind(AssertUnwindSafe(|| predictor.predict(&input, n)))
                            .unwrap_or_else(|payload| {
                                Err(format!(
                                    "inference crashed: {}",
                                    panic_text(payload.as_ref())
                                ))
                            })
                            .map_err(|error| {
                                if self.stop.load(Ordering::SeqCst) {
                                    "stopped: eov is shutting down".to_string()
                                } else {
                                    format!("inference failed: {error}")
                                }
                            })?;
                    self.counts.windows_predicted += n as u64;
                    for (k, (window, analysis)) in tissue.into_iter().enumerate() {
                        let (planes, unusual) =
                            window_planes(&prediction, k, markers, info, channels, &analysis.mask);
                        if unusual {
                            self.counts.windows_unusual += 1;
                        }
                        for (cell, dx, dy) in grid.window_cells(window) {
                            if !in_strip(cell) || finished.contains(&cell) {
                                continue;
                            }
                            let accum = accums.entry(cell).or_insert_with(|| {
                                Accum::new(
                                    channels.count(),
                                    grid.stride,
                                    expected_bits(cell.i, cell.j),
                                )
                            });
                            accum.add(dx, dy, &self.ramp, &planes, grid.window, &analysis.blocks);
                            if accum.is_complete()
                                && let Some(accum) = accums.remove(&cell)
                            {
                                finished.insert(cell);
                                let (values, bases) = accum.finish(&self.ramp);
                                let data =
                                    finalize(&values, &bases, grid.stride, channels, markers);
                                self.cell(cell, &data)
                                    .map_err(|e| format!("could not write {CELLS_CSV}: {e}"))?;
                            }
                        }
                    }
                }
                // Windows go row by row, so only cells of the row being
                // finished and the next one are still blending.
                debug_assert!(
                    accums.len() as i64 <= 2 * (STRIP + 2),
                    "{} cells blending",
                    accums.len()
                );
                done += chunk.len() as u64;
                self.progress(done, total);
                if self.full() {
                    return Ok(true);
                }
            }
            // Every window of the strip was seen; nothing should be left.
            self.counts.failed_cells += accums.len() as u64;
        }
        Ok(false)
    }

    fn read(&self, window: Idx) -> Result<WindowImage, String> {
        let plan = self.grid.read_plan(window);
        let mut last = String::new();
        for _ in 0..READ_ATTEMPTS {
            match self
                .host
                .read_region(plan.request)
                .map_err(|e| format!("the slide could not be read: {e}"))
                .and_then(|image| resample::resample(&image, &plan, self.grid.window))
            {
                Ok(image) => return Ok(image),
                Err(error) => last = error,
            }
        }
        Err(last)
    }

    /// Ends the strip's cells of a window that is glass, dark or unreadable,
    /// as the overlay does.
    fn drop_cells(
        &mut self,
        window: Idx,
        in_strip: &dyn Fn(Idx) -> bool,
        accums: &mut HashMap<Idx, Accum>,
        finished: &mut HashSet<Idx>,
        why: Ending,
    ) {
        for (cell, _, _) in self.grid.window_cells(window) {
            if !in_strip(cell) || !finished.insert(cell) {
                continue;
            }
            accums.remove(&cell);
            match why {
                Ending::Glass => self.counts.glass_cells += 1,
                Ending::Dark => self.counts.dark_cells += 1,
                Ending::Failed => self.counts.failed_cells += 1,
            }
        }
    }

    fn full(&self) -> bool {
        self.max_cells > 0 && self.counts.tissue_cells >= self.max_cells
    }

    /// A finished cell: its tissue blocks into the summary, and a row.
    fn cell(&mut self, cell: Idx, data: &CellData) -> std::io::Result<()> {
        if self.full() {
            // Finished in the same run as the last one wanted.
            return Ok(());
        }
        let tissue = match data {
            CellData::Empty => {
                self.counts.glass_cells += 1;
                return Ok(());
            }
            CellData::Dark => {
                self.counts.dark_cells += 1;
                return Ok(());
            }
            CellData::Tissue(tissue) => tissue,
        };
        let per = self.grid.blocks_per_side();
        let threshold = self.info.recon_threshold;
        let (mut blocks, mut dark) = (0u64, 0u64);
        let (mut od, mut recon, mut recon_blocks) = (0f64, 0f64, 0u64);
        let mut sums = vec![0f64; self.markers.len()];
        for (index, block) in tissue.blocks.iter().enumerate() {
            let usage = block_use(block);
            if usage == BlockUse::Skip {
                continue;
            }
            if let Some(region) = &self.region {
                let (x, y) = self.grid.block_center(cell, index % per, index / per);
                if !region.contains(x, y) {
                    continue;
                }
            }
            if usage == BlockUse::Dark {
                dark += 1;
                continue;
            }
            blocks += 1;
            od += f64::from(block.od);
            for (k, mean) in block.means.iter().enumerate() {
                let mean = f64::from(*mean);
                sums[k] += mean;
                self.fits[k].add(f64::from(block.od), mean);
                self.hists[k].add(mean);
            }
            if let Some(value) = block.recon {
                recon += f64::from(value);
                recon_blocks += 1;
                if threshold.is_some_and(|t| value > t) {
                    self.counts.unusual_blocks += 1;
                }
            }
        }
        self.counts.dark_blocks += dark;
        self.counts.blocks += blocks;
        self.counts.od_sum += od;
        self.counts.recon_sum += recon;
        self.counts.recon_blocks += recon_blocks;
        if blocks == 0 {
            self.counts.unmeasured_cells += 1;
            return Ok(());
        }
        self.counts.tissue_cells += 1;
        let rect = self.grid.cell_rect(cell);
        let n = blocks as f64;
        let recon = if recon_blocks > 0 {
            format!("{:.4}", recon / recon_blocks as f64)
        } else {
            String::new()
        };
        write!(
            self.csv,
            "{:.1},{:.1},{:.2},{:.2},{:.2},{blocks},{dark},{:.4},{recon}",
            rect.x,
            rect.y,
            rect.width,
            rect.height,
            self.cell_um(),
            od / n
        )?;
        for sum in sums {
            write!(self.csv, ",{:.4}", sum / n)?;
        }
        writeln!(self.csv)
    }

    fn progress(&mut self, done: u64, total: u64) {
        if self.last_progress.elapsed() < PROGRESS_INTERVAL && done < total {
            return;
        }
        self.last_progress = Instant::now();
        let fraction = if total == 0 {
            1.0
        } else {
            done as f32 / total as f32
        };
        let elapsed = self.started.elapsed().as_secs_f64();
        let left = if done == 0 {
            String::new()
        } else {
            format!(
                ", {} left",
                present::duration(elapsed / done as f64 * total.saturating_sub(done) as f64)
            )
        };
        let _ = self.host.batch_progress(
            self.run_id,
            Some(fraction),
            Some(&format!(
                "{done} of {total} windows, {} cells with tissue{left}",
                self.counts.tissue_cells
            )),
        );
    }

    /// The slide's summary for the manifest, and a note when it is
    /// incomplete.
    fn summary(
        &self,
        file: &FileInfo,
        region: Option<Rect>,
        source: &str,
        stopped_early: bool,
    ) -> (Value, Option<String>) {
        let c = &self.counts;
        let info = self.info;
        let mut warnings = present::slide_notes(
            info,
            Some(self.grid),
            file,
            WindowCounts {
                dark: c.windows_dark as usize,
                tissue: c.windows_predicted as usize,
                unusual: c.windows_unusual as usize,
            },
        );
        warnings.extend(info.notes.iter().cloned());
        let mut incomplete = Vec::new();
        if c.windows_failed > 0 {
            incomplete.push(format!(
                "{} windows could not be read ({}); their {} cells are not in the table",
                c.windows_failed,
                self.first_error.as_deref().unwrap_or("no reason given"),
                c.failed_cells
            ));
        }
        if stopped_early {
            incomplete.push(format!(
                "stopped after {} cells with tissue (max_cells); the rest was not predicted",
                self.max_cells
            ));
        }
        if c.tissue_cells == 0 && !stopped_early {
            warnings.push("no tissue was found".into());
        }
        warnings.extend(incomplete.iter().cloned());
        let (version, library) = runtime_parts(&info.runtime);
        let percentiles = |hist: &Histogram| -> Value {
            let mut out = Map::new();
            for p in PERCENTILES {
                out.insert(format!("p{p}"), json!(hist.percentile(p)));
            }
            Value::Object(out)
        };
        let markers: Vec<Value> = self
            .markers
            .iter()
            .zip(&self.fits)
            .zip(&self.hists)
            .map(|((marker, fit), hist)| {
                json!({
                    "name": marker.name,
                    "channel": marker.channel,
                    "mean": (fit.blocks() > 0).then(|| fit.mean()),
                    "spread": (fit.blocks() > 0).then(|| fit.spread()),
                    "density_r2": fit.r2(),
                    "percentiles": percentiles(hist),
                })
            })
            .collect();
        let scale = &self.grid.scale;
        let summary = json!({
            "notice": RESEARCH_NOTICE,
            "plugin": format!("cellaris {}", env!("CARGO_PKG_VERSION")),
            "complete": incomplete.is_empty(),
            "values": if info.raw_intensity {
                "mean predicted arcsinh(x/5) intensity over tissue blocks"
            } else {
                "mean predicted intensity, as the model outputs it, over tissue blocks"
            },
            "model": {
                "name": info.title,
                "version": info.version,
                "file": info.file_name(),
                "sha256": info.sha256,
                "bytes": info.bytes,
                "mpp": info.mpp,
                "mpp_recorded": info.mpp_recorded,
                "window_px": info.window,
                "channels": info.channel_names,
                "raw_intensity": info.raw_intensity,
                "reconstruction_error": info.recon,
                "recon_error_threshold": info.recon_threshold,
                "training_data": info.training_data,
                "license": info.license,
            },
            "runtime": {
                "onnx_runtime": version,
                "library": library,
                "device": "cpu",
                "threads": info.threads,
                "batch_size": self.batch,
            },
            "slide": {
                "width": file.width,
                "height": file.height,
                "mpp_x": scale.mpp_x,
                "mpp_y": scale.mpp_y,
                "pixel_size": match scale.source {
                    ScaleSource::Reported => "reported",
                    ScaleSource::OneAxis => "one axis reported",
                    ScaleSource::Assumed => "assumed (assume_mpp)",
                },
                "level_read": self.grid.level,
                "downsample": self.grid.downsample,
                "upsampling": scale.upsampling(info.mpp),
            },
            "region": {
                "source": source,
                "rect": region.map(|r| json!({
                    "x": r.x, "y": r.y, "width": r.width, "height": r.height,
                })),
            },
            "grid": {
                "cell_um": self.cell_um(),
                "cell_px": [self.grid.cell_size().0, self.grid.cell_size().1],
                "block_um": BLOCK as f64 * info.mpp,
                "stride_px": self.grid.stride,
            },
            "cells": {
                "in_region": c.cells,
                "with_tissue": c.tissue_cells,
                "glass": c.glass_cells,
                "dark": c.dark_cells,
                "no_measurable_tissue": c.unmeasured_cells,
                "failed": c.failed_cells,
            },
            "windows": {
                "read": c.windows_read,
                "predicted": c.windows_predicted,
                "glass": c.windows_glass,
                "dark": c.windows_dark,
                "failed": c.windows_failed,
                "unusual": c.windows_unusual,
            },
            "tissue": {
                "blocks": c.blocks,
                "area_mm2": c.blocks as f64 * self.grid.block_area_um2() / 1e6,
                "dark_blocks": c.dark_blocks,
                "od_mean": (c.blocks > 0).then(|| c.od_sum / c.blocks as f64),
                "mean_recon_error": (c.recon_blocks > 0).then(|| c.recon_sum / c.recon_blocks as f64),
                "unusual_blocks": c.unusual_blocks,
            },
            "markers": markers,
            "warnings": warnings,
            "seconds": self.started.elapsed().as_secs_f64(),
        });
        let message = (!incomplete.is_empty()).then(|| incomplete.join("; "));
        (summary, message)
    }
}

/// Why a window's cells end without a prediction.
#[derive(Clone, Copy)]
enum Ending {
    Glass,
    Dark,
    Failed,
}

fn panic_text(payload: &(dyn std::any::Any + Send)) -> String {
    if let Some(text) = payload.downcast_ref::<&str>() {
        (*text).to_string()
    } else if let Some(text) = payload.downcast_ref::<String>() {
        text.clone()
    } else {
        "unknown panic".to_string()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn params(pairs: &[(&str, Value)]) -> Result<Params, String> {
        let map: Map<String, Value> = pairs
            .iter()
            .map(|(k, v)| (k.to_string(), v.clone()))
            .collect();
        Params::parse(&map, &Prefs::default())
    }

    #[test]
    fn parameters_come_in_the_forms_eov_passes_them() {
        let defaults = params(&[]).unwrap();
        assert_eq!(
            defaults.markers,
            Markers::Named(vec!["Hoechst".into(), "Pan-CK".into(), "SMA".into()])
        );
        assert!(!defaults.assume_mpp && defaults.region.is_none() && defaults.max_cells == 0);
        // `--param markers=Hoechst,CD8a`, `=all`, or a JSON list.
        assert_eq!(
            params(&[("markers", json!(" Hoechst, CD8a "))])
                .unwrap()
                .markers,
            Markers::Named(vec!["Hoechst".into(), "CD8a".into()])
        );
        assert_eq!(
            params(&[("markers", json!("ALL"))]).unwrap().markers,
            Markers::All
        );
        assert_eq!(
            params(&[("markers", json!(["SMA"]))]).unwrap().markers,
            Markers::Named(vec!["SMA".into()])
        );
        assert!(params(&[("markers", json!(""))]).is_err());
        assert!(params(&[("markers", json!(3))]).is_err());
        // Booleans as JSON or words.
        assert!(params(&[("assume_mpp", json!(true))]).unwrap().assume_mpp);
        assert!(params(&[("assume_mpp", json!("yes"))]).unwrap().assume_mpp);
        assert!(params(&[("assume_mpp", json!("maybe"))]).is_err());
        // Regions as text, a list or an object.
        let rect = Rect {
            x: 10.0,
            y: 20.0,
            width: 30.0,
            height: 40.0,
        };
        for value in [
            json!("10, 20, 30, 40"),
            json!([10, 20, 30, 40]),
            json!({"x": 10, "y": 20, "width": 30, "height": 40}),
        ] {
            assert_eq!(params(&[("region", value)]).unwrap().region, Some(rect));
        }
        for value in [json!("1,2,3"), json!("1,2,0,4"), json!("a,b,c,d")] {
            let error = params(&[("region", value)]).unwrap_err();
            assert!(error.contains("x,y,width,height"), "{error}");
        }
        assert!(params(&[("batch_size", json!(0))]).is_err());
        assert_eq!(params(&[("threads", json!(3))]).unwrap().threads, 3);
        assert_eq!(
            params(&[("model", json!("/m.onnx"))]).unwrap().model,
            Some(PathBuf::from("/m.onnx"))
        );
        // The command declares every parameter it reads, with defaults.
        let declared: Vec<String> = command().params.into_iter().map(|p| p.name).collect();
        for name in [
            "markers",
            "assume_mpp",
            "region",
            "max_cells",
            "model",
            "runtime",
            "threads",
            "batch_size",
        ] {
            assert!(declared.iter().any(|d| d == name), "{name} is not declared");
        }
    }

    #[test]
    fn percentiles_come_from_fixed_bins() {
        let mut hist = Histogram::default();
        assert_eq!(hist.percentile(50), None);
        for k in 0..100 {
            hist.add(5.0 + f64::from(k) * 0.01);
        }
        hist.add(f64::NAN);
        let p50 = hist.percentile(50).unwrap();
        assert!((p50 - 5.495).abs() < 0.011, "{p50}");
        assert!(hist.percentile(1).unwrap() < hist.percentile(99).unwrap());
        // Out of range values count at the ends.
        hist.add(100.0);
        assert!(hist.percentile(100).unwrap() > 11.9);
    }

    #[test]
    fn provenance_names_no_local_paths() {
        assert_eq!(
            runtime_parts("ONNX Runtime 1.30.0 from /home/me/lib/libonnxruntime.so.1.30.0"),
            (
                "1.30.0".to_string(),
                Some("libonnxruntime.so.1.30.0".to_string())
            )
        );
        assert_eq!(runtime_parts("none"), ("none".to_string(), None));
        assert_eq!(csv_field("Pan-CK_mean"), "Pan-CK_mean");
        assert_eq!(csv_field("a,\"b\""), "\"a,\"\"b\"\"\"");
    }
}
