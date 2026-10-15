//! The prediction engine: shared state, and the worker thread that plans,
//! reads, predicts, blends and publishes.
//!
//! The UI thread only records what the user did and wakes the worker; it
//! never reads pixels or runs the model. The worker:
//!
//! 1. reads the host's snapshot for every pane's view (CEL-7: each pane's
//!    demand counts, so split views do not starve each other);
//! 2. loads the model when one is needed and asked for (CEL-6: failures are
//!    shown and retried on request, never cached forever);
//! 3. plans the windows the views and readouts need, nearest each view's
//!    centre first, panes and readouts taking turns (CEL-8), skipping
//!    windows off the slide (CEL-7) and backing off failed ones (CEL-6);
//! 4. reads and resamples them to the model's pixel size (CEL-3), skips
//!    glass and non-bright-field windows, and predicts the rest in batches;
//! 5. blends finished cells (CEL-5) and uploads them as raster objects of an
//!    overlay layer, which the host draws off the UI thread on any backend
//!    (CEL-8).

use crate::blend::{Accum, CellData, Channels, Ramp, TissueCell, bit, expected_bits, finalize};
use crate::cache::{self, Shelf};
use crate::config::Prefs;
use crate::geometry::{Grid, Idx, ReadPlan, Scale, ScaleProblem};
use crate::markers::{self, Marker};
use crate::model::{Backend, LoadError, LoadRequest, ModelInfo, Predictor, Stopper};
use crate::present;
use crate::region::{Readout, ReadoutState, Region};
use crate::resample::{self, BlockBase, WindowKind};
use eov_plugin_api::prelude::*;
use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet};
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Condvar, Mutex, MutexGuard};
use std::time::{Duration, Instant};

/// A pane needing more cells than this is "zoomed too far out" and computes
/// nothing (the model is only run at its own pixel size).
pub const MAX_VIEW_CELLS: u64 = 600;
/// Finished cells kept in memory, and as rasters in the host.
pub const CELL_BUDGET: usize = 1500;
/// Cells waiting for the rest of their windows.
pub const ACCUM_BUDGET: usize = 320;
/// A region needing more windows than this asks before it runs.
pub const READOUT_CONFIRM_WINDOWS: usize = 600;
/// More than this is refused.
pub const READOUT_MAX_WINDOWS: usize = 6000;
/// Failed windows are retried this many times, backing off, then left
/// until the user asks again.
pub const MAX_ATTEMPTS: u32 = 5;
/// Disk cells loaded per pass.
const DISK_LOADS: usize = 256;
/// Displays re-rendered per pass after a display change.
const RERENDER: usize = 200;
/// How often the panel and status follow progress.
const UI_INTERVAL: Duration = Duration::from_millis(250);
/// The worker looks at the host this often even when nothing happens.
const IDLE_POLL: Duration = Duration::from_secs(3);
/// An idle worker never waits less than this, whatever the bookkeeping says.
const MIN_WAIT: Duration = Duration::from_millis(50);
/// Seconds per window assumed before anything has been measured.
pub const DEFAULT_SECONDS_PER_WINDOW: f64 = 1.2;
/// Cells' value tooltips are offered once a cell is this many screen
/// pixels wide; below that, a pointer covers several cells.
pub const TOOLTIP_CELL_PX: f64 = 24.0;

pub fn pred_layer(file: FileId) -> String {
    format!("cellaris-predicted-{file}")
}

pub fn marks_layer(file: FileId) -> String {
    format!("cellaris-marks-{file}")
}

pub fn region_layer(file: FileId) -> String {
    format!("cellaris-region-{file}")
}

/// Transparent cells carrying the predicted values as tooltips.
pub fn values_layer(file: FileId) -> String {
    format!("cellaris-values-{file}")
}

pub fn cell_object(cell: Idx) -> String {
    format!("c{}_{}", cell.i, cell.j)
}

/// The plugin's directories.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Paths {
    pub plugin_root: PathBuf,
    pub data_dir: PathBuf,
    pub cache_dir: PathBuf,
}

/// Where the model stands.
#[derive(Clone, Debug, Default)]
pub enum ModelState {
    /// Not loaded yet (nothing has needed it).
    #[default]
    Idle,
    Loading,
    Ready(Arc<ModelInfo>),
    Failed(LoadError),
}

impl ModelState {
    pub fn info(&self) -> Option<&Arc<ModelInfo>> {
        match self {
            ModelState::Ready(info) => Some(info),
            _ => None,
        }
    }
}

/// A window that failed, and when it may be tried again.
#[derive(Clone, Debug, PartialEq)]
pub struct Failure {
    pub attempts: u32,
    pub retry_at: Instant,
    pub error: String,
}

impl Failure {
    pub fn given_up(&self) -> bool {
        self.attempts >= MAX_ATTEMPTS
    }

    pub fn blocks(&self, now: Instant) -> bool {
        self.given_up() || now < self.retry_at
    }
}

/// What a pane's view needs.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PaneDemand {
    /// Computing, or all done (the slide counts its cells in view).
    Active,
    /// Zoomed out past [`MAX_VIEW_CELLS`].
    TooFar { cells: u64 },
    /// The slide cannot be predicted (see the file's problem).
    Blocked,
}

/// A readout with its host job.
pub struct ReadoutJob {
    pub readout: Readout,
    pub job: Option<Job>,
    /// The confirmation asked for a long run.
    pub confirm: Option<String>,
    pub started: Instant,
}

/// One open slide.
pub struct FileState {
    pub info: FileInfo,
    pub layers: bool,
    pub grid: Option<Result<Arc<Grid>, ScaleProblem>>,
    pub cells: HashMap<Idx, (CellData, u64)>,
    pub accums: HashMap<Idx, Accum>,
    pub failures: HashMap<Idx, Failure>,
    pub inflight: HashSet<Idx>,
    /// Cells with a raster in the host.
    pub uploaded: HashSet<Idx>,
    /// Uploaded cells to draw again after a display change.
    pub stale: BTreeSet<Idx>,
    /// Cells known not to be on disk.
    pub disk_misses: HashSet<Idx>,
    pub shelf: Option<Shelf>,
    pub region: Option<Region>,
    /// The region is the slide's region of interest, and follows it.
    pub region_from_roi: bool,
    /// A readout waiting to be started (the region changed).
    pub readout_wanted: bool,
    pub readout: Option<ReadoutJob>,
    pub panes: BTreeMap<PaneId, PaneDemand>,
    /// Cells in any pane's view (each counted once), those still being
    /// computed, and those whose windows failed.
    pub in_view: usize,
    pub pending: Vec<Idx>,
    pub failed: Vec<Idx>,
    /// Failed cells that wait on a window given up: they stay failed until
    /// the user retries.
    pub stuck: usize,
    pub dark_windows: usize,
    pub tissue_windows: usize,
    /// Windows above the model's reconstruction-error threshold.
    pub unusual_windows: usize,
    /// Fingerprints of what the marks and region layers last showed.
    pub marks_print: Option<u64>,
    pub region_print: Option<u64>,
    /// The minimum zoom given to the values layer, once the grid is known.
    pub values_zoom: Option<f64>,
}

impl FileState {
    pub fn new(info: FileInfo) -> Self {
        Self {
            info,
            layers: false,
            grid: None,
            cells: HashMap::new(),
            accums: HashMap::new(),
            failures: HashMap::new(),
            inflight: HashSet::new(),
            uploaded: HashSet::new(),
            stale: BTreeSet::new(),
            disk_misses: HashSet::new(),
            shelf: None,
            region: None,
            region_from_roi: false,
            readout_wanted: false,
            readout: None,
            panes: BTreeMap::new(),
            in_view: 0,
            pending: Vec::new(),
            failed: Vec::new(),
            stuck: 0,
            dark_windows: 0,
            tissue_windows: 0,
            unusual_windows: 0,
            marks_print: None,
            region_print: None,
            values_zoom: None,
        }
    }

    /// Forgets every prediction (the model, markers or grid changed).
    /// Returns the cells whose rasters must go.
    pub fn reset(&mut self) -> Vec<Idx> {
        self.grid = None;
        self.cells.clear();
        self.accums.clear();
        self.failures.clear();
        self.inflight.clear();
        self.stale.clear();
        self.disk_misses.clear();
        self.shelf = None;
        self.in_view = 0;
        self.pending.clear();
        self.failed.clear();
        self.stuck = 0;
        self.values_zoom = None;
        self.dark_windows = 0;
        self.tissue_windows = 0;
        self.unusual_windows = 0;
        if self.region.is_some() {
            self.readout_wanted = true;
        }
        self.uploaded.drain().collect()
    }

    pub fn grid(&self) -> Option<&Arc<Grid>> {
        self.grid.as_ref().and_then(|g| g.as_ref().ok())
    }

    pub fn problem(&self) -> Option<&ScaleProblem> {
        self.grid.as_ref().and_then(|g| g.as_ref().err())
    }
}

/// How fast windows go.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Throughput {
    pub windows: u64,
    pub seconds: f64,
}

impl Throughput {
    pub fn seconds_per_window(&self) -> f64 {
        if self.windows == 0 {
            DEFAULT_SECONDS_PER_WINDOW
        } else {
            self.seconds / self.windows as f64
        }
    }
}

/// Everything the UI thread and the worker share.
pub struct State {
    pub prefs: Prefs,
    pub overlay: bool,
    pub files: BTreeMap<FileId, FileState>,
    pub views: BTreeMap<PaneId, ViewInfo>,
    pub active_file: Option<FileId>,
    /// The pane the user is working in.
    pub focused_pane: Option<PaneId>,
    pub model: ModelState,
    /// The resolved markers, or why they could not be.
    pub markers: Result<Vec<Marker>, String>,
    /// Load (or reload) the model at the next chance.
    pub load_wanted: bool,
    /// Drop the loaded model first (its file or settings changed).
    pub reload: bool,
    /// Bumped whenever cached predictions stop being valid.
    pub generation: u64,
    pub tick: u64,
    pub sidebar: Option<Surface>,
    pub sidebar_wanted: bool,
    pub readout_seq: u64,
    pub throughput: Throughput,
    /// Problems the user was already told about.
    pub notified: HashSet<String>,
    /// Problems with the settings, shown in the panel.
    pub setting_problems: Vec<String>,
    pub woken: bool,
    pub busy: bool,
    /// The runtime chosen after one was already loaded.
    pub runtime_restart: bool,
    pub paths: Paths,
    /// What the status bar and the panel last showed.
    pub last_status: Option<Option<String>>,
    pub last_sidebar: Option<Map<String, Value>>,
    pub last_panel_open: Option<bool>,
    /// Whether the host shows badges; hosts older than the badge method
    /// get the notice drawn on the slide instead.
    pub badges: Support,
    /// The badges shown: id → (slide, text).
    pub shown_badges: BTreeMap<String, (Option<FileId>, String)>,
    /// How each pane displays its slide, as far as the host says.
    pub displays: BTreeMap<PaneId, DisplayAdjustments>,
    /// Whether the host answers `view.display`.
    pub displays_known: Support,
    /// Whether the host shows the tooltips of non-interactive layers. Hosts
    /// without `layer.set_min_zoom` predate them, and get no values layer.
    pub tooltips: Support,
}

/// Whether the host has a method that came after API 1.0's first release.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum Support {
    /// Not tried yet.
    #[default]
    Unknown,
    Yes,
    No,
}

impl State {
    pub fn new(prefs: Prefs, paths: Paths) -> Self {
        Self {
            prefs,
            overlay: false,
            files: BTreeMap::new(),
            views: BTreeMap::new(),
            active_file: None,
            focused_pane: None,
            model: ModelState::Idle,
            markers: Err("the model is not loaded".to_string()),
            load_wanted: false,
            reload: false,
            generation: 0,
            tick: 0,
            sidebar: None,
            sidebar_wanted: false,
            readout_seq: 0,
            throughput: Throughput::default(),
            notified: HashSet::new(),
            setting_problems: Vec::new(),
            woken: false,
            busy: false,
            runtime_restart: false,
            paths,
            last_status: None,
            last_sidebar: None,
            last_panel_open: None,
            badges: Support::Unknown,
            shown_badges: BTreeMap::new(),
            displays: BTreeMap::new(),
            displays_known: Support::Unknown,
            tooltips: Support::Unknown,
        }
    }

    /// Whether the model is needed by anything right now.
    pub fn needs_model(&self) -> bool {
        self.overlay
            || self.files.values().any(|f| {
                f.readout_wanted || f.readout.as_ref().is_some_and(|r| r.readout.is_running())
            })
    }

    /// Invalidates every prediction; returns rasters to remove per file.
    pub fn invalidate(&mut self) -> Vec<(FileId, Vec<Idx>)> {
        self.generation += 1;
        self.files
            .iter_mut()
            .map(|(id, file)| (*id, file.reset()))
            .collect()
    }

    /// The markers as the overlay draws them.
    pub fn display(&self) -> Option<Display> {
        let markers = self.markers.as_ref().ok()?;
        Some(Display {
            colours: markers.iter().map(|m| m.colour).collect(),
            visible: markers
                .iter()
                .map(|m| self.prefs.is_visible(&m.name))
                .collect(),
            mode: self.prefs.mode,
        })
    }
}

/// How cells are drawn.
#[derive(Clone, Debug, PartialEq)]
pub struct Display {
    pub colours: Vec<[u8; 3]>,
    pub visible: Vec<bool>,
    pub mode: markers::DisplayMode,
}

impl Display {
    pub fn render(&self, cell: &crate::blend::TissueCell) -> Vec<u8> {
        let mut out = Vec::new();
        markers::composite(
            &cell.planes,
            &cell.mask,
            &self.colours,
            &self.visible,
            self.mode,
            &mut out,
        );
        out
    }
}

/// The state, the worker's wake-up, and its stop flag.
pub struct Shared {
    state: Mutex<State>,
    wake: Condvar,
    pub stop: AtomicBool,
    /// Interrupts a running prediction (shutdown).
    pub stopper: Mutex<Option<Stopper>>,
}

impl Shared {
    pub fn new(state: State) -> Self {
        Self {
            state: Mutex::new(state),
            wake: Condvar::new(),
            stop: AtomicBool::new(false),
            stopper: Mutex::new(None),
        }
    }

    /// The state; a panic elsewhere must not wedge the plugin.
    pub fn lock(&self) -> MutexGuard<'_, State> {
        self.state.lock().unwrap_or_else(|e| e.into_inner())
    }

    /// Wakes the worker.
    pub fn wake(&self) {
        self.lock().woken = true;
        self.wake.notify_all();
    }

    /// Stops the worker and interrupts a running prediction.
    pub fn shut_down(&self) {
        self.stop.store(true, Ordering::SeqCst);
        if let Some(stop) = self
            .stopper
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .as_ref()
        {
            stop();
        }
        self.wake();
    }

    fn wait(&self, timeout: Duration) {
        let mut state = self.lock();
        let deadline = Instant::now() + timeout;
        while !state.woken && !self.stop.load(Ordering::SeqCst) {
            let now = Instant::now();
            if now >= deadline {
                break;
            }
            state = self
                .wake
                .wait_timeout(state, deadline - now)
                .map(|(guard, _)| guard)
                .unwrap_or_else(|e| e.into_inner().0);
        }
        state.woken = false;
    }
}

// ---------------------------------------------------------------------------
// Planning
// ---------------------------------------------------------------------------

#[derive(Clone, Debug)]
struct PlannedWindow {
    file: FileId,
    window: Idx,
    plan: ReadPlan,
    grid: Arc<Grid>,
}

#[derive(Default)]
struct Plan {
    generation: u64,
    windows: Vec<PlannedWindow>,
    disk: Vec<(FileId, Idx, Shelf)>,
    batch: usize,
}

/// The cells a source wants, in priority order.
struct Demand {
    file: FileId,
    cells: Vec<Idx>,
}

/// Retry delay after `attempts` failures: 2, 4, 8, 16, 30 s.
pub fn backoff(attempts: u32) -> Duration {
    Duration::from_secs((1u64 << attempts.min(5)).min(30))
}

fn scale_for(info: &FileInfo, model: &ModelInfo, prefs: &Prefs) -> Result<Arc<Grid>, ScaleProblem> {
    let scale = Scale::of(info, model.mpp, prefs.assume_mpp)?;
    Grid::new(info, scale, model.mpp, model.window).map(Arc::new)
}

/// Maps every slide onto the model grid, and opens its disk shelf once the
/// slide's identity is known (the host computes it after opening).
fn ensure_grids(st: &mut State, model: &ModelInfo, markers: &[Marker]) {
    let disk_on = st.prefs.disk_cache_mb > 0;
    for file in st.files.values_mut() {
        if file.grid.is_none() {
            file.grid = Some(scale_for(&file.info, model, &st.prefs));
        }
        if file.shelf.is_none()
            && disk_on
            && let (Some(grid), Some(identity)) = (file.grid(), &file.info.identity)
        {
            file.shelf = Some(Shelf::new(
                &st.paths.cache_dir.join("cells"),
                &cache::key(identity, &model.sha256, grid, markers, model.recon),
                grid.stride,
                markers.len(),
                model.recon,
            ));
            file.disk_misses.clear();
        }
    }
}

/// The windows a cell still needs.
fn missing_windows(file: &FileState, grid: &Grid, cell: Idx) -> Vec<Idx> {
    let accum = file.accums.get(&cell);
    grid.covering(cell)
        .filter(|(_, dx, dy)| accum.is_none_or(|a| !a.has(bit(*dx, *dy))))
        .map(|(window, _, _)| window)
        .collect()
}

// ---------------------------------------------------------------------------
// Results of a pass
// ---------------------------------------------------------------------------

enum Outcome {
    Background,
    Dark,
    Tissue {
        /// Markers (arcsinh), mask, reconstruction error: `channels × S²`.
        planes: Vec<f32>,
        blocks: Vec<BlockBase>,
        unusual: bool,
    },
    Failed(String),
}

struct WindowResult {
    file: FileId,
    window: Idx,
    grid: Arc<Grid>,
    outcome: Outcome,
}

/// A cell to draw and upload.
pub struct Render {
    file: FileId,
    cell: Idx,
    /// Its level-0 rectangle.
    rect: Rect,
    tissue: Arc<TissueCell>,
    /// A new cell, whose value tooltip goes up too (a redraw after a
    /// display change only changes its colours).
    new: bool,
}

/// What a pass hands to the host.
#[derive(Default)]
pub struct Publish {
    /// The generation the cells were computed for.
    generation: u64,
    /// How cells are drawn, as it was when they were queued.
    display: Option<Display>,
    /// Raster side in model pixels.
    side: u32,
    /// Whether cells get value tooltips, with the markers' names and the
    /// model's reconstruction-error threshold for them.
    tooltips: bool,
    names: Vec<String>,
    threshold: Option<f32>,
    renders: Vec<Render>,
    removals: Vec<(FileId, Vec<Idx>)>,
    writes: Vec<(Shelf, Idx, CellData)>,
    /// Values layers to limit to zooms where a cell is big enough to point at.
    min_zooms: Vec<(FileId, f64)>,
}

impl Publish {
    /// Notes how the state draws cells now, for the cells queued with it.
    fn follow(&mut self, st: &State) {
        self.display = st.display();
        self.side = st.model.info().map_or(0, |m| (m.window / 2) as u32);
        self.names = st
            .markers
            .as_ref()
            .map(|markers| markers.iter().map(|m| m.name.clone()).collect())
            .unwrap_or_default();
        self.threshold = st.model.info().and_then(|m| m.recon_threshold);
        self.tooltips = st.tooltips != Support::No;
    }
}

/// Removes cells from the host: their rasters, and their value tooltips
/// when the host has them.
pub fn remove_rasters(host: &Host, removals: Vec<(FileId, Vec<Idx>)>, tooltips: bool) {
    for (file, cells) in removals {
        remove_cells(host, file, cells, tooltips);
    }
}

/// Removes one slide's cells from the host.
pub fn remove_cells(host: &Host, file: FileId, cells: Vec<Idx>, tooltips: bool) {
    if cells.is_empty() {
        return;
    }
    let ids: Vec<String> = cells.into_iter().map(cell_object).collect();
    if tooltips {
        let _ = host.layer(values_layer(file)).remove(ids.clone());
    }
    let _ = host.layer(pred_layer(file)).remove(ids);
}

// ---------------------------------------------------------------------------
// The worker
// ---------------------------------------------------------------------------

pub struct Worker<B: Backend> {
    host: Host,
    shared: Arc<Shared>,
    backend: Arc<B>,
    predictor: Option<Box<dyn Predictor>>,
    last_ui: Instant,
    writes: usize,
}

impl<B: Backend> Worker<B> {
    pub fn new(host: Host, shared: Arc<Shared>, backend: Arc<B>) -> Self {
        Self {
            host,
            shared,
            backend,
            predictor: None,
            last_ui: Instant::now() - UI_INTERVAL,
            writes: 0,
        }
    }

    fn stopping(&self) -> bool {
        self.shared.stop.load(Ordering::SeqCst)
    }

    /// The worker loop. Every pass is guarded: a panic is reported and the
    /// worker carries on (CEL-6).
    pub fn run(mut self) {
        self.prune_disk();
        while !self.stopping() {
            let busy = match catch_unwind(AssertUnwindSafe(|| self.pass())) {
                Ok(busy) => busy,
                Err(payload) => {
                    let message = panic_text(payload.as_ref());
                    self.predictor = None;
                    self.host.notify_error(
                        "Cellaris hit an internal error and reset its model",
                        message.clone(),
                    );
                    let mut st = self.shared.lock();
                    st.model = ModelState::Failed(LoadError::new(
                        "Cellaris hit an internal error",
                        message,
                    ));
                    for file in st.files.values_mut() {
                        file.inflight.clear();
                    }
                    false
                }
            };
            if !busy {
                let timeout = self
                    .next_deadline()
                    .unwrap_or(IDLE_POLL)
                    .clamp(MIN_WAIT, IDLE_POLL);
                self.shared.wait(timeout);
            }
        }
        self.publish_ui(true);
    }

    /// One pass; returns whether there may be more to do right away.
    fn pass(&mut self) -> bool {
        self.sync_with_host();
        if self.stopping() {
            return false;
        }
        if self.load_if_wanted() {
            return true;
        }
        self.start_readouts();
        let plan = self.plan();
        let mut publish = Publish {
            generation: plan.generation,
            ..Default::default()
        };
        let mut busy = false;
        if !plan.disk.is_empty() {
            busy = true;
            let loaded: Vec<(FileId, Idx, Option<CellData>)> = plan
                .disk
                .iter()
                .map(|(file, cell, shelf)| (*file, *cell, shelf.load(*cell)))
                .collect();
            self.integrate_disk(plan.generation, loaded, &mut publish);
        }
        if !plan.windows.is_empty() {
            busy = true;
            let results = self.compute(&plan);
            self.integrate(plan.generation, results, &mut publish);
        }
        if self.rerender(&mut publish) {
            busy = true;
        }
        self.publish(publish);
        self.publish_ui(!busy);
        {
            let mut st = self.shared.lock();
            st.busy = busy;
        }
        busy
    }

    /// The earliest moment a failed window may be retried.
    /// The next retry still to come. A retry that is due and wanted is
    /// planned in the same pass (which is then busy); one nobody wants any
    /// more must not wake the worker, or it would spin.
    fn next_deadline(&self) -> Option<Duration> {
        let st = self.shared.lock();
        let now = Instant::now();
        st.files
            .values()
            .flat_map(|f| f.failures.values())
            .filter(|f| !f.given_up() && f.retry_at > now)
            .map(|f| f.retry_at - now)
            .min()
    }

    // -- the host's view of things ------------------------------------------

    /// Views of every pane, and files opened or closed without an event.
    fn sync_with_host(&mut self) {
        let Ok(snapshot) = self.host.snapshot() else {
            return;
        };
        let mut st = self.shared.lock();
        st.views = snapshot
            .panes
            .iter()
            .filter_map(|pane| pane.view.map(|view| (pane.pane, view)))
            .collect();
        st.focused_pane = snapshot.focused_pane;
        st.active_file = snapshot
            .focused_pane
            .and_then(|pane| snapshot.panes.iter().find(|p| p.pane == pane))
            .and_then(|pane| pane.file_id)
            .or(st.active_file.filter(|id| snapshot.file(*id).is_some()));
        let open: BTreeSet<FileId> = snapshot.files.iter().map(|f| f.file_id).collect();
        let closed: Vec<FileId> = st
            .files
            .keys()
            .filter(|id| !open.contains(id))
            .copied()
            .collect();
        for id in closed {
            if let Some(file) = st.files.remove(&id)
                && let Some(job) = file.readout
                && let Some(host_job) = job.job
            {
                host_job.finish(Some("the slide was closed".into()));
            }
        }
        for info in &snapshot.files {
            match st.files.get_mut(&info.file_id) {
                Some(file) if file.info != *info => file.info = info.clone(),
                Some(_) => {}
                None => {
                    st.files.insert(info.file_id, FileState::new(info.clone()));
                }
            }
        }
        let created = ensure_layers(&self.host, &mut st);
        let state = &mut *st;
        let views = &state.views;
        state.displays.retain(|pane, _| views.contains_key(pane));
        let unknown: Vec<PaneId> = if state.displays_known == Support::No {
            Vec::new()
        } else {
            views
                .keys()
                .filter(|pane| !state.displays.contains_key(pane))
                .copied()
                .collect()
        };
        drop(st);
        for file in created {
            self.refresh_file(file);
        }
        // Later changes arrive as `DisplayChanged`; a pane seen for the
        // first time is asked once.
        for pane in unknown {
            match self.host.display_adjustments(Some(pane)) {
                Ok(display) => {
                    let mut st = self.shared.lock();
                    st.displays_known = Support::Yes;
                    st.displays.entry(pane).or_insert(display);
                }
                Err(error) if error.is_unsupported() => {
                    self.shared.lock().displays_known = Support::No;
                    break;
                }
                Err(_) => {}
            }
        }
    }

    /// Loads the model when something needs it and loading was asked for.
    fn load_if_wanted(&mut self) -> bool {
        let request = {
            let mut guard = self.shared.lock();
            let st = &mut *guard;
            if st.reload {
                st.reload = false;
                self.predictor = None;
                st.model = ModelState::Idle;
                st.markers = Err("the model is not loaded".to_string());
                let tooltips = st.tooltips != Support::No;
                remove_rasters(&self.host, st.invalidate(), tooltips);
            }
            if self.predictor.is_some() || !st.load_wanted || !st.needs_model() {
                return false;
            }
            st.load_wanted = false;
            st.model = ModelState::Loading;
            LoadRequest {
                model: st.prefs.model_path.clone().map(PathBuf::from),
                runtime: st.prefs.runtime_path.clone().map(PathBuf::from),
                plugin_root: st.paths.plugin_root.clone(),
                data_dir: st.paths.data_dir.clone(),
                threads: if st.prefs.threads == 0 {
                    crate::model::default_threads()
                } else {
                    st.prefs.threads
                },
                ..Default::default()
            }
        };
        self.publish_ui(true);
        let started = Instant::now();
        let loaded = catch_unwind(AssertUnwindSafe(|| self.backend.load(&request))).unwrap_or_else(
            |payload| {
                Err(LoadError::new(
                    "Loading the model crashed",
                    panic_text(payload.as_ref()),
                ))
            },
        );
        let mut guard = self.shared.lock();
        let st = &mut *guard;
        match loaded {
            Ok(predictor) => {
                let info = Arc::new(predictor.info().clone());
                self.host.info(format!(
                    "Loaded {} ({} channels, {:.3} \u{b5}m/px, sha256 {}) with {} in {:.1} s",
                    info.path.display(),
                    info.channel_names.len(),
                    info.mpp,
                    info.short_hash(),
                    info.runtime,
                    started.elapsed().as_secs_f64()
                ));
                *self
                    .shared
                    .stopper
                    .lock()
                    .unwrap_or_else(|e| e.into_inner()) = predictor.stopper();
                st.markers = markers::select(&info.channel_names, &st.prefs.markers);
                if let Err(problem) = st.markers.clone() {
                    notify_once(
                        &self.host,
                        st,
                        "Cellaris cannot show the chosen markers",
                        &problem,
                    );
                }
                st.model = ModelState::Ready(info);
                self.predictor = Some(predictor);
                let tooltips = st.tooltips != Support::No;
                remove_rasters(&self.host, st.invalidate(), tooltips);
            }
            Err(error) => {
                self.host
                    .error(format!("Cellaris: {}: {}", error.summary, error.detail));
                let key = format!("{}\n{}", error.summary, error.detail);
                if st.notified.insert(key) {
                    self.host
                        .notify_error(format!("Cellaris: {}", error.summary), error.detail.clone());
                }
                st.model = ModelState::Failed(error.clone());
                for file in st.files.values_mut() {
                    if file.readout_wanted {
                        file.readout_wanted = false;
                        file.readout = Some(ReadoutJob {
                            readout: failed_readout(
                                &mut st.readout_seq,
                                file.region.clone(),
                                format!("The model did not load: {}", error.summary),
                            ),
                            job: None,
                            confirm: None,
                            started: Instant::now(),
                        });
                    }
                }
            }
        }
        true
    }

    /// Picks the next windows, and the cells to look for on disk.
    fn plan(&self) -> Plan {
        let mut guard = self.shared.lock();
        let st = &mut *guard;
        st.tick += 1;
        let tick = st.tick;
        let now = Instant::now();
        let Some(model) = st.model.info().cloned() else {
            return Plan::default();
        };
        let Ok(markers) = st.markers.clone() else {
            return Plan::default();
        };
        let batch = model.batch.size(st.prefs.batch_size);
        ensure_grids(st, &model, &markers);

        // Demand: every pane's view, then readouts.
        let mut demands: Vec<Demand> = Vec::new();
        for file in st.files.values_mut() {
            file.panes.clear();
            file.in_view = 0;
            file.pending.clear();
            file.failed.clear();
            file.stuck = 0;
        }
        let views: Vec<ViewInfo> = st.views.values().copied().collect();
        let mut seen: HashSet<(FileId, Idx)> = HashSet::new();
        let mut in_view: HashSet<(FileId, Idx)> = HashSet::new();
        if st.overlay {
            for view in views {
                let Some(file) = st.files.get_mut(&view.file_id) else {
                    continue;
                };
                let Some(grid) = file.grid().cloned() else {
                    file.panes.insert(view.pane, PaneDemand::Blocked);
                    continue;
                };
                let count = grid.cell_count(&view.visible);
                if count > MAX_VIEW_CELLS {
                    file.panes
                        .insert(view.pane, PaneDemand::TooFar { cells: count });
                    continue;
                }
                let cells = grid.cells_by_distance(&view.visible, view.center_x, view.center_y);
                file.panes.insert(view.pane, PaneDemand::Active);
                for cell in &cells {
                    if in_view.insert((view.file_id, *cell)) {
                        file.in_view += 1;
                    }
                }
                for cell in &cells {
                    if !file.cells.contains_key(cell) && seen.insert((view.file_id, *cell)) {
                        file.pending.push(*cell);
                        let blocking: Vec<&Failure> = missing_windows(file, &grid, *cell)
                            .iter()
                            .filter_map(|w| file.failures.get(w))
                            .filter(|f| f.blocks(now))
                            .collect();
                        if !blocking.is_empty() {
                            let stuck = blocking.iter().any(|f| f.given_up());
                            file.failed.push(*cell);
                            if stuck {
                                file.stuck += 1;
                            }
                        }
                    }
                }
                demands.push(Demand {
                    file: view.file_id,
                    cells,
                });
            }
        }
        for (id, file) in st.files.iter() {
            if let Some(job) = &file.readout
                && job.readout.is_running()
            {
                demands.push(Demand {
                    file: *id,
                    cells: job
                        .readout
                        .cells
                        .iter()
                        .filter(|c| job.readout.remaining.contains(c))
                        .copied()
                        .collect(),
                });
            }
        }

        // Keep what is wanted fresh for eviction.
        for demand in &demands {
            if let Some(file) = st.files.get_mut(&demand.file) {
                for cell in &demand.cells {
                    if let Some(entry) = file.cells.get_mut(cell) {
                        entry.1 = tick;
                    }
                    if let Some(accum) = file.accums.get_mut(cell) {
                        accum.last_used = tick;
                    }
                }
            }
        }

        // Sources take turns, nearest first.
        let mut plan = Plan {
            generation: st.generation,
            batch,
            ..Default::default()
        };
        let mut chosen: HashSet<(FileId, Idx)> = HashSet::new();
        let longest = demands.iter().map(|d| d.cells.len()).max().unwrap_or(0);
        'outer: for position in 0..longest {
            for demand in &demands {
                let Some(cell) = demand.cells.get(position) else {
                    continue;
                };
                let Some(file) = st.files.get_mut(&demand.file) else {
                    continue;
                };
                if file.cells.contains_key(cell) {
                    continue;
                }
                let Some(grid) = file.grid().cloned() else {
                    continue;
                };
                if let Some(shelf) = &file.shelf
                    && !file.disk_misses.contains(cell)
                    && !file.accums.contains_key(cell)
                {
                    if plan.disk.len() < DISK_LOADS
                        && !plan
                            .disk
                            .iter()
                            .any(|(f, c, _)| *f == demand.file && c == cell)
                    {
                        plan.disk.push((demand.file, *cell, shelf.clone()));
                    }
                    continue;
                }
                for window in missing_windows(file, &grid, *cell) {
                    if plan.windows.len() >= batch {
                        break 'outer;
                    }
                    if file.inflight.contains(&window)
                        || file.failures.get(&window).is_some_and(|f| f.blocks(now))
                        || !chosen.insert((demand.file, window))
                    {
                        continue;
                    }
                    file.inflight.insert(window);
                    plan.windows.push(PlannedWindow {
                        file: demand.file,
                        window,
                        plan: grid.read_plan(window),
                        grid: Arc::clone(&grid),
                    });
                }
            }
        }

        // Readouts whose remaining cells can only wait on failed windows.
        for file in st.files.values_mut() {
            let Some(grid) = file.grid().cloned() else {
                continue;
            };
            let Some(job) = file.readout.as_mut() else {
                continue;
            };
            if !job.readout.is_running() {
                continue;
            }
            let blocked = job.readout.remaining.iter().all(|cell| {
                missing_windows_of(&file.accums, &grid, *cell)
                    .iter()
                    .any(|w| file.failures.get(w).is_some_and(Failure::given_up))
            });
            if blocked && !job.readout.remaining.is_empty() {
                let error = file
                    .failures
                    .values()
                    .find(|f| f.given_up())
                    .map(|f| f.error.clone())
                    .unwrap_or_default();
                let missing = job.readout.remaining.len();
                job.readout.state = ReadoutState::Failed(format!(
                    "{missing} of {} cells could not be computed: {error}",
                    job.readout.cells.len()
                ));
                if let Some(host_job) = job.job.take() {
                    host_job.finish(Some(format!("{missing} cells failed: {error}")));
                }
            }
        }
        plan
    }

    /// Starts the readouts of files whose region changed. The cells a
    /// region needs are found without holding the state lock (a large
    /// outline takes a while), and the readout is installed only if the
    /// region did not change meanwhile.
    fn start_readouts(&self) {
        let requests: Vec<(FileId, Region, Arc<Grid>)> = {
            let mut guard = self.shared.lock();
            let st = &mut *guard;
            let (Some(model), Ok(markers)) = (st.model.info().cloned(), st.markers.clone()) else {
                return;
            };
            ensure_grids(st, &model, &markers);
            let mut requests = Vec::new();
            for (id, file) in st.files.iter_mut() {
                if !file.readout_wanted {
                    continue;
                }
                let Some(region) = file.region.clone() else {
                    file.readout_wanted = false;
                    continue;
                };
                match &file.grid {
                    Some(Ok(grid)) => requests.push((*id, region, Arc::clone(grid))),
                    Some(Err(problem)) => {
                        file.readout_wanted = false;
                        file.readout = Some(ReadoutJob {
                            readout: failed_readout(
                                &mut st.readout_seq,
                                Some(region),
                                problem.message(),
                            ),
                            job: None,
                            confirm: None,
                            started: Instant::now(),
                        });
                    }
                    None => {}
                }
            }
            requests
        };
        for (id, region, grid) in requests {
            let cells = region.cells(&grid, READOUT_MAX_WINDOWS * 2);
            let windows: HashSet<Idx> = cells
                .iter()
                .flat_map(|cell| grid.covering(*cell).map(|(w, _, _)| w))
                .collect();
            let mut guard = self.shared.lock();
            let st = &mut *guard;
            let markers = st.markers.as_ref().map(Vec::len).unwrap_or(0);
            let threshold = st.model.info().and_then(|m| m.recon_threshold);
            let seconds = st.throughput.seconds_per_window();
            let Some(file) = st.files.get_mut(&id) else {
                continue;
            };
            let unchanged = file.readout_wanted
                && file.region.as_ref() == Some(&region)
                && file.grid().is_some_and(|g| Arc::ptr_eq(g, &grid));
            if !unchanged {
                // A newer outline or grid: the next pass takes it.
                continue;
            }
            file.readout_wanted = false;
            if let Some(old) = file.readout.take()
                && let Some(job) = old.job
            {
                job.finish(Some("replaced by a new region".into()));
            }
            st.readout_seq += 1;
            let mut readout = Readout::new(
                st.readout_seq,
                region,
                cells.clone(),
                markers,
                windows.len(),
            );
            if cells.is_empty() {
                readout.state =
                    ReadoutState::Failed("No part of that region is on the slide.".into());
            } else if windows.len() > READOUT_MAX_WINDOWS {
                readout.state = ReadoutState::Failed(format!(
                    "That region needs more than {READOUT_MAX_WINDOWS} windows. Draw a smaller one."
                ));
            }
            let mut job = ReadoutJob {
                readout,
                job: None,
                confirm: None,
                started: Instant::now(),
            };
            if job.readout.is_running() {
                // Cells already known count straight away.
                for cell in &cells {
                    if let Some((data, _)) = file.cells.get(cell) {
                        job.readout.consume(*cell, data, &grid, threshold);
                    }
                }
                let left: usize = job
                    .readout
                    .remaining
                    .iter()
                    .map(|cell| missing_windows_of(&file.accums, &grid, *cell).len())
                    .sum::<usize>()
                    .min(windows.len());
                if left > READOUT_CONFIRM_WINDOWS {
                    job.readout.state = ReadoutState::Confirming;
                    job.confirm = self
                        .host
                        .confirm(ConfirmArgs {
                            title: "Measure a large region?".into(),
                            message: format!(
                                "This region needs about {left} more windows of predictions, \
                                 roughly {} on this computer. The overlay keeps working \
                                 meanwhile, and the readout can be cancelled.",
                                present::duration(left as f64 * seconds)
                            ),
                            confirm_label: "Measure".into(),
                            cancel_label: "Cancel".into(),
                            destructive: false,
                        })
                        .ok();
                    if job.confirm.is_none() {
                        job.readout.state = ReadoutState::Running;
                    }
                }
                if job.readout.is_running() {
                    job.job = self.host.start_job("Cellaris region readout", true).ok();
                }
            }
            file.readout = Some(job);
        }
    }

    // -- computing -----------------------------------------------------------

    fn compute(&mut self, plan: &Plan) -> Vec<WindowResult> {
        let (markers, info) = {
            let st = self.shared.lock();
            (
                st.markers.clone().unwrap_or_default(),
                st.model.info().cloned(),
            )
        };
        let Some(info) = info else {
            // Every planned window must come back, or it would stay in
            // flight and never be planned again.
            return plan
                .windows
                .iter()
                .map(|planned| WindowResult {
                    file: planned.file,
                    window: planned.window,
                    grid: Arc::clone(&planned.grid),
                    outcome: Outcome::Failed("the model is not loaded".to_string()),
                })
                .collect();
        };
        let side = info.window;
        let channels = Channels {
            markers: markers.len(),
            recon: info.recon,
        };
        let started = Instant::now();

        // Read and classify.
        let mut prepared: Vec<(usize, resample::Analysis)> = Vec::new();
        let mut input: Vec<f32> = Vec::new();
        let mut results: Vec<WindowResult> = Vec::with_capacity(plan.windows.len());
        for planned in &plan.windows {
            if self.stopping() {
                break;
            }
            let outcome = self
                .host
                .read_region(planned.plan.request)
                .map_err(|e| format!("the slide could not be read: {e}"))
                .and_then(|image| resample::resample(&image, &planned.plan, side));
            let outcome = match outcome {
                Err(error) => Outcome::Failed(error),
                Ok(window) => {
                    let analysis = resample::analyse(&window);
                    match analysis.kind {
                        WindowKind::Background => Outcome::Background,
                        WindowKind::Dark => Outcome::Dark,
                        WindowKind::Tissue => {
                            let offset = input.len();
                            input.resize(offset + 3 * side * side, 0.0);
                            resample::model_input(&window, &mut input[offset..]);
                            prepared.push((results.len(), analysis));
                            // Filled in after inference.
                            Outcome::Failed(String::new())
                        }
                    }
                }
            };
            results.push(WindowResult {
                file: planned.file,
                window: planned.window,
                grid: Arc::clone(&planned.grid),
                outcome,
            });
        }

        // Predict, in runs of the batch size.
        let per = 3 * side * side;
        let run = plan.batch.max(1);
        let mut prepared = prepared.into_iter().peekable();
        let mut offset = 0;
        while prepared.peek().is_some() {
            if self.stopping() {
                for (index, _) in prepared.by_ref() {
                    results[index].outcome = Outcome::Failed("stopping".to_string());
                }
                break;
            }
            let n = (input.len() / per - offset).min(run);
            let slice = &input[offset * per..(offset + n) * per];
            let predicted = match self.predictor.as_mut() {
                Some(predictor) => catch_unwind(AssertUnwindSafe(|| predictor.predict(slice, n)))
                    .unwrap_or_else(|payload| {
                        Err(format!(
                            "inference crashed: {}",
                            panic_text(payload.as_ref())
                        ))
                    }),
                None => Err("the model is not loaded".to_string()),
            };
            if let Err(error) = &predicted
                && error.starts_with("inference crashed")
            {
                // Loaded again before the next run.
                self.predictor = None;
                let mut st = self.shared.lock();
                st.model = ModelState::Idle;
                st.load_wanted = true;
            }
            for k in 0..n {
                let Some((index, analysis)) = prepared.next() else {
                    break;
                };
                results[index].outcome = match &predicted {
                    Err(error) => Outcome::Failed(format!("inference failed: {error}")),
                    Ok(prediction) => {
                        tissue_outcome(prediction, k, &markers, &info, channels, analysis)
                    }
                };
            }
            offset += n;
        }

        let tissue = results
            .iter()
            .filter(|r| matches!(r.outcome, Outcome::Tissue { .. }))
            .count();
        if tissue > 0 {
            let mut st = self.shared.lock();
            st.throughput.windows += tissue as u64;
            st.throughput.seconds += started.elapsed().as_secs_f64();
        }
        results
    }

    // -- integrating -----------------------------------------------------------

    fn integrate(&mut self, generation: u64, results: Vec<WindowResult>, publish: &mut Publish) {
        let mut guard = self.shared.lock();
        let st = &mut *guard;
        if st.generation != generation {
            for result in &results {
                if let Some(file) = st.files.get_mut(&result.file) {
                    file.inflight.remove(&result.window);
                }
            }
            return;
        }
        let Ok(markers) = st.markers.clone() else {
            return;
        };
        let info = st.model.info().cloned();
        let threshold = info.as_ref().and_then(|m| m.recon_threshold);
        let channels = Channels {
            markers: markers.len(),
            recon: info.as_ref().is_some_and(|m| m.recon),
        };
        publish.follow(st);
        let tick = st.tick;
        let now = Instant::now();
        let mut failures = Vec::new();
        for result in results {
            let Some(file) = st.files.get_mut(&result.file) else {
                continue;
            };
            file.inflight.remove(&result.window);
            if file.grid().is_none_or(|g| **g != *result.grid) {
                continue;
            }
            let grid = &result.grid;
            let mut finished: Vec<(Idx, CellData)> = Vec::new();
            match result.outcome {
                Outcome::Failed(error) => {
                    let attempts = file.failures.get(&result.window).map_or(0, |f| f.attempts) + 1;
                    file.failures.insert(
                        result.window,
                        Failure {
                            attempts,
                            retry_at: now + backoff(attempts),
                            error: error.clone(),
                        },
                    );
                    failures.push(error);
                    continue;
                }
                outcome @ (Outcome::Background | Outcome::Dark) => {
                    let dark = matches!(outcome, Outcome::Dark);
                    if dark {
                        file.dark_windows += 1;
                    }
                    file.failures.remove(&result.window);
                    for (cell, _, _) in grid.window_cells(result.window) {
                        if !file.cells.contains_key(&cell) {
                            file.accums.remove(&cell);
                            finished.push((
                                cell,
                                if dark {
                                    CellData::Dark
                                } else {
                                    CellData::Empty
                                },
                            ));
                        }
                    }
                }
                Outcome::Tissue {
                    planes,
                    blocks,
                    unusual,
                } => {
                    file.tissue_windows += 1;
                    if unusual {
                        file.unusual_windows += 1;
                    }
                    file.failures.remove(&result.window);
                    let ramp = Ramp::new(grid.window);
                    for (cell, dx, dy) in grid.window_cells(result.window) {
                        if file.cells.contains_key(&cell) {
                            continue;
                        }
                        let accum = file.accums.entry(cell).or_insert_with(|| {
                            Accum::new(channels.count(), grid.stride, expected_bits(cell.i, cell.j))
                        });
                        accum.last_used = tick;
                        accum.add(dx, dy, &ramp, &planes, grid.window, &blocks);
                        if accum.is_complete()
                            && let Some(accum) = file.accums.remove(&cell)
                        {
                            let (values, bases) = accum.finish(&ramp);
                            finished.push((
                                cell,
                                finalize(&values, &bases, grid.stride, channels, &markers),
                            ));
                        }
                    }
                }
            }
            for (cell, data) in finished {
                finish_cell(file, grid, cell, data, tick, threshold, publish, false);
            }
        }
        if let Some(first) = failures.first() {
            notify_once(
                &self.host,
                st,
                "Cellaris could not compute part of the slide; retrying",
                first,
            );
        }
        evict(st, publish);
    }

    fn integrate_disk(
        &mut self,
        generation: u64,
        loaded: Vec<(FileId, Idx, Option<CellData>)>,
        publish: &mut Publish,
    ) {
        let mut guard = self.shared.lock();
        let st = &mut *guard;
        if st.generation != generation {
            return;
        }
        publish.follow(st);
        let threshold = st.model.info().and_then(|m| m.recon_threshold);
        let tick = st.tick;
        for (id, cell, data) in loaded {
            let Some(file) = st.files.get_mut(&id) else {
                continue;
            };
            let Some(grid) = file.grid().cloned() else {
                continue;
            };
            match data {
                Some(data) if !file.cells.contains_key(&cell) => {
                    file.accums.remove(&cell);
                    finish_cell(file, &grid, cell, data, tick, threshold, publish, true);
                }
                Some(_) => {}
                None => {
                    file.disk_misses.insert(cell);
                }
            }
        }
        evict(st, publish);
    }

    /// Draws cells again after the display changed, a pass at a time.
    fn rerender(&mut self, publish: &mut Publish) -> bool {
        let mut guard = self.shared.lock();
        let st = &mut *guard;
        if st.display().is_none() || st.generation != publish.generation {
            return false;
        }
        publish.follow(st);
        let mut budget = RERENDER;
        for (id, file) in st.files.iter_mut() {
            let Some(grid) = file.grid().cloned() else {
                file.stale.clear();
                continue;
            };
            while budget > 0 {
                let Some(cell) = file.stale.pop_first() else {
                    break;
                };
                if let Some((CellData::Tissue(tissue), _)) = file.cells.get(&cell) {
                    publish.renders.push(Render {
                        file: *id,
                        cell,
                        rect: grid.cell_rect(cell),
                        tissue: Arc::clone(tissue),
                        new: false,
                    });
                    budget -= 1;
                }
            }
        }
        budget < RERENDER
    }

    // -- publishing --------------------------------------------------------------

    /// Sends a pass's results. Cells are drawn and uploaded without the
    /// state lock (it would stall the UI thread's handlers); a cell that was
    /// invalidated, reset or evicted meanwhile is then removed again, so the
    /// host never keeps a stale raster.
    fn publish(&mut self, mut publish: Publish) {
        let tooltips = publish.tooltips;
        remove_rasters(&self.host, std::mem::take(&mut publish.removals), tooltips);
        for (file, zoom) in std::mem::take(&mut publish.min_zooms) {
            let _ = self.host.layer(values_layer(file)).set_min_zoom(Some(zoom));
        }
        if let Some(display) = publish.display.take()
            && !publish.renders.is_empty()
            && publish.side > 0
        {
            let mut sent = Vec::with_capacity(publish.renders.len());
            let mut values: BTreeMap<FileId, Vec<OverlayObject>> = BTreeMap::new();
            for render in publish.renders.drain(..) {
                let image = Image {
                    width: publish.side,
                    height: publish.side,
                    data: display.render(&render.tissue),
                };
                match self.host.layer(pred_layer(render.file)).put_raster(
                    cell_object(render.cell),
                    render.rect,
                    image,
                    true,
                ) {
                    Ok(()) => sent.push((render.file, render.cell)),
                    Err(error) => {
                        self.host.debug(format!("raster upload failed: {error}"));
                        continue;
                    }
                }
                if render.new
                    && tooltips
                    && let Some(object) = present::cell_values(
                        render.cell,
                        render.rect,
                        &render.tissue,
                        &publish.names,
                        publish.threshold,
                    )
                {
                    values.entry(render.file).or_default().push(object);
                }
            }
            for (file, objects) in values {
                if let Err(error) = self.host.layer(values_layer(file)).upsert(objects) {
                    self.host
                        .debug(format!("value tooltips could not be sent: {error}"));
                }
            }
            let stale: BTreeMap<FileId, Vec<Idx>> = {
                let st = self.shared.lock();
                let mut stale: BTreeMap<FileId, Vec<Idx>> = BTreeMap::new();
                for (file, cell) in sent {
                    let current = st.generation == publish.generation
                        && st
                            .files
                            .get(&file)
                            .is_some_and(|f| f.uploaded.contains(&cell));
                    if !current {
                        stale.entry(file).or_default().push(cell);
                    }
                }
                stale
            };
            remove_rasters(&self.host, stale.into_iter().collect(), tooltips);
        }
        for (shelf, cell, data) in publish.writes {
            if let Err(error) = shelf.store(cell, &data) {
                self.host
                    .debug(format!("could not cache a cell on disk: {error}"));
            }
            self.writes += 1;
        }
        if self.writes >= 500 {
            self.writes = 0;
            self.prune_disk();
        }
    }

    fn prune_disk(&self) {
        let (root, budget) = {
            let st = self.shared.lock();
            (
                st.paths.cache_dir.join("cells"),
                st.prefs.disk_cache_mb.saturating_mul(1 << 20),
            )
        };
        if let Err(error) = cache::prune(&root, budget) {
            self.host
                .debug(format!("could not prune the cell cache: {error}"));
        }
    }

    /// The panel, the status bar, readout jobs and every file's marks.
    fn publish_ui(&mut self, force: bool) {
        if !force && self.last_ui.elapsed() < UI_INTERVAL {
            return;
        }
        self.last_ui = Instant::now();
        let mut st = self.shared.lock();
        present::refresh(&self.host, &mut st);
    }

    fn refresh_file(&self, file: FileId) {
        let mut st = self.shared.lock();
        present::refresh_file(&self.host, &mut st, file);
    }
}

fn missing_windows_of(accums: &HashMap<Idx, Accum>, grid: &Grid, cell: Idx) -> Vec<Idx> {
    let accum = accums.get(&cell);
    grid.covering(cell)
        .filter(|(_, dx, dy)| accum.is_none_or(|a| !a.has(bit(*dx, *dy))))
        .map(|(window, _, _)| window)
        .collect()
}

fn failed_readout(seq: &mut u64, region: Option<Region>, message: String) -> Readout {
    *seq += 1;
    let region = region.unwrap_or(Region::Rectangle(Rect::default()));
    let mut readout = Readout::new(*seq, region, Vec::new(), 0, 0);
    readout.state = ReadoutState::Failed(message);
    readout
}

/// The planes of one predicted window: the selected markers on the arcsinh
/// scale, then the tissue mask, then the reconstruction error.
fn tissue_outcome(
    prediction: &crate::model::Prediction,
    k: usize,
    markers: &[Marker],
    info: &ModelInfo,
    channels: Channels,
    analysis: resample::Analysis,
) -> Outcome {
    let (planes, unusual) = window_planes(prediction, k, markers, info, channels, &analysis.mask);
    Outcome::Tissue {
        planes,
        blocks: analysis.blocks,
        unusual,
    }
}

/// Window `k` of a prediction as the planes its cells blend
/// (`channels × side²`), and whether its tissue is above the model's
/// reconstruction-error threshold.
pub(crate) fn window_planes(
    prediction: &crate::model::Prediction,
    k: usize,
    markers: &[Marker],
    info: &ModelInfo,
    channels: Channels,
    mask: &[f32],
) -> (Vec<f32>, bool) {
    let side = prediction.side;
    let area = side * side;
    let mut planes = vec![0f32; channels.count() * area];
    let base = k * prediction.channels * area;
    for (m, marker) in markers.iter().enumerate() {
        let source =
            &prediction.markers[base + marker.channel * area..base + (marker.channel + 1) * area];
        for (out, value) in planes[m * area..(m + 1) * area].iter_mut().zip(source) {
            *out = markers::intensity(*value, info.raw_intensity);
        }
    }
    planes[channels.mask() * area..(channels.mask() + 1) * area].copy_from_slice(mask);
    let mut unusual = false;
    if let (Some(index), Some(recon)) = (channels.recon(), &prediction.recon) {
        let source = &recon[k * area..(k + 1) * area];
        planes[index * area..(index + 1) * area].copy_from_slice(source);
        if let Some(threshold) = info.recon_threshold {
            let (sum, weight) = source
                .iter()
                .zip(mask)
                .fold((0f32, 0f32), |(s, w), (r, m)| (s + r * m, w + m));
            unusual = weight > 0.0 && sum / weight > threshold;
        }
    }
    (planes, unusual)
}

/// Records a finished cell: readout, raster (drawn later, outside the
/// lock) and disk (unless it came from there).
#[allow(clippy::too_many_arguments)]
fn finish_cell(
    file: &mut FileState,
    grid: &Grid,
    cell: Idx,
    data: CellData,
    tick: u64,
    threshold: Option<f32>,
    publish: &mut Publish,
    from_disk: bool,
) {
    if let Some(job) = file.readout.as_mut() {
        let was_running = job.readout.is_running();
        job.readout.consume(cell, &data, grid, threshold);
        if was_running
            && job.readout.state == ReadoutState::Done
            && let Some(host_job) = job.job.take()
        {
            host_job.finish(None);
        }
    }
    if let CellData::Tissue(tissue) = &data
        && file.layers
    {
        publish.renders.push(Render {
            file: grid.file_id,
            cell,
            rect: grid.cell_rect(cell),
            tissue: Arc::clone(tissue),
            new: true,
        });
        file.uploaded.insert(cell);
        if publish.tooltips && file.values_zoom.is_none() {
            let (w, h) = grid.cell_size();
            let zoom = TOOLTIP_CELL_PX / w.min(h).max(1e-9);
            file.values_zoom = Some(zoom);
            publish.min_zooms.push((grid.file_id, zoom));
        }
    }
    if let Some(shelf) = &file.shelf {
        if !from_disk {
            publish.writes.push((shelf.clone(), cell, data.clone()));
        }
        file.disk_misses.remove(&cell);
    }
    file.cells.insert(cell, (data, tick));
}

/// Keeps memory (and host rasters) within budget, dropping what is not in
/// view and was used least recently.
fn evict(st: &mut State, publish: &mut Publish) {
    let tick = st.tick;
    let total: usize = st.files.values().map(|f| f.cells.len()).sum();
    if total > CELL_BUDGET {
        let mut candidates: Vec<(u64, FileId, Idx)> = st
            .files
            .iter()
            .flat_map(|(id, f)| {
                f.cells
                    .iter()
                    .filter(|(_, (_, used))| *used < tick)
                    .map(move |(cell, (_, used))| (*used, *id, *cell))
            })
            .collect();
        candidates.sort();
        let excess = total - CELL_BUDGET * 9 / 10;
        let mut removed: BTreeMap<FileId, Vec<Idx>> = BTreeMap::new();
        for (_, id, cell) in candidates.into_iter().take(excess) {
            if let Some(file) = st.files.get_mut(&id) {
                file.cells.remove(&cell);
                file.stale.remove(&cell);
                if file.uploaded.remove(&cell) {
                    removed.entry(id).or_default().push(cell);
                }
            }
        }
        publish.removals.extend(removed);
    }
    let pending: usize = st.files.values().map(|f| f.accums.len()).sum();
    if pending > ACCUM_BUDGET {
        let mut candidates: Vec<(u64, FileId, Idx)> = st
            .files
            .iter()
            .flat_map(|(id, f)| {
                f.accums
                    .iter()
                    .filter(|(_, a)| a.last_used < tick)
                    .map(move |(cell, a)| (a.last_used, *id, *cell))
            })
            .collect();
        candidates.sort();
        for (_, id, cell) in candidates.into_iter().take(pending - ACCUM_BUDGET * 9 / 10) {
            if let Some(file) = st.files.get_mut(&id) {
                file.accums.remove(&cell);
            }
        }
    }
}

/// Creates the layers of files that have none yet; returns those files.
pub fn ensure_layers(host: &Host, st: &mut State) -> Vec<FileId> {
    let mut created = Vec::new();
    let opacity = st.prefs.opacity;
    let overlay = st.overlay;
    for (id, file) in st.files.iter_mut() {
        if file.layers {
            continue;
        }
        let predicted = LayerSpec {
            opacity,
            visible: overlay,
            ..LayerSpec::new(pred_layer(*id), *id)
                .named("Cellaris \u{2014} predicted markers (research use only)")
                // Under the host ROI's dimming and every annotation.
                .at_z(-60)
        };
        let marks = LayerSpec::new(marks_layer(*id), *id)
            .named("Cellaris \u{2014} progress")
            .at_z(160);
        let region = LayerSpec::new(region_layer(*id), *id)
            .named("Cellaris \u{2014} readout region")
            .at_z(95)
            .editable(true)
            .with_style(present::region_style());
        let mut specs = vec![predicted, marks, region];
        if st.tooltips != Support::No {
            // Not interactive, so it takes no clicks from other tools: the
            // host shows the tooltips of such layers when nothing else is
            // hovered.
            specs.push(LayerSpec {
                visible: overlay,
                ..LayerSpec::new(values_layer(*id), *id)
                    .named("Cellaris \u{2014} predicted values (research use only)")
                    .at_z(-59)
                    .with_style(present::values_style())
            });
        }
        if !specs
            .into_iter()
            .all(|spec| host.create_layer(spec).is_ok())
        {
            continue;
        }
        if st.tooltips == Support::Unknown {
            // Hosts without `set_min_zoom` also predate passive tooltips:
            // there the layer would hold objects nobody ever sees.
            match host.layer(values_layer(*id)).set_min_zoom(None) {
                Ok(()) => st.tooltips = Support::Yes,
                Err(error) if error.is_unsupported() => {
                    st.tooltips = Support::No;
                    let _ = host.layer(values_layer(*id)).remove_layer();
                }
                Err(_) => {}
            }
        }
        file.layers = true;
        created.push(*id);
    }
    created
}

/// Tells the user about a problem once per kind of message: the digits
/// (coordinates, sizes, counts) are left out of the comparison, so a read
/// error repeated for every window of a corrupt area is one notification.
pub fn notify_once(host: &Host, st: &mut State, title: &str, detail: &str) {
    let key: String = format!("{title}\n{detail}")
        .chars()
        .filter(|c| !c.is_ascii_digit())
        .collect();
    if st.notified.len() > 256 {
        st.notified.clear();
    }
    if st.notified.insert(key) {
        host.notify_error(title.to_string(), detail.to_string());
    }
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
    use crate::model::{Batch, ModelInfo};
    use eov_plugin_api::testing::MockHost;

    #[derive(Default)]
    struct NoBackend;

    impl Backend for NoBackend {
        fn load(&self, _: &LoadRequest) -> Result<Box<dyn Predictor>, LoadError> {
            Err(LoadError::new("no model", "tests do not load one"))
        }
    }

    fn model() -> ModelInfo {
        ModelInfo {
            path: "m.onnx".into(),
            bytes: 0,
            sha256: "0".repeat(64),
            channel_names: crate::markers::DEFAULT_MARKERS.map(String::from).to_vec(),
            raw_intensity: true,
            mpp: 0.325,
            mpp_recorded: true,
            window: 64,
            batch: Batch::Dynamic,
            recon: false,
            recon_threshold: None,
            title: None,
            version: None,
            description: None,
            producer: None,
            training_data: None,
            license: None,
            runtime: String::new(),
            threads: 1,
            notes: Vec::new(),
        }
    }

    #[test]
    fn a_raster_drawn_for_a_cell_dropped_meanwhile_is_taken_back() {
        let mock = MockHost::new();
        let host = Host::from_ffi(mock.to_ffi());
        host.create_layer(LayerSpec::new(pred_layer(1), 1)).unwrap();
        host.create_layer(LayerSpec::new(values_layer(1), 1))
            .unwrap();
        let mut st = State::new(Prefs::default(), Paths::default());
        let info = model();
        st.markers = markers::select(&info.channel_names, &st.prefs.markers);
        st.model = ModelState::Ready(Arc::new(info));
        let mut file = FileState::new(FileInfo {
            file_id: 1,
            ..Default::default()
        });
        file.layers = true;
        // Cell (0, 0) is still wanted; (1, 0) was evicted while it was drawn.
        file.uploaded.insert(Idx::new(0, 0));
        st.files.insert(1, file);
        let mut publish = Publish::default();
        publish.follow(&st);
        let shared = Arc::new(Shared::new(st));
        let tissue = Arc::new(TissueCell {
            planes: vec![200; 3 * 32 * 32],
            mask: vec![255; 32 * 32],
            blocks: vec![crate::blend::BlockStat {
                od: 0.3,
                coverage: 1.0,
                dark: 0.0,
                means: vec![5.0; 3],
                recon: None,
            }],
        });
        let rect = |i: f64| Rect {
            x: i * 32.0,
            y: 0.0,
            width: 32.0,
            height: 32.0,
        };
        let render = |i: i64, tissue: Arc<TissueCell>| Render {
            file: 1,
            cell: Idx::new(i, 0),
            rect: rect(i as f64),
            tissue,
            new: true,
        };
        publish.renders = vec![render(0, Arc::clone(&tissue)), render(1, tissue)];
        let mut worker = Worker::new(host, Arc::clone(&shared), Arc::new(NoBackend));
        worker.publish(publish);
        let state = mock.state();
        let rasters: Vec<&String> = state.layers[&pred_layer(1)].rasters.keys().collect();
        assert_eq!(rasters, vec!["c0_0"], "a dropped cell kept its raster");
        let values: Vec<&String> = state.layers[&values_layer(1)].objects.keys().collect();
        assert_eq!(values, vec!["c0_0"], "a dropped cell kept its tooltip");

        // After an invalidation every raster of the old generation goes.
        drop(state);
        let mut publish = Publish::default();
        publish.follow(&shared.lock());
        let removals = shared.lock().invalidate();
        assert_eq!(removals, vec![(1, vec![Idx::new(0, 0)])]);
        publish.renders = vec![render(
            0,
            Arc::new(TissueCell {
                planes: vec![9; 3 * 32 * 32],
                mask: vec![255; 32 * 32],
                blocks: Vec::new(),
            }),
        )];
        remove_rasters(&Host::from_ffi(mock.to_ffi()), removals, true);
        worker.publish(publish);
        assert!(mock.state().layers[&pred_layer(1)].rasters.is_empty());
        assert!(mock.state().layers[&values_layer(1)].objects.is_empty());
    }

    #[test]
    fn only_failures_of_cells_in_view_decide_whether_they_are_retried() {
        let mock = MockHost::new();
        let host = Host::from_ffi(mock.to_ffi());
        let mut st = State::new(Prefs::default(), Paths::default());
        let model = model();
        st.markers = markers::select(&model.channel_names, &st.prefs.markers);
        st.model = ModelState::Ready(Arc::new(model.clone()));
        st.overlay = true;
        let info = FileInfo {
            file_id: 1,
            width: 10_000,
            height: 10_000,
            mpp_x: Some(0.325),
            mpp_y: Some(0.325),
            ..Default::default()
        };
        let grid = Arc::new(scale_for(&info, &model, &st.prefs).unwrap());
        let mut file = FileState::new(info);
        file.grid = Some(Ok(Arc::clone(&grid)));
        st.files.insert(1, file);
        // 2 × 2 cells of 32 px in view.
        let visible = Rect {
            x: 3_200.0,
            y: 3_200.0,
            width: 64.0,
            height: 64.0,
        };
        st.views.insert(
            1,
            ViewInfo {
                pane: 1,
                file_id: 1,
                zoom: 2.0,
                visible,
                center_x: 3_232.0,
                center_y: 3_232.0,
                ..Default::default()
            },
        );
        let in_view: HashSet<Idx> = grid
            .cells_by_distance(&visible, 3_232.0, 3_232.0)
            .into_iter()
            .flat_map(|cell| grid.covering(cell).map(|(window, _, _)| window))
            .collect();
        let now = Instant::now();
        let failure = |attempts| Failure {
            attempts,
            retry_at: now + Duration::from_secs(60),
            error: "unreadable".into(),
        };
        let shared = Arc::new(Shared::new(st));
        let worker = Worker::new(host, Arc::clone(&shared), Arc::new(NoBackend));
        let line = || {
            present::badges(&shared.lock())
                .get(&present::slide_badge(1))
                .map(|(_, text)| text.clone())
                .unwrap_or_default()
        };

        // Every window in view was given up; a window elsewhere still waits
        // for its retry, but it is never retried while nobody needs it.
        {
            let mut st = shared.lock();
            let file = st.files.get_mut(&1).unwrap();
            for window in &in_view {
                file.failures.insert(*window, failure(MAX_ATTEMPTS));
            }
            file.failures.insert(Idx::new(0, 0), failure(1));
        }
        let plan = worker.plan();
        assert!(plan.windows.is_empty());
        {
            let st = shared.lock();
            assert_eq!((st.files[&1].failed.len(), st.files[&1].stuck), (4, 4));
        }
        assert!(
            line().contains("4 failed (Retry in the Cellaris panel)"),
            "{}",
            line()
        );

        // In view they wait for a retry; one given up elsewhere does not
        // matter.
        {
            let mut st = shared.lock();
            let file = st.files.get_mut(&1).unwrap();
            for window in &in_view {
                file.failures.insert(*window, failure(1));
            }
            file.failures.insert(Idx::new(0, 0), failure(MAX_ATTEMPTS));
        }
        worker.plan();
        {
            let st = shared.lock();
            assert_eq!((st.files[&1].failed.len(), st.files[&1].stuck), (4, 0));
        }
        assert!(line().contains("4 failed, retrying"), "{}", line());
    }

    #[test]
    fn retries_back_off_and_then_stop() {
        assert_eq!(backoff(1), Duration::from_secs(2));
        assert_eq!(backoff(4), Duration::from_secs(16));
        assert_eq!(backoff(10), Duration::from_secs(30));
        let now = Instant::now();
        let failure = Failure {
            attempts: 1,
            retry_at: now + Duration::from_secs(2),
            error: "x".into(),
        };
        assert!(failure.blocks(now));
        assert!(!failure.blocks(now + Duration::from_secs(3)));
        let given_up = Failure {
            attempts: MAX_ATTEMPTS,
            ..failure
        };
        assert!(given_up.blocks(now + Duration::from_secs(3600)));
    }
}
