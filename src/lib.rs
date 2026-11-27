//! Cellaris: predicted multiplex-IF markers overlaid on H&E, for eov.
//!
//! **Research use only.** The overlay shows what a model *predicts* the
//! Orion immunofluorescence channels would look like; it is not a
//! measurement and must not be used for diagnosis. Every view of a
//! prediction carries that statement.
//!
//! - [`geometry`] maps each slide onto a grid at the model's own pixel size.
//! - [`resample`] reads windows onto it; [`blend`] feathers them into cells.
//! - [`model`] and [`runtime`] load the ONNX model and ONNX Runtime.
//! - [`engine`] runs it all on a worker thread; [`present`] shows it.
//! - [`region`] measures a drawn region on the same blended cells.
//! - [`batch`] predicts whole slides without a window, for `eov batch`.
//! - [`text`] keeps texts until they are shown, in the language of the
//!   interface then (and in English for logs and data).
//!
//! The interface is translated (`translations/`, see the README): texts
//! made here are marked with `tr!`, the panel's with `@tr`. What eov keeps
//! showing for the plugin (toolbar tooltips, the tool, layer names, the
//! panel's properties, badges) is sent again when the language changes.
//! Batch runs write English: their output is read by programs and people's
//! tools, as eov's own batch output is.

pub mod batch;
pub mod blend;
pub mod cache;
pub mod config;
pub mod engine;
pub mod geometry;
pub mod markers;
pub mod model;
pub mod present;
pub mod region;
pub mod resample;
pub mod runtime;
pub mod text;

#[cfg(test)]
mod tests;

use config::{
    OVERLAY_BUTTON, OVERLAY_ICON, PANEL_BUTTON, PANEL_COMPONENT, PANEL_ICON, PANEL_UI, PANEL_WIDTH,
    Prefs, REGION_ICON, REGION_TOOL, keys,
};
use engine::{
    FileState, ModelState, Paths, Shared, State, Support, Worker, pred_layer, region_layer,
    values_layer,
};
use eov_plugin_api::prelude::*;
use markers::DisplayMode;
use model::{Backend, OrtBackend};
use region::{ReadoutState, Region};
use std::marker::PhantomData;
use std::path::PathBuf;
use std::sync::{Arc, Mutex};
use std::thread::JoinHandle;
use std::time::{Duration, Instant};
use text::{Text, text};

/// What an open file dialog is choosing.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Choice {
    Model,
    Runtime,
}

/// The plugin, generic over how models are loaded so tests can run without
/// ONNX Runtime.
pub struct Cellaris<B: Backend = OrtBackend> {
    host: Host,
    shared: Arc<Shared>,
    worker: Mutex<Option<JoinHandle<()>>>,
    dialogs: Mutex<Vec<(String, Choice)>>,
    /// Without a window (`eov batch`): only batch slides are handled.
    headless: bool,
    batch: batch::BatchRunner,
    /// For batch runs: the directories, and the settings eov started with.
    paths: Paths,
    prefs: Prefs,
    _backend: PhantomData<fn() -> B>,
}

eov_plugin_api::export_plugin!(Cellaris<OrtBackend>);

fn host_err(error: HostError) -> String {
    error.message
}

/// The overlay button's tooltip (eov adds its key).
fn overlay_tooltip() -> String {
    // Translators: the tooltip of the toolbar button that shows or hides the
    // predicted markers; H&E is haematoxylin and eosin. Keep "research use
    // only".
    tr!("Cellaris: predicted markers over H&E (research use only)")
}

/// The overlay button, with the key the settings give it (eov adds the key
/// to the tooltip, or refuses it if it is taken).
fn overlay_button(prefs: &Prefs, active: bool) -> ToolbarButtonSpec {
    ToolbarButtonSpec {
        button_id: OVERLAY_BUTTON.into(),
        tooltip: overlay_tooltip(),
        icon_svg: OVERLAY_ICON.into(),
        hotkey: (!prefs.overlay_key.is_empty()).then(|| Hotkey::key(&prefs.overlay_key)),
        active,
        order: 0,
        ..Default::default()
    }
}

/// The panel button's tooltip.
fn panel_tooltip() -> String {
    // Translators: the tooltip of the toolbar button that shows or hides the
    // Cellaris panel.
    tr!("Cellaris panel: legend, region readout and model")
}

/// Why eov refused a registration, as a kept text: in English for logs and
/// data, and as eov showed it (`shown_reason`) on screen. eov before 1.1
/// sends no shown reason: its English one is shown, kept in its order in
/// right-to-left text.
fn refusal_reason(reason: String, shown_reason: Option<String>) -> Text {
    Text::new(move || {
        if text::writing_english() {
            reason.clone()
        } else {
            shown_reason
                .clone()
                .unwrap_or_else(|| text::isolate(&reason))
        }
    })
}

/// The tool that outlines a region to measure.
fn region_tool() -> ToolSpec {
    let mut tool = ToolSpec::new(
        REGION_TOOL,
        engine::readout_job_title(),
        REGION_ICON,
        ToolKind::Polygon,
    );
    tool.preview_style = Some(present::region_style());
    tool.edit_layers = Vec::new();
    tool
}

impl<B: Backend> Cellaris<B> {
    fn lock(&self) -> std::sync::MutexGuard<'_, State> {
        self.shared.lock()
    }

    /// Saves a setting. When it cannot be saved, the log says which, in
    /// English, and the user is told in their language: eov shows logged
    /// warnings to the user as they are (PLG-15).
    fn persist<T: serde::Serialize + ?Sized>(&self, key: &str, value: &T) {
        if let Err(error) = self.host.set_setting(key, value) {
            self.host.info(format!(
                "Cellaris could not save the setting {key}: {error}"
            ));
            self.host.notify(
                NotifyLevel::Warning,
                // Translators: a warning; {} is eov's reason, in English.
                tr!(
                    "Cellaris could not save its settings: {}",
                    text::isolate(&error.message)
                ),
            );
        }
    }

    fn refresh(&self, st: &mut State) {
        present::refresh(&self.host, st);
    }

    // -- the overlay ------------------------------------------------------------

    fn set_overlay(&self, on: bool) {
        let mut st = self.lock();
        st.overlay = on;
        if on && !matches!(st.model, ModelState::Ready(_) | ModelState::Loading) {
            // Also a retry after a failed load (CEL-6).
            st.load_wanted = true;
        }
        let tooltips = st.tooltips != Support::No;
        for (id, file) in &st.files {
            if file.layers {
                let _ = self.host.layer(pred_layer(*id)).set_visible(on);
                if tooltips {
                    let _ = self.host.layer(values_layer(*id)).set_visible(on);
                }
            }
        }
        let _ = self.host.set_toolbar_button_active(OVERLAY_BUTTON, on);
        self.refresh(&mut st);
        drop(st);
        self.shared.wake();
    }

    fn set_opacity(&self, opacity: f32) {
        let opacity = opacity.clamp(0.1, 1.0);
        let mut st = self.lock();
        st.prefs.opacity = opacity;
        for (id, file) in &st.files {
            if file.layers {
                let _ = self.host.layer(pred_layer(*id)).update(LayerUpdate {
                    opacity: Some(opacity),
                    ..Default::default()
                });
            }
        }
        self.refresh(&mut st);
        drop(st);
        self.persist(keys::OPACITY, &opacity);
    }

    /// Redraws every cell after a display change (on the worker).
    fn redraw_all(&self, st: &mut State) {
        for file in st.files.values_mut() {
            let uploaded: Vec<_> = file.uploaded.iter().copied().collect();
            file.stale.extend(uploaded);
        }
    }

    fn set_marker_visible(&self, index: usize, visible: bool) {
        let mut st = self.lock();
        let Some(name) = st.prefs.markers.get(index).cloned() else {
            return;
        };
        st.prefs.hidden.retain(|hidden| *hidden != name);
        if !visible {
            st.prefs.hidden.push(name);
        }
        let hidden = st.prefs.hidden.clone();
        self.redraw_all(&mut st);
        self.refresh(&mut st);
        drop(st);
        self.persist(keys::HIDDEN, &hidden);
        self.shared.wake();
    }

    /// Sets (`Some`) or resets (`None`) a marker's display window. Only the
    /// drawing changes: cells keep their values and are drawn again, and
    /// readouts, tooltips and batch runs are not affected.
    fn set_window(&self, index: usize, window: Option<(f32, f32)>) {
        let mut st = self.lock();
        let Some(name) = st.prefs.markers.get(index).cloned() else {
            return;
        };
        let Some(default) = markers::default_window(&name) else {
            return;
        };
        let current = st
            .markers
            .as_ref()
            .ok()
            .and_then(|m| m.iter().find(|m| m.name == name))
            .map_or(default, |m| (m.low, m.high));
        let (low, high) = match window {
            Some(wanted) => markers::fit_window(&name, wanted, current),
            None => default,
        };
        if (low, high) == current && window.is_some() {
            return;
        }
        if (low - default.0).abs() < 1e-4 && (high - default.1).abs() < 1e-4 {
            st.prefs.windows.remove(&name);
        } else {
            st.prefs.windows.insert(name.clone(), [low, high]);
        }
        if let Ok(selected) = &mut st.markers {
            for marker in selected.iter_mut().filter(|m| m.name == name) {
                marker.low = low;
                marker.high = high;
            }
        }
        let windows = st.prefs.windows.clone();
        self.redraw_all(&mut st);
        self.refresh(&mut st);
        drop(st);
        self.persist(keys::WINDOWS, &windows);
        self.shared.wake();
    }

    /// Opens a marker's window editor in the panel, or closes it.
    fn edit_window(&self, index: usize) {
        let mut st = self.lock();
        st.window_editor = (st.window_editor != Some(index)).then_some(index);
        self.refresh(&mut st);
    }

    fn set_mode(&self, mode: DisplayMode) {
        let mut st = self.lock();
        st.prefs.mode = mode;
        self.redraw_all(&mut st);
        self.refresh(&mut st);
        drop(st);
        self.persist(keys::MODE, &mode);
        self.shared.wake();
    }

    fn set_assume_mpp(&self, assume: bool) {
        let mut st = self.lock();
        st.prefs.assume_mpp = assume;
        // Only slides without a usable pixel size are affected: those the
        // grid found none for, and those mapped with an assumed one.
        let ids: Vec<FileId> = st
            .files
            .iter()
            .filter(|(_, f)| {
                matches!(f.problem(), Some(geometry::ScaleProblem::Unknown))
                    || f.grid()
                        .is_some_and(|g| g.scale.source == geometry::ScaleSource::Assumed)
            })
            .map(|(id, _)| *id)
            .collect();
        let tooltips = st.tooltips != Support::No;
        for id in ids {
            if let Some(file) = st.files.get_mut(&id) {
                let cells = file.reset();
                engine::remove_cells(&self.host, id, cells, tooltips);
            }
        }
        self.refresh(&mut st);
        drop(st);
        self.persist(keys::ASSUME_MPP, &assume);
        self.shared.wake();
    }

    // -- the panel --------------------------------------------------------------

    fn open_panel(&self) {
        let mut st = self.lock();
        if st.sidebar.is_some() {
            return;
        }
        let mut spec = SurfaceSpec::new(SurfaceKind::Sidebar, PANEL_UI, PANEL_COMPONENT)
            .with_title("Cellaris")
            .with_size(PANEL_WIDTH, None);
        spec.properties = present::sidebar_properties(&st);
        spec.capture_keyboard = false;
        match self.host.open_surface(spec) {
            Ok(surface) => {
                st.sidebar = Some(surface);
                st.last_sidebar = None;
            }
            Err(error) => self
                .host
                .notify_error(tr!("The Cellaris panel could not open"), error.message),
        }
        self.refresh(&mut st);
    }

    fn close_panel(&self) {
        let mut st = self.lock();
        if let Some(surface) = st.sidebar.take() {
            let _ = surface.close();
        }
        st.last_sidebar = None;
        self.refresh(&mut st);
    }

    // -- regions ----------------------------------------------------------------

    /// Measures `region` on a slide (`None`: stops measuring). A region
    /// taken from the slide's region of interest follows it.
    fn set_region(&self, file_id: FileId, region: Option<Region>, from_roi: bool) {
        let mut st = self.lock();
        let needs_model = !matches!(st.model, ModelState::Ready(_) | ModelState::Loading);
        let Some(file) = st.files.get_mut(&file_id) else {
            return;
        };
        if let Some(old) = file.readout.take()
            && let Some(job) = old.job
        {
            job.finish(None);
        }
        file.readout_wanted = region.is_some();
        file.region_from_roi = from_roi && region.is_some();
        file.region = region;
        if file.readout_wanted && needs_model {
            st.load_wanted = true;
        }
        st.active_file = Some(file_id);
        self.refresh(&mut st);
        drop(st);
        self.shared.wake();
    }

    /// The slide the user is looking at.
    fn active_file(&self) -> Option<FileId> {
        let active = self.lock().active_file;
        active.or_else(|| {
            let snapshot = self.host.snapshot().ok()?;
            Some(snapshot.focused_view()?.file_id)
        })
    }

    fn measure_roi(&self) -> Result<(), String> {
        let Some(file_id) = self.active_file() else {
            self.host
                .notify(NotifyLevel::Info, tr!("Open a slide first."));
            return Ok(());
        };
        match self.host.roi(file_id).map_err(host_err)? {
            Some(rect) if rect.width > 0.0 && rect.height > 0.0 => {
                self.set_region(file_id, Some(Region::Rectangle(rect)), true);
            }
            _ => self.host.notify(
                NotifyLevel::Info,
                tr!(
                    // Translators: the ROI tool is eov's region-of-interest tool;
                    // "Cellaris region readout" is the name of Cellaris's tool (use
                    // the same words as there).
                    "This slide has no region of interest; draw one with the ROI tool, or outline a region with the Cellaris region readout tool."
                ),
            ),
        }
        Ok(())
    }

    /// A region measured from the region of interest follows it: drawn
    /// again, moved, undone or removed.
    fn roi_changed(&self, file_id: FileId, roi: Option<Rect>) {
        let region = roi
            .filter(|rect| rect.width > 0.0 && rect.height > 0.0)
            .map(Region::Rectangle);
        let follows = self
            .lock()
            .files
            .get(&file_id)
            .is_some_and(|file| file.region_from_roi && file.region != region);
        if follows {
            self.set_region(file_id, region, true);
        }
    }

    fn cancel_readout(&self, file_id: Option<FileId>, job_id: Option<&str>) {
        let mut st = self.lock();
        for (id, file) in st.files.iter_mut() {
            let Some(job) = file.readout.as_mut() else {
                continue;
            };
            let matches = match job_id {
                Some(job_id) => job.job.as_ref().is_some_and(|j| j.id() == job_id),
                None => Some(*id) == file_id,
            };
            if matches
                && matches!(
                    job.readout.state,
                    ReadoutState::Running | ReadoutState::Confirming
                )
            {
                job.readout.state = ReadoutState::Cancelled;
                if let Some(host_job) = job.job.take() {
                    host_job.finish(None);
                }
            }
        }
        self.refresh(&mut st);
    }

    fn confirmed(&self, request_id: &str, confirmed: bool) {
        let mut st = self.lock();
        for file in st.files.values_mut() {
            let Some(job) = file.readout.as_mut() else {
                continue;
            };
            if job.confirm.as_deref() != Some(request_id) {
                continue;
            }
            job.confirm = None;
            if job.readout.state != ReadoutState::Confirming {
                continue;
            }
            if confirmed {
                job.readout.state = ReadoutState::Running;
                job.started = Instant::now();
                job.job = self.host.start_job(engine::readout_job_title(), true).ok();
            } else {
                job.readout.state = ReadoutState::Cancelled;
            }
        }
        self.refresh(&mut st);
        drop(st);
        self.shared.wake();
    }

    fn copy_readout(&self) -> Result<(), String> {
        let text = {
            let st = self.lock();
            st.active_file.and_then(|id| present::copy_text(&st, id))
        };
        match text {
            Some(text) => {
                self.host.copy_text(text).map_err(host_err)?;
                self.host.notify(
                    NotifyLevel::Success,
                    tr!("Readout copied as tab-separated text."),
                );
            }
            None => self.host.notify(
                NotifyLevel::Info,
                tr!("There is no finished readout to copy."),
            ),
        }
        Ok(())
    }

    // -- the model ----------------------------------------------------------------

    fn retry(&self) {
        let mut st = self.lock();
        for file in st.files.values_mut() {
            file.failures.clear();
            if file.region.is_some()
                && file
                    .readout
                    .as_ref()
                    .is_some_and(|job| matches!(job.readout.state, ReadoutState::Failed(_)))
            {
                file.readout_wanted = true;
            }
        }
        if matches!(st.model, ModelState::Failed(_)) {
            st.notified.clear();
            st.load_wanted = true;
            st.reload = true;
        }
        self.refresh(&mut st);
        drop(st);
        self.shared.wake();
    }

    fn choose(&self, choice: Choice) -> Result<(), String> {
        // Translators: a kind of file in the file dialog.
        let runtime_files = || tr!("ONNX Runtime library");
        let filters = match choice {
            Choice::Model => vec![FileFilter {
                // Translators: a kind of file in the file dialog.
                name: tr!("ONNX model"),
                extensions: vec!["onnx".into()],
            }],
            // Versioned Linux names (libonnxruntime.so.1.22.0) have no
            // usable extension, so nothing is filtered there.
            Choice::Runtime if cfg!(target_os = "windows") => vec![FileFilter {
                name: runtime_files(),
                extensions: vec!["dll".into()],
            }],
            Choice::Runtime if cfg!(target_os = "macos") => vec![FileFilter {
                name: runtime_files(),
                extensions: vec!["dylib".into()],
            }],
            Choice::Runtime => Vec::new(),
        };
        let title = match choice {
            // Translators: the title of the file dialog.
            Choice::Model => tr!("Choose a Cellaris marker model"),
            // Translators: the title of the file dialog.
            Choice::Runtime => tr!("Choose the ONNX Runtime library"),
        };
        let request = self
            .host
            .open_file_dialog(FileDialogArgs {
                title: Some(title),
                filters,
                ..Default::default()
            })
            .map_err(host_err)?;
        self.dialogs
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .push((request, choice));
        Ok(())
    }

    fn chosen(&self, request_id: &str, paths: Vec<String>) {
        let choice = {
            let mut dialogs = self.dialogs.lock().unwrap_or_else(|e| e.into_inner());
            let Some(index) = dialogs.iter().position(|(id, _)| id == request_id) else {
                return;
            };
            dialogs.remove(index).1
        };
        let Some(path) = paths.into_iter().next() else {
            return;
        };
        let mut st = self.lock();
        match choice {
            Choice::Model => {
                st.prefs.model_path = Some(path.clone());
                st.reload = true;
                st.load_wanted = true;
                st.notified.clear();
                drop(st);
                self.persist(keys::MODEL, &path);
            }
            Choice::Runtime => {
                st.prefs.runtime_path = Some(path.clone());
                if matches!(st.model, ModelState::Ready(_)) {
                    // ONNX Runtime loads once per process.
                    st.runtime_restart = true;
                } else {
                    st.load_wanted = true;
                    st.reload = true;
                    st.notified.clear();
                }
                drop(st);
                self.persist(keys::RUNTIME, &path);
            }
        }
        let mut st = self.lock();
        self.refresh(&mut st);
        drop(st);
        self.shared.wake();
    }

    // -- downloading the model ------------------------------------------------------

    fn toggle_download(&self) {
        let mut st = self.lock();
        st.download_open = !st.download_open;
        st.download_note = None;
        self.refresh(&mut st);
    }

    /// *Download*: asks eov for the model at `url` with `sha256`, which are
    /// kept for next time. eov answers at once when its model store has
    /// the model; otherwise it asks the user, downloads it and checks it.
    fn download(&self, url: &str, sha256: &str) {
        let mut st = self.lock();
        if st.model_fetch.is_some() {
            return;
        }
        let Some(sha256) = config::parse_sha256(sha256) else {
            st.download_note = Some(text!(tr!(
                // Translators: SHA-256 is a checksum (not translated); hex digits
                // are 0–9 and a–f.
                "Enter the model's SHA-256 (64 hex digits): the download is checked against it."
            )));
            self.refresh(&mut st);
            return;
        };
        let url = Some(url.trim().to_string()).filter(|url| !url.is_empty());
        st.prefs.model_url = url.clone();
        st.prefs.model_sha256 = Some(sha256.clone());
        drop(st);
        match &url {
            Some(url) => self.persist(keys::MODEL_URL, url),
            None => {
                let _ = self.host.remove_setting(keys::MODEL_URL);
            }
        }
        self.persist(keys::MODEL_SHA256, &sha256);
        let asked = self.host.fetch_model(ModelFetchArgs {
            url,
            sha256,
            name: Some("Cellaris marker model".into()),
            ..Default::default()
        });
        let mut st = self.lock();
        match asked {
            Ok(request) => {
                st.model_fetch = Some(request);
                st.model_store = Support::Yes;
                st.download_note = None;
            }
            Err(error) if error.is_unsupported() => {
                st.model_store = Support::No;
                st.download_note = Some(text!(tr!(
                    // Translators: "Choose model…" is the panel's button: use the
                    // same words as there.
                    "This eov cannot download models: choose the model file with Choose model\u{2026}"
                )));
            }
            Err(error) => st.download_note = Some(Text::plain(error.message)),
        }
        self.refresh(&mut st);
    }

    /// eov's answer: the downloaded model is the model from now on, loaded
    /// and checked like one chosen in the panel.
    fn fetched(
        &self,
        request_id: &str,
        path: Option<String>,
        error: Option<String>,
        cancelled: bool,
    ) {
        let mut st = self.lock();
        if st.model_fetch.as_deref() != Some(request_id) {
            return;
        }
        st.model_fetch = None;
        let Some(path) = path else {
            // eov has told the user why; the panel keeps it in view.
            st.download_note = error.filter(|_| !cancelled).map(Text::plain);
            self.refresh(&mut st);
            return;
        };
        st.prefs.model_path = Some(path.clone());
        st.reload = true;
        st.load_wanted = true;
        st.notified.clear();
        st.download_open = false;
        st.download_note = None;
        drop(st);
        self.persist(keys::MODEL, &path);
        let mut st = self.lock();
        self.refresh(&mut st);
        drop(st);
        self.shared.wake();
    }

    /// The SDK has switched `tr!` to the language the user chose: what eov
    /// shows for the plugin is sent again in it. The panel's own texts
    /// follow by themselves; its properties, the badges, the status bar,
    /// the marks and labels on slides go again with the refresh, and the
    /// cells' value tooltips with their rasters.
    fn language_changed(&self) {
        // eov said in the old language why it refused the overlay's key: the
        // button is added again (with its state), and eov answers in the new
        // language if it still refuses it, without warning the user twice.
        let overlay_again = {
            let mut st = self.lock();
            st.refused
                .remove(OVERLAY_BUTTON)
                .map(|_| overlay_button(&st.prefs, st.overlay))
        };
        let refused = overlay_again.is_some();
        if let Some(button) = overlay_again {
            let _ = self.host.add_toolbar_button(button);
        }
        for (button_id, tooltip) in [
            (OVERLAY_BUTTON, overlay_tooltip()),
            (PANEL_BUTTON, panel_tooltip()),
        ] {
            if button_id == OVERLAY_BUTTON && refused {
                continue;
            }
            let _ = self.host.update_toolbar_button(ButtonUpdate {
                button_id: button_id.into(),
                tooltip: Some(tooltip),
                ..Default::default()
            });
        }
        let _ = self.host.register_tool(region_tool());
        let mut st = self.lock();
        engine::rename_layers(&self.host, &st);
        st.relabel();
        self.refresh(&mut st);
        drop(st);
        self.shared.wake();
    }

    fn add_file(&self, info: FileInfo) {
        let mut st = self.lock();
        let id = info.file_id;
        match st.files.get_mut(&id) {
            Some(file) => file.info = info,
            None => {
                st.files.insert(id, FileState::new(info));
            }
        }
        if st.active_file.is_none() {
            st.active_file = Some(id);
        }
        let _ = engine::ensure_layers(&self.host, &mut st);
        present::refresh_file(&self.host, &mut st, id);
        drop(st);
        self.shared.wake();
    }
}

/// The model the site set for *Download model…* (`model_sha256`) is the
/// model when eov's model store has it and none was chosen, also in batch
/// runs. Says whether the host keeps models.
fn use_stored_model(host: &Host, prefs: &mut Prefs) -> Support {
    let Some(sha256) = prefs.model_sha256.clone() else {
        return Support::Unknown;
    };
    match host.resolve_model(&sha256) {
        Ok(found) => {
            if prefs.model_path.is_none()
                && let Some(model) = found
            {
                prefs.model_path = Some(model.path);
            }
            Support::Yes
        }
        Err(error) if error.is_unsupported() => Support::No,
        Err(_) => Support::Unknown,
    }
}

impl<B: Backend> Plugin for Cellaris<B> {
    fn create(host: Host, init: PluginInit) -> Result<Self, String> {
        let (mut prefs, problems) = Prefs::from_settings(&init.settings);
        for problem in &problems {
            host.info(format!("Cellaris {}", problem.english()));
        }
        let model_store = use_stored_model(&host, &mut prefs);
        let paths = Paths {
            plugin_root: PathBuf::from(&init.plugin_root),
            data_dir: PathBuf::from(&init.data_dir),
            cache_dir: PathBuf::from(&init.cache_dir),
        };
        // Offered in the window too, for `eov batch list`; hosts from
        // before batch runs answer Unsupported. Anything else is a mistake
        // in the command's fixed spec: for the log, not a warning (eov
        // shows those to the user, untranslated, PLG-15).
        if let Err(error) = host.register_batch_command(batch::command())
            && !error.is_unsupported()
        {
            host.info(format!(
                "Cellaris could not offer its batch command: {error}"
            ));
        }
        if init.headless {
            // No window: no buttons, tool, panel or overlay worker.
            let state = State::new(prefs.clone(), paths.clone());
            return Ok(Self {
                host,
                shared: Arc::new(Shared::new(state)),
                worker: Mutex::new(None),
                dialogs: Mutex::new(Vec::new()),
                headless: true,
                batch: batch::BatchRunner::default(),
                paths,
                prefs,
                _backend: PhantomData,
            });
        }
        host.add_toolbar_button(overlay_button(&prefs, false))
            .map_err(host_err)?;
        host.add_toolbar_button(ToolbarButtonSpec {
            button_id: PANEL_BUTTON.into(),
            tooltip: panel_tooltip(),
            icon_svg: PANEL_ICON.into(),
            order: 1,
            ..Default::default()
        })
        .map_err(host_err)?;
        host.register_tool(region_tool()).map_err(host_err)?;

        let mut state = State::new(prefs.clone(), paths.clone());
        state.setting_problems = problems;
        state.model_store = model_store;
        let shared = Arc::new(Shared::new(state));
        let worker = Worker::new(host.clone(), Arc::clone(&shared), Arc::new(B::default()));
        let handle = std::thread::Builder::new()
            .name("cellaris-worker".into())
            .spawn(move || worker.run())
            // In English: eov logs it and writes it to the audit log as it
            // is, and shows it to the user inside its own translated
            // message.
            .map_err(|e| format!("could not start the Cellaris worker: {e}"))?;
        Ok(Self {
            host,
            shared,
            worker: Mutex::new(Some(handle)),
            dialogs: Mutex::new(Vec::new()),
            headless: false,
            batch: batch::BatchRunner::default(),
            paths,
            prefs,
            _backend: PhantomData,
        })
    }

    fn on_event(&self, event: HostEvent) -> Result<(), String> {
        match event {
            HostEvent::BatchRun {
                run_id,
                command_id,
                file,
                output_dir,
                params,
            } => self.batch.submit::<B>(
                &self.host,
                &self.paths,
                &self.prefs,
                batch::Job {
                    run_id,
                    command_id,
                    file,
                    output_dir: PathBuf::from(output_dir),
                    params,
                },
            ),
            HostEvent::Shutdown => {
                self.shared.shut_down();
                self.batch.stop();
            }
            // Without a window, slides come only as batch runs.
            _ if self.headless => {}
            HostEvent::Ready { restore_sidebar } => {
                if let Ok(snapshot) = self.host.snapshot() {
                    for file in snapshot.files {
                        self.add_file(file);
                    }
                }
                if restore_sidebar {
                    self.open_panel();
                }
            }
            HostEvent::FileOpened { file } => self.add_file(file),
            HostEvent::FileClosed { file_id } => {
                let mut st = self.lock();
                if let Some(file) = st.files.remove(&file_id)
                    && let Some(job) = file.readout
                    && let Some(host_job) = job.job
                {
                    host_job.finish(Some(engine::slide_closed()));
                }
                if st.active_file == Some(file_id) {
                    st.active_file = None;
                }
                self.refresh(&mut st);
            }
            HostEvent::ActiveChanged { pane, file_id } => {
                let mut st = self.lock();
                st.active_file = file_id;
                st.focused_pane = pane;
                self.refresh(&mut st);
            }
            HostEvent::ViewChanged { view } => {
                let mut st = self.lock();
                st.views.insert(view.pane, view);
                present::refresh_file(&self.host, &mut st, view.file_id);
                drop(st);
                self.shared.wake();
            }
            HostEvent::ToolbarButtonClicked { button_id } => match button_id.as_str() {
                OVERLAY_BUTTON => {
                    let on = !self.lock().overlay;
                    self.set_overlay(on);
                }
                PANEL_BUTTON => {
                    if self.lock().sidebar.is_some() {
                        self.close_panel();
                    } else {
                        self.open_panel();
                    }
                }
                _ => {}
            },
            HostEvent::ToolActivated { tool_id } if tool_id == REGION_TOOL => self.open_panel(),
            HostEvent::ShapeDrawn {
                tool_id,
                file_id,
                shape,
                ..
            } if tool_id == REGION_TOOL => match Region::from_shape(&shape) {
                Some(region) => self.set_region(file_id, Some(region), false),
                None => self.host.notify(
                    NotifyLevel::Info,
                    tr!("Outline a region with at least three points."),
                ),
            },
            HostEvent::ObjectEdited { object, shape } => {
                let file_id = self
                    .lock()
                    .files
                    .keys()
                    .copied()
                    .find(|id| region_layer(*id) == object.layer_id);
                if let Some(file_id) = file_id {
                    // Reshaped by hand, it no longer follows the ROI.
                    self.set_region(file_id, Region::from_shape(&shape), false);
                }
            }
            HostEvent::ObjectsDeleteRequested { objects } => {
                let files: Vec<FileId> = {
                    let st = self.lock();
                    st.files
                        .keys()
                        .copied()
                        .filter(|id| objects.iter().any(|o| o.layer_id == region_layer(*id)))
                        .collect()
                };
                for file_id in files {
                    self.set_region(file_id, None, false);
                }
            }
            HostEvent::ConfirmResult {
                request_id,
                confirmed,
            } => self.confirmed(&request_id, confirmed),
            HostEvent::JobCancelRequested { job_id } => self.cancel_readout(None, Some(&job_id)),
            HostEvent::DialogResult { request_id, paths } => self.chosen(&request_id, paths),
            HostEvent::ModelFetched {
                request_id,
                path,
                error,
                cancelled,
                ..
            } => self.fetched(&request_id, path, error, cancelled),
            HostEvent::SurfaceClosed { surface_id } => {
                let mut st = self.lock();
                if st.sidebar.as_ref().is_some_and(|s| s.id() == surface_id) {
                    st.sidebar = None;
                    st.last_sidebar = None;
                }
                self.refresh(&mut st);
            }
            HostEvent::RoiChanged { file_id, roi } => self.roi_changed(file_id, roi),
            HostEvent::LanguageChanged { .. } => self.language_changed(),
            HostEvent::DisplayChanged { pane, display } => {
                let mut st = self.lock();
                st.displays.insert(pane, display);
                self.refresh(&mut st);
            }
            HostEvent::RegistrationRefused {
                what,
                id,
                reason,
                shown_reason,
                ..
            } => {
                // eov has warned and logged it; the panel keeps it in view,
                // in eov's words as it shows them (from eov before 1.1, its
                // English ones). A newer answer replaces an older one.
                let reason = refusal_reason(reason, shown_reason);
                let refused = id.clone();
                let problem = match (what.as_str(), id.as_str()) {
                    ("hotkey", OVERLAY_BUTTON) => text!(tr!(
                        // Translators: {reason} is eov's explanation, a whole
                        // sentence; {key} is the name of a setting in
                        // settings.json, and "" an empty value there (neither
                        // is translated).
                        "{reason} Another key can be set as {key} in Cellaris's settings.json (\"\" for none).",
                        reason,
                        key = keys::OVERLAY_KEY
                    )),
                    ("hotkey", _) => reason,
                    _ => text!(tr!(
                        // Translators: eov did not accept something Cellaris
                        // registered: {id} is its identifier and {what} the kind of
                        // registration as eov names it in English, such as
                        // "hotkey" (neither is translated); {reason} is eov's
                        // explanation, a whole sentence.
                        "eov refused to register {id} ({what}): {reason}",
                        what = text::isolate(&what),
                        id = text::isolate(&id),
                        reason
                    )),
                };
                let mut st = self.lock();
                st.refused.insert(refused, problem);
                self.refresh(&mut st);
            }
            _ => {}
        }
        Ok(())
    }

    fn ui_callback(&self, _surface_id: &str, name: &str, args: &[Value]) -> Result<Value, String> {
        if self.headless {
            return Ok(Value::Null);
        }
        let flag = || args.first().and_then(Value::as_bool).unwrap_or(false);
        match name {
            "toggle-overlay" => {
                let on = !self.lock().overlay;
                self.set_overlay(on);
            }
            "set-opacity" => {
                if let Some(opacity) = args.first().and_then(Value::as_f64) {
                    self.set_opacity(opacity as f32);
                }
            }
            "set-marker-visible" => {
                let index = args.first().and_then(Value::as_u64).unwrap_or(u64::MAX);
                let visible = args.get(1).and_then(Value::as_bool).unwrap_or(true);
                self.set_marker_visible(index as usize, visible);
            }
            "set-fluorescence" => self.set_mode(if flag() {
                DisplayMode::Fluorescence
            } else {
                DisplayMode::Overlay
            }),
            "set-assume-mpp" => self.set_assume_mpp(flag()),
            "set-window" | "reset-window" | "edit-window" => {
                let index = args.first().and_then(Value::as_u64).unwrap_or(u64::MAX) as usize;
                let value = |k: usize| args.get(k).and_then(Value::as_f64).map(|v| v as f32);
                match name {
                    "set-window" => {
                        if let (Some(low), Some(high)) = (value(1), value(2)) {
                            self.set_window(index, Some((low, high)));
                        }
                    }
                    "reset-window" => self.set_window(index, None),
                    _ => self.edit_window(index),
                }
            }
            "measure-roi" => self.measure_roi()?,
            "draw-region" => self.host.activate_tool(REGION_TOOL).map_err(host_err)?,
            "clear-region" => {
                if let Some(file_id) = self.active_file() {
                    self.set_region(file_id, None, false);
                }
            }
            "cancel-readout" => {
                let active = self.active_file();
                self.cancel_readout(active, None);
            }
            "copy-readout" => self.copy_readout()?,
            "choose-model" => self.choose(Choice::Model)?,
            "choose-runtime" => self.choose(Choice::Runtime)?,
            "toggle-download" => self.toggle_download(),
            "download-model" => {
                let text = |index: usize| args.get(index).and_then(Value::as_str).unwrap_or("");
                self.download(text(0), text(1));
            }
            "retry" => self.retry(),
            "close-panel" => self.close_panel(),
            _ => {}
        }
        Ok(Value::Null)
    }

    fn shutdown(&self) {
        self.shared.shut_down();
        // A slide being predicted ends as failed ("stopped").
        self.batch.shut_down(Duration::from_secs(3));
        let handle = self.worker.lock().unwrap_or_else(|e| e.into_inner()).take();
        if let Some(handle) = handle {
            // A prediction is interrupted, but a slow slide read is not; do
            // not hold the host up for it (plugin libraries are never
            // unloaded, so a lingering thread is harmless).
            let deadline = Instant::now() + Duration::from_secs(3);
            while !handle.is_finished() && Instant::now() < deadline {
                std::thread::sleep(Duration::from_millis(10));
            }
            if handle.is_finished() {
                let _ = handle.join();
            }
        }
    }
}
