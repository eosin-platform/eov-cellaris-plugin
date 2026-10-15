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

#[cfg(test)]
mod tests;

use config::{
    OVERLAY_BUTTON, OVERLAY_ICON, PANEL_BUTTON, PANEL_COMPONENT, PANEL_ICON, PANEL_UI, PANEL_WIDTH,
    Prefs, REGION_ICON, REGION_TOOL, keys,
};
use engine::{FileState, ModelState, Paths, Shared, State, Worker, pred_layer, region_layer};
use eov_plugin_api::prelude::*;
use markers::DisplayMode;
use model::{Backend, OrtBackend};
use region::{ReadoutState, Region};
use std::marker::PhantomData;
use std::path::PathBuf;
use std::sync::{Arc, Mutex};
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

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
    _backend: PhantomData<fn() -> B>,
}

eov_plugin_api::export_plugin!(Cellaris<OrtBackend>);

fn host_err(error: HostError) -> String {
    error.message
}

impl<B: Backend> Cellaris<B> {
    fn lock(&self) -> std::sync::MutexGuard<'_, State> {
        self.shared.lock()
    }

    fn persist<T: serde::Serialize + ?Sized>(&self, key: &str, value: &T) {
        if let Err(error) = self.host.set_setting(key, value) {
            self.host.warn(format!(
                "Cellaris could not save the setting {key}: {error}"
            ));
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
        for (id, file) in &st.files {
            if file.layers {
                let _ = self.host.layer(pred_layer(*id)).set_visible(on);
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
        // Only slides without a pixel size are affected.
        let ids: Vec<FileId> = st
            .files
            .iter()
            .filter(|(_, f)| f.info.mpp_x.is_none() && f.info.mpp_y.is_none())
            .map(|(id, _)| *id)
            .collect();
        for id in ids {
            if let Some(file) = st.files.get_mut(&id) {
                let cells = file.reset();
                if !cells.is_empty() {
                    let _ = self
                        .host
                        .layer(pred_layer(id))
                        .remove(cells.into_iter().map(engine::cell_object).collect());
                }
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
                .notify_error("The Cellaris panel could not open", error.message),
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

    fn set_region(&self, file_id: FileId, region: Option<Region>) {
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
            self.host.notify(NotifyLevel::Info, "Open a slide first.");
            return Ok(());
        };
        match self.host.roi(file_id).map_err(host_err)? {
            Some(rect) if rect.width > 0.0 && rect.height > 0.0 => {
                self.set_region(file_id, Some(Region::Rectangle(rect)));
            }
            _ => self.host.notify(
                NotifyLevel::Info,
                "This slide has no region of interest; draw one with the ROI tool, or outline a \
                 region with the Cellaris region tool.",
            ),
        }
        Ok(())
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
                job.job = self.host.start_job("Cellaris region readout", true).ok();
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
                    "Readout copied as tab-separated text.",
                );
            }
            None => self
                .host
                .notify(NotifyLevel::Info, "There is no finished readout to copy."),
        }
        Ok(())
    }

    // -- the model ----------------------------------------------------------------

    fn retry(&self) {
        let mut st = self.lock();
        for file in st.files.values_mut() {
            file.failures.clear();
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
        let filters = match choice {
            Choice::Model => vec![FileFilter {
                name: "ONNX model".into(),
                extensions: vec!["onnx".into()],
            }],
            // Versioned Linux names (libonnxruntime.so.1.22.0) have no
            // usable extension, so nothing is filtered there.
            Choice::Runtime if cfg!(target_os = "windows") => vec![FileFilter {
                name: "ONNX Runtime library".into(),
                extensions: vec!["dll".into()],
            }],
            Choice::Runtime if cfg!(target_os = "macos") => vec![FileFilter {
                name: "ONNX Runtime library".into(),
                extensions: vec!["dylib".into()],
            }],
            Choice::Runtime => Vec::new(),
        };
        let title = match choice {
            Choice::Model => "Choose a Cellaris marker model",
            Choice::Runtime => "Choose the ONNX Runtime library",
        };
        let request = self
            .host
            .open_file_dialog(FileDialogArgs {
                title: Some(title.into()),
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

impl<B: Backend> Plugin for Cellaris<B> {
    fn create(host: Host, init: PluginInit) -> Result<Self, String> {
        let (prefs, problems) = Prefs::from_settings(&init.settings);
        for problem in &problems {
            host.info(format!("Cellaris {problem}"));
        }
        host.add_toolbar_button(ToolbarButtonSpec {
            button_id: OVERLAY_BUTTON.into(),
            tooltip: "Cellaris: predicted markers over H&E (research use only)".into(),
            icon_svg: OVERLAY_ICON.into(),
            order: 0,
            ..Default::default()
        })
        .map_err(host_err)?;
        host.add_toolbar_button(ToolbarButtonSpec {
            button_id: PANEL_BUTTON.into(),
            tooltip: "Cellaris panel: legend, region readout and model".into(),
            icon_svg: PANEL_ICON.into(),
            order: 1,
            ..Default::default()
        })
        .map_err(host_err)?;
        let mut tool = ToolSpec::new(
            REGION_TOOL,
            "Cellaris region readout",
            REGION_ICON,
            ToolKind::Polygon,
        );
        tool.preview_style = Some(present::region_style());
        tool.edit_layers = Vec::new();
        host.register_tool(tool).map_err(host_err)?;

        let paths = Paths {
            plugin_root: PathBuf::from(&init.plugin_root),
            data_dir: PathBuf::from(&init.data_dir),
            cache_dir: PathBuf::from(&init.cache_dir),
        };
        let mut state = State::new(prefs, paths);
        state.setting_problems = problems;
        let shared = Arc::new(Shared::new(state));
        let worker = Worker::new(host.clone(), Arc::clone(&shared), Arc::new(B::default()));
        let handle = std::thread::Builder::new()
            .name("cellaris-worker".into())
            .spawn(move || worker.run())
            .map_err(|e| format!("could not start the Cellaris worker: {e}"))?;
        Ok(Self {
            host,
            shared,
            worker: Mutex::new(Some(handle)),
            dialogs: Mutex::new(Vec::new()),
            _backend: PhantomData,
        })
    }

    fn on_event(&self, event: HostEvent) -> Result<(), String> {
        match event {
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
                    host_job.finish(Some("the slide was closed".into()));
                }
                if st.active_file == Some(file_id) {
                    st.active_file = None;
                }
                self.refresh(&mut st);
            }
            HostEvent::ActiveChanged { file_id, .. } => {
                let mut st = self.lock();
                st.active_file = file_id;
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
                Some(region) => self.set_region(file_id, Some(region)),
                None => self.host.notify(
                    NotifyLevel::Info,
                    "Outline a region with at least three points.",
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
                    self.set_region(file_id, Region::from_shape(&shape));
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
                    self.set_region(file_id, None);
                }
            }
            HostEvent::ConfirmResult {
                request_id,
                confirmed,
            } => self.confirmed(&request_id, confirmed),
            HostEvent::JobCancelRequested { job_id } => self.cancel_readout(None, Some(&job_id)),
            HostEvent::DialogResult { request_id, paths } => self.chosen(&request_id, paths),
            HostEvent::SurfaceClosed { surface_id } => {
                let mut st = self.lock();
                if st.sidebar.as_ref().is_some_and(|s| s.id() == surface_id) {
                    st.sidebar = None;
                    st.last_sidebar = None;
                }
                self.refresh(&mut st);
            }
            HostEvent::Shutdown => self.shared.shut_down(),
            _ => {}
        }
        Ok(())
    }

    fn ui_callback(&self, _surface_id: &str, name: &str, args: &[Value]) -> Result<Value, String> {
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
            "measure-roi" => self.measure_roi()?,
            "draw-region" => self.host.activate_tool(REGION_TOOL).map_err(host_err)?,
            "clear-region" => {
                if let Some(file_id) = self.active_file() {
                    self.set_region(file_id, None);
                }
            }
            "cancel-readout" => {
                let active = self.active_file();
                self.cancel_readout(active, None);
            }
            "copy-readout" => self.copy_readout()?,
            "choose-model" => self.choose(Choice::Model)?,
            "choose-runtime" => self.choose(Choice::Runtime)?,
            "retry" => self.retry(),
            "close-panel" => self.close_panel(),
            _ => {}
        }
        Ok(Value::Null)
    }

    fn shutdown(&self) {
        self.shared.shut_down();
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
