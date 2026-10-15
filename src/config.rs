//! Identifiers and persisted preferences.

use crate::markers::{DEFAULT_MARKERS, DisplayMode, MAX_MARKERS};
use eov_plugin_api::prelude::{Map, Value};
use serde::Serialize;

/// Toggles the predicted-marker overlay on every open slide.
pub const OVERLAY_BUTTON: &str = "cellaris_overlay";
/// The overlay button's key unless the `overlay_key` setting says
/// otherwise. eov binds f, m, n, r, d, i, the digits and + = - (users can
/// change those); the annotations plugin's tools take v . c p g b o l t h.
pub const DEFAULT_OVERLAY_KEY: &str = "x";
/// Shows or hides the Cellaris panel.
pub const PANEL_BUTTON: &str = "cellaris_panel";
/// Draws a region to measure.
pub const REGION_TOOL: &str = "cellaris_region";
/// The sidebar.
pub const PANEL_UI: &str = "ui/cellaris.slint";
pub const PANEL_COMPONENT: &str = "CellarisPanel";
pub const PANEL_WIDTH: f32 = 340.0;

/// The statement shown wherever predictions are (CEL-9, §14.6).
pub const RESEARCH_NOTICE: &str = "PREDICTED \u{2014} research use only, not for diagnosis";

pub const OVERLAY_ICON: &str = "<svg xmlns=\"http://www.w3.org/2000/svg\" viewBox=\"0 0 24 24\" \
    fill=\"none\" stroke=\"currentColor\" stroke-width=\"1.8\"><circle cx=\"12\" cy=\"12\" r=\"8\"/>\
    <circle cx=\"12\" cy=\"12\" r=\"3\"/></svg>";
pub const PANEL_ICON: &str = "<svg xmlns=\"http://www.w3.org/2000/svg\" viewBox=\"0 0 24 24\" \
    fill=\"none\" stroke=\"currentColor\" stroke-width=\"1.8\" stroke-linejoin=\"round\">\
    <rect x=\"4\" y=\"4\" width=\"16\" height=\"16\" rx=\"2\"/><path d=\"M14 4v16\"/>\
    <path d=\"M7 8h4M7 12h4M7 16h4\"/></svg>";
pub const REGION_ICON: &str = "<svg xmlns=\"http://www.w3.org/2000/svg\" viewBox=\"0 0 24 24\" \
    fill=\"none\" stroke=\"currentColor\" stroke-width=\"1.8\" stroke-linejoin=\"round\">\
    <path d=\"M4 7l7-3 9 5-3 10-11 1z\"/><circle cx=\"12\" cy=\"12\" r=\"2\"/></svg>";

/// What the user chose, kept in the host's settings store, one key each.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct Prefs {
    /// Overlay opacity, 0.1–1.
    pub opacity: f32,
    /// Markers computed, in colour order.
    pub markers: Vec<String>,
    /// Computed markers not drawn.
    pub hidden: Vec<String>,
    pub mode: DisplayMode,
    /// Windows per inference run.
    pub batch_size: usize,
    /// Inference threads; 0 picks half the CPUs.
    pub threads: usize,
    pub model_path: Option<String>,
    pub runtime_path: Option<String>,
    /// Run on slides without a pixel size as if they matched the model.
    pub assume_mpp: bool,
    /// Disk cache budget; 0 turns it off.
    pub disk_cache_mb: u64,
    /// The overlay button's key; empty for none.
    pub overlay_key: String,
}

impl Default for Prefs {
    fn default() -> Self {
        Self {
            opacity: 0.8,
            markers: DEFAULT_MARKERS.map(String::from).to_vec(),
            hidden: Vec::new(),
            mode: DisplayMode::Overlay,
            batch_size: 4,
            threads: 0,
            model_path: None,
            runtime_path: None,
            assume_mpp: false,
            disk_cache_mb: 2048,
            overlay_key: DEFAULT_OVERLAY_KEY.to_string(),
        }
    }
}

pub mod keys {
    pub const OPACITY: &str = "opacity";
    pub const MARKERS: &str = "markers";
    pub const HIDDEN: &str = "hidden_markers";
    pub const MODE: &str = "display_mode";
    pub const BATCH: &str = "batch_size";
    pub const THREADS: &str = "threads";
    pub const MODEL: &str = "model_path";
    pub const RUNTIME: &str = "runtime_path";
    pub const ASSUME_MPP: &str = "assume_mpp";
    pub const DISK_CACHE: &str = "disk_cache_mb";
    pub const OVERLAY_KEY: &str = "overlay_key";
}

impl Prefs {
    /// Reads the settings the host kept; a bad value falls back to its
    /// default and is reported.
    pub fn from_settings(settings: &Map<String, Value>) -> (Prefs, Vec<String>) {
        let mut prefs = Prefs::default();
        let mut problems = Vec::new();
        let mut bad = |key: &str, value: &Value| {
            problems.push(format!("ignored the setting {key} = {value}"));
        };
        for (key, value) in settings {
            match key.as_str() {
                keys::OPACITY => match value.as_f64() {
                    Some(v) if (0.0..=1.0).contains(&v) => prefs.opacity = (v as f32).max(0.1),
                    _ => bad(key, value),
                },
                keys::MARKERS => match serde_json::from_value::<Vec<String>>(value.clone()) {
                    Ok(list) if !list.is_empty() && list.len() <= MAX_MARKERS => {
                        prefs.markers = list
                    }
                    _ => bad(key, value),
                },
                keys::HIDDEN => match serde_json::from_value::<Vec<String>>(value.clone()) {
                    Ok(list) => prefs.hidden = list,
                    _ => bad(key, value),
                },
                keys::MODE => match serde_json::from_value::<DisplayMode>(value.clone()) {
                    Ok(mode) => prefs.mode = mode,
                    _ => bad(key, value),
                },
                keys::BATCH => match value.as_u64() {
                    Some(n @ 1..=32) => prefs.batch_size = n as usize,
                    _ => bad(key, value),
                },
                keys::THREADS => match value.as_u64() {
                    Some(n @ 0..=64) => prefs.threads = n as usize,
                    _ => bad(key, value),
                },
                keys::MODEL => match value.as_str() {
                    Some(path) if !path.is_empty() => prefs.model_path = Some(path.into()),
                    _ => bad(key, value),
                },
                keys::RUNTIME => match value.as_str() {
                    Some(path) if !path.is_empty() => prefs.runtime_path = Some(path.into()),
                    _ => bad(key, value),
                },
                keys::ASSUME_MPP => match value.as_bool() {
                    Some(flag) => prefs.assume_mpp = flag,
                    _ => bad(key, value),
                },
                keys::DISK_CACHE => match value.as_u64() {
                    Some(mb) => prefs.disk_cache_mb = mb,
                    _ => bad(key, value),
                },
                // A character or a named key such as "F2"; eov refuses keys
                // it or another plugin uses.
                keys::OVERLAY_KEY => match value.as_str().map(str::trim) {
                    Some(name) if name.chars().count() <= 16 => {
                        prefs.overlay_key = name.to_string()
                    }
                    _ => bad(key, value),
                },
                _ => {}
            }
        }
        (prefs, problems)
    }

    pub fn is_visible(&self, marker: &str) -> bool {
        !self.hidden.iter().any(|hidden| hidden == marker)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use eov_plugin_api::prelude::json;

    #[test]
    fn settings_are_read_one_key_at_a_time() {
        let mut settings = Map::new();
        settings.insert("opacity".into(), json!(0.5));
        settings.insert("markers".into(), json!(["Hoechst", "CD8a"]));
        settings.insert("batch_size".into(), json!(1000));
        settings.insert("display_mode".into(), json!("fluorescence"));
        settings.insert("overlay_key".into(), json!(" F7 "));
        settings.insert("unrelated".into(), json!(true));
        let (prefs, problems) = Prefs::from_settings(&settings);
        assert_eq!(prefs.opacity, 0.5);
        assert_eq!(prefs.markers, vec!["Hoechst", "CD8a"]);
        assert_eq!(prefs.mode, DisplayMode::Fluorescence);
        assert_eq!(prefs.batch_size, Prefs::default().batch_size);
        assert_eq!(prefs.overlay_key, "F7");
        assert_eq!(problems.len(), 1, "{problems:?}");
        // No key at all, or a value that is no key.
        for (value, key) in [
            (json!(""), ""),
            (json!(3), "x"),
            (json!("x".repeat(17)), "x"),
        ] {
            let settings = Map::from_iter([("overlay_key".to_string(), value)]);
            assert_eq!(Prefs::from_settings(&settings).0.overlay_key, key);
        }
    }

    #[test]
    fn defaults_are_the_three_validated_markers() {
        let prefs = Prefs::default();
        assert_eq!(prefs.markers, vec!["Hoechst", "Pan-CK", "SMA"]);
        assert!(prefs.is_visible("SMA"));
        assert!(!prefs.assume_mpp, "unknown pixel sizes must be opted into");
    }
}
