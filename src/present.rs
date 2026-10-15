//! What the user sees: the badges over each pane — the research-use
//! notice, the legend, progress and warnings (CEL-9) — the cells' values
//! on hover, the readout region, the panel, the status bar, and the
//! copyable readout.

use crate::blend::TissueCell;
use crate::config::{PANEL_BUTTON, RESEARCH_NOTICE};
use crate::engine::{
    FileState, ModelState, PaneDemand, State, Support, cell_object, marks_layer, region_layer,
};
use crate::geometry::{Grid, Idx, SCALE_TOLERANCE, ScaleSource};
use crate::markers::{self, DisplayMode};
use crate::region::{BlockUse, Readout, ReadoutState, block_use, utc_timestamp};
use eov_plugin_api::prelude::*;
use std::collections::BTreeMap;
use std::hash::{Hash, Hasher};

/// The research-use notice, over every pane showing a slide.
pub const NOTICE_BADGE: &str = "1-notice";
/// The legend, or why there is nothing to show.
pub const MODEL_BADGE: &str = "2-model";
/// The host shows at most 8 badges per plugin: the two above, and one per
/// slide on screen for this many slides.
pub const SLIDE_BADGES: usize = 6;
/// The host's limit on a badge's length, in characters.
pub const BADGE_CHARS: usize = 200;

/// A slide's own badge: its progress and warnings.
pub fn slide_badge(file: FileId) -> String {
    format!("3-slide-{file}")
}

const TEXT: Rgba = Rgba(0xFF, 0xFF, 0xFF, 0xFF);
const WARNING: Rgba = Rgba(0xFF, 0xC1, 0x07, 0xFF);
const PENDING: Rgba = Rgba(0x28, 0x3C, 0x5A, 0x30);
const FAILED: Rgba = Rgba(0xE5, 0x39, 0x35, 0xFF);
const REGION: Rgba = Rgba(0x00, 0xC8, 0xFF, 0xFF);

/// "about 3 min".
pub fn duration(seconds: f64) -> String {
    if seconds < 90.0 {
        format!("about {} s", seconds.round().max(1.0))
    } else if seconds < 90.0 * 60.0 {
        format!("about {} min", (seconds / 60.0).round())
    } else {
        format!("about {:.1} h", seconds / 3600.0)
    }
}

pub fn region_style() -> Style {
    Style {
        fill: Rgba::TRANSPARENT,
        stroke: REGION,
        stroke_width: 2.0,
        text_color: TEXT,
        ..Style::default()
    }
}

/// The values layer draws nothing; it is there for its tooltips.
pub fn values_style() -> Style {
    Style {
        fill: Rgba::TRANSPARENT,
        stroke: Rgba::TRANSPARENT,
        stroke_width: 0.0,
        ..Style::default()
    }
}

/// A cell's predicted values, as the tooltip of an invisible rectangle
/// over it: each marker's mean over the cell's tissue blocks, the same
/// blocks a region readout measures. Cells with no such block get none.
pub fn cell_values(
    cell: Idx,
    rect: Rect,
    tissue: &TissueCell,
    names: &[String],
    threshold: Option<f32>,
) -> Option<OverlayObject> {
    let mut sums = vec![0f64; names.len()];
    let (mut blocks, mut dark, mut unusual) = (0usize, 0usize, 0usize);
    for block in &tissue.blocks {
        match block_use(block) {
            BlockUse::Skip => {}
            BlockUse::Dark => dark += 1,
            BlockUse::Tissue => {
                blocks += 1;
                for (sum, mean) in sums.iter_mut().zip(&block.means) {
                    *sum += f64::from(*mean);
                }
                if let (Some(recon), Some(threshold)) = (block.recon, threshold)
                    && recon > threshold
                {
                    unusual += 1;
                }
            }
        }
    }
    if blocks == 0 || names.is_empty() {
        return None;
    }
    let values: Vec<String> = names
        .iter()
        .zip(&sums)
        .map(|(name, sum)| format!("{name} {:.2}", sum / blocks as f64))
        .collect();
    let mut text = format!(
        "Predicted \u{2014} research use only\n{}\nMean arcsinh intensity over {blocks} tissue \
         blocks",
        values.join(" \u{b7} ")
    );
    if unusual > 0 {
        text.push_str(&format!(", {unusual} unlike the training data"));
    }
    if dark > 0 {
        text.push_str(&format!("; {dark} near-black blocks left out"));
    }
    Some(
        OverlayObject::new(
            cell_object(cell),
            Shape::Rectangle {
                x: rect.x,
                y: rect.y,
                width: rect.width,
                height: rect.height,
            },
        )
        .with_tooltip(text)
        .with_editable(false),
    )
}

/// Merges cells into rectangles: runs along each row, then identical runs
/// in consecutive rows. Returns `(i0, i1, j0, j1)`, end-exclusive.
pub fn merge_cells(cells: &[Idx]) -> Vec<(i64, i64, i64, i64)> {
    let mut rows: BTreeMap<i64, Vec<i64>> = BTreeMap::new();
    for cell in cells {
        rows.entry(cell.j).or_default().push(cell.i);
    }
    let mut runs: Vec<(i64, i64, i64)> = Vec::new();
    for (j, mut columns) in rows {
        columns.sort_unstable();
        columns.dedup();
        let mut start = columns[0];
        let mut previous = start;
        for &i in &columns[1..] {
            if i != previous + 1 {
                runs.push((start, previous + 1, j));
                start = i;
            }
            previous = i;
        }
        runs.push((start, previous + 1, j));
    }
    let mut merged: Vec<(i64, i64, i64, i64)> = Vec::new();
    for (i0, i1, j) in runs {
        if let Some(open) = merged
            .iter_mut()
            .find(|(a0, a1, _, b1)| *a0 == i0 && *a1 == i1 && *b1 == j)
        {
            open.3 = j + 1;
        } else {
            merged.push((i0, i1, j, j + 1));
        }
    }
    merged
}

fn rect_shape(grid: &Grid, (i0, i1, j0, j1): (i64, i64, i64, i64)) -> Shape {
    let a = grid.cell_rect(Idx::new(i0, j0));
    let b = grid.cell_rect(Idx::new(i1 - 1, j1 - 1));
    Shape::Rectangle {
        x: a.x,
        y: a.y,
        width: b.x + b.width - a.x,
        height: b.y + b.height - a.y,
    }
}

fn text(id: String, x: f64, y: f64, label: String, colour: Rgba) -> OverlayObject {
    OverlayObject::new(id, Shape::Point { x, y })
        .with_style(Style {
            fill: Rgba::TRANSPARENT,
            stroke: Rgba::TRANSPARENT,
            point_radius: 0.0,
            text_color: colour,
            ..Style::default()
        })
        .with_label(label)
        .with_editable(false)
}

/// Short notes on how a slide was matched to the model.
pub fn scale_notes(st: &State, file: &FileState) -> Vec<String> {
    let Some(model) = st.model.info() else {
        return Vec::new();
    };
    slide_notes(
        model,
        file.grid().map(|grid| &**grid),
        &file.info,
        WindowCounts {
            dark: file.dark_windows,
            tissue: file.tissue_windows,
            unusual: file.unusual_windows,
        },
    )
}

/// What a slide's windows were found to be, for its notes.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct WindowCounts {
    /// Skipped as not bright-field.
    pub dark: usize,
    pub tissue: usize,
    /// Tissue windows above the model's reconstruction-error threshold.
    pub unusual: usize,
}

/// Short notes on how a slide was matched to the model and what its
/// windows showed: shown in its badge and panel, and in batch summaries.
pub fn slide_notes(
    model: &crate::model::ModelInfo,
    grid: Option<&Grid>,
    info: &FileInfo,
    windows: WindowCounts,
) -> Vec<String> {
    let mut notes = Vec::new();
    if let Some(grid) = grid {
        match grid.scale.source {
            ScaleSource::Assumed => notes.push(format!(
                "pixel size unknown \u{2014} assumed {:.3} \u{b5}m/px",
                model.mpp
            )),
            ScaleSource::OneAxis => notes.push("pixel size reported for one axis only".into()),
            ScaleSource::Reported => {}
        }
        let up = grid.scale.upsampling(model.mpp);
        if up > SCALE_TOLERANCE {
            notes.push(format!(
                "upsampled \u{d7}{up:.2} from {:.3} \u{b5}m/px \u{2014} less reliable",
                grid.scale.mpp_x.max(grid.scale.mpp_y)
            ));
        }
        if grid.scale.is_anisotropic() {
            notes.push("non-square pixels, resampled per axis".into());
        }
    }
    if let Some(stain) = &info.stain {
        let lower = stain.to_ascii_lowercase();
        if !(lower.contains("h&e")
            || lower.contains("hematoxylin")
            || lower.contains("haematoxylin"))
        {
            notes.push(format!(
                "the slide says {stain}; the model was trained on H&E"
            ));
        }
    }
    if windows.dark > 0 {
        notes.push(format!(
            "{} windows skipped as not bright-field",
            windows.dark
        ));
    }
    if windows.unusual > 0 {
        notes.push(format!(
            "{} of {} windows look unlike the training data",
            windows.unusual, windows.tissue
        ));
    }
    if !model.mpp_recorded {
        notes.push("the model does not record its pixel size".into());
    }
    notes
}

/// A line of text, and whether it warns.
type Line = (String, bool);

/// Why nothing can be predicted on any slide, while that is so.
fn model_line(st: &State) -> Option<Line> {
    match &st.model {
        ModelState::Failed(error) => Some((
            format!("Not computed: {} (see the Cellaris panel)", error.summary),
            true,
        )),
        ModelState::Loading | ModelState::Idle => Some(("Loading the model\u{2026}".into(), false)),
        ModelState::Ready(_) => st
            .markers
            .as_ref()
            .err()
            .map(|problem| (format!("Not computed: {problem}"), true)),
    }
}

/// The markers shown and their colours, by name: a badge has no swatches.
fn legend_line(st: &State) -> Option<String> {
    let markers = st.markers.as_ref().ok()?;
    let shown: Vec<String> = markers
        .iter()
        .filter(|m| st.prefs.is_visible(&m.name))
        .map(|m| format!("{}: {}", m.name, markers::colour_name(m.colour)))
        .collect();
    if shown.is_empty() {
        return Some("Every marker is hidden (see the Cellaris panel)".into());
    }
    let hidden = markers.len() - shown.len();
    let mut line = shown.join(" \u{b7} ");
    if hidden > 0 {
        line.push_str(&format!(" ({hidden} hidden)"));
    }
    Some(line)
}

/// Where a slide's predictions stand, over every pane showing it (a badge
/// belongs to a slide, and shows in each of its panes).
fn slide_line(st: &State, file: &FileState) -> Option<Line> {
    if let Some(problem) = file.problem() {
        return Some((format!("Not computed: {}", short_problem(problem)), true));
    }
    if st.model.info().is_none() || st.markers.is_err() {
        // The model's line says why.
        return None;
    }
    let left = file.pending.len();
    if left > 0 {
        let done = file.in_view.saturating_sub(left);
        let mut line = format!("Computing {done} of {} cells", file.in_view);
        let failed = file.failed.len();
        if failed > 0 {
            // Only failures of cells in view count: others are not retried.
            line.push_str(&format!(
                " \u{b7} {failed} failed{}",
                if file.stuck > 0 {
                    " (Retry in the Cellaris panel)"
                } else {
                    ", retrying"
                }
            ));
        } else {
            line.push_str(&format!(
                " \u{b7} {} left",
                duration(left as f64 * st.throughput.seconds_per_window())
            ));
        }
        return Some((line, failed > 0));
    }
    let too_far = file
        .panes
        .values()
        .filter_map(|demand| match demand {
            PaneDemand::TooFar { cells } => Some(*cells),
            _ => None,
        })
        .max()?;
    let which = if file
        .panes
        .values()
        .any(|demand| matches!(demand, PaneDemand::Active))
    {
        "one view of this slide"
    } else {
        "this view"
    };
    Some((
        format!(
            "Zoom in to predict: {which} needs {too_far} cells (limit {})",
            crate::engine::MAX_VIEW_CELLS
        ),
        true,
    ))
}

fn short_problem(problem: &crate::geometry::ScaleProblem) -> String {
    match problem {
        crate::geometry::ScaleProblem::Unknown => "pixel size unknown".into(),
        crate::geometry::ScaleProblem::TooCoarse { slide_mpp, .. } => {
            format!("slide too coarse ({slide_mpp:.2} \u{b5}m/px)")
        }
        crate::geometry::ScaleProblem::BadSlide(reason) => reason.clone(),
    }
}

/// At most [`BADGE_CHARS`] characters, ending in an ellipsis if cut.
fn clip(text: &str) -> String {
    if text.chars().count() <= BADGE_CHARS {
        return text.to_string();
    }
    let mut out: String = text.chars().take(BADGE_CHARS - 1).collect();
    out.push('\u{2026}');
    out
}

/// A slide's badge: its line and its notes.
fn slide_text(st: &State, file: &FileState) -> Option<String> {
    let mut parts: Vec<String> = slide_line(st, file)
        .map(|(line, _)| line)
        .into_iter()
        .collect();
    parts.extend(scale_notes(st, file));
    (!parts.is_empty()).then(|| clip(&parts.join(" \u{b7} ")))
}

/// The badges to show while the overlay is on: id → (slide, text). The
/// host shows them at the top of the panes, fixed to the pane, in id order;
/// a slide's own badge shows only in the panes showing that slide.
pub fn badges(st: &State) -> BTreeMap<String, (Option<FileId>, String)> {
    let mut out = BTreeMap::new();
    if !st.overlay {
        return out;
    }
    out.insert(
        NOTICE_BADGE.to_string(),
        (None, format!("{RESEARCH_NOTICE} \u{b7} Cellaris")),
    );
    if let Some(line) = model_line(st)
        .map(|(line, _)| line)
        .or_else(|| legend_line(st))
    {
        out.insert(MODEL_BADGE.to_string(), (None, clip(&line)));
    }
    // The slides on screen, the focused pane's first.
    let focused = st
        .focused_pane
        .and_then(|pane| st.views.get(&pane))
        .map(|view| view.file_id);
    let mut shown: Vec<FileId> = Vec::new();
    for id in focused
        .into_iter()
        .chain(st.views.values().map(|view| view.file_id))
    {
        if st.files.contains_key(&id) && !shown.contains(&id) {
            shown.push(id);
        }
    }
    for id in shown.into_iter().take(SLIDE_BADGES) {
        if let Some(text) = slide_text(st, &st.files[&id]) {
            out.insert(slide_badge(id), (Some(id), text));
        }
    }
    out
}

/// Brings the host's badges up to date, sending only what changed. A host
/// without badges is noted, and the slides carry the notice instead.
pub fn refresh_badges(host: &Host, st: &mut State) {
    if st.badges == Support::No {
        return;
    }
    let wanted = badges(st);
    let gone: Vec<String> = st
        .shown_badges
        .keys()
        .filter(|id| !wanted.contains_key(*id))
        .cloned()
        .collect();
    let mut sends: Vec<(String, Option<FileId>, Option<String>)> =
        gone.into_iter().map(|id| (id, None, None)).collect();
    for (id, (file, text)) in wanted {
        if st.shown_badges.get(&id) != Some(&(file, text.clone())) {
            sends.push((id, file, Some(text)));
        }
    }
    for (id, file, text) in sends {
        match host.set_badge(&id, file, text.as_deref()) {
            Err(error) if error.is_unsupported() => {
                host.info(
                    "Cellaris draws its research-use notice on the slide: this eov has no badges",
                );
                st.badges = Support::No;
                st.shown_badges.clear();
                return;
            }
            Err(error) => host.debug(format!("Cellaris could not set badge {id}: {error}")),
            Ok(()) => st.badges = Support::Yes,
        }
        // Not sent again until it changes, even after an error.
        match text {
            Some(text) => {
                st.shown_badges.insert(id, (file, text));
            }
            None => {
                st.shown_badges.remove(&id);
            }
        }
    }
}

/// The objects of a file's marks layer: areas still computing are veiled,
/// and areas that failed are outlined. On hosts without badges the notice,
/// progress, notes and legend are drawn here too.
pub fn marks(st: &State, id: FileId) -> Vec<OverlayObject> {
    let mut objects = Vec::new();
    let Some(file) = st.files.get(&id) else {
        return objects;
    };
    if !st.overlay {
        return objects;
    }
    if let Some(grid) = file.grid() {
        let failed: Vec<Idx> = file.failed.clone();
        let pending: Vec<Idx> = file
            .pending
            .iter()
            .filter(|c| !failed.contains(c))
            .copied()
            .collect();
        for (n, run) in merge_cells(&pending).into_iter().enumerate() {
            objects.push(
                OverlayObject::new(format!("pending-{n}"), rect_shape(grid, run))
                    .with_style(Style {
                        fill: PENDING,
                        stroke: Rgba::TRANSPARENT,
                        stroke_width: 0.0,
                        ..Style::default()
                    })
                    .with_editable(false),
            );
        }
        for (n, run) in merge_cells(&failed).into_iter().enumerate() {
            objects.push(
                OverlayObject::new(format!("failed-{n}"), rect_shape(grid, run))
                    .with_style(Style {
                        fill: Rgba::TRANSPARENT,
                        stroke: FAILED,
                        stroke_width: 1.5,
                        dash: Dash::Dashed,
                        ..Style::default()
                    })
                    .with_editable(false),
            );
        }
    }
    if st.badges == Support::No {
        in_slide_badge(st, id, file, &mut objects);
    }
    objects
}

/// The badge drawn on the slide, for hosts without badges. A layer belongs
/// to a slide and is drawn in every pane showing it, so there is one, in
/// the focused pane when it shows this slide, else in the first that does;
/// it follows the view once the view settles.
fn in_slide_badge(st: &State, id: FileId, file: &FileState, objects: &mut Vec<OverlayObject>) {
    let shows = |pane: &PaneId| st.views.get(pane).is_some_and(|v| v.file_id == id);
    let pane = st.focused_pane.filter(shows).or_else(|| {
        st.views
            .iter()
            .find(|(_, v)| v.file_id == id)
            .map(|(pane, _)| *pane)
    });
    let Some((pane, view)) = pane.and_then(|pane| st.views.get_key_value(&pane)) else {
        return;
    };
    let zoom = view.zoom.max(1e-9);
    let x = view.visible.x + 12.0 / zoom;
    let mut y = view.visible.y + 14.0 / zoom;
    let step = 22.0 / zoom;
    objects.push(text(
        format!("badge-{pane}"),
        x,
        y,
        format!("{RESEARCH_NOTICE} \u{b7} Cellaris"),
        TEXT,
    ));
    y += step;
    if let Some((line, warning)) = model_line(st).or_else(|| slide_line(st, file)) {
        let colour = if warning { WARNING } else { TEXT };
        objects.push(text(format!("status-{pane}"), x, y, line, colour));
        y += step;
    }
    let notes = scale_notes(st, file);
    if !notes.is_empty() {
        objects.push(text(
            format!("notes-{pane}"),
            x,
            y,
            notes.join(" \u{b7} "),
            WARNING,
        ));
        y += step;
    }
    if let Ok(markers) = &st.markers {
        for (k, marker) in markers.iter().enumerate() {
            if !st.prefs.is_visible(&marker.name) {
                continue;
            }
            let [r, g, b] = marker.colour;
            objects.push(
                OverlayObject::new(
                    format!("legend-{pane}-{k}"),
                    Shape::Point {
                        x: x + 6.0 / zoom,
                        y: y + 2.0 / zoom,
                    },
                )
                .with_style(Style {
                    fill: Rgba::rgb(r, g, b),
                    stroke: Rgba(0x11, 0x13, 0x18, 0xFF),
                    stroke_width: 1.0,
                    point_radius: 6.0,
                    text_color: TEXT,
                    ..Style::default()
                })
                .with_label(format!("{} (predicted)", marker.name))
                .with_editable(false),
            );
            y += 20.0 / zoom;
        }
    }
}

/// The objects of a file's region layer.
pub fn region_objects(st: &State, id: FileId) -> Vec<OverlayObject> {
    let Some(file) = st.files.get(&id) else {
        return Vec::new();
    };
    let Some(region) = &file.region else {
        return Vec::new();
    };
    let label = match file.readout.as_ref().map(|r| &r.readout) {
        Some(readout) if readout.is_running() => format!(
            "Cellaris region \u{b7} measuring {:.0} %",
            readout.progress() * 100.0
        ),
        _ => "Cellaris region".to_string(),
    };
    let mut style = region_style();
    if file
        .readout
        .as_ref()
        .is_some_and(|r| r.readout.is_running())
    {
        style.dash = Dash::Dashed;
    }
    vec![
        OverlayObject::new("region", region.to_shape())
            .with_style(style)
            .with_label(label)
            .with_editable(true),
    ]
}

fn fingerprint(objects: &[OverlayObject]) -> u64 {
    let mut hasher = std::collections::hash_map::DefaultHasher::new();
    serde_json::to_string(objects)
        .unwrap_or_default()
        .hash(&mut hasher);
    hasher.finish()
}

/// Updates a file's marks and region layers if they changed.
pub fn refresh_file(host: &Host, st: &mut State, id: FileId) {
    let marks = marks(st, id);
    let region = region_objects(st, id);
    let Some(file) = st.files.get_mut(&id) else {
        return;
    };
    if !file.layers {
        return;
    }
    let print = fingerprint(&marks);
    if file.marks_print != Some(print) {
        file.marks_print = Some(print);
        let _ = host.layer(marks_layer(id)).replace_all(marks);
    }
    let print = fingerprint(&region);
    if file.region_print != Some(print) {
        file.region_print = Some(print);
        let _ = host.layer(region_layer(id)).replace_all(region);
    }
}

/// Everything shown, brought up to date.
pub fn refresh(host: &Host, st: &mut State) {
    // First: a host without badges gets the notice in the marks layers.
    refresh_badges(host, st);
    let ids: Vec<FileId> = st.files.keys().copied().collect();
    for id in ids {
        refresh_file(host, st, id);
        if let Some(file) = st.files.get(&id)
            && let Some(job) = &file.readout
            && let Some(host_job) = &job.job
            && job.readout.is_running()
        {
            host_job.update(
                Some(job.readout.progress()),
                Some(&format!(
                    "{} of {} cells",
                    job.readout.cells.len() - job.readout.remaining.len(),
                    job.readout.cells.len()
                )),
            );
        }
    }
    let status = status_line(st);
    if st.last_status.as_ref() != Some(&status) {
        host.set_status(status.as_deref());
        st.last_status = Some(status);
    }
    let open = st.sidebar.is_some();
    if st.last_panel_open != Some(open) {
        let _ = host.set_toolbar_button_active(PANEL_BUTTON, open);
        st.last_panel_open = Some(open);
    }
    if let Some(sidebar) = &st.sidebar {
        let properties = sidebar_properties(st);
        if st.last_sidebar.as_ref() != Some(&properties) {
            if sidebar.set_many(properties.clone()).is_ok() {
                st.last_sidebar = Some(properties);
            } else {
                // The host closed it.
                st.sidebar = None;
                st.last_sidebar = None;
            }
        }
    }
}

/// The plugin's status-bar segment.
pub fn status_line(st: &State) -> Option<String> {
    match &st.model {
        ModelState::Loading => return Some("Cellaris: loading the model\u{2026}".into()),
        ModelState::Failed(error) if st.overlay => {
            return Some(format!("Cellaris: {}", error.summary));
        }
        _ => {}
    }
    let mut cells = 0usize;
    let mut left = 0usize;
    let mut failed = 0usize;
    for file in st.files.values() {
        cells += file.in_view;
        left += file.pending.len();
        failed += file.failed.len();
    }
    let readouts = st
        .files
        .values()
        .filter(|f| f.readout.as_ref().is_some_and(|r| r.readout.is_running()))
        .count();
    if !st.overlay && readouts == 0 {
        return None;
    }
    let mut parts = vec!["Cellaris (predicted, research use only)".to_string()];
    if left > 0 {
        parts.push(format!(
            "{}/{cells} cells, {:.1} s per window",
            cells.saturating_sub(left),
            st.throughput.seconds_per_window()
        ));
    }
    if failed > 0 {
        parts.push(format!("{failed} failed"));
    }
    if readouts > 0 {
        parts.push("measuring a region".into());
    }
    Some(parts.join(" \u{b7} "))
}

fn rows(readout: &Readout, st: &State) -> Vec<Value> {
    let Ok(markers) = &st.markers else {
        return Vec::new();
    };
    if readout.blocks() == 0 {
        return Vec::new();
    }
    markers
        .iter()
        .zip(&readout.fits)
        .map(|(marker, fit)| {
            json!({
                "name": marker.name,
                "colour": markers::hex(marker.colour),
                "mean": format!("{:.2}", fit.mean()),
                "spread": format!("{:.2}", fit.spread()),
                "density": fit.r2().map_or("\u{2014}".to_string(), |r2| format!("{r2:.2}")),
            })
        })
        .collect()
}

/// Tissue and region areas of a readout, in mm².
fn areas(readout: &Readout, grid: &Grid) -> (f64, f64) {
    let tissue = readout.blocks() as f64 * grid.block_area_um2() / 1e6;
    let region = readout.region.area() * grid.pixel_area_um2() / 1e6;
    (tissue, region)
}

/// The readout part of the panel for the active slide.
fn readout_properties(st: &State, out: &mut Map<String, Value>) {
    let file = st.active_file.and_then(|id| st.files.get(&id));
    let job = file.and_then(|f| f.readout.as_ref());
    let has_region = file.is_some_and(|f| f.region.is_some());
    let waiting = match (&st.model, &st.markers) {
        (ModelState::Ready(_), Err(problem)) => format!("Cannot measure: {problem}"),
        _ => "Waiting for the model\u{2026}".to_string(),
    };
    let (status, detail, progress, rows, running, can_copy) = match job {
        None if has_region => (waiting, String::new(), 0.0, Vec::new(), false, false),
        None => (
            "Pick the Cellaris region tool and outline some tissue, or measure the region of \
             interest."
                .to_string(),
            String::new(),
            0.0,
            Vec::new(),
            false,
            false,
        ),
        Some(job) => {
            let readout = &job.readout;
            let grid = file.and_then(FileState::grid);
            let blocks = readout.blocks();
            let area = grid.map(|g| areas(readout, g));
            let summary = match area {
                Some((tissue, region)) => format!(
                    "{blocks} tissue blocks, {tissue:.3} mm\u{b2} of tissue in {region:.3} mm\u{b2}"
                ),
                None => format!("{blocks} tissue blocks"),
            };
            let status = match &readout.state {
                ReadoutState::Confirming => "Waiting for you to confirm a long measurement.".into(),
                ReadoutState::Running => format!(
                    "Measuring: {} of {} cells \u{b7} {summary}",
                    readout.cells.len() - readout.remaining.len(),
                    readout.cells.len()
                ),
                ReadoutState::Done => summary.clone(),
                ReadoutState::Cancelled => format!("Cancelled \u{b7} {summary}"),
                ReadoutState::Failed(message) => message.clone(),
            };
            let mut details = Vec::new();
            if readout.is_running() {
                let left = readout.remaining.len() as f64 * st.throughput.seconds_per_window();
                details.push(format!("{} left on this computer.", duration(left)));
            } else if blocks > 0 && blocks < crate::region::MIN_BLOCKS {
                details.push(format!(
                    "Fewer than {} tissue blocks, too few for a density r\u{b2}.",
                    crate::region::MIN_BLOCKS
                ));
            }
            if let Some(recon) = readout.mean_recon() {
                let unusual = if readout.unusual_blocks > 0 {
                    format!(
                        ", {:.0} % of blocks above the model's threshold",
                        100.0 * readout.unusual_blocks as f64 / readout.recon_blocks as f64
                    )
                } else {
                    String::new()
                };
                details.push(format!("Mean reconstruction error {recon:.3}{unusual}."));
            }
            if readout.dark_cells > 0 || readout.dark_blocks > 0 {
                details.push(format!(
                    "{} blocks and {} cells skipped as near-black (not bright-field H&E).",
                    readout.dark_blocks, readout.dark_cells
                ));
            }
            if let Some(file) = file {
                details.extend(
                    scale_notes(st, file)
                        .into_iter()
                        .map(|n| format!("Note: {n}.")),
                );
            }
            (
                status,
                details.join(" "),
                readout.progress(),
                rows(readout, st),
                readout.is_running() || readout.state == ReadoutState::Confirming,
                blocks > 0 && !readout.is_running(),
            )
        }
    };
    out.insert("readout-status".into(), json!(status));
    out.insert("readout-detail".into(), json!(detail));
    out.insert("readout-progress".into(), json!(progress));
    out.insert("readout-rows".into(), json!(rows));
    out.insert("has-region".into(), json!(has_region));
    out.insert("readout-running".into(), json!(running));
    out.insert("can-copy".into(), json!(can_copy));
}

fn model_card(st: &State) -> Vec<Value> {
    let row = |key: &str, value: String| json!({ "key": key, "value": value });
    let mut card = Vec::new();
    match &st.model {
        ModelState::Ready(info) => {
            card.push(row(
                "Model",
                info.title.clone().unwrap_or_else(|| info.file_name()),
            ));
            if let Some(version) = &info.version {
                card.push(row("Version", version.clone()));
            }
            card.push(row("File", info.path.display().to_string()));
            card.push(row(
                "SHA-256",
                format!("{} ({:.1} MB)", info.short_hash(), info.bytes as f64 / 1e6),
            ));
            card.push(row(
                "Trained at",
                format!(
                    "{:.3} \u{b5}m/px{}",
                    info.mpp,
                    if info.mpp_recorded { "" } else { " (assumed)" }
                ),
            ));
            card.push(row(
                "Input",
                format!(
                    "{0}\u{d7}{0} px windows, {1}",
                    info.window,
                    match info.batch {
                        crate::model::Batch::Dynamic => "any batch size".to_string(),
                        crate::model::Batch::Fixed(n) => format!("batches of {n}"),
                    }
                ),
            ));
            card.push(row(
                "Channels",
                format!(
                    "{} ({})",
                    info.channel_names.len(),
                    info.channel_names.join(", ")
                ),
            ));
            card.push(row(
                "Outputs",
                format!(
                    "{}{}",
                    if info.raw_intensity {
                        "raw intensities, shown as arcsinh(x/5)"
                    } else {
                        "arcsinh intensities"
                    },
                    if info.recon {
                        ", reconstruction error"
                    } else {
                        ""
                    }
                ),
            ));
            if let Some(training) = &info.training_data {
                card.push(row("Trained on", training.clone()));
            }
            if let Some(license) = &info.license {
                card.push(row("Licence", license.clone()));
            }
            if let Some(description) = &info.description {
                card.push(row("About", description.clone()));
            }
            if let Some(producer) = &info.producer {
                card.push(row("Exported by", producer.clone()));
            }
            card.push(row(
                "Runtime",
                format!("{}, {} threads", info.runtime, info.threads),
            ));
            for note in &info.notes {
                card.push(row("Note", note.clone()));
            }
        }
        _ => {
            card.push(row(
                "Model",
                st.prefs.model_path.clone().unwrap_or_else(|| {
                    st.paths
                        .data_dir
                        .join(crate::model::MODEL_FILE)
                        .display()
                        .to_string()
                }),
            ));
        }
    }
    if st.runtime_restart {
        card.push(row(
            "Runtime",
            "the new choice takes effect when eov restarts".into(),
        ));
    }
    card
}

/// Said when a pane shows the slide with its colours adjusted: the model
/// still sees the slide's own pixels.
pub fn display_note(st: &State, id: FileId) -> Option<String> {
    let shows = |pane: &PaneId| st.views.get(pane).is_some_and(|v| v.file_id == id);
    let display = st
        .focused_pane
        .filter(shows)
        .into_iter()
        .chain(st.views.keys().copied().filter(shows))
        .filter_map(|pane| st.displays.get(&pane))
        .find(|display| display.modified)?;
    let what = display.badge.trim();
    Some(if what.is_empty() {
        "the display is adjusted; predictions use the slide's unadjusted colours".to_string()
    } else {
        format!("the display is adjusted ({what}); predictions use the slide's unadjusted colours")
    })
}

/// The panel's properties.
pub fn sidebar_properties(st: &State) -> Map<String, Value> {
    let mut out = Map::new();
    out.insert("notice".into(), json!(RESEARCH_NOTICE));
    out.insert("overlay-on".into(), json!(st.overlay));
    let (status, level, detail, retry) = match &st.model {
        ModelState::Idle => (
            "The model loads when the overlay is turned on or a region is measured.".to_string(),
            0,
            String::new(),
            false,
        ),
        ModelState::Loading => (
            "Loading the model\u{2026}".to_string(),
            1,
            String::new(),
            false,
        ),
        ModelState::Failed(error) => (error.summary.clone(), 3, error.detail.clone(), true),
        ModelState::Ready(info) => match &st.markers {
            Err(problem) => (problem.clone(), 3, String::new(), false),
            Ok(_) => {
                let failed: usize = st.files.values().map(|f| f.failed.len()).sum();
                if failed > 0 {
                    let error = st
                        .files
                        .values()
                        .flat_map(|f| f.failures.values())
                        .map(|f| f.error.clone())
                        .next()
                        .unwrap_or_default();
                    (
                        format!("{failed} cells in view could not be computed"),
                        2,
                        error,
                        true,
                    )
                } else if st.busy {
                    (
                        format!(
                            "Computing \u{b7} {:.1} s per window",
                            st.throughput.seconds_per_window()
                        ),
                        1,
                        String::new(),
                        false,
                    )
                } else {
                    (
                        format!(
                            "Ready: {}",
                            info.title.clone().unwrap_or_else(|| info.file_name())
                        ),
                        0,
                        String::new(),
                        false,
                    )
                }
            }
        },
    };
    let mut detail = detail;
    for problem in &st.setting_problems {
        if !detail.is_empty() {
            detail.push('\n');
        }
        detail.push_str(problem);
    }
    out.insert("status".into(), json!(status));
    out.insert("status-level".into(), json!(level));
    out.insert("status-detail".into(), json!(detail));
    out.insert("can-retry".into(), json!(retry));

    let file = st.active_file.and_then(|id| st.files.get(&id));
    let (slide, notes) = match file {
        None => ("No slide".to_string(), String::new()),
        Some(file) => {
            let scale = match file.grid() {
                Some(grid) => format!(
                    "{:.3} \u{d7} {:.3} \u{b5}m/px, read from level {} (\u{d7}{:.2})",
                    grid.scale.mpp_x, grid.scale.mpp_y, grid.level, grid.downsample
                ),
                None => match (file.info.mpp_x, file.info.mpp_y) {
                    (Some(x), Some(y)) => format!("{x:.3} \u{d7} {y:.3} \u{b5}m/px"),
                    _ => "pixel size not reported".to_string(),
                },
            };
            let mut notes = vec![scale];
            if let Some(problem) = file.problem() {
                notes.push(problem.message());
            }
            notes.extend(scale_notes(st, file));
            notes.extend(display_note(st, file.info.file_id));
            (file.info.filename.clone(), notes.join("\n"))
        }
    };
    out.insert("slide-name".into(), json!(slide));
    out.insert("slide-notes".into(), json!(notes));

    let (cells, done) = file
        .map(|f| (f.in_view, f.in_view.saturating_sub(f.pending.len())))
        .unwrap_or((0, 0));
    out.insert(
        "progress".into(),
        json!(if cells > 0 && st.overlay {
            done as f64 / cells as f64
        } else {
            -1.0
        }),
    );
    out.insert(
        "progress-text".into(),
        json!(if cells > 0 && st.overlay {
            format!("{done} of {cells} cells in view")
        } else {
            String::new()
        }),
    );
    out.insert("overlay-opacity".into(), json!(st.prefs.opacity));
    out.insert(
        "fluorescence".into(),
        json!(st.prefs.mode == DisplayMode::Fluorescence),
    );
    out.insert("assume-mpp".into(), json!(st.prefs.assume_mpp));
    let rows: Vec<Value> = match &st.markers {
        Ok(markers) => markers
            .iter()
            .map(|m| {
                json!({
                    "name": m.name,
                    "colour": markers::hex(m.colour),
                    "visible": st.prefs.is_visible(&m.name),
                    "window": format!("arcsinh {:.2}\u{2013}{:.2}", m.low, m.high),
                })
            })
            .collect(),
        Err(_) => st
            .prefs
            .markers
            .iter()
            .enumerate()
            .map(|(k, name)| {
                json!({
                    "name": name,
                    "colour": markers::hex(markers::PALETTE[k.min(markers::PALETTE.len() - 1)]),
                    "visible": st.prefs.is_visible(name),
                    "window": "",
                })
            })
            .collect(),
    };
    out.insert("markers".into(), json!(rows));
    out.insert("model-card".into(), json!(model_card(st)));
    readout_properties(st, &mut out);
    out
}

/// The active slide's readout as tab-separated text, with its provenance.
pub fn copy_text(st: &State, id: FileId) -> Option<String> {
    let file = st.files.get(&id)?;
    let job = file.readout.as_ref()?;
    let readout = &job.readout;
    let grid = file.grid()?;
    let model = st.model.info()?;
    let markers = st.markers.as_ref().ok()?;
    let (tissue, region) = areas(readout, grid);
    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default();
    let mut lines = vec![
        format!("# Cellaris region readout \u{2014} {RESEARCH_NOTICE}"),
        format!("# slide\t{}", file.info.filename),
        format!(
            "# model\t{} sha256:{}",
            model.title.clone().unwrap_or_else(|| model.file_name()),
            model.sha256
        ),
        format!(
            "# model pixel size\t{:.4} um/px{}",
            model.mpp,
            if model.mpp_recorded { "" } else { " (assumed)" }
        ),
        format!(
            "# slide pixel size\t{:.4} x {:.4} um/px ({:?})",
            grid.scale.mpp_x, grid.scale.mpp_y, grid.scale.source
        ),
        format!(
            "# windows\t{0}x{0} px at a {1} px stride, feathered; read from level {2}",
            grid.window, grid.stride, grid.level
        ),
        format!("# region area\t{region:.4} mm2"),
        format!(
            "# tissue\t{} blocks of {:.1} um (optical density >= {}), {tissue:.4} mm2",
            readout.blocks(),
            crate::geometry::BLOCK as f64 * model.mpp,
            crate::resample::TISSUE_OD
        ),
        format!("# state\t{:?}", readout.state),
        format!(
            "# generated\t{} by cellaris {}",
            utc_timestamp(now),
            env!("CARGO_PKG_VERSION")
        ),
        "marker\tmean_arcsinh\tspread\tdensity_r2".to_string(),
    ];
    for note in scale_notes(st, file)
        .into_iter()
        .chain(display_note(st, id))
    {
        lines.insert(lines.len() - 1, format!("# note\t{note}"));
    }
    for (marker, fit) in markers.iter().zip(&readout.fits) {
        lines.push(format!(
            "{}\t{:.4}\t{:.4}\t{}",
            marker.name,
            fit.mean(),
            fit.spread(),
            fit.r2().map_or("NA".to_string(), |r| format!("{r:.4}"))
        ));
    }
    Some(lines.join("\n") + "\n")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cells_merge_into_few_rectangles() {
        let mut cells = Vec::new();
        for j in 0..4 {
            for i in 0..5 {
                cells.push(Idx::new(i, j));
            }
        }
        cells.push(Idx::new(9, 0));
        let merged = merge_cells(&cells);
        assert!(merged.contains(&(0, 5, 0, 4)), "{merged:?}");
        assert!(merged.contains(&(9, 10, 0, 1)), "{merged:?}");
        assert_eq!(merged.len(), 2);
        assert!(merge_cells(&[]).is_empty());
    }

    #[test]
    fn durations_read_naturally() {
        assert_eq!(duration(0.2), "about 1 s");
        assert_eq!(duration(600.0), "about 10 min");
        assert_eq!(duration(3.0 * 3600.0), "about 3.0 h");
    }
}

/// The panel's callbacks, as `ui/cellaris.slint` declares them.
pub const PANEL_CALLBACKS: &[&str] = &[
    "toggle-overlay",
    "set-opacity",
    "set-marker-visible",
    "set-fluorescence",
    "set-assume-mpp",
    "measure-roi",
    "draw-region",
    "clear-region",
    "cancel-readout",
    "copy-readout",
    "choose-model",
    "choose-runtime",
    "retry",
    "close-panel",
];

/// A state with a loaded model, a slide, a view and a finished readout, for
/// checking the panel against what the plugin sends it.
#[doc(hidden)]
pub fn example_state() -> State {
    use crate::config::Prefs;
    use crate::engine::{FileState, Paths, ReadoutJob};
    use crate::model::{Batch, ModelInfo};
    use crate::region::{Readout, Region};
    use std::sync::Arc;

    let mut st = State::new(Prefs::default(), Paths::default());
    let names: Vec<String> = ["Hoechst", "AF1", "Pan-CK", "SMA"]
        .map(String::from)
        .to_vec();
    st.markers = markers::select(&names, &st.prefs.markers);
    st.model = ModelState::Ready(Arc::new(ModelInfo {
        path: "/models/markers.onnx".into(),
        bytes: 123_456_789,
        sha256: "0123456789abcdef".repeat(4),
        channel_names: names,
        raw_intensity: true,
        mpp: 0.325,
        mpp_recorded: false,
        window: 256,
        batch: Batch::Dynamic,
        recon: true,
        recon_threshold: Some(0.1),
        title: Some("cellaris markers".into()),
        version: Some("1".into()),
        description: None,
        producer: Some("pytorch".into()),
        training_data: Some("Orion CRC".into()),
        license: None,
        runtime: "ONNX Runtime 1.22.0 from /usr/lib/libonnxruntime.so".into(),
        threads: 4,
        notes: vec!["the training pixel size is not recorded".into()],
    }));
    st.overlay = true;
    let info = FileInfo {
        file_id: 1,
        filename: "slide.svs".into(),
        width: 50_000,
        height: 40_000,
        mpp_x: Some(0.25),
        mpp_y: Some(0.25),
        ..Default::default()
    };
    let mut file = FileState::new(info.clone());
    let model = st.model.info().cloned();
    if let Some(model) = model {
        let scale = crate::geometry::Scale::of(&info, model.mpp, false).ok();
        file.grid = scale.map(|scale| {
            crate::geometry::Grid::new(&info, scale, model.mpp, model.window).map(Arc::new)
        });
    }
    let region = Region::Rectangle(Rect {
        x: 1000.0,
        y: 1000.0,
        width: 2000.0,
        height: 2000.0,
    });
    let mut readout = Readout::new(1, region.clone(), vec![], 3, 10);
    for k in 0..40 {
        for fit in &mut readout.fits {
            fit.add(0.1 + k as f64 * 0.01, 5.0 + (k % 7) as f64 * 0.1);
        }
    }
    readout.state = crate::region::ReadoutState::Done;
    file.region = Some(region);
    file.readout = Some(ReadoutJob {
        readout,
        job: None,
        confirm: None,
        started: std::time::Instant::now(),
    });
    file.panes.insert(1, PaneDemand::Active);
    file.in_view = 100;
    file.pending = (0..60).map(|i| Idx::new(i, 0)).collect();
    st.files.insert(1, file);
    st.active_file = Some(1);
    st.views.insert(
        1,
        ViewInfo {
            pane: 1,
            file_id: 1,
            zoom: 0.5,
            width: 800.0,
            height: 600.0,
            visible: Rect {
                x: 0.0,
                y: 0.0,
                width: 1600.0,
                height: 1200.0,
            },
            ..Default::default()
        },
    );
    st
}

#[cfg(test)]
mod panel_tests {
    use super::*;

    /// `in property <type> name` declarations of the panel's root.
    fn declared(kind: &str) -> Vec<String> {
        let source = include_str!("../ui/cellaris.slint");
        let root = source
            .split_once("export component CellarisPanel")
            .map_or(source, |(_, root)| root);
        root.lines()
            .map(str::trim)
            .filter_map(|line| match kind {
                "property" => line
                    .strip_prefix("in property <")
                    .and_then(|rest| rest.split_once('>'))
                    .map(|(_, rest)| rest.trim().trim_end_matches(';')),
                _ => line
                    .strip_prefix("callback ")
                    .map(|rest| rest.split('(').next().unwrap_or(rest)),
            })
            .map(|name| name.split(':').next().unwrap_or(name).trim().to_string())
            .collect()
    }

    #[test]
    fn the_panel_declares_every_property_and_callback_used() {
        let properties = declared("property");
        for state in [
            example_state(),
            State::new(Default::default(), Default::default()),
        ] {
            for name in sidebar_properties(&state).keys() {
                assert!(
                    properties.contains(name),
                    "the panel has no property {name}"
                );
            }
        }
        let callbacks = declared("callback");
        for name in PANEL_CALLBACKS {
            assert!(callbacks.iter().any(|c| c == name), "no callback {name}");
        }
        assert_eq!(callbacks.len(), PANEL_CALLBACKS.len(), "{callbacks:?}");
    }

    #[test]
    fn the_badges_say_predicted_and_research_use_only_and_name_the_colours() {
        let st = example_state();
        let shown = badges(&st);
        let (file, notice) = &shown[NOTICE_BADGE];
        assert_eq!(*file, None, "the notice is over every pane");
        assert!(notice.contains("PREDICTED") && notice.contains("research use only"));
        // The legend names each visible marker's colour: there are no
        // swatches in a badge.
        assert_eq!(
            shown[MODEL_BADGE],
            (
                None,
                "Hoechst: yellow \u{b7} Pan-CK: magenta \u{b7} SMA: bluish green".to_string()
            )
        );
        // The slide's own badge, only over panes showing it.
        let (file, line) = &shown[&slide_badge(1)];
        assert_eq!(*file, Some(1));
        assert!(
            line.starts_with("Computing 40 of 100 cells \u{b7} about"),
            "{line}"
        );
        assert!(
            line.contains("the model does not record its pixel size"),
            "{line}"
        );
        // Nothing is drawn on the slide but what belongs to it.
        assert!(marks(&st, 1).iter().all(|o| o.label.is_none()));

        let mut hidden = example_state();
        hidden.prefs.hidden = vec!["Pan-CK".into()];
        assert_eq!(
            badges(&hidden)[MODEL_BADGE].1,
            "Hoechst: yellow \u{b7} SMA: bluish green (1 hidden)"
        );
        let mut off = example_state();
        off.overlay = false;
        assert!(badges(&off).is_empty());
        assert!(marks(&off, 1).is_empty());
    }

    #[test]
    fn badges_stay_within_the_hosts_limits() {
        let mut st = example_state();
        let template = st.files.remove(&1).unwrap();
        // Ten slides on screen, each with long notes.
        for id in 1..=10 {
            let info = FileInfo {
                file_id: id,
                filename: format!("slide-{id}.svs"),
                stain: Some(format!(
                    "{} immunohistochemistry",
                    "very long stain name ".repeat(8)
                )),
                ..template.info.clone()
            };
            let mut file = FileState::new(info);
            file.grid = template.grid.clone();
            file.dark_windows = 123_456;
            file.unusual_windows = 7;
            file.tissue_windows = 9;
            st.files.insert(id, file);
            let pane = id as PaneId;
            st.views.insert(
                pane,
                ViewInfo {
                    pane,
                    file_id: id,
                    ..st.views[&1]
                },
            );
        }
        st.focused_pane = Some(9);
        let shown = badges(&st);
        assert!(shown.len() <= 8, "{} badges", shown.len());
        for (id, (_, text)) in &shown {
            assert!(!id.is_empty() && id.len() <= 64, "{id}");
            let count = text.trim().chars().count();
            assert!(
                (1..=BADGE_CHARS).contains(&count),
                "{id}: {count} characters"
            );
        }
        assert!(
            shown.contains_key(&slide_badge(9)),
            "the focused slide comes first"
        );
        assert!(shown[&slide_badge(9)].1.ends_with('\u{2026}'));
    }

    #[test]
    fn hosts_without_badges_get_the_notice_drawn_on_the_slide() {
        let mut st = example_state();
        st.badges = Support::No;
        let objects = marks(&st, 1);
        let badge = objects
            .iter()
            .find(|o| o.object_id == "badge-1")
            .expect("no badge in view");
        let label = badge.label.as_deref().unwrap_or_default();
        assert!(label.contains("PREDICTED") && label.contains("research use only"));
        let status = objects
            .iter()
            .find(|o| o.object_id == "status-1")
            .and_then(|o| o.label.as_deref())
            .unwrap_or_default();
        assert!(status.starts_with("Computing 40 of 100"), "{status}");
        // A legend entry per visible marker, in its colour.
        let legend: Vec<_> = objects
            .iter()
            .filter(|o| o.object_id.starts_with("legend-1-"))
            .collect();
        assert_eq!(legend.len(), 3);
        assert_eq!(
            legend[0].style.as_ref().unwrap().fill,
            Rgba::rgb(0xF0, 0xE4, 0x42)
        );
        // The badge sits at the top left of what the pane shows.
        let Shape::Point { x, y } = badge.shape else {
            panic!("the badge is not a point");
        };
        assert!(x > 0.0 && x < 100.0 && y > 0.0 && y < 100.0, "({x}, {y})");
        assert!(badges(&st).len() > 1, "badges are still worked out");
    }

    #[test]
    fn a_cells_tooltip_gives_its_predicted_values() {
        use crate::blend::BlockStat;
        let block = |od: f32, dark: f32, hoechst: f32, recon: f32| BlockStat {
            od,
            coverage: 1.0,
            dark,
            means: vec![hoechst, 5.0],
            recon: Some(recon),
        };
        let tissue = TissueCell {
            planes: Vec::new(),
            mask: Vec::new(),
            blocks: vec![
                block(0.3, 0.0, 6.0, 0.05),
                block(0.3, 0.0, 7.0, 0.5),
                // Glass, and a near-black block: left out, as in a readout.
                block(0.01, 0.0, 99.0, 0.05),
                block(0.3, 0.9, 99.0, 0.05),
            ],
        };
        let names = ["Hoechst".to_string(), "SMA".to_string()];
        let rect = Rect {
            x: 10.0,
            y: 20.0,
            width: 30.0,
            height: 30.0,
        };
        let object = cell_values(Idx::new(3, 4), rect, &tissue, &names, Some(0.1)).unwrap();
        assert_eq!(object.object_id, "c3_4");
        assert_eq!(
            object.shape,
            Shape::Rectangle {
                x: 10.0,
                y: 20.0,
                width: 30.0,
                height: 30.0
            }
        );
        let tooltip = object.tooltip.unwrap();
        assert!(
            tooltip.starts_with("Predicted \u{2014} research use only\n"),
            "{tooltip}"
        );
        assert!(
            tooltip.contains("Hoechst 6.50 \u{b7} SMA 5.00"),
            "{tooltip}"
        );
        assert!(
            tooltip.contains("over 2 tissue blocks, 1 unlike the training data"),
            "{tooltip}"
        );
        assert!(
            tooltip.contains("1 near-black blocks left out"),
            "{tooltip}"
        );
        // A cell without a measurable block has no tooltip.
        let glass = TissueCell {
            blocks: vec![block(0.01, 0.0, 1.0, 0.0)],
            ..tissue
        };
        assert!(cell_values(Idx::new(0, 0), rect, &glass, &names, None).is_none());
    }

    #[test]
    fn an_adjusted_display_is_noted_where_the_slide_is_described() {
        let mut st = example_state();
        assert_eq!(display_note(&st, 1), None);
        st.displays.insert(
            1,
            DisplayAdjustments {
                modified: true,
                badge: "Colour-normalised (Macenko)".into(),
                ..Default::default()
            },
        );
        let note = display_note(&st, 1).unwrap();
        assert!(note.contains("(Colour-normalised (Macenko))"), "{note}");
        assert!(note.contains("unadjusted colours"), "{note}");
        let notes = sidebar_properties(&st)["slide-notes"]
            .as_str()
            .unwrap()
            .to_string();
        assert!(notes.contains(&note), "{notes}");
        let text = copy_text(&st, 1).unwrap();
        assert!(text.contains(&format!("# note\t{note}")), "{text}");
        // A pane showing another slide does not count.
        assert_eq!(display_note(&st, 2), None);
    }

    #[test]
    fn the_copied_readout_carries_its_provenance() {
        let st = example_state();
        let text = copy_text(&st, 1).unwrap();
        assert!(text.starts_with("# Cellaris region readout"));
        for needle in [
            "research use only",
            "slide.svs",
            "sha256:0123456789abcdef",
            "0.3250 um/px (assumed)",
            "0.2500 x 0.2500 um/px",
            "marker\tmean_arcsinh\tspread\tdensity_r2",
            "\nHoechst\t",
        ] {
            assert!(text.contains(needle), "missing {needle:?} in\n{text}");
        }
    }
}
