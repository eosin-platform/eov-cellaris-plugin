//! What the user sees: the badges over each pane — the research-use
//! notice, the legend, progress and warnings (CEL-9) — the cells' values
//! on hover, the readout region, the panel, the status bar, and the
//! copyable readout.

use crate::blend::TissueCell;
use crate::config::{PANEL_BUTTON, RESEARCH_NOTICE, research_notice};
use crate::engine::{
    FileState, ModelState, PaneDemand, State, Support, cell_object, marks_layer, region_layer,
};
use crate::geometry::{Grid, Idx, SCALE_TOLERANCE, ScaleSource};
use crate::markers::{self, DisplayMode};
use crate::region::{BlockUse, Readout, ReadoutState, block_use, utc_timestamp};
use crate::text::{Text, decimal, isolate, text, writing_english};
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
pub fn duration(seconds: f64) -> Text {
    if seconds < 90.0 {
        let seconds = seconds.round().max(1.0) as u64;
        // Translators: how long something will still take; "s" is seconds.
        text!(tr!("about {n} s" | "about {n} s" % seconds))
    } else if seconds < 90.0 * 60.0 {
        let minutes = (seconds / 60.0).round() as u64;
        // Translators: how long something will still take; "min" is minutes.
        text!(tr!("about {n} min" | "about {n} min" % minutes))
    } else {
        let hours = seconds / 3600.0;
        // Translators: how long something will still take, in hours ("h").
        text!(tr!("about {} h", decimal(hours, 1)))
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
        .map(|(name, sum)| format!("{name} {}", decimal(sum / blocks as f64, 2)))
        .collect();
    let mut mean = tr!(
        // Translators: in a cell's tooltip; a tissue block is a 5 µm square
        // of tissue, not a paraffin block.
        "Mean predicted arcsinh intensity over {n} tissue block"
            | "Mean predicted arcsinh intensity over {n} tissue blocks" % blocks
    );
    if unusual > 0 {
        let unusual = tr!(
            // Translators: in a cell's tooltip: {n} of the tissue blocks look
            // unlike the slides the model learnt from.
            "{n} unlike the training data" | "{n} unlike the training data" % unusual
        );
        // Translators: joins two parts of a cell's tooltip: "Mean predicted
        // arcsinh intensity over 2 tissue blocks" and "1 unlike the training
        // data".
        mean = tr!("{mean}, {unusual}", mean, unusual);
    }
    if dark > 0 {
        // Translators: in a cell's tooltip: tissue blocks (5 µm squares of
        // tissue, not paraffin blocks) too dark to be bright-field H&E.
        let dark = tr!("{n} near-black block left out" | "{n} near-black blocks left out" % dark);
        // Translators: joins two parts of a cell's tooltip: the mean of the
        // predicted values ("Mean predicted arcsinh intensity over 2 tissue
        // blocks", …) and "1 near-black block left out".
        mean = tr!("{mean}; {dark}", mean, dark);
    }
    let text = format!(
        "{}\n{}\n{mean}",
        // Translators: the first line of a cell's tooltip. Keep all of its
        // meaning: the values are a model's prediction, for research only.
        tr!("Predicted \u{2014} research use only"),
        values.join(" \u{b7} ")
    );
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

/// `object` labelled with `label`: in the language of the interface on
/// screen, and in English where eov burns the label into images (exports,
/// copies of the view), whose font has Latin and Greek letters only.
fn labelled(object: OverlayObject, label: &Text) -> OverlayObject {
    object
        .with_label(label.to_string())
        .with_export_label(label.english())
}

fn text(id: String, x: f64, y: f64, label: &Text, colour: Rgba) -> OverlayObject {
    let object = OverlayObject::new(id, Shape::Point { x, y }).with_style(Style {
        fill: Rgba::TRANSPARENT,
        stroke: Rgba::TRANSPARENT,
        point_radius: 0.0,
        text_color: colour,
        ..Style::default()
    });
    labelled(object, label).with_editable(false)
}

/// Short notes on how a slide was matched to the model.
pub fn scale_notes(st: &State, file: &FileState) -> Vec<Text> {
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
///
/// The notes are [`Text`]s: in the language of the interface where they are
/// shown, in English (`Text::english`) in copied readouts and batch
/// summaries.
pub fn slide_notes(
    model: &crate::model::ModelInfo,
    grid: Option<&Grid>,
    info: &FileInfo,
    windows: WindowCounts,
) -> Vec<Text> {
    let mut notes = Vec::new();
    if let Some(grid) = grid {
        match grid.scale.source {
            ScaleSource::Assumed => {
                let mpp = model.mpp;
                notes.push(text!(tr!(
                    // Translators: a note on a slide; µm/px is micrometres per
                    // pixel.
                    "pixel size unknown \u{2014} assumed {} \u{b5}m/px",
                    decimal(mpp, 3)
                )));
            }
            ScaleSource::OneAxis => {
                // Translators: a note on a slide: its pixel size is known for
                // its width or its height only.
                notes.push(text!(tr!("pixel size reported for one axis only")));
            }
            ScaleSource::Reported => {}
        }
        let up = grid.scale.upsampling(model.mpp);
        if up > SCALE_TOLERANCE {
            let from = grid.scale.mpp_x.max(grid.scale.mpp_y);
            notes.push(text!(tr!(
                // Translators: a note on a slide: it was enlarged {factor} times
                // from its pixel size, {mpp} micrometres per pixel.
                "upsampled \u{d7}{factor} from {mpp} \u{b5}m/px \u{2014} less reliable",
                factor = decimal(up, 2),
                mpp = decimal(from, 3)
            )));
        }
        if grid.scale.is_anisotropic() {
            // Translators: a note on a slide whose pixels are not square.
            notes.push(text!(tr!("non-square pixels, resampled per axis")));
        }
    }
    if let Some(stain) = &info.stain {
        let lower = stain.to_ascii_lowercase();
        if !(lower.contains("h&e")
            || lower.contains("hematoxylin")
            || lower.contains("haematoxylin"))
        {
            let stain = stain.clone();
            notes.push(text!(tr!(
                // Translators: a note on a slide; {} is the stain its metadata
                // names (not translated); H&E is haematoxylin and eosin.
                "the slide says {}; the model was trained on H&E",
                isolate(&stain)
            )));
        }
    }
    if windows.dark > 0 {
        let dark = windows.dark;
        notes.push(text!(tr!(
            // Translators: a note on a slide; a window is one of the square
            // images the model predicts from.
            "{n} window skipped as not bright-field"
                | "{n} windows skipped as not bright-field" % dark
        )));
    }
    if windows.unusual > 0 {
        let (unusual, tissue) = (windows.unusual, windows.tissue);
        notes.push(text!(tr!(
            // Translators: a note on a slide: {n} of its {total} windows with
            // tissue look unlike the slides the model learnt from. The form
            // follows {n}, the windows that do; {total} is a bare number.
            "{n} window of {total} looks unlike the training data"
                | "{n} windows of {total} look unlike the training data" % unusual,
            total = tissue
        )));
    }
    if !model.mpp_recorded {
        // Translators: a note on a slide: the model does not say at which
        // pixel size it was trained.
        notes.push(text!(tr!("the model does not record its pixel size")));
    }
    notes
}

/// A line of text, and whether it warns. A [`Text`]: badges show it in the
/// language of the interface, and on hosts without badges it is a label on
/// the slide, which images get in English.
type Line = (Text, bool);

/// Why nothing can be predicted on any slide, while that is so.
fn model_line(st: &State) -> Option<Line> {
    match &st.model {
        ModelState::Failed(error) => {
            let summary = error.summary.clone();
            Some((
                // Translators: over the slide; {} says why nothing was predicted.
                text!(tr!("Not computed: {} (see the Cellaris panel)", summary)),
                true,
            ))
        }
        ModelState::Loading | ModelState::Idle => {
            Some((text!(tr!("Loading the model\u{2026}")), false))
        }
        ModelState::Ready(_) => st.markers.as_ref().err().map(|problem| {
            let problem = problem.clone();
            // Translators: over the slide; {} says why nothing was predicted.
            (text!(tr!("Not computed: {}", problem)), true)
        }),
    }
}

/// The markers shown and their colours, by name: a badge has no swatches.
fn legend_line(st: &State) -> Option<String> {
    let markers = st.markers.as_ref().ok()?;
    let shown: Vec<String> = markers
        .iter()
        .filter(|m| st.prefs.is_visible(&m.name))
        .map(|m| {
            let (marker, colour) = (isolate(&m.name), markers::colour_name(m.colour));
            // An adjusted window is named: brightness then depends on it.
            if m.is_adjusted() {
                // Translators: a legend entry over the slide, such as
                // "Pan-CK: magenta (arcsinh 4.50–6.50)": a marker, its colour,
                // and the display window the user set for it.
                tr!(
                    "{marker}: {colour} ({window})",
                    marker,
                    colour,
                    window = m.window_text()
                )
            } else {
                // Translators: a legend entry over the slide, such as
                // "Hoechst: yellow": a marker and its colour.
                tr!("{marker}: {colour}", marker, colour)
            }
        })
        .collect();
    if shown.is_empty() {
        return Some(tr!("Every marker is hidden (see the Cellaris panel)"));
    }
    let hidden = markers.len() - shown.len();
    let line = shown.join(" \u{b7} ");
    if hidden > 0 {
        // Translators: {markers} is the legend over the slide; {n} markers are
        // hidden.
        return Some(tr!(
            "{markers} ({n} hidden)" | "{markers} ({n} hidden)" % hidden,
            markers = line
        ));
    }
    Some(line)
}

/// Where a slide's predictions stand, over every pane showing it (a badge
/// belongs to a slide, and shows in each of its panes).
fn slide_line(st: &State, file: &FileState) -> Option<Line> {
    if let Some(problem) = file.problem() {
        let problem = short_problem(problem);
        // Translators: over the slide; {} says why nothing was predicted.
        return Some((text!(tr!("Not computed: {}", problem)), true));
    }
    if st.model.info().is_none() || st.markers.is_err() {
        // The model's line says why.
        return None;
    }
    let left = file.pending.len();
    if left > 0 {
        let (done, in_view) = (file.in_view.saturating_sub(left), file.in_view);
        let computing = text!(tr!(
            // Translators: over the slide; a cell is a small square of the
            // slide that is predicted at once.
            "Computing {done} of {n} cell" | "Computing {done} of {n} cells" % in_view,
            done
        ));
        let failed = file.failed.len();
        // Only failures of cells in view count: others are not retried.
        let more = if failed > 0 && file.stuck > 0 {
            text!(tr!(
                // Translators: {n} cells could not be computed; Retry is a
                // button in the Cellaris panel.
                "{n} failed (Retry in the Cellaris panel)"
                    | "{n} failed (Retry in the Cellaris panel)" % failed
            ))
        } else if failed > 0 {
            // Translators: {n} cells could not be computed and are tried again.
            text!(tr!("{n} failed, retrying" | "{n} failed, retrying" % failed))
        } else {
            let time = duration(left as f64 * st.throughput.seconds_per_window());
            // Translators: {} is a duration such as "about 3 min".
            text!(tr!("{} left", time))
        };
        let line = Text::new(move || format!("{computing} \u{b7} {more}"));
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
    let limit = crate::engine::MAX_VIEW_CELLS;
    let line = if file
        .panes
        .values()
        .any(|demand| matches!(demand, PaneDemand::Active))
    {
        text!(tr!(
            // Translators: over the slide, when it is shown in several views:
            // one of them shows too much of it to predict ({n} cells; at most
            // {limit}). A cell is one square of the grid the slide is predicted
            // in, not a biological cell.
            "Zoom in to predict: one view of this slide needs {n} cell (limit {limit})"
                | "Zoom in to predict: one view of this slide needs {n} cells (limit {limit})"
                    % too_far,
            limit
        ))
    } else {
        text!(tr!(
            // Translators: over the slide: the view shows too much of it to
            // predict ({n} cells; at most {limit}). A cell is one square of the
            // grid the slide is predicted in, not a biological cell.
            "Zoom in to predict: this view needs {n} cell (limit {limit})"
                | "Zoom in to predict: this view needs {n} cells (limit {limit})" % too_far,
            limit
        ))
    };
    Some((line, true))
}

fn short_problem(problem: &crate::geometry::ScaleProblem) -> Text {
    match problem {
        // Translators: why a slide is not predicted (over the slide).
        crate::geometry::ScaleProblem::Unknown => text!(tr!("pixel size unknown")),
        crate::geometry::ScaleProblem::TooCoarse { slide_mpp, .. } => {
            let slide_mpp = *slide_mpp;
            // Translators: why a slide is not predicted (over the slide); µm/px
            // is micrometres per pixel.
            text!(tr!(
                "slide too coarse ({} \u{b5}m/px)",
                decimal(slide_mpp, 2)
            ))
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
        .map(|(line, _)| line.to_string())
        .into_iter()
        .collect();
    parts.extend(scale_notes(st, file).iter().map(Text::to_string));
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
        (None, format!("{} \u{b7} Cellaris", research_notice())),
    );
    if let Some(line) = model_line(st)
        .map(|(line, _)| line.to_string())
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
    let notice = Text::new(|| {
        let notice = if writing_english() {
            RESEARCH_NOTICE.to_string()
        } else {
            research_notice()
        };
        format!("{notice} \u{b7} Cellaris")
    });
    objects.push(text(format!("badge-{pane}"), x, y, &notice, TEXT));
    y += step;
    if let Some((line, warning)) = model_line(st).or_else(|| slide_line(st, file)) {
        let colour = if warning { WARNING } else { TEXT };
        objects.push(text(format!("status-{pane}"), x, y, &line, colour));
        y += step;
    }
    let notes = scale_notes(st, file);
    if !notes.is_empty() {
        let notes = Text::new(move || {
            let notes: Vec<String> = notes.iter().map(Text::to_string).collect();
            notes.join(" \u{b7} ")
        });
        objects.push(text(format!("notes-{pane}"), x, y, &notes, WARNING));
        y += step;
    }
    if let Ok(markers) = &st.markers {
        for (k, marker) in markers.iter().enumerate() {
            if !st.prefs.is_visible(&marker.name) {
                continue;
            }
            let [r, g, b] = marker.colour;
            let label = if marker.is_adjusted() {
                let marker = marker.clone();
                text!(tr!(
                    // Translators: a legend entry drawn on the slide: a
                    // marker, and the display window the user set for it.
                    "{marker} (predicted, {window})",
                    marker = isolate(&marker.name),
                    window = marker.window_text()
                ))
            } else {
                let name = marker.name.clone();
                // Translators: a legend entry drawn on the slide: a marker.
                text!(tr!("{} (predicted)", isolate(&name)))
            };
            let entry = OverlayObject::new(
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
            });
            objects.push(labelled(entry, &label).with_editable(false));
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
        Some(readout) if readout.is_running() => {
            let percent = format!("{:.0}", readout.progress() * 100.0);
            text!(tr!(
                // Translators: the label drawn on a region while its readout is
                // computed; {} is how much of it is done, in percent.
                "Cellaris region \u{b7} {} % computed",
                percent
            ))
        }
        // Translators: the label drawn on the region of a readout (a readout
        // summarises predicted values; it is not a measurement).
        _ => text!(tr!("Cellaris region")),
    };
    let mut style = region_style();
    if file
        .readout
        .as_ref()
        .is_some_and(|r| r.readout.is_running())
    {
        style.dash = Dash::Dashed;
    }
    let region = OverlayObject::new("region", region.to_shape()).with_style(style);
    vec![labelled(region, &label).with_editable(true)]
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
            let total = job.readout.cells.len();
            let done = total - job.readout.remaining.len();
            host_job.update(
                Some(job.readout.progress()),
                Some(&tr!(
                    // Translators: the progress of a region readout: {done} of the
                    // region's {n} cells (squares of the grid the slide is
                    // predicted in, not biological cells).
                    "{done} of {n} cell" | "{done} of {n} cells" % total,
                    done
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
        ModelState::Loading => return Some(tr!("Cellaris: loading the model\u{2026}")),
        ModelState::Failed(error) if st.overlay => {
            // Translators: the status bar while the overlay is on; {} says
            // what went wrong.
            return Some(tr!("Cellaris: {}", error.summary));
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
    // Translators: in the status bar while predictions are shown. Keep all
    // of its meaning: the colours are predicted, for research only.
    let mut parts = vec![tr!("Cellaris (predicted, research use only)")];
    if left > 0 {
        parts.push(tr!(
            // Translators: in the status bar: {done} of {n} cells computed, and
            // how long each window (square image the model predicts from)
            // takes, in seconds. A cell is one square of the grid the slide is
            // predicted in, not a biological cell.
            "{done}/{n} cell, {seconds} s per window"
                | "{done}/{n} cells, {seconds} s per window" % cells,
            done = cells.saturating_sub(left),
            seconds = decimal(st.throughput.seconds_per_window(), 1)
        ));
    }
    if failed > 0 {
        // Translators: in the status bar: {n} cells could not be computed.
        parts.push(tr!("{n} failed" | "{n} failed" % failed));
    }
    if readouts > 0 {
        // Translators: in the status bar while a region readout is computed.
        parts.push(tr!("computing a region readout"));
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
                "mean": decimal(fit.mean(), 2),
                "spread": decimal(fit.spread(), 2),
                "density": fit.r2().map_or("\u{2014}".to_string(), |r2| decimal(r2, 2)),
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
        // Translators: {} says why the region's readout cannot be computed
        // (such as "no markers are selected").
        (ModelState::Ready(_), Err(problem)) => tr!("Cannot compute the readout: {}", problem),
        _ => tr!("Waiting for the model\u{2026}"),
    };
    let (status, detail, progress, rows, running, can_copy) = match job {
        None if has_region => (waiting, String::new(), 0.0, Vec::new(), false, false),
        None => (
            tr!(
                // Translators: "Cellaris region readout" is the name of the tool
                // (use the same words as there); the region of interest is eov's
                // ROI (the panel's button Use ROI).
                "Outline some tissue with the Cellaris region readout tool, or use the region of interest."
            ),
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
                Some((tissue, region)) => tr!(
                    // Translators: what a region readout covers: {n} tissue blocks
                    // (5 µm squares of tissue, not paraffin blocks), their area and
                    // the region's, in square millimetres.
                    "{n} tissue block, {tissue} mm\u{b2} of tissue in {region} mm\u{b2}"
                        | "{n} tissue blocks, {tissue} mm\u{b2} of tissue in {region} mm\u{b2}"
                            % blocks,
                    tissue = decimal(tissue, 3),
                    region = decimal(region, 3)
                ),
                // Translators: what a region readout covers: {n} tissue blocks (5 µm
                // squares of tissue, not paraffin blocks).
                None => tr!("{n} tissue block" | "{n} tissue blocks" % blocks),
            };
            let status = match &readout.state {
                ReadoutState::Confirming => tr!("Waiting for you to confirm a long readout."),
                ReadoutState::Running => tr!(
                    // Translators: {done} of the region's {n} cells (squares of the
                    // grid the slide is predicted in, not biological cells) are
                    // computed; {summary} is what the readout covers so far, such
                    // as "12 tissue blocks, …".
                    "Computing: {done} of {n} cell \u{b7} {summary}"
                        | "Computing: {done} of {n} cells \u{b7} {summary}" % readout.cells.len(),
                    done = readout.cells.len() - readout.remaining.len(),
                    summary = summary.clone()
                ),
                ReadoutState::Done => summary.clone(),
                // Translators: {} is what the readout covered until it was
                // cancelled, such as "12 tissue blocks, …".
                ReadoutState::Cancelled => tr!("Cancelled \u{b7} {}", summary),
                ReadoutState::Failed(message) => message.to_string(),
            };
            let mut details = Vec::new();
            if readout.is_running() {
                let left = readout.remaining.len() as f64 * st.throughput.seconds_per_window();
                // Translators: {} is a duration such as "about 3 min".
                details.push(tr!("{} left on this computer.", duration(left)));
            } else if blocks > 0 && blocks < crate::region::MIN_BLOCKS {
                details.push(tr!(
                    // Translators: density r² is a column of the readout; a tissue
                    // block is a 5 µm square of tissue, not a paraffin block.
                    "Fewer than {n} tissue block, too few for a density r\u{b2}."
                        | "Fewer than {n} tissue blocks, too few for a density r\u{b2}."
                            % crate::region::MIN_BLOCKS
                ));
            }
            if let Some(recon) = readout.mean_recon() {
                let recon = decimal(recon, 3);
                details.push(if readout.unusual_blocks > 0 {
                    let percent = format!(
                        "{:.0}",
                        100.0 * readout.unusual_blocks as f64 / readout.recon_blocks as f64
                    );
                    tr!(
                        // Translators: the reconstruction error says how unlike the
                        // training data the tissue looks; {percent} is a percentage;
                        // blocks are 5 µm squares of tissue, not paraffin blocks.
                        "Mean reconstruction error {recon}, {percent} % of blocks above the model's threshold.",
                        recon,
                        percent
                    )
                } else {
                    // Translators: the reconstruction error says how unlike the
                    // training data the tissue looks.
                    tr!("Mean reconstruction error {}.", recon)
                });
            }
            if readout.dark_cells > 0 || readout.dark_blocks > 0 {
                let (blocks, cells) = (readout.dark_blocks, readout.dark_cells);
                details.push(tr!(
                    // Translators: {blocks} is a number of blocks ("3 blocks") and
                    // {cells} a number of cells ("1 cell"), both translated
                    // separately.
                    "{blocks} and {cells} skipped as near-black (not bright-field H&E).",
                    // Translators: a number of tissue blocks (5 µm squares of
                    // tissue, not paraffin blocks), in "3 blocks and 1 cell skipped
                    // as near-black".
                    blocks = tr!("{n} block" | "{n} blocks" % blocks),
                    // Translators: a number of cells (squares of the grid the slide
                    // is predicted in, not biological cells), in "3 blocks and 1
                    // cell skipped as near-black".
                    cells = tr!("{n} cell" | "{n} cells" % cells)
                ));
            }
            if let Some(file) = file {
                details.extend(
                    scale_notes(st, file)
                        .into_iter()
                        // Translators: {} is a note on the slide.
                        .map(|n| tr!("Note: {}.", n)),
                );
            }
            (
                status,
                eov_plugin_api::i18n::join_sentences(&details),
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
    let row = |key: String, value: String| json!({ "key": key, "value": value });
    // Translators: the labels of the model card in the Cellaris panel.
    let model = || tr!("Model card" => "Model");
    let runtime = || tr!("Model card" => "Runtime");
    let mut card = Vec::new();
    match &st.model {
        ModelState::Ready(info) => {
            card.push(row(
                model(),
                info.title.clone().unwrap_or_else(|| info.file_name()),
            ));
            if let Some(version) = &info.version {
                card.push(row(tr!("Model card" => "Version"), version.clone()));
            }
            card.push(row(
                tr!("Model card" => "File"),
                info.path.display().to_string(),
            ));
            card.push(row(
                "SHA-256".to_string(),
                // Translators: the start of the model file's SHA-256, and its
                // size in megabytes.
                tr!(
                    "{hash} ({size} MB)",
                    hash = info.short_hash(),
                    size = decimal(info.bytes as f64 / 1e6, 1)
                ),
            ));
            let mpp = decimal(info.mpp, 3);
            card.push(row(
                // Translators: the pixel size the model was trained at.
                tr!("Model card" => "Trained at"),
                if info.mpp_recorded {
                    format!("{mpp} \u{b5}m/px")
                } else {
                    // Translators: a pixel size the model does not record, so it
                    // is assumed; µm/px is micrometres per pixel.
                    tr!("{} \u{b5}m/px (assumed)", mpp)
                },
            ));
            card.push(row(
                // Translators: what the model takes in.
                tr!("Model card" => "Input"),
                match info.batch {
                    crate::model::Batch::Dynamic => tr!(
                        // Translators: the model's input: square windows of {side}
                        // pixels, as many at once as wanted.
                        "{side}\u{d7}{side} px windows, any batch size",
                        side = info.window
                    ),
                    crate::model::Batch::Fixed(n) => tr!(
                        // Translators: the model's input: square windows of {side}
                        // pixels, exactly {n} at once.
                        "{side}\u{d7}{side} px windows, batches of {n}"
                            | "{side}\u{d7}{side} px windows, batches of {n}" % n,
                        side = info.window
                    ),
                },
            ));
            card.push(row(
                // Translators: the markers the model predicts.
                tr!("Model card" => "Channels"),
                format!(
                    "{} ({})",
                    info.channel_names.len(),
                    info.channel_names.join(", ")
                ),
            ));
            card.push(row(
                // Translators: what the model computes.
                tr!("Model card" => "Outputs"),
                match (info.raw_intensity, info.recon) {
                    (true, true) => {
                        // Translators: the model card's Outputs row: the model's
                        // predicted intensities are on a linear ("raw",
                        // untransformed) scale and shown as arcsinh(x/5); "raw" does
                        // not mean measured signal.
                        tr!("raw intensities, shown as arcsinh(x/5), reconstruction error")
                    }
                    // Translators: the model card's Outputs row: the model's predicted
                    // intensities are on a linear ("raw", untransformed) scale and
                    // shown as arcsinh(x/5); "raw" does not mean measured signal.
                    (true, false) => tr!("raw intensities, shown as arcsinh(x/5)"),
                    (false, true) => tr!("arcsinh intensities, reconstruction error"),
                    (false, false) => tr!("arcsinh intensities"),
                },
            ));
            if let Some(training) = &info.training_data {
                // Translators: the data the model was trained on.
                card.push(row(tr!("Model card" => "Trained on"), training.clone()));
            }
            if let Some(license) = &info.license {
                card.push(row(tr!("Model card" => "Licence"), license.clone()));
            }
            if let Some(description) = &info.description {
                // Translators: the model's own description follows.
                card.push(row(tr!("Model card" => "About"), description.clone()));
            }
            if let Some(producer) = &info.producer {
                // Translators: the program that wrote the model file.
                card.push(row(tr!("Model card" => "Exported by"), producer.clone()));
            }
            card.push(row(runtime(), runtime_text(&info.runtime, info.threads)));
            for note in &info.notes {
                card.push(row(tr!("Model card" => "Note"), note.to_string()));
            }
        }
        _ => {
            card.push(row(
                model(),
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
            runtime(),
            tr!("the new choice takes effect when eov restarts"),
        ));
    }
    card
}

/// The model card's runtime: "ONNX Runtime 1.22.0 from /usr/lib/…, 4
/// threads".
fn runtime_text(runtime: &str, threads: usize) -> String {
    match crate::model::runtime_parts(runtime) {
        Some((Some(version), library)) => tr!(
            // Translators: {version} is ONNX Runtime's version, {library} its
            // file; {n} threads compute predictions.
            "ONNX Runtime {version} from {library}, {n} thread"
                | "ONNX Runtime {version} from {library}, {n} threads" % threads,
            version,
            library = isolate(library)
        ),
        Some((None, library)) => tr!(
            // Translators: {library} is ONNX Runtime's file; {n} threads compute
            // predictions.
            "ONNX Runtime (version unknown) from {library}, {n} thread"
                | "ONNX Runtime (version unknown) from {library}, {n} threads" % threads,
            library = isolate(library)
        ),
        None => tr!(
            // Translators: {runtime} names ONNX Runtime and its file; {n}
            // threads compute predictions.
            "{runtime}, {n} thread" | "{runtime}, {n} threads" % threads,
            runtime = isolate(runtime)
        ),
    }
}

/// Said when a pane shows the slide with its colours adjusted: the model
/// still sees the slide's own pixels.
///
/// eov's badge says how: in English for data such as the copied readout
/// (`badge`), and as the window shows it, in the interface's language, in
/// the panel (`shown_badge`; hosts before plugin API 1.1 send only `badge`,
/// in English then too).
pub fn display_note(st: &State, id: FileId) -> Option<Text> {
    let shows = |pane: &PaneId| st.views.get(pane).is_some_and(|v| v.file_id == id);
    let display = st
        .focused_pane
        .filter(shows)
        .into_iter()
        .chain(st.views.keys().copied().filter(shows))
        .filter_map(|pane| st.displays.get(&pane))
        .find(|display| display.modified)?;
    let english = display.badge.trim().to_string();
    let shown = display
        .shown_badge
        .as_deref()
        .map_or(english.as_str(), str::trim)
        .to_string();
    // Which badge, and so which sentence, is decided as the note is written.
    Some(Text::new(move || {
        let what = if writing_english() { &english } else { &shown };
        let note = if what.is_empty() {
            // Translators: a note on a slide: eov shows it with adjusted colours,
            // but the model sees the slide's own.
            text!(tr!(
                "the display is adjusted; predictions use the slide's unadjusted colours"
            ))
        } else {
            let what = what.clone();
            text!(tr!(
                // Translators: a note on a slide: eov shows it with adjusted colours
                // ({} says how, in eov's words), but the model sees the slide's own.
                "the display is adjusted ({}); predictions use the slide's unadjusted colours",
                what
            ))
        };
        note.to_string()
    }))
}

/// The panel's properties.
pub fn sidebar_properties(st: &State) -> Map<String, Value> {
    let mut out = Map::new();
    out.insert("notice".into(), json!(research_notice()));
    out.insert("overlay-on".into(), json!(st.overlay));
    let (status, level, detail, retry) = match &st.model {
        ModelState::Idle => (
            tr!("The model loads when the overlay is turned on or a region readout starts."),
            0,
            String::new(),
            false,
        ),
        ModelState::Loading => (tr!("Loading the model\u{2026}"), 1, String::new(), false),
        ModelState::Failed(error) => (error.summary.to_string(), 3, error.detail.to_string(), true),
        ModelState::Ready(info) => match &st.markers {
            Err(problem) => (problem.to_string(), 3, String::new(), false),
            Ok(_) => {
                let failed: usize = st.files.values().map(|f| f.failed.len()).sum();
                if failed > 0 {
                    let error = st
                        .files
                        .values()
                        .flat_map(|f| f.failures.values())
                        .map(|f| f.error.to_string())
                        .next()
                        .unwrap_or_default();
                    (
                        tr!(
                            // Translators: cells are the small squares of the slide
                            // that are predicted at once.
                            "{n} cell in view could not be computed"
                                | "{n} cells in view could not be computed" % failed
                        ),
                        2,
                        error,
                        true,
                    )
                } else if st.busy {
                    (
                        tr!(
                            // Translators: {} is how long each window (square image
                            // the model predicts from) takes, in seconds.
                            "Computing \u{b7} {} s per window",
                            decimal(st.throughput.seconds_per_window(), 1)
                        ),
                        1,
                        String::new(),
                        false,
                    )
                } else {
                    let model = info.title.clone().unwrap_or_else(|| info.file_name());
                    (
                        // Translators: {} is the name of the model that is loaded.
                        tr!("Ready: {}", isolate(&model)),
                        0,
                        String::new(),
                        false,
                    )
                }
            }
        },
    };
    let mut detail = detail;
    for problem in st.setting_problems.iter().chain(st.refused.values()) {
        if !detail.is_empty() {
            detail.push('\n');
        }
        detail.push_str(&problem.to_string());
    }
    out.insert("status".into(), json!(status));
    out.insert("status-level".into(), json!(level));
    out.insert("status-detail".into(), json!(detail));
    out.insert("can-retry".into(), json!(retry));

    let file = st.active_file.and_then(|id| st.files.get(&id));
    let (slide, notes) = match file {
        None => (tr!("No slide"), String::new()),
        Some(file) => {
            let scale = match file.grid() {
                Some(grid) => tr!(
                    // Translators: the slide's pixel size in micrometres per pixel
                    // (µm/px), the pyramid level it is read from, and how much
                    // smaller that level is (×4.00).
                    "{x} \u{d7} {y} \u{b5}m/px, read from level {level} (\u{d7}{downsample})",
                    x = decimal(grid.scale.mpp_x, 3),
                    y = decimal(grid.scale.mpp_y, 3),
                    level = grid.level,
                    downsample = decimal(grid.downsample, 2)
                ),
                None => match (file.info.mpp_x, file.info.mpp_y) {
                    (Some(x), Some(y)) => {
                        format!("{} \u{d7} {} \u{b5}m/px", decimal(x, 3), decimal(y, 3))
                    }
                    // Translators: the slide does not say its pixel size.
                    _ => tr!("pixel size not reported"),
                },
            };
            let mut notes = vec![scale];
            if let Some(problem) = file.problem() {
                notes.push(problem.message().to_string());
            }
            notes.extend(scale_notes(st, file).iter().map(Text::to_string));
            notes.extend(display_note(st, file.info.file_id).map(|note| note.to_string()));
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
            tr!(
                // Translators: under the progress bar in the Cellaris panel; a
                // cell is one square of the grid the slide is predicted in, not a
                // biological cell.
                "{done} of {n} cell in view" | "{done} of {n} cells in view" % cells,
                done
            )
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
    let adjusted = |name: &str, (low, high): (f32, f32)| {
        markers::default_window(name)
            .is_some_and(|(l, h)| (l - low).abs() > 1e-4 || (h - high).abs() > 1e-4)
    };
    let row = |name: &str, colour: [u8; 3], (low, high): (f32, f32), shown: bool| {
        let adjusted = adjusted(name, (low, high));
        json!({
            "name": name,
            "colour": markers::hex(colour),
            "visible": st.prefs.is_visible(name),
            "window": if !shown {
                String::new()
            } else if adjusted {
                // Translators: {} is a marker's display window ("arcsinh 4.50–6.50")
                // that the user set.
                tr!("{}, yours", markers::window_text(low, high))
            } else {
                markers::window_text(low, high)
            },
            "adjusted": adjusted,
        })
    };
    let rows: Vec<Value> = match &st.markers {
        Ok(markers) => markers
            .iter()
            .map(|m| row(&m.name, m.colour, (m.low, m.high), true))
            .collect(),
        Err(_) => st
            .prefs
            .markers
            .iter()
            .enumerate()
            .map(|(k, name)| {
                let window = st
                    .prefs
                    .windows
                    .get(name)
                    .map(|&[low, high]| (low, high))
                    .or_else(|| markers::default_window(name))
                    .unwrap_or((0.0, 1.0));
                let colour = markers::PALETTE[k.min(markers::PALETTE.len() - 1)];
                row(name, colour, window, false)
            })
            .collect(),
    };
    // The window editor: one marker at a time, below the list, so that the
    // list can change while a slider is dragged.
    let edited = st.window_editor.and_then(|index| {
        let marker = st.markers.as_ref().ok()?.get(index)?;
        Some((index, marker))
    });
    out.insert("markers".into(), json!(rows));
    let (index, low, high, (min, max), is_adjusted) = match edited {
        Some((index, m)) => (
            index as i64,
            m.low,
            m.high,
            markers::window_range(&m.name),
            m.is_adjusted(),
        ),
        None => (-1, 0.0, 1.0, (0.0, 1.0), false),
    };
    out.insert("window-editor".into(), json!(index));
    out.insert("window-low".into(), json!(low));
    out.insert("window-high".into(), json!(high));
    out.insert("window-min".into(), json!(min));
    out.insert("window-max".into(), json!(max));
    out.insert("window-adjusted".into(), json!(is_adjusted));
    out.insert("model-card".into(), json!(model_card(st)));
    download_properties(st, &mut out);
    readout_properties(st, &mut out);
    out
}

/// The *Download model…* form.
fn download_properties(st: &State, out: &mut Map<String, Value>) {
    let (note, error) = match (&st.model_fetch, &st.download_note) {
        (Some(_), _) => (
            // Translators: eov first asks the user whether to download the model
            // (nothing is downloaded without that answer), then downloads it.
            tr!(
                "eov is asking you whether to download the model, or downloading it (see the status bar)\u{2026}"
            ),
            false,
        ),
        (None, Some(note)) => (note.to_string(), true),
        (None, None) => (
            tr!(
                // Translators: under the model's address and SHA-256 in the
                // Cellaris panel; the model store is where eov keeps models.
                "eov checks the download against this SHA-256 and keeps it in its model store; it is then Cellaris's model. Without an address, the copy your administrator gave eov is used."
            ),
            false,
        ),
    };
    out.insert(
        "can-download".into(),
        json!(st.model_store != crate::engine::Support::No),
    );
    out.insert("download-open".into(), json!(st.download_open));
    out.insert("download-busy".into(), json!(st.model_fetch.is_some()));
    out.insert(
        "download-url".into(),
        json!(st.prefs.model_url.clone().unwrap_or_default()),
    );
    out.insert(
        "download-sha256".into(),
        json!(st.prefs.model_sha256.clone().unwrap_or_default()),
    );
    out.insert("download-note".into(), json!(note));
    out.insert("download-note-error".into(), json!(error));
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
    // In English, as files are: the readout is data.
    for note in scale_notes(st, file)
        .into_iter()
        .chain(display_note(st, id))
    {
        lines.insert(lines.len() - 1, format!("# note\t{}", note.english()));
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
    "set-window",
    "reset-window",
    "edit-window",
    "toggle-download",
    "download-model",
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
    // Pan-CK's display window is the user's, and open in the panel.
    st.prefs.windows.insert("Pan-CK".into(), [4.5, 6.5]);
    if let Ok(markers) = &mut st.markers {
        markers::apply_windows(markers, &st.prefs.windows);
    }
    st.window_editor = Some(1);
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
                    .or_else(|| line.strip_prefix("in-out property <"))
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
        // Pan-CK's window is the user's: the legend says so.
        assert_eq!(
            shown[MODEL_BADGE],
            (
                None,
                "Hoechst: yellow \u{b7} Pan-CK: magenta (arcsinh 4.50\u{2013}6.50) \u{b7} SMA: \
                 bluish green"
                    .to_string()
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
        assert!(tooltip.contains("1 near-black block left out"), "{tooltip}");
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
        // A host before plugin API 1.1: the badge only, in English.
        st.displays.insert(
            1,
            DisplayAdjustments {
                modified: true,
                badge: "Colour-normalised (Macenko)".into(),
                ..Default::default()
            },
        );
        let note = display_note(&st, 1).unwrap().to_string();
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
    fn the_panel_names_the_adjustments_as_eov_shows_them_and_the_readout_in_english() {
        let mut st = example_state();
        let english = "Colour-normalised (Macenko) \u{b7} Sharpened";
        let shown = "Farbnormalisiert (Macenko) \u{b7} Gesch\u{e4}rft";
        st.displays.insert(
            1,
            DisplayAdjustments {
                modified: true,
                sharpness: 0.5,
                stain_normalization: Some("macenko".into()),
                badge: english.into(),
                shown_badge: Some(shown.into()),
                stain_colors: Some(true),
                ..Default::default()
            },
        );
        let notes = sidebar_properties(&st)["slide-notes"]
            .as_str()
            .unwrap()
            .to_string();
        assert!(
            notes.contains(&format!("the display is adjusted ({shown});")),
            "{notes}"
        );
        // The copied readout is data, in English.
        let text = copy_text(&st, 1).unwrap();
        assert!(
            text.contains(&format!(
                "# note\tthe display is adjusted ({english}); predictions use the slide's \
                 unadjusted colours\n"
            )),
            "{text}"
        );
        assert!(!text.contains("Farbnormalisiert"), "{text}");
        // Each says how only when its own badge does.
        st.displays.get_mut(&1).unwrap().badge = String::new();
        let note = display_note(&st, 1).unwrap();
        assert_eq!(
            note.english(),
            "the display is adjusted; predictions use the slide's unadjusted colours"
        );
        assert!(note.to_string().contains(shown), "{note}");
    }

    #[test]
    fn windows_unlike_the_training_data_are_counted_in_their_own_number() {
        let st = example_state();
        let model = st.model.info().unwrap();
        let file = &st.files[&1];
        let note = |unusual, tissue| {
            let windows = WindowCounts {
                dark: 0,
                tissue,
                unusual,
            };
            slide_notes(model, file.grid().map(|grid| &**grid), &file.info, windows)
                .iter()
                .map(Text::english)
                .find(|note| note.contains("training data"))
                .unwrap()
        };
        assert_eq!(note(1, 40), "1 window of 40 looks unlike the training data");
        assert_eq!(note(3, 40), "3 windows of 40 look unlike the training data");
        assert_eq!(note(1, 1), "1 window of 1 looks unlike the training data");
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
