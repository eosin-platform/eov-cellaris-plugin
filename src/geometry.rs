//! The model grid: where windows and cells sit on a slide, and how each
//! window is read from the pyramid (CEL-3).
//!
//! The model was trained at one pixel size (`model_mpp`, 0.325 µm/px for the
//! Orion cohort). Every slide is mapped onto a virtual grid of pixels that
//! size, whatever its scanner: a window is `window` *model* pixels across
//! (≈ 83 µm), not 256 slide pixels. Windows sit at a stride of half their
//! size and every `stride`-sized cell is the feathered blend of the windows
//! over it (see [`crate::blend`]).
//!
//! Each window is read from the coarsest pyramid level that is still at
//! least as fine as the model's pixels, so it is only ever downsampled, and
//! resampled to the model grid with an area filter ([`crate::resample`]).
//! Slides coarser than the model are upsampled (and flagged) up to
//! [`MAX_UPSAMPLE`]; beyond that they are refused.

use eov_plugin_api::abi::{MAX_REGION_PIXELS, RegionRequest};
use eov_plugin_api::protocol::{FileId, FileInfo, Rect};

/// Side of a readout block in model pixels: 5.2 µm at 0.325 µm/px, about a
/// cell.
pub const BLOCK: usize = 16;
/// Slides whose pixels are at most this much coarser than the model's are
/// treated as matching.
pub const SCALE_TOLERANCE: f64 = 1.10;
/// Slides coarser than the model by more than this factor are refused.
pub const MAX_UPSAMPLE: f64 = 2.0;
/// Source pixels read around each window for the resampling filter.
const PAD: f64 = 2.0;

/// A cell or window position on the model grid.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct Idx {
    pub i: i64,
    pub j: i64,
}

impl Idx {
    pub const fn new(i: i64, j: i64) -> Self {
        Self { i, j }
    }
}

/// Where a slide's pixel size came from.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ScaleSource {
    /// Both axes reported by the slide.
    Reported,
    /// Only one axis was reported; it is used for both.
    OneAxis,
    /// Nothing was reported and the user chose to assume the model's.
    Assumed,
}

/// The pixel size of a slide's level 0, in µm per pixel.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Scale {
    pub mpp_x: f64,
    pub mpp_y: f64,
    pub source: ScaleSource,
}

/// Why a slide cannot be mapped onto the model grid.
#[derive(Clone, Debug, PartialEq)]
pub enum ScaleProblem {
    /// The slide reports no pixel size.
    Unknown,
    /// The slide is too coarse to resample to the model's pixel size.
    TooCoarse { slide_mpp: f64, model_mpp: f64 },
    /// The pyramid cannot serve windows (bad levels, or none close enough).
    BadSlide(String),
}

impl ScaleProblem {
    /// One line for the user.
    pub fn message(&self) -> String {
        match self {
            ScaleProblem::Unknown => "This slide does not report its pixel size, so it cannot be \
                matched to the model's. Turn on \u{201c}Assume the model's pixel size\u{201d} in \
                the Cellaris panel to run anyway."
                .to_string(),
            ScaleProblem::TooCoarse {
                slide_mpp,
                model_mpp,
            } => format!(
                "This slide is scanned at {slide_mpp:.3} \u{b5}m/px, more than {MAX_UPSAMPLE:.0}\u{d7} \
                 coarser than the {model_mpp:.3} \u{b5}m/px the model was trained at; predictions \
                 would be meaningless."
            ),
            ScaleProblem::BadSlide(reason) => reason.clone(),
        }
    }
}

fn plausible_mpp(value: Option<f64>) -> Option<f64> {
    value.filter(|v| v.is_finite() && *v > 0.0 && *v < 1000.0)
}

impl Scale {
    /// The slide's pixel size. With `assume_when_unknown`, a slide without
    /// one is taken to be at `model_mpp`.
    pub fn of(
        info: &FileInfo,
        model_mpp: f64,
        assume_when_unknown: bool,
    ) -> Result<Scale, ScaleProblem> {
        match (plausible_mpp(info.mpp_x), plausible_mpp(info.mpp_y)) {
            (Some(mpp_x), Some(mpp_y)) => Ok(Scale {
                mpp_x,
                mpp_y,
                source: ScaleSource::Reported,
            }),
            (Some(mpp), None) | (None, Some(mpp)) => Ok(Scale {
                mpp_x: mpp,
                mpp_y: mpp,
                source: ScaleSource::OneAxis,
            }),
            (None, None) if assume_when_unknown => Ok(Scale {
                mpp_x: model_mpp,
                mpp_y: model_mpp,
                source: ScaleSource::Assumed,
            }),
            (None, None) => Err(ScaleProblem::Unknown),
        }
    }

    /// How many times coarser than the model's the slide's pixels are
    /// (below 1: finer).
    pub fn upsampling(&self, model_mpp: f64) -> f64 {
        self.mpp_x.max(self.mpp_y) / model_mpp
    }

    /// Whether x and y differ by more than 2 %.
    pub fn is_anisotropic(&self) -> bool {
        let mean = (self.mpp_x + self.mpp_y) / 2.0;
        (self.mpp_x - self.mpp_y).abs() > 0.02 * mean
    }
}

/// How one window is read and where it sits in what was read.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ReadPlan {
    pub request: RegionRequest,
    /// The window's top-left corner in the read region's pixel coordinates
    /// (pixel edges, so pixel `k` spans `[k, k + 1)`).
    pub origin_x: f64,
    pub origin_y: f64,
    /// Region pixels per model pixel.
    pub step_x: f64,
    pub step_y: f64,
}

/// One slide mapped onto the model grid.
#[derive(Clone, Debug, PartialEq)]
pub struct Grid {
    pub file_id: FileId,
    /// Level-0 size of the slide.
    pub width: f64,
    pub height: f64,
    /// The level-0 area holding image data: the slide, cut to the scanned
    /// bounds when the slide reports them.
    pub area: Rect,
    /// Level-0 pixels per model pixel.
    pub sx: f64,
    pub sy: f64,
    /// Window side and stride, in model pixels.
    pub window: usize,
    pub stride: usize,
    /// Cells across and down.
    pub cols: i64,
    pub rows: i64,
    /// The pyramid level windows are read from.
    pub level: u32,
    pub downsample: f64,
    pub scale: Scale,
    pub model_mpp: f64,
}

impl Grid {
    pub fn new(
        info: &FileInfo,
        scale: Scale,
        model_mpp: f64,
        window: usize,
    ) -> Result<Grid, ScaleProblem> {
        if !(model_mpp.is_finite() && model_mpp > 0.0) {
            return Err(ScaleProblem::BadSlide(format!(
                "the model's pixel size {model_mpp} is not usable"
            )));
        }
        if window < 2 * BLOCK || !window.is_multiple_of(2 * BLOCK) {
            return Err(ScaleProblem::BadSlide(format!(
                "the model's window of {window} px is not a multiple of {}",
                2 * BLOCK
            )));
        }
        let upsampling = scale.upsampling(model_mpp);
        if upsampling > MAX_UPSAMPLE {
            return Err(ScaleProblem::TooCoarse {
                slide_mpp: scale.mpp_x.max(scale.mpp_y),
                model_mpp,
            });
        }
        if info.width == 0 || info.height == 0 {
            return Err(ScaleProblem::BadSlide("the slide is empty".to_string()));
        }
        let sx = model_mpp / scale.mpp_x;
        let sy = model_mpp / scale.mpp_y;

        // The coarsest level whose pixels are still no larger than the
        // model's along either axis; level 0 when even that is coarser.
        let finest = sx.min(sy);
        let mut levels: Vec<(u32, f64)> = info
            .levels
            .iter()
            .filter(|level| level.downsample.is_finite() && level.downsample >= 0.999)
            .map(|level| (level.level, level.downsample))
            .collect();
        if levels.is_empty() {
            levels.push((0, 1.0));
        }
        levels.sort_by(|a, b| a.1.total_cmp(&b.1));
        let (level, downsample) = levels
            .iter()
            .rev()
            .find(|(_, downsample)| *downsample <= finest * 1.001)
            .copied()
            .unwrap_or(levels[0]);

        let side = |step: f64| (window as f64 * step + 2.0 * PAD + 2.0).ceil() as u64;
        let (read_w, read_h) = (side(sx / downsample), side(sy / downsample));
        if read_w * read_h > MAX_REGION_PIXELS || read_w > u64::from(u32::MAX) {
            return Err(ScaleProblem::BadSlide(format!(
                "no pyramid level is close to the model's {model_mpp:.3} \u{b5}m/px; a window \
                 would need {read_w}\u{d7}{read_h} pixels"
            )));
        }

        let width = info.width as f64;
        let height = info.height as f64;
        let full = Rect {
            x: 0.0,
            y: 0.0,
            width,
            height,
        };
        let area = match info.bounds {
            Some(bounds) if bounds.width > 0.0 && bounds.height > 0.0 => {
                intersect(&full, &bounds).unwrap_or(full)
            }
            _ => full,
        };
        let stride = window / 2;
        let cols = (width / (stride as f64 * sx)).ceil() as i64;
        let rows = (height / (stride as f64 * sy)).ceil() as i64;
        Ok(Grid {
            file_id: info.file_id,
            width,
            height,
            area,
            sx,
            sy,
            window,
            stride,
            cols,
            rows,
            level,
            downsample,
            scale,
            model_mpp,
        })
    }

    /// Level-0 size of a cell.
    pub fn cell_size(&self) -> (f64, f64) {
        (self.stride as f64 * self.sx, self.stride as f64 * self.sy)
    }

    /// The level-0 rectangle of a cell.
    pub fn cell_rect(&self, cell: Idx) -> Rect {
        let (w, h) = self.cell_size();
        Rect {
            x: cell.i as f64 * w,
            y: cell.j as f64 * h,
            width: w,
            height: h,
        }
    }

    /// The level-0 rectangle a window covers.
    pub fn window_rect(&self, window: Idx) -> Rect {
        let (w, h) = self.cell_size();
        Rect {
            x: window.i as f64 * w,
            y: window.j as f64 * h,
            width: 2.0 * w,
            height: 2.0 * h,
        }
    }

    /// The range of cells (`i0..i1`, `j0..j1`) overlapping a level-0
    /// rectangle and the slide's image area.
    pub fn cell_range(&self, rect: &Rect) -> (i64, i64, i64, i64) {
        let Some(clipped) = intersect(rect, &self.area) else {
            return (0, 0, 0, 0);
        };
        let (w, h) = self.cell_size();
        let i0 = ((clipped.x / w).floor() as i64).clamp(0, self.cols);
        let j0 = ((clipped.y / h).floor() as i64).clamp(0, self.rows);
        let i1 = (((clipped.x + clipped.width) / w).ceil() as i64).clamp(i0, self.cols);
        let j1 = (((clipped.y + clipped.height) / h).ceil() as i64).clamp(j0, self.rows);
        (i0, i1, j0, j1)
    }

    /// How many cells a level-0 rectangle needs.
    pub fn cell_count(&self, rect: &Rect) -> u64 {
        let (i0, i1, j0, j1) = self.cell_range(rect);
        (i1 - i0) as u64 * (j1 - j0) as u64
    }

    /// The cells overlapping a level-0 rectangle, nearest to `(cx, cy)`
    /// first.
    pub fn cells_by_distance(&self, rect: &Rect, cx: f64, cy: f64) -> Vec<Idx> {
        let (i0, i1, j0, j1) = self.cell_range(rect);
        let (w, h) = self.cell_size();
        let mut cells: Vec<(f64, Idx)> = (j0..j1)
            .flat_map(|j| (i0..i1).map(move |i| Idx::new(i, j)))
            .map(|cell| {
                let dx = (cell.i as f64 + 0.5) * w - cx;
                let dy = (cell.j as f64 + 0.5) * h - cy;
                (dx * dx + dy * dy, cell)
            })
            .collect();
        cells.sort_by(|a, b| a.0.total_cmp(&b.0).then(a.1.cmp(&b.1)));
        cells.into_iter().map(|(_, cell)| cell).collect()
    }

    pub fn contains_cell(&self, cell: Idx) -> bool {
        (0..self.cols).contains(&cell.i) && (0..self.rows).contains(&cell.j)
    }

    /// The windows over a cell, each with the cell's quadrant in it
    /// (`dx`, `dy`: 0 = the window's left/top half).
    pub fn covering(&self, cell: Idx) -> impl Iterator<Item = (Idx, usize, usize)> + '_ {
        [(0usize, 0usize), (1, 0), (0, 1), (1, 1)]
            .into_iter()
            .map(move |(dx, dy)| (Idx::new(cell.i - dx as i64, cell.j - dy as i64), dx, dy))
            .filter(|(window, _, _)| window.i >= 0 && window.j >= 0)
    }

    /// The cells of a window that are on the grid, with their quadrants.
    pub fn window_cells(&self, window: Idx) -> impl Iterator<Item = (Idx, usize, usize)> + '_ {
        [(0usize, 0usize), (1, 0), (0, 1), (1, 1)]
            .into_iter()
            .map(move |(dx, dy)| (Idx::new(window.i + dx as i64, window.j + dy as i64), dx, dy))
            .filter(|(cell, _, _)| self.contains_cell(*cell))
    }

    /// How a window is read: the region request, and where the window lies
    /// in it.
    pub fn read_plan(&self, window: Idx) -> ReadPlan {
        let d = self.downsample;
        let step_x = self.sx / d;
        let step_y = self.sy / d;
        let lx = window.i as f64 * self.stride as f64 * step_x;
        let ly = window.j as f64 * self.stride as f64 * step_y;
        let side = self.window as f64;
        let px0 = (lx - PAD).floor();
        let py0 = (ly - PAD).floor();
        let px1 = (lx + side * step_x + PAD).ceil();
        let py1 = (ly + side * step_y + PAD).ceil();
        let x = (px0 * d).round() as i64;
        let y = (py0 * d).round() as i64;
        ReadPlan {
            request: RegionRequest {
                file_id: self.file_id,
                level: self.level,
                x,
                y,
                width: (px1 - px0) as u32,
                height: (py1 - py0) as u32,
                // Straight alpha: transparency is composited onto white here,
                // and kept as coverage for the tissue mask.
                background: 0,
            },
            origin_x: lx - x as f64 / d,
            origin_y: ly - y as f64 / d,
            step_x,
            step_y,
        }
    }

    /// Blocks per cell side.
    pub fn blocks_per_side(&self) -> usize {
        self.stride / BLOCK
    }

    /// The level-0 centre of block `(bx, by)` of a cell.
    pub fn block_center(&self, cell: Idx, bx: usize, by: usize) -> (f64, f64) {
        let u = (cell.i as f64 * self.stride as f64) + (bx * BLOCK) as f64 + BLOCK as f64 / 2.0;
        let v = (cell.j as f64 * self.stride as f64) + (by * BLOCK) as f64 + BLOCK as f64 / 2.0;
        (u * self.sx, v * self.sy)
    }

    /// Area of one block in µm² (in the assumed scale when it is assumed).
    pub fn block_area_um2(&self) -> f64 {
        let side = BLOCK as f64 * self.model_mpp;
        side * side
    }

    /// Area of a level-0 pixel in µm².
    pub fn pixel_area_um2(&self) -> f64 {
        self.scale.mpp_x * self.scale.mpp_y
    }
}

/// The overlap of two rectangles, if they overlap.
pub fn intersect(a: &Rect, b: &Rect) -> Option<Rect> {
    let x0 = a.x.max(b.x);
    let y0 = a.y.max(b.y);
    let x1 = (a.x + a.width).min(b.x + b.width);
    let y1 = (a.y + a.height).min(b.y + b.height);
    (x1 > x0 && y1 > y0).then_some(Rect {
        x: x0,
        y: y0,
        width: x1 - x0,
        height: y1 - y0,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use eov_plugin_api::protocol::LevelInfo;

    const ORION: f64 = 0.325;

    fn slide(mpp: Option<f64>, downsamples: &[f64]) -> FileInfo {
        FileInfo {
            file_id: 7,
            width: 100_000,
            height: 80_000,
            mpp_x: mpp,
            mpp_y: mpp,
            levels: downsamples
                .iter()
                .enumerate()
                .map(|(level, d)| LevelInfo {
                    level: level as u32,
                    width: (100_000.0 / d) as u64,
                    height: (80_000.0 / d) as u64,
                    downsample: *d,
                })
                .collect(),
            ..Default::default()
        }
    }

    fn grid(info: &FileInfo) -> Result<Grid, ScaleProblem> {
        let scale = Scale::of(info, ORION, false)?;
        Grid::new(info, scale, ORION, 256)
    }

    #[test]
    fn a_window_is_the_same_tissue_on_every_scanner() {
        // 83.2 µm across whatever the slide's pixel size.
        for mpp in [0.25, ORION, 0.5] {
            let g = grid(&slide(Some(mpp), &[1.0, 4.0, 16.0])).unwrap();
            let rect = g.window_rect(Idx::new(3, 2));
            let microns = rect.width * mpp;
            assert!(
                (microns - 256.0 * ORION).abs() < 1e-9,
                "{mpp}: {microns} µm"
            );
        }
    }

    #[test]
    fn the_coarsest_level_still_finer_than_the_model_is_read() {
        // 0.1625 µm/px: two slide pixels per model pixel, so level 1 (×2).
        let g = grid(&slide(Some(0.1625), &[1.0, 2.0, 8.0])).unwrap();
        assert_eq!((g.level, g.downsample), (1, 2.0));
        let plan = g.read_plan(Idx::new(0, 0));
        assert!((plan.step_x - 1.0).abs() < 1e-12);

        // Aperio downsamples are not exact: 4.0003 still serves 4×.
        let g = grid(&slide(Some(0.08125), &[1.0, 4.0003, 16.001])).unwrap();
        assert_eq!(g.level, 1);

        // 40× (0.25 µm/px) with 1, 4, 16: level 0, downsampled 1.3×.
        let g = grid(&slide(Some(0.25), &[1.0, 4.0, 16.0])).unwrap();
        assert_eq!(g.level, 0);
        assert!((g.sx - 1.3).abs() < 1e-12);

        // 20× (0.5 µm/px) is coarser than the model: level 0, upsampled.
        let g = grid(&slide(Some(0.5), &[1.0, 4.0])).unwrap();
        assert_eq!(g.level, 0);
        assert!(g.sx < 1.0);
        assert!(g.scale.upsampling(ORION) > SCALE_TOLERANCE);
    }

    #[test]
    fn unknown_or_far_too_coarse_pixel_sizes_are_refused() {
        assert_eq!(grid(&slide(None, &[1.0])), Err(ScaleProblem::Unknown));
        let assumed = Scale::of(&slide(None, &[1.0]), ORION, true).unwrap();
        assert_eq!(assumed.source, ScaleSource::Assumed);
        assert!(matches!(
            grid(&slide(Some(1.0), &[1.0])),
            Err(ScaleProblem::TooCoarse { .. })
        ));
        // Nonsense values count as unknown.
        assert_eq!(grid(&slide(Some(0.0), &[1.0])), Err(ScaleProblem::Unknown));
        assert_eq!(
            grid(&slide(Some(f64::NAN), &[1.0])),
            Err(ScaleProblem::Unknown)
        );
    }

    #[test]
    fn one_reported_axis_is_used_for_both_and_anisotropy_is_kept() {
        let mut info = slide(Some(0.25), &[1.0]);
        info.mpp_y = None;
        let scale = Scale::of(&info, ORION, false).unwrap();
        assert_eq!(scale.source, ScaleSource::OneAxis);
        assert_eq!(scale.mpp_y, 0.25);

        let mut info = slide(Some(0.25), &[1.0]);
        info.mpp_y = Some(0.26);
        let g = grid(&info).unwrap();
        assert!(g.scale.is_anisotropic());
        let plan = g.read_plan(Idx::new(1, 1));
        assert!((plan.step_x - 1.3).abs() < 1e-12);
        assert!((plan.step_y - 1.25).abs() < 1e-12);
    }

    #[test]
    fn the_read_region_holds_the_window_with_a_margin() {
        for mpp in [0.1625, 0.25, ORION, 0.5] {
            let g = grid(&slide(Some(mpp), &[1.0, 2.0, 4.0])).unwrap();
            for window in [Idx::new(0, 0), Idx::new(17, 5), Idx::new(101, 77)] {
                let plan = g.read_plan(window);
                let d = g.downsample;
                // The region's first pixel is at level-0 x, so the window's
                // level-0 origin maps back exactly.
                let level0 = plan.request.x as f64 + plan.origin_x * d;
                let expected = window.i as f64 * g.stride as f64 * g.sx;
                assert!((level0 - expected).abs() < 1e-6, "{mpp} {window:?}");
                assert!(plan.origin_x >= 1.0 && plan.origin_y >= 1.0);
                let far_x = plan.origin_x + g.window as f64 * plan.step_x;
                let far_y = plan.origin_y + g.window as f64 * plan.step_y;
                assert!(far_x + 1.0 <= plan.request.width as f64, "{mpp} {window:?}");
                assert!(
                    far_y + 1.0 <= plan.request.height as f64,
                    "{mpp} {window:?}"
                );
                assert_eq!(plan.request.level, g.level);
                assert_eq!(plan.request.background, 0);
            }
        }
    }

    #[test]
    fn cells_stay_on_the_slide_and_its_scanned_area() {
        let mut info = slide(Some(ORION), &[1.0]);
        info.width = 1000;
        info.height = 300;
        let g = grid(&info).unwrap();
        assert_eq!((g.cols, g.rows), (8, 3));
        // Past the right and bottom edges, and before the origin.
        let everything = Rect {
            x: -5000.0,
            y: -5000.0,
            width: 20_000.0,
            height: 20_000.0,
        };
        assert_eq!(g.cell_count(&everything), 24);
        let beyond = Rect {
            x: 2000.0,
            y: 0.0,
            width: 500.0,
            height: 500.0,
        };
        assert_eq!(g.cell_count(&beyond), 0);

        // Scanned bounds cut the area further.
        info.bounds = Some(Rect {
            x: 300.0,
            y: 0.0,
            width: 200.0,
            height: 100.0,
        });
        let g = grid(&info).unwrap();
        assert_eq!(g.cell_range(&everything), (2, 4, 0, 1));
    }

    #[test]
    fn windows_and_cells_cover_each_other() {
        let g = grid(&slide(Some(ORION), &[1.0])).unwrap();
        let cell = Idx::new(4, 3);
        let windows: Vec<_> = g.covering(cell).collect();
        assert_eq!(windows.len(), 4);
        for (window, dx, dy) in &windows {
            assert!(
                g.window_cells(*window)
                    .any(|(c, cdx, cdy)| c == cell && cdx == *dx && cdy == *dy)
            );
            let w = g.window_rect(*window);
            let c = g.cell_rect(cell);
            assert!(w.x <= c.x && c.x + c.width <= w.x + w.width + 1e-9);
            assert!(w.y <= c.y && c.y + c.height <= w.y + w.height + 1e-9);
        }
        // At the slide's corner only one window exists.
        assert_eq!(g.covering(Idx::new(0, 0)).count(), 1);
        // The last window's right cells are off the grid.
        let last = Idx::new(g.cols - 1, 0);
        assert_eq!(g.window_cells(last).count(), 2);
    }

    #[test]
    fn cells_are_ordered_outward_from_the_centre() {
        let g = grid(&slide(Some(ORION), &[1.0])).unwrap();
        let (w, h) = g.cell_size();
        let rect = Rect {
            x: 0.0,
            y: 0.0,
            width: 5.0 * w,
            height: 5.0 * h,
        };
        let cells = g.cells_by_distance(&rect, 2.5 * w, 2.5 * h);
        assert_eq!(cells.len(), 25);
        assert_eq!(cells[0], Idx::new(2, 2));
        let corner = cells.iter().position(|c| *c == Idx::new(0, 0)).unwrap();
        assert!(corner >= 20, "a corner came {corner}th");
    }

    #[test]
    fn blocks_are_defined_in_microns() {
        for mpp in [0.25, 0.5] {
            let g = grid(&slide(Some(mpp), &[1.0])).unwrap();
            assert!((g.block_area_um2() - (16.0 * ORION).powi(2)).abs() < 1e-9);
            let (x0, _) = g.block_center(Idx::new(0, 0), 0, 0);
            let (x1, _) = g.block_center(Idx::new(0, 0), 1, 0);
            assert!(((x1 - x0) * mpp - 16.0 * ORION).abs() < 1e-9);
        }
    }
}
