//! Region readout: the prediction inside a drawn region, beside how much of
//! it optical density alone reproduces.
//!
//! The control is cellaris's tissue-density control, measured over blocks
//! of [`BLOCK`] model pixels (5.2 µm, about a cell) because the viewer has
//! no segmentation. Blocks come from the same feathered cells the overlay
//! paints (CEL-5), and are defined in micrometres (CEL-3).

use crate::blend::CellData;
use crate::geometry::{BLOCK, Grid, Idx};
use crate::resample::TISSUE_OD;
use eov_plugin_api::protocol::{Rect, Shape};
use std::collections::BTreeSet;

/// Fewer tissue blocks than this give no r².
pub const MIN_BLOCKS: usize = 30;
/// Blocks must lie this fully on the slide to be measured.
const MIN_COVERAGE: f32 = 0.99;
/// Blocks with more dark (not bright-field) pixels than this are skipped.
const MAX_DARK: f32 = 0.25;

/// A region drawn or chosen on a slide, in level-0 pixels.
#[derive(Clone, Debug, PartialEq)]
pub enum Region {
    /// Rings of a polygon; the first is the outline, the rest are holes.
    Polygon(Vec<Vec<[f64; 2]>>),
    Rectangle(Rect),
    Ellipse {
        cx: f64,
        cy: f64,
        rx: f64,
        ry: f64,
    },
}

fn finite(points: &[[f64; 2]]) -> bool {
    points.iter().all(|p| p[0].is_finite() && p[1].is_finite())
}

impl Region {
    /// A region from a host shape; `None` for shapes without an inside.
    pub fn from_shape(shape: &Shape) -> Option<Region> {
        if !shape.is_valid() {
            return None;
        }
        match shape {
            Shape::Polygon { exterior, holes } if exterior.len() >= 3 && finite(exterior) => {
                let mut rings = vec![exterior.clone()];
                rings.extend(holes.iter().filter(|h| h.len() >= 3).cloned());
                Some(Region::Polygon(rings))
            }
            Shape::Rectangle {
                x,
                y,
                width,
                height,
            } if *width > 0.0 && *height > 0.0 => Some(Region::Rectangle(Rect {
                x: *x,
                y: *y,
                width: *width,
                height: *height,
            })),
            Shape::Ellipse { cx, cy, rx, ry } if *rx > 0.0 && *ry > 0.0 => Some(Region::Ellipse {
                cx: *cx,
                cy: *cy,
                rx: *rx,
                ry: *ry,
            }),
            _ => None,
        }
    }

    /// The shape drawn for the region.
    pub fn to_shape(&self) -> Shape {
        match self {
            Region::Polygon(rings) => Shape::Polygon {
                exterior: rings[0].clone(),
                holes: rings[1..].to_vec(),
            },
            Region::Rectangle(r) => Shape::Rectangle {
                x: r.x,
                y: r.y,
                width: r.width,
                height: r.height,
            },
            Region::Ellipse { cx, cy, rx, ry } => Shape::Ellipse {
                cx: *cx,
                cy: *cy,
                rx: *rx,
                ry: *ry,
            },
        }
    }

    /// Even-odd containment, so holes and self-crossing outlines work.
    pub fn contains(&self, x: f64, y: f64) -> bool {
        match self {
            Region::Polygon(rings) => {
                let mut inside = false;
                for ring in rings {
                    let mut previous = ring[ring.len() - 1];
                    for &point in ring {
                        let (xi, yi) = (point[0], point[1]);
                        let (xj, yj) = (previous[0], previous[1]);
                        if (yi > y) != (yj > y) && x < (xj - xi) * (y - yi) / (yj - yi) + xi {
                            inside = !inside;
                        }
                        previous = point;
                    }
                }
                inside
            }
            Region::Rectangle(r) => x >= r.x && x < r.x + r.width && y >= r.y && y < r.y + r.height,
            Region::Ellipse { cx, cy, rx, ry } => {
                let (u, v) = ((x - cx) / rx, (y - cy) / ry);
                u * u + v * v <= 1.0
            }
        }
    }

    pub fn bounds(&self) -> Rect {
        self.to_shape().bounds()
    }

    /// Area in level-0 pixels² (holes subtracted).
    pub fn area(&self) -> f64 {
        fn ring_area(ring: &[[f64; 2]]) -> f64 {
            let mut twice = 0.0;
            let mut previous = ring[ring.len() - 1];
            for &point in ring {
                twice += previous[0] * point[1] - point[0] * previous[1];
                previous = point;
            }
            (twice / 2.0).abs()
        }
        match self {
            Region::Polygon(rings) => {
                let holes: f64 = rings[1..].iter().map(|ring| ring_area(ring)).sum();
                (ring_area(&rings[0]) - holes).max(0.0)
            }
            Region::Rectangle(r) => r.width * r.height,
            Region::Ellipse { rx, ry, .. } => std::f64::consts::PI * rx * ry,
        }
    }

    /// The cells with at least one block centre inside the region, nearest
    /// the region's centre first. Stops past `limit` cells.
    pub fn cells(&self, grid: &Grid, limit: usize) -> Vec<Idx> {
        let bounds = self.bounds();
        let (cx, cy) = (
            bounds.x + bounds.width / 2.0,
            bounds.y + bounds.height / 2.0,
        );
        let per = grid.blocks_per_side();
        let mut found = Vec::new();
        for cell in grid.cells_by_distance(&bounds, cx, cy) {
            let inside = (0..per * per).any(|index| {
                let (x, y) = grid.block_center(cell, index % per, index / per);
                self.contains(x, y)
            });
            if inside {
                found.push(cell);
                if found.len() > limit {
                    break;
                }
            }
        }
        found
    }
}

/// Running least-squares of a marker on density.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Fit {
    n: f64,
    sx: f64,
    sy: f64,
    sxx: f64,
    syy: f64,
    sxy: f64,
}

impl Fit {
    pub fn add(&mut self, density: f64, value: f64) {
        self.n += 1.0;
        self.sx += density;
        self.sy += value;
        self.sxx += density * density;
        self.syy += value * value;
        self.sxy += density * value;
    }

    pub fn blocks(&self) -> usize {
        self.n as usize
    }

    pub fn mean(&self) -> f64 {
        if self.n == 0.0 { 0.0 } else { self.sy / self.n }
    }

    /// Standard deviation of the prediction across blocks. Near zero, a low
    /// r² only means there was nothing to explain.
    pub fn spread(&self) -> f64 {
        if self.n == 0.0 {
            0.0
        } else {
            (self.syy / self.n - self.mean() * self.mean())
                .max(0.0)
                .sqrt()
        }
    }

    /// Share of the prediction's variance a straight line in density
    /// reproduces. Squared, so a control pointing either way counts.
    pub fn r2(&self) -> Option<f64> {
        if self.blocks() < MIN_BLOCKS {
            return None;
        }
        let vx = self.sxx - self.sx * self.sx / self.n;
        let vy = self.syy - self.sy * self.sy / self.n;
        let cov = self.sxy - self.sx * self.sy / self.n;
        (vx > 1e-12 && vy > 1e-12).then(|| (cov * cov / (vx * vy)).clamp(0.0, 1.0))
    }
}

/// Where a readout stands.
#[derive(Clone, Debug, PartialEq)]
pub enum ReadoutState {
    /// Waiting for the user to accept a long run.
    Confirming,
    Running,
    Done,
    Cancelled,
    Failed(String),
}

/// The measurement of one region.
#[derive(Clone, Debug)]
pub struct Readout {
    pub id: u64,
    pub region: Region,
    /// Cells the region needs, nearest its centre first.
    pub cells: Vec<Idx>,
    pub remaining: BTreeSet<Idx>,
    pub fits: Vec<Fit>,
    pub recon_sum: f64,
    pub recon_blocks: usize,
    /// Blocks above the model's reconstruction-error threshold.
    pub unusual_blocks: usize,
    /// Cells skipped as not bright-field.
    pub dark_cells: usize,
    /// Tissue blocks skipped for dark (not bright-field) pixels.
    pub dark_blocks: usize,
    pub state: ReadoutState,
    /// Windows the region needed when it was drawn.
    pub windows: usize,
}

impl Readout {
    pub fn new(id: u64, region: Region, cells: Vec<Idx>, markers: usize, windows: usize) -> Self {
        Self {
            id,
            remaining: cells.iter().copied().collect(),
            cells,
            region,
            fits: vec![Fit::default(); markers],
            recon_sum: 0.0,
            recon_blocks: 0,
            unusual_blocks: 0,
            dark_cells: 0,
            dark_blocks: 0,
            state: ReadoutState::Running,
            windows,
        }
    }

    pub fn is_running(&self) -> bool {
        self.state == ReadoutState::Running
    }

    /// Share of the region's cells measured.
    pub fn progress(&self) -> f32 {
        if self.cells.is_empty() {
            1.0
        } else {
            1.0 - self.remaining.len() as f32 / self.cells.len() as f32
        }
    }

    /// Adds a finished cell's blocks inside the region. Each cell counts
    /// once; cells the region does not need are ignored.
    pub fn consume(
        &mut self,
        cell: Idx,
        data: &CellData,
        grid: &Grid,
        recon_threshold: Option<f32>,
    ) {
        if !self.remaining.remove(&cell) {
            return;
        }
        match data {
            CellData::Empty => {}
            CellData::Dark => self.dark_cells += 1,
            CellData::Tissue(tissue) => {
                let per = grid.stride / BLOCK;
                for (index, block) in tissue.blocks.iter().enumerate() {
                    if block.coverage < MIN_COVERAGE || block.od < TISSUE_OD {
                        continue;
                    }
                    let (x, y) = grid.block_center(cell, index % per, index / per);
                    if !self.region.contains(x, y) {
                        continue;
                    }
                    if block.dark > MAX_DARK {
                        self.dark_blocks += 1;
                        continue;
                    }
                    for (fit, mean) in self.fits.iter_mut().zip(&block.means) {
                        fit.add(f64::from(block.od), f64::from(*mean));
                    }
                    if let Some(recon) = block.recon {
                        self.recon_sum += f64::from(recon);
                        self.recon_blocks += 1;
                        if recon_threshold.is_some_and(|t| recon > t) {
                            self.unusual_blocks += 1;
                        }
                    }
                }
            }
        }
        if self.remaining.is_empty() && self.state == ReadoutState::Running {
            self.state = ReadoutState::Done;
        }
    }

    /// Tissue blocks measured so far.
    pub fn blocks(&self) -> usize {
        self.fits.first().map_or(0, Fit::blocks)
    }

    pub fn mean_recon(&self) -> Option<f64> {
        (self.recon_blocks > 0).then(|| self.recon_sum / self.recon_blocks as f64)
    }
}

/// `2026-10-15T12:34:56Z` for a time since the Unix epoch.
pub fn utc_timestamp(since_epoch: std::time::Duration) -> String {
    let secs = since_epoch.as_secs() as i64;
    let (days, rem) = (secs.div_euclid(86_400), secs.rem_euclid(86_400));
    // Civil date from days (Howard Hinnant's algorithm).
    let z = days + 719_468;
    let era = z.div_euclid(146_097);
    let doe = z.rem_euclid(146_097);
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let day = doy - (153 * mp + 2) / 5 + 1;
    let month = if mp < 10 { mp + 3 } else { mp - 9 };
    let year = yoe + era * 400 + i64::from(month <= 2);
    format!(
        "{year:04}-{month:02}-{day:02}T{:02}:{:02}:{:02}Z",
        rem / 3600,
        rem / 60 % 60,
        rem % 60
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::blend::{BlockStat, TissueCell};
    use crate::geometry::Scale;
    use eov_plugin_api::protocol::FileInfo;

    fn grid(mpp: f64) -> Grid {
        let info = FileInfo {
            file_id: 1,
            width: 100_000,
            height: 100_000,
            mpp_x: Some(mpp),
            mpp_y: Some(mpp),
            ..Default::default()
        };
        Grid::new(&info, Scale::of(&info, 0.325, false).unwrap(), 0.325, 256).unwrap()
    }

    fn square(left: f64, top: f64, side: f64) -> Region {
        Region::from_shape(&Shape::Polygon {
            exterior: vec![
                [left, top],
                [left + side, top],
                [left + side, top + side],
                [left, top + side],
            ],
            holes: vec![],
        })
        .unwrap()
    }

    #[test]
    fn a_concave_outline_excludes_its_notch_and_holes_are_outside() {
        let l_shape = Region::from_shape(&Shape::Polygon {
            exterior: vec![
                [0.0, 0.0],
                [10.0, 0.0],
                [10.0, 4.0],
                [4.0, 4.0],
                [4.0, 10.0],
                [0.0, 10.0],
            ],
            holes: vec![],
        })
        .unwrap();
        assert!(l_shape.contains(2.0, 8.0));
        assert!(l_shape.contains(8.0, 2.0));
        assert!(!l_shape.contains(8.0, 8.0), "the notch counted as inside");

        let ring = Region::from_shape(&Shape::Polygon {
            exterior: vec![[0.0, 0.0], [10.0, 0.0], [10.0, 10.0], [0.0, 10.0]],
            holes: vec![vec![[3.0, 3.0], [7.0, 3.0], [7.0, 7.0], [3.0, 7.0]]],
        })
        .unwrap();
        assert!(ring.contains(1.0, 1.0));
        assert!(!ring.contains(5.0, 5.0), "the hole counted as inside");
        assert!((ring.area() - 84.0).abs() < 1e-9);

        assert!(Region::from_shape(&Shape::Point { x: 0.0, y: 0.0 }).is_none());
        let ellipse = Region::from_shape(&Shape::Ellipse {
            cx: 0.0,
            cy: 0.0,
            rx: 10.0,
            ry: 5.0,
        })
        .unwrap();
        assert!(ellipse.contains(9.0, 0.0) && !ellipse.contains(0.0, 6.0));
    }

    #[test]
    fn r2_is_one_on_a_line_and_absent_when_it_cannot_be_measured() {
        let mut line = Fit::default();
        for i in 0..50 {
            line.add(i as f64, 3.0 - 2.0 * i as f64);
        }
        assert!((line.r2().unwrap() - 1.0).abs() < 1e-9);

        let mut few = Fit::default();
        for i in 0..MIN_BLOCKS - 1 {
            few.add(i as f64, i as f64);
        }
        assert!(few.r2().is_none(), "r² reported from too few blocks");

        let mut flat = Fit::default();
        for i in 0..50 {
            flat.add(i as f64, 5.0);
        }
        assert!(flat.r2().is_none(), "a constant prediction has no r²");
        assert!((flat.mean() - 5.0).abs() < 1e-12);
        assert!(flat.spread() < 1e-6);

        let mut two = Fit::default();
        two.add(0.0, 4.0);
        two.add(0.0, 6.0);
        assert!((two.spread() - 1.0).abs() < 1e-12);
    }

    #[test]
    fn a_region_needs_the_cells_under_its_blocks_only() {
        let g = grid(0.325);
        // One cell is 128 px at 0.325 µm/px: a 100 px square inside cell
        // (1, 1) needs that cell alone.
        let cells = square(140.0, 140.0, 100.0).cells(&g, 100);
        assert_eq!(cells, vec![Idx::new(1, 1)]);
        let big = square(0.0, 0.0, 128.0 * 30.0).cells(&g, 100);
        assert_eq!(big.len(), 101, "the walk did not stop at the limit");
    }

    fn tissue(per: usize, od: f32, means: impl Fn(usize) -> f32) -> CellData {
        CellData::Tissue(std::sync::Arc::new(TissueCell {
            planes: vec![],
            mask: vec![],
            blocks: (0..per * per)
                .map(|index| BlockStat {
                    od,
                    coverage: 1.0,
                    dark: 0.0,
                    means: vec![means(index)],
                    recon: Some(0.1),
                })
                .collect(),
        }))
    }

    #[test]
    fn blocks_outside_the_region_glass_and_off_slide_are_left_out() {
        let g = grid(0.325);
        // Cell (0, 0) spans 0..128; the region keeps its left half.
        let region = square(0.0, 0.0, 64.0);
        let cells = region.cells(&g, 10);
        let mut readout = Readout::new(1, region, cells.clone(), 1, 4);
        readout.consume(cells[0], &tissue(8, 0.3, |i| i as f32), &g, Some(0.05));
        // 4 × 4 of the cell's 8 × 8 blocks have their centre inside.
        assert_eq!(readout.blocks(), 16);
        assert_eq!(readout.state, ReadoutState::Done);
        assert_eq!(readout.unusual_blocks, 16);
        // A second delivery of the same cell is ignored.
        readout.consume(cells[0], &tissue(8, 0.3, |_| 0.0), &g, None);
        assert_eq!(readout.blocks(), 16);

        let mut glass = Readout::new(2, square(0.0, 0.0, 64.0), cells.clone(), 1, 4);
        glass.consume(cells[0], &tissue(8, 0.01, |_| 1.0), &g, None);
        assert_eq!(glass.blocks(), 0, "glass blocks were measured");
    }

    #[test]
    fn blocks_are_the_same_tissue_on_every_scanner() {
        // A 0.5 mm square holds the same number of blocks at 0.25 and 0.5
        // µm/px, because blocks are 5.2 µm, not 16 slide pixels.
        let counts: Vec<usize> = [0.25, 0.5]
            .iter()
            .map(|mpp| {
                let g = grid(*mpp);
                let side = 500.0 / mpp;
                let region = square(10_000.0, 10_000.0, side);
                let cells = region.cells(&g, 10_000);
                let mut readout = Readout::new(1, region, cells.clone(), 1, 0);
                for cell in cells {
                    readout.consume(cell, &tissue(8, 0.3, |_| 1.0), &g, None);
                }
                readout.blocks()
            })
            .collect();
        let expected = (500.0f64 / (16.0 * 0.325)).powi(2);
        for count in &counts {
            let off = (*count as f64 - expected).abs() / expected;
            assert!(off < 0.03, "{count} blocks, expected about {expected:.0}");
        }
    }

    #[test]
    fn timestamps_are_utc_dates() {
        use std::time::Duration;
        assert_eq!(utc_timestamp(Duration::ZERO), "1970-01-01T00:00:00Z");
        assert_eq!(
            utc_timestamp(Duration::from_secs(1_791_984_896)),
            "2026-10-14T13:34:56Z"
        );
        assert_eq!(
            utc_timestamp(Duration::from_secs(951_782_400)),
            "2000-02-29T00:00:00Z"
        );
    }
}
