//! Feathered blending of overlapping windows into cells.
//!
//! The model sees less context at the edge of its input than at the centre,
//! so unblended windows show their joins (a seam score of 4.88 against 0.91
//! blended, measured in cellaris). Windows sit at a stride of half their
//! size; every cell is covered by four of them and is the weighted mean of
//! their predictions, with raised-cosine weights that taper to ~0 at each
//! window's edge and sum to one wherever two windows overlap.
//!
//! The overlay *and* the region readout use these blended cells, so what is
//! measured is what is painted (CEL-5).

use crate::geometry::BLOCK;
use crate::markers::Marker;
use crate::resample::BlockBase;

/// Weights across one window side.
#[derive(Clone, Debug, PartialEq)]
pub struct Ramp {
    weights: Vec<f32>,
    stride: usize,
}

impl Ramp {
    /// Raised-cosine taper over each half of a `window`-wide side.
    pub fn new(window: usize) -> Self {
        let stride = window / 2;
        let mut weights = vec![1f32; window];
        for i in 0..stride {
            let t = std::f32::consts::PI * (i as f32 + 0.5) / stride as f32;
            let taper = (0.5 * (1.0 - t.cos())).max(1e-3);
            weights[i] = taper;
            weights[window - 1 - i] = taper;
        }
        Self { weights, stride }
    }

    pub fn weight(&self, index: usize) -> f32 {
        self.weights[index]
    }

    /// Sum of the weights at position `x` of a cell over the windows present
    /// along one axis (`low`: the window whose right half holds the cell;
    /// `high`: the one whose left half does).
    fn axis_total(&self, x: usize, low: bool, high: bool) -> f32 {
        let mut total = 0.0;
        if high {
            total += self.weights[x];
        }
        if low {
            total += self.weights[x + self.stride];
        }
        total
    }
}

/// The bit of a covering window, by the cell's quadrant in it.
pub fn bit(dx: usize, dy: usize) -> u8 {
    1 << (dy * 2 + dx)
}

/// The bits of the windows that exist over cell `(i, j)` (windows with a
/// negative index lie before the slide and are never computed).
pub fn expected_bits(i: i64, j: i64) -> u8 {
    let mut bits = 0;
    for dy in 0..2 {
        for dx in 0..2 {
            if i - dx as i64 >= 0 && j - dy as i64 >= 0 {
                bits |= bit(dx, dy);
            }
        }
    }
    bits
}

/// Weighted sums for one cell while its windows arrive.
#[derive(Clone, Debug)]
pub struct Accum {
    channels: usize,
    stride: usize,
    sums: Vec<f32>,
    received: u8,
    expected: u8,
    /// The cell's blocks, from the first window that arrived (the input is
    /// the same slide in every window).
    blocks: Option<Vec<BlockBase>>,
    pub last_used: u64,
}

impl Accum {
    pub fn new(channels: usize, stride: usize, expected: u8) -> Self {
        Self {
            channels,
            stride,
            sums: vec![0.0; channels * stride * stride],
            received: 0,
            expected,
            blocks: None,
            last_used: 0,
        }
    }

    pub fn has(&self, bit: u8) -> bool {
        self.received & bit != 0
    }

    pub fn is_complete(&self) -> bool {
        self.received & self.expected == self.expected
    }

    /// Adds one window's planes (`channels × window²`) for the cell in its
    /// quadrant `(dx, dy)`. A window that already contributed is ignored.
    #[allow(clippy::too_many_arguments)]
    pub fn add(
        &mut self,
        dx: usize,
        dy: usize,
        ramp: &Ramp,
        planes: &[f32],
        window: usize,
        window_blocks: &[BlockBase],
    ) {
        let b = bit(dx, dy);
        if self.has(b) || self.expected & b == 0 {
            return;
        }
        self.received |= b;
        let stride = self.stride;
        let plane = window * window;
        let (ox, oy) = (dx * stride, dy * stride);
        for channel in 0..self.channels {
            let source = &planes[channel * plane..(channel + 1) * plane];
            let target = &mut self.sums[channel * stride * stride..(channel + 1) * stride * stride];
            for y in 0..stride {
                let wy = ramp.weight(oy + y);
                let row = &source[(oy + y) * window + ox..(oy + y) * window + ox + stride];
                let out = &mut target[y * stride..(y + 1) * stride];
                for (x, (sum, value)) in out.iter_mut().zip(row).enumerate() {
                    *sum += ramp.weight(ox + x) * wy * value;
                }
            }
        }
        if self.blocks.is_none() {
            let per_window = window / BLOCK;
            let per_cell = stride / BLOCK;
            let (bx0, by0) = (dx * per_cell, dy * per_cell);
            let mut blocks = Vec::with_capacity(per_cell * per_cell);
            for by in 0..per_cell {
                for bx in 0..per_cell {
                    blocks.push(window_blocks[(by0 + by) * per_window + bx0 + bx]);
                }
            }
            self.blocks = Some(blocks);
        }
    }

    /// The blended planes (`channels × stride²`) and the cell's blocks.
    pub fn finish(self, ramp: &Ramp) -> (Vec<f32>, Vec<BlockBase>) {
        let stride = self.stride;
        let has = |dx, dy| self.expected & bit(dx, dy) != 0;
        // The expected windows form a product set, so weights separate.
        let left = has(1, 0) || has(1, 1);
        let up = has(0, 1) || has(1, 1);
        let wx: Vec<f32> = (0..stride)
            .map(|x| ramp.axis_total(x, left, true))
            .collect();
        let wy: Vec<f32> = (0..stride).map(|y| ramp.axis_total(y, up, true)).collect();
        let mut values = self.sums;
        for channel in 0..self.channels {
            let plane = &mut values[channel * stride * stride..(channel + 1) * stride * stride];
            for y in 0..stride {
                for x in 0..stride {
                    plane[y * stride + x] /= wx[x] * wy[y];
                }
            }
        }
        let per = stride / BLOCK;
        let blocks = self
            .blocks
            .unwrap_or_else(|| vec![BlockBase::default(); per * per]);
        (values, blocks)
    }
}

/// Per-block readout statistics of a finished cell.
#[derive(Clone, Debug, PartialEq)]
pub struct BlockStat {
    pub od: f32,
    pub coverage: f32,
    /// Mean predicted intensity per marker (arcsinh scale).
    pub means: Vec<f32>,
    /// Mean reconstruction error, when the model reports one.
    pub recon: Option<f32>,
}

/// A cell with tissue in it.
#[derive(Clone, Debug, PartialEq)]
pub struct TissueCell {
    /// Display-scaled markers, `markers × stride²`, 0–255.
    pub planes: Vec<u8>,
    /// Paint mask, `stride²`, 0–255.
    pub mask: Vec<u8>,
    /// Row-major blocks, `(stride / BLOCK)²`.
    pub blocks: Vec<BlockStat>,
}

/// A finished cell.
#[derive(Clone, Debug, PartialEq)]
pub enum CellData {
    /// Glass or no data: nothing painted or measured.
    Empty,
    /// Not bright-field H&E (mostly black); skipped.
    Dark,
    Tissue(Box<TissueCell>),
}

impl CellData {
    pub fn tissue(&self) -> Option<&TissueCell> {
        match self {
            CellData::Tissue(cell) => Some(cell),
            _ => None,
        }
    }
}

/// Channel layout of the planes a window contributes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Channels {
    pub markers: usize,
    pub recon: bool,
}

impl Channels {
    /// Markers, then the tissue mask, then the reconstruction error.
    pub fn count(&self) -> usize {
        self.markers + 1 + usize::from(self.recon)
    }

    pub fn mask(&self) -> usize {
        self.markers
    }

    pub fn recon(&self) -> Option<usize> {
        self.recon.then_some(self.markers + 1)
    }
}

/// Turns blended planes into display planes, a mask and block statistics.
pub fn finalize(
    values: &[f32],
    blocks: &[BlockBase],
    stride: usize,
    channels: Channels,
    markers: &[Marker],
) -> CellData {
    let area = stride * stride;
    let mask_plane = &values[channels.mask() * area..(channels.mask() + 1) * area];
    let mask: Vec<u8> = mask_plane
        .iter()
        .map(|m| (m.clamp(0.0, 1.0) * 255.0).round() as u8)
        .collect();
    if mask.iter().all(|m| *m == 0) {
        return CellData::Empty;
    }
    let mut planes = vec![0u8; markers.len() * area];
    for (k, marker) in markers.iter().enumerate() {
        for (out, value) in planes[k * area..(k + 1) * area]
            .iter_mut()
            .zip(&values[k * area..(k + 1) * area])
        {
            *out = marker.display(*value);
        }
    }
    let per = stride / BLOCK;
    let recon = channels.recon().map(|c| &values[c * area..(c + 1) * area]);
    let n = (BLOCK * BLOCK) as f32;
    let stats = blocks
        .iter()
        .enumerate()
        .map(|(index, base)| {
            let (bx, by) = (index % per, index / per);
            let mean = |plane: &[f32]| -> f32 {
                let mut sum = 0f32;
                for y in by * BLOCK..(by + 1) * BLOCK {
                    for x in bx * BLOCK..(bx + 1) * BLOCK {
                        sum += plane[y * stride + x];
                    }
                }
                sum / n
            };
            BlockStat {
                od: base.od,
                coverage: base.coverage,
                means: (0..markers.len())
                    .map(|k| mean(&values[k * area..(k + 1) * area]))
                    .collect(),
                recon: recon.map(mean),
            }
        })
        .collect();
    CellData::Tissue(Box::new(TissueCell {
        planes,
        mask,
        blocks: stats,
    }))
}

#[cfg(test)]
mod tests {
    use super::*;

    const WINDOW: usize = 64;
    const STRIDE: usize = 32;

    #[test]
    fn neighbouring_windows_sum_to_one() {
        let ramp = Ramp::new(256);
        for i in 0..128 {
            let total = ramp.weight(i) + ramp.weight(i + 128);
            assert!((total - 1.0).abs() <= 1e-3, "{total} at {i}");
        }
    }

    #[test]
    fn expected_windows_skip_those_before_the_slide() {
        assert_eq!(expected_bits(0, 0), bit(0, 0));
        assert_eq!(expected_bits(3, 0), bit(0, 0) | bit(1, 0));
        assert_eq!(expected_bits(2, 5), 0b1111);
    }

    /// Planes of one channel plus a full mask for a window at `(wi, wj)`,
    /// with values from a function of absolute model coordinates.
    fn window_planes(wi: i64, wj: i64, f: &dyn Fn(f32, f32) -> f32) -> Vec<f32> {
        let mut planes = vec![0f32; 2 * WINDOW * WINDOW];
        for y in 0..WINDOW {
            for x in 0..WINDOW {
                let u = (wi * STRIDE as i64) as f32 + x as f32;
                let v = (wj * STRIDE as i64) as f32 + y as f32;
                planes[y * WINDOW + x] = f(u, v);
                planes[WINDOW * WINDOW + y * WINDOW + x] = 1.0;
            }
        }
        planes
    }

    fn blend_cell(i: i64, j: i64, f: &dyn Fn(f32, f32) -> f32) -> Vec<f32> {
        let ramp = Ramp::new(WINDOW);
        let blocks = vec![BlockBase::default(); (WINDOW / BLOCK).pow(2)];
        let mut accum = Accum::new(2, STRIDE, expected_bits(i, j));
        for (dx, dy) in [(0usize, 0usize), (1, 0), (0, 1), (1, 1)] {
            let (wi, wj) = (i - dx as i64, j - dy as i64);
            if wi < 0 || wj < 0 {
                continue;
            }
            accum.add(dx, dy, &ramp, &window_planes(wi, wj, f), WINDOW, &blocks);
        }
        assert!(accum.is_complete());
        accum.finish(&ramp).0
    }

    #[test]
    fn a_consistent_field_comes_back_unchanged() {
        // Linear in position: blending is a partition of unity, so exact.
        let field = |u: f32, v: f32| 3.0 + 0.25 * u - 0.5 * v;
        for (i, j) in [(0, 0), (1, 0), (0, 3), (4, 2)] {
            let values = blend_cell(i, j, &field);
            for y in 0..STRIDE {
                for x in 0..STRIDE {
                    let u = (i * STRIDE as i64) as f32 + x as f32;
                    let v = (j * STRIDE as i64) as f32 + y as f32;
                    let got = values[y * STRIDE + x];
                    assert!((got - field(u, v)).abs() < 1e-3, "({i},{j}) {x},{y}: {got}");
                }
            }
        }
    }

    #[test]
    fn edge_bias_in_each_window_is_feathered_away() {
        // Every window adds +5 within 4 px of its own border: the bias a
        // model shows where it sees no context.
        let ramp = Ramp::new(WINDOW);
        let blocks = vec![BlockBase::default(); (WINDOW / BLOCK).pow(2)];
        let mut accum = Accum::new(2, STRIDE, expected_bits(2, 2));
        for (dx, dy) in [(0usize, 0usize), (1, 0), (0, 1), (1, 1)] {
            let mut planes = window_planes(2 - dx as i64, 2 - dy as i64, &|_, _| 1.0);
            for y in 0..WINDOW {
                for x in 0..WINDOW {
                    let edge = x.min(y).min(WINDOW - 1 - x).min(WINDOW - 1 - y);
                    if edge < 4 {
                        planes[y * WINDOW + x] += 5.0;
                    }
                }
            }
            accum.add(dx, dy, &ramp, &planes, WINDOW, &blocks);
        }
        let values = accum.finish(&ramp).0;
        let worst = values[..STRIDE * STRIDE]
            .iter()
            .fold(0f32, |m, v| m.max((v - 1.0).abs()));
        // Unblended, the join would read +5; the 64-px test window tapers
        // more steeply than the real 256-px one, which does better still.
        assert!(worst < 0.35, "a seam of {worst} survived blending");
    }

    #[test]
    fn a_window_is_counted_once() {
        let ramp = Ramp::new(WINDOW);
        let blocks = vec![BlockBase::default(); (WINDOW / BLOCK).pow(2)];
        let planes = window_planes(0, 0, &|_, _| 2.0);
        let mut accum = Accum::new(2, STRIDE, expected_bits(0, 0));
        accum.add(0, 0, &ramp, &planes, WINDOW, &blocks);
        accum.add(0, 0, &ramp, &planes, WINDOW, &blocks);
        let values = accum.finish(&ramp).0;
        assert!(
            values[..STRIDE * STRIDE]
                .iter()
                .all(|v| (v - 2.0).abs() < 1e-4)
        );
    }

    #[test]
    fn blocks_carry_the_blended_means() {
        let marker = Marker {
            name: "Hoechst".into(),
            channel: 0,
            low: 0.0,
            high: 10.0,
            colour: [255, 255, 0],
        };
        let channels = Channels {
            markers: 1,
            recon: false,
        };
        let values = blend_cell(1, 1, &|u, _| u);
        let per = STRIDE / BLOCK;
        let bases = vec![
            BlockBase {
                od: 0.3,
                coverage: 1.0
            };
            per * per
        ];
        let CellData::Tissue(cell) = finalize(&values, &bases, STRIDE, channels, &[marker]) else {
            panic!("a fully masked cell was not tissue");
        };
        // Block 0 spans u = 32..48: mean 39.5.
        assert!((cell.blocks[0].means[0] - 39.5).abs() < 1e-2);
        assert!((cell.blocks[1].means[0] - 55.5).abs() < 1e-2);
        assert_eq!(cell.blocks[0].od, 0.3);
        assert_eq!(
            cell.planes[0], 255,
            "values past the window clip in display only"
        );
    }

    #[test]
    fn a_cell_without_mask_is_empty() {
        let channels = Channels {
            markers: 1,
            recon: false,
        };
        let values = vec![0f32; 2 * STRIDE * STRIDE];
        let bases = vec![BlockBase::default(); (STRIDE / BLOCK).pow(2)];
        assert_eq!(
            finalize(&values, &bases, STRIDE, channels, &[]),
            CellData::Empty
        );
    }
}
