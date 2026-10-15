//! Resampling a read region onto the model grid, and what a window holds.
//!
//! Windows are read with straight alpha. Pixels the slide has no data for
//! (outside the scanned area, transparent tiles) are composited onto white,
//! which is what glass looks like to the model, and their coverage is kept
//! so that they are never painted or measured (CEL-7).
//!
//! Downsampling uses an exact area (box) filter, so no detail aliases into
//! the model's input; the rare upsampling case is bilinear.

use crate::geometry::{BLOCK, ReadPlan};
use eov_plugin_api::sdk::Image;

/// Block mean optical density above which a block counts as tissue: the gap
/// between glass (≈0.01) and tissue (from ≈0.06) on CRC11. Kept from the
/// first readout, which measured it.
pub const TISSUE_OD: f32 = 0.04;
/// Smoothed optical density at which the paint mask starts and saturates.
pub const MASK_LOW: f32 = 0.02;
pub const MASK_HIGH: f32 = 0.06;
/// Radius, in model pixels, of the box blur under the paint mask.
const MASK_BLUR: usize = 4;
/// Pixels darker than this in every channel count as "dark".
const DARK_LEVEL: f32 = 30.0 / 255.0;
/// A window that is mostly dark is not a bright-field image.
const DARK_SHARE: f32 = 0.6;

/// A window on the model grid.
#[derive(Clone, Debug, PartialEq)]
pub struct WindowImage {
    pub side: usize,
    /// RGB composited onto white, planar (`3 × side²`), 0–1.
    pub rgb: Vec<f32>,
    /// The share of each pixel the slide has data for, 0–1.
    pub coverage: Vec<f32>,
}

/// Source pixels contributing to one output pixel, with their weights.
#[derive(Clone, Debug, PartialEq)]
struct Tap {
    first: usize,
    weights: Vec<f32>,
}

/// Taps along one axis: output pixel `a` covers source `[origin + a·step,
/// origin + (a+1)·step)`.
fn axis_taps(origin: f64, step: f64, count: usize, len: usize) -> Vec<Tap> {
    let last = len.saturating_sub(1);
    (0..count)
        .map(|a| {
            let mut pairs: Vec<(usize, f64)> = Vec::new();
            if step >= 1.0 {
                let lo = origin + a as f64 * step;
                let hi = lo + step;
                let mut k = lo.floor();
                while k < hi {
                    let overlap = (k + 1.0).min(hi) - k.max(lo);
                    if overlap > 0.0 {
                        pairs.push(((k.max(0.0) as usize).min(last), overlap));
                    }
                    k += 1.0;
                }
            } else {
                // Bilinear at the pixel centre (pixel k's centre is k + 0.5).
                let centre = origin + (a as f64 + 0.5) * step - 0.5;
                let k = centre.floor();
                let t = centre - k;
                let at = |k: f64| (k.max(0.0) as usize).min(last);
                pairs.push((at(k), 1.0 - t));
                pairs.push((at(k + 1.0), t));
            }
            let first = pairs.iter().map(|p| p.0).min().unwrap_or(0);
            let end = pairs.iter().map(|p| p.0).max().unwrap_or(0) + 1;
            let mut weights = vec![0f64; end - first];
            for (index, weight) in pairs {
                weights[index - first] += weight;
            }
            let total: f64 = weights.iter().sum();
            Tap {
                first,
                weights: weights
                    .into_iter()
                    .map(|w| if total > 0.0 { (w / total) as f32 } else { 0.0 })
                    .collect(),
            }
        })
        .collect()
}

/// Resamples a region read for `plan` onto a `side`×`side` window.
pub fn resample(image: &Image, plan: &ReadPlan, side: usize) -> Result<WindowImage, String> {
    let (width, height) = (image.width as usize, image.height as usize);
    if width == 0 || height == 0 || image.data.len() != width * height * 4 {
        return Err(format!(
            "the host returned {} bytes for a {width}\u{d7}{height} region",
            image.data.len()
        ));
    }
    if (width, height) != (plan.request.width as usize, plan.request.height as usize) {
        return Err(format!(
            "asked for {}\u{d7}{} pixels, got {width}\u{d7}{height}",
            plan.request.width, plan.request.height
        ));
    }

    // Premultiplied planes: r·a, g·a, b·a, a.
    let pixels = width * height;
    let mut source = vec![0f32; 4 * pixels];
    for (index, px) in image.data.as_chunks::<4>().0.iter().enumerate() {
        let alpha = f32::from(px[3]) / 255.0;
        for channel in 0..3 {
            source[channel * pixels + index] = f32::from(px[channel]) / 255.0 * alpha;
        }
        source[3 * pixels + index] = alpha;
    }

    let taps_x = axis_taps(plan.origin_x, plan.step_x, side, width);
    let taps_y = axis_taps(plan.origin_y, plan.step_y, side, height);

    // Horizontal pass: 4 × height × side.
    let mut rows = vec![0f32; 4 * height * side];
    for channel in 0..4 {
        let plane = &source[channel * pixels..(channel + 1) * pixels];
        let out = &mut rows[channel * height * side..(channel + 1) * height * side];
        for y in 0..height {
            let line = &plane[y * width..(y + 1) * width];
            for (a, tap) in taps_x.iter().enumerate() {
                out[y * side + a] = tap
                    .weights
                    .iter()
                    .zip(&line[tap.first..])
                    .map(|(w, v)| w * v)
                    .sum();
            }
        }
    }

    // Vertical pass, then composite onto white.
    let area = side * side;
    let mut premultiplied = vec![0f32; 4 * area];
    for channel in 0..4 {
        let plane = &rows[channel * height * side..(channel + 1) * height * side];
        let out = &mut premultiplied[channel * area..(channel + 1) * area];
        for (b, tap) in taps_y.iter().enumerate() {
            for a in 0..side {
                out[b * side + a] = tap
                    .weights
                    .iter()
                    .enumerate()
                    .map(|(k, w)| w * plane[(tap.first + k) * side + a])
                    .sum();
            }
        }
    }
    let coverage: Vec<f32> = premultiplied[3 * area..]
        .iter()
        .map(|a| a.clamp(0.0, 1.0))
        .collect();
    let mut rgb = vec![0f32; 3 * area];
    for channel in 0..3 {
        for index in 0..area {
            rgb[channel * area + index] =
                (premultiplied[channel * area + index] + (1.0 - coverage[index])).clamp(0.0, 1.0);
        }
    }
    Ok(WindowImage {
        side,
        rgb,
        coverage,
    })
}

/// The model's input for a window: planar RGB scaled to [−1, 1].
pub fn model_input(window: &WindowImage, out: &mut [f32]) {
    for (target, value) in out.iter_mut().zip(&window.rgb) {
        *target = value * 2.0 - 1.0;
    }
}

/// Optical density of a 0–1 channel value, as the first readout defined it
/// for 8-bit values: −log10((c + 1) / 256).
pub fn optical_density(value: f32) -> f32 {
    -((value * 255.0 + 1.0) / 256.0).log10()
}

/// What a window holds.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum WindowKind {
    /// Tissue to predict.
    Tissue,
    /// Glass or no data only: nothing to predict, paint or measure.
    Background,
    /// Mostly near-black: not a bright-field image (fluorescence, pen,
    /// debris); the model was not trained on anything like it.
    Dark,
}

/// Optical density and coverage of one readout block.
#[derive(Clone, Copy, Debug, PartialEq, Default)]
pub struct BlockBase {
    pub od: f32,
    pub coverage: f32,
}

/// A window's tissue mask and block statistics.
#[derive(Clone, Debug, PartialEq)]
pub struct Analysis {
    pub kind: WindowKind,
    /// Paint mask, `side²`, 0–1.
    pub mask: Vec<f32>,
    /// Blocks of [`BLOCK`] model pixels, row-major, `(side / BLOCK)²`.
    pub blocks: Vec<BlockBase>,
}

fn smoothstep(low: f32, high: f32, value: f32) -> f32 {
    let t = ((value - low) / (high - low)).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

/// Box blur with clamped edges (each output averages the pixels that exist).
fn box_blur(plane: &[f32], side: usize, radius: usize) -> Vec<f32> {
    let pass = |input: &[f32], horizontal: bool| {
        let mut out = vec![0f32; input.len()];
        for line in 0..side {
            let at = |k: usize| {
                if horizontal {
                    line * side + k
                } else {
                    k * side + line
                }
            };
            let mut prefix = vec![0f32; side + 1];
            for k in 0..side {
                prefix[k + 1] = prefix[k] + input[at(k)];
            }
            for k in 0..side {
                let lo = k.saturating_sub(radius);
                let hi = (k + radius + 1).min(side);
                out[at(k)] = (prefix[hi] - prefix[lo]) / (hi - lo) as f32;
            }
        }
        out
    };
    pass(&pass(plane, true), false)
}

/// Classifies a window and computes its paint mask and block statistics.
pub fn analyse(window: &WindowImage) -> Analysis {
    let side = window.side;
    let area = side * side;
    let mut od = vec![0f32; area];
    let mut dark = 0usize;
    let mut covered = 0usize;
    for (index, density) in od.iter_mut().enumerate() {
        let r = window.rgb[index];
        let g = window.rgb[area + index];
        let b = window.rgb[2 * area + index];
        *density = (optical_density(r) + optical_density(g) + optical_density(b)) / 3.0;
        if window.coverage[index] >= 0.5 {
            covered += 1;
            if r.max(g).max(b) < DARK_LEVEL {
                dark += 1;
            }
        }
    }
    let blurred = box_blur(&od, side, MASK_BLUR);
    let mask: Vec<f32> = blurred
        .iter()
        .zip(&window.coverage)
        .map(|(od, coverage)| smoothstep(MASK_LOW, MASK_HIGH, *od) * coverage)
        .collect();

    let per = side / BLOCK;
    let mut blocks = vec![BlockBase::default(); per * per];
    for (index, block) in blocks.iter_mut().enumerate() {
        let (bx, by) = (index % per, index / per);
        let mut od_sum = 0f32;
        let mut coverage_sum = 0f32;
        for y in by * BLOCK..(by + 1) * BLOCK {
            for x in bx * BLOCK..(bx + 1) * BLOCK {
                od_sum += od[y * side + x];
                coverage_sum += window.coverage[y * side + x];
            }
        }
        let n = (BLOCK * BLOCK) as f32;
        *block = BlockBase {
            od: od_sum / n,
            coverage: coverage_sum / n,
        };
    }

    let kind = if covered > 0 && dark as f32 > DARK_SHARE * covered as f32 {
        WindowKind::Dark
    } else if mask.iter().all(|m| *m <= 0.0) {
        WindowKind::Background
    } else {
        WindowKind::Tissue
    };
    Analysis { kind, mask, blocks }
}

#[cfg(test)]
mod tests {
    use super::*;
    use eov_plugin_api::abi::RegionRequest;

    fn plan(width: u32, height: u32, origin_x: f64, origin_y: f64, step: f64) -> ReadPlan {
        ReadPlan {
            request: RegionRequest {
                file_id: 1,
                level: 0,
                x: 0,
                y: 0,
                width,
                height,
                background: 0,
            },
            origin_x,
            origin_y,
            step_x: step,
            step_y: step,
        }
    }

    fn image(width: u32, height: u32, f: impl Fn(u32, u32) -> [u8; 4]) -> Image {
        let mut data = Vec::with_capacity((width * height * 4) as usize);
        for y in 0..height {
            for x in 0..width {
                data.extend_from_slice(&f(x, y));
            }
        }
        Image {
            width,
            height,
            data,
        }
    }

    #[test]
    fn taps_conserve_weight_and_follow_the_interval() {
        let taps = axis_taps(2.0, 1.3, 10, 20);
        for tap in &taps {
            let total: f32 = tap.weights.iter().sum();
            assert!((total - 1.0).abs() < 1e-6);
        }
        // Output 0 covers [2.0, 3.3): all of pixel 2, 0.3 of pixel 3.
        assert_eq!(taps[0].first, 2);
        assert!((taps[0].weights[0] - 1.0 / 1.3).abs() < 1e-6);
        assert!((taps[0].weights[1] - 0.3 / 1.3).abs() < 1e-6);
        // Integer steps average whole pixels.
        let taps = axis_taps(0.0, 2.0, 3, 6);
        assert_eq!(taps[1].first, 2);
        assert_eq!(taps[1].weights, vec![0.5, 0.5]);
    }

    #[test]
    fn a_ramp_survives_downsampling_at_the_right_positions() {
        // Grey level = source column, read at 2 source px per model px.
        let src = image(40, 40, |x, _| [x as u8 * 5, x as u8 * 5, x as u8 * 5, 255]);
        let out = resample(&src, &plan(40, 40, 2.0, 2.0, 2.0), 16).unwrap();
        for a in 0..16 {
            // Model pixel a averages source columns 2 + 2a and 3 + 2a.
            let expected = ((2 + 2 * a) as f32 + 0.5) * 5.0 / 255.0;
            assert!((out.rgb[a] - expected).abs() < 1e-5, "{a}: {}", out.rgb[a]);
        }
        assert!(out.coverage.iter().all(|c| (*c - 1.0).abs() < 1e-6));
    }

    #[test]
    fn transparency_becomes_white_with_its_coverage_kept() {
        // Left half has no data (0,0,0,0), right half is opaque black.
        let src = image(20, 8, |x, _| if x < 10 { [0; 4] } else { [0, 0, 0, 255] });
        let out = resample(&src, &plan(20, 8, 2.0, 2.0, 1.0), 4).unwrap();
        let area = 16;
        // Model pixels 0..4 cover source columns 2..6: no data.
        assert!(
            (out.rgb[0] - 1.0).abs() < 1e-6,
            "no data is white, not black"
        );
        assert!(out.coverage[0] < 1e-6);
        let half = image(20, 8, |x, _| if x < 10 { [0; 4] } else { [0, 0, 0, 255] });
        let straddle = resample(&half, &plan(20, 8, 8.0, 2.0, 1.0), 4).unwrap();
        // Model pixel 1 covers source column 9..10 (none), 2 covers 10..11.
        assert!(straddle.coverage[1] < 1e-6 && (straddle.coverage[2] - 1.0).abs() < 1e-6);
        assert!(straddle.rgb[2 * area + 2] < 1e-6, "data stays black");
    }

    #[test]
    fn mismatched_regions_are_reported_not_trusted() {
        let src = image(10, 10, |_, _| [0, 0, 0, 255]);
        assert!(resample(&src, &plan(12, 10, 1.0, 1.0, 1.0), 4).is_err());
        let broken = Image {
            width: 10,
            height: 10,
            data: vec![0; 7],
        };
        assert!(resample(&broken, &plan(10, 10, 1.0, 1.0, 1.0), 4).is_err());
    }

    fn window(side: usize, f: impl Fn(usize, usize) -> ([f32; 3], f32)) -> WindowImage {
        let area = side * side;
        let mut rgb = vec![0f32; 3 * area];
        let mut coverage = vec![0f32; area];
        for y in 0..side {
            for x in 0..side {
                let (c, cov) = f(x, y);
                for k in 0..3 {
                    rgb[k * area + y * side + x] = c[k];
                }
                coverage[y * side + x] = cov;
            }
        }
        WindowImage {
            side,
            rgb,
            coverage,
        }
    }

    #[test]
    fn glass_and_no_data_are_background_and_tissue_is_masked_in() {
        let glass = window(64, |_, _| ([0.98, 0.98, 0.98], 1.0));
        assert_eq!(analyse(&glass).kind, WindowKind::Background);
        let nothing = window(64, |_, _| ([1.0; 3], 0.0));
        assert_eq!(analyse(&nothing).kind, WindowKind::Background);

        // Pink tissue on the right half.
        let half = window(64, |x, _| {
            if x >= 32 {
                ([0.85, 0.55, 0.7], 1.0)
            } else {
                ([0.98; 3], 1.0)
            }
        });
        let analysis = analyse(&half);
        assert_eq!(analysis.kind, WindowKind::Tissue);
        assert!(analysis.mask[10 * 64 + 4] < 1e-6, "glass was painted");
        assert!(analysis.mask[10 * 64 + 60] > 0.99, "tissue was not painted");
        // Blocks: 4 per side; the right two columns are tissue.
        assert!(analysis.blocks[0].od < TISSUE_OD);
        assert!(analysis.blocks[3].od > TISSUE_OD);
        assert!((analysis.blocks[3].coverage - 1.0).abs() < 1e-6);
    }

    #[test]
    fn a_mostly_black_window_is_out_of_domain() {
        let dark = window(64, |_, _| ([0.02, 0.03, 0.02], 1.0));
        assert_eq!(analyse(&dark).kind, WindowKind::Dark);
        let dense = window(64, |_, _| ([0.35, 0.2, 0.45], 1.0));
        assert_eq!(analyse(&dense).kind, WindowKind::Tissue);
    }

    #[test]
    fn optical_density_matches_the_8_bit_table() {
        assert!(optical_density(1.0).abs() < 1e-6);
        assert!((optical_density(0.0) - 2.408_24).abs() < 1e-4);
    }

    #[test]
    fn the_model_sees_minus_one_to_one() {
        let w = window(2, |_, _| ([0.0, 0.5, 1.0], 1.0));
        let mut input = vec![0f32; 12];
        model_input(&w, &mut input);
        assert_eq!(input[0], -1.0);
        assert_eq!(input[4], 0.0);
        assert_eq!(input[8], 1.0);
    }
}
