//! Marker selection, fixed display windows, the palette, and compositing.

use serde::{Deserialize, Serialize};

/// Markers predicted by default: the ones whose gain over the controls
/// clears zero on Orion, and in SMA's case on Xenium too.
pub const DEFAULT_MARKERS: [&str; 3] = ["Hoechst", "Pan-CK", "SMA"];

/// At most this many markers are computed at once.
pub const MAX_MARKERS: usize = 6;

/// arcsinh(x / 5) at the 1st and 99.5th percentile of measured Orion IF,
/// 400 tiles across 16 slides. Fixed, so an empty tile looks empty.
const WINDOWS: [(&str, f32, f32); 19] = [
    ("Hoechst", 4.08, 8.82),
    ("AF1", 4.49, 6.73),
    ("CD31", 4.33, 5.77),
    ("CD45", 4.33, 7.26),
    ("CD68", 4.31, 6.24),
    ("Argo550", 4.20, 5.11),
    ("CD4", 4.37, 5.95),
    ("FOXP3", 4.21, 5.13),
    ("CD8a", 4.29, 6.10),
    ("CD45RO", 4.45, 6.37),
    ("CD20", 4.25, 5.70),
    ("PD-L1", 4.43, 6.05),
    ("CD3e", 4.25, 6.41),
    ("CD163", 3.35, 6.35),
    ("E-cadherin", 4.08, 7.14),
    ("PD-1", 4.34, 5.39),
    ("Ki67", 4.34, 5.91),
    ("Pan-CK", 4.17, 6.88),
    ("SMA", 3.87, 6.64),
];

/// The arcsinh cofactor of the display transform.
const COFACTOR: f32 = 5.0;

/// Colours in marker order. The first three (yellow, magenta, bluish green)
/// stay at least ΔE 60 apart under simulated protanopia, deuteranopia and
/// tritanopia, and are bright against both haematoxylin and eosin (CEL-9:
/// the first version's green Pan-CK and amber SMA were a red-green
/// confusion pair). The rest are Okabe–Ito colours for extra markers; past
/// three, toggle markers to tell them apart.
pub const PALETTE: [[u8; 3]; MAX_MARKERS] = [
    [0xF0, 0xE4, 0x42], // yellow
    [0xFF, 0x00, 0xFF], // magenta
    [0x00, 0xBE, 0x8C], // bluish green
    [0x56, 0xB4, 0xE9], // sky blue
    [0xE6, 0x9F, 0x00], // orange
    [0xD5, 0x5E, 0x00], // vermillion
];

/// A marker the plugin computes, resolved against the model's channels.
#[derive(Clone, Debug, PartialEq)]
pub struct Marker {
    pub name: String,
    /// Channel index in the model's output.
    pub channel: usize,
    /// Display window on the arcsinh scale.
    pub low: f32,
    pub high: f32,
    pub colour: [u8; 3],
}

impl Marker {
    /// One value on the arcsinh scale as 0–255 of the fixed window.
    pub fn display(&self, value: f32) -> u8 {
        let unit = ((value - self.low) / (self.high - self.low)).clamp(0.0, 1.0);
        (unit * 255.0).round() as u8
    }
}

/// Resolves markers by name (case-sensitive, surrounding blanks ignored)
/// against the model's own channel order.
pub fn select(channel_names: &[String], wanted: &[String]) -> Result<Vec<Marker>, String> {
    if wanted.is_empty() {
        return Err("no markers are selected".to_string());
    }
    if wanted.len() > MAX_MARKERS {
        return Err(format!(
            "at most {MAX_MARKERS} markers can be shown at once"
        ));
    }
    let mut markers: Vec<Marker> = Vec::with_capacity(wanted.len());
    for (index, name) in wanted.iter().enumerate() {
        let name = name.trim();
        if markers.iter().any(|m| m.name == name) {
            return Err(format!("{name} is selected twice"));
        }
        let channel = channel_names
            .iter()
            .position(|candidate| candidate.trim() == name)
            .ok_or_else(|| format!("the model has no channel named {name}"))?;
        let &(_, low, high) = WINDOWS
            .iter()
            .find(|(known, _, _)| *known == name)
            .ok_or_else(|| format!("no display window is known for {name}"))?;
        markers.push(Marker {
            name: name.to_string(),
            channel,
            low,
            high,
            colour: PALETTE[index],
        });
    }
    Ok(markers)
}

/// The model channels a display window is known for, in model order.
pub fn displayable(channel_names: &[String]) -> Vec<String> {
    channel_names
        .iter()
        .map(|name| name.trim().to_string())
        .filter(|name| WINDOWS.iter().any(|(known, _, _)| known == name))
        .collect()
}

/// A model output on the arcsinh scale the windows are in.
pub fn intensity(value: f32, raw_intensity: bool) -> f32 {
    if raw_intensity {
        (value / COFACTOR).asinh()
    } else {
        value
    }
}

/// How predicted markers are drawn over the H&E.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DisplayMode {
    /// Colour where a marker is predicted, H&E elsewhere: opacity follows
    /// the strongest marker.
    #[default]
    Overlay,
    /// Fluorescence style: additive colour on black over all tissue.
    Fluorescence,
}

/// Composites display-scaled marker planes (`markers × n`, 0–255) and a
/// tissue mask (`n`, 0–255) into straight-alpha RGBA.
pub fn composite(
    planes: &[u8],
    mask: &[u8],
    colours: &[[u8; 3]],
    visible: &[bool],
    mode: DisplayMode,
    out: &mut Vec<u8>,
) {
    let n = mask.len();
    out.clear();
    out.resize(n * 4, 0);
    let tints: Vec<[f32; 3]> = colours
        .iter()
        .map(|c| c.map(|lane| f32::from(lane) / 255.0))
        .collect();
    for index in 0..n {
        let tissue = f32::from(mask[index]) / 255.0;
        if tissue <= 0.0 {
            continue;
        }
        let mut rgb = [0f32; 3];
        let mut strongest = 0f32;
        for (k, tint) in tints.iter().enumerate() {
            if !visible.get(k).copied().unwrap_or(false) {
                continue;
            }
            let value = f32::from(planes[k * n + index]) / 255.0;
            strongest = strongest.max(value);
            for (lane, t) in rgb.iter_mut().zip(tint) {
                *lane += value * t;
            }
        }
        let (rgb, alpha) = match mode {
            DisplayMode::Overlay => {
                if strongest <= 0.0 {
                    continue;
                }
                (rgb.map(|lane| lane / strongest), strongest * tissue)
            }
            DisplayMode::Fluorescence => (rgb, tissue),
        };
        let px = &mut out[index * 4..index * 4 + 4];
        for (target, lane) in px.iter_mut().zip(rgb) {
            *target = (lane.clamp(0.0, 1.0) * 255.0).round() as u8;
        }
        px[3] = (alpha.clamp(0.0, 1.0) * 255.0).round() as u8;
    }
}

/// `#RRGGBB` of a colour.
pub fn hex(colour: [u8; 3]) -> String {
    format!("#{:02X}{:02X}{:02X}", colour[0], colour[1], colour[2])
}

/// The palette's colours by name, for legends that cannot show a swatch.
pub const PALETTE_NAMES: [&str; MAX_MARKERS] = [
    "yellow",
    "magenta",
    "bluish green",
    "sky blue",
    "orange",
    "vermilion",
];

/// A colour's name in the palette, or its hex code.
pub fn colour_name(colour: [u8; 3]) -> String {
    PALETTE
        .iter()
        .position(|c| *c == colour)
        .map_or_else(|| hex(colour), |k| PALETTE_NAMES[k].to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn orion() -> Vec<String> {
        WINDOWS
            .iter()
            .map(|(name, _, _)| name.to_string())
            .collect()
    }

    fn defaults() -> Vec<String> {
        DEFAULT_MARKERS.map(String::from).to_vec()
    }

    #[test]
    fn markers_resolve_by_name_not_position() {
        let markers = select(&orion(), &defaults()).unwrap();
        let channels: Vec<usize> = markers.iter().map(|m| m.channel).collect();
        assert_eq!(channels, vec![0, 17, 18], "Pan-CK is 17 and SMA 18, not 16");

        let mut shuffled = orion();
        shuffled.swap(16, 17);
        let moved = select(&shuffled, &defaults()).unwrap();
        assert_eq!(moved[1].channel, 16, "selection did not follow the model");
    }

    #[test]
    fn channel_names_with_blanks_still_resolve() {
        let names: Vec<String> = orion().iter().map(|n| format!(" {n} ")).collect();
        assert_eq!(select(&names, &defaults()).unwrap()[2].channel, 18);
    }

    #[test]
    fn missing_duplicate_or_unknown_markers_are_refused() {
        let mut names = orion();
        names.retain(|name| name != "SMA");
        assert!(select(&names, &defaults()).is_err());
        let twice = vec!["SMA".to_string(), "SMA".to_string()];
        assert!(select(&orion(), &twice).is_err());
        let mut odd = orion();
        odd.push("Mystery".into());
        assert!(select(&odd, &["Mystery".to_string()]).is_err());
        assert!(select(&orion(), &[]).is_err());
        assert_eq!(displayable(&odd).len(), 19);
    }

    #[test]
    fn the_window_is_fixed_not_stretched() {
        let marker = select(&orion(), &["Hoechst".to_string()])
            .unwrap()
            .remove(0);
        let quiet = marker.display(intensity((4.08f32).sinh() * COFACTOR, true));
        let bright = marker.display(intensity((8.82f32).sinh() * COFACTOR, true));
        assert!(quiet <= 1, "the window floor painted {quiet}");
        assert!(bright >= 254, "the window ceiling painted {bright}");
        assert_eq!(marker.display(intensity(0.0, true)), 0);
        assert_eq!(intensity(5.0, false), 5.0);
    }

    #[test]
    fn overlay_mode_leaves_glass_and_unstained_tissue_alone() {
        let colours = [PALETTE[0], PALETTE[1]];
        // Pixels: glass, tissue with no signal, tissue with marker 0 at half.
        let planes = [0, 0, 128, 0, 0, 0];
        let mask = [0, 255, 255];
        let mut out = Vec::new();
        composite(
            &planes,
            &mask,
            &colours,
            &[true, true],
            DisplayMode::Overlay,
            &mut out,
        );
        assert_eq!(&out[0..4], &[0, 0, 0, 0], "glass was painted");
        assert_eq!(out[7], 0, "tissue without signal was darkened");
        assert_eq!(&out[8..11], &PALETTE[0], "a lone marker keeps its colour");
        assert!((127..=129).contains(&out[11]), "alpha follows the signal");

        // Hidden markers do not show.
        composite(
            &planes,
            &mask,
            &colours,
            &[false, true],
            DisplayMode::Overlay,
            &mut out,
        );
        assert_eq!(out[11], 0);

        // Fluorescence mode paints all tissue, black where nothing is.
        composite(
            &planes,
            &mask,
            &colours,
            &[true, true],
            DisplayMode::Fluorescence,
            &mut out,
        );
        assert_eq!(&out[4..8], &[0, 0, 0, 255]);
        assert_eq!(out[3], 0, "glass is still left alone");
    }

    // -- colour-vision deficiency ------------------------------------------

    /// Machado, Oliveira & Fernandes (2009), severity 1, linear RGB.
    const PROTAN: [[f64; 3]; 3] = [
        [0.152286, 1.052583, -0.204868],
        [0.114503, 0.786281, 0.099216],
        [-0.003882, -0.048116, 1.051998],
    ];
    const DEUTAN: [[f64; 3]; 3] = [
        [0.367322, 0.860646, -0.227968],
        [0.280085, 0.672501, 0.047413],
        [-0.011820, 0.042940, 0.968881],
    ];
    const TRITAN: [[f64; 3]; 3] = [
        [1.255528, -0.076749, -0.178779],
        [-0.078411, 0.930809, 0.147602],
        [0.004733, 0.691367, 0.303900],
    ];

    fn linear(c: u8) -> f64 {
        let c = f64::from(c) / 255.0;
        if c <= 0.04045 {
            c / 12.92
        } else {
            ((c + 0.055) / 1.055).powf(2.4)
        }
    }

    fn lab(rgb: [f64; 3]) -> [f64; 3] {
        let [r, g, b] = rgb;
        let x = (0.4124 * r + 0.3576 * g + 0.1805 * b) / 0.95047;
        let y = 0.2126 * r + 0.7152 * g + 0.0722 * b;
        let z = (0.0193 * r + 0.1192 * g + 0.9505 * b) / 1.08883;
        let f = |t: f64| {
            if t > 0.008856 {
                t.cbrt()
            } else {
                7.787 * t + 16.0 / 116.0
            }
        };
        [
            116.0 * f(y) - 16.0,
            500.0 * (f(x) - f(y)),
            200.0 * (f(y) - f(z)),
        ]
    }

    fn seen(colour: [u8; 3], matrix: Option<&[[f64; 3]; 3]>) -> [f64; 3] {
        let l = colour.map(linear);
        let simulated = match matrix {
            None => l,
            Some(m) => [0, 1, 2]
                .map(|i| (m[i][0] * l[0] + m[i][1] * l[1] + m[i][2] * l[2]).clamp(0.0, 1.0)),
        };
        lab(simulated)
    }

    fn distance(a: [f64; 3], b: [f64; 3]) -> f64 {
        ((a[0] - b[0]).powi(2) + (a[1] - b[1]).powi(2) + (a[2] - b[2]).powi(2)).sqrt()
    }

    #[test]
    fn the_default_markers_stay_distinct_for_colour_blind_viewers() {
        for (vision, matrix) in [
            ("normal", None),
            ("protanopia", Some(&PROTAN)),
            ("deuteranopia", Some(&DEUTAN)),
            ("tritanopia", Some(&TRITAN)),
        ] {
            for (a, first) in PALETTE[..3].iter().enumerate() {
                for (b, second) in PALETTE[..3].iter().enumerate().skip(a + 1) {
                    let d = distance(seen(*first, matrix), seen(*second, matrix));
                    assert!(
                        d > 50.0,
                        "{vision}: colours {a} and {b} only ΔE {d:.0} apart"
                    );
                }
            }
        }
        // The first version's Pan-CK green and SMA amber, for comparison.
        let old = distance(
            seen([26, 255, 64], Some(&DEUTAN)),
            seen([255, 191, 0], Some(&DEUTAN)),
        );
        assert!(
            old < 25.0,
            "the old pair was {old:.0} apart under deuteranopia"
        );
    }

    #[test]
    fn the_defaults_stand_out_from_haematoxylin_and_eosin() {
        for stain in [[75u8, 46, 131], [231, 165, 195]] {
            for colour in &PALETTE[..3] {
                let d = distance(seen(*colour, None), seen(stain, None));
                assert!(
                    d > 50.0,
                    "{} is only ΔE {d:.0} from {stain:?}",
                    hex(*colour)
                );
            }
        }
    }
}
