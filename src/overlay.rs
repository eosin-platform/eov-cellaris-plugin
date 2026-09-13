//! Marker selection, fixed display windows, and compositing.

/// Markers painted by default: the ones whose gain over the controls clears
/// zero on Orion, and in SMA's case on Xenium too.
pub const DEFAULT_MARKERS: [(&str, [f32; 3]); 3] = [
    ("Hoechst", [0.25, 0.45, 1.00]),
    ("Pan-CK", [0.10, 1.00, 0.25]),
    ("SMA", [1.00, 0.75, 0.00]),
];

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

const COFACTOR: f32 = 5.0;

#[derive(Clone, Copy, Debug)]
pub struct Marker {
    pub channel: usize,
    pub colour: [f32; 3],
    low: f32,
    high: f32,
}

/// Resolve markers by name against the model's own channel order.
pub fn select(
    channel_names: &[String],
    wanted: &[(&str, [f32; 3])],
) -> Result<Vec<Marker>, String> {
    wanted
        .iter()
        .map(|(name, colour)| {
            let channel = channel_names
                .iter()
                .position(|candidate| candidate == name)
                .ok_or_else(|| format!("model has no channel named {name}"))?;
            let &(_, low, high) = WINDOWS
                .iter()
                .find(|(known, _, _)| known == name)
                .ok_or_else(|| format!("no display window for {name}"))?;
            Ok(Marker {
                channel,
                colour: *colour,
                low,
                high,
            })
        })
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

/// One marker value as 0..=255 of its fixed window.
pub fn scale(value: f32, marker: &Marker, raw_intensity: bool) -> u8 {
    let y = intensity(value, raw_intensity);
    let unit = ((y - marker.low) / (marker.high - marker.low)).clamp(0.0, 1.0);
    (unit * 255.0).round() as u8
}

/// Additive colour of already-scaled marker values, clipped per lane.
pub fn composite(values: &[f32], markers: &[Marker]) -> [u8; 3] {
    let mut rgb = [0f32; 3];
    for (value, marker) in values.iter().zip(markers) {
        for (lane, tint) in rgb.iter_mut().zip(marker.colour) {
            *lane += value / 255.0 * tint;
        }
    }
    rgb.map(|lane| (lane.clamp(0.0, 1.0) * 255.0).round() as u8)
}

pub fn blend_pixel(base: &mut [u8], overlay: &[u8], alpha: u16) {
    let inverse = 255 - alpha;
    for lane in 0..3 {
        base[lane] =
            ((u16::from(base[lane]) * inverse + u16::from(overlay[lane]) * alpha) / 255) as u8;
    }
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

    #[test]
    fn markers_resolve_by_name_not_position() {
        let markers = select(&orion(), &DEFAULT_MARKERS).unwrap();
        let channels: Vec<usize> = markers.iter().map(|m| m.channel).collect();
        assert_eq!(channels, vec![0, 17, 18], "Pan-CK is 17 and SMA 18, not 16");

        let mut shuffled = orion();
        shuffled.swap(16, 17);
        let moved = select(&shuffled, &DEFAULT_MARKERS).unwrap();
        assert_eq!(
            moved[1].channel, 16,
            "selection did not follow the model's order"
        );
    }

    #[test]
    fn a_missing_marker_is_refused() {
        let mut names = orion();
        names.retain(|name| name != "SMA");
        assert!(select(&names, &DEFAULT_MARKERS).is_err());
    }

    #[test]
    fn the_window_is_fixed_not_stretched() {
        let marker = select(&orion(), &[("Hoechst", [1.0; 3])]).unwrap()[0];
        let quiet = scale((4.08f32).sinh() * COFACTOR, &marker, true);
        let bright = scale((8.82f32).sinh() * COFACTOR, &marker, true);
        assert!(quiet <= 1, "the window floor painted {quiet}");
        assert!(bright >= 254, "the window ceiling painted {bright}");
        assert_eq!(
            scale(0.0, &marker, true),
            0,
            "an empty pixel was lifted off zero"
        );
    }

    #[test]
    fn colours_add_and_clip() {
        let markers = select(&orion(), &DEFAULT_MARKERS).unwrap();
        assert_eq!(composite(&[0.0, 0.0, 0.0], &markers), [0, 0, 0]);
        let full = composite(&[255.0, 255.0, 255.0], &markers);
        assert!(
            full.iter().all(|lane| *lane == 255),
            "all three at full did not clip to white"
        );
    }
}
