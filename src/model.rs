//! The marker model: what the rest of the plugin needs from it
//! ([`Predictor`]), how it is found, and its ONNX Runtime implementation.
//!
//! The model is a `markers.onnx` exported by cellaris: input `[B, 3, H, W]`
//! in [−1, 1], outputs `markers [B, C, H, W]` and optionally
//! `recon_error [B, 1, H, W]` (the VAE's reconstruction error, a hint of
//! how familiar the input looks). Its metadata names the channels
//! (`channel_names`), says whether outputs are raw intensities
//! (`raw_intensity`) and, ideally, the pixel size it was trained at (`mpp`,
//! µm/px). Everything read from the file is validated; nothing is guessed
//! silently, and every assumption shows up in the model card (CEL-11).

use crate::runtime;
use sha2::{Digest, Sha256};
use std::ffi::OsString;
use std::io::Read;
use std::path::{Path, PathBuf};
use std::sync::Arc;

/// The model file looked for by default.
pub const MODEL_FILE: &str = "markers.onnx";
/// The pixel size the Orion H&E was scanned at (level 0), assumed for
/// models that do not record theirs.
pub const DEFAULT_MPP: f64 = 0.325;
/// The size the model was trained on. It does not generalise past it: a
/// 384-px input puts a ring of +2 arcsinh round every tile.
pub const DEFAULT_WINDOW: usize = 256;

/// How many inputs one run takes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Batch {
    Dynamic,
    /// The graph was exported for exactly this many.
    Fixed(usize),
}

impl Batch {
    /// The batch size to use when `wanted` is asked for.
    pub fn size(&self, wanted: usize) -> usize {
        match self {
            Batch::Dynamic => wanted.max(1),
            Batch::Fixed(n) => *n,
        }
    }
}

/// Everything known about a loaded model, for the model card and for
/// provenance.
#[derive(Clone, Debug, PartialEq)]
pub struct ModelInfo {
    pub path: PathBuf,
    pub bytes: u64,
    /// Lower-case hex SHA-256 of the file.
    pub sha256: String,
    pub channel_names: Vec<String>,
    pub raw_intensity: bool,
    /// µm per pixel the model was trained at.
    pub mpp: f64,
    pub mpp_recorded: bool,
    /// Input side in pixels.
    pub window: usize,
    pub batch: Batch,
    /// The model reports a reconstruction error.
    pub recon: bool,
    /// Reconstruction error above which an input is unlike the training
    /// data, if the model records one.
    pub recon_threshold: Option<f32>,
    pub title: Option<String>,
    pub version: Option<String>,
    pub description: Option<String>,
    pub producer: Option<String>,
    pub training_data: Option<String>,
    pub license: Option<String>,
    /// e.g. "ONNX Runtime 1.22.0 from /usr/lib/libonnxruntime.so".
    pub runtime: String,
    pub threads: usize,
    /// Assumptions made while loading, shown in the model card.
    pub notes: Vec<String>,
}

impl ModelInfo {
    pub fn file_name(&self) -> String {
        self.path
            .file_name()
            .map(|n| n.to_string_lossy().into_owned())
            .unwrap_or_default()
    }

    pub fn short_hash(&self) -> &str {
        &self.sha256[..self.sha256.len().min(12)]
    }
}

/// One run's outputs.
#[derive(Clone, Debug, PartialEq)]
pub struct Prediction {
    pub n: usize,
    pub channels: usize,
    pub side: usize,
    /// `n × channels × side²`, model order.
    pub markers: Vec<f32>,
    /// `n × side²`.
    pub recon: Option<Vec<f32>>,
}

/// Asks a running prediction to stop, from any thread.
pub type Stopper = Arc<dyn Fn() + Send + Sync>;

/// A loaded model.
pub trait Predictor: Send {
    fn info(&self) -> &ModelInfo;

    /// Runs `n` windows; `input` is `n × 3 × side²` in [−1, 1].
    fn predict(&mut self, input: &[f32], n: usize) -> Result<Prediction, String>;

    /// A way to interrupt `predict` from another thread, if there is one.
    fn stopper(&self) -> Option<Stopper> {
        None
    }
}

/// What loading needs.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct LoadRequest {
    /// The model chosen in the panel.
    pub model: Option<PathBuf>,
    /// The runtime chosen in the panel.
    pub runtime: Option<PathBuf>,
    pub plugin_root: PathBuf,
    pub data_dir: PathBuf,
    /// Intra-op threads.
    pub threads: usize,
}

/// Why a model could not be loaded: a line for the user and the details.
#[derive(Clone, Debug, PartialEq)]
pub struct LoadError {
    pub summary: String,
    pub detail: String,
}

impl LoadError {
    pub fn new(summary: impl Into<String>, detail: impl Into<String>) -> Self {
        Self {
            summary: summary.into(),
            detail: detail.into(),
        }
    }
}

/// Loads models. The plugin is generic over it so that tests can run
/// without ONNX Runtime.
pub trait Backend: Send + Sync + Default + 'static {
    fn load(&self, request: &LoadRequest) -> Result<Box<dyn Predictor>, LoadError>;
}

// ---------------------------------------------------------------------------
// Locating the model
// ---------------------------------------------------------------------------

/// The model to load: `CELLARIS_MODEL`, the panel's choice, then
/// `markers.onnx` in the data directory or the plugin package. Explicit
/// choices that point nowhere are errors.
pub fn locate_model(
    env: Option<OsString>,
    setting: Option<&Path>,
    data_dir: &Path,
    plugin_root: &Path,
    exists: &dyn Fn(&Path) -> bool,
) -> Result<PathBuf, String> {
    if let Some(path) = env.map(PathBuf::from) {
        return if exists(&path) {
            Ok(path)
        } else {
            Err(format!(
                "CELLARIS_MODEL points at {}, which does not exist",
                path.display()
            ))
        };
    }
    if let Some(path) = setting {
        return if exists(path) {
            Ok(path.to_path_buf())
        } else {
            Err(format!(
                "the model chosen in the Cellaris panel, {}, does not exist",
                path.display()
            ))
        };
    }
    for dir in [data_dir, plugin_root] {
        let path = dir.join(MODEL_FILE);
        if exists(&path) {
            return Ok(path);
        }
    }
    Err(format!(
        "No {MODEL_FILE}: put one in {}, or choose a model in the Cellaris panel.",
        data_dir.display()
    ))
}

// ---------------------------------------------------------------------------
// Metadata and shapes
// ---------------------------------------------------------------------------

/// What the model's metadata says.
#[derive(Clone, Debug, PartialEq)]
pub struct Meta {
    pub channel_names: Vec<String>,
    pub raw_intensity: bool,
    pub mpp: f64,
    pub mpp_recorded: bool,
    pub window: Option<usize>,
    pub recon_threshold: Option<f32>,
    pub title: Option<String>,
    pub version: Option<String>,
    pub description: Option<String>,
    pub training_data: Option<String>,
    pub license: Option<String>,
    pub notes: Vec<String>,
}

/// Channel names as a comma-separated list or a JSON array.
pub fn parse_channel_names(text: &str) -> Result<Vec<String>, String> {
    let trimmed = text.trim();
    let names: Vec<String> = if trimmed.starts_with('[') {
        serde_json::from_str::<Vec<String>>(trimmed)
            .map_err(|e| format!("channel_names is not a list of names: {e}"))?
    } else {
        trimmed.split(',').map(str::to_string).collect()
    };
    let names: Vec<String> = names.iter().map(|n| n.trim().to_string()).collect();
    if names.is_empty() || names.iter().any(String::is_empty) {
        return Err(format!("channel_names has an empty name: {text:?}"));
    }
    for (index, name) in names.iter().enumerate() {
        if names[..index].contains(name) {
            return Err(format!("channel_names names {name} twice"));
        }
    }
    Ok(names)
}

/// A boolean flag; anything unrecognised is an error, not "true".
pub fn parse_flag(key: &str, text: &str) -> Result<bool, String> {
    match text.trim().to_ascii_lowercase().as_str() {
        "true" | "1" | "yes" => Ok(true),
        "false" | "0" | "no" => Ok(false),
        other => Err(format!("{key} must be true or false, not {other:?}")),
    }
}

fn parse_number<T: std::str::FromStr + PartialOrd + Copy + std::fmt::Display>(
    key: &str,
    text: &str,
    min: T,
    max: T,
) -> Result<T, String> {
    let value: T = text
        .trim()
        .parse()
        .map_err(|_| format!("{key} is not a number: {text:?}"))?;
    if value < min || value > max {
        return Err(format!("{key} = {value} is outside {min}–{max}"));
    }
    Ok(value)
}

/// Reads and validates the metadata through `get(key)`.
pub fn read_meta(get: &dyn Fn(&str) -> Option<String>) -> Result<Meta, String> {
    let first = |keys: &[&'static str]| keys.iter().find_map(|key| get(key).map(|v| (*key, v)));
    let mut notes = Vec::new();

    let channel_names = parse_channel_names(&get("channel_names").ok_or(
        "the model carries no channel_names, so its channels cannot be told apart; re-export it \
         with cellaris's export_marker_onnx",
    )?)?;

    let raw_intensity = match get("raw_intensity") {
        Some(text) => parse_flag("raw_intensity", &text)?,
        None => {
            notes.push(
                "raw_intensity is not recorded; outputs are taken to be raw intensities".into(),
            );
            true
        }
    };

    let (mpp, mpp_recorded) = match first(&["mpp", "microns_per_pixel", "mpp_um"]) {
        Some((key, text)) => (parse_number::<f64>(key, &text, 0.01, 100.0)?, true),
        None => {
            notes.push(format!(
                "the training pixel size is not recorded; {DEFAULT_MPP} \u{b5}m/px (Orion level 0) \
                 is assumed"
            ));
            (DEFAULT_MPP, false)
        }
    };

    let window = match first(&["tile_size", "window"]) {
        Some((key, text)) => Some(parse_number::<usize>(key, &text, 64, 1024)?),
        None => None,
    };
    let recon_threshold = match get("recon_error_threshold") {
        Some(text) => Some(parse_number::<f32>(
            "recon_error_threshold",
            &text,
            0.0,
            10.0,
        )?),
        None => None,
    };
    let text = |keys: &[&'static str]| {
        first(keys)
            .map(|(_, v)| v.trim().to_string())
            .filter(|v| !v.is_empty())
    };
    Ok(Meta {
        channel_names,
        raw_intensity,
        mpp,
        mpp_recorded,
        window,
        recon_threshold,
        title: text(&["name", "model_name"]),
        version: text(&["version", "model_version"]),
        description: text(&["description"]),
        training_data: text(&["training_data", "dataset"]),
        license: text(&["license"]),
        notes,
    })
}

/// The batch policy and fixed side (if any) of an input `[B, 3, H, W]`.
pub fn input_layout(dims: &[i64]) -> Result<(Batch, Option<usize>), String> {
    if dims.len() != 4 {
        return Err(format!(
            "the input has {} dimensions, not [batch, 3, height, width]",
            dims.len()
        ));
    }
    if dims[1] > 0 && dims[1] != 3 {
        return Err(format!("the input takes {} channels, not RGB", dims[1]));
    }
    let batch = if dims[0] > 0 {
        Batch::Fixed(dims[0] as usize)
    } else {
        Batch::Dynamic
    };
    let side = match (dims[2], dims[3]) {
        (h, w) if h > 0 && w > 0 && h != w => {
            return Err(format!("the input is {h}×{w}; only square inputs are used"));
        }
        (h, _) if h > 0 => Some(h as usize),
        (_, w) if w > 0 => Some(w as usize),
        _ => None,
    };
    Ok((batch, side))
}

/// Checks an output's shape against `[n, channels, side, side]`.
pub fn check_output(
    name: &str,
    dims: &[i64],
    n: usize,
    channels: usize,
    side: usize,
) -> Result<(), String> {
    let expected = [n as i64, channels as i64, side as i64, side as i64];
    if dims != expected {
        return Err(format!(
            "the model's {name} output is {dims:?}, expected {expected:?}"
        ));
    }
    Ok(())
}

/// The input side to use: the graph's, else the metadata's, else 256.
pub fn choose_window(fixed: Option<usize>, meta: Option<usize>) -> Result<usize, String> {
    let window = fixed.or(meta).unwrap_or(DEFAULT_WINDOW);
    if !(64..=1024).contains(&window) || !window.is_multiple_of(32) {
        return Err(format!(
            "the model's input of {window} px is not a multiple of 32 between 64 and 1024"
        ));
    }
    Ok(window)
}

/// Lower-case hex SHA-256 of a file.
pub fn hash_file(path: &Path) -> std::io::Result<(String, u64)> {
    let mut file = std::fs::File::open(path)?;
    let mut hasher = Sha256::new();
    let mut buffer = vec![0u8; 1 << 20];
    let mut bytes = 0u64;
    loop {
        let read = file.read(&mut buffer)?;
        if read == 0 {
            break;
        }
        bytes += read as u64;
        hasher.update(&buffer[..read]);
    }
    let digest = hasher.finalize();
    Ok((digest.iter().map(|b| format!("{b:02x}")).collect(), bytes))
}

/// Threads for inference: half the logical CPUs, at most 8, so tile
/// loading and the UI keep a share (CEL-8).
pub fn default_threads() -> usize {
    std::thread::available_parallelism()
        .map(|n| (n.get() / 2).clamp(1, 8))
        .unwrap_or(2)
}

// ---------------------------------------------------------------------------
// ONNX Runtime
// ---------------------------------------------------------------------------

/// Loads models with ONNX Runtime.
#[derive(Default)]
pub struct OrtBackend;

struct OrtPredictor {
    session: ort::session::Session,
    run_options: Arc<ort::session::RunOptions>,
    input: String,
    markers: String,
    recon: Option<String>,
    info: ModelInfo,
}

fn outlet_dims(outlet: &ort::value::Outlet) -> Option<Vec<i64>> {
    outlet.dtype().tensor_shape().map(|shape| shape.to_vec())
}

fn is_f32(outlet: &ort::value::Outlet) -> bool {
    outlet.dtype().tensor_type() == Some(ort::value::TensorElementType::Float32)
}

impl Backend for OrtBackend {
    fn load(&self, request: &LoadRequest) -> Result<Box<dyn Predictor>, LoadError> {
        let search = runtime::Search::from_env(
            request.runtime.clone(),
            Some(request.plugin_root.clone()),
            Some(request.data_dir.clone()),
        );
        let loaded = runtime::load(&search).map_err(|f| LoadError::new(f.summary, f.detail))?;
        let runtime = format!(
            "ONNX Runtime {} from {}",
            loaded.version.as_deref().unwrap_or("(version unknown)"),
            loaded.path.display()
        );

        let path = locate_model(
            std::env::var_os("CELLARIS_MODEL").filter(|v| !v.is_empty()),
            request.model.as_deref(),
            &request.data_dir,
            &request.plugin_root,
            &|path| path.is_file(),
        )
        .map_err(|reason| LoadError::new("No marker model", reason))?;
        let fail = |summary: &str, detail: String| {
            LoadError::new(summary, format!("{}: {detail}", path.display()))
        };
        let (sha256, bytes) =
            hash_file(&path).map_err(|e| fail("The model could not be read", e.to_string()))?;

        let threads = request.threads.max(1);
        let session = ort::session::Session::builder()
            .and_then(|b| b.with_intra_threads(threads).map_err(Into::into))
            .and_then(|mut b| b.commit_from_file(&path))
            .map_err(|e| fail("The model could not be opened", e.to_string()))?;

        let input = session
            .inputs()
            .iter()
            .find(|i| i.name() == "input")
            .or_else(|| session.inputs().first())
            .ok_or_else(|| fail("Not a marker model", "it has no inputs".into()))?;
        if !is_f32(input) {
            return Err(fail(
                "Not a marker model",
                "its input is not float32".into(),
            ));
        }
        let dims = outlet_dims(input)
            .ok_or_else(|| fail("Not a marker model", "its input is not a tensor".into()))?;
        let (batch, fixed_side) = input_layout(&dims).map_err(|e| fail("Not a marker model", e))?;
        let input_name = input.name().to_string();
        let mut notes = Vec::new();
        if session.inputs().len() > 1 {
            notes.push(format!(
                "the model has {} inputs; only {input_name} is fed",
                session.inputs().len()
            ));
        }

        let markers = session
            .outputs()
            .iter()
            .find(|o| o.name() == "markers")
            .ok_or_else(|| {
                fail(
                    "Not a marker model",
                    "it has no output named `markers`; not a cellaris export".into(),
                )
            })?;
        if !is_f32(markers) {
            return Err(fail("Not a marker model", "markers is not float32".into()));
        }
        let recon = session
            .outputs()
            .iter()
            .find(|o| o.name() == "recon_error")
            .filter(|o| is_f32(o))
            .map(|o| o.name().to_string());

        let metadata = session
            .metadata()
            .map_err(|e| fail("The model's metadata could not be read", e.to_string()))?;
        let meta = read_meta(&|key| metadata.custom(key))
            .map_err(|e| fail("The model's metadata is not usable", e))?;
        let producer = metadata.producer().filter(|p| !p.is_empty());
        let graph_title = metadata.name().filter(|n| !n.is_empty());
        drop(metadata);

        if let Some(dims) = outlet_dims(markers)
            && dims.len() == 4
            && dims[1] > 0
            && dims[1] as usize != meta.channel_names.len()
        {
            return Err(fail(
                "The model's metadata is not usable",
                format!(
                    "it names {} channels but outputs {}",
                    meta.channel_names.len(),
                    dims[1]
                ),
            ));
        }
        let window = choose_window(fixed_side, meta.window).map_err(|e| fail("Not usable", e))?;
        notes.extend(meta.notes);
        if recon.is_none() {
            notes.push("the model reports no reconstruction error".into());
        }

        let run_options = ort::session::RunOptions::new()
            .map(Arc::new)
            .map_err(|e| fail("ONNX Runtime failed", e.to_string()))?;
        let info = ModelInfo {
            path: path.clone(),
            bytes,
            sha256,
            channel_names: meta.channel_names,
            raw_intensity: meta.raw_intensity,
            mpp: meta.mpp,
            mpp_recorded: meta.mpp_recorded,
            window,
            batch,
            recon: recon.is_some(),
            recon_threshold: meta.recon_threshold,
            title: meta.title.or(graph_title),
            version: meta.version,
            description: meta.description,
            producer,
            training_data: meta.training_data,
            license: meta.license,
            runtime,
            threads,
            notes,
        };
        Ok(Box::new(OrtPredictor {
            session,
            run_options,
            input: input_name,
            markers: "markers".to_string(),
            recon,
            info,
        }))
    }
}

impl Predictor for OrtPredictor {
    fn info(&self) -> &ModelInfo {
        &self.info
    }

    fn predict(&mut self, input: &[f32], n: usize) -> Result<Prediction, String> {
        let side = self.info.window;
        let per = 3 * side * side;
        if n == 0 || input.len() != n * per {
            return Err(format!(
                "expected {n} windows of {per} values, got {}",
                input.len()
            ));
        }
        let run_n = match self.info.batch {
            Batch::Dynamic => n,
            Batch::Fixed(b) if b >= n => b,
            Batch::Fixed(b) => return Err(format!("the model takes {b} windows per run, not {n}")),
        };
        let padded;
        let data: &[f32] = if run_n > n {
            // White, which the model sees as glass.
            let mut buffer = input.to_vec();
            buffer.resize(run_n * per, 1.0);
            padded = buffer;
            &padded
        } else {
            input
        };
        let _ = self.run_options.unterminate();
        let tensor = ort::value::TensorRef::from_array_view(([run_n, 3, side, side], data))
            .map_err(|e| e.to_string())?;
        let outputs = self
            .session
            .run_with_options(
                ort::inputs![self.input.as_str() => tensor],
                &self.run_options,
            )
            .map_err(|e| e.to_string())?;
        let channels = self.info.channel_names.len();
        let markers = {
            let value = outputs
                .get(self.markers.as_str())
                .ok_or("the model returned no markers")?;
            let (shape, values) = value
                .try_extract_tensor::<f32>()
                .map_err(|e| e.to_string())?;
            check_output("markers", shape, run_n, channels, side)?;
            values[..n * channels * side * side].to_vec()
        };
        let recon = match &self.recon {
            Some(name) => {
                let value = outputs
                    .get(name.as_str())
                    .ok_or("recon_error went missing")?;
                let (shape, values) = value
                    .try_extract_tensor::<f32>()
                    .map_err(|e| e.to_string())?;
                check_output("recon_error", shape, run_n, 1, side)?;
                Some(values[..n * side * side].to_vec())
            }
            None => None,
        };
        Ok(Prediction {
            n,
            channels,
            side,
            markers,
            recon,
        })
    }

    fn stopper(&self) -> Option<Stopper> {
        let options = Arc::clone(&self.run_options);
        Some(Arc::new(move || {
            let _ = options.terminate();
        }))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::BTreeMap;

    fn meta(pairs: &[(&str, &str)]) -> Result<Meta, String> {
        let map: BTreeMap<String, String> = pairs
            .iter()
            .map(|(k, v)| (k.to_string(), v.to_string()))
            .collect();
        read_meta(&|key| map.get(key).cloned())
    }

    #[test]
    fn channel_names_are_trimmed_and_checked() {
        assert_eq!(
            parse_channel_names("Hoechst, AF1 ,CD31").unwrap(),
            vec!["Hoechst", "AF1", "CD31"]
        );
        assert_eq!(
            parse_channel_names(r#"["Hoechst", "Pan-CK"]"#).unwrap(),
            vec!["Hoechst", "Pan-CK"]
        );
        assert!(parse_channel_names("Hoechst,,SMA").is_err());
        assert!(parse_channel_names("SMA, SMA").is_err());
        assert!(parse_channel_names("[1, 2]").is_err());
    }

    #[test]
    fn raw_intensity_is_read_strictly() {
        assert!(!parse_flag("raw_intensity", " False ").unwrap());
        assert!(parse_flag("raw_intensity", "1").unwrap());
        assert!(
            parse_flag("raw_intensity", "maybe").is_err(),
            "anything but false used to count as raw"
        );
    }

    #[test]
    fn assumptions_are_noted_and_bad_values_refused() {
        let m = meta(&[("channel_names", "Hoechst,Pan-CK,SMA")]).unwrap();
        assert!(m.raw_intensity);
        assert_eq!((m.mpp, m.mpp_recorded), (DEFAULT_MPP, false));
        assert_eq!(m.notes.len(), 2, "{:?}", m.notes);

        let m = meta(&[
            ("channel_names", "Hoechst"),
            ("raw_intensity", "false"),
            ("mpp", "0.5"),
            ("recon_error_threshold", "0.12"),
            ("name", "cellaris markers"),
        ])
        .unwrap();
        assert_eq!((m.mpp, m.mpp_recorded), (0.5, true));
        assert_eq!(m.recon_threshold, Some(0.12));
        assert_eq!(m.title.as_deref(), Some("cellaris markers"));
        assert!(m.notes.is_empty());

        assert!(
            meta(&[]).is_err(),
            "a model without channel names was accepted"
        );
        assert!(meta(&[("channel_names", "A"), ("mpp", "-1")]).is_err());
        assert!(meta(&[("channel_names", "A"), ("mpp", "fast")]).is_err());
        assert!(meta(&[("channel_names", "A"), ("tile_size", "100000")]).is_err());
    }

    #[test]
    fn input_and_output_shapes_are_validated() {
        assert_eq!(
            input_layout(&[-1, 3, -1, -1]).unwrap(),
            (Batch::Dynamic, None)
        );
        assert_eq!(
            input_layout(&[32, 3, 256, 256]).unwrap(),
            (Batch::Fixed(32), Some(256))
        );
        assert!(input_layout(&[1, 4, 256, 256]).is_err());
        assert!(input_layout(&[1, 3, 256]).is_err());
        assert!(input_layout(&[1, 3, 256, 128]).is_err());

        assert!(check_output("markers", &[2, 19, 256, 256], 2, 19, 256).is_ok());
        // A smaller output used to panic the worker on a slice.
        assert!(check_output("markers", &[2, 3, 256, 256], 2, 19, 256).is_err());
        assert!(check_output("markers", &[2, 19, 128, 128], 2, 19, 256).is_err());

        assert_eq!(choose_window(None, None).unwrap(), 256);
        assert_eq!(choose_window(Some(512), Some(256)).unwrap(), 512);
        assert!(choose_window(None, Some(100)).is_err());
        assert_eq!(Batch::Fixed(4).size(2), 4);
        assert_eq!(Batch::Dynamic.size(0), 1);
    }

    #[test]
    fn the_model_is_found_where_documented() {
        let data = Path::new("/data/cellaris");
        let root = Path::new("/pkg");
        let nothing = |_: &Path| false;
        let everything = |_: &Path| true;
        let error = locate_model(None, None, data, root, &nothing).unwrap_err();
        assert!(error.contains("/data/cellaris"), "{error}");
        assert_eq!(
            locate_model(None, None, data, root, &everything).unwrap(),
            data.join(MODEL_FILE)
        );
        let bundled = |p: &Path| p.starts_with("/pkg");
        assert_eq!(
            locate_model(None, None, data, root, &bundled).unwrap(),
            root.join(MODEL_FILE)
        );
        let error = locate_model(Some("/gone.onnx".into()), None, data, root, &|p: &Path| {
            p != Path::new("/gone.onnx")
        })
        .unwrap_err();
        assert!(error.contains("CELLARIS_MODEL") && error.contains("/gone.onnx"));
        let chosen = Path::new("/models/m.onnx");
        assert_eq!(
            locate_model(None, Some(chosen), data, root, &everything).unwrap(),
            chosen
        );
        assert!(locate_model(None, Some(chosen), data, root, &nothing).is_err());
    }

    #[test]
    fn files_are_hashed() {
        let dir = std::env::temp_dir().join(format!("cellaris-hash-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("m.onnx");
        std::fs::write(&path, b"abc").unwrap();
        let (hash, bytes) = hash_file(&path).unwrap();
        assert_eq!(bytes, 3);
        assert_eq!(
            hash,
            "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
        );
        let _ = std::fs::remove_dir_all(dir);
    }
}
