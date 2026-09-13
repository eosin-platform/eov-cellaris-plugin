//! ONNX session and single-window inference.
//!
//! The model is a `markers.onnx` from `export_marker_onnx`, which records its
//! channel order and intensity convention in the model metadata. It is looked
//! for in `~/.eov/plugins/cellaris/`, or wherever `CELLARIS_MODEL` and
//! `CELLARIS_ORT_DYLIB` say.

use std::env;
use std::path::{Path, PathBuf};
use std::sync::{Mutex, OnceLock};

use ndarray::Array4;
use ort::session::Session;
use ort::value::TensorRef;

/// Where a plugin's own files live, beside the `.eop` that installed it.
fn data_dir() -> Option<PathBuf> {
    let home = env::var_os("HOME").or_else(|| env::var_os("USERPROFILE"))?;
    Some(
        PathBuf::from(home)
            .join(".eov")
            .join("plugins")
            .join("cellaris"),
    )
}

fn resolve(
    from_env: Option<PathBuf>,
    dir: Option<PathBuf>,
    file: &str,
    key: &str,
    exists: &dyn Fn(&Path) -> bool,
) -> Result<PathBuf, String> {
    if let Some(path) = from_env {
        return if exists(&path) {
            Ok(path)
        } else {
            Err(format!(
                "{key} points at {}, which does not exist",
                path.display()
            ))
        };
    }
    match dir.map(|dir| dir.join(file)) {
        Some(path) if exists(&path) => Ok(path),
        Some(path) => Err(format!(
            "no {file}: put one at {}, or set {key}",
            path.display()
        )),
        None => Err(format!("no home directory to find {file} in; set {key}")),
    }
}

fn locate(file: &str, key: &str) -> Result<PathBuf, String> {
    resolve(
        env::var_os(key).map(PathBuf::from),
        data_dir(),
        file,
        key,
        &|path| path.exists(),
    )
}

/// The size the model was trained at. It does not generalise past it: a 384px
/// input puts a ring of +2 arcsinh round every tile.
pub const WINDOW: u32 = 256;

/// One session for the overlay and the readout, loaded by whichever asks first.
/// Call it from a worker thread: loading takes seconds.
pub fn shared() -> Result<&'static Mutex<Model>, String> {
    static SHARED: OnceLock<Result<Mutex<Model>, String>> = OnceLock::new();
    SHARED
        .get_or_init(|| Model::load().map(Mutex::new))
        .as_ref()
        .map_err(Clone::clone)
}

pub struct Model {
    session: Session,
    input_name: String,
    marker_output: String,
    pub path: PathBuf,
    pub channel_names: Vec<String>,
    pub raw_intensity: bool,
}

impl Model {
    pub fn load() -> Result<Self, String> {
        // A runtime we cannot find is not fatal: ort falls back to its own
        // search, which finds a system-wide install.
        if let Ok(dylib) = locate("onnxruntime.dll", "CELLARIS_ORT_DYLIB") {
            if let Ok(builder) = ort::init_from(dylib.to_string_lossy().as_ref()) {
                let _ = builder.commit();
            }
        }
        let path = locate("markers.onnx", "CELLARIS_MODEL")?;

        let session = Session::builder()
            .map_err(|error| error.to_string())?
            .commit_from_file(&path)
            .map_err(|error| error.to_string())?;

        let input_name = session
            .inputs()
            .first()
            .map(|input| input.name().to_string())
            .ok_or_else(|| "model has no inputs".to_string())?;

        let marker_output = session
            .outputs()
            .iter()
            .find(|output| output.name() == "markers")
            .map(|output| output.name().to_string())
            .ok_or_else(|| "no output named `markers`; not a cellaris export".to_string())?;

        let metadata = session.metadata().map_err(|error| error.to_string())?;
        let channel_names: Vec<String> = metadata
            .custom("channel_names")
            .ok_or_else(|| {
                "model carries no channel_names; re-export it with export_marker_onnx".to_string()
            })?
            .split(',')
            .map(str::to_string)
            .collect();
        let raw_intensity = metadata.custom("raw_intensity").as_deref() != Some("false");
        drop(metadata);

        Ok(Self {
            session,
            input_name,
            marker_output,
            path,
            channel_names,
            raw_intensity,
        })
    }

    /// `WINDOW`x`WINDOW` RGBA8 to `(channels, WINDOW, WINDOW)`, input in `[-1, 1]`.
    pub fn predict(&mut self, rgba: &[u8]) -> Result<Vec<f32>, String> {
        let side = WINDOW as usize;
        let pixels = side * side;
        if rgba.len() < pixels * 4 {
            return Err(format!(
                "expected {} bytes of RGBA, got {}",
                pixels * 4,
                rgba.len()
            ));
        }

        let mut planar = vec![0f32; pixels * 3];
        for index in 0..pixels {
            for channel in 0..3 {
                planar[channel * pixels + index] =
                    f32::from(rgba[index * 4 + channel]) / 255.0 * 2.0 - 1.0;
            }
        }
        let input = Array4::from_shape_vec((1, 3, side, side), planar)
            .map_err(|error| error.to_string())?;

        let outputs = self
            .session
            .run(ort::inputs![
                self.input_name.as_str() =>
                    TensorRef::from_array_view(input.view())
                        .map_err(|error| error.to_string())?
            ])
            .map_err(|error| error.to_string())?;

        let array = outputs[self.marker_output.as_str()]
            .try_extract_array::<f32>()
            .map_err(|error| error.to_string())?;
        Ok(array.iter().copied().collect())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn nothing_exists(_: &Path) -> bool {
        false
    }

    #[test]
    fn the_environment_wins_when_it_points_somewhere_real() {
        let got = resolve(
            Some(PathBuf::from("/models/other.onnx")),
            Some(PathBuf::from("/home/.eov/plugins/cellaris")),
            "markers.onnx",
            "CELLARIS_MODEL",
            &|_| true,
        );
        assert_eq!(got.unwrap(), PathBuf::from("/models/other.onnx"));
    }

    #[test]
    fn a_missing_env_path_is_named_rather_than_ignored() {
        let error = resolve(
            Some(PathBuf::from("/models/gone.onnx")),
            Some(PathBuf::from("/home/.eov/plugins/cellaris")),
            "markers.onnx",
            "CELLARIS_MODEL",
            &nothing_exists,
        )
        .unwrap_err();
        assert!(error.contains("gone.onnx"), "{error}");
        assert!(error.contains("CELLARIS_MODEL"), "{error}");
    }

    #[test]
    fn the_plugin_directory_is_the_default() {
        let dir = PathBuf::from("/home/.eov/plugins/cellaris");
        let got = resolve(
            None,
            Some(dir.clone()),
            "markers.onnx",
            "CELLARIS_MODEL",
            &|_| true,
        );
        assert_eq!(got.unwrap(), dir.join("markers.onnx"));
    }

    #[test]
    fn with_nothing_anywhere_the_message_says_where_to_put_it() {
        let error = resolve(
            None,
            Some(PathBuf::from("/home/.eov/plugins/cellaris")),
            "markers.onnx",
            "CELLARIS_MODEL",
            &nothing_exists,
        )
        .unwrap_err();
        assert!(error.contains("/home/.eov/plugins/cellaris"), "{error}");
        assert!(error.contains("CELLARIS_MODEL"), "{error}");
    }
}
