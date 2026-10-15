//! Finding and loading ONNX Runtime (CEL-4).
//!
//! The runtime is loaded at run time (`ort`'s `load-dynamic`), from the
//! first of:
//!
//! 1. `CELLARIS_ORT_DYLIB`, the runtime chosen in the Cellaris panel, or
//!    `ORT_DYLIB_PATH` — a library file or a directory holding one. These
//!    are explicit: if they are wrong, that is reported and nothing else is
//!    tried.
//! 2. The plugin package (`package.py --onnxruntime` bundles it there) and
//!    the plugin's data directory.
//! 3. The system: the dynamic loader's search path, plus Homebrew's and
//!    `/usr/local/lib` on macOS and Linux.
//!
//! Every attempt that fails is reported with its path and reason, and
//! nothing here can panic the worker: the version check happens before
//! `ort` is used, and loading runs inside `catch_unwind`.

use std::ffi::OsString;
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::path::{Path, PathBuf};

/// The oldest ONNX Runtime this build accepts (it is built for API 17).
pub const MIN_VERSION: &str = "1.17";

/// The runtime's unversioned file names on this platform.
pub fn library_names() -> &'static [&'static str] {
    if cfg!(target_os = "windows") {
        &["onnxruntime.dll"]
    } else if cfg!(target_os = "macos") {
        &["libonnxruntime.dylib"]
    } else {
        &["libonnxruntime.so", "libonnxruntime.so.1"]
    }
}

/// Whether `name` is an ONNX Runtime library on `os` ("windows", "macos",
/// anything else is treated as Linux), including versioned names such as
/// `libonnxruntime.so.1.20.1` and `libonnxruntime.1.20.1.dylib`, but not
/// execution-provider libraries.
pub fn is_runtime_name(name: &str, os: &str) -> bool {
    let lower = name.to_ascii_lowercase();
    if lower.contains("providers") {
        return false;
    }
    match os {
        "windows" => lower == "onnxruntime.dll",
        "macos" => {
            lower == "libonnxruntime.dylib"
                || (lower.starts_with("libonnxruntime.") && lower.ends_with(".dylib"))
        }
        _ => lower == "libonnxruntime.so" || lower.starts_with("libonnxruntime.so."),
    }
}

fn current_os() -> &'static str {
    if cfg!(target_os = "windows") {
        "windows"
    } else if cfg!(target_os = "macos") {
        "macos"
    } else {
        "linux"
    }
}

/// Unversioned names first, then the newest version.
fn preference(name: &str) -> (bool, Vec<u64>) {
    let numbers: Vec<u64> = name
        .split(|c: char| !c.is_ascii_digit())
        .filter(|part| !part.is_empty())
        .filter_map(|part| part.parse().ok())
        .collect();
    (numbers.is_empty(), numbers)
}

/// What the search needs from the file system (a seam for tests).
pub trait Probe {
    fn exists(&self, path: &Path) -> bool;
    fn is_dir(&self, path: &Path) -> bool;
    fn list(&self, dir: &Path) -> Vec<String>;
}

/// The real file system.
pub struct Disk;

impl Probe for Disk {
    fn exists(&self, path: &Path) -> bool {
        path.exists()
    }

    fn is_dir(&self, path: &Path) -> bool {
        path.is_dir()
    }

    fn list(&self, dir: &Path) -> Vec<String> {
        std::fs::read_dir(dir)
            .map(|entries| {
                entries
                    .filter_map(Result::ok)
                    .filter_map(|entry| entry.file_name().into_string().ok())
                    .collect()
            })
            .unwrap_or_default()
    }
}

/// Where a candidate came from.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Source {
    Env(&'static str),
    Setting,
    Bundled,
    DataDir,
    System,
}

impl Source {
    pub fn describe(&self) -> String {
        match self {
            Source::Env(name) => name.to_string(),
            Source::Setting => "the runtime chosen in the Cellaris panel".to_string(),
            Source::Bundled => "the plugin package".to_string(),
            Source::DataDir => "the plugin's data directory".to_string(),
            Source::System => "the system library path".to_string(),
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct Candidate {
    pub path: PathBuf,
    pub source: Source,
}

/// Inputs of the search.
#[derive(Clone, Debug, Default)]
pub struct Search {
    pub env_cellaris: Option<OsString>,
    pub env_ort: Option<OsString>,
    pub setting: Option<PathBuf>,
    pub plugin_root: Option<PathBuf>,
    pub data_dir: Option<PathBuf>,
}

impl Search {
    /// The search for this process.
    pub fn from_env(
        setting: Option<PathBuf>,
        plugin_root: Option<PathBuf>,
        data_dir: Option<PathBuf>,
    ) -> Self {
        let var = |name: &str| std::env::var_os(name).filter(|value| !value.is_empty());
        Self {
            env_cellaris: var("CELLARIS_ORT_DYLIB"),
            env_ort: var("ORT_DYLIB_PATH"),
            setting,
            plugin_root,
            data_dir,
        }
    }
}

/// The best runtime library in `dir` (or its `lib/`), if any.
fn find_in(dir: &Path, os: &str, probe: &dyn Probe) -> Option<PathBuf> {
    for dir in [dir.to_path_buf(), dir.join("lib")] {
        let mut names: Vec<String> = probe
            .list(&dir)
            .into_iter()
            .filter(|name| is_runtime_name(name, os))
            .collect();
        names.sort_by_key(|name| std::cmp::Reverse(preference(name)));
        if let Some(name) = names.first() {
            return Some(dir.join(name));
        }
    }
    None
}

/// The candidates to try, in order. An explicit setting that points nowhere
/// is an error rather than a reason to look elsewhere.
pub fn candidates(search: &Search, os: &str, probe: &dyn Probe) -> Result<Vec<Candidate>, String> {
    let names = match os {
        "windows" => "onnxruntime.dll",
        "macos" => "libonnxruntime.dylib",
        _ => "libonnxruntime.so",
    };
    let explicit = [
        (
            search.env_cellaris.clone().map(PathBuf::from),
            Source::Env("CELLARIS_ORT_DYLIB"),
        ),
        (search.setting.clone(), Source::Setting),
        (
            search.env_ort.clone().map(PathBuf::from),
            Source::Env("ORT_DYLIB_PATH"),
        ),
    ];
    for (path, source) in explicit {
        let Some(path) = path else { continue };
        if !probe.exists(&path) {
            return Err(format!(
                "{} points at {}, which does not exist",
                source.describe(),
                path.display()
            ));
        }
        if probe.is_dir(&path) {
            return match find_in(&path, os, probe) {
                Some(found) => Ok(vec![Candidate {
                    path: found,
                    source,
                }]),
                None => Err(format!(
                    "{} points at {}, which holds no {names}",
                    source.describe(),
                    path.display()
                )),
            };
        }
        return Ok(vec![Candidate { path, source }]);
    }

    let mut found = Vec::new();
    for (dir, source) in [
        (search.plugin_root.as_deref(), Source::Bundled),
        (search.data_dir.as_deref(), Source::DataDir),
    ] {
        if let Some(path) = dir.and_then(|dir| find_in(dir, os, probe)) {
            found.push(Candidate { path, source });
        }
    }
    let system: &[&str] = match os {
        "windows" => &["onnxruntime.dll"],
        "macos" => &[
            "/opt/homebrew/lib/libonnxruntime.dylib",
            "/usr/local/lib/libonnxruntime.dylib",
            "libonnxruntime.dylib",
        ],
        _ => &[
            "libonnxruntime.so",
            "libonnxruntime.so.1",
            "/usr/local/lib/libonnxruntime.so",
        ],
    };
    for name in system {
        let path = PathBuf::from(name);
        // Bare names are left to the dynamic loader's search; paths are
        // only worth a try when they are there. (Not `is_absolute`: on
        // Windows, a path without a drive letter is not absolute.)
        let bare = path.parent().is_none_or(|dir| dir.as_os_str().is_empty());
        if !bare && !probe.exists(&path) {
            continue;
        }
        found.push(Candidate {
            path,
            source: Source::System,
        });
    }
    Ok(found)
}

/// A runtime that loaded.
#[derive(Clone, Debug, PartialEq)]
pub struct Loaded {
    pub path: PathBuf,
    pub source: Source,
    /// e.g. "1.22.0", from the runtime's build information.
    pub version: Option<String>,
}

/// Why no runtime could be loaded.
#[derive(Clone, Debug, PartialEq)]
pub struct Failure {
    pub summary: String,
    pub detail: String,
}

fn install_hint() -> String {
    let platform = match current_os() {
        "windows" => "onnxruntime.dll from the onnxruntime-win-x64 release",
        "macos" => {
            "`brew install onnxruntime`, or libonnxruntime.dylib from the onnxruntime-osx release"
        }
        _ => "libonnxruntime.so from the onnxruntime-linux release, or your distribution's package",
    };
    format!(
        "Install ONNX Runtime {MIN_VERSION} or newer ({platform}), then choose it in the \
         Cellaris panel, set CELLARIS_ORT_DYLIB, or put it in the plugin's data directory."
    )
}

/// The version ONNX Runtime reports through `OrtApiBase::GetVersionString`.
fn reported_version(path: &Path) -> Option<String> {
    // ort resolves a relative name next to the executable first; so does
    // this, so that it asks the same library.
    let path = match std::env::current_exe() {
        Ok(exe) if path.is_relative() => exe
            .parent()
            .map(|dir| dir.join(path))
            .filter(|p| p.exists())
            .unwrap_or_else(|| path.to_path_buf()),
        _ => path.to_path_buf(),
    };
    // SAFETY: ort has just loaded this library, so opening it again only
    // takes another reference to it. OrtGetApiBase and GetVersionString
    // take no arguments, and the string they return is static.
    unsafe {
        let library = libloading::Library::new(&path).ok()?;
        let get_base: libloading::Symbol<unsafe extern "C" fn() -> *const ort::sys::OrtApiBase> =
            library.get(b"OrtGetApiBase").ok()?;
        let base = get_base();
        if base.is_null() {
            return None;
        }
        let text = ((*base).GetVersionString)();
        if text.is_null() {
            return None;
        }
        Some(
            std::ffi::CStr::from_ptr(text)
                .to_string_lossy()
                .into_owned(),
        )
    }
}

/// The version in a versioned library name ("libonnxruntime.so.1.22.0").
pub fn version_from_name(path: &Path) -> Option<String> {
    let name = path.file_name()?.to_str()?;
    let version = name.strip_prefix("libonnxruntime.so.").or_else(|| {
        name.strip_prefix("libonnxruntime.")
            .and_then(|rest| rest.strip_suffix(".dylib"))
    })?;
    (version.contains('.') && version.chars().all(|c| c.is_ascii_digit() || c == '.'))
        .then(|| version.to_string())
}

/// The version in `ort::info()`'s build string ("git-branch=rel-1.22.0").
pub fn version_from_build_info(info: &str) -> Option<String> {
    let start = info.find("rel-")? + 4;
    let version: String = info[start..]
        .chars()
        .take_while(|c| c.is_ascii_digit() || *c == '.')
        .collect();
    (!version.is_empty()).then_some(version)
}

/// Loads the first candidate that works. Loading is once per process: after
/// a success, later calls return the same runtime.
pub fn load(search: &Search) -> Result<Loaded, Failure> {
    let candidates = candidates(search, current_os(), &Disk).map_err(|reason| Failure {
        summary: "ONNX Runtime was not found".to_string(),
        detail: format!("{reason}.\n{}", install_hint()),
    })?;
    if candidates.is_empty() {
        return Err(Failure {
            summary: "ONNX Runtime was not found".to_string(),
            detail: install_hint(),
        });
    }
    let mut attempts = Vec::new();
    for candidate in candidates {
        let path = candidate.path.clone();
        let outcome = catch_unwind(AssertUnwindSafe(|| {
            ort::init_from(&path).map(|builder| {
                // Keep ONNX Runtime's telemetry off for slides.
                builder.with_name("cellaris").with_telemetry(false).commit();
            })
        }));
        match outcome {
            Ok(Ok(())) => {
                let version = catch_unwind(AssertUnwindSafe(|| reported_version(&candidate.path)))
                    .ok()
                    .flatten()
                    .or_else(|| version_from_name(&candidate.path))
                    .or_else(|| {
                        catch_unwind(ort::info)
                            .ok()
                            .and_then(version_from_build_info)
                    });
                return Ok(Loaded {
                    path: candidate.path,
                    source: candidate.source,
                    version,
                });
            }
            Ok(Err(error)) => attempts.push(format!(
                "{} ({}): {error}",
                candidate.path.display(),
                candidate.source.describe()
            )),
            Err(_) => attempts.push(format!(
                "{} ({}): loading it crashed",
                candidate.path.display(),
                candidate.source.describe()
            )),
        }
        if !matches!(
            candidate.source,
            Source::Bundled | Source::DataDir | Source::System
        ) {
            break;
        }
    }
    let incompatible = attempts.iter().any(|a| a.contains("not compatible"));
    Err(Failure {
        summary: if incompatible {
            format!("The ONNX Runtime found is older than {MIN_VERSION}")
        } else {
            "ONNX Runtime could not be loaded".to_string()
        },
        detail: format!("Tried:\n{}\n\n{}", attempts.join("\n"), install_hint()),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::BTreeMap;

    /// Paths that exist, and directory listings.
    #[derive(Default)]
    struct Fake {
        files: Vec<PathBuf>,
        dirs: BTreeMap<PathBuf, Vec<String>>,
    }

    impl Probe for Fake {
        fn exists(&self, path: &Path) -> bool {
            self.files.iter().any(|f| f == path) || self.dirs.contains_key(path)
        }

        fn is_dir(&self, path: &Path) -> bool {
            self.dirs.contains_key(path)
        }

        fn list(&self, dir: &Path) -> Vec<String> {
            self.dirs.get(dir).cloned().unwrap_or_default()
        }
    }

    fn search() -> Search {
        Search {
            plugin_root: Some("/pkg".into()),
            data_dir: Some("/data".into()),
            ..Default::default()
        }
    }

    #[test]
    fn library_names_are_per_platform() {
        assert!(is_runtime_name("onnxruntime.dll", "windows"));
        assert!(
            !is_runtime_name("onnxruntime.dll", "linux"),
            "the old lookup"
        );
        assert!(is_runtime_name("libonnxruntime.so.1.20.1", "linux"));
        assert!(is_runtime_name("libonnxruntime.1.20.1.dylib", "macos"));
        assert!(!is_runtime_name(
            "libonnxruntime_providers_shared.so",
            "linux"
        ));
        assert!(!is_runtime_name("libonnxruntime.so", "macos"));
        assert!(!library_names().is_empty());
    }

    #[test]
    fn a_wrong_explicit_path_is_reported_not_skipped() {
        let fake = Fake::default();
        let error = candidates(
            &Search {
                env_cellaris: Some("/nope/libonnxruntime.so".into()),
                ..search()
            },
            "linux",
            &fake,
        )
        .unwrap_err();
        assert!(
            error.contains("CELLARIS_ORT_DYLIB") && error.contains("/nope"),
            "{error}"
        );

        let mut fake = Fake::default();
        fake.dirs.insert("/empty".into(), vec!["README".into()]);
        let error = candidates(
            &Search {
                setting: Some("/empty".into()),
                ..search()
            },
            "linux",
            &fake,
        )
        .unwrap_err();
        assert!(error.contains("holds no libonnxruntime.so"), "{error}");
    }

    #[test]
    fn explicit_paths_win_and_directories_are_searched() {
        let mut fake = Fake::default();
        fake.dirs.insert("/ort".into(), vec!["include".into()]);
        fake.dirs.insert(
            "/ort/lib".into(),
            vec![
                "libonnxruntime.so.1.20.1".into(),
                "libonnxruntime.so.1.22.0".into(),
                "libonnxruntime_providers_shared.so".into(),
            ],
        );
        fake.dirs
            .insert("/pkg".into(), vec!["libonnxruntime.so".into()]);
        let found = candidates(
            &Search {
                env_ort: Some("/ort".into()),
                ..search()
            },
            "linux",
            &fake,
        )
        .unwrap();
        assert_eq!(
            found,
            vec![Candidate {
                path: "/ort/lib/libonnxruntime.so.1.22.0".into(),
                source: Source::Env("ORT_DYLIB_PATH"),
            }]
        );
    }

    #[test]
    fn the_bundled_runtime_comes_before_the_system() {
        let mut fake = Fake::default();
        fake.dirs.insert(
            "/pkg".into(),
            vec!["libcellaris.so".into(), "libonnxruntime.so.1.22.0".into()],
        );
        fake.dirs
            .insert("/data".into(), vec!["libonnxruntime.so".into()]);
        let found = candidates(&search(), "linux", &fake).unwrap();
        assert_eq!(
            found[0].path,
            PathBuf::from("/pkg/libonnxruntime.so.1.22.0")
        );
        assert_eq!(found[0].source, Source::Bundled);
        assert_eq!(found[1].path, PathBuf::from("/data/libonnxruntime.so"));
        assert!(found[2..].iter().all(|c| c.source == Source::System));
        assert_eq!(found[2].path, PathBuf::from("libonnxruntime.so"));
        // System paths (not bare names) are only tried when they exist.
        assert!(!found.iter().any(|c| c.path.starts_with("/usr/local")));

        let mut fake = Fake::default();
        fake.files
            .push("/opt/homebrew/lib/libonnxruntime.dylib".into());
        let found = candidates(&search(), "macos", &fake).unwrap();
        assert_eq!(
            found[0].path,
            PathBuf::from("/opt/homebrew/lib/libonnxruntime.dylib")
        );
        let found = candidates(&search(), "windows", &Fake::default()).unwrap();
        assert_eq!(
            found,
            vec![Candidate {
                path: "onnxruntime.dll".into(),
                source: Source::System
            }]
        );
    }

    #[test]
    fn versioned_names_give_their_version() {
        assert_eq!(
            version_from_name(Path::new("/x/libonnxruntime.so.1.29.0")).as_deref(),
            Some("1.29.0")
        );
        assert_eq!(
            version_from_name(Path::new("libonnxruntime.1.20.1.dylib")).as_deref(),
            Some("1.20.1")
        );
        assert_eq!(version_from_name(Path::new("libonnxruntime.so")), None);
        assert_eq!(version_from_name(Path::new("libonnxruntime.so.1")), None);
    }

    #[test]
    fn the_runtime_version_is_read_from_its_build_string() {
        let info =
            "ORT Build Info: git-branch=rel-1.22.0, git-commit-id=0b7e3a2, build type=Release";
        assert_eq!(version_from_build_info(info).as_deref(), Some("1.22.0"));
        assert_eq!(version_from_build_info("custom build"), None);
    }

    #[test]
    fn a_missing_runtime_is_an_error_not_a_panic() {
        let failure = load(&Search {
            env_cellaris: Some("/definitely/not/here/libonnxruntime.so".into()),
            ..Default::default()
        })
        .unwrap_err();
        assert!(
            failure.detail.contains("/definitely/not/here"),
            "{failure:?}"
        );
        assert!(failure.detail.contains(MIN_VERSION));
    }
}
