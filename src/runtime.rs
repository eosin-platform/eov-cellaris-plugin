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

use crate::text::{Text, isolate, text};
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
    /// A batch run's `--param runtime`.
    Param,
    Env(&'static str),
    Setting,
    Bundled,
    DataDir,
    System,
}

impl Source {
    /// Where the candidate came from, for the list of libraries tried.
    pub fn describe(&self) -> Text {
        match self {
            Source::Param => Text::plain("--param runtime"),
            Source::Env(name) => Text::plain(*name),
            // Translators: where an ONNX Runtime library was found, shown in
            // parentheses after it in the list of libraries tried.
            Source::Setting => text!(tr!("the runtime chosen in the Cellaris panel")),
            Source::Bundled => text!(tr!("the plugin package")),
            Source::DataDir => text!(tr!("the plugin's data directory")),
            Source::System => text!(tr!("the system library path")),
        }
    }

    /// `path`, given by this source, does not exist.
    fn points_nowhere(&self, path: &Path) -> Text {
        let path = path.display().to_string();
        match self {
            Source::Setting => text!(tr!(
                // Translators: {} is the path the user chose for the ONNX
                // Runtime library.
                "the runtime chosen in the Cellaris panel points at {}, which does not exist",
                isolate(&path)
            )),
            _ => {
                let source = self.describe();
                text!(tr!(
                    // Translators: {source} is an environment variable or a
                    // command-line option (not translated), {path} a path.
                    "{source} points at {path}, which does not exist",
                    source = isolate(&source),
                    path = isolate(&path)
                ))
            }
        }
    }

    /// `path`, a folder given by this source, holds no `library`.
    fn holds_none(&self, path: &Path, library: &'static str) -> Text {
        let path = path.display().to_string();
        match self {
            Source::Setting => text!(tr!(
                // Translators: {path} is the folder the user chose for ONNX
                // Runtime, {library} the file name of its library.
                "the runtime chosen in the Cellaris panel points at {path}, which holds no {library}",
                path = isolate(&path),
                library
            )),
            _ => {
                let source = self.describe();
                text!(tr!(
                    // Translators: {source} is an environment variable or a
                    // command-line option (not translated), {path} a folder,
                    // {library} the file name of ONNX Runtime's library.
                    "{source} points at {path}, which holds no {library}",
                    source = isolate(&source),
                    path = isolate(&path),
                    library
                ))
            }
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
    /// A batch run's own choice: it wins over the environment.
    pub param: Option<PathBuf>,
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
            param: None,
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
pub fn candidates(search: &Search, os: &str, probe: &dyn Probe) -> Result<Vec<Candidate>, Text> {
    let names = match os {
        "windows" => "onnxruntime.dll",
        "macos" => "libonnxruntime.dylib",
        _ => "libonnxruntime.so",
    };
    let explicit = [
        (search.param.clone(), Source::Param),
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
            return Err(source.points_nowhere(&path));
        }
        if probe.is_dir(&path) {
            return match find_in(&path, os, probe) {
                Some(found) => Ok(vec![Candidate {
                    path: found,
                    source,
                }]),
                None => Err(source.holds_none(&path, names)),
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
    pub summary: Text,
    pub detail: Text,
}

fn install_hint() -> Text {
    let platform = match current_os() {
        // Translators: where to get ONNX Runtime on Windows; file and release
        // names are not translated.
        "windows" => text!(tr!("onnxruntime.dll from the onnxruntime-win-x64 release")),
        // Translators: where to get ONNX Runtime on macOS; `brew install
        // onnxruntime` is a command, and file and release names are not
        // translated.
        "macos" => text!(tr!(
            "`brew install onnxruntime`, or libonnxruntime.dylib from the onnxruntime-osx release"
        )),
        // Translators: where to get ONNX Runtime on Linux; file and release
        // names are not translated.
        _ => text!(tr!(
            "libonnxruntime.so from the onnxruntime-linux release, or your distribution's package"
        )),
    };
    text!(tr!(
        // Translators: {version} is a version number, {platform} says where to
        // get ONNX Runtime for this computer; CELLARIS_ORT_DYLIB is an
        // environment variable (not translated).
        "Install ONNX Runtime {version} or newer ({platform}), then choose it in the Cellaris panel, set CELLARIS_ORT_DYLIB, or put it in the plugin's data directory.",
        version = MIN_VERSION,
        platform
    ))
}

/// Why a candidate was passed over.
struct Refusal {
    reason: Text,
    /// It is ONNX Runtime, older than [`MIN_VERSION`].
    too_old: bool,
}

impl From<Text> for Refusal {
    fn from(reason: Text) -> Self {
        Refusal {
            reason,
            too_old: false,
        }
    }
}

fn not_found() -> Text {
    text!(tr!("ONNX Runtime was not found"))
}

fn could_not_load() -> Text {
    text!(tr!("ONNX Runtime could not be loaded"))
}

/// One line of the libraries tried: the library, where it came from, and
/// what went wrong. The source is isolated as the path is: an environment
/// variable or `--param runtime` keeps its order in right-to-left text and
/// does not make the line run from left to right.
fn attempt(path: &Path, source: &Source, what: Text) -> Text {
    let (path, source) = (path.display().to_string(), source.describe());
    text!(tr!(
        // Translators: {path} is an ONNX Runtime library, {source} where it was
        // found, {what} what went wrong with it.
        "{path} ({source}): {what}",
        path = isolate(&path),
        source = isolate(&source),
        what
    ))
}

/// Where `ort` looks for `path`: a relative name next to the executable
/// first, as it does.
fn resolve_like_ort(path: &Path) -> PathBuf {
    match std::env::current_exe() {
        Ok(exe) if path.is_relative() => exe
            .parent()
            .map(|dir| dir.join(path))
            .filter(|p| p.exists())
            .unwrap_or_else(|| path.to_path_buf()),
        _ => path.to_path_buf(),
    }
}

/// A candidate that opened, still open, and its version.
struct Checked {
    path: PathBuf,
    version: Option<String>,
    _library: libloading::Library,
}

/// Opens a candidate without `ort` and checks that it is ONNX Runtime and
/// new enough. `ort` can try only once per process: its 2.0.0-rc.13 keeps a
/// failed load as if it had succeeded (and then uses an empty library), so
/// it is only given a library known to open.
fn check(path: &Path) -> Result<Checked, Refusal> {
    let path = resolve_like_ort(path);
    // SAFETY: opening a library runs its initialisers, as loading it would.
    // OrtGetApiBase and GetVersionString take no arguments, and the string
    // they return is static.
    unsafe {
        let library = libloading::Library::new(&path).map_err(|e| {
            let e = e.to_string();
            // Translators: "it" is a library file; {} is why the system could
            // not open it.
            Refusal::from(text!(tr!("it could not be opened: {}", e)))
        })?;
        let get_base: libloading::Symbol<unsafe extern "C" fn() -> *const ort::sys::OrtApiBase> =
            library.get(b"OrtGetApiBase").map_err(|_| {
                // Translators: "it" is a library file; OrtGetApiBase is a name
                // in ONNX Runtime (not translated).
                Refusal::from(text!(tr!(
                    "it is not ONNX Runtime (it has no OrtGetApiBase)"
                )))
            })?;
        let base = get_base();
        if base.is_null() {
            return Err(Refusal::from(text!(tr!(
                // Translators: "it" is a library file; OrtGetApiBase is a name in
                // ONNX Runtime (not translated).
                "it is not ONNX Runtime (OrtGetApiBase returned nothing)"
            ))));
        }
        let text = ((*base).GetVersionString)();
        let version = (!text.is_null()).then(|| {
            std::ffi::CStr::from_ptr(text)
                .to_string_lossy()
                .into_owned()
        });
        let minor = |v: &str| v.split('.').nth(1).and_then(|m| m.parse::<u32>().ok());
        let needed = minor(MIN_VERSION).unwrap_or(0);
        if let Some(version) = &version
            && minor(version).is_some_and(|m| m < needed)
        {
            let version = version.clone();
            return Err(Refusal {
                reason: text!(tr!(
                    // Translators: "it" is a library file; {version} and {needed}
                    // are version numbers.
                    "it is ONNX Runtime {version}, older than {needed}",
                    version,
                    needed = MIN_VERSION
                )),
                too_old: true,
            });
        }
        Ok(Checked {
            path,
            version,
            _library: library,
        })
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

/// How loading went in this process: ONNX Runtime loads once per process.
static LOADED: std::sync::OnceLock<Result<Loaded, Failure>> = std::sync::OnceLock::new();

/// Whether this process has already tried `ort` (and so cannot again).
pub fn tried() -> bool {
    LOADED.get().is_some()
}

/// Loads the first candidate that works. Loading is once per process: after
/// a success, later calls return the same runtime; after `ort` itself
/// failed, only a restart helps.
pub fn load(search: &Search) -> Result<Loaded, Failure> {
    if let Some(done) = LOADED.get() {
        return done.clone();
    }
    let candidates = candidates(search, current_os(), &Disk).map_err(|reason| {
        let hint = install_hint();
        Failure {
            summary: not_found(),
            // Translators: {reason} is why ONNX Runtime was not found, {hint}
            // how to install it.
            detail: text!(tr!("{reason}.\n{hint}", reason, hint)),
        }
    })?;
    if candidates.is_empty() {
        return Err(Failure {
            summary: not_found(),
            detail: install_hint(),
        });
    }
    let mut attempts: Vec<Text> = Vec::new();
    let mut too_old = false;
    for candidate in candidates {
        let explicit = !matches!(
            candidate.source,
            Source::Bundled | Source::DataDir | Source::System
        );
        let checked = match catch_unwind(AssertUnwindSafe(|| check(&candidate.path))) {
            Ok(Ok(checked)) => checked,
            Ok(Err(refusal)) => {
                too_old |= refusal.too_old;
                attempts.push(attempt(&candidate.path, &candidate.source, refusal.reason));
                if explicit {
                    break;
                }
                continue;
            }
            Err(_) => {
                attempts.push(attempt(
                    &candidate.path,
                    &candidate.source,
                    // Translators: "it" is an ONNX Runtime library.
                    text!(tr!("opening it crashed")),
                ));
                if explicit {
                    break;
                }
                continue;
            }
        };
        // The one try `ort` gets.
        let outcome = catch_unwind(AssertUnwindSafe(|| {
            ort::init_from(&checked.path).map(|builder| {
                // Keep ONNX Runtime's telemetry off for slides.
                builder.with_name("cellaris").with_telemetry(false).commit();
            })
        }));
        let result = match outcome {
            Ok(Ok(())) => Ok(Loaded {
                version: checked
                    .version
                    .clone()
                    .or_else(|| version_from_name(&checked.path)),
                path: candidate.path,
                source: candidate.source,
            }),
            Ok(Err(error)) => {
                let error = error.to_string();
                Err(Failure {
                    summary: could_not_load(),
                    detail: attempt(
                        &checked.path,
                        &candidate.source,
                        // Translators: {} is why ONNX Runtime could not be
                        // loaded, in its own words.
                        text!(tr!("{}. Restart eov to try again.", error)),
                    ),
                })
            }
            Err(_) => Err(Failure {
                summary: could_not_load(),
                detail: attempt(
                    &checked.path,
                    &candidate.source,
                    // Translators: "it" is an ONNX Runtime library.
                    text!(tr!("loading it crashed. Restart eov to try again.")),
                ),
            }),
        };
        return LOADED.get_or_init(|| result).clone();
    }
    let hint = install_hint();
    Err(Failure {
        summary: if too_old {
            // Translators: {} is a version number.
            text!(tr!("The ONNX Runtime found is older than {}", MIN_VERSION))
        } else {
            could_not_load()
        },
        detail: text!(tr!(
            // Translators: {attempts} lists the libraries tried, one per line,
            // {hint} says how to install ONNX Runtime.
            "Tried:\n{attempts}\n\n{hint}",
            attempts = attempts
                .iter()
                .map(Text::to_string)
                .collect::<Vec<_>>()
                .join("\n"),
            hint
        )),
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
        .unwrap_err()
        .to_string();
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
        .unwrap_err()
        .to_string();
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
        let detail = failure.detail.to_string();
        assert!(detail.contains("/definitely/not/here"), "{failure:?}");
        assert!(detail.contains(MIN_VERSION));
    }

    /// In a right-to-left language the line of a library tried isolates
    /// where the library came from, as it isolates the path: otherwise an
    /// environment variable's name makes the line run from left to right
    /// and `--param runtime` shows as `param runtime--`. The English text,
    /// for logs and batch output, has no marks.
    ///
    /// The language is global to the library, so the right-to-left half
    /// runs in a process of its own: this test binary running only this
    /// test, in Hebrew without a catalog (English texts, with the values
    /// they insert isolated as in every right-to-left language).
    #[test]
    fn a_tried_library_keeps_its_source_in_order_in_right_to_left_text() {
        const CHILD: &str = "CELLARIS_TEST_RIGHT_TO_LEFT";
        let lines = || {
            [
                (Source::Env("ORT_DYLIB_PATH"), "ORT_DYLIB_PATH"),
                (Source::Param, "--param runtime"),
                (Source::Bundled, "the plugin package"),
            ]
            .map(|(source, shown)| {
                let line = attempt(
                    Path::new("/opt/ort/libonnxruntime.so"),
                    &source,
                    Text::plain("it could not be opened"),
                );
                let english =
                    format!("/opt/ort/libonnxruntime.so ({shown}): it could not be opened");
                (line, shown, english)
            })
        };
        if std::env::var_os(CHILD).is_none() {
            for (line, _, english) in lines() {
                assert_eq!(line.to_string(), english);
                assert_eq!(line.english(), english);
            }
            let module = module_path!().split_once("::").unwrap().1;
            let name = format!(
                "{module}::a_tried_library_keeps_its_source_in_order_in_right_to_left_text"
            );
            let output = std::process::Command::new(std::env::current_exe().unwrap())
                .args([name.as_str(), "--exact"])
                .env(CHILD, "1")
                .output()
                .unwrap();
            let report = format!(
                "{}{}",
                String::from_utf8_lossy(&output.stdout),
                String::from_utf8_lossy(&output.stderr)
            );
            assert!(output.status.success(), "{report}");
            assert!(report.contains("1 passed"), "{report}");
            return;
        }
        let hebrew = eov_plugin_api::i18n::by_tag("he").unwrap();
        eov_plugin_api::i18n::set_current(hebrew);
        for (line, shown, english) in lines() {
            assert_eq!(
                line.to_string(),
                format!(
                    "\u{2068}/opt/ort/libonnxruntime.so\u{2069} (\u{2068}{shown}\u{2069}): \
                     it could not be opened"
                )
            );
            assert_eq!(line.english(), english);
        }
    }
}
