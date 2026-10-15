//! Loads a built Cellaris library the way eov does — `lib_header_from_path`
//! and `init_root_module`, which check the layout of every type in the
//! interface — creates the plugin through its exported root module against
//! a mock host, and drives it through a few events.
//!
//!     cargo run --manifest-path tools/check/Cargo.toml --bin load -- \
//!         target/release/libcellaris.so [--package-root DIR]
//!
//! Without a model, turning the overlay on must end in a reported error, not
//! a crash. By default the runtime search is pointed at an empty directory,
//! so the error is about ONNX Runtime. With `--package-root` (an extracted
//! .eop) the plugin searches as it would in eov, and a package that bundles
//! ONNX Runtime gets as far as the missing model.

use abi_stable::std_types::RStr;
use eov_plugin_api::abi::{ABI_VERSION, API_VERSION, PluginModuleRef};
use eov_plugin_api::prelude::*;
use eov_plugin_api::testing::{MockHost, test_init};
use std::path::PathBuf;
use std::process::ExitCode;
use std::time::{Duration, Instant};

fn main() -> ExitCode {
    match run() {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => {
            eprintln!("{error}");
            ExitCode::FAILURE
        }
    }
}

fn send(plugin: &eov_plugin_api::abi::PluginBox, event: &HostEvent) -> Result<(), String> {
    let json = serde_json::to_string(event).map_err(|e| e.to_string())?;
    plugin
        .handle_event(RStr::from_str(&json))
        .into_result()
        .map(|_| ())
        .map_err(|e| format!("{event:?} failed: {}", e.message))
}

fn run() -> Result<(), String> {
    let mut args = std::env::args_os().skip(1);
    let path: PathBuf = args
        .next()
        .ok_or("usage: load <path to libcellaris> [--package-root DIR]")?
        .into();
    let package_root: Option<PathBuf> = match args.next() {
        Some(flag) if flag == "--package-root" => Some(
            args.next()
                .ok_or("--package-root needs a directory")?
                .into(),
        ),
        Some(other) => return Err(format!("unexpected argument {other:?}")),
        None => None,
    };
    let header = abi_stable::library::lib_header_from_path(&path)
        .map_err(|e| format!("{}: {e}", path.display()))?;
    let module = header
        .init_root_module::<PluginModuleRef>()
        .map_err(|e| format!("incompatible library: {e}"))?;
    if module.abi_version() != ABI_VERSION {
        return Err(format!("ABI {} != {ABI_VERSION}", module.abi_version()));
    }
    let api = module.api_version().as_str();
    if api.split('.').next() != API_VERSION.split('.').next() {
        return Err(format!(
            "built for plugin API {api}, expected {API_VERSION}"
        ));
    }
    println!("{}: plugin API {api}, ABI {ABI_VERSION}", path.display());

    let host = MockHost::new();
    let file = FileInfo {
        file_id: 1,
        filename: "slide.svs".into(),
        width: 10_000,
        height: 10_000,
        mpp_x: Some(0.25),
        mpp_y: Some(0.25),
        ..Default::default()
    };
    host.state().snapshot.files = vec![file.clone()];
    let mut init = test_init("cellaris");
    match &package_root {
        Some(root) => init.plugin_root = root.to_string_lossy().into_owned(),
        None => {
            // Keep the ONNX Runtime search away from anything installed on
            // this machine: an empty, explicit directory.
            let empty = PathBuf::from(&init.data_dir).join("no-runtime");
            std::fs::create_dir_all(&empty).map_err(|e| e.to_string())?;
            init.settings
                .insert("runtime_path".into(), json!(empty.to_string_lossy()));
        }
    }
    let init_json = serde_json::to_string(&init).map_err(|e| e.to_string())?;
    let create = module.create();
    let plugin = create(host.to_ffi(), RStr::from_str(&init_json))
        .into_result()
        .map_err(|e| format!("create failed: {}", e.message))?;

    {
        let state = host.state();
        for button in ["cellaris_overlay", "cellaris_panel"] {
            if !state.toolbar_buttons.contains_key(button) {
                return Err(format!("no toolbar button {button}"));
            }
        }
        if !state.tools.contains_key("cellaris_region") {
            return Err("no region tool".into());
        }
    }
    send(
        &plugin,
        &HostEvent::Ready {
            restore_sidebar: false,
        },
    )?;
    send(&plugin, &HostEvent::FileOpened { file })?;
    send(
        &plugin,
        &HostEvent::ToolbarButtonClicked {
            button_id: "cellaris_panel".into(),
        },
    )?;
    send(
        &plugin,
        &HostEvent::ToolbarButtonClicked {
            button_id: "cellaris_overlay".into(),
        },
    )?;
    let deadline = Instant::now() + Duration::from_secs(20);
    let error = loop {
        if let Some(note) = host
            .state()
            .notifications
            .iter()
            .find(|n| n.level == NotifyLevel::Error)
            .cloned()
        {
            break note;
        }
        if Instant::now() > deadline {
            return Err("no error was reported for the missing runtime".into());
        }
        std::thread::sleep(Duration::from_millis(20));
    };
    println!(
        "reported: {}\n{}",
        error.message,
        error.detail.unwrap_or_default()
    );
    let layers = host.state().layers.len();
    if layers != 3 {
        return Err(format!("expected 3 layers for the slide, found {layers}"));
    }
    plugin.shutdown();
    println!("loaded, created, drove and shut down the plugin");
    Ok(())
}
