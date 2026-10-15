//! Compiles `ui/cellaris.slint` the way eov does and checks the plugin's
//! side of the contract: every property it pushes exists and converts, and
//! every callback it handles is declared.
//!
//!     cargo run --manifest-path tools/ui-check/Cargo.toml

use cellaris::present;
use i_slint_compiler::langtype::Type;
use serde_json::Value as Json;
use std::collections::HashMap;
use std::path::Path;
use std::process::ExitCode;

fn spin_on<T>(future: impl std::future::Future<Output = T>) -> T {
    use std::task::{Context, Poll, Waker};
    let mut cx = Context::from_waker(Waker::noop());
    let mut future = std::pin::pin!(future);
    loop {
        if let Poll::Ready(value) = future.as_mut().poll(&mut cx) {
            return value;
        }
    }
}

/// The conversion eov applies to each pushed property (host_ui/surfaces.rs).
fn convert(ty: &Type, json: &Json) -> Result<(), String> {
    match (ty, json) {
        (Type::Array(inner), Json::Array(rows)) => rows
            .iter()
            .try_for_each(|row| slint_interpreter::json::value_from_json(inner, row).map(|_| ())),
        _ => slint_interpreter::json::value_from_json(ty, json).map(|_| ()),
    }
}

fn main() -> ExitCode {
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    let path = root.join(cellaris::config::PANEL_UI);
    let mut compiler = slint_interpreter::Compiler::default();
    compiler.set_include_paths(vec![root.clone()]);
    let result = spin_on(compiler.build_from_path(&path));
    let mut failed = false;
    for diagnostic in result.diagnostics() {
        eprintln!("{diagnostic}");
        if diagnostic.level() == slint_interpreter::DiagnosticLevel::Error {
            failed = true;
        }
    }
    let Some(definition) = result.component(cellaris::config::PANEL_COMPONENT) else {
        eprintln!(
            "{} does not export {}",
            path.display(),
            cellaris::config::PANEL_COMPONENT
        );
        return ExitCode::FAILURE;
    };

    let mut properties: HashMap<String, Type> = HashMap::new();
    let mut callbacks = Vec::new();
    for (name, (ty, _)) in definition.properties_and_callbacks() {
        match &ty {
            Type::Callback(_) => callbacks.push(name),
            ty if ty.is_property_type() => {
                properties.insert(name, ty.clone());
            }
            _ => {}
        }
    }

    let states = [
        ("a loaded model", present::example_state()),
        (
            "nothing loaded",
            cellaris::engine::State::new(Default::default(), Default::default()),
        ),
    ];
    for (label, state) in &states {
        for (name, json) in present::sidebar_properties(state) {
            match properties.get(&name) {
                None => {
                    eprintln!("{label}: the panel has no property {name}");
                    failed = true;
                }
                Some(ty) => {
                    if let Err(error) = convert(ty, &json) {
                        eprintln!("{label}: {name} = {json} does not convert: {error}");
                        failed = true;
                    }
                }
            }
        }
    }
    for name in present::PANEL_CALLBACKS {
        if !callbacks.iter().any(|c| c == name) {
            eprintln!("the panel declares no callback {name}");
            failed = true;
        }
    }
    for name in &callbacks {
        if !present::PANEL_CALLBACKS.contains(&name.as_str()) {
            eprintln!("the plugin does not handle the panel's callback {name}");
            failed = true;
        }
    }
    if failed {
        ExitCode::FAILURE
    } else {
        println!(
            "{}: {} properties and {} callbacks check out",
            path.display(),
            properties.len(),
            callbacks.len()
        );
        ExitCode::SUCCESS
    }
}
