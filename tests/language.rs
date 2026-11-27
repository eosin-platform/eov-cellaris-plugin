//! When the user picks another language, what the plugin handed eov is
//! handed over again in it: the toolbar tooltips, the tool, the layers'
//! names, and the panel's texts and numbers, also texts the plugin kept.
//!
//! A test binary of its own: the language is global to the plugin library,
//! so it would change the language of tests running beside it.

use cellaris::Cellaris;
use cellaris::config::{OVERLAY_BUTTON, PANEL_BUTTON, REGION_TOOL};
use cellaris::model::{Backend, LoadError, LoadRequest, Predictor};
use eov_plugin_api::prelude::*;
use eov_plugin_api::testing::{MockHost, TestHarness, test_init};
use std::path::PathBuf;
use std::sync::{Mutex, MutexGuard};
use std::time::{Duration, Instant};

/// One test at a time: they change the language of the library.
fn serial() -> MutexGuard<'static, ()> {
    static SERIAL: Mutex<()> = Mutex::new(());
    SERIAL.lock().unwrap_or_else(|e| e.into_inner())
}

#[derive(Default)]
struct NoModel;

impl Backend for NoModel {
    fn load(&self, _: &LoadRequest) -> Result<Box<dyn Predictor>, LoadError> {
        Err(LoadError::new("No marker model", "none in this test"))
    }
}

/// A German catalog for a few of the plugin's texts.
const GERMAN: &str = r#"msgid ""
msgstr ""
"Language: de\n"
"Content-Type: text/plain; charset=UTF-8\n"
"Plural-Forms: nplurals=2; plural=(n != 1);\n"

msgid "Cellaris panel: legend, region readout and model"
msgstr "Cellaris-Bereich: Legende, Regionsmessung und Modell"

msgid "Cellaris region readout"
msgstr "Cellaris-Regionsmessung"

msgid "Cellaris — readout region"
msgstr "Cellaris — Messregion"

msgid "Cellaris region"
msgstr "Cellaris-Region"

msgid "Enter the model's SHA-256 (64 hex digits): the download is checked against it."
msgstr "SHA-256 des Modells eingeben (64 Hexadezimalziffern): der Download wird damit geprüft."

msgid "{reason} Another key can be set as {key} in Cellaris's settings.json (\"\" for none)."
msgstr "{reason} In der settings.json von Cellaris kann {key} eine andere Taste festlegen (\"\" für keine)."
"#;

fn wait_for(what: &str, mut done: impl FnMut() -> bool) {
    let deadline = Instant::now() + Duration::from_secs(20);
    while !done() {
        assert!(Instant::now() < deadline, "timed out waiting for {what}");
        std::thread::sleep(Duration::from_millis(20));
    }
}

/// The region's label on screen and in images (exports, copies of the
/// view).
fn region_labels(harness: &TestHarness<Cellaris<NoModel>>) -> (Option<String>, Option<String>) {
    harness
        .state()
        .objects("cellaris-region-1")
        .into_iter()
        .find(|object| object.object_id == "region")
        .map(|region| (region.label, region.export_label))
        .unwrap_or_default()
}

fn panel(harness: &TestHarness<Cellaris<NoModel>>, name: &str) -> String {
    let state = harness.state();
    state
        .surfaces
        .values()
        .find(|surface| surface.open)
        .and_then(|surface| surface.properties.get(name))
        .and_then(Value::as_str)
        .unwrap_or_default()
        .to_string()
}

/// The plugin's start, as eov unpacks its package: the manifest and a
/// catalog.
fn translated_init() -> PluginInit {
    let init = test_init("cellaris");
    let root = PathBuf::from(&init.plugin_root);
    std::fs::write(root.join("plugin.toml"), include_str!("../plugin.toml")).unwrap();
    let catalogs = root.join("translations/de/LC_MESSAGES");
    std::fs::create_dir_all(&catalogs).unwrap();
    std::fs::write(catalogs.join("cellaris.po"), GERMAN).unwrap();
    init
}

#[test]
fn a_language_change_hands_eov_the_plugins_texts_again() {
    let _serial = serial();
    let init = translated_init();

    let host = MockHost::new();
    host.state().snapshot.files = vec![FileInfo {
        file_id: 1,
        filename: "slide.svs".into(),
        width: 10_000,
        height: 10_000,
        mpp_x: Some(0.25),
        mpp_y: Some(0.25),
        ..Default::default()
    }];
    let harness = TestHarness::<Cellaris<NoModel>>::with_host(host, init).unwrap();
    harness
        .send(HostEvent::Ready {
            restore_sidebar: true,
        })
        .unwrap();
    // A text the plugin keeps until it is shown.
    harness
        .ui_callback(
            "panel",
            "download-model",
            vec![json!("https://example.org/markers.onnx"), json!("abc")],
        )
        .unwrap();
    wait_for("the English panel", || {
        panel(&harness, "slide-notes").starts_with("0.250 \u{d7} 0.250")
            && panel(&harness, "download-note").starts_with("Enter the model's SHA-256")
    });
    // A region to measure, labelled on the slide.
    harness
        .send(HostEvent::ShapeDrawn {
            tool_id: REGION_TOOL.into(),
            pane: 1,
            file_id: 1,
            shape: Shape::Polygon {
                exterior: vec![
                    [1_000.0, 1_000.0],
                    [2_000.0, 1_000.0],
                    [2_000.0, 2_000.0],
                    [1_000.0, 2_000.0],
                ],
                holes: vec![],
            },
            modifiers: Modifiers::default(),
        })
        .unwrap();
    wait_for("the region's label", || {
        region_labels(&harness)
            == (
                Some("Cellaris region".into()),
                Some("Cellaris region".into()),
            )
    });
    {
        let state = harness.state();
        assert_eq!(
            state.toolbar_buttons[PANEL_BUTTON].tooltip,
            "Cellaris panel: legend, region readout and model"
        );
        assert_eq!(state.tools[REGION_TOOL].label, "Cellaris region readout");
        assert_eq!(
            state.layers["cellaris-region-1"].spec.name,
            "Cellaris \u{2014} readout region"
        );
    }

    harness
        .send(HostEvent::LanguageChanged {
            language: "de".into(),
        })
        .unwrap();
    {
        let state = harness.state();
        assert_eq!(
            state.toolbar_buttons[PANEL_BUTTON].tooltip,
            "Cellaris-Bereich: Legende, Regionsmessung und Modell"
        );
        assert_eq!(state.tools[REGION_TOOL].label, "Cellaris-Regionsmessung");
        assert_eq!(
            state.layers["cellaris-region-1"].spec.name,
            "Cellaris \u{2014} Messregion"
        );
        // Untranslated texts stay in English.
        assert_eq!(
            state.layers["cellaris-marks-1"].spec.name,
            "Cellaris \u{2014} progress"
        );
    }
    // German numbers, and the kept text written again in German.
    wait_for("the German panel", || {
        panel(&harness, "slide-notes").starts_with("0,250 \u{d7} 0,250")
            && panel(&harness, "download-note").starts_with("SHA-256 des Modells eingeben")
    });
    // The region's label follows the language on screen; images (exports,
    // copies of the view) keep it in English, for their font and readers.
    wait_for("the German region label", || {
        region_labels(&harness)
            == (
                Some("Cellaris-Region".into()),
                Some("Cellaris region".into()),
            )
    });
}

/// eov's reason for refusing the overlay's key is shown in the panel as eov
/// says it in the language of the interface, and asked for again after a
/// change of language.
#[test]
fn a_refused_key_is_explained_in_the_language_of_the_interface() {
    let _serial = serial();
    let harness =
        TestHarness::<Cellaris<NoModel>>::with_host(MockHost::new(), translated_init()).unwrap();
    harness
        .send(HostEvent::ToolbarButtonClicked {
            button_id: PANEL_BUTTON.into(),
        })
        .unwrap();
    const IN_ENGLISH: &str = "Hotkey 'x' was ignored: it is the same key as other:tool.";
    const IN_GERMAN: &str = "Tastenkürzel „x“ wurde ignoriert: other:tool hat dieselbe Taste.";
    let refusal = |language: &str| HostEvent::RegistrationRefused {
        what: "hotkey".into(),
        id: OVERLAY_BUTTON.into(),
        reason: IN_ENGLISH.into(),
        shown_reason: Some(
            if language == "de" {
                IN_GERMAN
            } else {
                IN_ENGLISH
            }
            .into(),
        ),
    };
    harness
        .send(HostEvent::LanguageChanged {
            language: "de".into(),
        })
        .unwrap();
    harness.send(refusal("de")).unwrap();
    let german = format!(
        "{IN_GERMAN} In der settings.json von Cellaris kann overlay_key eine andere Taste festlegen (\"\" für keine)."
    );
    wait_for("the German note", || {
        panel(&harness, "status-detail").contains(&german)
    });
    assert!(
        !panel(&harness, "status-detail").contains(IN_ENGLISH),
        "{}",
        panel(&harness, "status-detail")
    );
    // Back in English: the button is offered again, and eov's new answer
    // replaces the German one.
    let added = harness.calls_to(methods::TOOLBAR_ADD).len();
    harness
        .send(HostEvent::LanguageChanged {
            language: "en".into(),
        })
        .unwrap();
    assert_eq!(harness.calls_to(methods::TOOLBAR_ADD).len(), added + 1);
    harness.send(refusal("en")).unwrap();
    wait_for("the English note", || {
        let detail = panel(&harness, "status-detail");
        detail.contains(&format!(
            "{IN_ENGLISH} Another key can be set as overlay_key in Cellaris's settings.json"
        )) && !detail.contains(IN_GERMAN)
    });
}
