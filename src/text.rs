//! Texts that are kept until they are shown: why the model did not load, a
//! slide's notes, a readout that failed.
//!
//! A [`Text`] is written each time it is shown, so it is in the language of
//! the interface at that moment and changes when the user picks another
//! one. [`Text::english`] writes it in English, for logs and for data (the
//! copied readout, batch results), whatever the interface's language.
//!
//! Make one from a `tr!` text with [`text!`]: `text!(tr!("…", args))`. The
//! arguments are evaluated each time the text is written, so they must own
//! their values. Inside them, write numbers with [`decimal`], insert names
//! with [`isolate`], and insert other kept texts as [`Text`]s (not as
//! strings made with `tr!`): all three follow the language being written.

use eov_plugin_api::i18n;
use std::cell::Cell;
use std::fmt;
use std::sync::Arc;

thread_local! {
    /// Set while a text is written in English (see [`Text::english`]).
    static ENGLISH: Cell<bool> = const { Cell::new(false) };
}

/// Whether texts are being written in English (see [`Text::english`]).
pub fn writing_english() -> bool {
    ENGLISH.with(Cell::get)
}

/// The language texts are being written in.
fn language() -> &'static i18n::Language {
    if writing_english() {
        i18n::source()
    } else {
        i18n::current()
    }
}

/// `value` with `decimals` digits after the decimal separator of the
/// language being written (`1,50` in German).
pub fn decimal(value: f64, decimals: usize) -> String {
    i18n::format_decimal_for(language(), value, decimals)
}

/// `value` for insertion into a text: a file name, a path, a marker's name
/// keeps its own order in right-to-left languages (see [`i18n::isolate`]).
pub fn isolate(value: impl fmt::Display) -> String {
    i18n::isolate_for(language(), value)
}

/// A text the user reads, written when it is shown (see the module
/// documentation).
#[derive(Clone)]
pub struct Text(Arc<dyn Fn() -> String + Send + Sync>);

impl Text {
    /// A text written by `write` whenever it is shown; see [`text!`].
    pub fn new(write: impl Fn() -> String + Send + Sync + 'static) -> Self {
        Text(Arc::new(write))
    }

    /// A text shown as it is in every language: eov's or a library's words,
    /// a name, a number.
    pub fn plain(text: impl Into<String>) -> Self {
        let text: Arc<str> = Arc::from(text.into());
        Text::new(move || text.to_string())
    }

    /// The text in English, for logs and data.
    pub fn english(&self) -> String {
        /// Puts the language back, also if writing panics.
        struct Restore(bool);
        impl Drop for Restore {
            fn drop(&mut self) {
                ENGLISH.with(|english| english.set(self.0));
            }
        }
        let _restore = Restore(ENGLISH.with(|english| english.replace(true)));
        (self.0)()
    }
}

impl fmt::Display for Text {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&(self.0)())
    }
}

impl fmt::Debug for Text {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Debug::fmt(&self.english(), f)
    }
}

impl Default for Text {
    fn default() -> Self {
        Text::plain("")
    }
}

/// Texts are equal when they say the same in English.
impl PartialEq for Text {
    fn eq(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.0, &other.0) || self.english() == other.english()
    }
}

impl Eq for Text {}

/// Compares the text as it is shown.
impl PartialEq<&str> for Text {
    fn eq(&self, other: &&str) -> bool {
        (self.0)() == *other
    }
}

impl From<String> for Text {
    fn from(text: String) -> Self {
        Text::plain(text)
    }
}

impl From<&String> for Text {
    fn from(text: &String) -> Self {
        Text::plain(text.as_str())
    }
}

impl From<&str> for Text {
    fn from(text: &str) -> Self {
        Text::plain(text)
    }
}

/// A [`Text`] from a `tr!` text, written in the language of the interface
/// when it is shown and in English for [`Text::english`]:
/// `text!(tr!("{} is selected twice", isolate(&name)))`. The `tr!` is
/// spelled out so that the texts are found where they are written.
macro_rules! text {
    (tr!($($t:tt)*)) => {
        $crate::text::Text::new(move || {
            if $crate::text::writing_english() {
                $crate::text::english!($($t)*)
            } else {
                ::eov_plugin_api::tr!($($t)*)
            }
        })
    };
}
pub(crate) use text;

/// `tr!` without its translation: the English text, filled in the same
/// way (one form or the other of a plural text by its number).
macro_rules! english {
    ($context:tt => $one:tt | $many:tt % $n:expr $(, $($args:tt)*)?) => {{
        let n = $n;
        ::eov_plugin_api::i18n::format(
            if n as u64 == 1 { $one } else { $many },
            $crate::text::arguments!([] $($($args)*)? , n = n),
        )
    }};
    ($context:tt => $msgid:tt $(, $($args:tt)*)?) => {
        ::eov_plugin_api::i18n::format($msgid, $crate::text::arguments!([] $($($args)*)?))
    };
    ($one:tt | $many:tt % $n:expr $(, $($args:tt)*)?) => {{
        let n = $n;
        ::eov_plugin_api::i18n::format(
            if n as u64 == 1 { $one } else { $many },
            $crate::text::arguments!([] $($($args)*)? , n = n),
        )
    }};
    ($msgid:tt $(, $($args:tt)*)?) => {
        ::eov_plugin_api::i18n::format($msgid, $crate::text::arguments!([] $($($args)*)?))
    };
}
pub(crate) use english;

/// The arguments of [`english!`] as `(name, value)` pairs, as `tr!` takes
/// them: `value`, `name = value`, or `name` for `name = name`.
macro_rules! arguments {
    ([$($done:tt)*]) => { &[$($done)*] };
    ([$($done:tt)*] ,) => { &[$($done)*] };
    ([$($done:tt)*] , $($rest:tt)+) => { $crate::text::arguments!([$($done)*] $($rest)+) };
    ([$($done:tt)*] $name:ident = $value:expr , $($rest:tt)*) => {
        $crate::text::arguments!(
            [$($done)* (::core::stringify!($name), &$value as &dyn ::core::fmt::Display),] , $($rest)*
        )
    };
    ([$($done:tt)*] $name:ident = $value:expr) => {
        $crate::text::arguments!(
            [$($done)* (::core::stringify!($name), &$value as &dyn ::core::fmt::Display),]
        )
    };
    ([$($done:tt)*] $name:ident , $($rest:tt)*) => {
        $crate::text::arguments!(
            [$($done)* (::core::stringify!($name), &$name as &dyn ::core::fmt::Display),] , $($rest)*
        )
    };
    ([$($done:tt)*] $name:ident) => {
        $crate::text::arguments!(
            [$($done)* (::core::stringify!($name), &$name as &dyn ::core::fmt::Display),]
        )
    };
    ([$($done:tt)*] $value:expr , $($rest:tt)*) => {
        $crate::text::arguments!(
            [$($done)* ("", &$value as &dyn ::core::fmt::Display),] , $($rest)*
        )
    };
    ([$($done:tt)*] $value:expr) => {
        $crate::text::arguments!([$($done)* ("", &$value as &dyn ::core::fmt::Display),])
    };
}
pub(crate) use arguments;

// The texts are checked through `english!` and closures: a `tr!` here would
// put test texts in the catalogs.
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn english_is_filled_in_as_tr_fills_it_in() {
        let name = "Pan-CK";
        let count = 3usize;
        assert_eq!(english!("{} twice", name), "Pan-CK twice");
        assert_eq!(
            english!("{name}, {other}", name, other = "SMA"),
            "Pan-CK, SMA"
        );
        assert_eq!(english!("One cell" | "{n} cells" % count), "3 cells");
        assert_eq!(english!("One cell" | "{n} cells" % 1), "One cell");
        assert_eq!(
            english!("Ctx" => "{n} of {total}" | "{n} of all {total}" % 1, total = count),
            "1 of 3"
        );
        assert_eq!(english!("Ctx" => "{{x}} {}", decimal(1.5, 2)), "{x} 1.50");
    }

    #[test]
    fn kept_texts_inside_texts_follow_the_language_being_written() {
        let inner = Text::new(|| {
            if writing_english() {
                "in English".to_string()
            } else {
                "as shown".to_string()
            }
        });
        let outer = {
            let inner = inner.clone();
            Text::new(move || english!("outer: {}", inner))
        };
        assert_eq!(outer.english(), "outer: in English");
        assert_eq!(outer.to_string(), "outer: as shown");
        assert!(!writing_english(), "English mode leaked");
        assert_eq!(Text::plain("as is").english(), "as is");
        assert_eq!(Text::from("a"), Text::plain("a".to_string()));
        assert_eq!(Text::default(), "");
    }
}
