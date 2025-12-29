//! SQL syntax highlighting for the zarr-cli REPL
//!
//! Uses sqlparser for tokenization and nu-ansi-term for colors.

use nu_ansi_term::{Color, Style};
use rustyline::highlight::{CmdKind, Highlighter};
use rustyline::validate::{ValidationContext, ValidationResult, Validator};
use rustyline::{Completer, Hinter};
use sqlparser::dialect::GenericDialect;
use sqlparser::keywords::Keyword;
use sqlparser::tokenizer::{Token, Tokenizer};
use std::borrow::Cow;

/// SQL Helper for rustyline with syntax highlighting
#[derive(Completer, Hinter)]
pub struct SqlHelper;

impl SqlHelper {
    pub fn new() -> Self {
        Self
    }
}

impl rustyline::Helper for SqlHelper {}

impl Validator for SqlHelper {
    fn validate(&self, _ctx: &mut ValidationContext) -> rustyline::Result<ValidationResult> {
        Ok(ValidationResult::Valid(None))
    }
}

impl Highlighter for SqlHelper {
    fn highlight<'l>(&self, line: &'l str, _pos: usize) -> Cow<'l, str> {
        if line.is_empty() {
            return Cow::Borrowed(line);
        }

        match highlight_sql(line) {
            Some(highlighted) => Cow::Owned(highlighted),
            None => Cow::Borrowed(line),
        }
    }

    fn highlight_prompt<'b, 's: 'b, 'p: 'b>(
        &'s self,
        prompt: &'p str,
        _default: bool,
    ) -> Cow<'b, str> {
        Cow::Owned(Style::new().bold().paint(prompt).to_string())
    }

    fn highlight_hint<'h>(&self, hint: &'h str) -> Cow<'h, str> {
        Cow::Owned(Color::DarkGray.paint(hint).to_string())
    }

    fn highlight_char(&self, _line: &str, _pos: usize, _kind: CmdKind) -> bool {
        true // Always re-highlight on changes
    }
}

/// Color scheme for SQL tokens
struct Colors;

impl Colors {
    fn keyword() -> Style {
        Style::new().bold().fg(Color::Green)
    }

    fn string() -> Style {
        Style::new().fg(Color::Yellow)
    }

    fn number() -> Style {
        Style::new().fg(Color::Yellow)
    }

    fn operator() -> Style {
        Style::new().fg(Color::Blue)
    }

    fn comment() -> Style {
        Style::new().fg(Color::DarkGray)
    }

    fn identifier() -> Style {
        Style::new().fg(Color::White)
    }
}

/// Highlight SQL string using sqlparser tokenizer
fn highlight_sql(sql: &str) -> Option<String> {
    let dialect = GenericDialect {};
    let mut tokenizer = Tokenizer::new(&dialect, sql);

    let tokens = match tokenizer.tokenize() {
        Ok(tokens) => tokens,
        Err(_) => return None, // Fall back to plain text on error
    };

    let mut result = String::with_capacity(sql.len() * 2);

    for token in tokens {
        let colored = colorize_token(&token);
        result.push_str(&colored);
    }

    Some(result)
}

/// Apply color to a single token
fn colorize_token(token: &Token) -> String {
    match token {
        // Keywords (using sqlparser's keyword detection)
        Token::Word(word) if word.keyword != Keyword::NoKeyword => {
            Colors::keyword().paint(&word.value).to_string()
        }

        // Custom keywords not in sqlparser
        Token::Word(word) if word.value.eq_ignore_ascii_case("ZARR") => {
            Colors::keyword().paint(&word.value).to_string()
        }

        // Regular identifiers (includes function names - parsed later by sqlparser)
        Token::Word(word) => Colors::identifier().paint(&word.value).to_string(),

        // Strings
        Token::SingleQuotedString(s) => Colors::string().paint(format!("'{}'", s)).to_string(),
        Token::DoubleQuotedString(s) => Colors::string().paint(format!("\"{}\"", s)).to_string(),
        Token::NationalStringLiteral(s) => Colors::string().paint(format!("N'{}'", s)).to_string(),
        Token::HexStringLiteral(s) => Colors::string().paint(format!("X'{}'", s)).to_string(),

        // Numbers
        Token::Number(n, _) => Colors::number().paint(n.as_str()).to_string(),

        // Comments
        Token::Whitespace(sqlparser::tokenizer::Whitespace::SingleLineComment {
            comment,
            prefix,
        }) => Colors::comment()
            .paint(format!("{}{}", prefix, comment))
            .to_string(),
        Token::Whitespace(sqlparser::tokenizer::Whitespace::MultiLineComment(comment)) => {
            Colors::comment()
                .paint(format!("/*{}*/", comment))
                .to_string()
        }

        // Whitespace (preserve as-is)
        Token::Whitespace(ws) => ws.to_string(),

        // Operators
        token if is_operator(token) => Colors::operator().paint(token.to_string()).to_string(),

        // Everything else (punctuation, etc.) - no color
        _ => token.to_string(),
    }
}

/// Check if a token is an operator.
///
/// Unlike keywords (where sqlparser provides `word.keyword != Keyword::NoKeyword`),
/// sqlparser does not provide an `is_operator()` method or `Operator` enum.
/// Operators are individual Token variants, so we must match them manually.
fn is_operator(token: &Token) -> bool {
    matches!(
        token,
        // Comparison
        Token::Eq
            | Token::Neq
            | Token::Lt
            | Token::Gt
            | Token::LtEq
            | Token::GtEq
            // Arithmetic
            | Token::Plus
            | Token::Minus
            | Token::Mul
            | Token::Div
            | Token::Mod
            // Bitwise
            | Token::Ampersand
            | Token::Pipe
            | Token::Caret
            | Token::ShiftLeft
            | Token::ShiftRight
            // Other
            | Token::ExclamationMark
            | Token::DoubleEq
            | Token::Spaceship
            | Token::Tilde
            | Token::AtSign
            | Token::Arrow
            | Token::LongArrow
            | Token::HashArrow
            | Token::HashLongArrow
    )
}
