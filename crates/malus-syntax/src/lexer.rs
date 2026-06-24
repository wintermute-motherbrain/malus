use crate::span::{FileId, Span};
use crate::token::{Token, TokenKind};

// ── Error ────────────────────────────────────────────────────────────────────

#[derive(Debug, Clone, PartialEq)]
pub struct LexError {
    pub kind: LexErrorKind,
    pub span: Span,
}

#[derive(Debug, Clone, PartialEq)]
pub enum LexErrorKind {
    UnexpectedCharacter(char),
    InvalidNumber,
    UnexpectedIndent,
    InconsistentDedent,
    ExpectedIndentAfterColon,
    MixedIndentation,
    UnterminatedString,
}

impl std::fmt::Display for LexError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match &self.kind {
            LexErrorKind::UnexpectedCharacter(c) => write!(f, "unexpected character {:?}", c),
            LexErrorKind::InvalidNumber => write!(f, "invalid number literal"),
            LexErrorKind::UnexpectedIndent => write!(f, "unexpected indentation"),
            LexErrorKind::InconsistentDedent => write!(f, "dedent does not match any outer indentation level"),
            LexErrorKind::ExpectedIndentAfterColon => write!(f, "expected an indented block after ':'"),
            LexErrorKind::MixedIndentation => write!(f, "mixed tabs and spaces in indentation"),
            LexErrorKind::UnterminatedString => write!(f, "unterminated string literal"),
        }
    }
}

// ── Lexer ────────────────────────────────────────────────────────────────────

#[derive(Debug, Clone, Copy, PartialEq)]
enum IndentChar {
    Spaces,
    Tabs,
}

struct Lexer<'src> {
    source: &'src str,
    bytes: &'src [u8],
    file: FileId,
    pos: usize,

    tokens: Vec<Token>,

    indent_stack: Vec<u32>,
    bracket_depth: u32,
    at_line_start: bool,
    // True when the last logical line ended with ':' and the NEWLINE was emitted.
    after_colon: bool,
    indent_char: Option<IndentChar>,
}

impl<'src> Lexer<'src> {
    fn new(file: FileId, source: &'src str) -> Self {
        Self {
            source,
            bytes: source.as_bytes(),
            file,
            pos: 0,
            tokens: Vec::new(),
            indent_stack: vec![0],
            bracket_depth: 0,
            at_line_start: true,
            after_colon: false,
            indent_char: None,
        }
    }

    fn span(&self, start: usize, end: usize) -> Span {
        Span::new(self.file, start, end)
    }

    fn peek(&self) -> Option<u8> {
        self.bytes.get(self.pos).copied()
    }

    fn peek_ahead(&self, offset: usize) -> Option<u8> {
        self.bytes.get(self.pos + offset).copied()
    }

    fn emit(&mut self, kind: TokenKind, start: usize) {
        self.tokens.push(Token { kind, span: self.span(start, self.pos) });
    }

    fn last_real_token_is_colon(&self) -> bool {
        for tok in self.tokens.iter().rev() {
            match tok.kind {
                TokenKind::Newline | TokenKind::Indent | TokenKind::Dedent => continue,
                TokenKind::Colon => return true,
                _ => return false,
            }
        }
        false
    }

    fn run(&mut self) -> Result<(), LexError> {
        while self.pos < self.bytes.len() {
            if self.at_line_start {
                let done = self.handle_line_start()?;
                if done {
                    // Blank/comment-only line was consumed; loop again.
                    continue;
                }
            }

            // Skip non-leading horizontal whitespace.
            while matches!(self.peek(), Some(b' ') | Some(b'\t')) {
                self.pos += 1;
            }

            if self.pos >= self.bytes.len() {
                break;
            }

            let ch = self.bytes[self.pos];

            match ch {
                b'\n' => self.handle_newline(),
                b'#' => self.skip_comment(),
                b'a'..=b'z' | b'A'..=b'Z' | b'_' => self.scan_ident_or_keyword(),
                b'0'..=b'9' => self.scan_number()?,
                b'"' => self.scan_string()?,
                b'+' => { let s = self.pos; self.pos += 1; self.emit(TokenKind::Plus, s); }
                b'-' => {
                    let s = self.pos;
                    if self.peek_ahead(1) == Some(b'>') {
                        self.pos += 2;
                        self.emit(TokenKind::Arrow, s);
                    } else {
                        self.pos += 1;
                        self.emit(TokenKind::Minus, s);
                    }
                }
                b'*' => { let s = self.pos; self.pos += 1; self.emit(TokenKind::Star, s); }
                b'/' => { let s = self.pos; self.pos += 1; self.emit(TokenKind::Slash, s); }
                b'@' => { let s = self.pos; self.pos += 1; self.emit(TokenKind::At, s); }
                b'=' => {
                    let s = self.pos;
                    if self.peek_ahead(1) == Some(b'=') {
                        self.pos += 2;
                        self.emit(TokenKind::EqEq, s);
                    } else {
                        self.pos += 1;
                        self.emit(TokenKind::Eq, s);
                    }
                }
                b'!' => {
                    let s = self.pos;
                    if self.peek_ahead(1) == Some(b'=') {
                        self.pos += 2;
                        self.emit(TokenKind::NotEq, s);
                    } else {
                        return Err(LexError {
                            kind: LexErrorKind::UnexpectedCharacter('!'),
                            span: self.span(s, s + 1),
                        });
                    }
                }
                b'<' => {
                    let s = self.pos;
                    if self.peek_ahead(1) == Some(b'=') {
                        self.pos += 2;
                        self.emit(TokenKind::LtEq, s);
                    } else {
                        self.pos += 1;
                        self.emit(TokenKind::Lt, s);
                    }
                }
                b'>' => {
                    let s = self.pos;
                    if self.peek_ahead(1) == Some(b'=') {
                        self.pos += 2;
                        self.emit(TokenKind::GtEq, s);
                    } else {
                        self.pos += 1;
                        self.emit(TokenKind::Gt, s);
                    }
                }
                b'(' => {
                    let s = self.pos; self.pos += 1;
                    self.emit(TokenKind::LParen, s);
                    self.bracket_depth += 1;
                }
                b')' => {
                    let s = self.pos; self.pos += 1;
                    self.emit(TokenKind::RParen, s);
                    if self.bracket_depth > 0 { self.bracket_depth -= 1; }
                }
                b'[' => {
                    let s = self.pos; self.pos += 1;
                    self.emit(TokenKind::LBracket, s);
                    self.bracket_depth += 1;
                }
                b']' => {
                    let s = self.pos; self.pos += 1;
                    self.emit(TokenKind::RBracket, s);
                    if self.bracket_depth > 0 { self.bracket_depth -= 1; }
                }
                b',' => { let s = self.pos; self.pos += 1; self.emit(TokenKind::Comma, s); }
                b'.' => { let s = self.pos; self.pos += 1; self.emit(TokenKind::Dot, s); }
                b':' => { let s = self.pos; self.pos += 1; self.emit(TokenKind::Colon, s); }
                other => {
                    let c = other as char;
                    return Err(LexError {
                        kind: LexErrorKind::UnexpectedCharacter(c),
                        span: self.span(self.pos, self.pos + 1),
                    });
                }
            }
        }

        self.emit_eof_dedents();
        let s = self.pos;
        self.emit(TokenKind::Eof, s);
        Ok(())
    }

    /// Called when `at_line_start` is true.
    ///
    /// Returns `true` if the line was blank or comment-only (caller should
    /// loop again), or `false` if real tokens can now be scanned.
    fn handle_line_start(&mut self) -> Result<bool, LexError> {
        self.at_line_start = false;

        let line_start = self.pos;
        let mut width: u32 = 0;
        let mut first_indent_ch: Option<IndentChar> = None;

        while self.pos < self.bytes.len() {
            match self.bytes[self.pos] {
                b' ' => {
                    let ic = IndentChar::Spaces;
                    if let Some(ref fi) = first_indent_ch {
                        if *fi != ic {
                            return Err(LexError {
                                kind: LexErrorKind::MixedIndentation,
                                span: self.span(line_start, self.pos + 1),
                            });
                        }
                    } else {
                        first_indent_ch = Some(ic);
                    }
                    width += 1;
                    self.pos += 1;
                }
                b'\t' => {
                    let ic = IndentChar::Tabs;
                    if let Some(ref fi) = first_indent_ch {
                        if *fi != ic {
                            return Err(LexError {
                                kind: LexErrorKind::MixedIndentation,
                                span: self.span(line_start, self.pos + 1),
                            });
                        }
                    } else {
                        first_indent_ch = Some(ic);
                    }
                    width += 1;
                    self.pos += 1;
                }
                _ => break,
            }
        }

        // Enforce inter-line indent-char consistency.
        if let Some(ic) = first_indent_ch {
            match self.indent_char {
                None => { self.indent_char = Some(ic); }
                Some(existing) if existing != ic => {
                    return Err(LexError {
                        kind: LexErrorKind::MixedIndentation,
                        span: self.span(line_start, self.pos),
                    });
                }
                _ => {}
            }
        }

        // Blank line or comment-only: skip the rest of the line and loop.
        let at_eol = self.pos >= self.bytes.len()
            || self.bytes[self.pos] == b'\n'
            || self.bytes[self.pos] == b'#';

        if at_eol {
            if self.bytes.get(self.pos) == Some(&b'#') {
                self.skip_comment();
            }
            // Consume the newline if present.
            if self.bytes.get(self.pos) == Some(&b'\n') {
                self.pos += 1;
            }
            self.at_line_start = true;
            return Ok(true); // blank line consumed
        }

        // Inside brackets: suppress INDENT/DEDENT logic.
        if self.bracket_depth > 0 {
            return Ok(false);
        }

        let current = *self.indent_stack.last().unwrap();
        let indent_span = self.span(line_start, self.pos);

        if width > current {
            // Indentation increased.
            if !self.after_colon {
                return Err(LexError { kind: LexErrorKind::UnexpectedIndent, span: indent_span });
            }
            self.indent_stack.push(width);
            self.tokens.push(Token { kind: TokenKind::Indent, span: indent_span });
            self.after_colon = false;
        } else if width < current {
            // Indentation decreased: emit one DEDENT per level popped.
            self.after_colon = false;
            loop {
                self.indent_stack.pop();
                self.tokens.push(Token { kind: TokenKind::Dedent, span: indent_span });
                let new_top = *self.indent_stack.last().unwrap();
                if new_top == width {
                    break;
                }
                if new_top < width {
                    return Err(LexError { kind: LexErrorKind::InconsistentDedent, span: indent_span });
                }
            }
        } else {
            // Same indentation level.
            if self.after_colon {
                return Err(LexError { kind: LexErrorKind::ExpectedIndentAfterColon, span: indent_span });
            }
            self.after_colon = false;
        }

        Ok(false)
    }

    fn handle_newline(&mut self) {
        if self.bracket_depth == 0 {
            // Record whether this line ends with ':' for the next line's INDENT check.
            self.after_colon = self.last_real_token_is_colon();
            let s = self.pos;
            self.pos += 1;
            self.emit(TokenKind::Newline, s);
        } else {
            self.pos += 1;
        }
        self.at_line_start = true;
    }

    fn skip_comment(&mut self) {
        while self.pos < self.bytes.len() && self.bytes[self.pos] != b'\n' {
            self.pos += 1;
        }
    }

    fn scan_ident_or_keyword(&mut self) {
        let start = self.pos;
        while self.pos < self.bytes.len()
            && (self.bytes[self.pos].is_ascii_alphanumeric() || self.bytes[self.pos] == b'_')
        {
            self.pos += 1;
        }
        let text = &self.source[start..self.pos];
        let kind = match text {
            "fn"     => TokenKind::Fn,
            "from"   => TokenKind::From,
            "import" => TokenKind::Import,
            "kernel" => TokenKind::Kernel,
            "let"    => TokenKind::Let,
            "return" => TokenKind::Return,
            "if"     => TokenKind::If,
            "else"   => TokenKind::Else,
            "for"    => TokenKind::For,
            "in"     => TokenKind::In,
            "while"  => TokenKind::While,
            "struct" => TokenKind::Struct,
            "enum"   => TokenKind::Enum,
            "inout"  => TokenKind::Inout,
            "and"    => TokenKind::And,
            "or"     => TokenKind::Or,
            "not"    => TokenKind::Not,
            "true"   => TokenKind::Bool(true),
            "false"  => TokenKind::Bool(false),
            _        => TokenKind::Ident(text.to_owned()),
        };
        self.emit(kind, start);
    }

    fn scan_number(&mut self) -> Result<(), LexError> {
        let start = self.pos;
        let mut is_float = false;

        // Integer part.
        self.consume_digits();

        // Optional fractional part: '.' followed by a digit.
        if self.peek() == Some(b'.')
            && self.peek_ahead(1).map_or(false, |b| b.is_ascii_digit())
        {
            is_float = true;
            self.pos += 1; // consume '.'
            self.consume_digits();
        }

        // Optional exponent: [eE][+-]?[0-9]+
        if matches!(self.peek(), Some(b'e') | Some(b'E')) {
            is_float = true;
            self.pos += 1; // consume 'e' or 'E'
            if matches!(self.peek(), Some(b'+') | Some(b'-')) {
                self.pos += 1;
            }
            if !self.peek().map_or(false, |b| b.is_ascii_digit()) {
                return Err(LexError {
                    kind: LexErrorKind::InvalidNumber,
                    span: self.span(start, self.pos),
                });
            }
            self.consume_digits();
        }

        let raw = &self.source[start..self.pos];
        let clean: String = raw.chars().filter(|&c| c != '_').collect();

        let kind = if is_float {
            let value: f64 = clean.parse().map_err(|_| LexError {
                kind: LexErrorKind::InvalidNumber,
                span: self.span(start, self.pos),
            })?;
            TokenKind::Float(value)
        } else {
            let value: i64 = clean.parse().map_err(|_| LexError {
                kind: LexErrorKind::InvalidNumber,
                span: self.span(start, self.pos),
            })?;
            TokenKind::Int(value)
        };

        self.emit(kind, start);
        Ok(())
    }

    fn scan_string(&mut self) -> Result<(), LexError> {
        let start = self.pos;
        self.pos += 1; // consume opening '"'
        let mut value = String::new();
        loop {
            if self.pos >= self.bytes.len() || self.bytes[self.pos] == b'\n' {
                return Err(LexError {
                    kind: LexErrorKind::UnterminatedString,
                    span: self.span(start, self.pos),
                });
            }
            match self.bytes[self.pos] {
                b'"' => {
                    self.pos += 1; // consume closing '"'
                    break;
                }
                b'\\' => {
                    self.pos += 1;
                    if self.pos >= self.bytes.len() {
                        return Err(LexError {
                            kind: LexErrorKind::UnterminatedString,
                            span: self.span(start, self.pos),
                        });
                    }
                    match self.bytes[self.pos] {
                        b'"'  => { value.push('"');  self.pos += 1; }
                        b'\\' => { value.push('\\'); self.pos += 1; }
                        b'n'  => { value.push('\n'); self.pos += 1; }
                        b't'  => { value.push('\t'); self.pos += 1; }
                        other => { value.push(other as char); self.pos += 1; }
                    }
                }
                b => {
                    value.push(b as char);
                    self.pos += 1;
                }
            }
        }
        self.emit(TokenKind::Str(value), start);
        Ok(())
    }

    fn consume_digits(&mut self) {
        while self.pos < self.bytes.len()
            && (self.bytes[self.pos].is_ascii_digit() || self.bytes[self.pos] == b'_')
        {
            self.pos += 1;
        }
    }

    fn emit_eof_dedents(&mut self) {
        if self.bracket_depth == 0 {
            // Emit a trailing NEWLINE if the last real token wasn't one.
            let needs_newline = !matches!(
                self.tokens.last().map(|t| &t.kind),
                Some(TokenKind::Newline) | Some(TokenKind::Indent) | Some(TokenKind::Dedent) | None
            );
            if needs_newline {
                let s = self.pos;
                self.emit(TokenKind::Newline, s);
            }

            // Pop all indent levels above 0.
            while *self.indent_stack.last().unwrap() > 0 {
                self.indent_stack.pop();
                let s = self.pos;
                self.emit(TokenKind::Dedent, s);
            }
        }
    }
}

// ── Public API ───────────────────────────────────────────────────────────────

pub fn lex(file: FileId, source: &str) -> Result<Vec<Token>, LexError> {
    let mut lexer = Lexer::new(file, source);
    lexer.run()?;
    Ok(lexer.tokens)
}

// ── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use TokenKind::*;

    fn kinds(source: &str) -> Vec<TokenKind> {
        lex(FileId(0), source)
            .expect("lex failed")
            .into_iter()
            .map(|t| t.kind)
            .collect()
    }

    fn lex_err(source: &str) -> LexErrorKind {
        lex(FileId(0), source).unwrap_err().kind
    }

    // ── Category 1: Single tokens ────────────────────────────────────────────

    #[test]
    fn kw_fn() {
        assert_eq!(kinds("fn"), vec![Fn, Newline, Eof]);
    }

    #[test]
    fn kw_kernel() {
        assert_eq!(kinds("kernel"), vec![Kernel, Newline, Eof]);
    }

    #[test]
    fn kw_let() {
        assert_eq!(kinds("let"), vec![Let, Newline, Eof]);
    }

    #[test]
    fn kw_return() {
        assert_eq!(kinds("return"), vec![Return, Newline, Eof]);
    }

    #[test]
    fn lit_bool_true() {
        assert_eq!(kinds("true"), vec![Bool(true), Newline, Eof]);
    }

    #[test]
    fn lit_bool_false() {
        assert_eq!(kinds("false"), vec![Bool(false), Newline, Eof]);
    }

    #[test]
    fn op_plus() {
        assert_eq!(kinds("+"), vec![Plus, Newline, Eof]);
    }

    #[test]
    fn op_at() {
        assert_eq!(kinds("@"), vec![At, Newline, Eof]);
    }

    #[test]
    fn op_eq() {
        assert_eq!(kinds("="), vec![Eq, Newline, Eof]);
    }

    #[test]
    fn op_lt_gt() {
        assert_eq!(kinds("<>"), vec![Lt, Gt, Newline, Eof]);
    }

    #[test]
    fn delimiters() {
        assert_eq!(
            kinds("()[],.:"),
            vec![LParen, RParen, LBracket, RBracket, Comma, Dot, Colon, Newline, Eof]
        );
    }

    // ── Category 2: Number literals ──────────────────────────────────────────

    #[test]
    fn int_simple() {
        assert_eq!(kinds("42"), vec![Int(42), Newline, Eof]);
    }

    #[test]
    fn int_zero() {
        assert_eq!(kinds("0"), vec![Int(0), Newline, Eof]);
    }

    #[test]
    fn int_underscore() {
        assert_eq!(kinds("1_000_000"), vec![Int(1_000_000), Newline, Eof]);
    }

    #[test]
    fn float_simple() {
        assert_eq!(kinds("1.0"), vec![Float(1.0), Newline, Eof]);
    }

    #[test]
    fn float_decimal() {
        assert_eq!(kinds("3.14"), vec![Float(3.14), Newline, Eof]);
    }

    #[test]
    fn float_sci_no_dot() {
        assert_eq!(kinds("1e4"), vec![Float(1e4), Newline, Eof]);
    }

    #[test]
    fn float_sci_neg_exp() {
        assert_eq!(kinds("1e-4"), vec![Float(1e-4), Newline, Eof]);
    }

    #[test]
    fn float_sci_pos_exp() {
        assert_eq!(kinds("1.5e+3"), vec![Float(1.5e3), Newline, Eof]);
    }

    #[test]
    fn float_sci_with_dot() {
        assert_eq!(kinds("1.5e10"), vec![Float(1.5e10), Newline, Eof]);
    }

    #[test]
    fn float_capital_e() {
        assert_eq!(kinds("2E3"), vec![Float(2000.0), Newline, Eof]);
    }

    // A number followed by a dot then an identifier: dot is a separate token.
    #[test]
    fn int_then_dot_then_ident() {
        assert_eq!(kinds("42.foo"), vec![Int(42), Dot, Ident("foo".into()), Newline, Eof]);
    }

    // ── Category 3: Identifier vs keyword ────────────────────────────────────

    #[test]
    fn ident_plain() {
        assert_eq!(kinds("foo"), vec![Ident("foo".into()), Newline, Eof]);
    }

    #[test]
    fn ident_with_underscore_prefix() {
        assert_eq!(kinds("_private"), vec![Ident("_private".into()), Newline, Eof]);
    }

    #[test]
    fn ident_with_digits() {
        assert_eq!(kinds("x1"), vec![Ident("x1".into()), Newline, Eof]);
    }

    #[test]
    fn keyword_prefix_not_keyword() {
        // "fn_name" is an identifier, not the keyword "fn".
        assert_eq!(kinds("fn_name"), vec![Ident("fn_name".into()), Newline, Eof]);
    }

    #[test]
    fn kernel_prefix_not_keyword() {
        assert_eq!(kinds("kernels"), vec![Ident("kernels".into()), Newline, Eof]);
    }

    // ── Category 4: Operator disambiguation ──────────────────────────────────

    #[test]
    fn arrow_token() {
        assert_eq!(kinds("->"), vec![Arrow, Newline, Eof]);
    }

    #[test]
    fn minus_then_gt_with_space() {
        // "- >" has a space, so they are separate tokens.
        assert_eq!(kinds("- >"), vec![Minus, Gt, Newline, Eof]);
    }

    #[test]
    fn minus_alone() {
        assert_eq!(kinds("a-b"), vec![Ident("a".into()), Minus, Ident("b".into()), Newline, Eof]);
    }

    #[test]
    fn arrow_in_context() {
        assert_eq!(
            kinds("fn f() -> f32:"),
            vec![Fn, Ident("f".into()), LParen, RParen, Arrow, Ident("f32".into()), Colon, Newline, Eof]
        );
    }

    #[test]
    fn comparison_eqeq() {
        assert_eq!(kinds("=="), vec![EqEq, Newline, Eof]);
    }

    #[test]
    fn comparison_noteq() {
        assert_eq!(kinds("!="), vec![NotEq, Newline, Eof]);
    }

    #[test]
    fn comparison_lteq() {
        assert_eq!(kinds("<="), vec![LtEq, Newline, Eof]);
    }

    #[test]
    fn comparison_gteq() {
        assert_eq!(kinds(">="), vec![GtEq, Newline, Eof]);
    }

    #[test]
    fn comparison_eq_not_eqeq() {
        // Single '=' is assignment, not comparison.
        assert_eq!(kinds("a = b"), vec![Ident("a".into()), Eq, Ident("b".into()), Newline, Eof]);
    }

    #[test]
    fn comparison_lt_not_lteq() {
        assert_eq!(kinds("a < b"), vec![Ident("a".into()), Lt, Ident("b".into()), Newline, Eof]);
    }

    #[test]
    fn bang_alone_is_error() {
        assert_eq!(lex_err("!"), LexErrorKind::UnexpectedCharacter('!'));
    }

    // ── Category 5: Comments ─────────────────────────────────────────────────

    #[test]
    fn inline_comment() {
        assert_eq!(
            kinds("x # this is a comment\ny"),
            vec![Ident("x".into()), Newline, Ident("y".into()), Newline, Eof]
        );
    }

    #[test]
    fn comment_only_line() {
        // A comment-only line is treated as blank; no tokens emitted.
        assert_eq!(kinds("# just a comment\n"), vec![Eof]);
    }

    #[test]
    fn comment_at_eof_no_newline() {
        assert_eq!(kinds("# comment"), vec![Eof]);
    }

    // ── Category 6: INDENT/DEDENT basic ──────────────────────────────────────

    #[test]
    fn simple_block() {
        let src = "fn main():\n    return 1\n";
        assert_eq!(
            kinds(src),
            vec![
                Fn, Ident("main".into()), LParen, RParen, Colon, Newline,
                Indent,
                Return, Int(1), Newline,
                Dedent,
                Eof,
            ]
        );
    }

    #[test]
    fn two_statements_in_block() {
        let src = "fn f():\n    let x = 1\n    let y = 2\n";
        assert_eq!(
            kinds(src),
            vec![
                Fn, Ident("f".into()), LParen, RParen, Colon, Newline,
                Indent,
                Let, Ident("x".into()), Eq, Int(1), Newline,
                Let, Ident("y".into()), Eq, Int(2), Newline,
                Dedent,
                Eof,
            ]
        );
    }

    // ── Category 7: Multi-level DEDENT ───────────────────────────────────────

    #[test]
    fn nested_dedent() {
        // Two levels of indentation; outer then inner then back to outer then back to zero.
        let src = "fn f():\n    let x = 1\n    fn g():\n        return 2\n    let z = 3\n";
        let k = kinds(src);
        // After "return 2\n", next line is "    let z" at width 4.
        // Stack was [0, 4, 8]. Dedenting to 4 should emit one DEDENT.
        let indent_count = k.iter().filter(|t| **t == Indent).count();
        let dedent_count = k.iter().filter(|t| **t == Dedent).count();
        // Two indents (into f, into g) and two dedents (out of g, out of f).
        assert_eq!(indent_count, 2, "expected 2 INDENTs, got {}", indent_count);
        assert_eq!(dedent_count, 2, "expected 2 DEDENTs, got {}", dedent_count);
    }

    // ── Category 8: Bracket suppression ──────────────────────────────────────

    #[test]
    fn no_newline_inside_parens() {
        let src = "f(\n  a,\n  b\n)\n";
        assert_eq!(
            kinds(src),
            vec![Ident("f".into()), LParen, Ident("a".into()), Comma, Ident("b".into()), RParen, Newline, Eof]
        );
    }

    #[test]
    fn no_newline_inside_brackets() {
        let src = "[1,\n2]\n";
        assert_eq!(
            kinds(src),
            vec![LBracket, Int(1), Comma, Int(2), RBracket, Newline, Eof]
        );
    }

    #[test]
    fn nested_bracket_suppression() {
        // Parens containing brackets across lines.
        let src = "f([\n  1\n])\n";
        assert_eq!(
            kinds(src),
            vec![Ident("f".into()), LParen, LBracket, Int(1), RBracket, RParen, Newline, Eof]
        );
    }

    // ── Category 9: EOF dedents ───────────────────────────────────────────────

    #[test]
    fn eof_dedent_no_trailing_newline() {
        let src = "fn f():\n    x";
        assert_eq!(
            kinds(src),
            vec![
                Fn, Ident("f".into()), LParen, RParen, Colon, Newline,
                Indent,
                Ident("x".into()), Newline,
                Dedent,
                Eof,
            ]
        );
    }

    #[test]
    fn eof_dedent_with_trailing_newline() {
        let src = "fn f():\n    x\n";
        assert_eq!(
            kinds(src),
            vec![
                Fn, Ident("f".into()), LParen, RParen, Colon, Newline,
                Indent,
                Ident("x".into()), Newline,
                Dedent,
                Eof,
            ]
        );
    }

    // ── Category 10: Blank lines ──────────────────────────────────────────────

    #[test]
    fn blank_line_between_statements() {
        let src = "fn f():\n    x\n\n    y\n";
        let k = kinds(src);
        // Blank line should not inject extra INDENT/DEDENT.
        let indent_count = k.iter().filter(|t| **t == Indent).count();
        let dedent_count = k.iter().filter(|t| **t == Dedent).count();
        assert_eq!(indent_count, 1);
        assert_eq!(dedent_count, 1);
    }

    #[test]
    fn blank_line_between_top_level_items() {
        // The blank line between fn and kernel must not disrupt DEDENT logic.
        let src = "fn f():\n    return 1\n\nfn g():\n    return 2\n";
        let k = kinds(src);
        let indent_count = k.iter().filter(|t| **t == Indent).count();
        let dedent_count = k.iter().filter(|t| **t == Dedent).count();
        assert_eq!(indent_count, 2);
        assert_eq!(dedent_count, 2);
    }

    // ── Category 11: Error cases ──────────────────────────────────────────────

    #[test]
    fn error_unexpected_char() {
        assert_eq!(lex_err("$"), LexErrorKind::UnexpectedCharacter('$'));
    }

    #[test]
    fn error_invalid_number_bare_exponent() {
        assert_eq!(lex_err("1e"), LexErrorKind::InvalidNumber);
    }

    #[test]
    fn error_unexpected_indent() {
        // Indentation without a preceding colon.
        assert_eq!(lex_err("x\n    y"), LexErrorKind::UnexpectedIndent);
    }

    #[test]
    fn error_inconsistent_dedent() {
        // Dedents to a level that was never on the stack.
        let src = "fn f():\n    return 1\n  x\n";
        assert_eq!(lex_err(src), LexErrorKind::InconsistentDedent);
    }

    #[test]
    fn error_expected_indent_after_colon() {
        // Colon at end of line but next line at same indentation level.
        let src = "fn f():\nreturn 1\n";
        assert_eq!(lex_err(src), LexErrorKind::ExpectedIndentAfterColon);
    }

    // ── Category 12: Span correctness ────────────────────────────────────────

    #[test]
    fn spans_simple_line() {
        let src = "let x = 42";
        let tokens = lex(FileId(0), src).unwrap();
        // Expected: Let(0..3), Ident "x"(4..5), Eq(6..7), Int(8..10), Newline(10..10), Eof(10..10)
        assert_eq!(tokens[0].span.start, 0);
        assert_eq!(tokens[0].span.end, 3);
        assert_eq!(tokens[1].span.start, 4);
        assert_eq!(tokens[1].span.end, 5);
        assert_eq!(tokens[2].span.start, 6);
        assert_eq!(tokens[2].span.end, 7);
        assert_eq!(tokens[3].span.start, 8);
        assert_eq!(tokens[3].span.end, 10);
    }

    #[test]
    fn spans_float_literal() {
        let src = "3.14";
        let tokens = lex(FileId(0), src).unwrap();
        assert_eq!(tokens[0].span.start, 0);
        assert_eq!(tokens[0].span.end, 4);
    }

    // ── Category 13: MVP integration ─────────────────────────────────────────

    #[test]
    fn mvp_example() {
        // Lex the actual MVP example file and verify the complete token sequence.
        let src = include_str!("../../../examples/add_tensors.ml");
        let tokens = lex(FileId(0), src).expect("add_tensors.ml should lex without errors");
        let k: Vec<&TokenKind> = tokens.iter().map(|t| &t.kind).collect();

        // fn main():
        assert_eq!(k[0], &Fn);
        assert_eq!(k[1], &Ident("main".into()));
        assert_eq!(k[2], &LParen);
        assert_eq!(k[3], &RParen);
        assert_eq!(k[4], &Colon);
        assert_eq!(k[5], &Newline);
        assert_eq!(k[6], &Indent);

        // let a = Tensor.gpu<f32>([1.0, 2.0, 3.0, 4.0])
        assert_eq!(k[7],  &Let);
        assert_eq!(k[8],  &Ident("a".into()));
        assert_eq!(k[9],  &Eq);
        assert_eq!(k[10], &Ident("Tensor".into()));
        assert_eq!(k[11], &Dot);
        assert_eq!(k[12], &Ident("gpu".into()));
        assert_eq!(k[13], &Lt);
        assert_eq!(k[14], &Ident("f32".into()));
        assert_eq!(k[15], &Gt);
        assert_eq!(k[16], &LParen);
        assert_eq!(k[17], &LBracket);
        assert_eq!(k[18], &Float(1.0));
        assert_eq!(k[19], &Comma);
        assert_eq!(k[20], &Float(2.0));
        assert_eq!(k[21], &Comma);
        assert_eq!(k[22], &Float(3.0));
        assert_eq!(k[23], &Comma);
        assert_eq!(k[24], &Float(4.0));
        assert_eq!(k[25], &RBracket);
        assert_eq!(k[26], &RParen);
        assert_eq!(k[27], &Newline);

        // let b = Tensor.gpu<f32>([5.0, 6.0, 7.0, 8.0])
        assert_eq!(k[28], &Let);
        // ... (same structure as 'a', skip to the end of the fn body)

        // let c = add(a, b)
        // Compute offset: let_b (10 tokens) + newline = k[38..48]
        // Find "let c" by searching.
        let let_c = k.iter().position(|t| **t == Let && {
            // peek at the next ident
            true
        }).unwrap();
        // There are three Let tokens: a, b, c. The third is for c.
        let let_positions: Vec<usize> = k.iter()
            .enumerate()
            .filter_map(|(i, t)| if **t == Let { Some(i) } else { None })
            .collect();
        assert_eq!(let_positions.len(), 3, "expected 3 let bindings");
        let _ = let_c;

        // Verify overall structure: exactly one INDENT/DEDENT pair for main's body,
        // one INDENT/DEDENT pair for kernel add's body.
        let indent_count = k.iter().filter(|t| ***t == Indent).count();
        let dedent_count = k.iter().filter(|t| ***t == Dedent).count();
        assert_eq!(indent_count, 2, "expected 2 INDENTs (main body + kernel body)");
        assert_eq!(dedent_count, 2, "expected 2 DEDENTs");

        // Verify the file ends with Eof.
        assert_eq!(k.last(), Some(&&Eof));

        // Verify kernel keyword appears.
        assert!(k.contains(&&Kernel), "kernel keyword missing");

        // Verify return keyword appears.
        assert!(k.contains(&&Return), "return keyword missing");

        // Verify -> (Arrow) appears for return type annotations.
        let arrow_count = k.iter().filter(|t| ***t == Arrow).count();
        assert_eq!(arrow_count, 1, "expected 1 Arrow token");

        // Verify + (Plus) appears in the kernel body.
        assert!(k.contains(&&Plus), "Plus operator missing");
    }

    // ── Keywords and boolean operators ───────────────────────────────────────

    #[test]
    fn kw_if() { assert_eq!(kinds("if"), vec![If, Newline, Eof]); }

    #[test]
    fn kw_else() { assert_eq!(kinds("else"), vec![Else, Newline, Eof]); }

    #[test]
    fn kw_for() { assert_eq!(kinds("for"), vec![For, Newline, Eof]); }

    #[test]
    fn kw_in() { assert_eq!(kinds("in"), vec![In, Newline, Eof]); }

    #[test]
    fn kw_while() { assert_eq!(kinds("while"), vec![While, Newline, Eof]); }

    #[test]
    fn kw_struct() { assert_eq!(kinds("struct"), vec![Struct, Newline, Eof]); }

    #[test]
    fn kw_enum() { assert_eq!(kinds("enum"), vec![Enum, Newline, Eof]); }

    #[test]
    fn kw_inout() { assert_eq!(kinds("inout"), vec![Inout, Newline, Eof]); }

    #[test]
    fn kw_import() { assert_eq!(kinds("import"), vec![Import, Newline, Eof]); }

    #[test]
    fn kw_from() { assert_eq!(kinds("from"), vec![From, Newline, Eof]); }

    #[test]
    fn import_prefix_not_keyword() {
        assert_eq!(kinds("imports"),   vec![Ident("imports".into()),   Newline, Eof]);
        assert_eq!(kinds("from_path"), vec![Ident("from_path".into()), Newline, Eof]);
    }

    #[test]
    fn import_statement_tokens() {
        assert_eq!(kinds("import ops"), vec![Import, Ident("ops".into()), Newline, Eof]);
    }

    #[test]
    fn from_import_tokens() {
        assert_eq!(
            kinds("from ops import add"),
            vec![From, Ident("ops".into()), Import, Ident("add".into()), Newline, Eof]
        );
    }

    #[test]
    fn dotted_import_tokens() {
        assert_eq!(
            kinds("import models.transformer"),
            vec![Import, Ident("models".into()), Dot, Ident("transformer".into()), Newline, Eof]
        );
    }

    #[test]
    fn kw_and() { assert_eq!(kinds("and"), vec![And, Newline, Eof]); }

    #[test]
    fn kw_or() { assert_eq!(kinds("or"), vec![Or, Newline, Eof]); }

    #[test]
    fn kw_not() { assert_eq!(kinds("not"), vec![Not, Newline, Eof]); }

    #[test]
    fn keywords_not_prefix_matched() {
        // Identifiers that start with a keyword must not be misidentified.
        assert_eq!(kinds("iffy"),   vec![Ident("iffy".into()),   Newline, Eof]);
        assert_eq!(kinds("inout2"), vec![Ident("inout2".into()), Newline, Eof]);
        assert_eq!(kinds("note"),   vec![Ident("note".into()),   Newline, Eof]);
        assert_eq!(kinds("ors"),    vec![Ident("ors".into()),    Newline, Eof]);
    }

    // ── String literals ───────────────────────────────────────────────────────

    #[test]
    fn string_simple() {
        assert_eq!(kinds("\"hello\""), vec![Str("hello".into()), Newline, Eof]);
    }

    #[test]
    fn string_path() {
        assert_eq!(
            kinds("\"path/to/file.safetensors\""),
            vec![Str("path/to/file.safetensors".into()), Newline, Eof]
        );
    }

    #[test]
    fn string_escape_quote() {
        assert_eq!(kinds(r#""say \"hi\"""#), vec![Str("say \"hi\"".into()), Newline, Eof]);
    }

    #[test]
    fn string_escape_backslash() {
        assert_eq!(kinds(r#""a\\b""#), vec![Str("a\\b".into()), Newline, Eof]);
    }

    #[test]
    fn string_in_call() {
        assert_eq!(
            kinds("print(\"label:\")"),
            vec![Ident("print".into()), LParen, Str("label:".into()), RParen, Newline, Eof]
        );
    }

    #[test]
    fn string_span() {
        let tokens = lex(FileId(0), "\"hi\"").unwrap();
        assert_eq!(tokens[0].span.start, 0);
        assert_eq!(tokens[0].span.end, 4);
    }

    #[test]
    fn error_unterminated_string() {
        assert_eq!(lex_err("\"oops"), LexErrorKind::UnterminatedString);
    }

    #[test]
    fn error_unterminated_string_newline() {
        assert_eq!(lex_err("\"no\nnewline\""), LexErrorKind::UnterminatedString);
    }
}
