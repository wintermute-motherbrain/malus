use crate::span::Span;

#[derive(Debug, Clone, PartialEq)]
pub struct Token {
    pub kind: TokenKind,
    pub span: Span,
}

#[derive(Debug, Clone, PartialEq)]
pub enum TokenKind {
    // Keywords
    Fn,
    Kernel,
    Let,
    Return,
    If,
    Else,
    For,
    From,
    Import,
    In,
    While,
    Struct,
    Enum,
    Inout,

    // Boolean keyword-operators
    And,
    Or,
    Not,

    // Literals
    Int(i64),
    Float(f64),
    Bool(bool),
    Str(String),

    // Identifier
    Ident(String),

    // Operators
    Plus,       // +
    Minus,      // -
    Star,       // *
    Slash,      // /
    At,         // @
    Eq,         // =
    EqEq,       // ==
    NotEq,      // !=
    Arrow,      // ->
    Lt,         // <
    LtEq,       // <=
    Gt,         // >
    GtEq,       // >=

    // Delimiters
    LParen,     // (
    RParen,     // )
    LBracket,   // [
    RBracket,   // ]
    Comma,      // ,
    Dot,        // .
    Colon,      // :

    // Synthetic structural tokens
    Newline,
    Indent,
    Dedent,

    Eof,
}
