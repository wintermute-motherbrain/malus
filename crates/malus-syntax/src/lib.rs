// Lexer, parser, and AST for the malus language.
// Handles: fn/kernel declarations, let bindings, tensor types,
// control flow, bracket indexing, and operator expressions.

pub mod span;
pub mod token;
pub mod lexer;
pub mod ast;
pub mod parser;

pub use span::{FileId, Span};
pub use token::{Token, TokenKind};
pub use lexer::{lex, LexError, LexErrorKind};
pub use parser::{parse, ParseError};
