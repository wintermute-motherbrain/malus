use crate::ast::*;
use crate::span::{FileId, Span};
use crate::token::{Token, TokenKind};

// ── Error ─────────────────────────────────────────────────────────────────────

#[derive(Debug, Clone, PartialEq)]
pub struct ParseError {
    pub message: String,
    pub span: Span,
}

impl ParseError {
    fn new(msg: impl Into<String>, span: Span) -> Self {
        Self { message: msg.into(), span }
    }
}

impl std::fmt::Display for ParseError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.message)
    }
}

// ── Parser ────────────────────────────────────────────────────────────────────

struct Parser {
    tokens: Vec<Token>,
    pos: usize,
}

impl Parser {
    fn new(tokens: Vec<Token>) -> Self {
        Self { tokens, pos: 0 }
    }

    // ── Token navigation ──────────────────────────────────────────────────────

    fn current(&self) -> &Token {
        &self.tokens[self.pos]
    }

    fn current_kind(&self) -> &TokenKind {
        &self.current().kind
    }

    fn current_span(&self) -> Span {
        self.current().span
    }

    #[allow(dead_code)]
    fn peek(&self) -> &TokenKind {
        // Look past Newline tokens one step ahead.
        let mut i = self.pos + 1;
        while i < self.tokens.len() {
            match &self.tokens[i].kind {
                TokenKind::Newline => i += 1,
                k => return k,
            }
        }
        &TokenKind::Eof
    }

    fn advance(&mut self) -> &Token {
        let t = &self.tokens[self.pos];
        if self.pos + 1 < self.tokens.len() {
            self.pos += 1;
        }
        t
    }

    fn skip_newlines(&mut self) {
        while matches!(self.current_kind(), TokenKind::Newline) {
            self.advance();
        }
    }

    fn expect(&mut self, kind: &TokenKind) -> Result<Span, ParseError> {
        if self.current_kind() == kind {
            Ok(self.advance().span)
        } else {
            Err(ParseError::new(
                format!("expected {:?}, got {:?}", kind, self.current_kind()),
                self.current_span(),
            ))
        }
    }

    fn expect_ident(&mut self) -> Result<(String, Span), ParseError> {
        match self.current_kind().clone() {
            TokenKind::Ident(s) => {
                let span = self.current_span();
                self.advance();
                Ok((s, span))
            }
            _ => Err(ParseError::new(
                format!("expected identifier, got {:?}", self.current_kind()),
                self.current_span(),
            )),
        }
    }

    fn at_end(&self) -> bool {
        matches!(self.current_kind(), TokenKind::Eof)
    }

    // ── Types ─────────────────────────────────────────────────────────────────

    fn parse_scalar_ty(name: &str) -> Option<ScalarTy> {
        match name {
            "f32"  => Some(ScalarTy::F32),
            "f16"  => Some(ScalarTy::F16),
            "bf16" => Some(ScalarTy::Bf16),
            "i8"   => Some(ScalarTy::I8),
            "i16"  => Some(ScalarTy::I16),
            "i32"  => Some(ScalarTy::I32),
            "i64"  => Some(ScalarTy::I64),
            "u8"   => Some(ScalarTy::U8),
            "u16"  => Some(ScalarTy::U16),
            "u32"  => Some(ScalarTy::U32),
            "u64"  => Some(ScalarTy::U64),
            _      => None,
        }
    }

    fn parse_type(&mut self) -> Result<Ty, ParseError> {
        let span = self.current_span();

        // Tensor<dtype>
        if let TokenKind::Ident(name) = self.current_kind().clone() {
            if name == "Tensor" {
                self.advance();
                self.expect(&TokenKind::Lt)?;
                let (dtype_name, dtype_span) = self.expect_ident()?;
                let dtype = Self::parse_scalar_ty(&dtype_name).ok_or_else(|| {
                    ParseError::new(format!("unknown dtype '{}'", dtype_name), dtype_span)
                })?;
                self.expect(&TokenKind::Gt)?;
                return Ok(Ty::Tensor { dtype });
            }

            // Scalar type
            if let Some(s) = Self::parse_scalar_ty(&name) {
                self.advance();
                return Ok(Ty::Scalar(s));
            }
        }

        // bool keyword
        if matches!(self.current_kind(), TokenKind::Bool(_)) {
            self.advance();
            return Ok(Ty::Bool);
        }

        // Tuple ( ty, ty, ... ) — two or more types
        if matches!(self.current_kind(), TokenKind::LParen) {
            self.advance();
            let first = self.parse_type()?;
            self.expect(&TokenKind::Comma)?;
            let mut tys = vec![first, self.parse_type()?];
            while matches!(self.current_kind(), TokenKind::Comma) {
                self.advance();
                tys.push(self.parse_type()?);
            }
            self.expect(&TokenKind::RParen)?;
            return Ok(Ty::Tuple(tys));
        }

        // Named type (struct / enum name)
        if let TokenKind::Ident(name) = self.current_kind().clone() {
            self.advance();
            return Ok(Ty::Named(name));
        }

        Err(ParseError::new(
            format!("expected type, got {:?}", self.current_kind()),
            span,
        ))
    }

    // ── Parameters ────────────────────────────────────────────────────────────

    fn parse_params(&mut self) -> Result<Vec<Param>, ParseError> {
        self.expect(&TokenKind::LParen)?;
        let mut params = Vec::new();
        while !matches!(self.current_kind(), TokenKind::RParen | TokenKind::Eof) {
            let start = self.current_span();
            let (name, _) = self.expect_ident()?;
            self.expect(&TokenKind::Colon)?;
            let ty = self.parse_type()?;
            let end = self.current_span();
            params.push(Param {
                name,
                ty,
                span: Span::new(start.file, start.start as usize, end.start as usize),
            });
            if matches!(self.current_kind(), TokenKind::Comma) {
                self.advance();
            } else {
                break;
            }
        }
        self.expect(&TokenKind::RParen)?;
        Ok(params)
    }

    fn parse_kernel_params(&mut self) -> Result<Vec<KernelParam>, ParseError> {
        self.expect(&TokenKind::LParen)?;
        let mut params = Vec::new();
        while !matches!(self.current_kind(), TokenKind::RParen | TokenKind::Eof) {
            let start = self.current_span();
            let inout = if matches!(self.current_kind(), TokenKind::Inout) {
                self.advance();
                true
            } else {
                false
            };
            let (name, _) = self.expect_ident()?;
            self.expect(&TokenKind::Colon)?;
            let ty = self.parse_type()?;
            let end = self.current_span();
            params.push(KernelParam {
                inout,
                name,
                ty,
                span: Span::new(start.file, start.start as usize, end.start as usize),
            });
            if matches!(self.current_kind(), TokenKind::Comma) {
                self.advance();
            } else {
                break;
            }
        }
        self.expect(&TokenKind::RParen)?;
        Ok(params)
    }

    // ── Body (INDENT stmts DEDENT) ────────────────────────────────────────────

    fn parse_body(&mut self) -> Result<Vec<Stmt>, ParseError> {
        self.expect(&TokenKind::Newline)?;
        if !matches!(self.current_kind(), TokenKind::Indent) {
            return Err(ParseError::new(
                "expected an indented block",
                self.current_span(),
            ));
        }
        self.advance(); // consume INDENT
        let mut stmts = Vec::new();
        loop {
            self.skip_newlines();
            if matches!(self.current_kind(), TokenKind::Dedent | TokenKind::Eof) {
                break;
            }
            stmts.push(self.parse_stmt()?);
        }
        if matches!(self.current_kind(), TokenKind::Dedent) {
            self.advance(); // consume DEDENT
        }
        if stmts.is_empty() {
            // The lexer already enforced that an indent follows ':', but
            // a block with only newlines would reach here empty.
            return Err(ParseError::new("empty block", self.current_span()));
        }
        Ok(stmts)
    }

    // ── Statements ────────────────────────────────────────────────────────────

    fn parse_stmt(&mut self) -> Result<Stmt, ParseError> {
        let start = self.current_span();

        match self.current_kind().clone() {
            TokenKind::Let => {
                self.advance();
                let (name, _) = self.expect_ident()?;
                self.expect(&TokenKind::Eq)?;
                let expr = self.parse_expr()?;
                let end = expr.span;
                self.expect_newline_or_eof()?;
                Ok(Stmt {
                    kind: StmtKind::Let { name, expr },
                    span: Span::new(start.file, start.start as usize, end.end as usize),
                })
            }
            TokenKind::Return => {
                self.advance();
                let expr = self.parse_expr()?;
                let end = expr.span;
                self.expect_newline_or_eof()?;
                Ok(Stmt {
                    kind: StmtKind::Return { expr },
                    span: Span::new(start.file, start.start as usize, end.end as usize),
                })
            }
            _ => {
                let expr = self.parse_expr()?;
                let end = expr.span;
                self.expect_newline_or_eof()?;
                Ok(Stmt {
                    kind: StmtKind::Expr(expr),
                    span: Span::new(start.file, start.start as usize, end.end as usize),
                })
            }
        }
    }

    fn expect_newline_or_eof(&mut self) -> Result<(), ParseError> {
        match self.current_kind() {
            TokenKind::Newline => { self.advance(); Ok(()) }
            TokenKind::Eof | TokenKind::Dedent => Ok(()),
            _ => Err(ParseError::new(
                format!("expected newline, got {:?}", self.current_kind()),
                self.current_span(),
            )),
        }
    }

    // ── Expressions (Pratt) ───────────────────────────────────────────────────

    fn parse_expr(&mut self) -> Result<Expr, ParseError> {
        self.parse_expr_bp(0)
    }

    fn infix_bp(kind: &TokenKind) -> Option<(u8, u8)> {
        match kind {
            TokenKind::Or                  => Some((1, 2)),
            TokenKind::And                 => Some((3, 4)),
            TokenKind::EqEq | TokenKind::NotEq
            | TokenKind::Lt | TokenKind::LtEq
            | TokenKind::Gt | TokenKind::GtEq => Some((5, 6)),
            TokenKind::Plus | TokenKind::Minus => Some((7, 8)),
            TokenKind::Star | TokenKind::Slash | TokenKind::At => Some((9, 10)),
            _ => None,
        }
    }

    fn token_to_binop(kind: &TokenKind) -> Option<BinOp> {
        match kind {
            TokenKind::Plus  => Some(BinOp::Add),
            TokenKind::Minus => Some(BinOp::Sub),
            TokenKind::Star  => Some(BinOp::Mul),
            TokenKind::Slash => Some(BinOp::Div),
            TokenKind::At    => Some(BinOp::Matmul),
            TokenKind::EqEq  => Some(BinOp::Eq),
            TokenKind::NotEq => Some(BinOp::NotEq),
            TokenKind::Lt    => Some(BinOp::Lt),
            TokenKind::LtEq  => Some(BinOp::LtEq),
            TokenKind::Gt    => Some(BinOp::Gt),
            TokenKind::GtEq  => Some(BinOp::GtEq),
            TokenKind::And   => Some(BinOp::And),
            TokenKind::Or    => Some(BinOp::Or),
            _ => None,
        }
    }

    fn parse_expr_bp(&mut self, min_bp: u8) -> Result<Expr, ParseError> {
        let mut lhs = self.parse_unary()?;

        loop {
            let kind = self.current_kind().clone();
            let Some((l_bp, r_bp)) = Self::infix_bp(&kind) else { break };
            if l_bp < min_bp { break; }
            let op_span = self.current_span();
            self.advance();
            let rhs = self.parse_expr_bp(r_bp)?;
            let span = Span::new(lhs.span.file, lhs.span.start as usize, rhs.span.end as usize);
            let op = Self::token_to_binop(&kind).unwrap();
            lhs = Expr { kind: ExprKind::BinOp { op, lhs: Box::new(lhs), rhs: Box::new(rhs) }, span };
            let _ = op_span;
        }

        Ok(lhs)
    }

    fn parse_unary(&mut self) -> Result<Expr, ParseError> {
        let start = self.current_span();
        match self.current_kind().clone() {
            TokenKind::Minus => {
                self.advance();
                let operand = self.parse_unary()?;
                let end = operand.span;
                Ok(Expr {
                    kind: ExprKind::Unary { op: UnaryOp::Neg, operand: Box::new(operand) },
                    span: Span::new(start.file, start.start as usize, end.end as usize),
                })
            }
            TokenKind::Not => {
                self.advance();
                let operand = self.parse_unary()?;
                let end = operand.span;
                Ok(Expr {
                    kind: ExprKind::Unary { op: UnaryOp::Not, operand: Box::new(operand) },
                    span: Span::new(start.file, start.start as usize, end.end as usize),
                })
            }
            _ => self.parse_postfix(),
        }
    }

    fn parse_postfix(&mut self) -> Result<Expr, ParseError> {
        let mut base = self.parse_primary()?;

        loop {
            match self.current_kind().clone() {
                // Function call: base(args)
                TokenKind::LParen => {
                    self.advance();
                    let mut args = Vec::new();
                    while !matches!(self.current_kind(), TokenKind::RParen | TokenKind::Eof) {
                        args.push(self.parse_expr()?);
                        if matches!(self.current_kind(), TokenKind::Comma) {
                            self.advance();
                        } else {
                            break;
                        }
                    }
                    let end = self.current_span();
                    self.expect(&TokenKind::RParen)?;
                    let span = Span::new(base.span.file, base.span.start as usize, end.end as usize);
                    base = Expr { kind: ExprKind::Call { callee: Box::new(base), args }, span };
                }
                // Index: base[indices]
                TokenKind::LBracket => {
                    self.advance();
                    let mut indices = Vec::new();
                    while !matches!(self.current_kind(), TokenKind::RBracket | TokenKind::Eof) {
                        indices.push(self.parse_expr()?);
                        if matches!(self.current_kind(), TokenKind::Comma) {
                            self.advance();
                        } else {
                            break;
                        }
                    }
                    let end = self.current_span();
                    self.expect(&TokenKind::RBracket)?;
                    let span = Span::new(base.span.file, base.span.start as usize, end.end as usize);
                    base = Expr { kind: ExprKind::Index { base: Box::new(base), indices }, span };
                }
                // Field / method access: base.name  or  Tensor.gpu<dtype>([...])
                TokenKind::Dot => {
                    self.advance();
                    let (name, name_span) = self.expect_ident()?;
                    let span = Span::new(base.span.file, base.span.start as usize, name_span.end as usize);

                    // Tensor literal: Tensor.cpu<dtype>([...]) or Tensor.gpu<dtype>([...])
                    if let ExprKind::Ident(ref base_name) = base.kind.clone() {
                        if base_name == "Tensor" && (name == "cpu" || name == "gpu") {
                            // Check for <dtype>([...])
                            if matches!(self.current_kind(), TokenKind::Lt) {
                                base = self.parse_tensor_literal(
                                    if name == "gpu" { Placement::Gpu } else { Placement::Cpu },
                                    base.span,
                                )?;
                                continue;
                            }
                        }
                    }

                    // Plain field access — represent as a Call on a dot-chained ident for now.
                    // The parser models `a.b` as ident "b" accessed via postfix; we represent
                    // it as a call with zero args using a synthetic callee for field access.
                    // For M1 this is sufficient — only Tensor.gpu<...>([...]) needs special handling.
                    let field_expr = Expr { kind: ExprKind::Ident(name), span: name_span };
                    base = Expr {
                        kind: ExprKind::Call { callee: Box::new(field_expr), args: vec![base] },
                        span,
                    };
                }
                _ => break,
            }
        }

        Ok(base)
    }

    /// Parse `<dtype>([elements])` after `Tensor.gpu` / `Tensor.cpu` has been consumed.
    fn parse_tensor_literal(&mut self, placement: Placement, start: Span) -> Result<Expr, ParseError> {
        self.expect(&TokenKind::Lt)?;
        let (dtype_name, dtype_span) = self.expect_ident()?;
        let dtype = Self::parse_scalar_ty(&dtype_name).ok_or_else(|| {
            ParseError::new(format!("unknown dtype '{}'", dtype_name), dtype_span)
        })?;
        self.expect(&TokenKind::Gt)?;
        self.expect(&TokenKind::LParen)?;
        self.expect(&TokenKind::LBracket)?;
        let mut elements = Vec::new();
        while !matches!(self.current_kind(), TokenKind::RBracket | TokenKind::Eof) {
            elements.push(self.parse_expr()?);
            if matches!(self.current_kind(), TokenKind::Comma) {
                self.advance();
            } else {
                break;
            }
        }
        let end = self.current_span();
        self.expect(&TokenKind::RBracket)?;
        self.expect(&TokenKind::RParen)?;
        let span = Span::new(start.file, start.start as usize, end.end as usize);
        Ok(Expr { kind: ExprKind::TensorLiteral { placement, dtype, elements }, span })
    }

    fn parse_primary(&mut self) -> Result<Expr, ParseError> {
        let span = self.current_span();
        match self.current_kind().clone() {
            TokenKind::Int(v) => { self.advance(); Ok(Expr { kind: ExprKind::Lit(Lit::Int(v)), span }) }
            TokenKind::Float(v) => { self.advance(); Ok(Expr { kind: ExprKind::Lit(Lit::Float(v)), span }) }
            TokenKind::Bool(v) => { self.advance(); Ok(Expr { kind: ExprKind::Lit(Lit::Bool(v)), span }) }
            TokenKind::Str(v) => { self.advance(); Ok(Expr { kind: ExprKind::Lit(Lit::Str(v)), span }) }
            TokenKind::Ident(name) => { self.advance(); Ok(Expr { kind: ExprKind::Ident(name), span }) }
            TokenKind::LParen => {
                self.advance();
                let inner = self.parse_expr()?;
                self.expect(&TokenKind::RParen)?;
                Ok(inner)
            }
            _ => Err(ParseError::new(
                format!("expected expression, got {:?}", self.current_kind()),
                span,
            )),
        }
    }

    // ── Top-level items ───────────────────────────────────────────────────────

    fn parse_fn(&mut self) -> Result<Item, ParseError> {
        let start = self.current_span();
        self.expect(&TokenKind::Fn)?;
        let (name, _) = self.expect_ident()?;
        let params = self.parse_params()?;
        let return_ty = if matches!(self.current_kind(), TokenKind::Arrow) {
            self.advance();
            Some(self.parse_type()?)
        } else {
            None
        };
        self.expect(&TokenKind::Colon)?;
        let body = self.parse_body()?;
        let end = self.current_span();
        Ok(Item {
            kind: ItemKind::Fn { name, params, return_ty, body },
            span: Span::new(start.file, start.start as usize, end.start as usize),
        })
    }

    fn parse_kernel(&mut self) -> Result<Item, ParseError> {
        let start = self.current_span();
        self.expect(&TokenKind::Kernel)?;
        let (name, _) = self.expect_ident()?;
        let params = self.parse_kernel_params()?;
        self.expect(&TokenKind::Arrow)?;
        let return_ty = self.parse_type()?;
        self.expect(&TokenKind::Colon)?;
        let body = self.parse_body()?;
        let end = self.current_span();
        Ok(Item {
            kind: ItemKind::Kernel { name, params, return_ty, body },
            span: Span::new(start.file, start.start as usize, end.start as usize),
        })
    }

    fn parse_program(&mut self) -> Result<Program, ParseError> {
        let mut items = Vec::new();
        self.skip_newlines();
        while !self.at_end() {
            match self.current_kind() {
                TokenKind::Fn => items.push(self.parse_fn()?),
                TokenKind::Kernel => items.push(self.parse_kernel()?),
                _ => return Err(ParseError::new(
                    format!("expected 'fn' or 'kernel', got {:?}", self.current_kind()),
                    self.current_span(),
                )),
            }
            self.skip_newlines();
        }
        Ok(Program { items })
    }
}

// ── Public API ────────────────────────────────────────────────────────────────

pub fn parse(file: FileId, source: &str) -> Result<Program, ParseError> {
    let tokens = crate::lexer::lex(file, source).map_err(|e| {
        ParseError::new(e.to_string(), e.span)
    })?;
    let mut parser = Parser::new(tokens);
    parser.parse_program()
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::span::FileId;

    fn parse_ok(src: &str) -> Program {
        parse(FileId(0), src).expect("parse failed")
    }

    #[allow(dead_code)]
    fn parse_err(src: &str) -> String {
        parse(FileId(0), src).unwrap_err().message
    }

    // ── M1 milestone tests ────────────────────────────────────────────────────

    #[test]
    fn mvp_add_tensors() {
        let src = include_str!("../../../examples/add_tensors.ml");
        let prog = parse_ok(src);
        assert_eq!(prog.items.len(), 2);
        assert!(matches!(prog.items[0].kind, ItemKind::Fn { .. }));
        assert!(matches!(prog.items[1].kind, ItemKind::Kernel { .. }));
    }

    #[test]
    fn fn_no_body_is_error() {
        // A fn with no body should produce an error at a valid span.
        let err = parse(FileId(0), "fn f():\n").unwrap_err();
        assert!(!err.message.is_empty());
        // Span should exist (non-zero file id matches).
        assert_eq!(err.span.file, FileId(0));
    }

    #[test]
    fn tensor_literal_node() {
        let src = "fn f():\n    return Tensor.gpu<f32>([1.0])\n";
        let prog = parse_ok(src);
        let ItemKind::Fn { body, .. } = &prog.items[0].kind else { panic!() };
        let StmtKind::Return { expr } = &body[0].kind else { panic!() };
        assert!(matches!(
            &expr.kind,
            ExprKind::TensorLiteral { placement: Placement::Gpu, dtype: ScalarTy::F32, elements }
            if elements.len() == 1
        ));
    }

    #[test]
    fn binop_add() {
        let src = "fn f():\n    return a + b\n";
        let prog = parse_ok(src);
        let ItemKind::Fn { body, .. } = &prog.items[0].kind else { panic!() };
        let StmtKind::Return { expr } = &body[0].kind else { panic!() };
        assert!(matches!(
            &expr.kind,
            ExprKind::BinOp { op: BinOp::Add, .. }
        ));
    }

    #[test]
    fn binop_matmul() {
        let src = "fn f():\n    return a @ b\n";
        let prog = parse_ok(src);
        let ItemKind::Fn { body, .. } = &prog.items[0].kind else { panic!() };
        let StmtKind::Return { expr } = &body[0].kind else { panic!() };
        assert!(matches!(
            &expr.kind,
            ExprKind::BinOp { op: BinOp::Matmul, .. }
        ));
    }

    // ── Structural tests ──────────────────────────────────────────────────────

    #[test]
    fn fn_params_and_return_ty() {
        let src = "fn add(a: f32, b: f32) -> f32:\n    return a + b\n";
        let prog = parse_ok(src);
        let ItemKind::Fn { params, return_ty, .. } = &prog.items[0].kind else { panic!() };
        assert_eq!(params.len(), 2);
        assert_eq!(params[0].name, "a");
        assert!(matches!(params[0].ty, Ty::Scalar(ScalarTy::F32)));
        assert!(matches!(return_ty, Some(Ty::Scalar(ScalarTy::F32))));
    }

    #[test]
    fn kernel_params_and_return_ty() {
        let src = "kernel add(a: Tensor<f32>, b: Tensor<f32>) -> Tensor<f32>:\n    return a + b\n";
        let prog = parse_ok(src);
        let ItemKind::Kernel { params, return_ty, .. } = &prog.items[0].kind else { panic!() };
        assert_eq!(params.len(), 2);
        assert!(matches!(&params[0].ty, Ty::Tensor { dtype: ScalarTy::F32 }));
        assert!(matches!(return_ty, Ty::Tensor { dtype: ScalarTy::F32 }));
    }

    #[test]
    fn let_binding() {
        let src = "fn f():\n    let x = 42\n";
        let prog = parse_ok(src);
        let ItemKind::Fn { body, .. } = &prog.items[0].kind else { panic!() };
        let StmtKind::Let { name, expr } = &body[0].kind else { panic!() };
        assert_eq!(name, "x");
        assert!(matches!(expr.kind, ExprKind::Lit(Lit::Int(42))));
    }

    #[test]
    fn fn_call_expr() {
        let src = "fn f():\n    print(x)\n";
        let prog = parse_ok(src);
        let ItemKind::Fn { body, .. } = &prog.items[0].kind else { panic!() };
        let StmtKind::Expr(expr) = &body[0].kind else { panic!() };
        assert!(matches!(&expr.kind, ExprKind::Call { args, .. } if args.len() == 1));
    }

    #[test]
    fn precedence_mul_before_add() {
        // a + b * c should parse as a + (b * c)
        let src = "fn f():\n    return a + b * c\n";
        let prog = parse_ok(src);
        let ItemKind::Fn { body, .. } = &prog.items[0].kind else { panic!() };
        let StmtKind::Return { expr } = &body[0].kind else { panic!() };
        let ExprKind::BinOp { op: BinOp::Add, rhs, .. } = &expr.kind else { panic!() };
        assert!(matches!(rhs.kind, ExprKind::BinOp { op: BinOp::Mul, .. }));
    }

    #[test]
    fn unary_neg() {
        let src = "fn f():\n    return -x\n";
        let prog = parse_ok(src);
        let ItemKind::Fn { body, .. } = &prog.items[0].kind else { panic!() };
        let StmtKind::Return { expr } = &body[0].kind else { panic!() };
        assert!(matches!(expr.kind, ExprKind::Unary { op: UnaryOp::Neg, .. }));
    }

    #[test]
    fn multiple_items() {
        let src = "fn f():\n    return 1\n\nfn g():\n    return 2\n";
        let prog = parse_ok(src);
        assert_eq!(prog.items.len(), 2);
    }

    #[test]
    fn no_params_fn() {
        let src = "fn main():\n    return 0\n";
        let prog = parse_ok(src);
        let ItemKind::Fn { params, .. } = &prog.items[0].kind else { panic!() };
        assert!(params.is_empty());
    }

    #[test]
    fn tensor_cpu_literal() {
        let src = "fn f():\n    return Tensor.cpu<i32>([1, 2, 3])\n";
        let prog = parse_ok(src);
        let ItemKind::Fn { body, .. } = &prog.items[0].kind else { panic!() };
        let StmtKind::Return { expr } = &body[0].kind else { panic!() };
        assert!(matches!(
            &expr.kind,
            ExprKind::TensorLiteral { placement: Placement::Cpu, dtype: ScalarTy::I32, elements }
            if elements.len() == 3
        ));
    }

    #[test]
    fn comparison_ops() {
        let src = "fn f():\n    return a == b\n";
        let prog = parse_ok(src);
        let ItemKind::Fn { body, .. } = &prog.items[0].kind else { panic!() };
        let StmtKind::Return { expr } = &body[0].kind else { panic!() };
        assert!(matches!(expr.kind, ExprKind::BinOp { op: BinOp::Eq, .. }));
    }

    #[test]
    fn bool_and_or() {
        let src = "fn f():\n    return a and b\n";
        let prog = parse_ok(src);
        let ItemKind::Fn { body, .. } = &prog.items[0].kind else { panic!() };
        let StmtKind::Return { expr } = &body[0].kind else { panic!() };
        assert!(matches!(expr.kind, ExprKind::BinOp { op: BinOp::And, .. }));
    }

    #[test]
    fn empty_tensor_literal() {
        let src = "fn f():\n    return Tensor.gpu<f32>([])\n";
        let prog = parse_ok(src);
        let ItemKind::Fn { body, .. } = &prog.items[0].kind else { panic!() };
        let StmtKind::Return { expr } = &body[0].kind else { panic!() };
        assert!(matches!(
            &expr.kind,
            ExprKind::TensorLiteral { elements, .. } if elements.is_empty()
        ));
    }

    #[test]
    fn inout_kernel_param() {
        let src = "kernel relu(inout a: Tensor<f32>) -> Tensor<f32>:\n    return a\n";
        let prog = parse_ok(src);
        let ItemKind::Kernel { params, .. } = &prog.items[0].kind else { panic!() };
        assert!(params[0].inout);
    }
}
