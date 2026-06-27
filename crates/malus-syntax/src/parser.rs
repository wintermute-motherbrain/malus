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
                let mutable = if self.current_kind() == &TokenKind::Mut {
                    self.advance();
                    true
                } else {
                    false
                };
                let (name, _) = self.expect_ident()?;
                self.expect(&TokenKind::Eq)?;
                let expr = self.parse_expr()?;
                let end = expr.span;
                self.expect_newline_or_eof()?;
                let kind = if mutable {
                    StmtKind::LetMut { name, expr }
                } else {
                    StmtKind::Let { name, expr }
                };
                Ok(Stmt {
                    kind,
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
            // ── Control flow ─────────────────────────────────────────────────
            //
            // These stmts end on a DEDENT (not a Newline), so they do NOT call
            // `expect_newline_or_eof` — mirroring how `parse_fn`/`parse_kernel`
            // consume `Colon` then `parse_body` and stop.
            TokenKind::If => {
                self.advance(); // consume 'if'
                let condition = self.parse_expr()?;
                self.expect(&TokenKind::Colon)?;
                let then_body = self.parse_body()?;
                // Optional `else:` clause.  `else if` is written as an `if`
                // inside the else block — we do not support `else if` directly.
                let else_body = if matches!(self.current_kind(), TokenKind::Else) {
                    self.advance(); // consume 'else'
                    self.expect(&TokenKind::Colon)?;
                    Some(self.parse_body()?)
                } else {
                    None
                };
                let end = self.current_span();
                Ok(Stmt {
                    kind: StmtKind::If { condition, then_body, else_body },
                    span: Span::new(start.file, start.start as usize, end.start as usize),
                })
            }
            TokenKind::While => {
                self.advance(); // consume 'while'
                let condition = self.parse_expr()?;
                self.expect(&TokenKind::Colon)?;
                let body = self.parse_body()?;
                let end = self.current_span();
                Ok(Stmt {
                    kind: StmtKind::While { condition, body },
                    span: Span::new(start.file, start.start as usize, end.start as usize),
                })
            }
            TokenKind::Match => {
                self.advance(); // consume 'match'
                let scrutinee = self.parse_expr()?;
                self.expect(&TokenKind::Colon)?;
                self.expect(&TokenKind::Newline)?;
                if !matches!(self.current_kind(), TokenKind::Indent) {
                    return Err(ParseError::new("expected indented match arms", self.current_span()));
                }
                self.advance(); // consume INDENT
                let mut arms = Vec::new();
                loop {
                    self.skip_newlines();
                    if matches!(self.current_kind(), TokenKind::Dedent | TokenKind::Eof) {
                        break;
                    }
                    let arm_start = self.current_span();
                    let (variant, _) = self.expect_ident()?;
                    let mut bindings = Vec::new();
                    if matches!(self.current_kind(), TokenKind::LParen) {
                        self.advance(); // consume '('
                        while !matches!(self.current_kind(), TokenKind::RParen | TokenKind::Eof) {
                            let (binding, _) = self.expect_ident()?;
                            bindings.push(binding);
                            if matches!(self.current_kind(), TokenKind::Comma) {
                                self.advance();
                            } else {
                                break;
                            }
                        }
                        self.expect(&TokenKind::RParen)?;
                    }
                    self.expect(&TokenKind::Colon)?;
                    let body = self.parse_body()?;
                    let arm_end = self.current_span();
                    arms.push(MatchArm {
                        variant,
                        bindings,
                        body,
                        span: Span::new(arm_start.file, arm_start.start as usize, arm_end.start as usize),
                    });
                }
                if matches!(self.current_kind(), TokenKind::Dedent) {
                    self.advance(); // consume DEDENT
                }
                if arms.is_empty() {
                    return Err(ParseError::new("match must have at least one arm", self.current_span()));
                }
                let end = self.current_span();
                Ok(Stmt {
                    kind: StmtKind::Match { scrutinee, arms },
                    span: Span::new(start.file, start.start as usize, end.start as usize),
                })
            }
            TokenKind::For => {
                self.advance(); // consume 'for'
                let (var, _) = self.expect_ident()?; // loop variable
                self.expect(&TokenKind::In)?;

                // `range` is syntactic sugar, not a runtime function.
                // Accepted forms: `range(end)` → start=0  and `range(start, end)`.
                let range_span = self.current_span();
                match self.current_kind().clone() {
                    TokenKind::Ident(ref name) if name == "range" => {
                        self.advance(); // consume 'range'
                    }
                    _ => {
                        return Err(ParseError::new(
                            format!(
                                "'for' loops must iterate over 'range(...)'; got {:?}",
                                self.current_kind()
                            ),
                            range_span,
                        ));
                    }
                }
                self.expect(&TokenKind::LParen)?;
                let first_arg = self.parse_expr()?;
                let (start_expr, end_expr) = if matches!(self.current_kind(), TokenKind::Comma) {
                    self.advance(); // consume ','
                    let second = self.parse_expr()?;
                    (first_arg, second)
                } else {
                    // range(n) → start = 0, end = n
                    let zero = Expr {
                        kind: ExprKind::Lit(Lit::Int(0)),
                        span: start, // synthetic span at the 'for' keyword
                    };
                    (zero, first_arg)
                };
                self.expect(&TokenKind::RParen)?;
                self.expect(&TokenKind::Colon)?;
                let body = self.parse_body()?;
                let end = self.current_span();
                Ok(Stmt {
                    kind: StmtKind::For { var, start: start_expr, end: end_expr, body },
                    span: Span::new(start.file, start.start as usize, end.start as usize),
                })
            }
            _ => {
                let expr = self.parse_expr()?;
                // Check for assignment: <ident> = <expr>
                if let ExprKind::Ident(target) = &expr.kind {
                    if self.current_kind() == &TokenKind::Eq {
                        let target = target.clone();
                        self.advance();
                        let rhs = self.parse_expr()?;
                        let end = rhs.span;
                        self.expect_newline_or_eof()?;
                        return Ok(Stmt {
                            kind: StmtKind::Assign { target, expr: rhs },
                            span: Span::new(start.file, start.start as usize, end.end as usize),
                        });
                    }
                }
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
                // Function call or struct/enum constructor: base(args)
                TokenKind::LParen => {
                    self.advance();
                    let mut args = Vec::new();
                    while !matches!(self.current_kind(), TokenKind::RParen | TokenKind::Eof) {
                        // Detect named arg: Ident followed immediately by `=` (not `==`).
                        let arg = if matches!(self.current_kind(), TokenKind::Ident(_))
                            && self.pos + 1 < self.tokens.len()
                            && matches!(self.tokens[self.pos + 1].kind, TokenKind::Eq)
                        {
                            let (name, _) = self.expect_ident()?;
                            self.advance(); // consume `=`
                            let value = self.parse_expr()?;
                            CallArg { name: Some(name), value }
                        } else {
                            let value = self.parse_expr()?;
                            CallArg { name: None, value }
                        };
                        args.push(arg);
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

                    base = Expr {
                        kind: ExprKind::FieldAccess { base: Box::new(base), field: name },
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

    // ── Import declarations ───────────────────────────────────────────────────

    fn parse_module_path(&mut self) -> Result<ModulePath, ParseError> {
        let start = self.current_span();
        let (first, _) = self.expect_ident()?;
        let mut segments = vec![first];
        while matches!(self.current_kind(), TokenKind::Dot) {
            self.advance(); // consume '.'
            let (seg, _) = self.expect_ident()?;
            segments.push(seg);
        }
        let end = self.current_span();
        Ok(ModulePath {
            segments,
            span: Span::new(start.file, start.start as usize, end.start as usize),
        })
    }

    fn parse_import(&mut self) -> Result<Item, ParseError> {
        let start = self.current_span();
        self.expect(&TokenKind::Import)?;
        let path = self.parse_module_path()?;
        let end = path.span;
        self.expect_newline_or_eof()?;
        Ok(Item {
            kind: ItemKind::Import { path },
            span: Span::new(start.file, start.start as usize, end.end as usize),
        })
    }

    fn parse_from_import(&mut self) -> Result<Item, ParseError> {
        let start = self.current_span();
        self.expect(&TokenKind::From)?;
        let path = self.parse_module_path()?;
        self.expect(&TokenKind::Import)?;
        // Parse `ident (',' ident)*`
        let mut names: Vec<(String, Span)> = Vec::new();
        let (first_name, first_span) = self.expect_ident()?;
        names.push((first_name, first_span));
        while matches!(self.current_kind(), TokenKind::Comma) {
            self.advance();
            let (name, span) = self.expect_ident()?;
            names.push((name, span));
        }
        let end = names.last().map(|(_, s)| *s).unwrap_or(path.span);
        self.expect_newline_or_eof()?;
        Ok(Item {
            kind: ItemKind::FromImport { path, names },
            span: Span::new(start.file, start.start as usize, end.end as usize),
        })
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

    fn parse_struct(&mut self) -> Result<Item, ParseError> {
        let start = self.current_span();
        self.expect(&TokenKind::Struct)?;
        let (name, _) = self.expect_ident()?;
        self.expect(&TokenKind::Colon)?;
        self.expect(&TokenKind::Newline)?;
        if !matches!(self.current_kind(), TokenKind::Indent) {
            return Err(ParseError::new("expected indented struct fields", self.current_span()));
        }
        self.advance(); // consume INDENT
        let mut fields = Vec::new();
        loop {
            self.skip_newlines();
            if matches!(self.current_kind(), TokenKind::Dedent | TokenKind::Eof) {
                break;
            }
            let field_start = self.current_span();
            let (fname, _) = self.expect_ident()?;
            self.expect(&TokenKind::Colon)?;
            let ty = self.parse_type()?;
            let field_end = self.current_span();
            self.expect_newline_or_eof()?;
            fields.push(FieldDef {
                name: fname,
                ty,
                span: Span::new(field_start.file, field_start.start as usize, field_end.start as usize),
            });
        }
        if matches!(self.current_kind(), TokenKind::Dedent) {
            self.advance(); // consume DEDENT
        }
        if fields.is_empty() {
            return Err(ParseError::new("struct must have at least one field", self.current_span()));
        }
        let end = self.current_span();
        Ok(Item {
            kind: ItemKind::Struct { name, fields },
            span: Span::new(start.file, start.start as usize, end.start as usize),
        })
    }

    fn parse_enum(&mut self) -> Result<Item, ParseError> {
        let start = self.current_span();
        self.expect(&TokenKind::Enum)?;
        let (name, _) = self.expect_ident()?;
        self.expect(&TokenKind::Colon)?;
        self.expect(&TokenKind::Newline)?;
        if !matches!(self.current_kind(), TokenKind::Indent) {
            return Err(ParseError::new("expected indented enum variants", self.current_span()));
        }
        self.advance(); // consume INDENT
        let mut variants = Vec::new();
        loop {
            self.skip_newlines();
            if matches!(self.current_kind(), TokenKind::Dedent | TokenKind::Eof) {
                break;
            }
            let var_start = self.current_span();
            let (vname, _) = self.expect_ident()?;
            // Optional data-carrying fields: Variant(name: Type, ...)
            let mut vfields = Vec::new();
            if matches!(self.current_kind(), TokenKind::LParen) {
                self.advance(); // consume '('
                while !matches!(self.current_kind(), TokenKind::RParen | TokenKind::Eof) {
                    let field_start = self.current_span();
                    let (fname, _) = self.expect_ident()?;
                    self.expect(&TokenKind::Colon)?;
                    let ty = self.parse_type()?;
                    let field_end = self.current_span();
                    vfields.push(FieldDef {
                        name: fname,
                        ty,
                        span: Span::new(field_start.file, field_start.start as usize, field_end.start as usize),
                    });
                    if matches!(self.current_kind(), TokenKind::Comma) {
                        self.advance();
                    } else {
                        break;
                    }
                }
                self.expect(&TokenKind::RParen)?;
            }
            let var_end = self.current_span();
            self.expect_newline_or_eof()?;
            variants.push(VariantDef {
                name: vname,
                fields: vfields,
                span: Span::new(var_start.file, var_start.start as usize, var_end.start as usize),
            });
        }
        if matches!(self.current_kind(), TokenKind::Dedent) {
            self.advance(); // consume DEDENT
        }
        if variants.is_empty() {
            return Err(ParseError::new("enum must have at least one variant", self.current_span()));
        }
        let end = self.current_span();
        Ok(Item {
            kind: ItemKind::Enum { name, variants },
            span: Span::new(start.file, start.start as usize, end.start as usize),
        })
    }

    fn parse_program(&mut self) -> Result<Program, ParseError> {
        let mut items = Vec::new();
        self.skip_newlines();

        // Phase 1: import declarations (must precede all fn/kernel definitions).
        while !self.at_end() {
            match self.current_kind() {
                TokenKind::Import => items.push(self.parse_import()?),
                TokenKind::From   => items.push(self.parse_from_import()?),
                _ => break,
            }
            self.skip_newlines();
        }

        // Phase 2: fn, kernel, struct, and enum definitions.
        while !self.at_end() {
            match self.current_kind() {
                TokenKind::Fn     => items.push(self.parse_fn()?),
                TokenKind::Kernel => items.push(self.parse_kernel()?),
                TokenKind::Struct => items.push(self.parse_struct()?),
                TokenKind::Enum   => items.push(self.parse_enum()?),
                TokenKind::Import | TokenKind::From => return Err(ParseError::new(
                    "import declarations must appear before function and kernel definitions",
                    self.current_span(),
                )),
                _ => return Err(ParseError::new(
                    format!("expected 'fn', 'kernel', 'struct', or 'enum', got {:?}", self.current_kind()),
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

    #[test]
    fn field_access_node() {
        let src = "fn f():\n    return a.b\n";
        let prog = parse_ok(src);
        let ItemKind::Fn { body, .. } = &prog.items[0].kind else { panic!() };
        let StmtKind::Return { expr } = &body[0].kind else { panic!() };
        assert!(matches!(&expr.kind, ExprKind::FieldAccess { field, .. } if field == "b"));
    }

    // ── Import parsing ────────────────────────────────────────────────────────

    #[test]
    fn parse_simple_import() {
        let src = "import ops\n\nfn main():\n    return 0\n";
        let prog = parse_ok(src);
        assert_eq!(prog.items.len(), 2);
        match &prog.items[0].kind {
            ItemKind::Import { path } => assert_eq!(path.segments, vec!["ops"]),
            _ => panic!("expected Import"),
        }
        assert!(matches!(prog.items[1].kind, ItemKind::Fn { .. }));
    }

    #[test]
    fn parse_from_import_single() {
        let src = "from ops import add\n\nfn main():\n    return 0\n";
        let prog = parse_ok(src);
        match &prog.items[0].kind {
            ItemKind::FromImport { path, names } => {
                assert_eq!(path.segments, vec!["ops"]);
                assert_eq!(names.len(), 1);
                assert_eq!(names[0].0, "add");
            }
            _ => panic!("expected FromImport"),
        }
    }

    #[test]
    fn parse_from_import_multiple_names() {
        let src = "from ops import add, mul, sub\n\nfn main():\n    return 0\n";
        let prog = parse_ok(src);
        match &prog.items[0].kind {
            ItemKind::FromImport { names, .. } => {
                assert_eq!(names.len(), 3);
                assert_eq!(names[0].0, "add");
                assert_eq!(names[1].0, "mul");
                assert_eq!(names[2].0, "sub");
            }
            _ => panic!("expected FromImport"),
        }
    }

    #[test]
    fn parse_dotted_import() {
        let src = "import models.transformer\n\nfn main():\n    return 0\n";
        let prog = parse_ok(src);
        match &prog.items[0].kind {
            ItemKind::Import { path } => {
                assert_eq!(path.segments, vec!["models", "transformer"]);
                assert_eq!(path.name(), "transformer");
            }
            _ => panic!("expected Import"),
        }
    }

    #[test]
    fn parse_multiple_imports() {
        let src = "import ops\nfrom utils import helper\n\nfn main():\n    return 0\n";
        let prog = parse_ok(src);
        assert_eq!(prog.items.len(), 3);
        assert!(matches!(prog.items[0].kind, ItemKind::Import { .. }));
        assert!(matches!(prog.items[1].kind, ItemKind::FromImport { .. }));
        assert!(matches!(prog.items[2].kind, ItemKind::Fn { .. }));
    }

    #[test]
    fn parse_import_only_file() {
        let src = "import ops\n";
        let prog = parse_ok(src);
        assert_eq!(prog.items.len(), 1);
        assert!(matches!(prog.items[0].kind, ItemKind::Import { .. }));
    }

    #[test]
    fn parse_deep_dotted_import() {
        let src = "import a.b.c.d\n";
        let prog = parse_ok(src);
        match &prog.items[0].kind {
            ItemKind::Import { path } => assert_eq!(path.segments, vec!["a", "b", "c", "d"]),
            _ => panic!(),
        }
    }

    #[test]
    fn import_after_fn_is_error() {
        let src = "fn f():\n    return 0\n\nimport ops\n";
        let err = parse(FileId(0), src).unwrap_err();
        assert!(err.message.contains("import declarations must appear before"));
    }

    #[test]
    fn from_import_after_fn_is_error() {
        let src = "fn f():\n    return 0\n\nfrom ops import add\n";
        let err = parse(FileId(0), src).unwrap_err();
        assert!(err.message.contains("import declarations must appear before"));
    }

    // ── M9: if / while / for parsing ─────────────────────────────────────────

    #[test]
    fn parse_if_no_else() {
        let src = "fn f():\n    if x > 0:\n        print(x)\n";
        let prog = parse_ok(src);
        let ItemKind::Fn { body, .. } = &prog.items[0].kind else { panic!() };
        let StmtKind::If { then_body, else_body, .. } = &body[0].kind else {
            panic!("expected If, got {:?}", body[0].kind)
        };
        assert_eq!(then_body.len(), 1);
        assert!(else_body.is_none());
    }

    #[test]
    fn parse_if_else() {
        let src = "fn f():\n    if x > 0:\n        print(x)\n    else:\n        print(y)\n";
        let prog = parse_ok(src);
        let ItemKind::Fn { body, .. } = &prog.items[0].kind else { panic!() };
        let StmtKind::If { then_body, else_body, .. } = &body[0].kind else { panic!() };
        assert_eq!(then_body.len(), 1);
        assert!(else_body.as_ref().unwrap().len() == 1);
    }

    #[test]
    fn parse_for_range_one_arg() {
        let src = "fn f():\n    for i in range(10):\n        print(i)\n";
        let prog = parse_ok(src);
        let ItemKind::Fn { body, .. } = &prog.items[0].kind else { panic!() };
        let StmtKind::For { var, start, end, body } = &body[0].kind else {
            panic!("expected For, got {:?}", body[0].kind)
        };
        assert_eq!(var, "i");
        // start should be literal 0
        assert!(matches!(start.kind, ExprKind::Lit(Lit::Int(0))));
        // end should be literal 10
        assert!(matches!(end.kind, ExprKind::Lit(Lit::Int(10))));
        assert_eq!(body.len(), 1);
    }

    #[test]
    fn parse_for_range_two_args() {
        let src = "fn f():\n    for i in range(a, b):\n        print(i)\n";
        let prog = parse_ok(src);
        let ItemKind::Fn { body, .. } = &prog.items[0].kind else { panic!() };
        let StmtKind::For { var, start, end, .. } = &body[0].kind else { panic!() };
        assert_eq!(var, "i");
        assert!(matches!(&start.kind, ExprKind::Ident(n) if n == "a"));
        assert!(matches!(&end.kind, ExprKind::Ident(n) if n == "b"));
    }

    #[test]
    fn parse_while() {
        let src = "fn f():\n    while x > 0:\n        print(x)\n";
        let prog = parse_ok(src);
        let ItemKind::Fn { body, .. } = &prog.items[0].kind else { panic!() };
        let StmtKind::While { body: inner, .. } = &body[0].kind else {
            panic!("expected While, got {:?}", body[0].kind)
        };
        assert_eq!(inner.len(), 1);
    }

    #[test]
    fn parse_nested_for_and_if() {
        let src = concat!(
            "fn f():\n",
            "    for i in range(5):\n",
            "        let x = i\n",
            "        if i > 2:\n",
            "            print(i)\n",
        );
        let prog = parse_ok(src);
        let ItemKind::Fn { body, .. } = &prog.items[0].kind else { panic!() };
        let StmtKind::For { body: for_body, .. } = &body[0].kind else { panic!() };
        // for body: let x, if
        assert_eq!(for_body.len(), 2);
        assert!(matches!(&for_body[0].kind, StmtKind::Let { .. }));
        assert!(matches!(&for_body[1].kind, StmtKind::If { .. }));
    }

    #[test]
    fn parse_for_non_range_is_error() {
        let src = "fn f():\n    for i in mylist:\n        print(i)\n";
        let err = parse(FileId(0), src).unwrap_err();
        assert!(err.message.contains("range"), "error was: {}", err.message);
    }
}
