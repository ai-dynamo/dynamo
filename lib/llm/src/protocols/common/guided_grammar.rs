// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Conservative check that a `guided_grammar` start rule can produce a finite string.
//!
//! The scanner understands the GBNF / xgrammar EBNF subset: `name ::= body` rules, `|`
//! alternatives, quoted literals, `[...]` character classes, `(...)` groups, the `*`,
//! `+`, `?` and `{m,n}` quantifiers, and `#` comments. On any other syntax it gives no
//! verdict and the grammar goes to the engine unchanged, so the check can miss a bad
//! grammar but never rejects a good one.

use std::collections::HashMap;

use super::MAX_GRAMMAR_NESTING_DEPTH;

/// Start rule name used by xgrammar and the engines.
pub(super) const ROOT_RULE: &str = "root";

/// True only when the grammar defines `root` and `root` cannot derive any finite string,
/// such as `root ::= root root`. Engines accept such a grammar and then fail on the first
/// token (SGLang aborts the request with HTTP 500), so it is rejected up front.
pub(super) fn root_cannot_terminate(grammar: &str) -> bool {
    let Some(tokens) = tokenize(grammar) else {
        return false;
    };
    let Some(parsed) = Parser::parse(&tokens) else {
        return false;
    };
    let Some(&root) = parsed.rules.get(ROOT_RULE) else {
        return false;
    };
    !productive_symbols(parsed.symbols, &parsed.productions)[root]
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Token<'a> {
    Ident(&'a str),
    Assign,
    /// A literal or character class.
    Terminal,
    Open,
    Close,
    Pipe,
    /// `*`, `?` or `{0,n}`: the item may be absent.
    Optional,
    /// `+` or `{m,n}` with `m >= 1`: the item must appear.
    Required,
}

/// Gives no verdict when `(` groups nest deeper than `MAX_GRAMMAR_NESTING_DEPTH`. That
/// bounds the parser's recursion; the caller's byte-level nesting check does not, because
/// brackets inside literals lower its count.
fn tokenize(grammar: &str) -> Option<Vec<Token<'_>>> {
    let bytes = grammar.as_bytes();
    let mut tokens = Vec::new();
    let mut group_depth: usize = 0;
    let mut i = 0;
    while i < bytes.len() {
        let token = match bytes[i] {
            b' ' | b'\t' | b'\r' | b'\n' => {
                i += 1;
                continue;
            }
            b'#' => {
                while i < bytes.len() && bytes[i] != b'\n' {
                    i += 1;
                }
                continue;
            }
            quote @ (b'"' | b'\'') => {
                i = skip_delimited(bytes, i + 1, quote)?;
                Token::Terminal
            }
            b'[' => {
                i = skip_delimited(bytes, i + 1, b']')?;
                Token::Terminal
            }
            b'{' => {
                let end = i + bytes[i..].iter().position(|&b| b == b'}')?;
                let bounds = grammar[i + 1..end].split(',').collect::<Vec<_>>();
                if bounds.len() > 2
                    || !bounds
                        .iter()
                        .all(|b| b.trim().bytes().all(|c| c.is_ascii_digit()))
                {
                    return None;
                }
                let min = bounds[0].trim();
                i = end + 1;
                if min.is_empty() || min.bytes().all(|c| c == b'0') {
                    Token::Optional
                } else {
                    Token::Required
                }
            }
            b':' if bytes[i..].starts_with(b"::=") => {
                i += 3;
                Token::Assign
            }
            b'(' | b')' | b'|' | b'*' | b'?' | b'+' => {
                i += 1;
                match bytes[i - 1] {
                    b'(' => {
                        group_depth += 1;
                        if group_depth > MAX_GRAMMAR_NESTING_DEPTH {
                            return None;
                        }
                        Token::Open
                    }
                    b')' => {
                        group_depth = group_depth.saturating_sub(1);
                        Token::Close
                    }
                    b'|' => Token::Pipe,
                    b'+' => Token::Required,
                    _ => Token::Optional,
                }
            }
            c if c.is_ascii_alphabetic() || c == b'_' => {
                let start = i;
                while i < bytes.len()
                    && (bytes[i].is_ascii_alphanumeric() || bytes[i] == b'_' || bytes[i] == b'-')
                {
                    i += 1;
                }
                Token::Ident(&grammar[start..i])
            }
            // Lookahead `(= ...)`, Lark `:`, regex `/.../`, and anything else unknown.
            _ => return None,
        };
        tokens.push(token);
    }
    Some(tokens)
}

/// Index just past the closing delimiter, skipping backslash escapes.
fn skip_delimited(bytes: &[u8], mut i: usize, close: u8) -> Option<usize> {
    while i < bytes.len() {
        match bytes[i] {
            b'\\' => i += 2,
            c if c == close => return Some(i + 1),
            _ => i += 1,
        }
    }
    None
}

/// The grammar as plain BNF over symbol ids: rules first, then one symbol per group.
/// Each production lists only the symbols that must be productive for it to be;
/// terminals, optional items, and references to undefined rules are dropped.
struct Parser<'t, 'a> {
    tokens: &'t [Token<'a>],
    pos: usize,
    rules: HashMap<&'a str, usize>,
    symbols: usize,
    productions: Vec<(usize, Vec<usize>)>,
}

impl<'t, 'a> Parser<'t, 'a> {
    /// Index rule names before reading bodies so forward references can be resolved.
    /// Duplicate definitions or unsupported syntax give no verdict.
    fn parse(tokens: &'t [Token<'a>]) -> Option<Self> {
        let mut rules = HashMap::new();
        for pair in tokens.windows(2) {
            if let [Token::Ident(name), Token::Assign] = pair {
                let id = rules.len();
                if rules.insert(*name, id).is_some() {
                    return None;
                }
            }
        }
        let mut parser = Self {
            tokens,
            pos: 0,
            symbols: rules.len(),
            rules,
            productions: Vec::new(),
        };
        while parser.pos < tokens.len() {
            let Token::Ident(name) = tokens[parser.pos] else {
                return None;
            };
            if parser.peek_at(1) != Some(Token::Assign) {
                return None;
            }
            parser.pos += 2;
            let lhs = parser.rules[name];
            parser.alternatives(lhs)?;
        }
        Some(parser)
    }

    fn peek_at(&self, offset: usize) -> Option<Token<'a>> {
        self.tokens.get(self.pos + offset).copied()
    }

    /// Record each alternative separately; any one can make `lhs` productive.
    fn alternatives(&mut self, lhs: usize) -> Option<()> {
        loop {
            let required = self.sequence()?;
            self.productions.push((lhs, required));
            if self.peek_at(0) != Some(Token::Pipe) {
                return Some(());
            }
            self.pos += 1;
        }
    }

    /// Recursion depth is bounded by the group depth limit in `tokenize`.
    fn sequence(&mut self) -> Option<Vec<usize>> {
        let mut required = Vec::new();
        loop {
            let item = match self.peek_at(0) {
                Some(Token::Ident(name)) if self.peek_at(1) != Some(Token::Assign) => {
                    self.pos += 1;
                    // Undefined rules count as productive; the engine reports them.
                    self.rules.get(name).copied()
                }
                Some(Token::Terminal) => {
                    self.pos += 1;
                    None
                }
                Some(Token::Open) => {
                    self.pos += 1;
                    let group = self.symbols;
                    self.symbols += 1;
                    self.alternatives(group)?;
                    if self.peek_at(0) != Some(Token::Close) {
                        return None;
                    }
                    self.pos += 1;
                    Some(group)
                }
                // `|`, `)`, the next rule, or the end of input.
                _ => return Some(required),
            };
            let mut optional = false;
            while let Some(quantifier @ (Token::Optional | Token::Required)) = self.peek_at(0) {
                self.pos += 1;
                optional |= quantifier == Token::Optional;
            }
            if !optional {
                required.extend(item);
            }
        }
    }
}

/// Standard counter-and-worklist productivity: linear in grammar size, because grammar
/// text is client-controlled and a fixed-point loop is quadratic on long rule chains.
fn productive_symbols(symbols: usize, productions: &[(usize, Vec<usize>)]) -> Vec<bool> {
    let mut productive = vec![false; symbols];
    let mut uses = vec![Vec::new(); symbols];
    let mut pending = Vec::with_capacity(productions.len());
    let mut worklist = Vec::new();
    for (production, (lhs, required)) in productions.iter().enumerate() {
        for &symbol in required {
            uses[symbol].push(production);
        }
        pending.push(required.len());
        if required.is_empty() && !productive[*lhs] {
            productive[*lhs] = true;
            worklist.push(*lhs);
        }
    }
    while let Some(symbol) = worklist.pop() {
        for &production in &uses[symbol] {
            pending[production] -= 1;
            let lhs = productions[production].0;
            if pending[production] == 0 && !productive[lhs] {
                productive[lhs] = true;
                worklist.push(lhs);
            }
        }
    }
    productive
}
