// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Parser for prometheus_names.rs to extract constants and modules

use anyhow::{Context, Result};
use std::collections::HashMap;
use syn::{File, Item, ItemConst, ItemMacro, ItemMod};

#[derive(Debug, Clone)]
pub struct ConstantDef {
    pub name: String,
    pub value: String,
    pub doc_comment: String,
}

#[derive(Debug, Clone)]
pub struct ModuleDef {
    pub name: String,
    pub constants: Vec<ConstantDef>,
    pub doc_comment: String,
    pub is_macro_generated: bool,
    pub macro_prefix: Option<String>,
    /// Nested `pub mod` blocks, sorted by name so generated output is stable.
    /// `transport::tcp` and the `frontend_service` label-value modules live here.
    pub submodules: Vec<ModuleDef>,
}

pub struct PrometheusParser {
    pub modules: HashMap<String, ModuleDef>,
}

impl PrometheusParser {
    pub fn parse_file(content: &str) -> Result<Self> {
        let ast: File = syn::parse_str(content).context("Failed to parse Rust file")?;

        let mut modules = HashMap::new();

        for item in ast.items {
            if let Item::Mod(module) = item {
                if let Some(parsed_module) = Self::parse_module(&module)? {
                    modules.insert(parsed_module.name.clone(), parsed_module);
                }
            }
        }

        Ok(Self { modules })
    }

    fn parse_module(module: &ItemMod) -> Result<Option<ModuleDef>> {
        // Only process public modules
        if !matches!(module.vis, syn::Visibility::Public(_)) {
            return Ok(None);
        }

        let module_name = module.ident.to_string();
        let doc_comment = Self::extract_doc_comment(&module.attrs);

        let (_, items) = match &module.content {
            Some(content) => content,
            None => return Ok(None),
        };

        let mut constants = Vec::new();
        let mut submodules: Vec<ModuleDef> = Vec::new();
        let mut is_macro_generated = false;
        let mut macro_prefix = None;

        for item in items {
            match item {
                Item::Const(const_item) => {
                    if let Some(const_def) = Self::parse_const(const_item)? {
                        constants.push(const_def);
                    }
                }
                Item::Macro(macro_item) => {
                    // Check if this is a macro_rules! that generates names with a prefix
                    if let Some(prefix) = Self::extract_macro_prefix(macro_item) {
                        is_macro_generated = true;
                        macro_prefix = Some(prefix);
                    }
                }
                Item::Mod(nested) => {
                    // Recurse so `transport::tcp` reaches the generated Python as a
                    // nested class. Skipping these used to emit `class transport:` with
                    // an empty body, which parses fine and silently loses every name.
                    if let Some(child) = Self::parse_module(nested)? {
                        submodules.push(child);
                    }
                }
                _ => {}
            }
        }

        // Deterministic output: the caller sorts top-level modules, so sort children too.
        submodules.sort_by(|a, b| a.name.cmp(&b.name));

        // Rust keeps consts and mods in separate namespaces; Python class attributes are
        // one namespace. `pub const status` next to `pub mod status` compiles in Rust and
        // would emit two `status` members in the same class body, where the nested class
        // silently wins because it is written last. Fail the build instead — a silently
        // shadowed metric name is the exact failure mode this recursion was added to end.
        for child in &submodules {
            if let Some(clash) = constants.iter().find(|c| c.name == child.name) {
                anyhow::bail!(
                    "module `{module_name}` declares both a constant and a nested module named \
                     `{}`; Python cannot express both as attributes of one class. Rename one of \
                     them (the constant currently resolves to \"{}\").",
                    child.name,
                    clash.value
                );
            }
        }

        // Apply macro prefix to constants if needed. Deliberately NOT applied to
        // submodules: the nested modules hold label *values* (`operation::TOKENIZE`,
        // `error_type::VALIDATION`), not metric names, so prefixing them would corrupt
        // the value a caller compares against.
        if is_macro_generated {
            if let Some(prefix) = macro_prefix.as_ref() {
                for constant in &mut constants {
                    // Only apply if the constant doesn't already have the prefix
                    if constant.name == "PREFIX" {
                        // PREFIX constant should be just the prefix with trailing underscore
                        continue;
                    }
                    // Check if value looks like it should have prefix applied
                    // (doesn't already start with the prefix)
                    if !constant.value.starts_with(prefix) {
                        constant.value = format!("{}_{}", prefix, constant.value);
                    }
                }
            }
        }

        Ok(Some(ModuleDef {
            name: module_name,
            constants,
            doc_comment,
            is_macro_generated,
            macro_prefix,
            submodules,
        }))
    }

    fn parse_const(const_item: &ItemConst) -> Result<Option<ConstantDef>> {
        // Only process public constants
        if !matches!(const_item.vis, syn::Visibility::Public(_)) {
            return Ok(None);
        }

        // Only process &str constants
        let is_str_type = matches!(&*const_item.ty, syn::Type::Reference(type_ref)
            if matches!(&*type_ref.elem, syn::Type::Path(path)
                if path.path.segments.last().map(|s| s.ident == "str").unwrap_or(false)));

        if !is_str_type {
            return Ok(None);
        }

        let name = const_item.ident.to_string();
        let doc_comment = Self::extract_doc_comment(&const_item.attrs);

        // Extract the string value
        let value = Self::extract_string_value(&const_item.expr)?;

        Ok(Some(ConstantDef {
            name,
            value,
            doc_comment,
        }))
    }

    fn extract_string_value(expr: &syn::Expr) -> Result<String> {
        match expr {
            // Direct string literal: "value"
            syn::Expr::Lit(lit_expr) => {
                if let syn::Lit::Str(lit_str) = &lit_expr.lit {
                    Ok(lit_str.value())
                } else {
                    anyhow::bail!("Expected string literal")
                }
            }
            // Macro invocation: some_macro!("value")
            syn::Expr::Macro(macro_expr) => {
                // Try to extract the string from macro arguments
                Self::extract_from_macro_tokens(&macro_expr.mac.tokens)
            }
            // Method call: "value".to_string()
            syn::Expr::MethodCall(method_call) => Self::extract_string_value(&method_call.receiver),
            _ => anyhow::bail!("Unsupported expression type for constant value"),
        }
    }

    fn extract_from_macro_tokens(tokens: &proc_macro2::TokenStream) -> Result<String> {
        // Parse the tokens to find string literals
        let tokens_str = tokens.to_string();

        // Look for string literals in the token stream
        // This handles cases like: concat!("prefix_", "value")
        let parts: Vec<&str> = tokens_str
            .split('"')
            .enumerate()
            .filter(|(i, _)| i % 2 == 1)
            .map(|(_, s)| s)
            .collect();

        if parts.is_empty() {
            anyhow::bail!("No string literals found in macro");
        }

        // Concatenate all string parts (for concat! macro)
        Ok(parts.join(""))
    }

    fn extract_macro_prefix(macro_item: &ItemMacro) -> Option<String> {
        // Check if this is a macro_rules! with a name ending in "_name"
        let macro_name = macro_item.ident.as_ref()?.to_string();
        if !macro_name.ends_with("_name") {
            return None;
        }

        // Try to extract the prefix from the macro body
        // Looking for patterns like: concat!("prefix_", $name)
        let tokens_str = macro_item.mac.tokens.to_string();

        // Look for concat! with a string literal
        // Pattern: concat ! ( "prefix_" , ...
        if let Some(concat_start) = tokens_str.find("concat !") {
            let after_concat = &tokens_str[concat_start..];
            // Find the first string literal after concat!
            if let Some(quote_start) = after_concat.find('"') {
                let after_quote = &after_concat[quote_start + 1..];
                if let Some(quote_end) = after_quote.find('"') {
                    let prefix = &after_quote[..quote_end];
                    // Remove trailing underscore if present
                    return Some(prefix.trim_end_matches('_').to_string());
                }
            }
        }

        None
    }

    fn extract_doc_comment(attrs: &[syn::Attribute]) -> String {
        let mut doc_lines = Vec::new();

        for attr in attrs {
            if attr.path().is_ident("doc") {
                if let syn::Meta::NameValue(meta) = &attr.meta {
                    if let syn::Expr::Lit(lit) = &meta.value {
                        if let syn::Lit::Str(lit_str) = &lit.lit {
                            let line = lit_str.value().trim().to_string();
                            if !line.is_empty() {
                                doc_lines.push(line);
                            }
                        }
                    }
                }
            }
        }

        doc_lines.join("\n")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Mirrors the real `transport` module: a parent whose constants all live in nested
    /// `pub mod` blocks. Before recursion landed, this parsed to a parent with zero
    /// constants and no children, which the generator turned into an empty Python class.
    const NESTED_SOURCE: &str = r#"
/// Transport-specific metrics (TCP / NATS)
pub mod transport {
    pub mod tcp {
        pub const BYTES_SENT_TOTAL: &str = "tcp_bytes_sent_total";
        pub const ERRORS_TOTAL: &str = "tcp_errors_total";
    }
    pub mod nats {
        pub const ERRORS_TOTAL: &str = "nats_errors_total";
    }
}
"#;

    fn parse(src: &str) -> PrometheusParser {
        PrometheusParser::parse_file(src).expect("source should parse")
    }

    fn child<'a>(module: &'a ModuleDef, name: &str) -> &'a ModuleDef {
        module
            .submodules
            .iter()
            .find(|m| m.name == name)
            .unwrap_or_else(|| panic!("expected submodule `{name}`"))
    }

    /// The regression this whole change exists for: nested modules must survive parsing
    /// with their constants attached. Reverting the `Item::Mod` arm leaves `submodules`
    /// empty and fails the first assertion.
    #[test]
    fn nested_modules_are_parsed_with_their_constants() {
        let parser = parse(NESTED_SOURCE);
        let transport = &parser.modules["transport"];

        assert_eq!(
            transport.submodules.len(),
            2,
            "transport must carry both nested modules, got {:?}",
            transport
                .submodules
                .iter()
                .map(|m| &m.name)
                .collect::<Vec<_>>()
        );

        let tcp = child(transport, "tcp");
        assert_eq!(
            tcp.constants
                .iter()
                .map(|c| (c.name.as_str(), c.value.as_str()))
                .collect::<Vec<_>>(),
            vec![
                ("BYTES_SENT_TOTAL", "tcp_bytes_sent_total"),
                ("ERRORS_TOTAL", "tcp_errors_total"),
            ]
        );
        assert_eq!(child(transport, "nats").constants.len(), 1);
    }

    /// Submodule order must not depend on source order, or the generated Python file
    /// churns whenever someone reorders the Rust modules.
    #[test]
    fn submodules_are_sorted_by_name() {
        let parser = parse(NESTED_SOURCE);
        let names: Vec<&str> = parser.modules["transport"]
            .submodules
            .iter()
            .map(|m| m.name.as_str())
            .collect();
        assert_eq!(
            names,
            vec!["nats", "tcp"],
            "source declares tcp before nats; output must still be sorted"
        );
    }

    /// A nested module holding label *values* must keep those values verbatim. This is
    /// the shape of `frontend_service::error_type`, where `NONE` is deliberately the
    /// empty string and any prefixing would corrupt what callers compare against.
    #[test]
    fn nested_label_value_modules_keep_raw_values() {
        let parser = parse(
            r#"
pub mod frontend_service {
    pub const REQUESTS_TOTAL: &str = "requests_total";
    /// Error type label values
    pub mod error_type {
        pub const NONE: &str = "";
        pub const VALIDATION: &str = "validation";
    }
}
"#,
        );
        let frontend = &parser.modules["frontend_service"];
        assert_eq!(frontend.constants.len(), 1, "parent constants still parse");

        let error_type = child(frontend, "error_type");
        assert_eq!(error_type.doc_comment, "Error type label values");
        assert_eq!(
            error_type
                .constants
                .iter()
                .map(|c| (c.name.as_str(), c.value.as_str()))
                .collect::<Vec<_>>(),
            vec![("NONE", ""), ("VALIDATION", "validation")]
        );
    }

    /// Recursion is not depth-limited to one level, and a private nested module stays
    /// out of the public Python surface.
    #[test]
    fn recursion_is_deep_and_skips_private_modules() {
        let parser = parse(
            r#"
pub mod outer {
    pub mod middle {
        pub mod inner {
            pub const LEAF: &str = "leaf_value";
        }
    }
    mod private_mod {
        pub const HIDDEN: &str = "hidden_value";
    }
}
"#,
        );
        let outer = &parser.modules["outer"];
        assert_eq!(
            outer.submodules.len(),
            1,
            "the private module must not be exported"
        );
        let inner = child(child(outer, "middle"), "inner");
        assert_eq!(inner.constants[0].value, "leaf_value");
    }

    /// Rust allows a const and a mod to share a name; Python cannot. The generator must
    /// refuse rather than emit two `status` attributes where the nested class silently
    /// overwrites the constant.
    #[test]
    fn const_and_submodule_name_collision_is_rejected() {
        let err = PrometheusParser::parse_file(
            r#"
pub mod frontend_service {
    pub const status: &str = "status_value";
    pub mod status {
        pub const SUCCESS: &str = "success";
    }
}
"#,
        )
        .err()
        .expect("a const/mod name collision must not parse silently");

        let msg = err.to_string();
        assert!(
            msg.contains("frontend_service") && msg.contains("`status`"),
            "error must name the module and the clashing symbol, got: {msg}"
        );
    }

    /// Top-level modules that never had nesting must be unaffected — this guards the
    /// flat namespace the Python file already exposes.
    #[test]
    fn flat_modules_are_unchanged() {
        let parser = parse(
            r#"
pub mod kvrouter {
    pub const KV_CACHE_EVENTS_APPLIED: &str = "kv_cache_events_applied";
}
"#,
        );
        let kvrouter = &parser.modules["kvrouter"];
        assert!(kvrouter.submodules.is_empty());
        assert_eq!(kvrouter.constants.len(), 1);
    }
}
