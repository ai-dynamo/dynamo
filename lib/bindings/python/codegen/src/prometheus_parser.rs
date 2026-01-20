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
                _ => {}
            }
        }

        // Apply macro prefix to constants if needed
        if is_macro_generated && macro_prefix.is_some() {
            let prefix = macro_prefix.as_ref().unwrap();
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

        Ok(Some(ModuleDef {
            name: module_name,
            constants,
            doc_comment,
            is_macro_generated,
            macro_prefix,
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
