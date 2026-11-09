// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! When the Unified `mistral` family installs its grammar.
//!
//! The grammar is mistral-common's, which upstream vLLM installs for v11+ tekken
//! tokenizers: it forces the call for forced choices, keeps every call's
//! arguments a JSON object under `auto`, and carries a JSON schema for the
//! reply, while keeping the `[TOOL_CALLS]name[ARGS]{json}` format the Unified
//! parser reads. frontend-crates renders it beside that parser
//! (`dynamo_parsers_v2::unified::mistral::ForcedToolGrammar`). It applies when
//! the Unified family serves the model, which model setup checks, the worker
//! compiles llguidance lark grammars, and the tokenizer has the call special
//! tokens.

use std::cell::RefCell;

use anyhow::Context as _;
use dynamo_parsers::tool_calling::ToolDefinition;
use dynamo_parsers_v2::unified::mistral::ForcedToolGrammar;
use serde_json::Value;

pub(crate) use dynamo_parsers_v2::unified::mistral::CallMode;

use crate::local_model::runtime_config::{LLGUIDANCE_LARK_GRAMMAR_RUNTIME_KEY, ModelRuntimeConfig};
use crate::tokenizers::traits::Tokenizer;

/// The tool-call grammar for one model, resolved at setup.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct MistralToolCallGrammar(ForcedToolGrammar);

impl MistralToolCallGrammar {
    /// Resolve the grammar for a model, or why tool calls must run without one.
    pub(crate) fn for_model(
        runtime_config: &ModelRuntimeConfig,
        tokenizer: &dyn Tokenizer,
    ) -> anyhow::Result<Self> {
        let lark = runtime_config
            .get_engine_specific::<bool>(LLGUIDANCE_LARK_GRAMMAR_RUNTIME_KEY)
            .with_context(|| format!("invalid {LLGUIDANCE_LARK_GRAMMAR_RUNTIME_KEY}"))?;
        anyhow::ensure!(
            lark == Some(true),
            "the backend does not advertise llguidance lark grammars"
        );
        let special = tokenizer.special_token_ids()?;
        // A backend that cannot look tokens up reports that, not a missing marker.
        let lookup_error = RefCell::new(None);
        let grammar = ForcedToolGrammar::resolve(|token| match tokenizer.token_to_id(token) {
            Ok(id) => id.filter(|id| special.contains(id)),
            Err(error) => {
                lookup_error.borrow_mut().get_or_insert(error);
                None
            }
        });
        if let Some(error) = lookup_error.into_inner() {
            return Err(error);
        }
        grammar
            .map(Self)
            .context("the tokenizer has no [TOOL_CALLS] and [ARGS] special tokens")
    }

    /// The grammar for one `tool_choice` mode. `parallel_tool_calls` allows one
    /// or more calls; otherwise one. A `reply_schema` also allows a reply that
    /// is only JSON matching it. Upstream vLLM also constrains `none` and
    /// tool-less requests without a reply schema; those stay unconstrained here.
    pub(crate) fn render(
        &self,
        mode: CallMode<'_>,
        tools: Vec<ToolDefinition>,
        parallel_tool_calls: bool,
        reply_schema: Option<&Value>,
    ) -> String {
        let tools = tools
            .into_iter()
            .map(|tool| dynamo_parsers_v2::Tool {
                name: tool.name,
                description: None,
                parameters: tool.parameters.unwrap_or(Value::Null),
                // Copied as is: `structural_tag/v2.rs` `to_v2_tool` reads an
                // omitted `strict` as true, which would change this grammar.
                strict: tool.strict,
            })
            .collect::<Vec<_>>();
        self.0
            .render(mode, &tools, parallel_tool_calls, reply_schema)
    }

    /// The grammar for a reply that is only JSON matching `reply_schema`, for
    /// `none` and tool-less requests.
    pub(crate) fn render_json(&self, reply_schema: &Value) -> String {
        self.0.render_json(reply_schema)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn resolves_markers_only_for_llguidance_backends() {
        let path = concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/tests/data/sample-models/mock-mistral-tekken/tekken.json"
        );
        let tokenizer = crate::tokenizers::TekkenTokenizer::from_file(path).unwrap();
        let mut runtime_config = ModelRuntimeConfig::default();
        assert!(MistralToolCallGrammar::for_model(&runtime_config, &tokenizer).is_err());

        runtime_config
            .set_engine_specific(LLGUIDANCE_LARK_GRAMMAR_RUNTIME_KEY, true)
            .unwrap();
        let grammar = MistralToolCallGrammar::for_model(&runtime_config, &tokenizer).unwrap();
        let tool = ToolDefinition {
            name: "get_weather".to_string(),
            parameters: Some(serde_json::json!({"type": "object", "required": ["city"]})),
            strict: Some(true),
        };
        let rendered = grammar.render(CallMode::Required, vec![tool], false, None);
        assert!(
            rendered.contains("think: <[34]> content <[35]>")
                && rendered.contains(
                    r#"<[9]> SAFE_WS? "get_weather" <[32]> SAFE_WS? %json {"type":"object","required":["city"]}"#
                ),
            "{rendered}"
        );
    }
}
