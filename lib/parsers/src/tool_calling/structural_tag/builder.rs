// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Public structural tag builder API used by the Dynamo preprocessor.

use dynamo_protocols::types::{ChatCompletionTool, ChatCompletionToolChoiceOption};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};

use super::dsml::{self, DsmlToolCallsConfig};
use super::format::{AnyTokensFormat, Format, StructuralTag, TagFormat};
use super::triggered_tags::{self, TriggeredTagsConfig};

/// Controls whether tools get their real parameter schema or an
/// unconstrained one inside structural tags.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, Default)]
#[serde(rename_all = "snake_case")]
pub enum StructuralTagSchemaMode {
    /// Real schema only for tools with `strict: true`; others get an
    /// unconstrained schema (`true` in xgrammar).
    #[default]
    Auto,
    /// Real parameter schema for all tools regardless of `strict` flag.
    Strict,
}

/// Request-scoped inputs for building a tool-call structural tag.
#[derive(Debug, Clone, Copy)]
pub struct ToolCallFormatBuildContext<'a> {
    /// Resolved `tool_choice` from the request.
    pub tool_choice: &'a ChatCompletionToolChoiceOption,
    /// All tools from the request.
    pub tools: &'a [ChatCompletionTool],
    /// From the request; `Some(false)` sets `stop_after_first` in the tag.
    pub parallel_tool_calls: Option<bool>,
    /// Schema strictness mode for tool arguments.
    pub schema_mode: StructuralTagSchemaMode,
}

impl ToolCallFormatBuildContext<'_> {
    /// Whether we should stop after the first matched tool-call tag.
    pub(crate) fn stop_after_first(&self) -> bool {
        self.parallel_tool_calls.is_some_and(|v| !v)
    }

    /// Whether all tools should use their request-provided parameter schema.
    pub(crate) fn strict_schema(&self) -> bool {
        self.schema_mode == StructuralTagSchemaMode::Strict
    }
}

/// Select tools for `tool_choice` and whether at least one call is required.
pub(crate) fn resolve_tools_to_include<'a>(
    ctx: &ToolCallFormatBuildContext<'a>,
) -> anyhow::Result<(Vec<&'a ChatCompletionTool>, bool)> {
    match ctx.tool_choice {
        ChatCompletionToolChoiceOption::None => Ok((vec![], false)),
        ChatCompletionToolChoiceOption::Auto => Ok((ctx.tools.iter().collect(), false)),
        ChatCompletionToolChoiceOption::Required => {
            anyhow::ensure!(
                !ctx.tools.is_empty(),
                "tool_choice is \"required\" but tools is empty"
            );
            Ok((ctx.tools.iter().collect(), true))
        }
        ChatCompletionToolChoiceOption::Named(named) => {
            let tool = ctx
                .tools
                .iter()
                .find(|t| t.function.name == named.function.name)
                .ok_or_else(|| {
                    anyhow::anyhow!(
                        "tool named \"{}\" in tool_choice is not present in tools",
                        named.function.name
                    )
                })?;
            Ok((vec![tool], true))
        }
    }
}

/// Resolve one tool's argument schema for structural tag generation.
pub(crate) fn resolve_tool_schema(tool: &ChatCompletionTool, strict_schema: bool) -> Value {
    // xgrammar uses `true` for syntactically valid but schema-unconstrained JSON.
    let default_schema = json!(true);

    let use_tool_schema = strict_schema || tool.function.strict.unwrap_or(false);
    if use_tool_schema {
        tool.function.parameters.clone().unwrap_or(default_schema)
    } else {
        default_schema
    }
}

/// Builder for model-family-specific tool-call structural tags.
#[derive(Debug, Clone)]
pub enum StructuralTagBuilder {
    /// Simple `triggered_tags` format with one tag template per tool.
    TriggeredTags(TriggeredTagsConfig),

    /// DeepSeek DSML format with a `triggered_tags` wrapper and invoke list.
    DsmlToolCalls(DsmlToolCallsConfig),
}

impl StructuralTagBuilder {
    /// Build the structural tag for the given request context.
    ///
    /// Returns `Ok(None)` when `tool_choice="none"` (use
    /// [`build_tool_call_ban`](Self::build_tool_call_ban) for that case)
    /// or when the tools list is empty for `tool_choice="auto"`.
    ///
    /// Returns `Err` when the request is invalid (e.g. named tool not found
    /// in the tools list, or empty tools for `tool_choice="required"`).
    pub fn build_tool_call_format(
        &self,
        ctx: &ToolCallFormatBuildContext<'_>,
    ) -> anyhow::Result<Option<Value>> {
        let structural_tag = match self {
            Self::TriggeredTags(config) => triggered_tags::build_triggered_tags(config, ctx)?,
            Self::DsmlToolCalls(config) => dsml::build_dsml_tool_calls(config, ctx)?,
        };

        structural_tag
            .map(|tag| serde_json::to_value(tag).map_err(Into::into))
            .transpose()
    }

    /// Build a structural tag that prevents tool-call generation for
    /// `tool_choice="none"`.
    ///
    /// Returns `Ok(None)` when no ban tokens are configured.
    pub fn build_tool_call_ban(&self) -> anyhow::Result<Option<Value>> {
        let tokens = self.ban_tokens();
        if tokens.is_empty() {
            return Ok(None);
        }

        let content = Format::AnyTokens(AnyTokensFormat {
            exclude_tokens: tokens.to_vec(),
        });

        let tag = StructuralTag {
            format: Format::Tag(TagFormat {
                begin: String::new(),
                content: Box::new(content),
                end: String::new(),
            }),
        };

        serde_json::to_value(tag).map(Some).map_err(Into::into)
    }

    /// Returns the tokens to ban for `tool_choice="none"`.
    pub fn ban_tokens(&self) -> &[String] {
        match self {
            Self::TriggeredTags(config) => &config.tool_call_ban_tokens,
            Self::DsmlToolCalls(config) => &config.tool_call_ban_tokens,
        }
    }
}
