// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Public structural tag builder API used by the Dynamo preprocessor.

use serde::{Deserialize, Serialize};
use serde_json::{Value, json};

use crate::tool_calling::{ToolChoice, ToolDefinition};

use super::dsml::{self, DsmlToolCallsConfig};
use super::format::{
    AnyTextFormat, AnyTokensFormat, Format, SequenceFormat, StructuralTag, TagFormat,
};
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
    pub tool_choice: &'a ToolChoice,
    /// All tools from the request.
    pub tools: &'a [ToolDefinition],
    /// From the request; `Some(false)` sets `stop_after_first` in the tag.
    pub parallel_tool_calls: Option<bool>,
    /// Schema strictness mode for tool arguments.
    pub schema_mode: StructuralTagSchemaMode,
    /// Whether generation starts inside an already-opened reasoning block.
    pub starts_in_reasoning: bool,
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
) -> anyhow::Result<(Vec<&'a ToolDefinition>, bool)> {
    match ctx.tool_choice {
        ToolChoice::None => Ok((vec![], false)),
        ToolChoice::Auto => Ok((ctx.tools.iter().collect(), false)),
        ToolChoice::Required => {
            anyhow::ensure!(
                !ctx.tools.is_empty(),
                "tool_choice is \"required\" but tools is empty"
            );
            Ok((ctx.tools.iter().collect(), true))
        }
        ToolChoice::Named(name) => {
            let tool = ctx.tools.iter().find(|t| t.name == *name).ok_or_else(|| {
                anyhow::anyhow!(
                    "tool named \"{}\" in tool_choice is not present in tools",
                    name
                )
            })?;
            Ok((vec![tool], true))
        }
    }
}

/// Resolve one tool's argument schema for structural tag generation.
pub(crate) fn resolve_tool_schema(tool: &ToolDefinition, strict_schema: bool) -> Value {
    // xgrammar uses `true` for syntactically valid but schema-unconstrained JSON.
    let default_schema = json!(true);

    let use_tool_schema = strict_schema || tool.strict.unwrap_or(false);
    if use_tool_schema {
        tool.parameters.clone().unwrap_or(default_schema)
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
            .map(|tag| self.wrap_reasoning_prefix_if_needed(ctx, tag))
            .transpose()?
            .map(|tag| serde_json::to_value(tag).map_err(Into::into))
            .transpose()
    }

    fn wrap_reasoning_prefix_if_needed(
        &self,
        ctx: &ToolCallFormatBuildContext<'_>,
        suffix: StructuralTag,
    ) -> anyhow::Result<StructuralTag> {
        let should_wrap_with_reasoning_tag = ctx.starts_in_reasoning
            && matches!(ctx.tool_choice, ToolChoice::Required | ToolChoice::Named(_));
        if !should_wrap_with_reasoning_tag {
            return Ok(suffix);
        }

        let reasoning_end = self.reasoning_end().ok_or_else(|| {
            anyhow::anyhow!("reasoning end tag is not configured for structural tag")
        })?;
        let reasoning_prefix = Format::Tag(TagFormat {
            begin: String::new(),
            content: Box::new(Format::AnyText(AnyTextFormat { excludes: vec![] })),
            end: reasoning_end.to_string(),
        });

        Ok(StructuralTag {
            format: Format::Sequence(SequenceFormat {
                elements: vec![reasoning_prefix, suffix.format],
            }),
        })
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

    fn reasoning_end(&self) -> Option<&str> {
        match self {
            Self::TriggeredTags(config) => config.reasoning_end.as_deref(),
            Self::DsmlToolCalls(config) => config.reasoning_end.as_deref(),
        }
    }
}
