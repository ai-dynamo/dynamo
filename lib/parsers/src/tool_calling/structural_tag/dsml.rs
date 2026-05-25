// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! [`StructuralTagBuilder::DsmlToolCalls`] implementation.
//!
//! DSML uses two structural tag levels: an outer `triggered_tags` block and
//! an inner `tags_with_separator` list of per-tool invoke tags.

use serde::{Deserialize, Serialize};

use super::TOOL_NAME_PLACEHOLDER;
use super::builder::{ToolCallFormatBuildContext, resolve_tool_schema, resolve_tools_to_include};
use super::format::{
    Format, JsonSchemaFormat, JsonSchemaStyle, StructuralTag, TagFormat, TagsWithSeparatorFormat,
    TriggeredTagsFormat,
};

/// Configuration for the nested DSML tool-call structural tag.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct DsmlToolCallsConfig {
    /// Trigger for the outer DSML block.
    pub trigger: String,

    /// Begin string for the outer block tag.
    pub block_begin: String,

    /// End string for the outer block tag.
    pub block_end: String,

    /// Invoke begin template; [`TOOL_NAME_PLACEHOLDER`] is replaced per tool.
    pub invoke_begin_template: String,

    /// End string for each invoke tag.
    pub invoke_end: String,

    /// Separator between consecutive invoke tags.
    pub separator: String,

    /// Tokens to ban when `tool_choice="none"`.
    #[serde(default)]
    pub tool_call_ban_tokens: Vec<String>,

    /// Closing tag for prompt-injected reasoning, if this parser supports it.
    #[serde(default)]
    pub reasoning_end: Option<String>,
}

/// Build a DSML tool-calls structural tag from config and request context.
pub(crate) fn build_dsml_tool_calls(
    config: &DsmlToolCallsConfig,
    ctx: &ToolCallFormatBuildContext<'_>,
) -> anyhow::Result<Option<StructuralTag>> {
    let (tools_to_include, outer_at_least_one) = resolve_tools_to_include(ctx)?;

    if tools_to_include.is_empty() {
        return Ok(None);
    }

    // Per-tool invoke tags inside the DSML block.
    let invoke_tags: Vec<TagFormat> = tools_to_include
        .iter()
        .map(|tool| {
            // function name validated by validate_tools() in the request handler
            let begin = config
                .invoke_begin_template
                .replace(TOOL_NAME_PLACEHOLDER, &tool.function.name);

            TagFormat {
                begin,
                content: Box::new(Format::JsonSchema(JsonSchemaFormat {
                    json_schema: resolve_tool_schema(tool, ctx.strict_schema()),
                    style: JsonSchemaStyle::DeepseekXml,
                })),
                end: config.invoke_end.clone(),
            }
        })
        .collect();

    // Inner list of invokes.
    let inner_content = Format::TagsWithSeparator(TagsWithSeparatorFormat {
        tags: invoke_tags,
        separator: config.separator.clone(),
        at_least_one: true,
        stop_after_first: ctx.stop_after_first(),
    });

    // Outer trigger that opens the DSML block.
    let block_tag = TagFormat {
        begin: config.block_begin.clone(),
        content: Box::new(inner_content),
        end: config.block_end.clone(),
    };

    Ok(Some(StructuralTag {
        format: Format::TriggeredTags(TriggeredTagsFormat {
            triggers: vec![config.trigger.clone()],
            tags: vec![block_tag],
            at_least_one: outer_at_least_one,
            stop_after_first: ctx.stop_after_first(),
        }),
    }))
}
