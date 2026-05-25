// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

pub use super::config::ToolCallConfig;
pub use super::parsers::{detect_and_parse_tool_call, detect_and_parse_tool_call_with_recovery};
pub use super::response::{
    CalledFunctionStream, ToolCallResponse, ToolCallResponseChunk, ToolCallType,
};

/// Try parsing a string as a structured tool call, for aggregation usage.
///
/// If successful, returns the parser-native [`ToolCallResponse`] values.
/// Consumers that need protocol/wire types map these locally.
///
/// Streaming jail callers (`should_exit_jail_early`, mid-stream early-exit
/// confirmation) MUST keep using this function — `allow_eof_recovery` stays
/// off so the parser doesn't claim a complete tool call before the end-token
/// has actually arrived.
pub async fn try_tool_call_parse_aggregate(
    message: &str,
    parser_str: Option<&str>,
    tools: Option<&[super::ToolDefinition]>,
) -> anyhow::Result<(Vec<ToolCallResponse>, Option<String>)> {
    if parser_str.is_none() {
        tracing::debug!("No tool parser provided. Trying parsing with default parser.");
    } else {
        tracing::debug!("Using tool parser: {:?}", parser_str);
    }
    let (parsed, content) = detect_and_parse_tool_call(message, parser_str, tools).await?;
    if parsed.is_empty() {
        return Ok((vec![], content));
    }
    Ok((parsed, content))
}

/// Finalize-only variant of [`try_tool_call_parse_aggregate`] that enables
/// EOF recovery (missing outer end-token, truncated JSON args). Use this from
/// stream-end / non-streaming aggregator paths only — never from streaming
/// jail early-exit logic.
pub async fn try_tool_call_parse_aggregate_finalize(
    message: &str,
    parser_str: Option<&str>,
    tools: Option<&[super::ToolDefinition]>,
) -> anyhow::Result<(Vec<ToolCallResponse>, Option<String>)> {
    let (parsed, content) =
        detect_and_parse_tool_call_with_recovery(message, parser_str, tools).await?;
    if parsed.is_empty() {
        return Ok((vec![], content));
    }
    Ok((parsed, content))
}

/// Try parsing a string as a structured tool call, for streaming (delta) usage.
///
/// If successful, returns parser-native [`ToolCallResponseChunk`] values.
/// Consumers that need protocol/wire types map these locally.
pub async fn try_tool_call_parse_stream(
    message: &str,
    parser_str: Option<&str>,
    tools: Option<&[super::ToolDefinition]>,
) -> anyhow::Result<(Vec<ToolCallResponseChunk>, Option<String>)> {
    let (parsed, content) = detect_and_parse_tool_call(message, parser_str, tools).await?;
    if parsed.is_empty() {
        return Ok((vec![], content));
    }
    Ok((
        parsed
            .into_iter()
            .enumerate()
            .map(|(idx, parsed)| ToolCallResponseChunk {
                index: idx as u32,
                id: Some(parsed.id),
                tp: Some(ToolCallType::Function),
                function: Some(CalledFunctionStream {
                    name: Some(parsed.function.name),
                    arguments: Some(parsed.function.arguments),
                }),
            })
            .collect(),
        content,
    ))
}
