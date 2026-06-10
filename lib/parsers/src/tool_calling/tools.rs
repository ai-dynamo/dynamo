// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

pub use super::config::ToolCallConfig;
#[allow(deprecated)]
pub use super::parsers::detect_and_parse_tool_call_with_stream_finalize_recovery;
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
/// the common EOF recovery paths (missing outer end-token, truncated JSON
/// args, and — for DSML/DeepSeek V4 — a missing outer `</｜DSML｜tool_calls>`
/// wrapper). Use this from finalize paths: both non-streaming aggregate and
/// stream-end jail finalization. Never use it from streaming jail early-exit
/// logic, which must keep `allow_eof_recovery=false`.
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

/// Deprecated compatibility shim retained for the published `dynamo-parsers`
/// API. Batch/non-streaming and stream-end finalize now share one recovery
/// path; call [`try_tool_call_parse_aggregate_finalize`] directly.
#[deprecated(
    note = "batch and stream finalize now share one recovery path; use try_tool_call_parse_aggregate_finalize"
)]
pub async fn try_tool_call_parse_aggregate_stream_finalize(
    message: &str,
    parser_str: Option<&str>,
    tools: Option<&[super::ToolDefinition]>,
) -> anyhow::Result<(Vec<ToolCallResponse>, Option<String>)> {
    try_tool_call_parse_aggregate_finalize(message, parser_str, tools).await
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

#[cfg(test)]
mod tests {
    use super::*;

    // Hermes-style single tool call.
    const SINGLE: &str = r#"<tool_call>{"name":"get_weather","arguments":{"location":"San Francisco, CA","unit":"celsius"}}</tool_call>"#;

    // Two parallel hermes-style tool calls.
    const PARALLEL: &str = r#"<tool_call>{"name":"a","arguments":{"k":"v1"}}</tool_call>
<tool_call>{"name":"b","arguments":{"k":"v2"}}</tool_call>"#;

    // Tool call with empty arguments object.
    const EMPTY_ARGS: &str = r#"<tool_call>{"name":"ping","arguments":{}}</tool_call>"#;

    /// `try_tool_call_parse_aggregate` returns native `ToolCallResponse`
    /// values with the parsed id/type/name/arguments.
    #[tokio::test]
    async fn aggregate_returns_native_tool_call_response() {
        let (calls, _content): (Vec<ToolCallResponse>, _) =
            try_tool_call_parse_aggregate(SINGLE, Some("hermes"), None)
                .await
                .unwrap();
        assert_eq!(calls.len(), 1);
        assert!(!calls[0].id.is_empty());
        assert!(matches!(calls[0].tp, ToolCallType::Function));
        assert_eq!(calls[0].function.name, "get_weather");
        assert_eq!(
            calls[0].function.arguments,
            r#"{"location":"San Francisco, CA","unit":"celsius"}"#
        );
    }

    /// Parallel calls preserve ordering and per-call argument byte spans.
    #[tokio::test]
    async fn aggregate_returns_native_for_parallel_calls() {
        let (calls, _): (Vec<ToolCallResponse>, _) =
            try_tool_call_parse_aggregate(PARALLEL, Some("hermes"), None)
                .await
                .unwrap();
        assert_eq!(calls.len(), 2);
        assert_eq!(calls[0].function.name, "a");
        assert_eq!(calls[0].function.arguments, r#"{"k":"v1"}"#);
        assert_eq!(calls[1].function.name, "b");
        assert_eq!(calls[1].function.arguments, r#"{"k":"v2"}"#);
    }

    /// The finalize variant also returns native `ToolCallResponse`.
    #[tokio::test]
    async fn aggregate_finalize_returns_native_tool_call_response() {
        let (calls, _): (Vec<ToolCallResponse>, _) =
            try_tool_call_parse_aggregate_finalize(SINGLE, Some("hermes"), None)
                .await
                .unwrap();
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].function.name, "get_weather");
        assert_eq!(
            calls[0].function.arguments,
            r#"{"location":"San Francisco, CA","unit":"celsius"}"#
        );
    }

    /// `try_tool_call_parse_stream` returns native `ToolCallResponseChunk`
    /// values populating index/id/type and the nested `CalledFunctionStream`.
    #[tokio::test]
    async fn stream_returns_native_tool_call_response_chunk() {
        let (chunks, _content): (Vec<ToolCallResponseChunk>, _) =
            try_tool_call_parse_stream(SINGLE, Some("hermes"), None)
                .await
                .unwrap();
        assert_eq!(chunks.len(), 1);
        let chunk = &chunks[0];
        assert_eq!(chunk.index, 0);
        assert!(chunk.id.as_deref().is_some_and(|id| !id.is_empty()));
        assert!(matches!(chunk.tp, Some(ToolCallType::Function)));
        let func = chunk.function.as_ref().expect("function present");
        assert_eq!(func.name.as_deref(), Some("get_weather"));
        assert_eq!(
            func.arguments.as_deref(),
            Some(r#"{"location":"San Francisco, CA","unit":"celsius"}"#)
        );
    }

    /// Streaming chunks are indexed sequentially for parallel calls.
    #[tokio::test]
    async fn stream_indexes_parallel_calls_sequentially() {
        let (chunks, _): (Vec<ToolCallResponseChunk>, _) =
            try_tool_call_parse_stream(PARALLEL, Some("hermes"), None)
                .await
                .unwrap();
        assert_eq!(chunks.len(), 2);
        assert_eq!(chunks[0].index, 0);
        assert_eq!(chunks[1].index, 1);
        assert_eq!(
            chunks[0].function.as_ref().unwrap().name.as_deref(),
            Some("a")
        );
        assert_eq!(
            chunks[1].function.as_ref().unwrap().name.as_deref(),
            Some("b")
        );
    }

    /// Empty arguments objects survive as `{}` through both entrypoints.
    #[tokio::test]
    async fn empty_arguments_preserved() {
        let (calls, _): (Vec<ToolCallResponse>, _) =
            try_tool_call_parse_aggregate(EMPTY_ARGS, Some("hermes"), None)
                .await
                .unwrap();
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].function.name, "ping");
        assert_eq!(calls[0].function.arguments, "{}");

        let (chunks, _): (Vec<ToolCallResponseChunk>, _) =
            try_tool_call_parse_stream(EMPTY_ARGS, Some("hermes"), None)
                .await
                .unwrap();
        assert_eq!(chunks.len(), 1);
        assert_eq!(
            chunks[0].function.as_ref().unwrap().arguments.as_deref(),
            Some("{}")
        );
    }

    /// No tool call -> empty result, content passed through.
    #[tokio::test]
    async fn no_tool_call_returns_empty() {
        let (calls, _): (Vec<ToolCallResponse>, _) =
            try_tool_call_parse_aggregate("just some prose", Some("hermes"), None)
                .await
                .unwrap();
        assert!(calls.is_empty());

        let (chunks, _): (Vec<ToolCallResponseChunk>, _) =
            try_tool_call_parse_stream("just some prose", Some("hermes"), None)
                .await
                .unwrap();
        assert!(chunks.is_empty());
    }
}
