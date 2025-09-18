// SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

pub use super::config::ToolCallConfig;
pub use super::parsers::detect_and_parse_tool_call;

/// Try parsing a string as a structured tool call, for aggregation usage.
///
/// If successful, returns a `ChatCompletionMessageToolCall`.
pub fn try_tool_call_parse_aggregate(
    message: &str,
    parser_str: Option<&str>,
) -> anyhow::Result<(
    Vec<dynamo_async_openai::types::ChatCompletionMessageToolCall>,
    Option<String>,
)> {
    if parser_str.is_none() {
        tracing::info!("No tool parser provided. Trying parsing with default parser.");
    } else {
        tracing::info!("Using tool parser: {:?}", parser_str);
    }
    let (parsed, content) = detect_and_parse_tool_call(message, parser_str)?;
    if parsed.is_empty() {
        return Ok((vec![], content));
    }
    Ok((
        parsed
            .into_iter()
            .map(
                |parsed| dynamo_async_openai::types::ChatCompletionMessageToolCall {
                    id: parsed.id,
                    r#type: dynamo_async_openai::types::ChatCompletionToolType::Function,
                    function: dynamo_async_openai::types::FunctionCall {
                        name: parsed.function.name,
                        arguments: parsed.function.arguments,
                    },
                },
            )
            .collect(),
        content,
    ))
}

/// Try parsing a string as a structured tool call, for streaming (delta) usage.
///
/// If successful, returns a `ChatCompletionMessageToolCallChunk`.
pub fn try_tool_call_parse_stream(
    message: &str,
    parser_str: Option<&str>,
) -> anyhow::Result<(
    Vec<dynamo_async_openai::types::ChatCompletionMessageToolCallChunk>,
    Option<String>,
)> {
    let (parsed, content) = detect_and_parse_tool_call(message, parser_str)?;
    if parsed.is_empty() {
        return Ok((vec![], content));
    }
    Ok((
        parsed
            .into_iter()
            .enumerate()
            .map(
                |(idx, parsed)| dynamo_async_openai::types::ChatCompletionMessageToolCallChunk {
                    index: idx as u32,
                    id: Some(parsed.id),
                    r#type: Some(dynamo_async_openai::types::ChatCompletionToolType::Function),
                    function: Some(dynamo_async_openai::types::FunctionCallStream {
                        name: Some(parsed.function.name),
                        arguments: Some(parsed.function.arguments),
                    }),
                    // Add other fields as needed if required by the struct definition
                },
            )
            .collect(),
        content,
    ))
}
