// SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

pub use super::response::*;
pub use crate::preprocessor::tools::request::*;

// Import json_parser from postprocessor module
pub use super::json_parser::*;
pub use super::parsers::{detect_and_parse_tool_call, ToolCallConfig};

/// Try parsing a string as a structured tool call, for aggregation usage.
///
/// If successful, returns a `ChatCompletionMessageToolCall`.
pub fn try_tool_call_parse_aggregate(
    message: &str,
    parser_str: Option<&str>,
) -> anyhow::Result<Vec<async_openai::types::ChatCompletionMessageToolCall>> {
    let parsed = detect_and_parse_tool_call(message, parser_str)?;
    if parsed.is_empty() {
        return Ok(vec![]);
    }
    Ok(parsed
        .into_iter()
        .map(
            |parsed| async_openai::types::ChatCompletionMessageToolCall {
                id: parsed.id,
                r#type: async_openai::types::ChatCompletionToolType::Function,
                function: async_openai::types::FunctionCall {
                    name: parsed.function.name,
                    arguments: parsed.function.arguments,
                },
            },
        )
        .collect())
}

/// Try parsing a string as a structured tool call, for streaming (delta) usage.
///
/// If successful, returns a `ChatCompletionMessageToolCallChunk`.
pub fn try_tool_call_parse_stream(
    message: &str,
    parser_str: Option<&str>,
) -> anyhow::Result<Vec<async_openai::types::ChatCompletionMessageToolCallChunk>> {
    let parsed = detect_and_parse_tool_call(message, parser_str)?;
    if parsed.is_empty() {
        return Ok(vec![]);
    }
    Ok(parsed
        .into_iter()
        .enumerate()
        .map(
            |(idx, parsed)| async_openai::types::ChatCompletionMessageToolCallChunk {
                index: idx as u32,
                id: Some(parsed.id),
                r#type: Some(async_openai::types::ChatCompletionToolType::Function),
                function: Some(async_openai::types::FunctionCallStream {
                    name: Some(parsed.function.name),
                    arguments: Some(parsed.function.arguments),
                }),
                // Add other fields as needed if required by the struct definition
            },
        )
        .collect())
}
