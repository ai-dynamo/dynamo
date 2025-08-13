// SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

pub use super::response::*;
pub use crate::preprocessor::tools::request::*;

// Import json_parser from postprocessor module
pub use super::json_parser::*;
pub use super::parsers::{try_tool_call_parse, ToolCallConfig};

/// Try parsing a string as a structured tool call, for aggregation usage.
///
/// If successful, returns a `ChatCompletionMessageToolCall`.
pub fn try_tool_call_parse_aggregate(
    message: &str,
) -> anyhow::Result<Option<async_openai::types::ChatCompletionMessageToolCall>> {
    let config = ToolCallConfig::default();
    let parsed = try_tool_call_parse(message, &config)?;
    if let Some(parsed) = parsed {
        Ok(Some(async_openai::types::ChatCompletionMessageToolCall {
            id: parsed.id,
            r#type: async_openai::types::ChatCompletionToolType::Function,
            function: async_openai::types::FunctionCall {
                name: parsed.function.name,
                arguments: parsed.function.arguments,
            },
        }))
    } else {
        Ok(None)
    }
}

/// Try parsing a string as a structured tool call, for streaming (delta) usage.
///
/// If successful, returns a `ChatCompletionMessageToolCallChunk`.
pub fn try_tool_call_parse_stream(
    message: &str,
) -> anyhow::Result<Option<async_openai::types::ChatCompletionMessageToolCallChunk>> {
    let config = ToolCallConfig::default();
    let parsed = try_tool_call_parse(message, &config)?;
    if let Some(parsed) = parsed {
        Ok(Some(
            async_openai::types::ChatCompletionMessageToolCallChunk {
                index: 0,
                id: Some(parsed.id),
                r#type: Some(async_openai::types::ChatCompletionToolType::Function),
                function: Some(async_openai::types::FunctionCallStream {
                    name: Some(parsed.function.name),
                    arguments: Some(parsed.function.arguments),
                }),
            },
        ))
    } else {
        Ok(None)
    }
}
