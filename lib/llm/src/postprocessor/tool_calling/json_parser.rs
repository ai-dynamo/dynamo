// SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::collections::HashMap;

use serde_json::Value;
use uuid::Uuid;

use super::parsers::JsonParserConfig;
use super::response::{CalledFunction, ToolCallResponse, ToolCallType};

// Same as CalledFunction with named parameters
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct CalledFunctionParameters {
    pub name: String,
    pub parameters: HashMap<String, Value>,
}

// Same as CalledFunction with named parameters
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct CalledFunctionArguments {
    pub name: String,
    pub arguments: HashMap<String, Value>,
}

/// Attempts to parse a tool call from a raw LLM message string into a unified [`ToolCallResponse`] format.
///
/// This is a flexible helper that handles a variety of potential formats emitted by LLMs for function/tool calls,
/// including wrapped payloads (`<TOOLCALL>[...]</TOOLCALL>`, `<|python_tag|>...`) and JSON representations
/// with either `parameters` or `arguments` fields.
///
/// # Supported Formats
///
/// The input `message` may be one of:
///
/// - `<TOOLCALL>[{ "name": ..., "parameters": { ... } }]</TOOLCALL>`
/// - `<|python_tag|>{ "name": ..., "arguments": { ... } }`
/// - Raw JSON of:
///     - `CalledFunctionParameters`: `{ "name": ..., "parameters": { ... } }`
///     - `CalledFunctionArguments`: `{ "name": ..., "arguments": { ... } }`
///     - Or a list of either of those types: `[ { "name": ..., "arguments": { ... } }, ... ]`
///
/// # Return
///
/// - `Ok(Some(ToolCallResponse))` if parsing succeeds
/// - `Ok(None)` if input format is unrecognized or invalid JSON
/// - `Err(...)` if JSON is valid but deserialization or argument re-serialization fails
///
/// # Note on List Handling
///
/// When the input contains a list of tool calls (either with `parameters` or `arguments`),
/// only the **last item** in the list is returned. This design choice assumes that the
/// most recent tool call in a list is the one to execute.
///
/// # Errors
///
/// Returns a `Result::Err` only if an inner `serde_json::to_string(...)` fails
/// (e.g., if the arguments are not serializable).
///
/// # Examples
///
/// ```ignore
/// let input = r#"<TOOLCALL>[{ "name": "search", "parameters": { "query": "rust" } }]</TOOLCALL>"#;
/// let result = try_tool_call_parse_json(input)?;
/// assert!(result.is_some());
/// ```
pub fn try_tool_call_parse_json(
    message: &str,
    config: &JsonParserConfig,
) -> anyhow::Result<Option<ToolCallResponse>> {
    // Log the config we are using
    tracing::debug!("Using JSON parser config: {:?}", config);
    let trimmed = message.trim();

    // Support <TOOLCALL>[ ... ] or <tool_call>[ ... ]
    let json = if let Some(stripped) = trimmed.strip_prefix("<TOOLCALL>[") {
        if let Some(stripped) = stripped.strip_suffix("]</TOOLCALL>") {
            tracing::debug!("Stripping <TOOLCALL> wrapper from tool call payload");
            stripped
        } else {
            trimmed
        }

    // Support custom/LLM-formatted `<|python_tag|>` preamble
    } else if let Some(stripped) = trimmed.strip_prefix("<|python_tag|>") {
        tracing::debug!("Stripping <|python_tag|> prefix from tool call payload");
        stripped

    // Otherwise, assume input is clean JSON
    } else {
        trimmed
    };

    // Anonymous function to attempt deserialization into a known representation
    let parse = |name: String, args: HashMap<String, Value>| -> anyhow::Result<_> {
        Ok(ToolCallResponse {
            id: format!("call-{}", Uuid::new_v4()),
            tp: ToolCallType::Function,
            function: CalledFunction {
                name,
                arguments: serde_json::to_string(&args)?,
            },
        })
    };

    // CalledFunctionParameters: Single { name, parameters }
    // Example:
    // {
    //   "name": "search_docs",
    //   "parameters": {
    //     "query": "how to use Rust",
    //     "limit": 5
    //   }
    // }
    if let Ok(single) = serde_json::from_str::<CalledFunctionParameters>(json) {
        return parse(single.name, single.parameters).map(Some);

    // CalledFunctionArguments: Single { name, arguments }
    // Example:
    // {
    //   "name": "summarize",
    //   "arguments": {
    //     "text": "Rust is a systems programming language.",
    //     "length": "short"
    //   }
    // }
    } else if let Ok(single) = serde_json::from_str::<CalledFunctionArguments>(json) {
        return parse(single.name, single.arguments).map(Some);

    // Vec<CalledFunctionParameters>: List of { name, parameters }
    // Example:
    // [
    //   { "name": "lookup_user", "parameters": { "user_id": "123" } },
    //   { "name": "send_email", "parameters": { "to": "user@example.com", "subject": "Welcome!" } }
    // ]
    // We pop the last item in the list to use.
    } else if let Ok(mut list) = serde_json::from_str::<Vec<CalledFunctionParameters>>(json) {
        if let Some(item) = list.pop() {
            return parse(item.name, item.parameters).map(Some);
        }

    // Vec<CalledFunctionArguments>: List of { name, arguments }
    // Example:
    // [
    //   {
    //     "name": "get_weather",
    //     "arguments": {
    //       "location": "San Francisco",
    //       "units": "celsius"
    //     }
    //   }
    // ]
    // Again, we take the last item for processing.
    } else if let Ok(mut list) = serde_json::from_str::<Vec<CalledFunctionArguments>>(json) {
        if let Some(item) = list.pop() {
            return parse(item.name, item.arguments).map(Some);
        }
    }

    Ok(None)
}
