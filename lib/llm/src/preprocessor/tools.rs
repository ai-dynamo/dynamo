// SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

use std::collections::HashMap;

use serde_json::Value;
use uuid::Uuid;

mod request;
mod response;

pub use request::*;
pub use response::*;

/// Matches and processes tool calling patterns in LLM responses
///
/// Supports multiple formats for tool calls:
/// - Single/multiple function calls with parameters/arguments
/// - Auto or user selected tool usage
pub struct ToolCallingMatcher {
    tool_choice: ToolChoice,
}

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

impl ToolCallingMatcher {
    pub fn new(tool_choice: ToolChoice) -> anyhow::Result<Self> {
        Ok(Self { tool_choice })
    }

    pub fn get_call(&self, message: &str) -> anyhow::Result<Vec<ToolCallResponse>> {
        if matches!(self.tool_choice, ToolChoice::None) {
            return Ok(Vec::new());
        }

        if let Ok(deser) = serde_json::from_str::<CalledFunctionParameters>(message) {
            let id = format!("call-{}", Uuid::new_v4());
            Ok(vec![ToolCallResponse {
                id,
                tp: ToolCallType::Function,
                function: CalledFunction {
                    name: deser.name,
                    arguments: serde_json::to_string(&deser.parameters)?,
                },
            }])
        } else if let Ok(deser) = serde_json::from_str::<Vec<CalledFunctionParameters>>(message) {
            Ok(deser
                .into_iter()
                .map(|deser| {
                    let id = format!("call-{}", Uuid::new_v4());
                    Ok(ToolCallResponse {
                        id,
                        tp: ToolCallType::Function,
                        function: CalledFunction {
                            name: deser.name,
                            arguments: serde_json::to_string(&deser.parameters)?,
                        },
                    })
                })
                .collect::<anyhow::Result<Vec<_>>>()?)
        } else if let Ok(deser) = serde_json::from_str::<CalledFunctionArguments>(message) {
            let id = format!("call-{}", Uuid::new_v4());
            Ok(vec![ToolCallResponse {
                id,
                tp: ToolCallType::Function,
                function: CalledFunction {
                    name: deser.name,
                    arguments: serde_json::to_string(&deser.arguments)?,
                },
            }])
        } else if let Ok(deser) = serde_json::from_str::<Vec<CalledFunctionArguments>>(message) {
            Ok(deser
                .into_iter()
                .map(|deser| {
                    let id = format!("call-{}", Uuid::new_v4());
                    Ok(ToolCallResponse {
                        id,
                        tp: ToolCallType::Function,
                        function: CalledFunction {
                            name: deser.name,
                            arguments: serde_json::to_string(&deser.arguments)?,
                        },
                    })
                })
                .collect::<anyhow::Result<Vec<_>>>()?)
        } else {
            if matches!(self.tool_choice, ToolChoice::Tool(_)) {
                anyhow::bail!("Tool choice was required but no tools were called.")
            }
            Ok(Vec::new())
        }
    }
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
/// let result = try_parse_call_common(input)?;
/// assert!(result.is_some());
/// ```
pub fn try_parse_call_common(message: &str) -> anyhow::Result<Option<ToolCallResponse>> {
    let trimmed = message.trim();

    // Support <TOOLCALL>[ ... ] or <tool_call>[ ... ]
    let json = if trimmed.starts_with("<TOOLCALL>[") && trimmed.ends_with("]</TOOLCALL>") {
        tracing::debug!("Stripping <TOOLCALL> wrapper from tool call payload");
        &trimmed["<TOOLCALL>[".len()..trimmed.len() - "]</TOOLCALL>".len()]

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

    // CalledFunctinoArguments: Single { name, arguments }
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

/// Try parsing a string as a structured tool call, for aggregation usage.
///
/// If successful, returns a `ChatCompletionMessageToolCall`.
pub fn try_parse_tool_call_aggregate(
    message: &str,
) -> anyhow::Result<Option<async_openai::types::ChatCompletionMessageToolCall>> {
    let parsed = try_parse_call_common(message)?;
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
pub fn try_parse_tool_call_stream(
    message: &str,
) -> anyhow::Result<Option<async_openai::types::ChatCompletionMessageToolCallChunk>> {
    let parsed = try_parse_call_common(message)?;
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

#[cfg(test)]
mod tests {
    use super::*;

    fn extract_name_and_args(call: ToolCallResponse) -> (String, serde_json::Value) {
        let args: serde_json::Value = serde_json::from_str(&call.function.arguments).unwrap();
        (call.function.name, args)
    }

    #[test]
    fn parses_single_parameters_object() {
        let input = r#"{ "name": "hello", "parameters": { "x": 1, "y": 2 } }"#;
        let result = try_parse_call_common(input).unwrap().unwrap();
        let (name, args) = extract_name_and_args(result);
        assert_eq!(name, "hello");
        assert_eq!(args["x"], 1);
        assert_eq!(args["y"], 2);
    }

    #[test]
    fn parses_single_arguments_object() {
        let input = r#"{ "name": "world", "arguments": { "a": "abc", "b": 42 } }"#;
        let result = try_parse_call_common(input).unwrap().unwrap();
        let (name, args) = extract_name_and_args(result);
        assert_eq!(name, "world");
        assert_eq!(args["a"], "abc");
        assert_eq!(args["b"], 42);
    }

    #[test]
    fn parses_vec_of_parameters_and_takes_last() {
        let input = r#"[{ "name": "first", "parameters": { "a": 1 } }, { "name": "second", "parameters": { "b": 2 } }]"#;
        let result = try_parse_call_common(input).unwrap().unwrap();
        let (name, args) = extract_name_and_args(result);
        assert_eq!(name, "second");
        assert_eq!(args["b"], 2);
    }

    #[test]
    fn parses_vec_of_arguments_and_takes_last() {
        let input = r#"[{ "name": "alpha", "arguments": { "a": "x" } }, { "name": "omega", "arguments": { "z": "y" } }]"#;
        let result = try_parse_call_common(input).unwrap().unwrap();
        let (name, args) = extract_name_and_args(result);
        assert_eq!(name, "omega");
        assert_eq!(args["z"], "y");
    }

    #[test]
    fn parses_toolcall_wrapped_payload() {
        let input =
            r#"<TOOLCALL>[{ "name": "wrapped", "parameters": { "foo": "bar" } }]</TOOLCALL>"#;
        let result = try_parse_call_common(input).unwrap().unwrap();
        let (name, args) = extract_name_and_args(result);
        assert_eq!(name, "wrapped");
        assert_eq!(args["foo"], "bar");
    }

    #[test]
    fn parses_python_tag_prefixed_payload() {
        let input = r#"<|python_tag|>{ "name": "pyfunc", "arguments": { "k": "v" } }"#;
        let result = try_parse_call_common(input).unwrap().unwrap();
        let (name, args) = extract_name_and_args(result);
        assert_eq!(name, "pyfunc");
        assert_eq!(args["k"], "v");
    }

    #[test]
    fn returns_none_on_invalid_input() {
        let input = r#"not even json"#;
        let result = try_parse_call_common(input).unwrap();
        assert!(result.is_none());
    }

    #[test]
    fn returns_none_on_valid_json_wrong_shape() {
        let input = r#"{ "foo": "bar" }"#;
        let result = try_parse_call_common(input).unwrap();
        assert!(result.is_none());
    }
}
