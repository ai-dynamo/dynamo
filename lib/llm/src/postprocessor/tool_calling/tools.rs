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

pub use crate::preprocessor::tools::request::*;
pub use super::response::*;

// Import json_parser from postprocessor module
pub use super::json_parser::*;



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


/// Try parsing a string as a structured tool call, for aggregation usage.
///
/// If successful, returns a `ChatCompletionMessageToolCall`.
pub fn try_tool_call_parse_aggregate(
    message: &str,
) -> anyhow::Result<Option<async_openai::types::ChatCompletionMessageToolCall>> {
    let parsed = try_tool_call_parse_json(message)?;
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
    let parsed = try_tool_call_parse_json(message)?;
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
        let result = try_tool_call_parse_json(input).unwrap().unwrap();
        let (name, args) = extract_name_and_args(result);
        assert_eq!(name, "hello");
        assert_eq!(args["x"], 1);
        assert_eq!(args["y"], 2);
    }

    #[test]
    fn parses_single_arguments_object() {
        let input = r#"{ "name": "world", "arguments": { "a": "abc", "b": 42 } }"#;
        let result = try_tool_call_parse_json(input).unwrap().unwrap();
        let (name, args) = extract_name_and_args(result);
        assert_eq!(name, "world");
        assert_eq!(args["a"], "abc");
        assert_eq!(args["b"], 42);
    }

    #[test]
    fn parses_vec_of_parameters_and_takes_last() {
        let input = r#"[{ "name": "first", "parameters": { "a": 1 } }, { "name": "second", "parameters": { "b": 2 } }]"#;
        let result = try_tool_call_parse_json(input).unwrap().unwrap();
        let (name, args) = extract_name_and_args(result);
        assert_eq!(name, "second");
        assert_eq!(args["b"], 2);
    }

    #[test]
    fn parses_vec_of_arguments_and_takes_last() {
        let input = r#"[{ "name": "alpha", "arguments": { "a": "x" } }, { "name": "omega", "arguments": { "z": "y" } }]"#;
        let result = try_tool_call_parse_json(input).unwrap().unwrap();
        let (name, args) = extract_name_and_args(result);
        assert_eq!(name, "omega");
        assert_eq!(args["z"], "y");
    }

    #[test]
    fn parses_toolcall_wrapped_payload() {
        let input =
            r#"<TOOLCALL>[{ "name": "wrapped", "parameters": { "foo": "bar" } }]</TOOLCALL>"#;
        let result = try_tool_call_parse_json(input).unwrap().unwrap();
        let (name, args) = extract_name_and_args(result);
        assert_eq!(name, "wrapped");
        assert_eq!(args["foo"], "bar");
    }

    #[test]
    fn parses_python_tag_prefixed_payload() {
        let input = r#"<|python_tag|>{ "name": "pyfunc", "arguments": { "k": "v" } }"#;
        let result = try_tool_call_parse_json(input).unwrap().unwrap();
        let (name, args) = extract_name_and_args(result);
        assert_eq!(name, "pyfunc");
        assert_eq!(args["k"], "v");
    }

    #[test]
    fn returns_none_on_invalid_input() {
        let input = r#"not even json"#;
        let result = try_tool_call_parse_json(input).unwrap();
        assert!(result.is_none());
    }

    #[test]
    fn returns_none_on_valid_json_wrong_shape() {
        let input = r#"{ "foo": "bar" }"#;
        let result = try_tool_call_parse_json(input).unwrap();
        assert!(result.is_none());
    }
}
