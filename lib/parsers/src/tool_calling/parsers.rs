// SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use super::json_parser::try_tool_call_parse_json;
use super::response::ToolCallResponse;

/// Represents the format type for tool calls
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub enum ToolCallParserType {
    /// JSON format: `{"name": "function", "arguments": {...}}`
    Json,
    Pythonic,
    Harmony,
    /// <function_call>```typescript
    /// functions.get_current_weather({"location": "Shanghai"})
    /// ```
    Typescript,
    Xml,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct JsonParserConfig {
    /// Start token for list of parallel tool calls (e.g., "<TOOLCALLS>")
    pub parallel_tool_calls_start_tokens: Vec<String>,
    /// End token for list of parallel tool calls (e.g., "</TOOLCALLS>")
    pub parallel_tool_calls_end_tokens: Vec<String>,
    /// Start token for individual tool calls (e.g., "<TOOLCALL>")
    pub tool_call_start_tokens: Vec<String>,
    /// End token for individual tool calls (e.g., "</TOOLCALL>")
    pub tool_call_end_tokens: Vec<String>,
    /// The key for the function name in the tool call
    /// i.e. `{"name": "function", "arguments": {...}}` it would be
    /// "name"
    pub function_name_keys: Vec<String>,
    /// The key for the arguments in the tool call
    /// i.e. `{"name": "function", "arguments": {...}}` it would be
    /// "arguments"
    pub arguments_keys: Vec<String>,
}

impl Default for JsonParserConfig {
    fn default() -> Self {
        Self {
            parallel_tool_calls_start_tokens: vec![],
            parallel_tool_calls_end_tokens: vec![],
            tool_call_start_tokens: vec!["<TOOLCALL>".to_string(), "<|python_tag|>".to_string()],
            tool_call_end_tokens: vec!["</TOOLCALL>".to_string(), "".to_string()],
            function_name_keys: vec!["name".to_string()],
            arguments_keys: vec!["arguments".to_string(), "parameters".to_string()],
        }
    }
}

impl Default for ToolCallConfig {
    fn default() -> Self {
        Self {
            format: ToolCallParserType::Json,
            json: JsonParserConfig::default(),
        }
    }
}

impl ToolCallConfig {
    /// Default configuration for hermes tool calls
    /// <tool_call>{"name": "get_weather", "arguments": {"location": "San Francisco, CA", "unit": "fahrenheit"}}\n</tool_call>
    pub fn hermes() -> Self {
        Self {
            format: ToolCallParserType::Json,
            json: JsonParserConfig {
                tool_call_start_tokens: vec!["<tool_call>".to_string()],
                tool_call_end_tokens: vec!["\n</tool_call>".to_string()],
                ..Default::default()
            },
        }
    }

    /// Default configuration for nemotron tool calls
    /// <TOOLCALL>[{"name": "get_weather", "arguments": {"location": "San Francisco, CA", "unit": "fahrenheit"}}]</TOOLCALL>
    pub fn nemotron_deci() -> Self {
        Self {
            format: ToolCallParserType::Json,
            json: JsonParserConfig {
                tool_call_start_tokens: vec!["<TOOLCALL>".to_string()],
                tool_call_end_tokens: vec!["</TOOLCALL>".to_string()],
                ..Default::default()
            },
        }
    }

    pub fn llama3_json() -> Self {
        // <|python_tag|>{ "name": "get_weather", "arguments": {"location": "San Francisco, CA", "unit": "fahrenheit"} }
        // or { "name": "get_weather", "arguments": {"location": "San Francisco, CA", "unit": "fahrenheit"} }
        Self {
            format: ToolCallParserType::Json,
            json: JsonParserConfig {
                tool_call_start_tokens: vec!["<|python_tag|>".to_string()],
                tool_call_end_tokens: vec!["".to_string()],
                ..Default::default()
            },
        }
    }

    pub fn mistral() -> Self {
        Self {
            format: ToolCallParserType::Json,
            json: JsonParserConfig {
                tool_call_start_tokens: vec!["[TOOL_CALLS]".to_string()],
                tool_call_end_tokens: vec!["".to_string()],
                ..Default::default()
            },
        }
    }

    pub fn phi4() -> Self {
        Self {
            format: ToolCallParserType::Json,
            json: JsonParserConfig {
                tool_call_start_tokens: vec!["functools".to_string()],
                tool_call_end_tokens: vec!["".to_string()],
                ..Default::default()
            },
        }
    }
}

/// Configuration for parsing tool calls with different formats
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct ToolCallConfig {
    /// The format type for tool calls
    pub format: ToolCallParserType,
    /// The config for the JSON parser
    pub json: JsonParserConfig,
}

pub fn try_tool_call_parse(
    message: &str,
    config: &ToolCallConfig,
) -> anyhow::Result<Vec<ToolCallResponse>> {
    // Use match statement (Rust's switch statement) to call the appropriate parser
    match config.format {
        ToolCallParserType::Json => try_tool_call_parse_json(message, &config.json),
        ToolCallParserType::Harmony => {
            anyhow::bail!("Harmony parser not implemented");
        }
        ToolCallParserType::Pythonic => {
            anyhow::bail!("Pythonic parser not implemented");
        }
        ToolCallParserType::Typescript => {
            anyhow::bail!("Typescript parser not implemented");
        }
        ToolCallParserType::Xml => {
            anyhow::bail!("Xml parser not implemented");
        }
    }
}

// Base Detector to call for all tool parsing
pub fn detect_and_parse_tool_call(
    message: &str,
    parser_str: Option<&str>,
) -> anyhow::Result<Vec<ToolCallResponse>> {
    let mut parser_map: std::collections::HashMap<&str, ToolCallConfig> =
        std::collections::HashMap::new();
    parser_map.insert("hermes", ToolCallConfig::hermes());
    parser_map.insert("nemotron_deci", ToolCallConfig::nemotron_deci());
    parser_map.insert("llama3_json", ToolCallConfig::llama3_json());
    parser_map.insert("mistral", ToolCallConfig::mistral());
    parser_map.insert("phi4", ToolCallConfig::phi4());
    parser_map.insert("default", ToolCallConfig::default()); // Add default key

    // Handle None or empty string by defaulting to "default"
    let parser_key = match parser_str {
        Some(s) if !s.is_empty() => s,
        _ => "default", // None or empty string
    };

    match parser_map.get(parser_key) {
        Some(config) => try_tool_call_parse(message, config),
        None => anyhow::bail!("Parser for the given config is not implemented"), // Original message
    }
}

// Tests
// cargo test postprocessor::tool_calling::parsers
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
        let result = try_tool_call_parse(input, &ToolCallConfig::default()).unwrap();
        assert!(!result.is_empty());
        assert_eq!(result.len(), 1);
        let (name, args) = extract_name_and_args(result[0].clone());
        assert_eq!(name, "hello");
        assert_eq!(args["x"], 1);
        assert_eq!(args["y"], 2);
    }

    #[test]
    fn parses_single_arguments_object() {
        let input = r#"{ "name": "world", "arguments": { "a": "abc", "b": 42 } }"#;
        let result = try_tool_call_parse(input, &ToolCallConfig::default()).unwrap();
        assert!(!result.is_empty());
        assert_eq!(result.len(), 1);
        let (name, args) = extract_name_and_args(result[0].clone());
        assert_eq!(name, "world");
        assert_eq!(args["a"], "abc");
        assert_eq!(args["b"], 42);
    }

    #[test]
    fn parses_vec_of_parameters() {
        let input = r#"[{ "name": "first", "parameters": { "a": 1 } }, { "name": "second", "parameters": { "b": 2 } }]"#;
        let result = try_tool_call_parse(input, &ToolCallConfig::default()).unwrap();
        assert!(!result.is_empty());
        assert_eq!(result.len(), 2);
        let (name, args) = extract_name_and_args(result[0].clone());
        assert_eq!(name, "first");
        assert_eq!(args["a"], 1);
        let (name, args) = extract_name_and_args(result[1].clone());
        assert_eq!(name, "second");
        assert_eq!(args["b"], 2);
    }

    #[test]
    fn parses_vec_of_arguments() {
        let input = r#"[{ "name": "alpha", "arguments": { "a": "x" } }, { "name": "omega", "arguments": { "z": "y" } }]"#;
        let result = try_tool_call_parse(input, &ToolCallConfig::default()).unwrap();
        assert!(!result.is_empty());
        assert_eq!(result.len(), 2);
        let (name, args) = extract_name_and_args(result[0].clone());
        assert_eq!(name, "alpha");
        assert_eq!(args["a"], "x");
        let (name, args) = extract_name_and_args(result[1].clone());
        assert_eq!(name, "omega");
        assert_eq!(args["z"], "y");
    }

    #[test]
    fn parses_toolcall_wrapped_payload() {
        let input =
            r#"<TOOLCALL>[{ "name": "wrapped", "parameters": { "foo": "bar" } }]</TOOLCALL>"#;
        let result = try_tool_call_parse(input, &ToolCallConfig::default()).unwrap();
        assert!(!result.is_empty());
        assert_eq!(result.len(), 1);
        let (name, args) = extract_name_and_args(result[0].clone());
        assert_eq!(name, "wrapped");
        assert_eq!(args["foo"], "bar");
    }

    #[test]
    fn parses_python_tag_prefixed_payload() {
        let input = r#"<|python_tag|>{ "name": "pyfunc", "arguments": { "k": "v" } }"#;
        let result = try_tool_call_parse(
            input,
            &ToolCallConfig {
                format: ToolCallParserType::Json,
                json: JsonParserConfig {
                    tool_call_start_tokens: vec!["<|python_tag|>".to_string()],
                    tool_call_end_tokens: vec!["".to_string()],
                    ..Default::default()
                },
            },
        )
        .unwrap();
        assert!(!result.is_empty());
        assert_eq!(result.len(), 1);
        let (name, args) = extract_name_and_args(result[0].clone());
        assert_eq!(name, "pyfunc");
        assert_eq!(args["k"], "v");
    }

    #[test]
    fn returns_none_on_invalid_input() {
        let input = r#"not even json"#;
        let result = try_tool_call_parse(input, &ToolCallConfig::default()).unwrap();
        assert!(result.is_empty());
    }

    #[test]
    fn returns_none_on_valid_json_wrong_shape() {
        let input = r#"{ "foo": "bar" }"#;
        let result = try_tool_call_parse(input, &ToolCallConfig::default()).unwrap();
        assert!(result.is_empty());
    }

    // Tests for real model outputs - disabled by default
    #[test]
    fn test_nvidia_llama3_nemotron_super_49b_simple() {
        let input = r#"<think>
Okay, the user is asking for the weather in San Francisco in Fahrenheit. Let me check the tools available.
</think>

<TOOLCALL>[{"name": "get_weather", "arguments": {"location": "San Francisco, CA", "unit": "fahrenheit"}}]</TOOLCALL>"#;
        let result = detect_and_parse_tool_call(input, Some("nemotron_deci")).unwrap();
        assert!(!result.is_empty());
        assert_eq!(result.len(), 1);
        let (name, args) = extract_name_and_args(result[0].clone());
        assert_eq!(name, "get_weather");
        assert_eq!(args["location"], "San Francisco, CA");
        assert_eq!(args["unit"], "fahrenheit");
    }

    #[test]
    fn test_nvidia_llama3_nemotron_super_49b_with_function_array() {
        let input = r#"<think>
Okay, the user is asking for the weather in San Francisco in Fahrenheit. Let me check the tools available.
</think>

<TOOLCALL>[{"name": "get_weather", "arguments": {"location": "San Francisco, CA", "unit": "fahrenheit"}}, {"name": "get_weather", "arguments": {"location": "New York, NY", "unit": "fahrenheit"}}]</TOOLCALL>"#;
        let config = ToolCallConfig::nemotron_deci();
        let result = try_tool_call_parse(input, &config).unwrap();
        assert!(!result.is_empty());
        assert_eq!(result.len(), 2);
        let (name, args) = extract_name_and_args(result[0].clone());
        assert_eq!(name, "get_weather");
        assert_eq!(args["location"], "San Francisco, CA");
        assert_eq!(args["unit"], "fahrenheit");
        let (name, args) = extract_name_and_args(result[1].clone());
        assert_eq!(name, "get_weather");
        assert_eq!(args["location"], "New York, NY");
        assert_eq!(args["unit"], "fahrenheit");
    }

    #[test]
    fn test_nvidia_llama3_nemotron_super_49b_with_function_array_with_new_lines() {
        let input = r#"<think>
Okay, the user is asking for the weather in San Francisco in Fahrenheit. Let me check the tools available.
</think>

<TOOLCALL>
[{"name": "get_weather",
 "arguments": {"location": "San Francisco, CA",
  "unit": "fahrenheit"}},
  {"name": "get_weather",
   "arguments":
  {"location": "New York, NY",
  "unit": "fahrenheit"}}]
  </TOOLCALL>
  "#;
        let config = ToolCallConfig::nemotron_deci();
        let result = try_tool_call_parse(input, &config).unwrap();
        assert!(!result.is_empty());
        assert_eq!(result.len(), 2);
        let (name, args) = extract_name_and_args(result[0].clone());
        assert_eq!(name, "get_weather");
        assert_eq!(args["location"], "San Francisco, CA");
        assert_eq!(args["unit"], "fahrenheit");
        let (name, args) = extract_name_and_args(result[1].clone());
        assert_eq!(name, "get_weather");
        assert_eq!(args["location"], "New York, NY");
        assert_eq!(args["unit"], "fahrenheit");
    }

    #[test]
    fn test_qwen_qwq_32b_simple() {
        let input = r#"<tool_call>
{"name": "get_weather", "arguments": {"location": "San Francisco, CA", "unit": "fahrenheit"}}
</tool_call>"#;
        let result = detect_and_parse_tool_call(input, Some("hermes")).unwrap();
        assert!(!result.is_empty());
        assert_eq!(result.len(), 1);
        let (name, args) = extract_name_and_args(result[0].clone());
        assert_eq!(name, "get_weather");
        assert_eq!(args["location"], "San Francisco, CA");
        assert_eq!(args["unit"], "fahrenheit");
    }

    #[test]
    fn test_nousresearch_hermes3_llama31_8b_simple() {
        let input = r#"<tool_call>
{"name": "get_weather", "arguments": {"location": "San Francisco, CA", "unit": "fahrenheit"}}
</tool_call>"#;
        let result = detect_and_parse_tool_call(input, Some("hermes")).unwrap();
        assert!(!result.is_empty());
        assert_eq!(result.len(), 1);
        let (name, args) = extract_name_and_args(result[0].clone());
        assert_eq!(name, "get_weather");
        assert_eq!(args["location"], "San Francisco, CA");
        assert_eq!(args["unit"], "fahrenheit");
    }

    #[test]
    fn test_qwen_qwq_32b_multiple_tool_calls() {
        let input = r#"<tool_call>
{"name": "get_weather", "arguments": {"location": "San Francisco, CA", "unit": "fahrenheit"}}
</tool_call>
<tool_call>
{"name": "get_weather", "arguments": {"location": "New York, NY", "unit": "fahrenheit"}}
</tool_call>
"#;
        let config = ToolCallConfig::hermes();
        let result = try_tool_call_parse(input, &config).unwrap();
        assert!(!result.is_empty());
        assert_eq!(result.len(), 2);
        let (name, args) = extract_name_and_args(result[0].clone());
        assert_eq!(name, "get_weather");
        assert_eq!(args["location"], "San Francisco, CA");
        assert_eq!(args["unit"], "fahrenheit");
        let (name, args) = extract_name_and_args(result[1].clone());
        assert_eq!(name, "get_weather");
        assert_eq!(args["location"], "New York, NY");
        assert_eq!(args["unit"], "fahrenheit");
    }

    #[test]
    fn test_qwen_qwq_32b_multiple_tool_calls_with_new_lines() {
        let input = r#"<tool_call>
{"name": "get_weather",
"arguments": {"location": "San Francisco, CA",
"unit": "fahrenheit"}}
</tool_call>
<tool_call>
{"name": "get_weather", "arguments":
{"location": "New York, NY", "unit":
"fahrenheit"}}
</tool_call>
"#;
        let config = ToolCallConfig::hermes();
        let result = try_tool_call_parse(input, &config).unwrap();
        assert!(!result.is_empty());
        assert_eq!(result.len(), 2);
        let (name, args) = extract_name_and_args(result[0].clone());
        assert_eq!(name, "get_weather");
        assert_eq!(args["location"], "San Francisco, CA");
        assert_eq!(args["unit"], "fahrenheit");
        let (name, args) = extract_name_and_args(result[1].clone());
        assert_eq!(name, "get_weather");
        assert_eq!(args["location"], "New York, NY");
        assert_eq!(args["unit"], "fahrenheit");
    }

    #[test]
    #[ignore]
    fn test_ibm_granite_40_tiny_preview_simple() {
        let input = r#"[{"arguments": {"location": "San Francisco, CA", "unit": "fahrenheit"}, "name": "get_weather"}]"#;
        let config = ToolCallConfig {
            format: ToolCallParserType::Json,
            json: JsonParserConfig {
                tool_call_start_tokens: vec![],
                tool_call_end_tokens: vec![],
                arguments_keys: vec!["arguments".to_string()],
                ..Default::default()
            },
        };
        let result = try_tool_call_parse(input, &config).unwrap();
        assert!(!result.is_empty());
        assert_eq!(result.len(), 1);
        let (name, args) = extract_name_and_args(result[0].clone());
        assert_eq!(name, "get_weather");
        assert_eq!(args["location"], "San Francisco, CA");
        assert_eq!(args["unit"], "fahrenheit");
    }

    #[test]
    fn test_mistralai_mistral_7b_instruct_v03_simple() {
        let input = r#" [{"name": "get_weather", "arguments": {"location": "San Francisco, CA", "unit": "fahrenheit"}}]"#;
        let config = ToolCallConfig {
            format: ToolCallParserType::Json,
            json: JsonParserConfig {
                tool_call_start_tokens: vec![],
                tool_call_end_tokens: vec![],
                arguments_keys: vec!["arguments".to_string()],
                ..Default::default()
            },
        };
        let result = try_tool_call_parse(input, &config).unwrap();
        assert!(!result.is_empty());
        assert_eq!(result.len(), 1);
        let (name, args) = extract_name_and_args(result[0].clone());
        assert_eq!(name, "get_weather");
        assert_eq!(args["location"], "San Francisco, CA");
        assert_eq!(args["unit"], "fahrenheit");
    }

    #[test]
    fn test_mistralai_mistral_7b_instruct_v03_simple_with_new_lines() {
        let input = r#"
        [{"name": "get_weather",
        "arguments": {"location":
        "San Francisco, CA",
        "unit": "fahrenheit"}}]
        "#;
        let config = ToolCallConfig {
            format: ToolCallParserType::Json,
            json: JsonParserConfig {
                tool_call_start_tokens: vec![],
                tool_call_end_tokens: vec![],
                arguments_keys: vec!["arguments".to_string()],
                ..Default::default()
            },
        };
        let result = try_tool_call_parse(input, &config).unwrap();
        assert!(!result.is_empty());
        assert_eq!(result.len(), 1);
        let (name, args) = extract_name_and_args(result[0].clone());
        assert_eq!(name, "get_weather");
        assert_eq!(args["location"], "San Francisco, CA");
        assert_eq!(args["unit"], "fahrenheit");
    }

    #[test]
    fn test_mistralai_mistral_7b_instruct_v03_multiple() {
        let input = r#" [{"name": "get_weather", "arguments": {"location": "San Francisco, CA", "unit": "fahrenheit"}}, {"name": "get_weather", "arguments": {"location": "New York, NY", "unit": "fahrenheit"}}]"#;
        let config = ToolCallConfig::mistral();
        let result = try_tool_call_parse(input, &config).unwrap();
        assert!(!result.is_empty());
        assert_eq!(result.len(), 2);
        let (name, args) = extract_name_and_args(result[0].clone());
        assert_eq!(name, "get_weather");
        assert_eq!(args["location"], "San Francisco, CA");
        assert_eq!(args["unit"], "fahrenheit");
        let (name, args) = extract_name_and_args(result[1].clone());
        assert_eq!(name, "get_weather");
        assert_eq!(args["location"], "New York, NY");
        assert_eq!(args["unit"], "fahrenheit");
    }

    #[test]
    fn test_mistralai_mistral_7b_instruct_v03_multiple_with_new_lines() {
        let input = r#"
        [{"name": "get_weather",
        "arguments": {"location": "San Francisco, CA",
        "unit": "fahrenheit"}}, {"name": "get_weather", "arguments":
        {"location": "New York, NY", "unit": "fahrenheit"}}]
        "#;
        let config = ToolCallConfig::mistral();
        let result = try_tool_call_parse(input, &config).unwrap();
        assert!(!result.is_empty());
        assert_eq!(result.len(), 2);
        let (name, args) = extract_name_and_args(result[0].clone());
        assert_eq!(name, "get_weather");
        assert_eq!(args["location"], "San Francisco, CA");
        assert_eq!(args["unit"], "fahrenheit");
        let (name, args) = extract_name_and_args(result[1].clone());
        assert_eq!(name, "get_weather");
        assert_eq!(args["location"], "New York, NY");
        assert_eq!(args["unit"], "fahrenheit");
    }

    #[test]
    fn test_mistralai_mistral_7b_instruct_v03_single_with_start_token() {
        let input = r#"[TOOL_CALLS] [{"name": "get_weather", "arguments": {"location": "San Francisco, CA", "unit": "fahrenheit"}}]"#;
        let config = ToolCallConfig::mistral();
        let result = try_tool_call_parse(input, &config).unwrap();
        assert!(!result.is_empty());
        assert_eq!(result.len(), 1);
        let (name, args) = extract_name_and_args(result[0].clone());
        assert_eq!(name, "get_weather");
        assert_eq!(args["location"], "San Francisco, CA");
        assert_eq!(args["unit"], "fahrenheit");
    }

    #[test]
    fn test_mistralai_mistral_7b_instruct_v03_single_with_start_tokenwith_new_lines() {
        let input = r#"
        [TOOL_CALLS]
        [{"name": "get_weather",
        "arguments": {"location":
        "San Francisco, CA",
        "unit": "fahrenheit"}}]
        "#;
        let config = ToolCallConfig::mistral();
        let result = try_tool_call_parse(input, &config).unwrap();
        assert!(!result.is_empty());
        assert_eq!(result.len(), 1);
        let (name, args) = extract_name_and_args(result[0].clone());
        assert_eq!(name, "get_weather");
        assert_eq!(args["location"], "San Francisco, CA");
        assert_eq!(args["unit"], "fahrenheit");
    }

    #[test]
    fn test_mistralai_mistral_7b_instruct_v03_single_with_start_token_multiple() {
        let input = r#"[TOOL_CALLS] [{"name": "get_weather", "arguments": {"location": "San Francisco, CA", "unit": "fahrenheit"}}, {"name": "get_weather", "arguments": {"location": "New York, NY", "unit": "fahrenheit"}}]"#;
        let config = ToolCallConfig::mistral();
        let result = try_tool_call_parse(input, &config).unwrap();
        assert!(!result.is_empty());
        assert_eq!(result.len(), 2);
        let (name, args) = extract_name_and_args(result[0].clone());
        assert_eq!(name, "get_weather");
        assert_eq!(args["location"], "San Francisco, CA");
        assert_eq!(args["unit"], "fahrenheit");
        let (name, args) = extract_name_and_args(result[1].clone());
        assert_eq!(name, "get_weather");
        assert_eq!(args["location"], "New York, NY");
        assert_eq!(args["unit"], "fahrenheit");
    }

    #[test]
    fn test_mistralai_mistral_7b_instruct_v03_single_with_start_token_multiple_with_new_lines() {
        let input = r#"
        [TOOL_CALLS]
        [{"name": "get_weather",
        "arguments": {"location":
        "San Francisco, CA",
        "unit": "fahrenheit"}},
        {"name": "get_weather", "arguments":
        {"location": "New York, NY", "unit":
        "fahrenheit"}}]
        "#;
        let config = ToolCallConfig::mistral();
        let result = try_tool_call_parse(input, &config).unwrap();
        assert!(!result.is_empty());
        assert_eq!(result.len(), 2);
        let (name, args) = extract_name_and_args(result[0].clone());
        assert_eq!(name, "get_weather");
        assert_eq!(args["location"], "San Francisco, CA");
        assert_eq!(args["unit"], "fahrenheit");
        let (name, args) = extract_name_and_args(result[1].clone());
        assert_eq!(name, "get_weather");
        assert_eq!(args["location"], "New York, NY");
        assert_eq!(args["unit"], "fahrenheit");
    }

    #[test]
    fn test_meta_llama_llama31_8b_instruct_simple() {
        let input = r#"{"name": "get_weather", "parameters": {"location": "San Francisco, CA", "unit": "fahrenheit"}}"#;
        let result = detect_and_parse_tool_call(input, Some("llama3_json")).unwrap();
        assert!(!result.is_empty());
        assert_eq!(result.len(), 1);
        let (name, args) = extract_name_and_args(result[0].clone());
        assert_eq!(name, "get_weather");
        assert_eq!(args["location"], "San Francisco, CA");
        assert_eq!(args["unit"], "fahrenheit");
    }

    #[test]
    fn test_meta_llama_llama31_8b_instruct_with_new_lines() {
        let input = r#"
        {"name": "get_weather",
        "parameters": {"location": "San Francisco, CA", "unit": "fahrenheit"}}
        "#;
        let result = detect_and_parse_tool_call(input, Some("llama3_json")).unwrap();
        assert!(!result.is_empty());
        assert_eq!(result.len(), 1);
        let (name, args) = extract_name_and_args(result[0].clone());
        assert_eq!(name, "get_weather");
        assert_eq!(args["location"], "San Francisco, CA");
        assert_eq!(args["unit"], "fahrenheit");
    }

    #[test]
    fn test_meta_llama_llama31_8b_instruct_with_python_tag() {
        let input = r#"<|python_tag|>{ "name": "get_weather", "parameters": {"location": "San Francisco, CA", "unit": "fahrenheit" } }"#;
        let result = detect_and_parse_tool_call(input, Some("llama3_json")).unwrap();
        assert!(!result.is_empty());
        assert_eq!(result.len(), 1);
        let (name, args) = extract_name_and_args(result[0].clone());
        assert_eq!(name, "get_weather");
        assert_eq!(args["location"], "San Francisco, CA");
        assert_eq!(args["unit"], "fahrenheit");
    }

    #[test]
    fn test_meta_llama_llama31_8b_instruct_with_python_tag_with_new_lines() {
        let input = r#"
        <|python_tag|>
        {"name": "get_weather", "parameters": {"location": "San Francisco, CA", "unit": "fahrenheit"}}
        "#;
        let result = detect_and_parse_tool_call(input, Some("llama3_json")).unwrap();
        assert!(!result.is_empty());
        assert_eq!(result.len(), 1);
        let (name, args) = extract_name_and_args(result[0].clone());
        assert_eq!(name, "get_weather");
        assert_eq!(args["location"], "San Francisco, CA");
        assert_eq!(args["unit"], "fahrenheit");
    }

    #[test]
    fn test_meta_llama_llama31_8b_instruct_with_python_tag_multiple_with_new_lines() {
        let input = r#"
        <|python_tag|>
        {"name": "get_weather", "parameters": {"location": "San Francisco, CA", "unit": "fahrenheit" }}
        <|python_tag|>
        {"name": "get_weather", "parameters": {"location": "New York, NY", "unit": "fahrenheit" }}
        "#;
        let result = detect_and_parse_tool_call(input, Some("llama3_json")).unwrap();
        assert!(!result.is_empty());
        assert_eq!(result.len(), 2);
        let (name, args) = extract_name_and_args(result[0].clone());
        assert_eq!(name, "get_weather");
        assert_eq!(args["location"], "San Francisco, CA");
        assert_eq!(args["unit"], "fahrenheit");
        let (name, args) = extract_name_and_args(result[1].clone());
        assert_eq!(name, "get_weather");
        assert_eq!(args["location"], "New York, NY");
        assert_eq!(args["unit"], "fahrenheit");
    }

    #[test]
    fn test_detect_and_parse_tool_call_error_handling() {
        // Unknown parser string should return an error
        let input = r#"{"name": "get_weather", "arguments": {"location": "San Francisco, CA"}}"#;
        let result = detect_and_parse_tool_call(input, Some("unknown_parser"));
        assert!(result.is_err());
        let err = result.unwrap_err().to_string();
        assert!(
            err.contains("is not implemented"),
            "Unexpected error message: {}",
            err
        );

        // Known parser, but invalid input (not JSON) should return Ok(None)
        let input = "not a json";
        let result = detect_and_parse_tool_call(input, Some("hermes"));
        assert!(result.is_ok());
        assert!(result.unwrap().is_empty());

        // Known parser, but valid JSON with wrong shape should return Ok(None)
        let input = r#"{"foo": "bar"}"#;
        let result = detect_and_parse_tool_call(input, Some("hermes"));
        assert!(result.is_ok());
        assert!(result.unwrap().is_empty());
    }

    #[test]
    #[ignore]
    fn test_internlm_internlm2_5_7b_chat_simple() {
        let input = r#"San Francisco's weather is known for its mild climate with plenty of fog, especially along the coast. Here's an overview of the weather in Fahrenheit:

- **Summer (June to August)**: Average highs range from the mid-60s to low 70s Fahrenheit, with cooler mornings and evenings. Coastal areas may be cooler than inland spots.

Remember, San Francisco weather can be quite unpredictable, particularly with its famous fog, which can significantly lower temperatures. Always check a local weather forecast for the most accurate and up-to-date information."#;
        let result = try_tool_call_parse(input, &ToolCallConfig::default()).unwrap();
        assert!(result.is_empty()); // This model doesn't produce tool calls
    }

    #[test]
    #[ignore]
    fn test_ai21labs_ai21_jamba_15_mini_simple() {
        let input = r#" [
    {"name": "get_weather", "arguments": {"location": "San Francisco, CA", "unit": "fahrenheit"}}
]"#;
        let config = ToolCallConfig {
            format: ToolCallParserType::Json,
            json: JsonParserConfig {
                tool_call_start_tokens: vec![],
                tool_call_end_tokens: vec![],
                arguments_keys: vec!["arguments".to_string()],
                ..Default::default()
            },
        };
        let result = try_tool_call_parse(input, &config).unwrap();
        assert!(!result.is_empty());
        assert_eq!(result.len(), 1);
        let (name, args) = extract_name_and_args(result[0].clone());
        assert_eq!(name, "get_weather");
        assert_eq!(args["location"], "San Francisco, CA");
        assert_eq!(args["unit"], "fahrenheit");
    }

    #[test]
    #[ignore]
    fn test_salesforce_llama_xlam_2_8b_fc_r_simple() {
        let input = r#"[{"name": "get_weather", "arguments": {"location": "San Francisco, CA", "unit": "fahrenheit"}}]"#;
        let config = ToolCallConfig {
            format: ToolCallParserType::Json,
            json: JsonParserConfig {
                tool_call_start_tokens: vec![],
                tool_call_end_tokens: vec![],
                arguments_keys: vec!["arguments".to_string()],
                ..Default::default()
            },
        };
        let result = try_tool_call_parse(input, &config).unwrap();
        assert!(!result.is_empty());
        assert_eq!(result.len(), 1);
        let (name, args) = extract_name_and_args(result[0].clone());
        assert_eq!(name, "get_weather");
        assert_eq!(args["location"], "San Francisco, CA");
        assert_eq!(args["unit"], "fahrenheit");
    }

    #[test]
    fn test_detect_and_parse_tool_call_default_parser_nemotron_deci() {
        let input = r#"<TOOLCALL>[{"name": "get_weather", "arguments": {"location": "San Francisco, CA", "unit": "fahrenheit"}}]</TOOLCALL>"#;
        let result = detect_and_parse_tool_call(input, None).unwrap();
        assert!(!result.is_empty());
        assert_eq!(result.len(), 1);
        let (name, args) = extract_name_and_args(result[0].clone());
        assert_eq!(name, "get_weather");
        assert_eq!(args["location"], "San Francisco, CA");
        assert_eq!(args["unit"], "fahrenheit");
    }

    #[test]
    fn test_detect_and_parse_tool_call_default_parser_nemotron_deci_multiple() {
        let input = r#"<TOOLCALL>[{"name": "get_weather", "arguments": {"location": "San Francisco, CA", "unit": "fahrenheit"}}, {"name": "get_weather", "arguments": {"location": "New York, NY", "unit": "fahrenheit"}}]</TOOLCALL>"#;
        let result = detect_and_parse_tool_call(input, None).unwrap();
        assert!(!result.is_empty());
        assert_eq!(result.len(), 2);
        let (name, args) = extract_name_and_args(result[0].clone());
        assert_eq!(name, "get_weather");
        assert_eq!(args["location"], "San Francisco, CA");
        assert_eq!(args["unit"], "fahrenheit");
        let (name, args) = extract_name_and_args(result[1].clone());
        assert_eq!(name, "get_weather");
        assert_eq!(args["location"], "New York, NY");
        assert_eq!(args["unit"], "fahrenheit");
    }

    #[test]
    fn test_detect_and_parse_tool_call_default_parser_llama3_json_with_python_tag() {
        let input = r#"<|python_tag|>{ "name": "get_weather", "arguments": {"location": "San Francisco, CA", "unit": "fahrenheit" } }"#;
        let result = detect_and_parse_tool_call(input, None).unwrap();
        assert!(!result.is_empty());
        assert_eq!(result.len(), 1);
        let (name, args) = extract_name_and_args(result[0].clone());
        assert_eq!(name, "get_weather");
        assert_eq!(args["location"], "San Francisco, CA");
        assert_eq!(args["unit"], "fahrenheit");
    }

    #[test]
    fn test_detect_and_parse_tool_call_default_parser_llama3_json_with_python_tag_with_new_lines() {
        let input = r#"
        <|python_tag|>
        {"name":
        "get_weather",
         "arguments":
          {"location": "San Francisco, CA",
          "unit": "fahrenheit" }}
        "#;
        let result = detect_and_parse_tool_call(input, None).unwrap();
        assert!(!result.is_empty());
        assert_eq!(result.len(), 1);
        let (name, args) = extract_name_and_args(result[0].clone());
        assert_eq!(name, "get_weather");
        assert_eq!(args["location"], "San Francisco, CA");
        assert_eq!(args["unit"], "fahrenheit");
    }

    #[test]
    fn test_detect_and_parse_tool_call_default_parser_llama3_json_without_python_tag_multiple_with_new_lines()
     {
        let input = r#"
        {"name": "get_weather", "arguments":
         {"location": "San Francisco, CA",
          "unit": "fahrenheit" }}
        "#;
        let result = detect_and_parse_tool_call(input, None).unwrap();
        assert!(!result.is_empty());
        assert_eq!(result.len(), 1);
        let (name, args) = extract_name_and_args(result[0].clone());
        assert_eq!(name, "get_weather");
        assert_eq!(args["location"], "San Francisco, CA");
        assert_eq!(args["unit"], "fahrenheit");
    }

    #[test]
    fn test_detect_and_parse_tool_call_default_parser_llama3_json_without_python_tag() {
        let input = r#"{ "name": "get_weather", "arguments": {"location": "San Francisco, CA", "unit": "fahrenheit" } }"#;
        let result = detect_and_parse_tool_call(input, None).unwrap();
        assert!(!result.is_empty());
        assert_eq!(result.len(), 1);
        let (name, args) = extract_name_and_args(result[0].clone());
        assert_eq!(name, "get_weather");
        assert_eq!(args["location"], "San Francisco, CA");
        assert_eq!(args["unit"], "fahrenheit");
    }

    #[test]
    fn test_phi4_single_function_call() {
        let input =
            r#"functools[{"name": "get_country_capital", "arguments": {"country": "Poland"}}]"#;
        let result = detect_and_parse_tool_call(input, Some("phi4")).unwrap();
        assert_eq!(result.len(), 1);
        let (name, args) = extract_name_and_args(result[0].clone());
        assert_eq!(name, "get_country_capital");
        assert_eq!(args["country"], "Poland");
    }

    #[test]
    fn test_phi4_multiple_function_calls_simple_arguments() {
        let input = r#"functools[
  {"name": "get_country_capital", "arguments": {"country": "Poland"}},
  {"name": "get_population", "arguments": {"city": "Warsaw"}}
]"#;
        let result = detect_and_parse_tool_call(input, Some("phi4")).unwrap();
        assert_eq!(result.len(), 2);

        let (name1, args1) = extract_name_and_args(result[0].clone());
        assert_eq!(name1, "get_country_capital");
        assert_eq!(args1["country"], "Poland");

        let (name2, args2) = extract_name_and_args(result[1].clone());
        assert_eq!(name2, "get_population");
        assert_eq!(args2["city"], "Warsaw");
    }

    #[test]
    fn test_phi4_single_function_call_nested_json_arguments() {
        let input = r#"functools[{"name": "get_weather_forecast", "arguments":
        {"location": {"city": "San Francisco",
        "state": "CA"}, "date": "2023-10-05"}}]"#;
        let result = detect_and_parse_tool_call(input, Some("phi4")).unwrap();
        assert_eq!(result.len(), 1);
        let (name, args) = extract_name_and_args(result[0].clone());
        assert_eq!(name, "get_weather_forecast");
        assert_eq!(args["date"], "2023-10-05");
        assert_eq!(args["location"]["city"], "San Francisco");
        assert_eq!(args["location"]["state"], "CA");
    }

    #[test]
    fn test_phi4_function_call_with_parameters_instead_of_arguments() {
        let input = r#"functools[{"name": "calculate_distance",
         "parameters": {"from": "New York", "to": "Los Angeles"}}]"#;
        let result = detect_and_parse_tool_call(input, Some("phi4")).unwrap();
        assert_eq!(result.len(), 1);
        let (name, args) = extract_name_and_args(result[0].clone());
        assert_eq!(name, "calculate_distance");
        assert_eq!(args["from"], "New York");
        assert_eq!(args["to"], "Los Angeles");
    }
}
