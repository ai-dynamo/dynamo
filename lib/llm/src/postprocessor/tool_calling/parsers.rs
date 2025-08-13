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
            tool_call_end_tokens: vec!["</TOOLCALL>".to_string()],
            function_name_keys: vec!["name".to_string()],
            arguments_keys: vec!["arguments".to_string(), "parameters".to_string()],
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

impl Default for ToolCallConfig {
    fn default() -> Self {
        Self {
            format: ToolCallParserType::Json,
            json: JsonParserConfig::default(),
        }
    }
}

pub fn try_tool_call_parse(
    message: &str,
    config: &ToolCallConfig,
) -> anyhow::Result<Option<ToolCallResponse>> {
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
        let result = try_tool_call_parse(input, &ToolCallConfig::default())
            .unwrap()
            .unwrap();
        let (name, args) = extract_name_and_args(result);
        assert_eq!(name, "hello");
        assert_eq!(args["x"], 1);
        assert_eq!(args["y"], 2);
    }

    #[test]
    fn parses_single_arguments_object() {
        let input = r#"{ "name": "world", "arguments": { "a": "abc", "b": 42 } }"#;
        let result = try_tool_call_parse(input, &ToolCallConfig::default())
            .unwrap()
            .unwrap();
        let (name, args) = extract_name_and_args(result);
        assert_eq!(name, "world");
        assert_eq!(args["a"], "abc");
        assert_eq!(args["b"], 42);
    }

    #[test]
    fn parses_vec_of_parameters_and_takes_last() {
        let input = r#"[{ "name": "first", "parameters": { "a": 1 } }, { "name": "second", "parameters": { "b": 2 } }]"#;
        let result = try_tool_call_parse(input, &ToolCallConfig::default())
            .unwrap()
            .unwrap();
        let (name, args) = extract_name_and_args(result);
        assert_eq!(name, "second");
        assert_eq!(args["b"], 2);
    }

    #[test]
    fn parses_vec_of_arguments_and_takes_last() {
        let input = r#"[{ "name": "alpha", "arguments": { "a": "x" } }, { "name": "omega", "arguments": { "z": "y" } }]"#;
        let result = try_tool_call_parse(input, &ToolCallConfig::default())
            .unwrap()
            .unwrap();
        let (name, args) = extract_name_and_args(result);
        assert_eq!(name, "omega");
        assert_eq!(args["z"], "y");
    }

    #[test]
    fn parses_toolcall_wrapped_payload() {
        let input =
            r#"<TOOLCALL>[{ "name": "wrapped", "parameters": { "foo": "bar" } }]</TOOLCALL>"#;
        let result = try_tool_call_parse(input, &ToolCallConfig::default())
            .unwrap()
            .unwrap();
        let (name, args) = extract_name_and_args(result);
        assert_eq!(name, "wrapped");
        assert_eq!(args["foo"], "bar");
    }

    #[test]
    fn parses_python_tag_prefixed_payload() {
        let input = r#"<|python_tag|>{ "name": "pyfunc", "arguments": { "k": "v" } }"#;
        let result = try_tool_call_parse(input, &ToolCallConfig::default())
            .unwrap()
            .unwrap();
        let (name, args) = extract_name_and_args(result);
        assert_eq!(name, "pyfunc");
        assert_eq!(args["k"], "v");
    }

    #[test]
    fn returns_none_on_invalid_input() {
        let input = r#"not even json"#;
        let result = try_tool_call_parse(input, &ToolCallConfig::default()).unwrap();
        assert!(result.is_none());
    }

    #[test]
    fn returns_none_on_valid_json_wrong_shape() {
        let input = r#"{ "foo": "bar" }"#;
        let result = try_tool_call_parse(input, &ToolCallConfig::default()).unwrap();
        assert!(result.is_none());
    }

    // Tests for real model outputs - disabled by default
    #[test]
    #[ignore]
    fn test_nvidia_llama3_nemotron_super_49b_simple() {
        let input = r#"<think>
Okay, the user is asking for the weather in San Francisco in Fahrenheit. Let me check the tools available.
</think>

<TOOLCALL>[{"name": "get_weather", "arguments": {"location": "San Francisco, CA", "unit": "fahrenheit"}}]</TOOLCALL>"#;
        let result = try_tool_call_parse(input, &ToolCallConfig::default())
            .unwrap()
            .unwrap();
        let (name, args) = extract_name_and_args(result);
        assert_eq!(name, "get_weather");
        assert_eq!(args["location"], "San Francisco, CA");
        assert_eq!(args["unit"], "fahrenheit");
    }

    #[test]
    #[ignore]
    fn test_qwen_qwq_32b_simple() {
        let input = r#"<tool_call>
{"name": "get_weather", "arguments": {"location": "San Francisco, CA", "unit": "fahrenheit"}}
</tool_call>"#;
        let config = ToolCallConfig {
            format: ToolCallParserType::Json,
            json: JsonParserConfig {
                tool_call_start_tokens: vec!["<tool_call>".to_string()],
                tool_call_end_tokens: vec!["</tool_call>".to_string()],
                arguments_keys: vec!["arguments".to_string()],
                ..Default::default()
            },
        };
        let result = try_tool_call_parse(input, &config).unwrap().unwrap();
        let (name, args) = extract_name_and_args(result);
        assert_eq!(name, "get_weather");
        assert_eq!(args["location"], "San Francisco, CA");
        assert_eq!(args["unit"], "fahrenheit");
    }

    #[test]
    #[ignore]
    fn test_nousresearch_hermes3_llama31_8b_simple() {
        let input = r#"<tool_call>
{"name": "get_weather", "arguments": {"location": "San Francisco, CA", "unit": "fahrenheit"}}
</tool_call>"#;
        let config = ToolCallConfig {
            format: ToolCallParserType::Json,
            json: JsonParserConfig {
                tool_call_start_tokens: vec!["<tool_call>".to_string()],
                tool_call_end_tokens: vec!["</tool_call>".to_string()],
                arguments_keys: vec!["arguments".to_string()],
                ..Default::default()
            },
        };
        let result = try_tool_call_parse(input, &config).unwrap().unwrap();
        let (name, args) = extract_name_and_args(result);
        assert_eq!(name, "get_weather");
        assert_eq!(args["location"], "San Francisco, CA");
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
        let result = try_tool_call_parse(input, &config).unwrap().unwrap();
        let (name, args) = extract_name_and_args(result);
        assert_eq!(name, "get_weather");
        assert_eq!(args["location"], "San Francisco, CA");
        assert_eq!(args["unit"], "fahrenheit");
    }

    #[test]
    #[ignore]
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
        let result = try_tool_call_parse(input, &config).unwrap().unwrap();
        let (name, args) = extract_name_and_args(result);
        assert_eq!(name, "get_weather");
        assert_eq!(args["location"], "San Francisco, CA");
        assert_eq!(args["unit"], "fahrenheit");
    }

    #[test]
    #[ignore]
    fn test_meta_llama_llama31_8b_instruct_simple() {
        let input = r#"{"name": "get_weather", "parameters": {"location": "San Francisco, CA", "unit": "fahrenheit"}}"#;
        let config = ToolCallConfig {
            format: ToolCallParserType::Json,
            json: JsonParserConfig {
                tool_call_start_tokens: vec![],
                tool_call_end_tokens: vec![],
                arguments_keys: vec!["parameters".to_string()],
                ..Default::default()
            },
        };
        let result = try_tool_call_parse(input, &config).unwrap().unwrap();
        let (name, args) = extract_name_and_args(result);
        assert_eq!(name, "get_weather");
        assert_eq!(args["location"], "San Francisco, CA");
        assert_eq!(args["unit"], "fahrenheit");
    }

    #[test]
    #[ignore]
    fn test_internlm_internlm2_5_7b_chat_simple() {
        let input = r#"San Francisco's weather is known for its mild climate with plenty of fog, especially along the coast. Here's an overview of the weather in Fahrenheit:

- **Summer (June to August)**: Average highs range from the mid-60s to low 70s Fahrenheit, with cooler mornings and evenings. Coastal areas may be cooler than inland spots.

Remember, San Francisco weather can be quite unpredictable, particularly with its famous fog, which can significantly lower temperatures. Always check a local weather forecast for the most accurate and up-to-date information."#;
        let result = try_tool_call_parse(input, &ToolCallConfig::default()).unwrap();
        assert!(result.is_none()); // This model doesn't produce tool calls
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
        let result = try_tool_call_parse(input, &config).unwrap().unwrap();
        let (name, args) = extract_name_and_args(result);
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
        let result = try_tool_call_parse(input, &config).unwrap().unwrap();
        let (name, args) = extract_name_and_args(result);
        assert_eq!(name, "get_weather");
        assert_eq!(args["location"], "San Francisco, CA");
        assert_eq!(args["unit"], "fahrenheit");
    }
}
