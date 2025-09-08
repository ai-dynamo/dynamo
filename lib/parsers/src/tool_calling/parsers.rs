// SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use super::config::{ToolCallConfig, ToolCallParserType};
use super::harmony::parse_tool_calls_harmony;
use super::json::try_tool_call_parse_json;
use super::pythonic::try_tool_call_parse_pythonic;
use super::response::ToolCallResponse;
use std::collections::HashMap;
use std::sync::OnceLock;

static PARSER_MAP: OnceLock<HashMap<&'static str, ToolCallConfig>> = OnceLock::new();

// Always update this parsermap when adding a new parser
pub fn get_tool_parser_map() -> &'static HashMap<&'static str, ToolCallConfig> {
    PARSER_MAP.get_or_init(|| {
        let mut map = HashMap::new();
        map.insert("hermes", ToolCallConfig::hermes());
        map.insert("nemotron_deci", ToolCallConfig::nemotron_deci());
        map.insert("llama3_json", ToolCallConfig::llama3_json());
        map.insert("mistral", ToolCallConfig::mistral());
        map.insert("phi4", ToolCallConfig::phi4());
        map.insert("pythonic", ToolCallConfig::pythonic());
        map.insert("harmony", ToolCallConfig::harmony());
        map.insert("deepseek_v3_1", ToolCallConfig::deepseek_v3_1());
        map.insert("default", ToolCallConfig::default());
        map
    })
}

pub fn get_available_tool_parsers() -> Vec<&'static str> {
    get_tool_parser_map().keys().copied().collect()
}

pub fn try_tool_call_parse(
    message: &str,
    config: &ToolCallConfig,
) -> anyhow::Result<(Vec<ToolCallResponse>, Option<String>)> {
    // Use match statement (Rust's switch statement) to call the appropriate parser
    match config.format {
        ToolCallParserType::Json => {
            let (results, normal_content) = try_tool_call_parse_json(message, &config.json)?;
            Ok((results, normal_content))
        }
        ToolCallParserType::Harmony => {
            let (results, normal_content) = parse_tool_calls_harmony(message, &config.json)?;
            Ok((results, normal_content))
        }
        ToolCallParserType::Pythonic => {
            let (results, normal_content) = try_tool_call_parse_pythonic(message)?;
            Ok((results, normal_content))
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
) -> anyhow::Result<(Vec<ToolCallResponse>, Option<String>)> {
    // Get the tool parser map
    let parser_map = get_tool_parser_map();

    // Handle None or empty string by defaulting to "default"
    let parser_key = match parser_str {
        Some(s) if !s.is_empty() => s,
        _ => "default", // None or empty string
    };

    match parser_map.get(parser_key) {
        Some(config) => {
            let (results, normal_content) = try_tool_call_parse(message, config)?;
            Ok((results, normal_content))
        }
        None => anyhow::bail!(
            "Parser '{}' is not implemented. Available parsers: {:?}",
            parser_key,
            get_available_tool_parsers()
        ),
    }
}

// Tests
// cargo test postprocessor::tool_calling::parsers
#[cfg(test)]
mod tests {
    use super::super::config::JsonParserConfig;
    use super::*;

    fn extract_name_and_args(call: ToolCallResponse) -> (String, serde_json::Value) {
        let args: serde_json::Value = serde_json::from_str(&call.function.arguments).unwrap();
        (call.function.name, args)
    }

    #[test]
    fn test_get_available_tool_parsers() {
        let parsers = get_available_tool_parsers();
        assert!(!parsers.is_empty());
        // Update this list when adding a new parser
        let available_parsers = [
            "hermes",
            "llama3_json",
            "harmony",
            "nemotron_deci",
            "mistral",
            "phi4",
            "default",
            "pythonic",
            "deepseek_v3_1",
        ];
        for parser in available_parsers {
            assert!(parsers.contains(&parser));
        }
    }

    #[test]
    fn parses_single_parameters_object() {
        let input = r#"{ "name": "hello", "parameters": { "x": 1, "y": 2 } }"#;
        let (result, content) = try_tool_call_parse(input, &ToolCallConfig::default()).unwrap();
        assert_eq!(content, Some("".to_string()));
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
        let (result, content) = try_tool_call_parse(input, &ToolCallConfig::default()).unwrap();
        assert_eq!(content, Some("".to_string()));
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
        let (result, content) = try_tool_call_parse(input, &ToolCallConfig::default()).unwrap();
        assert_eq!(content, Some("".to_string()));
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
        let (result, content) = try_tool_call_parse(input, &ToolCallConfig::default()).unwrap();
        assert_eq!(content, Some("".to_string()));
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
        let (result, content) = try_tool_call_parse(input, &ToolCallConfig::default()).unwrap();
        assert_eq!(content, Some("".to_string()));
        assert!(!result.is_empty());
        assert_eq!(result.len(), 1);
        let (name, args) = extract_name_and_args(result[0].clone());
        assert_eq!(name, "wrapped");
        assert_eq!(args["foo"], "bar");
    }

    #[test]
    fn parses_python_tag_prefixed_payload() {
        let input = r#"<|python_tag|>{ "name": "pyfunc", "arguments": { "k": "v" } }"#;
        let (result, content) = try_tool_call_parse(
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
        assert_eq!(content, Some("".to_string()));
        assert!(!result.is_empty());
        assert_eq!(result.len(), 1);
        let (name, args) = extract_name_and_args(result[0].clone());
        assert_eq!(name, "pyfunc");
        assert_eq!(args["k"], "v");
    }

    #[test]
    fn returns_none_on_invalid_input() {
        let input = r#"not even json"#;
        let (result, content) = try_tool_call_parse(input, &ToolCallConfig::default()).unwrap();
        assert_eq!(content, Some("not even json".to_string()));
        assert!(result.is_empty());
    }

    #[test]
    fn returns_none_on_valid_json_wrong_shape() {
        let input = r#"{ "foo": "bar" }"#;
        let (result, content) = try_tool_call_parse(input, &ToolCallConfig::default()).unwrap();
        assert_eq!(content, Some("{ \"foo\": \"bar\" }".to_string()));
        assert!(result.is_empty());
    }

    // Tests for real model outputs - disabled by default
    #[test]
    fn test_nvidia_llama3_nemotron_super_49b_simple() {
        let input = r#"<think>
Okay, the user is asking for the weather in San Francisco in Fahrenheit. Let me check the tools available.
</think>

<TOOLCALL>[{"name": "get_weather", "arguments": {"location": "San Francisco, CA", "unit": "fahrenheit"}}]</TOOLCALL>"#;
        let (result, content) = detect_and_parse_tool_call(input, Some("nemotron_deci")).unwrap();
        assert!(!result.is_empty());
        assert_eq!(result.len(), 1);
        assert_eq!(content, Some("<think>\nOkay, the user is asking for the weather in San Francisco in Fahrenheit. Let me check the tools available.\n</think>".to_string()));
        let (name, args) = extract_name_and_args(result[0].clone());
        assert_eq!(name, "get_weather");
        assert_eq!(args["location"], "San Francisco, CA");
        assert_eq!(args["unit"], "fahrenheit");
    }

    #[test]
    fn test_nvidia_llama3_nemotron_super_49b_simple_with_no_think() {
        let input = r#"<TOOLCALL>[{"name": "get_weather", "arguments": {"location": "San Francisco, CA", "unit": "fahrenheit"}}]</TOOLCALL>"#;
        let (result, content) = detect_and_parse_tool_call(input, Some("nemotron_deci")).unwrap();
        assert!(!result.is_empty());
        assert_eq!(result.len(), 1);
        assert_eq!(content, Some("".to_string()));
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
        let (result, content) = try_tool_call_parse(input, &config).unwrap();
        assert_eq!(content, Some("<think>\nOkay, the user is asking for the weather in San Francisco in Fahrenheit. Let me check the tools available.\n</think>".to_string()));
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
        let (result, content) = try_tool_call_parse(input, &config).unwrap();
        assert_eq!(content, Some("<think>\nOkay, the user is asking for the weather in San Francisco in Fahrenheit. Let me check the tools available.\n</think>".to_string()));
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
        let (result, content) = detect_and_parse_tool_call(input, Some("hermes")).unwrap();
        assert_eq!(content, Some("".to_string()));
        assert!(!result.is_empty());
        assert_eq!(result.len(), 1);
        let (name, args) = extract_name_and_args(result[0].clone());
        assert_eq!(name, "get_weather");
        assert_eq!(args["location"], "San Francisco, CA");
        assert_eq!(args["unit"], "fahrenheit");
    }

    #[test]
    fn test_qwen_qwq_32b_simple_with_normal_text() {
        let input = r#"Hey How are you? <tool_call>
{"name": "get_weather", "arguments": {"location": "San Francisco, CA", "unit": "fahrenheit"}}
</tool_call>"#;
        let (result, content) = detect_and_parse_tool_call(input, Some("hermes")).unwrap();
        assert_eq!(content, Some("Hey How are you?".to_string()));
        assert!(!result.is_empty());
        assert_eq!(result.len(), 1);
    }

    #[test]
    fn test_nousresearch_hermes3_llama31_8b_simple() {
        let input = r#"<tool_call>
{"name": "get_weather", "arguments": {"location": "San Francisco, CA", "unit": "fahrenheit"}}
</tool_call>"#;
        let (result, content) = detect_and_parse_tool_call(input, Some("hermes")).unwrap();
        assert_eq!(content, Some("".to_string()));
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
        let (result, content) = try_tool_call_parse(input, &config).unwrap();
        assert_eq!(content, Some("".to_string()));
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
    fn test_qwen_qwq_32b_multiple_tool_calls_with_normal_text() {
        let input = r#"Hey How are you? <tool_call>
{"name": "get_weather", "arguments": {"location": "San Francisco, CA", "unit": "fahrenheit"}}
</tool_call>
<tool_call>
{"name": "get_weather", "arguments": {"location": "New York, NY", "unit": "fahrenheit"}}
</tool_call>
"#;
        let config = ToolCallConfig::hermes();
        let (result, content) = try_tool_call_parse(input, &config).unwrap();
        assert_eq!(content, Some("Hey How are you?".to_string()));
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
        let (result, content) = try_tool_call_parse(input, &config).unwrap();
        assert_eq!(content, Some("".to_string()));
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
        let (result, content) = try_tool_call_parse(input, &config).unwrap();
        assert_eq!(content, Some("".to_string()));
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
        let config = ToolCallConfig::mistral();
        let (result, content) = try_tool_call_parse(input, &config).unwrap();
        assert_eq!(content, Some("".to_string()));
        assert!(!result.is_empty());
        assert_eq!(result.len(), 1);
        let (name, args) = extract_name_and_args(result[0].clone());
        assert_eq!(name, "get_weather");
        assert_eq!(args["location"], "San Francisco, CA");
        assert_eq!(args["unit"], "fahrenheit");
    }

    #[test]
    fn test_mistralai_mistral_7b_instruct_v03_simple_with_normal_text() {
        let input = r#"Hey How are you? [{"name": "get_weather", "arguments": {"location": "San Francisco, CA", "unit": "fahrenheit"}}]"#;
        let config = ToolCallConfig::mistral();
        let (result, content) = try_tool_call_parse(input, &config).unwrap();
        assert_eq!(content, Some("Hey How are you?".to_string()));
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
        let config = ToolCallConfig::mistral();
        let (result, content) = try_tool_call_parse(input, &config).unwrap();
        assert_eq!(content, Some("".to_string()));
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
        let (result, content) = try_tool_call_parse(input, &config).unwrap();
        assert_eq!(content, Some("".to_string()));
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
    fn test_mistralai_mistral_7b_instruct_v03_multiple_with_normal_text() {
        let input = r#"Hey How are you? [{"name": "get_weather", "arguments": {"location": "San Francisco, CA", "unit": "fahrenheit"}}, {"name": "get_weather", "arguments": {"location": "New York, NY", "unit": "fahrenheit"}}]"#;
        let config = ToolCallConfig::mistral();
        let (result, content) = try_tool_call_parse(input, &config).unwrap();
        assert_eq!(content, Some("Hey How are you?".to_string()));
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
        "arguments": {"location":
        "San Francisco, CA",
        "unit": "fahrenheit"}},
        {"name": "get_weather", "arguments":
        {"location": "New York, NY", "unit":
        "fahrenheit"}}]
        "#;
        let config = ToolCallConfig::mistral();
        let (result, content) = try_tool_call_parse(input, &config).unwrap();
        assert_eq!(content, Some("".to_string()));
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
        let (result, content) = try_tool_call_parse(input, &config).unwrap();
        assert_eq!(content, Some("".to_string()));
        assert!(!result.is_empty());
        assert_eq!(result.len(), 1);
        let (name, args) = extract_name_and_args(result[0].clone());
        assert_eq!(name, "get_weather");
        assert_eq!(args["location"], "San Francisco, CA");
        assert_eq!(args["unit"], "fahrenheit");
    }

    #[test]
    fn test_mistralai_mistral_7b_instruct_v03_single_with_start_token_with_normal_text() {
        let input = r#"Hey How are you? [TOOL_CALLS] [{"name": "get_weather", "arguments": {"location": "San Francisco, CA", "unit": "fahrenheit"}}]"#;
        let config = ToolCallConfig::mistral();
        let (result, content) = try_tool_call_parse(input, &config).unwrap();
        assert_eq!(content, Some("Hey How are you?".to_string()));
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
        let (result, content) = try_tool_call_parse(input, &config).unwrap();
        assert_eq!(content, Some("".to_string()));
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
        let (result, content) = try_tool_call_parse(input, &config).unwrap();
        assert_eq!(content, Some("".to_string()));
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
    fn test_mistralai_mistral_7b_instruct_v03_single_with_start_token_multiple_with_normal_text() {
        let input = r#"Hey How are you? [TOOL_CALLS] [{"name": "get_weather", "arguments": {"location": "San Francisco, CA", "unit": "fahrenheit"}}, {"name": "get_weather", "arguments": {"location": "New York, NY", "unit": "fahrenheit"}}]"#;
        let config = ToolCallConfig::mistral();
        let (result, content) = try_tool_call_parse(input, &config).unwrap();
        assert_eq!(content, Some("Hey How are you?".to_string()));
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
        let (result, content) = try_tool_call_parse(input, &config).unwrap();
        assert_eq!(content, Some("".to_string()));
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
        let (result, content) = try_tool_call_parse(input, &ToolCallConfig::mistral()).unwrap();
        assert_eq!(content, Some("".to_string()));
        assert!(!result.is_empty());
        assert_eq!(result.len(), 1);
        let (name, args) = extract_name_and_args(result[0].clone());
        assert_eq!(name, "get_weather");
        assert_eq!(args["location"], "San Francisco, CA");
        assert_eq!(args["unit"], "fahrenheit");
    }

    #[test]
    fn test_meta_llama_llama31_8b_instruct_simple_with_normal_text() {
        let input = r#"Hey How are you? {"name": "get_weather", "parameters": {"location": "San Francisco, CA", "unit": "fahrenheit"}}"#;
        let (result, content) = try_tool_call_parse(input, &ToolCallConfig::mistral()).unwrap();
        assert_eq!(content, Some("Hey How are you?".to_string()));
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
        let (result, content) = detect_and_parse_tool_call(input, Some("llama3_json")).unwrap();
        assert_eq!(content, Some("".to_string()));
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
        let (result, content) = detect_and_parse_tool_call(input, Some("llama3_json")).unwrap();
        assert_eq!(content, Some("".to_string()));
        assert!(!result.is_empty());
        assert_eq!(result.len(), 1);
        let (name, args) = extract_name_and_args(result[0].clone());
        assert_eq!(name, "get_weather");
        assert_eq!(args["location"], "San Francisco, CA");
        assert_eq!(args["unit"], "fahrenheit");
    }

    #[test]
    fn test_meta_llama_llama31_8b_instruct_with_python_tag_with_normal_text() {
        let input = r#"Hey How are you? <|python_tag|>{ "name": "get_weather", "parameters": {"location": "San Francisco, CA", "unit": "fahrenheit" } }"#;
        let (result, content) = detect_and_parse_tool_call(input, Some("llama3_json")).unwrap();
        assert_eq!(content, Some("Hey How are you?".to_string()));
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
        let (result, content) = detect_and_parse_tool_call(input, Some("llama3_json")).unwrap();
        assert_eq!(content, Some("".to_string()));
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
        let (result, content) = detect_and_parse_tool_call(input, Some("llama3_json")).unwrap();
        assert_eq!(content, Some("".to_string()));
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
        let (result, content) = detect_and_parse_tool_call(input, Some("hermes")).unwrap();
        assert_eq!(content, Some("not a json".to_string()));
        assert!(result.is_empty());

        // Known parser, but valid JSON with wrong shape should return Ok(None)
        let input = r#"{"foo": "bar"}"#;
        let (result, content) = detect_and_parse_tool_call(input, Some("hermes")).unwrap();
        assert_eq!(content, Some(r#"{"foo": "bar"}"#.to_string()));
        assert!(result.is_empty());
    }

    #[test]
    #[ignore]
    fn test_internlm_internlm2_5_7b_chat_simple() {
        let input = r#"San Francisco's weather is known for its mild climate with plenty of fog, especially along the coast. Here's an overview of the weather in Fahrenheit:

- **Summer (June to August)**: Average highs range from the mid-60s to low 70s Fahrenheit, with cooler mornings and evenings. Coastal areas may be cooler than inland spots.

Remember, San Francisco weather can be quite unpredictable, particularly with its famous fog, which can significantly lower temperatures. Always check a local weather forecast for the most accurate and up-to-date information."#;
        let (result, content) = try_tool_call_parse(input, &ToolCallConfig::default()).unwrap();
        assert_eq!(content, Some(input.to_string()));
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
        let (result, content) = try_tool_call_parse(input, &config).unwrap();
        assert_eq!(content, Some("".to_string()));
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
        let (result, content) = try_tool_call_parse(input, &config).unwrap();
        assert_eq!(content, Some("".to_string()));
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
        let (result, content) = detect_and_parse_tool_call(input, None).unwrap();
        assert_eq!(content, Some("".to_string()));
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
        let (result, content) = detect_and_parse_tool_call(input, None).unwrap();
        assert_eq!(content, Some("".to_string()));
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
    fn test_detect_and_parse_tool_call_default_parser_nemotron_deci_multiple_with_normal_text() {
        let input = r#"Hey How are you? <TOOLCALL>[{"name": "get_weather", "arguments": {"location": "San Francisco, CA", "unit": "fahrenheit"}}, {"name": "get_weather", "arguments": {"location": "New York, NY", "unit": "fahrenheit"}}]</TOOLCALL>"#;
        let (result, content) = detect_and_parse_tool_call(input, None).unwrap();
        assert_eq!(content, Some("Hey How are you?".to_string()));
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
        let (result, content) = detect_and_parse_tool_call(input, None).unwrap();
        assert_eq!(content, Some("".to_string()));
        assert!(!result.is_empty());
        assert_eq!(result.len(), 1);
        let (name, args) = extract_name_and_args(result[0].clone());
        assert_eq!(name, "get_weather");
        assert_eq!(args["location"], "San Francisco, CA");
        assert_eq!(args["unit"], "fahrenheit");
    }

    #[test]
    fn test_detect_and_parse_tool_call_default_parser_llama3_json_with_python_tag_with_normal_text()
    {
        let input = r#"Hey How are you? <|python_tag|>{ "name": "get_weather", "arguments": {"location": "San Francisco, CA", "unit": "fahrenheit" } }"#;
        let (result, content) = detect_and_parse_tool_call(input, None).unwrap();
        assert_eq!(content, Some("Hey How are you?".to_string()));
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
        let (result, content) = detect_and_parse_tool_call(input, None).unwrap();
        assert_eq!(content, Some("".to_string()));
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
        let (result, content) = detect_and_parse_tool_call(input, None).unwrap();
        assert_eq!(content, Some("".to_string()));
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
        let (result, content) = try_tool_call_parse(input, &ToolCallConfig::mistral()).unwrap();
        assert_eq!(content, Some("".to_string()));
        assert!(!result.is_empty());
        assert_eq!(result.len(), 1);
        let (name, args) = extract_name_and_args(result[0].clone());
        assert_eq!(name, "get_weather");
        assert_eq!(args["location"], "San Francisco, CA");
        assert_eq!(args["unit"], "fahrenheit");
    }

    #[test]
    fn test_detect_and_parse_tool_call_default_parser_llama3_json_without_python_tag_with_normal_text()
     {
        let input = r#"Hey How are you? { "name": "get_weather", "arguments": {"location": "San Francisco, CA", "unit": "fahrenheit" } }"#;
        let (result, content) = try_tool_call_parse(input, &ToolCallConfig::mistral()).unwrap();
        assert_eq!(content, Some("Hey How are you?".to_string()));
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
        let (result, content) = detect_and_parse_tool_call(input, Some("phi4")).unwrap();
        assert_eq!(content, Some("".to_string()));
        assert_eq!(result.len(), 1);
        let (name, args) = extract_name_and_args(result[0].clone());
        assert_eq!(name, "get_country_capital");
        assert_eq!(args["country"], "Poland");
    }

    #[test]
    fn test_phi4_single_function_call_with_normal_text() {
        let input = r#"Hey How are you? functools[{"name": "get_country_capital", "arguments": {"country": "Poland"}}]"#;
        let (result, content) = detect_and_parse_tool_call(input, Some("phi4")).unwrap();
        assert_eq!(content, Some("Hey How are you?".to_string()));
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
        let (result, content) = detect_and_parse_tool_call(input, Some("phi4")).unwrap();
        assert_eq!(content, Some("".to_string()));
        assert_eq!(result.len(), 2);

        let (name1, args1) = extract_name_and_args(result[0].clone());
        assert_eq!(name1, "get_country_capital");
        assert_eq!(args1["country"], "Poland");

        let (name2, args2) = extract_name_and_args(result[1].clone());
        assert_eq!(name2, "get_population");
        assert_eq!(args2["city"], "Warsaw");
    }

    #[test]
    fn test_phi4_multiple_function_calls_simple_arguments_with_normal_text() {
        let input = r#"Hey How are you? functools[
  {"name": "get_country_capital", "arguments": {"country": "Poland"}},
  {"name": "get_population", "arguments": {"city": "Warsaw"}}
]"#;
        let (result, content) = detect_and_parse_tool_call(input, Some("phi4")).unwrap();
        assert_eq!(content, Some("Hey How are you?".to_string()));
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
        let (result, content) = detect_and_parse_tool_call(input, Some("phi4")).unwrap();
        assert_eq!(content, Some("".to_string()));
        assert_eq!(result.len(), 1);
        let (name, args) = extract_name_and_args(result[0].clone());
        assert_eq!(name, "get_weather_forecast");
        assert_eq!(args["date"], "2023-10-05");
        assert_eq!(args["location"]["city"], "San Francisco");
        assert_eq!(args["location"]["state"], "CA");
    }

    #[test]
    fn test_phi4_single_function_call_nested_json_arguments_with_normal_text() {
        let input = r#"Hey How are you? functools[{"name": "get_weather_forecast", "arguments":
        {"location": {"city": "San Francisco",
        "state": "CA"}, "date": "2023-10-05"}}]"#;
        let (result, content) = detect_and_parse_tool_call(input, Some("phi4")).unwrap();
        assert_eq!(content, Some("Hey How are you?".to_string()));
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
        let (result, content) = detect_and_parse_tool_call(input, Some("phi4")).unwrap();
        assert_eq!(content, Some("".to_string()));
        assert_eq!(result.len(), 1);
        let (name, args) = extract_name_and_args(result[0].clone());
        assert_eq!(name, "calculate_distance");
        assert_eq!(args["from"], "New York");
        assert_eq!(args["to"], "Los Angeles");
    }

    #[test]
    fn test_phi4_function_call_with_parameters_instead_of_arguments_with_normal_text() {
        let input = r#"Hey How are you? functools[{"name": "calculate_distance",
         "parameters": {"from": "New York", "to": "Los Angeles"}}]"#;
        let (result, content) = detect_and_parse_tool_call(input, Some("phi4")).unwrap();
        assert_eq!(content, Some("Hey How are you?".to_string()));
        assert_eq!(result.len(), 1);
        let (name, args) = extract_name_and_args(result[0].clone());
        assert_eq!(name, "calculate_distance");
        assert_eq!(args["from"], "New York");
        assert_eq!(args["to"], "Los Angeles");
    }

    #[test]
    fn test_pythonic_parser_basic_with_constants() {
        let input = r#"[get_weather(location="San Francisco", unit="fahrenheit"), get_weather(location="New York", unit="fahrenheit")]"#;
        let (result, content) = detect_and_parse_tool_call(input, Some("pythonic")).unwrap();
        assert_eq!(content, Some("".to_string()));
        assert_eq!(result.len(), 2);
        let (name, args) = extract_name_and_args(result[0].clone());
        assert_eq!(name, "get_weather");
        assert_eq!(args["location"], "San Francisco");
        assert_eq!(args["unit"], "fahrenheit");
        let (name, args) = extract_name_and_args(result[1].clone());
        assert_eq!(name, "get_weather");
        assert_eq!(args["location"], "New York");
        assert_eq!(args["unit"], "fahrenheit");
    }

    #[test]
    #[ignore]
    fn test_pythonic_parser_with_constants_and_normal_text() {
        let input = r#"Hey How are you? [get_weather(location="San Francisco", unit="fahrenheit"), get_weather(location="New York", unit="fahrenheit")]"#;
        let (result, content) = detect_and_parse_tool_call(input, Some("pythonic")).unwrap();
        assert_eq!(content, Some("Hey How are you?".to_string()));
        assert_eq!(result.len(), 2);

        let (name, args) = extract_name_and_args(result[0].clone());
        assert_eq!(name, "get_weather");
        assert_eq!(args["location"], "San Francisco");
        assert_eq!(args["unit"], "fahrenheit");
        let (name, args) = extract_name_and_args(result[1].clone());
        assert_eq!(name, "get_weather");
        assert_eq!(args["location"], "New York");
        assert_eq!(args["unit"], "fahrenheit");
    }

    #[test]
    fn test_harmony_parser_basic() {
        let input = r#"
        <|channel|>analysis<|message|>Need to use function get_current_weather.<|end|>
        <|start|>assistant<|channel|>commentary to=functions.get_current_weather <|constrain|>json
        <|message|>{"location":"San Francisco", "unit":"fahrenheit"}<|call|>
        "#;
        let (result, content) = detect_and_parse_tool_call(input, Some("harmony")).unwrap();
        assert_eq!(
            content,
            Some("Need to use function get_current_weather.".to_string())
        );
        assert_eq!(result.len(), 1);
        let (name, args) = extract_name_and_args(result[0].clone());
        assert_eq!(name, "get_current_weather");
        assert_eq!(args["location"], "San Francisco");
        assert_eq!(args["unit"], "fahrenheit");
    }

    #[test]
    fn test_deepseek_v3_1_parser_basic() {
        let input = r#"<｜tool▁calls▁begin｜><｜tool▁call▁begin｜>get_current_weather<｜tool▁sep｜>{"location": "Tokyo"}<｜tool▁call▁end｜><｜tool▁call▁begin｜>get_current_weather<｜tool▁sep｜>{"location": "Paris"}<｜tool▁call▁end｜><｜tool▁calls▁end｜><｜end▁of▁sentence｜>"#;
        let (result, content) = detect_and_parse_tool_call(input, Some("deepseek_v3_1")).unwrap();
        assert_eq!(content, Some("".to_string()));
        assert_eq!(result.len(), 2);
        let (name, args) = extract_name_and_args(result[0].clone());
        assert_eq!(name, "get_current_weather");
        assert_eq!(args["location"], "Tokyo");
        let (name, args) = extract_name_and_args(result[1].clone());
        assert_eq!(name, "get_current_weather");
        assert_eq!(args["location"], "Paris");
    }
}
