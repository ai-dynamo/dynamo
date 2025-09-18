// SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use super::config::{ToolCallConfig, ToolCallParserType};
use super::harmony::{detect_tool_call_start_harmony, parse_tool_calls_harmony};
use super::json::{detect_tool_call_start_json, try_tool_call_parse_json};
use super::pythonic::{detect_tool_call_start_pythonic, try_tool_call_parse_pythonic};
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

pub async fn try_tool_call_parse(
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
            let (results, normal_content) = parse_tool_calls_harmony(message, &config.json).await?;
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
pub async fn detect_and_parse_tool_call(
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
            let (results, normal_content) = try_tool_call_parse(message, config).await?;
            Ok((results, normal_content))
        }
        None => anyhow::bail!(
            "Parser '{}' is not implemented. Available parsers: {:?}",
            parser_key,
            get_available_tool_parsers()
        ),
    }
}

pub fn detect_tool_call_start(chunk: &str, parser_str: Option<&str>) -> anyhow::Result<bool> {
    let parser_map = get_tool_parser_map();
    let parser_key = match parser_str {
        Some(s) if !s.is_empty() => s,
        _ => "default", // None or empty string
    };

    match parser_map.get(parser_key) {
        Some(config) => match config.format {
            ToolCallParserType::Json => Ok(detect_tool_call_start_json(chunk, &config.json)),
            ToolCallParserType::Harmony => {
                Ok(detect_tool_call_start_harmony(chunk, &config.json, false))
            }
            ToolCallParserType::Pythonic => Ok(detect_tool_call_start_pythonic(chunk)),
            ToolCallParserType::Typescript => {
                anyhow::bail!("Typescript parser not implemented");
            }
            ToolCallParserType::Xml => {
                anyhow::bail!("Xml parser not implemented");
            }
        },
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

    #[tokio::test]
    async fn parses_single_parameters_object() {
        let input = r#"{ "name": "hello", "parameters": { "x": 1, "y": 2 } }"#;
        let (result, content) = try_tool_call_parse(input, &ToolCallConfig::default())
            .await
            .unwrap();
        assert_eq!(content, Some("".to_string()));
        assert!(!result.is_empty());
        assert_eq!(result.len(), 1);
        let (name, args) = extract_name_and_args(result[0].clone());
        assert_eq!(name, "hello");
        assert_eq!(args["x"], 1);
        assert_eq!(args["y"], 2);
    }

    #[tokio::test]
    async fn parses_single_arguments_object() {
        let input = r#"{ "name": "world", "arguments": { "a": "abc", "b": 42 } }"#;
        let (result, content) = try_tool_call_parse(input, &ToolCallConfig::default())
            .await
            .unwrap();
        assert_eq!(content, Some("".to_string()));
        assert!(!result.is_empty());
        assert_eq!(result.len(), 1);
        let (name, args) = extract_name_and_args(result[0].clone());
        assert_eq!(name, "world");
        assert_eq!(args["a"], "abc");
        assert_eq!(args["b"], 42);
    }

    #[tokio::test]
    async fn parses_vec_of_parameters() {
        let input = r#"[{ "name": "first", "parameters": { "a": 1 } }, { "name": "second", "parameters": { "b": 2 } }]"#;
        let (result, content) = try_tool_call_parse(input, &ToolCallConfig::default())
            .await
            .unwrap();
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

    #[tokio::test]
    async fn parses_vec_of_arguments() {
        let input = r#"[{ "name": "alpha", "arguments": { "a": "x" } }, { "name": "omega", "arguments": { "z": "y" } }]"#;
        let (result, content) = try_tool_call_parse(input, &ToolCallConfig::default())
            .await
            .unwrap();
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

    #[tokio::test]
    async fn parses_toolcall_wrapped_payload() {
        let input =
            r#"<TOOLCALL>[{ "name": "wrapped", "parameters": { "foo": "bar" } }]</TOOLCALL>"#;
        let (result, content) = try_tool_call_parse(input, &ToolCallConfig::default())
            .await
            .unwrap();
        assert_eq!(content, Some("".to_string()));
        assert!(!result.is_empty());
        assert_eq!(result.len(), 1);
        let (name, args) = extract_name_and_args(result[0].clone());
        assert_eq!(name, "wrapped");
        assert_eq!(args["foo"], "bar");
    }

    #[tokio::test]
    async fn parses_python_tag_prefixed_payload() {
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
        .await
        .unwrap();
        assert_eq!(content, Some("".to_string()));
        assert!(!result.is_empty());
        assert_eq!(result.len(), 1);
        let (name, args) = extract_name_and_args(result[0].clone());
        assert_eq!(name, "pyfunc");
        assert_eq!(args["k"], "v");
    }

    #[tokio::test]
    async fn returns_none_on_invalid_input() {
        let input = r#"not even json"#;
        let (result, content) = try_tool_call_parse(input, &ToolCallConfig::default())
            .await
            .unwrap();
        assert_eq!(content, Some("not even json".to_string()));
        assert!(result.is_empty());
    }

    #[tokio::test]
    async fn returns_none_on_valid_json_wrong_shape() {
        let input = r#"{ "foo": "bar" }"#;
        let (result, content) = try_tool_call_parse(input, &ToolCallConfig::default())
            .await
            .unwrap();
        assert_eq!(content, Some("{ \"foo\": \"bar\" }".to_string()));
        assert!(result.is_empty());
    }

    // Tests for real model outputs - disabled by default
    #[tokio::test]
    async fn test_nvidia_llama3_nemotron_super_49b_simple() {
        let input = r#"<think>
Okay, the user is asking for the weather in San Francisco in Fahrenheit. Let me check the tools available.
</think>

<TOOLCALL>[{"name": "get_weather", "arguments": {"location": "San Francisco, CA", "unit": "fahrenheit"}}]</TOOLCALL>"#;
        let (result, content) = detect_and_parse_tool_call(input, Some("nemotron_deci"))
            .await
            .unwrap();
        assert!(!result.is_empty());
        assert_eq!(result.len(), 1);
        assert_eq!(content, Some("<think>\nOkay, the user is asking for the weather in San Francisco in Fahrenheit. Let me check the tools available.\n</think>".to_string()));
        let (name, args) = extract_name_and_args(result[0].clone());
        assert_eq!(name, "get_weather");
        assert_eq!(args["location"], "San Francisco, CA");
        assert_eq!(args["unit"], "fahrenheit");
    }

    #[tokio::test]
    async fn test_nvidia_llama3_nemotron_super_49b_simple_with_no_think() {
        let input = r#"<TOOLCALL>[{"name": "get_weather", "arguments": {"location": "San Francisco, CA", "unit": "fahrenheit"}}]</TOOLCALL>"#;
        let (result, content) = detect_and_parse_tool_call(input, Some("nemotron_deci"))
            .await
            .unwrap();
        assert!(!result.is_empty());
        assert_eq!(result.len(), 1);
        assert_eq!(content, Some("".to_string()));
        let (name, args) = extract_name_and_args(result[0].clone());
        assert_eq!(name, "get_weather");
        assert_eq!(args["location"], "San Francisco, CA");
        assert_eq!(args["unit"], "fahrenheit");
    }

    #[tokio::test]
    async fn test_nvidia_llama3_nemotron_super_49b_with_function_array() {
        let input = r#"<think>
Okay, the user is asking for the weather in San Francisco in Fahrenheit. Let me check the tools available.
</think>

<TOOLCALL>[{"name": "get_weather", "arguments": {"location": "San Francisco, CA", "unit": "fahrenheit"}}, {"name": "get_weather", "arguments": {"location": "New York, NY", "unit": "fahrenheit"}}]</TOOLCALL>"#;
        let config = ToolCallConfig::nemotron_deci();
        let (result, content) = try_tool_call_parse(input, &config).await.unwrap();
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

    #[tokio::test]
    async fn test_nvidia_llama3_nemotron_super_49b_with_function_array_with_new_lines() {
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
        let (result, content) = try_tool_call_parse(input, &config).await.unwrap();
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

    #[tokio::test]
    async fn test_qwen_qwq_32b_simple() {
        let input = r#"<tool_call>
{"name": "get_weather", "arguments": {"location": "San Francisco, CA", "unit": "fahrenheit"}}
</tool_call>"#;
        let (result, content) = detect_and_parse_tool_call(input, Some("hermes"))
            .await
            .unwrap();
        assert_eq!(content, Some("".to_string()));
        assert!(!result.is_empty());
        assert_eq!(result.len(), 1);
        let (name, args) = extract_name_and_args(result[0].clone());
        assert_eq!(name, "get_weather");
        assert_eq!(args["location"], "San Francisco, CA");
        assert_eq!(args["unit"], "fahrenheit");
    }

    #[tokio::test]
    async fn test_qwen_qwq_32b_simple_with_normal_text() {
        let input = r#"Hey How are you? <tool_call>
{"name": "get_weather", "arguments": {"location": "San Francisco, CA", "unit": "fahrenheit"}}
</tool_call>"#;
        let (result, content) = detect_and_parse_tool_call(input, Some("hermes"))
            .await
            .unwrap();
        assert_eq!(content, Some("Hey How are you?".to_string()));
        assert!(!result.is_empty());
        assert_eq!(result.len(), 1);
    }

    #[tokio::test]
    async fn test_nousresearch_hermes3_llama31_8b_simple() {
        let input = r#"<tool_call>
{"name": "get_weather", "arguments": {"location": "San Francisco, CA", "unit": "fahrenheit"}}
</tool_call>"#;
        let (result, content) = detect_and_parse_tool_call(input, Some("hermes"))
            .await
            .unwrap();
        assert_eq!(content, Some("".to_string()));
        assert!(!result.is_empty());
        assert_eq!(result.len(), 1);
        let (name, args) = extract_name_and_args(result[0].clone());
        assert_eq!(name, "get_weather");
        assert_eq!(args["location"], "San Francisco, CA");
        assert_eq!(args["unit"], "fahrenheit");
    }

    #[tokio::test]
    async fn test_qwen_qwq_32b_multiple_tool_calls() {
        let input = r#"<tool_call>
{"name": "get_weather", "arguments": {"location": "San Francisco, CA", "unit": "fahrenheit"}}
</tool_call>
<tool_call>
{"name": "get_weather", "arguments": {"location": "New York, NY", "unit": "fahrenheit"}}
</tool_call>
"#;
        let config = ToolCallConfig::hermes();
        let (result, content) = try_tool_call_parse(input, &config).await.unwrap();
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

    #[tokio::test]
    async fn test_qwen_qwq_32b_multiple_tool_calls_with_normal_text() {
        let input = r#"Hey How are you? <tool_call>
{"name": "get_weather", "arguments": {"location": "San Francisco, CA", "unit": "fahrenheit"}}
</tool_call>
<tool_call>
{"name": "get_weather", "arguments": {"location": "New York, NY", "unit": "fahrenheit"}}
</tool_call>
"#;
        let config = ToolCallConfig::hermes();
        let (result, content) = try_tool_call_parse(input, &config).await.unwrap();
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

    #[tokio::test]
    async fn test_qwen_qwq_32b_multiple_tool_calls_with_new_lines() {
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
        let (result, content) = try_tool_call_parse(input, &config).await.unwrap();
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

    #[tokio::test]
    #[ignore]
    async fn test_ibm_granite_40_tiny_preview_simple() {
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
        let (result, content) = try_tool_call_parse(input, &config).await.unwrap();
        assert_eq!(content, Some("".to_string()));
        assert!(!result.is_empty());
        assert_eq!(result.len(), 1);
        let (name, args) = extract_name_and_args(result[0].clone());
        assert_eq!(name, "get_weather");
        assert_eq!(args["location"], "San Francisco, CA");
        assert_eq!(args["unit"], "fahrenheit");
    }

    #[tokio::test]
    async fn test_mistralai_mistral_7b_instruct_v03_simple() {
        let input = r#" [{"name": "get_weather", "arguments": {"location": "San Francisco, CA", "unit": "fahrenheit"}}]"#;
        let config = ToolCallConfig::mistral();
        let (result, content) = try_tool_call_parse(input, &config).await.unwrap();
        assert_eq!(content, Some("".to_string()));
        assert!(!result.is_empty());
        assert_eq!(result.len(), 1);
        let (name, args) = extract_name_and_args(result[0].clone());
        assert_eq!(name, "get_weather");
        assert_eq!(args["location"], "San Francisco, CA");
        assert_eq!(args["unit"], "fahrenheit");
    }

    #[tokio::test]
    async fn test_mistralai_mistral_7b_instruct_v03_simple_with_normal_text() {
        let input = r#"Hey How are you? [{"name": "get_weather", "arguments": {"location": "San Francisco, CA", "unit": "fahrenheit"}}]"#;
        let config = ToolCallConfig::mistral();
        let (result, content) = try_tool_call_parse(input, &config).await.unwrap();
        assert_eq!(content, Some("Hey How are you?".to_string()));
        assert!(!result.is_empty());
        assert_eq!(result.len(), 1);
        let (name, args) = extract_name_and_args(result[0].clone());
        assert_eq!(name, "get_weather");
        assert_eq!(args["location"], "San Francisco, CA");
        assert_eq!(args["unit"], "fahrenheit");
    }

    #[tokio::test]
    async fn test_mistralai_mistral_7b_instruct_v03_simple_with_new_lines() {
        let input = r#"
        [{"name": "get_weather",
        "arguments": {"location":
        "San Francisco, CA",
        "unit": "fahrenheit"}}]
        "#;
        let config = ToolCallConfig::mistral();
        let (result, content) = try_tool_call_parse(input, &config).await.unwrap();
        assert_eq!(content, Some("".to_string()));
        assert!(!result.is_empty());
        assert_eq!(result.len(), 1);
        let (name, args) = extract_name_and_args(result[0].clone());
        assert_eq!(name, "get_weather");
        assert_eq!(args["location"], "San Francisco, CA");
        assert_eq!(args["unit"], "fahrenheit");
    }

    #[tokio::test]
    async fn test_mistralai_mistral_7b_instruct_v03_multiple() {
        let input = r#" [{"name": "get_weather", "arguments": {"location": "San Francisco, CA", "unit": "fahrenheit"}}, {"name": "get_weather", "arguments": {"location": "New York, NY", "unit": "fahrenheit"}}]"#;
        let config = ToolCallConfig::mistral();
        let (result, content) = try_tool_call_parse(input, &config).await.unwrap();
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

    #[tokio::test]
    async fn test_mistralai_mistral_7b_instruct_v03_multiple_with_normal_text() {
        let input = r#"Hey How are you? [{"name": "get_weather", "arguments": {"location": "San Francisco, CA", "unit": "fahrenheit"}}, {"name": "get_weather", "arguments": {"location": "New York, NY", "unit": "fahrenheit"}}]"#;
        let config = ToolCallConfig::mistral();
        let (result, content) = try_tool_call_parse(input, &config).await.unwrap();
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

    #[tokio::test]
    async fn test_mistralai_mistral_7b_instruct_v03_multiple_with_new_lines() {
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
        let (result, content) = try_tool_call_parse(input, &config).await.unwrap();
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

    #[tokio::test]
    async fn test_mistralai_mistral_7b_instruct_v03_single_with_start_token() {
        let input = r#"[TOOL_CALLS] [{"name": "get_weather", "arguments": {"location": "San Francisco, CA", "unit": "fahrenheit"}}]"#;
        let config = ToolCallConfig::mistral();
        let (result, content) = try_tool_call_parse(input, &config).await.unwrap();
        assert_eq!(content, Some("".to_string()));
        assert!(!result.is_empty());
        assert_eq!(result.len(), 1);
        let (name, args) = extract_name_and_args(result[0].clone());
        assert_eq!(name, "get_weather");
        assert_eq!(args["location"], "San Francisco, CA");
        assert_eq!(args["unit"], "fahrenheit");
    }

    #[tokio::test]
    async fn test_mistralai_mistral_7b_instruct_v03_single_with_start_token_with_normal_text() {
        let input = r#"Hey How are you? [TOOL_CALLS] [{"name": "get_weather", "arguments": {"location": "San Francisco, CA", "unit": "fahrenheit"}}]"#;
        let config = ToolCallConfig::mistral();
        let (result, content) = try_tool_call_parse(input, &config).await.unwrap();
        assert_eq!(content, Some("Hey How are you?".to_string()));
        assert!(!result.is_empty());
        assert_eq!(result.len(), 1);
        let (name, args) = extract_name_and_args(result[0].clone());
        assert_eq!(name, "get_weather");
        assert_eq!(args["location"], "San Francisco, CA");
        assert_eq!(args["unit"], "fahrenheit");
    }

    #[tokio::test]
    async fn test_mistralai_mistral_7b_instruct_v03_single_with_start_tokenwith_new_lines() {
        let input = r#"
        [TOOL_CALLS]
        [{"name": "get_weather",
        "arguments": {"location":
        "San Francisco, CA",
        "unit": "fahrenheit"}}]
        "#;
        let config = ToolCallConfig::mistral();
        let (result, content) = try_tool_call_parse(input, &config).await.unwrap();
        assert_eq!(content, Some("".to_string()));
        assert!(!result.is_empty());
        assert_eq!(result.len(), 1);
        let (name, args) = extract_name_and_args(result[0].clone());
        assert_eq!(name, "get_weather");
        assert_eq!(args["location"], "San Francisco, CA");
        assert_eq!(args["unit"], "fahrenheit");
    }

    #[tokio::test]
    async fn test_mistralai_mistral_7b_instruct_v03_single_with_start_token_multiple() {
        let input = r#"[TOOL_CALLS] [{"name": "get_weather", "arguments": {"location": "San Francisco, CA", "unit": "fahrenheit"}}, {"name": "get_weather", "arguments": {"location": "New York, NY", "unit": "fahrenheit"}}]"#;
        let config = ToolCallConfig::mistral();
        let (result, content) = try_tool_call_parse(input, &config).await.unwrap();
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

    #[tokio::test]
    async fn test_mistralai_mistral_7b_instruct_v03_single_with_start_token_multiple_with_normal_text()
     {
        let input = r#"Hey How are you? [TOOL_CALLS] [{"name": "get_weather", "arguments": {"location": "San Francisco, CA", "unit": "fahrenheit"}}, {"name": "get_weather", "arguments": {"location": "New York, NY", "unit": "fahrenheit"}}]"#;
        let config = ToolCallConfig::mistral();
        let (result, content) = try_tool_call_parse(input, &config).await.unwrap();
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

    #[tokio::test]
    async fn test_mistralai_mistral_7b_instruct_v03_single_with_start_token_multiple_with_new_lines()
     {
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
        let (result, content) = try_tool_call_parse(input, &config).await.unwrap();
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

    #[tokio::test]
    async fn test_meta_llama_llama31_8b_instruct_simple() {
        let input = r#"{"name": "get_weather", "parameters": {"location": "San Francisco, CA", "unit": "fahrenheit"}}"#;
        let (result, content) = try_tool_call_parse(input, &ToolCallConfig::mistral())
            .await
            .unwrap();
        assert_eq!(content, Some("".to_string()));
        assert!(!result.is_empty());
        assert_eq!(result.len(), 1);
        let (name, args) = extract_name_and_args(result[0].clone());
        assert_eq!(name, "get_weather");
        assert_eq!(args["location"], "San Francisco, CA");
        assert_eq!(args["unit"], "fahrenheit");
    }

    #[tokio::test]
    async fn test_meta_llama_llama31_8b_instruct_simple_with_normal_text() {
        let input = r#"Hey How are you? {"name": "get_weather", "parameters": {"location": "San Francisco, CA", "unit": "fahrenheit"}}"#;
        let (result, content) = try_tool_call_parse(input, &ToolCallConfig::mistral())
            .await
            .unwrap();
        assert_eq!(content, Some("Hey How are you?".to_string()));
        assert!(!result.is_empty());
        assert_eq!(result.len(), 1);
        let (name, args) = extract_name_and_args(result[0].clone());
        assert_eq!(name, "get_weather");
        assert_eq!(args["location"], "San Francisco, CA");
        assert_eq!(args["unit"], "fahrenheit");
    }

    #[tokio::test]
    async fn test_meta_llama_llama31_8b_instruct_with_new_lines() {
        let input = r#"
        {"name": "get_weather",
        "parameters": {"location": "San Francisco, CA", "unit": "fahrenheit"}}
        "#;
        let (result, content) = detect_and_parse_tool_call(input, Some("llama3_json"))
            .await
            .unwrap();
        assert_eq!(content, Some("".to_string()));
        assert!(!result.is_empty());
        assert_eq!(result.len(), 1);
        let (name, args) = extract_name_and_args(result[0].clone());
        assert_eq!(name, "get_weather");
        assert_eq!(args["location"], "San Francisco, CA");
        assert_eq!(args["unit"], "fahrenheit");
    }

    #[tokio::test]
    async fn test_meta_llama_llama31_8b_instruct_with_python_tag() {
        let input = r#"<|python_tag|>{ "name": "get_weather", "parameters": {"location": "San Francisco, CA", "unit": "fahrenheit" } }"#;
        let (result, content) = detect_and_parse_tool_call(input, Some("llama3_json"))
            .await
            .unwrap();
        assert_eq!(content, Some("".to_string()));
        assert!(!result.is_empty());
        assert_eq!(result.len(), 1);
        let (name, args) = extract_name_and_args(result[0].clone());
        assert_eq!(name, "get_weather");
        assert_eq!(args["location"], "San Francisco, CA");
        assert_eq!(args["unit"], "fahrenheit");
    }

    #[tokio::test]
    async fn test_meta_llama_llama31_8b_instruct_with_python_tag_with_normal_text() {
        let input = r#"Hey How are you? <|python_tag|>{ "name": "get_weather", "parameters": {"location": "San Francisco, CA", "unit": "fahrenheit" } }"#;
        let (result, content) = detect_and_parse_tool_call(input, Some("llama3_json"))
            .await
            .unwrap();
        assert_eq!(content, Some("Hey How are you?".to_string()));
        assert!(!result.is_empty());
        assert_eq!(result.len(), 1);
        let (name, args) = extract_name_and_args(result[0].clone());
        assert_eq!(name, "get_weather");
        assert_eq!(args["location"], "San Francisco, CA");
        assert_eq!(args["unit"], "fahrenheit");
    }

    #[tokio::test]
    async fn test_meta_llama_llama31_8b_instruct_with_python_tag_with_new_lines() {
        let input = r#"
        <|python_tag|>
        {"name": "get_weather", "parameters": {"location": "San Francisco, CA", "unit": "fahrenheit"}}
        "#;
        let (result, content) = detect_and_parse_tool_call(input, Some("llama3_json"))
            .await
            .unwrap();
        assert_eq!(content, Some("".to_string()));
        assert!(!result.is_empty());
        assert_eq!(result.len(), 1);
        let (name, args) = extract_name_and_args(result[0].clone());
        assert_eq!(name, "get_weather");
        assert_eq!(args["location"], "San Francisco, CA");
        assert_eq!(args["unit"], "fahrenheit");
    }

    #[tokio::test]
    async fn test_meta_llama_llama31_8b_instruct_with_python_tag_multiple_with_new_lines() {
        let input = r#"
        <|python_tag|>
        {"name": "get_weather", "parameters": {"location": "San Francisco, CA", "unit": "fahrenheit" }}
        <|python_tag|>
        {"name": "get_weather", "parameters": {"location": "New York, NY", "unit": "fahrenheit" }}
        "#;
        let (result, content) = detect_and_parse_tool_call(input, Some("llama3_json"))
            .await
            .unwrap();
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

    #[tokio::test]
    async fn test_detect_and_parse_tool_call_error_handling() {
        // Unknown parser string should return an error
        let input = r#"{"name": "get_weather", "arguments": {"location": "San Francisco, CA"}}"#;
        let result = detect_and_parse_tool_call(input, Some("unknown_parser")).await;
        assert!(result.is_err());
        let err = result.unwrap_err().to_string();
        assert!(
            err.contains("is not implemented"),
            "Unexpected error message: {}",
            err
        );

        // Known parser, but invalid input (not JSON) should return Ok(None)
        let input = "not a json";
        let (result, content) = detect_and_parse_tool_call(input, Some("hermes"))
            .await
            .unwrap();
        assert_eq!(content, Some("not a json".to_string()));
        assert!(result.is_empty());

        // Known parser, but valid JSON with wrong shape should return Ok(None)
        let input = r#"{"foo": "bar"}"#;
        let (result, content) = detect_and_parse_tool_call(input, Some("hermes"))
            .await
            .unwrap();
        assert_eq!(content, Some(r#"{"foo": "bar"}"#.to_string()));
        assert!(result.is_empty());
    }

    #[tokio::test]
    #[ignore]
    async fn test_internlm_internlm2_5_7b_chat_simple() {
        let input = r#"San Francisco's weather is known for its mild climate with plenty of fog, especially along the coast. Here's an overview of the weather in Fahrenheit:

- **Summer (June to August)**: Average highs range from the mid-60s to low 70s Fahrenheit, with cooler mornings and evenings. Coastal areas may be cooler than inland spots.

Remember, San Francisco weather can be quite unpredictable, particularly with its famous fog, which can significantly lower temperatures. Always check a local weather forecast for the most accurate and up-to-date information."#;
        let (result, content) = try_tool_call_parse(input, &ToolCallConfig::default())
            .await
            .unwrap();
        assert_eq!(content, Some(input.to_string()));
        assert!(result.is_empty()); // This model doesn't produce tool calls
    }

    #[tokio::test]
    #[ignore]
    async fn test_ai21labs_ai21_jamba_15_mini_simple() {
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
        let (result, content) = try_tool_call_parse(input, &config).await.unwrap();
        assert_eq!(content, Some("".to_string()));
        assert!(!result.is_empty());
        assert_eq!(result.len(), 1);
        let (name, args) = extract_name_and_args(result[0].clone());
        assert_eq!(name, "get_weather");
        assert_eq!(args["location"], "San Francisco, CA");
        assert_eq!(args["unit"], "fahrenheit");
    }

    #[tokio::test]
    #[ignore]
    async fn test_salesforce_llama_xlam_2_8b_fc_r_simple() {
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
        let (result, content) = try_tool_call_parse(input, &config).await.unwrap();
        assert_eq!(content, Some("".to_string()));
        assert!(!result.is_empty());
        assert_eq!(result.len(), 1);
        let (name, args) = extract_name_and_args(result[0].clone());
        assert_eq!(name, "get_weather");
        assert_eq!(args["location"], "San Francisco, CA");
        assert_eq!(args["unit"], "fahrenheit");
    }

    #[tokio::test]
    async fn test_detect_and_parse_tool_call_default_parser_nemotron_deci() {
        let input = r#"<TOOLCALL>[{"name": "get_weather", "arguments": {"location": "San Francisco, CA", "unit": "fahrenheit"}}]</TOOLCALL>"#;
        let (result, content) = detect_and_parse_tool_call(input, None).await.unwrap();
        assert_eq!(content, Some("".to_string()));
        assert!(!result.is_empty());
        assert_eq!(result.len(), 1);
        let (name, args) = extract_name_and_args(result[0].clone());
        assert_eq!(name, "get_weather");
        assert_eq!(args["location"], "San Francisco, CA");
        assert_eq!(args["unit"], "fahrenheit");
    }

    #[tokio::test]
    async fn test_detect_and_parse_tool_call_default_parser_nemotron_deci_multiple() {
        let input = r#"<TOOLCALL>[{"name": "get_weather", "arguments": {"location": "San Francisco, CA", "unit": "fahrenheit"}}, {"name": "get_weather", "arguments": {"location": "New York, NY", "unit": "fahrenheit"}}]</TOOLCALL>"#;
        let (result, content) = detect_and_parse_tool_call(input, None).await.unwrap();
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

    #[tokio::test]
    async fn test_detect_and_parse_tool_call_default_parser_nemotron_deci_multiple_with_normal_text()
     {
        let input = r#"Hey How are you? <TOOLCALL>[{"name": "get_weather", "arguments": {"location": "San Francisco, CA", "unit": "fahrenheit"}}, {"name": "get_weather", "arguments": {"location": "New York, NY", "unit": "fahrenheit"}}]</TOOLCALL>"#;
        let (result, content) = detect_and_parse_tool_call(input, None).await.unwrap();
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

    #[tokio::test]
    async fn test_detect_and_parse_tool_call_default_parser_llama3_json_with_python_tag() {
        let input = r#"<|python_tag|>{ "name": "get_weather", "arguments": {"location": "San Francisco, CA", "unit": "fahrenheit" } }"#;
        let (result, content) = detect_and_parse_tool_call(input, None).await.unwrap();
        assert_eq!(content, Some("".to_string()));
        assert!(!result.is_empty());
        assert_eq!(result.len(), 1);
        let (name, args) = extract_name_and_args(result[0].clone());
        assert_eq!(name, "get_weather");
        assert_eq!(args["location"], "San Francisco, CA");
        assert_eq!(args["unit"], "fahrenheit");
    }

    #[tokio::test]
    async fn test_detect_and_parse_tool_call_default_parser_llama3_json_with_python_tag_with_normal_text()
     {
        let input = r#"Hey How are you? <|python_tag|>{ "name": "get_weather", "arguments": {"location": "San Francisco, CA", "unit": "fahrenheit" } }"#;
        let (result, content) = detect_and_parse_tool_call(input, None).await.unwrap();
        assert_eq!(content, Some("Hey How are you?".to_string()));
        assert!(!result.is_empty());
        assert_eq!(result.len(), 1);
        let (name, args) = extract_name_and_args(result[0].clone());
        assert_eq!(name, "get_weather");
        assert_eq!(args["location"], "San Francisco, CA");
        assert_eq!(args["unit"], "fahrenheit");
    }

    #[tokio::test]
    async fn test_detect_and_parse_tool_call_default_parser_llama3_json_with_python_tag_with_new_lines()
     {
        let input = r#"
        <|python_tag|>
        {"name":
        "get_weather",
         "arguments":
          {"location": "San Francisco, CA",
          "unit": "fahrenheit" }}
        "#;
        let (result, content) = detect_and_parse_tool_call(input, None).await.unwrap();
        assert_eq!(content, Some("".to_string()));
        assert!(!result.is_empty());
        assert_eq!(result.len(), 1);
        let (name, args) = extract_name_and_args(result[0].clone());
        assert_eq!(name, "get_weather");
        assert_eq!(args["location"], "San Francisco, CA");
        assert_eq!(args["unit"], "fahrenheit");
    }

    #[tokio::test]
    async fn test_detect_and_parse_tool_call_default_parser_llama3_json_without_python_tag_multiple_with_new_lines()
     {
        let input = r#"
        {"name": "get_weather", "arguments":
         {"location": "San Francisco, CA",
          "unit": "fahrenheit" }}
        "#;
        let (result, content) = detect_and_parse_tool_call(input, None).await.unwrap();
        assert_eq!(content, Some("".to_string()));
        assert!(!result.is_empty());
        assert_eq!(result.len(), 1);
        let (name, args) = extract_name_and_args(result[0].clone());
        assert_eq!(name, "get_weather");
        assert_eq!(args["location"], "San Francisco, CA");
        assert_eq!(args["unit"], "fahrenheit");
    }

    #[tokio::test]
    async fn test_detect_and_parse_tool_call_default_parser_llama3_json_without_python_tag() {
        let input = r#"{ "name": "get_weather", "arguments": {"location": "San Francisco, CA", "unit": "fahrenheit" } }"#;
        let (result, content) = try_tool_call_parse(input, &ToolCallConfig::mistral())
            .await
            .unwrap();
        assert_eq!(content, Some("".to_string()));
        assert!(!result.is_empty());
        assert_eq!(result.len(), 1);
        let (name, args) = extract_name_and_args(result[0].clone());
        assert_eq!(name, "get_weather");
        assert_eq!(args["location"], "San Francisco, CA");
        assert_eq!(args["unit"], "fahrenheit");
    }

    #[tokio::test]
    async fn test_detect_and_parse_tool_call_default_parser_llama3_json_without_python_tag_with_normal_text()
     {
        let input = r#"Hey How are you? { "name": "get_weather", "arguments": {"location": "San Francisco, CA", "unit": "fahrenheit" } }"#;
        let (result, content) = try_tool_call_parse(input, &ToolCallConfig::mistral())
            .await
            .unwrap();
        assert_eq!(content, Some("Hey How are you?".to_string()));
        assert!(!result.is_empty());
        assert_eq!(result.len(), 1);
        let (name, args) = extract_name_and_args(result[0].clone());
        assert_eq!(name, "get_weather");
        assert_eq!(args["location"], "San Francisco, CA");
        assert_eq!(args["unit"], "fahrenheit");
    }

    #[tokio::test]
    async fn test_phi4_single_function_call() {
        let input =
            r#"functools[{"name": "get_country_capital", "arguments": {"country": "Poland"}}]"#;
        let (result, content) = detect_and_parse_tool_call(input, Some("phi4"))
            .await
            .unwrap();
        assert_eq!(content, Some("".to_string()));
        assert_eq!(result.len(), 1);
        let (name, args) = extract_name_and_args(result[0].clone());
        assert_eq!(name, "get_country_capital");
        assert_eq!(args["country"], "Poland");
    }

    #[tokio::test]
    async fn test_phi4_single_function_call_with_normal_text() {
        let input = r#"Hey How are you? functools[{"name": "get_country_capital", "arguments": {"country": "Poland"}}]"#;
        let (result, content) = detect_and_parse_tool_call(input, Some("phi4"))
            .await
            .unwrap();
        assert_eq!(content, Some("Hey How are you?".to_string()));
        assert_eq!(result.len(), 1);
        let (name, args) = extract_name_and_args(result[0].clone());
        assert_eq!(name, "get_country_capital");
        assert_eq!(args["country"], "Poland");
    }

    #[tokio::test]
    async fn test_phi4_multiple_function_calls_simple_arguments() {
        let input = r#"functools[
  {"name": "get_country_capital", "arguments": {"country": "Poland"}},
  {"name": "get_population", "arguments": {"city": "Warsaw"}}
]"#;
        let (result, content) = detect_and_parse_tool_call(input, Some("phi4"))
            .await
            .unwrap();
        assert_eq!(content, Some("".to_string()));
        assert_eq!(result.len(), 2);

        let (name1, args1) = extract_name_and_args(result[0].clone());
        assert_eq!(name1, "get_country_capital");
        assert_eq!(args1["country"], "Poland");

        let (name2, args2) = extract_name_and_args(result[1].clone());
        assert_eq!(name2, "get_population");
        assert_eq!(args2["city"], "Warsaw");
    }

    #[tokio::test]
    async fn test_phi4_multiple_function_calls_simple_arguments_with_normal_text() {
        let input = r#"Hey How are you? functools[
  {"name": "get_country_capital", "arguments": {"country": "Poland"}},
  {"name": "get_population", "arguments": {"city": "Warsaw"}}
]"#;
        let (result, content) = detect_and_parse_tool_call(input, Some("phi4"))
            .await
            .unwrap();
        assert_eq!(content, Some("Hey How are you?".to_string()));
        assert_eq!(result.len(), 2);

        let (name1, args1) = extract_name_and_args(result[0].clone());
        assert_eq!(name1, "get_country_capital");
        assert_eq!(args1["country"], "Poland");

        let (name2, args2) = extract_name_and_args(result[1].clone());
        assert_eq!(name2, "get_population");
        assert_eq!(args2["city"], "Warsaw");
    }

    #[tokio::test]
    async fn test_phi4_single_function_call_nested_json_arguments() {
        let input = r#"functools[{"name": "get_weather_forecast", "arguments":
        {"location": {"city": "San Francisco",
        "state": "CA"}, "date": "2023-10-05"}}]"#;
        let (result, content) = detect_and_parse_tool_call(input, Some("phi4"))
            .await
            .unwrap();
        assert_eq!(content, Some("".to_string()));
        assert_eq!(result.len(), 1);
        let (name, args) = extract_name_and_args(result[0].clone());
        assert_eq!(name, "get_weather_forecast");
        assert_eq!(args["date"], "2023-10-05");
        assert_eq!(args["location"]["city"], "San Francisco");
        assert_eq!(args["location"]["state"], "CA");
    }

    #[tokio::test]
    async fn test_phi4_single_function_call_nested_json_arguments_with_normal_text() {
        let input = r#"Hey How are you? functools[{"name": "get_weather_forecast", "arguments":
        {"location": {"city": "San Francisco",
        "state": "CA"}, "date": "2023-10-05"}}]"#;
        let (result, content) = detect_and_parse_tool_call(input, Some("phi4"))
            .await
            .unwrap();
        assert_eq!(content, Some("Hey How are you?".to_string()));
        assert_eq!(result.len(), 1);
        let (name, args) = extract_name_and_args(result[0].clone());
        assert_eq!(name, "get_weather_forecast");
        assert_eq!(args["date"], "2023-10-05");
        assert_eq!(args["location"]["city"], "San Francisco");
        assert_eq!(args["location"]["state"], "CA");
    }

    #[tokio::test]
    async fn test_phi4_function_call_with_parameters_instead_of_arguments() {
        let input = r#"functools[{"name": "calculate_distance",
         "parameters": {"from": "New York", "to": "Los Angeles"}}]"#;
        let (result, content) = detect_and_parse_tool_call(input, Some("phi4"))
            .await
            .unwrap();
        assert_eq!(content, Some("".to_string()));
        assert_eq!(result.len(), 1);
        let (name, args) = extract_name_and_args(result[0].clone());
        assert_eq!(name, "calculate_distance");
        assert_eq!(args["from"], "New York");
        assert_eq!(args["to"], "Los Angeles");
    }

    #[tokio::test]
    async fn test_phi4_function_call_with_parameters_instead_of_arguments_with_normal_text() {
        let input = r#"Hey How are you? functools[{"name": "calculate_distance",
         "parameters": {"from": "New York", "to": "Los Angeles"}}]"#;
        let (result, content) = detect_and_parse_tool_call(input, Some("phi4"))
            .await
            .unwrap();
        assert_eq!(content, Some("Hey How are you?".to_string()));
        assert_eq!(result.len(), 1);
        let (name, args) = extract_name_and_args(result[0].clone());
        assert_eq!(name, "calculate_distance");
        assert_eq!(args["from"], "New York");
        assert_eq!(args["to"], "Los Angeles");
    }

    #[tokio::test]
    async fn test_phi4_token_leak_reproduction() {
        // Reproduce the issue where "functools" appears in content field
        // This might happen when there's malformed JSON or parsing issues
        let input = r#"functools{"name": "get_weather","arguments":{"location":"San Francisco"}}"#;
        let (result, content) = detect_and_parse_tool_call(input, Some("phi4"))
            .await
            .unwrap();
        // Content should be empty, not contain "functools"
        assert_eq!(content, Some("".to_string()));
        assert_eq!(result.len(), 1);
        let (name, args) = extract_name_and_args(result[0].clone());
        assert_eq!(name, "get_weather");
        assert_eq!(args["location"], "San Francisco");
    }

    #[tokio::test]
    async fn test_phi4_token_leak_edge_case() {
        // Test the case where only the token appears without JSON
        // This case is less critical but shouldn't leak the full token
        let input = r#"functools"#;
        let (result, _content) = detect_and_parse_tool_call(input, Some("phi4"))
            .await
            .unwrap();
        // Content may contain the token if no valid JSON follows, but shouldn't crash
        // The important thing is that no tool calls are returned
        assert_eq!(result.len(), 0); // No tool calls found
        // Content behavior is less critical for this edge case
    }

    #[tokio::test]
    async fn test_phi4_token_with_invalid_json() {
        // Test the case where token is followed by invalid JSON
        let input = r#"functools{invalid json}"#;
        let (result, content) = detect_and_parse_tool_call(input, Some("phi4"))
            .await
            .unwrap();
        // Content should be empty, not contain "functools" or leak the token
        assert_eq!(content, Some("".to_string()));
        assert_eq!(result.len(), 0); // No tool calls found due to invalid JSON
    }

    #[tokio::test]
    async fn test_phi4_streaming_partial_tokens() {
        // Test that our fix handles the actual streaming scenario described by the user
        // Where "fun", "ct", "ools" arrive as separate chunks

        // Test that "fun" is detected as a potential tool call start (for streaming jailing)
        let config = super::get_tool_parser_map().get("phi4").unwrap();

        // Test detection of partial tokens
        use super::super::json::detect_tool_call_start_json;
        assert!(
            detect_tool_call_start_json("fun", &config.json),
            "'fun' should be detected as potential start"
        );
        assert!(
            detect_tool_call_start_json("f", &config.json),
            "'f' should be detected as potential start"
        );
        assert!(
            detect_tool_call_start_json("func", &config.json),
            "'func' should be detected as potential start"
        );
        assert!(
            detect_tool_call_start_json("functo", &config.json),
            "'functo' should be detected as potential start"
        );

        // Test that unrelated text is not detected
        assert!(
            !detect_tool_call_start_json("hello", &config.json),
            "'hello' should not be detected"
        );
        assert!(
            !detect_tool_call_start_json("xyz", &config.json),
            "'xyz' should not be detected"
        );
    }

    #[tokio::test]
    async fn test_phi4_false_positive_words() {
        // Test that words like "funk" or text starting with "func" but not "functools"
        // are correctly treated as normal content, not tool calls

        let input = r#"funk music is great"#;
        let (result, content) = detect_and_parse_tool_call(input, Some("phi4"))
            .await
            .unwrap();
        // Should be treated as normal content, not tool call
        assert_eq!(
            result.len(),
            0,
            "No tool calls should be found in 'funk music is great'"
        );
        assert_eq!(
            content,
            Some("funk music is great".to_string()),
            "Content should contain the original text"
        );
    }

    #[tokio::test]
    async fn test_phi4_partial_but_complete_words() {
        // Test words that start with "func" but are not "functools"

        let input = r#"The function works well"#;
        let (result, content) = detect_and_parse_tool_call(input, Some("phi4"))
            .await
            .unwrap();
        assert_eq!(
            result.len(),
            0,
            "No tool calls should be found in 'The function works well'"
        );
        assert_eq!(content, Some("The function works well".to_string()));

        let input = r#"functional programming"#;
        let (result, content) = detect_and_parse_tool_call(input, Some("phi4"))
            .await
            .unwrap();
        assert_eq!(
            result.len(),
            0,
            "No tool calls should be found in 'functional programming'"
        );
        assert_eq!(content, Some("functional programming".to_string()));
    }

    #[tokio::test]
    async fn test_phi4_funk_variations() {
        // Test various "funk" related words to ensure they're not treated as tool calls

        let test_cases = vec![
            "funk",
            "funky",
            "funktion", // German word for function
            "funked",
            "I love funk music",
            "This is funky stuff",
        ];

        for test_input in test_cases {
            let (result, content) = detect_and_parse_tool_call(test_input, Some("phi4"))
                .await
                .unwrap();
            assert_eq!(
                result.len(),
                0,
                "No tool calls should be found in '{}'",
                test_input
            );
            assert_eq!(
                content,
                Some(test_input.to_string()),
                "Content should match input for '{}'",
                test_input
            );
        }
    }

    #[tokio::test]
    async fn test_phi4_func_but_not_functools() {
        // Test words starting with "func" that are complete words, not partial "functools"

        let test_cases = vec![
            "func()",  // Programming syntax
            "funcdef", // Python keyword variant
            "functions are useful",
            "functionally speaking",
        ];

        for test_input in test_cases {
            let (result, content) = detect_and_parse_tool_call(test_input, Some("phi4"))
                .await
                .unwrap();
            assert_eq!(
                result.len(),
                0,
                "No tool calls should be found in '{}'",
                test_input
            );
            assert_eq!(
                content,
                Some(test_input.to_string()),
                "Content should match input for '{}'",
                test_input
            );
        }
    }

    #[tokio::test]
    async fn test_pythonic_parser_basic_with_constants() {
        let input = r#"[get_weather(location="San Francisco", unit="fahrenheit"), get_weather(location="New York", unit="fahrenheit")]"#;
        let (result, content) = detect_and_parse_tool_call(input, Some("pythonic"))
            .await
            .unwrap();
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

    #[tokio::test]
    #[ignore]
    async fn test_pythonic_parser_with_constants_and_normal_text() {
        let input = r#"Hey How are you? [get_weather(location="San Francisco", unit="fahrenheit"), get_weather(location="New York", unit="fahrenheit")]"#;
        let (result, content) = detect_and_parse_tool_call(input, Some("pythonic"))
            .await
            .unwrap();
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

    #[tokio::test]
    async fn test_harmony_parser_basic() {
        let input = r#"
        <|channel|>analysis<|message|>Need to use function get_current_weather.<|end|>
        <|start|>assistant<|channel|>commentary to=functions.get_current_weather <|constrain|>json
        <|message|>{"location":"San Francisco", "unit":"fahrenheit"}<|call|>
        "#;
        let (result, content) = detect_and_parse_tool_call(input, Some("harmony"))
            .await
            .unwrap();
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

    #[tokio::test]
    async fn test_deepseek_v3_1_parser_basic() {
        let input = r#"<｜tool▁calls▁begin｜><｜tool▁call▁begin｜>get_current_weather<｜tool▁sep｜>{"location": "Tokyo"}<｜tool▁call▁end｜><｜tool▁call▁begin｜>get_current_weather<｜tool▁sep｜>{"location": "Paris"}<｜tool▁call▁end｜><｜tool▁calls▁end｜><｜end▁of▁sentence｜>"#;
        let (result, content) = detect_and_parse_tool_call(input, Some("deepseek_v3_1"))
            .await
            .unwrap();
        assert_eq!(content, Some("".to_string()));
        assert_eq!(result.len(), 2);
        let (name, args) = extract_name_and_args(result[0].clone());
        assert_eq!(name, "get_current_weather");
        assert_eq!(args["location"], "Tokyo");
        let (name, args) = extract_name_and_args(result[1].clone());
        assert_eq!(name, "get_current_weather");
        assert_eq!(args["location"], "Paris");
    }

    #[tokio::test]
    async fn test_hermes_parser_without_new_line() {
        let input = r#"<tool_call>{"name": "get_weather", "arguments": {"location": "San Francisco, CA", "unit": "celsius"}}</tool_call>"
        "#;
        let (result, content) = detect_and_parse_tool_call(input, Some("hermes"))
            .await
            .unwrap();
        assert_eq!(content, Some("".to_string()));
        assert_eq!(result.len(), 1);
        let (name, args) = extract_name_and_args(result[0].clone());
        assert_eq!(name, "get_weather");
        assert_eq!(args["location"], "San Francisco, CA");
        assert_eq!(args["unit"], "celsius");
    }
}

#[cfg(test)]
// Just e2e tests to test the flow. Detailed tests are covered in the individual parsers
mod detect_parser_tests {
    use super::*;

    #[test]
    fn test_e2e_detect_tool_call_start_harmony() {
        let text = r#"<|start|>assistant<|channel|>commentary to=functions.get_current_weather <|constrain|>json"#;
        let result = detect_tool_call_start(text, Some("harmony")).unwrap();
        assert!(result);
    }

    #[test]
    fn test_e2e_detect_tool_call_start_hermes() {
        let text = r#"{"name": "get_current_weather", "parameters": {"location": "Tokyo"}}"#;
        let result = detect_tool_call_start(text, Some("hermes")).unwrap();
        assert!(result);
    }

    #[test]
    fn test_e2e_detect_tool_call_start_pythonic() {
        let text = r#"foo(a=1, b=2), bar(x=3)]"#;
        let result = detect_tool_call_start(text, Some("pythonic")).unwrap();
        assert!(!result);
    }

    #[test]
    fn test_e2e_detect_tool_call_start_nemotron_deci() {
        let text = r#"<TOOLCALL>[{"name": "get_current_weather", "parameters": {"location": "Tokyo"}}]</TOOLCALL>"#;
        let result = detect_tool_call_start(text, Some("nemotron_deci")).unwrap();
        assert!(result);
    }

    #[test]
    fn test_e2e_detect_tool_call_start_phi4() {
        let text =
            r#"functools{"name": "get_current_weather", "parameters": {"location": "Tokyo"}}"#;
        let result = detect_tool_call_start(text, Some("phi4")).unwrap();
        assert!(result);
    }

    #[test]
    fn test_e2e_detect_tool_call_start_llama3_json() {
        let text = r#"<|python_tag|>{ "name": "get_current_weather", "parameters": {"location": "Tokyo"}}"#;
        let result = detect_tool_call_start(text, Some("llama3_json")).unwrap();
        assert!(result);
    }

    #[test]
    fn test_e2e_detect_tool_call_start_mistral() {
        let text =
            r#"[TOOL_CALLS]{"name": "get_current_weather", "parameters": {"location": "Tokyo"}}"#;
        let result = detect_tool_call_start(text, Some("mistral")).unwrap();
        assert!(result);
    }

    #[test]
    fn test_e2e_detect_tool_call_start_deepseek_v3_1() {
        let text = r#"<｜tool▁calls▁begin｜><｜tool▁call▁begin｜>get_current_weather{"location": "Tokyo"}<｜tool▁call▁end｜>"#;
        let result = detect_tool_call_start(text, Some("deepseek_v3_1")).unwrap();
        assert!(result);
    }
}
