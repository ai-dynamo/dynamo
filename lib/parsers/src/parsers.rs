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