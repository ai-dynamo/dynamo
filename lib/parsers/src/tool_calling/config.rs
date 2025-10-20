// SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use super::json::JsonParserType;

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
    /// Start token for individual tool calls (e.g., "<TOOLCALL>")
    pub tool_call_start_tokens: Vec<String>,
    /// End token for individual tool calls (e.g., "</TOOLCALL>")
    pub tool_call_end_tokens: Vec<String>,
    /// Separator tokens between function name and arguments
    /// (e.g., "<｜tool▁sep｜>" for DeepSeek v3.1)
    /// Used by some models to separate function name from arguments
    pub tool_call_separator_tokens: Vec<String>,
    /// The key for the function name in the tool call
    /// i.e. `{"name": "function", "arguments": {...}}` it would be
    /// "name"
    pub function_name_keys: Vec<String>,
    /// The key for the arguments in the tool call
    /// i.e. `{"name": "function", "arguments": {...}}` it would be
    /// "arguments"
    pub arguments_keys: Vec<String>,

    /// The type of JSON parser to use
    #[serde(default)]
    pub parser_type: JsonParserType,
}

impl Default for JsonParserConfig {
    fn default() -> Self {
        Self {
            tool_call_start_tokens: vec!["<TOOLCALL>".to_string(), "<|python_tag|>".to_string()],
            tool_call_end_tokens: vec!["</TOOLCALL>".to_string(), "".to_string()],
            tool_call_separator_tokens: vec![],
            function_name_keys: vec!["name".to_string()],
            arguments_keys: vec!["arguments".to_string(), "parameters".to_string()],
            parser_type: JsonParserType::Basic,
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
                tool_call_end_tokens: vec!["</tool_call>".to_string()],
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
                tool_call_end_tokens: vec!["[/TOOL_CALLS]".to_string(), "".to_string()],
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

    pub fn pythonic() -> Self {
        Self {
            format: ToolCallParserType::Pythonic,
            json: JsonParserConfig::default(), // This is noop here, but we keep it for consistency
        }
    }

    pub fn harmony() -> Self {
        Self {
            format: ToolCallParserType::Harmony,
            json: JsonParserConfig {
                tool_call_start_tokens: vec!["<|start|>assistant<|channel|>commentary".to_string()],
                tool_call_end_tokens: vec!["<|call|>".to_string()],
                ..Default::default()
            },
        }
    }

    pub fn deepseek_v3_1() -> Self {
        Self {
            format: ToolCallParserType::Json,
            json: JsonParserConfig {
                tool_call_start_tokens: vec![
                    "<｜tool▁calls▁begin｜>".to_string(),
                    "<｜tool▁call▁begin｜>".to_string(),
                ],
                tool_call_end_tokens: vec![
                    "<｜tool▁calls▁end｜>".to_string(),
                    "<｜tool▁call▁end｜>".to_string(),
                ],
                tool_call_separator_tokens: vec!["<｜tool▁sep｜>".to_string()],
                parser_type: JsonParserType::DeepseekV31,
                ..Default::default()
            },
        }
    }
}
