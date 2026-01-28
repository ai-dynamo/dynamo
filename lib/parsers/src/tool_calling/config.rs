// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use super::json::JsonParserType;

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

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct XmlParserConfig {
    /// Start token for individual tool calls (e.g., "<tool_call>")
    pub tool_call_start_token: String,
    /// End token for individual tool calls (e.g., "</tool_call>")
    pub tool_call_end_token: String,
    /// Start token for function name (e.g., "<function=")
    pub function_start_token: String,
    /// End token for function (e.g., "</function>")
    pub function_end_token: String,
    /// Start token for parameter (e.g., "<parameter=")
    pub parameter_start_token: String,
    /// End token for parameter (e.g., "</parameter>")
    pub parameter_end_token: String,
}

impl Default for XmlParserConfig {
    fn default() -> Self {
        Self {
            tool_call_start_token: "<tool_call>".to_string(),
            tool_call_end_token: "</tool_call>".to_string(),
            function_start_token: "<function=".to_string(),
            function_end_token: "</function>".to_string(),
            parameter_start_token: "<parameter=".to_string(),
            parameter_end_token: "</parameter>".to_string(),
        }
    }
}

/// Configuration for DSML-style tool call parser (DeepSeek V3.2+)
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct DsmlParserConfig {
    /// Start token for function_calls block (e.g., "<｜DSML｜function_calls>")
    pub function_calls_start: String,
    /// End token for function_calls block (e.g., "</｜DSML｜function_calls>")
    pub function_calls_end: String,
    /// Start prefix for invoke (e.g., "<｜DSML｜invoke name=")
    pub invoke_start_prefix: String,
    /// End token for invoke (e.g., "</｜DSML｜invoke>")
    pub invoke_end: String,
    /// Start prefix for parameter (e.g., "<｜DSML｜parameter name=")
    pub parameter_prefix: String,
    /// End token for parameter (e.g., "</｜DSML｜parameter>")
    pub parameter_end: String,
}

impl Default for DsmlParserConfig {
    fn default() -> Self {
        Self {
            function_calls_start: "<｜DSML｜function_calls>".to_string(),
            function_calls_end: "</｜DSML｜function_calls>".to_string(),
            invoke_start_prefix: "<｜DSML｜invoke name=".to_string(),
            invoke_end: "</｜DSML｜invoke>".to_string(),
            parameter_prefix: "<｜DSML｜parameter name=".to_string(),
            parameter_end: "</｜DSML｜parameter>".to_string(),
        }
    }
}

/// Parser-specific configuration
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ParserConfig {
    Json(JsonParserConfig),
    Xml(XmlParserConfig),
    Pythonic,
    Harmony(JsonParserConfig),
    Typescript,
    Dsml(DsmlParserConfig),
}

impl ParserConfig {
    /// Get the tool call start tokens for this parser configuration
    /// Returns a vector of start tokens that indicate the beginning of a tool call
    pub fn tool_call_start_tokens(&self) -> Vec<String> {
        match self {
            ParserConfig::Json(config) => config.tool_call_start_tokens.clone(),
            ParserConfig::Harmony(config) => config.tool_call_start_tokens.clone(),
            ParserConfig::Xml(config) => vec![config.tool_call_start_token.clone()],
            ParserConfig::Pythonic => vec![],
            ParserConfig::Typescript => vec![],
            ParserConfig::Dsml(config) => vec![config.function_calls_start.clone()],
        }
    }

    /// Get the tool call end tokens for this parser configuration
    /// Returns a vector of end tokens that indicate the end of a tool call
    pub fn tool_call_end_tokens(&self) -> Vec<String> {
        match self {
            ParserConfig::Json(config) => config.tool_call_end_tokens.clone(),
            ParserConfig::Harmony(config) => config.tool_call_end_tokens.clone(),
            ParserConfig::Xml(config) => vec![config.tool_call_end_token.clone()],
            ParserConfig::Pythonic => vec![],
            ParserConfig::Typescript => vec![],
            ParserConfig::Dsml(config) => vec![config.function_calls_end.clone()],
        }
    }
}

/// Configuration for parsing tool calls with different formats
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct ToolCallConfig {
    /// Parser-specific configuration.
    pub parser_config: ParserConfig,
}

impl Default for ToolCallConfig {
    fn default() -> Self {
        Self {
            parser_config: ParserConfig::Json(JsonParserConfig::default()),
        }
    }
}

impl ToolCallConfig {
    /// Default configuration for hermes tool calls
    /// <tool_call>{"name": "get_weather", "arguments": {"location": "San Francisco, CA", "unit": "fahrenheit"}}\n</tool_call>
    pub fn hermes() -> Self {
        Self {
            parser_config: ParserConfig::Json(JsonParserConfig {
                tool_call_start_tokens: vec!["<tool_call>".to_string()],
                tool_call_end_tokens: vec!["</tool_call>".to_string()],
                ..Default::default()
            }),
        }
    }

    /// Default configuration for nemotron tool calls
    /// <TOOLCALL>[{"name": "get_weather", "arguments": {"location": "San Francisco, CA", "unit": "fahrenheit"}}]</TOOLCALL>
    pub fn nemotron_deci() -> Self {
        Self {
            parser_config: ParserConfig::Json(JsonParserConfig {
                tool_call_start_tokens: vec!["<TOOLCALL>".to_string()],
                tool_call_end_tokens: vec!["</TOOLCALL>".to_string()],
                ..Default::default()
            }),
        }
    }

    pub fn llama3_json() -> Self {
        // <|python_tag|>{ "name": "get_weather", "arguments": {"location": "San Francisco, CA", "unit": "fahrenheit"} }
        // or { "name": "get_weather", "arguments": {"location": "San Francisco, CA", "unit": "fahrenheit"} }
        Self {
            parser_config: ParserConfig::Json(JsonParserConfig {
                tool_call_start_tokens: vec!["<|python_tag|>".to_string()],
                tool_call_end_tokens: vec!["".to_string()],
                ..Default::default()
            }),
        }
    }

    pub fn mistral() -> Self {
        Self {
            parser_config: ParserConfig::Json(JsonParserConfig {
                tool_call_start_tokens: vec!["[TOOL_CALLS]".to_string()],
                tool_call_end_tokens: vec!["[/TOOL_CALLS]".to_string(), "".to_string()],
                ..Default::default()
            }),
        }
    }

    pub fn phi4() -> Self {
        Self {
            parser_config: ParserConfig::Json(JsonParserConfig {
                tool_call_start_tokens: vec!["functools".to_string()],
                tool_call_end_tokens: vec!["".to_string()],
                ..Default::default()
            }),
        }
    }

    pub fn pythonic() -> Self {
        Self {
            parser_config: ParserConfig::Pythonic,
        }
    }

    pub fn harmony() -> Self {
        Self {
            parser_config: ParserConfig::Harmony(JsonParserConfig {
                tool_call_start_tokens: vec!["<|start|>assistant<|channel|>commentary".to_string()],
                tool_call_end_tokens: vec!["<|call|>".to_string()],
                ..Default::default()
            }),
        }
    }

    pub fn deepseek_v3_1() -> Self {
        // The whole tool calls block is wrapped between
        // <｜tool▁calls▁begin｜> ... <｜tool▁calls▁end｜>
        // regardless of number of tool calls. For external use of this
        // config, we want them to only be operating on the whole block,
        // so the tool parser can properly consume all tool call tokens.
        // https://huggingface.co/deepseek-ai/DeepSeek-V3.1#toolcall
        Self {
            parser_config: ParserConfig::Json(JsonParserConfig {
                tool_call_start_tokens: vec![
                    "<｜tool▁calls▁begin｜>".to_string(),
                    // "<｜tool▁call▁begin｜>".to_string(),
                ],
                tool_call_end_tokens: vec![
                    "<｜tool▁calls▁end｜>".to_string(),
                    // "<｜tool▁call▁end｜>".to_string(),
                ],
                tool_call_separator_tokens: vec!["<｜tool▁sep｜>".to_string()],
                parser_type: JsonParserType::DeepseekV31,
                ..Default::default()
            }),
        }
    }

    pub fn deepseek_v3() -> Self {
        // DeepSeek V3 format:
        // <｜tool▁calls▁begin｜><｜tool▁call▁begin｜>{type}<｜tool▁sep｜>{function_name}\n```json\n{arguments}\n```<｜tool▁call▁end｜><｜tool▁calls▁end｜>
        // There are some differences between DeepSeek V3 and DeepSeek V3.1
        Self {
            parser_config: ParserConfig::Json(JsonParserConfig {
                tool_call_start_tokens: vec!["<｜tool▁calls▁begin｜>".to_string()],
                tool_call_end_tokens: vec!["<｜tool▁calls▁end｜>".to_string()],
                tool_call_separator_tokens: vec!["<｜tool▁sep｜>".to_string()],
                parser_type: JsonParserType::DeepseekV3,
                ..Default::default()
            }),
        }
    }

    pub fn qwen3_coder() -> Self {
        // <tool_call><function=name><parameter=key>value</parameter></function></tool_call>
        Self {
            parser_config: ParserConfig::Xml(XmlParserConfig::default()),
        }
    }

    pub fn jamba() -> Self {
        Self {
            parser_config: ParserConfig::Json(JsonParserConfig {
                tool_call_start_tokens: vec!["<tool_calls>".to_string()],
                tool_call_end_tokens: vec!["</tool_calls>".to_string()],
                ..Default::default()
            }),
        }
    }

    pub fn deepseek_v3_2() -> Self {
        // DeepSeek V3.2 format (DSML):
        // <｜DSML｜function_calls>
        // <｜DSML｜invoke name="function_name">
        // <｜DSML｜parameter name="param_name" string="true|false">value</｜DSML｜parameter>
        // </｜DSML｜invoke>
        // </｜DSML｜function_calls>
        Self {
            parser_config: ParserConfig::Dsml(DsmlParserConfig::default()),
        }
    }

    pub fn minimax_m2() -> Self {
        // MiniMax-M2.1 format:
        // <minimax:tool_call>
        // <invoke name="function_name">
        // <parameter name="param_name">value</parameter>
        // </invoke>
        // </minimax:tool_call>
        // Reference: https://huggingface.co/MiniMaxAI/MiniMax-M2.1/blob/main/docs/tool_calling_guide.md
        Self {
            parser_config: ParserConfig::Xml(XmlParserConfig {
                tool_call_start_token: "<minimax:tool_call>".to_string(),
                tool_call_end_token: "</minimax:tool_call>".to_string(),
                function_start_token: "<invoke name=".to_string(),
                function_end_token: "</invoke>".to_string(),
                parameter_start_token: "<parameter name=".to_string(),
                parameter_end_token: "</parameter>".to_string(),
            }),
        }
    }
}
