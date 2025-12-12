// SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

pub mod base_json_parser;
pub mod deepseek_v3_1_parser;
pub mod deepseek_v3_parser;

pub use super::{config, response};
pub use base_json_parser::{detect_tool_call_start_basic_json, try_tool_call_parse_basic_json};
pub use deepseek_v3_1_parser::{
    detect_tool_call_start_deepseek_v3_1, parse_tool_calls_deepseek_v3_1,
};
pub use deepseek_v3_parser::{detect_tool_call_start_deepseek_v3, parse_tool_calls_deepseek_v3};

pub use super::config::JsonParserConfig;
pub use super::response::ToolCallResponse;

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub enum JsonParserType {
    // Basic is generic json parser which can handle most of the cases
    Basic,
    // Model Specific JSON Parsers
    DeepseekV3,
    DeepseekV31,
}

impl Default for JsonParserType {
    fn default() -> Self {
        Self::Basic
    }
}

pub fn try_tool_call_parse_json(
    message: &str,
    config: &JsonParserConfig,
    tools: Option<&[super::ToolDefinition]>,
) -> anyhow::Result<(Vec<ToolCallResponse>, Option<String>)> {
    match config.parser_type {
        JsonParserType::Basic => try_tool_call_parse_basic_json(message, config, tools),
        JsonParserType::DeepseekV3 => parse_tool_calls_deepseek_v3(message, config, tools),
        JsonParserType::DeepseekV31 => parse_tool_calls_deepseek_v3_1(message, config, tools),
    }
}

pub fn detect_tool_call_start_json(chunk: &str, config: &JsonParserConfig) -> bool {
    match config.parser_type {
        JsonParserType::Basic => detect_tool_call_start_basic_json(chunk, config),
        JsonParserType::DeepseekV3 => detect_tool_call_start_deepseek_v3(chunk, config),
        JsonParserType::DeepseekV31 => detect_tool_call_start_deepseek_v3_1(chunk, config),
    }
}

pub fn find_tool_call_end_position_json(
    chunk: &str,
    parser: &str,
    config: &JsonParserConfig,
) -> usize {
    match parser {
        "hermes" | "nemotron_deci" => {
            if let Some(end_token) = config.tool_call_end_tokens.first() {
                if let Some(pos) = chunk.find(end_token) {
                    pos + end_token.len()
                } else {
                    chunk.len()
                }
            } else {
                chunk.len()
            }
        }
        "mistral" | "phi4" => {
            if let Some(pos) = chunk.rfind(']') {
                pos + 1
            } else {
                chunk.len()
            }
        }
        _ => chunk.len(),
    }
}
