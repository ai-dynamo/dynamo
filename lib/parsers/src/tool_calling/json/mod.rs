// SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

pub mod base_json_parser;
pub mod deepseek_parser;

pub use super::{config, response};
pub use base_json_parser::{detect_tool_call_start_basic_json, try_tool_call_parse_basic_json};
pub use deepseek_parser::{detect_tool_call_start_deepseek_v3_1, parse_tool_calls_deepseek_v3_1};

pub use super::config::JsonParserConfig;
pub use super::response::ToolCallResponse;

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub enum JsonParserType {
    // Basic is generic json parser which can handle most of the cases
    Basic,
    // Model Specific JSON Parsers
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
) -> anyhow::Result<(Vec<ToolCallResponse>, Option<String>)> {
    match config.parser_type {
        JsonParserType::Basic => try_tool_call_parse_basic_json(message, config),
        JsonParserType::DeepseekV31 => parse_tool_calls_deepseek_v3_1(message, config),
    }
}

pub fn detect_tool_call_start_json(chunk: &str, config: &JsonParserConfig) -> bool {
    match config.parser_type {
        JsonParserType::Basic => detect_tool_call_start_basic_json(chunk, config),
        JsonParserType::DeepseekV31 => detect_tool_call_start_deepseek_v3_1(chunk, config),
    }
}
