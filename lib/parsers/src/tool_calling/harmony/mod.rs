// SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

pub mod harmony_parser;

pub use super::config::JsonParserConfig;
pub use super::{config, response};
pub use harmony_parser::{
    detect_tool_call_start_harmony, parse_tool_calls_harmony, parse_tool_calls_harmony_complete,
};

pub fn find_tool_call_end_position_harmony(chunk: &str, config: &JsonParserConfig) -> usize {
    let end_token = config
        .tool_call_end_tokens
        .first()
        .map_or("<|call|>", |v| v);
    if let Some(pos) = chunk.rfind(end_token) {
        pos + end_token.len()
    } else {
        chunk.len()
    }
}
