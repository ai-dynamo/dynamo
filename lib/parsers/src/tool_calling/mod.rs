// SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

pub mod config;
pub mod harmony;
pub mod json;
pub mod parsers;
pub mod pythonic;
pub mod response;
#[cfg(test)]
pub mod tests;
pub mod tools;

// Re-export main types and functions for convenience
pub use config::{JsonParserConfig, ToolCallConfig, ToolCallParserType};
pub use harmony::{parse_tool_calls_harmony, parse_tool_calls_harmony_complete};
pub use json::try_tool_call_parse_json;
pub use parsers::{
    detect_and_parse_tool_call, detect_tool_call_start, find_tool_call_end_position,
    try_tool_call_parse,
};
pub use pythonic::try_tool_call_parse_pythonic;
pub use response::{CalledFunction, ToolCallResponse, ToolCallType};
pub use tools::{try_tool_call_parse_aggregate, try_tool_call_parse_stream};
