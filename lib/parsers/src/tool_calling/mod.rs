// SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

pub mod json_parser;
pub mod parsers;
pub mod response;
pub mod tools;

// Re-export main types and functions for convenience
pub use json_parser::{
    CalledFunctionArguments, CalledFunctionParameters, try_tool_call_parse_json,
};
pub use parsers::{
    JsonParserConfig, ToolCallConfig, ToolCallParserType, detect_and_parse_tool_call,
};
pub use response::{CalledFunction, ToolCallResponse, ToolCallType};
pub use tools::{try_tool_call_parse_aggregate, try_tool_call_parse_stream};
