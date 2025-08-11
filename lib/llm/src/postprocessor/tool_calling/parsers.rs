// SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

//! Parsers for tool calls
//! 
//! This module contains the parsers for tool calls.
//! 
//! The parsers are responsible for parsing the tool calls from the response.


/// Represents the format type for tool calls
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub enum ToolCallParserType {
    /// JSON format: `{"name": "function", "arguments": {...}}`
    Json,
    /// Pythonic format: `function_name(arg1=val1, arg2=val2)`
    Pythonic,
    /// XML format: `<function name="function"><arguments>...</arguments></function>`
    Xml,
}

/// Configuration for parsing tool calls with different formats
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct ToolCallConfig {
    /// The format type for tool calls
    pub format: ToolCallParserType,
    /// Start token for list of parallel tool calls (e.g., "<TOOLCALLS>")
    pub parallel_tool_calls_start_token: Option<String>,
    /// End token for list of parallel tool calls (e.g., "</TOOLCALLS>")
    pub parallel_tool_calls_end_token: Option<String>,
    /// Start token for individual tool calls (e.g., "<TOOLCALL>")
    pub tool_call_start_token: Option<String>,
    /// End token for individual tool calls (e.g., "</TOOLCALL>")
    pub tool_call_end_token: Option<String>,
}

impl Default for ToolCallConfig {
    fn default() -> Self {
        Self {
            format: ToolCallParserType::Json,
            parallel_tool_calls_start_token: None,
            parallel_tool_calls_end_token: None,
            tool_call_start_token: "<TOOLCALL",
            tool_call_end_token: "</TOOLCALL>",
        }
    }
}
