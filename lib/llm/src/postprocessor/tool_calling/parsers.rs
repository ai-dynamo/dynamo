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

use super::response::ToolCallResponse;
use super::json_parser::try_tool_call_parse_json;
use super::pythonic_parser::try_tool_call_parse_pythonic;
use super::xml_parser::try_tool_call_parse_xml;
use regex::Regex;

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
            tool_call_start_token: Some("<TOOLCALL>".to_string()),
            tool_call_end_token: Some("</TOOLCALL>".to_string()),
        }
    }
}

fn split_messages_with_tool_call_tokens(original_message: &str, config: &ToolCallConfig) -> Vec<String> {
    // Check if parallel_tool_calls_start_token is not None
    // If not None then remove it and everything before it
    // For parallel_tool_calls_start_end, do the same but remove everything after it
    let mut message = original_message.clone();
    if config.parallel_tool_calls_start_token.is_some() {
        let start_token = config.parallel_tool_calls_start_token.as_ref().unwrap();
        let index = message.find(start_token);
        if index.is_some() {
            message = &message[index.unwrap() + start_token.len()..];
        }
    }
    if config.parallel_tool_calls_end_token.is_some() {
        let end_token = config.parallel_tool_calls_end_token.as_ref().unwrap();
        let index = message.rfind(end_token);
        if index.is_some() {
            message = &message[..index.unwrap()];
        }
    }

    // Now find all submessages between tool_call_start_token and tool_call_end_token
    // Use a compiled regex to find all submessages, the regex is based on the config
    let mut regex_str = String::new();
    if config.tool_call_start_token.is_some() {
        let start_token = config.tool_call_start_token.as_ref().unwrap();
        regex_str.push_str(&format!(r#"{}\s*"#, start_token));
    }
    if config.tool_call_end_token.is_some() {
        let end_token = config.tool_call_end_token.as_ref().unwrap();
        regex_str.push_str(&format!(r#"{}\s*"#, end_token));
    }

    let regex = Regex::new(&regex_str).unwrap();
    let matches = regex.find_iter(&message);
    
    let mut messages = Vec::new();
    let mut current_message = String::new();
    for m in matches {
        current_message.push_str(&message[..m.start()]);
        messages.push(current_message);
        current_message = String::new();
    }
    messages
}

pub fn try_tool_call_parse(message: &str, config: &ToolCallConfig) -> anyhow::Result<Option<ToolCallResponse>> {
    // Use match statement (Rust's switch statement) to call the appropriate parser
    match config.format {
        ToolCallParserType::Json => {
            try_tool_call_parse_json(&message)
        }
        ToolCallParserType::Pythonic => {
            try_tool_call_parse_pythonic(&message)
        }
        ToolCallParserType::Xml => {
            try_tool_call_parse_xml(&message)
        }
    }
}