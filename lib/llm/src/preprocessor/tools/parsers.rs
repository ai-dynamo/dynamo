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

use std::collections::HashMap;
use serde_json::Value;
use uuid::Uuid;

use super::response::{ToolCallResponse, ToolCallType, CalledFunction};

/// Represents the format type for tool calls
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub enum ToolCallFormat {
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
    pub format: ToolCallFormat,
    /// Start token for list of parallel tool calls (e.g., "<TOOLCALLS>")
    pub parallel_tool_start_token: Option<String>,
    /// End token for list of parallel tool calls (e.g., "</TOOLCALLS>")
    pub parallel_tool_end_token: Option<String>,
    /// Start token for individual tool calls (e.g., "<TOOLCALL>")
    pub call_start_token: Option<String>,
    /// End token for individual tool calls (e.g., "</TOOLCALL>")
    pub call_end_token: Option<String>,
    /// Whether to expect JSON structure with "parameters" field instead of "arguments"
    pub use_parameters_field: Option<bool>,
}

impl Default for ToolCallConfig {
    fn default() -> Self {
        Self {
            format: ToolCallFormat::Json,
            parallel_tool_start_token: None,
            parallel_tool_end_token: None,
            call_start_token: None,
            call_end_token: None,
            use_parameters_field: None,
        }
    }
}

/// Trait for parsing tool calls from different formats
pub trait ToolCallParser {
    fn parse(&self, message: &str) -> anyhow::Result<Option<ToolCallResponse>>;
    fn parse_multiple(&self, message: &str) -> anyhow::Result<Vec<ToolCallResponse>>;
}

/// JSON-based tool call parser
pub struct JsonToolCallParser {
    config: ToolCallConfig,
}

impl JsonToolCallParser {
    pub fn new(config: ToolCallConfig) -> Self {
        Self { config }
    }

    fn extract_json_content(&self, message: &str) -> &str {
        let trimmed = message.trim();
        
        // Handle list wrapper tokens
        if let (Some(start), Some(end)) = (&self.config.parallel_tool_start_token, &self.config.parallel_tool_end_token) {
            if trimmed.starts_with(start) && trimmed.ends_with(end) {
                return &trimmed[start.len()..trimmed.len() - end.len()];
            }
        }
        
        // Handle individual call wrapper tokens
        if let (Some(start), Some(end)) = (&self.config.call_start_token, &self.config.call_end_token) {
            if trimmed.starts_with(start) && trimmed.ends_with(end) {
                return &trimmed[start.len()..trimmed.len() - end.len()];
            }
        }
        
        // Handle special prefixes like <|python_tag|>
        if let Some(stripped) = trimmed.strip_prefix("<|python_tag|>") {
            return stripped;
        }
        
        trimmed
    }

    fn parse_single_call(&self, json_content: &str) -> anyhow::Result<Option<ToolCallResponse>> {
        // Try parsing as single function call
        if let Ok(function_call) = self.parse_function_call(json_content)? {
            return Ok(Some(function_call));
        }
        
        // Try parsing as list and take the last one
        if let Ok(mut calls) = self.parse_function_calls_list(json_content)? {
            if let Some(last_call) = calls.pop() {
                return Ok(Some(last_call));
            }
        }
        
        Ok(None)
    }

    fn parse_function_call(&self, json_content: &str) -> anyhow::Result<ToolCallResponse> {
        let use_parameters = self.config.use_parameters_field.unwrap_or(false);
        
        if use_parameters {
            // Try with "parameters" field
            if let Ok(deser) = serde_json::from_str::<HashMap<String, Value>>(json_content) {
                if let (Some(name), Some(params)) = (deser.get("name"), deser.get("parameters")) {
                    if let (Some(name_str), Some(params_obj)) = (name.as_str(), params.as_object()) {
                        return Ok(ToolCallResponse {
                            id: format!("call-{}", Uuid::new_v4()),
                            tp: ToolCallType::Function,
                            function: CalledFunction {
                                name: name_str.to_string(),
                                arguments: serde_json::to_string(params_obj)?,
                            },
                        });
                    }
                }
            }
        } else {
            // Try with "arguments" field
            if let Ok(deser) = serde_json::from_str::<HashMap<String, Value>>(json_content) {
                if let (Some(name), Some(args)) = (deser.get("name"), deser.get("arguments")) {
                    if let (Some(name_str), Some(args_obj)) = (name.as_str(), args.as_object()) {
                        return Ok(ToolCallResponse {
                            id: format!("call-{}", Uuid::new_v4()),
                            tp: ToolCallType::Function,
                            function: CalledFunction {
                                name: name_str.to_string(),
                                arguments: serde_json::to_string(args_obj)?,
                            },
                        });
                    }
                }
            }
        }
        
        anyhow::bail!("Failed to parse function call from JSON")
    }

    fn parse_function_calls_list(&self, json_content: &str) -> anyhow::Result<Vec<ToolCallResponse>> {
        let use_parameters = self.config.use_parameters_field.unwrap_or(false);
        
        if use_parameters {
            // Try parsing as list with "parameters" field
            if let Ok(list) = serde_json::from_str::<Vec<HashMap<String, Value>>>(json_content) {
                return list.into_iter()
                    .filter_map(|item| {
                        if let (Some(name), Some(params)) = (item.get("name"), item.get("parameters")) {
                            if let (Some(name_str), Some(params_obj)) = (name.as_str(), params.as_object()) {
                                return Some(Ok(ToolCallResponse {
                                    id: format!("call-{}", Uuid::new_v4()),
                                    tp: ToolCallType::Function,
                                    function: CalledFunction {
                                        name: name_str.to_string(),
                                        arguments: serde_json::to_string(params_obj).ok()?,
                                    },
                                }));
                            }
                        }
                        None
                    })
                    .collect();
            }
        } else {
            // Try parsing as list with "arguments" field
            if let Ok(list) = serde_json::from_str::<Vec<HashMap<String, Value>>>(json_content) {
                return list.into_iter()
                    .filter_map(|item| {
                        if let (Some(name), Some(args)) = (item.get("name"), item.get("arguments")) {
                            if let (Some(name_str), Some(args_obj)) = (name.as_str(), args.as_object()) {
                                return Some(Ok(ToolCallResponse {
                                    id: format!("call-{}", Uuid::new_v4()),
                                    tp: ToolCallType::Function,
                                    function: CalledFunction {
                                        name: name_str.to_string(),
                                        arguments: serde_json::to_string(args_obj).ok()?,
                                    },
                                }));
                            }
                        }
                        None
                    })
                    .collect();
            }
        }
        
        anyhow::bail!("Failed to parse function calls list from JSON")
    }
}

impl ToolCallParser for JsonToolCallParser {
    fn parse(&self, message: &str) -> anyhow::Result<Option<ToolCallResponse>> {
        let json_content = self.extract_json_content(message);
        self.parse_single_call(json_content)
    }

    fn parse_multiple(&self, message: &str) -> anyhow::Result<Vec<ToolCallResponse>> {
        let json_content = self.extract_json_content(message);
        
        // Try parsing as list first
        if let Ok(calls) = self.parse_function_calls_list(json_content) {
            return Ok(calls);
        }
        
        // Try parsing as single call
        if let Ok(Some(call)) = self.parse_single_call(json_content) {
            return Ok(vec![call]);
        }
        
        Ok(Vec::new())
    }
}

/// Pythonic tool call parser
pub struct PythonicToolCallParser {
    config: ToolCallConfig,
}

impl PythonicToolCallParser {
    pub fn new(config: ToolCallConfig) -> Self {
        Self { config }
    }

    fn parse_pythonic_call(&self, content: &str) -> anyhow::Result<Option<ToolCallResponse>> {
        // Simple regex-like parsing for function_name(arg1=val1, arg2=val2)
        // This is a basic implementation - could be enhanced with proper parsing
        let trimmed = content.trim();
        
        if let Some(open_paren) = trimmed.find('(') {
            let function_name = trimmed[..open_paren].trim();
            if let Some(close_paren) = trimmed.rfind(')') {
                let args_str = &trimmed[open_paren + 1..close_paren];
                
                // Parse arguments as key=value pairs
                let mut args = HashMap::new();
                for arg in args_str.split(',') {
                    let arg = arg.trim();
                    if let Some(eq_pos) = arg.find('=') {
                        let key = arg[..eq_pos].trim();
                        let value = arg[eq_pos + 1..].trim();
                        
                        // Try to parse value as JSON
                        if let Ok(json_value) = serde_json::from_str(value) {
                            args.insert(key.to_string(), json_value);
                        } else {
                            // Treat as string if not valid JSON
                            args.insert(key.to_string(), Value::String(value.to_string()));
                        }
                    }
                }
                
                return Ok(Some(ToolCallResponse {
                    id: format!("call-{}", Uuid::new_v4()),
                    tp: ToolCallType::Function,
                    function: CalledFunction {
                        name: function_name.to_string(),
                        arguments: serde_json::to_string(&args)?,
                    },
                }));
            }
        }
        
        Ok(None)
    }
}

impl ToolCallParser for PythonicToolCallParser {
    fn parse(&self, message: &str) -> anyhow::Result<Option<ToolCallResponse>> {
        let content = self.extract_content(message);
        self.parse_pythonic_call(content)
    }

    fn parse_multiple(&self, message: &str) -> anyhow::Result<Vec<ToolCallResponse>> {
        let content = self.extract_content(message);
        
        // For pythonic format, we'll treat multiple calls as separate lines or semicolon-separated
        let calls: Vec<&str> = content
            .split(';')
            .filter(|s| !s.trim().is_empty())
            .collect();
        
        let mut results = Vec::new();
        for call in calls {
            if let Ok(Some(result)) = self.parse_pythonic_call(call.trim()) {
                results.push(result);
            }
        }
        
        Ok(results)
    }

    fn extract_content(&self, message: &str) -> &str {
        let trimmed = message.trim();
        
        // Handle wrapper tokens similar to JSON parser
        if let (Some(start), Some(end)) = (&self.config.call_start_token, &self.config.call_end_token) {
            if trimmed.starts_with(start) && trimmed.ends_with(end) {
                return &trimmed[start.len()..trimmed.len() - end.len()];
            }
        }
        
        trimmed
    }
}

/// XML tool call parser
pub struct XmlToolCallParser {
    config: ToolCallConfig,
}

impl XmlToolCallParser {
    pub fn new(config: ToolCallConfig) -> Self {
        Self { config }
    }

    fn parse_xml_call(&self, content: &str) -> anyhow::Result<Option<ToolCallResponse>> {
        // Basic XML parsing - could be enhanced with proper XML parser
        let content = content.trim();
        
        // Look for <function name="..."> pattern
        if let Some(name_start) = content.find("name=\"") {
            let name_start = name_start + 6; // "name=\"" length
            if let Some(name_end) = content[name_start..].find('"') {
                let function_name = &content[name_start..name_start + name_end];
                
                // Look for arguments section
                if let Some(args_start) = content.find("<arguments>") {
                    let args_start = args_start + 11; // "<arguments>" length
                    if let Some(args_end) = content[args_start..].find("</arguments>") {
                        let args_content = &content[args_start..args_start + args_end];
                        
                        // Try to parse arguments as JSON
                        if let Ok(args_value) = serde_json::from_str::<Value>(args_content) {
                            return Ok(Some(ToolCallResponse {
                                id: format!("call-{}", Uuid::new_v4()),
                                tp: ToolCallType::Function,
                                function: CalledFunction {
                                    name: function_name.to_string(),
                                    arguments: serde_json::to_string(&args_value)?,
                                },
                            }));
                        }
                    }
                }
            }
        }
        
        Ok(None)
    }
}

impl ToolCallParser for XmlToolCallParser {
    fn parse(&self, message: &str) -> anyhow::Result<Option<ToolCallResponse>> {
        let content = self.extract_content(message);
        self.parse_xml_call(content)
    }

    fn parse_multiple(&self, message: &str) -> anyhow::Result<Vec<ToolCallResponse>> {
        let content = self.extract_content(message);
        
        // For XML, we'll look for multiple <function> tags
        let mut results = Vec::new();
        let mut pos = 0;
        
        while let Some(function_start) = content[pos..].find("<function") {
            let start_pos = pos + function_start;
            if let Some(function_end) = content[start_pos..].find("</function>") {
                let function_content = &content[start_pos..start_pos + function_end + 10];
                if let Ok(Some(result)) = self.parse_xml_call(function_content) {
                    results.push(result);
                }
                pos = start_pos + function_end + 10;
            } else {
                break;
            }
        }
        
        Ok(results)
    }

    fn extract_content(&self, message: &str) -> &str {
        let trimmed = message.trim();
        
        // Handle wrapper tokens
        if let (Some(start), Some(end)) = (&self.config.call_start_token, &self.config.call_end_token) {
            if trimmed.starts_with(start) && trimmed.ends_with(end) {
                return &trimmed[start.len()..trimmed.len() - end.len()];
            }
        }
        
        trimmed
    }
}

/// Factory function to create appropriate parser based on config
pub fn create_parser(config: ToolCallConfig) -> Box<dyn ToolCallParser> {
    match config.format {
        ToolCallFormat::Json => Box::new(JsonToolCallParser::new(config)),
        ToolCallFormat::Pythonic => Box::new(PythonicToolCallParser::new(config)),
        ToolCallFormat::Xml => Box::new(XmlToolCallParser::new(config)),
    }
}

/// Load parser configuration from JSON file
pub fn load_config_from_file(path: &str) -> anyhow::Result<ToolCallConfig> {
    let content = std::fs::read_to_string(path)?;
    let config: ToolCallConfig = serde_json::from_str(&content)?;
    Ok(config)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_json_parser_basic() {
        let config = ToolCallConfig::default();
        let parser = JsonToolCallParser::new(config);
        
        let input = r#"{"name": "test_function", "arguments": {"param1": "value1"}}"#;
        let result = parser.parse(input).unwrap().unwrap();
        
        assert_eq!(result.function.name, "test_function");
        let args: Value = serde_json::from_str(&result.function.arguments).unwrap();
        assert_eq!(args["param1"], "value1");
    }

    #[test]
    fn test_json_parser_with_wrapper() {
        let mut config = ToolCallConfig::default();
        config.call_start_token = Some("<TOOLCALL>".to_string());
        config.call_end_token = Some("</TOOLCALL>".to_string());
        
        let parser = JsonToolCallParser::new(config);
        
        let input = r#"<TOOLCALL>{"name": "wrapped_function", "arguments": {"x": 42}}</TOOLCALL>"#;
        let result = parser.parse(input).unwrap().unwrap();
        
        assert_eq!(result.function.name, "wrapped_function");
        let args: Value = serde_json::from_str(&result.function.arguments).unwrap();
        assert_eq!(args["x"], 42);
    }

    #[test]
    fn test_pythonic_parser() {
        let config = ToolCallConfig::default();
        let parser = PythonicToolCallParser::new(config);
        
        let input = r#"test_function(param1="value1", param2=42)"#;
        let result = parser.parse(input).unwrap().unwrap();
        
        assert_eq!(result.function.name, "test_function");
        let args: Value = serde_json::from_str(&result.function.arguments).unwrap();
        assert_eq!(args["param1"], "value1");
        assert_eq!(args["param2"], 42);
    }

    #[test]
    fn test_xml_parser() {
        let config = ToolCallConfig::default();
        let parser = XmlToolCallParser::new(config);
        
        let input = r#"<function name="xml_function"><arguments>{"param1": "value1"}</arguments></function>"#;
        let result = parser.parse(input).unwrap().unwrap();
        
        assert_eq!(result.function.name, "xml_function");
        let args: Value = serde_json::from_str(&result.function.arguments).unwrap();
        assert_eq!(args["param1"], "value1");
    }
} 