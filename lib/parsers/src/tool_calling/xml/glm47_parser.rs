// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// GLM-4.7 Tool Call Parser
// Format: <tool_call>function_name<arg_key>param1</arg_key><arg_value>value1</arg_value></tool_call>
// Reference: https://huggingface.co/zai-org/GLM-4.7/blob/main/chat_template.jinja

use std::collections::HashMap;
use regex::Regex;
use serde_json::Value;
use uuid::Uuid;

use super::super::ToolDefinition;
use super::super::config::Glm47ParserConfig;
use super::response::{CalledFunction, ToolCallResponse, ToolCallType};

/// Check if a chunk contains the start of a GLM-4.7 tool call.
/// Format: <tool_call>function_name<arg_key>...</arg_key><arg_value>...</arg_value></tool_call>
pub fn detect_tool_call_start_glm47(chunk: &str, config: &Glm47ParserConfig) -> bool {
    let start_token = &config.tool_call_start;

    // Check if we have the complete start token
    if chunk.contains(start_token.as_str()) {
        return true;
    }

    // Check for partial match at the end of the chunk (for streaming)
    for i in 1..start_token.len() {
        if chunk.ends_with(&start_token[..i]) {
            return true;
        }
    }

    false
}

/// Find the end position of a GLM-4.7 tool call.
/// Returns the position after </tool_call> or the length of the chunk if not found.
pub fn find_tool_call_end_position_glm47(chunk: &str, config: &Glm47ParserConfig) -> usize {
    let end_token = &config.tool_call_end;

    if let Some(pos) = chunk.find(end_token.as_str()) {
        pos + end_token.len()
    } else {
        chunk.len()
    }
}

/// Try to parse GLM-4.7 formatted tool calls from a message.
/// Format: <tool_call>function_name<arg_key>param1</arg_key><arg_value>value1</arg_value></tool_call>
/// Returns (parsed_tool_calls, normal_text_content)
pub fn try_tool_call_parse_glm47(
    message: &str,
    config: &Glm47ParserConfig,
    tools: Option<&[ToolDefinition]>,
) -> anyhow::Result<(Vec<ToolCallResponse>, Option<String>)> {
    let (normal_text, tool_calls) = extract_tool_calls(message, config, tools)?;

    let normal_content = if normal_text.is_empty() {
        Some("".to_string())
    } else {
        Some(normal_text)
    };

    Ok((tool_calls, normal_content))
}

/// Extract tool calls and normal text from message.
fn extract_tool_calls(
    text: &str,
    config: &Glm47ParserConfig,
    tools: Option<&[ToolDefinition]>,
) -> anyhow::Result<(String, Vec<ToolCallResponse>)> {
    let mut normal_parts = Vec::new();
    let mut calls = Vec::new();
    let mut cursor = 0;

    let start_token = &config.tool_call_start;
    let end_token = &config.tool_call_end;

    while cursor < text.len() {
        // Find next tool call start
        if let Some(start_pos) = text[cursor..].find(start_token.as_str()) {
            let abs_start = cursor + start_pos;

            // Add text before tool call to normal parts
            normal_parts.push(&text[cursor..abs_start]);

            // Find the corresponding end token
            if let Some(end_pos) = text[abs_start..].find(end_token.as_str()) {
                let abs_end = abs_start + end_pos + end_token.len();
                let block = &text[abs_start..abs_end];

                // Parse this tool call block
                if let Ok(parsed_call) = parse_tool_call_block(block, config, tools) {
                    calls.push(parsed_call);
                }

                cursor = abs_end;
            } else {
                // No end token found -> treat the rest as normal text
                normal_parts.push(&text[abs_start..]);
                break;
            }
        } else {
            // No more tool calls
            normal_parts.push(&text[cursor..]);
            break;
        }
    }

    let normal_text = normal_parts.join("").trim().to_string();
    Ok((normal_text, calls))
}

/// Parse a single GLM-4.7 tool call block
/// Format: <tool_call>function_name<arg_key>key1</arg_key><arg_value>value1</arg_value>...</tool_call>
fn parse_tool_call_block(
    block: &str,
    config: &Glm47ParserConfig,
    tools: Option<&[ToolDefinition]>,
) -> anyhow::Result<ToolCallResponse> {
    // Remove the outer <tool_call> tags
    let start_token = &config.tool_call_start;
    let end_token = &config.tool_call_end;

    let content = block
        .strip_prefix(start_token.as_str())
        .and_then(|s| s.strip_suffix(end_token.as_str()))
        .ok_or_else(|| anyhow::anyhow!("Invalid tool call block format"))?;

    // Extract function name (everything before first <arg_key> or end)
    let arg_key_start = &config.arg_key_start;
    let function_name = if let Some(pos) = content.find(arg_key_start.as_str()) {
        content[..pos].trim().to_string()
    } else {
        // No arguments, just function name
        content.trim().to_string()
    };

    if function_name.is_empty() {
        anyhow::bail!("Empty function name in tool call");
    }

    // Parse key-value pairs
    let mut arguments = HashMap::new();
    let args_section = &content[function_name.len()..];

    // Build regex patterns
    let arg_key_start_escaped = regex::escape(&config.arg_key_start);
    let arg_key_end_escaped = regex::escape(&config.arg_key_end);
    let arg_value_start_escaped = regex::escape(&config.arg_value_start);
    let arg_value_end_escaped = regex::escape(&config.arg_value_end);

    // Pattern to match: <arg_key>key</arg_key><arg_value>value</arg_value>
    // (?s) enables dotall mode so (.*?) matches across newlines — required
    // because models often emit multi-line content in arg values.
    let pattern = format!(
        r"(?s){}([^<]+){}{}(.*?){}",
        arg_key_start_escaped,
        arg_key_end_escaped,
        arg_value_start_escaped,
        arg_value_end_escaped
    );

    let regex = Regex::new(&pattern)?;

    for cap in regex.captures_iter(args_section) {
        let key = cap.get(1).map(|m| m.as_str().trim()).unwrap_or("");
        let value = cap.get(2).map(|m| m.as_str()).unwrap_or("");

        if !key.is_empty() {
            // Try to parse value as JSON, otherwise use as string
            let json_value: Value = if value.trim().starts_with('{') || value.trim().starts_with('[') || value.trim().starts_with('"') {
                serde_json::from_str(value).unwrap_or_else(|_| Value::String(value.to_string()))
            } else {
                Value::String(value.to_string())
            };

            arguments.insert(key.to_string(), json_value);
        }
    }

    // Validate function against tools if provided
    if let Some(tools_list) = tools {
        let tool_exists = tools_list.iter().any(|t| t.name == function_name);
        if !tool_exists {
            anyhow::bail!("Function '{}' not found in available tools", function_name);
        }
    }

    Ok(ToolCallResponse {
        id: Uuid::new_v4().to_string(),
        tp: ToolCallType::Function,
        function: CalledFunction {
            name: function_name,
            arguments: serde_json::to_string(&arguments)?,
        },
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn get_test_config() -> Glm47ParserConfig {
        Glm47ParserConfig::default()
    }

    #[test]
    fn test_detect_tool_call_start() {
        let config = get_test_config();

        // Complete start token
        assert!(detect_tool_call_start_glm47("<tool_call>get_weather", &config));

        // Partial start token (streaming)
        assert!(detect_tool_call_start_glm47("Some text <tool", &config));
        assert!(detect_tool_call_start_glm47("Some text <tool_c", &config));

        // No tool call
        assert!(!detect_tool_call_start_glm47("Just normal text", &config));
    }

    #[test]
    fn test_parse_simple_tool_call() {
        let config = get_test_config();
        let message = "<tool_call>get_weather<arg_key>location</arg_key><arg_value>San Francisco</arg_value></tool_call>";

        let (calls, normal_text) = try_tool_call_parse_glm47(message, &config, None).unwrap();

        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].function.name, "get_weather");

        let args: HashMap<String, Value> = serde_json::from_str(&calls[0].function.arguments).unwrap();
        assert_eq!(args.get("location").unwrap().as_str().unwrap(), "San Francisco");
        assert_eq!(normal_text, Some("".to_string()));
    }

    #[test]
    fn test_parse_tool_call_with_multiple_args() {
        let config = get_test_config();
        let message = "<tool_call>book_flight<arg_key>from</arg_key><arg_value>NYC</arg_value><arg_key>to</arg_key><arg_value>LAX</arg_value><arg_key>date</arg_key><arg_value>2026-03-15</arg_value></tool_call>";

        let (calls, _) = try_tool_call_parse_glm47(message, &config, None).unwrap();

        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].function.name, "book_flight");

        let args: HashMap<String, Value> = serde_json::from_str(&calls[0].function.arguments).unwrap();
        assert_eq!(args.get("from").unwrap().as_str().unwrap(), "NYC");
        assert_eq!(args.get("to").unwrap().as_str().unwrap(), "LAX");
        assert_eq!(args.get("date").unwrap().as_str().unwrap(), "2026-03-15");
    }

    #[test]
    fn test_parse_tool_call_with_json_value() {
        let config = get_test_config();
        let message = r#"<tool_call>search<arg_key>filters</arg_key><arg_value>{"category": "books", "price_max": 50}</arg_value></tool_call>"#;

        let (calls, _) = try_tool_call_parse_glm47(message, &config, None).unwrap();

        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].function.name, "search");

        let args: HashMap<String, Value> = serde_json::from_str(&calls[0].function.arguments).unwrap();
        let filters = args.get("filters").unwrap();
        assert!(filters.is_object());
    }

    #[test]
    fn test_parse_multiple_tool_calls() {
        let config = get_test_config();
        let message = "<tool_call>get_weather<arg_key>location</arg_key><arg_value>NYC</arg_value></tool_call><tool_call>get_time<arg_key>timezone</arg_key><arg_value>EST</arg_value></tool_call>";

        let (calls, _) = try_tool_call_parse_glm47(message, &config, None).unwrap();

        assert_eq!(calls.len(), 2);
        assert_eq!(calls[0].function.name, "get_weather");
        assert_eq!(calls[1].function.name, "get_time");
    }

    #[test]
    fn test_parse_with_normal_text() {
        let config = get_test_config();
        let message = "I'll check the weather for you. <tool_call>get_weather<arg_key>location</arg_key><arg_value>Paris</arg_value></tool_call>";

        let (calls, normal_text) = try_tool_call_parse_glm47(message, &config, None).unwrap();

        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].function.name, "get_weather");
        assert_eq!(normal_text, Some("I'll check the weather for you.".to_string()));
    }

    #[test]
    fn test_parse_tool_call_no_args() {
        let config = get_test_config();
        let message = "<tool_call>get_current_time</tool_call>";

        let (calls, _) = try_tool_call_parse_glm47(message, &config, None).unwrap();

        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].function.name, "get_current_time");

        let args: HashMap<String, Value> = serde_json::from_str(&calls[0].function.arguments).unwrap();
        assert!(args.is_empty());
    }

    #[test]
    fn test_find_tool_call_end_position() {
        let config = get_test_config();
        let chunk = "<tool_call>func<arg_key>k</arg_key><arg_value>v</arg_value></tool_call>more text";

        let end_pos = find_tool_call_end_position_glm47(chunk, &config);
        assert_eq!(&chunk[..end_pos], "<tool_call>func<arg_key>k</arg_key><arg_value>v</arg_value></tool_call>");
    }

    #[test]
    fn test_parse_multiline_arg_value() {
        let config = get_test_config();
        let message = "<tool_call>write_file<arg_key>path</arg_key><arg_value>/tmp/hello.py</arg_value><arg_key>content</arg_key><arg_value>#!/usr/bin/env python3\nprint(\"Hello, World!\")\n</arg_value></tool_call>";

        let (calls, _) = try_tool_call_parse_glm47(message, &config, None).unwrap();

        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].function.name, "write_file");

        let args: HashMap<String, Value> = serde_json::from_str(&calls[0].function.arguments).unwrap();
        assert_eq!(args.get("path").unwrap().as_str().unwrap(), "/tmp/hello.py");
        assert!(
            args.get("content").is_some(),
            "content argument must be parsed even when it contains newlines"
        );
        let content = args.get("content").unwrap().as_str().unwrap();
        assert!(content.contains("print(\"Hello, World!\")"));
    }

    #[test]
    fn test_malformed_tool_call() {
        let config = get_test_config();

        // Missing end tag
        let message = "<tool_call>get_weather";
        let result = try_tool_call_parse_glm47(message, &config, None);
        assert!(result.is_ok()); // Should handle gracefully, no calls extracted

        let (calls, _) = result.unwrap();
        assert_eq!(calls.len(), 0);
    }
}
