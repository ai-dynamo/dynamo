// SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// Reference implementation:
// https://huggingface.co/deepseek-ai/DeepSeek-V3.2/tree/main/encoding/encoding_dsv32.py

use regex::Regex;
use uuid::Uuid;

use super::super::config::DsmlParserConfig;
use super::super::response::{CalledFunction, ToolCallResponse, ToolCallType};

/// DeepSeek V3.2 uses DSML (DeepSeek Markup Language) format for tool calls:
///
/// <｜DSML｜function_calls>
/// <｜DSML｜invoke name="function_name">
/// <｜DSML｜parameter name="param_name" string="true|false">value</｜DSML｜parameter>
/// ...
/// </｜DSML｜invoke>
/// </｜DSML｜function_calls>
/// Check if a chunk contains the start of a DSML tool call
pub fn detect_tool_call_start_dsml(chunk: &str, config: &DsmlParserConfig) -> bool {
    let start_token = &config.function_calls_start;

    // Check for complete start token
    if chunk.contains(start_token.as_str()) {
        return true;
    }

    // Check for partial match at the end (streaming scenario)
    let start_chars: Vec<char> = start_token.chars().collect();
    for i in 1..start_chars.len() {
        let partial: String = start_chars[..i].iter().collect();
        if chunk.ends_with(&partial) {
            return true;
        }
    }

    false
}

/// Find the end position of a DSML tool call block
pub fn find_tool_call_end_position_dsml(chunk: &str, config: &DsmlParserConfig) -> usize {
    let end_token = &config.function_calls_end;

    if let Some(pos) = chunk.find(end_token.as_str()) {
        pos + end_token.len()
    } else {
        chunk.len()
    }
}

/// Parse DSML formatted tool calls from a message
/// Returns (parsed_tool_calls, normal_text_content)
pub fn try_tool_call_parse_dsml(
    message: &str,
    config: &DsmlParserConfig,
) -> anyhow::Result<(Vec<ToolCallResponse>, Option<String>)> {
    let trimmed = message.trim();

    // Early exit if no content
    if trimmed.is_empty() {
        return Ok((vec![], Some(String::new())));
    }

    // Check if tool call block exists
    if !trimmed.contains(&config.function_calls_start) {
        return Ok((vec![], Some(trimmed.to_string())));
    }

    // Extract normal text before tool calls
    let normal_text = if let Some(start_idx) = trimmed.find(&config.function_calls_start) {
        let text = trimmed[..start_idx].trim();
        if text.is_empty() {
            String::new()
        } else {
            text.to_string()
        }
    } else {
        String::new()
    };

    // Extract tool calls blocks
    let tool_calls = extract_tool_calls(trimmed, config)?;

    if tool_calls.is_empty() {
        // No valid tool calls found
        return Ok((vec![], Some(trimmed.to_string())));
    }

    Ok((tool_calls, Some(normal_text)))
}

/// Extract all tool calls from the DSML formatted text
fn extract_tool_calls(
    text: &str,
    config: &DsmlParserConfig,
) -> anyhow::Result<Vec<ToolCallResponse>> {
    let mut tool_calls = Vec::new();

    // Find all function_calls blocks
    // Matches: <｜DSML｜function_calls> ... </｜DSML｜function_calls>
    // Pattern: (?s) = dot matches newlines
    //          \s*(.*?)\s* = capture content between start/end tags (non-greedy)
    let block_pattern = format!(
        r"(?s){}\s*(.*?)\s*{}",
        regex::escape(&config.function_calls_start),
        regex::escape(&config.function_calls_end)
    );
    let block_regex = Regex::new(&block_pattern)?;

    for block_match in block_regex.captures_iter(text) {
        if let Some(block_content) = block_match.get(1) {
            let block = block_content.as_str();

            // Extract individual invokes from this block
            let invokes = extract_invokes(block, config)?;
            tool_calls.extend(invokes);
        }
    }

    Ok(tool_calls)
}

/// Extract individual invoke blocks from function_calls content
fn extract_invokes(
    block: &str,
    config: &DsmlParserConfig,
) -> anyhow::Result<Vec<ToolCallResponse>> {
    let mut invokes = Vec::new();

    // Regex to match: <｜DSML｜invoke name="function_name">..content..</｜DSML｜invoke>
    // Note: invoke_start_prefix is "<｜DSML｜invoke name=" (no quotes, we add them in pattern)
    let invoke_pattern = format!(
        r#"(?s){}\"([^"]+)\"\s*>(.*?){}"#,
        regex::escape(&config.invoke_start_prefix),
        regex::escape(&config.invoke_end)
    );
    let invoke_regex = Regex::new(&invoke_pattern)?;

    for invoke_match in invoke_regex.captures_iter(block) {
        if let (Some(name_match), Some(content_match)) = (invoke_match.get(1), invoke_match.get(2))
        {
            let function_name = name_match.as_str().trim().to_string();
            let invoke_content = content_match.as_str();

            // Parse parameters from invoke content
            let parameters = parse_parameters(invoke_content, config)?;

            // Create tool call response
            let arguments_json = serde_json::to_string(&parameters)?;

            invokes.push(ToolCallResponse {
                id: format!("call-{}", Uuid::new_v4()),
                tp: ToolCallType::Function,
                function: CalledFunction {
                    name: function_name,
                    arguments: arguments_json,
                },
            });
        }
    }

    Ok(invokes)
}

/// Parse parameters from invoke content
fn parse_parameters(
    content: &str,
    config: &DsmlParserConfig,
) -> anyhow::Result<serde_json::Map<String, serde_json::Value>> {
    let mut parameters = serde_json::Map::new();

    // Build pattern with proper escaping
    // Match: <｜DSML｜parameter name="param_name" string="true|false">value</｜DSML｜parameter>
    // Note: parameter_prefix is "<｜DSML｜parameter name=" (no quotes, we add them in pattern)
    let prefix_escaped = regex::escape(&config.parameter_prefix);
    let end_escaped = regex::escape(&config.parameter_end);

    let param_pattern = format!(
        r#"(?s){}\"([^"]+)\"\s+string=\"(true|false)\"\s*>(.*?){}"#,
        prefix_escaped, end_escaped
    );

    let param_regex = Regex::new(&param_pattern)?;

    for param_match in param_regex.captures_iter(content) {
        if let (Some(name_match), Some(string_match), Some(value_match)) =
            (param_match.get(1), param_match.get(2), param_match.get(3))
        {
            let param_name = name_match.as_str().trim();
            let is_string = string_match.as_str() == "true";
            let param_value = value_match.as_str().trim();

            // Parse value based on string attribute
            let value = if is_string {
                // String type - use as-is
                serde_json::Value::String(param_value.to_string())
            } else {
                // Non-string type - parse as JSON
                serde_json::from_str(param_value).unwrap_or_else(|_| {
                    // Fallback to string if JSON parsing fails
                    serde_json::Value::String(param_value.to_string())
                })
            };

            parameters.insert(param_name.to_string(), value);
        }
    }

    Ok(parameters)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn extract_name_and_args(call: ToolCallResponse) -> (String, serde_json::Value) {
        let args: serde_json::Value = serde_json::from_str(&call.function.arguments).unwrap();
        (call.function.name, args)
    }

    fn get_test_config() -> DsmlParserConfig {
        DsmlParserConfig::default()
    }

    #[test]
    fn test_detect_tool_call_start() {
        let config = get_test_config();
        assert!(detect_tool_call_start_dsml(
            "<｜DSML｜function_calls>",
            &config
        ));
        assert!(detect_tool_call_start_dsml(
            "text <｜DSML｜function_calls>",
            &config
        ));
        assert!(detect_tool_call_start_dsml("<｜DSML｜function_c", &config)); // Partial
        assert!(!detect_tool_call_start_dsml("no tool call here", &config));
    }

    #[test]
    fn test_find_tool_call_end_position() {
        let config = get_test_config();
        let text = "<｜DSML｜function_calls><｜DSML｜invoke name=\"test\"></｜DSML｜invoke></｜DSML｜function_calls>more";
        let pos = find_tool_call_end_position_dsml(text, &config);
        assert_eq!(&text[pos..], "more");
    }

    #[test]
    fn test_parse_single_tool_call_string_param() {
        let input = r#"<｜DSML｜function_calls>
<｜DSML｜invoke name="get_weather">
<｜DSML｜parameter name="location" string="true">San Francisco</｜DSML｜parameter>
</｜DSML｜invoke>
</｜DSML｜function_calls>"#;

        let config = get_test_config();
        let result = try_tool_call_parse_dsml(input, &config);
        if let Err(e) = &result {
            eprintln!("Parse error: {:?}", e);
        }
        let (calls, normal) = result.unwrap();

        if calls.is_empty() {
            eprintln!("Input: {}", input);
            eprintln!("No calls parsed!");
        }

        assert_eq!(calls.len(), 1, "Expected 1 tool call, got {}", calls.len());
        assert_eq!(normal, Some("".to_string()));

        let (name, args) = extract_name_and_args(calls[0].clone());
        assert_eq!(name, "get_weather");
        assert_eq!(args["location"], "San Francisco");
    }

    #[test]
    fn test_parse_single_tool_call_mixed_params() {
        let input = r#"<｜DSML｜function_calls>
<｜DSML｜invoke name="search">
<｜DSML｜parameter name="query" string="true">test query</｜DSML｜parameter>
<｜DSML｜parameter name="topn" string="false">10</｜DSML｜parameter>
</｜DSML｜invoke>
</｜DSML｜function_calls>"#;

        let config = get_test_config();
        let (calls, _) = try_tool_call_parse_dsml(input, &config).unwrap();
        assert_eq!(calls.len(), 1);

        let (name, args) = extract_name_and_args(calls[0].clone());
        assert_eq!(name, "search");
        assert_eq!(args["query"], "test query");
        assert_eq!(args["topn"], 10);
    }

    #[test]
    fn test_parse_multiple_tool_calls() {
        let input = r#"<｜DSML｜function_calls>
<｜DSML｜invoke name="get_weather">
<｜DSML｜parameter name="location" string="true">Beijing</｜DSML｜parameter>
<｜DSML｜parameter name="date" string="true">2024-01-16</｜DSML｜parameter>
</｜DSML｜invoke>
<｜DSML｜invoke name="get_weather">
<｜DSML｜parameter name="location" string="true">Hangzhou</｜DSML｜parameter>
<｜DSML｜parameter name="date" string="true">2024-01-16</｜DSML｜parameter>
</｜DSML｜invoke>
</｜DSML｜function_calls>"#;

        let config = get_test_config();
        let (calls, _) = try_tool_call_parse_dsml(input, &config).unwrap();
        assert_eq!(calls.len(), 2);

        let (name1, args1) = extract_name_and_args(calls[0].clone());
        assert_eq!(name1, "get_weather");
        assert_eq!(args1["location"], "Beijing");

        let (name2, args2) = extract_name_and_args(calls[1].clone());
        assert_eq!(name2, "get_weather");
        assert_eq!(args2["location"], "Hangzhou");
    }

    #[test]
    fn test_parse_with_normal_text() {
        let input = r#"Here's the result: <｜DSML｜function_calls>
<｜DSML｜invoke name="test">
<｜DSML｜parameter name="value" string="true">test</｜DSML｜parameter>
</｜DSML｜invoke>
</｜DSML｜function_calls>"#;

        let config = get_test_config();
        let (calls, normal) = try_tool_call_parse_dsml(input, &config).unwrap();
        assert_eq!(calls.len(), 1);
        assert_eq!(normal, Some("Here's the result:".to_string()));
    }

    #[test]
    fn test_parse_no_tool_calls() {
        let input = "This is just normal text without any tool calls.";
        let config = get_test_config();
        let (calls, normal) = try_tool_call_parse_dsml(input, &config).unwrap();
        assert_eq!(calls.len(), 0);
        assert_eq!(normal, Some(input.to_string()));
    }

    #[test]
    fn test_parse_json_parameter_value() {
        let input = r#"<｜DSML｜function_calls>
<｜DSML｜invoke name="process">
<｜DSML｜parameter name="config" string="false">{"key": "value", "count": 42}</｜DSML｜parameter>
</｜DSML｜invoke>
</｜DSML｜function_calls>"#;

        let config = get_test_config();
        let (calls, _) = try_tool_call_parse_dsml(input, &config).unwrap();
        assert_eq!(calls.len(), 1);

        let (_, args) = extract_name_and_args(calls[0].clone());
        assert!(args["config"].is_object());
        assert_eq!(args["config"]["key"], "value");
        assert_eq!(args["config"]["count"], 42);
    }

    #[test]
    fn test_parse_array_parameter_value() {
        let input = r#"<｜DSML｜function_calls>
<｜DSML｜invoke name="process">
<｜DSML｜parameter name="items" string="false">[1, 2, 3]</｜DSML｜parameter>
</｜DSML｜invoke>
</｜DSML｜function_calls>"#;

        let config = get_test_config();
        let (calls, _) = try_tool_call_parse_dsml(input, &config).unwrap();
        assert_eq!(calls.len(), 1);

        let (_, args) = extract_name_and_args(calls[0].clone());
        assert!(args["items"].is_array());
        assert_eq!(args["items"][0], 1);
        assert_eq!(args["items"][2], 3);
    }

    #[test]
    fn test_parse_boolean_parameters() {
        let input = r#"<｜DSML｜function_calls>
<｜DSML｜invoke name="config">
<｜DSML｜parameter name="enabled" string="false">true</｜DSML｜parameter>
<｜DSML｜parameter name="disabled" string="false">false</｜DSML｜parameter>
</｜DSML｜invoke>
</｜DSML｜function_calls>"#;

        let config = get_test_config();
        let (calls, _) = try_tool_call_parse_dsml(input, &config).unwrap();
        assert_eq!(calls.len(), 1);

        let (_, args) = extract_name_and_args(calls[0].clone());
        assert_eq!(args["enabled"], true);
        assert_eq!(args["disabled"], false);
    }

    #[test]
    fn test_parse_number_parameters() {
        let input = r#"<｜DSML｜function_calls>
<｜DSML｜invoke name="calculate">
<｜DSML｜parameter name="integer" string="false">42</｜DSML｜parameter>
<｜DSML｜parameter name="float" string="false">2.7</｜DSML｜parameter>
<｜DSML｜parameter name="negative" string="false">-100</｜DSML｜parameter>
</｜DSML｜invoke>
</｜DSML｜function_calls>"#;

        let config = get_test_config();
        let (calls, _) = try_tool_call_parse_dsml(input, &config).unwrap();
        assert_eq!(calls.len(), 1);

        let (_, args) = extract_name_and_args(calls[0].clone());
        assert_eq!(args["integer"], 42);
        assert_eq!(args["float"], 2.7);
        assert_eq!(args["negative"], -100);
    }

    #[test]
    fn test_parse_mixed_types_realistic() {
        // Realistic example based on test data
        let input = r#"<｜DSML｜function_calls>
<｜DSML｜invoke name="search">
<｜DSML｜parameter name="query" string="true">search agent benchmark 2024</｜DSML｜parameter>
<｜DSML｜parameter name="topn" string="false">10</｜DSML｜parameter>
<｜DSML｜parameter name="source" string="true">web</｜DSML｜parameter>
</｜DSML｜invoke>
</｜DSML｜function_calls>"#;

        let config = get_test_config();
        let (calls, _) = try_tool_call_parse_dsml(input, &config).unwrap();
        assert_eq!(calls.len(), 1);

        let (name, args) = extract_name_and_args(calls[0].clone());
        assert_eq!(name, "search");
        assert_eq!(args["query"], "search agent benchmark 2024");
        assert_eq!(args["topn"], 10); // Should be number, not string
        assert_eq!(args["source"], "web");
    }

    #[test]
    fn test_parse_nested_object_parameter() {
        let input = r#"<｜DSML｜function_calls>
<｜DSML｜invoke name="configure">
<｜DSML｜parameter name="settings" string="false">{"timeout": 30, "retry": true, "endpoints": ["a", "b"]}</｜DSML｜parameter>
</｜DSML｜invoke>
</｜DSML｜function_calls>"#;

        let config = get_test_config();
        let (calls, _) = try_tool_call_parse_dsml(input, &config).unwrap();
        assert_eq!(calls.len(), 1);

        let (_, args) = extract_name_and_args(calls[0].clone());
        assert!(args["settings"].is_object());
        assert_eq!(args["settings"]["timeout"], 30);
        assert_eq!(args["settings"]["retry"], true);
        assert!(args["settings"]["endpoints"].is_array());
        assert_eq!(args["settings"]["endpoints"][0], "a");
    }

    #[test]
    fn test_parse_empty_string_parameter() {
        let input = r#"<｜DSML｜function_calls>
<｜DSML｜invoke name="test">
<｜DSML｜parameter name="empty" string="true"></｜DSML｜parameter>
</｜DSML｜invoke>
</｜DSML｜function_calls>"#;

        let config = get_test_config();
        let (calls, _) = try_tool_call_parse_dsml(input, &config).unwrap();
        assert_eq!(calls.len(), 1);

        let (_, args) = extract_name_and_args(calls[0].clone());
        assert_eq!(args["empty"], "");
    }

    #[test]
    fn test_parse_null_parameter() {
        let input = r#"<｜DSML｜function_calls>
<｜DSML｜invoke name="test">
<｜DSML｜parameter name="value" string="false">null</｜DSML｜parameter>
</｜DSML｜invoke>
</｜DSML｜function_calls>"#;

        let config = get_test_config();
        let (calls, _) = try_tool_call_parse_dsml(input, &config).unwrap();
        assert_eq!(calls.len(), 1);

        let (_, args) = extract_name_and_args(calls[0].clone());
        assert!(args["value"].is_null());
    }
}
