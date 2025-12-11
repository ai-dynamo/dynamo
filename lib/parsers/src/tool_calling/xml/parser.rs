// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// Reference implementation:
// https://github.com/sgl-project/sglang/blob/44da737770e4bcd9bfa27751f0a0751c9b5c06e1/python/sglang/srt/function_call/qwen3_coder_detector.py

use std::collections::HashMap;

use regex::Regex;
use uuid::Uuid;

use super::super::config::XmlParserConfig;
use super::response::{CalledFunction, ToolCallResponse, ToolCallType};

/// Check if a chunk contains the start of a xml-style tool call.
/// Format: <tool_call><function=name><parameter=foo>...</parameter></function></tool_call>
pub fn detect_tool_call_start_xml(chunk: &str, config: &XmlParserConfig) -> bool {
    // Check for complete or partial start token.
    let start_token = &config.tool_call_start_token;

    // Check if we have the complete start token.
    if chunk.contains(start_token.as_str()) {
        return true;
    }

    // Check for partial match at the end of the chunk (for streaming).
    for i in 1..start_token.len() {
        if chunk.ends_with(&start_token[..i]) {
            return true;
        }
    }

    false
}

/// Find the end position of a Qwen3Coder tool call.
/// Returns the position after </tool_call> or the length of the chunk if not found.
pub fn find_tool_call_end_position_xml(chunk: &str, config: &XmlParserConfig) -> usize {
    let end_token = &config.tool_call_end_token;

    if let Some(pos) = chunk.find(end_token.as_str()) {
        pos + end_token.len()
    } else {
        chunk.len()
    }
}

/// Try to parse Qwen3Coder formatted tool calls from a message.
/// Format: <tool_call><function=name><parameter=key>value</parameter></function></tool_call>
/// Returns (parsed_tool_calls, normal_text_content)
pub fn try_tool_call_parse_xml(
    message: &str,
    config: &XmlParserConfig,
) -> anyhow::Result<(Vec<ToolCallResponse>, Option<String>)> {
    let (normal_text, tool_calls) = extract_tool_calls(message, config)?;

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
    config: &XmlParserConfig,
) -> anyhow::Result<(String, Vec<ToolCallResponse>)> {
    let mut normal_parts = Vec::new();
    let mut calls = Vec::new();
    let mut cursor = 0;

    let start_token = &config.tool_call_start_token;
    let end_token = &config.tool_call_end_token;

    while cursor < text.len() {
        // Find next tool call start.
        if let Some(start_pos) = text[cursor..].find(start_token.as_str()) {
            let abs_start = cursor + start_pos;

            // Add text before tool call to normal parts.
            normal_parts.push(&text[cursor..abs_start]);

            // Find the corresponding end token.
            if let Some(end_pos) = text[abs_start..].find(end_token.as_str()) {
                let abs_end = abs_start + end_pos + end_token.len();
                let block = &text[abs_start..abs_end];

                // Parse this tool call block.
                if let Ok(mut parsed_calls) = parse_tool_call_block(block, config) {
                    calls.append(&mut parsed_calls);
                }

                cursor = abs_end;
            } else {
                // No end token found -> treat the rest as normal text.
                normal_parts.push(&text[abs_start..]);
                break;
            }
        } else {
            // No more tool calls.
            normal_parts.push(&text[cursor..]);
            break;
        }
    }

    let normal_text = normal_parts.join("").trim().to_string();
    Ok((normal_text, calls))
}

/// Parse a single tool call block
/// Format: <tool_call><function=name><parameter=key>value</parameter>...</function></tool_call>
fn parse_tool_call_block(
    block: &str,
    config: &XmlParserConfig,
) -> anyhow::Result<Vec<ToolCallResponse>> {
    // Build regex patterns based on config
    let function_start = regex::escape(&config.function_start_token);
    let function_end = regex::escape(&config.function_end_token);
    let parameter_start = regex::escape(&config.parameter_start_token);
    let parameter_end = regex::escape(&config.parameter_end_token);

    let function_pattern = format!(r"(?s){}([^>]+)>(.*?)(?:{}|$)", function_start, function_end);
    let parameter_pattern = format!(
        r"(?s){}([^>]+)>(.*?)(?:{}|$)",
        parameter_start, parameter_end
    );

    let function_regex = Regex::new(&function_pattern)?;
    let parameter_regex = Regex::new(&parameter_pattern)?;

    let mut results = Vec::new();

    // Find all function blocks.
    for func_cap in function_regex.captures_iter(block) {
        let function_name = func_cap.get(1).map(|m| m.as_str().trim()).unwrap_or("");
        let function_body = func_cap.get(2).map(|m| m.as_str()).unwrap_or("");

        if function_name.is_empty() {
            continue;
        }

        // Parse parameters from the function body.
        let mut parameters: HashMap<String, serde_json::Value> = HashMap::new();

        for param_cap in parameter_regex.captures_iter(function_body) {
            let param_name = param_cap.get(1).map(|m| m.as_str().trim()).unwrap_or("");
            let param_value = param_cap.get(2).map(|m| m.as_str()).unwrap_or("");

            if !param_name.is_empty() {
                let parsed_value = safe_parse_value(param_value);
                parameters.insert(param_name.to_string(), parsed_value);
            }
        }

        // Create tool call response.
        let arguments_json = serde_json::to_string(&parameters)?;

        let tool_call = ToolCallResponse {
            id: format!("call-{}", Uuid::new_v4()),
            tp: ToolCallType::Function,
            function: CalledFunction {
                name: function_name.to_string(),
                arguments: arguments_json,
            },
        };

        results.push(tool_call);
    }

    Ok(results)
}

/// Safely parse a value - tries JSON, then falls back to string.
/// Mimics SGLang's `_safe_val` function in spirit.
fn safe_parse_value(raw: &str) -> serde_json::Value {
    // HTML unescape
    let unescaped = html_unescape(raw.trim());

    if let Ok(value) = serde_json::from_str::<serde_json::Value>(&unescaped) {
        return value;
    }

    if let Ok(num) = unescaped.parse::<i64>() {
        return serde_json::Value::Number(num.into());
    }

    if let Ok(num) = unescaped.parse::<f64>()
        && let Some(num_val) = serde_json::Number::from_f64(num)
    {
        return serde_json::Value::Number(num_val);
    }

    match unescaped.to_lowercase().as_str() {
        "true" => return serde_json::Value::Bool(true),
        "false" => return serde_json::Value::Bool(false),
        "null" | "none" => return serde_json::Value::Null,
        _ => {}
    }

    // Default to string, stripping newlines from start and end.
    serde_json::Value::String(unescaped.trim_matches('\n').to_string())
}

/// Simple HTML unescape for common entities.
fn html_unescape(s: &str) -> String {
    s.replace("&lt;", "<")
        .replace("&gt;", ">")
        .replace("&amp;", "&")
        .replace("&quot;", "\"")
        .replace("&#x27;", "'")
        .replace("&#39;", "'")
}

#[cfg(test)]
mod tests {
    use super::*;
    use rstest::rstest;

    #[test]
    fn test_detect_tool_call_start() {
        let config = XmlParserConfig::default();
        assert!(detect_tool_call_start_xml("<tool_call>", &config));
        assert!(detect_tool_call_start_xml("text <tool_call>", &config));
        assert!(detect_tool_call_start_xml("<tool_c", &config)); // Partial match
        assert!(detect_tool_call_start_xml("<", &config)); // Partial match
        assert!(!detect_tool_call_start_xml("no tool call here", &config));
        assert!(!detect_tool_call_start_xml("toolcall", &config));
    }

    #[test]
    fn test_find_tool_call_end_position() {
        let config = XmlParserConfig::default();
        let text = "<tool_call><function=test></function></tool_call>more text";
        let pos = find_tool_call_end_position_xml(text, &config);
        assert_eq!(pos, 49); // Position after </tool_call>
        assert_eq!(&text[pos..], "more text");

        let text_no_end = "<tool_call><function=test>";
        let pos = find_tool_call_end_position_xml(text_no_end, &config);
        assert_eq!(pos, text_no_end.len());
    }

    #[rstest]
    #[case(r#"{"key": "value"}"#, serde_json::json!({"key": "value"}), "JSON object")]
    #[case(r#"[1, 2, 3]"#, serde_json::json!([1, 2, 3]), "JSON array")]
    #[case("42", serde_json::json!(42), "integer")]
    #[case("3.15", serde_json::json!(3.15), "float")]
    #[case("true", serde_json::json!(true), "boolean true")]
    #[case("false", serde_json::json!(false), "boolean false")]
    #[case("null", serde_json::json!(null), "null")]
    #[case("hello", serde_json::json!("hello"), "unquoted string")]
    #[case("  text  ", serde_json::json!("text"), "trimmed string")]
    fn test_safe_parse_value(
        #[case] input: &str,
        #[case] expected: serde_json::Value,
        #[case] _description: &str,
    ) {
        assert_eq!(safe_parse_value(input), expected);
    }

    #[rstest]
    #[case("&lt;div&gt;", "<div>", "HTML tags")]
    #[case("a &amp; b", "a & b", "ampersand")]
    #[case("&quot;quoted&quot;", "\"quoted\"", "quotes")]
    fn test_html_unescape(#[case] input: &str, #[case] expected: &str, #[case] _description: &str) {
        assert_eq!(html_unescape(input), expected);
    }

    #[test]
    fn test_parse_simple_tool_call() {
        let input = r#"<tool_call>
<function=execute_bash>
<parameter=command>
pwd && ls
</parameter>
</function>
</tool_call>"#;

        let (calls, normal) = try_tool_call_parse_xml(input, &XmlParserConfig::default()).unwrap();
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].function.name, "execute_bash");
        assert_eq!(normal, Some("".to_string()));

        let args: serde_json::Value = serde_json::from_str(&calls[0].function.arguments).unwrap();
        assert_eq!(args["command"], "pwd && ls");
    }

    #[test]
    fn test_parse_multiple_parameters() {
        let input = r#"<tool_call>
<function=get_weather>
<parameter=city>
San Francisco
</parameter>
<parameter=state>
CA
</parameter>
<parameter=unit>
fahrenheit
</parameter>
</function>
</tool_call>"#;

        let (calls, _) = try_tool_call_parse_xml(input, &XmlParserConfig::default()).unwrap();
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].function.name, "get_weather");

        let args: serde_json::Value = serde_json::from_str(&calls[0].function.arguments).unwrap();
        assert_eq!(args["city"], "San Francisco");
        assert_eq!(args["state"], "CA");
        assert_eq!(args["unit"], "fahrenheit");
    }

    #[test]
    fn test_parse_with_normal_text() {
        let input = r#"I'll help you with that. <tool_call>
<function=get_weather>
<parameter=city>
Dallas
</parameter>
</function>
</tool_call> Let me check that for you."#;

        let (calls, normal) = try_tool_call_parse_xml(input, &XmlParserConfig::default()).unwrap();
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].function.name, "get_weather");
        assert_eq!(
            normal,
            Some("I'll help you with that.  Let me check that for you.".to_string())
        );
    }

    #[test]
    fn test_parse_multiple_tool_calls() {
        let input = r#"<tool_call>
<function=get_weather>
<parameter=city>
Dallas
</parameter>
</function>
</tool_call>
<tool_call>
<function=get_weather>
<parameter=city>
Orlando
</parameter>
</function>
</tool_call>"#;

        let (calls, _) = try_tool_call_parse_xml(input, &XmlParserConfig::default()).unwrap();
        assert_eq!(calls.len(), 2);
        assert_eq!(calls[0].function.name, "get_weather");
        assert_eq!(calls[1].function.name, "get_weather");

        let args0: serde_json::Value = serde_json::from_str(&calls[0].function.arguments).unwrap();
        let args1: serde_json::Value = serde_json::from_str(&calls[1].function.arguments).unwrap();
        assert_eq!(args0["city"], "Dallas");
        assert_eq!(args1["city"], "Orlando");
    }

    #[test]
    fn test_parse_json_parameter_value() {
        let input = r#"<tool_call>
<function=process_data>
<parameter=config>
{"setting": "value", "count": 42}
</parameter>
</function>
</tool_call>"#;

        let (calls, _) = try_tool_call_parse_xml(input, &XmlParserConfig::default()).unwrap();
        assert_eq!(calls.len(), 1);

        let args: serde_json::Value = serde_json::from_str(&calls[0].function.arguments).unwrap();
        assert!(args["config"].is_object());
        assert_eq!(args["config"]["setting"], "value");
        assert_eq!(args["config"]["count"], 42);
    }

    #[test]
    fn test_parse_no_tool_calls() {
        let input = "This is just normal text without any tool calls.";
        let (calls, normal) = try_tool_call_parse_xml(input, &XmlParserConfig::default()).unwrap();
        assert_eq!(calls.len(), 0);
        assert_eq!(normal, Some(input.to_string()));
    }

    #[test]
    fn test_parse_malformed_tool_call() {
        let input = r#"<tool_call>
<function=incomplete>
<parameter=test>
value
</tool_call>"#;

        // Should handle gracefully - might parse or return empty
        let result = try_tool_call_parse_xml(input, &XmlParserConfig::default());
        assert!(result.is_ok());
    }

    #[test]
    fn test_parse_missing_parameter_closing_tag() {
        let input = r#"<tool_call>
<function=execute_bash>
<parameter=command>
ls -la
</function>
</tool_call>"#;

        let (calls, _) = try_tool_call_parse_xml(input, &XmlParserConfig::default()).unwrap();
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].function.name, "execute_bash");

        let args: serde_json::Value = serde_json::from_str(&calls[0].function.arguments).unwrap();
        assert_eq!(args["command"], "ls -la");
    }

    #[test]
    fn test_parse_missing_function_closing_tag() {
        let input = r#"<tool_call>
<function=get_weather>
<parameter=city>
Boston
</parameter>
</tool_call>"#;

        let (calls, _) = try_tool_call_parse_xml(input, &XmlParserConfig::default()).unwrap();
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].function.name, "get_weather");

        let args: serde_json::Value = serde_json::from_str(&calls[0].function.arguments).unwrap();
        assert_eq!(args["city"], "Boston");
    }

    #[test]
    fn test_parse_missing_both_closing_tags() {
        let input = r#"<tool_call>
<function=run_query>
<parameter=sql>
SELECT * FROM users
</tool_call>"#;

        let (calls, _) = try_tool_call_parse_xml(input, &XmlParserConfig::default()).unwrap();
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].function.name, "run_query");

        let args: serde_json::Value = serde_json::from_str(&calls[0].function.arguments).unwrap();
        // This matches the original SGLang python implementation.
        assert_eq!(args["sql"], "SELECT * FROM users\n</tool_call>");
    }

    #[test]
    fn test_parse_multiple_parameters_missing_closing_tags() {
        let input = r#"<tool_call>
<function=search>
<parameter=query>
rust programming
<parameter=limit>
10
</function>
</tool_call>"#;

        let (calls, _) = try_tool_call_parse_xml(input, &XmlParserConfig::default()).unwrap();
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].function.name, "search");

        let args: serde_json::Value = serde_json::from_str(&calls[0].function.arguments).unwrap();
        // This matches the original SGLang python implementation.
        assert_eq!(args["query"], "rust programming\n<parameter=limit>\n10");
    }
}
