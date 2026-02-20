// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// Reference implementation:
// https://github.com/sgl-project/sglang/blob/44da737770e4bcd9bfa27751f0a0751c9b5c06e1/python/sglang/srt/function_call/qwen3_coder_detector.py

use std::collections::HashMap;

use regex::Regex;
use serde_json::Value;
use uuid::Uuid;

use super::super::ToolDefinition;
use super::super::config::XmlParserConfig;
use super::response::{CalledFunction, ToolCallResponse, ToolCallType};

/// Strip surrounding quotes from a string if present
fn strip_quotes(s: &str) -> &str {
    let trimmed = s.trim();
    if (trimmed.starts_with('"') && trimmed.ends_with('"'))
        || (trimmed.starts_with('\'') && trimmed.ends_with('\''))
    {
        &trimmed[1..trimmed.len() - 1]
    } else {
        trimmed
    }
}

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
    config: &XmlParserConfig,
    tools: Option<&[ToolDefinition]>,
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
                if let Ok(mut parsed_calls) = parse_tool_call_block(block, config, tools) {
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
    tools: Option<&[ToolDefinition]>,
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
        let function_name_raw = func_cap.get(1).map(|m| m.as_str().trim()).unwrap_or("");
        let function_name = strip_quotes(function_name_raw);
        let function_body = func_cap.get(2).map(|m| m.as_str()).unwrap_or("");

        if function_name.is_empty() {
            continue;
        }

        // Get parameter config for this function
        let param_config = get_arguments_config(function_name, tools);

        // Parse parameters from the function body.
        let mut parameters: HashMap<String, serde_json::Value> = HashMap::new();

        for param_cap in parameter_regex.captures_iter(function_body) {
            let param_name_raw = param_cap.get(1).map(|m| m.as_str().trim()).unwrap_or("");
            let param_name = strip_quotes(param_name_raw);
            let param_value = param_cap.get(2).map(|m| m.as_str()).unwrap_or("");

            if !param_name.is_empty() {
                let parsed_value =
                    convert_param_value(param_value, param_name, &param_config, function_name);
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

/// Extract argument configuration for a function from the tool definitions.
/// Returns a HashMap of parameter names to their schema definitions.
fn get_arguments_config(
    func_name: &str,
    tools: Option<&[ToolDefinition]>,
) -> HashMap<String, Value> {
    let Some(tools) = tools else {
        return HashMap::new();
    };

    for tool in tools {
        if tool.name == func_name {
            if let Some(params) = &tool.parameters {
                // Try to extract "properties" from the parameters schema
                if let Some(properties) = params.get("properties") {
                    if let Some(props_obj) = properties.as_object() {
                        return props_obj
                            .iter()
                            .map(|(k, v)| (k.clone(), v.clone()))
                            .collect();
                    }
                } else if let Some(params_obj) = params.as_object() {
                    // If no "properties" field, treat the whole thing as the config
                    return params_obj
                        .iter()
                        .map(|(k, v)| (k.clone(), v.clone()))
                        .collect();
                }
            }
            return HashMap::new();
        }
    }

    tracing::warn!("Tool '{}' is not defined in the tools list.", func_name);
    HashMap::new()
}

/// Convert parameter value based on its type in the schema.
/// This matches the behavior of the Python implementation.
/// Converts a string parameter value from XML into a typed JSON Value.
///
/// # Examples
///
/// **String types:**
/// ```text
/// Input:  param_value="hello world", param_type="string"
/// Output: Value::String("hello world")
/// ```
///
/// ```text
/// Input:  param_value="42", param_type="string"
/// Output: Value::String("42")
/// ```
///
/// **Integer types:**
/// ```text
/// Input:  param_value="42", param_type="integer"
/// Output: Value::Number(42)
///
/// Input:  param_value="not_a_number", param_type="int"
/// Output: Value::String("not_a_number")  // Falls back to string with warning
/// ```
///
/// **Float/Number types:**
/// ```text
/// Input:  param_value="3.14", param_type="number"
/// Output: Value::Number(3.14)
///
/// Input:  param_value="42.0", param_type="float"
/// Output: Value::Number(42)  // Whole numbers stored as integers
/// ```
///
/// **Boolean types:**
/// ```text
/// Input:  param_value="true", param_type="boolean"
/// Output: Value::Bool(true)
///
/// Input:  param_value="yes", param_type="bool"
/// Output: Value::Bool(false)  // Falls back to false with warning
/// ```
///
/// **Complex types (objects/arrays):**
/// ```text
/// Input:  param_value='{"key": "value"}', param_type="object"
/// Output: Value::Object({"key": "value"})
///
/// Input:  param_value="[1, 2, 3]", param_type="array"
/// Output: Value::Array([1, 2, 3])
///
/// Input:  param_value="{'key': 'value'}", param_type="dict"
/// Output: Value::Object({"key": "value"})  // Uses ast.literal_eval-style parsing
/// ```
///
/// **Special cases:**
/// ```text
/// Input:  param_value="null", param_type=<any>
/// Output: Value::Null  // Handled before type checking
///
/// Input:  param_value="&lt;tag&gt;", param_type="string"
/// Output: Value::String("<tag>")  // HTML entities are unescaped
///
/// Input:  param_value="123", param_type=<undefined/not in schema>
/// Output: Value::String("123")  // Unknown params returned as strings
/// ```
///
/// # Arguments
///
/// * `param_value` - The raw string value from XML parameter
/// * `param_name` - The parameter name (used for schema lookup and error messages)
/// * `param_config` - Schema defining expected types for each parameter
/// * `func_name` - The function/tool name (used for error messages)
///
/// # Type Aliases
///
/// The function recognizes various type name aliases:
/// - Strings: "string", "str", "text", "varchar", "char", "enum"
/// - Integers: "int", "integer", "int32", "int64", "uint", "long", "short", "unsigned"
/// - Numbers: "number", "num", "float", "float32", "float64", "double"
/// - Booleans: "boolean", "bool", "binary"
/// - Objects: "object", "dict", "dictionary"
/// - Arrays: "array", "arr", "list"
fn convert_param_value(
    param_value: &str,
    param_name: &str,
    param_config: &HashMap<String, Value>,
    func_name: &str,
) -> Value {
    // HTML unescape and trim
    let param_value = html_unescape(param_value.trim());

    // Handle null
    if param_value.to_lowercase() == "null" {
        return Value::Null;
    }

    // Check if parameter is in config
    if !param_config.contains_key(param_name) {
        tracing::debug!(
            "Parsed parameter '{}' is not defined in the tool parameters for tool '{}', directly returning the string value.",
            param_name,
            func_name
        );
        return Value::String(param_value);
    }

    // Get the type from schema
    let param_type = param_config
        .get(param_name)
        .and_then(|v| v.get("type"))
        .and_then(|t| t.as_str())
        .unwrap_or("string")
        .to_lowercase();

    // The follow `match` block follows this rough pattern for each block:
    // 1. Match `param_type` against predefined string representations of each type,
    // 2. Parse the string value and convert it to the appropriate Rust JSON Value type.
    // Each branch handles a category of type aliases (e.g., "int"/"integer"/"int32" all map to i64).
    // If parsing fails, we log a warning and fall back to returning the value as a string.
    match param_type.as_str() {
        // String types: Return value as-is (already HTML-unescaped above)
        "string" | "str" | "text" | "varchar" | "char" | "enum" => Value::String(param_value),

        // Integer types: Parse as i64, fall back to string on error.
        // Matches: "int", "integer", "int32", "uint", "unsigned", "long", "short", etc.
        t if t.starts_with("int")
            || t.starts_with("uint")
            || t.starts_with("long")
            || t.starts_with("short")
            || t.starts_with("unsigned") =>
        {
            match param_value.parse::<i64>() {
                Ok(int_val) => Value::Number(int_val.into()),
                Err(_) => {
                    tracing::warn!(
                        "Parsed value '{}' of parameter '{}' is not an integer in tool '{}', degenerating to string.",
                        param_value,
                        param_name,
                        func_name
                    );
                    Value::String(param_value)
                }
            }
        }

        // Float/Number types: Parse as f64.
        // Matches: "number", "num", "float", "float32", "float64", "double", etc.
        // Note: Whole numbers (e.g., 42.0) are stored as integers for better JSON compatibility.
        t if t.starts_with("num") || t.starts_with("float") => {
            match param_value.parse::<f64>() {
                Ok(float_val) => {
                    // Return int if it's a whole number, otherwise float.
                    if float_val.fract() == 0.0 && float_val.is_finite() {
                        Value::Number((float_val as i64).into())
                    } else if let Some(num) = serde_json::Number::from_f64(float_val) {
                        Value::Number(num)
                    } else {
                        tracing::warn!(
                            "Parsed value '{}' of parameter '{}' is not a valid float in tool '{}', degenerating to string.",
                            param_value,
                            param_name,
                            func_name
                        );
                        Value::String(param_value)
                    }
                }
                Err(_) => {
                    tracing::warn!(
                        "Parsed value '{}' of parameter '{}' is not a float in tool '{}', degenerating to string.",
                        param_value,
                        param_name,
                        func_name
                    );
                    Value::String(param_value)
                }
            }
        }

        // Boolean types: Only "true" or "false" (case-insensitive) are valid.
        // Any other value defaults to false with a warning.
        "boolean" | "bool" | "binary" => {
            let lower_val = param_value.to_lowercase();
            if lower_val != "true" && lower_val != "false" {
                tracing::warn!(
                    "Parsed value '{}' of parameter '{}' is not a boolean (`true` or `false`) in tool '{}', degenerating to false.",
                    param_value,
                    param_name,
                    func_name
                );
            }
            Value::Bool(lower_val == "true")
        }

        // Complex types (objects/arrays): Try JSON parsing, then fall back to Python-style
        // `ast.literal_eval` (or our own barebones version of it for the purposes of this
        // parser).
        // Matches: "object", "array", "arr", "dict", "dictionary", "list", etc.
        // This handles both JSON syntax ({"a": 1}) and Python syntax ({'a': 1}).
        t if t == "object"
            || t == "array"
            || t == "arr"
            || t.starts_with("dict")
            || t.starts_with("list") =>
        {
            // Try JSON parsing first (standard JSON with double quotes).
            if let Ok(json_val) = serde_json::from_str::<Value>(&param_value) {
                return json_val;
            }

            tracing::warn!(
                "Parsed value '{}' of parameter '{}' cannot be parsed with json.loads in tool '{}', will try other methods to parse it.",
                param_value,
                param_name,
                func_name
            );

            // Try `ast.literal_eval` equivalent (handles Python-style single quotes, etc.).
            if let Ok(json_val) = try_literal_eval(&param_value) {
                return json_val;
            }

            tracing::warn!(
                "Parsed value '{}' of parameter '{}' cannot be converted via Python `ast.literal_eval()` in tool '{}', degenerating to string.",
                param_value,
                param_name,
                func_name
            );
            Value::String(param_value)
        }

        // Unknown/custom types: Attempt best-effort parsing via `literal_eval`.
        // This allows for flexible type names while still trying to parse structured data
        _ => {
            // Unknown type, try `literal_eval`.
            if let Ok(json_val) = try_literal_eval(&param_value) {
                return json_val;
            }

            tracing::warn!(
                "Parsed value '{}' of parameter '{}' cannot be converted via Python `ast.literal_eval()` in tool '{}', degenerating to string.",
                param_value,
                param_name,
                func_name
            );
            Value::String(param_value)
        }
    }
}

/// Try to parse a value similar to Python's ast.literal_eval.
/// This is a simplified version that handles common cases.
fn try_literal_eval(s: &str) -> Result<Value, ()> {
    // First try standard JSON
    if let Ok(val) = serde_json::from_str::<Value>(s) {
        return Ok(val);
    }

    // Try to handle Python-style literals (single quotes, True/False/None)
    let normalized = s
        .replace('\'', "\"") // Replace single quotes with double quotes
        .replace("True", "true")
        .replace("False", "false")
        .replace("None", "null");

    serde_json::from_str::<Value>(&normalized).map_err(|_| ())
}

/// Safely parse a value - tries JSON, then falls back to string.
/// Mimics SGLang's `_safe_val` function in spirit.
/// NOTE: This function is deprecated and kept for reference. Use convert_param_value instead.
#[allow(dead_code)]
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

        let (calls, normal) =
            try_tool_call_parse_xml(input, &XmlParserConfig::default(), None).unwrap();
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

        let (calls, _) = try_tool_call_parse_xml(input, &XmlParserConfig::default(), None).unwrap();
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

        let (calls, normal) =
            try_tool_call_parse_xml(input, &XmlParserConfig::default(), None).unwrap();
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

        let (calls, _) = try_tool_call_parse_xml(input, &XmlParserConfig::default(), None).unwrap();
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
        // With schema-aware parsing, we need to provide a schema to parse JSON objects
        let tools = vec![ToolDefinition {
            name: "process_data".to_string(),
            parameters: Some(serde_json::json!({
                "type": "object",
                "properties": {
                    "config": {"type": "object"}
                }
            })),
        }];

        let input = r#"<tool_call>
<function=process_data>
<parameter=config>
{"setting": "value", "count": 42}
</parameter>
</function>
</tool_call>"#;

        let (calls, _) =
            try_tool_call_parse_xml(input, &XmlParserConfig::default(), Some(&tools)).unwrap();
        assert_eq!(calls.len(), 1);

        let args: serde_json::Value = serde_json::from_str(&calls[0].function.arguments).unwrap();
        assert!(args["config"].is_object());
        assert_eq!(args["config"]["setting"], "value");
        assert_eq!(args["config"]["count"], 42);
    }

    #[test]
    fn test_parse_no_tool_calls() {
        let input = "This is just normal text without any tool calls.";
        let (calls, normal) =
            try_tool_call_parse_xml(input, &XmlParserConfig::default(), None).unwrap();
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
        let result = try_tool_call_parse_xml(input, &XmlParserConfig::default(), None);
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

        let (calls, _) = try_tool_call_parse_xml(input, &XmlParserConfig::default(), None).unwrap();
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

        let (calls, _) = try_tool_call_parse_xml(input, &XmlParserConfig::default(), None).unwrap();
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

        let (calls, _) = try_tool_call_parse_xml(input, &XmlParserConfig::default(), None).unwrap();
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

        let (calls, _) = try_tool_call_parse_xml(input, &XmlParserConfig::default(), None).unwrap();
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].function.name, "search");

        let args: serde_json::Value = serde_json::from_str(&calls[0].function.arguments).unwrap();
        // This matches the original SGLang python implementation.
        assert_eq!(args["query"], "rust programming\n<parameter=limit>\n10");
    }

    #[test]
    fn test_schema_aware_type_conversion() {
        // This test matches the Python test_parse_streaming_increment_multiple_parameters
        // from the diff, showing schema-aware type conversion
        let tools = vec![ToolDefinition {
            name: "multi_param_func".to_string(),
            parameters: Some(serde_json::json!({
                "type": "object",
                "properties": {
                    "param1": {"type": "string"},
                    "param2": {"type": "float"},
                    "param3": {"type": "integer"},
                    "param4": {"type": "boolean"},
                    "param5": {"type": "object"},
                    "param6": {"type": "array"},
                    "param7": {"type": "null"},
                    "param8": {"type": "other_type"}
                },
                "required": ["param1", "param2", "param3", "param4", "param5", "param6", "param7", "param8"]
            })),
        }];

        let input = r#"<tool_call>
<function=multi_param_func>
<parameter=param1>42</parameter>
<parameter=param2>41.9</parameter>
<parameter=param3>42</parameter>
<parameter=param4>true</parameter>
<parameter=param5>{"key": "value"}</parameter>
<parameter=param6>[1, 2, 3]</parameter>
<parameter=param7>null</parameter>
<parameter=param8>{'arg1': 3, 'arg2': [1, 2]}</parameter>
</function>
</tool_call>"#;

        let (calls, _) =
            try_tool_call_parse_xml(input, &XmlParserConfig::default(), Some(&tools)).unwrap();
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].function.name, "multi_param_func");

        let args: serde_json::Value = serde_json::from_str(&calls[0].function.arguments).unwrap();

        // param1 is type "string" so "42" stays as string
        assert_eq!(args["param1"], "42");

        // param2 is type "float" so 41.9 is parsed as float
        assert_eq!(args["param2"], 41.9);

        // param3 is type "integer" so 42 is parsed as integer
        assert_eq!(args["param3"], 42);

        // param4 is type "boolean" so "true" is parsed as bool
        assert_eq!(args["param4"], true);

        // param5 is type "object" so JSON is parsed
        assert_eq!(args["param5"], serde_json::json!({"key": "value"}));

        // param6 is type "array" so JSON array is parsed
        assert_eq!(args["param6"], serde_json::json!([1, 2, 3]));

        // param7 is type "null" so "null" is parsed as null
        assert_eq!(args["param7"], serde_json::Value::Null);

        // param8 is other_type, uses literal_eval which converts Python-style dict
        assert_eq!(
            args["param8"],
            serde_json::json!({"arg1": 3, "arg2": [1, 2]})
        );
    }

    #[test]
    fn test_schema_aware_type_conversion_fallback() {
        // Test that invalid values fall back to strings with warnings
        let tools = vec![ToolDefinition {
            name: "test_func".to_string(),
            parameters: Some(serde_json::json!({
                "type": "object",
                "properties": {
                    "int_param": {"type": "integer"},
                    "float_param": {"type": "float"},
                    "bool_param": {"type": "boolean"}
                }
            })),
        }];

        let input = r#"<tool_call>
<function=test_func>
<parameter=int_param>not_an_int</parameter>
<parameter=float_param>not_a_float</parameter>
<parameter=bool_param>not_a_bool</parameter>
</function>
</tool_call>"#;

        let (calls, _) =
            try_tool_call_parse_xml(input, &XmlParserConfig::default(), Some(&tools)).unwrap();
        assert_eq!(calls.len(), 1);

        let args: serde_json::Value = serde_json::from_str(&calls[0].function.arguments).unwrap();

        // All should fall back to strings
        assert_eq!(args["int_param"], "not_an_int");
        assert_eq!(args["float_param"], "not_a_float");
        // bool_param with invalid value defaults to false
        assert_eq!(args["bool_param"], false);
    }

    #[test]
    fn test_no_schema_fallback_behavior() {
        // Without schema, behavior should match old safe_parse_value logic
        let input = r#"<tool_call>
<function=unknown_func>
<parameter=param1>42</parameter>
<parameter=param2>true</parameter>
<parameter=param3>hello</parameter>
</function>
</tool_call>"#;

        let (calls, _) = try_tool_call_parse_xml(input, &XmlParserConfig::default(), None).unwrap();
        assert_eq!(calls.len(), 1);

        let args: serde_json::Value = serde_json::from_str(&calls[0].function.arguments).unwrap();

        // Without schema, all values are returned as strings (no type inference)
        assert_eq!(args["param1"], "42");
        assert_eq!(args["param2"], "true");
        assert_eq!(args["param3"], "hello");
    }
}
