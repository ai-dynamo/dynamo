// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// Reference implementation:
// https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/function_call/kimik2_detector.py
// https://github.com/vllm-project/vllm/blob/main/vllm/tool_parsers/kimi_k2_tool_parser.py

use std::sync::OnceLock;

use regex::Regex;

use super::super::ToolDefinition;
use super::super::config::KimiK2ParserConfig;
use super::response::{CalledFunction, ToolCallResponse, ToolCallType};

static ID_REGEX: OnceLock<Regex> = OnceLock::new();

static TOOL_CALL_REGEX: OnceLock<Regex> = OnceLock::new();

/// Returns the cached regex that captures `function_id` (e.g. `functions.get_weather:0`) and
/// `arguments` (JSON object) between the configured `call_start`, `argument_begin`, and
/// `call_end` tokens.
///
/// The `function_id` pattern `[\w.]+:\d+` matches the `functions.name:index` format used by
/// Kimi K2, consistent with sglang/vllm reference implementations.
fn get_tool_call_regex(config: &KimiK2ParserConfig) -> &'static Regex {
    TOOL_CALL_REGEX.get_or_init(|| {
        let pattern = format!(
            r"(?s){}\s*(?P<function_id>[\w.]+:\d+)\s*{}\s*(?P<arguments>\{{.*?\}})\s*{}",
            regex::escape(&config.call_start),
            regex::escape(&config.argument_begin),
            regex::escape(&config.call_end),
        );
        Regex::new(&pattern).expect("Failed to compile kimi k2 tool call regex")
    })
}

fn get_id_regex() -> &'static Regex {
    ID_REGEX.get_or_init(|| {
        Regex::new(r"^(?:functions\.)?(?P<name>[\w\.]+):(?P<index>\d+)$")
            .expect("Failed to compile kimi k2 id regex")
    })
}

/// Check if a chunk contains the start of a Kimi K2 style tool call.
/// Detects `<|tool_calls_section_begin|>` (or singular variant) or partial match for streaming.
pub fn detect_tool_call_start_kimi_k2(chunk: &str, config: &KimiK2ParserConfig) -> bool {
    for start_token in &config.section_start_variants {
        debug_assert!(
            start_token.is_ascii(),
            "Kimi K2 section tokens must be ASCII for safe byte slicing, got: {start_token:?}"
        );

        // Check for complete start token.
        if chunk.contains(start_token.as_str()) {
            return true;
        }

        // Check for partial match at the end of the chunk (for streaming).
        for i in 1..start_token.len() {
            if chunk.ends_with(&start_token[..i]) {
                return true;
            }
        }
    }

    false
}

/// Returns the position after `<|tool_calls_section_end|>` (or singular variant) or the length
/// of the chunk if not found.
pub fn find_tool_call_end_position_kimi_k2(chunk: &str, config: &KimiK2ParserConfig) -> usize {
    // Find the earliest matching end token variant.
    let mut earliest: Option<usize> = None;
    for end_token in &config.section_end_variants {
        if let Some(pos) = chunk.find(end_token.as_str()) {
            let end_pos = pos + end_token.len();
            earliest = Some(earliest.map_or(end_pos, |e: usize| e.min(end_pos)));
        }
    }
    earliest.unwrap_or(chunk.len())
}

/// Format:
/// ```text
/// <|tool_calls_section_begin|>
/// <|tool_call_begin|>functions.get_weather:0<|tool_call_argument_begin|>{"location":"NYC"}<|tool_call_end|>
/// <|tool_calls_section_end|>
/// ```
///
/// Returns (parsed_tool_calls, normal_text_content)
pub fn try_tool_call_parse_kimi_k2(
    message: &str,
    config: &KimiK2ParserConfig,
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

/// Find the first occurrence of any section start variant in `text[cursor..]`.
/// Returns `(relative_position, matched_token_length)` or `None`.
fn find_section_start(
    text: &str,
    cursor: usize,
    config: &KimiK2ParserConfig,
) -> Option<(usize, usize)> {
    let mut best: Option<(usize, usize)> = None;
    for variant in &config.section_start_variants {
        if let Some(pos) = text[cursor..].find(variant.as_str())
            && best.is_none_or(|(bp, _)| pos < bp)
        {
            best = Some((pos, variant.len()));
        }
    }
    best
}

/// Find the first occurrence of any section end variant in `text[from..]`.
/// Returns `(relative_position, matched_token_length)` or `None`.
fn find_section_end(
    text: &str,
    from: usize,
    config: &KimiK2ParserConfig,
) -> Option<(usize, usize)> {
    let mut best: Option<(usize, usize)> = None;
    for variant in &config.section_end_variants {
        if let Some(pos) = text[from..].find(variant.as_str())
            && best.is_none_or(|(bp, _)| pos < bp)
        {
            best = Some((pos, variant.len()));
        }
    }
    best
}

/// Extract tool calls and normal text from message.
fn extract_tool_calls(
    text: &str,
    config: &KimiK2ParserConfig,
    tools: Option<&[ToolDefinition]>,
) -> anyhow::Result<(String, Vec<ToolCallResponse>)> {
    let mut normal_parts = Vec::new();
    let mut calls = Vec::new();
    let mut cursor = 0;

    while cursor < text.len() {
        if let Some((start_pos, _start_len)) = find_section_start(text, cursor, config) {
            let abs_start = cursor + start_pos;

            // Add text before tool call section to normal parts.
            normal_parts.push(&text[cursor..abs_start]);

            if let Some((end_pos, end_len)) = find_section_end(text, abs_start, config) {
                let abs_end = abs_start + end_pos + end_len;
                let block = &text[abs_start..abs_end];

                // Parse individual tool calls within this section block.
                if let Ok(mut parsed_calls) = parse_section_block(block, config, tools) {
                    calls.append(&mut parsed_calls);
                }

                cursor = abs_end;
            } else {
                // No end token found -> treat the rest as normal text.
                normal_parts.push(&text[abs_start..]);
                break;
            }
        } else {
            // No more tool call sections.
            normal_parts.push(&text[cursor..]);
            break;
        }
    }

    let normal_text = normal_parts.join("").trim().to_string();
    Ok((normal_text, calls))
}

/// Parse a tool calls section block, extracting individual tool calls.
///
/// The block is between `<|tool_calls_section_begin|>` and `<|tool_calls_section_end|>`.
/// Each individual call is between `<|tool_call_begin|>` and `<|tool_call_end|>`.
fn parse_section_block(
    block: &str,
    config: &KimiK2ParserConfig,
    tools: Option<&[ToolDefinition]>,
) -> anyhow::Result<Vec<ToolCallResponse>> {
    let tool_call_regex = get_tool_call_regex(config);
    let id_regex = get_id_regex();

    let mut results = Vec::new();

    for cap in tool_call_regex.captures_iter(block) {
        let function_id = cap
            .name("function_id")
            .map(|m| m.as_str().trim())
            .unwrap_or("");
        let arguments_raw = cap
            .name("arguments")
            .map(|m| m.as_str().trim())
            .unwrap_or("{}");

        // Parse function ID
        let function_name = if let Some(id_cap) = id_regex.captures(function_id) {
            id_cap
                .name("name")
                .map(|m| m.as_str().to_string())
                .unwrap_or_default()
        } else {
            // Fallback: use the whole ID as the function name
            tracing::warn!(
                "Unexpected tool_call_id format: '{}', using as-is",
                function_id
            );
            function_id.to_string()
        };

        if function_name.is_empty() {
            continue;
        }

        // Validate function name against tools if provided
        if let Some(tools) = tools
            && !tools.iter().any(|t| t.name == function_name)
        {
            tracing::warn!("Tool '{}' is not defined in the tools list.", function_name);
        }

        // Validate JSON arguments
        let arguments_json = match serde_json::from_str::<serde_json::Value>(arguments_raw) {
            Ok(val) => serde_json::to_string(&val)?,
            Err(e) => {
                tracing::warn!(
                    "Failed to parse JSON arguments for tool '{}': {}. Using raw string.",
                    function_name,
                    e,
                );
                arguments_raw.to_string()
            }
        };

        // NOTE: Unlike other parsers (XML, DSML) which generate `call-{UUID}` IDs,
        // we preserve the model's native function_id (e.g., "functions.bash:0") here.
        // This matches the behavior of vllm/sglang and is required for Kimi K2 compatibility.
        let tool_call = ToolCallResponse {
            id: function_id.to_string(),
            tp: ToolCallType::Function,
            function: CalledFunction {
                name: function_name,
                arguments: arguments_json,
            },
        };

        results.push(tool_call);
    }

    Ok(results)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn default_config() -> KimiK2ParserConfig {
        KimiK2ParserConfig::default()
    }

    #[test]
    fn test_detect_tool_call_start() {
        let config = default_config();
        assert!(detect_tool_call_start_kimi_k2(
            "<|tool_calls_section_begin|>",
            &config
        ));
        assert!(detect_tool_call_start_kimi_k2(
            "text <|tool_calls_section_begin|>",
            &config
        ));
        // Partial match at end
        assert!(detect_tool_call_start_kimi_k2("<|tool_calls_sec", &config));
        assert!(detect_tool_call_start_kimi_k2("<|", &config));
        // No match
        assert!(!detect_tool_call_start_kimi_k2(
            "no tool call here",
            &config
        ));
        assert!(!detect_tool_call_start_kimi_k2("toolcall", &config));
    }

    #[test]
    fn test_find_tool_call_end_position() {
        let config = default_config();
        let text = "<|tool_calls_section_begin|><|tool_call_begin|>functions.test:0<|tool_call_argument_begin|>{}<|tool_call_end|><|tool_calls_section_end|>more text";
        let pos = find_tool_call_end_position_kimi_k2(text, &config);
        assert_eq!(&text[pos..], "more text");

        let text_no_end = "<|tool_calls_section_begin|><|tool_call_begin|>functions.test:0";
        let pos = find_tool_call_end_position_kimi_k2(text_no_end, &config);
        assert_eq!(pos, text_no_end.len());
    }

    #[test]
    fn test_parse_simple_tool_call() {
        let config = default_config();
        let input = r#"<|tool_calls_section_begin|><|tool_call_begin|>functions.get_weather:0<|tool_call_argument_begin|>{"location":"NYC"}<|tool_call_end|><|tool_calls_section_end|>"#;

        let (calls, normal) = try_tool_call_parse_kimi_k2(input, &config, None).unwrap();
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].function.name, "get_weather");
        assert_eq!(normal, Some("".to_string()));

        let args: serde_json::Value = serde_json::from_str(&calls[0].function.arguments).unwrap();
        assert_eq!(args["location"], "NYC");
    }

    #[test]
    fn test_parse_multiple_args() {
        let config = default_config();
        let input = r#"<|tool_calls_section_begin|><|tool_call_begin|>functions.get_weather:0<|tool_call_argument_begin|>{"location":"San Francisco, CA","unit":"fahrenheit"}<|tool_call_end|><|tool_calls_section_end|>"#;

        let (calls, _) = try_tool_call_parse_kimi_k2(input, &config, None).unwrap();
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].function.name, "get_weather");

        let args: serde_json::Value = serde_json::from_str(&calls[0].function.arguments).unwrap();
        assert_eq!(args["location"], "San Francisco, CA");
        assert_eq!(args["unit"], "fahrenheit");
    }

    #[test]
    fn test_parse_multiple_tool_calls() {
        let config = default_config();
        let input = r#"<|tool_calls_section_begin|><|tool_call_begin|>functions.get_weather:0<|tool_call_argument_begin|>{"location":"NYC"}<|tool_call_end|><|tool_call_begin|>functions.get_time:1<|tool_call_argument_begin|>{"timezone":"EST"}<|tool_call_end|><|tool_calls_section_end|>"#;

        let (calls, normal) = try_tool_call_parse_kimi_k2(input, &config, None).unwrap();
        assert_eq!(calls.len(), 2);
        assert_eq!(calls[0].function.name, "get_weather");
        assert_eq!(calls[1].function.name, "get_time");
        assert_eq!(normal, Some("".to_string()));

        let args0: serde_json::Value = serde_json::from_str(&calls[0].function.arguments).unwrap();
        let args1: serde_json::Value = serde_json::from_str(&calls[1].function.arguments).unwrap();
        assert_eq!(args0["location"], "NYC");
        assert_eq!(args1["timezone"], "EST");
    }

    #[test]
    fn test_parse_with_normal_text() {
        let config = default_config();
        let input = r#"I'll help you with that. <|tool_calls_section_begin|><|tool_call_begin|>functions.get_weather:0<|tool_call_argument_begin|>{"location":"Dallas"}<|tool_call_end|><|tool_calls_section_end|> Let me check."#;

        let (calls, normal) = try_tool_call_parse_kimi_k2(input, &config, None).unwrap();
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].function.name, "get_weather");
        assert_eq!(
            normal,
            Some("I'll help you with that.  Let me check.".to_string())
        );
    }

    #[test]
    fn test_parse_no_arg_call() {
        let config = default_config();
        let input = r#"<|tool_calls_section_begin|><|tool_call_begin|>functions.get_current_time:0<|tool_call_argument_begin|>{}<|tool_call_end|><|tool_calls_section_end|>"#;

        let (calls, _) = try_tool_call_parse_kimi_k2(input, &config, None).unwrap();
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].function.name, "get_current_time");

        let args: serde_json::Value = serde_json::from_str(&calls[0].function.arguments).unwrap();
        assert!(args.as_object().unwrap().is_empty());
    }

    #[test]
    fn test_parse_no_tool_calls() {
        let config = default_config();
        let input = "This is just normal text without any tool calls.";

        let (calls, normal) = try_tool_call_parse_kimi_k2(input, &config, None).unwrap();
        assert_eq!(calls.len(), 0);
        assert_eq!(normal, Some(input.to_string()));
    }

    #[test]
    fn test_parse_without_functions_prefix() {
        let config = default_config();
        // Some models may emit without the "functions." prefix
        let input = r#"<|tool_calls_section_begin|><|tool_call_begin|>get_weather:0<|tool_call_argument_begin|>{"location":"NYC"}<|tool_call_end|><|tool_calls_section_end|>"#;

        let (calls, _) = try_tool_call_parse_kimi_k2(input, &config, None).unwrap();
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].function.name, "get_weather");
    }

    #[test]
    fn test_parse_with_tool_validation() {
        let config = default_config();
        let tools = vec![ToolDefinition {
            name: "get_weather".to_string(),
            parameters: Some(serde_json::json!({
                "type": "object",
                "properties": {
                    "location": {"type": "string"}
                }
            })),
        }];

        let input = r#"<|tool_calls_section_begin|><|tool_call_begin|>functions.get_weather:0<|tool_call_argument_begin|>{"location":"NYC"}<|tool_call_end|><|tool_calls_section_end|>"#;

        let (calls, _) = try_tool_call_parse_kimi_k2(input, &config, Some(&tools)).unwrap();
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].function.name, "get_weather");
    }

    #[test]
    fn test_parse_malformed_no_section_end() {
        let config = default_config();
        let input = r#"<|tool_calls_section_begin|><|tool_call_begin|>functions.get_weather:0<|tool_call_argument_begin|>{"location":"NYC"}<|tool_call_end|>"#;

        // Should handle gracefully - section_end not found so whole text is treated as normal
        let (calls, normal) = try_tool_call_parse_kimi_k2(input, &config, None).unwrap();
        assert_eq!(
            calls.len(),
            0,
            "No tool calls should be parsed without section end"
        );
        assert_eq!(
            normal,
            Some(input.to_string()),
            "Input should be preserved as normal text"
        );
    }

    #[test]
    fn test_parse_with_whitespace() {
        let config = default_config();
        let input = "<|tool_calls_section_begin|>\n<|tool_call_begin|> functions.search:0 <|tool_call_argument_begin|> {\"query\":\"rust programming\"} <|tool_call_end|>\n<|tool_calls_section_end|>";

        let (calls, _) = try_tool_call_parse_kimi_k2(input, &config, None).unwrap();
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].function.name, "search");

        let args: serde_json::Value = serde_json::from_str(&calls[0].function.arguments).unwrap();
        assert_eq!(args["query"], "rust programming");
    }

    #[test]
    fn test_parse_complex_json_arguments() {
        let config = default_config();
        let input = r#"<|tool_calls_section_begin|><|tool_call_begin|>functions.process_data:0<|tool_call_argument_begin|>{"items":[1,2,3],"config":{"nested":true}}<|tool_call_end|><|tool_calls_section_end|>"#;

        let (calls, _) = try_tool_call_parse_kimi_k2(input, &config, None).unwrap();
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].function.name, "process_data");

        let args: serde_json::Value = serde_json::from_str(&calls[0].function.arguments).unwrap();
        assert_eq!(args["items"], serde_json::json!([1, 2, 3]));
        assert_eq!(args["config"]["nested"], true);
    }

    #[test]
    fn test_parse_deeply_nested_json_multiple_calls() {
        let config = default_config();
        // Multiple tool calls with deeply nested JSON - stress test for regex backtracking
        let input = r#"<|tool_calls_section_begin|><|tool_call_begin|>functions.create_config:0<|tool_call_argument_begin|>{"database":{"primary":{"host":"db1.example.com","port":5432,"options":{"ssl":true,"pool":{"min":5,"max":20}}},"replica":{"host":"db2.example.com","port":5432}},"features":["auth","logging"]}<|tool_call_end|><|tool_call_begin|>functions.deploy:1<|tool_call_argument_begin|>{"env":"production","services":[{"name":"api","replicas":3,"config":{"memory":"2Gi","cpu":"1000m"}},{"name":"worker","replicas":2,"config":{"memory":"4Gi","cpu":"2000m"}}]}<|tool_call_end|><|tool_call_begin|>functions.notify:2<|tool_call_argument_begin|>{"channels":["slack","email"],"message":"Deployment started"}<|tool_call_end|><|tool_calls_section_end|>"#;

        let (calls, normal) = try_tool_call_parse_kimi_k2(input, &config, None).unwrap();
        assert_eq!(calls.len(), 3);

        assert_eq!(calls[0].function.name, "create_config");
        assert_eq!(calls[0].id, "functions.create_config:0");
        let args0: serde_json::Value = serde_json::from_str(&calls[0].function.arguments).unwrap();
        assert_eq!(args0["database"]["primary"]["options"]["pool"]["max"], 20);

        assert_eq!(calls[1].function.name, "deploy");
        assert_eq!(calls[1].id, "functions.deploy:1");
        let args1: serde_json::Value = serde_json::from_str(&calls[1].function.arguments).unwrap();
        assert_eq!(args1["services"][0]["config"]["memory"], "2Gi");

        assert_eq!(calls[2].function.name, "notify");
        assert_eq!(calls[2].id, "functions.notify:2");
        let args2: serde_json::Value = serde_json::from_str(&calls[2].function.arguments).unwrap();
        assert_eq!(args2["channels"], serde_json::json!(["slack", "email"]));

        assert_eq!(normal, Some("".to_string()));
    }

    #[test]
    fn test_detect_singular_section_start() {
        let config = default_config();
        // Singular variant: <|tool_call_section_begin|> (without 's')
        assert!(detect_tool_call_start_kimi_k2(
            "<|tool_call_section_begin|>",
            &config
        ));
        // Partial match of singular variant
        assert!(detect_tool_call_start_kimi_k2(
            "text <|tool_call_section_b",
            &config
        ));
    }

    #[test]
    fn test_parse_with_singular_section_tokens() {
        let config = default_config();
        // Use singular form: tool_call_section_begin/end (without 's')
        let input = r#"<|tool_call_section_begin|><|tool_call_begin|>functions.get_weather:0<|tool_call_argument_begin|>{"location":"NYC"}<|tool_call_end|><|tool_call_section_end|>"#;

        let (calls, normal) = try_tool_call_parse_kimi_k2(input, &config, None).unwrap();
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].function.name, "get_weather");
        assert_eq!(normal, Some("".to_string()));
    }

    #[test]
    fn test_find_end_position_singular_variant() {
        let config = default_config();
        // Singular variant end token
        let text = "<|tool_call_section_begin|><|tool_call_begin|>functions.test:0<|tool_call_argument_begin|>{}<|tool_call_end|><|tool_call_section_end|>more text";
        let pos = find_tool_call_end_position_kimi_k2(text, &config);
        assert_eq!(&text[pos..], "more text");
    }

    // --- Tests inspired by vllm/sglang coverage gaps ---

    #[test]
    fn test_parse_invalid_json_falls_back_to_raw_string() {
        // vllm: test_extract_tool_calls_invalid_json
        // Invalid JSON in arguments should fall back to raw string, not panic
        let config = default_config();
        let input = r#"<|tool_calls_section_begin|><|tool_call_begin|>functions.get_weather:0<|tool_call_argument_begin|>{invalid json here}<|tool_call_end|><|tool_calls_section_end|>"#;

        let (calls, _) = try_tool_call_parse_kimi_k2(input, &config, None).unwrap();
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].function.name, "get_weather");
        // Arguments should be preserved as raw string since JSON parsing failed
        assert_eq!(calls[0].function.arguments, "{invalid json here}");
    }

    #[test]
    fn test_parse_invalid_function_id_rejected_by_regex() {
        // vllm: test_extract_tool_calls_invalid_funcall
        // sglang: test_invalid_tool_call
        // After C2 fix, function_id regex requires [\w.]+:\d+ — IDs without :digit are rejected
        let config = default_config();

        // No colon+digit suffix at all
        let input1 = r#"<|tool_calls_section_begin|><|tool_call_begin|>just_a_name<|tool_call_argument_begin|>{"key":"val"}<|tool_call_end|><|tool_calls_section_end|>"#;
        let (calls, _) = try_tool_call_parse_kimi_k2(input1, &config, None).unwrap();
        assert_eq!(calls.len(), 0, "ID without :digit should be rejected");

        // Colon but non-digit suffix
        let input2 = r#"<|tool_calls_section_begin|><|tool_call_begin|>functions.get_weather:abc<|tool_call_argument_begin|>{"key":"val"}<|tool_call_end|><|tool_calls_section_end|>"#;
        let (calls, _) = try_tool_call_parse_kimi_k2(input2, &config, None).unwrap();
        assert_eq!(calls.len(), 0, "ID with :non-digit should be rejected");

        // Multiple colons (garbage)
        let input3 = r#"<|tool_calls_section_begin|><|tool_call_begin|>:::0<|tool_call_argument_begin|>{"key":"val"}<|tool_call_end|><|tool_calls_section_end|>"#;
        let (calls, _) = try_tool_call_parse_kimi_k2(input3, &config, None).unwrap();
        assert_eq!(calls.len(), 0, "Garbage ID should be rejected");

        // Valid call mixed with invalid — only valid should be extracted
        let input4 = r#"<|tool_calls_section_begin|><|tool_call_begin|>no_colon<|tool_call_argument_begin|>{"a":"b"}<|tool_call_end|><|tool_call_begin|>functions.valid:0<|tool_call_argument_begin|>{"x":"y"}<|tool_call_end|><|tool_calls_section_end|>"#;
        let (calls, _) = try_tool_call_parse_kimi_k2(input4, &config, None).unwrap();
        assert_eq!(calls.len(), 1, "Only valid call should be extracted");
        assert_eq!(calls[0].function.name, "valid");
    }

    #[test]
    fn test_parse_angle_brackets_in_json_arguments() {
        // vllm: angle_brackets_in_json
        // JSON values containing <tag> constructs should not confuse the parser,
        // since Kimi markers use <| prefix which is distinct from bare <
        let config = default_config();
        let input = r#"<|tool_calls_section_begin|><|tool_call_begin|>functions.render_html:0<|tool_call_argument_begin|>{"template":"<div class=\"main\"><h1>Title</h1><p>Content</p></div>","format":"html"}<|tool_call_end|><|tool_calls_section_end|>"#;

        let (calls, _) = try_tool_call_parse_kimi_k2(input, &config, None).unwrap();
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].function.name, "render_html");

        let args: serde_json::Value = serde_json::from_str(&calls[0].function.arguments).unwrap();
        assert!(args["template"].as_str().unwrap().contains("<div"));
        assert!(args["template"].as_str().unwrap().contains("</div>"));
        assert_eq!(args["format"], "html");
    }

    #[test]
    fn test_parse_three_concatenated_calls_no_spacing() {
        // vllm: concatenated_tool_calls_bug_fix, three_concatenated_tool_calls
        // Three tool calls concatenated with zero whitespace between them
        let config = default_config();
        let input = "<|tool_calls_section_begin|>\
            <|tool_call_begin|>functions.search:0<|tool_call_argument_begin|>{\"q\":\"rust\"}<|tool_call_end|>\
            <|tool_call_begin|>functions.search:1<|tool_call_argument_begin|>{\"q\":\"python\"}<|tool_call_end|>\
            <|tool_call_begin|>functions.search:2<|tool_call_argument_begin|>{\"q\":\"go\"}<|tool_call_end|>\
            <|tool_calls_section_end|>";

        let (calls, normal) = try_tool_call_parse_kimi_k2(input, &config, None).unwrap();
        assert_eq!(calls.len(), 3);
        assert_eq!(calls[0].function.name, "search");
        assert_eq!(calls[0].id, "functions.search:0");
        assert_eq!(calls[1].function.name, "search");
        assert_eq!(calls[1].id, "functions.search:1");
        assert_eq!(calls[2].function.name, "search");
        assert_eq!(calls[2].id, "functions.search:2");

        let a0: serde_json::Value = serde_json::from_str(&calls[0].function.arguments).unwrap();
        let a1: serde_json::Value = serde_json::from_str(&calls[1].function.arguments).unwrap();
        let a2: serde_json::Value = serde_json::from_str(&calls[2].function.arguments).unwrap();
        assert_eq!(a0["q"], "rust");
        assert_eq!(a1["q"], "python");
        assert_eq!(a2["q"], "go");
        assert_eq!(normal, Some("".to_string()));
    }

    #[test]
    fn test_parse_newlines_in_json_arguments() {
        // vllm: newlines_in_json
        // Multi-line formatted JSON arguments (model may emit pretty-printed JSON)
        let config = default_config();
        let input = "<|tool_calls_section_begin|><|tool_call_begin|>functions.create_user:0<|tool_call_argument_begin|>{\n  \"name\": \"John Doe\",\n  \"address\": {\n    \"street\": \"123 Main St\",\n    \"city\": \"Springfield\"\n  },\n  \"tags\": [\n    \"admin\",\n    \"active\"\n  ]\n}<|tool_call_end|><|tool_calls_section_end|>";

        let (calls, _) = try_tool_call_parse_kimi_k2(input, &config, None).unwrap();
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].function.name, "create_user");

        let args: serde_json::Value = serde_json::from_str(&calls[0].function.arguments).unwrap();
        assert_eq!(args["name"], "John Doe");
        assert_eq!(args["address"]["city"], "Springfield");
        assert_eq!(args["tags"], serde_json::json!(["admin", "active"]));
    }

    #[test]
    fn test_parse_empty_tool_section() {
        // vllm: test_empty_tool_section
        // Section begin immediately followed by section end, no tool calls inside
        let config = default_config();
        let input = "Here is my answer. <|tool_calls_section_begin|><|tool_calls_section_end|> And more text.";

        let (calls, normal) = try_tool_call_parse_kimi_k2(input, &config, None).unwrap();
        assert_eq!(calls.len(), 0, "Empty section should produce no tool calls");
        assert_eq!(
            normal,
            Some("Here is my answer.  And more text.".to_string()),
            "Text around empty section should be preserved"
        );
    }
}
