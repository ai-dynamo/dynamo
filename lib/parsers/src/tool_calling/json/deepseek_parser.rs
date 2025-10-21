// SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use regex::RegexBuilder;
use serde_json::Value;
use uuid::Uuid;

use super::config::JsonParserConfig;
use super::response::{CalledFunction, ToolCallResponse, ToolCallType};

/// Extract individual tool call blocks from the input string.
/// Returns a list of strings, each representing one tool call block.
///
/// DeepSeek format: <｜tool▁call▁begin｜>{name}<｜tool▁sep｜>{args}<｜tool▁call▁end｜>
///
/// DeepSeek uses nested tokens:
/// - Wrapper tokens: <｜tool▁calls▁begin｜> ... <｜tool▁calls▁end｜> (wraps all tool calls)
/// - Individual tokens: <｜tool▁call▁begin｜> ... <｜tool▁call▁end｜> (individual call)
fn extract_tool_call_blocks(
    input: &str,
    start_tokens: &[String],
    end_tokens: &[String],
) -> Vec<String> {
    let mut blocks = Vec::new();

    // Filter tokens to find individual call markers (not the wrapper "calls" versions)
    let individual_start_tokens: Vec<&String> = start_tokens
        .iter()
        .filter(|t| t.contains("tool_call_begin") || t.contains("tool▁call▁begin"))
        .collect();

    let individual_end_tokens: Vec<&String> = end_tokens
        .iter()
        .filter(|t| t.contains("tool_call_end") || t.contains("tool▁call▁end"))
        .collect();

    // Try all combinations of individual start and end tokens
    for start_token in individual_start_tokens.iter() {
        for end_token in individual_end_tokens.iter() {
            if start_token.is_empty() || end_token.is_empty() {
                continue;
            }

            // Build regex pattern with escaped tokens
            let escaped_start = regex::escape(start_token);
            let escaped_end = regex::escape(end_token);
            let pattern = format!(r"{}(.*?){}", escaped_start, escaped_end);

            if let Ok(regex) = RegexBuilder::new(&pattern)
                .dot_matches_new_line(true)
                .build()
            {
                for capture in regex.captures_iter(input) {
                    if let Some(matched) = capture.get(1) {
                        // Don't trim the content - preserve whitespace for multiline JSON
                        let content = matched.as_str();
                        if !content.trim().is_empty() {
                            blocks.push(content.to_string());
                        }
                    }
                }

                // If we found matches with this token pair, don't try other combinations
                if !blocks.is_empty() {
                    return blocks;
                }
            }
        }
    }

    blocks
}

/// Parse a single tool call block that contains function name and arguments separated by a separator token.
///
/// Format: {function_name}<｜tool▁sep｜>{json_arguments}
fn parse_single_tool_call(block: &str, separator_tokens: &[String]) -> Option<(String, Value)> {
    // Try each separator token
    for sep_token in separator_tokens.iter() {
        if sep_token.is_empty() {
            continue;
        }

        if let Some((name_part, args_part)) = block.split_once(sep_token) {
            let function_name = name_part.trim();
            let args_str = args_part.trim();

            // Validate function name (should not be empty and should not contain JSON-like chars)
            if function_name.is_empty() || function_name.contains(['{', '}', '[', ']']) {
                continue;
            }

            // Try to parse arguments as JSON
            // First try parsing as-is
            if let Ok(arguments) = serde_json::from_str::<Value>(args_str) {
                return Some((function_name.to_string(), arguments));
            }

            // If that fails, try normalizing the JSON (handle multiline strings with unescaped newlines)
            // This is a lenient approach for malformed JSON that may come from LLMs
            let normalized = args_str
                .lines()
                .map(|line| line.trim_start())
                .collect::<Vec<_>>()
                .join(" ");

            if let Ok(arguments) = serde_json::from_str::<Value>(&normalized) {
                return Some((function_name.to_string(), arguments));
            }
        }
    }

    None
}

pub fn parse_tool_calls_deepseek_v3_1(
    message: &str,
    config: &JsonParserConfig,
) -> anyhow::Result<(Vec<ToolCallResponse>, Option<String>)> {
    // Format Structure:
    // <｜tool▁calls▁begin｜><｜tool▁call▁begin｜>{function_name}<｜tool▁sep｜>{json_arguments}<｜tool▁call▁end｜><｜tool▁calls▁end｜>
    let trimmed = message.trim();

    // Early exit if no content
    if trimmed.is_empty() {
        return Ok((vec![], Some(String::new())));
    }

    let tool_call_start_tokens = &config.tool_call_start_tokens;
    let tool_call_end_tokens = &config.tool_call_end_tokens;
    let separator_tokens = &config.tool_call_separator_tokens;

    // Early exit if no tokens configured
    if tool_call_start_tokens.is_empty() || separator_tokens.is_empty() {
        return Ok((vec![], Some(trimmed.to_string())));
    }

    // Check if tool call start token is present
    if !detect_tool_call_start_deepseek_v3_1(trimmed, config) {
        return Ok((vec![], Some(trimmed.to_string())));
    }

    // Extract normal text (content before the first wrapper start token)
    // Look for wrapper tokens like <｜tool▁calls▁begin｜> (note: "calls" not "call")
    let wrapper_tokens: Vec<&String> = tool_call_start_tokens
        .iter()
        .filter(|t| t.contains("tool_calls_begin") || t.contains("tool▁calls▁begin"))
        .collect();

    let normal_text = if !wrapper_tokens.is_empty() {
        wrapper_tokens
            .iter()
            .find_map(|token| {
                trimmed
                    .find(token.as_str())
                    .map(|idx| trimmed[..idx].to_string())
            })
            .unwrap_or_else(String::new)
    } else {
        // Fallback to first individual call token if no wrapper found
        tool_call_start_tokens
            .iter()
            .filter(|token| !token.is_empty())
            .find_map(|token| trimmed.find(token).map(|idx| trimmed[..idx].to_string()))
            .unwrap_or_else(String::new)
    };

    // Extract individual tool call blocks
    let blocks = extract_tool_call_blocks(trimmed, tool_call_start_tokens, tool_call_end_tokens);

    if blocks.is_empty() {
        // Found start token but no valid blocks
        return Ok((vec![], Some(trimmed.to_string())));
    }

    // Parse each block to extract function name and arguments
    let mut tool_calls: Vec<ToolCallResponse> = Vec::new();
    for block in blocks {
        if let Some((function_name, arguments)) = parse_single_tool_call(&block, separator_tokens) {
            tool_calls.push(ToolCallResponse {
                id: format!("call-{}", Uuid::new_v4()),
                tp: ToolCallType::Function,
                function: CalledFunction {
                    name: function_name,
                    arguments: serde_json::to_string(&arguments)?,
                },
            });
        }
    }

    // If no valid tool calls were parsed, return everything as normal text
    if tool_calls.is_empty() {
        return Ok((vec![], Some(trimmed.to_string())));
    }

    Ok((tool_calls, Some(normal_text)))
}

pub fn detect_tool_call_start_deepseek_v3_1(chunk: &str, config: &JsonParserConfig) -> bool {
    let trimmed = chunk.trim();
    if trimmed.is_empty() {
        return false;
    }

    // Check for complete start tokens first
    let has_complete_token = config
        .tool_call_start_tokens
        .iter()
        .any(|token| !token.is_empty() && trimmed.contains(token));

    if has_complete_token {
        return true;
    }

    // Check for partial start tokens (streaming scenario)
    // This handles cases where start tokens are split across multiple chunks
    config.tool_call_start_tokens.iter().any(|token| {
        if token.is_empty() {
            return false;
        }
        // Check if the chunk could be a prefix of this start token
        // Handle Unicode character boundaries properly
        for i in 1..=token.chars().count() {
            if let Some(prefix) = token.chars().take(i).collect::<String>().get(..) {
                let prefix_str = &prefix[..prefix.len()];
                if trimmed == prefix_str || trimmed.ends_with(prefix_str) {
                    return true;
                }
            }
        }
        false
    })
}

#[cfg(test)]
mod tests {
    use super::super::config::ToolCallConfig;
    use super::*;

    fn extract_name_and_args(call: ToolCallResponse) -> (String, serde_json::Value) {
        let args: serde_json::Value = serde_json::from_str(&call.function.arguments).unwrap();
        (call.function.name, args)
    }

    #[test]
    fn test_parse_tool_calls_deepseek_v3_1_basic() {
        let text = r#"<｜tool▁calls▁begin｜><｜tool▁call▁begin｜>get_current_weather<｜tool▁sep｜>{"location": "Tokyo"}<｜tool▁call▁end｜><｜tool▁call▁begin｜>get_current_weather<｜tool▁sep｜>{"location": "Paris"}<｜tool▁call▁end｜><｜tool▁calls▁end｜><｜end▁of▁sentence｜>"#;
        let config = ToolCallConfig::deepseek_v3_1().json;
        let (result, content) = parse_tool_calls_deepseek_v3_1(text, &config).unwrap();
        assert_eq!(content, Some("".to_string()));
        assert_eq!(result.len(), 2);
        let (name, args) = extract_name_and_args(result[0].clone());
        assert_eq!(name, "get_current_weather");
        assert_eq!(args["location"], "Tokyo");
        let (name, args) = extract_name_and_args(result[1].clone());
        assert_eq!(name, "get_current_weather");
        assert_eq!(args["location"], "Paris");
    }

    #[test]
    fn test_parse_tool_calls_deepseek_v3_1_with_normal_text() {
        let text = r#"The following tool call retrieves weather information: <｜tool▁calls▁begin｜><｜tool▁call▁begin｜>get_current_weather<｜tool▁sep｜>{"location": "New York"}<｜tool▁call▁end｜><｜tool▁calls▁end｜><｜end▁of▁sentence｜>"#;
        let config = ToolCallConfig::deepseek_v3_1().json;
        let (result, content) = parse_tool_calls_deepseek_v3_1(text, &config).unwrap();
        assert_eq!(
            content,
            Some("The following tool call retrieves weather information: ".to_string())
        );
        assert_eq!(result.len(), 1);
        let (name, args) = extract_name_and_args(result[0].clone());
        assert_eq!(name, "get_current_weather");
        assert_eq!(args["location"], "New York");
    }

    #[test]
    fn test_parse_tool_calls_deepseek_v3_1_without_tool_call_start_token() {
        let text = r#"<｜tool▁call▁begin｜>get_current_weather宽带}{location": "Tokyo"}<｜tool▁call▁end｜><｜tool▁calls▁end｜>"#;
        let config = ToolCallConfig::deepseek_v3_1().json;
        let (result, content) = parse_tool_calls_deepseek_v3_1(text, &config).unwrap();
        assert_eq!(content, Some(text.to_string()));
        assert_eq!(result.len(), 0);
    }

    #[test]
    fn test_parse_tool_calls_deepseek_v3_1_with_multi_tool_calls_with_multiple_args() {
        let text = r#"<｜tool▁calls▁begin｜><｜tool▁call▁begin｜>get_current_weather<｜tool▁sep｜>{"location": "Berlin", "units": "metric"}<｜tool▁call▁end｜><｜tool▁call▁begin｜>get_weather_forecast<｜tool▁sep｜>{"location": "Berlin", "days": 7, "units": "imperial"}<｜tool▁call▁end｜><｜tool▁call▁begin｜>get_air_quality<｜tool▁sep｜>{"location": "Berlin", "radius": 50}<｜tool▁call▁end｜><｜tool▁calls▁end｜><｜end▁of▁sentence｜>"#;
        let config = ToolCallConfig::deepseek_v3_1().json;
        let (result, content) = parse_tool_calls_deepseek_v3_1(text, &config).unwrap();
        assert_eq!(content, Some("".to_string()));
        assert_eq!(result.len(), 3);
        let (name, args) = extract_name_and_args(result[0].clone());
        assert_eq!(name, "get_current_weather");
        assert_eq!(args["location"], "Berlin");
        assert_eq!(args["units"], "metric");
        let (name, args) = extract_name_and_args(result[1].clone());
        assert_eq!(name, "get_weather_forecast");
        assert_eq!(args["location"], "Berlin");
        assert_eq!(args["days"], 7);
        assert_eq!(args["units"], "imperial");
        let (name, args) = extract_name_and_args(result[2].clone());
        assert_eq!(name, "get_air_quality");
        assert_eq!(args["location"], "Berlin");
        assert_eq!(args["radius"], 50);
    }

    #[test]
    fn test_parse_tool_calls_deepseek_v3_1_with_invalid_json() {
        // Everything is normal text in case of invalid json
        let text = r#"<｜tool▁calls▁begin｜><｜tool▁call▁begin｜>get_current_weather}{location": "Tokyo"}<｜tool▁call▁end｜><｜tool▁calls▁end｜>"#;
        let config = ToolCallConfig::deepseek_v3_1().json;
        let (result, content) = parse_tool_calls_deepseek_v3_1(text, &config).unwrap();
        assert_eq!(content, Some(text.trim().to_string()));
        assert_eq!(result.len(), 0);
    }

    #[test]
    fn test_parse_tool_calls_deepseek_v3_1_with_multi_tool_calls_with_normal_text() {
        // Everything is normal text in case of invalid json
        let text = r#"The following tool calls retrieve weather information: <｜tool▁calls▁begin｜><｜tool▁call▁begin｜>get_current_weather宽带}{location": "Tokyo"}<｜tool▁call▁end｜><｜tool▁call▁begin｜>get_weather_forecast宽带}{location": "Berlin", "days": 7, "units": "imperial"}<｜tool▁call▁end｜><｜tool▁call▁begin｜>get_air_quality宽带}{location": "Berlin", "radius": 50}<｜tool▁call▁end｜><｜tool▁calls▁end｜>"#;
        let config = ToolCallConfig::deepseek_v3_1().json;
        let (result, content) = parse_tool_calls_deepseek_v3_1(text, &config).unwrap();
        assert_eq!(content, Some(text.trim().to_string()));
        assert_eq!(result.len(), 0);
    }

    #[test]
    fn test_parse_tool_calls_deepseek_v3_1_with_multiline_json() {
        let text = r#"I'll help you understand this codebase. Let me start by exploring the structure and key
  files to provide you with a comprehensive
  explanation.<｜tool▁calls▁begin｜><｜tool▁call▁begin｜>TodoWrite<｜tool▁sep｜>{"todos":
  [{"content": "Explore the root directory structure", "status": "in_progress", "activeForm":
   "Exploring the root directory structure"}, {"content": "Examine package.json and
  configuration files", "status": "pending", "activeForm": "Examining package.json and
  configuration files"}, {"content": "Analyze source code structure and key modules",
  "status": "pending", "activeForm": "Analyzing source code structure and key modules"},
  {"content": "Identify main entry points and architectural patterns", "status": "pending",
  "activeForm": "Identifying main entry points and architectural patterns"}, {"content":
  "Summarize the codebase purpose and functionality", "status": "pending", "activeForm":
  "Summarizing the codebase purpose and
  functionality"}]}<｜tool▁call▁end｜><｜tool▁calls▁end｜>"#;
        let config = ToolCallConfig::deepseek_v3_1().json;

        let (tool_call_results, normal_content) =
            parse_tool_calls_deepseek_v3_1(text, &config).unwrap();

        assert_eq!(tool_call_results.len(), 1);

        let (name, args) = extract_name_and_args(tool_call_results[0].clone());
        assert_eq!(name, "TodoWrite");
        assert_eq!(tool_call_results[0].tp, ToolCallType::Function);

        let todos_array = args["todos"].as_array().unwrap();
        assert_eq!(todos_array.len(), 5);

        assert_eq!(
            todos_array[0]["content"],
            "Explore the root directory structure"
        );
        assert_eq!(todos_array[0]["status"], "in_progress");
        assert_eq!(
            todos_array[0]["activeForm"],
            "Exploring the root directory structure"
        );

        assert_eq!(
            todos_array[1]["content"],
            "Examine package.json and configuration files"
        );
        assert_eq!(todos_array[1]["status"], "pending");

        assert_eq!(
            todos_array[4]["content"],
            "Summarize the codebase purpose and functionality"
        );
        assert_eq!(todos_array[4]["status"], "pending");

        assert_eq!(
            normal_content,
            Some("I'll help you understand this codebase. Let me start by exploring the structure and key\n  files to provide you with a comprehensive\n  explanation.".to_string())
        );
    }
}

#[cfg(test)]
mod detect_parser_tests {
    use super::super::config::ToolCallConfig;
    use super::*;
    #[test]
    fn test_detect_tool_call_start_deepseek_v3_1_chunk_with_tool_call_start_token() {
        let text = r#"<｜tool▁calls▁begin｜><｜tool▁call▁begin｜>get_current_weather宽带}"#;
        let config = ToolCallConfig::deepseek_v3_1().json;
        let result = detect_tool_call_start_deepseek_v3_1(text, &config);
        assert!(result);
    }

    #[test]
    fn test_detect_tool_call_start_deepseek_v3_1_chunk_without_tool_call_start_token() {
        let text = r#"<｜tool▁call▁begin｜>get_current_weather宽带}"#;
        let config = ToolCallConfig::deepseek_v3_1().json;
        let result = detect_tool_call_start_deepseek_v3_1(text, &config);
        assert!(result);
    }

    #[test]
    fn test_detect_tool_call_start_deepseek_v3_1_chunk_with_tool_call_start_token_in_middle() {
        let text = r#"The following tool calls retrieve weather information: <｜tool▁calls▁begin｜><｜tool▁call▁begin｜>get_current_weather宽带}"#;
        let config = ToolCallConfig::deepseek_v3_1().json;
        let result = detect_tool_call_start_deepseek_v3_1(text, &config);
        assert!(result);
    }

    #[test]
    fn test_detect_tool_call_start_deepseek_v3_1_partial_tokens() {
        // Test partial token detection for streaming scenarios with unicode characters
        let config = ToolCallConfig::deepseek_v3_1().json;

        // Test various partial prefixes
        assert!(
            detect_tool_call_start_deepseek_v3_1("<", &config),
            "'<' should be detected as potential start"
        );
        assert!(
            detect_tool_call_start_deepseek_v3_1("<｜", &config),
            "'<｜' should be detected as potential start"
        );
        assert!(
            detect_tool_call_start_deepseek_v3_1("<｜tool", &config),
            "'<｜tool' should be detected as potential start"
        );
        assert!(
            detect_tool_call_start_deepseek_v3_1("<｜tool▁calls", &config),
            "'<｜tool▁calls' should be detected as potential start"
        );

        // Test that unrelated text is not detected
        assert!(
            !detect_tool_call_start_deepseek_v3_1("hello world", &config),
            "'hello world' should not be detected"
        );
        assert!(
            !detect_tool_call_start_deepseek_v3_1("xyz", &config),
            "'xyz' should not be detected"
        );
    }
}
