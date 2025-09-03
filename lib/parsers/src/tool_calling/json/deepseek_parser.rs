// SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use regex::Regex;
use serde_json::Value;
use std::sync::OnceLock;

use super::config::JsonParserConfig;
use super::response::{CalledFunction, ToolCallResponse, ToolCallType};

static DEEPSEEK_V3_1_OUTER_REGEX: OnceLock<Regex> = OnceLock::new();
static DEEPSEEK_V3_1_INNER_REGEX: OnceLock<Regex> = OnceLock::new();

pub fn get_deepseek_v3_1_outer_regex() -> &'static Regex {
    DEEPSEEK_V3_1_OUTER_REGEX.get_or_init(|| {
        // Outer regex: matches the entire tool call block
        Regex::new(r"(?s)<｜tool▁call▁begin｜>.*?<｜tool▁call▁end｜>")
            .expect("Failed to compile deepseek v3.1 outer regex pattern")
    })
}

pub fn get_deepseek_v3_1_inner_regex() -> &'static Regex {
    DEEPSEEK_V3_1_INNER_REGEX.get_or_init(|| {
        // Inner regex: captures function name and arguments between sep tokens
        Regex::new(r"(?s)<｜tool▁call▁begin｜>(.*?)<｜tool▁sep｜>(.*?)<｜tool▁call▁end｜>")
            .expect("Failed to compile deepseek v3.1 inner regex pattern")
    })
}

pub fn parse_tool_calls_deepseek_v3_1(
    message: &str,
    config: &JsonParserConfig,
) -> anyhow::Result<(Vec<ToolCallResponse>, Option<String>)> {
    // Format Structure:
    // <｜tool▁calls▁begin｜><｜tool▁call▁begin｜>{function_name}<｜tool▁sep｜>{json_arguments}<｜tool▁calls▁end｜><｜end▁of▁sentence｜>
    let trimmed = message.trim();

    let tool_call_start_tokens = &config.tool_call_start_tokens;

    // Early exit if no content or tool_call_start_tokens is empty
    if trimmed.is_empty() || tool_call_start_tokens.is_empty() {
        return Ok((vec![], Some(trimmed.to_string())));
    }

    // If tool call start token is not present then, no tool calls are there, return empty tool calls and the original trimmed string
    if let Some(start_token) = tool_call_start_tokens.first() {
        if !trimmed.contains(start_token) {
            return Ok((vec![], Some(trimmed.to_string())));
        }
    } else {
        // Invalid start token
        return Ok((vec![], Some(trimmed.to_string())));
    }

    let outer_re = get_deepseek_v3_1_outer_regex();
    let inner_re = get_deepseek_v3_1_inner_regex();

    let outer_matches = outer_re.find_iter(trimmed);

    let mut tool_calls: Vec<ToolCallResponse> = Vec::new();
    let mut call_idx = 0usize;
    // Two matches are there, first one using outer regex to extract multiple tool calls
    // Second one using inner regex to extract the structure of the tool call
    for outer_match in outer_matches {
        for grp in inner_re.captures_iter(outer_match.as_str()) {
            let Some(function_name) = grp.get(1).map(|x| x.as_str()) else {
                continue; // Skip if function name is not found
            };

            let Some(arg_match) = grp.get(2) else {
                continue; // Skip if arguments Match is not found.
            };

            let arguments = match serde_json::from_str::<Value>(arg_match.as_str()) {
                Ok(args) => args,
                Err(_) => {
                    continue; // Skip if arguments are not valid JSON
                }
            };

            call_idx += 1;
            tool_calls.push(ToolCallResponse {
                id: format!("call-{}", call_idx),
                tp: ToolCallType::Function,
                function: CalledFunction {
                    name: function_name.to_string(),
                    arguments: serde_json::to_string(&arguments)?,
                },
            });
        }
    }

    // Fast path: if no tool calls, just return early
    // This may happen due to invalid json or any other parsing error reasons
    if tool_calls.is_empty() {
        return Ok((vec![], Some(trimmed.to_string())));
    }

    // Safety: We already checked above that tool_call_start_tokens.first() is Some
    let start_token = tool_call_start_tokens.first().unwrap();
    let normal_text = trimmed
        .split_once(start_token)
        .map(|(before, _)| before.to_string())
        .unwrap_or_else(|| trimmed.to_string());

    Ok((tool_calls, Some(normal_text)))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn extract_name_and_args(call: ToolCallResponse) -> (String, serde_json::Value) {
        let args: serde_json::Value = serde_json::from_str(&call.function.arguments).unwrap();
        (call.function.name, args)
    }

    #[test]
    fn test_parse_tool_calls_deepseek_v3_1_basic() {
        let text = r#"<｜tool▁calls▁begin｜><｜tool▁call▁begin｜>get_current_weather<｜tool▁sep｜>{"location": "Tokyo"}<｜tool▁call▁end｜><｜tool▁call▁begin｜>get_current_weather<｜tool▁sep｜>{"location": "Paris"}<｜tool▁call▁end｜><｜tool▁calls▁end｜><｜end▁of▁sentence｜>"#;
        let config = JsonParserConfig {
            tool_call_start_tokens: vec!["<｜tool▁calls▁begin｜>".to_string()],
            tool_call_end_tokens: vec!["<｜tool▁calls▁end｜>".to_string()],
            ..Default::default()
        };
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
        let config = JsonParserConfig {
            tool_call_start_tokens: vec!["<｜tool▁calls▁begin｜>".to_string()],
            tool_call_end_tokens: vec!["<｜tool▁calls▁end｜>".to_string()],
            ..Default::default()
        };
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
        let config = JsonParserConfig {
            tool_call_start_tokens: vec!["<｜tool▁calls▁begin｜>".to_string()],
            tool_call_end_tokens: vec!["<｜tool▁calls▁end｜>".to_string()],
            ..Default::default()
        };
        let (result, content) = parse_tool_calls_deepseek_v3_1(text, &config).unwrap();
        assert_eq!(content, Some(text.to_string()));
        assert_eq!(result.len(), 0);
    }

    #[test]
    fn test_parse_tool_calls_deepseek_v3_1_with_multi_tool_calls_with_multiple_args() {
        let text = r#"<｜tool▁calls▁begin｜><｜tool▁call▁begin｜>get_current_weather<｜tool▁sep｜>{"location": "Berlin", "units": "metric"}<｜tool▁call▁end｜><｜tool▁call▁begin｜>get_weather_forecast<｜tool▁sep｜>{"location": "Berlin", "days": 7, "units": "imperial"}<｜tool▁call▁end｜><｜tool▁call▁begin｜>get_air_quality<｜tool▁sep｜>{"location": "Berlin", "radius": 50}<｜tool▁call▁end｜><｜tool▁calls▁end｜><｜end▁of▁sentence｜>"#;
        let config = JsonParserConfig {
            tool_call_start_tokens: vec!["<｜tool▁calls▁begin｜>".to_string()],
            tool_call_end_tokens: vec!["<｜tool▁calls▁end｜>".to_string()],
            ..Default::default()
        };
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
        let config = JsonParserConfig {
            tool_call_start_tokens: vec!["<｜tool▁calls▁begin｜>".to_string()],
            tool_call_end_tokens: vec!["<｜tool▁calls▁end｜>".to_string()],
            ..Default::default()
        };
        let (result, content) = parse_tool_calls_deepseek_v3_1(text, &config).unwrap();
        assert_eq!(content, Some(text.trim().to_string()));
        assert_eq!(result.len(), 0);
    }

    #[test]
    fn test_parse_tool_calls_deepseek_v3_1_with_multi_tool_calls_with_normal_text() {
        // Everything is normal text in case of invalid json
        let text = r#"The following tool calls retrieve weather information: <｜tool▁calls▁begin｜><｜tool▁call▁begin｜>get_current_weather宽带}{location": "Tokyo"}<｜tool▁call▁end｜><｜tool▁call▁begin｜>get_weather_forecast宽带}{location": "Berlin", "days": 7, "units": "imperial"}<｜tool▁call▁end｜><｜tool▁call▁begin｜>get_air_quality宽带}{location": "Berlin", "radius": 50}<｜tool▁call▁end｜><｜tool▁calls▁end｜>"#;
        let config = JsonParserConfig {
            tool_call_start_tokens: vec!["<｜tool▁calls▁begin｜>".to_string()],
            tool_call_end_tokens: vec!["<｜tool▁calls▁end｜>".to_string()],
            ..Default::default()
        };
        let (result, content) = parse_tool_calls_deepseek_v3_1(text, &config).unwrap();
        assert_eq!(content, Some(text.trim().to_string()));
        assert_eq!(result.len(), 0);
    }
}
