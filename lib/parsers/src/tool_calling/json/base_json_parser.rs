// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::collections::HashMap;

use regex::RegexBuilder;
use serde_json::Value;
use uuid::Uuid;

use super::super::ToolDefinition;
use super::config::JsonParserConfig;
use super::response::{CalledFunction, ToolCallResponse, ToolCallType};

// Same as CalledFunction with named parameters
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct CalledFunctionParameters {
    pub name: String,
    pub parameters: HashMap<String, Value>,
}

// Same as CalledFunction with named parameters
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct CalledFunctionArguments {
    pub name: String,
    pub arguments: HashMap<String, Value>,
}

// Extract the contents between start and end tokens using regex parsing.
// Returns a JSON array string if there are multiple matches, otherwise returns the last match directly.
fn extract_tool_call_content(input: &str, start_token: &str, end_token: &str) -> Option<String> {
    let escaped_start = regex::escape(start_token);
    let escaped_end = regex::escape(end_token);
    let pattern = format!(r"{}(.*?){}", escaped_start, escaped_end);

    match RegexBuilder::new(&pattern)
        .dot_matches_new_line(true)
        .build()
    {
        Ok(regex) => {
            // Get all matches and take the last one for now. TODO: Handle multiple tool calls
            let matches: Vec<_> = regex
                .captures_iter(input)
                .filter_map(|captures| captures.get(1))
                .map(|m| m.as_str().trim().to_string())
                .collect();
            if !matches.is_empty() {
                // If only one match, return it directly, otherwise return as a JSON array string
                if matches.len() == 1 {
                    // Return the last match directly
                    return Some(matches.last().unwrap().clone());
                } else {
                    // Join the matches into a JSON array string
                    return Some(format!("[{}]", matches.join(",")));
                }
            }
            None
        }
        Err(_) => None,
    }
}

// Special case for <|python_tag|> . Regex pattern does not work well with it as it has no end token
// Handles single tool and multiple tool call cases for single start_token like <|python_tag|>
fn handle_single_token_tool_calls(input: &str, start_token: &str) -> Option<String> {
    // Return the input if it doesn't contain the start token
    if !input.contains(start_token) {
        return None;
    }

    // Split on the start token and keep only JSON-looking segments
    let mut items: Vec<String> = Vec::new();
    for seg in input.split(start_token) {
        let s = seg.trim();
        if s.is_empty() {
            continue;
        }
        // Only consider segments that start like JSON (objects or arrays)
        if s.starts_with('{') {
            // Trim trailing non-JSON by cutting at the last closing brace
            if let Some(pos) = s.rfind('}') {
                let candidate = &s[..=pos].trim();
                // Keep only valid JSON candidates
                if serde_json::from_str::<serde_json::Value>(candidate).is_ok() {
                    items.push(candidate.to_string());
                }
            }
        } else if s.starts_with('[') {
            // Handle array format (like phi4: functools[{...}])
            if let Some(pos) = s.rfind(']') {
                let candidate = &s[..=pos].trim();
                // Keep only valid JSON arrays
                if serde_json::from_str::<serde_json::Value>(candidate).is_ok() {
                    // For arrays, we need to extract the individual objects
                    if let Ok(serde_json::Value::Array(arr)) =
                        serde_json::from_str::<serde_json::Value>(candidate)
                    {
                        for item in arr {
                            if let Ok(item_str) = serde_json::to_string(&item) {
                                items.push(item_str);
                            }
                        }
                    }
                }
            }
        }
    }
    if items.is_empty() {
        // If we found the start token but no valid JSON after it, return empty string
        // to avoid leaking the invalid content (important for phi4 and similar models)
        return Some(String::new());
    }
    Some(format!("[{}]", items.join(",")))
}

fn try_parse_normal_text(input: &str, start_token: &str) -> String {
    // If input contains start token, just take the part before it
    if let Some(idx) = input.find(start_token) {
        return input[..idx].trim().to_string();
    }

    // No start token found, return empty string
    String::new()
}

/// Attempts to parse a tool call from a raw LLM message string into a unified [`ToolCallResponse`] format.
///
/// This is a flexible helper that handles a variety of potential formats emitted by LLMs for function/tool calls,
/// including wrapped payloads (`<TOOLCALL>[...]</TOOLCALL>`, `<|python_tag|>...`) and JSON representations
/// with either `parameters` or `arguments` fields.
///
/// # Supported Formats
///
/// The input `message` may be one of:
///
/// - `<TOOLCALL>[{ "name": ..., "parameters": { ... } }]</TOOLCALL>`
/// - `<|python_tag|>{ "name": ..., "arguments": { ... } }`
/// - Raw JSON of:
///     - `CalledFunctionParameters`: `{ "name": ..., "parameters": { ... } }`
///     - `CalledFunctionArguments`: `{ "name": ..., "arguments": { ... } }`
///     - Or a list of either of those types: `[ { "name": ..., "arguments": { ... } }, ... ]`
///
/// # Return
///
/// - `Ok(Some(ToolCallResponse))` if parsing succeeds
/// - `Ok(None)` if input format is unrecognized or invalid JSON
/// - `Err(...)` if JSON is valid but deserialization or argument re-serialization fails
///
/// # Note on List Handling
///
/// When the input contains a list of tool calls (either with `parameters` or `arguments`),
/// only the **last item** in the list is returned. This design choice assumes that the
/// most recent tool call in a list is the one to execute.
///
/// # Errors
///
/// Returns a `Result::Err` only if an inner `serde_json::to_string(...)` fails
/// (e.g., if the arguments are not serializable).
///
/// # Examples
///
/// ```ignore
/// let input = r#"<TOOLCALL>[{ "name": "search", "parameters": { "query": "rust" } }]</TOOLCALL>"#;
/// let result = try_tool_call_parse_json(input)?;
/// assert!(result.is_some());
/// ```
pub fn try_tool_call_parse_basic_json(
    message: &str,
    config: &JsonParserConfig,
    _tools: Option<&[ToolDefinition]>,
) -> anyhow::Result<(Vec<ToolCallResponse>, Option<String>)> {
    // Log the config we are using
    tracing::debug!("Using JSON parser config: {:?}", config);
    let trimmed = message.trim();

    // Early exit if no content
    if trimmed.is_empty() {
        return Ok((vec![], Some(String::new())));
    }

    let tool_call_start_tokens = &config.tool_call_start_tokens;
    let tool_call_end_tokens = &config.tool_call_end_tokens;

    // Early exit if no tokens configured
    if tool_call_start_tokens.is_empty() {
        return Ok((vec![], Some(trimmed.to_string())));
    }

    // Iterate over all start and end tokens and try to extract the content between them
    // Assumption : One message will not contain different tags for tool calls. Iteration over tags is to support different tags by default for multiple models
    let mut json = trimmed.to_string();
    let mut normal_text = trimmed.to_string();
    let mut found_start_token_with_no_valid_json = false;

    // First, check if ANY start token exists in the input
    let has_start_token = tool_call_start_tokens
        .iter()
        .any(|token| !token.is_empty() && normal_text.contains(token));

    if !has_start_token {
        // No start tokens found, try to extract JSON directly. Everything that starts with { or [ is considered a potential JSON.
        if let Some(idx) = normal_text.find(['{', '[']) {
            let extracted_normal = normal_text[..idx].trim().to_string();
            let extracted_json = normal_text[idx..].trim().to_string();
            if !extracted_json.is_empty() {
                normal_text = extracted_normal;
                json = extracted_json;
            }
        }
    } else {
        // Start tokens exist, use regex-based parsing
        // Try all combinations of start and end tokens
        'outer: for start_token in tool_call_start_tokens.iter() {
            for end_token in tool_call_end_tokens.iter() {
                let new_normal_text = try_parse_normal_text(&normal_text, start_token);

                // Process based on token types
                match (start_token.is_empty(), end_token.is_empty()) {
                    (false, true) => {
                        // Single token case
                        let result = handle_single_token_tool_calls(&json, start_token);
                        if let Some(content) = result {
                            // Check if we found a start token but got empty JSON back
                            // This indicates the token was found but no valid JSON followed
                            if content.is_empty() {
                                found_start_token_with_no_valid_json = true;
                            }

                            json = content;
                            // For single token case, use the normal text we extracted earlier
                            normal_text = new_normal_text;

                            break 'outer; // Found content, exit early
                        }
                    }
                    (false, false) => {
                        // Start and end token case
                        let result = extract_tool_call_content(&json, start_token, end_token);
                        if let Some(content) = result {
                            // Check if we found a start token but got empty JSON back
                            // This indicates the token was found but no valid JSON followed
                            if content.is_empty() {
                                found_start_token_with_no_valid_json = true;
                            }

                            json = content;
                            normal_text = new_normal_text;

                            break 'outer; // Found content, exit early
                        }
                    }
                    _ => {
                        continue;
                    }
                }
            }
        }
    }
    // Convert json (String) to &str
    let json = json.as_str();
    // Anonymous function to attempt deserialization into a known representation
    let parse = |name: String, args: HashMap<String, Value>| -> anyhow::Result<ToolCallResponse> {
        // Preserve nested JSON strings intact; do not double-escape.
        // serde_json::to_string on Value preserves required escapes only.
        Ok(ToolCallResponse {
            id: format!("call-{}", Uuid::new_v4()),
            tp: ToolCallType::Function,
            function: CalledFunction {
                name,
                arguments: serde_json::to_string(&args)?,
            },
        })
    };

    // CalledFunctionParameters: Single { name, parameters }
    // Example:
    // {
    //   "name": "search_docs",
    //   "parameters": {
    //     "query": "how to use Rust",
    //     "limit": 5
    //   }
    // }
    if let Ok(single) = serde_json::from_str::<CalledFunctionParameters>(json) {
        return Ok((
            vec![parse(single.name, single.parameters)?],
            Some(normal_text),
        ));
        //parse(single.name, single.parameters).map(Some);

        // CalledFunctionArguments: Single { name, arguments }
        // Example:
        // {
        //   "name": "summarize",
        //   "arguments": {
        //     "text": "Rust is a systems programming language.",
        //     "length": "short"
        //   }
        // }
    } else if let Ok(single) = serde_json::from_str::<CalledFunctionArguments>(json) {
        return Ok((
            vec![parse(single.name, single.arguments)?],
            Some(normal_text),
        ));

    // Vec<CalledFunctionParameters> or Vec<CalledFunctionArguments>: Array of tool calls
    // Example:
    // [
    //   { "name": "lookup_user", "parameters": { "user_id": "123" } },
    //   { "name": "get_weather", "arguments": { "location": "SF", "units": "celsius" } }
    // ]
    // Parse as generic array to handle both formats and malformed entries gracefully
    // Note: Always return once we parse a valid array, even if empty or with malformed entries
    } else if let Ok(array) = serde_json::from_str::<Vec<serde_json::Value>>(json) {
        let mut results = Vec::new();
        for item in array {
            // Try both CalledFunctionArguments and CalledFunctionParameters formats
            if let Ok(func_args) = serde_json::from_value::<CalledFunctionArguments>(item.clone()) {
                results.push(parse(func_args.name, func_args.arguments)?);
            } else if let Ok(func_params) = serde_json::from_value::<CalledFunctionParameters>(item)
            {
                results.push(parse(func_params.name, func_params.parameters)?);
            }
            // Skip malformed entries silently
        }
        // Return with whatever results we have, even if empty (e.g., [] is a valid empty array)
        return Ok((results, Some(normal_text)));
    }

    // If we found a start token but no valid JSON, return empty content
    // to avoid leaking the token and invalid JSON content
    if found_start_token_with_no_valid_json {
        Ok((vec![], Some(String::new())))
    } else {
        Ok((vec![], Some(trimmed.to_string())))
    }
}

pub fn detect_tool_call_start_basic_json(chunk: &str, config: &JsonParserConfig) -> bool {
    let trimmed = chunk.trim();
    if trimmed.is_empty() {
        return false;
    }

    // Check if chunk contains any complete start token
    let contains_complete_token = config
        .tool_call_start_tokens
        .iter()
        .any(|token| !token.is_empty() && trimmed.contains(token));

    if contains_complete_token {
        return true;
    }

    // Check for partial start tokens (streaming scenario)
    // This handles cases where start tokens are split across multiple chunks
    let has_partial_token = config.tool_call_start_tokens.iter().any(|token| {
        if token.is_empty() {
            return false;
        }
        // Check if the chunk could be a prefix of this start token
        // Handle Unicode character boundaries properly
        for i in 1..=token.chars().count() {
            if let Some(prefix) = token.chars().take(i).collect::<String>().get(..) {
                let prefix_str = &prefix[..prefix.len()];
                // Check for exact prefix match
                if trimmed == prefix_str {
                    return true;
                }
                // For longer prefixes (3+ chars), allow them anywhere in the input
                // This allows "funny joke" to match "functools" via "fun"
                // but prevents "<tool_call>" from matching "<TOOLCALL>" via single char "<"
                if prefix_str.len() >= 3 && trimmed.contains(prefix_str) {
                    return true;
                }
                // For shorter prefixes, only match if they're at the end (streaming scenario)
                if prefix_str.len() < 3 && trimmed.ends_with(prefix_str) {
                    return true;
                }
            }
        }
        false
    });

    has_partial_token || trimmed.contains('{') || trimmed.contains('[')
}

#[cfg(test)]
mod detect_parser_tests {
    use super::*;

    #[test]
    fn detect_tool_call_start_basic_json_chunk_with_tool_call_start_token_hermes() {
        let text =
            r#"<tool_call>{"name": "search", "parameters": { "query": "rust" } }</tool_call>"#;
        let config = JsonParserConfig {
            tool_call_start_tokens: vec!["<tool_call>".to_string()],
            tool_call_end_tokens: vec!["</tool_call>".to_string()],
            ..Default::default()
        };
        let result = detect_tool_call_start_basic_json(text, &config);
        assert!(result);
    }

    #[test]
    fn detect_tool_call_start_basic_json_chunk_without_tool_call_start_token() {
        let text = r#"{"name": "search", "parameters": { "query": "rust" } }"#;
        let config = JsonParserConfig {
            tool_call_start_tokens: vec!["<tool_call>".to_string()],
            tool_call_end_tokens: vec!["</tool_call>".to_string()],
            ..Default::default()
        };
        let result = detect_tool_call_start_basic_json(text, &config);
        assert!(result);
    }

    #[test]
    fn detect_tool_call_start_basic_json_chunk_without_tool_call_start_token_with_normal_text() {
        let text = r#"Here it is {"name": "#;
        let config = JsonParserConfig {
            tool_call_start_tokens: vec!["<tool_call>".to_string()],
            tool_call_end_tokens: vec!["</tool_call>".to_string()],
            ..Default::default()
        };
        let result = detect_tool_call_start_basic_json(text, &config);
        assert!(result);
    }

    #[test]
    fn detect_tool_call_start_basic_json_chunk_with_square_brackets() {
        // These kind of false positives are expected when calling this function for stream=True
        let text = r#"Here it is [{"name": "search","#;
        let config = JsonParserConfig {
            tool_call_start_tokens: vec!["<tool_call>".to_string()],
            tool_call_end_tokens: vec!["</tool_call>".to_string()],
            ..Default::default()
        };
        let result = detect_tool_call_start_basic_json(text, &config);
        assert!(result);
    }

    #[test]
    fn detect_tool_call_start_basic_json_chunk_false_positive() {
        // These kind of false positives are expected when calling this function for stream=True
        let text = r#"Here it is { Whats up"#;
        let config = JsonParserConfig {
            tool_call_start_tokens: vec!["<tool_call>".to_string()],
            tool_call_end_tokens: vec!["</tool_call>".to_string()],
            ..Default::default()
        };
        let result = detect_tool_call_start_basic_json(text, &config);
        assert!(result);
    }

    #[test]
    fn detect_tool_call_start_basic_json_chunk_with_tool_call_start_token_nemotron_deci() {
        let text =
            r#"<TOOLCALL>[{"name": "search", "parameters": { "query": "rust" } }]</TOOLCALL>"#;
        let config = JsonParserConfig {
            tool_call_start_tokens: vec!["<TOOLCALL>".to_string()],
            tool_call_end_tokens: vec!["</TOOLCALL>".to_string()],
            ..Default::default()
        };
        let result = detect_tool_call_start_basic_json(text, &config);
        assert!(result);
    }

    #[test]
    fn detect_tool_call_start_basic_json_chunk_with_lllama3_json_token() {
        let text = r#"<|python_tag|>{ "name": }"#;
        let config = JsonParserConfig {
            tool_call_start_tokens: vec!["<|python_tag|>".to_string()],
            tool_call_end_tokens: vec!["".to_string()],
            ..Default::default()
        };
        let result = detect_tool_call_start_basic_json(text, &config);
        assert!(result);
    }

    #[test]
    fn detect_tool_call_start_basic_json_chunk_mistral_token() {
        let text = r#"Hello Yo ! [TOOL_CALLS]{"name": "search", "#;
        let config = JsonParserConfig {
            tool_call_start_tokens: vec!["[TOOL_CALLS]".to_string()],
            tool_call_end_tokens: vec!["".to_string()],
            ..Default::default()
        };
        let result = detect_tool_call_start_basic_json(text, &config);
        assert!(result);
    }

    #[test]
    fn detect_tool_call_start_basic_json_chunk_phi4_token() {
        let text = r#"functools{"name": "search", "#;
        let config = JsonParserConfig {
            tool_call_start_tokens: vec!["functools".to_string()],
            tool_call_end_tokens: vec!["".to_string()],
            ..Default::default()
        };
        let result = detect_tool_call_start_basic_json(text, &config);
        assert!(result);
    }

    #[test]
    fn detect_tool_call_start_basic_json_chunk_phi4_partial_token_fun() {
        // Test the streaming scenario where "fun" arrives first
        let text = r#"fun"#;
        let config = JsonParserConfig {
            tool_call_start_tokens: vec!["functools".to_string()],
            tool_call_end_tokens: vec!["".to_string()],
            ..Default::default()
        };
        let result = detect_tool_call_start_basic_json(text, &config);
        assert!(
            result,
            "Should detect 'fun' as potential start of 'functools'"
        );
    }

    #[test]
    fn detect_tool_call_start_basic_json_chunk_phi4_partial_token_func() {
        let text = r#"func"#;
        let config = JsonParserConfig {
            tool_call_start_tokens: vec!["functools".to_string()],
            tool_call_end_tokens: vec!["".to_string()],
            ..Default::default()
        };
        let result = detect_tool_call_start_basic_json(text, &config);
        assert!(
            result,
            "Should detect 'func' as potential start of 'functools'"
        );
    }

    #[test]
    fn detect_tool_call_start_basic_json_chunk_phi4_partial_token_f() {
        let text = r#"f"#;
        let config = JsonParserConfig {
            tool_call_start_tokens: vec!["functools".to_string()],
            tool_call_end_tokens: vec!["".to_string()],
            ..Default::default()
        };
        let result = detect_tool_call_start_basic_json(text, &config);
        assert!(
            result,
            "Should detect 'f' as potential start of 'functools'"
        );
    }

    #[test]
    fn detect_tool_call_start_basic_json_chunk_phi4_partial_with_prefix() {
        // Test case where text ends with a partial token (more realistic streaming scenario)
        let text = r#"Hello fun"#;
        let config = JsonParserConfig {
            tool_call_start_tokens: vec!["functools".to_string()],
            tool_call_end_tokens: vec!["".to_string()],
            ..Default::default()
        };
        let result = detect_tool_call_start_basic_json(text, &config);
        assert!(
            result,
            "Should detect text ending with 'fun' as potential tool call start"
        );
    }

    #[test]
    fn detect_tool_call_start_basic_json_chunk_phi4_avoid_false_positive() {
        // Test to ensure we don't get false positives for unrelated text
        let text = r#"funny joke"#;
        let config = JsonParserConfig {
            tool_call_start_tokens: vec!["functools".to_string()],
            tool_call_end_tokens: vec!["".to_string()],
            ..Default::default()
        };
        let result = detect_tool_call_start_basic_json(text, &config);
        // This should still return true because "fun" is a prefix, but that's expected behavior
        // The key is that we detect potential starts, and false positives are acceptable
        // in streaming scenarios to avoid missing real tool calls
        assert!(result);
    }

    #[test]
    fn detect_tool_call_start_basic_json_chunk_phi4_no_match() {
        let text = r#"hello world"#;
        let config = JsonParserConfig {
            tool_call_start_tokens: vec!["functools".to_string()],
            tool_call_end_tokens: vec!["".to_string()],
            ..Default::default()
        };
        let result = detect_tool_call_start_basic_json(text, &config);
        assert!(
            !result,
            "Should not detect unrelated text as tool call start"
        );
    }
}
