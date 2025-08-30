// SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::collections::HashMap;

use regex::RegexBuilder;
use serde_json::Value;
use uuid::Uuid;

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
        // Only consider segments that start like JSON
        if s.starts_with('{') {
            // Trim trailing non-JSON by cutting at the last closing brace/bracket
            if let Some(pos) = s.rfind('}') {
                let candidate = &s[..=pos].trim();
                // Keep only valid JSON candidates
                if serde_json::from_str::<serde_json::Value>(candidate).is_ok() {
                    items.push(candidate.to_string());
                }
            }
        }
    }
    if items.is_empty() {
        // Remove everything up to and including the first occurrence of the start token
        if let Some(idx) = input.find(start_token) {
            let rest = &input[idx + start_token.len()..];
            return Some(rest.trim_start().to_string());
        } else {
            // Shouldn't happen because we checked contains() above, but be defensive
            return None;
        }
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
pub fn try_tool_call_parse_json(
    message: &str,
    config: &JsonParserConfig,
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
        for (start_token, end_token) in tool_call_start_tokens
            .iter()
            .zip(tool_call_end_tokens.iter())
        {
            let new_normal_text = try_parse_normal_text(&normal_text, start_token);

            // Process based on token types
            match (start_token.is_empty(), end_token.is_empty()) {
                (false, true) => {
                    // Single token case
                    let result = handle_single_token_tool_calls(&json, start_token);
                    if let Some(content) = result {
                        json = content;
                        // For single token case, use the normal text we extracted earlier
                        normal_text = new_normal_text;
                        break; // Found content, exit early
                    }
                }
                (false, false) => {
                    // Start and end token case
                    let result = extract_tool_call_content(&json, start_token, end_token);
                    if let Some(content) = result {
                        json = content;
                        normal_text = new_normal_text;
                        break; // Found content, exit early
                    }
                }
                _ => {
                    continue;
                }
            }
        }
    }
    // Convert json (String) to &str
    let json = json.as_str();
    // Anonymous function to attempt deserialization into a known representation
    let parse = |name: String, args: HashMap<String, Value>| -> anyhow::Result<_> {
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

    // Vec<CalledFunctionParameters>: List of { name, parameters }
    // Example:
    // [
    //   { "name": "lookup_user", "parameters": { "user_id": "123" } },
    //   { "name": "send_email", "parameters": { "to": "user@example.com", "subject": "Welcome!" } }
    // ]
    // We pop the last item in the list to use.
    } else if let Ok(list) = serde_json::from_str::<Vec<CalledFunctionParameters>>(json) {
        let mut results = Vec::new();
        for item in list {
            results.push(parse(item.name, item.parameters)?);
        }
        return Ok((results, Some(normal_text)));

    // Vec<CalledFunctionArguments>: List of { name, arguments }
    // Example:
    // [
    //   {
    //     "name": "get_weather",
    //     "arguments": {
    //       "location": "San Francisco",
    //       "units": "celsius"
    //     }
    //   }
    // ]
    // Again, we take the last item for processing.
    } else if let Ok(list) = serde_json::from_str::<Vec<CalledFunctionArguments>>(json) {
        let mut results = Vec::new();
        for item in list {
            results.push(parse(item.name, item.arguments)?);
        }
        return Ok((results, Some(normal_text)));
    }

    Ok((vec![], Some(trimmed.to_string())))
}
