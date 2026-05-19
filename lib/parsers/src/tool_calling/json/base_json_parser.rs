// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use regex::RegexBuilder;
use serde_json::value::RawValue;
use uuid::Uuid;

use super::super::ToolDefinition;
use super::config::JsonParserConfig;
use super::response::{CalledFunction, ToolCallResponse, ToolCallType};

// Same as CalledFunction with named parameters.
//
// `parameters` / `arguments` are deserialized as `Box<RawValue>` so the
// original byte span — including key order, whitespace, and number
// formatting — is preserved verbatim. Going through `HashMap<String, Value>`
// here would normalize via `serde_json::to_string`, which strips spaces and
// reorders keys based on HashMap iteration. That broke append-only KV-cache
// prefix matching: when the model's emitted `arguments` were re-rendered
// through this parser and round-tripped back into the next turn's prompt,
// the bytes diverged from what the model originally emitted.
#[derive(Debug, serde::Deserialize)]
pub struct CalledFunctionParameters {
    pub name: String,
    pub parameters: Box<RawValue>,
}

#[derive(Debug, serde::Deserialize)]
pub struct CalledFunctionArguments {
    pub name: String,
    pub arguments: Box<RawValue>,
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

/// EOF-as-end-token recovery — finalize-only path. Returns the JSON-looking
/// tail after `start_token` when the outer end-token never arrived. Gated on
/// `JsonParserConfig::allow_eof_recovery` so streaming early-exit doesn't
/// fire mid-stream before the end-token has shown up.
fn extract_tool_call_content_eof_recovery(input: &str, start_token: &str) -> Option<String> {
    let start_pos = input.find(start_token)?;
    let tail = input[start_pos + start_token.len()..].trim();
    if tail.starts_with('{') || tail.starts_with('[') {
        Some(tail.to_string())
    } else {
        None
    }
}

// Special case for <|python_tag|> . Regex pattern does not work well with it as it has no end token
// Handles single tool and multiple tool call cases for single start_token like <|python_tag|>
fn handle_single_token_tool_calls(input: &str, start_token: &str) -> Option<String> {
    // Return the input if it doesn't contain the start token
    if !input.contains(start_token) {
        return None;
    }

    // Split on the start token and collect valid JSON objects/arrays.
    let mut items: Vec<String> = Vec::new();
    for seg in input.split(start_token) {
        let s = seg.trim();
        if s.is_empty() {
            continue;
        }
        if s.starts_with('{') {
            // Stream consecutive JSON objects from the segment, skipping ';'
            // separators between them.  This correctly handles both:
            //   • a single call whose argument contains ';' — the streaming
            //     deserializer parses the whole object in one shot without
            //     ever looking at the internal semicolon.
            //   • parallel calls separated by ';' where one argument also
            //     contains a ';' inside a string — the deserializer tracks
            //     string/depth context so byte_offset() lands exactly after
            //     the closing '}' of each complete object.
            let mut remaining = s.trim_start();
            while !remaining.is_empty() {
                // Use StreamDeserializer (.into_iter().next()) rather than
                // from_str so the parse succeeds even when there is trailing
                // non-JSON text after the closing '}' — e.g.
                //   {"name":"q","arguments":{}} Let me know if you need more
                // from_str would Err on the trailing text; StreamDeserializer
                // reads one value and stops.
                let mut stream =
                    serde_json::Deserializer::from_str(remaining).into_iter::<Box<RawValue>>();
                match stream.next() {
                    Some(Ok(rv)) => {
                        let raw = rv.get();
                        if raw.is_empty() {
                            break; // defensive: zero-advance guard
                        }
                        items.push(raw.to_string());
                        // Advance past the consumed bytes.  `RawValue` captures
                        // exactly the JSON token bytes (no surrounding whitespace),
                        // and `remaining` starts at a non-whitespace byte because
                        // we called `trim_start()` at every step.
                        remaining = remaining[raw.len()..].trim_start();
                        // Skip the ';' separator between parallel calls (if any).
                        if let Some(rest) = remaining.strip_prefix(';') {
                            remaining = rest.trim_start();
                        } else {
                            break; // no separator → only one object or done
                        }
                    }
                    _ => break, // None (end of input) or Some(Err(_)) (malformed)
                }
            }
        } else if s.starts_with('[') {
            // Array format used by phi4 (functools[{...}]) and similar models.
            // Parse as Vec<Box<RawValue>> to preserve each element's original byte
            // span — serde_json::Value + to_string would reorder keys and strip
            // whitespace, breaking append-only KV-cache prefix matching.
            if let Some(pos) = s.rfind(']') {
                let candidate = &s[..=pos].trim();
                if let Ok(arr) = serde_json::from_str::<Vec<Box<RawValue>>>(candidate) {
                    for item in arr {
                        items.push(item.get().to_string());
                    }
                }
            }
        }
        // Segments that start with neither '{' nor '[' are silently dropped.
        // Note: a separate symptom of issue #8732 is that the model occasionally
        // echoes back unfilled response-template text (e.g. "WinRM: [status]")
        // after the start token instead of a tool call. That is a model-side
        // behaviour (likely caused by an incorrect system prompt) and is tracked
        // separately; it is not addressed by this parser change.
    }
    if items.is_empty() {
        // Start token was found but no valid JSON followed it — return empty to
        // avoid leaking the start token or invalid content into normal_text.
        return Some(String::new());
    }
    Some(format!("[{}]", items.join(",")))
}

/// Attempt to repair JSON truncated by max_tokens / EOS. Walks the input
/// tracking string state and brace/bracket nesting; on EOF closes any
/// open string and pops outstanding closers. Returns `Some(repaired)` only
/// when at least one closer needed to be appended (so we don't churn
/// already-valid JSON).
pub(crate) fn try_repair_truncated_json(s: &str) -> Option<String> {
    let mut stack: Vec<char> = Vec::new();
    let mut in_string = false;
    let mut escape = false;
    for c in s.chars() {
        if escape {
            escape = false;
            continue;
        }
        if in_string {
            match c {
                '\\' => escape = true,
                '"' => in_string = false,
                _ => {}
            }
            continue;
        }
        match c {
            '"' => in_string = true,
            '{' => stack.push('}'),
            '[' => stack.push(']'),
            '}' | ']' => {
                stack.pop();
            }
            _ => {}
        }
    }
    if !escape && !in_string && stack.is_empty() {
        return None;
    }
    let mut repaired = s.to_string();
    // EOF mid-escape sequence: pair the trailing `\` with another `\` so the
    // closing quote we append next isn't itself escaped.
    if escape {
        repaired.push('\\');
    }
    if in_string {
        repaired.push('"');
    }
    while let Some(closer) = stack.pop() {
        repaired.push(closer);
    }
    Some(repaired)
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

    // Early exit if no tokens configured (unless bare_json_mode forces the
    // no-marker extraction path).
    if tool_call_start_tokens.is_empty() && !config.bare_json_mode {
        return Ok((vec![], Some(trimmed.to_string())));
    }

    // Iterate over all start and end tokens and try to extract the content between them
    // Assumption : One message will not contain different tags for tool calls. Iteration over tags is to support different tags by default for multiple models
    let mut json = trimmed.to_string();
    let mut normal_text = trimmed.to_string();
    let mut found_start_token_with_no_valid_json = false;

    // First, check if ANY start token exists in the input. `bare_json_mode`
    // short-circuits this to false so we always take the no-marker branch.
    let has_start_token = !config.bare_json_mode
        && tool_call_start_tokens
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
                            // handle_single_token_tool_calls returns either:
                            //   Some("[{...}, ...]") — one or more extracted calls
                            //   Some("")             — start token found, no valid JSON followed
                            // Only the "[..." form means extraction succeeded. Anything else
                            // means the start token was present but produced no calls; set the
                            // flag so the caller returns "" rather than leaking the start token
                            // or the raw invalid content into normal_text.
                            if !content.starts_with('[') {
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
                        let mut result = extract_tool_call_content(&json, start_token, end_token);
                        // EOF recovery: only when explicitly opted in (finalize
                        // path). Streaming jails leave `allow_eof_recovery=false`
                        // so the parser doesn't claim a complete call before
                        // the end-token has actually arrived.
                        if result.is_none()
                            && config.allow_eof_recovery
                            && json.contains(start_token.as_str())
                        {
                            result = extract_tool_call_content_eof_recovery(&json, start_token);
                        }
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
    // Anonymous function to attempt deserialization into a known representation.
    //
    // We pass through the original byte span via `RawValue::get()` rather than
    // re-serializing a parsed `HashMap` / `Value`. That keeps `arguments`
    // byte-identical to what the model emitted, which is required for KV-cache
    // append-only prefix matching across multi-step tool use.
    let parse = |name: String, args: &RawValue| -> anyhow::Result<ToolCallResponse> {
        Ok(ToolCallResponse {
            id: format!("call-{}", Uuid::new_v4()),
            tp: ToolCallType::Function,
            function: CalledFunction {
                name,
                arguments: args.get().to_string(),
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
            vec![parse(single.name, &single.parameters)?],
            Some(normal_text),
        ));

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
            vec![parse(single.name, &single.arguments)?],
            Some(normal_text),
        ));

    // Vec<CalledFunctionParameters> or Vec<CalledFunctionArguments>: Array of tool calls
    // Example:
    // [
    //   { "name": "lookup_user", "parameters": { "user_id": "123" } },
    //   { "name": "get_weather", "arguments": { "location": "SF", "units": "celsius" } }
    // ]
    // Parse the array as `Vec<Box<RawValue>>` so each element retains its
    // original byte span. Going through `Vec<serde_json::Value>` would
    // already have mangled the inner `arguments` / `parameters` bytes by the
    // time we tried to extract them.
    } else if let Ok(array) = serde_json::from_str::<Vec<Box<RawValue>>>(json) {
        let mut results = Vec::new();
        for item in array {
            let item_str = item.get();
            // Try both CalledFunctionArguments and CalledFunctionParameters formats
            if let Ok(func_args) = serde_json::from_str::<CalledFunctionArguments>(item_str) {
                results.push(parse(func_args.name, &func_args.arguments)?);
            } else if let Ok(func_params) =
                serde_json::from_str::<CalledFunctionParameters>(item_str)
            {
                results.push(parse(func_params.name, &func_params.parameters)?);
            }
            // Skip malformed entries silently
        }
        // Return with whatever results we have, even if empty (e.g., [] is a valid empty array)
        return Ok((results, Some(normal_text)));
    }

    // Truncation recovery: balance unclosed strings/braces (common
    // max_tokens / EOS pattern) and retry the same three parses. Gated on
    // `allow_eof_recovery` so streaming jails don't claim a complete tool
    // call while the model is still emitting JSON tokens.
    if config.allow_eof_recovery
        && let Some(repaired) = try_repair_truncated_json(json)
    {
        let repaired = repaired.as_str();
        if let Ok(single) = serde_json::from_str::<CalledFunctionParameters>(repaired) {
            return Ok((
                vec![parse(single.name, &single.parameters)?],
                Some(normal_text),
            ));
        } else if let Ok(single) = serde_json::from_str::<CalledFunctionArguments>(repaired) {
            return Ok((
                vec![parse(single.name, &single.arguments)?],
                Some(normal_text),
            ));
        } else if let Ok(array) = serde_json::from_str::<Vec<Box<RawValue>>>(repaired) {
            let mut results = Vec::new();
            for item in array {
                let item_str = item.get();
                if let Ok(func_args) = serde_json::from_str::<CalledFunctionArguments>(item_str) {
                    results.push(parse(func_args.name, &func_args.arguments)?);
                } else if let Ok(func_params) =
                    serde_json::from_str::<CalledFunctionParameters>(item_str)
                {
                    results.push(parse(func_params.name, &func_params.parameters)?);
                }
            }
            if !results.is_empty() {
                return Ok((results, Some(normal_text)));
            }
        }
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
mod repair_tests {
    use super::*;

    // EOF inside an escape sequence (`{"k":"a\` → `{"k":"a\\"}`). Without
    // the `escape` guard, the appended `"` would itself be escaped and the
    // resulting JSON would still be invalid.
    #[test]
    fn test_repair_eof_after_backslash() {
        let repaired = try_repair_truncated_json(r#"{"k":"a\"#).expect("must repair");
        assert!(
            serde_json::from_str::<serde_json::Value>(&repaired).is_ok(),
            "repaired must parse: {:?}",
            repaired
        );
    }
}

#[cfg(test)]
mod detect_parser_tests {
    use super::*;

    #[test] // helper
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

    #[test] // helper
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

    #[test] // helper, PARSER.batch.8
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

    #[test] // helper
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

    #[test] // helper
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

    #[test] // helper
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

    #[test] // helper
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

    #[test] // helper
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

    #[test] // helper
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

    #[test] // helper, PARSER.stream.3
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

    #[test] // helper, PARSER.stream.3
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

    #[test] // helper, PARSER.stream.3
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

    #[test] // helper, PARSER.stream.3
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

    #[test] // helper
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

    #[test] // helper
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
