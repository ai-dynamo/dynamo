// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use regex::RegexBuilder;
use serde_json::{Value, value::RawValue};
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

#[derive(Debug, serde::Deserialize)]
pub struct CalledFunctionNameOnly {
    pub name: String,
}

fn tool_allows_empty_arguments(name: &str, tools: Option<&[ToolDefinition]>) -> bool {
    let Some(tools) = tools else {
        return false;
    };
    tools
        .iter()
        .any(|tool| tool.name == name && schema_allows_empty_arguments(tool.parameters.as_ref()))
}

fn schema_allows_empty_arguments(schema: Option<&Value>) -> bool {
    let Some(schema) = schema else {
        return true;
    };
    schema
        .get("required")
        .and_then(Value::as_array)
        .map(|required| required.is_empty())
        .unwrap_or(true)
}

fn contains_marker_token(text: &str, config: &JsonParserConfig) -> bool {
    config
        .tool_call_start_tokens
        .iter()
        .chain(config.tool_call_end_tokens.iter())
        .chain(config.tool_call_sentinel_tokens.iter())
        .any(|token| !token.is_empty() && text.contains(token))
}

fn strip_marker_tokens(input: &str, config: &JsonParserConfig) -> String {
    let mut stripped = input.to_string();
    for token in config
        .tool_call_start_tokens
        .iter()
        .chain(config.tool_call_end_tokens.iter())
        .chain(config.tool_call_sentinel_tokens.iter())
        .filter(|token| !token.is_empty())
    {
        stripped = stripped.replace(token, "");
    }
    stripped.trim().to_string()
}

fn strip_trailing_end_tokens(input: &str, end_tokens: &[String]) -> Option<String> {
    let mut trimmed = input.trim_end();
    let mut changed = false;

    loop {
        let mut matched = false;
        for token in end_tokens.iter().filter(|token| !token.is_empty()) {
            if let Some(prefix) = trimmed.strip_suffix(token.as_str()) {
                trimmed = prefix.trim_end();
                changed = true;
                matched = true;
                break;
            }
        }
        if !matched {
            break;
        }
    }

    if changed && (trimmed.starts_with('{') || trimmed.starts_with('[')) {
        Some(trimmed.to_string())
    } else {
        None
    }
}

// Legacy extraction for established JSON parser families. Keep this behavior
// stable unless a family opts into JSON-aware wrapper recovery.
fn extract_tool_call_content_regex(
    input: &str,
    start_token: &str,
    end_token: &str,
) -> Option<String> {
    let escaped_start = regex::escape(start_token);
    let escaped_end = regex::escape(end_token);
    let pattern = format!(r"{}(.*?){}", escaped_start, escaped_end);

    match RegexBuilder::new(&pattern)
        .dot_matches_new_line(true)
        .build()
    {
        Ok(regex) => {
            let matches: Vec<_> = regex
                .captures_iter(input)
                .filter_map(|captures| captures.get(1))
                .map(|m| m.as_str().trim().to_string())
                .collect();
            if matches.is_empty() {
                None
            } else if matches.len() == 1 {
                matches.last().cloned()
            } else {
                Some(format!("[{}]", matches.join(",")))
            }
        }
        Err(_) => None,
    }
}

// Extract complete JSON values after each start token. This deliberately uses a
// JSON parser instead of a `start(.*?)end` regex so marker-looking text inside a
// JSON string is treated as data, and so a malformed wrapper can be skipped
// before resynchronizing on a later valid wrapper.
fn extract_tool_call_content_with_recovery(
    input: &str,
    start_tokens: &[String],
    end_token: &str,
    allow_eof_recovery: bool,
) -> Option<String> {
    let mut cursor = 0;
    let mut values: Vec<String> = Vec::new();
    let mut saw_start = false;
    let mut saw_closed_wrapper = false;
    let mut saw_unclosed_wrapper = false;

    while let Some((start_pos, start_token)) = find_next_start_token(input, cursor, start_tokens) {
        saw_start = true;
        let body_pos = start_pos + start_token.len();
        let body = &input[body_pos..];
        let trimmed_body = body.trim_start();
        let leading_ws = body.len() - trimmed_body.len();
        let json_pos = body_pos + leading_ws;

        if !(trimmed_body.starts_with('{') || trimmed_body.starts_with('[')) {
            match next_wrapper_boundary(input, body_pos, start_tokens, end_token) {
                WrapperBoundary::End(pos) => {
                    tracing::warn!(
                        why = "non_json_wrapper_body",
                        skipped_bytes = pos.saturating_sub(body_pos),
                        "JSON tool-call recovery: skipped malformed marker wrapper whose body did not start with JSON."
                    );
                    saw_closed_wrapper = true;
                    cursor = pos + end_token.len();
                }
                WrapperBoundary::Start(pos) => {
                    tracing::warn!(
                        why = "non_json_wrapper_body_resync",
                        skipped_bytes = pos.saturating_sub(body_pos),
                        "JSON tool-call recovery: skipped malformed marker wrapper and resynchronized at the next start token."
                    );
                    saw_unclosed_wrapper = true;
                    cursor = pos;
                }
                WrapperBoundary::None => {
                    if allow_eof_recovery {
                        tracing::warn!(
                            why = "non_json_wrapper_body_no_boundary",
                            skipped_bytes = body.len(),
                            "JSON tool-call recovery: suppressed malformed marker wrapper with no later boundary."
                        );
                    }
                    saw_unclosed_wrapper = true;
                    break;
                }
            }
            continue;
        }

        let mut stream =
            serde_json::Deserializer::from_str(trimmed_body).into_iter::<Box<RawValue>>();
        match stream.next() {
            Some(Ok(raw)) => {
                let raw_json = raw.get();
                let after_raw = &trimmed_body[raw_json.len()..];
                let after_raw_trimmed = after_raw.trim_start();
                let raw_trailing_ws = after_raw.len() - after_raw_trimmed.len();
                if after_raw_trimmed.starts_with(end_token) {
                    values.push(raw_json.to_string());
                    cursor = json_pos + raw_json.len() + raw_trailing_ws + end_token.len();
                    saw_closed_wrapper = true;
                } else if allow_eof_recovery && after_raw_trimmed.trim().is_empty() {
                    tracing::warn!(
                        why = "missing_end_token",
                        recovered_bytes = raw_json.len(),
                        "JSON tool-call recovery: accepted complete JSON at EOF because end token was missing."
                    );
                    values.push(raw_json.to_string());
                    break;
                } else {
                    match next_wrapper_boundary(input, body_pos, start_tokens, end_token) {
                        WrapperBoundary::End(pos) => {
                            tracing::warn!(
                                why = "invalid_trailing_wrapper_bytes",
                                skipped_bytes = pos.saturating_sub(body_pos),
                                "JSON tool-call recovery: skipped marker wrapper with bytes after the JSON body before the end token."
                            );
                            saw_closed_wrapper = true;
                            cursor = pos + end_token.len();
                        }
                        WrapperBoundary::Start(pos) => {
                            tracing::warn!(
                                why = "invalid_trailing_wrapper_bytes_resync",
                                skipped_bytes = pos.saturating_sub(body_pos),
                                "JSON tool-call recovery: skipped marker wrapper with invalid trailing bytes and resynchronized at the next start token."
                            );
                            saw_unclosed_wrapper = true;
                            cursor = pos;
                        }
                        WrapperBoundary::None => {
                            if allow_eof_recovery {
                                tracing::warn!(
                                    why = "invalid_trailing_wrapper_bytes_no_boundary",
                                    skipped_bytes = body.len(),
                                    "JSON tool-call recovery: suppressed marker wrapper with invalid trailing bytes and no later boundary."
                                );
                            }
                            saw_unclosed_wrapper = true;
                            break;
                        }
                    }
                }
            }
            _ => {
                let boundary = next_wrapper_boundary(input, body_pos, start_tokens, end_token);
                if allow_eof_recovery {
                    let repair_candidate = match boundary {
                        WrapperBoundary::End(end_pos) => Some(input[json_pos..end_pos].trim()),
                        WrapperBoundary::None => Some(trimmed_body),
                        WrapperBoundary::Start(_) => None,
                    };
                    if let Some(candidate) = repair_candidate
                        && let Some(repaired) = try_repair_truncated_json(candidate)
                        && serde_json::from_str::<Box<RawValue>>(&repaired).is_ok()
                    {
                        tracing::warn!(
                            why = "truncated_json_repaired",
                            original_bytes = candidate.len(),
                            repaired_bytes = repaired.len(),
                            "JSON tool-call recovery: repaired truncated JSON before parsing tool call."
                        );
                        values.push(repaired);
                        if let WrapperBoundary::End(end_pos) = boundary {
                            saw_closed_wrapper = true;
                            cursor = end_pos + end_token.len();
                            continue;
                        }
                        break;
                    }
                }
                match boundary {
                    WrapperBoundary::End(pos) => {
                        tracing::warn!(
                            why = "malformed_json_wrapper",
                            skipped_bytes = pos.saturating_sub(body_pos),
                            "JSON tool-call recovery: skipped marker wrapper whose JSON body could not be parsed or repaired."
                        );
                        saw_closed_wrapper = true;
                        cursor = pos + end_token.len();
                    }
                    WrapperBoundary::Start(pos) => {
                        tracing::warn!(
                            why = "malformed_json_wrapper_resync",
                            skipped_bytes = pos.saturating_sub(body_pos),
                            "JSON tool-call recovery: skipped malformed JSON wrapper and resynchronized at the next start token."
                        );
                        saw_unclosed_wrapper = true;
                        cursor = pos;
                    }
                    WrapperBoundary::None => {
                        if allow_eof_recovery {
                            tracing::warn!(
                                why = "malformed_json_wrapper_no_boundary",
                                skipped_bytes = body.len(),
                                "JSON tool-call recovery: suppressed malformed JSON wrapper with no later boundary."
                            );
                        }
                        saw_unclosed_wrapper = true;
                        break;
                    }
                }
            }
        }
    }

    if values.len() == 1 {
        Some(values.remove(0))
    } else if !values.is_empty() {
        Some(format!("[{}]", values.join(",")))
    } else if saw_closed_wrapper || (saw_start && !saw_unclosed_wrapper) {
        Some(String::new())
    } else {
        None
    }
}

enum WrapperBoundary {
    End(usize),
    Start(usize),
    None,
}

fn find_next_start_token<'a>(
    input: &str,
    search_from: usize,
    start_tokens: &'a [String],
) -> Option<(usize, &'a str)> {
    start_tokens
        .iter()
        .filter(|token| !token.is_empty())
        .filter_map(|token| {
            input[search_from..]
                .find(token)
                .map(|pos| (search_from + pos, token.as_str()))
        })
        .min_by_key(|(pos, token)| (*pos, std::cmp::Reverse(token.len())))
}

fn next_wrapper_boundary(
    input: &str,
    search_from: usize,
    start_tokens: &[String],
    end_token: &str,
) -> WrapperBoundary {
    let tail = &input[search_from..];
    let next_start = find_next_start_token(input, search_from, start_tokens).map(|(pos, _)| pos);
    let next_end = tail.find(end_token).map(|pos| search_from + pos);
    match (next_start, next_end) {
        (Some(start), Some(end)) if start < end => WrapperBoundary::Start(start),
        (_, Some(end)) => WrapperBoundary::End(end),
        (Some(start), None) => WrapperBoundary::Start(start),
        (None, None) => WrapperBoundary::None,
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

fn try_parse_normal_text_from_tokens(input: &str, start_tokens: &[String]) -> String {
    find_next_start_token(input, 0, start_tokens)
        .map(|(idx, _)| input[..idx].trim().to_string())
        .unwrap_or_default()
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
/// Parse `payload` into tool calls, trying the three canonical JSON shapes in
/// order: an array of calls, a single `{name, arguments}`, then a single
/// `{name, parameters}`. Within an array, each element is tried as
/// `arguments` then `parameters`.
///
/// Returns:
/// - `Ok(Some(calls))` when `payload` matched one of the shapes. The vec may be
///   empty (e.g. a literal `[]`, or an array whose elements were all malformed),
///   which still counts as "recognized" so the caller returns rather than
///   falling through to truncation repair / strict recovery.
/// - `Ok(None)` when `payload` matched none of the shapes.
///
/// `arguments` bytes are passed through verbatim via `RawValue::get()` rather
/// than re-serializing a parsed `HashMap` / `Value`, which keeps them
/// byte-identical to what the model emitted (required for KV-cache append-only
/// prefix matching across multi-step tool use).
fn parse_calls(
    payload: &str,
    allow_name_only: bool,
    tools: Option<&[ToolDefinition]>,
) -> anyhow::Result<Option<Vec<ToolCallResponse>>> {
    let mk = |name: String, args: &RawValue| ToolCallResponse {
        id: format!("call-{}", Uuid::new_v4()),
        tp: ToolCallType::Function,
        function: CalledFunction {
            name,
            arguments: args.get().to_string(),
        },
    };
    // Name-only call (`{"name": ...}` with no args) — opt-in via
    // `allow_name_only` and only when the tool schema permits empty arguments.
    let mk_empty = |name: String| ToolCallResponse {
        id: format!("call-{}", Uuid::new_v4()),
        tp: ToolCallType::Function,
        function: CalledFunction {
            name,
            arguments: "{}".to_string(),
        },
    };

    if let Ok(array) = serde_json::from_str::<Vec<Box<RawValue>>>(payload) {
        let mut calls = Vec::new();
        for item in array {
            let item_str = item.get();
            if let Ok(func_args) = serde_json::from_str::<CalledFunctionArguments>(item_str) {
                calls.push(mk(func_args.name, &func_args.arguments));
            } else if let Ok(func_params) =
                serde_json::from_str::<CalledFunctionParameters>(item_str)
            {
                calls.push(mk(func_params.name, &func_params.parameters));
            } else if allow_name_only
                && let Ok(func_name) = serde_json::from_str::<CalledFunctionNameOnly>(item_str)
                && tool_allows_empty_arguments(&func_name.name, tools)
            {
                calls.push(mk_empty(func_name.name));
            }
            // Skip malformed entries silently.
        }
        return Ok(Some(calls));
    }
    if let Ok(single) = serde_json::from_str::<CalledFunctionParameters>(payload) {
        return Ok(Some(vec![mk(single.name, &single.parameters)]));
    }
    if let Ok(single) = serde_json::from_str::<CalledFunctionArguments>(payload) {
        return Ok(Some(vec![mk(single.name, &single.arguments)]));
    }
    if allow_name_only
        && let Ok(single) = serde_json::from_str::<CalledFunctionNameOnly>(payload)
        && tool_allows_empty_arguments(&single.name, tools)
    {
        return Ok(Some(vec![mk_empty(single.name)]));
    }
    Ok(None)
}

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
    let mut extracted_marker_wrapped_content = false;

    // First, check if ANY start token exists in the input. `bare_json_mode`
    // short-circuits this to false so we always take the no-marker branch.
    let has_start_token = !config.bare_json_mode
        && tool_call_start_tokens
            .iter()
            .any(|token| !token.is_empty() && normal_text.contains(token));
    let has_marker_token = !config.bare_json_mode && contains_marker_token(trimmed, config);

    if !has_start_token {
        // No start tokens found, try to extract JSON directly. Everything that starts with { or [ is considered a potential JSON.
        if let Some(idx) = normal_text.find(['{', '[']) {
            let extracted_normal = normal_text[..idx].trim().to_string();
            let extracted_json = normal_text[idx..].trim().to_string();
            if !extracted_json.is_empty() {
                normal_text = if config.suppress_marker_tokens_on_parse_failure && has_marker_token
                {
                    let stripped_normal = strip_marker_tokens(&extracted_normal, config);
                    if stripped_normal != extracted_normal {
                        tracing::warn!(
                            why = "marker_prefix_before_recovered_raw_json",
                            "JSON tool-call recovery: stripped marker tokens before recovered raw JSON."
                        );
                    }
                    stripped_normal
                } else {
                    extracted_normal
                };
                json = if config.recover_orphan_end_token && has_marker_token {
                    if let Some(stripped) =
                        strip_trailing_end_tokens(&extracted_json, tool_call_end_tokens)
                    {
                        tracing::warn!(
                            why = "orphan_end_token",
                            original_bytes = extracted_json.len(),
                            stripped_bytes = extracted_json.len().saturating_sub(stripped.len()),
                            "JSON tool-call recovery: stripped trailing orphan end token before parsing raw JSON."
                        );
                        stripped
                    } else {
                        extracted_json
                    }
                } else {
                    extracted_json
                };
            }
        }
    } else {
        // Start tokens exist, use regex-based parsing
        // Try all combinations of start and end tokens
        'outer: for start_token in tool_call_start_tokens.iter() {
            for end_token in tool_call_end_tokens.iter() {
                let new_normal_text = if config.recover_malformed_wrappers {
                    try_parse_normal_text_from_tokens(&normal_text, tool_call_start_tokens)
                } else {
                    try_parse_normal_text(&normal_text, start_token)
                };

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
                            extracted_marker_wrapped_content = true;

                            break 'outer; // Found content, exit early
                        }
                    }
                    (false, false) => {
                        // Start and end token case
                        let mut result = if config.recover_malformed_wrappers {
                            extract_tool_call_content_with_recovery(
                                &json,
                                tool_call_start_tokens,
                                end_token,
                                config.allow_eof_recovery,
                            )
                        } else {
                            extract_tool_call_content_regex(&json, start_token, end_token)
                        };
                        // EOF recovery: only when explicitly opted in (finalize
                        // path). Streaming jails leave `allow_eof_recovery=false`
                        // so the parser doesn't claim a complete call before
                        // the end-token has actually arrived.
                        if result.is_none()
                            && config.allow_eof_recovery
                            && let Some((_, eof_start_token)) =
                                find_next_start_token(&json, 0, tool_call_start_tokens)
                        {
                            result = extract_tool_call_content_eof_recovery(&json, eof_start_token);
                            if let Some(content) = result.as_ref() {
                                tracing::warn!(
                                    why = "missing_end_token_eof_fallback",
                                    recovered_bytes = content.len(),
                                    "JSON tool-call recovery: treated EOF as the end token after wrapper extraction failed."
                                );
                            }
                        }
                        if let Some(content) = result {
                            // Check if we found a start token but got empty JSON back
                            // This indicates the token was found but no valid JSON followed
                            if content.is_empty() {
                                found_start_token_with_no_valid_json = true;
                            }

                            json = content;
                            normal_text = new_normal_text;
                            extracted_marker_wrapped_content = true;

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
    // Try the three canonical JSON shapes (single object with `parameters` or
    // `arguments`, or an array of either) plus the opt-in name-only shape
    // (`{"name": ...}` with no args). A recognized shape returns here —
    // including an empty array, which is a valid empty result and must not fall
    // through to truncation recovery.
    if let Some(calls) = parse_calls(json, config.allow_name_only_tool_calls, _tools)? {
        return Ok((calls, Some(normal_text)));
    }

    // Truncation recovery: balance unclosed strings/braces (common
    // max_tokens / EOS pattern) and retry the same three parses. Gated on
    // `allow_eof_recovery` so streaming jails don't claim a complete tool
    // call while the model is still emitting JSON tokens.
    if config.allow_eof_recovery
        && let Some(repaired) = try_repair_truncated_json(json)
        && let Some(calls) =
            parse_calls(repaired.as_str(), config.allow_name_only_tool_calls, _tools)?
        && !calls.is_empty()
    {
        tracing::warn!(
            why = "truncated_json_repaired_parse_retry",
            original_bytes = json.len(),
            "JSON tool-call recovery: repaired truncated JSON after the initial parse failed."
        );
        return Ok((calls, Some(normal_text)));
    }

    // If we found a start token but no valid JSON, return empty content
    // to avoid leaking the token and invalid JSON content
    if found_start_token_with_no_valid_json {
        if config.suppress_marker_tokens_on_parse_failure {
            tracing::warn!(
                why = "marker_wrapped_content_unparseable",
                "JSON tool-call recovery: suppressed marker-wrapped content that did not parse as a supported tool-call shape."
            );
            return Ok((vec![], Some(normal_text)));
        }
        return Ok((vec![], Some(String::new())));
    }

    // Strict recovery (opt-in via `strip_markup_on_recovery`, e.g. nemotron_deci):
    // every parse above failed, so the fall-through below would return the raw
    // text verbatim — which leaks the wrapper markers (`<TOOLCALL>` /
    // `</TOOLCALL>`) into `normal_text`. Instead, strip all configured markers
    // and retry a strict parse of the remaining payload: recover any well-formed
    // call (this is what salvages orphan-close framing like
    // `[{...}]</TOOLCALL>`), otherwise drop the content. Markers never reach the
    // user either way; `tracing::warn!` records what was recovered or dropped.
    // Gated on `allow_eof_recovery` so this only runs on finalize / non-streaming
    // aggregate paths — never on a mid-stream chunk. Firing mid-stream would
    // claim a "complete" call before the end token arrives (same hazard as
    // `allow_eof_recovery` itself), which strands the trailing `</TOOLCALL>` as
    // leaked normal_text on the next chunk.
    if config.strip_markup_on_recovery && config.allow_eof_recovery {
        // Only intervene when a wrapper marker is actually present. Plain text
        // with no tool-call marker is a normal (non-tool) response and MUST
        // pass through unchanged — it must never be dropped or treated as a
        // failed tool call.
        let has_marker = config
            .tool_call_start_tokens
            .iter()
            .chain(config.tool_call_end_tokens.iter())
            .any(|token| !token.is_empty() && trimmed.contains(token.as_str()));

        if has_marker {
            // Strip wrapper markers only at the boundaries — start tokens from
            // the front, end tokens from the end — never globally. A global
            // replace would corrupt literal marker text inside a JSON string
            // value (e.g. an argument that mentions "</TOOLCALL>"); boundary
            // stripping leaves the JSON bytes handed to serde untouched.
            //
            // Base the payload on `json` (already split from `normal_text` by
            // the extraction stages above), not `trimmed`. With a preamble like
            // `Let me check.[{...}]</TOOLCALL>`, `trimmed` re-glues the prose
            // onto the JSON so it never parses and the call is dropped; `json`
            // is just `[{...}]</TOOLCALL>` and recovers. `has_marker` still
            // checks `trimmed` because extraction may have already consumed the
            // markers from `json`.
            let mut payload = json;
            loop {
                payload = payload.trim();
                match config
                    .tool_call_start_tokens
                    .iter()
                    .filter(|token| !token.is_empty())
                    .find_map(|token| payload.strip_prefix(token.as_str()))
                {
                    Some(rest) => payload = rest,
                    None => break,
                }
            }
            loop {
                payload = payload.trim();
                match config
                    .tool_call_end_tokens
                    .iter()
                    .filter(|token| !token.is_empty())
                    .find_map(|token| payload.strip_suffix(token.as_str()))
                {
                    Some(rest) => payload = rest,
                    None => break,
                }
            }
            let payload = payload.trim();

            let calls = parse_calls(payload, config.allow_name_only_tool_calls, _tools)?
                .unwrap_or_default();

            if !calls.is_empty() {
                tracing::warn!(
                    recovered_calls = calls.len(),
                    "Recovered {} tool call(s) from malformed tool-call framing; stripped wrapper markers instead of leaking them into normal_text",
                    calls.len()
                );
                return Ok((calls, Some(String::new())));
            }

            tracing::warn!(
                dropped_content = %trimmed,
                "Dropping unparseable tool-call content; wrapper markers stripped, no valid tool call recovered"
            );
            return Ok((vec![], Some(String::new())));
        }
    }

    if config.suppress_marker_tokens_on_parse_failure && extracted_marker_wrapped_content {
        tracing::warn!(
            why = "marker_wrapped_content_unparseable",
            "JSON tool-call recovery: suppressed marker-wrapped content that did not parse as a supported tool-call shape."
        );
        Ok((vec![], Some(normal_text)))
    } else if config.suppress_marker_tokens_on_parse_failure && has_marker_token {
        tracing::warn!(
            why = "marker_token_without_parse",
            "JSON tool-call recovery: suppressed tool marker tokens that did not produce a valid tool call."
        );
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
mod wrapper_recovery_tests {
    use super::*;

    fn internlm_recovery_config() -> JsonParserConfig {
        JsonParserConfig {
            tool_call_start_tokens: vec![
                "<|action_start|><|plugin|>".to_string(),
                "<|action_start|>".to_string(),
            ],
            tool_call_end_tokens: vec!["<|action_end|>".to_string()],
            tool_call_sentinel_tokens: vec![
                "<|action_start|>".to_string(),
                "<|plugin|>".to_string(),
                "<|action_end|>".to_string(),
            ],
            recover_malformed_wrappers: true,
            recover_orphan_end_token: true,
            suppress_marker_tokens_on_parse_failure: true,
            ..Default::default()
        }
    }

    #[test]
    fn test_recovery_scans_mixed_internlm_start_tokens() {
        let config = internlm_recovery_config();
        let input = concat!(
            "prefix ",
            r#"<|action_start|><|plugin|>{"name":"get_weather","parameters":{"city":"NYC"}}<|action_end|>"#,
            r#"<|action_start|>{"name":"current_time","parameters":{"timezone":"UTC"}}<|action_end|>"#
        );

        let (calls, normal_text) = try_tool_call_parse_basic_json(input, &config, None).unwrap();
        assert_eq!(normal_text.as_deref(), Some("prefix"));
        assert_eq!(calls.len(), 2);
        assert_eq!(calls[0].function.name, "get_weather");
        assert_eq!(calls[1].function.name, "current_time");

        let first_args: serde_json::Value =
            serde_json::from_str(&calls[0].function.arguments).unwrap();
        let second_args: serde_json::Value =
            serde_json::from_str(&calls[1].function.arguments).unwrap();
        assert_eq!(first_args["city"], "NYC");
        assert_eq!(second_args["timezone"], "UTC");
    }

    #[test]
    fn test_internlm_recovery_strips_sentinel_only_prefix() {
        let config = internlm_recovery_config();
        let input =
            r#"<|plugin|>{"name":"get_weather","parameters":{"location":"NYC"}}<|action_end|>"#;

        let (calls, normal_text) = try_tool_call_parse_basic_json(input, &config, None).unwrap();
        assert_eq!(normal_text.as_deref(), Some(""));
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].function.name, "get_weather");

        let args: serde_json::Value = serde_json::from_str(&calls[0].function.arguments).unwrap();
        assert_eq!(args["location"], "NYC");
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

    #[test] // helper, TOOLCALLING.batch.8
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

    #[test] // helper, TOOLCALLING.stream.3
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

    #[test] // helper, TOOLCALLING.stream.3
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

    #[test] // helper, TOOLCALLING.stream.3
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

    #[test] // helper, TOOLCALLING.stream.3
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
