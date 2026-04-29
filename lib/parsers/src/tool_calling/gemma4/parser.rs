// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// Reference implementations:
// https://github.com/vllm-project/vllm/pull/.../vllm/tool_parsers/gemma4_tool_parser.py
//
// Gemma 4 uses a custom non-JSON serialization format for tool calls:
//
//     <|tool_call>call:func_name{key:<|"|>value<|"|>,num:42}<tool_call|>
//
// Strings are delimited by `<|"|>`, keys are bare (unquoted), nested objects
// and arrays are supported, and multiple tool calls are concatenated without
// separators. The parser converts this grammar into JSON-shaped arguments so
// downstream callers see the same `serde_json::Value` shape as every other
// parser family.

use std::sync::OnceLock;

use regex::Regex;
use serde_json::{Map, Value};
use uuid::Uuid;

use super::super::ToolDefinition;
use super::super::response::{CalledFunction, ToolCallResponse, ToolCallType};

pub(crate) const TOOL_CALL_START: &str = "<|tool_call>";
pub(crate) const TOOL_CALL_END: &str = "<tool_call|>";
pub(crate) const STRING_DELIM: &str = "<|\"|>";
const CALL_PREFIX: &str = "call:";

static TOOL_CALL_REGEX: OnceLock<Regex> = OnceLock::new();

/// Captures the function-name + raw-args body of a single complete tool call.
/// `(?s)` enables dot-all so nested arg bodies that span newlines parse correctly.
fn tool_call_regex() -> &'static Regex {
    TOOL_CALL_REGEX.get_or_init(|| {
        let pattern = format!(
            r"(?s){}{}(?P<name>[\w\-\.]+)\{{(?P<args>.*?)\}}{}",
            regex::escape(TOOL_CALL_START),
            regex::escape(CALL_PREFIX),
            regex::escape(TOOL_CALL_END),
        );
        Regex::new(&pattern).expect("Failed to compile gemma4 tool call regex")
    })
}

/// Detect whether `chunk` contains the start of a Gemma 4 tool call, including
/// partial-prefix matches at the chunk boundary so streaming pipelines can hold
/// off emitting bytes that may belong to a tool-call marker.
pub fn detect_tool_call_start_gemma4(chunk: &str) -> bool {
    if chunk.contains(TOOL_CALL_START) {
        return true;
    }
    for i in 1..TOOL_CALL_START.len() {
        if !TOOL_CALL_START.is_char_boundary(i) {
            continue;
        }
        if chunk.ends_with(&TOOL_CALL_START[..i]) {
            return true;
        }
    }
    false
}

/// Returns the position immediately after the *last* `<tool_call|>` end marker
/// in `chunk`, or `None` if no end marker has arrived yet (caller should keep
/// accumulating). Mirrors the kimi_k2 / dsml convention so the streaming jail
/// behaves consistently across parser families.
pub fn find_tool_call_end_position_gemma4(chunk: &str) -> Option<usize> {
    let mut last_end: Option<usize> = None;
    let mut search_from = 0;
    while let Some(pos) = chunk[search_from..].find(TOOL_CALL_END) {
        let abs = search_from + pos + TOOL_CALL_END.len();
        last_end = Some(abs);
        search_from = abs;
    }
    last_end
}

/// Parse a Gemma 4 model response into structured tool calls + leftover text.
///
/// Returns `(parsed_tool_calls, normal_text_content)`. Text outside the
/// `<|tool_call>...<tool_call|>` markers is returned as `normal_text`; truly
/// truncated calls (start marker present, end marker missing) are dropped from
/// the call list and their raw bytes are NOT re-emitted as user-visible text
/// (mirrors kimi_k2's behavior — surfacing half-parsed markers in user output
/// is worse than dropping the truncated call).
pub fn try_tool_call_parse_gemma4(
    message: &str,
    tools: Option<&[ToolDefinition]>,
) -> anyhow::Result<(Vec<ToolCallResponse>, Option<String>)> {
    let mut calls = Vec::new();
    let mut normal_parts = Vec::new();
    let mut cursor = 0;

    while cursor < message.len() {
        let Some(start_rel) = message[cursor..].find(TOOL_CALL_START) else {
            normal_parts.push(&message[cursor..]);
            break;
        };
        let abs_start = cursor + start_rel;
        normal_parts.push(&message[cursor..abs_start]);

        let Some(end_rel) = message[abs_start..].find(TOOL_CALL_END) else {
            // Truncated tool call — consume the rest of the buffer silently.
            cursor = message.len();
            break;
        };
        let abs_end = abs_start + end_rel + TOOL_CALL_END.len();
        let block = &message[abs_start..abs_end];

        if let Some(call) = parse_single_call(block, tools)? {
            calls.push(call);
        }

        cursor = abs_end;
    }

    let normal_text = normal_parts.join("").trim().to_string();
    let normal_content = if normal_text.is_empty() {
        Some(String::new())
    } else {
        Some(normal_text)
    };

    Ok((calls, normal_content))
}

fn parse_single_call(
    block: &str,
    tools: Option<&[ToolDefinition]>,
) -> anyhow::Result<Option<ToolCallResponse>> {
    let Some(caps) = tool_call_regex().captures(block) else {
        return Ok(None);
    };
    let name = caps
        .name("name")
        .map(|m| m.as_str().to_string())
        .unwrap_or_default();
    if name.is_empty() {
        return Ok(None);
    }
    let args_raw = caps.name("args").map(|m| m.as_str()).unwrap_or("");

    if let Some(tools) = tools
        && !tools.iter().any(|t| t.name == name)
    {
        tracing::warn!(
            "Tool '{}' is not defined in the tools list (Gemma 4 parser).",
            name
        );
    }

    let args_value = match parse_args_object(args_raw) {
        Ok(v) => v,
        Err(e) => {
            tracing::warn!(
                "Failed to parse Gemma 4 args for '{}': {}. Falling back to empty object.",
                name,
                e
            );
            Value::Object(Map::new())
        }
    };
    let arguments = serde_json::to_string(&args_value)?;

    Ok(Some(ToolCallResponse {
        id: format!("call-{}", Uuid::new_v4()),
        tp: ToolCallType::Function,
        function: CalledFunction {
            name,
            arguments,
        },
    }))
}

// ---------------------------------------------------------------------------
// Recursive-descent parser for the Gemma 4 argument grammar
// ---------------------------------------------------------------------------
//
// Grammar (informal):
//
//   args     = (entry ("," entry)*)?
//   entry    = key ":" value
//   key      = bare-identifier (no quoting in Gemma 4 emit)
//   value    = string | number | bool | null | object | array
//   string   = "<|\"|>" .* "<|\"|>"
//   number   = -? [0-9]+ ( "." [0-9]+ )?
//   bool     = "true" | "false"
//   null     = "null" | "none" | "nil"
//   object   = "{" args "}"
//   array    = "[" (value ("," value)*)? "]"
//
// We parse straight into `serde_json::Value` so the rest of the pipeline sees
// the same shape every other parser produces.

struct Cursor<'a> {
    src: &'a str,
    pos: usize,
}

impl<'a> Cursor<'a> {
    fn new(src: &'a str) -> Self {
        Self { src, pos: 0 }
    }

    fn rest(&self) -> &'a str {
        &self.src[self.pos..]
    }

    fn eof(&self) -> bool {
        self.pos >= self.src.len()
    }

    fn skip_whitespace(&mut self) {
        let bytes = self.src.as_bytes();
        while self.pos < bytes.len() && bytes[self.pos].is_ascii_whitespace() {
            self.pos += 1;
        }
    }

    fn peek_byte(&self) -> Option<u8> {
        self.src.as_bytes().get(self.pos).copied()
    }

    fn consume_byte(&mut self, b: u8) -> bool {
        if self.peek_byte() == Some(b) {
            self.pos += 1;
            true
        } else {
            false
        }
    }

    fn consume_str(&mut self, s: &str) -> bool {
        if self.rest().starts_with(s) {
            self.pos += s.len();
            true
        } else {
            false
        }
    }
}

pub(crate) fn parse_args_object(input: &str) -> anyhow::Result<Value> {
    let mut cur = Cursor::new(input);
    cur.skip_whitespace();
    let val = parse_object_body(&mut cur)?;
    cur.skip_whitespace();
    if !cur.eof() {
        anyhow::bail!(
            "trailing characters after Gemma 4 args object at offset {}: {:?}",
            cur.pos,
            cur.rest()
        );
    }
    Ok(val)
}

fn parse_object_body(cur: &mut Cursor) -> anyhow::Result<Value> {
    let mut map = Map::new();
    cur.skip_whitespace();
    if cur.eof() || cur.peek_byte() == Some(b'}') {
        return Ok(Value::Object(map));
    }
    loop {
        cur.skip_whitespace();
        let key = parse_key(cur)?;
        cur.skip_whitespace();
        if !cur.consume_byte(b':') {
            anyhow::bail!("expected ':' after key '{}' at offset {}", key, cur.pos);
        }
        cur.skip_whitespace();
        let value = parse_value(cur)?;
        map.insert(key, value);
        cur.skip_whitespace();
        if !cur.consume_byte(b',') {
            break;
        }
    }
    Ok(Value::Object(map))
}

fn parse_key(cur: &mut Cursor) -> anyhow::Result<String> {
    let bytes = cur.src.as_bytes();
    let start = cur.pos;
    while cur.pos < bytes.len() {
        let b = bytes[cur.pos];
        if b.is_ascii_alphanumeric() || b == b'_' || b == b'-' || b == b'.' {
            cur.pos += 1;
        } else {
            break;
        }
    }
    if cur.pos == start {
        anyhow::bail!("expected bare key at offset {}", start);
    }
    Ok(cur.src[start..cur.pos].to_string())
}

fn parse_value(cur: &mut Cursor) -> anyhow::Result<Value> {
    cur.skip_whitespace();

    // Delimited string: <|"|>...<|"|>
    if cur.rest().starts_with(STRING_DELIM) {
        let open_at = cur.pos;
        cur.pos += STRING_DELIM.len();
        let body_start = cur.pos;
        let Some(end_rel) = cur.src[body_start..].find(STRING_DELIM) else {
            anyhow::bail!("unterminated <|\"|> string starting at offset {open_at}");
        };
        let body_end = body_start + end_rel;
        let s = cur.src[body_start..body_end].to_string();
        cur.pos = body_end + STRING_DELIM.len();
        return Ok(Value::String(s));
    }

    // Object
    if cur.consume_byte(b'{') {
        let v = parse_object_body(cur)?;
        cur.skip_whitespace();
        if !cur.consume_byte(b'}') {
            anyhow::bail!("expected '}}' to close object at offset {}", cur.pos);
        }
        return Ok(v);
    }

    // Array
    if cur.consume_byte(b'[') {
        return parse_array(cur);
    }

    // Booleans / nulls (case-sensitive match — Gemma 4 emits lowercase, but we
    // accept the same null-aliases vLLM does for parity with offline parsing).
    if cur.consume_str("true") {
        return Ok(Value::Bool(true));
    }
    if cur.consume_str("false") {
        return Ok(Value::Bool(false));
    }
    for null_token in ["null", "none", "nil", "None", "NULL", "Nil"] {
        if cur.consume_str(null_token) {
            return Ok(Value::Null);
        }
    }

    // Number
    parse_number(cur)
}

fn parse_array(cur: &mut Cursor) -> anyhow::Result<Value> {
    let mut items = Vec::new();
    cur.skip_whitespace();
    if cur.consume_byte(b']') {
        return Ok(Value::Array(items));
    }
    loop {
        cur.skip_whitespace();
        items.push(parse_value(cur)?);
        cur.skip_whitespace();
        if cur.consume_byte(b']') {
            return Ok(Value::Array(items));
        }
        if !cur.consume_byte(b',') {
            anyhow::bail!("expected ',' or ']' in array at offset {}", cur.pos);
        }
    }
}

fn parse_number(cur: &mut Cursor) -> anyhow::Result<Value> {
    let start = cur.pos;
    let bytes = cur.src.as_bytes();
    if cur.peek_byte() == Some(b'-') {
        cur.pos += 1;
    }
    let int_start = cur.pos;
    while cur.pos < bytes.len() && bytes[cur.pos].is_ascii_digit() {
        cur.pos += 1;
    }
    if cur.pos == int_start {
        anyhow::bail!(
            "expected value at offset {} but got: {:?}",
            start,
            &cur.src[start..]
        );
    }
    let mut is_float = false;
    if cur.peek_byte() == Some(b'.') {
        is_float = true;
        cur.pos += 1;
        while cur.pos < bytes.len() && bytes[cur.pos].is_ascii_digit() {
            cur.pos += 1;
        }
    }
    let lex = &cur.src[start..cur.pos];
    if is_float {
        let f: f64 = lex.parse()?;
        Ok(serde_json::json!(f))
    } else {
        let i: i64 = lex.parse()?;
        Ok(serde_json::json!(i))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn extract_first(input: &str) -> (String, Value) {
        let (calls, _) = try_tool_call_parse_gemma4(input, None).unwrap();
        assert_eq!(calls.len(), 1, "expected exactly one tool call");
        let args: Value = serde_json::from_str(&calls[0].function.arguments).unwrap();
        (calls[0].function.name.clone(), args)
    }

    #[test] // CASE.20 — detection helper
    fn detect_full_and_partial_start() {
        assert!(detect_tool_call_start_gemma4("<|tool_call>"));
        assert!(detect_tool_call_start_gemma4("blah <|tool_call>"));
        assert!(detect_tool_call_start_gemma4("<|tool_"));
        assert!(detect_tool_call_start_gemma4("<|"));
        assert!(!detect_tool_call_start_gemma4("nothing here"));
        assert!(!detect_tool_call_start_gemma4("toolcall"));
    }

    #[test] // CASE.20 — end-position helper
    fn find_end_returns_position_after_last_marker() {
        let text = "<|tool_call>call:f{}<tool_call|>more";
        let pos = find_tool_call_end_position_gemma4(text).unwrap();
        assert_eq!(&text[pos..], "more");
        assert_eq!(find_tool_call_end_position_gemma4("<|tool_call>call:f{"), None);
    }

    #[test] // CASE.1 — single string argument
    fn parse_single_string_argument() {
        let input = r#"<|tool_call>call:get_weather{location:<|"|>Tokyo<|"|>}<tool_call|>"#;
        let (name, args) = extract_first(input);
        assert_eq!(name, "get_weather");
        assert_eq!(args["location"], "Tokyo");
    }

    #[test] // CASE.1, CASE.7 — multiple typed arguments
    fn parse_multiple_typed_arguments() {
        let input = r#"<|tool_call>call:f{loc:<|"|>San Francisco, CA<|"|>,unit:<|"|>celsius<|"|>,count:42,flag:true,nope:null}<tool_call|>"#;
        let (name, args) = extract_first(input);
        assert_eq!(name, "f");
        assert_eq!(args["loc"], "San Francisco, CA");
        assert_eq!(args["unit"], "celsius");
        assert_eq!(args["count"], 42);
        assert_eq!(args["flag"], true);
        assert_eq!(args["nope"], Value::Null);
    }

    #[test] // CASE.6 — empty argument object
    fn parse_no_arg_call() {
        let input = "<|tool_call>call:get_time{}<tool_call|>";
        let (name, args) = extract_first(input);
        assert_eq!(name, "get_time");
        assert!(args.as_object().unwrap().is_empty());
    }

    #[test] // CASE.7 — nested object value
    fn parse_nested_object_value() {
        let input = r#"<|tool_call>call:f{cfg:{ssl:true,pool:{min:5,max:20}}}<tool_call|>"#;
        let (_name, args) = extract_first(input);
        assert_eq!(args["cfg"]["ssl"], true);
        assert_eq!(args["cfg"]["pool"]["min"], 5);
        assert_eq!(args["cfg"]["pool"]["max"], 20);
    }

    #[test] // CASE.7 — array of strings
    fn parse_array_of_strings() {
        let input = r#"<|tool_call>call:f{tags:[<|"|>a<|"|>,<|"|>b<|"|>,<|"|>c<|"|>]}<tool_call|>"#;
        let (_name, args) = extract_first(input);
        assert_eq!(args["tags"], serde_json::json!(["a", "b", "c"]));
    }

    #[test] // CASE.7 — array of mixed primitives
    fn parse_array_of_mixed_primitives() {
        let input = "<|tool_call>call:f{xs:[1,2,3.5,true,false,null]}<tool_call|>";
        let (_name, args) = extract_first(input);
        assert_eq!(args["xs"][0], 1);
        assert_eq!(args["xs"][1], 2);
        assert!((args["xs"][2].as_f64().unwrap() - 3.5).abs() < 1e-9);
        assert_eq!(args["xs"][3], true);
        assert_eq!(args["xs"][4], false);
        assert_eq!(args["xs"][5], Value::Null);
    }

    #[test] // CASE.2 — multiple parallel calls, zero spacing
    fn parse_multiple_parallel_calls() {
        let input = concat!(
            "<|tool_call>call:a{x:1}<tool_call|>",
            "<|tool_call>call:b{y:<|\"|>two<|\"|>}<tool_call|>",
            "<|tool_call>call:c{}<tool_call|>",
        );
        let (calls, normal) = try_tool_call_parse_gemma4(input, None).unwrap();
        assert_eq!(calls.len(), 3);
        assert_eq!(calls[0].function.name, "a");
        assert_eq!(calls[1].function.name, "b");
        assert_eq!(calls[2].function.name, "c");
        assert_eq!(normal, Some(String::new()));
    }

    #[test] // CASE.13 — surrounding normal text preserved
    fn parse_with_surrounding_text() {
        let input = r#"Sure thing. <|tool_call>call:f{x:1}<tool_call|> All set."#;
        let (calls, normal) = try_tool_call_parse_gemma4(input, None).unwrap();
        assert_eq!(calls.len(), 1);
        assert_eq!(normal, Some("Sure thing.  All set.".to_string()));
    }

    #[test] // CASE.3 — no tool calls at all
    fn parse_no_tool_calls() {
        let (calls, normal) =
            try_tool_call_parse_gemma4("just plain prose here", None).unwrap();
        assert_eq!(calls.len(), 0);
        assert_eq!(normal, Some("just plain prose here".to_string()));
    }

    #[test] // CASE.5, CASE.16 — truncated mid-args is dropped, complete prior calls survive
    fn truncated_call_dropped_complete_prior_survives() {
        let input = concat!(
            "<|tool_call>call:complete{x:1}<tool_call|>",
            "<|tool_call>call:partial{y:<|\"|>incomp", // no end marker
        );
        let (calls, _normal) = try_tool_call_parse_gemma4(input, None).unwrap();
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].function.name, "complete");
    }

    #[test] // CASE.4 — malformed args body falls back to empty object, call still emitted
    fn malformed_args_falls_back_to_empty_object() {
        let input = "<|tool_call>call:f{garbage no colons here}<tool_call|>";
        let (calls, _) = try_tool_call_parse_gemma4(input, None).unwrap();
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].function.name, "f");
        let args: Value = serde_json::from_str(&calls[0].function.arguments).unwrap();
        assert!(args.as_object().unwrap().is_empty());
    }

    #[test] // CASE.21 — function names with hyphens, dots, underscores
    fn parse_function_names_with_special_chars() {
        for (input, expected_name) in [
            ("<|tool_call>call:list-tasklists{}<tool_call|>", "list-tasklists"),
            ("<|tool_call>call:mcp__portal__search-doc{}<tool_call|>", "mcp__portal__search-doc"),
            ("<|tool_call>call:my.namespaced.fn{}<tool_call|>", "my.namespaced.fn"),
        ] {
            let (calls, _) = try_tool_call_parse_gemma4(input, None).unwrap();
            assert_eq!(calls.len(), 1, "input: {input}");
            assert_eq!(calls[0].function.name, expected_name);
        }
    }

    #[test] // CASE.7 — angle brackets / HTML inside string values
    fn parse_html_in_string_value() {
        let input = r#"<|tool_call>call:render{html:<|"|><div class="x"><h1>Hi</h1></div><|"|>}<tool_call|>"#;
        let (_name, args) = extract_first(input);
        assert_eq!(args["html"], "<div class=\"x\"><h1>Hi</h1></div>");
    }

    #[test] // CASE.7 — newlines inside string values
    fn parse_newlines_in_string_value() {
        let input = "<|tool_call>call:f{body:<|\"|>line1\nline2\nline3<|\"|>}<tool_call|>";
        let (_name, args) = extract_first(input);
        assert_eq!(args["body"], "line1\nline2\nline3");
    }

    #[test] // CASE.22 — whitespace tolerance inside args
    fn parse_with_internal_whitespace() {
        let input = r#"<|tool_call>call:f{ x : 1 , y : <|"|>z<|"|> }<tool_call|>"#;
        let (_name, args) = extract_first(input);
        assert_eq!(args["x"], 1);
        assert_eq!(args["y"], "z");
    }

    #[test] // CASE.7 — negative numbers and floats
    fn parse_signed_numbers_and_floats() {
        let input = "<|tool_call>call:f{a:-1,b:-2.5,c:0,d:0.0}<tool_call|>";
        let (_name, args) = extract_first(input);
        assert_eq!(args["a"], -1);
        assert!((args["b"].as_f64().unwrap() - -2.5).abs() < 1e-9);
        assert_eq!(args["c"], 0);
        assert!((args["d"].as_f64().unwrap()).abs() < 1e-9);
    }

    #[test] // CASE.21 — tool validation warns but doesn't drop
    fn parse_with_tool_validation() {
        let input = r#"<|tool_call>call:get_weather{x:1}<tool_call|>"#;
        let tools = vec![ToolDefinition {
            name: "get_weather".to_string(),
            parameters: None,
        }];
        let (calls, _) = try_tool_call_parse_gemma4(input, Some(&tools)).unwrap();
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].function.name, "get_weather");
    }

    #[test] // CASE.24 — empty wrapper between calls
    fn parse_empty_text_between_calls() {
        let input = concat!(
            "<|tool_call>call:a{}<tool_call|>",
            "<|tool_call>call:b{}<tool_call|>",
        );
        let (calls, normal) = try_tool_call_parse_gemma4(input, None).unwrap();
        assert_eq!(calls.len(), 2);
        assert_eq!(normal, Some(String::new()));
    }

    // Argument-grammar unit tests (parse_args_object directly)

    #[test]
    fn args_grammar_empty() {
        let v = parse_args_object("").unwrap();
        assert_eq!(v, serde_json::json!({}));
    }

    #[test]
    fn args_grammar_string_with_special_chars() {
        let v = parse_args_object(r#"x:<|"|>has,comma:and{brace}<|"|>"#).unwrap();
        assert_eq!(v["x"], "has,comma:and{brace}");
    }

    #[test]
    fn args_grammar_deeply_nested() {
        let v = parse_args_object("a:{b:{c:{d:{e:1}}}}").unwrap();
        assert_eq!(v["a"]["b"]["c"]["d"]["e"], 1);
    }

    #[test]
    fn args_grammar_array_of_objects() {
        let v = parse_args_object(r#"items:[{n:<|"|>x<|"|>},{n:<|"|>y<|"|>}]"#).unwrap();
        assert_eq!(v["items"][0]["n"], "x");
        assert_eq!(v["items"][1]["n"], "y");
    }

    #[test]
    fn args_grammar_unterminated_string_errors() {
        let err = parse_args_object(r#"x:<|"|>oops"#).unwrap_err();
        assert!(err.to_string().contains("unterminated"));
    }
}
