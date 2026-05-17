// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// GLM-4.7 Tool Call Parser
// Format: <tool_call>function_name<arg_key>param1</arg_key><arg_value>value1</arg_value></tool_call>
// Reference: https://huggingface.co/zai-org/GLM-4.7/blob/main/chat_template.jinja

use regex::Regex;
use serde_json::Value;
use std::collections::HashMap;
use tracing::warn;
use uuid::Uuid;

use super::super::ToolDefinition;
use super::super::config::Glm47ParserConfig;
use super::response::{CalledFunction, ToolCallResponse, ToolCallType};

/// Render a tool_call block snippet for logs. Bounded so a huge truncated
/// argument body doesn't blow up the log line; control chars are escaped
/// because raw newlines/tabs make the warning unreadable in grep/jq.
fn truncate_for_log(s: &str) -> String {
    const MAX: usize = 200;
    let mut out = String::with_capacity(MAX.min(s.len()) + 16);
    let mut bytes = 0usize;
    for ch in s.chars() {
        if bytes >= MAX {
            out.push('…');
            break;
        }
        match ch {
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c => out.push(c),
        }
        bytes += ch.len_utf8();
    }
    out
}

/// Check if a chunk contains the start of a GLM-4.7 tool call.
/// Format: <tool_call>function_name<arg_key>...</arg_key><arg_value>...</arg_value></tool_call>
pub fn detect_tool_call_start_glm47(chunk: &str, config: &Glm47ParserConfig) -> bool {
    let start_token = &config.tool_call_start;

    // Check if we have the complete start token
    if chunk.contains(start_token.as_str()) {
        return true;
    }

    // Check for partial match at the end of the chunk (for streaming)
    for i in 1..start_token.len() {
        if chunk.ends_with(&start_token[..i]) {
            return true;
        }
    }

    false
}

/// Find the end position of all consecutive GLM-4.7 tool calls.
/// When a model emits multiple parallel tool calls in one chunk
/// (e.g. `<tool_call>A</tool_call><tool_call>B</tool_call>`), this
/// function advances past every consecutive start→end pair so the
/// entire group is captured as a single jailed region.  Returns the
/// position after the last `</tool_call>` found, or the length of the
/// chunk when no end token is present.
pub fn find_tool_call_end_position_glm47(chunk: &str, config: &Glm47ParserConfig) -> usize {
    let start_token = &config.tool_call_start;
    let end_token = &config.tool_call_end;

    let Some(first_end) = chunk.find(end_token.as_str()) else {
        return chunk.len();
    };

    let mut cursor = first_end + end_token.len();

    loop {
        let rest = &chunk[cursor..];
        let trimmed = rest.trim_start();
        if !trimmed.starts_with(start_token.as_str()) {
            break;
        }
        let trim_offset = rest.len() - trimmed.len();
        let search_from = cursor + trim_offset + start_token.len();
        if let Some(end_pos) = chunk[search_from..].find(end_token.as_str()) {
            cursor = search_from + end_pos + end_token.len();
        } else {
            break;
        }
    }

    cursor
}

/// Try to parse GLM-4.7 formatted tool calls from a message.
/// Format: <tool_call>function_name<arg_key>param1</arg_key><arg_value>value1</arg_value></tool_call>
/// Returns (parsed_tool_calls, normal_text_content)
pub fn try_tool_call_parse_glm47(
    message: &str,
    config: &Glm47ParserConfig,
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
    config: &Glm47ParserConfig,
    tools: Option<&[ToolDefinition]>,
) -> anyhow::Result<(String, Vec<ToolCallResponse>)> {
    let mut normal_parts = Vec::new();
    let mut calls = Vec::new();
    let mut cursor = 0;

    let start_token = &config.tool_call_start;
    let end_token = &config.tool_call_end;

    while cursor < text.len() {
        // Find next tool call start
        if let Some(start_pos) = text[cursor..].find(start_token.as_str()) {
            let abs_start = cursor + start_pos;

            // Only surface normal text that precedes the first parsed call.
            // Text after any </tool_call> is not response content; matches the
            // convention ported into the generic XML parser by PR #9350 and
            // vLLM's glm47_moe_tool_parser.
            if calls.is_empty() {
                normal_parts.push(&text[cursor..abs_start]);
            }

            // Find the corresponding end token
            if let Some(end_pos) = text[abs_start..].find(end_token.as_str()) {
                let abs_end = abs_start + end_pos + end_token.len();
                let block = &text[abs_start..abs_end];

                // Parse this tool call block. Unparseable blocks (malformed
                // <tool_call>...</tool_call> markup the parser can't extract)
                // are dropped — emitting the raw markup as normal_text leaks
                // wire tags downstream. vLLM and SGLang both drop on this
                // path; aligning Dynamo to that contract.
                match parse_tool_call_block(block, config, tools) {
                    Ok(parsed_call) => calls.push(parsed_call),
                    Err(e) => {
                        warn!(
                            reason = %e,
                            why = "block has open + close fence but content failed to parse \
                                   as a GLM-4.7 tool call (e.g. empty function name, \
                                   missing <arg_key>, malformed args); dropping to avoid \
                                   leaking wire tags through normal_text",
                            dropped_block = %truncate_for_log(block),
                            "GLM-4.7 parser dropping unparseable tool_call block"
                        );
                    }
                }

                cursor = abs_end;
            } else {
                // Recovery: outer </tool_call> absent (max_tokens / EOS
                // truncation). Gated on `allow_eof_recovery` so streaming
                // early-exit doesn't fire mid-stream. Also requires an
                // `<arg_key>` opener in the trailing slice as the structural
                // signal that a real tool call was emitted.
                let block = &text[abs_start..];
                let arg_key_start = &config.arg_key_start;
                if config.allow_eof_recovery && block.contains(arg_key_start.as_str()) {
                    match parse_tool_call_block(block, config, tools) {
                        Ok(parsed_call) => {
                            calls.push(parsed_call);
                            cursor = text.len();
                            continue;
                        }
                        Err(e) => {
                            warn!(
                                reason = %e,
                                why = "EOF recovery enabled and <arg_key> opener present, \
                                       but parse_tool_call_block failed on the truncated \
                                       tail; dropping to avoid leaking wire tags through \
                                       normal_text",
                                dropped_block = %truncate_for_log(block),
                                "GLM-4.7 parser dropping truncated tool_call block (recovery attempt failed)"
                            );
                        }
                    }
                } else {
                    // Either recovery disabled (production default for GLM-4.7)
                    // or no <arg_key> in the tail (so this is plausibly not a
                    // real tool call at all, just a stray <tool_call> token).
                    let reason = if !config.allow_eof_recovery {
                        "allow_eof_recovery=false (production default for GLM-4.7 to match \
                         vLLM/SGLang on truncated tool calls)"
                    } else {
                        "no <arg_key> in the tail after the <tool_call> start fence, so the \
                         block does not look like a structurally-real GLM-4.7 tool call"
                    };
                    warn!(
                        why = %reason,
                        dropped_block = %truncate_for_log(block),
                        "GLM-4.7 parser dropping truncated tool_call block (no end fence)"
                    );
                }
                // Drop the truncated/unrecoverable tail. Emitting the raw
                // <tool_call>...<arg_key>...<arg_value>... prefix as
                // normal_text would leak wire tags into message.content; vLLM
                // strips the same way on truncation.
                break;
            }
        } else {
            // No more tool calls
            if calls.is_empty() {
                normal_parts.push(&text[cursor..]);
            }
            break;
        }
    }

    let normal_text = normal_parts.join("");
    let normal_text = if calls.is_empty() {
        normal_text.trim().to_string()
    } else {
        normal_text
    };
    Ok((normal_text, calls))
}

/// Decode XML character entities in a string.
/// Handles the five predefined XML entities: &lt; &gt; &amp; &quot; &apos;
fn decode_xml_entities(s: &str) -> String {
    s.replace("&lt;", "<")
        .replace("&gt;", ">")
        .replace("&amp;", "&")
        .replace("&quot;", "\"")
        .replace("&apos;", "'")
}

/// Coerce a raw string value to a JSON Value using the tool's parameter schema.
/// Falls back to string if no schema is available or the type is unrecognized.
fn coerce_value(raw: &str, schema_type: Option<&str>) -> Value {
    let trimmed = raw.trim();

    // If the value already looks like JSON (object, array, or quoted string), parse it directly
    if (trimmed.starts_with('{') || trimmed.starts_with('[') || trimmed.starts_with('"'))
        && let Ok(v) = serde_json::from_str(trimmed)
    {
        return v;
    }

    // Use schema type hints for coercion when available
    match schema_type {
        Some("integer") | Some("int") => {
            if let Ok(n) = trimmed.parse::<i64>() {
                return Value::Number(n.into());
            }
        }
        Some("number") | Some("float") | Some("double") => {
            if let Ok(n) = trimmed.parse::<f64>()
                && let Some(num) = serde_json::Number::from_f64(n)
            {
                return Value::Number(num);
            }
        }
        Some("boolean") | Some("bool") => match trimmed.to_lowercase().as_str() {
            "true" | "1" | "yes" => return Value::Bool(true),
            "false" | "0" | "no" => return Value::Bool(false),
            _ => {}
        },
        Some("array") => {
            // Try JSON parse first, then fall back to comma-separated splitting
            if let Ok(v) = serde_json::from_str::<Value>(trimmed)
                && v.is_array()
            {
                return v;
            }
            let items: Vec<Value> = trimmed
                .split(',')
                .map(|s| Value::String(s.trim().to_string()))
                .collect();
            return Value::Array(items);
        }
        Some("null") => {
            if trimmed == "null" || trimmed == "None" || trimmed.is_empty() {
                return Value::Null;
            }
        }
        _ => {}
    }

    Value::String(raw.to_string())
}

/// Look up the JSON Schema type for a parameter by name from a tool's parameter schema.
fn get_param_schema_type<'a>(
    tools: Option<&'a [ToolDefinition]>,
    function_name: &str,
    param_name: &str,
) -> Option<&'a str> {
    let tool = tools?.iter().find(|t| t.name == function_name)?;
    let schema = tool.parameters.as_ref()?;
    let props = schema.get("properties")?;
    let param = props.get(param_name)?;
    param.get("type")?.as_str()
}

/// Parse a single GLM-4.7 tool call block
/// Format: <tool_call>function_name<arg_key>key1</arg_key><arg_value>value1</arg_value>...</tool_call>
fn parse_tool_call_block(
    block: &str,
    config: &Glm47ParserConfig,
    tools: Option<&[ToolDefinition]>,
) -> anyhow::Result<ToolCallResponse> {
    // Remove the outer <tool_call> tags
    let start_token = &config.tool_call_start;
    let end_token = &config.tool_call_end;

    // Strip the outer start token. The end token is optional so we can
    // recover from max_tokens / EOS truncation that drops `</tool_call>`.
    let after_start = block
        .strip_prefix(start_token.as_str())
        .ok_or_else(|| anyhow::anyhow!("Invalid tool call block format"))?;
    let content = after_start
        .strip_suffix(end_token.as_str())
        .unwrap_or(after_start);

    // Extract function name (everything before first <arg_key> or end)
    let arg_key_start = &config.arg_key_start;
    let function_name = if let Some(pos) = content.find(arg_key_start.as_str()) {
        content[..pos].trim().to_string()
    } else {
        // No arguments, just function name
        content.trim().to_string()
    };

    if function_name.is_empty() {
        anyhow::bail!("Empty function name in tool call");
    }

    // Parse key-value pairs
    let mut arguments = HashMap::new();
    let args_section = &content[function_name.len()..];

    // Build regex patterns
    let arg_key_start_escaped = regex::escape(&config.arg_key_start);
    let arg_key_end_escaped = regex::escape(&config.arg_key_end);
    let arg_value_start_escaped = regex::escape(&config.arg_value_start);
    let arg_value_end_escaped = regex::escape(&config.arg_value_end);

    // Pattern to match: <arg_key>key</arg_key><arg_value>value</arg_value>
    // (?s) enables dotall mode so (.*?) matches across newlines — required
    // because models often emit multi-line content in arg values.
    let pattern = format!(
        r"(?s){}([^<]+){}{}(.*?){}",
        arg_key_start_escaped, arg_key_end_escaped, arg_value_start_escaped, arg_value_end_escaped
    );

    let regex = Regex::new(&pattern)?;

    for cap in regex.captures_iter(args_section) {
        let key = cap.get(1).map(|m| m.as_str().trim()).unwrap_or("");
        let raw_value = cap.get(2).map(|m| m.as_str()).unwrap_or("");

        if !key.is_empty() {
            // Decode XML entities (e.g. &lt; → <, &amp; → &) before parsing
            let decoded = decode_xml_entities(raw_value);

            // Look up the expected type from the tool's parameter schema
            let schema_type = get_param_schema_type(tools, &function_name, key);
            let json_value = coerce_value(&decoded, schema_type);

            arguments.insert(key.to_string(), json_value);
        }
    }

    // Validate function against tools if provided
    if let Some(tools_list) = tools {
        let tool_exists = tools_list.iter().any(|t| t.name == function_name);
        if !tool_exists {
            anyhow::bail!("Function '{}' not found in available tools", function_name);
        }
    }

    Ok(ToolCallResponse {
        id: Uuid::new_v4().to_string(),
        tp: ToolCallType::Function,
        function: CalledFunction {
            name: function_name,
            arguments: serde_json::to_string(&arguments)?,
        },
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn get_test_config() -> Glm47ParserConfig {
        Glm47ParserConfig::default()
    }

    #[test] // helper
    fn test_detect_tool_call_start() {
        let config = get_test_config();

        // Complete start token
        assert!(detect_tool_call_start_glm47(
            "<tool_call>get_weather",
            &config
        ));

        // Partial start token (streaming)
        assert!(detect_tool_call_start_glm47("Some text <tool", &config));
        assert!(detect_tool_call_start_glm47("Some text <tool_c", &config));

        // No tool call
        assert!(!detect_tool_call_start_glm47("Just normal text", &config));
    }

    #[test] // PARSER.batch.1
    fn test_parse_simple_tool_call() {
        let config = get_test_config();
        let message = "<tool_call>get_weather<arg_key>location</arg_key><arg_value>San Francisco</arg_value></tool_call>";

        let (calls, normal_text) = try_tool_call_parse_glm47(message, &config, None).unwrap();

        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].function.name, "get_weather");

        let args: HashMap<String, Value> =
            serde_json::from_str(&calls[0].function.arguments).unwrap();
        assert_eq!(
            args.get("location").unwrap().as_str().unwrap(),
            "San Francisco"
        );
        assert_eq!(normal_text, Some("".to_string()));
    }

    #[test] // PARSER.batch.1, PARSER.batch.7
    fn test_parse_tool_call_with_multiple_args() {
        let config = get_test_config();
        let message = "<tool_call>book_flight<arg_key>from</arg_key><arg_value>NYC</arg_value><arg_key>to</arg_key><arg_value>LAX</arg_value><arg_key>date</arg_key><arg_value>2026-03-15</arg_value></tool_call>";

        let (calls, _) = try_tool_call_parse_glm47(message, &config, None).unwrap();

        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].function.name, "book_flight");

        let args: HashMap<String, Value> =
            serde_json::from_str(&calls[0].function.arguments).unwrap();
        assert_eq!(args.get("from").unwrap().as_str().unwrap(), "NYC");
        assert_eq!(args.get("to").unwrap().as_str().unwrap(), "LAX");
        assert_eq!(args.get("date").unwrap().as_str().unwrap(), "2026-03-15");
    }

    #[test] // PARSER.batch.7
    fn test_parse_tool_call_with_json_value() {
        let config = get_test_config();
        let message = r#"<tool_call>search<arg_key>filters</arg_key><arg_value>{"category": "books", "price_max": 50}</arg_value></tool_call>"#;

        let (calls, _) = try_tool_call_parse_glm47(message, &config, None).unwrap();

        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].function.name, "search");

        let args: HashMap<String, Value> =
            serde_json::from_str(&calls[0].function.arguments).unwrap();
        let filters = args.get("filters").unwrap();
        assert!(filters.is_object());
    }

    #[test] // PARSER.batch.2
    fn test_parse_multiple_tool_calls() {
        let config = get_test_config();
        let message = "<tool_call>get_weather<arg_key>location</arg_key><arg_value>NYC</arg_value></tool_call><tool_call>get_time<arg_key>timezone</arg_key><arg_value>EST</arg_value></tool_call>";

        let (calls, _) = try_tool_call_parse_glm47(message, &config, None).unwrap();

        assert_eq!(calls.len(), 2);
        assert_eq!(calls[0].function.name, "get_weather");
        assert_eq!(calls[1].function.name, "get_time");
    }

    #[test] // PARSER.batch.8
    fn test_parse_with_normal_text() {
        let config = get_test_config();
        let message = "I'll check the weather for you. <tool_call>get_weather<arg_key>location</arg_key><arg_value>Paris</arg_value></tool_call>";

        let (calls, normal_text) = try_tool_call_parse_glm47(message, &config, None).unwrap();

        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].function.name, "get_weather");
        assert_eq!(
            normal_text,
            Some("I'll check the weather for you. ".to_string())
        );
    }

    #[test] // PARSER.batch.6
    fn test_parse_tool_call_no_args() {
        let config = get_test_config();
        let message = "<tool_call>get_current_time</tool_call>";

        let (calls, _) = try_tool_call_parse_glm47(message, &config, None).unwrap();

        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].function.name, "get_current_time");

        let args: HashMap<String, Value> =
            serde_json::from_str(&calls[0].function.arguments).unwrap();
        assert!(args.is_empty());
    }

    #[test] // helper
    fn test_find_tool_call_end_position() {
        let config = get_test_config();
        let chunk =
            "<tool_call>func<arg_key>k</arg_key><arg_value>v</arg_value></tool_call>more text";

        let end_pos = find_tool_call_end_position_glm47(chunk, &config);
        assert_eq!(
            &chunk[..end_pos],
            "<tool_call>func<arg_key>k</arg_key><arg_value>v</arg_value></tool_call>"
        );
    }

    #[test] // helper — parallel calls: end position must advance past ALL blocks
    fn test_find_tool_call_end_position_parallel() {
        let config = get_test_config();
        let chunk = "<tool_call>get_weather<arg_key>location</arg_key><arg_value>SF</arg_value></tool_call><tool_call>get_weather<arg_key>location</arg_key><arg_value>NYC</arg_value></tool_call>trailing";

        let end_pos = find_tool_call_end_position_glm47(chunk, &config);
        assert_eq!(
            &chunk[..end_pos],
            "<tool_call>get_weather<arg_key>location</arg_key><arg_value>SF</arg_value></tool_call><tool_call>get_weather<arg_key>location</arg_key><arg_value>NYC</arg_value></tool_call>",
            "Must advance past ALL consecutive tool call blocks"
        );
    }

    #[test] // helper — parallel calls with whitespace between blocks
    fn test_find_tool_call_end_position_parallel_with_whitespace() {
        let config = get_test_config();
        let chunk = "<tool_call>a<arg_key>k</arg_key><arg_value>1</arg_value></tool_call>\n<tool_call>b<arg_key>k</arg_key><arg_value>2</arg_value></tool_call>";

        let end_pos = find_tool_call_end_position_glm47(chunk, &config);
        assert_eq!(
            end_pos,
            chunk.len(),
            "Must handle whitespace/newlines between consecutive blocks"
        );
    }

    #[test] // helper — parallel calls: second block incomplete (streaming)
    fn test_find_tool_call_end_position_parallel_second_incomplete() {
        let config = get_test_config();
        let chunk = "<tool_call>a<arg_key>k</arg_key><arg_value>1</arg_value></tool_call><tool_call>b<arg_key>k</arg_key><arg_value>2</arg_value>";

        let end_pos = find_tool_call_end_position_glm47(chunk, &config);
        assert_eq!(
            &chunk[..end_pos],
            "<tool_call>a<arg_key>k</arg_key><arg_value>1</arg_value></tool_call>",
            "Must stop at first complete block when second is incomplete"
        );
    }

    #[test] // PARSER.batch.2 + PARSER.batch.8 — bug report repro: text + parallel calls
    fn test_parse_text_then_parallel_calls() {
        let config = get_test_config();
        let message = "I'll check the weather for both cities at the same time!<tool_call>get_weather<arg_key>location</arg_key><arg_value>San Francisco</arg_value></tool_call><tool_call>get_weather<arg_key>location</arg_key><arg_value>New York</arg_value></tool_call>";

        let (calls, normal_text) = try_tool_call_parse_glm47(message, &config, None).unwrap();

        assert_eq!(calls.len(), 2, "Both parallel calls must be extracted");
        assert_eq!(calls[0].function.name, "get_weather");
        assert_eq!(calls[1].function.name, "get_weather");

        let args0: HashMap<String, Value> =
            serde_json::from_str(&calls[0].function.arguments).unwrap();
        let args1: HashMap<String, Value> =
            serde_json::from_str(&calls[1].function.arguments).unwrap();
        assert_eq!(
            args0.get("location").unwrap().as_str().unwrap(),
            "San Francisco"
        );
        assert_eq!(args1.get("location").unwrap().as_str().unwrap(), "New York");

        let text = normal_text.unwrap();
        assert_eq!(
            text,
            "I'll check the weather for both cities at the same time!"
        );
    }

    #[test] // PARSER.batch.7, PARSER.fmt.2
    fn test_parse_multiline_arg_value() {
        let config = get_test_config();
        let message = "<tool_call>write_file<arg_key>path</arg_key><arg_value>/tmp/hello.py</arg_value><arg_key>content</arg_key><arg_value>#!/usr/bin/env python3\nprint(\"Hello, World!\")\n</arg_value></tool_call>";

        let (calls, _) = try_tool_call_parse_glm47(message, &config, None).unwrap();

        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].function.name, "write_file");

        let args: HashMap<String, Value> =
            serde_json::from_str(&calls[0].function.arguments).unwrap();
        assert_eq!(args.get("path").unwrap().as_str().unwrap(), "/tmp/hello.py");
        assert!(
            args.contains_key("content"),
            "content argument must be parsed even when it contains newlines"
        );
        let content = args.get("content").unwrap().as_str().unwrap();
        assert!(content.contains("print(\"Hello, World!\")"));
    }

    #[test] // PARSER.batch.4
    fn test_malformed_tool_call() {
        let config = get_test_config();

        // Missing end tag
        let message = "<tool_call>get_weather";
        let result = try_tool_call_parse_glm47(message, &config, None);
        assert!(result.is_ok()); // Should handle gracefully, no calls extracted

        let (calls, _) = result.unwrap();
        assert_eq!(calls.len(), 0);
    }

    // Recovery for missing outer </tool_call> (max_tokens / EOS truncation):
    // when the inner arg pairs are well-formed, treat EOF as the end token
    // and extract the call. The arg_key opener gates recovery so plain text
    // that happens to start with `<tool_call>` is still preserved verbatim.
    #[test] // PARSER.batch.5
    fn test_parse_no_end_tag_complete_args_recovers() {
        let config = Glm47ParserConfig {
            allow_eof_recovery: true,
            ..get_test_config()
        };
        // Args complete, only outer </tool_call> missing.
        let message = "<tool_call>get_weather<arg_key>location</arg_key><arg_value>NYC</arg_value>";

        let (calls, _) = try_tool_call_parse_glm47(message, &config, None).unwrap();
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].function.name, "get_weather");
        let args: serde_json::Value = serde_json::from_str(&calls[0].function.arguments).unwrap();
        assert_eq!(args["location"], "NYC");
    }

    #[test] // PARSER.batch.5
    fn test_parse_no_end_tag_multiple_calls_recovers() {
        let config = Glm47ParserConfig {
            allow_eof_recovery: true,
            ..get_test_config()
        };
        // Two complete inner calls, missing only the trailing </tool_call> on the second.
        let message = "<tool_call>get_weather<arg_key>city</arg_key><arg_value>NYC</arg_value></tool_call><tool_call>get_time<arg_key>tz</arg_key><arg_value>EST</arg_value>";

        let (calls, _) = try_tool_call_parse_glm47(message, &config, None).unwrap();
        assert_eq!(calls.len(), 2);
        assert_eq!(calls[0].function.name, "get_weather");
        assert_eq!(calls[1].function.name, "get_time");
    }

    #[test] // PARSER.batch.4, PARSER.batch.8
    fn test_unparseable_block_dropped_no_tag_leak() {
        let config = get_test_config();
        let tools = vec![ToolDefinition {
            name: "get_weather".to_string(),
            parameters: None,
        }];

        // Tool call block references a function not in the tools list — the
        // whole block (including <tool_call>...<arg_key>...<arg_value>... wire
        // markup) must be dropped, not leaked through normal_text.
        let message = "Here is the result: <tool_call>unknown_func<arg_key>x</arg_key><arg_value>1</arg_value></tool_call> done";
        let (calls, normal_text) =
            try_tool_call_parse_glm47(message, &config, Some(&tools)).unwrap();

        assert_eq!(calls.len(), 0);
        let text = normal_text.unwrap();
        assert!(
            !text.contains("unknown_func"),
            "Unparseable block must be dropped to avoid tag leakage, got: {text}"
        );
        assert!(
            !text.contains("<tool_call>") && !text.contains("<arg_key>"),
            "Wire-format tags must not leak into normal_text, got: {text}"
        );
        assert!(
            text.contains("Here is the result:") && text.contains("done"),
            "Surrounding prose must be preserved, got: {text}"
        );
    }

    #[test] // helper
    fn test_xml_entity_decoding() {
        let config = get_test_config();
        let message = r#"<tool_call>write_file<arg_key>content</arg_key><arg_value>x &lt; y &amp;&amp; y &gt; z</arg_value></tool_call>"#;

        let (calls, _) = try_tool_call_parse_glm47(message, &config, None).unwrap();

        assert_eq!(calls.len(), 1);
        let args: HashMap<String, Value> =
            serde_json::from_str(&calls[0].function.arguments).unwrap();
        assert_eq!(
            args.get("content").unwrap().as_str().unwrap(),
            "x < y && y > z"
        );
    }

    #[test] // helper
    fn test_type_coercion_with_schema() {
        let config = get_test_config();
        let tools = vec![ToolDefinition {
            name: "set_temperature".to_string(),
            parameters: Some(serde_json::json!({
                "type": "object",
                "properties": {
                    "degrees": {"type": "number"},
                    "enabled": {"type": "boolean"},
                    "count": {"type": "integer"},
                    "label": {"type": "string"}
                }
            })),
        }];

        let message = "<tool_call>set_temperature<arg_key>degrees</arg_key><arg_value>72.5</arg_value><arg_key>enabled</arg_key><arg_value>true</arg_value><arg_key>count</arg_key><arg_value>3</arg_value><arg_key>label</arg_key><arg_value>warm</arg_value></tool_call>";

        let (calls, _) = try_tool_call_parse_glm47(message, &config, Some(&tools)).unwrap();
        assert_eq!(calls.len(), 1);

        let args: HashMap<String, Value> =
            serde_json::from_str(&calls[0].function.arguments).unwrap();

        // number coercion
        assert_eq!(args.get("degrees").unwrap().as_f64().unwrap(), 72.5);
        // boolean coercion
        assert!(args.get("enabled").unwrap().as_bool().unwrap());
        // integer coercion
        assert_eq!(args.get("count").unwrap().as_i64().unwrap(), 3);
        // string stays string
        assert_eq!(args.get("label").unwrap().as_str().unwrap(), "warm");
    }

    #[test] // helper
    fn test_type_coercion_array_comma_separated() {
        let config = get_test_config();
        let tools = vec![ToolDefinition {
            name: "tag_item".to_string(),
            parameters: Some(serde_json::json!({
                "type": "object",
                "properties": {
                    "tags": {"type": "array"}
                }
            })),
        }];

        // Model emits comma-separated values without JSON brackets
        let message = "<tool_call>tag_item<arg_key>tags</arg_key><arg_value>rust, python, go</arg_value></tool_call>";
        let (calls, _) = try_tool_call_parse_glm47(message, &config, Some(&tools)).unwrap();

        let args: HashMap<String, Value> =
            serde_json::from_str(&calls[0].function.arguments).unwrap();
        let tags = args.get("tags").unwrap().as_array().unwrap();
        assert_eq!(tags.len(), 3);
        assert_eq!(tags[0].as_str().unwrap(), "rust");
        assert_eq!(tags[1].as_str().unwrap(), "python");
        assert_eq!(tags[2].as_str().unwrap(), "go");
    }

    #[test] // helper
    fn test_type_coercion_array_json() {
        let config = get_test_config();
        let tools = vec![ToolDefinition {
            name: "tag_item".to_string(),
            parameters: Some(serde_json::json!({
                "type": "object",
                "properties": {
                    "ids": {"type": "array"}
                }
            })),
        }];

        // Model emits proper JSON array
        let message = r#"<tool_call>tag_item<arg_key>ids</arg_key><arg_value>[1, 2, 3]</arg_value></tool_call>"#;
        let (calls, _) = try_tool_call_parse_glm47(message, &config, Some(&tools)).unwrap();

        let args: HashMap<String, Value> =
            serde_json::from_str(&calls[0].function.arguments).unwrap();
        let ids = args.get("ids").unwrap().as_array().unwrap();
        assert_eq!(ids.len(), 3);
        assert_eq!(ids[0].as_i64().unwrap(), 1);
    }

    #[test] // helper
    fn test_type_coercion_falls_back_to_string() {
        let config = get_test_config();
        let tools = vec![ToolDefinition {
            name: "test_func".to_string(),
            parameters: Some(serde_json::json!({
                "type": "object",
                "properties": {
                    "count": {"type": "integer"}
                }
            })),
        }];

        // "not_a_number" can't be parsed as integer — should fall back to string
        let message = "<tool_call>test_func<arg_key>count</arg_key><arg_value>not_a_number</arg_value></tool_call>";
        let (calls, _) = try_tool_call_parse_glm47(message, &config, Some(&tools)).unwrap();

        let args: HashMap<String, Value> =
            serde_json::from_str(&calls[0].function.arguments).unwrap();
        assert!(
            args.get("count").unwrap().is_string(),
            "Should fall back to string when coercion fails"
        );
    }

    /// Parser-level invariant: the glm47 parser is byte-stable — it doesn't
    /// see `finish_reason` and produces the same output regardless of the
    /// upstream stream-end reason. Real PIPELINE.finish_reason coverage (stop / tool_calls
    /// / length mapping) lives in `lib/llm/tests/test_streaming_tool_parsers.rs`
    /// and belongs in the cross-parser finish_reason mapping work-item
    /// (tracked separately).
    #[test]
    fn test_glm47_parser_output_independent_of_upstream_finish() {
        let config = get_test_config();
        let input = "<tool_call>get_weather<arg_key>location</arg_key><arg_value>NYC</arg_value></tool_call>";
        let (calls, _) = try_tool_call_parse_glm47(input, &config, None).unwrap();
        assert_eq!(calls.len(), 1);
    }

    /// PARSER.batch.9 — empty / null content variants. Truly-empty (zero bytes)
    /// and whitespace-only inputs must yield no tool calls; normal_text
    /// collapses to the empty string.
    #[test] // PARSER.batch.9
    fn test_parse_glm47_empty_and_whitespace_inputs() {
        let config = get_test_config();
        for input in &["", " ", "\n", "\t\n  \t"] {
            let (calls, normal) = try_tool_call_parse_glm47(input, &config, None).unwrap();
            assert!(
                calls.is_empty(),
                "Empty/whitespace input must yield no calls (input={:?})",
                input
            );
            assert_eq!(
                normal.as_deref(),
                Some(""),
                "Empty/whitespace input collapses to empty normal_text (input={:?})",
                input
            );
        }
    }

    /// PARSER.batch.10 — duplicate calls (same function name twice in one section).
    /// Universal gap noted in the test taxonomy; pin parser-level behavior —
    /// both calls returned with distinct ids.
    #[test] // PARSER.batch.10
    fn test_parse_glm47_duplicate_calls_same_name() {
        let config = get_test_config();
        let input = "<tool_call>get_weather<arg_key>location</arg_key><arg_value>NYC</arg_value></tool_call><tool_call>get_weather<arg_key>location</arg_key><arg_value>LA</arg_value></tool_call>";
        let (calls, _) = try_tool_call_parse_glm47(input, &config, None).unwrap();
        assert_eq!(calls.len(), 2, "Both duplicate-name calls must be returned");
        assert_eq!(calls[0].function.name, "get_weather");
        assert_eq!(calls[1].function.name, "get_weather");
        assert_ne!(
            calls[0].id, calls[1].id,
            "Duplicate calls must have distinct ids"
        );
        let args0: HashMap<String, Value> =
            serde_json::from_str(&calls[0].function.arguments).unwrap();
        let args1: HashMap<String, Value> =
            serde_json::from_str(&calls[1].function.arguments).unwrap();
        assert_eq!(args0.get("location").unwrap().as_str().unwrap(), "NYC");
        assert_eq!(args1.get("location").unwrap().as_str().unwrap(), "LA");
    }
}
