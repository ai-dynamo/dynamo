// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! DeepSeek V4 native prompt formatting
//!
//! Native Rust port of DeepSeek V4's chat encoding (encoding_dsv4.py).
//!
//! Reference: DeepSeek-V4-Pro/encoding/encoding_dsv4.py

use anyhow::{Context, Result};
use serde_json::Value as JsonValue;

/// Special tokens for DeepSeek V4
pub mod tokens {
    pub const BOS: &str = "<｜begin▁of▁sentence｜>";
    pub const EOS: &str = "<｜end▁of▁sentence｜>";
    pub const THINKING_START: &str = "<think>";
    pub const THINKING_END: &str = "</think>";
    pub const DSML_TOKEN: &str = "｜DSML｜";
    pub const USER_START: &str = "<｜User｜>";
    pub const ASSISTANT_START: &str = "<｜Assistant｜>";
    pub const LATEST_REMINDER: &str = "<｜latest_reminder｜>";

    // Quick-instruction task tokens
    pub const TASK_ACTION: &str = "<｜action｜>";
    pub const TASK_QUERY: &str = "<｜query｜>";
    pub const TASK_AUTHORITY: &str = "<｜authority｜>";
    pub const TASK_DOMAIN: &str = "<｜domain｜>";
    pub const TASK_TITLE: &str = "<｜title｜>";
    pub const TASK_READ_URL: &str = "<｜read_url｜>";
}

/// DSML outer block name for tool-call groups: `<｜DSML｜tool_calls>...</｜DSML｜tool_calls>`.
const TOOL_CALLS_BLOCK_NAME: &str = "tool_calls";

/// Wire-format tags that wrap a tool result inside a user content block.
const TOOL_RESULT_OPEN: &str = "<tool_result>";
const TOOL_RESULT_CLOSE: &str = "</tool_result>";

/// Preamble line that introduces a response-format schema in both the
/// system and developer roles. The `{}` placeholder is filled by the
/// caller with the JSON-serialized schema.
const RESPONSE_FORMAT_PREAMBLE: &str =
    "## Response Format:\n\nYou MUST strictly adhere to the following schema to reply:\n{}";

/// Roles whose messages are always kept by `drop_thinking_messages`. All other
/// roles before the last user/developer message are dropped (assistant turns
/// keep their non-`reasoning_content` fields; everything else is removed).
const KEEP_ROLES: &[&str] = &[
    "user",
    "system",
    "tool",
    "latest_reminder",
    "direct_search_results",
];

/// Placeholder the Python reference inserts when it can't render an
/// unsupported content-block or tool-result item type. Rendered into the
/// assistant-visible prompt so a human can tell something was skipped;
/// dynamo additionally emits a `tracing::warn!` so ops see it too.
const UNSUPPORTED_PLACEHOLDER_FMT: &str = "[Unsupported {}]";

const REASONING_EFFORT_MAX: &str = "Reasoning Effort: Absolute maximum with no shortcuts permitted.\nYou MUST be very thorough in your thinking and comprehensively decompose the problem to resolve the root cause, rigorously stress-testing your logic against all potential paths, edge cases, and adversarial scenarios.\nExplicitly write out your entire deliberation process, documenting every intermediate step, considered alternative, and rejected hypothesis to ensure absolutely no assumption is left unchecked.\n\n";

/// Thinking mode for the model
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ThinkingMode {
    Chat,
    Thinking,
}

impl ThinkingMode {
    /// Tiny branch-free mapping to a static string. `#[inline]` because this
    /// is called from inside the per-message render hot path.
    #[inline]
    pub fn as_str(&self) -> &'static str {
        match self {
            ThinkingMode::Chat => "chat",
            ThinkingMode::Thinking => "thinking",
        }
    }
}

/// Reasoning effort level. `None` conveyed as `Option<ReasoningEffort>`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReasoningEffort {
    Max,
    High,
}

/// Serialize a JSON value to match Python's `json.dumps(ensure_ascii=False)` spacing.
/// Python's default separators are `(', ', ': ')`; we use a custom `Formatter`
/// so escape sequences inside strings can't confuse state tracking.
///
/// Returns `Result` so serialization / UTF-8 errors propagate to the caller
/// rather than silently collapsing to `"{}"` (the old error-swallow behavior
/// dropped the entire request payload with no signal up the stack).
fn to_json(value: &JsonValue) -> Result<String> {
    use serde::Serialize;
    use serde_json::ser::Formatter;
    use std::io;

    struct PythonFormatter;

    impl Formatter for PythonFormatter {
        fn begin_array_value<W: ?Sized + io::Write>(
            &mut self,
            writer: &mut W,
            first: bool,
        ) -> io::Result<()> {
            if first {
                Ok(())
            } else {
                writer.write_all(b", ")
            }
        }

        fn begin_object_key<W: ?Sized + io::Write>(
            &mut self,
            writer: &mut W,
            first: bool,
        ) -> io::Result<()> {
            if first {
                Ok(())
            } else {
                writer.write_all(b", ")
            }
        }

        fn begin_object_value<W: ?Sized + io::Write>(&mut self, writer: &mut W) -> io::Result<()> {
            writer.write_all(b": ")
        }
    }

    // Size the buffer from a compact pre-serialization. Python-style spacing
    // adds exactly one byte per structural separator (`,` → `, `, `:` → `: `),
    // which bounds the final length by ~1.125× the compact length. This keeps
    // small payloads cheap and avoids the 5–9 reallocations the prior fixed
    // 64-byte hint forced on KB-sized tool schemas and response formats.
    let compact_len = serde_json::to_string(value)
        .context("to_json: compact pre-serialization failed")?
        .len();
    let capacity = compact_len.saturating_add(compact_len / 8).max(256);
    let mut buf = Vec::with_capacity(capacity);
    let mut ser = serde_json::Serializer::with_formatter(&mut buf, PythonFormatter);
    value
        .serialize(&mut ser)
        .context("to_json: Python-formatter serialization failed")?;
    String::from_utf8(buf).context("to_json: serialized output is not valid UTF-8")
}

/// Extract function definitions from OpenAI-format tool list.
fn tools_from_openai_format(tools: &[JsonValue]) -> Vec<JsonValue> {
    tools
        .iter()
        .filter_map(|tool| tool.get("function").cloned())
        .collect()
}

/// Render tool schemas into the system prompt format.
///
/// Previously built via four sequential `String::replace` passes over a
/// `{placeholder}`-based template, which allocated a fresh copy of the whole
/// (schema-inlined, potentially kB-scale) string on each substitution. A
/// single `format!` with named arguments collapses that to one allocation
/// sized from the final length.
fn render_tools(tools: &[JsonValue]) -> Result<String> {
    let tools_json: Vec<String> = tools_from_openai_format(tools)
        .iter()
        .map(to_json)
        .collect::<Result<_>>()?;
    let schemas = tools_json.join("\n");

    Ok(format!(
        r#"## Tools

You have access to a set of tools to help answer the user's question. You can invoke tools by writing a "<{dsml}tool_calls>" block like the following:

<{dsml}tool_calls>
<{dsml}invoke name="$TOOL_NAME">
<{dsml}parameter name="$PARAMETER_NAME" string="true|false">$PARAMETER_VALUE</{dsml}parameter>
...
</{dsml}invoke>
<{dsml}invoke name="$TOOL_NAME2">
...
</{dsml}invoke>
</{dsml}tool_calls>

String parameters should be specified as is and set `string="true"`. For all other types (numbers, booleans, arrays, objects), pass the value in JSON format and set `string="false"`.

If thinking_mode is enabled (triggered by {think_open}), you MUST output your complete reasoning inside {think_open}...{think_close} BEFORE any tool calls or final response.

Otherwise, output directly after {think_close} with tool calls or final response.

### Available Tool Schemas

{schemas}

You MUST strictly follow the above defined tool name and parameter schemas to invoke tool calls.
"#,
        dsml = tokens::DSML_TOKEN,
        think_open = tokens::THINKING_START,
        think_close = tokens::THINKING_END,
        schemas = schemas,
    ))
}

/// Find the index of the last user/developer message.
///
/// Returns `None` when no such message exists. Callers should treat `None`
/// as Python's `-1` sentinel: `idx >= -1` is always true in Python, so use
/// `Option::is_none_or(|u| idx >= u)` (or `>`) to match the reference encoder.
fn find_last_user_index(messages: &[JsonValue]) -> Option<usize> {
    messages
        .iter()
        .enumerate()
        .rev()
        .find(|(_, msg)| {
            msg.get("role")
                .and_then(JsonValue::as_str)
                .is_some_and(|r| matches!(r, "user" | "developer"))
        })
        .map(|(idx, _)| idx)
}

/// Extract visible text from OpenAI-style message content.
///
/// Returns `Result` because the `_ => to_json(content)` fallback can now fail
/// (to_json itself returns `Result`). Callers already sit in `Result` context.
fn extract_visible_text(content: &JsonValue) -> Result<String> {
    Ok(match content {
        JsonValue::String(text) => text.clone(),
        JsonValue::Array(items) => items
            .iter()
            .filter_map(|item| {
                if let Some(text) = item.as_str() {
                    return Some(text.to_string());
                }
                let item_type = item.get("type").and_then(JsonValue::as_str);
                if item_type == Some("text") {
                    return item
                        .get("text")
                        .and_then(JsonValue::as_str)
                        .map(|text| text.to_string());
                }
                tracing::warn!(
                    chunk_type = item_type.unwrap_or("unknown"),
                    "DeepSeek V4 formatter dropped non-text content chunk while normalizing message content",
                );
                None
            })
            .collect::<String>(),
        _ => to_json(content)?,
    })
}

/// Normalize message `content` fields for text-only DeepSeek V4 rendering.
fn normalize_message_contents(messages: &mut [JsonValue]) -> Result<()> {
    for msg in messages {
        let Some(content) = msg.get("content") else {
            continue;
        };
        // Leave non-string/non-array content untouched (null, etc.)
        if !content.is_string() && !content.is_array() {
            continue;
        }
        let normalized = extract_visible_text(content)?;
        if let Some(obj) = msg.as_object_mut() {
            obj.insert("content".to_string(), JsonValue::String(normalized));
        }
    }
    Ok(())
}

/// Encode tool call arguments into DSML parameter format.
fn encode_arguments_to_dsml(tool_call: &JsonValue) -> Result<String> {
    let arguments_str = tool_call
        .get("arguments")
        .and_then(JsonValue::as_str)
        .context("Missing or invalid 'arguments' field")?;

    // Python falls back to `{"arguments": raw_string}` on parse failure.
    let arguments: JsonValue = match serde_json::from_str(arguments_str) {
        Ok(v) => v,
        Err(_) => serde_json::json!({ "arguments": arguments_str }),
    };

    let arguments_obj = arguments
        .as_object()
        .context("Arguments must be a JSON object")?;

    let mut params = Vec::new();
    for (key, value) in arguments_obj {
        // Dispatch on the concrete variant so we don't do the `is_string` / `as_str`
        // / `unwrap` dance (unwrap is technically safe after is_string but fragile
        // against future refactors).
        let (is_string, value_str) = match value {
            JsonValue::String(s) => (true, s.clone()),
            _ => (false, to_json(value)?),
        };
        params.push(format!(
            "<{}parameter name=\"{}\" string=\"{}\">{}</{}parameter>",
            tokens::DSML_TOKEN,
            key,
            if is_string { "true" } else { "false" },
            value_str,
            tokens::DSML_TOKEN
        ));
    }

    Ok(params.join("\n"))
}

/// Lookup the task token for a quick-instruction task.
///
/// Called once per assistant-turn in `render_message` and once per transition
/// token lookup; `#[inline]` because the static-str match is smaller than the
/// call overhead.
#[inline]
fn task_token(task: &str) -> Option<&'static str> {
    match task {
        "action" => Some(tokens::TASK_ACTION),
        "query" => Some(tokens::TASK_QUERY),
        "authority" => Some(tokens::TASK_AUTHORITY),
        "domain" => Some(tokens::TASK_DOMAIN),
        "title" => Some(tokens::TASK_TITLE),
        "read_url" => Some(tokens::TASK_READ_URL),
        _ => None,
    }
}

/// Append the "Response Format" schema block to `prompt` if the message has one.
fn append_response_format(msg: &JsonValue, prompt: &mut String) -> Result<()> {
    if let Some(response_format) = msg.get("response_format") {
        prompt.push_str("\n\n");
        prompt.push_str(&RESPONSE_FORMAT_PREAMBLE.replace("{}", &to_json(response_format)?));
    }
    Ok(())
}

/// Append the tools section to `prompt` if the message has a `tools` array.
fn append_tools_section(msg: &JsonValue, prompt: &mut String) -> Result<()> {
    if let Some(tools) = msg.get("tools").and_then(JsonValue::as_array) {
        prompt.push_str("\n\n");
        prompt.push_str(&render_tools(tools)?);
    }
    Ok(())
}

/// Render the `system` role: raw content, then optional tools + response-format.
fn render_system_role(msg: &JsonValue, prompt: &mut String) -> Result<()> {
    let content = msg.get("content").and_then(JsonValue::as_str).unwrap_or("");
    prompt.push_str(content);
    append_tools_section(msg, prompt)?;
    append_response_format(msg, prompt)?;
    Ok(())
}

/// Render the `developer` role: wraps content in USER_START, then optional
/// tools + response-format. Developer content is required (non-empty).
fn render_developer_role(msg: &JsonValue, prompt: &mut String) -> Result<()> {
    let content = msg
        .get("content")
        .and_then(JsonValue::as_str)
        .filter(|s| !s.is_empty())
        .context("Developer role requires content")?;

    prompt.push_str(tokens::USER_START);
    prompt.push_str(content);
    append_tools_section(msg, prompt)?;
    append_response_format(msg, prompt)?;
    Ok(())
}

/// Render the `user` role: either content-blocks (with tool_result wrapping)
/// or a plain string `content` field.
fn render_user_role(msg: &JsonValue, prompt: &mut String) -> Result<()> {
    prompt.push_str(tokens::USER_START);
    if let Some(blocks) = msg.get("content_blocks").and_then(JsonValue::as_array) {
        let mut parts: Vec<String> = Vec::with_capacity(blocks.len());
        for block in blocks {
            let block_type = block.get("type").and_then(JsonValue::as_str).unwrap_or("");
            match block_type {
                "text" => {
                    let text = block.get("text").and_then(JsonValue::as_str).unwrap_or("");
                    parts.push(text.to_string());
                }
                "tool_result" => {
                    let rendered = render_tool_result_content(
                        block.get("content").unwrap_or(&JsonValue::Null),
                    )?;
                    parts.push(format!(
                        "{}{}{}",
                        TOOL_RESULT_OPEN, rendered, TOOL_RESULT_CLOSE
                    ));
                }
                other => {
                    tracing::warn!(
                        block_type = other,
                        "DeepSeek V4 formatter emitted placeholder for unsupported user content block type",
                    );
                    parts.push(UNSUPPORTED_PLACEHOLDER_FMT.replace("{}", other));
                }
            }
        }
        prompt.push_str(&parts.join("\n\n"));
    } else {
        let content = msg.get("content").and_then(JsonValue::as_str).unwrap_or("");
        prompt.push_str(content);
    }
    Ok(())
}

/// Render the `latest_reminder` role: LATEST_REMINDER token + content.
fn render_latest_reminder_role(msg: &JsonValue, prompt: &mut String) {
    let content = msg.get("content").and_then(JsonValue::as_str).unwrap_or("");
    prompt.push_str(tokens::LATEST_REMINDER);
    prompt.push_str(content);
}

/// Render the `assistant` role: optional thinking prefix, content, optional
/// `tool_calls` DSML block, optional EOS.
///
/// Needs `index` and `messages` to peek at the previous turn's `task` field
/// (suppresses thinking prefix when the prior message was a task message).
fn render_assistant_role(
    msg: &JsonValue,
    index: usize,
    messages: &[JsonValue],
    thinking_mode: ThinkingMode,
    drop_thinking: bool,
    last_user_idx: Option<usize>,
    prompt: &mut String,
) -> Result<()> {
    let content = msg.get("content").and_then(JsonValue::as_str).unwrap_or("");
    let reasoning = msg
        .get("reasoning_content")
        .and_then(JsonValue::as_str)
        .unwrap_or("");
    let wo_eos = msg
        .get("wo_eos")
        .and_then(JsonValue::as_bool)
        .unwrap_or(false);

    let prev_has_task = index > 0
        && messages[index - 1]
            .get("task")
            .map(|v| !v.is_null())
            .unwrap_or(false);

    if thinking_mode == ThinkingMode::Thinking && !prev_has_task {
        let render_thinking = !drop_thinking || last_user_idx.is_none_or(|u| index > u);
        if render_thinking {
            prompt.push_str(reasoning);
            prompt.push_str(tokens::THINKING_END);
        }
    }

    prompt.push_str(content);

    if let Some(tool_calls) = msg.get("tool_calls").and_then(JsonValue::as_array)
        && !tool_calls.is_empty()
    {
        prompt.push_str("\n\n");
        prompt.push_str(&format!(
            "<{}{}>\n",
            tokens::DSML_TOKEN,
            TOOL_CALLS_BLOCK_NAME
        ));

        let mut invocations = Vec::with_capacity(tool_calls.len());
        for tc in tool_calls {
            // Accept both OpenAI-format (nested `function`) and internal
            // `{name, arguments}` shape, matching Python's `tool_calls_from_openai_format`.
            let fn_obj = tc.get("function").unwrap_or(tc);
            let name = fn_obj
                .get("name")
                .and_then(JsonValue::as_str)
                .context("Missing tool call name")?;
            let arguments = encode_arguments_to_dsml(fn_obj)?;
            invocations.push(format!(
                "<{}invoke name=\"{}\">\n{}\n</{}invoke>",
                tokens::DSML_TOKEN,
                name,
                arguments,
                tokens::DSML_TOKEN
            ));
        }
        prompt.push_str(&invocations.join("\n"));
        prompt.push_str(&format!(
            "\n</{}{}>",
            tokens::DSML_TOKEN,
            TOOL_CALLS_BLOCK_NAME
        ));
    }

    if !wo_eos {
        prompt.push_str(tokens::EOS);
    }
    Ok(())
}

/// Render a single message at the given index.
fn render_message(
    index: usize,
    messages: &[JsonValue],
    thinking_mode: ThinkingMode,
    drop_thinking: bool,
    reasoning_effort: Option<ReasoningEffort>,
    last_user_idx: Option<usize>,
) -> Result<String> {
    let msg = &messages[index];

    let role = msg
        .get("role")
        .and_then(JsonValue::as_str)
        .context("Missing 'role' field")?;

    let mut prompt = String::new();

    // Reasoning effort prefix (only at index 0 in thinking mode with max effort).
    if index == 0
        && thinking_mode == ThinkingMode::Thinking
        && reasoning_effort == Some(ReasoningEffort::Max)
    {
        prompt.push_str(REASONING_EFFORT_MAX);
    }

    match role {
        "system" => render_system_role(msg, &mut prompt)?,
        "developer" => render_developer_role(msg, &mut prompt)?,
        "user" => render_user_role(msg, &mut prompt)?,
        "latest_reminder" => render_latest_reminder_role(msg, &mut prompt),
        "tool" => anyhow::bail!(
            "deepseek_v4 merges tool messages into user; preprocess with merge_tool_messages()"
        ),
        "assistant" => render_assistant_role(
            msg,
            index,
            messages,
            thinking_mode,
            drop_thinking,
            last_user_idx,
            &mut prompt,
        )?,
        other => anyhow::bail!("Unknown role: {other}"),
    }

    // Early return if the next message is not assistant/latest_reminder — no transition appended.
    if index + 1 < messages.len() {
        let next_role = messages[index + 1].get("role").and_then(JsonValue::as_str);
        if !matches!(next_role, Some("assistant") | Some("latest_reminder")) {
            return Ok(prompt);
        }
    }

    // Transition tokens based on task field and role.
    let task = msg.get("task").and_then(JsonValue::as_str);
    if let Some(task) = task {
        let sp = task_token(task).with_context(|| format!("Invalid task: '{}'", task))?;
        if task != "action" {
            prompt.push_str(sp);
        } else {
            prompt.push_str(tokens::ASSISTANT_START);
            prompt.push_str(if thinking_mode != ThinkingMode::Thinking {
                tokens::THINKING_END
            } else {
                tokens::THINKING_START
            });
            prompt.push_str(sp);
        }
    } else if matches!(role, "user" | "developer") {
        prompt.push_str(tokens::ASSISTANT_START);
        let seed_thinking = thinking_mode == ThinkingMode::Thinking
            && (!drop_thinking || last_user_idx.is_none_or(|u| index >= u));
        prompt.push_str(if seed_thinking {
            tokens::THINKING_START
        } else {
            tokens::THINKING_END
        });
    }

    Ok(prompt)
}

/// Render a tool_result `content` payload (string or content-block list).
fn render_tool_result_content(content: &JsonValue) -> Result<String> {
    Ok(match content {
        JsonValue::String(s) => s.clone(),
        JsonValue::Array(items) => {
            let mut parts: Vec<String> = Vec::with_capacity(items.len());
            for item in items {
                let item_type = item.get("type").and_then(JsonValue::as_str).unwrap_or("");
                if item_type == "text" {
                    parts.push(
                        item.get("text")
                            .and_then(JsonValue::as_str)
                            .unwrap_or("")
                            .to_string(),
                    );
                } else {
                    tracing::warn!(
                        item_type,
                        "DeepSeek V4 formatter emitted placeholder for unsupported tool_result content item type",
                    );
                    parts.push(UNSUPPORTED_PLACEHOLDER_FMT.replace("{}", item_type));
                }
            }
            parts.join("\n\n")
        }
        JsonValue::Null => String::new(),
        _ => to_json(content)?,
    })
}

/// Merge `tool` role messages into preceding user `content_blocks` and collapse
/// consecutive user turns, matching Python's `merge_tool_messages`.
///
/// Iterates the input by reference. Each message is cloned at most once — and
/// only when the control flow actually moves it into `merged` (the "other
/// role" pass-through branch). The `tool` and `user` branches extract the few
/// fields they need and build fresh JSON objects, so cloning the whole input
/// message up front (as the original implementation did) was pure overhead
/// on long chat histories.
pub fn merge_tool_messages(messages: &[JsonValue]) -> Vec<JsonValue> {
    let mut merged: Vec<JsonValue> = Vec::with_capacity(messages.len());

    for msg in messages {
        let role = msg.get("role").and_then(JsonValue::as_str).unwrap_or("");

        if role == "tool" {
            let tool_block = serde_json::json!({
                "type": "tool_result",
                "tool_use_id": msg.get("tool_call_id").cloned().unwrap_or(JsonValue::String(String::new())),
                "content": msg.get("content").cloned().unwrap_or(JsonValue::String(String::new())),
            });

            let can_merge = merged
                .last()
                .map(|m| {
                    m.get("role").and_then(JsonValue::as_str) == Some("user")
                        && m.get("content_blocks").is_some()
                })
                .unwrap_or(false);

            if can_merge {
                // `can_merge` already checked `content_blocks.is_some()`; if
                // the subsequent `as_array_mut()` ever fails (data invariant
                // violation) we match the original behavior and silently
                // drop rather than falling through to push a new user msg.
                if let Some(last) = merged.last_mut()
                    && let Some(blocks) = last
                        .as_object_mut()
                        .and_then(|o| o.get_mut("content_blocks"))
                        .and_then(JsonValue::as_array_mut)
                {
                    blocks.push(tool_block);
                }
            } else {
                merged.push(serde_json::json!({
                    "role": "user",
                    "content_blocks": [tool_block],
                }));
            }
        } else if role == "user" {
            let text = msg
                .get("content")
                .and_then(JsonValue::as_str)
                .unwrap_or("")
                .to_string();
            let text_block = serde_json::json!({ "type": "text", "text": text });

            let can_merge = merged
                .last()
                .map(|m| {
                    m.get("role").and_then(JsonValue::as_str) == Some("user")
                        && m.get("content_blocks").is_some()
                        && m.get("task").map(|v| v.is_null()).unwrap_or(true)
                })
                .unwrap_or(false);

            if can_merge {
                if let Some(last) = merged.last_mut()
                    && let Some(blocks) = last
                        .as_object_mut()
                        .and_then(|o| o.get_mut("content_blocks"))
                        .and_then(JsonValue::as_array_mut)
                {
                    blocks.push(text_block);
                }
            } else {
                let mut new_msg = serde_json::json!({
                    "role": "user",
                    "content": text,
                    "content_blocks": [text_block],
                });
                // Preserve extra fields.
                if let Some(obj) = new_msg.as_object_mut() {
                    for key in ["task", "wo_eos", "mask"] {
                        if let Some(v) = msg.get(key) {
                            obj.insert(key.to_string(), v.clone());
                        }
                    }
                }
                merged.push(new_msg);
            }
        } else {
            // Pass-through: clone only when we're actually moving the message.
            merged.push(msg.clone());
        }
    }

    merged
}

/// Sort `tool_result` blocks within user messages by the `tool_calls[].id` order
/// of the preceding assistant message.
pub fn sort_tool_results_by_call_order(mut messages: Vec<JsonValue>) -> Vec<JsonValue> {
    use std::collections::HashMap;
    let mut last_order: HashMap<String, usize> = HashMap::new();

    for msg in &mut messages {
        let role = msg.get("role").and_then(JsonValue::as_str).unwrap_or("");
        if role == "assistant" {
            if let Some(tcs) = msg.get("tool_calls").and_then(JsonValue::as_array) {
                last_order.clear();
                for (idx, tc) in tcs.iter().enumerate() {
                    let id = tc
                        .get("id")
                        .and_then(JsonValue::as_str)
                        .or_else(|| {
                            tc.get("function")
                                .and_then(|f| f.get("id"))
                                .and_then(JsonValue::as_str)
                        })
                        .unwrap_or("");
                    if !id.is_empty() {
                        last_order.insert(id.to_string(), idx);
                    }
                }
            }
        } else if role == "user" && !last_order.is_empty() {
            let Some(blocks) = msg
                .as_object_mut()
                .and_then(|o| o.get_mut("content_blocks"))
                .and_then(JsonValue::as_array_mut)
            else {
                continue;
            };

            // Collect tool_result blocks with their positions.
            let tool_positions: Vec<usize> = blocks
                .iter()
                .enumerate()
                .filter(|(_, b)| b.get("type").and_then(JsonValue::as_str) == Some("tool_result"))
                .map(|(i, _)| i)
                .collect();

            if tool_positions.len() > 1 {
                let mut tool_blocks: Vec<JsonValue> =
                    tool_positions.iter().map(|&i| blocks[i].clone()).collect();
                tool_blocks.sort_by_key(|b| {
                    let id = b
                        .get("tool_use_id")
                        .and_then(JsonValue::as_str)
                        .unwrap_or("");
                    *last_order.get(id).unwrap_or(&0)
                });
                for (sorted_idx, &pos) in tool_positions.iter().enumerate() {
                    blocks[pos] = tool_blocks[sorted_idx].clone();
                }
            }
        }
    }

    messages
}

/// Drop reasoning and non-essential messages before the last user message.
///
/// Takes `last_user_idx` (the pre-drop position of the last user/developer
/// message, as returned by [`find_last_user_index`]) from the caller so we
/// don't rescan the vector; returns the post-drop position of the same
/// message so the caller doesn't have to rescan either.
fn drop_thinking_messages(
    messages: Vec<JsonValue>,
    last_user_idx: Option<usize>,
) -> (Vec<JsonValue>, Option<usize>) {
    let mut out = Vec::with_capacity(messages.len());
    let mut new_last_user_idx: Option<usize> = None;
    for (idx, mut msg) in messages.into_iter().enumerate() {
        let role = msg.get("role").and_then(JsonValue::as_str).unwrap_or("");
        if KEEP_ROLES.contains(&role) || last_user_idx.is_none_or(|u| idx >= u) {
            if last_user_idx == Some(idx) {
                new_last_user_idx = Some(out.len());
            }
            out.push(msg);
        } else if role == "assistant" {
            if let Some(obj) = msg.as_object_mut() {
                obj.remove("reasoning_content");
            }
            out.push(msg);
        }
        // developer and other roles before last_user_idx are dropped.
    }
    (out, new_last_user_idx)
}

/// Encode messages to prompt string with default options.
///
/// Equivalent to `encode_messages_with_options(.., drop_thinking=true, reasoning_effort=None)`.
pub fn encode_messages(
    messages: &[JsonValue],
    thinking_mode: ThinkingMode,
    add_bos_token: bool,
) -> Result<String> {
    encode_messages_with_options(messages, thinking_mode, add_bos_token, true, None)
}

/// Encode messages to prompt string.
///
/// # Arguments
/// * `messages` - Array of messages in OpenAI format
/// * `thinking_mode` - Chat or Thinking
/// * `add_bos_token` - Whether to prepend BOS token
/// * `drop_thinking` - Drop reasoning_content from earlier turns (auto-disabled if tools present)
/// * `reasoning_effort` - Optional reasoning effort level (Max prepends a verbatim block)
pub fn encode_messages_with_options(
    messages: &[JsonValue],
    thinking_mode: ThinkingMode,
    add_bos_token: bool,
    drop_thinking: bool,
    reasoning_effort: Option<ReasoningEffort>,
) -> Result<String> {
    let merged = merge_tool_messages(messages);
    let mut full = sort_tool_results_by_call_order(merged);

    let mut prompt = String::new();
    if add_bos_token {
        prompt.push_str(tokens::BOS);
    }

    // Auto-disable drop_thinking when any message carries a `tools` field.
    let has_tools = full.iter().any(|m| {
        m.get("tools")
            .map(|v| match v {
                JsonValue::Array(a) => !a.is_empty(),
                JsonValue::Null => false,
                _ => true,
            })
            .unwrap_or(false)
    });
    let effective_drop_thinking = drop_thinking && !has_tools;

    // Locate the last user/developer message once. `drop_thinking_messages`
    // both consumes this (to know what to keep) and returns the new index
    // (to avoid a second full scan over its output list).
    let mut last_user_idx = find_last_user_index(&full);

    if thinking_mode == ThinkingMode::Thinking && effective_drop_thinking {
        let (dropped, new_last_user) = drop_thinking_messages(full, last_user_idx);
        full = dropped;
        last_user_idx = new_last_user;
    }

    for idx in 0..full.len() {
        let part = render_message(
            idx,
            &full,
            thinking_mode,
            effective_drop_thinking,
            reasoning_effort,
            last_user_idx,
        )?;
        prompt.push_str(&part);
    }

    Ok(prompt)
}

