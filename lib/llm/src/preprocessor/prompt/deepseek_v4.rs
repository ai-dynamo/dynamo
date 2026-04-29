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

const TOOL_CALLS_BLOCK_NAME: &str = "tool_calls";

const RESPONSE_FORMAT_TEMPLATE: &str =
    "## Response Format:\n\nYou MUST strictly adhere to the following schema to reply:\n{schema}";

const TOOLS_TEMPLATE: &str = r#"## Tools

You have access to a set of tools to help answer the user's question. You can invoke tools by writing a "<{dsml_token}tool_calls>" block like the following:

<{dsml_token}tool_calls>
<{dsml_token}invoke name="$TOOL_NAME">
<{dsml_token}parameter name="$PARAMETER_NAME" string="true|false">$PARAMETER_VALUE</{dsml_token}parameter>
...
</{dsml_token}invoke>
<{dsml_token}invoke name="$TOOL_NAME2">
...
</{dsml_token}invoke>
</{dsml_token}tool_calls>

String parameters should be specified as is and set `string="true"`. For all other types (numbers, booleans, arrays, objects), pass the value in JSON format and set `string="false"`.

If thinking_mode is enabled (triggered by {thinking_start_token}), you MUST output your complete reasoning inside {thinking_start_token}...{thinking_end_token} BEFORE any tool calls or final response.

Otherwise, output directly after {thinking_end_token} with tool calls or final response.

### Available Tool Schemas

{tool_schemas}

You MUST strictly follow the above defined tool name and parameter schemas to invoke tool calls.
"#;

const REASONING_EFFORT_MAX: &str = "Reasoning Effort: Absolute maximum with no shortcuts permitted.\nYou MUST be very thorough in your thinking and comprehensively decompose the problem to resolve the root cause, rigorously stress-testing your logic against all potential paths, edge cases, and adversarial scenarios.\nExplicitly write out your entire deliberation process, documenting every intermediate step, considered alternative, and rejected hypothesis to ensure absolutely no assumption is left unchecked.\n\n";

/// Thinking mode for the model
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ThinkingMode {
    Chat,
    Thinking,
}

impl ThinkingMode {
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
fn to_json(value: &JsonValue) -> String {
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

    let mut buf = Vec::with_capacity(64);
    let mut ser = serde_json::Serializer::with_formatter(&mut buf, PythonFormatter);
    if let Err(e) = value.serialize(&mut ser) {
        tracing::warn!(error = %e, "to_json: serialize failed; falling back to empty object");
        return "{}".to_string();
    }
    String::from_utf8(buf).unwrap_or_else(|e| {
        tracing::warn!(error = %e, "to_json: serialized output not valid UTF-8; falling back to empty object");
        "{}".to_string()
    })
}

/// Extract function definitions from OpenAI-format tool list.
fn tools_from_openai_format(tools: &[JsonValue]) -> Vec<JsonValue> {
    tools
        .iter()
        .filter_map(|tool| tool.get("function").cloned())
        .collect()
}

/// Render tool schemas into the system prompt format.
fn render_tools(tools: &[JsonValue]) -> String {
    let tools_json: Vec<String> = tools_from_openai_format(tools)
        .iter()
        .map(to_json)
        .collect();

    TOOLS_TEMPLATE
        .replace("{tool_schemas}", &tools_json.join("\n"))
        .replace("{dsml_token}", tokens::DSML_TOKEN)
        .replace("{thinking_start_token}", tokens::THINKING_START)
        .replace("{thinking_end_token}", tokens::THINKING_END)
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
                .and_then(|r| r.as_str())
                .map(|r| r == "user" || r == "developer")
                .unwrap_or(false)
        })
        .map(|(idx, _)| idx)
}

/// Extract visible text from OpenAI-style message content.
fn extract_visible_text(content: &JsonValue) -> String {
    match content {
        JsonValue::String(text) => text.clone(),
        JsonValue::Array(items) => items
            .iter()
            .filter_map(|item| {
                if let Some(text) = item.as_str() {
                    return Some(text.to_string());
                }
                let item_type = item.get("type").and_then(|v| v.as_str());
                if item_type == Some("text") {
                    return item
                        .get("text")
                        .and_then(|v| v.as_str())
                        .map(|text| text.to_string());
                }
                tracing::warn!(
                    chunk_type = item_type.unwrap_or("unknown"),
                    "DeepSeek V4 formatter dropped non-text content chunk while normalizing message content",
                );
                None
            })
            .collect::<String>(),
        _ => to_json(content),
    }
}

/// Normalize message `content` fields for text-only DeepSeek V4 rendering.
fn normalize_message_contents(messages: &mut [JsonValue]) {
    for msg in messages {
        let Some(content) = msg.get("content") else {
            continue;
        };
        // Leave non-string/non-array content untouched (null, etc.)
        if !content.is_string() && !content.is_array() {
            continue;
        }
        let normalized = extract_visible_text(content);
        if let Some(obj) = msg.as_object_mut() {
            obj.insert("content".to_string(), JsonValue::String(normalized));
        }
    }
}

/// Encode tool call arguments into DSML parameter format.
fn encode_arguments_to_dsml(tool_call: &JsonValue) -> Result<String> {
    let arguments_str = tool_call
        .get("arguments")
        .and_then(|a| a.as_str())
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
        let is_string = value.is_string();
        let value_str = if is_string {
            value.as_str().unwrap().to_string()
        } else {
            to_json(value)
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
        .and_then(|r| r.as_str())
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
        "system" => {
            let content = msg.get("content").and_then(|c| c.as_str()).unwrap_or("");
            prompt.push_str(content);
            if let Some(tools) = msg.get("tools").and_then(|t| t.as_array()) {
                prompt.push_str("\n\n");
                prompt.push_str(&render_tools(tools));
            }
            if let Some(response_format) = msg.get("response_format") {
                prompt.push_str("\n\n");
                prompt.push_str(
                    &RESPONSE_FORMAT_TEMPLATE.replace("{schema}", &to_json(response_format)),
                );
            }
        }

        "developer" => {
            let content = msg
                .get("content")
                .and_then(|c| c.as_str())
                .filter(|s| !s.is_empty())
                .context("Developer role requires content")?;

            let mut content_developer = String::from(tokens::USER_START);
            content_developer.push_str(content);

            if let Some(tools) = msg.get("tools").and_then(|t| t.as_array()) {
                content_developer.push_str("\n\n");
                content_developer.push_str(&render_tools(tools));
            }
            if let Some(response_format) = msg.get("response_format") {
                content_developer.push_str("\n\n");
                content_developer.push_str(
                    &RESPONSE_FORMAT_TEMPLATE.replace("{schema}", &to_json(response_format)),
                );
            }
            prompt.push_str(&content_developer);
        }

        "user" => {
            prompt.push_str(tokens::USER_START);
            if let Some(blocks) = msg.get("content_blocks").and_then(|b| b.as_array()) {
                let mut parts: Vec<String> = Vec::with_capacity(blocks.len());
                for block in blocks {
                    let block_type = block.get("type").and_then(|v| v.as_str()).unwrap_or("");
                    match block_type {
                        "text" => {
                            let text = block.get("text").and_then(|v| v.as_str()).unwrap_or("");
                            parts.push(text.to_string());
                        }
                        "tool_result" => {
                            let rendered = render_tool_result_content(
                                block.get("content").unwrap_or(&JsonValue::Null),
                            );
                            parts.push(format!("<tool_result>{}</tool_result>", rendered));
                        }
                        other => {
                            parts.push(format!("[Unsupported {}]", other));
                        }
                    }
                }
                prompt.push_str(&parts.join("\n\n"));
            } else {
                let content = msg.get("content").and_then(|c| c.as_str()).unwrap_or("");
                prompt.push_str(content);
            }
        }

        "latest_reminder" => {
            let content = msg.get("content").and_then(|c| c.as_str()).unwrap_or("");
            prompt.push_str(tokens::LATEST_REMINDER);
            prompt.push_str(content);
        }

        "tool" => {
            anyhow::bail!(
                "deepseek_v4 merges tool messages into user; preprocess with merge_tool_messages()"
            );
        }

        "assistant" => {
            let content = msg.get("content").and_then(|c| c.as_str()).unwrap_or("");
            let reasoning = msg
                .get("reasoning_content")
                .and_then(|c| c.as_str())
                .unwrap_or("");
            let wo_eos = msg.get("wo_eos").and_then(|v| v.as_bool()).unwrap_or(false);

            let prev_has_task = index > 0
                && messages[index - 1]
                    .get("task")
                    .map(|v| !v.is_null())
                    .unwrap_or(false);

            let mut thinking_part = String::new();
            if thinking_mode == ThinkingMode::Thinking && !prev_has_task {
                let render_thinking = !drop_thinking || last_user_idx.is_none_or(|u| index > u);
                if render_thinking {
                    thinking_part.push_str(reasoning);
                    thinking_part.push_str(tokens::THINKING_END);
                }
            }

            prompt.push_str(&thinking_part);
            prompt.push_str(content);

            if let Some(tool_calls) = msg.get("tool_calls").and_then(|t| t.as_array())
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
                        .and_then(|n| n.as_str())
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
        }

        other => anyhow::bail!("Unknown role: {}", other),
    }

    // Early return if the next message is not assistant/latest_reminder — no transition appended.
    if index + 1 < messages.len() {
        let next_role = messages[index + 1].get("role").and_then(|r| r.as_str());
        if !matches!(next_role, Some("assistant") | Some("latest_reminder")) {
            return Ok(prompt);
        }
    }

    // Transition tokens based on task field and role.
    let task = msg.get("task").and_then(|v| v.as_str());
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
fn render_tool_result_content(content: &JsonValue) -> String {
    match content {
        JsonValue::String(s) => s.clone(),
        JsonValue::Array(items) => {
            let mut parts: Vec<String> = Vec::with_capacity(items.len());
            for item in items {
                let item_type = item.get("type").and_then(|v| v.as_str()).unwrap_or("");
                if item_type == "text" {
                    parts.push(
                        item.get("text")
                            .and_then(|v| v.as_str())
                            .unwrap_or("")
                            .to_string(),
                    );
                } else {
                    parts.push(format!("[Unsupported {}]", item_type));
                }
            }
            parts.join("\n\n")
        }
        JsonValue::Null => String::new(),
        _ => to_json(content),
    }
}

/// Merge `tool` role messages into preceding user `content_blocks` and collapse
/// consecutive user turns, matching Python's `merge_tool_messages`.
pub fn merge_tool_messages(messages: &[JsonValue]) -> Vec<JsonValue> {
    let mut merged: Vec<JsonValue> = Vec::with_capacity(messages.len());

    for msg in messages {
        let msg = msg.clone();
        let role = msg.get("role").and_then(|r| r.as_str()).unwrap_or("");

        if role == "tool" {
            let tool_block = serde_json::json!({
                "type": "tool_result",
                "tool_use_id": msg.get("tool_call_id").cloned().unwrap_or(JsonValue::String(String::new())),
                "content": msg.get("content").cloned().unwrap_or(JsonValue::String(String::new())),
            });

            let can_merge = merged
                .last()
                .map(|m| {
                    m.get("role").and_then(|r| r.as_str()) == Some("user")
                        && m.get("content_blocks").is_some()
                })
                .unwrap_or(false);

            if can_merge {
                let last = merged.last_mut().unwrap();
                if let Some(blocks) = last
                    .as_object_mut()
                    .and_then(|o| o.get_mut("content_blocks"))
                    .and_then(|v| v.as_array_mut())
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
                .and_then(|c| c.as_str())
                .unwrap_or("")
                .to_string();
            let text_block = serde_json::json!({ "type": "text", "text": text });

            let can_merge = merged
                .last()
                .map(|m| {
                    m.get("role").and_then(|r| r.as_str()) == Some("user")
                        && m.get("content_blocks").is_some()
                        && m.get("task").map(|v| v.is_null()).unwrap_or(true)
                })
                .unwrap_or(false);

            if can_merge {
                let last = merged.last_mut().unwrap();
                if let Some(blocks) = last
                    .as_object_mut()
                    .and_then(|o| o.get_mut("content_blocks"))
                    .and_then(|v| v.as_array_mut())
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
            merged.push(msg);
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
        let role = msg.get("role").and_then(|r| r.as_str()).unwrap_or("");
        if role == "assistant" {
            if let Some(tcs) = msg.get("tool_calls").and_then(|t| t.as_array()) {
                last_order.clear();
                for (idx, tc) in tcs.iter().enumerate() {
                    let id = tc
                        .get("id")
                        .and_then(|v| v.as_str())
                        .or_else(|| {
                            tc.get("function")
                                .and_then(|f| f.get("id"))
                                .and_then(|v| v.as_str())
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
                .and_then(|v| v.as_array_mut())
            else {
                continue;
            };

            // Collect tool_result blocks with their positions.
            let tool_positions: Vec<usize> = blocks
                .iter()
                .enumerate()
                .filter(|(_, b)| b.get("type").and_then(|v| v.as_str()) == Some("tool_result"))
                .map(|(i, _)| i)
                .collect();

            if tool_positions.len() > 1 {
                let mut tool_blocks: Vec<JsonValue> =
                    tool_positions.iter().map(|&i| blocks[i].clone()).collect();
                tool_blocks.sort_by_key(|b| {
                    let id = b.get("tool_use_id").and_then(|v| v.as_str()).unwrap_or("");
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
fn drop_thinking_messages(messages: Vec<JsonValue>) -> Vec<JsonValue> {
    let last_user_idx = find_last_user_index(&messages);
    let mut out = Vec::with_capacity(messages.len());
    const KEEP: &[&str] = &[
        "user",
        "system",
        "tool",
        "latest_reminder",
        "direct_search_results",
    ];

    for (idx, mut msg) in messages.into_iter().enumerate() {
        let role = msg.get("role").and_then(|r| r.as_str()).unwrap_or("");
        if KEEP.contains(&role) || last_user_idx.is_none_or(|u| idx >= u) {
            out.push(msg);
        } else if role == "assistant" {
            if let Some(obj) = msg.as_object_mut() {
                obj.remove("reasoning_content");
            }
            out.push(msg);
        }
        // developer and other roles before last_user_idx are dropped.
    }
    out
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

    if thinking_mode == ThinkingMode::Thinking && effective_drop_thinking {
        full = drop_thinking_messages(full);
    }

    let last_user_idx = find_last_user_index(&full);
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

/// DeepSeek V4 Prompt Formatter
#[derive(Debug)]
pub struct DeepSeekV4Formatter {
    thinking_mode: ThinkingMode,
}

impl DeepSeekV4Formatter {
    pub fn new(thinking_mode: ThinkingMode) -> Self {
        Self { thinking_mode }
    }

    /// Create formatter with thinking mode enabled (default for DSV4)
    pub fn new_thinking() -> Self {
        Self::new(ThinkingMode::Thinking)
    }

    /// Create formatter with chat mode
    pub fn new_chat() -> Self {
        Self::new(ThinkingMode::Chat)
    }

    fn resolve_reasoning_effort(
        args: Option<&std::collections::HashMap<String, serde_json::Value>>,
    ) -> Option<ReasoningEffort> {
        let args = args?;
        let v = args.get("reasoning_effort")?;
        match v.as_str() {
            Some("max") => Some(ReasoningEffort::Max),
            Some("high") => Some(ReasoningEffort::High),
            _ => {
                tracing::warn!(
                    value = ?v,
                    "chat_template_args.reasoning_effort must be a string of \"max\" or \"high\"; ignoring and using default (none)"
                );
                None
            }
        }
    }

    fn resolve_drop_thinking(
        args: Option<&std::collections::HashMap<String, serde_json::Value>>,
    ) -> bool {
        let Some(args) = args else { return true };
        let Some(v) = args.get("drop_thinking") else {
            return true;
        };
        if let Some(b) = v.as_bool() {
            return b;
        }
        tracing::warn!(
            value = ?v,
            "chat_template_args.drop_thinking must be a bool; ignoring and using default (true)"
        );
        true
    }

    fn resolve_thinking_mode(
        &self,
        args: Option<&std::collections::HashMap<String, serde_json::Value>>,
    ) -> ThinkingMode {
        if let Some(enabled) = super::thinking_bool_from_args(args) {
            return if enabled {
                ThinkingMode::Thinking
            } else {
                ThinkingMode::Chat
            };
        }
        if let Some(args) = args
            && let Some(mode) = args.get("thinking_mode").and_then(|v| v.as_str())
        {
            match mode {
                "chat" => return ThinkingMode::Chat,
                "thinking" => return ThinkingMode::Thinking,
                _ => {}
            }
        }
        self.thinking_mode
    }
}

impl super::OAIPromptFormatter for DeepSeekV4Formatter {
    fn supports_add_generation_prompt(&self) -> bool {
        true
    }

    fn render(&self, req: &dyn super::OAIChatLikeRequest) -> Result<String> {
        let args = req.chat_template_args();
        let thinking_mode = self.resolve_thinking_mode(args);
        let reasoning_effort = Self::resolve_reasoning_effort(args);
        let drop_thinking = Self::resolve_drop_thinking(args);

        let messages_value = req.messages();
        let messages_json =
            serde_json::to_value(&messages_value).context("Failed to convert messages to JSON")?;

        let mut messages_array = messages_json
            .as_array()
            .context("Messages is not an array")?
            .clone();

        normalize_message_contents(&mut messages_array);

        let tools_json = req
            .tools()
            .map(|t| serde_json::to_value(&t))
            .transpose()
            .context("Failed to convert tools to JSON")?;

        let response_format_json = req
            .response_format()
            .map(|rf| serde_json::to_value(&rf))
            .transpose()
            .context("Failed to convert response_format to JSON")?;

        if tools_json.is_some() || response_format_json.is_some() {
            let system_idx = messages_array
                .iter()
                .position(|msg| msg.get("role").and_then(|r| r.as_str()) == Some("system"));

            if let Some(idx) = system_idx {
                if let Some(msg) = messages_array.get_mut(idx)
                    && let Some(obj) = msg.as_object_mut()
                {
                    if let Some(tools) = tools_json {
                        obj.insert("tools".to_string(), tools);
                    }
                    if let Some(rf) = response_format_json {
                        obj.insert("response_format".to_string(), rf);
                    }
                }
            } else {
                let mut system_msg = serde_json::json!({
                    "role": "system",
                    "content": ""
                });
                if let Some(obj) = system_msg.as_object_mut() {
                    if let Some(tools) = tools_json {
                        obj.insert("tools".to_string(), tools);
                    }
                    if let Some(rf) = response_format_json {
                        obj.insert("response_format".to_string(), rf);
                    }
                }
                messages_array.insert(0, system_msg);
            }
        }

        encode_messages_with_options(
            &messages_array,
            thinking_mode,
            true,
            drop_thinking,
            reasoning_effort,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn test_simple_conversation() {
        let messages = json!([
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "reasoning_content": "greet", "content": "Hi!"},
            {"role": "user", "content": "What is 2+2?"}
        ]);
        let out =
            encode_messages(messages.as_array().unwrap(), ThinkingMode::Thinking, true).unwrap();
        assert!(out.starts_with(tokens::BOS));
        assert!(out.ends_with(&format!(
            "{}{}",
            tokens::ASSISTANT_START,
            tokens::THINKING_START
        )));
        // drop_thinking default true → earlier reasoning stripped
        assert!(!out.contains("greet"));
    }

    #[test]
    fn test_reasoning_effort_max_prefix() {
        let messages = json!([
            {"role": "system", "content": "hi"},
            {"role": "user", "content": "hello"}
        ]);
        let out = encode_messages_with_options(
            messages.as_array().unwrap(),
            ThinkingMode::Thinking,
            true,
            true,
            Some(ReasoningEffort::Max),
        )
        .unwrap();
        assert!(out.contains("Reasoning Effort: Absolute maximum"));
        // Prefix comes between BOS and system content.
        let after_bos = &out[tokens::BOS.len()..];
        assert!(after_bos.starts_with("Reasoning Effort:"));

        // High and None do not emit the prefix.
        let out2 = encode_messages_with_options(
            messages.as_array().unwrap(),
            ThinkingMode::Thinking,
            true,
            true,
            Some(ReasoningEffort::High),
        )
        .unwrap();
        assert!(!out2.contains("Reasoning Effort: Absolute maximum"));
    }

    #[test]
    fn test_content_blocks_with_tool_result() {
        // `merge_tool_messages` turns a `tool` role followed by a plain user text
        // into a single user turn whose `content_blocks` interleave the tool result
        // with the text, joined by "\n\n" at render time. Users don't construct
        // `content_blocks` directly — both the Python reference and this port
        // overwrite any user-supplied `content_blocks` with a single text block.
        let messages = json!([
            {"role": "user", "content": "call tool"},
            {"role": "assistant", "content": "", "tool_calls": [{
                "id": "c1", "type": "function",
                "function": {"name": "f", "arguments": "{}"}
            }]},
            {"role": "tool", "tool_call_id": "c1", "content": "RESULT"},
            {"role": "user", "content": "thanks"}
        ]);
        let out = encode_messages(messages.as_array().unwrap(), ThinkingMode::Chat, true).unwrap();
        assert!(
            out.contains("<tool_result>RESULT</tool_result>\n\nthanks"),
            "expected tool_result block followed by 'thanks' in the merged user turn, got:\n{}",
            out
        );
    }

    #[test]
    fn test_drop_thinking_auto_disable_when_tools_present() {
        let messages = json!([
            {"role": "system", "content": "s", "tools": [{
                "type": "function",
                "function": {"name": "f", "description": "", "parameters": {"type": "object", "properties": {}}}
            }]},
            {"role": "user", "content": "hi"},
            {"role": "assistant", "reasoning_content": "PRIOR_REASONING", "content": "reply"},
            {"role": "user", "content": "again"}
        ]);
        let out =
            encode_messages(messages.as_array().unwrap(), ThinkingMode::Thinking, true).unwrap();
        // Tools present → drop_thinking auto-disabled → earlier reasoning preserved.
        assert!(out.contains("PRIOR_REASONING"));
    }

    // ---- Regression tests for known divergences from the Python reference ----

    /// Bug: `last_user_idx = None` (no user/developer in history) should behave
    /// like Python's `-1` sentinel — `index >= -1` / `idx >= -1` always true, so
    /// earlier reasoning is preserved and the assistant's reasoning block is
    /// rendered. Rust defaulting `None` to `usize::MAX` / `is_some_and` silently
    /// stripped reasoning instead.
    ///
    /// Byte-equivalent to Python reference with the same input:
    /// `<BOS>sysREASONING_BLOCK</think>hello<EOS>`
    #[test]
    fn test_assistant_reasoning_preserved_when_no_user_in_history() {
        let messages = json!([
            {"role": "system", "content": "sys"},
            {"role": "assistant", "content": "hello", "reasoning_content": "REASONING_BLOCK"}
        ]);
        let out =
            encode_messages(messages.as_array().unwrap(), ThinkingMode::Thinking, true).unwrap();
        assert_eq!(
            out, "<｜begin▁of▁sentence｜>sysREASONING_BLOCK</think>hello<｜end▁of▁sentence｜>",
            "Output must match Python reference byte-for-byte when no user/developer in history"
        );
    }

    /// Bug: `to_json` tracks in-string state via `prev_char != '\\'` which
    /// mis-handles consecutive backslashes. A value containing `\\` (one literal
    /// backslash in JSON) makes the helper think the closing `"` is escaped,
    /// so it stops inserting Python-compatible spaces after subsequent `:`/`,`.
    ///
    /// Python `json.dumps({"path": "\\", "count": 5}, ensure_ascii=False)`
    /// emits `{"path": "\\", "count": 5}` — space after every `:` and `,`.
    #[test]
    fn test_to_json_preserves_spacing_past_escaped_backslash() {
        let v = json!({"path": "\\", "count": 5});
        let got = to_json(&v);
        assert_eq!(
            got, r#"{"path": "\\", "count": 5}"#,
            "to_json must match Python's json.dumps formatting past an escaped backslash"
        );
    }

    #[test]
    fn test_resolve_drop_thinking_warns_on_malformed_value() {
        use std::collections::HashMap;
        // String "false" where a bool is expected → fall back to default (true) and warn.
        let mut args = HashMap::new();
        args.insert(
            "drop_thinking".to_string(),
            serde_json::Value::String("false".to_string()),
        );
        assert!(DeepSeekV4Formatter::resolve_drop_thinking(Some(&args)));
        // Malformed reasoning_effort falls back to None.
        let mut args2 = HashMap::new();
        args2.insert(
            "reasoning_effort".to_string(),
            serde_json::Value::String("HIGH".to_string()),
        );
        assert_eq!(
            DeepSeekV4Formatter::resolve_reasoning_effort(Some(&args2)),
            None
        );
    }

    #[test]
    fn test_resolve_thinking_mode_honors_enable_thinking() {
        use std::collections::HashMap;
        let f = DeepSeekV4Formatter::new_thinking();
        let mut args = HashMap::new();
        args.insert(
            "enable_thinking".to_string(),
            serde_json::Value::Bool(false),
        );
        assert_eq!(f.resolve_thinking_mode(Some(&args)), ThinkingMode::Chat);
        args.insert("enable_thinking".to_string(), serde_json::Value::Bool(true));
        assert_eq!(f.resolve_thinking_mode(Some(&args)), ThinkingMode::Thinking);
    }

    struct MockRequest {
        messages: JsonValue,
        chat_template_args: Option<std::collections::HashMap<String, JsonValue>>,
    }

    impl MockRequest {
        fn new(messages: JsonValue) -> Self {
            Self {
                messages,
                chat_template_args: None,
            }
        }

        fn with_chat_template_args(
            mut self,
            args: std::collections::HashMap<String, JsonValue>,
        ) -> Self {
            self.chat_template_args = Some(args);
            self
        }
    }

    impl super::super::OAIChatLikeRequest for MockRequest {
        fn model(&self) -> String {
            "deepseek-v4".to_string()
        }

        fn messages(&self) -> minijinja::value::Value {
            minijinja::value::Value::from_serialize(&self.messages)
        }

        fn should_add_generation_prompt(&self) -> bool {
            true
        }

        fn chat_template_args(
            &self,
        ) -> Option<&std::collections::HashMap<String, serde_json::Value>> {
            self.chat_template_args.as_ref()
        }
    }

    #[test]
    fn test_render_wires_reasoning_effort_max_from_chat_template_args() {
        use super::super::OAIPromptFormatter;
        use std::collections::HashMap;

        let mut args = HashMap::new();
        args.insert("reasoning_effort".to_string(), json!("max"));

        let req = MockRequest::new(json!([
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "hi"}
        ]))
        .with_chat_template_args(args);

        let formatter = DeepSeekV4Formatter::new_thinking();
        let out = formatter.render(&req).unwrap();

        assert!(out.starts_with(tokens::BOS));
        let after_bos = &out[tokens::BOS.len()..];
        assert!(
            after_bos.starts_with("Reasoning Effort:"),
            "REASONING_EFFORT_MAX preamble should appear at start (after BOS), got:\n{}",
            out
        );
    }

    #[test]
    fn test_render_drop_thinking_override_from_chat_template_args() {
        use super::super::OAIPromptFormatter;
        use std::collections::HashMap;

        let messages = json!([
            {"role": "user", "content": "first"},
            {"role": "assistant", "reasoning_content": "PRIOR", "content": "reply"},
            {"role": "user", "content": "again"}
        ]);

        // Default (drop_thinking=true): prior reasoning stripped.
        let req_default = MockRequest::new(messages.clone());
        let formatter = DeepSeekV4Formatter::new_thinking();
        let out_default = formatter.render(&req_default).unwrap();
        assert!(
            !out_default.contains("PRIOR"),
            "default drop_thinking=true should strip prior reasoning, got:\n{}",
            out_default
        );

        // drop_thinking=false override: prior reasoning survives.
        let mut args = HashMap::new();
        args.insert("drop_thinking".to_string(), json!(false));
        let req_keep = MockRequest::new(messages).with_chat_template_args(args);
        let out_keep = formatter.render(&req_keep).unwrap();
        assert!(
            out_keep.contains("PRIOR"),
            "drop_thinking=false override should preserve prior reasoning, got:\n{}",
            out_keep
        );
    }

    // N4: developer-role interactions with drop_thinking.
    // find_last_user_index returns the index of user OR developer messages; the
    // drop_thinking reasoning cutoff and the thinking-seed insertion treat
    // user and developer identically.

    #[test]
    fn test_developer_only_conversation_renders_developer_content() {
        let messages = json!([
            {"role": "system", "content": "sys"},
            {"role": "developer", "content": "x"},
            {"role": "assistant", "reasoning_content": "R", "content": "ok"}
        ]);
        let out =
            encode_messages(messages.as_array().unwrap(), ThinkingMode::Thinking, true).unwrap();
        assert!(
            out.contains("x"),
            "developer content should appear in output, got:\n{}",
            out
        );
    }

    #[test]
    fn test_developer_as_last_user_index_controls_reasoning_cutoff() {
        // Indices: 0=user, 1=assistant(FIRST), 2=developer(y), 3=assistant(SECOND).
        // find_last_user_index = 2 (developer). With drop_thinking=true:
        //   - assistant idx=1 < 2  → reasoning_content stripped.
        //   - assistant idx=3 >= 2 → reasoning_content preserved.
        let messages = json!([
            {"role": "user", "content": "a"},
            {"role": "assistant", "reasoning_content": "FIRST", "content": "r1"},
            {"role": "developer", "content": "y"},
            {"role": "assistant", "reasoning_content": "SECOND", "content": "r2"}
        ]);
        let out =
            encode_messages(messages.as_array().unwrap(), ThinkingMode::Thinking, true).unwrap();
        assert!(
            !out.contains("FIRST"),
            "reasoning before last user/developer (idx 1 < 2) should be stripped, got:\n{}",
            out
        );
        assert!(
            out.contains("SECOND"),
            "reasoning at/after last user/developer (idx 3 > 2) should survive, got:\n{}",
            out
        );
    }
}
