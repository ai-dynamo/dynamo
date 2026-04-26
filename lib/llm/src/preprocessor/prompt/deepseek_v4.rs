// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! DeepSeek V4 prompt formatting.
//!
//! Production rendering goes through `jinja_chat::render_v4` and the inline
//! template at `templates/deepseek_v4_inline.jinja`. This module only keeps:
//!   - public types (`ThinkingMode`, `ReasoningEffort`, `DeepSeekV4Formatter`)
//!     and constants (`tokens::*`)
//!   - the Rust pre-pass the template assumes already ran
//!     (`merge_tool_messages`, `sort_tool_results_by_call_order`)
//!   - request-side glue (`normalize_message_contents`, the
//!     `OAIPromptFormatter` impl that injects tools / response_format into
//!     the system message before handing the array to the renderer)
//!   - thin shim entry points (`encode_messages`,
//!     `encode_messages_with_options`) that route through
//!     `jinja_chat::render_v4`.
//!
//! Reference: DeepSeek-V4-Pro/encoding/encoding_dsv4.py.

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

/// Thinking mode for the model
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ThinkingMode {
    Chat,
    Thinking,
}

impl ThinkingMode {
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

/// Flatten message `content` from OAI multi-modal-block form (list of
/// `{type: "text", text: ...}` etc.) into a plain string. Non-string,
/// non-array content is left untouched (the template handles `null`).
fn normalize_message_contents(messages: &mut [JsonValue]) {
    for msg in messages {
        let Some(content) = msg.get("content") else {
            continue;
        };
        let normalized = match content {
            JsonValue::String(s) => s.clone(),
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
                            .map(str::to_string);
                    }
                    tracing::warn!(
                        chunk_type = item_type.unwrap_or("unknown"),
                        "DeepSeek V4 formatter dropped non-text content chunk while normalizing message content",
                    );
                    None
                })
                .collect::<String>(),
            _ => continue,
        };
        if let Some(obj) = msg.as_object_mut() {
            obj.insert("content".to_string(), JsonValue::String(normalized));
        }
    }
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

/// Encode messages to prompt string with default options.
///
/// Thin shim over `jinja_chat::render_v4`. Equivalent to
/// `encode_messages_with_options(.., drop_thinking=true, reasoning_effort=None)`.
pub fn encode_messages(
    messages: &[JsonValue],
    thinking_mode: ThinkingMode,
    add_bos_token: bool,
) -> Result<String> {
    encode_messages_with_options(messages, thinking_mode, add_bos_token, true, None)
}

/// Encode messages to prompt string.
///
/// Thin shim over `jinja_chat::render_v4` — production rendering lives in the
/// inline Jinja template at `templates/deepseek_v4_inline.jinja`. This entry
/// point is kept so existing callers (tests, the formatter trait impl)
/// continue to work unchanged.
pub fn encode_messages_with_options(
    messages: &[JsonValue],
    thinking_mode: ThinkingMode,
    add_bos_token: bool,
    drop_thinking: bool,
    reasoning_effort: Option<ReasoningEffort>,
) -> Result<String> {
    use super::jinja_chat;
    let jinja_mode = match thinking_mode {
        ThinkingMode::Chat => jinja_chat::ThinkingMode::Chat,
        ThinkingMode::Thinking => jinja_chat::ThinkingMode::Thinking,
    };
    let jinja_effort = reasoning_effort.and_then(|e| match e {
        ReasoningEffort::Max => Some(jinja_chat::ReasoningEffort::Max),
        // Only `Max` injects a prefix in the template; other levels are no-ops.
        ReasoningEffort::High => None,
    });
    jinja_chat::render_v4(
        messages,
        jinja_mode,
        add_bos_token,
        drop_thinking,
        jinja_effort,
    )
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

    fn resolve_thinking_mode(
        &self,
        args: Option<&std::collections::HashMap<String, serde_json::Value>>,
    ) -> ThinkingMode {
        if let Some(args) = args
            && let Some(thinking) = args.get("thinking").and_then(JsonValue::as_bool)
        {
            return if thinking {
                ThinkingMode::Thinking
            } else {
                ThinkingMode::Chat
            };
        }
        if let Some(args) = args
            && let Some(mode) = args.get("thinking_mode").and_then(JsonValue::as_str)
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
        let thinking_mode = self.resolve_thinking_mode(req.chat_template_args());

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
                .position(|msg| msg.get("role").and_then(JsonValue::as_str) == Some("system"));

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

        encode_messages(&messages_array, thinking_mode, true)
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
        // with the text, joined by "\n\n" at render time.
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

    /// Bug: `last_user_idx = None` (no user/developer in history) should behave
    /// like Python's `-1` sentinel — `index >= -1` / `idx >= -1` always true, so
    /// earlier reasoning is preserved and the assistant's reasoning block is
    /// rendered.
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
}
