// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! DeepSeek V4 native prompt formatting
//!
//! Native Rust port of DeepSeek V4's chat encoding (encoding_dsv4.py).
//!
//! Reference: DeepSeek-V4-Pro/encoding/encoding_dsv4.py

use anyhow::{Context, Result};
use serde_json::Value as JsonValue;

use super::deepseek_common::{
    NormalizeNonText, REASONING_EFFORT_MAX, RESPONSE_FORMAT_TEMPLATE, TOOL_CALLS_BLOCK_NAME,
    TOOLS_TEMPLATE, drop_thinking_messages, encode_arguments_to_dsml, find_last_user_index,
    merge_tool_messages, normalize_message_contents, render_tools, sort_tool_results_by_call_order,
    task_token, to_json,
};
pub use super::deepseek_common::{ReasoningEffort, ThinkingMode, tokens};

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
                prompt.push_str(&render_tools(TOOLS_TEMPLATE, tools));
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
                content_developer.push_str(&render_tools(TOOLS_TEMPLATE, tools));
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
}

impl super::OAIPromptFormatter for DeepSeekV4Formatter {
    fn supports_add_generation_prompt(&self) -> bool {
        true
    }

    fn render(&self, req: &dyn super::OAIChatLikeRequest) -> Result<String> {
        let args = req.chat_template_args();
        let thinking_mode = super::deepseek_common::resolve_thinking_mode(args, self.thinking_mode);
        let reasoning_effort = Self::resolve_reasoning_effort(args);
        let drop_thinking = Self::resolve_drop_thinking(args);

        let messages_value = req.messages();
        let messages_json =
            serde_json::to_value(&messages_value).context("Failed to convert messages to JSON")?;

        let mut messages_array = messages_json
            .as_array()
            .context("Messages is not an array")?
            .clone();

        normalize_message_contents(&mut messages_array, NormalizeNonText::LeaveUntouched);

        super::deepseek_common::inject_tools_and_response_format(&mut messages_array, req)?;

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
    fn test_user_task_preserved_when_merged_after_tool_result() {
        let messages = json!([
            {"role": "assistant", "content": "", "tool_calls": [{
                "id": "c1", "type": "function",
                "function": {"name": "search", "arguments": "{}"}
            }]},
            {"role": "tool", "tool_call_id": "c1", "content": "RESULT"},
            {"role": "user", "content": "Search", "task": "action"},
            {"role": "assistant", "content": "OK"}
        ]);

        let out = encode_messages(messages.as_array().unwrap(), ThinkingMode::Chat, true).unwrap();
        assert!(
            out.contains(&format!(
                "{}Search{}{}{}OK",
                "<tool_result>RESULT</tool_result>\n\n",
                tokens::ASSISTANT_START,
                tokens::THINKING_END,
                tokens::TASK_ACTION
            )),
            "expected merged user text to keep the action task transition, got:\n{}",
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
        let mut args = HashMap::new();
        args.insert(
            "enable_thinking".to_string(),
            serde_json::Value::Bool(false),
        );
        assert_eq!(
            super::super::deepseek_common::resolve_thinking_mode(
                Some(&args),
                ThinkingMode::Thinking
            ),
            ThinkingMode::Chat
        );
        args.insert("enable_thinking".to_string(), serde_json::Value::Bool(true));
        assert_eq!(
            super::super::deepseek_common::resolve_thinking_mode(
                Some(&args),
                ThinkingMode::Thinking
            ),
            ThinkingMode::Thinking
        );
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
    fn test_render_leaves_null_assistant_tool_content_empty() {
        use super::super::OAIPromptFormatter;

        let req = MockRequest::new(json!([
            {"role": "user", "content": "call tool"},
            {"role": "assistant", "content": null, "tool_calls": [{
                "id": "c1", "type": "function",
                "function": {"name": "f", "arguments": "{}"}
            }]}
        ]));

        let formatter = DeepSeekV4Formatter::new_chat();
        let out = formatter.render(&req).unwrap();

        assert!(out.contains(&format!(
            "<{}{}>",
            tokens::DSML_TOKEN,
            TOOL_CALLS_BLOCK_NAME
        )));
        assert!(!out.contains("null"));
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
