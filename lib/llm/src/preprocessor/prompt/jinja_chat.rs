// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Jinja-based chat-template renderer for DeepSeek V4 / V3.2.
//!
//! Replaces the hand-written `deepseek_v4.rs` / `deepseek_v32.rs` encoders with
//! a thin Rust pre-pass plus a minijinja render of the bundled
//! `deepseek_v{4,32}_inline.jinja` templates. The templates are byte-for-byte
//! ports of upstream `encoding_dsv4.py` / `encoding_dsv32.py`; the only Rust
//! responsibilities here are the pre-pass that the templates assume already
//! ran (merge parallel tool messages, sort tool results by call order) and the
//! minijinja env config that makes `tojson` produce Python-style spacing.

use anyhow::{Context, Result};
use minijinja::{Environment, Value};
use serde_json::Value as JsonValue;

use super::deepseek_v4::{merge_tool_messages, sort_tool_results_by_call_order};

/// Embedded inline templates — single-file form, no Jinja `include` lookups.
const V4_INLINE_TEMPLATE: &str =
    include_str!("templates/deepseek_v4_inline.jinja");
const V32_INLINE_TEMPLATE: &str =
    include_str!("templates/deepseek_v32_inline.jinja");

/// `serde_json::ser::Formatter` matching Python's `json.dumps` default spacing
/// (`, ` between elements, `: ` between key and value). The default
/// serde_json `CompactFormatter` emits no spaces, which would diverge from
/// the upstream Python references' fixture output.
struct PythonStyleFormatter;

impl serde_json::ser::Formatter for PythonStyleFormatter {
    fn begin_array_value<W: ?Sized + std::io::Write>(
        &mut self,
        w: &mut W,
        first: bool,
    ) -> std::io::Result<()> {
        if first { Ok(()) } else { w.write_all(b", ") }
    }
    fn begin_object_key<W: ?Sized + std::io::Write>(
        &mut self,
        w: &mut W,
        first: bool,
    ) -> std::io::Result<()> {
        if first { Ok(()) } else { w.write_all(b", ") }
    }
    fn begin_object_value<W: ?Sized + std::io::Write>(&mut self, w: &mut W) -> std::io::Result<()> {
        w.write_all(b": ")
    }
}

fn python_style_to_string(v: &JsonValue) -> Result<String> {
    let mut buf = Vec::new();
    let mut ser = serde_json::Serializer::with_formatter(&mut buf, PythonStyleFormatter);
    serde::Serialize::serialize(v, &mut ser).context("python_style_to_string: serialize failed")?;
    String::from_utf8(buf).context("python_style_to_string: non-utf8 output")
}

/// Build a fresh minijinja `Environment` configured the way the DeepSeek
/// templates expect: trim/lstrip blocks on, plus `tojson` and `fromjson`
/// filters that match jinja2 + Python `json` semantics.
pub fn make_chat_env() -> Environment<'static> {
    let mut env = Environment::new();
    env.set_lstrip_blocks(true);
    env.set_trim_blocks(true);
    env.add_filter("tojson", |v: Value| -> Result<String, minijinja::Error> {
        let json_v: JsonValue = serde_json::to_value(v).map_err(|e| {
            minijinja::Error::new(
                minijinja::ErrorKind::InvalidOperation,
                format!("tojson: {e}"),
            )
        })?;
        python_style_to_string(&json_v).map_err(|e| {
            minijinja::Error::new(
                minijinja::ErrorKind::InvalidOperation,
                format!("tojson: {e}"),
            )
        })
    });
    env.add_filter(
        "fromjson",
        |s: String| -> Result<Value, minijinja::Error> {
            serde_json::from_str::<JsonValue>(&s)
                .map(Value::from_serialize)
                .map_err(|e| {
                    minijinja::Error::new(
                        minijinja::ErrorKind::InvalidOperation,
                        format!("fromjson: {e}"),
                    )
                })
        },
    );
    env
}

/// Thinking-mode wire string accepted by both inline templates. Kept as a
/// dedicated enum so callers don't pass a freeform string.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ThinkingMode {
    Chat,
    Thinking,
}

impl ThinkingMode {
    fn as_template_str(self) -> &'static str {
        match self {
            ThinkingMode::Chat => "chat",
            ThinkingMode::Thinking => "thinking",
        }
    }
}

/// Reasoning-effort prefix opt-in. Currently only `Max` has any effect on the
/// template (it injects a verbatim instruction block at idx 0); kept as an
/// enum so future levels (`High`, etc.) can land without a signature change.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReasoningEffort {
    Max,
}

impl ReasoningEffort {
    fn as_template_str(self) -> &'static str {
        match self {
            ReasoningEffort::Max => "max",
        }
    }
}

/// Render messages through `deepseek_v4_inline.jinja`.
///
/// Applies the same pre-pass the existing `deepseek_v4::encode_messages`
/// applies (merge parallel tool messages, sort tool results by call order),
/// then hands the prepared messages to minijinja. The template handles
/// last-user-index detection, drop-thinking, and tool-call rendering itself.
pub fn render_v4(
    messages: &[JsonValue],
    thinking_mode: ThinkingMode,
    add_bos_token: bool,
    drop_thinking: bool,
    reasoning_effort: Option<ReasoningEffort>,
) -> Result<String> {
    let merged = merge_tool_messages(messages);
    let prepared = sort_tool_results_by_call_order(merged);

    let mut env = make_chat_env();
    env.add_template("deepseek_v4_inline.jinja", V4_INLINE_TEMPLATE)
        .context("register deepseek_v4_inline.jinja")?;

    let reasoning_effort_val: Value = match reasoning_effort {
        Some(level) => Value::from(level.as_template_str()),
        None => Value::from(()),
    };

    env.get_template("deepseek_v4_inline.jinja")
        .context("lookup deepseek_v4_inline.jinja")?
        .render(minijinja::context! {
            messages => Value::from_serialize(&prepared),
            thinking_mode => thinking_mode.as_template_str(),
            drop_thinking => drop_thinking,
            add_bos_token => add_bos_token,
            reasoning_effort => reasoning_effort_val,
        })
        .context("render deepseek_v4_inline.jinja")
}

/// Flatten array-form `reasoning_content` (`["seg1", "seg2", ""]`, emitted by
/// interleaved-thinking models like GLM-5) into a single newline-joined
/// string, dropping empty segments. The V3.2 inline template only handles
/// string-form reasoning_content; array form would render as `[object]`.
fn flatten_reasoning_segments(messages: &[JsonValue]) -> Vec<JsonValue> {
    messages
        .iter()
        .map(|msg| {
            let JsonValue::Array(segments) = msg.get("reasoning_content").unwrap_or(&JsonValue::Null)
            else {
                return msg.clone();
            };
            let joined = segments
                .iter()
                .filter_map(JsonValue::as_str)
                .filter(|s| !s.is_empty())
                .collect::<Vec<_>>()
                .join("\n");
            let mut out = msg.clone();
            if let Some(obj) = out.as_object_mut() {
                obj.insert("reasoning_content".to_string(), JsonValue::String(joined));
            }
            out
        })
        .collect()
}

/// Render messages through `deepseek_v32_inline.jinja`.
///
/// V3.2 has no parallel-tool merge step. The only Rust pre-pass here is
/// flattening array-form `reasoning_content` into the string form the
/// template expects (the upstream Python reference encoded
/// `reasoning_content` as a string).
pub fn render_v32(
    messages: &[JsonValue],
    thinking_mode: ThinkingMode,
    add_bos_token: bool,
    drop_thinking: bool,
) -> Result<String> {
    let prepared = flatten_reasoning_segments(messages);

    let mut env = make_chat_env();
    env.add_template("deepseek_v32_inline.jinja", V32_INLINE_TEMPLATE)
        .context("register deepseek_v32_inline.jinja")?;

    env.get_template("deepseek_v32_inline.jinja")
        .context("lookup deepseek_v32_inline.jinja")?
        .render(minijinja::context! {
            messages => Value::from_serialize(&prepared),
            thinking_mode => thinking_mode.as_template_str(),
            drop_thinking => drop_thinking,
            add_bos_token => add_bos_token,
        })
        .context("render deepseek_v32_inline.jinja")
}
