// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! What an OpenAI response reports to the router, read off the body Envoy
//! relays: output progress, the finish reason, token usage and errors.
//!
//! Streaming events are read by a byte scan rather than a JSON parse, since one
//! arrives per generated token. Only an event carrying a non-null `usage`
//! object, normally the last one, is parsed.

use crate::picker::ResponseUsage;

/// One response event: an SSE `data:` payload, or a whole buffered body.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct ResponseEvent {
    /// Output-bearing fields in this event: non-empty `content`,
    /// `reasoning_content` or `text`, or a non-empty `tool_calls` array.
    /// Streaming engines send one token per field, so this approximates the
    /// tokens generated. Always 0 for a buffered body.
    pub content_deltas: usize,
    /// The last non-null `finish_reason` in the event.
    pub finish_reason: Option<String>,
    pub usage: Option<ResponseUsage>,
    /// The event carries a non-null `error`.
    pub error: bool,
}

impl ResponseEvent {
    fn is_empty(&self) -> bool {
        *self == Self::default()
    }

    /// Read one SSE `data:` payload. `None` for an event that reports nothing,
    /// such as a role-only delta or `[DONE]`.
    pub fn from_sse_data(payload: &[u8]) -> Option<Self> {
        let mut event = Self::default();
        let mut has_usage = false;
        for_each_member(payload, |key, value| match key {
            b"content" | b"reasoning_content" | b"text" if non_empty_string(value) => {
                event.content_deltas += 1;
            }
            b"tool_calls" if non_empty_array(value) => event.content_deltas += 1,
            b"finish_reason" => {
                if let Some(reason) = string_value(value) {
                    event.finish_reason = Some(reason.to_owned());
                }
            }
            b"error" => event.error |= !value.starts_with(b"null"),
            b"usage" => has_usage |= value.starts_with(b"{"),
            _ => {}
        });
        if has_usage {
            event.usage = serde_json::from_slice(payload)
                .ok()
                .as_ref()
                .and_then(usage_from_json);
        }
        (!event.is_empty()).then_some(event)
    }

    /// Read a buffered (non-streaming) JSON response body with one parse.
    pub fn from_body(body: &[u8]) -> Self {
        let Ok(value) = serde_json::from_slice::<serde_json::Value>(body) else {
            return Self::default();
        };
        let finish_reason = value
            .get("choices")
            .and_then(|choices| choices.as_array())
            .and_then(|choices| {
                choices
                    .iter()
                    .filter_map(|choice| choice.get("finish_reason")?.as_str())
                    .next_back()
            })
            .map(str::to_owned);
        Self {
            content_deltas: 0,
            finish_reason,
            usage: usage_from_json(&value),
            error: value.get("error").is_some_and(|error| !error.is_null()),
        }
    }
}

/// Extract [`ResponseUsage`] from a JSON object's `usage` member.
pub(crate) fn usage_from_json(value: &serde_json::Value) -> Option<ResponseUsage> {
    let usage = value.get("usage")?;
    if usage.is_null() {
        return None;
    }
    let field = |name: &str| usage.get(name).and_then(serde_json::Value::as_u64);
    Some(ResponseUsage {
        prompt_tokens: field("prompt_tokens"),
        completion_tokens: field("completion_tokens"),
        total_tokens: field("total_tokens"),
        cached_tokens: usage
            .get("prompt_tokens_details")
            .and_then(|details| details.get("cached_tokens"))
            .and_then(serde_json::Value::as_u64),
    })
}

/// Call `f(key, value)` for every object member in a JSON document, where
/// `value` is the rest of the input from the member's value onward. Strings are
/// skipped escape-aware in one pass, so text inside a string value is never
/// mistaken for a key.
fn for_each_member(json: &[u8], mut f: impl FnMut(&[u8], &[u8])) {
    let mut at = 0;
    while let Some(open) = json[at..].iter().position(|&b| b == b'"') {
        let start = at + open + 1;
        let Some(end) = string_end(json, start) else {
            return;
        };
        let after = json[end + 1..].trim_ascii_start();
        if let Some(value) = after.strip_prefix(b":") {
            f(&json[start..end], value.trim_ascii_start());
        }
        at = end + 1;
    }
}

/// Index of the quote closing a string whose contents start at `start`.
fn string_end(json: &[u8], start: usize) -> Option<usize> {
    let mut i = start;
    while i < json.len() {
        match json[i] {
            b'\\' => i += 2,
            b'"' => return Some(i),
            _ => i += 1,
        }
    }
    None
}

fn non_empty_string(value: &[u8]) -> bool {
    value.starts_with(b"\"") && value.get(1).is_some_and(|&b| b != b'"')
}

fn non_empty_array(value: &[u8]) -> bool {
    value
        .strip_prefix(b"[")
        .is_some_and(|rest| rest.trim_ascii_start().first().is_some_and(|&b| b != b']'))
}

/// The contents of a string value; `None` for `null` or a non-string.
fn string_value(value: &[u8]) -> Option<&str> {
    let contents = value.strip_prefix(b"\"")?;
    let end = string_end(contents, 0)?;
    std::str::from_utf8(&contents[..end]).ok()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn event(payload: &str) -> Option<ResponseEvent> {
        ResponseEvent::from_sse_data(payload.as_bytes())
    }

    #[test]
    fn counts_output_fields_in_compact_and_spaced_json() {
        for payload in [
            r#"{"choices":[{"index":0,"delta":{"content":"Hel"}}]}"#,
            r#"{"choices": [{"index": 0, "delta": {"content": "Hel"}}]}"#,
            r#"{"choices":[{"delta":{"reasoning_content":"so"}}]}"#,
            r#"{"choices":[{"text":"lo","index":0}]}"#,
            r#"{"choices":[{"delta":{"tool_calls":[{"index":0,"function":{"arguments":"{"}}]}}]}"#,
        ] {
            assert_eq!(event(payload).unwrap().content_deltas, 1, "{payload}");
        }
        let two_choices = r#"{"choices":[{"index":0,"delta":{"content":"a"}},{"index":1,"delta":{"content":"b"}}]}"#;
        assert_eq!(event(two_choices).unwrap().content_deltas, 2);
    }

    #[test]
    fn events_without_output_report_nothing() {
        for payload in [
            r#"{"choices":[{"delta":{"role":"assistant"}}]}"#,
            r#"{"choices":[{"delta":{"role":"assistant","content":""}}]}"#,
            r#"{"choices":[{"delta":{"content":null,"tool_calls":[]}}]}"#,
            r#"{"choices":[{"delta":{},"logprobs":{"content":[]}}]}"#,
            "[DONE]",
        ] {
            assert_eq!(event(payload), None, "{payload}");
        }
    }

    /// Keys inside string values (here, tool-call arguments) are text, not keys.
    #[test]
    fn keys_inside_string_values_are_ignored() {
        let payload = r#"{"choices":[{"delta":{"tool_calls":[{"function":{"arguments":"{\"error\":\"x\",\"content\":\"y\"}"}}]}}]}"#;
        let event = event(payload).unwrap();
        assert_eq!(event.content_deltas, 1);
        assert!(!event.error);
    }

    #[test]
    fn reads_finish_reason_usage_and_errors() {
        let last = event(
            r#"{"choices":[{"index":0,"delta":{},"finish_reason":"length"}],"usage":{"prompt_tokens":4,"completion_tokens":2,"total_tokens":6,"prompt_tokens_details":{"cached_tokens":3}}}"#,
        )
        .unwrap();
        assert_eq!(last.finish_reason.as_deref(), Some("length"));
        let usage = last.usage.unwrap();
        assert_eq!(usage.completion_tokens, Some(2));
        assert_eq!(usage.cached_tokens, Some(3));

        let null_usage =
            r#"{"choices":[{"delta":{"content":"a"},"finish_reason":null}],"usage":null}"#;
        assert_eq!(
            event(null_usage).unwrap(),
            ResponseEvent {
                content_deltas: 1,
                ..Default::default()
            }
        );

        assert!(
            event(r#"{"error":{"message":"boom","code":500}}"#)
                .unwrap()
                .error
        );
    }

    #[test]
    fn a_buffered_body_is_read_once() {
        let body = br#"{"choices":[{"index":0,"message":{"role":"assistant","content":"hi"},"finish_reason":"stop"}],"usage":{"prompt_tokens":4,"completion_tokens":9,"total_tokens":13}}"#;
        let event = ResponseEvent::from_body(body);
        assert_eq!(event.content_deltas, 0);
        assert_eq!(event.finish_reason.as_deref(), Some("stop"));
        assert_eq!(event.usage.unwrap().completion_tokens, Some(9));
        assert!(!event.error);

        assert!(ResponseEvent::from_body(br#"{"error":{"message":"bad"}}"#).error);
        assert_eq!(
            ResponseEvent::from_body(b"not json"),
            ResponseEvent::default()
        );
    }
}
