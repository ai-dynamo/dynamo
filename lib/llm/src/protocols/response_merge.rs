// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Joining the pieces of one engine response.
//!
//! A response reaches the frontend in pieces at two seams: an engine streams it
//! in chunks, and in P/D only the prefill worker computes prompt-side outputs.
//! A [`ResponseMerger`] owns one engine's rules for both seams; [`RuleSet`]
//! implements it from declarative [`FieldRule`]s, the intended plugin format.
//! Merges move or swap values, so no field is copied.

use std::sync::Arc;

use base64::{Engine as _, engine::general_purpose::STANDARD};
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};

/// Request annotation asking the prefill router to send prefill's final
/// `engine_data` into the decode stream, in an item with this event name. It
/// arrives before decode's final output.
pub const PREFILL_OUTPUT_ANNOTATION: &str = "prefill_output";
/// With [`PREFILL_OUTPUT_ANNOTATION`]: hold decode output until it arrives.
pub const PREFILL_OUTPUT_FIRST_ANNOTATION: &str = "prefill_output_first";

/// When a response needs prefill's output.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PrefillNeed {
    None,
    /// Only the final chunk carries prefill's values.
    BeforeLast,
    /// Every chunk carries prefill's values.
    BeforeFirst,
}

pub trait ResponseMerger: Send + Sync {
    /// Whether, and how early, the request needs output only prefill produces.
    fn prefill_need(&self, request: &Value) -> PrefillNeed;

    /// Join the next streamed chunk onto the response so far. Fields without
    /// a rule take `next`'s value.
    fn join_step(&self, joined: &mut Value, next: Value);

    /// Move prefill's prompt-side values into a decode response or chunk.
    fn join_prefill(&self, prefill: &mut Value, decode: &mut Value);

    /// Undo [`Self::join_prefill`]'s moves, so `prefill` serves the next chunk.
    fn restore_prefill(&self, prefill: &mut Value, decode: &mut Value);
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum JoinOp {
    /// Streamed chunks: earlier items or text, then later ones.
    Append,
    /// P/D: prefill's value replaces decode's.
    TakePrefill,
    /// P/D: prefill's base64 bytes overwrite the start of decode's.
    OverlayBase64Prefix,
}

/// One rule; `path` is a JSON pointer.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct FieldRule {
    pub path: String,
    pub op: JoinOp,
    /// Remove decode's value when prefill has none, for values that look
    /// complete but are wrong without prefill's part.
    #[serde(default)]
    pub drop_if_missing: bool,
}

struct Rule {
    parent: Vec<String>,
    key: String,
    op: JoinOp,
    drop_if_missing: bool,
}

pub struct RuleSet {
    rules: Vec<Rule>,
    before_first: Vec<Vec<String>>,
    before_last: Vec<Vec<String>>,
}

impl RuleSet {
    /// A request needs prefill's output when every top-level field of some
    /// group is set: `true`, a non-negative number, or a list holding one.
    /// `before_first` groups ask for values every chunk carries; `before_last`
    /// groups for values only the final chunk carries.
    pub fn new(
        rules: impl IntoIterator<Item = FieldRule>,
        before_first: &[&[&str]],
        before_last: &[&[&str]],
    ) -> Self {
        let rules = rules
            .into_iter()
            .map(|rule| {
                let mut parent: Vec<String> = rule.path.split('/').skip(1).map(unescape).collect();
                let key = parent.pop().unwrap_or_default();
                Rule {
                    parent,
                    key,
                    op: rule.op,
                    drop_if_missing: rule.drop_if_missing,
                }
            })
            .collect();
        let groups = |groups: &[&[&str]]| {
            groups
                .iter()
                .map(|group| group.iter().map(|field| field.to_string()).collect())
                .collect()
        };
        Self {
            rules,
            before_first: groups(before_first),
            before_last: groups(before_last),
        }
    }
}

fn unescape(token: &str) -> String {
    token.replace("~1", "/").replace("~0", "~")
}

fn object<'a>(mut value: &'a mut Value, path: &[String]) -> Option<&'a mut Map<String, Value>> {
    for key in path {
        value = value.get_mut(key.as_str())?;
    }
    value.as_object_mut()
}

/// The field's value, inserting `null` first when the key is absent.
fn slot<'a>(object: &'a mut Map<String, Value>, key: &str) -> &'a mut Value {
    if !object.contains_key(key) {
        object.insert(key.to_owned(), Value::Null);
    }
    object.get_mut(key).expect("key was just inserted")
}

impl ResponseMerger for RuleSet {
    fn prefill_need(&self, request: &Value) -> PrefillNeed {
        fn set(value: &Value) -> bool {
            match value {
                Value::Bool(value) => *value,
                Value::Number(number) => number.as_f64().is_some_and(|n| n >= 0.0),
                Value::Array(values) => values.iter().any(set),
                _ => false,
            }
        }
        let any = |groups: &[Vec<String>]| {
            groups
                .iter()
                .any(|group| group.iter().all(|f| request.get(f).is_some_and(set)))
        };
        if any(&self.before_first) {
            PrefillNeed::BeforeFirst
        } else if any(&self.before_last) {
            PrefillNeed::BeforeLast
        } else {
            PrefillNeed::None
        }
    }

    fn join_step(&self, joined: &mut Value, mut next: Value) {
        for rule in self.rules.iter().filter(|rule| rule.op == JoinOp::Append) {
            let Some(head) = object(joined, &rule.parent).and_then(|o| o.remove(&rule.key)) else {
                continue;
            };
            let Some(target) = object(&mut next, &rule.parent) else {
                continue;
            };
            match (head, slot(target, &rule.key)) {
                (Value::Array(mut head), Value::Array(tail)) => {
                    head.append(tail);
                    *tail = head;
                }
                (Value::String(mut head), Value::String(tail)) => {
                    head.push_str(tail);
                    *tail = head;
                }
                (head, tail) if tail.is_null() => *tail = head,
                _ => tracing::warn!(key = %rule.key, "cannot append mismatched values"),
            }
        }
        *joined = next;
    }

    fn join_prefill(&self, prefill: &mut Value, decode: &mut Value) {
        self.swap_prefill(prefill, decode, true);
    }

    fn restore_prefill(&self, prefill: &mut Value, decode: &mut Value) {
        self.swap_prefill(prefill, decode, false);
    }
}

impl RuleSet {
    /// TakePrefill values are swapped, so swapping again restores them; only
    /// a join overlays and drops.
    fn swap_prefill(&self, prefill: &mut Value, decode: &mut Value, join: bool) {
        for rule in self.rules.iter().filter(|rule| rule.op != JoinOp::Append) {
            let Some(target) = object(decode, &rule.parent) else {
                continue;
            };
            let source = object(prefill, &rule.parent).and_then(|o| o.get_mut(&rule.key));
            let joined = match (rule.op, source) {
                (JoinOp::TakePrefill, Some(source)) => {
                    std::mem::swap(source, slot(target, &rule.key));
                    true
                }
                (JoinOp::OverlayBase64Prefix, _) if !join => true,
                (JoinOp::OverlayBase64Prefix, Some(Value::String(source))) => {
                    match target.get_mut(&rule.key) {
                        Some(Value::String(target)) => overlay_base64_prefix(source, target),
                        _ => true,
                    }
                }
                _ => false,
            };
            if join && !joined && rule.drop_if_missing {
                target.remove(&rule.key);
            }
        }
    }
}

/// Make `target` encode `prefix`'s bytes followed by `target`'s bytes past
/// that length, in place. Whole 4-character groups are copied as text; only
/// the group where `prefix`'s bytes end is decoded. Returns false, leaving
/// `target` unchanged, when either value is not padded standard base64.
fn overlay_base64_prefix(prefix: &str, target: &mut String) -> bool {
    let decoded_len = |text: &str| text.trim_end_matches('=').len() * 3 / 4;
    let len = decoded_len(prefix);
    let whole = len / 3 * 4;
    if !is_padded_base64(prefix) || !is_padded_base64(target) {
        return false;
    }
    if decoded_len(target) <= len {
        target.replace_range(.., prefix);
        return true;
    }
    // The 4-character group holding prefix's last bytes and target's next ones.
    let mut group = [0u8; 4];
    let mut group_len = 0;
    if !len.is_multiple_of(3) {
        let (mut head, mut tail) = ([0u8; 3], [0u8; 3]);
        let Some(target_group) = target.get(whole..whole + 4) else {
            return false;
        };
        let (Ok(_), Ok(tail_len)) = (
            STANDARD.decode_slice(&prefix[whole..], &mut head),
            STANDARD.decode_slice(target_group, &mut tail),
        ) else {
            return false;
        };
        tail[..len % 3].copy_from_slice(&head[..len % 3]);
        let Ok(written) = STANDARD.encode_slice(&tail[..tail_len], &mut group) else {
            return false;
        };
        group_len = written;
    }
    // Overwrite in place: memcpy, then one validation pass to get a `String` back.
    let mut bytes = std::mem::take(target).into_bytes();
    bytes[..whole].copy_from_slice(&prefix.as_bytes()[..whole]);
    bytes[whole..whole + group_len].copy_from_slice(&group[..group_len]);
    *target = String::from_utf8(bytes).expect("base64 is ASCII");
    true
}

/// Whole 4-character groups of the standard alphabet, with at most two `=`
/// at the end. A byte scan, so the overlay can copy groups without decoding.
fn is_padded_base64(text: &str) -> bool {
    let body = text.trim_end_matches('=');
    text.len().is_multiple_of(4)
        && text.len() - body.len() <= 2
        && body
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || byte == b'+' || byte == b'/')
}

/// Builds one client response from engine pieces.
pub struct ResponseAssembler {
    merger: Arc<dyn ResponseMerger>,
    prefill: Option<Value>,
    joined: Option<Value>,
}

impl ResponseAssembler {
    pub fn new(merger: Arc<dyn ResponseMerger>) -> Self {
        Self {
            merger,
            prefill: None,
            joined: None,
        }
    }

    pub fn set_prefill(&mut self, prefill: Value) {
        self.prefill = Some(prefill);
    }

    /// Streaming: `render` sees `chunk` with prefill's values swapped in;
    /// they are swapped back out afterwards for the next chunk.
    pub fn render_chunk<T>(&mut self, mut chunk: Value, render: impl FnOnce(&Value) -> T) -> T {
        let Some(prefill) = &mut self.prefill else {
            return render(&chunk);
        };
        self.merger.join_prefill(prefill, &mut chunk);
        let rendered = render(&chunk);
        self.merger.restore_prefill(prefill, &mut chunk);
        rendered
    }

    /// Unary: join `chunk` onto the response so far.
    pub fn push(&mut self, chunk: Value) {
        match &mut self.joined {
            Some(joined) => self.merger.join_step(joined, chunk),
            None => self.joined = Some(chunk),
        }
    }

    /// Unary: the joined response, or `None` when no chunk arrived.
    pub fn finish(self) -> Option<Value> {
        let mut response = self.joined?;
        if let Some(mut prefill) = self.prefill {
            self.merger.join_prefill(&mut prefill, &mut response);
        }
        Some(response)
    }
}

#[cfg(test)]
mod tests {
    use proptest::prelude::*;
    use serde_json::json;

    use super::*;

    /// `/text` and `/m/ids` stream as deltas, `/m/prompt` and `/m/blob` come
    /// from prefill, and `/m/n` is a snapshot without a rule.
    fn rules() -> Arc<dyn ResponseMerger> {
        let rule = |path: &str, op, drop_if_missing| FieldRule {
            path: path.into(),
            op,
            drop_if_missing,
        };
        Arc::new(RuleSet::new(
            [
                rule("/text", JoinOp::Append, false),
                rule("/m/ids", JoinOp::Append, false),
                rule("/m/prompt", JoinOp::TakePrefill, false),
                rule("/m/blob", JoinOp::OverlayBase64Prefix, true),
            ],
            &[&["need", "start"]],
            &[&["want"]],
        ))
    }

    /// One streamed chunk: a text delta, an optional ids delta (absent key
    /// when `None`), and the current snapshot `n`.
    fn step() -> impl Strategy<Value = (String, Option<Vec<u32>>, u32)> {
        (
            "\\PC{0,4}",
            prop::option::of(prop::collection::vec(any::<u32>(), 0..4)),
            any::<u32>(),
        )
    }

    /// A decode chunk: an optional prompt value and an optional blob, plus a
    /// snapshot that prefill must not touch.
    fn decode_chunk() -> impl Strategy<Value = Value> {
        (
            prop::option::of(prop::collection::vec(any::<u32>(), 0..3)),
            prop::option::of(prop::collection::vec(any::<u8>(), 0..24)),
            any::<u32>(),
        )
            .prop_map(|(prompt, blob, n)| {
                let mut m = json!({ "n": n });
                if let Some(prompt) = prompt {
                    m["prompt"] = json!(prompt);
                }
                if let Some(blob) = blob {
                    m["blob"] = json!(STANDARD.encode(blob));
                }
                json!({ "m": m })
            })
    }

    fn prefill() -> impl Strategy<Value = Value> {
        (
            prop::option::of(prop::collection::vec(any::<u32>(), 0..3)),
            prop::option::of(prop::collection::vec(any::<u8>(), 0..24)),
        )
            .prop_map(|(prompt, blob)| {
                let mut m = json!({ "other": "prefill-only" });
                if let Some(prompt) = prompt {
                    m["prompt"] = json!(prompt);
                }
                if let Some(blob) = blob {
                    m["blob"] = json!(STANDARD.encode(blob));
                }
                json!({ "m": m })
            })
    }

    /// What one decode chunk must look like with prefill's values joined in.
    fn expected_join(prefill: &Value, decode: &Value) -> Value {
        let mut expected = decode.clone();
        let m = expected["m"].as_object_mut().unwrap();
        if let Some(prompt) = prefill["m"].get("prompt") {
            m.insert("prompt".into(), prompt.clone());
        }
        match (prefill["m"].get("blob"), m.get("blob").cloned()) {
            (Some(head), Some(tail)) => {
                let mut bytes = STANDARD.decode(head.as_str().unwrap()).unwrap();
                let tail = STANDARD.decode(tail.as_str().unwrap()).unwrap();
                bytes.extend_from_slice(tail.get(bytes.len()..).unwrap_or_default());
                m.insert("blob".into(), json!(STANDARD.encode(bytes)));
            }
            (None, Some(_)) => {
                m.remove("blob");
            }
            _ => {}
        }
        expected
    }

    proptest! {
        /// Joining streamed chunks appends the delta fields in order and keeps
        /// the last chunk's other fields; a delta field missing from every
        /// chunk stays missing (SGLang's `_coalesce_streaming_chunks`).
        #[test]
        fn steps_append_deltas_and_keep_the_last_snapshot(
            steps in prop::collection::vec(step(), 1..6)
        ) {
            let mut assembler = ResponseAssembler::new(rules());
            for (text, ids, n) in &steps {
                let mut m = json!({ "n": n });
                if let Some(ids) = ids {
                    m["ids"] = json!(ids);
                }
                assembler.push(json!({ "text": text, "m": m }));
            }
            let mut expected = json!({
                "text": steps.iter().map(|(text, ..)| text.as_str()).collect::<String>(),
                "m": { "n": steps.last().unwrap().2 },
            });
            if steps.iter().any(|(_, ids, _)| ids.is_some()) {
                let ids: Vec<u32> = steps.iter().flat_map(|(_, ids, _)| ids.iter().flatten()).copied().collect();
                expected["m"]["ids"] = json!(ids);
            }
            prop_assert_eq!(assembler.finish().unwrap(), expected);
        }

        /// Streaming joins prefill's values into every chunk and leaves
        /// prefill intact for the next one.
        #[test]
        fn streamed_chunks_get_prefill_values_and_prefill_survives(
            prefill in prefill(),
            chunks in prop::collection::vec(decode_chunk(), 1..5),
        ) {
            let mut assembler = ResponseAssembler::new(rules());
            assembler.set_prefill(prefill.clone());
            for chunk in chunks {
                let expected = expected_join(&prefill, &chunk);
                prop_assert_eq!(assembler.render_chunk(chunk, Value::clone), expected);
                prop_assert_eq!(assembler.prefill.as_ref(), Some(&prefill));
            }
        }

        /// A client that joins the streamed chunks itself gets the unary
        /// response: joining prefill commutes with joining chunks.
        #[test]
        fn unary_response_equals_joined_streamed_chunks(
            prefill in prefill(),
            chunks in prop::collection::vec(decode_chunk(), 1..5),
        ) {
            let mut streamed = ResponseAssembler::new(rules());
            streamed.set_prefill(prefill.clone());
            let mut client = ResponseAssembler::new(rules());
            let mut unary = ResponseAssembler::new(rules());
            unary.set_prefill(prefill);
            for chunk in chunks {
                client.push(streamed.render_chunk(chunk.clone(), Value::clone));
                unary.push(chunk);
            }
            prop_assert_eq!(unary.finish(), client.finish());
        }

        /// The in-place overlay equals decode, splice, encode.
        #[test]
        fn overlay_matches_decode_splice_encode(
            prefix in prop::collection::vec(any::<u8>(), 0..64),
            target in prop::collection::vec(any::<u8>(), 0..64),
        ) {
            let mut expected = prefix.clone();
            expected.extend_from_slice(target.get(prefix.len()..).unwrap_or_default());
            let mut encoded = STANDARD.encode(&target);
            prop_assert!(overlay_base64_prefix(&STANDARD.encode(&prefix), &mut encoded));
            prop_assert_eq!(encoded, STANDARD.encode(&expected));
        }

        /// Near-base64 text (stray `=`, invalid and multi-byte characters)
        /// never panics or yields invalid base64; a refused overlay leaves
        /// the target unchanged.
        #[test]
        fn overlay_of_arbitrary_text_is_safe(
            prefix in "[A-Za-z0-9+/=!é]{0,12}",
            target in "[A-Za-z0-9+/=!é]{0,16}",
        ) {
            let mut result = target.clone();
            if overlay_base64_prefix(&prefix, &mut result) {
                prop_assert!(is_padded_base64(&prefix) && is_padded_base64(&result));
            } else {
                prop_assert_eq!(result, target);
            }
        }
    }

    #[test]
    fn prefill_need_requires_every_field_of_a_group() {
        let merger = rules();
        for (request, need) in [
            (json!({"want": [false, true]}), PrefillNeed::BeforeLast),
            (
                json!({"want": true, "need": true, "start": 0}),
                PrefillNeed::BeforeFirst,
            ),
            (json!({"want": false}), PrefillNeed::None),
            (json!({"need": true}), PrefillNeed::None),
            (json!({"need": true, "start": -1}), PrefillNeed::None),
        ] {
            assert_eq!(merger.prefill_need(&request), need, "{request}");
        }
    }

    #[test]
    fn rules_have_a_declarative_format() {
        let rule: FieldRule = serde_json::from_value(json!({
            "path": "/meta_info/routed_experts", "op": "overlay_base64_prefix", "drop_if_missing": true,
        }))
        .unwrap();
        assert_eq!(rule.op, JoinOp::OverlayBase64Prefix);
        assert!(rule.drop_if_missing);
    }
}
