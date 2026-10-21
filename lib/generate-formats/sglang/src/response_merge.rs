// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Joining the pieces of one engine response.
//!
//! A field table joins streamed chunks and combines prefill/decode output.
//! Joins move or swap values, so streaming copies no field per chunk.

use std::collections::BTreeMap;
use std::sync::Arc;

use base64::{Engine as _, engine::general_purpose::STANDARD};
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};

use dynamo_llm::protocols::response_merge::PrefillNeed;

/// How one response field joins chunks and prefill/decode output.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum FieldClass {
    /// Output-side arrays or text appended across chunks.
    Delta,
    /// Prompt-side values supplied by prefill in disaggregated serving.
    Prompt,
    /// Final Base64 rows: prefill's bytes replace decode's stale prompt rows.
    SequenceFinal,
    /// A snapshot: the last chunk's value wins.
    Attempt,
}

struct Field {
    path: String,
    parent: Vec<String>,
    key: String,
    class: FieldClass,
}

/// A request needs prefill's output when every top-level field of some group
/// is set: `true`, a non-negative number, or a list holding one.
#[derive(Debug, Clone, Default, Deserialize, Serialize)]
pub struct PrefillFlags {
    /// Groups asking for values every chunk carries.
    #[serde(default)]
    pub before_first: Vec<Vec<String>>,
    /// Groups asking for values only the final chunk carries.
    #[serde(default)]
    pub before_last: Vec<Vec<String>>,
}

#[derive(Deserialize)]
struct RulesFile {
    fields: BTreeMap<String, FieldClass>,
    #[serde(default)]
    prefill_request_flags: PrefillFlags,
}

/// One format's join table. Fields without a class keep the last value.
pub struct RuleSet {
    fields: Vec<Field>,
    prefill_flags: PrefillFlags,
}

impl RuleSet {
    /// `fields` maps JSON pointers to classes.
    pub fn new(
        fields: impl IntoIterator<Item = (String, FieldClass)>,
        prefill_flags: PrefillFlags,
    ) -> Self {
        let fields: Vec<Field> = fields
            .into_iter()
            .map(|(path, class)| {
                let mut parent: Vec<String> = path.split('/').skip(1).map(unescape).collect();
                let key = parent.pop().unwrap_or_default();
                Field {
                    path,
                    parent,
                    key,
                    class,
                }
            })
            .collect();
        Self {
            fields,
            prefill_flags,
        }
    }

    /// Parse `{"fields": {pointer: class}, "prefill_request_flags": {...}}`.
    pub fn from_json(json: &str) -> Result<Self, serde_json::Error> {
        let file: RulesFile = serde_json::from_str(json)?;
        Ok(Self::new(file.fields, file.prefill_request_flags))
    }

    pub fn fields(&self) -> impl Iterator<Item = (&str, FieldClass)> {
        self.fields.iter().map(|f| (f.path.as_str(), f.class))
    }

    pub fn prefill_need(&self, request: &Map<String, Value>) -> PrefillNeed {
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
        if any(&self.prefill_flags.before_first) {
            PrefillNeed::BeforeFirst
        } else if any(&self.prefill_flags.before_last) {
            PrefillNeed::BeforeLast
        } else {
            PrefillNeed::None
        }
    }

    fn of(&self, class: FieldClass) -> impl Iterator<Item = &Field> {
        self.fields.iter().filter(move |f| f.class == class)
    }

    /// Join the next streamed chunk onto the response so far.
    fn join_step(&self, joined: &mut Value, mut next: Value) {
        for field in self.of(FieldClass::Delta) {
            let Some(head) = object(joined, &field.parent).and_then(|o| o.remove(&field.key))
            else {
                continue;
            };
            let Some(target) = object(&mut next, &field.parent) else {
                continue;
            };
            match (head, slot(target, &field.key)) {
                (Value::Array(mut head), Value::Array(tail)) => {
                    head.append(tail);
                    *tail = head;
                }
                (Value::String(mut head), Value::String(tail)) => {
                    head.push_str(tail);
                    *tail = head;
                }
                (head, tail) if tail.is_null() => *tail = head,
                _ => tracing::warn!(key = %field.key, "cannot append mismatched values"),
            }
        }
        *joined = next;
    }

    /// Swap `class` values present in `source` into `target`. Swapping again
    /// restores both.
    fn swap(&self, class: FieldClass, source: &mut Value, target: &mut Value) {
        for field in self.of(class) {
            let Some(from) = object(source, &field.parent).and_then(|o| o.get_mut(&field.key))
            else {
                continue;
            };
            if let Some(to) = object(target, &field.parent) {
                std::mem::swap(from, slot(to, &field.key));
            }
        }
    }

    /// P/D: prompt values are swapped in; prefill's base64 prefix overwrites
    /// decode's in place, and decode's value is dropped without one.
    fn join_prefill(&self, prefill: &mut Value, decode: &mut Value) {
        self.swap(FieldClass::Prompt, prefill, decode);
        for field in self.of(FieldClass::SequenceFinal) {
            let Some(target) = object(decode, &field.parent) else {
                continue;
            };
            let source = object(prefill, &field.parent).and_then(|o| o.get(&field.key));
            let joined = match (source, target.get_mut(&field.key)) {
                (_, None) => true,
                (Some(Value::String(source)), Some(Value::String(target))) => {
                    overlay_base64_prefix(source, target)
                }
                _ => false,
            };
            if !joined {
                target.remove(&field.key);
            }
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
        // Check padding and unused bits with the canonical decoder. Only the
        // final group needs decoding; the bulk overlay remains in place.
        && (text.is_empty()
            || text.get(text.len() - 4..).is_some_and(|tail| {
                STANDARD.decode_slice(tail, &mut [0; 3]).is_ok()
            }))
        && body
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || byte == b'+' || byte == b'/')
}

/// Builds one client response from streamed chunks and prefill output.
pub struct Assembler {
    rules: Arc<RuleSet>,
    prefill: Option<Value>,
    joined: Option<Value>,
}

impl Assembler {
    pub fn new(rules: Arc<RuleSet>) -> Self {
        Self {
            rules,
            prefill: None,
            joined: None,
        }
    }

    pub fn set_prefill(&mut self, prefill: Value) {
        self.prefill = Some(prefill);
    }

    /// Streaming: join prefill values for rendering, then restore them.
    pub fn render<T>(&mut self, mut chunk: Value, render: impl FnOnce(&Value) -> T) -> T {
        if let Some(prefill) = &mut self.prefill {
            self.rules.join_prefill(prefill, &mut chunk);
        }
        let rendered = render(&chunk);
        if let Some(prefill) = &mut self.prefill {
            self.rules.swap(FieldClass::Prompt, prefill, &mut chunk);
        }
        rendered
    }

    /// Unary: join a chunk onto the response so far.
    pub fn push(&mut self, chunk: Value) {
        match &mut self.joined {
            Some(joined) => self.rules.join_step(joined, chunk),
            None => self.joined = Some(chunk),
        }
    }

    /// Unary: combine prefill output with the joined response.
    pub fn finish(&mut self) -> Option<Value> {
        let mut response = self.joined.take()?;
        if let Some(prefill) = &mut self.prefill {
            self.rules.join_prefill(prefill, &mut response);
        }
        Some(response)
    }
}

#[cfg(test)]
mod tests {
    //! Joining and P/D are checked end to end by the
    //! `testkit` laws; these pin the byte-level overlay.
    use proptest::prelude::*;
    use serde_json::json;

    use super::*;

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(512))]

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

        /// Near-base64 text never panics or yields invalid base64; a refused
        /// overlay leaves the target unchanged.
        #[test]
        fn overlay_of_arbitrary_text_is_safe(
            prefix in prop_oneof![Just("AB==".to_string()), "[A-Za-z0-9+/=!é]{0,12}"],
            target in "[A-Za-z0-9+/=!é]{0,16}",
        ) {
            let mut result = target.clone();
            if overlay_base64_prefix(&prefix, &mut result) {
                prop_assert!(STANDARD.decode(&prefix).is_ok());
                prop_assert!(STANDARD.decode(&result).is_ok());
            } else {
                prop_assert_eq!(result, target);
            }
        }
    }

    #[test]
    fn prefill_need_requires_every_field_of_a_group() {
        let rules = RuleSet::from_json(
            r#"{"fields": {}, "prefill_request_flags":
                {"before_first": [["need", "start"]], "before_last": [["want"]]}}"#,
        )
        .unwrap();
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
            assert_eq!(
                rules.prefill_need(request.as_object().unwrap()),
                need,
                "{request}"
            );
        }
    }
}
