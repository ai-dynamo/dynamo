// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Joining the pieces of one engine response.
//!
//! A response reaches the frontend in pieces at three seams: an engine streams
//! it in chunks, in P/D only the prefill worker computes prompt-side outputs,
//! and a migrated request continues on another worker. Each response field has
//! one [`FieldClass`], which fixes how it joins at all three; [`RuleSet`] holds
//! a format's table and [`Assembler`] applies it to one request. Joins move or
//! swap values, so streaming copies no field per chunk.

use std::collections::BTreeMap;
use std::sync::Arc;

use base64::{Engine as _, engine::general_purpose::STANDARD};
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};

use dynamo_llm::protocols::response_merge::PrefillNeed;

use crate::{ClientCodec, Native};

/// How one response field behaves across chunks, P/D, and migration.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum FieldClass {
    /// Output-side deltas: chunks append, decode reports them, and a retry
    /// keeps appending.
    Delta,
    /// A prompt-side value repeated in full on every chunk: prefill computes
    /// it in P/D, and the first attempt's value survives migration because a
    /// retry's prompt includes the delivered tokens.
    Prompt,
    /// Base64 rows for the whole sequence, on the final chunk only: prefill's
    /// bytes replace decode's stale prompt rows, and a retry's value already
    /// covers the whole sequence. Dropped when prefill sends none, since
    /// decode's prompt rows are wrong on their own.
    SequenceFinal,
    /// A snapshot fixed by the first attempt, such as prompt length or cache
    /// hits; a retry reports its own longer prompt.
    FirstAttempt,
    /// A count over the output: a retry reports only its own part, so earlier
    /// attempts' last counts are added.
    Cumulative,
    /// A count measured from the attempt's first output token, such as
    /// completed tokens or the budget a length finish reports: tokens
    /// delivered by earlier attempts are added.
    OutputOffset,
    /// A snapshot about one attempt: the last value wins everywhere.
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
    cumulative: Vec<usize>,
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
        let cumulative = (0..fields.len())
            .filter(|&i| fields[i].class == FieldClass::Cumulative)
            .collect();
        Self {
            fields,
            cumulative,
            prefill_flags,
        }
    }

    /// Parse `{"fields": {pointer: class}, "prefill_request_flags": {...}}`.
    pub fn from_json(json: &str) -> Result<Self, serde_json::Error> {
        let file: RulesFile = serde_json::from_str(json)?;
        Ok(Self::new(file.fields, file.prefill_request_flags))
    }

    /// Add or override field classes, retaining the native request flags.
    /// Additional flag groups request prefill output for new response fields.
    pub fn extend(
        &self,
        fields: impl IntoIterator<Item = (String, FieldClass)>,
        flags: PrefillFlags,
    ) -> Self {
        let mut combined: BTreeMap<_, _> = self
            .fields()
            .map(|(path, class)| (path.to_owned(), class))
            .collect();
        combined.extend(fields);
        let mut prefill_flags = self.prefill_flags.clone();
        prefill_flags.before_first.extend(flags.before_first);
        prefill_flags.before_last.extend(flags.before_last);
        Self::new(combined, prefill_flags)
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

    /// P/D: prompt values are swapped in; prefill's sequence prefix overwrites
    /// decode's in place, and decode's value is dropped without one.
    fn join_prefill(&self, prefill: &mut Value, decode: &mut Value, codec: &impl ClientCodec) {
        // Prefill generates the first token. If it is also the last, decode
        // never runs a model step and has no sequence value of its own.
        // Read the attempt-local count before migration offsets are added.
        let prefill_only = decode
            .pointer("/meta_info/completion_tokens")
            .and_then(Value::as_u64)
            == Some(1)
            && matches!(
                decode
                    .pointer("/meta_info/finish_reason/type")
                    .and_then(Value::as_str),
                Some("stop" | "length")
            );
        self.swap(FieldClass::Prompt, prefill, decode);
        for field in self.of(FieldClass::SequenceFinal) {
            let Some(target) = object(decode, &field.parent) else {
                continue;
            };
            let source = object(prefill, &field.parent).and_then(|o| o.get(&field.key));
            if prefill_only
                && !target.contains_key(&field.key)
                && let Some(source) = source
            {
                target.insert(field.key.clone(), source.clone());
            }
            let joined = match (source, target.get_mut(&field.key)) {
                (_, None) => true,
                (Some(source), Some(target)) => codec.overlay_sequence(&field.path, source, target),
                _ => false,
            };
            if !joined {
                target.remove(&field.key);
            }
        }
    }

    /// A copy of the `class` values in `from`, or `None` when it has none.
    fn capture(&self, class: FieldClass, from: &Value) -> Option<Value> {
        let mut captured = Value::Object(Map::new());
        let mut any = false;
        for field in self.of(class) {
            let Some(value) = lookup(from, field) else {
                continue;
            };
            object_or_insert(&mut captured, &field.parent)
                .expect("captured parents are objects")
                .insert(field.key.clone(), value.clone());
            any = true;
        }
        any.then_some(captured)
    }

    fn has(&self, class: FieldClass, value: &Value) -> bool {
        self.of(class).any(|field| lookup(value, field).is_some())
    }

    /// Keep `old`'s prompt values in `new`: the first prefill computed them
    /// over the client's prompt.
    fn adopt_prompt(&self, old: &mut Value, new: &mut Value) {
        for field in self.of(FieldClass::Prompt) {
            let Some(value) = object(old, &field.parent).and_then(|o| o.remove(&field.key)) else {
                continue;
            };
            if let Some(target) = object_or_insert(new, &field.parent) {
                target.insert(field.key.clone(), value);
            }
        }
    }

    fn cumulative_values(&self, chunk: &Value, seen: &mut [Option<u64>]) {
        for (slot, &index) in seen.iter_mut().zip(&self.cumulative) {
            if let Some(n) = lookup(chunk, &self.fields[index]).and_then(Value::as_u64) {
                *slot = Some(n);
            }
        }
    }

    fn add_offsets(&self, offsets: &[u64], delivered: u64, chunk: &mut Value) {
        let add = |field: &Field, chunk: &mut Value, offset: u64| {
            if let Some(value) = object(chunk, &field.parent).and_then(|o| o.get_mut(&field.key))
                && let Some(n) = value.as_u64()
            {
                *value = Value::from(n.saturating_add(offset));
            }
        };
        for (&offset, &index) in offsets.iter().zip(&self.cumulative) {
            add(&self.fields[index], chunk, offset);
        }
        for field in self.of(FieldClass::OutputOffset) {
            add(field, chunk, delivered);
        }
    }
}

fn unescape(token: &str) -> String {
    token.replace("~1", "/").replace("~0", "~")
}

fn lookup<'a>(value: &'a Value, field: &Field) -> Option<&'a Value> {
    field
        .parent
        .iter()
        .try_fold(value, |v, k| v.get(k))?
        .get(&field.key)
}

fn object<'a>(mut value: &'a mut Value, path: &[String]) -> Option<&'a mut Map<String, Value>> {
    for key in path {
        value = value.get_mut(key.as_str())?;
    }
    value.as_object_mut()
}

fn object_or_insert<'a>(
    mut value: &'a mut Value,
    path: &[String],
) -> Option<&'a mut Map<String, Value>> {
    for key in path {
        value = value
            .as_object_mut()?
            .entry(key.clone())
            .or_insert_with(|| Value::Object(Map::new()));
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
pub(crate) fn overlay_base64_prefix(prefix: &str, target: &mut String) -> bool {
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

/// Builds one client response from engine pieces across every attempt.
pub struct Assembler<C = Native> {
    codec: Arc<C>,
    rules: Arc<RuleSet>,
    /// Keep what a retry needs from the first attempt.
    capture: bool,
    prefill: Option<Value>,
    /// Prompt values from the first attempt, without P/D.
    prompt: Option<Value>,
    first: Option<Value>,
    /// Earlier attempts' cumulative counts, and the current attempt's.
    offsets: Vec<u64>,
    seen: Vec<Option<u64>>,
    /// Generated tokens delivered before the current attempt.
    delivered: u64,
    attempt: u32,
    joined: Option<Value>,
}

impl Assembler {
    /// `capture` keeps the first attempt's prompt-side values for migration.
    pub fn new(rules: Arc<RuleSet>, capture: bool) -> Self {
        Self::with_codec(rules, capture, Arc::new(Native))
    }
}

impl<C: ClientCodec> Assembler<C> {
    pub fn with_codec(rules: Arc<RuleSet>, capture: bool, codec: Arc<C>) -> Self {
        let n = rules.cumulative.len();
        Self {
            codec,
            rules,
            capture,
            prefill: None,
            prompt: None,
            first: None,
            offsets: vec![0; n],
            seen: vec![None; n],
            delivered: 0,
            attempt: 0,
            joined: None,
        }
    }

    /// A later prefill keeps the first one's prompt values.
    pub fn set_prefill(&mut self, mut prefill: Value) {
        if let Some(mut old) = self.prefill.take() {
            self.rules.adopt_prompt(&mut old, &mut prefill);
        }
        self.prefill = Some(prefill);
    }

    /// Whether the client has prompt-side values a retry need not recompute.
    pub fn prompt_delivered(&self) -> bool {
        self.prompt.is_some()
            || self
                .prefill
                .as_ref()
                .is_some_and(|p| self.rules.has(FieldClass::Prompt, p))
    }

    pub fn attempt(&self) -> u32 {
        self.attempt
    }

    /// End the current attempt after `delivered` generated tokens in total;
    /// later chunks continue it.
    pub fn migrate(&mut self, delivered: u64) {
        for (offset, seen) in self.offsets.iter_mut().zip(&mut self.seen) {
            *offset = offset.saturating_add(seen.take().unwrap_or(0));
        }
        self.delivered = delivered;
        self.attempt += 1;
    }

    fn observe(&mut self, chunk: &Value) {
        self.rules.cumulative_values(chunk, &mut self.seen);
        // Until a token is delivered, a retry's prompt is the client's.
        if self.capture && self.delivered == 0 {
            if self.first.is_none() {
                self.first = self.rules.capture(FieldClass::FirstAttempt, chunk);
            }
            if self.prompt.is_none() && self.prefill.is_none() {
                self.prompt = self.rules.capture(FieldClass::Prompt, chunk);
            }
        }
    }

    /// Swap the request's joined values into `chunk`; `undo` swaps them back.
    fn apply(&mut self, chunk: &mut Value, undo: bool) {
        let retry = self.attempt > 0;
        if let Some(prefill) = &mut self.prefill {
            if undo {
                self.rules.swap(FieldClass::Prompt, prefill, chunk);
            } else {
                self.rules.join_prefill(prefill, chunk, self.codec.as_ref());
            }
        } else if retry && let Some(prompt) = &mut self.prompt {
            self.rules.swap(FieldClass::Prompt, prompt, chunk);
        }
        if retry && let Some(first) = &mut self.first {
            self.rules.swap(FieldClass::FirstAttempt, first, chunk);
        }
        if retry && !undo {
            self.rules.add_offsets(&self.offsets, self.delivered, chunk);
        }
    }

    /// Streaming: `render` sees `chunk` joined with prefill and earlier
    /// attempts; the joined values are swapped back out afterwards.
    pub fn render<T>(&mut self, mut chunk: Value, render: impl FnOnce(&Value) -> T) -> T {
        self.observe(&chunk);
        self.apply(&mut chunk, false);
        let rendered = render(&chunk);
        self.apply(&mut chunk, true);
        rendered
    }

    /// Unary: join `chunk` onto the response so far.
    pub fn push(&mut self, chunk: Value) {
        self.observe(&chunk);
        match &mut self.joined {
            Some(joined) => self.rules.join_step(joined, chunk),
            None => self.joined = Some(chunk),
        }
    }

    /// Unary: the joined response, or `None` when no chunk arrived.
    pub fn finish(&mut self) -> Option<Value> {
        let mut response = self.joined.take()?;
        self.apply(&mut response, false);
        Some(response)
    }
}

#[cfg(test)]
mod tests {
    //! Joining, P/D, and migration are checked end to end by the
    //! `testkit` laws; these pin the byte-level overlay.
    use proptest::prelude::*;
    use serde_json::json;

    use super::*;

    #[test]
    fn missing_decode_sequence_requires_complete_valid_prefill() {
        for count in [Value::Null, json!(0), json!(1), json!(2)] {
            for finish in [Value::Null, json!("stop"), json!("length"), json!("abort")] {
                for prefix in [Value::Null, json!("AB=="), json!("AQID")] {
                    let complete = count == 1
                        && matches!(finish.as_str(), Some("stop" | "length"))
                        && prefix == "AQID";
                    let mut prefill = json!({"meta_info": {"routed_experts": prefix}});
                    let mut response = json!({"meta_info": {
                        "completion_tokens": count, "finish_reason": {"type": finish}
                    }});
                    crate::rules().join_prefill(&mut prefill, &mut response, &Native);
                    assert_eq!(
                        response.pointer("/meta_info/routed_experts"),
                        complete.then_some(&prefix),
                        "count={count}, finish={finish}, prefix={prefix}"
                    );
                }
            }
        }
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(512))]

        #[test]
        fn retry_counters_do_not_wrap(
            counts in prop::collection::vec(any::<u64>(), 1..4),
            final_count in prop_oneof![Just(u64::MAX), any::<u64>()],
            delivered in 1u64..10,
        ) {
            let rules = RuleSet::from_json(r#"{"fields": {
                "/total": "cumulative", "/completed": "output_offset"
            }}"#).unwrap();
            let mut assembler = Assembler::new(Arc::new(rules), true);
            for count in &counts {
                assembler.render(json!({"total": count}), |_| {});
                assembler.migrate(delivered);
            }
            assembler.render(json!({"total": final_count, "completed": final_count}), |joined| {
                let total: u128 = counts.iter().map(|&n| u128::from(n)).sum();
                assert_eq!(joined["total"], (total + u128::from(final_count)).min(u128::from(u64::MAX)) as u64);
                assert_eq!(joined["completed"], (u128::from(final_count) + u128::from(delivered)).min(u128::from(u64::MAX)) as u64);
            });
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
