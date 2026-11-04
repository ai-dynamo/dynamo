// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Sessions for requests whose headers name none, derived from their input.
//!
//! A key is derived from the request alone, the same way in every process,
//! and becomes an ordinary session ID, so its pin is kept and shared like any
//! other session's. Change the `v1` in the seed whenever the derivation
//! changes: processes on different versions then miss each other's pins
//! instead of misreading them.

use std::fmt;

use serde::de::{DeserializeSeed, Deserializer, IgnoredAny, MapAccess, SeqAccess, Visitor};
use serde_json::value::RawValue;
use xxhash_rust::const_xxh3::xxh3_64 as const_xxh3_64;
use xxhash_rust::xxh3::xxh3_64_with_seed;

const PREFIX_SEED: u64 = const_xxh3_64(b"dynamo.global-router.prefix-key.v1");

/// Fields that clients change in messages they send again: cache
/// breakpoints, and reasoning, which they drop from earlier turns.
const SKIPPED_FIELDS: &[&str] = &["cache_control", "reasoning", "reasoning_content"];

/// Objects, by their `type`, that clients drop from earlier turns: Responses
/// reasoning items and Anthropic thinking parts. Other types are not text.
const SKIPPED_TYPES: &[&str] = &["reasoning", "thinking", "redacted_thinking"];

// Seeds for each kind of value, so that `"1"` and `1` differ.
const STRING: u64 = 1;
const NUMBER: u64 = 2;
const BOOL: u64 = 3;

/// One prefix of a request's input: see [`RequestInput::prefix_keys`].
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct PrefixKey(pub(crate) u64);

impl PrefixKey {
    /// The session ID of the conversation whose input this prefix is. It
    /// starts with a control character, which header values cannot hold, so
    /// it never names a client's session.
    pub fn session_id(self) -> String {
        format!("\u{1}px:{:016x}", self.0)
    }
}

/// The fields of a request body that hold its input, borrowed from the body:
/// Anthropic's `system`, `messages` (chat completions and Anthropic), and
/// `input` (Responses).
#[derive(Clone, Copy, Debug, Default)]
pub struct RequestInput<'a> {
    pub system: Option<&'a RawValue>,
    pub messages: Option<&'a RawValue>,
    pub input: Option<&'a RawValue>,
}

impl RequestInput<'_> {
    /// One key for each prefix of the input, shortest first: the input is a
    /// list of elements (`system`, then each message or `input` item, an
    /// `input` string being one user message), and each key covers `model`
    /// and the elements up to its own, as the tokenizer cache keys each
    /// prefix of a prompt. A conversation's next turn sends this turn's input
    /// again with more after it, so its keys include this turn's last key.
    ///
    /// An element counts by its text: the set of its string, number, and bool
    /// values, without field names, structure, order, [`SKIPPED_FIELDS`], or
    /// objects of [`SKIPPED_TYPES`]. So a client may reorder fields, move
    /// cache breakpoints, drop reasoning, or turn a string into text parts
    /// between turns and keep its keys. Elements with no text count as none.
    /// Elements whose values differ only in arrangement collide, which at
    /// worst routes a request with another conversation's.
    ///
    /// Empty if the input is not as described.
    pub fn prefix_keys(&self, model: &str) -> Vec<PrefixKey> {
        let mut chain = Chain {
            key: xxh3_64_with_seed(model.as_bytes(), PREFIX_SEED),
            keys: Vec::new(),
        };
        let mut read = || -> serde_json::Result<()> {
            if let Some(system) = self.system {
                chain.push(Text.deserialize(&mut json(system))?);
            }
            for list in [self.messages, self.input].into_iter().flatten() {
                json(list).deserialize_any(Elements(&mut chain))?;
            }
            Ok(())
        };
        match read() {
            Ok(()) => chain.keys,
            Err(_) => Vec::new(),
        }
    }
}

fn json(value: &RawValue) -> serde_json::Deserializer<serde_json::de::StrRead<'_>> {
    serde_json::Deserializer::from_str(value.get())
}

fn leaf(seed: u64, bytes: &[u8]) -> u64 {
    xxh3_64_with_seed(bytes, seed)
}

/// Empty strings count as absent, as nulls do.
fn string(value: &str) -> u64 {
    if value.is_empty() {
        0
    } else {
        leaf(STRING, value.as_bytes())
    }
}

/// The key of each prefix so far.
struct Chain {
    key: u64,
    keys: Vec<PrefixKey>,
}

impl Chain {
    fn push(&mut self, text: u64) {
        if text != 0 {
            self.key = xxh3_64_with_seed(&text.to_le_bytes(), self.key);
            self.keys.push(PrefixKey(self.key));
        }
    }
}

/// Adds a list of elements to a chain: each item of an array, or a string as
/// the user message it stands for.
struct Elements<'c>(&'c mut Chain);

impl<'de> Visitor<'de> for Elements<'_> {
    type Value = ();

    fn expecting(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        formatter.write_str("an array or a string")
    }

    fn visit_seq<A: SeqAccess<'de>>(self, mut seq: A) -> Result<(), A::Error> {
        while let Some(text) = seq.next_element_seed(Text)? {
            self.0.push(text);
        }
        Ok(())
    }

    fn visit_str<E>(self, value: &str) -> Result<(), E> {
        self.0.push(string("user").wrapping_add(string(value)));
        Ok(())
    }
}

/// A JSON value's text, as a sum of its values' hashes: see
/// [`RequestInput::prefix_keys`]. Zero for none.
struct Text;

impl<'de> DeserializeSeed<'de> for Text {
    type Value = u64;

    fn deserialize<D: Deserializer<'de>>(self, deserializer: D) -> Result<u64, D::Error> {
        deserializer.deserialize_any(self)
    }
}

impl<'de> Visitor<'de> for Text {
    type Value = u64;

    fn expecting(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        formatter.write_str("JSON")
    }

    fn visit_bool<E>(self, value: bool) -> Result<u64, E> {
        Ok(leaf(BOOL, &[value as u8]))
    }

    fn visit_i64<E>(self, value: i64) -> Result<u64, E> {
        Ok(leaf(NUMBER, &value.to_le_bytes()))
    }

    fn visit_u64<E>(self, value: u64) -> Result<u64, E> {
        Ok(leaf(NUMBER, &value.to_le_bytes()))
    }

    fn visit_f64<E>(self, value: f64) -> Result<u64, E> {
        Ok(leaf(NUMBER, &value.to_le_bytes()))
    }

    fn visit_str<E>(self, value: &str) -> Result<u64, E> {
        Ok(string(value))
    }

    fn visit_unit<E>(self) -> Result<u64, E> {
        Ok(0)
    }

    fn visit_seq<A: SeqAccess<'de>>(self, mut seq: A) -> Result<u64, A::Error> {
        let mut text = 0u64;
        while let Some(item) = seq.next_element_seed(Text)? {
            text = text.wrapping_add(item);
        }
        Ok(text)
    }

    fn visit_map<A: MapAccess<'de>>(self, mut map: A) -> Result<u64, A::Error> {
        let mut text = 0u64;
        let mut skipped = false;
        while let Some(field) = map.next_key_seed(FieldName)? {
            match field {
                Field::Type => skipped |= map.next_value_seed(SkippedType)?,
                Field::Skipped => drop(map.next_value::<IgnoredAny>()?),
                Field::Text => text = text.wrapping_add(map.next_value_seed(Text)?),
            }
        }
        Ok(if skipped { 0 } else { text })
    }
}

enum Field {
    /// `type`: read only to find [`SKIPPED_TYPES`].
    Type,
    Skipped,
    Text,
}

/// Reads a field name.
struct FieldName;

impl<'de> DeserializeSeed<'de> for FieldName {
    type Value = Field;

    fn deserialize<D: Deserializer<'de>>(self, deserializer: D) -> Result<Field, D::Error> {
        deserializer.deserialize_str(self)
    }
}

impl<'de> Visitor<'de> for FieldName {
    type Value = Field;

    fn expecting(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        formatter.write_str("a field name")
    }

    fn visit_str<E>(self, name: &str) -> Result<Field, E> {
        Ok(match name {
            "type" => Field::Type,
            name if SKIPPED_FIELDS.contains(&name) => Field::Skipped,
            _ => Field::Text,
        })
    }
}

/// Reads a `type` value: whether it is one of [`SKIPPED_TYPES`].
struct SkippedType;

impl<'de> DeserializeSeed<'de> for SkippedType {
    type Value = bool;

    fn deserialize<D: Deserializer<'de>>(self, deserializer: D) -> Result<bool, D::Error> {
        deserializer.deserialize_str(self)
    }
}

impl<'de> Visitor<'de> for SkippedType {
    type Value = bool;

    fn expecting(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        formatter.write_str("a type name")
    }

    fn visit_str<E>(self, name: &str) -> Result<bool, E> {
        Ok(SKIPPED_TYPES.contains(&name))
    }
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::*;

    /// The keys of a chat completions request with these messages.
    fn chat(messages: serde_json::Value) -> Vec<PrefixKey> {
        let messages = serde_json::to_string(&messages).unwrap();
        let messages = RawValue::from_string(messages).unwrap();
        let input = RequestInput {
            messages: Some(&messages),
            ..Default::default()
        };
        input.prefix_keys("qwen")
    }

    #[test]
    fn the_next_turn_extends_this_turns_keys() {
        let first = chat(json!([
            {"role": "system", "content": "You are terse."},
            {"role": "user", "content": "hi"},
        ]));
        let second = chat(json!([
            {"role": "system", "content": "You are terse."},
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "Hello."},
            {"role": "user", "content": "bye"},
        ]));
        assert_eq!(first.len(), 2);
        assert_eq!(second[..2], first[..]);
        // Another conversation shares only the system prompt's key.
        let other = chat(json!([
            {"role": "system", "content": "You are terse."},
            {"role": "user", "content": "hello"},
        ]));
        assert_eq!(other[0], first[0]);
        assert_ne!(other[1], first[1]);
    }

    #[test]
    fn rewrites_that_keep_the_text_keep_the_keys() {
        let sent = chat(json!([
            {"role": "user", "content": [
                {"type": "text", "text": "read it", "cache_control": {"type": "ephemeral"}},
            ]},
            {"type": "reasoning", "id": "rs_1", "summary": [{"type": "summary_text", "text": "hmm"}]},
            {"role": "assistant", "reasoning_content": "they want it read", "content": [
                {"type": "thinking", "thinking": "hmm", "signature": "abc"},
                {"type": "tool_use", "id": "t1", "name": "read", "input": {"path": "a.rs"}},
            ]},
            {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "t1", "content": "fn main"}]},
        ]));
        // Sent again with the fields reordered, and without cache breakpoints
        // or reasoning.
        let resent = RawValue::from_string(
            r#"[
                {"content": "read it", "role": "user"},
                {"content": [{"input": {"path": "a.rs"}, "name": "read", "id": "t1", "type": "tool_use"}], "role": "assistant"},
                {"content": [{"content": "fn main", "tool_use_id": "t1", "type": "tool_result"}], "role": "user"}
            ]"#
            .to_owned(),
        )
        .unwrap();
        let resent = RequestInput {
            messages: Some(&resent),
            ..Default::default()
        };
        assert_eq!(sent, resent.prefix_keys("qwen"));
        // Any change to the text changes the keys from there on.
        let edited = chat(json!([
            {"role": "user", "content": "read it"},
            {"role": "assistant", "content": [
                {"type": "tool_use", "id": "t1", "name": "read", "input": {"path": "b.rs"}},
            ]},
        ]));
        assert_eq!(edited[0], sent[0]);
        assert_ne!(edited[1], sent[1]);
    }

    #[test]
    fn keys_cover_the_model_system_and_input() {
        let system = RawValue::from_string(r#""Be brief.""#.to_owned()).unwrap();
        let messages =
            RawValue::from_string(r#"[{"role":"user","content":"hi"}]"#.to_owned()).unwrap();
        let anthropic = RequestInput {
            system: Some(&system),
            messages: Some(&messages),
            ..Default::default()
        };
        let keys = anthropic.prefix_keys("claude");
        assert_eq!(keys.len(), 2);
        assert_ne!(keys, anthropic.prefix_keys("qwen"));

        // A Responses string input is the user message the next turn sends.
        let string = RawValue::from_string(r#""hi""#.to_owned()).unwrap();
        let items = RawValue::from_string(
            r#"[{"type":"message","role":"user","content":[{"type":"input_text","text":"hi"}]},
                {"type":"message","role":"assistant","content":[{"type":"output_text","text":"Hello."}]}]"#
                .to_owned(),
        )
        .unwrap();
        let responses = |input| {
            RequestInput {
                input: Some(input),
                ..Default::default()
            }
            .prefix_keys("qwen")
        };
        assert_eq!(responses(&string)[..], responses(&items)[..1]);
    }

    #[test]
    fn no_keys_without_valid_input() {
        assert!(RequestInput::default().prefix_keys("qwen").is_empty());
        let object = RawValue::from_string(r#"{"role":"user"}"#.to_owned()).unwrap();
        let input = RequestInput {
            messages: Some(&object),
            ..Default::default()
        };
        assert!(input.prefix_keys("qwen").is_empty());
    }

    /// Session IDs are shared between processes: a change here breaks pins
    /// across versions, so it must come with a new seed.
    #[test]
    fn session_ids_are_stable() {
        // A literal, since `json!` orders fields by serde_json's features.
        let messages =
            RawValue::from_string(r#"[{"role":"user","content":"hi"}]"#.to_owned()).unwrap();
        let input = RequestInput {
            messages: Some(&messages),
            ..Default::default()
        };
        let keys = input.prefix_keys("qwen");
        assert_eq!(keys, [PrefixKey(0x7d30_387a_9afe_37a7)]);
        assert_eq!(keys[0].session_id(), "\u{1}px:7d30387a9afe37a7");
    }
}
