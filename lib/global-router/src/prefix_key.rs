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

use serde::de::{
    DeserializeSeed, Deserializer, IgnoredAny, IntoDeserializer, MapAccess, SeqAccess, Visitor,
};
use serde_json::value::RawValue;
use xxhash_rust::const_xxh3::xxh3_64 as const_xxh3_64;
use xxhash_rust::xxh3::Xxh3;

const PREFIX_SEED: u64 = const_xxh3_64(b"dynamo.global-router.prefix-key.v1");

/// Fields that say nothing about the conversation, or that clients change
/// in messages they send again: cache breakpoints, part types, and
/// reasoning, which clients drop from earlier turns.
const SKIPPED_FIELDS: &[&str] = &[
    "cache_control",
    "type",
    "thinking",
    "signature",
    "reasoning",
    "reasoning_content",
    "encrypted_content",
];

// What each hashed leaf starts with, so that leaves cannot run together.
const STRING: u8 = 1;
const NUMBER: u8 = 2;
const BOOL: u8 = 3;
const ELEMENT_END: u8 = 4;

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
    /// list of elements (`system`, then each message or `input` item, or an
    /// `input` string as one element), and each key covers `model` and the
    /// elements up to its own, as the tokenizer cache keys each prefix of a
    /// prompt. A conversation's next turn sends this turn's input again with
    /// more after it, so its keys include this turn's last key.
    ///
    /// Only an element's text counts: its string, number, and bool values in
    /// order, without field names, structure, or [`SKIPPED_FIELDS`]. So a
    /// client may move cache breakpoints, drop reasoning, or turn a string
    /// into text parts between turns and keep its keys.
    ///
    /// Empty if the input has no elements or is not as described.
    pub fn prefix_keys(&self, model: &str) -> Vec<PrefixKey> {
        let mut hasher = Xxh3::with_seed(PREFIX_SEED);
        hasher.update(&(model.len() as u64).to_le_bytes());
        hasher.update(model.as_bytes());
        let mut keys = Vec::new();
        let mut elements = |value: &RawValue, list: bool| {
            let mut json = serde_json::Deserializer::from_str(value.get());
            let element = Element {
                hasher: &mut hasher,
                keys: &mut keys,
            };
            if list {
                (&mut json).deserialize_any(Elements(element))
            } else {
                element.deserialize(&mut json)
            }
        };
        let read = self
            .system
            .map_or(Ok(()), |system| elements(system, false))
            .and_then(|()| {
                self.messages
                    .map_or(Ok(()), |messages| elements(messages, true))
            })
            .and_then(|()| self.input.map_or(Ok(()), |input| elements(input, true)));
        if read.is_err() {
            return Vec::new();
        }
        keys
    }
}

/// Hashes one element's text, then keys the prefix it ends.
struct Element<'h> {
    hasher: &'h mut Xxh3,
    keys: &'h mut Vec<PrefixKey>,
}

impl<'de> DeserializeSeed<'de> for Element<'_> {
    type Value = ();

    fn deserialize<D: Deserializer<'de>>(self, deserializer: D) -> Result<(), D::Error> {
        Text(&mut *self.hasher).deserialize(deserializer)?;
        self.hasher.update(&[ELEMENT_END]);
        self.keys.push(PrefixKey(self.hasher.digest()));
        Ok(())
    }
}

/// Reads a list of elements, in one pass: each item of an array, or a string
/// as one element.
struct Elements<'h>(Element<'h>);

impl<'de> Visitor<'de> for Elements<'_> {
    type Value = ();

    fn expecting(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        formatter.write_str("an array or a string")
    }

    fn visit_seq<A: SeqAccess<'de>>(self, mut seq: A) -> Result<(), A::Error> {
        let Element { hasher, keys } = self.0;
        while seq
            .next_element_seed(Element {
                hasher: &mut *hasher,
                keys: &mut *keys,
            })?
            .is_some()
        {}
        Ok(())
    }

    fn visit_str<E: serde::de::Error>(self, value: &str) -> Result<(), E> {
        self.0.deserialize(value.into_deserializer())
    }
}

/// Feeds a JSON value's text into a hasher: see [`RequestInput::prefix_keys`].
struct Text<'h>(&'h mut Xxh3);

impl<'de> DeserializeSeed<'de> for Text<'_> {
    type Value = ();

    fn deserialize<D: Deserializer<'de>>(self, deserializer: D) -> Result<(), D::Error> {
        deserializer.deserialize_any(self)
    }
}

impl<'de> Visitor<'de> for Text<'_> {
    type Value = ();

    fn expecting(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        formatter.write_str("JSON")
    }

    fn visit_bool<E>(self, value: bool) -> Result<(), E> {
        self.0.update(&[BOOL, value as u8]);
        Ok(())
    }

    fn visit_i64<E>(self, value: i64) -> Result<(), E> {
        self.0.update(&[NUMBER]);
        self.0.update(&value.to_le_bytes());
        Ok(())
    }

    fn visit_u64<E>(self, value: u64) -> Result<(), E> {
        self.0.update(&[NUMBER]);
        self.0.update(&value.to_le_bytes());
        Ok(())
    }

    fn visit_f64<E>(self, value: f64) -> Result<(), E> {
        self.0.update(&[NUMBER]);
        self.0.update(&value.to_le_bytes());
        Ok(())
    }

    /// Empty strings count as absent, as nulls do.
    fn visit_str<E>(self, value: &str) -> Result<(), E> {
        if !value.is_empty() {
            self.0.update(&[STRING]);
            self.0.update(&(value.len() as u64).to_le_bytes());
            self.0.update(value.as_bytes());
        }
        Ok(())
    }

    fn visit_unit<E>(self) -> Result<(), E> {
        Ok(())
    }

    fn visit_seq<A: SeqAccess<'de>>(self, mut seq: A) -> Result<(), A::Error> {
        while seq.next_element_seed(Text(&mut *self.0))?.is_some() {}
        Ok(())
    }

    fn visit_map<A: MapAccess<'de>>(self, mut map: A) -> Result<(), A::Error> {
        while let Some(skipped) = map.next_key_seed(Skipped)? {
            if skipped {
                map.next_value::<IgnoredAny>()?;
            } else {
                map.next_value_seed(Text(&mut *self.0))?;
            }
        }
        Ok(())
    }
}

/// Reads a field name: whether it is one of [`SKIPPED_FIELDS`].
struct Skipped;

impl<'de> DeserializeSeed<'de> for Skipped {
    type Value = bool;

    fn deserialize<D: Deserializer<'de>>(self, deserializer: D) -> Result<bool, D::Error> {
        deserializer.deserialize_str(self)
    }
}

impl<'de> Visitor<'de> for Skipped {
    type Value = bool;

    fn expecting(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        formatter.write_str("a field name")
    }

    fn visit_str<E>(self, name: &str) -> Result<bool, E> {
        Ok(SKIPPED_FIELDS.contains(&name))
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
            {"role": "assistant", "reasoning_content": "they want it read", "content": [
                {"type": "thinking", "thinking": "hmm", "signature": "abc"},
                {"type": "tool_use", "id": "t1", "name": "read", "input": {"path": "a.rs"}},
            ]},
            {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "t1", "content": "fn main"}]},
        ]));
        let resent = chat(json!([
            {"role": "user", "content": "read it"},
            {"role": "assistant", "content": [
                {"type": "tool_use", "id": "t1", "name": "read", "input": {"path": "a.rs"}},
            ]},
            {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "t1", "content": "fn main"}]},
        ]));
        assert_eq!(sent, resent);
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

        let string = RawValue::from_string(r#""hi""#.to_owned()).unwrap();
        let items = RawValue::from_string(
            r#"[{"type":"message","role":"user","content":[{"type":"input_text","text":"hi"}]}]"#
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
        assert_eq!(responses(&string).len(), 1);
        assert_eq!(responses(&items).len(), 1);
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
        assert_eq!(keys, [PrefixKey(0x02d3_874b_28bf_12a9)]);
        assert_eq!(keys[0].session_id(), "\u{1}px:02d3874b28bf12a9");
    }
}
