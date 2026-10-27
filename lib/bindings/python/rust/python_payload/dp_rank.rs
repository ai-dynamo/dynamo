// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Records the DP rank a Python worker's request body names while the body is
//! decoded, for backend admission.
//!
//! The body is transcoded straight from the request-plane codec into the Python
//! object its handler receives, with no Rust value tree in between, and backend
//! admission must not read that object back. [`DpRankProbe`] wraps the codec's
//! deserializer for that one transcoding pass: every event still reaches the
//! transcoder unchanged, and the probe only watches the top-level `routing`
//! entry and, inside it, the `dp_rank` entry.
//!
//! It reads the rank as the handler reads its dict, so the two agree:
//!
//! - Only a string key matches, as only a `str` key equals `"routing"` in
//!   Python.
//! - A repeated key is decided by its last entry, as a dict keeps the last value
//!   set for a key.
//! - A rank is an integer that fits a `u32`. `null`, any other value, or no entry
//!   names no rank.
//!
//! The probe is only ever driven by `serde_transcode`, which asks each
//! deserializer for `deserialize_any` alone, so every other request is
//! forwarded to it.

use std::cell::Cell;
use std::fmt;

use serde::de::{self, DeserializeSeed, Deserializer, EnumAccess, MapAccess, SeqAccess, Visitor};

/// The value a probe wraps.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Level {
    /// The whole request body.
    Body,
    /// The body's `routing` entry.
    Routing,
    /// The `dp_rank` entry of `routing`.
    DpRank,
}

impl Level {
    /// The key of the one entry a map at this level is watched for, and the
    /// level of that entry's value.
    fn watched_entry(self) -> Option<(&'static str, Level)> {
        match self {
            Level::Body => Some(("routing", Level::Routing)),
            Level::Routing => Some(("dp_rank", Level::DpRank)),
            Level::DpRank => None,
        }
    }
}

/// A deserializer that records into `rank` the DP rank of the request body it
/// yields, while passing every event through unchanged.
pub(super) struct DpRankProbe<'a, D> {
    inner: D,
    level: Level,
    rank: &'a Cell<Option<u32>>,
}

impl<'a, D> DpRankProbe<'a, D> {
    /// Probe a whole request body. `rank` must start as `None`, and holds the
    /// body's rank once the body is fully deserialized.
    pub(super) fn body(inner: D, rank: &'a Cell<Option<u32>>) -> Self {
        Self {
            inner,
            level: Level::Body,
            rank,
        }
    }
}

impl<'de, D: Deserializer<'de>> Deserializer<'de> for DpRankProbe<'_, D> {
    type Error = D::Error;

    fn deserialize_any<V: Visitor<'de>>(self, visitor: V) -> Result<V::Value, D::Error> {
        self.inner.deserialize_any(ProbeVisitor {
            inner: visitor,
            level: self.level,
            rank: self.rank,
        })
    }

    fn is_human_readable(&self) -> bool {
        self.inner.is_human_readable()
    }

    serde::forward_to_deserialize_any! {
        bool i8 i16 i32 i64 i128 u8 u16 u32 u64 u128 f32 f64 char str string bytes
        byte_buf option unit unit_struct newtype_struct seq tuple tuple_struct map
        struct enum identifier ignored_any
    }
}

/// Forward each listed visit to the wrapped visitor unchanged.
macro_rules! forward_visits {
    ($($method:ident($ty:ty)),* $(,)?) => {$(
        fn $method<E: de::Error>(self, value: $ty) -> Result<Self::Value, E> {
            self.inner.$method(value)
        }
    )*};
}

/// Record each listed integer visit as the rank, at the `dp_rank` level, then
/// forward it.
macro_rules! record_integers {
    ($($method:ident($ty:ty)),* $(,)?) => {$(
        fn $method<E: de::Error>(self, value: $ty) -> Result<Self::Value, E> {
            if self.level == Level::DpRank {
                self.rank.set(u32::try_from(value).ok());
            }
            self.inner.$method(value)
        }
    )*};
}

struct ProbeVisitor<'a, V> {
    inner: V,
    level: Level,
    rank: &'a Cell<Option<u32>>,
}

impl<'a, V> ProbeVisitor<'a, V> {
    /// The same probe over a value this one wraps transparently, as `Some` and
    /// newtypes are transparent in the Python object.
    fn wrap<D>(&self, inner: D) -> DpRankProbe<'a, D> {
        DpRankProbe {
            inner,
            level: self.level,
            rank: self.rank,
        }
    }
}

impl<'de, V: Visitor<'de>> Visitor<'de> for ProbeVisitor<'_, V> {
    type Value = V::Value;

    fn expecting(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        self.inner.expecting(formatter)
    }

    record_integers! {
        visit_i8(i8), visit_i16(i16), visit_i32(i32), visit_i64(i64), visit_i128(i128),
        visit_u8(u8), visit_u16(u16), visit_u32(u32), visit_u64(u64), visit_u128(u128),
    }

    forward_visits! {
        visit_bool(bool), visit_f32(f32), visit_f64(f64), visit_char(char),
        visit_str(&str), visit_borrowed_str(&'de str), visit_string(String),
        visit_bytes(&[u8]), visit_borrowed_bytes(&'de [u8]), visit_byte_buf(Vec<u8>),
    }

    fn visit_none<E: de::Error>(self) -> Result<Self::Value, E> {
        self.inner.visit_none()
    }

    fn visit_unit<E: de::Error>(self) -> Result<Self::Value, E> {
        self.inner.visit_unit()
    }

    fn visit_some<D: Deserializer<'de>>(self, deserializer: D) -> Result<Self::Value, D::Error> {
        let probe = self.wrap(deserializer);
        self.inner.visit_some(probe)
    }

    fn visit_newtype_struct<D: Deserializer<'de>>(
        self,
        deserializer: D,
    ) -> Result<Self::Value, D::Error> {
        let probe = self.wrap(deserializer);
        self.inner.visit_newtype_struct(probe)
    }

    fn visit_seq<A: SeqAccess<'de>>(self, seq: A) -> Result<Self::Value, A::Error> {
        self.inner.visit_seq(seq)
    }

    fn visit_map<A: MapAccess<'de>>(self, map: A) -> Result<Self::Value, A::Error> {
        match self.level.watched_entry() {
            Some((key, value_level)) => self.inner.visit_map(ProbeMap {
                inner: map,
                key,
                value_level,
                rank: self.rank,
                watched_next: false,
            }),
            None => self.inner.visit_map(map),
        }
    }

    fn visit_enum<A: EnumAccess<'de>>(self, data: A) -> Result<Self::Value, A::Error> {
        self.inner.visit_enum(data)
    }
}

/// A map whose entry under `key` is probed at `value_level`; every other entry
/// passes through unwrapped.
struct ProbeMap<'a, A> {
    inner: A,
    key: &'static str,
    value_level: Level,
    rank: &'a Cell<Option<u32>>,
    /// Whether the key just read is `key`, so its value is the one to probe.
    watched_next: bool,
}

impl<'de, A: MapAccess<'de>> MapAccess<'de> for ProbeMap<'_, A> {
    type Error = A::Error;

    fn next_key_seed<K: DeserializeSeed<'de>>(
        &mut self,
        seed: K,
    ) -> Result<Option<K::Value>, A::Error> {
        self.watched_next = false;
        self.inner.next_key_seed(KeySeed {
            inner: seed,
            key: self.key,
            matched: &mut self.watched_next,
        })
    }

    fn next_value_seed<S: DeserializeSeed<'de>>(&mut self, seed: S) -> Result<S::Value, A::Error> {
        if !std::mem::take(&mut self.watched_next) {
            return self.inner.next_value_seed(seed);
        }
        // A repeated key replaces the earlier entry, as in a Python dict, so the
        // rank is decided afresh by this one.
        self.rank.set(None);
        self.inner.next_value_seed(ValueSeed {
            inner: seed,
            level: self.value_level,
            rank: self.rank,
        })
    }

    fn size_hint(&self) -> Option<usize> {
        self.inner.size_hint()
    }
}

/// A map key, checked against the watched key as it passes through.
struct KeySeed<'b, S> {
    inner: S,
    key: &'static str,
    matched: &'b mut bool,
}

impl<'de, S: DeserializeSeed<'de>> DeserializeSeed<'de> for KeySeed<'_, S> {
    type Value = S::Value;

    fn deserialize<D: Deserializer<'de>>(self, deserializer: D) -> Result<S::Value, D::Error> {
        self.inner.deserialize(KeyProbe {
            inner: deserializer,
            key: self.key,
            matched: self.matched,
        })
    }
}

struct KeyProbe<'b, D> {
    inner: D,
    key: &'static str,
    matched: &'b mut bool,
}

impl<'de, D: Deserializer<'de>> Deserializer<'de> for KeyProbe<'_, D> {
    type Error = D::Error;

    fn deserialize_any<V: Visitor<'de>>(self, visitor: V) -> Result<V::Value, D::Error> {
        self.inner.deserialize_any(KeyVisitor {
            inner: visitor,
            key: self.key,
            matched: self.matched,
        })
    }

    fn is_human_readable(&self) -> bool {
        self.inner.is_human_readable()
    }

    serde::forward_to_deserialize_any! {
        bool i8 i16 i32 i64 i128 u8 u16 u32 u64 u128 f32 f64 char str string bytes
        byte_buf option unit unit_struct newtype_struct seq tuple tuple_struct map
        struct enum identifier ignored_any
    }
}

/// Sets `matched` when the key is the string `key`; any other key, string or
/// not, leaves it unset.
struct KeyVisitor<'b, V> {
    inner: V,
    key: &'static str,
    matched: &'b mut bool,
}

impl<'de, V: Visitor<'de>> Visitor<'de> for KeyVisitor<'_, V> {
    type Value = V::Value;

    fn expecting(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        self.inner.expecting(formatter)
    }

    fn visit_str<E: de::Error>(self, value: &str) -> Result<Self::Value, E> {
        *self.matched = value == self.key;
        self.inner.visit_str(value)
    }

    fn visit_borrowed_str<E: de::Error>(self, value: &'de str) -> Result<Self::Value, E> {
        *self.matched = value == self.key;
        self.inner.visit_borrowed_str(value)
    }

    fn visit_string<E: de::Error>(self, value: String) -> Result<Self::Value, E> {
        *self.matched = value == self.key;
        self.inner.visit_string(value)
    }

    forward_visits! {
        visit_bool(bool), visit_i8(i8), visit_i16(i16), visit_i32(i32), visit_i64(i64),
        visit_i128(i128), visit_u8(u8), visit_u16(u16), visit_u32(u32), visit_u64(u64),
        visit_u128(u128), visit_f32(f32), visit_f64(f64), visit_char(char),
        visit_bytes(&[u8]), visit_borrowed_bytes(&'de [u8]), visit_byte_buf(Vec<u8>),
    }

    fn visit_none<E: de::Error>(self) -> Result<Self::Value, E> {
        self.inner.visit_none()
    }

    fn visit_unit<E: de::Error>(self) -> Result<Self::Value, E> {
        self.inner.visit_unit()
    }

    fn visit_some<D: Deserializer<'de>>(self, deserializer: D) -> Result<Self::Value, D::Error> {
        self.inner.visit_some(deserializer)
    }

    fn visit_newtype_struct<D: Deserializer<'de>>(
        self,
        deserializer: D,
    ) -> Result<Self::Value, D::Error> {
        self.inner.visit_newtype_struct(deserializer)
    }

    fn visit_seq<A: SeqAccess<'de>>(self, seq: A) -> Result<Self::Value, A::Error> {
        self.inner.visit_seq(seq)
    }

    fn visit_map<A: MapAccess<'de>>(self, map: A) -> Result<Self::Value, A::Error> {
        self.inner.visit_map(map)
    }

    fn visit_enum<A: EnumAccess<'de>>(self, data: A) -> Result<Self::Value, A::Error> {
        self.inner.visit_enum(data)
    }
}

/// The value of a watched entry, probed at `level`.
struct ValueSeed<'a, S> {
    inner: S,
    level: Level,
    rank: &'a Cell<Option<u32>>,
}

impl<'de, S: DeserializeSeed<'de>> DeserializeSeed<'de> for ValueSeed<'_, S> {
    type Value = S::Value;

    fn deserialize<D: Deserializer<'de>>(self, deserializer: D) -> Result<S::Value, D::Error> {
        self.inner.deserialize(DpRankProbe {
            inner: deserializer,
            level: self.level,
            rank: self.rank,
        })
    }
}

#[cfg(test)]
mod tests {
    // Free of Python C API calls, like every Rust test in this crate: an
    // `rmpv::Value` stands in for the Python object, as it keeps every map entry
    // in order, repeated and non-string keys included. The built extension's
    // own behavior is covered by tests/test_request_plane_admission_rank.py.

    use super::*;

    use dynamo_runtime::pipeline::network::RequestPlanePayloadCodec;
    use rmpv::Value;
    use serde::Deserialize;
    use serde::de::Error as _;
    use serde_transcode::Transcoder;

    /// What `PythonPayload` decodes, into an `rmpv::Value` instead of Python.
    struct Probed {
        body: Value,
        dp_rank: Option<u32>,
    }

    impl<'de> Deserialize<'de> for Probed {
        fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
            let dp_rank = Cell::new(None);
            let body =
                rmpv::ext::to_value(Transcoder::new(DpRankProbe::body(deserializer, &dp_rank)))
                    .map_err(D::Error::custom)?;
            Ok(Self {
                body,
                dp_rank: dp_rank.get(),
            })
        }
    }

    /// The same transcoding without the probe.
    struct Plain(Value);

    impl<'de> Deserialize<'de> for Plain {
        fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
            rmpv::ext::to_value(Transcoder::new(deserializer))
                .map(Plain)
                .map_err(D::Error::custom)
        }
    }

    /// The rank a Python handler reads from the dict `body` decodes to:
    /// `(body.get("routing") or {}).get("dp_rank")`, where only a `str` key
    /// matches and the last entry for a key wins, if it is an integer that fits a
    /// `u32`.
    fn handler_rank(body: &Value) -> Option<u32> {
        fn last<'v>(map: &'v Value, key: &str) -> Option<&'v Value> {
            map.as_map()?
                .iter()
                .rev()
                .find(|(entry, _)| entry.as_str() == Some(key))
                .map(|(_, value)| value)
        }
        match last(last(body, "routing")?, "dp_rank")? {
            Value::Integer(rank) => rank.as_u64().and_then(|rank| u32::try_from(rank).ok()),
            _ => None,
        }
    }

    /// Decode `bytes` as a ranked Python request would be, and require that the
    /// probe changed nothing and agrees with the handler. Returns the rank.
    fn decode(codec: RequestPlanePayloadCodec, bytes: &[u8]) -> Option<u32> {
        let probed: Probed = codec.decode(bytes).unwrap();
        let Plain(plain) = codec.decode(bytes).unwrap();
        assert_eq!(probed.body, plain, "the probe must not change the body");
        assert_eq!(
            probed.dp_rank,
            handler_rank(&probed.body),
            "the recorded rank must be the handler's: {}",
            probed.body
        );
        probed.dp_rank
    }

    fn json(text: &str) -> Option<u32> {
        decode(RequestPlanePayloadCodec::Json, text.as_bytes())
    }

    fn msgpack(body: &Value) -> Option<u32> {
        let mut bytes = Vec::new();
        rmpv::encode::write_value(&mut bytes, body).unwrap();
        decode(RequestPlanePayloadCodec::Msgpack, &bytes)
    }

    fn map(entries: Vec<(Value, Value)>) -> Value {
        Value::Map(entries)
    }

    fn key(name: &str) -> Value {
        Value::from(name)
    }

    fn routing(dp_rank: Value) -> Value {
        map(vec![(key("routing"), map(vec![(key("dp_rank"), dp_rank)]))])
    }

    #[test]
    fn json_bodies_name_the_rank_their_handler_reads() {
        // A request as the frontend sends it, with its rank among other routing
        // hints.
        assert_eq!(
            json(
                r#"{"model":"m","token_ids":[1,2],"routing":{"backend_instance_id":7,"dp_rank":4,"prefill_dp_rank":2},"annotations":[]}"#
            ),
            Some(4)
        );
        assert_eq!(json(r#"{"routing":{"dp_rank":0}}"#), Some(0));
        assert_eq!(
            json(r#"{"routing":{"dp_rank":4294967295}}"#),
            Some(u32::MAX)
        );
        // Escaped keys name the same keys once decoded.
        assert_eq!(json(r#"{"rout\u0069ng":{"dp_\u0072ank":7}}"#), Some(7));
    }

    #[test]
    fn json_bodies_without_a_usable_rank_name_none() {
        for body in [
            // Missing or null.
            r#"{}"#,
            r#"{"model":"m"}"#,
            r#"{"routing":null}"#,
            r#"{"routing":{}}"#,
            r#"{"routing":{"dp_rank":null}}"#,
            // Malformed: not an integer that fits a u32.
            r#"{"routing":{"dp_rank":"5"}}"#,
            r#"{"routing":{"dp_rank":5.0}}"#,
            r#"{"routing":{"dp_rank":1e2}}"#,
            r#"{"routing":{"dp_rank":true}}"#,
            r#"{"routing":{"dp_rank":-1}}"#,
            r#"{"routing":{"dp_rank":4294967296}}"#,
            r#"{"routing":{"dp_rank":[5]}}"#,
            r#"{"routing":{"dp_rank":{"rank":5}}}"#,
            r#"{"routing":5}"#,
            r#"{"routing":[{"dp_rank":5}]}"#,
            // Unrelated nesting: only `routing.dp_rank` of the body itself counts.
            r#"{"dp_rank":5}"#,
            r#"{"nested":{"routing":{"dp_rank":5}}}"#,
            r#"{"routing":{"nested":{"dp_rank":5}}}"#,
            r#"{"routing":{"prefill_dp_rank":5}}"#,
            // Bodies that are not maps.
            r#"[{"routing":{"dp_rank":5}}]"#,
            r#""routing""#,
            r#"5"#,
            r#"null"#,
        ] {
            assert_eq!(json(body), None, "{body}");
        }
    }

    #[test]
    fn json_repeated_keys_are_decided_by_their_last_entry() {
        assert_eq!(
            json(r#"{"routing":{"dp_rank":4},"routing":{"dp_rank":5}}"#),
            Some(5)
        );
        assert_eq!(json(r#"{"routing":{"dp_rank":5},"routing":null}"#), None);
        assert_eq!(json(r#"{"routing":{"dp_rank":5},"routing":{}}"#), None);
        assert_eq!(json(r#"{"routing":null,"routing":{"dp_rank":6}}"#), Some(6));
        assert_eq!(json(r#"{"routing":{"dp_rank":4,"dp_rank":null}}"#), None);
        assert_eq!(json(r#"{"routing":{"dp_rank":4,"dp_rank":"x"}}"#), None);
        assert_eq!(json(r#"{"routing":{"dp_rank":null,"dp_rank":6}}"#), Some(6));
        // Entries after the rank, at either level, leave it alone.
        assert_eq!(
            json(r#"{"routing":{"dp_rank":6,"priority":1},"model":"m","dp_rank":7}"#),
            Some(6)
        );
        // A nested `routing` inside `routing` is not the body's.
        assert_eq!(
            json(r#"{"routing":{"dp_rank":5,"routing":{"dp_rank":6}}}"#),
            Some(5)
        );
    }

    #[test]
    fn msgpack_bodies_name_the_rank_their_handler_reads() {
        assert_eq!(msgpack(&routing(Value::from(5))), Some(5));
        assert_eq!(msgpack(&routing(Value::from(u32::MAX))), Some(u32::MAX));
        // Non-string keys and binary values elsewhere in the body pass through.
        assert_eq!(
            msgpack(&map(vec![
                (Value::from(1), key("one")),
                (Value::Binary(vec![0, 0xff]), Value::Binary(vec![1, 2])),
                (
                    key("routing"),
                    map(vec![
                        (Value::from(2), Value::Binary(vec![0])),
                        (key("dp_rank"), Value::from(6)),
                        (Value::Boolean(true), Value::Nil),
                    ]),
                ),
                (key("nested"), routing(Value::from(7))),
            ])),
            Some(6)
        );
    }

    #[test]
    fn msgpack_keys_match_only_as_strings() {
        // A binary key is `bytes` in Python, never equal to the `str` key.
        assert_eq!(
            msgpack(&map(vec![(
                Value::Binary(b"routing".to_vec()),
                map(vec![(key("dp_rank"), Value::from(5))]),
            )])),
            None
        );
        assert_eq!(
            msgpack(&map(vec![(
                key("routing"),
                map(vec![(Value::Binary(b"dp_rank".to_vec()), Value::from(5))]),
            )])),
            None
        );
        // So a binary `routing` after the string one replaces nothing.
        assert_eq!(
            msgpack(&map(vec![
                (key("routing"), map(vec![(key("dp_rank"), Value::from(4))])),
                (
                    Value::Binary(b"routing".to_vec()),
                    map(vec![(key("dp_rank"), Value::from(5))]),
                ),
            ])),
            Some(4)
        );
        assert_eq!(
            msgpack(&map(vec![(Value::from(0), routing(Value::from(5)))])),
            None
        );
    }

    #[test]
    fn msgpack_repeated_keys_are_decided_by_their_last_entry() {
        let rank = |value: Value| map(vec![(key("dp_rank"), value)]);
        assert_eq!(
            msgpack(&map(vec![
                (key("routing"), rank(Value::from(4))),
                (key("routing"), rank(Value::from(5))),
            ])),
            Some(5)
        );
        assert_eq!(
            msgpack(&map(vec![
                (key("routing"), rank(Value::from(5))),
                (key("routing"), Value::Nil),
            ])),
            None
        );
        assert_eq!(
            msgpack(&map(vec![(
                key("routing"),
                map(vec![
                    (key("dp_rank"), Value::from(5)),
                    (key("dp_rank"), Value::Nil),
                ]),
            )])),
            None
        );
        assert_eq!(
            msgpack(&map(vec![(
                key("routing"),
                map(vec![
                    (key("dp_rank"), Value::Nil),
                    (key("dp_rank"), Value::from(6)),
                ]),
            )])),
            Some(6)
        );
    }

    /// Every msgpack integer encoding a sender may choose, by marker, and the
    /// values that are not ranks.
    #[test]
    fn msgpack_ranks_are_read_from_every_integer_encoding() {
        let mut prefix = vec![0x81, 0xa7];
        prefix.extend_from_slice(b"routing");
        prefix.extend_from_slice(&[0x81, 0xa7]);
        prefix.extend_from_slice(b"dp_rank");
        let cases: &[(&[u8], Option<u32>)] = &[
            (&[0x05], Some(5)),
            (&[0xcc, 0x05], Some(5)),
            (&[0xcd, 0x00, 0x05], Some(5)),
            (&[0xce, 0xff, 0xff, 0xff, 0xff], Some(u32::MAX)),
            (&[0xcf, 0, 0, 0, 0, 0, 0, 0, 0x05], Some(5)),
            (&[0xcf, 0, 0, 0, 0x01, 0, 0, 0, 0], None),
            (&[0xd0, 0x05], Some(5)),
            (&[0xd1, 0x00, 0x05], Some(5)),
            (&[0xd2, 0, 0, 0, 0x05], Some(5)),
            (&[0xd3, 0, 0, 0, 0, 0, 0, 0, 0x07], Some(7)),
            (&[0xd0, 0xff], None),
            (&[0xff], None),
            (&[0xca, 0x40, 0xa0, 0x00, 0x00], None),
            (&[0xc0], None),
            (&[0xc3], None),
            (&[0xa1, b'5'], None),
            (&[0xc4, 0x01, 0x05], None),
            (&[0x91, 0x05], None),
        ];
        for (encoded, expected) in cases {
            let bytes = [prefix.as_slice(), encoded].concat();
            assert_eq!(
                decode(RequestPlanePayloadCodec::Msgpack, &bytes),
                *expected,
                "{encoded:02x?}"
            );
        }
        // A `str8`-encoded key is the same string as a fixstr one.
        let mut bytes = vec![0x81, 0xd9, 0x07];
        bytes.extend_from_slice(b"routing");
        bytes.extend_from_slice(&[0x81, 0xa7]);
        bytes.extend_from_slice(b"dp_rank");
        bytes.push(0x06);
        assert_eq!(decode(RequestPlanePayloadCodec::Msgpack, &bytes), Some(6));
    }

    /// A body the codec rejects is rejected with or without the probe.
    #[test]
    fn malformed_payloads_fail_as_before() {
        for (codec, bytes) in [
            (
                RequestPlanePayloadCodec::Json,
                &br#"{"routing":{"dp_rank":5"#[..],
            ),
            (RequestPlanePayloadCodec::Msgpack, &[0x81, 0xa7, b'r'][..]),
        ] {
            assert!(codec.decode::<Probed>(bytes).is_err());
            assert!(codec.decode::<Plain>(bytes).is_err());
        }
    }
}
