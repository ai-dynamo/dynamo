// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Decode-side view of a streamed `GenerateResponse`.
//!
//! SGLang sends its whole `meta_info` dict on every streamed chunk, and the
//! generated `pb::GenerateResponse` decodes it into a `HashMap<String, String>`
//! that the sidecar builds and drops per token while reading a few keys.
//! [`GenerateChunk`] decodes the same wire format but keeps only [`META_KEYS`].

use prost::bytes::{Buf, BufMut};
use prost::encoding::{self, DecodeContext, WireType};
use prost::{DecodeError, Message};

/// The `meta_info` keys the sidecar reads; decoding skips every other entry.
const META_KEYS: [&str; 8] = [
    "prompt_tokens",
    "cached_tokens",
    "finish_reason",
    "routed_experts",
    "input_token_logprobs",
    "input_top_logprobs",
    "output_token_logprobs",
    "output_top_logprobs",
];

fn meta_key_index(key: &[u8]) -> Option<usize> {
    META_KEYS.iter().position(|known| known.as_bytes() == key)
}

/// `pb::GenerateResponse` as the sidecar decodes it from the Generate stream.
/// `text` is skipped: the sidecar never sets `return_text`.
#[derive(Clone, Debug, Default, PartialEq)]
pub(crate) struct GenerateChunk {
    pub(crate) output_ids: Vec<i32>,
    pub(crate) meta_info: MetaInfo,
    pub(crate) finished: bool,
}

/// The [`META_KEYS`] entries of a response's `meta_info`.
#[derive(Clone, Debug, Default, PartialEq)]
pub(crate) struct MetaInfo([Option<String>; META_KEYS.len()]);

impl MetaInfo {
    pub(crate) fn get(&self, key: &str) -> Option<&str> {
        let index = meta_key_index(key.as_bytes());
        debug_assert!(index.is_some(), "meta_info key `{key}` is not in META_KEYS");
        self.0[index?].as_deref()
    }

    /// Merges one `map<string, string>` entry, allocating only for [`META_KEYS`].
    fn merge_entry(&mut self, buf: &mut impl Buf, ctx: DecodeContext) -> Result<(), DecodeError> {
        // `None` until the key is read: protobuf allows the value to come first.
        let mut index: Option<Option<usize>> = None;
        let mut value = String::new();
        encoding::merge_loop(&mut (), buf, ctx, |_, buf, ctx| {
            let (tag, wire_type) = encoding::decode_key(buf)?;
            match tag {
                1 => {
                    index = Some(decode_meta_key(wire_type, buf)?);
                    Ok(())
                }
                2 if index == Some(None) => encoding::skip_field(wire_type, tag, buf, ctx),
                2 => encoding::string::merge(wire_type, &mut value, buf, ctx),
                _ => encoding::skip_field(wire_type, tag, buf, ctx),
            }
        })?;
        if let Some(Some(index)) = index {
            self.0[index] = Some(value);
        }
        Ok(())
    }
}

#[cfg(test)]
impl FromIterator<(String, String)> for MetaInfo {
    fn from_iter<I: IntoIterator<Item = (String, String)>>(entries: I) -> Self {
        let mut meta = Self::default();
        for (key, value) in entries {
            if let Some(index) = meta_key_index(key.as_bytes()) {
                meta.0[index] = Some(value);
            }
        }
        meta
    }
}

/// Reads a `meta_info` key without allocating and returns its [`META_KEYS`] index.
fn decode_meta_key(wire_type: WireType, buf: &mut impl Buf) -> Result<Option<usize>, DecodeError> {
    encoding::check_wire_type(WireType::LengthDelimited, wire_type)?;
    let len = encoding::decode_varint(buf)?;
    if len > buf.remaining() as u64 {
        return Err(DecodeError::new("buffer underflow"));
    }
    let len = len as usize;
    // Longer than any of META_KEYS.
    let mut key = [0; 32];
    if len > key.len() {
        buf.advance(len);
        return Ok(None);
    }
    buf.copy_to_slice(&mut key[..len]);
    Ok(meta_key_index(&key[..len]))
}

impl Message for GenerateChunk {
    fn merge_field(
        &mut self,
        tag: u32,
        wire_type: WireType,
        buf: &mut impl Buf,
        ctx: DecodeContext,
    ) -> Result<(), DecodeError> {
        // Field tags of `GenerateResponse` in proto/sglang.proto.
        match tag {
            1 => encoding::int32::merge_repeated(wire_type, &mut self.output_ids, buf, ctx),
            2 => {
                encoding::check_wire_type(WireType::LengthDelimited, wire_type)?;
                self.meta_info.merge_entry(buf, ctx)
            }
            3 => encoding::bool::merge(wire_type, &mut self.finished, buf, ctx),
            _ => encoding::skip_field(wire_type, tag, buf, ctx),
        }
    }

    fn clear(&mut self) {
        *self = Self::default();
    }

    // The sidecar only receives `GenerateResponse`; it never encodes one.
    fn encode_raw(&self, _buf: &mut impl BufMut) {
        unreachable!("GenerateChunk is decode-only")
    }

    fn encoded_len(&self) -> usize {
        unreachable!("GenerateChunk is decode-only")
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use super::*;
    use crate::proto as pb;

    #[test]
    fn decodes_generate_response_keeping_read_meta_keys() {
        let mut meta_info: HashMap<String, String> = META_KEYS
            .iter()
            .map(|key| (key.to_string(), format!("\"{key}\"")))
            .collect();
        // Entries SGLang sends on every chunk that the sidecar does not read.
        for key in ["id", "weight_version", "completion_tokens", &"k".repeat(40)] {
            meta_info.insert(key.to_string(), "1".to_string());
        }
        let response = pb::GenerateResponse {
            output_ids: vec![7, 8],
            meta_info,
            finished: true,
            text: Some("ok".to_string()),
        };

        let chunk = GenerateChunk::decode(response.encode_to_vec().as_slice()).unwrap();

        assert_eq!(chunk.output_ids, [7, 8]);
        assert!(chunk.finished);
        for key in META_KEYS {
            assert_eq!(
                chunk.meta_info.get(key),
                Some(format!("\"{key}\"").as_str())
            );
        }
    }

    #[test]
    fn meta_entry_value_may_precede_its_key() {
        let mut entry = Vec::new();
        encoding::string::encode(2, &"3".to_string(), &mut entry);
        encoding::string::encode(1, &"prompt_tokens".to_string(), &mut entry);
        let mut response = Vec::new();
        encoding::bytes::encode(2, &entry, &mut response);

        let chunk = GenerateChunk::decode(response.as_slice()).unwrap();

        assert_eq!(chunk.meta_info.get("prompt_tokens"), Some("3"));
    }
}
