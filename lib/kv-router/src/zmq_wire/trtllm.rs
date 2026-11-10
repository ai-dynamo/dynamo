// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Canonicalize TRT-LLM V2's digest-first image cache-key tokens.

use std::sync::atomic::{AtomicU64, Ordering};

use serde::Deserialize;

use super::types::KvTokenIds;

/// UUID namespace reserved for adapter-generated TRT-LLM image routing hints.
pub const IMAGE_ROUTING_UUID_PREFIX: &str = "dynamo-image-v1:";

/// Process-wide, because the serde visitor that drops these events has no state.
static STORES_WITHOUT_IMAGE_ROUTING: AtomicU64 = AtomicU64::new(0);

/// Records a digest-first `BlockStored` that carries no `trtllm_image_routing`
/// metadata. The caller still drops the event: without the metadata its image
/// tokens cannot be mapped to the hashes the frontend computes. Warns once per
/// process, so that image inference on a runtime without image routing is not
/// mistaken for image-aware routing, then logs at debug.
pub(super) fn record_store_without_image_routing(block_hashes: usize) {
    let dropped = STORES_WITHOUT_IMAGE_ROUTING.fetch_add(1, Ordering::Relaxed) + 1;
    if dropped == 1 {
        tracing::warn!(
            dropped,
            block_hashes,
            "Dropped a TensorRT-LLM image BlockStored event: it has digest-first image tokens but no \
             trtllm_image_routing metadata. Image KV blocks will not enter the routing index, so image \
             requests route by text prefix only. Image-aware KV routing needs a TensorRT-LLM runtime \
             with image routing v1 (NVIDIA/TensorRT-LLM#21125)"
        );
    } else {
        tracing::debug!(
            dropped,
            block_hashes,
            "Dropped a TensorRT-LLM image BlockStored event without trtllm_image_routing metadata"
        );
    }
}

#[cfg(test)]
pub(super) fn stores_without_image_routing() -> u64 {
    STORES_WITHOUT_IMAGE_ROUTING.load(Ordering::Relaxed)
}

pub(super) enum MapTokenIds {
    Plain(KvTokenIds),
    DigestFirst(Vec<Token>),
}

impl<'de> Deserialize<'de> for MapTokenIds {
    fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        // Nested untagged derives clone the buffered sequence on every event.
        #[derive(Deserialize)]
        #[serde(untagged)]
        enum Wire {
            Single(Vec<u64>),
            Bigram(Vec<(u64, u64)>),
            DigestFirst(Vec<Token>),
        }

        Ok(match Wire::deserialize(deserializer)? {
            Wire::Single(ids) => Self::Plain(KvTokenIds::Single(ids)),
            Wire::Bigram(ids) => Self::Plain(KvTokenIds::Bigram(ids)),
            Wire::DigestFirst(tokens) => Self::DigestFirst(tokens),
        })
    }
}

#[derive(Deserialize)]
#[serde(untagged)]
pub(super) enum Token {
    Id(u32),
    Digest(String),
}

#[derive(Deserialize)]
pub(super) struct MediaKey {
    hash: String,
    #[serde(default)]
    uuid: Option<String>,
}

#[derive(Deserialize)]
#[serde(untagged)]
pub(super) enum RoutingMetadata {
    Fields(RoutingFields),
    Ignored(serde::de::IgnoredAny),
}

#[derive(Deserialize)]
pub(super) struct RoutingFields {
    pub version: u32,
    pub token_id_offset: u32,
}

#[derive(Deserialize)]
#[serde(untagged)]
pub(super) enum RoutingKeys {
    Fields(Vec<Vec<MediaKey>>),
    Ignored(serde::de::IgnoredAny),
}

pub(super) fn token_ids(
    ids: MapTokenIds,
    metadata: RoutingMetadata,
    keys: Option<RoutingKeys>,
    block_size: usize,
    blocks: usize,
) -> Option<Vec<u32>> {
    let RoutingMetadata::Fields(metadata) = metadata else {
        return None;
    };
    if metadata.version != 1 {
        return None;
    }
    let tokens = match ids {
        MapTokenIds::Plain(KvTokenIds::Single(ids)) => ids
            .into_iter()
            .map(|id| u32::try_from(id).ok().map(Token::Id))
            .collect::<Option<Vec<_>>>()?,
        MapTokenIds::DigestFirst(tokens) => tokens,
        MapTokenIds::Plain(KvTokenIds::Bigram(_)) => return None,
    };
    let RoutingKeys::Fields(keys) = keys? else {
        return None;
    };
    normalize(tokens, metadata.token_id_offset, keys, block_size, blocks)
}

/// Missing/evicted identities make only this stored event ineligible for indexing.
fn normalize(
    tokens: Vec<Token>,
    offset: u32,
    keys: Vec<Vec<MediaKey>>,
    block_size: usize,
    blocks: usize,
) -> Option<Vec<u32>> {
    if block_size == 0 || keys.len() != blocks || block_size.checked_mul(blocks)? != tokens.len() {
        return None;
    }
    let mut result = Vec::with_capacity(tokens.len());
    let mut tokens = tokens.into_iter();
    for block_keys in keys {
        let mut keys = block_keys.into_iter();
        let mut current_pad = None;
        for token in tokens.by_ref().take(block_size) {
            match token {
                Token::Digest(digest) => {
                    let key = keys.next()?;
                    if digest != key.hash {
                        return None;
                    }
                    let pad = key_pad(&key)?;
                    current_pad = Some(pad);
                    result.push(pad);
                }
                Token::Id(id) if id > offset => {
                    let pad = match current_pad {
                        Some(pad) => pad,
                        None => key_pad(&keys.next()?)?,
                    };
                    current_pad = Some(pad);
                    result.push(pad);
                }
                Token::Id(id) => {
                    current_pad = None;
                    result.push(id);
                }
            }
        }
        if keys.next().is_some() {
            return None;
        }
    }
    Some(result)
}

fn key_pad(key: &MediaKey) -> Option<u32> {
    if key.hash.len() != 64 || !key.hash.bytes().all(|b| b.is_ascii_hexdigit()) {
        return None;
    }
    let hash = key
        .uuid
        .as_deref()?
        .strip_prefix(IMAGE_ROUTING_UUID_PREFIX)?;
    if hash.len() != 16 || !hash.bytes().all(|byte| byte.is_ascii_hexdigit()) {
        return None;
    }
    let hash = u64::from_str_radix(hash, 16).ok()?;
    Some(crate::protocols::pad_value_for_mm_hash(hash))
}
