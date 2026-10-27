// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Canonicalize TRT-LLM V2's digest-first image cache-key tokens.

use serde::Deserialize;

#[derive(Deserialize)]
#[serde(untagged)]
pub(super) enum MapTokenIds {
    Single(Vec<u32>),
    Bigram(Vec<(u32, u32)>),
    DigestFirst(Vec<Token>),
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
    pub keys: Vec<Vec<MediaKey>>,
}

/// Missing/evicted identities make only this stored event ineligible for indexing.
pub(super) fn normalize(
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
                    if digest != key.hash
                        || digest.len() != 64
                        || !digest.bytes().all(|b| b.is_ascii_hexdigit())
                    {
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
    let hash = key.uuid.as_deref()?.strip_prefix("dynamo-image-v1:")?;
    if hash.len() != 16 {
        return None;
    }
    let hash = u64::from_str_radix(hash, 16).ok()?;
    Some(crate::protocols::pad_value_for_mm_hash(hash))
}
