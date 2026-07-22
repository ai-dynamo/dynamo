// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use crate::protocols::{BlockExtraInfo, BlockMmObjectInfo};

use super::types::ExtraKeyItem;

/// Parse MM hash from extra_keys string:
/// - Only accept canonical vLLM MM identifiers (64-char hex digest)
/// - Convert by taking the first 16 hex chars as u64
pub fn parse_mm_hash_from_extra_key(s: &str) -> Option<u64> {
    // extra_keys mixes MM identifiers with LoRA/cache_salt/prompt-embed metadata.
    // Only MM identifiers should be mapped into BlockExtraInfo.
    if s.len() == 64 && s.chars().all(|c| c.is_ascii_hexdigit()) {
        return u64::from_str_radix(&s[..16], 16).ok();
    }
    None
}

/// Convert vLLM BlockStored extra_keys to block-level MM infos.
/// extra_keys is a list aligned with blocks:
/// - None => no MM content in that block
/// - ["hash1", "hash2", ...] => one or more MM objects in that block
/// - [[hash, start_offset], ...] => one or more MM objects with block-relative
///   start offsets (vLLM 0.19+)
pub fn extra_keys_to_block_mm_infos(
    extra_keys: Option<Vec<Option<Vec<ExtraKeyItem>>>>,
) -> Option<Vec<Option<BlockExtraInfo>>> {
    let extra_keys = extra_keys?;
    if extra_keys.is_empty() {
        return None;
    }

    let infos: Vec<Option<BlockExtraInfo>> = extra_keys
        .into_iter()
        .map(|block_keys| {
            let mm_objects: Vec<BlockMmObjectInfo> = block_keys
                .unwrap_or_default()
                .iter()
                .filter_map(|key| match key {
                    ExtraKeyItem::Hash(hash)
                    | ExtraKeyItem::HashWithSignedOffset((hash, _))
                    | ExtraKeyItem::HashWithUnsignedOffset((hash, _)) => {
                        parse_mm_hash_from_extra_key(hash)
                    }
                    ExtraKeyItem::Bytes(_)
                    | ExtraKeyItem::Signed(_)
                    | ExtraKeyItem::Unsigned(_)
                    | ExtraKeyItem::Float(_)
                    | ExtraKeyItem::Bool(_) => None,
                })
                .map(|mm_hash| BlockMmObjectInfo {
                    mm_hash,
                    // vLLM extra_keys exposes MM start offsets but not MM lengths.
                    // Dynamo's block hash only depends on mm_hash today, so keep
                    // offsets empty rather than inventing a synthetic range.
                    offsets: vec![],
                })
                .collect();

            if mm_objects.is_empty() {
                None
            } else {
                Some(BlockExtraInfo { mm_objects })
            }
        })
        .collect();

    if infos.iter().all(|i| i.is_none()) {
        return None;
    }

    Some(infos)
}
