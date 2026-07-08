// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use crate::protocols::{BlockExtraInfo, BlockMmObjectInfo};

use super::types::ExtraKeyItem;

// Must match _DYNAMO_CACHE_SALT_PREFIX in components/src/dynamo/vllm/handlers.py.
const DYNAMO_CACHE_SALT_PREFIX: &str = "dynamo-cache-salt:";

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

/// Extract a vLLM cache salt from `extra_keys` when a producer does not emit
/// top-level `cache_salt`. vLLM aligns `extra_keys` with blocks and includes
/// cache salt only in the first block. Dynamo tags the opaque value before
/// passing it to vLLM because bare strings are otherwise ambiguous with LoRA
/// names and legacy multimodal hashes. The first matching LoRA item is skipped
/// before looking for the tag so a salt equal to the LoRA name still works.
pub fn extra_keys_to_cache_namespace(
    extra_keys: Option<&[Option<Vec<ExtraKeyItem>>]>,
    lora_name: Option<&str>,
) -> Option<String> {
    let first_block = extra_keys?.first()?.as_ref()?;
    let mut unmatched_lora = lora_name.filter(|name| !name.is_empty());
    first_block.iter().find_map(|key| {
        let ExtraKeyItem::Hash(value) = key else {
            return None;
        };
        if unmatched_lora.is_some_and(|name| name == value) {
            unmatched_lora = None;
            return None;
        }
        value
            .strip_prefix(DYNAMO_CACHE_SALT_PREFIX)
            .filter(|namespace| !namespace.is_empty())
            .map(str::to_owned)
    })
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn prompt_embedding_bytes_are_not_cache_namespace() {
        let extra_keys = [Some(vec![ExtraKeyItem::Bytes(b"prompt-embed".to_vec())])];
        assert_eq!(extra_keys_to_cache_namespace(Some(&extra_keys), None), None);
    }

    #[test]
    fn untagged_strings_are_not_cache_namespaces() {
        let extra_keys = [Some(vec![ExtraKeyItem::Hash("tenant-a".to_string())])];
        assert_eq!(extra_keys_to_cache_namespace(Some(&extra_keys), None), None);
    }

    #[test]
    fn tagged_cache_namespace_is_decoded() {
        let extra_keys = [Some(vec![ExtraKeyItem::Hash(
            "dynamo-cache-salt:tenant-a".to_string(),
        )])];
        assert_eq!(
            extra_keys_to_cache_namespace(Some(&extra_keys), None).as_deref(),
            Some("tenant-a")
        );
    }

    #[test]
    fn cache_namespace_equal_to_lora_name_is_decoded() {
        let extra_keys = [Some(vec![
            ExtraKeyItem::Hash("adapter-a".to_string()),
            ExtraKeyItem::Hash("dynamo-cache-salt:adapter-a".to_string()),
        ])];
        assert_eq!(
            extra_keys_to_cache_namespace(Some(&extra_keys), Some("adapter-a")).as_deref(),
            Some("adapter-a")
        );
    }
}
