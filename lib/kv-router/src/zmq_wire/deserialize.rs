// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::fmt;

use serde::Deserialize;
use serde::Deserializer;
use serde::de::{self, IgnoredAny, MapAccess, SeqAccess, Visitor};

use crate::protocols::BlockExtraInfo;

use super::extra_keys::extra_keys_to_block_mm_infos;
use super::filter::{BlockStoredTrailingField, KvCacheEventMetadata, KvCacheEventTrailingField};
use super::types::{BlockHashValue, ExtraKeyItem, KvTokenIds, RawKvEvent};

/// Our producers use msgspec with `tag=True` and `array_like=True`, which
/// encodes each event as either a tagged map or a tagged tuple. To be tolerant of
/// additional fields that may be appended in the future, we implement a custom
/// deserializer that ignores unknown keys and any extra positional elements.
///
/// This keeps us compatible with older payloads while safely
/// accepting newer ones that include extra metadata.
impl<'de> Deserialize<'de> for RawKvEvent {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        deserializer.deserialize_any(RawKvEventVisitor)
    }
}

struct RawKvEventVisitor;

impl<'de> Visitor<'de> for RawKvEventVisitor {
    type Value = RawKvEvent;

    fn expecting(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        formatter.write_str("a kv event encoded as a tagged map or sequence")
    }

    fn visit_map<A>(self, mut map: A) -> Result<Self::Value, A::Error>
    where
        A: MapAccess<'de>,
    {
        let mut event_type: Option<String> = None;
        let mut block_hashes: Option<Vec<BlockHashValue>> = None;
        let mut parent_block_hash: Option<Option<BlockHashValue>> = None;
        let mut token_ids: Option<KvTokenIds> = None;
        let mut block_size: Option<usize> = None;
        let mut medium: Option<Option<String>> = None;
        let mut lora_name: Option<Option<String>> = None;
        let mut extra_keys: Option<Option<Vec<Option<Vec<ExtraKeyItem>>>>> = None;
        let mut block_mm_infos: Option<Option<Vec<Option<BlockExtraInfo>>>> = None;
        let mut metadata = KvCacheEventMetadata::default();

        while let Some(key) = map.next_key::<String>()? {
            match key.as_str() {
                "type" => {
                    event_type = Some(map.next_value()?);
                }
                "block_hashes" => {
                    block_hashes = Some(map.next_value()?);
                }
                "parent_block_hash" => {
                    parent_block_hash = Some(map.next_value()?);
                }
                "token_ids" => {
                    token_ids = Some(map.next_value()?);
                }
                "block_size" => {
                    block_size = Some(map.next_value()?);
                }
                "medium" => {
                    medium = Some(map.next_value()?);
                }
                "lora_name" => {
                    lora_name = Some(map.next_value()?);
                }
                "extra_keys" => {
                    extra_keys = Some(map.next_value()?);
                }
                "block_mm_infos" => {
                    block_mm_infos = Some(map.next_value()?);
                }
                "group_idx" => {
                    metadata.group_idx = map.next_value()?;
                }
                "kv_cache_spec_kind" => {
                    metadata.kv_cache_spec_kind = map.next_value()?;
                }
                "kv_cache_spec_sliding_window" => {
                    metadata.kv_cache_spec_sliding_window = map.next_value()?;
                }
                _ => {
                    map.next_value::<IgnoredAny>()?;
                }
            }
        }

        match event_type.as_deref() {
            Some("BlockStored") => {
                let block_hashes =
                    block_hashes.ok_or_else(|| de::Error::missing_field("block_hashes"))?;
                let token_ids = token_ids.ok_or_else(|| de::Error::missing_field("token_ids"))?;
                let (raw_token_ids, is_eagle) = normalize_token_ids(token_ids);
                let block_size =
                    block_size.ok_or_else(|| de::Error::missing_field("block_size"))?;
                let medium = medium.unwrap_or(None);
                let block_mm_infos = block_mm_infos
                    .unwrap_or(None)
                    .or_else(|| extra_keys_to_block_mm_infos(extra_keys.unwrap_or(None)));
                Ok(RawKvEvent::BlockStored {
                    block_hashes,
                    parent_block_hash: parent_block_hash.unwrap_or(None),
                    token_ids: raw_token_ids,
                    block_size,
                    medium,
                    lora_name: lora_name.unwrap_or(None),
                    block_mm_infos,
                    is_eagle: Some(is_eagle),
                    group_idx: metadata.group_idx,
                    kv_cache_spec_kind: metadata.kv_cache_spec_kind,
                    kv_cache_spec_sliding_window: metadata.kv_cache_spec_sliding_window,
                })
            }
            Some("BlockRemoved") => {
                let block_hashes =
                    block_hashes.ok_or_else(|| de::Error::missing_field("block_hashes"))?;
                let medium = medium.unwrap_or(None);
                Ok(RawKvEvent::BlockRemoved {
                    block_hashes,
                    medium,
                    group_idx: metadata.group_idx,
                    kv_cache_spec_kind: metadata.kv_cache_spec_kind,
                    kv_cache_spec_sliding_window: metadata.kv_cache_spec_sliding_window,
                })
            }
            Some("AllBlocksCleared") => Ok(RawKvEvent::AllBlocksCleared),
            Some("Ignored") => Ok(RawKvEvent::Ignored),
            Some(other) => Err(de::Error::unknown_variant(
                other,
                &["BlockStored", "BlockRemoved", "AllBlocksCleared", "Ignored"],
            )),
            None => Err(de::Error::missing_field("type")),
        }
    }

    fn visit_seq<A>(self, mut seq: A) -> Result<Self::Value, A::Error>
    where
        A: SeqAccess<'de>,
    {
        let tag: Option<String> = seq.next_element()?;
        let Some(tag) = tag else {
            return Err(de::Error::invalid_length(
                0,
                &"sequence must start with event tag",
            ));
        };

        match tag.as_str() {
            "BlockStored" => {
                let block_hashes: Vec<BlockHashValue> = seq
                    .next_element()?
                    .ok_or_else(|| de::Error::invalid_length(1, &"missing block_hashes"))?;
                let parent_block_hash: Option<BlockHashValue> = seq.next_element()?.unwrap_or(None);
                let token_ids: KvTokenIds = seq
                    .next_element()?
                    .ok_or_else(|| de::Error::invalid_length(3, &"missing token_ids"))?;
                let block_size: usize = seq
                    .next_element()?
                    .ok_or_else(|| de::Error::invalid_length(4, &"missing block_size"))?;
                // Position 5 was lora_id in older formats; consume and discard for compat.
                let _lora_id: Option<u64> = seq.next_element()?.unwrap_or(None);
                let medium: Option<String> = seq.next_element()?.unwrap_or(None);
                let lora_name: Option<String> = seq.next_element()?.unwrap_or(None);
                let extra_keys: Option<Vec<Option<Vec<ExtraKeyItem>>>> =
                    seq.next_element()?.unwrap_or(None);
                let mut block_mm_infos: Option<Vec<Option<BlockExtraInfo>>> = None;
                let mut metadata = KvCacheEventMetadata::default();

                for _ in 0..4 {
                    let trailing: Option<BlockStoredTrailingField> =
                        seq.next_element()?.unwrap_or(None);
                    match trailing {
                        Some(BlockStoredTrailingField::Common(trailing)) => {
                            metadata.record_trailing(trailing);
                        }
                        Some(BlockStoredTrailingField::BlockMmInfos(infos)) => {
                            block_mm_infos = Some(infos);
                        }
                        None => {}
                    }
                }

                while seq.next_element::<IgnoredAny>()?.is_some() {}

                let block_mm_infos =
                    block_mm_infos.or_else(|| extra_keys_to_block_mm_infos(extra_keys));
                let (raw_token_ids, is_eagle) = normalize_token_ids(token_ids);

                Ok(RawKvEvent::BlockStored {
                    block_hashes,
                    parent_block_hash,
                    token_ids: raw_token_ids,
                    block_size,
                    medium,
                    lora_name,
                    block_mm_infos,
                    is_eagle: Some(is_eagle),
                    group_idx: metadata.group_idx,
                    kv_cache_spec_kind: metadata.kv_cache_spec_kind,
                    kv_cache_spec_sliding_window: metadata.kv_cache_spec_sliding_window,
                })
            }
            "BlockRemoved" => {
                let block_hashes: Vec<BlockHashValue> = seq
                    .next_element()?
                    .ok_or_else(|| de::Error::invalid_length(1, &"missing block_hashes"))?;
                let medium: Option<String> = seq.next_element()?.unwrap_or(None);
                let mut metadata = KvCacheEventMetadata::default();

                for _ in 0..3 {
                    let trailing: Option<KvCacheEventTrailingField> =
                        seq.next_element()?.unwrap_or(None);
                    if let Some(trailing) = trailing {
                        metadata.record_trailing(trailing);
                    }
                }

                while seq.next_element::<IgnoredAny>()?.is_some() {}

                Ok(RawKvEvent::BlockRemoved {
                    block_hashes,
                    medium,
                    group_idx: metadata.group_idx,
                    kv_cache_spec_kind: metadata.kv_cache_spec_kind,
                    kv_cache_spec_sliding_window: metadata.kv_cache_spec_sliding_window,
                })
            }
            "AllBlocksCleared" => {
                while seq.next_element::<IgnoredAny>()?.is_some() {}
                Ok(RawKvEvent::AllBlocksCleared)
            }
            "Ignored" => {
                while seq.next_element::<IgnoredAny>()?.is_some() {}
                Ok(RawKvEvent::Ignored)
            }
            other => Err(de::Error::unknown_variant(
                other,
                &["BlockStored", "BlockRemoved", "AllBlocksCleared", "Ignored"],
            )),
        }
    }
}

fn normalize_token_ids(token_ids: KvTokenIds) -> (Vec<u32>, bool) {
    match token_ids {
        KvTokenIds::Single(tids) => (tids, false),
        KvTokenIds::Bigram(tids) => {
            let mut new_tids: Vec<u32> = tids.iter().map(|&(first, _)| first).collect();
            if !tids.is_empty() {
                let last_token = tids.last().map(|&(_, second)| second).unwrap();
                new_tids.push(last_token);
            }
            (new_tids, true)
        }
    }
}
