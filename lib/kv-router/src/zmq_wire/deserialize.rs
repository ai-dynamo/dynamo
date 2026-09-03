// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::fmt;

use serde::Deserialize;
use serde::Deserializer;
use serde::de::{self, IgnoredAny, MapAccess, SeqAccess, Visitor};

use crate::protocols::BlockExtraInfo;

use super::extra_keys::{extra_keys_to_block_mm_infos, extra_keys_to_cache_namespace};
use super::filter::{
    BlockStoredTrailingField, KvCacheEventMetadata, KvCacheEventTrailingField, KvCacheSpecKind,
};
use super::types::{BlockHashValue, ExtraKeyItem, KvTokenIds, Locality, RawKvEvent};

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
        let mut cache_namespace: Option<Option<String>> = None;
        let mut extra_keys: Option<Option<Vec<Option<Vec<ExtraKeyItem>>>>> = None;
        let mut block_mm_infos: Option<Option<Vec<Option<BlockExtraInfo>>>> = None;
        let mut locality: Option<Option<Locality>> = None;
        let mut ownership: Option<Option<String>> = None;
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
                "cache_salt" => {
                    cache_namespace = Some(map.next_value()?);
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
                "locality" => {
                    locality = Some(map.next_value()?);
                }
                "ownership" => {
                    ownership = Some(map.next_value()?);
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
                let medium = normalize_medium(medium.unwrap_or(None));
                let lora_name = lora_name.unwrap_or(None);
                let extra_keys = extra_keys.unwrap_or(None);
                let cache_namespace = cache_namespace.unwrap_or(None).or_else(|| {
                    extra_keys_to_cache_namespace(extra_keys.as_deref(), lora_name.as_deref())
                });
                let block_mm_infos = block_mm_infos
                    .unwrap_or(None)
                    .or_else(|| extra_keys_to_block_mm_infos(extra_keys));
                Ok(RawKvEvent::BlockStored {
                    block_hashes,
                    parent_block_hash: parent_block_hash.unwrap_or(None),
                    token_ids: raw_token_ids,
                    block_size,
                    medium,
                    lora_name,
                    cache_namespace,
                    block_mm_infos,
                    is_eagle: Some(is_eagle),
                    group_idx: metadata.group_idx,
                    kv_cache_spec_kind: metadata.kv_cache_spec_kind,
                    kv_cache_spec_sliding_window: metadata.kv_cache_spec_sliding_window,
                    locality: locality.unwrap_or(None),
                    ownership: ownership.unwrap_or(None),
                })
            }
            Some("BlockRemoved") => {
                let block_hashes =
                    block_hashes.ok_or_else(|| de::Error::missing_field("block_hashes"))?;
                let medium = normalize_medium(medium.unwrap_or(None));
                Ok(RawKvEvent::BlockRemoved {
                    block_hashes,
                    medium,
                    group_idx: metadata.group_idx,
                    kv_cache_spec_kind: metadata.kv_cache_spec_kind,
                    kv_cache_spec_sliding_window: metadata.kv_cache_spec_sliding_window,
                    locality: locality.unwrap_or(None),
                    ownership: ownership.unwrap_or(None),
                })
            }
            Some("AllBlocksCleared") => Ok(RawKvEvent::AllBlocksCleared {
                ownership: ownership.unwrap_or(None),
            }),
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
                let medium: Option<String> = normalize_medium(seq.next_element()?.unwrap_or(None));
                let lora_name: Option<String> = seq.next_element()?.unwrap_or(None);
                let extra_keys: Option<Vec<Option<Vec<ExtraKeyItem>>>> =
                    seq.next_element()?.unwrap_or(None);
                let mut trailing = std::array::from_fn(|_| None);
                let mut trailing_len = 0;
                while trailing_len < trailing.len() {
                    let Some(field) = seq.next_element::<Option<BlockStoredTrailingField>>()?
                    else {
                        break;
                    };
                    trailing[trailing_len] = field;
                    trailing_len += 1;
                }
                while seq.next_element::<IgnoredAny>()?.is_some() {}

                let parsed = parse_block_stored_trailing::<A::Error>(trailing, trailing_len)?;

                let cache_namespace =
                    extra_keys_to_cache_namespace(extra_keys.as_deref(), lora_name.as_deref());
                let block_mm_infos = parsed
                    .block_mm_infos
                    .or_else(|| extra_keys_to_block_mm_infos(extra_keys));
                let (raw_token_ids, is_eagle) = normalize_token_ids(token_ids);

                Ok(RawKvEvent::BlockStored {
                    block_hashes,
                    parent_block_hash,
                    token_ids: raw_token_ids,
                    block_size,
                    medium,
                    lora_name,
                    cache_namespace,
                    block_mm_infos,
                    is_eagle: Some(is_eagle),
                    group_idx: parsed.metadata.group_idx,
                    kv_cache_spec_kind: parsed.metadata.kv_cache_spec_kind,
                    kv_cache_spec_sliding_window: parsed.metadata.kv_cache_spec_sliding_window,
                    locality: parsed.locality,
                    ownership: parsed.ownership,
                })
            }
            "BlockRemoved" => {
                let block_hashes: Vec<BlockHashValue> = seq
                    .next_element()?
                    .ok_or_else(|| de::Error::invalid_length(1, &"missing block_hashes"))?;
                let medium: Option<String> = normalize_medium(seq.next_element()?.unwrap_or(None));
                let mut trailing = std::array::from_fn(|_| None);
                let mut trailing_len = 0;
                while trailing_len < trailing.len() {
                    let Some(field) = seq.next_element::<Option<KvCacheEventTrailingField>>()?
                    else {
                        break;
                    };
                    trailing[trailing_len] = field;
                    trailing_len += 1;
                }
                while seq.next_element::<IgnoredAny>()?.is_some() {}
                let parsed = parse_common_trailing::<A::Error>(trailing, trailing_len)?;

                Ok(RawKvEvent::BlockRemoved {
                    block_hashes,
                    medium,
                    group_idx: parsed.metadata.group_idx,
                    kv_cache_spec_kind: parsed.metadata.kv_cache_spec_kind,
                    kv_cache_spec_sliding_window: parsed.metadata.kv_cache_spec_sliding_window,
                    locality: parsed.locality,
                    ownership: parsed.ownership,
                })
            }
            "AllBlocksCleared" => {
                let ownership: Option<String> = seq.next_element()?.unwrap_or(None);
                while seq.next_element::<IgnoredAny>()?.is_some() {}
                Ok(RawKvEvent::AllBlocksCleared { ownership })
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

struct ParsedCommonTrailing {
    metadata: KvCacheEventMetadata,
    locality: Option<Locality>,
    ownership: Option<String>,
}

struct ParsedBlockStoredTrailing {
    metadata: KvCacheEventMetadata,
    block_mm_infos: Option<Vec<Option<BlockExtraInfo>>>,
    locality: Option<Locality>,
    ownership: Option<String>,
}

fn parse_block_stored_trailing<E>(
    trailing: [Option<BlockStoredTrailingField>; 5],
    trailing_len: usize,
) -> Result<ParsedBlockStoredTrailing, E>
where
    E: de::Error,
{
    let trailing_fields = &trailing[..trailing_len];
    let has_legacy_mm_infos = trailing_fields
        .iter()
        .flatten()
        .any(|field| matches!(field, BlockStoredTrailingField::BlockMmInfos(_)));

    if trailing_len >= 4
        && !has_legacy_mm_infos
        && fixed_trailing_layout_matches(|index| {
            trailing_fields.get(index).and_then(|field| match field {
                Some(BlockStoredTrailingField::Common(field)) => Some(field),
                Some(BlockStoredTrailingField::BlockMmInfos(_)) | None => None,
            })
        })
    {
        let common = trailing.map(|field| match field {
            Some(BlockStoredTrailingField::Common(field)) => Some(field),
            Some(BlockStoredTrailingField::BlockMmInfos(_)) => unreachable!(),
            None => None,
        });
        let parsed = parse_fixed_trailing::<E>(common)?;
        return Ok(ParsedBlockStoredTrailing {
            metadata: parsed.metadata,
            block_mm_infos: None,
            locality: parsed.locality,
            ownership: parsed.ownership,
        });
    }

    // Compatibility with v1.2 vLLM tuples during v1.4 rolling upgrades. These
    // forms appended metadata opportunistically and could include block_mm_infos.
    // TODO(v1.5): Remove when v1.2 falls outside the N-2 compatibility window.
    let mut metadata = KvCacheEventMetadata::default();
    let mut block_mm_infos = None;
    for field in trailing.into_iter().take(trailing_len).flatten() {
        match field {
            BlockStoredTrailingField::Common(field) => metadata.record_legacy_trailing(field),
            BlockStoredTrailingField::BlockMmInfos(infos) => block_mm_infos = Some(infos),
        }
    }
    Ok(ParsedBlockStoredTrailing {
        metadata,
        block_mm_infos,
        locality: None,
        ownership: None,
    })
}

fn parse_common_trailing<E>(
    trailing: [Option<KvCacheEventTrailingField>; 5],
    trailing_len: usize,
) -> Result<ParsedCommonTrailing, E>
where
    E: de::Error,
{
    if trailing_len >= 4
        && fixed_trailing_layout_matches(|index| {
            trailing[..trailing_len].get(index).and_then(Option::as_ref)
        })
    {
        return parse_fixed_trailing::<E>(trailing);
    }

    // Compatibility with v1.2 vLLM tuples during v1.4 rolling upgrades. These
    // forms omitted unused slots and therefore had no fixed trailing layout.
    // TODO(v1.5): Remove when v1.2 falls outside the N-2 compatibility window.
    let mut metadata = KvCacheEventMetadata::default();
    for field in trailing.into_iter().take(trailing_len).flatten() {
        metadata.record_legacy_trailing(field);
    }
    Ok(ParsedCommonTrailing {
        metadata,
        locality: None,
        ownership: None,
    })
}

fn fixed_trailing_layout_matches<'a>(
    field_at: impl Fn(usize) -> Option<&'a KvCacheEventTrailingField>,
) -> bool {
    let accepts_unsigned = |field: Option<&KvCacheEventTrailingField>| {
        field.is_none_or(|field| matches!(field, KvCacheEventTrailingField::Unsigned(_)))
    };
    let accepts_text = |field: Option<&KvCacheEventTrailingField>| {
        field.is_none_or(|field| matches!(field, KvCacheEventTrailingField::Text(_)))
    };

    accepts_unsigned(field_at(0))
        && accepts_text(field_at(1))
        && accepts_unsigned(field_at(2))
        && accepts_text(field_at(3))
        && accepts_text(field_at(4))
}

fn parse_fixed_trailing<E>(
    trailing: [Option<KvCacheEventTrailingField>; 5],
) -> Result<ParsedCommonTrailing, E>
where
    E: de::Error,
{
    // Current production order is group, cache kind, sliding window, locality,
    // then ownership. New fields must remain append-only.
    let mut fields = trailing.into_iter();
    let group_idx = parse_unsigned::<E>(fields.next().flatten(), "group_idx")?;
    let kv_cache_spec_kind = parse_text::<E>(fields.next().flatten(), "kv_cache_spec_kind")?
        .map(|kind| KvCacheSpecKind::from_wire(&kind));
    let kv_cache_spec_sliding_window =
        parse_unsigned::<E>(fields.next().flatten(), "kv_cache_spec_sliding_window")?;
    let locality = parse_text::<E>(fields.next().flatten(), "locality")?.map(|locality| {
        match locality.as_str() {
            "LOCAL" => Locality::Local,
            "REMOTE" => Locality::Remote,
            _ => Locality::Unknown,
        }
    });
    let ownership = parse_text::<E>(fields.next().flatten(), "ownership")?;

    Ok(ParsedCommonTrailing {
        metadata: KvCacheEventMetadata {
            group_idx,
            kv_cache_spec_kind,
            kv_cache_spec_sliding_window,
        },
        locality,
        ownership,
    })
}

fn parse_unsigned<E>(
    field: Option<KvCacheEventTrailingField>,
    name: &'static str,
) -> Result<Option<u32>, E>
where
    E: de::Error,
{
    match field {
        Some(KvCacheEventTrailingField::Unsigned(value)) => Ok(Some(value)),
        Some(KvCacheEventTrailingField::Text(_) | KvCacheEventTrailingField::Ignored(_)) => {
            tracing::debug!(field = name, "Ignoring incompatible KV event tuple field");
            Ok(None)
        }
        None => Ok(None),
    }
}

fn parse_text<E>(
    field: Option<KvCacheEventTrailingField>,
    name: &'static str,
) -> Result<Option<String>, E>
where
    E: de::Error,
{
    match field {
        Some(KvCacheEventTrailingField::Text(value)) => Ok(Some(value)),
        Some(KvCacheEventTrailingField::Unsigned(_) | KvCacheEventTrailingField::Ignored(_)) => {
            tracing::debug!(field = name, "Ignoring incompatible KV event tuple field");
            Ok(None)
        }
        None => Ok(None),
    }
}

/// vLLM omits `medium` for device (GPU) events; treat an empty string the same
/// as an absent field so an unset medium stays on the default device tier
/// instead of failing closed as an unknown medium. Mirrors the empty-string
/// normalization applied to `cache_salt`.
fn normalize_medium(medium: Option<String>) -> Option<String> {
    medium.filter(|value| !value.is_empty())
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
