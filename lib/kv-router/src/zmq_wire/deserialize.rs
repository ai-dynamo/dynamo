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
use super::trtllm::{self, MapTokenIds};
use super::types::{
    BlockHashValue, ExtraKeyItem, KvTokenIds, Locality, RawKvEvent, WIDE_TOKEN_SENTINEL,
};

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

/// SGLang appends typed metadata where vLLM carries its optional LoRA name.
#[derive(Deserialize)]
#[serde(untagged)]
enum BlockStoredNamespace {
    LoraName(String),
    SglangMetadata { cache_salt: String },
}

/// Positional `lookahead_token_ids` slot; any other shape is ignored.
#[derive(Deserialize)]
#[serde(untagged)]
enum LookaheadSlot {
    Tokens(Vec<Vec<u64>>),
    #[allow(dead_code)]
    Other(IgnoredAny),
}

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
        let mut token_ids: Option<MapTokenIds> = None;
        let mut trtllm_routing: Option<trtllm::RoutingMetadata> = None;
        let mut mm_keys: Option<trtllm::RoutingKeys> = None;
        let mut block_size: Option<usize> = None;
        let mut medium: Option<Option<String>> = None;
        let mut lora_name: Option<Option<String>> = None;
        let mut cache_namespace: Option<Option<String>> = None;
        let mut extra_keys: Option<Option<Vec<Option<Vec<ExtraKeyItem>>>>> = None;
        let mut block_mm_infos: Option<Option<Vec<Option<BlockExtraInfo>>>> = None;
        let mut locality: Option<Option<Locality>> = None;
        let mut ownership: Option<Option<String>> = None;
        let mut session_id: Option<Option<String>> = None;
        let mut lookahead_token_ids: Option<Option<Vec<Vec<u64>>>> = None;
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
                "mm_keys" => {
                    mm_keys = map.next_value()?;
                }
                "trtllm_image_routing" => {
                    trtllm_routing = map.next_value()?;
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
                "session_id" => {
                    session_id = Some(map.next_value()?);
                }
                "lookahead_token_ids" => {
                    lookahead_token_ids = Some(map.next_value()?);
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
                let block_size =
                    block_size.ok_or_else(|| de::Error::missing_field("block_size"))?;
                let (raw_token_ids, lookahead_tokens) = match (token_ids, trtllm_routing) {
                    (MapTokenIds::Plain(ids), None) => normalize_token_ids(ids),
                    (ids, Some(metadata)) => {
                        let Some(tokens) = trtllm::token_ids(
                            ids,
                            metadata,
                            mm_keys,
                            block_size,
                            block_hashes.len(),
                        ) else {
                            return Ok(RawKvEvent::Ignored);
                        };
                        (tokens, 0)
                    }
                    _ => return Ok(RawKvEvent::Ignored),
                };
                let (raw_token_ids, lookahead_tokens) = apply_lookahead_token_ids(
                    raw_token_ids,
                    lookahead_tokens,
                    block_size,
                    lookahead_token_ids.unwrap_or(None),
                );
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
                    lookahead_tokens: Some(lookahead_tokens),
                    group_idx: metadata.group_idx,
                    kv_cache_spec_kind: metadata.kv_cache_spec_kind,
                    kv_cache_spec_sliding_window: metadata.kv_cache_spec_sliding_window,
                    locality: locality.unwrap_or(None),
                    ownership: ownership.unwrap_or(None),
                    session_id: session_id.unwrap_or(None),
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
                let namespace: Option<BlockStoredNamespace> = seq.next_element()?.unwrap_or(None);
                let (lora_name, cache_namespace) = match namespace {
                    Some(BlockStoredNamespace::LoraName(name)) => (Some(name), None),
                    Some(BlockStoredNamespace::SglangMetadata { cache_salt }) => {
                        (None, (!cache_salt.is_empty()).then_some(cache_salt))
                    }
                    None => (None, None),
                };
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
                // vLLM's fixed layout continues with `session_id` (index 14) and
                // `lookahead_token_ids` (index 15) after the five trailing fields.
                let mut lookahead_token_ids = None;
                if trailing_len == trailing.len() {
                    let _session_id = seq.next_element::<IgnoredAny>()?;
                    if let Some(Some(LookaheadSlot::Tokens(tokens))) =
                        seq.next_element::<Option<LookaheadSlot>>()?
                    {
                        lookahead_token_ids = Some(tokens);
                    }
                }
                while seq.next_element::<IgnoredAny>()?.is_some() {}

                let parsed = parse_block_stored_trailing::<A::Error>(trailing, trailing_len)?;

                let cache_namespace = cache_namespace.or_else(|| {
                    extra_keys_to_cache_namespace(extra_keys.as_deref(), lora_name.as_deref())
                });
                let block_mm_infos = parsed
                    .block_mm_infos
                    .or_else(|| extra_keys_to_block_mm_infos(extra_keys));
                let (raw_token_ids, lookahead_tokens) = normalize_token_ids(token_ids);
                let (raw_token_ids, lookahead_tokens) = apply_lookahead_token_ids(
                    raw_token_ids,
                    lookahead_tokens,
                    block_size,
                    lookahead_token_ids,
                );

                Ok(RawKvEvent::BlockStored {
                    block_hashes,
                    parent_block_hash,
                    token_ids: raw_token_ids,
                    block_size,
                    medium,
                    lora_name,
                    cache_namespace,
                    block_mm_infos,
                    lookahead_tokens: Some(lookahead_tokens),
                    group_idx: parsed.metadata.group_idx,
                    kv_cache_spec_kind: parsed.metadata.kv_cache_spec_kind,
                    kv_cache_spec_sliding_window: parsed.metadata.kv_cache_spec_sliding_window,
                    locality: parsed.locality,
                    ownership: parsed.ownership,
                    // vLLM emits session_id in its named-map format. Legacy
                    // positional layouts have no unambiguous session slot.
                    session_id: None,
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

fn narrow_token_id(token_id: u64) -> u32 {
    u32::try_from(token_id).unwrap_or(WIDE_TOKEN_SENTINEL)
}

/// Fold vLLM lookahead block hashes into the router's lookahead representation.
///
/// vLLM hashes each block over its tokens plus the next `L` tokens and reports
/// those tokens per block in `lookahead_token_ids`. The router hashes the same
/// windows (`block_size + L` tokens, stride `block_size`) from one flat token
/// stream, so the last block's lookahead is appended to the event's tokens.
/// Layouts that do not describe one consistent stream fall back to plain
/// hashing.
fn apply_lookahead_token_ids(
    mut token_ids: Vec<u32>,
    lookahead_tokens: u32,
    block_size: usize,
    lookahead: Option<Vec<Vec<u64>>>,
) -> (Vec<u32>, u32) {
    let Some(lookahead) = lookahead else {
        return (token_ids, lookahead_tokens);
    };
    if lookahead_tokens > 0 || lookahead.is_empty() {
        return (token_ids, lookahead_tokens);
    }

    let width = lookahead[0].len();
    let uniform = width > 0 && lookahead.iter().all(|tokens| tokens.len() == width);
    let valid = uniform
        && block_size > 0
        && token_ids.len() == lookahead.len() * block_size
        && u32::try_from(width).is_ok();
    if valid {
        let tail = lookahead[lookahead.len() - 1]
            .iter()
            .map(|&t| narrow_token_id(t));
        token_ids.extend(tail);
        // Every block's lookahead must be exactly the tokens that follow it.
        let consistent = lookahead.iter().enumerate().all(|(block_idx, tokens)| {
            let start = (block_idx + 1) * block_size;
            token_ids.get(start..start + width).is_some_and(|next| {
                next.iter()
                    .zip(tokens)
                    .all(|(&a, &b)| a == narrow_token_id(b))
            })
        });
        if consistent {
            return (token_ids, width as u32);
        }
        token_ids.truncate(lookahead.len() * block_size);
    }

    tracing::warn!(
        blocks = lookahead.len(),
        "Inconsistent lookahead_token_ids layout; hashing without lookahead"
    );
    (token_ids, 0)
}

fn normalize_token_ids(token_ids: KvTokenIds) -> (Vec<u32>, u32) {
    match token_ids {
        KvTokenIds::Single(tids) => (tids.into_iter().map(narrow_token_id).collect(), 0),
        KvTokenIds::Bigram(tids) => {
            let mut new_tids: Vec<u32> = tids
                .iter()
                .map(|&(first, _)| narrow_token_id(first))
                .collect();
            if let Some(&(_, last_token)) = tids.last() {
                new_tids.push(narrow_token_id(last_token));
            }
            (new_tids, 1)
        }
    }
}
