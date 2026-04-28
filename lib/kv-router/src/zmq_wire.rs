// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Wire-format types for vLLM ZMQ KV event streams.
//!
//! These types mirror the Python `msgspec`-defined structures emitted by vLLM
//! engines over ZMQ PUB sockets. They are independent of the dynamo runtime
//! and can be used by any crate that needs to decode the raw ZMQ payloads.

use std::collections::HashSet;
use std::fmt;
use std::sync::Arc;
use std::sync::atomic::{AtomicU32, Ordering};

use serde::Deserialize;
use serde::Serialize;
use serde::de::{self, Deserializer, IgnoredAny, MapAccess, SeqAccess, Visitor};

use crate::protocols::{
    BlockExtraInfo, BlockHashOptions, BlockMmObjectInfo, ExternalSequenceBlockHash, KvCacheEvent,
    KvCacheEventData, KvCacheRemoveData, KvCacheStoreData, KvCacheStoredBlockData, Placement,
    PlacementEvent, StorageTier, WorkerWithDpRank, compute_block_hash_for_seq,
};

// -------------------------------------------------------------------------
// Types mirroring the Python msgspec-defined structures -------------------
// -------------------------------------------------------------------------

#[derive(Debug, Serialize)]
pub struct KvEventBatch {
    pub ts: f64,
    pub events: Vec<RawKvEvent>,
    #[serde(alias = "dp_rank")]
    pub data_parallel_rank: Option<i32>,
}

impl<'de> Deserialize<'de> for KvEventBatch {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        // Deserialize from array format: [timestamp, [events], data_parallel_rank]
        let arr: (f64, Vec<RawKvEvent>, Option<i32>) = Deserialize::deserialize(deserializer)?;
        Ok(KvEventBatch {
            ts: arr.0,
            events: arr.1,
            data_parallel_rank: arr.2,
        })
    }
}

#[derive(Debug, Serialize, Deserialize, Clone, Copy)]
#[serde(untagged)]
pub enum BlockHashValue {
    Signed(i64),
    Unsigned(u64),
}

impl BlockHashValue {
    pub fn into_u64(self) -> u64 {
        match self {
            BlockHashValue::Signed(v) => v.cast_unsigned(),
            BlockHashValue::Unsigned(v) => v,
        }
    }
}

#[derive(Serialize, Deserialize, Debug, Clone)]
#[serde(untagged)]
pub enum KvTokenIds {
    Single(Vec<u32>),
    Bigram(Vec<(u32, u32)>),
}

#[derive(Debug, Serialize, Clone)]
#[serde(tag = "type")] // msgspec encodes variant tag as a string when `tag=True`
pub enum RawKvEvent {
    BlockStored {
        /// Block hashes may be emitted as either signed or unsigned 64-bit values.
        /// We normalize them to `u64` while deserializing to support both producers.
        block_hashes: Vec<BlockHashValue>,
        parent_block_hash: Option<BlockHashValue>,
        token_ids: Vec<u32>,
        block_size: usize,
        #[serde(skip_serializing_if = "Option::is_none")]
        medium: Option<String>,
        /// LoRA adapter name for adapter-aware block hashing
        #[serde(default, skip_serializing_if = "Option::is_none")]
        lora_name: Option<String>,
        /// Multimodal extra info for each block (length should match block_hashes)
        #[serde(default, skip_serializing_if = "Option::is_none")]
        block_mm_infos: Option<Vec<Option<BlockExtraInfo>>>,
        #[serde(skip_serializing_if = "Option::is_none")]
        is_eagle: Option<bool>,
    },
    BlockRemoved {
        block_hashes: Vec<BlockHashValue>,
        #[serde(skip_serializing_if = "Option::is_none")]
        medium: Option<String>,
    },
    AllBlocksCleared,
    Ignored,
}

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

#[derive(Debug, Deserialize, Clone)]
#[serde(untagged)]
pub enum ExtraKeyItem {
    Hash(String),
    HashWithSignedOffset((String, i64)),
    HashWithUnsignedOffset((String, u64)),
    Bytes(Vec<u8>),
    Signed(i64),
    Unsigned(u64),
    Float(f64),
    Bool(bool),
}

#[derive(Debug, Deserialize)]
#[serde(untagged)]
enum KvCacheEventTrailingField {
    GroupIdx(u32),
    KvCacheSpecKind(KvCacheSpecKind),
}

#[derive(Debug, Deserialize)]
#[serde(untagged)]
enum BlockStoredTrailingField {
    Common(KvCacheEventTrailingField),
    BlockMmInfos(Vec<Option<BlockExtraInfo>>),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum KvCacheSpecKind {
    FullAttention,
    MlaAttention,
    SlidingWindow,
    SlidingWindowMla,
    Mamba,
    ChunkedLocalAttention,
    SinkFullAttention,
    EncoderOnlyAttention,
    CrossAttention,
    Unknown,
}

impl KvCacheSpecKind {
    fn from_wire(value: &str) -> Self {
        match value {
            "full_attention" => Self::FullAttention,
            "mla_attention" => Self::MlaAttention,
            "sliding_window" => Self::SlidingWindow,
            "sliding_window_mla" => Self::SlidingWindowMla,
            "mamba" => Self::Mamba,
            "chunked_local_attention" => Self::ChunkedLocalAttention,
            "sink_full_attention" => Self::SinkFullAttention,
            "encoder_only_attention" => Self::EncoderOnlyAttention,
            "cross_attention" => Self::CrossAttention,
            unknown => {
                tracing::warn!(
                    kv_cache_spec_kind = unknown,
                    "Unknown KV cache spec kind; treating KV event as non-main"
                );
                Self::Unknown
            }
        }
    }

    fn is_main_attention(self) -> bool {
        matches!(
            self,
            Self::FullAttention | Self::MlaAttention | Self::SinkFullAttention
        )
    }
}

impl<'de> Deserialize<'de> for KvCacheSpecKind {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let value = String::deserialize(deserializer)?;
        Ok(Self::from_wire(&value))
    }
}

#[derive(Debug, Default, Clone, Copy)]
struct KvCacheEventFilter {
    group_idx: Option<u32>,
    kv_cache_spec_kind: Option<KvCacheSpecKind>,
}

impl KvCacheEventFilter {
    fn record_trailing(&mut self, trailing: KvCacheEventTrailingField) {
        match trailing {
            KvCacheEventTrailingField::GroupIdx(group_idx) if self.kv_cache_spec_kind.is_none() => {
                self.group_idx = Some(group_idx);
            }
            KvCacheEventTrailingField::GroupIdx(_) => {}
            KvCacheEventTrailingField::KvCacheSpecKind(kind) => {
                self.kv_cache_spec_kind = Some(kind);
            }
        }
    }

    fn should_ignore(self) -> bool {
        if let Some(kind) = self.kv_cache_spec_kind {
            return !kind.is_main_attention();
        }

        matches!(self.group_idx, Some(group_idx) if group_idx != 0)
    }
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

// -------------------------------------------------------------------------
// Custom deserializer for RawKvEvent --------------------------------------
// -------------------------------------------------------------------------

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
        let mut filter = KvCacheEventFilter::default();

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
                    filter.group_idx = map.next_value()?;
                }
                "kv_cache_spec_kind" => {
                    filter.kv_cache_spec_kind = map.next_value()?;
                }
                "kv_cache_spec_sliding_window" => {
                    map.next_value::<Option<u32>>()?;
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
                let (raw_token_ids, is_eagle) = match token_ids {
                    KvTokenIds::Single(tids) => (tids, false),
                    KvTokenIds::Bigram(tids) => {
                        let mut new_tids: Vec<u32> = tids.iter().map(|&(first, _)| first).collect();
                        if !tids.is_empty() {
                            let last_token = tids.last().map(|&(_, second)| second).unwrap();
                            new_tids.push(last_token);
                        }
                        (new_tids, true)
                    }
                };
                let block_size =
                    block_size.ok_or_else(|| de::Error::missing_field("block_size"))?;
                let medium = medium.unwrap_or(None);
                if filter.should_ignore() {
                    return Ok(RawKvEvent::Ignored);
                }
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
                })
            }
            Some("BlockRemoved") => {
                let block_hashes =
                    block_hashes.ok_or_else(|| de::Error::missing_field("block_hashes"))?;
                let medium = medium.unwrap_or(None);
                if filter.should_ignore() {
                    return Ok(RawKvEvent::Ignored);
                }
                Ok(RawKvEvent::BlockRemoved {
                    block_hashes,
                    medium,
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
                // Position 5 was lora_id in older formats; consume and discard for compat
                let _lora_id: Option<u64> = seq.next_element()?.unwrap_or(None);
                let medium: Option<String> = seq.next_element()?.unwrap_or(None);
                let lora_name: Option<String> = seq.next_element()?.unwrap_or(None);
                let extra_keys: Option<Vec<Option<Vec<ExtraKeyItem>>>> =
                    seq.next_element()?.unwrap_or(None);
                let mut block_mm_infos: Option<Vec<Option<BlockExtraInfo>>> = None;
                let mut filter = KvCacheEventFilter::default();

                for _ in 0..3 {
                    let trailing: Option<BlockStoredTrailingField> =
                        seq.next_element()?.unwrap_or(None);
                    match trailing {
                        Some(BlockStoredTrailingField::Common(trailing)) => {
                            filter.record_trailing(trailing);
                        }
                        Some(BlockStoredTrailingField::BlockMmInfos(infos)) => {
                            block_mm_infos = Some(infos);
                        }
                        None => {}
                    }
                }

                while seq.next_element::<IgnoredAny>()?.is_some() {}

                if filter.should_ignore() {
                    return Ok(RawKvEvent::Ignored);
                }

                let block_mm_infos =
                    block_mm_infos.or_else(|| extra_keys_to_block_mm_infos(extra_keys));

                let (raw_token_ids, is_eagle) = match token_ids {
                    KvTokenIds::Single(tids) => (tids, false),
                    KvTokenIds::Bigram(tids) => {
                        let mut new_tids: Vec<u32> = tids.iter().map(|&(first, _)| first).collect();
                        if !tids.is_empty() {
                            let last_token = tids.last().map(|&(_, second)| second).unwrap();
                            new_tids.push(last_token);
                        }
                        (new_tids, true)
                    }
                };

                Ok(RawKvEvent::BlockStored {
                    block_hashes,
                    parent_block_hash,
                    token_ids: raw_token_ids,
                    block_size,
                    medium,
                    lora_name,
                    block_mm_infos,
                    is_eagle: Some(is_eagle),
                })
            }
            "BlockRemoved" => {
                let block_hashes: Vec<BlockHashValue> = seq
                    .next_element()?
                    .ok_or_else(|| de::Error::invalid_length(1, &"missing block_hashes"))?;
                let medium: Option<String> = seq.next_element()?.unwrap_or(None);
                let mut filter = KvCacheEventFilter::default();

                for _ in 0..3 {
                    let trailing: Option<KvCacheEventTrailingField> =
                        seq.next_element()?.unwrap_or(None);
                    if let Some(trailing) = trailing {
                        filter.record_trailing(trailing);
                    }
                }

                while seq.next_element::<IgnoredAny>()?.is_some() {}

                if filter.should_ignore() {
                    return Ok(RawKvEvent::Ignored);
                }

                Ok(RawKvEvent::BlockRemoved {
                    block_hashes,
                    medium,
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

// -------------------------------------------------------------------------
// Event conversion --------------------------------------------------------
// -------------------------------------------------------------------------

/// Convert a raw event coming from the ZMQ channel into a placement-aware worker event.
pub fn convert_event(
    raw: RawKvEvent,
    event_id: u64,
    kv_block_size: u32,
    worker: WorkerWithDpRank,
    warning_count: &Arc<AtomicU32>,
) -> Option<PlacementEvent> {
    let storage_tier = match &raw {
        RawKvEvent::BlockStored { medium, .. } | RawKvEvent::BlockRemoved { medium, .. } => {
            StorageTier::from_kv_medium_or_default(medium.as_deref())
        }
        RawKvEvent::AllBlocksCleared => StorageTier::Device,
        RawKvEvent::Ignored => return None,
    };
    let dp_rank = worker.dp_rank;
    let event = match raw {
        RawKvEvent::BlockStored {
            block_hashes,
            parent_block_hash,
            token_ids,
            block_size,
            lora_name,
            block_mm_infos,
            medium: _,
            is_eagle,
        } => {
            // Reject self-referencing blocks: all block hashes (including parent) must be unique.
            {
                let mut seen = HashSet::with_capacity(block_hashes.len() + 1);
                if let Some(parent) = parent_block_hash {
                    seen.insert(parent.into_u64());
                }
                let has_duplicate = block_hashes.iter().any(|h| !seen.insert(h.into_u64()));
                if has_duplicate {
                    tracing::warn!(
                        event_id,
                        "Self-referencing block detected: duplicate hash in store event; dropping"
                    );
                    // Return an empty Removed instead of Cleared to avoid nuking
                    // the worker's entire index state. An empty Removed is a no-op
                    // in the radix tree (zero iterations, returns Ok(())).
                    return Some(PlacementEvent::new(
                        Placement::local_worker(worker.worker_id, worker.dp_rank, storage_tier),
                        KvCacheEvent {
                            event_id,
                            data: KvCacheEventData::Removed(KvCacheRemoveData {
                                block_hashes: vec![],
                            }),
                            dp_rank,
                        },
                    ));
                }
            }

            let num_block_tokens = vec![block_size as u64; block_hashes.len()];
            let block_hashes_u64: Vec<u64> = block_hashes
                .into_iter()
                .map(BlockHashValue::into_u64)
                .collect();
            KvCacheEvent {
                event_id,
                data: KvCacheEventData::Stored(KvCacheStoreData {
                    parent_hash: parent_block_hash
                        .map(BlockHashValue::into_u64)
                        .map(ExternalSequenceBlockHash::from),
                    start_position: None,
                    blocks: create_stored_blocks(
                        kv_block_size,
                        &token_ids,
                        &num_block_tokens,
                        &block_hashes_u64,
                        lora_name.as_deref(),
                        warning_count,
                        block_mm_infos.as_deref(),
                        is_eagle,
                    ),
                }),
                dp_rank,
            }
        }
        RawKvEvent::BlockRemoved { block_hashes, .. } => {
            let hashes = block_hashes
                .into_iter()
                .map(BlockHashValue::into_u64)
                .map(ExternalSequenceBlockHash::from)
                .collect();
            KvCacheEvent {
                event_id,
                data: KvCacheEventData::Removed(KvCacheRemoveData {
                    block_hashes: hashes,
                }),
                dp_rank,
            }
        }
        RawKvEvent::AllBlocksCleared => KvCacheEvent {
            event_id,
            data: KvCacheEventData::Cleared,
            dp_rank,
        },
        RawKvEvent::Ignored => unreachable!("ignored events return before conversion"),
    };

    Some(PlacementEvent::new(
        Placement::local_worker(worker.worker_id, worker.dp_rank, storage_tier),
        event,
    ))
}

pub fn create_stored_block_from_parts(
    kv_block_size: u32,
    block_hash: u64,
    token_ids: &[u32],
    lora_name: Option<&str>,
    mm_extra_info: Option<BlockExtraInfo>,
    is_eagle: Option<bool>,
) -> KvCacheStoredBlockData {
    let block_mm_infos = mm_extra_info.as_ref().map(|info| vec![Some(info.clone())]);
    let tokens_hash = compute_block_hash_for_seq(
        token_ids,
        kv_block_size,
        BlockHashOptions {
            block_mm_infos: block_mm_infos.as_deref(),
            lora_name,
            is_eagle,
        },
    )[0];

    tracing::trace!(
        "Creating stored block: external_block_hash={}, tokens_hash={}, token_ids={:?}, kv_block_size={}, mm_extra_info={:?}",
        block_hash,
        tokens_hash.0,
        token_ids,
        kv_block_size,
        mm_extra_info
    );
    KvCacheStoredBlockData {
        block_hash: ExternalSequenceBlockHash::from(block_hash),
        tokens_hash,
        mm_extra_info,
    }
}

#[allow(clippy::too_many_arguments)]
pub fn create_stored_blocks(
    kv_block_size: u32,
    token_ids: &[u32],
    num_block_tokens: &[u64],
    block_hashes: &[u64],
    lora_name: Option<&str>,
    warning_count: &Arc<AtomicU32>,
    block_mm_infos: Option<&[Option<BlockExtraInfo>]>,
    is_eagle: Option<bool>,
) -> Vec<KvCacheStoredBlockData> {
    let mut blocks: Vec<KvCacheStoredBlockData> = Vec::new();

    let mut token_offset: usize = 0;
    let append = is_eagle.unwrap_or(false) as usize;

    for (block_idx, (num_tokens_it, block_hash_it)) in
        num_block_tokens.iter().zip(block_hashes.iter()).enumerate()
    {
        if *num_tokens_it != kv_block_size as u64 {
            if warning_count.fetch_add(1, Ordering::Relaxed) < 3 {
                tracing::warn!(
                    "Block not published. Block size must be {} tokens to be published. Block size is: {}",
                    kv_block_size,
                    *num_tokens_it
                );
            }
            break;
        }

        let end = token_offset + append + *num_tokens_it as usize;
        if end > token_ids.len() {
            if warning_count.fetch_add(1, Ordering::Relaxed) < 3 {
                tracing::warn!(
                    "Block not published. token_ids too short: need {}, got {}",
                    end,
                    token_ids.len()
                );
            }
            break;
        }

        let tokens = &token_ids[token_offset..end];
        let mm_extra_info = block_mm_infos
            .and_then(|infos| infos.get(block_idx))
            .and_then(|opt| opt.clone());

        blocks.push(create_stored_block_from_parts(
            kv_block_size,
            *block_hash_it,
            tokens,
            lora_name,
            mm_extra_info,
            is_eagle,
        ));
        token_offset += *num_tokens_it as usize;
    }

    blocks
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;
    use std::sync::atomic::AtomicU32;

    use rmp_serde::{from_slice, to_vec};
    use rstest::rstest;

    use super::*;

    #[derive(Clone, Copy, Debug)]
    enum TestEventKind {
        BlockStored,
        BlockRemoved,
    }

    #[test]
    fn test_deserialize_bigram_block_stored_sequence() {
        let raw_event = (
            "BlockStored",
            vec![BlockHashValue::Unsigned(11), BlockHashValue::Unsigned(12)],
            Option::<BlockHashValue>::None,
            vec![(10u32, 11u32), (11, 12), (12, 13), (13, 14)],
            2usize,
            Option::<u64>::None,
            Option::<String>::None,
            Option::<String>::None,
        );
        let encoded = to_vec(&raw_event).unwrap();
        let event: RawKvEvent = from_slice(&encoded).unwrap();

        match event {
            RawKvEvent::BlockStored {
                token_ids,
                block_size,
                is_eagle,
                ..
            } => {
                assert_eq!(token_ids, vec![10, 11, 12, 13, 14]);
                assert_eq!(block_size, 2);
                assert_eq!(is_eagle, Some(true));
            }
            other => panic!("expected BlockStored, got {other:?}"),
        }
    }

    fn block_stored_sequence(
        group_idx: Option<u32>,
        kv_cache_spec_kind: Option<&'static str>,
    ) -> Vec<u8> {
        match (group_idx, kv_cache_spec_kind) {
            (Some(group_idx), Some(kv_cache_spec_kind)) => to_vec(&(
                "BlockStored",
                vec![BlockHashValue::Unsigned(11)],
                Option::<BlockHashValue>::None,
                vec![10u32, 11],
                2usize,
                Option::<u64>::None,
                Option::<String>::None,
                Option::<String>::None,
                Option::<u8>::None,
                group_idx,
                kv_cache_spec_kind,
            ))
            .unwrap(),
            (Some(group_idx), None) => to_vec(&(
                "BlockStored",
                vec![BlockHashValue::Unsigned(11)],
                Option::<BlockHashValue>::None,
                vec![10u32, 11],
                2usize,
                Option::<u64>::None,
                Option::<String>::None,
                Option::<String>::None,
                Option::<u8>::None,
                group_idx,
            ))
            .unwrap(),
            (None, Some(kv_cache_spec_kind)) => to_vec(&(
                "BlockStored",
                vec![BlockHashValue::Unsigned(11)],
                Option::<BlockHashValue>::None,
                vec![10u32, 11],
                2usize,
                Option::<u64>::None,
                Option::<String>::None,
                Option::<String>::None,
                Option::<u8>::None,
                Option::<u32>::None,
                kv_cache_spec_kind,
            ))
            .unwrap(),
            (None, None) => to_vec(&(
                "BlockStored",
                vec![BlockHashValue::Unsigned(11)],
                Option::<BlockHashValue>::None,
                vec![10u32, 11],
                2usize,
                Option::<u64>::None,
                Option::<String>::None,
                Option::<String>::None,
            ))
            .unwrap(),
        }
    }

    fn block_removed_sequence(
        group_idx: Option<u32>,
        kv_cache_spec_kind: Option<&'static str>,
    ) -> Vec<u8> {
        match (group_idx, kv_cache_spec_kind) {
            (Some(group_idx), Some(kv_cache_spec_kind)) => to_vec(&(
                "BlockRemoved",
                vec![BlockHashValue::Unsigned(11)],
                Option::<String>::None,
                group_idx,
                kv_cache_spec_kind,
            ))
            .unwrap(),
            (Some(group_idx), None) => to_vec(&(
                "BlockRemoved",
                vec![BlockHashValue::Unsigned(11)],
                Option::<String>::None,
                group_idx,
            ))
            .unwrap(),
            (None, Some(kv_cache_spec_kind)) => to_vec(&(
                "BlockRemoved",
                vec![BlockHashValue::Unsigned(11)],
                Option::<String>::None,
                Option::<u32>::None,
                kv_cache_spec_kind,
            ))
            .unwrap(),
            (None, None) => to_vec(&(
                "BlockRemoved",
                vec![BlockHashValue::Unsigned(11)],
                Option::<String>::None,
            ))
            .unwrap(),
        }
    }

    fn sequence_with_group_idx(event_kind: TestEventKind, group_idx: Option<u32>) -> Vec<u8> {
        match event_kind {
            TestEventKind::BlockStored => block_stored_sequence(group_idx, None),
            TestEventKind::BlockRemoved => block_removed_sequence(group_idx, None),
        }
    }

    fn sequence_with_cache_spec_kind(
        event_kind: TestEventKind,
        group_idx: Option<u32>,
        kv_cache_spec_kind: &'static str,
    ) -> Vec<u8> {
        match event_kind {
            TestEventKind::BlockStored => {
                block_stored_sequence(group_idx, Some(kv_cache_spec_kind))
            }
            TestEventKind::BlockRemoved => {
                block_removed_sequence(group_idx, Some(kv_cache_spec_kind))
            }
        }
    }

    fn sequence_with_cache_spec_kind_without_group_idx_slot(
        event_kind: TestEventKind,
        kv_cache_spec_kind: &'static str,
    ) -> Vec<u8> {
        match event_kind {
            TestEventKind::BlockStored => to_vec(&(
                "BlockStored",
                vec![BlockHashValue::Unsigned(11)],
                Option::<BlockHashValue>::None,
                vec![10u32, 11],
                2usize,
                Option::<u64>::None,
                Option::<String>::None,
                Option::<String>::None,
                Option::<u8>::None,
                kv_cache_spec_kind,
            ))
            .unwrap(),
            TestEventKind::BlockRemoved => to_vec(&(
                "BlockRemoved",
                vec![BlockHashValue::Unsigned(11)],
                Option::<String>::None,
                kv_cache_spec_kind,
            ))
            .unwrap(),
        }
    }

    fn sequence_with_cache_spec_kind_and_sliding_window(
        event_kind: TestEventKind,
        group_idx: u32,
        kv_cache_spec_kind: &'static str,
        kv_cache_spec_sliding_window: u32,
    ) -> Vec<u8> {
        match event_kind {
            TestEventKind::BlockStored => to_vec(&(
                "BlockStored",
                vec![BlockHashValue::Unsigned(11)],
                Option::<BlockHashValue>::None,
                vec![10u32, 11],
                2usize,
                Option::<u64>::None,
                Option::<String>::None,
                Option::<String>::None,
                Option::<u8>::None,
                group_idx,
                kv_cache_spec_kind,
                kv_cache_spec_sliding_window,
            ))
            .unwrap(),
            TestEventKind::BlockRemoved => to_vec(&(
                "BlockRemoved",
                vec![BlockHashValue::Unsigned(11)],
                Option::<String>::None,
                group_idx,
                kv_cache_spec_kind,
                kv_cache_spec_sliding_window,
            ))
            .unwrap(),
        }
    }

    fn assert_parsed_event_kind(event: RawKvEvent, expected_kind: TestEventKind) {
        match (event, expected_kind) {
            (RawKvEvent::BlockStored { .. }, TestEventKind::BlockStored)
            | (RawKvEvent::BlockRemoved { .. }, TestEventKind::BlockRemoved) => {}
            (event, expected_kind) => {
                panic!("expected {expected_kind:?}, got {event:?}");
            }
        }
    }

    #[rstest]
    #[case(TestEventKind::BlockStored)]
    #[case(TestEventKind::BlockRemoved)]
    fn test_deserialize_sequence_accepts_main_group_idx(#[case] event_kind: TestEventKind) {
        let event: RawKvEvent = from_slice(&sequence_with_group_idx(event_kind, Some(0))).unwrap();

        assert_parsed_event_kind(event, event_kind);
    }

    #[rstest]
    #[case(TestEventKind::BlockStored)]
    #[case(TestEventKind::BlockRemoved)]
    fn test_deserialize_sequence_ignores_non_main_group_idx(#[case] event_kind: TestEventKind) {
        let event: RawKvEvent = from_slice(&sequence_with_group_idx(event_kind, Some(1))).unwrap();

        assert!(matches!(event, RawKvEvent::Ignored));
    }

    #[rstest]
    #[case(TestEventKind::BlockStored)]
    #[case(TestEventKind::BlockRemoved)]
    fn test_deserialize_sequence_accepts_missing_group_idx(#[case] event_kind: TestEventKind) {
        let event: RawKvEvent = from_slice(&sequence_with_group_idx(event_kind, None)).unwrap();

        assert_parsed_event_kind(event, event_kind);
    }

    #[rstest]
    #[case(TestEventKind::BlockStored)]
    #[case(TestEventKind::BlockRemoved)]
    fn test_deserialize_sequence_accepts_main_attention_kind_with_nonzero_group_idx(
        #[case] event_kind: TestEventKind,
    ) {
        let event: RawKvEvent = from_slice(&sequence_with_cache_spec_kind(
            event_kind,
            Some(3),
            "full_attention",
        ))
        .unwrap();

        assert_parsed_event_kind(event, event_kind);
    }

    #[rstest]
    #[case(TestEventKind::BlockStored)]
    #[case(TestEventKind::BlockRemoved)]
    fn test_deserialize_sequence_accepts_main_attention_kind_without_group_idx_slot(
        #[case] event_kind: TestEventKind,
    ) {
        let event: RawKvEvent = from_slice(&sequence_with_cache_spec_kind_without_group_idx_slot(
            event_kind,
            "full_attention",
        ))
        .unwrap();

        assert_parsed_event_kind(event, event_kind);
    }

    #[rstest]
    #[case(TestEventKind::BlockStored)]
    #[case(TestEventKind::BlockRemoved)]
    fn test_deserialize_sequence_accepts_main_attention_kind_with_sliding_window(
        #[case] event_kind: TestEventKind,
    ) {
        let event: RawKvEvent = from_slice(&sequence_with_cache_spec_kind_and_sliding_window(
            event_kind,
            3,
            "full_attention",
            128,
        ))
        .unwrap();

        assert_parsed_event_kind(event, event_kind);
    }

    #[rstest]
    #[case(TestEventKind::BlockStored)]
    #[case(TestEventKind::BlockRemoved)]
    fn test_deserialize_sequence_ignores_non_main_attention_kind_with_group_idx_zero(
        #[case] event_kind: TestEventKind,
    ) {
        let event: RawKvEvent =
            from_slice(&sequence_with_cache_spec_kind(event_kind, Some(0), "mamba")).unwrap();

        assert!(matches!(event, RawKvEvent::Ignored));
    }

    #[test]
    fn test_convert_event_bigram_emits_eagle_windows() {
        let raw_event = RawKvEvent::BlockStored {
            block_hashes: vec![BlockHashValue::Unsigned(21), BlockHashValue::Unsigned(22)],
            parent_block_hash: None,
            token_ids: vec![10, 11, 12, 13, 14],
            block_size: 2,
            medium: None,
            lora_name: None,
            block_mm_infos: None,
            is_eagle: Some(true),
        };
        let warning_count = Arc::new(AtomicU32::new(0));
        let placement_event =
            convert_event(raw_event, 7, 2, WorkerWithDpRank::new(3, 0), &warning_count);

        match placement_event.unwrap().event.data {
            KvCacheEventData::Stored(store_data) => {
                assert_eq!(store_data.blocks.len(), 2);
                assert_eq!(
                    store_data.blocks[0].block_hash,
                    ExternalSequenceBlockHash(21)
                );
                assert_eq!(
                    store_data.blocks[1].block_hash,
                    ExternalSequenceBlockHash(22)
                );

                let expected_first = compute_block_hash_for_seq(
                    &[10, 11, 12],
                    2,
                    BlockHashOptions {
                        is_eagle: Some(true),
                        ..Default::default()
                    },
                );
                let expected_second = compute_block_hash_for_seq(
                    &[12, 13, 14],
                    2,
                    BlockHashOptions {
                        is_eagle: Some(true),
                        ..Default::default()
                    },
                );

                assert_eq!(store_data.blocks[0].tokens_hash, expected_first[0]);
                assert_eq!(store_data.blocks[1].tokens_hash, expected_second[0]);
            }
            other => panic!("expected Stored event, got {other:?}"),
        }
    }
}
