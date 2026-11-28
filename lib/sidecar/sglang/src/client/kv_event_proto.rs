// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Converts a response of SGLang's `KvEvents` stream into the batch the ZMQ
//! decoder returns for the same events, through the decoder's own helpers.

use anyhow::{Context as _, Result};
use dynamo_sidecar_common::kv_events::KvBatch;
use dynamo_sidecar_common::kv_events::wire::{
    BlockHashValue, KvEventBatch, KvTokenIds, RawKvEvent, normalize_medium, normalize_token_ids,
};
use sglang_kv_types::kv::v1 as pb;

#[cfg(test)]
mod tests;

/// The batch of a stream response; `None` for the stream's start.
pub(crate) fn batch(response: pb::SubscribeKvEventsResponse) -> Option<KvBatch> {
    let pb::subscribe_kv_events_response::Item::Batch(batch) = response.item? else {
        return None;
    };
    let events = batch.events.into_iter().map(event).collect::<Result<_>>();
    Some(events.map(|events| {
        (
            batch.sequence_number,
            KvEventBatch {
                ts: batch.ts,
                events,
                data_parallel_rank: batch.attn_dp_rank.map(u32::cast_signed),
            },
        )
    }))
}

fn event(event: pb::KvCacheEvent) -> Result<RawKvEvent> {
    use pb::kv_cache_event::Event;
    Ok(
        match event.event.context("KV cache event without a type")? {
            Event::BlockStored(stored) => block_stored(stored)?,
            Event::BlockRemoved(removed) => RawKvEvent::BlockRemoved {
                block_hashes: hashes(removed.block_hashes),
                medium: normalize_medium(removed.medium),
                group_idx: None,
                kv_cache_spec_kind: None,
                kv_cache_spec_sliding_window: None,
                locality: None,
                ownership: None,
            },
            Event::AllBlocksCleared(_) => RawKvEvent::AllBlocksCleared { ownership: None },
        },
    )
}

fn block_stored(stored: pb::BlockStored) -> Result<RawKvEvent> {
    use pb::block_stored::TokenIds;
    let token_ids = match stored.token_ids.context("BlockStored without token_ids")? {
        TokenIds::Single(list) => KvTokenIds::Single(list.token_ids),
        TokenIds::Pairs(list) => KvTokenIds::Bigram(
            list.pairs
                .into_iter()
                .map(|pair| (pair.first, pair.second))
                .collect(),
        ),
    };
    let (token_ids, is_eagle) = normalize_token_ids(token_ids);
    Ok(RawKvEvent::BlockStored {
        block_hashes: hashes(stored.block_hashes),
        parent_block_hash: stored.parent_block_hash.map(BlockHashValue::Signed),
        token_ids,
        block_size: stored.block_size as usize,
        medium: normalize_medium(stored.medium),
        lora_name: stored.lora_name,
        cache_namespace: stored.cache_salt,
        block_mm_infos: None,
        is_eagle: Some(is_eagle),
        group_idx: None,
        kv_cache_spec_kind: None,
        kv_cache_spec_sliding_window: None,
        locality: None,
        ownership: None,
        session_id: stored.session_id,
    })
}

fn hashes(hashes: Vec<i64>) -> Vec<BlockHashValue> {
    hashes.into_iter().map(BlockHashValue::Signed).collect()
}
