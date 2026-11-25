// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Converts the responses of vLLM's `KvEvents` service into the batches the
//! ZMQ decoder returns for the same events, through the decoder's own helpers.

use anyhow::{Context as _, Result, bail};
use dynamo_sidecar_common::kv_events::wire::{
    BlockHashValue, ExtraKeyItem, KvCacheSpecKind, KvEventBatch, Locality, RawKvEvent,
    extra_keys_to_block_mm_infos, extra_keys_to_cache_namespace, mark_first_raw_key_block,
    normalize_medium,
};
use dynamo_sidecar_common::kv_events::{KvBatch, KvSnapshotItem};

use crate::proto as pb;

#[cfg(test)]
mod tests;

/// The batch of a stream response; `None` for a response without one.
pub(crate) fn batch(response: pb::SubscribeKvEventsResponse) -> Option<KvBatch> {
    let pb::subscribe_kv_events_response::Item::Batch(batch) = response.item?;
    let events = batch.events.into_iter().map(event).collect::<Result<_>>();
    Some(events.map(|events| {
        (
            batch.sequence_number,
            KvEventBatch {
                ts: batch.ts,
                events,
                data_parallel_rank: batch.data_parallel_rank.map(u32::cast_signed),
            },
        )
    }))
}

/// The stores or the end of a snapshot response; `None` for a response without either.
pub(crate) fn snapshot_item(response: pb::GetKvSnapshotResponse) -> Option<Result<KvSnapshotItem>> {
    use pb::get_kv_snapshot_response::Item;
    Some(match response.item? {
        Item::Blocks(blocks) => {
            let stores = blocks.stored.into_iter().map(block_stored);
            stores.collect::<Result<_>>().map(KvSnapshotItem::Stores)
        }
        Item::End(end) => Ok(KvSnapshotItem::End {
            next_sequence: end.next_sequence_number,
            is_complete: end.complete,
        }),
    })
}

fn event(event: pb::KvCacheEvent) -> Result<RawKvEvent> {
    use pb::kv_cache_event::Event;
    Ok(
        match event.event.context("KV cache event without a type")? {
            Event::BlockStored(stored) => block_stored(stored)?,
            Event::BlockRemoved(removed) => RawKvEvent::BlockRemoved {
                block_hashes: hashes(removed.block_hashes)?,
                medium: normalize_medium(removed.medium),
                group_idx: removed.group_idx,
                kv_cache_spec_kind: None,
                kv_cache_spec_sliding_window: None,
                locality: removed.locality.as_deref().map(Locality::from_wire),
                ownership: removed.ownership,
            },
            Event::AllBlocksCleared(_) => RawKvEvent::AllBlocksCleared { ownership: None },
        },
    )
}

fn block_stored(stored: pb::BlockStored) -> Result<RawKvEvent> {
    let mut token_ids = stored.token_ids;
    let block_size = stored.block_size as usize;
    // The decoder's helpers read an empty entry as they read `None`.
    let extra_keys: Vec<Option<Vec<ExtraKeyItem>>> = stored
        .extra_keys
        .into_iter()
        .map(|block| {
            let keys = block.keys.into_iter().map(extra_key);
            keys.collect::<Result<_>>().map(Some)
        })
        .collect::<Result<_>>()?;
    mark_first_raw_key_block(&mut token_ids, block_size, Some(&extra_keys));
    let cache_namespace =
        extra_keys_to_cache_namespace(Some(&extra_keys), stored.lora_name.as_deref());
    Ok(RawKvEvent::BlockStored {
        block_hashes: hashes(stored.block_hashes)?,
        parent_block_hash: stored.parent_block_hash.map(hash).transpose()?,
        token_ids,
        block_size,
        medium: normalize_medium(stored.medium),
        lora_name: stored.lora_name,
        cache_namespace,
        block_mm_infos: extra_keys_to_block_mm_infos(Some(extra_keys)),
        is_eagle: Some(false),
        group_idx: stored.group_idx,
        kv_cache_spec_kind: stored
            .kv_cache_spec_kind
            .as_deref()
            .map(KvCacheSpecKind::from_wire),
        kv_cache_spec_sliding_window: stored.kv_cache_spec_sliding_window,
        locality: stored.locality.as_deref().map(Locality::from_wire),
        ownership: stored.ownership,
        session_id: stored.session_id,
    })
}

fn hashes(hashes: Vec<pb::BlockHash>) -> Result<Vec<BlockHashValue>> {
    hashes.into_iter().map(hash).collect()
}

fn hash(hash: pb::BlockHash) -> Result<BlockHashValue> {
    use pb::block_hash::Value;
    match hash.value {
        Some(Value::UnsignedValue(value)) => Ok(BlockHashValue::Unsigned(value)),
        Some(Value::SignedValue(value)) => Ok(BlockHashValue::Signed(value)),
        // The ZMQ decoder rejects byte hashes as well.
        Some(Value::Digest(_)) => bail!("byte block hashes are not supported"),
        None => bail!("block hash without a value"),
    }
}

fn extra_key(key: pb::ExtraKeyValue) -> Result<ExtraKeyItem> {
    use pb::extra_key_value::Value;
    Ok(match key.value.context("extra key without a value")? {
        Value::Text(text) => ExtraKeyItem::Hash(text),
        Value::Data(data) => ExtraKeyItem::Bytes(data),
        Value::Integer(value) => ExtraKeyItem::Signed(value),
        Value::Tuple(tuple) => match <[pb::ExtraKeyValue; 2]>::try_from(tuple.items) {
            Ok(
                [
                    pb::ExtraKeyValue {
                        value: Some(Value::Text(hash)),
                    },
                    pb::ExtraKeyValue {
                        value: Some(Value::Integer(offset)),
                    },
                ],
            ) => ExtraKeyItem::HashWithSignedOffset((hash, offset)),
            _ => bail!("unsupported extra key tuple"),
        },
    })
}
