// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Replay order for the lower-tier stores of a tree dump.

use rustc_hash::FxHashMap;

use crate::prelude::{KvCacheEventData, KvCacheStoreData, RouterEvent};

/// Reorders each stretch of consecutive stores to one lower tier so that a
/// block's store comes after its parent's. No store moves past an event of
/// another kind or tier, so the replay ends in the same state.
///
/// The lower tier stores consecutive blocks together only when a block arrives
/// after its parent. Senders through v1.7 list a lower tier's blocks in hash
/// order, which would rebuild a chain as runs of about two blocks.
///
/// Compatibility with v1.6 and v1.7 senders during rolling upgrades.
/// TODO(v1.10): Remove when v1.7 falls outside the N-2 compatibility window.
pub fn order_lower_tier_dump(events: Vec<RouterEvent>) -> Vec<RouterEvent> {
    let mut ordered = Vec::with_capacity(events.len());
    let mut events = events.into_iter().peekable();
    let mut stores = Vec::new();
    while let Some(event) = events.next() {
        if event.storage_tier.is_gpu() || stored(&event).is_none() {
            ordered.push(event);
            continue;
        }
        let tier = event.storage_tier;
        stores.push(event);
        while let Some(next) =
            events.next_if(|next| next.storage_tier == tier && stored(next).is_some())
        {
            stores.push(next);
        }
        append_parents_first(&mut stores, &mut ordered);
    }
    ordered
}

fn stored(event: &RouterEvent) -> Option<&KvCacheStoreData> {
    match &event.event.data {
        KvCacheEventData::Stored(store) => Some(store),
        _ => None,
    }
}

/// Moves `stores` to the end of `ordered`, parents first.
fn append_parents_first(stores: &mut Vec<RouterEvent>, ordered: &mut Vec<RouterEvent>) {
    if stores.len() == 1 {
        ordered.append(stores);
        return;
    }
    // Which store holds each block.
    let mut stored_by = FxHashMap::default();
    for (index, event) in stores.iter().enumerate() {
        for block in stored(event).into_iter().flat_map(|store| &store.blocks) {
            stored_by.entry(block.block_hash).or_insert(index);
        }
    }
    let mut pending: Vec<_> = stores.drain(..).map(Some).collect();
    let mut chain = Vec::new();
    for start in 0..pending.len() {
        // Walk up through the stores not moved yet, then move them top down.
        let mut next = Some(start);
        while let Some(index) = next
            && let Some(event) = pending[index].take()
        {
            next = stored(&event)
                .and_then(|store| store.parent_hash)
                .and_then(|hash| stored_by.get(&hash).copied());
            chain.push(event);
        }
        ordered.extend(chain.drain(..).rev());
    }
}

#[cfg(test)]
mod tests {
    use super::super::{LowerTierIndexer, WorkerBlockIndex};
    use super::order_lower_tier_dump;
    use crate::prelude::{
        ExternalSequenceBlockHash, KvCacheEvent, KvCacheEventData, KvCacheRemoveData,
        KvCacheStoreData, LocalBlockHash, RouterEvent, StorageTier,
    };
    use crate::testing::stored_blocks_with_sequence_hashes;

    fn event(id: u64, tier: StorageTier, data: KvCacheEventData) -> RouterEvent {
        let event = KvCacheEvent {
            event_id: id,
            dp_rank: 0,
            data,
        };
        RouterEvent::with_storage_tier(7, event, tier)
    }

    /// A single-block store of block `n` of one chain; block 0 is the root.
    fn store(n: u64, tier: StorageTier) -> RouterEvent {
        let data = KvCacheEventData::Stored(KvCacheStoreData {
            parent_hash: (n > 0).then(|| ExternalSequenceBlockHash(100 + n - 1)),
            start_position: None,
            blocks: stored_blocks_with_sequence_hashes(&[LocalBlockHash(n)], &[100 + n]),
        });
        event(n, tier, data)
    }

    #[test]
    fn a_hash_ordered_dump_rebuilds_whole_runs() {
        let host = StorageTier::HostPinned;
        let remove = KvCacheEventData::Removed(KvCacheRemoveData {
            block_hashes: vec![ExternalSequenceBlockHash(120)],
        });
        // A device store, one host chain of 40 blocks with children before
        // parents, as a v1.7 sender can list it, then a live tail.
        let tail = [event(40, host, remove), store(20, host)];
        let mut dump = vec![store(0, StorageTier::Device)];
        dump.extend((0..40).rev().map(|n| store(n, host)));
        dump.extend(tail.clone());

        let ordered = order_lower_tier_dump(dump.clone());

        let mut expected = vec![dump[0].clone()];
        expected.extend((0..40).map(|n| store(n, host)));
        expected.extend(tail);
        assert_eq!(ordered, expected);

        let index = LowerTierIndexer::new();
        let mut lane = WorkerBlockIndex::default();
        for event in ordered.into_iter().skip(1) {
            index.apply_event(&mut lane, event).unwrap();
        }
        let census = index.run_census();
        // Tests cap a run at 16 blocks.
        assert_eq!((census.runs, census.held_blocks), (3, 40));
    }
}
