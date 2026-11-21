// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! The hybrid-worker mode of `mooncake_bench` (`--recurrent-checkpoints`).
//!
//! A hybrid model keeps recurrent state next to its attention KV cache. A worker can resume a
//! cached prefix only at a block where it holds a state checkpoint for every recurrent group, so
//! the router caps each hybrid worker's match at its deepest checkpoint. In this mode every
//! worker declares a layout of [`RECURRENT_GROUPS`] and publishes each checkpoint as vLLM does:
//! one tagged event per group.

use std::collections::{HashMap, HashSet};

use dynamo_kv_events::event::{
    KvCacheEvent, KvCacheEventData, KvCacheRemoveData, KvCacheStoreData,
};
use dynamo_kv_events::residency::{KvCacheGroupKind, KvCacheGroupTag, StorageTier};

use super::mooncake_shared::{WorkerTrace, WorkerTraceEntry};

/// A worker's recurrent cache groups. Qwen3.5 has three.
pub(crate) const RECURRENT_GROUPS: [u32; 3] = [0, 1, 2];

/// Which blocks of a worker's stores carry a checkpoint.
#[derive(Clone, Copy, Debug, PartialEq, Eq, clap::ValueEnum)]
pub enum CheckpointPolicy {
    /// No block. Workers declare the layout and never publish a checkpoint.
    None,
    /// The last block of each store.
    StoreEnd,
    /// Every stored block.
    EveryBlock,
}

pub(crate) fn group_tag(group: u32) -> KvCacheGroupTag {
    KvCacheGroupTag {
        group_idx: group,
        kind: KvCacheGroupKind::Mamba,
        expected_group_count: Some(RECURRENT_GROUPS.len() as u32),
        alignment_blocks: Some(1),
        eagle_margin_blocks: None,
    }
}

/// Adds the checkpoint layer to one worker's timeline. After each store the worker owns, on any
/// tier, the blocks `policy` picks become checkpoints on that tier: one `Stored` per group. A
/// removal takes its blocks' checkpoints with it: one `Removed` per group. KVCR cache owners
/// publish no checkpoint.
pub(crate) fn derive_checkpoint_layer(
    timeline: Vec<WorkerTrace>,
    policy: CheckpointPolicy,
) -> Vec<WorkerTrace> {
    let mut live: HashMap<StorageTier, HashSet<u64>> = HashMap::new();
    let mut event_id = 0u64;
    let mut out = Vec::with_capacity(timeline.len());
    for entry in timeline {
        let derived = match &entry.entry {
            WorkerTraceEntry::Event {
                event,
                storage_tier,
                cache_owner: false,
            } => {
                let tier_live = live.entry(*storage_tier).or_default();
                let data = match &event.data {
                    KvCacheEventData::Stored(store) => {
                        let picked = match policy {
                            CheckpointPolicy::None => &[][..],
                            CheckpointPolicy::StoreEnd => {
                                &store.blocks[store.blocks.len().saturating_sub(1)..]
                            }
                            CheckpointPolicy::EveryBlock => &store.blocks[..],
                        };
                        let blocks: Vec<_> = picked
                            .iter()
                            .filter(|block| tier_live.insert(block.block_hash.0))
                            .cloned()
                            .collect();
                        let nonempty = !blocks.is_empty();
                        nonempty.then_some(KvCacheEventData::Stored(KvCacheStoreData {
                            parent_hash: None,
                            start_position: None,
                            blocks,
                        }))
                    }
                    KvCacheEventData::Removed(remove) => {
                        let block_hashes: Vec<_> = remove
                            .block_hashes
                            .iter()
                            .filter(|hash| tier_live.remove(&hash.0))
                            .copied()
                            .collect();
                        let nonempty = !block_hashes.is_empty();
                        nonempty.then_some(KvCacheEventData::Removed(KvCacheRemoveData {
                            block_hashes,
                        }))
                    }
                    // The router drops a worker's checkpoints on its untagged clear.
                    KvCacheEventData::Cleared => {
                        tier_live.clear();
                        None
                    }
                };
                data.map(|data| (data, *storage_tier, event.dp_rank))
            }
            _ => None,
        };
        let timestamp_us = entry.timestamp_us;
        out.push(entry);
        let Some((data, storage_tier, dp_rank)) = derived else {
            continue;
        };
        for group in RECURRENT_GROUPS {
            out.push(WorkerTrace {
                timestamp_us,
                entry: WorkerTraceEntry::Checkpoint {
                    event: KvCacheEvent {
                        event_id,
                        data: data.clone(),
                        dp_rank,
                    },
                    storage_tier,
                    group,
                },
            });
            event_id += 1;
        }
    }
    out
}
