// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use tokio::sync::oneshot;

use crate::block_manager::block::transfer::remote::RemoteKey;
use crate::block_manager::distributed::registry::{NoMetadata, PositionalKey};

/// In-flight async G4 lookup state for a request slot.
/// Host/disk matches are preserved while remote lookup runs.
pub struct PendingG4Lookup<H, D> {
    pub num_computed_tokens: usize,
    pub host_blocks: H,
    pub disk_blocks: D,
    pub world_size: usize,
    pub receiver: oneshot::Receiver<Vec<(PositionalKey, RemoteKey, NoMetadata)>>,
}

/// Compute strict TP positional consensus from positional-key registry matches.
pub fn compute_tp_consensus_hashes(
    matches: Vec<(PositionalKey, RemoteKey, NoMetadata)>,
    world_size: usize,
) -> Vec<u64> {
    if world_size == 0 {
        return vec![];
    }

    let mut per_worker: Vec<std::collections::BTreeMap<u32, u64>> =
        vec![std::collections::BTreeMap::new(); world_size];
    for (key, _, _) in matches {
        let wid = key.worker_id as usize;
        if wid < world_size {
            per_worker[wid]
                .entry(key.position)
                .or_insert(key.sequence_hash);
        }
    }

    let mut consensus = Vec::new();
    let mut pos: u32 = 0;
    loop {
        let Some(hash0) = per_worker[0].get(&pos).copied() else {
            break;
        };
        if per_worker
            .iter()
            .all(|worker| worker.get(&pos).copied() == Some(hash0))
        {
            consensus.push(hash0);
            pos = pos.saturating_add(1);
        } else {
            break;
        }
    }
    consensus
}
