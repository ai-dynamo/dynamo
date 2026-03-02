// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use crate::block_manager::block::transfer::remote::{DiskKey, ObjectKey, RemoteKey};
use crate::block_manager::config::RemoteStorageConfig;
use crate::block_manager::distributed::remote::PositionalRemoteHandle;
use crate::block_manager::distributed::{RemoteHashOperations, RemoteHashOperationsSync};
use crate::tokens::SequenceHash;

/// Lookup hashes with TP consensus.
pub async fn match_prefix_tp(
    handle: &PositionalRemoteHandle,
    hashes: &[SequenceHash],
    world_size: usize,
) -> Vec<SequenceHash> {
    if hashes.is_empty() || world_size == 0 {
        return vec![];
    }

    if world_size == 1 {
        return handle.lookup_hashes(hashes, 0).await;
    }

    let all_keys: Vec<_> = (0..world_size)
        .flat_map(|wid| {
            hashes.iter().enumerate().map(move |(pos, &hash)| {
                crate::block_manager::distributed::registry::PositionalKey {
                    worker_id: wid as u64,
                    sequence_hash: hash,
                    position: pos as u32,
                }
            })
        })
        .collect();

    let matches = handle.match_prefix(all_keys).await;
    let mut per_worker: Vec<Vec<SequenceHash>> = vec![vec![]; world_size];
    for (key, _, _) in &matches {
        if (key.worker_id as usize) < world_size {
            per_worker[key.worker_id as usize].push(key.sequence_hash);
        }
    }

    let consensus = find_common_prefix(&per_worker);
    if consensus.len() < per_worker.iter().map(|r| r.len()).max().unwrap_or(0) {
        tracing::warn!(
            world_size,
            consensus = consensus.len(),
            per_worker = ?per_worker.iter().map(|r| r.len()).collect::<Vec<_>>(),
            "TP consensus reduced - partial data across workers"
        );
    }

    consensus
}

/// Blocking version of `match_prefix_tp`.
pub fn match_prefix_tp_blocking(
    handle: &PositionalRemoteHandle,
    hashes: &[SequenceHash],
    world_size: usize,
) -> Vec<SequenceHash> {
    if hashes.is_empty() || world_size == 0 {
        return vec![];
    }

    if world_size == 1 {
        return handle.lookup_hashes_blocking(hashes, 0);
    }

    let all_keys: Vec<_> = (0..world_size)
        .flat_map(|wid| {
            hashes.iter().enumerate().map(move |(pos, &hash)| {
                crate::block_manager::distributed::registry::PositionalKey {
                    worker_id: wid as u64,
                    sequence_hash: hash,
                    position: pos as u32,
                }
            })
        })
        .collect();

    let handle_clone = handle.clone();
    let matches = handle.block_on(async move { handle_clone.match_prefix(all_keys).await });
    let mut per_worker: Vec<Vec<SequenceHash>> = vec![vec![]; world_size];
    for (key, _, _) in &matches {
        if (key.worker_id as usize) < world_size {
            per_worker[key.worker_id as usize].push(key.sequence_hash);
        }
    }

    let consensus = find_common_prefix(&per_worker);
    if consensus.len() < per_worker.iter().map(|r| r.len()).max().unwrap_or(0) {
        tracing::warn!(
            world_size,
            consensus = consensus.len(),
            per_worker = ?per_worker.iter().map(|r| r.len()).collect::<Vec<_>>(),
            "TP consensus reduced - partial data across workers"
        );
    }

    consensus
}

/// Register hashes for ALL TP workers.
pub async fn register_tp(
    handle: &PositionalRemoteHandle,
    hashes_with_positions: &[(SequenceHash, u32)],
    storage_config: &RemoteStorageConfig,
    world_size: usize,
) {
    if hashes_with_positions.is_empty() || world_size == 0 {
        return;
    }

    tracing::debug!(
        entries = hashes_with_positions.len(),
        world_size,
        "register_tp"
    );

    for worker_id in 0..world_size {
        let entries = build_entries(hashes_with_positions, storage_config, worker_id, world_size);
        handle.register_hashes(&entries, worker_id as u64).await;
        tracing::trace!(worker_id, entries = entries.len(), "register_tp done");
    }
}

fn build_entries(
    hashes_with_positions: &[(SequenceHash, u32)],
    storage_config: &RemoteStorageConfig,
    worker_id: usize,
    world_size: usize,
) -> Vec<(SequenceHash, u32, RemoteKey)> {
    match storage_config {
        RemoteStorageConfig::Object { default_bucket, .. } => {
            let template = default_bucket.as_deref().unwrap_or("dynamo-kv-cache");
            let bucket = template.replace("{worker_id}", &worker_id.to_string());
            hashes_with_positions
                .iter()
                .map(|&(hash, pos)| {
                    let key = RemoteKey::Object(ObjectKey {
                        bucket: bucket.clone(),
                        key: format!("{:016x}", hash),
                    });
                    (hash, pos, key)
                })
                .collect()
        }
        RemoteStorageConfig::Disk { base_path, .. } => {
            let path = base_path.replace("{worker_id}", &worker_id.to_string());
            hashes_with_positions
                .iter()
                .map(|&(hash, pos)| {
                    let key = RemoteKey::Disk(DiskKey {
                        path: path.clone(),
                        key: format!("{:016x}_{}_{}", hash, worker_id, world_size),
                    });
                    (hash, pos, key)
                })
                .collect()
        }
    }
}

fn find_common_prefix(results: &[Vec<SequenceHash>]) -> Vec<SequenceHash> {
    if results.is_empty() {
        return vec![];
    }

    let min_len = results.iter().map(|r| r.len()).min().unwrap_or(0);
    if min_len == 0 {
        return vec![];
    }

    let reference = &results[0];
    let mut common_len = 0;

    for i in 0..min_len {
        let ref_hash = reference[i];
        if results.iter().skip(1).all(|r| r[i] == ref_hash) {
            common_len = i + 1;
        } else {
            break;
        }
    }

    reference[..common_len].to_vec()
}
