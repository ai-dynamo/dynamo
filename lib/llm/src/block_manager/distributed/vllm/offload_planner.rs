// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use crate::block_manager::block::transfer::remote::{RemoteBlockDescriptor, RemoteTransferPipeline};
use crate::block_manager::config::RemoteStorageConfig;
use crate::block_manager::distributed::remote::PositionalRemoteHandle;
use crate::block_manager::distributed::RemoteHashOperations;
use crate::tokens::SequenceHash;

/// Create remote block descriptors from hashes and storage config.
pub fn create_descriptors(
    hashes: &[SequenceHash],
    storage_config: &RemoteStorageConfig,
    block_size: usize,
    worker_id: usize,
    world_size: usize,
) -> Vec<RemoteBlockDescriptor> {
    match storage_config {
        RemoteStorageConfig::Object { default_bucket, .. } => {
            let bucket = default_bucket.as_deref().unwrap_or("dynamo-kv-cache");
            hashes
                .iter()
                .map(|&hash| RemoteBlockDescriptor::object_from_hash(bucket, hash, block_size))
                .collect()
        }
        RemoteStorageConfig::Disk { base_path, .. } => hashes
            .iter()
            .map(|&hash| {
                RemoteBlockDescriptor::disk_from_hash(
                    base_path, hash, block_size, worker_id, world_size,
                )
            })
            .collect(),
    }
}

/// Filter hashes for offload based on registry response.
pub fn filter_offload_hashes(
    sequence_hashes: &[SequenceHash],
    can_offload_hashes: &[SequenceHash],
    already_stored: &[SequenceHash],
    host_block_ids: Option<&[usize]>,
) -> Option<(Vec<(SequenceHash, u32)>, Option<Vec<usize>>)> {
    use std::collections::HashSet;

    if can_offload_hashes.is_empty() {
        return None;
    }

    let can_offload_set: HashSet<SequenceHash> = can_offload_hashes.iter().copied().collect();
    let hashes_with_positions: Vec<(SequenceHash, u32)> = sequence_hashes
        .iter()
        .enumerate()
        .filter(|(_, hash)| can_offload_set.contains(hash))
        .map(|(pos, &hash)| (hash, pos as u32))
        .collect();

    let filtered_host_ids = host_block_ids.map(|ids| {
        if already_stored.is_empty() {
            ids.to_vec()
        } else {
            let stored_set: HashSet<SequenceHash> = already_stored.iter().copied().collect();
            sequence_hashes
                .iter()
                .zip(ids.iter())
                .filter(|(hash, _)| !stored_set.contains(hash))
                .map(|(_, &id)| id)
                .collect()
        }
    });

    Some((hashes_with_positions, filtered_host_ids))
}

/// Determine if H2O (Host-to-Object) transfer should be triggered.
pub fn should_trigger_h2o(
    is_host_transfer: bool,
    remote_registry_enabled: bool,
    current_h2o_count: usize,
    max_concurrent_h2o: usize,
) -> bool {
    is_host_transfer && remote_registry_enabled && current_h2o_count < max_concurrent_h2o
}

/// Filter hashes for offload by querying the registry.
pub async fn filter_for_offload(
    handle: &PositionalRemoteHandle,
    sequence_hashes: &[SequenceHash],
    host_block_ids: Option<&[usize]>,
    worker_id: u64,
    is_onboard: bool,
) -> Option<(Vec<(SequenceHash, u32)>, Option<Vec<usize>>)> {
    if is_onboard {
        let hashes_with_positions: Vec<(SequenceHash, u32)> = sequence_hashes
            .iter()
            .enumerate()
            .map(|(pos, &hash)| (hash, pos as u32))
            .collect();
        return Some((hashes_with_positions, None));
    }

    let (can_offload_hashes, already_stored, _leased) =
        handle.can_offload_hashes(sequence_hashes, worker_id).await;

    tracing::debug!(
        can_offload = can_offload_hashes.len(),
        already_stored = already_stored.len(),
        "can_offload: {} blocks, {} already stored",
        can_offload_hashes.len(),
        already_stored.len()
    );

    filter_offload_hashes(
        sequence_hashes,
        &can_offload_hashes,
        &already_stored,
        host_block_ids,
    )
}

/// Create a remote transfer pipeline for onboard or offload.
pub fn create_pipeline(
    descriptors: Vec<RemoteBlockDescriptor>,
    is_onboard: bool,
    is_h2o: bool,
    bounce_block_ids: Vec<usize>,
    device_block_ids: Vec<usize>,
) -> RemoteTransferPipeline {
    if is_h2o {
        RemoteTransferPipeline::offload_with_bounce(descriptors, bounce_block_ids, vec![])
    } else if is_onboard {
        RemoteTransferPipeline::onboard_with_bounce(descriptors, bounce_block_ids, device_block_ids)
    } else {
        RemoteTransferPipeline::offload_with_bounce(descriptors, bounce_block_ids, device_block_ids)
    }
}

/// Create a transfer pipeline from hashes and storage config.
pub fn create_transfer_pipeline(
    hashes: &[SequenceHash],
    storage_config: &RemoteStorageConfig,
    block_size: usize,
    worker_id: usize,
    world_size: usize,
    is_onboard: bool,
    is_h2o: bool,
    bounce_block_ids: Vec<usize>,
    device_block_ids: Vec<usize>,
) -> RemoteTransferPipeline {
    let descriptors = create_descriptors(hashes, storage_config, block_size, worker_id, world_size);
    create_pipeline(
        descriptors,
        is_onboard,
        is_h2o,
        bounce_block_ids,
        device_block_ids,
    )
}
