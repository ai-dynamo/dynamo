// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! vLLM integration for distributed block management.
//!
//! Provides business logic for:
//! - **TP-aware registry**: Consensus lookup and multi-rank registration
//! - **Offload orchestration**: D2H, D2D, H2O transfer flows
//! - **Transfer pipelines**: Descriptor and pipeline construction

use crate::block_manager::block::transfer::remote::{
    ObjectKey, RemoteBlockDescriptor, RemoteKey, RemoteTransferPipeline,
};
use crate::block_manager::config::RemoteStorageConfig;
use crate::block_manager::connector::protocol::{RequestType, TransferType, WorkerTransferRequest};
use crate::block_manager::distributed::remote::PositionalRemoteHandle;
use crate::block_manager::distributed::{RemoteHashOperations, RemoteHashOperationsSync};
use crate::tokens::SequenceHash;

use std::sync::OnceLock;

/// Whether G4 checksum validation is enabled (from env var).
static G4_CHECKSUM_ENABLED_CELL: OnceLock<bool> = OnceLock::new();

/// Check if G4 checksum validation is enabled.
pub fn g4_checksum_enabled() -> bool {
    *G4_CHECKSUM_ENABLED_CELL.get_or_init(|| {
        std::env::var("DYN_KVBM_G4_CHECKSUM_VALIDATION")
            .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
            .unwrap_or(false)
    })
}

/// Lookup hashes with TP consensus.
///
/// Queries registry for all worker_ids (0..world_size) and returns the
/// common prefix - hashes that ALL workers have.
///
/// For TP=1, this is equivalent to a single worker lookup.
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

    // Query each worker
    let mut all_results: Vec<Vec<SequenceHash>> = Vec::with_capacity(world_size);
    for worker_id in 0..world_size {
        let matched = handle.lookup_hashes(hashes, worker_id as u64).await;
        tracing::trace!(worker_id, matched = matched.len(), "match_prefix_tp");
        all_results.push(matched);
    }

    let consensus = find_common_prefix(&all_results);

    if consensus.len() < all_results.iter().map(|r| r.len()).max().unwrap_or(0) {
        tracing::warn!(
            world_size,
            consensus = consensus.len(),
            per_worker = ?all_results.iter().map(|r| r.len()).collect::<Vec<_>>(),
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

    let mut all_results: Vec<Vec<SequenceHash>> = Vec::with_capacity(world_size);
    for worker_id in 0..world_size {
        let matched = handle.lookup_hashes_blocking(hashes, worker_id as u64);
        tracing::trace!(
            worker_id,
            matched = matched.len(),
            "match_prefix_tp_blocking"
        );
        all_results.push(matched);
    }

    let consensus = find_common_prefix(&all_results);

    if consensus.len() < all_results.iter().map(|r| r.len()).max().unwrap_or(0) {
        tracing::warn!(
            world_size,
            consensus = consensus.len(),
            per_worker = ?all_results.iter().map(|r| r.len()).collect::<Vec<_>>(),
            "TP consensus reduced - partial data across workers"
        );
    }

    consensus
}

/// Register hashes for ALL TP workers.
///
/// Each worker has its own bucket (via `{worker_id}` template substitution).
/// This ensures queries for any worker_id can discover the offloaded data.
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
        let entries = build_entries(hashes_with_positions, storage_config, worker_id);
        handle.register_hashes(&entries, worker_id as u64).await;
        tracing::trace!(worker_id, entries = entries.len(), "register_tp done");
    }
}

/// Build registry entries for a specific worker_id.
fn build_entries(
    hashes_with_positions: &[(SequenceHash, u32)],
    storage_config: &RemoteStorageConfig,
    worker_id: usize,
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
    }
}

/// Create remote block descriptors from hashes and storage config.
///
/// Maps sequence hashes to `RemoteBlockDescriptor` based on the storage type.
pub fn create_descriptors(
    hashes: &[SequenceHash],
    storage_config: &RemoteStorageConfig,
    block_size: usize,
) -> Vec<RemoteBlockDescriptor> {
    match storage_config {
        RemoteStorageConfig::Object { default_bucket, .. } => {
            let bucket = default_bucket.as_deref().unwrap_or("dynamo-kv-cache");
            hashes
                .iter()
                .map(|&hash| RemoteBlockDescriptor::object_from_hash(bucket, hash, block_size))
                .collect()
        }
    }
}

/// Filter hashes for offload based on registry response.
///
/// Takes the original sequence hashes, the `can_offload` response, and optional host block IDs.
/// Returns `(hashes_with_positions, filtered_host_ids)` preserving original positions.
///
/// Returns `None` if nothing can be offloaded (all already stored).
#[allow(clippy::type_complexity)]
pub fn filter_offload_hashes(
    sequence_hashes: &[SequenceHash],
    can_offload_hashes: &[SequenceHash],
    already_stored: &[SequenceHash],
    host_block_ids: Option<&[usize]>,
) -> Option<(Vec<(SequenceHash, u32)>, Option<Vec<usize>>)> {
    use std::collections::HashSet;

    // Nothing to offload
    if can_offload_hashes.is_empty() {
        return None;
    }

    // Build set for O(1) lookup
    let can_offload_set: HashSet<SequenceHash> = can_offload_hashes.iter().copied().collect();

    // Preserve original positions by filtering from original sequence
    let hashes_with_positions: Vec<(SequenceHash, u32)> = sequence_hashes
        .iter()
        .enumerate()
        .filter(|(_, hash)| can_offload_set.contains(hash))
        .map(|(pos, &hash)| (hash, pos as u32))
        .collect();

    // Filter host block IDs (for H2O transfers)
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
///
/// H2O is triggered when:
/// - Transfer is to host pool (D2H completed)
/// - Remote registry is enabled
/// - Current H2O count is below max concurrent limit
pub fn should_trigger_h2o(
    is_host_transfer: bool,
    remote_registry_enabled: bool,
    current_h2o_count: usize,
    max_concurrent_h2o: usize,
) -> bool {
    is_host_transfer && remote_registry_enabled && current_h2o_count < max_concurrent_h2o
}

/// Filter hashes for offload by querying the registry.
///
/// For onboard: returns all hashes with positions (no filtering needed).
/// For offload: queries `can_offload_hashes` and filters based on what's already stored.
///
/// Returns `None` if nothing can be transferred.
pub async fn filter_for_offload(
    handle: &PositionalRemoteHandle,
    sequence_hashes: &[SequenceHash],
    host_block_ids: Option<&[usize]>,
    worker_id: u64,
    is_onboard: bool,
) -> Option<(Vec<(SequenceHash, u32)>, Option<Vec<usize>>)> {
    if is_onboard {
        // For onboard, preserve original positions (no registry query needed)
        let hashes_with_positions: Vec<(SequenceHash, u32)> = sequence_hashes
            .iter()
            .enumerate()
            .map(|(pos, &hash)| (hash, pos as u32))
            .collect();
        return Some((hashes_with_positions, None));
    }

    // Query registry to find blocks we can offload
    let (can_offload_hashes, already_stored, _leased) =
        handle.can_offload_hashes(sequence_hashes, worker_id).await;

    tracing::debug!(
        can_offload = can_offload_hashes.len(),
        already_stored = already_stored.len(),
        "can_offload: {} blocks, {} already stored",
        can_offload_hashes.len(),
        already_stored.len()
    );

    // Use the pure filtering logic
    filter_offload_hashes(
        sequence_hashes,
        &can_offload_hashes,
        &already_stored,
        host_block_ids,
    )
}

/// Create a remote transfer pipeline for onboard or offload.
///
/// For H2O (host-to-object): uses provided host_block_ids as bounce buffers.
/// For onboard/offload: uses provided bounce_block_ids and device_block_ids.
pub fn create_pipeline(
    descriptors: Vec<RemoteBlockDescriptor>,
    is_onboard: bool,
    is_h2o: bool,
    bounce_block_ids: Vec<usize>,
    device_block_ids: Vec<usize>,
) -> RemoteTransferPipeline {
    if is_h2o {
        // H2O: host blocks are the bounce buffers, no device blocks
        RemoteTransferPipeline::offload_with_bounce(descriptors, bounce_block_ids, vec![])
    } else if is_onboard {
        RemoteTransferPipeline::onboard_with_bounce(descriptors, bounce_block_ids, device_block_ids)
    } else {
        // Invariant: callers use from_g4_params (is_onboard=true) or new_h2o (is_h2o=true),
        // so this branch where both are false should never be reached.
        unreachable!(
            "create_pipeline called with is_onboard=false and is_h2o=false; \
             use from_g4_params for onboard or new_h2o for host-to-object transfers"
        )
    }
}

/// Create a transfer pipeline from hashes and storage config.
///
/// Combines descriptor creation with pipeline construction.
/// Caller provides pre-allocated bounce buffer IDs.
pub fn create_transfer_pipeline(
    hashes: &[SequenceHash],
    storage_config: &RemoteStorageConfig,
    block_size: usize,
    is_onboard: bool,
    is_h2o: bool,
    bounce_block_ids: Vec<usize>,
    device_block_ids: Vec<usize>,
) -> RemoteTransferPipeline {
    let descriptors = create_descriptors(hashes, storage_config, block_size);
    create_pipeline(
        descriptors,
        is_onboard,
        is_h2o,
        bounce_block_ids,
        device_block_ids,
    )
}

/// Parameters for G4 onboard operation.
#[derive(Debug, Clone)]
pub struct G4OnboardParams {
    pub request_id: String,
    pub sequence_hashes: Vec<u64>,
    pub device_block_ids: Vec<usize>,
    pub operation_id: uuid::Uuid,
    pub block_size: usize,
}

/// Prepare G4 onboard operation.
///
/// Creates the operation parameters and worker request. The caller is responsible for:
/// - Creating the local transfer request from params
/// - Sending via the transfer channel
/// - Appending worker_req to pending operations
///
/// The `operation_id` must match the one used in the G4 state machine transition
/// to ensure response correlation.
///
/// Returns (G4OnboardParams, WorkerTransferRequest).
pub fn onboard_from_g4(
    request_id: String,
    sequence_hashes: Vec<u64>,
    device_block_ids: Vec<usize>,
    block_size: usize,
    operation_id: uuid::Uuid,
) -> (G4OnboardParams, WorkerTransferRequest) {
    let num_blocks = sequence_hashes.len();

    tracing::debug!(
        target: "kvbm-g4",
        request_id = %request_id,
        operation_id = %operation_id,
        num_blocks = num_blocks,
        "preparing onboard for {} blocks",
        num_blocks
    );

    let params = G4OnboardParams {
        request_id: request_id.clone(),
        sequence_hashes,
        device_block_ids: device_block_ids.clone(),
        operation_id,
        block_size,
    };

    let worker_req = WorkerTransferRequest {
        request_id,
        uuid: operation_id,
        transfer_type: TransferType::Load,
        request_type: RequestType::Immediate,
        block_ids: device_block_ids,
    };

    (params, worker_req)
}

// Re-export core checksum functions from the shared module
pub use crate::block_manager::block::transfer::checksum::{
    BlockChecksum, ChecksumBuilder, compute_checksum, compute_checksum_raw, verify_checksum,
};

/// G4 transfer checksum record for validation.
#[derive(Debug, Clone)]
pub struct G4ChecksumRecord {
    /// Sequence hash (block identifier)
    pub sequence_hash: u64,
    /// Blake3 checksum of the block data (hex string)
    pub checksum: BlockChecksum,
    /// Block size in bytes
    pub block_size: usize,
}

impl G4ChecksumRecord {
    pub fn new(sequence_hash: u64, checksum: BlockChecksum, block_size: usize) -> Self {
        Self {
            sequence_hash,
            checksum,
            block_size,
        }
    }
}

/// Compute checksums for multiple blocks in bounce buffers.
///
/// # Arguments
/// * `bounce_ptrs` - Pointers to bounce buffer blocks
/// * `block_size` - Size of each block in bytes
/// * `sequence_hashes` - Corresponding sequence hashes
///
/// # Safety
/// Each pointer must be valid for `block_size` bytes.
pub unsafe fn compute_g4_checksums(
    bounce_ptrs: &[*const u8],
    block_size: usize,
    sequence_hashes: &[u64],
) -> Vec<G4ChecksumRecord> {
    assert_eq!(bounce_ptrs.len(), sequence_hashes.len());

    bounce_ptrs
        .iter()
        .zip(sequence_hashes.iter())
        .map(|(&ptr, &hash)| {
            let checksum = unsafe { compute_checksum_raw(ptr, block_size) };
            G4ChecksumRecord::new(hash, checksum, block_size)
        })
        .collect()
}

/// Verify checksums for blocks after G4 read.
///
/// # Arguments
/// * `bounce_ptrs` - Pointers to bounce buffer blocks with received data
/// * `block_size` - Size of each block in bytes
/// * `expected` - Expected checksum records
///
/// # Returns
/// `Ok(())` if all checksums match, `Err` with details of first mismatch.
///
/// # Safety
/// Each pointer must be valid for `block_size` bytes.
pub unsafe fn verify_g4_checksums(
    bounce_ptrs: &[*const u8],
    block_size: usize,
    expected: &[G4ChecksumRecord],
) -> Result<(), G4ChecksumError> {
    if bounce_ptrs.len() != expected.len() {
        return Err(G4ChecksumError::CountMismatch {
            actual: bounce_ptrs.len(),
            expected: expected.len(),
        });
    }

    for (idx, (&ptr, record)) in bounce_ptrs.iter().zip(expected.iter()).enumerate() {
        let actual = unsafe { compute_checksum_raw(ptr, block_size) };
        if actual != record.checksum {
            return Err(G4ChecksumError::Mismatch {
                block_index: idx,
                sequence_hash: record.sequence_hash,
                expected: record.checksum.clone(),
                actual,
            });
        }
    }

    Ok(())
}

/// Error type for G4 checksum validation failures.
#[derive(Debug, Clone)]
pub enum G4ChecksumError {
    /// Number of blocks doesn't match expected
    CountMismatch { actual: usize, expected: usize },
    /// Checksum mismatch for a specific block
    Mismatch {
        block_index: usize,
        sequence_hash: u64,
        expected: String,
        actual: String,
    },
}

impl std::fmt::Display for G4ChecksumError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            G4ChecksumError::CountMismatch { actual, expected } => {
                write!(
                    f,
                    "Block count mismatch: got {}, expected {}",
                    actual, expected
                )
            }
            G4ChecksumError::Mismatch {
                block_index,
                sequence_hash,
                expected,
                actual,
            } => {
                write!(
                    f,
                    "Checksum mismatch at block {} (hash {:016x}): expected {}, got {}",
                    block_index, sequence_hash, expected, actual
                )
            }
        }
    }
}

impl std::error::Error for G4ChecksumError {}

/// Find the common prefix across multiple result sets.
///
/// Returns the longest prefix where ALL result sets have the same hashes.
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_find_common_prefix_all_match() {
        let results: Vec<Vec<SequenceHash>> =
            vec![vec![1, 2, 3, 4], vec![1, 2, 3, 4], vec![1, 2, 3, 4]];
        assert_eq!(find_common_prefix(&results), vec![1u64, 2, 3, 4]);
    }

    #[test]
    fn test_find_common_prefix_partial() {
        let results: Vec<Vec<SequenceHash>> = vec![vec![1, 2, 3, 4], vec![1, 2, 3], vec![1, 2]];
        assert_eq!(find_common_prefix(&results), vec![1u64, 2]);
    }

    #[test]
    fn test_find_common_prefix_mismatch() {
        let results: Vec<Vec<SequenceHash>> = vec![
            vec![1, 2, 3],
            vec![1, 9, 3], // different at position 1
        ];
        assert_eq!(find_common_prefix(&results), vec![1u64]);
    }

    #[test]
    fn test_find_common_prefix_empty() {
        let results: Vec<Vec<SequenceHash>> = vec![vec![], vec![1, 2]];
        let expected: Vec<SequenceHash> = vec![];
        assert_eq!(find_common_prefix(&results), expected);
    }

    #[test]
    fn test_find_common_prefix_no_results() {
        let results: Vec<Vec<SequenceHash>> = vec![];
        let expected: Vec<SequenceHash> = vec![];
        assert_eq!(find_common_prefix(&results), expected);
    }

    #[test]
    fn test_filter_offload_hashes_all_can_offload() {
        let sequence: Vec<SequenceHash> = vec![1, 2, 3, 4];
        let can_offload: Vec<SequenceHash> = vec![1, 2, 3, 4];
        let already_stored: Vec<SequenceHash> = vec![];

        let result = filter_offload_hashes(&sequence, &can_offload, &already_stored, None);
        assert!(result.is_some());

        let (hashes_with_pos, host_ids) = result.unwrap();
        assert_eq!(hashes_with_pos, vec![(1, 0), (2, 1), (3, 2), (4, 3)]);
        assert!(host_ids.is_none());
    }

    #[test]
    fn test_filter_offload_hashes_partial() {
        let sequence: Vec<SequenceHash> = vec![1, 2, 3, 4];
        let can_offload: Vec<SequenceHash> = vec![1, 3]; // 2 and 4 already stored
        let already_stored: Vec<SequenceHash> = vec![2, 4];

        let result = filter_offload_hashes(&sequence, &can_offload, &already_stored, None);
        assert!(result.is_some());

        let (hashes_with_pos, _) = result.unwrap();
        // Positions preserved: 1 at pos 0, 3 at pos 2
        assert_eq!(hashes_with_pos, vec![(1, 0), (3, 2)]);
    }

    #[test]
    fn test_filter_offload_hashes_with_host_ids() {
        let sequence: Vec<SequenceHash> = vec![1, 2, 3, 4];
        let can_offload: Vec<SequenceHash> = vec![1, 3];
        let already_stored: Vec<SequenceHash> = vec![2, 4];
        let host_ids: Vec<usize> = vec![10, 20, 30, 40];

        let result =
            filter_offload_hashes(&sequence, &can_offload, &already_stored, Some(&host_ids));
        assert!(result.is_some());

        let (_, filtered_host_ids) = result.unwrap();
        // Host IDs for hashes 1 and 3 (not in already_stored)
        assert_eq!(filtered_host_ids, Some(vec![10, 30]));
    }

    #[test]
    fn test_filter_offload_hashes_all_stored() {
        let sequence: Vec<SequenceHash> = vec![1, 2];
        let can_offload: Vec<SequenceHash> = vec![]; // Nothing to offload
        let already_stored: Vec<SequenceHash> = vec![1, 2];

        let result = filter_offload_hashes(&sequence, &can_offload, &already_stored, None);
        assert!(result.is_none());
    }

    #[test]
    fn test_should_trigger_h2o() {
        // All conditions met
        assert!(should_trigger_h2o(true, true, 0, 8));
        assert!(should_trigger_h2o(true, true, 7, 8));

        // At max concurrent
        assert!(!should_trigger_h2o(true, true, 8, 8));
        assert!(!should_trigger_h2o(true, true, 10, 8));

        // Not host transfer
        assert!(!should_trigger_h2o(false, true, 0, 8));

        // Registry disabled
        assert!(!should_trigger_h2o(true, false, 0, 8));
    }

    #[test]
    fn test_create_descriptors_object() {
        use crate::block_manager::block::transfer::remote::RemoteStorageKind;

        let hashes: Vec<SequenceHash> = vec![0x1234, 0x5678];
        let config = RemoteStorageConfig::object("my-bucket");
        let descriptors = create_descriptors(&hashes, &config, 4096);

        assert_eq!(descriptors.len(), 2);
        assert_eq!(descriptors[0].kind(), RemoteStorageKind::Object);
        assert_eq!(descriptors[0].sequence_hash(), Some(0x1234));
        assert_eq!(descriptors[1].sequence_hash(), Some(0x5678));
    }
}
