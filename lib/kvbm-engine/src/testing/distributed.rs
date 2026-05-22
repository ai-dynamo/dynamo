// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Distributed leader testing utilities.
//!
//! This module provides test infrastructure for:
//! - Single-leader tests with `TestInstanceLeader` and `InstanceLeaderPair`
//! - Multi-worker RDMA tests with `TestWorker` and `TestInstanceLeaderWithWorkers`

use anyhow::Result;
use std::collections::HashMap;
use std::sync::Arc;

use crate::{
    BlockId, G2, G3, InstanceId, SequenceHash,
    leader::InstanceLeader,
    worker::{DirectWorker, Worker},
};
use kvbm_logical::manager::BlockManager;
use kvbm_physical::manager::{LayoutHandle, TransferManager};
use kvbm_physical::transfer::StorageKind;
use kvbm_physical::{
    layout::LayoutConfig,
    transfer::{BlockChecksum, FillPattern},
};

use super::{managers, messenger, physical, token_blocks};

/// Number of layers for layerwise transfer tests.
pub const DEFAULT_NUM_LAYERS: usize = 3;

/// Container for a test InstanceLeader with its managers.
pub struct TestInstanceLeader {
    pub instance_id: InstanceId,
    pub leader: InstanceLeader,
    pub g2_manager: Arc<BlockManager<G2>>,
    pub g3_manager: Option<Arc<BlockManager<G3>>>,
}

/// Container for a pair of connected InstanceLeaders.
pub struct InstanceLeaderPair {
    pub leader_a: TestInstanceLeader,
    pub leader_b: TestInstanceLeader,
}

/// Create a pair of InstanceLeaders connected via Messenger for integration testing.
///
/// Setup:
/// - Two Messenger instances with TCP transport
/// - Bidirectional peer registration
/// - G2 BlockManagers for each leader
/// - Handlers registered for distributed communication
///
/// # Arguments
/// * `block_count` - Number of blocks in each G2 manager
/// * `block_size` - Tokens per block
///
/// # Returns
/// InstanceLeaderPair with both leaders ready for testing
///
/// # Example
/// ```ignore
/// let pair = create_instance_leader_pair(100, 16).await?;
///
/// // Populate leader A with blocks
/// let (_, hashes) = populate_leader_with_blocks(&pair.leader_a, 32, 16, 0)?;
///
/// // Leader B can search leader A
/// let result = pair.leader_b.leader.find_matches(&hashes)?;
/// ```
pub async fn create_instance_leader_pair(
    block_count: usize,
    block_size: usize,
) -> Result<InstanceLeaderPair> {
    // Create Messenger pair
    let messenger::MessengerPair {
        messenger_a,
        messenger_b,
    } = messenger::create_messenger_pair_tcp().await?;

    // Create G2 managers
    let registry_a = managers::TestRegistryBuilder::new().build();
    let registry_b = managers::TestRegistryBuilder::new().build();

    let g2_manager_a = Arc::new(
        managers::TestManagerBuilder::<G2>::new()
            .block_count(block_count)
            .block_size(block_size)
            .registry(registry_a.clone())
            .build(),
    );
    let g3_manager_a = Arc::new(
        managers::TestManagerBuilder::<G3>::new()
            .block_count(block_count)
            .block_size(block_size)
            .registry(registry_a.clone())
            .build(),
    );

    let g2_manager_b = Arc::new(
        managers::TestManagerBuilder::<G2>::new()
            .block_count(block_count)
            .block_size(block_size)
            .registry(registry_b.clone())
            .build(),
    );
    let g3_manager_b = Arc::new(
        managers::TestManagerBuilder::<G3>::new()
            .block_count(block_count)
            .block_size(block_size)
            .registry(registry_b.clone())
            .build(),
    );

    // Build InstanceLeader A
    let leader_a = InstanceLeader::builder()
        .messenger(messenger_a.clone())
        .registry(registry_a.clone())
        .g2_manager(g2_manager_a.clone())
        .g3_manager(g3_manager_a.clone())
        .workers(vec![]) // No workers for now (no transfers)
        .build()?;

    // Register handlers for A
    leader_a.register_handlers()?;

    // Build InstanceLeader B
    let leader_b = InstanceLeader::builder()
        .messenger(messenger_b.clone())
        .registry(registry_b.clone())
        .g2_manager(g2_manager_b.clone())
        .g3_manager(g3_manager_b.clone())
        .workers(vec![]) // No workers for now
        .build()?;

    // Register handlers for B
    leader_b.register_handlers()?;

    Ok(InstanceLeaderPair {
        leader_a: TestInstanceLeader {
            instance_id: messenger_a.instance_id(),
            leader: leader_a,
            g2_manager: g2_manager_a,
            g3_manager: Some(g3_manager_a),
        },
        leader_b: TestInstanceLeader {
            instance_id: messenger_b.instance_id(),
            leader: leader_b,
            g2_manager: g2_manager_b,
            g3_manager: Some(g3_manager_b),
        },
    })
}

/// Populate a leader's G2 manager with token blocks.
///
/// # Arguments
/// * `leader` - The test leader instance
/// * `num_blocks` - Number of blocks to create
/// * `block_size` - Tokens per block
/// * `start_token` - Starting token value
///
/// # Returns
/// (BlockManager, Vec<SequenceHash>) - Manager and sequence hashes of populated blocks
///
/// # Example
/// ```ignore
/// let pair = create_instance_leader_pair(100, 4).await?;
/// let (manager, hashes) = populate_leader_with_blocks(&pair.leader_a, 32, 4, 0)?;
/// assert_eq!(hashes.len(), 32);
/// ```
pub fn populate_leader_with_blocks(
    leader: &TestInstanceLeader,
    num_blocks: usize,
    block_size: usize,
    start_token: u32,
) -> Result<(Arc<BlockManager<G2>>, Vec<SequenceHash>)> {
    let token_sequence =
        super::token_blocks::create_token_sequence(num_blocks, block_size, start_token);
    let seq_hashes =
        managers::populate_manager_with_blocks(&leader.g2_manager, token_sequence.blocks())?;

    Ok((leader.g2_manager.clone(), seq_hashes))
}

// =============================================================================
// Multi-worker RDMA test infrastructure
// =============================================================================

/// Container for a test worker with its transfer infrastructure.
///
/// This wraps a DirectWorker with access to its TransferManager and registered layouts,
/// enabling fine-grained control over worker-level operations in tests.
pub struct TestWorker {
    /// Unique instance identifier (primary identity).
    pub instance_id: InstanceId,
    /// Unique worker identifier derived from instance_id (used in LayoutHandle encoding).
    pub worker_id: u64,
    /// The DirectWorker instance (implements Worker trait).
    pub worker: Arc<DirectWorker>,
    /// TransferManager owned by this worker (for direct transfer operations).
    pub manager: Arc<TransferManager>,
    /// G2 layout handle registered with this worker.
    pub g2_handle: LayoutHandle,
}

impl TestWorker {
    /// Fill G2 blocks with test data and return checksums.
    ///
    /// This uses the internal registry accessor to fill blocks in the
    /// registered G2 layout. Only works with System or Pinned storage.
    pub fn fill_g2_blocks(
        &self,
        block_ids: &[BlockId],
        pattern: FillPattern,
    ) -> Result<HashMap<BlockId, BlockChecksum>> {
        physical::fill_and_checksum_manager(&self.manager, self.g2_handle, block_ids, pattern)
    }

    /// Compute checksums for G2 blocks (for verification after transfers).
    ///
    /// This uses the internal registry accessor to compute checksums for
    /// blocks in the registered G2 layout.
    pub fn compute_g2_checksums(
        &self,
        block_ids: &[BlockId],
    ) -> Result<HashMap<BlockId, BlockChecksum>> {
        physical::compute_manager_checksums(&self.manager, self.g2_handle, block_ids)
    }
}

/// Container for a test InstanceLeader with accessible workers.
///
/// This extends TestInstanceLeader with actual DirectWorker instances,
/// allowing tests to access both the leader-level APIs and the underlying
/// worker infrastructure for RDMA operations.
pub struct TestInstanceLeaderWithWorkers {
    /// Instance identifier.
    pub instance_id: InstanceId,
    /// The InstanceLeader.
    pub leader: InstanceLeader,
    /// G2 BlockManager for logical block management.
    pub g2_manager: Arc<BlockManager<G2>>,
    /// G3 BlockManager for disk-backed blocks.
    pub g3_manager: Option<Arc<BlockManager<G3>>>,
    /// Workers with their transfer infrastructure.
    pub workers: Vec<TestWorker>,
}

impl TestInstanceLeaderWithWorkers {
    /// Get the G2 layout handle (from first worker).
    ///
    /// Returns `None` if there are no workers.
    pub fn g2_layout_handle(&self) -> Option<LayoutHandle> {
        self.workers.first().map(|w| w.g2_handle)
    }

    /// Populate G2 with blocks and return their sequence hashes.
    ///
    /// This is a convenience method that combines allocation, filling,
    /// and registration into one step.
    pub fn populate_g2_blocks(
        &self,
        num_blocks: usize,
        block_size: usize,
        start_token: u32,
    ) -> Result<(Vec<BlockId>, Vec<SequenceHash>)> {
        let token_sequence =
            token_blocks::create_token_sequence(num_blocks, block_size, start_token);
        let seq_hashes =
            managers::populate_manager_with_blocks(&self.g2_manager, token_sequence.blocks())?;

        // Get the block IDs that were allocated
        let matched = self.g2_manager.match_blocks(&seq_hashes);
        let block_ids: Vec<BlockId> = matched.into_iter().map(|b| b.block_id()).collect();

        Ok((block_ids, seq_hashes))
    }

    /// Fill blocks on all workers with a layer-specific pattern.
    ///
    /// Each layer gets a different fill byte: layer 0 = 0xA0, layer 1 = 0xA1, etc.
    /// This enables verification that the correct layer was transferred.
    pub fn fill_blocks_with_layer_pattern(
        &self,
        block_ids: &[BlockId],
        layer: usize,
    ) -> Result<HashMap<BlockId, BlockChecksum>> {
        let pattern = FillPattern::Constant(0xA0 + layer as u8);
        let mut all_checksums = HashMap::new();

        for worker in &self.workers {
            let checksums = worker.fill_g2_blocks(block_ids, pattern)?;
            all_checksums.extend(checksums);
        }

        Ok(all_checksums)
    }

    /// Verify that blocks have the expected layer pattern.
    ///
    /// Checks that blocks were transferred correctly by verifying
    /// the checksum matches the expected layer pattern.
    pub fn verify_layer_checksums(
        &self,
        block_ids: &[BlockId],
        expected_checksums: &HashMap<BlockId, BlockChecksum>,
    ) -> Result<()> {
        for worker in &self.workers {
            let actual_checksums = worker.compute_g2_checksums(block_ids)?;
            for block_id in block_ids {
                let expected = expected_checksums.get(block_id).ok_or_else(|| {
                    anyhow::anyhow!("Missing expected checksum for block {}", block_id)
                })?;
                let actual = actual_checksums.get(block_id).ok_or_else(|| {
                    anyhow::anyhow!("Missing actual checksum for block {}", block_id)
                })?;
                if expected != actual {
                    anyhow::bail!(
                        "Checksum mismatch for block {}: expected {:?}, got {:?}",
                        block_id,
                        expected,
                        actual
                    );
                }
            }
        }
        Ok(())
    }
}

// Instance Leader Pair with Workers
// =============================================================================

/// Container for a pair of leaders with workers for RDMA testing.
///
/// This is the primary test fixture for prefill-decode RDMA scenarios:
/// - `decode`: The source instance (has data to pull from)
/// - `prefill`: The destination instance (pulls data via RDMA)
pub struct InstanceLeaderPairWithWorkers {
    /// Decode leader (source of RDMA transfers).
    pub decode: TestInstanceLeaderWithWorkers,
    /// Prefill leader (destination of RDMA transfers).
    pub prefill: TestInstanceLeaderWithWorkers,
}

/// Create a DirectWorker with UCX backend and registered G2 layout.
///
/// # Arguments
/// * `instance_id` - Unique instance identifier for this worker
/// * `agent_name` - NIXL agent name (must be unique for RDMA addressing)
/// * `layout_config` - Configuration for the G2 physical layout
/// * `storage` - Storage type for the layout (typically Pinned for RDMA)
///
/// # Returns
/// TestWorker with TransferManager and registered G2 layout
///
/// # Worker ID Derivation
/// The worker_id is derived from instance_id using xxh3_64 hash, ensuring
/// unique LayoutHandles (worker_id, layout_id) for each worker.
///
/// # Backend Requirements
/// This function requires UCX backend for RDMA operations. Use
/// `physical::TestAgentBuilder` for more flexible backend handling.
pub fn create_direct_worker(
    instance_id: InstanceId,
    agent_name: &str,
    layout_config: &LayoutConfig,
    storage: StorageKind,
) -> Result<TestWorker> {
    // Derive worker_id from instance_id (deterministic hash)
    let worker_id = instance_id.worker_id().as_u64();

    // Create local EventManager (purely local event system for this worker)
    let event_system = velo::EventManager::local();

    // Create NixlAgent with UCX backend using TestAgentBuilder
    // UCX is required for RDMA operations
    let test_agent = physical::TestAgentBuilder::new(agent_name)
        .require_backend("UCX")
        .build()?;
    let agent = test_agent.into_nixl_agent();

    // Create TransferManager with the event_system
    let manager = TransferManager::builder()
        .event_system(Arc::new(event_system))
        .nixl_agent(agent.clone())
        .cuda_device_id(0)
        .build()?;

    // Create and register G2 physical layout
    // This will create LayoutHandle(worker_id, 0) - now unique per worker!
    let layout = physical::create_fc_layout_with_config(agent, storage, layout_config.clone());
    let g2_handle = manager.register_layout(layout)?;

    // Create DirectWorker with G2 handle via builder
    let direct_worker = DirectWorker::builder()
        .manager(manager.clone())
        .g2_handle(g2_handle)
        .build()?;

    Ok(TestWorker {
        instance_id,
        worker_id,
        worker: Arc::new(direct_worker),
        manager: Arc::new(manager),
        g2_handle,
    })
}

/// Create multiple DirectWorkers for a single leader.
///
/// Each worker gets:
/// - A unique InstanceId (UUID v4)
/// - A unique NixlAgent with UCX backend
/// - Its own TransferManager with unique worker_id
/// - A registered G2 physical layout
///
/// # Arguments
/// * `num_workers` - Number of workers to create
/// * `layout_config` - Configuration for G2 layouts
/// * `storage` - Storage type (typically Pinned for RDMA)
/// * `agent_name_prefix` - Prefix for agent names (e.g., "decode" -> "decode-worker-0")
///
/// # Returns
/// Vector of TestWorkers, one per worker, each with unique InstanceId
pub fn create_direct_workers(
    num_workers: usize,
    layout_config: &LayoutConfig,
    storage: StorageKind,
    agent_name_prefix: &str,
) -> Result<Vec<TestWorker>> {
    let mut workers = Vec::with_capacity(num_workers);
    for i in 0..num_workers {
        // Create unique InstanceId for this worker
        let instance_id = InstanceId::new_v4();
        let agent_name = format!("{}-worker-{}", agent_name_prefix, i);

        let worker = create_direct_worker(instance_id, &agent_name, layout_config, storage)?;
        workers.push(worker);
    }

    Ok(workers)
}

/// Create an InstanceLeader with DirectWorkers for RDMA testing.
///
/// # Arguments
/// * `block_count` - Number of blocks in G2 manager
/// * `block_size` - Tokens per block
/// * `num_workers` - Number of DirectWorkers to create
/// * `layout_config` - Configuration for worker G2 layouts
/// * `storage` - Storage type for layouts
/// * `messenger` - Messenger instance for leader communication
///
/// # Returns
/// TestInstanceLeaderWithWorkers with leader and worker infrastructure
#[allow(clippy::too_many_arguments)]
pub async fn create_instance_leader_with_workers(
    block_count: usize,
    block_size: usize,
    num_workers: usize,
    layout_config: &LayoutConfig,
    storage: StorageKind,
    messenger: Arc<velo::Messenger>,
    agent_name_prefix: &str,
) -> Result<TestInstanceLeaderWithWorkers> {
    // Create G2 and G3 managers
    let registry = managers::TestRegistryBuilder::new().build();
    let g2_manager = Arc::new(
        managers::TestManagerBuilder::<G2>::new()
            .block_count(block_count)
            .block_size(block_size)
            .registry(registry.clone())
            .build(),
    );
    let g3_manager = Arc::new(
        managers::TestManagerBuilder::<G3>::new()
            .block_count(block_count)
            .block_size(block_size)
            .registry(registry.clone())
            .build(),
    );

    // Create DirectWorkers
    let workers = create_direct_workers(num_workers, layout_config, storage, agent_name_prefix)?;

    // Extract worker references for the leader
    let worker_refs: Vec<Arc<dyn Worker>> = workers
        .iter()
        .map(|w| w.worker.clone() as Arc<dyn Worker>)
        .collect();

    // Build InstanceLeader
    let leader = InstanceLeader::builder()
        .messenger(messenger.clone())
        .registry(registry.clone())
        .g2_manager(g2_manager.clone())
        .g3_manager(g3_manager.clone())
        .workers(worker_refs)
        .build()?;

    // Register handlers
    leader.register_handlers()?;

    Ok(TestInstanceLeaderWithWorkers {
        instance_id: messenger.instance_id(),
        leader,
        g2_manager,
        g3_manager: Some(g3_manager),
        workers,
    })
}

/// Create a pair of InstanceLeaders with workers for RDMA integration testing.
///
/// Setup:
/// - Two Messenger instances with TCP transport
/// - Bidirectional peer registration
/// - N DirectWorkers per leader with UCX-registered layouts
/// - G2 BlockManagers for logical block management
///
/// # Arguments
/// * `block_count` - Number of blocks in each G2 manager
/// * `block_size` - Tokens per block
/// * `num_workers` - Number of workers per leader (must match for RDMA)
/// * `layout_config` - Configuration for worker G2 layouts
/// * `storage` - Storage type (typically Pinned for RDMA)
///
/// # Returns
/// InstanceLeaderPairWithWorkers ready for RDMA testing
///
/// # Example
/// ```ignore
/// let layout_config = custom_config(64, 3, 2, 4, 64, 2);
/// let pair = create_instance_leader_pair_with_workers(
///     64, 4, 2, &layout_config, StorageKind::Pinned
/// ).await?;
///
/// // Fill decode workers with data
/// for worker in &pair.decode.workers {
///     fill_and_checksum(&layout, &block_ids, FillPattern::Sequential)?;
/// }
/// ```
pub async fn create_instance_leader_pair_with_workers(
    block_count: usize,
    block_size: usize,
    num_workers: usize,
    layout_config: &LayoutConfig,
    storage: StorageKind,
) -> Result<InstanceLeaderPairWithWorkers> {
    // Create Messenger pair
    let messenger::MessengerPair {
        messenger_a,
        messenger_b,
    } = messenger::create_messenger_pair_tcp().await?;

    // Create Decode leader with workers
    let decode = create_instance_leader_with_workers(
        block_count,
        block_size,
        num_workers,
        layout_config,
        storage,
        messenger_a.clone(),
        "decode",
    )
    .await?;

    // Create Prefill leader with workers
    let prefill = create_instance_leader_with_workers(
        block_count,
        block_size,
        num_workers,
        layout_config,
        storage,
        messenger_b.clone(),
        "prefill",
    )
    .await?;

    Ok(InstanceLeaderPairWithWorkers { decode, prefill })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_create_instance_leader_pair() {
        let pair = create_instance_leader_pair(100, 16)
            .await
            .expect("Should create leader pair");

        // Verify different instance IDs
        assert_ne!(pair.leader_a.instance_id, pair.leader_b.instance_id);

        // Verify managers are configured correctly
        assert_eq!(pair.leader_a.g2_manager.total_blocks(), 100);
        assert_eq!(pair.leader_a.g2_manager.block_size(), 16);
        assert_eq!(pair.leader_b.g2_manager.total_blocks(), 100);
        assert_eq!(pair.leader_b.g2_manager.block_size(), 16);
    }

    #[tokio::test]
    async fn test_populate_leader_with_blocks() {
        let pair = create_instance_leader_pair(50, 4)
            .await
            .expect("Should create pair");

        let (manager, hashes) =
            populate_leader_with_blocks(&pair.leader_a, 10, 4, 0).expect("Should populate");

        assert_eq!(hashes.len(), 10);
        assert_eq!(manager.available_blocks(), 50); // All blocks available (10 in inactive)

        // Verify blocks can be matched
        let matched = manager.match_blocks(&hashes);
        assert_eq!(matched.len(), 10);
    }

    // =========================================================================
    // scan_with_policy Tests
    // =========================================================================

    /// Test simple linear scan policy.
    ///
    /// This tests the basic usage of scan_with_policy where the policy
    /// iterates through all hashes and yields each found block.
    #[tokio::test]
    async fn test_scan_with_policy_linear_scan() {
        use crate::leader::TieredBlock;

        let pair = create_instance_leader_pair(50, 4)
            .await
            .expect("Should create pair");

        // Populate with blocks
        let (_, hashes) =
            populate_leader_with_blocks(&pair.leader_a, 10, 4, 0).expect("Should populate");

        // Simple linear scan policy: find all blocks
        let blocks: Vec<TieredBlock> =
            pair.leader_a
                .leader
                .scan_with_policy(&hashes, true, |hashes, ctx| {
                    for hash in hashes {
                        if let Some(block) = ctx.accessor().find(*hash) {
                            ctx.yield_item(block);
                        }
                    }
                });

        // Should find all 10 blocks
        assert_eq!(blocks.len(), 10);

        // Verify all are G2 blocks (since we populated G2)
        for block in &blocks {
            assert!(block.is_g2());
        }

        // Verify positions are sequential (0-9)
        for (i, block) in blocks.iter().enumerate() {
            assert_eq!(block.position(), i as u64);
        }
    }

    /// Test scan_with_policy with partial matches.
    ///
    /// Tests that the policy correctly handles cases where some hashes
    /// are not found in the manager.
    #[tokio::test]
    async fn test_scan_with_policy_partial_matches() {
        use crate::leader::TieredBlock;

        let pair = create_instance_leader_pair(50, 4)
            .await
            .expect("Should create pair");

        // Populate with blocks at positions 0-4
        let (_, hashes) =
            populate_leader_with_blocks(&pair.leader_a, 5, 4, 0).expect("Should populate");

        // Create some hashes that won't be found
        let token_seq = token_blocks::create_token_sequence(3, 4, 1000);
        let nonexistent_hashes = token_blocks::generate_sequence_hashes(&token_seq);

        // Mix found and not-found hashes
        let mixed_hashes: Vec<_> = hashes
            .iter()
            .take(2)
            .chain(nonexistent_hashes.iter().take(2))
            .chain(hashes.iter().skip(2))
            .copied()
            .collect();

        // Linear scan should only find the existing blocks
        let blocks: Vec<TieredBlock> =
            pair.leader_a
                .leader
                .scan_with_policy(&mixed_hashes, true, |hashes, ctx| {
                    for hash in hashes {
                        if let Some(block) = ctx.accessor().find(*hash) {
                            ctx.yield_item(block);
                        }
                    }
                });

        // Should find only the 5 blocks that exist
        assert_eq!(blocks.len(), 5);
    }

    /// Test contiguous subsequence discovery policy with a single contiguous sequence.
    ///
    /// This tests that the policy correctly identifies a fully contiguous sequence
    /// as a single run.
    #[tokio::test]
    async fn test_scan_with_policy_contiguous_single_run() {
        use crate::leader::TieredBlock;

        let pair = create_instance_leader_pair(100, 4)
            .await
            .expect("Should create pair");

        // Create a single contiguous sequence (all positions are consecutive)
        let (_, hashes) =
            populate_leader_with_blocks(&pair.leader_a, 10, 4, 0).expect("Should populate");

        // Contiguous subsequence discovery policy
        let runs: Vec<Vec<TieredBlock>> =
            pair.leader_a
                .leader
                .scan_with_policy(&hashes, true, |hashes, ctx| {
                    let mut sorted_hashes = hashes.to_vec();
                    sorted_hashes.sort_by_key(|h| h.position());

                    let mut current_run = Vec::new();
                    let mut last_pos: Option<u64> = None;

                    for hash in &sorted_hashes {
                        if let Some(block) = ctx.accessor().find(*hash) {
                            let pos = block.position();
                            let is_contiguous = last_pos.is_none_or(|p| pos == p + 1);

                            if is_contiguous {
                                current_run.push(block);
                            } else {
                                if !current_run.is_empty() {
                                    ctx.yield_item(std::mem::take(&mut current_run));
                                }
                                current_run.push(block);
                            }
                            last_pos = Some(pos);
                        } else if !current_run.is_empty() {
                            ctx.yield_item(std::mem::take(&mut current_run));
                            last_pos = None;
                        }
                    }
                    if !current_run.is_empty() {
                        ctx.yield_item(current_run);
                    }
                });

        // Should find exactly 1 contiguous run containing all 10 blocks
        assert_eq!(runs.len(), 1, "Expected single contiguous run");
        assert_eq!(runs[0].len(), 10, "Run should contain all 10 blocks");

        // Verify positions are consecutive 0-9
        for (i, block) in runs[0].iter().enumerate() {
            assert_eq!(block.position(), i as u64);
        }
    }

    /// Test contiguous subsequence discovery policy with gaps.
    ///
    /// This tests that when some blocks are missing from the search,
    /// the policy correctly identifies separate runs.
    #[tokio::test]
    async fn test_scan_with_policy_contiguous_with_gaps() {
        use crate::leader::TieredBlock;

        let pair = create_instance_leader_pair(100, 4)
            .await
            .expect("Should create pair");

        // Create a contiguous sequence of 10 blocks (positions 0-9)
        let (_, all_hashes) =
            populate_leader_with_blocks(&pair.leader_a, 10, 4, 0).expect("Should populate");

        // Query only for blocks 0-2, 5-6, 8-9 (skipping 3-4 and 7)
        // This should create 3 runs: [0,1,2], [5,6], [8,9]
        let query_hashes: Vec<_> = all_hashes
            .iter()
            .enumerate()
            .filter(|(i, _)| matches!(*i, 0..=2 | 5..=6 | 8..=9))
            .map(|(_, h)| *h)
            .collect();

        // Contiguous subsequence discovery policy
        let runs: Vec<Vec<TieredBlock>> =
            pair.leader_a
                .leader
                .scan_with_policy(&query_hashes, true, |hashes, ctx| {
                    let mut sorted_hashes = hashes.to_vec();
                    sorted_hashes.sort_by_key(|h| h.position());

                    let mut current_run = Vec::new();
                    let mut last_pos: Option<u64> = None;

                    for hash in &sorted_hashes {
                        if let Some(block) = ctx.accessor().find(*hash) {
                            let pos = block.position();
                            let is_contiguous = last_pos.is_none_or(|p| pos == p + 1);

                            if is_contiguous {
                                current_run.push(block);
                            } else {
                                if !current_run.is_empty() {
                                    ctx.yield_item(std::mem::take(&mut current_run));
                                }
                                current_run.push(block);
                            }
                            last_pos = Some(pos);
                        } else if !current_run.is_empty() {
                            ctx.yield_item(std::mem::take(&mut current_run));
                            last_pos = None;
                        }
                    }
                    if !current_run.is_empty() {
                        ctx.yield_item(current_run);
                    }
                });

        // Should find 3 runs
        assert_eq!(runs.len(), 3, "Expected 3 contiguous runs");

        // First run: positions 0, 1, 2
        assert_eq!(runs[0].len(), 3);
        assert_eq!(runs[0][0].position(), 0);
        assert_eq!(runs[0][1].position(), 1);
        assert_eq!(runs[0][2].position(), 2);

        // Second run: positions 5, 6
        assert_eq!(runs[1].len(), 2);
        assert_eq!(runs[1][0].position(), 5);
        assert_eq!(runs[1][1].position(), 6);

        // Third run: positions 8, 9
        assert_eq!(runs[2].len(), 2);
        assert_eq!(runs[2][0].position(), 8);
        assert_eq!(runs[2][1].position(), 9);
    }

    /// Test scan_with_policy with tiered G2/G3 blocks.
    ///
    /// This tests the precedence behavior where G2 blocks are returned
    /// preferentially over G3 blocks when both exist.
    ///
    /// Setup:
    /// - 4 blocks total (positions 0, 1, 2, 3)
    /// - All 4 blocks are in G3
    /// - Even blocks (0, 2) are ALSO in G2
    ///
    /// Expected result:
    /// - Blocks 0, 2 should come from G2 (precedence)
    /// - Blocks 1, 3 should come from G3
    #[tokio::test]
    async fn test_scan_with_policy_tiered_g2_g3() {
        use crate::leader::TieredBlock;

        let pair = create_instance_leader_pair(50, 4)
            .await
            .expect("Should create pair");

        // Create 4 token blocks
        let token_sequence = token_blocks::create_token_sequence(4, 4, 0);
        let all_token_blocks = token_sequence.blocks();

        // Populate G3 with ALL 4 blocks
        let g3_manager = pair
            .leader_a
            .g3_manager
            .as_ref()
            .expect("G3 manager should exist");
        let g3_hashes =
            managers::populate_manager_with_blocks(g3_manager, all_token_blocks).expect("G3 pop");

        // Populate G2 with only EVEN blocks (positions 0, 2)
        let even_token_blocks: Vec<_> = all_token_blocks
            .iter()
            .enumerate()
            .filter(|(i, _)| i % 2 == 0)
            .map(|(_, b)| b.clone())
            .collect();
        let _g2_hashes =
            managers::populate_manager_with_blocks(&pair.leader_a.g2_manager, &even_token_blocks)
                .expect("G2 pop");

        // The hashes from G3 are for all blocks; G2 hashes are for even blocks only
        // We'll query using the G3 hashes (which cover all 4 blocks)
        // Note: The actual sequence hashes should match between G2 and G3 for the same token content

        // Simple linear scan to get all blocks
        let blocks: Vec<TieredBlock> =
            pair.leader_a
                .leader
                .scan_with_policy(&g3_hashes, true, |hashes, ctx| {
                    for hash in hashes {
                        if let Some(block) = ctx.accessor().find(*hash) {
                            ctx.yield_item(block);
                        }
                    }
                });

        // Should find all 4 blocks
        assert_eq!(blocks.len(), 4, "Should find all 4 blocks");

        // Count G2 vs G3 blocks
        let g2_count = blocks.iter().filter(|b| b.is_g2()).count();
        let g3_count = blocks.iter().filter(|b| b.is_g3()).count();

        // Even positions (0, 2) should be G2, odd positions (1, 3) should be G3
        assert_eq!(g2_count, 2, "Should have 2 G2 blocks (even positions)");
        assert_eq!(g3_count, 2, "Should have 2 G3 blocks (odd positions)");

        // Verify the specific tier for each position
        // Blocks are returned in the order we queried (g3_hashes order = 0, 1, 2, 3)
        assert!(blocks[0].is_g2(), "Block at position 0 should be G2 (even)");
        assert!(blocks[1].is_g3(), "Block at position 1 should be G3 (odd)");
        assert!(blocks[2].is_g2(), "Block at position 2 should be G2 (even)");
        assert!(blocks[3].is_g3(), "Block at position 3 should be G3 (odd)");

        // Verify positions are correct
        for (i, block) in blocks.iter().enumerate() {
            assert_eq!(
                block.position(),
                i as u64,
                "Block {} should be at position {}",
                i,
                i
            );
        }
    }

    /// Test scan_with_policy with empty input.
    #[tokio::test]
    async fn test_scan_with_policy_empty_hashes() {
        use crate::leader::TieredBlock;

        let pair = create_instance_leader_pair(50, 4)
            .await
            .expect("Should create pair");

        let empty_hashes: Vec<SequenceHash> = vec![];

        let blocks: Vec<TieredBlock> =
            pair.leader_a
                .leader
                .scan_with_policy(&empty_hashes, true, |hashes, ctx| {
                    for hash in hashes {
                        if let Some(block) = ctx.accessor().find(*hash) {
                            ctx.yield_item(block);
                        }
                    }
                });

        assert!(blocks.is_empty());
    }

    /// Test scan_with_policy with yield_items (batch yield).
    #[tokio::test]
    async fn test_scan_with_policy_yield_items() {
        use crate::leader::TieredBlock;

        let pair = create_instance_leader_pair(50, 4)
            .await
            .expect("Should create pair");

        // Populate with blocks
        let (_, hashes) =
            populate_leader_with_blocks(&pair.leader_a, 10, 4, 0).expect("Should populate");

        // Policy that uses yield_items to batch results
        let blocks: Vec<TieredBlock> =
            pair.leader_a
                .leader
                .scan_with_policy(&hashes, true, |hashes, ctx| {
                    let found: Vec<TieredBlock> = hashes
                        .iter()
                        .filter_map(|hash| ctx.accessor().find(*hash))
                        .collect();
                    ctx.yield_items(found);
                });

        assert_eq!(blocks.len(), 10);
    }

    /// Test scan_with_policy touch parameter.
    ///
    /// Verifies that the touch parameter is correctly passed to the accessor
    /// and affects frequency tracking behavior.
    #[tokio::test]
    async fn test_scan_with_policy_touch_parameter() {
        use crate::leader::TieredBlock;

        let pair = create_instance_leader_pair(50, 4)
            .await
            .expect("Should create pair");

        let (_, hashes) =
            populate_leader_with_blocks(&pair.leader_a, 5, 4, 0).expect("Should populate");

        // Scan with touch=false
        let blocks_no_touch: Vec<TieredBlock> =
            pair.leader_a
                .leader
                .scan_with_policy(&hashes, false, |hashes, ctx| {
                    // Verify accessor has correct touch setting
                    assert!(!ctx.accessor().touch());
                    for hash in hashes {
                        if let Some(block) = ctx.accessor().find(*hash) {
                            ctx.yield_item(block);
                        }
                    }
                });

        // Drop blocks so they return to the pool
        drop(blocks_no_touch);

        // Scan with touch=true
        let blocks_with_touch: Vec<TieredBlock> =
            pair.leader_a
                .leader
                .scan_with_policy(&hashes, true, |hashes, ctx| {
                    // Verify accessor has correct touch setting
                    assert!(ctx.accessor().touch());
                    for hash in hashes {
                        if let Some(block) = ctx.accessor().find(*hash) {
                            ctx.yield_item(block);
                        }
                    }
                });

        assert_eq!(blocks_with_touch.len(), 5);
    }
}
