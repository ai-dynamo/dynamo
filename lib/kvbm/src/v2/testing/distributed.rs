// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
    BlockId,
    logical::manager::BlockManager,
    physical::manager::{LayoutHandle, TransferManager},
    v2::{
        InstanceId,
        distributed::{
            leader::InstanceLeader,
            worker::{DirectWorker, Worker},
        },
        integrations::{G2, G3},
        logical::pools::SequenceHash,
        physical::{
            layout::LayoutConfig,
            transfer::{BlockChecksum, FillPattern},
        },
    },
};
use dynamo_memory::{StorageKind, nixl::NixlAgent};

use super::{managers, nova, physical};

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

/// Create a pair of InstanceLeaders connected via Nova for integration testing.
///
/// Setup:
/// - Two Nova instances with TCP transport
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
    // Create Nova pair
    let nova::NovaPair { nova_a, nova_b } = nova::create_nova_pair_tcp().await?;

    // Create G2 managers
    let g2_manager_a = Arc::new(managers::create_test_manager::<G2>(block_count, block_size));
    let g3_manager_a = Arc::new(managers::create_test_manager::<G3>(block_count, block_size));

    let g2_manager_b = Arc::new(managers::create_test_manager::<G2>(block_count, block_size));
    let g3_manager_b = Arc::new(managers::create_test_manager::<G3>(block_count, block_size));

    // Build InstanceLeader A
    let leader_a = InstanceLeader::builder()
        .nova(nova_a.clone())
        .g2_manager(g2_manager_a.clone())
        .g3_manager(g3_manager_a.clone())
        .workers(vec![]) // No workers for now (no transfers)
        .remote_leaders(vec![nova_b.instance_id()])
        .build()?;

    // Register handlers for A
    leader_a.register_handlers()?;

    // Build InstanceLeader B
    let leader_b = InstanceLeader::builder()
        .nova(nova_b.clone())
        .g2_manager(g2_manager_b.clone())
        .g3_manager(g3_manager_b.clone())
        .workers(vec![]) // No workers for now
        .remote_leaders(vec![nova_a.instance_id()])
        .build()?;

    // Register handlers for B
    leader_b.register_handlers()?;

    Ok(InstanceLeaderPair {
        leader_a: TestInstanceLeader {
            instance_id: nova_a.instance_id(),
            leader: leader_a,
            g2_manager: g2_manager_a,
            g3_manager: Some(g3_manager_a),
        },
        leader_b: TestInstanceLeader {
            instance_id: nova_b.instance_id(),
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
pub fn create_direct_worker(
    instance_id: InstanceId,
    agent_name: &str,
    layout_config: &LayoutConfig,
    storage: StorageKind,
) -> Result<TestWorker> {
    // Derive worker_id from instance_id (deterministic hash)
    let worker_id = instance_id.worker_id().as_u64();

    // Create LocalEventSystem with the derived worker_id (already returns Arc<Self>)
    let event_system = dynamo_nova::events::LocalEventSystem::new(worker_id);

    // Create NixlAgent with UCX backend
    let agent = NixlAgent::with_backends(agent_name, &["UCX"])?;

    // Create TransferManager with the event_system
    let manager = TransferManager::builder()
        .event_system(event_system)
        .nixl_agent(agent.clone())
        .cuda_device_id(0)
        .build()?;

    // Create and register G2 physical layout
    // This will create LayoutHandle(worker_id, 0) - now unique per worker!
    let layout = physical::create_fc_layout_with_config(agent, storage, layout_config.clone());
    let g2_handle = manager.register_layout(layout)?;

    // Create DirectWorker with G2 handle
    let direct_worker = DirectWorker::new(manager.clone()).with_g2_handle(g2_handle);

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
/// * `nova` - Nova instance for leader communication
/// * `remote_leaders` - Instance IDs of remote leaders
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
    nova: Arc<dynamo_nova::am::Nova>,
    remote_leaders: Vec<InstanceId>,
    agent_name_prefix: &str,
) -> Result<TestInstanceLeaderWithWorkers> {
    // Create G2 and G3 managers
    let g2_manager = Arc::new(managers::create_test_manager::<G2>(block_count, block_size));
    let g3_manager = Arc::new(managers::create_test_manager::<G3>(block_count, block_size));

    // Create DirectWorkers
    let workers = create_direct_workers(num_workers, layout_config, storage, agent_name_prefix)?;

    // Extract worker references for the leader
    let worker_refs: Vec<Arc<dyn Worker>> = workers
        .iter()
        .map(|w| w.worker.clone() as Arc<dyn Worker>)
        .collect();

    // Build InstanceLeader
    let leader = InstanceLeader::builder()
        .nova(nova.clone())
        .g2_manager(g2_manager.clone())
        .g3_manager(g3_manager.clone())
        .workers(worker_refs)
        .remote_leaders(remote_leaders)
        .build()?;

    // Register handlers
    leader.register_handlers()?;

    Ok(TestInstanceLeaderWithWorkers {
        instance_id: nova.instance_id(),
        leader,
        g2_manager,
        g3_manager: Some(g3_manager),
        workers,
    })
}

/// Create a pair of InstanceLeaders with workers for RDMA integration testing.
///
/// Setup:
/// - Two Nova instances with TCP transport
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
    // Create Nova pair
    let nova::NovaPair { nova_a, nova_b } = nova::create_nova_pair_tcp().await?;

    // Create Decode leader with workers
    let decode = create_instance_leader_with_workers(
        block_count,
        block_size,
        num_workers,
        layout_config,
        storage,
        nova_a.clone(),
        vec![nova_b.instance_id()],
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
        nova_b.clone(),
        vec![nova_a.instance_id()],
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
    // RDMA Transfer Tests (require UCX and CUDA)
    // =========================================================================

    // Test constants - scaling up to 2 workers, 4 blocks each
    const NUM_WORKERS: usize = 2; // Two workers now
    const LAYOUT_BLOCKS: usize = 16; // Blocks per layout
    const TEST_BLOCKS: usize = 4; // Test 4 blocks at once
    const BLOCK_SIZE: usize = 4; // Tokens per block
    const NUM_LAYERS: usize = 2; // Layers
    const OUTER_DIM: usize = 1; // Outer dim
    const PAGE_SIZE: usize = 4;
    const INNER_DIM: usize = 64;
    const DTYPE_WIDTH: usize = 2; // bf16
    const MANAGER_BLOCKS: usize = 16; // Blocks in G2 BlockManager

    fn test_layout_config() -> crate::v2::physical::layout::LayoutConfig {
        physical::custom_config(
            LAYOUT_BLOCKS,
            NUM_LAYERS,
            OUTER_DIM,
            PAGE_SIZE,
            INNER_DIM,
            DTYPE_WIDTH,
        )
    }

    /// Full RDMA transfer test with checksum verification.
    ///
    /// This test (simplified to 1 block for debugging):
    /// 1. Creates a pair of leaders with 1 worker each
    /// 2. Fills Decode worker's G2 layout with 0xAA pattern
    /// 3. Fills Prefill worker's G2 destination with 0xBB pattern
    /// 4. Prefill pulls block via RDMA
    /// 5. Verifies: Decode unchanged (still 0xAA), Prefill has Decode's data (now 0xAA)
    ///
    /// If the transfer goes the wrong direction (PUT instead of GET):
    /// - Decode would have 0xBB (wrong!)
    /// - Prefill would have 0xBB (unchanged, wrong!)
    #[tokio::test(flavor = "multi_thread")]
    async fn test_rdma_transfer_with_checksum_verification() {
        use crate::v2::distributed::leader::ControllableSessionOptions;
        use std::time::Duration;
        use tokio::time::timeout;

        let layout_config = test_layout_config();

        // 1. Create leader pair with workers
        let pair = create_instance_leader_pair_with_workers(
            MANAGER_BLOCKS,
            BLOCK_SIZE,
            NUM_WORKERS,
            &layout_config,
            StorageKind::Pinned,
        )
        .await
        .expect("Should create leader pair with workers");

        println!(
            "\n=== RDMA Direction Test (1 block) ===\n\
             Decode (source): instance={}, {} workers\n\
             Prefill (dest): instance={}, {} workers",
            pair.decode.instance_id,
            pair.decode.workers.len(),
            pair.prefill.instance_id,
            pair.prefill.workers.len()
        );

        // 2. Define block IDs - multiple blocks now
        let src_block_ids: Vec<BlockId> = (0..TEST_BLOCKS as BlockId).collect();
        // Use non-overlapping block IDs for destination
        let dst_block_ids: Vec<BlockId> = (TEST_BLOCKS..(TEST_BLOCKS * 2) as BlockId).collect();

        println!(
            "Testing {} blocks x {} workers: src={:?}, dst={:?}",
            TEST_BLOCKS, NUM_WORKERS, src_block_ids, dst_block_ids
        );

        // 3. Fill ALL DECODE workers' source blocks with 0xAA pattern
        let mut decode_checksums_before_by_worker = Vec::new();
        for (i, worker) in pair.decode.workers.iter().enumerate() {
            let checksums = worker
                .fill_g2_blocks(&src_block_ids, FillPattern::Constant(0xAA))
                .expect("Should fill Decode G2 blocks");
            println!(
                "BEFORE transfer - Decode worker {} blocks: {:?}",
                i, src_block_ids
            );
            decode_checksums_before_by_worker.push(checksums);
        }

        // 4. Fill ALL PREFILL workers' destination blocks with 0xBB pattern (different!)
        let mut prefill_checksums_before_by_worker = Vec::new();
        for (i, worker) in pair.prefill.workers.iter().enumerate() {
            let checksums = worker
                .fill_g2_blocks(&dst_block_ids, FillPattern::Constant(0xBB))
                .expect("Should fill Prefill G2 blocks");
            println!(
                "BEFORE transfer - Prefill worker {} blocks: {:?}",
                i, dst_block_ids
            );
            prefill_checksums_before_by_worker.push(checksums);
        }

        // Sanity check: they should be different
        assert_ne!(
            decode_checksums_before_by_worker[0][&0],
            prefill_checksums_before_by_worker[0][&dst_block_ids[0]],
            "Pre-transfer: Decode and Prefill should have different data"
        );

        // 5. Populate Decode's logical manager with blocks
        let test_leader = TestInstanceLeader {
            instance_id: pair.decode.instance_id,
            leader: pair.decode.leader.clone(),
            g2_manager: pair.decode.g2_manager.clone(),
            g3_manager: pair.decode.g3_manager.clone(),
        };
        let (_, sequence_hashes) =
            populate_leader_with_blocks(&test_leader, TEST_BLOCKS, BLOCK_SIZE, 0)
                .expect("Should populate leader");

        // 6. Create controllable session on Decode
        let session_result = pair
            .decode
            .leader
            .create_controllable_session_with_options(
                &sequence_hashes,
                ControllableSessionOptions { auto_stage: false },
            )
            .expect("Should create controllable session");
        println!(
            "Decode session created: {} G2 blocks",
            session_result.local_g2_count
        );

        // 7. Prefill attaches
        let mut handle = pair
            .prefill
            .leader
            .attach_remote_session(pair.decode.instance_id, session_result.session_id)
            .await
            .expect("Should attach");

        // 8. Wait for initial state
        let state = timeout(Duration::from_secs(5), handle.wait_for_initial_state())
            .await
            .expect("Timeout")
            .expect("Should get state");
        println!(
            "Prefill sees {} G2 blocks from Decode",
            state.g2_blocks.len()
        );

        // 9. Execute RDMA PULL: Prefill pulls FROM Decode
        println!("\n--- Executing RDMA pull: Decode block 0 -> Prefill block 1 ---");
        let notification = handle
            .pull_blocks_rdma(&state.g2_blocks, &dst_block_ids)
            .await
            .expect("Should initiate RDMA pull");
        notification.await.expect("Transfer should complete");
        println!("Transfer complete!\n");

        // 10. SPMD replication: ALL workers have ALL blocks
        // Each Prefill worker N pulled from Decode worker N
        // So verify ALL blocks on ALL workers
        println!("\nVerifying SPMD replication - all workers have all blocks:");
        println!(
            "  Each worker: src={:?} -> dst={:?}",
            src_block_ids, dst_block_ids
        );

        // 11. Verify transfer for each worker - all workers have all blocks
        for (worker_idx, (decode_worker, prefill_worker)) in pair
            .decode
            .workers
            .iter()
            .zip(pair.prefill.workers.iter())
            .enumerate()
        {
            let decode_checksums_after = decode_worker
                .compute_g2_checksums(&src_block_ids)
                .expect("compute Decode checksums");
            let prefill_checksums_after = prefill_worker
                .compute_g2_checksums(&dst_block_ids)
                .expect("compute Prefill checksums");

            println!(
                "\nWorker {} verification ({} blocks):",
                worker_idx, TEST_BLOCKS
            );

            let decode_checksums_before = &decode_checksums_before_by_worker[worker_idx];

            for i in 0..TEST_BLOCKS {
                let src_id = src_block_ids[i];
                let dst_id = dst_block_ids[i];

                // Decode source should be unchanged (still 0xAA)
                assert_eq!(
                    decode_checksums_before[&src_id], decode_checksums_after[&src_id],
                    "Decode block {} was modified!",
                    src_id
                );

                // Prefill dest should have Decode's data (now 0xAA, not 0xBB)
                assert_eq!(
                    decode_checksums_before[&src_id], prefill_checksums_after[&dst_id],
                    "Prefill block {} doesn't have Decode block {}'s data",
                    dst_id, src_id
                );

                println!("  ✓ Worker {} block {} -> {}", worker_idx, src_id, dst_id);
            }
        }

        println!(
            "\n=== SUCCESS: {} blocks correctly transferred across {} workers (SPMD) ===",
            TEST_BLOCKS, NUM_WORKERS
        );

        // 12. Cleanup
        handle.mark_blocks_pulled(sequence_hashes).await.ok();
        handle.detach().await.ok();
    }
}
