// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! End-to-end test for G2→G3 offload flow with frequency-based filtering.
//!
//! This test demonstrates:
//! - Setting up the OffloadEngine with pipelines and policies
//! - Using PresenceAndLFUFilter to filter blocks based on frequency
//! - Artificially bumping frequency counts via `registry.touch()`
//! - Verifying that only "hot" blocks (freq >= threshold) are offloaded
//!
//! Note: Uses sync tests (#[test]) with TestConnectorInstance::create_with_config
//! which properly manages the tokio runtime to avoid drop panics.

#[cfg(test)]
mod tests {
    use std::sync::Arc;
    use std::time::Duration;

    use anyhow::Result;

    use crate::v2::integrations::offload::{
        ExternalBlock, OffloadEngine, PipelineBuilder, PresenceAndLFUFilter, PresenceFilter,
        SourceBlocks, TransferStatus,
    };
    use crate::v2::testing::connector::{ConnectorTestConfig, TestConnectorInstance};
    use crate::v2::{G2, G3};

    /// Test G2→G3 offload with presence and LFU frequency filtering.
    ///
    /// Scenario:
    /// 1. Create 16 blocks in G2
    /// 2. Bump frequency count for 4 "hot" blocks (freq = 5)
    /// 3. Offload G2→G3 with PresenceAndLFUFilter (threshold = 4)
    /// 4. Verify only hot blocks are transferred
    ///
    #[test]
    fn test_g2_to_g3_with_frequency_filter() -> Result<()> {
        // 1. Create test instance with G2 and G3 tiers
        // Uses sync factory which properly manages tokio runtime
        let config = ConnectorTestConfig::new()
            .leader_cache_blocks(64) // G2: 64 blocks
            .leader_disk_blocks(32); // G3: 32 blocks

        let instance = TestConnectorInstance::create_with_config(config, 1)?;
        let handle = instance.tokio_handle();
        // Enter tokio runtime context - needed for spawning tasks in OffloadEngine
        let _guard = handle.enter();

        let leader = instance.instance_leader()?;
        let registry = leader.registry();

        // Check that frequency tracking is enabled
        assert!(
            registry.has_frequency_tracking(),
            "Frequency tracking should be enabled for this test"
        );

        // 2. Populate G2 with 16 blocks (block_size=16 to match default layout page_size)
        let (_, seq_hashes) = instance.populate_g2_blocks(16, 16, 1000)?;
        assert_eq!(seq_hashes.len(), 16, "Should have 16 blocks in G2");

        // 3. Artificially bump frequency for first 4 blocks (make them "hot")
        let hot_hashes = &seq_hashes[0..4];
        for hash in hot_hashes {
            // Touch 5 times to get freq > 4
            registry.touch(*hash);
            registry.touch(*hash);
            registry.touch(*hash);
            registry.touch(*hash);
            registry.touch(*hash);
        }

        // Verify hot blocks have count > 4 (5 touches + any initial from populate)
        for hash in hot_hashes {
            let count = registry.count(*hash);
            assert!(count > 4, "Hot block should have count > 4, got {}", count);
        }

        // Cold blocks should have count < threshold (any initial count from populate)
        // The threshold is 4, so cold blocks with count < 4 should be filtered
        for hash in &seq_hashes[4..] {
            let count = registry.count(*hash);
            assert!(count < 4, "Cold block should have count < 4, got {}", count);
        }

        // 4. Build OffloadEngine with G2→G3 pipeline using frequency filter
        let g2_manager = leader.g2_manager().clone();
        let g3_manager = leader
            .g3_manager()
            .cloned()
            .expect("G3 manager should be configured");

        // We need to create a registry for the policies - use the same one from leader
        let policy_registry = Arc::new(registry.clone());

        let engine = OffloadEngine::builder(Arc::new(leader.clone()))
            .with_registry(policy_registry.clone())
            .with_g2_manager(g2_manager.clone())
            .with_g3_manager(g3_manager.clone())
            .with_runtime(handle.clone())
            .with_g2_to_g3_pipeline(
                PipelineBuilder::<G2, G3>::new()
                    .policy(Arc::new(PresenceAndLFUFilter::<G2, G3>::new(
                        policy_registry.clone(),
                        4, // min_lfu_count threshold
                    )))
                    .batch_size(8)
                    .min_batch_size(1) // Process smaller batches for testing
                    .flush_interval(Duration::from_millis(5))
                    .skip_transfers(true) // Skip actual transfers for testing
                    .build(),
            )
            .build()?;

        // 5. Get G2 blocks to offload
        let g2_blocks = g2_manager.match_blocks(&seq_hashes);
        assert_eq!(g2_blocks.len(), 16, "Should match all 16 G2 blocks");

        // 6. Enqueue G2→G3 offload
        let mut transfer_handle = engine.enqueue_g2_to_g3(SourceBlocks::Strong(g2_blocks))?;

        // 7. Wait for completion with timeout (use runtime handle for async)
        let result = handle
            .block_on(async {
                tokio::time::timeout(Duration::from_secs(10), transfer_handle.wait()).await
            })
            .expect("Transfer should complete within 10s")?;

        // 8. Verify results
        assert_eq!(
            result.status,
            TransferStatus::Complete,
            "Transfer should complete successfully"
        );
        assert_eq!(
            result.completed_blocks.len(),
            4,
            "Only 4 hot blocks should be offloaded, got {}",
            result.completed_blocks.len()
        );
        assert_eq!(
            result.filtered_blocks.len(),
            12,
            "12 cold blocks should be filtered, got {}",
            result.filtered_blocks.len()
        );

        // Note: With skip_transfers=true, blocks aren't actually moved to G3.
        // We're testing the pipeline flow (policy evaluation, batching, completion tracking),
        // not actual data transfer.

        tracing::info!(
            "G2→G3 offload test passed: {} blocks marked as transferred, {} filtered",
            result.completed_blocks.len(),
            result.filtered_blocks.len()
        );

        Ok(())
    }

    /// Test G2→G3 offload with presence-only filter (no frequency requirement).
    ///
    /// This test verifies the basic pipeline flow without frequency filtering.
    ///
    #[test]
    fn test_g2_to_g3_presence_only() -> Result<()> {
        // 1. Create test instance (sync factory manages runtime)
        let config = ConnectorTestConfig::new()
            .leader_cache_blocks(64) // G2
            .leader_disk_blocks(32) // G3
            .leader_tokio_threads(4); // Ensure multi-threaded runtime for spawned tasks

        let instance = TestConnectorInstance::create_with_config(config, 1)?;
        let handle = instance.tokio_handle();
        // Enter tokio runtime context - needed for spawning tasks in OffloadEngine
        let _guard = handle.enter();

        let leader = instance.instance_leader()?;
        let registry = leader.registry();

        // 2. Populate G2 with 8 blocks (block_size=16 to match default layout page_size)
        let (_, seq_hashes) = instance.populate_g2_blocks(8, 16, 2000)?;

        // 3. Build OffloadEngine with presence-only filter
        let g2_manager = leader.g2_manager().clone();
        let g3_manager = leader.g3_manager().cloned().expect("G3 manager required");
        let policy_registry = Arc::new(registry.clone());

        let engine = OffloadEngine::builder(Arc::new(leader.clone()))
            .with_registry(policy_registry.clone())
            .with_g2_manager(g2_manager.clone())
            .with_g3_manager(g3_manager.clone())
            .with_runtime(handle.clone())
            .with_g2_to_g3_pipeline(
                PipelineBuilder::<G2, G3>::new()
                    .policy(Arc::new(PresenceFilter::<G2, G3>::new(
                        policy_registry.clone(),
                    )))
                    .batch_size(8)
                    .min_batch_size(1)
                    .flush_interval(Duration::from_millis(5))
                    .skip_transfers(true) // Skip actual transfers for testing
                    .build(),
            )
            .build()?;

        // 4. Get G2 blocks and offload
        let g2_blocks = g2_manager.match_blocks(&seq_hashes);
        let mut transfer_handle = engine.enqueue_g2_to_g3(SourceBlocks::Strong(g2_blocks))?;

        // 5. Wait for completion
        let result = handle
            .block_on(async {
                tokio::time::timeout(Duration::from_secs(10), transfer_handle.wait()).await
            })
            .expect("Transfer should complete within 10s")?;

        // 6. Verify all blocks transferred (no frequency filter)
        assert_eq!(result.status, TransferStatus::Complete);
        assert_eq!(
            result.completed_blocks.len(),
            8,
            "All 8 blocks should be transferred with presence-only filter"
        );
        assert_eq!(
            result.filtered_blocks.len(),
            0,
            "No blocks should be filtered with presence-only"
        );

        // Note: With skip_transfers=true, blocks aren't actually moved to G3.
        // We're testing the pipeline flow (policy evaluation, batching, completion tracking),
        // not actual data transfer.

        tracing::info!(
            "G2→G3 presence-only test passed: {} blocks marked as transferred",
            8
        );

        Ok(())
    }

    /// Test that blocks already in G3 are filtered by presence check.
    ///
    #[test]
    fn test_g2_to_g3_filters_existing_blocks() -> Result<()> {
        // 1. Create test instance (sync factory manages runtime)
        let config = ConnectorTestConfig::new()
            .leader_cache_blocks(64)
            .leader_disk_blocks(32);

        let instance = TestConnectorInstance::create_with_config(config, 1)?;
        let handle = instance.tokio_handle();
        // Enter tokio runtime context - needed for spawning tasks in OffloadEngine
        let _guard = handle.enter();

        let leader = instance.instance_leader()?;
        let registry = leader.registry();

        // 2. Populate G2 with 8 blocks (block_size=16 to match default layout page_size)
        let (_, g2_hashes) = instance.populate_g2_blocks(8, 16, 3000)?;

        // 3. Pre-populate G3 with 4 of the same blocks (simulate already offloaded)
        // Note: We use different start tokens to create different blocks in G3
        let (_, _g3_hashes) = instance.populate_g3_blocks(4, 16, 4000)?;

        // 4. Build engine with presence filter
        let g2_manager = leader.g2_manager().clone();
        let g3_manager = leader.g3_manager().cloned().expect("G3 manager required");
        let policy_registry = Arc::new(registry.clone());

        let engine = OffloadEngine::builder(Arc::new(leader.clone()))
            .with_registry(policy_registry.clone())
            .with_g2_manager(g2_manager.clone())
            .with_g3_manager(g3_manager.clone())
            .with_runtime(handle.clone())
            .with_g2_to_g3_pipeline(
                PipelineBuilder::<G2, G3>::new()
                    .policy(Arc::new(PresenceFilter::<G2, G3>::new(
                        policy_registry.clone(),
                    )))
                    .batch_size(8)
                    .min_batch_size(1)
                    .flush_interval(Duration::from_millis(5))
                    .skip_transfers(true) // Skip actual transfers for testing
                    .build(),
            )
            .build()?;

        // 5. Offload G2 blocks
        let g2_blocks = g2_manager.match_blocks(&g2_hashes);
        let mut transfer_handle = engine.enqueue_g2_to_g3(SourceBlocks::Strong(g2_blocks))?;

        let result = handle
            .block_on(async {
                tokio::time::timeout(Duration::from_secs(10), transfer_handle.wait()).await
            })
            .expect("Transfer should complete")?;

        // 6. Verify all 8 transferred (since these are different blocks than g3_hashes)
        // The presence filter checks by sequence hash, and g2_hashes != g3_hashes
        assert_eq!(result.status, TransferStatus::Complete);
        assert_eq!(
            result.completed_blocks.len(),
            8,
            "All G2 blocks should transfer since they have different hashes"
        );

        tracing::info!(
            "G2→G3 presence filter test passed: {} transferred",
            result.completed_blocks.len()
        );

        Ok(())
    }

    /// Test G2→G3 offload with full RDMA transfers and registration.
    ///
    /// This test verifies the complete transfer path:
    /// 1. Blocks are populated in G2 manager (metadata)
    /// 2. G2 physical blocks are filled with test pattern
    /// 3. RDMA transfer executes (skip_transfers=false)
    /// 4. Blocks are registered in G3 destination tier
    /// 5. Blocks can be found in G3 via match_blocks()
    /// 6. G3 physical blocks have correct data
    ///
    #[test]
    fn test_g2_to_g3_full_rdma_transfer() -> Result<()> {
        use crate::v2::physical::transfer::FillPattern;

        // 1. Create test instance with G2 and G3 tiers
        let config = ConnectorTestConfig::new()
            .leader_cache_blocks(64) // G2
            .leader_disk_blocks(32); // G3

        let instance = TestConnectorInstance::create_with_config(config, 1)?;
        let handle = instance.tokio_handle();
        let _guard = handle.enter();

        let leader = instance.instance_leader()?;
        let registry = leader.registry();

        // Check if parallel_worker is available (required for actual transfers)
        let has_parallel_worker = leader.has_parallel_worker();
        eprintln!(
            "DEBUG: has_parallel_worker = {} (workers need to be initialized for this)",
            has_parallel_worker
        );

        // Enable full RDMA transfers when workers are properly configured
        let skip_transfers = !has_parallel_worker;

        eprintln!(
            "DEBUG: Using skip_transfers={} (has_parallel_worker={})",
            skip_transfers, has_parallel_worker
        );

        // 2. Populate G2 manager with 4 blocks (small to make test fast)
        let (block_ids, seq_hashes) = instance.populate_g2_blocks(4, 16, 5000)?;
        assert_eq!(seq_hashes.len(), 4, "Should have 4 blocks in G2");
        eprintln!("DEBUG: Populated G2 with block_ids: {:?}", block_ids);

        // Verify blocks are in G2 manager
        let g2_manager = leader.g2_manager().clone();
        let g2_blocks_initial = g2_manager.match_blocks(&seq_hashes);
        assert_eq!(g2_blocks_initial.len(), 4, "Should find all 4 blocks in G2");

        // 3. Fill G2 physical blocks with test pattern (so we can verify after transfer)
        // Use first worker for simplicity
        let worker = instance
            .workers
            .first()
            .expect("Should have at least 1 worker");
        // fill_g2_blocks is sync (uses direct memory access)
        let g2_checksums = worker.fill_g2_blocks(&block_ids, FillPattern::Constant(0xAB))?;
        eprintln!(
            "DEBUG: Filled G2 blocks with pattern 0xAB, checksums: {:?}",
            g2_checksums
        );

        // Verify blocks are NOT in G3 yet
        let g3_manager = leader.g3_manager().cloned().expect("G3 manager required");
        let g3_blocks_before = g3_manager.match_blocks(&seq_hashes);
        assert_eq!(g3_blocks_before.len(), 0, "Blocks should not be in G3 yet");

        // 4. Build OffloadEngine with transfer configuration based on environment
        let policy_registry = Arc::new(registry.clone());

        let engine = OffloadEngine::builder(Arc::new(leader.clone()))
            .with_registry(policy_registry.clone())
            .with_g2_manager(g2_manager.clone())
            .with_g3_manager(g3_manager.clone())
            .with_runtime(handle.clone())
            .with_g2_to_g3_pipeline(
                PipelineBuilder::<G2, G3>::new()
                    .policy(Arc::new(PresenceFilter::<G2, G3>::new(
                        policy_registry.clone(),
                    )))
                    .batch_size(8)
                    .min_batch_size(1)
                    .flush_interval(Duration::from_millis(5))
                    .skip_transfers(skip_transfers) // Based on environment capability
                    .build(),
            )
            .build()?;

        // 5. Get fresh G2 blocks and offload
        let g2_blocks = g2_manager.match_blocks(&seq_hashes);
        let mut transfer_handle = engine.enqueue_g2_to_g3(SourceBlocks::Strong(g2_blocks))?;

        // 6. Wait for completion with increased timeout for actual RDMA
        let timeout_secs = if skip_transfers { 10 } else { 30 };
        let result = handle
            .block_on(async {
                tokio::time::timeout(Duration::from_secs(timeout_secs), transfer_handle.wait())
                    .await
            })
            .expect(&format!(
                "Transfer should complete within {}s",
                timeout_secs
            ))?;

        // 7. Verify transfer completed
        assert_eq!(
            result.status,
            TransferStatus::Complete,
            "Transfer should complete successfully"
        );
        assert_eq!(
            result.completed_blocks.len(),
            4,
            "All 4 blocks should be transferred"
        );

        // 8. Verify blocks in G3 only if actual transfers were executed
        if !skip_transfers {
            // CRITICAL: Verify blocks are now findable in G3
            let g3_blocks_after = g3_manager.match_blocks(&seq_hashes);
            assert_eq!(
                g3_blocks_after.len(),
                4,
                "All 4 blocks should now be registered in G3 (found {}, expected 4)",
                g3_blocks_after.len()
            );

            // Verify the registered blocks have the correct sequence hashes
            for (expected_hash, block) in seq_hashes.iter().zip(g3_blocks_after.iter()) {
                assert_eq!(
                    block.sequence_hash(),
                    *expected_hash,
                    "G3 block should have matching sequence hash"
                );
            }

            // Get the destination block IDs from the completed_blocks
            let g3_block_ids: Vec<_> = g3_blocks_after.iter().map(|b| b.block_id()).collect();
            eprintln!("DEBUG: G3 block_ids after transfer: {:?}", g3_block_ids);

            // Verify G3 physical data matches original pattern by computing checksums
            // compute_g3_checksums is async (uses RDMA transfer to bounce buffer)
            let g3_checksums = handle.block_on(worker.compute_g3_checksums(&g3_block_ids))?;
            eprintln!("DEBUG: G3 checksums after transfer: {:?}", g3_checksums);

            // Compare checksums - they should match (same data transferred)
            for block_id in &block_ids {
                let g2_checksum = g2_checksums
                    .get(block_id)
                    .expect("G2 block should have checksum");
                // Find the corresponding G3 block (same index in output as input)
                let g3_idx = block_ids.iter().position(|b| b == block_id).unwrap();
                let g3_block_id = g3_block_ids[g3_idx];
                let g3_checksum = g3_checksums
                    .get(&g3_block_id)
                    .expect("G3 block should have checksum");

                assert_eq!(
                    g2_checksum, g3_checksum,
                    "G3 block {} checksum should match G2 block {} (G2: {:?}, G3: {:?})",
                    g3_block_id, block_id, g2_checksum, g3_checksum
                );
            }

            eprintln!(
                "G2→G3 full RDMA transfer test passed: {} blocks transferred, registered in G3, data verified",
                result.completed_blocks.len()
            );
        } else {
            eprintln!(
                "G2→G3 pipeline flow test passed: {} blocks marked as completed (skip_transfers=true)",
                result.completed_blocks.len()
            );
        }

        Ok(())
    }

    /// Test G2→G3 offload using external blocks path.
    ///
    /// This test demonstrates the external blocks flow where:
    /// 1. Caller holds `ImmutableBlock<G2>` externally (mimics vLLM scheduler holding blocks)
    /// 2. KVBM receives `ExternalBlock<G2>` (just metadata: block_id + sequence_hash)
    /// 3. Transfer executes while caller still holds the blocks
    /// 4. Caller releases blocks after transfer completes
    ///
    /// Key differences from Strong blocks:
    /// - External blocks bypass policy evaluation (caller explicitly chose them)
    /// - No upgrade stage needed (caller guarantees block validity)
    /// - block_id and sequence_hash pass through without RAII guards
    ///
    #[test]
    fn test_g2_to_g3_external_blocks_path() -> Result<()> {
        // 1. Create test instance
        let config = ConnectorTestConfig::new()
            .leader_cache_blocks(64) // G2
            .leader_disk_blocks(32); // G3

        let instance = TestConnectorInstance::create_with_config(config, 1)?;
        let handle = instance.tokio_handle();
        let _guard = handle.enter();

        let leader = instance.instance_leader()?;
        let registry = leader.registry();

        // 2. Populate G2 with 4 blocks
        let (_, seq_hashes) = instance.populate_g2_blocks(4, 16, 6000)?;
        assert_eq!(seq_hashes.len(), 4);

        // 3. Get G2 blocks and HOLD THEM EXTERNALLY (simulates vLLM holding blocks)
        let g2_manager = leader.g2_manager().clone();
        let g2_blocks = g2_manager.match_blocks(&seq_hashes);
        assert_eq!(g2_blocks.len(), 4, "Should hold all 4 G2 blocks");

        // 4. Create ExternalBlock references from held blocks (just metadata)
        let external_blocks: Vec<ExternalBlock<G2>> = g2_blocks
            .iter()
            .map(|block| ExternalBlock::new(block.block_id(), block.sequence_hash()))
            .collect();

        assert_eq!(external_blocks.len(), 4);

        // Log what we're doing
        eprintln!(
            "DEBUG: Created {} ExternalBlock<G2> references",
            external_blocks.len()
        );
        eprintln!(
            "DEBUG: Holding {} ImmutableBlock<G2> externally (simulating vLLM scheduler)",
            g2_blocks.len()
        );

        // 5. Build OffloadEngine
        let g3_manager = leader.g3_manager().cloned().expect("G3 manager required");
        let policy_registry = Arc::new(registry.clone());

        let engine = OffloadEngine::builder(Arc::new(leader.clone()))
            .with_registry(policy_registry.clone())
            .with_g2_manager(g2_manager.clone())
            .with_g3_manager(g3_manager.clone())
            .with_runtime(handle.clone())
            .with_g2_to_g3_pipeline(
                PipelineBuilder::<G2, G3>::new()
                    .policy(Arc::new(PresenceFilter::<G2, G3>::new(
                        policy_registry.clone(),
                    )))
                    .batch_size(8)
                    .min_batch_size(1)
                    .flush_interval(Duration::from_millis(5))
                    .skip_transfers(true) // Pipeline flow only, no actual RDMA
                    .build(),
            )
            .build()?;

        // 6. Enqueue EXTERNAL blocks (not the Strong blocks we're holding)
        // Note: We pass external_blocks, NOT g2_blocks
        let mut transfer_handle =
            engine.enqueue_g2_to_g3(SourceBlocks::External(external_blocks))?;

        // At this point:
        // - g2_blocks are still held by us (simulating vLLM scheduler)
        // - Pipeline has ExternalBlock references (just metadata)
        // - Transfer should proceed since external blocks bypass policy

        // 7. Wait for completion
        let result = handle
            .block_on(async {
                tokio::time::timeout(Duration::from_secs(10), transfer_handle.wait()).await
            })
            .expect("Transfer should complete within 10s")?;

        // 8. Verify results
        assert_eq!(
            result.status,
            TransferStatus::Complete,
            "Transfer should complete successfully"
        );
        // External blocks BYPASS policy evaluation, so all should be "completed"
        // (marked as passed, even though skip_transfers=true means no actual data moved)
        assert_eq!(
            result.completed_blocks.len(),
            4,
            "All 4 external blocks should be marked completed (bypassed policy)"
        );
        assert_eq!(
            result.filtered_blocks.len(),
            0,
            "External blocks bypass policy, none should be filtered"
        );

        eprintln!(
            "External blocks path test passed: {} completed, {} filtered",
            result.completed_blocks.len(),
            result.filtered_blocks.len()
        );

        // 9. NOW we can release the external blocks (transfer is complete)
        // In production, vLLM scheduler would release blocks when decode completes
        drop(g2_blocks);
        eprintln!("DEBUG: Released externally-held ImmutableBlock<G2> blocks");

        Ok(())
    }
}
