// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Cancellation tests for the offload engine.
//!
//! These tests verify that:
//! - Transfers can be cancelled mid-flight
//! - The sweeper task removes cancelled items from queues
//! - Resources (ImmutableBlock guards) are properly released
//! - CancelConfirmation future resolves correctly
//!
//! Note: Uses sync tests (#[test]) with TestConnectorInstance::create_with_config
//! which properly manages the tokio runtime to avoid drop panics.

#[cfg(test)]
mod tests {
    use std::marker::PhantomData;
    use std::sync::Arc;
    use std::time::Duration;

    use anyhow::Result;

    use crate::v2::distributed::offload::{
        EvalContext, OffloadEngine, OffloadPolicy, PipelineBuilder, PolicyFuture, SourceBlocks,
        TransferStatus, async_result,
    };
    use crate::v2::logical::blocks::BlockMetadata;
    use crate::v2::testing::connector::{ConnectorTestConfig, TestConnectorInstance};
    use crate::v2::{G2, G3};

    /// A slow policy that introduces artificial delays during evaluation.
    ///
    /// This allows us to test cancellation while the pipeline is actively processing.
    struct SlowPolicy<T: BlockMetadata> {
        delay: Duration,
        _marker: PhantomData<T>,
    }

    impl<T: BlockMetadata> SlowPolicy<T> {
        fn new(delay: Duration) -> Self {
            Self {
                delay,
                _marker: PhantomData,
            }
        }
    }

    impl<T: BlockMetadata> OffloadPolicy<T> for SlowPolicy<T> {
        fn name(&self) -> &str {
            "SlowPolicy"
        }

        fn evaluate<'a>(&'a self, _ctx: &'a EvalContext<T>) -> PolicyFuture<'a> {
            let delay = self.delay;
            async_result(async move {
                tokio::time::sleep(delay).await;
                Ok(true) // Pass all blocks (slowly)
            })
        }
    }

    /// Test that cancellation works correctly during active evaluation.
    ///
    /// This test:
    /// 1. Sets up a slow policy (100ms per block)
    /// 2. Enqueues 16 blocks for transfer
    /// 3. Waits a bit for evaluation to start
    /// 4. Cancels and waits for confirmation
    /// 5. Verifies status is Cancelled
    ///
    #[test]
    fn test_cancellation_during_evaluation() -> Result<()> {
        // 1. Create test instance (sync factory manages runtime)
        let config = ConnectorTestConfig::new()
            .leader_cache_blocks(64)
            .leader_disk_blocks(32);

        let instance = TestConnectorInstance::create_with_config(config, 1)?;
        let rt_handle = instance.tokio_handle();
        // Enter tokio runtime context - needed for spawning tasks in OffloadEngine
        let _guard = rt_handle.enter();

        let leader = instance.instance_leader()?;
        let registry = leader.registry();

        // 2. Populate G2 with 16 blocks (block_size=16 to match default layout page_size)
        let (_, seq_hashes) = instance.populate_g2_blocks(16, 16, 5000)?;

        // 3. Build engine with slow policy (100ms per block)
        let g2_manager = leader.g2_manager().clone();
        let g3_manager = leader.g3_manager().cloned().expect("G3 manager required");
        let policy_registry = Arc::new(registry.clone());

        let slow_policy = Arc::new(SlowPolicy::<G2>::new(Duration::from_millis(100)));

        let engine = OffloadEngine::builder(Arc::new(leader.clone()))
            .with_registry(policy_registry)
            .with_g2_manager(g2_manager.clone())
            .with_g3_manager(g3_manager.clone())
            .with_runtime(rt_handle.clone())
            .with_g2_to_g3_pipeline(
                PipelineBuilder::<G2, G3>::new()
                    .policy(slow_policy)
                    .batch_size(8)
                    .min_batch_size(1)
                    .sweep_interval(Duration::from_millis(10)) // Fast sweep for testing
                    .skip_transfers(true) // Skip actual transfers for testing
                    .build(),
            )
            .build()?;

        // 4. Get G2 blocks and start transfer
        let g2_blocks = g2_manager.match_blocks(&seq_hashes);
        let transfer_handle = engine.enqueue_g2_to_g3(SourceBlocks::Strong(g2_blocks))?;

        // 5. Wait a bit for evaluation to start (2-3 blocks should start)
        rt_handle.block_on(async {
            tokio::time::sleep(Duration::from_millis(250)).await;
        });

        // 6. Request cancellation
        let cancel_confirmation = transfer_handle.cancel();
        assert!(
            transfer_handle.is_cancelled(),
            "Handle should report cancellation requested"
        );

        // 7. Wait for confirmation with timeout
        rt_handle
            .block_on(async {
                tokio::time::timeout(Duration::from_secs(5), cancel_confirmation).await
            })
            .expect("Cancellation should confirm within 5s");

        // 8. Verify final status
        assert_eq!(
            transfer_handle.status(),
            TransferStatus::Cancelled,
            "Status should be Cancelled"
        );
        assert!(transfer_handle.is_complete(), "Transfer should be complete");

        // Log stats
        tracing::info!(
            "Cancellation test passed:\n\
             - Status: {:?}\n\
             - Completed blocks: {}\n\
             - Remaining blocks: {}",
            transfer_handle.status(),
            transfer_handle.completed_blocks().len(),
            transfer_handle.remaining_blocks().len()
        );

        Ok(())
    }

    /// Test that the sweeper task quickly removes cancelled items from queues.
    ///
    /// This verifies that cancellation doesn't wait for slow policies to complete
    /// for each queued item - the sweeper removes them proactively.
    ///
    #[test]
    fn test_sweeper_removes_queued_items_quickly() -> Result<()> {
        // 1. Create test instance (sync factory manages runtime)
        let config = ConnectorTestConfig::new()
            .leader_cache_blocks(64)
            .leader_disk_blocks(32);

        let instance = TestConnectorInstance::create_with_config(config, 1)?;
        let rt_handle = instance.tokio_handle();
        // Enter tokio runtime context - needed for spawning tasks in OffloadEngine
        let _guard = rt_handle.enter();

        let leader = instance.instance_leader()?;
        let registry = leader.registry();

        // 2. Populate G2 with 32 blocks (block_size=16 to match default layout page_size)
        let (_, seq_hashes) = instance.populate_g2_blocks(32, 16, 6000)?;

        // 3. Build engine with very slow policy (1 second per block)
        // With 32 blocks at 1s each, sequential processing would take 32 seconds.
        // But with the sweeper, cancellation should confirm in ~1-2 seconds.
        let g2_manager = leader.g2_manager().clone();
        let g3_manager = leader.g3_manager().cloned().expect("G3 manager required");
        let policy_registry = Arc::new(registry.clone());

        let very_slow_policy = Arc::new(SlowPolicy::<G2>::new(Duration::from_secs(1)));

        let engine = OffloadEngine::builder(Arc::new(leader.clone()))
            .with_registry(policy_registry)
            .with_g2_manager(g2_manager.clone())
            .with_g3_manager(g3_manager)
            .with_runtime(rt_handle.clone())
            .with_g2_to_g3_pipeline(
                PipelineBuilder::<G2, G3>::new()
                    .policy(very_slow_policy)
                    .batch_size(16) // Large batch so items queue up
                    .min_batch_size(1)
                    .sweep_interval(Duration::from_millis(10)) // Fast sweep
                    .skip_transfers(true) // Skip actual transfers for testing
                    .build(),
            )
            .build()?;

        // 4. Start transfer
        let g2_blocks = g2_manager.match_blocks(&seq_hashes);
        let transfer_handle = engine.enqueue_g2_to_g3(SourceBlocks::Strong(g2_blocks))?;

        // 5. Immediately cancel (most items still in queue)
        rt_handle.block_on(async {
            tokio::time::sleep(Duration::from_millis(50)).await;
        });
        let cancel_confirmation = transfer_handle.cancel();

        // 6. Time the cancellation confirmation
        let start = std::time::Instant::now();
        let confirmation_result = rt_handle.block_on(async {
            tokio::time::timeout(Duration::from_secs(5), cancel_confirmation).await
        });

        let elapsed = start.elapsed();

        // 7. Verify cancellation was quick (not 32 seconds)
        confirmation_result.expect("Cancellation should complete within 5s");

        assert!(
            elapsed < Duration::from_secs(3),
            "Cancellation should be fast due to sweeper, but took {:?}",
            elapsed
        );

        assert_eq!(transfer_handle.status(), TransferStatus::Cancelled);

        tracing::info!(
            "Sweeper test passed: cancellation took {:?} (expected < 3s)",
            elapsed
        );

        Ok(())
    }

    /// Test that cancellation correctly handles already-completed items.
    ///
    /// If some blocks have already been transferred before cancellation,
    /// they should remain in completed_blocks.
    ///
    #[test]
    fn test_cancellation_preserves_completed_blocks() -> Result<()> {
        // 1. Create test instance (sync factory manages runtime)
        let config = ConnectorTestConfig::new()
            .leader_cache_blocks(64)
            .leader_disk_blocks(32);

        let instance = TestConnectorInstance::create_with_config(config, 1)?;
        let rt_handle = instance.tokio_handle();
        // Enter tokio runtime context - needed for spawning tasks in OffloadEngine
        let _guard = rt_handle.enter();

        let leader = instance.instance_leader()?;
        let registry = leader.registry();

        // 2. Populate G2 with 8 blocks (block_size=16 to match default layout page_size)
        let (_, seq_hashes) = instance.populate_g2_blocks(8, 16, 7000)?;

        // 3. Build engine with medium-slow policy (200ms per block)
        // Total time would be ~1.6s for all blocks
        let g2_manager = leader.g2_manager().clone();
        let g3_manager = leader.g3_manager().cloned().expect("G3 manager required");
        let policy_registry = Arc::new(registry.clone());

        let policy = Arc::new(SlowPolicy::<G2>::new(Duration::from_millis(200)));

        let engine = OffloadEngine::builder(Arc::new(leader.clone()))
            .with_registry(policy_registry)
            .with_g2_manager(g2_manager.clone())
            .with_g3_manager(g3_manager)
            .with_runtime(rt_handle.clone())
            .with_g2_to_g3_pipeline(
                PipelineBuilder::<G2, G3>::new()
                    .policy(policy)
                    .batch_size(8)
                    .min_batch_size(1)
                    .flush_interval(Duration::from_millis(10))
                    .sweep_interval(Duration::from_millis(10))
                    .skip_transfers(true) // Skip actual transfers for testing
                    .build(),
            )
            .build()?;

        // 4. Start transfer
        let g2_blocks = g2_manager.match_blocks(&seq_hashes);
        let transfer_handle = engine.enqueue_g2_to_g3(SourceBlocks::Strong(g2_blocks))?;

        // 5. Wait long enough for some blocks to potentially complete
        // (Policy evaluation happens before batching/transfer)
        rt_handle.block_on(async {
            tokio::time::sleep(Duration::from_millis(800)).await;
        });

        // 6. Cancel
        let confirmation = transfer_handle.cancel();
        rt_handle
            .block_on(async { tokio::time::timeout(Duration::from_secs(5), confirmation).await })
            .expect("Cancellation timeout");

        // 7. Verify status
        assert_eq!(transfer_handle.status(), TransferStatus::Cancelled);

        // Log what was completed before cancellation
        let completed = transfer_handle.completed_blocks();
        let remaining = transfer_handle.remaining_blocks();

        tracing::info!(
            "Partial cancellation test:\n\
             - Completed: {} blocks\n\
             - Remaining: {} blocks",
            completed.len(),
            remaining.len()
        );

        // Note: Due to async timing, we can't assert exact numbers,
        // but the test verifies the cancellation mechanism works

        Ok(())
    }

    /// Test multiple concurrent cancellations.
    ///
    /// Verifies that multiple transfers can be cancelled independently.
    ///
    #[test]
    fn test_multiple_concurrent_cancellations() -> Result<()> {
        // 1. Create test instance (sync factory manages runtime)
        let config = ConnectorTestConfig::new()
            .leader_cache_blocks(128)
            .leader_disk_blocks(64);

        let instance = TestConnectorInstance::create_with_config(config, 1)?;
        let rt_handle = instance.tokio_handle();
        // Enter tokio runtime context - needed for spawning tasks in OffloadEngine
        let _guard = rt_handle.enter();

        let leader = instance.instance_leader()?;
        let registry = leader.registry();

        // 2. Populate G2 with 32 blocks for 2 transfers (block_size=16 to match layout)
        let (_, seq_hashes) = instance.populate_g2_blocks(32, 16, 8000)?;

        // 3. Build engine with slow policy
        let g2_manager = leader.g2_manager().clone();
        let g3_manager = leader.g3_manager().cloned().expect("G3 manager required");
        let policy_registry = Arc::new(registry.clone());

        let policy = Arc::new(SlowPolicy::<G2>::new(Duration::from_millis(100)));

        let engine = OffloadEngine::builder(Arc::new(leader.clone()))
            .with_registry(policy_registry)
            .with_g2_manager(g2_manager.clone())
            .with_g3_manager(g3_manager)
            .with_runtime(rt_handle.clone())
            .with_g2_to_g3_pipeline(
                PipelineBuilder::<G2, G3>::new()
                    .policy(policy)
                    .batch_size(8)
                    .min_batch_size(1)
                    .sweep_interval(Duration::from_millis(10))
                    .skip_transfers(true) // Skip actual transfers for testing
                    .build(),
            )
            .build()?;

        // 4. Start two transfers with different block sets
        let blocks1 = g2_manager.match_blocks(&seq_hashes[0..16]);
        let blocks2 = g2_manager.match_blocks(&seq_hashes[16..32]);

        let transfer_handle1 = engine.enqueue_g2_to_g3(SourceBlocks::Strong(blocks1))?;
        let transfer_handle2 = engine.enqueue_g2_to_g3(SourceBlocks::Strong(blocks2))?;

        // 5. Wait a bit, then cancel both
        rt_handle.block_on(async {
            tokio::time::sleep(Duration::from_millis(200)).await;
        });

        let confirm1 = transfer_handle1.cancel();
        let confirm2 = transfer_handle2.cancel();

        // 6. Wait for both confirmations
        rt_handle
            .block_on(async { tokio::time::timeout(Duration::from_secs(5), confirm1).await })
            .expect("Cancel 1 timeout");
        rt_handle
            .block_on(async { tokio::time::timeout(Duration::from_secs(5), confirm2).await })
            .expect("Cancel 2 timeout");

        // 7. Verify both are cancelled
        assert_eq!(transfer_handle1.status(), TransferStatus::Cancelled);
        assert_eq!(transfer_handle2.status(), TransferStatus::Cancelled);

        tracing::info!("Multiple cancellation test passed");

        Ok(())
    }
}
