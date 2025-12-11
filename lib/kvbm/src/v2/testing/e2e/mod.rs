// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! End-to-end tests for connector instances.
//!
//! This module provides E2E tests using TestConnectorCluster to test
//! multi-instance scenarios like bidirectional transfers.

#[cfg(test)]
mod find_blocks;

#[cfg(test)]
mod tests {
    use crate::v2::physical::transfer::FillPattern;
    use crate::v2::testing::connector::{ConnectorTestConfig, TestConnectorCluster};
    use std::time::Duration;
    use tokio::time::timeout;

    /// Test that a TestConnectorCluster can be created with 2 instances.
    ///
    /// This verifies:
    /// 1. Multiple instances can be created with cross-registered Nova
    /// 2. Workers are initialized successfully
    /// 3. InstanceLeaders are created and accessible
    /// 4. Sessions can be established between instances
    /// 5. Handler lists are refreshed after worker initialization (verifies cache fix)
    #[tokio::test(flavor = "multi_thread")]
    async fn test_cluster_creation_and_sessions() {
        let _ = tracing_subscriber::fmt()
            .with_max_level(tracing::Level::INFO)
            .with_test_writer()
            .try_init();

        // Create 2-instance cluster with 1 worker each
        let config = ConnectorTestConfig::new().leader_cache_blocks(128);

        let cluster = TestConnectorCluster::builder()
            .num_instances(2)
            .workers_per_instance(1)
            .test_config(config)
            .build()
            .await
            .expect("Should create cluster");

        let [decode, prefill]: &[_; 2] = cluster
            .instances()
            .try_into()
            .expect("Should have exactly 2 instances");

        println!(
            "\n=== Cluster Creation Test ===\n\
             Decode: instance={}, {} workers\n\
             Prefill: instance={}, {} workers",
            decode.instance_id(),
            decode.workers.len(),
            prefill.instance_id(),
            prefill.workers.len(),
        );

        // Verify instance leaders are accessible
        let decode_leader = decode.instance_leader().expect("Should have decode leader");
        let prefill_leader = prefill
            .instance_leader()
            .expect("Should have prefill leader");
        println!("✓ Both leaders accessible");

        // Populate blocks on each instance
        const BLOCK_SIZE: usize = 16;
        let (decode_block_ids, decode_hashes) = decode
            .populate_g2_blocks(4, BLOCK_SIZE, 0)
            .expect("Should populate Decode");
        decode
            .fill_g2_blocks(&decode_block_ids, FillPattern::Constant(0xCA))
            .expect("Should fill decode blocks");
        println!("✓ Decode populated with {} blocks", decode_block_ids.len());

        let (prefill_block_ids, prefill_hashes) = prefill
            .populate_g2_blocks(2, BLOCK_SIZE, 1000)
            .expect("Should populate Prefill");
        prefill
            .fill_g2_blocks(&prefill_block_ids, FillPattern::Constant(0xBB))
            .expect("Should fill prefill blocks");
        println!(
            "✓ Prefill populated with {} blocks",
            prefill_block_ids.len()
        );

        // Create endpoint session on Decode
        let (decode_session_id, _decode_handle) = decode_leader
            .create_endpoint_session(&decode_hashes)
            .expect("Should create decode endpoint session");
        println!("✓ Decode created endpoint session: {}", decode_session_id);

        // Prefill attaches to Decode's session
        let mut prefill_handle = prefill_leader
            .attach_session(decode.instance_id(), decode_session_id)
            .await
            .expect("Should attach to Decode");
        println!("✓ Prefill attached to Decode's session");

        // Wait for session to be ready
        let state = timeout(Duration::from_secs(5), prefill_handle.wait_for_ready())
            .await
            .expect("Timeout waiting for ready")
            .expect("Should get ready state");
        println!(
            "✓ Session ready: {} G2 blocks, phase: {:?}",
            state.g2_blocks.len(),
            state.phase
        );

        // Clean up
        prefill_handle.detach().await.ok();
        println!("✓ Prefill detached");

        // Create reverse session - Prefill exposes blocks, Decode attaches
        let (prefill_session_id, _prefill_session_handle) = prefill_leader
            .create_endpoint_session(&prefill_hashes)
            .expect("Should create prefill endpoint session");
        println!("✓ Prefill created endpoint session: {}", prefill_session_id);

        let mut decode_handle = decode_leader
            .attach_session(prefill.instance_id(), prefill_session_id)
            .await
            .expect("Should attach to Prefill");
        println!("✓ Decode attached to Prefill's session");

        let state = timeout(Duration::from_secs(5), decode_handle.wait_for_ready())
            .await
            .expect("Timeout waiting for ready")
            .expect("Should get ready state");
        println!(
            "✓ Reverse session ready: {} G2 blocks, phase: {:?}",
            state.g2_blocks.len(),
            state.phase
        );

        decode_handle.detach().await.ok();
        println!("✓ Decode detached");

        // Verify worker handlers are available (tests the handler cache fix)
        println!("\n--- Verifying Handler Cache Fix ---");
        for (i, instance) in cluster.instances().iter().enumerate() {
            for (j, worker) in instance.workers.iter().enumerate() {
                let handlers = instance
                    .leader_nova
                    .available_handlers(worker.instance_id)
                    .await
                    .expect("Should get worker handlers");

                // Verify critical worker handlers are present
                assert!(
                    handlers.contains(&"kvbm.worker.import_metadata".to_string()),
                    "Instance {} Worker {} should have kvbm.worker.import_metadata handler",
                    i,
                    j
                );
                assert!(
                    handlers.contains(&"kvbm.worker.export_metadata".to_string()),
                    "Instance {} Worker {} should have kvbm.worker.export_metadata handler",
                    i,
                    j
                );
                println!(
                    "✓ Instance {} Worker {} has {} handlers including worker handlers",
                    i,
                    j,
                    handlers.len()
                );
            }
        }

        println!("\n=== SUCCESS: Cluster, sessions, and handler cache tests passed ===");

        // Drop cluster in a blocking context to avoid runtime drop panic
        let _ = tokio::task::spawn_blocking(move || {
            drop(cluster);
        })
        .await;
    }

    /// Test bidirectional RDMA transfers using TestConnectorCluster.
    ///
    /// This replicates the pattern from test_bidirectional_layerwise_transfer
    /// using the new TestConnectorCluster abstraction.
    #[tokio::test(flavor = "multi_thread")]
    async fn test_bidirectional_rdma_transfer() {
        let _ = tracing_subscriber::fmt()
            .with_max_level(tracing::Level::INFO)
            .with_test_writer()
            .try_init();

        // Test parameters matching distributed test
        const CACHED_BLOCKS: usize = 4; // Blocks Decode has, Prefill pulls
        const NEW_BLOCKS: usize = 2; // Blocks Prefill has, Decode pulls
        const BLOCK_SIZE: usize = 16;

        // Create 2-instance cluster with 1 worker each
        let config = ConnectorTestConfig::new().leader_cache_blocks(128);

        let cluster = TestConnectorCluster::builder()
            .num_instances(2)
            .workers_per_instance(1)
            .test_config(config)
            .build()
            .await
            .expect("Should create cluster");

        let [decode, prefill]: &[_; 2] = cluster
            .instances()
            .try_into()
            .expect("Should have exactly 2 instances");

        println!(
            "\n=== Bidirectional RDMA Transfer E2E Test ===\n\
             Decode: instance={}, {} workers\n\
             Prefill: instance={}, {} workers\n\
             Cached blocks: {}, New blocks: {}",
            decode.instance_id(),
            decode.workers.len(),
            prefill.instance_id(),
            prefill.workers.len(),
            CACHED_BLOCKS,
            NEW_BLOCKS
        );

        // =====================================================================
        // Phase 1: Setup - Populate both instances with blocks
        // =====================================================================
        println!("\n--- Phase 1: Setup ---");

        let (decode_cached_block_ids, cached_hashes) = decode
            .populate_g2_blocks(CACHED_BLOCKS, BLOCK_SIZE, 0)
            .expect("Should populate Decode");
        decode
            .fill_g2_blocks(&decode_cached_block_ids, FillPattern::Constant(0xCA))
            .expect("Should fill cached blocks");
        println!(
            "Decode populated with {} cached blocks: {:?}",
            CACHED_BLOCKS, decode_cached_block_ids
        );

        let (prefill_new_block_ids, new_hashes) = prefill
            .populate_g2_blocks(NEW_BLOCKS, BLOCK_SIZE, 1000)
            .expect("Should populate Prefill");
        prefill
            .fill_g2_blocks(&prefill_new_block_ids, FillPattern::Constant(0xBB))
            .expect("Should fill new blocks");
        println!(
            "Prefill populated with {} new blocks: {:?}",
            NEW_BLOCKS, prefill_new_block_ids
        );

        // =====================================================================
        // Phase 2: Prefill Pulls Cached Blocks from Decode via RDMA
        // =====================================================================
        println!("\n--- Phase 2: Prefill Pulls from Decode ---");

        let decode_leader = decode.instance_leader().expect("Should have leader");
        let (decode_session_id, _decode_session_handle) = decode_leader
            .create_endpoint_session(&cached_hashes)
            .expect("Should create endpoint session");
        println!("Decode created endpoint session: {}", decode_session_id);

        let prefill_leader = prefill.instance_leader().expect("Should have leader");
        let mut prefill_handle = prefill_leader
            .attach_session(decode.instance_id(), decode_session_id)
            .await
            .expect("Should attach");

        let state = timeout(Duration::from_secs(5), prefill_handle.wait_for_ready())
            .await
            .expect("Timeout waiting for ready")
            .expect("Should get ready state");
        println!(
            "Prefill sees {} G2 blocks from Decode",
            state.g2_blocks.len()
        );

        // Prefill allocates destination blocks
        let prefill_dst_blocks = prefill_leader
            .g2_manager()
            .allocate_blocks(CACHED_BLOCKS)
            .expect("Should allocate destination blocks on Prefill");
        let prefill_dst_block_ids: Vec<_> =
            prefill_dst_blocks.iter().map(|b| b.block_id()).collect();
        println!(
            "Prefill allocated destination blocks: {:?}",
            prefill_dst_block_ids
        );

        // Prefill pulls cached blocks via RDMA
        let notification = prefill_handle
            .pull_blocks_rdma(&state.g2_blocks, &prefill_dst_block_ids)
            .await
            .expect("Should initiate RDMA pull");
        notification.await.expect("Transfer should complete");
        println!("Prefill pulled {} cached blocks via RDMA", CACHED_BLOCKS);

        // Verify Prefill received Decode's cached data (0xCA pattern)
        println!("Verifying Prefill received Decode's cached data...");
        let decode_checksums = decode
            .compute_g2_checksums(&decode_cached_block_ids)
            .expect("checksums");
        let prefill_checksums = prefill
            .compute_g2_checksums(&prefill_dst_block_ids)
            .expect("checksums");

        for worker_idx in 0..decode.workers.len() {
            for i in 0..CACHED_BLOCKS {
                let src_id = decode_cached_block_ids[i];
                let dst_id = prefill_dst_block_ids[i];
                assert_eq!(
                    decode_checksums[worker_idx][&src_id], prefill_checksums[worker_idx][&dst_id],
                    "Worker {}: Prefill block {} should match Decode block {}",
                    worker_idx, dst_id, src_id
                );
            }
            println!(
                "  ✓ Worker {} verified: Prefill has Decode's cached data",
                worker_idx
            );
        }

        prefill_handle.detach().await.ok();
        println!("Prefill detached (Decode keeps cached blocks)");

        // =====================================================================
        // Phase 3: Role Reversal - Decode Pulls New Blocks from Prefill via RDMA
        // =====================================================================
        println!("\n--- Phase 3: Role Reversal ---");

        let (prefill_session_id, prefill_session_handle) = prefill_leader
            .create_endpoint_session(&new_hashes)
            .expect("Should create endpoint session");
        println!("Prefill created endpoint session: {}", prefill_session_id);

        let mut decode_handle = decode_leader
            .attach_session(prefill.instance_id(), prefill_session_id)
            .await
            .expect("Should attach to Prefill's session");
        println!("Decode attached to Prefill's session");

        let state = timeout(Duration::from_secs(5), decode_handle.wait_for_ready())
            .await
            .expect("Timeout waiting for ready")
            .expect("Should get ready state");
        println!(
            "Decode sees {} G2 blocks from Prefill, phase: {:?}",
            state.g2_blocks.len(),
            state.phase
        );

        // =====================================================================
        // Phase 4: Layerwise Notification and RDMA Transfer
        // =====================================================================
        println!("\n--- Phase 4: Layerwise Transfer ---");

        let decode_dst_blocks = decode_leader
            .g2_manager()
            .allocate_blocks(NEW_BLOCKS)
            .expect("Should allocate destination blocks on Decode");
        let decode_dst_block_ids: Vec<_> = decode_dst_blocks.iter().map(|b| b.block_id()).collect();
        println!(
            "Decode allocated destination blocks: {:?}",
            decode_dst_block_ids
        );

        // Demonstrate layerwise notification (simulate compute completion)
        let num_layers = 4; // From layout config
        for layer in 0..num_layers {
            prefill_session_handle
                .notify_layers_ready(0..layer + 1)
                .await
                .expect("Should notify layer ready");
            println!("  Layer {}: Prefill notified ready", layer);
            tokio::time::sleep(Duration::from_millis(5)).await;
        }

        // After all layers ready, Decode pulls all blocks via RDMA
        println!("\n  Decode pulling all blocks via RDMA...");
        let notification = decode_handle
            .pull_blocks_rdma(&state.g2_blocks, &decode_dst_block_ids)
            .await
            .expect("Should initiate RDMA pull");
        notification.await.expect("Transfer should complete");
        println!("  Decode pulled all {} blocks via RDMA", NEW_BLOCKS);

        // Verify Decode received Prefill's data (0xBB pattern)
        println!("Verifying Decode received Prefill's data...");
        let prefill_checksums = prefill
            .compute_g2_checksums(&prefill_new_block_ids)
            .expect("checksums");
        let decode_checksums = decode
            .compute_g2_checksums(&decode_dst_block_ids)
            .expect("checksums");

        for worker_idx in 0..decode.workers.len() {
            for i in 0..NEW_BLOCKS {
                let src_id = prefill_new_block_ids[i];
                let dst_id = decode_dst_block_ids[i];
                assert_eq!(
                    prefill_checksums[worker_idx][&src_id], decode_checksums[worker_idx][&dst_id],
                    "Worker {}: Decode block {} should match Prefill block {}",
                    worker_idx, dst_id, src_id
                );
            }
            println!(
                "  ✓ Worker {} verified: Decode has Prefill's data (pattern 0xBB)",
                worker_idx
            );
        }

        // =====================================================================
        // Phase 5: Cleanup
        // =====================================================================
        println!("\n--- Phase 5: Cleanup ---");

        decode_handle
            .mark_blocks_pulled(new_hashes.clone())
            .await
            .ok();
        decode_handle.detach().await.ok();
        println!("Decode detached from Prefill's session");

        prefill_session_handle.close().await.ok();
        println!("Prefill closed endpoint session");

        println!(
            "\n=== SUCCESS: Bidirectional RDMA transfer completed ===\n\
             - {} cached blocks transferred Decode -> Prefill via RDMA\n\
             - {} new blocks transferred Prefill -> Decode via RDMA",
            CACHED_BLOCKS, NEW_BLOCKS
        );

        // Drop cluster in a blocking context to avoid runtime drop panic
        let _ = tokio::task::spawn_blocking(move || {
            drop(cluster);
        })
        .await;
    }
}
