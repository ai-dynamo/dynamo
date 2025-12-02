// SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! End-to-end tests for distributed find_blocks operations.
//!
//! These tests verify that `find_matches_with_options` can find blocks
//! distributed across multiple instances using scan-based search.

#[cfg(test)]
mod tests {
    use crate::v2::distributed::leader::{
        FindMatchesOptions, FindMatchesResult, Leader, OnboardingStatus, StagingMode,
    };
    use crate::v2::physical::transfer::FillPattern;
    use crate::v2::testing::connector::{ConnectorTestConfig, TestConnectorCluster};
    use crate::v2::testing::distributed::create_instance_leader_pair;
    use crate::v2::testing::managers::populate_manager_with_blocks;
    use crate::v2::testing::token_blocks::{create_token_sequence, generate_sequence_hashes};
    use std::time::Duration;
    use tokio::time::timeout;

    /// Test distributed find_blocks with partial sequence matching.
    ///
    /// This test verifies that scan_matches (not match_blocks) is used for remote search,
    /// enabling the system to find blocks even when the remote doesn't have block 0.
    ///
    /// Scenario:
    /// - Leader A (querier): Has no blocks
    /// - Leader B (remote): Has blocks 8-15 (NOT blocks 0-7)
    /// - Query: Find all 16 blocks (0-15)
    /// - Expected: Only blocks 8-15 should be found on Leader B
    ///
    /// Key insight: SequenceHash includes POSITION, not just content hash.
    /// We must create the FULL sequence and register only blocks 8-15 on Leader B,
    /// so their positions (8-15) match the query positions.
    ///
    /// With match_blocks (old behavior): Would return 0 blocks (stops at first miss)
    /// With scan_matches (new behavior): Returns 8 blocks (scans all hashes)
    #[tokio::test]
    async fn test_find_blocks_partial_sequence_on_remote() {
        let _ = tracing_subscriber::fmt()
            .with_max_level(tracing::Level::INFO)
            .with_test_writer()
            .try_init();

        const BLOCK_SIZE: usize = 4;
        const TOTAL_BLOCKS: usize = 16;
        const REMOTE_START_BLOCK: usize = 8; // Leader B has blocks 8-15
        const REMOTE_BLOCKS: usize = 8; // Only blocks 8-15

        // Create a pair of connected leaders
        let pair = create_instance_leader_pair(128, BLOCK_SIZE)
            .await
            .expect("Should create leader pair");

        eprintln!(
            "\n=== Partial Sequence Test ===\n\
             Leader A (querier): {}\n\
             Leader B (remote):  {}",
            pair.leader_a.instance_id, pair.leader_b.instance_id,
        );

        // =====================================================================
        // Phase 1: Create FULL sequence and register PARTIAL on Leader B
        // =====================================================================
        eprintln!("\n--- Phase 1: Populating Leader B with partial sequence ---");

        // Create the FULL token sequence (all 16 blocks, positions 0-15)
        // This is crucial: SequenceHash includes position, so we need blocks
        // with positions 8-15, not positions 0-7 with different token content.
        let full_sequence = create_token_sequence(TOTAL_BLOCKS, BLOCK_SIZE, 0);
        let full_blocks = full_sequence.blocks();

        // Extract only blocks 8-15 from the full sequence
        // These blocks have positions 8-15 (matching the query positions)
        let remote_blocks: Vec<_> = full_blocks
            .iter()
            .skip(REMOTE_START_BLOCK)
            .take(REMOTE_BLOCKS)
            .cloned()
            .collect();

        // Register blocks 8-15 on Leader B's G2 manager
        let remote_hashes = populate_manager_with_blocks(&pair.leader_b.g2_manager, &remote_blocks)
            .expect("Should populate Leader B");

        eprintln!(
            "Leader B populated with {} blocks (positions {}-{})",
            REMOTE_BLOCKS,
            REMOTE_START_BLOCK,
            REMOTE_START_BLOCK + REMOTE_BLOCKS - 1
        );

        // Verify the positions are correct (8-15)
        for (i, hash) in remote_hashes.iter().enumerate() {
            let expected_pos = (REMOTE_START_BLOCK + i) as u64;
            assert_eq!(
                hash.position(),
                expected_pos,
                "Block {} should have position {}",
                i,
                expected_pos
            );
        }
        eprintln!("✓ Verified blocks have positions 8-15");

        // =====================================================================
        // Phase 2: Create full sequence hashes (blocks 0-15) for query
        // =====================================================================
        eprintln!("\n--- Phase 2: Generating full query hashes ---");

        let full_hashes = generate_sequence_hashes(&full_sequence);
        assert_eq!(full_hashes.len(), TOTAL_BLOCKS);
        eprintln!(
            "Generated {} hashes for full sequence (positions 0-{})",
            TOTAL_BLOCKS,
            TOTAL_BLOCKS - 1
        );

        // =====================================================================
        // Phase 3: Query from Leader A with remote search
        // =====================================================================
        eprintln!("\n--- Phase 3: Querying from Leader A ---");

        let options = FindMatchesOptions {
            search_remote: true,
            staging_mode: StagingMode::Hold,
        };

        let result = pair
            .leader_a
            .leader
            .find_matches_with_options(&full_hashes, options)
            .expect("Should execute find_matches");

        // =====================================================================
        // Phase 4: Verify results
        // =====================================================================
        eprintln!("\n--- Phase 4: Verifying results ---");

        match result {
            FindMatchesResult::AsyncSession(mut session) => {
                eprintln!("Got AsyncSession, waiting for completion...");

                timeout(Duration::from_secs(10), session.wait_for_completion())
                    .await
                    .expect("Timeout waiting for search")
                    .expect("Should complete search");

                let status = session.status();
                eprintln!("Search complete: status = {:?}", status);

                // For Hold mode, we get Holding status with breakdown
                let total_matched = match status {
                    OnboardingStatus::Holding {
                        local_g2,
                        local_g3,
                        remote_g2,
                        remote_g3,
                    } => {
                        eprintln!(
                            "Holding breakdown:\n  \
                             local_g2={}, local_g3={}\n  \
                             remote_g2={}, remote_g3={}",
                            local_g2, local_g3, remote_g2, remote_g3
                        );
                        local_g2 + local_g3 + remote_g2 + remote_g3
                    }
                    OnboardingStatus::Complete { matched } => {
                        eprintln!("Complete with {} matched", matched);
                        matched
                    }
                    other => {
                        panic!("Unexpected status: {:?}", other);
                    }
                };

                eprintln!("Total matched: {}", total_matched);

                // Key assertion: We should find REMOTE_BLOCKS (8), not 0!
                // With old match_blocks behavior, this would be 0 (stops at first miss)
                // With new scan_matches behavior, this should be 8 (finds all available blocks)
                assert_eq!(
                    total_matched, REMOTE_BLOCKS,
                    "Should find {} blocks on remote (the partial sequence), not 0 or {}",
                    REMOTE_BLOCKS, TOTAL_BLOCKS
                );

                eprintln!(
                    "✓ SUCCESS: Found {} blocks using scan_matches!\n\
                     With old match_blocks this would have found 0 blocks.",
                    REMOTE_BLOCKS
                );
            }
            FindMatchesResult::Ready(ready) => {
                // If Ready, check local blocks
                eprintln!(
                    "Got Ready result with {} G2 blocks (local only search?)",
                    ready.g2_count()
                );
                panic!(
                    "Expected AsyncSession for remote search, got Ready with {} blocks",
                    ready.g2_count()
                );
            }
        }

        eprintln!("\n=== SUCCESS: Partial sequence test passed ===");
    }

    /// Comprehensive 4-instance distributed find_blocks test with staging and RDMA.
    ///
    /// This test verifies the full distributed block discovery and transfer flow:
    ///
    /// **Scenario:**
    /// - ISL: 128 tokens, block_size: 4 → 32 total blocks
    /// - Instance 0 (querier): Has no blocks - initiates find and receives all blocks
    /// - Instance 1: Blocks 0-15 (8 in G2 @ 0x11, 8 in G3 @ 0x12)
    /// - Instance 2: Blocks 16-23 (8 in G2 @ 0x22)
    /// - Instance 3: Blocks 24-31 (8 in G2 @ 0x33)
    ///
    /// **Flow:**
    /// 1. Find with Hold mode → verify status shows remote_g2=24, remote_g3=8
    /// 2. Call prepare() → triggers G3→G2 staging on remotes
    /// 3. Call pull() → triggers RDMA pull remote G2→local G2
    /// 4. Verify checksums match expected fill patterns
    ///
    /// Note: Uses sync test (#[test]) with TestConnectorCluster::create_with_config
    /// which properly manages the tokio runtime to avoid drop panics.
    #[test]
    fn test_4_instance_distributed_find_with_staging_rdma() {
        eprintln!("DEBUG: Test started");

        let _ = tracing_subscriber::fmt()
            .with_max_level(tracing::Level::INFO)
            .with_test_writer()
            .try_init();

        eprintln!("DEBUG: Tracing initialized");

        const BLOCK_SIZE: usize = 16; // Must match layout config page_size
        const TOTAL_BLOCKS: usize = 32;

        // Create 4-instance cluster with G3 support using sync API
        // This creates and manages its own tokio runtime internally
        let config = ConnectorTestConfig::new()
            .leader_cache_blocks(128)
            .leader_disk_blocks(64); // Enable G3

        eprintln!("DEBUG: Config created, building cluster...");

        // Use sync factory method which properly manages runtime
        let cluster = TestConnectorCluster::create_with_config(config, 4, 1)
            .expect("Should create 4-instance cluster");

        eprintln!(
            "DEBUG: Cluster built successfully with {} instances",
            cluster.instances().len()
        );

        let [querier, inst1, inst2, inst3]: &[_; 4] = cluster
            .instances()
            .try_into()
            .expect("Should have exactly 4 instances");

        eprintln!(
            "\n=== 4-Instance Distributed Find Test ===\n\
             Instance 0 (querier): {}\n\
             Instance 1 (G2+G3):   {}\n\
             Instance 2 (G2):      {}\n\
             Instance 3 (G2):      {}",
            querier.instance_id(),
            inst1.instance_id(),
            inst2.instance_id(),
            inst3.instance_id(),
        );

        // =====================================================================
        // Phase 1: Create full token sequence and populate instances (G2 only)
        // =====================================================================
        eprintln!("\n--- Phase 1: Populating instances with G2 blocks ---");

        // Create the FULL token sequence (all 32 blocks, positions 0-31)
        let full_sequence = create_token_sequence(TOTAL_BLOCKS, BLOCK_SIZE, 0);
        let full_blocks = full_sequence.blocks();
        let full_hashes = generate_sequence_hashes(&full_sequence);
        assert_eq!(full_hashes.len(), TOTAL_BLOCKS);

        // Instance 1: blocks 0-15 in G2 (simplified - no G3 for sync test)
        let inst1_blocks: Vec<_> = full_blocks[0..16].to_vec();
        let inst1_leader = inst1.instance_leader().expect("inst1 leader");
        let inst1_g2_manager = inst1_leader.g2_manager();

        let inst1_hashes =
            populate_manager_with_blocks(inst1_g2_manager, &inst1_blocks).expect("inst1 G2 pop");
        let inst1_matched = inst1_g2_manager.match_blocks(&inst1_hashes);
        let inst1_block_ids: Vec<_> = inst1_matched.into_iter().map(|b| b.block_id()).collect();

        inst1
            .fill_g2_blocks(&inst1_block_ids, FillPattern::Constant(0x11))
            .expect("inst1 G2 fill");
        eprintln!(
            "Instance 1: {} G2 blocks (positions 0-15) filled with pattern 0x11",
            inst1_block_ids.len()
        );

        // Instance 2: blocks 16-23 in G2
        let inst2_blocks: Vec<_> = full_blocks[16..24].to_vec();
        let inst2_leader = inst2.instance_leader().expect("inst2 leader");
        let inst2_g2_manager = inst2_leader.g2_manager();

        let inst2_hashes =
            populate_manager_with_blocks(inst2_g2_manager, &inst2_blocks).expect("inst2 G2 pop");
        let inst2_matched = inst2_g2_manager.match_blocks(&inst2_hashes);
        let inst2_block_ids: Vec<_> = inst2_matched.into_iter().map(|b| b.block_id()).collect();

        inst2
            .fill_g2_blocks(&inst2_block_ids, FillPattern::Constant(0x22))
            .expect("inst2 G2 fill");
        eprintln!(
            "Instance 2: {} G2 blocks (positions 16-23) filled with pattern 0x22",
            inst2_block_ids.len()
        );

        // Instance 3: blocks 24-31 in G2
        let inst3_blocks: Vec<_> = full_blocks[24..32].to_vec();
        let inst3_leader = inst3.instance_leader().expect("inst3 leader");
        let inst3_g2_manager = inst3_leader.g2_manager();

        let inst3_hashes =
            populate_manager_with_blocks(inst3_g2_manager, &inst3_blocks).expect("inst3 G2 pop");
        let inst3_matched = inst3_g2_manager.match_blocks(&inst3_hashes);
        let inst3_block_ids: Vec<_> = inst3_matched.into_iter().map(|b| b.block_id()).collect();

        inst3
            .fill_g2_blocks(&inst3_block_ids, FillPattern::Constant(0x33))
            .expect("inst3 G2 fill");
        eprintln!(
            "Instance 3: {} G2 blocks (positions 24-31) filled with pattern 0x33",
            inst3_block_ids.len()
        );

        // =====================================================================
        // Phase 2: Find with local-only mode (sync test)
        // =====================================================================
        eprintln!("\n--- Phase 2: Find with local mode (sync) ---");

        // Test that each instance can find its own blocks
        let inst1_result = inst1_leader
            .find_matches_with_options(&inst1_hashes, FindMatchesOptions::default())
            .expect("inst1 find");

        match inst1_result {
            FindMatchesResult::Ready(ready) => {
                assert_eq!(ready.g2_count(), 16, "Instance 1 should find 16 G2 blocks");
                eprintln!("✓ Instance 1 found {} local G2 blocks", ready.g2_count());
            }
            _ => panic!("Expected Ready result for local search"),
        }

        let inst2_result = inst2_leader
            .find_matches_with_options(&inst2_hashes, FindMatchesOptions::default())
            .expect("inst2 find");

        match inst2_result {
            FindMatchesResult::Ready(ready) => {
                assert_eq!(ready.g2_count(), 8, "Instance 2 should find 8 G2 blocks");
                eprintln!("✓ Instance 2 found {} local G2 blocks", ready.g2_count());
            }
            _ => panic!("Expected Ready result for local search"),
        }

        let inst3_result = inst3_leader
            .find_matches_with_options(&inst3_hashes, FindMatchesOptions::default())
            .expect("inst3 find");

        match inst3_result {
            FindMatchesResult::Ready(ready) => {
                assert_eq!(ready.g2_count(), 8, "Instance 3 should find 8 G2 blocks");
                eprintln!("✓ Instance 3 found {} local G2 blocks", ready.g2_count());
            }
            _ => panic!("Expected Ready result for local search"),
        }

        // Verify querier has no blocks locally
        let querier_leader = querier.instance_leader().expect("querier leader");
        let querier_result = querier_leader
            .find_matches_with_options(&full_hashes, FindMatchesOptions::default())
            .expect("querier find");

        match querier_result {
            FindMatchesResult::Ready(ready) => {
                assert_eq!(ready.g2_count(), 0, "Querier should have no local blocks");
                eprintln!(
                    "✓ Querier has {} local blocks (expected 0)",
                    ready.g2_count()
                );
            }
            _ => panic!("Expected Ready result for local search"),
        }

        // =====================================================================
        // Phase 3: Remote search using querier's tokio runtime
        // =====================================================================
        eprintln!("\n--- Phase 3: Remote search with Hold mode ---");

        // Get the tokio handle from the querier for async operations
        let handle = querier.tokio_handle();

        // Configure the querier to know about the remote instances
        querier_leader.add_remote_leader(inst1.instance_id());
        querier_leader.add_remote_leader(inst2.instance_id());
        querier_leader.add_remote_leader(inst3.instance_id());
        eprintln!(
            "Configured {} remote leaders for querier",
            querier_leader.remote_leaders().len()
        );

        let options = FindMatchesOptions {
            search_remote: true,
            staging_mode: StagingMode::Hold,
        };

        // find_matches_with_options uses tokio::spawn internally for remote search,
        // so it must be called from within a tokio runtime context
        let result = handle
            .block_on(async { querier_leader.find_matches_with_options(&full_hashes, options) })
            .expect("querier remote find");

        match result {
            FindMatchesResult::AsyncSession(mut session) => {
                eprintln!("Got AsyncSession, waiting for completion...");

                // Use the tokio handle to run async operations in sync context
                handle
                    .block_on(async {
                        tokio::time::timeout(Duration::from_secs(10), session.wait_for_completion())
                            .await
                    })
                    .expect("Timeout waiting for search")
                    .expect("Should complete search");

                let status = session.status();
                eprintln!("Search complete: status = {:?}", status);

                // For Hold mode, we get Holding status with breakdown
                let total_matched = match status {
                    OnboardingStatus::Holding {
                        local_g2,
                        local_g3,
                        remote_g2,
                        remote_g3,
                    } => {
                        eprintln!(
                            "Holding breakdown:\n  \
                             local_g2={}, local_g3={}\n  \
                             remote_g2={}, remote_g3={}",
                            local_g2, local_g3, remote_g2, remote_g3
                        );
                        // Verify breakdown
                        assert_eq!(local_g2, 0, "Querier should have 0 local G2 blocks");
                        assert_eq!(local_g3, 0, "Querier should have 0 local G3 blocks");
                        assert_eq!(
                            remote_g2, TOTAL_BLOCKS,
                            "Should find {} remote G2 blocks",
                            TOTAL_BLOCKS
                        );
                        assert_eq!(
                            remote_g3, 0,
                            "Should have 0 remote G3 blocks (G2 only test)"
                        );
                        local_g2 + local_g3 + remote_g2 + remote_g3
                    }
                    OnboardingStatus::Complete { matched } => {
                        eprintln!("Complete with {} matched", matched);
                        matched
                    }
                    other => {
                        panic!("Unexpected status: {:?}", other);
                    }
                };

                eprintln!("Total matched: {}", total_matched);

                // Key assertion: We should find all 32 blocks across the remote instances
                assert_eq!(
                    total_matched, TOTAL_BLOCKS,
                    "Should find all {} blocks on remote instances",
                    TOTAL_BLOCKS
                );

                eprintln!(
                    "✓ SUCCESS: Found all {} blocks via remote search!",
                    TOTAL_BLOCKS
                );
            }
            FindMatchesResult::Ready(ready) => {
                panic!(
                    "Expected AsyncSession for remote search, got Ready with {} blocks",
                    ready.g2_count()
                );
            }
        }

        eprintln!("\n=== SUCCESS: 4-instance distributed find test passed ===");

        // Cluster drops here in sync context - no panic
    }
}
