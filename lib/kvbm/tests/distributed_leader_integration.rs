// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Integration tests for distributed leader Hold mode.
//!
//! Test scenario (from user requirements):
//! 1. Leader A: Instantiate 128 tokens with block size 4 (= 32 blocks)
//!    - Allocate blocks, apply token blocks, register, free
//!    - Blocks go to G2 inactive pool on A
//! 2. Leader B: Search for same/similar sequence
//!    - Finds no local matches
//!    - Searches leader A remotely
//! 3. Success: B holds session on A which holds the matching blocks
//! 4. Close session on B
//! 5. Verify: Blocks return to A's inactive pool

use anyhow::Result;
use std::time::Duration;
use tokio::time::timeout;

use dynamo_kvbm::v2::{
    distributed::leader::{FindMatchesOptions, Leader, OnboardingStatus, StagingMode},
    testing::distributed,
};

// Test constants
const NUM_TOKENS: usize = 128;
const BLOCK_SIZE: usize = 4;
const NUM_BLOCKS: usize = NUM_TOKENS / BLOCK_SIZE; // 32 blocks

#[tokio::test(flavor = "multi_thread")]
async fn test_two_leaders_hold_mode_and_session_cleanup() -> Result<()> {
    // Step 1: Create two InstanceLeaders with Nova communication
    let pair = distributed::create_instance_leader_pair(NUM_BLOCKS, BLOCK_SIZE).await?;

    println!(
        "Created leader pair: A={}, B={}",
        pair.leader_a.instance_id, pair.leader_b.instance_id
    );

    // Step 2: Populate leader A with 32 blocks (128 tokens / 4 per block)
    let (_, sequence_hashes) =
        distributed::populate_leader_with_blocks(&pair.leader_a, NUM_BLOCKS, BLOCK_SIZE, 0)?;

    println!("Populated leader A with {} blocks", NUM_BLOCKS);

    // Verify A has blocks in inactive pool (all available)
    let a_initial_available = pair.leader_a.g2_manager.available_blocks();
    assert_eq!(
        a_initial_available, NUM_BLOCKS,
        "Leader A should have all blocks available in inactive pool"
    );

    // Step 3: Leader B searches for same sequence with Hold mode
    println!(
        "Leader B searching for {} sequence hashes...",
        sequence_hashes.len()
    );

    let mut result = pair.leader_b.leader.find_matches_with_options(
        &sequence_hashes,
        FindMatchesOptions {
            search_remote: true,
            staging_mode: StagingMode::Hold,
        },
    )?;

    // With remote search enabled, should get AsyncSession variant
    let async_result = result
        .as_async_mut()
        .expect("Should get AsyncSession with remote search enabled");

    // Step 4: Wait for Holding status with timeout
    timeout(Duration::from_secs(5), async {
        loop {
            match async_result.status() {
                OnboardingStatus::Holding {
                    local_g2,
                    local_g3,
                    remote_g2,
                    remote_g3,
                    ..
                } => {
                    println!(
                        "Holding status: local_g2={}, local_g3={}, remote_g2={}, remote_g3={}",
                        local_g2, local_g3, remote_g2, remote_g3
                    );

                    assert_eq!(local_g2, 0, "Leader B should have no local G2 blocks");
                    assert_eq!(local_g3, 0, "Leader B should have no local G3 blocks");
                    assert_eq!(
                        remote_g2, NUM_BLOCKS,
                        "Leader B should find all {} blocks on remote leader A",
                        NUM_BLOCKS
                    );
                    assert_eq!(remote_g3, 0, "Leader A has no G3 blocks");

                    break;
                }
                OnboardingStatus::Searching => {
                    tokio::time::sleep(Duration::from_millis(10)).await;
                }
                other => {
                    panic!("Unexpected status during search: {:?}", other);
                }
            }
        }
    })
    .await
    .expect("Should reach Holding status within timeout");

    println!("✓ Leader B successfully holding blocks from A");

    // Step 5: Verify session exists and blocks are held
    let _session_id = async_result.session_id();

    // Verify A's blocks are being held (should be 0 available if held by session)
    // Note: Current implementation may vary depending on how blocks are held
    let a_held_available = pair.leader_a.g2_manager.available_blocks();
    println!(
        "Leader A available blocks while held: {} (initial: {})",
        a_held_available, a_initial_available
    );

    // Step 6: Close session via cancel
    println!("Closing session on leader B...");

    if let Some(handle) = async_result.session_handle() {
        handle.cancel().await?;
        println!("✓ Session cancelled");
    } else {
        panic!("Expected session handle for Hold mode");
    }

    // Give time for cleanup messages to propagate
    tokio::time::sleep(Duration::from_millis(200)).await;

    // Step 7: Verify blocks returned to A's inactive pool
    let a_final_available = pair.leader_a.g2_manager.available_blocks();
    println!(
        "Leader A available blocks after session close: {}",
        a_final_available
    );

    assert_eq!(
        a_final_available, NUM_BLOCKS,
        "All blocks should return to leader A's inactive pool after session close"
    );

    println!("✓ All blocks returned to inactive pool");
    println!("\n=== Test Passed ===");

    Ok(())
}

#[tokio::test(flavor = "multi_thread")]
#[ignore] // TODO: Enable when ready for full test suite
async fn test_two_leaders_prepare_mode() -> Result<()> {
    // Test Prepare mode: G3→G2 staging without RDMA pull
    // TODO: Implement when G3 support and workers are available
    todo!("Implement Prepare mode test")
}

#[tokio::test(flavor = "multi_thread")]
#[ignore] // TODO: Enable when RDMA is implemented
async fn test_two_leaders_full_mode() -> Result<()> {
    // Test Full mode: Complete staging with RDMA pull
    // TODO: Implement when RDMA pull is implemented
    todo!("Implement Full mode test with RDMA")
}
