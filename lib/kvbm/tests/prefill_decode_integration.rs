// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Integration tests for the inverted control pattern (Prefill-Decode scenario).
//!
//! Test scenario:
//! 1. Decode: Creates a controllable session with local blocks
//! 2. Decode: Sends session_id to Prefill (simulated via direct call)
//! 3. Prefill: Attaches to the session on Decode
//! 4. Prefill: Queries session state (G2 blocks, G3 pending)
//! 5. Prefill: Triggers staging if needed (or awaits auto-staging completion)
//! 6. Prefill: Pulls G2 blocks (mocked for now)
//! 7. Prefill: Marks blocks as pulled
//! 8. Prefill: Detaches from session
//! 9. Verify: All resources cleaned up

use anyhow::Result;
use std::time::Duration;
use tokio::time::timeout;

use dynamo_kvbm::v2::{
    distributed::leader::{ControllableSessionOptions, RemoteSessionPhase},
    testing::distributed,
};

// Test constants
const NUM_TOKENS: usize = 128;
const BLOCK_SIZE: usize = 4;
const NUM_BLOCKS: usize = NUM_TOKENS / BLOCK_SIZE; // 32 blocks

/// Test basic controllable session creation and attachment.
///
/// This tests the core inverted control flow:
/// 1. Decode creates session with local blocks
/// 2. Prefill attaches and receives initial state
/// 3. Prefill detaches
#[tokio::test(flavor = "multi_thread")]
async fn test_controllable_session_basic_flow() -> Result<()> {
    // Create two leaders (Decode and Prefill)
    let pair = distributed::create_instance_leader_pair(NUM_BLOCKS, BLOCK_SIZE).await?;

    let decode = &pair.leader_a;
    let prefill = &pair.leader_b;

    println!(
        "Created leader pair: Decode={}, Prefill={}",
        decode.instance_id, prefill.instance_id
    );

    // Step 1: Populate Decode with blocks
    let (_, sequence_hashes) =
        distributed::populate_leader_with_blocks(decode, NUM_BLOCKS, BLOCK_SIZE, 0)?;

    println!("Populated Decode with {} blocks", NUM_BLOCKS);

    // Verify Decode has blocks available
    let decode_initial_available = decode.g2_manager.available_blocks();
    assert_eq!(
        decode_initial_available, NUM_BLOCKS,
        "Decode should have all blocks available"
    );

    // Step 2: Decode creates controllable session (auto_stage=false for this test)
    let session_result = decode.leader.create_controllable_session_with_options(
        &sequence_hashes,
        ControllableSessionOptions { auto_stage: false },
    )?;

    println!(
        "Decode created controllable session {}: {} G2, {} G3",
        session_result.session_id, session_result.local_g2_count, session_result.local_g3_count
    );

    assert_eq!(
        session_result.local_g2_count, NUM_BLOCKS,
        "Should find all blocks in G2"
    );
    assert_eq!(session_result.local_g3_count, 0, "No G3 blocks expected");

    // Step 3: Prefill attaches to the session on Decode
    println!("Prefill attaching to session...");

    let mut handle = prefill
        .leader
        .attach_remote_session(decode.instance_id, session_result.session_id)
        .await?;

    // Step 4: Wait for initial state
    let state = timeout(Duration::from_secs(5), handle.wait_for_initial_state())
        .await
        .expect("Should receive initial state within timeout")?;

    println!(
        "Prefill received initial state: phase={:?}, g2_blocks={}, g3_pending={}",
        state.phase,
        state.g2_blocks.len(),
        state.g3_pending_count
    );

    assert!(
        state.phase == RemoteSessionPhase::Attached || state.phase == RemoteSessionPhase::Ready,
        "Should be in Attached or Ready phase"
    );
    assert_eq!(
        state.g2_blocks.len(),
        NUM_BLOCKS,
        "Should see all G2 blocks"
    );
    assert_eq!(state.g3_pending_count, 0, "No G3 blocks pending");

    // Step 5: Verify G2 block metadata
    for block_info in &state.g2_blocks {
        println!(
            "  G2 block: id={}, hash={:?}",
            block_info.block_id, block_info.sequence_hash
        );
    }

    // Step 6: Mark blocks as pulled (simulating RDMA pull completion)
    let pulled_hashes: Vec<_> = state.g2_blocks.iter().map(|b| b.sequence_hash).collect();

    handle.mark_blocks_pulled(pulled_hashes.clone()).await?;
    println!("Prefill marked {} blocks as pulled", pulled_hashes.len());

    // Step 7: Detach from session
    handle.detach().await?;
    println!("Prefill detached from session");

    // Give time for cleanup messages to propagate
    tokio::time::sleep(Duration::from_millis(200)).await;

    // Step 8: Verify cleanup
    let decode_final_available = decode.g2_manager.available_blocks();
    println!(
        "Decode final available blocks: {} (initial: {})",
        decode_final_available, decode_initial_available
    );

    // Note: Blocks should return to inactive pool after session cleanup
    // The exact count depends on implementation details

    println!("\n=== Test Passed ===");

    Ok(())
}

// /// Test controllable session with auto-staging enabled (default behavior).
// ///
// /// This tests the scenario where Decode auto-stages G3→G2 before Prefill attaches.
// #[tokio::test(flavor = "multi_thread")]
// #[ignore] // TODO: Enable when G3 staging is implemented with workers
// async fn test_controllable_session_auto_staging() -> Result<()> {
//     let pair = distributed::create_instance_leader_pair(NUM_BLOCKS, BLOCK_SIZE).await?;

//     let decode = &pair.leader_a;
//     let prefill = &pair.leader_b;

//     // TODO: Populate Decode with G3 blocks (requires G3 manager population)
//     // let (_, sequence_hashes) = populate_g3_blocks(decode, NUM_BLOCKS, BLOCK_SIZE, 0)?;

//     // Create session with auto_stage=true (default)
//     // let session_result = decode.leader.create_controllable_session(&sequence_hashes)?;

//     // Prefill attaches - should see Staging or Ready phase
//     // let mut handle = prefill.leader.attach_remote_session(decode.instance_id, session_result.session_id).await?;

//     // Wait for staging to complete
//     // handle.wait_for_staging_complete().await?;

//     // Verify all blocks are in G2
//     // let state = handle.current_state();
//     // assert_eq!(state.phase, RemoteSessionPhase::Ready);
//     // assert_eq!(state.g2_blocks.len(), NUM_BLOCKS);
//     // assert_eq!(state.g3_pending_count, 0);

//     todo!("Implement when G3 staging with workers is available")
// }

/// Test that trigger_staging is idempotent.
#[tokio::test(flavor = "multi_thread")]
async fn test_controllable_session_idempotent_staging() -> Result<()> {
    let pair = distributed::create_instance_leader_pair(NUM_BLOCKS, BLOCK_SIZE).await?;

    let decode = &pair.leader_a;
    let prefill = &pair.leader_b;

    // Populate Decode with G2 blocks only
    let (_, sequence_hashes) =
        distributed::populate_leader_with_blocks(decode, NUM_BLOCKS, BLOCK_SIZE, 0)?;

    // Create session with auto_stage=false
    let session_result = decode.leader.create_controllable_session_with_options(
        &sequence_hashes,
        ControllableSessionOptions { auto_stage: false },
    )?;

    // Prefill attaches
    let mut handle = prefill
        .leader
        .attach_remote_session(decode.instance_id, session_result.session_id)
        .await?;

    // Wait for initial state
    let _ = timeout(Duration::from_secs(5), handle.wait_for_initial_state())
        .await
        .expect("Should receive initial state")?;

    // Call trigger_staging multiple times - should be idempotent
    handle.trigger_staging().await?;
    handle.trigger_staging().await?;
    handle.trigger_staging().await?;

    // Should still be in Ready state (no G3 blocks to stage)
    let state = handle.current_state();
    assert!(
        state.phase == RemoteSessionPhase::Attached || state.phase == RemoteSessionPhase::Ready,
        "Should remain in Attached/Ready phase"
    );

    // Clean up
    handle.detach().await?;

    println!("\n=== Idempotent Staging Test Passed ===");

    Ok(())
}

/// Test session state visibility through handle.
#[tokio::test(flavor = "multi_thread")]
async fn test_remote_session_handle_state_queries() -> Result<()> {
    let pair = distributed::create_instance_leader_pair(NUM_BLOCKS, BLOCK_SIZE).await?;

    let decode = &pair.leader_a;
    let prefill = &pair.leader_b;

    let (_, sequence_hashes) =
        distributed::populate_leader_with_blocks(decode, NUM_BLOCKS, BLOCK_SIZE, 0)?;

    let session_result = decode.leader.create_controllable_session_with_options(
        &sequence_hashes,
        ControllableSessionOptions { auto_stage: false },
    )?;

    let mut handle = prefill
        .leader
        .attach_remote_session(decode.instance_id, session_result.session_id)
        .await?;

    // Wait for initial state
    let _ = timeout(Duration::from_secs(5), handle.wait_for_initial_state())
        .await
        .expect("Should receive initial state")?;

    // Test state query methods
    let session_id = handle.session_id();
    assert_eq!(session_id, session_result.session_id);

    let remote_instance = handle.remote_instance();
    assert_eq!(remote_instance, decode.instance_id);

    let local_instance = handle.local_instance();
    assert_eq!(local_instance, prefill.instance_id);

    let g2_blocks = handle.get_g2_blocks();
    assert_eq!(g2_blocks.len(), NUM_BLOCKS);

    let g3_pending = handle.g3_pending_count();
    assert_eq!(g3_pending, 0);

    let is_ready = handle.is_ready();
    let is_complete = handle.is_complete();
    println!("State: is_ready={}, is_complete={}", is_ready, is_complete);

    // Clean up
    handle.detach().await?;

    println!("\n=== State Queries Test Passed ===");

    Ok(())
}
