// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! RDMA integration tests for the prefill-decode pattern.
//!
//! These tests verify actual RDMA transfers using UCX backend:
//! 1. Decode creates a controllable session with local blocks
//! 2. Prefill attaches to the session
//! 3. Prefill pulls G2 blocks via RDMA
//! 4. Data integrity is verified via checksums
//!
//! Test configuration:
//! - NUM_WORKERS: 2 workers per leader (SPMD pattern)
//! - LAYOUT_BLOCKS: 64 blocks allocated per layout (2x test usage)
//! - TEST_BLOCKS: 32 blocks used in test transfers
//! - Storage: Pinned (host memory, UCX-registered for RDMA)

use anyhow::Result;
use std::time::Duration;
use tokio::time::timeout;

use dynamo_kvbm::v2::{
    distributed::leader::{ControllableSessionOptions, RemoteSessionPhase},
    physical::layout::LayoutConfig,
    testing::{distributed, physical},
};
use dynamo_memory::StorageKind;

// Test constants
const NUM_WORKERS: usize = 2;
const LAYOUT_BLOCKS: usize = 64; // Blocks allocated per layout (2x test usage)
const TEST_BLOCKS: usize = 32; // Blocks used in transfers
const BLOCK_SIZE: usize = 4; // Tokens per block
const NUM_LAYERS: usize = 3;
const OUTER_DIM: usize = 2;
const PAGE_SIZE: usize = 4;
const INNER_DIM: usize = 64;
const DTYPE_WIDTH: usize = 2; // bf16
const MANAGER_BLOCKS: usize = 64; // Blocks in G2 BlockManager (2x test usage)

/// Create layout configuration for RDMA tests.
fn test_layout_config() -> LayoutConfig {
    physical::custom_config(
        LAYOUT_BLOCKS,
        NUM_LAYERS,
        OUTER_DIM,
        PAGE_SIZE,
        INNER_DIM,
        DTYPE_WIDTH,
    )
}

/// Test metadata exchange between leaders with workers.
///
/// This tests that:
/// 1. Leaders can export worker metadata
/// 2. Leaders can import remote metadata
/// 3. Metadata import is tracked correctly
#[tokio::test(flavor = "multi_thread")]
async fn test_metadata_exchange() -> Result<()> {
    let layout_config = test_layout_config();

    // Create leader pair with workers
    let pair = distributed::create_instance_leader_pair_with_workers(
        MANAGER_BLOCKS,
        BLOCK_SIZE,
        NUM_WORKERS,
        &layout_config,
        StorageKind::Pinned,
    )
    .await?;

    // Verify worker counts
    assert_eq!(pair.decode.workers.len(), NUM_WORKERS);
    assert_eq!(pair.prefill.workers.len(), NUM_WORKERS);
    assert_eq!(pair.decode.leader.worker_count(), NUM_WORKERS);
    assert_eq!(pair.prefill.leader.worker_count(), NUM_WORKERS);

    println!(
        "Created leader pair: Decode={} ({} workers), Prefill={} ({} workers)",
        pair.decode.instance_id, NUM_WORKERS, pair.prefill.instance_id, NUM_WORKERS
    );

    // Export metadata from Decode
    let decode_metadata = pair.decode.leader.export_worker_metadata().await?;
    assert_eq!(decode_metadata.len(), NUM_WORKERS);
    println!(
        "Exported {} metadata entries from Decode",
        decode_metadata.len()
    );

    // Prefill should not have Decode's metadata yet
    assert!(
        !pair
            .prefill
            .leader
            .has_remote_metadata(pair.decode.instance_id)
    );

    // Import Decode's metadata into Prefill
    // Handles are now stored internally by the parallel worker
    pair.prefill
        .leader
        .import_remote_metadata(pair.decode.instance_id, decode_metadata)
        .await?;

    println!(
        "Imported Decode metadata into Prefill's {} workers",
        NUM_WORKERS
    );

    // Now Prefill should have Decode's metadata
    assert!(
        pair.prefill
            .leader
            .has_remote_metadata(pair.decode.instance_id)
    );

    // Attempting to import again should fail
    let decode_metadata_2 = pair.decode.leader.export_worker_metadata().await?;
    let result = pair
        .prefill
        .leader
        .import_remote_metadata(pair.decode.instance_id, decode_metadata_2)
        .await;
    assert!(result.is_err());
    println!("Second import correctly rejected: {}", result.unwrap_err());

    println!("\n=== Metadata Exchange Test Passed ===");
    Ok(())
}

/// Test basic controllable session flow with workers.
///
/// This tests that the session can be created with workers
/// and that Prefill can attach with RDMA support.
#[tokio::test(flavor = "multi_thread")]
async fn test_controllable_session_with_workers() -> Result<()> {
    let layout_config = test_layout_config();

    let pair = distributed::create_instance_leader_pair_with_workers(
        MANAGER_BLOCKS,
        BLOCK_SIZE,
        NUM_WORKERS,
        &layout_config,
        StorageKind::Pinned,
    )
    .await?;

    println!(
        "Created leader pair: Decode={}, Prefill={}",
        pair.decode.instance_id, pair.prefill.instance_id
    );

    // Populate Decode with blocks
    let (_, sequence_hashes) = distributed::populate_leader_with_blocks(
        &pair.decode.leader_as_test_instance_leader(),
        TEST_BLOCKS,
        BLOCK_SIZE,
        0,
    )?;

    println!("Populated Decode with {} blocks", TEST_BLOCKS);

    // Create controllable session on Decode
    let session_result = pair
        .decode
        .leader
        .create_controllable_session_with_options(
            &sequence_hashes,
            ControllableSessionOptions { auto_stage: false },
        )?;

    println!(
        "Decode created controllable session {}: {} G2, {} G3",
        session_result.session_id, session_result.local_g2_count, session_result.local_g3_count
    );

    assert_eq!(session_result.local_g2_count, TEST_BLOCKS);

    // Prefill attaches with RDMA support (due to workers)
    let mut handle = pair
        .prefill
        .leader
        .attach_remote_session(pair.decode.instance_id, session_result.session_id)
        .await?;

    // Wait for initial state
    let state = timeout(Duration::from_secs(5), handle.wait_for_initial_state())
        .await
        .expect("Should receive initial state within timeout")?;

    println!(
        "Prefill received initial state: phase={:?}, g2_blocks={}",
        state.phase,
        state.g2_blocks.len()
    );

    assert!(
        state.phase == RemoteSessionPhase::Attached || state.phase == RemoteSessionPhase::Ready,
        "Should be in Attached or Ready phase"
    );
    assert_eq!(state.g2_blocks.len(), TEST_BLOCKS);

    // Clean up
    handle.detach().await?;

    println!("\n=== Controllable Session with Workers Test Passed ===");
    Ok(())
}

// Helper to get TestInstanceLeader from TestInstanceLeaderWithWorkers
trait AsTestInstanceLeader {
    fn leader_as_test_instance_leader(&self) -> distributed::TestInstanceLeader;
}

impl AsTestInstanceLeader for distributed::TestInstanceLeaderWithWorkers {
    fn leader_as_test_instance_leader(&self) -> distributed::TestInstanceLeader {
        distributed::TestInstanceLeader {
            instance_id: self.instance_id,
            leader: self.leader.clone(),
            g2_manager: self.g2_manager.clone(),
            g3_manager: self.g3_manager.clone(),
        }
    }
}

// Note: The full RDMA transfer test with checksum verification is in the
// internal testing module: lib/kvbm/src/v2/testing/distributed.rs
// (test_rdma_transfer_with_checksum_verification)
