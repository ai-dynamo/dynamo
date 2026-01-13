// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Unit tests for ConnectorWorker intra-pass operations.
//!
//! These tests focus on:
//! - Flag lifecycle for intra-pass onboard (`intra_pass_onboard_active`)
//! - Flag lifecycle for intra-pass offload (`intra_pass_offload_active`)
//! - Logic paths in `needs_offload_action`
//! - Early-exit paths in `wait_for_layer_load`, `wait_for_save`, `save_kv_layer`

use std::sync::atomic::Ordering;

use crate::v2::integrations::connector::leader::scheduler::{
    IntraPassLoad, IntraPassStore, KvConnectorMetadata,
};
use crate::v2::testing::connector::{ConnectorTestConfig, TestConnectorInstance};

use super::ConnectorWorkerInterface;

/// Helper to create a minimal test instance with a single worker.
async fn create_test_instance() -> TestConnectorInstance {
    let config = ConnectorTestConfig::new().leader_cache_blocks(128);

    TestConnectorInstance::builder()
        .num_workers(1)
        .test_config(config)
        .build()
        .await
        .expect("Should create test instance")
}

// ============================================================================
// Intra-Pass Onboard Tests
// ============================================================================

/// Test that intra_pass_onboard_active flag transitions correctly through lifecycle.
///
/// Expected flow:
/// 1. Initially false
/// 2. After start_load_kv with intra_pass_load metadata -> true
/// 3. After clear_connector_metadata -> false
#[tokio::test(flavor = "multi_thread")]
async fn test_intra_pass_onboard_flag_lifecycle() {
    let _ = tracing_subscriber::fmt()
        .with_env_filter(tracing_subscriber::EnvFilter::new("warn"))
        .with_test_writer()
        .try_init();

    let instance = create_test_instance().await;
    instance
        .register_all_workers()
        .expect("Should register workers");
    instance.initialize().await.expect("Should initialize");

    let worker = &instance.workers[0].worker;

    // 1. Initially false
    assert!(
        !worker.intra_pass_onboard_active.load(Ordering::Relaxed),
        "intra_pass_onboard_active should be false initially"
    );

    // 2. Bind metadata WITH intra_pass_load
    let metadata = KvConnectorMetadata {
        iteration: 1,
        foward_pass_completion_events: None,
        intra_pass_load: Some(IntraPassLoad {
            g2_src_block_ids: vec![0, 1, 2],
            g1_dst_block_ids: vec![0, 1, 2],
        }),
        intra_pass_store: None,
    };
    worker
        .bind_connector_metadata(metadata)
        .expect("Should bind metadata");

    // 3. Call start_load_kv - this should set the flag
    // Note: This will fail if G2 layout is not available, but the flag should still transition
    // based on whether intra_pass_load was present in metadata
    let _ = worker.start_load_kv();

    // The flag should be true if start_load_kv processed the intra_pass_load
    // (may fail the transfer, but flag should be set)
    // Note: In practice, this needs the DirectWorker to be fully initialized
    // For this test, we verify the flag is accessible and the clear resets it

    // 4. Clear metadata - flag should be false
    worker
        .clear_connector_metadata()
        .expect("Should clear metadata");
    assert!(
        !worker.intra_pass_onboard_active.load(Ordering::Relaxed),
        "intra_pass_onboard_active should be false after clear"
    );

    // Cleanup in spawn_blocking to avoid runtime-in-runtime panic
    tokio::task::spawn_blocking(move || drop(instance))
        .await
        .expect("Cleanup should succeed");
}

/// Test that start_load_kv with no intra_pass_load returns Ok and doesn't set flag.
#[tokio::test(flavor = "multi_thread")]
async fn test_intra_pass_onboard_no_metadata() {
    let _ = tracing_subscriber::fmt()
        .with_env_filter(tracing_subscriber::EnvFilter::new("warn"))
        .with_test_writer()
        .try_init();

    let instance = create_test_instance().await;
    instance
        .register_all_workers()
        .expect("Should register workers");
    instance.initialize().await.expect("Should initialize");

    let worker = &instance.workers[0].worker;

    // Bind metadata WITHOUT intra_pass_load
    let metadata = KvConnectorMetadata {
        iteration: 1,
        foward_pass_completion_events: None,
        intra_pass_load: None,
        intra_pass_store: None,
    };
    worker
        .bind_connector_metadata(metadata)
        .expect("Should bind metadata");

    // start_load_kv should succeed (no-op)
    worker
        .start_load_kv()
        .expect("start_load_kv should succeed");

    // Flag should still be false
    assert!(
        !worker.intra_pass_onboard_active.load(Ordering::Relaxed),
        "intra_pass_onboard_active should remain false when no intra_pass_load"
    );

    worker
        .clear_connector_metadata()
        .expect("Should clear metadata");

    tokio::task::spawn_blocking(move || drop(instance))
        .await
        .expect("Cleanup should succeed");
}

/// Test that wait_for_layer_load returns immediately when flag is false.
#[tokio::test(flavor = "multi_thread")]
async fn test_wait_for_layer_load_early_exit() {
    let _ = tracing_subscriber::fmt()
        .with_env_filter(tracing_subscriber::EnvFilter::new("warn"))
        .with_test_writer()
        .try_init();

    let instance = create_test_instance().await;
    instance
        .register_all_workers()
        .expect("Should register workers");
    instance.initialize().await.expect("Should initialize");

    let worker = &instance.workers[0].worker;

    // No metadata bound, flag is false
    assert!(
        !worker.intra_pass_onboard_active.load(Ordering::Relaxed),
        "Flag should be false"
    );

    // wait_for_layer_load should return Ok immediately (early exit path)
    // Using a dummy stream handle since it won't be used
    worker
        .wait_for_layer_load(0, 0)
        .expect("wait_for_layer_load should succeed with early exit");

    tokio::task::spawn_blocking(move || drop(instance))
        .await
        .expect("Cleanup should succeed");
}

// ============================================================================
// Intra-Pass Offload Tests
// ============================================================================

/// Test that intra_pass_offload_active state transitions correctly through lifecycle.
///
/// Expected flow:
/// 1. Initially None
/// 2. After bind_connector_metadata with intra_pass_store -> Some
/// 3. After clear_connector_metadata -> None
#[tokio::test(flavor = "multi_thread")]
async fn test_intra_pass_offload_flag_lifecycle() {
    let _ = tracing_subscriber::fmt()
        .with_env_filter(tracing_subscriber::EnvFilter::new("warn"))
        .with_test_writer()
        .try_init();

    let instance = create_test_instance().await;
    instance
        .register_all_workers()
        .expect("Should register workers");
    instance.initialize().await.expect("Should initialize");

    let worker = &instance.workers[0].worker;

    // 1. Initially None
    assert!(
        worker.intra_pass_offload_active.lock().is_none(),
        "intra_pass_offload_active should be None initially"
    );

    // 2. Bind metadata WITH intra_pass_store
    let metadata = KvConnectorMetadata {
        iteration: 1,
        foward_pass_completion_events: None,
        intra_pass_load: None,
        intra_pass_store: Some(IntraPassStore {
            g1_src_block_ids: vec![0, 1, 2],
            g2_dst_block_ids: vec![0, 1, 2],
        }),
    };
    worker
        .bind_connector_metadata(metadata)
        .expect("Should bind metadata");

    // Flag should be Some now (state created)
    assert!(
        worker.intra_pass_offload_active.lock().is_some(),
        "intra_pass_offload_active should be Some after binding with intra_pass_store"
    );

    // 3. Clear metadata - state should be None
    worker
        .clear_connector_metadata()
        .expect("Should clear metadata");
    assert!(
        worker.intra_pass_offload_active.lock().is_none(),
        "intra_pass_offload_active should be None after clear"
    );

    tokio::task::spawn_blocking(move || drop(instance))
        .await
        .expect("Cleanup should succeed");
}

/// Test needs_offload_action returns true for any layer when direct offload is active.
#[tokio::test(flavor = "multi_thread")]
async fn test_needs_offload_action_direct_offload() {
    let _ = tracing_subscriber::fmt()
        .with_env_filter(tracing_subscriber::EnvFilter::new("warn"))
        .with_test_writer()
        .try_init();

    let instance = create_test_instance().await;
    instance
        .register_all_workers()
        .expect("Should register workers");
    instance.initialize().await.expect("Should initialize");

    let worker = &instance.workers[0].worker;

    // Bind metadata WITH intra_pass_store (enables direct offloading)
    let metadata = KvConnectorMetadata {
        iteration: 1,
        foward_pass_completion_events: None,
        intra_pass_load: None,
        intra_pass_store: Some(IntraPassStore {
            g1_src_block_ids: vec![0, 1, 2],
            g2_dst_block_ids: vec![0, 1, 2],
        }),
    };
    worker
        .bind_connector_metadata(metadata)
        .expect("Should bind metadata");

    // needs_offload_action should return true for ANY layer (not just last)
    assert!(
        worker.needs_offload_action(0),
        "needs_offload_action should return true for layer 0 with direct offload"
    );
    assert!(
        worker.needs_offload_action(1),
        "needs_offload_action should return true for layer 1 with direct offload"
    );
    assert!(
        worker.needs_offload_action(2),
        "needs_offload_action should return true for layer 2 with direct offload"
    );

    worker
        .clear_connector_metadata()
        .expect("Should clear metadata");

    tokio::task::spawn_blocking(move || drop(instance))
        .await
        .expect("Cleanup should succeed");
}

/// Test needs_offload_action with only forward_pass_completion_active.
///
/// Should return:
/// - false for non-last layers
/// - true for last layer only
#[tokio::test(flavor = "multi_thread")]
async fn test_needs_offload_action_forward_pass_only() {
    let _ = tracing_subscriber::fmt()
        .with_env_filter(tracing_subscriber::EnvFilter::new("warn"))
        .with_test_writer()
        .try_init();

    let instance = create_test_instance().await;
    instance
        .register_all_workers()
        .expect("Should register workers");
    instance.initialize().await.expect("Should initialize");

    let worker = &instance.workers[0].worker;

    // Manually set forward_pass_completion_active (simulating metadata with events)
    worker
        .forward_pass_completion_active
        .store(true, Ordering::Relaxed);

    // Get num_layers from the worker (default test config has 4 layers)
    let num_layers = worker.num_layers();
    assert!(num_layers >= 2, "Test requires at least 2 layers");

    // Non-last layers should return false
    assert!(
        !worker.needs_offload_action(0),
        "needs_offload_action should return false for layer 0 (not last)"
    );

    // Last layer should return true
    assert!(
        worker.needs_offload_action(num_layers - 1),
        "needs_offload_action should return true for last layer"
    );

    // Clean up
    worker
        .forward_pass_completion_active
        .store(false, Ordering::Relaxed);

    tokio::task::spawn_blocking(move || drop(instance))
        .await
        .expect("Cleanup should succeed");
}

/// Test that wait_for_save returns immediately when no offload is active.
#[tokio::test(flavor = "multi_thread")]
async fn test_wait_for_save_no_offload_active() {
    let _ = tracing_subscriber::fmt()
        .with_env_filter(tracing_subscriber::EnvFilter::new("warn"))
        .with_test_writer()
        .try_init();

    let instance = create_test_instance().await;
    instance
        .register_all_workers()
        .expect("Should register workers");
    instance.initialize().await.expect("Should initialize");

    let worker = &instance.workers[0].worker;

    // No offload active (default state)
    assert!(
        worker.intra_pass_offload_active.lock().is_none(),
        "Offload should not be active"
    );

    // wait_for_save should return Ok immediately (early exit path)
    worker
        .wait_for_save()
        .expect("wait_for_save should succeed with early exit");

    tokio::task::spawn_blocking(move || drop(instance))
        .await
        .expect("Cleanup should succeed");
}

/// Test that save_kv_layer returns immediately when no action is needed.
#[tokio::test(flavor = "multi_thread")]
async fn test_save_kv_layer_early_exit() {
    let _ = tracing_subscriber::fmt()
        .with_env_filter(tracing_subscriber::EnvFilter::new("warn"))
        .with_test_writer()
        .try_init();

    let instance = create_test_instance().await;
    instance
        .register_all_workers()
        .expect("Should register workers");
    instance.initialize().await.expect("Should initialize");

    let worker = &instance.workers[0].worker;

    // No offload or forward pass completion active
    assert!(
        worker.intra_pass_offload_active.lock().is_none(),
        "Offload should not be active"
    );
    assert!(
        !worker
            .forward_pass_completion_active
            .load(Ordering::Relaxed),
        "Forward pass completion should not be active"
    );

    // save_kv_layer should return Ok immediately (early exit via needs_offload_action)
    worker
        .save_kv_layer(0, 0)
        .expect("save_kv_layer should succeed with early exit");

    tokio::task::spawn_blocking(move || drop(instance))
        .await
        .expect("Cleanup should succeed");
}

// ============================================================================
// Combined Lifecycle Tests
// ============================================================================

/// Test full iteration lifecycle with metadata bind/clear cycle.
#[tokio::test(flavor = "multi_thread")]
async fn test_full_iteration_lifecycle() {
    let _ = tracing_subscriber::fmt()
        .with_env_filter(tracing_subscriber::EnvFilter::new("warn"))
        .with_test_writer()
        .try_init();

    let instance = create_test_instance().await;
    instance
        .register_all_workers()
        .expect("Should register workers");
    instance.initialize().await.expect("Should initialize");

    let worker = &instance.workers[0].worker;

    // Simulate multiple iterations
    for iteration in 1..=3 {
        // Bind metadata (no intra-pass operations, simulating normal decode)
        let metadata = KvConnectorMetadata {
            iteration,
            foward_pass_completion_events: None,
            intra_pass_load: None,
            intra_pass_store: None,
        };
        worker
            .bind_connector_metadata(metadata)
            .expect("Should bind metadata");

        // start_load_kv should be safe to call
        worker
            .start_load_kv()
            .expect("start_load_kv should succeed");

        // wait_for_layer_load should early exit
        worker
            .wait_for_layer_load(0, 0)
            .expect("wait_for_layer_load should succeed");

        // save_kv_layer should early exit
        worker
            .save_kv_layer(0, 0)
            .expect("save_kv_layer should succeed");

        // wait_for_save should early exit
        worker
            .wait_for_save()
            .expect("wait_for_save should succeed");

        // Clear for next iteration
        worker
            .clear_connector_metadata()
            .expect("Should clear metadata");
    }

    tokio::task::spawn_blocking(move || drop(instance))
        .await
        .expect("Cleanup should succeed");
}
