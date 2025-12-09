// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! G4 (Object Storage) transfer tests.
//!
//! These tests verify the `execute_g4_offload` and `execute_g4_onboard` functionality
//! for transferring KV cache blocks to/from object storage via NIXL's OBJ plugin.
//!
//! # Requirements
//!
//! These tests require:
//! - NIXL built with OBJ plugin support
//! - S3-compatible object storage
//! - Proper environment configuration for the OBJ backend
//!
//! # Running
//!
//! ```bash
//! # Run with ignored tests enabled
//! cargo test -p dynamo-kvbm g4_transfers -- --ignored
//! ```

use anyhow::Result;

use crate::v2::{
    SequenceHash,
    distributed::worker::RemoteDescriptor,
    physical::{
        layout::{LayoutConfig, PhysicalLayout},
        manager::TransferManager,
        transfer::TransferOptions,
    },
};
use dynamo_memory::nixl::{NixlAgent, NixlBackendConfig};

/// Create a test SequenceHash from an index.
///
/// This creates a PositionalSequenceHash with:
/// - sequence_hash = index as lower 64 bits
/// - position = 0
/// - local_block_hash = 0
fn test_sequence_hash(index: u64) -> SequenceHash {
    SequenceHash::new(index, 0, 0)
}

/// Try to create a NIXL agent with OBJ backend.
///
/// Returns `Ok(Some(agent))` if OBJ plugin is available.
/// Returns `Ok(None)` if OBJ plugin is not installed.
/// Returns `Err` for other failures.
///
/// Reads backend parameters from environment variables:
/// - `DYN_KVBM_NIXL_BACKEND_OBJ=true` - enable OBJ backend
/// - `DYN_KVBM_NIXL_BACKEND_OBJ_BUCKET=name` - S3 bucket
/// - `DYN_KVBM_NIXL_BACKEND_OBJ_ENDPOINT_OVERRIDE=url` - custom S3 endpoint
fn try_create_obj_agent(name: &str) -> Result<Option<NixlAgent>> {
    // Read backend config from environment, then ensure required backends are enabled
    let mut backend_config = NixlBackendConfig::from_env()?;

    // OBJ backend for object storage
    if !backend_config.has_backend("OBJ") {
        backend_config = backend_config.with_backend("OBJ");
    }

    // POSIX backend for DRAM memory handling (required for DRAM→OBJ transfers)
    if !backend_config.has_backend("POSIX") {
        backend_config = backend_config.with_backend("POSIX");
    }

    match NixlAgent::from_nixl_backend_config(name, backend_config) {
        Ok(agent) => Ok(Some(agent)),
        Err(e) => {
            let err_msg = e.to_string();
            // Check if error is due to missing plugin
            if err_msg.contains("No OBJ plugin found") || err_msg.contains("plugin") {
                Ok(None)
            } else {
                Err(e)
            }
        }
    }
}

/// Create a standard test layout configuration.
fn test_config(num_blocks: usize) -> LayoutConfig {
    LayoutConfig::builder()
        .num_blocks(num_blocks)
        .num_layers(2)
        .outer_dim(2)
        .page_size(16)
        .inner_dim(128)
        .dtype_width_bytes(2)
        .build()
        .unwrap()
}

/// Create a test layout with system memory.
fn create_system_layout(agent: NixlAgent, num_blocks: usize) -> PhysicalLayout {
    PhysicalLayout::builder(agent)
        .with_config(test_config(num_blocks))
        .fully_contiguous()
        .allocate_system()
        .build()
        .unwrap()
}

/// Create a test layout with pinned memory.
fn create_pinned_layout(agent: NixlAgent, num_blocks: usize) -> PhysicalLayout {
    PhysicalLayout::builder(agent)
        .with_config(test_config(num_blocks))
        .fully_contiguous()
        .allocate_pinned(false)
        .build()
        .unwrap()
}

#[tokio::test]
#[ignore] // Requires NIXL OBJ backend and S3-compatible storage
async fn test_g4_offload_basic() {
    // This test verifies basic G4 offload functionality:
    // 1. Create a TransferManager with OBJ backend support
    // 2. Register a source layout with test data
    // 3. Execute offload to object storage
    // 4. Verify the transfer completes successfully

    // Try to create agent with OBJ backend
    let agent = match try_create_obj_agent("test-g4-offload") {
        Ok(Some(agent)) => agent,
        Ok(None) => {
            eprintln!("Skipping test: OBJ plugin not available");
            return;
        }
        Err(e) => panic!("Failed to create agent: {}", e),
    };

    let num_blocks = 4;
    let layout = create_system_layout(agent.clone(), num_blocks);

    // Create TransferManager
    let manager = TransferManager::builder()
        .nixl_agent(agent)
        .build()
        .expect("Failed to create manager");

    // Register the source layout
    let src_handle = manager
        .register_layout(layout)
        .expect("Failed to register layout");

    // Create destination descriptor for object storage
    // Each block maps to a unique sequence hash (object key)
    let dst_keys: Vec<SequenceHash> = (0..num_blocks)
        .map(|i| test_sequence_hash(i as u64 + 1000))
        .collect();

    let dst_descriptor = RemoteDescriptor::Object { keys: dst_keys };

    // Source block IDs
    let src_blocks: Vec<usize> = (0..num_blocks).collect();

    // Execute the offload
    let options = TransferOptions::default();
    let notification = manager
        .execute_g4_offload(src_handle, &src_blocks, dst_descriptor, options)
        .expect("Failed to execute G4 offload");

    // Wait for completion
    notification.await.expect("Transfer failed");
}

#[tokio::test]
#[ignore] // Requires NIXL OBJ backend and S3-compatible storage
async fn test_g4_offload_with_tp_rank() {
    // This test verifies G4 offload with tensor parallelism configuration.
    // In a multi-worker TP scenario, each worker handles a portion of the data.

    let agent = match try_create_obj_agent("test-g4-tp") {
        Ok(Some(agent)) => agent,
        Ok(None) => {
            eprintln!("Skipping test: OBJ plugin not available");
            return;
        }
        Err(e) => panic!("Failed to create agent: {}", e),
    };

    let num_blocks = 2;
    let layout = create_pinned_layout(agent.clone(), num_blocks);

    let manager = TransferManager::builder()
        .nixl_agent(agent)
        .build()
        .expect("Failed to create manager");

    let src_handle = manager
        .register_layout(layout)
        .expect("Failed to register layout");

    let dst_keys: Vec<SequenceHash> = (0..num_blocks)
        .map(|i| SequenceHash::new(i as u64 + 2000, 0, 0))
        .collect();

    let dst_descriptor = RemoteDescriptor::Object { keys: dst_keys };
    let src_blocks: Vec<usize> = (0..num_blocks).collect();

    // Configure TP options: worker 0 of 4
    let options = TransferOptions::builder()
        .tp_rank(0)
        .tp_size(4)
        .build()
        .unwrap();

    let notification = manager
        .execute_g4_offload(src_handle, &src_blocks, dst_descriptor, options)
        .expect("Failed to execute G4 offload with TP config");

    notification.await.expect("Transfer failed");
}

#[tokio::test]
#[ignore] // Requires NIXL OBJ backend and S3-compatible storage
async fn test_g4_offload_layer_range() {
    // This test verifies G4 offload with a specific layer range.
    // Only transfers layers 0..1 instead of all layers.

    let agent = match try_create_obj_agent("test-g4-layers") {
        Ok(Some(agent)) => agent,
        Ok(None) => {
            eprintln!("Skipping test: OBJ plugin not available");
            return;
        }
        Err(e) => panic!("Failed to create agent: {}", e),
    };

    let num_blocks = 2;
    let layout = create_system_layout(agent.clone(), num_blocks);

    let manager = TransferManager::builder()
        .nixl_agent(agent)
        .build()
        .expect("Failed to create manager");

    let src_handle = manager
        .register_layout(layout)
        .expect("Failed to register layout");

    let dst_keys: Vec<SequenceHash> = (0..num_blocks)
        .map(|i| SequenceHash::new(i as u64 + 3000, 0, 0))
        .collect();

    let dst_descriptor = RemoteDescriptor::Object { keys: dst_keys };
    let src_blocks: Vec<usize> = (0..num_blocks).collect();

    // Only transfer layer 0
    let options = TransferOptions::builder()
        .layer_range(0..1)
        .build()
        .unwrap();

    let notification = manager
        .execute_g4_offload(src_handle, &src_blocks, dst_descriptor, options)
        .expect("Failed to execute G4 offload with layer range");

    notification.await.expect("Transfer failed");
}

#[tokio::test]
#[ignore] // Requires NIXL OBJ backend and S3-compatible storage
async fn test_g4_offload_to_layout_fallback() {
    // This test verifies that G4 offload with RemoteDescriptor::Layout
    // falls back to the standard execute_transfer path.

    // This test doesn't need OBJ backend since it uses Layout descriptor (fallback path)
    let agent = NixlAgent::new("test-g4-fallback").expect("Failed to create agent");

    let num_blocks = 2;
    let src_layout = create_system_layout(agent.clone(), num_blocks);
    let dst_layout = create_pinned_layout(agent.clone(), num_blocks);

    let manager = TransferManager::builder()
        .nixl_agent(agent)
        .build()
        .expect("Failed to create manager");

    let src_handle = manager
        .register_layout(src_layout)
        .expect("Failed to register source");
    let dst_handle = manager
        .register_layout(dst_layout)
        .expect("Failed to register destination");

    // Use Layout variant instead of Object
    let dst_descriptor = RemoteDescriptor::Layout {
        handle: dst_handle,
        block_ids: (0..num_blocks).collect(),
    };

    let src_blocks: Vec<usize> = (0..num_blocks).collect();
    let options = TransferOptions::default();

    let notification = manager
        .execute_g4_offload(src_handle, &src_blocks, dst_descriptor, options)
        .expect("Failed to execute G4 offload (layout fallback)");

    notification.await.expect("Transfer failed");
}

#[test]
fn test_g4_offload_block_count_mismatch() {
    // This test verifies that mismatched block counts are rejected.
    // No NIXL OBJ backend required - tests validation logic only.

    let agent = NixlAgent::new("test-g4-mismatch").expect("Failed to create agent");

    let num_blocks = 4;
    let layout = create_system_layout(agent.clone(), num_blocks);

    let manager = TransferManager::builder()
        .nixl_agent(agent)
        .build()
        .expect("Failed to create manager");

    let src_handle = manager
        .register_layout(layout)
        .expect("Failed to register layout");

    // Mismatch: 4 source blocks but only 2 destination keys
    let dst_keys: Vec<SequenceHash> = vec![
        SequenceHash::new(100, 0, 0),
        SequenceHash::new(200, 0, 0),
    ];

    let dst_descriptor = RemoteDescriptor::Object { keys: dst_keys };
    let src_blocks: Vec<usize> = (0..num_blocks).collect();
    let options = TransferOptions::default();

    let result = manager.execute_g4_offload(src_handle, &src_blocks, dst_descriptor, options);

    match result {
        Ok(_) => panic!("Expected mismatch error, but got Ok"),
        Err(e) => {
            let err = e.to_string();
            assert!(
                err.contains("mismatch"),
                "Expected mismatch error, got: {}",
                err
            );
        }
    }
}
