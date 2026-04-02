// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! No-CUDA integration tests for remote (S3 / disk) block transfers.
//!
//! Gated on `testing-nixl` + `testing-remote-storage` features so they only
//! run in environments where NIXL and an S3/disk backend are available.
//!
//! Run disk tests:  `cargo test ... remote_tests::disk`
//! Run S3 tests:    `cargo test ... remote_tests::s3`

use std::sync::Arc;

use nixl_sys::Agent as NixlAgent;
use tokio_util::sync::CancellationToken;

use crate::block_manager::{
    LayoutConfig,
    block::{BasicMetadata, Block, BlockData, locality},
    block::data::{BlockDataExt, BlockDataProvider, BlockDataProviderMut},
    config::{RemoteStorageConfig, RemoteTransferContext, DISK_FLAGS_POSIX_BOTH},
    layout::{FullyContiguous, nixl::NixlLayout},
    storage::SystemAllocator,
};
use crate::block_manager::block::transfer::context::TransferContext;
use crate::block_manager::block::transfer::TransferError;

use super::nixl::execute_remote_transfer;
use super::remote::{RemoteBlockDescriptor, RemoteStorageKind, RemoteTransferDirection};

// ── Shared NIXL agents (POSIX + OBJ, no CUDA) ────────────────────────────────
//
// All agents are static lazy-initialized so that OBJ backend creation —
// which touches the global AWS SDK client — happens exactly once per agent
// and never concurrently with another agent's initialization. Concurrent
// `create_backend("OBJ", …)` calls in the same process would corrupt the
// shared S3-client state and cause NIXL_ERR_BACKEND on transfers.

/// Register an OBJ backend on an existing NixlAgent, pointing at `bucket`.
///
/// Reads `KVBM_TEST_S3_ENDPOINT`, `AWS_ACCESS_KEY_ID`, and
/// `AWS_SECRET_ACCESS_KEY` from the environment when present.
fn register_obj_backend(agent: &NixlAgent, agent_name: &str, bucket: &str) {
    if let Ok((_, params)) = agent.get_plugin_params("OBJ") {
        let result = params.clone().and_then(|mut p| {
            p.set("bucket", bucket)?;
            if let Ok(endpoint) = std::env::var(dynamo_runtime::config::environment_names::kvbm::testing::KVBM_TEST_S3_ENDPOINT) {
                // endpoint_override expects host:port only; scheme is a separate param.
                let (scheme, host_port) = if let Some(h) = endpoint.strip_prefix("http://") {
                    ("http", h)
                } else if let Some(h) = endpoint.strip_prefix("https://") {
                    ("https", h)
                } else {
                    ("https", endpoint.as_str())
                };
                p.set("endpoint_override", host_port)?;
                p.set("scheme", scheme)?;
            }
            // Pass credentials explicitly so the AWS SDK does not fall back to
            // the EC2 instance-metadata service (which would time out).
            if let Ok(key) = std::env::var("AWS_ACCESS_KEY_ID") {
                p.set("access_key", &key)?;
            }
            if let Ok(secret) = std::env::var("AWS_SECRET_ACCESS_KEY") {
                p.set("secret_key", &secret)?;
            }
            agent.create_backend("OBJ", &p)
        });
        match result {
            Ok(_) => eprintln!("[nocuda] OBJ backend ready ({agent_name} → {bucket})"),
            Err(e) => eprintln!("[nocuda] OBJ backend failed ({agent_name} → {bucket}): {e}"),
        }
    }
}

/// Create a NixlAgent with an OBJ backend pointed at `bucket`.
///
/// Used by TP-isolation tests that need a per-worker agent; each call
/// produces a fresh agent independent of the shared `TEST_AGENT`.
fn create_obj_agent(agent_name: &str, bucket: &str) -> Arc<Option<NixlAgent>> {
    let agent = NixlAgent::new(agent_name).expect("NixlAgent::new failed");
    register_obj_backend(&agent, agent_name, bucket);
    Arc::new(Some(agent))
}

lazy_static::lazy_static! {
    /// Shared agent with both POSIX and OBJ backends for the single-worker tests.
    static ref TEST_AGENT: Arc<Option<NixlAgent>> = {
        let agent = NixlAgent::new("kvbm-nocuda-test").expect("NixlAgent::new failed");

        if let Ok((_, params)) = agent.get_plugin_params("POSIX") {
            match agent.create_backend("POSIX", &params) {
                Ok(_)  => eprintln!("[nocuda] POSIX backend ready"),
                Err(e) => eprintln!("[nocuda] POSIX backend failed: {e}"),
            }
        }

        let bucket = std::env::var(dynamo_runtime::config::environment_names::kvbm::testing::KVBM_TEST_S3_BUCKET)
            .unwrap_or_else(|_| "kvbm-test-bucket".into());
        register_obj_backend(&agent, "kvbm-nocuda-test", &bucket);

        Arc::new(Some(agent))
    };
}

// ── Helpers ───────────────────────────────────────────────────────────────────

/// Build a `TransferContext` using the given agent (no CUDA).
fn make_transfer_ctx_for(agent: Arc<Option<NixlAgent>>) -> Arc<TransferContext> {
    let handle = tokio::runtime::Handle::current();
    Arc::new(TransferContext::without_cuda(agent, handle))
}

/// Build a `TransferContext` backed by the shared `TEST_AGENT`.
fn make_transfer_ctx() -> Arc<TransferContext> {
    make_transfer_ctx_for(TEST_AGENT.clone())
}

/// Allocate a `FullyContiguous<SystemStorage>` layout, register it with the
/// shared NIXL agent, and return `Block<SystemStorage>` instances for each slot.
fn alloc_system_blocks(
    num_blocks: usize,
    block_size: usize,
) -> (
    Arc<FullyContiguous<crate::block_manager::storage::SystemStorage>>,
    Vec<Block<crate::block_manager::storage::SystemStorage, locality::Local, BasicMetadata>>,
) {
    alloc_system_blocks_with_agent(
        num_blocks,
        block_size,
        TEST_AGENT.as_ref().as_ref().unwrap(),
    )
}

/// Same as `alloc_system_blocks` but registers with a caller-supplied agent.
///
/// Used by tests that create per-worker agents (e.g. TP isolation tests).
fn alloc_system_blocks_with_agent(
    num_blocks: usize,
    block_size: usize,
    agent: &NixlAgent,
) -> (
    Arc<FullyContiguous<crate::block_manager::storage::SystemStorage>>,
    Vec<Block<crate::block_manager::storage::SystemStorage, locality::Local, BasicMetadata>>,
) {
    // One-layer, one-page layout so block_size == inner_dim (bytes = inner_dim * dtype_width).
    // We use dtype_width=1 so inner_dim directly controls bytes per block.
    let config = LayoutConfig::builder()
        .num_blocks(num_blocks)
        .num_layers(1)
        .outer_dim(1)
        .page_size(1)
        .inner_dim(block_size)
        .dtype_width_bytes(1)
        .build()
        .unwrap();

    let mut layout = FullyContiguous::allocate(config, &SystemAllocator).unwrap();
    layout
        .nixl_register(agent, None)
        .expect("NIXL registration failed for SystemStorage");
    let layout = Arc::new(layout);

    let blocks: Vec<_> = (0..num_blocks)
        .map(|i| {
            let data = BlockData::new(layout.clone(), i, 0, 0);
            Block::new(data, BasicMetadata::default()).unwrap()
        })
        .collect();

    (layout, blocks)
}

fn disk_test_path() -> String {
    std::env::var(dynamo_runtime::config::environment_names::kvbm::testing::KVBM_TEST_DISK_PATH)
        .unwrap_or_else(|_| "/tmp/kvbm-nocuda-tests".into())
}

fn s3_bucket() -> String {
    std::env::var(dynamo_runtime::config::environment_names::kvbm::testing::KVBM_TEST_S3_BUCKET)
        .unwrap_or_else(|_| "kvbm-test-bucket".into())
}

fn s3_endpoint() -> Option<String> {
    std::env::var(dynamo_runtime::config::environment_names::kvbm::testing::KVBM_TEST_S3_ENDPOINT)
        .ok()
}

// ── POSIX disk tests ──────────────────────────────────────────────────────────

#[tokio::test]
async fn disk_offload_onboard_roundtrip() {
    let base_path = disk_test_path();
    std::fs::create_dir_all(&base_path).unwrap();

    let num_blocks = 4;
    let block_size = 4096;
    let (_layout, mut blocks) = alloc_system_blocks(num_blocks, block_size);

    // Fill blocks with deterministic data.
    for (i, block) in blocks.iter_mut().enumerate() {
        let mut view = block.block_data_mut().block_view_mut().unwrap();
        let slice =
            unsafe { std::slice::from_raw_parts_mut(view.as_mut_ptr(), view.size()) };
        for (j, byte) in slice.iter_mut().enumerate() {
            *byte = ((i * 37 + j) % 256) as u8;
        }
    }

    // Build descriptors — one file per block.
    let descriptors: Vec<_> = (0..num_blocks)
        .map(|i| RemoteBlockDescriptor::disk_from_hash(&base_path, i as u64, block_size, 0, 1))
        .collect();

    let base_ctx = make_transfer_ctx();
    let remote_ctx = RemoteTransferContext::new(
        base_ctx.clone(),
        RemoteStorageConfig::disk(&base_path, DISK_FLAGS_POSIX_BOTH),
    );
    let cancel = CancellationToken::new();

    // Offload.
    execute_remote_transfer(
        RemoteTransferDirection::Offload,
        RemoteStorageKind::Disk,
        &descriptors,
        &blocks,
        &remote_ctx,
        &cancel,
    )
    .await
    .expect("disk offload failed");

    // Zero out local blocks.
    for block in blocks.iter_mut() {
        let mut view = block.block_data_mut().block_view_mut().unwrap();
        let slice =
            unsafe { std::slice::from_raw_parts_mut(view.as_mut_ptr(), view.size()) };
        slice.fill(0);
    }

    // Onboard.
    execute_remote_transfer(
        RemoteTransferDirection::Onboard,
        RemoteStorageKind::Disk,
        &descriptors,
        &blocks,
        &remote_ctx,
        &cancel,
    )
    .await
    .expect("disk onboard failed");

    // Verify data survived the round-trip.
    for (i, block) in blocks.iter().enumerate() {
        let view = block.block_data().block_view().unwrap();
        let slice = unsafe { std::slice::from_raw_parts(view.as_ptr(), view.size()) };
        for (j, &byte) in slice.iter().enumerate() {
            let expected = ((i * 37 + j) % 256) as u8;
            assert_eq!(byte, expected, "disk: mismatch at block {i} byte {j}");
        }
    }
}

#[tokio::test]
async fn disk_transfer_cancellation_before_start() {
    let base_path = disk_test_path();
    std::fs::create_dir_all(&base_path).unwrap();

    let (_layout, blocks) = alloc_system_blocks(2, 4096);
    let descriptors: Vec<_> = (0..2)
        .map(|i| RemoteBlockDescriptor::disk_from_hash(&base_path, 0xCAFE + i as u64, 4096, 0, 1))
        .collect();

    let base_ctx = make_transfer_ctx();
    let remote_ctx = RemoteTransferContext::new(
        base_ctx,
        RemoteStorageConfig::disk(&base_path, DISK_FLAGS_POSIX_BOTH),
    );

    let cancel = CancellationToken::new();
    cancel.cancel(); // Cancel before the transfer starts.

    let result = execute_remote_transfer(
        RemoteTransferDirection::Offload,
        RemoteStorageKind::Disk,
        &descriptors,
        &blocks,
        &remote_ctx,
        &cancel,
    )
    .await;

    assert!(
        matches!(result, Err(TransferError::Cancelled)),
        "expected Cancelled, got {result:?}"
    );
}

#[tokio::test]
async fn disk_offload_rejects_mismatched_block_count() {
    let base_path = disk_test_path();
    std::fs::create_dir_all(&base_path).unwrap();

    let (_layout, blocks) = alloc_system_blocks(1, 4096);
    // Two descriptors but only one block.
    let descriptors = vec![
        RemoteBlockDescriptor::disk_from_hash(&base_path, 0x1111, 4096, 0, 1),
        RemoteBlockDescriptor::disk_from_hash(&base_path, 0x2222, 4096, 0, 1),
    ];

    let base_ctx = make_transfer_ctx();
    let remote_ctx = RemoteTransferContext::new(
        base_ctx,
        RemoteStorageConfig::disk(&base_path, DISK_FLAGS_POSIX_BOTH),
    );
    let cancel = CancellationToken::new();

    let result = execute_remote_transfer(
        RemoteTransferDirection::Offload,
        RemoteStorageKind::Disk,
        &descriptors,
        &blocks,
        &remote_ctx,
        &cancel,
    )
    .await;

    assert!(
        matches!(result, Err(TransferError::CountMismatch(2, 1))),
        "expected CountMismatch(2,1), got {result:?}"
    );
}

/// Simulate TP=2: two workers each offload then onboard the same sequence hash.
/// Workers must read back their own data, not each other's, because the file
/// name encodes `worker_id` and `world_size`.
#[tokio::test]
async fn disk_tp2_workers_are_isolated() {
    let base_path = disk_test_path();
    std::fs::create_dir_all(&base_path).unwrap();

    let block_size = 4096;
    let world_size = 2;
    let hash = 0xABCD_1234_u64;

    // Each worker writes a distinct pattern so we can detect cross-worker reads.
    for worker_id in 0..world_size {
        let (_layout, mut blocks) = alloc_system_blocks(1, block_size);
        let fill_byte = (0xA0 + worker_id) as u8;
        {
            let mut view = blocks[0].block_data_mut().block_view_mut().unwrap();
            let slice =
                unsafe { std::slice::from_raw_parts_mut(view.as_mut_ptr(), view.size()) };
            slice.fill(fill_byte);
        }

        let desc = vec![RemoteBlockDescriptor::disk_from_hash(
            &base_path, hash, block_size, worker_id, world_size,
        )];
        let ctx = RemoteTransferContext::new(
            make_transfer_ctx(),
            RemoteStorageConfig::disk(&base_path, DISK_FLAGS_POSIX_BOTH),
        )
        .with_topology(worker_id as u64, world_size);
        let cancel = CancellationToken::new();

        execute_remote_transfer(
            RemoteTransferDirection::Offload,
            RemoteStorageKind::Disk,
            &desc,
            &blocks,
            &ctx,
            &cancel,
        )
        .await
        .unwrap_or_else(|e| panic!("worker {worker_id} offload failed: {e:?}"));
    }

    // Each worker onboards and verifies it gets its own data.
    for worker_id in 0..world_size {
        let (_layout, mut blocks) = alloc_system_blocks(1, block_size);
        {
            let mut view = blocks[0].block_data_mut().block_view_mut().unwrap();
            let slice =
                unsafe { std::slice::from_raw_parts_mut(view.as_mut_ptr(), view.size()) };
            slice.fill(0);
        }

        let desc = vec![RemoteBlockDescriptor::disk_from_hash(
            &base_path, hash, block_size, worker_id, world_size,
        )];
        let ctx = RemoteTransferContext::new(
            make_transfer_ctx(),
            RemoteStorageConfig::disk(&base_path, DISK_FLAGS_POSIX_BOTH),
        )
        .with_topology(worker_id as u64, world_size);
        let cancel = CancellationToken::new();

        execute_remote_transfer(
            RemoteTransferDirection::Onboard,
            RemoteStorageKind::Disk,
            &desc,
            &blocks,
            &ctx,
            &cancel,
        )
        .await
        .unwrap_or_else(|e| panic!("worker {worker_id} onboard failed: {e:?}"));

        let expected_byte = (0xA0 + worker_id) as u8;
        let view = blocks[0].block_data().block_view().unwrap();
        let slice = unsafe { std::slice::from_raw_parts(view.as_ptr(), view.size()) };
        assert!(
            slice.iter().all(|&b| b == expected_byte),
            "worker {worker_id}: expected all bytes to be 0x{expected_byte:02x}",
        );
    }
}

// ── S3 / OBJ tests ────────────────────────────────────────────────────────────

#[tokio::test]
async fn s3_offload_onboard_roundtrip() {
    let bucket = s3_bucket();
    let endpoint = s3_endpoint();

    let num_blocks = 4;
    let block_size = 4096;
    let (_layout, mut blocks) = alloc_system_blocks(num_blocks, block_size);

    // Fill blocks with deterministic data.
    for (i, block) in blocks.iter_mut().enumerate() {
        let mut view = block.block_data_mut().block_view_mut().unwrap();
        let slice =
            unsafe { std::slice::from_raw_parts_mut(view.as_mut_ptr(), view.size()) };
        for (j, byte) in slice.iter_mut().enumerate() {
            *byte = ((i * 53 + j) % 256) as u8;
        }
    }

    let descriptors: Vec<_> = (0..num_blocks)
        .map(|i| RemoteBlockDescriptor::object_from_hash(&bucket, 0xDEAD_0000 + i as u64, block_size))
        .collect();

    let base_ctx = make_transfer_ctx();
    let remote_ctx = RemoteTransferContext::for_object_with_options(
        base_ctx.clone(),
        Some(bucket.clone()),
        endpoint.clone(),
        Some("us-east-1".into()),
        0,
    );
    let cancel = CancellationToken::new();

    // Offload to S3.
    execute_remote_transfer(
        RemoteTransferDirection::Offload,
        RemoteStorageKind::Object,
        &descriptors,
        &blocks,
        &remote_ctx,
        &cancel,
    )
    .await
    .expect("S3 offload failed");

    // Zero out local blocks.
    for block in blocks.iter_mut() {
        let mut view = block.block_data_mut().block_view_mut().unwrap();
        let slice =
            unsafe { std::slice::from_raw_parts_mut(view.as_mut_ptr(), view.size()) };
        slice.fill(0);
    }

    // Onboard from S3.
    execute_remote_transfer(
        RemoteTransferDirection::Onboard,
        RemoteStorageKind::Object,
        &descriptors,
        &blocks,
        &remote_ctx,
        &cancel,
    )
    .await
    .expect("S3 onboard failed");

    // Verify.
    for (i, block) in blocks.iter().enumerate() {
        let view = block.block_data().block_view().unwrap();
        let slice = unsafe { std::slice::from_raw_parts(view.as_ptr(), view.size()) };
        for (j, &byte) in slice.iter().enumerate() {
            let expected = ((i * 53 + j) % 256) as u8;
            assert_eq!(byte, expected, "S3: mismatch at block {i} byte {j}");
        }
    }
}

/// Simulate TP=2 with S3: each worker uses a separate bucket so the same
/// sequence hash maps to distinct S3 objects and TP ranks cannot cross-read.
///
/// Each worker has its own NixlAgent with an OBJ backend pointing at its
/// own bucket (`<base>-0`, `<base>-1`).  NIXL allows multiple OBJ agents
/// per process; in production each TP rank is a separate OS process anyway.
#[tokio::test]
async fn s3_tp2_workers_are_isolated() {
    let base_bucket = s3_bucket();

    let block_size = 4096;
    let world_size = 2_usize;
    let hash = 0xBEEF_5678_u64;

    // Offload from each worker into its own bucket.
    for worker_id in 0..world_size {
        let worker_bucket = format!("{base_bucket}-{worker_id}");
        let agent = create_obj_agent(
            &format!("kvbm-test-w{worker_id}-write"),
            &worker_bucket,
        );
        let agent_ref = agent.as_ref().as_ref().unwrap();
        let (_layout, mut blocks) =
            alloc_system_blocks_with_agent(1, block_size, agent_ref);
        let fill_byte = (0xB0 + worker_id) as u8;
        {
            let mut view = blocks[0].block_data_mut().block_view_mut().unwrap();
            let slice =
                unsafe { std::slice::from_raw_parts_mut(view.as_mut_ptr(), view.size()) };
            slice.fill(fill_byte);
        }

        let descs = vec![RemoteBlockDescriptor::object_from_hash(
            &worker_bucket,
            hash,
            block_size,
        )];
        let ctx = RemoteTransferContext::for_object_with_options(
            make_transfer_ctx_for(agent),
            Some(worker_bucket.clone()),
            s3_endpoint(),
            Some("us-east-1".into()),
            worker_id as u64,
        );
        let cancel = CancellationToken::new();

        execute_remote_transfer(
            RemoteTransferDirection::Offload,
            RemoteStorageKind::Object,
            &descs,
            &blocks,
            &ctx,
            &cancel,
        )
        .await
        .unwrap_or_else(|e| panic!("worker {worker_id} S3 offload failed: {e:?}"));
    }

    // Onboard from each worker's bucket and verify data isolation.
    for worker_id in 0..world_size {
        let worker_bucket = format!("{base_bucket}-{worker_id}");
        let agent = create_obj_agent(
            &format!("kvbm-test-w{worker_id}-read"),
            &worker_bucket,
        );
        let agent_ref = agent.as_ref().as_ref().unwrap();
        let (_layout, mut blocks) =
            alloc_system_blocks_with_agent(1, block_size, agent_ref);
        {
            let mut view = blocks[0].block_data_mut().block_view_mut().unwrap();
            let slice =
                unsafe { std::slice::from_raw_parts_mut(view.as_mut_ptr(), view.size()) };
            slice.fill(0);
        }

        let descs = vec![RemoteBlockDescriptor::object_from_hash(
            &worker_bucket,
            hash,
            block_size,
        )];
        let ctx = RemoteTransferContext::for_object_with_options(
            make_transfer_ctx_for(agent),
            Some(worker_bucket.clone()),
            s3_endpoint(),
            Some("us-east-1".into()),
            worker_id as u64,
        );
        let cancel = CancellationToken::new();

        execute_remote_transfer(
            RemoteTransferDirection::Onboard,
            RemoteStorageKind::Object,
            &descs,
            &blocks,
            &ctx,
            &cancel,
        )
        .await
        .unwrap_or_else(|e| panic!("worker {worker_id} S3 onboard failed: {e:?}"));

        let expected_byte = (0xB0 + worker_id) as u8;
        let view = blocks[0].block_data().block_view().unwrap();
        let slice = unsafe { std::slice::from_raw_parts(view.as_ptr(), view.size()) };
        assert!(
            slice.iter().all(|&b| b == expected_byte),
            "worker {worker_id}: expected all bytes 0x{expected_byte:02x}, isolation failed",
        );
    }
}

#[tokio::test]
async fn s3_transfer_cancellation_before_start() {
    let bucket = s3_bucket();
    let endpoint = s3_endpoint();

    let (_layout, blocks) = alloc_system_blocks(2, 4096);
    let descriptors: Vec<_> = (0..2)
        .map(|i| RemoteBlockDescriptor::object_from_hash(&bucket, 0xCAFE_0000 + i as u64, 4096))
        .collect();

    let base_ctx = make_transfer_ctx();
    let remote_ctx = RemoteTransferContext::for_object_with_options(
        base_ctx,
        Some(bucket),
        endpoint,
        Some("us-east-1".into()),
        0,
    );

    let cancel = CancellationToken::new();
    cancel.cancel();

    let result = execute_remote_transfer(
        RemoteTransferDirection::Offload,
        RemoteStorageKind::Object,
        &descriptors,
        &blocks,
        &remote_ctx,
        &cancel,
    )
    .await;

    assert!(
        matches!(result, Err(TransferError::Cancelled)),
        "expected Cancelled, got {result:?}"
    );
}
