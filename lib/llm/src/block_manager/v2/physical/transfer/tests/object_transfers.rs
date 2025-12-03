// SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Object storage transfer tests for verifying offload/onboard operations via NIXL OBJ backend.
//!
//! These tests require:
//! - NIXL with OBJ backend support
//! - Object-compatible object storage
//! - Environment variables for connection configuration
//!
//! # Environment Variables
//!
//! ```bash
//! export DYN_KVBM_NIXL_BACKEND_OBJ_ENDPOINT="http://localhost:9000"
//! export DYN_KVBM_NIXL_BACKEND_OBJ_BUCKET="test-bucket"
//! export DYN_KVBM_NIXL_BACKEND_OBJ_ACCESS_KEY=""
//! export DYN_KVBM_NIXL_BACKEND_OBJ_SECRET_KEY=""
//! ```

use super::*;
use crate::block_manager::v2::physical::layout::PhysicalLayout;
use crate::block_manager::v2::physical::transfer::executor::execute_transfer;
use crate::block_manager::v2::physical::transfer::{
    BounceBufferSpec, DescriptorHint, FillPattern, TransferCapabilities, TransferOptions,
    compute_block_checksums, fill_blocks,
};
use anyhow::Result;
use rstest::rstest;
use std::collections::HashMap;
use std::env;
use std::sync::{Arc, Once};

static INIT_TRACING: Once = Once::new();

/// Initialize tracing subscriber for tests (only once across all tests).
fn init_tracing() {
    INIT_TRACING.call_once(|| {
        tracing_subscriber::fmt()
            .with_env_filter(tracing_subscriber::EnvFilter::from_default_env())
            .with_test_writer()
            .init();
    });
}

// ============================================================================
// Configuration
// ============================================================================

/// Object storage configuration from environment variables.
#[derive(Debug, Clone)]
struct ObjectStorageConfig {
    #[allow(dead_code)]
    endpoint: String,
    bucket: String,
    #[allow(dead_code)]
    access_key: String,
    #[allow(dead_code)]
    secret_key: String,
}

impl ObjectStorageConfig {
    /// Load configuration from environment variables.
    /// Panics if any required variable is missing.
    fn from_env() -> Self {
        Self {
            endpoint: env::var("DYN_KVBM_NIXL_BACKEND_OBJ_ENDPOINT")
                .expect("DYN_KVBM_NIXL_BACKEND_OBJ_ENDPOINT env var required for object storage tests"),
            bucket: env::var("DYN_KVBM_NIXL_BACKEND_OBJ_BUCKET")
                .expect("DYN_KVBM_NIXL_BACKEND_OBJ_BUCKET env var required for object storage tests"),
            access_key: env::var("DYN_KVBM_NIXL_BACKEND_OBJ_ACCESS_KEY")
                .expect("DYN_KVBM_NIXL_BACKEND_OBJ_ACCESS_KEY env var required for object storage tests"),
            secret_key: env::var("DYN_KVBM_NIXL_BACKEND_OBJ_SECRET_KEY")
                .expect("DYN_KVBM_NIXL_BACKEND_OBJ_SECRET_KEY env var required for object storage tests"),
        }
    }
}

// ============================================================================
// Test Types (similar to local_transfers.rs)
// ============================================================================

/// Host storage type for object transfer tests.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HostStorageType {
    Pinned,
    Device,
}


// ============================================================================
// Helper Functions
// ============================================================================

/// Create an agent with OBJ backend for object storage tests.
fn create_object_test_agent(name: &str) -> Result<NixlAgent> {
    NixlAgent::new_with_backends(name, &["OBJ", "POSIX"])
}

/// Generate a unique object key for testing (based on timestamp).
fn generate_test_object_key() -> u64 {
    use std::time::{SystemTime, UNIX_EPOCH};
    let duration = SystemTime::now().duration_since(UNIX_EPOCH).unwrap();
    duration.as_nanos() as u64
}

/// Create an ObjectLayout with multiple Objects (one per block).
/// Used for WRITE operations where each block maps to a unique Object.
fn create_multi_object_layout(
    agent: NixlAgent,
    bucket: &str,
    base_key: u64,
    num_blocks: usize,
) -> Result<PhysicalLayout> {
    let config = standard_config(num_blocks);
    let keys: Vec<u64> = (0..num_blocks as u64).map(|i| base_key + i).collect();
    PhysicalLayout::builder(agent)
        .with_config(config)
        .object_layout()
        .allocate_objects(bucket.to_string(), keys)?
        .build()
}

/// Create a FullyContiguous layout backed by a SINGLE Object.
/// Used for READ operations where blocks read from different byte offsets.
fn create_single_object_layout(
    agent: NixlAgent,
    bucket: &str,
    key: u64,
    num_blocks: usize,
) -> Result<PhysicalLayout> {
    let config = standard_config(num_blocks);
    PhysicalLayout::builder(agent)
        .with_config(config)
        .fully_contiguous()
        .allocate_object(bucket.to_string(), key)
        .build()
}

/// Create a pinned memory layout for testing.
fn create_pinned_layout(agent: NixlAgent, num_blocks: usize) -> Result<PhysicalLayout> {
    let config = standard_config(num_blocks);
    Ok(PhysicalLayout::builder(agent)
        .with_config(config)
        .fully_contiguous()
        .allocate_pinned(false)
        .build()?)
}

/// Create a device memory layout for testing.
fn create_device_layout(agent: NixlAgent, num_blocks: usize) -> Result<PhysicalLayout> {
    let config = standard_config(num_blocks);
    Ok(PhysicalLayout::builder(agent)
        .with_config(config)
        .fully_contiguous()
        .allocate_device(0)
        .build()?)
}

/// Create host layout based on storage type.
fn create_host_layout(
    agent: NixlAgent,
    storage: HostStorageType,
    num_blocks: usize,
) -> Result<PhysicalLayout> {
    match storage {
        HostStorageType::Pinned => create_pinned_layout(agent, num_blocks),
        HostStorageType::Device => create_device_layout(agent, num_blocks),
    }
}

/// Create a transfer context for object storage tests.
fn create_object_transfer_context(
    agent: NixlAgent,
) -> Result<crate::block_manager::v2::physical::manager::TransportManager> {
    crate::block_manager::v2::physical::manager::TransportManager::builder()
        .capabilities(TransferCapabilities::default())
        .worker_id(0)
        .nixl_agent(agent)
        .cuda_device_id(0)
        .build()
}

/// Fill blocks and compute checksums (for pinned/system storage).
fn fill_and_checksum(
    layout: &PhysicalLayout,
    block_ids: &[usize],
    pattern: FillPattern,
) -> Result<HashMap<usize, crate::block_manager::v2::physical::transfer::BlockChecksum>> {
    fill_blocks(layout, block_ids, pattern)?;
    compute_block_checksums(layout, block_ids)
}

/// Verify checksums match by position.
fn verify_checksums_by_position(
    src_checksums: &HashMap<usize, crate::block_manager::v2::physical::transfer::BlockChecksum>,
    src_block_ids: &[usize],
    dst_layout: &PhysicalLayout,
    dst_block_ids: &[usize],
) -> Result<()> {
    assert_eq!(
        src_block_ids.len(),
        dst_block_ids.len(),
        "Source and destination block arrays must have same length"
    );

    let dst_checksums = compute_block_checksums(dst_layout, dst_block_ids)?;

    for (src_id, dst_id) in src_block_ids.iter().zip(dst_block_ids.iter()) {
        let src_checksum = src_checksums
            .get(src_id)
            .unwrap_or_else(|| panic!("Missing source checksum for block {}", src_id));
        let dst_checksum = dst_checksums
            .get(dst_id)
            .unwrap_or_else(|| panic!("Missing destination checksum for block {}", dst_id));

        assert_eq!(
            src_checksum, dst_checksum,
            "Checksum mismatch: src[{}] != dst[{}]: {} != {}",
            src_id, dst_id, src_checksum, dst_checksum
        );
    }

    Ok(())
}

// Bounce buffer helper for device transfers
struct TestBounceBuffer {
    layout: PhysicalLayout,
    block_ids: Vec<usize>,
}

impl BounceBufferSpec for TestBounceBuffer {
    fn layout(&self) -> &PhysicalLayout {
        &self.layout
    }
    fn block_ids(&self) -> &[usize] {
        &self.block_ids
    }
}

// ============================================================================
// Offload Tests (Host → Object Storage)
// ============================================================================

/// Implementation helper for offload tests.
async fn test_offload_impl(
    host_storage: HostStorageType,
    num_blocks: usize,
    name_suffix: &str,
) -> Result<()> {
    init_tracing();
    let config = ObjectStorageConfig::from_env();
    let test_name = format!("test-offload-{}-{}", name_suffix, generate_test_object_key());
    let agent = create_object_test_agent(&test_name)?;
    let object_key = generate_test_object_key();

    // Create layouts
    let src = create_host_layout(agent.clone(), host_storage, num_blocks)?;
    let dst = create_multi_object_layout(agent.clone(), &config.bucket, object_key, num_blocks)?;

    let block_ids: Vec<usize> = (0..num_blocks).collect();

    // For device storage, we need a bounce buffer
    let bounce_spec: Option<Arc<dyn BounceBufferSpec>> = match host_storage {
        HostStorageType::Device => {
            let bounce = create_pinned_layout(agent.clone(), num_blocks)?;
            Some(Arc::new(TestBounceBuffer {
                layout: bounce,
                block_ids: block_ids.clone(),
            }))
        }
        HostStorageType::Pinned => None,
    };

    // Fill source with test data (via bounce for device)
    let src_checksums = if host_storage == HostStorageType::Pinned {
        fill_and_checksum(&src, &block_ids, FillPattern::Sequential)?
    } else {
        // For device, we can't directly fill - just transfer and verify structure
        HashMap::new()
    };

    let ctx = create_object_transfer_context(agent.clone())?;

    // Build transfer options
    let mut options_builder = TransferOptions::builder()
        .descriptor_hint(DescriptorHint::PerBlockUniqueTarget);
    if let Some(bounce) = bounce_spec.clone() {
        options_builder = options_builder.bounce_buffer(bounce);
    }
    let options = options_builder.build()?;

    // Execute offload
    let notification = execute_transfer(
        &src,
        &dst,
        &block_ids,
        &block_ids,
        options,
        ctx.context(),
    )?;
    notification.await?;

    // Verify by reading back (only for pinned where we have checksums)
    if host_storage == HostStorageType::Pinned {
        let verify = create_pinned_layout(agent.clone(), num_blocks)?;
        let read_options = TransferOptions::builder()
            .descriptor_hint(DescriptorHint::PerBlockWithOffset)
            .build()?;
        let notification = execute_transfer(
            &dst,
            &verify,
            &block_ids,
            &block_ids,
            read_options,
            ctx.context(),
        )?;
        notification.await?;

        verify_checksums_by_position(&src_checksums, &block_ids, &verify, &block_ids)?;
    }

    Ok(())
}

#[rstest]
#[case(HostStorageType::Pinned, 4, "pinned_4")]
#[case(HostStorageType::Pinned, 8, "pinned_8")]
#[case(HostStorageType::Device, 4, "device_4")]
#[tokio::test]
async fn test_offload(
    #[case] host_storage: HostStorageType,
    #[case] num_blocks: usize,
    #[case] name_suffix: &str,
) -> Result<()> {
    test_offload_impl(host_storage, num_blocks, name_suffix).await
}

// ============================================================================
// Onboard Tests (Object Storage → Host)
// ============================================================================

/// Implementation helper for onboard tests.
async fn test_onboard_impl(
    host_storage: HostStorageType,
    num_blocks: usize,
    name_suffix: &str,
) -> Result<()> {
    init_tracing();
    let config = ObjectStorageConfig::from_env();
    let test_name = format!("test-onboard-{}-{}", name_suffix, generate_test_object_key());
    let agent = create_object_test_agent(&test_name)?;
    let object_key = generate_test_object_key();

    // First, create source data and offload to object storage
    let src = create_pinned_layout(agent.clone(), num_blocks)?;
    let obj = create_multi_object_layout(agent.clone(), &config.bucket, object_key, num_blocks)?;

    let block_ids: Vec<usize> = (0..num_blocks).collect();

    // Fill and checksum source
    let original_checksums = fill_and_checksum(&src, &block_ids, FillPattern::Sequential)?;

    let ctx = create_object_transfer_context(agent.clone())?;

    // Offload to object storage
    let write_options = TransferOptions::builder()
        .descriptor_hint(DescriptorHint::PerBlockUniqueTarget)
        .build()?;
    let notification = execute_transfer(
        &src,
        &obj,
        &block_ids,
        &block_ids,
        write_options,
        ctx.context(),
    )?;
    notification.await?;

    // Create destination layout
    let dst = create_host_layout(agent.clone(), host_storage, num_blocks)?;

    // For device storage, we need a bounce buffer
    let bounce_spec: Option<Arc<dyn BounceBufferSpec>> = match host_storage {
        HostStorageType::Device => {
            let bounce = create_pinned_layout(agent.clone(), num_blocks)?;
            Some(Arc::new(TestBounceBuffer {
                layout: bounce,
                block_ids: block_ids.clone(),
            }))
        }
        HostStorageType::Pinned => None,
    };

    // Build read options
    let mut options_builder = TransferOptions::builder()
        .descriptor_hint(DescriptorHint::PerBlockWithOffset);
    if let Some(bounce) = bounce_spec {
        options_builder = options_builder.bounce_buffer(bounce);
    }
    let options = options_builder.build()?;

    // Execute onboard
    let notification = execute_transfer(
        &obj,
        &dst,
        &block_ids,
        &block_ids,
        options,
        ctx.context(),
    )?;
    notification.await?;

    // Verify (only for pinned where we can compute checksums)
    if host_storage == HostStorageType::Pinned {
        verify_checksums_by_position(&original_checksums, &block_ids, &dst, &block_ids)?;
    }

    Ok(())
}

#[rstest]
#[case(HostStorageType::Pinned, 4, "pinned_4")]
#[case(HostStorageType::Pinned, 8, "pinned_8")]
#[case(HostStorageType::Device, 4, "device_4")]
#[tokio::test]
async fn test_onboard(
    #[case] host_storage: HostStorageType,
    #[case] num_blocks: usize,
    #[case] name_suffix: &str,
) -> Result<()> {
    test_onboard_impl(host_storage, num_blocks, name_suffix).await
}

// ============================================================================
// Roundtrip Tests (Host → Object → Host)
// ============================================================================

/// Implementation helper for roundtrip tests.
async fn test_roundtrip_impl(num_blocks: usize, name_suffix: &str) -> Result<()> {
    init_tracing();
    let config = ObjectStorageConfig::from_env();
    let test_name = format!("test-roundtrip-{}-{}", name_suffix, generate_test_object_key());
    let agent = create_object_test_agent(&test_name)?;
    let object_key = generate_test_object_key();

    let src = create_pinned_layout(agent.clone(), num_blocks)?;
    let obj = create_multi_object_layout(agent.clone(), &config.bucket, object_key, num_blocks)?;
    let dst = create_pinned_layout(agent.clone(), num_blocks)?;

    let block_ids: Vec<usize> = (0..num_blocks).collect();

    // Fill source
    let original_checksums = fill_and_checksum(&src, &block_ids, FillPattern::Sequential)?;

    let ctx = create_object_transfer_context(agent)?;

    // Step 1: Offload
    let write_options = TransferOptions::builder()
        .descriptor_hint(DescriptorHint::PerBlockUniqueTarget)
        .build()?;
    let notification = execute_transfer(
        &src,
        &obj,
        &block_ids,
        &block_ids,
        write_options,
        ctx.context(),
    )?;
    notification.await?;

    // Step 2: Onboard
    let read_options = TransferOptions::builder()
        .descriptor_hint(DescriptorHint::PerBlockWithOffset)
        .build()?;
    let notification = execute_transfer(
        &obj,
        &dst,
        &block_ids,
        &block_ids,
        read_options,
        ctx.context(),
    )?;
    notification.await?;

    // Step 3: Verify
    verify_checksums_by_position(&original_checksums, &block_ids, &dst, &block_ids)?;

    Ok(())
}

#[rstest]
#[case(4, "4_blocks")]
#[case(8, "8_blocks")]
#[case(16, "16_blocks")]
#[tokio::test]
async fn test_roundtrip(#[case] num_blocks: usize, #[case] name_suffix: &str) -> Result<()> {
    test_roundtrip_impl(num_blocks, name_suffix).await
}

// ============================================================================
// TP Partition Tests (byte-range reads from single Object)
// ============================================================================

/// Implementation helper for TP partition read tests.
///
/// This simulates a TP scenario where:
/// 1. One Object contains data for all tensor parallel partitions
/// 2. Each TP rank reads its partition from a different byte offset
///
/// ```text
/// Object:
/// ┌──────────────┬──────────────┬──────────────┬──────────────┐
/// │    TP0       │    TP1       │    TP2       │    TP3       │
/// │  offset 0    │  offset 1N   │  offset 2N   │  offset 3N   │
/// └──────────────┴──────────────┴──────────────┴──────────────┘
/// ```
async fn test_tp_partition_read_impl(
    num_tp_partitions: usize,
    read_all_at_once: bool,
    name_suffix: &str,
) -> Result<()> {
    init_tracing();
    let config = ObjectStorageConfig::from_env();
    let test_name = format!("test-tp-{}-{}", name_suffix, generate_test_object_key());
    let agent = create_object_test_agent(&test_name)?;
    let object_key = generate_test_object_key();

    // Create source data in pinned memory
    let src_pinned = create_pinned_layout(agent.clone(), num_tp_partitions)?;
    let block_ids: Vec<usize> = (0..num_tp_partitions).collect();

    // Fill with sequential pattern
    let original_checksums = fill_and_checksum(&src_pinned, &block_ids, FillPattern::Sequential)?;

    // Write ALL data to ONE Object (TP layout uses fully_contiguous)
    let obj_tp = create_single_object_layout(
        agent.clone(),
        &config.bucket,
        object_key,
        num_tp_partitions,
    )?;

    let ctx = create_object_transfer_context(agent.clone())?;

    // Coalesced write to single Object
    let write_options = TransferOptions::builder()
        .descriptor_hint(DescriptorHint::Coalesced)
        .build()?;
    let notification = execute_transfer(
        &src_pinned,
        &obj_tp,
        &block_ids,
        &block_ids,
        write_options,
        ctx.context(),
    )?;
    notification.await?;

    if read_all_at_once {
        // Read all partitions in one transfer
        let dst_pinned = create_pinned_layout(agent.clone(), num_tp_partitions)?;
        let read_options = TransferOptions::builder()
            .descriptor_hint(DescriptorHint::PerBlockWithOffset)
            .build()?;
        let notification = execute_transfer(
            &obj_tp,
            &dst_pinned,
            &block_ids,
            &block_ids,
            read_options,
            ctx.context(),
        )?;
        notification.await?;

        verify_checksums_by_position(&original_checksums, &block_ids, &dst_pinned, &block_ids)?;
    } else {
        // Read each TP partition individually (simulates TP ranks reading their data)
        for tp_rank in 0..num_tp_partitions {
            let dst_pinned = create_pinned_layout(agent.clone(), 1)?;

            let read_options = TransferOptions::builder()
                .descriptor_hint(DescriptorHint::PerBlockWithOffset)
                .build()?;
            let notification = execute_transfer(
                &obj_tp,
                &dst_pinned,
                &[tp_rank],
                &[0],
                read_options,
                ctx.context(),
            )?;
            notification.await?;

            // Verify this partition
            let dst_checksum = compute_block_checksums(&dst_pinned, &[0])?;
            let expected = original_checksums.get(&tp_rank).unwrap();
            let actual = dst_checksum.get(&0).unwrap();
            assert_eq!(
                expected, actual,
                "TP{} data mismatch: expected {}, got {}",
                tp_rank, expected, actual
            );
        }
    }

    Ok(())
}

#[rstest]
#[case(4, true, "tp4_batch")]
#[case(4, false, "tp4_individual")]
#[case(8, true, "tp8_batch")]
#[tokio::test]
async fn test_tp_partition_read(
    #[case] num_partitions: usize,
    #[case] read_all_at_once: bool,
    #[case] name_suffix: &str,
) -> Result<()> {
    test_tp_partition_read_impl(num_partitions, read_all_at_once, name_suffix).await
}

/// Test reading non-contiguous TP partitions (e.g., only TP0 and TP2).
async fn test_tp_selective_read_impl(
    num_tp_partitions: usize,
    selected: &[usize],
    name_suffix: &str,
) -> Result<()> {
    init_tracing();
    let config = ObjectStorageConfig::from_env();
    let test_name = format!("test-tp-selective-{}-{}", name_suffix, generate_test_object_key());
    let agent = create_object_test_agent(&test_name)?;
    let object_key = generate_test_object_key();

    // Create and populate source data
    let src_pinned = create_pinned_layout(agent.clone(), num_tp_partitions)?;
    let all_block_ids: Vec<usize> = (0..num_tp_partitions).collect();

    let original_checksums = fill_and_checksum(&src_pinned, &all_block_ids, FillPattern::Sequential)?;

    // Write all partitions to single Object
    let obj_tp = create_single_object_layout(
        agent.clone(),
        &config.bucket,
        object_key,
        num_tp_partitions,
    )?;
    let ctx = create_object_transfer_context(agent.clone())?;

    let write_options = TransferOptions::builder()
        .descriptor_hint(DescriptorHint::Coalesced)
        .build()?;
    let notification = execute_transfer(
        &src_pinned,
        &obj_tp,
        &all_block_ids,
        &all_block_ids,
        write_options,
        ctx.context(),
    )?;
    notification.await?;

    // Read only selected partitions
    let dst_pinned = create_pinned_layout(agent.clone(), selected.len())?;
    let dst_block_ids: Vec<usize> = (0..selected.len()).collect();

    let read_options = TransferOptions::builder()
        .descriptor_hint(DescriptorHint::PerBlockWithOffset)
        .build()?;
    let notification = execute_transfer(
        &obj_tp,
        &dst_pinned,
        selected,
        &dst_block_ids,
        read_options,
        ctx.context(),
    )?;
    notification.await?;

    // Verify selected partitions
    let dst_checksums = compute_block_checksums(&dst_pinned, &dst_block_ids)?;
    for (dst_idx, &src_tp) in selected.iter().enumerate() {
        let expected = original_checksums.get(&src_tp).unwrap();
        let actual = dst_checksums.get(&dst_idx).unwrap();
        assert_eq!(
            expected, actual,
            "TP{} -> dst[{}] mismatch: expected {}, got {}",
            src_tp, dst_idx, expected, actual
        );
    }

    Ok(())
}

#[rstest]
#[case(4, vec![0, 2], "tp4_0_2")]
#[case(4, vec![1, 3], "tp4_1_3")]
#[case(8, vec![0, 2, 4, 6], "tp8_even")]
#[case(8, vec![1, 3, 5, 7], "tp8_odd")]
#[tokio::test]
async fn test_tp_selective_read(
    #[case] num_partitions: usize,
    #[case] selected: Vec<usize>,
    #[case] name_suffix: &str,
) -> Result<()> {
    test_tp_selective_read_impl(num_partitions, &selected, name_suffix).await
}

// ============================================================================
// Batch Operation Tests
// ============================================================================

/// Test batch operations with multiple unique object keys.
async fn test_batch_impl(num_objects: usize, name_suffix: &str) -> Result<()> {
    init_tracing();
    let config = ObjectStorageConfig::from_env();
    let test_name = format!("test-batch-{}-{}", name_suffix, generate_test_object_key());
    let agent = create_object_test_agent(&test_name)?;

    // Create multiple object keys
    let base_key = generate_test_object_key();
    let keys: Vec<u64> = (0..num_objects as u64).map(|i| base_key + i).collect();

    let layout_config = standard_config(num_objects);

    // Create object layout with multiple keys
    let obj = PhysicalLayout::builder(agent.clone())
        .with_config(layout_config)
        .object_layout()
        .allocate_objects(config.bucket.clone(), keys)?
        .build()?;

    let src = create_pinned_layout(agent.clone(), num_objects)?;
    let block_ids: Vec<usize> = (0..num_objects).collect();

    let original_checksums = fill_and_checksum(&src, &block_ids, FillPattern::Sequential)?;

    let ctx = create_object_transfer_context(agent.clone())?;

    // Batch offload
    let write_options = TransferOptions::builder()
        .descriptor_hint(DescriptorHint::PerBlockUniqueTarget)
        .build()?;
    let notification = execute_transfer(
        &src,
        &obj,
        &block_ids,
        &block_ids,
        write_options,
        ctx.context(),
    )?;
    notification.await?;

    // Batch onboard to new buffer
    let dst = create_pinned_layout(agent.clone(), num_objects)?;
    let read_options = TransferOptions::builder()
        .descriptor_hint(DescriptorHint::PerBlockWithOffset)
        .build()?;
    let notification = execute_transfer(
        &obj,
        &dst,
        &block_ids,
        &block_ids,
        read_options,
        ctx.context(),
    )?;
    notification.await?;

    // Verify
    verify_checksums_by_position(&original_checksums, &block_ids, &dst, &block_ids)?;

    Ok(())
}

#[rstest]
#[case(4, "4_objects")]
#[case(8, "8_objects")]
#[case(16, "16_objects")]
#[tokio::test]
async fn test_batch(#[case] num_objects: usize, #[case] name_suffix: &str) -> Result<()> {
    test_batch_impl(num_objects, name_suffix).await
}
