// SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Integration tests for object storage transfers (S3-compatible backends).
//!
//! These tests verify NIXL transfers to/from object storage using MinIO or S3.
//! They require the NIXL OBJ backend to be enabled and configured via environment variables.
//!
//! # Configuration
//!
//! Set these environment variables before running:
//! ```bash
//! export DYN_KVBM_NIXL_BACKEND_OBJ_BUCKET=test-bucket
//! export DYN_KVBM_NIXL_BACKEND_OBJ_ENDPOINT_OVERRIDE=http://localhost:9000
//! export DYN_KVBM_NIXL_BACKEND_OBJ_ACCESS_KEY=minioadmin
//! export DYN_KVBM_NIXL_BACKEND_OBJ_SECRET_KEY=minioadmin
//! ```
//!
//! # Test Categories
//!
//! - **Offload**: Host/Device → Object storage
//! - **Onboard**: Object storage → Host/Device
//! - **Roundtrip**: Host → Object → Host verification
//! - **TP Partitions**: Byte-range reads from single object (tensor parallelism)
//! - **TP Selective**: Non-contiguous partition reads
//! - **Batch**: Multiple object operations

#![cfg(test)]

use super::{
    create_test_agent, fill_and_checksum, standard_config, verify_checksums_by_position,
};
use crate::{
    BlockId,
    v2::physical::{
        layout::{BlockDimension, PhysicalLayout},
        transfer::{
            BounceBuffer, DescriptorHint, LayoutHandle, TransferManager, TransferOptions,
            compute_block_checksums, fill_blocks,
        },
    },
};
use anyhow::Result;
use dynamo_memory::{NixlAgent, ObjectStorage, ObjectStorageConfig, StorageKind, nixl::NixlBackendConfig};
use rstest::rstest;
use std::collections::HashMap;
use std::sync::Arc;
use tracing_subscriber;

// ============================================================================
// Test Configuration
// ============================================================================

/// Storage type for host-side layouts in tests.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum HostStorageType {
    Pinned,
    Device,
}

/// Initialize tracing for tests (idempotent).
fn init_tracing() {
    let _ = tracing_subscriber::fmt()
        .with_max_level(tracing::Level::DEBUG)
        .with_test_writer()
        .try_init();
}

/// Create a test agent with OBJ backend support.
///
/// This agent must be configured with the NIXL OBJ plugin for object storage operations.
fn create_object_test_agent(name: &str) -> Result<NixlAgent> {
    let config = NixlBackendConfig::from_env()?;

    if !config.has_backend("OBJ") {
        anyhow::bail!("NIXL OBJ backend not configured. Set DYN_KVBM_NIXL_BACKEND_OBJ_* environment variables.");
    }

    NixlAgent::with_config(name, config)
}

/// Generate a unique object key for testing using timestamp and random component.
fn generate_test_object_key() -> u64 {
    use std::time::{SystemTime, UNIX_EPOCH};
    let timestamp = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_secs();
    let random: u64 = rand::random();
    timestamp.wrapping_mul(1_000_000).wrapping_add(random % 1_000_000)
}

// ============================================================================
// Layout Creation Helpers
// ============================================================================

/// Create an ObjectLayout with multiple separate objects (one per block).
///
/// This is used for offload tests where each block goes to its own S3 object.
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
        .allocate_objects(bucket, keys)?
        .build()
}

/// Create a fully contiguous layout backed by a single object.
///
/// This is used for TP tests where all data resides in one S3 object
/// and different ranks read from different byte offsets.
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
        .allocate_object(bucket, key)?
        .build()
}

/// Create a pinned host memory layout.
fn create_pinned_layout(agent: NixlAgent, num_blocks: usize) -> Result<PhysicalLayout> {
    let config = standard_config(num_blocks);

    PhysicalLayout::builder(agent)
        .with_config(config)
        .fully_contiguous()
        .allocate_pinned(false)?
        .build()
}

/// Create a GPU device memory layout.
fn create_device_layout(agent: NixlAgent, num_blocks: usize) -> Result<PhysicalLayout> {
    let config = standard_config(num_blocks);

    PhysicalLayout::builder(agent)
        .with_config(config)
        .fully_contiguous()
        .allocate_device(0)?
        .build()
}

/// Create a host layout based on storage type (pinned or device).
fn create_host_layout(
    agent: NixlAgent,
    storage_type: HostStorageType,
    num_blocks: usize,
) -> Result<PhysicalLayout> {
    match storage_type {
        HostStorageType::Pinned => create_pinned_layout(agent, num_blocks),
        HostStorageType::Device => create_device_layout(agent, num_blocks),
    }
}

/// Create a TransferManager for object storage tests.
fn create_object_transfer_manager(agent: NixlAgent) -> Result<TransferManager> {
    TransferManager::builder()
        .nixl_agent(agent)
        .cuda_device_id(0)
        .build()
}

// ============================================================================
// Offload Tests (Host → Object)
// ============================================================================

/// Implementation helper for offload tests.
///
/// Tests transferring data from host memory (pinned or device) to object storage.
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

    // Create source layout
    let src = create_host_layout(agent.clone(), host_storage, num_blocks)?;
    let dst = create_multi_object_layout(agent.clone(), &config.bucket, object_key, num_blocks)?;

    let block_ids: Vec<usize> = (0..num_blocks).collect();

    // Fill source and compute checksums
    use crate::v2::physical::transfer::FillPattern;
    let original_checksums = fill_and_checksum(&src, &block_ids, FillPattern::Sequential)?;

    let manager = create_object_transfer_manager(agent.clone())?;

    // Register layouts
    let src_handle = manager.register_layout(src)?;
    let dst_handle = manager.register_layout(dst)?;

    // For device storage, we need a bounce buffer
    let bounce_buffer = match host_storage {
        HostStorageType::Device => {
            let bounce = create_pinned_layout(agent.clone(), num_blocks)?;
            let bounce_handle = manager.register_layout(bounce)?;
            Some(BounceBuffer::from_handle(bounce_handle, block_ids.clone()))
        }
        HostStorageType::Pinned => None,
    };

    // Build transfer options with descriptor hint
    let mut options_builder = TransferOptions::builder()
        .descriptor_hint(DescriptorHint::PerBlockUniqueTarget);
    if let Some(bounce) = bounce_buffer {
        options_builder = options_builder.bounce_buffer(bounce);
    }
    let options = options_builder.build()?;

    // Execute transfer
    let notification = manager.transfer_blocks(
        src_handle,
        dst_handle,
        &block_ids,
        &block_ids,
        options,
    )?;
    notification.await?;

    // Verify by reading back to a new pinned buffer
    let verify_layout = create_pinned_layout(agent.clone(), num_blocks)?;
    let verify_handle = manager.register_layout(verify_layout.clone())?;

    let read_options = TransferOptions::builder()
        .descriptor_hint(DescriptorHint::PerBlockWithOffset)
        .build()?;

    let notification = manager.transfer_blocks(
        dst_handle,
        verify_handle,
        &block_ids,
        &block_ids,
        read_options,
    )?;
    notification.await?;

    // Verify data integrity
    verify_checksums_by_position(&original_checksums, &block_ids, &verify_layout, &block_ids)?;

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
// Onboard Tests (Object → Host)
// ============================================================================

/// Implementation helper for onboard tests.
///
/// Tests transferring data from object storage to host memory (pinned or device).
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
    use crate::v2::physical::transfer::FillPattern;
    let original_checksums = fill_and_checksum(&src, &block_ids, FillPattern::Sequential)?;

    let manager = create_object_transfer_manager(agent.clone())?;
    let src_handle = manager.register_layout(src)?;
    let obj_handle = manager.register_layout(obj)?;

    // Offload to object storage
    let write_options = TransferOptions::builder()
        .descriptor_hint(DescriptorHint::PerBlockUniqueTarget)
        .build()?;
    let notification = manager.transfer_blocks(
        src_handle,
        obj_handle,
        &block_ids,
        &block_ids,
        write_options,
    )?;
    notification.await?;

    // Create destination layout
    let dst = create_host_layout(agent.clone(), host_storage, num_blocks)?;
    let dst_handle = manager.register_layout(dst.clone())?;

    // For device storage, we need a bounce buffer
    let bounce_buffer = match host_storage {
        HostStorageType::Device => {
            let bounce = create_pinned_layout(agent.clone(), num_blocks)?;
            let bounce_handle = manager.register_layout(bounce)?;
            Some(BounceBuffer::from_handle(bounce_handle, block_ids.clone()))
        }
        HostStorageType::Pinned => None,
    };

    // Build read options
    let mut options_builder = TransferOptions::builder()
        .descriptor_hint(DescriptorHint::PerBlockWithOffset);
    if let Some(bounce) = bounce_buffer {
        options_builder = options_builder.bounce_buffer(bounce);
    }
    let options = options_builder.build()?;

    // Execute onboard
    let notification = manager.transfer_blocks(
        obj_handle,
        dst_handle,
        &block_ids,
        &block_ids,
        options,
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
    use crate::v2::physical::transfer::FillPattern;
    let original_checksums = fill_and_checksum(&src, &block_ids, FillPattern::Sequential)?;

    let manager = create_object_transfer_manager(agent)?;
    let src_handle = manager.register_layout(src)?;
    let obj_handle = manager.register_layout(obj)?;
    let dst_handle = manager.register_layout(dst.clone())?;

    // Step 1: Offload
    let write_options = TransferOptions::builder()
        .descriptor_hint(DescriptorHint::PerBlockUniqueTarget)
        .build()?;
    let notification = manager.transfer_blocks(
        src_handle,
        obj_handle,
        &block_ids,
        &block_ids,
        write_options,
    )?;
    notification.await?;

    // Step 2: Onboard
    let read_options = TransferOptions::builder()
        .descriptor_hint(DescriptorHint::PerBlockWithOffset)
        .build()?;
    let notification = manager.transfer_blocks(
        obj_handle,
        dst_handle,
        &block_ids,
        &block_ids,
        read_options,
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
    use crate::v2::physical::transfer::FillPattern;
    let original_checksums = fill_and_checksum(&src_pinned, &block_ids, FillPattern::Sequential)?;

    // Write ALL data to ONE Object (TP layout uses fully_contiguous)
    let obj_tp = create_single_object_layout(
        agent.clone(),
        &config.bucket,
        object_key,
        num_tp_partitions,
    )?;

    let manager = create_object_transfer_manager(agent.clone())?;
    let src_handle = manager.register_layout(src_pinned)?;
    let obj_handle = manager.register_layout(obj_tp)?;

    // Coalesced write to single Object
    let write_options = TransferOptions::builder()
        .descriptor_hint(DescriptorHint::Coalesced)
        .build()?;
    let notification = manager.transfer_blocks(
        src_handle,
        obj_handle,
        &block_ids,
        &block_ids,
        write_options,
    )?;
    notification.await?;

    if read_all_at_once {
        // Read all partitions in one transfer
        let dst_pinned = create_pinned_layout(agent.clone(), num_tp_partitions)?;
        let dst_handle = manager.register_layout(dst_pinned.clone())?;

        let read_options = TransferOptions::builder()
            .descriptor_hint(DescriptorHint::PerBlockWithOffset)
            .build()?;
        let notification = manager.transfer_blocks(
            obj_handle,
            dst_handle,
            &block_ids,
            &block_ids,
            read_options,
        )?;
        notification.await?;

        verify_checksums_by_position(&original_checksums, &block_ids, &dst_pinned, &block_ids)?;
    } else {
        // Read each TP partition individually (simulates TP ranks reading their data)
        for tp_rank in 0..num_tp_partitions {
            let dst_pinned = create_pinned_layout(agent.clone(), 1)?;
            let dst_handle = manager.register_layout(dst_pinned.clone())?;

            let read_options = TransferOptions::builder()
                .descriptor_hint(DescriptorHint::PerBlockWithOffset)
                .build()?;
            let notification = manager.transfer_blocks(
                obj_handle,
                dst_handle,
                &[tp_rank],
                &[0],
                read_options,
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

    use crate::v2::physical::transfer::FillPattern;
    let original_checksums = fill_and_checksum(&src_pinned, &all_block_ids, FillPattern::Sequential)?;

    // Write all partitions to single Object
    let obj_tp = create_single_object_layout(
        agent.clone(),
        &config.bucket,
        object_key,
        num_tp_partitions,
    )?;

    let manager = create_object_transfer_manager(agent.clone())?;
    let src_handle = manager.register_layout(src_pinned)?;
    let obj_handle = manager.register_layout(obj_tp)?;

    let write_options = TransferOptions::builder()
        .descriptor_hint(DescriptorHint::Coalesced)
        .build()?;
    let notification = manager.transfer_blocks(
        src_handle,
        obj_handle,
        &all_block_ids,
        &all_block_ids,
        write_options,
    )?;
    notification.await?;

    // Read only selected partitions
    let dst_pinned = create_pinned_layout(agent.clone(), selected.len())?;
    let dst_block_ids: Vec<usize> = (0..selected.len()).collect();
    let dst_handle = manager.register_layout(dst_pinned.clone())?;

    let read_options = TransferOptions::builder()
        .descriptor_hint(DescriptorHint::PerBlockWithOffset)
        .build()?;
    let notification = manager.transfer_blocks(
        obj_handle,
        dst_handle,
        selected,
        &dst_block_ids,
        read_options,
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

    use crate::v2::physical::transfer::FillPattern;
    let original_checksums = fill_and_checksum(&src, &block_ids, FillPattern::Sequential)?;

    let manager = create_object_transfer_manager(agent.clone())?;
    let src_handle = manager.register_layout(src)?;
    let obj_handle = manager.register_layout(obj)?;

    // Batch offload
    let write_options = TransferOptions::builder()
        .descriptor_hint(DescriptorHint::PerBlockUniqueTarget)
        .build()?;
    let notification = manager.transfer_blocks(
        src_handle,
        obj_handle,
        &block_ids,
        &block_ids,
        write_options,
    )?;
    notification.await?;

    // Batch onboard to new buffer
    let dst = create_pinned_layout(agent.clone(), num_objects)?;
    let dst_handle = manager.register_layout(dst.clone())?;

    let read_options = TransferOptions::builder()
        .descriptor_hint(DescriptorHint::PerBlockWithOffset)
        .build()?;
    let notification = manager.transfer_blocks(
        obj_handle,
        dst_handle,
        &block_ids,
        &block_ids,
        read_options,
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
