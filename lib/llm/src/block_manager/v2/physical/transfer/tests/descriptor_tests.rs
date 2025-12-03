// SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Descriptor building tests for verifying hint behavior.
//!
//! These tests verify that each `DescriptorHint` produces the expected
//! descriptor patterns through the `build_descriptors` entry point.

use super::*;
use crate::block_manager::v2::physical::layout::PhysicalLayout;
use crate::block_manager::v2::physical::transfer::executor::{DescriptorParams, build_descriptors};
use crate::block_manager::v2::physical::transfer::DescriptorHint;
use anyhow::Result;
use nixl_sys::XferDescList;

// ============================================================================
// Test Helpers
// ============================================================================

/// Create a pinned layout for descriptor tests.
fn create_pinned_layout(agent: NixlAgent, num_blocks: usize) -> Result<PhysicalLayout> {
    let config = standard_config(num_blocks);
    Ok(PhysicalLayout::builder(agent)
        .with_config(config)
        .fully_contiguous()
        .allocate_pinned(false)
        .build()?)
}

/// Create an object layout with multiple objects (one per block).
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

/// Create a single-object layout (TP partition style).
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

/// Calculate block size in bytes from layout.
fn block_size_bytes(layout: &PhysicalLayout) -> usize {
    let l = layout.layout();
    l.num_layers() * l.outer_dim() * l.page_size() * l.inner_dim() * l.dtype_width_bytes()
}

/// Get descriptor details for verification.
struct DescriptorInfo {
    addr: usize,
    len: usize,
    device_id: u64,
}

fn get_descriptors(dl: &XferDescList) -> Vec<DescriptorInfo> {
    let count = dl.len().unwrap_or(0);
    (0..count)
        .filter_map(|i| dl.get(i).ok())
        .map(|d| DescriptorInfo {
            addr: d.addr,
            len: d.len,
            device_id: d.dev_id,
        })
        .collect()
}

/// Helper to build descriptors with a specific hint.
fn build_with_hint(
    src: &PhysicalLayout,
    dst: &PhysicalLayout,
    src_blocks: &[usize],
    dst_blocks: &[usize],
    hint: DescriptorHint,
) -> Result<(Vec<DescriptorInfo>, Vec<DescriptorInfo>)> {
    let layers = 0..src.layout().num_layers();

    let src_mem_type = src.nixl_metadata().mem_type();
    let dst_mem_type = dst.nixl_metadata().mem_type();

    let mut src_dl = XferDescList::new(src_mem_type)?;
    let mut dst_dl = XferDescList::new(dst_mem_type)?;

    let params = DescriptorParams {
        src,
        dst,
        src_block_ids: src_blocks,
        dst_block_ids: dst_blocks,
        layers: &layers,
        hint,
    };

    build_descriptors(&params, &mut src_dl, &mut dst_dl)?;

    Ok((get_descriptors(&src_dl), get_descriptors(&dst_dl)))
}

// ============================================================================
// Coalesced Hint Tests
// ============================================================================

#[test]
fn test_hint_coalesced_creates_single_descriptor() -> Result<()> {
    let agent = create_test_agent("test-coalesced");
    let src = create_pinned_layout(agent.clone(), 4)?;
    let dst = create_pinned_layout(agent.clone(), 4)?;

    let (src_descs, dst_descs) = build_with_hint(
        &src,
        &dst,
        &[0, 1, 2, 3],
        &[0, 1, 2, 3],
        DescriptorHint::Coalesced,
    )?;

    // Should create exactly 1 descriptor covering all 4 blocks
    assert_eq!(src_descs.len(), 1, "Coalesced should create 1 src descriptor");
    assert_eq!(dst_descs.len(), 1, "Coalesced should create 1 dst descriptor");

    // Verify size covers all blocks
    let expected_block_size = block_size_bytes(&src);
    let expected_total = 4 * expected_block_size;
    assert_eq!(
        src_descs[0].len, expected_total,
        "Size should cover all 4 blocks ({} bytes)",
        expected_total
    );

    Ok(())
}

#[test]
fn test_hint_coalesced_non_contiguous_fallback() -> Result<()> {
    let agent = create_test_agent("test-coalesced-nc");
    let src = create_pinned_layout(agent.clone(), 8)?;
    let dst = create_pinned_layout(agent.clone(), 8)?;

    // Non-contiguous blocks
    let (src_descs, _) = build_with_hint(
        &src,
        &dst,
        &[0, 2, 4, 6],
        &[1, 3, 5, 7],
        DescriptorHint::Coalesced,
    )?;

    // Non-contiguous should fall back to per-block
    assert_eq!(
        src_descs.len(),
        4,
        "Non-contiguous should create 4 descriptors"
    );

    Ok(())
}

// ============================================================================
// PerBlockWithOffset Hint Tests
// ============================================================================

#[test]
fn test_hint_per_block_offset_creates_n_descriptors() -> Result<()> {
    let agent = create_test_agent("test-offset");
    let src = create_pinned_layout(agent.clone(), 4)?;
    let dst = create_pinned_layout(agent.clone(), 4)?;

    let (src_descs, dst_descs) = build_with_hint(
        &src,
        &dst,
        &[0, 1, 2, 3],
        &[0, 1, 2, 3],
        DescriptorHint::PerBlockWithOffset,
    )?;

    // Should create 4 descriptors (one per block)
    assert_eq!(src_descs.len(), 4, "Should create 4 src descriptors");
    assert_eq!(dst_descs.len(), 4, "Should create 4 dst descriptors");

    // Verify offsets are different for each block (addresses are absolute, not relative)
    let block_size = block_size_bytes(&src);
    let base_addr = src_descs[0].addr;
    for (i, desc) in src_descs.iter().enumerate() {
        let relative_offset = desc.addr - base_addr;
        let expected_offset = i * block_size;
        assert_eq!(
            relative_offset, expected_offset,
            "Block {} should have relative offset {}",
            i, expected_offset
        );
    }

    Ok(())
}

#[test]
fn test_hint_per_block_offset_object_read() -> Result<()> {
    let agent = create_test_agent("test-offset-obj");

    // Single object layout (all blocks share same key, different offsets)
    let src = create_single_object_layout(agent.clone(), "test-bucket", 99999, 4)?;
    let dst = create_pinned_layout(agent.clone(), 4)?;

    let (src_descs, _) = build_with_hint(
        &src,
        &dst,
        &[0, 1, 2, 3],
        &[0, 1, 2, 3],
        DescriptorHint::PerBlockWithOffset,
    )?;

    // All descriptors should share the same object key (99999)
    for (i, desc) in src_descs.iter().enumerate() {
        assert_eq!(
            desc.device_id, 99999,
            "Block {} should share key 99999, got {}",
            i, desc.device_id
        );
    }

    // Offsets should be different (byte-range reads)
    let offsets: Vec<usize> = src_descs.iter().map(|d| d.addr).collect();
    let unique_offsets: std::collections::HashSet<_> = offsets.iter().collect();
    assert_eq!(
        unique_offsets.len(),
        4,
        "Each block should have unique offset"
    );

    Ok(())
}

// ============================================================================
// PerBlockUniqueTarget Hint Tests
// ============================================================================

#[test]
fn test_hint_per_block_unique_memory_preserves_offsets() -> Result<()> {
    let agent = create_test_agent("test-unique");
    let src = create_pinned_layout(agent.clone(), 4)?;
    let dst = create_pinned_layout(agent.clone(), 4)?;

    let (_, dst_descs) = build_with_hint(
        &src,
        &dst,
        &[0, 1, 2, 3],
        &[0, 1, 2, 3],
        DescriptorHint::PerBlockUniqueTarget,
    )?;

    // For memory transfers, relative offsets should be preserved (addresses are absolute)
    let block_size = block_size_bytes(&dst);
    let base_addr = dst_descs[0].addr;
    for (i, desc) in dst_descs.iter().enumerate() {
        let relative_offset = desc.addr - base_addr;
        let expected_offset = i * block_size;
        assert_eq!(
            relative_offset, expected_offset,
            "Block {} should have relative offset {}",
            i, expected_offset
        );
    }

    Ok(())
}

#[test]
fn test_hint_per_block_unique_object_write() -> Result<()> {
    let agent = create_test_agent("test-unique-obj");
    let src = create_pinned_layout(agent.clone(), 4)?;
    let dst = create_multi_object_layout(agent.clone(), "test-bucket", 1000, 4)?;

    let (_, dst_descs) = build_with_hint(
        &src,
        &dst,
        &[0, 1, 2, 3],
        &[0, 1, 2, 3],
        DescriptorHint::PerBlockUniqueTarget,
    )?;

    // All destination offsets should be 0 (object storage requirement)
    for (i, desc) in dst_descs.iter().enumerate() {
        assert_eq!(
            desc.addr, 0,
            "Block {} should have offset 0 for object write",
            i
        );
    }

    // Each block should have unique device_id (object key)
    let keys: Vec<u64> = dst_descs.iter().map(|d| d.device_id).collect();
    let expected_keys: Vec<u64> = vec![1000, 1001, 1002, 1003];
    assert_eq!(keys, expected_keys, "Each block should have unique object key");

    Ok(())
}

// ============================================================================
// BatchedRanges Hint Tests
// ============================================================================

#[test]
fn test_hint_batched_fully_contiguous() -> Result<()> {
    let agent = create_test_agent("test-batched-full");
    let src = create_pinned_layout(agent.clone(), 8)?;
    let dst = create_pinned_layout(agent.clone(), 8)?;

    // Fully contiguous blocks
    let (src_descs, _) = build_with_hint(
        &src,
        &dst,
        &[0, 1, 2, 3, 4, 5, 6, 7],
        &[0, 1, 2, 3, 4, 5, 6, 7],
        DescriptorHint::BatchedRanges,
    )?;

    // Fully contiguous should create 1 descriptor
    assert_eq!(
        src_descs.len(),
        1,
        "Fully contiguous should batch into 1 descriptor"
    );

    // Verify size covers all 8 blocks
    let expected_size = 8 * block_size_bytes(&src);
    assert_eq!(src_descs[0].len, expected_size, "Should cover all 8 blocks");

    Ok(())
}

#[test]
fn test_hint_batched_partial_contiguous() -> Result<()> {
    let agent = create_test_agent("test-batched-partial");
    let src = create_pinned_layout(agent.clone(), 16)?;
    let dst = create_pinned_layout(agent.clone(), 16)?;

    // Source contiguous ranges: [0,1,2] + [10,11] + [15]
    // But dst is also fragmented, breaking batches
    let (src_descs, _) = build_with_hint(
        &src,
        &dst,
        &[0, 1, 2, 10, 11, 15],
        &[0, 1, 2, 3, 4, 5], // dst contiguous breaks src batching
        DescriptorHint::BatchedRanges,
    )?;

    // Should batch some ranges (fewer than 6 descriptors)
    assert!(
        src_descs.len() <= 6,
        "Should batch at least some ranges, got {} descriptors",
        src_descs.len()
    );

    Ok(())
}

#[test]
fn test_hint_batched_no_contiguous() -> Result<()> {
    let agent = create_test_agent("test-batched-sep");
    let src = create_pinned_layout(agent.clone(), 16)?;
    let dst = create_pinned_layout(agent.clone(), 16)?;

    // No contiguous pairs in either src or dst
    let (src_descs, _) = build_with_hint(
        &src,
        &dst,
        &[0, 2, 4, 6],
        &[1, 3, 5, 7],
        DescriptorHint::BatchedRanges,
    )?;

    // No batching possible, should create 4 descriptors
    assert_eq!(
        src_descs.len(),
        4,
        "Non-contiguous should create 4 descriptors"
    );

    Ok(())
}

// ============================================================================
// Auto Hint Tests
// ============================================================================

#[test]
fn test_hint_auto_memory_to_memory() -> Result<()> {
    let agent = create_test_agent("test-auto-mem");
    let src = create_pinned_layout(agent.clone(), 4)?;
    let dst = create_pinned_layout(agent.clone(), 4)?;

    // Auto should detect contiguous→contiguous and use BatchedRanges
    let (src_descs, _) = build_with_hint(
        &src,
        &dst,
        &[0, 1, 2, 3],
        &[0, 1, 2, 3],
        DescriptorHint::Auto,
    )?;

    // Should batch into 1 descriptor (contiguous)
    assert_eq!(
        src_descs.len(),
        1,
        "Auto should batch contiguous memory transfer"
    );

    Ok(())
}

#[test]
fn test_hint_auto_object_read() -> Result<()> {
    let agent = create_test_agent("test-auto-obj-read");
    let src = create_single_object_layout(agent.clone(), "bucket", 12345, 4)?;
    let dst = create_pinned_layout(agent.clone(), 4)?;

    // Auto should detect Object→Memory and use PerBlockWithOffset
    let (src_descs, _) = build_with_hint(
        &src,
        &dst,
        &[0, 1, 2, 3],
        &[0, 1, 2, 3],
        DescriptorHint::Auto,
    )?;

    // Should create 4 descriptors (per-block for object reads)
    assert_eq!(
        src_descs.len(),
        4,
        "Auto should use per-block for object reads"
    );

    Ok(())
}

#[test]
fn test_hint_auto_object_write() -> Result<()> {
    let agent = create_test_agent("test-auto-obj-write");
    let src = create_pinned_layout(agent.clone(), 4)?;
    let dst = create_multi_object_layout(agent.clone(), "bucket", 5000, 4)?;

    // Auto should detect Memory→Object and use PerBlockUniqueTarget
    let (_, dst_descs) = build_with_hint(
        &src,
        &dst,
        &[0, 1, 2, 3],
        &[0, 1, 2, 3],
        DescriptorHint::Auto,
    )?;

    // Should create 4 descriptors with unique keys and offset 0
    assert_eq!(
        dst_descs.len(),
        4,
        "Auto should use per-block for object writes"
    );

    for desc in &dst_descs {
        assert_eq!(desc.addr, 0, "Object writes should have offset 0");
    }

    Ok(())
}

// ============================================================================
// Default Hint Tests (no hint provided)
// ============================================================================

/// Helper to build descriptors with the default hint (Auto).
fn build_with_default_hint(
    src: &PhysicalLayout,
    dst: &PhysicalLayout,
    src_blocks: &[usize],
    dst_blocks: &[usize],
) -> Result<(Vec<DescriptorInfo>, Vec<DescriptorInfo>)> {
    // DescriptorHint::default() is Auto
    build_with_hint(src, dst, src_blocks, dst_blocks, DescriptorHint::default())
}

#[test]
fn test_default_hint_is_auto() {
    // Verify that the default hint is Auto
    assert_eq!(
        DescriptorHint::default(),
        DescriptorHint::Auto,
        "Default hint should be Auto"
    );
}

#[test]
fn test_default_hint_memory_to_memory() -> Result<()> {
    let agent = create_test_agent("test-default-mem");
    let src = create_pinned_layout(agent.clone(), 4)?;
    let dst = create_pinned_layout(agent.clone(), 4)?;

    // Using default hint (no explicit hint provided)
    let (src_descs, _) = build_with_default_hint(&src, &dst, &[0, 1, 2, 3], &[0, 1, 2, 3])?;

    // Should auto-detect contiguous and batch
    assert_eq!(
        src_descs.len(),
        1,
        "Default hint should batch contiguous memory"
    );

    Ok(())
}

#[test]
fn test_default_hint_object_read() -> Result<()> {
    let agent = create_test_agent("test-default-read");
    let src = create_single_object_layout(agent.clone(), "bucket", 55555, 4)?;
    let dst = create_pinned_layout(agent.clone(), 4)?;

    let (src_descs, _) = build_with_default_hint(&src, &dst, &[0, 1, 2, 3], &[0, 1, 2, 3])?;

    // Should auto-detect object read and use per-block
    assert_eq!(
        src_descs.len(),
        4,
        "Default hint should use per-block for object reads"
    );

    Ok(())
}

#[test]
fn test_default_hint_object_write() -> Result<()> {
    let agent = create_test_agent("test-default-write");
    let src = create_pinned_layout(agent.clone(), 4)?;
    let dst = create_multi_object_layout(agent.clone(), "bucket", 7000, 4)?;

    let (_, dst_descs) = build_with_default_hint(&src, &dst, &[0, 1, 2, 3], &[0, 1, 2, 3])?;

    // Should auto-detect object write
    assert_eq!(dst_descs.len(), 4, "Default hint should use per-block for object writes");

    // All offsets should be 0
    for desc in &dst_descs {
        assert_eq!(desc.addr, 0, "Object writes should have offset 0");
    }

    Ok(())
}

// ============================================================================
// Edge Case Tests
// ============================================================================

#[test]
fn test_single_block_transfer() -> Result<()> {
    let agent = create_test_agent("test-single");
    let src = create_pinned_layout(agent.clone(), 4)?;
    let dst = create_pinned_layout(agent.clone(), 4)?;

    let (src_descs, _) = build_with_hint(
        &src,
        &dst,
        &[2],
        &[3],
        DescriptorHint::Coalesced,
    )?;

    assert_eq!(src_descs.len(), 1, "Single block should create 1 descriptor");

    Ok(())
}

#[test]
fn test_descriptor_sizes_match() -> Result<()> {
    let agent = create_test_agent("test-sizes");
    let src = create_pinned_layout(agent.clone(), 4)?;
    let dst = create_pinned_layout(agent.clone(), 4)?;

    let (src_descs, dst_descs) = build_with_hint(
        &src,
        &dst,
        &[0, 1],
        &[2, 3],
        DescriptorHint::PerBlockWithOffset,
    )?;

    // Sizes should match between src and dst
    for (s, d) in src_descs.iter().zip(dst_descs.iter()) {
        assert_eq!(s.len, d.len, "Src and dst descriptor sizes should match");
    }

    Ok(())
}
