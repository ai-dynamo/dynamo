// SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Local transfer tests where source and destination use the same NIXL agent.
//!
//! These tests verify data integrity across:
//! - Different storage types (System, Pinned, Device)
//! - Different layout types (Fully Contiguous, Layer-wise)
//! - Different transfer strategies (Memcpy, CUDA H2D/D2H)

use super::*;
use crate::physical::transfer::TransferCapabilities;
use crate::physical::transfer::executor::TransferOptionsInternal;
use crate::v2::physical::transfer::executor::execute_transfer;
use anyhow::Result;
use rstest::rstest;

// ============================================================================
// System <=> System Tests (Memcpy)
// ============================================================================

#[derive(Clone)]
enum LayoutType {
    FC,
    LW,
}

fn build_layout(
    agent: NixlAgent,
    layout_type: LayoutType,
    storage_kind: StorageKind,
    num_blocks: usize,
) -> PhysicalLayout {
    match layout_type {
        LayoutType::FC => create_fc_layout(agent, storage_kind, num_blocks),
        LayoutType::LW => create_lw_layout(agent, storage_kind, num_blocks),
    }
}

fn build_agent_for_kinds(src_kind: StorageKind, dst_kind: StorageKind) -> Result<NixlAgent> {
    use std::collections::HashSet;

    let mut backends = HashSet::new();

    // Determine required backends for both source and destination
    for kind in [src_kind, dst_kind] {
        match kind {
            StorageKind::System | StorageKind::Pinned => {
                backends.insert("POSIX"); // Lightweight for DRAM
            }
            StorageKind::Device(_) => {
                backends.insert("UCX"); // Required for VRAM (expensive)
            }
            StorageKind::Disk(_) => {
                backends.insert("POSIX"); // Required for disk I/O
            }
        }
    }

    // Optional: Add GDS for Device <-> Disk optimization
    match (src_kind, dst_kind) {
        (StorageKind::Device(_), StorageKind::Disk(_))
        | (StorageKind::Disk(_), StorageKind::Device(_)) => {
            backends.insert("GDS_MT");
        }
        _ => {}
    }

    let backend_vec: Vec<&str> = backends.into_iter().collect();
    create_test_agent_with_backends("agent", &backend_vec)
}

#[rstest]
#[tokio::test]
async fn test_p2p(
    #[values(LayoutType::FC, LayoutType::LW)] src_layout: LayoutType,
    #[values(
        StorageKind::System,
        StorageKind::Pinned,
        StorageKind::Device(0),
        StorageKind::Disk(0)
    )]
    src_kind: StorageKind,
    #[values(LayoutType::FC, LayoutType::LW)] dst_layout: LayoutType,
    #[values(
        StorageKind::System,
        StorageKind::Pinned,
        StorageKind::Device(0),
        StorageKind::Disk(0)
    )]
    dst_kind: StorageKind,
) -> Result<()> {
    use crate::physical::transfer::{BounceBufferInternal, executor::TransferOptionsInternal};

    let agent = build_agent_for_kinds(src_kind, dst_kind)?;

    let src = build_layout(agent.clone(), src_layout, src_kind, 4);
    let dst = build_layout(agent.clone(), dst_layout, dst_kind, 4);

    let bounce_layout = build_layout(agent.clone(), LayoutType::FC, StorageKind::Pinned, 4);

    let bounce_buffer_spec = BounceBufferInternal::from_layout(bounce_layout, vec![0, 1]);

    let src_blocks = vec![0, 1];
    let dst_blocks = vec![2, 3];

    let checksums = fill_and_checksum(&src, &src_blocks, FillPattern::Sequential)?;
    let ctx = create_transfer_context(agent, None).unwrap();

    let options = TransferOptionsInternal::builder()
        .bounce_buffer(bounce_buffer_spec)
        .build()?;

    let notification =
        execute_transfer(&src, &dst, &src_blocks, &dst_blocks, options, ctx.context())?;
    notification.await?;

    verify_checksums_by_position(&checksums, &src_blocks, &dst, &dst_blocks)?;

    Ok(())
}

#[rstest]
#[tokio::test]
async fn test_roundtrip(
    #[values(LayoutType::FC, LayoutType::LW)] src_layout: LayoutType,
    #[values(StorageKind::System, StorageKind::Pinned, StorageKind::Device(0))]
    src_kind: StorageKind,
    #[values(LayoutType::FC, LayoutType::LW)] inter_layout: LayoutType,
    #[values(StorageKind::System, StorageKind::Pinned, StorageKind::Device(0))]
    inter_kind: StorageKind,
    #[values(LayoutType::FC, LayoutType::LW)] dst_layout: LayoutType,
    #[values(StorageKind::System, StorageKind::Pinned, StorageKind::Device(0))]
    dst_kind: StorageKind,
) -> Result<()> {
    use crate::physical::transfer::executor::TransferOptionsInternal;

    let agent = build_agent_for_kinds(src_kind, dst_kind)?;

    // Create layouts: source pinned, device intermediate, destination pinned
    let src = build_layout(agent.clone(), src_layout, src_kind, 4);
    let device = build_layout(agent.clone(), inter_layout, inter_kind, 4);
    let dst = build_layout(agent.clone(), dst_layout, dst_kind, 4);

    let src_blocks = vec![0, 1];
    let device_blocks = vec![0, 1];
    let dst_blocks = vec![2, 3];

    // Fill source and compute checksums
    let checksums = fill_and_checksum(&src, &src_blocks, FillPattern::Sequential)?;
    let ctx = create_transfer_context(agent, None).unwrap();

    // Transfer: Pinned[0,1] -> Device[0,1]
    let notification = execute_transfer(
        &src,
        &device,
        &src_blocks,
        &device_blocks,
        TransferOptionsInternal::default(),
        ctx.context(),
    )?;
    notification.await?;

    // Transfer: Device[0,1] -> Pinned[2,3]
    let notification = execute_transfer(
        &device,
        &dst,
        &device_blocks,
        &dst_blocks,
        TransferOptionsInternal::default(),
        ctx.context(),
    )?;
    notification.await?;

    // Verify checksums match
    verify_checksums_by_position(&checksums, &src_blocks, &dst, &dst_blocks)?;

    Ok(())
}

#[rstest]
#[case(StorageKind::Device(0), StorageKind::Disk(0))]
#[case(StorageKind::Disk(0), StorageKind::Device(0))]
#[tokio::test]
async fn test_gds(
    #[case] src_kind: StorageKind,
    #[values(LayoutType::FC, LayoutType::LW)] src_layout: LayoutType,
    #[case] dst_kind: StorageKind,
    #[values(LayoutType::FC, LayoutType::LW)] dst_layout: LayoutType,
) -> Result<()> {
    let capabilities = TransferCapabilities::default().with_gds_if_supported();

    if !capabilities.allow_gds {
        println!("System does not support GDS. Skipping test.");
        return Ok(());
    }

    let agent = build_agent_for_kinds(src_kind, dst_kind)?;

    let src = build_layout(agent.clone(), src_layout, src_kind, 4);
    let dst = build_layout(agent.clone(), dst_layout, dst_kind, 4);

    let src_blocks = vec![0, 1];
    let dst_blocks = vec![2, 3];

    let checksums = fill_and_checksum(&src, &src_blocks, FillPattern::Sequential)?;
    let ctx = create_transfer_context(agent, Some(capabilities)).unwrap();

    let notification = execute_transfer(
        &src,
        &dst,
        &src_blocks,
        &dst_blocks,
        TransferOptionsInternal::default(),
        ctx.context(),
    )?;
    notification.await?;

    verify_checksums_by_position(&checksums, &src_blocks, &dst, &dst_blocks)?;

    Ok(())
}

#[rstest]
#[case(1024)]
#[case(2048)]
#[case(4096)]
#[case(8192)]
#[case(16384)]
#[tokio::test]
async fn test_large_block_counts(#[case] block_count: usize) {
    let agent = create_test_agent(&format!("test_large_block_counts_{}", block_count));

    let src = create_fc_layout(agent.clone(), StorageKind::Pinned, block_count);
    let device = create_fc_layout(agent.clone(), StorageKind::Device(0), block_count);

    let src_blocks = (0..block_count).collect::<Vec<_>>();
    let device_blocks = (0..block_count).collect::<Vec<_>>();

    let ctx = create_transfer_context(agent, None).unwrap();
    let notification = execute_transfer(
        &src,
        &device,
        &src_blocks,
        &device_blocks,
        TransferOptionsInternal::default(),
        ctx.context(),
    )
    .unwrap();
    notification.await.unwrap();
}

// ============================================================================
// Parameterized Bounce Tests with Guard Block Validation
// ============================================================================

/// Test bounce transfers with guard block validation.
///
/// This test validates that:
/// 1. Data can be transferred: host[src_blocks] → bounce[src_blocks] → host[dst_blocks]
/// 2. Guard blocks adjacent to dst_blocks remain unchanged (no memory corruption)
/// 3. Works correctly with different storage types, layouts, and transfer modes
///
/// Test pattern (6 blocks total):
/// - Source blocks: [0, 1]
/// - Destination blocks: [3, 4]
/// - Guard blocks: [2, 5] (adjacent to destination, should remain unchanged)
#[rstest]
// Storage combinations (host, bounce)
#[case(StorageKind::System, StorageKind::Pinned, "sys_pin")]
#[case(StorageKind::Pinned, StorageKind::System, "pin_sys")]
#[case(StorageKind::Pinned, StorageKind::Device(0), "pin_dev")]
#[tokio::test]
async fn test_bounce_with_guards_fc_fc_full(
    #[case] host_storage: StorageKind,
    #[case] bounce_storage: StorageKind,
    #[case] name_suffix: &str,
) {
    test_bounce_with_guards_impl(
        host_storage,
        bounce_storage,
        LayoutKind::FC,
        LayoutKind::FC,
        TransferMode::FullBlocks,
        name_suffix,
    )
    .await
    .unwrap();
}

#[rstest]
#[case(StorageKind::System, StorageKind::Pinned, "sys_pin")]
#[case(StorageKind::Pinned, StorageKind::System, "pin_sys")]
#[case(StorageKind::Pinned, StorageKind::Device(0), "pin_dev")]
#[tokio::test]
async fn test_bounce_with_guards_fc_lw_full(
    #[case] host_storage: StorageKind,
    #[case] bounce_storage: StorageKind,
    #[case] name_suffix: &str,
) {
    test_bounce_with_guards_impl(
        host_storage,
        bounce_storage,
        LayoutKind::FC,
        LayoutKind::LW,
        TransferMode::FullBlocks,
        name_suffix,
    )
    .await
    .unwrap();
}

#[rstest]
#[case(StorageKind::System, StorageKind::Pinned, "sys_pin")]
#[case(StorageKind::Pinned, StorageKind::System, "pin_sys")]
#[case(StorageKind::Pinned, StorageKind::Device(0), "pin_dev")]
#[tokio::test]
async fn test_bounce_with_guards_lw_fc_full(
    #[case] host_storage: StorageKind,
    #[case] bounce_storage: StorageKind,
    #[case] name_suffix: &str,
) {
    test_bounce_with_guards_impl(
        host_storage,
        bounce_storage,
        LayoutKind::LW,
        LayoutKind::FC,
        TransferMode::FullBlocks,
        name_suffix,
    )
    .await
    .unwrap();
}

#[rstest]
#[case(StorageKind::System, StorageKind::Pinned, "sys_pin")]
#[case(StorageKind::Pinned, StorageKind::System, "pin_sys")]
#[case(StorageKind::Pinned, StorageKind::Device(0), "pin_dev")]
#[tokio::test]
async fn test_bounce_with_guards_lw_lw_full(
    #[case] host_storage: StorageKind,
    #[case] bounce_storage: StorageKind,
    #[case] name_suffix: &str,
) {
    test_bounce_with_guards_impl(
        host_storage,
        bounce_storage,
        LayoutKind::LW,
        LayoutKind::LW,
        TransferMode::FullBlocks,
        name_suffix,
    )
    .await
    .unwrap();
}

#[rstest]
#[case(StorageKind::Pinned, StorageKind::Device(0), "pin_dev")]
#[tokio::test]
async fn test_bounce_with_guards_fc_fc_layer0(
    #[case] host_storage: StorageKind,
    #[case] bounce_storage: StorageKind,
    #[case] name_suffix: &str,
) {
    test_bounce_with_guards_impl(
        host_storage,
        bounce_storage,
        LayoutKind::FC,
        LayoutKind::FC,
        TransferMode::FirstLayerOnly,
        name_suffix,
    )
    .await
    .unwrap();
}

#[rstest]
#[case(StorageKind::Pinned, StorageKind::Device(0), "pin_dev")]
#[tokio::test]
async fn test_bounce_with_guards_lw_lw_layer0(
    #[case] host_storage: StorageKind,
    #[case] bounce_storage: StorageKind,
    #[case] name_suffix: &str,
) {
    test_bounce_with_guards_impl(
        host_storage,
        bounce_storage,
        LayoutKind::LW,
        LayoutKind::LW,
        TransferMode::FirstLayerOnly,
        name_suffix,
    )
    .await
    .unwrap();
}

/// Implementation helper for bounce tests with guard blocks.
async fn test_bounce_with_guards_impl(
    host_storage: StorageKind,
    bounce_storage: StorageKind,
    host_layout: LayoutKind,
    bounce_layout: LayoutKind,
    mode: TransferMode,
    name_suffix: &str,
) -> Result<()> {
    let num_blocks = 6;
    let test_name = format!(
        "bounce_{}_{:?}_{:?}_{}_{}",
        name_suffix,
        host_layout,
        bounce_layout,
        mode.suffix(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_millis()
    );
    let agent = create_test_agent(&test_name);

    // Create layouts
    let host = create_layout(
        agent.clone(),
        LayoutSpec::new(host_layout, host_storage),
        num_blocks,
    );
    let bounce = create_layout(
        agent.clone(),
        LayoutSpec::new(bounce_layout, bounce_storage),
        num_blocks,
    );

    // Block assignments:
    // - Transfer: host[0,1] → bounce[0,1] → host[3,4]
    // - Guards: host[2,5] (should remain unchanged)
    let src_blocks = vec![0, 1];
    let dst_blocks = vec![3, 4];
    let guard_blocks = vec![2, 5];

    // Setup: Fill source blocks and guard blocks
    let src_checksums =
        fill_and_checksum_with_mode(&host, &src_blocks, FillPattern::Sequential, mode)?;
    let guard_checksums = create_guard_blocks(&host, &guard_blocks, FillPattern::Constant(0xFF))?;

    let ctx = create_transfer_context(agent, None)?;

    // Execute bounce: host[0,1] → bounce[0,1]
    let mut options_builder = TransferOptionsInternal::builder();
    if let Some(range) = mode.layer_range() {
        options_builder = options_builder.layer_range(range);
    }
    let notification = execute_transfer(
        &host,
        &bounce,
        &src_blocks,
        &src_blocks,
        options_builder.build()?,
        ctx.context(),
    )?;
    notification.await?;

    // Execute bounce: bounce[0,1] → host[3,4]
    let mut options_builder = TransferOptionsInternal::builder();
    if let Some(range) = mode.layer_range() {
        options_builder = options_builder.layer_range(range);
    }
    let notification = execute_transfer(
        &bounce,
        &host,
        &src_blocks,
        &dst_blocks,
        options_builder.build()?,
        ctx.context(),
    )?;
    notification.await?;

    // Verify: Data integrity + guards unchanged
    verify_checksums_by_position_with_mode(&src_checksums, &src_blocks, &host, &dst_blocks, mode)?;
    verify_guard_blocks_unchanged(&host, &guard_blocks, &guard_checksums)?;

    Ok(())
}

// ============================================================================
// Parameterized Direct Transfer Tests
// ============================================================================

/// Test direct transfers with parameterization over storage, layout, and transfer mode.
///
/// This demonstrates the DRY parameterized approach that can replace the 18 individual
/// tests above (System<=>System, Pinned<=>Pinned, cross-type, etc).
///
/// Note: Only tests System<=>System, Pinned<=>Pinned, and System<=>Pinned since we can only
/// fill/checksum System and Pinned storage. For Device tests, use bounce tests instead.
#[rstest]
// Storage combinations (only fillable storage types)
#[case(StorageKind::System, StorageKind::System, "sys_sys")]
#[case(StorageKind::Pinned, StorageKind::Pinned, "pin_pin")]
#[case(StorageKind::System, StorageKind::Pinned, "sys_pin")]
#[case(StorageKind::Pinned, StorageKind::System, "pin_sys")]
#[tokio::test]
async fn test_direct_transfer_fc_fc_full(
    #[case] src_storage: StorageKind,
    #[case] dst_storage: StorageKind,
    #[case] name_suffix: &str,
) {
    test_direct_transfer_impl(
        src_storage,
        dst_storage,
        LayoutKind::FC,
        LayoutKind::FC,
        TransferMode::FullBlocks,
        name_suffix,
    )
    .await
    .unwrap();
}

#[rstest]
#[case(StorageKind::System, StorageKind::Pinned, "sys_pin")]
#[case(StorageKind::Pinned, StorageKind::System, "pin_sys")]
#[tokio::test]
async fn test_direct_transfer_fc_lw_layer0(
    #[case] src_storage: StorageKind,
    #[case] dst_storage: StorageKind,
    #[case] name_suffix: &str,
) {
    test_direct_transfer_impl(
        src_storage,
        dst_storage,
        LayoutKind::FC,
        LayoutKind::LW,
        TransferMode::FirstLayerOnly,
        name_suffix,
    )
    .await
    .unwrap();
}

#[rstest]
#[case(StorageKind::Pinned, StorageKind::Pinned, "pin_pin")]
#[tokio::test]
async fn test_direct_transfer_lw_lw_layer1(
    #[case] src_storage: StorageKind,
    #[case] dst_storage: StorageKind,
    #[case] name_suffix: &str,
) {
    test_direct_transfer_impl(
        src_storage,
        dst_storage,
        LayoutKind::LW,
        LayoutKind::LW,
        TransferMode::SecondLayerOnly,
        name_suffix,
    )
    .await
    .unwrap();
}

/// Implementation helper for direct transfer tests.
async fn test_direct_transfer_impl(
    src_storage: StorageKind,
    dst_storage: StorageKind,
    src_layout: LayoutKind,
    dst_layout: LayoutKind,
    mode: TransferMode,
    name_suffix: &str,
) -> Result<()> {
    let num_blocks = 4;
    let test_name = format!(
        "direct_{}_{:?}_{:?}_{}_{}",
        name_suffix,
        src_layout,
        dst_layout,
        mode.suffix(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_millis()
    );
    let agent = create_test_agent(&test_name);

    // Create layouts
    let src = create_layout(
        agent.clone(),
        LayoutSpec::new(src_layout, src_storage),
        num_blocks,
    );
    let dst = create_layout(
        agent.clone(),
        LayoutSpec::new(dst_layout, dst_storage),
        num_blocks,
    );

    // Transfer src[0,1] -> dst[2,3]
    let src_blocks = vec![0, 1];
    let dst_blocks = vec![2, 3];

    // Fill source and compute checksums
    let src_checksums =
        fill_and_checksum_with_mode(&src, &src_blocks, FillPattern::Sequential, mode)?;

    let ctx = create_transfer_context(agent, None)?;

    // Execute transfer
    let mut options_builder = TransferOptionsInternal::builder();
    if let Some(range) = mode.layer_range() {
        options_builder = options_builder.layer_range(range);
    }
    let notification = execute_transfer(
        &src,
        &dst,
        &src_blocks,
        &dst_blocks,
        options_builder.build()?,
        ctx.context(),
    )?;
    notification.await?;

    // Verify data integrity
    verify_checksums_by_position_with_mode(&src_checksums, &src_blocks, &dst, &dst_blocks, mode)?;

    Ok(())
}
