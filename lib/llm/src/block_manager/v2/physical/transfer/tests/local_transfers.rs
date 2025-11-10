// SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Local transfer tests where source and destination use the same NIXL agent.
//!
//! These tests verify data integrity across:
//! - Different storage types (System, Pinned, Device)
//! - Different layout types (Fully Contiguous, Layer-wise)
//! - Different transfer strategies (Memcpy, CUDA H2D/D2H)

use super::*;
use crate::block_manager::v2::physical::layout::BlockDimension;
use crate::block_manager::v2::physical::transfer::executor::execute_transfer;
use crate::block_manager::v2::physical::transfer::{
    BlockChecksum, BounceBufferSpec, FillPattern, StorageKind, TransferCapabilities,
    TransferOptions, compute_block_checksums, compute_layer_checksums, fill_blocks, fill_layers,
};
use anyhow::Result;
use rstest::rstest;
use std::collections::HashMap;
use std::ops::Range;
use std::sync::Arc;

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

/// Layout kind for parameterized testing.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LayoutKind {
    /// Fully contiguous layout
    FC,
    /// Layer-wise (layer-separate) layout
    LW,
}

/// Storage and layout specification for creating test layouts.
#[derive(Debug, Clone, Copy)]
pub struct LayoutSpec {
    pub kind: LayoutKind,
    pub storage: StorageKind,
}

impl LayoutSpec {
    pub fn new(kind: LayoutKind, storage: StorageKind) -> Self {
        Self { kind, storage }
    }
}

/// Transfer mode for parameterized testing.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TransferMode {
    /// Transfer entire blocks (all layers)
    FullBlocks,
    /// Transfer only the first layer
    FirstLayerOnly,
    /// Transfer only the second layer
    SecondLayerOnly,
}

impl TransferMode {
    /// Convert to optional layer range for execute_transfer.
    pub fn layer_range(&self) -> Option<Range<usize>> {
        match self {
            TransferMode::FullBlocks => None,
            TransferMode::FirstLayerOnly => Some(0..1),
            TransferMode::SecondLayerOnly => Some(1..2),
        }
    }

    /// Get a descriptive suffix for test names.
    pub fn suffix(&self) -> &'static str {
        match self {
            TransferMode::FullBlocks => "full",
            TransferMode::FirstLayerOnly => "layer0",
            TransferMode::SecondLayerOnly => "layer1",
        }
    }
}

/// Create a fully contiguous physical layout with the specified storage type.
pub fn create_fc_layout(
    agent: NixlAgent,
    storage_kind: StorageKind,
    num_blocks: usize,
) -> PhysicalLayout {
    let config = standard_config(num_blocks);
    let builder = PhysicalLayout::builder(agent)
        .with_config(config)
        .fully_contiguous();

    match storage_kind {
        StorageKind::System => builder.allocate_system().build().unwrap(),
        StorageKind::Pinned => builder.allocate_pinned(false).build().unwrap(),
        StorageKind::Device(device_id) => builder.allocate_device(device_id).build().unwrap(),
        StorageKind::Disk(_) => builder.allocate_disk(None).build().unwrap(),
    }
}

/// Create a layer-separate physical layout with the specified storage type.
pub fn create_lw_layout(
    agent: NixlAgent,
    storage_kind: StorageKind,
    num_blocks: usize,
) -> PhysicalLayout {
    let config = standard_config(num_blocks);
    let builder = PhysicalLayout::builder(agent)
        .with_config(config)
        .layer_separate(BlockDimension::BlockIsFirstDim);

    match storage_kind {
        StorageKind::System => builder.allocate_system().build().unwrap(),
        StorageKind::Pinned => builder.allocate_pinned(false).build().unwrap(),
        StorageKind::Device(device_id) => builder.allocate_device(device_id).build().unwrap(),
        StorageKind::Disk(_) => builder.allocate_disk(None).build().unwrap(),
    }
}

/// Create a physical layout based on the specification.
///
/// This is a DRY helper that dispatches to create_fc_layout or create_lw_layout
/// based on the layout kind in the spec.
pub fn create_layout(agent: NixlAgent, spec: LayoutSpec, num_blocks: usize) -> PhysicalLayout {
    match spec.kind {
        LayoutKind::FC => create_fc_layout(agent, spec.storage, num_blocks),
        LayoutKind::LW => create_lw_layout(agent, spec.storage, num_blocks),
    }
}

/// Fill blocks or layers based on transfer mode and compute checksums.
///
/// This is a mode-aware version of fill_and_checksum that handles both
/// full block transfers and layer-wise transfers.
pub fn fill_and_checksum_with_mode(
    layout: &PhysicalLayout,
    block_ids: &[usize],
    pattern: FillPattern,
    mode: TransferMode,
) -> Result<HashMap<usize, BlockChecksum>> {
    match mode {
        TransferMode::FullBlocks => {
            fill_blocks(layout, block_ids, pattern)?;
            compute_block_checksums(layout, block_ids)
        }
        TransferMode::FirstLayerOnly => {
            fill_layers(layout, block_ids, 0..1, pattern)?;
            compute_layer_checksums(layout, block_ids, 0..1)
        }
        TransferMode::SecondLayerOnly => {
            fill_layers(layout, block_ids, 1..2, pattern)?;
            compute_layer_checksums(layout, block_ids, 1..2)
        }
    }
}

/// Verify checksums with transfer mode awareness.
///
/// This is a mode-aware version that handles both full block and layer-wise verification.
pub fn verify_checksums_by_position_with_mode(
    src_checksums: &HashMap<usize, BlockChecksum>,
    src_block_ids: &[usize],
    dst_layout: &PhysicalLayout,
    dst_block_ids: &[usize],
    mode: TransferMode,
) -> Result<()> {
    assert_eq!(
        src_block_ids.len(),
        dst_block_ids.len(),
        "Source and destination block arrays must have same length"
    );

    let dst_checksums = match mode {
        TransferMode::FullBlocks => compute_block_checksums(dst_layout, dst_block_ids)?,
        TransferMode::FirstLayerOnly => compute_layer_checksums(dst_layout, dst_block_ids, 0..1)?,
        TransferMode::SecondLayerOnly => compute_layer_checksums(dst_layout, dst_block_ids, 1..2)?,
    };

    for (src_id, dst_id) in src_block_ids.iter().zip(dst_block_ids.iter()) {
        let src_checksum = src_checksums
            .get(src_id)
            .unwrap_or_else(|| panic!("Missing source checksum for block {}", src_id));
        let dst_checksum = dst_checksums
            .get(dst_id)
            .unwrap_or_else(|| panic!("Missing destination checksum for block {}", dst_id));

        assert_eq!(
            src_checksum, dst_checksum,
            "Checksum mismatch (mode={:?}): src[{}] != dst[{}]: {} != {}",
            mode, src_id, dst_id, src_checksum, dst_checksum
        );
    }

    Ok(())
}

/// Create a test agent with specific backends.
pub fn create_test_agent_with_backends(name: &str, backends: &[&str]) -> Result<NixlAgent> {
    NixlAgent::new_with_backends(name, backends)
}

/// Create a transport manager for testing with the specified agent.
///
/// Note: The agent should already have backends configured. Use `create_test_agent`
/// or `build_agent_with_backends` to create properly configured agents.
pub fn create_transfer_context(
    agent: NixlAgent,
    capabilities: Option<TransferCapabilities>,
) -> Result<crate::block_manager::v2::physical::manager::TransportManager> {
    crate::block_manager::v2::physical::manager::TransportManager::builder()
        .capabilities(capabilities.unwrap_or_default())
        .worker_id(0) // Default worker ID for local tests
        .nixl_agent(agent)
        .cuda_device_id(0)
        .build()
}

/// Fill blocks and compute checksums.
///
/// This can only be called on System or Pinned layouts.
pub fn fill_and_checksum(
    layout: &PhysicalLayout,
    block_ids: &[usize],
    pattern: FillPattern,
) -> Result<HashMap<usize, BlockChecksum>> {
    fill_blocks(layout, block_ids, pattern)?;
    compute_block_checksums(layout, block_ids)
}

/// Verify that destination block checksums match the expected source checksums.
///
/// This function compares checksums in order, assuming the source and destination
/// block arrays have a 1:1 correspondence (src[i] was transferred to dst[i]).
pub fn verify_checksums_by_position(
    src_checksums: &HashMap<usize, BlockChecksum>,
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

/// Fill guard blocks and return their checksums for later verification.
///
/// Guard blocks are blocks adjacent to transfer destinations that should
/// remain unchanged during transfers. This function fills them with a
/// distinctive pattern and returns their checksums for later validation.
///
/// # Arguments
/// * `layout` - The physical layout containing the guard blocks
/// * `guard_block_ids` - Block IDs to use as guards
/// * `pattern` - Fill pattern for guard blocks (typically a constant like 0xFF)
///
/// # Returns
/// A map of block ID to checksum for all guard blocks
pub fn create_guard_blocks(
    layout: &PhysicalLayout,
    guard_block_ids: &[usize],
    pattern: FillPattern,
) -> Result<HashMap<usize, BlockChecksum>> {
    fill_blocks(layout, guard_block_ids, pattern)?;
    compute_block_checksums(layout, guard_block_ids)
}

/// Verify that guard blocks remain unchanged after transfers.
///
/// This function compares the current checksums of guard blocks against
/// their expected values. Any mismatch indicates memory corruption or
/// unintended overwrites during transfer operations.
///
/// # Arguments
/// * `layout` - The physical layout containing the guard blocks
/// * `guard_block_ids` - Block IDs to verify
/// * `expected_checksums` - Expected checksums from create_guard_blocks
///
/// # Errors
/// Returns an error if any guard block checksum has changed
pub fn verify_guard_blocks_unchanged(
    layout: &PhysicalLayout,
    guard_block_ids: &[usize],
    expected_checksums: &HashMap<usize, BlockChecksum>,
) -> Result<()> {
    let current_checksums = compute_block_checksums(layout, guard_block_ids)?;

    for &block_id in guard_block_ids {
        let expected = expected_checksums
            .get(&block_id)
            .unwrap_or_else(|| panic!("Missing expected checksum for guard block {}", block_id));
        let current = current_checksums
            .get(&block_id)
            .unwrap_or_else(|| panic!("Missing current checksum for guard block {}", block_id));

        if expected != current {
            return Err(anyhow::anyhow!(
                "Guard block {} was modified during transfer! Expected: {}, Got: {}",
                block_id,
                expected,
                current
            ));
        }
    }

    Ok(())
}

struct DummyBounceBufferSpec {
    pub layout: PhysicalLayout,
    pub block_ids: Vec<usize>,
}

impl BounceBufferSpec for DummyBounceBufferSpec {
    fn layout(&self) -> &PhysicalLayout {
        &self.layout
    }
    fn block_ids(&self) -> &[usize] {
        &self.block_ids
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
    use crate::block_manager::v2::physical::transfer::TransferOptions;

    let agent = build_agent_for_kinds(src_kind, dst_kind)?;

    let src = build_layout(agent.clone(), src_layout, src_kind, 4);
    let dst = build_layout(agent.clone(), dst_layout, dst_kind, 4);

    let bounce_layout = build_layout(agent.clone(), LayoutType::FC, StorageKind::Pinned, 4);

    let bounce_buffer_spec: Arc<dyn BounceBufferSpec> = Arc::new(DummyBounceBufferSpec {
        layout: bounce_layout,
        block_ids: vec![0, 1],
    });

    let src_blocks = vec![0, 1];
    let dst_blocks = vec![2, 3];

    let checksums = fill_and_checksum(&src, &src_blocks, FillPattern::Sequential)?;
    let ctx = create_transfer_context(agent, None).unwrap();

    let options = TransferOptions::builder()
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
        TransferOptions::default(),
        ctx.context(),
    )?;
    notification.await?;

    // Transfer: Device[0,1] -> Pinned[2,3]
    let notification = execute_transfer(
        &device,
        &dst,
        &device_blocks,
        &dst_blocks,
        TransferOptions::default(),
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
        TransferOptions::default(),
        ctx.context(),
    )?;
    notification.await?;

    verify_checksums_by_position(&checksums, &src_blocks, &dst, &dst_blocks)?;

    Ok(())
}

#[rstest]
#[case(StorageKind::Device(0), StorageKind::Disk(0))]
#[case(StorageKind::Disk(0), StorageKind::Device(0))]
#[tokio::test]
async fn test_buffered_transfer(
    #[case] src_kind: StorageKind,
    #[values(LayoutType::FC, LayoutType::LW)] src_layout: LayoutType,
    #[case] dst_kind: StorageKind,
    #[values(LayoutType::FC, LayoutType::LW)] dst_layout: LayoutType,
) -> Result<()> {
    let agent = build_agent_for_kinds(src_kind, dst_kind)?;

    let src = build_layout(agent.clone(), src_layout, src_kind, 5);
    let dst = build_layout(agent.clone(), dst_layout, dst_kind, 5);

    let src_blocks = vec![0, 1, 2, 3, 4];
    let dst_blocks = vec![4, 3, 2, 1, 0];

    let bounce_layout = build_layout(agent.clone(), LayoutType::FC, StorageKind::Pinned, 3);
    let bounce_buffer_spec: Arc<dyn BounceBufferSpec> = Arc::new(DummyBounceBufferSpec {
        layout: bounce_layout,
        block_ids: vec![0, 1, 2],
    });

    let checksums = fill_and_checksum(&src, &src_blocks, FillPattern::Sequential)?;
    let ctx = create_transfer_context(agent, None).unwrap();

    let notification = execute_transfer(
        &src,
        &dst,
        &src_blocks,
        &dst_blocks,
        TransferOptions::builder()
            .bounce_buffer(bounce_buffer_spec)
            .build()?,
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
        TransferOptions::default(),
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
    let notification = execute_transfer(
        &host,
        &bounce,
        &src_blocks,
        &src_blocks,
        TransferOptions::from_layer_range(mode.layer_range()),
        ctx.context(),
    )?;
    notification.await?;

    // Execute bounce: bounce[0,1] → host[3,4]
    let notification = execute_transfer(
        &bounce,
        &host,
        &src_blocks,
        &dst_blocks,
        TransferOptions::from_layer_range(mode.layer_range()),
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
    let notification = execute_transfer(
        &src,
        &dst,
        &src_blocks,
        &dst_blocks,
        TransferOptions::from_layer_range(mode.layer_range()),
        ctx.context(),
    )?;
    notification.await?;

    // Verify data integrity
    verify_checksums_by_position_with_mode(&src_checksums, &src_blocks, &dst, &dst_blocks, mode)?;

    Ok(())
}
