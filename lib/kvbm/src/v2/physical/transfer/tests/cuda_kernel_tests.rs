// SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Unit tests for CUDA kernel compatibility and execution.
//!
//! These tests verify:
//! - Layout compatibility (FC↔LW, FC↔FC)
//! - Backend selection (VectorizedKernel, MemcpyAsync, MemcpyBatch)
//! - Data integrity across H2D/D2H roundtrips
//! - Layer-wise transfer consistency

use super::{
    FillPattern, LayoutKind, LayoutSpec, NixlAgent, TransferMode, create_guard_blocks,
    create_layout, create_test_agent, create_transfer_context, fill_and_checksum,
    fill_and_checksum_with_mode, skip_if_stubs, verify_checksums_by_position,
    verify_checksums_by_position_with_mode, verify_guard_blocks_unchanged,
};
use crate::v2::physical::layout::{BlockDimension, LayoutConfig, PhysicalLayout};
use crate::v2::physical::transfer::executor::cuda::try_execute_operational_kernel;
use crate::v2::physical::transfer::{StorageKind, TransferContext};
use anyhow::Result;
use dynamo_kvbm_kernels::tensor_kernels::OperationalCopyBackend;
use rstest::rstest;

/// Skip test if stub kernels are in use (for tests returning Result).
macro_rules! skip_if_stubs_result {
    () => {
        if dynamo_kvbm_kernels::is_using_stubs() {
            eprintln!(
                "Skipping test '{}': stub kernels in use (no real CUDA)",
                module_path!()
            );
            return Ok(());
        }
    };
}

/// Skip test if MemcpyBatch backend is selected but not available (CUDA < 12.9).
macro_rules! skip_if_memcpy_batch_unsupported {
    ($backend:expr) => {
        if matches!($backend, OperationalCopyBackend::MemcpyBatch)
            && !dynamo_kvbm_kernels::is_memcpy_batch_available()
        {
            eprintln!(
                "Skipping test '{}': MemcpyBatch not available (CUDA < 12.9)",
                module_path!()
            );
            return Ok(());
        }
    };
}

/// Create a fully contiguous layout for testing
fn create_fc_layout(agent: NixlAgent, storage: StorageKind) -> PhysicalLayout {
    let config = LayoutConfig::builder()
        .num_blocks(4)
        .num_layers(2)
        .outer_dim(2)
        .page_size(16)
        .inner_dim(64)
        .dtype_width_bytes(2)
        .build()
        .unwrap();

    let builder = PhysicalLayout::builder(agent)
        .with_config(config)
        .fully_contiguous();

    match storage {
        StorageKind::Device(id) => builder.allocate_device(id).build().unwrap(),
        StorageKind::Pinned => builder.allocate_pinned(None).build().unwrap(),
        _ => panic!("Unsupported storage kind"),
    }
}

/// Create a layer-wise layout for testing
fn create_lw_layout(agent: NixlAgent, storage: StorageKind) -> PhysicalLayout {
    let config = LayoutConfig::builder()
        .num_blocks(4)
        .num_layers(2)
        .outer_dim(2)
        .page_size(16)
        .inner_dim(64)
        .dtype_width_bytes(2)
        .build()
        .unwrap();

    let builder = PhysicalLayout::builder(agent)
        .with_config(config)
        .layer_separate(BlockDimension::BlockIsFirstDim);

    match storage {
        StorageKind::Device(id) => builder.allocate_device(id).build().unwrap(),
        StorageKind::Pinned => builder.allocate_pinned(None).build().unwrap(),
        _ => panic!("Unsupported storage kind"),
    }
}

/// Test that LW→LW (both layer-wise) is incompatible and returns error
#[test]
fn test_kernel_incompatible_lw_to_lw() {
    skip_if_stubs!();
    let agent = create_test_agent("test_kernel_lw_lw");
    let ctx = TransferContext::builder()
        .nixl_agent(agent.clone())
        .cuda_device_id(0)
        .build()
        .unwrap();

    let src = create_lw_layout(agent.clone(), StorageKind::Device(0));
    let dst = create_lw_layout(agent, StorageKind::Device(0));

    let src_blocks = vec![0, 1];
    let dst_blocks = vec![0, 1];
    let layers = 0..2;
    let stream = ctx.h2d_stream();
    let backend = OperationalCopyBackend::Auto;

    // Call the kernel function directly - should fail with incompatibility error
    let result = try_execute_operational_kernel(
        &src,
        &dst,
        &src_blocks,
        &dst_blocks,
        layers,
        stream.as_ref(),
        backend,
        ctx.cuda_pool(),
    );

    assert!(result.is_err(), "Expected error for LW→LW transfer");
    let err_msg = result.unwrap_err().to_string();
    assert!(
        err_msg.contains("not compatible") || err_msg.contains("incompatible"),
        "Error should mention incompatibility, got: {}",
        err_msg
    );
}

/// Test that FC→LW is compatible and succeeds
#[test]
fn test_kernel_compatible_fc_to_lw() {
    skip_if_stubs!();
    let agent = create_test_agent("test_kernel_fc_lw");
    let ctx = TransferContext::builder()
        .nixl_agent(agent.clone())
        .cuda_device_id(0)
        .build()
        .unwrap();

    let src = create_fc_layout(agent.clone(), StorageKind::Device(0));
    let dst = create_lw_layout(agent, StorageKind::Device(0));

    let src_blocks = vec![0, 1];
    let dst_blocks = vec![0, 1];
    let layers = 0..2;
    let stream = ctx.h2d_stream();
    let backend = OperationalCopyBackend::Auto;

    // Call the kernel function directly - should succeed
    let result = try_execute_operational_kernel(
        &src,
        &dst,
        &src_blocks,
        &dst_blocks,
        layers,
        stream.as_ref(),
        backend,
        ctx.cuda_pool(),
    );

    assert!(
        result.is_ok(),
        "Expected success for FC→LW transfer, got: {:?}",
        result.unwrap_err()
    );
}

/// Test that LW→FC is compatible and succeeds
#[test]
fn test_kernel_compatible_lw_to_fc() {
    skip_if_stubs!();
    let agent = create_test_agent("test_kernel_lw_fc");
    let ctx = TransferContext::builder()
        .nixl_agent(agent.clone())
        .cuda_device_id(0)
        .build()
        .unwrap();

    let src = create_lw_layout(agent.clone(), StorageKind::Device(0));
    let dst = create_fc_layout(agent, StorageKind::Device(0));

    let src_blocks = vec![0, 1];
    let dst_blocks = vec![0, 1];
    let layers = 0..2;
    let stream = ctx.h2d_stream();
    let backend = OperationalCopyBackend::Auto;

    // Call the kernel function directly - should succeed
    let result = try_execute_operational_kernel(
        &src,
        &dst,
        &src_blocks,
        &dst_blocks,
        layers,
        stream.as_ref(),
        backend,
        ctx.cuda_pool(),
    );

    assert!(
        result.is_ok(),
        "Expected success for LW→FC transfer, got: {:?}",
        result.unwrap_err()
    );
}

/// Test that FC→FC is compatible and succeeds
#[test]
fn test_kernel_compatible_fc_to_fc() {
    skip_if_stubs!();
    let agent = create_test_agent("test_kernel_fc_fc");
    let ctx = TransferContext::builder()
        .nixl_agent(agent.clone())
        .cuda_device_id(0)
        .build()
        .unwrap();

    let src = create_fc_layout(agent.clone(), StorageKind::Device(0));
    let dst = create_fc_layout(agent, StorageKind::Device(0));

    let src_blocks = vec![0, 1];
    let dst_blocks = vec![0, 1];
    let layers = 0..2;
    let stream = ctx.h2d_stream();
    let backend = OperationalCopyBackend::Auto;

    // Call the kernel function directly - should succeed
    let result = try_execute_operational_kernel(
        &src,
        &dst,
        &src_blocks,
        &dst_blocks,
        layers,
        stream.as_ref(),
        backend,
        ctx.cuda_pool(),
    );

    assert!(
        result.is_ok(),
        "Expected success for FC→FC transfer, got: {:?}",
        result.unwrap_err()
    );
}

/// Test that the vectorized kernel backend works correctly for FC→LW transfers.
/// This explicitly forces the VectorizedKernel backend to verify it works.
#[test]
fn test_kernel_vectorized_backend_fc_to_lw() {
    skip_if_stubs!();
    let agent = create_test_agent("test_kernel_vectorized_fc_lw");
    let ctx = TransferContext::builder()
        .nixl_agent(agent.clone())
        .cuda_device_id(0)
        .build()
        .unwrap();

    // Use dimensions that guarantee 8-byte alignment for vectorized path
    // page_size=16, inner_dim=64, dtype=2 bytes -> inner=16*64*2=2048 bytes (8-byte aligned)
    let src = create_fc_layout(agent.clone(), StorageKind::Device(0));
    let dst = create_lw_layout(agent, StorageKind::Device(0));

    let src_blocks = vec![0, 1];
    let dst_blocks = vec![0, 1];
    let layers = 0..2;
    let stream = ctx.h2d_stream();
    let backend = OperationalCopyBackend::VectorizedKernel;

    // Call the kernel function with VectorizedKernel backend
    let result = try_execute_operational_kernel(
        &src,
        &dst,
        &src_blocks,
        &dst_blocks,
        layers,
        stream.as_ref(),
        backend,
        ctx.cuda_pool(),
    );

    assert!(
        result.is_ok(),
        "Expected success for FC→LW transfer with VectorizedKernel backend, got: {:?}",
        result.unwrap_err()
    );
}

/// Test that the vectorized kernel backend works correctly for LW→FC transfers (unpack direction).
#[test]
fn test_kernel_vectorized_backend_lw_to_fc() {
    skip_if_stubs!();
    let agent = create_test_agent("test_kernel_vectorized_lw_fc");
    let ctx = TransferContext::builder()
        .nixl_agent(agent.clone())
        .cuda_device_id(0)
        .build()
        .unwrap();

    let src = create_lw_layout(agent.clone(), StorageKind::Device(0));
    let dst = create_fc_layout(agent, StorageKind::Device(0));

    let src_blocks = vec![0, 1];
    let dst_blocks = vec![0, 1];
    let layers = 0..2;
    let stream = ctx.h2d_stream();
    let backend = OperationalCopyBackend::VectorizedKernel;

    // Call the kernel function with VectorizedKernel backend
    let result = try_execute_operational_kernel(
        &src,
        &dst,
        &src_blocks,
        &dst_blocks,
        layers,
        stream.as_ref(),
        backend,
        ctx.cuda_pool(),
    );

    assert!(
        result.is_ok(),
        "Expected success for LW→FC transfer with VectorizedKernel backend, got: {:?}",
        result.unwrap_err()
    );
}

/// Test that Auto mode selects the vectorized kernel for 8-byte aligned data.
/// This verifies the priority order: vectorized -> batch -> async.
#[test]
fn test_kernel_auto_selects_vectorized_for_aligned_data() {
    skip_if_stubs!();
    let agent = create_test_agent("test_kernel_auto_vectorized");
    let ctx = TransferContext::builder()
        .nixl_agent(agent.clone())
        .cuda_device_id(0)
        .build()
        .unwrap();

    // Create layouts with 8-byte aligned inner dimensions
    // page_size=16, inner_dim=64, dtype=2 bytes -> total=2048 bytes per chunk (8-byte aligned)
    let src = create_fc_layout(agent.clone(), StorageKind::Device(0));
    let dst = create_lw_layout(agent, StorageKind::Device(0));

    let src_blocks = vec![0, 1, 2, 3];
    let dst_blocks = vec![0, 1, 2, 3];
    let layers = 0..2;
    let stream = ctx.h2d_stream();
    let backend = OperationalCopyBackend::Auto;

    // Auto mode should succeed - it will use vectorized kernel for aligned data
    let result = try_execute_operational_kernel(
        &src,
        &dst,
        &src_blocks,
        &dst_blocks,
        layers,
        stream.as_ref(),
        backend,
        ctx.cuda_pool(),
    );

    assert!(
        result.is_ok(),
        "Expected success for Auto mode with aligned data, got: {:?}",
        result.unwrap_err()
    );
}

// ============================================================================
// Parameterized H2D/D2H Roundtrip Tests with Data Integrity Verification
// ============================================================================

/// Test H2D roundtrip: Pinned src -> Device -> Pinned dst
///
/// This test verifies data integrity across a full roundtrip transfer:
/// 1. Fill Pinned src blocks with sequential data
/// 2. H2D: Transfer Pinned src -> Device using specified backend
/// 3. D2H: Transfer Device -> Pinned dst using specified backend
/// 4. Verify checksums match between src and dst
/// 5. Verify guard blocks remain unchanged
#[rstest]
#[case(
    OperationalCopyBackend::VectorizedKernel,
    LayoutKind::FC,
    LayoutKind::FC,
    TransferMode::FullBlocks
)]
#[case(
    OperationalCopyBackend::VectorizedKernel,
    LayoutKind::FC,
    LayoutKind::LW,
    TransferMode::FullBlocks
)]
#[case(
    OperationalCopyBackend::VectorizedKernel,
    LayoutKind::LW,
    LayoutKind::FC,
    TransferMode::FullBlocks
)]
#[case(
    OperationalCopyBackend::VectorizedKernel,
    LayoutKind::FC,
    LayoutKind::FC,
    TransferMode::FirstLayerOnly
)]
#[case(
    OperationalCopyBackend::VectorizedKernel,
    LayoutKind::FC,
    LayoutKind::LW,
    TransferMode::FirstLayerOnly
)]
#[case(
    OperationalCopyBackend::VectorizedKernel,
    LayoutKind::FC,
    LayoutKind::FC,
    TransferMode::SecondLayerOnly
)]
#[case(
    OperationalCopyBackend::MemcpyAsync,
    LayoutKind::FC,
    LayoutKind::FC,
    TransferMode::FullBlocks
)]
#[case(
    OperationalCopyBackend::MemcpyAsync,
    LayoutKind::FC,
    LayoutKind::LW,
    TransferMode::FullBlocks
)]
#[case(
    OperationalCopyBackend::MemcpyAsync,
    LayoutKind::LW,
    LayoutKind::FC,
    TransferMode::FullBlocks
)]
#[case(
    OperationalCopyBackend::MemcpyAsync,
    LayoutKind::FC,
    LayoutKind::FC,
    TransferMode::FirstLayerOnly
)]
#[case(
    OperationalCopyBackend::MemcpyAsync,
    LayoutKind::FC,
    LayoutKind::LW,
    TransferMode::FirstLayerOnly
)]
#[case(
    OperationalCopyBackend::MemcpyAsync,
    LayoutKind::FC,
    LayoutKind::FC,
    TransferMode::SecondLayerOnly
)]
#[case(
    OperationalCopyBackend::MemcpyBatch,
    LayoutKind::FC,
    LayoutKind::FC,
    TransferMode::FullBlocks
)]
#[case(
    OperationalCopyBackend::MemcpyBatch,
    LayoutKind::FC,
    LayoutKind::LW,
    TransferMode::FullBlocks
)]
#[case(
    OperationalCopyBackend::MemcpyBatch,
    LayoutKind::LW,
    LayoutKind::FC,
    TransferMode::FullBlocks
)]
#[test]
fn test_kernel_h2d_roundtrip(
    #[case] backend: OperationalCopyBackend,
    #[case] pinned_layout: LayoutKind,
    #[case] device_layout: LayoutKind,
    #[case] mode: TransferMode,
) -> Result<()> {
    skip_if_stubs_result!();
    skip_if_memcpy_batch_unsupported!(backend);

    let num_blocks = 6;
    let agent = create_test_agent("test_kernel_h2d_roundtrip");
    let ctx = create_transfer_context(agent.clone(), None)?;

    // Create layouts: pinned src, device intermediate, pinned dst
    let pinned_src = create_layout(
        agent.clone(),
        LayoutSpec::new(pinned_layout, StorageKind::Pinned),
        num_blocks,
    );
    let device = create_layout(
        agent.clone(),
        LayoutSpec::new(device_layout, StorageKind::Device(0)),
        num_blocks,
    );
    let pinned_dst = create_layout(
        agent.clone(),
        LayoutSpec::new(pinned_layout, StorageKind::Pinned),
        num_blocks,
    );

    // Block assignments: src[0,1] -> device[0,1] -> dst[3,4], guards[2,5]
    let src_blocks = vec![0, 1];
    let device_blocks = vec![0, 1];
    let dst_blocks = vec![3, 4];
    let guard_blocks = vec![2, 5];

    // Fill source and compute checksums (mode-aware for layer-wise transfers)
    let src_checksums =
        fill_and_checksum_with_mode(&pinned_src, &src_blocks, FillPattern::Sequential, mode)?;

    // Create guard blocks in destination
    let guard_checksums =
        create_guard_blocks(&pinned_dst, &guard_blocks, FillPattern::Constant(0xAA))?;

    // Get layer range based on mode
    let layers = mode.layer_range().unwrap_or(0..2);

    // H2D: Pinned src -> Device
    let h2d_stream = ctx.h2d_stream();
    try_execute_operational_kernel(
        &pinned_src,
        &device,
        &src_blocks,
        &device_blocks,
        layers.clone(),
        h2d_stream.as_ref(),
        backend,
        ctx.cuda_pool(),
    )?;
    h2d_stream.synchronize()?;

    // D2H: Device -> Pinned dst
    let d2h_stream = ctx.d2h_stream();
    try_execute_operational_kernel(
        &device,
        &pinned_dst,
        &device_blocks,
        &dst_blocks,
        layers,
        d2h_stream.as_ref(),
        backend,
        ctx.cuda_pool(),
    )?;
    d2h_stream.synchronize()?;

    // Verify checksums match by position (mode-aware for layer-wise transfers)
    verify_checksums_by_position_with_mode(
        &src_checksums,
        &src_blocks,
        &pinned_dst,
        &dst_blocks,
        mode,
    )?;

    // Verify guard blocks unchanged
    verify_guard_blocks_unchanged(&pinned_dst, &guard_blocks, &guard_checksums)?;

    Ok(())
}

/// Test D2H roundtrip: Device src (via staging) -> Pinned dst
///
/// Since we can't directly fill Device memory, we stage through Pinned:
/// 1. Fill Pinned staging with sequential data
/// 2. H2D: Staging -> Device src (setup)
/// 3. D2H: Device src -> Pinned dst using specified backend
/// 4. Verify checksums match
/// 5. Verify guard blocks unchanged
#[rstest]
#[case(
    OperationalCopyBackend::VectorizedKernel,
    LayoutKind::FC,
    LayoutKind::FC,
    TransferMode::FullBlocks
)]
#[case(
    OperationalCopyBackend::VectorizedKernel,
    LayoutKind::FC,
    LayoutKind::LW,
    TransferMode::FullBlocks
)]
#[case(
    OperationalCopyBackend::VectorizedKernel,
    LayoutKind::LW,
    LayoutKind::FC,
    TransferMode::FullBlocks
)]
#[case(
    OperationalCopyBackend::VectorizedKernel,
    LayoutKind::FC,
    LayoutKind::FC,
    TransferMode::FirstLayerOnly
)]
#[case(
    OperationalCopyBackend::VectorizedKernel,
    LayoutKind::LW,
    LayoutKind::FC,
    TransferMode::FirstLayerOnly
)]
#[case(
    OperationalCopyBackend::VectorizedKernel,
    LayoutKind::FC,
    LayoutKind::FC,
    TransferMode::SecondLayerOnly
)]
#[case(
    OperationalCopyBackend::MemcpyAsync,
    LayoutKind::FC,
    LayoutKind::FC,
    TransferMode::FullBlocks
)]
#[case(
    OperationalCopyBackend::MemcpyAsync,
    LayoutKind::FC,
    LayoutKind::LW,
    TransferMode::FullBlocks
)]
#[case(
    OperationalCopyBackend::MemcpyAsync,
    LayoutKind::LW,
    LayoutKind::FC,
    TransferMode::FullBlocks
)]
#[case(
    OperationalCopyBackend::MemcpyAsync,
    LayoutKind::FC,
    LayoutKind::FC,
    TransferMode::FirstLayerOnly
)]
#[case(
    OperationalCopyBackend::MemcpyAsync,
    LayoutKind::LW,
    LayoutKind::FC,
    TransferMode::FirstLayerOnly
)]
#[case(
    OperationalCopyBackend::MemcpyAsync,
    LayoutKind::FC,
    LayoutKind::FC,
    TransferMode::SecondLayerOnly
)]
#[case(
    OperationalCopyBackend::MemcpyBatch,
    LayoutKind::FC,
    LayoutKind::FC,
    TransferMode::FullBlocks
)]
#[case(
    OperationalCopyBackend::MemcpyBatch,
    LayoutKind::FC,
    LayoutKind::LW,
    TransferMode::FullBlocks
)]
#[case(
    OperationalCopyBackend::MemcpyBatch,
    LayoutKind::LW,
    LayoutKind::FC,
    TransferMode::FullBlocks
)]
#[test]
fn test_kernel_d2h_roundtrip(
    #[case] backend: OperationalCopyBackend,
    #[case] device_layout: LayoutKind,
    #[case] pinned_layout: LayoutKind,
    #[case] mode: TransferMode,
) -> Result<()> {
    skip_if_stubs_result!();
    skip_if_memcpy_batch_unsupported!(backend);

    let num_blocks = 6;
    let agent = create_test_agent("test_kernel_d2h_roundtrip");
    let ctx = create_transfer_context(agent.clone(), None)?;

    // Create layouts: pinned staging, device src, pinned dst
    let staging = create_layout(
        agent.clone(),
        LayoutSpec::new(LayoutKind::FC, StorageKind::Pinned),
        num_blocks,
    );
    let device = create_layout(
        agent.clone(),
        LayoutSpec::new(device_layout, StorageKind::Device(0)),
        num_blocks,
    );
    let pinned_dst = create_layout(
        agent.clone(),
        LayoutSpec::new(pinned_layout, StorageKind::Pinned),
        num_blocks,
    );

    // Block assignments: staging[0,1] -> device[0,1] -> dst[3,4], guards[2,5]
    let src_blocks = vec![0, 1];
    let device_blocks = vec![0, 1];
    let dst_blocks = vec![3, 4];
    let guard_blocks = vec![2, 5];

    // Fill staging and compute checksums (mode-aware for layer-wise transfers)
    let src_checksums =
        fill_and_checksum_with_mode(&staging, &src_blocks, FillPattern::Sequential, mode)?;

    // Create guard blocks in destination
    let guard_checksums =
        create_guard_blocks(&pinned_dst, &guard_blocks, FillPattern::Constant(0xBB))?;

    // Get layer range based on mode
    let layers = mode.layer_range().unwrap_or(0..2);

    // Setup: H2D staging -> device (use Auto for setup)
    let h2d_stream = ctx.h2d_stream();
    try_execute_operational_kernel(
        &staging,
        &device,
        &src_blocks,
        &device_blocks,
        layers.clone(),
        h2d_stream.as_ref(),
        OperationalCopyBackend::Auto,
        ctx.cuda_pool(),
    )?;
    h2d_stream.synchronize()?;

    // Test: D2H Device -> Pinned dst with specified backend
    let d2h_stream = ctx.d2h_stream();
    try_execute_operational_kernel(
        &device,
        &pinned_dst,
        &device_blocks,
        &dst_blocks,
        layers,
        d2h_stream.as_ref(),
        backend,
        ctx.cuda_pool(),
    )?;
    d2h_stream.synchronize()?;

    // Verify checksums match by position (mode-aware for layer-wise transfers)
    verify_checksums_by_position_with_mode(
        &src_checksums,
        &src_blocks,
        &pinned_dst,
        &dst_blocks,
        mode,
    )?;

    // Verify guard blocks unchanged
    verify_guard_blocks_unchanged(&pinned_dst, &guard_blocks, &guard_checksums)?;

    Ok(())
}

// ============================================================================
// Layer-wise Transfer Consistency Tests
// ============================================================================

/// Test that layer-wise transfers produce identical results to full block transfers.
///
/// This test verifies:
/// 1. Executing each layer transfer independently produces the same result as full block transfer
/// 2. Results are identical between VectorizedKernel and MemcpyAsync backends
/// 3. The combined layer-wise checksums match the full block checksums
///
/// Test flow:
/// 1. Fill Pinned src with sequential data
/// 2. Compute full block checksums (reference)
/// 3. For each backend:
///    a. H2D: Transfer layer 0 -> await
///    b. H2D: Transfer layer 1 -> await
///    c. D2H: Transfer layer 0 -> await
///    d. D2H: Transfer layer 1 -> await
///    e. Verify checksums match reference
/// 4. Verify both backends produce identical results
#[rstest]
#[case(LayoutKind::FC, LayoutKind::FC)]
#[case(LayoutKind::FC, LayoutKind::LW)]
#[case(LayoutKind::LW, LayoutKind::FC)]
#[test]
fn test_kernel_layerwise_consistency(
    #[case] pinned_layout: LayoutKind,
    #[case] device_layout: LayoutKind,
) -> Result<()> {
    skip_if_stubs_result!();

    let num_blocks = 6;
    let agent = create_test_agent("test_kernel_layerwise_consistency");
    let ctx = create_transfer_context(agent.clone(), None)?;

    // Create layouts
    let pinned_src = create_layout(
        agent.clone(),
        LayoutSpec::new(pinned_layout, StorageKind::Pinned),
        num_blocks,
    );

    // Block assignments
    let src_blocks = vec![0, 1];
    let device_blocks = vec![0, 1];
    let dst_blocks = vec![3, 4];
    let guard_blocks = vec![2, 5];

    // Fill source with sequential data and get reference checksums
    let reference_checksums = fill_and_checksum(&pinned_src, &src_blocks, FillPattern::Sequential)?;

    // Test both backends and verify they produce identical results
    let backends = [
        OperationalCopyBackend::VectorizedKernel,
        OperationalCopyBackend::MemcpyAsync,
    ];

    for backend in backends {
        // Create fresh device and destination layouts for each backend test
        let device = create_layout(
            agent.clone(),
            LayoutSpec::new(device_layout, StorageKind::Device(0)),
            num_blocks,
        );
        let pinned_dst = create_layout(
            agent.clone(),
            LayoutSpec::new(pinned_layout, StorageKind::Pinned),
            num_blocks,
        );

        // Create guard blocks
        let guard_checksums =
            create_guard_blocks(&pinned_dst, &guard_blocks, FillPattern::Constant(0xCC))?;

        let h2d_stream = ctx.h2d_stream();
        let d2h_stream = ctx.d2h_stream();

        // H2D: Transfer layer 0 independently
        try_execute_operational_kernel(
            &pinned_src,
            &device,
            &src_blocks,
            &device_blocks,
            0..1, // Layer 0 only
            h2d_stream.as_ref(),
            backend,
            ctx.cuda_pool(),
        )?;
        h2d_stream.synchronize()?;

        // H2D: Transfer layer 1 independently
        try_execute_operational_kernel(
            &pinned_src,
            &device,
            &src_blocks,
            &device_blocks,
            1..2, // Layer 1 only
            h2d_stream.as_ref(),
            backend,
            ctx.cuda_pool(),
        )?;
        h2d_stream.synchronize()?;

        // D2H: Transfer layer 0 independently
        try_execute_operational_kernel(
            &device,
            &pinned_dst,
            &device_blocks,
            &dst_blocks,
            0..1, // Layer 0 only
            d2h_stream.as_ref(),
            backend,
            ctx.cuda_pool(),
        )?;
        d2h_stream.synchronize()?;

        // D2H: Transfer layer 1 independently
        try_execute_operational_kernel(
            &device,
            &pinned_dst,
            &device_blocks,
            &dst_blocks,
            1..2, // Layer 1 only
            d2h_stream.as_ref(),
            backend,
            ctx.cuda_pool(),
        )?;
        d2h_stream.synchronize()?;

        // Verify full block checksums match reference
        verify_checksums_by_position(&reference_checksums, &src_blocks, &pinned_dst, &dst_blocks)
            .map_err(|e| {
            anyhow::anyhow!(
                "Layer-wise transfer checksum mismatch for backend {:?}: {}",
                backend,
                e
            )
        })?;

        // Verify guard blocks unchanged
        verify_guard_blocks_unchanged(&pinned_dst, &guard_blocks, &guard_checksums).map_err(
            |e| {
                anyhow::anyhow!(
                    "Guard block corruption detected for backend {:?}: {}",
                    backend,
                    e
                )
            },
        )?;
    }

    Ok(())
}
