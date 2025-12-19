// SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Unit tests for CUDA kernel compatibility and execution.

use super::{NixlAgent, create_test_agent, skip_if_stubs};
use crate::v2::physical::layout::{BlockDimension, LayoutConfig, PhysicalLayout};
use crate::v2::physical::transfer::executor::cuda::try_execute_operational_kernel;
use crate::v2::physical::transfer::{StorageKind, TransferContext};
use dynamo_kvbm_kernels::tensor_kernels::OperationalCopyBackend;

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
