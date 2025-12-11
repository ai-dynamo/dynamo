// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::sync::Arc;

use anyhow::{Result, bail};

use crate::physical::layout::{BlockDimension, LayoutConfig};
use dynamo_memory::TensorDescriptor;

/// Determine the KV cache layout configuration and block dimension from tensor shapes.
///
/// # Arguments
/// * `num_device_blocks` - Expected number of device blocks (from vLLM's cache config)
/// * `page_size` - Block/page size for the KV cache
/// * `dtype_width_bytes` - Data type width in bytes (e.g., 2 for fp16)
/// * `kv_tensors` - KV cache tensors (one per layer), all must have the same shape
///
/// # Returns
/// A tuple of `(LayoutConfig, BlockDimension)` that describes:
/// - `LayoutConfig`: Configuration for physical layout construction
/// - `BlockDimension`: Whether blocks are in the first or second tensor dimension
pub fn determine_kv_layout(
    num_device_blocks: usize,
    page_size: usize,
    dtype_width_bytes: usize,
    kv_tensors: &[Arc<dyn TensorDescriptor>],
) -> Result<(LayoutConfig, BlockDimension)> {
    let first_tensor = kv_tensors
        .first()
        .ok_or(anyhow::anyhow!("No tensors provided"))?;
    let shape = validate_tensor_shapes(first_tensor, kv_tensors)?;

    let mut builder = LayoutConfig::builder();

    builder.num_blocks(num_device_blocks);
    builder.num_layers(kv_tensors.len());
    builder.page_size(page_size);
    builder.dtype_width_bytes(dtype_width_bytes);

    // Validate shape dimension count
    if shape.len() < 3 {
        bail!(
            "Tensor must have at least 3 dimensions (blocks, heads, head_dim), got {:?}",
            shape
        );
    }

    // Log strides for debugging (matching V1 behavior)
    for tensor in kv_tensors {
        let stride = tensor.stride();
        tracing::debug!("stride: {:?}", stride);
    }

    let block_dim = if shape[0] >= num_device_blocks {
        builder.outer_dim(shape[1]);
        BlockDimension::BlockIsFirstDim
    } else if shape[1] >= num_device_blocks {
        builder.outer_dim(shape[0]);
        BlockDimension::BlockIsSecondDim
    } else {
        bail!(
            "Unexpected tensor shape: {:?}; expected num_device_blocks: {num_device_blocks} to be present in the first or second dimension",
            shape
        );
    };

    // Calculate inner dimension (relaxed validation compared to strict page_size check)
    let inner_dim = shape[2..].iter().product::<usize>() / page_size;
    builder.inner_dim(inner_dim);

    let layout_config = builder.build()?;

    tracing::debug!(?layout_config, "Determined KV layout");

    Ok((layout_config, block_dim))
}

/// Validate tensors
fn validate_tensor_shapes(
    first: &Arc<dyn TensorDescriptor>,
    tensors: &[Arc<dyn TensorDescriptor>],
) -> Result<Vec<usize>> {
    let shape = first.shape();

    if !tensors.iter().all(|tensor| shape == tensor.shape()) {
        return Err(anyhow::anyhow!(
            "All tensors must have the same shape! Expected {:?}",
            shape
        ));
    }

    Ok(shape.to_vec())
}
