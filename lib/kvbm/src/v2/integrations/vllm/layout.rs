// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::sync::Arc;

use anyhow::{Result, bail};

use crate::physical::layout::{BlockDimension, LayoutConfig};
use dynamo_memory::TensorDescriptor;

pub fn determine_kv_layout(
    num_device_blocks: usize,
    page_size: usize,
    dtype_width_bytes: usize,
    kv_tensors: &[Arc<dyn TensorDescriptor>],
) -> Result<()> {
    let first_tensor = kv_tensors
        .first()
        .ok_or(anyhow::anyhow!("No tensors provided"))?;
    let shape = validate_tensor_shapes(first_tensor, kv_tensors)?;

    let mut builder = LayoutConfig::builder();

    builder.num_blocks(num_device_blocks);
    builder.num_layers(kv_tensors.len());
    builder.page_size(page_size);
    builder.dtype_width_bytes(dtype_width_bytes);

    let _block_dim = if shape[0] >= num_device_blocks {
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

    if shape[2] != page_size && shape[3] != page_size {
        bail!(
            "Unexpected tensor shape: {:?}; expected page_size: {page_size} to be present in the second or third dimension",
            shape
        );
    }

    let inner_dim = shape[2..].iter().product::<usize>() / page_size;
    builder.inner_dim(inner_dim);

    let _layout_config = builder.build()?;

    Ok(())
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
