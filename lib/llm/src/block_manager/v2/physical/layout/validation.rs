// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Tensor validation utilities for layout creation.

use anyhow::{Result, anyhow};
use std::sync::Arc;

use crate::block_manager::v2::memory::TorchTensor;

/// Format of tensor layout (for future TP translation).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TensorFormat {
    /// NHD format: [N, H, D] where N=block_size, H=heads, D=hidden
    NHD,
    /// HND format: [H, N, D] where H=heads, N=block_size, D=hidden
    HND,
    /// Unknown or ambiguous format
    Unknown,
}

/// Validate tensor strides and detect format.
///
/// This function checks that tensor strides are monotonically decreasing,
/// which ensures tensor-contiguous layout. The stride validation is flexible
/// at the inner dimension boundary to accommodate different layouts.
///
/// Additionally, it attempts to detect whether the layout is NHD or HND format,
/// which is important for future tensor parallel (TP) translation.
///
/// # Arguments
/// * `tensors` - Slice of tensors to validate
///
/// # Returns
/// The detected tensor format (NHD, HND, or Unknown)
pub fn validate_tensor_strides(tensors: &[Arc<dyn TorchTensor>]) -> Result<TensorFormat> {
    if tensors.is_empty() {
        return Err(anyhow!("Cannot validate empty tensor list"));
    }

    let mut format = TensorFormat::Unknown;

    for tensor in tensors {
        let stride = tensor.stride();
        let shape = tensor.shape();

        if stride.len() < 2 {
            return Err(anyhow!(
                "Tensor must have at least 2 dimensions, got stride: {:?}",
                stride
            ));
        }

        // Check monotonic decreasing stride
        // Note: We're flexible at the combined inner dimension boundary as per requirements
        let mut prev_stride = usize::MAX;
        for (i, &current_stride) in stride.iter().enumerate() {
            if current_stride > prev_stride {
                return Err(anyhow!(
                    "Tensor strides must be monotonically decreasing (until inner dimension). \
                     Got stride: {:?} at position {}",
                    stride,
                    i
                ));
            }
            prev_stride = current_stride;
        }

        // Attempt to detect NHD vs HND format based on shape and stride patterns
        // This is a heuristic and may need refinement based on actual usage
        if shape.len() >= 3 {
            // If the first dimension stride is smaller than the second, likely HND
            // If the first dimension stride is larger than the second, likely NHD
            if stride[0] < stride[1] {
                format = TensorFormat::HND;
            } else if stride[0] > stride[1] {
                format = TensorFormat::NHD;
            }
        }
    }

    Ok(format)
}

/// Validate that all tensors have consistent shapes.
///
/// # Arguments
/// * `tensors` - Slice of tensors to validate
///
/// # Returns
/// The common shape shared by all tensors
pub fn validate_tensor_shapes(tensors: &[Arc<dyn TorchTensor>]) -> Result<Vec<usize>> {
    if tensors.is_empty() {
        return Err(anyhow!("Cannot validate empty tensor list"));
    }

    let first_shape = tensors[0].shape();

    for tensor in &tensors[1..] {
        if tensor.shape() != first_shape {
            return Err(anyhow!(
                "All tensors must have the same shape. Expected {:?}, got {:?}",
                first_shape,
                tensor.shape()
            ));
        }
    }

    Ok(first_shape)
}

#[allow(dead_code)]
pub fn determine_compressed_shape(shape: &[usize]) -> usize {
    shape.iter().product()
}

#[cfg(test)]
mod tests {

    // Note: These tests would require mock TorchTensor implementations
    // which we can add if needed for testing infrastructure
}
