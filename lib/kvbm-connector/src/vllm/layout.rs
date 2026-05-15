// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::sync::Arc;

use anyhow::{Result, bail};

use dynamo_memory::TensorDescriptor;
use kvbm_physical::layout::{BlockDimension, LayoutConfig};

/// Determine the KV cache layout configuration and block dimension from tensor shapes.
///
/// # Arguments
/// * `num_device_blocks` - Expected number of device blocks (from vLLM's cache config)
/// * `page_size` - Block/page size for the KV cache
/// * `dtype_width_bytes` - Data type width in bytes (e.g., 2 for fp16)
/// * `kv_tensors` - KV cache tensors (one per layer), all must have the same shape
/// * `explicit_outer_dim` - Optional caller-supplied outer_dim. When `Some(_)`
///   `explicit_inner_dim` must also be `Some(_)`; the pair bypasses shape inference.
///   For MLA models Python should pass `Some(1)` since the cache is a fused latent
///   and shape-based inference is ambiguous (see `mla_deepseek_v2_shape` test).
/// * `explicit_inner_dim` - Optional caller-supplied inner_dim (paired with `explicit_outer_dim`).
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
    explicit_outer_dim: Option<usize>,
    explicit_inner_dim: Option<usize>,
) -> Result<(LayoutConfig, BlockDimension)> {
    // Cross-field coupling: explicit dims must be provided together.
    match (explicit_outer_dim, explicit_inner_dim) {
        (Some(o), _) if !(1..=2).contains(&o) => bail!(
            "explicit outer_dim must be in [1, 2] (1 = MLA fused, 2 = standard K/V split); got {}",
            o
        ),
        (_, Some(0)) => bail!("explicit inner_dim must be > 0"),
        (Some(_), None) | (None, Some(_)) => bail!(
            "explicit outer_dim and inner_dim must be provided together; got outer_dim={:?}, inner_dim={:?}",
            explicit_outer_dim,
            explicit_inner_dim
        ),
        _ => {}
    }

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

    // Locate the blocks dimension from shape (independent of explicit dims).
    let block_dim = if shape[0] >= num_device_blocks {
        BlockDimension::BlockIsFirstDim
    } else if shape[1] >= num_device_blocks {
        BlockDimension::BlockIsSecondDim
    } else {
        bail!(
            "Unexpected tensor shape: {:?}; expected num_device_blocks: {num_device_blocks} to be present in the first or second dimension",
            shape
        );
    };

    // Resolve outer_dim / inner_dim.
    //
    // Preferred path: caller (Python) supplied explicit dims by reading the vLLM
    // model config (`use_mla`). This is unambiguous and future-proof against
    // vLLM shape changes.
    //
    // Fallback path: infer from shape. Standard layouts present an explicit K/V
    // axis (outer_dim ∈ {1, 2}); MLA layouts omit it and surface
    // `[n_blocks, page_size, latent_dim]` (or the outer-contiguous equivalent),
    // so the neighbouring candidate lands on `page_size` and exceeds 2 — treat
    // as MLA and set outer_dim=1.
    let (outer_dim, inner_dim) =
        if let (Some(o), Some(i)) = (explicit_outer_dim, explicit_inner_dim) {
            (o, i)
        } else {
            let candidate_outer = match block_dim {
                BlockDimension::BlockIsFirstDim => shape[1],
                BlockDimension::BlockIsSecondDim => shape[0],
            };
            let (outer_dim, content_dims_start) = if candidate_outer <= 2 {
                (candidate_outer, 2)
            } else {
                (1, 1)
            };
            let inner_dim = shape[content_dims_start..].iter().product::<usize>() / page_size;
            (outer_dim, inner_dim)
        };
    builder.outer_dim(outer_dim);
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

#[cfg(test)]
mod tests {
    use super::*;

    use std::any::Any;

    use dynamo_memory::nixl::NixlDescriptor;
    use dynamo_memory::{MemoryDescriptor, StorageKind};

    #[derive(Debug)]
    struct TestTensor {
        shape: Vec<usize>,
        stride: Vec<usize>,
    }

    impl TestTensor {
        fn arc(shape: Vec<usize>) -> Arc<dyn TensorDescriptor> {
            // Row-major strides (unused by layout inference, but the trait requires them).
            let mut stride = vec![1usize; shape.len()];
            for i in (0..shape.len().saturating_sub(1)).rev() {
                stride[i] = stride[i + 1] * shape[i + 1];
            }
            Arc::new(Self { shape, stride })
        }
    }

    impl MemoryDescriptor for TestTensor {
        fn addr(&self) -> usize {
            0
        }
        fn size(&self) -> usize {
            0
        }
        fn storage_kind(&self) -> StorageKind {
            StorageKind::System
        }
        fn as_any(&self) -> &dyn Any {
            self
        }
        fn nixl_descriptor(&self) -> Option<NixlDescriptor> {
            None
        }
    }

    impl TensorDescriptor for TestTensor {
        fn shape(&self) -> &[usize] {
            &self.shape
        }
        fn stride(&self) -> &[usize] {
            &self.stride
        }
        fn element_size(&self) -> usize {
            2
        }
    }

    fn layers(shape: Vec<usize>, n: usize) -> Vec<Arc<dyn TensorDescriptor>> {
        (0..n).map(|_| TestTensor::arc(shape.clone())).collect()
    }

    /// Standard 4D cache with K/V split, block-first: [n_blocks, 2, page_size, inner_dim].
    #[test]
    fn inferred_standard_kv_split_block_first() {
        let tensors = layers(vec![100, 2, 16, 128], 4);
        let (cfg, block_dim) = determine_kv_layout(100, 16, 2, &tensors, None, None).unwrap();
        assert_eq!(cfg.outer_dim, 2);
        assert_eq!(cfg.inner_dim, 128);
        assert!(matches!(block_dim, BlockDimension::BlockIsFirstDim));
    }

    /// Standard 4D cache with K/V split, outer-first: [2, n_blocks, page_size, inner_dim].
    #[test]
    fn inferred_standard_kv_split_outer_first() {
        let tensors = layers(vec![2, 100, 16, 128], 4);
        let (cfg, block_dim) = determine_kv_layout(100, 16, 2, &tensors, None, None).unwrap();
        assert_eq!(cfg.outer_dim, 2);
        assert_eq!(cfg.inner_dim, 128);
        assert!(matches!(block_dim, BlockDimension::BlockIsSecondDim));
    }

    /// MLA cache (DeepSeek-V2-Lite reproduction), inference-only: 3D
    /// `[n_blocks, page_size, latent_dim]` with latent_dim = kv_lora_rank +
    /// qk_rope_head_dim = 512 + 64 = 576. Candidate outer_dim falls on
    /// page_size=16 (>2) so we fall back to outer_dim=1.
    #[test]
    fn inferred_mla_deepseek_v2_shape() {
        let tensors = layers(vec![26847, 16, 576], 27);
        let (cfg, block_dim) = determine_kv_layout(26847, 16, 2, &tensors, None, None).unwrap();
        assert_eq!(cfg.outer_dim, 1);
        assert_eq!(cfg.inner_dim, 576);
        assert!(matches!(block_dim, BlockDimension::BlockIsFirstDim));
    }

    /// Fused-outer MLA variant: outer_dim=1 explicitly in the tensor shape,
    /// block-first 4D. Exercises the `candidate_outer == 1` branch in inference.
    #[test]
    fn inferred_mla_explicit_outer_dim_one_in_shape() {
        let tensors = layers(vec![100, 1, 16, 576], 4);
        let (cfg, _) = determine_kv_layout(100, 16, 2, &tensors, None, None).unwrap();
        assert_eq!(cfg.outer_dim, 1);
        assert_eq!(cfg.inner_dim, 576);
    }

    /// Sanity: n_blocks present in neither of the first two dims is rejected.
    #[test]
    fn rejects_missing_block_dim() {
        let tensors = layers(vec![8, 16, 128], 4);
        assert!(determine_kv_layout(100, 16, 2, &tensors, None, None).is_err());
    }

    /// Caller supplies explicit MLA dims (outer_dim=1, inner_dim=latent). This is the
    /// preferred path — Python derives it from vLLM's `model_config.use_mla`, so the
    /// Rust side never has to guess from a potentially-ambiguous shape.
    #[test]
    fn explicit_mla_dims_are_used_directly() {
        let tensors = layers(vec![26847, 16, 576], 27);
        let (cfg, _) = determine_kv_layout(26847, 16, 2, &tensors, Some(1), Some(576)).unwrap();
        assert_eq!(cfg.outer_dim, 1);
        assert_eq!(cfg.inner_dim, 576);
    }

    /// Explicit standard K/V split: caller asserts outer_dim=2, inner_dim=128.
    #[test]
    fn explicit_standard_dims_are_used_directly() {
        let tensors = layers(vec![100, 2, 16, 128], 4);
        let (cfg, _) = determine_kv_layout(100, 16, 2, &tensors, Some(2), Some(128)).unwrap();
        assert_eq!(cfg.outer_dim, 2);
        assert_eq!(cfg.inner_dim, 128);
    }

    /// Coupling check: passing only outer_dim (without inner_dim) is an error.
    #[test]
    fn explicit_outer_without_inner_is_err() {
        let tensors = layers(vec![100, 2, 16, 128], 4);
        let err = determine_kv_layout(100, 16, 2, &tensors, Some(1), None)
            .unwrap_err()
            .to_string();
        assert!(err.contains("must be provided together"), "got: {err}");
    }

    /// Coupling check: passing only inner_dim (without outer_dim) is an error.
    #[test]
    fn explicit_inner_without_outer_is_err() {
        let tensors = layers(vec![100, 2, 16, 128], 4);
        let err = determine_kv_layout(100, 16, 2, &tensors, None, Some(128))
            .unwrap_err()
            .to_string();
        assert!(err.contains("must be provided together"), "got: {err}");
    }

    /// Range check: outer_dim=3 is rejected at the API boundary (before layout build).
    #[test]
    fn explicit_outer_dim_out_of_range_is_err() {
        let tensors = layers(vec![100, 2, 16, 128], 4);
        let err = determine_kv_layout(100, 16, 2, &tensors, Some(3), Some(128))
            .unwrap_err()
            .to_string();
        assert!(err.contains("outer_dim"), "got: {err}");
    }

    /// Range check: inner_dim=0 is rejected.
    #[test]
    fn explicit_inner_dim_zero_is_err() {
        let tensors = layers(vec![100, 2, 16, 128], 4);
        let err = determine_kv_layout(100, 16, 2, &tensors, Some(2), Some(0))
            .unwrap_err()
            .to_string();
        assert!(err.contains("inner_dim"), "got: {err}");
    }
}
