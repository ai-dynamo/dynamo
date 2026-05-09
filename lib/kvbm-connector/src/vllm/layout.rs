// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Translate a labeled [`KvDimLayout`] from Python into a `LayoutConfig` and
//! `BlockDimension` for the physical layer.
//!
//! Each axis of every registered KV-cache tensor is named explicitly by
//! Python (via `lib/bindings/kvbm/python/kvbm/v2/vllm/dim_probe.py`). This
//! module enforces the contract: tensor `shape()` matches the labeled
//! `sizes()`, the labels are coherent (single `Block`, etc.), and the
//! resulting `LayoutConfig.bytes_per_block() * num_blocks` matches each
//! tensor's `numel * element_size` to within layout-arithmetic accuracy.

use std::sync::Arc;

use anyhow::{Result, bail};

use dynamo_memory::TensorDescriptor;
use kvbm_common::{KvDim, KvDimLayout};
use kvbm_physical::layout::{BlockDimension, LayoutConfig};

/// Determine the KV cache layout configuration and block dimension from a
/// caller-supplied [`KvDimLayout`].
///
/// # Arguments
/// * `num_device_blocks` - Expected number of device blocks (from vLLM's
///   per-rank `kv_cache_config`). Used both as the value of `LayoutConfig.
///   num_blocks` and as a cross-check against the layout's `Block` axis
///   size — they must agree.
/// * `dtype_width_bytes` - KV-cache element width in bytes (e.g. 2 for fp16).
/// * `kv_tensors` - KV cache tensors. For per-layer registration there is
///   one tensor per layer; for cross-layer registration there is exactly
///   one tensor and the layout has a `KvDim::Layer` axis.
/// * `dim_layout` - Per-axis labels and sizes describing the tensors. Built
///   in Python by probing `attn_backend.get_kv_cache_shape(...)` with
///   sentinel values.
///
/// # Returns
/// `(LayoutConfig, BlockDimension)` — the `LayoutConfig` is fully
/// determined by the labels (`num_blocks`, `outer_dim`, `page_size`,
/// `inner_dim`, `num_heads`); the `BlockDimension` records whether the
/// `Block` axis sat at tensor position 0 or 1 (per-layer) — only meaningful
/// for layer-separate physical layouts.
pub fn determine_kv_layout(
    num_device_blocks: usize,
    dtype_width_bytes: usize,
    kv_tensors: &[Arc<dyn TensorDescriptor>],
    dim_layout: &KvDimLayout,
) -> Result<(LayoutConfig, BlockDimension)> {
    if kv_tensors.is_empty() {
        bail!("determine_kv_layout: no tensors provided");
    }

    // Cross-check: every tensor's shape must equal the labeled sizes.
    for (i, tensor) in kv_tensors.iter().enumerate() {
        let shape = tensor.shape();
        if shape.len() != dim_layout.sizes().len() {
            bail!(
                "tensor {i}: rank {} does not match labeled rank {} (shape {:?}, dims {:?})",
                shape.len(),
                dim_layout.sizes().len(),
                shape,
                dim_layout.dims(),
            );
        }
        for (axis, (&actual, (&expected, label))) in shape
            .iter()
            .zip(dim_layout.sizes().iter().zip(dim_layout.dims().iter()))
            .enumerate()
        {
            if actual != expected {
                bail!(
                    "tensor {i} axis {axis} ({label}): shape size {actual} != labeled size {expected}",
                );
            }
        }
    }

    // Cross-check: Block size must equal num_device_blocks. The layout is
    // the freshly-probed truth, but the caller still asserts the number for
    // belt-and-braces — surface mismatches loudly.
    let labeled_blocks = dim_layout.size_of(KvDim::Block).ok_or_else(|| {
        anyhow::anyhow!("determine_kv_layout: KvDimLayout is missing a Block axis")
    })?;
    if labeled_blocks != num_device_blocks {
        bail!("Block axis size ({labeled_blocks}) != num_device_blocks ({num_device_blocks})",);
    }

    // Milestone 1 supports per-layer registration only. Reject any
    // `KvDim::Layer` axis up front: the downstream `PhysicalLayoutBuilder
    // ::layer_separate(...)` path expects N tensors (one per layer), but a
    // Layer-axis layout describes a single cross-layer tensor. Milestone 2
    // (cross-layer / fully-contiguous default) will branch the build path
    // on this; until then a Layer-axis caller would only fail later at
    // `LayerSeparateLayout::new_with_block_layout` with a confusing
    // `memory.len() != num_layers` error.
    if dim_layout.position(KvDim::Layer).is_some() {
        bail!(
            "KvDim::Layer is not supported in Milestone 1 (per-layer registration only); \
             cross-layer / fully-contiguous registration lands in Milestone 2",
        );
    }

    // Locate the Block axis. For per-layer tensors only positions 0 or 1
    // are supported by the downstream layer-separate physical layout.
    let block_pos = dim_layout.block_axis()?;
    let block_dim = match block_pos {
        0 => BlockDimension::BlockIsFirstDim,
        1 => BlockDimension::BlockIsSecondDim,
        n => bail!(
            "Block axis at position {n} is not supported (per-layer registration expects 0 or 1)",
        ),
    };

    // Derive LayoutConfig from labels.
    let outer_dim = dim_layout.outer_size();
    let page_size = dim_layout.page_size()?;
    let inner_dim = dim_layout
        .inner_elements()
        .ok_or_else(|| anyhow::anyhow!("KvDimLayout has neither HeadSize nor Payload axis"))?;
    // Per-layer mode only in M1: `num_layers` is the tensor count.
    let num_layers = kv_tensors.len();

    let mut builder = LayoutConfig::builder();
    builder.num_blocks(num_device_blocks);
    builder.num_layers(num_layers);
    builder.outer_dim(outer_dim);
    builder.page_size(page_size);
    builder.inner_dim(inner_dim);
    builder.dtype_width_bytes(dtype_width_bytes);
    if let Some(nh) = dim_layout.head_count() {
        builder.num_heads(Some(nh));
    }

    // Field-level `#[validate]` attrs run at physical-layout build time
    // (`FullyContiguousLayout::new_internal` / `LayerSeparateLayout::new_internal`),
    // so we don't re-run them here. Cross-field invariants (Block size,
    // total bytes) are checked below.
    let layout_config = builder.build()?;

    // Per-tensor total-bytes cross-check: catches `dtype_width_bytes`
    // discrepancies and inner-dim arithmetic errors that the per-axis
    // size check would miss (the per-axis check works in element counts,
    // not bytes). Per-layer mode only — each tensor is one layer's worth
    // of `num_blocks * outer * page * inner * dtype_bytes`.
    let per_layer_bytes = num_device_blocks * outer_dim * page_size * inner_dim * dtype_width_bytes;
    for (i, tensor) in kv_tensors.iter().enumerate() {
        let observed = tensor.shape().iter().product::<usize>() * tensor.element_size();
        if observed != per_layer_bytes {
            bail!(
                "tensor {i}: total bytes {observed} != layout expectation {per_layer_bytes} \
                 (num_blocks={num_device_blocks}, num_layers={num_layers}, outer={outer_dim}, \
                  page={page_size}, inner={inner_dim}, dtype_bytes={dtype_width_bytes})",
            );
        }
    }

    tracing::debug!(
        ?layout_config,
        ?block_dim,
        ?dim_layout,
        "Resolved KV layout from labeled dim layout"
    );

    Ok((layout_config, block_dim))
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
        element_size: usize,
    }

    impl TestTensor {
        fn arc(shape: Vec<usize>, element_size: usize) -> Arc<dyn TensorDescriptor> {
            // Row-major strides (unused by layout inference, but the trait requires them).
            let mut stride = vec![1usize; shape.len()];
            for i in (0..shape.len().saturating_sub(1)).rev() {
                stride[i] = stride[i + 1] * shape[i + 1];
            }
            Arc::new(Self {
                shape,
                stride,
                element_size,
            })
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
            self.element_size
        }
    }

    fn layers(shape: Vec<usize>, n: usize, element_size: usize) -> Vec<Arc<dyn TensorDescriptor>> {
        (0..n)
            .map(|_| TestTensor::arc(shape.clone(), element_size))
            .collect()
    }

    fn nhd_per_layer_layout(
        num_blocks: usize,
        page_size: usize,
        nh: usize,
        hd: usize,
    ) -> KvDimLayout {
        // FlashAttn NHD per-layer: (2, num_blocks, page, nh, hd).
        KvDimLayout::new(
            vec![
                KvDim::Outer,
                KvDim::Block,
                KvDim::Page,
                KvDim::HeadCount,
                KvDim::HeadSize,
            ],
            vec![2, num_blocks, page_size, nh, hd],
        )
        .unwrap()
    }

    /// FlashAttn NHD per-layer: outer-first, Block at axis 1. Resolves to
    /// a layer-separate `LayoutConfig` with `outer_dim=2`, `inner_dim=nh*hd`.
    #[test]
    fn flashattn_nhd_per_layer_resolves() {
        let n_blocks = 1024;
        let page = 16;
        let nh = 8;
        let hd = 128;
        let dtype = 2;
        let dim_layout = nhd_per_layer_layout(n_blocks, page, nh, hd);
        let tensors = layers(vec![2, n_blocks, page, nh, hd], 32, dtype);

        let (cfg, block_dim) = determine_kv_layout(n_blocks, dtype, &tensors, &dim_layout).unwrap();

        assert_eq!(cfg.num_blocks, n_blocks);
        assert_eq!(cfg.num_layers, 32);
        assert_eq!(cfg.outer_dim, 2);
        assert_eq!(cfg.page_size, page);
        assert_eq!(cfg.inner_dim, nh * hd);
        assert_eq!(cfg.dtype_width_bytes, dtype);
        assert_eq!(cfg.num_heads, Some(nh));
        assert!(matches!(block_dim, BlockDimension::BlockIsSecondDim));
    }

    /// FlashInfer NHD per-layer: block-first, Block at axis 0.
    /// `(num_blocks, 2, page, nh, hd)`.
    #[test]
    fn flashinfer_nhd_per_layer_resolves() {
        let n_blocks = 1024;
        let page = 16;
        let dtype = 2;
        let dim_layout = KvDimLayout::new(
            vec![
                KvDim::Block,
                KvDim::Outer,
                KvDim::Page,
                KvDim::HeadCount,
                KvDim::HeadSize,
            ],
            vec![n_blocks, 2, page, 8, 128],
        )
        .unwrap();
        let tensors = layers(vec![n_blocks, 2, page, 8, 128], 32, dtype);

        let (cfg, block_dim) = determine_kv_layout(n_blocks, dtype, &tensors, &dim_layout).unwrap();

        assert_eq!(cfg.outer_dim, 2);
        assert_eq!(cfg.inner_dim, 8 * 128);
        assert!(matches!(block_dim, BlockDimension::BlockIsFirstDim));
    }

    /// FlashAttn HND per-layer: block-first, HeadCount before Page.
    /// `(2, num_blocks, nh, page, hd)`. Note `block_dim` is still
    /// `BlockIsSecondDim`; the Page/HeadCount permutation is captured by
    /// `KvBlockLayout` (passed separately to the builder), not here.
    #[test]
    fn flashattn_hnd_per_layer_resolves() {
        let n_blocks = 1024;
        let page = 16;
        let dtype = 2;
        let dim_layout = KvDimLayout::new(
            vec![
                KvDim::Outer,
                KvDim::Block,
                KvDim::HeadCount,
                KvDim::Page,
                KvDim::HeadSize,
            ],
            vec![2, n_blocks, 8, page, 128],
        )
        .unwrap();
        let tensors = layers(vec![2, n_blocks, 8, page, 128], 32, dtype);

        let (cfg, _block_dim) =
            determine_kv_layout(n_blocks, dtype, &tensors, &dim_layout).unwrap();
        assert_eq!(cfg.outer_dim, 2);
        assert_eq!(cfg.page_size, page);
        // inner_dim is purely label-derived: HeadCount * HeadSize, regardless
        // of axis position. The HND vs NHD distinction is the
        // KvBlockLayout's job.
        assert_eq!(cfg.inner_dim, 8 * 128);
        assert_eq!(cfg.num_heads, Some(8));
    }

    /// DeepSeek-V2-Lite MLA per-layer: `(num_blocks, page, head_size)`.
    /// No `Outer`, no `HeadCount` — `outer_dim` defaults to 1.
    #[test]
    fn mla_per_layer_resolves() {
        let n_blocks = 26847;
        let page = 16;
        let head_size = 576; // kv_lora_rank + qk_rope_head_dim
        let dtype = 2;
        let dim_layout = KvDimLayout::new(
            vec![KvDim::Block, KvDim::Page, KvDim::HeadSize],
            vec![n_blocks, page, head_size],
        )
        .unwrap();
        let tensors = layers(vec![n_blocks, page, head_size], 27, dtype);

        let (cfg, block_dim) = determine_kv_layout(n_blocks, dtype, &tensors, &dim_layout).unwrap();

        assert_eq!(cfg.outer_dim, 1);
        assert_eq!(cfg.page_size, page);
        assert_eq!(cfg.inner_dim, head_size);
        assert_eq!(cfg.num_heads, None);
        assert!(matches!(block_dim, BlockDimension::BlockIsFirstDim));
    }

    /// DiffKV-style: `Payload` trailing axis covers `head_size + head_size_v`.
    /// `(num_blocks, page, nh, payload)`.
    #[test]
    fn diffkv_payload_resolves() {
        let n_blocks = 1024;
        let page = 16;
        let nh = 8;
        let payload = 192; // head_size + head_size_v
        let dtype = 2;
        let dim_layout = KvDimLayout::new(
            vec![KvDim::Block, KvDim::Page, KvDim::HeadCount, KvDim::Payload],
            vec![n_blocks, page, nh, payload],
        )
        .unwrap();
        let tensors = layers(vec![n_blocks, page, nh, payload], 32, dtype);

        let (cfg, block_dim) = determine_kv_layout(n_blocks, dtype, &tensors, &dim_layout).unwrap();
        assert_eq!(cfg.outer_dim, 1); // no Outer axis
        assert_eq!(cfg.inner_dim, nh * payload);
        assert_eq!(cfg.num_heads, Some(nh));
        assert!(matches!(block_dim, BlockDimension::BlockIsFirstDim));
    }

    /// Tensor shape mismatch surfaces a clear error naming the axis label.
    #[test]
    fn rejects_shape_mismatch_with_axis_label() {
        let dim_layout = nhd_per_layer_layout(1024, 16, 8, 128);
        // Tensor reports 2048 blocks, layout claims 1024.
        let tensors = layers(vec![2, 2048, 16, 8, 128], 1, 2);
        let err = determine_kv_layout(1024, 2, &tensors, &dim_layout)
            .unwrap_err()
            .to_string();
        assert!(err.contains("Block"), "got: {err}");
        assert!(err.contains("1024"), "got: {err}");
        assert!(err.contains("2048"), "got: {err}");
    }

    /// Block size disagreement between `num_device_blocks` and the labeled
    /// `KvDim::Block` size is a hard error — they must agree.
    #[test]
    fn rejects_block_size_disagreement_between_num_blocks_and_layout() {
        let dim_layout = nhd_per_layer_layout(1024, 16, 8, 128);
        let tensors = layers(vec![2, 1024, 16, 8, 128], 1, 2);
        // Caller asserts 999 blocks but layout (and tensor) say 1024.
        let err = determine_kv_layout(999, 2, &tensors, &dim_layout)
            .unwrap_err()
            .to_string();
        assert!(err.contains("Block axis size"), "got: {err}");
    }

    /// Wrong `dtype_width_bytes` is caught by the total-bytes cross-check.
    #[test]
    fn rejects_dtype_width_mismatch() {
        let dim_layout = nhd_per_layer_layout(1024, 16, 8, 128);
        // Tensor element size is 2 (fp16) but caller passes 4.
        let tensors = layers(vec![2, 1024, 16, 8, 128], 1, 2);
        let err = determine_kv_layout(1024, 4, &tensors, &dim_layout)
            .unwrap_err()
            .to_string();
        assert!(err.contains("total bytes"), "got: {err}");
    }

    /// M1 only supports per-layer registration — a `KvDim::Layer` axis
    /// must be rejected up front so the failure surfaces at
    /// `register_kv_caches` rather than later at `LayerSeparateLayout::
    /// new_with_block_layout` with a confusing `memory.len() != num_layers`
    /// error. M2 will lift this restriction.
    #[test]
    fn rejects_layer_axis_in_milestone_1() {
        let dim_layout = KvDimLayout::new(
            vec![
                KvDim::Block,
                KvDim::Layer,
                KvDim::Outer,
                KvDim::Page,
                KvDim::HeadCount,
                KvDim::HeadSize,
            ],
            vec![1024, 32, 2, 16, 8, 128],
        )
        .unwrap();
        // The tensor list shape doesn't matter for this gate — the layout
        // alone triggers the rejection. Use a plausible cross-layer shape.
        let tensors = layers(vec![1024, 32, 2, 16, 8, 128], 1, 2);
        let err = determine_kv_layout(1024, 2, &tensors, &dim_layout)
            .unwrap_err()
            .to_string();
        assert!(
            err.contains("Layer is not supported in Milestone 1"),
            "got: {err}"
        );
    }

    /// Empty tensor list is rejected up front.
    #[test]
    fn rejects_no_tensors() {
        let dim_layout = nhd_per_layer_layout(1024, 16, 8, 128);
        let err = determine_kv_layout(1024, 2, &[], &dim_layout)
            .unwrap_err()
            .to_string();
        assert!(err.contains("no tensors"), "got: {err}");
    }
}
