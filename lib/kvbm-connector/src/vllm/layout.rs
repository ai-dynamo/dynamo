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
use kvbm_common::{KvBlockLayout, KvDim, KvDimLayout, KvDimStrides};
use kvbm_physical::layout::{BlockDimension, LayoutConfig};

/// Detect the permutation between a tensor's *logical* axis order
/// (what `dim_probe.py` labels) and its *physical* memory order
/// (revealed by `tensor.stride()`), and return the labelled layout
/// re-expressed in physical order.
///
/// vLLM's HND backends ship as `tensor.permute(inv_order)` views over
/// physically contiguous memory: the strides are non-row-major when
/// indexed by labelled position, but row-major when indexed by
/// physical (stride-descending) position. Sorting axes by descending
/// element-stride recovers the physical order; reordering the
/// `(label, size)` pairs by that permutation yields a `KvDimLayout`
/// whose schema matches the bytes in memory.
///
/// Errors only when the strides have **internal gaps** in physical
/// order (stride padding, strided slices) — i.e. they're not just a
/// permutation of a contiguous tensor. Plain HND views relabel
/// successfully and downstream sees `[Outer, Block, HeadCount, Page,
/// HeadSize]` instead of the post-permute logical view.
///
/// Rank check runs **before** any helper that walks the stride
/// vector — a malformed `TensorDescriptor` whose `stride()` and
/// `shape()` disagree on length surfaces here as a clear error
/// rather than a panic deeper in.
fn relabel_to_physical_order(
    tensor_idx: usize,
    tensor: &Arc<dyn TensorDescriptor>,
    logical_dim_layout: &KvDimLayout,
) -> Result<KvDimLayout> {
    let actual = tensor.stride();
    let elem_size = tensor.element_size();
    let n = logical_dim_layout.dims().len();

    if actual.len() != n {
        bail!(
            "tensor {tensor_idx}: stride rank {} does not match labelled rank {n}",
            actual.len(),
        );
    }

    // Sort axes by element stride descending. Ties (e.g. size-1 axes)
    // break by ascending logical position so the permutation is
    // deterministic.
    let mut perm: Vec<usize> = (0..n).collect();
    perm.sort_by(|&a, &b| actual[b].cmp(&actual[a]).then(a.cmp(&b)));

    let phys_dims: Vec<KvDim> = perm.iter().map(|&i| logical_dim_layout.dims()[i]).collect();
    let phys_sizes: Vec<usize> = perm.iter().map(|&i| logical_dim_layout.sizes()[i]).collect();
    let phys_layout = KvDimLayout::new(phys_dims, phys_sizes)?;

    let phys_byte_strides: Vec<usize> = perm.iter().map(|&i| actual[i] * elem_size).collect();
    let phys_strides = KvDimStrides::from_byte_strides(phys_byte_strides, elem_size)?;
    if !phys_strides.is_fully_contiguous(&phys_layout)? {
        let expected = KvDimStrides::contiguous_for(&phys_layout, elem_size);
        bail!(
            "tensor {tensor_idx}: strides have internal gaps in physical order \
             [{phys}]: actual byte strides {actual_bytes:?}, expected (row-major) \
             {expected_bytes:?}. This is not a simple permutation — it has stride \
             padding or strided slicing, which M1 does not support.",
            phys = phys_layout
                .dims()
                .iter()
                .map(|d| d.to_string())
                .collect::<Vec<_>>()
                .join(","),
            actual_bytes = phys_strides.as_bytes(),
            expected_bytes = expected.as_bytes(),
        );
    }
    Ok(phys_layout)
}

/// Derive `KvBlockLayout` from a *physical* `KvDimLayout` by inspecting
/// the relative position of `Page` and `HeadCount` (HND has HeadCount
/// before Page; NHD has Page before HeadCount). Returns `Unknown` for
/// MLA/Payload layouts that lack one of those axes — those models
/// don't have an NHD/HND distinction to make.
fn derive_block_layout_from_physical(phys_layout: &KvDimLayout) -> KvBlockLayout {
    match (
        phys_layout.position(KvDim::Page),
        phys_layout.position(KvDim::HeadCount),
    ) {
        (Some(p), Some(h)) if p < h => KvBlockLayout::OperationalNHD,
        (Some(p), Some(h)) if p > h => KvBlockLayout::OperationalHND,
        _ => KvBlockLayout::Unknown,
    }
}

/// Determine the KV cache layout configuration and block dimension from
/// a caller-supplied (logical) [`KvDimLayout`] plus the actual tensor
/// strides.
///
/// The logical `dim_layout` matches what `dim_probe.py` extracts from
/// `backend.get_kv_cache_shape()` — i.e. the post-permute view that
/// PyTorch hands across the FFI. For HND backends, that view's strides
/// don't match the labelled order. This function calls
/// [`relabel_to_physical_order`] on tensor 0 to produce the physical
/// labelled layout, asserts that tensors 1..N share tensor 0's strides
/// and element size, and then derives `LayoutConfig` and `BlockDimension`
/// from the **physical** layout — so a downstream `LayerSeparateLayout`
/// sees the correct in-tensor axis ordering for HND as well as NHD.
///
/// # Arguments
/// * `num_device_blocks` - Expected number of device blocks (from vLLM's
///   per-rank `kv_cache_config`). Cross-checked against the layout's
///   `Block` axis size — they must agree.
/// * `dtype_width_bytes` - KV-cache element width in bytes (e.g. 2 for fp16).
/// * `kv_tensors` - KV cache tensors (one per layer in M1 per-layer mode).
/// * `dim_layout` - Per-axis logical labels and sizes from `dim_probe.py`.
/// * `declared_block_layout` - The `KvBlockLayout` Python derived from
///   `backend.get_kv_cache_stride_order(False)`. Cross-checked against
///   what the relabeler infers from the physical layout; on disagreement
///   we log a warning and prefer the relabeler's view (it sees the
///   actual bytes).
///
/// # Returns
/// `(LayoutConfig, BlockDimension)` derived from the physical layout.
pub fn determine_kv_layout(
    num_device_blocks: usize,
    dtype_width_bytes: usize,
    kv_tensors: &[Arc<dyn TensorDescriptor>],
    dim_layout: &KvDimLayout,
    declared_block_layout: KvBlockLayout,
) -> Result<(LayoutConfig, BlockDimension)> {
    if kv_tensors.is_empty() {
        bail!("determine_kv_layout: no tensors provided");
    }

    // Cross-check: every tensor's shape must equal the labelled sizes
    // (logical order — that's how Python sends them).
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

    // Relabel tensor 0 to physical order. Tensors 1..N must share its
    // strides and element size — vLLM allocates all per-layer tensors
    // with the same backend, so divergence here is a bug we want to
    // surface at register time rather than later.
    let phys_layout = relabel_to_physical_order(0, &kv_tensors[0], dim_layout)?;
    let ref_strides = kv_tensors[0].stride();
    let ref_elem = kv_tensors[0].element_size();
    for (i, tensor) in kv_tensors.iter().enumerate().skip(1) {
        if tensor.stride() != ref_strides {
            bail!(
                "tensor {i}: stride {:?} disagrees with tensor 0 stride {:?}; \
                 per-layer-divergent strides are not supported",
                tensor.stride(),
                ref_strides,
            );
        }
        if tensor.element_size() != ref_elem {
            bail!(
                "tensor {i}: element_size {} disagrees with tensor 0 element_size {}",
                tensor.element_size(),
                ref_elem,
            );
        }
    }

    // Cross-check the relabeler's per-block ordering against the
    // `KvBlockLayout` Python derived from
    // `backend.get_kv_cache_stride_order(False)`. Disagreement is a
    // Python-side bug, not a tensor problem — warn and continue with
    // the relabeler's view (it inspected the actual bytes).
    let derived_block_layout = derive_block_layout_from_physical(&phys_layout);
    if declared_block_layout != KvBlockLayout::Unknown
        && derived_block_layout != KvBlockLayout::Unknown
        && declared_block_layout != derived_block_layout
    {
        tracing::warn!(
            declared = %declared_block_layout,
            derived = %derived_block_layout,
            phys_dims = ?phys_layout.dims(),
            "Python-declared KvBlockLayout disagrees with stride-derived physical \
             ordering; trusting strides",
        );
    }

    let permutation_is_identity = phys_layout.dims() == dim_layout.dims();
    tracing::info!(
        n_tensors = kv_tensors.len(),
        logical_dims = ?dim_layout.dims(),
        physical_dims = ?phys_layout.dims(),
        identity = permutation_is_identity,
        derived_block_layout = %derived_block_layout,
        declared_block_layout = %declared_block_layout,
        "Relabelled tensors to physical layout",
    );

    // Cross-check: Block size must equal num_device_blocks. Position-
    // agnostic; physical and logical agree on individual axis sizes.
    let labeled_blocks = phys_layout.size_of(KvDim::Block).ok_or_else(|| {
        anyhow::anyhow!("determine_kv_layout: KvDimLayout is missing a Block axis")
    })?;
    if labeled_blocks != num_device_blocks {
        bail!("Block axis size ({labeled_blocks}) != num_device_blocks ({num_device_blocks})",);
    }

    // Milestone 1: per-layer registration only. Reject any `KvDim::Layer`
    // axis. This applies to the *physical* layout — if Layer is present
    // in physical order, the registration is cross-layer and goes through
    // a different (M2) build path.
    if phys_layout.position(KvDim::Layer).is_some() {
        bail!(
            "KvDim::Layer is not supported in Milestone 1 (per-layer registration only); \
             cross-layer / fully-contiguous registration lands in Milestone 2",
        );
    }

    // Locate the Block axis in physical order. Per-layer tensors only
    // support positions 0 or 1 in the layer-separate physical layout.
    let block_pos = phys_layout.block_axis()?;
    let block_dim = match block_pos {
        0 => BlockDimension::BlockIsFirstDim,
        1 => BlockDimension::BlockIsSecondDim,
        n => bail!(
            "Block axis at position {n} (physical) is not supported (per-layer registration expects 0 or 1)",
        ),
    };

    // Derive LayoutConfig from the physical layout. Sizes are
    // position-agnostic, so this matches what the prior implementation
    // computed from logical for NHD; for HND, positions of HeadCount
    // and Page differ but their *sizes* don't, and `inner_elements()`
    // collapses both into the flat product.
    let outer_dim = phys_layout.outer_size();
    let page_size = phys_layout.page_size()?;
    let inner_dim = phys_layout
        .inner_elements()
        .ok_or_else(|| anyhow::anyhow!("KvDimLayout has neither HeadSize nor Payload axis"))?;
    let num_layers = kv_tensors.len();

    let mut builder = LayoutConfig::builder();
    builder.num_blocks(num_device_blocks);
    builder.num_layers(num_layers);
    builder.outer_dim(outer_dim);
    builder.page_size(page_size);
    builder.inner_dim(inner_dim);
    builder.dtype_width_bytes(dtype_width_bytes);
    if let Some(nh) = phys_layout.head_count() {
        builder.num_heads(Some(nh));
    }
    let layout_config = builder.build()?;

    // Per-tensor total-bytes cross-check.
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
        physical_dims = ?phys_layout.dims(),
        "Resolved KV layout from physical (relabelled) dim layout"
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
            // Row-major strides (the strict-contiguity validator requires them).
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

        /// Build a tensor with caller-supplied element strides — used by
        /// the strict-contiguity validator tests to exercise non-row-major
        /// layouts (e.g. FlashAttention HND's `(0, 1, 3, 2, 4)` permutation).
        fn arc_strided(
            shape: Vec<usize>,
            stride: Vec<usize>,
            element_size: usize,
        ) -> Arc<dyn TensorDescriptor> {
            assert_eq!(shape.len(), stride.len(), "shape/stride rank mismatch");
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

        let (cfg, block_dim) = determine_kv_layout(n_blocks, dtype, &tensors, &dim_layout, KvBlockLayout::Unknown).unwrap();

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

        let (cfg, block_dim) = determine_kv_layout(n_blocks, dtype, &tensors, &dim_layout, KvBlockLayout::Unknown).unwrap();

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
            determine_kv_layout(n_blocks, dtype, &tensors, &dim_layout, KvBlockLayout::Unknown).unwrap();
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

        let (cfg, block_dim) = determine_kv_layout(n_blocks, dtype, &tensors, &dim_layout, KvBlockLayout::Unknown).unwrap();

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

        let (cfg, block_dim) = determine_kv_layout(n_blocks, dtype, &tensors, &dim_layout, KvBlockLayout::Unknown).unwrap();
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
        let err = determine_kv_layout(1024, 2, &tensors, &dim_layout, KvBlockLayout::Unknown)
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
        let err = determine_kv_layout(999, 2, &tensors, &dim_layout, KvBlockLayout::Unknown)
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
        let err = determine_kv_layout(1024, 4, &tensors, &dim_layout, KvBlockLayout::Unknown)
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
        let err = determine_kv_layout(1024, 2, &tensors, &dim_layout, KvBlockLayout::Unknown)
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
        let err = determine_kv_layout(1024, 2, &[], &dim_layout, KvBlockLayout::Unknown)
            .unwrap_err()
            .to_string();
        assert!(err.contains("no tensors"), "got: {err}");
    }

    /// FlashAttention HND case: the *logical* labelled shape is
    /// `[Outer, Block, Page, HeadCount, HeadSize]` (= NHD ordering, the
    /// post-permute view PyTorch hands across the FFI) but the physical
    /// element strides correspond to memory laid out as
    /// `[Outer=2, Block=1024, HeadCount=8, Page=16, HeadSize=128]`.
    ///
    /// Element strides re-indexed to the *logical* axis order
    /// `[Outer, Block, Page, HeadCount, HeadSize]`:
    ///     Outer=16777216, Block=16384, Page=128, HeadCount=2048, HeadSize=1
    ///
    /// The relabeler sorts these descending → physical permutation
    /// `[0, 1, 3, 2, 4]` → physical labels
    /// `[Outer, Block, HeadCount, Page, HeadSize]`. Since the strides
    /// are row-major-with-no-gaps in that physical order,
    /// `relabel_to_physical_order` returns Ok and downstream sees the
    /// HND layout in its true memory ordering. The per-layer
    /// `LayerSeparateLayout` then computes correct addresses with
    /// `BlockIsSecondDim` (Block sits at physical position 1).
    #[test]
    fn relabels_hnd_permuted_strides_to_physical_order() {
        let n_blocks = 1024;
        let page = 16;
        let nh = 8;
        let hd = 128;
        let dtype = 2;
        let dim_layout = nhd_per_layer_layout(n_blocks, page, nh, hd);
        let shape = vec![2, n_blocks, page, nh, hd];
        // HND-permuted element strides re-indexed to logical-axis order.
        let stride = vec![16777216, 16384, 128, 2048, 1];
        let tensors: Vec<Arc<dyn TensorDescriptor>> = (0..2)
            .map(|_| TestTensor::arc_strided(shape.clone(), stride.clone(), dtype))
            .collect();
        let (cfg, block_dim) = determine_kv_layout(
            n_blocks,
            dtype,
            &tensors,
            &dim_layout,
            KvBlockLayout::OperationalHND,
        )
        .expect("HND-permuted tensors must relabel cleanly");
        // Sizes are position-agnostic — same as the NHD case.
        assert_eq!(cfg.outer_dim, 2);
        assert_eq!(cfg.page_size, page);
        assert_eq!(cfg.inner_dim, nh * hd);
        assert_eq!(cfg.num_heads, Some(nh));
        // Physical Block axis is position 1 → BlockIsSecondDim.
        assert!(matches!(block_dim, BlockDimension::BlockIsSecondDim));
    }

    /// Strided/sliced tensors (gaps in physical order) MUST still
    /// error — the relabeler accepts permutations of contiguous memory
    /// only. Construct a tensor whose innermost stride is `2 * elem_size`
    /// (every other element) — the sort produces the identity
    /// permutation (descending strides agree with row-major label order)
    /// but `is_fully_contiguous` fails because the innermost stride
    /// disagrees with `elem_size`.
    #[test]
    fn rejects_strided_with_internal_gaps() {
        let n_blocks = 1024;
        let page = 16;
        let nh = 8;
        let hd = 128;
        let dtype = 2;
        let dim_layout = nhd_per_layer_layout(n_blocks, page, nh, hd);
        let shape = vec![2, n_blocks, page, nh, hd];
        // Row-major element strides for shape [2, 1024, 16, 8, 128] are
        // [1024*16*8*128, 16*8*128, 8*128, 128, 1] = [16777216, 16384, 1024, 128, 1].
        // Doubling every entry preserves descending order (so the
        // permutation is identity = NHD physical) but makes each axis
        // skip every other slot in memory — internal gaps.
        let stride = vec![33554432, 32768, 2048, 256, 2];
        let tensors: Vec<Arc<dyn TensorDescriptor>> = (0..2)
            .map(|_| TestTensor::arc_strided(shape.clone(), stride.clone(), dtype))
            .collect();
        let err = determine_kv_layout(
            n_blocks,
            dtype,
            &tensors,
            &dim_layout,
            KvBlockLayout::OperationalNHD,
        )
        .expect_err("strided tensor with gaps must error")
        .to_string();
        assert!(
            err.contains("internal gaps"),
            "expected 'internal gaps' in err: {err}"
        );
    }

    /// Row-major NHD tensors relabel as identity (no permutation).
    /// Explicit positive-path assertion against regressions in the
    /// relabeler.
    #[test]
    fn relabels_row_major_tensors_to_identity_permutation() {
        let dim_layout = nhd_per_layer_layout(1024, 16, 8, 128);
        let tensors = layers(vec![2, 1024, 16, 8, 128], 32, 2);
        assert!(
            determine_kv_layout(1024, 2, &tensors, &dim_layout, KvBlockLayout::OperationalNHD)
                .is_ok()
        );
    }

    /// Per-layer-divergent strides must surface at registration time —
    /// vLLM's per-layer tensors all use the same backend, so divergent
    /// strides indicate a bug we want to catch loudly rather than
    /// silently miscompute addresses.
    #[test]
    fn rejects_per_layer_divergent_strides() {
        let n_blocks = 1024;
        let page = 16;
        let nh = 8;
        let hd = 128;
        let dtype = 2;
        let dim_layout = nhd_per_layer_layout(n_blocks, page, nh, hd);
        let shape = vec![2, n_blocks, page, nh, hd];
        // Tensor 0: row-major NHD strides.
        let stride_a = vec![16777216, 16384, 1024, 128, 1];
        // Tensor 1: HND-permuted strides — different from tensor 0.
        let stride_b = vec![16777216, 16384, 128, 2048, 1];
        let tensors: Vec<Arc<dyn TensorDescriptor>> = vec![
            TestTensor::arc_strided(shape.clone(), stride_a, dtype),
            TestTensor::arc_strided(shape.clone(), stride_b, dtype),
        ];
        let err = determine_kv_layout(
            n_blocks,
            dtype,
            &tensors,
            &dim_layout,
            KvBlockLayout::OperationalNHD,
        )
        .expect_err("divergent strides must error")
        .to_string();
        assert!(
            err.contains("disagrees with tensor 0 stride"),
            "expected divergence error: {err}"
        );
    }
}
