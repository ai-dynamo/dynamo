// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Per-axis labels for KV-cache tensors as registered with the connector.
//!
//! [`KvDim`] names a single tensor axis (`Block`, `Layer`, `Outer`,
//! `Page`, `HeadCount`, `HeadSize`, `Payload`). [`KvDimLayout`] is the
//! ordered list of those labels paired with their concrete sizes — the
//! single source of truth for what each dimension of a registered
//! `kv_cache` tensor means.
//!
//! The labels are populated by Python (in
//! `lib/bindings/kvbm/python/kvbm/v2/vllm/dim_probe.py`) by probing the
//! per-layer `AttentionBackend.get_kv_cache_shape(...)` with sentinel
//! values, and consumed by Rust to build a `LayoutConfig` deterministically
//! — replacing the previous shape-inference heuristic in
//! `kvbm-connector::vllm::layout::determine_kv_layout`.

use anyhow::{Result, anyhow, bail};
use serde::{Deserialize, Serialize};

/// Symbolic label for a single tensor axis.
///
/// `Payload` covers backends whose trailing axis is not pure `head_size`
/// (DiffKV `head_size + head_size_v`, TurboQuant `slot_size_aligned`,
/// FP8 DS-MLA `656`). It may only appear in trailing position.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum KvDim {
    /// Number of paged blocks (one tensor axis equals `num_gpu_blocks`).
    Block,
    /// Number of layers — present only for cross-layer (uniform) caches.
    Layer,
    /// K/V split: 1 (MLA / packed) or 2 (separate K and V).
    Outer,
    /// Tokens per block (page / `block_size`).
    Page,
    /// Number of attention heads on this rank (`num_kv_heads`).
    HeadCount,
    /// Per-head channel count (`head_size`, or
    /// `kv_lora_rank + qk_rope_head_dim` for MLA).
    HeadSize,
    /// Opaque trailing-axis payload (DiffKV, TurboQuant, FP8 DS-MLA).
    /// Only legal in trailing position.
    Payload,
}

impl std::fmt::Display for KvDim {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let s = match self {
            Self::Block => "Block",
            Self::Layer => "Layer",
            Self::Outer => "Outer",
            Self::Page => "Page",
            Self::HeadCount => "HeadCount",
            Self::HeadSize => "HeadSize",
            Self::Payload => "Payload",
        };
        f.write_str(s)
    }
}

/// Labeled, sized layout of a KV cache tensor.
///
/// Invariants (enforced by [`KvDimLayout::new`]):
/// - `dims.len() == sizes.len() > 0`
/// - exactly one `KvDim::Block`
/// - at most one `KvDim::Layer`
/// - at most one of `{KvDim::HeadSize, KvDim::Payload}`
/// - if present, `KvDim::Payload` is the trailing element of `dims`
/// - if present, `KvDim::Outer` has size `1` or `2`
/// - all sizes are `> 0`
///
/// Two non-`Layer`, non-`Outer` dimensions of the same kind are not
/// permitted (a tensor cannot have two `HeadCount` axes, etc.).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct KvDimLayout {
    dims: Vec<KvDim>,
    sizes: Vec<usize>,
}

impl KvDimLayout {
    /// Construct a labeled layout, validating the invariants above.
    pub fn new(dims: Vec<KvDim>, sizes: Vec<usize>) -> Result<Self> {
        if dims.is_empty() {
            bail!("KvDimLayout: dims must be non-empty");
        }
        if dims.len() != sizes.len() {
            bail!(
                "KvDimLayout: dims.len() ({}) != sizes.len() ({})",
                dims.len(),
                sizes.len()
            );
        }
        for (i, &s) in sizes.iter().enumerate() {
            if s == 0 {
                bail!(
                    "KvDimLayout: size at axis {} (label {}) is zero",
                    i,
                    dims[i]
                );
            }
        }

        // Count occurrences of each label that should appear at most once.
        let mut block_count = 0;
        let mut layer_count = 0;
        let mut outer_count = 0;
        let mut page_count = 0;
        let mut head_count = 0;
        let mut head_size_count = 0;
        let mut payload_count = 0;
        for d in &dims {
            match d {
                KvDim::Block => block_count += 1,
                KvDim::Layer => layer_count += 1,
                KvDim::Outer => outer_count += 1,
                KvDim::Page => page_count += 1,
                KvDim::HeadCount => head_count += 1,
                KvDim::HeadSize => head_size_count += 1,
                KvDim::Payload => payload_count += 1,
            }
        }
        if block_count != 1 {
            bail!("KvDimLayout: exactly one Block axis required, got {block_count}");
        }
        if layer_count > 1 {
            bail!("KvDimLayout: at most one Layer axis, got {layer_count}");
        }
        if outer_count > 1 {
            bail!("KvDimLayout: at most one Outer axis, got {outer_count}");
        }
        if page_count > 1 {
            bail!("KvDimLayout: at most one Page axis, got {page_count}");
        }
        if head_count > 1 {
            bail!("KvDimLayout: at most one HeadCount axis, got {head_count}");
        }
        if head_size_count + payload_count > 1 {
            bail!(
                "KvDimLayout: at most one of HeadSize/Payload, got HeadSize={head_size_count}, Payload={payload_count}"
            );
        }
        if payload_count == 1 && *dims.last().unwrap() != KvDim::Payload {
            bail!("KvDimLayout: Payload must be the trailing axis");
        }

        // Outer size must be 1 or 2 if present.
        if let Some(idx) = dims.iter().position(|d| *d == KvDim::Outer) {
            if !(1..=2).contains(&sizes[idx]) {
                bail!(
                    "KvDimLayout: Outer axis size must be 1 or 2, got {}",
                    sizes[idx]
                );
            }
        }

        Ok(Self { dims, sizes })
    }

    /// Ordered axis labels.
    pub fn dims(&self) -> &[KvDim] {
        &self.dims
    }

    /// Ordered axis sizes (matches `dims` index-for-index).
    pub fn sizes(&self) -> &[usize] {
        &self.sizes
    }

    /// Position of the first axis carrying `dim`, if any.
    pub fn position(&self, dim: KvDim) -> Option<usize> {
        self.dims.iter().position(|d| *d == dim)
    }

    /// Size of the first axis carrying `dim`, if any.
    pub fn size_of(&self, dim: KvDim) -> Option<usize> {
        self.position(dim).map(|i| self.sizes[i])
    }

    /// Product of all dimension sizes — the total element count of the
    /// tensor (compare to `tensor.numel()`).
    pub fn total_elements(&self) -> usize {
        self.sizes.iter().product()
    }

    /// Position of the `Block` axis.
    ///
    /// Errors if no `Block` axis is present (the `new` constructor
    /// rejects this, so this only ever returns `Ok(_)` on a valid layout —
    /// the result type is for symmetry with other accessors).
    pub fn block_axis(&self) -> Result<usize> {
        self.position(KvDim::Block)
            .ok_or_else(|| anyhow!("KvDimLayout: missing Block axis"))
    }

    /// Size of the `Outer` axis, defaulting to `1` when absent.
    ///
    /// MLA / packed K-V layouts omit the K/V split axis entirely; treating
    /// the absent case as `1` lets `LayoutConfig.outer_dim` be derived
    /// uniformly.
    pub fn outer_size(&self) -> usize {
        self.size_of(KvDim::Outer).unwrap_or(1)
    }

    /// Size of the `Page` axis (tokens per block). Required.
    pub fn page_size(&self) -> Result<usize> {
        self.size_of(KvDim::Page)
            .ok_or_else(|| anyhow!("KvDimLayout: missing Page axis"))
    }

    /// Size of the `HeadCount` axis (`num_kv_heads`), if present.
    pub fn head_count(&self) -> Option<usize> {
        self.size_of(KvDim::HeadCount)
    }

    /// Size of the `HeadSize` axis, if present.
    ///
    /// `None` for backends with a `Payload` trailing axis.
    pub fn head_size(&self) -> Option<usize> {
        self.size_of(KvDim::HeadSize)
    }

    /// Size of the `Payload` axis, if present.
    pub fn payload_size(&self) -> Option<usize> {
        self.size_of(KvDim::Payload)
    }

    /// Per-token bytes-per-block-token (i.e. the `inner_dim` factor in
    /// `LayoutConfig`'s `bytes_per_block`):
    /// `HeadCount? * (HeadSize | Payload)` × `dtype_bytes`.
    ///
    /// For MLA (`HeadCount` absent, `HeadSize` present):
    /// `head_size * dtype_bytes`.
    ///
    /// For DiffKV / TurboQuant (`HeadCount` + `Payload`):
    /// `head_count * payload * dtype_bytes`.
    ///
    /// Returns `None` if neither `HeadSize` nor `Payload` is present.
    pub fn inner_bytes(&self, dtype_bytes: usize) -> Option<usize> {
        let trailing = self.head_size().or_else(|| self.payload_size())?;
        let head_count = self.head_count().unwrap_or(1);
        Some(head_count * trailing * dtype_bytes)
    }

    /// Per-token element count (no dtype), suitable for direct use as
    /// `LayoutConfig.inner_dim`.
    pub fn inner_elements(&self) -> Option<usize> {
        self.inner_bytes(1)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Llama-3-8B per-layer FlashAttn NHD shape:
    /// `(2, num_blocks, page, num_kv_heads, head_size)`.
    #[test]
    fn flashattn_nhd_per_layer() {
        let l = KvDimLayout::new(
            vec![
                KvDim::Outer,
                KvDim::Block,
                KvDim::Page,
                KvDim::HeadCount,
                KvDim::HeadSize,
            ],
            vec![2, 1024, 16, 8, 128],
        )
        .unwrap();
        assert_eq!(l.block_axis().unwrap(), 1);
        assert_eq!(l.outer_size(), 2);
        assert_eq!(l.page_size().unwrap(), 16);
        assert_eq!(l.head_count(), Some(8));
        assert_eq!(l.head_size(), Some(128));
        assert_eq!(l.payload_size(), None);
        assert_eq!(l.total_elements(), 2 * 1024 * 16 * 8 * 128);
        assert_eq!(l.inner_elements(), Some(8 * 128)); // num_kv_heads * head_size
    }

    /// FlashInfer NHD: `(num_blocks, 2, page, num_kv_heads, head_size)`.
    #[test]
    fn flashinfer_nhd_per_layer() {
        let l = KvDimLayout::new(
            vec![
                KvDim::Block,
                KvDim::Outer,
                KvDim::Page,
                KvDim::HeadCount,
                KvDim::HeadSize,
            ],
            vec![1024, 2, 16, 8, 128],
        )
        .unwrap();
        assert_eq!(l.block_axis().unwrap(), 0);
        assert_eq!(l.outer_size(), 2);
    }

    /// DeepSeek-V2-Lite MLA: `(num_blocks, page, head_size)`. No `Outer`,
    /// no `HeadCount`. `outer_size` defaults to 1.
    #[test]
    fn mla_per_layer() {
        let l = KvDimLayout::new(
            vec![KvDim::Block, KvDim::Page, KvDim::HeadSize],
            vec![26847, 16, 576],
        )
        .unwrap();
        assert_eq!(l.outer_size(), 1);
        assert_eq!(l.head_count(), None);
        assert_eq!(l.head_size(), Some(576));
        assert_eq!(l.inner_elements(), Some(576));
    }

    /// DiffKV: `(num_blocks, page, num_kv_heads, head_size + head_size_v)`
    /// — the trailing axis is `Payload`, not `HeadSize`.
    #[test]
    fn diffkv_payload_trailing() {
        let l = KvDimLayout::new(
            vec![KvDim::Block, KvDim::Page, KvDim::HeadCount, KvDim::Payload],
            vec![1024, 16, 8, 192],
        )
        .unwrap();
        assert_eq!(l.head_size(), None);
        assert_eq!(l.payload_size(), Some(192));
        assert_eq!(l.inner_elements(), Some(8 * 192));
    }

    /// Cross-layer FlashAttn NHD:
    /// `(num_blocks, num_layers, 2, page, num_kv_heads, head_size)`.
    #[test]
    fn flashattn_nhd_cross_layer() {
        let l = KvDimLayout::new(
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
        assert_eq!(l.block_axis().unwrap(), 0);
        assert_eq!(l.size_of(KvDim::Layer), Some(32));
    }

    #[test]
    fn rejects_empty() {
        assert!(KvDimLayout::new(vec![], vec![]).is_err());
    }

    #[test]
    fn rejects_dim_size_length_mismatch() {
        assert!(KvDimLayout::new(vec![KvDim::Block], vec![1, 2]).is_err());
    }

    #[test]
    fn rejects_zero_size() {
        let err = KvDimLayout::new(vec![KvDim::Block], vec![0])
            .unwrap_err()
            .to_string();
        assert!(err.contains("zero"), "got: {err}");
    }

    #[test]
    fn rejects_missing_block() {
        let err = KvDimLayout::new(vec![KvDim::Page], vec![16])
            .unwrap_err()
            .to_string();
        assert!(err.contains("Block"), "got: {err}");
    }

    #[test]
    fn rejects_two_block_axes() {
        let err = KvDimLayout::new(vec![KvDim::Block, KvDim::Block], vec![16, 32])
            .unwrap_err()
            .to_string();
        assert!(err.contains("exactly one Block"), "got: {err}");
    }

    #[test]
    fn rejects_outer_size_three() {
        let err = KvDimLayout::new(
            vec![KvDim::Outer, KvDim::Block, KvDim::Page, KvDim::HeadSize],
            vec![3, 16, 8, 128],
        )
        .unwrap_err()
        .to_string();
        assert!(err.contains("Outer"), "got: {err}");
    }

    #[test]
    fn rejects_payload_not_trailing() {
        let err = KvDimLayout::new(
            vec![KvDim::Block, KvDim::Payload, KvDim::HeadCount],
            vec![16, 192, 8],
        )
        .unwrap_err()
        .to_string();
        assert!(err.contains("Payload"), "got: {err}");
    }

    #[test]
    fn rejects_both_head_size_and_payload() {
        let err = KvDimLayout::new(
            vec![KvDim::Block, KvDim::HeadSize, KvDim::Payload],
            vec![16, 128, 192],
        )
        .unwrap_err()
        .to_string();
        assert!(err.contains("HeadSize"), "got: {err}");
    }

    #[test]
    fn serde_roundtrip() {
        let l = KvDimLayout::new(
            vec![KvDim::Block, KvDim::Page, KvDim::HeadSize],
            vec![1024, 16, 576],
        )
        .unwrap();
        let json = serde_json::to_string(&l).unwrap();
        let de: KvDimLayout = serde_json::from_str(&json).unwrap();
        assert_eq!(l, de);
    }
}
