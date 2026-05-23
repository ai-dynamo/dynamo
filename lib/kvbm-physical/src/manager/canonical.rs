// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Build a [`CanonicalBlockShape`] from one worker's [`SerializedLayout`].
//!
//! The shape type itself (and its `require_equal` predicate) lives in
//! `kvbm_common::shape`. This module is the thin physical-side adapter that
//! reads the per-worker layout bytes and produces the canonical aggregate
//! used by universal-mode cross-leader compatibility checks. See
//! `kvbm_common::block_layout_mode` for the full semantic write-up.
//!
//! [`SerializedLayout`]: super::metadata::SerializedLayout

use anyhow::{Result, anyhow, bail};
use kvbm_common::{CanonicalBlockShape, KvBlockLayout, KvDim};

use super::metadata::{SerializedLayout, select_transfer_canonical_layout};
use crate::layout::LayoutTypeDetails;

/// Build the canonical descriptor from one worker's [`SerializedLayout`].
///
/// Any one worker is sufficient because `global_extents` is leader-wide
/// (every worker in a leader carries the same pre-shard extents). The
/// function picks the transfer-canonical layout via
/// [`select_transfer_canonical_layout`] (G2 → G3 → first) — all of a
/// worker's layouts share the same shape parameters, but the
/// `KvBlockLayout` guard is more accurate against the tier that peers
/// will actually transfer over (G2 normally, G3 in bypass-host mode
/// where G2 is skipped).
///
/// [`select_transfer_canonical_layout`]: super::metadata::select_transfer_canonical_layout
///
/// # Errors
///
/// - Worker carries no `ParallelismDescriptor` (legacy bytes or a
///   single-worker leader that never stamped one). Universal mode needs
///   the canonical extents and cannot fall back to per-worker
///   `num_heads` / `num_layers` because those are post-shard.
/// - Worker exports no layouts.
/// - Worker's layout is [`KvBlockLayout::Unknown`] — universal mode
///   requires every axis to be labeled.
/// - `global_extents` is missing one of the canonical axes
///   ([`KvDim::Layer`], [`KvDim::HeadCount`]).
/// - `inner_dim` is not divisible by `num_heads`.
pub fn canonical_shape_from_worker(layout: &SerializedLayout) -> Result<CanonicalBlockShape> {
    let unpacked = layout.unpack()?;

    let parallelism = unpacked.parallelism.as_ref().ok_or_else(|| {
        anyhow!(
            "canonical_shape_from_worker: worker {} carries no \
             ParallelismDescriptor (legacy bytes or unstamped leader). \
             Universal-mode compatibility requires global_extents.",
            unpacked.worker_address.worker_id,
        )
    })?;

    // c3: pick the transfer-canonical layout (G2 if present, else first).
    // The shape parameters in `layout_config` (num_layers, num_heads, etc.)
    // are identical across tiers, but the KvBlockLayout guard is more
    // accurate against the transfer-canonical tier (G2 in Universal mode).
    let canonical_layout =
        select_transfer_canonical_layout(&unpacked.layouts).ok_or_else(|| {
            anyhow!(
                "canonical_shape_from_worker: worker {} exported no layouts",
                unpacked.worker_address.worker_id,
            )
        })?;

    let kv_block_layout = match &canonical_layout.layout.layout_type_details {
        LayoutTypeDetails::FullyContiguous(d) => d.kv_block_layout,
        LayoutTypeDetails::LayerSeparate(d) => d.kv_block_layout,
    };
    if matches!(kv_block_layout, KvBlockLayout::Unknown) {
        bail!(
            "canonical_shape_from_worker: worker {} has \
             KvBlockLayout::Unknown; universal mode requires every axis \
             to be labeled",
            unpacked.worker_address.worker_id,
        );
    }

    let cfg = &canonical_layout.layout.layout_config;

    let num_layers_total =
        extent_of(&parallelism.global_extents, KvDim::Layer).ok_or_else(|| {
            anyhow!(
                "canonical_shape_from_worker: worker {} global_extents \
                 missing KvDim::Layer",
                unpacked.worker_address.worker_id,
            )
        })?;
    let num_heads_total =
        extent_of(&parallelism.global_extents, KvDim::HeadCount).ok_or_else(|| {
            anyhow!(
                "canonical_shape_from_worker: worker {} global_extents \
                 missing KvDim::HeadCount",
                unpacked.worker_address.worker_id,
            )
        })?;

    let per_worker_num_heads = cfg.num_heads.ok_or_else(|| {
        anyhow!(
            "canonical_shape_from_worker: worker {} LayoutConfig.num_heads \
             is unset; universal mode requires num_heads to derive head_dim",
            unpacked.worker_address.worker_id,
        )
    })?;
    if per_worker_num_heads == 0 || !cfg.inner_dim.is_multiple_of(per_worker_num_heads) {
        bail!(
            "canonical_shape_from_worker: worker {} inner_dim ({}) \
             not divisible by num_heads ({})",
            unpacked.worker_address.worker_id,
            cfg.inner_dim,
            per_worker_num_heads,
        );
    }
    let head_dim = cfg.inner_dim / per_worker_num_heads;

    Ok(CanonicalBlockShape {
        num_layers_total,
        outer_dim: cfg.outer_dim,
        page_size: cfg.page_size,
        num_heads_total,
        head_dim,
        dtype_width_bytes: cfg.dtype_width_bytes,
    })
}

fn extent_of(extents: &[(KvDim, usize)], axis: KvDim) -> Option<usize> {
    extents
        .iter()
        .find_map(|(d, n)| if *d == axis { Some(*n) } else { None })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn extent_of_finds_axis() {
        let v = vec![(KvDim::Layer, 32), (KvDim::HeadCount, 64)];
        assert_eq!(extent_of(&v, KvDim::Layer), Some(32));
        assert_eq!(extent_of(&v, KvDim::HeadCount), Some(64));
        assert_eq!(extent_of(&v, KvDim::Page), None);
    }
}
