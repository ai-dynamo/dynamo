// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Coordinate-space slicing of labelled KV layouts.
//!
//! [`AxisSlice`] restricts the addressable coordinate range of a single
//! axis without changing the underlying memory or strides. It is the
//! atomic unit of layout slicing: one TP-resharding `LayoutView` is the
//! base layout plus zero or more `AxisSlice`s, each restricting a
//! distinct axis to a `[start, start + len)` window of the global
//! coordinate space.
//!
//! Today every `LayoutView` carries at most one [`AxisSlice`] per axis
//! (covering TP, PP, and PP×TP composition with contiguous shards).
//! Non-contiguous shardings (DP-uneven, interleaved heads) will eventually
//! need multiple slices on the same axis; the storage type
//! (`Vec<AxisSlice>`) and the intersection algebra defined here are
//! designed to accept that without API breakage.
//!
//! [`LayoutSignature`] is the address-free, hashable description of a
//! sliced layout — it carries everything required to match two layouts
//! across processes (axes, local/global extents, strides, region-axis
//! identity) but no pointers. It is **not** a stable wire format yet:
//! intra-process planner consumers are the only audience for now. When
//! it is promoted to a cross-process control message (PR-7+) it will gain
//! an explicit version field; until then, treat the type as internal and
//! mark with `#[non_exhaustive]` to preserve forward compatibility.

use std::ops::Range;

use anyhow::{Result, bail};
use serde::{Deserialize, Serialize};

use crate::tensor::KvDim;

/// A contiguous coordinate-space restriction on a single axis.
///
/// `start` and `len` are in **global** coordinates — the same coordinate
/// system that the unrestricted base layout uses. Two views over the same
/// global coordinate system can be intersected via [`intersect_axis`] to
/// find the overlap (and translate it into each side's local coords).
///
/// Slicing does not change the underlying byte strides; it is a pure
/// coordinate-space restriction.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[non_exhaustive]
pub struct AxisSlice {
    pub dim: KvDim,
    pub start: usize,
    pub len: usize,
    pub global_len: usize,
}

impl AxisSlice {
    /// Full (unrestricted) slice: `start = 0`, `len = global_len`.
    pub fn full(dim: KvDim, global_len: usize) -> Self {
        Self {
            dim,
            start: 0,
            len: global_len,
            global_len,
        }
    }

    /// Construct a slice covering rank `r` of an even `world`-way shard
    /// over `global_len`. Errors when `world == 0`, `rank >= world`, or
    /// `global_len` is not divisible by `world`.
    pub fn shard(dim: KvDim, rank: usize, world: usize, global_len: usize) -> Result<Self> {
        if world == 0 {
            bail!("AxisSlice::shard: world must be > 0");
        }
        if rank >= world {
            bail!("AxisSlice::shard: rank {rank} out of range for world {world}");
        }
        if !global_len.is_multiple_of(world) {
            bail!(
                "AxisSlice::shard: global_len {global_len} is not divisible by world {world} \
                 (uneven sharding requires multi-slice support, not yet exposed)"
            );
        }
        let len = global_len / world;
        Ok(Self {
            dim,
            start: rank * len,
            len,
            global_len,
        })
    }

    /// Construct a slice with explicit bounds. Validates `start + len <=
    /// global_len` and `len > 0`.
    pub fn new(dim: KvDim, start: usize, len: usize, global_len: usize) -> Result<Self> {
        if len == 0 {
            bail!("AxisSlice::new: len must be > 0 (use AxisSlice::full or omit the slice)");
        }
        if start
            .checked_add(len)
            .map(|e| e > global_len)
            .unwrap_or(true)
        {
            bail!(
                "AxisSlice::new: start ({start}) + len ({len}) exceeds global_len ({global_len})"
            );
        }
        Ok(Self {
            dim,
            start,
            len,
            global_len,
        })
    }

    /// Inclusive lower bound (alias for `start`).
    pub fn begin(&self) -> usize {
        self.start
    }

    /// Exclusive upper bound (`start + len`).
    pub fn end(&self) -> usize {
        self.start + self.len
    }

    /// True iff this slice covers the entire global range.
    pub fn is_full(&self) -> bool {
        self.start == 0 && self.len == self.global_len
    }
}

/// Per-axis local/global extent pair. Used inside [`LayoutSignature`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[non_exhaustive]
pub struct AxisExtent {
    /// Size in local (post-slicing) coordinates — the range a coord may
    /// occupy in this view.
    pub local: usize,
    /// Size in global coordinates — the unrestricted axis size.
    pub global: usize,
    /// Local coordinate `0` corresponds to global coordinate `start`.
    pub start: usize,
}

impl AxisExtent {
    /// Build an extent representing an unrestricted axis.
    pub fn full(size: usize) -> Self {
        Self {
            local: size,
            global: size,
            start: 0,
        }
    }

    /// Build an extent from an [`AxisSlice`].
    pub fn from_slice(slice: &AxisSlice) -> Self {
        Self {
            local: slice.len,
            global: slice.global_len,
            start: slice.start,
        }
    }

    /// True iff this extent represents an unrestricted axis.
    pub fn is_full(&self) -> bool {
        self.local == self.global && self.start == 0
    }
}

/// Address-free, hashable description of a sliced layout.
///
/// Two layouts whose `LayoutSignature`s match describe exactly the same
/// addressable shape (local / global extents, byte strides, region-axis
/// identity). This is the type used by the planner to decide kernel /
/// candidate compatibility and — eventually — by cross-process pull
/// requests to identify a peer's exposed layout.
///
/// This is intentionally **not** a stable wire type yet: when it crosses
/// the process boundary in a future PR, it will gain a version field and
/// a documented schema. Today it is `#[non_exhaustive]` to keep the
/// in-process API additive.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[non_exhaustive]
pub struct LayoutSignature {
    /// Per-axis: (label, local/global extent + slice start). Order matches
    /// the underlying layout's axis order.
    pub axes: Vec<(KvDim, AxisExtent)>,
    /// Per-axis byte strides in the *base* tensor — slicing does not
    /// change strides.
    pub byte_strides: Vec<usize>,
    /// Element size (bytes) of the underlying tensor dtype.
    pub elem_size: usize,
    /// Which axis (if any) is region-partitioned: i.e. an axis whose
    /// coordinate selects between separate base allocations rather than
    /// indexing within one tensor.
    pub region_axis: Option<KvDim>,
}

impl LayoutSignature {
    /// Construct a signature. Required because `#[non_exhaustive]`
    /// blocks struct-literal construction outside this crate.
    pub fn new(
        axes: Vec<(KvDim, AxisExtent)>,
        byte_strides: Vec<usize>,
        elem_size: usize,
        region_axis: Option<KvDim>,
    ) -> Self {
        Self {
            axes,
            byte_strides,
            elem_size,
            region_axis,
        }
    }
}

/// Resulting overlap of two [`AxisSlice`]s on the same axis.
///
/// `src_local` / `dst_local` are coordinate ranges in each side's local
/// (post-slicing) coordinate space. The lengths are equal (`len ==
/// overlap`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AxisIntersection {
    pub dim: KvDim,
    pub src_local: Range<usize>,
    pub dst_local: Range<usize>,
}

impl AxisIntersection {
    /// Length of the overlap (same for src and dst by construction).
    pub fn len(&self) -> usize {
        self.src_local.end - self.src_local.start
    }

    /// True iff the overlap is zero. The constructor [`intersect_axis`]
    /// returns `None` for empty overlaps, so a constructed
    /// `AxisIntersection` is never empty — this method exists for clippy
    /// / `len()`-paired ergonomics.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

/// Intersect two slices on the same axis, returning the overlap in each
/// side's local coordinate space.
///
/// Returns:
/// - `Err` if the slices reference different axes or different
///   `global_len`s — those describe incompatible layouts (different model
///   configs entirely) and cannot be transferred between.
/// - `Ok(None)` if the slices are disjoint.
/// - `Ok(Some(intersection))` otherwise.
pub fn intersect_axis(src: &AxisSlice, dst: &AxisSlice) -> Result<Option<AxisIntersection>> {
    if src.dim != dst.dim {
        bail!(
            "intersect_axis: dim mismatch — src is {:?}, dst is {:?}",
            src.dim,
            dst.dim
        );
    }
    if src.global_len != dst.global_len {
        bail!(
            "intersect_axis ({:?}): global_len mismatch — src has {}, dst has {}; \
             layouts describe different model configs",
            src.dim,
            src.global_len,
            dst.global_len
        );
    }
    let global_start = src.start.max(dst.start);
    let global_end = src.end().min(dst.end());
    if global_end <= global_start {
        return Ok(None);
    }
    let len = global_end - global_start;
    let src_local = (global_start - src.start)..(global_start - src.start + len);
    let dst_local = (global_start - dst.start)..(global_start - dst.start + len);
    Ok(Some(AxisIntersection {
        dim: src.dim,
        src_local,
        dst_local,
    }))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn axis_slice_full_covers_global() {
        let s = AxisSlice::full(KvDim::HeadCount, 32);
        assert!(s.is_full());
        assert_eq!(s.begin(), 0);
        assert_eq!(s.end(), 32);
    }

    #[test]
    fn axis_slice_shard_partitions_evenly() {
        // 4 ranks over global_len = 32 → each rank gets 8.
        for rank in 0..4 {
            let s = AxisSlice::shard(KvDim::HeadCount, rank, 4, 32).unwrap();
            assert_eq!(s.len, 8);
            assert_eq!(s.start, rank * 8);
            assert_eq!(s.global_len, 32);
        }
    }

    #[test]
    fn axis_slice_shard_rejects_uneven() {
        // 32 / 3 is not even ⇒ must error today (non-contiguous shards
        // require multi-slice-per-axis, future work).
        assert!(AxisSlice::shard(KvDim::HeadCount, 0, 3, 32).is_err());
    }

    #[test]
    fn axis_slice_shard_rejects_oob_rank() {
        assert!(AxisSlice::shard(KvDim::HeadCount, 4, 4, 32).is_err());
    }

    #[test]
    fn axis_slice_new_validates_bounds() {
        assert!(AxisSlice::new(KvDim::HeadCount, 0, 0, 32).is_err());
        assert!(AxisSlice::new(KvDim::HeadCount, 24, 16, 32).is_err());
        assert!(AxisSlice::new(KvDim::HeadCount, 24, 8, 32).is_ok());
    }

    #[test]
    fn intersect_full_with_shard_yields_shard() {
        let full = AxisSlice::full(KvDim::HeadCount, 32);
        let shard = AxisSlice::shard(KvDim::HeadCount, 1, 4, 32).unwrap();
        let inter = intersect_axis(&full, &shard).unwrap().unwrap();
        // Global overlap [8, 16) ⇒ src (full) local [8, 16), dst (shard)
        // local [0, 8) (shard's local 0 == global 8).
        assert_eq!(inter.src_local, 8..16);
        assert_eq!(inter.dst_local, 0..8);
        assert_eq!(inter.len(), 8);
    }

    #[test]
    fn intersect_disjoint_returns_none() {
        // TP4 rank 2 puller (global [16, 24)) ∩ TP2 rank 0 source (global
        // [0, 16)) → disjoint.
        let puller = AxisSlice::new(KvDim::HeadCount, 16, 8, 32).unwrap();
        let source = AxisSlice::new(KvDim::HeadCount, 0, 16, 32).unwrap();
        assert!(intersect_axis(&puller, &source).unwrap().is_none());
    }

    #[test]
    fn intersect_partial_overlap() {
        // TP4 rank 2 puller (global [16, 24)) ∩ TP2 rank 1 source (global
        // [16, 32)) → overlap [16, 24); src_local [0, 8), dst_local [0, 8).
        let puller = AxisSlice::new(KvDim::HeadCount, 16, 8, 32).unwrap();
        let source = AxisSlice::new(KvDim::HeadCount, 16, 16, 32).unwrap();
        let inter = intersect_axis(&puller, &source).unwrap().unwrap();
        assert_eq!(inter.src_local, 0..8);
        assert_eq!(inter.dst_local, 0..8);
    }

    #[test]
    fn intersect_rejects_global_len_mismatch() {
        let a = AxisSlice::full(KvDim::HeadCount, 32);
        let b = AxisSlice::full(KvDim::HeadCount, 64);
        assert!(intersect_axis(&a, &b).is_err());
    }

    #[test]
    fn intersect_rejects_dim_mismatch() {
        let a = AxisSlice::full(KvDim::HeadCount, 32);
        let b = AxisSlice::full(KvDim::Page, 32);
        assert!(intersect_axis(&a, &b).is_err());
    }

    #[test]
    fn axis_extent_full_round_trip() {
        let e = AxisExtent::full(16);
        assert!(e.is_full());
        assert_eq!(e.local, 16);
        assert_eq!(e.global, 16);
        assert_eq!(e.start, 0);
    }

    #[test]
    fn axis_extent_from_slice() {
        let s = AxisSlice::shard(KvDim::HeadCount, 1, 4, 32).unwrap();
        let e = AxisExtent::from_slice(&s);
        assert_eq!(e.local, 8);
        assert_eq!(e.global, 32);
        assert_eq!(e.start, 8);
        assert!(!e.is_full());
    }

    #[test]
    fn layout_signature_serde_roundtrip() {
        let sig = LayoutSignature::new(
            vec![
                (KvDim::Block, AxisExtent::full(1024)),
                (KvDim::Page, AxisExtent::full(16)),
                (
                    KvDim::HeadCount,
                    AxisExtent::from_slice(&AxisSlice::shard(KvDim::HeadCount, 1, 4, 32).unwrap()),
                ),
                (KvDim::HeadSize, AxisExtent::full(128)),
            ],
            vec![32 * 16 * 128 * 2, 32 * 128 * 2, 128 * 2, 2],
            2,
            None,
        );
        let json = serde_json::to_string(&sig).unwrap();
        let de: LayoutSignature = serde_json::from_str(&json).unwrap();
        assert_eq!(sig, de);
    }
}
