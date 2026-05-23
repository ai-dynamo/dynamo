// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// LayoutView is the canonical addressable carrier the planner reasons
// about. Promoted from pub(crate) to pub in AB-1d so the leader-side
// cross-parallelism dispatcher (AB-2) can construct sliced views to
// describe per-shard pulls, and so the worker-side handler (AB-3) can
// consume them on receipt.

//! Sliced, stride-aware, label-driven layout views.
//!
//! [`LayoutView`] is the addressable carrier the planner reasons about.
//! It pairs a labelled axis layout with the byte strides describing the
//! underlying allocation, the per-region base addresses, and an optional
//! coordinate-space restriction (one [`AxisSlice`] per restricted axis).
//!
//! Slicing is a *coordinate-space* operation: it does not allocate, copy,
//! or reshape memory. The byte strides remain pinned to the underlying
//! allocation, and `addr_of` returns addresses inside that allocation —
//! shifted by `slice.start * stride` for in-tensor sliced axes, or
//! consumed by region-vec narrowing for region-axis slicing.
//!
//! Two views over the same global coordinate system can be intersected
//! via [`intersect_views`] to find the common transferable region — the
//! algebra used by the puller when resharding from a peer with a
//! different `(start, len)` partitioning of the same global axis.

use std::collections::HashMap;

use anyhow::{Result, bail};
use dynamo_memory::StorageKind;
use kvbm_common::{
    AxisExtent, AxisIntersection, AxisSlice, CoordByLabel, KvDim, KvDimLayout, KvDimStrides,
    LayoutSignature, intersect_axis,
};

/// A sliced, stride-aware, label-driven view over a registered KV layout.
///
/// Construct via [`LayoutView::full`] for an unrestricted view, then chain
/// [`LayoutView::slice`] / [`LayoutView::shard`] to apply coordinate-space
/// restrictions. Once built, [`LayoutView::addr_of`] is total — every
/// in-bounds [`CoordByLabel`] resolves to a byte address.
///
/// **Allocation lifetime is the caller's responsibility.** `regions`
/// holds raw byte addresses; if the underlying allocation (e.g. the
/// `Arc<dyn Layout>` the regions were extracted from) is dropped while a
/// `LayoutView` still references it, [`LayoutView::addr_of`] will hand
/// out dangling pointers. The view does not pin the allocation — keep
/// the source alive for the view's lifetime explicitly.
#[derive(Debug, Clone)]
pub struct LayoutView {
    /// Axis labels paired with **local** sizes. After slicing on axis
    /// `d`, `local_layout.size_of(d) == slice.len`. Other axes carry
    /// their original sizes.
    local_layout: KvDimLayout,

    /// Per-axis byte strides, parallel to `local_layout.dims()`. Pinned
    /// to the underlying allocation: slicing does **not** modify these.
    byte_strides: KvDimStrides,

    /// Per-region base byte addresses. When `region_axis` is sliced, this
    /// vec is narrowed at slice time so `regions[i]` (post-slice)
    /// corresponds to the slice's `i`-th region (i.e. global region
    /// `slice.start + i`).
    regions: Vec<usize>,

    /// Which axis (if any) is partitioned across separate base
    /// allocations rather than indexed within a single tensor.
    region_axis: Option<KvDim>,

    /// At most one slice per axis (today). Stored sorted by `KvDim`
    /// ordinal for stable signature generation. Axes with the trivial
    /// "full" restriction are omitted from this vec.
    ///
    /// **Forward compatibility**: non-contiguous shardings (DP-uneven,
    /// interleaved heads) will need to express multiple disjoint slices
    /// on the same axis. The storage type already supports that; the
    /// `single-slice-per-axis` invariant is enforced at construction
    /// time and can be relaxed without API breakage when the multi-slice
    /// intersection algebra is implemented.
    slices: Vec<AxisSlice>,

    /// Per-axis [`StorageKind`], parallel to `local_layout.dims()`.
    ///
    /// Today every producer populates this homogeneously (all axes carry
    /// the layout's single `StorageKind`) so production behaviour is
    /// unchanged. Future PRs (PR-7.7.1+) will populate heterogeneous
    /// vectors — e.g. disk axes mixed with device axes — when the planner
    /// gains heterogeneous-storage dispatch.
    ///
    /// Slicing does **not** modify this vec: axis indices stay stable and
    /// the storage kind for a sliced axis is unchanged.
    axis_storage_kinds: Vec<StorageKind>,
}

impl LayoutView {
    /// Build an unrestricted view from raw parts. Validates cross-field
    /// invariants:
    ///
    /// 1. `byte_strides` rank matches `layout` rank.
    /// 2. `axis_storage_kinds` length matches `layout` rank (one entry per
    ///    axis, parallel to `layout.dims()`).
    /// 3. If `region_axis` is `Some(d)`: `d` appears in `layout`,
    ///    `regions.len() == layout.size_of(d).unwrap()`, and
    ///    `regions.len() > 0`.
    /// 4. If `region_axis` is `None`: `regions.len() == 1`.
    ///
    /// Today's producers pass a homogeneous `axis_storage_kinds` (all
    /// axes carry the same [`StorageKind`]). Future producers can pass a
    /// mixed vec to express per-axis heterogeneous storage.
    pub fn full(
        layout: KvDimLayout,
        byte_strides: KvDimStrides,
        regions: Vec<usize>,
        region_axis: Option<KvDim>,
        axis_storage_kinds: Vec<StorageKind>,
    ) -> Result<Self> {
        let rank = layout.dims().len();
        if byte_strides.as_bytes().len() != rank {
            bail!(
                "LayoutView::full: stride rank ({}) != layout rank ({})",
                byte_strides.as_bytes().len(),
                rank
            );
        }
        if axis_storage_kinds.len() != rank {
            bail!(
                "LayoutView::full: axis_storage_kinds length ({}) != layout rank ({})",
                axis_storage_kinds.len(),
                rank
            );
        }
        match region_axis {
            Some(d) => {
                let region_size = layout.size_of(d).ok_or_else(|| {
                    anyhow::anyhow!(
                        "LayoutView::full: region_axis {d:?} not present in layout {:?}",
                        layout.dims()
                    )
                })?;
                if regions.len() != region_size {
                    bail!(
                        "LayoutView::full: regions vec has {} entries but region_axis {d:?} \
                         size is {region_size}",
                        regions.len()
                    );
                }
            }
            None => {
                if regions.len() != 1 {
                    bail!(
                        "LayoutView::full: regions vec must have exactly 1 entry when \
                         region_axis is None, got {}",
                        regions.len()
                    );
                }
            }
        }

        Ok(Self {
            local_layout: layout,
            byte_strides,
            regions,
            region_axis,
            slices: Vec::new(),
            axis_storage_kinds,
        })
    }

    /// Apply a coordinate-space restriction on `dim`, narrowing the
    /// addressable extent of that axis to `[start, start + len)` of the
    /// view's *current* (pre-slice) coordinate space.
    ///
    /// Composability: this method errors if `dim` is already sliced —
    /// re-slicing the same axis is not yet supported (it would need
    /// careful global-coord arithmetic for the post-slice signature, and
    /// no consumer needs it today). To re-slice, drop and rebuild from
    /// the base.
    pub fn slice(mut self, dim: KvDim, start: usize, len: usize) -> Result<Self> {
        if self.slices.iter().any(|s| s.dim == dim) {
            bail!("LayoutView::slice: axis {dim:?} already sliced; rebuild from base instead");
        }
        let global_len = self.local_layout.size_of(dim).ok_or_else(|| {
            anyhow::anyhow!(
                "LayoutView::slice: axis {dim:?} not present in layout {:?}",
                self.local_layout.dims()
            )
        })?;
        let axis_slice = AxisSlice::new(dim, start, len, global_len)?;

        // Rebuild local_layout with the sliced axis size shrunk.
        let new_sizes: Vec<usize> = self
            .local_layout
            .dims()
            .iter()
            .zip(self.local_layout.sizes())
            .map(|(d, &s)| if *d == dim { len } else { s })
            .collect();
        self.local_layout = KvDimLayout::new(self.local_layout.dims().to_vec(), new_sizes)?;

        // Narrow regions vec when slicing the region axis.
        if Some(dim) == self.region_axis {
            self.regions = self.regions[start..start + len].to_vec();
        }

        // Insert the slice in dim-ordinal order so the slices vec is
        // canonically ordered for signature generation.
        let insert_pos = self
            .slices
            .iter()
            .position(|s| dim_ordinal(s.dim) > dim_ordinal(dim))
            .unwrap_or(self.slices.len());
        self.slices.insert(insert_pos, axis_slice);

        Ok(self)
    }

    /// Even-split helper: shard `dim` into `world` equal pieces and keep
    /// the `rank`-th piece. Errors if `world == 0`, `rank >= world`, or
    /// `dim`'s extent is not divisible by `world`.
    pub fn shard(self, dim: KvDim, rank: usize, world: usize) -> Result<Self> {
        let global_len = self.local_layout.size_of(dim).ok_or_else(|| {
            anyhow::anyhow!(
                "LayoutView::shard: axis {dim:?} not present in layout {:?}",
                self.local_layout.dims()
            )
        })?;
        let s = AxisSlice::shard(dim, rank, world, global_len)?;
        self.slice(dim, s.start, s.len)
    }

    /// Local (post-slicing) layout. Every axis `d` reports `size =
    /// slice_for(d).len` if sliced, else its base size.
    pub fn local_layout(&self) -> &KvDimLayout {
        &self.local_layout
    }

    /// Per-axis byte strides over the underlying allocation. Pinned —
    /// slicing does not change these.
    pub fn byte_strides(&self) -> &KvDimStrides {
        &self.byte_strides
    }

    /// Per-region base addresses (post-narrowing if `region_axis` was
    /// sliced).
    pub fn regions(&self) -> &[usize] {
        &self.regions
    }

    pub fn region_axis(&self) -> Option<KvDim> {
        self.region_axis
    }

    /// Active axis slices, in `KvDim` ordinal order. Empty when the view
    /// has no restrictions. **Multi-slice-per-axis is not yet exposed;
    /// callers can rely on at most one entry per `dim` for now.**
    pub fn slices(&self) -> &[AxisSlice] {
        &self.slices
    }

    /// Per-axis [`StorageKind`], parallel to `local_layout().dims()`.
    ///
    /// Today's views are always homogeneous (all axes carry the same kind).
    /// Future views from heterogeneous-storage layouts (e.g. disk head
    /// axes mixed with device inner axes) will return a mixed slice.
    pub fn axis_storage_kinds(&self) -> &[StorageKind] {
        &self.axis_storage_kinds
    }

    /// Returns `true` when the view contains more than one distinct
    /// [`StorageKind`] across its axes.
    ///
    /// An empty view, a single-axis view, or any view where all axes
    /// carry the same kind returns `false`. A view with, say,
    /// `[Device(0), Device(0), System, System]` returns `true`.
    pub fn is_heterogeneous(&self) -> bool {
        let mut iter = self.axis_storage_kinds.iter();
        match iter.next() {
            None => false,
            Some(first) => iter.any(|k| k != first),
        }
    }

    /// Element size (bytes) the strides were built against.
    pub fn elem_size(&self) -> usize {
        self.byte_strides.elem_size()
    }

    /// Address-free, hashable description for catalog lookup and
    /// cross-process matching. See [`LayoutSignature`] for stability
    /// caveats.
    pub fn signature(&self) -> LayoutSignature {
        let slice_by_dim: HashMap<KvDim, &AxisSlice> =
            self.slices.iter().map(|s| (s.dim, s)).collect();
        let axes: Vec<(KvDim, AxisExtent)> = self
            .local_layout
            .dims()
            .iter()
            .zip(self.local_layout.sizes())
            .map(|(d, &local_size)| {
                let extent = match slice_by_dim.get(d) {
                    Some(s) => AxisExtent::from_slice(s),
                    None => AxisExtent::full(local_size),
                };
                (*d, extent)
            })
            .collect();

        LayoutSignature::new(
            axes,
            self.byte_strides.as_bytes().to_vec(),
            self.byte_strides.elem_size(),
            self.region_axis,
        )
    }

    /// Resolve a label-keyed coordinate to a byte address.
    ///
    /// Errors when:
    /// - `region_axis` is `Some(d)` but `coord.get(d)` is missing.
    /// - any in-tensor axis present in `local_layout` is missing from
    ///   `coord`.
    /// - any provided coord is `>=` its axis's local size.
    pub fn addr_of(&self, coord: &CoordByLabel) -> Result<usize> {
        // Resolve region base.
        let region_base = match self.region_axis {
            Some(d) => {
                let idx = coord.get(d).ok_or_else(|| {
                    anyhow::anyhow!("LayoutView::addr_of: missing coord for region_axis {d:?}")
                })?;
                let local_region_count = self
                    .local_layout
                    .size_of(d)
                    .expect("region_axis must appear in local_layout (validated at construction)");
                if idx >= local_region_count {
                    bail!(
                        "LayoutView::addr_of: region_axis {d:?} coord {idx} >= local size \
                         {local_region_count}"
                    );
                }
                self.regions[idx]
            }
            None => self.regions[0],
        };

        // Pre-build a per-axis slice-start lookup. Region axis is excluded
        // — its slice (if any) was consumed by region-vec narrowing.
        let mut offset = 0usize;
        let dims = self.local_layout.dims();
        let sizes = self.local_layout.sizes();
        let strides = self.byte_strides.as_bytes();
        let slice_by_dim: HashMap<KvDim, &AxisSlice> =
            self.slices.iter().map(|s| (s.dim, s)).collect();

        for (i, d) in dims.iter().enumerate() {
            if Some(*d) == self.region_axis {
                continue;
            }
            let c = coord.get(*d).ok_or_else(|| {
                anyhow::anyhow!("LayoutView::addr_of: missing coord for axis {d:?}")
            })?;
            if c >= sizes[i] {
                bail!(
                    "LayoutView::addr_of: axis {d:?} coord {c} >= local size {}",
                    sizes[i]
                );
            }
            let start = slice_by_dim.get(d).map(|s| s.start).unwrap_or(0);
            // Plain arithmetic on purpose: saturating math here would mask
            // genuine address-overflow bugs as `usize::MAX`-clamped
            // garbage. The bounds check above + KvDimStrides' positive-
            // stride invariant make overflow possible only via a
            // malformed layout — which should panic loudly, not silently
            // hand back a clamped address.
            offset += strides[i] * (start + c);
        }

        Ok(region_base + offset)
    }
}

/// Per-axis intersection of two `LayoutView`s.
///
/// Returns:
/// - `Err` if the two views don't carry the same set of `KvDim` labels.
///   Different label sets describe different schemas (e.g. one has
///   `Payload`, the other has `HeadSize`) and cannot be transferred
///   between by coordinate-space matching alone.
/// - `Err` if any axis has incompatible global extents (different model
///   configs entirely).
/// - `Ok(None)` if at least one axis is fully disjoint.
/// - `Ok(Some(vec))` with one [`AxisIntersection`] per axis where
///   *either* side is restricted. Axes that are full-on-both-sides are
///   omitted (the entire axis is implicitly transferred).
///
/// Notes on structural permutation tolerance: this function operates
/// purely on coordinate-space restrictions. It tolerates differences in
/// axis order between the two views, and tolerates `region_axis`
/// disagreement (e.g. src has `Layer` as region, dst has `Layer`
/// in-tensor) — the structural difference is the planner's problem to
/// resolve via per-region iteration, not the intersection algebra's.
pub fn intersect_views(
    src: &LayoutView,
    dst: &LayoutView,
) -> Result<Option<Vec<AxisIntersection>>> {
    let src_dims: Vec<KvDim> = src.local_layout.dims().to_vec();
    let dst_dims: Vec<KvDim> = dst.local_layout.dims().to_vec();
    if !same_label_set(&src_dims, &dst_dims) {
        bail!(
            "intersect_views: label-set mismatch — src has {:?}, dst has {:?}",
            src_dims,
            dst_dims
        );
    }

    let src_slice_by_dim: HashMap<KvDim, &AxisSlice> =
        src.slices.iter().map(|s| (s.dim, s)).collect();
    let dst_slice_by_dim: HashMap<KvDim, &AxisSlice> =
        dst.slices.iter().map(|s| (s.dim, s)).collect();

    let mut out: Vec<AxisIntersection> = Vec::new();
    for d in &src_dims {
        // Build per-side AxisSlices, defaulting to full when the side
        // has no restriction. This forces a global_len agreement check
        // even when neither side is sliced — exactly the structural
        // mismatch we want to surface.
        let src_global = src.global_size(*d);
        let dst_global = dst.global_size(*d);
        if src_global != dst_global {
            bail!(
                "intersect_views ({d:?}): global extent mismatch — src has {src_global}, \
                 dst has {dst_global}; layouts describe different model configs"
            );
        }
        let src_axis = src_slice_by_dim
            .get(d)
            .copied()
            .cloned()
            .unwrap_or_else(|| AxisSlice::full(*d, src_global));
        let dst_axis = dst_slice_by_dim
            .get(d)
            .copied()
            .cloned()
            .unwrap_or_else(|| AxisSlice::full(*d, dst_global));

        match intersect_axis(&src_axis, &dst_axis)? {
            Some(inter) => {
                if !(src_axis.is_full() && dst_axis.is_full()) {
                    out.push(inter);
                }
            }
            None => return Ok(None),
        }
    }

    Ok(Some(out))
}

impl LayoutView {
    /// Global (pre-slicing) size of `dim`. Equals `slice.global_len` when
    /// sliced, else the local layout's size for that axis.
    fn global_size(&self, dim: KvDim) -> usize {
        if let Some(s) = self.slices.iter().find(|s| s.dim == dim) {
            s.global_len
        } else {
            self.local_layout.size_of(dim).unwrap_or(0)
        }
    }
}

fn same_label_set(a: &[KvDim], b: &[KvDim]) -> bool {
    if a.len() != b.len() {
        return false;
    }
    let mut a_sorted: Vec<KvDim> = a.to_vec();
    let mut b_sorted: Vec<KvDim> = b.to_vec();
    a_sorted.sort_by_key(|d| dim_ordinal(*d));
    b_sorted.sort_by_key(|d| dim_ordinal(*d));
    a_sorted == b_sorted
}

const fn dim_ordinal(d: KvDim) -> usize {
    match d {
        KvDim::Block => 0,
        KvDim::Layer => 1,
        KvDim::Outer => 2,
        KvDim::Page => 3,
        KvDim::HeadCount => 4,
        KvDim::HeadSize => 5,
        KvDim::Payload => 6,
    }
}

#[cfg(all(test, feature = "testing-kvbm"))]
mod tests {
    use super::*;

    /// Build a per-layer NHD layout with HeadCount as the head axis.
    /// `[Outer, Block, Page, HeadCount, HeadSize]` × per-layer regions.
    fn nhd_per_layer_view(
        num_layers: usize,
        num_blocks: usize,
        page: usize,
        head_count: usize,
        head_size: usize,
        elem_size: usize,
    ) -> LayoutView {
        let layout = KvDimLayout::new(
            vec![
                KvDim::Layer,
                KvDim::Outer,
                KvDim::Block,
                KvDim::Page,
                KvDim::HeadCount,
                KvDim::HeadSize,
            ],
            vec![num_layers, 2, num_blocks, page, head_count, head_size],
        )
        .unwrap();
        // Layer is the region axis; in-tensor axes are
        // [Outer, Block, Page, HeadCount, HeadSize].
        let in_tensor_layout = KvDimLayout::new(
            vec![
                KvDim::Outer,
                KvDim::Block,
                KvDim::Page,
                KvDim::HeadCount,
                KvDim::HeadSize,
            ],
            vec![2, num_blocks, page, head_count, head_size],
        )
        .unwrap();
        let in_tensor_strides = KvDimStrides::contiguous_for(&in_tensor_layout, elem_size);
        // Layer axis stride is unused (region) but must be present and
        // positive. Set it to the per-layer tensor size so it agrees with
        // a hypothetical fully-contiguous parent.
        let region_size = 2 * num_blocks * page * head_count * head_size * elem_size;
        let mut byte_strides = vec![region_size];
        byte_strides.extend_from_slice(in_tensor_strides.as_bytes());
        let strides = KvDimStrides::from_byte_strides(byte_strides, elem_size).unwrap();

        // Hand-rolled region bases: each layer at a distinct, well-spaced
        // offset to make per-region arithmetic easy to verify.
        let regions: Vec<usize> = (0..num_layers)
            .map(|i| 0x1000_0000 + i * 0x10_0000)
            .collect();

        // Homogeneous System storage for all 6 axes (Layer + 5 in-tensor).
        let axis_storage_kinds = vec![StorageKind::System; layout.dims().len()];
        LayoutView::full(
            layout,
            strides,
            regions,
            Some(KvDim::Layer),
            axis_storage_kinds,
        )
        .unwrap()
    }

    #[test]
    fn full_view_addr_of_matches_stride_math() {
        let view = nhd_per_layer_view(4, 16, 8, 4, 64, 2);
        // Coord: layer=2, outer=1, block=3, page=5, head_count=2, head_size=10
        let c = CoordByLabel::new()
            .with(KvDim::Layer, 2)
            .with(KvDim::Outer, 1)
            .with(KvDim::Block, 3)
            .with(KvDim::Page, 5)
            .with(KvDim::HeadCount, 2)
            .with(KvDim::HeadSize, 10);
        let addr = view.addr_of(&c).unwrap();
        // strides (elem=2) over per-layer in-tensor shape
        // [Outer=2, Block=16, Page=8, HeadCount=4, HeadSize=64]:
        //   HeadSize=2, HeadCount=128, Page=512, Block=4096, Outer=65536.
        // expected = regions[2] + Σ stride[i] * coord[i]
        let expected =
            (0x1000_0000 + 2 * 0x10_0000) + 65536 + 4096 * 3 + 512 * 5 + 128 * 2 + 2 * 10;
        assert_eq!(addr, expected);
    }

    #[test]
    fn slice_does_not_change_byte_strides() {
        let base = nhd_per_layer_view(4, 16, 8, 4, 64, 2);
        let strides_before = base.byte_strides().as_bytes().to_vec();
        let sliced = base.slice(KvDim::HeadCount, 1, 2).unwrap();
        assert_eq!(sliced.byte_strides().as_bytes(), strides_before);
    }

    #[test]
    fn slice_addr_of_adds_start_offset_for_in_tensor_axis() {
        let base = nhd_per_layer_view(4, 16, 8, 4, 64, 2);
        let sliced = base.slice(KvDim::HeadCount, 1, 2).unwrap();
        // local HeadCount coord 0 ↔ global HeadCount coord 1.
        let local = CoordByLabel::new()
            .with(KvDim::Layer, 0)
            .with(KvDim::Outer, 0)
            .with(KvDim::Block, 0)
            .with(KvDim::Page, 0)
            .with(KvDim::HeadCount, 0)
            .with(KvDim::HeadSize, 0);
        let global = CoordByLabel::new()
            .with(KvDim::Layer, 0)
            .with(KvDim::Outer, 0)
            .with(KvDim::Block, 0)
            .with(KvDim::Page, 0)
            .with(KvDim::HeadCount, 1)
            .with(KvDim::HeadSize, 0);
        let sliced_addr = sliced.addr_of(&local).unwrap();
        let base_addr = nhd_per_layer_view(4, 16, 8, 4, 64, 2)
            .addr_of(&global)
            .unwrap();
        assert_eq!(sliced_addr, base_addr);
    }

    #[test]
    fn slice_rejects_oob_local_coord() {
        let view = nhd_per_layer_view(4, 16, 8, 4, 64, 2)
            .slice(KvDim::HeadCount, 1, 2)
            .unwrap();
        let c = CoordByLabel::new()
            .with(KvDim::Layer, 0)
            .with(KvDim::Outer, 0)
            .with(KvDim::Block, 0)
            .with(KvDim::Page, 0)
            .with(KvDim::HeadCount, 2) // local size = 2; valid: 0..=1
            .with(KvDim::HeadSize, 0);
        assert!(view.addr_of(&c).is_err());
    }

    #[test]
    fn shard_partitions_evenly() {
        let v = nhd_per_layer_view(4, 16, 8, 4, 64, 2)
            .shard(KvDim::HeadCount, 1, 2)
            .unwrap();
        assert_eq!(v.local_layout().size_of(KvDim::HeadCount), Some(2));
        let s = &v.slices()[0];
        assert_eq!(s.start, 2);
        assert_eq!(s.len, 2);
    }

    #[test]
    fn shard_rejects_uneven() {
        let v = nhd_per_layer_view(4, 16, 8, 4, 64, 2);
        assert!(v.shard(KvDim::HeadCount, 0, 3).is_err());
    }

    #[test]
    fn region_axis_slice_narrows_regions_vec() {
        let view = nhd_per_layer_view(4, 16, 8, 4, 64, 2)
            .slice(KvDim::Layer, 1, 2)
            .unwrap();
        assert_eq!(view.regions().len(), 2);
        assert_eq!(view.regions()[0], 0x1000_0000 + 0x10_0000);
        assert_eq!(view.regions()[1], 0x1000_0000 + 2 * 0x10_0000);

        // local Layer coord 0 ↔ global Layer coord 1 ↔ regions[1] in base.
        let c = CoordByLabel::new()
            .with(KvDim::Layer, 0)
            .with(KvDim::Outer, 0)
            .with(KvDim::Block, 0)
            .with(KvDim::Page, 0)
            .with(KvDim::HeadCount, 0)
            .with(KvDim::HeadSize, 0);
        let addr = view.addr_of(&c).unwrap();
        assert_eq!(addr, 0x1000_0000 + 0x10_0000);
    }

    #[test]
    fn multi_axis_slice_composes() {
        let view = nhd_per_layer_view(4, 16, 8, 4, 64, 2)
            .slice(KvDim::Layer, 1, 2)
            .unwrap()
            .slice(KvDim::HeadCount, 1, 2)
            .unwrap();
        assert_eq!(view.local_layout().size_of(KvDim::Layer), Some(2));
        assert_eq!(view.local_layout().size_of(KvDim::HeadCount), Some(2));

        // local (Layer=1, HeadCount=1) ↔ global (Layer=2, HeadCount=2).
        let local = CoordByLabel::new()
            .with(KvDim::Layer, 1)
            .with(KvDim::Outer, 0)
            .with(KvDim::Block, 0)
            .with(KvDim::Page, 0)
            .with(KvDim::HeadCount, 1)
            .with(KvDim::HeadSize, 0);
        let global = CoordByLabel::new()
            .with(KvDim::Layer, 2)
            .with(KvDim::Outer, 0)
            .with(KvDim::Block, 0)
            .with(KvDim::Page, 0)
            .with(KvDim::HeadCount, 2)
            .with(KvDim::HeadSize, 0);
        let actual = view.addr_of(&local).unwrap();
        let expected = nhd_per_layer_view(4, 16, 8, 4, 64, 2)
            .addr_of(&global)
            .unwrap();
        assert_eq!(actual, expected);
    }

    #[test]
    fn rejects_re_slicing_same_axis() {
        let result = nhd_per_layer_view(4, 16, 8, 4, 64, 2)
            .slice(KvDim::HeadCount, 1, 2)
            .unwrap()
            .slice(KvDim::HeadCount, 0, 1);
        assert!(result.is_err());
    }

    #[test]
    fn signature_includes_slice_extents() {
        let view = nhd_per_layer_view(4, 16, 8, 4, 64, 2)
            .shard(KvDim::HeadCount, 1, 2)
            .unwrap();
        let sig = view.signature();
        let head_count = sig
            .axes
            .iter()
            .find(|(d, _)| *d == KvDim::HeadCount)
            .unwrap();
        assert_eq!(head_count.1.local, 2);
        assert_eq!(head_count.1.global, 4);
        assert_eq!(head_count.1.start, 2);
    }

    #[test]
    fn signature_round_trips_serde() {
        let view = nhd_per_layer_view(4, 16, 8, 4, 64, 2)
            .slice(KvDim::HeadCount, 1, 2)
            .unwrap();
        let sig = view.signature();
        let json = serde_json::to_string(&sig).unwrap();
        let de: LayoutSignature = serde_json::from_str(&json).unwrap();
        assert_eq!(sig, de);
    }

    #[test]
    fn intersect_tp1_full_with_tp4_rank_1() {
        // Both views: HeadCount global = 32. TP1 puller is full; TP4
        // rank 1 is HeadCount[8, 16). Intersection covers exactly rank 1.
        let tp1 = build_view_with_head_count(32);
        let tp4_r1 = build_view_with_head_count(32)
            .shard(KvDim::HeadCount, 1, 4)
            .unwrap();
        let inters = intersect_views(&tp1, &tp4_r1).unwrap().unwrap();
        let head = inters.iter().find(|i| i.dim == KvDim::HeadCount).unwrap();
        assert_eq!(head.src_local, 8..16);
        assert_eq!(head.dst_local, 0..8);
    }

    #[test]
    fn intersect_tp4_rank_2_with_tp2_rank_0_is_disjoint() {
        // TP4 rank 2 covers HeadCount[16, 24); TP2 rank 0 covers [0, 16).
        // Disjoint.
        let puller = build_view_with_head_count(32)
            .shard(KvDim::HeadCount, 2, 4)
            .unwrap();
        let source = build_view_with_head_count(32)
            .shard(KvDim::HeadCount, 0, 2)
            .unwrap();
        assert!(intersect_views(&puller, &source).unwrap().is_none());
    }

    #[test]
    fn intersect_tp4_rank_2_with_tp2_rank_1_overlaps_at_local_origin() {
        // TP4 rank 2 covers HeadCount[16, 24); TP2 rank 1 covers
        // HeadCount[16, 32). Overlap [16, 24); both sides report local
        // [0, 8).
        let puller = build_view_with_head_count(32)
            .shard(KvDim::HeadCount, 2, 4)
            .unwrap();
        let source = build_view_with_head_count(32)
            .shard(KvDim::HeadCount, 1, 2)
            .unwrap();
        let inters = intersect_views(&puller, &source).unwrap().unwrap();
        let head = inters.iter().find(|i| i.dim == KvDim::HeadCount).unwrap();
        assert_eq!(head.src_local, 0..8);
        assert_eq!(head.dst_local, 0..8);
    }

    #[test]
    fn intersect_rejects_global_size_mismatch() {
        let a = build_view_with_head_count(32);
        let b = build_view_with_head_count(64);
        assert!(intersect_views(&a, &b).is_err());
    }

    #[test]
    fn intersect_full_full_returns_no_axes() {
        let a = build_view_with_head_count(32);
        let b = build_view_with_head_count(32);
        let inters = intersect_views(&a, &b).unwrap().unwrap();
        // Both sides full on every axis ⇒ no axis recorded.
        assert!(inters.is_empty());
    }

    /// Cross-layer ↔ per-layer: src has `Layer` as the region_axis (one
    /// per-layer tensor per region); dst has `Layer` in-tensor (a single
    /// fully-contiguous allocation). Both sides slice `Layer` to test
    /// that intersect_views is purely coordinate-space — it tolerates
    /// the structural difference and reports the global-coord overlap.
    #[test]
    fn intersect_tolerates_region_axis_disagreement() {
        // src: per-layer regions sliced to Layer[1, 3) of global 4 layers.
        let src = nhd_per_layer_view(4, 16, 8, 4, 64, 2)
            .slice(KvDim::Layer, 1, 2)
            .unwrap();

        // dst: cross-layer (Layer in-tensor, region_axis = None) sliced
        // to Layer[2, 4) of global 4 layers.
        let dst = cross_layer_view(4, 16, 8, 4, 64, 2)
            .slice(KvDim::Layer, 2, 2)
            .unwrap();

        let inters = intersect_views(&src, &dst).unwrap().unwrap();
        // Only Layer is partially restricted on either side; HeadCount /
        // Page / etc. are full-full and omitted.
        assert_eq!(inters.len(), 1);
        let layer = &inters[0];
        assert_eq!(layer.dim, KvDim::Layer);
        // Global overlap is [2, 3): src starts at 1 ⇒ src_local 1..2;
        // dst starts at 2 ⇒ dst_local 0..1.
        assert_eq!(layer.src_local, 1..2);
        assert_eq!(layer.dst_local, 0..1);
    }

    /// Build a view with `HeadCount = head_count` and other axes fixed —
    /// used by the intersection tests where only HeadCount varies.
    fn build_view_with_head_count(head_count: usize) -> LayoutView {
        nhd_per_layer_view(4, 16, 8, head_count, 64, 2)
    }

    // ── PR-7.7: per-axis StorageKind tests ──────────────────────────────────

    /// FC-like homogeneous view: all axes report `System`, `is_heterogeneous == false`.
    #[test]
    fn layout_view_axis_storage_homogeneous_single_storage_kind() {
        let view = cross_layer_view(4, 16, 8, 4, 64, 2);
        let kinds = view.axis_storage_kinds();
        assert!(!kinds.is_empty(), "axis_storage_kinds must be non-empty");
        assert!(
            kinds.iter().all(|k| *k == StorageKind::System),
            "all axes must report System"
        );
        assert!(
            !view.is_heterogeneous(),
            "homogeneous view must not be heterogeneous"
        );
    }

    /// LS-like per-layer homogeneous view: all axes report `System`,
    /// `is_heterogeneous == false`.
    #[test]
    fn layout_view_axis_storage_homogeneous_for_per_layer_view() {
        let view = nhd_per_layer_view(4, 16, 8, 4, 64, 2);
        let kinds = view.axis_storage_kinds();
        assert_eq!(kinds.len(), view.local_layout().dims().len());
        assert!(
            kinds.iter().all(|k| *k == StorageKind::System),
            "all axes must report System for a per-layer test view"
        );
        assert!(!view.is_heterogeneous());
    }

    /// Edge case: a single-axis view must not be heterogeneous.
    #[test]
    fn layout_view_homogeneous_with_single_axis() {
        let layout = KvDimLayout::new(vec![KvDim::Block], vec![8]).unwrap();
        let strides = KvDimStrides::from_byte_strides(vec![4096], 2).unwrap();
        let view = LayoutView::full(
            layout,
            strides,
            vec![0x1000],
            None,
            vec![StorageKind::Device(0)],
        )
        .unwrap();
        assert_eq!(view.axis_storage_kinds(), &[StorageKind::Device(0)]);
        assert!(!view.is_heterogeneous());
    }

    /// Synthetic heterogeneous fixture: mix Device(0) and System axes.
    /// `is_heterogeneous` must return `true` and `axis_storage_kinds`
    /// must faithfully report the input slice.
    #[test]
    fn layout_view_synthetic_heterogeneous_fixture() {
        // 4-axis layout: [Block, Layer, Page, HeadSize].
        // We label the first two axes Device(0), the last two System —
        // a contrived but structurally valid heterogeneous example.
        let layout = KvDimLayout::new(
            vec![KvDim::Block, KvDim::Layer, KvDim::Page, KvDim::HeadSize],
            vec![4, 2, 8, 64],
        )
        .unwrap();
        let strides = KvDimStrides::contiguous_for(&layout, 2);
        let mixed = vec![
            StorageKind::Device(0),
            StorageKind::Device(0),
            StorageKind::System,
            StorageKind::System,
        ];
        let view = LayoutView::full(layout, strides, vec![0x1000], None, mixed.clone()).unwrap();
        assert_eq!(view.axis_storage_kinds(), mixed.as_slice());
        assert!(
            view.is_heterogeneous(),
            "mixed storage kinds must be detected as heterogeneous"
        );
    }

    /// Build a fully-contiguous (cross-layer) NHD view with `Layer` as an
    /// in-tensor axis and `region_axis = None`. Single allocation.
    fn cross_layer_view(
        num_layers: usize,
        num_blocks: usize,
        page: usize,
        head_count: usize,
        head_size: usize,
        elem_size: usize,
    ) -> LayoutView {
        let layout = KvDimLayout::new(
            vec![
                KvDim::Block,
                KvDim::Layer,
                KvDim::Outer,
                KvDim::Page,
                KvDim::HeadCount,
                KvDim::HeadSize,
            ],
            vec![num_blocks, num_layers, 2, page, head_count, head_size],
        )
        .unwrap();
        let strides = KvDimStrides::contiguous_for(&layout, elem_size);
        let axis_storage_kinds = vec![StorageKind::System; layout.dims().len()];
        LayoutView::full(layout, strides, vec![0x2000_0000], None, axis_storage_kinds).unwrap()
    }
}
