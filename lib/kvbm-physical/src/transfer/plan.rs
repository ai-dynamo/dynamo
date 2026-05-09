// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Stride-aware, label-driven copy planner.
//!
//! Given two [`AnnotatedLayout`]s — each carrying a [`KvDimLayout`]
//! schema, per-axis byte strides, and per-region base addresses — plus
//! a list of `(src_block_id, dst_block_id)` pairs and a [`CopyPolicy`],
//! [`plan_copy`] emits a coalesced [`CopyPlan`]: either a
//! `Vec<CopyOp>` of `(src, dst, size)` triples (the `Direct` path) or
//! a deferred [`CopyPlan::Transform`] when the matching contiguous
//! tail is too small to amortise launch overhead.
//!
//! The planner is pure addressing math — no GPU, no NIXL, no
//! allocations beyond the output `Vec<CopyOp>`. It is **not yet wired
//! into the executor**; the existing `transfer/executor`'s
//! `select_transform_kernel(KvBlockLayout, KvBlockLayout)` keeps
//! working unchanged. Tests in this module are the only consumer.
//!
//! ## Block-pair semantics
//!
//! Real KV-cache transfers carry two parallel block-id lists (one per
//! side); the planner takes them as `&[(src_block_id, dst_block_id)]`
//! pairs. For a plain "copy this block back to itself" call, pass
//! pairs like `[(b, b), ...]`. The src block id selects the row of
//! the Block axis on the src `addr_of`; the dst id does the same on
//! dst.
//!
//! ## Transform permutation contract
//!
//! `Transform.permutation` is built over the *in-tensor* axes of each
//! side — `region_axis` is excluded from both before computing the
//! index map. When `src.region_axis()` and `dst.region_axis()` disagree
//! (e.g. src has `Layer` as its region axis while dst carries `Layer`
//! in-tensor), the planner emits per-region iteration in the outer
//! loop and does **not** fold `Layer` into the permutation vector. This
//! keeps `permutation` well-defined as a pure in-tensor index map and
//! isolates the partitioned-vs-contiguous concern to iteration
//! scaffolding.

use anyhow::{Result, bail};

use kvbm_common::{KvDim, KvDimLayout, KvDimStrides};

/// `KvDim` is a closed enum with seven variants; we use this constant
/// to size [`CoordByLabel`]'s fixed array.
const KV_DIM_COUNT: usize = 7;

fn kv_dim_index(d: KvDim) -> usize {
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

/// Per-axis coordinate keyed by label. `Some(v)` means the caller
/// supplied a value for that axis; `None` means the axis isn't part of
/// this layout (e.g. MLA has no `HeadCount`).
///
/// Backed by a fixed array because [`KvDim`] is a closed enum — adding
/// a new variant is a load-bearing semantic change that demands
/// recompiling every consumer of `CoordByLabel` anyway.
#[derive(Debug, Clone, Default)]
pub struct CoordByLabel([Option<usize>; KV_DIM_COUNT]);

impl CoordByLabel {
    pub fn new() -> Self {
        Self([None; KV_DIM_COUNT])
    }

    pub fn set(mut self, d: KvDim, v: usize) -> Self {
        self.0[kv_dim_index(d)] = Some(v);
        self
    }

    pub fn set_in_place(&mut self, d: KvDim, v: usize) {
        self.0[kv_dim_index(d)] = Some(v);
    }

    pub fn get(&self, d: KvDim) -> Option<usize> {
        self.0[kv_dim_index(d)]
    }
}

/// A label-annotated, stride-described, addressable layout.
///
/// Construct via [`AnnotatedLayout::new`], which enforces:
///
/// 1. `byte_strides.as_bytes().len() == dim_layout.dims().len()`.
/// 2. If `region_axis` is `Some(d)`: `d` appears in `dim_layout`, and
///    `regions.len() == dim_layout.size_of(d).unwrap()`.
/// 3. If `region_axis` is `None`: `regions.len() == 1`.
/// 4. Every `byte_strides` entry is positive (validated by
///    [`KvDimStrides`] at its own construction time).
///
/// [`AnnotatedLayout::new`] validates the *layout* — once it returns
/// `Ok`, the schema, strides, and region count are mutually consistent.
/// Per-call coordinate validity (each axis value is `< size_of(axis)`,
/// and the region-axis coordinate is supplied) is checked by
/// [`AnnotatedLayout::addr_of`], which returns `Result`.
///
/// [`new`]: AnnotatedLayout::new
#[derive(Debug, Clone)]
pub struct AnnotatedLayout {
    regions: Vec<usize>,
    region_axis: Option<KvDim>,
    dim_layout: KvDimLayout,
    byte_strides: KvDimStrides,
}

impl AnnotatedLayout {
    pub fn new(
        regions: Vec<usize>,
        region_axis: Option<KvDim>,
        dim_layout: KvDimLayout,
        byte_strides: KvDimStrides,
    ) -> Result<Self> {
        if byte_strides.as_bytes().len() != dim_layout.dims().len() {
            bail!(
                "AnnotatedLayout: byte_strides rank {} does not match dim_layout rank {}",
                byte_strides.as_bytes().len(),
                dim_layout.dims().len(),
            );
        }
        match region_axis {
            Some(d) => {
                let size = dim_layout.size_of(d).ok_or_else(|| {
                    anyhow::anyhow!(
                        "AnnotatedLayout: region_axis {d} is not present in dim_layout {:?}",
                        dim_layout.dims()
                    )
                })?;
                if regions.len() != size {
                    bail!(
                        "AnnotatedLayout: regions.len() ({}) != size of region_axis {d} ({size})",
                        regions.len(),
                    );
                }
            }
            None => {
                if regions.len() != 1 {
                    bail!(
                        "AnnotatedLayout: region_axis is None, regions.len() must be 1, got {}",
                        regions.len(),
                    );
                }
            }
        }
        Ok(Self {
            regions,
            region_axis,
            dim_layout,
            byte_strides,
        })
    }

    /// Byte address for a labelled coordinate.
    ///
    /// For each axis `i` in `dim_layout.dims()`:
    /// - If `Some(d) == region_axis`: the coordinate selects the
    ///   region rather than contributing to the in-tensor offset; the
    ///   coord MUST supply a value for `d`, otherwise this returns
    ///   `Err`.
    /// - Else: `off += byte_strides[i] * coord.get(d).unwrap_or(0)`.
    ///   A missing in-tensor coordinate defaults to `0`, treating the
    ///   axis as folded into the inner copy.
    ///
    /// Plus the region base: `regions[coord.get(region_axis).unwrap()]`.
    ///
    /// Errors when:
    /// - `region_axis = Some(d)` and `coord.get(d)` is `None`;
    /// - the region-axis coordinate is `>= regions.len()`;
    /// - any in-tensor coordinate is `>= size_of(axis)`.
    pub fn addr_of(&self, coord: &CoordByLabel) -> Result<usize> {
        let region_idx = match self.region_axis {
            Some(d) => coord
                .get(d)
                .ok_or_else(|| anyhow::anyhow!("addr_of: missing coord for region axis {d}"))?,
            None => 0,
        };
        if region_idx >= self.regions.len() {
            bail!(
                "addr_of: region index {region_idx} out of range (have {} regions)",
                self.regions.len(),
            );
        }
        let mut off = 0usize;
        let dims = self.dim_layout.dims();
        let sizes = self.dim_layout.sizes();
        for (i, &d) in dims.iter().enumerate() {
            if Some(d) == self.region_axis {
                continue;
            }
            let v = coord.get(d).unwrap_or(0);
            if v >= sizes[i] {
                bail!("addr_of: coord {d}={v} out of range (size {})", sizes[i],);
            }
            off += self.byte_strides.as_bytes()[i] * v;
        }
        Ok(self.regions[region_idx] + off)
    }

    pub fn dim_layout(&self) -> &KvDimLayout {
        &self.dim_layout
    }
    pub fn byte_strides(&self) -> &KvDimStrides {
        &self.byte_strides
    }
    pub fn region_axis(&self) -> Option<KvDim> {
        self.region_axis
    }
    pub fn regions(&self) -> &[usize] {
        &self.regions
    }
    pub fn elem_size(&self) -> usize {
        self.byte_strides.elem_size()
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CopyOp {
    pub src_addr: usize,
    pub dst_addr: usize,
    pub size: usize,
}

#[derive(Debug)]
pub enum CopyPlan {
    /// Coalesced contiguous transfers. Eventual mapping: N ×
    /// `cudaMemcpyAsync` / `memcpy` / one NIXL `XferDescList`.
    Direct(Vec<CopyOp>),
    /// Inner contiguous tail fell below `policy.min_inner_bytes` —
    /// defer to a kernel-side permutation. Carries the in-tensor
    /// permutation index map (`region_axis` excluded from both sides)
    /// alongside the original `(src_block_id, dst_block_id)` pairs so
    /// the kernel knows which rows to read and write.
    Transform {
        src: AnnotatedLayout,
        dst: AnnotatedLayout,
        block_pairs: Vec<(usize, usize)>,
        permutation: Vec<usize>,
    },
    /// Two-stage: src → intermediate → dst. Reserved for the case where
    /// staging through a different memory tier is faster than direct
    /// transform; not emitted by the prototype.
    Staged {
        first: Box<CopyPlan>,
        second: Box<CopyPlan>,
        intermediate: AnnotatedLayout,
    },
}

#[derive(Debug, Clone)]
pub struct CopyPolicy {
    /// Minimum contiguous bytes per emitted `CopyOp` to stay on the
    /// `Direct` path. Below this, fall back to `Transform`.
    pub min_inner_bytes: usize,
    /// Whether to coalesce adjacent `CopyOp`s after emission.
    pub coalesce: bool,
}

impl Default for CopyPolicy {
    fn default() -> Self {
        Self {
            min_inner_bytes: 4 * 1024,
            coalesce: true,
        }
    }
}

/// Plan a copy from `src` to `dst` over `block_pairs`, honouring
/// `policy`.
///
/// Each pair is `(src_block_id, dst_block_id)` — the planner copies
/// the same logical KV slab from src's row to dst's row.
pub fn plan_copy(
    src: &AnnotatedLayout,
    dst: &AnnotatedLayout,
    block_pairs: &[(usize, usize)],
    policy: &CopyPolicy,
) -> Result<CopyPlan> {
    // (a) Compatibility check: same multiset of (label, size) pairs.
    check_label_compatibility(src, dst)?;

    // Element sizes must agree — the planner emits byte-level copies,
    // and a dtype-width disagreement would silently miscount.
    if src.elem_size() != dst.elem_size() {
        bail!(
            "plan_copy: src.elem_size ({}) != dst.elem_size ({})",
            src.elem_size(),
            dst.elem_size(),
        );
    }

    // Validate block ids per pair against each side's Block-axis size.
    let src_block_size = src
        .dim_layout()
        .size_of(KvDim::Block)
        .ok_or_else(|| anyhow::anyhow!("plan_copy: src has no Block axis"))?;
    let dst_block_size = dst
        .dim_layout()
        .size_of(KvDim::Block)
        .ok_or_else(|| anyhow::anyhow!("plan_copy: dst has no Block axis"))?;
    for (i, &(s, d)) in block_pairs.iter().enumerate() {
        if s >= src_block_size {
            bail!(
                "plan_copy: block_pairs[{i}].0 = {s} out of range (src Block size {src_block_size})",
            );
        }
        if d >= dst_block_size {
            bail!(
                "plan_copy: block_pairs[{i}].1 = {d} out of range (dst Block size {dst_block_size})",
            );
        }
    }

    // (b) Three-way intersection.
    let matching_axes = matching_inner_suffix(src, dst);
    let (inner_bytes, accepted) = compute_inner_bytes(src, dst, &matching_axes)?;

    if inner_bytes < policy.min_inner_bytes {
        return Ok(CopyPlan::Transform {
            src: src.clone(),
            dst: dst.clone(),
            block_pairs: block_pairs.to_vec(),
            permutation: in_tensor_permutation(src, dst),
        });
    }

    // (c) Outer-iteration domain: labels not consumed by the
    //     *accepted* inner suffix, in the order they appear in
    //     src.dim_layout.dims(). Using `matching_axes` here would skip
    //     iteration of axes whose strides truncated the suffix below
    //     the matching label suffix — see compute_inner_bytes.
    let accepted_axes = &matching_axes[matching_axes.len() - accepted..];
    let inner_set: Vec<KvDim> = accepted_axes.iter().map(|(d, _)| *d).collect();
    let outer_axes: Vec<(KvDim, usize)> = src
        .dim_layout()
        .dims()
        .iter()
        .zip(src.dim_layout().sizes().iter())
        .filter(|(d, _)| !inner_set.contains(*d))
        .map(|(&d, &s)| (d, s))
        .collect();

    // (d) Emit triples by iterating the outer domain in src order.
    let mut ops: Vec<CopyOp> = Vec::new();
    let mut src_coord = CoordByLabel::new();
    let mut dst_coord = CoordByLabel::new();
    emit_outer(
        src,
        dst,
        &outer_axes,
        0,
        &mut src_coord,
        &mut dst_coord,
        block_pairs,
        inner_bytes,
        &mut ops,
    )?;

    // (e) Coalesce adjacent triples (optional).
    if policy.coalesce {
        ops = coalesce(ops);
    }

    Ok(CopyPlan::Direct(ops))
}

fn check_label_compatibility(src: &AnnotatedLayout, dst: &AnnotatedLayout) -> Result<()> {
    let src_dims = src.dim_layout().dims();
    let dst_dims = dst.dim_layout().dims();
    if src_dims.len() != dst_dims.len() {
        bail!(
            "plan_copy: rank mismatch (src {} vs dst {})",
            src_dims.len(),
            dst_dims.len(),
        );
    }
    // Every src label/size must appear in dst with the same size, and
    // vice versa. KvDimLayout already forbids duplicate labels (modulo
    // Outer), so a one-direction set check is sufficient.
    for (&d, &s) in src_dims.iter().zip(src.dim_layout().sizes().iter()) {
        let dst_size = dst.dim_layout().size_of(d).ok_or_else(|| {
            anyhow::anyhow!("plan_copy: src has label {d} but dst does not (dst dims {dst_dims:?})")
        })?;
        if dst_size != s {
            bail!("plan_copy: label {d} size disagrees — src={s}, dst={dst_size}",);
        }
    }
    for &d in dst_dims.iter() {
        if src.dim_layout().position(d).is_none() {
            bail!("plan_copy: dst has label {d} but src does not");
        }
    }
    Ok(())
}

/// Walk both layouts inside-out, recording the longest matching suffix
/// of `(label, size)` pairs. Stops at the first mismatch, OR at any
/// `KvDim::Block` axis (Block is always part of the outer iteration so
/// the planner can honour the caller's `blocks` list), OR at the
/// `region_axis` of either side (region partitioning ends contiguity).
fn matching_inner_suffix(src: &AnnotatedLayout, dst: &AnnotatedLayout) -> Vec<(KvDim, usize)> {
    let src_dims = src.dim_layout().dims();
    let src_sizes = src.dim_layout().sizes();
    let dst_dims = dst.dim_layout().dims();
    let dst_sizes = dst.dim_layout().sizes();
    let n = src_dims.len();
    let mut out = Vec::new();
    for k in 0..n {
        let i = n - 1 - k;
        let (sd, ss) = (src_dims[i], src_sizes[i]);
        let (dd, ds) = (dst_dims[i], dst_sizes[i]);
        if sd != dd || ss != ds {
            break;
        }
        if sd == KvDim::Block {
            break;
        }
        if Some(sd) == src.region_axis() || Some(sd) == dst.region_axis() {
            break;
        }
        out.push((sd, ss));
    }
    // out is innermost-first; reverse so it reads outermost-to-innermost.
    out.reverse();
    out
}

/// Compute `(inner_bytes, accepted_axis_count)` — the largest
/// `inner_bytes` ≤ matching-suffix bytes AND ≤ each side's effective
/// contiguous tail, plus the number of innermost matching axes that
/// fit into that byte budget.
///
/// Walks inside-out across `matching_axes`, accumulating bytes until
/// adding the next axis would exceed either side's tail. The caller
/// uses `accepted_axis_count` (not `matching_axes.len()`) to decide
/// which axes are folded into the inner copy versus iterated in the
/// outer loop. Without that distinction, stride-truncated cases would
/// silently skip iteration of axes that still need to vary.
fn compute_inner_bytes(
    src: &AnnotatedLayout,
    dst: &AnnotatedLayout,
    matching_axes: &[(KvDim, usize)],
) -> Result<(usize, usize)> {
    let elem = src.elem_size();
    let src_tail = effective_inner_tail_bytes(src)?;
    let dst_tail = effective_inner_tail_bytes(dst)?;
    let cap = src_tail.min(dst_tail);
    if elem > cap {
        return Ok((0, 0));
    }
    let mut bytes = elem;
    let mut accepted: usize = 0;
    for k in 0..matching_axes.len() {
        let (_, size) = matching_axes[matching_axes.len() - 1 - k];
        let next = bytes
            .checked_mul(size)
            .ok_or_else(|| anyhow::anyhow!("plan_copy: inner_bytes overflow"))?;
        if next > cap {
            break;
        }
        bytes = next;
        accepted = k + 1;
    }
    if accepted == 0 {
        Ok((0, 0))
    } else {
        Ok((bytes, accepted))
    }
}

/// Effective contiguous-tail bytes, capped at the region axis.
///
/// Even if the recorded byte strides happen to agree with row-major
/// past the region axis, the planner can't safely cross region
/// boundaries — each region has its own base address. The cap is
/// `elem_size * Π sizes of axes strictly inside the region axis`, i.e.
/// one full region's worth of bytes.
fn effective_inner_tail_bytes(layout: &AnnotatedLayout) -> Result<usize> {
    let raw = layout
        .byte_strides()
        .contiguous_tail_bytes(layout.dim_layout())?;
    let cap = match layout.region_axis() {
        Some(d) => {
            let pos = layout
                .dim_layout()
                .position(d)
                .expect("region_axis presence enforced by AnnotatedLayout::new");
            let sizes = layout.dim_layout().sizes();
            let mut bytes = layout.elem_size();
            for k in pos + 1..sizes.len() {
                bytes = bytes
                    .checked_mul(sizes[k])
                    .ok_or_else(|| anyhow::anyhow!("plan_copy: region cap overflow"))?;
            }
            bytes
        }
        None => usize::MAX,
    };
    Ok(raw.min(cap))
}

/// Index map from dst's in-tensor axes to src's in-tensor axes.
///
/// Both `src.region_axis()` and `dst.region_axis()` are excluded from
/// **both** sides before computing the map. That keeps the in-tensor
/// axis sets in agreement when one side has `Layer` as its region axis
/// while the other carries `Layer` in-tensor: the disagreement is
/// resolved by the per-region iteration in the outer loop, not by
/// folding `Layer` into the permutation vector.
fn in_tensor_permutation(src: &AnnotatedLayout, dst: &AnnotatedLayout) -> Vec<usize> {
    let exclude = |d: KvDim| Some(d) == src.region_axis() || Some(d) == dst.region_axis();
    let src_in: Vec<KvDim> = src
        .dim_layout()
        .dims()
        .iter()
        .copied()
        .filter(|&d| !exclude(d))
        .collect();
    let dst_in: Vec<KvDim> = dst
        .dim_layout()
        .dims()
        .iter()
        .copied()
        .filter(|&d| !exclude(d))
        .collect();
    dst_in
        .iter()
        .map(|d| {
            src_in
                .iter()
                .position(|s| s == d)
                .expect("plan_copy: in-tensor axis sets must agree after excluding region axes")
        })
        .collect()
}

/// Recursively walk the outer axes, threading the src/dst block
/// coordinates separately on the Block axis (since `block_pairs`
/// carries different ids for src and dst), and emit one `CopyOp` per
/// terminal cell.
///
/// Block-id range was validated up-front in `plan_copy`; `addr_of`
/// also re-checks every coordinate against axis size and surfaces any
/// remaining inconsistency as a `Result`, so this function bubbles
/// errors instead of asserting.
fn emit_outer(
    src: &AnnotatedLayout,
    dst: &AnnotatedLayout,
    outer_axes: &[(KvDim, usize)],
    depth: usize,
    src_coord: &mut CoordByLabel,
    dst_coord: &mut CoordByLabel,
    block_pairs: &[(usize, usize)],
    inner_bytes: usize,
    ops: &mut Vec<CopyOp>,
) -> Result<()> {
    if depth == outer_axes.len() {
        let src_addr = src.addr_of(src_coord)?;
        let dst_addr = dst.addr_of(dst_coord)?;
        ops.push(CopyOp {
            src_addr,
            dst_addr,
            size: inner_bytes,
        });
        return Ok(());
    }
    let (label, size) = outer_axes[depth];
    if label == KvDim::Block {
        for &(s, d) in block_pairs {
            src_coord.set_in_place(KvDim::Block, s);
            dst_coord.set_in_place(KvDim::Block, d);
            emit_outer(
                src,
                dst,
                outer_axes,
                depth + 1,
                src_coord,
                dst_coord,
                block_pairs,
                inner_bytes,
                ops,
            )?;
        }
    } else {
        for i in 0..size {
            src_coord.set_in_place(label, i);
            dst_coord.set_in_place(label, i);
            emit_outer(
                src,
                dst,
                outer_axes,
                depth + 1,
                src_coord,
                dst_coord,
                block_pairs,
                inner_bytes,
                ops,
            )?;
        }
    }
    Ok(())
}

/// Sort `ops` by `src_addr` and merge consecutive entries where
/// `(src_end, dst_end)` of op `k` equals `(src_start, dst_start)` of op
/// `k+1`. The result is the minimum-descriptor-count plan — critical
/// for NIXL.
fn coalesce(mut ops: Vec<CopyOp>) -> Vec<CopyOp> {
    if ops.len() <= 1 {
        return ops;
    }
    ops.sort_by_key(|o| o.src_addr);
    let mut out: Vec<CopyOp> = Vec::with_capacity(ops.len());
    for op in ops {
        match out.last_mut() {
            Some(last)
                if last.src_addr + last.size == op.src_addr
                    && last.dst_addr + last.size == op.dst_addr =>
            {
                last.size += op.size;
            }
            _ => out.push(op),
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Helper: build NHD-block-first AnnotatedLayout for per-layer
    /// registration. dim_layout = [Layer (region), Block, Outer, Page,
    /// HeadCount, HeadSize]; byte strides are row-major within each
    /// region; the Layer-axis stride is a sentinel (`1`) since
    /// addr_of skips it.
    ///
    /// Block-first (Outer at position 2, inner to Block) is chosen so
    /// the Outer axis can fold into the inner contiguous suffix —
    /// that's what makes test 1's
    /// `inner_bytes == outer * page * inner * dtype` hold. With
    /// FlashAttn-style outer-first ordering, Outer would be outermost
    /// and the inner suffix would stop at Page.
    fn nhd_per_layer(
        num_layers: usize,
        num_blocks: usize,
        outer: usize,
        page: usize,
        nh: usize,
        hd: usize,
        elem: usize,
        regions: Vec<usize>,
    ) -> AnnotatedLayout {
        assert_eq!(regions.len(), num_layers);
        let dim_layout = KvDimLayout::new(
            vec![
                KvDim::Layer,
                KvDim::Block,
                KvDim::Outer,
                KvDim::Page,
                KvDim::HeadCount,
                KvDim::HeadSize,
            ],
            vec![num_layers, num_blocks, outer, page, nh, hd],
        )
        .unwrap();
        // Within a region (Outer, Block, Page, HeadCount, HeadSize):
        // wait — region_axis is Layer at position 0, so axes 1.. are
        // in-tensor. Row-major byte strides over [Block, Outer, Page,
        // HeadCount, HeadSize]:
        let s_hs = elem;
        let s_hc = s_hs * hd;
        let s_pg = s_hc * nh;
        let s_ot = s_pg * page;
        let s_bk = s_ot * outer;
        // Layer-axis stride: sentinel (must be > 0).
        let s_la = 1;
        let strides =
            KvDimStrides::from_byte_strides(vec![s_la, s_bk, s_ot, s_pg, s_hc, s_hs], elem)
                .unwrap();
        AnnotatedLayout::new(regions, Some(KvDim::Layer), dim_layout, strides).unwrap()
    }

    /// Cross-layer fully-contiguous NHD: dim_layout = [Block, Layer,
    /// Outer, Page, HeadCount, HeadSize]; row-major end-to-end; one
    /// region (the single base address).
    fn nhd_cross_layer(
        num_blocks: usize,
        num_layers: usize,
        outer: usize,
        page: usize,
        nh: usize,
        hd: usize,
        elem: usize,
        base: usize,
    ) -> AnnotatedLayout {
        let dim_layout = KvDimLayout::new(
            vec![
                KvDim::Block,
                KvDim::Layer,
                KvDim::Outer,
                KvDim::Page,
                KvDim::HeadCount,
                KvDim::HeadSize,
            ],
            vec![num_blocks, num_layers, outer, page, nh, hd],
        )
        .unwrap();
        let s_hs = elem;
        let s_hc = s_hs * hd;
        let s_pg = s_hc * nh;
        let s_ot = s_pg * page;
        let s_la = s_ot * outer;
        let s_bk = s_la * num_layers;
        let strides =
            KvDimStrides::from_byte_strides(vec![s_bk, s_la, s_ot, s_pg, s_hc, s_hs], elem)
                .unwrap();
        AnnotatedLayout::new(vec![base], None, dim_layout, strides).unwrap()
    }

    /// HND per-layer: same as NHD per-layer except HeadCount and Page
    /// are swapped in dim_layout AND in byte strides. Each region's
    /// physical shape is [Block, Outer, HeadCount, Page, HeadSize].
    fn hnd_per_layer(
        num_layers: usize,
        num_blocks: usize,
        outer: usize,
        page: usize,
        nh: usize,
        hd: usize,
        elem: usize,
        regions: Vec<usize>,
    ) -> AnnotatedLayout {
        assert_eq!(regions.len(), num_layers);
        let dim_layout = KvDimLayout::new(
            vec![
                KvDim::Layer,
                KvDim::Block,
                KvDim::Outer,
                KvDim::HeadCount,
                KvDim::Page,
                KvDim::HeadSize,
            ],
            vec![num_layers, num_blocks, outer, nh, page, hd],
        )
        .unwrap();
        // Row-major byte strides over [Block, Outer, HeadCount, Page,
        // HeadSize] = [num_blocks, outer, nh, page, hd]:
        let s_hs = elem;
        let s_pg = s_hs * hd;
        let s_hc = s_pg * page;
        let s_ot = s_hc * nh;
        let s_bk = s_ot * outer;
        let s_la = 1;
        let strides =
            KvDimStrides::from_byte_strides(vec![s_la, s_bk, s_ot, s_hc, s_pg, s_hs], elem)
                .unwrap();
        AnnotatedLayout::new(regions, Some(KvDim::Layer), dim_layout, strides).unwrap()
    }

    /// Test 1: NHD per-layer ↔ NHD per-layer, same sizes. With
    /// `blocks = [0..16] ∪ [32..48]`, expect `CopyPlan::Direct` with
    /// `2 * num_layers` ops after coalescing — one per (layer,
    /// block-run). `inner_bytes == outer * page * inner * dtype`.
    #[test]
    fn nhd_per_layer_to_nhd_per_layer_direct_coalesces_block_runs() {
        let num_layers = 4;
        let num_blocks = 64;
        let (outer, page, nh, hd, elem) = (2, 16, 8, 128, 2);
        // Per-region size in bytes:
        let region_bytes = outer * num_blocks * page * nh * hd * elem;
        let src_regions: Vec<usize> = (0..num_layers)
            .map(|i| 0x100_0000 + i * region_bytes)
            .collect();
        let dst_regions: Vec<usize> = (0..num_layers)
            .map(|i| 0x800_0000 + i * region_bytes)
            .collect();
        let src = nhd_per_layer(
            num_layers,
            num_blocks,
            outer,
            page,
            nh,
            hd,
            elem,
            src_regions,
        );
        let dst = nhd_per_layer(
            num_layers,
            num_blocks,
            outer,
            page,
            nh,
            hd,
            elem,
            dst_regions,
        );

        // Two contiguous block runs: [0..16] and [32..48]; identity
        // src→dst block mapping.
        let mut block_ids: Vec<usize> = (0..16).collect();
        block_ids.extend(32..48);
        let block_pairs: Vec<(usize, usize)> = block_ids.iter().map(|&b| (b, b)).collect();

        let plan = plan_copy(&src, &dst, &block_pairs, &CopyPolicy::default()).unwrap();
        match plan {
            CopyPlan::Direct(ops) => {
                // inner_bytes == outer * page * nh * hd * elem
                let expected_inner = outer * page * nh * hd * elem;
                // After coalescing: one op per (layer, run), each
                // covering 16 blocks of `expected_inner` bytes. So
                // size == 16 * expected_inner per coalesced op.
                let coalesced_size = 16 * expected_inner;
                assert_eq!(
                    ops.len(),
                    2 * num_layers,
                    "expected 2*num_layers={} coalesced ops, got {}",
                    2 * num_layers,
                    ops.len()
                );
                for op in &ops {
                    assert_eq!(op.size, coalesced_size, "unexpected coalesced size: {op:?}");
                }
            }
            other => panic!("expected Direct, got {other:?}"),
        }
    }

    /// Test 2: NHD per-layer ↔ HND per-layer. inner contiguous tail
    /// collapses to `[HeadSize]` (256 B for fp16/hd=128), under
    /// `policy.min_inner_bytes = 4096` → expect `CopyPlan::Transform`
    /// with the in-tensor permutation `[0, 1, 3, 2, 4]`
    /// (Block, Outer, HeadCount→Page, Page→HeadCount, HeadSize).
    #[test]
    fn nhd_per_layer_to_hnd_per_layer_falls_back_to_transform() {
        let num_layers = 2;
        let num_blocks = 8;
        let (outer, page, nh, hd, elem) = (2, 16, 8, 128, 2);
        let region_bytes = outer * num_blocks * page * nh * hd * elem;
        let src_regions: Vec<usize> = (0..num_layers)
            .map(|i| 0x100_0000 + i * region_bytes)
            .collect();
        let dst_regions: Vec<usize> = (0..num_layers)
            .map(|i| 0x800_0000 + i * region_bytes)
            .collect();
        let src = nhd_per_layer(
            num_layers,
            num_blocks,
            outer,
            page,
            nh,
            hd,
            elem,
            src_regions,
        );
        let dst = hnd_per_layer(
            num_layers,
            num_blocks,
            outer,
            page,
            nh,
            hd,
            elem,
            dst_regions,
        );

        let block_pairs: Vec<(usize, usize)> = (0..num_blocks).map(|b| (b, b)).collect();
        let plan = plan_copy(&src, &dst, &block_pairs, &CopyPolicy::default()).unwrap();
        match plan {
            CopyPlan::Transform { permutation, .. } => {
                // src in-tensor axes: [Block, Outer, Page, HeadCount, HeadSize]
                // dst in-tensor axes: [Block, Outer, HeadCount, Page, HeadSize]
                // permutation[i] = position in src of dst[i]:
                //   dst[0]=Block     -> src[0]      -> 0
                //   dst[1]=Outer     -> src[1]      -> 1
                //   dst[2]=HeadCount -> src[3]      -> 3
                //   dst[3]=Page      -> src[2]      -> 2
                //   dst[4]=HeadSize  -> src[4]      -> 4
                assert_eq!(permutation, vec![0, 1, 3, 2, 4]);
            }
            other => panic!("expected Transform, got {other:?}"),
        }
    }

    /// Test 3: NHD per-layer ↔ NHD cross-layer. src has
    /// `region_axis = Some(Layer)`; dst is fully-contiguous with Layer
    /// in-tensor at position 1. Expect `CopyPlan::Direct` with one op
    /// per (layer, block) — Block can't be folded into the inner
    /// suffix, and Layer iteration is materialized in the outer loop
    /// (touch-up 4: per-region iteration when region axes disagree).
    #[test]
    fn nhd_per_layer_to_cross_layer_direct_per_region_iteration() {
        let num_layers = 4;
        let num_blocks = 8;
        let (outer, page, nh, hd, elem) = (2, 16, 8, 128, 2);
        let region_bytes = outer * num_blocks * page * nh * hd * elem;
        let src_regions: Vec<usize> = (0..num_layers)
            .map(|i| 0x100_0000 + i * region_bytes)
            .collect();
        let dst_base = 0x800_0000;
        let src = nhd_per_layer(
            num_layers,
            num_blocks,
            outer,
            page,
            nh,
            hd,
            elem,
            src_regions.clone(),
        );
        let dst = nhd_cross_layer(num_blocks, num_layers, outer, page, nh, hd, elem, dst_base);

        let block_pairs: Vec<(usize, usize)> = (0..num_blocks).map(|b| (b, b)).collect();
        let plan = plan_copy(&src, &dst, &block_pairs, &CopyPolicy::default()).unwrap();
        match plan {
            CopyPlan::Direct(ops) => {
                // matching suffix walk inside-out: HeadSize, HeadCount,
                // Page, Outer match in label and size — Outer is at the
                // same trailing position on both sides. Then src=Block
                // vs dst=Layer disagree → suffix = [Outer, Page,
                // HeadCount, HeadSize]. inner_bytes = elem * outer *
                // page * nh * hd = 65536.
                let expected_inner = outer * page * nh * hd * elem;
                // Before coalescing: 1 op per (layer, block) =
                // num_layers * num_blocks. Coalescing across blocks
                // requires consecutive (src, dst) addresses: src jumps
                // by region between layers (non-contiguous), dst is
                // [Block, Layer, ...] so consecutive blocks at the same
                // layer are NOT consecutive on dst (they differ by
                // dst.Block-stride which is num_layers*outer*...).
                // Result: no coalescing — each (layer, block) cell
                // stays a separate op.
                assert_eq!(ops.len(), num_layers * num_blocks);
                for op in &ops {
                    assert_eq!(op.size, expected_inner);
                }

                // Spot-check one address: (layer=2, block=3).
                let layer = 2usize;
                let block = 3usize;
                let coord = CoordByLabel::new()
                    .set(KvDim::Layer, layer)
                    .set(KvDim::Block, block);
                let expected_src = src.addr_of(&coord).unwrap();
                let expected_dst = dst.addr_of(&coord).unwrap();
                let found = ops
                    .iter()
                    .find(|o| o.src_addr == expected_src)
                    .expect("expected a CopyOp for (layer=2, block=3)");
                assert_eq!(found.dst_addr, expected_dst);
            }
            other => panic!("expected Direct, got {other:?}"),
        }
    }

    /// Test 4: layer-separate (Layer-as-region) ↔ fully-contiguous
    /// NHD with a different in-tensor ordering. Different orderings +
    /// different region partitionings → expect `CopyPlan::Transform`.
    /// Verifies that the prototype routes mismatched-ordering cases to
    /// the kernel-side path.
    #[test]
    fn layer_separate_to_fully_contiguous_different_order_routes_to_transform() {
        let num_layers = 2;
        let num_blocks = 4;
        let (outer, page, nh, hd, elem) = (2, 16, 8, 128, 2);
        let region_bytes = outer * num_blocks * page * nh * hd * elem;
        let src_regions: Vec<usize> = (0..num_layers)
            .map(|i| 0x100_0000 + i * region_bytes)
            .collect();
        // src is HND per-layer (Layer-as-region, HeadCount before Page in-tensor).
        let src = hnd_per_layer(
            num_layers,
            num_blocks,
            outer,
            page,
            nh,
            hd,
            elem,
            src_regions,
        );
        // dst is NHD cross-layer (no region-axis, Page before HeadCount).
        let dst_base = 0x800_0000;
        let dst = nhd_cross_layer(num_blocks, num_layers, outer, page, nh, hd, elem, dst_base);

        let block_pairs: Vec<(usize, usize)> = (0..num_blocks).map(|b| (b, b)).collect();
        let plan = plan_copy(&src, &dst, &block_pairs, &CopyPolicy::default()).unwrap();
        match plan {
            CopyPlan::Transform { permutation, .. } => {
                // src in-tensor: [Block, Outer, HeadCount, Page, HeadSize]
                // dst in-tensor: [Block, Outer, Page, HeadCount, HeadSize]
                //   dst[0]=Block     -> src[0] -> 0
                //   dst[1]=Outer     -> src[1] -> 1
                //   dst[2]=Page      -> src[3] -> 3
                //   dst[3]=HeadCount -> src[2] -> 2
                //   dst[4]=HeadSize  -> src[4] -> 4
                assert_eq!(permutation, vec![0, 1, 3, 2, 4]);
            }
            other => panic!("expected Transform, got {other:?}"),
        }
    }

    /// addr_of: spot-check a non-trivial coordinate. (layer=1,
    /// outer=1, block=3, page=5, head=2, head_size=10) on a per-layer
    /// NHD layout.
    #[test]
    fn addr_of_handcomputed() {
        let layout = nhd_per_layer(4, 8, 2, 16, 8, 128, 2, vec![0x1000, 0x2000, 0x3000, 0x4000]);
        let coord = CoordByLabel::new()
            .set(KvDim::Layer, 1)
            .set(KvDim::Block, 3)
            .set(KvDim::Outer, 1)
            .set(KvDim::Page, 5)
            .set(KvDim::HeadCount, 2)
            .set(KvDim::HeadSize, 10);
        // base for layer=1 = 0x2000.
        // strides (bytes) for [Block, Outer, Page, HeadCount, HeadSize]:
        //   Block = outer*page*nh*hd*elem = 2*16*8*128*2 = 65536
        //   Outer = page*nh*hd*elem       =   16*8*128*2 = 32768
        //   Page  = nh*hd*elem            =     8*128*2  =  2048
        //   HeadCount = hd*elem           =       128*2  =   256
        //   HeadSize = elem               =           2
        // off = 3*65536 + 1*32768 + 5*2048 + 2*256 + 10*2
        //     = 196608 + 32768 + 10240 + 512 + 20
        //     = 240148
        assert_eq!(layout.addr_of(&coord).unwrap(), 0x2000 + 240148);
    }

    /// `addr_of` errors on a missing region-axis coordinate — the
    /// region must be selected explicitly, not silently defaulted to
    /// region 0.
    #[test]
    fn addr_of_errors_on_missing_region_coord() {
        let layout = nhd_per_layer(4, 8, 2, 16, 8, 128, 2, vec![0x1000, 0x2000, 0x3000, 0x4000]);
        let coord = CoordByLabel::new().set(KvDim::Block, 0);
        assert!(layout.addr_of(&coord).is_err());
    }

    /// `addr_of` errors when an in-tensor coordinate exceeds the axis
    /// size.
    #[test]
    fn addr_of_errors_on_oob_coord() {
        let layout = nhd_per_layer(4, 8, 2, 16, 8, 128, 2, vec![0x1000, 0x2000, 0x3000, 0x4000]);
        let coord = CoordByLabel::new()
            .set(KvDim::Layer, 1)
            .set(KvDim::Block, 99); // 99 >= num_blocks (8)
        assert!(layout.addr_of(&coord).is_err());
    }

    /// `plan_copy` rejects block_pair ids that exceed the Block axis
    /// size on either side, returning an error rather than panicking.
    #[test]
    fn plan_copy_rejects_oob_block_id() {
        let layout = nhd_per_layer(2, 8, 2, 16, 8, 128, 2, vec![0x1000, 0x2000]);
        let block_pairs = vec![(0usize, 99usize)]; // dst id out of range.
        let res = plan_copy(&layout, &layout, &block_pairs, &CopyPolicy::default());
        assert!(res.is_err());
    }

    /// Item 2 (paired block ids): src and dst block ids may differ.
    /// Build a per-layer NHD ↔ NHD plan with `(0, 5)` and `(2, 1)`
    /// and assert the emitted addresses use the right side's id on
    /// each end.
    #[test]
    fn plan_copy_remaps_block_ids_per_side() {
        let num_layers = 2;
        let num_blocks = 8;
        let (outer, page, nh, hd, elem) = (2, 16, 8, 128, 2);
        let region_bytes = outer * num_blocks * page * nh * hd * elem;
        let src_regions: Vec<usize> = (0..num_layers)
            .map(|i| 0x100_0000 + i * region_bytes)
            .collect();
        let dst_regions: Vec<usize> = (0..num_layers)
            .map(|i| 0x800_0000 + i * region_bytes)
            .collect();
        let src = nhd_per_layer(
            num_layers,
            num_blocks,
            outer,
            page,
            nh,
            hd,
            elem,
            src_regions,
        );
        let dst = nhd_per_layer(
            num_layers,
            num_blocks,
            outer,
            page,
            nh,
            hd,
            elem,
            dst_regions,
        );

        let block_pairs = vec![(0usize, 5usize), (2usize, 1usize)];
        let plan = plan_copy(&src, &dst, &block_pairs, &CopyPolicy::default()).unwrap();
        let CopyPlan::Direct(ops) = plan else {
            panic!("expected Direct");
        };

        // Coalescing won't help — pairs (0,5) and (2,1) are not
        // adjacent on either side. So we get one op per (layer, pair)
        // = 2 * 2 = 4 ops. Verify each pair's src and dst use the
        // correct id.
        assert_eq!(ops.len(), num_layers * block_pairs.len());

        for layer in 0..num_layers {
            for &(sb, db) in &block_pairs {
                let src_coord = CoordByLabel::new()
                    .set(KvDim::Layer, layer)
                    .set(KvDim::Block, sb);
                let dst_coord = CoordByLabel::new()
                    .set(KvDim::Layer, layer)
                    .set(KvDim::Block, db);
                let want_src = src.addr_of(&src_coord).unwrap();
                let want_dst = dst.addr_of(&dst_coord).unwrap();
                let found = ops
                    .iter()
                    .find(|o| o.src_addr == want_src && o.dst_addr == want_dst)
                    .unwrap_or_else(|| {
                        panic!("missing op for layer={layer} src_block={sb} dst_block={db}");
                    });
                let expected_inner = outer * page * nh * hd * elem;
                assert_eq!(found.size, expected_inner);
            }
        }
    }

    /// Item 3 (accepted vs matching suffix): when stride caps truncate
    /// the contiguous tail below the matching label suffix, the
    /// planner must iterate the truncated axes in the outer loop.
    /// Setup: identical labels and sizes on src and dst, but src has
    /// a non-row-major Page stride that breaks contiguity above
    /// HeadCount. With `min_inner_bytes` set below the truncated
    /// tail size, the planner must still take the Direct path AND
    /// emit ops for every (Layer, Block, Outer, Page) cell — not just
    /// (Layer, Block) as the buggy implementation would.
    #[test]
    fn plan_copy_iterates_over_stride_truncated_axes() {
        // Layout: [Layer (region), Block, Outer, Page, HeadCount, HeadSize].
        // Sizes: 2 layers × 4 blocks × 2 outer × 4 page × 2 nh × 16 hd, fp16.
        let num_layers = 2usize;
        let num_blocks = 4usize;
        let (outer, page, nh, hd, elem) = (2usize, 4usize, 2usize, 16usize, 2usize);

        let dim_layout = KvDimLayout::new(
            vec![
                KvDim::Layer,
                KvDim::Block,
                KvDim::Outer,
                KvDim::Page,
                KvDim::HeadCount,
                KvDim::HeadSize,
            ],
            vec![num_layers, num_blocks, outer, page, nh, hd],
        )
        .unwrap();

        // Standard row-major-from-sizes byte strides for sanity:
        //   HeadSize  = 2
        //   HeadCount = 32  (= 16 * 2)
        //   Page      = 64  (= 32 * 2)
        //   Outer     = 256 (= 64 * 4)
        //   Block     = 512 (= 256 * 2)
        //   Layer     = sentinel
        // Here we INFLATE Page stride to 128 so that contiguity breaks
        // *between* HeadCount and Page (Page stride 128 ≠ HeadCount * 32 = 64).
        // Contiguous tail bytes = HeadCount * HeadSize * elem = 32 * 2 = ... wait.
        //   tail walk (inside-out):
        //     HeadSize stride = 2 = elem ✓ → tail = 2 * 16 = 32 bytes
        //     HeadCount stride 32 = 2*16 ✓ → tail = 32 * 2 = 64 bytes
        //     Page stride 128 ≠ 32*2=64 ✗ → stop. tail = 64 bytes.
        let s_hs: usize = elem;
        let s_hc: usize = s_hs * hd;
        let s_pg: usize = s_hc * nh * 2; // inflated by 2× → breaks contiguity here
        let s_ot: usize = s_pg * page;
        let s_bk: usize = s_ot * outer;
        let s_la: usize = 1;

        let strides =
            KvDimStrides::from_byte_strides(vec![s_la, s_bk, s_ot, s_pg, s_hc, s_hs], elem)
                .unwrap();
        let regions: Vec<usize> = (0..num_layers)
            .map(|i| 0x100_0000 + i * 0x10_0000)
            .collect();
        let src = AnnotatedLayout::new(
            regions.clone(),
            Some(KvDim::Layer),
            dim_layout.clone(),
            strides.clone(),
        )
        .unwrap();
        // Use an identical layout on dst so matching suffix is the
        // full tail [Outer, Page, HeadCount, HeadSize] (4 axes).
        let dst_regions: Vec<usize> = (0..num_layers)
            .map(|i| 0x800_0000 + i * 0x10_0000)
            .collect();
        let dst =
            AnnotatedLayout::new(dst_regions, Some(KvDim::Layer), dim_layout, strides).unwrap();

        // matching suffix: 4 axes (full Outer..HeadSize). compute_inner_bytes
        // truncates to 2 axes (HeadCount + HeadSize) because Page stride
        // breaks contiguity. inner_bytes = 64.
        // Set min_inner_bytes = 32 (≤ 64) so we stay on the Direct path.
        let policy = CopyPolicy {
            min_inner_bytes: 32,
            coalesce: false,
        };

        let block_pairs: Vec<(usize, usize)> = (0..num_blocks).map(|b| (b, b)).collect();
        let plan = plan_copy(&src, &dst, &block_pairs, &policy).unwrap();
        let CopyPlan::Direct(ops) = plan else {
            panic!("expected Direct (inner_bytes 64 ≥ min_inner_bytes 32)");
        };

        // Outer iteration must include Layer, Block, Outer, AND Page —
        // the buggy implementation would iterate only Layer + Block
        // and emit num_layers * num_blocks = 8 ops, missing Outer and
        // Page coverage. Correct implementation emits
        // num_layers * num_blocks * outer * page = 2*4*2*4 = 64 ops.
        let expected_ops = num_layers * num_blocks * outer * page;
        assert_eq!(
            ops.len(),
            expected_ops,
            "expected {expected_ops} ops covering Layer×Block×Outer×Page; got {}",
            ops.len()
        );
        // Each op carries the truncated inner_bytes = HeadCount * HeadSize * elem = 64.
        let expected_inner = nh * hd * elem;
        for op in &ops {
            assert_eq!(op.size, expected_inner);
        }
    }

    /// AnnotatedLayout::new rejects rank mismatches.
    #[test]
    fn annotated_layout_rejects_rank_mismatch() {
        let dim_layout = KvDimLayout::new(
            vec![KvDim::Block, KvDim::Page, KvDim::HeadSize],
            vec![16, 16, 128],
        )
        .unwrap();
        let strides = KvDimStrides::from_byte_strides(vec![1, 2], 2).unwrap();
        let res = AnnotatedLayout::new(vec![0x1000], None, dim_layout, strides);
        assert!(res.is_err());
    }

    /// AnnotatedLayout::new rejects regions whose count disagrees with
    /// the region-axis size.
    #[test]
    fn annotated_layout_rejects_region_count_mismatch() {
        let dim_layout = KvDimLayout::new(
            vec![KvDim::Layer, KvDim::Block, KvDim::HeadSize],
            vec![4, 16, 128],
        )
        .unwrap();
        let strides = KvDimStrides::from_byte_strides(vec![1, 256, 2], 2).unwrap();
        // region_axis = Some(Layer) with size 4 but only 3 regions.
        let res = AnnotatedLayout::new(
            vec![0x1000, 0x2000, 0x3000],
            Some(KvDim::Layer),
            dim_layout,
            strides,
        );
        assert!(res.is_err());
    }
}
