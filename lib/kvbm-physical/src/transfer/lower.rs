// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// Lowering output is consumed by `executor::planner` once the planner
// path is wired via `TransferOptions::use_planner` (PR-5). Suppress
// dead-code warnings until the executor reaches in.
#![allow(dead_code)]

//! Lowering [`CopyPlan`]s to executor candidates and projecting
//! [`PhysicalLayout`]s onto [`LayoutView`]s.
//!
//! This module is the bridge between the semantic planner
//! (`transfer::plan`) and the executor (`transfer::executor`):
//!
//! * [`physical_to_layout_view`] downcasts a [`PhysicalLayout`] to its
//!   concrete impl ([`FullyContiguousLayout`] /
//!   [`LayerSeparateLayout`]) and constructs the corresponding
//!   labelled [`LayoutView`]. The projection uses a skeletal axis
//!   labelling — Block, optional Layer (region or in-tensor), Outer,
//!   Page, Payload — collapsing the per-token NHD/HND substructure
//!   into a single opaque trailing [`KvDim::Payload`] axis sized to
//!   `inner_dim`. `Payload` (vs the more specific `HeadSize`) is the
//!   honest label here: PR-5 transfers don't reason about the
//!   per-token head structure, only about contiguous byte runs. PR-6
//!   will project true `HeadCount`/`HeadSize` axes when the kernel
//!   catalog needs to distinguish NHD from HND.
//!
//! * [`Candidate`] is the executor-side representation of a lowered
//!   plan. PR-5 emits only [`Candidate::DirectDma`] from
//!   [`CopyPlan::Direct`]; [`CopyPlan::Transform`] is rejected as
//!   "kernel catalog not wired yet" (PR-6 lands the kernel
//!   instantiations); [`CopyPlan::Staged`] is reserved.
//!
//! * [`lower_to_candidates`] performs the lowering. [`select_candidate`]
//!   picks one for execution.

use anyhow::{Result, bail};
use kvbm_common::{KvDim, KvDimLayout, KvDimStrides};

use crate::layout::{
    BlockDimension, FullyContiguousLayout, KvBlockLayout, LayerSeparateLayout, Layout, LayoutView,
    PhysicalLayout,
};
use crate::transfer::plan::{CopyOp, CopyPlan};

/// Project a [`PhysicalLayout`] to a [`LayoutView`] for the planner.
///
/// Thin wrapper around [`layout_to_view`] for callers that already have
/// a `PhysicalLayout` in hand.
pub(crate) fn physical_to_layout_view(physical: &PhysicalLayout) -> Result<LayoutView> {
    layout_to_view(physical.layout().as_ref())
}

/// Project a bare `&dyn Layout` to a [`LayoutView`].
///
/// Only [`FullyContiguousLayout`] and [`LayerSeparateLayout`] are
/// supported; other layout kinds return an error. The projection is
/// skeletal: the per-token NHD/HND substructure inside `inner_dim` is
/// collapsed into a single opaque [`KvDim::Payload`] trailing axis.
/// PR-6 will refine this when the kernel catalog needs to distinguish
/// the substructure.
///
/// **Design note (deferred from PR-5 review).** `Layout::as_any()`
/// downcasting is brittle: adding a new concrete `Layout` impl
/// requires editing this dispatch and unsupported layouts fail late
/// here rather than at the trait boundary. A follow-up PR can replace
/// this with a `Layout::layout_view(&self) -> Result<LayoutView>`
/// trait method (or a sibling projection trait) so each layout owns
/// its projection.
pub(crate) fn layout_to_view(layout: &dyn Layout) -> Result<LayoutView> {
    let cfg = layout.config();
    if let Some(fc) = layout.as_any().downcast_ref::<FullyContiguousLayout>() {
        return fc_to_view(fc, cfg);
    }
    if let Some(ls) = layout.as_any().downcast_ref::<LayerSeparateLayout>() {
        return ls_to_view(ls, cfg);
    }
    bail!(
        "layout_to_view: unsupported concrete layout type \
         (block_layout = {:?})",
        layout.block_layout()
    );
}

/// Resolve `(num_heads, head_dim)` from a config, requiring `num_heads`
/// to be set when the projection has been asked to honour a known
/// `KvBlockLayout` (the kernel catalog distinguishes NHD/HND/Universal
/// by axis order, which can only be expressed once `inner_dim` is split).
///
/// Validation lives at the projection site, not at `LayoutConfig::build()`,
/// so legacy callers that don't enable `use_planner = true` are not
/// disturbed.
fn resolve_head_dims(
    cfg: &crate::layout::LayoutConfig,
    block_layout: KvBlockLayout,
) -> Result<(usize, usize)> {
    let nh = cfg.num_heads.ok_or_else(|| {
        anyhow::anyhow!(
            "physical_to_layout_view: cfg.num_heads is required when KvBlockLayout = \
             {block_layout:?} (set num_heads on LayoutConfig)"
        )
    })?;
    if !cfg.inner_dim.is_multiple_of(nh) {
        bail!(
            "physical_to_layout_view: inner_dim ({}) is not divisible by num_heads ({}) \
             — cannot split into HeadCount × HeadSize",
            cfg.inner_dim,
            nh,
        );
    }
    Ok((nh, cfg.inner_dim / nh))
}

/// FC: single allocation, no region axis.
///
/// Axis order depends on `KvBlockLayout`:
/// - `OperationalNHD`: `[Block, Layer, Outer, Page, HeadCount, HeadSize]`
/// - `OperationalHND`: `[Block, Layer, Outer, HeadCount, Page, HeadSize]`
/// - `UniversalTP`:    `[Block, HeadCount, Layer, Outer, Page, HeadSize]`
/// - `UniversalPP`:    `[Block, Layer, HeadCount, Outer, Page, HeadSize]`
///
/// The four orderings are what makes the kernel catalog's
/// signature-keyed dispatch work — NHD vs HND vs UniversalTP/PP differ
/// only in where `HeadCount` sits relative to the inner axes.
fn fc_to_view(fc: &FullyContiguousLayout, cfg: &crate::layout::LayoutConfig) -> Result<LayoutView> {
    let block_layout = fc.kv_block_layout();
    if matches!(block_layout, KvBlockLayout::Unknown) {
        bail!("physical_to_layout_view: FullyContiguousLayout has Unknown block layout");
    }
    let (nh, hd) = resolve_head_dims(cfg, block_layout)?;
    let elem = cfg.dtype_width_bytes;
    let buffers = fc.memory_regions();
    if buffers.len() != 1 {
        bail!(
            "physical_to_layout_view: FullyContiguousLayout expects 1 Buffer, got {}",
            buffers.len()
        );
    }
    let regions = vec![buffers[0].addr()];

    let (dims, sizes, byte_strides) = match block_layout {
        KvBlockLayout::OperationalNHD => {
            // [Block, Layer, Outer, Page, HeadCount, HeadSize]
            let s_hs = elem;
            let s_hc = s_hs * hd;
            let s_pg = s_hc * nh;
            let s_ot = s_pg * cfg.page_size;
            let s_la = s_ot * cfg.outer_dim;
            let s_bk = s_la * cfg.num_layers;
            (
                vec![
                    KvDim::Block,
                    KvDim::Layer,
                    KvDim::Outer,
                    KvDim::Page,
                    KvDim::HeadCount,
                    KvDim::HeadSize,
                ],
                vec![
                    cfg.num_blocks,
                    cfg.num_layers,
                    cfg.outer_dim,
                    cfg.page_size,
                    nh,
                    hd,
                ],
                vec![s_bk, s_la, s_ot, s_pg, s_hc, s_hs],
            )
        }
        KvBlockLayout::OperationalHND => {
            // [Block, Layer, Outer, HeadCount, Page, HeadSize]
            let s_hs = elem;
            let s_pg = s_hs * hd;
            let s_hc = s_pg * cfg.page_size;
            let s_ot = s_hc * nh;
            let s_la = s_ot * cfg.outer_dim;
            let s_bk = s_la * cfg.num_layers;
            (
                vec![
                    KvDim::Block,
                    KvDim::Layer,
                    KvDim::Outer,
                    KvDim::HeadCount,
                    KvDim::Page,
                    KvDim::HeadSize,
                ],
                vec![
                    cfg.num_blocks,
                    cfg.num_layers,
                    cfg.outer_dim,
                    nh,
                    cfg.page_size,
                    hd,
                ],
                vec![s_bk, s_la, s_ot, s_hc, s_pg, s_hs],
            )
        }
        KvBlockLayout::UniversalTP => {
            // [Block, HeadCount, Layer, Outer, Page, HeadSize] — per-block
            // shape `[nh, nl, no, nt, hd]` (matches universal_from_block /
            // block_from_universal kernel layout in kvbm-kernels).
            let s_hs = elem;
            let s_pg = s_hs * hd;
            let s_ot = s_pg * cfg.page_size;
            let s_la = s_ot * cfg.outer_dim;
            let s_hc = s_la * cfg.num_layers;
            let s_bk = s_hc * nh;
            (
                vec![
                    KvDim::Block,
                    KvDim::HeadCount,
                    KvDim::Layer,
                    KvDim::Outer,
                    KvDim::Page,
                    KvDim::HeadSize,
                ],
                vec![
                    cfg.num_blocks,
                    nh,
                    cfg.num_layers,
                    cfg.outer_dim,
                    cfg.page_size,
                    hd,
                ],
                vec![s_bk, s_hc, s_la, s_ot, s_pg, s_hs],
            )
        }
        KvBlockLayout::UniversalPP => {
            // [Block, Layer, HeadCount, Outer, Page, HeadSize].
            let s_hs = elem;
            let s_pg = s_hs * hd;
            let s_ot = s_pg * cfg.page_size;
            let s_hc = s_ot * cfg.outer_dim;
            let s_la = s_hc * nh;
            let s_bk = s_la * cfg.num_layers;
            (
                vec![
                    KvDim::Block,
                    KvDim::Layer,
                    KvDim::HeadCount,
                    KvDim::Outer,
                    KvDim::Page,
                    KvDim::HeadSize,
                ],
                vec![
                    cfg.num_blocks,
                    cfg.num_layers,
                    nh,
                    cfg.outer_dim,
                    cfg.page_size,
                    hd,
                ],
                vec![s_bk, s_la, s_hc, s_ot, s_pg, s_hs],
            )
        }
        KvBlockLayout::Custom(_) => bail!(
            "physical_to_layout_view: KvBlockLayout::Custom is not supported by the \
             planner-driven path"
        ),
        KvBlockLayout::Unknown => unreachable!("Unknown rejected above"),
    };

    let layout = KvDimLayout::new(dims, sizes)?;
    let strides = KvDimStrides::from_byte_strides(byte_strides, elem)?;
    LayoutView::full(layout, strides, regions, None)
}

/// LS: per-layer regions, `region_axis = Some(Layer)`. Universal
/// layouts cannot be LS — `LayerSeparateLayout::build` rejects them
/// — so only operational variants are projected here.
///
/// In-region axis order depends on `BlockDimension` and `KvBlockLayout`:
/// - `BlockIsFirstDim` + NHD: `[Layer, Block, Outer, Page, HeadCount, HeadSize]`
/// - `BlockIsFirstDim` + HND: `[Layer, Block, Outer, HeadCount, Page, HeadSize]`
/// - `BlockIsSecondDim` + NHD: `[Layer, Outer, Block, Page, HeadCount, HeadSize]`
/// - `BlockIsSecondDim` + HND: `[Layer, Outer, Block, HeadCount, Page, HeadSize]`
fn ls_to_view(ls: &LayerSeparateLayout, cfg: &crate::layout::LayoutConfig) -> Result<LayoutView> {
    let block_layout = ls.kv_block_layout();
    if matches!(block_layout, KvBlockLayout::Unknown) {
        bail!("physical_to_layout_view: LayerSeparateLayout has Unknown block layout");
    }
    let (nh, hd) = resolve_head_dims(cfg, block_layout)?;
    let elem = cfg.dtype_width_bytes;

    // Per-region inner-axis strides (NHD vs HND).
    let (s_hs, s_pre_outer, inner_axes_after_outer): (
        usize,
        usize,
        (Vec<KvDim>, Vec<usize>, Vec<usize>),
    ) = match block_layout {
        KvBlockLayout::OperationalNHD => {
            // Inside Outer: [Page, HeadCount, HeadSize]
            let s_hs = elem;
            let s_hc = s_hs * hd;
            let s_pg = s_hc * nh;
            let s_pre_outer = s_pg * cfg.page_size;
            let inner = (
                vec![KvDim::Page, KvDim::HeadCount, KvDim::HeadSize],
                vec![cfg.page_size, nh, hd],
                vec![s_pg, s_hc, s_hs],
            );
            (s_hs, s_pre_outer, inner)
        }
        KvBlockLayout::OperationalHND => {
            // Inside Outer: [HeadCount, Page, HeadSize]
            let s_hs = elem;
            let s_pg = s_hs * hd;
            let s_hc = s_pg * cfg.page_size;
            let s_pre_outer = s_hc * nh;
            let inner = (
                vec![KvDim::HeadCount, KvDim::Page, KvDim::HeadSize],
                vec![nh, cfg.page_size, hd],
                vec![s_hc, s_pg, s_hs],
            );
            (s_hs, s_pre_outer, inner)
        }
        KvBlockLayout::UniversalTP | KvBlockLayout::UniversalPP => bail!(
            "physical_to_layout_view: LayerSeparateLayout cannot represent universal \
             KvBlockLayout (LayerSeparateLayout::build rejects these)"
        ),
        KvBlockLayout::Custom(_) => bail!(
            "physical_to_layout_view: KvBlockLayout::Custom is not supported by the \
             planner-driven path"
        ),
        KvBlockLayout::Unknown => unreachable!("Unknown rejected above"),
    };
    let _ = s_hs;

    // Per-region prefix order depends on which axis is outermost in
    // the region.
    let region_size = s_pre_outer * cfg.outer_dim;
    let (mut dims, mut sizes, mut byte_strides) = match ls.block_dim() {
        BlockDimension::BlockIsFirstDim => {
            let s_ot = s_pre_outer;
            let s_bk = s_ot * cfg.outer_dim;
            let s_la = s_bk * cfg.num_blocks; // Layer-axis sentinel
            let _ = region_size;
            (
                vec![KvDim::Layer, KvDim::Block, KvDim::Outer],
                vec![cfg.num_layers, cfg.num_blocks, cfg.outer_dim],
                vec![s_la, s_bk, s_ot],
            )
        }
        BlockDimension::BlockIsSecondDim => {
            let s_bk = s_pre_outer;
            let s_ot = s_bk * cfg.num_blocks;
            let s_la = s_ot * cfg.outer_dim;
            (
                vec![KvDim::Layer, KvDim::Outer, KvDim::Block],
                vec![cfg.num_layers, cfg.outer_dim, cfg.num_blocks],
                vec![s_la, s_ot, s_bk],
            )
        }
    };
    let (mut inner_dims, mut inner_sizes, mut inner_strides) = inner_axes_after_outer;
    dims.append(&mut inner_dims);
    sizes.append(&mut inner_sizes);
    byte_strides.append(&mut inner_strides);

    let layout = KvDimLayout::new(dims, sizes)?;
    let strides = KvDimStrides::from_byte_strides(byte_strides, elem)?;
    let regions: Vec<usize> = ls.memory_regions().iter().map(|b| b.addr()).collect();
    if regions.len() != cfg.num_layers {
        bail!(
            "physical_to_layout_view: LayerSeparateLayout has {} regions but {} layers",
            regions.len(),
            cfg.num_layers
        );
    }
    LayoutView::full(layout, strides, regions, Some(KvDim::Layer))
}

/// Executor candidate: one concrete way to perform the planned
/// transfer.
///
/// PR-6.1 emits [`Candidate::DirectDma`] from [`CopyPlan::Direct`] via
/// `lower_to_candidates`; [`Candidate::TransformKernel`] is constructed
/// directly by `executor::planner::plan_and_lower` once the catalog
/// resolves a kernel for `CopyPlan::Transform`. Other variants are
/// reserved for follow-up PRs:
/// * [`Candidate::BatchedDma`] groups ops by region/stream for
///   coalesced launches; arrives when stream-aware grouping is wired.
/// * [`Candidate::Staged`] handles two-hop transfers when direct
///   transform is impossible.
#[derive(Debug, Clone)]
pub(crate) enum Candidate {
    DirectDma {
        ops: Vec<CopyOp>,
    },
    BatchedDma {
        groups: Vec<Vec<CopyOp>>,
    },
    /// PR-6.1: kernel-driven transform candidate. `invocation` carries
    /// the resolved [`KernelKind`] + launch params; the executor reads
    /// per-side region pointers from the original `PhysicalLayout`s
    /// at dispatch.
    #[cfg(feature = "permute_kernels")]
    TransformKernel {
        invocation: crate::transfer::kernel_catalog::KernelInvocation,
    },
    Staged {/* spec lands later */},
}

/// Lower a [`CopyPlan`] to a vector of executor candidates.
///
/// PR-6.1 surface:
/// * [`CopyPlan::Direct`] yields a single [`Candidate::DirectDma`].
/// * [`CopyPlan::Transform`] errors here — Transform lowering happens
///   in `executor::planner::plan_and_lower`, which has the original
///   `PhysicalLayout`s needed for catalog lookup. (`AnnotatedLayout`
///   is purely structural and doesn't carry `KvBlockLayout`.)
/// * [`CopyPlan::Staged`] is reserved.
pub(crate) fn lower_to_candidates(plan: CopyPlan) -> Result<Vec<Candidate>> {
    match plan {
        CopyPlan::Direct(ops) => Ok(vec![Candidate::DirectDma { ops }]),
        CopyPlan::Transform { .. } => bail!(
            "lower_to_candidates: CopyPlan::Transform is lowered by \
             executor::planner::plan_and_lower (catalog lookup needs \
             KvBlockLayout, which AnnotatedLayout does not carry)"
        ),
        CopyPlan::Staged { .. } => bail!(
            "lower_to_candidates: CopyPlan::Staged is reserved and not \
             yet emitted by the prototype"
        ),
    }
}

/// Select a candidate to execute.
///
/// PR-5 picks the first [`Candidate::DirectDma`] in the list; future
/// selection logic (PR-7) will key off route, dtype, descriptor count,
/// and capability metadata. Errors when no executable candidate
/// exists.
pub(crate) fn select_candidate(candidates: &[Candidate]) -> Result<&Candidate> {
    candidates
        .iter()
        .find(|c| matches!(c, Candidate::DirectDma { .. }))
        .ok_or_else(|| {
            anyhow::anyhow!(
                "select_candidate: no executable Candidate::DirectDma in {} candidates",
                candidates.len()
            )
        })
}

#[cfg(all(test, feature = "testing-kvbm"))]
mod tests {
    use super::*;
    use crate::transfer::plan::AnnotatedLayout;

    #[test]
    fn lower_direct_to_dma_candidate() {
        let plan = CopyPlan::Direct(vec![CopyOp {
            src_addr: 0x1000,
            dst_addr: 0x2000,
            size: 64,
        }]);
        let cands = lower_to_candidates(plan).unwrap();
        assert_eq!(cands.len(), 1);
        assert!(matches!(cands[0], Candidate::DirectDma { .. }));
    }

    #[test]
    fn lower_transform_errors() {
        let layout = KvDimLayout::new(
            vec![KvDim::Block, KvDim::Page, KvDim::HeadSize],
            vec![4, 16, 128],
        )
        .unwrap();
        let strides = KvDimStrides::from_byte_strides(vec![16 * 128 * 2, 128 * 2, 2], 2).unwrap();
        let al = AnnotatedLayout::new(vec![0x1000], None, layout, strides).unwrap();
        let plan = CopyPlan::Transform {
            src: al.clone(),
            dst: al,
            block_pairs: vec![(0, 0)],
            permutation: vec![0, 1],
        };
        assert!(lower_to_candidates(plan).is_err());
    }

    #[test]
    fn select_picks_direct_dma() {
        // Use Staged as a non-DirectDma placeholder — this test pins
        // `select_candidate`'s preference for DirectDma when present.
        // (PR-7 will replace this with multi-candidate scoring.)
        let cands = vec![
            Candidate::Staged {},
            Candidate::DirectDma {
                ops: vec![CopyOp {
                    src_addr: 0,
                    dst_addr: 0,
                    size: 0,
                }],
            },
        ];
        let picked = select_candidate(&cands).unwrap();
        assert!(matches!(picked, Candidate::DirectDma { .. }));
    }

    #[test]
    fn select_no_direct_errors() {
        let cands = vec![Candidate::Staged {}];
        assert!(select_candidate(&cands).is_err());
    }

    /// FullyContiguousLayout (NHD): verify `addr_of` on the projected
    /// view agrees with `Layout::memory_region` + within-region
    /// stride math for a representative coord. PR-6.1 splits the
    /// inner axis into HeadCount × HeadSize, so the test now uses
    /// `(HeadCount, h_idx) + (HeadSize, hd_idx)` pairs.
    #[test]
    fn layout_to_view_fc_round_trips_with_memory_region() {
        use crate::layout::FullyContiguousLayout;
        use crate::layout::tests::MockMemory;
        use crate::layout::{Buffer, KvBlockLayout, Layout, LayoutConfig};

        let cfg = LayoutConfig::builder()
            .num_blocks(4)
            .num_layers(2)
            .outer_dim(2)
            .page_size(8)
            .inner_dim(64)
            .dtype_width_bytes(2)
            .num_heads(Some(8))
            .build()
            .unwrap();
        let mem = Buffer::from_arc(MockMemory::new(0x1_0000, cfg.required_bytes()));
        let fc = FullyContiguousLayout::builder()
            .config(cfg.clone())
            .memory(mem)
            .kv_block_layout(KvBlockLayout::OperationalNHD)
            .build()
            .unwrap();

        let view = layout_to_view(&fc as &dyn Layout).unwrap();
        let al = AnnotatedLayout::from_view(&view).unwrap();

        let block_id = 2usize;
        let layer_id = 1usize;
        let outer_id = 1usize;
        let page = 5usize;
        let h_idx = 3usize;
        let hd_idx = 5usize;
        let head_dim = cfg.head_dim().unwrap();
        let coord = kvbm_common::CoordByLabel::new()
            .with(KvDim::Block, block_id)
            .with(KvDim::Layer, layer_id)
            .with(KvDim::Outer, outer_id)
            .with(KvDim::Page, page)
            .with(KvDim::HeadCount, h_idx)
            .with(KvDim::HeadSize, hd_idx);

        let view_addr = al.addr_of(&coord).unwrap();
        let region = fc.memory_region(block_id, layer_id, outer_id).unwrap();
        // NHD inner: [Page, HeadCount, HeadSize] within (block, layer, outer).
        let expected = region.addr()
            + (page * cfg.inner_dim + h_idx * head_dim + hd_idx) * cfg.dtype_width_bytes;
        assert_eq!(view_addr, expected);
    }

    /// LayerSeparateLayout (BlockIsFirstDim, NHD): per-region in-tensor
    /// axis order is `[Block, Outer, Page, HeadCount, HeadSize]`.
    #[test]
    fn layout_to_view_ls_first_dim_round_trips_with_memory_region() {
        use crate::layout::LayerSeparateLayout;
        use crate::layout::tests::MockMemory;
        use crate::layout::{BlockDimension, Buffer, InnerShape, Layout, LayoutConfig};

        let cfg = LayoutConfig::builder()
            .num_blocks(4)
            .num_layers(2)
            .outer_dim(2)
            .page_size(8)
            .inner_dim(64)
            .dtype_width_bytes(2)
            .num_heads(Some(8))
            .build()
            .unwrap();
        let per_layer =
            cfg.num_blocks * cfg.outer_dim * cfg.page_size * cfg.inner_dim * cfg.dtype_width_bytes;
        let memory: Vec<Buffer> = (0..cfg.num_layers)
            .map(|i| Buffer::from_arc(MockMemory::new(0x1_0000_0000 + i * 0x10_0000, per_layer)))
            .collect();
        let ls = LayerSeparateLayout::builder()
            .config(cfg.clone())
            .memory(memory)
            .block_dim(BlockDimension::BlockIsFirstDim)
            .inner_shape(InnerShape::NHD)
            .build()
            .unwrap();

        let view = layout_to_view(&ls as &dyn Layout).unwrap();
        let al = AnnotatedLayout::from_view(&view).unwrap();

        let block_id = 1usize;
        let layer_id = 1usize;
        let outer_id = 1usize;
        let page = 3usize;
        let h_idx = 2usize;
        let hd_idx = 4usize;
        let head_dim = cfg.head_dim().unwrap();
        let coord = kvbm_common::CoordByLabel::new()
            .with(KvDim::Block, block_id)
            .with(KvDim::Layer, layer_id)
            .with(KvDim::Outer, outer_id)
            .with(KvDim::Page, page)
            .with(KvDim::HeadCount, h_idx)
            .with(KvDim::HeadSize, hd_idx);

        let view_addr = al.addr_of(&coord).unwrap();
        let region = ls.memory_region(block_id, layer_id, outer_id).unwrap();
        let expected = region.addr()
            + (page * cfg.inner_dim + h_idx * head_dim + hd_idx) * cfg.dtype_width_bytes;
        assert_eq!(view_addr, expected);
    }

    /// LayerSeparateLayout (BlockIsSecondDim, HND): per-region inner
    /// shape `[Outer, Block, HeadCount, Page, HeadSize]`.
    #[test]
    fn layout_to_view_ls_second_dim_round_trips_with_memory_region() {
        use crate::layout::LayerSeparateLayout;
        use crate::layout::tests::MockMemory;
        use crate::layout::{BlockDimension, Buffer, InnerShape, Layout, LayoutConfig};

        let cfg = LayoutConfig::builder()
            .num_blocks(4)
            .num_layers(2)
            .outer_dim(2)
            .page_size(8)
            .inner_dim(64)
            .dtype_width_bytes(2)
            .num_heads(Some(8))
            .build()
            .unwrap();
        let per_layer =
            cfg.num_blocks * cfg.outer_dim * cfg.page_size * cfg.inner_dim * cfg.dtype_width_bytes;
        let memory: Vec<Buffer> = (0..cfg.num_layers)
            .map(|i| Buffer::from_arc(MockMemory::new(0x2_0000_0000 + i * 0x10_0000, per_layer)))
            .collect();
        let ls = LayerSeparateLayout::builder()
            .config(cfg.clone())
            .memory(memory)
            .block_dim(BlockDimension::BlockIsSecondDim)
            .inner_shape(InnerShape::HND)
            .build()
            .unwrap();

        let view = layout_to_view(&ls as &dyn Layout).unwrap();
        let al = AnnotatedLayout::from_view(&view).unwrap();

        let block_id = 2usize;
        let layer_id = 1usize;
        let outer_id = 0usize;
        let page = 6usize;
        let h_idx = 5usize;
        let hd_idx = 1usize;
        let head_dim = cfg.head_dim().unwrap();
        let coord = kvbm_common::CoordByLabel::new()
            .with(KvDim::Block, block_id)
            .with(KvDim::Layer, layer_id)
            .with(KvDim::Outer, outer_id)
            .with(KvDim::Page, page)
            .with(KvDim::HeadCount, h_idx)
            .with(KvDim::HeadSize, hd_idx);

        let view_addr = al.addr_of(&coord).unwrap();
        let region = ls.memory_region(block_id, layer_id, outer_id).unwrap();
        // HND inner per-region (Outer-major, then Block, then HND):
        // expected = region + (block_id * inner_dim) [outer×block] +
        //            (h_idx * page_size * head_dim) +
        //            (page * head_dim) +
        //            hd_idx, all × elem.
        // But region addr already encodes (block, layer, outer); the
        // layout-aware projection uses the full inner stride table, so
        // we recompute via the same components.
        let elem = cfg.dtype_width_bytes;
        let expected =
            region.addr() + (h_idx * cfg.page_size * head_dim + page * head_dim + hd_idx) * elem;
        assert_eq!(view_addr, expected);
    }

    /// Building a layout with `KvBlockLayout::Unknown` is rejected by
    /// `layout_to_view` — the projection cannot honestly emit
    /// `Direct` ops without knowing the per-token substructure.
    #[test]
    fn layout_to_view_rejects_unknown_block_layout() {
        use crate::layout::FullyContiguousLayout;
        use crate::layout::tests::MockMemory;
        use crate::layout::{Buffer, KvBlockLayout, Layout, LayoutConfig};

        let cfg = LayoutConfig::builder()
            .num_blocks(2)
            .num_layers(1)
            .outer_dim(1)
            .page_size(4)
            .inner_dim(8)
            .dtype_width_bytes(2)
            .build()
            .unwrap();
        let mem = Buffer::from_arc(MockMemory::new(0x1_0000, cfg.required_bytes()));
        let fc = FullyContiguousLayout::builder()
            .config(cfg)
            .memory(mem)
            .kv_block_layout(KvBlockLayout::Unknown)
            .build()
            .unwrap();
        assert!(layout_to_view(&fc as &dyn Layout).is_err());
    }

    /// PR-6.1 projection requires `cfg.num_heads.is_some()` whenever
    /// the layout has a known `KvBlockLayout` (the catalog
    /// distinguishes NHD / HND / Universal by axis order, which can
    /// only be expressed once `inner_dim` is split into HeadCount
    /// and HeadSize). Validation lives at the projection site, not
    /// at `LayoutConfig::build()`, so legacy callers that don't
    /// enable `use_planner = true` are unaffected.
    #[test]
    fn layout_to_view_requires_num_heads_when_block_layout_is_known() {
        use crate::layout::FullyContiguousLayout;
        use crate::layout::tests::MockMemory;
        use crate::layout::{Buffer, KvBlockLayout, Layout, LayoutConfig};

        let cfg = LayoutConfig::builder()
            .num_blocks(2)
            .num_layers(1)
            .outer_dim(1)
            .page_size(4)
            .inner_dim(8)
            .dtype_width_bytes(2)
            // Note: no num_heads(...) — defaults to None.
            .build()
            .unwrap();
        let mem = Buffer::from_arc(MockMemory::new(0x1_0000, cfg.required_bytes()));
        let fc = FullyContiguousLayout::builder()
            .config(cfg)
            .memory(mem)
            .kv_block_layout(KvBlockLayout::OperationalNHD)
            .build()
            .unwrap();
        let err = layout_to_view(&fc as &dyn Layout)
            .expect_err("projection should error when num_heads is unset");
        let msg = format!("{err:#}");
        assert!(
            msg.contains("num_heads"),
            "expected num_heads-related error, got: {msg}"
        );
    }
}
