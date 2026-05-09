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

/// FC: single allocation, axes `[Block, Layer, Outer, Page, Payload]`,
/// no region axis.
fn fc_to_view(fc: &FullyContiguousLayout, cfg: &crate::layout::LayoutConfig) -> Result<LayoutView> {
    if matches!(fc.kv_block_layout(), KvBlockLayout::Unknown) {
        bail!("physical_to_layout_view: FullyContiguousLayout has Unknown block layout");
    }
    let elem = cfg.dtype_width_bytes;
    let dims = vec![
        KvDim::Block,
        KvDim::Layer,
        KvDim::Outer,
        KvDim::Page,
        KvDim::Payload,
    ];
    let sizes = vec![
        cfg.num_blocks,
        cfg.num_layers,
        cfg.outer_dim,
        cfg.page_size,
        cfg.inner_dim,
    ];
    let layout = KvDimLayout::new(dims, sizes)?;

    // Row-major byte strides over [Block, Layer, Outer, Page, Payload]:
    let s_hs = elem;
    let s_pg = s_hs * cfg.inner_dim;
    let s_ot = s_pg * cfg.page_size;
    let s_la = s_ot * cfg.outer_dim;
    let s_bk = s_la * cfg.num_layers;
    let strides = KvDimStrides::from_byte_strides(vec![s_bk, s_la, s_ot, s_pg, s_hs], elem)?;

    // FC has exactly one Buffer in memory_regions.
    let buffers = fc.memory_regions();
    if buffers.len() != 1 {
        bail!(
            "physical_to_layout_view: FullyContiguousLayout expects 1 Buffer, got {}",
            buffers.len()
        );
    }
    let regions = vec![buffers[0].addr()];
    LayoutView::full(layout, strides, regions, None)
}

/// LS: per-layer regions, `region_axis = Some(Layer)`. Inner axis order
/// depends on `BlockDimension`:
/// - BlockIsFirstDim:  `[Block, Outer, Page, Payload]`
/// - BlockIsSecondDim: `[Outer, Block, Page, Payload]`
fn ls_to_view(ls: &LayerSeparateLayout, cfg: &crate::layout::LayoutConfig) -> Result<LayoutView> {
    if matches!(ls.kv_block_layout(), KvBlockLayout::Unknown) {
        bail!("physical_to_layout_view: LayerSeparateLayout has Unknown block layout");
    }
    let elem = cfg.dtype_width_bytes;
    // Inner-axis stride math (per region):
    let s_hs = elem;
    let s_pg = s_hs * cfg.inner_dim;
    let region_size = s_pg * cfg.page_size;

    // Per-side outer strides depend on which axis is outermost in the
    // region.
    let (dims, sizes, byte_strides) = match ls.block_dim() {
        BlockDimension::BlockIsFirstDim => {
            let s_ot = region_size;
            let s_bk = s_ot * cfg.outer_dim;
            // Layer stride is unused for in-region addressing; use the
            // per-region byte size as a positive sentinel.
            let s_la = s_bk * cfg.num_blocks;
            (
                vec![
                    KvDim::Layer,
                    KvDim::Block,
                    KvDim::Outer,
                    KvDim::Page,
                    KvDim::Payload,
                ],
                vec![
                    cfg.num_layers,
                    cfg.num_blocks,
                    cfg.outer_dim,
                    cfg.page_size,
                    cfg.inner_dim,
                ],
                vec![s_la, s_bk, s_ot, s_pg, s_hs],
            )
        }
        BlockDimension::BlockIsSecondDim => {
            let s_bk = region_size;
            let s_ot = s_bk * cfg.num_blocks;
            let s_la = s_ot * cfg.outer_dim;
            (
                vec![
                    KvDim::Layer,
                    KvDim::Outer,
                    KvDim::Block,
                    KvDim::Page,
                    KvDim::Payload,
                ],
                vec![
                    cfg.num_layers,
                    cfg.outer_dim,
                    cfg.num_blocks,
                    cfg.page_size,
                    cfg.inner_dim,
                ],
                vec![s_la, s_ot, s_bk, s_pg, s_hs],
            )
        }
    };

    let layout = KvDimLayout::new(dims, sizes)?;
    let strides = KvDimStrides::from_byte_strides(byte_strides, elem)?;
    // LS exposes one Buffer per layer; their base addresses are the
    // per-region bases.
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
/// PR-5 emits only [`Candidate::DirectDma`] from [`CopyPlan::Direct`].
/// Other variants are reserved for follow-up PRs:
/// * [`Candidate::BatchedDma`] groups ops by region/stream for
///   coalesced launches; arrives when stream-aware grouping is wired.
/// * [`Candidate::TransformKernel`] consumes [`CopyPlan::Transform`]
///   via the `kvbm-kernels` catalog (PR-6).
/// * [`Candidate::Staged`] handles two-hop transfers when direct
///   transform is impossible.
#[derive(Debug, Clone)]
pub(crate) enum Candidate {
    DirectDma { ops: Vec<CopyOp> },
    BatchedDma { groups: Vec<Vec<CopyOp>> },
    TransformKernel {/* spec lands in PR-6 */},
    Staged {/* spec lands later */},
}

/// Lower a [`CopyPlan`] to a vector of executor candidates.
///
/// PR-5 surface:
/// * [`CopyPlan::Direct`] yields a single [`Candidate::DirectDma`].
/// * [`CopyPlan::Transform`] errors with `kernel catalog not wired
///   yet` — PR-6 introduces the `KernelSignature` registry and lowers
///   transforms into [`Candidate::TransformKernel`].
/// * [`CopyPlan::Staged`] is reserved; the prototype never emits it.
pub(crate) fn lower_to_candidates(plan: CopyPlan) -> Result<Vec<Candidate>> {
    match plan {
        CopyPlan::Direct(ops) => Ok(vec![Candidate::DirectDma { ops }]),
        CopyPlan::Transform { .. } => bail!(
            "lower_to_candidates: CopyPlan::Transform is not yet wired \
             (PR-6 lands the kernel catalog)"
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
        let cands = vec![
            Candidate::TransformKernel {},
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
        let cands = vec![Candidate::TransformKernel {}];
        assert!(select_candidate(&cands).is_err());
    }

    /// FullyContiguousLayout: verify that `addr_of` on the projected
    /// view agrees with `Layout::memory_region` + within-region stride
    /// math for a representative coord. This is the load-bearing
    /// correctness check for the planner-driven CudaAsync path.
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

        // Pick a non-trivial coord. addr_of(local) must match the
        // existing memory_region API + per-axis stride within the
        // region.
        let block_id = 2usize;
        let layer_id = 1usize;
        let outer_id = 1usize;
        let page = 5usize;
        let inner = 17usize;
        let coord = kvbm_common::CoordByLabel::new()
            .with(KvDim::Block, block_id)
            .with(KvDim::Layer, layer_id)
            .with(KvDim::Outer, outer_id)
            .with(KvDim::Page, page)
            .with(KvDim::Payload, inner);

        let view_addr = al.addr_of(&coord).unwrap();
        let region = fc.memory_region(block_id, layer_id, outer_id).unwrap();
        let expected = region.addr()
            + page * cfg.inner_dim * cfg.dtype_width_bytes
            + inner * cfg.dtype_width_bytes;
        assert_eq!(view_addr, expected);
    }

    /// LayerSeparateLayout (BlockIsFirstDim): same correctness check
    /// against the per-layer-region indexing.
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
        let inner = 7usize;
        let coord = kvbm_common::CoordByLabel::new()
            .with(KvDim::Block, block_id)
            .with(KvDim::Layer, layer_id)
            .with(KvDim::Outer, outer_id)
            .with(KvDim::Page, page)
            .with(KvDim::Payload, inner);

        let view_addr = al.addr_of(&coord).unwrap();
        let region = ls.memory_region(block_id, layer_id, outer_id).unwrap();
        let expected = region.addr()
            + page * cfg.inner_dim * cfg.dtype_width_bytes
            + inner * cfg.dtype_width_bytes;
        assert_eq!(view_addr, expected);
    }

    /// LayerSeparateLayout (BlockIsSecondDim): per-region inner shape
    /// `[Outer, Block, Page, Payload]` with the outermost-region
    /// stride larger than the per-block stride. Round-trip the same
    /// way as the BlockIsFirstDim variant.
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
        let inner = 11usize;
        let coord = kvbm_common::CoordByLabel::new()
            .with(KvDim::Block, block_id)
            .with(KvDim::Layer, layer_id)
            .with(KvDim::Outer, outer_id)
            .with(KvDim::Page, page)
            .with(KvDim::Payload, inner);

        let view_addr = al.addr_of(&coord).unwrap();
        let region = ls.memory_region(block_id, layer_id, outer_id).unwrap();
        let expected = region.addr()
            + page * cfg.inner_dim * cfg.dtype_width_bytes
            + inner * cfg.dtype_width_bytes;
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
}
