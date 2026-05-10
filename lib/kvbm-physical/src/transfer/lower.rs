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

use std::cmp::Reverse;

use anyhow::{Result, bail};

use crate::layout::{LayoutView, PhysicalLayout};
use crate::transfer::TransferCapabilities;
use crate::transfer::plan::{CopyOp, CopyPlan};
use crate::transfer::strategy::TransferStrategy;

/// Project a [`PhysicalLayout`] to a [`LayoutView`] for the planner.
///
/// Each concrete [`Layout`] impl owns its own projection via
/// [`Layout::layout_view`]; this helper is the convenience wrapper
/// for callers that already have a [`PhysicalLayout`] in hand.
pub(crate) fn physical_to_layout_view(physical: &PhysicalLayout) -> Result<LayoutView> {
    physical.layout().layout_view()
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

// ─────────────────────────── Selector scaffolding ────────────────────────────
//
// `SelectionContext` + `score_candidate` + `select_candidate` land in PR-7.2
// as scaffolding only. Today's candidate vectors always have ≤1 element, so
// the scorer is invoked but never has a real choice. The benefit: PR-7.3
// (small-strided-copy) and PR-7.4 (CudaGraphReplay) plug in new variants and
// the existing scorer automatically selects the best one.
//
// Scoring design:
//   - Higher score = preferred.
//   - Negative score (< 0) = placeholder / not-yet-ready; filtered before
//     selection so they can never be chosen even when no positive candidate
//     exists (that surfaces a precise error instead of a silent wrong pick).
//   - Tie-break: earlier element in the slice wins (stable ordering).

/// Context passed to [`score_candidate`] to let it key on transfer shape
/// rather than only the candidate variant.
///
/// # Route
///
/// `strategy: TransferStrategy` is used directly rather than a 2-variant
/// `enum Route { Cuda, Nixl }` abstraction because:
/// (a) `TransferStrategy` already encodes the Cuda*/Nixl* family, and
/// (b) PR-7.3+ scoring may distinguish D2D from H2D even within the same
///     Cuda family, so a 2-variant proxy would lose information.
/// The scorer uses `is_cuda_family()` / `is_nixl_family()` helpers for
/// the coarse routing check it needs today.
pub(crate) struct SelectionContext<'a> {
    /// Transfer strategy in use — determines the route family.
    pub strategy: TransferStrategy,

    /// Number of descriptors in the planned op-set.
    pub descriptor_count: usize,

    /// Total bytes to be moved across all ops.
    pub total_bytes: usize,

    /// Tensor element dtype when the kernel catalog is enabled.
    ///
    /// `None` when `cfg(not(feature = "permute_kernels"))` or when the
    /// caller's `LayoutConfig::dtype` is unset (same gate as in
    /// `LayoutConfig`).
    #[cfg(feature = "permute_kernels")]
    pub dtype: Option<kvbm_kernels::TensorDataType>,

    /// Capability flags in effect for this transfer.
    pub capabilities: &'a TransferCapabilities,
}

impl TransferStrategy {
    /// Returns `true` for `CudaAsync{H2D, D2H, D2D}`.
    ///
    /// Used by the scorer to distinguish Cuda-family routes from NIXL.
    pub(crate) fn is_cuda_family(self) -> bool {
        matches!(
            self,
            TransferStrategy::CudaAsyncH2D
                | TransferStrategy::CudaAsyncD2H
                | TransferStrategy::CudaAsyncD2D
        )
    }

    /// Returns `true` for `Nixl{Read, Write, ReadFlipped, WriteFlipped}`.
    pub(crate) fn is_nixl_family(self) -> bool {
        matches!(
            self,
            TransferStrategy::NixlRead
                | TransferStrategy::NixlWrite
                | TransferStrategy::NixlReadFlipped
                | TransferStrategy::NixlWriteFlipped
        )
    }
}

/// Scoring constants.
///
/// ```text
/// TransformKernel  1100   — dedicated permute kernel; preferred over raw DMA
///                             when the route is Cuda* (avoids per-coord
///                             descriptor explosion).
/// DirectDma        1000   — universal DMA fallback; available on all routes.
/// BatchedDma       1000   — equivalent to DirectDma today; PR-7.3 may
///                             differentiate via descriptor-count thresholds.
/// Staged              -1  — placeholder variant; spec not yet finalised.
///                             Negative score ensures it is *never* selected.
/// ```
///
/// Adding a new `Candidate` variant for PR-7.3+ means:
/// 1. Add its variant to `Candidate`.
/// 2. Add a match arm here with the appropriate base score.
/// 3. Adjust constants above if the new variant should outrank existing ones.
const SCORE_TRANSFORM_KERNEL: i64 = 1100;
const SCORE_DIRECT_DMA: i64 = 1000;
const SCORE_BATCHED_DMA: i64 = 1000;
const SCORE_STAGED: i64 = -1;

/// Score a single candidate given the selection context.
///
/// Returns a higher number for a more preferred candidate.
/// Negative scores mark placeholder variants that must be filtered out
/// before `select_candidate` makes a final choice.
///
/// # Notes on `TransformKernel`
///
/// In production today `TransformKernel` is never passed through
/// `select_candidate` — `plan_and_lower` routes it through
/// `PlanOutcome::Transform` and dispatches it directly. The arm exists
/// as scaffolding for PR-7.3/7.4 where future candidates may compete
/// with `TransformKernel` under a Cuda route.
#[cfg_attr(not(feature = "permute_kernels"), allow(unused_variables))]
pub(crate) fn score_candidate(candidate: &Candidate, ctx: &SelectionContext<'_>) -> i64 {
    match candidate {
        Candidate::DirectDma { .. } => SCORE_DIRECT_DMA,
        Candidate::BatchedDma { .. } => SCORE_BATCHED_DMA,
        #[cfg(feature = "permute_kernels")]
        Candidate::TransformKernel { .. } => {
            // TransformKernel is only meaningful on Cuda-family routes;
            // it earns a higher base score than DirectDma because the
            // dedicated permute kernel avoids the per-coord descriptor
            // explosion that CopyPlan::Direct would generate.
            // On NIXL routes, dispatch goes through dispatch_staged_nixl_transform,
            // not through select_candidate, so this arm is unreachable there.
            if ctx.strategy.is_cuda_family() {
                SCORE_TRANSFORM_KERNEL
            } else {
                SCORE_DIRECT_DMA // treat as equivalent if somehow reached on NIXL
            }
        }
        Candidate::Staged { .. } => SCORE_STAGED,
    }
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

/// Select the best candidate to execute.
///
/// Each candidate is scored via [`score_candidate`]; candidates with a
/// negative score (placeholders such as [`Candidate::Staged`]) are filtered
/// out before the max is taken. On score ties, the *earlier* element in
/// the slice wins (stable ordering: `max_by_key` with `(score, Reverse(idx))`
/// so a lower index beats a higher one at equal score).
///
/// Errors with a message containing `"no executable candidate"` when all
/// candidates score below zero.
pub(crate) fn select_candidate<'a>(
    candidates: &'a [Candidate],
    ctx: &SelectionContext<'_>,
) -> Result<&'a Candidate> {
    candidates
        .iter()
        .enumerate()
        .filter_map(|(idx, c)| {
            let s = score_candidate(c, ctx);
            if s >= 0 { Some((idx, c, s)) } else { None }
        })
        // Higher score wins; on equal score, lower index wins (Reverse keeps
        // the element with the smallest index when scores are equal).
        .max_by_key(|&(idx, _, s)| (s, Reverse(idx)))
        .map(|(_, c, _)| c)
        .ok_or_else(|| {
            anyhow::anyhow!(
                "select_candidate: no executable candidate in {} candidates \
                 (all scored < 0 or list is empty)",
                candidates.len()
            )
        })
}

#[cfg(all(test, feature = "testing-kvbm"))]
mod tests {
    use super::*;
    use kvbm_common::{KvDim, KvDimLayout, KvDimStrides};

    use crate::layout::Layout;
    use crate::transfer::plan::AnnotatedLayout;

    // ── helpers ──────────────────────────────────────────────────────────────

    fn cuda_ctx() -> SelectionContext<'static> {
        // A minimal SelectionContext for unit tests that don't need real
        // descriptor counts or bytes — just enough to drive the scorer.
        static CAPS: TransferCapabilities = TransferCapabilities {
            allow_gds: false,
            allow_gpu_rdma: false,
        };
        SelectionContext {
            strategy: TransferStrategy::CudaAsyncD2D,
            descriptor_count: 1,
            total_bytes: 4096,
            #[cfg(feature = "permute_kernels")]
            dtype: None,
            capabilities: &CAPS,
        }
    }

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
        // Staged scores -1 (filtered); DirectDma scores 1000 — assert it wins.
        let ctx = cuda_ctx();
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
        let picked = select_candidate(&cands, &ctx).unwrap();
        assert!(matches!(picked, Candidate::DirectDma { .. }));
    }

    #[test]
    fn select_no_direct_errors() {
        let ctx = cuda_ctx();
        let cands = vec![Candidate::Staged {}];
        assert!(select_candidate(&cands, &ctx).is_err());
    }

    // ── new PR-7.2 scorer tests ───────────────────────────────────────────────

    /// On a Cuda route, TransformKernel scores 1100 vs DirectDma's 1000 vs
    /// BatchedDma's 1000. TransformKernel should win.
    #[cfg(feature = "permute_kernels")]
    #[test]
    fn select_picks_highest_scoring_candidate() {
        use crate::transfer::kernel_catalog::{KernelInvocation, KernelKind};
        use kvbm_kernels::{BlockLayout, TensorDataType};

        let ctx = cuda_ctx();
        let invoc = KernelInvocation {
            kind: KernelKind::NhdHndTranspose,
            num_layers: 1,
            outer_dim: 1,
            page_size: 16,
            num_heads: 8,
            head_dim: 64,
            dtype: TensorDataType::F16,
            block_layout: BlockLayout::NHD,
        };
        let cands = vec![
            Candidate::BatchedDma { groups: vec![] },
            Candidate::DirectDma { ops: vec![] },
            Candidate::TransformKernel { invocation: invoc },
        ];
        let picked = select_candidate(&cands, &ctx).unwrap();
        assert!(
            matches!(picked, Candidate::TransformKernel { .. }),
            "expected TransformKernel (score {SCORE_TRANSFORM_KERNEL}), got {picked:?}"
        );
    }

    /// Staged scores -1 and must be filtered; DirectDma should win.
    #[test]
    fn select_filters_placeholder_candidates() {
        let ctx = cuda_ctx();
        let cands = vec![
            Candidate::Staged {},
            Candidate::DirectDma { ops: vec![] },
        ];
        let picked = select_candidate(&cands, &ctx).unwrap();
        assert!(matches!(picked, Candidate::DirectDma { .. }));
    }

    /// When all candidates score < 0 the error message must contain
    /// "no executable candidate".
    #[test]
    fn select_no_executable_candidates_errors() {
        let ctx = cuda_ctx();
        let cands = vec![Candidate::Staged {}];
        let err = select_candidate(&cands, &ctx).unwrap_err();
        let msg = format!("{err}");
        assert!(
            msg.contains("no executable candidate"),
            "expected 'no executable candidate' in error, got: {msg}"
        );
    }

    /// Two DirectDma candidates with equal scores — first one wins.
    #[test]
    fn select_stable_ordering_on_tie() {
        let ctx = cuda_ctx();
        let first = Candidate::DirectDma {
            ops: vec![CopyOp { src_addr: 0x1000, dst_addr: 0x2000, size: 64 }],
        };
        let second = Candidate::DirectDma {
            ops: vec![CopyOp { src_addr: 0x3000, dst_addr: 0x4000, size: 128 }],
        };
        let cands = vec![first, second];
        let picked = select_candidate(&cands, &ctx).unwrap();
        // The first candidate has src_addr 0x1000; assert that's what we got.
        match picked {
            Candidate::DirectDma { ops } => {
                assert_eq!(ops[0].src_addr, 0x1000, "expected first candidate to win tie");
            }
            other => panic!("expected DirectDma, got {other:?}"),
        }
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

        let view = (&fc as &dyn Layout).layout_view().unwrap();
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

        let view = (&ls as &dyn Layout).layout_view().unwrap();
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

        let view = (&ls as &dyn Layout).layout_view().unwrap();
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
        assert!((&fc as &dyn Layout).layout_view().is_err());
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
        let err = (&fc as &dyn Layout)
            .layout_view()
            .expect_err("projection should error when num_heads is unset");
        let msg = format!("{err:#}");
        assert!(
            msg.contains("num_heads"),
            "expected num_heads-related error, got: {msg}"
        );
    }
}
