// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Side-by-side equivalence tests for `TransferOptions::use_planner`.
//!
//! Each test runs the same source layout through two destination
//! layouts in sequence: one with `use_planner = false` (legacy
//! `select_strategy` path) and one with `use_planner = true` (PR-5
//! planner pipeline). Both destinations are checksummed and compared
//! against the source — if both pass, the planner path produced
//! byte-equivalent output to the legacy path.
//!
//! Coverage focuses on the *application* of the copy (the addressing
//! is already exercised by the address-math round-trip tests in
//! `transfer::lower::tests`). The four scenarios picked here are the
//! shapes most likely to surface a regression:
//! - FC↔FC D2D (whole-block fast path).
//! - LS↔LS D2D, BlockIsFirstDim (per-layer regions).
//! - FC↔LS D2D (heterogeneous layouts).
//! - FC↔FC H2D (host→device direction).
//!
//! NIXL paths land in PR-5.6 once `execute_planner_nixl_transfer` is
//! wired.

use anyhow::Result;

use super::gate::gpu_serial;
use super::local_transfers::{LayoutType, build_agent_for_kinds};
use super::*;
use crate::layout::{KvBlockLayout, PhysicalLayout};
use crate::transfer::executor::{TransferOptionsInternal, execute_transfer};

/// Build a layout the planner path can project — sets
/// `KvBlockLayout::OperationalNHD` explicitly. The shared
/// `super::create_fc_layout` / `create_lw_layout` helpers leave
/// `kv_block_layout` as the default `Unknown`, which the planner-path
/// projection rejects (the projection cannot honestly emit Direct ops
/// when the per-token substructure is unknown).
fn build_layout_for_planner(
    agent: NixlAgent,
    layout_type: LayoutType,
    storage_kind: StorageKind,
    num_blocks: usize,
) -> PhysicalLayout {
    let config = standard_config(num_blocks);
    let builder = PhysicalLayout::builder(agent)
        .with_config(config)
        .with_block_layout(KvBlockLayout::OperationalNHD);
    let typed = match layout_type {
        LayoutType::FC => builder.fully_contiguous(),
        LayoutType::LW => builder.layer_separate(BlockDimension::BlockIsFirstDim),
    };
    match storage_kind {
        StorageKind::System => typed.allocate_system().build().unwrap(),
        StorageKind::Pinned => typed.allocate_pinned(None).build().unwrap(),
        StorageKind::Device(id) => typed.allocate_device(id).build().unwrap(),
        StorageKind::Disk(_) => typed.allocate_disk(None).build().unwrap(),
    }
}

/// Run a transfer with the requested planner setting and return
/// destination checksums for comparison.
async fn transfer_and_checksum(
    src: &PhysicalLayout,
    dst: &PhysicalLayout,
    src_blocks: &[BlockId],
    dst_blocks: &[BlockId],
    use_planner: bool,
    ctx: &crate::transfer::context::TransferContext,
) -> Result<HashMap<BlockId, BlockChecksum>> {
    let options = TransferOptionsInternal::builder()
        .use_planner(use_planner)
        .build()?;
    let notification = execute_transfer(src, dst, src_blocks, dst_blocks, options, ctx)?;
    notification.await?;
    compute_block_checksums(dst, dst_blocks)
}

/// Side-by-side equivalence: build src, fill, transfer to two
/// destinations (legacy + planner), assert both match src checksums
/// AND match each other.
async fn assert_planner_matches_legacy(
    src_layout: LayoutType,
    src_kind: StorageKind,
    dst_layout: LayoutType,
    dst_kind: StorageKind,
) -> Result<()> {
    let agent = build_agent_for_kinds(&[src_kind, dst_kind])?;
    let src = build_layout_for_planner(agent.clone(), src_layout.clone(), src_kind, 4);
    let dst_legacy = build_layout_for_planner(agent.clone(), dst_layout.clone(), dst_kind, 4);
    let dst_planner = build_layout_for_planner(agent.clone(), dst_layout, dst_kind, 4);

    let src_blocks = vec![0, 1];
    let dst_blocks = vec![2, 3];

    // Fill src once; both destinations pull from the same data.
    let src_checksums = fill_and_checksum(&src, &src_blocks, FillPattern::Sequential)?;
    let ctx = create_transfer_context(agent, None).unwrap();

    let legacy_checksums = transfer_and_checksum(
        &src,
        &dst_legacy,
        &src_blocks,
        &dst_blocks,
        false,
        ctx.context(),
    )
    .await?;
    let planner_checksums = transfer_and_checksum(
        &src,
        &dst_planner,
        &src_blocks,
        &dst_blocks,
        true,
        ctx.context(),
    )
    .await?;

    // Both paths must reproduce src checksums on dst.
    verify_checksums_by_position(&src_checksums, &src_blocks, &dst_legacy, &dst_blocks)?;
    verify_checksums_by_position(&src_checksums, &src_blocks, &dst_planner, &dst_blocks)?;

    // And they must agree pairwise — if both produce the same wrong
    // result, the per-axis test would still pass; this catches the
    // (unlikely) collusion case.
    for (&legacy_id, &planner_id) in dst_blocks.iter().zip(dst_blocks.iter()) {
        let legacy = legacy_checksums.get(&legacy_id).expect("legacy checksum");
        let planner = planner_checksums
            .get(&planner_id)
            .expect("planner checksum");
        assert_eq!(
            legacy, planner,
            "planner / legacy checksum disagreement on dst block {}: \
             legacy={legacy} vs planner={planner}",
            legacy_id
        );
    }

    Ok(())
}

#[tokio::test]
async fn use_planner_matches_legacy_fc_fc_d2d() -> Result<()> {
    let src_kind = StorageKind::Device(0);
    let dst_kind = StorageKind::Device(0);
    skip_if_stubs_and_device!(src_kind, dst_kind);
    gpu_serial!();
    assert_planner_matches_legacy(LayoutType::FC, src_kind, LayoutType::FC, dst_kind).await
}

#[tokio::test]
async fn use_planner_matches_legacy_lw_lw_d2d() -> Result<()> {
    let src_kind = StorageKind::Device(0);
    let dst_kind = StorageKind::Device(0);
    skip_if_stubs_and_device!(src_kind, dst_kind);
    gpu_serial!();
    assert_planner_matches_legacy(LayoutType::LW, src_kind, LayoutType::LW, dst_kind).await
}

#[tokio::test]
async fn use_planner_matches_legacy_fc_lw_d2d() -> Result<()> {
    let src_kind = StorageKind::Device(0);
    let dst_kind = StorageKind::Device(0);
    skip_if_stubs_and_device!(src_kind, dst_kind);
    gpu_serial!();
    assert_planner_matches_legacy(LayoutType::FC, src_kind, LayoutType::LW, dst_kind).await
}

#[tokio::test]
async fn use_planner_matches_legacy_fc_fc_h2d() -> Result<()> {
    let src_kind = StorageKind::Pinned;
    let dst_kind = StorageKind::Device(0);
    skip_if_stubs_and_device!(src_kind, dst_kind);
    gpu_serial!();
    assert_planner_matches_legacy(LayoutType::FC, src_kind, LayoutType::FC, dst_kind).await
}

#[tokio::test]
async fn use_planner_matches_legacy_fc_fc_d2h() -> Result<()> {
    let src_kind = StorageKind::Device(0);
    let dst_kind = StorageKind::Pinned;
    skip_if_stubs_and_device!(src_kind, dst_kind);
    gpu_serial!();
    assert_planner_matches_legacy(LayoutType::FC, src_kind, LayoutType::FC, dst_kind).await
}

// ────────────────── PR-6.1: operational ↔ universal ──────────────────

/// Build an FC PhysicalLayout on Device(0) with an explicit
/// `KvBlockLayout`. PR-6.1 catalog dispatch needs the layout enum
/// set; the standard helper only configures NHD.
fn build_fc_with_block_layout(
    agent: NixlAgent,
    block_layout: KvBlockLayout,
    num_blocks: usize,
) -> PhysicalLayout {
    let config = standard_config(num_blocks);
    PhysicalLayout::builder(agent)
        .with_config(config)
        .with_block_layout(block_layout)
        .fully_contiguous()
        .allocate_device(0)
        .build()
        .unwrap()
}

/// Operational ↔ universal round-trip on Device(0). Fill an NHD
/// source with a deterministic pattern, transfer NHD → universal via
/// the planner-driven kernel catalog (`universal_from_block`), then
/// universal → fresh NHD via the inverse kernel
/// (`block_from_universal`). Verify the final NHD matches the source
/// byte-for-byte.
///
/// There's no legacy comparison path for transforms — the legacy
/// CudaAsync executor doesn't dispatch `universal_from_block` /
/// `block_from_universal`, so a side-by-side check is impossible.
/// The round-trip pattern is the strongest correctness check
/// available.
async fn assert_operational_universal_round_trip(operational: KvBlockLayout) -> Result<()> {
    let agent = build_agent_for_kinds(&[StorageKind::Device(0)])?;
    let src = build_fc_with_block_layout(agent.clone(), operational, 4);
    let mid = build_fc_with_block_layout(agent.clone(), KvBlockLayout::Universal, 4);
    let dst = build_fc_with_block_layout(agent.clone(), operational, 4);

    let src_blocks = vec![0, 1];
    let mid_blocks = vec![2, 3];
    let dst_blocks = vec![0, 1];

    let src_checksums = fill_and_checksum(&src, &src_blocks, FillPattern::Sequential)?;
    let ctx = create_transfer_context(agent, None).unwrap();
    let options = || -> Result<TransferOptionsInternal> {
        TransferOptionsInternal::builder().use_planner(true).build()
    };

    // Forward: operational → universal.
    let forward = execute_transfer(
        &src,
        &mid,
        &src_blocks,
        &mid_blocks,
        options()?,
        ctx.context(),
    )?;
    forward.await?;

    // Reverse: universal → fresh operational.
    let reverse = execute_transfer(
        &mid,
        &dst,
        &mid_blocks,
        &dst_blocks,
        options()?,
        ctx.context(),
    )?;
    reverse.await?;

    // dst[0,1] should hold the original src[0,1] pattern.
    verify_checksums_by_position(&src_checksums, &src_blocks, &dst, &dst_blocks)?;
    Ok(())
}

#[tokio::test]
async fn use_planner_round_trip_nhd_via_universal() -> Result<()> {
    skip_if_stubs_and_device!(StorageKind::Device(0));
    gpu_serial!();
    assert_operational_universal_round_trip(KvBlockLayout::OperationalNHD).await
}

#[tokio::test]
async fn use_planner_round_trip_hnd_via_universal() -> Result<()> {
    skip_if_stubs_and_device!(StorageKind::Device(0));
    gpu_serial!();
    assert_operational_universal_round_trip(KvBlockLayout::OperationalHND).await
}

// ────────────────── PR-6.3: NHD ↔ HND ──────────────────

/// Round-trip NHD → HND → NHD on Device(0) via the planner-driven
/// `nhd_hnd_transpose` kernel. Wiring test only — kernel correctness
/// is verified independently in `kvbm-kernels`'s `kernel_roundtrip`
/// suite, which compares each direction's output against ground-truth
/// chunks (not against the kernel's own inverse).
async fn assert_nhd_hnd_round_trip(src_layout: KvBlockLayout) -> Result<()> {
    let other = match src_layout {
        KvBlockLayout::OperationalNHD => KvBlockLayout::OperationalHND,
        KvBlockLayout::OperationalHND => KvBlockLayout::OperationalNHD,
        _ => panic!("assert_nhd_hnd_round_trip: src must be NHD or HND, got {src_layout:?}"),
    };
    let agent = build_agent_for_kinds(&[StorageKind::Device(0)])?;
    let src = build_fc_with_block_layout(agent.clone(), src_layout, 4);
    let mid = build_fc_with_block_layout(agent.clone(), other, 4);
    let dst = build_fc_with_block_layout(agent.clone(), src_layout, 4);

    let src_blocks = vec![0, 1];
    let mid_blocks = vec![2, 3];
    let dst_blocks = vec![0, 1];

    let src_checksums = fill_and_checksum(&src, &src_blocks, FillPattern::Sequential)?;
    let ctx = create_transfer_context(agent, None).unwrap();
    let options = || -> Result<TransferOptionsInternal> {
        TransferOptionsInternal::builder().use_planner(true).build()
    };

    let forward = execute_transfer(
        &src,
        &mid,
        &src_blocks,
        &mid_blocks,
        options()?,
        ctx.context(),
    )?;
    forward.await?;
    let reverse = execute_transfer(
        &mid,
        &dst,
        &mid_blocks,
        &dst_blocks,
        options()?,
        ctx.context(),
    )?;
    reverse.await?;

    verify_checksums_by_position(&src_checksums, &src_blocks, &dst, &dst_blocks)?;
    Ok(())
}

#[tokio::test]
async fn use_planner_round_trip_nhd_to_hnd() -> Result<()> {
    skip_if_stubs_and_device!(StorageKind::Device(0));
    gpu_serial!();
    assert_nhd_hnd_round_trip(KvBlockLayout::OperationalNHD).await
}

#[tokio::test]
async fn use_planner_round_trip_hnd_to_nhd() -> Result<()> {
    skip_if_stubs_and_device!(StorageKind::Device(0));
    gpu_serial!();
    assert_nhd_hnd_round_trip(KvBlockLayout::OperationalHND).await
}

// ────────────────── PR-7.3: small-inner threshold-fallback ──────────────────

/// Integration test: a same-layout D2D transfer whose inner contiguous tail
/// is below 4 KiB routes through `SmallStridedCopy` (vectorized_copy) and
/// produces byte-equivalent output to the legacy path.
///
/// Layout geometry chosen so inner_bytes < 4096:
///   page_size=1, inner_dim=128 (nh=8, hd=16), outer_dim=2, dtype=2 bytes
///   inner_bytes = outer×page×inner×dtype = 2×1×128×2 = 512 bytes < 4096
///
/// The planner path with `use_planner = true` should now succeed where
/// previously (before PR-7.3) it would have returned a "plan_copy emitted
/// Transform for same-KvBlockLayout pair" error.
#[tokio::test]
async fn use_planner_small_inner_threshold_fallback_d2d() -> Result<()> {
    let src_kind = StorageKind::Device(0);
    let dst_kind = StorageKind::Device(0);
    skip_if_stubs_and_device!(src_kind, dst_kind);
    gpu_serial!();

    // page_size=1: inner_bytes = 2×1×128×2 = 512 < 4096 → ThresholdFallback path.
    let agent = build_agent_for_kinds(&[src_kind, dst_kind])?;
    let config = standard_config_with_page_size(4, /*page_size=*/ 1);
    let build_layout = |bl: KvBlockLayout| {
        PhysicalLayout::builder(agent.clone())
            .with_config(config.clone())
            .with_block_layout(bl)
            .fully_contiguous()
            .allocate_device(0)
            .build()
            .unwrap()
    };
    let src = build_layout(KvBlockLayout::OperationalNHD);
    let dst_legacy = build_layout(KvBlockLayout::OperationalNHD);
    let dst_planner = build_layout(KvBlockLayout::OperationalNHD);

    let src_blocks = vec![0, 1];
    let dst_blocks = vec![2, 3];
    let src_checksums = fill_and_checksum(&src, &src_blocks, FillPattern::Sequential)?;
    let ctx = create_transfer_context(agent, None).unwrap();

    // Legacy path (use_planner = false).
    let legacy_checksums = transfer_and_checksum(
        &src,
        &dst_legacy,
        &src_blocks,
        &dst_blocks,
        false,
        ctx.context(),
    )
    .await?;

    // Planner path (use_planner = true) — must not error, must match legacy.
    let planner_checksums = transfer_and_checksum(
        &src,
        &dst_planner,
        &src_blocks,
        &dst_blocks,
        true,
        ctx.context(),
    )
    .await?;

    verify_checksums_by_position(&src_checksums, &src_blocks, &dst_legacy, &dst_blocks)?;
    verify_checksums_by_position(&src_checksums, &src_blocks, &dst_planner, &dst_blocks)?;
    for (&did, _) in dst_blocks.iter().zip(dst_blocks.iter()) {
        let lc = legacy_checksums.get(&did).expect("legacy");
        let pc = planner_checksums.get(&did).expect("planner");
        assert_eq!(lc, pc, "planner/legacy checksum mismatch at block {did}");
    }
    Ok(())
}

// ────────────────── c6: auto-promote on requires_transform ──────────────────
//
// Production layout builders historically set `dtype_width_bytes` but
// leave `LayoutConfig.dtype` as `None` (only the bench harness sets
// the typed enum). The c6 round-trip reproducers exercise the
// derive-from-width path by passing a `dtype=None` config; the
// `layer_range + transform` rejection test deliberately keeps the
// standard config to confirm the layer_range bail fires before the
// dtype derive.

/// Build an FC PhysicalLayout on Device(0) with an explicit
/// `KvBlockLayout` and `LayoutConfig.dtype = None`. Matches the shape
/// of layouts that production code (vLLM connector, engine builders)
/// emits today — `dtype_width_bytes` set, typed `dtype` unset. Used by
/// the c6 reproducers to exercise the `effective_dtype` derive path.
fn build_fc_dtype_none(
    agent: NixlAgent,
    block_layout: KvBlockLayout,
    num_blocks: usize,
) -> PhysicalLayout {
    let mut config = standard_config(num_blocks);
    config.dtype = None;
    PhysicalLayout::builder(agent)
        .with_config(config)
        .with_block_layout(block_layout)
        .fully_contiguous()
        .allocate_device(0)
        .build()
        .unwrap()
}

//
// c3 made G2 = KvBlockLayout::Universal in BlockLayoutMode::Universal
// while G1 stayed OperationalNHD. The offload pipeline's default
// TransferOptions (use_planner = false) routes through the legacy CUDA
// executor whose validate_layout_compatibility rejects cross-layout
// pairs — so G1↔G2 transfers under Universal mode bail at runtime even
// though the fused permute kernel is fully wired in the planner.
//
// c6 makes `executor::execute_transfer` detect `requires_transform` and
// auto-promote `use_planner = true`. These three reproducers pin that
// contract.

/// G1 (OperationalNHD) ↔ G2 (Universal) round-trip with default
/// TransferOptionsInternal — the exact call shape produced by
/// `kvbm-engine::offload::pipeline::execute_transfer` and
/// `worker/physical.rs::execute_local_transfer` under Universal mode.
///
/// On `wt/kvcc` HEAD (pre-c6) the first transfer fails in
/// `validate_layout_compatibility` with "Layout transformation not
/// supported: src=OperationalNHD, dst=Universal". After c6 both
/// transfers dispatch the fused permute kernel and the final
/// destination matches the source byte-for-byte.
///
/// Reproducer-first regression guard: src/dst hold 4 blocks while
/// the Universal intermediate holds 8. Production G1↔G2 have
/// differently-sized tiers (G1 GPU HBM is typically much larger
/// than G2 pinned host); the first cut of c6 missed this because
/// the reproducers used identical num_blocks, which trivially
/// passed `build_transform_invocation`'s per-block-shape gate. The
/// gate now ignores `num_blocks` (per-tier capacity, not
/// per-block geometry) but enforces everything else.
#[tokio::test]
async fn auto_promote_g1_nhd_to_g2_univ_default_options_dispatches_kernel() -> Result<()> {
    skip_if_stubs_and_device!(StorageKind::Device(0));
    gpu_serial!();

    let agent = build_agent_for_kinds(&[StorageKind::Device(0)])?;
    // dtype-None configs match production layout builders (vLLM
    // connector + engine sites); the c6 effective_dtype derive
    // turns this into BF16 at the catalog gate.
    let src = build_fc_dtype_none(agent.clone(), KvBlockLayout::OperationalNHD, 4);
    let mid = build_fc_dtype_none(agent.clone(), KvBlockLayout::Universal, 8);
    let dst = build_fc_dtype_none(agent.clone(), KvBlockLayout::OperationalNHD, 4);

    let src_blocks = vec![0, 1];
    let mid_blocks = vec![2, 3];
    let dst_blocks = vec![0, 1];

    let src_checksums = fill_and_checksum(&src, &src_blocks, FillPattern::Sequential)?;
    let ctx = create_transfer_context(agent, None).unwrap();

    // Regression guard: the contract is "default options must work for
    // requires_transform pairs because the executor auto-promotes."
    // If `TransferOptionsInternal::default()` ever changes to
    // `use_planner = true`, this guard becomes a no-op but the test
    // still exercises the same path.
    assert!(
        !TransferOptionsInternal::default().use_planner,
        "regression guard: TransferOptionsInternal::default() should be use_planner=false; \
         c6's auto-promote is the load-bearing mechanism for this test",
    );
    assert!(
        src.layout().config().dtype.is_none(),
        "regression guard: src config must have dtype=None to exercise the c6 \
         effective_dtype derive path (production layouts emit dtype=None today)",
    );

    let forward = execute_transfer(
        &src,
        &mid,
        &src_blocks,
        &mid_blocks,
        TransferOptionsInternal::default(),
        ctx.context(),
    )?;
    forward.await?;

    let reverse = execute_transfer(
        &mid,
        &dst,
        &mid_blocks,
        &dst_blocks,
        TransferOptionsInternal::default(),
        ctx.context(),
    )?;
    reverse.await?;

    verify_checksums_by_position(&src_checksums, &src_blocks, &dst, &dst_blocks)?;
    Ok(())
}

/// HND variant of the NHD round-trip above. Pins HND coverage
/// separately because the c3 Universal-mode rule applies to both
/// operational variants and the `kernel_catalog` dispatch picks a
/// different kernel direction per source layout. Same mismatched-
/// num_blocks shape (src=dst=4, mid=8) as the NHD test.
#[tokio::test]
async fn auto_promote_g1_hnd_to_g2_univ_default_options_dispatches_kernel() -> Result<()> {
    skip_if_stubs_and_device!(StorageKind::Device(0));
    gpu_serial!();

    let agent = build_agent_for_kinds(&[StorageKind::Device(0)])?;
    let src = build_fc_dtype_none(agent.clone(), KvBlockLayout::OperationalHND, 4);
    let mid = build_fc_dtype_none(agent.clone(), KvBlockLayout::Universal, 8);
    let dst = build_fc_dtype_none(agent.clone(), KvBlockLayout::OperationalHND, 4);

    let src_blocks = vec![0, 1];
    let mid_blocks = vec![2, 3];
    let dst_blocks = vec![0, 1];

    let src_checksums = fill_and_checksum(&src, &src_blocks, FillPattern::Sequential)?;
    let ctx = create_transfer_context(agent, None).unwrap();

    let forward = execute_transfer(
        &src,
        &mid,
        &src_blocks,
        &mid_blocks,
        TransferOptionsInternal::default(),
        ctx.context(),
    )?;
    forward.await?;
    let reverse = execute_transfer(
        &mid,
        &dst,
        &mid_blocks,
        &dst_blocks,
        TransferOptionsInternal::default(),
        ctx.context(),
    )?;
    reverse.await?;

    verify_checksums_by_position(&src_checksums, &src_blocks, &dst, &dst_blocks)?;
    Ok(())
}

// ────────────────── c6 phase 4b: layer-range + transform ──────────────────
//
// Universal storage is `[Block, HeadCount, Layer, Outer, Page, HeadSize]`,
// so a layer subrange of a universal block is interleaved across heads
// (per-head stride is `num_layers * outer * page * hd`, not a contiguous
// slab). Phase 4b extended the permute kernels with `nl_full` + `nl_offset`
// so per-layer scatter writes the correct slice without head-interleave
// corruption. These tests pin that contract end-to-end through the
// planner; the kernel-level head-interleave check lives in
// `kvbm-kernels/tests/kernel_roundtrip.rs::layer_subrange_*`.

/// Per-layer round-trip: N scatter calls (operational → universal,
/// one layer each) then N gather calls (universal → operational, one
/// layer each). Must reproduce src byte-for-byte. A stride-math bug
/// in the universal-side kernel (per-head stride driven by `nl` instead
/// of `nl_full`) would scribble heads into each other's slots; the
/// reverse gather then reads the wrong data and the checksum diverges.
async fn assert_layer_range_round_trip(operational: KvBlockLayout) -> Result<()> {
    let agent = build_agent_for_kinds(&[StorageKind::Device(0)])?;
    // dtype-None matches production layout builders (vLLM connector +
    // engine sites); the c6 effective_dtype derive picks BF16 at the
    // catalog gate. Mismatched num_blocks (src=dst=4, mid=8) mirrors
    // production G1↔G2 tier sizes, same as the c6 auto_promote
    // reproducers above.
    let src = build_fc_dtype_none(agent.clone(), operational, 4);
    let mid = build_fc_dtype_none(agent.clone(), KvBlockLayout::Universal, 8);
    let dst = build_fc_dtype_none(agent.clone(), operational, 4);

    let src_blocks = vec![0, 1];
    let mid_blocks = vec![2, 3];
    let dst_blocks = vec![0, 1];

    let src_checksums = fill_and_checksum(&src, &src_blocks, FillPattern::Sequential)?;
    let ctx = create_transfer_context(agent, None).unwrap();
    let nl = src.layout().config().num_layers;

    // Forward: operational → universal, one layer at a time.
    for layer in 0..nl {
        let options = TransferOptionsInternal::builder()
            .layer_range(layer..layer + 1)
            .build()?;
        let forward =
            execute_transfer(&src, &mid, &src_blocks, &mid_blocks, options, ctx.context())?;
        forward.await?;
    }

    // Reverse: universal → operational, one layer at a time. Must
    // reproduce the full src pattern on dst across all layers.
    for layer in 0..nl {
        let options = TransferOptionsInternal::builder()
            .layer_range(layer..layer + 1)
            .build()?;
        let reverse =
            execute_transfer(&mid, &dst, &mid_blocks, &dst_blocks, options, ctx.context())?;
        reverse.await?;
    }

    verify_checksums_by_position(&src_checksums, &src_blocks, &dst, &dst_blocks)?;
    Ok(())
}

#[tokio::test]
async fn per_layer_round_trip_nhd_via_universal() -> Result<()> {
    skip_if_stubs_and_device!(StorageKind::Device(0));
    gpu_serial!();
    assert_layer_range_round_trip(KvBlockLayout::OperationalNHD).await
}

#[tokio::test]
async fn per_layer_round_trip_hnd_via_universal() -> Result<()> {
    skip_if_stubs_and_device!(StorageKind::Device(0));
    gpu_serial!();
    assert_layer_range_round_trip(KvBlockLayout::OperationalHND).await
}

/// NHD ↔ HND transpose with `layer_range`. Both sides are operational
/// so the kernel doesn't need `nl_full` / `nl_offset` — the chunk
/// pointer table already encodes the layer slice — but the planner-
/// level wiring still has to walk only the slice's chunks. This pins
/// that contract on the op↔op path.
async fn assert_nhd_hnd_layer_range_round_trip(src_layout: KvBlockLayout) -> Result<()> {
    let other = match src_layout {
        KvBlockLayout::OperationalNHD => KvBlockLayout::OperationalHND,
        KvBlockLayout::OperationalHND => KvBlockLayout::OperationalNHD,
        _ => panic!("transpose layer_range test: src must be NHD or HND, got {src_layout:?}"),
    };
    let agent = build_agent_for_kinds(&[StorageKind::Device(0)])?;
    let src = build_fc_with_block_layout(agent.clone(), src_layout, 4);
    let mid = build_fc_with_block_layout(agent.clone(), other, 4);
    let dst = build_fc_with_block_layout(agent.clone(), src_layout, 4);

    let src_blocks = vec![0, 1];
    let mid_blocks = vec![2, 3];
    let dst_blocks = vec![0, 1];

    let src_checksums = fill_and_checksum(&src, &src_blocks, FillPattern::Sequential)?;
    let ctx = create_transfer_context(agent, None).unwrap();
    let nl = src.layout().config().num_layers;

    for layer in 0..nl {
        let options = TransferOptionsInternal::builder()
            .layer_range(layer..layer + 1)
            .use_planner(true)
            .build()?;
        let forward =
            execute_transfer(&src, &mid, &src_blocks, &mid_blocks, options, ctx.context())?;
        forward.await?;
    }
    for layer in 0..nl {
        let options = TransferOptionsInternal::builder()
            .layer_range(layer..layer + 1)
            .use_planner(true)
            .build()?;
        let reverse =
            execute_transfer(&mid, &dst, &mid_blocks, &dst_blocks, options, ctx.context())?;
        reverse.await?;
    }

    verify_checksums_by_position(&src_checksums, &src_blocks, &dst, &dst_blocks)?;
    Ok(())
}

#[tokio::test]
async fn per_layer_round_trip_nhd_to_hnd() -> Result<()> {
    skip_if_stubs_and_device!(StorageKind::Device(0));
    gpu_serial!();
    assert_nhd_hnd_layer_range_round_trip(KvBlockLayout::OperationalNHD).await
}

#[tokio::test]
async fn per_layer_round_trip_hnd_to_nhd() -> Result<()> {
    skip_if_stubs_and_device!(StorageKind::Device(0));
    gpu_serial!();
    assert_nhd_hnd_layer_range_round_trip(KvBlockLayout::OperationalHND).await
}

/// `layer_range` with `requires_transform = true` and `use_planner = false`
/// must auto-promote to the planner path under default options. Pins the
/// production call shape: `intra_pass_offload` sets `layer_range` and
/// `cuda_stream` but not `use_planner`, and the executor must transparently
/// route the call to the kernel catalog.
#[tokio::test]
async fn auto_promote_g1_nhd_to_g2_univ_with_layer_range() -> Result<()> {
    skip_if_stubs_and_device!(StorageKind::Device(0));
    gpu_serial!();

    let agent = build_agent_for_kinds(&[StorageKind::Device(0)])?;
    let src = build_fc_dtype_none(agent.clone(), KvBlockLayout::OperationalNHD, 4);
    let dst = build_fc_dtype_none(agent.clone(), KvBlockLayout::Universal, 8);

    let src_blocks = vec![0];
    let dst_blocks = vec![0];

    let ctx = create_transfer_context(agent, None).unwrap();

    // Default options + layer_range = the exact intra-pass-offload call shape
    // (use_planner stays false; auto-promote flips it inside execute_transfer).
    let options = TransferOptionsInternal::builder()
        .layer_range(0..1)
        .build()?;
    assert!(
        !options.use_planner,
        "regression guard: builder must not pre-set use_planner — the executor's \
         auto-promote on requires_transform is the load-bearing mechanism",
    );
    let forward = execute_transfer(&src, &dst, &src_blocks, &dst_blocks, options, ctx.context())?;
    forward.await?;
    Ok(())
}
