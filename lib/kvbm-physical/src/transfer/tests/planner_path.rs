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

use super::local_transfers::{LayoutType, build_agent_for_kinds};
use super::*;
use crate::layout::KvBlockLayout;
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
    assert_planner_matches_legacy(LayoutType::FC, src_kind, LayoutType::FC, dst_kind).await
}

#[tokio::test]
async fn use_planner_matches_legacy_lw_lw_d2d() -> Result<()> {
    let src_kind = StorageKind::Device(0);
    let dst_kind = StorageKind::Device(0);
    skip_if_stubs_and_device!(src_kind, dst_kind);
    assert_planner_matches_legacy(LayoutType::LW, src_kind, LayoutType::LW, dst_kind).await
}

#[tokio::test]
async fn use_planner_matches_legacy_fc_lw_d2d() -> Result<()> {
    let src_kind = StorageKind::Device(0);
    let dst_kind = StorageKind::Device(0);
    skip_if_stubs_and_device!(src_kind, dst_kind);
    assert_planner_matches_legacy(LayoutType::FC, src_kind, LayoutType::LW, dst_kind).await
}

#[tokio::test]
async fn use_planner_matches_legacy_fc_fc_h2d() -> Result<()> {
    let src_kind = StorageKind::Pinned;
    let dst_kind = StorageKind::Device(0);
    skip_if_stubs_and_device!(src_kind, dst_kind);
    assert_planner_matches_legacy(LayoutType::FC, src_kind, LayoutType::FC, dst_kind).await
}

#[tokio::test]
async fn use_planner_matches_legacy_fc_fc_d2h() -> Result<()> {
    let src_kind = StorageKind::Device(0);
    let dst_kind = StorageKind::Pinned;
    skip_if_stubs_and_device!(src_kind, dst_kind);
    assert_planner_matches_legacy(LayoutType::FC, src_kind, LayoutType::FC, dst_kind).await
}
