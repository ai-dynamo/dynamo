// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! 2-agent same-process NIXL pull tests for `use_planner = true`.
//!
//! Each test builds two `NixlAgent`s in the same process — one
//! "owner" agent (the source) and one "puller" agent (the
//! destination). Each agent registers its own `PhysicalLayout`. NIXL
//! metadata is exchanged via `agent.get_local_md()` /
//! `agent.load_remote_md(...)` so the puller can address the owner's
//! memory. The puller's [`crate::manager::TransferManager`] then
//! executes a `NixlReadFlipped` transfer with `src` on the owner side
//! and `dst` on the puller side; the side-by-side check runs the same
//! transfer once with `use_planner = false` (legacy `NixlTransferBuilder`)
//! and once with `use_planner = true` (new `execute_planner_nixl_transfer`)
//! and asserts both produce byte-identical destination checksums.
//!
//! Gated on UCX backend availability — without UCX no NIXL transfer
//! can complete and the test silently skips. CUDA stubs also skip
//! (Device(0) requires real CUDA).

use anyhow::Result;

use super::local_transfers::{LayoutType, is_nixl_backend_available};
use super::*;
use crate::layout::KvBlockLayout;
use crate::transfer::executor::{TransferOptionsInternal, execute_transfer};

/// Build a NIXL agent with the UCX backend enabled, or `None` if UCX
/// is unavailable on this host (which gates the entire test).
fn build_ucx_agent(name: &str) -> Result<Option<NixlAgent>> {
    let mut agent = NixlAgent::new(name)?;
    if agent.add_backend("UCX").is_err() {
        return Ok(None);
    }
    Ok(Some(agent))
}

/// Build a `Device(0)` `PhysicalLayout` with `KvBlockLayout::OperationalNHD`
/// on the given agent. Mirrors `planner_path::build_layout_for_planner`
/// but tied to the caller's specific agent (each owner / puller agent
/// needs its own layout, and the layout's `nixl_metadata().agent_name()`
/// must match the owning agent's name for locality logic to work).
fn build_layout_on_agent(
    agent: NixlAgent,
    layout_type: LayoutType,
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
    typed.allocate_device(0).build().unwrap()
}

/// Two-agent pull: agent B reads from agent A.
///
/// Returns the destination checksums (under whichever planner setting
/// was passed) so the caller can compare with the source / against
/// the other path.
async fn run_pull_with_planner(
    layout_type: LayoutType,
    use_planner: bool,
) -> Result<HashMap<BlockId, BlockChecksum>> {
    let owner = build_ucx_agent("planner-nixl-owner")?
        .expect("UCX backend missing — caller should have skipped");
    let puller = build_ucx_agent("planner-nixl-puller")?
        .expect("UCX backend missing — caller should have skipped");

    let src = build_layout_on_agent(owner.clone(), layout_type.clone(), 4);
    let dst = build_layout_on_agent(puller.clone(), layout_type, 4);

    // Cross-load NIXL metadata so the puller can address the owner's
    // registered memory.
    let owner_md = owner
        .get_local_md()
        .map_err(|e| anyhow::anyhow!("owner.get_local_md: {:?}", e))?;
    puller
        .load_remote_md(&owner_md)
        .map_err(|e| anyhow::anyhow!("puller.load_remote_md: {:?}", e))?;

    let src_blocks = vec![0, 1];
    let dst_blocks = vec![2, 3];

    // Fill the owner's source first (host-side write into device
    // memory via the underlying registration path); checksums on the
    // owner side don't drive the test, but we want a deterministic
    // pattern.
    fill_blocks(&src, &src_blocks, FillPattern::Sequential)?;

    // Build a TransferContext anchored on the puller — its NIXL
    // agent is what `select_strategy` uses to determine locality
    // (src.agent != ctx.agent ⇒ remote; dst.agent == ctx.agent ⇒
    // local). The resulting strategy is `NixlReadFlipped`.
    //
    // GPU RDMA must be enabled in capabilities so device-to-device
    // cross-agent transfers don't get rejected as "GPU RDMA is
    // disabled" — the puller's destination is on Device(0) and so
    // is the owner's source, and the pull crosses agents.
    let caps = crate::transfer::TransferCapabilities::default().with_gpu_rdma(true);
    let ctx = create_transfer_context(puller, Some(caps)).unwrap();

    let options = TransferOptionsInternal::builder()
        .use_planner(use_planner)
        .build()?;
    let notification =
        execute_transfer(&src, &dst, &src_blocks, &dst_blocks, options, ctx.context())?;
    notification.await?;

    compute_block_checksums(&dst, &dst_blocks)
}

/// Side-by-side equivalence: run the pull twice, compare checksums.
async fn assert_planner_matches_legacy_nixl_pull(layout_type: LayoutType) -> Result<()> {
    let legacy = run_pull_with_planner(layout_type.clone(), false).await?;
    let planner = run_pull_with_planner(layout_type, true).await?;
    assert_eq!(
        legacy.len(),
        planner.len(),
        "destination block count disagreement: legacy={} vs planner={}",
        legacy.len(),
        planner.len()
    );
    for (id, legacy_sum) in &legacy {
        let planner_sum = planner.get(id).unwrap_or_else(|| {
            panic!("missing planner checksum for dst block {id}");
        });
        assert_eq!(
            legacy_sum, planner_sum,
            "planner / legacy NIXL checksum disagreement on dst block {id}: \
             legacy={legacy_sum} vs planner={planner_sum}"
        );
    }
    Ok(())
}

#[tokio::test]
async fn use_planner_nixl_pull_matches_legacy_fc_fc() -> Result<()> {
    skip_if_stubs_and_device!(StorageKind::Device(0));
    if !is_nixl_backend_available("UCX") {
        eprintln!("Skipping NIXL planner test — UCX backend unavailable");
        return Ok(());
    }
    assert_planner_matches_legacy_nixl_pull(LayoutType::FC).await
}

#[tokio::test]
async fn use_planner_nixl_pull_matches_legacy_lw_lw() -> Result<()> {
    skip_if_stubs_and_device!(StorageKind::Device(0));
    if !is_nixl_backend_available("UCX") {
        eprintln!("Skipping NIXL planner test — UCX backend unavailable");
        return Ok(());
    }
    assert_planner_matches_legacy_nixl_pull(LayoutType::LW).await
}
