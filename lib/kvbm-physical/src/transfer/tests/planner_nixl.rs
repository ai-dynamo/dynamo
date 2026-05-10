// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! 2-agent same-process NIXL pull/push tests for `use_planner = true`.
//!
//! Each test builds two `NixlAgent`s in the same process — one
//! "owner" agent (the data origin) and one "remote" agent (the
//! other side). NIXL metadata is exchanged in both directions via
//! `agent.get_local_md()` / `agent.load_remote_md(...)` so either
//! side can address the other's registered memory. The local side's
//! [`crate::transfer::context::TransferContext`] then drives
//! [`crate::transfer::executor::execute_transfer`] in either pull
//! (Read) or push (Write) direction.
//!
//! - Pull (`*_pull_*`): `TransferContext` anchored on the puller;
//!   src is owned by the remote owner agent; dst is on the puller.
//!   `select_strategy` resolves to `NixlReadFlipped`.
//! - Push (`*_push_*`): `TransferContext` anchored on the pusher;
//!   src is owned by the local pusher agent; dst is on the remote
//!   receiver. `select_strategy` resolves to `NixlWrite`.
//!
//! Both directions run twice — once with `use_planner = false` (legacy
//! `NixlTransferBuilder`) and once with `use_planner = true` (new
//! `execute_planner_nixl_transfer`) — and assert: (1) both legacy and
//! planner destinations match the source by position (catches
//! both-paths-wrong collusion that pure legacy-vs-planner equality
//! would miss); (2) the resolved strategy matches the expected
//! variant before the transfer runs.
//!
//! Per-test agent name suffixes (see [`agent_pair_names`]) avoid
//! collisions between concurrently-running tests.
//!
//! Gated on UCX backend availability — without UCX no NIXL transfer
//! can complete and the test silently skips. CUDA stubs also skip
//! (Device(0) requires real CUDA).

use std::sync::atomic::{AtomicU64, Ordering};

use anyhow::{Result, bail};

use super::gate::nixl_serial;
use super::local_transfers::{LayoutType, is_nixl_backend_available};
use super::*;
use crate::layout::KvBlockLayout;
use crate::transfer::executor::{TransferOptionsInternal, execute_transfer};
use crate::transfer::strategy::{TransferPlan, TransferStrategy, select_strategy};

/// Direction of the NIXL transfer under test.
///
/// Determines (a) which agent anchors the `TransferContext`, (b) which
/// agent owns the source memory vs the destination memory, and (c) the
/// strategy `select_strategy` is expected to resolve to.
#[derive(Clone, Copy, Debug)]
enum Direction {
    /// Puller (dst-side) anchors the ctx; src lives on the remote
    /// owner. `select_strategy` ⇒ `NixlReadFlipped`.
    Pull,
    /// Pusher (src-side) anchors the ctx; dst lives on the remote
    /// receiver. `select_strategy` ⇒ `NixlWrite`.
    Push,
}

impl Direction {
    fn expected_strategy(&self) -> TransferStrategy {
        match self {
            Direction::Pull => TransferStrategy::NixlReadFlipped,
            Direction::Push => TransferStrategy::NixlWrite,
        }
    }

    fn label(&self) -> &'static str {
        match self {
            Direction::Pull => "pull",
            Direction::Push => "push",
        }
    }
}

/// Build a NIXL agent with the UCX backend enabled, or `None` if UCX
/// is unavailable on this host (which gates the entire test).
fn build_ucx_agent(name: &str) -> Result<Option<NixlAgent>> {
    let mut agent = NixlAgent::new(name)?;
    if agent.add_backend("UCX").is_err() {
        return Ok(None);
    }
    Ok(Some(agent))
}

/// Generate a fresh `(owner_name, remote_name)` pair for a test.
///
/// A monotonic atomic counter ensures concurrent tests don't collide
/// on agent names — each call returns a strictly-increasing suffix.
/// `role` distinguishes pull vs push runs (and legacy vs planner
/// runs within those) so failures point at the specific test.
fn agent_pair_names(role: &str) -> (String, String) {
    static SUFFIX: AtomicU64 = AtomicU64::new(0);
    let n = SUFFIX.fetch_add(1, Ordering::Relaxed);
    (
        format!("planner-nixl-{role}-owner-{n}"),
        format!("planner-nixl-{role}-remote-{n}"),
    )
}

/// Build a `Device(0)` `PhysicalLayout` with `KvBlockLayout::OperationalNHD`
/// on the given agent. Mirrors `planner_path::build_layout_for_planner`
/// but tied to the caller's specific agent (each agent needs its own
/// layout, and the layout's `nixl_metadata().agent_name()` must match
/// the owning agent's name for locality logic to work).
fn build_layout_on_agent(
    agent: NixlAgent,
    layout_type: LayoutType,
    num_blocks: usize,
) -> Result<PhysicalLayout> {
    let config = standard_config(num_blocks);
    let builder = PhysicalLayout::builder(agent)
        .with_config(config)
        .with_block_layout(KvBlockLayout::OperationalNHD);
    let typed = match layout_type {
        LayoutType::FC => builder.fully_contiguous(),
        LayoutType::LW => builder.layer_separate(BlockDimension::BlockIsFirstDim),
    };
    Ok(typed.allocate_device(0).build()?)
}

/// Run one cross-agent NIXL transfer and verify the destination
/// matches the source by position.
///
/// Returns the destination checksums (under whichever planner setting
/// was passed) so the caller can compare with the other path's
/// destination checksums for the side-by-side equality check.
///
/// `role_label` is folded into the agent names so concurrent runs
/// don't collide and failures point at the specific scenario.
async fn run_one_transfer(
    direction: Direction,
    layout_type: LayoutType,
    use_planner: bool,
    role_label: &str,
) -> Result<HashMap<BlockId, BlockChecksum>> {
    let (owner_name, remote_name) = agent_pair_names(role_label);
    let owner = build_ucx_agent(&owner_name)?
        .expect("UCX backend missing — caller should have skipped");
    let remote = build_ucx_agent(&remote_name)?
        .expect("UCX backend missing — caller should have skipped");

    // owner holds src memory; remote holds dst memory. The
    // TransferContext is anchored on whichever side `Direction`
    // designates as local.
    let src = build_layout_on_agent(owner.clone(), layout_type.clone(), 4)?;
    let dst = build_layout_on_agent(remote.clone(), layout_type, 4)?;

    // Cross-load metadata in both directions so either side can drive
    // the transfer. (Pull only strictly needs remote→puller and Push
    // only needs owner→pusher, but loading both keeps the helper
    // direction-agnostic.)
    let owner_md = owner
        .get_local_md()
        .map_err(|e| anyhow::anyhow!("owner.get_local_md: {:?}", e))?;
    let remote_md = remote
        .get_local_md()
        .map_err(|e| anyhow::anyhow!("remote.get_local_md: {:?}", e))?;
    owner
        .load_remote_md(&remote_md)
        .map_err(|e| anyhow::anyhow!("owner.load_remote_md: {:?}", e))?;
    remote
        .load_remote_md(&owner_md)
        .map_err(|e| anyhow::anyhow!("remote.load_remote_md: {:?}", e))?;

    let src_blocks = vec![0, 1];
    let dst_blocks = vec![2, 3];

    // Fill src with a deterministic pattern; the position-by-position
    // verification below confirms each dst block carries exactly the
    // pattern of its position-matched src block.
    let src_checksums = fill_and_checksum(&src, &src_blocks, FillPattern::Sequential)?;

    // GPU RDMA capability is required so device-to-device cross-agent
    // transfers don't get rejected as "GPU RDMA is disabled".
    let caps = crate::transfer::TransferCapabilities::default().with_gpu_rdma(true);
    let ctx_agent = match direction {
        Direction::Pull => remote,
        Direction::Push => owner,
    };
    let ctx = create_transfer_context(ctx_agent, Some(caps))?;

    // Strategy must resolve to the expected variant — catches drift
    // in select_strategy that would silently move the test off its
    // target path (e.g. NixlReadFlipped → NixlRead).
    let plan = select_strategy(&src, &dst, ctx.context())?;
    let resolved = match plan {
        TransferPlan::Direct(s) => s,
        other => bail!(
            "{:?}: expected TransferPlan::Direct, got {other:?}",
            direction
        ),
    };
    assert_eq!(
        resolved,
        direction.expected_strategy(),
        "{:?}: expected strategy {:?}, got {resolved:?}",
        direction,
        direction.expected_strategy(),
    );

    let options = TransferOptionsInternal::builder()
        .use_planner(use_planner)
        .build()?;
    let notification =
        execute_transfer(&src, &dst, &src_blocks, &dst_blocks, options, ctx.context())?;
    notification.await?;

    // Verify dst content matches src by position. This catches
    // both-paths-wrong cases (e.g. block swap, partial copy, identical
    // legacy/planner descriptor mistake) that pure legacy-vs-planner
    // equality would miss — see review-finding #2 on PR-5.6.
    verify_checksums_by_position(&src_checksums, &src_blocks, &dst, &dst_blocks)?;

    compute_block_checksums(&dst, &dst_blocks)
}

/// Side-by-side equivalence: run the same transfer twice (legacy +
/// planner), assert each destination matches the source by position,
/// and assert the legacy and planner destinations agree pairwise.
async fn assert_planner_matches_legacy(
    direction: Direction,
    layout_type: LayoutType,
    test_label: &str,
) -> Result<()> {
    let legacy_role = format!("{}-{}-legacy", direction.label(), test_label);
    let planner_role = format!("{}-{}-planner", direction.label(), test_label);

    let legacy = run_one_transfer(direction, layout_type.clone(), false, &legacy_role).await?;
    let planner = run_one_transfer(direction, layout_type, true, &planner_role).await?;

    assert_eq!(
        legacy.len(),
        planner.len(),
        "{:?}: destination block count disagreement: legacy={} vs planner={}",
        direction,
        legacy.len(),
        planner.len()
    );
    for (id, legacy_sum) in &legacy {
        let planner_sum = planner
            .get(id)
            .unwrap_or_else(|| panic!("missing planner checksum for dst block {id}"));
        assert_eq!(
            legacy_sum, planner_sum,
            "{:?}: planner / legacy NIXL checksum disagreement on dst block {id}: \
             legacy={legacy_sum} vs planner={planner_sum}",
            direction,
        );
    }
    Ok(())
}

// ──────────── pull (NixlReadFlipped) ────────────

#[tokio::test]
async fn use_planner_nixl_pull_matches_legacy_fc_fc() -> Result<()> {
    skip_if_stubs_and_device!(StorageKind::Device(0));
    if !is_nixl_backend_available("UCX") {
        eprintln!("Skipping NIXL planner test — UCX backend unavailable");
        return Ok(());
    }
    nixl_serial!();
    assert_planner_matches_legacy(Direction::Pull, LayoutType::FC, "fc_fc").await
}

#[tokio::test]
async fn use_planner_nixl_pull_matches_legacy_lw_lw() -> Result<()> {
    skip_if_stubs_and_device!(StorageKind::Device(0));
    if !is_nixl_backend_available("UCX") {
        eprintln!("Skipping NIXL planner test — UCX backend unavailable");
        return Ok(());
    }
    nixl_serial!();
    assert_planner_matches_legacy(Direction::Pull, LayoutType::LW, "lw_lw").await
}

// ──────────── push (NixlWrite) ────────────

#[tokio::test]
async fn use_planner_nixl_push_matches_legacy_fc_fc() -> Result<()> {
    skip_if_stubs_and_device!(StorageKind::Device(0));
    if !is_nixl_backend_available("UCX") {
        eprintln!("Skipping NIXL planner test — UCX backend unavailable");
        return Ok(());
    }
    nixl_serial!();
    assert_planner_matches_legacy(Direction::Push, LayoutType::FC, "fc_fc").await
}

#[tokio::test]
async fn use_planner_nixl_push_matches_legacy_lw_lw() -> Result<()> {
    skip_if_stubs_and_device!(StorageKind::Device(0));
    if !is_nixl_backend_available("UCX") {
        eprintln!("Skipping NIXL planner test — UCX backend unavailable");
        return Ok(());
    }
    nixl_serial!();
    assert_planner_matches_legacy(Direction::Push, LayoutType::LW, "lw_lw").await
}
