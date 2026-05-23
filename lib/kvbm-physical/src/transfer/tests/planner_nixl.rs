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
    typed.allocate_device(0).build()
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
    let owner =
        build_ucx_agent(&owner_name)?.expect("UCX backend missing — caller should have skipped");
    let remote =
        build_ucx_agent(&remote_name)?.expect("UCX backend missing — caller should have skipped");

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

// ──────────── PR-6.2: staged operational ↔ universal transforms ────────────

/// Build an FC `PhysicalLayout` on `Device(0)` with an explicit
/// `KvBlockLayout`, attached to the given agent. PR-6.2 staged
/// transforms need the bounce buffer registered on the local agent
/// AND a configurable `KvBlockLayout` for src/dst.
fn build_fc_with_block_layout_on_agent(
    agent: NixlAgent,
    block_layout: KvBlockLayout,
    num_blocks: usize,
) -> Result<PhysicalLayout> {
    let config = standard_config(num_blocks);
    PhysicalLayout::builder(agent)
        .with_config(config)
        .with_block_layout(block_layout)
        .fully_contiguous()
        .allocate_device(0)
        .build()
}

/// Verify the Pull-direction staged transform via round-trip:
///
/// 1. Fill `src(operational, owner)` with a deterministic pattern;
///    record its content-based checksums.
/// 2. Staged Pull `src → mid(Universal, remote)` (the path under
///    test — kernel runs on remote after raw NIXL pull).
/// 3. Local Cuda* transfer `mid → final(operational, remote)` —
///    PR-6.1's catalog dispatches `block_from_universal` locally.
///    This is the "inverse" of step 2's kernel, validated by PR-6.1's
///    own round-trip test.
/// 4. The final content-based checksums on remote must match src's
///    by position. Mismatch ⇒ staged Pull produced wrong bytes.
async fn assert_staged_pull_round_trip(operational: KvBlockLayout) -> Result<()> {
    use crate::transfer::BounceBufferInternal;

    let role = format!("pull-staged-{:?}", operational);
    let (owner_name, remote_name) = agent_pair_names(&role);
    let owner =
        build_ucx_agent(&owner_name)?.expect("UCX backend missing — caller should have skipped");
    let remote =
        build_ucx_agent(&remote_name)?.expect("UCX backend missing — caller should have skipped");

    // src on owner (operational), mid on remote (universal),
    // final on remote (operational), bounce on remote (operational
    // — matches src for the raw NIXL leg).
    let src = build_fc_with_block_layout_on_agent(owner.clone(), operational, 4)?;
    let mid = build_fc_with_block_layout_on_agent(remote.clone(), KvBlockLayout::Universal, 4)?;
    let final_dst = build_fc_with_block_layout_on_agent(remote.clone(), operational, 4)?;
    let bounce_layout = build_fc_with_block_layout_on_agent(remote.clone(), operational, 4)?;

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
    let mid_blocks = vec![2, 3];
    let final_blocks = vec![0, 1];
    let bounce_blocks = vec![2, 3];

    let src_checksums = fill_and_checksum(&src, &src_blocks, FillPattern::Sequential)?;

    let caps = crate::transfer::TransferCapabilities::default().with_gpu_rdma(true);
    let ctx = create_transfer_context(remote, Some(caps))?;

    // Stage 1: staged Pull through bounce + kernel.
    let bounce = BounceBufferInternal::from_layout(bounce_layout, bounce_blocks);
    let staged_options = TransferOptionsInternal::builder()
        .use_planner(true)
        .bounce_buffer(bounce)
        .build()?;
    let staged = execute_transfer(
        &src,
        &mid,
        &src_blocks,
        &mid_blocks,
        staged_options,
        ctx.context(),
    )?;
    staged.await?;

    // Stage 2: local Cuda* mid → final (PR-6.1 catalog dispatch on
    // the local agent).
    let local_options = TransferOptionsInternal::builder()
        .use_planner(true)
        .build()?;
    let local = execute_transfer(
        &mid,
        &final_dst,
        &mid_blocks,
        &final_blocks,
        local_options,
        ctx.context(),
    )?;
    local.await?;

    verify_checksums_by_position(&src_checksums, &src_blocks, &final_dst, &final_blocks)?;
    Ok(())
}

/// Mirror for Push direction: round-trip
/// `src(operational, owner) → mid(Universal, owner)` via the
/// local Cuda* catalog, then staged Push
/// `mid → final(operational, remote)`. The final's checksums on
/// remote must match src's by position.
async fn assert_staged_push_round_trip(operational: KvBlockLayout) -> Result<()> {
    use crate::transfer::BounceBufferInternal;

    let role = format!("push-staged-{:?}", operational);
    let (owner_name, remote_name) = agent_pair_names(&role);
    let owner =
        build_ucx_agent(&owner_name)?.expect("UCX backend missing — caller should have skipped");
    let remote =
        build_ucx_agent(&remote_name)?.expect("UCX backend missing — caller should have skipped");

    let src = build_fc_with_block_layout_on_agent(owner.clone(), operational, 4)?;
    let mid = build_fc_with_block_layout_on_agent(owner.clone(), KvBlockLayout::Universal, 4)?;
    let final_dst = build_fc_with_block_layout_on_agent(remote.clone(), operational, 4)?;
    let bounce_layout = build_fc_with_block_layout_on_agent(owner.clone(), operational, 4)?;

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
    let mid_blocks = vec![2, 3];
    let final_blocks = vec![0, 1];
    let bounce_blocks = vec![2, 3];

    let src_checksums = fill_and_checksum(&src, &src_blocks, FillPattern::Sequential)?;

    let caps = crate::transfer::TransferCapabilities::default().with_gpu_rdma(true);
    let ctx = create_transfer_context(owner, Some(caps))?;

    // Stage 1: local Cuda* src → mid (PR-6.1).
    let local_options = TransferOptionsInternal::builder()
        .use_planner(true)
        .build()?;
    let local = execute_transfer(
        &src,
        &mid,
        &src_blocks,
        &mid_blocks,
        local_options,
        ctx.context(),
    )?;
    local.await?;

    // Stage 2: staged Push through kernel + bounce.
    let bounce = BounceBufferInternal::from_layout(bounce_layout, bounce_blocks);
    let staged_options = TransferOptionsInternal::builder()
        .use_planner(true)
        .bounce_buffer(bounce)
        .build()?;
    let staged = execute_transfer(
        &mid,
        &final_dst,
        &mid_blocks,
        &final_blocks,
        staged_options,
        ctx.context(),
    )?;
    staged.await?;

    verify_checksums_by_position(&src_checksums, &src_blocks, &final_dst, &final_blocks)?;
    Ok(())
}

#[tokio::test]
async fn use_planner_nixl_pull_transform_nhd_to_universal() -> Result<()> {
    skip_if_stubs_and_device!(StorageKind::Device(0));
    if !is_nixl_backend_available("UCX") {
        eprintln!("Skipping NIXL planner test — UCX backend unavailable");
        return Ok(());
    }
    nixl_serial!();
    assert_staged_pull_round_trip(KvBlockLayout::OperationalNHD).await
}

#[tokio::test]
async fn use_planner_nixl_pull_transform_hnd_to_universal() -> Result<()> {
    skip_if_stubs_and_device!(StorageKind::Device(0));
    if !is_nixl_backend_available("UCX") {
        eprintln!("Skipping NIXL planner test — UCX backend unavailable");
        return Ok(());
    }
    nixl_serial!();
    assert_staged_pull_round_trip(KvBlockLayout::OperationalHND).await
}

#[tokio::test]
async fn use_planner_nixl_push_transform_universal_to_nhd() -> Result<()> {
    skip_if_stubs_and_device!(StorageKind::Device(0));
    if !is_nixl_backend_available("UCX") {
        eprintln!("Skipping NIXL planner test — UCX backend unavailable");
        return Ok(());
    }
    nixl_serial!();
    assert_staged_push_round_trip(KvBlockLayout::OperationalNHD).await
}

#[tokio::test]
async fn use_planner_nixl_push_transform_universal_to_hnd() -> Result<()> {
    skip_if_stubs_and_device!(StorageKind::Device(0));
    if !is_nixl_backend_available("UCX") {
        eprintln!("Skipping NIXL planner test — UCX backend unavailable");
        return Ok(());
    }
    nixl_serial!();
    assert_staged_push_round_trip(KvBlockLayout::OperationalHND).await
}

// ──────────── PR-6.3: staged NHD ↔ HND transforms ────────────
//
// Wiring tests for the staged NIXL path through the new transpose
// kernel. Verification uses the same `nhd_hnd_transpose` symbol in
// the inverse direction — kernel correctness against ground truth
// is exercised by `kvbm-kernels`'s `kernel_roundtrip` suite.

/// Round-trip via staged Pull `src(layout, owner) → mid(other, remote)`,
/// then local Cuda* `mid(other, remote) → final(layout, remote)`.
/// `final`'s checksums on remote must match `src`'s by position.
async fn assert_staged_pull_nhd_hnd_round_trip(src_layout: KvBlockLayout) -> Result<()> {
    use crate::transfer::BounceBufferInternal;

    let other = match src_layout {
        KvBlockLayout::OperationalNHD => KvBlockLayout::OperationalHND,
        KvBlockLayout::OperationalHND => KvBlockLayout::OperationalNHD,
        _ => panic!(
            "assert_staged_pull_nhd_hnd_round_trip: src must be NHD or HND, got {src_layout:?}"
        ),
    };

    let role = format!("pull-staged-nhd-hnd-{:?}", src_layout);
    let (owner_name, remote_name) = agent_pair_names(&role);
    let owner =
        build_ucx_agent(&owner_name)?.expect("UCX backend missing — caller should have skipped");
    let remote =
        build_ucx_agent(&remote_name)?.expect("UCX backend missing — caller should have skipped");

    // src on owner (src_layout); mid on remote (other) — the staged
    // pull's destination; final on remote (src_layout) — the local
    // inverse target; bounce on remote (src_layout) — matches src kv
    // for the NIXL leg of the staged pull.
    let src = build_fc_with_block_layout_on_agent(owner.clone(), src_layout, 4)?;
    let mid = build_fc_with_block_layout_on_agent(remote.clone(), other, 4)?;
    let final_dst = build_fc_with_block_layout_on_agent(remote.clone(), src_layout, 4)?;
    let bounce_layout = build_fc_with_block_layout_on_agent(remote.clone(), src_layout, 4)?;

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
    let mid_blocks = vec![2, 3];
    let final_blocks = vec![0, 1];
    let bounce_blocks = vec![2, 3];

    let src_checksums = fill_and_checksum(&src, &src_blocks, FillPattern::Sequential)?;

    let caps = crate::transfer::TransferCapabilities::default().with_gpu_rdma(true);
    let ctx = create_transfer_context(remote, Some(caps))?;

    // Stage 1 (under test): staged Pull src → mid through bounce + kernel.
    let bounce = BounceBufferInternal::from_layout(bounce_layout, bounce_blocks);
    let staged_options = TransferOptionsInternal::builder()
        .use_planner(true)
        .bounce_buffer(bounce)
        .build()?;
    let staged = execute_transfer(
        &src,
        &mid,
        &src_blocks,
        &mid_blocks,
        staged_options,
        ctx.context(),
    )?;
    staged.await?;

    // Stage 2 (verification): local Cuda* mid → final via the same
    // kernel symbol in the inverse direction.
    let local_options = TransferOptionsInternal::builder()
        .use_planner(true)
        .build()?;
    let local = execute_transfer(
        &mid,
        &final_dst,
        &mid_blocks,
        &final_blocks,
        local_options,
        ctx.context(),
    )?;
    local.await?;

    verify_checksums_by_position(&src_checksums, &src_blocks, &final_dst, &final_blocks)?;
    Ok(())
}

/// Round-trip via staged Push `src(layout, owner) →
/// final_dst(other, remote)`, then local Cuda* `final_dst(other, remote)
/// → final2(layout, remote)`. `final2`'s checksums on remote must match
/// `src`'s by position.
///
/// **Key invariant**: src and final_dst differ in `KvBlockLayout`, so
/// `requires_transform` is true and the executor routes through
/// `dispatch_staged_nixl_transform`. With only two operational layouts
/// in the system (NHD, HND), introducing an `other`-layout intermediate
/// on the *same* agent before the cross-agent hop would collapse the
/// staged hop into a same-layout Direct push and silently bypass the
/// path under test — so the staged push runs straight from
/// src(layout, owner) to final_dst(other, remote).
async fn assert_staged_push_nhd_hnd_round_trip(src_layout: KvBlockLayout) -> Result<()> {
    use crate::transfer::BounceBufferInternal;

    let other = match src_layout {
        KvBlockLayout::OperationalNHD => KvBlockLayout::OperationalHND,
        KvBlockLayout::OperationalHND => KvBlockLayout::OperationalNHD,
        _ => panic!(
            "assert_staged_push_nhd_hnd_round_trip: src must be NHD or HND, got {src_layout:?}"
        ),
    };

    let role = format!("push-staged-nhd-hnd-{:?}", src_layout);
    let (owner_name, remote_name) = agent_pair_names(&role);
    let owner =
        build_ucx_agent(&owner_name)?.expect("UCX backend missing — caller should have skipped");
    let remote =
        build_ucx_agent(&remote_name)?.expect("UCX backend missing — caller should have skipped");

    // src on owner (src_layout); final_dst on remote (other) — the
    // staged push's destination, kv-mismatched from src so the
    // executor routes through dispatch_staged_nixl_transform; bounce
    // on owner (other) — matches dst kv for the NIXL leg of the
    // staged push; final2 on remote (src_layout) — local inverse
    // target for verification.
    let src = build_fc_with_block_layout_on_agent(owner.clone(), src_layout, 4)?;
    let final_dst = build_fc_with_block_layout_on_agent(remote.clone(), other, 4)?;
    let bounce_layout = build_fc_with_block_layout_on_agent(owner.clone(), other, 4)?;
    let final2 = build_fc_with_block_layout_on_agent(remote.clone(), src_layout, 4)?;

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
    let final_blocks = vec![0, 1];
    let bounce_blocks = vec![2, 3];
    let final2_blocks = vec![2, 3];

    let src_checksums = fill_and_checksum(&src, &src_blocks, FillPattern::Sequential)?;

    let caps = crate::transfer::TransferCapabilities::default().with_gpu_rdma(true);
    let ctx_owner = create_transfer_context(owner, Some(caps))?;
    let ctx_remote = create_transfer_context(remote, Some(caps))?;

    // Stage 1 (under test): staged Push src → final_dst through
    // kernel (src→bounce locally on owner) + NIXL push (bounce →
    // final_dst on remote).
    let bounce = BounceBufferInternal::from_layout(bounce_layout, bounce_blocks);
    let staged_options = TransferOptionsInternal::builder()
        .use_planner(true)
        .bounce_buffer(bounce)
        .build()?;
    let staged = execute_transfer(
        &src,
        &final_dst,
        &src_blocks,
        &final_blocks,
        staged_options,
        ctx_owner.context(),
    )?;
    staged.await?;

    // Stage 2 (verification): local Cuda* final_dst → final2 on
    // remote via the inverse direction of the same kernel symbol.
    let local_options = TransferOptionsInternal::builder()
        .use_planner(true)
        .build()?;
    let local = execute_transfer(
        &final_dst,
        &final2,
        &final_blocks,
        &final2_blocks,
        local_options,
        ctx_remote.context(),
    )?;
    local.await?;

    verify_checksums_by_position(&src_checksums, &src_blocks, &final2, &final2_blocks)?;
    Ok(())
}

#[tokio::test]
async fn use_planner_nixl_pull_transform_nhd_to_hnd() -> Result<()> {
    skip_if_stubs_and_device!(StorageKind::Device(0));
    if !is_nixl_backend_available("UCX") {
        eprintln!("Skipping NIXL planner test — UCX backend unavailable");
        return Ok(());
    }
    nixl_serial!();
    assert_staged_pull_nhd_hnd_round_trip(KvBlockLayout::OperationalNHD).await
}

#[tokio::test]
async fn use_planner_nixl_pull_transform_hnd_to_nhd() -> Result<()> {
    skip_if_stubs_and_device!(StorageKind::Device(0));
    if !is_nixl_backend_available("UCX") {
        eprintln!("Skipping NIXL planner test — UCX backend unavailable");
        return Ok(());
    }
    nixl_serial!();
    assert_staged_pull_nhd_hnd_round_trip(KvBlockLayout::OperationalHND).await
}

#[tokio::test]
async fn use_planner_nixl_push_transform_nhd_to_hnd() -> Result<()> {
    skip_if_stubs_and_device!(StorageKind::Device(0));
    if !is_nixl_backend_available("UCX") {
        eprintln!("Skipping NIXL planner test — UCX backend unavailable");
        return Ok(());
    }
    nixl_serial!();
    assert_staged_push_nhd_hnd_round_trip(KvBlockLayout::OperationalNHD).await
}

#[tokio::test]
async fn use_planner_nixl_push_transform_hnd_to_nhd() -> Result<()> {
    skip_if_stubs_and_device!(StorageKind::Device(0));
    if !is_nixl_backend_available("UCX") {
        eprintln!("Skipping NIXL planner test — UCX backend unavailable");
        return Ok(());
    }
    nixl_serial!();
    assert_staged_push_nhd_hnd_round_trip(KvBlockLayout::OperationalHND).await
}

/// Negative test: NIXL transform without a bounce buffer must error
/// at the entrypoint with a precise message naming `bounce_buffer`.
#[tokio::test]
async fn use_planner_nixl_transform_without_bounce_errors() -> Result<()> {
    skip_if_stubs_and_device!(StorageKind::Device(0));
    if !is_nixl_backend_available("UCX") {
        eprintln!("Skipping NIXL planner test — UCX backend unavailable");
        return Ok(());
    }
    nixl_serial!();

    let (owner_name, remote_name) = agent_pair_names("transform-no-bounce");
    let owner =
        build_ucx_agent(&owner_name)?.expect("UCX backend missing — caller should have skipped");
    let remote =
        build_ucx_agent(&remote_name)?.expect("UCX backend missing — caller should have skipped");

    let src = build_fc_with_block_layout_on_agent(owner.clone(), KvBlockLayout::OperationalNHD, 4)?;
    let dst = build_fc_with_block_layout_on_agent(remote.clone(), KvBlockLayout::Universal, 4)?;

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

    let _ = fill_and_checksum(&src, &[0, 1], FillPattern::Sequential)?;
    let caps = crate::transfer::TransferCapabilities::default().with_gpu_rdma(true);
    let ctx = create_transfer_context(remote, Some(caps))?;

    let options = TransferOptionsInternal::builder()
        .use_planner(true)
        .build()?;
    let result = execute_transfer(&src, &dst, &[0, 1], &[2, 3], options, ctx.context());
    let err = match result {
        Ok(_) => panic!("execute_transfer should bail without bounce_buffer for NIXL transform"),
        Err(e) => e,
    };
    let msg = format!("{err:#}");
    assert!(
        msg.contains("bounce_buffer") || msg.contains("Bounce") || msg.contains("bounce"),
        "expected bounce_buffer-related error, got: {msg}"
    );
    Ok(())
}
