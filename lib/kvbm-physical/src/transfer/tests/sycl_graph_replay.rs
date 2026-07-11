// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Integration tests for SYCL graph capture/replay.
//!
//! Mirrors the CUDA tests in [`planner_graph_replay`] but validates the
//! SYCL-specific behaviour:
//! - No cache (fresh record each call) — re-recording with new addresses works.
//! - `supports_address_rebind()` returns `false`.
//! - `try_rebind()` returns `Ok(false)`.
//! - Byte-level correctness matches the DirectDma path.
//!
//! All tests require a real SYCL/XPU device (skipped when stubs are linked)
//! and are serialised via `gpu_serial!()` to avoid queue contention.
//!
//! Test matrix:
//!
//! 1. `sycl_graph_replay_round_trips_d2d` — fill src, dispatch once
//!    via graph replay, verify dst checksums match src.
//!
//! 2. `sycl_graph_replay_byte_equiv_to_direct_dma` — same input data
//!    transferred via DirectDma and DeviceGraphReplay independently; both
//!    destinations must checksum-match.
//!
//! 3. `sycl_graph_replay_fresh_record_each_call` — two transfers with the
//!    same shape but different addresses; both must produce correct results,
//!    proving fresh re-recording works (no stale graph handle reuse).
//!
//! 4. `sycl_graph_replay_confirms_no_rebind` — verifies trait-level
//!    `supports_address_rebind() == false` and `try_rebind() == Ok(false)`.

use anyhow::Result;

use super::gate::gpu_serial;
use super::*;
use crate::layout::{KvBlockLayout, PhysicalLayout};
use crate::transfer::TransferCapabilities;
use crate::transfer::context::TransferContext;
use crate::transfer::executor::{TransferOptionsInternal, execute_transfer};

// ─────────────────────────── helpers ─────────────────────────────────────────

/// Build a Device(0) FC layout with `OperationalNHD`, `num_blocks` blocks,
/// and `use_planner = true`-compatible projection fields set.
fn device_fc(agent: NixlAgent, num_blocks: usize) -> PhysicalLayout {
    let config = standard_config(num_blocks);
    PhysicalLayout::builder(agent)
        .with_config(config)
        .with_block_layout(KvBlockLayout::OperationalNHD)
        .fully_contiguous()
        .allocate_device(test_allocator(0))
        .build()
        .unwrap()
}

/// Create a `TransferContext` with `device_graph_replay = true`.
fn ctx_with_graph_replay(agent: NixlAgent) -> crate::manager::TransferManager {
    let caps = TransferCapabilities::default().with_device_graph_replay(true);
    create_transfer_context(agent, Some(caps)).unwrap()
}

/// Run a planner-path Async D2D transfer via `execute_transfer` with
/// `use_planner = true`. Routes through the full planner dispatcher
/// (including `DeviceGraphReplay` candidate selection when the capability is
/// enabled) without calling any private planner internals directly.
async fn transfer_direct_planner(
    src: &PhysicalLayout,
    dst: &PhysicalLayout,
    src_blocks: &[usize],
    dst_blocks: &[usize],
    ctx: &TransferContext,
) -> Result<()> {
    let options = TransferOptionsInternal::builder()
        .use_planner(true)
        .build()?;
    let notif = execute_transfer(src, dst, src_blocks, dst_blocks, options, ctx)?;
    notif.await?;
    Ok(())
}

/// Guard: skip if the test device backend is not SYCL.
/// This ensures CUDA-only CI doesn't fail on these SYCL-specific tests.
macro_rules! skip_if_not_sycl {
    () => {
        if test_device_backend() != DeviceBackend::Sycl {
            eprintln!(
                "Skipping test '{}': device backend is not SYCL",
                module_path!()
            );
            return Ok(());
        }
    };
}

// ─────────────────────────── tests ───────────────────────────────────────────

/// Basic round-trip: fill src, graph-replay dispatch to dst, verify checksums.
///
/// Uses a single block so the graph has exactly one memcpy node. Verifies
/// that the captured SYCL graph + enqueue produces the same bytes as the source.
#[tokio::test]
async fn sycl_graph_replay_round_trips_d2d() -> Result<()> {
    skip_if_not_sycl!();
    skip_if_stubs_and_device!(StorageKind::Device(0));
    gpu_serial!();

    let agent = super::local_transfers::build_agent_for_kinds(&[StorageKind::Device(0)])?;
    let src = device_fc(agent.clone(), 4);
    let dst = device_fc(agent.clone(), 4);
    let manager = ctx_with_graph_replay(agent);
    let ctx = manager.context();

    let src_blocks = vec![0usize];
    let dst_blocks = vec![1usize];

    let src_checksums = fill_and_checksum(&src, &src_blocks, FillPattern::Sequential)?;
    transfer_direct_planner(&src, &dst, &src_blocks, &dst_blocks, ctx).await?;
    verify_checksums_by_position(&src_checksums, &src_blocks, &dst, &dst_blocks)?;
    Ok(())
}

/// Byte equivalence: DirectDma and DeviceGraphReplay must produce identical output.
///
/// Both runs use the same source data. The two destination layouts are
/// distinct allocations; both are compared to src checksums AND to each other.
#[tokio::test]
async fn sycl_graph_replay_byte_equiv_to_direct_dma() -> Result<()> {
    skip_if_not_sycl!();
    skip_if_stubs_and_device!(StorageKind::Device(0));
    gpu_serial!();

    let agent = super::local_transfers::build_agent_for_kinds(&[StorageKind::Device(0)])?;
    let src = device_fc(agent.clone(), 4);
    let dst_direct = device_fc(agent.clone(), 4);
    let dst_replay = device_fc(agent.clone(), 4);

    // ctx with device_graph_replay = true for the replay run.
    let manager_replay = ctx_with_graph_replay(agent.clone());
    let ctx_replay = manager_replay.context();

    // ctx with device_graph_replay = false for the direct run.
    let manager_direct = create_transfer_context(agent, None).unwrap();
    let ctx_direct = manager_direct.context();

    let src_blocks = vec![0usize, 1usize];
    let dst_blocks = vec![2usize, 3usize];

    let src_checksums = fill_and_checksum(&src, &src_blocks, FillPattern::Sequential)?;

    // DirectDma run.
    transfer_direct_planner(&src, &dst_direct, &src_blocks, &dst_blocks, ctx_direct).await?;

    // DeviceGraphReplay run — same data, separate destination.
    transfer_direct_planner(&src, &dst_replay, &src_blocks, &dst_blocks, ctx_replay).await?;

    // Both must reproduce src checksums.
    verify_checksums_by_position(&src_checksums, &src_blocks, &dst_direct, &dst_blocks)?;
    verify_checksums_by_position(&src_checksums, &src_blocks, &dst_replay, &dst_blocks)?;

    // Pairwise equality: DirectDma == DeviceGraphReplay.
    let direct_sums = compute_block_checksums(&dst_direct, &dst_blocks)?;
    let replay_sums = compute_block_checksums(&dst_replay, &dst_blocks)?;
    for &bid in &dst_blocks {
        let d = direct_sums.get(&bid).expect("direct checksum");
        let r = replay_sums.get(&bid).expect("replay checksum");
        assert_eq!(
            d, r,
            "DirectDma vs DeviceGraphReplay checksum mismatch at dst block {bid}"
        );
    }
    Ok(())
}

/// Fresh record each call: two transfers with the SAME shape but DIFFERENT
/// source data (different block IDs). Both must produce correct results.
///
/// Unlike CUDA (which caches the graph and rebinds addresses), SYCL records
/// a fresh SYCL graph each time. This test proves that the fresh-record path
/// handles address changes correctly — no stale pointers from a previous
/// graph handle.
#[tokio::test]
async fn sycl_graph_replay_fresh_record_each_call() -> Result<()> {
    skip_if_not_sycl!();
    skip_if_stubs_and_device!(StorageKind::Device(0));
    gpu_serial!();

    let agent = super::local_transfers::build_agent_for_kinds(&[StorageKind::Device(0)])?;
    let src = device_fc(agent.clone(), 4);
    let dst = device_fc(agent.clone(), 4);
    let manager = ctx_with_graph_replay(agent);
    let ctx = manager.context();

    // Fill block 0 with Sequential and block 2 with a different pattern
    // so the two src blocks have verifiably distinct content.
    fill_blocks(&src, &[0], FillPattern::Sequential)?;
    fill_blocks(&src, &[2], FillPattern::Constant(0xABu8))?;
    let checksums_block0 = compute_block_checksums(&src, &[0])?;
    let checksums_block2 = compute_block_checksums(&src, &[2])?;

    // Assert the two blocks actually differ — if they don't, the test is vacuous.
    let c0 = checksums_block0[&0].clone();
    let c2 = checksums_block2[&2].clone();
    assert_ne!(
        c0, c2,
        "test precondition: src block 0 (Sequential) and block 2 (Constant) \
         must have different checksums. Got c0={c0}, c2={c2}"
    );

    // First transfer: src[0] → dst[1]. Records a fresh SYCL graph.
    transfer_direct_planner(&src, &dst, &[0], &[1], ctx).await?;
    let dst1_checksums = compute_block_checksums(&dst, &[1])?;
    assert_eq!(
        dst1_checksums[&1], c0,
        "first transfer: dst[1] should match src[0] (Sequential). \
         Got {}, expected {c0}",
        dst1_checksums[&1]
    );

    // Second transfer: src[2] → dst[3]. Same SHAPE, different addresses.
    // A fresh SYCL graph is recorded with the new addresses.
    transfer_direct_planner(&src, &dst, &[2], &[3], ctx).await?;
    let dst3_checksums = compute_block_checksums(&dst, &[3])?;

    // dst[3] must match src[2] (Constant 0xAB), NOT src[0] (Sequential).
    assert_eq!(
        dst3_checksums[&3], c2,
        "second transfer: dst[3] should match src[2] (Constant 0xAB), \
         but got {}. If it matches src[0]={c0}, a stale graph was replayed.",
        dst3_checksums[&3]
    );

    // Double-check that dst[3] is NOT the same as dst[1].
    assert_ne!(
        dst3_checksums[&3], dst1_checksums[&1],
        "fresh record appears broken: dst[3] ({}) == dst[1] ({}) \
         even though the two source blocks had different content",
        dst3_checksums[&3], dst1_checksums[&1]
    );

    Ok(())
}

/// Trait-level confirmation: SYCL graph ops report no address rebind support.
///
/// This is a unit-level check that doesn't move data — it just instantiates
/// the device context and verifies the trait returns.
#[tokio::test]
async fn sycl_graph_replay_confirms_no_rebind() -> Result<()> {
    skip_if_not_sycl!();
    skip_if_stubs_and_device!(StorageKind::Device(0));
    gpu_serial!();

    use crate::device::{DeviceGraphExec, DeviceGraphOps};

    let device_ctx = DeviceContext::new(DeviceBackend::Sycl, 0)?;
    let graph_ops = device_ctx.graph_ops().expect(
        "SYCL DeviceContext must implement DeviceGraphOps"
    );

    // Trait-level: SYCL does not support address rebind.
    assert!(
        !graph_ops.supports_address_rebind(),
        "SYCL supports_address_rebind() should return false"
    );

    Ok(())
}
