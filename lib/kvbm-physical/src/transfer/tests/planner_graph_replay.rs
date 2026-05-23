// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! PR-7.4.1: Integration tests for CUDA graph capture/replay.
//!
//! All tests require a real CUDA device (skipped when stubs are linked)
//! and are serialised via `gpu_serial!()` to avoid stream contention.
//!
//! Test matrix:
//!
//! 1. `cuda_graph_replay_round_trips_d2d_fc_fc` — fill src, dispatch once
//!    via graph replay, verify dst checksums match src.
//!
//! 2. `cuda_graph_replay_byte_equiv_to_direct_dma` — same input data
//!    transferred via DirectDma and CudaGraphReplay independently; both
//!    destinations must checksum-match.
//!
//! 3. `cuda_graph_replay_cache_reuse` — two transfers with the same
//!    shape; the second must reuse the cached exec (cache.len() stays at 1,
//!    not 2), and both results must be correct.
//!
//! 4. `cuda_graph_replay_address_rebind_works` — two transfers with the
//!    SAME shape but DIFFERENT block IDs; the second result must reflect
//!    the new block addresses (different src data), proving that the
//!    `cuGraphExecMemcpyNodeSetParams` rebind path is live and not a no-op.

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
        .allocate_device(0)
        .build()
        .unwrap()
}

/// Create a `TransferContext` with `cuda_graph_replay = true`.
fn ctx_with_graph_replay(agent: NixlAgent) -> crate::manager::TransferManager {
    let caps = TransferCapabilities::default().with_cuda_graph_replay(true);
    create_transfer_context(agent, Some(caps)).unwrap()
}

/// Run a planner-path CudaAsync D2D transfer via `execute_transfer` with
/// `use_planner = true`.  This routes through the full planner dispatcher
/// (including `CudaGraphReplay` candidate selection when the capability is
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

// ─────────────────────────── tests ───────────────────────────────────────────

/// Basic round-trip: fill src, replay-dispatch to dst, verify checksums.
///
/// Uses a single block so the graph has exactly one memcpy node. Verifies
/// that the captured graph + address rebind produces the same bytes as the
/// source.
#[tokio::test]
async fn cuda_graph_replay_round_trips_d2d_fc_fc() -> Result<()> {
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

/// Byte equivalence: DirectDma and CudaGraphReplay must produce identical output.
///
/// Both runs use the same source data. The two destination layouts are
/// distinct allocations; both are compared to src checksums AND to each other.
#[tokio::test]
async fn cuda_graph_replay_byte_equiv_to_direct_dma() -> Result<()> {
    skip_if_stubs_and_device!(StorageKind::Device(0));
    gpu_serial!();

    let agent = super::local_transfers::build_agent_for_kinds(&[StorageKind::Device(0)])?;
    let src = device_fc(agent.clone(), 4);
    let dst_direct = device_fc(agent.clone(), 4);
    let dst_replay = device_fc(agent.clone(), 4);

    // ctx with cuda_graph_replay = true for the replay run.
    let manager_replay = ctx_with_graph_replay(agent.clone());
    let ctx_replay = manager_replay.context();

    // ctx with cuda_graph_replay = false for the direct run.
    let manager_direct = create_transfer_context(agent, None).unwrap();
    let ctx_direct = manager_direct.context();

    let src_blocks = vec![0usize, 1usize];
    let dst_blocks = vec![2usize, 3usize];

    let src_checksums = fill_and_checksum(&src, &src_blocks, FillPattern::Sequential)?;

    // DirectDma run.
    transfer_direct_planner(&src, &dst_direct, &src_blocks, &dst_blocks, ctx_direct).await?;

    // CudaGraphReplay run — same data, separate destination.
    transfer_direct_planner(&src, &dst_replay, &src_blocks, &dst_blocks, ctx_replay).await?;

    // Both must reproduce src checksums.
    verify_checksums_by_position(&src_checksums, &src_blocks, &dst_direct, &dst_blocks)?;
    verify_checksums_by_position(&src_checksums, &src_blocks, &dst_replay, &dst_blocks)?;

    // Pairwise equality: DirectDma == CudaGraphReplay.
    let direct_sums = compute_block_checksums(&dst_direct, &dst_blocks)?;
    let replay_sums = compute_block_checksums(&dst_replay, &dst_blocks)?;
    for (&did, &rid) in dst_blocks.iter().zip(dst_blocks.iter()) {
        let d = direct_sums.get(&did).expect("direct checksum");
        let r = replay_sums.get(&rid).expect("replay checksum");
        assert_eq!(
            d, r,
            "DirectDma vs CudaGraphReplay checksum mismatch at dst block {did}"
        );
    }
    Ok(())
}

/// Cache reuse: the second transfer with the same shape must hit the cache
/// (cache size stays at 1) and produce correct output.
#[tokio::test]
async fn cuda_graph_replay_cache_reuse() -> Result<()> {
    skip_if_stubs_and_device!(StorageKind::Device(0));
    gpu_serial!();

    let agent = super::local_transfers::build_agent_for_kinds(&[StorageKind::Device(0)])?;
    let src = device_fc(agent.clone(), 4);
    let dst1 = device_fc(agent.clone(), 4);
    let dst2 = device_fc(agent.clone(), 4);
    let manager = ctx_with_graph_replay(agent);
    let ctx = manager.context();

    let src_blocks = vec![0usize];
    let dst_blocks_1 = vec![1usize];
    let dst_blocks_2 = vec![2usize];

    let src_checksums = fill_and_checksum(&src, &src_blocks, FillPattern::Sequential)?;

    // First transfer — cache miss, graph is captured.
    transfer_direct_planner(&src, &dst1, &src_blocks, &dst_blocks_1, ctx).await?;

    // Cache must have exactly 1 entry after the first transfer.
    let cache_len_after_first = ctx.graph_cache().len();
    assert_eq!(
        cache_len_after_first, 1,
        "expected 1 graph cache entry after first transfer, got {cache_len_after_first}"
    );

    // Second transfer — same shape, different dst block slot. Cache hit.
    transfer_direct_planner(&src, &dst2, &src_blocks, &dst_blocks_2, ctx).await?;

    // Cache size must still be 1 (reused, not duplicated).
    let cache_len_after_second = ctx.graph_cache().len();
    assert_eq!(
        cache_len_after_second, 1,
        "expected cache to stay at 1 entry after second (same-shape) transfer, \
         got {cache_len_after_second}"
    );

    // Both results must be correct.
    verify_checksums_by_position(&src_checksums, &src_blocks, &dst1, &dst_blocks_1)?;
    verify_checksums_by_position(&src_checksums, &src_blocks, &dst2, &dst_blocks_2)?;
    Ok(())
}

/// Address rebind: two transfers with the SAME shape but DIFFERENT source data
/// (different block IDs). The second result must reflect the NEW addresses, not
/// the captured ones, proving `cuGraphExecMemcpyNodeSetParams` is live.
#[tokio::test]
async fn cuda_graph_replay_address_rebind_works() -> Result<()> {
    skip_if_stubs_and_device!(StorageKind::Device(0));
    gpu_serial!();

    let agent = super::local_transfers::build_agent_for_kinds(&[StorageKind::Device(0)])?;
    let src = device_fc(agent.clone(), 4);
    let dst = device_fc(agent.clone(), 4);
    let manager = ctx_with_graph_replay(agent);
    let ctx = manager.context();

    // Fill block 0 with Sequential and block 2 with a different pattern (Constant)
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
         must have different checksums so rebind is observable. \
         Got c0={c0}, c2={c2}"
    );

    // First transfer: src[0] → dst[1]. Captures the graph.
    transfer_direct_planner(&src, &dst, &[0], &[1], ctx).await?;
    let dst1_checksums = compute_block_checksums(&dst, &[1])?;

    // Verify first result matches src block 0.
    assert_eq!(
        dst1_checksums[&1], c0,
        "first transfer: dst[1] should match src[0] (Sequential). \
         Got {}, expected {c0}",
        dst1_checksums[&1]
    );

    // Second transfer: src[2] → dst[3]. Same SHAPE as the first (same-shape
    // layout, 1 block, same byte count per block), but DIFFERENT addresses.
    // The rebind must update the captured graph's src/dst pointers.
    transfer_direct_planner(&src, &dst, &[2], &[3], ctx).await?;
    let dst3_checksums = compute_block_checksums(&dst, &[3])?;

    // dst[3] must match src[2] (Constant), NOT src[0] (Sequential).
    assert_eq!(
        dst3_checksums[&3], c2,
        "second transfer (rebind): dst[3] should match src[2] (Constant 0xAB), \
         but got {}. If it matches src[0]={c0}, the rebind is not working.",
        dst3_checksums[&3]
    );

    // Double-check that dst[3] is NOT the same as dst[1] (which would mean
    // the rebind was a no-op and the captured addresses were replayed instead).
    assert_ne!(
        dst3_checksums[&3], dst1_checksums[&1],
        "address rebind appears to be a no-op: dst[3] ({}) == dst[1] ({}) \
         even though the two source blocks had different content",
        dst3_checksums[&3], dst1_checksums[&1]
    );

    Ok(())
}
