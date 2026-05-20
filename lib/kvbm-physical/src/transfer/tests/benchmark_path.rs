// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! GPU integration tests for [`BenchmarkCache::benchmark_pair`] (PR-7.5.1).
//!
//! These tests exercise end-to-end timing for each `BenchmarkCandidate`
//! variant that requires a real CUDA device.  Pure-Rust / NIXL-locality
//! unit tests live in `transfer/benchmark.rs`'s own `mod tests` block.
//!
//! # Serialisation
//!
//! All tests that touch `Device(0)` acquire the [`gpu_serial!`] gate
//! so they don't race on GPU memory with concurrent tests.
//!
//! # Skip guard
//!
//! `skip_if_stubs!()` / `skip_if_stubs_and_device!()` at the top of
//! each test makes them silent no-ops when the `kvbm-kernels` crate is
//! built with stub implementations (e.g. in CPU-only CI).

use std::sync::Arc;

use anyhow::Result;

use super::gate::gpu_serial;
use super::local_transfers::build_agent_for_kinds;
use super::*;
use crate::layout::{KvBlockLayout, PhysicalLayout};
use crate::transfer::benchmark::{BenchmarkCache, BenchmarkCandidate, BenchmarkKey};
use crate::transfer::strategy::TransferStrategy;

// ── Layout helpers ────────────────────────────────────────────────────────────

/// Build a fully-contiguous FC layout on Device(0) with an explicit
/// `KvBlockLayout`.  Mirrors the helper in `planner_path.rs`.
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

/// Build a `BenchmarkKey` for a D2D transfer between two layouts.
///
/// Uses `CudaAsyncD2D` as the route discriminant — the key only needs
/// to be stable within a test.
fn make_d2d_key(src: &PhysicalLayout, dst: &PhysicalLayout) -> BenchmarkKey {
    // LayoutSignature is pub(crate) on LayoutView; construct a surrogate
    // key that's unique to this (src, dst, route) triple using the layout
    // signatures exposed through kvbm_common.
    use kvbm_common::{AxisExtent, KvDim, LayoutSignature};
    let cfg = src.layout().config();
    // Build a signature that captures the shape discriminators we care
    // about: num_blocks × (page_size, inner_dim).  Not byte-for-byte
    // identical to the planner's signature, but sufficient to make the
    // key unique for these tests (no other key will collide within a
    // single test run).
    let sig = LayoutSignature::new(
        vec![
            (KvDim::Block, AxisExtent::full(cfg.num_blocks)),
            (KvDim::Page, AxisExtent::full(cfg.page_size)),
            (KvDim::HeadSize, AxisExtent::full(cfg.inner_dim)),
        ],
        vec![
            cfg.page_size * cfg.inner_dim * cfg.dtype_width_bytes,
            cfg.inner_dim * cfg.dtype_width_bytes,
            cfg.dtype_width_bytes,
        ],
        cfg.dtype_width_bytes,
        None,
    );
    let _ = dst; // dst shape matches src for same-config pairs
    BenchmarkKey::new(
        sig.clone(),
        sig,
        Some(cfg.dtype_width_bytes as u32),
        TransferStrategy::CudaAsyncD2D,
    )
}

// ── PR-7.5.1 test: TransformKernel end-to-end timing ─────────────────────────

/// Benchmark `BenchmarkCandidate::TransformKernel` (OperationalNHD → Universal)
/// on Device(0) and assert the cache is populated with a `"TransformKernel"` winner.
///
/// # What this proves
///
/// 1. `dispatch_transform_kernel` is reachable from `benchmark.rs` via the
///    `crate::transfer::executor::dispatch_transform_kernel` re-export.
/// 2. `benchmark_pair` records a `BenchmarkOutcome` with `winner == "TransformKernel"`
///    when only that variant is submitted.
/// 3. End-to-end: the kernel launches, the stream synchronises, and the
///    elapsed time is non-zero (or at worst 0 µs — the assertion checks
///    the cache entry exists, not the specific latency).
#[tokio::test]
async fn benchmark_pair_times_transform_kernel_d2d() -> Result<()> {
    skip_if_stubs_and_device!(StorageKind::Device(0));
    gpu_serial!();

    let agent = build_agent_for_kinds(&[StorageKind::Device(0)])?;
    let src = Arc::new(build_fc_with_block_layout(
        agent.clone(),
        KvBlockLayout::OperationalNHD,
        4,
    ));
    let dst = Arc::new(build_fc_with_block_layout(
        agent.clone(),
        KvBlockLayout::Universal,
        4,
    ));

    // Transfer blocks [0, 1] from src (NHD) to dst (Universal).
    let block_pairs: Vec<(crate::BlockId, crate::BlockId)> = vec![(0, 2), (1, 3)];

    // Build the KernelInvocation directly — mirrors the shape produced by
    // `build_transform_invocation` for `standard_config` (num_layers=2,
    // outer_dim=2, page_size=16, num_heads=8, head_dim=16, dtype=F16).
    use crate::transfer::kernel_catalog::{KernelInvocation, KernelKind};
    let invocation = KernelInvocation {
        kind: KernelKind::UniversalFromBlock,
        num_layers: 2,
        outer_dim: 2,
        page_size: 16,
        num_heads: 8,
        head_dim: 16, // inner_dim (128) / num_heads (8)
        dtype: kvbm_kernels::TensorDataType::F16,
        block_layout: kvbm_kernels::BlockLayout::NHD,
    };

    let candidate = BenchmarkCandidate::TransformKernel {
        invocation,
        src: src.clone(),
        dst: dst.clone(),
        block_pairs,
    };

    let key = make_d2d_key(&src, &dst);
    let cache = Arc::new(BenchmarkCache::new());

    // Acquire a D2D-appropriate stream from a transfer context.
    let ctx_mgr = create_transfer_context(agent, None)?;
    let stream = ctx_mgr.context().acquire_d2h_stream();

    let outcome = cache.benchmark_pair(key.clone(), vec![candidate], &stream)?;

    assert_eq!(
        outcome.winner, "TransformKernel",
        "single TransformKernel candidate must be the winner"
    );
    assert_eq!(
        outcome.runs_compared, 1,
        "exactly one candidate was submitted"
    );

    // Cache must contain the outcome after benchmark_pair returns.
    let cached = cache.lookup(&key);
    assert!(
        cached.is_some(),
        "cache must hold the outcome after benchmark_pair"
    );
    assert_eq!(
        cached.unwrap().winner,
        "TransformKernel",
        "cached winner must match returned outcome"
    );

    Ok(())
}

// ── PR-7.5.1 test: three-candidate comparison ─────────────────────────────────

/// Submit three `BenchmarkCandidate` variants to `benchmark_pair` and verify
/// the cache records `runs_compared == 3`.
///
/// Uses:
/// - `DirectDma` (empty ops — no DMA issued, purely measures submit overhead)
/// - `DirectDma` with a distinct key nonce (simulates a second DirectDma shape)
/// - `TransformKernel` (NHD → Universal, real kernel dispatch)
///
/// We assert `runs_compared == 3` (counting logic) and that the cache is
/// populated.  The winner is not asserted — timing jitter may vary.
#[tokio::test]
async fn benchmark_pair_compares_three_candidates() -> Result<()> {
    skip_if_stubs_and_device!(StorageKind::Device(0));
    gpu_serial!();

    let agent = build_agent_for_kinds(&[StorageKind::Device(0)])?;
    let src = Arc::new(build_fc_with_block_layout(
        agent.clone(),
        KvBlockLayout::OperationalNHD,
        4,
    ));
    let dst = Arc::new(build_fc_with_block_layout(
        agent.clone(),
        KvBlockLayout::Universal,
        4,
    ));

    use crate::transfer::benchmark::CopyOp;
    use crate::transfer::kernel_catalog::{KernelInvocation, KernelKind};

    let block_pairs: Vec<(crate::BlockId, crate::BlockId)> = vec![(0, 2), (1, 3)];

    let invocation = KernelInvocation {
        kind: KernelKind::UniversalFromBlock,
        num_layers: 2,
        outer_dim: 2,
        page_size: 16,
        num_heads: 8,
        head_dim: 16,
        dtype: kvbm_kernels::TensorDataType::F16,
        block_layout: kvbm_kernels::BlockLayout::NHD,
    };

    let candidates = vec![
        // Candidate 1: DirectDma with no ops (measures submit-only overhead).
        BenchmarkCandidate::DirectDma { ops: vec![] },
        // Candidate 2: DirectDma again (different instance, same type — valid
        // for counting; real callers would never submit the same variant twice,
        // but the cache doesn't enforce uniqueness).
        BenchmarkCandidate::DirectDma {
            ops: vec![CopyOp {
                src_addr: 0,
                dst_addr: 0,
                size: 0,
            }],
        },
        // Candidate 3: TransformKernel.
        BenchmarkCandidate::TransformKernel {
            invocation,
            src: src.clone(),
            dst: dst.clone(),
            block_pairs,
        },
    ];

    let key = make_d2d_key(&src, &dst);
    let cache = Arc::new(BenchmarkCache::new());

    let ctx_mgr = create_transfer_context(agent, None)?;
    let stream = ctx_mgr.context().acquire_d2h_stream();

    let outcome = cache.benchmark_pair(key.clone(), candidates, &stream)?;

    assert_eq!(
        outcome.runs_compared, 3,
        "three candidates must be recorded as runs_compared"
    );

    let cached = cache.lookup(&key);
    assert!(
        cached.is_some(),
        "cache must hold the outcome after benchmark_pair with 3 candidates"
    );

    Ok(())
}
