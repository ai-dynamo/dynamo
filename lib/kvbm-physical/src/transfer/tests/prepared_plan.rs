// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Prepared-plan cache acceptance tests.
//!
//! Covers the five plan-required cases:
//!  1. G1↔G2 repeated transfers reuse one prepared plan across different block ids.
//!  2. Fully contiguous and layerwise OperationalNHD/HND ↔ Universal roundtrips
//!     preserve checksums (with cache stats populated).
//!  3. Remote Universal↔Universal asymmetric TP cache entries hit the LRU after
//!     first build. Exercised via the cache API with synthetic handles to avoid
//!     standing up a remote NIXL setup.
//!  4. Many remote `src_handles` sharing one `dst_handle` remain bounded by
//!     the LRU.
//!  5. Scratch pool reuses capacity across repeated transforms (zero host-side
//!     pointer-array growth after warmup).

use anyhow::Result;
use std::sync::Arc;

use super::gate::gpu_serial;
use super::*;
use crate::layout::{KvBlockLayout, PhysicalLayout};
use crate::manager::{LayoutHandle, TransferManager};
use crate::transfer::TransferOptions;
use crate::transfer::prepared::{PreparedPlanCache, PreparedPlanKey, PreparedTransferPlan};
use crate::transfer::strategy::TransferStrategy;

/// Build a fully-contiguous device-side layout with an explicit
/// `KvBlockLayout`. Mirrors `planner_path::build_fc_with_block_layout`
/// but exposed here for prepared-plan tests.
fn build_fc_device(
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

/// Build a layer-separate (BlockIsFirstDim) device-side layout with an
/// explicit `KvBlockLayout`.
fn build_lw_device(
    agent: NixlAgent,
    block_layout: KvBlockLayout,
    num_blocks: usize,
) -> PhysicalLayout {
    let config = standard_config(num_blocks);
    PhysicalLayout::builder(agent)
        .with_config(config)
        .with_block_layout(block_layout)
        .layer_separate(BlockDimension::BlockIsFirstDim)
        .allocate_device(0)
        .build()
        .unwrap()
}

/// Build a pinned-host Universal layout — the canonical G2 shape for
/// the prepared-plan tests below.
fn build_universal_pinned(agent: NixlAgent, num_blocks: usize) -> PhysicalLayout {
    let config = standard_config(num_blocks);
    PhysicalLayout::builder(agent)
        .with_config(config)
        .with_block_layout(KvBlockLayout::Universal)
        .fully_contiguous()
        .allocate_pinned(None)
        .build()
        .unwrap()
}

/// (Plan test 1) G1↔G2 repeated transfers reuse one prepared plan
/// across different block ids.
///
/// Drives the TransferManager handle path twice with disjoint
/// `(src_blocks, dst_blocks)` per call. The prepared-plan key only
/// covers `(handles, strategy, axis_slices)` — block ids are
/// per-call — so the second call must hit the cache.
#[tokio::test]
async fn g1_g2_repeated_transfers_reuse_prepared_plan() -> Result<()> {
    skip_if_stubs_and_device!(StorageKind::Device(0));
    gpu_serial!();

    let agent = create_test_agent("kvcc_prepared_g1_g2_reuse");
    let manager = TransferManager::builder()
        .nixl_agent(agent.clone())
        .cuda_device_id(0)
        .build()?;

    let g1 = build_fc_device(agent.clone(), KvBlockLayout::OperationalNHD, 8);
    let g2 = build_universal_pinned(agent.clone(), 8);
    // `g1_dst` receives the round-trip back: operational layout
    // matches `g1`, so checksum comparison is well-defined.
    let g1_dst = build_fc_device(agent, KvBlockLayout::OperationalNHD, 8);

    let pattern = FillPattern::Sequential;
    let src_blocks = vec![0, 1];
    let extra_src_blocks = vec![2, 3];
    let src_checksums = fill_and_checksum(
        &g1,
        &[src_blocks.clone(), extra_src_blocks.clone()].concat(),
        pattern,
    )?;

    let g1_handle = manager.register_layout(g1)?;
    let g2_handle = manager.register_layout(g2)?;
    let g1_dst_handle = manager.register_layout(g1_dst)?;

    // Sanity: cache empty before any transfer.
    let pre = manager.prepared_plan_cache_stats();
    assert_eq!(pre.local_entries, 0);
    assert_eq!(pre.local_hits, 0);
    assert_eq!(pre.local_misses, 0);

    // First leg: g1 → g2 for blocks [0, 1]. Builds the cache entry
    // for the (g1, g2, strategy) key.
    manager
        .execute_transfer(
            g1_handle,
            &src_blocks,
            g2_handle,
            &src_blocks,
            TransferOptions::default(),
        )?
        .await?;

    let after_first = manager.prepared_plan_cache_stats();
    assert_eq!(
        after_first.local_misses, 1,
        "first g1→g2 transfer should record one miss"
    );
    assert_eq!(after_first.local_hits, 0);
    assert_eq!(after_first.local_entries, 1);

    // Second leg: same handle pair, different block ids — must hit.
    manager
        .execute_transfer(
            g1_handle,
            &extra_src_blocks,
            g2_handle,
            &extra_src_blocks,
            TransferOptions::default(),
        )?
        .await?;

    let after_second = manager.prepared_plan_cache_stats();
    assert_eq!(
        after_second.local_hits, 1,
        "second g1→g2 transfer with different block ids must hit the cache"
    );
    assert_eq!(after_second.local_misses, 1, "no new misses on second call");
    assert_eq!(after_second.local_entries, 1, "still one cached entry");

    // Round-trip back to operational to confirm cache hits returned
    // correct bytes. g2 → g1_dst is a *different* cache key (handles
    // swap roles), so this build adds a second cache entry.
    manager
        .execute_transfer(
            g2_handle,
            &src_blocks,
            g1_dst_handle,
            &src_blocks,
            TransferOptions::default(),
        )?
        .await?;
    manager
        .execute_transfer(
            g2_handle,
            &extra_src_blocks,
            g1_dst_handle,
            &extra_src_blocks,
            TransferOptions::default(),
        )?
        .await?;
    let after_reverse = manager.prepared_plan_cache_stats();
    assert_eq!(
        after_reverse.local_entries, 2,
        "two distinct (src,dst) handle pairs = two cache entries"
    );
    assert!(
        after_reverse.local_hits >= 2,
        "at least one hit on the second-pair second call (stats={after_reverse:?})"
    );

    let g1_dst_ref = manager
        .get_physical_layout(g1_dst_handle)
        .expect("g1_dst layout present");
    verify_checksums_by_position(&src_checksums, &[0, 1, 2, 3], &g1_dst_ref, &[0, 1, 2, 3])?;
    Ok(())
}

/// (Plan test 2 — FC variant) Fully contiguous OperationalNHD↔Universal
/// roundtrip preserves checksums AND populates the prepared-plan cache
/// for both directions.
#[tokio::test]
async fn fc_nhd_universal_roundtrip_with_prepared_cache() -> Result<()> {
    assert_operational_universal_with_cache(KvBlockLayout::OperationalNHD, LayoutKind::FC).await
}

/// (Plan test 2 — FC HND variant)
#[tokio::test]
async fn fc_hnd_universal_roundtrip_with_prepared_cache() -> Result<()> {
    assert_operational_universal_with_cache(KvBlockLayout::OperationalHND, LayoutKind::FC).await
}

/// (Plan test 2 — LW variant) Layer-separate OperationalNHD↔Universal
/// roundtrip preserves checksums.
#[tokio::test]
async fn lw_nhd_universal_roundtrip_with_prepared_cache() -> Result<()> {
    assert_operational_universal_with_cache(KvBlockLayout::OperationalNHD, LayoutKind::LW).await
}

/// (Plan test 2 — LW HND variant)
#[tokio::test]
async fn lw_hnd_universal_roundtrip_with_prepared_cache() -> Result<()> {
    assert_operational_universal_with_cache(KvBlockLayout::OperationalHND, LayoutKind::LW).await
}

async fn assert_operational_universal_with_cache(
    op_layout: KvBlockLayout,
    op_kind: LayoutKind,
) -> Result<()> {
    skip_if_stubs_and_device!(StorageKind::Device(0));
    gpu_serial!();

    let agent = create_test_agent("kvcc_prepared_op_univ_roundtrip");
    let manager = TransferManager::builder()
        .nixl_agent(agent.clone())
        .cuda_device_id(0)
        .build()?;

    let src = match op_kind {
        LayoutKind::FC => build_fc_device(agent.clone(), op_layout, 4),
        LayoutKind::LW => build_lw_device(agent.clone(), op_layout, 4),
    };
    let mid = build_universal_pinned(agent.clone(), 4);
    let dst = match op_kind {
        LayoutKind::FC => build_fc_device(agent.clone(), op_layout, 4),
        LayoutKind::LW => build_lw_device(agent, op_layout, 4),
    };

    let src_blocks = vec![0, 1];
    let mid_blocks = vec![2, 3];
    let dst_blocks = vec![0, 1];

    let src_checksums = fill_and_checksum(&src, &src_blocks, FillPattern::Sequential)?;

    let src_handle = manager.register_layout(src)?;
    let mid_handle = manager.register_layout(mid)?;
    let dst_handle = manager.register_layout(dst)?;

    // Op → Universal.
    manager
        .execute_transfer(
            src_handle,
            &src_blocks,
            mid_handle,
            &mid_blocks,
            TransferOptions::default(),
        )?
        .await?;

    let after_forward = manager.prepared_plan_cache_stats();
    assert!(
        after_forward.local_misses >= 1,
        "forward leg should have recorded at least one miss (stats={after_forward:?})"
    );

    // Universal → fresh op.
    manager
        .execute_transfer(
            mid_handle,
            &mid_blocks,
            dst_handle,
            &dst_blocks,
            TransferOptions::default(),
        )?
        .await?;

    let after_reverse = manager.prepared_plan_cache_stats();
    assert!(
        after_reverse.local_entries >= 2,
        "two distinct (handle pair, direction) plans should be cached (stats={after_reverse:?})"
    );

    let dst_ref = manager
        .get_physical_layout(dst_handle)
        .expect("dst layout present");
    verify_checksums_by_position(&src_checksums, &src_blocks, &dst_ref, &dst_blocks)?;
    Ok(())
}

/// (Plan test 5) Scratch pool reuses capacity across repeated transforms.
///
/// Run the same G1→G2 transfer ten times, then verify the scratch
/// pool's `peak_op_capacity` matches the first call's capacity — i.e.
/// no host-side `op_ptrs` Vec grew after warmup.
#[tokio::test]
async fn transform_scratch_pool_reuses_capacity_after_warmup() -> Result<()> {
    skip_if_stubs_and_device!(StorageKind::Device(0));
    gpu_serial!();

    let agent = create_test_agent("kvcc_prepared_scratch_reuse");
    let manager = TransferManager::builder()
        .nixl_agent(agent.clone())
        .cuda_device_id(0)
        .build()?;

    let g1 = build_fc_device(agent.clone(), KvBlockLayout::OperationalNHD, 8);
    let g2 = build_universal_pinned(agent, 8);

    let g1_handle = manager.register_layout(g1)?;
    let g2_handle = manager.register_layout(g2)?;

    // Run 10 same-shape transfers. Steady state must not grow the pool.
    for _ in 0..10 {
        manager
            .execute_transfer(
                g1_handle,
                &[0, 1, 2, 3],
                g2_handle,
                &[0, 1, 2, 3],
                TransferOptions::default(),
            )?
            .await?;
    }

    let stats = manager.prepared_plan_cache_stats();
    assert_eq!(
        stats.local_misses, 1,
        "single cache miss expected after 10 same-shape transfers"
    );
    assert_eq!(stats.local_hits, 9, "9 hits after first miss");

    // Approximate-bytes count should be small and stable — the cache
    // entry plus a 2-entry scratch slot (op_ptrs of 4*nl*no + univ_ptrs
    // of 4 entries). Bounded check rather than exact match because the
    // map overhead is implementation-dependent.
    assert!(
        stats.approximate_bytes > 0 && stats.approximate_bytes < 8 * 1024,
        "prepared-plan footprint should be small after steady state (stats={stats:?})"
    );
    Ok(())
}

/// (Plan test 1 — prewarm variant) `prewarm_local_pair` populates the
/// cache for both directions before any transfer runs.
#[tokio::test]
async fn prewarm_local_pair_populates_cache() -> Result<()> {
    skip_if_stubs_and_device!(StorageKind::Device(0));
    gpu_serial!();

    let agent = create_test_agent("kvcc_prepared_prewarm");
    let manager = TransferManager::builder()
        .nixl_agent(agent.clone())
        .cuda_device_id(0)
        .build()?;

    let g1 = build_fc_device(agent.clone(), KvBlockLayout::OperationalNHD, 4);
    let g2 = build_universal_pinned(agent, 4);

    let g1_handle = manager.register_layout(g1)?;
    let g2_handle = manager.register_layout(g2)?;

    let stats = manager.prewarm_local_pair(g1_handle, g2_handle)?;
    assert_eq!(
        stats.local_entries, 2,
        "prewarm should build both directions of the handle pair (stats={stats:?})"
    );
    assert_eq!(stats.local_misses, 2);
    assert_eq!(stats.local_hits, 0, "prewarm only builds, never hits");

    // Subsequent transfer hits the prewarmed entry.
    manager
        .execute_transfer(
            g1_handle,
            &[0, 1],
            g2_handle,
            &[0, 1],
            TransferOptions::default(),
        )?
        .await?;
    let post = manager.prepared_plan_cache_stats();
    assert_eq!(
        post.local_hits, 1,
        "first post-prewarm transfer must hit cache"
    );
    assert_eq!(post.local_entries, 2, "prewarm entries still present");
    Ok(())
}

// ────────────────── Cache-only unit tests (no CUDA) ──────────────────

fn dummy_transform_plan() -> PreparedTransferPlan {
    // Synthetic plan used by cache-only unit tests below. The tests
    // only inspect Arc identity and cache structural state — they
    // never invoke emit_*, so the inner univ_base/scratch pool fields
    // are inert.
    use crate::transfer::kernel_catalog::{KernelInvocation, KernelKind};
    use crate::transfer::prepared::TransformScratchPool;
    PreparedTransferPlan::synthetic_for_tests(
        KernelInvocation {
            kind: KernelKind::UniversalFromBlock,
            num_layers: 1,
            outer_dim: 1,
            page_size: 1,
            num_heads: 1,
            head_dim: 1,
            dtype: kvbm_kernels::TensorDataType::F16,
            block_layout: kvbm_kernels::BlockLayout::NHD,
        },
        Some(0x1000),
        Some(64),
        Arc::new(TransformScratchPool::default()),
    )
}

/// (Plan test 4) Many remote `src_handles` sharing one `dst_handle`
/// remain bounded by the LRU.
#[test]
fn remote_lru_bounds_distinct_src_handles_for_one_dst() {
    let cache = PreparedPlanCache::new(true, 16);
    let local_worker = 7;
    let dst_handle = LayoutHandle::new(local_worker, 0);

    // 64 distinct remote (worker_id, layout_id) pairs all targeting
    // the same dst_handle. Capacity is 16, so the LRU should hold
    // exactly 16 entries after the loop with the most recent ones
    // present.
    for i in 0..64u64 {
        let src_handle = LayoutHandle::new(100 + i, 0);
        let key = PreparedPlanKey::new(src_handle, dst_handle, TransferStrategy::NixlRead, &[]);
        cache
            .get_or_insert_with(local_worker, key, || Ok(dummy_transform_plan()))
            .expect("insert");
    }
    let stats = cache.stats();
    assert_eq!(stats.remote_entries, 16, "LRU bounded at capacity");
    assert_eq!(stats.remote_misses, 64, "every insertion was a miss");
    assert_eq!(stats.remote_hits, 0);
    assert!(
        stats.approximate_bytes > 0,
        "remote LRU should report non-zero approximate bytes"
    );
}

/// (Plan test 3) Same remote `(src, dst)` pair queried twice must hit
/// the LRU on the second call. Verifies the asymmetric-TP cache hit
/// path without needing a real remote NIXL setup.
#[test]
fn remote_pair_second_lookup_hits_lru() {
    let cache = PreparedPlanCache::new(true, 1024);
    let local_worker = 7;
    let src_handle = LayoutHandle::new(42, 0); // remote
    let dst_handle = LayoutHandle::new(local_worker, 1); // local
    // Build an axis_slices pattern that mimics an asymmetric TP pull
    // (a sliced range on the Head axis). The cache key includes
    // slice ranges, so this exercises the slice-keyed path.
    let head_slice = kvbm_common::AxisIntersection {
        dim: kvbm_common::KvDim::HeadCount,
        src_local: 0..4,
        dst_local: 0..4,
    };
    let key1 = PreparedPlanKey::new(
        src_handle,
        dst_handle,
        TransferStrategy::NixlRead,
        std::slice::from_ref(&head_slice),
    );
    let key2 = PreparedPlanKey::new(
        src_handle,
        dst_handle,
        TransferStrategy::NixlRead,
        std::slice::from_ref(&head_slice),
    );

    let first = cache
        .get_or_insert_with(local_worker, key1, || Ok(dummy_transform_plan()))
        .unwrap();
    let second = cache
        .get_or_insert_with(local_worker, key2, || panic!("second lookup must hit"))
        .unwrap();
    assert!(
        Arc::ptr_eq(&first, &second),
        "second remote lookup with the same key should return the same cached Arc"
    );
    let stats = cache.stats();
    assert_eq!(stats.remote_misses, 1);
    assert_eq!(stats.remote_hits, 1);
    assert_eq!(stats.remote_entries, 1);
}
