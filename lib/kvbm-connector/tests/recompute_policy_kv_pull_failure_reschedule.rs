// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Recompute-policy KV pull-failure reschedule — coverage gap closure.
//!
//! # Why this test exists
//!
//! `kv_load_failure_policy=recompute` (vLLM's default for KVBM smoke deployments)
//! had **zero passing Rust unit-test coverage** prior to this file. This is the
//! exact failure mode that surfaced **CD USAA-1** in the disagg-smoke skill:
//! a `Session::pull` error mid-request triggers `cleanup_failed_request` →
//! `coordinator.release()`, which removes the coordinator's per-request entry.
//! Any subsequent scheduler callback that reaches the
//! `coordinator.session_for(...)` check at `decode_leader.rs:885` then observes
//! `None` and bails with `"CD USAA-1: coordinator has no session for {rid}"`.
//!
//! This test deterministically reproduces the **race window**:
//!
//!   1. Drive GNMT (decode opens the session, populates coordinator state).
//!   2. Drive USAA-1 (local kick spawned, remote pipeline spawned).
//!   3. `resolve_pull(0, Err(...))` — simulated KV transport failure.
//!   4. Wait for the failure cleanup chain to surface to vLLM.
//!   5. Assert `coordinator.session_for(rid)` returns `None` — proves the
//!      `coordinator.release()` path ran and the per-request entry is evicted.
//!
//! # Why the assertion isn't on the bail itself
//!
//! By the time `cleanup_failed_request` (in `decode_leader.rs`) finishes, BOTH
//! `cd_request_state` AND the coordinator entry are gone. A second
//! `update_state_after_alloc` would bail at `decode_leader.rs:758`
//! (`"CD request state missing for {} at USAA-1"`) BEFORE reaching line 885 —
//! you cannot deterministically reach the line-885 bail through this code path.
//!
//! Instead, this test reproduces the upstream RACE WINDOW. The line-885 bail
//! is the consequence: if any code path lands in `commit_usaa1` after this
//! window opens (e.g. an in-flight scheduler callback in production), it will
//! bail. The structural fix at `decode_leader.rs:888` is to soft-skip when the
//! session is gone (request was already cleaned up) instead of bailing.
//! This test STAYS GREEN under both the bail and the soft-skip behavior — its
//! invariant is the existence of the race window, not the choice of response.

use std::sync::Arc;

use anyhow::Result;
use kvbm_config::{DisaggConfig, DisaggregationRole};
use kvbm_connector::G2;
use kvbm_connector::common::Request;
use kvbm_connector::connector::leader::disagg::testing::{
    InMemoryRemotePrefillQueue, MockCdBlockTransport, MockCdWorkerHook, MockInnerLeaderShim,
    MockSlot, TEST_BLOCK_SIZE, wait_until,
};
use kvbm_connector::connector::leader::disagg::{
    AlwaysRemote, ConditionalDisaggCoordinator, ConnectorLeaderApi, DecodeDisaggLeader,
};
use kvbm_engine::disagg::session::{CommittedBlock, MockSessionFactory};
use kvbm_engine::testing::managers::{TestManagerBuilder, TestRegistryBuilder};
use kvbm_engine::testing::token_blocks::{create_token_sequence, generate_sequence_hashes};
use kvbm_logical::manager::BlockManager;

const COMPUTED_BLOCKS: usize = 2;
const LOCAL_BLOCKS: usize = 2;
const REMOTE_BLOCKS: usize = 4;
const TOTAL_BLOCKS: usize = COMPUTED_BLOCKS + LOCAL_BLOCKS + REMOTE_BLOCKS;
const BLOCK_SIZE: usize = TEST_BLOCK_SIZE;

fn make_request() -> Request {
    Request::builder()
        .request_id("req-1".to_string())
        .tokens(dynamo_tokens::Tokens::from(Vec::<u32>::new()))
        .build(None)
        .expect("build request")
}

fn build_g2_manager(capacity: usize) -> Arc<BlockManager<G2>> {
    let registry = TestRegistryBuilder::new().build();
    Arc::new(
        TestManagerBuilder::<G2>::new()
            .block_count(capacity)
            .block_size(BLOCK_SIZE)
            .registry(registry)
            .build(),
    )
}

fn committed_blocks(hashes: &[kvbm_logical::SequenceHash], base: usize) -> Vec<CommittedBlock> {
    hashes
        .iter()
        .enumerate()
        .map(|(i, hash)| CommittedBlock {
            hash: *hash,
            peer_block_id: base + i,
        })
        .collect()
}

/// Reproduces the CD USAA-1 race window under
/// `kv_load_failure_policy=recompute` semantics: a mid-request
/// `Session::pull` failure triggers the failure cascade, which
/// evicts the coordinator's per-request entry. After the cascade
/// settles, `coordinator.session_for(rid)` returns `None` — this
/// is the precise condition that the bail at
/// `decode_leader.rs:888` (or its post-fix soft-skip) guards
/// against.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn recompute_policy_pull_failure_evicts_coordinator_state() -> Result<()> {
    // ---------- Token sequence + manager ----------
    let g2_manager = build_g2_manager(32);
    let token_sequence = create_token_sequence(TOTAL_BLOCKS, BLOCK_SIZE, 100);
    let all_hashes = generate_sequence_hashes(&token_sequence);
    let token_blocks: Vec<_> = token_sequence.blocks().to_vec();

    // Pre-allocate + register the LOCAL-match G2 blocks so the local
    // kick has something to copy from.
    let mutables = g2_manager
        .allocate_blocks(LOCAL_BLOCKS)
        .expect("allocate local-match G2");
    let completes: Vec<_> = mutables
        .into_iter()
        .zip(token_blocks[COMPUTED_BLOCKS..COMPUTED_BLOCKS + LOCAL_BLOCKS].iter())
        .map(|(mutable, tb)| mutable.complete(tb).expect("complete local match"))
        .collect();
    let local_match_g2 = g2_manager.register_blocks(completes);

    // ---------- Wrapper + coordinator ----------
    let inner = MockInnerLeaderShim::new(BLOCK_SIZE, g2_manager.clone());
    let g1_block_ids: Vec<usize> = (1000..1000 + TOTAL_BLOCKS).collect();

    inner.install_slot(
        "req-1",
        MockSlot {
            block_size: BLOCK_SIZE,
            total_blocks: TOTAL_BLOCKS,
            computed_blocks: COMPUTED_BLOCKS,
            local_match_blocks: LOCAL_BLOCKS,
            all_hashes: all_hashes.clone(),
            token_blocks,
            local_match_g2: parking_lot::Mutex::new(Some(local_match_g2)),
            assigned_block_ids: parking_lot::Mutex::new(None),
            gnmt_result: (Some(LOCAL_BLOCKS * BLOCK_SIZE), true),
            usaa_passthrough_calls: parking_lot::Mutex::new(Vec::new()),
            transfer_params: None,
            ..MockSlot::default()
        },
    );

    let factory = MockSessionFactory::new();
    let queue = InMemoryRemotePrefillQueue::new();
    let transport = MockCdBlockTransport::new();
    let workers = MockCdWorkerHook::new();

    let coordinator = ConditionalDisaggCoordinator::new_with_decode(
        inner.clone(),
        transport.clone(),
        workers.clone(),
        factory.clone(),
        Arc::new(kvbm_connector::connector::leader::disagg::peer_resolver::NoopPeerResolver),
        tokio::runtime::Handle::current(),
        Arc::new(AlwaysRemote),
        queue.clone(),
    );

    let cfg = DisaggConfig {
        hub_url: "http://127.0.0.1:1337".to_string(),
        role: DisaggregationRole::Decode,
        max_inflight_remote_prefill_tokens: usize::MAX,
    };
    let wrapper = DecodeDisaggLeader::from_parts(
        inner.clone(),
        &cfg,
        coordinator.clone(),
        transport.clone(),
        workers.clone(),
        tokio::runtime::Handle::current(),
        None,
        None,
        None,
    );

    // ---------- 1. GNMT — opens session, populates coordinator state ----------
    wrapper.create_slot(make_request())?;
    let (count, async_flag) =
        wrapper.get_num_new_matched_tokens("req-1", COMPUTED_BLOCKS * BLOCK_SIZE)?;
    assert_eq!(count, Some((LOCAL_BLOCKS + REMOTE_BLOCKS) * BLOCK_SIZE));
    assert!(async_flag);

    // Sanity: coordinator now holds a session for this request.
    wait_until(|| coordinator.session_for("req-1").is_some()).await;
    let session = factory.last_opened().expect("decode opened a session");

    // ---------- 2. USAA-1 — local kick + remote pipeline spawned ----------
    wrapper.update_state_after_alloc(
        "req-1",
        g1_block_ids.clone(),
        (LOCAL_BLOCKS + REMOTE_BLOCKS) * BLOCK_SIZE,
    )?;
    transport.wait_onboard_count(1).await;
    transport.resolve_onboard(0, Ok(()));

    // Drive prefill peer's commits + availability so the remote
    // pipeline reaches `session.pull(...)`.
    let remote_hashes: Vec<_> = all_hashes[COMPUTED_BLOCKS + LOCAL_BLOCKS..].to_vec();
    session.inject_peer_commit(remote_hashes.clone());
    session.inject_peer_finish_commits();
    session.inject_peer_available(committed_blocks(&remote_hashes, 9000));
    session.inject_peer_drained();

    session.wait_pull_count(1).await;

    // ---------- 3. Inject the KV pull failure (simulated transport error) ----------
    // This is the synthetic equivalent of an RDMA / NIXL transport
    // failure mid-request — exactly what a real `Session::pull`
    // would surface from the worker side under `kv_load_failure_policy=recompute`.
    session.resolve_pull(
        0,
        Err(anyhow::anyhow!("simulated KV pull transport failure")),
    );

    // ---------- 4. Wait for the failure cascade to surface ----------
    // The chain is:
    //   `run_remote_pipeline` returns Err
    //     → spawned task calls `wrapper.cleanup_failed_request(rid, reason)`
    //       → `worker_hook.mark_failed_onboarding(rid, unfilled_g1)`
    //       → `wrapper.release_request(rid)` (evicts cd_request_state)
    //       → `coordinator.release(rid)`        (atomic remove from states map)
    wait_until(|| workers.failed_for("req-1").is_some()).await;
    let failed = workers.failed_for("req-1").unwrap();
    let mut got = failed.block_ids.clone();
    got.sort();
    let mut want: Vec<_> = g1_block_ids[COMPUTED_BLOCKS + LOCAL_BLOCKS..].to_vec();
    want.sort();
    assert_eq!(
        got, want,
        "failure cascade must surface the unfilled remote G1 slice to vLLM"
    );

    // ---------- 5. The race-window assertion ----------
    // After the cascade, ANY in-flight scheduler callback that reaches
    // `coordinator.session_for(rid)` will observe `None`. In production,
    // this is the trigger for the CD USAA-1 cascade: a stray
    // `commit_usaa1` (e.g. from a vLLM forward pass that crossed the
    // failure boundary in flight) lands at `decode_leader.rs:885` and
    // currently bails.
    //
    // This `wait_until` is robust to the spawn ordering between
    // `mark_failed_onboarding` (the failure_for signal) and
    // `coordinator.release` (the session_for signal): both run inside
    // the same spawned cleanup chain, but assert order does matter.
    wait_until(|| coordinator.session_for("req-1").is_none()).await;
    assert!(
        coordinator.session_for("req-1").is_none(),
        "after the pull-failure cascade, coordinator.session_for must return None — \
         this is the race window the CD USAA-1 bail (or its soft-skip fix) protects against"
    );

    // Idempotency: inflight budget reopened, no second cleanup needed.
    assert_eq!(wrapper.inflight_available(), usize::MAX);
    assert!(!wrapper.has_active_cd_request("req-1"));

    Ok(())
}
