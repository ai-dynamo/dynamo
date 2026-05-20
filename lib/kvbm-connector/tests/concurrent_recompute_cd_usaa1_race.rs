// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! CD USAA-1 race coverage under CONCURRENT recompute — sibling-fix to
//! `recompute_policy_kv_pull_failure_reschedule`.
//!
//! # Why this test exists
//!
//! The neighbor file proves the race **window** opens — that
//! `coordinator.session_for(rid)` goes None after a pull-failure
//! cascade. It does NOT exercise either of the two read sites the
//! `fc4173c496` fix targets:
//!
//!   * Site B — `decode_leader.rs` `cd_request_state.get(rid)` at
//!     `commit_usaa1` head (the soft-skip patch). Pre-fix bail:
//!     `"CD request state missing for {rid} at USAA-1"`.
//!   * Site A — `decode_leader.rs` `coordinator.session_for(rid)` at
//!     `commit_usaa1`'s remote-pipeline spawn. Pre-fix bail:
//!     `"CD USAA-1: coordinator has no session for {rid}"`.
//!
//! Under `kv_load_failure_policy=recompute` (run #9 trace: 8 recompute
//! events, 3 fatal), the scheduler reschedules the same request id
//! across multiple ticks. Between two ticks, a sibling
//! `cleanup_failed_request` for that rid can call `coordinator.release`
//! AFTER the wrapper-side `cd_request_state` was rebuilt by a fresh
//! gnmt but BEFORE the next USAA-1 lookups fire. That's the
//! observable state both sites guard.
//!
//! # Determinism mechanism
//!
//! `commit_usaa1` is fully synchronous; we cannot wedge a Tokio yield
//! between its two reads. Instead we reproduce the **observable state**
//! from the cross-tick race directly:
//!
//! 1. Build the wrapper + coordinator, drive a request through gnmt
//!    (populates both wrapper `cd_request_state[rid].session` and
//!    coordinator `states[rid]`).
//! 2. Call `coordinator.release(rid)` synchronously from the test
//!    thread — this is exactly the operation a sibling
//!    `cleanup_failed_request` would have completed under
//!    `kv_load_failure_policy=recompute`. Pre-fix code at line 888
//!    queries `coordinator.session_for(rid)` and bails because the
//!    coordinator state is gone. Post-fix code reads the session from
//!    the held `Arc<CdRequestState>::session`, which survived because
//!    `coordinator.release` only evicts coordinator state, not wrapper
//!    state.
//! 3. Drive USAA-1 via `wrapper.update_state_after_alloc(rid, ...)`,
//!    which routes through `decode_usaa` → `commit_usaa1`. Assert the
//!    call returns `Ok(())` (post-fix) instead of bailing with
//!    `"coordinator has no session for {rid}"` (pre-fix).
//! 4. Wait for the spawned remote pipeline to observe the
//!    coordinator-finalized session's `CommitDelta::Closed` (because
//!    `coordinator.release` finalized it). The pipeline's `?` routes
//!    the resulting Err to `cleanup_failed_request`, which surfaces
//!    `mark_failed_onboarding(rid, unfilled_g1)` to vLLM. Test
//!    succeeds: per-request graceful failure, no worker-fatal.
//!
//! Plus a second test: TWO concurrent requests, one fails via pull
//! failure, the other survives. Validates that the sibling-cleanup
//! does not knock out the unrelated request — the per-rid keying on
//! `cd_request_state` and `coordinator.states` holds.
//!
//! # Validation expectation
//!
//! On post-fix HEAD (`fc4173c496` and later), both tests PASS.
//! On pre-fix HEAD (revert decode_leader.rs to its parent), the first
//! test FAILS with `update_state_after_alloc` returning Err containing
//! `"CD USAA-1: coordinator has no session for req-1"` — the exact
//! Site A bail signature. See the commit body for the captured
//! pre-fix failure trace.

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
    AlwaysRemote, ConditionalDisaggCoordinator, ConnectorLeaderApi, CoordinatorParts,
    DecodeDisaggLeader, HubWiring,
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

fn make_request(request_id: &str) -> Request {
    Request::builder()
        .request_id(request_id.to_string())
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

/// Build a fully wired `(wrapper, coordinator, factory, transport,
/// workers, queue, inner)` tuple plus the pre-allocated local-match G2
/// blocks for the named slot. Centralizes the boilerplate across the
/// two test cases.
#[allow(dead_code)]
struct Harness {
    wrapper: Arc<DecodeDisaggLeader>,
    coordinator: Arc<ConditionalDisaggCoordinator>,
    factory: Arc<MockSessionFactory>,
    transport: Arc<MockCdBlockTransport>,
    workers: Arc<MockCdWorkerHook>,
    inner: Arc<MockInnerLeaderShim>,
    all_hashes: Vec<kvbm_logical::SequenceHash>,
}

fn build_harness(request_id: &str, g1_base: usize) -> (Harness, Vec<usize>) {
    let g2_manager = build_g2_manager(32);
    let token_sequence = create_token_sequence(TOTAL_BLOCKS, BLOCK_SIZE, g1_base as u32);
    let all_hashes = generate_sequence_hashes(&token_sequence);
    let token_blocks: Vec<_> = token_sequence.blocks().to_vec();

    let mutables = g2_manager
        .allocate_blocks(LOCAL_BLOCKS)
        .expect("allocate local-match G2");
    let completes: Vec<_> = mutables
        .into_iter()
        .zip(token_blocks[COMPUTED_BLOCKS..COMPUTED_BLOCKS + LOCAL_BLOCKS].iter())
        .map(|(mutable, tb)| mutable.complete(tb).expect("complete local match"))
        .collect();
    let local_match_g2 = g2_manager.register_blocks(completes);

    let inner = MockInnerLeaderShim::new(BLOCK_SIZE, g2_manager.clone());
    let g1_block_ids: Vec<usize> = (g1_base..g1_base + TOTAL_BLOCKS).collect();

    inner.install_slot(
        request_id,
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
        CoordinatorParts {
            inner: inner.clone(),
            transport: transport.clone(),
            worker_hook: workers.clone(),
            session_factory: factory.clone(),
            peer_resolver: Arc::new(
                kvbm_connector::connector::leader::disagg::peer_resolver::NoopPeerResolver,
            ),
            runtime: tokio::runtime::Handle::current(),
        },
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
        HubWiring {
            hub: None,
            client: None,
            hub_velo_id: None,
        },
    );

    (
        Harness {
            wrapper,
            coordinator,
            factory,
            transport,
            workers,
            inner,
            all_hashes,
        },
        g1_block_ids,
    )
}

/// Reproduces the **CD USAA-1 race at Site A** (the
/// `coordinator.session_for(rid)` lookup at the remote-pipeline spawn,
/// formerly `decode_leader.rs:888`).
///
/// ## Race topology
///
/// 1. Request "req-1" enters gnmt: wrapper installs
///    `cd_request_state[req-1].session = Some(S)`, coordinator
///    installs `states[req-1].session = Some(S)`.
/// 2. **Sibling cleanup fires** — modeled here by a direct
///    `coordinator.release("req-1")` call. In production this is the
///    second-half of `cleanup_failed_request` for an earlier
///    rescheduled instance of this rid under
///    `kv_load_failure_policy=recompute`. The wrapper's
///    `cd_request_state` is untouched (the bug exposes the
///    coordinator/wrapper desync).
/// 3. USAA-1 fires (`update_state_after_alloc`). The
///    `is_active = cd_request_state.contains_key(rid)` check at
///    `decode_leader.rs:656` is true (wrapper state is intact), so the
///    call descends into `commit_usaa1`.
/// 4. **Site A**: pre-fix code at `decode_leader.rs:888` queries
///    `coordinator.session_for(rid)` → None → `bail!("CD USAA-1:
///    coordinator has no session for {rid}")`. Post-fix reads
///    `updated.session.lock().clone()` (the held Arc) and proceeds.
///
/// ## Determinism
///
/// `commit_usaa1` is fully synchronous; the race is across scheduler
/// ticks (or, in production, across spawned tasks crossing tick
/// boundaries). Sequencing the cleanup BEFORE the USAA-1 lookup
/// faithfully reproduces the read site's observable state under the
/// race. No sleeps, no barriers — the assertion is causal.
///
/// ## Expected behavior on post-fix code
///
/// `update_state_after_alloc` returns `Ok(())`. The remote-pipeline
/// task spawned inside `commit_usaa1` holds the `Arc<dyn Session>`
/// from the wrapper's per-request state; that session was finalized
/// by `coordinator.release` (line 1056), so the pipeline observes
/// `CommitDelta::Closed` short of the expected hash count and bails.
/// The bail routes through `cleanup_failed_request` →
/// `mark_failed_onboarding(rid, unfilled_g1)`. We assert this is the
/// observable terminal state.
///
/// ## Expected behavior on pre-fix code
///
/// `update_state_after_alloc` returns `Err` containing the bail
/// string `"CD USAA-1: coordinator has no session for req-1"`. The `?`
/// in this test propagates the error and the test panics with that
/// signature. See the commit body for the captured failure trace.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn commit_usaa1_survives_coordinator_release_race() -> Result<()> {
    let (h, g1_block_ids) = build_harness("req-1", 1000);

    // ---------- 1. GNMT — opens session, installs wrapper + coordinator state ----------
    h.wrapper.create_slot(make_request("req-1"))?;
    let (count, async_flag) = h
        .wrapper
        .get_num_new_matched_tokens("req-1", COMPUTED_BLOCKS * BLOCK_SIZE)?;
    assert_eq!(count, Some((LOCAL_BLOCKS + REMOTE_BLOCKS) * BLOCK_SIZE));
    assert!(async_flag);

    // Sanity: both maps now hold this request.
    wait_until(|| h.coordinator.session_for("req-1").is_some()).await;
    assert!(
        h.wrapper.has_active_cd_request("req-1"),
        "wrapper-side cd_request_state must hold req-1 after gnmt"
    );
    let session_before = h
        .coordinator
        .session_for("req-1")
        .expect("coordinator session_for after gnmt");

    // ---------- 2. Race window: sibling cleanup runs `coordinator.release` ----------
    // This is the exact effect of a sibling `cleanup_failed_request`
    // that already finished `release_request` (which evicted nothing
    // here because the wrapper-side state belongs to the CURRENT
    // tick) and reached `coordinator.release` for this rid. The
    // wrapper-side cd_request_state is intentionally NOT touched —
    // that's the asymmetry the bug exposed.
    h.coordinator.release("req-1");
    assert!(
        h.coordinator.session_for("req-1").is_none(),
        "coordinator must drop session on release (Bug B's atomic-remove site)"
    );
    assert!(
        h.wrapper.has_active_cd_request("req-1"),
        "wrapper-side state MUST survive coordinator.release — that's the race the fix targets"
    );

    // ---------- 3. USAA-1 fires — pre-fix would bail at Site A ----------
    // is_active = true (wrapper state intact), so we descend into
    // commit_usaa1. The Site B lookup
    // (`cd_request_state.get(rid)`) succeeds (state intact). The
    // Site A lookup pre-fix did `coordinator.session_for(rid)` →
    // None → bail. Post-fix reads `updated.session.lock().clone()`
    // (held Arc from gnmt) → Some → proceeds.
    h.wrapper
        .update_state_after_alloc(
            "req-1",
            g1_block_ids.clone(),
            (LOCAL_BLOCKS + REMOTE_BLOCKS) * BLOCK_SIZE,
        )
        .map_err(|err| {
            anyhow::anyhow!(
                "POST-FIX REGRESSION: update_state_after_alloc bailed — \
                 indicates Site A or Site B fix is missing. Error: {err}"
            )
        })?;

    // ---------- 4. Local kick resolves (unblocks the spawned local-G2→G1 task) ----------
    // The local kick may complete BEFORE or AFTER cleanup_failed_request
    // runs, depending on tokio scheduling. Either ordering is correct
    // for the structural fix we're validating; the test asserts only
    // the invariants that hold regardless of that interleave.
    h.transport.wait_onboard_count(1).await;
    h.transport.resolve_onboard(0, Ok(()));

    // ---------- 5. Remote pipeline observes Closed (session was finalized by release) ----------
    // run_remote_pipeline subscribes to the held Arc<Session>'s
    // commits; coordinator.release called session.finalize, which
    // pushed CommitDelta::Closed onto the commits stream. The
    // pipeline's `?` routes Err to cleanup_failed_request, which
    // emits mark_failed_onboarding(rid, unfilled_g1). NO worker-fatal
    // anywhere in this chain.
    wait_until(|| h.workers.failed_for("req-1").is_some()).await;
    let failed = h
        .workers
        .failed_for("req-1")
        .expect("mark_failed_onboarding must surface unfilled g1 ids");

    // The unfilled set MUST include the full remote slice (the remote
    // pipeline bailed before any pull resolved). It MAY ALSO include
    // the local slice if cleanup_failed_request runs before
    // local_onboard_complete is set — that's a valid interleave and
    // not a bug. The invariant is: every unfilled id corresponds to a
    // block this rid was waiting on, and the REMOTE slice is fully
    // present (no partial-remote leakage that would mark a remote g1
    // as filled when it never was).
    let remote_g1: Vec<usize> = g1_block_ids[COMPUTED_BLOCKS + LOCAL_BLOCKS..].to_vec();
    let local_g1: Vec<usize> =
        g1_block_ids[COMPUTED_BLOCKS..COMPUTED_BLOCKS + LOCAL_BLOCKS].to_vec();
    let got_set: std::collections::HashSet<usize> = failed.block_ids.iter().copied().collect();
    let remote_set: std::collections::HashSet<usize> = remote_g1.iter().copied().collect();
    let local_set: std::collections::HashSet<usize> = local_g1.iter().copied().collect();
    let valid_universe: std::collections::HashSet<usize> =
        remote_set.union(&local_set).copied().collect();
    assert!(
        remote_set.is_subset(&got_set),
        "unfilled set must include the full REMOTE g1 slice — \
         remote pipeline bailed before any pull. got={:?} want_superset_of={:?}",
        failed.block_ids,
        remote_g1
    );
    assert!(
        got_set.is_subset(&valid_universe),
        "unfilled set must NOT include any block outside (LOCAL ∪ REMOTE) — \
         got={:?} valid_universe={:?}",
        failed.block_ids,
        valid_universe
    );

    // ---------- 6. Cleanup chain settled ----------
    // After cleanup_failed_request, both wrapper and coordinator are
    // empty. has_active_cd_request returns false; coordinator
    // session_for returns None.
    wait_until(|| !h.wrapper.has_active_cd_request("req-1")).await;
    assert!(h.coordinator.session_for("req-1").is_none());

    // Holding the original session Arc keeps it alive even though
    // coordinator and wrapper both released — explicit drop here so
    // the active-session gauge can decrement.
    drop(session_before);

    // No completed callback should have fired (the request failed,
    // not succeeded).
    assert!(
        !h.workers.completed_contains("req-1"),
        "failed request must NOT be marked complete"
    );

    Ok(())
}

/// Sibling-fail scenario with TWO concurrent requests sharing a
/// wrapper. Drives both through gnmt + USAA-1, fails req-A's pull
/// (which triggers cleanup → coordinator.release(A)), and asserts
/// req-B is unaffected. This is the multi-request expression of the
/// CD USAA-1 cascade observed in run #9 (8 recompute events, 3 fatal):
/// the bug was that a failure on one rid could fan out to a worker-
/// fatal that took down the whole engine.
///
/// Post-fix: req-A's failure is contained — req-B's session is
/// unrelated, its `cd_request_state` is unrelated, and it completes
/// normally. No `mark_failed_onboarding` for req-B. No bail anywhere.
///
/// (We can't re-use `Harness` because each request gets its own slot
/// install — but they SHARE the wrapper, coordinator, factory,
/// transport, and workers, which is the property the test wants to
/// exercise.)
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn concurrent_recompute_sibling_failure_does_not_cascade() -> Result<()> {
    // ---------- Shared infra: one wrapper, two slots ----------
    let g2_manager = build_g2_manager(64);
    let token_seq_a = create_token_sequence(TOTAL_BLOCKS, BLOCK_SIZE, 100);
    let token_seq_b = create_token_sequence(TOTAL_BLOCKS, BLOCK_SIZE, 200);
    let hashes_a = generate_sequence_hashes(&token_seq_a);
    let hashes_b = generate_sequence_hashes(&token_seq_b);
    let token_blocks_a: Vec<_> = token_seq_a.blocks().to_vec();
    let token_blocks_b: Vec<_> = token_seq_b.blocks().to_vec();

    // Local-match G2 blocks for both slots.
    let mut_a = g2_manager
        .allocate_blocks(LOCAL_BLOCKS)
        .expect("alloc A local match");
    let comp_a: Vec<_> = mut_a
        .into_iter()
        .zip(token_blocks_a[COMPUTED_BLOCKS..COMPUTED_BLOCKS + LOCAL_BLOCKS].iter())
        .map(|(m, tb)| m.complete(tb).expect("complete A local"))
        .collect();
    let local_g2_a = g2_manager.register_blocks(comp_a);

    let mut_b = g2_manager
        .allocate_blocks(LOCAL_BLOCKS)
        .expect("alloc B local match");
    let comp_b: Vec<_> = mut_b
        .into_iter()
        .zip(token_blocks_b[COMPUTED_BLOCKS..COMPUTED_BLOCKS + LOCAL_BLOCKS].iter())
        .map(|(m, tb)| m.complete(tb).expect("complete B local"))
        .collect();
    let local_g2_b = g2_manager.register_blocks(comp_b);

    let inner = MockInnerLeaderShim::new(BLOCK_SIZE, g2_manager.clone());
    let g1_a: Vec<usize> = (1000..1000 + TOTAL_BLOCKS).collect();
    let g1_b: Vec<usize> = (3000..3000 + TOTAL_BLOCKS).collect();

    inner.install_slot(
        "req-A",
        MockSlot {
            block_size: BLOCK_SIZE,
            total_blocks: TOTAL_BLOCKS,
            computed_blocks: COMPUTED_BLOCKS,
            local_match_blocks: LOCAL_BLOCKS,
            all_hashes: hashes_a.clone(),
            token_blocks: token_blocks_a,
            local_match_g2: parking_lot::Mutex::new(Some(local_g2_a)),
            assigned_block_ids: parking_lot::Mutex::new(None),
            gnmt_result: (Some(LOCAL_BLOCKS * BLOCK_SIZE), true),
            usaa_passthrough_calls: parking_lot::Mutex::new(Vec::new()),
            transfer_params: None,
            ..MockSlot::default()
        },
    );
    inner.install_slot(
        "req-B",
        MockSlot {
            block_size: BLOCK_SIZE,
            total_blocks: TOTAL_BLOCKS,
            computed_blocks: COMPUTED_BLOCKS,
            local_match_blocks: LOCAL_BLOCKS,
            all_hashes: hashes_b.clone(),
            token_blocks: token_blocks_b,
            local_match_g2: parking_lot::Mutex::new(Some(local_g2_b)),
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
        CoordinatorParts {
            inner: inner.clone(),
            transport: transport.clone(),
            worker_hook: workers.clone(),
            session_factory: factory.clone(),
            peer_resolver: Arc::new(
                kvbm_connector::connector::leader::disagg::peer_resolver::NoopPeerResolver,
            ),
            runtime: tokio::runtime::Handle::current(),
        },
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
        HubWiring {
            hub: None,
            client: None,
            hub_velo_id: None,
        },
    );

    // ---------- Drive A: gnmt + USAA-1 ----------
    wrapper.create_slot(make_request("req-A"))?;
    wrapper.get_num_new_matched_tokens("req-A", COMPUTED_BLOCKS * BLOCK_SIZE)?;
    wait_until(|| coordinator.session_for("req-A").is_some()).await;
    let session_a = factory.last_opened().expect("A session opened");

    wrapper.update_state_after_alloc(
        "req-A",
        g1_a.clone(),
        (LOCAL_BLOCKS + REMOTE_BLOCKS) * BLOCK_SIZE,
    )?;
    transport.wait_onboard_count(1).await;
    transport.resolve_onboard(0, Ok(()));

    // ---------- Drive B: gnmt + USAA-1 (interleaved with A's remote pipeline staging) ----------
    wrapper.create_slot(make_request("req-B"))?;
    wrapper.get_num_new_matched_tokens("req-B", COMPUTED_BLOCKS * BLOCK_SIZE)?;
    wait_until(|| coordinator.session_for("req-B").is_some()).await;
    let session_b = factory.last_opened().expect("B session opened");
    assert!(
        !Arc::ptr_eq(&session_a, &session_b),
        "A and B must hold distinct sessions"
    );

    wrapper.update_state_after_alloc(
        "req-B",
        g1_b.clone(),
        (LOCAL_BLOCKS + REMOTE_BLOCKS) * BLOCK_SIZE,
    )?;
    // B is the second onboard to be queued.
    transport.wait_onboard_count(2).await;
    transport.resolve_onboard(1, Ok(()));

    // ---------- Stage A's remote pipeline up to `session.pull` ----------
    let remote_hashes_a: Vec<_> = hashes_a[COMPUTED_BLOCKS + LOCAL_BLOCKS..].to_vec();
    session_a.inject_peer_commit(remote_hashes_a.clone());
    session_a.inject_peer_finish_commits();
    session_a.inject_peer_available(committed_blocks(&remote_hashes_a, 9000));
    session_a.inject_peer_drained();
    session_a.wait_pull_count(1).await;

    // ---------- Stage B's remote pipeline THROUGH completion ----------
    // B's pull resolves Ok; the remote G2→G1 onboard then runs and
    // completes B's pipeline.
    let remote_hashes_b: Vec<_> = hashes_b[COMPUTED_BLOCKS + LOCAL_BLOCKS..].to_vec();
    session_b.inject_peer_commit(remote_hashes_b.clone());
    session_b.inject_peer_finish_commits();
    session_b.inject_peer_available(committed_blocks(&remote_hashes_b, 19000));
    session_b.inject_peer_drained();
    session_b.wait_pull_count(1).await;
    session_b.resolve_pull(0, Ok(()));

    // ---------- Inject A's pull failure — triggers full cleanup cascade for A ----------
    session_a.resolve_pull(
        0,
        Err(anyhow::anyhow!(
            "simulated KV pull transport failure for req-A"
        )),
    );

    // ---------- A's failure surfaces, B's onboard for the remote slice resolves ----------
    wait_until(|| workers.failed_for("req-A").is_some()).await;
    let failed_a = workers.failed_for("req-A").expect("A failure surfaced");
    let mut got_a = failed_a.block_ids.clone();
    got_a.sort();
    let mut want_a: Vec<_> = g1_a[COMPUTED_BLOCKS + LOCAL_BLOCKS..].to_vec();
    want_a.sort();
    assert_eq!(
        got_a, want_a,
        "A's failure must surface its REMOTE slice (local already onboarded Ok)"
    );

    // Resolve B's remote-slice onboard so the B pipeline reaches
    // `mark_onboarding_complete`.
    transport.wait_onboard_count(3).await;
    transport.resolve_onboard(2, Ok(()));

    wait_until(|| workers.completed_contains("req-B")).await;

    // ---------- Final assertions: A failed, B succeeded, no cascade ----------
    assert!(
        workers.failed_for("req-A").is_some(),
        "req-A must surface mark_failed_onboarding"
    );
    assert!(
        workers.failed_for("req-B").is_none(),
        "req-B MUST NOT be marked failed — sibling cleanup must not cascade"
    );
    assert!(
        workers.completed_contains("req-B"),
        "req-B must reach mark_onboarding_complete"
    );
    assert!(
        !workers.completed_contains("req-A"),
        "req-A must NOT be marked complete (it failed)"
    );

    // Coordinator state for both is gone (A via cleanup, B via
    // request_finished or the maybe_complete teardown chain).
    wait_until(|| coordinator.session_for("req-A").is_none()).await;
    assert!(
        !wrapper.has_active_cd_request("req-A"),
        "wrapper state for A must be evicted by cleanup"
    );

    Ok(())
}
