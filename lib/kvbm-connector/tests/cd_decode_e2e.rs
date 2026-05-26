// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Decode-side end-to-end test for the disagg wrapper, against
//! the symmetric `Session` API (MockSession + MockSessionFactory). No
//! `KvbmRuntime`, no `nixl_agent`, no real RDMA.
//!
//! Sequence layout (block_size = 16, total = 8 blocks):
//!
//! ```text
//! block index   :  0   1     2   3       4   5   6   7
//! [0, COMPUTED) : [c   c]    -   -       -   -   -   -          (already in G1)
//! [COMP, X)     :          [l   l]                              (local G2 hits)
//! [X, N)        :                      [r   r   r   r]         (remote prefill)
//! ```
//!
//! - COMPUTED = 2 blocks, LOCAL = 2 blocks, REMOTE = 4 blocks.
//!
//! Each test drives:
//!   GNMT (wrapper opens session, commits + makes-available local-match,
//!         queues request)
//!   USAA-1 (wrapper kicks local G2→G1, spawns the remote pull pipeline)
//!   peer commits + availability injection (mimicking prefill output)
//!   session.pull resolution
//!   remote G2→G1 onboard
//!   completion / failure / panic.

use std::sync::Arc;
use std::time::Duration;

use anyhow::Result;
use kvbm_config::{DisaggConfig, DisaggregationRole};
use kvbm_connector::G2;
use kvbm_connector::common::Request;
use kvbm_connector::connector::leader::disagg::testing::{
    InMemoryRemotePrefillQueue, MockInnerLeaderShim, MockP2pBlockTransport, MockP2pWorkerHook,
    MockSlot, TEST_BLOCK_SIZE, wait_until,
};
use kvbm_connector::connector::leader::disagg::{
    AlwaysRemote, ConditionalDisaggCoordinator, ConnectorLeaderApi, CoordinatorParts,
    DecodeDisaggLeader, HubWiring,
};
use kvbm_engine::p2p::session::{CommittedBlock, LifecycleEvent, MockSessionFactory};
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

struct TestHarness {
    wrapper: Arc<DecodeDisaggLeader>,
    inner: Arc<MockInnerLeaderShim>,
    transport: Arc<MockP2pBlockTransport>,
    workers: Arc<MockP2pWorkerHook>,
    factory: Arc<MockSessionFactory>,
    queue: Arc<InMemoryRemotePrefillQueue>,
    coordinator: Arc<ConditionalDisaggCoordinator>,
    all_hashes: Vec<kvbm_logical::SequenceHash>,
    g1_block_ids: Vec<usize>,
}

fn build_harness() -> TestHarness {
    let g2_manager = build_g2_manager(32);

    let token_sequence = create_token_sequence(TOTAL_BLOCKS, BLOCK_SIZE, 100);
    let all_hashes = generate_sequence_hashes(&token_sequence);
    let token_blocks: Vec<_> = token_sequence.blocks().to_vec();

    // Pre-allocate + register the LOCAL-match G2 blocks.
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

    let g1_block_ids: Vec<usize> = (1000..1000 + TOTAL_BLOCKS).collect();

    let slot = MockSlot {
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
        transfer_params: parking_lot::Mutex::new(None),
        ..MockSlot::default()
    };
    inner.install_slot("req-1", slot);

    let factory = MockSessionFactory::new();
    let queue = InMemoryRemotePrefillQueue::new();
    let transport = MockP2pBlockTransport::new();
    let workers = MockP2pWorkerHook::new();

    let coordinator = ConditionalDisaggCoordinator::new_with_decode(
        CoordinatorParts {
            inner: inner.clone(),
            transport: transport.clone(),
            worker_hook: workers.clone(),
            session_factory: factory.clone(),
            peer_resolver: Arc::new(
                kvbm_connector::connector::leader::p2p::peer_resolver::NoopPeerResolver,
            ),
            runtime: tokio::runtime::Handle::current(),
        },
        Arc::new(AlwaysRemote),
        queue.clone(),
    );

    let cfg = DisaggConfig {
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

    TestHarness {
        wrapper,
        inner,
        transport,
        workers,
        factory,
        queue,
        coordinator,
        all_hashes,
        g1_block_ids,
    }
}

/// Build a `Vec<CommittedBlock>` for prefill's remote-output
/// peer-block_ids, mapping each remote hash to a synthetic peer
/// block_id starting at `base`.
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

fn local_match_hashes(h: &TestHarness) -> Vec<kvbm_logical::SequenceHash> {
    h.all_hashes[COMPUTED_BLOCKS..COMPUTED_BLOCKS + LOCAL_BLOCKS].to_vec()
}

fn remote_hashes(h: &TestHarness) -> Vec<kvbm_logical::SequenceHash> {
    h.all_hashes[COMPUTED_BLOCKS + LOCAL_BLOCKS..].to_vec()
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn cd_decode_happy_path() -> Result<()> {
    let h = build_harness();

    h.wrapper.create_slot(make_request())?;

    // 1. GNMT — wrapper opens session, commits + makes-available
    //    local-match, queues remote-prefill request.
    let (count, async_flag) = h
        .wrapper
        .get_num_new_matched_tokens("req-1", COMPUTED_BLOCKS * BLOCK_SIZE)?;
    assert_eq!(count, Some((LOCAL_BLOCKS + REMOTE_BLOCKS) * BLOCK_SIZE));
    assert!(async_flag);

    wait_until(|| h.queue.snapshot().len() == 1).await;
    let queued = h.queue.snapshot();
    assert_eq!(queued[0].request_id, "req-1");
    // num_provided_tokens (DNPT) folds both the vLLM-decode G1 prefix
    // and the G2 local-match window into a single absolute-tokens
    // commitment from position 0. PLH values are NOT on the wire —
    // prefill recomputes them locally from its own slot's full hash
    // chain over the same token range.
    assert_eq!(
        queued[0].num_provided_tokens,
        (COMPUTED_BLOCKS + local_match_hashes(&h).len()) * BLOCK_SIZE
    );
    // token_ids carries the FULL prompt up to the partial tail block —
    // prefill must build its TokenBlockSequence from the same tokens
    // decode hashed so the absolute-coord PositionalLineageHash chain
    // matches. `num_computed_tokens` rides separately on the wire so
    // prefill skips actually recomputing the prefix portion.
    assert_eq!(
        queued[0].token_ids.len(),
        (COMPUTED_BLOCKS + LOCAL_BLOCKS + REMOTE_BLOCKS) * BLOCK_SIZE
    );

    let session = h.factory.last_opened().expect("decode opened a session");
    // Stage 1: the GNMT-time commit covers BOTH the planned-promoted
    // prefix and the local-match window (the prefix is in vLLM's G1
    // but not yet in decode's G2 — `MockInnerLeaderShim`'s
    // `find_prefix_g2_blocks` returns empty, which triggers the
    // promotion planner). `make_available` at GNMT still exposes
    // only the currently-G2-resident set (local match); the
    // promotion task lands the prefix G2 blocks at USAA.
    let mut expected_first_commit = h.all_hashes[..COMPUTED_BLOCKS].to_vec();
    expected_first_commit.extend(local_match_hashes(&h));
    assert_eq!(session.commit_calls(), vec![expected_first_commit]);
    assert_eq!(session.make_available_calls(), vec![local_match_hashes(&h)]);

    // 2. USAA-1 — local kick spawns; remote pull pipeline subscribes.
    h.wrapper.update_state_after_alloc(
        "req-1",
        h.g1_block_ids.clone(),
        (LOCAL_BLOCKS + REMOTE_BLOCKS) * BLOCK_SIZE,
    )?;

    h.transport.wait_onboard_count(1).await;
    let local_call = h.transport.onboard_calls()[0].clone();
    assert_eq!(local_call.src_g2_block_ids.len(), LOCAL_BLOCKS);
    assert_eq!(
        local_call.dst_g1_block_ids,
        h.g1_block_ids[COMPUTED_BLOCKS..COMPUTED_BLOCKS + LOCAL_BLOCKS].to_vec()
    );
    h.transport.resolve_onboard(0, Ok(()));

    // 3. Inject prefill peer's commits + availability for the
    //    remote slice. Coordinator drains commits then pulls.
    session.inject_peer_commit(remote_hashes(&h));
    session.inject_peer_finish_commits();
    session.inject_peer_available(committed_blocks(&remote_hashes(&h), 9000));
    session.inject_peer_drained();

    // 4. Pull fires.
    session.wait_pull_count(1).await;
    let pull = session.pull_calls()[0].clone();
    assert_eq!(pull.0, remote_hashes(&h));
    session.resolve_pull(0, Ok(()));

    // 5. Remote G2→G1 onboard.
    h.transport.wait_onboard_count(2).await;
    let remote_call = h.transport.onboard_calls()[1].clone();
    assert_eq!(remote_call.src_g2_block_ids.len(), REMOTE_BLOCKS);
    assert_eq!(
        remote_call.dst_g1_block_ids,
        h.g1_block_ids[COMPUTED_BLOCKS + LOCAL_BLOCKS..].to_vec()
    );
    h.transport.resolve_onboard(1, Ok(()));

    // 6. Completion.
    wait_until(|| h.workers.completed_contains("req-1")).await;
    assert_eq!(h.wrapper.inflight_available(), usize::MAX);
    assert!(!h.wrapper.has_active_cd_request("req-1"));

    // Stage 1: with COMPUTED_BLOCKS > 0, the GNMT-time
    // `find_prefix_g2_blocks` returns empty → a promotion task is
    // in flight. `coordinator.release` defers `session.finalize`
    // until the task completes (per session CONTRACT §2.13,
    // finalize would otherwise emit Drained and pre-empt the
    // task's make_available). Resolve the promotion here so the
    // finalize-defer chain can run to completion.
    h.inner.wait_promotion_count(1).await;
    let slot = h.inner.slot("req-1").expect("slot installed");
    let prefix_token_blocks: Vec<_> = slot.token_blocks[..COMPUTED_BLOCKS].to_vec();
    let mutables = h
        .inner
        .g2_manager()
        .allocate_blocks(COMPUTED_BLOCKS)
        .expect("allocate prefix G2");
    let completes: Vec<_> = mutables
        .into_iter()
        .zip(prefix_token_blocks.iter())
        .map(|(m, tb)| m.complete(tb).expect("complete prefix G2"))
        .collect();
    let promoted_g2 = h.inner.g2_manager().register_blocks(completes);
    h.inner.resolve_promotion(0, Ok(promoted_g2));

    // Decode now signals cooperative finalize, NOT close().
    // close() is reserved for the abort path; happy-path
    // shutdown goes through finalize() and waits for the
    // peer's symmetric finalize to fire the rendezvous.
    wait_until(|| session.finished_reason().is_some()).await;
    assert!(
        session.closed_reason().is_none(),
        "decode must NOT call session.close() on the happy path"
    );

    Ok(())
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn cd_decode_local_kick_failure() -> Result<()> {
    let h = build_harness();

    h.wrapper.create_slot(make_request())?;
    let _ = h
        .wrapper
        .get_num_new_matched_tokens("req-1", COMPUTED_BLOCKS * BLOCK_SIZE)?;

    h.wrapper.update_state_after_alloc(
        "req-1",
        h.g1_block_ids.clone(),
        (LOCAL_BLOCKS + REMOTE_BLOCKS) * BLOCK_SIZE,
    )?;

    h.transport.wait_onboard_count(1).await;
    h.transport
        .resolve_onboard(0, Err(anyhow::anyhow!("simulated local-kick failure")));

    // mark_failed_onboarding fires with all G1 ids in the
    // [COMPUTED, N) range still unfilled.
    wait_until(|| h.workers.failed_for("req-1").is_some()).await;
    let failed = h.workers.failed_for("req-1").unwrap();
    let mut got = failed.block_ids.clone();
    got.sort();
    let mut want: Vec<_> = h.g1_block_ids[COMPUTED_BLOCKS..].to_vec();
    want.sort();
    assert_eq!(got, want);
    assert_eq!(h.wrapper.inflight_available(), usize::MAX);

    Ok(())
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn cd_decode_remote_pull_failure() -> Result<()> {
    let h = build_harness();

    h.wrapper.create_slot(make_request())?;
    let _ = h
        .wrapper
        .get_num_new_matched_tokens("req-1", COMPUTED_BLOCKS * BLOCK_SIZE)?;
    let session = h.factory.last_opened().expect("session");

    h.wrapper.update_state_after_alloc(
        "req-1",
        h.g1_block_ids.clone(),
        (LOCAL_BLOCKS + REMOTE_BLOCKS) * BLOCK_SIZE,
    )?;

    h.transport.wait_onboard_count(1).await;
    h.transport.resolve_onboard(0, Ok(()));

    session.inject_peer_commit(remote_hashes(&h));
    session.inject_peer_finish_commits();
    session.inject_peer_available(committed_blocks(&remote_hashes(&h), 9000));
    session.inject_peer_drained();

    session.wait_pull_count(1).await;
    session.resolve_pull(0, Err(anyhow::anyhow!("simulated pull failure")));

    wait_until(|| h.workers.failed_for("req-1").is_some()).await;
    let failed = h.workers.failed_for("req-1").unwrap();
    let mut got = failed.block_ids.clone();
    got.sort();
    let mut want: Vec<_> = h.g1_block_ids[COMPUTED_BLOCKS + LOCAL_BLOCKS..].to_vec();
    want.sort();
    assert_eq!(got, want, "only the remote slice should be unfilled");
    assert_eq!(h.wrapper.inflight_available(), usize::MAX);

    Ok(())
}

/// Bug fix (R-B Slice 7-B post-review #3): unexpected availability
/// hash must fail the request cleanly, not panic in a spawned task.
///
/// Before the fix, an `availability` delta carrying a hash outside
/// `expected_remote_hashes` panicked inside `run_remote_pipeline`'s
/// spawned task. The panic never reached the test thread; the request
/// hung forever. After the fix, the protocol violation `bail!`s, the
/// caller's `?` routes to `cleanup_failed_request`, and vLLM is
/// unblocked via `mark_failed_onboarding`.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn cd_decode_unexpected_hash_fails_request_clean() -> Result<()> {
    let h = build_harness();

    h.wrapper.create_slot(make_request())?;
    let _ = h
        .wrapper
        .get_num_new_matched_tokens("req-1", COMPUTED_BLOCKS * BLOCK_SIZE)?;
    let session = h.factory.last_opened().expect("session");

    h.wrapper.update_state_after_alloc(
        "req-1",
        h.g1_block_ids.clone(),
        (LOCAL_BLOCKS + REMOTE_BLOCKS) * BLOCK_SIZE,
    )?;
    h.transport.wait_onboard_count(1).await;
    h.transport.resolve_onboard(0, Ok(()));

    // Drive the commits drain to completion with the legitimate
    // remote hashes so run_remote_pipeline reaches availability
    // validation. Then inject availability carrying a hash from the
    // LOCAL-match range (not in expected_remote_hashes). Must result
    // in a clean failure surfaced to vLLM, NOT a panic.
    session.inject_peer_commit(remote_hashes(&h));
    session.inject_peer_finish_commits();
    let bogus = h.all_hashes[COMPUTED_BLOCKS];
    session.inject_peer_available(vec![CommittedBlock {
        hash: bogus,
        peer_block_id: 9999,
    }]);
    session.inject_peer_drained();

    wait_until(|| h.workers.failed_for("req-1").is_some()).await;
    let failed = h.workers.failed_for("req-1").unwrap();
    let mut got = failed.block_ids.clone();
    got.sort();
    let mut want: Vec<_> = h.g1_block_ids[COMPUTED_BLOCKS + LOCAL_BLOCKS..].to_vec();
    want.sort();
    assert_eq!(
        got, want,
        "remote slice must be marked failed (local match was already onboarded)"
    );
    Ok(())
}

/// Bug fix (review round 2 #P0): pre-USAA failure must NOT emit
/// `mark_failed_onboarding(rid, [])`. vLLM's connector contract treats
/// an empty `failed_block_ids` plus `finished_recving` as a successful
/// async load. Pre-USAA failures must be stashed and replayed at USAA
/// time with the now-known G1 ids.
///
/// Test scenario:
/// 1. Drive gnmt to install CD state. USAA has not been called.
/// 2. `coordinator.mark_failed` (mirrors queue-enqueue-failure path).
/// 3. Assert: `mark_failed_onboarding` is NOT yet called (failure stashed).
/// 4. Drive USAA with G1 destinations.
/// 5. Assert: `mark_failed_onboarding` is called exactly once with
///    those G1 ids and the request_id.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn cd_decode_pre_usaa_failure_stashes_until_usaa() -> Result<()> {
    let h = build_harness();

    h.wrapper.create_slot(make_request())?;
    let _ = h
        .wrapper
        .get_num_new_matched_tokens("req-1", COMPUTED_BLOCKS * BLOCK_SIZE)?;
    assert!(h.wrapper.has_active_cd_request("req-1"));

    // Pre-USAA failure stashes — must NOT emit mark_failed_onboarding.
    h.coordinator
        .mark_failed("req-1", "simulated queue enqueue failure".to_string());

    // Give the failure-sink chain time to run (it may spawn the cleanup
    // task; we want to confirm it lands at "stash" not "emit").
    tokio::time::sleep(Duration::from_millis(100)).await;
    assert!(
        h.workers.failed_for("req-1").is_none(),
        "pre-USAA failure must NOT emit mark_failed_onboarding (vLLM treats \
         empty failed_block_ids as success); failure must be stashed"
    );

    // Now drive USAA. The replay path should fire mark_failed_onboarding
    // with the just-arrived G1 ids and skip the rest of USAA bookkeeping.
    let _ = h.wrapper.update_state_after_alloc(
        "req-1",
        h.g1_block_ids.clone(),
        (LOCAL_BLOCKS + REMOTE_BLOCKS) * BLOCK_SIZE,
    );

    wait_until(|| h.workers.failed_for("req-1").is_some()).await;
    let failed = h.workers.failed_for("req-1").unwrap();
    let mut got = failed.block_ids.clone();
    got.sort();
    // Only the EXTERNAL slice (local + remote, excluding the
    // computed prefix) must be reported failed. vLLM truncates
    // request.num_computed_tokens at the first invalid block;
    // reporting the entire `block_ids` would force recomputation
    // from token 0 instead of just the external-load range.
    let mut want: Vec<_> = h.g1_block_ids[COMPUTED_BLOCKS..].to_vec();
    want.sort();
    assert_eq!(
        got, want,
        "USAA replay must emit ONLY the external G1 slice (excluding computed prefix)"
    );
    Ok(())
}

/// Bug fix (review round 2 #P1#1): GNMT must be idempotent.
///
/// vLLM may call gnmt multiple times for the same request without an
/// intervening USAA (allocation can fail after gnmt). The wrapper must
/// not double-invoke inner.gnmt or duplicate side effects (open a
/// second session, queue a second enqueue, etc.). Verify by counting
/// inner.gnmt calls across two wrapper.gnmt invocations.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn cd_decode_gnmt_called_twice_inner_called_once() -> Result<()> {
    let h = build_harness();
    h.wrapper.create_slot(make_request())?;

    let r1 = h
        .wrapper
        .get_num_new_matched_tokens("req-1", COMPUTED_BLOCKS * BLOCK_SIZE)?;
    let r2 = h
        .wrapper
        .get_num_new_matched_tokens("req-1", COMPUTED_BLOCKS * BLOCK_SIZE)?;
    assert_eq!(r1, r2, "idempotent gnmt must return identical tuples");

    // Inner.gnmt was called by the FIRST wrapper.gnmt only — the
    // second hit the cd_request_state cache and short-circuited.
    let inner_calls = h.inner.gnmt_call_count("req-1");
    assert_eq!(
        inner_calls, 1,
        "inner.gnmt must be called exactly once across two wrapper.gnmt calls (got {})",
        inner_calls
    );

    // Side effects (session opened, request queued) must also be once.
    assert_eq!(
        h.factory.open_count(),
        1,
        "session must be opened exactly once across two wrapper.gnmt calls"
    );
    wait_until(|| h.queue.snapshot().len() == 1).await;
    assert_eq!(
        h.queue.snapshot().len(),
        1,
        "remote-prefill request must be enqueued exactly once"
    );

    Ok(())
}

/// Bug fix (R-B Slice 7-B post-review #4): availability deltas covering
/// non-contiguous slot indices must be handled (split into contiguous
/// runs and processed individually), not rejected as a protocol error.
///
/// Before the fix, `pull_register_onboard_chunk` required all hashes
/// in a single call to be a contiguous slot range. The session API does
/// not guarantee contiguous availability — sparse or coalesced deltas
/// are valid shapes for distributed prefill. After the fix, the helper
/// splits non-contiguous input into maximal runs and processes each.
///
/// Test scenario: emit a single Available delta with the FIRST and
/// THIRD remote hashes (slot indices `[0, 2]`), then a second delta
/// with the SECOND and FOURTH (slot indices `[1, 3]`). The first delta
/// is non-contiguous; before the fix it bailed.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn cd_decode_non_contiguous_availability_succeeds() -> Result<()> {
    let h = build_harness();

    h.wrapper.create_slot(make_request())?;
    let _ = h
        .wrapper
        .get_num_new_matched_tokens("req-1", COMPUTED_BLOCKS * BLOCK_SIZE)?;
    let session = h.factory.last_opened().expect("session");

    h.wrapper.update_state_after_alloc(
        "req-1",
        h.g1_block_ids.clone(),
        (LOCAL_BLOCKS + REMOTE_BLOCKS) * BLOCK_SIZE,
    )?;
    h.transport.wait_onboard_count(1).await;
    h.transport.resolve_onboard(0, Ok(()));

    let remote = remote_hashes(&h);
    assert!(remote.len() >= 4, "test assumes ≥4 remote hashes");
    session.inject_peer_commit(remote.clone());
    session.inject_peer_finish_commits();

    // Single Available delta carrying slots [0, 2, 1, 3] — neither
    // contiguous nor sorted on the wire. Helper must sort by slot
    // index, then split into contiguous runs and process each.
    // After sort: [0, 1, 2, 3]. Split: one run of length 4 (all
    // contiguous after sort) → 1 pull. The reorder step exercises
    // the slot-index reorder path; the split step is exercised by
    // the next assertion below (sparse availability).
    session.inject_peer_available(vec![
        CommittedBlock {
            hash: remote[2],
            peer_block_id: 9002,
        },
        CommittedBlock {
            hash: remote[0],
            peer_block_id: 9000,
        },
        CommittedBlock {
            hash: remote[3],
            peer_block_id: 9003,
        },
        CommittedBlock {
            hash: remote[1],
            peer_block_id: 9001,
        },
    ]);
    session.inject_peer_drained();

    session.wait_pull_count(1).await;
    let pull = session.pull_calls()[0].clone();
    // After sort: hashes are in slot-index order (= sequence order)
    // for the contiguous run [0..4].
    assert_eq!(pull.0, remote, "pull must receive slot-ordered hashes");
    session.resolve_pull(0, Ok(()));

    h.transport.wait_onboard_count(2).await; // +1 for the local kick
    h.transport.resolve_onboard(1, Ok(()));

    wait_until(|| h.workers.completed_contains("req-1")).await;
    Ok(())
}

/// Bug fix (R-B Slice 7-B post-review #4) — sparse / non-contiguous
/// variant.
///
/// Availability arrives in a shape that *cannot* be coalesced into one
/// contiguous run even after sorting: slots `[0, 2]` arrive in the
/// first delta (skipping slot 1), then `[1, 3]` arrives in the second.
/// Each non-contiguous delta splits into 2 single-slot runs → 4 pulls
/// total. Before the fix, the first delta bailed with "non-contiguous
/// chunk".
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn cd_decode_sparse_availability_succeeds() -> Result<()> {
    let h = build_harness();

    h.wrapper.create_slot(make_request())?;
    let _ = h
        .wrapper
        .get_num_new_matched_tokens("req-1", COMPUTED_BLOCKS * BLOCK_SIZE)?;
    let session = h.factory.last_opened().expect("session");

    h.wrapper.update_state_after_alloc(
        "req-1",
        h.g1_block_ids.clone(),
        (LOCAL_BLOCKS + REMOTE_BLOCKS) * BLOCK_SIZE,
    )?;
    h.transport.wait_onboard_count(1).await;
    h.transport.resolve_onboard(0, Ok(()));

    let remote = remote_hashes(&h);
    assert!(remote.len() >= 4, "test assumes ≥4 remote hashes");
    session.inject_peer_commit(remote.clone());
    session.inject_peer_finish_commits();

    // First Available: slots [0, 2] — skipping slot 1. Splits into
    // runs [0] and [2] (sequentially: pull #1 then pull #2 after run 1
    // resolves).
    session.inject_peer_available(vec![
        CommittedBlock {
            hash: remote[0],
            peer_block_id: 9000,
        },
        CommittedBlock {
            hash: remote[2],
            peer_block_id: 9002,
        },
    ]);

    session.wait_pull_count(1).await;
    session.resolve_pull(0, Ok(()));
    h.transport.wait_onboard_count(2).await;
    h.transport.resolve_onboard(1, Ok(()));

    session.wait_pull_count(2).await;
    session.resolve_pull(1, Ok(()));
    h.transport.wait_onboard_count(3).await;
    h.transport.resolve_onboard(2, Ok(()));

    // Second Available: slots [1, 3] — also non-contiguous.
    session.inject_peer_available(vec![
        CommittedBlock {
            hash: remote[1],
            peer_block_id: 9001,
        },
        CommittedBlock {
            hash: remote[3],
            peer_block_id: 9003,
        },
    ]);
    session.inject_peer_drained();

    session.wait_pull_count(3).await;
    session.resolve_pull(2, Ok(()));
    h.transport.wait_onboard_count(4).await;
    h.transport.resolve_onboard(3, Ok(()));

    session.wait_pull_count(4).await;
    session.resolve_pull(3, Ok(()));
    h.transport.wait_onboard_count(5).await;
    h.transport.resolve_onboard(4, Ok(()));

    wait_until(|| h.workers.completed_contains("req-1")).await;
    Ok(())
}

/// Bug fix (R-B Slice 7-B post-review #6): `session.pull` returning
/// the wrong number of blocks must error explicitly, not silently
/// truncate via `zip`.
///
/// Exercises the real length-check path (Ok with short result), not
/// just the cleanup chain. `resolve_pull_short` returns 3 mutables
/// out of the requested 4; the helper's `filled.len() != chunk_size`
/// guard must bail.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn cd_decode_session_pull_length_mismatch_errors_clean() -> Result<()> {
    let h = build_harness();

    h.wrapper.create_slot(make_request())?;
    let _ = h
        .wrapper
        .get_num_new_matched_tokens("req-1", COMPUTED_BLOCKS * BLOCK_SIZE)?;
    let session = h.factory.last_opened().expect("session");

    h.wrapper.update_state_after_alloc(
        "req-1",
        h.g1_block_ids.clone(),
        (LOCAL_BLOCKS + REMOTE_BLOCKS) * BLOCK_SIZE,
    )?;
    h.transport.wait_onboard_count(1).await;
    h.transport.resolve_onboard(0, Ok(()));

    // Single Available covers all 4 remote blocks → one pull of
    // chunk_size=4. resolve_pull_short returns only 3 mutables.
    let remote = remote_hashes(&h);
    session.inject_peer_commit(remote.clone());
    session.inject_peer_finish_commits();
    session.inject_peer_available(committed_blocks(&remote, 9000));
    session.inject_peer_drained();

    session.wait_pull_count(1).await;
    session.resolve_pull_short(0, REMOTE_BLOCKS - 1);

    // Failure surfaces via cleanup → mark_failed_onboarding with the
    // remote G1 slice (post-USAA path, since USAA already ran).
    wait_until(|| h.workers.failed_for("req-1").is_some()).await;
    let failed = h.workers.failed_for("req-1").unwrap();
    let mut got = failed.block_ids.clone();
    got.sort();
    let mut want: Vec<_> = h.g1_block_ids[COMPUTED_BLOCKS + LOCAL_BLOCKS..].to_vec();
    want.sort();
    assert_eq!(got, want, "remote slice must be marked failed");
    Ok(())
}

/// Slice C — peer signals `Closed` on the commits stream before
/// all expected remote hashes have arrived. Decode treats this as
/// a protocol-level failure: prefill said "no more commits coming"
/// while still owing decode block_ids that vLLM's G1 destinations
/// depend on. `cleanup_failed_request` fires; vLLM aborts the
/// remote slice.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn cd_decode_commits_closed_short_fails_request() -> Result<()> {
    let h = build_harness();

    h.wrapper.create_slot(make_request())?;
    let _ = h
        .wrapper
        .get_num_new_matched_tokens("req-1", COMPUTED_BLOCKS * BLOCK_SIZE)?;
    let session = h.factory.last_opened().expect("session");

    h.wrapper.update_state_after_alloc(
        "req-1",
        h.g1_block_ids.clone(),
        (LOCAL_BLOCKS + REMOTE_BLOCKS) * BLOCK_SIZE,
    )?;

    h.transport.wait_onboard_count(1).await;
    h.transport.resolve_onboard(0, Ok(()));

    // Commit only HALF the remote hashes, then immediately Close.
    // run_remote_pipeline detects the deficit and bails before
    // touching availability.
    let remote = remote_hashes(&h);
    assert!(remote.len() >= 2, "test assumes ≥2 remote hashes");
    let half = remote.len() / 2;
    session.inject_peer_commit(remote[..half].to_vec());
    session.inject_peer_finish_commits();

    wait_until(|| h.workers.failed_for("req-1").is_some()).await;
    let failed = h.workers.failed_for("req-1").unwrap();
    let mut got = failed.block_ids.clone();
    got.sort();
    let mut want: Vec<_> = h.g1_block_ids[COMPUTED_BLOCKS + LOCAL_BLOCKS..].to_vec();
    want.sort();
    assert_eq!(
        got, want,
        "remote slice must be marked failed (local match was already onboarded)"
    );
    assert_eq!(h.wrapper.inflight_available(), usize::MAX);

    Ok(())
}

/// Concurrent `cleanup_failed_request` must invoke
/// `mark_failed_onboarding` at most once for the same request_id.
/// Five connector-spawned paths race here in production: the lifecycle
/// watcher's failure-sink route, commit_usaa1's local-kick / remote-
/// pipeline / session-missing spawns, and the enqueue-spawn Err.
/// Without the `cleanup_claimed` CAS on `CdRequestState`, two paths
/// can both observe non-empty `unfilled_ids` and both push a
/// `mark_failed_onboarding` call before either's `release_request`
/// removes the wrapper entry — vLLM gets double-notified for the
/// same failed G1 window.
///
/// Reproducer-first: with N=8 concurrent spawns on a multi-thread
/// runtime, the race reliably fires pre-CAS (verified by probing
/// with the guard temporarily disabled — got 2 calls instead of 1).
/// With the CAS, no concurrent invocation count produces more than
/// one `mark_failed_onboarding` call.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn cd_decode_concurrent_cleanup_marks_failed_once() -> Result<()> {
    use kvbm_connector::connector::leader::disagg::CdFailureSink;

    let h = build_harness();
    h.wrapper.create_slot(make_request())?;
    let _ = h
        .wrapper
        .get_num_new_matched_tokens("req-1", COMPUTED_BLOCKS * BLOCK_SIZE)?;
    h.wrapper.update_state_after_alloc(
        "req-1",
        h.g1_block_ids.clone(),
        (LOCAL_BLOCKS + REMOTE_BLOCKS) * BLOCK_SIZE,
    )?;

    const N: usize = 8;
    let sink: Arc<dyn CdFailureSink> = Arc::clone(&h.wrapper) as Arc<dyn CdFailureSink>;
    let mut handles = Vec::with_capacity(N);
    for i in 0..N {
        let s = Arc::clone(&sink);
        handles.push(tokio::spawn(async move {
            s.on_session_failure("req-1".to_string(), format!("race-{i}"))
                .await
        }));
    }
    for h_ in handles {
        let _ = h_.await;
    }

    let calls: Vec<_> = h
        .workers
        .failed()
        .into_iter()
        .filter(|c| c.request_id == "req-1")
        .collect();
    assert_eq!(
        calls.len(),
        1,
        "{N} concurrent cleanup_failed_request invocations must collapse to a single \
         mark_failed_onboarding call (got {})",
        calls.len(),
    );

    let mut got = calls[0].block_ids.clone();
    got.sort();
    let mut want: Vec<_> = h.g1_block_ids[COMPUTED_BLOCKS..].to_vec();
    want.sort();
    assert_eq!(got, want);
    assert_eq!(h.wrapper.inflight_available(), usize::MAX);

    Ok(())
}

/// `commit_usaa1`'s state rebuild must START the new state with
/// `cleanup_claimed=false`, regardless of the existing state's value.
///
/// Bug it prevents: `decode_usaa` reads `pending_failure` once at the
/// top, then calls `commit_usaa1` which reads it again at rebuild
/// time. A concurrent `cleanup_failed_request` that fires between
/// those two reads stashes `pending_failure=Some` AND sets the
/// existing state's `cleanup_claimed=true`. If commit_usaa1 threads
/// the flag forward, the new state inherits `cleanup_claimed=true`
/// while also carrying the freshly-stashed `pending_failure`. The
/// replay branch in `decode_usaa` was already bypassed, and no
/// future `cleanup_failed_request` can pass the CAS to surface the
/// stash — vLLM is never notified of the failure.
///
/// This test simulates the race without timing fragility by
/// directly mutating the existing state's `cleanup_claimed` via a
/// test-only accessor before driving USAA. Asserts the post-USAA
/// state's flag is `false` so a subsequent failure can be reported.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn cd_decode_usaa1_rebuild_resets_cleanup_claimed() -> Result<()> {
    let h = build_harness();
    h.wrapper.create_slot(make_request())?;
    let _ = h
        .wrapper
        .get_num_new_matched_tokens("req-1", COMPUTED_BLOCKS * BLOCK_SIZE)?;

    // After GNMT, the wrapper has the gnmt-time state with
    // cleanup_claimed=false. Force it to true to simulate a
    // concurrent cleanup_failed_request landing between
    // decode_usaa's pending-failure check and commit_usaa1's
    // rebuild.
    assert_eq!(
        h.wrapper.cleanup_claimed_for_test("req-1"),
        Some(false),
        "gnmt-time state must start with cleanup_claimed=false"
    );
    assert!(
        h.wrapper.force_cleanup_claimed_for_test("req-1", true),
        "force_cleanup_claimed_for_test must find the gnmt-time entry"
    );

    h.wrapper.update_state_after_alloc(
        "req-1",
        h.g1_block_ids.clone(),
        (LOCAL_BLOCKS + REMOTE_BLOCKS) * BLOCK_SIZE,
    )?;

    // Post-USAA: the new state must start with cleanup_claimed=false
    // so a future cleanup_failed_request can pass the CAS.
    assert_eq!(
        h.wrapper.cleanup_claimed_for_test("req-1"),
        Some(false),
        "USAA-1 rebuild must reset cleanup_claimed to false; pre-fix would thread \
         the existing `true` forward and silently swallow future failures"
    );

    // Exercise the recovered cleanup path end-to-end: forcing the
    // local-kick to fail must surface mark_failed_onboarding.
    h.transport.wait_onboard_count(1).await;
    h.transport
        .resolve_onboard(0, Err(anyhow::anyhow!("post-rebuild local kick failure")));
    wait_until(|| h.workers.failed_for("req-1").is_some()).await;
    let failed = h.workers.failed_for("req-1").expect("failure surfaced");
    assert!(
        !failed.block_ids.is_empty(),
        "mark_failed_onboarding must be called with the unfilled G1 slice"
    );

    Ok(())
}

/// `commit_usaa1` must re-check `pending_failure` after reading the
/// existing `cd_request_state` entry. Without the re-check, a
/// concurrent `cleanup_failed_request` that stashed a failure
/// between `decode_usaa`'s pending-check and `commit_usaa1`'s read
/// would proceed to apply block assignments, drain
/// `local_match_g2_pins`, build `remote_slots`, and spawn the
/// local-kick + remote-pipeline. If those happen to complete
/// successfully, `maybe_complete` fires `mark_onboarding_complete`
/// — vLLM is told the load SUCCEEDED for a request that was
/// supposed to be reported as a failure. The stash is never
/// surfaced.
///
/// Test enters `commit_usaa1` directly via a feature-gated test
/// helper to bypass `decode_usaa`'s outer pending_failure check —
/// the production race requires a stash to be observed by
/// commit_usaa1 but NOT by decode_usaa, which is microsecond
/// timing-fragile in real production code but trivially simulated
/// here.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn cd_decode_commit_usaa1_replays_late_pending_failure() -> Result<()> {
    let h = build_harness();
    h.wrapper.create_slot(make_request())?;
    let _ = h
        .wrapper
        .get_num_new_matched_tokens("req-1", COMPUTED_BLOCKS * BLOCK_SIZE)?;

    // Simulate a racing cleanup_failed_request landing between
    // decode_usaa's pending-failure check (None) and
    // commit_usaa1's read.
    assert!(
        h.wrapper
            .force_pending_failure_for_test("req-1", Some("late race failure".to_string())),
        "test accessor must find the gnmt-time entry"
    );

    // Enter commit_usaa1 directly — production decode_usaa would
    // catch the stash and replay there, but we are exercising the
    // race window inside commit_usaa1 itself.
    h.wrapper.commit_usaa1_for_test(
        "req-1",
        h.g1_block_ids.clone(),
        (LOCAL_BLOCKS + REMOTE_BLOCKS) * BLOCK_SIZE,
    )?;

    // Post-commit_usaa1: the inner replay path must have emitted
    // mark_failed_onboarding. The happy-path local-kick must NOT
    // have spawned — if it had, mark_onboarding_complete could
    // race the failure and report success.
    wait_until(|| h.workers.failed_for("req-1").is_some()).await;
    let failed = h.workers.failed_for("req-1").expect("failure surfaced");

    let mut got = failed.block_ids.clone();
    got.sort();
    let mut want: Vec<_> = h.g1_block_ids[COMPUTED_BLOCKS..].to_vec();
    want.sort();
    assert_eq!(
        got, want,
        "commit_usaa1 replay must report the full external G1 slice as failed"
    );

    // Confirm the happy path did NOT race to completion: no
    // mark_onboarding_complete for this request.
    assert!(
        !h.workers.completed_contains("req-1"),
        "mark_onboarding_complete must NOT fire for a stashed-failure request — \
         pre-fix the local-kick + remote-pipeline would spawn and could race the \
         failure to report SUCCESS to vLLM",
    );

    // Budget released.
    assert_eq!(h.wrapper.inflight_available(), usize::MAX);

    Ok(())
}

/// `commit_usaa1` must check `existing.pending_failure` AFTER the
/// insert of the rebuilt state, not just before. A
/// `cleanup_failed_request` that fires between the outer
/// pending_failure re-check and the insert stashes a failure on
/// EXISTING (OLD state, still reachable via `cd_request_state.get`
/// pre-insert, takes the pre-USAA branch because OLD has empty
/// remote_slots). After commit_usaa1's insert, OLD is unreachable
/// via DashMap but commit_usaa1 still holds the OLD Arc via
/// `existing`. Without the post-insert re-check, that stash is
/// orphaned, the pipelines spawn against NEW state, may complete
/// successfully, fire `mark_onboarding_complete`, and vLLM sees
/// SUCCESS for a failed request.
///
/// Test injects the stash via a hook on
/// `MockInnerLeaderShim::apply_block_assignments`, which is called
/// inside `commit_usaa1` AFTER the outer re-check and BEFORE the
/// insert — the exact race window.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn cd_decode_commit_usaa1_post_insert_replays_late_stash() -> Result<()> {
    let h = build_harness();
    h.wrapper.create_slot(make_request())?;
    let _ = h
        .wrapper
        .get_num_new_matched_tokens("req-1", COMPUTED_BLOCKS * BLOCK_SIZE)?;

    // Install hook: when commit_usaa1 calls
    // self.inner.apply_block_assignments (between the outer
    // pending_failure re-check and the rebuild's insert), force a
    // stash on the wrapper's cd_request_state entry. Models a
    // cleanup_failed_request landing in that race window.
    let wrapper_weak = Arc::downgrade(&h.wrapper);
    h.inner.set_apply_block_assignments_hook(Arc::new(move || {
        if let Some(wrapper) = wrapper_weak.upgrade() {
            wrapper.force_pending_failure_for_test(
                "req-1",
                Some("race in apply_block_assignments window".to_string()),
            );
        }
    }));

    // Drive commit_usaa1 directly so the outer pending_failure check
    // sees None (matches the production race semantics — the stash
    // lands AFTER decode_usaa's outer check).
    h.wrapper.commit_usaa1_for_test(
        "req-1",
        h.g1_block_ids.clone(),
        (LOCAL_BLOCKS + REMOTE_BLOCKS) * BLOCK_SIZE,
    )?;

    // Post-insert replay must have surfaced the stash to vLLM.
    wait_until(|| h.workers.failed_for("req-1").is_some()).await;
    let failed = h.workers.failed_for("req-1").expect("failure surfaced");
    let mut got = failed.block_ids.clone();
    got.sort();
    let mut want: Vec<_> = h.g1_block_ids[COMPUTED_BLOCKS..].to_vec();
    want.sort();
    assert_eq!(
        got, want,
        "post-insert replay must report the full external G1 slice as failed"
    );

    // Critical: mark_onboarding_complete must NOT fire. Pre-fix
    // the pipelines would spawn against the freshly-inserted NEW
    // state and could race to completion, reporting SUCCESS.
    assert!(
        !h.workers.completed_contains("req-1"),
        "mark_onboarding_complete must NOT fire for a stashed-failure request"
    );

    assert_eq!(h.wrapper.inflight_available(), usize::MAX);
    Ok(())
}

/// P0 #9 — Decode-side mid-output peer failure. When the prefill
/// peer crashes (or velo surfaces `Frame::Error`) mid-pull, the
/// decode side observes `LifecycleEvent::Failed` on its session.
/// The chain that follows:
///
/// 1. Decode's lifecycle watcher (`spawn_lifecycle_watcher` with
///    role="decode") routes Failed through
///    `invoke_decode_failure_sink_and_evict`.
/// 2. The failure sink calls
///    `DecodeDisaggLeader::cleanup_failed_request`.
/// 3. The wrapper's `cleanup_claimed` CAS (added 8e9fc64a691) holds
///    against concurrent cleanup callers.
/// 4. Post-USAA branch emits `mark_failed_onboarding` with the
///    unfilled external G1 slice; `release_request` removes the
///    wrapper state + releases the inflight budget;
///    `coordinator.release` removes coord state + finalizes the
///    session.
///
/// The test gates the lifecycle path in two stages:
///
/// - Stage 1: inject ONE `LifecycleEvent::Failed` and wait for
///   `mark_failed_onboarding`. No other test code touches cleanup yet.
///   Asserts the outcome contract (peer failure → cleanup) holds.
///   *Caveat:* on the decode side this outcome is reachable via two
///   internal paths — (a) the lifecycle-failure-sink route, and
///   (b) the watcher's cooperative-branch `state.cancel.cancel()`
///   propagating through `run_remote_pipeline`'s `tokio::select!` →
///   spawn-catch → `wrapper.cleanup_failed_request`. This test gates
///   the outcome regardless of which path won. The lifecycle wire
///   itself (`Self::lifecycle_failure_reason`) is shared with the
///   prefill watcher and is independently gated by the prefill-side
///   peer-failure test in `cd_prefill_e2e.rs`.
/// - Stage 2: fire a second `on_session_failure` after the lifecycle
///   path already claimed the CAS. The duplicate must be a no-op
///   (CAS short-circuits OR wrapper state already absent).
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn cd_decode_peer_failure_mid_output_cleans_up() -> Result<()> {
    use kvbm_connector::connector::leader::disagg::CdFailureSink;

    let h = build_harness();

    h.wrapper.create_slot(make_request())?;
    let _ = h
        .wrapper
        .get_num_new_matched_tokens("req-1", COMPUTED_BLOCKS * BLOCK_SIZE)?;
    let session = h.factory.last_opened().expect("decode opened a session");

    // USAA-1 installs the full G1 window. Wait for the local-kick
    // spawn to land but do NOT resolve it — we want the failure to
    // fire mid-flight with unfilled G1 ids on both the local-match
    // slice AND the remote slice.
    h.wrapper.update_state_after_alloc(
        "req-1",
        h.g1_block_ids.clone(),
        (LOCAL_BLOCKS + REMOTE_BLOCKS) * BLOCK_SIZE,
    )?;
    h.transport.wait_onboard_count(1).await;

    // Stage 1 — lifecycle path is the ONLY trigger.
    assert!(h.workers.failed_for("req-1").is_none());
    session.inject_lifecycle(LifecycleEvent::Failed {
        reason: "induced prefill peer crash mid-pull".to_string(),
    });
    wait_until(|| h.workers.failed_for("req-1").is_some()).await;
    let after_lifecycle: Vec<_> = h
        .workers
        .failed()
        .into_iter()
        .filter(|c| c.request_id == "req-1")
        .collect();
    assert_eq!(
        after_lifecycle.len(),
        1,
        "lifecycle Failed event alone must produce ONE mark_failed_onboarding (got {})",
        after_lifecycle.len(),
    );

    // The unfilled set covers the full external G1 slice (local-match
    // + remote) because neither local-kick nor remote-pipeline has
    // completed.
    let mut got = after_lifecycle[0].block_ids.clone();
    got.sort();
    let mut want: Vec<_> = h.g1_block_ids[COMPUTED_BLOCKS..].to_vec();
    want.sort();
    assert_eq!(got, want);

    // Wrapper state + budget reclaimed; coordinator state evicted.
    wait_until(|| !h.wrapper.has_active_cd_request("req-1")).await;
    assert_eq!(h.wrapper.inflight_available(), usize::MAX);
    wait_until(|| h.coordinator.active_count() == 0).await;

    // Stage 2 — duplicate cleanup via the failure-sink is a no-op.
    // After Stage 1, the wrapper's cd_request_state is empty so the
    // CAS check at the top of cleanup_failed_request short-circuits;
    // a duplicate call must NOT re-emit mark_failed_onboarding.
    let sink: Arc<dyn CdFailureSink> = Arc::clone(&h.wrapper) as Arc<dyn CdFailureSink>;
    sink.on_session_failure("req-1".to_string(), "duplicate via direct sink".to_string())
        .await;
    let after_dup = h
        .workers
        .failed()
        .into_iter()
        .filter(|c| c.request_id == "req-1")
        .count();
    assert_eq!(
        after_dup, 1,
        "duplicate on_session_failure must be a no-op (got {})",
        after_dup,
    );

    // mark_onboarding_complete must NOT fire.
    assert!(
        !h.workers.completed_contains("req-1"),
        "failed request must not also signal onboarding-complete to vLLM"
    );

    Ok(())
}

fn prefix_hashes(h: &TestHarness) -> Vec<kvbm_logical::SequenceHash> {
    h.all_hashes[..COMPUTED_BLOCKS].to_vec()
}

/// Stage 1: at decode GNMT, when vLLM reports a non-zero
/// `num_computed_tokens` (vLLM holds the prefix in G1) but the G2
/// cache has no record of the prefix (the all-or-nothing
/// `find_prefix_g2_blocks` returned empty), the wrapper must
/// arrange for the prefix blocks to be promoted G1→G2 at USAA and
/// the session's committed set must widen to include them.
///
/// What this test pins:
///
///   1. GNMT-time `session.commit` includes BOTH the local-match
///      hashes AND the planned-promoted prefix hashes. The
///      promoted hashes are committed up-front because the
///      `session.finish_commits` seal is taken later in the same
///      flow — the prefill peer must see the full promised set
///      before commits is sealed.
///   2. GNMT-time `session.make_available` exposes ONLY the
///      currently-G2-resident blocks (local match here). Promoted
///      prefix blocks are not yet G2-resident.
///   3. `session.finish_commits` fires at GNMT (commit set is
///      locked once we know the planned promotion).
///   4. `session.finish_availability` is DEFERRED — the prefill
///      peer would stall if it observed `finish_availability`
///      before the promoted G2 blocks are exposed.
///   5. At USAA, the wrapper invokes `promote_g1_to_g2` with the
///      `(block_id, sequence_hash)` pairs covering the prefix
///      window — these come from `block_ids[..num_prefix_blocks]`
///      paired with `all_sequence_hashes[..num_prefix_blocks]`.
///   6. When the promotion task resolves with the registered G2
///      blocks, the wrapper calls `session.make_available` with
///      them and then `session.finish_availability`.
///
/// This test is part of the Stage 1 reproducer-first scaffold and
/// MUST FAIL on the pre-Stage-1 codebase (the wrapper today
/// advertises an empty prefix on `find_prefix_g2_blocks` miss).
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn cd_decode_promotes_g1_prefix_to_g2_at_usaa() -> Result<()> {
    let h = build_harness();

    h.wrapper.create_slot(make_request())?;

    // 1. GNMT — wrapper opens session.
    let (count, async_flag) = h
        .wrapper
        .get_num_new_matched_tokens("req-1", COMPUTED_BLOCKS * BLOCK_SIZE)?;
    assert_eq!(count, Some((LOCAL_BLOCKS + REMOTE_BLOCKS) * BLOCK_SIZE));
    assert!(async_flag);

    let session = h.factory.last_opened().expect("decode opened a session");

    // The first commit must already carry BOTH the local-match
    // hashes and the planned-promoted prefix hashes — Stage 1
    // commits the full planned set up-front so prefill peers can
    // observe a sealed committed set after `finish_commits`.
    let mut expected_first_commit = prefix_hashes(&h);
    expected_first_commit.extend(local_match_hashes(&h));
    assert_eq!(
        session.commit_calls(),
        vec![expected_first_commit],
        "GNMT must commit planned prefix promotion alongside local match",
    );

    // make_available at GNMT exposes only the G2-resident set
    // (local match only — prefix promotion hasn't run yet).
    assert_eq!(
        session.make_available_calls(),
        vec![local_match_hashes(&h)],
        "GNMT must NOT expose planned-promotion blocks before they land",
    );

    assert!(
        session.finish_commits_called(),
        "GNMT must seal commits once the planned set is committed",
    );
    assert!(
        !session.finish_availability_called(),
        "GNMT must DEFER finish_availability — promotion has not landed",
    );

    assert_eq!(
        h.inner.promotion_count(),
        0,
        "promotion is triggered at USAA, not GNMT",
    );

    // 2. USAA — wrapper kicks the local G2→G1 transfer AND issues
    //    the G1→G2 prefix promotion request.
    h.wrapper.update_state_after_alloc(
        "req-1",
        h.g1_block_ids.clone(),
        (LOCAL_BLOCKS + REMOTE_BLOCKS) * BLOCK_SIZE,
    )?;

    // Allow the wrapper's USAA bookkeeping + the promotion-task
    // spawn to settle.
    h.inner.wait_promotion_count(1).await;

    let (source_block_ids, expected_hashes) = h
        .inner
        .snapshot_promotion(0)
        .expect("promotion #0 must be recorded");
    assert_eq!(
        source_block_ids,
        h.g1_block_ids[..COMPUTED_BLOCKS].to_vec(),
        "promotion source block ids must be the prefix slice of vLLM's g1 assignment",
    );
    assert_eq!(
        expected_hashes,
        prefix_hashes(&h),
        "promotion expected hashes must be the prefix hash range",
    );

    // 3. Build genuine G2 ImmutableBlocks for the prefix hashes
    //    and resolve the promotion. Mirrors what the production
    //    offload-pipeline's register step would have done.
    let slot = h.inner.slot("req-1").expect("slot installed");
    let prefix_token_blocks: Vec<_> = slot.token_blocks[..COMPUTED_BLOCKS].to_vec();
    let mutables = h
        .inner
        .g2_manager()
        .allocate_blocks(COMPUTED_BLOCKS)
        .expect("allocate prefix G2");
    let completes: Vec<_> = mutables
        .into_iter()
        .zip(prefix_token_blocks.iter())
        .map(|(m, tb)| m.complete(tb).expect("complete prefix G2"))
        .collect();
    let promoted_g2 = h.inner.g2_manager().register_blocks(completes);
    assert_eq!(promoted_g2.len(), COMPUTED_BLOCKS);

    h.inner.resolve_promotion(0, Ok(promoted_g2));

    // 4. After the promotion task drives session.make_available +
    //    session.finish_availability, both must be observable.
    wait_until(|| session.finish_availability_called()).await;

    // The second make_available call covers the promoted prefix.
    let make_avail = session.make_available_calls();
    assert_eq!(
        make_avail.len(),
        2,
        "expected GNMT-time + USAA-promotion make_available; got: {:?}",
        make_avail,
    );
    assert_eq!(
        make_avail[0],
        local_match_hashes(&h),
        "first make_available is local match at GNMT",
    );
    assert_eq!(
        make_avail[1],
        prefix_hashes(&h),
        "second make_available is the promoted prefix",
    );

    Ok(())
}

/// Stage 1 race: `request_finished` (and the implicit drop chain
/// `CdRequestStatePayload::Drop → coordinator.release →
/// session.finalize`) MUST NOT pre-empt an in-flight prefix
/// promotion task. Per session CONTRACT §2.13, `finalize` sends
/// `CommitsClosed` + `Drained` if not already sent — which would
/// arrive on the prefill peer BEFORE the promoted prefix blocks
/// are exposed via `session.make_available`, stranding prefill
/// with an unfillable expected_hashes window.
///
/// The release path must defer `session.finalize` until the
/// promotion task has finished (success or failure). This test
/// pins that ordering.
///
/// Scenario:
///   1. GNMT spawns the promotion task (NOT yet resolved).
///   2. `request_finished` fires — release_request +
///      coordinator.release run.
///   3. Assert `session.finished_reason()` is `None` (finalize
///      deferred — task is still in flight).
///   4. Resolve the promotion → task drives make_available +
///      finish_availability + then triggers the deferred
///      finalize.
///   5. Assert `session.finished_reason()` is now `Some` AND
///      make_available_calls includes the promoted prefix.
///
/// Reproducer-first: MUST FAIL on the pre-fix codebase where
/// coordinator.release calls session.finalize unconditionally.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn cd_decode_finalize_deferred_until_promotion_lands() -> Result<()> {
    let h = build_harness();

    h.wrapper.create_slot(make_request())?;
    let _ = h
        .wrapper
        .get_num_new_matched_tokens("req-1", COMPUTED_BLOCKS * BLOCK_SIZE)?;

    let session = h.factory.last_opened().expect("decode opened a session");

    h.wrapper.update_state_after_alloc(
        "req-1",
        h.g1_block_ids.clone(),
        (LOCAL_BLOCKS + REMOTE_BLOCKS) * BLOCK_SIZE,
    )?;

    h.inner.wait_promotion_count(1).await;

    // Fire request_finished BEFORE resolving the promotion.
    // Under the broken (pre-fix) flow this triggers
    // coordinator.release → session.finalize, which closes
    // the availability stream before the promotion's
    // make_available lands.
    let _ = h.wrapper.request_finished("req-1");

    // Give the runtime a moment to settle the release chain.
    // session.finalize would have been called synchronously
    // inside coordinator.release; if the fix is in place it
    // must be deferred.
    tokio::time::sleep(Duration::from_millis(50)).await;
    assert!(
        session.finished_reason().is_none(),
        "session.finalize must NOT fire while promotion task is in flight \
         (got finished_reason={:?})",
        session.finished_reason()
    );
    assert!(
        !session.finish_availability_called(),
        "finish_availability must NOT fire while promotion task is in flight",
    );

    // Now resolve the promotion. The task should drive
    // make_available + finish_availability, then the deferred
    // finalize fires.
    let slot = h.inner.slot("req-1").expect("slot installed");
    let prefix_token_blocks: Vec<_> = slot.token_blocks[..COMPUTED_BLOCKS].to_vec();
    let mutables = h
        .inner
        .g2_manager()
        .allocate_blocks(COMPUTED_BLOCKS)
        .expect("allocate prefix G2");
    let completes: Vec<_> = mutables
        .into_iter()
        .zip(prefix_token_blocks.iter())
        .map(|(m, tb)| m.complete(tb).expect("complete prefix G2"))
        .collect();
    let promoted_g2 = h.inner.g2_manager().register_blocks(completes);
    h.inner.resolve_promotion(0, Ok(promoted_g2));

    // After the task lands the promotion AND the deferred
    // finalize fires, both observables must hold.
    wait_until(|| session.finished_reason().is_some()).await;

    let make_avail = session.make_available_calls();
    assert!(
        make_avail.iter().any(|v| v == &prefix_hashes(&h)),
        "promoted prefix must be made-available BEFORE the deferred finalize fires; \
         observed make_available_calls = {make_avail:?}"
    );
    assert!(
        session.finish_availability_called(),
        "finish_availability must be called by the promotion task before deferred finalize",
    );

    Ok(())
}

/// Stage 1 corner: when the promotion task FAILS, it calls
/// `session.close(reason)` — which per CONTRACT §2.14 implies
/// `finish_commits + finish_availability` and sends a Detached
/// lifecycle event. If `request_finished` then fires (or already
/// fired and the deferred release awaits the failed task), the
/// deferred-release path MUST NOT also call `session.finalize` —
/// `finalize` on an already-closed session sends `Frame::Finished`
/// after the close terminators (undefined protocol behavior;
/// MockSession surfaces it as `finished_reason=Some` on top of
/// `closed_reason=Some`, both observable on the same session).
///
/// The promotion task must signal to the deferred-release path
/// whether it closed the session, so the deferred-release path can
/// skip the redundant finalize.
///
/// Scenario:
///   1. GNMT spawns the promotion task (not yet resolved).
///   2. `request_finished` fires — deferred release spawns,
///      awaits the task.
///   3. Resolve the promotion with `Err` — task calls
///      `session.close(reason)`. closed_reason becomes Some.
///   4. Deferred release wakes up after task completion.
///   5. Assert: `session.finished_reason()` stays None (no
///      finalize-after-close).
///
/// Reproducer-first: MUST FAIL on the current code where the
/// deferred release calls finalize unconditionally.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn cd_decode_deferred_release_skips_finalize_after_promotion_failure() -> Result<()> {
    let h = build_harness();

    h.wrapper.create_slot(make_request())?;
    let _ = h
        .wrapper
        .get_num_new_matched_tokens("req-1", COMPUTED_BLOCKS * BLOCK_SIZE)?;

    let session = h.factory.last_opened().expect("decode opened a session");

    h.wrapper.update_state_after_alloc(
        "req-1",
        h.g1_block_ids.clone(),
        (LOCAL_BLOCKS + REMOTE_BLOCKS) * BLOCK_SIZE,
    )?;

    h.inner.wait_promotion_count(1).await;

    // Fire request_finished BEFORE resolving promotion — the
    // deferred-release path spawns and awaits the task.
    let _ = h.wrapper.request_finished("req-1");
    tokio::time::sleep(Duration::from_millis(50)).await;

    // Resolve promotion with an error — the task calls
    // session.close(reason), draining both streams.
    h.inner
        .resolve_promotion(0, Err(anyhow::anyhow!("simulated promotion failure")));

    // Wait for close to land.
    wait_until(|| session.closed_reason().is_some()).await;
    // Give the deferred release a moment to (incorrectly) call
    // finalize on top of the close. If the fix is in place the
    // deferred release observes the task's "closed the session"
    // signal and skips finalize.
    tokio::time::sleep(Duration::from_millis(50)).await;

    assert!(
        session.closed_reason().is_some(),
        "promotion failure must call session.close"
    );
    assert!(
        session.finished_reason().is_none(),
        "deferred release must NOT call session.finalize after the promotion task \
         already closed the session — got finished_reason={:?} alongside \
         closed_reason={:?}",
        session.finished_reason(),
        session.closed_reason(),
    );

    Ok(())
}

#[allow(dead_code)]
fn _ensure_inner_used(h: &TestHarness) {
    let _ = h.inner.local_id();
    let _ = h.coordinator.active_count();
}

/// Stage 2c: when decode's G3 holds the prefix range that vLLM
/// reports (`num_computed_tokens > 0`) AND the G2 cache has no
/// record of it, the wrapper must plan a G3→G2 promotion (not a
/// G1→G2 one). Producer wiring lives in `commit_gnmt_remote`'s
/// empty-prefix arm: try `find_prefix_g3_hashes` first, fall
/// back to G1 promotion only if G3 misses.
///
/// What this test pins:
///   1. `find_prefix_g3_hashes` is called and returns the
///      prefix slice (the mock is pre-loaded with
///      `install_g3_prefix`).
///   2. The recorded `PendingTierPromotion.source_tier` is `G3`
///      (visible via the `pending_promotion_tier_for_test`
///      accessor).
///   3. The session-side commit shape is the same as the G1
///      path — `session.commit(prefix ∪ local_match)` +
///      `finish_commits` + deferred `finish_availability`.
///      Stage 2's promotion source change does not alter the
///      session-side committed set or sealing.
///   4. No `promote_g1_to_g2` call fires at this point (G1 path
///      not taken) and no `promote_g3_to_g2` call fires yet
///      (Stage 2d wires the USAA-time dispatch).
///
/// Discriminating quality: pre-Stage-2c code does not call
/// `find_prefix_g3_hashes`, so even with `install_g3_prefix`
/// the planned tier would be `G1` (the only branch in the
/// arm). The `Some("G3")` assertion fails on pre-2c code and
/// passes only when the empty-prefix arm consults G3 first.
///
/// This test does NOT drive USAA — Stage 2a's guard in
/// `commit_usaa1` bails on G3 dispatch until Stage 2d wires it.
/// The Stage 2d test will exercise the USAA path.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn cd_decode_plans_g3_to_g2_promotion_when_g3_holds_prefix() -> Result<()> {
    let h = build_harness();

    // Pre-load the mock's G3 universe with the prefix hashes so
    // `find_prefix_g3_hashes` returns Some on the GNMT-time query.
    h.inner.install_g3_prefix("req-1", prefix_hashes(&h));

    h.wrapper.create_slot(make_request())?;

    // GNMT — wrapper opens session, plans G3 promotion in the
    // empty-prefix arm.
    let (count, async_flag) = h
        .wrapper
        .get_num_new_matched_tokens("req-1", COMPUTED_BLOCKS * BLOCK_SIZE)?;
    assert_eq!(count, Some((LOCAL_BLOCKS + REMOTE_BLOCKS) * BLOCK_SIZE));
    assert!(async_flag);

    // Pin: the planned promotion's source tier is G3 (not G1).
    assert_eq!(
        h.wrapper.pending_promotion_tier_for_test("req-1"),
        Some("G3"),
        "with G3 pre-loaded, commit_gnmt_remote must plan a G3 promotion",
    );

    // Pin: session-side shape matches the G1 path. The committed
    // set is `prefix ∪ local_match`; `finish_commits` is sealed;
    // `finish_availability` is deferred until the promotion task
    // (lands in Stage 2d) drives `session.make_available` for the
    // promoted blocks.
    let session = h.factory.last_opened().expect("decode opened a session");
    let mut expected_first_commit = prefix_hashes(&h);
    expected_first_commit.extend(local_match_hashes(&h));
    assert_eq!(
        session.commit_calls(),
        vec![expected_first_commit],
        "GNMT must commit prefix ∪ local_match regardless of source tier",
    );
    assert_eq!(
        session.make_available_calls(),
        vec![local_match_hashes(&h)],
        "GNMT must NOT expose planned-promotion blocks before they land",
    );
    assert!(
        session.finish_commits_called(),
        "GNMT must seal commits at the GNMT-time planned set",
    );
    assert!(
        !session.finish_availability_called(),
        "GNMT must DEFER finish_availability — Stage 2d wires the G3 task",
    );

    // Pin: no G1 promotion fired at GNMT; no G3 promotion yet
    // either (USAA-time dispatch lands in Stage 2d).
    assert_eq!(
        h.inner.promotion_count(),
        0,
        "G1 promotion must NOT fire when G3 path wins the finder race",
    );
    assert_eq!(
        h.inner.g3_promotion_count(),
        0,
        "G3 promotion dispatch lands in Stage 2d, not here",
    );

    Ok(())
}

/// Stage 2c: when neither G2 nor G3 holds the prefix, the
/// wrapper must fall back to the existing Stage 1 G1 promotion
/// path. This is the existing happy-path behavior; the test
/// pins it explicitly so a future regression in the G3
/// short-circuit doesn't silently route G1-only flows through
/// the wrong branch.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn cd_decode_falls_back_to_g1_promotion_when_g3_misses() -> Result<()> {
    let h = build_harness();

    // NOTE: deliberately NOT calling `install_g3_prefix` — the
    // mock's G3 universe is empty, so `find_prefix_g3_hashes`
    // returns empty and `commit_gnmt_remote` falls back to G1.

    h.wrapper.create_slot(make_request())?;
    let _ = h
        .wrapper
        .get_num_new_matched_tokens("req-1", COMPUTED_BLOCKS * BLOCK_SIZE)?;

    assert_eq!(
        h.wrapper.pending_promotion_tier_for_test("req-1"),
        Some("G1"),
        "G3 miss must fall back to G1 promotion (existing Stage 1 path)",
    );

    Ok(())
}

/// Install a second slot ("req-2") on an existing harness. Mirrors
/// `build_harness`'s req-1 setup but with a different token seed so
/// the two slots have distinct hash chains (and therefore distinct
/// `Arc<CdRequestState>` / `Arc<CdRequest>` identities after GNMT).
fn install_second_slot(h: &TestHarness) -> Vec<kvbm_logical::SequenceHash> {
    let g2_manager = h.inner.g2_manager().clone();

    let token_sequence = create_token_sequence(TOTAL_BLOCKS, BLOCK_SIZE, 200);
    let all_hashes = generate_sequence_hashes(&token_sequence);
    let token_blocks: Vec<_> = token_sequence.blocks().to_vec();

    let mutables = g2_manager
        .allocate_blocks(LOCAL_BLOCKS)
        .expect("allocate req-2 local match");
    let completes: Vec<_> = mutables
        .into_iter()
        .zip(token_blocks[COMPUTED_BLOCKS..COMPUTED_BLOCKS + LOCAL_BLOCKS].iter())
        .map(|(mutable, tb)| mutable.complete(tb).expect("complete req-2 local match"))
        .collect();
    let local_match_g2 = g2_manager.register_blocks(completes);

    let slot = MockSlot {
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
        transfer_params: parking_lot::Mutex::new(None),
        ..MockSlot::default()
    };
    h.inner.install_slot("req-2", slot);
    all_hashes
}

fn make_request_for(request_id: &str) -> Request {
    Request::builder()
        .request_id(request_id.to_string())
        .tokens(dynamo_tokens::Tokens::from(Vec::<u32>::new()))
        .build(None)
        .expect("build request")
}

/// Stage 1 hardening: `release_request_if_matches` /
/// `release_if_matches` MUST refuse to remove a DashMap entry whose
/// `Arc` identity differs from the caller's captured snapshot. The
/// production race this guards against is the cross-lifecycle stale
/// release under `kv_load_failure_policy=recompute` — a spawn-replay
/// task from a prior lifecycle parked in `mark_failed_onboarding`
/// for unbounded time MUST NOT wipe the freshly-installed state of
/// a subsequent lifecycle for the same rid.
///
/// This test reproduces the structural failure mode using two
/// distinct rids ("req-1", "req-2"): cycle-1 captures `handle_a`,
/// cycle-2 captures `handle_b`, and the cross-handle release calls
/// must no-op. The Arc-identity predicate inside
/// `DashMap::remove_if` is the same code path the cross-lifecycle
/// case exercises — Arc-A captured, Arc-B currently at the key,
/// `Arc::ptr_eq` is false, remove_if returns None.
///
/// Pre-fix code did not expose `release_*_if_matches` at all; the
/// spawn-replays called `release_request` / `coordinator.release`
/// by name, which would wipe whatever state currently lived at the
/// rid regardless of identity. This test asserts both the
/// negative (stale handle no-ops) and the positive (matching
/// handle releases) for both wrapper and coordinator.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn release_if_matches_enforces_arc_identity() -> Result<()> {
    let h = build_harness();
    install_second_slot(&h);

    // Cycle-A: GNMT for req-1 installs state A in both maps.
    h.wrapper.create_slot(make_request_for("req-1"))?;
    let _ = h
        .wrapper
        .get_num_new_matched_tokens("req-1", COMPUTED_BLOCKS * BLOCK_SIZE)?;
    let handle_a = h
        .wrapper
        .snapshot_cd_state_for_test("req-1")
        .expect("wrapper state A");
    let coord_a = h
        .coordinator
        .state_for_test("req-1")
        .expect("coord state A");

    // Cycle-B: GNMT for req-2 installs state B in both maps.
    h.wrapper.create_slot(make_request_for("req-2"))?;
    let _ = h
        .wrapper
        .get_num_new_matched_tokens("req-2", COMPUTED_BLOCKS * BLOCK_SIZE)?;
    let handle_b = h
        .wrapper
        .snapshot_cd_state_for_test("req-2")
        .expect("wrapper state B");
    let coord_b = h
        .coordinator
        .state_for_test("req-2")
        .expect("coord state B");

    // Sanity: two distinct lifecycles must hold distinct Arcs.
    assert!(
        !handle_a.ptr_eq(&handle_b),
        "distinct lifecycles must yield distinct wrapper Arcs"
    );
    assert!(
        !Arc::ptr_eq(&coord_a, &coord_b),
        "distinct lifecycles must yield distinct coordinator Arcs"
    );

    // Stale-Arc release attempts MUST no-op. Each call looks up the
    // current entry at the target rid, observes that
    // `Arc::ptr_eq(captured, current)` is false, and returns false
    // without touching state.
    assert!(
        !h.wrapper
            .release_request_if_matches_for_test("req-1", &handle_b),
        "wrapper release with B's handle against req-1 must no-op"
    );
    assert!(
        h.wrapper.has_active_cd_request("req-1"),
        "req-1 wrapper state must survive stale release"
    );
    assert!(
        !h.coordinator.release_if_matches("req-1", &coord_b),
        "coordinator release with B's Arc against req-1 must no-op"
    );
    assert!(
        h.coordinator.state_for_test("req-1").is_some(),
        "req-1 coord state must survive stale release"
    );

    // Matching-Arc release MUST fire and remove the entry.
    assert!(
        h.wrapper
            .release_request_if_matches_for_test("req-1", &handle_a),
        "wrapper release with matching A handle must succeed"
    );
    assert!(
        !h.wrapper.has_active_cd_request("req-1"),
        "req-1 wrapper state must be removed after matching release"
    );
    assert!(
        h.coordinator.release_if_matches("req-1", &coord_a),
        "coordinator release with matching A Arc must succeed"
    );
    assert!(
        h.coordinator.state_for_test("req-1").is_none(),
        "req-1 coord state must be removed after matching release"
    );

    // Re-attempt with the same (now-stale) handles MUST no-op —
    // the entry is gone, `DashMap::remove_if` returns None without
    // running the predicate.
    assert!(
        !h.wrapper
            .release_request_if_matches_for_test("req-1", &handle_a),
        "wrapper release against missing entry must no-op"
    );
    assert!(
        !h.coordinator.release_if_matches("req-1", &coord_a),
        "coordinator release against missing entry must no-op"
    );

    // req-2 still alive; nothing has touched it.
    assert!(h.wrapper.has_active_cd_request("req-2"));
    assert!(h.coordinator.state_for_test("req-2").is_some());

    Ok(())
}
