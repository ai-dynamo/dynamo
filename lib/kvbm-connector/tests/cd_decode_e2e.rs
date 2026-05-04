// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Decode-side end-to-end test for the conditional-disagg wrapper, against
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
    InMemoryRemotePrefillQueue, MockCdBlockTransport, MockCdWorkerHook, MockInnerLeaderShim,
    MockSlot, TEST_BLOCK_SIZE, wait_until,
};
use kvbm_connector::connector::leader::disagg::{
    AlwaysRemote, ConnectorLeaderApi, ConditionalDisaggCoordinator, DecodeDisaggLeader,
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

struct TestHarness {
    wrapper: Arc<DecodeDisaggLeader>,
    inner: Arc<MockInnerLeaderShim>,
    transport: Arc<MockCdBlockTransport>,
    workers: Arc<MockCdWorkerHook>,
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
        transfer_params: None,
        ..MockSlot::default()
    };
    inner.install_slot("req-1", slot);

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
    // sequence_hashes carries the local-match (what prefill will pull
    // from us), not the remote slice — symmetric-API wire semantics.
    assert_eq!(queued[0].sequence_hashes, local_match_hashes(&h));
    // num_computed_tokens carries the decode-side gnmt argument
    // (decode-already-computed prefix), so prefill knows where the
    // sequence_hashes blocks live in absolute position.
    assert_eq!(queued[0].num_computed_tokens, COMPUTED_BLOCKS * BLOCK_SIZE);
    // token_ids carries only the prefill-window slice — decode keeps
    // its already-computed prefix and the partial tail block.
    assert_eq!(
        queued[0].token_ids.len(),
        (LOCAL_BLOCKS + REMOTE_BLOCKS) * BLOCK_SIZE
    );

    let session = h.factory.last_opened().expect("decode opened a session");
    assert_eq!(session.commit_calls(), vec![local_match_hashes(&h)]);
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
    // Decode now signals cooperative finalize, NOT close().
    // close() is reserved for the abort path; happy-path
    // shutdown goes through finalize() and waits for the
    // peer's symmetric finalize to fire the rendezvous.
    assert!(
        session.finished_reason().is_some(),
        "decode coordinator must signal session.finalize() on cooperative end"
    );
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

#[allow(dead_code)]
fn _ensure_inner_used(h: &TestHarness) {
    let _ = h.inner.local_id();
    let _ = h.coordinator.active_count();
}
