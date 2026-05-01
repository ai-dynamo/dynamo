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
    AlwaysRemote, ConnectorLeaderApi, DecodeDisaggLeader, RemotePrefillCoordinator,
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
    coordinator: Arc<RemotePrefillCoordinator>,
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
    let coordinator = RemotePrefillCoordinator::new(
        Arc::new(AlwaysRemote),
        factory.clone(),
        queue.clone(),
        tokio::runtime::Handle::current(),
    );

    let transport = MockCdBlockTransport::new();
    let workers = MockCdWorkerHook::new();

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

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
#[should_panic(expected = "availability carried hash")]
async fn cd_decode_unexpected_hash_panics() {
    let h = build_harness();

    h.wrapper.create_slot(make_request()).unwrap();
    let _ = h
        .wrapper
        .get_num_new_matched_tokens("req-1", COMPUTED_BLOCKS * BLOCK_SIZE)
        .unwrap();
    let session = h.factory.last_opened().expect("session");

    h.wrapper
        .update_state_after_alloc(
            "req-1",
            h.g1_block_ids.clone(),
            (LOCAL_BLOCKS + REMOTE_BLOCKS) * BLOCK_SIZE,
        )
        .unwrap();
    h.transport.wait_onboard_count(1).await;
    h.transport.resolve_onboard(0, Ok(()));

    // Inject availability with a hash from the LOCAL-match range
    // (not in expected_remote_hashes) — coordinator panics in
    // the spawned task.
    let bogus = h.all_hashes[COMPUTED_BLOCKS];
    session.inject_peer_commit(vec![bogus]);
    session.inject_peer_available(vec![CommittedBlock {
        hash: bogus,
        peer_block_id: 9999,
    }]);

    // The panic happens on a tokio worker; #[should_panic] only
    // catches panics on the test thread, so we drive a sleep loop
    // and then panic loud with a matching message if the
    // background panic didn't kill the test.
    for _ in 0..200 {
        tokio::time::sleep(Duration::from_millis(5)).await;
    }
    panic!("availability carried hash should have panicked the spawned task");
}

#[allow(dead_code)]
fn _ensure_inner_used(h: &TestHarness) {
    let _ = h.inner.local_id();
    let _ = h.coordinator.active_count();
}
