// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Decode-side end-to-end test for the conditional-disagg wrapper, using
//! mocks for the inner leader, the block transport, and the worker hook.
//! No `KvbmRuntime`, no `nixl_agent`, no real RDMA.
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
//! - COMPUTED = 2 blocks (32 tokens)
//! - LOCAL    = 2 blocks
//! - REMOTE   = 4 blocks
//!
//! `wrapper.get_num_new_matched_tokens` returns `(Some((LOCAL+REMOTE) * block_size), true)`,
//! commits `begin_remote_prefill`, and proceeds through USAA-1 → BlockSetsAdded
//! → maybe_complete using only the mocks.

use std::sync::Arc;
use std::time::Duration;

use anyhow::Result;
use kvbm_connector::common::Request;
use kvbm_connector::connector::leader::disagg::testing::{
    InMemoryRemotePrefillQueue, MockCdBlockTransport, MockCdWorkerHook,
    MockInnerLeaderShim, MockPrefillSessionFactory, MockSlot, TEST_BLOCK_SIZE, wait_until,
};
use kvbm_connector::connector::leader::disagg::{
    AlwaysRemote, DecodeDisaggLeader, ConnectorLeaderApi, RemotePrefillCoordinator,
    SessionEvent,
};
use kvbm_config::{DisaggConfig, DisaggregationRole};
use kvbm_engine::disagg::{RemoteBlockRef, RemoteBlockSet};
use kvbm_engine::testing::token_blocks::{create_token_sequence, generate_sequence_hashes};
use kvbm_engine::testing::managers::{TestManagerBuilder, TestRegistryBuilder};
use kvbm_connector::G2;
use kvbm_logical::manager::BlockManager;

fn make_request() -> Request {
    Request::builder()
        .request_id("req-1".to_string())
        .tokens(dynamo_tokens::Tokens::from(Vec::<u32>::new()))
        .build(None)
        .expect("build request")
}

const COMPUTED_BLOCKS: usize = 2;
const LOCAL_BLOCKS: usize = 2;
const REMOTE_BLOCKS: usize = 4;
const TOTAL_BLOCKS: usize = COMPUTED_BLOCKS + LOCAL_BLOCKS + REMOTE_BLOCKS;
const BLOCK_SIZE: usize = TEST_BLOCK_SIZE;

struct TestHarness {
    wrapper: Arc<DecodeDisaggLeader>,
    inner: Arc<MockInnerLeaderShim>,
    transport: Arc<MockCdBlockTransport>,
    workers: Arc<MockCdWorkerHook>,
    factory: Arc<MockPrefillSessionFactory>,
    queue: Arc<InMemoryRemotePrefillQueue>,
    coordinator: Arc<RemotePrefillCoordinator>,
    /// Token-block hashes for the full sequence (length TOTAL_BLOCKS).
    all_hashes: Vec<kvbm_logical::SequenceHash>,
    /// vLLM-allocated G1 block_ids (length TOTAL_BLOCKS).
    g1_block_ids: Vec<usize>,
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

fn build_harness() -> TestHarness {
    // Need slack for both local-match (2) + remote-dest (4) + a small margin.
    let g2_manager = build_g2_manager(32);

    // Build a real token sequence so the hashes the wrapper sees match the
    // ones we'll use in the BlockSetsAdded event.
    let token_sequence = create_token_sequence(TOTAL_BLOCKS, BLOCK_SIZE, 100);
    let all_hashes = generate_sequence_hashes(&token_sequence);
    let token_blocks: Vec<_> = token_sequence.blocks().to_vec();
    assert_eq!(all_hashes.len(), TOTAL_BLOCKS);
    assert_eq!(token_blocks.len(), TOTAL_BLOCKS);

    // Pre-allocate + register the LOCAL-match G2 blocks so they look like
    // real find-session output.
    let mutables = g2_manager
        .allocate_blocks(LOCAL_BLOCKS)
        .expect("allocate local-match G2");
    let completes: Vec<_> = mutables
        .into_iter()
        .zip(token_blocks[COMPUTED_BLOCKS..COMPUTED_BLOCKS + LOCAL_BLOCKS].iter())
        .map(|(mutable, tb)| mutable.complete(tb).expect("complete local match"))
        .collect();
    let local_match_g2 = g2_manager.register_blocks(completes);
    assert_eq!(local_match_g2.len(), LOCAL_BLOCKS);

    // Mock inner leader.
    let inner = MockInnerLeaderShim::new(BLOCK_SIZE, g2_manager.clone());

    // Synthetic G1 block_ids vLLM would give us.
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
        // Inner GNMT reports the local-match count. The wrapper will then
        // promote it to the full external (local + remote) count.
        gnmt_result: (Some(LOCAL_BLOCKS * BLOCK_SIZE), true),
        usaa_passthrough_calls: parking_lot::Mutex::new(Vec::new()),
        transfer_params: None,
    };
    inner.install_slot("req-1", slot);

    // Coordinator + session factory + queue.
    let factory = MockPrefillSessionFactory::new();
    let queue = InMemoryRemotePrefillQueue::new();
    let coordinator = RemotePrefillCoordinator::with_attach_timeout(
        Arc::new(AlwaysRemote),
        factory.clone(),
        queue.clone(),
        tokio::runtime::Handle::current(),
        Duration::from_secs(30),
    );

    // Mocks for transport and worker notifications.
    let transport = MockCdBlockTransport::new();
    let workers = MockCdWorkerHook::new();

    // Wrapper.
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

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn cd_decode_happy_path() -> Result<()> {
    let h = build_harness();

    // Stub create_slot — our mock ignores it, but the trait routes through.
    h.wrapper.create_slot(make_request())?;

    // 1. GNMT — wrapper promotes inner's local match to the full external
    //    count and fires begin_remote_prefill.
    let (count, async_flag) = h
        .wrapper
        .get_num_new_matched_tokens("req-1", COMPUTED_BLOCKS * BLOCK_SIZE)?;
    assert_eq!(
        count,
        Some((LOCAL_BLOCKS + REMOTE_BLOCKS) * BLOCK_SIZE),
        "GNMT should report full-block external for local + remote"
    );
    assert!(async_flag);

    // Coordinator received begin_remote_prefill: queue.enqueue runs on a
    // spawned task so we wait for it to land.
    let queue = h.queue.clone();
    wait_until(|| queue.snapshot().len() == 1).await;
    let queued = h.queue.snapshot();
    assert_eq!(queued[0].request_id, "req-1");
    let session = h.factory.last().expect("session created");

    // 2. Push Attached so on_block_sets_added has a peer_instance_id.
    let peer_id: kvbm_connector::InstanceId = uuid::Uuid::new_v4().into();
    session
        .push_event(SessionEvent::Attached {
            peer_instance_id: peer_id,
        })
        .expect("push attached");

    // 3. USAA-1.
    h.wrapper.update_state_after_alloc(
        "req-1",
        h.g1_block_ids.clone(),
        (LOCAL_BLOCKS + REMOTE_BLOCKS) * BLOCK_SIZE,
    )?;

    // Local kick should be the first onboard call.
    h.transport.wait_onboard_count(1).await;
    let local_call = h.transport.onboard_calls()[0].clone();
    assert_eq!(local_call.src_g2_block_ids.len(), LOCAL_BLOCKS);
    assert_eq!(
        local_call.dst_g1_block_ids,
        h.g1_block_ids[COMPUTED_BLOCKS..COMPUTED_BLOCKS + LOCAL_BLOCKS].to_vec()
    );
    h.transport.resolve_onboard(0, Ok(()));

    // 4. Push BlockSetsAdded for the remote slice — uses the same hashes
    //    the wrapper computed via slot_match_split.
    let remote_hashes = h.all_hashes[COMPUTED_BLOCKS + LOCAL_BLOCKS..].to_vec();
    let block_set = RemoteBlockSet {
        source_layout: kvbm_common::LogicalLayoutHandle::G2,
        blocks: remote_hashes
            .iter()
            .enumerate()
            .map(|(i, h)| RemoteBlockRef {
                block_id: 9000 + i,
                sequence_hash: *h,
            })
            .collect(),
    };
    session
        .push_event(SessionEvent::BlockSetsAdded {
            block_sets: vec![block_set],
        })
        .expect("push block sets added");

    // 5. Pull call.
    h.transport.wait_pull_count(1).await;
    let pull_call = h.transport.pull_calls()[0].clone();
    assert_eq!(pull_call.remote_instance, peer_id);
    assert_eq!(pull_call.local_dst_g2_block_ids.len(), REMOTE_BLOCKS);
    h.transport.resolve_pull(0, Ok(()));

    // 6. Second onboard call — remote G2 → G1 for the [X, N) range.
    h.transport.wait_onboard_count(2).await;
    let remote_call = h.transport.onboard_calls()[1].clone();
    assert_eq!(remote_call.src_g2_block_ids.len(), REMOTE_BLOCKS);
    assert_eq!(
        remote_call.dst_g1_block_ids,
        h.g1_block_ids[COMPUTED_BLOCKS + LOCAL_BLOCKS..].to_vec()
    );
    h.transport.resolve_onboard(1, Ok(()));

    // 7. Wait for completion.
    wait_until(|| h.workers.completed_contains("req-1")).await;

    // PullComplete fired on the session.
    assert_eq!(session.pull_completes().len(), 1);
    assert_eq!(session.pull_completes()[0].hashes, remote_hashes);

    // Budget refunded; CD state cleared.
    assert_eq!(h.wrapper.inflight_available(), usize::MAX);
    assert!(!h.wrapper.has_active_cd_request("req-1"));

    Ok(())
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn cd_decode_local_kick_failure() -> Result<()> {
    let h = build_harness();

    h.wrapper.create_slot(make_request())?;
    let _ = h
        .wrapper
        .get_num_new_matched_tokens("req-1", COMPUTED_BLOCKS * BLOCK_SIZE)?;
    let session = h.factory.last().expect("session created");
    session
        .push_event(SessionEvent::Attached {
            peer_instance_id: uuid::Uuid::new_v4().into(),
        })
        .expect("push attached");

    h.wrapper.update_state_after_alloc(
        "req-1",
        h.g1_block_ids.clone(),
        (LOCAL_BLOCKS + REMOTE_BLOCKS) * BLOCK_SIZE,
    )?;

    h.transport.wait_onboard_count(1).await;
    h.transport
        .resolve_onboard(0, Err(anyhow::anyhow!("simulated local-kick failure")));

    // mark_failed_onboarding fires with the unfilled G1 ids (everything
    // from COMPUTED to N).
    wait_until(|| h.workers.failed_for("req-1").is_some()).await;
    let failed = h.workers.failed_for("req-1").unwrap();
    let expected: Vec<_> = h.g1_block_ids[COMPUTED_BLOCKS..].to_vec();
    let mut got = failed.block_ids.clone();
    got.sort();
    let mut want = expected.clone();
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
    let session = h.factory.last().expect("session created");
    let peer_id: kvbm_connector::InstanceId = uuid::Uuid::new_v4().into();
    session
        .push_event(SessionEvent::Attached {
            peer_instance_id: peer_id,
        })
        .expect("push attached");

    h.wrapper.update_state_after_alloc(
        "req-1",
        h.g1_block_ids.clone(),
        (LOCAL_BLOCKS + REMOTE_BLOCKS) * BLOCK_SIZE,
    )?;

    // Resolve local kick OK.
    h.transport.wait_onboard_count(1).await;
    h.transport.resolve_onboard(0, Ok(()));

    // Push the remote chunk and fail the pull.
    let remote_hashes = h.all_hashes[COMPUTED_BLOCKS + LOCAL_BLOCKS..].to_vec();
    session
        .push_event(SessionEvent::BlockSetsAdded {
            block_sets: vec![RemoteBlockSet {
                source_layout: kvbm_common::LogicalLayoutHandle::G2,
                blocks: remote_hashes
                    .iter()
                    .enumerate()
                    .map(|(i, h)| RemoteBlockRef {
                        block_id: 9000 + i,
                        sequence_hash: *h,
                    })
                    .collect(),
            }],
        })
        .expect("push block sets added");
    h.transport.wait_pull_count(1).await;
    h.transport
        .resolve_pull(0, Err(anyhow::anyhow!("simulated pull failure")));

    wait_until(|| h.workers.failed_for("req-1").is_some()).await;
    let failed = h.workers.failed_for("req-1").unwrap();
    let expected: Vec<_> = h.g1_block_ids[COMPUTED_BLOCKS + LOCAL_BLOCKS..].to_vec();
    let mut got = failed.block_ids.clone();
    got.sort();
    let mut want = expected.clone();
    want.sort();
    assert_eq!(got, want, "only the remote slice should be unfilled");
    assert_eq!(h.wrapper.inflight_available(), usize::MAX);

    Ok(())
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
#[should_panic(expected = "BlockSetsAdded carried hash")]
async fn cd_decode_unexpected_hash_panics() {
    let h = build_harness();

    h.wrapper.create_slot(make_request()).unwrap();
    let _ = h
        .wrapper
        .get_num_new_matched_tokens("req-1", COMPUTED_BLOCKS * BLOCK_SIZE)
        .unwrap();
    let session = h.factory.last().expect("session created");
    session
        .push_event(SessionEvent::Attached {
            peer_instance_id: uuid::Uuid::new_v4().into(),
        })
        .expect("push attached");

    h.wrapper
        .update_state_after_alloc(
            "req-1",
            h.g1_block_ids.clone(),
            (LOCAL_BLOCKS + REMOTE_BLOCKS) * BLOCK_SIZE,
        )
        .unwrap();
    h.transport.wait_onboard_count(1).await;
    h.transport.resolve_onboard(0, Ok(()));

    // Push a BlockSetsAdded carrying a hash we don't expect (using the
    // hash for COMPUTED_BLOCKS, which is in the local-match range).
    let bogus_hash = h.all_hashes[COMPUTED_BLOCKS];
    session
        .push_event(SessionEvent::BlockSetsAdded {
            block_sets: vec![RemoteBlockSet {
                source_layout: kvbm_common::LogicalLayoutHandle::G2,
                blocks: vec![RemoteBlockRef {
                    block_id: 9999,
                    sequence_hash: bogus_hash,
                }],
            }],
        })
        .expect("push block sets added");

    // Wait for the panic to propagate via the spawned task — we need to
    // give it cycles. The #[should_panic] attribute only catches panics on
    // the test thread, so we drive a loop that yields and re-checks.
    for _ in 0..200 {
        tokio::time::sleep(Duration::from_millis(5)).await;
    }
    // If we reach here without the panic having been observed, fail the
    // test loud.
    panic!("BlockSetsAdded carried hash should have panicked the spawned task");
}

// Suppress unused-warnings until the corresponding fields/methods are
// observed in additional tests.
#[allow(dead_code)]
fn _ensure_inner_used(h: &TestHarness) {
    let _ = h.inner.local_id();
    let _ = h.coordinator.active_count();
}
