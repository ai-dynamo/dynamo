// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Prefill-side end-to-end test for the conditional-disagg wrapper.
//!
//! Mocks for the inner leader, the block transport, the worker hook, and
//! the session attacher. No `KvbmRuntime`, no `nixl_agent`, no real RDMA,
//! no real velo. The wrapper drives `PrefillCoordinatorImpl` through its
//! `ConnectorLeaderApi` surface against scripted collaborators.
//!
//! Sequence layout (block_size = 16, total = 4 blocks):
//!
//! ```text
//! block index   :  0   1   2   3
//! [0, N)        : [r   r   r   r]    (everything pulled from D)
//! ```
//!
//! GNMT-1 always returns `(Some(N * block_size), true)` for CD-bound
//! requests; non-CD requests fall through to the inner.
//!
//! See `/home/ryan/.claude/plans/cd-usaa-pipeline.md` §"Test strategy".

use std::sync::Arc;
use std::time::Duration;

use anyhow::Result;
use kvbm_connector::G2;
use kvbm_connector::common::Request;
use kvbm_connector::connector::leader::disagg::prefill_coordinator::{
    PrefillCoordinatorImpl, PrefillStatus,
};
use kvbm_connector::connector::leader::disagg::testing::{
    MockCdBlockTransport, MockCdWorkerHook, MockInnerLeaderShim, MockPrefillSessionAttacher,
    MockSlot, TEST_BLOCK_SIZE, wait_until,
};
use kvbm_connector::connector::leader::disagg::{
    ConnectorLeaderApi, PrefillDisaggLeader, SessionEvent,
};
use kvbm_disagg_protocol::{
    DISAGG_PROTOCOL_VERSION, RemotePrefillParams, SessionEndpoint, SessionId, TransferParams,
};
use kvbm_engine::disagg::{
    BlockSetResponse, HashSelection, PullComplete, RemoteBlockRef, RemoteBlockSet,
};
use kvbm_engine::testing::managers::{TestManagerBuilder, TestRegistryBuilder};
use kvbm_engine::testing::token_blocks::{create_token_sequence, generate_sequence_hashes};
use kvbm_logical::blocks::ImmutableBlock;
use kvbm_logical::manager::BlockManager;

const TOTAL_BLOCKS: usize = 4;
const BLOCK_SIZE: usize = TEST_BLOCK_SIZE;
const NUM_EXTERNAL: usize = TOTAL_BLOCKS * BLOCK_SIZE;

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

fn synthetic_decode_endpoint() -> SessionEndpoint {
    SessionEndpoint {
        kind: "mock_decode".to_string(),
        payload: serde_json::json!({"decode_endpoint": "test"}),
    }
}

/// Build a `TransferParams` carrying a remote-prefill marker
/// for `expected_hashes`.
fn cd_transfer_params(
    session_id: SessionId,
    initiator_instance_id: kvbm_connector::InstanceId,
    expected_hashes: Vec<kvbm_logical::SequenceHash>,
) -> TransferParams {
    TransferParams::remote_prefill(RemotePrefillParams {
        protocol_version: DISAGG_PROTOCOL_VERSION,
        session_id,
        initiator_instance_id,
        decode_endpoint: Some(synthetic_decode_endpoint()),
        sequence_hashes: expected_hashes,
    })
}

struct TestHarness {
    wrapper: Arc<PrefillDisaggLeader>,
    coordinator: Arc<PrefillCoordinatorImpl>,
    inner: Arc<MockInnerLeaderShim>,
    transport: Arc<MockCdBlockTransport>,
    workers: Arc<MockCdWorkerHook>,
    attacher: Arc<MockPrefillSessionAttacher>,
    /// Token-block hashes for the full sequence (length TOTAL_BLOCKS).
    all_hashes: Vec<kvbm_logical::SequenceHash>,
    /// Synthetic G1 block_ids vLLM would give us.
    g1_block_ids: Vec<usize>,
    /// Synthetic G2 source block_ids D would advertise in
    /// `BlockSetResponse.ready`.
    decode_g2_block_ids: Vec<usize>,
    /// Pre-built G2 source manager for synthesizing output blocks
    /// in `simulate_offload_complete`.
    output_blocks: Vec<ImmutableBlock<G2>>,
    /// `params.initiator_instance_id` — what the coordinator will
    /// pass as `remote_instance` on `pull_remote`. Captured so
    /// tests can assert on it.
    decode_instance_id: kvbm_connector::InstanceId,
}

/// Build the harness with a CD-bound slot. If `with_transfer_params`
/// is false, the slot has no transfer_params and the wrapper falls
/// through to inner.
fn build_harness(with_transfer_params: bool) -> TestHarness {
    // Generous capacity: TOTAL_BLOCKS pulled-G2 + a few output blocks.
    let g2_manager = build_g2_manager(64);

    // Real token sequence so coordinator's `MutableBlock::complete`
    // resolves against valid hashes.
    let token_sequence = create_token_sequence(TOTAL_BLOCKS, BLOCK_SIZE, 100);
    let all_hashes = generate_sequence_hashes(&token_sequence);
    let token_blocks: Vec<_> = token_sequence.blocks().to_vec();
    assert_eq!(all_hashes.len(), TOTAL_BLOCKS);

    // Pre-build a few output blocks to feed simulate_offload_complete.
    // Use a different start_token so hashes don't collide with input.
    let output_seq = create_token_sequence(2, BLOCK_SIZE, 9000);
    let output_token_blocks: Vec<_> = output_seq.blocks().to_vec();
    let output_mutables = g2_manager
        .allocate_blocks(2)
        .expect("allocate output mutables");
    let output_completes: Vec<_> = output_mutables
        .into_iter()
        .zip(output_token_blocks.iter())
        .map(|(m, tb)| m.complete(tb).expect("complete output"))
        .collect();
    let output_blocks = g2_manager.register_blocks(output_completes);
    assert_eq!(output_blocks.len(), 2);

    let inner = MockInnerLeaderShim::new(BLOCK_SIZE, g2_manager.clone());

    let g1_block_ids: Vec<usize> = (1000..1000 + TOTAL_BLOCKS).collect();
    let decode_g2_block_ids: Vec<usize> = (5000..5000 + TOTAL_BLOCKS).collect();

    let session_id = uuid::Uuid::new_v4();
    let decode_instance_id: kvbm_connector::InstanceId = uuid::Uuid::new_v4().into();
    let transfer_params = if with_transfer_params {
        Some(cd_transfer_params(
            session_id,
            decode_instance_id,
            all_hashes.clone(),
        ))
    } else {
        None
    };

    let slot = MockSlot {
        block_size: BLOCK_SIZE,
        total_blocks: TOTAL_BLOCKS,
        // Prefill side doesn't use computed/local-match split, but
        // the struct requires the fields. Set to no-match.
        computed_blocks: 0,
        local_match_blocks: 0,
        all_hashes: all_hashes.clone(),
        token_blocks,
        local_match_g2: parking_lot::Mutex::new(Some(Vec::new())),
        assigned_block_ids: parking_lot::Mutex::new(None),
        // Used only on non-CD passthrough. Exercise it with a
        // distinguishable value.
        gnmt_result: (Some(7 * BLOCK_SIZE), false),
        usaa_passthrough_calls: parking_lot::Mutex::new(Vec::new()),
        transfer_params,
    };
    inner.install_slot("req-1", slot);

    let transport = MockCdBlockTransport::new();
    let workers = MockCdWorkerHook::new();
    let attacher = MockPrefillSessionAttacher::new();

    let coordinator = PrefillCoordinatorImpl::new(
        inner.clone(),
        transport.clone(),
        workers.clone(),
        attacher.clone(),
        tokio::runtime::Handle::current(),
    );

    let wrapper = PrefillDisaggLeader::from_parts(
        inner.clone(),
        coordinator.clone(),
        workers.clone(),
    );

    TestHarness {
        wrapper,
        coordinator,
        inner,
        transport,
        workers,
        attacher,
        all_hashes,
        g1_block_ids,
        decode_g2_block_ids,
        output_blocks,
        decode_instance_id,
    }
}

/// Build a scripted `BlockSetResponse` advertising D's local G2
/// block_ids for every expected hash.
fn scripted_block_set_response(
    request_id: &str,
    decode_g2_block_ids: &[usize],
    expected_hashes: &[kvbm_logical::SequenceHash],
) -> BlockSetResponse {
    BlockSetResponse {
        request_id: request_id.to_string(),
        ready: vec![RemoteBlockSet {
            source_layout: kvbm_common::LogicalLayoutHandle::G2,
            blocks: expected_hashes
                .iter()
                .zip(decode_g2_block_ids.iter())
                .map(|(hash, block_id)| RemoteBlockRef {
                    block_id: *block_id,
                    sequence_hash: *hash,
                })
                .collect(),
        }],
        pending_hashes: Vec::new(),
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn cd_prefill_happy_path() -> Result<()> {
    let h = build_harness(true);

    h.wrapper.create_slot(make_request())?;

    // 1. GNMT — wrapper detects CD-bound transfer_params, calls
    //    ensure_started, returns (Some(N), true).
    let (count, async_flag) = h.wrapper.get_num_new_matched_tokens("req-1", 0)?;
    assert_eq!(count, Some(NUM_EXTERNAL));
    assert!(async_flag);
    assert_eq!(h.coordinator.active_count(), 1);

    // Coordinator spawned attach asynchronously; wait for it to land.
    wait_until(|| h.attacher.last().is_some()).await;
    let session = h.attacher.last().expect("session attached");
    assert_eq!(h.attacher.attach_calls().len(), 1);

    // 2. Coordinator issues `request_block_sets` against the
    //    session. The mock waits for a scripted response, so the
    //    test can deliver the response any time.
    wait_until(|| !session.requested_block_sets().is_empty()).await;
    let requests = session.requested_block_sets();
    assert_eq!(requests.len(), 1);
    assert_eq!(requests[0].hashes, HashSelection::All);
    session.enqueue_block_set_response(scripted_block_set_response(
        "req-1",
        &h.decode_g2_block_ids,
        &h.all_hashes,
    ));

    // 3. Pull D→P fires with peer_instance_id sourced from
    //    transfer_params.initiator_instance_id (NOT from a
    //    SessionEvent::Attached — that's a decode-side event,
    //    not prefill-side).
    h.transport.wait_pull_count(1).await;
    let pull = h.transport.pull_calls()[0].clone();
    assert_eq!(pull.remote_instance, h.decode_instance_id);
    assert_eq!(pull.local_dst_g2_block_ids.len(), TOTAL_BLOCKS);
    h.transport.resolve_pull(0, Ok(()));

    // 5. After register, status flips to Registered. Wait for it.
    wait_until(|| h.coordinator.status_for("req-1") == Some(PrefillStatus::Registered)).await;

    // 6. USAA arrives now (post-register). Wrapper calls inner USAA
    //    first, then on_usaa hands deltas to the coordinator. Since
    //    state is Registered, on_usaa spawns `kick_onboard`.
    h.wrapper
        .update_state_after_alloc("req-1", h.g1_block_ids.clone(), NUM_EXTERNAL)?;

    // Inner USAA passthrough recorded.
    let slot = h.inner.slot("req-1").unwrap();
    {
        let calls = slot.usaa_passthrough_calls.lock();
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].0, h.g1_block_ids);
        assert_eq!(calls[0].1, NUM_EXTERNAL);
    }

    // 7. G2→G1 onboard fires. Resolve OK.
    h.transport.wait_onboard_count(1).await;
    let onboard = h.transport.onboard_calls()[0].clone();
    assert_eq!(onboard.dst_g1_block_ids, h.g1_block_ids);
    assert_eq!(onboard.src_g2_block_ids.len(), TOTAL_BLOCKS);
    h.transport.resolve_onboard(0, Ok(()));

    // 8. mark_onboarding_complete fires.
    wait_until(|| h.workers.completed_contains("req-1")).await;
    wait_until(|| {
        h.coordinator.status_for("req-1") == Some(PrefillStatus::OnboardingComplete)
    })
    .await;

    // 9. Forward-pass-time: simulate a single chunk of G2 outputs
    //    landing via the offload pipeline observer.
    h.coordinator
        .simulate_offload_complete("req-1", h.output_blocks.clone())?;

    wait_until(|| !session.published_output_sets().is_empty()).await;
    let published = session.published_output_sets();
    assert_eq!(published.len(), 1);
    assert_eq!(published[0].len(), 1);
    assert_eq!(published[0][0].blocks.len(), 2);

    // 10. Slot is finished. Coordinator marks SlotDone but session
    //     stays alive holding output pins.
    let _status = h.wrapper.request_finished("req-1");
    wait_until(|| h.coordinator.status_for("req-1") == Some(PrefillStatus::SlotDone)).await;

    // 11. D's PullComplete arrives. Coordinator sends PullAck and
    //     drops the request state.
    let pull_id = 42_u64;
    session
        .push_event(SessionEvent::PullComplete(PullComplete {
            pull_id,
            hashes: h.all_hashes.clone(),
        }))
        .expect("push pull complete");

    wait_until(|| !session.pull_acks().is_empty()).await;
    assert_eq!(session.pull_acks()[0].pull_id, pull_id);

    wait_until(|| h.coordinator.active_count() == 0).await;

    Ok(())
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn cd_prefill_usaa_before_pull_completes() -> Result<()> {
    let h = build_harness(true);

    h.wrapper.create_slot(make_request())?;

    let (count, async_flag) = h.wrapper.get_num_new_matched_tokens("req-1", 0)?;
    assert_eq!(count, Some(NUM_EXTERNAL));
    assert!(async_flag);

    wait_until(|| h.attacher.last().is_some()).await;
    let session = h.attacher.last().expect("session attached");

    wait_until(|| !session.requested_block_sets().is_empty()).await;
    session.enqueue_block_set_response(scripted_block_set_response(
        "req-1",
        &h.decode_g2_block_ids,
        &h.all_hashes,
    ));

    // Wait for pull to be enqueued — but DO NOT resolve it yet.
    h.transport.wait_pull_count(1).await;
    assert!(matches!(
        h.coordinator.status_for("req-1"),
        Some(PrefillStatus::Pulling)
    ));

    // USAA arrives early. Coordinator should stash G1 ids.
    h.wrapper
        .update_state_after_alloc("req-1", h.g1_block_ids.clone(), NUM_EXTERNAL)?;
    // No onboard call yet — pull hasn't resolved.
    assert_eq!(h.transport.onboard_calls().len(), 0);

    // Now resolve the pull. Setup task picks up stashed G1 and kicks
    // onboard.
    h.transport.resolve_pull(0, Ok(()));

    h.transport.wait_onboard_count(1).await;
    let onboard = h.transport.onboard_calls()[0].clone();
    assert_eq!(onboard.dst_g1_block_ids, h.g1_block_ids);

    h.transport.resolve_onboard(0, Ok(()));
    wait_until(|| h.workers.completed_contains("req-1")).await;
    wait_until(|| {
        h.coordinator.status_for("req-1") == Some(PrefillStatus::OnboardingComplete)
    })
    .await;

    Ok(())
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn cd_prefill_multi_chunk_publish() -> Result<()> {
    let h = build_harness(true);

    h.wrapper.create_slot(make_request())?;
    let _ = h.wrapper.get_num_new_matched_tokens("req-1", 0)?;

    wait_until(|| h.attacher.last().is_some()).await;
    let session = h.attacher.last().expect("session attached");
    wait_until(|| !session.requested_block_sets().is_empty()).await;
    session.enqueue_block_set_response(scripted_block_set_response(
        "req-1",
        &h.decode_g2_block_ids,
        &h.all_hashes,
    ));

    h.transport.wait_pull_count(1).await;
    h.transport.resolve_pull(0, Ok(()));
    wait_until(|| h.coordinator.status_for("req-1") == Some(PrefillStatus::Registered)).await;

    h.wrapper
        .update_state_after_alloc("req-1", h.g1_block_ids.clone(), NUM_EXTERNAL)?;
    h.transport.wait_onboard_count(1).await;
    h.transport.resolve_onboard(0, Ok(()));
    wait_until(|| h.workers.completed_contains("req-1")).await;

    // First chunk: 1 block.
    h.coordinator
        .simulate_offload_complete("req-1", vec![h.output_blocks[0].clone()])?;
    // Second chunk: 1 block.
    h.coordinator
        .simulate_offload_complete("req-1", vec![h.output_blocks[1].clone()])?;

    wait_until(|| session.published_output_sets().len() == 2).await;
    let published = session.published_output_sets();
    assert_eq!(published.len(), 2);
    assert_eq!(published[0][0].blocks.len(), 1);
    assert_eq!(published[1][0].blocks.len(), 1);
    // Different hashes per chunk.
    assert_ne!(published[0][0].blocks[0].sequence_hash, published[1][0].blocks[0].sequence_hash);

    // Single PullComplete releases both chunks' pins.
    let _ = h.wrapper.request_finished("req-1");
    session
        .push_event(SessionEvent::PullComplete(PullComplete {
            pull_id: 7,
            hashes: vec![],
        }))
        .expect("push pull complete");

    wait_until(|| !session.pull_acks().is_empty()).await;
    assert_eq!(session.pull_acks().len(), 1);
    wait_until(|| h.coordinator.active_count() == 0).await;

    Ok(())
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn cd_prefill_non_cd_request_passes_through() -> Result<()> {
    let h = build_harness(false);

    h.wrapper.create_slot(make_request())?;

    // GNMT: no transfer_params on slot → wrapper falls through
    // to inner.GNMT, which our mock returns as (Some(7*16), false).
    let (count, async_flag) = h.wrapper.get_num_new_matched_tokens("req-1", 0)?;
    assert_eq!(count, Some(7 * BLOCK_SIZE));
    assert!(!async_flag);

    // No coordinator state was created for this request.
    assert_eq!(h.coordinator.active_count(), 0);
    assert_eq!(h.coordinator.status_for("req-1"), None);

    // No session attach happened.
    assert!(h.attacher.last().is_none());

    // USAA passthrough: inner gets called, coordinator's on_usaa
    // is a no-op for non-CD requests.
    h.wrapper
        .update_state_after_alloc("req-1", h.g1_block_ids.clone(), 0)?;
    let slot = h.inner.slot("req-1").unwrap();
    {
        let calls = slot.usaa_passthrough_calls.lock();
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].1, 0);
    }

    // request_finished: passes through; coordinator on_request_finished
    // is a no-op for non-CD requests.
    let _ = h.wrapper.request_finished("req-1");
    assert_eq!(h.coordinator.active_count(), 0);

    Ok(())
}

// Suppress unused-warnings for fields not exercised in every test.
#[allow(dead_code)]
fn _ensure_used(h: &TestHarness) {
    let _ = (
        &h.inner,
        &h.transport,
        &h.workers,
        &h.attacher,
        &h.all_hashes,
        &h.g1_block_ids,
        &h.decode_g2_block_ids,
        &h.output_blocks,
        Duration::from_secs(0),
    );
}
