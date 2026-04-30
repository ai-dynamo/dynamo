// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Prefill-side end-to-end test for the CD wrapper, against the new
//! symmetric `Session` API (MockSession + MockSessionFactory).
//!
//! Mocks: `MockInnerLeaderShim`, `MockCdBlockTransport` (for the
//! G2→G1 onboard only), `MockCdWorkerHook`, `MockSessionFactory`.
//! No velo, no real RDMA.

use std::sync::Arc;
use std::time::Duration;

use anyhow::Result;
use kvbm_connector::G2;
use kvbm_connector::common::Request;
use kvbm_connector::connector::leader::disagg::prefill_coordinator::{
    PrefillCoordinatorImpl, PrefillStatus,
};
use kvbm_connector::connector::leader::disagg::testing::{
    MockCdBlockTransport, MockCdWorkerHook, MockInnerLeaderShim, MockSlot, TEST_BLOCK_SIZE,
    wait_until,
};
use kvbm_connector::connector::leader::disagg::{ConnectorLeaderApi, PrefillDisaggLeader};
use kvbm_disagg_protocol::{
    DISAGG_PROTOCOL_VERSION, RemotePrefillParams, SessionEndpoint, SessionId, TransferParams,
};
use kvbm_engine::disagg::session::{CommittedBlock, MockSession, MockSessionFactory};
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
        num_computed_tokens: 0,
    })
}

struct TestHarness {
    wrapper: Arc<PrefillDisaggLeader>,
    coordinator: Arc<PrefillCoordinatorImpl>,
    inner: Arc<MockInnerLeaderShim>,
    transport: Arc<MockCdBlockTransport>,
    workers: Arc<MockCdWorkerHook>,
    factory: Arc<MockSessionFactory>,
    all_hashes: Vec<kvbm_logical::SequenceHash>,
    g1_block_ids: Vec<usize>,
    decode_g2_block_ids: Vec<usize>,
    output_blocks: Vec<ImmutableBlock<G2>>,
    decode_instance_id: kvbm_connector::InstanceId,
}

fn build_harness(with_transfer_params: bool) -> TestHarness {
    let g2_manager = build_g2_manager(64);

    let token_sequence = create_token_sequence(TOTAL_BLOCKS, BLOCK_SIZE, 100);
    let all_hashes = generate_sequence_hashes(&token_sequence);
    let token_blocks: Vec<_> = token_sequence.blocks().to_vec();
    assert_eq!(all_hashes.len(), TOTAL_BLOCKS);

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
        computed_blocks: 0,
        local_match_blocks: 0,
        all_hashes: all_hashes.clone(),
        token_blocks,
        local_match_g2: parking_lot::Mutex::new(Some(Vec::new())),
        assigned_block_ids: parking_lot::Mutex::new(None),
        gnmt_result: (Some(7 * BLOCK_SIZE), false),
        usaa_passthrough_calls: parking_lot::Mutex::new(Vec::new()),
        transfer_params,
        ..MockSlot::default()
    };
    inner.install_slot("req-1", slot);

    let transport = MockCdBlockTransport::new();
    let workers = MockCdWorkerHook::new();
    let factory = MockSessionFactory::new();

    let coordinator = PrefillCoordinatorImpl::new(
        inner.clone(),
        transport.clone(),
        workers.clone(),
        factory.clone(),
        Arc::new(kvbm_connector::connector::leader::disagg::peer_resolver::NoopPeerResolver),
        tokio::runtime::Handle::current(),
    );

    let wrapper =
        PrefillDisaggLeader::from_parts(inner.clone(), coordinator.clone(), workers.clone());

    TestHarness {
        wrapper,
        coordinator,
        inner,
        transport,
        workers,
        factory,
        all_hashes,
        g1_block_ids,
        decode_g2_block_ids,
        output_blocks,
        decode_instance_id,
    }
}

/// Build a `Vec<CommittedBlock>` carrying decode's local-match
/// hashes mapped to scripted peer block_ids.
fn committed_blocks(
    decode_g2_block_ids: &[usize],
    expected_hashes: &[kvbm_logical::SequenceHash],
) -> Vec<CommittedBlock> {
    expected_hashes
        .iter()
        .zip(decode_g2_block_ids.iter())
        .map(|(hash, id)| CommittedBlock {
            hash: *hash,
            peer_block_id: *id,
        })
        .collect()
}

/// Drive the standard prefill setup: wait for attach, inject
/// commits + availability, resolve pull. Returns the
/// MockSession.
async fn drive_setup(h: &TestHarness) -> Arc<MockSession> {
    wait_until(|| h.factory.last_attached().is_some()).await;
    let session = h.factory.last_attached().expect("session");

    // Verify attach passed peer_instance_id correctly.
    let calls = h.factory.attach_calls();
    assert_eq!(calls.len(), 1);
    assert_eq!(calls[0].1, h.decode_instance_id);

    // Inject the peer's commits, then the available blocks.
    session.inject_peer_commit(h.all_hashes.clone());
    session.inject_peer_finish_commits();
    session.inject_peer_available(committed_blocks(&h.decode_g2_block_ids, &h.all_hashes));
    session.inject_peer_drained();

    // Coordinator calls session.pull(...) once it's drained the
    // commit + availability streams. Resolve it.
    session.wait_pull_count(1).await;
    let pull = session.pull_calls()[0].clone();
    assert_eq!(pull.0.len(), TOTAL_BLOCKS);
    session.resolve_pull(0, Ok(()));

    session
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn cd_prefill_happy_path() -> Result<()> {
    let h = build_harness(true);

    h.wrapper.create_slot(make_request())?;
    let (count, async_flag) = h.wrapper.get_num_new_matched_tokens("req-1", 0)?;
    assert_eq!(count, Some(NUM_EXTERNAL));
    assert!(async_flag);
    assert_eq!(h.coordinator.active_count(), 1);

    let session = drive_setup(&h).await;

    // Wait for register-then-Registered.
    wait_until(|| h.coordinator.status_for("req-1") == Some(PrefillStatus::Registered)).await;

    // USAA arrives (post-register). Wrapper calls inner USAA, then
    // coordinator.on_usaa kicks the G2→G1 onboard via transport.
    h.wrapper
        .update_state_after_alloc("req-1", h.g1_block_ids.clone(), NUM_EXTERNAL)?;

    let slot = h.inner.slot("req-1").unwrap();
    {
        let calls = slot.usaa_passthrough_calls.lock();
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].0, h.g1_block_ids);
    }

    // G2→G1 onboard fires.
    h.transport.wait_onboard_count(1).await;
    let onboard = h.transport.onboard_calls()[0].clone();
    assert_eq!(onboard.dst_g1_block_ids, h.g1_block_ids);
    assert_eq!(onboard.src_g2_block_ids.len(), TOTAL_BLOCKS);
    h.transport.resolve_onboard(0, Ok(()));

    wait_until(|| h.workers.completed_contains("req-1")).await;
    wait_until(|| h.coordinator.status_for("req-1") == Some(PrefillStatus::OnboardingComplete))
        .await;

    // Forward-pass output: commit + make_available via the
    // production-shaped helper.
    h.coordinator
        .commit_output_blocks("req-1", h.output_blocks.clone())?;

    // The MockSession records commit + make_available calls.
    wait_until(|| !session.commit_calls().is_empty()).await;
    let commits = session.commit_calls();
    assert_eq!(commits.len(), 1);
    assert_eq!(commits[0].len(), 2);
    let avails = session.make_available_calls();
    assert_eq!(avails.len(), 1);
    assert_eq!(avails[0].len(), 2);

    // request_finished: coordinator finishes streams + closes session.
    let _ = h.wrapper.request_finished("req-1");
    wait_until(|| h.coordinator.active_count() == 0).await;
    assert!(session.closed_reason().is_some());

    Ok(())
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn cd_prefill_usaa_before_pull_completes() -> Result<()> {
    let h = build_harness(true);

    h.wrapper.create_slot(make_request())?;
    let (count, async_flag) = h.wrapper.get_num_new_matched_tokens("req-1", 0)?;
    assert_eq!(count, Some(NUM_EXTERNAL));
    assert!(async_flag);

    wait_until(|| h.factory.last_attached().is_some()).await;
    let session = h.factory.last_attached().expect("session");

    // Inject commits + availability — coordinator will call
    // session.pull() but we DON'T resolve it yet.
    session.inject_peer_commit(h.all_hashes.clone());
    session.inject_peer_finish_commits();
    session.inject_peer_available(committed_blocks(&h.decode_g2_block_ids, &h.all_hashes));

    session.wait_pull_count(1).await;
    assert!(matches!(
        h.coordinator.status_for("req-1"),
        Some(PrefillStatus::Pulling)
    ));

    // USAA arrives early. Coordinator should stash G1 ids.
    h.wrapper
        .update_state_after_alloc("req-1", h.g1_block_ids.clone(), NUM_EXTERNAL)?;
    assert_eq!(h.transport.onboard_calls().len(), 0);

    // Resolve the pull. Setup task picks up stashed G1 + kicks onboard.
    session.resolve_pull(0, Ok(()));

    h.transport.wait_onboard_count(1).await;
    let onboard = h.transport.onboard_calls()[0].clone();
    assert_eq!(onboard.dst_g1_block_ids, h.g1_block_ids);
    h.transport.resolve_onboard(0, Ok(()));

    wait_until(|| h.workers.completed_contains("req-1")).await;
    wait_until(|| h.coordinator.status_for("req-1") == Some(PrefillStatus::OnboardingComplete))
        .await;

    Ok(())
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn cd_prefill_multi_chunk_publish() -> Result<()> {
    let h = build_harness(true);

    h.wrapper.create_slot(make_request())?;
    let _ = h.wrapper.get_num_new_matched_tokens("req-1", 0)?;

    let session = drive_setup(&h).await;

    wait_until(|| h.coordinator.status_for("req-1") == Some(PrefillStatus::Registered)).await;

    h.wrapper
        .update_state_after_alloc("req-1", h.g1_block_ids.clone(), NUM_EXTERNAL)?;
    h.transport.wait_onboard_count(1).await;
    h.transport.resolve_onboard(0, Ok(()));
    wait_until(|| h.workers.completed_contains("req-1")).await;

    // Two output chunks: each commit_output_blocks call commits +
    // make_available.
    h.coordinator
        .commit_output_blocks("req-1", vec![h.output_blocks[0].clone()])?;
    h.coordinator
        .commit_output_blocks("req-1", vec![h.output_blocks[1].clone()])?;

    wait_until(|| session.commit_calls().len() == 2).await;
    let commits = session.commit_calls();
    assert_eq!(commits.len(), 2);
    assert_eq!(commits[0].len(), 1);
    assert_eq!(commits[1].len(), 1);
    assert_ne!(commits[0][0], commits[1][0]);

    let _ = h.wrapper.request_finished("req-1");
    wait_until(|| h.coordinator.active_count() == 0).await;

    Ok(())
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn cd_prefill_non_cd_request_passes_through() -> Result<()> {
    let h = build_harness(false);

    h.wrapper.create_slot(make_request())?;

    let (count, async_flag) = h.wrapper.get_num_new_matched_tokens("req-1", 0)?;
    assert_eq!(count, Some(7 * BLOCK_SIZE));
    assert!(!async_flag);

    assert_eq!(h.coordinator.active_count(), 0);
    assert_eq!(h.coordinator.status_for("req-1"), None);
    assert!(h.factory.last_attached().is_none());

    h.wrapper
        .update_state_after_alloc("req-1", h.g1_block_ids.clone(), 0)?;
    let slot = h.inner.slot("req-1").unwrap();
    {
        let calls = slot.usaa_passthrough_calls.lock();
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].1, 0);
    }

    let _ = h.wrapper.request_finished("req-1");
    assert_eq!(h.coordinator.active_count(), 0);

    Ok(())
}

/// Build a CD-bound harness where decode supplied **no** sequence
/// hashes for prefill to onboard from G2. `inner_gnmt` controls the
/// passthrough tuple the inner shim returns; the wrapper must surface
/// it verbatim instead of forging `(Some(0), true)`.
fn build_harness_cd_no_g2_hits(inner_gnmt: (Option<usize>, bool)) -> TestHarness {
    let g2_manager = build_g2_manager(64);
    let token_sequence = create_token_sequence(TOTAL_BLOCKS, BLOCK_SIZE, 100);
    let all_hashes = generate_sequence_hashes(&token_sequence);
    let token_blocks: Vec<_> = token_sequence.blocks().to_vec();

    let inner = MockInnerLeaderShim::new(BLOCK_SIZE, g2_manager.clone());
    let g1_block_ids: Vec<usize> = (1000..1000 + TOTAL_BLOCKS).collect();
    let decode_g2_block_ids: Vec<usize> = (5000..5000 + TOTAL_BLOCKS).collect();

    let session_id = uuid::Uuid::new_v4();
    let decode_instance_id: kvbm_connector::InstanceId = uuid::Uuid::new_v4().into();
    // CD-bound but with **empty** sequence_hashes — mirrors the
    // hub dispatcher payload when decode has no local-match cache to
    // forward (the common golden-path case).
    let transfer_params = Some(cd_transfer_params(
        session_id,
        decode_instance_id,
        Vec::new(),
    ));

    let slot = MockSlot {
        block_size: BLOCK_SIZE,
        total_blocks: TOTAL_BLOCKS,
        computed_blocks: 0,
        local_match_blocks: 0,
        all_hashes: all_hashes.clone(),
        token_blocks,
        local_match_g2: parking_lot::Mutex::new(Some(Vec::new())),
        assigned_block_ids: parking_lot::Mutex::new(None),
        gnmt_result: inner_gnmt,
        usaa_passthrough_calls: parking_lot::Mutex::new(Vec::new()),
        transfer_params,
        ..MockSlot::default()
    };
    inner.install_slot("req-1", slot);

    let transport = MockCdBlockTransport::new();
    let workers = MockCdWorkerHook::new();
    let factory = MockSessionFactory::new();
    let coordinator = PrefillCoordinatorImpl::new(
        inner.clone(),
        transport.clone(),
        workers.clone(),
        factory.clone(),
        Arc::new(kvbm_connector::connector::leader::disagg::peer_resolver::NoopPeerResolver),
        tokio::runtime::Handle::current(),
    );
    let wrapper =
        PrefillDisaggLeader::from_parts(inner.clone(), coordinator.clone(), workers.clone());

    TestHarness {
        wrapper,
        coordinator,
        inner,
        transport,
        workers,
        factory,
        all_hashes,
        g1_block_ids,
        decode_g2_block_ids,
        output_blocks: Vec::new(),
        decode_instance_id,
    }
}

/// Regression: CD-bound prefill request with `sequence_hashes=[]` (no
/// G2 cache hits to onboard) must NOT return `(Some(0), true)`.
///
/// vLLM's scheduler asserts `num_external_computed_tokens > 0` whenever
/// `load_kv_async` is true (`vllm/v1/core/sched/scheduler.py`, search
/// for `num_external_computed_tokens > 0` under `if load_kv_async:`).
/// This invariant is also encoded locally in
/// [`crate::connector::leader::slot::Slot::finalize_match_check`] —
/// `(Some(matched_tokens), matched_tokens > 0)`.
///
/// History: shipping the smoke surfaced this as `EngineCore` panicking
/// on the prefill side mid-request. The wrapper was unconditionally
/// returning `(Some(n), true)` from `ensure_started` even when n=0,
/// violating both the local invariant and vLLM's contract.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn cd_prefill_no_g2_hits_passes_through_inner_gnmt() -> Result<()> {
    // Inner returns the typical "fresh request, no local match" tuple.
    let inner_gnmt = (Some(0), false);
    let h = build_harness_cd_no_g2_hits(inner_gnmt);
    h.wrapper.create_slot(make_request())?;

    let result = h.wrapper.get_num_new_matched_tokens("req-1", 0)?;
    assert_eq!(
        result, inner_gnmt,
        "with empty sequence_hashes the wrapper must passthrough to inner — \
         (Some(0), true) violates vLLM's load_kv_async invariant",
    );

    // Inner shim must have been consulted (the passthrough call).
    // Coordinator's CD setup still ran inside ensure_started so the
    // worker observer can publish blocks during the upcoming forward
    // pass — but with no G2 hits there is no async onboard and the
    // gnmt return is purely the inner's verdict.

    Ok(())
}

/// Pin the `(Some(n>0), true)` half of the same invariant — the
/// existing `cd_prefill_happy_path` already covers this implicitly
/// (NUM_EXTERNAL > 0), but make the rule explicit alongside the n=0
/// regression so the contract is visible in one place.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn cd_prefill_with_g2_hits_returns_async_load() -> Result<()> {
    let h = build_harness(true);
    h.wrapper.create_slot(make_request())?;

    let (count, async_flag) = h.wrapper.get_num_new_matched_tokens("req-1", 0)?;
    let n = count.expect("matched-tokens count must be Some when CD onboards");
    assert!(n > 0, "n>0 required when async_load=true");
    assert!(async_flag, "async_load=true required when n>0 onboards");

    Ok(())
}

#[allow(dead_code)]
fn _ensure_used(h: &TestHarness) {
    let _ = (
        &h.inner,
        &h.transport,
        &h.workers,
        &h.factory,
        &h.all_hashes,
        &h.g1_block_ids,
        &h.decode_g2_block_ids,
        &h.output_blocks,
        Duration::from_secs(0),
    );
}
