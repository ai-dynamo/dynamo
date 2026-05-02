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
    build_harness_with_watchdog(with_transfer_params, Duration::from_secs(60))
}

fn build_harness_with_watchdog(with_transfer_params: bool, watchdog: Duration) -> TestHarness {
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

    let coordinator = PrefillCoordinatorImpl::new_with_watchdog(
        inner.clone(),
        transport.clone(),
        workers.clone(),
        factory.clone(),
        Arc::new(kvbm_connector::connector::leader::disagg::peer_resolver::NoopPeerResolver),
        tokio::runtime::Handle::current(),
        watchdog,
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

    // CD-bound requests bypass inner.update_state_after_alloc:
    // the coordinator's RequestState owns the onboarding flow, so
    // the prefill leader skips inner to avoid start_onboarding's
    // PreparingToOnboard precondition. usaa_passthrough_calls
    // must therefore be empty for CD-tracked requests.
    let slot = h.inner.slot("req-1").unwrap();
    {
        let calls = slot.usaa_passthrough_calls.lock();
        assert_eq!(calls.len(), 0, "CD-bound USAA must NOT delegate to inner");
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

    // request_finished: prefill calls session.finalize()
    // (cooperative — terminators + Frame::Finished). Session
    // stays alive until decode also calls .finalize() at which
    // point both sides reach the rendezvous and trigger velo
    // wire finalize independently.
    let _ = h.wrapper.request_finished("req-1");
    wait_until(|| session.finished_reason().is_some()).await;
    assert!(
        session.closed_reason().is_none(),
        "prefill must NOT call session.close() in cooperative path"
    );
    // Simulate decode also signalling finalize: in MockSession's
    // paired mode this would deliver Frame::Finished; here we
    // inject directly via the test scaffolding.
    session.inject_peer_finished();
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
    wait_until(|| session.finished_reason().is_some()).await;
    session.inject_peer_finished();
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

/// Regression for #24: dropping the prefill `CdOnboardingPayload`
/// (which simulates `process_finished_onboarding` taking the slot's
/// `OnboardingState` after async-load completes) must NOT close the
/// session. The prefill side still needs to forward-pass + offload
/// + publish net-new G2 blocks to decode via `commit_output_blocks`,
/// which calls `session.commit` / `session.make_available`. If the
/// Drop closes the session, those publishes hit a closed channel
/// and decode's `run_remote_pipeline` hangs forever.
///
/// History: in the two-request smoke, R2's prefill side completed
/// async-load → process_finished_onboarding fired → payload Drop
/// called `coordinator.on_request_finished` → session closed at
/// T+7ms. Offload G1→G2 of the 1 net-new block landed at T+18ms,
/// observer.observe → commit_output_blocks → commit on a closed
/// session = silent failure. Decode hung at curl-90s timeout.
/// Fix: payload Drop is now an audit-only no-op; session lifecycle
/// is owned by `PrefillDisaggLeader::request_finished`.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn cd_prefill_payload_drop_does_not_close_session() -> Result<()> {
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

    // Simulate `process_finished_onboarding` taking the cd_payload
    // off the slot. In production this fires immediately after
    // async-load (G2→G1) completes; the payload's Drop runs.
    let slot = h.inner.slot("req-1").unwrap();
    let payload = slot.installed_cd_payload.lock().take();
    assert!(payload.is_some(), "expected cd_payload to be installed");
    drop(payload);

    // Session must still be open AND coordinator state must still
    // be tracked — the offload-pipeline observer + publish-back to
    // decode happen AFTER this drop in production.
    assert!(
        session.closed_reason().is_none(),
        "Drop must not close the session — publish-back hasn't happened yet"
    );
    assert_eq!(
        h.coordinator.active_count(),
        1,
        "coordinator state must persist past async-load drop"
    );

    // The post-drop publish-back path: simulate the offload
    // observer calling commit_output_blocks. This must succeed.
    h.coordinator
        .commit_output_blocks("req-1", h.output_blocks.clone())?;
    wait_until(|| !session.commit_calls().is_empty()).await;
    let commits = session.commit_calls();
    assert_eq!(commits.len(), 1);
    assert_eq!(commits[0].len(), 2);

    // vLLM's `request_finished` triggers cooperative finalize
    // (terminators + Frame::Finished). Decode is then simulated
    // signalling its own finalize via inject_peer_finished; the
    // rendezvous fires and the watcher evicts RequestState.
    let _ = h.wrapper.request_finished("req-1");
    wait_until(|| session.finished_reason().is_some()).await;
    assert!(
        session.closed_reason().is_none(),
        "prefill on_request_finished must NOT call session.close()"
    );
    session.inject_peer_finished();
    wait_until(|| h.coordinator.active_count() == 0).await;

    Ok(())
}

/// Stage 1 verification: lifecycle watcher's watchdog evicts
/// RequestState if no peer Detach arrives. Belt-and-suspenders
/// against velo heartbeat misconfiguration.
///
/// Uses an injected short watchdog (200ms) via
/// [`PrefillCoordinatorImpl::new_with_watchdog`]; production is
/// 60s and is preserved by [`PrefillCoordinatorImpl::new`].
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn cd_prefill_lifecycle_watchdog_evicts_state() -> Result<()> {
    let h = build_harness_with_watchdog(true, Duration::from_millis(200));
    h.wrapper.create_slot(make_request())?;
    let _ = h.wrapper.get_num_new_matched_tokens("req-1", 0)?;
    let session = drive_setup(&h).await;
    wait_until(|| h.coordinator.status_for("req-1") == Some(PrefillStatus::Registered)).await;
    h.wrapper
        .update_state_after_alloc("req-1", h.g1_block_ids.clone(), NUM_EXTERNAL)?;
    h.transport.wait_onboard_count(1).await;
    h.transport.resolve_onboard(0, Ok(()));
    wait_until(|| h.workers.completed_contains("req-1")).await;
    let _ = h.wrapper.request_finished("req-1");
    wait_until(|| session.finished_reason().is_some()).await;
    // Do NOT inject Detach — let the watchdog fire.  At 200ms it
    // is well under the 60s ignored ceiling but enough to prove
    // the watchdog path runs without timing flakes.
    tokio::time::timeout(
        Duration::from_secs(5),
        wait_until(|| h.coordinator.active_count() == 0),
    )
    .await
    .expect("watchdog should evict within 5s when set to 200ms");
    Ok(())
}

/// Slice A — post-USAA failure surfaces the FULL G1 window to vLLM.
///
/// vLLM allocates G1 destinations for the entire prefill window
/// (local_match + remote-computed); when prefill fails mid-pipeline
/// after USAA has stashed those ids, every slot must be marked
/// failed so the scheduler aborts the whole request rather than
/// proceeding with partially-loaded blocks.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn cd_prefill_cleanup_post_usaa_surfaces_full_g1_window() -> Result<()> {
    let h = build_harness(true);

    h.wrapper.create_slot(make_request())?;
    let _ = h.wrapper.get_num_new_matched_tokens("req-1", 0)?;

    drive_setup(&h).await;
    wait_until(|| h.coordinator.status_for("req-1") == Some(PrefillStatus::Registered)).await;

    // USAA stashes the full G1 window in RequestState.
    h.wrapper
        .update_state_after_alloc("req-1", h.g1_block_ids.clone(), NUM_EXTERNAL)?;

    // Don't resolve onboard — induce a mid-flight failure path.
    h.coordinator
        .cleanup_failed_request("req-1", "induced post-USAA failure".to_string())
        .await;

    // mark_failed_onboarding fired with the FULL window so vLLM
    // aborts every allocated slot.
    let failed = h
        .workers
        .failed_for("req-1")
        .expect("mark_failed_onboarding must fire");
    assert_eq!(failed.request_id, "req-1");
    assert_eq!(
        failed.block_ids, h.g1_block_ids,
        "must surface the full G1 window vLLM allocated, not the local-match prefix"
    );

    // State entry evicted from DashMap.  Observer residual cleanup
    // is gated on dropping the LAST Arc<RequestState> — the
    // pending kick_onboard task still holds one in this scenario,
    // which is realistic (cleanup races mid-flight tasks).  Residual
    // drops once the task drains; covered by the
    // `observer_handle_drop_evicts_pending_entry` unit test.
    assert_eq!(h.coordinator.active_count(), 0);

    Ok(())
}

/// Slice A — pre-USAA failure surfaces the request_id with empty
/// block_ids; the worker-side handler still pairs the request_id
/// with `get_finished()`'s onboard set so vLLM moves the request
/// out of `WAITING_FOR_REMOTE_KVS`. Documented limitation per
/// `cd-error-path-design.md` §6 Q2: failed blocks cannot be
/// reported because no G1 ids exist before USAA.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn cd_prefill_cleanup_pre_usaa_surfaces_request_id_only() -> Result<()> {
    let h = build_harness(true);

    h.wrapper.create_slot(make_request())?;
    let _ = h.wrapper.get_num_new_matched_tokens("req-1", 0)?;
    assert_eq!(h.coordinator.active_count(), 1);

    // No drive_setup, no USAA — state is installed but g1_block_ids
    // is still None.  Simulate an early failure (e.g. attach error,
    // peer-resolve error, hub queue dispatch error).
    h.coordinator
        .cleanup_failed_request("req-1", "induced pre-USAA failure".to_string())
        .await;

    let failed = h
        .workers
        .failed_for("req-1")
        .expect("mark_failed_onboarding must fire even pre-USAA");
    assert_eq!(failed.request_id, "req-1");
    assert!(
        failed.block_ids.is_empty(),
        "pre-USAA failure has no G1 ids (worker handler pairs request_id regardless)"
    );

    assert_eq!(h.coordinator.active_count(), 0);

    Ok(())
}

/// Slice A — cleanup is idempotent against state-already-evicted.
/// Multiple failure-detection paths (run_setup Err, lifecycle
/// escalation, deadline timer) may converge on the same request;
/// only the first call propagates.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn cd_prefill_cleanup_is_idempotent() -> Result<()> {
    let h = build_harness(true);

    h.wrapper.create_slot(make_request())?;
    let _ = h.wrapper.get_num_new_matched_tokens("req-1", 0)?;

    h.coordinator
        .cleanup_failed_request("req-1", "first call".to_string())
        .await;
    h.coordinator
        .cleanup_failed_request("req-1", "second call".to_string())
        .await;

    let failed_calls: Vec<_> = h
        .workers
        .failed()
        .into_iter()
        .filter(|c| c.request_id == "req-1")
        .collect();
    assert_eq!(
        failed_calls.len(),
        1,
        "cleanup must not double-fire mark_failed_onboarding"
    );

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
