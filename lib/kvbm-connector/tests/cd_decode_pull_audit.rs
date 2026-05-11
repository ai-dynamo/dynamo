// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Observability tests for the worker NIXL pull at the
//! `pull_register_onboard_contiguous_chunk` `await ?` site
//! (`decode_leader.rs`'s `session.pull(...)`).
//!
//! Bug C from PR #9077 manifested as `worker_session_pull_call`
//! firing without a matching `worker_session_pull_returned`, with no
//! audit signal on the failure path itself — the `?` in the await
//! propagated up to `cleanup_failed_request` and the connector then
//! looked indistinguishable from a hang.
//!
//! This file uses the `MockSession` failure injection (via
//! `MockSessionFactory` + `wait_pull_count` + `resolve_pull(_, Err)`)
//! to drive the await into Err and asserts that:
//!
//! - `worker_session_pull_call` fires (baseline marker preserved).
//! - `worker_session_pull_error` fires AFTER the pull resolves Err,
//!   with `error_kind`, `error_msg`, and `elapsed_ms` populated.
//! - `worker_session_pull_returned` does NOT fire on the failed
//!   chunk (so post-mortem traces don't show a phantom completion).
//!
//! No new mocker types are introduced — this test reuses the same
//! `MockInnerLeaderShim` / `MockSessionFactory` infrastructure
//! `cd_decode_remote_pull_failure` already exercises.

mod audit_helpers;

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
    AlwaysRemote, ConditionalDisaggCoordinator, ConnectorLeaderApi, DecodeDisaggLeader,
};
use kvbm_engine::disagg::session::{CommittedBlock, MockSessionFactory};
use kvbm_engine::testing::managers::{TestManagerBuilder, TestRegistryBuilder};
use kvbm_engine::testing::token_blocks::{create_token_sequence, generate_sequence_hashes};
use kvbm_logical::manager::BlockManager;

use audit_helpers::{AuditEvent, install_collector};

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
    #[allow(dead_code)]
    inner: Arc<MockInnerLeaderShim>,
    transport: Arc<MockCdBlockTransport>,
    workers: Arc<MockCdWorkerHook>,
    factory: Arc<MockSessionFactory>,
    #[allow(dead_code)]
    queue: Arc<InMemoryRemotePrefillQueue>,
    #[allow(dead_code)]
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

fn committed_blocks(
    hashes: &[kvbm_logical::SequenceHash],
    base: usize,
) -> Vec<CommittedBlock> {
    hashes
        .iter()
        .enumerate()
        .map(|(i, hash)| CommittedBlock {
            hash: *hash,
            peer_block_id: base + i,
        })
        .collect()
}

fn remote_hashes(h: &TestHarness) -> Vec<kvbm_logical::SequenceHash> {
    h.all_hashes[COMPUTED_BLOCKS + LOCAL_BLOCKS..].to_vec()
}

/// Drive the decode pipeline through gnmt + USAA + the local-kick
/// completion, then commit + make-available the remote slice. Returns
/// the freshly-opened `MockSession` so the caller can `resolve_pull`
/// it.
async fn drive_decode_through_remote_pull_dispatch(
    h: &TestHarness,
) -> Result<Arc<kvbm_engine::disagg::session::MockSession>> {
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

    h.transport.wait_onboard_count(1).await;
    h.transport.resolve_onboard(0, Ok(()));

    session.inject_peer_commit(remote_hashes(h));
    session.inject_peer_finish_commits();
    session.inject_peer_available(committed_blocks(&remote_hashes(h), 9000));
    session.inject_peer_drained();

    Ok(session)
}

/// Find the FIRST audit event with the given name. Returns `None` if
/// no event with that name exists in the captured stream.
fn find_event<'a>(events: &'a [AuditEvent], name: &str) -> Option<&'a AuditEvent> {
    events.iter().find(|e| e.event == name)
}

/// Get a field value by name (sorted in `AuditEvent::fields`).
fn field<'a>(event: &'a AuditEvent, name: &str) -> Option<&'a str> {
    event
        .fields
        .iter()
        .find(|(k, _)| k == name)
        .map(|(_, v)| v.as_str())
}

/// Assert the audit event chain on a `session.pull` async failure
/// surfaces the new `worker_session_pull_error` with `error_kind`
/// = `XferReqCreate` (the "createXferReq fast-fail" / "hash not in
/// peer_available" semantic class).
///
/// The MockSession's manual-mode pull resolver lands in `Err(...)`
/// when the test calls `resolve_pull(0, Err(...))`. The connector's
/// `pull_register_onboard_contiguous_chunk` matches on the result;
/// on Err it emits `worker_session_pull_error` and returns the error
/// up the `?` chain to `cleanup_failed_request`. The terminal
/// `worker_session_pull_returned` event must NOT fire on the failed
/// chunk.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn cd_decode_pull_error_audit_emits_xfer_req_create() -> Result<()> {
    // Serialize against any other audit-capturing tests in the same
    // binary (the capture buffer is process-global).
    let _audit_guard = audit_helpers::audit_test_lock();
    let capture = install_collector();
    // Drain anything left over from a previous test.
    let _ = capture.drain();

    let h = build_harness();

    let session = drive_decode_through_remote_pull_dispatch(&h).await?;

    // Wait for the wrapper to dispatch ONE pull, then inject a
    // sync-style failure ("create_xfer_req: no backend matched").
    // The classifier maps any "create_xfer_req" / "createXferReq"
    // substring to `XferReqCreate`, which is the canonical fast-fail
    // class.
    session.wait_pull_count(1).await;
    session.resolve_pull(
        0,
        Err(anyhow::anyhow!(
            "createXferReq failed: no NIXL backend matches Unknown layout"
        )),
    );

    // Wait for cleanup to land — `mark_failed_onboarding` is the
    // terminal sentinel that cleanup_failed_request has run.
    wait_until(|| h.workers.failed_for("req-1").is_some()).await;

    // Drain the captured audit stream and look for the new event.
    let events = capture.drain();

    // 1. Baseline: `worker_session_pull_call` must precede the error
    //    so post-mortem ordering holds.
    let call_idx = events
        .iter()
        .position(|e| e.event == "worker_session_pull_call")
        .expect("worker_session_pull_call must fire on dispatch");
    assert_eq!(
        find_event(&events, "worker_session_pull_call")
            .and_then(|e| e.role.as_deref()),
        Some("decode")
    );

    // 2. `worker_session_pull_error` must fire AFTER the call audit,
    //    carry the right kind/msg/elapsed fields, and reference the
    //    same request_id.
    let err_idx = events
        .iter()
        .position(|e| e.event == "worker_session_pull_error")
        .expect(
            "worker_session_pull_error must fire when session.pull() resolves Err — \
             this is the regression-guard for Bug C observability",
        );
    assert!(
        err_idx > call_idx,
        "worker_session_pull_error must fire AFTER worker_session_pull_call, \
         got call_idx={} err_idx={}",
        call_idx,
        err_idx
    );

    let err_event = &events[err_idx];
    assert_eq!(err_event.role.as_deref(), Some("decode"));
    assert_eq!(err_event.request_id.as_deref(), Some("req-1"));

    let kind = field(err_event, "error_kind").expect("error_kind field present");
    assert_eq!(
        kind, "XferReqCreate",
        "createXferReq-shaped failure must classify as XferReqCreate, got {kind:?}"
    );

    let msg = field(err_event, "error_msg").expect("error_msg field present");
    assert!(
        msg.to_ascii_lowercase().contains("createxferreq")
            || msg.to_ascii_lowercase().contains("create_xfer_req")
            || msg.to_ascii_lowercase().contains("xferreq"),
        "error_msg should preserve the original anyhow chain: got {msg:?}"
    );

    let elapsed = field(err_event, "elapsed_ms").expect("elapsed_ms field present");
    let _: u64 = elapsed
        .parse()
        .expect("elapsed_ms must parse as a non-negative integer");

    // 3. `worker_session_pull_returned` must NOT fire on the failed
    //    chunk — that's the trace-readability invariant we're
    //    actually defending.
    assert!(
        find_event(&events, "worker_session_pull_returned").is_none(),
        "worker_session_pull_returned must NOT fire on a chunk whose pull resolved Err"
    );

    // Belt-and-suspenders: the workers see the failure surface to
    // vLLM (not absorbed by the audit hook).
    let failed = h
        .workers
        .failed_for("req-1")
        .expect("mark_failed_onboarding must still fire after the audit hook");
    assert!(
        !failed.block_ids.is_empty(),
        "audit hook must not swallow the failure: vLLM still gets the unfilled G1 ids"
    );

    Ok(())
}

/// Sanity check: a successful pull keeps the existing
/// `call → returned` event pair intact and does NOT emit
/// `worker_session_pull_error`. Guards against the new error-emit
/// branch leaking onto the success path.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn cd_decode_pull_success_does_not_emit_error_audit() -> Result<()> {
    let _audit_guard = audit_helpers::audit_test_lock();
    let capture = install_collector();
    let _ = capture.drain();

    let h = build_harness();
    let session = drive_decode_through_remote_pull_dispatch(&h).await?;

    session.wait_pull_count(1).await;
    session.resolve_pull(0, Ok(()));

    // Wait for the pipeline to complete cleanly. `pull_progress` with
    // `terminal=true` is the last audit event we control before
    // remote_pipeline_complete_set fires; `mark_onboarding_complete`
    // is the sentinel that the wrapper has finished onboarding.
    wait_until(|| {
        let snap = capture.snapshot();
        snap.iter().any(|e| e.event == "mark_onboarding_complete")
    })
    .await;

    let events = capture.drain();
    assert!(
        find_event(&events, "worker_session_pull_call").is_some(),
        "call event must fire on dispatch"
    );
    assert!(
        find_event(&events, "worker_session_pull_returned").is_some(),
        "returned event must fire on Ok"
    );
    assert!(
        find_event(&events, "worker_session_pull_error").is_none(),
        "error event must NOT fire on a successful pull"
    );

    // pull_progress should have a terminal=true row when filled
    // reached the expected count. Asserting just on its presence is
    // enough — the field shape is exercised in tests covering the
    // remote-pipeline progress more directly.
    let progress_terminal = events
        .iter()
        .filter(|e| e.event == "pull_progress")
        .filter_map(|e| field(e, "terminal"))
        .any(|v| v == "true");
    assert!(
        progress_terminal,
        "pull_progress with terminal=true must fire when filled.len() reaches expected"
    );

    Ok(())
}

/// The `Duration` import is used by future expansions (e.g. asserting
/// elapsed_ms is plausibly > 0 when artificial latency is injected).
/// Keep it here so the test file follows the surrounding convention
/// in `cd_decode_e2e.rs`.
#[allow(dead_code)]
fn _keep_duration_used(d: Duration) -> Duration {
    d
}

/// Regression guard for Bug C's exact failure mode: `session.pull(...)`
/// hangs forever (never resolves Ok or Err). Pre-heartbeat, this looked
/// indistinguishable in audit traces from a sync-fail-and-cleanup —
/// `worker_session_pull_call` would fire, then dead silence until
/// `cleanup_failed_request` (if the request even reached that path).
///
/// The new `pull_heartbeat` audit MUST emit at the configured interval
/// while the await is suspended. We tighten the interval to 50ms via
/// `KVBM_PULL_HEARTBEAT_MS=50` and wait ~250ms — that should reliably
/// produce at least 2 heartbeats. Each heartbeat must carry
/// `chunk_start_slot`, `hashes_in_chunk`, `elapsed_ms`, and reference the
/// same `request_id`.
///
/// We deliberately do NOT call `resolve_pull` — the pull future is left
/// pending and the harness is torn down at scope exit. This mirrors
/// Bug C's "stuck inside session.pull" symptom exactly.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn cd_decode_pull_hang_emits_heartbeats() -> Result<()> {
    let _audit_guard = audit_helpers::audit_test_lock();
    // SAFETY: env-var write is process-global; the audit_test_lock above
    // serializes audit-capturing tests, so no other test reads this var
    // concurrently. Tighter interval keeps test runtime down to ~300ms.
    // Restored at function exit via the guard pattern below.
    struct EnvGuard {
        key: &'static str,
        prior: Option<String>,
    }
    impl Drop for EnvGuard {
        fn drop(&mut self) {
            match self.prior.take() {
                Some(v) => unsafe { std::env::set_var(self.key, v) },
                None => unsafe { std::env::remove_var(self.key) },
            }
        }
    }
    let _env = EnvGuard {
        key: "KVBM_PULL_HEARTBEAT_MS",
        prior: std::env::var("KVBM_PULL_HEARTBEAT_MS").ok(),
    };
    unsafe { std::env::set_var("KVBM_PULL_HEARTBEAT_MS", "50") };

    let capture = install_collector();
    let _ = capture.drain();

    let h = build_harness();
    let session = drive_decode_through_remote_pull_dispatch(&h).await?;

    // Wait for the wrapper to start ONE pull; then deliberately DO NOT
    // resolve it. The select! race in pull_register_onboard_contiguous_chunk
    // should fall through to `heartbeat.tick()` every ~50ms and emit
    // `pull_heartbeat`.
    session.wait_pull_count(1).await;

    // ~6 heartbeat windows at 50ms each. Plenty of headroom for CI jitter
    // while still keeping the test fast.
    tokio::time::sleep(Duration::from_millis(300)).await;

    let events = capture.snapshot();

    // 1. Must see worker_session_pull_call exactly once on dispatch.
    assert_eq!(
        events
            .iter()
            .filter(|e| e.event == "worker_session_pull_call")
            .count(),
        1,
        "worker_session_pull_call should fire once on chunk dispatch"
    );

    // 2. Must see >=2 pull_heartbeat events while the pull stays
    //    suspended. >=2 guarantees the ticker is actually firing on
    //    repeat (not just the first iteration of the loop).
    let heartbeats: Vec<&AuditEvent> = events
        .iter()
        .filter(|e| e.event == "pull_heartbeat")
        .collect();
    assert!(
        heartbeats.len() >= 2,
        "expected >=2 pull_heartbeat events in 300ms at 50ms interval, got {} (events: {:?})",
        heartbeats.len(),
        events.iter().map(|e| &e.event).collect::<Vec<_>>()
    );

    // 3. Each heartbeat must carry the required diagnostic fields.
    for hb in &heartbeats {
        assert_eq!(hb.role.as_deref(), Some("decode"));
        assert_eq!(hb.request_id.as_deref(), Some("req-1"));
        let chunk_slot = field(hb, "chunk_start_slot")
            .expect("chunk_start_slot field present on pull_heartbeat");
        let _: usize = chunk_slot
            .parse()
            .expect("chunk_start_slot must parse as usize");
        let hashes = field(hb, "hashes_in_chunk")
            .expect("hashes_in_chunk field present on pull_heartbeat");
        let n_hashes: usize = hashes.parse().expect("hashes_in_chunk must parse");
        assert!(n_hashes > 0, "hashes_in_chunk must be > 0");
        let elapsed = field(hb, "elapsed_ms").expect("elapsed_ms field present");
        let _: u64 = elapsed.parse().expect("elapsed_ms must parse");
    }

    // 4. elapsed_ms must be monotonically non-decreasing across the
    //    heartbeat stream — the field is `pull_started.elapsed()`, which
    //    is wall-clock-derived and only increases.
    let elapsed_ms_series: Vec<u64> = heartbeats
        .iter()
        .map(|e| {
            field(e, "elapsed_ms")
                .and_then(|v| v.parse::<u64>().ok())
                .unwrap_or(0)
        })
        .collect();
    for w in elapsed_ms_series.windows(2) {
        assert!(
            w[1] >= w[0],
            "pull_heartbeat elapsed_ms must be monotonically non-decreasing, got {:?}",
            elapsed_ms_series
        );
    }

    // 5. Neither pull_returned nor pull_error should fire — we never
    //    resolved the pull. This is the "looks like a hang in audit
    //    traces" failure-mode we're catching.
    assert!(
        find_event(&events, "worker_session_pull_returned").is_none(),
        "pull_returned must NOT fire while the pull is suspended"
    );
    assert!(
        find_event(&events, "worker_session_pull_error").is_none(),
        "pull_error must NOT fire while the pull is suspended"
    );

    // Resolve the pull with an error so cleanup can run and the test
    // tears down cleanly instead of leaking a hung task.
    session.resolve_pull(0, Err(anyhow::anyhow!("test teardown — resolve pending pull")));
    Ok(())
}
