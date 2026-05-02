// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Trace-equivalence tests for `UnifiedDisaggLeader`.
//!
//! For each test we run a fixed workload twice — once through the
//! bare role-specific leader (the baseline), once through
//! `UnifiedDisaggLeader` wrapping the same role-specific leader (the
//! "under-test" run).  We capture `kvbm_audit` events for each run
//! and assert the (event, role, request_id) signature sequences
//! match.
//!
//! ### Why this is the right test shape
//!
//! Today, the unified leader's per-request methods *delegate* to the
//! wrapped flow — so the wrapped leader is the source of all audits.
//! If the dispatch ever mis-classifies (e.g. routes a no-params
//! request to inner directly instead of the decode flow), the
//! audit stream would be MISSING entries and the equivalence
//! assertion fires.  When we later refactor the unified leader's
//! tick-level methods to "call inner once, decorate per flow," the
//! same equivalence assertion becomes the regression bar.

mod audit_helpers;

use std::sync::Arc;

use anyhow::Result;
use kvbm_config::{DisaggConfig, DisaggregationRole};
use kvbm_connector::G2;
use kvbm_connector::common::Request;
use kvbm_connector::connector::leader::disagg::testing::{
    InMemoryRemotePrefillQueue, MockCdBlockTransport, MockCdWorkerHook, MockInnerLeaderShim,
    MockSlot, TEST_BLOCK_SIZE,
};
use kvbm_connector::connector::leader::disagg::{
    AlwaysRemote, ConnectorLeaderApi, DecodeDisaggLeader, RemotePrefillCoordinator,
    UnifiedDisaggLeader,
};
use kvbm_engine::disagg::session::MockSessionFactory;
use kvbm_engine::testing::managers::{TestManagerBuilder, TestRegistryBuilder};
use kvbm_engine::testing::token_blocks::{create_token_sequence, generate_sequence_hashes};
use kvbm_logical::manager::BlockManager;

use audit_helpers::{
    AuditCaptureHandle, assert_event_signatures_equal, audit_test_lock, install_collector,
};

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

/// One run's collaborators — fresh per run so cd_request_state and
/// coordinator state don't leak between runs.
struct DecodeRig {
    inner: Arc<MockInnerLeaderShim>,
    decode: Arc<DecodeDisaggLeader>,
    queue: Arc<InMemoryRemotePrefillQueue>,
}

fn build_decode_rig(request_id: &str) -> DecodeRig {
    let g2_manager = build_g2_manager(32);

    let token_sequence = create_token_sequence(TOTAL_BLOCKS, BLOCK_SIZE, 100);
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
    inner.install_slot(request_id, slot);

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
    let decode = DecodeDisaggLeader::from_parts(
        inner.clone(),
        &cfg,
        coordinator,
        transport,
        workers,
        tokio::runtime::Handle::current(),
        None,
        None,
        None,
    );

    DecodeRig {
        inner,
        decode,
        queue,
    }
}

/// Drive the synchronous portion of decode's CD flow up through GNMT.
/// Both the baseline and the under-test run go through this same
/// driver — only the `api` arg differs.
fn drive_create_then_gnmt(
    api: &dyn ConnectorLeaderApi,
    request_id: &str,
) -> Result<(Option<usize>, bool)> {
    api.create_slot(make_request(request_id))?;
    api.get_num_new_matched_tokens(request_id, COMPUTED_BLOCKS * BLOCK_SIZE)
}

/// Settle pending audits from any tasks the synchronous workload
/// kicked off (e.g. the decode coordinator's enqueue spawn).
async fn settle_audits(handle: &AuditCaptureHandle, expected_queue: &Arc<InMemoryRemotePrefillQueue>) {
    // Wait for the spawned enqueue task to land its event.  Mirrors
    // `cd_decode_e2e`'s `wait_until` pattern but bounded shorter.
    for _ in 0..200 {
        if expected_queue.snapshot().len() >= 1 {
            // small spin so the post-enqueue audit (if any) lands too
            tokio::time::sleep(std::time::Duration::from_millis(2)).await;
            break;
        }
        tokio::time::sleep(std::time::Duration::from_millis(2)).await;
    }
    // No-op consume to ensure handle is touched.
    let _ = handle.snapshot();
}

/// Sequence-equivalence on the synchronous create_slot + GNMT path.
///
/// The baseline run drives `DecodeDisaggLeader` directly; the under-
/// test run drives `UnifiedDisaggLeader` wrapping a fresh
/// `DecodeDisaggLeader`.  The two leaders cannot be the same instance
/// (their internal cd_request_state would be polluted from the first
/// run) — but the *audit emissions* should match because both runs
/// take the same code path through the decode leader.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn decode_create_and_gnmt_audit_streams_match() -> Result<()> {
    let _audit_guard = audit_test_lock();
    let handle = install_collector();

    // ---- Baseline run: bare DecodeDisaggLeader ----
    let baseline_rig = build_decode_rig("req-1");
    handle.drain(); // throw away anything emitted during construction
    let baseline_result = drive_create_then_gnmt(baseline_rig.decode.as_ref(), "req-1")?;
    settle_audits(&handle, &baseline_rig.queue).await;
    let baseline_events = handle.drain();

    // ---- Under-test run: UnifiedDisaggLeader wrapping DecodeDisaggLeader ----
    let unified_rig = build_decode_rig("req-1");
    let unified = UnifiedDisaggLeader::builder(unified_rig.inner.clone())
        .with_decode(unified_rig.decode.clone())
        .build()
        .expect("build unified leader");
    handle.drain();
    let unified_result = drive_create_then_gnmt(unified.as_ref(), "req-1")?;
    settle_audits(&handle, &unified_rig.queue).await;
    let unified_events = handle.drain();

    // Functional results match.
    assert_eq!(
        baseline_result, unified_result,
        "baseline vs unified GNMT result mismatch"
    );
    assert_eq!(baseline_rig.queue.snapshot().len(), 1);
    assert_eq!(unified_rig.queue.snapshot().len(), 1);

    // Audit signature sequences match.
    assert_event_signatures_equal(
        &baseline_events,
        &unified_events,
        "decode create_slot + GNMT (sync portion)",
    );

    // Sanity: ensure we actually captured something.  If the audit
    // collector were broken we'd silently pass.
    assert!(
        !baseline_events.is_empty(),
        "baseline captured zero audit events — collector likely broken"
    );
    assert!(
        baseline_events.iter().any(|e| e.event == "create_slot"),
        "expected create_slot audit in baseline; got: {:?}",
        baseline_events
            .iter()
            .map(|e| &e.event)
            .collect::<Vec<_>>()
    );
    assert!(
        baseline_events.iter().any(|e| e.event == "gnmt_entry"),
        "expected gnmt_entry audit in baseline"
    );
    assert!(
        baseline_events
            .iter()
            .any(|e| e.event == "remote_prefill_queued"),
        "expected remote_prefill_queued audit in baseline"
    );

    Ok(())
}

/// Sanity check the audit collector itself: emitting an audit event
/// directly captures it; non-`kvbm_audit` events are filtered out.
#[tokio::test(flavor = "multi_thread", worker_threads = 1)]
async fn audit_collector_captures_only_kvbm_audit_target() {
    let _audit_guard = audit_test_lock();
    let handle = install_collector();
    handle.drain();

    kvbm_connector::audit!(
        "test_marker",
        role = "decode",
        request_id = "req-sanity",
        n = 7
    );
    tracing::info!(target: "some_other_target", "should not be captured");

    // Brief yield so the event lands.
    tokio::task::yield_now().await;
    let captured = handle.drain();
    assert_eq!(
        captured.len(),
        1,
        "expected exactly one captured event, got: {captured:?}"
    );
    assert_eq!(captured[0].event, "test_marker");
    assert_eq!(captured[0].role.as_deref(), Some("decode"));
    assert_eq!(captured[0].request_id.as_deref(), Some("req-sanity"));
    assert!(
        captured[0]
            .fields
            .iter()
            .any(|(k, v)| k == "n" && v == "7"),
        "expected n=7 in fields, got: {:?}",
        captured[0].fields
    );
}
