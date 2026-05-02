// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Prefill-side trace-equivalence tests for `UnifiedDisaggLeader`.
//!
//! Symmetric counterpart to `cd_unified_audit_equiv.rs`'s decode
//! tests.  For each scenario we run a fixed synchronous workload
//! twice — once through bare `PrefillDisaggLeader`, once through
//! `UnifiedDisaggLeader::with_prefill(prefill, coord)` — and assert
//! the captured `(event, role, request_id)` audit-signature
//! sequences match.
//!
//! ### Async work caveat
//!
//! `PrefillDisaggLeader::get_num_new_matched_tokens` for a CD-bound
//! request spawns `PrefillCoordinatorImpl::run_setup` on the tokio
//! runtime.  That task contains only `tracing::info!` calls (no
//! `audit!`), so it does not pollute the audit stream.  The
//! lifecycle watcher *can* emit async audits (`prefill_lifecycle_*`)
//! on terminal events, but those don't fire unless we trigger
//! detach / failure / watchdog — none of which the equivalence
//! workloads do.  As a defense-in-depth we filter any
//! `prefill_lifecycle_*` events out of both captured streams before
//! comparing.

mod audit_helpers;

use std::sync::Arc;

use anyhow::Result;
use kvbm_connector::G2;
use kvbm_connector::common::Request;
use kvbm_connector::connector::leader::disagg::prefill_coordinator::PrefillCoordinatorImpl;
use kvbm_connector::connector::leader::disagg::testing::{
    MockCdBlockTransport, MockCdWorkerHook, MockInnerLeaderShim, MockSlot, TEST_BLOCK_SIZE,
};
use kvbm_connector::connector::leader::disagg::{
    ConnectorLeaderApi, PrefillDisaggLeader, UnifiedDisaggLeader,
};
use kvbm_disagg_protocol::{
    DISAGG_PROTOCOL_VERSION, RemotePrefillParams, SessionEndpoint, SessionId, TransferParams,
};
use kvbm_engine::disagg::session::MockSessionFactory;
use kvbm_engine::testing::managers::{TestManagerBuilder, TestRegistryBuilder};
use kvbm_engine::testing::token_blocks::{create_token_sequence, generate_sequence_hashes};
use kvbm_logical::manager::BlockManager;

use audit_helpers::{
    AuditEvent, assert_event_signatures_equal, audit_test_lock, filter_out_event_prefixes,
    install_collector,
};

const TOTAL_BLOCKS: usize = 4;
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

/// One run's collaborators — fresh per run so coordinator state
/// doesn't bleed across runs (the audit collector is process-wide
/// but `audit_test_lock` serializes us against other tests).
struct PrefillRig {
    prefill: Arc<PrefillDisaggLeader>,
    coordinator: Arc<PrefillCoordinatorImpl>,
    inner: Arc<MockInnerLeaderShim>,
}

/// Build a prefill rig.  `with_cd_params=true` installs a slot whose
/// `transfer_params` carries `RemotePrefillParams` (CD-bound path);
/// `false` leaves it as None (non-CD passthrough).  `expected_hashes`
/// is non-empty only when we want `ensure_started` to return n>0; the
/// equivalence tests stick with empty hashes for the CD-bound case so
/// the `ensure_started_zero_passthrough` synchronous chain runs.
fn build_prefill_rig(with_cd_params: bool) -> PrefillRig {
    let g2_manager = build_g2_manager(64);

    let token_sequence = create_token_sequence(TOTAL_BLOCKS, BLOCK_SIZE, 100);
    let all_hashes = generate_sequence_hashes(&token_sequence);
    let token_blocks: Vec<_> = token_sequence.blocks().to_vec();

    let inner = MockInnerLeaderShim::new(BLOCK_SIZE, g2_manager.clone());

    let session_id = uuid::Uuid::new_v4();
    let decode_instance_id: kvbm_connector::InstanceId = uuid::Uuid::new_v4().into();
    let transfer_params = if with_cd_params {
        Some(cd_transfer_params(
            session_id,
            decode_instance_id,
            // empty hashes ⇒ ensure_started returns 0 ⇒ leader takes
            // the synchronous "ensure_started_zero_passthrough" branch
            Vec::new(),
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
        transport,
        workers.clone(),
        factory,
        Arc::new(kvbm_connector::connector::leader::disagg::peer_resolver::NoopPeerResolver),
        tokio::runtime::Handle::current(),
    );
    let prefill = PrefillDisaggLeader::from_parts(inner.clone(), coordinator.clone(), workers);

    PrefillRig {
        prefill,
        coordinator,
        inner,
    }
}

/// Run the synchronous portion of prefill: create_slot + GNMT.
fn drive_create_then_gnmt(api: &dyn ConnectorLeaderApi) -> Result<(Option<usize>, bool)> {
    api.create_slot(make_request())?;
    api.get_num_new_matched_tokens("req-1", 0)
}

/// Filter any `prefill_lifecycle_*` events the spawned watcher might
/// emit between the sync call returning and our drain.  In normal
/// equivalence runs none should fire, but defense-in-depth keeps the
/// assertion stable across CI scheduling jitter.
fn strip_async_lifecycle(events: Vec<AuditEvent>) -> Vec<AuditEvent> {
    filter_out_event_prefixes(events, &["prefill_lifecycle_"])
}

// ============================================================================
// Tests
// ============================================================================

/// Prefill non-CD passthrough: a slot without `transfer_params` runs
/// the synchronous `gnmt_passthrough_non_cd` audit chain.  No async
/// work is spawned.  The two audit streams must match exactly.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn prefill_non_cd_passthrough_audit_streams_match() -> Result<()> {
    let _audit_guard = audit_test_lock();
    let handle = install_collector();

    // Baseline: bare PrefillDisaggLeader.
    let baseline = build_prefill_rig(false);
    handle.drain();
    let baseline_result = drive_create_then_gnmt(baseline.prefill.as_ref())?;
    let baseline_events = strip_async_lifecycle(handle.drain());

    // Under-test: UnifiedDisaggLeader wrapping fresh PrefillDisaggLeader
    // + coordinator (independent state from baseline).
    let unified_rig = build_prefill_rig(false);
    let unified = UnifiedDisaggLeader::builder(unified_rig.inner.clone())
        .with_prefill(unified_rig.prefill.clone(), unified_rig.coordinator.clone())
        .build()
        .expect("build unified leader");
    handle.drain();
    let unified_result = drive_create_then_gnmt(unified.as_ref())?;
    let unified_events = strip_async_lifecycle(handle.drain());

    // Functional equivalence.
    assert_eq!(
        baseline_result, unified_result,
        "baseline vs unified GNMT result mismatch"
    );

    // Audit signature sequences match.
    assert_event_signatures_equal(
        &baseline_events,
        &unified_events,
        "prefill non-CD passthrough (sync portion)",
    );

    // Sanity: expected events present in baseline.
    assert!(
        !baseline_events.is_empty(),
        "baseline captured zero audits — collector likely broken"
    );
    assert!(
        baseline_events.iter().any(|e| e.event == "create_slot"),
        "expected create_slot audit"
    );
    assert!(
        baseline_events
            .iter()
            .any(|e| e.event == "gnmt_passthrough_non_cd"),
        "expected gnmt_passthrough_non_cd audit (the non-CD route signal)"
    );

    Ok(())
}

/// Prefill CD-bound n=0 passthrough: slot has
/// `RemotePrefillParams::sequence_hashes = []`, so `ensure_started`
/// returns 0 and the leader takes the
/// `ensure_started_zero_passthrough` synchronous branch.  A run_setup
/// task is spawned but contains only `tracing::info!` (no `audit!`),
/// so the captured stream is stable.  Async lifecycle audits are
/// filtered out as defense-in-depth.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn prefill_cd_n_zero_passthrough_audit_streams_match() -> Result<()> {
    let _audit_guard = audit_test_lock();
    let handle = install_collector();

    let baseline = build_prefill_rig(true);
    handle.drain();
    let baseline_result = drive_create_then_gnmt(baseline.prefill.as_ref())?;
    let baseline_events = strip_async_lifecycle(handle.drain());
    // Hold the rig until events are drained so the spawned setup
    // task's tracing scope is alive when audits are visited.
    let _ = baseline.coordinator.active_count();

    let unified_rig = build_prefill_rig(true);
    let unified = UnifiedDisaggLeader::builder(unified_rig.inner.clone())
        .with_prefill(unified_rig.prefill.clone(), unified_rig.coordinator.clone())
        .build()
        .expect("build unified leader");
    handle.drain();
    let unified_result = drive_create_then_gnmt(unified.as_ref())?;
    let unified_events = strip_async_lifecycle(handle.drain());
    let _ = unified_rig.coordinator.active_count();

    assert_eq!(
        baseline_result, unified_result,
        "baseline vs unified GNMT result mismatch (CD n=0 path)"
    );

    assert_event_signatures_equal(
        &baseline_events,
        &unified_events,
        "prefill CD-bound n=0 passthrough (sync portion)",
    );

    // Sanity: the CD-bound branch hit `cd_bound_ensure_started` and
    // `ensure_started_zero_passthrough`.
    assert!(
        baseline_events
            .iter()
            .any(|e| e.event == "cd_bound_ensure_started"),
        "expected cd_bound_ensure_started audit"
    );
    assert!(
        baseline_events
            .iter()
            .any(|e| e.event == "ensure_started_zero_passthrough"),
        "expected ensure_started_zero_passthrough audit"
    );
    assert!(
        baseline_events
            .iter()
            .any(|e| e.event == "session_setup_spawned"),
        "expected session_setup_spawned audit (emitted on sync thread)"
    );

    Ok(())
}
