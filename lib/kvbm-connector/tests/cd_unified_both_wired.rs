// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Both-flows-wired tests for `UnifiedDisaggLeader`.
//!
//! These tests assert the *new* tick-method semantics added in Step 4:
//! `build_connector_meta`, `update_connector_output`, and
//! `create_slot` must call `inner` exactly once and run each wired
//! flow's audit/observe decorator on top — never short-circuit to one
//! flow.
//!
//! They also pin the per-request dispatch under both-wired:
//! `TransferParams::remote_prefill = Some(..)` routes exclusively to
//! the prefill flow (no decode-side audits leak); absence routes
//! exclusively to the decode flow.

mod audit_helpers;

use std::collections::HashSet;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use anyhow::Result;
use futures::{FutureExt, future::BoxFuture};
use kvbm_config::{DisaggConfig, DisaggregationRole};
use kvbm_connector::G2;
use kvbm_connector::common::{Request, SchedulerOutput};
use kvbm_connector::connector::leader::FinishedStatus;
use kvbm_connector::connector::leader::SlotMatchSplit;
use kvbm_connector::connector::leader::scheduler::KvConnectorMetadata;
use kvbm_connector::connector::leader::disagg::testing::{
    InMemoryRemotePrefillQueue, MockCdBlockTransport, MockCdWorkerHook, MockInnerLeaderShim,
    MockSlot, TEST_BLOCK_SIZE,
};
use kvbm_connector::connector::leader::disagg::{
    AlwaysRemote, ConnectorLeaderApi, DecodeDisaggLeader, InnerLeaderShim, PrefillCoordinator,
    PrefillDisaggLeader, RemotePrefillCoordinator, UnifiedDisaggLeader,
};
use kvbm_disagg_protocol::{RemotePrefillParams, TransferParams};
use kvbm_engine::disagg::session::MockSessionFactory;
use kvbm_engine::testing::managers::{TestManagerBuilder, TestRegistryBuilder};
use kvbm_logical::blocks::{CompleteBlock, ImmutableBlock, MutableBlock};
use kvbm_logical::manager::BlockManager;
use parking_lot::Mutex;

use audit_helpers::{AuditEvent, audit_test_lock, install_collector};

const BLOCK_SIZE: usize = TEST_BLOCK_SIZE;

// ============================================================================
// Mocks
// ============================================================================

/// `MockPrefillCoordinator` records each trait call.  The unified
/// leader's tick path consults `observe_forward`; per-request dispatch
/// consults `has_active_request`, `ensure_started`, `on_usaa`,
/// `on_request_finished`.
#[derive(Default)]
struct MockPrefillCoordinator {
    has_active: Mutex<HashSet<String>>,
    ensure_started_calls: AtomicUsize,
    observe_forward_calls: AtomicUsize,
    on_usaa_calls: AtomicUsize,
    on_request_finished_calls: AtomicUsize,
}

impl MockPrefillCoordinator {
    fn new() -> Arc<Self> {
        Arc::new(Self::default())
    }

    #[allow(dead_code)]
    fn mark_active(&self, request_id: &str) {
        self.has_active.lock().insert(request_id.to_string());
    }
}

impl PrefillCoordinator for MockPrefillCoordinator {
    fn has_active_request(&self, request_id: &str) -> bool {
        self.has_active.lock().contains(request_id)
    }

    fn ensure_started(
        &self,
        _request_id: &str,
        _params: &kvbm_disagg_protocol::RemotePrefillParams,
    ) -> Result<usize> {
        self.ensure_started_calls.fetch_add(1, Ordering::Relaxed);
        Ok(0)
    }

    fn on_usaa(
        &self,
        _request_id: &str,
        _block_ids: &[usize],
        _num_external_tokens: usize,
    ) -> Result<()> {
        self.on_usaa_calls.fetch_add(1, Ordering::Relaxed);
        Ok(())
    }

    fn observe_forward(
        &self,
        _request_id: &str,
        _meta: &KvConnectorMetadata,
    ) -> Result<()> {
        self.observe_forward_calls.fetch_add(1, Ordering::Relaxed);
        Ok(())
    }

    fn on_request_finished(&self, _request_id: &str) {
        self.on_request_finished_calls
            .fetch_add(1, Ordering::Relaxed);
    }
}

/// Wraps a `MockInnerLeaderShim` to count inner-method invocations
/// and return a default-shaped `KvConnectorMetadata` from
/// `build_connector_meta` (the underlying mock errors there).
struct CountingInner {
    inner: Arc<MockInnerLeaderShim>,
    create_slot_calls: AtomicUsize,
    build_meta_calls: AtomicUsize,
    update_connector_output_calls: AtomicUsize,
}

impl CountingInner {
    fn new(inner: Arc<MockInnerLeaderShim>) -> Arc<Self> {
        Arc::new(Self {
            inner,
            create_slot_calls: AtomicUsize::new(0),
            build_meta_calls: AtomicUsize::new(0),
            update_connector_output_calls: AtomicUsize::new(0),
        })
    }
}

fn empty_meta() -> KvConnectorMetadata {
    KvConnectorMetadata {
        iteration: 0,
        foward_pass_completion_events: None,
        intra_pass_load: None,
        intra_pass_store: None,
    }
}

impl InnerLeaderShim for CountingInner {
    fn create_slot(&self, request: Request) -> Result<()> {
        self.create_slot_calls.fetch_add(1, Ordering::Relaxed);
        self.inner.create_slot(request)
    }

    fn has_slot(&self, request_id: &str) -> bool {
        self.inner.has_slot(request_id)
    }

    fn extend_slot_tokens(&self, request_id: &str, tokens: Vec<u32>) -> Result<()> {
        self.inner.extend_slot_tokens(request_id, tokens)
    }

    fn get_num_new_matched_tokens(
        &self,
        request_id: &str,
        num_computed_tokens: usize,
    ) -> Result<(Option<usize>, bool)> {
        self.inner
            .get_num_new_matched_tokens(request_id, num_computed_tokens)
    }

    fn update_state_after_alloc(
        &self,
        request_id: &str,
        block_ids: Vec<usize>,
        num_external_tokens: usize,
    ) -> Result<()> {
        self.inner
            .update_state_after_alloc(request_id, block_ids, num_external_tokens)
    }

    fn build_connector_meta(&self, _output: SchedulerOutput) -> Result<KvConnectorMetadata> {
        self.build_meta_calls.fetch_add(1, Ordering::Relaxed);
        Ok(empty_meta())
    }

    fn update_connector_output(
        &self,
        finished_sending: HashSet<String>,
        finished_recving: HashSet<String>,
    ) -> Result<()> {
        self.update_connector_output_calls
            .fetch_add(1, Ordering::Relaxed);
        self.inner
            .update_connector_output(finished_sending, finished_recving)
    }

    fn request_finished(&self, request_id: &str) -> FinishedStatus {
        self.inner.request_finished(request_id)
    }

    fn block_size(&self) -> usize {
        self.inner.block_size()
    }

    fn get_slot_total_tokens(&self, request_id: &str) -> Result<usize> {
        self.inner.get_slot_total_tokens(request_id)
    }

    fn slot_match_split(&self, request_id: &str) -> Result<SlotMatchSplit> {
        self.inner.slot_match_split(request_id)
    }

    fn slot_token_ids(&self, request_id: &str) -> Result<Vec<u32>> {
        self.inner.slot_token_ids(request_id)
    }

    fn local_instance_id(&self) -> kvbm_connector::InstanceId {
        self.inner.local_instance_id()
    }

    fn apply_block_assignments(&self, request_id: &str, block_ids: Vec<usize>) -> Result<()> {
        self.inner.apply_block_assignments(request_id, block_ids)
    }

    fn take_local_match_g2_blocks(&self, request_id: &str) -> Result<Vec<ImmutableBlock<G2>>> {
        self.inner.take_local_match_g2_blocks(request_id)
    }

    fn token_blocks_for_range(
        &self,
        request_id: &str,
        range: std::ops::Range<usize>,
    ) -> Result<Vec<dynamo_tokens::TokenBlock>> {
        self.inner.token_blocks_for_range(request_id, range)
    }

    fn slot_transfer_params(&self, request_id: &str) -> Result<Option<TransferParams>> {
        self.inner.slot_transfer_params(request_id)
    }

    fn allocate_g2_blocks(&self, count: usize) -> Result<Vec<MutableBlock<G2>>> {
        self.inner.allocate_g2_blocks(count)
    }

    fn register_g2_blocks(
        &self,
        blocks: Vec<CompleteBlock<G2>>,
    ) -> Result<Vec<ImmutableBlock<G2>>> {
        self.inner.register_g2_blocks(blocks)
    }

    fn install_cd_onboarding_payload(
        &self,
        request_id: &str,
        cd_payload: Box<dyn kvbm_connector::connector::leader::CdOnboardingPayload>,
    ) -> Result<()> {
        self.inner
            .install_cd_onboarding_payload(request_id, cd_payload)
    }
}

// ============================================================================
// Harness
// ============================================================================

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

struct BothWiredHarness {
    leader: Arc<UnifiedDisaggLeader>,
    counting_inner: Arc<CountingInner>,
    coord: Arc<MockPrefillCoordinator>,
    inner: Arc<MockInnerLeaderShim>,
}

fn build_both_wired() -> BothWiredHarness {
    let g2 = build_g2_manager(8);
    let inner = MockInnerLeaderShim::new(BLOCK_SIZE, g2.clone());
    let counting = CountingInner::new(inner.clone());
    let counting_dyn: Arc<dyn InnerLeaderShim> = counting.clone();

    // Decode flow.
    let factory = MockSessionFactory::new();
    let queue = InMemoryRemotePrefillQueue::new();
    let decode_coord = RemotePrefillCoordinator::new(
        Arc::new(AlwaysRemote),
        factory.clone(),
        queue,
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
        counting_dyn.clone(),
        &cfg,
        decode_coord,
        transport.clone(),
        workers.clone(),
        tokio::runtime::Handle::current(),
        None,
        None,
        None,
    );

    // Prefill flow.
    let mock_coord = MockPrefillCoordinator::new();
    let prefill = PrefillDisaggLeader::from_parts(
        counting_dyn.clone(),
        mock_coord.clone(),
        workers.clone(),
    );

    let leader = UnifiedDisaggLeader::builder(counting_dyn)
        .with_decode(decode)
        .with_prefill(prefill, mock_coord.clone())
        .build()
        .expect("build unified leader");

    BothWiredHarness {
        leader,
        counting_inner: counting,
        coord: mock_coord,
        inner,
    }
}

fn role_count(events: &[AuditEvent], event_name: &str, role: &str) -> usize {
    events
        .iter()
        .filter(|e| e.event == event_name && e.role.as_deref() == Some(role))
        .count()
}

// ============================================================================
// Tests
// ============================================================================

/// `build_connector_meta` with both flows wired must:
/// - emit `build_meta_entry` with role=decode AND role=prefill
/// - call `inner.build_connector_meta` exactly once
/// - call `prefill_coordinator.observe_forward` exactly once
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn both_wired_build_connector_meta_decorates_single_inner_call() -> Result<()> {
    let _audit_guard = audit_test_lock();
    let handle = install_collector();
    let h = build_both_wired();
    handle.drain();

    // Iteration % 20 == 0 → audit_build_meta emits even with no
    // requests scheduled.
    let output = SchedulerOutput {
        iteration: 0,
        ..SchedulerOutput::default()
    };
    let _meta = h.leader.build_connector_meta(output)?;

    let events = handle.drain();
    assert_eq!(
        h.counting_inner.build_meta_calls.load(Ordering::Relaxed),
        1,
        "inner.build_connector_meta should be called exactly once"
    );
    assert_eq!(
        h.coord.observe_forward_calls.load(Ordering::Relaxed),
        1,
        "prefill coordinator's observe_forward should be called exactly once"
    );
    assert_eq!(
        role_count(&events, "build_meta_entry", "decode"),
        1,
        "expected 1 decode-role build_meta_entry; got events: {events:#?}"
    );
    assert_eq!(
        role_count(&events, "build_meta_entry", "prefill"),
        1,
        "expected 1 prefill-role build_meta_entry; got events: {events:#?}"
    );
    Ok(())
}

/// `update_connector_output` with both flows wired must:
/// - emit per-rid `uco_*` audits for both roles
/// - decode-role audits carry `cd_tracked` field; prefill-role do not
/// - call `inner.update_connector_output` exactly once
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn both_wired_update_connector_output_decorates_single_inner_call() -> Result<()> {
    let _audit_guard = audit_test_lock();
    let handle = install_collector();
    let h = build_both_wired();
    handle.drain();

    let mut sending = HashSet::new();
    sending.insert("rid-A".to_string());
    let mut recving = HashSet::new();
    recving.insert("rid-B".to_string());

    h.leader.update_connector_output(sending, recving)?;

    let events = handle.drain();
    assert_eq!(
        h.counting_inner
            .update_connector_output_calls
            .load(Ordering::Relaxed),
        1,
        "inner.update_connector_output should be called exactly once"
    );
    assert_eq!(role_count(&events, "uco_finished_sending", "decode"), 1);
    assert_eq!(role_count(&events, "uco_finished_sending", "prefill"), 1);
    assert_eq!(role_count(&events, "uco_finished_recving", "decode"), 1);
    assert_eq!(role_count(&events, "uco_finished_recving", "prefill"), 1);

    // Decode-role uco_* audits must carry `cd_tracked`; prefill-role
    // must not.  This pins the field-shape divergence between the two
    // role tags so future refactors don't accidentally homogenize it
    // (which would silently break log-pipeline parsers downstream).
    for e in events
        .iter()
        .filter(|e| e.event.starts_with("uco_") && e.role.as_deref() == Some("decode"))
    {
        assert!(
            e.fields.iter().any(|(k, _)| k == "cd_tracked"),
            "decode-role uco audit missing cd_tracked field: {e:?}"
        );
    }
    for e in events
        .iter()
        .filter(|e| e.event.starts_with("uco_") && e.role.as_deref() == Some("prefill"))
    {
        assert!(
            !e.fields.iter().any(|(k, _)| k == "cd_tracked"),
            "prefill-role uco audit must NOT carry cd_tracked: {e:?}"
        );
    }
    Ok(())
}

/// `create_slot` with both flows wired must emit one create_slot
/// audit per role and call inner exactly once.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn both_wired_create_slot_decorates_single_inner_call() -> Result<()> {
    let _audit_guard = audit_test_lock();
    let handle = install_collector();
    let h = build_both_wired();
    handle.drain();

    let request = Request::builder()
        .request_id("req-cs".to_string())
        .tokens(dynamo_tokens::Tokens::from(Vec::<u32>::new()))
        .build(None)
        .expect("build request");
    h.leader.create_slot(request)?;

    let events = handle.drain();
    assert_eq!(
        h.counting_inner.create_slot_calls.load(Ordering::Relaxed),
        1,
        "inner.create_slot should be called exactly once"
    );
    assert_eq!(role_count(&events, "create_slot", "decode"), 1);
    assert_eq!(role_count(&events, "create_slot", "prefill"), 1);
    Ok(())
}

/// Per-request dispatch under both-wired:
///
/// A slot whose `transfer_params` carries `RemotePrefillParams`
/// classifies as Prefill — GNMT routes through the prefill leader's
/// `gnmt_entry` audit (role=prefill) and never the decode leader's
/// (role=decode).
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn both_wired_remote_prefill_request_routes_only_to_prefill_flow() -> Result<()> {
    let _audit_guard = audit_test_lock();
    let handle = install_collector();
    let h = build_both_wired();

    h.inner.install_slot(
        "rid-prefill",
        MockSlot {
            block_size: BLOCK_SIZE,
            gnmt_result: (Some(0), false),
            transfer_params: Some(TransferParams::remote_prefill(RemotePrefillParams::new(
                uuid::Uuid::new_v4(),
                uuid::Uuid::new_v4().into(),
            ))),
            ..MockSlot::default()
        },
    );
    handle.drain();

    let _ = h.leader.get_num_new_matched_tokens("rid-prefill", 0)?;
    let events = handle.drain();

    assert!(
        role_count(&events, "gnmt_entry", "prefill") >= 1,
        "expected at least one prefill-role gnmt_entry; got: {events:#?}"
    );
    assert_eq!(
        role_count(&events, "gnmt_entry", "decode"),
        0,
        "decode-role gnmt_entry must NOT fire for a prefill-classified request: {events:#?}"
    );
    Ok(())
}

/// Per-request dispatch under both-wired:
///
/// A slot without `transfer_params` classifies as Decode — GNMT
/// routes through the decode leader and never the prefill leader.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn both_wired_no_params_request_routes_only_to_decode_flow() -> Result<()> {
    let _audit_guard = audit_test_lock();
    let handle = install_collector();
    let h = build_both_wired();

    h.inner.install_slot(
        "rid-decode",
        MockSlot {
            block_size: BLOCK_SIZE,
            // gnmt_result returns None so decode_gnmt's
            // `inner_result` short-circuits to passthrough — keeps
            // the test focused on dispatch routing without dragging
            // in policy / queue setup.
            gnmt_result: (None, false),
            transfer_params: None,
            ..MockSlot::default()
        },
    );
    handle.drain();

    let _ = h.leader.get_num_new_matched_tokens("rid-decode", 0)?;
    let events = handle.drain();

    assert!(
        role_count(&events, "gnmt_entry", "decode") >= 1,
        "expected at least one decode-role gnmt_entry; got: {events:#?}"
    );
    assert_eq!(
        role_count(&events, "gnmt_entry", "prefill"),
        0,
        "prefill-role gnmt_entry must NOT fire for a decode-classified request: {events:#?}"
    );
    Ok(())
}

/// Multi-request smoke: drive N sequential GNMTs through the unified
/// leader (alternating decode-only and prefill-only classifications)
/// and assert each request's `gnmt_entry` fires under exactly one
/// role tag.  This proves that classify state isn't leaking between
/// requests and that the dispatch is per-call, not cached.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn both_wired_alternating_requests_route_independently() -> Result<()> {
    let _audit_guard = audit_test_lock();
    let handle = install_collector();
    let h = build_both_wired();

    const N: usize = 6;
    for i in 0..N {
        let rid = format!("rid-{i}");
        let slot = if i % 2 == 0 {
            MockSlot {
                block_size: BLOCK_SIZE,
                gnmt_result: (None, false),
                transfer_params: None,
                ..MockSlot::default()
            }
        } else {
            MockSlot {
                block_size: BLOCK_SIZE,
                gnmt_result: (Some(0), false),
                transfer_params: Some(TransferParams::remote_prefill(
                    RemotePrefillParams::new(
                        uuid::Uuid::new_v4(),
                        uuid::Uuid::new_v4().into(),
                    ),
                )),
                ..MockSlot::default()
            }
        };
        h.inner.install_slot(&rid, slot);
    }
    handle.drain();

    for i in 0..N {
        let rid = format!("rid-{i}");
        let _ = h.leader.get_num_new_matched_tokens(&rid, 0)?;
    }
    let events = handle.drain();

    for i in 0..N {
        let rid = format!("rid-{i}");
        let entries: Vec<_> = events
            .iter()
            .filter(|e| e.event == "gnmt_entry" && e.request_id.as_deref() == Some(&rid))
            .collect();
        assert!(
            !entries.is_empty(),
            "no gnmt_entry seen for {rid}"
        );
        let role = entries[0].role.as_deref().unwrap_or("-");
        if i % 2 == 0 {
            assert_eq!(role, "decode", "rid={rid} expected decode role");
        } else {
            assert_eq!(role, "prefill", "rid={rid} expected prefill role");
        }
        // No cross-role contamination per request.
        let cross = events
            .iter()
            .filter(|e| {
                e.event == "gnmt_entry"
                    && e.request_id.as_deref() == Some(&rid)
                    && e.role.as_deref() != Some(role)
            })
            .count();
        assert_eq!(
            cross, 0,
            "rid={rid} saw cross-role gnmt_entry: {entries:#?}"
        );
    }
    Ok(())
}

// Suppress unused import warnings on items only used in trait
// signatures via `as_ref()` / `&BoxFuture` — kept here to keep this
// file self-contained against future signature changes.
#[allow(dead_code)]
fn _unused_imports_anchor() -> BoxFuture<'static, Result<()>> {
    async { Ok(()) }.boxed()
}
