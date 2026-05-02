// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Bidirectional conditional-disagg end-to-end test.
//!
//! Proves that a single instance can hold BOTH a `RemotePrefillCoordinator`
//! (decode role) AND a `PrefillCoordinatorImpl` (prefill role) concurrently,
//! driven against the same `MockInnerLeaderShim` / `MockCdBlockTransport` /
//! `MockCdWorkerHook`.
//!
//! This is the empirical foundation for the R-B refactor: the rest of the
//! refactor assumes this dual-role composition works correctly before
//! collapsing the two wrappers into a single `UnifiedDisaggLeader`.
//!
//! # Instance layout
//!
//! Two instances, A and B, each wired with:
//!   - one `DecodeDisaggLeader` (owns `RemotePrefillCoordinator`)
//!   - one `PrefillDisaggLeader` (owns `PrefillCoordinatorImpl`)
//!   - shared `MockInnerLeaderShim`, `MockCdBlockTransport`, `MockCdWorkerHook`
//!   - a `MockSessionFactory` paired via `make_paired()` with the other instance
//!
//! # Scenario 1 — sequential, same instance
//!
//! Request X: A is decode, B is prefill. Drive to completion.
//! Request Y: A is prefill, B is decode. Drive to completion.
//! Assert both requests completed on the pull side, no failures, all
//! coordinators idle after cleanup.
//!
//! # Scenario 2 — concurrent, same instance
//!
//! X and Y are both in flight simultaneously. Session frames for X and Y
//! are injected in interleaved order; transports and pulls are resolved in
//! alternating order. Both must complete cleanly.
//!
//! # Inflight-budget finding
//!
//! The inflight budget is a field of `DecodeDisaggLeader` only, consumed at
//! `decode_leader.rs:418` via `self.inflight_budget.try_reserve(...)`.
//! `PrefillCoordinatorImpl` (`prefill_coordinator.rs`) has no budget field.
//! Therefore: when instance A acts as *prefill* for request Y, it does NOT
//! consume from A's decode-side inflight budget. The two roles' budgets are
//! fully independent by construction.
//!
//! # Token layout (per direction)
//!
//! ```text
//! COMPUTED=0, LOCAL=2, REMOTE=2, TOTAL=4 blocks of 16 tokens each
//! ```
//!
//! Critical: prefill's output blocks must be built from the SAME token_blocks
//! slice [COMPUTED+LOCAL..TOTAL] as the decode side's original sequence so
//! the sequence hashes match `state.remote_slot_index` in `run_remote_pipeline`
//! (see `decode_leader.rs:902-907`).

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
    PrefillDisaggLeader,
};
use kvbm_disagg_protocol::{
    DISAGG_PROTOCOL_VERSION, RemotePrefillParams, SessionEndpoint, SessionId, TransferParams,
};
use kvbm_engine::disagg::session::MockSessionFactory;
use kvbm_engine::testing::managers::{TestManagerBuilder, TestRegistryBuilder};
use kvbm_engine::testing::token_blocks::{create_token_sequence, generate_sequence_hashes};
use kvbm_logical::manager::BlockManager;

const COMPUTED_BLOCKS: usize = 0;
const LOCAL_BLOCKS: usize = 2;
const REMOTE_BLOCKS: usize = 2;
const TOTAL_BLOCKS: usize = COMPUTED_BLOCKS + LOCAL_BLOCKS + REMOTE_BLOCKS;
const BLOCK_SIZE: usize = TEST_BLOCK_SIZE;

// ============================================================================
// Helpers
// ============================================================================

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

fn make_disagg_config() -> DisaggConfig {
    DisaggConfig {
        hub_url: "http://127.0.0.1:1337".to_string(),
        role: DisaggregationRole::Decode,
        max_inflight_remote_prefill_tokens: usize::MAX,
    }
}

/// One instance: carries both a decode-side `DecodeDisaggLeader` and a
/// prefill-side `PrefillDisaggLeader`, wired through shared mocks.
struct DualRoleInstance {
    decode_wrapper: Arc<DecodeDisaggLeader>,
    prefill_wrapper: Arc<PrefillDisaggLeader>,
    decode_coord: Arc<ConditionalDisaggCoordinator>,
    prefill_coord: Arc<ConditionalDisaggCoordinator>,
    inner: Arc<MockInnerLeaderShim>,
    transport: Arc<MockCdBlockTransport>,
    workers: Arc<MockCdWorkerHook>,
    queue: Arc<InMemoryRemotePrefillQueue>,
    g2_manager: Arc<BlockManager<G2>>,
}

fn build_instance(factory: Arc<MockSessionFactory>) -> DualRoleInstance {
    let g2_manager = build_g2_manager(128);
    let inner = MockInnerLeaderShim::new(BLOCK_SIZE, g2_manager.clone());
    let transport = MockCdBlockTransport::new();
    let workers = MockCdWorkerHook::new();
    let queue = InMemoryRemotePrefillQueue::new();

    let decode_coord = ConditionalDisaggCoordinator::new_with_decode(
        inner.clone(),
        transport.clone(),
        workers.clone(),
        factory.clone(),
        Arc::new(kvbm_connector::connector::leader::disagg::peer_resolver::NoopPeerResolver),
        tokio::runtime::Handle::current(),
        Arc::new(AlwaysRemote),
        queue.clone(),
    );

    let decode_wrapper = DecodeDisaggLeader::from_parts(
        inner.clone(),
        &make_disagg_config(),
        decode_coord.clone(),
        transport.clone(),
        workers.clone(),
        tokio::runtime::Handle::current(),
        None,
        None,
        None,
    );

    let prefill_coord = ConditionalDisaggCoordinator::new(
        inner.clone(),
        transport.clone(),
        workers.clone(),
        factory.clone(),
        Arc::new(kvbm_connector::connector::leader::disagg::peer_resolver::NoopPeerResolver),
        tokio::runtime::Handle::current(),
    );

    let prefill_wrapper =
        PrefillDisaggLeader::from_parts(inner.clone(), prefill_coord.clone(), workers.clone());

    DualRoleInstance {
        decode_wrapper,
        prefill_wrapper,
        decode_coord,
        prefill_coord,
        inner,
        transport,
        workers,
        queue,
        g2_manager,
    }
}

/// Data for one request direction: token sequence + derived hashes.
///
/// Decode installs a slot with `all_hashes` over the full TOTAL_BLOCKS
/// range. Prefill needs the same `token_blocks[COMPUTED+LOCAL..]` to
/// build output blocks whose hashes match decode's `remote_slot_index`.
struct RequestSeq {
    all_hashes: Vec<kvbm_logical::SequenceHash>,
    token_blocks: Vec<dynamo_tokens::TokenBlock>,
}

impl RequestSeq {
    fn new(seed: u32) -> Self {
        let seq = create_token_sequence(TOTAL_BLOCKS, BLOCK_SIZE, seed);
        let all_hashes = generate_sequence_hashes(&seq);
        let token_blocks = seq.blocks().to_vec();
        Self {
            all_hashes,
            token_blocks,
        }
    }

    fn local_hashes(&self) -> Vec<kvbm_logical::SequenceHash> {
        self.all_hashes[COMPUTED_BLOCKS..COMPUTED_BLOCKS + LOCAL_BLOCKS].to_vec()
    }

    fn remote_hashes(&self) -> Vec<kvbm_logical::SequenceHash> {
        self.all_hashes[COMPUTED_BLOCKS + LOCAL_BLOCKS..].to_vec()
    }

    fn remote_token_blocks(&self) -> &[dynamo_tokens::TokenBlock] {
        &self.token_blocks[COMPUTED_BLOCKS + LOCAL_BLOCKS..]
    }

    fn local_token_blocks(&self) -> &[dynamo_tokens::TokenBlock] {
        &self.token_blocks[COMPUTED_BLOCKS..COMPUTED_BLOCKS + LOCAL_BLOCKS]
    }
}

/// Install a decode-flavor slot. Returns the pre-registered local-match G2
/// blocks (so they're available for inspection if needed).
fn install_decode_slot(
    inst: &DualRoleInstance,
    request_id: &str,
    seq: &RequestSeq,
) {
    let mutables = inst
        .g2_manager
        .allocate_blocks(LOCAL_BLOCKS)
        .expect("alloc local-match G2");
    let completes: Vec<_> = mutables
        .into_iter()
        .zip(seq.local_token_blocks().iter())
        .map(|(m, tb)| m.complete(tb).expect("complete local"))
        .collect();
    let local_match_g2 = inst.g2_manager.register_blocks(completes);

    inst.inner.install_slot(
        request_id,
        MockSlot {
            block_size: BLOCK_SIZE,
            total_blocks: TOTAL_BLOCKS,
            computed_blocks: COMPUTED_BLOCKS,
            local_match_blocks: LOCAL_BLOCKS,
            all_hashes: seq.all_hashes.clone(),
            token_blocks: seq.token_blocks.clone(),
            local_match_g2: parking_lot::Mutex::new(Some(local_match_g2)),
            assigned_block_ids: parking_lot::Mutex::new(None),
            gnmt_result: (Some(LOCAL_BLOCKS * BLOCK_SIZE), true),
            usaa_passthrough_calls: parking_lot::Mutex::new(Vec::new()),
            transfer_params: None,
            ..MockSlot::default()
        },
    );
}

/// Install a prefill-flavor slot with the given session back-reference.
///
/// The slot's `all_hashes` is set to ONLY the local-match hashes (the portion
/// that prefill pulls from decode). This matches the cd_loopback pattern and
/// ensures `expected_outputs = all_hashes - sequence_hashes = empty`, so the
/// observer's pending set is never populated. Without this,
/// `on_request_finished`'s finalize-deferral spawned task would spin on
/// `has_pending()` forever (since we call `commit_output_blocks` directly
/// rather than via the offload pipeline observer).
fn install_prefill_slot(
    inst: &DualRoleInstance,
    request_id: &str,
    seq: &RequestSeq,
    session_id: SessionId,
    initiator_instance_id: kvbm_connector::InstanceId,
    decode_endpoint: SessionEndpoint,
) {
    let local_hashes = seq.local_hashes();
    inst.inner.install_slot(
        request_id,
        MockSlot {
            block_size: BLOCK_SIZE,
            total_blocks: LOCAL_BLOCKS,
            computed_blocks: 0,
            local_match_blocks: 0,
            // all_hashes = local_hashes only: expected_outputs = local-local = empty.
            // Prevents the finalize-deferral watchdog from blocking indefinitely.
            all_hashes: local_hashes.clone(),
            token_blocks: seq.token_blocks[..LOCAL_BLOCKS].to_vec(),
            local_match_g2: parking_lot::Mutex::new(Some(Vec::new())),
            assigned_block_ids: parking_lot::Mutex::new(None),
            gnmt_result: (Some(0), false),
            usaa_passthrough_calls: parking_lot::Mutex::new(Vec::new()),
            transfer_params: Some(TransferParams::remote_prefill(RemotePrefillParams {
                protocol_version: DISAGG_PROTOCOL_VERSION,
                session_id,
                initiator_instance_id,
                decode_endpoint: Some(decode_endpoint),
                sequence_hashes: local_hashes,
                num_computed_tokens: COMPUTED_BLOCKS * BLOCK_SIZE,
            })),
            ..MockSlot::default()
        },
    );
}

/// Drive a decode→prefill exchange to full completion.
///
/// `decode_inst` opens the session, publishes local-match, drives pull.
/// `prefill_inst` attaches, pulls local-match from decode, publishes remote
/// output blocks.
///
/// `decode_g1`: full G1 window on decode side (TOTAL_BLOCKS).
/// `prefill_g1`: G1 slice for prefill's local-match (LOCAL_BLOCKS).
/// `seq`: token sequence used by the decode slot (prefill output blocks are
///        built from `seq.remote_token_blocks()` so hashes match decode's
///        `expected_remote_hashes`).
async fn drive_one_exchange(
    request_id: &str,
    seq: &RequestSeq,
    decode_inst: &DualRoleInstance,
    prefill_inst: &DualRoleInstance,
    decode_g1: Vec<usize>,
    prefill_g1: Vec<usize>,
) -> Result<()> {
    // 1. Install decode slot and open session via GNMT.
    install_decode_slot(decode_inst, request_id, seq);
    decode_inst
        .decode_wrapper
        .create_slot(make_request(request_id))?;
    let (d_count, d_async) = decode_inst
        .decode_wrapper
        .get_num_new_matched_tokens(request_id, COMPUTED_BLOCKS * BLOCK_SIZE)?;
    assert_eq!(
        d_count,
        Some((LOCAL_BLOCKS + REMOTE_BLOCKS) * BLOCK_SIZE),
        "[{}] decode GNMT must report local+remote external",
        request_id
    );
    assert!(d_async, "[{}] decode GNMT must be async", request_id);

    // 2. Wait for queue entry; capture session_id and decode endpoint.
    wait_until(|| {
        decode_inst
            .queue
            .snapshot()
            .iter()
            .any(|r| r.request_id == request_id)
    })
    .await;
    let queued = decode_inst.queue.snapshot();
    let q = queued
        .iter()
        .find(|r| r.request_id == request_id)
        .expect("queued entry");
    let session_id: SessionId = q.session_id;
    let decode_endpoint: SessionEndpoint = q
        .decode_endpoint
        .clone()
        .expect("queued decode_endpoint");
    let local_match_hashes = q.sequence_hashes.clone();
    assert_eq!(local_match_hashes.len(), LOCAL_BLOCKS);

    // 3. Install prefill slot referencing the opened session.
    install_prefill_slot(
        prefill_inst,
        request_id,
        seq,
        session_id,
        decode_inst.inner.local_id(),
        decode_endpoint,
    );

    // 4. Decode USAA-1 — kicks local G2→G1 onboard.
    decode_inst.decode_wrapper.update_state_after_alloc(
        request_id,
        decode_g1.clone(),
        (LOCAL_BLOCKS + REMOTE_BLOCKS) * BLOCK_SIZE,
    )?;
    let x_local_dst = decode_g1[COMPUTED_BLOCKS..COMPUTED_BLOCKS + LOCAL_BLOCKS].to_vec();
    wait_until(|| {
        decode_inst
            .transport
            .onboard_calls()
            .iter()
            .any(|c| c.dst_g1_block_ids == x_local_dst)
    })
    .await;
    let d_local_idx = decode_inst
        .transport
        .onboard_calls()
        .iter()
        .position(|c| c.dst_g1_block_ids == x_local_dst)
        .expect("decode local-match onboard");
    decode_inst.transport.resolve_onboard(d_local_idx, Ok(()));

    // 5. Prefill GNMT — coordinator attaches (pairs sessions) and drains
    //    commits + availability from the now-open decode session.
    prefill_inst
        .prefill_wrapper
        .create_slot(make_request(request_id))?;
    let (p_count, p_async) = prefill_inst
        .prefill_wrapper
        .get_num_new_matched_tokens(request_id, 0)?;
    assert_eq!(
        p_count,
        Some(LOCAL_BLOCKS * BLOCK_SIZE),
        "[{}] prefill GNMT must report local-match token count",
        request_id
    );
    assert!(p_async, "[{}] prefill GNMT must be async", request_id);

    // 6. Prefill USAA — kicks G2→G1 onboard for the pulled blocks.
    prefill_inst.prefill_wrapper.update_state_after_alloc(
        request_id,
        prefill_g1.clone(),
        LOCAL_BLOCKS * BLOCK_SIZE,
    )?;
    wait_until(|| {
        prefill_inst
            .transport
            .onboard_calls()
            .iter()
            .any(|c| c.dst_g1_block_ids == prefill_g1)
    })
    .await;
    let p_onboard_idx = prefill_inst
        .transport
        .onboard_calls()
        .iter()
        .position(|c| c.dst_g1_block_ids == prefill_g1)
        .expect("prefill onboard");
    prefill_inst
        .transport
        .resolve_onboard(p_onboard_idx, Ok(()));

    // 7. Wait for prefill's mark_onboarding_complete.
    wait_until(|| prefill_inst.workers.completed_contains(request_id)).await;

    // 8. Prefill publishes output blocks back to decode.
    //    Critical: blocks must be completed with seq.remote_token_blocks()
    //    so their hashes match decode's expected_remote_hashes in
    //    run_remote_pipeline (decode_leader.rs:902-907).
    let out_mutables = prefill_inst
        .g2_manager
        .allocate_blocks(REMOTE_BLOCKS)
        .expect("alloc prefill output");
    let out_completes: Vec<_> = out_mutables
        .into_iter()
        .zip(seq.remote_token_blocks().iter())
        .map(|(m, tb)| m.complete(tb).expect("complete output"))
        .collect();
    let out_blocks = prefill_inst.g2_manager.register_blocks(out_completes);
    // Verify hashes match what decode expects before publishing.
    let expected_remote = seq.remote_hashes();
    for (block, expected) in out_blocks.iter().zip(expected_remote.iter()) {
        assert_eq!(
            block.sequence_hash(),
            *expected,
            "[{}] output block hash must match decode's expected_remote_hashes",
            request_id
        );
    }
    prefill_inst
        .prefill_coord
        .commit_output_blocks(request_id, out_blocks)?;

    // 9. Decode's run_remote_pipeline receives output, pulls, kicks remote
    //    G2→G1 onboard.
    let remote_dst = decode_g1[COMPUTED_BLOCKS + LOCAL_BLOCKS..].to_vec();
    wait_until(|| {
        decode_inst
            .transport
            .onboard_calls()
            .iter()
            .any(|c| c.dst_g1_block_ids == remote_dst)
    })
    .await;
    let d_remote_idx = decode_inst
        .transport
        .onboard_calls()
        .iter()
        .position(|c| c.dst_g1_block_ids == remote_dst)
        .expect("decode remote-slice onboard");
    decode_inst.transport.resolve_onboard(d_remote_idx, Ok(()));

    // 10. Decode's mark_onboarding_complete.
    wait_until(|| decode_inst.workers.completed_contains(request_id)).await;

    // 11. Teardown.
    let _ = prefill_inst.prefill_wrapper.request_finished(request_id);
    let _ = decode_inst.decode_wrapper.request_finished(request_id);

    Ok(())
}

// ============================================================================
// Scenario 1 — sequential
// ============================================================================

/// One instance ("A") serves both decode (for req-X) and prefill (for req-Y)
/// sequentially. The two requests are driven one after the other; no
/// interleaving. Proves the dual-coordinator wiring is functionally correct
/// before testing concurrent scheduling.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn bidirectional_sequential_same_instance() -> Result<()> {
    // Paired factories: A↔B cross-wire sessions in-process.
    let (a_factory, b_factory) = MockSessionFactory::make_paired();
    let instance_a = build_instance(a_factory);
    let instance_b = build_instance(b_factory);

    // Assert dual-role state holds: both coordinators present and idle.
    assert_eq!(instance_a.decode_coord.active_count(), 0);
    assert_eq!(instance_a.prefill_coord.active_count(), 0);
    assert_eq!(instance_b.decode_coord.active_count(), 0);
    assert_eq!(instance_b.prefill_coord.active_count(), 0);

    // ---- Request X: A is decode, B is prefill ----
    let x_seq = RequestSeq::new(100);
    drive_one_exchange(
        "req-X",
        &x_seq,
        &instance_a,
        &instance_b,
        (1000..1000 + TOTAL_BLOCKS).collect(), // A-decode G1
        (1500..1500 + LOCAL_BLOCKS).collect(), // B-prefill G1
    )
    .await?;

    assert!(
        instance_a.workers.completed_contains("req-X"),
        "decode-A must complete req-X"
    );
    assert!(
        instance_b.workers.completed_contains("req-X"),
        "prefill-B must complete req-X"
    );
    assert!(instance_a.workers.failed_for("req-X").is_none());
    assert!(instance_b.workers.failed_for("req-X").is_none());

    wait_until(|| instance_a.decode_coord.active_count() == 0).await;
    wait_until(|| instance_b.prefill_coord.active_count() == 0).await;

    // ---- Request Y: A is prefill, B is decode ----
    let y_seq = RequestSeq::new(200);
    drive_one_exchange(
        "req-Y",
        &y_seq,
        &instance_b, // decode instance
        &instance_a, // prefill instance
        (3000..3000 + TOTAL_BLOCKS).collect(), // B-decode G1
        (2500..2500 + LOCAL_BLOCKS).collect(), // A-prefill G1
    )
    .await?;

    assert!(
        instance_b.workers.completed_contains("req-Y"),
        "decode-B must complete req-Y"
    );
    assert!(
        instance_a.workers.completed_contains("req-Y"),
        "prefill-A must complete req-Y"
    );
    assert!(instance_a.workers.failed_for("req-Y").is_none());
    assert!(instance_b.workers.failed_for("req-Y").is_none());

    // All coordinators must be idle after both requests finish.
    wait_until(|| instance_a.decode_coord.active_count() == 0).await;
    wait_until(|| instance_a.prefill_coord.active_count() == 0).await;
    wait_until(|| instance_b.decode_coord.active_count() == 0).await;
    wait_until(|| instance_b.prefill_coord.active_count() == 0).await;

    assert_eq!(instance_a.decode_coord.active_count(), 0);
    assert_eq!(instance_a.prefill_coord.active_count(), 0);
    assert_eq!(instance_b.decode_coord.active_count(), 0);
    assert_eq!(instance_b.prefill_coord.active_count(), 0);

    // Inflight budget: decode-A's budget was consumed by req-X but released
    // after completion. A's prefill role for req-Y never touched the decode
    // budget — PrefillCoordinatorImpl has no InflightBudget field. Budget
    // independence confirmed (see prefill_coordinator.rs and decode_leader.rs:418).
    assert_eq!(instance_a.decode_wrapper.inflight_available(), usize::MAX);
    assert_eq!(instance_b.decode_wrapper.inflight_available(), usize::MAX);

    let _ = Duration::from_secs(0);
    Ok(())
}

// ============================================================================
// Scenario 2 — concurrent
// ============================================================================

/// Same dual-coordinator setup as Scenario 1, but X and Y are in flight
/// simultaneously. Both decode openings happen first (so both session_ids
/// are known), then both prefill attachments happen in parallel, then
/// decode/prefill USAAs and outputs are driven in interleaved order.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn bidirectional_concurrent_same_instance() -> Result<()> {
    let (a_factory, b_factory) = MockSessionFactory::make_paired();
    let instance_a = build_instance(a_factory);
    let instance_b = build_instance(b_factory);

    // Two request sequences: disjoint seeds → disjoint hashes.
    let x_seq = RequestSeq::new(300);
    let y_seq = RequestSeq::new(400);

    // Disjoint G1 ranges across all four roles:
    //   A-decode for X: 5000..5004
    //   B-prefill for X: 5500..5502
    //   B-decode for Y: 6000..6004
    //   A-prefill for Y: 6500..6502
    let a_decode_x_g1: Vec<usize> = (5000..5000 + TOTAL_BLOCKS).collect();
    let b_prefill_x_g1: Vec<usize> = (5500..5500 + LOCAL_BLOCKS).collect();
    let b_decode_y_g1: Vec<usize> = (6000..6000 + TOTAL_BLOCKS).collect();
    let a_prefill_y_g1: Vec<usize> = (6500..6500 + LOCAL_BLOCKS).collect();

    // ---- Step 1: Install both decode slots ----

    // X: A is decode.
    let x_local_mutables = instance_a
        .g2_manager
        .allocate_blocks(LOCAL_BLOCKS)
        .expect("alloc X local");
    let x_local_completes: Vec<_> = x_local_mutables
        .into_iter()
        .zip(x_seq.local_token_blocks().iter())
        .map(|(m, tb)| m.complete(tb).expect("complete X local"))
        .collect();
    let x_local_g2 = instance_a.g2_manager.register_blocks(x_local_completes);
    instance_a.inner.install_slot(
        "req-X",
        MockSlot {
            block_size: BLOCK_SIZE,
            total_blocks: TOTAL_BLOCKS,
            computed_blocks: COMPUTED_BLOCKS,
            local_match_blocks: LOCAL_BLOCKS,
            all_hashes: x_seq.all_hashes.clone(),
            token_blocks: x_seq.token_blocks.clone(),
            local_match_g2: parking_lot::Mutex::new(Some(x_local_g2)),
            assigned_block_ids: parking_lot::Mutex::new(None),
            gnmt_result: (Some(LOCAL_BLOCKS * BLOCK_SIZE), true),
            usaa_passthrough_calls: parking_lot::Mutex::new(Vec::new()),
            transfer_params: None,
            ..MockSlot::default()
        },
    );

    // Y: B is decode.
    let y_local_mutables = instance_b
        .g2_manager
        .allocate_blocks(LOCAL_BLOCKS)
        .expect("alloc Y local");
    let y_local_completes: Vec<_> = y_local_mutables
        .into_iter()
        .zip(y_seq.local_token_blocks().iter())
        .map(|(m, tb)| m.complete(tb).expect("complete Y local"))
        .collect();
    let y_local_g2 = instance_b.g2_manager.register_blocks(y_local_completes);
    instance_b.inner.install_slot(
        "req-Y",
        MockSlot {
            block_size: BLOCK_SIZE,
            total_blocks: TOTAL_BLOCKS,
            computed_blocks: COMPUTED_BLOCKS,
            local_match_blocks: LOCAL_BLOCKS,
            all_hashes: y_seq.all_hashes.clone(),
            token_blocks: y_seq.token_blocks.clone(),
            local_match_g2: parking_lot::Mutex::new(Some(y_local_g2)),
            assigned_block_ids: parking_lot::Mutex::new(None),
            gnmt_result: (Some(LOCAL_BLOCKS * BLOCK_SIZE), true),
            usaa_passthrough_calls: parking_lot::Mutex::new(Vec::new()),
            transfer_params: None,
            ..MockSlot::default()
        },
    );

    // ---- Step 2: Open both sessions (decode GNMT, interleaved) ----
    instance_a
        .decode_wrapper
        .create_slot(make_request("req-X"))?;
    let _ = instance_a
        .decode_wrapper
        .get_num_new_matched_tokens("req-X", COMPUTED_BLOCKS * BLOCK_SIZE)?;

    instance_b
        .decode_wrapper
        .create_slot(make_request("req-Y"))?;
    let _ = instance_b
        .decode_wrapper
        .get_num_new_matched_tokens("req-Y", COMPUTED_BLOCKS * BLOCK_SIZE)?;

    // ---- Step 3: Capture both session_ids ----
    wait_until(|| {
        instance_a
            .queue
            .snapshot()
            .iter()
            .any(|r| r.request_id == "req-X")
    })
    .await;
    wait_until(|| {
        instance_b
            .queue
            .snapshot()
            .iter()
            .any(|r| r.request_id == "req-Y")
    })
    .await;

    let x_session_id: SessionId;
    let x_decode_endpoint: SessionEndpoint;
    {
        let snap = instance_a.queue.snapshot();
        let q = snap
            .iter()
            .find(|r| r.request_id == "req-X")
            .expect("X queued");
        x_session_id = q.session_id;
        x_decode_endpoint = q.decode_endpoint.clone().expect("X endpoint");
    }

    let y_session_id: SessionId;
    let y_decode_endpoint: SessionEndpoint;
    {
        let snap = instance_b.queue.snapshot();
        let q = snap
            .iter()
            .find(|r| r.request_id == "req-Y")
            .expect("Y queued");
        y_session_id = q.session_id;
        y_decode_endpoint = q.decode_endpoint.clone().expect("Y endpoint");
    }

    // ---- Step 4: Install both prefill slots ----

    // X prefill on B.
    // all_hashes = local_hashes only: expected_outputs = empty → observer
    // does not block on_request_finished's finalize-deferral task.
    let x_local_h = x_seq.local_hashes();
    instance_b.inner.install_slot(
        "req-X",
        MockSlot {
            block_size: BLOCK_SIZE,
            total_blocks: LOCAL_BLOCKS,
            computed_blocks: 0,
            local_match_blocks: 0,
            all_hashes: x_local_h.clone(),
            token_blocks: x_seq.token_blocks[..LOCAL_BLOCKS].to_vec(),
            local_match_g2: parking_lot::Mutex::new(Some(Vec::new())),
            assigned_block_ids: parking_lot::Mutex::new(None),
            gnmt_result: (Some(0), false),
            usaa_passthrough_calls: parking_lot::Mutex::new(Vec::new()),
            transfer_params: Some(TransferParams::remote_prefill(RemotePrefillParams {
                protocol_version: DISAGG_PROTOCOL_VERSION,
                session_id: x_session_id,
                initiator_instance_id: instance_a.inner.local_id(),
                decode_endpoint: Some(x_decode_endpoint),
                sequence_hashes: x_local_h,
                num_computed_tokens: COMPUTED_BLOCKS * BLOCK_SIZE,
            })),
            ..MockSlot::default()
        },
    );

    // Y prefill on A.
    let y_local_h = y_seq.local_hashes();
    instance_a.inner.install_slot(
        "req-Y",
        MockSlot {
            block_size: BLOCK_SIZE,
            total_blocks: LOCAL_BLOCKS,
            computed_blocks: 0,
            local_match_blocks: 0,
            all_hashes: y_local_h.clone(),
            token_blocks: y_seq.token_blocks[..LOCAL_BLOCKS].to_vec(),
            local_match_g2: parking_lot::Mutex::new(Some(Vec::new())),
            assigned_block_ids: parking_lot::Mutex::new(None),
            gnmt_result: (Some(0), false),
            usaa_passthrough_calls: parking_lot::Mutex::new(Vec::new()),
            transfer_params: Some(TransferParams::remote_prefill(RemotePrefillParams {
                protocol_version: DISAGG_PROTOCOL_VERSION,
                session_id: y_session_id,
                initiator_instance_id: instance_b.inner.local_id(),
                decode_endpoint: Some(y_decode_endpoint),
                sequence_hashes: y_local_h,
                num_computed_tokens: COMPUTED_BLOCKS * BLOCK_SIZE,
            })),
            ..MockSlot::default()
        },
    );

    // ---- Step 5: Interleaved decode USAAs ----
    instance_a.decode_wrapper.update_state_after_alloc(
        "req-X",
        a_decode_x_g1.clone(),
        (LOCAL_BLOCKS + REMOTE_BLOCKS) * BLOCK_SIZE,
    )?;
    instance_b.decode_wrapper.update_state_after_alloc(
        "req-Y",
        b_decode_y_g1.clone(),
        (LOCAL_BLOCKS + REMOTE_BLOCKS) * BLOCK_SIZE,
    )?;

    // Resolve both decode local-match onboards (interleaved).
    let x_local_dst = a_decode_x_g1[COMPUTED_BLOCKS..COMPUTED_BLOCKS + LOCAL_BLOCKS].to_vec();
    let y_local_dst = b_decode_y_g1[COMPUTED_BLOCKS..COMPUTED_BLOCKS + LOCAL_BLOCKS].to_vec();

    wait_until(|| {
        instance_a
            .transport
            .onboard_calls()
            .iter()
            .any(|c| c.dst_g1_block_ids == x_local_dst)
    })
    .await;
    let x_d_local_idx = instance_a
        .transport
        .onboard_calls()
        .iter()
        .position(|c| c.dst_g1_block_ids == x_local_dst)
        .unwrap();
    instance_a.transport.resolve_onboard(x_d_local_idx, Ok(()));

    wait_until(|| {
        instance_b
            .transport
            .onboard_calls()
            .iter()
            .any(|c| c.dst_g1_block_ids == y_local_dst)
    })
    .await;
    let y_d_local_idx = instance_b
        .transport
        .onboard_calls()
        .iter()
        .position(|c| c.dst_g1_block_ids == y_local_dst)
        .unwrap();
    instance_b.transport.resolve_onboard(y_d_local_idx, Ok(()));

    // ---- Step 6: Interleaved prefill GNMTs and USAAs ----
    instance_b
        .prefill_wrapper
        .create_slot(make_request("req-X"))?;
    let _ = instance_b
        .prefill_wrapper
        .get_num_new_matched_tokens("req-X", 0)?;

    instance_a
        .prefill_wrapper
        .create_slot(make_request("req-Y"))?;
    let _ = instance_a
        .prefill_wrapper
        .get_num_new_matched_tokens("req-Y", 0)?;

    instance_b.prefill_wrapper.update_state_after_alloc(
        "req-X",
        b_prefill_x_g1.clone(),
        LOCAL_BLOCKS * BLOCK_SIZE,
    )?;
    instance_a.prefill_wrapper.update_state_after_alloc(
        "req-Y",
        a_prefill_y_g1.clone(),
        LOCAL_BLOCKS * BLOCK_SIZE,
    )?;

    // Resolve both prefill onboards.
    wait_until(|| {
        instance_b
            .transport
            .onboard_calls()
            .iter()
            .any(|c| c.dst_g1_block_ids == b_prefill_x_g1)
    })
    .await;
    let x_p_idx = instance_b
        .transport
        .onboard_calls()
        .iter()
        .position(|c| c.dst_g1_block_ids == b_prefill_x_g1)
        .unwrap();
    instance_b.transport.resolve_onboard(x_p_idx, Ok(()));

    wait_until(|| {
        instance_a
            .transport
            .onboard_calls()
            .iter()
            .any(|c| c.dst_g1_block_ids == a_prefill_y_g1)
    })
    .await;
    let y_p_idx = instance_a
        .transport
        .onboard_calls()
        .iter()
        .position(|c| c.dst_g1_block_ids == a_prefill_y_g1)
        .unwrap();
    instance_a.transport.resolve_onboard(y_p_idx, Ok(()));

    // Wait for both prefill completions.
    wait_until(|| instance_b.workers.completed_contains("req-X")).await;
    wait_until(|| instance_a.workers.completed_contains("req-Y")).await;

    // ---- Step 7: Interleaved output publishing ----

    // X output from B (uses x_seq.remote_token_blocks() for correct hashes).
    let x_out_mutables = instance_b
        .g2_manager
        .allocate_blocks(REMOTE_BLOCKS)
        .expect("alloc X output");
    let x_out_completes: Vec<_> = x_out_mutables
        .into_iter()
        .zip(x_seq.remote_token_blocks().iter())
        .map(|(m, tb)| m.complete(tb).expect("complete X output"))
        .collect();
    let x_out_blocks = instance_b.g2_manager.register_blocks(x_out_completes);
    instance_b
        .prefill_coord
        .commit_output_blocks("req-X", x_out_blocks)?;

    // Y output from A (uses y_seq.remote_token_blocks() for correct hashes).
    let y_out_mutables = instance_a
        .g2_manager
        .allocate_blocks(REMOTE_BLOCKS)
        .expect("alloc Y output");
    let y_out_completes: Vec<_> = y_out_mutables
        .into_iter()
        .zip(y_seq.remote_token_blocks().iter())
        .map(|(m, tb)| m.complete(tb).expect("complete Y output"))
        .collect();
    let y_out_blocks = instance_a.g2_manager.register_blocks(y_out_completes);
    instance_a
        .prefill_coord
        .commit_output_blocks("req-Y", y_out_blocks)?;

    // ---- Step 7b: Dual-role in-flight assertions ----
    // At this exact point both requests are mid-flight: req-X is actively
    // being decoded by A while A simultaneously holds a live prefill session
    // for req-Y (output committed but decode pull not yet resolved). Likewise
    // B is decoding req-Y while its prefill session for req-X is still live.
    // Both coordinators on each instance must be non-zero simultaneously.
    assert!(
        instance_a.decode_coord.active_count() >= 1,
        "instance_a decode must still be active for req-X (awaiting remote pull)"
    );
    assert!(
        instance_a.prefill_coord.active_count() >= 1,
        "instance_a prefill must still be active for req-Y (peer rendezvous pending)"
    );
    assert!(
        instance_b.decode_coord.active_count() >= 1,
        "instance_b decode must still be active for req-Y (awaiting remote pull)"
    );
    assert!(
        instance_b.prefill_coord.active_count() >= 1,
        "instance_b prefill must still be active for req-X (peer rendezvous pending)"
    );

    // ---- Step 8: Resolve both remote-slice onboards ----
    let x_remote_dst = a_decode_x_g1[COMPUTED_BLOCKS + LOCAL_BLOCKS..].to_vec();
    let y_remote_dst = b_decode_y_g1[COMPUTED_BLOCKS + LOCAL_BLOCKS..].to_vec();

    wait_until(|| {
        instance_a
            .transport
            .onboard_calls()
            .iter()
            .any(|c| c.dst_g1_block_ids == x_remote_dst)
    })
    .await;
    let x_d_remote_idx = instance_a
        .transport
        .onboard_calls()
        .iter()
        .position(|c| c.dst_g1_block_ids == x_remote_dst)
        .unwrap();
    instance_a.transport.resolve_onboard(x_d_remote_idx, Ok(()));

    wait_until(|| {
        instance_b
            .transport
            .onboard_calls()
            .iter()
            .any(|c| c.dst_g1_block_ids == y_remote_dst)
    })
    .await;
    let y_d_remote_idx = instance_b
        .transport
        .onboard_calls()
        .iter()
        .position(|c| c.dst_g1_block_ids == y_remote_dst)
        .unwrap();
    instance_b.transport.resolve_onboard(y_d_remote_idx, Ok(()));

    // ---- Step 9: Wait for both decode completions ----
    wait_until(|| instance_a.workers.completed_contains("req-X")).await;
    wait_until(|| instance_b.workers.completed_contains("req-Y")).await;

    // ---- Step 10: Teardown ----
    let _ = instance_b.prefill_wrapper.request_finished("req-X");
    let _ = instance_a.decode_wrapper.request_finished("req-X");
    let _ = instance_a.prefill_wrapper.request_finished("req-Y");
    let _ = instance_b.decode_wrapper.request_finished("req-Y");

    // ---- Step 11: Final assertions ----
    assert!(instance_a.workers.completed_contains("req-X"));
    assert!(instance_b.workers.completed_contains("req-X"));
    assert!(instance_b.workers.completed_contains("req-Y"));
    assert!(instance_a.workers.completed_contains("req-Y"));

    assert!(instance_a.workers.failed_for("req-X").is_none());
    assert!(instance_b.workers.failed_for("req-X").is_none());
    assert!(instance_a.workers.failed_for("req-Y").is_none());
    assert!(instance_b.workers.failed_for("req-Y").is_none());

    wait_until(|| instance_a.decode_coord.active_count() == 0).await;
    wait_until(|| instance_a.prefill_coord.active_count() == 0).await;
    wait_until(|| instance_b.decode_coord.active_count() == 0).await;
    wait_until(|| instance_b.prefill_coord.active_count() == 0).await;

    assert_eq!(instance_a.decode_coord.active_count(), 0);
    assert_eq!(instance_a.prefill_coord.active_count(), 0);
    assert_eq!(instance_b.decode_coord.active_count(), 0);
    assert_eq!(instance_b.prefill_coord.active_count(), 0);

    // Inflight-budget independence: A's decode budget (consumed by req-X at
    // decode_leader.rs:418) is fully independent of A's prefill coordinator
    // (for req-Y). PrefillCoordinatorImpl has no InflightBudget field.
    assert_eq!(instance_a.decode_wrapper.inflight_available(), usize::MAX);
    assert_eq!(instance_b.decode_wrapper.inflight_available(), usize::MAX);

    let _ = Duration::from_secs(0);
    Ok(())
}
