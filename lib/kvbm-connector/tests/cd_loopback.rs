// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Cross-wrapper loopback test — proves `DecodeDisaggLeader` and
//! `PrefillDisaggLeader` compose correctly around a real
//! `DisaggSession` pair (real velo, no nixl, no RDMA).
//!
//! Mocks: `CdBlockTransport` (replaces RDMA on both sides),
//! `CdWorkerHook` (records worker calls on both sides),
//! `MockInnerLeaderShim` on both sides (simulated slots).
//! Real coordinators on both sides
//! (`RemotePrefillCoordinator` for decode,
//! `PrefillCoordinatorImpl` for prefill). Real velo via
//! TCP loopback.
//!
//! Layout: COMPUTED=0, LOCAL=2, REMOTE=2, TOTAL=4 blocks of 16 tokens.
//! D has cached the LOCAL slice; P pulls those, prefills the REMOTE
//! slice, publishes the outputs back to D, D pulls them, sends
//! `PullComplete`, P acks. Full goal: `mark_onboarding_complete`
//! fires on D (its remote slice landed) and on P (its external
//! tokens onboarded).
//!
//! See the canonical plan
//! (`/home/ryan/.claude/plans/cd-usaa-pipeline.md` §A.6) for
//! the full sequence diagram.

use std::sync::Arc;
use std::time::Duration;

use anyhow::Result;
use kvbm_connector::G2;
use kvbm_connector::common::Request;
use kvbm_connector::connector::leader::disagg::prefill_coordinator::PrefillCoordinatorImpl;
use kvbm_connector::connector::leader::disagg::testing::{
    InMemoryRemotePrefillQueue, MockCdBlockTransport, MockCdWorkerHook, MockInnerLeaderShim,
    MockSlot, TEST_BLOCK_SIZE, wait_until,
};
use kvbm_connector::connector::leader::disagg::transport::VeloPrefillSessionAttacher;
use kvbm_connector::connector::leader::disagg::{
    AlwaysRemote, ConnectorLeaderApi, DecodeDisaggLeader, PrefillDisaggLeader,
    RemotePrefillCoordinator, VeloPrefillSessionFactory,
};
use kvbm_config::{DisaggConfig, DisaggregationRole};
use kvbm_disagg_protocol::{
    DISAGG_PROTOCOL_VERSION, RemotePrefillParams, SessionEndpoint, SessionId, TransferParams,
};
use kvbm_engine::testing::managers::{TestManagerBuilder, TestRegistryBuilder};
use kvbm_engine::testing::token_blocks::{create_token_sequence, generate_sequence_hashes};
use kvbm_logical::manager::BlockManager;
use velo::backend::tcp::TcpTransportBuilder;

const COMPUTED_BLOCKS: usize = 0;
const LOCAL_BLOCKS: usize = 2;
const REMOTE_BLOCKS: usize = 2;
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

fn new_velo_transport() -> Arc<velo::backend::tcp::TcpTransport> {
    let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
    Arc::new(
        TcpTransportBuilder::new()
            .from_listener(listener)
            .unwrap()
            .build()
            .unwrap(),
    )
}

async fn new_velo() -> Arc<velo::Velo> {
    velo::Velo::builder()
        .add_transport(new_velo_transport())
        .build()
        .await
        .unwrap()
}

// Temporarily disabled during the session-API refactor (Stage 4
// rewrites this test against the new symmetric Session trait).
#[cfg(any())]
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn cd_loopback_decode_prefill_session() -> Result<()> {
    // ------------------------------------------------------------------
    // Velo plumbing
    // ------------------------------------------------------------------
    let d_velo = new_velo().await;
    let p_velo = new_velo().await;
    // Mutual peer registration so attach can route both directions.
    d_velo.register_peer(p_velo.peer_info()).unwrap();
    p_velo.register_peer(d_velo.peer_info()).unwrap();

    let d_instance_id: kvbm_connector::InstanceId = d_velo.instance_id();
    let p_instance_id: kvbm_connector::InstanceId = p_velo.instance_id();

    // ------------------------------------------------------------------
    // Build shared token sequence so D's slot hashes line up with P's.
    // ------------------------------------------------------------------
    let token_sequence = create_token_sequence(TOTAL_BLOCKS, BLOCK_SIZE, 100);
    let all_hashes = generate_sequence_hashes(&token_sequence);
    let token_blocks: Vec<_> = token_sequence.blocks().to_vec();
    let local_match_hashes: Vec<_> =
        all_hashes[COMPUTED_BLOCKS..COMPUTED_BLOCKS + LOCAL_BLOCKS].to_vec();

    // ------------------------------------------------------------------
    // Decode side
    // ------------------------------------------------------------------
    let d_g2_manager = build_g2_manager(64);

    // Pre-build + register D's local-match G2 blocks so they show up
    // in the session's ready_blocks.
    let d_local_mutables = d_g2_manager
        .allocate_blocks(LOCAL_BLOCKS)
        .expect("alloc d local match");
    let d_local_completes: Vec<_> = d_local_mutables
        .into_iter()
        .zip(token_blocks[COMPUTED_BLOCKS..COMPUTED_BLOCKS + LOCAL_BLOCKS].iter())
        .map(|(m, tb)| m.complete(tb).expect("complete d local"))
        .collect();
    let d_local_g2 = d_g2_manager.register_blocks(d_local_completes);

    let d_inner = MockInnerLeaderShim::new(BLOCK_SIZE, d_g2_manager.clone());
    let d_g1: Vec<usize> = (1000..1000 + TOTAL_BLOCKS).collect();
    d_inner.install_slot(
        "req-1",
        MockSlot {
            block_size: BLOCK_SIZE,
            total_blocks: TOTAL_BLOCKS,
            computed_blocks: COMPUTED_BLOCKS,
            local_match_blocks: LOCAL_BLOCKS,
            all_hashes: all_hashes.clone(),
            token_blocks: token_blocks.clone(),
            local_match_g2: parking_lot::Mutex::new(Some(d_local_g2)),
            assigned_block_ids: parking_lot::Mutex::new(None),
            gnmt_result: (Some(LOCAL_BLOCKS * BLOCK_SIZE), true),
            usaa_passthrough_calls: parking_lot::Mutex::new(Vec::new()),
            transfer_params: None,
        },
    );

    let d_factory = VeloPrefillSessionFactory::new(Arc::clone(&d_velo));
    let d_queue = InMemoryRemotePrefillQueue::new();
    let d_coordinator = RemotePrefillCoordinator::with_attach_timeout(
        Arc::new(AlwaysRemote),
        d_factory.clone(),
        d_queue.clone(),
        tokio::runtime::Handle::current(),
        Duration::from_secs(30),
    );
    let d_transport = MockCdBlockTransport::new();
    let d_workers = MockCdWorkerHook::new();

    let d_cfg = DisaggConfig {
        hub_url: "http://127.0.0.1:1337".to_string(),
        role: DisaggregationRole::Decode,
        max_inflight_remote_prefill_tokens: usize::MAX,
    };
    let d_wrapper = DecodeDisaggLeader::from_parts(
        d_inner.clone(),
        &d_cfg,
        d_coordinator.clone(),
        d_transport.clone(),
        d_workers.clone(),
        tokio::runtime::Handle::current(),
        None,
        None,
        None,
    );

    // ------------------------------------------------------------------
    // Prefill side
    // ------------------------------------------------------------------
    let p_g2_manager = build_g2_manager(64);
    let p_inner = MockInnerLeaderShim::new(BLOCK_SIZE, p_g2_manager.clone());
    let p_g1: Vec<usize> = (2000..2000 + LOCAL_BLOCKS).collect();

    // P's slot will be installed once we know the session_id from D.
    // Build a placeholder for token_blocks indexed [0, LOCAL_BLOCKS).
    let p_token_blocks: Vec<_> = token_blocks[..LOCAL_BLOCKS].to_vec();

    let p_attacher = VeloPrefillSessionAttacher::new(Arc::clone(&p_velo));
    let p_transport = MockCdBlockTransport::new();
    let p_workers = MockCdWorkerHook::new();
    let p_coordinator = PrefillCoordinatorImpl::new(
        p_inner.clone(),
        p_transport.clone(),
        p_workers.clone(),
        p_attacher.clone(),
        tokio::runtime::Handle::current(),
    );
    let p_wrapper = PrefillDisaggLeader::from_parts(
        p_inner.clone(),
        p_coordinator.clone(),
        p_workers.clone(),
    );

    // ------------------------------------------------------------------
    // Drive D-side GNMT to create the session.
    // ------------------------------------------------------------------
    d_wrapper.create_slot(make_request("req-1"))?;
    let (d_count, d_async) =
        d_wrapper.get_num_new_matched_tokens("req-1", COMPUTED_BLOCKS * BLOCK_SIZE)?;
    assert_eq!(
        d_count,
        Some((LOCAL_BLOCKS + REMOTE_BLOCKS) * BLOCK_SIZE),
        "D GNMT should report local+remote external"
    );
    assert!(d_async);

    // Wait for the queue.enqueue spawned task to land so we can read
    // session_id and decode_endpoint.
    wait_until(|| !d_queue.snapshot().is_empty()).await;
    let queued = d_queue.snapshot();
    let session_id: SessionId = queued[0].session_id;
    let decode_endpoint: SessionEndpoint = queued[0]
        .decode_endpoint
        .clone()
        .expect("queued decode_endpoint");

    // ------------------------------------------------------------------
    // P slot installs with transfer_params pointing at D's session.
    // sequence_hashes carry the LOCAL_MATCH hashes — what D will
    // serve when P sends BlockSetRequest{All}.
    // ------------------------------------------------------------------
    let p_transfer = TransferParams::remote_prefill(RemotePrefillParams {
        protocol_version: DISAGG_PROTOCOL_VERSION,
        session_id,
        initiator_instance_id: d_instance_id,
        decode_endpoint: Some(decode_endpoint),
        sequence_hashes: local_match_hashes.clone(),
    });
    p_inner.install_slot(
        "req-1",
        MockSlot {
            block_size: BLOCK_SIZE,
            total_blocks: LOCAL_BLOCKS,
            computed_blocks: 0,
            local_match_blocks: 0,
            all_hashes: local_match_hashes.clone(),
            token_blocks: p_token_blocks.clone(),
            local_match_g2: parking_lot::Mutex::new(Some(Vec::new())),
            assigned_block_ids: parking_lot::Mutex::new(None),
            gnmt_result: (Some(0), false),
            usaa_passthrough_calls: parking_lot::Mutex::new(Vec::new()),
            transfer_params: Some(p_transfer),
        },
    );

    // ------------------------------------------------------------------
    // P GNMT — coordinator attaches and asks D for blocks.
    // ------------------------------------------------------------------
    p_wrapper.create_slot(make_request("req-1"))?;
    let (p_count, p_async) = p_wrapper.get_num_new_matched_tokens("req-1", 0)?;
    assert_eq!(p_count, Some(LOCAL_BLOCKS * BLOCK_SIZE));
    assert!(p_async);
    // P's pull_remote fires after attach + BlockSetRequest round-trips
    // through real velo.
    p_transport.wait_pull_count(1).await;
    let p_pull = p_transport.pull_calls()[0].clone();
    assert_eq!(p_pull.remote_instance, d_instance_id);
    assert_eq!(p_pull.local_dst_g2_block_ids.len(), LOCAL_BLOCKS);
    p_transport.resolve_pull(0, Ok(()));

    // ------------------------------------------------------------------
    // D-side USAA-1 — gives the wrapper the full G1 list.
    //
    // The wrapper kicks the local G2→G1 transfer via D's
    // CdBlockTransport. We resolve it to satisfy `local_onboard_complete`.
    // ------------------------------------------------------------------
    d_wrapper.update_state_after_alloc(
        "req-1",
        d_g1.clone(),
        (LOCAL_BLOCKS + REMOTE_BLOCKS) * BLOCK_SIZE,
    )?;
    d_transport.wait_onboard_count(1).await;
    d_transport.resolve_onboard(0, Ok(()));

    // ------------------------------------------------------------------
    // P-side USAA — onboards the pulled G2 blocks into G1.
    // ------------------------------------------------------------------
    p_wrapper.update_state_after_alloc("req-1", p_g1.clone(), LOCAL_BLOCKS * BLOCK_SIZE)?;
    p_transport.wait_onboard_count(1).await;
    let p_onboard = p_transport.onboard_calls()[0].clone();
    assert_eq!(p_onboard.dst_g1_block_ids, p_g1);
    p_transport.resolve_onboard(0, Ok(()));

    wait_until(|| p_workers.completed_contains("req-1")).await;

    // ------------------------------------------------------------------
    // P produces forward-pass output blocks and publishes to D.
    //
    // Output hashes must match D's `expected_remote_hashes` (D's
    // wrapper validates against the slot's hash list). We use the
    // SAME token sequence as the input, just at positions
    // [LOCAL_BLOCKS, TOTAL_BLOCKS) — that way the positional
    // lineage hashes line up with D's expectations.
    // ------------------------------------------------------------------
    let p_output_mutables = p_g2_manager
        .allocate_blocks(REMOTE_BLOCKS)
        .expect("alloc p output");
    let p_output_completes: Vec<_> = p_output_mutables
        .into_iter()
        .zip(token_blocks[COMPUTED_BLOCKS + LOCAL_BLOCKS..].iter())
        .map(|(m, tb)| m.complete(tb).expect("complete p output"))
        .collect();
    let p_output_g2 = p_g2_manager.register_blocks(p_output_completes);
    // Sanity-check that the output blocks carry the hashes D expects.
    let expected_remote_hashes: Vec<_> = all_hashes[COMPUTED_BLOCKS + LOCAL_BLOCKS..].to_vec();
    for (block, expected) in p_output_g2.iter().zip(expected_remote_hashes.iter()) {
        assert_eq!(block.sequence_hash(), *expected);
    }

    // simulate_offload_complete drives the publish via the
    // coordinator's session.
    p_coordinator.simulate_offload_complete("req-1", p_output_g2)?;

    // ------------------------------------------------------------------
    // D's drive_block_sets_added picks up the BlockSetsAdded event
    // and fires the remote pull (mocked).
    // ------------------------------------------------------------------
    d_transport.wait_pull_count(1).await;
    let d_pull = d_transport.pull_calls()[0].clone();
    assert_eq!(d_pull.remote_instance, p_instance_id);
    assert_eq!(d_pull.local_dst_g2_block_ids.len(), REMOTE_BLOCKS);
    d_transport.resolve_pull(0, Ok(()));

    // Then D fires the second onboard (remote G2 → G1).
    d_transport.wait_onboard_count(2).await;
    d_transport.resolve_onboard(1, Ok(()));

    // ------------------------------------------------------------------
    // D sends PullComplete; P receives it, sends PullAck; D's
    // maybe_complete fires mark_workers_onboarding_complete on D.
    // ------------------------------------------------------------------
    wait_until(|| d_workers.completed_contains("req-1")).await;

    // request_finished on P detaches the slot side; the session
    // continues holding output pins until D's `PullComplete`
    // (already received) drains them.
    let _ = p_wrapper.request_finished("req-1");
    let _ = d_wrapper.request_finished("req-1");

    wait_until(|| p_coordinator.active_count() == 0).await;

    // ------------------------------------------------------------------
    // Assertions
    // ------------------------------------------------------------------
    assert!(d_workers.completed_contains("req-1"));
    assert!(p_workers.completed_contains("req-1"));
    assert!(!d_wrapper.has_active_cd_request("req-1"));
    assert_eq!(p_coordinator.active_count(), 0);
    assert!(d_workers.failed_for("req-1").is_none());
    assert!(p_workers.failed_for("req-1").is_none());

    Ok(())
}
