// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Integration test for `UnifiedDisaggLeader` per-request dispatch.
//!
//! Mirrors the harness pattern from `cd_decode_e2e.rs` but invokes
//! the wrapped flow through `UnifiedDisaggLeader::ConnectorLeaderApi`
//! to prove the dispatch lands at the right leader.
//!
//! The unified leader's classification uses
//! `inner.slot_transfer_params(...)`:
//!
//! - Slot without `TransferParams` ⇒ decode flow.
//! - Slot with `TransferParams::remote_prefill = Some(..)` ⇒ prefill
//!   flow (covered by `cd_prefill_e2e.rs`'s harness equivalents — out
//!   of scope here; this test exclusively validates the decode
//!   routing path because spinning up a full PrefillCoordinatorImpl
//!   in test requires a SessionFactory + PeerResolver stack already
//!   exercised end-to-end elsewhere).
//!
//! What this test does NOT cover:
//!
//! - Real session attach / pull / onboard (those run inside the
//!   wrapped `DecodeDisaggLeader` and are covered by `cd_decode_e2e`).
//! - Prefill-side dispatch (covered by `cd_prefill_e2e` once the
//!   unified leader is wired into init.rs and a corresponding
//!   integration test is added).
//! - Builder error / classify edge cases (unit tests in `unified.rs`).

use std::sync::Arc;

use anyhow::Result;
use kvbm_config::{DisaggConfig, DisaggregationRole};
use kvbm_connector::G2;
use kvbm_connector::common::Request;
use kvbm_connector::connector::leader::disagg::testing::{
    InMemoryRemotePrefillQueue, MockCdBlockTransport, MockCdWorkerHook, MockInnerLeaderShim,
    MockSlot, TEST_BLOCK_SIZE, wait_until,
};
use kvbm_connector::connector::leader::disagg::{
    AlwaysRemote, ConnectorLeaderApi, DecodeDisaggLeader, RemotePrefillCoordinator,
    UnifiedDisaggLeader,
};
use kvbm_engine::disagg::session::MockSessionFactory;
use kvbm_engine::testing::managers::{TestManagerBuilder, TestRegistryBuilder};
use kvbm_engine::testing::token_blocks::{create_token_sequence, generate_sequence_hashes};
use kvbm_logical::manager::BlockManager;

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

struct DecodeOnlyHarness {
    leader: Arc<UnifiedDisaggLeader>,
    queue: Arc<InMemoryRemotePrefillQueue>,
}

fn build_decode_only_harness() -> DecodeOnlyHarness {
    let g2_manager = build_g2_manager(32);

    let token_sequence = create_token_sequence(TOTAL_BLOCKS, BLOCK_SIZE, 100);
    let all_hashes = generate_sequence_hashes(&token_sequence);
    let token_blocks: Vec<_> = token_sequence.blocks().to_vec();

    // Pre-allocate + register the LOCAL-match G2 blocks so the
    // decode wrapper's GNMT can take them via the inner shim.
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
    inner.install_slot("req-1", slot);

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

    let leader = UnifiedDisaggLeader::builder(inner)
        .with_decode(decode)
        .build()
        .expect("build unified leader");

    DecodeOnlyHarness { leader, queue }
}

/// Decode-only wiring: a request without `TransferParams` is routed
/// to the decode flow, which runs the policy, sizes the remote
/// window, and enqueues a `RemotePrefillRequest` on the in-memory
/// queue.  Asserting `queue.snapshot().len() == 1` proves the
/// dispatch landed at the decode leader's GNMT path.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn unified_decode_only_routes_no_params_request_to_decode_flow() -> Result<()> {
    let h = build_decode_only_harness();

    h.leader.create_slot(make_request())?;

    let (count, async_flag) = h
        .leader
        .get_num_new_matched_tokens("req-1", COMPUTED_BLOCKS * BLOCK_SIZE)?;

    // The decode flow's policy is `AlwaysRemote`, sizing
    // `(LOCAL + REMOTE) * BLOCK_SIZE` external tokens.  If the
    // dispatch failed and the call landed on the inner instead, we
    // would see (Some(LOCAL_BLOCKS * BLOCK_SIZE), true) — the mock
    // slot's `gnmt_result`.
    assert_eq!(
        count,
        Some((LOCAL_BLOCKS + REMOTE_BLOCKS) * BLOCK_SIZE),
        "expected decode flow to size full remote window"
    );
    assert!(async_flag, "expected async-load flag from decode flow");

    wait_until(|| h.queue.snapshot().len() == 1).await;
    let enqueued = h.queue.snapshot();
    assert_eq!(
        enqueued.len(),
        1,
        "expected one remote-prefill request to be enqueued"
    );
    assert_eq!(enqueued[0].request_id, "req-1");

    Ok(())
}

/// Decode-only wiring: regardless of `transfer_params` shape, the
/// unified leader routes every request to the decode flow (the bare
/// leader's behavior on a decode-only instance).  The decode flow
/// ignores `transfer_params` and runs its own policy; with
/// `total_tokens=0` the policy hits the
/// `policy_remote_passthrough_zero_block` branch and returns the
/// inner result without enqueueing.
///
/// This pins the post-classify-refactor semantic: "single-flow wired
/// → route everything to that flow."  Pre-refactor behavior was to
/// fall through to inner directly, which broke trace-equivalence
/// because it skipped the wrapped leader's audit chain.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn unified_decode_only_routes_remote_prefill_request_through_decode_flow() -> Result<()> {
    let g2_manager = build_g2_manager(8);
    let inner = MockInnerLeaderShim::new(BLOCK_SIZE, g2_manager.clone());

    let slot = MockSlot {
        block_size: BLOCK_SIZE,
        gnmt_result: (Some(64), true),
        transfer_params: Some(kvbm_disagg_protocol::TransferParams::remote_prefill(
            kvbm_disagg_protocol::RemotePrefillParams::new(
                uuid::Uuid::new_v4(),
                uuid::Uuid::new_v4().into(),
            ),
        )),
        ..MockSlot::default()
    };
    inner.install_slot("req-x", slot);

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
    let leader = UnifiedDisaggLeader::builder(inner)
        .with_decode(decode)
        .build()
        .expect("build unified leader");

    let (count, async_flag) = leader.get_num_new_matched_tokens("req-x", 0)?;
    // Decode flow's policy_remote_passthrough_zero_block branch
    // returns the inner result verbatim because total_tokens=0
    // ⇒ prefill_window=0 ⇒ full_block_external_tokens=0.
    assert_eq!(count, Some(64));
    assert!(async_flag);
    // Decode flow ran and decided "passthrough" — queue stays empty
    // because no remote prefill was queued.
    assert_eq!(
        queue.snapshot().len(),
        0,
        "decode flow's zero-block-passthrough branch must not enqueue"
    );

    Ok(())
}
