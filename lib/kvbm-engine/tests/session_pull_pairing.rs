// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#![cfg(feature = "testing")]

//! Pull-pairing invariant tests.
//!
//! Contract: `Session::pull(hashes, dst)` lands data such that
//! `dst[i]` receives the block the peer committed for `hashes[i]`,
//! regardless of the order of `hashes` relative to commit order.
//!
//! These tests pin two layers of the contract:
//!
//! 1. **Trait-level (MockSession paired mode)**: the implementation
//!    records the (hashes, dst_block_ids) pair in the caller's
//!    order. A bug that reorders dst or hashes inside the impl
//!    fails this test.
//!
//! 2. **Wire-level (VeloSession loopback)**: `Frame::Pull` arriving
//!    at the holder carries the puller's shuffled hash list in
//!    caller order. A bug that reorders hashes before/after sending
//!    fails this test.
//!
//! The RDMA happy-path (data actually landing at the dst block) is
//! NOT exercised here because the velo loopback test infra runs
//! with `workers(vec![])` and cannot drive NIXL/CUDA transfers; the
//! `peer_block_ids ↔ dst_block_ids` zip in `velo.rs::pull` is
//! correct by construction (sequential lookup + zip on the same
//! index `i`) and is verified by inspection until worker-equipped
//! E2E coverage lands in a follow-up.

use std::sync::Arc;
use std::time::Duration;

use anyhow::Result;
use futures::StreamExt;
use kvbm_engine::G2;
use kvbm_engine::leader::InstanceLeader;
use kvbm_engine::p2p::session::{
    AvailabilityDelta, CommitDelta, MockSessionFactory, Session, SessionFactory, VeloSessionFactory,
};
use kvbm_engine::testing::managers::{TestManagerBuilder, TestRegistryBuilder};
use kvbm_engine::testing::token_blocks::create_token_sequence;
use kvbm_logical::blocks::{ImmutableBlock, MutableBlock};
use kvbm_logical::manager::BlockManager;
use velo::InstanceId;
use velo::transports::tcp::TcpTransportBuilder;

const BLOCK_SIZE: usize = 16;

// ============================================================================
// Shared block-construction helpers
// ============================================================================

fn make_g2_manager() -> Arc<BlockManager<G2>> {
    let registry = TestRegistryBuilder::new().build();
    Arc::new(
        TestManagerBuilder::<G2>::new()
            .block_count(32)
            .block_size(BLOCK_SIZE)
            .registry(registry)
            .build(),
    )
}

fn make_blocks(
    g2: &Arc<BlockManager<G2>>,
    count: usize,
    start_token: u32,
) -> Vec<ImmutableBlock<G2>> {
    let token_sequence = create_token_sequence(count, BLOCK_SIZE, start_token);
    let mutables = g2.allocate_blocks(count).expect("alloc immutable seed");
    let completes: Vec<_> = mutables
        .into_iter()
        .zip(token_sequence.blocks().iter())
        .map(|(m, tb)| m.complete(tb).expect("complete block"))
        .collect();
    g2.register_blocks(completes)
}

fn alloc_mutable(g2: &Arc<BlockManager<G2>>, count: usize) -> Vec<MutableBlock<G2>> {
    g2.allocate_blocks(count).expect("alloc dst")
}

// ============================================================================
// MockSession paired-mode: trait-level pairing
// ============================================================================

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn mock_paired_pull_preserves_caller_order() -> Result<()> {
    let (h_factory, p_factory) = MockSessionFactory::make_paired();
    let session_id = uuid::Uuid::new_v4();

    let h_session = h_factory.open(session_id)?;
    let h_endpoint = h_session.endpoint().expect("holder endpoint");
    let _p_session_dyn = p_factory
        .attach(session_id, InstanceId::new_v4(), h_endpoint)
        .await?;
    let p_session = p_factory
        .last_attached()
        .expect("puller MockSession registered on attach");

    // Holder publishes 3 blocks in natural order: A, B, C.
    let g2_holder = make_g2_manager();
    let blocks = make_blocks(&g2_holder, 3, 100);
    let hashes_in_order: Vec<_> = blocks.iter().map(|b| b.sequence_hash()).collect();
    let (h_a, h_b, h_c) = (hashes_in_order[0], hashes_in_order[1], hashes_in_order[2]);

    h_session.commit(hashes_in_order.clone())?;
    h_session.make_available(blocks)?;
    h_session.finish_commits()?;
    h_session.finish_availability()?;

    // Puller drains both streams to the closed terminator (sealed sets).
    let mut commits = p_session.commits();
    while let Some(d) = commits.next().await {
        if matches!(d, CommitDelta::Closed) {
            break;
        }
    }
    let mut avail = p_session.availability();
    while let Some(d) = avail.next().await {
        if matches!(d, AvailabilityDelta::Drained) {
            break;
        }
    }

    // Puller pulls with hashes shuffled: [C, A, B]; dst in natural order.
    let g2_puller = make_g2_manager();
    let dst = alloc_mutable(&g2_puller, 3);
    let dst_ids: Vec<_> = dst.iter().map(|b| b.block_id()).collect();
    let shuffled = vec![h_c, h_a, h_b];

    let returned = p_session.pull(shuffled.clone(), dst).await?;
    assert_eq!(returned.len(), 3, "pull must return all 3 dst blocks");
    let returned_ids: Vec<_> = returned.iter().map(|b| b.block_id()).collect();
    assert_eq!(
        returned_ids, dst_ids,
        "returned dst must preserve caller-provided order"
    );

    // Trait-level contract: the implementation recorded the call with
    // (hashes, dst_block_ids) in the caller-provided order. A bug that
    // reorders either inside the implementation fails here.
    let calls = p_session.pull_calls();
    assert_eq!(calls.len(), 1, "exactly one pull call recorded");
    let (recorded_hashes, recorded_dst_ids) = &calls[0];
    assert_eq!(
        recorded_hashes, &shuffled,
        "recorded hashes must equal caller's shuffled input"
    );
    assert_eq!(
        recorded_dst_ids, &dst_ids,
        "recorded dst_block_ids must equal caller's dst order"
    );

    Ok(())
}

// ============================================================================
// VeloSession loopback: wire-level pairing
// ============================================================================

fn new_velo_transport() -> Arc<velo::transports::tcp::TcpTransport> {
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

struct Side {
    velo: Arc<velo::Velo>,
    factory: Arc<VeloSessionFactory>,
    g2_manager: Arc<BlockManager<G2>>,
}

async fn build_side() -> Side {
    let velo = new_velo().await;
    let registry = TestRegistryBuilder::new().build();
    let g2_manager = Arc::new(
        TestManagerBuilder::<G2>::new()
            .block_count(32)
            .block_size(BLOCK_SIZE)
            .registry(registry.clone())
            .build(),
    );
    let leader = Arc::new(
        InstanceLeader::builder()
            .messenger(velo.messenger().clone())
            .registry(registry)
            .g2_manager(g2_manager.clone())
            .workers(vec![])
            .build()
            .expect("build leader"),
    );
    leader
        .register_handlers()
        .expect("register leader handlers");
    let factory =
        VeloSessionFactory::new(Arc::clone(&velo), leader, tokio::runtime::Handle::current());
    Side {
        velo,
        factory,
        g2_manager,
    }
}

async fn paired_sides() -> (Side, Side) {
    let h = build_side().await;
    let p = build_side().await;
    h.velo.register_peer(p.velo.peer_info()).unwrap();
    p.velo.register_peer(h.velo.peer_info()).unwrap();
    (h, p)
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn velo_loopback_frame_pull_carries_caller_order_hashes() -> Result<()> {
    let (h, p) = paired_sides().await;
    let session_id = uuid::Uuid::new_v4();

    // Holder via the concrete accessor so we can call
    // `test_inbound_pull_hashes()` for assertions.
    let h_session = h.factory.open_concrete(session_id)?;
    let h_endpoint = h_session.endpoint().expect("holder endpoint");
    let p_session = p
        .factory
        .attach(session_id, h.velo.instance_id(), h_endpoint)
        .await?;

    // Holder publishes 3 blocks A, B, C and seals both sets.
    let blocks = make_blocks(&h.g2_manager, 3, 200);
    let hashes_in_order: Vec<_> = blocks.iter().map(|b| b.sequence_hash()).collect();
    let (h_a, h_b, h_c) = (hashes_in_order[0], hashes_in_order[1], hashes_in_order[2]);
    h_session.commit(hashes_in_order.clone())?;
    h_session.make_available(blocks)?;
    h_session.finish_commits()?;
    h_session.finish_availability()?;

    // Puller drains streams to the sealed terminators.
    let mut commits = p_session.commits();
    while let Some(d) = tokio::time::timeout(Duration::from_secs(5), commits.next()).await? {
        if matches!(d, CommitDelta::Closed) {
            break;
        }
    }
    let mut avail = p_session.availability();
    while let Some(d) = tokio::time::timeout(Duration::from_secs(5), avail.next()).await? {
        if matches!(d, AvailabilityDelta::Drained) {
            break;
        }
    }

    // Puller calls pull with shuffled hashes. The future reaches the
    // wire (Frame::Pull sent + PullComplete received) and then errors
    // during `rdma_pull_with_opts` because this side has no workers.
    // That's the assertion vehicle — only the wire-level payload
    // (Frame::Pull's hash list) is under test here.
    let dst = alloc_mutable(&p.g2_manager, 3);
    let shuffled = vec![h_c, h_a, h_b];
    let result = tokio::time::timeout(
        Duration::from_secs(5),
        p_session.pull(shuffled.clone(), dst),
    )
    .await?;
    assert!(
        result.is_err(),
        "pull should error in the RDMA step (no workers configured)"
    );

    // Wire-level contract: the holder recorded Frame::Pull's hashes
    // in the caller-provided shuffled order. PullAck is NOT sent
    // (the puller's rdma_pull failed) so the entry remains.
    let inbound = h_session.test_inbound_pull_hashes();
    assert_eq!(
        inbound.len(),
        1,
        "exactly one authorized-but-unacked inbound pull"
    );
    assert_eq!(
        inbound[0], shuffled,
        "Frame::Pull hashes must equal caller-provided shuffled order"
    );

    Ok(())
}
