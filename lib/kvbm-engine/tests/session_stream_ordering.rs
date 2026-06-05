// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#![cfg(feature = "testing")]

//! Stream-order invariant tests.
//!
//! Contract: `Session::commits()` and `Session::availability()` yield
//! deltas in **wire-receive order** — and wire-receive order equals
//! the holder's call order on a single connection.
//!
//! Concretely: holder calling `commit([A,B,C])` then `commit([D,E])`
//! produces a stream whose concatenated `CommitDelta::Added` payloads
//! equal `[A,B,C,D,E]`. The pre-subscribe replay coalescer may merge
//! consecutive `Added` items into one (`velo.rs::build_commit_stream`,
//! `testing.rs::drain_commit_buffer`); live deltas after subscribe
//! stay separate. Either way, concatenation preserves order.
//!
//! Same shape for `make_available` and `availability()`.
//!
//! These tests exercise the multi-call axis. Single-call replay
//! coverage already exists in `velo_session_loopback.rs::
//! replay_on_late_subscribe_coalesces` and in the inline
//! `testing.rs::tests` module (`replay_on_subscribe_single_added`).

use std::sync::Arc;
use std::time::Duration;

use anyhow::Result;
use futures::StreamExt;
use kvbm_engine::G2;
use kvbm_engine::leader::InstanceLeader;
use kvbm_engine::p2p::session::{
    AvailabilityDelta, CommitDelta, CommittedBlock, MockSessionFactory, Session, SessionFactory,
    VeloSessionFactory,
};
use kvbm_engine::testing::managers::{TestManagerBuilder, TestRegistryBuilder};
use kvbm_engine::testing::token_blocks::create_token_sequence;
use kvbm_logical::SequenceHash;
use kvbm_logical::blocks::ImmutableBlock;
use kvbm_logical::manager::BlockManager;
use velo::InstanceId;
use velo::transports::tcp::TcpTransportBuilder;

const BLOCK_SIZE: usize = 16;
const STREAM_TIMEOUT: Duration = Duration::from_secs(5);

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

// ============================================================================
// Stream-drain helpers — return the concatenated payload up to (and
// including consumption of) the closed terminator.
// ============================================================================

async fn drain_commits_to_closed<S>(mut stream: S) -> Result<Vec<SequenceHash>>
where
    S: futures::Stream<Item = CommitDelta> + Unpin,
{
    let mut out: Vec<SequenceHash> = Vec::new();
    loop {
        let next = tokio::time::timeout(STREAM_TIMEOUT, stream.next())
            .await?
            .ok_or_else(|| anyhow::anyhow!("commits stream ended before Closed"))?;
        match next {
            CommitDelta::Added(hs) => out.extend(hs),
            CommitDelta::Closed => return Ok(out),
        }
    }
}

async fn drain_avail_to_drained<S>(mut stream: S) -> Result<Vec<CommittedBlock>>
where
    S: futures::Stream<Item = AvailabilityDelta> + Unpin,
{
    let mut out: Vec<CommittedBlock> = Vec::new();
    loop {
        let next = tokio::time::timeout(STREAM_TIMEOUT, stream.next())
            .await?
            .ok_or_else(|| anyhow::anyhow!("availability stream ended before Drained"))?;
        match next {
            AvailabilityDelta::Available(bs) => out.extend(bs),
            AvailabilityDelta::Drained => return Ok(out),
        }
    }
}

// ============================================================================
// MockSession paired-mode: order preservation across multiple
// commit / make_available calls.
// ============================================================================

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn mock_paired_commits_preserves_order_across_calls() -> Result<()> {
    let (h_factory, p_factory) = MockSessionFactory::make_paired();
    let session_id = uuid::Uuid::new_v4();

    let h_session = h_factory.open(session_id)?;
    let h_endpoint = h_session.endpoint().expect("holder endpoint");
    let _p_dyn = p_factory
        .attach(session_id, InstanceId::new_v4(), h_endpoint)
        .await?;
    let p_session = p_factory.last_attached().expect("puller registered");

    let g2 = make_g2_manager();
    // Two separate commit() calls: [A,B,C] then [D,E].
    let batch_abc = make_blocks(&g2, 3, 100);
    let batch_de = make_blocks(&g2, 2, 200);
    let abc: Vec<_> = batch_abc.iter().map(|b| b.sequence_hash()).collect();
    let de: Vec<_> = batch_de.iter().map(|b| b.sequence_hash()).collect();
    let expected: Vec<SequenceHash> = abc.iter().chain(de.iter()).copied().collect();

    h_session.commit(abc)?;
    h_session.commit(de)?;
    h_session.finish_commits()?;

    let commits = p_session.commits();
    let drained = drain_commits_to_closed(commits).await?;
    assert_eq!(
        drained, expected,
        "concatenated Added payloads must equal holder's commit order across multiple calls"
    );

    Ok(())
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn mock_paired_availability_preserves_order_across_calls() -> Result<()> {
    let (h_factory, p_factory) = MockSessionFactory::make_paired();
    let session_id = uuid::Uuid::new_v4();

    let h_session = h_factory.open(session_id)?;
    let h_endpoint = h_session.endpoint().expect("holder endpoint");
    let _p_dyn = p_factory
        .attach(session_id, InstanceId::new_v4(), h_endpoint)
        .await?;
    let p_session = p_factory.last_attached().expect("puller registered");

    let g2 = make_g2_manager();
    let batch_abc = make_blocks(&g2, 3, 300);
    let batch_de = make_blocks(&g2, 2, 400);
    let all_hashes: Vec<_> = batch_abc
        .iter()
        .chain(batch_de.iter())
        .map(|b| b.sequence_hash())
        .collect();
    let expected_hashes: Vec<SequenceHash> = all_hashes.clone();

    // Must commit before make_available (available ⊆ committed
    // invariant).
    h_session.commit(all_hashes)?;
    h_session.make_available(batch_abc)?;
    h_session.make_available(batch_de)?;
    h_session.finish_availability()?;

    let avail = p_session.availability();
    let drained = drain_avail_to_drained(avail).await?;
    let drained_hashes: Vec<_> = drained.iter().map(|b| b.hash).collect();
    assert_eq!(
        drained_hashes, expected_hashes,
        "concatenated Available payloads must equal holder's make_available order"
    );

    Ok(())
}

// ============================================================================
// VeloSession loopback: order preservation across multiple
// commit / make_available calls.
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
    leader.register_handlers().expect("register handlers");
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
async fn velo_loopback_commits_preserves_order_across_calls() -> Result<()> {
    let (h, p) = paired_sides().await;
    let session_id = uuid::Uuid::new_v4();

    let h_session = h.factory.open(session_id)?;
    let h_endpoint = h_session.endpoint().expect("holder endpoint");
    let p_session = p
        .factory
        .attach(session_id, h.velo.instance_id(), h_endpoint)
        .await?;

    let batch_abc = make_blocks(&h.g2_manager, 3, 500);
    let batch_de = make_blocks(&h.g2_manager, 2, 600);
    let abc: Vec<_> = batch_abc.iter().map(|b| b.sequence_hash()).collect();
    let de: Vec<_> = batch_de.iter().map(|b| b.sequence_hash()).collect();
    let expected: Vec<SequenceHash> = abc.iter().chain(de.iter()).copied().collect();

    h_session.commit(abc)?;
    h_session.commit(de)?;
    h_session.finish_commits()?;

    // Slight delay so frames have settled on the puller's inbound
    // demux before we subscribe — this puts the test on the
    // pre-subscribe replay path (coalesced into one Added) which is
    // the order-sensitive case worth pinning.
    tokio::time::sleep(Duration::from_millis(200)).await;

    let commits = p_session.commits();
    let drained = drain_commits_to_closed(commits).await?;
    assert_eq!(
        drained, expected,
        "concatenated Added payloads must equal holder's commit order across multiple calls"
    );

    Ok(())
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn velo_loopback_availability_preserves_order_across_calls() -> Result<()> {
    let (h, p) = paired_sides().await;
    let session_id = uuid::Uuid::new_v4();

    let h_session = h.factory.open(session_id)?;
    let h_endpoint = h_session.endpoint().expect("holder endpoint");
    let p_session = p
        .factory
        .attach(session_id, h.velo.instance_id(), h_endpoint)
        .await?;

    let batch_abc = make_blocks(&h.g2_manager, 3, 700);
    let batch_de = make_blocks(&h.g2_manager, 2, 800);
    let all_hashes: Vec<_> = batch_abc
        .iter()
        .chain(batch_de.iter())
        .map(|b| b.sequence_hash())
        .collect();
    let expected_hashes: Vec<SequenceHash> = all_hashes.clone();

    h_session.commit(all_hashes)?;
    h_session.make_available(batch_abc)?;
    h_session.make_available(batch_de)?;
    h_session.finish_availability()?;

    tokio::time::sleep(Duration::from_millis(200)).await;

    let avail = p_session.availability();
    let drained = drain_avail_to_drained(avail).await?;
    let drained_hashes: Vec<_> = drained.iter().map(|b| b.hash).collect();
    assert_eq!(
        drained_hashes, expected_hashes,
        "concatenated Available payloads must equal holder's make_available order"
    );

    Ok(())
}
