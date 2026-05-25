// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#![cfg(feature = "testing")]

//! Bidirectional invariant test.
//!
//! Contract: the [`Session`] trait surface is role-symmetric. Both
//! sides — the one that called `factory.open(...)` AND the one that
//! called `factory.attach(...)` — may concurrently:
//!
//! * commit hashes, mark blocks available, finish both sets
//! * drain peer's `commits()` / `availability()` streams
//! * observe a `Sealed` snapshot once peer finishes
//! * call `pull(...)` on peer's hashes
//!
//! There is no implicit "holder commits before puller pulls"
//! sequencing in the state machine. A bug that special-cases one
//! role into a publish-only or consume-only path fails this test.
//!
//! Out of scope: actual RDMA happy-path data landing. The velo
//! loopback test runs with `workers(vec![])`; `pull(...)` reaches
//! the wire (Frame::Pull → PullComplete) and errors during
//! `rdma_pull_with_opts`. That progression is itself proof that
//! the consume path is reachable from both sides.

use std::sync::Arc;
use std::time::Duration;

use anyhow::Result;
use futures::StreamExt;
use kvbm_engine::G2;
use kvbm_engine::leader::InstanceLeader;
use kvbm_engine::p2p::session::{
    AvailabilityDelta, CommitDelta, CommittedBlock, SessionFactory, VeloSessionFactory,
};
use kvbm_engine::testing::managers::{TestManagerBuilder, TestRegistryBuilder};
use kvbm_engine::testing::token_blocks::create_token_sequence;
use kvbm_logical::SequenceHash;
use kvbm_logical::blocks::ImmutableBlock;
use kvbm_logical::manager::BlockManager;
use velo::transports::tcp::TcpTransportBuilder;

const BLOCK_SIZE: usize = 16;
const STREAM_TIMEOUT: Duration = Duration::from_secs(5);

// ============================================================================
// Velo loopback infrastructure (same shape as other integration tests)
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
    let a = build_side().await;
    let b = build_side().await;
    a.velo.register_peer(b.velo.peer_info()).unwrap();
    b.velo.register_peer(a.velo.peer_info()).unwrap();
    (a, b)
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
// Bidirectional publish + drain + pull
// ============================================================================

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn velo_loopback_bidirectional_publish_drain_pull() -> Result<()> {
    let (a, b) = paired_sides().await;
    let session_id = uuid::Uuid::new_v4();

    // A opens, B attaches.
    let a_session = a.factory.open(session_id)?;
    let a_endpoint = a_session.endpoint().expect("A endpoint");
    let b_session = b
        .factory
        .attach(session_id, a.velo.instance_id(), a_endpoint)
        .await?;

    // Both sides materialize blocks to publish — symmetric setup,
    // no holder/puller distinction.
    let blocks_a = make_blocks(&a.g2_manager, 2, 1_000);
    let blocks_b = make_blocks(&b.g2_manager, 2, 2_000);
    let hashes_a: Vec<_> = blocks_a.iter().map(|b| b.sequence_hash()).collect();
    let hashes_b: Vec<_> = blocks_b.iter().map(|b| b.sequence_hash()).collect();

    // ------------------------------------------------------------
    // Phase 1 — concurrent publish on both sides.
    // ------------------------------------------------------------
    let a_pub = {
        let a_session = Arc::clone(&a_session);
        let hashes_a = hashes_a.clone();
        async move {
            a_session.commit(hashes_a)?;
            a_session.make_available(blocks_a)?;
            a_session.finish_commits()?;
            a_session.finish_availability()?;
            anyhow::Ok(())
        }
    };
    let b_pub = {
        let b_session = Arc::clone(&b_session);
        let hashes_b = hashes_b.clone();
        async move {
            b_session.commit(hashes_b)?;
            b_session.make_available(blocks_b)?;
            b_session.finish_commits()?;
            b_session.finish_availability()?;
            anyhow::Ok(())
        }
    };
    tokio::try_join!(a_pub, b_pub)?;

    // ------------------------------------------------------------
    // Phase 2 — concurrent drain on both sides. Each side must see
    // the OTHER side's published hashes in the OTHER side's call
    // order.
    // ------------------------------------------------------------
    let a_drain = {
        let a_session = Arc::clone(&a_session);
        async move {
            let commits = a_session.commits();
            let drained_c = drain_commits_to_closed(commits).await?;
            let avail = a_session.availability();
            let drained_a = drain_avail_to_drained(avail).await?;
            anyhow::Ok((drained_c, drained_a))
        }
    };
    let b_drain = {
        let b_session = Arc::clone(&b_session);
        async move {
            let commits = b_session.commits();
            let drained_c = drain_commits_to_closed(commits).await?;
            let avail = b_session.availability();
            let drained_a = drain_avail_to_drained(avail).await?;
            anyhow::Ok((drained_c, drained_a))
        }
    };
    let ((a_seen_commits, a_seen_avail), (b_seen_commits, b_seen_avail)) =
        tokio::try_join!(a_drain, b_drain)?;

    assert_eq!(
        a_seen_commits, hashes_b,
        "A must see B's commits in B's call order"
    );
    assert_eq!(
        b_seen_commits, hashes_a,
        "B must see A's commits in A's call order"
    );
    let a_avail_hashes: Vec<_> = a_seen_avail.iter().map(|cb| cb.hash).collect();
    let b_avail_hashes: Vec<_> = b_seen_avail.iter().map(|cb| cb.hash).collect();
    assert_eq!(
        a_avail_hashes, hashes_b,
        "A must see B's availability in B's call order"
    );
    assert_eq!(
        b_avail_hashes, hashes_a,
        "B must see A's availability in A's call order"
    );

    // After both terminators arrived, snapshots on both sides must
    // be Sealed. This pins the Phase A enum's seal semantics over
    // the bidirectional path.
    assert!(
        a_session.peer_committed().is_sealed(),
        "A's peer_committed must be Sealed after B finished commits"
    );
    assert!(
        a_session.peer_available().is_sealed(),
        "A's peer_available must be Sealed after B finished availability"
    );
    assert!(
        b_session.peer_committed().is_sealed(),
        "B's peer_committed must be Sealed after A finished commits"
    );
    assert!(
        b_session.peer_available().is_sealed(),
        "B's peer_available must be Sealed after A finished availability"
    );

    // ------------------------------------------------------------
    // Phase 3 — concurrent pulls on both sides. Each side pulls
    // the other's hashes. Both will reach the wire (Frame::Pull →
    // PullComplete) and then error during `rdma_pull_with_opts`
    // because the test infra has no workers. That progression is
    // the proof the consume path is reachable from both sides
    // simultaneously; an asymmetry that locks one side out would
    // surface as a synchronous error before the wire is touched.
    // ------------------------------------------------------------
    let a_dst = a.g2_manager.allocate_blocks(2).expect("A alloc dst");
    let b_dst = b.g2_manager.allocate_blocks(2).expect("B alloc dst");

    let (a_pull_result, b_pull_result) = tokio::join!(
        tokio::time::timeout(
            Duration::from_secs(5),
            a_session.pull(hashes_b.clone(), a_dst)
        ),
        tokio::time::timeout(
            Duration::from_secs(5),
            b_session.pull(hashes_a.clone(), b_dst)
        ),
    );
    let a_err = a_pull_result?.expect_err("A's pull must reach rdma step and error");
    let b_err = b_pull_result?.expect_err("B's pull must reach rdma step and error");
    // Errors should come from rdma_pull_with_opts, NOT from a
    // synchronous role-asymmetry guard. Confirm by checking the
    // error chain contains rdma context.
    let a_msg = format!("{a_err:#}");
    let b_msg = format!("{b_err:#}");
    assert!(
        a_msg.contains("rdma_pull_with_opts"),
        "A's pull must error in the rdma step, not earlier: {a_msg}"
    );
    assert!(
        b_msg.contains("rdma_pull_with_opts"),
        "B's pull must error in the rdma step, not earlier: {b_msg}"
    );

    Ok(())
}
