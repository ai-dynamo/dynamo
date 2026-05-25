// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#![cfg(feature = "testing")]

//! Seal-immutability invariant tests.
//!
//! Contract: once `peer_committed()` returns `PeerCommitted::Sealed(v)`,
//! the inner `v` is final. Subsequent snapshot calls return the same
//! set — never growing. Same for `peer_available()` → `Sealed`.
//!
//! Two enforcement layers:
//!
//! 1. **Holder-side**: `commit()` errors after `finish_commits()`;
//!    `make_available()` errors after `finish_availability()`. A
//!    well-behaved holder cannot enqueue post-terminator frames.
//!
//! 2. **Puller-side dispatch**: a `Frame::Commit` arriving after
//!    `Frame::CommitsClosed` (e.g. from a buggy or older peer) is
//!    dropped with a `tracing::error!`; `peer_committed` is not
//!    mutated. Same for `Frame::Available` after `Frame::Drained`.
//!
//! This file pins both layers.

use std::sync::Arc;
use std::time::Duration;

use anyhow::Result;
use futures::StreamExt;
use kvbm_engine::G2;
use kvbm_engine::leader::InstanceLeader;
use kvbm_engine::p2p::session::{
    AvailabilityDelta, CommitDelta, Frame, MockSessionFactory, PeerAvailable, PeerCommitted,
    Session, SessionFactory, VeloSessionFactory,
};
use kvbm_engine::testing::managers::{TestManagerBuilder, TestRegistryBuilder};
use kvbm_engine::testing::token_blocks::create_token_sequence;
use kvbm_logical::blocks::ImmutableBlock;
use kvbm_logical::manager::BlockManager;
use velo::InstanceId;
use velo::transports::tcp::TcpTransportBuilder;

const BLOCK_SIZE: usize = 16;

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
// Mock holder-side: commit/make_available error after finish_*
// ============================================================================

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn mock_commit_after_finish_commits_errors() -> Result<()> {
    let (h_factory, p_factory) = MockSessionFactory::make_paired();
    let session_id = uuid::Uuid::new_v4();

    let h_session = h_factory.open(session_id)?;
    let h_endpoint = h_session.endpoint().expect("holder endpoint");
    let _ = p_factory
        .attach(session_id, InstanceId::new_v4(), h_endpoint)
        .await?;

    let g2 = make_g2_manager();
    let batch_a = make_blocks(&g2, 1, 100);
    let batch_b = make_blocks(&g2, 1, 200);
    let hashes_a: Vec<_> = batch_a.iter().map(|b| b.sequence_hash()).collect();
    let hashes_b: Vec<_> = batch_b.iter().map(|b| b.sequence_hash()).collect();

    h_session.commit(hashes_a)?;
    h_session.finish_commits()?;

    let err = h_session
        .commit(hashes_b)
        .expect_err("commit after finish_commits must error");
    assert!(
        err.to_string().contains("finish_commits"),
        "error must mention finish_commits: {err}"
    );

    Ok(())
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn mock_make_available_after_finish_availability_errors() -> Result<()> {
    let (h_factory, p_factory) = MockSessionFactory::make_paired();
    let session_id = uuid::Uuid::new_v4();

    let h_session = h_factory.open(session_id)?;
    let h_endpoint = h_session.endpoint().expect("holder endpoint");
    let _ = p_factory
        .attach(session_id, InstanceId::new_v4(), h_endpoint)
        .await?;

    let g2 = make_g2_manager();
    let batch_a = make_blocks(&g2, 1, 300);
    let batch_b = make_blocks(&g2, 1, 400);
    let hashes_a: Vec<_> = batch_a.iter().map(|b| b.sequence_hash()).collect();
    let hashes_b: Vec<_> = batch_b.iter().map(|b| b.sequence_hash()).collect();

    // Commit both up front (available ⊆ committed precondition).
    let mut all = hashes_a.clone();
    all.extend(hashes_b.clone());
    h_session.commit(all)?;
    h_session.make_available(batch_a)?;
    h_session.finish_availability()?;

    let err = h_session
        .make_available(batch_b)
        .expect_err("make_available after finish_availability must error");
    assert!(
        err.to_string().contains("finish_availability"),
        "error must mention finish_availability: {err}"
    );

    Ok(())
}

// ============================================================================
// Velo loopback infrastructure
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

// ============================================================================
// Velo holder-side: commit/make_available error after finish_*
// ============================================================================

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn velo_commit_after_finish_commits_errors() -> Result<()> {
    let h = build_side().await;
    let session_id = uuid::Uuid::new_v4();
    let h_session = h.factory.open_concrete(session_id)?;

    let batch_a = make_blocks(&h.g2_manager, 1, 500);
    let batch_b = make_blocks(&h.g2_manager, 1, 600);
    let hashes_a: Vec<_> = batch_a.iter().map(|b| b.sequence_hash()).collect();
    let hashes_b: Vec<_> = batch_b.iter().map(|b| b.sequence_hash()).collect();

    h_session.commit(hashes_a)?;
    h_session.finish_commits()?;

    let err = h_session
        .commit(hashes_b)
        .expect_err("commit after finish_commits must error");
    assert!(
        err.to_string().contains("finish_commits"),
        "error must mention finish_commits: {err}"
    );

    Ok(())
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn velo_make_available_after_finish_availability_errors() -> Result<()> {
    let h = build_side().await;
    let session_id = uuid::Uuid::new_v4();
    let h_session = h.factory.open_concrete(session_id)?;

    let batch_a = make_blocks(&h.g2_manager, 1, 700);
    let batch_b = make_blocks(&h.g2_manager, 1, 800);
    let hashes_a: Vec<_> = batch_a.iter().map(|b| b.sequence_hash()).collect();
    let hashes_b: Vec<_> = batch_b.iter().map(|b| b.sequence_hash()).collect();

    let mut all = hashes_a.clone();
    all.extend(hashes_b.clone());
    h_session.commit(all)?;
    h_session.make_available(batch_a)?;
    h_session.finish_availability()?;

    let err = h_session
        .make_available(batch_b)
        .expect_err("make_available after finish_availability must error");
    assert!(
        err.to_string().contains("finish_availability"),
        "error must mention finish_availability: {err}"
    );

    Ok(())
}

// ============================================================================
// Velo puller-side dispatch: post-terminator frames must not mutate snapshot
// ============================================================================

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn velo_sealed_committed_does_not_grow_under_post_closed_commit_frame() -> Result<()> {
    let h = build_side().await;
    let session_id = uuid::Uuid::new_v4();
    // Use the holder side as the snapshot subject — it has the
    // test_inject_inbound_frame accessor and can play the role of
    // "puller observing inbound Frame::Commit + Frame::CommitsClosed".
    let h_session = h.factory.open_concrete(session_id)?;

    // Forge inbound Frame::Commit for [X], then Frame::CommitsClosed,
    // then a violating Frame::Commit for [Y] (post-terminator).
    let blocks = make_blocks(&h.g2_manager, 3, 900);
    let hash_x = blocks[0].sequence_hash();
    let hash_y = blocks[1].sequence_hash();

    h_session.test_inject_inbound_frame(Frame::Commit {
        hashes: vec![hash_x],
    });
    h_session.test_inject_inbound_frame(Frame::CommitsClosed);

    // Snapshot now must be Sealed([X]).
    match h_session.peer_committed() {
        PeerCommitted::Sealed(v) => {
            assert_eq!(v, vec![hash_x], "Sealed must contain only X")
        }
        other => panic!("expected Sealed, got {other:?}"),
    }

    // Violating post-terminator Frame::Commit. After the fix, this
    // must be dropped — peer_committed must remain Sealed([X]).
    h_session.test_inject_inbound_frame(Frame::Commit {
        hashes: vec![hash_y],
    });

    // Small settle so the dispatch finishes.
    tokio::time::sleep(Duration::from_millis(50)).await;

    match h_session.peer_committed() {
        PeerCommitted::Sealed(v) => {
            assert_eq!(
                v,
                vec![hash_x],
                "post-CommitsClosed Frame::Commit must NOT grow Sealed snapshot"
            );
        }
        other => panic!("expected Sealed after violating frame, got {other:?}"),
    }

    Ok(())
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn velo_sealed_available_does_not_grow_under_post_drained_avail_frame() -> Result<()> {
    let (h, p) = paired_sides().await;
    let session_id = uuid::Uuid::new_v4();
    // We need an attached pair so peer_instance_id is set and
    // Frame::Available payloads are coherent. Use h as the subject.
    let h_session = h.factory.open_concrete(session_id)?;
    let h_endpoint = h_session.endpoint().expect("holder endpoint");
    let _p_session = p
        .factory
        .attach(session_id, h.velo.instance_id(), h_endpoint)
        .await?;

    let blocks = make_blocks(&h.g2_manager, 3, 1_000);
    let block_x = kvbm_engine::p2p::session::CommittedBlock {
        hash: blocks[0].sequence_hash(),
        peer_block_id: blocks[0].block_id(),
    };
    let block_y = kvbm_engine::p2p::session::CommittedBlock {
        hash: blocks[1].sequence_hash(),
        peer_block_id: blocks[1].block_id(),
    };

    h_session.test_inject_inbound_frame(Frame::Available {
        blocks: vec![block_x.clone()],
    });
    h_session.test_inject_inbound_frame(Frame::Drained);

    match h_session.peer_available() {
        PeerAvailable::Sealed(v) => {
            assert_eq!(v.len(), 1, "Sealed must contain only X");
            assert_eq!(v[0].hash, block_x.hash);
        }
        other => panic!("expected Sealed, got {other:?}"),
    }

    h_session.test_inject_inbound_frame(Frame::Available {
        blocks: vec![block_y],
    });

    tokio::time::sleep(Duration::from_millis(50)).await;

    match h_session.peer_available() {
        PeerAvailable::Sealed(v) => {
            assert_eq!(
                v.len(),
                1,
                "post-Drained Frame::Available must NOT grow Sealed snapshot"
            );
            assert_eq!(v[0].hash, block_x.hash);
        }
        other => panic!("expected Sealed after violating frame, got {other:?}"),
    }

    Ok(())
}

// ============================================================================
// End-to-end Sealed stability: drain → Sealed → snapshot is stable
// ============================================================================

// ============================================================================
// Concurrent commit + finish_commits must not lose accepted commits on the wire
// ============================================================================
//
// Race the holder side: thread A loops calling commit(); thread B calls
// finish_commits() after a small delay. Every commit() that returned Ok must
// be visible in the puller's drained commit stream. A TOCTOU between
// commit's flag-check and its enqueue would silently lose accepted commits
// (puller's defense-in-depth guard would drop the post-CommitsClosed
// Frame::Commit, but the holder already returned Ok to its caller).

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn velo_concurrent_commit_and_finish_commits_no_silent_drops() -> Result<()> {
    let (h, p) = paired_sides().await;
    let session_id = uuid::Uuid::new_v4();

    let h_session = h.factory.open_concrete(session_id)?;
    let h_endpoint = h_session.endpoint().expect("holder endpoint");
    let p_session = p
        .factory
        .attach(session_id, h.velo.instance_id(), h_endpoint)
        .await?;

    let blocks = make_blocks(&h.g2_manager, 32, 5_000);
    let hashes: Vec<_> = blocks.iter().map(|b| b.sequence_hash()).collect();

    let h_a = Arc::clone(&h_session);
    let hashes_a = hashes.clone();
    let commit_task = tokio::task::spawn_blocking(move || {
        // Sync loop on a blocking thread — exposes the race window
        // between commit's flag-check and its enqueue without tokio's
        // cooperative scheduling masking it.
        let mut accepted: Vec<_> = Vec::new();
        for h in hashes_a {
            match h_a.commit(vec![h]) {
                Ok(()) => accepted.push(h),
                Err(_) => break,
            }
        }
        accepted
    });

    let h_b = Arc::clone(&h_session);
    let finish_task = tokio::task::spawn_blocking(move || {
        // Brief sleep so a handful of commits land first, then race.
        std::thread::sleep(Duration::from_micros(500));
        h_b.finish_commits().expect("finish_commits");
    });

    let accepted = commit_task.await?;
    finish_task.await?;

    // Wait briefly for in-flight frames to flush.
    tokio::time::sleep(Duration::from_millis(200)).await;

    // Every commit that returned Ok must be visible in the puller's drained
    // set.
    let drained = {
        let mut commits = p_session.commits();
        let mut out = Vec::new();
        loop {
            let next = tokio::time::timeout(Duration::from_secs(5), commits.next())
                .await?
                .expect("commits stream");
            match next {
                CommitDelta::Added(hs) => out.extend(hs),
                CommitDelta::Closed => break,
            }
        }
        out
    };
    let drained_set: std::collections::HashSet<_> = drained.iter().copied().collect();
    for h in &accepted {
        assert!(
            drained_set.contains(h),
            "puller missing accepted commit {h:?} \
             (race: TOCTOU between commit's flag-check and enqueue lost data)"
        );
    }

    Ok(())
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn velo_loopback_sealed_snapshot_is_stable_across_calls() -> Result<()> {
    let (h, p) = paired_sides().await;
    let session_id = uuid::Uuid::new_v4();

    let h_session = h.factory.open_concrete(session_id)?;
    let h_endpoint = h_session.endpoint().expect("holder endpoint");
    let p_session = p
        .factory
        .attach(session_id, h.velo.instance_id(), h_endpoint)
        .await?;

    let blocks = make_blocks(&h.g2_manager, 2, 1_100);
    let hashes: Vec<_> = blocks.iter().map(|b| b.sequence_hash()).collect();
    h_session.commit(hashes.clone())?;
    h_session.make_available(blocks)?;
    h_session.finish_commits()?;
    h_session.finish_availability()?;

    // Drain puller's streams so the seal flags are set.
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

    // Now two snapshots back-to-back must agree.
    let snap1_c = p_session.peer_committed();
    let snap2_c = p_session.peer_committed();
    assert_eq!(snap1_c, snap2_c, "Sealed peer_committed must be stable");
    assert!(snap1_c.is_sealed());

    let snap1_a = p_session.peer_available();
    let snap2_a = p_session.peer_available();
    assert_eq!(snap1_a, snap2_a, "Sealed peer_available must be stable");
    assert!(snap1_a.is_sealed());

    Ok(())
}
