// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#![cfg(feature = "testing")]

//! Integration tests for `VeloSession` / `VeloSessionFactory` over a real
//! velo TCP loopback pair.
//!
//! Coverage gaps these tests close (cd-session-refactor plan §4b):
//!   * Replay-on-late-subscribe of `commit` + `make_available` deltas.
//!   * `close()` from holder propagates terminators (`CommitDelta::Closed`,
//!     `AvailabilityDelta::Drained`) + `LifecycleEvent::Detached` to the
//!     puller — closes the Mock/Velo behavioral parity gap (MockSession
//!     pushes terminators on close; VeloSession needs to match).
//!   * `pull(...)` with a hash not in `peer_available` resolves
//!     synchronously to `Err` without sending any frame on the wire.
//!
//! Out of scope here (covered by the rewritten `cd_loopback`):
//!   * `pull(...)` happy path resolving `Ok` after RDMA — needs a worker-
//!     equipped `InstanceLeader` pair (NIXL/CUDA). The pin-release-on-
//!     PullAck invariant is asserted indirectly by `cd_loopback` because
//!     the wrapper layer drives the full round-trip there.

use std::sync::Arc;
use std::time::Duration;

use anyhow::Result;
use futures::StreamExt;
use kvbm_engine::G2;
use kvbm_engine::disagg::session::{
    AvailabilityDelta, CommitDelta, Frame, LifecycleEvent, Session, SessionFactory,
    VeloSessionFactory,
};
use kvbm_engine::leader::InstanceLeader;
use kvbm_engine::testing::managers::{TestManagerBuilder, TestRegistryBuilder};
use kvbm_engine::testing::token_blocks::create_token_sequence;
use kvbm_logical::blocks::ImmutableBlock;
use kvbm_logical::manager::BlockManager;
use velo::backend::tcp::TcpTransportBuilder;

const BLOCK_SIZE: usize = 16;

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
    // No workers — these tests don't drive RDMA pulls, just the
    // wire protocol around session state.
    let leader = Arc::new(
        InstanceLeader::builder()
            .messenger(velo.messenger().clone())
            .registry(registry)
            .g2_manager(g2_manager.clone())
            .workers(vec![])
            .remote_leaders(vec![])
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

fn make_blocks(
    g2: &Arc<BlockManager<G2>>,
    count: usize,
    start_token: u32,
) -> Vec<ImmutableBlock<G2>> {
    let token_sequence = create_token_sequence(count, BLOCK_SIZE, start_token);
    let mutables = g2.allocate_blocks(count).expect("alloc");
    let completes: Vec<_> = mutables
        .into_iter()
        .zip(token_sequence.blocks().iter())
        .map(|(m, tb)| m.complete(tb).expect("complete"))
        .collect();
    g2.register_blocks(completes)
}

// ============================================================================
// Case: replay-on-late-subscribe coalesces commits + availability
// ============================================================================

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn replay_on_late_subscribe_coalesces() -> Result<()> {
    let (h, p) = paired_sides().await;
    let session_id = uuid::Uuid::new_v4();

    let h_session = h.factory.open(session_id)?;
    let h_endpoint = h_session.endpoint().expect("holder endpoint");
    let p_session = p
        .factory
        .attach(session_id, h.velo.instance_id(), h_endpoint)
        .await?;

    // Holder publishes eagerly; puller does NOT subscribe yet.
    let blocks = make_blocks(&h.g2_manager, 3, 100);
    let hashes: Vec<_> = blocks.iter().map(|b| b.sequence_hash()).collect();
    h_session.commit(hashes.clone())?;
    h_session.make_available(blocks)?;
    h_session.finish_commits()?;
    h_session.finish_availability()?;

    // Wait for frames to flow + buffer on the puller side.
    tokio::time::sleep(Duration::from_millis(200)).await;

    let mut commits = p_session.commits();
    match tokio::time::timeout(Duration::from_secs(30), commits.next())
        .await?
        .expect("first commit delta")
    {
        CommitDelta::Added(hs) => {
            assert_eq!(hs.len(), 3, "all 3 hashes coalesced into one Added");
            assert_eq!(hs, hashes);
        }
        other => panic!("expected Added, got {other:?}"),
    }
    let next = tokio::time::timeout(Duration::from_secs(30), commits.next()).await?;
    assert!(
        matches!(next, Some(CommitDelta::Closed)),
        "expected Closed terminator, got {next:?}"
    );

    let mut avail = p_session.availability();
    match tokio::time::timeout(Duration::from_secs(30), avail.next())
        .await?
        .expect("first avail delta")
    {
        AvailabilityDelta::Available(bs) => {
            assert_eq!(bs.len(), 3, "all 3 blocks coalesced into one Available");
        }
        other => panic!("expected Available, got {other:?}"),
    }
    let next = tokio::time::timeout(Duration::from_secs(30), avail.next()).await?;
    assert!(
        matches!(next, Some(AvailabilityDelta::Drained)),
        "expected Drained terminator, got {next:?}"
    );

    Ok(())
}

// ============================================================================
// Case: close() from holder terminates streams + emits Detached on puller
// ============================================================================

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn close_from_holder_terminates_streams() -> Result<()> {
    let (h, p) = paired_sides().await;
    let session_id = uuid::Uuid::new_v4();

    let h_session = h.factory.open(session_id)?;
    let h_endpoint = h_session.endpoint().expect("holder endpoint");
    let p_session = p
        .factory
        .attach(session_id, h.velo.instance_id(), h_endpoint)
        .await?;

    // Wait for the bidi link to settle by observing the holder's
    // Attached event (only the holder sees Attached — the puller
    // sends Attach but never receives one).
    let mut h_lifecycle = h_session.lifecycle();
    let evt = tokio::time::timeout(Duration::from_secs(30), h_lifecycle.next())
        .await?
        .expect("holder attached lifecycle");
    assert!(
        matches!(evt, LifecycleEvent::Attached { .. }),
        "holder got {evt:?}"
    );

    let mut lifecycle = p_session.lifecycle();
    let mut commits = p_session.commits();
    let mut avail = p_session.availability();

    // Holder closes — must imply finish_commits + finish_availability
    // (per Session trait doc) so puller sees stream terminators.
    h_session.close(Some("test-close".to_string()));

    // Detached arrives on puller.
    let evt = tokio::time::timeout(Duration::from_secs(30), lifecycle.next())
        .await?
        .expect("detached lifecycle");
    assert!(
        matches!(evt, LifecycleEvent::Detached { .. }),
        "got {evt:?}"
    );

    // Commits stream sees Closed within timeout.
    let next = tokio::time::timeout(Duration::from_secs(30), commits.next()).await?;
    assert!(
        matches!(next, Some(CommitDelta::Closed)),
        "expected Closed on commits, got {next:?}"
    );

    // Availability stream sees Drained within timeout.
    let next = tokio::time::timeout(Duration::from_secs(30), avail.next()).await?;
    assert!(
        matches!(next, Some(AvailabilityDelta::Drained)),
        "expected Drained on availability, got {next:?}"
    );

    Ok(())
}

// ============================================================================
// Case: Frame::PullAck on the holder's inbound dispatch drops pins
// ============================================================================
//
// Asserts the load-bearing invariant required by plan §5 stage-1 review:
// after a holder receives Frame::PullAck for a pull_id it previously
// authorized via Frame::Pull, the pins for the corresponding hashes are
// dropped from `available_pins`. Forges the inbound frames directly via
// the test-only `test_inject_inbound_frame` helper so the test doesn't
// need real RDMA workers.

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn pull_ack_drops_holder_pins() -> Result<()> {
    let h = build_side().await;

    let session_id = uuid::Uuid::new_v4();
    let h_session = h.factory.open_concrete(session_id)?;

    // Holder commits + makes-available 1 hash. After this the
    // hash is pinned in `available_pins`.
    let blocks = make_blocks(&h.g2_manager, 1, 200);
    let hashes: Vec<_> = blocks.iter().map(|b| b.sequence_hash()).collect();
    h_session.commit(hashes.clone())?;
    h_session.make_available(blocks)?;
    assert_eq!(
        h_session.test_available_pin_count(),
        1,
        "make_available should pin the block"
    );

    // Forge inbound Frame::Pull on holder's dispatch. This
    // records the pull_id → hashes mapping in `inbound_pulls`
    // and (via dispatch_frame) emits Frame::PullComplete on the
    // outbound — but the outbound has no peer attached yet, so
    // the PullComplete send fails silently. That's fine for
    // this test; we only care that the holder records the pull.
    let pull_id: u64 = 42;
    h_session.test_inject_inbound_frame(Frame::Pull {
        pull_id,
        hashes: hashes.clone(),
    });

    // Forge inbound Frame::PullAck — this is what plan §5
    // promises drops the pins. Assert the pin-release
    // invariant directly.
    h_session.test_inject_inbound_frame(Frame::PullAck { pull_id });

    assert_eq!(
        h_session.test_available_pin_count(),
        0,
        "PullAck must drop holder pins for the acked pull_id"
    );

    Ok(())
}

// ============================================================================
// Case: pull() with hash not in peer_available errors synchronously
// ============================================================================

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn pull_validation_synchronous_error() -> Result<()> {
    let (h, p) = paired_sides().await;
    let session_id = uuid::Uuid::new_v4();

    let h_session = h.factory.open(session_id)?;
    let h_endpoint = h_session.endpoint().expect("holder endpoint");
    let p_session = p
        .factory
        .attach(session_id, h.velo.instance_id(), h_endpoint)
        .await?;

    // Wait for attach to settle on holder (so holder's peer_*
    // state reflects steady-state).
    let mut h_lifecycle = h_session.lifecycle();
    let _ = tokio::time::timeout(Duration::from_secs(30), h_lifecycle.next()).await;

    // Puller calls pull(...) for a hash NOT in peer_available — should
    // resolve synchronously to Err.
    let bogus = kvbm_logical::SequenceHash::new(999, None, 999);
    let dst = p.g2_manager.allocate_blocks(1).expect("alloc dst");
    let err = p_session
        .pull(vec![bogus], dst)
        .await
        .expect_err("pull must error when hash absent");
    assert!(
        err.to_string().contains("not in peer_available"),
        "unexpected error: {err}"
    );

    // Indirect proof that no Frame::Pull went on the wire: holder's
    // peer_committed / peer_available remain empty after a settling
    // delay. (No outgoing frames would mutate holder's peer state.)
    tokio::time::sleep(Duration::from_millis(150)).await;
    assert!(
        h_session.peer_committed().is_empty(),
        "holder saw unexpected commits: {:?}",
        h_session.peer_committed()
    );
    assert!(
        h_session.peer_available().is_empty(),
        "holder saw unexpected availability: {:?}",
        h_session.peer_available()
    );

    Ok(())
}
