// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#![cfg(feature = "testing")]

//! Regression tests for the "peer not registered" attach failure on
//! cross-instance disaggregation.
//!
//! Production bug surfaced on PR #9393 (ryan/kvbm-rdma-pull) where decode's
//! vLLM smoke fails ~10ms after warm-up with
//!
//! ```text
//! attach outbound on Attach failed
//!   transport bind failed: TCP streaming: peer <u64-worker-id> not registered
//!   (call register_peer first)
//! ```
//!
//! That error originates in `kvbm_engine::p2p::session::velo`'s
//! `dispatch_frame` handler for `Frame::Attach` (holder side), which
//! calls `velo.attach_anchor` without first ensuring the puller's
//! `PeerInfo` is in the local streaming registry. The fix wires a
//! `PeerResolver` hook into `VeloSessionFactory` and invokes it from the
//! Frame::Attach handler before `attach_anchor`.
//!
//! Two tests pin the contract:
//!
//! 1. `attach_without_peer_registration_fails` — without any resolver
//!    AND without manual `velo.register_peer` calls, an attach must
//!    surface a clear "can't find peer" error and not hang the runtime.
//!
//! 2. `resolver_is_invoked_on_frame_attach_path` — verifies the wiring:
//!    when a `Frame::Attach` arrives at the holder, the configured
//!    resolver MUST be called with the puller's `InstanceId` before
//!    `attach_anchor` is attempted. We use a recording resolver that
//!    captures the arguments and (for stability of the test) returns
//!    a synthetic error so the rest of the wire protocol short-circuits
//!    without needing a real bidi roundtrip.

use std::sync::Arc;
use std::time::Duration;

use anyhow::Result;
use futures::StreamExt;
use futures::future::BoxFuture;
use kvbm_engine::G2;
use kvbm_engine::leader::InstanceLeader;
use kvbm_engine::p2p::session::{LifecycleEvent, PeerResolver, SessionFactory, VeloSessionFactory};
use kvbm_engine::testing::managers::{TestManagerBuilder, TestRegistryBuilder};
use kvbm_logical::manager::BlockManager;
use parking_lot::Mutex;
use velo::InstanceId;
use velo::transports::tcp::TcpTransportBuilder;

const BLOCK_SIZE: usize = 16;

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
    leader: Arc<InstanceLeader>,
    _g2_manager: Arc<BlockManager<G2>>,
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
    Side {
        velo,
        leader,
        _g2_manager: g2_manager,
    }
}

// ============================================================================
// Control case: attach without peer registration must fail (not hang) with
// a "can't find peer" error
// ============================================================================

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn attach_without_peer_registration_fails() {
    let h = build_side().await;
    let p = build_side().await;
    // NOTE: deliberately NO `velo.register_peer` exchange. This is the
    // production hot path before the peer-resolver fix: peers learn each
    // other via the hub, but the local streaming registry stays empty.

    let h_factory = VeloSessionFactory::new(
        Arc::clone(&h.velo),
        Arc::clone(&h.leader),
        tokio::runtime::Handle::current(),
    );
    let p_factory = VeloSessionFactory::new(
        Arc::clone(&p.velo),
        Arc::clone(&p.leader),
        tokio::runtime::Handle::current(),
    );

    let session_id = uuid::Uuid::new_v4();
    let h_session = h_factory.open(session_id).expect("open holder");
    let h_endpoint = h_session.endpoint().expect("holder endpoint");

    // Bound the call so a hang is reported as a test failure, not a
    // 6-minute CI stall.
    let result = tokio::time::timeout(
        Duration::from_secs(20),
        p_factory.attach(session_id, h.velo.instance_id(), h_endpoint),
    )
    .await
    .expect("attach must return within 20s (not hang) when peer is not registered");

    let err = result
        .err()
        .expect("attach must fail without peer registration");
    let chain = format!("{err:?}");
    // The exact surface message depends on whether velo has a discovery
    // backend wired in. Production hits "peer X not registered (call
    // register_peer first)"; bare-test (no discovery backend) hits
    // "Cannot resolve worker X" / "No discovery backend configured".
    // Both are the same root cause — accept either signature.
    let has_signature = chain.contains("not registered")
        || chain.contains("not velo-registered")
        || chain.contains("Cannot resolve worker")
        || chain.contains("No discovery backend");
    assert!(
        has_signature,
        "expected a 'can't find peer' error signature, got: {chain}"
    );
}

// ============================================================================
// Wiring check: holder-side Frame::Attach handler invokes the resolver
// ============================================================================

/// Recording resolver: captures every `resolve_and_register` invocation,
/// then returns a synthetic error so the engine layer's Frame::Attach
/// handler short-circuits without needing a real bidi attach.
///
/// We intentionally fail the resolver so the test doesn't depend on
/// downstream velo behavior — the only thing being pinned is that
/// `resolve_and_register` was called for the expected `InstanceId`
/// before `attach_anchor` would have a chance to run.
struct RecordingResolver {
    calls: Arc<Mutex<Vec<InstanceId>>>,
}

impl PeerResolver for RecordingResolver {
    fn resolve_and_register(&self, instance_id: InstanceId) -> BoxFuture<'_, Result<()>> {
        let calls = Arc::clone(&self.calls);
        Box::pin(async move {
            calls.lock().push(instance_id);
            // Synthetic error → Frame::Attach handler emits
            // LifecycleEvent::Failed instead of calling attach_anchor.
            // We pin the resolver-was-called behavior; the rest of the
            // attach path is covered by `velo_session_loopback.rs`.
            anyhow::bail!("recording resolver: intentionally short-circuit")
        })
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn resolver_is_invoked_on_frame_attach_path() -> Result<()> {
    let h = build_side().await;
    let p = build_side().await;
    // Manually register peers in the legacy direction so the puller's
    // attach_anchor succeeds (Frame::Attach reaches the holder). This
    // mirrors production where prefill's coordinator-layer hook
    // (`coordinator.rs:422`) already registered decode on prefill's
    // velo before the engine layer ran.
    p.velo.register_peer(h.velo.peer_info()).unwrap();
    h.velo.register_peer(p.velo.peer_info()).unwrap();

    let calls: Arc<Mutex<Vec<InstanceId>>> = Arc::new(Mutex::new(Vec::new()));
    let h_resolver: Arc<dyn PeerResolver> = Arc::new(RecordingResolver {
        calls: Arc::clone(&calls),
    });
    // Holder factory carries the recording resolver — this is the
    // call-site under test (Frame::Attach handler in dispatch_frame).
    let h_factory = VeloSessionFactory::with_peer_resolver(
        Arc::clone(&h.velo),
        Arc::clone(&h.leader),
        tokio::runtime::Handle::current(),
        h_resolver,
    );
    // Puller factory has NO resolver — production's puller side doesn't
    // need the engine-layer hook because the connector layer already
    // resolved the peer.
    let p_factory = VeloSessionFactory::new(
        Arc::clone(&p.velo),
        Arc::clone(&p.leader),
        tokio::runtime::Handle::current(),
    );

    let session_id = uuid::Uuid::new_v4();
    let h_session = h_factory.open(session_id)?;
    let h_endpoint = h_session.endpoint().expect("holder endpoint");

    let _p_session = tokio::time::timeout(
        Duration::from_secs(15),
        p_factory.attach(session_id, h.velo.instance_id(), h_endpoint),
    )
    .await
    .expect("attach must complete within 15s with peers pre-registered")?;

    // Holder Frame::Attach handler should invoke the recording
    // resolver with the puller's InstanceId, then emit a Failed
    // lifecycle event (because the resolver intentionally errors).
    let mut h_lifecycle = h_session.lifecycle();
    let evt = tokio::time::timeout(Duration::from_secs(15), h_lifecycle.next())
        .await
        .expect("holder lifecycle must emit within 15s")
        .expect("holder lifecycle closed before emitting");
    let reason = match evt {
        LifecycleEvent::Failed { reason } => reason,
        other => panic!("expected LifecycleEvent::Failed from resolver error, got {other:?}"),
    };
    assert!(
        reason.contains("recording resolver")
            || reason.contains("intentionally short-circuit")
            || reason.contains("resolve peer"),
        "Failed reason should include the recorded short-circuit, got: {reason}"
    );

    let recorded = calls.lock().clone();
    assert_eq!(
        recorded.len(),
        1,
        "expected exactly one resolver call from Frame::Attach handler, got {recorded:?}"
    );
    assert_eq!(
        recorded[0],
        p.velo.instance_id(),
        "resolver was called with wrong InstanceId; expected puller={}, got {}",
        p.velo.instance_id(),
        recorded[0]
    );

    Ok(())
}
