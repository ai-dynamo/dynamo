// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Graceful-shutdown integration tests.
//!
//! Drives the full `KvbmServiceGrpc` over a real UDS, exercises the
//! shutdown coordinator (which lives on `KvbmService`), and verifies:
//!
//! 1. `ServerShutdownInitiated` is enqueued onto every active stream.
//! 2. The container's `on_server_shutdown` hook fires exactly once.
//! 3. New `Register` calls during drain are rejected with `Unavailable`.
//! 4. After the grace period elapses on a non-detaching client, the
//!    coordinator sends `ServerShutdownTimedOut` and force-closes the stream.

use std::path::PathBuf;
use std::sync::Arc;
use std::sync::atomic::{AtomicU32, Ordering};
use std::time::Duration;

use async_trait::async_trait;
use futures::StreamExt;
use kvbm_client::{KvbmServiceClient, proto as cproto};
use kvbm_service::container::{ContainerError, ServiceContainer};
use kvbm_service::instance::RegistrationInstance;
use kvbm_service::metrics::ServiceMetrics;
use kvbm_service::proto::v1::kvbm_service_server::KvbmServiceServer;
use kvbm_service::registry::{RegistrationId, Registry};
use kvbm_service::server::grpc::KvbmServiceGrpc;
use parking_lot::Mutex;
use tempfile::TempDir;
use tokio::net::UnixListener;
use tokio_stream::wrappers::UnixListenerStream;
use tokio_util::sync::CancellationToken;

/// Test container that records every callback so suites can assert
/// ordering and counts deterministically.
struct RecordingContainer {
    registers: AtomicU32,
    unregisters: AtomicU32,
    shutdowns: AtomicU32,
    last_grace: Mutex<Option<Option<Duration>>>,
}

impl RecordingContainer {
    fn new() -> Arc<Self> {
        Arc::new(Self {
            registers: AtomicU32::new(0),
            unregisters: AtomicU32::new(0),
            shutdowns: AtomicU32::new(0),
            last_grace: Mutex::new(None),
        })
    }
}

#[async_trait]
impl ServiceContainer for RecordingContainer {
    fn name(&self) -> &str {
        "recording"
    }
    async fn on_register(
        &self,
        _id: RegistrationId,
        _instance: &RegistrationInstance,
    ) -> Result<(), ContainerError> {
        self.registers.fetch_add(1, Ordering::SeqCst);
        Ok(())
    }
    async fn on_unregister(&self, _id: RegistrationId) {
        self.unregisters.fetch_add(1, Ordering::SeqCst);
    }
    async fn on_server_shutdown(&self, grace: Option<Duration>) {
        self.shutdowns.fetch_add(1, Ordering::SeqCst);
        *self.last_grace.lock() = Some(grace);
    }
}

struct Harness {
    _tmp: TempDir,
    uds: PathBuf,
    cancel: CancellationToken,
    registry: Registry,
    container: Arc<RecordingContainer>,
}

impl Harness {
    async fn spawn() -> Self {
        let metrics = ServiceMetrics::new();
        let registry = Registry::new(8, metrics);
        let container = RecordingContainer::new();
        let tmp = tempfile::tempdir().expect("tempdir");
        let uds = tmp.path().join("kvbm.sock");

        // Heartbeat off (3600s) so it doesn't interleave with shutdown events.
        let grpc = KvbmServiceGrpc::with_heartbeat(
            registry.clone(),
            container.clone() as Arc<dyn ServiceContainer>,
            Duration::from_secs(3600),
        );
        let listener = UnixListener::bind(&uds).expect("bind UDS");
        let stream = UnixListenerStream::new(listener);
        let cancel = CancellationToken::new();
        let cancel_clone = cancel.clone();
        tokio::spawn(async move {
            let _ = tonic::transport::Server::builder()
                .add_service(KvbmServiceServer::new(grpc))
                .serve_with_incoming_shutdown(stream, cancel_clone.cancelled_owned())
                .await;
        });
        tokio::time::sleep(Duration::from_millis(50)).await;
        Self {
            _tmp: tmp,
            uds,
            cancel,
            registry,
            container,
        }
    }
}

fn make_instance(model: &str) -> cproto::RegistrationInstance {
    cproto::RegistrationInstance {
        kind: Some(cproto::registration_instance::Kind::Kvbm(
            cproto::KvbmInstance {
                model_name: model.into(),
                layout_mode: Some(cproto::LayoutMode {
                    kind: Some(cproto::layout_mode::Kind::UniversalTp1Canonical(vec![
                        1, 2, 3,
                    ])),
                }),
                tp_size: 2,
                block_size: 64,
                mode: cproto::ServiceMode::Kvbm as i32,
            },
        )),
    }
}

#[tokio::test]
async fn shutdown_broadcasts_initiated_to_every_client() {
    let h = Harness::spawn().await;

    // Two attached clients sharing the same key.
    let mut client_a = KvbmServiceClient::connect_uds(&h.uds).await.unwrap();
    let mut handle_a = client_a
        .register("c-a", make_instance("llm"))
        .await
        .unwrap();
    let mut client_b = KvbmServiceClient::connect_uds(&h.uds).await.unwrap();
    let mut handle_b = client_b
        .register("c-b", make_instance("llm"))
        .await
        .unwrap();

    assert_eq!(h.container.registers.load(Ordering::SeqCst), 2);
    assert_eq!(h.registry.snapshot().used_slots, 4);

    // Kick off the coordinator on the registry+container. Use a long grace
    // (90s) — exceeds the minimum and the test never has to wait on it
    // because both clients detach immediately after observing the event.
    let registry = h.registry.clone();
    let container = h.container.clone();
    let coord = tokio::spawn(async move {
        let lcs = registry.begin_drain();
        let initiated: Vec<_> = lcs
            .iter()
            .map(|lc| lc.send_shutdown_initiated(Some(Duration::from_secs(90))))
            .collect();
        futures::future::join_all(initiated).await;
        let combined = async {
            let _ = tokio::join!(
                container.on_server_shutdown(Some(Duration::from_secs(90))),
                registry.wait_until_empty(),
            );
        };
        let _ = tokio::time::timeout(Duration::from_secs(30), combined).await;
    });

    // Each client should receive ServerShutdownInitiated next.
    let ev = handle_a.next().await.expect("event a").unwrap();
    match ev.kind {
        Some(cproto::event::Kind::ServerShutdownInitiated(body)) => {
            assert_eq!(body.grace_period_ms, 90_000);
        }
        other => panic!("client A expected ServerShutdownInitiated, got {other:?}"),
    }
    let ev = handle_b.next().await.expect("event b").unwrap();
    assert!(matches!(
        ev.kind,
        Some(cproto::event::Kind::ServerShutdownInitiated(_))
    ));

    // Clients detach: drop their handles.
    drop(handle_a);
    drop(handle_b);

    // Coordinator finishes.
    coord.await.unwrap();

    assert_eq!(h.container.shutdowns.load(Ordering::SeqCst), 1);
    assert_eq!(
        *h.container.last_grace.lock(),
        Some(Some(Duration::from_secs(90)))
    );
    // Both clients flushed through on_unregister.
    assert_eq!(h.container.unregisters.load(Ordering::SeqCst), 2);
    assert_eq!(h.registry.snapshot().clients.len(), 0);

    h.cancel.cancel();
}

/// Race regression: a Register that races with `shutdown_graceful` must
/// either complete cleanly with `Accepted` arriving FIRST on the stream,
/// or fail with `Unavailable` and never enqueue any event the client sees.
/// In particular, the stream must never start with
/// `ServerShutdownInitiated` (or any event other than `Accepted`).
#[tokio::test]
async fn racing_register_either_sees_accepted_first_or_fails_cleanly() {
    use kvbm_service::registry::StreamLifecycle;

    // Drive enough iterations that we hit the window across thread schedules.
    for _ in 0..50 {
        let h = Harness::spawn().await;
        let registry = h.registry.clone();

        // Spawn the register call so we can race it.
        let uds = h.uds.clone();
        let register_task = tokio::spawn(async move {
            let mut client = KvbmServiceClient::connect_uds(&uds).await.unwrap();
            client.register("racy", make_instance("llm")).await
        });

        // Yield a couple of times to let the gRPC handler reach (or get
        // close to) the container.on_register await.
        tokio::task::yield_now().await;
        tokio::task::yield_now().await;

        // Fire shutdown concurrently.
        let drain_lifecycles: Vec<Arc<dyn StreamLifecycle>> = registry.begin_drain();
        for lc in &drain_lifecycles {
            lc.send_shutdown_initiated(Some(Duration::from_secs(90)))
                .await;
        }

        match register_task.await.unwrap() {
            Ok(mut handle) => {
                // Success path: first event MUST be Accepted (we know this
                // because the client crate already asserts that), and any
                // subsequent ServerShutdownInitiated must come after.
                let acc = handle.accepted();
                assert!(
                    !acc.registration_id.is_empty(),
                    "registration_id must be non-empty"
                );
                // Pull the next event if there is one — if shutdown reached
                // us post-commit, it should be ServerShutdownInitiated, not
                // anything weirder.
                if let Ok(Some(Ok(ev))) =
                    tokio::time::timeout(Duration::from_millis(200), handle.next()).await
                {
                    assert!(
                        matches!(
                            ev.kind,
                            Some(cproto::event::Kind::ServerShutdownInitiated(_))
                                | Some(cproto::event::Kind::Heartbeat(_))
                                | Some(cproto::event::Kind::ServerShutdownTimedOut(_))
                        ),
                        "unexpected event after Accepted: {ev:?}"
                    );
                }
                drop(handle);
            }
            Err(kvbm_client::ClientError::Rpc(status)) => {
                // Lost the race — should be Unavailable from commit_register
                // tripping over Draining. Client never observed Accepted.
                assert_eq!(status.code(), tonic::Code::Unavailable, "{status}");
            }
            Err(other) => panic!("unexpected client error: {other:?}"),
        }

        h.cancel.cancel();
    }
}

#[tokio::test]
async fn register_during_drain_returns_unavailable() {
    let h = Harness::spawn().await;
    h.registry.begin_drain();
    let mut client = KvbmServiceClient::connect_uds(&h.uds).await.unwrap();
    match client.register("c", make_instance("llm")).await {
        Err(kvbm_client::ClientError::Rpc(status)) => {
            assert_eq!(status.code(), tonic::Code::Unavailable);
        }
        Err(other) => panic!("expected Rpc error, got {other:?}"),
        Ok(_) => panic!("expected register to fail during drain"),
    }
    h.cancel.cancel();
}

#[tokio::test(start_paused = true)]
async fn timed_out_force_closes_stragglers() {
    let h = Harness::spawn().await;
    let mut client = KvbmServiceClient::connect_uds(&h.uds).await.unwrap();
    let mut handle = client
        .register("stuck", make_instance("llm"))
        .await
        .unwrap();

    let registry = h.registry.clone();
    let container = h.container.clone();
    let grace = Duration::from_secs(60);
    let coord = tokio::spawn(async move {
        let lcs = registry.begin_drain();
        let initiated: Vec<_> = lcs
            .iter()
            .map(|lc| lc.send_shutdown_initiated(Some(grace)))
            .collect();
        futures::future::join_all(initiated).await;
        let combined = async {
            let _ = tokio::join!(
                container.on_server_shutdown(Some(grace)),
                registry.wait_until_empty(),
            );
        };
        let timed_out = tokio::time::timeout(grace, combined).await.is_err();
        if timed_out {
            let stragglers = registry.begin_drain();
            let force: Vec<_> = stragglers
                .iter()
                .map(|lc| lc.send_shutdown_timed_out(grace))
                .collect();
            futures::future::join_all(force).await;
        }
        timed_out
    });

    // Drain the Initiated event.
    let ev = handle.next().await.unwrap().unwrap();
    assert!(matches!(
        ev.kind,
        Some(cproto::event::Kind::ServerShutdownInitiated(_))
    ));

    // Advance past the grace period without detaching.
    tokio::time::advance(grace + Duration::from_millis(100)).await;

    // The coordinator should have routed the TimedOut event by now.
    let ev = handle.next().await.unwrap().unwrap();
    match ev.kind {
        Some(cproto::event::Kind::ServerShutdownTimedOut(body)) => {
            assert_eq!(body.grace_period_ms, grace.as_millis() as u64);
        }
        other => panic!("expected ServerShutdownTimedOut, got {other:?}"),
    }
    // And the stream closes.
    assert!(
        handle.next().await.is_none(),
        "stream must close after TimedOut"
    );

    let timed_out = coord.await.unwrap();
    assert!(timed_out, "coordinator must report timeout");
    h.cancel.cancel();
}
