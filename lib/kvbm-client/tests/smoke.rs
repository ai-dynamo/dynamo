// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Smoke test: spin up an in-process kvbm-service gRPC server on a temp UDS,
//! connect with KvbmServiceClient, register, then drop and verify cleanup.

use std::sync::Arc;
use std::time::{Duration, Instant};

use kvbm_client::{KvbmServiceClient, proto};
use kvbm_service::container::NoopContainer;
use kvbm_service::metrics::ServiceMetrics;
use kvbm_service::proto::v1::kvbm_service_server::KvbmServiceServer;
use kvbm_service::registry::Registry;
use kvbm_service::server::grpc::KvbmServiceGrpc;
use tempfile::tempdir;
use tokio::net::UnixListener;
use tokio_stream::wrappers::UnixListenerStream;
use tonic::transport::Server;

fn make_registry(capacity: u32) -> Registry {
    Registry::new(capacity, ServiceMetrics::new())
}

fn registration_instance() -> proto::RegistrationInstance {
    proto::RegistrationInstance {
        kind: Some(proto::registration_instance::Kind::Kvbm(
            proto::KvbmInstance {
                model_name: "llm".into(),
                layout_mode: Some(proto::LayoutMode {
                    kind: Some(proto::layout_mode::Kind::UniversalTp1Canonical(vec![
                        1, 2, 3,
                    ])),
                }),
                tp_size: 2,
                block_size: 64,
                mode: proto::ServiceMode::Kvbm as i32,
            },
        )),
    }
}

#[tokio::test]
async fn smoke_register_and_drop() {
    let dir = tempdir().expect("tempdir");
    let sock_path = dir.path().join("kvbm.sock");

    let registry = make_registry(4);
    let registry_clone = registry.clone();

    // Bind the UDS listener before spawning the server so the path exists
    // by the time we call connect_uds.
    let listener = UnixListener::bind(&sock_path).expect("bind UDS");
    let incoming = UnixListenerStream::new(listener);

    let grpc_svc = KvbmServiceGrpc::with_heartbeat(
        registry.clone(),
        Arc::new(NoopContainer),
        Duration::from_secs(3600),
    );
    let server = Server::builder()
        .add_service(KvbmServiceServer::new(grpc_svc))
        .serve_with_incoming(incoming);

    let server_handle = tokio::spawn(server);

    // Brief yield to let the server start accepting.
    tokio::task::yield_now().await;

    let mut client = KvbmServiceClient::connect_uds(&sock_path)
        .await
        .expect("connect_uds");

    let handle = client
        .register("client-a", registration_instance())
        .await
        .expect("register");

    assert!(
        !handle.registration_id().is_empty(),
        "registration_id must be non-empty"
    );
    assert_eq!(handle.reserved_slots(), 2, "tp_size=2 → 2 reserved slots");

    // Snapshot should show one client active.
    let snap = registry_clone.snapshot();
    assert_eq!(snap.used_slots, 2);
    assert_eq!(snap.clients.len(), 1);

    // Drop the handle — server should detect stream close and unregister.
    drop(handle);

    // Poll registry.snapshot() for up to 2 s waiting for Empty.
    let deadline = Instant::now() + Duration::from_secs(2);
    loop {
        let snap = registry_clone.snapshot();
        if snap.state == "Empty" {
            break;
        }
        if Instant::now() > deadline {
            panic!(
                "registry did not return to Empty within 2 s; state={}",
                snap.state
            );
        }
        tokio::time::sleep(Duration::from_millis(20)).await;
    }

    // Clean up: abort the server task.
    server_handle.abort();
}
