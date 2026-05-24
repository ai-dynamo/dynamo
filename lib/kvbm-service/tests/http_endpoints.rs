// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! End-to-end smoke for the axum sidecar driven by the real `KvbmService`
//! (gRPC + HTTP both up).

use std::time::Duration;

use kvbm_service::{KvbmService, ServiceConfig};

#[tokio::test]
async fn http_sidecar_reports_ready_and_discovers_socket() {
    let tmp = tempfile::tempdir().unwrap();
    let cfg = ServiceConfig {
        http_addr: "127.0.0.1:0".parse().unwrap(),
        uds_path: None,
        uds_dir: tmp.path().to_path_buf(),
        shutdown_grace_ms: None,
        pool: Default::default(),
    };
    let svc = KvbmService::start(cfg).await.expect("KvbmService start");
    let http = svc.http_addr;
    let uds_expected = svc.uds_path.display().to_string();

    let client = reqwest::Client::new();

    // /live always 200.
    let r = client
        .get(format!("http://{http}/live"))
        .send()
        .await
        .unwrap();
    assert_eq!(r.status(), 200);
    assert_eq!(r.text().await.unwrap(), "ok");

    // /ready should be 200 since uds was set before serve() returned.
    let r = client
        .get(format!("http://{http}/ready"))
        .send()
        .await
        .unwrap();
    assert_eq!(r.status(), 200);

    // /v1/discovery/socket returns the UDS path the gRPC bound.
    let r = client
        .get(format!("http://{http}/v1/discovery/socket"))
        .send()
        .await
        .unwrap();
    assert_eq!(r.status(), 200);
    let body: serde_json::Value = r.json().await.unwrap();
    assert_eq!(body["uds_path"], uds_expected);

    // /v1/registrations starts Empty.
    let r = client
        .get(format!("http://{http}/v1/registrations"))
        .send()
        .await
        .unwrap();
    assert_eq!(r.status(), 200);
    let body: serde_json::Value = r.json().await.unwrap();
    assert_eq!(body["state"], "Empty");

    // /metrics returns Prometheus text with our metric names.
    let r = client
        .get(format!("http://{http}/metrics"))
        .send()
        .await
        .unwrap();
    assert_eq!(r.status(), 200);
    let body = r.text().await.unwrap();
    assert!(body.contains("kvbm_service_capacity_slots"));
    assert!(body.contains("kvbm_service_registered_clients"));

    // /v1/discovery/resources returns a JSON object.
    let r = client
        .get(format!("http://{http}/v1/discovery/resources"))
        .send()
        .await
        .unwrap();
    assert_eq!(r.status(), 200);
    let body: serde_json::Value = r.json().await.unwrap();
    assert!(
        body.is_object(),
        "/v1/discovery/resources must be an object"
    );

    svc.shutdown().await;
    // Give the OS a moment to release the socket file before tempdir cleanup.
    tokio::time::sleep(Duration::from_millis(50)).await;
}
