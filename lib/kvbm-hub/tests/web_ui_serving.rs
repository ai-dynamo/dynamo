// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Phase E — embedded UI asset routes return 200 with the right
//! Content-Type. Discovery hub only (no velo needed for static assets).

use std::net::{IpAddr, Ipv4Addr};
use std::time::Duration;

use kvbm_hub::HubServer;

async fn start_hub() -> HubServer {
    kvbm_hub::create_server_builder()
        .bind_addr(IpAddr::V4(Ipv4Addr::LOCALHOST))
        .discovery_port(0)
        .control_port(0)
        .heartbeat_interval(Duration::from_secs(3600))
        .heartbeat_max_failures(u32::MAX)
        .registration_ttl(Duration::from_secs(3600))
        .serve()
        .await
        .expect("start hub")
}

fn http() -> reqwest::Client {
    reqwest::Client::new()
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn serves_index_and_assets_with_correct_content_type() {
    let server = start_hub().await;
    let base = format!("http://{}", server.control_addr());

    let cases = &[
        ("/", "text/html; charset=utf-8", "<title>KVBM Hub</title>"),
        ("/app.js", "application/javascript; charset=utf-8", "hubApp"),
        ("/style.css", "text/css; charset=utf-8", "--accent"),
        ("/tokens.css", "text/css; charset=utf-8", "--accent"),
        (
            "/vendor/alpine.min.js",
            "application/javascript; charset=utf-8",
            "",
        ),
    ];

    for (path, expected_ct, must_contain) in cases {
        let url = format!("{base}{path}");
        let resp = http().get(&url).send().await.expect("GET");
        assert_eq!(
            resp.status().as_u16(),
            200,
            "expected 200 for {path}, got {}",
            resp.status()
        );
        let ct = resp
            .headers()
            .get(reqwest::header::CONTENT_TYPE)
            .and_then(|v| v.to_str().ok())
            .unwrap_or("")
            .to_owned();
        assert_eq!(&ct, expected_ct, "wrong Content-Type for {path}");
        let body = resp.text().await.expect("body");
        if !must_contain.is_empty() {
            assert!(
                body.contains(must_contain),
                "body for {path} missing marker {must_contain:?}"
            );
        }
        assert!(!body.is_empty(), "body for {path} unexpectedly empty");
    }

    server.shutdown().await.unwrap();
}
