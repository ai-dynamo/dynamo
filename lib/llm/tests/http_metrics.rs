// SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use dynamo_llm::http::service::metrics::{self, Endpoint};
use dynamo_llm::http::service::service_v2::HttpService;
use dynamo_runtime::CancellationToken;
use serial_test::serial;
use std::{env, time::Duration};

#[path = "common/ports.rs"]
mod ports;
use ports::get_random_port;

#[tokio::test]
#[serial]
async fn metrics_prefix_default_then_env_override() {
    // Case 1: default prefix
    env::remove_var(metrics::METRICS_PREFIX_ENV);
    let p1 = get_random_port().await;
    let svc1 = HttpService::builder().port(p1).build().unwrap();
    let token1 = CancellationToken::new();
    let h1 = svc1.spawn(token1.clone()).await;
    wait_for_metrics_ready(p1).await;

    // Populate labeled metrics
    let s1 = svc1.state_clone();
    {
        let _g = s1.metrics_clone().create_inflight_guard(
            "test-model",
            Endpoint::ChatCompletions,
            false,
        );
    }
    let body1 = reqwest::get(format!("http://localhost:{}/metrics", p1))
        .await
        .unwrap()
        .text()
        .await
        .unwrap();
    assert!(body1.contains("dynamo_frontend_requests_total"));
    token1.cancel();
    let _ = h1.await; // ensure port is released

    // Case 2: env override to prefix
    env::set_var(metrics::METRICS_PREFIX_ENV, "custom_prefix");
    let p2 = get_random_port().await;
    let svc2 = HttpService::builder().port(p2).build().unwrap();
    let token2 = CancellationToken::new();
    let h2 = svc2.spawn(token2.clone()).await;
    wait_for_metrics_ready(p2).await;

    // Populate labeled metrics
    let s2 = svc2.state_clone();
    {
        let _g =
            s2.metrics_clone()
                .create_inflight_guard("test-model", Endpoint::ChatCompletions, true);
    }
    // Single fetch and assert
    let body2 = reqwest::get(format!("http://localhost:{}/metrics", p2))
        .await
        .unwrap()
        .text()
        .await
        .unwrap();
    assert!(body2.contains("custom_prefix_requests_total"));
    assert!(!body2.contains("dynamo_frontend_requests_total"));
    token2.cancel();
    let _ = h2.await;

    // Case 3: invalid env prefix is sanitized
    env::set_var(metrics::METRICS_PREFIX_ENV, "nv-llm/http service");
    let p3 = get_random_port().await;
    let svc3 = HttpService::builder().port(p3).build().unwrap();
    let token3 = CancellationToken::new();
    let h3 = svc3.spawn(token3.clone()).await;
    wait_for_metrics_ready(p3).await;

    let s3 = svc3.state_clone();
    {
        let _g =
            s3.metrics_clone()
                .create_inflight_guard("test-model", Endpoint::ChatCompletions, true);
    }
    let body3 = reqwest::get(format!("http://localhost:{}/metrics", p3))
        .await
        .unwrap()
        .text()
        .await
        .unwrap();
    assert!(body3.contains("nv_llm_http_service_requests_total"));
    assert!(!body3.contains("dynamo_frontend_requests_total"));
    token3.cancel();
    let _ = h3.await;

    // Cleanup env to avoid leaking state
    env::remove_var(metrics::METRICS_PREFIX_ENV);
}

// Poll /metrics until ready or timeout
async fn wait_for_metrics_ready(port: u16) {
    let url = format!("http://localhost:{}/metrics", port);
    let start = tokio::time::Instant::now();
    let timeout = Duration::from_secs(5);
    loop {
        if start.elapsed() > timeout {
            panic!("Timed out waiting for metrics endpoint at {}", url);
        }
        match reqwest::get(&url).await {
            Ok(resp) if resp.status().is_success() => break,
            _ => tokio::time::sleep(Duration::from_millis(50)).await,
        }
    }
}
