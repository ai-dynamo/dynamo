// SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use dynamo_llm::http::service::metrics::{self, Endpoint};
use dynamo_llm::http::service::service_v2::HttpService;
use dynamo_runtime::CancellationToken;
use serial_test::serial;
use std::{env, time::Duration};

#[tokio::test]
#[serial]
async fn metrics_prefix_default_then_env_override() {
    // Case 1: default prefix
    env::remove_var(metrics::METRICS_PREFIX_ENV);
    let svc1 = HttpService::builder().port(9101).build().unwrap();
    let token1 = CancellationToken::new();
    let _h1 = svc1.spawn(token1.clone()).await;
    wait_for_metrics_ready(9101).await;

    // Populate labeled metrics
    let s1 = svc1.state_clone();
    {
        let _g = s1.metrics_clone().create_inflight_guard(
            "test-model",
            Endpoint::ChatCompletions,
            false,
        );
    }
    let body1 = reqwest::get("http://localhost:9101/metrics")
        .await
        .unwrap()
        .text()
        .await
        .unwrap();
    assert!(body1.contains("dynamo_frontend_requests_total"));
    token1.cancel();

    // Case 2: env override to prefix
    env::set_var(metrics::METRICS_PREFIX_ENV, "custom_prefix");
    let svc2 = HttpService::builder().port(9102).build().unwrap();
    let token2 = CancellationToken::new();
    let _h2 = svc2.spawn(token2.clone()).await;
    wait_for_metrics_ready(9102).await;

    // Populate labeled metrics
    let s2 = svc2.state_clone();
    {
        let _g =
            s2.metrics_clone()
                .create_inflight_guard("test-model", Endpoint::ChatCompletions, true);
    }
    // Single fetch and assert
    let body2 = reqwest::get("http://localhost:9102/metrics")
        .await
        .unwrap()
        .text()
        .await
        .unwrap();
    assert!(body2.contains("custom_prefix_requests_total"));
    assert!(!body2.contains("dynamo_frontend_requests_total"));
    token2.cancel();

    // Case 3: invalid env prefix is sanitized
    env::set_var(metrics::METRICS_PREFIX_ENV, "nv-llm/http service");
    let svc3 = HttpService::builder().port(9103).build().unwrap();
    let token3 = CancellationToken::new();
    let _h3 = svc3.spawn(token3.clone()).await;
    wait_for_metrics_ready(9103).await;

    let s3 = svc3.state_clone();
    {
        let _g =
            s3.metrics_clone()
                .create_inflight_guard("test-model", Endpoint::ChatCompletions, true);
    }
    let body3 = reqwest::get("http://localhost:9103/metrics")
        .await
        .unwrap()
        .text()
        .await
        .unwrap();
    assert!(body3.contains("nv_llm_http_service_requests_total"));
    assert!(!body3.contains("dynamo_frontend_requests_total"));
    token3.cancel();

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
