// SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use dynamo_llm::http::service::metrics::Endpoint;
use dynamo_llm::http::service::service_v2::HttpService;
use dynamo_runtime::CancellationToken;
use dynamo_runtime::metrics::prometheus_names::frontend_service::METRICS_PREFIX_ENV;
use std::time::Duration;

#[path = "common/ports.rs"]
mod ports;
use ports::get_random_port;

#[tokio::test]
async fn test_metrics_prefix_default() {
    // Test default prefix when no env var is set
    temp_env::async_with_vars([(METRICS_PREFIX_ENV, None::<&str>)], async {
        let port = get_random_port().await;
        let service = HttpService::builder().port(port).build().unwrap();
        let token = CancellationToken::new();
        let handle = service.spawn(token.clone()).await;
        wait_for_metrics_ready(port).await;

        // Populate labeled metrics
        let state = service.state_clone();
        {
            let _guard = state.metrics_clone().create_inflight_guard(
                "test-model",
                Endpoint::ChatCompletions,
                false,
            );
        }

        let body = reqwest::get(format!("http://localhost:{}/metrics", port))
            .await
            .unwrap()
            .text()
            .await
            .unwrap();
        assert!(body.contains("dynamo_frontend_requests_total"));

        token.cancel();
        let _ = handle.await;
    })
    .await;
}

#[tokio::test]
async fn test_metrics_prefix_custom() {
    // Test custom prefix override via environment variable
    temp_env::async_with_vars([(METRICS_PREFIX_ENV, Some("custom_prefix"))], async {
        let port = get_random_port().await;
        let service = HttpService::builder().port(port).build().unwrap();
        let token = CancellationToken::new();
        let handle = service.spawn(token.clone()).await;
        wait_for_metrics_ready(port).await;

        // Populate labeled metrics
        let state = service.state_clone();
        {
            let _guard = state.metrics_clone().create_inflight_guard(
                "test-model",
                Endpoint::ChatCompletions,
                true,
            );
        }

        let body = reqwest::get(format!("http://localhost:{}/metrics", port))
            .await
            .unwrap()
            .text()
            .await
            .unwrap();
        assert!(body.contains("custom_prefix_requests_total"));
        assert!(!body.contains("dynamo_frontend_requests_total"));

        token.cancel();
        let _ = handle.await;
    })
    .await;
}

#[tokio::test]
async fn test_metrics_prefix_sanitized() {
    // Test that invalid prefix characters are sanitized
    temp_env::async_with_vars([(METRICS_PREFIX_ENV, Some("nv-llm/http service"))], async {
        let port = get_random_port().await;
        let service = HttpService::builder().port(port).build().unwrap();
        let token = CancellationToken::new();
        let handle = service.spawn(token.clone()).await;
        wait_for_metrics_ready(port).await;

        let state = service.state_clone();
        {
            let _guard = state.metrics_clone().create_inflight_guard(
                "test-model",
                Endpoint::ChatCompletions,
                true,
            );
        }

        let body = reqwest::get(format!("http://localhost:{}/metrics", port))
            .await
            .unwrap()
            .text()
            .await
            .unwrap();
        assert!(body.contains("nv_llm_http_service_requests_total"));
        assert!(!body.contains("dynamo_frontend_requests_total"));

        token.cancel();
        let _ = handle.await;
    })
    .await;
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
