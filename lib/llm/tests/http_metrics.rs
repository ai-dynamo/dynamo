
// SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

//! Block Manager Dynamo Integration Tests
//!
//! This module both the integration components in the `llm_kvbm` module
//! and the tests for the `llm_kvbm` module.
//!
//! The intent is to move [llm_kvbm] to a separate crate in the future.

use dynamo_llm::http::service::metrics::{self, Endpoint};
use dynamo_llm::http::service::service_v2::HttpService;
use dynamo_runtime::CancellationToken;
use std::{env, time::Duration};

#[tokio::test]
async fn metrics_prefix_default_then_env_override() {
    // Case 1: default prefix
    env::remove_var(metrics::METRICS_PREFIX_ENV);
    let svc1 = HttpService::builder().port(9101).build().unwrap();
    let token1 = CancellationToken::new();
    let _h1 = svc1.spawn(token1.clone()).await;
    tokio::time::sleep(Duration::from_millis(100)).await;

    // Populate labeled metrics
    let s1 = svc1.state_clone();
    { let _g = s1.metrics_clone().create_inflight_guard("test-model", Endpoint::ChatCompletions, false); }
    let body1 = reqwest::get("http://localhost:9101/metrics").await.unwrap().text().await.unwrap();
    assert!(body1.contains("dynamo_frontend_requests_total"));
    token1.cancel();

    // Case 2: env override to NIM prefix
    env::set_var(metrics::METRICS_PREFIX_ENV, "nv_llm_http_service");
    let svc2 = HttpService::builder().port(9102).build().unwrap();
    let token2 = CancellationToken::new();
    let _h2 = svc2.spawn(token2.clone()).await;
    tokio::time::sleep(Duration::from_millis(100)).await;

    // Populate labeled metrics
    let s2 = svc2.state_clone();
    { let _g = s2.metrics_clone().create_inflight_guard("test-model", Endpoint::ChatCompletions, true); }
    let body2 = reqwest::get("http://localhost:9102/metrics").await.unwrap().text().await.unwrap();
    assert!(body2.contains("nv_llm_http_service_requests_total"));
    token2.cancel();
}
