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

use dynamo_llm::http::service::service_v2::HttpService;
use dynamo_llm::http::service::metrics::{self, Endpoint};
use dynamo_runtime::CancellationToken;
use std::{env, time::Duration};

#[tokio::test]
async fn metrics_prefix_env_override() {
    env::set_var(metrics::METRICS_PREFIX_ENV, "nv_llm_http_service");
    let svc = HttpService::builder().port(9102).build().unwrap();
    let token = CancellationToken::new();
    let _handle = svc.spawn(token.clone()).await;
    tokio::time::sleep(Duration::from_millis(100)).await;

    // Force-create a labeled metric
    let state = svc.state_clone();
    {
        let _guard = state
            .metrics_clone()
            .create_inflight_guard("test-model", Endpoint::ChatCompletions, true);
    }

    let body = reqwest::get("http://localhost:9102/metrics")
        .await.unwrap()
        .text().await.unwrap();
    assert!(body.contains("nv_llm_http_service_requests_total"));
    token.cancel();
}
