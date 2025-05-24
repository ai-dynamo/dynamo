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

use super::{service_v2, RouteDoc};
use axum::{
    extract::Path, http::Method, http::StatusCode, response::IntoResponse, routing::get, Router,
};
use dynamo_runtime::{DistributedRuntime, Runtime};
use std::sync::Arc;

pub fn health_check_router(
    state: Arc<service_v2::State>,
    path_override: Option<String>,
) -> (Vec<RouteDoc>, Router) {
    let path = path_override.unwrap_or_else(|| "/health".to_string());
    let path_namespace = format!("{path}/{{namespace}}");

    let docs: Vec<RouteDoc> = vec![
        RouteDoc::new(Method::GET, &path),
        RouteDoc::new(Method::GET, &path_namespace),
    ];

    let router = Router::new()
        .route(&path, get(health_handler))
        .route(&path_namespace, get(health_namespace_handler))
        .with_state(state);

    (docs, router)
}

async fn health_handler() -> impl IntoResponse {
    return (StatusCode::OK, "OK");
}

// A namespace health check will return if the namespace exists in ETCD and will return a list of components currently registered
async fn health_namespace_handler(Path(namespace): Path<String>) -> impl IntoResponse {
    let runtime = Runtime::from_current();
    match runtime {
        Ok(runtime) => {
            let drt = DistributedRuntime::from_settings(runtime).await.unwrap();

            let target_key = format!("instances/{namespace}");
            let kvpairs = drt
                .etcd_client()
                .unwrap()
                .kv_get_prefix(target_key)
                .await
                .unwrap();
            let mut components: Vec<String> = vec![];
            for kvpair in kvpairs {
                let val = kvpair.value();
                components.push(String::from_utf8(val.to_vec()).expect("Invalid UTF-8"));
            }
            return (StatusCode::OK, components.join(","));
        }
        Err(e) => {
            return (StatusCode::INTERNAL_SERVER_ERROR, e.to_string());
        }
    }
}
