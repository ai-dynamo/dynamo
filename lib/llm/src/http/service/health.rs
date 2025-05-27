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
use dynamo_runtime::component::{Instance, INSTANCE_ROOT_PATH};
use dynamo_runtime::{DistributedRuntime, Runtime};
use std::sync::Arc;

pub fn health_check_router(
    state: Arc<service_v2::State>,
    path_override: Option<String>,
) -> (Vec<RouteDoc>, Router) {
    let path = path_override.unwrap_or_else(|| "/health".to_string());
    let path_namespace = format!("{path}/{{namespace}}");
    let path_component = format!("{path}/{{namespace}}/{{component}}");

    let docs: Vec<RouteDoc> = vec![
        RouteDoc::new(Method::GET, &path),
        RouteDoc::new(Method::GET, &path_namespace),
        RouteDoc::new(Method::GET, &path_component),
    ];

    let router = Router::new()
        .route(&path, get(health_handler))
        .route(&path_namespace, get(health_namespace_handler))
        .route(&path_component, get(health_component_handler))
        .with_state(state);

    (docs, router)
}

async fn health_handler() -> impl IntoResponse {
    (StatusCode::OK, "OK")
}

// A namespace health check will return if the namespace exists in ETCD and will return a list of components currently registered
async fn health_namespace_handler(Path(namespace): Path<String>) -> impl IntoResponse {
    let target_key = format!("{INSTANCE_ROOT_PATH}/{namespace}");

    match get_instances(&target_key).await {
        Ok(instances) if instances.is_empty() => (
            StatusCode::NOT_FOUND,
            format!("Namespace '{}' not found", namespace),
        ),
        Ok(instances) => {
            let components: Vec<String> = instances
                .iter()
                .map(|i| format!("{}.{}", i.component, i.endpoint))
                .collect();

            (
                StatusCode::OK,
                format!(
                    "Namespace '{}' is healthy: {} instances - {}",
                    namespace,
                    components.len(),
                    components.join(", ")
                ),
            )
        }
        Err(error_response) => error_response,
    }
}

// A component health check will return the endpoints and instance IDs for a specific component
async fn health_component_handler(
    Path((namespace, component)): Path<(String, String)>,
) -> impl IntoResponse {
    let target_key = format!("{INSTANCE_ROOT_PATH}/{namespace}/{component}");

    match get_instances(&target_key).await {
        Ok(instances) if instances.is_empty() => (
            StatusCode::NOT_FOUND,
            format!("Component '{}.{}' not found", namespace, component),
        ),
        Ok(instances) => {
            let instance_info: Vec<String> = instances
                .iter()
                .map(|i| format!("{}.{} (id: {})", i.component, i.endpoint, i.instance_id))
                .collect();

            (
                StatusCode::OK,
                format!(
                    "Component '{}.{}' is healthy: {} instances - {}",
                    namespace,
                    component,
                    instances.len(),
                    instance_info.join(", ")
                ),
            )
        }
        Err(error_response) => error_response,
    }
}

// Helper function to get distributed runtime
async fn get_drt() -> Result<DistributedRuntime, (StatusCode, String)> {
    let runtime =
        Runtime::from_current().map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?;

    DistributedRuntime::from_settings(runtime)
        .await
        .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))
}

// Helper function to get instances from etcd
async fn get_instances(target_key: &str) -> Result<Vec<Instance>, (StatusCode, String)> {
    let drt = get_drt().await?;

    let kvpairs = drt
        .etcd_client()
        .ok_or((
            StatusCode::INTERNAL_SERVER_ERROR,
            "No etcd client".to_string(),
        ))?
        .kv_get_prefix(target_key)
        .await
        .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?;

    let mut instances = Vec::new();
    for kvpair in kvpairs {
        match serde_json::from_slice::<Instance>(kvpair.value()) {
            Ok(instance) => instances.push(instance),
            Err(e) => tracing::error!("Failed to parse instance from etcd: {}", e),
        }
    }

    Ok(instances)
}
