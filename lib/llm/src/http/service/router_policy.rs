// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Frontend-local worker-selection policy administration.

use std::{sync::Arc, time::Duration};

use axum::{
    Json, Router,
    extract::{
        DefaultBodyLimit, Query, State,
        rejection::{JsonRejection, QueryRejection},
    },
    http::{Method, StatusCode},
    response::{IntoResponse, Response},
    routing::get,
};
use dynamo_kv_router::plugins::worker_selection::{
    RuntimePolicyConfig, RuntimePolicyError, RuntimePolicyUpdate,
};
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};
use uuid::Uuid;

use super::RouteDoc;
use crate::discovery::{ModelManager, router_policy::RouterPolicyTarget};

const CONTROL_TIMEOUT: Duration = Duration::from_secs(5);

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct PolicyQuery {
    model: Option<String>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct PolicyPatch {
    router_id: Uuid,
    expected_revision: u64,
    parameters: Option<Map<String, Value>>,
    #[serde(default)]
    reset: bool,
}

#[derive(Serialize)]
struct PolicyResponse {
    #[serde(flatten)]
    target: RouterPolicyTarget,
    /// Null means the selected policy has not opted in to runtime configuration.
    configuration: Option<RuntimePolicyConfig>,
}

#[derive(Serialize)]
struct PolicyListResponse {
    routers: Vec<PolicyResponse>,
}

fn error(status: StatusCode, message: impl ToString) -> Response {
    (
        status,
        Json(serde_json::json!({"error": message.to_string()})),
    )
        .into_response()
}

fn policy_error(error_value: RuntimePolicyError) -> Response {
    let status = match &error_value {
        RuntimePolicyError::Conflict { .. } => StatusCode::CONFLICT,
        RuntimePolicyError::Unsupported | RuntimePolicyError::InvalidParameters(_) => {
            StatusCode::UNPROCESSABLE_ENTITY
        }
        RuntimePolicyError::Unavailable => StatusCode::SERVICE_UNAVAILABLE,
    };
    error(status, error_value)
}

pub(crate) fn router_policy_router(manager: Arc<ModelManager>) -> (Vec<RouteDoc>, Router) {
    let (mut docs, thresholds) =
        super::busy_threshold::router(manager.clone(), "/router_policy/busy_threshold");
    docs.extend([
        RouteDoc::new(Method::GET, "/router_policy"),
        RouteDoc::new(Method::PATCH, "/router_policy"),
    ]);
    (
        docs,
        Router::new()
            .route("/router_policy", get(list).patch(update))
            .layer(DefaultBodyLimit::max(16 * 1024))
            .with_state(manager)
            .merge(thresholds),
    )
}

async fn list(
    State(manager): State<Arc<ModelManager>>,
    query: Result<Query<PolicyQuery>, QueryRejection>,
) -> Response {
    let Query(query) = match query {
        Ok(query) => query,
        Err(rejection) => return error(rejection.status(), rejection.body_text()),
    };
    let entries = manager.router_policies.list(query.model.as_deref());
    let results =
        futures::future::join_all(entries.into_iter().map(|(target, control)| async move {
            (
                target,
                tokio::time::timeout(CONTROL_TIMEOUT, control.get()).await,
            )
        }))
        .await;
    let routers = results
        .into_iter()
        .filter_map(|(target, result)| {
            let configuration = match result {
                Ok(Ok(configuration)) => Some(configuration),
                Ok(Err(RuntimePolicyError::Unsupported)) => None,
                // Retired or unresponsive routers must not hide healthy instances.
                _ => return None,
            };
            Some(PolicyResponse {
                target,
                configuration,
            })
        })
        .collect();
    Json(PolicyListResponse { routers }).into_response()
}

async fn update(
    State(manager): State<Arc<ModelManager>>,
    payload: Result<Json<PolicyPatch>, JsonRejection>,
) -> Response {
    let Json(request) = match payload {
        Ok(request) => request,
        Err(rejection) => return error(rejection.status(), rejection.body_text()),
    };
    let update = match (request.reset, request.parameters) {
        (true, None) => RuntimePolicyUpdate::reset(request.expected_revision),
        (false, Some(parameters)) if !parameters.is_empty() => {
            RuntimePolicyUpdate::patch(request.expected_revision, parameters)
        }
        _ => {
            return error(
                StatusCode::BAD_REQUEST,
                "provide either nonempty parameters or reset: true",
            );
        }
    };
    let Some((target, control)) = manager.router_policies.get(request.router_id) else {
        return error(
            StatusCode::NOT_FOUND,
            "router instance not found; query /router_policy on this frontend",
        );
    };
    match tokio::time::timeout(CONTROL_TIMEOUT, control.update(update)).await {
        Ok(Ok(configuration)) => {
            tracing::info!(router_id = %target.router_id, model = %target.model, revision = configuration.revision(), "Updated router policy parameters");
            Json(PolicyResponse {
                target,
                configuration: Some(configuration),
            })
            .into_response()
        }
        Ok(Err(error)) => policy_error(error),
        Err(_) => policy_error(RuntimePolicyError::Unavailable),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::{
        body::{Body, to_bytes},
        http::Request,
    };
    use dynamo_custom_policy_builtin::default_factory;
    use dynamo_kv_router::services::selection::{
        SelectionCacheConfig, SelectionCore, WorkerRequest,
    };
    use dynamo_kv_router::{KvRouterConfig, RoutingPartitionId, WorkerType};
    use dynamo_runtime::protocols::EndpointId;
    use tower::ServiceExt;

    async fn fixture(manager: Arc<ModelManager>) -> (Router, SelectionCore, Uuid, Uuid) {
        let core = SelectionCore::try_new_local(
            KvRouterConfig {
                use_kv_events: false,
                ..Default::default()
            },
            1,
            Default::default(),
            SelectionCacheConfig::default(),
            default_factory(),
        )
        .unwrap();
        let mut ids = Vec::new();
        for (worker_id, model) in [(1, "first"), (2, "second")] {
            core.upsert_worker(WorkerRequest {
                worker_id,
                model_name: model.into(),
                routing_group: "default".into(),
                endpoint: Some(format!("http://worker-{worker_id}:8000")),
                block_size: Some(4),
                ..Default::default()
            })
            .await
            .unwrap();
            let partition = core
                .partition(&RoutingPartitionId::new(model, "default"))
                .unwrap();
            ids.push(manager.router_policies.register(
                model.into(),
                EndpointId {
                    namespace: "test".into(),
                    component: model.into(),
                    name: "generate".into(),
                },
                WorkerType::Aggregated,
                partition.scheduler().policy_control(),
            ));
        }
        (router_policy_router(manager).1, core, ids[0], ids[1])
    }

    async fn call(app: &Router, method: Method, uri: &str, body: Value) -> (StatusCode, Value) {
        let response = app
            .clone()
            .oneshot(
                Request::builder()
                    .method(method)
                    .uri(uri)
                    .header("content-type", "application/json")
                    .body(Body::from(body.to_string()))
                    .unwrap(),
            )
            .await
            .unwrap();
        let status = response.status();
        let bytes = to_bytes(response.into_body(), 1024 * 1024).await.unwrap();
        (status, serde_json::from_slice(&bytes).unwrap())
    }

    #[tokio::test]
    async fn busy_threshold_subresource_and_legacy_route_share_model_state() {
        use crate::discovery::{LoadThresholdConfig, LoadThresholdHandle, WorkerSet};

        let manager = Arc::new(ModelManager::new());
        let mut first_handles = Vec::new();
        for (model, namespace) in [("first", "old"), ("first", "new"), ("second", "other")] {
            let thresholds = LoadThresholdHandle::new(LoadThresholdConfig::default());
            let mut workers =
                WorkerSet::new(namespace.into(), "checksum".into(), Default::default());
            workers.load_thresholds = Some(thresholds.clone());
            manager
                .get_or_create_model(model)
                .add_worker_set(namespace.into(), Arc::new(workers));
            if model == "first" {
                first_handles.push(thresholds);
            }
        }
        let (app, core, first, _) = fixture(manager.clone()).await;
        let app =
            app.merge(super::super::busy_threshold::router(manager.clone(), "/busy_threshold").1);
        let path = "/router_policy/busy_threshold";
        let (status, updated) = call(&app, Method::POST, path, serde_json::json!({
            "model": "first", "active_decode_blocks_threshold": 0.85,
            "active_prefill_tokens_threshold": 1000, "active_prefill_tokens_threshold_frac": 0.8,
        })).await;
        assert_eq!(status, StatusCode::OK, "{updated}");
        let (status, legacy) = call(
            &app,
            Method::POST,
            "/busy_threshold",
            serde_json::json!({"model": "first"}),
        )
        .await;
        assert_eq!(status, StatusCode::OK);
        assert_eq!(legacy, updated);
        for handle in &first_handles {
            let config = handle.get();
            assert_eq!(config.active_decode_blocks_threshold, Some(0.85));
            assert_eq!(config.active_prefill_tokens_threshold, Some(1000));
            assert_eq!(config.active_prefill_tokens_threshold_frac, Some(0.8));
        }
        assert!(
            !manager
                .load_threshold_config("second", None)
                .unwrap()
                .is_configured()
        );

        let (status, legacy) = call(
            &app,
            Method::POST,
            "/busy_threshold",
            serde_json::json!({
                "model": "first", "active_prefill_tokens_threshold": 2000,
                "active_prefill_tokens_threshold_frac": null,
            }),
        )
        .await;
        assert_eq!(status, StatusCode::OK);
        let (_, current) = call(
            &app,
            Method::POST,
            path,
            serde_json::json!({"model": "first"}),
        )
        .await;
        assert_eq!(current, legacy);
        assert_eq!(current["active_decode_blocks_threshold"], 0.85);
        assert_eq!(current["active_prefill_tokens_threshold"], 2000);
        assert_eq!(current["active_prefill_tokens_threshold_frac"], 0.8);
        let (status, listing) = call(&app, Method::GET, path, Value::Null).await;
        assert_eq!(status, StatusCode::OK);
        assert_eq!(listing["thresholds"].as_array().unwrap().len(), 2);
        let (_, legacy_listing) = call(&app, Method::GET, "/busy_threshold", Value::Null).await;
        assert_eq!(listing, legacy_listing);

        for (body, expected) in [
            (
                serde_json::json!({"model": "first", "active_decode_blocks_threshold": 1.1, "active_prefill_tokens_threshold": 10}),
                StatusCode::BAD_REQUEST,
            ),
            (
                serde_json::json!({"model": "first", "active_prefill_tokens_threshold": -1}),
                StatusCode::UNPROCESSABLE_ENTITY,
            ),
            (
                serde_json::json!({"model": "missing", "active_decode_blocks_threshold": 0.9}),
                StatusCode::NOT_FOUND,
            ),
        ] {
            let (status, response) = call(&app, Method::POST, path, body).await;
            assert_eq!(status, expected, "{response}");
            assert!(response["error"].is_string());
            for handle in &first_handles {
                assert_eq!(handle.get().active_prefill_tokens_threshold, Some(2000));
                assert_eq!(handle.get().active_decode_blocks_threshold, Some(0.85));
            }
        }
        let (_, policy) = call(&app, Method::GET, "/router_policy?model=first", Value::Null).await;
        assert_eq!(policy["routers"][0]["router_id"], first.to_string());
        assert_eq!(policy["routers"][0]["configuration"]["revision"], 0);
        core.shutdown();
    }

    #[test]
    fn listing_skips_timed_out_routers_and_keeps_healthy_routers_editable() {
        let manager = Arc::new(ModelManager::new());
        // Leave these live actors on a runtime that stops polling after setup.
        let stalled_runtime = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .unwrap();
        let (_, stalled_core, stalled_id, _) =
            stalled_runtime.block_on(fixture(Arc::clone(&manager)));
        let healthy_runtime = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .start_paused(true)
            .build()
            .unwrap();
        healthy_runtime.block_on(async {
            let (app, core, first, second) = fixture(manager).await;
            let (status, listing) = call(&app, Method::GET, "/router_policy", Value::Null).await;
            assert_eq!(status, StatusCode::OK, "{listing}");
            let routers = listing["routers"].as_array().unwrap();
            assert_eq!(routers.len(), 2);
            for id in [first, second] {
                let router = routers
                    .iter()
                    .find(|r| r["router_id"] == id.to_string())
                    .unwrap();
                assert_eq!(router["configuration"]["revision"], 0);
            }
            let (status, updated) = call(
                &app,
                Method::PATCH,
                "/router_policy",
                serde_json::json!({"router_id": first, "expected_revision": 0,
                    "parameters": {"prefill_load_scale": 2.0}}),
            )
            .await;
            assert_eq!(status, StatusCode::OK, "{updated}");
            assert_eq!(
                updated["configuration"]["parameters"]["prefill_load_scale"],
                2.0
            );
            let (status, _) = call(
                &app,
                Method::PATCH,
                "/router_policy",
                serde_json::json!({"router_id": stalled_id, "expected_revision": 0, "reset": true}),
            )
            .await;
            assert_eq!(status, StatusCode::SERVICE_UNAVAILABLE);
            core.shutdown();
        });
        stalled_core.shutdown();
    }

    #[tokio::test]
    async fn http_updates_reach_the_live_policy_and_preserve_other_instances() {
        let (app, core, first, second) = fixture(Arc::new(ModelManager::new())).await;
        let (status, initial) =
            call(&app, Method::GET, "/router_policy?model=first", Value::Null).await;
        assert_eq!(status, StatusCode::OK);
        assert_eq!(initial["routers"].as_array().unwrap().len(), 1);
        assert_eq!(initial["routers"][0]["router_id"], first.to_string());
        assert_eq!(initial["routers"][0]["worker_type"], "aggregated");
        let startup = initial["routers"][0]["configuration"]["parameters"].clone();

        let (status, changed) = call(&app, Method::PATCH, "/router_policy", serde_json::json!({
            "router_id": first, "expected_revision": 0, "parameters": {"overlap_score_credit": 2.5, "decode_active_request_weight": 32.0}
        })).await;
        assert_eq!(status, StatusCode::OK, "{changed}");
        assert_eq!(changed["configuration"]["revision"], 1);
        let partition = core
            .partition(&RoutingPartitionId::new("first", "default"))
            .unwrap();
        let actual = partition.scheduler().policy_control().get().await.unwrap();
        assert_eq!(actual.parameters()["overlap_score_credit"], 2.5);
        assert_eq!(actual.parameters()["decode_active_request_weight"], 32.0);
        let (_, untouched) = call(
            &app,
            Method::GET,
            "/router_policy?model=second",
            Value::Null,
        )
        .await;
        assert_eq!(untouched["routers"][0]["router_id"], second.to_string());
        assert_eq!(untouched["routers"][0]["configuration"]["revision"], 0);

        for (body, expected) in [
            (
                serde_json::json!({"router_id": first, "expected_revision": 0, "reset": true}),
                StatusCode::CONFLICT,
            ),
            (
                serde_json::json!({"router_id": first, "expected_revision": 1, "parameters": {"overlap_score_credit": 8.0, "router_temperature": -1.0}}),
                StatusCode::UNPROCESSABLE_ENTITY,
            ),
            (
                serde_json::json!({"router_id": first, "expected_revision": 1, "parameters": {"routing_mode": "random"}}),
                StatusCode::UNPROCESSABLE_ENTITY,
            ),
            (
                serde_json::json!({"router_id": first, "expected_revision": 1, "parameters": {}, "reset": true}),
                StatusCode::BAD_REQUEST,
            ),
            (
                serde_json::json!({"router_id": Uuid::new_v4(), "expected_revision": 0, "reset": true}),
                StatusCode::NOT_FOUND,
            ),
            (
                serde_json::json!({"router_id": first, "reset": true}),
                StatusCode::UNPROCESSABLE_ENTITY,
            ),
            (
                serde_json::json!({"router_id": first, "expected_revision": 1, "parameters": {"router_temperature": "x".repeat(16 * 1024)}}),
                StatusCode::PAYLOAD_TOO_LARGE,
            ),
        ] {
            let (status, response) = call(&app, Method::PATCH, "/router_policy", body).await;
            assert_eq!(status, expected, "{response}");
            assert_eq!(
                partition.scheduler().policy_control().get().await.unwrap(),
                actual
            );
        }
        let (status, reset) = call(
            &app,
            Method::PATCH,
            "/router_policy",
            serde_json::json!({"router_id": first, "expected_revision": 1, "reset": true}),
        )
        .await;
        assert_eq!(status, StatusCode::OK);
        assert_eq!(reset["configuration"]["revision"], 2);
        assert_eq!(reset["configuration"]["parameters"], startup);
        core.shutdown();
    }
}
