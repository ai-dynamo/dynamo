// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! First aggregated Global Router HTTP request path.
//!
//! A regional ingress invokes this router once. The selected pool's endpoint
//! is a private local Frontend base URL, never another regional ingress.

use std::sync::Arc;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use axum::Router;
use axum::body::{Body, to_bytes};
use axum::extract::State;
use axum::http::{HeaderMap, StatusCode, header};
use axum::response::{IntoResponse, Response};
use axum::routing::post;
use dynamo_kv_router::global_view::selection::LoadPoolSelector;
use dynamo_kv_router::global_view::state::{FreshnessPolicy, PoolStateRepository};
use serde_json::{Value, json};

const MAX_REQUEST_BYTES: usize = 16 * 1024 * 1024;
const ROUTER_HOP_HEADER: &str = "x-dynamo-global-router-hop";

/// Load-based aggregated routing from a regional ingress to private Frontends.
pub struct GlobalRouterHttp {
    selector: LoadPoolSelector,
    client: reqwest::Client,
}

impl GlobalRouterHttp {
    pub fn new(
        repository: Arc<dyn PoolStateRepository>,
        freshness: FreshnessPolicy,
    ) -> Result<Self, reqwest::Error> {
        let client = reqwest::Client::builder()
            .connect_timeout(Duration::from_secs(5))
            .redirect(reqwest::redirect::Policy::none())
            .build()?;
        Ok(Self {
            selector: LoadPoolSelector::new(repository, freshness),
            client,
        })
    }

    /// The caller can merge this with Global View's optional diagnostics router.
    pub fn router(self: Arc<Self>) -> Router {
        Router::new()
            .route("/v1/chat/completions", post(chat_completions))
            .route("/v1/completions", post(completions))
            .with_state(self)
    }

    async fn forward(&self, path: &'static str, headers: HeaderMap, body: Body) -> Response {
        if headers.contains_key(ROUTER_HOP_HEADER) {
            return error(
                StatusCode::LOOP_DETECTED,
                "request already crossed Global Router",
            );
        }
        let bytes = match to_bytes(body, MAX_REQUEST_BYTES).await {
            Ok(bytes) => bytes,
            Err(_) => return error(StatusCode::PAYLOAD_TOO_LARGE, "request body is too large"),
        };
        let parsed: Value = match serde_json::from_slice(&bytes) {
            Ok(parsed) => parsed,
            Err(_) => return error(StatusCode::BAD_REQUEST, "request body must be JSON"),
        };
        let Some(model) = parsed
            .get("model")
            .and_then(Value::as_str)
            .filter(|model| !model.is_empty())
        else {
            return error(StatusCode::BAD_REQUEST, "request must name a model");
        };
        let now_unix_ms = match SystemTime::now().duration_since(UNIX_EPOCH) {
            Ok(elapsed) => u64::try_from(elapsed.as_millis()).unwrap_or(u64::MAX),
            Err(_) => return error(StatusCode::INTERNAL_SERVER_ERROR, "system clock is invalid"),
        };
        let Some(decision) = self.selector.select(model, now_unix_ms) else {
            return error(
                StatusCode::SERVICE_UNAVAILABLE,
                "no ready pool serves the requested model",
            );
        };
        let mut endpoint = match reqwest::Url::parse(&decision.frontend_endpoint) {
            Ok(endpoint)
                if matches!(endpoint.scheme(), "http" | "https")
                    && endpoint.host().is_some()
                    && endpoint.path() == "/"
                    && endpoint.query().is_none()
                    && endpoint.fragment().is_none()
                    && endpoint.username().is_empty()
                    && endpoint.password().is_none() =>
            {
                endpoint
            }
            _ => {
                tracing::error!(pool_id = %decision.pool_id, "pool has invalid private Frontend URL");
                return error(
                    StatusCode::BAD_GATEWAY,
                    "selected pool has an invalid Frontend URL",
                );
            }
        };
        endpoint.set_path(path);
        tracing::info!(
            pool_id = %decision.pool_id,
            model,
            cost = decision.cost,
            basis = ?decision.basis,
            "forwarding to selected pool"
        );
        let mut request = self
            .client
            .post(endpoint)
            .header(header::CONTENT_TYPE, "application/json")
            .header(ROUTER_HOP_HEADER, "1")
            .body(bytes);
        for name in [header::ACCEPT, header::USER_AGENT] {
            if let Some(value) = headers.get(&name) {
                request = request.header(name, value.clone());
            }
        }
        if let Some(value) = headers.get("x-request-id") {
            request = request.header("x-request-id", value.clone());
        }
        let upstream = match request.send().await {
            Ok(upstream) => upstream,
            Err(cause) => {
                tracing::warn!(pool_id = %decision.pool_id, %cause, "private Frontend request failed");
                return error(StatusCode::BAD_GATEWAY, "selected Frontend is unavailable");
            }
        };
        if upstream.status().is_redirection() {
            tracing::warn!(pool_id = %decision.pool_id, status = %upstream.status(), "private Frontend redirected request");
            return error(
                StatusCode::BAD_GATEWAY,
                "selected Frontend redirected the request",
            );
        }
        let status = upstream.status();
        let response_headers = upstream.headers().clone();
        let mut response = Response::new(Body::from_stream(upstream.bytes_stream()));
        *response.status_mut() = status;
        for name in [header::CONTENT_TYPE, header::CACHE_CONTROL] {
            if let Some(value) = response_headers.get(&name) {
                response.headers_mut().insert(name, value.clone());
            }
        }
        if let Some(value) = response_headers.get("x-request-id") {
            response.headers_mut().insert("x-request-id", value.clone());
        }
        response
    }
}

async fn chat_completions(
    State(router): State<Arc<GlobalRouterHttp>>,
    headers: HeaderMap,
    body: Body,
) -> Response {
    router.forward("/v1/chat/completions", headers, body).await
}

async fn completions(
    State(router): State<Arc<GlobalRouterHttp>>,
    headers: HeaderMap,
    body: Body,
) -> Response {
    router.forward("/v1/completions", headers, body).await
}

fn error(status: StatusCode, message: &'static str) -> Response {
    (
        status,
        axum::Json(json!({ "error": { "message": message } })),
    )
        .into_response()
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;

    use super::*;
    use dynamo_kv_router::global_view::state::{
        InMemoryPoolStateRepository, PoolCapacity, PoolDescriptors, PoolLoad, PoolLocation,
        PoolRole, PoolSignalStatus, PoolState, PoolStateSink, ServingReadiness, SignalState,
        SignalStatus,
    };
    use dynamo_kv_router::global_view::{PoolIdDeriver, PoolKey, V1PoolIdDeriver};

    fn freshness() -> FreshnessPolicy {
        FreshnessPolicy {
            catalog_max_age_ms: 60_000,
            readiness_max_age_ms: 60_000,
            capacity_max_age_ms: 60_000,
            load_max_age_ms: 60_000,
            kv_usage_max_age_ms: 60_000,
            kv_overlap_max_age_ms: 60_000,
        }
    }

    fn repository(frontend: String) -> Arc<InMemoryPoolStateRepository> {
        let now = u64::try_from(
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_millis(),
        )
        .unwrap();
        let status = SignalStatus {
            state: SignalState::Complete,
            received_at_unix_ms: Some(now),
            ..Default::default()
        };
        let repo = Arc::new(InMemoryPoolStateRepository::default());
        repo.replace(PoolState {
            pool_id: V1PoolIdDeriver.derive(&PoolKey::new("ohio", "dynamo", "mocker").unwrap()),
            descriptors: PoolDescriptors {
                site_id: "ohio".into(),
                namespace: "dynamo".into(),
                dgd_name: "mocker".into(),
                location: PoolLocation {
                    region: "us-east-2".into(),
                    availability_zone: None,
                    cluster: None,
                    datacenter: None,
                },
                models: vec!["model".into()],
                model_readiness: BTreeMap::from([("model".into(), ServingReadiness::Ready)]),
                roles: vec![PoolRole::Aggregated],
                frontend_endpoint: Some(frontend),
                hardware: Vec::new(),
            },
            capacity: PoolCapacity::default(),
            load: PoolLoad::default(),
            signal_status: PoolSignalStatus {
                catalog: status.clone(),
                readiness: status,
                ..Default::default()
            },
        });
        repo
    }

    #[tokio::test]
    async fn forwards_json_to_private_frontend_and_returns_sse() {
        async fn backend(headers: HeaderMap, body: axum::body::Bytes) -> Response {
            assert_eq!(headers.get(ROUTER_HOP_HEADER).unwrap(), "1");
            assert_eq!(headers.get("x-request-id").unwrap(), "caller-id");
            assert!(headers.get(header::AUTHORIZATION).is_none());
            assert_eq!(
                body.as_ref(),
                br#"{"model":"model","stream":true,"prompt":"hi"}"#
            );
            Response::builder()
                .status(StatusCode::OK)
                .header(header::CONTENT_TYPE, "text/event-stream")
                .header("x-request-id", "backend-id")
                .body(Body::from("data: one\n\ndata: [DONE]\n\n"))
                .unwrap()
        }
        let backend_listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let backend_address = backend_listener.local_addr().unwrap();
        let backend_task = tokio::spawn(async move {
            axum::serve(
                backend_listener,
                Router::new().route("/v1/chat/completions", post(backend)),
            )
            .await
            .unwrap();
        });

        let router = Arc::new(
            GlobalRouterHttp::new(repository(format!("http://{backend_address}")), freshness())
                .unwrap(),
        );
        let router_listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let router_address = router_listener.local_addr().unwrap();
        let router_task = tokio::spawn(async move {
            axum::serve(router_listener, router.router()).await.unwrap();
        });
        let response = reqwest::Client::new()
            .post(format!("http://{router_address}/v1/chat/completions"))
            .header("x-request-id", "caller-id")
            .header(header::AUTHORIZATION, "Bearer caller-secret")
            .body(r#"{"model":"model","stream":true,"prompt":"hi"}"#)
            .send()
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::OK);
        assert_eq!(
            response.headers()[header::CONTENT_TYPE],
            "text/event-stream"
        );
        assert_eq!(response.headers()["x-request-id"], "backend-id");
        assert_eq!(
            response.text().await.unwrap(),
            "data: one\n\ndata: [DONE]\n\n"
        );
        router_task.abort();
        backend_task.abort();
    }

    #[tokio::test]
    async fn rejects_frontend_redirect_back_to_regional_ingress() {
        async fn backend() -> Response {
            Response::builder()
                .status(StatusCode::TEMPORARY_REDIRECT)
                .header(
                    header::LOCATION,
                    "https://us-west-2.api.dynamo.com/v1/completions",
                )
                .body(Body::empty())
                .unwrap()
        }
        let backend_listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let backend_address = backend_listener.local_addr().unwrap();
        let backend_task = tokio::spawn(async move {
            axum::serve(
                backend_listener,
                Router::new().route("/v1/completions", post(backend)),
            )
            .await
            .unwrap();
        });
        let router = Arc::new(
            GlobalRouterHttp::new(repository(format!("http://{backend_address}")), freshness())
                .unwrap(),
        );
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let address = listener.local_addr().unwrap();
        let task = tokio::spawn(async move {
            axum::serve(listener, router.router()).await.unwrap();
        });
        let response = reqwest::Client::new()
            .post(format!("http://{address}/v1/completions"))
            .body(r#"{"model":"model","prompt":"hi"}"#)
            .send()
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::BAD_GATEWAY);
        assert!(response.headers().get(header::LOCATION).is_none());
        task.abort();
        backend_task.abort();
    }

    #[tokio::test]
    async fn rejects_second_global_routing_hop() {
        let router = Arc::new(
            GlobalRouterHttp::new(repository("http://127.0.0.1:1".into()), freshness()).unwrap(),
        );
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let address = listener.local_addr().unwrap();
        let task = tokio::spawn(async move {
            axum::serve(listener, router.router()).await.unwrap();
        });
        let response = reqwest::Client::new()
            .post(format!("http://{address}/v1/completions"))
            .header(ROUTER_HOP_HEADER, "1")
            .body(r#"{"model":"model","prompt":"hi"}"#)
            .send()
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::LOOP_DETECTED);
        task.abort();
    }
}
