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
use dynamo_kv_router::global_view::overlap::KvOverlapScorer;
use dynamo_kv_router::global_view::selection::LoadPoolSelector;
use dynamo_kv_router::global_view::state::{FreshnessPolicy, PoolStateRepository};
use serde_json::{Value, json};

use super::credit_selection::CreditPoolSelector;
use super::scheduler_metrics::SchedulerLoadRepository;

const MAX_REQUEST_BYTES: usize = 16 * 1024 * 1024;
const MAX_OVERLAP_TOKEN_IDS: usize = 32 * 1024;
const ROUTER_HOP_HEADER: &str = "x-dynamo-global-router-hop";

/// Load-based aggregated routing from a regional ingress to private Frontends.
pub struct GlobalRouterHttp {
    selector: LoadPoolSelector,
    overlap: Option<Arc<dyn KvOverlapScorer>>,
    credit: Option<CreditPoolSelector>,
    client: reqwest::Client,
}

impl GlobalRouterHttp {
    pub fn new(
        repository: Arc<dyn PoolStateRepository>,
        freshness: FreshnessPolicy,
    ) -> Result<Self, reqwest::Error> {
        Self::new_with_optional_overlap(repository, freshness, None)
    }

    /// Enable the first KV experiment for single token-ID completion prompts.
    /// All other request shapes continue to use the load policy.
    pub fn new_with_overlap(
        repository: Arc<dyn PoolStateRepository>,
        freshness: FreshnessPolicy,
        overlap: Arc<dyn KvOverlapScorer>,
    ) -> Result<Self, reqwest::Error> {
        Self::new_with_optional_overlap(repository, freshness, Some(overlap))
    }

    fn new_with_optional_overlap(
        repository: Arc<dyn PoolStateRepository>,
        freshness: FreshnessPolicy,
        overlap: Option<Arc<dyn KvOverlapScorer>>,
    ) -> Result<Self, reqwest::Error> {
        let client = reqwest::Client::builder()
            .connect_timeout(Duration::from_secs(5))
            .redirect(reqwest::redirect::Policy::none())
            .build()?;
        Ok(Self {
            selector: LoadPoolSelector::new(repository, freshness),
            overlap,
            credit: None,
            client,
        })
    }

    pub fn new_with_credit(
        repository: Arc<dyn PoolStateRepository>,
        freshness: FreshnessPolicy,
        overlap: Arc<dyn KvOverlapScorer>,
        scheduler: Arc<dyn SchedulerLoadRepository>,
        credit: f64,
    ) -> Result<Self, reqwest::Error> {
        let mut router = Self::new(repository.clone(), freshness)?;
        router.credit = Some(CreditPoolSelector::new(
            repository, scheduler, overlap, freshness, credit,
        ));
        Ok(router)
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
        // The first global cache contract is unsalted base-model token IDs.
        // Extension-bearing requests retain the existing load-only behavior.
        let token_ids = (path == "/v1/completions"
            && parsed.get("nvext").is_none()
            && parsed.get("cache_salt").is_none()
            && parsed.get("cache_namespace").is_none())
        .then(|| parsed.get("prompt").and_then(single_token_id_prompt))
        .flatten();
        let (decision, policy) =
            if let (Some(credit), Some(tokens)) = (&self.credit, token_ids.as_deref()) {
                match credit.select(model, tokens, now_unix_ms) {
                    Some(decision) => (Some(decision), "dynamo_credit"),
                    None => (
                        self.selector.select(model, now_unix_ms),
                        "credit_fallback_request_count",
                    ),
                }
            } else {
                match (self.overlap.as_ref(), token_ids.as_deref()) {
                    (Some(scorer), Some(token_ids)) => (
                        self.selector.select_with_overlap(
                            model,
                            token_ids,
                            scorer.as_ref(),
                            now_unix_ms,
                        ),
                        "prefix_then_load",
                    ),
                    _ => (self.selector.select(model, now_unix_ms), "request_count"),
                }
            };
        let Some(decision) = decision else {
            return error(
                StatusCode::SERVICE_UNAVAILABLE,
                "no ready pool serves the requested model",
            );
        };
        let Some(mut endpoint) = parse_private_frontend_base(&decision.frontend_endpoint) else {
            tracing::error!(pool_id = %decision.pool_id, "pool has invalid private Frontend URL");
            return error(
                StatusCode::BAD_GATEWAY,
                "selected pool has an invalid Frontend URL",
            );
        };
        endpoint.set_path(path);
        tracing::info!(
            request_id = headers.get("x-request-id").and_then(|v| v.to_str().ok()).unwrap_or(""),
            policy,
            load_sample_age_ms = decision.load_status.received_at_unix_ms.map(|v| now_unix_ms.saturating_sub(v)),
            assignments_since_sample = decision.assignments_since_sample,
            pool_id = %decision.pool_id,
            target_region = %decision.region,
            model,
            cost = decision.cost,
            basis = ?decision.basis,
            matched_prefix_tokens = ?decision.matched_prefix_tokens,
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
        // Private POC diagnostics let concurrent clients attribute each response
        // without inferring its destination from global counters.
        for (name, value) in [
            ("x-dynamo-pool-id", decision.pool_id.to_string()),
            ("x-dynamo-target-region", decision.region.clone()),
            ("x-dynamo-routing-cost", decision.cost.to_string()),
            ("x-dynamo-routing-basis", policy.to_owned()),
            (
                "x-dynamo-load-age-ms",
                decision
                    .load_status
                    .received_at_unix_ms
                    .map(|v| now_unix_ms.saturating_sub(v).to_string())
                    .unwrap_or_default(),
            ),
        ] {
            if let Ok(value) = value.parse() {
                response.headers_mut().insert(name, value);
            }
        }
        if let Some(tokens) = decision.matched_prefix_tokens {
            response.headers_mut().insert(
                "x-dynamo-matched-prefix-tokens",
                tokens.to_string().parse().unwrap(),
            );
        }
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

/// OpenAI completions can carry the exact token IDs that the Frontend serves.
/// Text, batched, malformed, or oversized prompts use the existing load path.
fn single_token_id_prompt(value: &Value) -> Option<Vec<u32>> {
    let values = value.as_array()?;
    if values.is_empty() || values.len() > MAX_OVERLAP_TOKEN_IDS {
        return None;
    }
    values
        .iter()
        .map(|value| u32::try_from(value.as_u64()?).ok())
        .collect()
}

/// Validate the configured local Frontend base URL before a request is sent.
/// Private reachability is enforced by deployment networking, not by DNS text.
pub(crate) fn parse_private_frontend_base(raw: &str) -> Option<reqwest::Url> {
    let endpoint = reqwest::Url::parse(raw).ok()?;
    (matches!(endpoint.scheme(), "http" | "https")
        && endpoint.host().is_some()
        && endpoint.path() == "/"
        && endpoint.query().is_none()
        && endpoint.fragment().is_none()
        && endpoint.username().is_empty()
        && endpoint.password().is_none())
    .then_some(endpoint)
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
    use std::collections::{BTreeMap, HashMap};

    use super::*;
    use dynamo_kv_router::global_view::state::{
        InMemoryPoolStateRepository, ModelRequestLoad, PoolCapacity, PoolDescriptors, PoolLoad,
        PoolLocation, PoolRole, PoolSignalStatus, PoolState, PoolStateSink, ServingReadiness,
        SignalState, SignalStatus,
    };
    use dynamo_kv_router::global_view::{PoolId, PoolIdDeriver, PoolKey, V1PoolIdDeriver};

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
    async fn ohio_ingress_forwards_to_lower_load_west_frontend() {
        let ohio_listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let ohio_address = ohio_listener.local_addr().unwrap();
        let ohio_task = tokio::spawn(async move {
            axum::serve(
                ohio_listener,
                Router::new().route("/v1/completions", post(|| async { "ohio" })),
            )
            .await
            .unwrap();
        });
        let west_listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let west_address = west_listener.local_addr().unwrap();
        let west_task = tokio::spawn(async move {
            axum::serve(
                west_listener,
                Router::new().route("/v1/completions", post(|| async { "west" })),
            )
            .await
            .unwrap();
        });

        let repo = repository(format!("http://{ohio_address}"));
        let now = u64::try_from(
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_millis(),
        )
        .unwrap();
        let mut ohio = repo.list(now, &freshness()).pop().unwrap();
        let mut west = ohio.clone();
        west.pool_id = V1PoolIdDeriver.derive(&PoolKey::new("west", "dynamo", "mocker").unwrap());
        west.descriptors.site_id = "west".into();
        west.descriptors.location.region = "us-west-2".into();
        west.descriptors.frontend_endpoint = Some(format!("http://{west_address}"));
        for (pool, requests) in [(&mut ohio, 20), (&mut west, 0)] {
            pool.load.request_plane.insert(
                "model".into(),
                ModelRequestLoad {
                    pending_first_output_requests: Some(requests),
                    output_generation_requests: Some(0),
                    ..Default::default()
                },
            );
            pool.signal_status.load = SignalStatus {
                state: SignalState::Complete,
                received_at_unix_ms: Some(now),
                ..Default::default()
            };
        }
        repo.replace(ohio);
        repo.replace(west);
        let router = Arc::new(GlobalRouterHttp::new(repo, freshness()).unwrap());
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
        assert_eq!(response.status(), StatusCode::OK);
        assert_eq!(response.text().await.unwrap(), "west");
        task.abort();
        ohio_task.abort();
        west_task.abort();
    }

    struct FixedOverlap(HashMap<PoolId, u64>);

    impl KvOverlapScorer for FixedOverlap {
        fn estimate_matched_prefix_tokens(
            &self,
            pool_id: &PoolId,
            _model: &str,
            _token_ids: &[u32],
        ) -> Option<u64> {
            self.0.get(pool_id).copied()
        }
    }

    #[tokio::test]
    async fn credit_ablation_and_missing_scheduler_fallback_are_observable() {
        use super::super::scheduler_metrics::SchedulerSnapshot;
        struct Samples(parking_lot::RwLock<HashMap<PoolId, SchedulerSnapshot>>);
        impl SchedulerLoadRepository for Samples {
            fn snapshot(&self, pool: &PoolId, _: &str, now: u64) -> Option<SchedulerSnapshot> {
                let sample = *self.0.read().get(pool)?;
                (now.saturating_sub(sample.collected_at_unix_ms) < 2_000).then_some(sample)
            }
        }
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let endpoint = format!("http://{}", listener.local_addr().unwrap());
        let task = tokio::spawn(async move {
            axum::serve(
                listener,
                Router::new().route("/v1/completions", post(|| async { "ok" })),
            )
            .await
            .unwrap();
        });
        let repo = repository(endpoint);
        let now = super::super::scheduler_metrics::now_ms();
        let mut ohio = repo.list(now, &freshness()).pop().unwrap();
        let mut west = ohio.clone();
        west.pool_id = V1PoolIdDeriver.derive(&PoolKey::new("west", "dynamo", "mocker").unwrap());
        west.descriptors.location.region = "us-west-2".into();
        let ohio_id = ohio.pool_id.clone();
        let west_id = west.pool_id.clone();
        for (pool, count) in [(&mut ohio, 0), (&mut west, 1)] {
            pool.capacity.live_workers = Some(2);
            pool.capacity.max_concurrency = Some(512);
            pool.load.request_plane.insert(
                "model".into(),
                ModelRequestLoad {
                    pending_first_output_requests: Some(count),
                    output_generation_requests: Some(0),
                    ..Default::default()
                },
            );
            pool.signal_status.capacity = pool.signal_status.catalog.clone();
            pool.signal_status.load = pool.signal_status.catalog.clone();
            pool.signal_status.kv_overlap = pool.signal_status.catalog.clone();
        }
        repo.replace(ohio);
        repo.replace(west);
        let samples = Arc::new(Samples(parking_lot::RwLock::new(HashMap::from([
            (
                ohio_id.clone(),
                SchedulerSnapshot {
                    prefill_tokens: 0,
                    decode_blocks: 0,
                    block_size: 64,
                    collected_at_unix_ms: now,
                },
            ),
            (
                west_id.clone(),
                SchedulerSnapshot {
                    prefill_tokens: 64,
                    decode_blocks: 0,
                    block_size: 64,
                    collected_at_unix_ms: now,
                },
            ),
        ]))));
        let overlap = Arc::new(FixedOverlap(HashMap::from([
            (ohio_id, 0),
            (west_id.clone(), 1024),
        ])));
        let body = || {
            Body::from(serde_json::json!({"model": "model", "prompt": vec![1; 1024]}).to_string())
        };
        let zero = GlobalRouterHttp::new_with_credit(
            repo.clone(),
            freshness(),
            overlap.clone(),
            samples.clone(),
            0.0,
        )
        .unwrap();
        let one =
            GlobalRouterHttp::new_with_credit(repo, freshness(), overlap, samples.clone(), 1.0)
                .unwrap();
        let response = zero
            .forward("/v1/completions", HeaderMap::new(), body())
            .await;
        assert_eq!(response.headers()["x-dynamo-target-region"], "us-east-2");
        assert_eq!(
            response.headers()["x-dynamo-routing-basis"],
            "dynamo_credit"
        );
        let response = one
            .forward("/v1/completions", HeaderMap::new(), body())
            .await;
        assert_eq!(response.headers()["x-dynamo-target-region"], "us-west-2");
        assert_eq!(response.headers()["x-dynamo-matched-prefix-tokens"], "1024");
        // Missing a ready pool's scheduler sample must not make it look idle or
        // remove it from eligibility. The host falls back across BOTH pools.
        samples.0.write().remove(&west_id);
        let response = one
            .forward("/v1/completions", HeaderMap::new(), body())
            .await;
        assert_eq!(
            response.headers()["x-dynamo-routing-basis"],
            "credit_fallback_request_count"
        );
        assert_eq!(response.headers()["x-dynamo-target-region"], "us-east-2");
        let response = one.forward("/v1/completions", HeaderMap::new(), Body::from(
            serde_json::json!({"model":"model", "prompt":[1,2], "nvext":{"cache_salt":"tenant"}}).to_string()
        )).await;
        assert_eq!(
            response.headers()["x-dynamo-routing-basis"],
            "request_count"
        );
        task.abort();
    }

    #[tokio::test]
    async fn token_id_completion_uses_kv_while_text_completion_uses_load() {
        let ohio_listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let ohio_address = ohio_listener.local_addr().unwrap();
        let ohio_task = tokio::spawn(async move {
            axum::serve(
                ohio_listener,
                Router::new().route("/v1/completions", post(|| async { "ohio" })),
            )
            .await
            .unwrap();
        });
        let west_listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let west_address = west_listener.local_addr().unwrap();
        let west_task = tokio::spawn(async move {
            axum::serve(
                west_listener,
                Router::new().route("/v1/completions", post(|| async { "west" })),
            )
            .await
            .unwrap();
        });
        let repo = repository(format!("http://{ohio_address}"));
        let now = u64::try_from(
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_millis(),
        )
        .unwrap();
        let mut ohio = repo.list(now, &freshness()).pop().unwrap();
        let mut west = ohio.clone();
        west.pool_id = V1PoolIdDeriver.derive(&PoolKey::new("west", "dynamo", "mocker").unwrap());
        west.descriptors.site_id = "west".into();
        west.descriptors.location.region = "us-west-2".into();
        west.descriptors.frontend_endpoint = Some(format!("http://{west_address}"));
        let ohio_id = ohio.pool_id.clone();
        let west_id = west.pool_id.clone();
        for (pool, requests) in [(&mut ohio, 20), (&mut west, 0)] {
            pool.load.request_plane.insert(
                "model".into(),
                ModelRequestLoad {
                    pending_first_output_requests: Some(requests),
                    output_generation_requests: Some(0),
                    ..Default::default()
                },
            );
            pool.signal_status.load = SignalStatus {
                state: SignalState::Complete,
                received_at_unix_ms: Some(now),
                ..Default::default()
            };
            pool.signal_status.kv_overlap = SignalStatus {
                state: SignalState::Complete,
                received_at_unix_ms: Some(now),
                ..Default::default()
            };
        }
        repo.replace(ohio);
        repo.replace(west);
        let scorer: Arc<dyn KvOverlapScorer> =
            Arc::new(FixedOverlap(HashMap::from([(ohio_id, 64), (west_id, 0)])));
        let router =
            Arc::new(GlobalRouterHttp::new_with_overlap(repo, freshness(), scorer).unwrap());
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let address = listener.local_addr().unwrap();
        let task = tokio::spawn(async move {
            axum::serve(listener, router.router()).await.unwrap();
        });
        let client = reqwest::Client::new();
        let kv_response = client
            .post(format!("http://{address}/v1/completions"))
            .body(r#"{"model":"model","prompt":[1,2,3,4]}"#)
            .send()
            .await
            .unwrap();
        assert_eq!(kv_response.status(), StatusCode::OK);
        assert_eq!(kv_response.text().await.unwrap(), "ohio");
        let text_response = client
            .post(format!("http://{address}/v1/completions"))
            .body(r#"{"model":"model","prompt":"hello"}"#)
            .send()
            .await
            .unwrap();
        assert_eq!(text_response.status(), StatusCode::OK);
        assert_eq!(text_response.text().await.unwrap(), "west");
        task.abort();
        ohio_task.abort();
        west_task.abort();
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
