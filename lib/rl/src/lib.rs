// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Dynamo RL admin control plane — generic fan-out facade for `/v1/rl/engine`.
//!
//! ## Architecture
//!
//! Workers register an `rl` endpoint on the request plane at startup:
//! ```text
//! dyn://<namespace>.<component>.rl
//! ```
//!
//! The frontend discovers live `rl` instances via the discovery plane and fans
//! out `{method, kwargs}` payloads via strict request-plane direct routing
//! (NATS / shared TCP). The Python `rl_dispatch` handler on each worker
//! receives the payload and dispatches to the appropriate registered handler.
//!
//! ## HTTP surface (mounted on the frontend)
//!
//! ```text
//! POST /v1/rl/engine   fan-out or direct call, selected by body
//! GET  /v1/rl/engine   list registered methods per live worker
//! ```
//!
//! Enabled when `DYN_ENABLE_RL_ENDPOINTS=true`, on port `DYN_RL_PORT`
//! (default: 8002 if unset, or shares the main port).

use std::{
    collections::{HashMap, hash_map::DefaultHasher},
    hash::{Hash, Hasher},
    sync::Arc,
    time::Duration,
};

use axum::{Json, Router, extract::State, http::StatusCode, response::IntoResponse, routing::post};
use dynamo_runtime::{
    DistributedRuntime,
    component::Client,
    discovery::{DiscoveryInstance, DiscoveryQuery},
    pipeline::{
        SingleIn,
        network::egress::push_router::{PushRouter, RouterMode},
    },
    protocols::annotated::Annotated,
};
use futures::{FutureExt, StreamExt};

// ---------------------------------------------------------------------------
// Public constants
// ---------------------------------------------------------------------------

pub const DEFAULT_RL_ENDPOINT: &str = "rl";
pub const DEFAULT_REQUEST_TIMEOUT_SECS: u64 = 30;

// ---------------------------------------------------------------------------
// Error types
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
pub enum RlError {
    NoWorkers {
        namespace: String,
        rl_endpoint: String,
    },
    MembershipChanged {
        before_epoch: u64,
        after_epoch: u64,
    },
    UnknownInstance {
        instance_id: u64,
    },
}

impl std::fmt::Display for RlError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            RlError::NoWorkers {
                namespace,
                rl_endpoint,
            } => write!(
                f,
                "no live RL workers found in namespace '{namespace}' for endpoint '{rl_endpoint}'"
            ),
            RlError::MembershipChanged {
                before_epoch,
                after_epoch,
            } => write!(
                f,
                "RL worker membership changed during fan-out (before={before_epoch}, after={after_epoch})"
            ),
            RlError::UnknownInstance { instance_id } => {
                write!(f, "instance_id {instance_id} not found in live worker set")
            }
        }
    }
}

impl std::error::Error for RlError {}

// ---------------------------------------------------------------------------
// Client configuration
// ---------------------------------------------------------------------------

pub struct RlClientConfig {
    pub runtime: Arc<DistributedRuntime>,
    pub namespace: String,
    /// Worker endpoint name (default: "rl")
    pub rl_endpoint: String,
    /// Per-worker call timeout default; overridden per call via `timeout_secs`
    pub default_request_timeout: Duration,
}

impl RlClientConfig {
    pub fn new(runtime: Arc<DistributedRuntime>, namespace: impl Into<String>) -> Self {
        Self {
            runtime,
            namespace: namespace.into(),
            rl_endpoint: DEFAULT_RL_ENDPOINT.to_string(),
            default_request_timeout: Duration::from_secs(DEFAULT_REQUEST_TIMEOUT_SECS),
        }
    }
}

// ---------------------------------------------------------------------------
// Wire types
// ---------------------------------------------------------------------------

/// Payload sent from the frontend to each worker's `rl` endpoint.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct RlRequest {
    pub method: String,
    #[serde(default)]
    pub kwargs: serde_json::Value,
}

/// Per-worker result collected by the frontend after fan-out.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct WorkerResult {
    pub instance_id: u64,
    pub component: String,
    pub status: WorkerStatus,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub response: Option<serde_json::Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum WorkerStatus {
    Ok,
    Error,
}

impl WorkerResult {
    fn ok(target: &WorkerTarget, response: serde_json::Value) -> Self {
        Self {
            instance_id: target.instance_id,
            component: target.component.clone(),
            status: WorkerStatus::Ok,
            response: Some(response),
            error: None,
        }
    }

    fn error(target: &WorkerTarget, error: impl Into<String>) -> Self {
        Self {
            instance_id: target.instance_id,
            component: target.component.clone(),
            status: WorkerStatus::Error,
            response: None,
            error: Some(error.into()),
        }
    }
}

// ---------------------------------------------------------------------------
// Discovery / membership
// ---------------------------------------------------------------------------

#[derive(
    Debug, Clone, serde::Serialize, serde::Deserialize, PartialEq, Eq, Hash, PartialOrd, Ord,
)]
pub struct WorkerTarget {
    pub namespace: String,
    pub component: String,
    pub endpoint: String,
    pub instance_id: u64,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct MembershipSnapshot {
    pub epoch: u64,
    pub targets: Vec<WorkerTarget>,
}

impl MembershipSnapshot {
    fn new(mut targets: Vec<WorkerTarget>) -> Self {
        targets.sort();
        targets.dedup();

        let mut hasher = DefaultHasher::new();
        targets.hash(&mut hasher);
        let epoch = hasher.finish();

        Self { epoch, targets }
    }

    pub fn is_empty(&self) -> bool {
        self.targets.is_empty()
    }
}

/// Aggregated result from a fan-out call.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct FanoutReport {
    pub epoch: u64,
    pub workers: Vec<WorkerResult>,
}

// ---------------------------------------------------------------------------
// Per-call options
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
pub struct CallOptions {
    pub timeout: Duration,
    /// Component names to restrict fan-out. `None` = all components.
    pub components: Option<Vec<String>>,
}

impl CallOptions {
    pub fn new(timeout: Duration) -> Self {
        Self {
            timeout,
            components: None,
        }
    }

    pub fn with_components(mut self, components: Vec<String>) -> Self {
        self.components = Some(components);
        self
    }
}

// ---------------------------------------------------------------------------
// RlClient
// ---------------------------------------------------------------------------

#[derive(Clone)]
pub struct RlClient {
    runtime: Arc<DistributedRuntime>,
    namespace: String,
    rl_endpoint: String,
    default_request_timeout: Duration,
}

impl RlClient {
    pub fn new(config: RlClientConfig) -> anyhow::Result<Self> {
        if config.namespace.trim().is_empty() {
            anyhow::bail!("RlClientConfig.namespace must not be empty");
        }
        if config.rl_endpoint.trim().is_empty() {
            anyhow::bail!("RlClientConfig.rl_endpoint must not be empty");
        }
        Ok(Self {
            runtime: config.runtime,
            namespace: config.namespace,
            rl_endpoint: config.rl_endpoint,
            default_request_timeout: config.default_request_timeout,
        })
    }

    /// Snapshot the current live worker set.
    pub async fn snapshot(&self, opts: &CallOptions) -> anyhow::Result<MembershipSnapshot> {
        let instances = self
            .runtime
            .discovery()
            .list(DiscoveryQuery::NamespacedEndpoints {
                namespace: self.namespace.clone(),
            })
            .await?;

        let targets = instances
            .into_iter()
            .filter_map(|instance| match instance {
                DiscoveryInstance::Endpoint(ep) if ep.endpoint == self.rl_endpoint => Some(ep),
                _ => None,
            })
            .filter(|ep| {
                opts.components
                    .as_ref()
                    .map(|cs| cs.iter().any(|c| c == &ep.component))
                    .unwrap_or(true)
            })
            .map(|ep| WorkerTarget {
                namespace: ep.namespace,
                component: ep.component,
                endpoint: ep.endpoint,
                instance_id: ep.instance_id,
            })
            .collect();

        Ok(MembershipSnapshot::new(targets))
    }

    /// Fan out `method` + `kwargs` to all live workers (or those matched by
    /// `opts.components`). Returns a `FanoutReport` with per-worker results.
    /// Returns 503 if zero workers are discovered.
    /// Returns 409 if the worker set changes between pre- and post-fan-out
    /// snapshots (abort-on-membership-change).
    pub async fn engine_call(
        &self,
        method: &str,
        kwargs: serde_json::Value,
        opts: CallOptions,
    ) -> anyhow::Result<FanoutReport> {
        let snapshot = self.snapshot(&opts).await?;
        if snapshot.is_empty() {
            return Err(RlError::NoWorkers {
                namespace: self.namespace.clone(),
                rl_endpoint: self.rl_endpoint.clone(),
            }
            .into());
        }
        self.fanout_snapshot(&snapshot, method, kwargs, opts.timeout)
            .await
    }

    /// Call exactly one worker (strict-direct). Returns 409 if the instance
    /// has vanished from the live set before the call is dispatched.
    pub async fn engine_call_one(
        &self,
        method: &str,
        kwargs: serde_json::Value,
        instance_id: u64,
        opts: CallOptions,
    ) -> anyhow::Result<FanoutReport> {
        let snapshot = self.snapshot(&opts).await?;
        let target = snapshot
            .targets
            .iter()
            .find(|t| t.instance_id == instance_id)
            .cloned()
            .ok_or(RlError::UnknownInstance { instance_id })?;

        let result = call_worker_target(&self.runtime, &target, method, kwargs, opts.timeout).await;
        Ok(FanoutReport {
            epoch: snapshot.epoch,
            workers: vec![result],
        })
    }

    async fn fanout_snapshot(
        &self,
        snapshot: &MembershipSnapshot,
        method: &str,
        kwargs: serde_json::Value,
        timeout: Duration,
    ) -> anyhow::Result<FanoutReport> {
        // Group targets by (namespace, component, endpoint) — one PushRouter per group.
        let mut grouped: HashMap<(String, String, String), Vec<WorkerTarget>> = HashMap::new();
        for target in &snapshot.targets {
            grouped
                .entry((
                    target.namespace.clone(),
                    target.component.clone(),
                    target.endpoint.clone(),
                ))
                .or_default()
                .push(target.clone());
        }

        let mut calls: Vec<futures::future::BoxFuture<'static, WorkerResult>> = Vec::new();

        for ((namespace, component, endpoint_name), targets) in grouped {
            let endpoint = match self
                .runtime
                .namespace(&namespace)
                .and_then(|ns| ns.component(&component))
            {
                Ok(comp) => comp.endpoint(endpoint_name),
                Err(err) => {
                    for target in targets {
                        let err_str = format!("endpoint build failed: {err}");
                        calls.push(
                            futures::future::ready(WorkerResult::error(&target, err_str)).boxed(),
                        );
                    }
                    continue;
                }
            };

            let client = match endpoint.client().await {
                Ok(c) => c,
                Err(err) => {
                    for target in targets {
                        let err_str = format!("client create failed: {err}");
                        calls.push(
                            futures::future::ready(WorkerResult::error(&target, err_str)).boxed(),
                        );
                    }
                    continue;
                }
            };

            let target_ids: Vec<u64> = targets.iter().map(|t| t.instance_id).collect();
            wait_for_client_targets(&client, &target_ids, Duration::from_secs(5)).await;

            let router =
                match PushRouter::<serde_json::Value, Annotated<serde_json::Value>>::from_client(
                    client,
                    RouterMode::Direct,
                )
                .await
                {
                    Ok(r) => r,
                    Err(err) => {
                        for target in targets {
                            let err_str = format!("PushRouter build failed: {err}");
                            calls.push(
                                futures::future::ready(WorkerResult::error(&target, err_str))
                                    .boxed(),
                            );
                        }
                        continue;
                    }
                };

            for target in targets {
                calls.push(
                    call_worker(
                        router.clone(),
                        target,
                        method.to_string(),
                        kwargs.clone(),
                        timeout,
                    )
                    .boxed(),
                );
            }
        }

        let workers = futures::future::join_all(calls).await;

        // Abort-on-membership-change: check epoch after fan-out completes.
        let after_opts = CallOptions::new(timeout);
        let after = self.snapshot(&after_opts).await?;
        if after.epoch != snapshot.epoch {
            return Err(RlError::MembershipChanged {
                before_epoch: snapshot.epoch,
                after_epoch: after.epoch,
            }
            .into());
        }

        Ok(FanoutReport {
            epoch: snapshot.epoch,
            workers,
        })
    }

    pub fn default_timeout(&self) -> Duration {
        self.default_request_timeout
    }
}

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

async fn wait_for_client_targets(client: &Client, target_ids: &[u64], timeout: Duration) {
    let wait = async {
        loop {
            let ids = client.instance_ids();
            if target_ids.iter().all(|id| ids.contains(id)) {
                break;
            }
            tokio::time::sleep(Duration::from_millis(25)).await;
        }
    };
    let _ = tokio::time::timeout(timeout, wait).await;
}

async fn call_worker_target(
    runtime: &Arc<DistributedRuntime>,
    target: &WorkerTarget,
    method: &str,
    kwargs: serde_json::Value,
    timeout: Duration,
) -> WorkerResult {
    let endpoint = match runtime
        .namespace(&target.namespace)
        .and_then(|ns| ns.component(&target.component))
    {
        Ok(comp) => comp.endpoint(target.endpoint.clone()),
        Err(err) => return WorkerResult::error(target, format!("endpoint build failed: {err}")),
    };

    let client = match endpoint.client().await {
        Ok(c) => c,
        Err(err) => return WorkerResult::error(target, format!("client create failed: {err}")),
    };

    wait_for_client_targets(&client, &[target.instance_id], Duration::from_secs(5)).await;

    let router = match PushRouter::<serde_json::Value, Annotated<serde_json::Value>>::from_client(
        client,
        RouterMode::Direct,
    )
    .await
    {
        Ok(r) => r,
        Err(err) => return WorkerResult::error(target, format!("PushRouter build failed: {err}")),
    };

    call_worker(router, target.clone(), method.to_string(), kwargs, timeout).await
}

async fn call_worker(
    router: PushRouter<serde_json::Value, Annotated<serde_json::Value>>,
    target: WorkerTarget,
    method: String,
    kwargs: serde_json::Value,
    timeout: Duration,
) -> WorkerResult {
    let request_value = serde_json::json!({
        "method": method,
        "kwargs": kwargs,
    });

    let instance_id = target.instance_id;

    let dispatch = async {
        let req = SingleIn::new(request_value);
        let mut stream = router.direct(req, instance_id).await?;

        while let Some(chunk) = stream.next().await {
            if let Some(data) = chunk.data {
                return anyhow::Ok(data);
            }
            if let Some(err) = chunk.error {
                anyhow::bail!(err.to_string());
            }
        }

        anyhow::bail!("empty response stream from worker");
    };

    match tokio::time::timeout(timeout, dispatch).await {
        Ok(Ok(response)) => WorkerResult::ok(&target, response),
        Ok(Err(err)) => WorkerResult::error(&target, format!("dispatch failed: {err}")),
        Err(_) => WorkerResult::error(
            &target,
            format!("dispatch timed out after {}s", timeout.as_secs()),
        ),
    }
}

// ---------------------------------------------------------------------------
// HTTP facade
// ---------------------------------------------------------------------------

/// HTTP request body for `POST /v1/rl/engine`.
#[derive(Debug, serde::Deserialize)]
pub struct RlEngineRequest {
    pub method: String,
    #[serde(default)]
    pub kwargs: serde_json::Value,
    /// If present, call only this instance (strict-direct). Absent = fan-out.
    pub instance_id: Option<u64>,
    /// Per-call timeout override (seconds). Falls back to `RlClientConfig.default_request_timeout`.
    pub timeout_secs: Option<f64>,
    /// Restrict fan-out to these component names. `None` = all.
    pub components: Option<Vec<String>>,
}

impl RlEngineRequest {
    fn call_options(&self, default_timeout: Duration) -> CallOptions {
        let timeout = self
            .timeout_secs
            .map(Duration::from_secs_f64)
            .unwrap_or(default_timeout);
        let mut opts = CallOptions::new(timeout);
        if let Some(ref cs) = self.components {
            opts = opts.with_components(cs.clone());
        }
        opts
    }
}

/// Shared state for the RL HTTP facade.
#[derive(Clone)]
pub struct RlState {
    pub client: Arc<RlClient>,
}

impl RlState {
    pub fn new(client: RlClient) -> Self {
        Self {
            client: Arc::new(client),
        }
    }
}

/// Build the RL axum router.
///
/// ```text
/// POST /v1/rl/engine   — fan-out or direct call
/// GET  /v1/rl/engine   — describe: list registered methods per worker
/// ```
pub fn rl_router(state: RlState) -> Router {
    Router::new()
        .route("/v1/rl/engine", post(engine_handler).get(describe_handler))
        .with_state(state)
}

fn rl_error_response(err: anyhow::Error) -> (StatusCode, Json<serde_json::Value>) {
    let (status, error_type) = match err.downcast_ref::<RlError>() {
        Some(RlError::NoWorkers { .. }) => (StatusCode::SERVICE_UNAVAILABLE, "no_workers"),
        Some(RlError::MembershipChanged { .. }) => (StatusCode::CONFLICT, "membership_changed"),
        Some(RlError::UnknownInstance { .. }) => (StatusCode::CONFLICT, "unknown_instance"),
        None => (StatusCode::INTERNAL_SERVER_ERROR, "fanout_failed"),
    };

    (
        status,
        Json(serde_json::json!({
            "error": error_type,
            "message": err.to_string(),
        })),
    )
}

fn fanout_report_to_response(report: FanoutReport) -> Json<serde_json::Value> {
    Json(serde_json::json!({
        "epoch": report.epoch,
        "workers": report.workers,
    }))
}

/// `POST /v1/rl/engine` — fan-out or direct call.
async fn engine_handler(
    State(state): State<RlState>,
    Json(req): Json<RlEngineRequest>,
) -> impl IntoResponse {
    let opts = req.call_options(state.client.default_timeout());

    let result = if let Some(id) = req.instance_id {
        state
            .client
            .engine_call_one(&req.method, req.kwargs, id, opts)
            .await
    } else {
        state
            .client
            .engine_call(&req.method, req.kwargs, opts)
            .await
    };

    match result {
        Ok(report) => fanout_report_to_response(report).into_response(),
        Err(err) => {
            let (status, body) = rl_error_response(err);
            (status, body).into_response()
        }
    }
}

/// `GET /v1/rl/engine` — return live worker set with registered method lists.
///
/// Sends `{"method": "__describe__"}` to each worker's `rl_dispatch` and
/// aggregates `registered_methods` lists.
async fn describe_handler(State(state): State<RlState>) -> impl IntoResponse {
    let opts = CallOptions::new(state.client.default_timeout());
    let result = state
        .client
        .engine_call("__describe__", serde_json::json!({}), opts)
        .await;

    match result {
        Ok(report) => fanout_report_to_response(report).into_response(),
        Err(err) => {
            let (status, body) = rl_error_response(err);
            (status, body).into_response()
        }
    }
}
