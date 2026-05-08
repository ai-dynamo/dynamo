// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Dynamo RL admin control plane — handlers, state, fan-out for `/v1/rl/*`.
//!
//! See `plans/rl-crate.md` and `plans/weight-transfer-config.md`.
//!
//! **PR B status:** request-plane fan-out via the dynamo discovery plane.
//! Workers register one endpoint `dyn://<ns>.<component>.rl` (see
//! `worker_factory.py::rl_endpoint.serve_endpoint(handler.rl_dispatch, …)`)
//! and the frontend dispatches by snapshotting live `rl` instances and calling
//! each via strict request-plane direct routing. The legacy `register_engine_route`
//! HTTP-on-system-port mechanism + `DYN_RL_WORKER_SYSTEM_URLS` static URL
//! list are gone.

use std::{
    collections::{HashMap, hash_map::DefaultHasher},
    hash::{Hash, Hasher},
    sync::Arc,
    time::Duration,
};

use axum::{
    Json, Router,
    extract::State,
    http::{Method, StatusCode},
    response::IntoResponse,
    routing::post,
};
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

pub const DEFAULT_RL_ENDPOINT: &str = "rl";

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
        }
    }
}

impl std::error::Error for RlError {}

#[derive(Debug, Clone)]
pub struct RlClientConfig {
    pub runtime: Arc<DistributedRuntime>,
    pub namespace: String,
    pub rl_endpoint: String,
    pub policy: FanoutPolicy,
}

#[derive(Debug, Clone)]
pub struct FanoutPolicy {
    pub min_workers: usize,
    pub membership_timeout: Duration,
    pub request_timeout: Duration,
    pub strict_direct: bool,
    pub abort_on_membership_change: bool,
    pub component_filter: Option<Vec<String>>,
}

impl FanoutPolicy {
    pub fn default_admin() -> Self {
        Self {
            min_workers: 1,
            membership_timeout: Duration::from_secs(5),
            request_timeout: Duration::from_secs(30),
            strict_direct: true,
            abort_on_membership_change: true,
            component_filter: None,
        }
    }

    pub fn with_component_filter(mut self, components: Vec<String>) -> Self {
        let components: Vec<String> = components
            .into_iter()
            .map(|c| c.trim().to_string())
            .filter(|c| !c.is_empty())
            .collect();
        self.component_filter = if components.is_empty() {
            None
        } else {
            Some(components)
        };
        self
    }
}

impl Default for FanoutPolicy {
    fn default() -> Self {
        Self::default_admin()
    }
}

#[derive(
    Debug, Clone, serde::Serialize, serde::Deserialize, PartialEq, Eq, Hash, PartialOrd, Ord,
)]
pub struct WorkerTarget {
    pub namespace: String,
    pub component: String,
    pub endpoint: String,
    pub instance_id: u64,
}

impl WorkerTarget {
    fn endpoint_key(&self) -> (String, String, String) {
        (
            self.namespace.clone(),
            self.component.clone(),
            self.endpoint.clone(),
        )
    }
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

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct RlRequest {
    pub op: String,
    #[serde(default)]
    pub body: serde_json::Value,
}

impl RlRequest {
    pub fn new(op: impl Into<String>, body: serde_json::Value) -> Self {
        Self {
            op: op.into(),
            body,
        }
    }

    pub fn describe(_req: DescribeRequest) -> Self {
        Self::new("describe", serde_json::json!({}))
    }

    pub fn pause(req: PauseRequest) -> Self {
        Self::new("pause", serde_json::to_value(req).unwrap_or_default())
    }

    pub fn resume(_req: ResumeRequest) -> Self {
        Self::new("resume", serde_json::json!({}))
    }

    pub fn init_transport(req: InitTransportRequest) -> Self {
        Self::new("init_transport", req.0)
    }

    pub fn update_weights(req: UpdateWeightsRequest) -> Self {
        Self::new("update_weights", req.into_body())
    }
}

#[derive(Debug, Clone, Default, serde::Serialize, serde::Deserialize)]
pub struct DescribeRequest {}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct PauseRequest {
    pub mode: String,
    pub clear_cache: bool,
}

impl Default for PauseRequest {
    fn default() -> Self {
        Self {
            mode: "keep".to_string(),
            clear_cache: false,
        }
    }
}

#[derive(Debug, Clone, Default, serde::Serialize, serde::Deserialize)]
pub struct ResumeRequest {}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
#[serde(transparent)]
pub struct InitTransportRequest(pub serde_json::Value);

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct UpdateWeightsRequest {
    pub version: String,
    pub target: serde_json::Value,
    pub transport: serde_json::Value,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub pause_mode: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub clear_cache: Option<bool>,
}

impl UpdateWeightsRequest {
    fn into_body(self) -> serde_json::Value {
        let mut body = serde_json::json!({
            "version": self.version,
            "target": self.target,
            "transport": self.transport,
        });
        if let Some(pause_mode) = self.pause_mode {
            body["pause_mode"] = serde_json::Value::String(pause_mode);
        }
        if let Some(clear_cache) = self.clear_cache {
            body["clear_cache"] = serde_json::Value::Bool(clear_cache);
        }
        body
    }
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct WorkerResult {
    pub target: WorkerTarget,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub response: Option<serde_json::Value>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
}

impl WorkerResult {
    fn ok(target: WorkerTarget, response: serde_json::Value) -> Self {
        Self {
            target,
            response: Some(response),
            error: None,
        }
    }

    fn error(target: WorkerTarget, error: impl Into<String>) -> Self {
        Self {
            target,
            response: None,
            error: Some(error.into()),
        }
    }

    pub fn is_ok(&self) -> bool {
        self.error.is_none()
            && self
                .response
                .as_ref()
                .and_then(|r| r.get("status"))
                .and_then(|s| s.as_str())
                == Some("ok")
    }

    pub fn payload(&self) -> serde_json::Value {
        match (&self.response, &self.error) {
            (Some(response), None) => response.clone(),
            (_, Some(error)) => serde_json::json!({
                "status": "error",
                "namespace": self.target.namespace,
                "component": self.target.component,
                "endpoint": self.target.endpoint,
                "instance_id": self.target.instance_id,
                "message": error,
            }),
            _ => serde_json::json!({
                "status": "error",
                "namespace": self.target.namespace,
                "component": self.target.component,
                "endpoint": self.target.endpoint,
                "instance_id": self.target.instance_id,
                "message": "missing worker response",
            }),
        }
    }
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct FanoutReport {
    pub snapshot: MembershipSnapshot,
    pub workers: Vec<WorkerResult>,
}

impl FanoutReport {
    pub fn all_ok(&self) -> bool {
        !self.workers.is_empty() && self.workers.iter().all(WorkerResult::is_ok)
    }

    pub fn worker_payloads(&self) -> Vec<serde_json::Value> {
        self.workers.iter().map(WorkerResult::payload).collect()
    }
}

#[derive(Clone)]
pub struct RlClient {
    runtime: Arc<DistributedRuntime>,
    namespace: String,
    rl_endpoint: String,
    policy: FanoutPolicy,
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
            policy: config.policy,
        })
    }

    pub async fn snapshot(&self) -> anyhow::Result<MembershipSnapshot> {
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
                DiscoveryInstance::Endpoint(instance) if instance.endpoint == self.rl_endpoint => {
                    Some(instance)
                }
                _ => None,
            })
            .filter(|instance| {
                self.policy
                    .component_filter
                    .as_ref()
                    .map(|components| components.iter().any(|c| c == &instance.component))
                    .unwrap_or(true)
            })
            .map(|instance| WorkerTarget {
                namespace: instance.namespace,
                component: instance.component,
                endpoint: instance.endpoint,
                instance_id: instance.instance_id,
            })
            .collect();

        Ok(MembershipSnapshot::new(targets))
    }

    pub async fn describe(&self) -> anyhow::Result<FanoutReport> {
        self.fanout(RlRequest::describe(DescribeRequest::default()))
            .await
    }

    pub async fn pause(&self, req: PauseRequest) -> anyhow::Result<FanoutReport> {
        self.fanout(RlRequest::pause(req)).await
    }

    pub async fn resume(&self, req: ResumeRequest) -> anyhow::Result<FanoutReport> {
        self.fanout(RlRequest::resume(req)).await
    }

    pub async fn init_transport(&self, req: InitTransportRequest) -> anyhow::Result<FanoutReport> {
        self.fanout(RlRequest::init_transport(req)).await
    }

    pub async fn update_weights(&self, req: UpdateWeightsRequest) -> anyhow::Result<FanoutReport> {
        self.fanout(RlRequest::update_weights(req)).await
    }

    pub async fn fanout(&self, request: RlRequest) -> anyhow::Result<FanoutReport> {
        let snapshot = self.snapshot().await?;
        self.fanout_snapshot(snapshot, request).await
    }

    pub async fn fanout_snapshot(
        &self,
        snapshot: MembershipSnapshot,
        request: RlRequest,
    ) -> anyhow::Result<FanoutReport> {
        if snapshot.targets.len() < self.policy.min_workers {
            return Err(RlError::NoWorkers {
                namespace: self.namespace.clone(),
                rl_endpoint: self.rl_endpoint.clone(),
            }
            .into());
        }

        let mut grouped: HashMap<(String, String, String), Vec<WorkerTarget>> = HashMap::new();
        for target in &snapshot.targets {
            grouped
                .entry(target.endpoint_key())
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
                Ok(component) => component.endpoint(endpoint_name),
                Err(err) => {
                    for target in targets {
                        calls.push(
                            futures::future::ready(WorkerResult::error(
                                target,
                                format!("endpoint build failed: {err}"),
                            ))
                            .boxed(),
                        );
                    }
                    continue;
                }
            };

            let client = match endpoint.client().await {
                Ok(client) => client,
                Err(err) => {
                    for target in targets {
                        calls.push(
                            futures::future::ready(WorkerResult::error(
                                target,
                                format!("client create failed: {err}"),
                            ))
                            .boxed(),
                        );
                    }
                    continue;
                }
            };

            let target_ids: Vec<u64> = targets.iter().map(|target| target.instance_id).collect();
            wait_for_client_targets(&client, &target_ids, self.policy.membership_timeout).await;

            let router =
                match PushRouter::<serde_json::Value, Annotated<serde_json::Value>>::from_client(
                    client,
                    RouterMode::Direct,
                )
                .await
                {
                    Ok(router) => router,
                    Err(err) => {
                        for target in targets {
                            calls.push(
                                futures::future::ready(WorkerResult::error(
                                    target,
                                    format!("PushRouter build failed: {err}"),
                                ))
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
                        request.clone(),
                        self.policy.request_timeout,
                        self.policy.strict_direct,
                    )
                    .boxed(),
                );
            }
        }

        let workers = futures::future::join_all(calls).await;

        if self.policy.abort_on_membership_change {
            let after = self.snapshot().await?;
            if after.epoch != snapshot.epoch {
                return Err(RlError::MembershipChanged {
                    before_epoch: snapshot.epoch,
                    after_epoch: after.epoch,
                }
                .into());
            }
        }

        Ok(FanoutReport { snapshot, workers })
    }
}

async fn wait_for_client_targets(client: &Client, target_ids: &[u64], timeout: Duration) {
    let wait = async {
        loop {
            let instance_ids = client.instance_ids();
            if target_ids.iter().all(|id| instance_ids.contains(id)) {
                break;
            }
            tokio::time::sleep(Duration::from_millis(25)).await;
        }
    };

    let _ = tokio::time::timeout(timeout, wait).await;
}

async fn call_worker(
    router: PushRouter<serde_json::Value, Annotated<serde_json::Value>>,
    target: WorkerTarget,
    request: RlRequest,
    timeout: Duration,
    strict_direct: bool,
) -> WorkerResult {
    let request_value = match serde_json::to_value(request) {
        Ok(value) => value,
        Err(err) => return WorkerResult::error(target, format!("request encode failed: {err}")),
    };

    let instance_id = target.instance_id;
    let dispatch = async {
        let req = SingleIn::new(request_value);
        let mut stream = if strict_direct {
            router.direct_strict(req, instance_id).await?
        } else {
            router.direct(req, instance_id).await?
        };

        while let Some(chunk) = stream.next().await {
            if let Some(data) = chunk.data {
                return anyhow::Ok(data);
            }
            if let Some(err) = chunk.error {
                anyhow::bail!(err.to_string());
            }
        }

        anyhow::bail!("empty response stream");
    };

    match tokio::time::timeout(timeout, dispatch).await {
        Ok(Ok(response)) => WorkerResult::ok(target, response),
        Ok(Err(err)) => WorkerResult::error(target, format!("dispatch failed: {err}")),
        Err(_) => WorkerResult::error(
            target,
            format!("dispatch timed out after {}s", timeout.as_secs()),
        ),
    }
}

/// Documentation tuple for an RL admin route. The dynamo-llm caller wraps
/// each tuple into its own `RouteDoc` for `/openapi.json` aggregation.
#[derive(Debug, Clone)]
pub struct RlRouteDoc {
    pub method: Method,
    pub path: String,
}

impl RlRouteDoc {
    fn new(method: Method, path: impl Into<String>) -> Self {
        Self {
            method,
            path: path.into(),
        }
    }
}

/// Shared state for the RL admin HTTP facade.
#[derive(Clone)]
struct RlState {
    client: RlClient,
}

impl RlState {
    fn new(client: RlClient) -> Self {
        Self { client }
    }

    async fn fan_out(&self, route: &str, body: serde_json::Value) -> anyhow::Result<FanoutReport> {
        self.client
            .fanout(RlRequest::new(route_to_op(route), body))
            .await
    }
}

#[derive(Clone)]
pub struct RlHttpDeps {
    pub client: RlClient,
}

/// Map a legacy engine-route name to the corresponding `rl_dispatch` op.
fn route_to_op(route: &str) -> &str {
    match route {
        "pause_generation" => "pause",
        "resume_generation" => "resume",
        "weight_transport_init" => "init_transport",
        "weight_transport_update" => "update_weights",
        // Anything else — pass through verbatim so `rl_dispatch` can return
        // a meaningful "unknown op" error instead of us silently rewriting.
        other => other,
    }
}

/// `POST /v1/rl/pause` — fan out `pause_generation` to all workers.
///
/// Query params (both optional):
/// - `mode`: `keep` | `wait` | `abort` (default `keep`)
/// - `clear_cache`: `true` | `false` (default `false`)
///
/// Three-mode pause matches what vLLM exposes (abort / wait / keep). The
/// default `mode=keep&clear_cache=false` preserves the original single-mode
/// pause behavior so existing callers keep working without changes.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Deserialize, serde::Serialize)]
#[serde(rename_all = "lowercase")]
enum PauseMode {
    Keep,
    Wait,
    Abort,
}

impl PauseMode {
    fn as_str(self) -> &'static str {
        match self {
            PauseMode::Keep => "keep",
            PauseMode::Wait => "wait",
            PauseMode::Abort => "abort",
        }
    }
}

impl Default for PauseMode {
    fn default() -> Self {
        PauseMode::Keep
    }
}

#[derive(Debug, serde::Deserialize)]
struct RlPauseQuery {
    /// Axum returns 400 automatically if this fails to deserialize as a
    /// `PauseMode` (i.e. on `mode=invalid`), so we don't need a runtime check.
    #[serde(default)]
    mode: Option<PauseMode>,
    #[serde(default)]
    clear_cache: Option<bool>,
}

fn rl_error_response(err: anyhow::Error) -> (StatusCode, Json<serde_json::Value>) {
    let (status, error_type) = match err.downcast_ref::<RlError>() {
        Some(RlError::NoWorkers { .. }) => (StatusCode::SERVICE_UNAVAILABLE, "no_workers"),
        Some(RlError::MembershipChanged { .. }) => (StatusCode::CONFLICT, "membership_changed"),
        None => (StatusCode::BAD_GATEWAY, "fanout_failed"),
    };

    (
        status,
        Json(serde_json::json!({
            "status": "error",
            "error_type": error_type,
            "message": err.to_string(),
        })),
    )
}

async fn rl_pause(
    State(state): State<Arc<RlState>>,
    axum::extract::Query(q): axum::extract::Query<RlPauseQuery>,
) -> impl IntoResponse {
    let mode = q.mode.unwrap_or_default();
    let clear_cache = q.clear_cache.unwrap_or(false);
    let report = match state
        .fan_out(
            "pause_generation",
            serde_json::json!({"mode": mode.as_str(), "clear_cache": clear_cache}),
        )
        .await
    {
        Ok(report) => report,
        Err(err) => return rl_error_response(err),
    };

    let workers = report.worker_payloads();
    if report.all_ok() {
        tracing::info!(
            worker_count = workers.len(),
            membership_epoch = report.snapshot.epoch,
            mode = %mode.as_str(),
            clear_cache,
            "RL pause: all workers paused"
        );
        (
            StatusCode::OK,
            Json(serde_json::json!({
                "status": "ok",
                "mode": mode.as_str(),
                "clear_cache": clear_cache,
                "membership_epoch": report.snapshot.epoch,
                "workers": workers,
            })),
        )
    } else {
        tracing::warn!(?workers, "RL pause: some workers failed");
        (
            StatusCode::BAD_GATEWAY,
            Json(serde_json::json!({
                "status": "error",
                "membership_epoch": report.snapshot.epoch,
                "workers": workers,
            })),
        )
    }
}

/// `POST /v1/rl/resume` — fan out `resume_generation` to all workers.
async fn rl_resume(State(state): State<Arc<RlState>>) -> impl IntoResponse {
    let report = match state
        .fan_out("resume_generation", serde_json::json!({}))
        .await
    {
        Ok(report) => report,
        Err(err) => return rl_error_response(err),
    };

    let workers = report.worker_payloads();
    if report.all_ok() {
        tracing::info!(
            worker_count = workers.len(),
            membership_epoch = report.snapshot.epoch,
            "RL resume: all workers resumed"
        );
        (
            StatusCode::OK,
            Json(serde_json::json!({
                "status": "ok",
                "membership_epoch": report.snapshot.epoch,
                "workers": workers,
            })),
        )
    } else {
        tracing::warn!(?workers, "RL resume: some workers failed");
        (
            StatusCode::BAD_GATEWAY,
            Json(serde_json::json!({
                "status": "error",
                "membership_epoch": report.snapshot.epoch,
                "workers": workers,
            })),
        )
    }
}

/// `POST /v1/rl/update_weights` — fan out `flush_cache → update_weights_from_path` to all workers.
///
/// **Not atomic.** If `update_weights_from_path` succeeds on workers `0..N-1`
/// and fails on worker `N`, the fleet is left in a mixed-version state: the
/// successful workers serve the new version while worker `N` still runs the
/// previous one. The response carries per-worker status so callers can
/// retry / drain manually; a true rollback layer is a follow-up.
///
/// Two body shapes are accepted:
///
/// **Legacy** (Phase 1 backward-compat):
/// ```json
/// {
///   "weight_dir": "/path/to/checkpoint" | null,   // null → NCCL mode no-op
///   "weight_version": "step_42",                   // optional; derived from
///                                                  //   weight_dir basename if missing
///   "reset_prefix_cache": true
/// }
/// ```
///
/// **WeightTransferConfig** (new, single shape across backends):
/// ```json
/// {
///   "version": "step_42",
///   "target":    {"kind": "base"} | {"kind": "lora", "name": "...", "op": "load|swap|unload"},
///   "transport": {
///     "backend": "filesystem" | "nccl",
///     "filesystem": {"path": "...", "require_marker": "STABLE"},
///     "nccl":       {"transport_id": "...", "weight_names": [...], "dtype": "bf16"}
///   }
/// }
/// ```
///
/// Returns `{ "status": "ok", "applied_weight_version": "step_42", "workers": [...] }` on success.
///
/// The pause/resume envelope is left to the caller; full-FT updates MUST
/// bracket this call with `/v1/rl/pause` and `/v1/rl/resume`.
///
/// **Phase 3 (PR C):** the legacy `{weight_dir, weight_version, reset_prefix_cache}`
/// body is gone. Every caller now provides `version`, `target`, and
/// `transport`. LoRA load/swap/unload also go through this same body via
/// `target.kind = "lora"` — see `weight-transfer-config.md` § 2.
#[derive(Debug, serde::Deserialize)]
struct RlUpdateWeightsBody {
    version: String,
    target: serde_json::Value,
    transport: serde_json::Value,
    #[serde(default)]
    pause_mode: Option<String>,
    #[serde(default)]
    clear_cache: Option<bool>,
}

async fn rl_update_weights(
    State(state): State<Arc<RlState>>,
    body: axum::extract::Json<RlUpdateWeightsBody>,
) -> impl IntoResponse {
    let RlUpdateWeightsBody {
        version,
        target,
        transport,
        pause_mode,
        clear_cache,
    } = body.0;
    rl_update_weights_inner(state, version, target, transport, pause_mode, clear_cache).await
}

/// WeightTransferConfig path — fans out to ``weight_transport_update``.
async fn rl_update_weights_inner(
    state: Arc<RlState>,
    version: String,
    target: serde_json::Value,
    transport: serde_json::Value,
    pause_mode: Option<String>,
    clear_cache: Option<bool>,
) -> (StatusCode, Json<serde_json::Value>) {
    let backend = transport
        .get("backend")
        .and_then(|v| v.as_str())
        .unwrap_or("");
    tracing::info!(
        version = %version,
        backend = %backend,
        ?target,
        "RL update_weights"
    );
    let mut body = serde_json::json!({
        "version": version,
        "target": target,
        "transport": transport,
    });
    if let Some(pm) = pause_mode {
        body["pause_mode"] = serde_json::Value::String(pm);
    }
    if let Some(cc) = clear_cache {
        body["clear_cache"] = serde_json::Value::Bool(cc);
    }
    let report = match state.fan_out("weight_transport_update", body).await {
        Ok(report) => report,
        Err(err) => return rl_error_response(err),
    };

    let workers = report.worker_payloads();
    if report.all_ok() {
        tracing::info!(
            worker_count = workers.len(),
            membership_epoch = report.snapshot.epoch,
            backend = %backend,
            version = %version,
            "RL update_weights: all workers updated"
        );
        (
            StatusCode::OK,
            Json(serde_json::json!({
                "status": "ok",
                "applied_weight_version": version,
                "backend": backend,
                "membership_epoch": report.snapshot.epoch,
                "workers": workers,
            })),
        )
    } else {
        tracing::warn!(?workers, backend = %backend, "RL update_weights: some workers failed");
        (
            StatusCode::BAD_GATEWAY,
            Json(serde_json::json!({
                "status": "error",
                "stage": "weight_transport_update",
                "backend": backend,
                "membership_epoch": report.snapshot.epoch,
                "workers": workers,
            })),
        )
    }
}

/// `POST /v1/rl/init_transport` — idempotent one-time setup for a weight
/// transport (filesystem / nccl). Replaces backend-specific bring-up
/// endpoints with a single discriminated body.
///
/// Body:
/// ```json
/// {
///   "transport_id": "rl-weights-step",
///   "backend": "filesystem" | "nccl",
///   "filesystem": { … } | "nccl": { … }
/// }
/// ```
///
/// `filesystem` is a no-op (transport state goes ``ready`` immediately).
/// `nccl` triggers the worker-side group bootstrap.
async fn rl_init_transport(
    State(state): State<Arc<RlState>>,
    body: axum::extract::Json<serde_json::Value>,
) -> impl IntoResponse {
    let body = body.0;
    let backend = body
        .get("backend")
        .and_then(|v| v.as_str())
        .unwrap_or("")
        .to_string();
    let transport_id = body
        .get("transport_id")
        .and_then(|v| v.as_str())
        .unwrap_or(&backend)
        .to_string();
    tracing::info!(%backend, %transport_id, "RL init_transport");

    let report = match state.fan_out("weight_transport_init", body).await {
        Ok(report) => report,
        Err(err) => return rl_error_response(err),
    };

    let workers = report.worker_payloads();
    if report.all_ok() {
        tracing::info!(
            worker_count = workers.len(),
            membership_epoch = report.snapshot.epoch,
            %backend,
            %transport_id,
            "RL init_transport: all workers ready"
        );
        (
            StatusCode::OK,
            Json(serde_json::json!({
                "status": "ok",
                "transport_id": transport_id,
                "backend": backend,
                "ready": true,
                "membership_epoch": report.snapshot.epoch,
                "workers": workers,
            })),
        )
    } else {
        tracing::warn!(?workers, %backend, "RL init_transport: some workers failed");
        (
            StatusCode::BAD_GATEWAY,
            Json(serde_json::json!({
                "status": "error",
                "transport_id": transport_id,
                "backend": backend,
                "membership_epoch": report.snapshot.epoch,
                "workers": workers,
            })),
        )
    }
}

/// Create an Axum [`Router`] for the RL admin endpoints at `/v1/rl/*`.
///
/// Fan-out goes through [`RlClient`], which snapshots the discovery plane,
/// groups live `<namespace>.<component>.rl` workers, and dispatches with
/// request-plane strict direct calls over NATS / TCP / HTTP.
///
/// **Surface:** four POST routes after Phase 3.
/// `pause`, `resume`, `init_transport`, `update_weights`. Read-side
/// endpoints (`state`, `health`, `ready`, `liveness`, `weight_version`)
/// and the dedicated LoRA routes (`load_lora_adapter`, `unload_lora_adapter`)
/// are dropped — replacements piggyback on the frontend's existing `/live`
/// and `/health`, and LoRA flows through `update_weights {target.kind="lora"}`.
/// See `weight-transfer-config.md` § "Constraints from existing surface".
///
/// Mounted on the dedicated `/v1/rl/*` listener when
/// `DYN_ENABLE_RL_ENDPOINTS=true`. prime-rl usage:
/// `admin_base_url = "http://dynamo-frontend:8002/v1/rl"`.
pub fn rl_router(deps: RlHttpDeps) -> anyhow::Result<(Vec<RlRouteDoc>, Router)> {
    let rl_state_arc = Arc::new(RlState::new(deps.client));
    let docs = vec![
        // Pause / resume bracket.
        RlRouteDoc::new(axum::http::Method::POST, "/v1/rl/pause"),
        RlRouteDoc::new(axum::http::Method::POST, "/v1/rl/resume"),
        // WeightTransferConfig API: init + discriminated update_weights body
        // covering both base-model reload and LoRA load/swap/unload.
        RlRouteDoc::new(axum::http::Method::POST, "/v1/rl/init_transport"),
        RlRouteDoc::new(axum::http::Method::POST, "/v1/rl/update_weights"),
    ];
    let router = Router::new()
        .route("/v1/rl/pause", post(rl_pause))
        .route("/v1/rl/resume", post(rl_resume))
        .route("/v1/rl/init_transport", post(rl_init_transport))
        .route("/v1/rl/update_weights", post(rl_update_weights))
        .with_state(rl_state_arc);
    Ok((docs, router))
}
