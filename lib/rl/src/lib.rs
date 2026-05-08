// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Dynamo RL admin control plane — handlers, state, fan-out for `/v1/rl/*`.
//!
//! See `plans/rl-crate.md` and `plans/weight-transfer-config.md`.
//!
//! **PR B status:** request-plane fan-out via the dynamo discovery plane.
//! Workers register one endpoint `dyn://<ns>.<component>.rl` (see
//! `worker_factory.py::rl_endpoint.serve_endpoint(handler.rl_dispatch, …)`)
//! and the frontend dispatches by listing live `rl` instances and calling
//! each via [`PushRouter::direct`]. The legacy `register_engine_route`
//! HTTP-on-system-port mechanism + `DYN_RL_WORKER_SYSTEM_URLS` static URL
//! list are gone.

use std::sync::Arc;

use axum::{
    Json, Router,
    extract::State,
    http::{Method, StatusCode},
    response::IntoResponse,
    routing::post,
};
use dynamo_runtime::{
    DistributedRuntime,
    pipeline::{
        SingleIn,
        network::egress::push_router::{PushRouter, RouterMode},
    },
    protocols::annotated::Annotated,
};
use futures::StreamExt;

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

/// Shared state for the RL admin router.
///
/// Holds a runtime handle, a target `<namespace>.<component>` pair, and the
/// name of the unified RL endpoint (always `"rl"`). Each fan-out call:
///
/// 1. Lists live instances of `<ns>.<comp>.rl` via discovery.
/// 2. Builds a [`PushRouter`] over the runtime's request plane (NATS / shared TCP).
/// 3. Calls [`PushRouter::direct`] per `instance_id` with a JSON
///    `{"op": <route_name>, "body": <payload>}` envelope.
/// 4. Drains the response stream and extracts the first `Annotated.data`.
#[derive(Clone)]
struct RlState {
    drt: Arc<DistributedRuntime>,
    namespace: String,
    component: String,
    /// The endpoint name workers serve their RL dispatcher on. Always `"rl"`.
    rl_endpoint: String,
}

impl RlState {
    fn from_env(drt: Arc<DistributedRuntime>) -> anyhow::Result<Self> {
        let namespace = std::env::var("DYN_NAMESPACE").unwrap_or_else(|_| "dynamo".into());
        // Workers default to component="backend" (vLLM, sglang). Allow
        // override for disagg / multi-component deployments.
        let component = std::env::var("DYN_RL_COMPONENT").unwrap_or_else(|_| "backend".into());
        let rl_endpoint = "rl".to_string();
        tracing::info!(
            ns = %namespace,
            comp = %component,
            rl_endpoint = %rl_endpoint,
            "RL admin router configured (request-plane discovery)"
        );
        Ok(Self {
            drt,
            namespace,
            component,
            rl_endpoint,
        })
    }

    /// Fan out an admin op to every live worker via the request plane.
    ///
    /// `route` is the legacy engine-route name (`pause_generation`,
    /// `resume_generation`, `weight_transport_init`, `weight_transport_update`)
    /// preserved from the call sites; we map it to the unified op name on
    /// the wire.
    ///
    /// Source of truth for "which workers are live" is the
    /// [`Client::instance_source`] watcher (etcd-backed), not a one-shot
    /// discovery `list()`. PushRouter's `direct()` checks the same client
    /// view internally — going through the client avoids the race where a
    /// freshly-built client hasn't populated yet.
    async fn fan_out(&self, route: &str, body: serde_json::Value) -> Vec<serde_json::Value> {
        let op = route_to_op(route);

        let endpoint = match self
            .drt
            .namespace(&self.namespace)
            .and_then(|ns| ns.component(&self.component))
        {
            Ok(comp) => comp.endpoint(&self.rl_endpoint),
            Err(err) => {
                tracing::warn!(%err, route, "RL fan_out: failed to build endpoint");
                return vec![serde_json::json!({
                    "status": "error",
                    "message": format!("endpoint build failed: {err}"),
                })];
            }
        };

        let client = match endpoint.client().await {
            Ok(c) => c,
            Err(err) => {
                tracing::warn!(%err, route, "RL fan_out: failed to create endpoint client");
                return vec![serde_json::json!({
                    "status": "error",
                    "message": format!("client create failed: {err}"),
                })];
            }
        };

        // Bound the watcher-population race: wait until the client sees
        // ≥1 instance (or a short deadline elapses, in which case we
        // surface the empty-fanout warning below). 5s is generous —
        // workers register synchronously on serve_endpoint() before they
        // start serving traffic, so by the time anything POSTs `/v1/rl/*`
        // they should already be in etcd.
        let _ = tokio::time::timeout(
            std::time::Duration::from_secs(5),
            client.wait_for_instances(),
        )
        .await;

        let instance_ids: Vec<u64> = client.instance_ids();
        if instance_ids.is_empty() {
            tracing::warn!(
                ns = %self.namespace,
                comp = %self.component,
                route,
                "RL fan_out: no live workers under {}.{}.rl; \
                 check DYN_NAMESPACE / DYN_RL_COMPONENT vs worker --component",
                self.namespace,
                self.component,
            );
            return Vec::new();
        }

        let router =
            match PushRouter::<serde_json::Value, Annotated<serde_json::Value>>::from_client(
                client,
                RouterMode::Direct,
            )
            .await
            {
                Ok(r) => r,
                Err(err) => {
                    tracing::warn!(%err, route, "RL fan_out: failed to build PushRouter");
                    return vec![serde_json::json!({
                        "status": "error",
                        "message": format!("PushRouter build failed: {err}"),
                    })];
                }
            };

        let envelope = serde_json::json!({"op": op, "body": body});

        let futures: Vec<_> = instance_ids
            .iter()
            .copied()
            .map(|id| {
                let router = router.clone();
                let envelope = envelope.clone();
                async move {
                    let req = SingleIn::new(envelope.clone());
                    match router.direct(req, id).await {
                        Ok(mut stream) => {
                            // Drain the first non-empty data chunk from the
                            // worker's async-generator response.
                            while let Some(chunk) = stream.next().await {
                                if let Some(data) = chunk.data {
                                    return data;
                                }
                                if let Some(err) = chunk.error {
                                    return serde_json::json!({
                                        "status": "error",
                                        "instance_id": id,
                                        "message": err.to_string(),
                                    });
                                }
                            }
                            serde_json::json!({
                                "status": "error",
                                "instance_id": id,
                                "message": "empty response stream",
                            })
                        }
                        Err(err) => serde_json::json!({
                            "status": "error",
                            "instance_id": id,
                            "message": format!("dispatch failed: {err}"),
                        }),
                    }
                }
            })
            .collect();
        futures::future::join_all(futures).await
    }

    /// Returns true only if every result is `status: "ok"` AND there is at
    /// least one. Empty fan-out (no workers found) is `503`, not silent OK.
    fn all_ok(results: &[serde_json::Value]) -> bool {
        !results.is_empty()
            && results
                .iter()
                .all(|r| r.get("status").and_then(|s| s.as_str()) == Some("ok"))
    }
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

async fn rl_pause(
    State(state): State<Arc<RlState>>,
    axum::extract::Query(q): axum::extract::Query<RlPauseQuery>,
) -> impl IntoResponse {
    let mode = q.mode.unwrap_or_default();
    let clear_cache = q.clear_cache.unwrap_or(false);
    let results = state
        .fan_out(
            "pause_generation",
            serde_json::json!({"mode": mode.as_str(), "clear_cache": clear_cache}),
        )
        .await;
    if RlState::all_ok(&results) {
        tracing::info!(
            worker_count = results.len(),
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
                "workers": results,
            })),
        )
    } else {
        tracing::warn!(?results, "RL pause: some workers failed");
        (
            StatusCode::BAD_GATEWAY,
            Json(serde_json::json!({"status": "error", "workers": results})),
        )
    }
}

/// `POST /v1/rl/resume` — fan out `resume_generation` to all workers.
async fn rl_resume(State(state): State<Arc<RlState>>) -> impl IntoResponse {
    let results = state
        .fan_out("resume_generation", serde_json::json!({}))
        .await;
    if RlState::all_ok(&results) {
        tracing::info!(
            worker_count = results.len(),
            "RL resume: all workers resumed"
        );
        (
            StatusCode::OK,
            Json(serde_json::json!({"status": "ok", "workers": results})),
        )
    } else {
        tracing::warn!(?results, "RL resume: some workers failed");
        (
            StatusCode::BAD_GATEWAY,
            Json(serde_json::json!({"status": "error", "workers": results})),
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
    let results = state.fan_out("weight_transport_update", body).await;
    if RlState::all_ok(&results) {
        tracing::info!(
            worker_count = results.len(),
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
                "workers": results,
            })),
        )
    } else {
        tracing::warn!(?results, backend = %backend, "RL update_weights: some workers failed");
        (
            StatusCode::BAD_GATEWAY,
            Json(serde_json::json!({
                "status": "error",
                "stage": "weight_transport_update",
                "backend": backend,
                "workers": results,
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

    let results = state.fan_out("weight_transport_init", body).await;
    if RlState::all_ok(&results) {
        tracing::info!(
            worker_count = results.len(),
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
                "workers": results,
            })),
        )
    } else {
        tracing::warn!(?results, %backend, "RL init_transport: some workers failed");
        (
            StatusCode::BAD_GATEWAY,
            Json(serde_json::json!({
                "status": "error",
                "transport_id": transport_id,
                "backend": backend,
                "workers": results,
            })),
        )
    }
}

/// Create an Axum [`Router`] for the RL admin endpoints at `/v1/rl/*`.
///
/// **PR B:** fan-out goes through the dynamo discovery plane + request
/// plane. Workers register `<DYN_NAMESPACE>.<DYN_RL_COMPONENT>.rl` (default
/// `dynamo.backend.rl`) on the request plane via
/// `runtime.endpoint(...).serve_endpoint(handler.rl_dispatch, ...)`. The
/// frontend lists live instances via [`DistributedRuntime::discovery`]
/// + [`DiscoveryQuery::NamespacedEndpoints`] and dispatches each call via
/// [`PushRouter::direct`] over NATS / shared TCP.
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
/// `admin_base_url = "http://dynamo-frontend:8000/v1/rl"`.
pub fn rl_router(drt: Arc<DistributedRuntime>) -> anyhow::Result<(Vec<RlRouteDoc>, Router)> {
    let rl_state_arc = Arc::new(RlState::from_env(drt)?);
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
