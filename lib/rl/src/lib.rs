// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Dynamo RL admin control plane — handlers, state, fan-out for `/v1/rl/*`.
//!
//! See `plans/rl-crate.md` and `plans/weight-transfer-config.md`.
//!
//! **PR A status:** pure refactor — handlers + state moved verbatim out of
//! `lib/llm/src/http/service/openai.rs` so the admin code lives in its own
//! crate. Behavior unchanged. Future work (per the plan):
//!
//! - **PR B:** replace `worker_system_urls: Vec<String>` (HTTP system-port
//!   fan-out, env-driven) with discovery-backed fan-out via the dynamo
//!   request plane. Drop `reqwest::Client`. Drop `DYN_RL_WORKER_SYSTEM_URLS`.
//! - **PR C:** introduce `DYN_ENABLE_RL_ENDPOINTS` (frontend-only) to gate
//!   this router on a separate Axum listener (`DYN_RL_PORT` / `--rl-port`).
//!   `DYN_ENABLE_RL` keeps its meaning as the inference-plane RL extensions
//!   gate plus worker-side engine-route registration.

use std::sync::Arc;

use axum::{
    Json, Router,
    extract::State,
    http::{Method, StatusCode},
    response::IntoResponse,
    routing::post,
};

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

/// Environment variable for comma-separated worker system HTTP URLs.
/// Defaults to `http://localhost:8081` when not set.
const DYN_RL_WORKER_SYSTEM_URLS_ENV: &str = "DYN_RL_WORKER_SYSTEM_URLS";

/// Shared state for the RL admin router.
#[derive(Clone)]
struct RlState {
    /// Worker system HTTP base URLs (e.g. `http://localhost:8081`).
    /// Set via `DYN_RL_WORKER_SYSTEM_URLS` (comma-separated list).
    /// PR B (deferred) replaces this with discovery-backed enumeration.
    worker_system_urls: Vec<String>,
    /// Shared HTTP client for all fan-out calls to worker system ports.
    http_client: reqwest::Client,
}

impl RlState {
    fn from_env() -> anyhow::Result<Self> {
        let worker_system_urls = std::env::var(DYN_RL_WORKER_SYSTEM_URLS_ENV)
            .unwrap_or_else(|_| "http://localhost:8081".to_string())
            .split(',')
            .map(|s| s.trim().to_string())
            .filter(|s| !s.is_empty())
            .collect::<Vec<_>>();
        tracing::info!(
            worker_count = worker_system_urls.len(),
            ?worker_system_urls,
            "RL admin router configured"
        );
        let http_client = reqwest::Client::builder()
            .timeout(std::time::Duration::from_secs(600))
            .build()
            .map_err(|e| anyhow::anyhow!("Failed to build RL router HTTP client: {e}"))?;
        Ok(Self::new(worker_system_urls, http_client))
    }

    /// Test-friendly constructor — bypasses env reading so tests can pass in
    /// fake worker URLs and a stubbed `reqwest::Client`.
    fn new(worker_system_urls: Vec<String>, http_client: reqwest::Client) -> Self {
        Self {
            worker_system_urls,
            http_client,
        }
    }

    /// Call a single engine route on one worker. Returns the JSON body.
    async fn call_engine_route(
        &self,
        url: &str,
        route: &str,
        body: &serde_json::Value,
    ) -> serde_json::Value {
        let endpoint = format!("{url}/engine/{route}");
        match self.http_client.post(&endpoint).json(body).send().await {
            Ok(resp) => {
                let status = resp.status();
                match resp.json::<serde_json::Value>().await {
                    Ok(v) => v,
                    Err(e) => serde_json::json!({
                        "status": "error",
                        "message": format!("Failed to decode response from {endpoint}: {e}"),
                        "http_status": status.as_u16()
                    }),
                }
            }
            Err(e) => serde_json::json!({
                "status": "error",
                "message": format!("Request to {endpoint} failed: {e}")
            }),
        }
    }

    /// Fan out an engine route call to all configured workers concurrently.
    async fn fan_out(&self, route: &str, body: serde_json::Value) -> Vec<serde_json::Value> {
        let futures: Vec<_> = self
            .worker_system_urls
            .iter()
            .map(|url| self.call_engine_route(url, route, &body))
            .collect();
        futures::future::join_all(futures).await
    }

    /// Returns true only if all results have `status: "ok"`.
    fn all_ok(results: &[serde_json::Value]) -> bool {
        results
            .iter()
            .all(|r| r.get("status").and_then(|s| s.as_str()) == Some("ok"))
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
/// Worker system URLs are read from the `DYN_RL_WORKER_SYSTEM_URLS` environment
/// variable (comma-separated, defaults to `http://localhost:8081`). Phase B of
/// `rl-crate.md` will replace this with discovery-backed fan-out; until then
/// the static URL list is the source of truth.
///
/// **Surface:** four POST routes after Phase 3 (this PR). Read-side endpoints
/// (`state`, `health`, `ready`, `liveness`, `weight_version`) and the
/// dedicated LoRA routes (`load_lora_adapter`, `unload_lora_adapter`) are
/// dropped — replacements piggyback on the frontend's existing `/live` and
/// `/health`, and LoRA flows through `update_weights {target.kind="lora"}`.
/// See `weight-transfer-config.md` § "Constraints from existing surface".
///
/// Mounted on the dedicated `/v1/rl/*` listener when
/// `DYN_ENABLE_RL_ENDPOINTS=true`. prime-rl usage:
/// `admin_base_url = "http://dynamo-frontend:8002/v1/rl"`.
pub fn rl_router() -> anyhow::Result<(Vec<RlRouteDoc>, Router)> {
    let rl_state_arc = Arc::new(RlState::from_env()?);
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
