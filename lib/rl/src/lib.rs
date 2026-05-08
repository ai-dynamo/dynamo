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
    routing::{get, post},
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
    worker_system_urls: Vec<String>,
    /// Shared HTTP client for all fan-out calls to worker system ports.
    http_client: reqwest::Client,
    /// Per-worker probe timeout for `/v1/rl/liveness` and `/v1/rl/ready`.
    /// Read once from `DYN_RL_LIVENESS_TIMEOUT_MS` at construction.
    probe_timeout: std::time::Duration,
}

impl RlState {
    fn from_env() -> anyhow::Result<Self> {
        let worker_system_urls = std::env::var(DYN_RL_WORKER_SYSTEM_URLS_ENV)
            .unwrap_or_else(|_| "http://localhost:8081".to_string())
            .split(',')
            .map(|s| s.trim().to_string())
            .filter(|s| !s.is_empty())
            .collect::<Vec<_>>();
        let probe_timeout_ms = std::env::var("DYN_RL_LIVENESS_TIMEOUT_MS")
            .ok()
            .and_then(|v| v.parse::<u64>().ok())
            .unwrap_or(5000);
        tracing::info!(
            worker_count = worker_system_urls.len(),
            ?worker_system_urls,
            probe_timeout_ms,
            "RL admin router configured"
        );
        let http_client = reqwest::Client::builder()
            .timeout(std::time::Duration::from_secs(600))
            .build()
            .map_err(|e| anyhow::anyhow!("Failed to build RL router HTTP client: {e}"))?;
        Ok(Self::new(
            worker_system_urls,
            http_client,
            std::time::Duration::from_millis(probe_timeout_ms),
        ))
    }

    /// Test-friendly constructor — bypasses env reading so tests can pass in
    /// fake worker URLs and a stubbed `reqwest::Client`.
    fn new(
        worker_system_urls: Vec<String>,
        http_client: reqwest::Client,
        probe_timeout: std::time::Duration,
    ) -> Self {
        Self {
            worker_system_urls,
            http_client,
            probe_timeout,
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

/// `GET /v1/rl/ready` — composite readiness check: worker health via system port.
///
/// Bounded with a per-worker probe timeout (default 5s, override via
/// `DYN_RL_LIVENESS_TIMEOUT_MS`) so a wedged worker fails fast as 503 instead
/// of hanging on the shared 600s `http_client` timeout.
async fn rl_ready(State(state): State<Arc<RlState>>) -> impl IntoResponse {
    let timeout = state.probe_timeout;
    let futures: Vec<_> = state
        .worker_system_urls
        .iter()
        .map(|url| {
            let client = state.http_client.clone();
            let health_url = format!("{url}/health");
            async move {
                match tokio::time::timeout(timeout, client.get(&health_url).send()).await {
                    Ok(Ok(resp)) => resp.status().is_success(),
                    Ok(Err(_)) | Err(_) => false,
                }
            }
        })
        .collect();
    let results = futures::future::join_all(futures).await;
    let all_ready = !results.is_empty() && results.iter().all(|ok| *ok);
    if all_ready {
        (StatusCode::OK, Json(serde_json::json!({"status": "ready"})))
    } else {
        (
            StatusCode::SERVICE_UNAVAILABLE,
            Json(serde_json::json!({
                "status": "not_ready",
                "workers_ready": results.iter().filter(|ok| **ok).count(),
                "workers_total": results.len()
            })),
        )
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
#[derive(Debug, serde::Deserialize)]
#[serde(untagged)]
enum RlUpdateWeightsBody {
    /// New shape — required field is `transport`. Serde tries this variant
    /// first; falls back to legacy if it fails.
    NewShape {
        version: String,
        target: serde_json::Value,
        transport: serde_json::Value,
        #[serde(default)]
        pause_mode: Option<String>,
        #[serde(default)]
        clear_cache: Option<bool>,
    },
    /// Legacy single-arg body kept live during Phase 1 / 2.
    Legacy {
        weight_dir: Option<String>,
        #[serde(default)]
        weight_version: Option<String>,
        #[serde(default = "default_reset_prefix_cache")]
        reset_prefix_cache: bool,
    },
}

fn default_reset_prefix_cache() -> bool {
    true
}

async fn rl_update_weights(
    State(state): State<Arc<RlState>>,
    body: axum::extract::Json<RlUpdateWeightsBody>,
) -> impl IntoResponse {
    // Dispatch on body shape. New shape goes through the WeightTransferConfig
    // worker route; legacy keeps the existing flush_cache → update_weights_from_path
    // sequence so unmigrated callers continue to work.
    match body.0 {
        RlUpdateWeightsBody::NewShape {
            version,
            target,
            transport,
            pause_mode,
            clear_cache,
        } => {
            return rl_update_weights_new_shape(
                state,
                version,
                target,
                transport,
                pause_mode,
                clear_cache,
            )
            .await;
        }
        RlUpdateWeightsBody::Legacy {
            weight_dir,
            weight_version,
            reset_prefix_cache,
        } => {
            return rl_update_weights_legacy(state, weight_dir, weight_version, reset_prefix_cache)
                .await;
        }
    }
}

/// New WeightTransferConfig path — fans out to ``weight_transport_update``.
async fn rl_update_weights_new_shape(
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
        "RL update_weights (new shape)"
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
            "RL update_weights (new shape): all workers updated"
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
        tracing::warn!(?results, backend = %backend, "RL update_weights (new shape): some workers failed");
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

/// Legacy single-arg body — Phase 1 backward-compat.
async fn rl_update_weights_legacy(
    state: Arc<RlState>,
    weight_dir: Option<String>,
    weight_version: Option<String>,
    reset_prefix_cache: bool,
) -> (StatusCode, Json<serde_json::Value>) {
    // Treat empty string the same as missing/null (NCCL no-op). Otherwise
    // an empty string would reach the engine as `path=""` and fail
    // confusingly downstream.
    let weight_dir = weight_dir
        .as_ref()
        .map(|s| s.trim())
        .filter(|s| !s.is_empty())
        .map(str::to_string);
    let Some(weight_dir) = weight_dir else {
        tracing::info!("RL update_weights: weight_dir=null (NCCL mode, no-op on Dynamo side)");
        return (
            StatusCode::OK,
            Json(serde_json::json!({
                "status": "ok",
                "message": "NCCL mode, no-op on Dynamo side"
            })),
        );
    };

    let version = weight_version.clone().unwrap_or_else(|| {
        std::path::Path::new(&weight_dir)
            .file_name()
            .and_then(|n| n.to_str())
            .unwrap_or("unknown")
            .to_string()
    });
    tracing::info!(
        weight_dir = %weight_dir,
        version = %version,
        reset_prefix_cache,
        "RL update_weights"
    );

    // Step 1 (optional): flush_cache across all workers.
    if reset_prefix_cache {
        let flush_results = state.fan_out("flush_cache", serde_json::json!({})).await;
        if !RlState::all_ok(&flush_results) {
            tracing::warn!(?flush_results, "RL update_weights: flush_cache failed");
            return (
                StatusCode::BAD_GATEWAY,
                Json(serde_json::json!({
                    "status": "error",
                    "stage": "flush_cache",
                    "workers": flush_results
                })),
            );
        }
    }

    // Step 2: update_weights_from_path across all workers.
    let load_body = serde_json::json!({"path": &weight_dir, "version": version});
    let load_results = state.fan_out("update_weights_from_path", load_body).await;
    if RlState::all_ok(&load_results) {
        tracing::info!(
            worker_count = load_results.len(),
            weight_dir = %weight_dir,
            "RL update_weights: all workers updated"
        );
        (
            StatusCode::OK,
            Json(serde_json::json!({
                "status": "ok",
                "applied_weight_version": version,
                "workers": load_results,
            })),
        )
    } else {
        tracing::warn!(
            ?load_results,
            "RL update_weights: update_weights_from_path failed"
        );
        (
            StatusCode::BAD_GATEWAY,
            Json(serde_json::json!({
                "status": "error",
                "stage": "update_weights_from_path",
                "workers": load_results
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

/// `POST /v1/rl/load_lora_adapter` — hot-load/swap a LoRA adapter from a filesystem path.
///
/// Expected body: `{"lora_name": "r16-a32.0", "lora_path": "/path/to/adapter_dir"}`
///
/// The adapter directory must contain PEFT-style `adapter_model.safetensors` and
/// `adapter_config.json`. This is the RL-specific LoRA path used by Prime-RL every
/// training step (separate from Dynamo's URI-based `load_lora` gRPC endpoint which
/// downloads adapters from S3/file URIs and publishes a new ModelDeploymentCard).
///
/// Hot-swap semantics: calling with a `lora_name` that is already loaded removes
/// the previous adapter and loads the new one under the same deterministic int ID,
/// then resets the prefix cache so stale KV entries don't poison new rollouts.
///
/// Pair with `/v1/rl/pause` and `/v1/rl/resume` for a full drain-swap-resume cycle.
async fn rl_load_lora_adapter(
    State(state): State<Arc<RlState>>,
    body: axum::extract::Json<serde_json::Value>,
) -> impl IntoResponse {
    let lora_name = body.get("lora_name").and_then(|v| v.as_str());
    let lora_path = body.get("lora_path").and_then(|v| v.as_str());

    let (lora_name, lora_path) = match (lora_name, lora_path) {
        (Some(n), Some(p)) if !n.is_empty() && !p.is_empty() => (n.to_string(), p.to_string()),
        _ => {
            return (
                StatusCode::BAD_REQUEST,
                Json(serde_json::json!({
                    "status": "error",
                    "message": "Expected body: {\"lora_name\": str, \"lora_path\": str} (both required, non-empty)"
                })),
            );
        }
    };

    tracing::info!(%lora_name, %lora_path, "RL load_lora_adapter");
    let results = state
        .fan_out(
            "load_lora_adapter",
            serde_json::json!({"lora_name": &lora_name, "lora_path": &lora_path}),
        )
        .await;

    if RlState::all_ok(&results) {
        tracing::info!(
            worker_count = results.len(),
            %lora_name,
            %lora_path,
            "RL load_lora_adapter: all workers loaded"
        );
        (
            StatusCode::OK,
            Json(serde_json::json!({"status": "ok", "workers": results})),
        )
    } else {
        tracing::warn!(?results, %lora_name, "RL load_lora_adapter: some workers failed");
        (
            StatusCode::BAD_GATEWAY,
            Json(serde_json::json!({"status": "error", "workers": results})),
        )
    }
}

/// `POST /v1/rl/unload_lora_adapter` — remove a previously loaded LoRA adapter by name.
///
/// Expected body: `{"lora_name": "r16-a32.0"}`
///
/// Idempotent: unloading an already-absent LoRA returns `status: ok` so callers
/// can retry safely without special-casing not-found.
async fn rl_unload_lora_adapter(
    State(state): State<Arc<RlState>>,
    body: axum::extract::Json<serde_json::Value>,
) -> impl IntoResponse {
    let lora_name = body
        .get("lora_name")
        .and_then(|v| v.as_str())
        .map(str::to_string);

    let lora_name = match lora_name {
        Some(n) if !n.is_empty() => n,
        _ => {
            return (
                StatusCode::BAD_REQUEST,
                Json(serde_json::json!({
                    "status": "error",
                    "message": "Expected body: {\"lora_name\": str} (required, non-empty)"
                })),
            );
        }
    };

    tracing::info!(%lora_name, "RL unload_lora_adapter");
    let results = state
        .fan_out(
            "unload_lora_adapter",
            serde_json::json!({"lora_name": &lora_name}),
        )
        .await;

    if RlState::all_ok(&results) {
        tracing::info!(
            worker_count = results.len(),
            %lora_name,
            "RL unload_lora_adapter: all workers unloaded"
        );
        (
            StatusCode::OK,
            Json(serde_json::json!({"status": "ok", "workers": results})),
        )
    } else {
        tracing::warn!(?results, %lora_name, "RL unload_lora_adapter: some workers failed");
        (
            StatusCode::BAD_GATEWAY,
            Json(serde_json::json!({"status": "error", "workers": results})),
        )
    }
}

/// `GET /v1/rl/weight_version` — query weight version from all workers.
async fn rl_weight_version(State(state): State<Arc<RlState>>) -> impl IntoResponse {
    let results = state
        .fan_out("get_weight_version", serde_json::json!({}))
        .await;

    // Collect distinct versions and check for consistency
    let versions: Vec<_> = results
        .iter()
        .filter_map(|r| {
            r.get("version")
                .and_then(|v| v.as_str())
                .map(str::to_string)
        })
        .collect();

    let unique: std::collections::HashSet<&str> = versions.iter().map(String::as_str).collect();
    if unique.len() == 1 {
        (
            StatusCode::OK,
            Json(serde_json::json!({
                "status": "ok",
                "version": unique.into_iter().next().unwrap_or(""),
                "workers": results
            })),
        )
    } else {
        (
            StatusCode::OK,
            Json(serde_json::json!({
                "status": "inconsistent",
                "versions": unique.into_iter().collect::<Vec<_>>(),
                "workers": results
            })),
        )
    }
}

async fn rl_health() -> impl IntoResponse {
    (StatusCode::OK, Json(serde_json::json!({"status": "ok"})))
}

/// `GET /v1/rl/liveness` — engine event-loop probe via the `liveness_probe`
/// engine route. The legacy `/v1/rl/health` returns OK as long as the
/// frontend process is up; this endpoint round-trips through the engine so
/// a hung event loop or wedged worker surfaces as 503.
///
/// Each per-worker call carries a 5s timeout (override via
/// `DYN_RL_LIVENESS_TIMEOUT_MS`). Returns 200 only when every worker
/// reports `alive: true` within the deadline; 503 otherwise.
async fn rl_liveness(State(state): State<Arc<RlState>>) -> impl IntoResponse {
    if state.worker_system_urls.is_empty() {
        return (
            StatusCode::SERVICE_UNAVAILABLE,
            Json(serde_json::json!({
                "status": "error",
                "alive": false,
                "message": "no workers registered"
            })),
        );
    }
    let timeout = state.probe_timeout;

    let futures: Vec<_> = state
        .worker_system_urls
        .iter()
        .map(|url| {
            let client = state.http_client.clone();
            let endpoint = format!("{url}/engine/liveness_probe");
            async move {
                tokio::time::timeout(
                    timeout,
                    async {
                        match client.post(&endpoint).json(&serde_json::json!({})).send().await {
                            Ok(resp) => resp
                                .json::<serde_json::Value>()
                                .await
                                .unwrap_or_else(|e| serde_json::json!({
                                    "status": "error",
                                    "alive": false,
                                    "message": format!("decode failed: {e}")
                                })),
                            Err(e) => serde_json::json!({
                                "status": "error",
                                "alive": false,
                                "message": format!("request failed: {e}")
                            }),
                        }
                    },
                )
                .await
                .unwrap_or_else(|_| serde_json::json!({
                    "status": "error",
                    "alive": false,
                    "message": format!("liveness_probe timed out after {}ms", timeout.as_millis())
                }))
            }
        })
        .collect();
    let results = futures::future::join_all(futures).await;
    let all_alive = results
        .iter()
        .all(|r| r.get("alive").and_then(|v| v.as_bool()) == Some(true));
    if all_alive {
        (
            StatusCode::OK,
            Json(serde_json::json!({
                "status": "ok",
                "alive": true,
                "workers": results,
            })),
        )
    } else {
        (
            StatusCode::SERVICE_UNAVAILABLE,
            Json(serde_json::json!({
                "status": "error",
                "alive": false,
                "workers": results,
            })),
        )
    }
}

/// `GET /v1/rl/state` — composite RL fleet state snapshot.
///
/// Replaces three v1 endpoints (`/v1/rl/health` + `/v1/rl/ready` +
/// `/v1/rl/weight_version`) with a single composite, scoped to RL-specific
/// readiness (engine alive, pause state, applied weight version, loaded
/// LoRAs).
///
/// Aggregates per-worker `get_state` engine-route responses into:
///
/// ```json
/// {
///   "ready": bool,
///   "ingress_alive": true,
///   "engine_alive": bool,            // every worker's engine.check_health() ok
///   "pause_state": "running"|"paused"|"mixed",
///   "applied_weight_version": str,   // when consistent across workers; null if mixed
///   "loras": [{name, loaded_on: [worker_idx]}],
///   "workers": [<per-worker get_state payloads>]
/// }
/// ```
///
/// `ingress_alive` is unconditionally `true` because reaching this handler
/// means the frontend HTTP listener is up. `ready = ingress_alive AND
/// engine_alive AND len(workers) > 0`.
async fn rl_state(State(state): State<Arc<RlState>>) -> impl IntoResponse {
    if state.worker_system_urls.is_empty() {
        return (
            StatusCode::SERVICE_UNAVAILABLE,
            Json(serde_json::json!({
                "ready": false,
                "ingress_alive": true,
                "engine_alive": false,
                "pause_state": "running",
                "applied_weight_version": null,
                "loras": [],
                "workers": [],
                "status": "error",
                "message": "no workers registered"
            })),
        );
    }
    let results = state.fan_out("get_state", serde_json::json!({})).await;

    let engine_alive = results
        .iter()
        .all(|r| r.get("engine_alive").and_then(|v| v.as_bool()) == Some(true));

    // Aggregate pause_state: if all workers agree, surface that; else "mixed".
    let pause_states: std::collections::HashSet<&str> = results
        .iter()
        .filter_map(|r| r.get("pause_state").and_then(|v| v.as_str()))
        .collect();
    let pause_state = if pause_states.len() == 1 {
        pause_states
            .into_iter()
            .next()
            .unwrap_or("running")
            .to_string()
    } else if pause_states.is_empty() {
        "running".to_string()
    } else {
        "mixed".to_string()
    };

    // applied_weight_version is reported only when consistent.
    let weight_versions: std::collections::HashSet<&str> = results
        .iter()
        .filter_map(|r| r.get("applied_weight_version").and_then(|v| v.as_str()))
        .collect();
    let applied_weight_version: Option<String> = if weight_versions.len() == 1 {
        weight_versions.into_iter().next().map(|s| s.to_string())
    } else {
        None
    };

    // LoRA name → list of worker indices that have it loaded.
    let mut lora_loaded_on: std::collections::BTreeMap<String, Vec<usize>> =
        std::collections::BTreeMap::new();
    for (idx, worker) in results.iter().enumerate() {
        if let Some(loras) = worker.get("loras").and_then(|v| v.as_array()) {
            for lora in loras {
                if let Some(name) = lora.get("name").and_then(|v| v.as_str()) {
                    lora_loaded_on
                        .entry(name.to_string())
                        .or_default()
                        .push(idx);
                }
            }
        }
    }
    let loras: Vec<serde_json::Value> = lora_loaded_on
        .into_iter()
        .map(|(name, loaded_on)| serde_json::json!({"name": name, "loaded_on": loaded_on}))
        .collect();

    let ready = engine_alive && !results.is_empty();
    let body = serde_json::json!({
        "ready": ready,
        "ingress_alive": true,
        "engine_alive": engine_alive,
        "pause_state": pause_state,
        "applied_weight_version": applied_weight_version,
        "loras": loras,
        "workers": results,
    });
    let status = if ready {
        StatusCode::OK
    } else {
        StatusCode::SERVICE_UNAVAILABLE
    };
    (status, Json(body))
}

/// Create an Axum [`Router`] for the RL admin endpoints at `/v1/rl/*`.
///
/// Worker system URLs are read from the `DYN_RL_WORKER_SYSTEM_URLS` environment
/// variable (comma-separated, defaults to `http://localhost:8081`).
///
/// Exposed only when `DYN_ENABLE_RL=true` or `HttpServiceConfig.enable_rl` is set.
///
/// Prime-RL usage: set `admin_base_url = ["http://dynamo-frontend:8000/v1/rl"]`
/// in the orchestrator config. Prime-RL strips the trailing `/v1` suffix only
/// if present, so `/v1/rl` is preserved and all routes resolve correctly.
pub fn rl_router() -> anyhow::Result<(Vec<RlRouteDoc>, Router)> {
    let rl_state_arc = Arc::new(RlState::from_env()?);
    let docs = vec![
        // Phase 1: composite endpoints.
        RlRouteDoc::new(axum::http::Method::GET, "/v1/rl/state"),
        RlRouteDoc::new(axum::http::Method::GET, "/v1/rl/liveness"),
        // Pause / resume / update_weights bracket.
        RlRouteDoc::new(axum::http::Method::POST, "/v1/rl/pause"),
        RlRouteDoc::new(axum::http::Method::POST, "/v1/rl/resume"),
        RlRouteDoc::new(axum::http::Method::POST, "/v1/rl/update_weights"),
        // WeightTransferConfig API (Phase 1+4) — idempotent transport setup.
        RlRouteDoc::new(axum::http::Method::POST, "/v1/rl/init_transport"),
        // LoRA hot-swap.
        RlRouteDoc::new(axum::http::Method::POST, "/v1/rl/load_lora_adapter"),
        // Legacy (deprecated; subsumed by /v1/rl/state — Phase 5 will drop):
        RlRouteDoc::new(axum::http::Method::GET, "/v1/rl/health"),
        RlRouteDoc::new(axum::http::Method::GET, "/v1/rl/ready"),
        RlRouteDoc::new(axum::http::Method::GET, "/v1/rl/weight_version"),
        RlRouteDoc::new(axum::http::Method::POST, "/v1/rl/unload_lora_adapter"),
    ];
    let router = Router::new()
        // Phase 1: composite read-only endpoints.
        .route("/v1/rl/state", get(rl_state))
        .route("/v1/rl/liveness", get(rl_liveness))
        // Pause / resume / update_weights bracket.
        .route("/v1/rl/pause", post(rl_pause))
        .route("/v1/rl/resume", post(rl_resume))
        .route("/v1/rl/update_weights", post(rl_update_weights))
        // WeightTransferConfig API (Phase 1+4) — idempotent transport setup.
        .route("/v1/rl/init_transport", post(rl_init_transport))
        // LoRA hot-swap.
        .route("/v1/rl/load_lora_adapter", post(rl_load_lora_adapter))
        // Legacy endpoints — kept for back-compat until existing clients
        // migrate to /v1/rl/state. Removed in a follow-up.
        .route("/v1/rl/health", get(rl_health))
        .route("/v1/rl/ready", get(rl_ready))
        .route("/v1/rl/weight_version", get(rl_weight_version))
        .route("/v1/rl/unload_lora_adapter", post(rl_unload_lora_adapter))
        .with_state(rl_state_arc);
    Ok((docs, router))
}
