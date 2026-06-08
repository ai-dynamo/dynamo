// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::collections::HashMap;
use std::sync::Arc;

use axum::extract::{DefaultBodyLimit, Query, State};
use axum::http::StatusCode;
use axum::response::IntoResponse;
use axum::routing::{get, post};
use axum::{Json, Router};
#[cfg(feature = "metrics")]
use prometheus::Encoder;
use serde::{Deserialize, Serialize};

use crate::indexer::TieredMatchDetails;
use crate::protocols::{
    BlockHashOptions, LocalBlockHash, StorageTier, WorkerId, compute_block_hash_for_seq,
};

use super::backend::Indexer;
use super::registry::{IndexerKey, ListenerControlError, WorkerRegistry};

/// We need to fit one million tokens as JSON text, this should do it.
const QUERY_REQUEST_BODY_LIMIT_BYTES: usize = 8 * 1024 * 1024;

/// Gates the listener-control test endpoints; unset in production, set by the
/// e2e harness. Accepts `1`/`true`/`yes`/`on`.
const DYN_KV_INDEXER_TEST_ENDPOINTS: &str = "DYN_KV_INDEXER_TEST_ENDPOINTS";

fn test_endpoints_enabled() -> bool {
    matches!(
        std::env::var(DYN_KV_INDEXER_TEST_ENDPOINTS)
            .unwrap_or_default()
            .trim()
            .to_ascii_lowercase()
            .as_str(),
        "1" | "true" | "yes" | "on"
    )
}

pub struct AppState {
    pub registry: Arc<WorkerRegistry>,
    #[cfg(feature = "metrics")]
    pub prom_registry: prometheus::Registry,
}

fn default_tenant() -> String {
    "default".to_string()
}

#[derive(Deserialize)]
pub struct RegisterRequest {
    pub instance_id: WorkerId,
    pub endpoint: String,
    pub model_name: String,
    #[serde(default = "default_tenant")]
    pub tenant_id: String,
    pub block_size: u32,
    #[serde(default)]
    pub dp_rank: Option<u32>,
    #[serde(default)]
    pub replay_endpoint: Option<String>,
    /// Optional per-tenant salt (Mooncake RFC #1403 `additionalsalt`).
    /// Currently accepted but not yet mixed into hashes — engines apply
    /// their own salt internally. Plumbed for forward compatibility.
    #[serde(default, alias = "additionalsalt")]
    pub additional_salt: Option<String>,
}

#[derive(Deserialize)]
pub struct UnregisterRequest {
    pub instance_id: WorkerId,
    pub model_name: String,
    #[serde(default)]
    pub tenant_id: Option<String>,
    #[serde(default)]
    pub dp_rank: Option<u32>,
}

#[derive(Deserialize)]
pub struct QueryRequest {
    pub token_ids: Vec<u32>,
    pub model_name: String,
    #[serde(default = "default_tenant")]
    pub tenant_id: String,
    #[serde(default)]
    pub lora_name: Option<String>,
    /// Optional per-request cache salt (Mooncake RFC #1403). Currently accepted
    /// but not yet mixed into hashes — engines apply their own internally.
    #[serde(default)]
    pub cache_salt: Option<String>,
}

#[derive(Deserialize)]
pub struct QueryByHashRequest {
    pub block_hashes: Vec<i64>,
    pub model_name: String,
    #[serde(default = "default_tenant")]
    pub tenant_id: String,
    /// Optional per-request cache salt (Mooncake RFC #1403). Currently accepted
    /// but not yet mixed into hashes — engines apply their own internally.
    #[serde(default)]
    pub cache_salt: Option<String>,
}

/// Response shape for `/query` and `/query_by_hash`.
///
/// The flat `scores`/`frequencies` fields are kept for backward compatibility
/// with existing callers. New callers should consume the `instances` map,
/// which mirrors the per-instance, per-tier breakdown proposed in Mooncake
/// RFC #1403 (kvcache-ai/Mooncake#1403):
/// `{instance_id: {longest_matched, gpu, dp: {rank: count}, cpu, disk}}`.
#[derive(Serialize)]
struct ScoreResponse {
    scores: HashMap<String, HashMap<String, u32>>,
    frequencies: Vec<usize>,
    /// Per-instance tier breakdown (Mooncake RFC #1403 alignment).
    instances: HashMap<String, InstanceTierBreakdown>,
}

/// Per-instance match summary in Mooncake RFC #1403 shape.
///
/// All counts are in *tokens* (block count × `block_size`), matching the flat
/// `scores` fields. The tier counts are CUMULATIVE through each tier's walk:
/// `cpu` includes everything reachable through device → host-pinned, and
/// `disk` includes everything reachable through device → host → disk. Under a
/// natural offload pipeline where blocks flow device → host → disk, these
/// satisfy `gpu ≤ cpu ≤ disk`. `longest_matched` is the max across the three
/// and is useful as a single-number "best prefix length" the gateway can use.
#[derive(Serialize, Default)]
struct InstanceTierBreakdown {
    longest_matched: u32,
    gpu: u32,
    /// Per-`dp_rank` device-tier match counts.
    dp: HashMap<String, u32>,
    cpu: u32,
    disk: u32,
}

async fn register(
    State(state): State<Arc<AppState>>,
    Json(req): Json<RegisterRequest>,
) -> impl IntoResponse {
    if let Err(error) =
        super::validate_listener_endpoints(&req.endpoint, req.replay_endpoint.as_deref())
    {
        return (
            StatusCode::BAD_REQUEST,
            Json(serde_json::json!({"error": error.to_string()})),
        );
    }

    match state
        .registry
        .register(
            req.instance_id,
            req.endpoint,
            req.dp_rank.unwrap_or(0),
            req.model_name,
            req.tenant_id,
            req.block_size,
            req.replay_endpoint,
        )
        .await
    {
        Ok(()) => (
            StatusCode::CREATED,
            Json(serde_json::json!({"status": "ok"})),
        ),
        Err(e) => (
            StatusCode::CONFLICT,
            Json(serde_json::json!({"error": e.to_string()})),
        ),
    }
}

async fn unregister(
    State(state): State<Arc<AppState>>,
    Json(req): Json<UnregisterRequest>,
) -> impl IntoResponse {
    let result = match req.tenant_id {
        Some(tenant_id) => match req.dp_rank {
            Some(dp_rank) => {
                state
                    .registry
                    .deregister_dp_rank(req.instance_id, dp_rank, &req.model_name, &tenant_id)
                    .await
            }
            None => {
                state
                    .registry
                    .deregister(req.instance_id, &req.model_name, &tenant_id)
                    .await
            }
        },
        None => {
            state
                .registry
                .deregister_all_tenants(req.instance_id, &req.model_name)
                .await
        }
    };
    match result {
        Ok(()) => (StatusCode::OK, Json(serde_json::json!({"status": "ok"}))),
        Err(e) => (
            StatusCode::NOT_FOUND,
            Json(serde_json::json!({"error": e.to_string()})),
        ),
    }
}

/// Optional query parameters for `GET /workers`.
///
/// Both fields are independent filters; omitting one skips that dimension.
/// Example: `GET /workers?model_name=llama3&tenant_id=acme`
#[derive(Deserialize)]
struct WorkersQuery {
    model_name: Option<String>,
    tenant_id: Option<String>,
}

async fn list_workers(
    State(state): State<Arc<AppState>>,
    Query(params): Query<WorkersQuery>,
) -> impl IntoResponse {
    Json(
        state
            .registry
            .list_filtered(params.model_name.as_deref(), params.tenant_id.as_deref()),
    )
}

/// Build the [`ScoreResponse`] in both the flat (legacy) and per-instance
/// (Mooncake RFC #1403) shapes from a tiered match result.
///
/// All token counts are scaled from blocks → tokens via `block_size`.
fn build_score_response(tiered: &TieredMatchDetails, block_size: u32) -> ScoreResponse {
    // Flat fields (unchanged) come from the device-tier overlap.
    let device = &tiered.device.overlap_scores;

    let mut scores: HashMap<String, HashMap<String, u32>> = HashMap::new();
    for (k, v) in &device.scores {
        scores
            .entry(k.worker_id.to_string())
            .or_default()
            .insert(k.dp_rank.to_string(), v * block_size);
    }

    // Per-worker (instance + dp_rank) reaches: cumulative through each tier.
    // The lower-tier indexer reports per-tier *extension* blocks beyond the
    // previous tier; we accumulate them here so the per-tier counts answer
    // "how many prefix tokens does this worker have through this tier" —
    // which is the natural reading of Mooncake RFC #1403's `GPU`/`CPU`/`DISK`
    // fields. Each worker's tier counts therefore satisfy gpu ≤ cpu ≤ disk
    // (since lower tiers extend the device match rather than shrink it).
    let host_extension = tiered.lower_tier.get(&StorageTier::HostPinned);
    let disk_extension = tiered.lower_tier.get(&StorageTier::Disk);
    let external_extension = tiered.lower_tier.get(&StorageTier::External);

    // Helper: blocks for `worker` in `extension`, defaulting to 0.
    let ext = |extension: Option<&crate::indexer::LowerTierMatchDetails>,
               worker: &crate::protocols::WorkerWithDpRank|
     -> u32 {
        extension
            .and_then(|e| e.hits.get(worker))
            .map(|&n| n as u32)
            .unwrap_or(0)
    };

    let mut instances: HashMap<String, InstanceTierBreakdown> = HashMap::new();

    // Collect the union of all workers seen in device tier and extension tiers.
    let mut all_workers = std::collections::HashSet::new();
    for worker in device.scores.keys() {
        all_workers.insert(*worker);
    }
    for extension in [host_extension, disk_extension, external_extension]
        .iter()
        .filter_map(|&e| e)
    {
        for worker in extension.hits.keys() {
            all_workers.insert(*worker);
        }
    }

    for worker in all_workers {
        let gpu_blocks = device.scores.get(&worker).copied().unwrap_or(0);
        let cpu_blocks = gpu_blocks + ext(host_extension, &worker);
        // Treat External as further-away storage and roll it into the disk
        // bucket alongside Disk; both extensions stack on top of host-pinned.
        let disk_blocks =
            cpu_blocks + ext(disk_extension, &worker) + ext(external_extension, &worker);

        let gpu_tokens = gpu_blocks * block_size;
        let cpu_tokens = cpu_blocks * block_size;
        let disk_tokens = disk_blocks * block_size;

        let entry = instances.entry(worker.worker_id.to_string()).or_default();

        entry.dp.insert(worker.dp_rank.to_string(), gpu_tokens);
        entry.gpu = entry.gpu.max(gpu_tokens);
        entry.cpu = entry.cpu.max(cpu_tokens);
        entry.disk = entry.disk.max(disk_tokens);
    }

    for entry in instances.values_mut() {
        entry.longest_matched = entry.gpu.max(entry.cpu).max(entry.disk);
    }

    ScoreResponse {
        scores,
        frequencies: tiered.device.overlap_scores.frequencies.clone(),
        instances,
    }
}

/// Run a tiered query and serialize the result, returning the appropriate
/// HTTP status. Shared between `/query` and `/query_by_hash`.
async fn run_tiered_query(
    indexer: &Indexer,
    block_hashes: Vec<LocalBlockHash>,
    block_size: u32,
) -> (StatusCode, Json<serde_json::Value>) {
    match indexer.find_tiered_matches(block_hashes).await {
        Ok(tiered) => (
            StatusCode::OK,
            Json(serde_json::json!(build_score_response(&tiered, block_size))),
        ),
        Err(e) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(serde_json::json!({"error": e.to_string()})),
        ),
    }
}

async fn query(
    State(state): State<Arc<AppState>>,
    Json(req): Json<QueryRequest>,
) -> impl IntoResponse {
    let key = IndexerKey {
        model_name: req.model_name,
        tenant_id: req.tenant_id,
    };
    let Some(ie) = state.registry.get_indexer(&key) else {
        return (
            StatusCode::NOT_FOUND,
            Json(serde_json::json!({
                "error": format!("no indexer for model={} tenant={}", key.model_name, key.tenant_id)
            })),
        );
    };
    let block_size = ie.block_size;
    let indexer = ie.indexer.clone();
    drop(ie);

    let block_hashes = compute_block_hash_for_seq(
        &req.token_ids,
        block_size,
        BlockHashOptions {
            lora_name: req.lora_name.as_deref(),
            ..Default::default()
        },
    );
    run_tiered_query(&indexer, block_hashes, block_size).await
}

async fn query_by_hash(
    State(state): State<Arc<AppState>>,
    Json(req): Json<QueryByHashRequest>,
) -> impl IntoResponse {
    let key = IndexerKey {
        model_name: req.model_name,
        tenant_id: req.tenant_id,
    };
    let Some(ie) = state.registry.get_indexer(&key) else {
        return (
            StatusCode::NOT_FOUND,
            Json(serde_json::json!({
                "error": format!("no indexer for model={} tenant={}", key.model_name, key.tenant_id)
            })),
        );
    };
    let block_size = ie.block_size;
    let indexer = ie.indexer.clone();
    drop(ie);

    let block_hashes: Vec<LocalBlockHash> = req
        .block_hashes
        .iter()
        .map(|h| LocalBlockHash(*h as u64))
        .collect();
    run_tiered_query(&indexer, block_hashes, block_size).await
}

#[derive(Deserialize)]
struct ListenerControlRequest {
    instance_id: WorkerId,
    #[serde(default)]
    dp_rank: Option<u32>,
}

async fn test_pause_listener(
    State(state): State<Arc<AppState>>,
    Json(req): Json<ListenerControlRequest>,
) -> impl IntoResponse {
    match state
        .registry
        .pause_listener(req.instance_id, req.dp_rank.unwrap_or(0))
    {
        Ok(()) => (StatusCode::OK, Json(serde_json::json!({"status": "ok"}))),
        Err(error) => listener_control_error_response(error),
    }
}

async fn test_resume_listener(
    State(state): State<Arc<AppState>>,
    Json(req): Json<ListenerControlRequest>,
) -> impl IntoResponse {
    match state
        .registry
        .resume_listener(req.instance_id, req.dp_rank.unwrap_or(0))
        .await
    {
        Ok(()) => (StatusCode::OK, Json(serde_json::json!({"status": "ok"}))),
        Err(error) => listener_control_error_response(error),
    }
}

fn listener_control_error_response(
    error: ListenerControlError,
) -> (StatusCode, Json<serde_json::Value>) {
    let status = match &error {
        ListenerControlError::WorkerNotFound { .. }
        | ListenerControlError::ListenerNotFound { .. } => StatusCode::NOT_FOUND,
        ListenerControlError::InvalidPauseState { .. }
        | ListenerControlError::InvalidResumeState { .. } => StatusCode::CONFLICT,
    };
    (
        status,
        Json(serde_json::json!({"error": error.to_string()})),
    )
}

#[derive(Deserialize)]
struct PeerRequest {
    url: String,
}

async fn register_peer(
    State(state): State<Arc<AppState>>,
    Json(req): Json<PeerRequest>,
) -> impl IntoResponse {
    state.registry.register_peer(req.url);
    (
        StatusCode::CREATED,
        Json(serde_json::json!({"status": "ok"})),
    )
}

async fn deregister_peer(
    State(state): State<Arc<AppState>>,
    Json(req): Json<PeerRequest>,
) -> impl IntoResponse {
    if state.registry.deregister_peer(&req.url) {
        (StatusCode::OK, Json(serde_json::json!({"status": "ok"})))
    } else {
        (
            StatusCode::NOT_FOUND,
            Json(serde_json::json!({"error": "peer not found"})),
        )
    }
}

async fn list_peers(State(state): State<Arc<AppState>>) -> impl IntoResponse {
    Json(state.registry.list_peers())
}

async fn dump_events(State(state): State<Arc<AppState>>) -> impl IntoResponse {
    let all = state.registry.all_indexers_with_block_size();
    let mut handles = Vec::with_capacity(all.len());

    for (key, indexer, block_size) in all {
        handles.push(tokio::spawn(async move {
            let events = indexer.dump_events().await;
            (key, events, block_size)
        }));
    }

    let mut result: HashMap<String, serde_json::Value> = HashMap::new();
    for handle in handles {
        match handle.await {
            Ok((key, Ok(events), block_size)) => {
                let map_key = format!("{}:{}", key.model_name, key.tenant_id);
                result.insert(
                    map_key,
                    serde_json::json!({
                        "block_size": block_size,
                        "events": events,
                    }),
                );
            }
            Ok((key, Err(e), _)) => {
                let map_key = format!("{}:{}", key.model_name, key.tenant_id);
                result.insert(map_key, serde_json::json!({"error": e.to_string()}));
            }
            Err(e) => {
                tracing::warn!("dump task join error: {e}");
            }
        }
    }
    (StatusCode::OK, Json(serde_json::json!(result)))
}

async fn handle_health() -> StatusCode {
    StatusCode::OK
}

#[cfg(feature = "metrics")]
async fn handle_metrics(State(state): State<Arc<AppState>>) -> impl IntoResponse {
    state.registry.refresh_metrics();
    let encoder = prometheus::TextEncoder::new();
    let mut buf = Vec::new();
    encoder
        .encode(&state.prom_registry.gather(), &mut buf)
        .unwrap();
    (
        StatusCode::OK,
        [(
            axum::http::header::CONTENT_TYPE,
            prometheus::TEXT_FORMAT.to_string(),
        )],
        buf,
    )
}

pub fn create_router(state: Arc<AppState>) -> Router {
    build_router(state, test_endpoints_enabled())
}

/// Mounts the listener-control test endpoints only when `test_endpoints` is
/// true; the explicit parameter lets tests exercise both states.
fn build_router(state: Arc<AppState>, test_endpoints: bool) -> Router {
    let router = Router::new()
        .route("/register", post(register))
        .route("/unregister", post(unregister))
        .route("/workers", get(list_workers))
        .route(
            "/query",
            post(query).layer(DefaultBodyLimit::max(QUERY_REQUEST_BODY_LIMIT_BYTES)),
        )
        .route("/query_by_hash", post(query_by_hash))
        .route("/dump", get(dump_events))
        .route("/register_peer", post(register_peer))
        .route("/deregister_peer", post(deregister_peer))
        .route("/peers", get(list_peers))
        .route("/health", get(handle_health));

    let mut router = router;
    if test_endpoints {
        tracing::warn!(
            "Mounting listener-control test endpoints (/test/pause_listener, \
             /test/resume_listener). These manipulate worker listener state and must never be \
             enabled in production."
        );
        router = router
            .route("/test/pause_listener", post(test_pause_listener))
            .route("/test/resume_listener", post(test_resume_listener));
    }
    let router = router.with_state(state.clone());

    #[cfg(feature = "metrics")]
    let router = {
        let metrics_route = Router::new()
            .route("/metrics", get(handle_metrics))
            .with_state(state);
        router
            .layer(axum::middleware::from_fn(
                super::metrics::metrics_middleware,
            ))
            .merge(metrics_route)
    };

    router
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::indexer::KvIndexerInterface;
    use crate::services::indexer::backend::create_indexer;
    use crate::services::indexer::backend::test_util::store_event;
    use axum::body::Body;
    use axum::http::{Request, StatusCode, header};
    use tower::ServiceExt;

    /// Drive a tiered query through `build_score_response` after feeding
    /// mixed-tier events. The response must carry both shapes:
    /// - flat `scores`/`tree_sizes` (legacy; used by existing callers), and
    /// - `instances` map keyed by stringified `worker_id` with per-tier
    ///   counts plus `longest_matched`, matching Mooncake RFC #1403.
    #[tokio::test]
    async fn build_score_response_contains_per_instance_tier_breakdown() {
        let block_size: u32 = 4;
        let indexer = create_indexer(block_size, 1);

        // Worker 7 owns 2 device blocks and a 3rd anchored on host-pinned.
        // Worker 8 owns the same 2 device blocks with no lower tier.
        for &worker_id in &[7u64, 8] {
            indexer
                .apply_event_routed(store_event(
                    worker_id,
                    0,
                    1,
                    &[],
                    &[11, 12],
                    StorageTier::Device,
                ))
                .await;
        }
        indexer
            .apply_event_routed(store_event(
                7,
                0,
                2,
                &[11, 12],
                &[13],
                StorageTier::HostPinned,
            ))
            .await;

        // Flush primary + lower tiers.
        if let Indexer::Single {
            primary,
            lower_tier,
        } = &indexer
        {
            let _ = primary.flush().await;
            for inner in lower_tier.all() {
                let _ = inner.dump_events().await.unwrap();
            }
        }

        let sequence = vec![LocalBlockHash(11), LocalBlockHash(12), LocalBlockHash(13)];
        let tiered = indexer.find_tiered_matches(sequence).await.unwrap();
        let response = build_score_response(&tiered, block_size);

        // Flat shape (legacy callers) carries device-tier overlap scaled by block_size.
        assert_eq!(
            response
                .scores
                .get("7")
                .and_then(|by_dp| by_dp.get("0").copied()),
            Some(2 * block_size),
            "legacy `scores` must still reflect device-tier hits"
        );

        // Per-instance breakdown (Mooncake RFC #1403 alignment).
        // Tier counts are CUMULATIVE through each tier's walk: cpu includes
        // device's reach plus the host-pinned extension; disk includes
        // everything below it. Without a disk extension, disk == cpu.
        let inst_7 = response
            .instances
            .get("7")
            .expect("instance 7 must appear with tier breakdown");
        assert_eq!(inst_7.gpu, 2 * block_size, "instance 7 device count");
        assert_eq!(
            inst_7.cpu,
            3 * block_size,
            "instance 7 host-pinned cumulative count = device + host extension"
        );
        assert_eq!(
            inst_7.disk,
            3 * block_size,
            "instance 7 disk cumulative falls back to cpu when no disk extension exists"
        );
        assert_eq!(
            inst_7.dp.get("0").copied(),
            Some(2 * block_size),
            "instance 7 dp_rank=0 device count"
        );
        assert_eq!(
            inst_7.longest_matched,
            3 * block_size,
            "longest_matched should be the max across device/host/disk"
        );

        let inst_8 = response
            .instances
            .get("8")
            .expect("instance 8 must appear with tier breakdown");
        assert_eq!(inst_8.gpu, 2 * block_size);
        assert_eq!(
            inst_8.cpu,
            2 * block_size,
            "instance 8 cpu falls back to device when no host extension exists"
        );
        assert_eq!(inst_8.disk, 2 * block_size);
        assert_eq!(inst_8.longest_matched, 2 * block_size);
    }

    fn oversized_query_body() -> String {
        let mut body = String::from(r#"{"token_ids":["#);
        let mut first = true;

        while body.len() <= QUERY_REQUEST_BODY_LIMIT_BYTES {
            if !first {
                body.push(',');
            }
            first = false;
            body.push('0');
        }

        body.push_str(r#"],"model_name":"model"}"#);
        body
    }

    #[tokio::test]
    async fn query_rejects_request_bodies_over_limit() {
        let app = create_router(Arc::new(AppState {
            registry: Arc::new(WorkerRegistry::new(1)),
            #[cfg(feature = "metrics")]
            prom_registry: prometheus::Registry::new(),
        }));

        let response = app
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/query")
                    .header(header::CONTENT_TYPE, "application/json")
                    .body(Body::from(oversized_query_body()))
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(response.status(), StatusCode::PAYLOAD_TOO_LARGE);
    }

    fn empty_indexer_router(test_endpoints: bool) -> Router {
        build_router(
            Arc::new(AppState {
                registry: Arc::new(WorkerRegistry::new(1)),
                #[cfg(feature = "metrics")]
                prom_registry: prometheus::Registry::new(),
            }),
            test_endpoints,
        )
    }

    /// Malformed body so the states are distinguishable: an unmounted route
    /// 404s before parsing, a mounted handler rejects it with a non-404. (A
    /// valid body would 404 via `WorkerNotFound` either way.)
    async fn post_malformed(app: &Router, uri: &str) -> StatusCode {
        app.clone()
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri(uri)
                    .header(header::CONTENT_TYPE, "application/json")
                    .body(Body::from("not json"))
                    .unwrap(),
            )
            .await
            .unwrap()
            .status()
    }

    const LISTENER_CONTROL_ROUTES: [&str; 2] = ["/test/pause_listener", "/test/resume_listener"];

    #[tokio::test]
    async fn listener_control_endpoints_are_not_mounted_by_default() {
        let app = empty_indexer_router(false);
        for uri in LISTENER_CONTROL_ROUTES {
            assert_eq!(
                post_malformed(&app, uri).await,
                StatusCode::NOT_FOUND,
                "{uri} must not be mounted unless test endpoints are explicitly enabled"
            );
        }
    }

    #[tokio::test]
    async fn listener_control_endpoints_are_mounted_when_enabled() {
        let app = empty_indexer_router(true);
        for uri in LISTENER_CONTROL_ROUTES {
            let status = post_malformed(&app, uri).await;
            assert_ne!(
                status,
                StatusCode::NOT_FOUND,
                "{uri} should be mounted and reject the malformed body, not 404"
            );
            assert!(
                status.is_client_error(),
                "{uri} should reject the malformed body with a 4xx, got {status}"
            );
        }
    }

    // ── /workers endpoint tests ───────────────────────────────────────────────

    #[tokio::test]
    async fn get_workers_returns_registered_workers_with_metadata() {
        let registry = Arc::new(WorkerRegistry::new(1));
        registry.signal_ready();

        // Worker 20: llama3 / acme, block_size=4
        registry
            .register(
                20,
                "tcp://127.0.0.1:15590".to_string(),
                0,
                "llama3".to_string(),
                "acme".to_string(),
                4,
                None,
            )
            .await
            .unwrap();

        // Worker 21: mistral / acme, block_size=8
        registry
            .register(
                21,
                "tcp://127.0.0.1:15591".to_string(),
                0,
                "mistral".to_string(),
                "acme".to_string(),
                8,
                None,
            )
            .await
            .unwrap();

        let app = create_router(Arc::new(AppState {
            registry,
            #[cfg(feature = "metrics")]
            prom_registry: prometheus::Registry::new(),
        }));

        // Unfiltered: both workers
        let response = app
            .clone()
            .oneshot(
                Request::builder()
                    .method("GET")
                    .uri("/workers")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(response.status(), StatusCode::OK);
        let bytes = axum::body::to_bytes(response.into_body(), usize::MAX)
            .await
            .unwrap();
        let workers: Vec<serde_json::Value> = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(workers.len(), 2);

        let mut model_names: Vec<_> = workers
            .iter()
            .filter_map(|w| w["model_name"].as_str())
            .collect();
        model_names.sort();
        assert_eq!(model_names, ["llama3", "mistral"]);

        let llama = workers
            .iter()
            .find(|w| w["model_name"] == "llama3")
            .unwrap();
        assert_eq!(llama["block_size"], 4);
        assert_eq!(llama["tenant_id"], "acme");

        // Filtered by model_name=llama3: only one result
        let response = app
            .clone()
            .oneshot(
                Request::builder()
                    .method("GET")
                    .uri("/workers?model_name=llama3")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(response.status(), StatusCode::OK);
        let bytes = axum::body::to_bytes(response.into_body(), usize::MAX)
            .await
            .unwrap();
        let filtered: Vec<serde_json::Value> = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(filtered.len(), 1);
        assert_eq!(filtered[0]["model_name"], "llama3");

        // Filtered by nonexistent model: empty array, not a 404
        let response = app
            .oneshot(
                Request::builder()
                    .method("GET")
                    .uri("/workers?model_name=nonexistent")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(response.status(), StatusCode::OK);
        let bytes = axum::body::to_bytes(response.into_body(), usize::MAX)
            .await
            .unwrap();
        let empty: Vec<serde_json::Value> = serde_json::from_slice(&bytes).unwrap();
        assert!(empty.is_empty());
    }
}
