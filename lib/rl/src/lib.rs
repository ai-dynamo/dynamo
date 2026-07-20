// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Dynamo RL worker discovery surface.
//!
//! Workers that run with `DYN_ENABLE_RL` / `--enable-rl` register an `rl`
//! request-plane endpoint:
//!
//! ```text
//! dyn://<namespace>.<component>.rl
//! ```
//!
//! This crate exposes a read-only frontend route. The frontend discovers live
//! `rl` endpoint instances from Dynamo discovery, then asks each worker for its
//! available Dynamo system routes with `{"method": "routes"}` over the request
//! plane. It does not expose a frontend fan-out method endpoint.

use std::{
    collections::{HashMap, HashSet},
    sync::Arc,
    time::Duration,
};

use axum::{Json, Router, extract::State, http::StatusCode, response::IntoResponse, routing::get};
use dynamo_runtime::{
    DistributedRuntime,
    component::{Client, Instance, TransportType},
    discovery::{DiscoveryInstance, DiscoveryQuery},
    pipeline::{
        SingleIn,
        network::egress::push_router::{PushRouter, RouterMode},
    },
    protocols::annotated::Annotated,
};
use futures::{StreamExt, future::join_all};

const DEFAULT_NAMESPACE: &str = "dynamo";
const DEFAULT_RL_ENDPOINT: &str = "rl";
const DEFAULT_REQUEST_TIMEOUT_SECS: u64 = 30;
/// Global cap on concurrent per-worker probes (across all in-flight discovery
/// requests), so a large fleet or many concurrent callers can't fan out without bound.
const DEFAULT_MAX_CONCURRENT_PROBES: usize = 32;
pub const RL_WORKERS_PROTOCOL_VERSION: u32 = 1;

type ModelKey = (String, String, u64);

#[derive(Clone)]
pub struct RlDiscoveryConfig {
    pub runtime: Arc<DistributedRuntime>,
    pub namespace: String,
    pub rl_endpoint: String,
    pub component_filter: Option<Vec<String>>,
    pub request_timeout: Duration,
    pub max_concurrent_probes: usize,
}

impl RlDiscoveryConfig {
    pub fn from_env(runtime: Arc<DistributedRuntime>) -> Self {
        let namespace = std::env::var("DYN_NAMESPACE").unwrap_or_else(|_| DEFAULT_NAMESPACE.into());
        let rl_endpoint =
            std::env::var("DYN_RL_ENDPOINT").unwrap_or_else(|_| DEFAULT_RL_ENDPOINT.into());
        let component_filter = parse_csv_env("DYN_RL_COMPONENTS")
            .or_else(|| std::env::var("DYN_RL_COMPONENT").ok().map(|c| vec![c]));
        let request_timeout = std::env::var("DYN_RL_REQUEST_TIMEOUT_SECS")
            .ok()
            .and_then(|value| value.parse::<u64>().ok())
            .map(Duration::from_secs)
            .unwrap_or_else(|| Duration::from_secs(DEFAULT_REQUEST_TIMEOUT_SECS));
        let max_concurrent_probes = std::env::var("DYN_RL_MAX_CONCURRENT_PROBES")
            .ok()
            .and_then(|value| value.parse::<usize>().ok())
            .filter(|value| *value > 0)
            .unwrap_or(DEFAULT_MAX_CONCURRENT_PROBES);

        Self {
            runtime,
            namespace,
            rl_endpoint,
            component_filter,
            request_timeout,
            max_concurrent_probes,
        }
    }
}

fn parse_csv_env(name: &str) -> Option<Vec<String>> {
    let values = std::env::var(name).ok()?;
    let parsed = values
        .split(',')
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .map(ToString::to_string)
        .collect::<Vec<_>>();
    (!parsed.is_empty()).then_some(parsed)
}

#[derive(Debug, Clone, serde::Serialize)]
pub struct RlWorkerInfo {
    pub namespace: String,
    pub component: String,
    pub endpoint: String,
    pub instance_id: u64,
    pub transport: TransportType,
    pub request_plane_url: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub system_url: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub admin_base_url: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub world_size: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub model: Option<String>,
    pub system_routes: Vec<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
}

#[derive(Debug, Clone, serde::Serialize)]
pub struct RlWorkersResponse {
    pub protocol_version: u32,
    pub namespace: String,
    pub workers: Vec<RlWorkerInfo>,
}

type EndpointKey = (String, String, String);

#[derive(Clone)]
pub struct RlDiscoveryState {
    config: Arc<RlDiscoveryConfig>,
    /// Cache of request-plane clients keyed by (namespace, component, endpoint).
    /// A `Client` spawns a runtime-lived instance-monitor task and has no per-client
    /// Drop cleanup, so building one per request would leak a task per (request*worker).
    /// Cache and reuse one client per endpoint instead.
    clients: Arc<tokio::sync::Mutex<HashMap<EndpointKey, Client>>>,
    /// Caps concurrent per-worker probes across all in-flight discovery requests,
    /// so a large fleet (or many concurrent callers) can't fan out without bound.
    probe_semaphore: Arc<tokio::sync::Semaphore>,
}

impl RlDiscoveryState {
    pub fn new(config: RlDiscoveryConfig) -> Self {
        let permits = config.max_concurrent_probes.max(1);
        Self {
            config: Arc::new(config),
            clients: Arc::new(tokio::sync::Mutex::new(HashMap::new())),
            probe_semaphore: Arc::new(tokio::sync::Semaphore::new(permits)),
        }
    }

    /// Get (or lazily create and cache) the request-plane client for an endpoint.
    /// The lock is held across creation so concurrent fan-out for the same endpoint
    /// creates exactly one client (cloning a `Client` shares its monitor task).
    async fn client_for(
        &self,
        namespace: &str,
        component: &str,
        endpoint: &str,
    ) -> anyhow::Result<Client> {
        let key = (
            namespace.to_string(),
            component.to_string(),
            endpoint.to_string(),
        );
        let mut guard = self.clients.lock().await;
        if let Some(client) = guard.get(&key) {
            return Ok(client.clone());
        }
        let client = self
            .config
            .runtime
            .namespace(namespace)?
            .component(component)?
            .endpoint(endpoint.to_string())
            .client()
            .await?;
        guard.insert(key, client.clone());
        Ok(client)
    }

    /// Drop cached clients for endpoints no longer present, so the cache can't grow
    /// without bound under endpoint/component churn. Stable components (the common
    /// case) are always in `live`, so this is a no-op for them.
    ///
    /// Note: dropping a `Client` frees its cache slot but does not stop the
    /// runtime-lived instance-monitor task it spawned — the `dynamo-runtime` `Client`
    /// has no per-client cancellation/`Drop`. For RL discovery the live endpoint set is
    /// small and stable, so this is bounded in practice; fully reclaiming the monitor
    /// task on drop is a runtime `Client` lifecycle change tracked outside this crate.
    async fn retain_endpoints(&self, live: &HashSet<EndpointKey>) {
        let mut guard = self.clients.lock().await;
        guard.retain(|key, _| live.contains(key));
    }
}

pub fn rl_router(state: RlDiscoveryState) -> Router {
    Router::new()
        .route("/v1/rl/workers", get(workers_handler))
        .with_state(state)
}

async fn workers_handler(State(state): State<RlDiscoveryState>) -> impl IntoResponse {
    match list_workers(&state).await {
        Ok(workers) => Json(RlWorkersResponse {
            protocol_version: RL_WORKERS_PROTOCOL_VERSION,
            namespace: state.config.namespace.clone(),
            workers,
        })
        .into_response(),
        Err(err) => {
            tracing::error!("failed to list RL workers: {err}");
            (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(serde_json::json!({
                    "error": "discovery_failed",
                    "message": err.to_string(),
                })),
            )
                .into_response()
        }
    }
}

async fn list_workers(state: &RlDiscoveryState) -> anyhow::Result<Vec<RlWorkerInfo>> {
    let config = &state.config;
    let endpoint_instances = config
        .runtime
        .discovery()
        .list(DiscoveryQuery::NamespacedEndpoints {
            namespace: config.namespace.clone(),
        })
        .await?;

    let model_instances = config
        .runtime
        .discovery()
        .list(DiscoveryQuery::NamespacedModels {
            namespace: config.namespace.clone(),
        })
        .await?;

    let models = model_map(model_instances);
    let rl_endpoints = endpoint_instances
        .into_iter()
        .filter_map(|instance| match instance {
            DiscoveryInstance::Endpoint(endpoint) => Some(endpoint),
            _ => None,
        })
        .filter(|endpoint| endpoint.endpoint == config.rl_endpoint)
        .filter(|endpoint| {
            config
                .component_filter
                .as_ref()
                .map(|components| components.iter().any(|c| c == &endpoint.component))
                .unwrap_or(true)
        })
        .collect::<Vec<_>>();

    // Bound the client cache (N2): drop clients for endpoints that are no longer
    // present. Live endpoints are retained, so the fan-out below still reuses their
    // cached clients rather than recreating them.
    let live_endpoints: HashSet<EndpointKey> = rl_endpoints
        .iter()
        .map(|endpoint| {
            (
                endpoint.namespace.clone(),
                endpoint.component.clone(),
                endpoint.endpoint.clone(),
            )
        })
        .collect();
    state.retain_endpoints(&live_endpoints).await;

    let mut workers = join_all(rl_endpoints.into_iter().map(|endpoint| {
        let state = state.clone();
        let timeout = config.request_timeout;
        let model = models
            .get(&(
                endpoint.namespace.clone(),
                endpoint.component.clone(),
                endpoint.instance_id,
            ))
            .cloned();
        async move { describe_worker(&state, endpoint, model, timeout).await }
    }))
    .await;

    workers.sort_by(|a, b| {
        (&a.namespace, &a.component, &a.endpoint, a.instance_id).cmp(&(
            &b.namespace,
            &b.component,
            &b.endpoint,
            b.instance_id,
        ))
    });

    workers.dedup_by(|a, b| {
        a.namespace == b.namespace
            && a.component == b.component
            && a.endpoint == b.endpoint
            && a.instance_id == b.instance_id
    });

    Ok(workers)
}

async fn describe_worker(
    state: &RlDiscoveryState,
    endpoint: Instance,
    model: Option<String>,
    timeout: Duration,
) -> RlWorkerInfo {
    // Bound the whole per-worker probe — INCLUDING the wait for a global concurrency
    // permit (N1) — by request_timeout, so neither a slow worker nor a saturated
    // semaphore can make per-worker latency unbounded (true end-to-end deadline, F2).
    let probe = async {
        let _permit = state
            .probe_semaphore
            .acquire()
            .await
            .map_err(|_| anyhow::anyhow!("rl discovery is shutting down"))?;
        call_worker_routes(state, &endpoint, timeout).await
    };
    match tokio::time::timeout(timeout, probe).await {
        Ok(Ok(routes)) => worker_info(
            endpoint,
            model,
            routes.routes,
            routes.system_url,
            routes.admin_base_url,
            routes.world_size,
            None,
        ),
        Ok(Err(err)) => worker_info(
            endpoint,
            model,
            Vec::new(),
            None,
            None,
            None,
            Some(err.to_string()),
        ),
        Err(_) => worker_info(
            endpoint,
            model,
            Vec::new(),
            None,
            None,
            None,
            Some(format!(
                "worker discovery timed out after {}s",
                timeout.as_secs()
            )),
        ),
    }
}

#[derive(Debug, Default)]
struct WorkerRoutes {
    routes: Vec<String>,
    system_url: Option<String>,
    admin_base_url: Option<String>,
    world_size: Option<u32>,
}

async fn call_worker_routes(
    state: &RlDiscoveryState,
    target: &Instance,
    timeout: Duration,
) -> anyhow::Result<WorkerRoutes> {
    // Reuse a cached client per endpoint instead of constructing one per worker/request
    // (a fresh client leaks a runtime-lived monitor task — see RlDiscoveryState::client_for).
    let client = state
        .client_for(&target.namespace, &target.component, &target.endpoint)
        .await?;

    // Bound the readiness wait by the configured request timeout (capped at 5s so a large
    // request_timeout doesn't block discovery on a single slow-to-register worker). The
    // caller (describe_worker) also enforces the overall request_timeout deadline.
    let readiness_timeout = timeout.min(Duration::from_secs(5));
    wait_for_client_targets(&client, &[target.instance_id], readiness_timeout).await?;

    let router = PushRouter::<serde_json::Value, Annotated<serde_json::Value>>::from_client(
        client,
        RouterMode::Direct,
    )
    .await?;

    let request_value = serde_json::json!({
        "method": "routes",
    });
    let instance_id = target.instance_id;

    let request = SingleIn::new(request_value);
    let mut stream = router.direct(request, instance_id).await?;

    while let Some(chunk) = stream.next().await {
        if let Some(data) = chunk.data {
            return parse_worker_routes(data);
        }
        if let Some(err) = chunk.error {
            anyhow::bail!(err.to_string());
        }
    }

    anyhow::bail!("empty routes response from worker")
}

async fn wait_for_client_targets(
    client: &Client,
    target_ids: &[u64],
    timeout: Duration,
) -> anyhow::Result<()> {
    let wait = async {
        loop {
            let ids = client.instance_ids();
            if target_ids.iter().all(|id| ids.contains(id)) {
                break;
            }
            tokio::time::sleep(Duration::from_millis(25)).await;
        }
    };
    tokio::time::timeout(timeout, wait).await.map_err(|_| {
        anyhow::anyhow!(
            "timed out after {}s waiting for worker instance(s) to become discoverable",
            timeout.as_secs()
        )
    })
}

fn parse_worker_routes(value: serde_json::Value) -> anyhow::Result<WorkerRoutes> {
    if value
        .get("status")
        .and_then(|status| status.as_str())
        .is_some_and(|status| status == "error")
    {
        anyhow::bail!(
            "{}",
            value
                .get("message")
                .and_then(|message| message.as_str())
                .unwrap_or("worker routes request failed")
        );
    }

    let routes_array = value
        .get("routes")
        .and_then(|routes| routes.as_array())
        .ok_or_else(|| anyhow::anyhow!("worker routes response missing 'routes' array"))?;
    let mut routes = Vec::with_capacity(routes_array.len());
    for route in routes_array {
        // Surface a protocol error for malformed entries instead of silently dropping
        // them (which would report a truncated capability list as success).
        let name = route.as_str().ok_or_else(|| {
            anyhow::anyhow!("worker routes response contains a non-string route entry")
        })?;
        if name.is_empty() {
            anyhow::bail!("worker routes response contains an empty route entry");
        }
        routes.push(name.to_string());
    }

    let system_url = value
        .get("system_url")
        .and_then(|url| url.as_str())
        .map(str::trim)
        .filter(|url| !url.is_empty())
        .map(ToString::to_string);

    let admin_base_url = value
        .get("admin_base_url")
        .and_then(|url| url.as_str())
        .map(str::trim)
        .filter(|url| !url.is_empty())
        .map(ToString::to_string);
    let world_size = match value.get("world_size") {
        None | Some(serde_json::Value::Null) => None,
        Some(value) => Some(
            value
                .as_u64()
                .and_then(|size| u32::try_from(size).ok())
                .filter(|size| *size > 0)
                .ok_or_else(|| {
                    anyhow::anyhow!("worker routes response has invalid 'world_size'")
                })?,
        ),
    };

    Ok(WorkerRoutes {
        routes,
        system_url,
        admin_base_url,
        world_size,
    })
}

fn worker_info(
    endpoint: Instance,
    model: Option<String>,
    mut system_routes: Vec<String>,
    system_url: Option<String>,
    admin_base_url: Option<String>,
    world_size: Option<u32>,
    error: Option<String>,
) -> RlWorkerInfo {
    system_routes.sort();
    system_routes.dedup();

    RlWorkerInfo {
        request_plane_url: request_plane_url(&endpoint),
        namespace: endpoint.namespace,
        component: endpoint.component,
        endpoint: endpoint.endpoint,
        instance_id: endpoint.instance_id,
        transport: endpoint.transport,
        system_url,
        admin_base_url,
        world_size,
        model,
        system_routes,
        error,
    }
}

fn request_plane_url(endpoint: &Instance) -> String {
    format!(
        "dyn://{}.{}.{}",
        endpoint.namespace, endpoint.component, endpoint.endpoint
    )
}

fn model_map(instances: Vec<DiscoveryInstance>) -> HashMap<ModelKey, String> {
    // A worker registers its model under its serving endpoint (e.g. "generate"), not the
    // "rl" endpoint, so association is by (namespace, component, instance_id) and intentionally
    // ignores the model's own endpoint. If one instance advertises multiple distinct base
    // models we cannot pick one safely, so omit it rather than report an arbitrary/wrong model.
    let mut by_key: HashMap<ModelKey, std::collections::BTreeSet<String>> = HashMap::new();
    for instance in instances {
        if let DiscoveryInstance::Model {
            namespace,
            component,
            endpoint: _,
            instance_id,
            card_json,
            model_suffix,
        } = instance
            && model_suffix.as_ref().is_none_or(|suffix| suffix.is_empty())
            && let Some(name) = card_json
                .get("display_name")
                .and_then(|value| value.as_str())
        {
            by_key
                .entry((namespace, component, instance_id))
                .or_default()
                .insert(name.to_string());
        }
    }
    by_key
        .into_iter()
        .filter_map(|(key, names)| match names.len() {
            1 => names.into_iter().next().map(|name| (key, name)),
            _ => None,
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn worker_response_names_dynamo_capabilities_system_routes() {
        let worker = RlWorkerInfo {
            namespace: "dynamo".to_string(),
            component: "backend".to_string(),
            endpoint: "rl".to_string(),
            instance_id: 1,
            transport: TransportType::Tcp("127.0.0.1:1".to_string()),
            request_plane_url: "dyn://dynamo.backend.rl".to_string(),
            system_url: Some("http://worker:8181".to_string()),
            admin_base_url: Some("http://worker:8120".to_string()),
            world_size: Some(2),
            model: Some("model".to_string()),
            system_routes: vec!["update/load_lora".to_string()],
            error: None,
        };

        let value = serde_json::to_value(worker).unwrap();
        assert_eq!(value["system_routes"], json!(["update/load_lora"]));
        assert!(value.get("routes").is_none());
    }

    #[test]
    fn workers_response_advertises_protocol_version() {
        let response = RlWorkersResponse {
            protocol_version: RL_WORKERS_PROTOCOL_VERSION,
            namespace: "dynamo".to_string(),
            workers: Vec::new(),
        };
        let value = serde_json::to_value(response).unwrap();
        assert_eq!(value["protocol_version"], json!(1));
    }

    fn model_instance(
        namespace: &str,
        component: &str,
        endpoint: &str,
        instance_id: u64,
        display_name: &str,
        model_suffix: Option<&str>,
    ) -> DiscoveryInstance {
        DiscoveryInstance::Model {
            namespace: namespace.to_string(),
            component: component.to_string(),
            endpoint: endpoint.to_string(),
            instance_id,
            card_json: json!({ "display_name": display_name }),
            model_suffix: model_suffix.map(ToString::to_string),
        }
    }

    #[test]
    fn parse_worker_routes_accepts_valid_payload() {
        let parsed = parse_worker_routes(json!({
            "routes": ["pause_generation", "resume_generation"],
            "system_url": "  http://worker:8080  ",
            "admin_base_url": "  http://worker:8120  ",
            "world_size": 2,
        }))
        .expect("valid payload");
        let routes: Vec<&str> = parsed.routes.iter().map(String::as_str).collect();
        assert_eq!(routes, ["pause_generation", "resume_generation"]);
        // system_url is trimmed.
        assert_eq!(parsed.system_url.as_deref(), Some("http://worker:8080"));
        assert_eq!(parsed.admin_base_url.as_deref(), Some("http://worker:8120"));
        assert_eq!(parsed.world_size, Some(2));
    }

    #[test]
    fn parse_worker_routes_rejects_invalid_world_size() {
        for world_size in [json!(0), json!(-1), json!("2")] {
            let err = parse_worker_routes(json!({
                "routes": [],
                "admin_base_url": "http://worker:8120",
                "world_size": world_size,
            }))
            .unwrap_err();
            assert!(err.to_string().contains("world_size"));
        }
    }

    #[test]
    fn parse_worker_routes_blank_system_url_is_none() {
        let parsed = parse_worker_routes(json!({ "routes": [], "system_url": "   " }))
            .expect("valid payload");
        assert!(parsed.routes.is_empty());
        assert!(parsed.system_url.is_none());
    }

    #[test]
    fn parse_worker_routes_requires_routes_array() {
        let err = parse_worker_routes(json!({ "system_url": "http://x" })).unwrap_err();
        assert!(err.to_string().contains("missing 'routes' array"));
    }

    #[test]
    fn parse_worker_routes_rejects_non_string_entry() {
        // Must surface a protocol error, not silently drop the bad entry (finding F4).
        let err = parse_worker_routes(json!({ "routes": ["pause", 7] })).unwrap_err();
        assert!(err.to_string().contains("non-string route entry"));
    }

    #[test]
    fn parse_worker_routes_rejects_empty_entry() {
        let err = parse_worker_routes(json!({ "routes": ["pause", ""] })).unwrap_err();
        assert!(err.to_string().contains("empty route entry"));
    }

    #[test]
    fn parse_worker_routes_propagates_worker_error_status() {
        let err = parse_worker_routes(json!({ "status": "error", "message": "engine is dead" }))
            .unwrap_err();
        assert!(err.to_string().contains("engine is dead"));
    }

    #[test]
    fn model_map_associates_single_model_ignoring_endpoint() {
        let map = model_map(vec![model_instance(
            "dynamo",
            "backend",
            "generate",
            1,
            "Qwen/Qwen3-0.6B",
            None,
        )]);
        assert_eq!(
            map.get(&("dynamo".to_string(), "backend".to_string(), 1u64))
                .map(String::as_str),
            Some("Qwen/Qwen3-0.6B")
        );
    }

    #[test]
    fn model_map_omits_instance_with_conflicting_models() {
        // One instance advertising two distinct base models must be omitted, not
        // arbitrarily resolved to one of them (finding F5).
        let map = model_map(vec![
            model_instance("dynamo", "backend", "generate", 1, "Qwen/Qwen3-0.6B", None),
            model_instance(
                "dynamo",
                "backend",
                "embed",
                1,
                "Qwen/Qwen3-Embedding-4B",
                None,
            ),
        ]);
        assert!(!map.contains_key(&("dynamo".to_string(), "backend".to_string(), 1u64)));
    }

    #[test]
    fn model_map_dedupes_identical_model_across_endpoints() {
        let map = model_map(vec![
            model_instance("dynamo", "backend", "generate", 1, "Qwen/Qwen3-0.6B", None),
            model_instance("dynamo", "backend", "rl", 1, "Qwen/Qwen3-0.6B", None),
        ]);
        assert_eq!(
            map.get(&("dynamo".to_string(), "backend".to_string(), 1u64))
                .map(String::as_str),
            Some("Qwen/Qwen3-0.6B")
        );
    }

    #[test]
    fn model_map_skips_lora_suffix_entries() {
        let map = model_map(vec![model_instance(
            "dynamo",
            "backend",
            "generate",
            1,
            "adapter",
            Some("lora-1"),
        )]);
        assert!(map.is_empty());
    }
}
