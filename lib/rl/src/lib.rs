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
//! available RL admin routes with `{"method": "routes"}` over the request
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
    namespace::{GLOBAL_NAMESPACE, NamespaceFilter},
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
const RL_WORKERS_PROTOCOL_VERSION: u32 = 1;

/// Validated base URL for worker-specific RL HTTP administration routes.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct RlAdminBaseUrl(String);

impl RlAdminBaseUrl {
    pub fn parse(raw: &str) -> anyhow::Result<Self> {
        let value = raw.trim();
        if value.is_empty() {
            anyhow::bail!("RL admin base URL must not be blank");
        }
        let parsed = url::Url::parse(value)
            .map_err(|error| anyhow::anyhow!("invalid RL admin base URL: {error}"))?;
        let has_valid_authority = value
            .split_once("://")
            .is_some_and(|(_, authority)| !authority.is_empty() && !authority.starts_with('/'));
        if !matches!(parsed.scheme(), "http" | "https")
            || parsed.host_str().is_none()
            || !has_valid_authority
        {
            anyhow::bail!("RL admin base URL must use HTTP or HTTPS and include a host");
        }
        if !parsed.username().is_empty() || parsed.password().is_some() {
            anyhow::bail!("RL admin base URL must not include user information");
        }
        if parsed.query().is_some() {
            anyhow::bail!("RL admin base URL must not include a query string");
        }
        if parsed.fragment().is_some() {
            anyhow::bail!("RL admin base URL must not include a fragment");
        }
        Ok(Self(value.to_string()))
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }

    pub fn into_string(self) -> String {
        self.0
    }
}

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

/// Resolve the namespace scope for RL discovery from the three namespace inputs.
///
/// This is deliberately pure — it reads no environment — so the composition rule can be
/// tested without mutating process-global state.
///
/// Precedence, highest first:
///
/// 1. `namespace_prefix` (`DYN_NAMESPACE_PREFIX`): match every namespace under that
///    prefix. Kubernetes deployments get this on the frontend container automatically,
///    and it is what lets one listener see several worker generations during a rolling
///    update.
/// 2. `worker_suffix` (`DYN_NAMESPACE_WORKER_SUFFIX`): match exactly `{base}-{suffix}`.
/// 3. Neither: match `base` exactly.
///
/// `base` is `namespace` (`DYN_NAMESPACE`) whenever that variable is set, and
/// [`DEFAULT_NAMESPACE`] only when it is unset. An explicitly empty value is kept, so a
/// deployment that sets `DYN_NAMESPACE=` is searched under the same empty base its
/// workers register with.
///
/// Both rules mirror `get_worker_namespace` in
/// `components/src/dynamo/common/utils/namespace.py`, which is how workers pick the
/// namespace they register under. Its `os.environ.get("DYN_NAMESPACE", "dynamo")` falls
/// back only for an unset variable, and its `if suffix:` skips an empty suffix; the
/// composition is a single ASCII hyphen, no trimming, no case folding.
///
/// The prefix has no counterpart in that helper. An empty one would match every
/// namespace, so it counts as absent rather than as a scope over everything.
///
/// The no-prefix, no-suffix case stays [`NamespaceFilter::Exact`] rather than going
/// through `NamespaceFilter::from_namespace_and_prefix`, because that constructor maps
/// the literal `dynamo` to `NamespaceFilter::Global`. Routing the default through it
/// would silently widen RL discovery from one namespace to all of them for everyone who
/// leaves `DYN_NAMESPACE` unset.
pub fn resolve_namespace_filter(
    namespace: Option<&str>,
    namespace_prefix: Option<&str>,
    worker_suffix: Option<&str>,
) -> NamespaceFilter {
    fn present(value: Option<&str>) -> Option<&str> {
        value.filter(|value| !value.is_empty())
    }

    if let Some(prefix) = present(namespace_prefix) {
        return NamespaceFilter::Prefix(prefix.to_string());
    }

    let base = namespace.unwrap_or(DEFAULT_NAMESPACE);
    match present(worker_suffix) {
        Some(suffix) => NamespaceFilter::Exact(format!("{base}-{suffix}")),
        None => NamespaceFilter::Exact(base.to_string()),
    }
}

/// The scope reported back to the caller in [`RlWorkersResponse::namespace`].
///
/// Protocol version 1 types that field as a plain string, so a prefix scope reports the
/// prefix itself. `Global` has no string form of its own and reports `GLOBAL_NAMESPACE`;
/// [`resolve_namespace_filter`] never produces it, so this arm only covers a config built
/// directly by a caller.
fn namespace_scope(filter: &NamespaceFilter) -> &str {
    match filter {
        NamespaceFilter::Global => GLOBAL_NAMESPACE,
        NamespaceFilter::Exact(namespace) => namespace,
        NamespaceFilter::Prefix(prefix) => prefix,
    }
}

/// Whether `namespace` is inside `filter` for the purposes of RL discovery.
///
/// [`NamespaceFilter::Prefix`] matches on a bare `starts_with`, which also admits a
/// sibling deployment whose name merely begins with the prefix: under
/// `DYN_NAMESPACE_PREFIX=myns-dgd` it would take in `myns-dgd2`. The endpoints reached
/// here are RL control endpoints, so the scope is the prefix itself plus the
/// hyphen-delimited worker generations beneath it — the same shape
/// `DYN_NAMESPACE_WORKER_SUFFIX` produces.
fn namespace_in_scope(filter: &NamespaceFilter, namespace: &str) -> bool {
    match filter {
        NamespaceFilter::Prefix(prefix) => namespace
            .strip_prefix(prefix.as_str())
            .is_some_and(|rest| rest.is_empty() || rest.starts_with('-')),
        filter => filter.matches(namespace),
    }
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
    pub model: Option<String>,
    pub routes: Vec<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub world_size: Option<u32>,
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
    namespace_filter: NamespaceFilter,
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
        let namespace_filter = NamespaceFilter::Exact(config.namespace.clone());
        Self::new_with_namespace_filter(config, namespace_filter)
    }

    /// Constructs the listener state using namespace-scope environment variables.
    ///
    /// This is deliberately separate from [`Self::new`] so callers that supply an
    /// explicit configuration retain an exact namespace scope.
    pub fn new_from_env(config: RlDiscoveryConfig) -> Self {
        let namespace_filter = resolve_namespace_filter(
            Some(&config.namespace),
            std::env::var("DYN_NAMESPACE_PREFIX").ok().as_deref(),
            std::env::var("DYN_NAMESPACE_WORKER_SUFFIX").ok().as_deref(),
        );
        Self::new_with_namespace_filter(config, namespace_filter)
    }

    fn new_with_namespace_filter(
        config: RlDiscoveryConfig,
        namespace_filter: NamespaceFilter,
    ) -> Self {
        let permits = config.max_concurrent_probes.max(1);
        Self {
            config: Arc::new(config),
            namespace_filter,
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
            namespace: namespace_scope(&state.namespace_filter).to_string(),
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
    // `DiscoveryQuery` has no prefix-scoped variant, so a prefix or global scope lists
    // everything and filters here; an exact scope keeps its narrow query unchanged.
    let (endpoint_query, model_query) = match &state.namespace_filter {
        NamespaceFilter::Exact(namespace) => (
            DiscoveryQuery::NamespacedEndpoints {
                namespace: namespace.clone(),
            },
            DiscoveryQuery::NamespacedModels {
                namespace: namespace.clone(),
            },
        ),
        NamespaceFilter::Prefix(_) | NamespaceFilter::Global => {
            (DiscoveryQuery::AllEndpoints, DiscoveryQuery::AllModels)
        }
    };

    let endpoint_instances = config.runtime.discovery().list(endpoint_query).await?;

    let model_instances = config
        .runtime
        .discovery()
        .list(model_query)
        .await
        .unwrap_or_default()
        .into_iter()
        .filter(|instance| match instance {
            DiscoveryInstance::Model { namespace, .. } => {
                namespace_in_scope(&state.namespace_filter, namespace)
            }
            _ => true,
        })
        .collect();

    let models = model_map(model_instances);
    let rl_endpoints = endpoint_instances
        .into_iter()
        .filter_map(|instance| match instance {
            DiscoveryInstance::Endpoint(endpoint) => Some(endpoint),
            _ => None,
        })
        .filter(|endpoint| namespace_in_scope(&state.namespace_filter, &endpoint.namespace))
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
        Ok(Ok(routes)) => worker_info(endpoint, model, routes, None),
        Ok(Err(err)) => worker_info(
            endpoint,
            model,
            WorkerRoutes::default(),
            Some(err.to_string()),
        ),
        Err(_) => worker_info(
            endpoint,
            model,
            WorkerRoutes::default(),
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
        .map(|value| {
            let raw = value.as_str().ok_or_else(|| {
                anyhow::anyhow!("worker routes response has invalid 'admin_base_url'")
            })?;
            RlAdminBaseUrl::parse(raw)
                .map(RlAdminBaseUrl::into_string)
                .map_err(|error| {
                    anyhow::anyhow!("worker routes response has invalid 'admin_base_url': {error}")
                })
        })
        .transpose()?;

    let world_size = value
        .get("world_size")
        .map(|value| {
            let value = value.as_u64().ok_or_else(|| {
                anyhow::anyhow!("worker routes response has invalid 'world_size'")
            })?;
            u32::try_from(value)
                .ok()
                .filter(|value| *value > 0)
                .ok_or_else(|| anyhow::anyhow!("worker routes response has invalid 'world_size'"))
        })
        .transpose()?;
    if admin_base_url.is_some() && world_size.is_none() {
        anyhow::bail!("worker routes response has 'admin_base_url' without valid 'world_size'");
    }
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
    mut discovered: WorkerRoutes,
    error: Option<String>,
) -> RlWorkerInfo {
    discovered.routes.sort();
    discovered.routes.dedup();

    RlWorkerInfo {
        request_plane_url: request_plane_url(&endpoint),
        namespace: endpoint.namespace,
        component: endpoint.component,
        endpoint: endpoint.endpoint,
        instance_id: endpoint.instance_id,
        transport: endpoint.transport,
        system_url: discovered.system_url,
        admin_base_url: discovered.admin_base_url,
        model,
        routes: discovered.routes,
        world_size: discovered.world_size,
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
    use dynamo_runtime::{
        component::StartedEndpoint,
        discovery::DiscoverySpec,
        pipeline::{
            AsyncEngine, AsyncEngineContextProvider, ManyOut, ResponseStream, async_trait,
            network::Ingress,
        },
    };
    use futures::stream;
    use serde_json::json;

    struct TestRoutesHandler;

    #[async_trait]
    impl
        AsyncEngine<
            SingleIn<serde_json::Value>,
            ManyOut<Annotated<serde_json::Value>>,
            anyhow::Error,
        > for TestRoutesHandler
    {
        async fn generate(
            &self,
            input: SingleIn<serde_json::Value>,
        ) -> anyhow::Result<ManyOut<Annotated<serde_json::Value>>> {
            let (_, context) = input.into_parts();
            Ok(ResponseStream::new(
                Box::pin(stream::once(async {
                    Annotated::from_data(json!({"status": "ok", "routes": []}))
                })),
                context.context(),
            ))
        }
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
            "world_size": 4,
            "weight_transfer_backend": "nccl",
        }))
        .expect("valid payload");
        let routes: Vec<&str> = parsed.routes.iter().map(String::as_str).collect();
        assert_eq!(routes, ["pause_generation", "resume_generation"]);
        // system_url is trimmed.
        assert_eq!(parsed.system_url.as_deref(), Some("http://worker:8080"));
        assert_eq!(parsed.admin_base_url.as_deref(), Some("http://worker:8120"));
        assert_eq!(parsed.world_size, Some(4));
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
    fn parse_worker_routes_rejects_invalid_rl_metadata() {
        let zero = parse_worker_routes(json!({ "routes": [], "world_size": 0 })).unwrap_err();
        assert!(zero.to_string().contains("world_size"));
    }

    #[test]
    fn parse_worker_routes_rejects_invalid_admin_base_url() {
        for value in [
            json!("   "),
            json!(42),
            json!("https://user:token@worker.example.com/admin"),
            json!("https://worker.example.com/admin?token=secret"),
            json!("https://worker.example.com/admin#fragment"),
        ] {
            let err = parse_worker_routes(json!({
                "routes": [],
                "admin_base_url": value,
            }))
            .unwrap_err();
            assert!(err.to_string().contains("admin_base_url"));
        }
    }

    #[test]
    fn parse_worker_routes_requires_world_size_with_admin_base_url() {
        let err = parse_worker_routes(json!({
            "routes": [],
            "admin_base_url": "http://worker:8120",
        }))
        .unwrap_err();
        assert!(err.to_string().contains("world_size"));

        let parsed = parse_worker_routes(json!({ "routes": [], "world_size": 1 }))
            .expect("world size does not require an admin URL");
        assert_eq!(parsed.world_size, Some(1));
        assert!(parsed.admin_base_url.is_none());
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

    /// A `DistributedRuntime` backed by a private in-memory discovery store, so every test
    /// that calls this sees only the endpoints it registers itself.
    async fn test_runtime() -> Arc<DistributedRuntime> {
        let runtime = dynamo_runtime::Runtime::from_current().expect("test runtime");
        Arc::new(
            DistributedRuntime::new(
                runtime,
                dynamo_runtime::distributed::DistributedConfig::process_local(),
            )
            .await
            .expect("distributed runtime"),
        )
    }

    /// Register a live `rl` endpoint under `namespace`, exactly as a worker started with
    /// `DYN_ENABLE_RL` does. The caller must `shutdown()` the returned handle.
    async fn start_rl_endpoint(
        distributed: &Arc<DistributedRuntime>,
        namespace: &str,
    ) -> StartedEndpoint {
        let ingress = Ingress::for_engine(Arc::new(TestRoutesHandler)).expect("test ingress");
        distributed
            .namespace(namespace)
            .expect("namespace")
            .component("backend")
            .expect("component")
            .endpoint("rl")
            .endpoint_builder()
            .handler(ingress)
            .start_with_registration()
            .await
            .expect("RL endpoint")
    }

    fn discovery_state(
        distributed: &Arc<DistributedRuntime>,
        namespace_filter: NamespaceFilter,
    ) -> RlDiscoveryState {
        RlDiscoveryState::new_with_namespace_filter(
            RlDiscoveryConfig {
                runtime: distributed.clone(),
                namespace: "ns".to_string(),
                rl_endpoint: "rl".to_string(),
                component_filter: None,
                request_timeout: Duration::from_secs(1),
                max_concurrent_probes: 1,
            },
            namespace_filter,
        )
    }

    #[tokio::test]
    async fn list_workers_prefix_scope_excludes_other_namespaces() {
        let distributed = test_runtime().await;
        let matching = start_rl_endpoint(&distributed, "ns-abc123").await;
        let other = start_rl_endpoint(&distributed, "other-ns").await;
        let state = discovery_state(&distributed, NamespaceFilter::Prefix("ns".to_string()));

        let workers = list_workers(&state).await.expect("list");
        let namespaces: Vec<&str> = workers.iter().map(|w| w.namespace.as_str()).collect();
        assert_eq!(namespaces, ["ns-abc123"]);

        matching.shutdown().await.expect("endpoint shutdown");
        other.shutdown().await.expect("endpoint shutdown");
    }

    #[tokio::test]
    async fn list_workers_exact_scope_excludes_suffixed_namespace() {
        let distributed = test_runtime().await;
        let started = start_rl_endpoint(&distributed, "ns-abc123").await;
        let state = discovery_state(&distributed, NamespaceFilter::Exact("ns".to_string()));

        let workers = list_workers(&state).await.expect("list");
        assert!(workers.is_empty(), "unexpected workers: {workers:?}");

        started.shutdown().await.expect("endpoint shutdown");
    }

    #[tokio::test]
    #[serial_test::serial]
    async fn from_env_discovers_worker_in_suffix_namespace() {
        temp_env::async_with_vars(
            [
                ("DYN_NAMESPACE", Some("ns")),
                ("DYN_NAMESPACE_PREFIX", None::<&str>),
                ("DYN_NAMESPACE_WORKER_SUFFIX", Some("abc123")),
            ],
            async {
                let distributed = test_runtime().await;
                let started = start_rl_endpoint(&distributed, "ns-abc123").await;
                let state = RlDiscoveryState::new_from_env(RlDiscoveryConfig::from_env(
                    distributed.clone(),
                ));

                let workers = list_workers(&state).await.expect("list");
                let namespaces: Vec<&str> = workers
                    .iter()
                    .map(|worker| worker.namespace.as_str())
                    .collect();
                assert_eq!(namespaces, ["ns-abc123"]);

                started.shutdown().await.expect("endpoint shutdown");
            },
        )
        .await;
    }

    #[tokio::test]
    #[serial_test::serial]
    async fn explicit_config_ignores_environment_namespace_scope() {
        temp_env::async_with_vars(
            [
                ("DYN_NAMESPACE_PREFIX", Some("other")),
                ("DYN_NAMESPACE_WORKER_SUFFIX", Some("abc123")),
            ],
            async {
                let distributed = test_runtime().await;
                let state = RlDiscoveryState::new(RlDiscoveryConfig {
                    runtime: distributed,
                    namespace: "ns".to_string(),
                    rl_endpoint: "rl".to_string(),
                    component_filter: None,
                    request_timeout: Duration::from_secs(1),
                    max_concurrent_probes: 1,
                });

                assert_eq!(namespace_scope(&state.namespace_filter), "ns");
            },
        )
        .await;
    }

    #[test]
    fn prefix_scope_stops_at_a_hyphen() {
        let filter = NamespaceFilter::Prefix("myns-dgd".to_string());

        assert!(namespace_in_scope(&filter, "myns-dgd"));
        assert!(namespace_in_scope(&filter, "myns-dgd-abc123"));
        assert!(!namespace_in_scope(&filter, "myns-dgd2"));
        assert!(!namespace_in_scope(&filter, "myns"));
    }

    #[test]
    fn resolve_namespace_filter_precedence() {
        let cases = [
            (
                "prefix wins over a suffix that is also set",
                Some("ns"),
                Some("ns"),
                Some("abc123"),
                NamespaceFilter::Prefix("ns".to_string()),
            ),
            (
                "suffix composes the worker namespace",
                Some("ns"),
                None,
                Some("abc123"),
                NamespaceFilter::Exact("ns-abc123".to_string()),
            ),
            (
                "an empty suffix counts as absent",
                Some("ns"),
                None,
                Some(""),
                NamespaceFilter::Exact("ns".to_string()),
            ),
            (
                "nothing set falls back to the default namespace",
                None,
                None,
                None,
                NamespaceFilter::Exact(DEFAULT_NAMESPACE.to_string()),
            ),
            (
                "an explicitly empty namespace is kept, not defaulted",
                Some(""),
                None,
                None,
                NamespaceFilter::Exact(String::new()),
            ),
            (
                "an explicitly empty namespace still takes the suffix",
                Some(""),
                None,
                Some("abc123"),
                NamespaceFilter::Exact("-abc123".to_string()),
            ),
            (
                "an empty prefix counts as absent rather than matching everything",
                Some("ns"),
                Some(""),
                None,
                NamespaceFilter::Exact("ns".to_string()),
            ),
        ];

        for (description, namespace, prefix, suffix, expected) in cases {
            assert_eq!(
                resolve_namespace_filter(namespace, prefix, suffix),
                expected,
                "{description}"
            );
        }
    }

    #[tokio::test]
    async fn list_workers_keeps_endpoints_without_unambiguous_model_metadata() {
        let distributed = test_runtime().await;
        let started = start_rl_endpoint(&distributed, "dynamo").await;
        let state = discovery_state(&distributed, NamespaceFilter::Exact("dynamo".to_string()));

        let workers = list_workers(&state).await.expect("workers without models");
        assert_eq!(workers.len(), 1);
        assert!(workers[0].model.is_none());

        for (endpoint, display_name) in [("generate", "model-a"), ("embed", "model-b")] {
            distributed
                .discovery()
                .register(DiscoverySpec::Model {
                    namespace: "dynamo".to_string(),
                    component: "backend".to_string(),
                    endpoint: endpoint.to_string(),
                    card_json: json!({"display_name": display_name}),
                    model_suffix: None,
                })
                .await
                .expect("model registration");
        }

        let workers = list_workers(&state)
            .await
            .expect("workers with ambiguous models");
        assert_eq!(workers.len(), 1);
        assert!(workers[0].model.is_none());

        started.shutdown().await.expect("endpoint shutdown");
    }
}
