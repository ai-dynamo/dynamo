// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use super::{AsyncEngineContextProvider, ResponseStream};
use crate::error::{BackendError, DynamoError, ErrorType, match_error_chain};
use crate::{
    component::{
        Client, DeviceType, Endpoint, Instance, RoutingInstances, RoutingOccupancyState,
        get_or_create_routing_occupancy_state,
    },
    discovery::EndpointInstanceId,
    dynamo_nvtx_range,
    engine::{AsyncEngine, AsyncEngineContext, Data},
    metrics::frontend_perf::{STAGE_DURATION_SECONDS, STAGE_ROUTE},
    pipeline::{
        AddressedPushRouter, AddressedRequest, Error, ManyIn, ManyOut, SingleIn,
        error::{PipelineError, PipelineErrorExt},
    },
    protocols::{EndpointId, maybe_error::MaybeError},
    traits::DistributedRuntimeProvider,
};
use async_trait::async_trait;
use futures::Stream;
use rand::Rng;
use serde::{Deserialize, Serialize};
use std::{
    collections::{HashMap, HashSet},
    marker::PhantomData,
    pin::Pin,
    sync::{
        Arc,
        atomic::{AtomicU64, Ordering},
    },
    task::Poll,
    time::Instant,
};
use tokio_stream::StreamExt;
use tracing::Instrument;

/// Check if an error chain indicates the worker should be reported as down.
fn is_inhibited(err: &(dyn std::error::Error + 'static)) -> bool {
    const INHIBITED: &[ErrorType] = &[
        ErrorType::CannotConnect,
        ErrorType::Disconnected,
        ErrorType::ConnectionTimeout,
        ErrorType::ResponseTimeout,
        ErrorType::Backend(BackendError::EngineShutdown),
    ];
    match_error_chain(err, INHIBITED, &[])
}

/// Read the backend response inactivity timeout from the environment.
/// Reuses `DYN_HTTP_BACKEND_STREAM_TIMEOUT_SECS` — the same env var
/// as the HTTP-layer safety net in `disconnect.rs`.
fn response_inactivity_timeout() -> Option<std::time::Duration> {
    use crate::config::environment_names::llm::DYN_HTTP_BACKEND_STREAM_TIMEOUT_SECS;
    std::env::var(DYN_HTTP_BACKEND_STREAM_TIMEOUT_SECS)
        .ok()
        .and_then(|s| s.parse::<u64>().ok())
        .filter(|&secs| secs > 0)
        .map(std::time::Duration::from_secs)
}

/// RAII handle for one in-flight unit of work charged against
/// [`RoutingOccupancyState`]. The counter is incremented at construction; the
/// matching decrement is emitted on drop (or by [`Self::into_tracked_stream`]).
struct OccupancyPermit {
    state: Arc<RoutingOccupancyState>,
    instance_id: u64,
    armed: bool,
}

impl OccupancyPermit {
    fn new(state: Arc<RoutingOccupancyState>, instance_id: u64) -> Self {
        Self {
            state,
            instance_id,
            armed: true,
        }
    }

    fn retarget(&mut self, instance_id: u64) {
        if self.instance_id == instance_id {
            return;
        }
        self.state.increment(instance_id);
        self.state.decrement(self.instance_id);
        self.instance_id = instance_id;
    }

    fn into_tracked_stream<U: Data>(mut self, stream: ManyOut<U>) -> ManyOut<U> {
        self.armed = false;
        let engine_ctx = stream.context();
        ResponseStream::new(
            Box::pin(OccupancyTrackedStream {
                inner: stream,
                state: self.state.clone(),
                instance_id: self.instance_id,
                released: false,
            }),
            engine_ctx,
        )
    }
}

impl Drop for OccupancyPermit {
    fn drop(&mut self) {
        if self.armed {
            self.state.decrement(self.instance_id);
        }
    }
}

/// Trait for monitoring worker load and determining overload state.
/// Implementations can define custom load metrics and overload thresholds.
#[async_trait]
pub trait WorkerLoadMonitor: Send + Sync {
    /// Start background monitoring of worker load.
    /// This should spawn background tasks that update the client's overloaded instances.
    async fn start_monitoring(&self) -> anyhow::Result<()>;
}

/// Query interface for routing against multimodal embedding cache state.
pub trait MultimodalCacheIndex: Send + Sync {
    fn workers_with_cache_key_hits(&self, cache_keys: &[String]) -> Vec<(u64, usize)>;
    fn remove_worker(&self, worker_id: u64);
}

pub type MultimodalCacheKeyExtractor<T> = Arc<dyn Fn(&T) -> Vec<String> + Send + Sync>;

#[derive(Clone)]
pub struct PushRouter<T, U>
where
    T: Data + Serialize,
    U: Data + for<'de> Deserialize<'de>,
{
    // TODO: This shouldn't be pub, but lib/bindings/python/rust/lib.rs exposes it.
    /// The Client is how we gather remote endpoint information from etcd.
    pub client: Client,

    /// How we choose which instance to send traffic to.
    ///
    /// Setting this to KV means we never intend to call `generate` on this PushRouter. We are
    /// not using it as an AsyncEngine.
    /// Instead we will decide whether to call random/round_robin/direct ourselves and call them directly.
    /// dynamo-llm's KV Routing does this.
    router_mode: RouterMode,

    /// Number of round robin requests handled. Used to decide which server is next.
    round_robin_counter: Arc<AtomicU64>,

    /// The next step in the chain. PushRouter (this object) picks an instances,
    /// addresses it, then passes it to AddressedPushRouter which does the network traffic.
    addressed: Arc<AddressedPushRouter>,

    /// When false, `generate_with_fault_detection` skips fault detection logic:
    /// it won't call `report_instance_down` on errors, and it uses the raw discovery
    /// instance list instead of the filtered avail list. Use for recovery/query paths
    /// where transient failures are expected.
    fault_detection_enabled: bool,

    /// Cached response inactivity timeout. Read once at construction from
    /// [`environment_names::llm::DYN_HTTP_BACKEND_STREAM_TIMEOUT_SECS`](crate::config::environment_names::llm::DYN_HTTP_BACKEND_STREAM_TIMEOUT_SECS) to avoid a syscall per request.
    response_timeout: Option<std::time::Duration>,

    /// Shared request occupancy state for tracked routing modes.
    occupancy_state: Option<Arc<RoutingOccupancyState>>,

    /// Optional cache index for direct multimodal embedding cache lookups.
    /// Currently consumed by `RouterMode::DeviceAwareWeighted`.
    multimodal_cache_indexer: Option<Arc<dyn MultimodalCacheIndex>>,

    /// Optional typed request extractor for multimodal embedding cache keys.
    multimodal_cache_key_extractor: Option<MultimodalCacheKeyExtractor<T>>,

    /// An internal Rust type. This says that PushRouter is generic over the T and U types,
    /// which are the input and output types of it's `generate` function. It allows the
    /// compiler to specialize us at compile time.
    _phantom: PhantomData<(T, U)>,
}

#[derive(Default, Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RouterMode {
    #[default]
    RoundRobin,
    Random,
    PowerOfTwoChoices,
    KV,
    Direct,
    LeastLoaded,
    /// Device-aware weighted routing for heterogeneous workers.
    DeviceAwareWeighted,
}

#[derive(Clone, Copy)]
enum TransportFallback<'a> {
    Allow,
    Deny,
    Within(&'a HashSet<u64>),
}

struct DeviceAwareCandidates {
    candidates: Vec<u64>,
    device_type_map: HashMap<u64, Option<DeviceType>>,
    embedding_cache_hit: bool,
    full_embedding_cache_hit: bool,
    request_cache_keys: usize,
}

impl RouterMode {
    pub fn is_kv_routing(&self) -> bool {
        *self == RouterMode::KV
    }

    pub fn is_direct_routing(&self) -> bool {
        *self == RouterMode::Direct
    }
}

/// Pick the instance with lower in-flight count from two random candidates.
/// Returns the single instance if only one is available.
fn p2c_select_from(occupancy_state: &RoutingOccupancyState, instance_ids: &[u64]) -> u64 {
    let count = instance_ids.len();
    if count == 1 {
        let worker_id = instance_ids[0];
        tracing::info!(
            router_mode = "power-of-two-choices",
            worker_id,
            candidate_count = count,
            load = occupancy_state.load(worker_id),
            "Selected worker"
        );
        return worker_id;
    }
    let mut rng = rand::rng();
    let idx1 = rng.random_range(0..count);
    let idx2 = (idx1 + 1 + rng.random_range(0..count - 1)) % count;
    let id1 = instance_ids[idx1];
    let id2 = instance_ids[idx2];
    let load1 = occupancy_state.load(id1);
    let load2 = occupancy_state.load(id2);
    let selected = if load1 <= load2 { id1 } else { id2 };
    tracing::info!(
        router_mode = "power-of-two-choices",
        worker_id = selected,
        candidate_count = count,
        load = std::cmp::min(load1, load2),
        candidate_a = id1,
        candidate_a_load = load1,
        candidate_b = id2,
        candidate_b_load = load2,
        "Selected worker"
    );
    selected
}

/// Select the target device group for the next request in `DeviceAwareWeighted` mode.
///
/// If only one class exists (all CPU or all non-CPU), returns that class directly.
/// If both classes exist, compares capability-normalized load and returns the less-loaded group.
///
/// Budget check (integer form):
/// `allowed_cpu_inflight = total_non_cpu_inflight * cpu_count / (ratio * non_cpu_count)`
/// and choose CPU when `total_cpu_inflight < allowed_cpu_inflight`.
///
/// `ratio` is `non_cpu_to_cpu_ratio` (from `DYN_ENCODER_CUDA_TO_CPU_RATIO`,
/// default `8` in `device_aware_weighted`).
fn device_aware_candidate_group(
    state: &RoutingOccupancyState,
    instance_ids: &[u64],
    device_type_map: &HashMap<u64, Option<DeviceType>>,
    non_cpu_to_cpu_ratio: usize,
) -> Vec<u64> {
    let cpu_ids: Vec<u64> = instance_ids
        .iter()
        .copied()
        .filter(|id| matches!(device_type_map.get(id), Some(Some(DeviceType::Cpu))))
        .collect();
    let non_cpu_ids: Vec<u64> = instance_ids
        .iter()
        .copied()
        .filter(|id| !matches!(device_type_map.get(id), Some(Some(DeviceType::Cpu))))
        .collect();

    if cpu_ids.is_empty() {
        return non_cpu_ids;
    }
    if non_cpu_ids.is_empty() {
        return cpu_ids;
    }

    // Both classes exist: compute a budget for CPU in-flight requests.
    let total_non_cpu_inflight: u64 = non_cpu_ids.iter().map(|id| state.load(*id)).sum();
    let total_cpu_inflight: u64 = cpu_ids.iter().map(|id| state.load(*id)).sum();
    let cpu_count = cpu_ids.len() as u64;
    let non_cpu_count = non_cpu_ids.len() as u64;
    let allowed_cpu_inflight = total_non_cpu_inflight.saturating_mul(cpu_count)
        / ((non_cpu_to_cpu_ratio as u64).saturating_mul(non_cpu_count));

    if total_cpu_inflight < allowed_cpu_inflight {
        cpu_ids
    } else {
        non_cpu_ids
    }
}

/// At most one `list_and_watch` per endpoint, across all `PushRouter`
/// instances. Entry removed on watcher exit so a later router can re-arm.
static ENDPOINT_WATCHER_ACTIVE: std::sync::OnceLock<dashmap::DashMap<EndpointId, ()>> =
    std::sync::OnceLock::new();

/// At most one multimodal cache cleanup watcher per endpoint.
static ENDPOINT_CACHE_INDEXER_WATCHER_ACTIVE: std::sync::OnceLock<
    dashmap::DashMap<EndpointId, ()>,
> = std::sync::OnceLock::new();

/// Watch discovery for instance removals and cancel pending response-stream
/// registrations on the removed instance, unblocking queued requests with
/// a migratable `Disconnected` error. Uses raw `list_and_watch` events
/// (not a coalesced snapshot diff) so a rapid remove→re-add of the same
/// identity is not silently swallowed. Keyed by full `EndpointInstanceId`.
fn spawn_instance_removal_watcher(
    endpoint: Endpoint,
    addressed: Arc<AddressedPushRouter>,
    cancel_token: tokio_util::sync::CancellationToken,
) {
    use crate::discovery::{
        DiscoveryEvent, DiscoveryInstance, DiscoveryInstanceId, DiscoveryQuery,
    };
    use tokio_stream::StreamExt as _;

    // One watcher per endpoint: if one is already running, skip.
    let guard = ENDPOINT_WATCHER_ACTIVE.get_or_init(dashmap::DashMap::new);
    let endpoint_id = endpoint.id();
    if guard.insert(endpoint_id.clone(), ()).is_some() {
        tracing::debug!(
            ?endpoint_id,
            "Instance removal watcher already running for this endpoint, skipping"
        );
        return;
    }

    let endpoint_name = endpoint.name().to_string();

    tokio::spawn(async move {
        // Release on every exit path (including panic); a leaked entry
        // silently disables removal cancellation until process restart.
        struct GuardRelease(EndpointId);
        impl Drop for GuardRelease {
            fn drop(&mut self) {
                if let Some(map) = ENDPOINT_WATCHER_ACTIVE.get() {
                    map.remove(&self.0);
                }
            }
        }
        let _release = GuardRelease(endpoint_id);

        let namespace = endpoint.component().namespace().name();
        let component = endpoint.component().name().to_string();

        // Reconnect on transient discovery failure; cancel-aware backoff.
        const RECONNECT_BACKOFF: std::time::Duration = std::time::Duration::from_secs(5);
        'reconnect: loop {
            let query = DiscoveryQuery::Endpoint {
                namespace: namespace.clone(),
                component: component.clone(),
                endpoint: endpoint_name.clone(),
            };

            let mut stream = match endpoint.drt().discovery().list_and_watch(query, None).await {
                Ok(s) => s,
                Err(e) => {
                    tracing::warn!(
                        endpoint = %endpoint_name,
                        "Failed to start instance removal watcher (will retry): {e}"
                    );
                    tokio::select! {
                        _ = tokio::time::sleep(RECONNECT_BACKOFF) => continue 'reconnect,
                        _ = cancel_token.cancelled() => break 'reconnect,
                    }
                }
            };

            loop {
                tokio::select! {
                    event = stream.next() => {
                        match event {
                            Some(Ok(DiscoveryEvent::Removed(id))) => {
                                if let DiscoveryInstanceId::Endpoint(eid) = &id {
                                    let n = addressed.cancel_instance_streams(eid).await;
                                    if n > 0 {
                                        tracing::warn!(
                                            namespace = %eid.namespace,
                                            component = %eid.component,
                                            endpoint = %eid.endpoint,
                                            instance_id = eid.instance_id,
                                            cancelled = n,
                                            "Cancelled pending response streams for removed \
                                             instance (discovery-driven cleanup)"
                                        );
                                    }
                                }
                            }
                            Some(Ok(DiscoveryEvent::Added(DiscoveryInstance::Endpoint(inst)))) => {
                                let eid: EndpointInstanceId = inst.endpoint_instance_id();
                                addressed.clear_instance_tombstone(&eid).await;
                            }
                            Some(Ok(_)) => {}
                            Some(Err(e)) => {
                                tracing::warn!(
                                    endpoint = %endpoint_name,
                                    "Instance removal watcher stream error: {e}"
                                );
                            }
                            None => {
                                tracing::warn!(
                                    endpoint = %endpoint_name,
                                    "Instance removal watcher stream ended; reconnecting"
                                );
                                continue 'reconnect;
                            }
                        }
                    }
                    _ = cancel_token.cancelled() => {
                        break 'reconnect;
                    }
                }
            }
        }

        tracing::debug!(endpoint = %endpoint_name, "Instance removal watcher exiting");
    });
}

/// Watch discovery removals for cache-aware routers and drop stale worker cache entries.
fn spawn_multimodal_cache_cleanup_watcher(
    endpoint: Endpoint,
    indexer: Arc<dyn MultimodalCacheIndex>,
    cancel_token: tokio_util::sync::CancellationToken,
) {
    use crate::discovery::{DiscoveryEvent, DiscoveryInstanceId, DiscoveryQuery};
    use tokio_stream::StreamExt as _;

    let guard = ENDPOINT_CACHE_INDEXER_WATCHER_ACTIVE.get_or_init(dashmap::DashMap::new);
    let endpoint_id = endpoint.id();
    if guard.insert(endpoint_id.clone(), ()).is_some() {
        tracing::debug!(
            ?endpoint_id,
            "Multimodal cache cleanup watcher already running for this endpoint, skipping"
        );
        return;
    }

    let endpoint_name = endpoint.name().to_string();
    let namespace = endpoint.component().namespace().name();
    let component = endpoint.component().name().to_string();

    tokio::spawn(async move {
        struct GuardRelease(EndpointId);
        impl Drop for GuardRelease {
            fn drop(&mut self) {
                if let Some(map) = ENDPOINT_CACHE_INDEXER_WATCHER_ACTIVE.get() {
                    map.remove(&self.0);
                }
            }
        }
        let _release = GuardRelease(endpoint_id);

        const RECONNECT_BACKOFF: std::time::Duration = std::time::Duration::from_secs(5);
        'reconnect: loop {
            let query = DiscoveryQuery::Endpoint {
                namespace: namespace.clone(),
                component: component.clone(),
                endpoint: endpoint_name.clone(),
            };

            let mut stream = match endpoint.drt().discovery().list_and_watch(query, None).await {
                Ok(stream) => stream,
                Err(error) => {
                    tracing::warn!(
                        endpoint = %endpoint_name,
                        "Failed to start multimodal cache cleanup watcher (will retry): {error}"
                    );
                    tokio::select! {
                        _ = tokio::time::sleep(RECONNECT_BACKOFF) => continue 'reconnect,
                        _ = cancel_token.cancelled() => break 'reconnect,
                    }
                }
            };

            loop {
                tokio::select! {
                    event = stream.next() => {
                        match event {
                            Some(Ok(DiscoveryEvent::Removed(DiscoveryInstanceId::Endpoint(eid)))) => {
                                indexer.remove_worker(eid.instance_id);
                            }
                            Some(Ok(_)) => {}
                            Some(Err(error)) => {
                                tracing::warn!(
                                    endpoint = %endpoint_name,
                                    "Multimodal cache cleanup watcher stream error: {error}"
                                );
                                continue 'reconnect;
                            }
                            None => {
                                tracing::warn!(
                                    endpoint = %endpoint_name,
                                    "Multimodal cache cleanup watcher stream ended; reconnecting"
                                );
                                continue 'reconnect;
                            }
                        }
                    }
                    _ = cancel_token.cancelled() => break 'reconnect,
                }
            }
        }

        tracing::debug!(endpoint = %endpoint_name, "Multimodal cache cleanup watcher exiting");
    });
}

async fn addressed_router(endpoint: &Endpoint) -> anyhow::Result<Arc<AddressedPushRouter>> {
    AddressedPushRouter::from_runtime_provider(endpoint).await
}

impl<T, U> PushRouter<T, U>
where
    T: Data + Serialize,
    U: Data + for<'de> Deserialize<'de> + MaybeError,
{
    /// Create a new PushRouter without a worker load monitor (no overload detection)
    pub async fn from_client(client: Client, router_mode: RouterMode) -> anyhow::Result<Self> {
        Self::from_client_with_monitor(client, router_mode, None).await
    }

    /// Create a new PushRouter with fault detection disabled.
    ///
    /// Unlike `from_client`, this router will not call `report_instance_down` on
    /// transient errors, and `direct()` uses the raw discovery instance list instead
    /// of the filtered avail list. Use for recovery/query paths.
    pub async fn from_client_no_fault_detection(
        client: Client,
        router_mode: RouterMode,
    ) -> anyhow::Result<Self> {
        let addressed = addressed_router(&client.endpoint).await?;

        let occupancy_state = if matches!(
            router_mode,
            RouterMode::PowerOfTwoChoices
                | RouterMode::LeastLoaded
                | RouterMode::DeviceAwareWeighted
        ) {
            Some(get_or_create_routing_occupancy_state(&client.endpoint).await)
        } else {
            None
        };

        // Cancel orphaned pending response streams when workers die.
        spawn_instance_removal_watcher(
            client.endpoint.clone(),
            addressed.clone(),
            client.endpoint.drt().primary_token(),
        );

        Ok(PushRouter {
            client,
            addressed,
            router_mode,
            round_robin_counter: Arc::new(AtomicU64::new(0)),
            fault_detection_enabled: false,
            response_timeout: response_inactivity_timeout(),
            occupancy_state,
            multimodal_cache_indexer: None,
            multimodal_cache_key_extractor: None,
            _phantom: PhantomData,
        })
    }

    /// Create a new PushRouter with an optional worker load monitor.
    ///
    /// The rejection path is gated by `fault_detection_enabled` (true here);
    /// overload detection itself is driven by the monitor via `client.set_overloaded_instances(...)`.
    /// If no thresholds are configured on the monitor (or no monitor is provided),
    /// the routing snapshot reports at least one free instance and the gate never rejects.
    pub async fn from_client_with_monitor(
        client: Client,
        router_mode: RouterMode,
        worker_monitor: Option<Arc<dyn WorkerLoadMonitor>>,
    ) -> anyhow::Result<Self> {
        Self::from_client_with_state(client, router_mode, worker_monitor, None, None).await
    }

    /// Create a new PushRouter with optional load monitoring and multimodal cache indexing.
    pub async fn from_client_with_state(
        client: Client,
        router_mode: RouterMode,
        worker_monitor: Option<Arc<dyn WorkerLoadMonitor>>,
        multimodal_cache_indexer: Option<Arc<dyn MultimodalCacheIndex>>,
        multimodal_cache_key_extractor: Option<MultimodalCacheKeyExtractor<T>>,
    ) -> anyhow::Result<Self> {
        let addressed = addressed_router(&client.endpoint).await?;

        // Start worker monitor if provided and in dynamic mode
        if let Some(monitor) = worker_monitor.as_ref() {
            monitor.start_monitoring().await?;
        }

        let occupancy_state = if matches!(
            router_mode,
            RouterMode::PowerOfTwoChoices
                | RouterMode::LeastLoaded
                | RouterMode::DeviceAwareWeighted
        ) {
            Some(get_or_create_routing_occupancy_state(&client.endpoint).await)
        } else {
            None
        };

        // Cancel orphaned pending response streams when workers die.
        spawn_instance_removal_watcher(
            client.endpoint.clone(),
            addressed.clone(),
            client.endpoint.drt().primary_token(),
        );

        // Drop stale cache-index entries when workers leave discovery.
        if let Some(indexer) = multimodal_cache_indexer.clone() {
            spawn_multimodal_cache_cleanup_watcher(
                client.endpoint.clone(),
                indexer,
                client.endpoint.drt().primary_token(),
            );
        }

        let router = PushRouter {
            client,
            addressed,
            router_mode,
            round_robin_counter: Arc::new(AtomicU64::new(0)),
            fault_detection_enabled: true,
            response_timeout: response_inactivity_timeout(),
            occupancy_state,
            multimodal_cache_indexer,
            multimodal_cache_key_extractor,
            _phantom: PhantomData,
        };

        Ok(router)
    }

    /// `ResourceExhausted` when workers are routable but all overloaded;
    /// `Unavailable` when no routable workers exist.
    fn empty_free_pool_error(&self, routing_instances: &RoutingInstances) -> anyhow::Error {
        if !routing_instances.routable_ids().is_empty() {
            let cause = PipelineError::ServiceOverloaded(
                "All workers are busy, please retry later".to_string(),
            );
            return DynamoError::builder()
                .error_type(ErrorType::ResourceExhausted)
                .message("All workers are busy, please retry later")
                .cause(cause)
                .build()
                .into();
        }
        DynamoError::builder()
            .error_type(ErrorType::Unavailable)
            .message(format!(
                "No workers available for endpoint {}",
                self.client.endpoint.id()
            ))
            .build()
            .into()
    }

    /// Issue a request to the next available instance in a round-robin fashion
    pub async fn round_robin(&self, request: SingleIn<T>) -> anyhow::Result<ManyOut<U>> {
        self.round_robin_prepared(request, |_, _| Ok(()))
            .await
            .map(|(_, stream)| stream)
    }

    async fn round_robin_prepared<M, F>(
        &self,
        request: SingleIn<T>,
        prepare: F,
    ) -> anyhow::Result<(M, ManyOut<U>)>
    where
        F: FnOnce(&mut T, u64) -> anyhow::Result<M>,
    {
        let counter = self.round_robin_counter.fetch_add(1, Ordering::Relaxed) as usize;

        let (instance_id, candidate_count) = {
            let routing_instances = self.client.routing_instances();
            let count = routing_instances.free_ids().len();
            if count == 0 {
                return Err(self.empty_free_pool_error(&routing_instances));
            }
            (routing_instances.free_ids()[counter % count], count)
        };
        tracing::info!(
            router_mode = "round-robin",
            worker_id = instance_id,
            candidate_count,
            "Selected worker"
        );

        self.dispatch_selected(instance_id, request, None, prepare)
            .await
    }

    /// Issue a request to a random endpoint
    pub async fn random(&self, request: SingleIn<T>) -> anyhow::Result<ManyOut<U>> {
        self.random_prepared(request, |_, _| Ok(()))
            .await
            .map(|(_, stream)| stream)
    }

    async fn random_prepared<M, F>(
        &self,
        request: SingleIn<T>,
        prepare: F,
    ) -> anyhow::Result<(M, ManyOut<U>)>
    where
        F: FnOnce(&mut T, u64) -> anyhow::Result<M>,
    {
        let (instance_id, candidate_count) = {
            let routing_instances = self.client.routing_instances();
            let count = routing_instances.free_ids().len();
            if count == 0 {
                return Err(self.empty_free_pool_error(&routing_instances));
            }
            let counter = rand::rng().random::<u64>() as usize;
            (routing_instances.free_ids()[counter % count], count)
        };
        tracing::info!(
            router_mode = "random",
            worker_id = instance_id,
            candidate_count,
            "Selected worker"
        );

        self.dispatch_selected(instance_id, request, None, prepare)
            .await
    }

    /// Issue a request using power-of-two-choices: pick 2 random healthy workers,
    /// route to the one with fewer in-flight requests.
    pub async fn power_of_two_choices(&self, request: SingleIn<T>) -> anyhow::Result<ManyOut<U>> {
        self.power_of_two_choices_prepared(request, |_, _| Ok(()))
            .await
            .map(|(_, stream)| stream)
    }

    async fn power_of_two_choices_prepared<M, F>(
        &self,
        request: SingleIn<T>,
        prepare: F,
    ) -> anyhow::Result<(M, ManyOut<U>)>
    where
        F: FnOnce(&mut T, u64) -> anyhow::Result<M>,
    {
        let state = self.occupancy_state()?;
        let instance_id = {
            let routing_instances = self.client.routing_instances();
            if routing_instances.free_ids().is_empty() {
                return Err(self.empty_free_pool_error(&routing_instances));
            }
            p2c_select_from(state.as_ref(), routing_instances.free_ids())
        };
        state.increment(instance_id);
        let permit = OccupancyPermit::new(state, instance_id);
        self.dispatch_selected(instance_id, request, Some(permit), prepare)
            .await
    }

    /// Issue a request to a specific endpoint
    pub async fn direct(
        &self,
        request: SingleIn<T>,
        instance_id: u64,
    ) -> anyhow::Result<ManyOut<U>> {
        self.direct_within(request, instance_id, None).await
    }

    /// Like [`direct`], but if the selected instance disappears between selection and dispatch,
    /// the internal reselection is constrained to `allowed_fallback` (when `Some`). Callers that
    /// pre-narrowed the candidate set (e.g. LoRA replica-set filtering) pass that set so the
    /// vanished-instance fallback cannot escape it and route to an arbitrary worker.
    pub async fn direct_within(
        &self,
        request: SingleIn<T>,
        instance_id: u64,
        allowed_fallback: Option<&HashSet<u64>>,
    ) -> anyhow::Result<ManyOut<U>> {
        self.direct_within_prepared(request, instance_id, allowed_fallback, |_, _| Ok(()))
            .await
            .map(|(_, stream)| stream)
    }

    /// Like [`Self::direct_within`], but prepares the request after transport resolution and
    /// returns the preparation metadata alongside the response stream.
    pub async fn direct_within_prepared<M, F>(
        &self,
        request: SingleIn<T>,
        instance_id: u64,
        allowed_fallback: Option<&HashSet<u64>>,
        prepare: F,
    ) -> anyhow::Result<(M, ManyOut<U>)>
    where
        F: FnOnce(&mut T, u64) -> anyhow::Result<M>,
    {
        // When fault detection is disabled, check the raw discovery list
        // (not filtered by report_instance_down) so transient failures
        // don't poison the instance for subsequent retries.
        let found = {
            if self.fault_detection_enabled {
                let routing_instances = self.client.routing_instances();
                routing_instances.routable_ids().contains(&instance_id)
            } else {
                self.client.instance_ids().contains(&instance_id)
            }
        };

        if !found {
            return Err(DynamoError::builder()
                .error_type(ErrorType::CannotConnect)
                .message(format!(
                    "instance_id={instance_id} not found for endpoint {}",
                    self.client.endpoint.id()
                ))
                .build()
                .into());
        }

        tracing::info!(
            router_mode = "direct",
            worker_id = instance_id,
            "Selected worker"
        );

        let fallback = allowed_fallback
            .map(TransportFallback::Within)
            .unwrap_or(TransportFallback::Allow);
        self.generate_with_fault_detection_prepared(instance_id, request, fallback, prepare)
            .await
    }

    /// Dispatch to exactly one worker without transport fallback.
    ///
    /// The worker is revalidated against the latest discovery and overload
    /// state immediately before dispatch.
    pub async fn dispatch_exact(
        &self,
        request: SingleIn<T>,
        instance_id: u64,
    ) -> anyhow::Result<ManyOut<U>> {
        self.generate_with_fault_detection(instance_id, request, TransportFallback::Deny)
            .await
    }

    /// Select and book one worker, prepare the request for that exact worker,
    /// then dispatch without reselection or transport fallback.
    pub async fn select_and_dispatch_exact<M, F>(
        &self,
        mut request: SingleIn<T>,
        pinned_worker: Option<u64>,
        prepare: F,
    ) -> anyhow::Result<(M, ManyOut<U>)>
    where
        F: FnOnce(&mut T, u64) -> anyhow::Result<M>,
    {
        let (instance_id, permit) = self
            .select_exact_target(request.content(), pinned_worker)
            .await?;
        let metadata = prepare(&mut request, instance_id)?;
        let stream = self.dispatch_exact(request, instance_id).await?;
        let stream = match permit {
            Some(permit) => permit.into_tracked_stream(stream),
            None => stream,
        };
        Ok((metadata, stream))
    }

    /// Select a worker using the configured routing mode, prepare the request with the worker
    /// that survives transport resolution, then dispatch with normal fallback behavior.
    pub async fn select_and_dispatch<M, F>(
        &self,
        request: SingleIn<T>,
        prepare: F,
    ) -> anyhow::Result<(M, ManyOut<U>)>
    where
        F: FnOnce(&mut T, u64) -> anyhow::Result<M>,
    {
        match self.router_mode {
            RouterMode::Random => self.random_prepared(request, prepare).await,
            RouterMode::RoundRobin => self.round_robin_prepared(request, prepare).await,
            RouterMode::PowerOfTwoChoices => {
                self.power_of_two_choices_prepared(request, prepare).await
            }
            RouterMode::LeastLoaded => self.least_loaded_prepared(request, prepare).await,
            RouterMode::DeviceAwareWeighted => {
                self.device_aware_weighted_prepared(request, prepare).await
            }
            RouterMode::KV => anyhow::bail!("KV routing should not call select_and_dispatch"),
            RouterMode::Direct => anyhow::bail!(
                "Direct routing should use direct_within_prepared instead of select_and_dispatch"
            ),
        }
    }

    async fn dispatch_selected<M, F>(
        &self,
        instance_id: u64,
        request: SingleIn<T>,
        mut permit: Option<OccupancyPermit>,
        prepare: F,
    ) -> anyhow::Result<(M, ManyOut<U>)>
    where
        F: FnOnce(&mut T, u64) -> anyhow::Result<M>,
    {
        let (metadata, stream) = self
            .generate_with_fault_detection_prepared(
                instance_id,
                request,
                TransportFallback::Allow,
                |request, resolved_instance_id| {
                    if let Some(permit) = permit.as_mut() {
                        permit.retarget(resolved_instance_id);
                    }
                    prepare(request, resolved_instance_id)
                },
            )
            .await?;
        let stream = match permit {
            Some(permit) => permit.into_tracked_stream(stream),
            None => stream,
        };
        Ok((metadata, stream))
    }

    /// Issue a request using device-aware weighted routing.
    ///
    /// Instances are partitioned by device type (CPU vs non-CPU), then the router
    /// applies a budget policy and selects the least-loaded instance within the
    /// chosen group.
    ///
    /// If only one device class exists (all CPU or all non-CPU), this naturally
    /// degenerates to least-loaded routing over the available instances.
    pub async fn device_aware_weighted(&self, request: SingleIn<T>) -> anyhow::Result<ManyOut<U>> {
        self.device_aware_weighted_prepared(request, |_, _| Ok(()))
            .await
            .map(|(_, stream)| stream)
    }

    async fn device_aware_weighted_prepared<M, F>(
        &self,
        request: SingleIn<T>,
        prepare: F,
    ) -> anyhow::Result<(M, ManyOut<U>)>
    where
        F: FnOnce(&mut T, u64) -> anyhow::Result<M>,
    {
        let state = self.occupancy_state()?;
        let routing_instances = self.client.routing_instances();
        let instance_ids = routing_instances.free_ids().to_vec();

        if instance_ids.is_empty() {
            return Err(self.empty_free_pool_error(&routing_instances));
        }

        // Apply a unified policy for all endpoints.
        let endpoint_id = self.client.endpoint.id();

        let selection =
            self.device_aware_candidates(request.content(), state.as_ref(), &instance_ids);

        // Only full cache hits bypass weighted accounting; partial hits still follow the
        // device-aware ratio because some image encoding remains for this request.
        let instance_id = if selection.full_embedding_cache_hit {
            state.select_exact_min(&selection.candidates).await
        } else {
            state
                .select_exact_min_and_increment(&selection.candidates)
                .await
        }
        .ok_or_else(|| self.empty_free_pool_error(&routing_instances))?;
        let permit = if selection.full_embedding_cache_hit {
            None
        } else {
            Some(OccupancyPermit::new(state.clone(), instance_id))
        };
        let is_cpu = matches!(
            selection.device_type_map.get(&instance_id),
            Some(Some(DeviceType::Cpu))
        );
        tracing::info!(
            router_mode = "device-aware-weighted",
            worker_id = instance_id,
            candidate_count = selection.candidates.len(),
            load = state.load(instance_id),
            endpoint = %endpoint_id,
            is_cpu,
            embedding_cache_hit = selection.embedding_cache_hit,
            request_cache_keys = selection.request_cache_keys,
            "Selected worker"
        );

        self.dispatch_selected(instance_id, request, permit, prepare)
            .await
    }

    fn device_aware_candidates(
        &self,
        request: &T,
        state: &RoutingOccupancyState,
        instance_ids: &[u64],
    ) -> DeviceAwareCandidates {
        let device_type_map = self
            .client
            .instances()
            .iter()
            .map(|instance| (instance.instance_id, instance.device_type.clone()))
            .collect();
        let cuda_to_cpu_ratio = std::env::var("DYN_ENCODER_CUDA_TO_CPU_RATIO")
            .ok()
            .and_then(|value| value.parse::<usize>().ok())
            .filter(|value| *value >= 1)
            .unwrap_or(8);

        let (request_cache_keys, cache_matched_candidates) =
            if let (Some(indexer), Some(extractor)) = (
                self.multimodal_cache_indexer.as_ref(),
                self.multimodal_cache_key_extractor.as_ref(),
            ) {
                let request_cache_keys = extractor(request);
                let matched = if request_cache_keys.is_empty() {
                    Vec::new()
                } else {
                    let mut matched = indexer.workers_with_cache_key_hits(&request_cache_keys);
                    matched.retain(|(id, _)| instance_ids.contains(id));
                    matched
                };
                (request_cache_keys, matched)
            } else {
                (Vec::new(), Vec::new())
            };

        let embedding_cache_hit = !cache_matched_candidates.is_empty();
        let request_cache_key_count = request_cache_keys
            .iter()
            .collect::<std::collections::HashSet<_>>()
            .len();
        let full_cache_candidates = cache_matched_candidates
            .iter()
            .filter_map(|(worker_id, hits)| {
                (*hits >= request_cache_key_count).then_some(*worker_id)
            })
            .collect::<Vec<_>>();
        let full_embedding_cache_hit = !full_cache_candidates.is_empty();
        let candidates = if full_embedding_cache_hit {
            full_cache_candidates
        } else {
            device_aware_candidate_group(state, instance_ids, &device_type_map, cuda_to_cpu_ratio)
        };

        DeviceAwareCandidates {
            candidates,
            device_type_map,
            embedding_cache_hit,
            full_embedding_cache_hit,
            request_cache_keys: request_cache_keys.len(),
        }
    }

    /// Issue a request to the instance with the fewest active connections.
    pub async fn least_loaded(&self, request: SingleIn<T>) -> anyhow::Result<ManyOut<U>> {
        self.least_loaded_prepared(request, |_, _| Ok(()))
            .await
            .map(|(_, stream)| stream)
    }

    async fn least_loaded_prepared<M, F>(
        &self,
        request: SingleIn<T>,
        prepare: F,
    ) -> anyhow::Result<(M, ManyOut<U>)>
    where
        F: FnOnce(&mut T, u64) -> anyhow::Result<M>,
    {
        let state = self.occupancy_state()?;
        let routing_instances = self.client.routing_instances();
        let instance_ids = routing_instances.free_ids().to_vec();
        let instance_id = state
            .select_exact_min_and_increment(&instance_ids)
            .await
            .ok_or_else(|| self.empty_free_pool_error(&routing_instances))?;
        let permit = OccupancyPermit::new(state.clone(), instance_id);
        tracing::info!(
            router_mode = "least-loaded",
            worker_id = instance_id,
            candidate_count = instance_ids.len(),
            load = state.load(instance_id),
            "Selected worker"
        );

        self.dispatch_selected(instance_id, request, Some(permit), prepare)
            .await
    }

    /// Select the next worker according to the routing mode.
    /// Increments round-robin counter if applicable.
    /// Returns None for modes that require request lifecycle tracking or explicit routing hints.
    pub fn select_next_worker(&self) -> Option<u64> {
        let routing_instances = self.client.routing_instances();
        let count = routing_instances.free_ids().len();
        if count == 0 {
            return None;
        }

        match self.router_mode {
            RouterMode::RoundRobin => {
                let counter = self.round_robin_counter.fetch_add(1, Ordering::Relaxed) as usize;
                Some(routing_instances.free_ids()[counter % count])
            }
            RouterMode::Random => {
                let counter = rand::rng().random::<u64>() as usize;
                Some(routing_instances.free_ids()[counter % count])
            }
            RouterMode::PowerOfTwoChoices
            | RouterMode::Direct
            | RouterMode::LeastLoaded
            | RouterMode::DeviceAwareWeighted => None,
            RouterMode::KV => {
                panic!(
                    "select_next_worker should not be called for {:?} routing mode",
                    self.router_mode
                )
            }
        }
    }

    /// Peek the next worker according to the routing mode without incrementing the counter.
    /// Useful for checking if a worker is suitable before committing to it.
    ///
    /// `None` for [`RouterMode::Direct`] (caller-supplied routing); panics for
    /// [`RouterMode::KV`], which selects via `kv_chooser::find_best_match`.
    pub fn peek_next_worker(&self) -> Option<u64> {
        // Select among free (admission-eligible) workers — see select_next_worker
        // for the per-mode selection rationale.
        let instance_ids = self.client.routing_instances().free_ids().to_vec();
        let count = instance_ids.len();
        if count == 0 {
            return None;
        }

        match self.router_mode {
            RouterMode::RoundRobin => {
                // Just peek at the current counter value without incrementing
                let counter = self.round_robin_counter.load(Ordering::Relaxed) as usize;
                Some(instance_ids[counter % count])
            }
            RouterMode::Random => {
                // For random, peeking implies a fresh random selection since it's stateless.
                // Note: The caller must realize that select_next_worker() will pick a DIFFERENT random worker.
                let counter = rand::rng().random::<u64>() as usize;
                Some(instance_ids[counter % count])
            }
            RouterMode::LeastLoaded => self.occupancy_state.as_deref()?.peek_min(&instance_ids),
            RouterMode::PowerOfTwoChoices => Some(p2c_select_from(
                self.occupancy_state.as_deref()?,
                &instance_ids,
            )),
            RouterMode::DeviceAwareWeighted => {
                let state = self.occupancy_state.as_deref()?;
                let device_type_map: HashMap<u64, Option<DeviceType>> = self
                    .client
                    .instances()
                    .iter()
                    .map(|instance| (instance.instance_id, instance.device_type.clone()))
                    .collect();
                let cuda_to_cpu_ratio = std::env::var("DYN_ENCODER_CUDA_TO_CPU_RATIO")
                    .ok()
                    .and_then(|value| value.parse::<usize>().ok())
                    .filter(|value| *value >= 1)
                    .unwrap_or(8);
                let candidates = device_aware_candidate_group(
                    state,
                    &instance_ids,
                    &device_type_map,
                    cuda_to_cpu_ratio,
                );
                state.peek_min(&candidates)
            }
            RouterMode::Direct => None,
            RouterMode::KV => {
                panic!(
                    "peek_next_worker should not be called for {:?} routing mode",
                    self.router_mode
                )
            }
        }
    }

    async fn select_exact_target(
        &self,
        request: &T,
        pinned_worker: Option<u64>,
    ) -> anyhow::Result<(u64, Option<OccupancyPermit>)> {
        if let Some(instance_id) = pinned_worker {
            let routing_instances = self.client.routing_instances();
            if !routing_instances.routable_ids().contains(&instance_id) {
                return Err(anyhow::anyhow!(
                    "instance_id={instance_id} not found for endpoint {}",
                    self.client.endpoint.id()
                ));
            }
            let permit = match self.router_mode {
                RouterMode::LeastLoaded
                | RouterMode::PowerOfTwoChoices
                | RouterMode::DeviceAwareWeighted => {
                    let state = self.occupancy_state()?;
                    state.increment(instance_id);
                    Some(OccupancyPermit::new(state, instance_id))
                }
                RouterMode::RoundRobin
                | RouterMode::Random
                | RouterMode::Direct
                | RouterMode::KV => None,
            };
            return Ok((instance_id, permit));
        }

        match self.router_mode {
            RouterMode::LeastLoaded
            | RouterMode::PowerOfTwoChoices
            | RouterMode::DeviceAwareWeighted => {
                let state = self.occupancy_state()?;
                let routing_instances = self.client.routing_instances();
                let instance_ids = routing_instances.free_ids().to_vec();
                if instance_ids.is_empty() {
                    return Err(self.empty_free_pool_error(&routing_instances));
                }

                let instance_id = match self.router_mode {
                    RouterMode::LeastLoaded => state
                        .select_exact_min_and_increment(&instance_ids)
                        .await
                        .ok_or_else(|| self.empty_free_pool_error(&routing_instances))?,
                    RouterMode::PowerOfTwoChoices => {
                        let instance_id = p2c_select_from(state.as_ref(), &instance_ids);
                        state.increment(instance_id);
                        instance_id
                    }
                    RouterMode::DeviceAwareWeighted => {
                        let selection =
                            self.device_aware_candidates(request, state.as_ref(), &instance_ids);
                        let instance_id = if selection.full_embedding_cache_hit {
                            state.select_exact_min(&selection.candidates).await
                        } else {
                            state
                                .select_exact_min_and_increment(&selection.candidates)
                                .await
                        }
                        .ok_or_else(|| self.empty_free_pool_error(&routing_instances))?;
                        let permit = (!selection.full_embedding_cache_hit)
                            .then(|| OccupancyPermit::new(state, instance_id));
                        return Ok((instance_id, permit));
                    }
                    _ => unreachable!(),
                };
                Ok((instance_id, Some(OccupancyPermit::new(state, instance_id))))
            }
            RouterMode::RoundRobin | RouterMode::Random => self
                .select_next_worker()
                .map(|instance_id| (instance_id, None))
                .ok_or_else(|| {
                    let routing_instances = self.client.routing_instances();
                    self.empty_free_pool_error(&routing_instances)
                }),
            RouterMode::Direct => Err(anyhow::anyhow!(
                "Worker ID required for exact dispatch in Direct routing mode"
            )),
            RouterMode::KV => Err(anyhow::anyhow!(
                "select_and_dispatch_exact cannot select workers in KV routing mode"
            )),
        }
    }

    fn occupancy_state(&self) -> anyhow::Result<Arc<RoutingOccupancyState>> {
        self.occupancy_state.clone().ok_or_else(|| {
            anyhow::anyhow!(
                "routing occupancy state not initialized for endpoint {}",
                self.client.endpoint.id()
            )
        })
    }

    /*
    pub async fn r#static(&self, request: SingleIn<T>) -> anyhow::Result<ManyOut<U>> {
        let subject = self.client.endpoint.subject();
        tracing::debug!("static got subject: {subject}");
        let request = request.map(|req| AddressedRequest::new(req, subject));
        tracing::debug!("router generate");
        self.addressed.generate(request).await
    }
    */

    async fn generate_with_fault_detection(
        &self,
        instance_id: u64,
        request: SingleIn<T>,
        fallback: TransportFallback<'_>,
    ) -> anyhow::Result<ManyOut<U>> {
        self.generate_with_fault_detection_prepared(instance_id, request, fallback, |_, _| Ok(()))
            .await
            .map(|(_, stream)| stream)
    }

    async fn generate_with_fault_detection_prepared<M, F>(
        &self,
        instance_id: u64,
        mut request: SingleIn<T>,
        fallback: TransportFallback<'_>,
        prepare: F,
    ) -> anyhow::Result<(M, ManyOut<U>)>
    where
        F: FnOnce(&mut T, u64) -> anyhow::Result<M>,
    {
        let route_start = Instant::now();
        let request_id = request.id().to_string();
        let route_span = if matches!(self.router_mode, RouterMode::KV) {
            tracing::Span::none()
        } else {
            tracing::info_span!(
                "router.route_request",
                request_id = %request_id,
                worker_id = instance_id,
                router_mode = ?self.router_mode,
            )
        };

        self.check_workers_available(instance_id, &request_id)?;

        let (instance_id, address, transport_kind, instance) =
            self.resolve_transport(instance_id, fallback)?;
        let metadata = prepare(&mut request, instance_id)?;
        let request = request.map(|req| AddressedRequest::with_instance(req, address, instance));

        STAGE_DURATION_SECONDS
            .with_label_values(&[STAGE_ROUTE])
            .observe(route_start.elapsed().as_secs_f64());

        let _nvtx_transport = dynamo_nvtx_range!(transport_kind);
        let stream = self
            .addressed
            .generate(request)
            .instrument(route_span)
            .await;
        let stream = self.wrap_with_fault_detection(stream, instance_id)?;
        Ok((metadata, stream))
    }

    /// Reject early if the selected worker is overloaded and fault detection
    /// is enabled. The request_id is only used for the debug-level "checked
    /// worker overload state" trace; pass an empty string from callers that
    /// don't have one handy.
    fn check_workers_available(&self, instance_id: u64, request_id: &str) -> anyhow::Result<()> {
        if !self.fault_detection_enabled {
            return Ok(());
        }
        let routing_instances = self.client.routing_instances();
        let selected_worker_overloaded = routing_instances.is_overloaded(instance_id);
        let counts = routing_instances.counts();
        if tracing::enabled!(tracing::Level::DEBUG) {
            tracing::debug!(
                request_id,
                instance_id,
                router_mode = ?self.router_mode,
                free_workers = counts.free,
                overloaded_workers = counts.overloaded,
                total_workers = counts.discovered,
                selected_worker_overloaded,
                "checked worker overload state"
            );
        }
        if !selected_worker_overloaded {
            return Ok(());
        }
        tracing::warn!(
            instance_id,
            overloaded_workers = counts.overloaded,
            total_workers = counts.discovered,
            "Rejecting request: selected worker is overloaded"
        );
        let cause = PipelineError::ServiceOverloaded(
            "Selected worker is overloaded, please retry later".into(),
        );
        Err(DynamoError::builder()
            .error_type(ErrorType::ResourceExhausted)
            .message("Selected worker is overloaded, please retry later")
            .cause(cause)
            .build()
            .into())
    }

    /// Resolve `(instance_id, address, transport_kind_label, Instance)` for
    /// the selected worker. If the instance has disappeared between selection
    /// and dispatch, fall back to one other instance from `free_ids` (same
    /// filter as pre-selection) and return the updated id so the caller can
    /// `report_instance_down` the right worker on later failures.
    fn resolve_transport(
        &self,
        instance_id: u64,
        fallback: TransportFallback<'_>,
    ) -> anyhow::Result<(u64, String, &'static str, Instance)> {
        use crate::component::TransportType;

        let lookup = |id: u64| {
            self.client
                .instances()
                .iter()
                .find(|i| i.instance_id == id)
                .map(|instance| {
                    let (addr, kind) = match &instance.transport {
                        TransportType::Tcp(tcp_endpoint) => {
                            (tcp_endpoint.clone(), "transport.tcp.request")
                        }
                        TransportType::Nats(subject) => (subject.clone(), "transport.nats.request"),
                    };
                    (addr, kind, instance.clone())
                })
        };

        if let Some((addr, kind, inst)) = lookup(instance_id) {
            return Ok((instance_id, addr, kind, inst));
        }
        let allowed_fallback = match fallback {
            TransportFallback::Allow => None,
            TransportFallback::Deny => {
                return Err(anyhow::anyhow!(
                    "Instance {} not found for endpoint {}",
                    instance_id,
                    self.client.endpoint.id()
                ));
            }
            TransportFallback::Within(allowed) => Some(allowed),
        };

        let routing_instances = self.client.routing_instances();
        let fallback_id = routing_instances.free_ids().iter().copied().find(|&id| {
            id != instance_id && allowed_fallback.is_none_or(|allowed| allowed.contains(&id))
        });
        match fallback_id {
            Some(id) => {
                tracing::warn!(
                    original_instance = instance_id,
                    fallback_instance = id,
                    "Instance disappeared during routing, reselecting"
                );
                let (addr, kind, inst) = lookup(id).ok_or_else(|| {
                    anyhow::anyhow!(
                        "Fallback instance {} also not found for endpoint {}",
                        id,
                        self.client.endpoint.id()
                    )
                })?;
                Ok((id, addr, kind, inst))
            }
            None => Err(anyhow::anyhow!(
                "Instance {} not found and no other instances available for endpoint {}",
                instance_id,
                self.client.endpoint.id()
            )),
        }
    }

    /// Wrap a dispatched stream with fault detection + inactivity timeout.
    /// `is_inhibited` errors trigger `report_instance_down`; the timeout
    /// (driven by `DYN_HTTP_BACKEND_STREAM_TIMEOUT_SECS`) yields a synthetic
    /// `ResponseTimeout` and quarantines the worker.
    fn wrap_with_fault_detection(
        &self,
        stream: anyhow::Result<ManyOut<U>>,
        instance_id: u64,
    ) -> anyhow::Result<ManyOut<U>> {
        let stream = match stream {
            Ok(stream) => stream,
            Err(err) => {
                if self.fault_detection_enabled {
                    if is_inhibited(err.as_ref()) {
                        tracing::debug!(
                            "Reporting instance {instance_id} down due to error: {err}"
                        );
                        self.client.report_instance_down(instance_id);
                    } else if match_error_chain(err.as_ref(), &[ErrorType::ResourceExhausted], &[])
                    {
                        // Backpressure: worker said "my queue is full,
                        // retry later". Mark overloaded so this FE skips it on
                        // the next selection; the next ActiveLoad event from the
                        // worker monitor overwrites the overloaded set from fresh
                        // metrics. This is NOT report_instance_down (fault path).
                        tracing::debug!(
                            "Marking instance {instance_id} overloaded due to backpressure: {err}"
                        );
                        self.client.mark_overloaded_immediate(instance_id);
                    }
                }
                return Err(err);
            }
        };

        if !self.fault_detection_enabled {
            return Ok(stream);
        }

        let engine_ctx = stream.context();
        let client = self.client.clone();
        let client_for_timeout = self.client.clone();
        let stream = stream.map(move |res| {
            if let Some(err) = res.err()
                && is_inhibited(&err)
            {
                tracing::debug!(
                    "Reporting instance {instance_id} down due to migratable error: {err}"
                );
                client.report_instance_down(instance_id);
            }
            res
        });

        let stream: Pin<Box<dyn Stream<Item = U> + Send>> =
            if let Some(timeout) = self.response_timeout {
                Box::pin(async_stream::stream! {
                    let mut inner = Box::pin(stream);
                    loop {
                        tokio::select! {
                            biased;
                            item = inner.next() => {
                                match item {
                                    Some(item) => yield item,
                                    None => break,
                                }
                            }
                            _ = tokio::time::sleep(timeout) => {
                                tracing::warn!(
                                    instance_id,
                                    timeout_secs = timeout.as_secs(),
                                    "backend response inactivity timeout — quarantining worker"
                                );
                                client_for_timeout.report_instance_down(instance_id);
                                yield U::from_err(
                                    crate::error::DynamoError::builder()
                                        .error_type(crate::error::ErrorType::ResponseTimeout)
                                        .message("backend response inactivity timeout")
                                        .build()
                                );
                                break;
                            }
                        }
                    }
                })
            } else {
                Box::pin(stream)
            };

        Ok(ResponseStream::new(stream, engine_ctx))
    }
}

#[async_trait]
impl<T, U> AsyncEngine<SingleIn<T>, ManyOut<U>, Error> for PushRouter<T, U>
where
    T: Data + Serialize,
    U: Data + for<'de> Deserialize<'de> + MaybeError,
{
    async fn generate(&self, request: SingleIn<T>) -> Result<ManyOut<U>, Error> {
        match self.router_mode {
            RouterMode::Random => self.random(request).await,
            RouterMode::RoundRobin => self.round_robin(request).await,
            RouterMode::PowerOfTwoChoices => self.power_of_two_choices(request).await,
            RouterMode::KV => {
                anyhow::bail!("KV routing should not call generate on PushRouter");
            }
            RouterMode::Direct => {
                anyhow::bail!(
                    "Direct routing should not call generate on PushRouter directly; use DirectRoutingRouter wrapper"
                );
            }
            RouterMode::LeastLoaded => self.least_loaded(request).await,
            RouterMode::DeviceAwareWeighted => self.device_aware_weighted(request).await,
        }
    }
}

impl<T, U> PushRouter<T, U>
where
    T: Data + Serialize,
    U: Data + for<'de> Deserialize<'de> + MaybeError,
{
    /// Bidirectional sibling of [`Self::generate_with_fault_detection`].
    async fn bidirectional_dispatch(
        &self,
        instance_id: u64,
        input: ManyIn<T>,
    ) -> anyhow::Result<ManyOut<U>> {
        let route_start = Instant::now();
        let request_id = input.context().id().to_string();
        let route_span = tracing::info_span!(
            "router.route_request_bidirectional",
            request_id = %request_id,
            worker_id = instance_id,
            router_mode = ?self.router_mode,
        );

        self.check_workers_available(instance_id, &request_id)?;
        let (instance_id, address, transport_kind, instance) =
            self.resolve_transport(instance_id, TransportFallback::Allow)?;

        STAGE_DURATION_SECONDS
            .with_label_values(&[STAGE_ROUTE])
            .observe(route_start.elapsed().as_secs_f64());

        let _nvtx_transport = dynamo_nvtx_range!(transport_kind);
        let stream: anyhow::Result<ManyOut<U>> = self
            .addressed
            .generate_bidirectional(instance, address, input)
            .instrument(route_span)
            .await;
        self.wrap_with_fault_detection(stream, instance_id)
    }
}

/// Bidirectional `AsyncEngine` impl for streaming-input workloads (e.g. the
/// OpenAI Realtime API). Reserves a sticky worker up front — before any
/// inbound frame is observed — and binds the whole input stream to that
/// worker. KV and Direct modes inherit the same `bail!` invariants as the
/// unary impl.
///
/// **Reserve-before-observe rationale.** The router-mode strategies
/// (`RoundRobin`, `Random`, `PowerOfTwoChoices`, `LeastLoaded`,
/// `DeviceAwareWeighted`) don't depend on frame contents, so selection
/// runs immediately and connection setup proceeds in parallel with the
/// client producing its first frame. A client that connects but never
/// sends one still releases the slot via the response-stream-drop path;
/// the dispatch-side `cancel_both` cleanup covers the early-bail case.
#[async_trait]
impl<T, U> AsyncEngine<ManyIn<T>, ManyOut<U>, Error> for PushRouter<T, U>
where
    T: Data + Serialize,
    U: Data + for<'de> Deserialize<'de> + MaybeError,
{
    async fn generate(&self, input: ManyIn<T>) -> Result<ManyOut<U>, Error> {
        match self.router_mode {
            RouterMode::KV => {
                anyhow::bail!("KV routing should not call generate on PushRouter");
            }
            RouterMode::Direct => {
                anyhow::bail!(
                    "Direct routing should not call generate on PushRouter directly; use DirectRoutingRouter wrapper"
                );
            }
            // These modes drive `select_next_worker()` to `None` — they rely on
            // the occupancy/load-aware selection the bidirectional path does not
            // wire yet, which would otherwise surface as a misleading "no
            // instances available" error below. Reject them explicitly until
            // bidirectional support lands; tracked in
            // https://github.com/ai-dynamo/dynamo/issues/10320.
            RouterMode::PowerOfTwoChoices
            | RouterMode::LeastLoaded
            | RouterMode::DeviceAwareWeighted => {
                anyhow::bail!(
                    "{:?} routing is not yet supported for bidirectional dispatch",
                    self.router_mode
                );
            }
            RouterMode::RoundRobin | RouterMode::Random => {}
        }

        let instance_id = self
            .select_next_worker()
            .ok_or_else(|| anyhow::anyhow!("no instances available for bidirectional routing"))?;

        self.bidirectional_dispatch(instance_id, input).await
    }
}

struct OccupancyTrackedStream<U: Data> {
    inner: ManyOut<U>,
    state: Arc<RoutingOccupancyState>,
    instance_id: u64,
    released: bool,
}

impl<U: Data> Drop for OccupancyTrackedStream<U> {
    fn drop(&mut self) {
        if !self.released {
            self.state.decrement(self.instance_id);
        }
    }
}

impl<U: Data> std::fmt::Debug for OccupancyTrackedStream<U> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("OccupancyTrackedStream")
            .field("instance_id", &self.instance_id)
            .finish()
    }
}

impl<U: Data> Stream for OccupancyTrackedStream<U> {
    type Item = U;

    fn poll_next(
        mut self: Pin<&mut Self>,
        cx: &mut std::task::Context<'_>,
    ) -> Poll<Option<Self::Item>> {
        let poll = self.inner.as_mut().poll_next(cx);
        if matches!(poll, Poll::Ready(None)) && !self.released {
            self.state.decrement(self.instance_id);
            self.released = true;
        }
        poll
    }
}

impl<U: Data> AsyncEngineContextProvider for OccupancyTrackedStream<U> {
    fn context(&self) -> Arc<dyn AsyncEngineContext> {
        self.inner.context()
    }
}

impl<U: Data> crate::engine::AsyncEngineStream<U> for OccupancyTrackedStream<U> {}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        DistributedRuntime, Runtime,
        distributed::DistributedConfig,
        error::DynamoError,
        pipeline::{
            RequestStream, ResponseStream,
            context::{Context, Controller},
        },
    };
    use serde::{Deserialize, Serialize};

    #[derive(Clone, Debug, Deserialize, Serialize)]
    struct TestResponse {
        error: Option<DynamoError>,
    }

    impl MaybeError for TestResponse {
        fn from_err(err: impl std::error::Error + 'static) -> Self {
            Self {
                error: Some(DynamoError::from(
                    Box::new(err) as Box<dyn std::error::Error + 'static>
                )),
            }
        }

        fn err(&self) -> Option<DynamoError> {
            self.error.clone()
        }
    }

    struct StaticMultimodalCacheIndex {
        worker_id: u64,
    }

    impl MultimodalCacheIndex for StaticMultimodalCacheIndex {
        fn workers_with_cache_key_hits(&self, cache_keys: &[String]) -> Vec<(u64, usize)> {
            vec![(self.worker_id, cache_keys.len())]
        }

        fn remove_worker(&self, _worker_id: u64) {}
    }

    #[test]
    fn p2c_selects_lower_load_worker() {
        let state = RoutingOccupancyState::default();
        for _ in 0..10 {
            state.increment(1);
        }
        state.increment(2);

        // With only two workers, p2c_select_from must pick both and choose id=2 (lower load).
        let result = p2c_select_from(&state, &[1, 2]);
        assert_eq!(result, 2);
    }

    #[test]
    fn p2c_selects_single_worker() {
        let state = RoutingOccupancyState::default();
        assert_eq!(p2c_select_from(&state, &[42]), 42);
    }

    #[test]
    fn p2c_treats_missing_counts_as_zero() {
        let state = RoutingOccupancyState::default();
        for _ in 0..5 {
            state.increment(1);
        }
        // Worker 2 has no entry — should be treated as 0, so it wins.
        let result = p2c_select_from(&state, &[1, 2]);
        assert_eq!(result, 2);
    }

    #[test]
    fn p2c_returns_valid_worker_on_tie() {
        let state = RoutingOccupancyState::default();
        for _ in 0..3 {
            state.increment(1);
            state.increment(2);
        }

        for _ in 0..100 {
            let result = p2c_select_from(&state, &[1, 2]);
            assert!(result == 1 || result == 2);
        }
    }

    #[test]
    fn occupancy_permit_decrements_before_stream_creation() {
        let state = Arc::new(RoutingOccupancyState::default());
        state.increment(42);
        let permit = OccupancyPermit::new(state.clone(), 42);
        assert_eq!(state.load(42), 1);
        drop(permit);
        assert_eq!(state.load(42), 0);
    }

    #[test]
    fn occupancy_tracked_stream_decrements_on_drop() {
        let state = Arc::new(RoutingOccupancyState::default());
        state.increment(7);
        let permit = OccupancyPermit::new(state.clone(), 7);
        let ctx: Arc<dyn AsyncEngineContext> = Arc::new(Controller::default());
        let stream = permit.into_tracked_stream(ResponseStream::new(
            Box::pin(tokio_stream::iter(vec![1u64])),
            ctx,
        ));
        assert_eq!(state.load(7), 1);
        drop(stream);
        assert_eq!(state.load(7), 0);
    }

    #[tokio::test]
    async fn occupancy_tracked_stream_decrements_on_completion() {
        let state = Arc::new(RoutingOccupancyState::default());
        state.increment(7);
        let permit = OccupancyPermit::new(state.clone(), 7);
        let ctx: Arc<dyn AsyncEngineContext> = Arc::new(Controller::default());
        let mut stream = permit.into_tracked_stream(ResponseStream::new(
            Box::pin(tokio_stream::iter(vec![1u64])),
            ctx,
        ));

        assert_eq!(stream.next().await, Some(1));
        assert_eq!(state.load(7), 1);
        assert_eq!(stream.next().await, None);
        assert_eq!(state.load(7), 0);
        drop(stream);
        assert_eq!(state.load(7), 0, "drop must not release twice after EOF");
    }

    #[test]
    fn p2c_lifecycle_tracks_inflight_counts_with_shared_tracker() {
        let state = Arc::new(RoutingOccupancyState::default());
        let mut permits = Vec::new();
        for _ in 0..5 {
            let selected = p2c_select_from(&state, &[1, 2]);
            state.increment(selected);
            permits.push(OccupancyPermit::new(state.clone(), selected));
        }

        let total = state.load(1) + state.load(2);
        assert_eq!(total, 5, "5 in-flight requests should be tracked");

        drop(permits);
        let total = state.load(1) + state.load(2);
        assert_eq!(total, 0, "All guards dropped, counts should be 0");
    }

    #[test]
    fn p2c_never_selects_dominated_worker() {
        let state = RoutingOccupancyState::default();
        for _ in 0..100 {
            state.increment(3);
        }

        let mut selected = [0u32; 3];
        for _ in 0..1000 {
            let result = p2c_select_from(&state, &[1, 2, 3]);
            match result {
                1 => selected[0] += 1,
                2 => selected[1] += 1,
                3 => selected[2] += 1,
                _ => panic!("unexpected worker id"),
            }
        }
        assert_eq!(
            selected[2], 0,
            "Worker 3 (load=100) should never be selected against load=0 workers, but got {} times",
            selected[2]
        );
    }

    #[tokio::test]
    async fn least_loaded_selects_exact_min_and_tracks_counts() {
        let state = Arc::new(RoutingOccupancyState::default());
        state.increment(1);
        state.increment(1);
        state.increment(2);

        let selected = state
            .select_exact_min_and_increment(&[1, 2, 3])
            .await
            .unwrap();
        assert_eq!(selected, 3);

        let permit = OccupancyPermit::new(state.clone(), selected);
        assert_eq!(state.load(selected), 1);
        drop(permit);
        assert_eq!(state.load(selected), 0);
    }

    #[tokio::test]
    async fn bidirectional_generate_bails_with_no_instances() {
        let rt = Runtime::from_current().unwrap();
        let drt = DistributedRuntime::new(rt.clone(), DistributedConfig::process_local())
            .await
            .unwrap();
        let ns = drt.namespace("test_bidi_no_instances".to_string()).unwrap();
        let component = ns.component("test_component".to_string()).unwrap();
        let endpoint = component.endpoint("test_endpoint".to_string());
        let client = endpoint.client().await.unwrap();

        let router = PushRouter::<u64, TestResponse>::from_client(client, RouterMode::RoundRobin)
            .await
            .unwrap();

        let input: ManyIn<u64> =
            Context::new(RequestStream::new(Box::pin(tokio_stream::iter(vec![
                1u64, 2u64,
            ]))));
        let result = router.generate(input).await;
        assert!(
            result.is_err(),
            "bidirectional generate must bail when no instances are registered"
        );

        rt.shutdown();
    }

    #[tokio::test]
    async fn bidirectional_generate_bails_for_kv_router_mode() {
        let rt = Runtime::from_current().unwrap();
        let drt = DistributedRuntime::new(rt.clone(), DistributedConfig::process_local())
            .await
            .unwrap();
        let ns = drt.namespace("test_bidi_kv_mode".to_string()).unwrap();
        let component = ns.component("test_component".to_string()).unwrap();
        let endpoint = component.endpoint("test_endpoint".to_string());
        let client = endpoint.client().await.unwrap();

        let router = PushRouter::<u64, TestResponse>::from_client(client, RouterMode::KV)
            .await
            .unwrap();

        let input: ManyIn<u64> =
            Context::new(RequestStream::new(Box::pin(tokio_stream::iter(vec![1u64]))));
        let result = router.generate(input).await;
        assert!(
            result.is_err(),
            "bidirectional generate must bail for RouterMode::KV"
        );
        let err_msg = format!("{:?}", result.unwrap_err());
        assert!(
            err_msg.contains("KV") || err_msg.contains("kv"),
            "error should mention KV: got {err_msg}"
        );

        rt.shutdown();
    }

    #[tokio::test]
    async fn bidirectional_generate_bails_for_direct_router_mode() {
        let rt = Runtime::from_current().unwrap();
        let drt = DistributedRuntime::new(rt.clone(), DistributedConfig::process_local())
            .await
            .unwrap();
        let ns = drt.namespace("test_bidi_direct_mode".to_string()).unwrap();
        let component = ns.component("test_component".to_string()).unwrap();
        let endpoint = component.endpoint("test_endpoint".to_string());
        let client = endpoint.client().await.unwrap();

        let router = PushRouter::<u64, TestResponse>::from_client(client, RouterMode::Direct)
            .await
            .unwrap();

        let input: ManyIn<u64> =
            Context::new(RequestStream::new(Box::pin(tokio_stream::iter(vec![1u64]))));
        let result = router.generate(input).await;
        assert!(
            result.is_err(),
            "bidirectional generate must bail for RouterMode::Direct"
        );
        let err_msg = format!("{:?}", result.unwrap_err());
        assert!(
            err_msg.contains("Direct") || err_msg.contains("direct"),
            "error should mention Direct: got {err_msg}"
        );

        rt.shutdown();
    }

    #[tokio::test]
    async fn bidirectional_generate_rejects_unsupported_load_aware_modes() {
        let rt = Runtime::from_current().unwrap();
        let drt = DistributedRuntime::new(rt.clone(), DistributedConfig::process_local())
            .await
            .unwrap();
        let ns = drt.namespace("test_bidi_load_aware".to_string()).unwrap();
        let component = ns.component("test_component".to_string()).unwrap();

        for mode in [
            RouterMode::PowerOfTwoChoices,
            RouterMode::LeastLoaded,
            RouterMode::DeviceAwareWeighted,
        ] {
            let endpoint = component.endpoint("test_endpoint".to_string());
            let client = endpoint.client().await.unwrap();
            let router = PushRouter::<u64, TestResponse>::from_client(client, mode)
                .await
                .unwrap();

            let input: ManyIn<u64> =
                Context::new(RequestStream::new(Box::pin(tokio_stream::iter(vec![1u64]))));
            let result = router.generate(input).await;
            assert!(
                result.is_err(),
                "bidirectional generate must reject {mode:?} (not yet supported)"
            );
            let err_msg = format!("{:?}", result.unwrap_err());
            assert!(
                err_msg.contains("not yet supported for bidirectional dispatch"),
                "error should explain the mode is unsupported, not 'no instances': got {err_msg}"
            );
        }

        rt.shutdown();
    }

    #[tokio::test]
    async fn least_loaded_peek_returns_available_worker_select_stays_none() {
        let rt = Runtime::from_current().unwrap();
        let drt = DistributedRuntime::new(rt.clone(), DistributedConfig::process_local())
            .await
            .unwrap();
        let ns = drt
            .namespace("test_least_loaded_router".to_string())
            .unwrap();
        let component = ns.component("test_component".to_string()).unwrap();
        let endpoint = component.endpoint("test_endpoint".to_string());
        let client = endpoint.client().await.unwrap();

        endpoint.register_endpoint_instance().await.unwrap();
        client.wait_for_instances().await.unwrap();

        let router = PushRouter::<u64, TestResponse>::from_client(client, RouterMode::LeastLoaded)
            .await
            .unwrap();

        // LeastLoaded selection tracks request occupancy, so the advisory API is
        // separate from select_next_worker().
        assert_eq!(router.select_next_worker(), None);
        assert!(
            router.peek_next_worker().is_some(),
            "LeastLoaded peek must return the available worker for disagg bootstrap"
        );

        rt.shutdown();
    }

    #[tokio::test]
    async fn exact_selection_releases_occupancy_when_preparation_fails() {
        let rt = Runtime::from_current().unwrap();
        let drt = DistributedRuntime::new(rt.clone(), DistributedConfig::process_local())
            .await
            .unwrap();
        let ns = drt
            .namespace("test_exact_prepare_failure".to_string())
            .unwrap();
        let component = ns.component("test_component".to_string()).unwrap();
        let endpoint = component.endpoint("test_endpoint".to_string());
        let client = endpoint.client().await.unwrap();
        endpoint.register_endpoint_instance().await.unwrap();
        let worker_id = client.wait_for_instances().await.unwrap()[0].id();

        let router = PushRouter::<u64, TestResponse>::from_client(client, RouterMode::LeastLoaded)
            .await
            .unwrap();
        let state = router.occupancy_state.clone().unwrap();
        let result = router
            .select_and_dispatch_exact(SingleIn::new(42), None, |_, _| {
                Err::<(), _>(anyhow::anyhow!("metadata preparation failed"))
            })
            .await;

        assert!(result.is_err());
        assert_eq!(
            state.load(worker_id),
            0,
            "preparation failure must release the selected worker"
        );
        rt.shutdown();
    }

    #[tokio::test]
    async fn exact_dispatch_revalidates_overload_after_preparation() {
        let rt = Runtime::from_current().unwrap();
        let drt = DistributedRuntime::new(rt.clone(), DistributedConfig::process_local())
            .await
            .unwrap();
        let ns = drt
            .namespace("test_exact_overload_revalidation".to_string())
            .unwrap();
        let component = ns.component("test_component".to_string()).unwrap();
        let endpoint = component.endpoint("test_endpoint".to_string());
        let client = endpoint.client().await.unwrap();
        endpoint.register_endpoint_instance().await.unwrap();
        let worker_id = client.wait_for_instances().await.unwrap()[0].id();

        let router =
            PushRouter::<u64, TestResponse>::from_client(client.clone(), RouterMode::LeastLoaded)
                .await
                .unwrap();
        let state = router.occupancy_state.clone().unwrap();
        let result = router
            .select_and_dispatch_exact(SingleIn::new(42), Some(worker_id), |_, worker_id| {
                client.set_overloaded_instances(&[worker_id]);
                Ok(())
            })
            .await;

        assert!(result.is_err());
        assert_eq!(
            state.load(worker_id),
            0,
            "validation failure must release the selected worker"
        );
        rt.shutdown();
    }

    #[tokio::test]
    async fn selected_overloaded_worker_is_rejected_before_dispatch() {
        const TEST_RECONCILE_INTERVAL: std::time::Duration = std::time::Duration::from_millis(50);

        let rt = Runtime::from_current().unwrap();
        let drt = DistributedRuntime::new(rt.clone(), DistributedConfig::process_local())
            .await
            .unwrap();
        let ns = drt
            .namespace("test_selected_overloaded_worker_rejected".to_string())
            .unwrap();
        let component = ns.component("test_component".to_string()).unwrap();
        let endpoint = component.endpoint("test_endpoint".to_string());
        let client = Client::with_reconcile_interval(endpoint.clone(), TEST_RECONCILE_INTERVAL)
            .await
            .unwrap();

        endpoint.register_endpoint_instance().await.unwrap();
        let instances = client.wait_for_instances().await.unwrap();
        let worker_id = instances[0].id();

        for _ in 0..10 {
            if client.instance_ids_avail().contains(&worker_id) {
                break;
            }
            tokio::time::sleep(TEST_RECONCILE_INTERVAL).await;
        }
        assert!(
            client.instance_ids_avail().contains(&worker_id),
            "worker should be routable before marking it overloaded"
        );

        client.set_overloaded_instances(&[worker_id]);
        let router = PushRouter::<u64, TestResponse>::from_client(client, RouterMode::RoundRobin)
            .await
            .unwrap();

        let result = router.generate(SingleIn::new(42u64)).await;
        assert!(result.is_err());
        let msg = format!("{}", result.unwrap_err());
        // With pre-selection filtering on free_ids, the single-overloaded-worker
        // case is now caught before selection rather than after — the chosen
        // worker is never overloaded because the candidate pool excludes it.
        // The post-selection check in route() remains as a race-condition
        // backstop.
        assert!(
            msg.contains("All workers are busy"),
            "expected empty-free-pool rejection, got: {msg}"
        );

        rt.shutdown();
    }

    #[tokio::test]
    async fn direct_within_rejects_overloaded_constrained_target() {
        const TEST_RECONCILE_INTERVAL: std::time::Duration = std::time::Duration::from_secs(3600);

        let rt = Runtime::from_current().unwrap();
        let drt = DistributedRuntime::new(rt.clone(), DistributedConfig::process_local())
            .await
            .unwrap();
        let ns = drt
            .namespace("test_direct_within_overload_rejection".to_string())
            .unwrap();
        let component = ns.component("test_component".to_string()).unwrap();
        let endpoint = component.endpoint("test_endpoint".to_string());
        let client = Client::with_reconcile_interval(endpoint.clone(), TEST_RECONCILE_INTERVAL)
            .await
            .unwrap();

        endpoint.register_endpoint_instance().await.unwrap();
        let worker_id = client.wait_for_instances().await.unwrap()[0].id();
        client.set_overloaded_instances(&[worker_id]);

        let router = PushRouter::<u64, TestResponse>::from_client(client, RouterMode::RoundRobin)
            .await
            .unwrap();
        let allowed = HashSet::from([worker_id]);
        let error = router
            .direct_within(SingleIn::new(42), worker_id, Some(&allowed))
            .await
            .unwrap_err();

        assert!(match_error_chain(
            error.as_ref(),
            &[ErrorType::ResourceExhausted],
            &[]
        ));
        assert!(
            error.to_string().contains("Selected worker is overloaded"),
            "expected overload rejection, got: {error}"
        );

        rt.shutdown();
    }

    #[tokio::test]
    async fn no_workers_is_reported_as_unavailable() {
        let rt = Runtime::from_current().unwrap();
        let drt = DistributedRuntime::new(rt.clone(), DistributedConfig::process_local())
            .await
            .unwrap();
        let ns = drt
            .namespace("test_no_workers_unavailable".to_string())
            .unwrap();
        let component = ns.component("test_component".to_string()).unwrap();
        let endpoint = component.endpoint("test_endpoint".to_string());
        let client = endpoint.client().await.unwrap();
        let router = PushRouter::<u64, TestResponse>::from_client(client, RouterMode::RoundRobin)
            .await
            .unwrap();

        let error = router.generate(SingleIn::new(42)).await.unwrap_err();
        assert!(match_error_chain(
            error.as_ref(),
            &[ErrorType::Unavailable],
            &[]
        ));

        rt.shutdown();
    }

    #[tokio::test]
    async fn round_robin_excludes_overloaded_workers_from_candidates() {
        // Long reconcile interval so the synthetic override below survives
        // the test. We still register a real endpoint instance up front so
        // the initial reconcile (which fires immediately when the monitor
        // task spawns) settles on a non-empty source — without that, the
        // first reconcile would clobber the override before it takes effect.
        const TEST_RECONCILE_INTERVAL: std::time::Duration = std::time::Duration::from_secs(3600);

        let rt = Runtime::from_current().unwrap();
        let drt = DistributedRuntime::new(rt.clone(), DistributedConfig::process_local())
            .await
            .unwrap();
        let ns = drt
            .namespace("test_round_robin_excludes_overloaded".to_string())
            .unwrap();
        let component = ns.component("test_component".to_string()).unwrap();
        let endpoint = component.endpoint("test_endpoint".to_string());
        let client = Client::with_reconcile_interval(endpoint.clone(), TEST_RECONCILE_INTERVAL)
            .await
            .unwrap();

        endpoint.register_endpoint_instance().await.unwrap();
        let instances = client.wait_for_instances().await.unwrap();
        let real_id = instances[0].id();
        for _ in 0..50 {
            if client.instance_ids_avail().contains(&real_id) {
                break;
            }
            tokio::time::sleep(std::time::Duration::from_millis(20)).await;
        }

        // Now override with two synthetic IDs and mark one overloaded.
        // round_robin must never select the overloaded one — that's the
        // whole point of selecting from free_ids instead of routable_ids.
        // The post-selection overload check in route() would otherwise return 529
        // one of N requests on each pass, which is the bug this PR closes
        // for non-KV selectors.
        client.override_instance_avail(vec![1, 2]);
        client.set_overloaded_instances(&[1]);

        let router = PushRouter::<u64, TestResponse>::from_client(client, RouterMode::RoundRobin)
            .await
            .unwrap();

        // Round-robin over N requests should land on worker 2 every time.
        // We use peek_next_worker for a side-effect-free probe.
        for _ in 0..6 {
            let selected = router
                .peek_next_worker()
                .expect("peek should succeed with a free worker");
            assert_eq!(
                selected, 2,
                "overloaded worker 1 must not appear in the candidate set"
            );
        }

        rt.shutdown();
    }

    #[tokio::test]
    async fn round_robin_cycles_all_workers_after_membership_refresh() {
        const TEST_RECONCILE_INTERVAL: std::time::Duration = std::time::Duration::from_secs(3600);

        let rt = Runtime::from_current().unwrap();
        let drt = DistributedRuntime::new(rt.clone(), DistributedConfig::process_local())
            .await
            .unwrap();
        let ns = drt
            .namespace("test_round_robin_all_workers".to_string())
            .unwrap();
        let component = ns.component("test_component".to_string()).unwrap();
        let endpoint = component.endpoint("test_endpoint".to_string());
        let client = Client::with_reconcile_interval(endpoint.clone(), TEST_RECONCILE_INTERVAL)
            .await
            .unwrap();

        // Register a real endpoint instance and wait for discovery to settle on
        // non-empty membership before applying the synthetic override. The
        // monitor_instance_source task runs an initial reconcile immediately
        // when the client is constructed, regardless of the long reconcile
        // interval; without a real instance for that reconcile to land on, it
        // can clobber the override below and leave the router with no workers.
        endpoint.register_endpoint_instance().await.unwrap();
        let instances = client.wait_for_instances().await.unwrap();
        let real_id = instances[0].id();
        for _ in 0..50 {
            if client.instance_ids_avail().contains(&real_id) {
                break;
            }
            tokio::time::sleep(std::time::Duration::from_millis(20)).await;
        }

        client.override_instance_avail(vec![30, 10, 40, 20]);
        let router =
            PushRouter::<u64, TestResponse>::from_client(client.clone(), RouterMode::RoundRobin)
                .await
                .unwrap();

        let selected: Vec<u64> = (0..8).filter_map(|_| router.select_next_worker()).collect();
        assert_eq!(selected, vec![10, 20, 30, 40, 10, 20, 30, 40]);

        client.override_instance_avail(vec![40, 20, 50, 30]);
        let selected: Vec<u64> = (0..8).filter_map(|_| router.select_next_worker()).collect();
        assert_eq!(selected, vec![20, 30, 40, 50, 20, 30, 40, 50]);

        rt.shutdown();
    }

    #[tokio::test]
    async fn device_aware_cpu_only_selects_least_loaded_instance() {
        let state = RoutingOccupancyState::default();
        // All candidates are CPU. Make worker 2 the least-loaded one.
        for _ in 0..3 {
            state.increment(1);
        }
        state.increment(3);

        let instance_ids = vec![1, 2, 3];
        let device_type_map = HashMap::from([
            (1, Some(DeviceType::Cpu)),
            (2, Some(DeviceType::Cpu)),
            (3, Some(DeviceType::Cpu)),
        ]);

        let candidates = device_aware_candidate_group(&state, &instance_ids, &device_type_map, 8);
        assert_eq!(candidates, vec![1, 2, 3]);

        let selected = state
            .select_exact_min_and_increment(&candidates)
            .await
            .unwrap();
        assert_eq!(selected, 2);
    }

    #[tokio::test]
    async fn device_aware_non_cpu_only_selects_least_loaded_instance() {
        let state = RoutingOccupancyState::default();
        // All candidates are non-CPU. Make worker 2 the least-loaded one.
        for _ in 0..3 {
            state.increment(1);
        }
        state.increment(3);

        let instance_ids = vec![1, 2, 3];
        let device_type_map = HashMap::from([
            (1, Some(DeviceType::Cuda)),
            (2, Some(DeviceType::Cuda)),
            (3, Some(DeviceType::Cuda)),
        ]);

        let candidates = device_aware_candidate_group(&state, &instance_ids, &device_type_map, 8);
        assert_eq!(candidates, vec![1, 2, 3]);

        let selected = state
            .select_exact_min_and_increment(&candidates)
            .await
            .unwrap();
        assert_eq!(selected, 2);
    }

    #[test]
    fn device_aware_group_uses_ratio_budget() {
        let state = RoutingOccupancyState::default();
        // CPU ids: 1,2 ; non-CPU ids: 3,4
        for _ in 0..4 {
            state.increment(3);
            state.increment(4);
        }
        // CPU inflight can differ across instances; budgeting uses total CPU inflight.
        for _ in 0..3 {
            state.increment(1);
        }
        // total_non_cpu_inflight=8, cpu_count=2, non_cpu_count=2, ratio=2
        // allowed_cpu_inflight = 8*2/(2*2)=4
        // total_cpu_inflight=3 < 4 => choose CPU group.
        let instance_ids = vec![1, 2, 3, 4];
        let device_type_map = HashMap::from([
            (1, Some(DeviceType::Cpu)),
            (2, Some(DeviceType::Cpu)),
            (3, Some(DeviceType::Cuda)),
            (4, Some(DeviceType::Cuda)),
        ]);

        let candidates = device_aware_candidate_group(&state, &instance_ids, &device_type_map, 2);
        assert_eq!(candidates, vec![1, 2]);

        // Within selected CPU group, final choice should be the least-loaded instance (id=2).
        let selected =
            futures::executor::block_on(state.select_exact_min_and_increment(&candidates)).unwrap();
        assert_eq!(selected, 2);
    }

    #[tokio::test]
    async fn device_aware_weighted_peek_returns_available_worker_select_stays_none() {
        let rt = Runtime::from_current().unwrap();
        let drt = DistributedRuntime::new(rt.clone(), DistributedConfig::process_local())
            .await
            .unwrap();
        let ns = drt
            .namespace("test_device_aware_router".to_string())
            .unwrap();
        let component = ns.component("test_component".to_string()).unwrap();
        let endpoint = component.endpoint("test_endpoint".to_string());
        let client = endpoint.client().await.unwrap();

        endpoint.register_endpoint_instance().await.unwrap();
        client.wait_for_instances().await.unwrap();

        let router =
            PushRouter::<u64, TestResponse>::from_client(client, RouterMode::DeviceAwareWeighted)
                .await
                .unwrap();

        // DeviceAwareWeighted degenerates to least-loaded for peek (device-class
        // partitioning happens at dispatch); select_next_worker stays None.
        assert_eq!(router.select_next_worker(), None);
        assert!(
            router.peek_next_worker().is_some(),
            "DeviceAwareWeighted peek must return the available worker for disagg bootstrap"
        );

        rt.shutdown();
    }

    #[tokio::test]
    async fn device_aware_exact_selection_preserves_full_multimodal_cache_hit() {
        let rt = Runtime::from_current().unwrap();
        let drt = DistributedRuntime::new(rt.clone(), DistributedConfig::process_local())
            .await
            .unwrap();
        let ns = drt
            .namespace("test_device_aware_affinity_cache".to_string())
            .unwrap();
        let component = ns.component("test_component".to_string()).unwrap();
        let endpoint = component.endpoint("test_endpoint".to_string());
        let client = endpoint.client().await.unwrap();
        endpoint.register_endpoint_instance().await.unwrap();
        let cache_worker = client.wait_for_instances().await.unwrap()[0].id();

        let router = PushRouter::<u64, TestResponse>::from_client_with_state(
            client,
            RouterMode::DeviceAwareWeighted,
            None,
            Some(Arc::new(StaticMultimodalCacheIndex {
                worker_id: cache_worker,
            })),
            Some(Arc::new(|_| vec!["image-key".to_string()])),
        )
        .await
        .unwrap();

        let (worker_id, permit) = router.select_exact_target(&42, None).await.unwrap();
        assert_eq!(worker_id, cache_worker);
        assert!(
            permit.is_none(),
            "full cache hits bypass occupancy charging"
        );

        rt.shutdown();
    }

    /// When the router selects an instance that has deregistered between selection
    /// and transport resolution, it should fall back to another available instance
    /// rather than returning a 500 error.
    #[tokio::test]
    async fn transport_resolution_falls_back_when_selected_instance_disappears() {
        let rt = Runtime::from_current().unwrap();
        let drt = DistributedRuntime::new(rt.clone(), DistributedConfig::process_local())
            .await
            .unwrap();
        let ns = drt
            .namespace("test_transport_fallback".to_string())
            .unwrap();
        let component = ns.component("test_component".to_string()).unwrap();
        let endpoint = component.endpoint("test_endpoint".to_string());
        let client = endpoint.client().await.unwrap();

        // Register one real instance so it appears in instance_source.
        endpoint.register_endpoint_instance().await.unwrap();
        client.wait_for_instances().await.unwrap();

        let real_id = client.instance_ids()[0];

        // Inject a stale ID into instance_avail that does NOT exist in
        // instance_source. This simulates the race window where an instance
        // deregistered after selection but before transport resolution.
        let stale_id = real_id + 1000;
        client.override_instance_avail(vec![stale_id, real_id]);

        // Build a router and call direct() targeting the *real* instance to
        // verify the router can still resolve transport for known instances.
        let router =
            PushRouter::<u64, TestResponse>::from_client(client.clone(), RouterMode::RoundRobin)
                .await
                .unwrap();

        // Round robin should succeed — even if it picks stale_id first, the
        // fallback logic should resolve transport via real_id.
        // We cannot fully test the network send without a worker, but we can
        // verify it doesn't fail at the transport resolution stage by checking
        // that the error (if any) is a transport/network error, not
        // "Instance not found".
        let request = SingleIn::new(42u64);
        let result = router.generate(request).await;

        // The request may fail at the network level (no actual worker), but it
        // must NOT fail with "Instance X not found" — that would mean the
        // fallback did not work.
        if let Err(err) = &result {
            let msg = format!("{err}");
            assert!(
                !msg.contains("not found"),
                "Transport resolution should have fallen back, but got: {msg}"
            );
        }

        rt.shutdown();
    }

    #[tokio::test]
    async fn prepared_dispatch_observes_worker_after_transport_fallback() {
        let rt = Runtime::from_current().unwrap();
        let drt = DistributedRuntime::new(rt.clone(), DistributedConfig::process_local())
            .await
            .unwrap();
        let endpoint = drt
            .namespace("test_prepared_transport_fallback".to_string())
            .unwrap()
            .component("test_component".to_string())
            .unwrap()
            .endpoint("test_endpoint".to_string());
        let client = endpoint.client().await.unwrap();
        endpoint.register_endpoint_instance().await.unwrap();
        let real_id = client.wait_for_instances().await.unwrap()[0].id();
        let stale_id = real_id.wrapping_add(1);
        client.override_instance_avail(vec![stale_id, real_id]);
        let router = PushRouter::<u64, TestResponse>::from_client(client, RouterMode::LeastLoaded)
            .await
            .unwrap();
        let state = router.occupancy_state.clone().unwrap();
        state.increment(real_id);
        let state_for_prepare = state.clone();
        let observed = Arc::new(AtomicU64::new(0));
        let observed_for_prepare = observed.clone();

        let _ = tokio::time::timeout(
            std::time::Duration::from_millis(100),
            router.select_and_dispatch(SingleIn::new(42), move |_, worker_id| {
                assert_eq!(state_for_prepare.load(stale_id), 0);
                assert_eq!(state_for_prepare.load(worker_id), 2);
                observed_for_prepare.store(worker_id, Ordering::Relaxed);
                Ok(())
            }),
        )
        .await;

        assert_eq!(observed.load(Ordering::Relaxed), real_id);
        assert_eq!(state.load(real_id), 1);
        state.decrement(real_id);
        rt.shutdown();
    }

    /// When no instances are available at all (both primary and fallback),
    /// the router should return a clear error.
    #[tokio::test]
    async fn transport_resolution_errors_when_no_instances_available() {
        let rt = Runtime::from_current().unwrap();
        let drt = DistributedRuntime::new(rt.clone(), DistributedConfig::process_local())
            .await
            .unwrap();
        let ns = drt
            .namespace("test_transport_no_fallback".to_string())
            .unwrap();
        let component = ns.component("test_component".to_string()).unwrap();
        let endpoint = component.endpoint("test_endpoint".to_string());
        let client = endpoint.client().await.unwrap();

        // Register an instance so we can create the router (needs transport setup).
        endpoint.register_endpoint_instance().await.unwrap();
        client.wait_for_instances().await.unwrap();

        let router =
            PushRouter::<u64, TestResponse>::from_client(client.clone(), RouterMode::RoundRobin)
                .await
                .unwrap();

        // Override avail to contain only a stale ID with no real backing
        // instance AND no other available fallback.
        let stale_id = 99999;
        client.override_instance_avail(vec![stale_id]);

        let request = SingleIn::new(42u64);
        let result = router.generate(request).await;

        assert!(result.is_err());
        let msg = format!("{}", result.unwrap_err());
        assert!(
            msg.contains("not found") && msg.contains("no other instances available"),
            "Expected clear error about missing instance with no fallback, got: {msg}"
        );

        rt.shutdown();
    }

    #[tokio::test]
    async fn transport_resolution_honors_fallback_policy() {
        let rt = Runtime::from_current().unwrap();
        let drt = DistributedRuntime::new(rt.clone(), DistributedConfig::process_local())
            .await
            .unwrap();
        let ns = drt
            .namespace("test_exact_transport_no_fallback".to_string())
            .unwrap();
        let component = ns.component("test_component".to_string()).unwrap();
        let endpoint = component.endpoint("test_endpoint".to_string());
        let client = endpoint.client().await.unwrap();
        endpoint.register_endpoint_instance().await.unwrap();
        let instances = client.wait_for_instances().await.unwrap();
        let real_id = instances[0].id();

        let router =
            PushRouter::<u64, TestResponse>::from_client(client.clone(), RouterMode::RoundRobin)
                .await
                .unwrap();
        let stale_id = real_id.wrapping_add(1);
        client.override_instance_avail(vec![stale_id, real_id]);

        assert!(
            router
                .resolve_transport(stale_id, TransportFallback::Allow)
                .is_ok(),
            "normal dispatch should preserve transport fallback"
        );
        let allowed = HashSet::from([real_id]);
        assert!(
            router
                .resolve_transport(stale_id, TransportFallback::Within(&allowed))
                .is_ok(),
            "constrained dispatch should fall back within the allowed worker set"
        );
        let disallowed = HashSet::new();
        assert!(
            router
                .resolve_transport(stale_id, TransportFallback::Within(&disallowed))
                .is_err(),
            "constrained dispatch must not fall back outside the allowed worker set"
        );
        let error = router
            .resolve_transport(stale_id, TransportFallback::Deny)
            .unwrap_err();
        assert!(
            error.to_string().contains("not found"),
            "exact dispatch must reject the missing selected worker"
        );

        rt.shutdown();
    }

    /// The watcher dedup guard must be released even if the spawned task panics.
    /// Without this, a panic anywhere in the watcher body would leave a stale
    /// `ENDPOINT_WATCHER_ACTIVE` entry, silently disabling orphaned-pending-
    /// request cancellation for that endpoint until process restart.
    ///
    /// We exercise the Drop-guard pattern directly against the same static
    /// rather than driving `spawn_instance_removal_watcher` end-to-end (which
    /// would require staging a panicking discovery stream). The test mirrors
    /// the production code's GuardRelease shape; if the production code stops
    /// using a Drop guard, the integration would regress and the existing
    /// orphan-cancellation tests would fail.
    #[tokio::test]
    async fn watcher_dedup_guard_released_on_panic() {
        let endpoint_id = EndpointId {
            namespace: "panic-test-ns".to_string(),
            component: "panic-test-comp".to_string(),
            name: "panic-test-endpoint".to_string(),
        };

        // Mimic the production code's pre-spawn dedup insert.
        let map = ENDPOINT_WATCHER_ACTIVE.get_or_init(dashmap::DashMap::new);
        map.insert(endpoint_id.clone(), ());

        let endpoint_id_clone = endpoint_id.clone();
        let join = tokio::spawn(async move {
            // Same shape as in spawn_instance_removal_watcher.
            struct GuardRelease(EndpointId);
            impl Drop for GuardRelease {
                fn drop(&mut self) {
                    if let Some(map) = ENDPOINT_WATCHER_ACTIVE.get() {
                        map.remove(&self.0);
                    }
                }
            }
            let _release = GuardRelease(endpoint_id_clone);
            panic!("simulated watcher-task panic");
        });

        let result = join.await;
        assert!(result.is_err() && result.unwrap_err().is_panic());
        assert!(
            !map.contains_key(&endpoint_id),
            "Drop guard must release the dedup entry even on panic"
        );
    }

    /// Normal-exit path: the Drop guard releases the entry when the task
    /// finishes without panicking. This is the everyday case (cancel_token
    /// fires or discovery stream closes).
    #[tokio::test]
    async fn watcher_dedup_guard_released_on_normal_exit() {
        let endpoint_id = EndpointId {
            namespace: "normal-test-ns".to_string(),
            component: "normal-test-comp".to_string(),
            name: "normal-test-endpoint".to_string(),
        };

        let map = ENDPOINT_WATCHER_ACTIVE.get_or_init(dashmap::DashMap::new);
        map.insert(endpoint_id.clone(), ());

        let endpoint_id_clone = endpoint_id.clone();
        tokio::spawn(async move {
            struct GuardRelease(EndpointId);
            impl Drop for GuardRelease {
                fn drop(&mut self) {
                    if let Some(map) = ENDPOINT_WATCHER_ACTIVE.get() {
                        map.remove(&self.0);
                    }
                }
            }
            let _release = GuardRelease(endpoint_id_clone);
            // task body returns normally
        })
        .await
        .unwrap();

        assert!(!map.contains_key(&endpoint_id));
    }
}
