// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use super::{AsyncEngineContextProvider, ResponseStream};
use crate::error::{BackendError, DynamoError, ErrorType, match_error_chain};
use crate::{
    component::{
        Client, DeviceType, Endpoint, RoutingOccupancyState, get_or_create_routing_occupancy_state,
    },
    dynamo_nvtx_range,
    engine::{AsyncEngine, AsyncEngineContext, Data},
    metrics::frontend_perf::STAGE_DURATION_SECONDS,
    pipeline::{
        AddressedPushRouter, AddressedRequest, Error, ManyOut, SingleIn,
        error::{PipelineError, PipelineErrorExt},
    },
    protocols::maybe_error::MaybeError,
    traits::DistributedRuntimeProvider,
};
use async_trait::async_trait;
use futures::Stream;
use rand::Rng;
use serde::{Deserialize, Serialize};
use std::{
    collections::HashMap,
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

    fn into_tracked_stream<U: Data>(mut self, stream: ManyOut<U>) -> ManyOut<U> {
        self.armed = false;
        let engine_ctx = stream.context();
        ResponseStream::new(
            Box::pin(OccupancyTrackedStream {
                inner: stream,
                state: self.state.clone(),
                instance_id: self.instance_id,
            }),
            engine_ctx,
        )
    }

    fn instance_id(&self) -> u64 {
        self.instance_id
    }
}

impl Drop for OccupancyPermit {
    fn drop(&mut self) {
        if self.armed {
            self.state.decrement(self.instance_id);
        }
    }
}

/// Trait for monitoring worker load and determining busy state.
/// Implementations can define custom load metrics and busy thresholds.
#[async_trait]
pub trait WorkerLoadMonitor: Send + Sync {
    /// Start background monitoring of worker load.
    /// This should spawn background tasks that update the client's free instances.
    async fn start_monitoring(&self) -> anyhow::Result<()>;
}

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

    /// Threshold for determining when a worker is busy (0.0 to 1.0)
    /// If None, busy detection is disabled
    busy_threshold: Option<f64>,

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

    /// An internal Rust type. This says that PushRouter is generic over the T and U types,
    /// which are the input and output types of it's `generate` function. It allows the
    /// compiler to specialize us at compile time.
    _phantom: PhantomData<(T, U)>,
}

#[derive(Default, Debug, Clone, Copy, PartialEq)]
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
        return instance_ids[0];
    }
    let mut rng = rand::rng();
    let idx1 = rng.random_range(0..count);
    let idx2 = (idx1 + 1 + rng.random_range(0..count - 1)) % count;
    let id1 = instance_ids[idx1];
    let id2 = instance_ids[idx2];
    let load1 = occupancy_state.load(id1);
    let load2 = occupancy_state.load(id2);
    let selected = if load1 <= load2 { id1 } else { id2 };
    tracing::debug!(
        candidate_a = id1,
        candidate_a_load = load1,
        candidate_b = id2,
        candidate_b_load = load2,
        selected = selected,
        "p2c selection"
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

async fn addressed_router(endpoint: &Endpoint) -> anyhow::Result<Arc<AddressedPushRouter>> {
    // Get network manager and create client (no mode checks!)
    let manager = endpoint.drt().network_manager();
    let req_client = manager.create_client()?;
    let resp_transport = endpoint.drt().tcp_server().await?;

    tracing::debug!(
        transport = req_client.transport_name(),
        "Creating AddressedPushRouter with request plane client"
    );

    AddressedPushRouter::new(req_client, resp_transport)
}

impl<T, U> PushRouter<T, U>
where
    T: Data + Serialize,
    U: Data + for<'de> Deserialize<'de> + MaybeError,
{
    /// Create a new PushRouter without busy threshold (no busy detection)
    pub async fn from_client(client: Client, router_mode: RouterMode) -> anyhow::Result<Self> {
        Self::from_client_with_threshold(client, router_mode, None, None).await
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

        Ok(PushRouter {
            client,
            addressed,
            router_mode,
            round_robin_counter: Arc::new(AtomicU64::new(0)),
            busy_threshold: None,
            fault_detection_enabled: false,
            response_timeout: response_inactivity_timeout(),
            occupancy_state,
            _phantom: PhantomData,
        })
    }

    /// Create a new PushRouter with optional busy threshold and worker load monitor
    pub async fn from_client_with_threshold(
        client: Client,
        router_mode: RouterMode,
        busy_threshold: Option<f64>,
        worker_monitor: Option<Arc<dyn WorkerLoadMonitor>>,
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

        let router = PushRouter {
            client,
            addressed,
            router_mode,
            round_robin_counter: Arc::new(AtomicU64::new(0)),
            busy_threshold,
            fault_detection_enabled: true,
            response_timeout: response_inactivity_timeout(),
            occupancy_state,
            _phantom: PhantomData,
        };

        Ok(router)
    }

    /// Issue a request to the next available instance in a round-robin fashion
    pub async fn round_robin(&self, request: SingleIn<T>) -> anyhow::Result<ManyOut<U>> {
        let counter = self.round_robin_counter.fetch_add(1, Ordering::Relaxed) as usize;

        let instance_id = {
            let instance_ids = self.client.instance_ids_avail();
            let count = instance_ids.len();
            if count == 0 {
                return Err(anyhow::anyhow!(
                    "no instances found for endpoint {}",
                    self.client.endpoint.id()
                ));
            }
            instance_ids[counter % count]
        };
        tracing::trace!("round robin router selected {instance_id}");

        self.generate_with_fault_detection(instance_id, request)
            .await
    }

    /// Issue a request to a random endpoint
    pub async fn random(&self, request: SingleIn<T>) -> anyhow::Result<ManyOut<U>> {
        let instance_id = {
            let instance_ids = self.client.instance_ids_avail();
            let count = instance_ids.len();
            if count == 0 {
                return Err(anyhow::anyhow!(
                    "no instances found for endpoint {}",
                    self.client.endpoint.id()
                ));
            }
            let counter = rand::rng().random::<u64>() as usize;
            instance_ids[counter % count]
        };
        tracing::trace!("random router selected {instance_id}");

        self.generate_with_fault_detection(instance_id, request)
            .await
    }

    /// Issue a request using power-of-two-choices: pick 2 random healthy workers,
    /// route to the one with fewer in-flight requests.
    pub async fn power_of_two_choices(&self, request: SingleIn<T>) -> anyhow::Result<ManyOut<U>> {
        let state = self.occupancy_state()?;
        let instance_id = {
            let instance_ids = self
                .client
                .instance_ids_avail()
                .iter()
                .copied()
                .collect::<Vec<_>>();
            if instance_ids.is_empty() {
                return Err(anyhow::anyhow!(
                    "no instances found for endpoint {}",
                    self.client.endpoint.id()
                ));
            }
            p2c_select_from(state.as_ref(), &instance_ids)
        };
        state.increment(instance_id);
        let permit = OccupancyPermit::new(state, instance_id);

        match self
            .generate_with_fault_detection(instance_id, request)
            .await
        {
            Ok(stream) => Ok(permit.into_tracked_stream(stream)),
            Err(err) => Err(err),
        }
    }

    /// Issue a request to a specific endpoint
    pub async fn direct(
        &self,
        request: SingleIn<T>,
        instance_id: u64,
    ) -> anyhow::Result<ManyOut<U>> {
        // When fault detection is disabled, check the raw discovery list
        // (not filtered by report_instance_down) so transient failures
        // don't poison the instance for subsequent retries.
        let found = if self.fault_detection_enabled {
            self.client.instance_ids_avail().contains(&instance_id)
        } else {
            self.client.instance_ids().contains(&instance_id)
        };

        if !found {
            return Err(anyhow::anyhow!(
                "instance_id={instance_id} not found for endpoint {}",
                self.client.endpoint.id()
            ));
        }

        self.generate_with_fault_detection(instance_id, request)
            .await
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
        let state = self.occupancy_state()?;
        let instance_ids = self
            .client
            .instance_ids_avail()
            .iter()
            .copied()
            .collect::<Vec<_>>();

        if instance_ids.is_empty() {
            return Err(anyhow::anyhow!(
                "no instances found for endpoint {}",
                self.client.endpoint.id()
            ));
        }

        // Apply a unified policy for all endpoints.
        let endpoint_id = self.client.endpoint.id();

        // For encoder endpoints, partition by device type
        let instances = self.client.instances();
        let device_type_map: std::collections::HashMap<u64, Option<DeviceType>> = instances
            .iter()
            .map(|inst| (inst.instance_id, inst.device_type.clone()))
            .collect();

        // Apply budget-based routing to determine which group to send to
        let cuda_to_cpu_ratio = std::env::var("DYN_ENCODER_CUDA_TO_CPU_RATIO")
            .ok()
            .and_then(|v| v.parse::<usize>().ok())
            .filter(|v| *v >= 1)
            .unwrap_or(8);
        let candidates = device_aware_candidate_group(
            state.as_ref(),
            &instance_ids,
            &device_type_map,
            cuda_to_cpu_ratio,
        );

        // Select least-loaded within the chosen group
        let instance_id = state
            .select_exact_min_and_increment(&candidates)
            .await
            .ok_or_else(|| {
                anyhow::anyhow!(
                    "no instances in selected device group for endpoint {}",
                    endpoint_id
                )
            })?;
        let permit = OccupancyPermit::new(state.clone(), instance_id);
        let is_cpu = matches!(
            device_type_map.get(&instance_id),
            Some(Some(DeviceType::Cpu))
        );
        tracing::info!(
            endpoint = %endpoint_id,
            selected_instance = instance_id,
            is_cpu,
            "DeviceAwareWeighted selected instance"
        );

        match self
            .generate_with_fault_detection(instance_id, request)
            .await
        {
            Ok(stream) => Ok(permit.into_tracked_stream(stream)),
            Err(err) => Err(err),
        }
    }

    /// Issue a request to the instance with the fewest active connections.
    pub async fn least_loaded(&self, request: SingleIn<T>) -> anyhow::Result<ManyOut<U>> {
        let state = self.occupancy_state()?;
        let instance_ids = self
            .client
            .instance_ids_avail()
            .iter()
            .copied()
            .collect::<Vec<_>>();
        let instance_id = state
            .select_exact_min_and_increment(&instance_ids)
            .await
            .ok_or_else(|| {
                anyhow::anyhow!(
                    "no instances found for endpoint {}",
                    self.client.endpoint.id()
                )
            })?;
        let permit = OccupancyPermit::new(state.clone(), instance_id);
        tracing::trace!(
            "least loaded router selected {instance_id} (connections: {})",
            state.load(instance_id)
        );

        match self
            .generate_with_fault_detection(instance_id, request)
            .await
        {
            Ok(stream) => Ok(permit.into_tracked_stream(stream)),
            Err(err) => Err(err),
        }
    }

    /// Select the next worker according to the routing mode.
    /// Increments round-robin counter if applicable.
    /// Returns None for modes that require request lifecycle tracking or explicit routing hints.
    pub fn select_next_worker(&self) -> Option<u64> {
        let instance_ids = self.client.instance_ids_avail();
        let count = instance_ids.len();
        if count == 0 {
            return None;
        }

        match self.router_mode {
            RouterMode::RoundRobin => {
                let counter = self.round_robin_counter.fetch_add(1, Ordering::Relaxed) as usize;
                Some(instance_ids[counter % count])
            }
            RouterMode::Random => {
                let counter = rand::rng().random::<u64>() as usize;
                Some(instance_ids[counter % count])
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
    /// Returns None for modes that require request lifecycle tracking or explicit routing hints.
    pub fn peek_next_worker(&self) -> Option<u64> {
        let instance_ids = self.client.instance_ids_avail();
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
            RouterMode::PowerOfTwoChoices
            | RouterMode::Direct
            | RouterMode::LeastLoaded
            | RouterMode::DeviceAwareWeighted => None,
            RouterMode::KV => {
                panic!(
                    "peek_next_worker should not be called for {:?} routing mode",
                    self.router_mode
                )
            }
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
        mut instance_id: u64,
        request: SingleIn<T>,
    ) -> anyhow::Result<ManyOut<U>> {
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

        // Check if all workers are busy (only if busy threshold is set and fault detection enabled)
        if self.fault_detection_enabled && self.busy_threshold.is_some() {
            let free_instances = self.client.instance_ids_free();
            if free_instances.is_empty() {
                // Check if we actually have any instances at all
                let all_instances = self.client.instance_ids();
                if !all_instances.is_empty() {
                    tracing::warn!(
                        instance_id,
                        total_workers = all_instances.len(),
                        "Rejecting request: all workers are busy"
                    );
                    let cause = PipelineError::ServiceOverloaded(
                        "All workers are busy, please retry later".to_string(),
                    );
                    return Err(DynamoError::builder()
                        .error_type(ErrorType::ResourceExhausted)
                        .message("All workers are busy, please retry later")
                        .cause(cause)
                        .build()
                        .into());
                }
            }
        }

        // Get the address based on discovered transport type.
        // If the selected instance disappeared between selection and dispatch
        // (e.g. deregistered during scale-down), fall back to another available
        // instance rather than returning a spurious 500.
        let (address, _transport_kind) = {
            use crate::component::TransportType;

            let resolve_transport = |id: u64| {
                let instances = self.client.instances();
                instances
                    .iter()
                    .find(|i| i.instance_id == id)
                    .map(|instance| match &instance.transport {
                        TransportType::Http(http_endpoint) => {
                            tracing::debug!(
                                instance_id = id,
                                http_endpoint = %http_endpoint,
                                "Using HTTP transport for instance"
                            );
                            (http_endpoint.clone(), "transport.http.request")
                        }
                        TransportType::Tcp(tcp_endpoint) => {
                            tracing::debug!(
                                instance_id = id,
                                tcp_endpoint = %tcp_endpoint,
                                "Using TCP transport for instance"
                            );
                            (tcp_endpoint.clone(), "transport.tcp.request")
                        }
                        TransportType::Nats(subject) => {
                            tracing::debug!(
                                instance_id = id,
                                subject = %subject,
                                "Using NATS transport for instance"
                            );
                            (subject.clone(), "transport.nats.request")
                        }
                    })
            };

            if let Some(result) = resolve_transport(instance_id) {
                result
            } else {
                // Instance vanished — pick a different one from the current
                // availability list and retry the lookup once.
                let avail = self.client.instance_ids_avail();
                let fallback_id = avail.iter().copied().find(|&id| id != instance_id);
                match fallback_id {
                    Some(id) => {
                        tracing::warn!(
                            original_instance = instance_id,
                            fallback_instance = id,
                            "Instance disappeared during routing, reselecting"
                        );
                        instance_id = id;
                        resolve_transport(id).ok_or_else(|| {
                            anyhow::anyhow!(
                                "Fallback instance {} also not found for endpoint {}",
                                id,
                                self.client.endpoint.id()
                            )
                        })?
                    }
                    None => {
                        return Err(anyhow::anyhow!(
                            "Instance {} not found and no other instances available \
                             for endpoint {}",
                            instance_id,
                            self.client.endpoint.id()
                        ));
                    }
                }
            }
        };

        let request = request.map(|req| AddressedRequest::new(req, address));

        STAGE_DURATION_SECONDS
            .with_label_values(&["route"])
            .observe(route_start.elapsed().as_secs_f64());

        let _nvtx_transport = dynamo_nvtx_range!(_transport_kind);
        let stream: anyhow::Result<ManyOut<U>> = self
            .addressed
            .generate(request)
            .instrument(route_span)
            .await;
        match stream {
            Ok(stream) => {
                if !self.fault_detection_enabled {
                    return Ok(stream);
                }
                let engine_ctx = stream.context();
                let client = self.client.clone();
                let client_for_timeout = self.client.clone();
                let stream = stream.map(move |res| {
                    // Check if the error is migratable (indicates worker/connection failure)
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

                // Request-plane inactivity timeout: emit a ResponseTimeout error item
                // when the backend stops producing output. This triggers is_inhibited()
                // → report_instance_down() to quarantine the worker.
                let stream: Pin<Box<dyn Stream<Item = U> + Send>> = if let Some(timeout) =
                    self.response_timeout
                {
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
            Err(err) => {
                if self.fault_detection_enabled && is_inhibited(err.as_ref()) {
                    tracing::debug!("Reporting instance {instance_id} down due to error: {err}");
                    self.client.report_instance_down(instance_id);
                }
                Err(err)
            }
        }
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

struct OccupancyTrackedStream<U: Data> {
    inner: ManyOut<U>,
    state: Arc<RoutingOccupancyState>,
    instance_id: u64,
}

impl<U: Data> Drop for OccupancyTrackedStream<U> {
    fn drop(&mut self) {
        self.state.decrement(self.instance_id);
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
        self.inner.as_mut().poll_next(cx)
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
        pipeline::{ResponseStream, context::Controller},
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
    async fn least_loaded_select_and_peek_return_none_with_available_worker() {
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

        assert_eq!(router.select_next_worker(), None);
        assert_eq!(router.peek_next_worker(), None);

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
    async fn device_aware_weighted_select_and_peek_return_none_with_available_worker() {
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

        assert_eq!(router.select_next_worker(), None);
        assert_eq!(router.peek_next_worker(), None);

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
}
