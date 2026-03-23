// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use super::{AsyncEngineContextProvider, ResponseStream};
use crate::error::{BackendError, DynamoError, ErrorType, match_error_chain};

/// Check if an error chain indicates the worker should be reported as down.
fn is_inhibited(err: &(dyn std::error::Error + 'static)) -> bool {
    const INHIBITED: &[ErrorType] = &[
        ErrorType::CannotConnect,
        ErrorType::Disconnected,
        ErrorType::ConnectionTimeout,
        ErrorType::Backend(BackendError::EngineShutdown),
    ];
    match_error_chain(err, INHIBITED, &[])
}
use crate::{
    component::{Client, Endpoint},
    dynamo_nvtx_range,
    engine::{AsyncEngine, Data},
    metrics::frontend_perf::STAGE_DURATION_SECONDS,
    pipeline::{
        AddressedPushRouter, AddressedRequest, Error, ManyOut, SingleIn,
        error::{PipelineError, PipelineErrorExt},
    },
    protocols::maybe_error::MaybeError,
    traits::DistributedRuntimeProvider,
};
use async_trait::async_trait;
use dashmap::DashMap;
use rand::Rng;
use serde::{Deserialize, Serialize};
use std::{
    future::Future,
    marker::PhantomData,
    sync::{
        Arc,
        atomic::{AtomicU64, Ordering},
    },
    time::Instant,
};
use tokio_stream::StreamExt;
use tracing::Instrument;

/// RAII guard that decrements a per-instance in-flight counter on drop.
/// Used by PowerOfTwoChoices routing to track request occupancy.
struct P2CGuard {
    in_flight_counts: Arc<DashMap<u64, AtomicU64>>,
    instance_id: u64,
}

impl Drop for P2CGuard {
    fn drop(&mut self) {
        if let Some(counter) = self.in_flight_counts.get(&self.instance_id) {
            counter.value().fetch_sub(1, Ordering::Relaxed);
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

    /// Per-instance in-flight request counts for PowerOfTwoChoices routing.
    in_flight_counts: Arc<DashMap<u64, AtomicU64>>,

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
fn p2c_select_from(in_flight_counts: &DashMap<u64, AtomicU64>, instance_ids: &[u64]) -> u64 {
    let count = instance_ids.len();
    if count == 1 {
        return instance_ids[0];
    }
    let mut rng = rand::rng();
    let idx1 = rng.random_range(0..count);
    let idx2 = (idx1 + 1 + rng.random_range(0..count - 1)) % count;
    let id1 = instance_ids[idx1];
    let id2 = instance_ids[idx2];
    let load1 = in_flight_counts
        .get(&id1)
        .map(|c| c.value().load(Ordering::Relaxed))
        .unwrap_or(0);
    let load2 = in_flight_counts
        .get(&id2)
        .map(|c| c.value().load(Ordering::Relaxed))
        .unwrap_or(0);
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

        Ok(PushRouter {
            client: client.clone(),
            addressed,
            router_mode,
            round_robin_counter: Arc::new(AtomicU64::new(0)),
            in_flight_counts: Arc::new(DashMap::new()),
            busy_threshold: None,
            fault_detection_enabled: false,
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

        let router = PushRouter {
            client: client.clone(),
            addressed,
            router_mode,
            round_robin_counter: Arc::new(AtomicU64::new(0)),
            in_flight_counts: Arc::new(DashMap::new()),
            busy_threshold,
            fault_detection_enabled: true,
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
        let instance_id = {
            let instance_ids = self.client.instance_ids_avail();
            if instance_ids.is_empty() {
                return Err(anyhow::anyhow!(
                    "no instances found for endpoint {}",
                    self.client.endpoint.id()
                ));
            }
            p2c_select_from(&self.in_flight_counts, &instance_ids)
        };
        // Guard created before the await so error paths also decrement.
        self.in_flight_counts
            .entry(instance_id)
            .or_insert_with(|| AtomicU64::new(0))
            .value()
            .fetch_add(1, Ordering::Relaxed);
        let guard = P2CGuard {
            in_flight_counts: self.in_flight_counts.clone(),
            instance_id,
        };

        let stream = self
            .generate_with_fault_detection(instance_id, request)
            .await?;
        let engine_ctx = stream.context();
        let stream = stream.map(move |res| {
            let _guard = &guard;
            res
        });
        Ok(ResponseStream::new(Box::pin(stream), engine_ctx))
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

    /// Select the next worker according to the routing mode.
    /// Increments round-robin counter if applicable.
    /// Returns None for Direct mode - requires explicit worker IDs via routing hints
    /// Panics for KV mode which has its own selection via find_best_match.
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
            // P2C needs lifecycle tracking (P2CGuard); use generate() instead.
            RouterMode::PowerOfTwoChoices | RouterMode::Direct => None,
            _ => {
                panic!(
                    "select_next_worker should not be called for {:?} routing mode",
                    self.router_mode
                )
            }
        }
    }

    /// Peek the next worker according to the routing mode without incrementing the counter.
    /// Useful for checking if a worker is suitable before committing to it.
    /// Returns None for Direct mode - requires explicit worker IDs via routing hints.
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
            // P2C needs lifecycle tracking (P2CGuard); use generate() instead.
            RouterMode::PowerOfTwoChoices | RouterMode::Direct => None,
            _ => {
                panic!(
                    "peek_next_worker should not be called for {:?} routing mode",
                    self.router_mode
                )
            }
        }
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

        // Get the address based on discovered transport type
        let (address, _transport_kind) = {
            use crate::component::TransportType;

            // Get the instance and use its actual transport type
            let instances = self.client.instances();
            let instance = instances
                .iter()
                .find(|i| i.instance_id == instance_id)
                .ok_or_else(|| {
                    anyhow::anyhow!("Instance {} not found in available instances", instance_id)
                })?;

            match &instance.transport {
                TransportType::Http(http_endpoint) => {
                    tracing::debug!(
                        instance_id = instance_id,
                        http_endpoint = %http_endpoint,
                        "Using HTTP transport for instance"
                    );
                    (http_endpoint.clone(), "transport.http.request")
                }
                TransportType::Tcp(tcp_endpoint) => {
                    tracing::debug!(
                        instance_id = instance_id,
                        tcp_endpoint = %tcp_endpoint,
                        "Using TCP transport for instance"
                    );
                    (tcp_endpoint.clone(), "transport.tcp.request")
                }
                TransportType::Nats(subject) => {
                    tracing::debug!(
                        instance_id = instance_id,
                        subject = %subject,
                        "Using NATS transport for instance"
                    );
                    (subject.clone(), "transport.nats.request")
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
                Ok(ResponseStream::new(Box::pin(stream), engine_ctx))
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
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn p2c_selects_lower_load_worker() {
        let counts = DashMap::new();
        counts.insert(1, AtomicU64::new(10));
        counts.insert(2, AtomicU64::new(1));

        // With only two workers, p2c_select_from must pick both and choose id=2 (lower load).
        let result = p2c_select_from(&counts, &[1, 2]);
        assert_eq!(result, 2);
    }

    #[test]
    fn p2c_selects_single_worker() {
        let counts = DashMap::new();
        assert_eq!(p2c_select_from(&counts, &[42]), 42);
    }

    #[test]
    fn p2c_treats_missing_counts_as_zero() {
        let counts = DashMap::new();
        counts.insert(1, AtomicU64::new(5));
        // Worker 2 has no entry — should be treated as 0, so it wins.
        let result = p2c_select_from(&counts, &[1, 2]);
        assert_eq!(result, 2);
    }

    #[test]
    fn p2c_returns_valid_worker_on_tie() {
        let counts = DashMap::new();
        counts.insert(1, AtomicU64::new(3));
        counts.insert(2, AtomicU64::new(3));

        for _ in 0..100 {
            let result = p2c_select_from(&counts, &[1, 2]);
            assert!(result == 1 || result == 2);
        }
    }

    #[test]
    fn p2c_lifecycle_tracks_inflight_counts() {
        let counts = Arc::new(DashMap::new());
        let mut guards = Vec::new();
        for _ in 0..5 {
            let selected = p2c_select_from(&counts, &[1, 2]);
            counts
                .entry(selected)
                .or_insert_with(|| AtomicU64::new(0))
                .value()
                .fetch_add(1, Ordering::Relaxed);
            guards.push(P2CGuard {
                in_flight_counts: counts.clone(),
                instance_id: selected,
            });
        }

        let total = counts
            .get(&1)
            .map(|c| c.value().load(Ordering::Relaxed))
            .unwrap_or(0)
            + counts
                .get(&2)
                .map(|c| c.value().load(Ordering::Relaxed))
                .unwrap_or(0);
        assert_eq!(total, 5, "5 in-flight requests should be tracked");

        drop(guards);
        let total = counts
            .get(&1)
            .map(|c| c.value().load(Ordering::Relaxed))
            .unwrap_or(0)
            + counts
                .get(&2)
                .map(|c| c.value().load(Ordering::Relaxed))
                .unwrap_or(0);
        assert_eq!(total, 0, "All guards dropped, counts should be 0");
    }

    #[test]
    fn p2c_never_selects_dominated_worker() {
        let counts = DashMap::new();
        counts.insert(1, AtomicU64::new(0));
        counts.insert(2, AtomicU64::new(0));
        counts.insert(3, AtomicU64::new(100));

        let mut selected = [0u32; 3];
        for _ in 0..1000 {
            let result = p2c_select_from(&counts, &[1, 2, 3]);
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
}
