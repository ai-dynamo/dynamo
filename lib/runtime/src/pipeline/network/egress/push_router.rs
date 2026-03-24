// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use super::{AsyncEngineContextProvider, ResponseStream};
use crate::error::{BackendError, ErrorType, match_error_chain};

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
    component::{Client, DeviceType, Endpoint},
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
use rand::Rng;
use serde::{Deserialize, Serialize};
use std::{
    collections::{HashMap, HashSet},
    env,
    future::Future,
    marker::PhantomData,
    sync::{
        Arc, Mutex,
        atomic::{AtomicU64, Ordering},
    },
    time::Instant,
};
use tokio_stream::StreamExt;
use tracing::Instrument;

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

    /// Best-effort in-flight accounting per instance for hetero-policy scheduling.
    inflight_by_instance: Arc<Mutex<HashMap<u64, usize>>>,

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
    /// Device-aware weighted routing for heterogeneous workers.
    DeviceAwareWeighted,
    Random,
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

#[derive(Debug, Clone)]
struct DeviceAwareWeightedConfig {
    cuda_to_cpu_ratio: usize,
}

fn device_aware_weighted_config() -> DeviceAwareWeightedConfig {
    let ratio = env::var("DYN_ENCODER_CUDA_TO_CPU_RATIO")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .filter(|v| *v >= 1)
        .unwrap_or(8);

    tracing::debug!(ratio = ratio, "Loaded DeviceAwareWeighted config");

    DeviceAwareWeightedConfig {
        cuda_to_cpu_ratio: ratio,
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

        Ok(PushRouter {
            client: client.clone(),
            addressed,
            router_mode,
            round_robin_counter: Arc::new(AtomicU64::new(0)),
            inflight_by_instance: Arc::new(Mutex::new(HashMap::new())),
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
            inflight_by_instance: Arc::new(Mutex::new(HashMap::new())),
            busy_threshold,
            fault_detection_enabled: true,
            _phantom: PhantomData,
        };

        Ok(router)
    }

    fn is_image_encode_endpoint(&self) -> bool {
        let endpoint_id = self.client.endpoint.id();
        endpoint_id.component == "encoder" && endpoint_id.name == "generate"
    }

    fn weight_for_instance(
        instance_id: u64,
        cfg: &DeviceAwareWeightedConfig,
        device_types: &HashMap<u64, Option<DeviceType>>,
    ) -> usize {
        if matches!(device_types.get(&instance_id), Some(Some(DeviceType::Cpu))) {
            1
        } else {
            cfg.cuda_to_cpu_ratio
        }
    }

    fn select_device_aware_weighted_instance(&self) -> anyhow::Result<u64> {
        let instance_ids = self.client.instance_ids_avail();
        let count = instance_ids.len();
        if count == 0 {
            anyhow::bail!(
                "no instances found for endpoint {}",
                self.client.endpoint.id()
            );
        }

        // Restrict device-aware weighted behavior to image encode endpoint only.
        if !self.is_image_encode_endpoint() {
            let counter = self.round_robin_counter.fetch_add(1, Ordering::Relaxed) as usize;
            tracing::trace!(
                endpoint = %self.client.endpoint.id(),
                selected_instance = instance_ids[counter % count],
                "DeviceAwareWeighted bypassed (non-encoder endpoint), using round-robin"
            );
            return Ok(instance_ids[counter % count]);
        }

        let cfg = device_aware_weighted_config();
        let device_types: HashMap<u64, Option<DeviceType>> = self
            .client
            .instances()
            .into_iter()
            .map(|inst| (inst.instance_id, inst.device_type))
            .collect();

        let cpu_count = instance_ids
            .iter()
            .filter(|id| matches!(device_types.get(id), Some(Some(DeviceType::Cpu))))
            .count();

        if cpu_count == 0 {
            tracing::warn!(
                endpoint = %self.client.endpoint.id(),
                ratio = cfg.cuda_to_cpu_ratio,
                "No CPU encode instances found from discovery metadata; DeviceAwareWeighted will treat all instances as CUDA"
            );
        }
        let mut inflight = self
            .inflight_by_instance
            .lock()
            .map_err(|_| anyhow::anyhow!("inflight lock poisoned"))?;

        let active_set = instance_ids.iter().copied().collect::<HashSet<_>>();
        inflight.retain(|id, _| active_set.contains(id));
        for id in instance_ids.iter() {
            inflight.entry(*id).or_insert(0);
        }

        let start = self.round_robin_counter.fetch_add(1, Ordering::Relaxed) as usize % count;

        let mut best_instance = instance_ids[start];
        let mut best_inflight = *inflight.get(&best_instance).unwrap_or(&0);
        let mut best_weight = Self::weight_for_instance(best_instance, &cfg, &device_types);

        tracing::debug!(
            endpoint = %self.client.endpoint.id(),
            start_index = start,
            candidates = ?instance_ids,
            inflight = ?*inflight,
            device_types = ?device_types,
            "DeviceAwareWeighted selecting instance"
        );

        for offset in 1..count {
            let idx = (start + offset) % count;
            let candidate = instance_ids[idx];
            let candidate_inflight = *inflight.get(&candidate).unwrap_or(&0);
            let candidate_weight = Self::weight_for_instance(candidate, &cfg, &device_types);

            // Compare load ratios without float: inflight / weight.
            let left = candidate_inflight.saturating_mul(best_weight);
            let right = best_inflight.saturating_mul(candidate_weight);
            tracing::trace!(
                candidate_instance = candidate,
                candidate_inflight = candidate_inflight,
                candidate_weight = candidate_weight,
                best_instance = best_instance,
                best_inflight = best_inflight,
                best_weight = best_weight,
                left = left,
                right = right,
                "DeviceAwareWeighted compare candidate vs best"
            );
            if left < right {
                best_instance = candidate;
                best_inflight = candidate_inflight;
                best_weight = candidate_weight;
            }
        }

        *inflight.entry(best_instance).or_insert(0) += 1;
        let selected_inflight = *inflight.get(&best_instance).unwrap_or(&0);
        tracing::info!(
            endpoint = %self.client.endpoint.id(),
            selected_instance = best_instance,
            selected_weight = best_weight,
            selected_inflight = selected_inflight,
            "DeviceAwareWeighted selected instance"
        );
        Ok(best_instance)
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
            RouterMode::DeviceAwareWeighted => {
                let counter = self.round_robin_counter.fetch_add(1, Ordering::Relaxed) as usize;
                Some(instance_ids[counter % count])
            }
            RouterMode::Random => {
                let counter = rand::rng().random::<u64>() as usize;
                Some(instance_ids[counter % count])
            }
            RouterMode::Direct => None,
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
            RouterMode::DeviceAwareWeighted => {
                // Keep parity with select_next_worker until policy-aware scoring lands.
                let counter = self.round_robin_counter.load(Ordering::Relaxed) as usize;
                Some(instance_ids[counter % count])
            }
            RouterMode::Random => {
                // For random, peeking implies a fresh random selection since it's stateless.
                // Note: The caller must realize that select_next_worker() will pick a DIFFERENT random worker.
                let counter = rand::rng().random::<u64>() as usize;
                Some(instance_ids[counter % count])
            }
            RouterMode::Direct => None,
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
                    return Err(PipelineError::ServiceOverloaded(
                        "All workers are busy, please retry later".to_string(),
                    )
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
            RouterMode::DeviceAwareWeighted => {
                let instance_id = self.select_device_aware_weighted_instance()?;
                let stream = self
                    .generate_with_fault_detection(instance_id, request)
                    .await?;

                // Track in-flight lifecycle only for image encode path.
                if self.is_image_encode_endpoint() {
                    let ctx = stream.context();
                    let inflight_clone = self.inflight_by_instance.clone();

                    let wrapped = async_stream::stream! {
                        let mut first = true;
                        let mut inner = stream;
                        while let Some(item) = inner.next().await {
                            if first {
                                // Encoder completed and returned first response data.
                                // Decrement inflight immediately, not waiting for downstream consumption.
                                first = false;
                                if let Ok(mut inflight) = inflight_clone.lock() {
                                    let value = inflight.entry(instance_id).or_insert(0);
                                    if *value > 0 {
                                        *value -= 1;
                                    }
                                    tracing::debug!(
                                        instance_id = instance_id,
                                        inflight_after_decrement = *value,
                                        "DeviceAwareWeighted inflight decremented on first encoder response"
                                    );
                                }
                            }
                            yield item;
                        }
                    };

                    Ok(ResponseStream::new(Box::pin(wrapped), ctx))
                } else {
                    Ok(stream)
                }
            }
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
