// SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use super::{AsyncEngineContextProvider, ResponseStream};
use crate::traits::events::EventSubscriber;
use crate::{
    component::{Client, Endpoint, InstanceSource},
    engine::{AsyncEngine, Data},
    pipeline::{
        error::{PipelineError, PipelineErrorExt},
        AddressedPushRouter, AddressedRequest, Error, ManyOut, SingleIn,
    },
    protocols::maybe_error::MaybeError,
    traits::DistributedRuntimeProvider,
};
use async_nats::client::{
    RequestError as NatsRequestError, RequestErrorKind::NoResponders as NatsNoResponders,
};
use async_trait::async_trait;
use rand::Rng;
use serde::{Deserialize, Serialize};
use std::{
    collections::HashMap,
    future::Future,
    marker::PhantomData,
    sync::{
        atomic::{AtomicU64, Ordering},
        Arc,
    },
};
use tokio::sync::{watch, RwLock};
use tokio_stream::StreamExt;

/// Worker load monitoring state
#[derive(Clone, Debug)]
struct WorkerLoadState {
    kv_active_blocks: Option<u64>,
    kv_total_blocks: Option<u64>,
}

impl WorkerLoadState {
    fn is_busy(&self, threshold: f64) -> bool {
        match (self.kv_active_blocks, self.kv_total_blocks) {
            (Some(active), Some(total)) if total > 0 => {
                (active as f64) > (threshold * total as f64)
            }
            _ => false,
        }
    }
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

    /// Worker load states for monitoring KV cache usage
    worker_load_states: Arc<RwLock<HashMap<i64, WorkerLoadState>>>,

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
    Direct(i64),
    // Marker value, KV routing itself is in dynamo-llm
    KV,
}

impl RouterMode {
    pub fn is_kv_routing(&self) -> bool {
        *self == RouterMode::KV
    }
}

async fn addressed_router(endpoint: &Endpoint) -> anyhow::Result<Arc<AddressedPushRouter>> {
    AddressedPushRouter::new(
        endpoint.drt().nats_client.client().clone(),
        endpoint.drt().tcp_server().await?,
    )
}

impl<T, U> PushRouter<T, U>
where
    T: Data + Serialize,
    U: Data + for<'de> Deserialize<'de> + MaybeError,
{
    pub async fn from_client(client: Client, router_mode: RouterMode) -> anyhow::Result<Self> {
        let addressed = addressed_router(&client.endpoint).await?;
        let worker_load_states = Arc::new(RwLock::new(HashMap::new()));

        let router = PushRouter {
            client: client.clone(),
            addressed,
            router_mode,
            round_robin_counter: Arc::new(AtomicU64::new(0)),
            worker_load_states: worker_load_states.clone(),
            _phantom: PhantomData,
        };

        // Start background monitoring if in dynamic mode
        if let InstanceSource::Dynamic(_) = client.instance_source.as_ref() {
            router.start_worker_monitoring().await?;
        }

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
                    "no instances found for endpoint {:?}",
                    self.client.endpoint.etcd_root()
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
                    "no instances found for endpoint {:?}",
                    self.client.endpoint.etcd_root()
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
        instance_id: i64,
    ) -> anyhow::Result<ManyOut<U>> {
        let found = self.client.instance_ids_avail().contains(&instance_id);

        if !found {
            return Err(anyhow::anyhow!(
                "instance_id={instance_id} not found for endpoint {:?}",
                self.client.endpoint.etcd_root()
            ));
        }

        self.generate_with_fault_detection(instance_id, request)
            .await
    }

    pub async fn r#static(&self, request: SingleIn<T>) -> anyhow::Result<ManyOut<U>> {
        let subject = self.client.endpoint.subject();
        tracing::debug!("static got subject: {subject}");
        let request = request.map(|req| AddressedRequest::new(req, subject));
        tracing::debug!("router generate");
        self.addressed.generate(request).await
    }

    async fn generate_with_fault_detection(
        &self,
        instance_id: i64,
        request: SingleIn<T>,
    ) -> anyhow::Result<ManyOut<U>> {
        // Check if all workers are busy
        let free_instances = self.client.instance_ids_free();
        if free_instances.is_empty() {
            // Check if we actually have any instances at all
            let all_instances = self.client.instance_ids();
            if !all_instances.is_empty() {
                return Err(PipelineError::ServiceOverloaded(
                    "All workers are busy, please retry later".to_string(),
                )
                .into());
            }
        }

        let subject = self.client.endpoint.subject_to(instance_id);
        let request = request.map(|req| AddressedRequest::new(req, subject));

        let stream: anyhow::Result<ManyOut<U>> = self.addressed.generate(request).await;
        match stream {
            Ok(stream) => {
                let engine_ctx = stream.context();
                let client = self.client.clone();
                let stream = stream.map(move |res| {
                    if let Some(err) = res.err() {
                        const STREAM_ERR_MSG: &str = "Stream ended before generation completed";
                        if format!("{:?}", err) == STREAM_ERR_MSG {
                            client.report_instance_down(instance_id);
                        }
                    }
                    res
                });
                Ok(ResponseStream::new(Box::pin(stream), engine_ctx))
            }
            Err(err) => {
                if let Some(req_err) = err.downcast_ref::<NatsRequestError>() {
                    if matches!(req_err.kind(), NatsNoResponders) {
                        self.client.report_instance_down(instance_id);
                    }
                }
                Err(err)
            }
        }
    }

    /// Start background monitoring of worker KV cache usage
    async fn start_worker_monitoring(&self) -> anyhow::Result<()> {
        // Constants
        const KV_METRICS_SUBJECT: &str = "kv_metrics";
        const MODEL_ROOT_PATH: &str = "models";
        const BUSY_THRESHOLD: f64 = 0.95;

        #[derive(serde::Deserialize)]
        struct LoadEvent {
            worker_id: i64,
            data: ForwardPassMetrics,
        }

        #[derive(serde::Deserialize)]
        struct ForwardPassMetrics {
            kv_stats: KvStats,
        }

        #[derive(serde::Deserialize)]
        struct KvStats {
            kv_active_blocks: u64,
        }

        #[derive(serde::Deserialize)]
        struct ModelEntry {
            runtime_config: Option<RuntimeConfig>,
        }

        #[derive(serde::Deserialize)]
        struct RuntimeConfig {
            total_kv_blocks: Option<u64>,
        }

        let endpoint = &self.client.endpoint;
        let component = endpoint.component();

        let Some(etcd_client) = component.drt().etcd_client() else {
            // Static mode, no monitoring needed
            return Ok(());
        };

        // Use the generic etcd watcher to watch model runtime configs
        use crate::utils::typed_prefix_watcher::{key_extractors, watch_prefix_with_extraction};

        let runtime_configs_watcher = watch_prefix_with_extraction(
            etcd_client,
            MODEL_ROOT_PATH,
            key_extractors::lease_id,
            |entry: ModelEntry| entry.runtime_config.and_then(|rc| rc.total_kv_blocks),
            component.drt().child_token(),
        )
        .await?;
        let mut config_events_rx = runtime_configs_watcher.receiver();

        // Subscribe to KV metrics events
        let mut kv_metrics_rx = component.subscribe(KV_METRICS_SUBJECT).await?;

        let worker_load_states = self.worker_load_states.clone();
        let client = self.client.clone();
        let cancellation_token = component.drt().child_token();

        // Spawn background monitoring task
        tokio::spawn(async move {
            loop {
                tokio::select! {
                    _ = cancellation_token.cancelled() => {
                        tracing::debug!("Worker monitoring cancelled");
                        break;
                    }

                    // Handle runtime config updates - now receives full HashMap
                    _ = config_events_rx.changed() => {
                        let runtime_configs = config_events_rx.borrow().clone();

                        let mut states = worker_load_states.write().await;
                        states.retain(|lease_id, _| runtime_configs.contains_key(lease_id));

                        // Update worker load states with total blocks
                        for (lease_id, total_blocks) in runtime_configs.iter() {
                            let state = states.entry(*lease_id).or_insert(WorkerLoadState {
                                kv_active_blocks: None,
                                kv_total_blocks: None,
                            });
                            state.kv_total_blocks = Some(*total_blocks);
                        }
                    }

                    // Handle KV metrics updates
                    kv_event = kv_metrics_rx.next() => {
                        let Some(event) = kv_event else {
                            tracing::debug!("KV metrics stream closed");
                            break;
                        };

                        if let Ok(load_event) = serde_json::from_slice::<LoadEvent>(&event.payload) {
                            let worker_id = load_event.worker_id;
                            let active_blocks = load_event.data.kv_stats.kv_active_blocks;

                            // Update worker load state
                            let mut states = worker_load_states.write().await;
                            let state = states.entry(worker_id).or_insert(WorkerLoadState {
                                kv_active_blocks: None,
                                kv_total_blocks: None,
                            });
                            state.kv_active_blocks = Some(active_blocks);
                            drop(states);

                            // Recalculate all busy instances and update
                            let states = worker_load_states.read().await;
                            let busy_instances: Vec<i64> = states
                                .iter()
                                .filter_map(|(&id, state)| {
                                    state.is_busy(BUSY_THRESHOLD).then_some(id)
                                })
                                .collect();
                            drop(states);

                            client.update_free_instances(&busy_instances);
                        }
                    }
                }
            }

            tracing::info!("Worker monitoring task exiting");
        });

        Ok(())
    }
}

#[async_trait]
impl<T, U> AsyncEngine<SingleIn<T>, ManyOut<U>, Error> for PushRouter<T, U>
where
    T: Data + Serialize,
    U: Data + for<'de> Deserialize<'de> + MaybeError,
{
    async fn generate(&self, request: SingleIn<T>) -> Result<ManyOut<U>, Error> {
        match self.client.instance_source.as_ref() {
            InstanceSource::Static => self.r#static(request).await,
            InstanceSource::Dynamic(_) => match self.router_mode {
                RouterMode::Random => self.random(request).await,
                RouterMode::RoundRobin => self.round_robin(request).await,
                RouterMode::Direct(instance_id) => self.direct(request, instance_id).await,
                RouterMode::KV => {
                    anyhow::bail!("KV routing should not call generate on PushRouter");
                }
            },
        }
    }
}
