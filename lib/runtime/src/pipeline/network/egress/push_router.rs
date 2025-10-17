// SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use super::{AsyncEngineContextProvider, ResponseStream, STREAM_ERR_MSG};
use crate::{
    component::{Client, Endpoint, InstanceSource},
    engine::{AsyncEngine, Data},
    pipeline::{
        AddressedPushRouter, AddressedRequest, Error, ManyOut, SingleIn,
        error::{PipelineError, PipelineErrorExt},
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
    future::Future,
    marker::PhantomData,
    sync::{
        Arc,
        atomic::{AtomicU64, Ordering},
    },
};
use tokio_stream::StreamExt;

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
    let Some(nats_client) = endpoint.drt().nats_client() else {
        anyhow::bail!("Missing NATS. Please ensure it is running and accessible.");
    };
    AddressedPushRouter::new(
        nats_client.client().clone(),
        endpoint.drt().tcp_server().await?,
    )
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

    /// Create a new PushRouter with optional busy threshold and worker load monitor
    pub async fn from_client_with_threshold(
        client: Client,
        router_mode: RouterMode,
        busy_threshold: Option<f64>,
        worker_monitor: Option<Arc<dyn WorkerLoadMonitor>>,
    ) -> anyhow::Result<Self> {
        let addressed = addressed_router(&client.endpoint).await?;

        // Start worker monitor if provided and in dynamic mode
        if let Some(monitor) = worker_monitor.as_ref()
            && matches!(client.instance_source.as_ref(), InstanceSource::Dynamic(_))
        {
            monitor.start_monitoring().await?;
        }

        let router = PushRouter {
            client: client.clone(),
            addressed,
            router_mode,
            round_robin_counter: Arc::new(AtomicU64::new(0)),
            busy_threshold,
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
        // Check if all workers are busy (only if busy threshold is set)
        if self.busy_threshold.is_some() {
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
        }

        let subject = self.client.endpoint.subject_to(instance_id);
        let request = request.map(|req| AddressedRequest::new(req, subject));

        let stream: anyhow::Result<ManyOut<U>> = self.addressed.generate(request).await;
        match stream {
            Ok(stream) => {
                let engine_ctx = stream.context();
                let client = self.client.clone();
                let stream = stream.map(move |res| {
                    // TODO: Standardize error type to avoid using string matching DIS-364
                    if let Some(err) = res.err()
                        && format!("{:?}", err) == STREAM_ERR_MSG
                    {
                        tracing::debug!(
                            "Reporting instance {instance_id} down due to stream error: {err}"
                        );
                        client.report_instance_down(instance_id);
                    }
                    res
                });
                Ok(ResponseStream::new(Box::pin(stream), engine_ctx))
            }
            Err(err) => {
                if let Some(req_err) = err.downcast_ref::<NatsRequestError>()
                    && matches!(req_err.kind(), NatsNoResponders)
                {
                    tracing::debug!(
                        "Reporting instance {instance_id} down due to request error: {req_err}"
                    );
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
