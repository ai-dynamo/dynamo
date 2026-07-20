// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

use anyhow::Result;
use tokio::sync::mpsc;
use tokio_util::sync::CancellationToken;

use dynamo_kv_router::indexer::{KvIndexerMetrics, LocalKvIndexer};
use dynamo_kv_router::protocols::*;
pub use dynamo_kv_router::zmq_wire::create_stored_blocks;
#[cfg(test)]
use dynamo_kv_router::zmq_wire::*;
use dynamo_runtime::component::{Component, Endpoint};
use dynamo_runtime::discovery::{DiscoverySpec, EventScope};
use dynamo_runtime::protocols::EndpointId;
use dynamo_runtime::traits::DistributedRuntimeProvider;

use crate::discovery::KvEventSource as DiscoveredKvEventSource;
use crate::kv_router::{
    KV_EVENT_SUBJECT, WORKER_KV_INDEXER_BUFFER_SIZE, indexer::start_worker_kv_query_endpoint,
    metrics::KvPublisherMetrics,
};

mod batching;
mod dedup;
mod event_processor;
mod multimodal_embedding_cache;
mod sinks;
#[cfg(test)]
mod tests;
mod worker_metrics;
mod zmq_listener;

#[cfg(test)]
use batching::BatchingState;
#[cfg(test)]
use dedup::EventDedupFilter;
#[cfg(test)]
use event_processor::run_event_processor_loop;
use event_processor::start_event_processor;
pub use multimodal_embedding_cache::{
    MultimodalEmbeddingCacheEvent, MultimodalEmbeddingCachePublisher,
    MultimodalEmbeddingCacheUpdate,
};
use sinks::EventPlanePublisher;
pub use worker_metrics::WorkerMetricsPublisher;
use zmq_listener::start_zmq_listener;

const MAX_BATCHING_TIMEOUT_MS: u64 = 15_000;
pub const DEFAULT_BATCHING_TIMEOUT_MS: Option<u64> = None;
const DEFAULT_MAX_BATCH_BLOCKS: usize = 128;

/// Configure the source of KV events.
/// Currently, only ZMQ is supported.
pub enum KvEventSourceConfig {
    Zmq {
        endpoint: String,
        topic: String,
        /// Model image-placeholder token id, used by the normalizer to rewrite
        /// vLLM BlockStored events to the canonical pad_value scheme. `None`
        /// for text-only / non-MM deployments (normalization is a no-op).
        image_token_id: Option<u32>,
    },
}

enum KvEventSource {
    Zmq {
        zmq_handle: tokio::task::JoinHandle<()>,
    },
}

impl KvEventSource {
    fn start(
        component: Component,
        worker_id: WorkerId,
        kv_block_size: u32,
        source_config: KvEventSourceConfig,
        cancellation_token: CancellationToken,
        tx: mpsc::UnboundedSender<Vec<PlacementEvent>>,
        next_event_id: Arc<AtomicU64>,
    ) -> Result<Self> {
        match source_config {
            KvEventSourceConfig::Zmq {
                endpoint,
                topic,
                image_token_id,
            } => {
                let zmq_handle = component
                    .drt()
                    .runtime()
                    .secondary()
                    .spawn(start_zmq_listener(
                        endpoint,
                        topic,
                        worker_id,
                        tx,
                        cancellation_token.clone(),
                        kv_block_size,
                        next_event_id,
                        image_token_id,
                    ));

                Ok(KvEventSource::Zmq { zmq_handle })
            }
        }
    }

    fn shutdown(&self) {
        match self {
            KvEventSource::Zmq { zmq_handle } => {
                zmq_handle.abort();
            }
        }
    }
}

/// A publisher of KV events.
pub struct KvEventPublisher {
    /// The size of the KV block.
    kv_block_size: u32,
    /// The source of KV events.
    /// Can be `None` if all events provided through [`KvEventPublisher::publish`].
    source: Option<KvEventSource>,
    /// The cancellation token.
    cancellation_token: CancellationToken,
    /// The ID of the local worker emitting placement events.
    worker_id: WorkerId,
    /// The channel to send events to.
    tx: mpsc::UnboundedSender<Vec<PlacementEvent>>,
    /// Internal monotonic event ID counter. Shared with the ZMQ listener if present.
    next_event_id: Arc<AtomicU64>,
}

impl KvEventPublisher {
    pub fn new(
        endpoint: Endpoint,
        kv_block_size: u32,
        source_config: Option<KvEventSourceConfig>,
    ) -> Result<Self> {
        Self::new_with_local_indexer(
            endpoint,
            kv_block_size,
            source_config,
            false,
            0,
            DEFAULT_BATCHING_TIMEOUT_MS,
        )
    }

    pub fn new_with_local_indexer(
        endpoint: Endpoint,
        kv_block_size: u32,
        source_config: Option<KvEventSourceConfig>,
        enable_local_indexer: bool,
        dp_rank: DpRank,
        batching_timeout_ms: Option<u64>,
    ) -> Result<Self> {
        let kv_state_endpoint = endpoint.id();
        Self::new_with_local_indexer_at(
            endpoint,
            kv_state_endpoint,
            kv_block_size,
            source_config,
            enable_local_indexer,
            dp_rank,
            batching_timeout_ms,
        )
    }

    pub fn new_with_local_indexer_at(
        endpoint: Endpoint,
        kv_state_endpoint: EndpointId,
        kv_block_size: u32,
        source_config: Option<KvEventSourceConfig>,
        enable_local_indexer: bool,
        dp_rank: DpRank,
        batching_timeout_ms: Option<u64>,
    ) -> Result<Self> {
        Self::new_with_local_indexer_and_worker_id_at(
            endpoint,
            kv_state_endpoint,
            None,
            kv_block_size,
            source_config,
            enable_local_indexer,
            dp_rank,
            batching_timeout_ms,
        )
    }

    pub fn new_with_local_indexer_and_worker_id(
        endpoint: Endpoint,
        worker_id: Option<WorkerId>,
        kv_block_size: u32,
        source_config: Option<KvEventSourceConfig>,
        enable_local_indexer: bool,
        dp_rank: DpRank,
        batching_timeout_ms: Option<u64>,
    ) -> Result<Self> {
        let kv_state_endpoint = endpoint.id();
        Self::new_with_local_indexer_and_worker_id_at(
            endpoint,
            kv_state_endpoint,
            worker_id,
            kv_block_size,
            source_config,
            enable_local_indexer,
            dp_rank,
            batching_timeout_ms,
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub fn new_with_local_indexer_and_worker_id_at(
        endpoint: Endpoint,
        kv_state_endpoint: EndpointId,
        worker_id: Option<WorkerId>,
        kv_block_size: u32,
        source_config: Option<KvEventSourceConfig>,
        enable_local_indexer: bool,
        dp_rank: DpRank,
        batching_timeout_ms: Option<u64>,
    ) -> Result<Self> {
        let component = endpoint.component().clone();
        let cancellation_token = CancellationToken::new();
        let batching_timeout_ms = batching_timeout_ms
            .filter(|&ms| {
                if ms > MAX_BATCHING_TIMEOUT_MS {
                    tracing::warn!(
                        requested_ms = ms,
                        max_ms = MAX_BATCHING_TIMEOUT_MS,
                        "batching_timeout_ms too high, capping to 15s"
                    );
                }
                ms > 0
            })
            .map(|ms| ms.min(MAX_BATCHING_TIMEOUT_MS));

        let (tx, rx) = mpsc::unbounded_channel::<Vec<PlacementEvent>>();
        let worker_id = worker_id.unwrap_or_else(|| component.drt().connection_id());

        let _ = KvPublisherMetrics::from_component(&component);

        let endpoint_id = endpoint.id();
        tracing::info!(
            %kv_state_endpoint,
            "Initializing KvEventPublisher for worker {worker_id} on serving endpoint {endpoint_id}"
        );

        if enable_local_indexer {
            tracing::info!(
                "LocalKvIndexer enabled for worker {worker_id} on endpoint {endpoint_id}"
            );
        }

        let next_event_id = Arc::new(AtomicU64::new(0));

        let mut source = None;
        if let Some(config) = source_config {
            source = Some(KvEventSource::start(
                component.clone(),
                worker_id,
                kv_block_size,
                config,
                cancellation_token.clone(),
                tx.clone(),
                next_event_id.clone(),
            )?);
        }

        let local_indexer = if enable_local_indexer {
            let metrics = Arc::new(KvIndexerMetrics::new_unregistered());
            Some(Arc::new(LocalKvIndexer::new(
                cancellation_token.clone(),
                kv_block_size,
                metrics,
                WORKER_KV_INDEXER_BUFFER_SIZE,
            )))
        } else {
            None
        };

        let cancellation_token_clone = cancellation_token.clone();
        let local_indexer_clone = local_indexer.clone();

        tracing::info!("Using event plane for KV event publishing");
        let endpoint_clone = endpoint.clone();
        component.drt().runtime().secondary().spawn(async move {
            let event_publisher =
                match dynamo_runtime::transports::event_plane::EventPublisher::for_endpoint_id(
                    endpoint_clone.drt(),
                    &kv_state_endpoint,
                    KV_EVENT_SUBJECT,
                )
                .await
                {
                    Ok(publisher) => publisher,
                    Err(e) => {
                        tracing::error!("Failed to create event publisher: {}", e);
                        return;
                    }
                };
            let publisher_id = event_publisher.publisher_id();

            let recovery_endpoint = if let Some(local_indexer) = local_indexer_clone.as_ref() {
                match start_worker_kv_query_endpoint(
                    component.clone(),
                    publisher_id,
                    worker_id,
                    dp_rank,
                    local_indexer.clone(),
                )
                .await
                {
                    Ok(endpoint) => Some(endpoint),
                    Err(error) => {
                        tracing::error!(
                            %error,
                            worker_id,
                            dp_rank,
                            publisher_id,
                            "KV recovery endpoint failed; advertising a live-only KV source"
                        );
                        None
                    }
                }
            } else {
                None
            };

            let source = DiscoveredKvEventSource {
                kv_state_endpoint: kv_state_endpoint.clone(),
                worker: WorkerWithDpRank::new(worker_id, dp_rank),
                publisher_id,
                recovery_target: recovery_endpoint
                    .as_ref()
                    .map(|endpoint| endpoint.instance().clone()),
            };
            let source_spec = DiscoverySpec::EventSource {
                scope: EventScope::Endpoint {
                    endpoint: kv_state_endpoint.clone(),
                },
                topic: KV_EVENT_SUBJECT.to_string(),
                publisher_id,
                metadata: match serde_json::to_value(&source) {
                    Ok(metadata) => metadata,
                    Err(error) => {
                        tracing::error!(%error, "Failed to encode KV event source advertisement");
                        if let Some(endpoint) = recovery_endpoint {
                            let _ = endpoint.shutdown().await;
                        }
                        return;
                    }
                },
            };
            let source_instance = match component.drt().discovery().register(source_spec).await {
                Ok(instance) => instance,
                Err(error) => {
                    tracing::error!(%error, "Failed to advertise KV event source");
                    if let Some(endpoint) = recovery_endpoint {
                        let _ = endpoint.shutdown().await;
                    }
                    return;
                }
            };

            start_event_processor(
                EventPlanePublisher(event_publisher),
                worker_id,
                cancellation_token_clone,
                rx,
                local_indexer_clone,
                batching_timeout_ms,
            )
            .await;

            if let Err(error) = component
                .drt()
                .discovery()
                .unregister(source_instance)
                .await
            {
                tracing::warn!(%error, publisher_id, "Failed to unregister KV event source");
            }
            if let Some(endpoint) = recovery_endpoint
                && let Err(error) = endpoint.shutdown().await
            {
                tracing::warn!(%error, publisher_id, "Failed to stop KV recovery endpoint");
            }
        });

        Ok(Self {
            kv_block_size,
            source,
            cancellation_token,
            worker_id,
            tx,
            next_event_id,
        })
    }

    pub fn publish(&self, event: KvCacheEvent) -> Result<(), mpsc::error::SendError<KvCacheEvent>> {
        self.send_singleton(PlacementEvent::local_gpu(self.worker_id, event))
    }

    pub fn publish_with_storage_tier(
        &self,
        event: KvCacheEvent,
        storage_tier: StorageTier,
    ) -> Result<(), mpsc::error::SendError<KvCacheEvent>> {
        let placement_event = PlacementEvent::new(
            Placement::local_worker(self.worker_id, event.dp_rank, storage_tier),
            event,
        );
        self.send_singleton(placement_event)
    }

    fn send_singleton(
        &self,
        event: PlacementEvent,
    ) -> Result<(), mpsc::error::SendError<KvCacheEvent>> {
        self.tx.send(vec![event]).map_err(|err| {
            mpsc::error::SendError(
                err.0
                    .into_iter()
                    .next()
                    .expect("singleton publish returned an empty failed batch")
                    .event,
            )
        })
    }

    pub fn next_event_id(&self) -> u64 {
        self.next_event_id.fetch_add(1, Ordering::SeqCst)
    }

    pub fn kv_block_size(&self) -> u32 {
        self.kv_block_size
    }

    pub fn shutdown(&mut self) {
        if !self.cancellation_token.is_cancelled() {
            self.cancellation_token.cancel();
        }

        if let Some(source) = self.source.take() {
            source.shutdown();
        }
    }
}

impl Drop for KvEventPublisher {
    fn drop(&mut self) {
        self.shutdown();
    }
}
