// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::sync::Arc;
use std::sync::OnceLock;
use std::sync::atomic::{AtomicU64, Ordering};

use anyhow::Result;
use tokio::sync::mpsc;
use tokio_util::sync::CancellationToken;

use dynamo_kv_router::indexer::{KvIndexerMetrics, LocalKvIndexer};
use dynamo_kv_router::protocols::*;
pub use dynamo_kv_router::zmq_wire::create_stored_blocks;
#[cfg(test)]
use dynamo_kv_router::zmq_wire::*;
use dynamo_runtime::config::environment_names::nats as env_nats;
use dynamo_runtime::metrics::MetricsHierarchy;
use dynamo_runtime::metrics::prometheus_names::kv_publisher;
use dynamo_runtime::traits::DistributedRuntimeProvider;
use dynamo_runtime::{
    component::Component,
    transports::nats::{NatsQueue, Slug},
};

use crate::kv_router::{
    KV_EVENT_SUBJECT, WORKER_KV_INDEXER_BUFFER_SIZE, worker_query::start_worker_kv_query_endpoint,
};

mod event_processor;
#[cfg(test)]
mod tests;
mod worker_metrics;
mod zmq_listener;

#[cfg(test)]
use event_processor::{BatchingState, run_event_processor_loop};
use event_processor::{
    EventPlanePublisher, start_event_processor, start_event_processor_jetstream,
};
pub use worker_metrics::WorkerMetricsPublisher;
use zmq_listener::start_zmq_listener;
#[cfg(test)]
use zmq_listener::{
    INITIAL_BACKOFF_MS, MAX_BACKOFF_EXPONENT, MAX_BACKOFF_MS, MAX_CONSECUTIVE_ERRORS,
    calculate_backoff_ms,
};

const MAX_BATCHING_TIMEOUT_MS: u64 = 15_000;
pub const DEFAULT_BATCHING_TIMEOUT_MS: Option<u64> = None;
const DEFAULT_MAX_BATCH_BLOCKS: usize = 128;

/// Helper function to create a KV stream name from a component and subject.
///
/// Generates a slugified stream name in the format:
/// `namespace-{namespace}-component-{component}-{subject}`
fn create_kv_stream_name(component: &Component, subject: &str) -> String {
    Slug::slugify(&format!(
        "namespace.{}.component.{}.{}",
        component.namespace().name(),
        component.name(),
        subject
    ))
    .to_string()
    .replace("_", "-")
}

/// Metrics for the KV publisher, created via the MetricsHierarchy API.
/// This provides automatic `dynamo_namespace`, `dynamo_component`, and other
/// hierarchy labels for free.
pub(super) struct KvPublisherMetrics {
    /// Total number of raw events dropped by engines before reaching publisher
    pub engines_dropped_events_total: prometheus::IntCounterVec,
}

static KV_PUBLISHER_METRICS: OnceLock<Arc<KvPublisherMetrics>> = OnceLock::new();

impl KvPublisherMetrics {
    /// Create from a Component, memoized in a static OnceLock.
    /// Uses the MetricsHierarchy API which auto-prepends `dynamo_component_`,
    /// injects hierarchy labels, and registers with the DRT `MetricsRegistry`.
    pub fn from_component(component: &Component) -> Arc<Self> {
        KV_PUBLISHER_METRICS
            .get_or_init(|| {
                let metrics = component.metrics();
                match metrics.create_intcountervec(
                    kv_publisher::ENGINES_DROPPED_EVENTS_TOTAL,
                    "Total number of raw events dropped by engines before reaching publisher (detected via event_id gaps)",
                    &["worker_id"],
                    &[],
                ) {
                    Ok(engines_dropped_events_total) => {
                        Arc::new(Self { engines_dropped_events_total })
                    }
                    Err(e) => {
                        tracing::warn!("Failed to create kv_publisher metrics from component: {}. Using unregistered metrics as fallback.", e);
                        Arc::new(Self::new_unregistered())
                    }
                }
            })
            .clone()
    }

    /// Creates unregistered metrics for use when the MetricsRegistry is not available.
    /// This is used as a fallback when metric creation fails.
    pub fn new_unregistered() -> Self {
        Self {
            engines_dropped_events_total: prometheus::IntCounterVec::new(
                prometheus::Opts::new(
                    kv_publisher::ENGINES_DROPPED_EVENTS_TOTAL,
                    "Total number of raw events dropped by engines before reaching publisher (detected via event_id gaps)",
                ),
                &["worker_id"],
            )
            .expect("failed to create engines_dropped_events_total counter"),
        }
    }

    /// Increment the engines dropped events counter by the given amount.
    pub fn increment_engines_dropped_events(&self, worker_id: u64, count: u64) {
        self.engines_dropped_events_total
            .with_label_values(&[&worker_id.to_string()])
            .inc_by(count);
    }
}

fn kv_publisher_metrics() -> Option<Arc<KvPublisherMetrics>> {
    KV_PUBLISHER_METRICS.get().cloned()
}

/// Configure the source of KV events.
/// Currently, only ZMQ is supported.
pub enum KvEventSourceConfig {
    Zmq { endpoint: String, topic: String },
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
        tx: mpsc::UnboundedSender<PlacementEvent>,
        next_event_id: Arc<AtomicU64>,
    ) -> Result<Self> {
        match source_config {
            KvEventSourceConfig::Zmq { endpoint, topic } => {
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
    tx: mpsc::UnboundedSender<PlacementEvent>,
    /// Internal monotonic event ID counter. Shared with the ZMQ listener if present.
    next_event_id: Arc<AtomicU64>,
}

impl KvEventPublisher {
    pub fn new(
        component: Component,
        kv_block_size: u32,
        source_config: Option<KvEventSourceConfig>,
    ) -> Result<Self> {
        Self::new_with_local_indexer(
            component,
            kv_block_size,
            source_config,
            false,
            0,
            DEFAULT_BATCHING_TIMEOUT_MS,
        )
    }

    pub fn new_with_local_indexer(
        component: Component,
        kv_block_size: u32,
        source_config: Option<KvEventSourceConfig>,
        enable_local_indexer: bool,
        dp_rank: DpRank,
        batching_timeout_ms: Option<u64>,
    ) -> Result<Self> {
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

        let (tx, rx) = mpsc::unbounded_channel::<PlacementEvent>();
        let worker_id = component.drt().connection_id();

        KvPublisherMetrics::from_component(&component);

        let component_name = component.name();
        tracing::info!(
            "Initializing KvEventPublisher for worker {worker_id} in component {component_name}"
        );

        if enable_local_indexer {
            tracing::info!(
                "LocalKvIndexer enabled for worker {worker_id} in component {component_name}"
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

        let _local_indexer_query_handle = local_indexer.as_ref().map(|local_indexer_ref| {
            let component = component.clone();
            let local_indexer = local_indexer_ref.clone();

            component
                .drt()
                .runtime()
                .secondary()
                .spawn(start_worker_kv_query_endpoint(
                    component,
                    worker_id,
                    dp_rank,
                    local_indexer,
                ))
        });

        let cancellation_token_clone = cancellation_token.clone();
        let local_indexer_clone = local_indexer.clone();

        if enable_local_indexer {
            tracing::info!("Using event plane for KV event publishing (local_indexer mode)");
            let component_clone = component.clone();
            component.drt().runtime().secondary().spawn(async move {
                let event_publisher =
                    match dynamo_runtime::transports::event_plane::EventPublisher::for_component(
                        &component_clone,
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

                start_event_processor(
                    EventPlanePublisher(event_publisher),
                    worker_id,
                    cancellation_token_clone,
                    rx,
                    local_indexer_clone,
                    batching_timeout_ms,
                )
                .await
            });
        } else {
            let stream_name = create_kv_stream_name(&component, KV_EVENT_SUBJECT);
            let nats_server = std::env::var(env_nats::NATS_SERVER)
                .unwrap_or_else(|_| "nats://localhost:4222".to_string());
            let mut nats_queue = NatsQueue::new_without_consumer(
                stream_name,
                nats_server,
                std::time::Duration::from_secs(60),
            );

            component.drt().runtime().secondary().spawn(async move {
                if let Err(e) = nats_queue.connect().await {
                    tracing::error!("Failed to connect NatsQueue: {e}");
                    return;
                }
                start_event_processor_jetstream(
                    nats_queue,
                    worker_id,
                    cancellation_token_clone,
                    rx,
                    local_indexer_clone,
                    batching_timeout_ms,
                )
                .await
            });
        }

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
        let placement_event = PlacementEvent::local_gpu(self.worker_id, event);
        match self.tx.send(placement_event) {
            Ok(()) => Ok(()),
            Err(err) => Err(mpsc::error::SendError(err.0.event)),
        }
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
