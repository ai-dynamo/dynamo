// SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use crate::kv_router::{
    KV_EVENT_SUBJECT, KV_METRICS_SUBJECT,
    indexer::{RouterEvent, compute_block_hash_for_seq},
    protocols::*,
    scoring::LoadEvent,
};
use dynamo_runtime::metrics::{MetricsRegistry, prometheus_names::kvstats};
use dynamo_runtime::traits::{DistributedRuntimeProvider, events::EventPublisher};
use dynamo_runtime::{
    Result,
    component::{Component, Namespace},
    transports::nats::{NatsQueue, QUEUE_NAME, Slug},
};
use std::sync::{Arc, OnceLock};
use tokio::sync::mpsc;
use tokio_util::sync::CancellationToken;

use rmp_serde as rmps;
use serde::Deserialize;
use serde::Serialize;
use serde::de::{self, Deserializer, IgnoredAny, MapAccess, SeqAccess, Visitor};
use std::fmt;
use std::sync::atomic::{AtomicU32, Ordering};
use std::time::Duration;
use zeromq::{Socket, SocketRecv, SubSocket};

// -------------------------------------------------------------------------
// KV Event Publishers -----------------------------------------------------
// -------------------------------------------------------------------------

/// Configure the source of KV events.
/// Currently, only ZMQ is supported.
pub enum KvEventSourceConfig {
    Zmq { endpoint: String, topic: String },
}

/// The source of KV events.
enum KvEventSource {
    Zmq {
        zmq_handle: tokio::task::JoinHandle<()>,
    },
}

impl KvEventSource {
    /// Start the event source from a [`KvEventSourceConfig`].
    fn start(
        component: Component,
        kv_block_size: u32,
        source_config: KvEventSourceConfig,
        cancellation_token: CancellationToken,
        tx: mpsc::UnboundedSender<KvCacheEvent>,
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
                        tx,
                        cancellation_token.clone(),
                        kv_block_size,
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
    /// The channel to send events to.
    tx: mpsc::UnboundedSender<KvCacheEvent>,
}

impl KvEventPublisher {
    pub fn new(
        component: Component,
        worker_id: i64,
        kv_block_size: u32,
        source_config: Option<KvEventSourceConfig>,
    ) -> Result<Self> {
        let cancellation_token = CancellationToken::new();

        let (tx, rx) = mpsc::unbounded_channel::<KvCacheEvent>();

        // Create our event source (if any)
        let mut source = None;
        if let Some(config) = source_config {
            source = Some(KvEventSource::start(
                component.clone(),
                kv_block_size,
                config,
                cancellation_token.clone(),
                tx.clone(),
            )?);
        }

        let stream_name = Slug::slugify(&format!("{}.{}", component.subject(), KV_EVENT_SUBJECT))
            .to_string()
            .replace("_", "-");
        let nats_server =
            std::env::var("NATS_SERVER").unwrap_or_else(|_| "nats://localhost:4222".to_string());
        // Create NatsQueue without consumer since we're only publishing
        let mut nats_queue = NatsQueue::new_without_consumer(
            stream_name,
            nats_server,
            std::time::Duration::from_secs(60), // 1 minute timeout
        );

        // Connect the NatsQueue before passing it to the event processor
        let cancellation_token_clone = cancellation_token.clone();
        component.drt().runtime().secondary().spawn(async move {
            if let Err(e) = nats_queue.connect().await {
                tracing::error!("Failed to connect NatsQueue: {}", e);
                return;
            }
            start_event_processor(nats_queue, worker_id, cancellation_token_clone, rx).await
        });

        Ok(Self {
            kv_block_size,
            source,
            cancellation_token,
            tx,
        })
    }

    pub fn publish(&self, event: KvCacheEvent) -> Result<(), mpsc::error::SendError<KvCacheEvent>> {
        self.tx.send(event)
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

async fn start_event_processor<P: EventPublisher + Send + Sync + 'static>(
    publisher: P,
    worker_id: i64,
    cancellation_token: CancellationToken,
    mut rx: mpsc::UnboundedReceiver<KvCacheEvent>,
) {
    loop {
        tokio::select! {
            _ = cancellation_token.cancelled() => {
                tracing::info!("KV Event source received cancellation signal");
                break;
            }
            event = rx.recv() => {
                let Some(event) = event else {
                    tracing::debug!("Event processor channel closed.");
                    break;
                };

                // Encapsulate in a router event and publish.
                tracing::trace!("Event processor for worker_id {} processing event: {:?}", worker_id, event.data);
                let router_event = RouterEvent::new(worker_id, event);
                if let Err(e) = publisher.publish(QUEUE_NAME, &router_event).await {
                    tracing::error!("Failed to publish event: {}", e);
                }
            }
        }
    }
}

// Error handling configuration for ZMQ operations
const INITIAL_BACKOFF_MS: u64 = 10;
const MAX_BACKOFF_MS: u64 = 5000;
const MAX_CONSECUTIVE_ERRORS: u32 = 10;
const MAX_BACKOFF_EXPONENT: u32 = 8; // Cap at 2^8 = 256x multiplier to prevent overflow

/// Calculate exponential backoff duration based on consecutive error count
fn calculate_backoff_ms(consecutive_errors: u32) -> u64 {
    std::cmp::min(
        INITIAL_BACKOFF_MS * 2_u64.pow(consecutive_errors.min(MAX_BACKOFF_EXPONENT)),
        MAX_BACKOFF_MS,
    )
}

pub async fn start_zmq_listener(
    zmq_endpoint: String,
    zmq_topic: String,
    tx: mpsc::UnboundedSender<KvCacheEvent>,
    cancellation_token: CancellationToken,
    kv_block_size: u32,
) {
    tracing::debug!(
        "KVEventPublisher connecting to ZMQ endpoint {} (topic '{}')",
        zmq_endpoint,
        zmq_topic
    );

    let warning_count = Arc::new(AtomicU32::new(0));

    let mut socket = SubSocket::new();

    // Subscribe to the requested topic (empty string == all topics)
    if let Err(e) = socket.subscribe(&zmq_topic).await {
        tracing::error!("Failed to subscribe on ZMQ socket: {}", e);
        return;
    }

    if let Err(e) = socket.connect(&zmq_endpoint).await {
        tracing::error!("Failed to connect ZMQ SUB socket: {}", e);
        return;
    }

    let mut consecutive_errors = 0u32;
    #[allow(unused_assignments)]
    let mut exit_reason = "unknown";
    let mut messages_processed = 0u64;

    'main: loop {
        tokio::select! {
            biased;

            // Check for cancellation
            _ = cancellation_token.cancelled() => {
                tracing::debug!("ZMQ listener received cancellation signal");
                exit_reason = "cancellation token cancelled";
                break 'main;
            }

            // Receive message
            msg_result = socket.recv() => {
                let Ok(msg) = msg_result else {
                    let e = msg_result.unwrap_err();
                    consecutive_errors += 1;

                    if consecutive_errors >= MAX_CONSECUTIVE_ERRORS {
                        tracing::error!(
                            error=%e,
                            consecutive_errors=%consecutive_errors,
                            "Too many consecutive ZMQ errors, terminating listener"
                        );
                        exit_reason = "too many consecutive errors";
                        break 'main;
                    }

                    // Simple exponential backoff with max exponent to prevent overflow
                    let backoff_ms = calculate_backoff_ms(consecutive_errors);

                    tracing::warn!(
                        error=%e,
                        consecutive_errors=%consecutive_errors,
                        backoff_ms=%backoff_ms,
                        "Error reading from ZMQ socket, applying exponential backoff"
                    );

                    tokio::time::sleep(Duration::from_millis(backoff_ms)).await;
                    continue;
                };
                // Reset error count on successful message
                consecutive_errors = 0;

                // We expect multipart frames: [topic, seq, payload]
                let mut frames: Vec<Vec<u8>> = msg.into_vec().into_iter().map(|frame| frame.to_vec()).collect();

                if frames.len() != 3 {
                    tracing::warn!(expected=3, actual=%frames.len(), "Received unexpected ZMQ frame count");
                    continue;
                }

                // Extract the payload and sequence number.
                let payload = frames.pop().unwrap();
                let seq_bytes = frames.pop().unwrap();

                if seq_bytes.len() != 8 {
                    tracing::warn!(expected=8, actual=%seq_bytes.len(), "Invalid sequence number byte length");
                    continue;
                }

                let seq = u64::from_be_bytes(seq_bytes.try_into().unwrap());

                // Decode our batch of events.
                let batch_result = rmps::from_slice::<KvEventBatch>(&payload);
                let Ok(batch) = batch_result else {
                    let e = batch_result.unwrap_err();
                    tracing::warn!(error=%e, "Failed to decode KVEventBatch msgpack");
                    continue;
                };

                tracing::trace!(
                    "ZMQ listener on {} received batch with {} events (seq={}, dp_rank={})",
                    zmq_endpoint,
                    batch.events.len(),
                    seq,
                    batch.data_parallel_rank
                );

                let dp_rank = batch.data_parallel_rank;
                for raw_event in batch.events.into_iter() {
                    let event = convert_event(raw_event, seq, kv_block_size, dp_rank, &warning_count);
                    if tx.send(event).is_err() {
                        tracing::warn!("Failed to send message to channel - receiver dropped");
                        exit_reason = "channel receiver dropped";
                        break 'main;
                    }
                    messages_processed += 1;
                }
            }
        }
    }
    tracing::debug!(
        "ZMQ listener exiting, reason: {}, messages processed: {}",
        exit_reason,
        messages_processed
    );
}

/// Convert a raw event coming from the ZMQ channel into the internal
/// [`KvCacheEvent`] representation used by the router.
fn convert_event(
    raw: RawKvEvent,
    event_id: u64,
    kv_block_size: u32,
    dp_rank: u32,
    warning_count: &Arc<AtomicU32>,
) -> KvCacheEvent {
    match raw {
        RawKvEvent::BlockStored {
            block_hashes,
            parent_block_hash,
            token_ids,
            block_size,
            lora_id,
            ..
        } => {
            let num_block_tokens = vec![block_size as u64; block_hashes.len()];
            let block_hashes_u64: Vec<u64> = block_hashes
                .into_iter()
                .map(BlockHashValue::into_u64)
                .collect();
            KvCacheEvent {
                event_id,
                data: KvCacheEventData::Stored(KvCacheStoreData {
                    parent_hash: parent_block_hash
                        .map(BlockHashValue::into_u64)
                        .map(ExternalSequenceBlockHash::from),
                    blocks: create_stored_blocks(
                        kv_block_size,
                        &token_ids,
                        &num_block_tokens,
                        &block_hashes_u64,
                        lora_id.unwrap_or(0),
                        warning_count,
                    ),
                }),
                dp_rank,
            }
        }
        RawKvEvent::BlockRemoved { block_hashes, .. } => {
            let hashes = block_hashes
                .into_iter()
                .map(BlockHashValue::into_u64)
                .map(ExternalSequenceBlockHash::from)
                .collect();
            KvCacheEvent {
                event_id,
                data: KvCacheEventData::Removed(KvCacheRemoveData {
                    block_hashes: hashes,
                }),
                dp_rank,
            }
        }
        RawKvEvent::AllBlocksCleared => KvCacheEvent {
            event_id,
            data: KvCacheEventData::Cleared,
            dp_rank,
        },
    }
}

pub fn create_stored_block_from_parts(
    kv_block_size: u32,
    block_hash: u64,
    token_ids: &[u32],
    _lora_id: u64,
) -> KvCacheStoredBlockData {
    let tokens_hash = compute_block_hash_for_seq(token_ids, kv_block_size)[0];
    tracing::trace!(
        "Creating stored block: external_block_hash={}, tokens_hash={}, token_ids={:?}, kv_block_size={}",
        block_hash,
        tokens_hash.0,
        token_ids,
        kv_block_size
    );
    KvCacheStoredBlockData {
        block_hash: ExternalSequenceBlockHash::from(block_hash),
        tokens_hash,
    }
}

pub fn create_stored_blocks(
    kv_block_size: u32,
    token_ids: &[u32],
    num_block_tokens: &[u64],
    block_hashes: &[u64],
    lora_id: u64,
    warning_count: &Arc<AtomicU32>,
) -> Vec<KvCacheStoredBlockData> {
    let mut blocks: Vec<KvCacheStoredBlockData> = Vec::new();

    let mut token_offset: usize = 0;
    for (num_tokens_it, block_hash_it) in num_block_tokens.iter().zip(block_hashes.iter()) {
        if *num_tokens_it != kv_block_size as u64 {
            if warning_count.fetch_add(1, Ordering::Relaxed) < 3 {
                tracing::warn!(
                    "Block not published. Block size must be {} tokens to be published. Block size is: {}",
                    kv_block_size,
                    *num_tokens_it
                );
            }
            break;
        }

        let tokens = &token_ids[token_offset..(token_offset + *num_tokens_it as usize)];
        blocks.push(create_stored_block_from_parts(
            kv_block_size,
            *block_hash_it,
            tokens,
            lora_id,
        ));
        token_offset += *num_tokens_it as usize;
    }

    blocks
}

// -------------------------------------------------------------------------
// Types mirroring the Python msgspec-defined structures -------------------
// -------------------------------------------------------------------------

#[derive(Debug, Deserialize, Serialize)]
struct KvEventBatch {
    ts: f64,
    events: Vec<RawKvEvent>,
    #[serde(alias = "dp_rank")]
    data_parallel_rank: u32, // we are ignoring this for now
}

#[derive(Debug, Serialize, Deserialize, Clone, Copy)]
#[serde(untagged)]
enum BlockHashValue {
    Signed(i64),
    Unsigned(u64),
}

impl BlockHashValue {
    fn into_u64(self) -> u64 {
        match self {
            BlockHashValue::Signed(v) => v as u64,
            BlockHashValue::Unsigned(v) => v,
        }
    }
}

#[derive(Debug, Serialize)]
#[serde(tag = "type")] // msgspec encodes variant tag as a string when `tag=True`
enum RawKvEvent {
    BlockStored {
        /// Block hashes may be emitted as either signed or unsigned 64-bit values.
        /// We normalize them to `u64` while deserializing to support both producers.
        block_hashes: Vec<BlockHashValue>,
        parent_block_hash: Option<BlockHashValue>,
        token_ids: Vec<u32>,
        block_size: usize,
        lora_id: Option<u64>,
        #[serde(skip_serializing_if = "Option::is_none")]
        medium: Option<String>,
    },
    BlockRemoved {
        block_hashes: Vec<BlockHashValue>,
        #[serde(skip_serializing_if = "Option::is_none")]
        medium: Option<String>,
    },
    AllBlocksCleared,
}

/// Our producers use msgspec with `tag=True` and `array_like=True`, which
/// encodes each event as either a tagged map or a tagged tuple. To be tolerant of
/// additional fields that may be appended in the future, we implement a custom
/// deserializer that ignores unknown keys and any extra positional elements.
///
/// This keeps us compatible with older payloads while safely
/// accepting newer ones that include extra metadata.
impl<'de> Deserialize<'de> for RawKvEvent {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        deserializer.deserialize_any(RawKvEventVisitor)
    }
}

struct RawKvEventVisitor;

impl<'de> Visitor<'de> for RawKvEventVisitor {
    type Value = RawKvEvent;

    fn expecting(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        formatter.write_str("a kv event encoded as a tagged map or sequence")
    }

    fn visit_map<A>(self, mut map: A) -> Result<Self::Value, A::Error>
    where
        A: MapAccess<'de>,
    {
        let mut event_type: Option<String> = None;
        let mut block_hashes: Option<Vec<BlockHashValue>> = None;
        let mut parent_block_hash: Option<Option<BlockHashValue>> = None;
        let mut token_ids: Option<Vec<u32>> = None;
        let mut block_size: Option<usize> = None;
        let mut lora_id: Option<Option<u64>> = None;
        let mut medium: Option<Option<String>> = None;

        while let Some(key) = map.next_key::<String>()? {
            match key.as_str() {
                "type" => {
                    event_type = Some(map.next_value()?);
                }
                "block_hashes" => {
                    block_hashes = Some(map.next_value()?);
                }
                "parent_block_hash" => {
                    parent_block_hash = Some(map.next_value()?);
                }
                "token_ids" => {
                    token_ids = Some(map.next_value()?);
                }
                "block_size" => {
                    block_size = Some(map.next_value()?);
                }
                "lora_id" => {
                    lora_id = Some(map.next_value()?);
                }
                "medium" => {
                    medium = Some(map.next_value()?);
                }
                _ => {
                    map.next_value::<IgnoredAny>()?;
                }
            }
        }

        match event_type.as_deref() {
            Some("BlockStored") => {
                let block_hashes =
                    block_hashes.ok_or_else(|| de::Error::missing_field("block_hashes"))?;
                let token_ids = token_ids.ok_or_else(|| de::Error::missing_field("token_ids"))?;
                let block_size =
                    block_size.ok_or_else(|| de::Error::missing_field("block_size"))?;
                Ok(RawKvEvent::BlockStored {
                    block_hashes,
                    parent_block_hash: parent_block_hash.unwrap_or(None),
                    token_ids,
                    block_size,
                    lora_id: lora_id.unwrap_or(None),
                    medium: medium.unwrap_or(None),
                })
            }
            Some("BlockRemoved") => {
                let block_hashes =
                    block_hashes.ok_or_else(|| de::Error::missing_field("block_hashes"))?;
                Ok(RawKvEvent::BlockRemoved {
                    block_hashes,
                    medium: medium.unwrap_or(None),
                })
            }
            Some("AllBlocksCleared") => Ok(RawKvEvent::AllBlocksCleared),
            Some(other) => Err(de::Error::unknown_variant(
                other,
                &["BlockStored", "BlockRemoved", "AllBlocksCleared"],
            )),
            None => Err(de::Error::missing_field("type")),
        }
    }

    fn visit_seq<A>(self, mut seq: A) -> Result<Self::Value, A::Error>
    where
        A: SeqAccess<'de>,
    {
        let tag: Option<String> = seq.next_element()?;
        let Some(tag) = tag else {
            return Err(de::Error::invalid_length(
                0,
                &"sequence must start with event tag",
            ));
        };

        match tag.as_str() {
            "BlockStored" => {
                let block_hashes: Vec<BlockHashValue> = seq
                    .next_element()?
                    .ok_or_else(|| de::Error::invalid_length(1, &"missing block_hashes"))?;
                let parent_block_hash: Option<BlockHashValue> = seq.next_element()?.unwrap_or(None);
                let token_ids: Vec<u32> = seq
                    .next_element()?
                    .ok_or_else(|| de::Error::invalid_length(3, &"missing token_ids"))?;
                let block_size: usize = seq
                    .next_element()?
                    .ok_or_else(|| de::Error::invalid_length(4, &"missing block_size"))?;
                let lora_id: Option<u64> = seq.next_element()?.unwrap_or(None);
                let medium: Option<String> = seq.next_element()?.unwrap_or(None);

                while seq.next_element::<IgnoredAny>()?.is_some() {}

                Ok(RawKvEvent::BlockStored {
                    block_hashes,
                    parent_block_hash,
                    token_ids,
                    block_size,
                    lora_id,
                    medium,
                })
            }
            "BlockRemoved" => {
                let block_hashes: Vec<BlockHashValue> = seq
                    .next_element()?
                    .ok_or_else(|| de::Error::invalid_length(1, &"missing block_hashes"))?;
                let medium: Option<String> = seq.next_element()?.unwrap_or(None);

                while seq.next_element::<IgnoredAny>()?.is_some() {}

                Ok(RawKvEvent::BlockRemoved {
                    block_hashes,
                    medium,
                })
            }
            "AllBlocksCleared" => {
                while seq.next_element::<IgnoredAny>()?.is_some() {}
                Ok(RawKvEvent::AllBlocksCleared)
            }
            other => Err(de::Error::unknown_variant(
                other,
                &["BlockStored", "BlockRemoved", "AllBlocksCleared"],
            )),
        }
    }
}

// -------------------------------------------------------------------------
// Metrics Publishers ------------------------------------------------------
// -------------------------------------------------------------------------

pub struct WorkerMetricsPublisher {
    tx: tokio::sync::watch::Sender<Arc<ForwardPassMetrics>>,
    rx: tokio::sync::watch::Receiver<Arc<ForwardPassMetrics>>,
    /// Prometheus gauges for KvStats metrics
    /// We use OnceLock for efficient one-time initialization and lock-free reads
    /// The gauges are set once during register_prometheus_metrics and then only read
    prometheus_gauges: OnceLock<KvStatsPrometheusGauges>,
}

struct KvStatsPrometheusGauges {
    kv_active_blocks_gauge: prometheus::Gauge,
    kv_total_blocks_gauge: prometheus::Gauge,
    gpu_cache_usage_gauge: prometheus::Gauge,
    gpu_prefix_cache_hit_rate_gauge: prometheus::Gauge,
}

impl KvStatsPrometheusGauges {
    /// Create a new KvStatsPrometheusGauges instance with all metrics registered
    fn new(component: &Component) -> Result<Self> {
        let kv_active_blocks_gauge = component.create_gauge(
            kvstats::ACTIVE_BLOCKS,
            "Number of active KV cache blocks currently in use",
            &[],
        )?;

        let kv_total_blocks_gauge = component.create_gauge(
            kvstats::TOTAL_BLOCKS,
            "Total number of KV cache blocks available",
            &[],
        )?;

        let gpu_cache_usage_gauge = component.create_gauge(
            kvstats::GPU_CACHE_USAGE_PERCENT,
            "GPU cache usage as a percentage (0.0-1.0)",
            &[],
        )?;

        let gpu_prefix_cache_hit_rate_gauge = component.create_gauge(
            kvstats::GPU_PREFIX_CACHE_HIT_RATE,
            "GPU prefix cache hit rate as a percentage (0.0-1.0)",
            &[],
        )?;

        tracing::info!("Registered KvStats Prometheus metrics");

        Ok(KvStatsPrometheusGauges {
            kv_active_blocks_gauge,
            kv_total_blocks_gauge,
            gpu_cache_usage_gauge,
            gpu_prefix_cache_hit_rate_gauge,
        })
    }

    /// Update all gauges with values from KvStats
    fn update_from_kvstats(&self, kv_stats: &KvStats) {
        self.kv_active_blocks_gauge
            .set(kv_stats.kv_active_blocks as f64);
        self.kv_total_blocks_gauge
            .set(kv_stats.kv_total_blocks as f64);
        self.gpu_cache_usage_gauge
            .set(kv_stats.gpu_cache_usage_perc as f64);
        self.gpu_prefix_cache_hit_rate_gauge
            .set(kv_stats.gpu_prefix_cache_hit_rate as f64);
    }
}

impl WorkerMetricsPublisher {
    pub fn new() -> Result<Self> {
        let (tx, rx) = tokio::sync::watch::channel(Arc::new(ForwardPassMetrics::default()));
        Ok(WorkerMetricsPublisher {
            tx,
            rx,
            prometheus_gauges: OnceLock::new(),
        })
    }

    pub fn publish(
        &self,
        metrics: Arc<ForwardPassMetrics>,
    ) -> Result<(), tokio::sync::watch::error::SendError<Arc<ForwardPassMetrics>>> {
        tracing::trace!("Publish metrics: {metrics:?}");

        // Update Prometheus gauges - OnceLock provides lock-free reads after initialization
        // This is the hot path - we only read the Arc, no locking overhead
        if let Some(gauges) = self.prometheus_gauges.get() {
            gauges.update_from_kvstats(&metrics.kv_stats);
        }

        self.tx.send(metrics)
    }

    /// Register KvStats Prometheus metrics with the component's registry
    pub fn register_prometheus_metrics(&self, component: &Component) -> Result<()> {
        // Use get_or_init for thread-safe one-time initialization
        // This will only initialize once, subsequent calls will return immediately
        self.prometheus_gauges.get_or_init(|| {
            KvStatsPrometheusGauges::new(component).expect("Failed to create Prometheus gauges")
        });

        Ok(())
    }

    pub async fn create_endpoint(&self, component: Component) -> Result<()> {
        let worker_id = component
            .drt()
            .primary_lease()
            .map(|lease| lease.id())
            .unwrap_or_else(|| {
                tracing::warn!("Component is static, assuming worker_id of 0");
                0
            });

        self.start_nats_metrics_publishing(component.namespace().clone(), worker_id);
        Ok(())
    }

    /// Starts a background task to publish metrics over NATS
    ///
    /// This task monitors metric changes (specifically kv_active_blocks and num_requests_waiting)
    /// and publishes stable metrics to NATS after they've been unchanged for 1ms.
    fn start_nats_metrics_publishing(&self, namespace: Namespace, worker_id: i64) {
        let nats_rx = self.rx.clone();

        tokio::spawn(async move {
            let mut rx = nats_rx;
            let mut last_kv_active_blocks: Option<u64> = None;
            let mut last_num_requests_waiting: Option<u64> = None;
            let mut pending_publish: Option<Arc<ForwardPassMetrics>> = None;
            let mut publish_timer =
                Box::pin(tokio::time::sleep(tokio::time::Duration::from_secs(0)));
            publish_timer.as_mut().reset(tokio::time::Instant::now()); // Complete immediately

            loop {
                tokio::select! {
                    // Handle metrics changes
                    result = rx.changed() => {
                        if result.is_err() {
                            tracing::debug!(
                                "Metrics publisher sender dropped, stopping NATS background task"
                            );
                            break;
                        }

                        let metrics = rx.borrow_and_update().clone();

                        // Extract the values we care about
                        let current_kv_active_blocks = metrics.kv_stats.kv_active_blocks;
                        let current_num_requests_waiting =
                            metrics.worker_stats.num_requests_waiting;

                        // Check if these specific metrics have changed
                        let has_changed = match (last_kv_active_blocks, last_num_requests_waiting) {
                            (Some(last_kv), Some(last_requests)) => {
                                last_kv != current_kv_active_blocks
                                    || last_requests != current_num_requests_waiting
                            }
                            _ => true, // First time, consider it changed
                        };

                        // If load metrics changed, schedule a publish
                        if has_changed {
                            pending_publish = Some(metrics.clone());
                            last_kv_active_blocks = Some(current_kv_active_blocks);
                            last_num_requests_waiting = Some(current_num_requests_waiting);

                            // Start the 1ms timer
                            publish_timer.as_mut().reset(
                                tokio::time::Instant::now() + tokio::time::Duration::from_millis(1)
                            );
                        }
                    }
                    // Timer expired - publish if we have pending metrics
                    _ = &mut publish_timer => {
                        if let Some(metrics) = pending_publish.take() {
                            // Create LoadEvent wrapping the metrics
                            let load_event = LoadEvent {
                                worker_id,
                                data: (*metrics).clone(),
                            };

                            if let Err(e) =
                                namespace.publish(KV_METRICS_SUBJECT, &load_event).await
                            {
                                tracing::warn!("Failed to publish metrics over NATS: {}", e);
                            }
                        }
                    }
                }
            }
        });
    }
}

// -------------------------------------------------------------------------
// Testing -----------------------------------------------------------------
// -------------------------------------------------------------------------

#[cfg(test)]
mod test_event_processing {
    use super::*;
    use crate::kv_router::indexer::compute_block_hash_for_seq;

    // ---------------------------------------------------------------------
    // create_stored_block_from_parts --------------------------------------
    // ---------------------------------------------------------------------
    #[test]
    fn test_create_stored_block_from_parts() {
        let kv_block_size = 4;
        let token_ids = vec![10, 20, 30, 40];
        let blk_hash = 0xdead_beef;

        let stored = create_stored_block_from_parts(kv_block_size, blk_hash, &token_ids, 0);

        assert_eq!(stored.block_hash.0, blk_hash);
        let expected_hash = compute_block_hash_for_seq(&token_ids, 4)[0];
        assert_eq!(stored.tokens_hash, expected_hash);
    }

    // ---------------------------------------------------------------------
    // create_stored_blocks -------------------------------------------------
    // ---------------------------------------------------------------------
    #[test]
    fn test_create_stored_blocks_ok() {
        let kv_block_size = 4;
        // two blocks, each of size 4
        let token_ids = vec![1, 2, 3, 4, 5, 6, 7, 8];
        let num_block_tokens = vec![4_u64, 4_u64];
        let block_hashes = vec![111_u64, 222_u64];

        let blocks = create_stored_blocks(
            kv_block_size,
            &token_ids,
            &num_block_tokens,
            &block_hashes,
            /*lora_id=*/ 0,
            &Arc::new(AtomicU32::new(0)),
        );

        assert_eq!(blocks.len(), 2);
        assert_eq!(blocks[0].block_hash.0, 111);
        assert_eq!(blocks[1].block_hash.0, 222);
    }

    #[test]
    fn test_create_stored_blocks_wrong_size_triggers_warning() {
        let kv_block_size = 4;
        // second block is the wrong size
        let token_ids = vec![1, 2, 3, 4, 5, 6, 7];
        let num_block_tokens = vec![4_u64, 3_u64];
        let block_hashes = vec![111_u64, 222_u64];
        let warning_count = Arc::new(AtomicU32::new(0));

        let blocks = create_stored_blocks(
            kv_block_size,
            &token_ids,
            &num_block_tokens,
            &block_hashes,
            /*lora_id=*/ 0,
            &warning_count,
        );

        // should early-exit as second has mismatch
        assert!(blocks.len() == 1);
        assert!(warning_count.load(Ordering::Relaxed) == 1)
    }

    // ---------------------------------------------------------------------
    // convert_event --------------------------------------------------------
    // ---------------------------------------------------------------------
    #[test]
    fn test_convert_event_block_stored() {
        let kv_block_size = 4;
        let raw_evt = RawKvEvent::BlockStored {
            block_hashes: vec![BlockHashValue::Unsigned(10), BlockHashValue::Unsigned(11)],
            parent_block_hash: Some(BlockHashValue::Unsigned(99)),
            token_ids: vec![1, 2, 3, 4, 5, 6, 7, 8],
            block_size: 4,
            lora_id: Some(0),
            medium: None,
        };

        let out = convert_event(raw_evt, 42, kv_block_size, 0, &Arc::new(AtomicU32::new(0)));
        assert!(matches!(out.data, KvCacheEventData::Stored(_)));
    }

    #[test]
    fn test_convert_event_block_removed() {
        let kv_block_size = 4;
        let raw_evt = RawKvEvent::BlockRemoved {
            block_hashes: vec![BlockHashValue::Unsigned(123), BlockHashValue::Signed(456)],
            medium: None,
        };
        let out = convert_event(raw_evt, 7, kv_block_size, 0, &Arc::new(AtomicU32::new(0)));

        assert!(matches!(out.data, KvCacheEventData::Removed(_)));
    }

    #[test]
    fn test_convert_event_all_blocks_cleared() {
        let kv_block_size = 4;
        let raw_evt = RawKvEvent::AllBlocksCleared;
        let out = convert_event(raw_evt, 1, kv_block_size, 0, &Arc::new(AtomicU32::new(0)));
        assert!(matches!(out.data, KvCacheEventData::Cleared));
    }
}

#[cfg(test)]
mod tests_startup_helpers {
    use super::*;
    use crate::kv_router::protocols::ExternalSequenceBlockHash;
    use async_trait;
    use bytes::Bytes;
    use std::sync::{Arc, Mutex};
    use zeromq::{PubSocket, Socket, SocketSend, ZmqMessage};

    // Type alias to resolve clippy::type_complexity warning
    type PublishedEvents = Arc<Mutex<Vec<(String, Vec<u8>)>>>;

    //--------------------------------------------------------------------
    // A tiny stand-in for Component that just records every publish call
    //--------------------------------------------------------------------
    #[derive(Default)]
    struct MockComponent {
        published: PublishedEvents,
    }

    impl MockComponent {
        fn new() -> (Self, PublishedEvents) {
            let published = Arc::new(Mutex::new(Vec::new()));
            (
                Self {
                    published: published.clone(),
                },
                published,
            )
        }
    }

    #[async_trait::async_trait]
    impl EventPublisher for MockComponent {
        async fn publish(
            &self,
            event_name: impl AsRef<str> + Send + Sync,
            event: &(impl serde::Serialize + Send + Sync),
        ) -> dynamo_runtime::Result<()> {
            let bytes = rmp_serde::to_vec(event).unwrap();
            self.published
                .lock()
                .unwrap()
                .push((event_name.as_ref().to_string(), bytes));
            Ok(())
        }

        async fn publish_bytes(
            &self,
            event_name: impl AsRef<str> + Send + Sync,
            bytes: Vec<u8>,
        ) -> dynamo_runtime::Result<()> {
            self.published
                .lock()
                .unwrap()
                .push((event_name.as_ref().to_string(), bytes));
            Ok(())
        }

        fn subject(&self) -> String {
            "mock.subject".into()
        }
    }

    //--------------------------------------------------------------------
    // Test start_event_processor
    //--------------------------------------------------------------------
    #[tokio::test]
    async fn test_start_event_processor() {
        let (component, published) = MockComponent::new();

        let event = KvCacheEvent {
            event_id: 1,
            data: KvCacheEventData::Removed(KvCacheRemoveData {
                block_hashes: vec![ExternalSequenceBlockHash(1), ExternalSequenceBlockHash(2)],
            }),
            dp_rank: 0,
        };

        let token = CancellationToken::new();
        let (tx, rx) = mpsc::unbounded_channel::<KvCacheEvent>();
        tx.send(event).unwrap();
        drop(tx);

        let handle = tokio::spawn(start_event_processor(component, 1, token, rx));

        tokio::time::timeout(tokio::time::Duration::from_secs(1), handle)
            .await
            .unwrap()
            .unwrap();

        let published = published.lock().unwrap();
        assert_eq!(published.len(), 1);
        let (subject, _) = &published[0];
        assert_eq!(subject, QUEUE_NAME);
    }

    //--------------------------------------------------------------------
    // Test start_zmq_listener without a real socket
    //   (feed it frames through a ZMQ PAIR tcp socket)
    //--------------------------------------------------------------------
    #[tokio::test]
    async fn test_start_zmq_listener_pushes_to_channel() {
        // Prepare channel that listener should fill
        let (tx, mut rx) = mpsc::unbounded_channel::<KvCacheEvent>();

        // ZMQ TCP endpoint using localhost with fixed port
        let endpoint = "tcp://127.0.0.1:15555";
        let topic = "".to_string(); // subscribe to all

        // Publisher side - set up first
        let mut pub_socket = PubSocket::new();
        pub_socket.bind(endpoint).await.unwrap();

        // Cancellation token so we can stop the listener
        let token = dynamo_runtime::CancellationToken::new();

        // Spawn async listener
        let listener_handle = tokio::spawn({
            let token = token.clone();
            start_zmq_listener(endpoint.to_string(), topic, tx, token, 4)
        });

        // Give time for the connection to establish
        tokio::time::sleep(tokio::time::Duration::from_millis(100)).await;

        // Send synthetic 3-frame message: [topic, seq(8B), payload]
        let seq: u64 = 77;

        let events = vec![RawKvEvent::BlockStored {
            block_hashes: vec![BlockHashValue::Unsigned(42)],
            parent_block_hash: None,
            token_ids: vec![0, 1, 2, 3],
            block_size: 4,
            lora_id: None,
            medium: None,
        }];

        let batch = KvEventBatch {
            ts: 0.0,
            events,
            data_parallel_rank: 1,
        };

        let payload = Bytes::from(rmps::to_vec(&batch).unwrap());

        let frames = vec![
            Bytes::from(""),
            Bytes::from(seq.to_be_bytes().to_vec()),
            payload.clone(),
        ];

        // Create a proper multipart message
        let msg = ZmqMessage::try_from(frames).expect("Failed to create ZmqMessage");

        // Send the multipart message
        pub_socket.send(msg).await.unwrap();

        // Wait for message to be received
        tokio::time::sleep(tokio::time::Duration::from_millis(100)).await;

        // Check that we received the message
        let event = rx.try_recv().expect("no message received");

        let KvCacheEventData::Stored(KvCacheStoreData {
            parent_hash,
            blocks,
        }) = event.data
        else {
            panic!("expected KvCacheStoreData");
        };

        assert!(parent_hash.is_none());
        assert_eq!(blocks.len(), 1);
        assert_eq!(blocks[0].block_hash.0, 42);

        // Stop the listener
        token.cancel();
        let _ = listener_handle.await;
    }
}

#[cfg(test)]
mod test_exponential_backoff {
    use super::*;

    #[test]
    fn test_backoff_calculation_progression() {
        // Test the exponential progression
        assert_eq!(calculate_backoff_ms(0), 10); // 10 * 2^0 = 10
        assert_eq!(calculate_backoff_ms(1), 20); // 10 * 2^1 = 20
        assert_eq!(calculate_backoff_ms(2), 40); // 10 * 2^2 = 40
        assert_eq!(calculate_backoff_ms(3), 80); // 10 * 2^3 = 80
        assert_eq!(calculate_backoff_ms(4), 160); // 10 * 2^4 = 160
        assert_eq!(calculate_backoff_ms(5), 320); // 10 * 2^5 = 320
        assert_eq!(calculate_backoff_ms(6), 640); // 10 * 2^6 = 640
        assert_eq!(calculate_backoff_ms(7), 1280); // 10 * 2^7 = 1280
        assert_eq!(calculate_backoff_ms(8), 2560); // 10 * 2^8 = 2560
    }

    #[test]
    fn test_backoff_caps_at_max_exponent() {
        // After MAX_BACKOFF_EXPONENT, should stay at 2^8 = 2560ms
        assert_eq!(calculate_backoff_ms(8), 2560);
        assert_eq!(calculate_backoff_ms(9), 2560); // Same as 8
        assert_eq!(calculate_backoff_ms(100), 2560); // Same as 8
    }

    #[test]
    fn test_backoff_never_exceeds_max() {
        // Even if we somehow had a huge exponent, never exceed MAX_BACKOFF_MS
        for i in 0..20 {
            assert!(calculate_backoff_ms(i) <= MAX_BACKOFF_MS);
        }
    }

    #[test]
    #[allow(clippy::assertions_on_constants)]
    fn test_backoff_constants_are_sane() {
        // Verify our constants make sense together
        assert!(INITIAL_BACKOFF_MS > 0);
        assert!(MAX_BACKOFF_MS > INITIAL_BACKOFF_MS);
        assert!(MAX_BACKOFF_EXPONENT <= 10); // Prevent crazy exponents
        assert!(MAX_CONSECUTIVE_ERRORS > 0);

        // Max calculated value should be less than MAX_BACKOFF_MS
        let max_calculated = INITIAL_BACKOFF_MS * 2_u64.pow(MAX_BACKOFF_EXPONENT);
        assert!(max_calculated <= MAX_BACKOFF_MS);
    }
}

#[cfg(all(test, feature = "integration"))]
mod test_integration_publisher {
    use super::*;
    use crate::kv_router::protocols::{ForwardPassMetrics, KvStats, WorkerStats};
    use dynamo_runtime::distributed_test_utils::create_test_drt_async;
    use dynamo_runtime::traits::events::EventSubscriber;
    use futures::StreamExt;

    #[tokio::test]
    #[ignore] // Mark as ignored as requested, because CI's integrations still don't have NATS
    async fn test_metrics_publishing_behavior() -> Result<()> {
        // Set up runtime and namespace
        let drt = create_test_drt_async().await;
        let namespace = drt.namespace("ns2001".to_string())?;

        // Create a subscriber for the metrics events using subscribe_with_type
        let mut subscriber = namespace
            .subscribe_with_type::<LoadEvent>(KV_METRICS_SUBJECT)
            .await
            .unwrap();

        // Create WorkerMetricsPublisher
        let publisher = WorkerMetricsPublisher::new().unwrap();
        let worker_id = 1234;

        // Start NATS metrics publishing
        publisher.start_nats_metrics_publishing(namespace.clone(), worker_id);

        // Allow some time for the background task to start
        tokio::time::sleep(tokio::time::Duration::from_millis(10)).await;

        // Test 1: Publish 10 different metrics with 0.5ms intervals
        // Only the last one should be published after 1ms of stability
        for i in 0..10 {
            let metrics = Arc::new(ForwardPassMetrics {
                kv_stats: KvStats {
                    kv_active_blocks: (i * 100) as u64, // Changing load metric
                    kv_total_blocks: 1000,
                    gpu_cache_usage_perc: 0.5,
                    gpu_prefix_cache_hit_rate: 0.8,
                },
                worker_stats: WorkerStats {
                    num_requests_waiting: (i * 10) as u64, // Changing load metric
                    data_parallel_rank: None,
                    request_active_slots: 50,
                    request_total_slots: 100,
                },
                spec_decode_stats: None,
            });

            publisher.publish(metrics).unwrap();
            tokio::time::sleep(tokio::time::Duration::from_micros(100)).await;
        }

        // Wait a bit more than 1ms to ensure the last metric is published
        tokio::time::sleep(tokio::time::Duration::from_millis(10)).await;

        // Verify we receive exactly one event with the last metric values
        let result =
            tokio::time::timeout(tokio::time::Duration::from_millis(500), subscriber.next())
                .await
                .unwrap();

        let event = result.unwrap().unwrap(); // Unwrap the Option and the Result
        assert_eq!(event.worker_id, worker_id);
        assert_eq!(event.data.kv_stats.kv_active_blocks, 900); // Last value: 9 * 100
        assert_eq!(event.data.worker_stats.num_requests_waiting, 90); // Last value: 9 * 10

        // Ensure no more events are waiting
        let no_msg =
            tokio::time::timeout(tokio::time::Duration::from_millis(50), subscriber.next()).await;
        assert!(no_msg.is_err(), "Expected no more messages, but found one");

        // Test 2: Publish 10 more metrics where everything changes EXCEPT the load metrics
        for i in 0..10 {
            let metrics = Arc::new(ForwardPassMetrics {
                kv_stats: KvStats {
                    kv_active_blocks: 900,                         // Keep same as last published
                    kv_total_blocks: 1000 + (i * 100) as u64,      // Change other metrics
                    gpu_cache_usage_perc: 0.3 + (i as f32 * 0.05), // Change other metrics
                    gpu_prefix_cache_hit_rate: 0.7 + (i as f32 * 0.01), // Change other metrics
                },
                worker_stats: WorkerStats {
                    num_requests_waiting: 90, // Keep same as last published
                    data_parallel_rank: None,
                    request_active_slots: 40 + (i * 5) as u64, // Change other metrics
                    request_total_slots: 100 + (i * 10) as u64, // Change other metrics
                },
                spec_decode_stats: None,
            });

            publisher.publish(metrics).unwrap();
            tokio::time::sleep(tokio::time::Duration::from_micros(100)).await;
        }

        // Wait to ensure no events are published
        tokio::time::sleep(tokio::time::Duration::from_millis(10)).await;

        // Verify no events are received
        let no_msg =
            tokio::time::timeout(tokio::time::Duration::from_millis(50), subscriber.next()).await;
        assert!(
            no_msg.is_err(),
            "Expected no messages when load metrics don't change"
        );

        drt.shutdown();

        Ok(())
    }

    #[tokio::test]
    #[ignore] // Mark as ignored as requested, because CI's integrations still don't have NATS
    async fn test_kvstats_prometheus_gauge_updates() {
        use crate::kv_router::publisher::kvstats;
        use dynamo_runtime::metrics::MetricsRegistry;

        // Test that publish() updates Prometheus gauges correctly using real Component
        let publisher = WorkerMetricsPublisher::new().unwrap();

        // Create a real DRT and component for integration testing
        let drt = create_test_drt_async().await;
        let namespace = drt.namespace("ns2002".to_string()).unwrap();
        let component = namespace.component("comp2002".to_string()).unwrap();

        // Register Prometheus metrics using the real constructor
        publisher.register_prometheus_metrics(&component).unwrap();

        // Get references to the gauges for testing
        let gauges = publisher.prometheus_gauges.get().unwrap();
        let active_blocks_gauge = gauges.kv_active_blocks_gauge.clone();
        let total_blocks_gauge = gauges.kv_total_blocks_gauge.clone();
        let cache_usage_gauge = gauges.gpu_cache_usage_gauge.clone();
        let hit_rate_gauge = gauges.gpu_prefix_cache_hit_rate_gauge.clone();

        // Create test metrics with specific values
        let test_metrics = Arc::new(ForwardPassMetrics {
            worker_stats: WorkerStats {
                data_parallel_rank: None,
                request_active_slots: 5,
                request_total_slots: 100,
                num_requests_waiting: 2,
            },
            kv_stats: KvStats {
                kv_active_blocks: 42,
                kv_total_blocks: 12894,
                gpu_cache_usage_perc: 0.5,
                gpu_prefix_cache_hit_rate: 0.75,
            },
            spec_decode_stats: None,
        });

        // Test 1: Initial gauge values should be 0
        assert_eq!(active_blocks_gauge.get(), 0.0);
        assert_eq!(total_blocks_gauge.get(), 0.0);
        assert_eq!(cache_usage_gauge.get(), 0.0);
        assert_eq!(hit_rate_gauge.get(), 0.0);

        // Test 2: publish() should update all gauges with correct values
        let result = publisher.publish(test_metrics);
        assert!(result.is_ok());

        // Test 3: Verify gauges were updated correctly
        assert_eq!(active_blocks_gauge.get(), 42.0);
        assert_eq!(total_blocks_gauge.get(), 12894.0);
        assert_eq!(cache_usage_gauge.get(), 0.5);
        assert_eq!(hit_rate_gauge.get(), 0.75);

        // Test 4: Verify metrics are properly registered in the component's registry
        // Component implements MetricsRegistry trait which provides prometheus_expfmt()
        let prometheus_output = component.prometheus_expfmt().unwrap();

        // Verify metric names are present
        assert!(prometheus_output.contains(kvstats::ACTIVE_BLOCKS));
        assert!(prometheus_output.contains(kvstats::TOTAL_BLOCKS));
        assert!(prometheus_output.contains(kvstats::GPU_CACHE_USAGE_PERCENT));
        assert!(prometheus_output.contains(kvstats::GPU_PREFIX_CACHE_HIT_RATE));

        // Test 5: Verify the prometheus output contains the actual values
        // Print the output to debug format issues
        println!("Prometheus output:\n{}", prometheus_output);

        // Check for metric values - the format includes labels so we need to be more flexible
        assert!(prometheus_output.contains("kvstats_active_blocks"));
        assert!(prometheus_output.contains("42")); // The value should be there
        assert!(prometheus_output.contains("kvstats_total_blocks"));
        assert!(prometheus_output.contains("12894")); // The value should be there
        assert!(prometheus_output.contains("kvstats_gpu_cache_usage_percent"));
        assert!(prometheus_output.contains("kvstats_gpu_prefix_cache_hit_rate"));

        println!(
            "✅ KvStatsPrometheusGauges constructor and publish() work correctly with real Component"
        );
    }
}
