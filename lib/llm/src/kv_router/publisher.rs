// SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::fmt;
use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::{Arc, OnceLock};
use std::time::Duration;

use anyhow::Result;
use rmp_serde as rmps;
use serde::Deserialize;
use serde::Serialize;
use serde::de::{self, Deserializer, IgnoredAny, MapAccess, SeqAccess, Visitor};
use tokio::sync::mpsc;
use tokio_util::sync::CancellationToken;
use zeromq::{Socket, SocketRecv, SubSocket};

use dynamo_runtime::metrics::{MetricsHierarchy, prometheus_names::kvstats};
use dynamo_runtime::traits::{
    DistributedRuntimeProvider, events::EventPublisher, events::EventSubscriber,
};
use dynamo_runtime::{
    component::{Component, Namespace},
    transports::nats::{NatsQueue, Slug},
};
use futures::StreamExt;

use crate::kv_router::{
    KV_EVENT_SUBJECT, KV_METRICS_SUBJECT, WORKER_KV_INDEXER_BUFFER_SIZE,
    WORKER_KV_INDEXER_QUERY_SUBJECT,
    indexer::{
        KvIndexerMetrics, LocalKvIndexer, RouterEvent, WorkerKvQueryRequest,
        compute_block_hash_for_seq,
    },
    protocols::*,
};
use dynamo_runtime::config::environment_names::nats as env_nats;

// Error handling configuration for ZMQ operations
const INITIAL_BACKOFF_MS: u64 = 10;
const MAX_BACKOFF_MS: u64 = 5000;
const MAX_CONSECUTIVE_ERRORS: u32 = 10;
const MAX_BACKOFF_EXPONENT: u32 = 8; // Cap at 2^8 = 256x multiplier to prevent overflow

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
        kv_block_size: u32,
        source_config: Option<KvEventSourceConfig>,
    ) -> Result<Self> {
        Self::new_with_local_indexer(component, kv_block_size, source_config, false)
    }

    pub fn new_with_local_indexer(
        component: Component,
        kv_block_size: u32,
        source_config: Option<KvEventSourceConfig>,
        enable_local_indexer: bool,
    ) -> Result<Self> {
        let cancellation_token = CancellationToken::new();

        let (tx, rx) = mpsc::unbounded_channel::<KvCacheEvent>();

        // Infer worker_id from component's connection
        let worker_id = component.drt().connection_id();

        let component_name = component.name();
        tracing::info!(
            "Initializing KvEventPublisher for worker {worker_id} in component {component_name}"
        );

        if enable_local_indexer {
            tracing::info!(
                "LocalKvIndexer enabled for worker {worker_id} in component {component_name}"
            );
        }

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

        // Create local indexer if requested
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

        // Spawn runtime for router->local indexer comm if requested
        let _local_indexer_query_handle = local_indexer.as_ref().map(|local_indexer_ref| {
            let component = component.clone();
            let local_indexer = local_indexer_ref.clone();

            component
                .drt()
                .runtime()
                .secondary()
                .spawn(start_worker_kv_query_service(
                    component,
                    worker_id,
                    local_indexer,
                    cancellation_token.clone(),
                ))
        });

        // Connect the NatsQueue before passing it to the event processor
        let cancellation_token_clone = cancellation_token.clone();
        let local_indexer_clone = local_indexer.clone();

        if enable_local_indexer {
            // When local indexer is enabled, use NATS Core (Component) for publishing.
            // This is simpler and doesn't require JetStream durability since recovery
            // is handled via the local indexer's event buffer.
            tracing::info!("Using NATS Core for KV event publishing (local_indexer mode)");
            let component_clone = component.clone();
            component.drt().runtime().secondary().spawn(async move {
                start_event_processor(
                    component_clone,
                    worker_id,
                    cancellation_token_clone,
                    rx,
                    local_indexer_clone,
                )
                .await
            });
        } else {
            // When local indexer is disabled, use JetStream (NatsQueue) for durability.
            let stream_name =
                Slug::slugify(&format!("{}.{}", component.subject(), KV_EVENT_SUBJECT))
                    .to_string()
                    .replace("_", "-");
            let nats_server = std::env::var(env_nats::NATS_SERVER)
                .unwrap_or_else(|_| "nats://localhost:4222".to_string());
            let mut nats_queue = NatsQueue::new_without_consumer(
                stream_name,
                nats_server,
                std::time::Duration::from_secs(60), // 1 minute timeout
            );

            component.drt().runtime().secondary().spawn(async move {
                if let Err(e) = nats_queue.connect().await {
                    tracing::error!("Failed to connect NatsQueue: {e}");
                    return;
                }
                start_event_processor(
                    nats_queue,
                    worker_id,
                    cancellation_token_clone,
                    rx,
                    local_indexer_clone,
                )
                .await
            });
        }

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
    worker_id: u64,
    cancellation_token: CancellationToken,
    mut rx: mpsc::UnboundedReceiver<KvCacheEvent>,
    local_indexer: Option<Arc<LocalKvIndexer>>,
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

                // Encapsulate in a router event.
                tracing::trace!("Event processor for worker_id {} processing event: {:?}", worker_id, event.data);
                let router_event = RouterEvent::new(worker_id, event);

                // Apply to local indexer first (if present)
                if let Some(indexer) = &local_indexer {
                    // Adds event into local indexer, and logs it into internal buffer
                    if let Err(e) = indexer.apply_event_with_buffer(router_event.clone()).await {
                        tracing::warn!(
                            "Failed to send event to local indexer for worker {}: {}",
                            worker_id,
                            e
                        );
                    }
                }

                // Then publish to NATS for global distribution
                // Use KV_EVENT_SUBJECT so both JetStream and NATS Core subscribers
                // can receive events on the expected subject.
                if let Err(e) = publisher.publish(KV_EVENT_SUBJECT, &router_event).await {
                    tracing::error!("Failed to publish event to NATS: {}", e);
                }

            }
        }
    }
}

// Processor for Router -> LocalKvIndexer query service
async fn start_worker_kv_query_service(
    component: Component,
    worker_id: u64,
    local_indexer: Arc<LocalKvIndexer>,
    cancellation_token: CancellationToken,
) {
    // Create NATS subscriber on a subject specific to worker's id
    let subject = format!("{}.{}", WORKER_KV_INDEXER_QUERY_SUBJECT, worker_id);
    let mut subscriber = match component.subscribe(&subject).await {
        Ok(sub) => sub,
        Err(e) => {
            tracing::error!(
                "Query service failed to subscribe for worker {worker_id} on subject {subject}: {e}"
            );
            return;
        }
    };
    tracing::info!("Query service listening on NATS for worker {worker_id} on subject {subject}");

    // Receive query request from router, retrieve event(s) from LocalKvIndexer, return response
    loop {
        tokio::select! {
            _ = cancellation_token.cancelled() => {
                tracing::info!("Query service received cancellation signal for worker {worker_id}");
                break;
            }

            msg = subscriber.next() => {
                let Some(msg) = msg else {
                    tracing::warn!("Query service NATS stream ended for worker {worker_id}");
                    break;
                };

                // deserialize from msg (async_nats::Message)
                let request: WorkerKvQueryRequest = match serde_json::from_slice(&msg.payload) {
                    Ok(request) => request,
                    Err(e) => {
                        tracing::error!("Failed to deserialize WorkerKvQueryRequest for worker {worker_id}: {e}");
                        continue;
                    }
                };

                tracing::debug!("Received query request for worker {worker_id}: {request:?}");

                // Query events based on optional start/end ids
                let response = local_indexer
                    .get_events_in_id_range(request.start_event_id, request.end_event_id)
                    .await;

                // Send reply back (if reply subject exists)
                if let Some(reply_subject) = msg.reply {
                    let payload = match serde_json::to_vec(&response) {
                        Ok(p) => p,
                        Err(e) => {
                            tracing::error!("Failed to serialize response for worker {worker_id}: {e}");
                            continue;
                        }
                    };

                    // Publish through DRT/NATS directly instead of namespace (adds a prefix)
                    if let Err(e) = component
                        .drt()
                        .kv_router_nats_publish(reply_subject.to_string(), payload.into())
                        .await
                    {
                        tracing::error!("Failed to send reply for worker {worker_id}: {e}");
                    }
                }
            }
        }
    }
}

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
                    tracing::warn!("Received unexpected ZMQ frame count: expected 3, actual {}", frames.len());
                    continue;
                }

                // Extract the payload and sequence number.
                let payload = frames.pop().unwrap();
                let seq_bytes = frames.pop().unwrap();

                if seq_bytes.len() != 8 {
                    tracing::warn!("Invalid sequence number byte length: expected 8, actual {}", seq_bytes.len());
                    continue;
                }

                let seq = u64::from_be_bytes(seq_bytes.try_into().unwrap());

                // Decode our batch of events.
                let batch_result = rmps::from_slice::<KvEventBatch>(&payload);
                let Ok(batch) = batch_result else {
                    let e = batch_result.unwrap_err();
                    tracing::warn!("Failed to decode KVEventBatch msgpack: {e}");
                    continue;
                };

                tracing::trace!(
                    "ZMQ listener on {} received batch with {} events (seq={}, dp_rank={})",
                    zmq_endpoint,
                    batch.events.len(),
                    seq,
                    batch.data_parallel_rank.unwrap_or(0)
                );

                let dp_rank = batch.data_parallel_rank.unwrap_or(0) as u32;
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
            block_mm_infos,
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
                        block_mm_infos.as_deref(),
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
    mm_extra_info: Option<BlockExtraInfo>,
) -> KvCacheStoredBlockData {
    // Compute tokens_hash including MM info if present
    let block_mm_infos = mm_extra_info.as_ref().map(|info| vec![Some(info.clone())]);
    let tokens_hash =
        compute_block_hash_for_seq(token_ids, kv_block_size, block_mm_infos.as_deref())[0];

    tracing::trace!(
        "Creating stored block: external_block_hash={}, tokens_hash={}, token_ids={:?}, kv_block_size={}, mm_extra_info={:?}",
        block_hash,
        tokens_hash.0,
        token_ids,
        kv_block_size,
        mm_extra_info
    );
    KvCacheStoredBlockData {
        block_hash: ExternalSequenceBlockHash::from(block_hash),
        tokens_hash,
        mm_extra_info,
    }
}

pub fn create_stored_blocks(
    kv_block_size: u32,
    token_ids: &[u32],
    num_block_tokens: &[u64],
    block_hashes: &[u64],
    lora_id: u64,
    warning_count: &Arc<AtomicU32>,
    block_mm_infos: Option<&[Option<BlockExtraInfo>]>,
) -> Vec<KvCacheStoredBlockData> {
    let mut blocks: Vec<KvCacheStoredBlockData> = Vec::new();

    let mut token_offset: usize = 0;
    for (block_idx, (num_tokens_it, block_hash_it)) in
        num_block_tokens.iter().zip(block_hashes.iter()).enumerate()
    {
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
        let mm_extra_info = block_mm_infos
            .and_then(|infos| infos.get(block_idx))
            .and_then(|opt| opt.clone());

        blocks.push(create_stored_block_from_parts(
            kv_block_size,
            *block_hash_it,
            tokens,
            lora_id,
            mm_extra_info,
        ));
        token_offset += *num_tokens_it as usize;
    }

    blocks
}

// -------------------------------------------------------------------------
// Types mirroring the Python msgspec-defined structures -------------------
// -------------------------------------------------------------------------

#[derive(Debug, Serialize)]
struct KvEventBatch {
    ts: f64,
    events: Vec<RawKvEvent>,
    #[serde(alias = "dp_rank")]
    data_parallel_rank: Option<i32>,
}

impl<'de> Deserialize<'de> for KvEventBatch {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        // Deserialize from array format: [timestamp, [events], data_parallel_rank]
        let arr: (f64, Vec<RawKvEvent>, Option<i32>) = Deserialize::deserialize(deserializer)?;
        Ok(KvEventBatch {
            ts: arr.0,
            events: arr.1,
            data_parallel_rank: arr.2,
        })
    }
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

#[derive(Debug, Serialize, Clone)]
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
        /// Multimodal extra info for each block (length should match block_hashes)
        #[serde(default, skip_serializing_if = "Option::is_none")]
        block_mm_infos: Option<Vec<Option<BlockExtraInfo>>>,
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
        let mut block_mm_infos: Option<Option<Vec<Option<BlockExtraInfo>>>> = None;

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
                "block_mm_infos" => {
                    block_mm_infos = Some(map.next_value()?);
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
                    block_mm_infos: block_mm_infos.unwrap_or(None),
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
                let block_mm_infos: Option<Vec<Option<BlockExtraInfo>>> =
                    seq.next_element()?.unwrap_or(None);

                while seq.next_element::<IgnoredAny>()?.is_some() {}

                Ok(RawKvEvent::BlockStored {
                    block_hashes,
                    parent_block_hash,
                    token_ids,
                    block_size,
                    lora_id,
                    medium,
                    block_mm_infos,
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
        let kv_active_blocks_gauge = component.metrics().create_gauge(
            kvstats::ACTIVE_BLOCKS,
            "Number of active KV cache blocks currently in use",
            &[],
        )?;

        let kv_total_blocks_gauge = component.metrics().create_gauge(
            kvstats::TOTAL_BLOCKS,
            "Total number of KV cache blocks available",
            &[],
        )?;

        let gpu_cache_usage_gauge = component.metrics().create_gauge(
            kvstats::GPU_CACHE_USAGE_PERCENT,
            "GPU cache usage as a percentage (0.0-1.0)",
            &[],
        )?;

        let gpu_prefix_cache_hit_rate_gauge = component.metrics().create_gauge(
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
        let worker_id = component.drt().connection_id();
        self.start_nats_metrics_publishing(component.namespace().clone(), worker_id);
        Ok(())
    }

    /// Starts a background task to publish metrics over NATS
    ///
    /// This task monitors metric changes (specifically kv_active_blocks and num_requests_waiting)
    /// and publishes stable metrics to NATS after they've been unchanged for 1ms.
    fn start_nats_metrics_publishing(&self, namespace: Namespace, worker_id: u64) {
        let nats_rx = self.rx.clone();

        tokio::spawn(async move {
            let mut rx = nats_rx;
            let mut last_kv_active_blocks: Option<u64> = Some(0);
            let mut last_num_requests_waiting: Option<u64> = Some(0);
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
                            // Create ActiveLoad with only active_decode_blocks (worker doesn't know prefill tokens)
                            let active_load = ActiveLoad {
                                worker_id,
                                dp_rank: metrics.worker_stats.data_parallel_rank.unwrap_or(0),
                                active_decode_blocks: Some(metrics.kv_stats.kv_active_blocks),
                                active_prefill_tokens: None,
                            };

                            if let Err(e) =
                                namespace.publish(KV_METRICS_SUBJECT, &active_load).await
                            {
                                tracing::warn!("Failed to publish metrics over NATS: {}", e);
                            }
                        }

                        // Reset timer to pending state to avoid tight loop
                        // It will be reset to 1ms when metrics actually change
                        publish_timer.as_mut().reset(
                            tokio::time::Instant::now() + tokio::time::Duration::from_secs(3600)
                        );
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

        let stored = create_stored_block_from_parts(kv_block_size, blk_hash, &token_ids, 0, None);

        assert_eq!(stored.block_hash.0, blk_hash);
        let expected_hash = compute_block_hash_for_seq(&token_ids, 4, None)[0];
        assert_eq!(stored.tokens_hash, expected_hash);
        assert!(stored.mm_extra_info.is_none());
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
            None,
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
            None,
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
            block_mm_infos: None,
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
    use crate::kv_router::KvIndexer;
    use crate::kv_router::indexer::KvIndexerInterface;
    use crate::kv_router::protocols::{ExternalSequenceBlockHash, LocalBlockHash};
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
        ) -> anyhow::Result<()> {
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
        ) -> anyhow::Result<()> {
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

        let handle = tokio::spawn(start_event_processor(component, 1, token, rx, None));

        tokio::time::timeout(tokio::time::Duration::from_secs(1), handle)
            .await
            .unwrap()
            .unwrap();

        let published = published.lock().unwrap();
        assert_eq!(published.len(), 1);
        let (subject, _) = &published[0];
        assert_eq!(subject, KV_EVENT_SUBJECT);
    }

    //--------------------------------------------------------------------
    // Test start_event_processor with local indexer
    //--------------------------------------------------------------------
    #[tokio::test]
    async fn test_start_event_processor_with_local_indexer() {
        let (component, published) = MockComponent::new();

        // Create a local indexer
        let token = CancellationToken::new();
        let metrics = Arc::new(KvIndexerMetrics::new_unregistered());
        let local_indexer = Arc::new(LocalKvIndexer::new(token.clone(), 4, metrics, 100));

        // Create BlockStored event
        let event = KvCacheEvent {
            event_id: 1,
            data: KvCacheEventData::Stored(KvCacheStoreData {
                parent_hash: None,
                blocks: vec![
                    KvCacheStoredBlockData {
                        block_hash: ExternalSequenceBlockHash(100),
                        tokens_hash: LocalBlockHash(200),
                        mm_extra_info: None,
                    },
                    KvCacheStoredBlockData {
                        block_hash: ExternalSequenceBlockHash(101),
                        tokens_hash: LocalBlockHash(201),
                        mm_extra_info: None,
                    },
                ],
            }),
            dp_rank: 0,
        };

        let (tx, rx) = mpsc::unbounded_channel::<KvCacheEvent>();
        tx.send(event).unwrap();
        drop(tx);

        // Start event processor with local indexer
        let handle = tokio::spawn(start_event_processor(
            component,
            1,
            token.clone(),
            rx,
            Some(local_indexer.clone()), // arc::clone just increments atomic counters
        ));

        // Wait for processing
        tokio::time::timeout(tokio::time::Duration::from_secs(1), handle)
            .await
            .unwrap()
            .unwrap();

        // Verify event was published to NATS (same as test_start_event_processor)
        {
            let published_events = published.lock().unwrap();
            assert_eq!(published_events.len(), 1);
            let (subject, _) = &published_events[0];
            assert_eq!(subject, KV_EVENT_SUBJECT);
        } // drop lock

        // Verify event was applied to local indexer
        // We can check by querying the workers that have blocks
        let get_workers_tx = local_indexer.get_workers_sender();
        let mut found = false;
        for _ in 0..20 {
            // Try up to 20 times (200ms total)
            let (resp_tx, resp_rx) = tokio::sync::oneshot::channel();
            get_workers_tx
                .send(crate::kv_router::indexer::GetWorkersRequest { resp: resp_tx })
                .await
                .unwrap();
            let workers: Vec<u64> = resp_rx.await.unwrap();

            if workers.contains(&1) {
                found = true;
                break;
            }

            // Wait before retrying
            tokio::time::sleep(tokio::time::Duration::from_millis(10)).await;
        }

        // Worker 1 should be in the set (we used worker_id=1)
        assert!(
            found,
            "Worker 1 was not found in the indexer after processing"
        );

        // Cleanup
        token.cancel();
    }

    //--------------------------------------------------------------------
    // Test BlockRemoved event with local indexer
    //--------------------------------------------------------------------
    #[tokio::test]
    async fn test_event_processor_block_removed_with_local_indexer() {
        let (component, published) = MockComponent::new();

        let token = CancellationToken::new();
        let metrics = Arc::new(KvIndexerMetrics::new_unregistered());
        let local_indexer = Arc::new(LocalKvIndexer::new(token.clone(), 4, metrics, 100));

        // First, store a block
        let store_event = KvCacheEvent {
            event_id: 1,
            data: KvCacheEventData::Stored(KvCacheStoreData {
                parent_hash: None,
                blocks: vec![KvCacheStoredBlockData {
                    block_hash: ExternalSequenceBlockHash(100),
                    tokens_hash: LocalBlockHash(200),
                    mm_extra_info: None,
                }],
            }),
            dp_rank: 0,
        };

        let (tx, rx) = mpsc::unbounded_channel::<KvCacheEvent>();
        tx.send(store_event).unwrap();

        // Start event processor with local indexer
        let handle = tokio::spawn(start_event_processor(
            component,
            1,
            token.clone(),
            rx,
            Some(local_indexer.clone()),
        ));

        // Then remove same event
        let remove_event = KvCacheEvent {
            event_id: 2,
            data: KvCacheEventData::Removed(KvCacheRemoveData {
                block_hashes: vec![ExternalSequenceBlockHash(100)],
            }),
            dp_rank: 0,
        };
        tx.send(remove_event).unwrap();
        drop(tx);

        tokio::time::timeout(tokio::time::Duration::from_secs(1), handle)
            .await
            .unwrap()
            .unwrap();

        // Local indexer should have no block
        let mut no_blocks = false;
        for _ in 0..20 {
            // Try up to 20 times (200ms total)
            let scores = local_indexer
                .find_matches(vec![LocalBlockHash(200)])
                .await
                .unwrap();
            if scores.scores.is_empty() {
                no_blocks = true;
                break;
            }
            tokio::time::sleep(tokio::time::Duration::from_millis(10)).await;
        }
        assert!(no_blocks, "worker should have no blocks after removal");

        // Global kvindexer should have recieved two events (create/remove)
        let published = published.lock().unwrap();
        assert_eq!(
            published.len(),
            2,
            "expected 2 published events, found {}",
            published.len()
        );

        token.cancel();
    }

    //--------------------------------------------------------------------
    // Test AllBlocksCleared event with local indexer
    //--------------------------------------------------------------------
    #[tokio::test]
    async fn test_event_processor_all_blocks_cleared_with_local_indexer() {
        let (component, published) = MockComponent::new();

        let token = CancellationToken::new();
        let metrics = Arc::new(KvIndexerMetrics::new_unregistered());
        let local_indexer = Arc::new(LocalKvIndexer::new(token.clone(), 4, metrics, 100));

        // Store a block
        let store_event = KvCacheEvent {
            event_id: 1,
            data: KvCacheEventData::Stored(KvCacheStoreData {
                parent_hash: None,
                blocks: vec![KvCacheStoredBlockData {
                    block_hash: ExternalSequenceBlockHash(100),
                    tokens_hash: LocalBlockHash(200),
                    mm_extra_info: None,
                }],
            }),
            dp_rank: 0,
        };

        let (tx, rx) = mpsc::unbounded_channel::<KvCacheEvent>();
        tx.send(store_event).unwrap();

        // Clear all blocks
        let clear_event = KvCacheEvent {
            event_id: 2,
            data: KvCacheEventData::Cleared,
            dp_rank: 0,
        };
        tx.send(clear_event).unwrap();
        drop(tx);

        // Create event processor and wait
        let handle = tokio::spawn(start_event_processor(
            component,
            1,
            token.clone(),
            rx,
            Some(local_indexer.clone()),
        ));

        tokio::time::timeout(tokio::time::Duration::from_secs(1), handle)
            .await
            .unwrap()
            .unwrap();

        // Local indexer should have no block
        let mut no_blocks = false;
        for _ in 0..20 {
            // Try up to 20 times (200ms total)
            let scores = local_indexer
                .find_matches(vec![LocalBlockHash(200)])
                .await
                .unwrap();
            if scores.scores.is_empty() {
                no_blocks = true;
                break;
            }
            tokio::time::sleep(tokio::time::Duration::from_millis(10)).await;
        }
        assert!(no_blocks, "worker should have no blocks after clearing");

        // Global kvindexer should have recieved two events (create/remove)
        let published = published.lock().unwrap();
        assert_eq!(
            published.len(),
            2,
            "expected 2 published events, found {}",
            published.len()
        );

        token.cancel();
    }

    //--------------------------------------------------------------------
    // Test that local indexer failure doesn't break NATS publishing
    //--------------------------------------------------------------------
    #[tokio::test]
    async fn test_event_processor_local_indexer_failure_continues() {
        let (component, published) = MockComponent::new();

        let token = CancellationToken::new();
        let metrics = Arc::new(KvIndexerMetrics::new_unregistered());
        let local_indexer = Arc::new(LocalKvIndexer::new(token.clone(), 4, metrics, 100));

        // cancel indexer immediately to simulate failure
        token.cancel();

        let event = KvCacheEvent {
            event_id: 1,
            data: KvCacheEventData::Removed(KvCacheRemoveData {
                block_hashes: vec![ExternalSequenceBlockHash(1)],
            }),
            dp_rank: 0,
        };

        let new_token = CancellationToken::new();
        let (tx, rx) = mpsc::unbounded_channel::<KvCacheEvent>();
        tx.send(event).unwrap();
        drop(tx);

        // Despite local indexer being cancelled, event processor should continue
        let handle = tokio::spawn(start_event_processor(
            component,
            1,
            new_token,
            rx,
            Some(local_indexer),
        ));

        tokio::time::timeout(tokio::time::Duration::from_secs(1), handle)
            .await
            .unwrap()
            .unwrap();

        // Verify event was still published to NATS despite local indexer failure
        let published_events = published.lock().unwrap();
        assert_eq!(published_events.len(), 1);
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
            block_mm_infos: None,
        }];

        let batch = KvEventBatch {
            ts: 0.0,
            events,
            data_parallel_rank: Some(1),
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

    //--------------------------------------------------------------------
    // Test distributed recovery: Router queries worker's LocalKvIndexer after outage
    //--------------------------------------------------------------------
    #[tokio::test]
    async fn test_distributed_kvindexer_recovery_from_outage() {
        let worker_1_id = 1u64;
        let block_size = 4u32;
        let token = CancellationToken::new();

        // === SETUP: Worker Components ===
        let (worker_component, worker_published) = MockComponent::new();
        let local_indexer_1 = Arc::new(LocalKvIndexer::new(
            token.clone(),
            block_size,
            Arc::new(KvIndexerMetrics::new_unregistered()),
            100, // buffer size
        ));

        let (worker_tx, worker_rx) = mpsc::unbounded_channel::<KvCacheEvent>();

        // Start worker's event processor
        tokio::spawn(start_event_processor(
            worker_component,
            worker_1_id,
            token.clone(),
            worker_rx,
            Some(local_indexer_1.clone()),
        ));

        // === SETUP: Router Components ===
        let router_indexer = Arc::new(KvIndexer::new(
            token.clone(),
            block_size,
            Arc::new(KvIndexerMetrics::new_unregistered()),
        ));

        // === STEP 1: Normal Operation ===
        let event_1 = KvCacheEvent {
            event_id: 1,
            data: KvCacheEventData::Stored(KvCacheStoreData {
                parent_hash: None,
                blocks: vec![
                    KvCacheStoredBlockData {
                        block_hash: ExternalSequenceBlockHash(100),
                        tokens_hash: LocalBlockHash(200),
                        mm_extra_info: None,
                    },
                    KvCacheStoredBlockData {
                        block_hash: ExternalSequenceBlockHash(101),
                        tokens_hash: LocalBlockHash(201),
                        mm_extra_info: None,
                    },
                ],
            }),
            dp_rank: 0,
        };

        worker_tx.send(event_1.clone()).unwrap();
        tokio::time::sleep(tokio::time::Duration::from_millis(100)).await;

        // Simulate JetStream: forward worker's published event to router
        let (subject, bytes) = {
            let published = worker_published.lock().unwrap();
            assert_eq!(published.len(), 1, "Worker should have published 1 event");
            (published[0].0.clone(), published[0].1.clone())
        }; // drop worker_published before await
        assert_eq!(subject, KV_EVENT_SUBJECT);

        let router_event: RouterEvent = rmp_serde::from_slice(&bytes).unwrap();
        router_indexer
            .event_sender()
            .send(router_event)
            .await
            .unwrap();

        tokio::time::sleep(tokio::time::Duration::from_millis(100)).await;

        // assert: Router's indexer has event
        let get_workers_tx = router_indexer.get_workers_sender();
        let mut router_has_worker = false;
        for _ in 0..20 {
            let (resp_tx, resp_rx) = tokio::sync::oneshot::channel();
            get_workers_tx
                .send(crate::kv_router::indexer::GetWorkersRequest { resp: resp_tx })
                .await
                .unwrap();
            let workers: Vec<u64> = resp_rx.await.unwrap();
            if workers.contains(&worker_1_id) {
                router_has_worker = true;
                break;
            }
            tokio::time::sleep(tokio::time::Duration::from_millis(10)).await;
        }
        assert!(
            router_has_worker,
            "Router should see worker 1 after normal operation"
        );

        // assert: Worker's local indexer buffered event
        let buffered = local_indexer_1.get_all_events_in_buffer();
        assert_eq!(buffered.len(), 1, "Local indexer should buffer 1 event");

        // === STEP 2 & 3: Simulate Outage - Stop forwarding to router ===
        let event_2 = KvCacheEvent {
            event_id: 2,
            data: KvCacheEventData::Stored(KvCacheStoreData {
                parent_hash: None,
                blocks: vec![
                    KvCacheStoredBlockData {
                        block_hash: ExternalSequenceBlockHash(100), // Shared prefix
                        tokens_hash: LocalBlockHash(200),
                        mm_extra_info: None,
                    },
                    KvCacheStoredBlockData {
                        block_hash: ExternalSequenceBlockHash(102), // New block
                        tokens_hash: LocalBlockHash(202),
                        mm_extra_info: None,
                    },
                ],
            }),
            dp_rank: 0,
        };

        worker_tx.send(event_2.clone()).unwrap(); // send to worker but not to router
        tokio::time::sleep(tokio::time::Duration::from_millis(100)).await;

        // assert: Worker published event_2 to "NATS" (MockComponent)
        {
            let published = worker_published.lock().unwrap();
            assert_eq!(
                published.len(),
                2,
                "Worker should have published 2 events total"
            );
        }

        // assert: Worker's local indexer has both events
        let buffered = local_indexer_1.get_all_events_in_buffer();
        assert_eq!(
            buffered.len(),
            2,
            "Local indexer should have both events during outage"
        );

        // assert: Router DOESN'T have event_2
        let block_hashes_2 = vec![LocalBlockHash(200), LocalBlockHash(202)];
        let overlap = router_indexer
            .find_matches(block_hashes_2.clone())
            .await
            .unwrap();
        let router_overlap = overlap
            .scores
            .get(&crate::kv_router::protocols::WorkerWithDpRank::from_worker_id(worker_1_id))
            .copied()
            .unwrap_or(0);
        assert_eq!(
            router_overlap, 1,
            "Router should only see 1 shared block (not the new block from event_2)"
        );

        // === STEP 4 & 5: Recovery - Query worker's local indexer for missed events ===
        // In practice, the subscriber detects gaps and triggers recovery automatically.
        // Here we simulate that by querying for events after event_id=1.
        let last_known_id = 1u64; // Router only received event_1
        let response = local_indexer_1
            .get_events_in_id_range(Some(last_known_id + 1), None)
            .await;
        let missed_events = match response {
            crate::kv_router::indexer::WorkerKvQueryResponse::Events(e) => e,
            crate::kv_router::indexer::WorkerKvQueryResponse::TreeDump(e) => e,
            other => panic!("Unexpected response: {:?}", other),
        };
        assert_eq!(
            missed_events.len(),
            1,
            "Should get 1 missed event (event_2 with id=2)"
        );

        // Step 5: Apply missed events to router
        for router_event in missed_events {
            router_indexer
                .event_sender()
                .send(router_event)
                .await
                .unwrap();
        }

        tokio::time::sleep(tokio::time::Duration::from_millis(100)).await;

        // assert: Router now has complete state
        let overlap = router_indexer.find_matches(block_hashes_2).await.unwrap();
        let router_overlap_after = overlap
            .scores
            .get(&crate::kv_router::protocols::WorkerWithDpRank::from_worker_id(worker_1_id))
            .copied()
            .unwrap_or(0);
        assert_eq!(
            router_overlap_after, 2,
            "Router should now see both blocks after recovery"
        );

        token.cancel();
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
    use crate::kv_router::protocols::{ActiveLoad, ForwardPassMetrics, KvStats, WorkerStats};
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
            .subscribe_with_type::<ActiveLoad>(KV_METRICS_SUBJECT)
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
        assert_eq!(event.active_decode_blocks, Some(900)); // Last value: 9 * 100
        assert_eq!(event.active_prefill_tokens, None); // Worker doesn't publish prefill tokens

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
        let prometheus_output = component.metrics().prometheus_expfmt().unwrap();

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

#[cfg(all(test, feature = "integration"))]
mod test_integration_publisher_with_kvindexer {
    use super::*;

    use crate::kv_router::scheduler::DefaultWorkerSelector;
    use crate::kv_router::{KvPushRouter, KvRouter, KvRouterConfig};
    use crate::local_model::LocalModelBuilder;
    use crate::local_model::runtime_config::ModelRuntimeConfig;
    use crate::mocker::engine::{MOCKER_COMPONENT, MockVllmEngine};
    use crate::mocker::protocols::MockEngineArgs;
    use crate::protocols::common::llm_backend::{LLMEngineOutput, PreprocessedRequest};
    use crate::protocols::common::{OutputOptions, SamplingOptions, StopConditions};
    use dynamo_runtime::distributed_test_utils::create_test_shared_drt_async;
    use dynamo_runtime::engine::AsyncEngine;
    use dynamo_runtime::pipeline::{Context, PushRouter, RouterMode, network::Ingress};
    use dynamo_runtime::protocols::annotated::Annotated;

    /// Integration test: KvPushRouter end-to-end routing with mock engines.
    #[tokio::test(flavor = "multi_thread")]
    #[ignore] // Requires NATS/etcd. Run with: cargo test --package dynamo-llm --lib --features integration test_distributed_kvindexer_e2e -- --ignored --nocapture
    async fn test_distributed_kvindexer_e2e() -> anyhow::Result<()> {
        const BLOCK_SIZE: u32 = 4;
        const NUM_REQUESTS: usize = 4;

        dynamo_runtime::logging::init();

        // === SETUP: Distributed runtimes and namespaces ===
        let shared_store_dir = tempfile::tempdir()?;
        let shared_store_path = shared_store_dir.path().to_path_buf();

        // Make both runtimes point at the same file-backed storage backend so worker
        // registrations and heartbeats remain visible to every DRT instance.
        let distributed1 = create_test_shared_drt_async(&shared_store_path).await;
        let distributed2 = create_test_shared_drt_async(&shared_store_path).await;
        let component1 = distributed1
            .namespace("test_e2e_router")?
            .component(MOCKER_COMPONENT)?;
        let component2 = distributed2
            .namespace("test_e2e_router")?
            .component(MOCKER_COMPONENT)?;

        // === SETUP: Start mocker workers  ===
        let mocker_args = MockEngineArgs::builder()
            .block_size(BLOCK_SIZE as usize)
            .dp_size(1) // single worker per runtime
            .enable_prefix_caching(true)
            .enable_local_indexer(true) // affects scheduler/publisher args
            .build()?;

        let worker_components = vec![component1.clone(), component2.clone()];
        let mut server_handles = Vec::new();
        let mut worker_ids = Vec::new();

        for comp in worker_components {
            let engine = Arc::new(MockVllmEngine::new(mocker_args.clone()));
            engine.start(comp.clone()).await?;
            tracing::info!("MockVllmEngine started for {:?}", comp);

            // Register MDC with runtime_config so router can discover enable_local_indexer.
            // (Without this step, the MDC-based assert in query_worker() in worker_query.rs will fail.)
            // This inlines code which in the Python path would be performed by:
            // - local_model.rs: LocalModelBuilder::build() sets runtime_config from MockEngineArgs
            // - entrypoint/input/endpoint.rs: LocalModel::attach() registers MDC via discovery
            let endpoint = comp.endpoint("generate");
            let runtime_config = ModelRuntimeConfig {
                enable_local_indexer: true,
                ..Default::default()
            };
            let mut builder = LocalModelBuilder::default();
            builder
                .model_name(Some("mock".to_string()))
                .kv_cache_block_size(Some(BLOCK_SIZE))
                .runtime_config(runtime_config);
            let mut local_model = builder.build().await?;
            local_model
                .attach(
                    &endpoint,
                    crate::model_type::ModelType::Chat,
                    crate::model_type::ModelInput::Tokens,
                    None,
                )
                .await?;

            let ingress = Ingress::for_engine(engine.clone())?;
            let endpoint_component = comp.clone();
            let handle = tokio::spawn(async move {
                if let Err(e) = endpoint_component
                    .endpoint("generate")
                    .endpoint_builder()
                    .handler(ingress)
                    .start()
                    .await
                {
                    tracing::error!("Generate endpoint failed: {e}");
                }
            });
            server_handles.push(handle);
            worker_ids.push(comp.drt().connection_id());
        }
        tracing::info!("Generate endpoint servers launched");

        tokio::time::sleep(Duration::from_millis(500)).await;

        // === SETUP: Build KvPushRouter ===
        let router_distributed = create_test_shared_drt_async(&shared_store_path).await;
        let router_namespace = router_distributed.namespace("test_e2e_router")?;
        let backend_component = router_namespace.component(MOCKER_COMPONENT)?;
        let backend_endpoint = backend_component.endpoint("generate");
        let client = backend_endpoint.client().await?;
        let kv_router_config = KvRouterConfig::default();
        let selector = Box::new(DefaultWorkerSelector::new(Some(kv_router_config)));
        let consumer_id = format!("test-router-{}", router_distributed.connection_id());

        let kv_router: Arc<KvRouter> = Arc::new(
            KvRouter::new(
                backend_endpoint.clone(),
                client.clone(),
                BLOCK_SIZE,
                Some(selector),
                Some(kv_router_config),
                consumer_id,
            )
            .await?,
        );

        let push_router =
               PushRouter::<PreprocessedRequest, Annotated<LLMEngineOutput>>::from_client_with_threshold(
                   client,
                   RouterMode::KV,
                   None,
                   None,
               )
               .await?;

        let kv_push_router = KvPushRouter::new(push_router, kv_router.clone());

        // ===== TEST PART 1: ROUTE & SEND REQUESTS TO WORKERS (ROUTER -> WORKER) =====
        let create_request = |tokens: Vec<u32>| {
            PreprocessedRequest::builder()
                .model("mock".to_string())
                .token_ids(tokens)
                .stop_conditions(StopConditions {
                    max_tokens: Some(10),
                    ..Default::default()
                })
                .sampling_options(SamplingOptions::default())
                .output_options(OutputOptions::default())
                .build()
                .unwrap()
        }; // from mocker/engine.rs

        for i in 0..NUM_REQUESTS {
            tracing::info!("Sending routed request {}", i + 1);
            let tokens = vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10, i as u32];
            let request = create_request(tokens.clone());

            let response_stream = kv_push_router.generate(Context::new(request)).await?;
            let responses: Vec<Annotated<LLMEngineOutput>> = response_stream.collect().await;
            assert!(
                !responses.is_empty(),
                "Request {} should produce at least one response",
                i + 1
            );
        }

        tracing::info!("KvPushRouter generate() succeeded for {NUM_REQUESTS} requests");

        // ===== TEST PART 2: QUERY WORKER-LOCAL KVINDEXERS DIRECTLY =====
        // TODO: This could be refactored as router function (e.g. router.refresh_from_worker(worker_id))
        // (which should also update the global kvIndexer with the buffer from the local kvIndexer)
        let mut best_worker_info: Option<(u64, usize)> = None;

        // Exactly one worker should have been routed requests. Find that worker
        for &worker_id in &worker_ids {
            let response = kv_router
                .query_worker_local_kv(worker_id, None, None)
                .await?;
            let events = match response {
                crate::kv_router::indexer::WorkerKvQueryResponse::Events(e) => e,
                crate::kv_router::indexer::WorkerKvQueryResponse::TreeDump(e) => e,
                _ => vec![],
            };
            if events.is_empty() {
                continue;
            }

            let event_count = events.len();
            tracing::info!(
                worker_id,
                events = event_count,
                "Worker query on worker {worker_id} returned buffered KV events"
            );
            best_worker_info = Some((worker_id, event_count));
            break;
        }

        // Verify that only one worker has KV events in buffer
        let (best_worker_id, best_worker_event_count) =
            best_worker_info.expect("At least one worker should have buffered KV events");

        tracing::info!(
            "Best worker is {best_worker_id} with {best_worker_event_count} buffered KV events"
        );

        for &worker_id in &worker_ids {
            if worker_id == best_worker_id {
                continue;
            }

            let response = kv_router
                .query_worker_local_kv(worker_id, None, None)
                .await?;
            let events = match response {
                crate::kv_router::indexer::WorkerKvQueryResponse::Events(e) => e,
                crate::kv_router::indexer::WorkerKvQueryResponse::TreeDump(e) => e,
                _ => vec![],
            };
            assert!(
                events.is_empty(),
                "Worker {worker_id} should not report buffered KV events; best worker {best_worker_id} reported {best_worker_event_count}"
            );
        }

        // === Cleanup ===
        for handle in server_handles {
            handle.abort();
        }
        distributed1.shutdown();
        distributed2.shutdown();
        router_distributed.shutdown();

        Ok(())
    }

    #[tokio::test(flavor = "multi_thread")]
    #[ignore]
    async fn test_distributed_kvindexer_e2e_startup() -> anyhow::Result<()> {
        const BLOCK_SIZE: u32 = 4;

        dynamo_runtime::logging::init();

        // === SETUP: Distributed runtimes and namespaces ===
        let shared_store_dir = tempfile::tempdir()?;
        let shared_store_path = shared_store_dir.path().to_path_buf();

        // Use a unique namespace per test run for full isolation
        let test_namespace = format!("test_e2e_{}", uuid::Uuid::new_v4().simple());

        // Make both runtimes point at the same file-backed storage backend so worker
        // registrations and heartbeats remain visible to every DRT instance.
        let distributed1 = create_test_shared_drt_async(&shared_store_path).await;
        let distributed2 = create_test_shared_drt_async(&shared_store_path).await;
        let component1 = distributed1
            .namespace(&test_namespace)?
            .component(MOCKER_COMPONENT)?;
        let component2 = distributed2
            .namespace(&test_namespace)?
            .component(MOCKER_COMPONENT)?;

        // === SETUP: Start mocker workers  ===
        let mocker_args = MockEngineArgs::builder()
            .block_size(BLOCK_SIZE as usize)
            .dp_size(1) // single worker per runtime
            .enable_prefix_caching(true)
            .enable_local_indexer(true) // affects scheduler/publisher args
            .build()?;

        let worker_components = vec![component1.clone(), component2.clone()];
        let mut server_handles = Vec::new();
        let mut worker_ids = Vec::new();

        for comp in worker_components {
            let engine: Arc<MockVllmEngine> = Arc::new(MockVllmEngine::new(mocker_args.clone()));
            engine.start(comp.clone()).await?;
            tracing::info!("MockVllmEngine started for {:?}", comp);

            // Register MDC with runtime_config so router can discover enable_local_indexer.
            // (Without this step, the MDC-based assert in query_worker() in worker_query.rs will fail.)
            // This inlines code which in the Python path would be performed by:
            // - local_model.rs: LocalModelBuilder::build() sets runtime_config from MockEngineArgs
            // - entrypoint/input/endpoint.rs: LocalModel::attach() registers MDC via discovery
            let endpoint = comp.endpoint("generate");
            let runtime_config = ModelRuntimeConfig {
                enable_local_indexer: true,
                ..Default::default()
            };
            let mut builder = LocalModelBuilder::default();
            builder
                .model_name(Some("mock".to_string()))
                .kv_cache_block_size(Some(BLOCK_SIZE))
                .runtime_config(runtime_config);
            let mut local_model = builder.build().await?;
            local_model
                .attach(
                    &endpoint,
                    crate::model_type::ModelType::Chat,
                    crate::model_type::ModelInput::Tokens,
                    None,
                )
                .await?;

            let ingress = Ingress::for_engine(engine.clone())?;
            let endpoint_component = comp.clone();
            let handle = tokio::spawn(async move {
                if let Err(e) = endpoint_component
                    .endpoint("generate")
                    .endpoint_builder()
                    .handler(ingress)
                    .start()
                    .await
                {
                    tracing::error!("Generate endpoint failed: {e}");
                }
            });
            server_handles.push(handle);
            worker_ids.push(comp.drt().connection_id());
        }
        tracing::info!("Generate endpoint servers launched");

        tokio::time::sleep(Duration::from_millis(500)).await;

        // === STEP 1: Send request to worker_ids[0] to populate its local indexer ===
        // This simulates a situation where KvPushRouter is initialized
        // to route to workers which already have KV events
        let pre_router_distributed = create_test_shared_drt_async(&shared_store_path).await;
        let pre_backend_endpoint = pre_router_distributed
            .namespace(&test_namespace)?
            .component(MOCKER_COMPONENT)?
            .endpoint("generate");
        let pre_client = pre_backend_endpoint.client().await?;

        // Wait for the client to discover both workers
        let discovery_timeout = Duration::from_secs(5);
        let discovery_start = std::time::Instant::now();
        loop {
            let instances = pre_client.instance_source.as_ref().borrow().clone();
            if instances.len() >= 2 {
                tracing::info!("Discovered {} workers", instances.len());
                break;
            }
            if discovery_start.elapsed() > discovery_timeout {
                anyhow::bail!(
                    "Timed out waiting for worker discovery: expected 2, found {}",
                    instances.len()
                );
            }
            tokio::time::sleep(Duration::from_millis(50)).await;
        }

        // Create a PushRouter to send requests directly to a specific worker
        let pre_push_router =
            PushRouter::<PreprocessedRequest, Annotated<LLMEngineOutput>>::from_client_with_threshold(
                pre_client,
                RouterMode::Random, // We'll use direct() so mode doesn't matter
                None,
                None,
            )
            .await?;

        // Force sending one requests each to the two workers
        for &worker_id in &worker_ids {
            let tokens: Vec<u32> = vec![0, 1, 2, 3];
            let request = PreprocessedRequest::builder()
                .model("mock".to_string())
                .token_ids(tokens.clone())
                .sampling_options(SamplingOptions::default())
                .output_options(OutputOptions::default())
                .stop_conditions(StopConditions {
                    max_tokens: Some(5),
                    ..Default::default()
                })
                .build()?;
            let response_stream = pre_push_router
                .direct(Context::new(request), worker_id)
                .await?;
            // Consume the stream to complete the request
            let _responses: Vec<_> = response_stream.collect().await;
            tracing::debug!(
                "Sent request {:?} directly to worker {} to populate its local indexer",
                tokens,
                worker_id
            );
        }
        tokio::time::sleep(Duration::from_millis(1000)).await;

        // === SETUP: Build KvPushRouter ===
        let router_distributed = create_test_shared_drt_async(&shared_store_path).await;
        let router_namespace = router_distributed.namespace(&test_namespace)?;
        let backend_component = router_namespace.component(MOCKER_COMPONENT)?;
        let backend_endpoint = backend_component.endpoint("generate");
        let client = backend_endpoint.client().await?;
        let kv_router_config = KvRouterConfig::default();
        let selector = Box::new(DefaultWorkerSelector::new(Some(kv_router_config)));
        let consumer_id = format!("test-router-{}", router_distributed.connection_id());

        let kv_router: Arc<KvRouter> = Arc::new(
            KvRouter::new(
                backend_endpoint.clone(),
                client.clone(),
                BLOCK_SIZE,
                Some(selector),
                Some(kv_router_config),
                consumer_id,
            )
            .await?,
        );

        // The KvRouter now starts its subscriber asynchronously in a background task
        // that waits for runtime_configs. Poll until events appear or timeout.
        // Each request generates 2 events: input block (parent_hash: None) + output block (parent_hash: Some)
        // With 2 workers, that's 4 events total.
        let expected_events = 4;
        let max_wait = Duration::from_secs(10);
        let poll_interval = Duration::from_millis(100);
        let start = std::time::Instant::now();

        let global_kv_events = loop {
            let events = kv_router.indexer.dump_events().await?;
            tracing::debug!("Global KV events ({}): {:?}", events.len(), events);
            if events.len() >= expected_events {
                break events;
            }
            if start.elapsed() > max_wait {
                anyhow::bail!(
                    "Timed out waiting for KV events: expected {}, got {}",
                    expected_events,
                    events.len()
                );
            }
            tokio::time::sleep(poll_interval).await;
        };

        assert_eq!(global_kv_events.len(), expected_events); // 2 workers × 2 events per request (input + output)

        // === Cleanup ===
        for handle in server_handles {
            handle.abort();
        }
        distributed1.shutdown();
        distributed2.shutdown();
        router_distributed.shutdown();

        Ok(())
    }
}
