// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Publishers for KV Events Consolidator
//!
//! Publishes consolidated KV cache events to the router:
//! - ZMQ Publisher: For vLLM (publishes in vLLM format)
//! - NATS Publisher: For TensorRT-LLM (publishes RouterEvent format)

use anyhow::{Context, Result};
use bytes::Bytes;
use dynamo_runtime::{
    traits::events::EventPublisher,
    transports::nats::{NatsQueue, QUEUE_NAME, Slug},
};
use rmp_serde::Serializer;
use serde::Serialize;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use tokio::sync::RwLock;
use tokio::task::JoinHandle;
use tokio_util::sync::CancellationToken;
use zeromq::{PubSocket, Socket, SocketSend};

use super::tracker::{CacheStatusTracker, ConsolidatedEvent};
use crate::kv_router::KV_EVENT_SUBJECT;
use crate::kv_router::indexer::RouterEvent;
use crate::kv_router::protocols::{
    ExternalSequenceBlockHash, KvCacheEvent, KvCacheEventData, KvCacheRemoveData, KvCacheStoreData,
    KvCacheStoredBlockData, LocalBlockHash,
};

/// Event batch structure matching vLLM's format (array_like=True)
/// Format: [timestamp, [events], data_parallel_rank]
///
/// Note: This uses a tuple struct to serialize as an array [ts, events, rank]
/// rather than an object {"ts": ..., "events": ..., "rank": ...} for vLLM compatibility.
#[derive(Debug, Serialize)]
struct EventBatch(
    f64,         // ts
    Vec<Event>,  // events
    Option<i32>, // data_parallel_rank
);

/// Event types matching vLLM's format
/// Note: block_hashes are u64 to match vLLM's ExternalBlockHash type
#[derive(Debug, Serialize)]
#[serde(tag = "type")]
enum Event {
    #[serde(rename = "BlockStored")]
    BlockStored {
        block_hashes: Vec<u64>,
        parent_block_hash: Option<u64>,
        token_ids: Vec<i32>,
        block_size: i32,
        lora_id: Option<i32>,
    },
    #[serde(rename = "BlockRemoved")]
    BlockRemoved { block_hashes: Vec<u64> },
    #[serde(rename = "AllBlocksCleared")]
    AllBlocksCleared {},
}

impl Event {
    /// Convert from ConsolidatedEvent to vLLM Event format
    /// Parses string block hashes back to u64 for router compatibility
    /// Note: source field is kept in ConsolidatedEvent for internal logging but not sent to router
    ///
    /// Returns an error if block hash parsing fails to prevent sending corrupted events to the router
    fn from_consolidated(event: ConsolidatedEvent) -> Result<Self> {
        match event {
            ConsolidatedEvent::Store {
                block_hash,
                parent_hash,
                token_ids,
                block_size,
                lora_id,
                source: _, // Source used for logging only, not sent to router
            } => {
                // Parse block hash - fail if invalid to prevent corruption
                let parsed_hash = block_hash
                    .parse::<u64>()
                    .with_context(|| format!("Failed to parse block_hash: {}", block_hash))?;

                // Parse parent hash if present - fail if invalid
                let parsed_parent = parent_hash
                    .map(|h| {
                        h.parse::<u64>()
                            .with_context(|| format!("Failed to parse parent_hash: {}", h))
                    })
                    .transpose()?;

                // Convert u32 token_ids to i32 for vLLM compatibility
                // Token IDs should never exceed i32::MAX in practice, but we handle it gracefully
                let token_ids_i32: Vec<i32> = token_ids
                    .into_iter()
                    .map(|t| {
                        i32::try_from(t).unwrap_or_else(|_| {
                            tracing::warn!("Token ID {} exceeds i32::MAX, clamping to i32::MAX", t);
                            i32::MAX
                        })
                    })
                    .collect();

                // Convert usize block_size to i32 for vLLM compatibility
                let block_size_i32 = i32::try_from(block_size).unwrap_or_else(|_| {
                    tracing::warn!(
                        "Block size {} exceeds i32::MAX, clamping to i32::MAX",
                        block_size
                    );
                    i32::MAX
                });

                Ok(Event::BlockStored {
                    block_hashes: vec![parsed_hash],
                    parent_block_hash: parsed_parent,
                    token_ids: token_ids_i32,
                    block_size: block_size_i32,
                    lora_id,
                })
            }
            ConsolidatedEvent::Remove {
                block_hash,
                source: _,
            } => {
                // Parse block hash - fail if invalid to prevent corruption
                let parsed_hash = block_hash.parse::<u64>().with_context(|| {
                    format!("Failed to parse block_hash for removal: {}", block_hash)
                })?;

                Ok(Event::BlockRemoved {
                    block_hashes: vec![parsed_hash],
                })
            }
            ConsolidatedEvent::ClearAll => Ok(Event::AllBlocksCleared {}),
        }
    }
}

/// ZMQ Publisher for consolidated events
pub struct KvEventConsolidatorPublisher {
    endpoint: String,
    tracker: Arc<RwLock<CacheStatusTracker>>,
    sequence: Arc<AtomicU64>,
    task_handle: Option<JoinHandle<()>>,
}

impl KvEventConsolidatorPublisher {
    /// Create a new publisher
    pub fn new(endpoint: &str, tracker: Arc<RwLock<CacheStatusTracker>>) -> Result<Self> {
        let endpoint = endpoint.to_string();
        let sequence = Arc::new(AtomicU64::new(0));

        let publisher = Self {
            endpoint: endpoint.clone(),
            tracker: tracker.clone(),
            sequence: sequence.clone(),
            task_handle: None,
        };

        // Start the publisher task
        let handle = tokio::spawn(async move {
            if let Err(e) = Self::run_publisher_loop(endpoint, tracker, sequence).await {
                // Bind failures and other critical errors should crash the process
                panic!("Publisher task failed: {}", e);
            }
        });

        Ok(Self {
            endpoint: publisher.endpoint,
            tracker: publisher.tracker,
            sequence: publisher.sequence,
            task_handle: Some(handle),
        })
    }

    /// Stop the publisher task
    pub async fn shutdown(self) -> Result<()> {
        if let Some(handle) = self.task_handle {
            handle.abort();
            let _ = handle.await;
        }
        Ok(())
    }

    /// Main publisher loop
    async fn run_publisher_loop(
        endpoint: String,
        tracker: Arc<RwLock<CacheStatusTracker>>,
        sequence: Arc<AtomicU64>,
    ) -> Result<()> {
        tracing::info!("Starting consolidated event publisher on {}", endpoint);

        // Create ZMQ PUB socket and bind
        let mut socket = PubSocket::new();
        socket
            .bind(&endpoint)
            .await
            .with_context(|| format!("Failed to bind publisher to {}", endpoint))?;

        tracing::info!("Publisher bound to {}", endpoint);

        // Publish loop - check for events every 50ms
        let mut interval = tokio::time::interval(tokio::time::Duration::from_millis(50));

        loop {
            interval.tick().await;

            // Drain events from tracker
            let events = {
                let mut tracker_guard = tracker.write().await;
                tracker_guard.drain_events()
            };

            if events.is_empty() {
                continue;
            }

            tracing::debug!(
                "Publishing {} consolidated event(s) to router",
                events.len()
            );

            // Convert to vLLM format, filtering out events with invalid hashes
            let vllm_events: Vec<Event> = events
                .into_iter()
                .filter_map(|event| match Event::from_consolidated(event) {
                    Ok(e) => Some(e),
                    Err(err) => {
                        tracing::error!("Failed to convert consolidated event, skipping: {}", err);
                        None
                    }
                })
                .collect();

            // Skip publishing if all events were invalid
            if vllm_events.is_empty() {
                tracing::warn!("All consolidated events failed validation, skipping publish");
                continue;
            }

            let num_events = vllm_events.len(); // Save length before move

            let batch = EventBatch(
                std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap()
                    .as_secs_f64(), // ts
                vllm_events, // events
                Some(0),     // data_parallel_rank (default)
            );

            // Serialize to msgpack
            let mut payload = Vec::new();
            batch
                .serialize(&mut Serializer::new(&mut payload))
                .context("Failed to serialize event batch")?;

            // Get sequence number
            let seq = sequence.fetch_add(1, Ordering::SeqCst);
            let seq_bytes = seq.to_be_bytes();

            // Send multipart message: [topic, sequence, payload]
            // Empty topic means all subscribers receive it
            let frames = vec![
                Bytes::from(""),
                Bytes::from(seq_bytes.to_vec()),
                Bytes::from(payload),
            ];

            let msg = match zeromq::ZmqMessage::try_from(frames) {
                Ok(m) => m,
                Err(e) => {
                    tracing::error!("Failed to create multipart ZMQ message: {:?}", e);
                    continue;
                }
            };

            if let Err(e) = socket.send(msg).await {
                tracing::error!("Failed to send consolidated events: {}", e);
            } else {
                tracing::debug!(
                    "Published batch with {} event(s) to router (seq={})",
                    num_events,
                    seq
                );
            }
        }
    }
}

/// NATS Publisher for consolidated events (TensorRT-LLM)
pub struct KvEventConsolidatorPublisherNats {
    #[allow(dead_code)]
    tracker: Arc<RwLock<CacheStatusTracker>>,
    #[allow(dead_code)]
    sequence: Arc<AtomicU64>,
    task_handle: Option<JoinHandle<()>>,
    cancellation_token: CancellationToken,
}

impl KvEventConsolidatorPublisherNats {
    /// Create a new NATS publisher
    ///
    /// Uses a hardcoded stream name for TensorRT-LLM: namespace.dynamo.component.tensorrt_llm.kv_events
    /// This matches what the router subscribes to.
    pub fn new(tracker: Arc<RwLock<CacheStatusTracker>>) -> Result<Self> {
        let sequence = Arc::new(AtomicU64::new(0));
        let cancellation_token = CancellationToken::new();

        // Get runtime handle from current context
        // For BlockManager context, component is not available, so we use the current runtime handle
        let runtime_handle = tokio::runtime::Handle::try_current()
            .map_err(|_| anyhow::anyhow!("No tokio runtime available for NATS publisher"))?;

        let tracker_clone = tracker.clone();
        let sequence_clone = sequence.clone();
        let cancellation_token_clone = cancellation_token.clone();

        // Start the publisher task
        let handle = runtime_handle.spawn(async move {
            tracing::info!("KV Event Consolidator: NATS publisher task starting...");
            match Self::run_publisher_loop(tracker_clone, sequence_clone, cancellation_token_clone)
                .await
            {
                Ok(()) => {
                    tracing::debug!(
                        "KV Event Consolidator: NATS publisher task completed normally"
                    );
                }
                Err(e) => {
                    tracing::error!("KV Event Consolidator: NATS publisher task failed: {}", e);
                    tracing::error!(
                        "KV Event Consolidator: NATS publisher error details: {:?}",
                        e
                    );
                }
            }
        });

        Ok(Self {
            tracker,
            sequence,
            task_handle: Some(handle),
            cancellation_token,
        })
    }

    /// Stop the publisher task
    pub async fn shutdown(self) -> Result<()> {
        self.cancellation_token.cancel();
        if let Some(handle) = self.task_handle {
            let _ = handle.await;
        }
        Ok(())
    }

    /// Main publisher loop
    async fn run_publisher_loop(
        tracker: Arc<RwLock<CacheStatusTracker>>,
        _sequence: Arc<AtomicU64>,
        cancellation_token: CancellationToken,
    ) -> Result<()> {
        // Create NATS queue for publishing
        // The router subscribes to: namespace.dynamo.component.tensorrt_llm.kv_events
        let stream_name = Slug::slugify(&format!(
            "namespace.dynamo.component.tensorrt_llm.{}",
            KV_EVENT_SUBJECT
        ))
        .to_string()
        .replace("_", "-");

        tracing::info!(
            "KV Event Consolidator: Starting NATS publisher loop, will publish to stream: {}",
            stream_name
        );

        let nats_server =
            std::env::var(dynamo_runtime::config::environment_names::nats::NATS_SERVER)
                .unwrap_or_else(|_| "nats://localhost:4222".to_string());

        tracing::info!(
            "KV Event Consolidator: Connecting to NATS server: {}",
            nats_server
        );

        let mut nats_queue = NatsQueue::new_without_consumer(
            stream_name.clone(),
            nats_server.clone(),
            std::time::Duration::from_secs(60),
        );

        // Connect to NATS
        nats_queue
            .connect()
            .await
            .context("Failed to connect to NATS")?;

        tracing::info!(
            "KV Event Consolidator: Successfully connected to NATS, publishing to stream: {}",
            stream_name
        );

        // Use worker_id 0 for consolidator (consolidated events don't have a specific worker)
        let worker_id = 0u64;

        // Publish loop - check for events every 50ms
        let mut interval = tokio::time::interval(tokio::time::Duration::from_millis(50));

        loop {
            tokio::select! {
                _ = cancellation_token.cancelled() => {
                    break;
                }
                _ = interval.tick() => {
                    // Drain events from tracker
                    let events = {
                        let mut tracker_guard = tracker.write().await;
                        tracker_guard.drain_events()
                    };

                    if events.is_empty() {
                        continue;
                    }

                    tracing::debug!(
                        "KV Event Consolidator: Drained {} event(s) from tracker, preparing to publish",
                        events.len()
                    );

                    // Convert to RouterEvent format and publish
                    let mut published_count = 0;
                    for consolidated_event in events {
                        match Self::consolidated_to_router_event(consolidated_event, worker_id) {
                            Ok(router_event) => {
                                match nats_queue.publish(QUEUE_NAME, &router_event).await {
                                    Ok(_) => {
                                        published_count += 1;
                                    }
                                    Err(e) => {
                                        tracing::error!("Failed to publish consolidated event to NATS: {}", e);
                                    }
                                }
                            }
                            Err(err) => {
                                tracing::error!(
                                    "Failed to convert consolidated event, skipping: {}",
                                    err
                                );
                            }
                        }
                    }

                    // Log published events (matches test pattern: "Publish(?:ing|ed).*event.*to router")
                    if published_count > 0 {
                        tracing::debug!(
                            "Published {} consolidated event(s) to router",
                            published_count
                        );
                    }
                }
            }
        }

        Ok(())
    }

    /// Convert ConsolidatedEvent to RouterEvent
    fn consolidated_to_router_event(
        event: ConsolidatedEvent,
        worker_id: u64,
    ) -> Result<RouterEvent> {
        let kv_cache_event = match event {
            ConsolidatedEvent::Store {
                block_hash,
                parent_hash,
                token_ids,
                block_size: _,
                lora_id: _,
                source: _,
            } => {
                // Parse block hash
                let parsed_hash = block_hash
                    .parse::<u64>()
                    .with_context(|| format!("Failed to parse block_hash: {}", block_hash))?;

                // Parse parent hash if present
                let parsed_parent = parent_hash
                    .map(|h| {
                        h.parse::<u64>()
                            .with_context(|| format!("Failed to parse parent_hash: {}", h))
                    })
                    .transpose()?;

                // Compute tokens_hash from token_ids (LocalBlockHash)
                let tokens_hash = Self::compute_tokens_hash(&token_ids);

                // Create stored block data
                let stored_block = KvCacheStoredBlockData {
                    block_hash: ExternalSequenceBlockHash(parsed_hash),
                    tokens_hash: LocalBlockHash(tokens_hash),
                };

                KvCacheEvent {
                    event_id: 0, // Event ID not used for consolidated events
                    data: KvCacheEventData::Stored(KvCacheStoreData {
                        parent_hash: parsed_parent.map(ExternalSequenceBlockHash),
                        blocks: vec![stored_block],
                    }),
                    dp_rank: 0,
                }
            }
            ConsolidatedEvent::Remove {
                block_hash,
                source: _,
            } => {
                // Parse block hash
                let parsed_hash = block_hash.parse::<u64>().with_context(|| {
                    format!("Failed to parse block_hash for removal: {}", block_hash)
                })?;

                KvCacheEvent {
                    event_id: 0,
                    data: KvCacheEventData::Removed(KvCacheRemoveData {
                        block_hashes: vec![ExternalSequenceBlockHash(parsed_hash)],
                    }),
                    dp_rank: 0,
                }
            }
            ConsolidatedEvent::ClearAll => KvCacheEvent {
                event_id: 0,
                data: KvCacheEventData::Cleared,
                dp_rank: 0,
            },
        };

        let router_event = RouterEvent::new(worker_id, kv_cache_event);
        Ok(router_event)
    }

    /// Compute tokens hash from token IDs (matching router's computation)
    fn compute_tokens_hash(token_ids: &[u32]) -> u64 {
        use xxhash_rust::xxh3;
        const XXH3_SEED: u64 = 1337; // Must match router's seed

        let bytes: Vec<u8> = token_ids
            .iter()
            .flat_map(|&num| num.to_le_bytes())
            .collect();
        xxh3::xxh3_64_with_seed(&bytes, XXH3_SEED)
    }
}
