// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Publishers for KV Events Consolidator
//!
//! Publishes consolidated KV cache events to ZMQ (in vLLM format).
//! Worker-side publishers subscribe to this ZMQ stream and add worker_id before publishing to NATS.

use anyhow::{Context, Result};
use bytes::Bytes;
use rmp_serde::Serializer;
use serde::Serialize;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use tokio::sync::RwLock;
use tokio::task::JoinHandle;
use zeromq::{PubSocket, Socket, SocketSend};

use super::tracker::{CacheStatusTracker, ConsolidatedEvent};

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
/// Note: Uses i32 for token_ids, block_size, and lora_id to match vLLM's ZmqEventPublisher
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
        #[serde(default, skip_serializing_if = "Option::is_none")]
        medium: Option<String>,
    },
    #[serde(rename = "BlockRemoved")]
    BlockRemoved {
        block_hashes: Vec<u64>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        medium: Option<String>,
    },
    #[serde(rename = "AllBlocksCleared")]
    AllBlocksCleared {},
}

impl Event {
    /// Convert from ConsolidatedEvent to router Event format
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

                // lora_id is already Option<i32> in ConsolidatedEvent::Store
                let lora_id_i32 = lora_id;

                Ok(Event::BlockStored {
                    block_hashes: vec![parsed_hash],
                    parent_block_hash: parsed_parent,
                    token_ids: token_ids_i32,
                    block_size: block_size_i32,
                    lora_id: lora_id_i32,
                    medium: None, // Not provided by ConsolidatedEvent
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
                    medium: None, // Not provided by ConsolidatedEvent
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
                    "Consolidator: Published batch with {} event(s) to ZMQ (seq={})",
                    num_events,
                    seq
                );
            }
        }
    }
}
