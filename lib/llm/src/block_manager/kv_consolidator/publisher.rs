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
#[derive(Debug, Serialize)]
#[serde(tag = "type")]
enum Event {
    #[serde(rename = "BlockStored")]
    BlockStored {
        block_hashes: Vec<u64>,
        parent_block_hash: Option<u64>,
        token_ids: Vec<i32>,
        block_size: i32,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        lora_name: Option<String>,
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
    /// Returns the converted Event and its dp_rank.
    /// Returns an error if block hash parsing fails to prevent sending corrupted events to the router
    fn from_consolidated(event: ConsolidatedEvent) -> Result<(Self, Option<i32>)> {
        match event {
            ConsolidatedEvent::Store {
                block_hash,
                parent_hash,
                token_ids,
                block_size,
                lora_name,
                source: _, // Source used for logging only, not sent to router
                dp_rank,
            } => {
                let parsed_hash = block_hash
                    .parse::<u64>()
                    .with_context(|| format!("Failed to parse block_hash: {}", block_hash))?;

                let parsed_parent = parent_hash
                    .map(|h| {
                        h.parse::<u64>()
                            .with_context(|| format!("Failed to parse parent_hash: {}", h))
                    })
                    .transpose()?;

                let token_ids_i32: Vec<i32> = token_ids
                    .into_iter()
                    .map(|t| {
                        i32::try_from(t).unwrap_or_else(|_| {
                            tracing::warn!("Token ID {} exceeds i32::MAX, clamping to i32::MAX", t);
                            i32::MAX
                        })
                    })
                    .collect();

                let block_size_i32 = i32::try_from(block_size).unwrap_or_else(|_| {
                    tracing::warn!(
                        "Block size {} exceeds i32::MAX, clamping to i32::MAX",
                        block_size
                    );
                    i32::MAX
                });

                Ok((Event::BlockStored {
                    block_hashes: vec![parsed_hash],
                    parent_block_hash: parsed_parent,
                    token_ids: token_ids_i32,
                    block_size: block_size_i32,
                    lora_name,
                    medium: None,
                }, dp_rank))
            }
            ConsolidatedEvent::Remove {
                block_hash,
                source: _,
                dp_rank,
            } => {
                // Parse block hash - fail if invalid to prevent corruption
                let parsed_hash = block_hash.parse::<u64>().with_context(|| {
                    format!("Failed to parse block_hash for removal: {}", block_hash)
                })?;

                Ok((
                    Event::BlockRemoved {
                        block_hashes: vec![parsed_hash],
                        medium: None, // Not provided by ConsolidatedEvent
                    },
                    dp_rank,
                ))
            }
            ConsolidatedEvent::ClearAll { dp_rank } => {
                Ok((Event::AllBlocksCleared {}, dp_rank))
            }
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

            // Convert to vLLM format, grouping by dp_rank
            // Events with different dp_ranks must be published in separate batches
            let mut events_by_rank: std::collections::HashMap<Option<i32>, Vec<Event>> =
                std::collections::HashMap::new();

            for event in events {
                match Event::from_consolidated(event) {
                    Ok((e, dp_rank)) => {
                        events_by_rank.entry(dp_rank).or_default().push(e);
                    }
                    Err(err) => {
                        tracing::error!("Failed to convert consolidated event, skipping: {}", err);
                    }
                }
            }

            // Skip publishing if all events were invalid
            if events_by_rank.is_empty() {
                tracing::warn!("All consolidated events failed validation, skipping publish");
                continue;
            }

            // Publish one batch per dp_rank
            for (dp_rank, vllm_events) in events_by_rank {
                let num_events = vllm_events.len();

                let batch = EventBatch(
                    std::time::SystemTime::now()
                        .duration_since(std::time::UNIX_EPOCH)
                        .unwrap()
                        .as_secs_f64(), // ts
                    vllm_events, // events
                    dp_rank,     // data_parallel_rank from actual events
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
                        "Consolidator: Published batch with {} event(s) to ZMQ (seq={}, dp_rank={:?})",
                        num_events,
                        seq,
                        dp_rank
                    );
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use super::super::tracker::ConsolidatedEvent;

    #[test]
    fn test_from_consolidated_store_returns_dp_rank() {
        let event = ConsolidatedEvent::Store {
            block_hash: "12345".to_string(),
            parent_hash: None,
            token_ids: vec![1, 2, 3, 4],
            block_size: 4,
            lora_id: None,
            source: "trtllm".to_string(),
            dp_rank: Some(7),
        };

        let (converted, dp_rank) = Event::from_consolidated(event).unwrap();
        assert_eq!(dp_rank, Some(7));
        assert!(matches!(converted, Event::BlockStored { .. }));
    }

    #[test]
    fn test_from_consolidated_store_returns_none_dp_rank() {
        let event = ConsolidatedEvent::Store {
            block_hash: "12345".to_string(),
            parent_hash: None,
            token_ids: vec![1, 2, 3, 4],
            block_size: 4,
            lora_id: None,
            source: "vllm".to_string(),
            dp_rank: None,
        };

        let (_, dp_rank) = Event::from_consolidated(event).unwrap();
        assert_eq!(dp_rank, None);
    }

    #[test]
    fn test_from_consolidated_remove_returns_dp_rank() {
        let event = ConsolidatedEvent::Remove {
            block_hash: "12345".to_string(),
            source: "trtllm".to_string(),
            dp_rank: Some(2),
        };

        let (converted, dp_rank) = Event::from_consolidated(event).unwrap();
        assert_eq!(dp_rank, Some(2));
        assert!(matches!(converted, Event::BlockRemoved { .. }));
    }

    #[test]
    fn test_from_consolidated_clear_all_returns_dp_rank() {
        let event = ConsolidatedEvent::ClearAll { dp_rank: Some(3) };

        let (converted, dp_rank) = Event::from_consolidated(event).unwrap();
        assert_eq!(dp_rank, Some(3));
        assert!(matches!(converted, Event::AllBlocksCleared {}));
    }

    #[test]
    fn test_from_consolidated_clear_all_returns_none_dp_rank() {
        let event = ConsolidatedEvent::ClearAll { dp_rank: None };

        let (_, dp_rank) = Event::from_consolidated(event).unwrap();
        assert_eq!(dp_rank, None);
    }

    #[test]
    fn test_events_grouped_by_dp_rank() {
        // Simulate the grouping logic used in the publisher loop
        let events = vec![
            ConsolidatedEvent::Store {
                block_hash: "100".to_string(),
                parent_hash: None,
                token_ids: vec![1, 2, 3, 4],
                block_size: 4,
                lora_id: None,
                source: "trtllm".to_string(),
                dp_rank: Some(0),
            },
            ConsolidatedEvent::Store {
                block_hash: "200".to_string(),
                parent_hash: None,
                token_ids: vec![5, 6, 7, 8],
                block_size: 4,
                lora_id: None,
                source: "trtllm".to_string(),
                dp_rank: Some(1),
            },
            ConsolidatedEvent::Store {
                block_hash: "300".to_string(),
                parent_hash: None,
                token_ids: vec![9, 10, 11, 12],
                block_size: 4,
                lora_id: None,
                source: "trtllm".to_string(),
                dp_rank: Some(0),
            },
        ];

        let mut events_by_rank: std::collections::HashMap<Option<i32>, Vec<Event>> =
            std::collections::HashMap::new();

        for event in events {
            match Event::from_consolidated(event) {
                Ok((e, dp_rank)) => {
                    events_by_rank.entry(dp_rank).or_default().push(e);
                }
                Err(err) => panic!("Unexpected error: {}", err),
            }
        }

        // Should have 2 groups: dp_rank=0 (2 events) and dp_rank=1 (1 event)
        assert_eq!(events_by_rank.len(), 2);
        assert_eq!(events_by_rank[&Some(0)].len(), 2);
        assert_eq!(events_by_rank[&Some(1)].len(), 1);
    }
}
