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
use tokio::task::JoinHandle;

use super::SharedCacheStatusTracker;
use super::tracker::ConsolidatedEvent;
use crate::utils::zmq::{bind_pub_socket, send_multipart};

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
    /// Returns an error if block hash parsing fails to prevent sending corrupted events to the router
    fn from_consolidated(event: ConsolidatedEvent) -> Result<Self> {
        match event {
            ConsolidatedEvent::Store {
                block_hash,
                parent_hash,
                token_ids,
                block_size,
                lora_name,
                source: _,
                tier,
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

                Ok(Event::BlockStored {
                    block_hashes: vec![parsed_hash],
                    parent_block_hash: parsed_parent,
                    token_ids: token_ids_i32,
                    block_size: block_size_i32,
                    lora_name,
                    medium: tier.map(|t| t.to_vllm_medium().to_string()),
                })
            }
            ConsolidatedEvent::Remove {
                block_hash,
                source: _,
                tier,
            } => {
                // Parse block hash - fail if invalid to prevent corruption
                let parsed_hash = block_hash.parse::<u64>().with_context(|| {
                    format!("Failed to parse block_hash for removal: {}", block_hash)
                })?;

                Ok(Event::BlockRemoved {
                    block_hashes: vec![parsed_hash],
                    medium: tier.map(|t| t.to_vllm_medium().to_string()),
                })
            }
            ConsolidatedEvent::ClearAll => Ok(Event::AllBlocksCleared {}),
        }
    }
}

/// ZMQ Publisher for consolidated events
pub struct KvEventConsolidatorPublisher {
    endpoint: String,
    tracker: SharedCacheStatusTracker,
    sequence: Arc<AtomicU64>,
    task_handle: Option<JoinHandle<()>>,
}

impl KvEventConsolidatorPublisher {
    /// Create a new publisher
    pub fn new(endpoint: &str, tracker: SharedCacheStatusTracker) -> Result<Self> {
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
        tracker: SharedCacheStatusTracker,
        sequence: Arc<AtomicU64>,
    ) -> Result<()> {
        tracing::info!("Starting consolidated event publisher on {}", endpoint);

        // Create ZMQ PUB socket and bind
        let socket = bind_pub_socket(&endpoint)
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

            // Serialize to msgpack.
            // Keep the outer batch as a tuple [ts, events, dp_rank], but force
            // inner struct-like events to use named fields so RawKvEvent's map
            // deserializer can safely decode optional fields like `medium`.
            let mut payload = Vec::new();
            batch
                .serialize(&mut Serializer::new(&mut payload).with_struct_map())
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
            let frames = frames.into_iter().map(|frame| frame.to_vec()).collect();

            if let Err(e) = send_multipart(&socket, frames).await {
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

#[cfg(test)]
mod tests {
    use super::*;
    use dynamo_kv_router::zmq_wire::{KvEventBatch, RawKvEvent};
    use rmp_serde as rmps;

    #[test]
    fn test_block_stored_with_medium_and_no_lora_name_decodes() {
        let batch = EventBatch(
            0.0,
            vec![Event::BlockStored {
                block_hashes: vec![42],
                parent_block_hash: Some(7),
                token_ids: vec![1, 2, 3, 4],
                block_size: 4,
                lora_name: None,
                medium: Some("CPU_TIER1".to_string()),
            }],
            Some(0),
        );

        let mut payload = Vec::new();
        batch
            .serialize(&mut Serializer::new(&mut payload).with_struct_map())
            .unwrap();

        let decoded: KvEventBatch = rmps::from_slice(&payload).unwrap();
        assert_eq!(decoded.events.len(), 1);
        assert_eq!(decoded.data_parallel_rank, Some(0));

        let RawKvEvent::BlockStored {
            block_hashes,
            parent_block_hash,
            token_ids,
            block_size,
            medium,
            lora_name,
            ..
        } = &decoded.events[0]
        else {
            panic!("expected BlockStored");
        };

        assert_eq!(block_hashes.len(), 1);
        assert!(parent_block_hash.is_some());
        assert_eq!(token_ids, &vec![1, 2, 3, 4]);
        assert_eq!(*block_size, 4);
        assert_eq!(medium.as_deref(), Some("CPU_TIER1"));
        assert_eq!(lora_name.as_deref(), None);
    }
}
