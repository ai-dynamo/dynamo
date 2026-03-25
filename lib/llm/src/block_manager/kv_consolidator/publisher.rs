// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Publisher for the consolidated KV event stream.
//!
//! The consolidator republishes raw KV events after tier normalization. It does
//! not rewrite hashes or deduplicate across sources.

use anyhow::{Context, Result};
use bytes::Bytes;
use rmp_serde::Serializer;
use serde::Serialize;
use std::convert::TryFrom;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use tokio::sync::RwLock;
use tokio::task::JoinHandle;
use zeromq::{PubSocket, Socket, SocketSend, ZmqMessage};

use dynamo_kv_router::zmq_wire::RawKvEvent;

use super::tracker::{CacheStatusTracker, ConsolidatedEventBatch};

/// Event batch structure matching the engine ZMQ format:
/// `[timestamp, [events], data_parallel_rank]`.
#[derive(Debug, Serialize)]
struct EventBatch(f64, Vec<RawKvEvent>, Option<i32>);

/// ZMQ publisher for consolidated events.
pub struct KvEventConsolidatorPublisher {
    endpoint: String,
    tracker: Arc<RwLock<CacheStatusTracker>>,
    sequence: Arc<AtomicU64>,
    task_handle: Option<JoinHandle<()>>,
}

impl KvEventConsolidatorPublisher {
    /// Create a new publisher.
    pub fn new(endpoint: &str, tracker: Arc<RwLock<CacheStatusTracker>>) -> Result<Self> {
        let endpoint = endpoint.to_string();
        let sequence = Arc::new(AtomicU64::new(0));

        let publisher = Self {
            endpoint: endpoint.clone(),
            tracker: tracker.clone(),
            sequence: sequence.clone(),
            task_handle: None,
        };

        let handle = tokio::spawn(async move {
            if let Err(e) = Self::run_publisher_loop(endpoint, tracker, sequence).await {
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

    /// Stop the publisher task.
    pub async fn shutdown(self) -> Result<()> {
        if let Some(handle) = self.task_handle {
            handle.abort();
            let _ = handle.await;
        }
        Ok(())
    }

    /// Main publisher loop.
    async fn run_publisher_loop(
        endpoint: String,
        tracker: Arc<RwLock<CacheStatusTracker>>,
        sequence: Arc<AtomicU64>,
    ) -> Result<()> {
        tracing::info!("Starting consolidated event publisher on {}", endpoint);

        let mut socket = PubSocket::new();
        socket
            .bind(&endpoint)
            .await
            .with_context(|| format!("Failed to bind publisher to {}", endpoint))?;

        tracing::info!("Publisher bound to {}", endpoint);

        let mut interval = tokio::time::interval(tokio::time::Duration::from_millis(50));

        loop {
            interval.tick().await;

            let batches = {
                let mut tracker_guard = tracker.write().await;
                tracker_guard.drain_events()
            };

            if batches.is_empty() {
                continue;
            }

            tracing::debug!(
                "Publishing {} consolidated batch(es) to downstream publisher",
                batches.len()
            );

            for batch in batches {
                Self::publish_batch(&mut socket, &sequence, batch).await?;
            }
        }
    }

    async fn publish_batch(
        socket: &mut PubSocket,
        sequence: &AtomicU64,
        batch: ConsolidatedEventBatch,
    ) -> Result<()> {
        let num_events = batch.events.len();
        let payload_batch = EventBatch(
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_secs_f64(),
            batch.events,
            batch.data_parallel_rank,
        );

        let mut payload = Vec::new();
        payload_batch
            .serialize(&mut Serializer::new(&mut payload))
            .context("Failed to serialize consolidated event batch")?;

        let seq = sequence.fetch_add(1, Ordering::SeqCst);
        let seq_bytes = seq.to_be_bytes();
        let frames = vec![
            Bytes::from(""),
            Bytes::from(seq_bytes.to_vec()),
            Bytes::from(payload),
        ];

        let message = ZmqMessage::try_from(frames)
            .map_err(|_| anyhow::anyhow!("Failed to build multipart ZMQ message"))?;

        socket
            .send(message)
            .await
            .context("Failed to send consolidated ZMQ event batch")?;

        tracing::debug!(
            "Published consolidated batch seq={} containing {} event(s)",
            seq,
            num_events
        );

        Ok(())
    }
}
