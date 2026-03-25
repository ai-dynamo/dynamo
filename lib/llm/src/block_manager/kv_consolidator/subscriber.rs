// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Simple ZMQ subscriber for engine KV events.
//!
//! The consolidator preserves raw engine events and only normalizes tier metadata.

use anyhow::{Context, Result};
use rmp_serde::Deserializer;
use serde::Deserialize;
use std::sync::Arc;
use tokio::sync::RwLock;
use tokio::task::JoinHandle;
use tokio_util::sync::CancellationToken;
use zeromq::{Socket, SocketRecv, SubSocket};

use dynamo_kv_router::zmq_wire::RawKvEvent;

use super::tracker::{CacheStatusTracker, EventSource};

/// Event batch received from vLLM/TensorRT-LLM (array format)
/// Format: [timestamp, [events], data_parallel_rank]
///
/// Note: This uses a tuple struct to deserialize from array [ts, events, rank]
/// rather than an object {"ts": ..., "events": ..., "rank": ...} for vLLM/TensorRT-LLM compatibility.
#[derive(Debug, Deserialize)]
struct VllmEventBatch(
    f64,             // ts
    Vec<RawKvEvent>, // events — reuses the same custom deserializer as the router publisher
    Option<i32>,     // data_parallel_rank
);

/// Start ZMQ listener and process events into tracker
pub async fn start_simple_zmq_listener(
    endpoint: String,
    tracker: Arc<RwLock<CacheStatusTracker>>,
    cancellation_token: CancellationToken,
    engine_source: EventSource,
) -> Result<JoinHandle<()>> {
    let handle = tokio::spawn(async move {
        if let Err(e) =
            run_listener_loop(endpoint, tracker, cancellation_token, engine_source).await
        {
            tracing::error!("ZMQ listener task failed: {}", e);
        }
    });

    Ok(handle)
}

async fn run_listener_loop(
    endpoint: String,
    tracker: Arc<RwLock<CacheStatusTracker>>,
    cancellation_token: CancellationToken,
    engine_source: EventSource,
) -> Result<()> {
    tracing::info!(
        "KV event consolidator ZMQ listener connecting to {}",
        endpoint
    );

    let mut socket = SubSocket::new();
    socket
        .connect(&endpoint)
        .await
        .context("Failed to connect to ZMQ endpoint")?;
    socket
        .subscribe("")
        .await
        .context("Failed to subscribe to ZMQ topics")?;

    tracing::info!(
        "KV event consolidator ZMQ listener successfully connected to {}",
        endpoint
    );

    loop {
        tokio::select! {
            biased;

            _ = cancellation_token.cancelled() => {
                tracing::debug!("ZMQ listener received cancellation signal");
                break;
            }

            msg_result = socket.recv() => {
                let Ok(msg) = msg_result else {
                    tracing::warn!("Error receiving ZMQ message: {:?}", msg_result.unwrap_err());
                    tokio::time::sleep(tokio::time::Duration::from_millis(100)).await;
                    continue;
                };

                // Parse multipart message: supports both formats
                // - 2 frames: [topic, payload]
                // - 3 frames: [topic, sequence, payload]
                let frames: Vec<Vec<u8>> = msg.into_vec().into_iter().map(|f| f.to_vec()).collect();

                let payload = match frames.len() {
                    2 => &frames[1],  // [topic, payload]
                    3 => &frames[2],  // [topic, sequence, payload]
                    _ => {
                        tracing::warn!("Unexpected frame count: {} (expected 2 or 3)", frames.len());
                        continue;
                    }
                };

                // Deserialize event batch
                let mut deserializer = Deserializer::new(&payload[..]);
                let batch: VllmEventBatch = match Deserialize::deserialize(&mut deserializer) {
                    Ok(b) => b,
                    Err(e) => {
                        tracing::warn!("Failed to deserialize event batch: {}", e);
                        continue;
                    }
                };

                let VllmEventBatch(ts, events, dp_rank) = batch;
                tracing::debug!(
                    "Consolidator received event batch with {} events (ts={:.2}, dp_rank={:?})",
                    events.len(),
                    ts,
                    dp_rank
                );

                let mut tracker_guard = tracker.write().await;
                tracker_guard.handle_batch(events, dp_rank, engine_source);
            }
        }
    }

    Ok(())
}
