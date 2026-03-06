// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Simple ZMQ Subscriber for vLLM KV Events
//!
//! This is a simplified subscriber that deserializes raw vLLM/TensorRT-LLM events.

use anyhow::{Context, Result};
use rmp_serde::Deserializer;
use serde::Deserialize;
use std::sync::Arc;
use tokio::sync::RwLock;
use tokio::task::JoinHandle;
use tokio_util::sync::CancellationToken;
use zeromq::{Socket, SocketRecv, SubSocket};

use dynamo_kv_router::zmq_wire::RawKvEvent;

use super::tracker::{CacheStatusTracker, EventSource, StorageTier};

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

impl VllmEventBatch {
    fn ts(&self) -> f64 {
        self.0
    }

    fn events(&self) -> &Vec<RawKvEvent> {
        &self.1
    }

    fn data_parallel_rank(&self) -> Option<i32> {
        self.2
    }
}

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

                let dp_rank = batch.data_parallel_rank();
                tracing::debug!(
                    "Consolidator received event batch with {} events (ts={:.2}, dp_rank={:?})",
                    batch.events().len(),
                    batch.ts(),
                    dp_rank
                );

                // Process events
                let mut tracker_guard = tracker.write().await;
                for event in batch.events() {
                    process_event(&mut tracker_guard, event.clone(), dp_rank, engine_source);
                }
            }
        }
    }

    Ok(())
}

fn process_event(
    tracker: &mut CacheStatusTracker,
    event: RawKvEvent,
    data_parallel_rank: Option<i32>,
    engine_source: EventSource,
) {
    match event {
        RawKvEvent::BlockStored {
            block_hashes,
            parent_block_hash,
            token_ids,
            block_size,
            medium,
            lora_name,
            .. // block_mm_infos not used in consolidator
        } => {
            let storage_tier = medium
                .as_ref()
                .and_then(|m| StorageTier::from_vllm_medium(m))
                .unwrap_or(StorageTier::Device);

            tracing::debug!(
                "Processing BlockStored: {} blocks, tier={:?}, tokens={}, block_size={}, parent={:?}, dp_rank={:?}",
                block_hashes.len(),
                storage_tier,
                token_ids.len(),
                block_size,
                parent_block_hash,
                data_parallel_rank
            );

            // block_size is already usize; guard against 0 to avoid chunks() panic
            if block_size == 0 {
                tracing::warn!("Invalid block_size 0 (must be positive), skipping event to avoid chunks() panic");
                return;
            }

            // token_ids is already Vec<u32>; split directly into per-block chunks
            let token_chunks: Vec<Vec<u32>> = token_ids
                .chunks(block_size)
                .map(|chunk| chunk.to_vec())
                .collect();

            if token_chunks.len() != block_hashes.len() {
                tracing::warn!(
                    "Token chunks ({}) don't match block hashes ({}), skipping event",
                    token_chunks.len(),
                    block_hashes.len()
                );
                return;
            }

            // For batches, chain the blocks: each block's parent is the previous block
            let mut current_parent = parent_block_hash.map(|h| h.into_u64().to_string());

            for (i, block_hash) in block_hashes.into_iter().enumerate() {
                let block_tokens = token_chunks[i].clone();
                let block_hash_u64 = block_hash.into_u64();

                tracker.handle_store(
                    block_hash_u64.to_string(),
                    engine_source,
                    block_tokens,
                    current_parent.clone(),
                    block_size,
                    lora_name.clone(),
                    Some(storage_tier),
                    data_parallel_rank,
                );

                // Next block's parent is this block (only if hash was valid)
                current_parent = Some(block_hash_u64.to_string());
            }
        }

        RawKvEvent::BlockRemoved { block_hashes, medium } => {
            let storage_tier = medium
                .as_ref()
                .and_then(|m| StorageTier::from_vllm_medium(m))
                .unwrap_or(StorageTier::Device);

            tracing::debug!(
                "Processing BlockRemoved: {} blocks, tier={:?}",
                block_hashes.len(),
                storage_tier
            );

            for block_hash in block_hashes {
                tracker.handle_remove(&block_hash.into_u64().to_string(), engine_source);
            }
        }

        RawKvEvent::AllBlocksCleared => {
            tracing::debug!("Processing AllBlocksCleared (dp_rank={:?})", data_parallel_rank);
            tracker.handle_clear_all_with_rank(data_parallel_rank);
        }
    }
}
