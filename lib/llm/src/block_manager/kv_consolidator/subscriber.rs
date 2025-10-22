// SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Simple ZMQ Subscriber for vLLM KV Events
//!
//! This is a simplified subscriber that deserializes raw vLLM events.

use anyhow::{Context, Result};
use rmp_serde::Deserializer;
use serde::Deserialize;
use std::sync::Arc;
use tokio::sync::RwLock;
use tokio::task::JoinHandle;
use tokio_util::sync::CancellationToken;
use zeromq::{Socket, SocketRecv, SubSocket};

use super::tracker::{CacheStatusTracker, StorageTier};

/// vLLM's raw event batch structure (matches msgspec encoding with array_like=True)
/// Format: [timestamp, [events], data_parallel_rank]

#[derive(Debug, Deserialize)]
struct VllmEventBatch(
    f64,               // ts
    Vec<VllmRawEvent>, // events
    Option<i32>,       // data_parallel_rank
);

impl VllmEventBatch {
    fn ts(&self) -> f64 {
        self.0
    }

    fn events(&self) -> &Vec<VllmRawEvent> {
        &self.1
    }

    fn data_parallel_rank(&self) -> Option<i32> {
        self.2
    }
}

/// Block hash can be either an integer or a string (bytes hex-encoded)
#[derive(Debug, Clone, Deserialize)]
#[serde(untagged)]
enum BlockHash {
    Int(u64),
    Str(String),
}

impl std::fmt::Display for BlockHash {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            BlockHash::Int(n) => write!(f, "{}", n),
            BlockHash::Str(s) => write!(f, "{}", s),
        }
    }
}

/// Raw vLLM event format (preserves all data including token_ids)
#[derive(Debug, Clone, Deserialize)]
#[serde(tag = "type")]
enum VllmRawEvent {
    #[serde(rename = "BlockStored")]
    BlockStored {
        block_hashes: Vec<BlockHash>,
        parent_block_hash: Option<BlockHash>,
        token_ids: Vec<i32>,
        block_size: i32,
        lora_id: Option<i32>,
        #[serde(default)]
        medium: Option<String>,
    },
    #[serde(rename = "BlockRemoved")]
    BlockRemoved {
        block_hashes: Vec<BlockHash>,
        #[serde(default)]
        medium: Option<String>,
    },
    #[serde(rename = "AllBlocksCleared")]
    AllBlocksCleared {},
}

/// Start ZMQ listener and process events into tracker
pub async fn start_simple_zmq_listener(
    endpoint: String,
    tracker: Arc<RwLock<CacheStatusTracker>>,
    cancellation_token: CancellationToken,
) -> Result<JoinHandle<()>> {
    let handle = tokio::spawn(async move {
        if let Err(e) = run_listener_loop(endpoint, tracker, cancellation_token).await {
            tracing::error!("ZMQ listener task failed: {}", e);
        }
    });

    Ok(handle)
}

async fn run_listener_loop(
    endpoint: String,
    tracker: Arc<RwLock<CacheStatusTracker>>,
    cancellation_token: CancellationToken,
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
                    process_event(&mut tracker_guard, event.clone(), dp_rank);
                }
            }
        }
    }

    Ok(())
}

fn process_event(
    tracker: &mut CacheStatusTracker,
    event: VllmRawEvent,
    data_parallel_rank: Option<i32>,
) {
    match event {
        VllmRawEvent::BlockStored {
            block_hashes,
            parent_block_hash,
            token_ids,
            block_size,
            lora_id,
            medium,
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

            // Convert block_size from i32 to usize for chunking
            // SAFETY: Must validate block_size > 0 to prevent panic in chunks()
            let block_size_usize = match usize::try_from(block_size) {
                Ok(size) if size > 0 => size,
                _ => {
                    tracing::warn!(
                        "Invalid block_size {} (must be positive), skipping event to avoid chunks() panic",
                        block_size
                    );
                    return;
                }
            };

            // Convert token_ids from i32 to u32 and split into chunks
            let token_ids_u32: Vec<u32> = token_ids
                .into_iter()
                .filter_map(|t| {
                    u32::try_from(t).ok().or_else(|| {
                        tracing::warn!("Invalid token ID {}, skipping", t);
                        None
                    })
                })
                .collect();

            let token_chunks: Vec<Vec<u32>> = token_ids_u32
                .chunks(block_size_usize)
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

            // Process each block with its corresponding token chunk
            // For batches, chain the blocks: each block's parent is the previous block in the batch
            let mut current_parent = parent_block_hash.as_ref().map(|h| h.to_string());

            for (i, block_hash) in block_hashes.iter().enumerate() {
                let block_tokens = token_chunks[i].clone();

                tracker.handle_store(
                    block_hash.to_string(),
                    crate::block_manager::kv_consolidator::EventSource::Vllm,
                    block_tokens,
                    current_parent.clone(),
                    block_size_usize,
                    lora_id,
                    Some(storage_tier),
                    data_parallel_rank,
                );

                // Next block's parent is this block
                current_parent = Some(block_hash.to_string());
            }
        }

        VllmRawEvent::BlockRemoved {
            block_hashes,
            medium,
        } => {
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
                tracker.handle_remove(
                    &block_hash.to_string(),
                    crate::block_manager::kv_consolidator::EventSource::Vllm,
                );
            }
        }

        VllmRawEvent::AllBlocksCleared {} => {
            tracing::debug!("Processing AllBlocksCleared");
            tracker.handle_clear_all();
        }
    }
}
