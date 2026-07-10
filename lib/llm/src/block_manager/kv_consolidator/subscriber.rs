// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Simple ZMQ Subscriber for vLLM KV Events
//!
//! This is a simplified subscriber that deserializes raw vLLM/TensorRT-LLM events.

use anyhow::{Context, Result};
use futures::StreamExt;
use rmp_serde::Deserializer;
use serde::Deserialize;
use tokio::task::JoinHandle;
use tokio_util::sync::CancellationToken;

use dynamo_kv_router::protocols::StorageTier as RouterStorageTier;
use dynamo_kv_router::zmq_wire::{RawKvEvent, worker_local_event_tier};

use super::SharedCacheStatusTracker;
use super::tracker::{
    CacheStatusTracker, EventSource, RemoveEventInput, StorageTier, StoreEventInput,
};
use crate::utils::zmq::{connect_sub_socket, multipart_message};

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
    tracker: SharedCacheStatusTracker,
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
    tracker: SharedCacheStatusTracker,
    cancellation_token: CancellationToken,
    engine_source: EventSource,
) -> Result<()> {
    tracing::info!(
        "KV event consolidator ZMQ listener connecting to {}",
        endpoint
    );

    let socket = connect_sub_socket(&endpoint, None)
        .await
        .with_context(|| format!("Failed to connect to ZMQ endpoint {endpoint}"))?;
    let mut socket = socket;

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

            msg_result = socket.next() => {
                let frames = match msg_result {
                    Some(Ok(frames)) => multipart_message(frames),
                    Some(Err(error)) => {
                        tracing::error!("Error receiving ZMQ message: {error}");
                        break;
                    }
                    None => break,
                };

                // Parse multipart message: supports both formats
                // - 2 frames: [topic, payload]
                // - 3 frames: [topic, sequence, payload]
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
                    process_event(&mut **tracker_guard, event.clone(), dp_rank, engine_source);
                }
            }
        }
    }

    Ok(())
}

fn process_event(
    tracker: &mut dyn CacheStatusTracker,
    event: RawKvEvent,
    data_parallel_rank: Option<i32>,
    engine_source: EventSource,
) {
    // This source represents the engine's G1 cache. Native lower-tier events
    // are outside this consolidator's contract.
    if worker_local_event_tier(&event) != Some(RouterStorageTier::Device) {
        return;
    }

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

                tracker.handle_store(StoreEventInput {
                    block_hash: block_hash_u64.to_string(),
                    source: engine_source,
                    token_ids: block_tokens,
                    parent_hash: current_parent.clone(),
                    block_size,
                    lora_name: lora_name.clone(),
                    tier: Some(storage_tier),
                    data_parallel_rank,
                });

                // Next block's parent is this block (only if hash was valid)
                current_parent = Some(block_hash_u64.to_string());
            }
        }

        RawKvEvent::BlockRemoved {
            block_hashes,
            medium,
            ..
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
                tracker.handle_remove(RemoveEventInput {
                    block_hash: block_hash.into_u64().to_string(),
                    source: engine_source,
                    tier: Some(storage_tier),
                });
            }
        }

        RawKvEvent::AllBlocksCleared => {
            tracing::debug!("Processing AllBlocksCleared");
            tracker.handle_clear_all();
        }

        RawKvEvent::Ignored => {}
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use dynamo_kv_router::zmq_wire::{BlockHashValue, Locality};

    use super::super::tracker::{ConsolidatedEvent, PassthroughCacheStatusTracker, StorageTier};

    fn stored_event(medium: &str, locality: Option<Locality>) -> RawKvEvent {
        RawKvEvent::BlockStored {
            block_hashes: vec![BlockHashValue::Unsigned(1)],
            parent_block_hash: None,
            token_ids: vec![10, 11],
            block_size: 2,
            medium: Some(medium.to_string()),
            lora_name: None,
            cache_namespace: None,
            block_mm_infos: None,
            is_eagle: Some(false),
            group_idx: None,
            kv_cache_spec_kind: None,
            kv_cache_spec_sliding_window: None,
            locality,
        }
    }

    #[test]
    fn process_event_only_tracks_worker_local_g1_events() {
        let mut tracker = PassthroughCacheStatusTracker::new();

        process_event(
            &mut tracker,
            stored_event("GPU", Some(Locality::Remote)),
            None,
            EventSource::Vllm,
        );
        process_event(
            &mut tracker,
            stored_event("FS", Some(Locality::Local)),
            None,
            EventSource::Vllm,
        );
        process_event(
            &mut tracker,
            stored_event("FS", None),
            None,
            EventSource::Vllm,
        );
        assert!(tracker.drain_events().is_empty());

        process_event(
            &mut tracker,
            stored_event("GPU", None),
            None,
            EventSource::Vllm,
        );
        assert!(matches!(
            tracker.drain_events().as_slice(),
            [ConsolidatedEvent::Store {
                tier: Some(StorageTier::Device),
                ..
            }]
        ));
    }
}
