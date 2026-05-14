// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! ZMQ SUB ingress for vLLM / TRT-LLM event streams.
//!
//! Wire:
//!   frame layout: `[topic, payload]` (2-frame) or `[topic, sequence, payload]` (3-frame)
//!   payload     : msgpack-encoded [`crate::wire::vllm_in::KvEventBatch`]
//!
//! Behavior:
//!   - malformed frame counts / bad msgpack → `WARN`, loop survives
//!   - `BlockStored` → chunks `token_ids` by `block_size`, chains parents left-to-right
//!   - `BlockRemoved` → per-hash remove
//!   - `AllBlocksCleared` → clear_all
//!
//! Wave1-C implements this body.

use std::sync::Arc;

use anyhow::Result;
use futures::StreamExt;
use tokio::sync::RwLock;
use tokio::task::JoinHandle;
use tokio_util::sync::CancellationToken;

use crate::source::EventSource;
use crate::tracker::Tracker;
use crate::wire::vllm_in::{KvEventBatch, RawKvEvent};
use crate::zmq_util::{connect_sub_socket, multipart_message};

/// Spawn the ZMQ listener. Returns immediately with a [`JoinHandle`] for the task.
pub async fn spawn(
    endpoint: String,
    tracker: Arc<RwLock<Tracker>>,
    engine_source: EventSource,
    cancel: CancellationToken,
) -> Result<JoinHandle<()>> {
    let mut socket = connect_sub_socket(&endpoint, None).await?;

    let handle = tokio::spawn(async move {
        loop {
            tokio::select! {
                biased;

                _ = cancel.cancelled() => break,

                msg = socket.next() => {
                    let frames = match msg {
                        Some(Ok(f)) => multipart_message(f),
                        Some(Err(e)) => {
                            tracing::error!("ZMQ listener task failed: {e}");
                            break;
                        }
                        None => break,
                    };

                    let payload = match frames.len() {
                        2 => &frames[1],
                        3 => &frames[2],
                        n => {
                            tracing::warn!("Unexpected frame count: {}", n);
                            continue;
                        }
                    };

                    let batch: KvEventBatch = match rmp_serde::from_slice(payload) {
                        Ok(b) => b,
                        Err(e) => {
                            tracing::warn!("Failed to deserialize event batch: {e}");
                            continue;
                        }
                    };

                    // Acquire the write lock per event rather than per batch so the
                    // publisher drain and kvbm-bridge writes can interleave between
                    // events. Each `RawKvEvent` is the atomic unit — the parent chain
                    // inside `BlockStored` lives entirely within one event, so dropping
                    // the lock between events does not break chaining.
                    for event in batch.events {
                        let mut guard = tracker.write().await;
                        process_event(&mut guard, event, engine_source);
                    }
                }
            }
        }
    });

    Ok(handle)
}

fn process_event(tracker: &mut Tracker, event: RawKvEvent, engine_source: EventSource) {
    match event {
        RawKvEvent::BlockStored {
            block_hashes,
            parent_block_hash,
            token_ids,
            block_size,
            lora_name,
            is_eagle,
            block_mm_infos,
            ..
        } => {
            if block_size == 0 {
                tracing::warn!(
                    "Invalid block_size 0 (must be positive), skipping event to avoid chunks() panic"
                );
                return;
            }

            // Eagle (speculative-decode) events use overlapping `block_size + 1` token
            // windows, not contiguous block_size-sized chunks. Naive chunking would
            // miscount and either fail the chunk/hash count check or hash a misaligned
            // window, polluting the dedup map. Drop until we have a real Eagle-aware
            // path.
            if matches!(is_eagle, Some(true)) {
                tracing::warn!(
                    "Eagle BlockStored event not supported by consolidator yet; skipping ({} hashes)",
                    block_hashes.len()
                );
                return;
            }

            // Multimodal info participates in the per-block hash (the placeholder slots
            // are encoded with their `mm_hash`, not their raw token IDs). We don't yet
            // thread `block_mm_infos` through to `kv_hashing::Request::mm_info`, so
            // hashing as plain tokens would produce the wrong PLH and silently break
            // cross-source dedup. Drop until plumbed through.
            if block_mm_infos
                .as_ref()
                .is_some_and(|v| v.iter().any(Option::is_some))
            {
                tracing::warn!(
                    "Multimodal BlockStored event not supported by consolidator yet; skipping ({} hashes)",
                    block_hashes.len()
                );
                return;
            }

            let token_chunks: Vec<Vec<u32>> =
                token_ids.chunks(block_size).map(|c| c.to_vec()).collect();

            if token_chunks.len() != block_hashes.len() {
                tracing::warn!(
                    "Token chunks ({}) don't match block hashes ({}), skipping event",
                    token_chunks.len(),
                    block_hashes.len()
                );
                return;
            }

            let mut current_parent = parent_block_hash.map(|h| h.into_u64().to_string());

            for (i, block_hash) in block_hashes.into_iter().enumerate() {
                let hash_str = block_hash.into_u64().to_string();
                tracker.handle_store(
                    engine_source,
                    hash_str.clone(),
                    current_parent.clone(),
                    token_chunks[i].clone(),
                    block_size,
                    lora_name.clone(),
                );
                current_parent = Some(hash_str);
            }
        }

        RawKvEvent::BlockRemoved { block_hashes, .. } => {
            for hash in block_hashes {
                tracker.handle_remove(engine_source, &hash.into_u64().to_string());
            }
        }

        RawKvEvent::AllBlocksCleared => {
            tracker.handle_clear_all();
        }

        RawKvEvent::Ignored => {}
    }
}
