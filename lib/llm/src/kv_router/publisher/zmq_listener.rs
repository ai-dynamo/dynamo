// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

use futures::StreamExt;
use tokio::sync::mpsc;
use tokio_util::sync::CancellationToken;

use dynamo_kv_router::protocols::*;
use dynamo_kv_router::zmq_wire::*;

use crate::utils::zmq::{connect_sub_socket, multipart_message};

pub(super) async fn start_zmq_listener(
    zmq_endpoint: String,
    zmq_topic: String,
    worker_id: WorkerId,
    tx: mpsc::UnboundedSender<PlacementEvent>,
    cancellation_token: CancellationToken,
    kv_block_size: u32,
    next_event_id: Arc<AtomicU64>,
) {
    tracing::debug!(
        "KVEventPublisher connecting to ZMQ endpoint {} (topic '{}')",
        zmq_endpoint,
        zmq_topic
    );

    let mut normalizer = ZmqEventNormalizer::new(kv_block_size);
    let socket = match connect_sub_socket(&zmq_endpoint, Some(&zmq_topic)).await {
        Ok(socket) => socket,
        Err(error) => {
            tracing::error!(endpoint = %zmq_endpoint, topic = %zmq_topic, error = %error, "ZMQ listener failed to connect");
            return;
        }
    };
    let mut socket = socket;

    if cancellation_token.is_cancelled() {
        return;
    }

    let mut messages_processed = 0u64;

    let exit_reason = 'main: loop {
        tokio::select! {
            biased;

            _ = cancellation_token.cancelled() => {
                tracing::debug!("ZMQ listener received cancellation signal");
                break 'main String::from("cancellation token cancelled");
            }

            msg_result = socket.next() => {
                let frames = match msg_result {
                    Some(Ok(frames)) => multipart_message(frames),
                    Some(Err(error)) => {
                        tracing::error!(endpoint = %zmq_endpoint, error = %error, "ZMQ listener recv failed");
                        break 'main format!("ZMQ recv failed: {error}");
                    }
                    None => break 'main String::from("ZMQ stream ended"),
                };
                let mut frames = frames;

                if frames.len() != 3 {
                    tracing::warn!(
                        "Received unexpected ZMQ frame count: expected 3, actual {}",
                        frames.len()
                    );
                    continue;
                }

                let payload = frames.pop().unwrap();
                let seq_bytes = frames.pop().unwrap();

                if seq_bytes.len() != 8 {
                    tracing::warn!(
                        "Invalid sequence number byte length: expected 8, actual {}",
                        seq_bytes.len()
                    );
                    continue;
                }

                let engine_seq = u64::from_be_bytes(seq_bytes.try_into().unwrap());

                let batch_result = decode_event_batch(&payload);
                let Ok(batch) = batch_result else {
                    let e = batch_result.unwrap_err();
                    tracing::warn!("Failed to decode KVEventBatch msgpack: {e}");
                    continue;
                };

                tracing::trace!(
                    "ZMQ listener on {} received batch with {} events (engine_seq={}, dp_rank={})",
                    zmq_endpoint,
                    batch.events.len(),
                    engine_seq,
                    batch.data_parallel_rank.unwrap_or(0)
                );

                let dp_rank = batch.data_parallel_rank.unwrap_or(0).cast_unsigned();
                for raw_event in batch.events {
                    let worker = WorkerWithDpRank::new(worker_id, dp_rank);
                    let Some(raw_event) = normalizer.preprocess(raw_event, worker) else {
                        continue;
                    };
                    let event_id = next_event_id.fetch_add(1, Ordering::SeqCst);
                    let Some(event) =
                        normalizer.normalize_preprocessed(raw_event, event_id, worker)
                    else {
                        continue;
                    };
                    if tx.send(event).is_err() {
                        tracing::warn!("Failed to send message to channel - receiver dropped");
                        break 'main String::from("channel receiver dropped");
                    }
                    messages_processed += 1;
                }
            }
        }
    };

    tracing::debug!(
        "ZMQ listener exiting, reason: {}, messages processed: {}",
        exit_reason,
        messages_processed
    );
}
