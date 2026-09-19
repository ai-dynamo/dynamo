// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::collections::HashMap;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

use anyhow::{Context, Result};
use futures::StreamExt;
use tokio::sync::mpsc;
use tokio_util::sync::CancellationToken;

use dynamo_kv_router::protocols::*;
use dynamo_kv_router::zmq_wire::*;

use crate::kv_router::metrics::kv_publisher_metrics;
use crate::utils::zmq::{connect_sub_socket, multipart_message};

pub(super) struct DecodedZmqKvBatch {
    pub(super) source_cursor: u64,
    pub(super) batch: KvEventBatch,
}

/// Decode the transport envelope shared by legacy and residency-aware inputs.
///
/// Callers retain their own malformed-input and protocol-version policies.
pub(super) fn decode_zmq_kv_batch(
    mut frames: crate::utils::zmq::MultipartMessage,
) -> Result<DecodedZmqKvBatch> {
    if frames.len() != 3 {
        anyhow::bail!("expected three ZMQ frames, received {}", frames.len());
    }
    let payload = frames.pop().expect("frame count was validated");
    let sequence = frames.pop().expect("frame count was validated");
    let sequence: [u8; 8] = sequence.try_into().map_err(|sequence: Vec<u8>| {
        anyhow::anyhow!(
            "ZMQ sequence must contain eight bytes, received {}",
            sequence.len()
        )
    })?;
    let batch = decode_event_batch(&payload).context("failed to decode KV event batch")?;
    Ok(DecodedZmqKvBatch {
        source_cursor: u64::from_be_bytes(sequence),
        batch,
    })
}

#[allow(clippy::too_many_arguments)]
pub(super) async fn start_zmq_listener(
    zmq_endpoint: String,
    zmq_topic: String,
    worker_id: WorkerId,
    tx: mpsc::UnboundedSender<Vec<PlacementEvent>>,
    cancellation_token: CancellationToken,
    kv_block_size: u32,
    next_event_id: Arc<AtomicU64>,
    image_token_id: Option<u32>,
    video_token_id: Option<u32>,
    liveness: Option<(u32, std::time::Duration)>,
) {
    tracing::debug!(
        "KVEventPublisher connecting to ZMQ endpoint {} (topic '{}')",
        zmq_endpoint,
        zmq_topic
    );

    let mut normalizer = ZmqEventNormalizer::new(kv_block_size)
        .with_image_token_id(image_token_id)
        .with_video_token_id(video_token_id);
    let socket = match connect_sub_socket(&zmq_endpoint, Some(&zmq_topic)).await {
        Ok(socket) => socket,
        Err(error) => {
            tracing::error!(endpoint = %zmq_endpoint, topic = %zmq_topic, error = %error, "ZMQ listener failed to connect");
            return;
        }
    };
    let mut socket = socket;
    let metrics = kv_publisher_metrics();

    if cancellation_token.is_cancelled() {
        return;
    }

    let mut messages_processed = 0u64;
    let mut last_source_cursors = HashMap::<u32, u64>::new();
    let mut last_message = tokio::time::Instant::now();
    let mut expired = false;

    let exit_reason = 'main: loop {
        tokio::select! {
            biased;

            _ = cancellation_token.cancelled() => {
                tracing::debug!("ZMQ listener received cancellation signal");
                break 'main String::from("cancellation token cancelled");
            }

            _ = async {
                if let Some((_, timeout)) = liveness {
                    tokio::time::sleep_until(last_message + timeout).await;
                } else {
                    std::future::pending::<()>().await;
                }
            }, if !expired => {
                let (rank, _) = liveness.expect("deadline requires negotiated liveness");
                let worker = WorkerWithDpRank::new(worker_id, rank);
                let clear = normalizer.preprocess_with_reason(
                    RawKvEvent::AllBlocksCleared { ownership: None }, worker
                ).expect("clear is accepted");
                let clear = normalizer.normalize_preprocessed(
                    clear, next_event_id.fetch_add(1, Ordering::SeqCst), worker
                ).expect("clear normalizes");
                if tx.send(vec![clear]).is_err() {
                    break 'main String::from("channel receiver dropped");
                }
                expired = true;
                tracing::warn!(%zmq_endpoint, rank, "KV event source heartbeat expired; cleared residency");
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
                let DecodedZmqKvBatch {
                    source_cursor: engine_seq,
                    batch,
                } = match decode_zmq_kv_batch(frames) {
                    Ok(decoded) => decoded,
                    Err(error) => {
                        tracing::warn!(%error, "Failed to decode ZMQ KV batch");
                        continue;
                    }
                };

                tracing::trace!(
                    "ZMQ listener on {} received batch with {} events (engine_seq={}, dp_rank={})",
                    zmq_endpoint,
                    batch.events.len(),
                    engine_seq,
                    batch.data_parallel_rank.unwrap_or(0)
                );

                let dp_rank = batch.data_parallel_rank.unwrap_or(0).cast_unsigned();
                if let Some((expected_rank, _)) = liveness
                    && (batch.data_parallel_rank.is_none() || dp_rank != expected_rank)
                {
                    tracing::warn!(%zmq_endpoint, dp_rank, expected_rank, "Ignoring mismatched KV source rank");
                    continue;
                }
                last_message = tokio::time::Instant::now();
                expired = false;
                let discontinuity = last_source_cursors
                    .insert(dp_rank, engine_seq)
                    .is_some_and(|previous| previous.checked_add(1) != Some(engine_seq));
                let mut events = Vec::with_capacity(batch.events.len() + usize::from(discontinuity));
                if discontinuity {
                    tracing::warn!(
                        endpoint = %zmq_endpoint,
                        source_cursor = engine_seq,
                        dp_rank,
                        "ZMQ KV event sequence discontinuity; clearing stale router state"
                    );
                    let worker = WorkerWithDpRank::new(worker_id, dp_rank);
                    let raw_event = RawKvEvent::AllBlocksCleared { ownership: None };
                    let raw_event = normalizer
                        .preprocess_with_reason(raw_event, worker)
                        .expect("synthetic clear event is always accepted");
                    let event_id = next_event_id.fetch_add(1, Ordering::SeqCst);
                    let event = normalizer
                        .normalize_preprocessed(raw_event, event_id, worker)
                        .expect("synthetic clear event always normalizes");
                    events.push(event);
                }
                for raw_event in batch.events {
                    let event_type = raw_event.event_type_label();
                    if let Some(metrics) = &metrics {
                        metrics.increment_zmq_event("received", event_type);
                    }
                    let worker = WorkerWithDpRank::new(worker_id, dp_rank);
                    let raw_event = match normalizer.preprocess_with_reason(raw_event, worker) {
                        Ok(raw_event) => raw_event,
                        Err(reason) => {
                            if let Some(metrics) = &metrics {
                                metrics.increment_zmq_filtered_event(event_type, reason.as_label());
                            }
                            continue;
                        }
                    };
                    if let Some(metrics) = &metrics {
                        metrics.increment_zmq_event("accepted", event_type);
                    }
                    let event_id = next_event_id.fetch_add(1, Ordering::SeqCst);
                    let Some(event) =
                        normalizer.normalize_preprocessed(raw_event, event_id, worker)
                    else {
                        if let Some(metrics) = &metrics {
                            metrics.increment_zmq_conversion_issue(event_type, "conversion_none");
                        }
                        continue;
                    };
                    if matches!(event.event.data, KvCacheEventData::Stored(ref data) if data.blocks.is_empty())
                        && let Some(metrics) = &metrics
                    {
                        metrics.increment_zmq_suspicious_event(event_type, "empty_store_blocks");
                    }
                    events.push(event);
                }
                if !events.is_empty() {
                    let event_count = events.len() as u64;
                    if tx.send(events).is_err() {
                        tracing::warn!("Failed to send message to channel - receiver dropped");
                        break 'main String::from("channel receiver dropped");
                    }
                    messages_processed += event_count;
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
