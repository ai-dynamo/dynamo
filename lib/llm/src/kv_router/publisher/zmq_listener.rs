// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

use anyhow::{Context, Result};
use futures::StreamExt;
use tokio::sync::mpsc;
use tokio_util::sync::CancellationToken;

use dynamo_kv_router::protocols::*;
use dynamo_kv_router::zmq_wire::*;

use super::{PublisherInput, ZmqBootstrapConfig};
use crate::kv_router::metrics::kv_publisher_metrics;
use crate::utils::zmq::{connect_sub_socket, connect_sub_socket_with_monitor, multipart_message};

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
    tx: mpsc::UnboundedSender<PublisherInput>,
    cancellation_token: CancellationToken,
    kv_block_size: u32,
    next_event_id: Arc<AtomicU64>,
    image_token_id: Option<u32>,
    video_token_id: Option<u32>,
    bootstrap: Option<ZmqBootstrapConfig>,
) {
    tracing::debug!(
        "KVEventPublisher connecting to ZMQ endpoint {} (topic '{}')",
        zmq_endpoint,
        zmq_topic
    );

    let mut normalizer = ZmqEventNormalizer::new(kv_block_size)
        .with_image_token_id(image_token_id)
        .with_video_token_id(video_token_id);
    let connection = if bootstrap.is_some() {
        connect_sub_socket_with_monitor(&zmq_endpoint, &zmq_topic)
            .await
            .map(|(socket, monitor)| (socket, Some(monitor)))
    } else {
        connect_sub_socket(&zmq_endpoint, Some(&zmq_topic))
            .await
            .map(|socket| (socket, None))
    };
    let (mut socket, monitor) = match connection {
        Ok(connection) => connection,
        Err(error) => {
            tracing::error!(%error, "ZMQ listener failed to connect");
            return; // Dropping the completion sender reports uncertainty.
        }
    };
    let mut bootstrap_next = 0;
    let mut expected_rank = None;
    if let Some(config) = bootstrap {
        expected_rank = Some(config.dp_rank);
        let result = tokio::select! {
            biased;
            _ = cancellation_token.cancelled() => return,
            result = super::zmq_bootstrap::bootstrap(
                &mut socket, monitor.expect("bootstrap monitor"), &mut normalizer,
                &tx, &next_event_id, worker_id, &config,
            ) => result,
        };
        let (outcome, next) = match result {
            Ok(next) => (super::BootstrapOutcome::Success, Some(next)),
            Err(outcome) => (outcome, None),
        };
        let accepted = config.completion.send(outcome).is_ok();
        let Some(next) = next.filter(|_| accepted) else {
            return;
        };
        bootstrap_next = next;
    }

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

                if engine_seq < bootstrap_next {
                    continue; // Already applied during bootstrap.
                }
                let dp_rank = batch.data_parallel_rank.unwrap_or(expected_rank.unwrap_or(0) as i32).cast_unsigned();
                if expected_rank.is_some_and(|rank| rank != dp_rank) {
                    tracing::warn!(dp_rank, "KV batch belongs to the wrong rank");
                    continue;
                }
                let events = normalize_batch(batch, &mut normalizer, WorkerWithDpRank::new(worker_id, dp_rank), &next_event_id);
                if !events.is_empty() {
                    let event_count = events.len() as u64;
                    if tx.send(PublisherInput::Events(events)).is_err() {
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

/// Both startup replay and ordinary live input use the same normalizer and
/// event ID allocator; only their application/acknowledgement policy differs.
pub(super) fn normalize_batch(
    batch: KvEventBatch,
    normalizer: &mut ZmqEventNormalizer,
    worker: WorkerWithDpRank,
    next_event_id: &AtomicU64,
) -> Vec<PlacementEvent> {
    let metrics = kv_publisher_metrics();
    let mut events = Vec::with_capacity(batch.events.len());
    for raw_event in batch.events {
        let event_type = raw_event.event_type_label();
        if let Some(metrics) = &metrics {
            metrics.increment_zmq_event("received", event_type);
        }
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
        let Some(event) = normalizer.normalize_preprocessed(raw_event, event_id, worker) else {
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
    events
}
