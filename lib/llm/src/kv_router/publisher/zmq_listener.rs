// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::sync::Arc;
use std::sync::atomic::{AtomicU32, AtomicU64, Ordering};
use std::time::Duration;

use rmp_serde as rmps;
use tokio::sync::mpsc;
use tokio_util::sync::CancellationToken;
use zeromq::SocketRecv;

use crate::utils::zmq::connect_sub_socket_with_retry;
use dynamo_kv_router::protocols::*;
use dynamo_kv_router::zmq_wire::*;

pub(super) const INITIAL_BACKOFF_MS: u64 = 10;
pub(super) const MAX_BACKOFF_MS: u64 = 5000;
pub(super) const MAX_CONSECUTIVE_ERRORS: u32 = 10;
pub(super) const MAX_BACKOFF_EXPONENT: u32 = 8;

pub(super) fn calculate_backoff_ms(consecutive_errors: u32) -> u64 {
    std::cmp::min(
        INITIAL_BACKOFF_MS * 2_u64.pow(consecutive_errors.min(MAX_BACKOFF_EXPONENT)),
        MAX_BACKOFF_MS,
    )
}

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

    let warning_count = Arc::new(AtomicU32::new(0));
    let Some(mut socket) = connect_sub_socket_with_retry(
        &zmq_endpoint,
        Some(&zmq_topic),
        &cancellation_token,
        "ZMQ listener",
    )
    .await
    else {
        return;
    };

    let mut consecutive_errors = 0u32;
    #[expect(unused_assignments)]
    let mut exit_reason = "unknown";
    let mut messages_processed = 0u64;

    'main: loop {
        tokio::select! {
            biased;

            _ = cancellation_token.cancelled() => {
                tracing::debug!("ZMQ listener received cancellation signal");
                exit_reason = "cancellation token cancelled";
                break 'main;
            }

            msg_result = socket.recv() => {
                let Ok(msg) = msg_result else {
                    let e = msg_result.unwrap_err();
                    consecutive_errors += 1;

                    if consecutive_errors >= MAX_CONSECUTIVE_ERRORS {
                        tracing::error!(
                            error=%e,
                            consecutive_errors=%consecutive_errors,
                            "Too many consecutive ZMQ errors, terminating listener"
                        );
                        exit_reason = "too many consecutive errors";
                        break 'main;
                    }

                    let backoff_ms = calculate_backoff_ms(consecutive_errors);

                    tracing::warn!(
                        error=%e,
                        consecutive_errors=%consecutive_errors,
                        backoff_ms=%backoff_ms,
                        "Error reading from ZMQ socket, applying exponential backoff"
                    );

                    tokio::time::sleep(Duration::from_millis(backoff_ms)).await;
                    continue;
                };
                consecutive_errors = 0;

                let mut frames: Vec<Vec<u8>> =
                    msg.into_vec().into_iter().map(|frame| frame.to_vec()).collect();

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

                let batch_result = rmps::from_slice::<KvEventBatch>(&payload);
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
                    let event_id = next_event_id.fetch_add(1, Ordering::SeqCst);
                    let worker = WorkerWithDpRank::new(worker_id, dp_rank);
                    let event =
                        convert_event(raw_event, event_id, kv_block_size, worker, &warning_count);
                    if tx.send(event).is_err() {
                        tracing::warn!("Failed to send message to channel - receiver dropped");
                        exit_reason = "channel receiver dropped";
                        break 'main;
                    }
                    messages_processed += 1;
                }
            }
        }
    }

    tracing::debug!(
        "ZMQ listener exiting, reason: {}, messages processed: {}",
        exit_reason,
        messages_processed
    );
}
