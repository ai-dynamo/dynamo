// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::sync::Arc;
use std::sync::atomic::{AtomicU32, AtomicU64, Ordering};
use std::time::Duration;

use rmp_serde as rmps;
use tokio_util::sync::CancellationToken;
use zeromq::{Socket, SocketRecv, SubSocket};

use dynamo_kv_router::protocols::{RouterEvent, WorkerId};
use dynamo_kv_router::zmq_wire::{KvEventBatch, convert_event};

use super::indexer::Indexer;

const INITIAL_BACKOFF_MS: u64 = 10;
const MAX_BACKOFF_MS: u64 = 5000;
const MAX_CONSECUTIVE_ERRORS: u32 = 10;
const MAX_BACKOFF_EXPONENT: u32 = 8;

fn calculate_backoff_ms(consecutive_errors: u32) -> u64 {
    std::cmp::min(
        INITIAL_BACKOFF_MS * 2_u64.pow(consecutive_errors.min(MAX_BACKOFF_EXPONENT)),
        MAX_BACKOFF_MS,
    )
}

pub async fn run_zmq_listener(
    worker_id: WorkerId,
    dp_rank: u32,
    zmq_address: String,
    block_size: u32,
    indexer: Indexer,
    cancel: CancellationToken,
) {
    tracing::info!(worker_id, dp_rank, zmq_address, "ZMQ listener starting");

    let mut socket = SubSocket::new();

    if let Err(e) = socket.subscribe("").await {
        tracing::error!("Failed to subscribe on ZMQ socket: {e}");
        return;
    }

    if let Err(e) = socket.connect(&zmq_address).await {
        tracing::error!("Failed to connect ZMQ SUB socket to {zmq_address}: {e}");
        return;
    }

    let next_event_id = AtomicU64::new(0);
    let warning_count = Arc::new(AtomicU32::new(0));
    let mut consecutive_errors = 0u32;
    #[allow(unused_assignments)]
    let mut exit_reason = "unknown";
    let mut messages_processed = 0u64;

    'main: loop {
        tokio::select! {
            biased;

            _ = cancel.cancelled() => {
                exit_reason = "cancelled";
                break 'main;
            }

            msg_result = socket.recv() => {
                let Ok(msg) = msg_result else {
                    let e = msg_result.unwrap_err();
                    consecutive_errors += 1;

                    if consecutive_errors >= MAX_CONSECUTIVE_ERRORS {
                        tracing::error!(
                            error=%e,
                            consecutive_errors,
                            worker_id,
                            "Too many consecutive ZMQ errors, terminating listener"
                        );
                        exit_reason = "too many consecutive errors";
                        break 'main;
                    }

                    let backoff_ms = calculate_backoff_ms(consecutive_errors);
                    tracing::warn!(
                        error=%e,
                        consecutive_errors,
                        backoff_ms,
                        worker_id,
                        "ZMQ recv error, backing off"
                    );
                    tokio::time::sleep(Duration::from_millis(backoff_ms)).await;
                    continue;
                };

                consecutive_errors = 0;

                let mut frames: Vec<Vec<u8>> = msg.into_vec().into_iter().map(|f| f.to_vec()).collect();
                if frames.len() != 3 {
                    tracing::warn!(worker_id, "Unexpected ZMQ frame count: {}", frames.len());
                    continue;
                }

                let payload = frames.pop().unwrap();
                let seq_bytes = frames.pop().unwrap();

                if seq_bytes.len() != 8 {
                    tracing::warn!(worker_id, "Invalid sequence number length: {}", seq_bytes.len());
                    continue;
                }

                let batch_result = rmps::from_slice::<KvEventBatch>(&payload);
                let Ok(batch) = batch_result else {
                    tracing::warn!(worker_id, "Failed to decode KvEventBatch: {}", batch_result.unwrap_err());
                    continue;
                };

                let effective_dp_rank = batch.data_parallel_rank.map_or(dp_rank, |r| r as u32);
                for raw_event in batch.events.into_iter() {
                    let event_id = next_event_id.fetch_add(1, Ordering::SeqCst);
                    let kv_event = convert_event(raw_event, event_id, block_size, effective_dp_rank, &warning_count);
                    let router_event = RouterEvent::new(worker_id, kv_event);
                    indexer.apply_event(router_event).await;
                    messages_processed += 1;
                }
            }
        }
    }

    tracing::info!(
        worker_id,
        dp_rank,
        exit_reason,
        messages_processed,
        "ZMQ listener exiting"
    );
}
