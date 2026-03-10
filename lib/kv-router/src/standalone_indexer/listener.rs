// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::sync::Arc;
use std::sync::atomic::AtomicU32;
use std::time::Duration;

use rmp_serde as rmps;
use tokio::sync::watch;
use tokio_util::sync::CancellationToken;
use zeromq::{Socket, SocketRecv, SubSocket};

use crate::protocols::{RouterEvent, WorkerId};
use crate::zmq_wire::{KvEventBatch, convert_event};

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
    mut ready: watch::Receiver<bool>,
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

    tokio::select! {
        biased;
        _ = cancel.cancelled() => {
            tracing::info!(worker_id, dp_rank, "ZMQ listener cancelled before ready");
            return;
        }
        result = ready.wait_for(|&v| v) => {
            if result.is_err() {
                tracing::error!(worker_id, dp_rank, "Ready channel closed before signaling");
                return;
            }
        }
    }

    tracing::info!(worker_id, dp_rank, "ZMQ listener ready, starting recv loop");

    let mut next_event_id = 0u64;
    let warning_count = Arc::new(AtomicU32::new(0));
    let mut consecutive_errors = 0u32;
    #[expect(unused_assignments)]
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

                if msg.len() != 3 {
                    tracing::warn!(worker_id, "Unexpected ZMQ frame count: {}", msg.len());
                    continue;
                }

                let seq_bytes = msg.get(1).unwrap();
                if seq_bytes.len() != 8 {
                    tracing::warn!(worker_id, "Invalid sequence number length: {}", seq_bytes.len());
                    continue;
                }

                let payload = msg.get(2).unwrap();
                let batch_result = rmps::from_slice::<KvEventBatch>(payload);
                let Ok(batch) = batch_result else {
                    tracing::warn!(worker_id, "Failed to decode KvEventBatch: {}", batch_result.unwrap_err());
                    continue;
                };

                let effective_dp_rank = batch.data_parallel_rank.map_or(dp_rank, |r| r.cast_unsigned());
                for raw_event in batch.events {
                    let event_id = next_event_id;
                    next_event_id += 1;
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

#[cfg(test)]
mod tests {
    use zeromq::{PubSocket, Socket, SocketRecv, SocketSend, SubSocket};

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn zmq_buffers_messages_during_brief_delay() {
        let mut pub_socket = PubSocket::new();
        let bound_endpoint = pub_socket.bind("tcp://127.0.0.1:0").await.unwrap();

        let mut sub_socket = SubSocket::new();
        sub_socket.subscribe("").await.unwrap();
        sub_socket
            .connect(&bound_endpoint.to_string())
            .await
            .unwrap();

        let (tx, mut rx) = tokio::sync::mpsc::channel::<SubSocket>(1);
        tokio::spawn(async move {
            let _ = sub_socket.recv().await.unwrap();
            let _ = tx.send(sub_socket).await;
        });
        loop {
            pub_socket.send("probe".into()).await.unwrap();
            tokio::time::sleep(std::time::Duration::from_millis(50)).await;
            if let Ok(sub) = rx.try_recv() {
                sub_socket = sub;
                break;
            }
        }

        let num_messages = 10u64;

        for i in 0..num_messages {
            pub_socket
                .send(i.to_le_bytes().to_vec().into())
                .await
                .unwrap();
        }

        tokio::time::sleep(std::time::Duration::from_millis(500)).await;

        for i in 0u64..num_messages {
            let msg = tokio::time::timeout(std::time::Duration::from_secs(5), sub_socket.recv())
                .await
                .expect("timed out waiting for ZMQ message")
                .expect("ZMQ recv error");

            let payload = msg.get(0).unwrap();
            let received = u64::from_le_bytes(payload[..8].try_into().unwrap());
            assert_eq!(received, i, "message {i} arrived out of order");
        }
    }
}
