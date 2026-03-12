// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::sync::Arc;
use std::sync::atomic::{AtomicU32, AtomicU64, Ordering};
use std::time::Duration;

use bytes::Bytes;
use rmp_serde as rmps;
use tokio::sync::watch;
use tokio_util::sync::CancellationToken;
use zeromq::{Socket, SocketRecv, SocketSend, SubSocket};

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

/// Sentinel value for `watermark`: indicates no batch has been processed yet.
const WATERMARK_UNSET: u64 = u64::MAX;

/// Replay missed batches from the engine's ROUTER socket.
///
/// Uses a DEALER socket (no send/recv lockstep) to send one request and
/// receive multiple response frames. Each response is `[empty, seq, payload]`;
/// an empty payload signals end of replay.
#[expect(clippy::too_many_arguments)]
async fn replay_gap(
    replay_socket: &mut zeromq::DealerSocket,
    start_seq: u64,
    end_seq: u64,
    worker_id: WorkerId,
    dp_rank: u32,
    block_size: u32,
    indexer: &Indexer,
    warning_count: &Arc<AtomicU32>,
    watermark: &Arc<AtomicU64>,
) -> u64 {
    tracing::info!(
        worker_id,
        dp_rank,
        start_seq,
        end_seq,
        "Requesting replay from engine"
    );

    // DEALER must manually prepend the empty delimiter that REQ adds automatically.
    let req_frames = vec![Bytes::new(), Bytes::from(start_seq.to_be_bytes().to_vec())];
    let Ok(req_msg) = zeromq::ZmqMessage::try_from(req_frames) else {
        tracing::error!(worker_id, dp_rank, "Failed to build replay request");
        return 0;
    };
    if let Err(e) = replay_socket.send(req_msg).await {
        tracing::error!(worker_id, dp_rank, error = %e, "Failed to send replay request");
        return 0;
    }

    let mut replayed = 0u64;
    loop {
        let Ok(msg) = replay_socket.recv().await else {
            tracing::error!(worker_id, dp_rank, "Replay recv error");
            break;
        };
        // ROUTER sends [identity, empty, seq, payload]; DEALER strips identity,
        // so we receive [empty, seq, payload].
        if msg.len() < 3 {
            tracing::warn!(
                worker_id,
                dp_rank,
                "Unexpected replay frame count: {}",
                msg.len()
            );
            break;
        }

        let payload = msg.get(2).unwrap();
        if payload.is_empty() {
            break;
        }

        let seq_bytes = msg.get(1).unwrap();
        if seq_bytes.len() != 8 {
            tracing::warn!(
                worker_id,
                dp_rank,
                "Invalid replay seq length: {}",
                seq_bytes.len()
            );
            break;
        }
        let seq = u64::from_be_bytes(seq_bytes[..8].try_into().unwrap());

        let Ok(batch) = rmps::from_slice::<KvEventBatch>(payload) else {
            tracing::warn!(worker_id, dp_rank, seq, "Failed to decode replayed batch");
            continue;
        };

        let effective_dp_rank = batch
            .data_parallel_rank
            .map_or(dp_rank, |r| r.cast_unsigned());
        for raw_event in batch.events {
            let kv_event =
                convert_event(raw_event, seq, block_size, effective_dp_rank, warning_count);
            let router_event = RouterEvent::new(worker_id, kv_event);
            indexer.apply_event(router_event).await;
        }
        watermark.store(seq, Ordering::Release);
        replayed += 1;
    }

    tracing::info!(worker_id, dp_rank, replayed, "Replay complete");
    replayed
}

// TODO: assumes one dp_rank per ZMQ socket. Seq counter is per-socket so gap
// detection works regardless, but replay semantics may differ if a single
// socket multiplexes dp_ranks.

/// Connect the ZMQ SUB socket, then spawn a background task that waits for
/// the ready signal before entering the recv loop.
///
/// Returns once the SUB socket is connected (subscription handshake begins
/// immediately in the background). The ready gate and recv loop run in a
/// spawned task so `register()` is never blocked waiting for `signal_ready()`.
#[expect(clippy::too_many_arguments)]
pub async fn run_zmq_listener(
    worker_id: WorkerId,
    dp_rank: u32,
    zmq_address: String,
    block_size: u32,
    indexer: Indexer,
    cancel: CancellationToken,
    ready: watch::Receiver<bool>,
    replay_endpoint: Option<String>,
    watermark: Arc<AtomicU64>,
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

    // Spawn the ready-wait + recv loop so the caller returns immediately.
    // The ZMQ subscription handshake proceeds in the background while P2P
    // recovery runs; once signal_ready() fires the recv loop starts draining
    // any buffered messages.
    tokio::spawn(zmq_wait_ready_then_recv(
        worker_id,
        dp_rank,
        block_size,
        indexer,
        cancel,
        ready,
        socket,
        replay_endpoint,
        watermark,
    ));
}

#[expect(clippy::too_many_arguments)]
async fn zmq_wait_ready_then_recv(
    worker_id: WorkerId,
    dp_rank: u32,
    block_size: u32,
    indexer: Indexer,
    cancel: CancellationToken,
    mut ready: watch::Receiver<bool>,
    socket: SubSocket,
    replay_endpoint: Option<String>,
    watermark: Arc<AtomicU64>,
) {
    // Wait for the ready signal before entering the recv loop.
    // During P2P recovery, this delay lets the recovery code fetch the dump
    // from a peer while ZMQ subscription handshakes complete in the background.
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

    // Connect DEALER socket once if replay_endpoint is configured.
    // DEALER (not REQ) because we send one request and receive multiple responses.
    let mut replay_socket = None;
    if let Some(ref ep) = replay_endpoint {
        let mut sock = zeromq::DealerSocket::new();
        if let Err(e) = sock.connect(ep).await {
            tracing::error!(worker_id, dp_rank, error = %e, "Failed to connect replay socket to {ep}");
        } else {
            tracing::info!(
                worker_id,
                dp_rank,
                replay_endpoint = ep,
                "Replay socket connected"
            );
            replay_socket = Some(sock);
        }
    }

    zmq_recv_loop(
        worker_id,
        dp_rank,
        block_size,
        indexer,
        cancel,
        socket,
        replay_socket,
        watermark,
    )
    .await;
}

#[expect(clippy::too_many_arguments)]
async fn zmq_recv_loop(
    worker_id: WorkerId,
    dp_rank: u32,
    block_size: u32,
    indexer: Indexer,
    cancel: CancellationToken,
    mut socket: SubSocket,
    mut replay_socket: Option<zeromq::DealerSocket>,
    watermark: Arc<AtomicU64>,
) {
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

                let seq = u64::from_be_bytes(seq_bytes[..8].try_into().unwrap());

                // Gap detection
                let prev = watermark.load(Ordering::Acquire);
                if prev != WATERMARK_UNSET && seq > prev + 1 {
                    let gap_start = prev + 1;
                    tracing::warn!(
                        worker_id, dp_rank,
                        expected = gap_start, got = seq,
                        "Gap detected: expected seq {gap_start}, got {seq}"
                    );
                    match replay_socket.as_mut() {
                        Some(sock) => {
                            replay_gap(
                                sock, gap_start, seq, worker_id, dp_rank,
                                block_size, &indexer, &warning_count, &watermark,
                            ).await;
                        }
                        None => tracing::warn!(
                            worker_id, dp_rank,
                            gap_size = seq - gap_start,
                            "No replay endpoint configured, {gap_size} batches lost",
                            gap_size = seq - gap_start,
                        ),
                    }
                }

                // After replay, watermark may have advanced past the current
                // batch — skip to avoid double-apply. Exclude the sentinel
                // (WATERMARK_UNSET) so the very first message is not skipped.
                let current_wm = watermark.load(Ordering::Acquire);
                if current_wm != WATERMARK_UNSET && current_wm >= seq {
                    continue;
                }

                let payload = msg.get(2).unwrap();
                let batch_result = rmps::from_slice::<KvEventBatch>(payload);
                let Ok(batch) = batch_result else {
                    tracing::warn!(worker_id, "Failed to decode KvEventBatch: {}", batch_result.unwrap_err());
                    continue;
                };

                let effective_dp_rank = batch.data_parallel_rank.map_or(dp_rank, |r| r.cast_unsigned());
                // Use the engine's ZMQ sequence number as event_id so downstream
                // consumers can detect gaps and request replay.
                for raw_event in batch.events {
                    let kv_event = convert_event(raw_event, seq, block_size, effective_dp_rank, &warning_count);
                    let router_event = RouterEvent::new(worker_id, kv_event);
                    indexer.apply_event(router_event).await;
                    messages_processed += 1;
                }
                watermark.store(seq, Ordering::Release);
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

    /// Verify that the `zeromq` crate buffers a small number of messages in
    /// TCP kernel buffers when `recv()` is not being called. The PUB socket
    /// uses `try_send` with a noop waker — once the TCP send buffer is full
    /// it silently drops messages (per ZMQ spec RFC 29). This test confirms
    /// that a brief delay (simulating P2P recovery) doesn't lose messages.
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

        // Wait for SUB handshake: spawn recv in a background task so the
        // PUB's accept/subscription processing can proceed concurrently.
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

        // Send messages without calling recv() — simulates the brief window
        // between ZMQ connect and ready signal during P2P recovery.
        for i in 0..num_messages {
            pub_socket
                .send(i.to_le_bytes().to_vec().into())
                .await
                .unwrap();
        }

        // Simulate recovery delay
        tokio::time::sleep(std::time::Duration::from_millis(500)).await;

        // All messages should be buffered in TCP kernel buffers
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
