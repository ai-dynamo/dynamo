// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::sync::Arc;
use std::sync::atomic::{AtomicU32, AtomicU64, Ordering};
use std::time::Duration;

use bytes::Bytes;
use rmp_serde as rmps;
use tokio::sync::watch;
use tokio_util::sync::CancellationToken;
use zeromq::{DealerSocket, Socket, SocketRecv, SocketSend, SubSocket};

use crate::protocols::{WorkerId, WorkerWithDpRank};
use crate::recovery::{CursorObservation, CursorState};
use crate::zmq_wire::{KvEventBatch, convert_event};

use super::indexer::Indexer;
use super::registry::ListenerRecord;

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

const WATERMARK_UNSET: u64 = u64::MAX;

fn cursor_from_watermark(watermark: u64) -> CursorState {
    if watermark == WATERMARK_UNSET {
        CursorState::Initial
    } else {
        CursorState::Live(watermark)
    }
}

struct ListenerLoop {
    worker_id: WorkerId,
    dp_rank: u32,
    block_size: u32,
    indexer: Indexer,
    cancel: CancellationToken,
    socket: SubSocket,
    replay_socket: Option<DealerSocket>,
    watermark: Arc<AtomicU64>,
    warning_count: Arc<AtomicU32>,
    consecutive_errors: u32,
    messages_processed: u64,
}

impl ListenerLoop {
    #[expect(clippy::too_many_arguments)]
    fn new(
        worker_id: WorkerId,
        dp_rank: u32,
        block_size: u32,
        indexer: Indexer,
        cancel: CancellationToken,
        socket: SubSocket,
        replay_socket: Option<DealerSocket>,
        watermark: Arc<AtomicU64>,
    ) -> Self {
        Self {
            worker_id,
            dp_rank,
            block_size,
            indexer,
            cancel,
            socket,
            replay_socket,
            watermark,
            warning_count: Arc::new(AtomicU32::new(0)),
            consecutive_errors: 0,
            messages_processed: 0,
        }
    }

    fn cursor(&self) -> CursorState {
        cursor_from_watermark(self.watermark.load(Ordering::Acquire))
    }

    async fn replay_gap(&mut self, start_seq: u64, end_seq: u64) -> u64 {
        tracing::info!(
            self.worker_id,
            self.dp_rank,
            start_seq,
            end_seq,
            "Requesting replay from engine"
        );

        let Some(replay_socket) = self.replay_socket.as_mut() else {
            tracing::warn!(
                self.worker_id,
                self.dp_rank,
                gap_size = end_seq.saturating_sub(start_seq),
                "No replay endpoint configured; batches lost"
            );
            return 0;
        };

        let worker_id = self.worker_id;
        let dp_rank = self.dp_rank;
        let block_size = self.block_size;
        let indexer = &self.indexer;
        let warning_count = &self.warning_count;
        let watermark = &self.watermark;

        let req_frames = vec![Bytes::new(), Bytes::from(start_seq.to_be_bytes().to_vec())];
        let Ok(req_msg) = zeromq::ZmqMessage::try_from(req_frames) else {
            tracing::error!(worker_id, dp_rank, "Failed to build replay request");
            return 0;
        };
        if let Err(error) = replay_socket.send(req_msg).await {
            tracing::error!(worker_id, dp_rank, error = %error, "Failed to send replay request");
            return 0;
        }

        let mut replayed = 0u64;
        loop {
            let Ok(msg) = replay_socket.recv().await else {
                tracing::error!(worker_id, dp_rank, "Replay recv error");
                break;
            };
            if msg.len() < 3 {
                tracing::warn!(
                    worker_id,
                    dp_rank,
                    "Unexpected replay frame count: {}",
                    msg.len()
                );
                break;
            }

            let payload = msg.get(2).expect("frame count checked above");
            if payload.is_empty() {
                break;
            }

            let seq_bytes = msg.get(1).expect("frame count checked above");
            if seq_bytes.len() != 8 {
                tracing::warn!(
                    worker_id,
                    dp_rank,
                    "Invalid replay seq length: {}",
                    seq_bytes.len()
                );
                break;
            }
            let seq = u64::from_be_bytes(seq_bytes[..8].try_into().expect("length checked above"));

            let Ok(batch) = rmps::from_slice::<KvEventBatch>(payload) else {
                tracing::warn!(worker_id, dp_rank, seq, "Failed to decode replayed batch");
                continue;
            };

            let effective_dp_rank = batch
                .data_parallel_rank
                .map_or(dp_rank, |rank| rank.cast_unsigned());
            for raw_event in batch.events {
                let placement_event = convert_event(
                    raw_event,
                    seq,
                    block_size,
                    WorkerWithDpRank::new(worker_id, effective_dp_rank),
                    warning_count,
                );
                if !placement_event.placement.is_local_gpu() {
                    continue;
                }
                let router_event = placement_event
                    .into_router_event()
                    .expect("local worker placement must convert to router event");
                indexer.apply_event(router_event).await;
            }
            watermark.store(seq, Ordering::Release);
            replayed += 1;
        }

        tracing::info!(worker_id, dp_rank, replayed, "Replay complete");
        replayed
    }

    async fn handle_gap(&mut self, seq: u64) {
        match self.cursor().observe(seq) {
            CursorObservation::Initial { got } if got > 0 => {
                tracing::warn!(
                    self.worker_id,
                    self.dp_rank,
                    expected = 0,
                    got,
                    "Gap detected: expected seq 0, got {got}"
                );
                self.replay_gap(0, got).await;
            }
            CursorObservation::Gap { expected, got } => {
                tracing::warn!(
                    self.worker_id,
                    self.dp_rank,
                    expected,
                    got,
                    "Gap detected: expected seq {expected}, got {got}"
                );
                self.replay_gap(expected, got).await;
            }
            CursorObservation::Initial { .. }
            | CursorObservation::Contiguous { .. }
            | CursorObservation::Stale { .. }
            | CursorObservation::FreshAfterBarrier { .. } => {}
        }
    }

    async fn apply_live_batch(&mut self, seq: u64, payload: &[u8]) {
        let batch = match rmps::from_slice::<KvEventBatch>(payload) {
            Ok(batch) => batch,
            Err(error) => {
                tracing::warn!(
                    self.worker_id,
                    self.dp_rank,
                    "Failed to decode KvEventBatch: {error}"
                );
                return;
            }
        };

        let effective_dp_rank = batch
            .data_parallel_rank
            .map_or(self.dp_rank, |rank| rank.cast_unsigned());
        for raw_event in batch.events {
            let placement_event = convert_event(
                raw_event,
                seq,
                self.block_size,
                WorkerWithDpRank::new(self.worker_id, effective_dp_rank),
                &self.warning_count,
            );
            if !placement_event.placement.is_local_gpu() {
                continue;
            }
            let router_event = placement_event
                .into_router_event()
                .expect("local worker placement must convert to router event");
            self.indexer.apply_event(router_event).await;
            self.messages_processed += 1;
        }
        self.watermark.store(seq, Ordering::Release);
    }

    async fn handle_message(&mut self, msg: zeromq::ZmqMessage) {
        if msg.len() != 3 {
            tracing::warn!(
                self.worker_id,
                self.dp_rank,
                "Unexpected ZMQ frame count: {}",
                msg.len()
            );
            return;
        }

        let seq_bytes = msg.get(1).expect("frame count checked above");
        if seq_bytes.len() != 8 {
            tracing::warn!(
                self.worker_id,
                self.dp_rank,
                "Invalid sequence number length: {}",
                seq_bytes.len()
            );
            return;
        }

        let seq = u64::from_be_bytes(seq_bytes[..8].try_into().expect("length checked above"));
        self.handle_gap(seq).await;

        if matches!(self.cursor().observe(seq), CursorObservation::Stale { .. }) {
            return;
        }

        let payload = msg.get(2).expect("frame count checked above");
        self.apply_live_batch(seq, payload).await;
    }

    async fn run(mut self) -> Result<(), String> {
        loop {
            let msg = tokio::select! {
                biased;

                _ = self.cancel.cancelled() => {
                    tracing::info!(
                        self.worker_id,
                        self.dp_rank,
                        self.messages_processed,
                        "ZMQ listener exiting after cancellation"
                    );
                    return Ok(());
                }

                msg_result = self.socket.recv() => {
                    match msg_result {
                        Ok(msg) => {
                            self.consecutive_errors = 0;
                            msg
                        }
                        Err(error) => {
                            self.consecutive_errors += 1;

                            if self.consecutive_errors >= MAX_CONSECUTIVE_ERRORS {
                                return Err(format!(
                                    "too many consecutive ZMQ recv errors for worker {} dp_rank {}: {error}",
                                    self.worker_id,
                                    self.dp_rank,
                                ));
                            }

                            let backoff_ms = calculate_backoff_ms(self.consecutive_errors);
                            tracing::warn!(
                                error = %error,
                                consecutive_errors = self.consecutive_errors,
                                backoff_ms,
                                worker_id = self.worker_id,
                                dp_rank = self.dp_rank,
                                "ZMQ recv error, backing off"
                            );
                            tokio::time::sleep(Duration::from_millis(backoff_ms)).await;
                            continue;
                        }
                    }
                }
            };

            self.handle_message(msg).await;
        }
    }
}

pub fn spawn_zmq_listener(
    worker_id: WorkerId,
    dp_rank: u32,
    record: Arc<ListenerRecord>,
    ready: watch::Receiver<bool>,
    generation: u64,
    cancel: CancellationToken,
) {
    tokio::spawn(async move {
        if let Err(error) = run_listener(
            worker_id,
            dp_rank,
            record.clone(),
            ready,
            generation,
            cancel,
        )
        .await
        {
            tracing::error!(worker_id, dp_rank, error = %error, "ZMQ listener failed");
            record.try_mark_failed(generation, error);
        }
    });
}

async fn run_listener(
    worker_id: WorkerId,
    dp_rank: u32,
    record: Arc<ListenerRecord>,
    mut ready: watch::Receiver<bool>,
    generation: u64,
    cancel: CancellationToken,
) -> Result<(), String> {
    let endpoint = record.endpoint().to_string();
    let replay_endpoint = record.replay_endpoint().map(str::to_string);
    let block_size = record.block_size();
    let indexer = record.indexer();
    let watermark = record.watermark();

    tracing::info!(worker_id, dp_rank, endpoint, "ZMQ listener starting");

    if cancel.is_cancelled() {
        return Ok(());
    }

    let mut socket = SubSocket::new();
    socket
        .subscribe("")
        .await
        .map_err(|e| format!("failed to subscribe on ZMQ socket: {e}"))?;

    tokio::select! {
        _ = cancel.cancelled() => return Ok(()),
        result = socket.connect(&endpoint) => {
            result.map_err(|e| format!("failed to connect ZMQ SUB socket to {endpoint}: {e}"))?;
        }
    }

    tokio::select! {
        _ = cancel.cancelled() => return Ok(()),
        result = ready.wait_for(|&value| value) => {
            result.map_err(|_| "ready channel closed before signaling".to_string())?;
        }
    }

    if !record.try_mark_active(generation) {
        tracing::debug!(
            worker_id,
            dp_rank,
            "Listener attempt is stale after readiness gate; exiting"
        );
        return Ok(());
    }

    tracing::info!(worker_id, dp_rank, "ZMQ listener ready, starting recv loop");

    let replay_socket =
        connect_replay_socket(worker_id, dp_rank, replay_endpoint.as_deref(), &cancel).await;
    if cancel.is_cancelled() || !record.is_current_attempt(generation) {
        return Ok(());
    }

    ListenerLoop::new(
        worker_id,
        dp_rank,
        block_size,
        indexer,
        cancel,
        socket,
        replay_socket,
        watermark,
    )
    .run()
    .await
}

async fn connect_replay_socket(
    worker_id: WorkerId,
    dp_rank: u32,
    replay_endpoint: Option<&str>,
    cancel: &CancellationToken,
) -> Option<DealerSocket> {
    let endpoint = replay_endpoint?;

    let mut socket = DealerSocket::new();
    tokio::select! {
        _ = cancel.cancelled() => None,
        result = socket.connect(endpoint) => {
            match result {
                Ok(()) => {
                    tracing::info!(
                        worker_id,
                        dp_rank,
                        replay_endpoint = endpoint,
                        "Replay socket connected"
                    );
                    Some(socket)
                }
                Err(e) => {
                    tracing::error!(
                        worker_id,
                        dp_rank,
                        error = %e,
                        "Failed to connect replay socket to {endpoint}"
                    );
                    None
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{WATERMARK_UNSET, cursor_from_watermark};
    use crate::recovery::CursorObservation;
    use zeromq::{PubSocket, Socket, SocketRecv, SocketSend, SubSocket};

    #[test]
    fn initial_gap_replays_from_zero_and_replayed_seq_becomes_stale() {
        let replay_start = match cursor_from_watermark(WATERMARK_UNSET).observe(5) {
            CursorObservation::Initial { got } if got > 0 => Some(0),
            CursorObservation::Gap { expected, .. } => Some(expected),
            _ => None,
        };
        assert_eq!(replay_start, Some(0));
        assert!(matches!(
            cursor_from_watermark(5).observe(5),
            CursorObservation::Stale {
                got: 5,
                last_applied: Some(5),
            }
        ));
    }

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
