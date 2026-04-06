// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::sync::Arc;
use std::sync::atomic::{AtomicU32, AtomicU64, Ordering};

use rmp_serde as rmps;
use tokio::sync::watch;
use tokio_util::sync::CancellationToken;

use crate::protocols::{WorkerId, WorkerWithDpRank};
use crate::recovery::{CursorObservation, CursorState};
use crate::zmq_wire::{KvEventBatch, convert_event};

use super::indexer::Indexer;
use super::registry::ListenerRecord;
use super::zmq::{
    MultipartMessage, SharedSocket, connect_dealer_socket, connect_sub_socket, recv_multipart,
    send_multipart,
};

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
    live_socket: SharedSocket,
    replay_socket: Option<SharedSocket>,
    watermark: Arc<AtomicU64>,
    warning_count: Arc<AtomicU32>,
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
        live_socket: SharedSocket,
        replay_socket: Option<SharedSocket>,
        watermark: Arc<AtomicU64>,
    ) -> Self {
        Self {
            worker_id,
            dp_rank,
            block_size,
            indexer,
            cancel,
            live_socket,
            replay_socket,
            watermark,
            warning_count: Arc::new(AtomicU32::new(0)),
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

        let Some(replay_socket) = self.replay_socket.as_ref() else {
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

        let req_frames = vec![Vec::new(), start_seq.to_be_bytes().to_vec()];
        if let Err(error) = send_multipart(replay_socket, req_frames).await {
            tracing::error!(worker_id, dp_rank, error = %error, "Failed to send replay request");
            return 0;
        }

        let mut replayed = 0u64;
        loop {
            let msg = tokio::select! {
                _ = self.cancel.cancelled() => break,
                result = recv_multipart(replay_socket) => {
                    match result {
                        Ok(msg) => msg,
                        Err(error) => {
                            tracing::error!(worker_id, dp_rank, error = %error, "Replay recv error");
                            break;
                        }
                    }
                }
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

    async fn handle_message(&mut self, msg: MultipartMessage) {
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

                result = recv_multipart(&self.live_socket) => {
                    match result {
                        Ok(msg) => msg,
                        Err(error) => {
                            return Err(format!(
                                "ZMQ recv failed for worker {} dp_rank {}: {error}",
                                self.worker_id,
                                self.dp_rank,
                            ));
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

    let socket = connect_sub_socket(&endpoint)
        .map_err(|e| format!("failed to connect ZMQ SUB socket to {endpoint}: {e}"))?;

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
) -> Option<SharedSocket> {
    let endpoint = replay_endpoint?;

    if cancel.is_cancelled() {
        return None;
    }

    match connect_dealer_socket(endpoint) {
        Ok(socket) => {
            tracing::info!(
                worker_id,
                dp_rank,
                replay_endpoint = endpoint,
                "Replay socket connected"
            );
            Some(socket)
        }
        Err(error) => {
            tracing::error!(
                worker_id,
                dp_rank,
                error = %error,
                "Failed to connect replay socket to {endpoint}"
            );
            None
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{WATERMARK_UNSET, cursor_from_watermark};
    use crate::recovery::CursorObservation;
    use crate::standalone_indexer::zmq::{
        SharedSocket, bind_pub_socket, connect_sub_socket, recv_multipart, send_multipart,
    };

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
        let reserved_listener = reserve_open_port();
        let endpoint = format!(
            "tcp://127.0.0.1:{}",
            reserved_listener
                .local_addr()
                .expect("failed to read reserved listener address")
                .port()
        );
        drop(reserved_listener);
        let pub_socket = bind_pub_socket(&endpoint).unwrap();
        let mut sub_socket = connect_sub_socket(&endpoint).unwrap();

        let (tx, mut rx) = tokio::sync::mpsc::channel::<SharedSocket>(1);
        tokio::spawn(async move {
            let _ = recv_multipart(&sub_socket).await.unwrap();
            let _ = tx.send(sub_socket).await;
        });
        loop {
            send_multipart(&pub_socket, vec![b"probe".to_vec()])
                .await
                .unwrap();
            tokio::time::sleep(std::time::Duration::from_millis(50)).await;
            if let Ok(sub) = rx.try_recv() {
                sub_socket = sub;
                break;
            }
        }

        let num_messages = 10u64;

        for i in 0..num_messages {
            send_multipart(&pub_socket, vec![i.to_le_bytes().to_vec()])
                .await
                .unwrap();
        }

        tokio::time::sleep(std::time::Duration::from_millis(500)).await;

        for i in 0u64..num_messages {
            let msg = tokio::time::timeout(
                std::time::Duration::from_secs(5),
                recv_multipart(&sub_socket),
            )
            .await
            .expect("timed out waiting for ZMQ message")
            .unwrap();

            let payload = msg.first().unwrap();
            let received = u64::from_le_bytes(payload[..8].try_into().unwrap());
            assert_eq!(received, i, "message {i} arrived out of order");
        }
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn zmq_subscriber_connects_before_publisher_bind() {
        let reserved_listener = reserve_open_port();
        let endpoint = format!(
            "tcp://127.0.0.1:{}",
            reserved_listener
                .local_addr()
                .expect("failed to read reserved listener address")
                .port()
        );
        drop(reserved_listener);
        let sub_socket = connect_sub_socket(&endpoint).unwrap();

        tokio::time::sleep(std::time::Duration::from_millis(100)).await;

        let pub_socket = bind_pub_socket(&endpoint).unwrap();
        for _ in 0..5 {
            send_multipart(&pub_socket, vec![b"probe".to_vec()])
                .await
                .unwrap();
            tokio::time::sleep(std::time::Duration::from_millis(50)).await;
        }

        let msg = tokio::time::timeout(
            std::time::Duration::from_secs(5),
            recv_multipart(&sub_socket),
        )
        .await
        .expect("timed out waiting for ZMQ message")
        .unwrap();

        assert_eq!(msg, vec![b"probe".to_vec()]);
    }

    fn reserve_open_port() -> std::net::TcpListener {
        std::net::TcpListener::bind("127.0.0.1:0").expect("failed to bind probe listener")
    }
}
