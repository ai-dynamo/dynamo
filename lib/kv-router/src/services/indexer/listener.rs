// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

use tokio::sync::watch;
use tokio_util::sync::CancellationToken;

use crate::protocols::{WorkerId, WorkerWithDpRank};
use crate::recovery::{CursorObservation, CursorState};
use crate::zmq_wire::{ZmqEventNormalizer, decode_event_batch};

use super::backend::Indexer;
use super::registry::ListenerRecord;
use crate::services::common::zmq::{
    MultipartMessage, SharedSocket, connect_dealer_socket, connect_sub_socket, recv_multipart,
    send_multipart, try_recv_multipart,
};

const WATERMARK_UNSET: u64 = u64::MAX;

fn cursor_from_watermark(watermark: u64) -> CursorState {
    if watermark == WATERMARK_UNSET {
        CursorState::Initial
    } else {
        CursorState::Live(watermark)
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum ReplayRecoveryFailureReason {
    Empty,
    StartedAfterRequested,
    NonContiguous,
    EndedBeforeTarget,
}

impl ReplayRecoveryFailureReason {
    fn as_str(self) -> &'static str {
        match self {
            Self::Empty => "empty",
            Self::StartedAfterRequested => "started_after_requested",
            Self::NonContiguous => "non_contiguous",
            Self::EndedBeforeTarget => "ended_before_target",
        }
    }
}

struct ReplayRecoveryProgress {
    start_seq: u64,
    end_seq: u64,
    first_replayed: Option<u64>,
    last_replayed: Option<u64>,
    expected_next: u64,
    replayed: u64,
    non_contiguous: bool,
}

impl ReplayRecoveryProgress {
    fn new(start_seq: u64, end_seq: u64) -> Self {
        Self {
            start_seq,
            end_seq,
            first_replayed: None,
            last_replayed: None,
            expected_next: start_seq,
            replayed: 0,
            non_contiguous: false,
        }
    }

    fn record_batch(&mut self, seq: u64) {
        if self.first_replayed.is_none() {
            self.first_replayed = Some(seq);
        }
        if seq != self.expected_next {
            self.non_contiguous = true;
        }
        self.expected_next = seq.saturating_add(1);
        self.last_replayed = Some(seq);
        self.replayed += 1;
    }

    fn failure_reason(&self) -> Option<ReplayRecoveryFailureReason> {
        if self.start_seq >= self.end_seq {
            return None;
        }
        if self.replayed == 0 {
            return Some(ReplayRecoveryFailureReason::Empty);
        }
        if self
            .first_replayed
            .is_some_and(|first| first > self.start_seq)
        {
            return Some(ReplayRecoveryFailureReason::StartedAfterRequested);
        }
        if self.non_contiguous {
            return Some(ReplayRecoveryFailureReason::NonContiguous);
        }
        if self
            .last_replayed
            .is_some_and(|last| last < self.end_seq - 1)
        {
            return Some(ReplayRecoveryFailureReason::EndedBeforeTarget);
        }
        None
    }

    fn replayed(&self) -> u64 {
        self.replayed
    }

    fn warn_if_incomplete(&self, worker_id: WorkerId, dp_rank: u32) {
        let Some(reason) = self.failure_reason() else {
            return;
        };

        let reason = reason.as_str();
        let start_seq = self.start_seq;
        let end_seq = self.end_seq;
        let replayed = self.replayed;
        let first_replayed_display = self
            .first_replayed
            .map(|seq| seq.to_string())
            .unwrap_or_else(|| "none".to_string());
        let last_replayed_display = self
            .last_replayed
            .map(|seq| seq.to_string())
            .unwrap_or_else(|| "none".to_string());
        tracing::warn!(
            worker_id,
            dp_rank,
            requested_start = start_seq,
            requested_end = end_seq,
            first_replayed = ?self.first_replayed,
            last_replayed = ?self.last_replayed,
            replayed,
            reason,
            "Replay incomplete: requested=[{start_seq},{end_seq}), first={}, last={}, replayed={replayed}, reason={reason}",
            first_replayed_display,
            last_replayed_display,
        );
    }
}

struct ListenerLoop {
    worker_id: WorkerId,
    dp_rank: u32,
    indexer: Indexer,
    cancel: CancellationToken,
    live_socket: SharedSocket,
    replay_socket: Option<SharedSocket>,
    watermark: Arc<AtomicU64>,
    snapshot_bootstrap_first_batch: Arc<std::sync::atomic::AtomicBool>,
    normalizer: ZmqEventNormalizer,
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
        snapshot_bootstrap_first_batch: Arc<std::sync::atomic::AtomicBool>,
    ) -> Self {
        Self {
            worker_id,
            dp_rank,
            indexer,
            cancel,
            live_socket,
            replay_socket,
            watermark,
            snapshot_bootstrap_first_batch,
            normalizer: ZmqEventNormalizer::new(block_size),
            messages_processed: 0,
        }
    }

    fn cursor(&self) -> CursorState {
        cursor_from_watermark(self.watermark.load(Ordering::Acquire))
    }

    async fn replay_gap(&mut self, start_seq: u64, end_seq: u64) -> Result<u64, String> {
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
            return Ok(0);
        };

        let worker_id = self.worker_id;
        let dp_rank = self.dp_rank;
        let indexer = &self.indexer;
        let watermark = &self.watermark;

        let req_frames = vec![Vec::new(), start_seq.to_be_bytes().to_vec()];
        if let Err(error) = send_multipart(replay_socket, req_frames).await {
            tracing::error!(worker_id, dp_rank, error = %error, "Failed to send replay request");
            return Ok(0);
        }

        let mut replay_progress = ReplayRecoveryProgress::new(start_seq, end_seq);
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
            if msg.len() != 4 {
                tracing::warn!(
                    worker_id,
                    dp_rank,
                    "Unexpected replay frame count: {}",
                    msg.len()
                );
                break;
            }

            // vLLM replay responses include the DEALER identity delimiter plus the
            // original PUB topic, so the payload shape is:
            // [empty delimiter, topic, sequence number, encoded event batch].
            let payload = msg.get(3).expect("frame count checked above");
            if payload.is_empty() {
                break;
            }

            let seq_bytes = msg.get(2).expect("frame count checked above");
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

            let Ok(batch) = decode_event_batch(payload) else {
                tracing::warn!(worker_id, dp_rank, seq, "Failed to decode replayed batch");
                continue;
            };

            let effective_dp_rank = batch
                .data_parallel_rank
                .map_or(dp_rank, |rank| rank.cast_unsigned());
            for raw_event in batch.events {
                let Some(placement_event) = self.normalizer.normalize(
                    raw_event,
                    seq,
                    WorkerWithDpRank::new(worker_id, effective_dp_rank),
                ) else {
                    continue;
                };
                let router_event = placement_event
                    .into_router_event()
                    .expect("local worker placement must convert to router event");
                indexer
                    .apply_event_routed(router_event)
                    .await
                    .map_err(|error| {
                        format!(
                            "failed to apply replayed event for worker {worker_id} dp_rank {dp_rank}: {error}"
                        )
                    })?;
            }
            watermark.store(seq, Ordering::Release);
            replay_progress.record_batch(seq);
        }

        replay_progress.warn_if_incomplete(worker_id, dp_rank);

        let replayed = replay_progress.replayed();
        tracing::info!(worker_id, dp_rank, replayed, "Replay complete");
        Ok(replayed)
    }

    async fn handle_gap(&mut self, seq: u64) -> Result<(), String> {
        let observation = self.cursor().observe(seq);
        if matches!(observation, CursorObservation::Initial { .. })
            && self
                .snapshot_bootstrap_first_batch
                .swap(false, Ordering::AcqRel)
        {
            tracing::debug!(
                self.worker_id,
                self.dp_rank,
                seq,
                "Establishing live cursor from first batch after peer snapshot"
            );
            return Ok(());
        }

        match observation {
            CursorObservation::Initial { got } if got > 0 => {
                tracing::warn!(
                    self.worker_id,
                    self.dp_rank,
                    expected = 0,
                    got,
                    "Gap detected: expected seq 0, got {got}"
                );
                self.replay_gap(0, got).await?;
            }
            CursorObservation::Gap { expected, got } => {
                tracing::warn!(
                    self.worker_id,
                    self.dp_rank,
                    expected,
                    got,
                    "Gap detected: expected seq {expected}, got {got}"
                );
                self.replay_gap(expected, got).await?;
            }
            CursorObservation::Initial { .. }
            | CursorObservation::Contiguous { .. }
            | CursorObservation::Stale { .. } => {}
        }
        Ok(())
    }

    async fn apply_live_batch(&mut self, seq: u64, payload: &[u8]) -> Result<(), String> {
        let batch = match decode_event_batch(payload) {
            Ok(batch) => batch,
            Err(error) => {
                tracing::warn!(
                    self.worker_id,
                    self.dp_rank,
                    "Failed to decode KvEventBatch: {error}"
                );
                return Ok(());
            }
        };

        let effective_dp_rank = batch
            .data_parallel_rank
            .map_or(self.dp_rank, |rank| rank.cast_unsigned());
        for raw_event in batch.events {
            let Some(placement_event) = self.normalizer.normalize(
                raw_event,
                seq,
                WorkerWithDpRank::new(self.worker_id, effective_dp_rank),
            ) else {
                continue;
            };
            let router_event = placement_event
                .into_router_event()
                .expect("local worker placement must convert to router event");
            self.indexer
                .apply_event_routed(router_event)
                .await
                .map_err(|error| {
                    format!(
                        "failed to apply live event for worker {} dp_rank {}: {error}",
                        self.worker_id, self.dp_rank
                    )
                })?;
            self.messages_processed += 1;
        }
        self.watermark.store(seq, Ordering::Release);
        Ok(())
    }

    async fn handle_message(&mut self, msg: MultipartMessage) -> Result<(), String> {
        if msg.len() != 3 {
            tracing::warn!(
                self.worker_id,
                self.dp_rank,
                "Unexpected ZMQ frame count: {}",
                msg.len()
            );
            return Ok(());
        }

        let seq_bytes = msg.get(1).expect("frame count checked above");
        if seq_bytes.len() != 8 {
            tracing::warn!(
                self.worker_id,
                self.dp_rank,
                "Invalid sequence number length: {}",
                seq_bytes.len()
            );
            return Ok(());
        }

        let seq = u64::from_be_bytes(seq_bytes[..8].try_into().expect("length checked above"));
        self.handle_gap(seq).await?;

        if matches!(self.cursor().observe(seq), CursorObservation::Stale { .. }) {
            return Ok(());
        }

        let payload = msg.get(2).expect("frame count checked above");
        self.apply_live_batch(seq, payload).await
    }

    async fn drain_buffered(&mut self) -> Result<(), String> {
        while let Some(message) = try_recv_multipart(&self.live_socket)
            .await
            .map_err(|error| format!("ZMQ buffered recv failed: {error}"))?
        {
            self.handle_message(message).await?;
        }
        Ok(())
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

            self.handle_message(msg).await?;
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
    let snapshot_bootstrap_first_batch = record.snapshot_bootstrap_first_batch();

    tracing::info!(worker_id, dp_rank, endpoint, "ZMQ listener starting");

    if cancel.is_cancelled() {
        return Ok(());
    }

    let socket = connect_sub_socket(&endpoint)
        .map_err(|e| format!("failed to connect ZMQ SUB socket to {endpoint}: {e}"))?;

    if !record.try_mark_buffering(generation) {
        return Ok(());
    }

    tokio::select! {
        _ = cancel.cancelled() => return Ok(()),
        result = ready.wait_for(|&value| value) => {
            result.map_err(|_| "ready channel closed before signaling".to_string())?;
        }
    }

    let replay_socket =
        connect_replay_socket(worker_id, dp_rank, replay_endpoint.as_deref(), &cancel).await;
    if cancel.is_cancelled() || !record.is_current_attempt(generation) {
        return Ok(());
    }

    if !record.try_mark_catching_up(generation) {
        tracing::debug!(
            worker_id,
            dp_rank,
            "Listener attempt is stale before catch-up; exiting"
        );
        return Ok(());
    }

    let mut listener = ListenerLoop::new(
        worker_id,
        dp_rank,
        block_size,
        indexer,
        cancel,
        socket,
        replay_socket,
        watermark,
        snapshot_bootstrap_first_batch,
    );
    listener.drain_buffered().await?;
    if !record.try_mark_active(generation) {
        return Ok(());
    }

    tracing::info!(worker_id, dp_rank, "ZMQ listener ready, starting recv loop");
    listener.run().await
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
