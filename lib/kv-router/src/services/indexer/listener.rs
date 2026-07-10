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
    ) -> Self {
        Self {
            worker_id,
            dp_rank,
            indexer,
            cancel,
            live_socket,
            replay_socket,
            watermark,
            normalizer: ZmqEventNormalizer::new(block_size),
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
        let indexer = &self.indexer;
        let watermark = &self.watermark;

        let req_frames = vec![Vec::new(), start_seq.to_be_bytes().to_vec()];
        if let Err(error) = send_multipart(replay_socket, req_frames).await {
            tracing::error!(worker_id, dp_rank, error = %error, "Failed to send replay request");
            return 0;
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
                let Some(router_event) = placement_event.into_router_event() else {
                    continue;
                };
                indexer.apply_event_routed(router_event).await;
            }
            watermark.store(seq, Ordering::Release);
            replay_progress.record_batch(seq);
        }

        replay_progress.warn_if_incomplete(worker_id, dp_rank);

        let replayed = replay_progress.replayed();
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
        let batch = match decode_event_batch(payload) {
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
            let Some(placement_event) = self.normalizer.normalize(
                raw_event,
                seq,
                WorkerWithDpRank::new(self.worker_id, effective_dp_rank),
            ) else {
                continue;
            };
            let Some(router_event) = placement_event.into_router_event() else {
                continue;
            };
            self.indexer.apply_event_routed(router_event).await;
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
    use std::time::Duration;

    use super::*;
    use crate::protocols::KvCacheEventData;
    use crate::services::indexer::backend::create_indexer;
    use crate::zmq_wire::{BlockHashValue, Locality, RawKvEvent};

    const WORKER_ID: WorkerId = 7;
    const BLOCK_SIZE: u32 = 2;
    const LOCAL_BLOCK_HASH: u64 = 0xA1;
    const SHARED_FS_BLOCK_HASH: u64 = 0xB2;
    const SHARED_GPU_BLOCK_HASH: u64 = 0xC3;

    /// Reserve an OS-assigned TCP port by binding+dropping a listener
    /// (mirrors `tests/standalone_indexer_http.rs`).
    fn reserve_zmq_endpoint() -> String {
        let listener = std::net::TcpListener::bind("127.0.0.1:0").expect("bind probe listener");
        let port = listener
            .local_addr()
            .expect("local_addr on probe listener")
            .port();
        drop(listener);
        format!("tcp://127.0.0.1:{port}")
    }

    fn raw_block_stored(
        block_hash: u64,
        token_ids: Vec<u32>,
        medium: &str,
        locality: Option<Locality>,
    ) -> RawKvEvent {
        RawKvEvent::BlockStored {
            block_hashes: vec![BlockHashValue::Unsigned(block_hash)],
            parent_block_hash: None,
            token_ids,
            block_size: BLOCK_SIZE as usize,
            medium: Some(medium.to_string()),
            lora_name: None,
            cache_namespace: None,
            block_mm_infos: None,
            is_eagle: None,
            group_idx: None,
            kv_cache_spec_kind: None,
            kv_cache_spec_sliding_window: None,
            locality,
        }
    }

    /// One-batch payload mixing shared and local placements. The shared
    /// FS+REMOTE event comes first so the event loop must skip past it and
    /// still index the local event behind it.
    fn mixed_placement_payload() -> Vec<u8> {
        let events = vec![
            raw_block_stored(
                SHARED_FS_BLOCK_HASH,
                vec![20, 21],
                "FS",
                Some(Locality::Remote),
            ),
            raw_block_stored(LOCAL_BLOCK_HASH, vec![10, 11], "GPU", None),
            raw_block_stored(
                SHARED_GPU_BLOCK_HASH,
                vec![30, 31],
                "GPU",
                Some(Locality::Remote),
            ),
        ];
        rmp_serde::to_vec_named(&(0.0_f64, events, Some(0_i32))).expect("serialize KvEventBatch")
    }

    fn listener_loop(indexer: Indexer, replay_socket: Option<SharedSocket>) -> ListenerLoop {
        ListenerLoop::new(
            WORKER_ID,
            0,
            BLOCK_SIZE,
            indexer,
            CancellationToken::new(),
            connect_sub_socket(&reserve_zmq_endpoint()).expect("connect live SUB socket"),
            replay_socket,
            Arc::new(AtomicU64::new(WATERMARK_UNSET)),
        )
    }

    /// External block hashes of every stored block across all tiers.
    /// `Indexer::dump_events` drains pending events first, so this doubles as
    /// the FIFO barrier the other indexer tests rely on, and it covers the
    /// device primary plus every allocated lower-tier indexer.
    async fn stored_block_hashes(indexer: &Indexer) -> Vec<u64> {
        let mut hashes: Vec<u64> = indexer
            .dump_events()
            .await
            .expect("dump events")
            .iter()
            .filter_map(|event| match &event.event.data {
                KvCacheEventData::Stored(store) => Some(&store.blocks),
                _ => None,
            })
            .flatten()
            .map(|block| block.block_hash.0)
            .collect();
        hashes.sort_unstable();
        hashes
    }

    /// Live path: a batch mixing shared (FS+REMOTE, GPU+REMOTE) and local
    /// (GPU, no locality) events must index only the local block. Shared
    /// placements yield `into_router_event() == None` and must be skipped,
    /// not panic the listener (this used to be an `.expect(...)`).
    #[tokio::test]
    async fn live_batch_skips_shared_placements_and_indexes_local_block() {
        let indexer = create_indexer(BLOCK_SIZE, 1);
        let mut listener = listener_loop(indexer.clone(), None);

        listener
            .apply_live_batch(5, &mixed_placement_payload())
            .await;

        assert_eq!(
            stored_block_hashes(&indexer).await,
            vec![LOCAL_BLOCK_HASH],
            "only the local GPU block may reach the index"
        );
        assert_eq!(
            listener.messages_processed, 1,
            "shared events must be skipped, not counted as processed"
        );
        assert_eq!(listener.watermark.load(Ordering::Acquire), 5);
    }

    /// Replay path: replayed batches flow through the same fail-closed gate.
    /// A replayed batch containing shared events must still complete the
    /// replay (batch counted, watermark advanced) while indexing only the
    /// local block.
    #[tokio::test]
    async fn replayed_batch_skips_shared_placements_and_indexes_local_block() {
        let replay_endpoint = reserve_zmq_endpoint();
        let router = zmq::Context::new()
            .socket(zmq::ROUTER)
            .expect("create ROUTER socket");
        router.set_linger(0).expect("set_linger");
        router.set_rcvtimeo(10_000).expect("set_rcvtimeo");
        router.bind(&replay_endpoint).expect("bind ROUTER socket");

        let payload = mixed_placement_payload();
        let server = std::thread::spawn(move || {
            // Engine side of the replay protocol: answer the DEALER's
            // `[empty, start_seq]` request with one batch at seq 0, then an
            // empty-payload terminator.
            let request = router.recv_multipart(0).expect("recv replay request");
            let identity = request[0].as_slice();
            let seq0 = 0_u64.to_be_bytes();
            let batch: [&[u8]; 4] = [identity, b"", &seq0, payload.as_slice()];
            router.send_multipart(batch, 0).expect("send replay batch");
            let end_seq = 1_u64.to_be_bytes();
            let terminator: [&[u8]; 4] = [identity, b"", &end_seq, b""];
            router
                .send_multipart(terminator, 0)
                .expect("send replay terminator");
        });

        let indexer = create_indexer(BLOCK_SIZE, 1);
        let replay_socket =
            connect_dealer_socket(&replay_endpoint).expect("connect replay DEALER socket");
        let mut listener = listener_loop(indexer.clone(), Some(replay_socket));

        let replayed = tokio::time::timeout(Duration::from_secs(10), listener.replay_gap(0, 1))
            .await
            .expect("replay_gap timed out");

        assert_eq!(
            replayed, 1,
            "shared events must not abort the replayed batch"
        );
        assert_eq!(
            stored_block_hashes(&indexer).await,
            vec![LOCAL_BLOCK_HASH],
            "only the local GPU block may reach the index from replay"
        );
        assert_eq!(listener.watermark.load(Ordering::Acquire), 0);
        server.join().expect("replay server thread");
    }
}
