// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Raw-UDS `AsyncShardHandle` implementation.
//!
//! [`RawUdsShardClient`] connects to a [`RawUdsShardServer`] over a Unix Domain
//! Socket and implements [`AsyncShardHandle`] without any Velo / NATS middleware.
//!
//! ## Connection model
//!
//! One persistent `UnixStream` per `RawUdsShardClient`.  A background reader
//! task drains incoming [`ShardResponse`] frames and routes each one to the
//! waiting `oneshot::Sender` keyed by `req_id`.
//!
//! ## Write ordering
//!
//! All outbound frames — both fire-and-forget writes and RPC requests — travel
//! through a single `mpsc::UnboundedSender<Vec<u8>>` drained by one writer
//! task.  FIFO ordering is therefore guaranteed: an `EnqueueAnchor` sent before
//! an `ApplyEvent` for the same worker will always arrive first.
//!
//! ## `enqueue_anchor` sync constraint
//!
//! The [`AsyncShardHandle`] contract requires `enqueue_anchor` to be sync
//! (returns `Result<(), KvRouterError>`, not a future).  This is satisfied
//! because the underlying channel `send()` is non-blocking.

use std::path::PathBuf;
use std::sync::Arc;
use std::sync::atomic::{AtomicU32, Ordering};

use dashmap::DashMap;
use tokio::io::{AsyncWriteExt, BufReader};
use tokio::net::UnixStream;
use tokio::sync::{mpsc, oneshot};
use tracing::{error, warn};

use crate::indexer::{AnchorRef, AnchorTask, AsyncShardHandle, KvRouterError, ShardSizeSnapshot};
use crate::protocols::{
    DpRank, LocalBlockHash, OverlapScores, RouterEvent, WorkerId, WorkerWithDpRank,
};

use super::wire::{ShardRequest, ShardResponse, encode_msg, read_msg};

// ---------------------------------------------------------------------------
// Pending RPC map
// ---------------------------------------------------------------------------

/// Dispatch target for a single in-flight RPC.
enum PendingRpc {
    FindMatchesFromAnchor(oneshot::Sender<Result<OverlapScores, KvRouterError>>),
    DumpEvents(oneshot::Sender<Result<Vec<RouterEvent>, KvRouterError>>),
    ShardSizes(oneshot::Sender<ShardSizeSnapshot>),
    Flush(oneshot::Sender<usize>),
}

// ---------------------------------------------------------------------------
// RawUdsShardClient
// ---------------------------------------------------------------------------

/// [`AsyncShardHandle`] that communicates with a remote shard over a raw UDS.
///
/// # Cloning
///
/// `RawUdsShardClient` wraps its state in an `Arc` — cloning is cheap and all
/// clones share the same underlying connection.
#[derive(Clone)]
pub struct RawUdsShardClient {
    inner: Arc<ClientInner>,
}

struct ClientInner {
    /// Encodes and enqueues outbound frames (FIFO).
    write_tx: mpsc::UnboundedSender<Vec<u8>>,
    /// Monotonically-increasing request-id counter.
    next_req_id: AtomicU32,
    /// In-flight RPCs waiting for a server response.
    /// Wrapped in `Arc` so it can be shared with the background reader task.
    pending: Arc<DashMap<u32, PendingRpc>>,
}

impl RawUdsShardClient {
    /// Connect to a `RawUdsShardServer` at `socket_path`.
    ///
    /// Spawns two background tasks (writer + reader) that run until the
    /// connection drops or [`shutdown`] is called.
    pub async fn connect(socket_path: impl Into<PathBuf>) -> anyhow::Result<Self> {
        let path = socket_path.into();
        let stream = UnixStream::connect(&path).await?;
        let (read_half, mut write_half) = stream.into_split();

        let (write_tx, mut write_rx) = mpsc::unbounded_channel::<Vec<u8>>();
        let pending: Arc<DashMap<u32, PendingRpc>> = Arc::new(DashMap::new());

        // ── writer task ──────────────────────────────────────────────────
        tokio::spawn(async move {
            while let Some(frame) = write_rx.recv().await {
                if let Err(e) = write_half.write_all(&frame).await {
                    error!(path = %path.display(), error = %e, "shard client writer error");
                    break;
                }
            }
        });

        // ── reader task ──────────────────────────────────────────────────
        let pending_reader = Arc::clone(&pending);
        tokio::spawn(async move {
            let mut reader = BufReader::new(read_half);
            loop {
                match read_msg::<ShardResponse, _>(&mut reader).await {
                    Ok(Some(resp)) => {
                        dispatch_response(resp, &pending_reader);
                    }
                    Ok(None) => {
                        // server closed connection
                        break;
                    }
                    Err(e) => {
                        error!(error = %e, "shard client reader error");
                        break;
                    }
                }
            }
            // Drain pending RPCs with an error so callers don't hang.
            drain_pending_with_error(&pending_reader);
        });

        Ok(Self {
            inner: Arc::new(ClientInner {
                write_tx,
                next_req_id: AtomicU32::new(1),
                pending,
            }),
        })
    }

    fn next_req_id(&self) -> u32 {
        self.inner.next_req_id.fetch_add(1, Ordering::Relaxed)
    }

    fn send_frame(&self, frame: Vec<u8>) -> Result<(), KvRouterError> {
        self.inner
            .write_tx
            .send(frame)
            .map_err(|_| KvRouterError::IndexerOffline)
    }

    fn send_msg<T: serde::Serialize>(&self, msg: &T) -> Result<(), KvRouterError> {
        let frame = encode_msg(msg).map_err(|_| KvRouterError::IndexerOffline)?;
        self.send_frame(frame)
    }

    /// Insert a pending RPC slot, send the request, and clean up the slot on
    /// send failure so the entry never leaks.
    fn send_rpc<T: serde::Serialize>(
        &self,
        req_id: u32,
        slot: PendingRpc,
        msg: &T,
    ) -> Result<(), KvRouterError> {
        self.inner.pending.insert(req_id, slot);
        match self.send_msg(msg) {
            Ok(()) => Ok(()),
            Err(e) => {
                self.inner.pending.remove(&req_id);
                Err(e)
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Response dispatcher
// ---------------------------------------------------------------------------

fn dispatch_response(resp: ShardResponse, pending: &Arc<DashMap<u32, PendingRpc>>) {
    let req_id = resp.req_id();
    let Some((_, slot)) = pending.remove(&req_id) else {
        warn!(req_id, "received response for unknown req_id");
        return;
    };
    match (resp, slot) {
        (
            ShardResponse::FindMatchesFromAnchor { result, .. },
            PendingRpc::FindMatchesFromAnchor(tx),
        ) => {
            let r = result
                .map(OverlapScores::from)
                .map_err(|e| KvRouterError::Unsupported(e));
            let _ = tx.send(r);
        }
        (ShardResponse::DumpEvents { result, .. }, PendingRpc::DumpEvents(tx)) => {
            let r = result.map_err(|e| KvRouterError::Unsupported(e));
            let _ = tx.send(r);
        }
        (ShardResponse::ShardSizes { snapshot, .. }, PendingRpc::ShardSizes(tx)) => {
            let _ = tx.send(snapshot);
        }
        (ShardResponse::Flush { queue_depth, .. }, PendingRpc::Flush(tx)) => {
            let _ = tx.send(queue_depth);
        }
        _ => {
            warn!(req_id, "response/pending type mismatch for req_id={req_id}");
        }
    }
}

fn drain_pending_with_error(pending: &Arc<DashMap<u32, PendingRpc>>) {
    // Dropping all entries closes each oneshot::Sender, causing receivers to
    // wake with `RecvError` — callers mapped to `KvRouterError::IndexerOffline`.
    pending.clear();
}

// ---------------------------------------------------------------------------
// AsyncShardHandle impl
// ---------------------------------------------------------------------------

impl AsyncShardHandle for RawUdsShardClient {
    async fn apply_event(&self, event: RouterEvent) {
        let req = ShardRequest::ApplyEvent(event);
        if let Err(e) = self.send_msg(&req) {
            warn!(error = %e, "apply_event: failed to send frame");
        }
    }

    fn enqueue_anchor(
        &self,
        worker: WorkerWithDpRank,
        anchor: AnchorTask,
    ) -> Result<(), KvRouterError> {
        let req = ShardRequest::EnqueueAnchor { worker, anchor };
        self.send_msg(&req)
    }

    async fn find_matches_from_anchor(
        &self,
        anchor: AnchorRef,
        suffix: Vec<LocalBlockHash>,
    ) -> Result<OverlapScores, KvRouterError> {
        let req_id = self.next_req_id();
        let (tx, rx) = oneshot::channel();
        let req = ShardRequest::FindMatchesFromAnchor {
            req_id,
            anchor,
            suffix,
        };
        self.send_rpc(req_id, PendingRpc::FindMatchesFromAnchor(tx), &req)?;
        rx.await.map_err(|_| KvRouterError::IndexerOffline)?
    }

    async fn remove_worker(&self, worker_id: WorkerId) {
        let req = ShardRequest::RemoveWorker(worker_id);
        if let Err(e) = self.send_msg(&req) {
            warn!(error = %e, "remove_worker: failed to send frame");
        }
    }

    async fn remove_worker_dp_rank(&self, worker_id: WorkerId, dp_rank: DpRank) {
        let req = ShardRequest::RemoveWorkerDpRank(worker_id, dp_rank);
        if let Err(e) = self.send_msg(&req) {
            warn!(error = %e, "remove_worker_dp_rank: failed to send frame");
        }
    }

    async fn dump_events(&self) -> Result<Vec<RouterEvent>, KvRouterError> {
        let req_id = self.next_req_id();
        let (tx, rx) = oneshot::channel();
        let req = ShardRequest::DumpEvents { req_id };
        self.send_rpc(req_id, PendingRpc::DumpEvents(tx), &req)?;
        rx.await.map_err(|_| KvRouterError::IndexerOffline)?
    }

    async fn shard_sizes(&self) -> ShardSizeSnapshot {
        let req_id = self.next_req_id();
        let (tx, rx) = oneshot::channel();
        let req = ShardRequest::ShardSizes { req_id };
        if self
            .send_rpc(req_id, PendingRpc::ShardSizes(tx), &req)
            .is_err()
        {
            return ShardSizeSnapshot {
                shard_idx: 0,
                worker_count: 0,
                block_count: 0,
                node_count: 0,
            };
        }
        rx.await.unwrap_or(ShardSizeSnapshot {
            shard_idx: 0,
            worker_count: 0,
            block_count: 0,
            node_count: 0,
        })
    }

    async fn flush(&self) -> usize {
        let req_id = self.next_req_id();
        let (tx, rx) = oneshot::channel();
        let req = ShardRequest::Flush { req_id };
        if self.send_rpc(req_id, PendingRpc::Flush(tx), &req).is_err() {
            return 0;
        }
        rx.await.unwrap_or(0)
    }

    fn shutdown(&self) {
        let req = ShardRequest::Shutdown;
        if let Err(e) = self.send_msg(&req) {
            warn!(error = %e, "shutdown: failed to send frame");
        }
    }

    fn node_edge_lengths(&self) -> Vec<usize> {
        // Remote shards don't expose trie internals over the wire.
        Vec::new()
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::indexer::ThreadPoolIndexer;
    use crate::indexer::concurrent_radix_tree_compressed::ConcurrentRadixTreeCompressed;
    use crate::protocols::{KvCacheEvent, KvCacheEventData};
    use crate::shard_router::uds_raw_server::RawUdsShardServer;
    use tempfile::TempDir;

    type TestShard = ThreadPoolIndexer<ConcurrentRadixTreeCompressed>;

    fn make_shard() -> TestShard {
        ThreadPoolIndexer::new(ConcurrentRadixTreeCompressed::new(), 2, 32)
    }

    async fn start_server_and_client(
        dir: &TempDir,
    ) -> (RawUdsShardServer<TestShard>, RawUdsShardClient) {
        let socket_path = dir.path().join("test.sock");
        let shard = make_shard();
        let server = RawUdsShardServer::bind(socket_path.clone(), shard)
            .await
            .expect("bind failed");
        let client = RawUdsShardClient::connect(socket_path)
            .await
            .expect("connect failed");
        (server, client)
    }

    fn cleared_event(worker_id: WorkerId) -> RouterEvent {
        RouterEvent::new(
            worker_id,
            KvCacheEvent {
                event_id: 0,
                data: KvCacheEventData::Cleared,
                dp_rank: 0,
            },
        )
    }

    /// `flush` on an idle shard returns 0.
    #[tokio::test]
    async fn flush_returns_zero_on_idle_shard() {
        let dir = TempDir::new().unwrap();
        let (_server, client) = start_server_and_client(&dir).await;
        let depth = client.flush().await;
        assert_eq!(depth, 0);
    }

    /// Fire-and-forget `apply_event` + `flush` round-trip completes without error.
    #[tokio::test]
    async fn apply_event_and_flush() {
        let dir = TempDir::new().unwrap();
        let (_server, client) = start_server_and_client(&dir).await;
        client.apply_event(cleared_event(1)).await;
        let depth = client.flush().await;
        // flush queues the Cleared event, depth is non-deterministic but non-error
        let _ = depth;
    }

    /// `dump_events` on a fresh shard returns an empty vec.
    #[tokio::test]
    async fn dump_events_empty_on_fresh_shard() {
        let dir = TempDir::new().unwrap();
        let (_server, client) = start_server_and_client(&dir).await;
        client.flush().await;
        let events = client.dump_events().await.unwrap();
        assert!(events.is_empty(), "fresh shard should have no events");
    }

    /// `shard_sizes` returns a valid snapshot.
    #[tokio::test]
    async fn shard_sizes_returns_snapshot() {
        let dir = TempDir::new().unwrap();
        let (_server, client) = start_server_and_client(&dir).await;
        let snap = client.shard_sizes().await;
        // Just verify it didn't panic and returns a coherent (zero) snapshot.
        assert_eq!(snap.worker_count, 0);
    }
}
