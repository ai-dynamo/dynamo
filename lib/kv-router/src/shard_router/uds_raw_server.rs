// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Raw-UDS shard server.
//!
//! [`RawUdsShardServer`] binds a Unix Domain Socket, accepts one connection at
//! a time, and dispatches incoming [`ShardRequest`] frames to an underlying
//! `S: AsyncShardHandle`.  Response frames are sent back for RPC variants;
//! fire-and-forget variants are processed and no reply is sent.
//!
//! ## Single-connection model
//!
//! The server accepts a single connection (from the matching
//! [`RawUdsShardClient`]).  If that connection closes, the server loops back
//! to `accept()` and waits for a new one.  This matches the benchmark scenario
//! where one router process connects to one shard-server process.

use std::path::PathBuf;
use std::sync::Arc;

use tokio::io::BufReader;
use tokio::net::{UnixListener, UnixStream};
use tracing::{error, info, warn};

use crate::indexer::AsyncShardHandle;

use super::wire::{ShardRequest, ShardResponse, WireOverlapScores, read_msg, write_msg};

// ---------------------------------------------------------------------------
// RawUdsShardServer
// ---------------------------------------------------------------------------

/// Listens on a UDS socket and dispatches requests to `shard`.
///
/// When the handle is dropped the accept-loop task is **aborted** (via
/// `JoinHandle::abort`) and the socket file is removed.  Any connection
/// currently being processed will be interrupted at the next `.await` point.
pub struct RawUdsShardServer<S> {
    /// Shared reference to the underlying shard (also held by background task).
    _shard: Arc<S>,
    /// Background accept-loop task handle; aborted on drop.
    task: tokio::task::JoinHandle<()>,
    /// Path to the bound socket (removed on drop via guard).
    _socket_guard: SocketPathGuard,
}

impl<S> Drop for RawUdsShardServer<S> {
    fn drop(&mut self) {
        self.task.abort();
    }
}

impl<S: AsyncShardHandle> RawUdsShardServer<S> {
    /// Bind `socket_path` and start the accept loop.
    ///
    /// The path is removed from the filesystem when the returned
    /// `RawUdsShardServer` is dropped.
    pub async fn bind(socket_path: impl Into<PathBuf>, shard: S) -> anyhow::Result<Self> {
        let path = socket_path.into();
        // Remove stale socket if present.
        let _ = std::fs::remove_file(&path);
        let listener = UnixListener::bind(&path)?;
        let shard = Arc::new(shard);
        let shard_task = Arc::clone(&shard);
        let task = tokio::spawn(accept_loop(listener, shard_task));
        Ok(Self {
            _shard: shard,
            task,
            _socket_guard: SocketPathGuard(path),
        })
    }
}

// ---------------------------------------------------------------------------
// Accept loop
// ---------------------------------------------------------------------------

async fn accept_loop<S: AsyncShardHandle>(listener: UnixListener, shard: Arc<S>) {
    loop {
        match listener.accept().await {
            Ok((stream, _addr)) => {
                info!("shard server: accepted new connection");
                handle_connection(stream, Arc::clone(&shard)).await;
                info!("shard server: connection closed, waiting for next");
            }
            Err(e) => {
                error!(error = %e, "shard server: accept error");
                // Brief pause to avoid spinning on persistent errors.
                tokio::time::sleep(std::time::Duration::from_millis(10)).await;
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Connection handler
// ---------------------------------------------------------------------------

async fn handle_connection<S: AsyncShardHandle>(stream: UnixStream, shard: Arc<S>) {
    let (read_half, write_half) = stream.into_split();
    let mut reader = BufReader::new(read_half);
    // Writer is wrapped in a Mutex so the async handler closures can share it.
    // In practice only one request is in flight per read iteration, so
    // contention is zero.  We use a simple `tokio::sync::Mutex` here.
    let writer = Arc::new(tokio::sync::Mutex::new(write_half));

    loop {
        match read_msg::<ShardRequest, _>(&mut reader).await {
            Ok(Some(req)) => {
                let shard = Arc::clone(&shard);
                let writer = Arc::clone(&writer);
                // Process each request inline (not spawned) to preserve ordering
                // for fire-and-forget writes.  This serialises all requests on
                // the single connection, making it intentionally minimal: one
                // in-flight RPC at a time.  That is the head-of-line bottleneck
                // visible in benchmark results.  Pipelining or per-request task
                // spawning would reduce it but is out of scope for this baseline.
                process_request(req, shard, writer).await;
            }
            Ok(None) => break, // client disconnected cleanly
            Err(e) => {
                warn!(error = %e, "shard server: frame read error");
                break;
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Request processor
// ---------------------------------------------------------------------------

type LockedWriter = Arc<tokio::sync::Mutex<tokio::net::unix::OwnedWriteHalf>>;

async fn process_request<S: AsyncShardHandle>(
    req: ShardRequest,
    shard: Arc<S>,
    writer: LockedWriter,
) {
    match req {
        // ── fire-and-forget ──────────────────────────────────────────────
        ShardRequest::ApplyEvent(event) => {
            shard.apply_event(event).await;
        }
        ShardRequest::EnqueueAnchor { worker, anchor } => {
            if let Err(e) = shard.enqueue_anchor(worker, anchor) {
                warn!(error = %e, "shard server: enqueue_anchor error");
            }
        }
        ShardRequest::RemoveWorker(worker_id) => {
            shard.remove_worker(worker_id).await;
        }
        ShardRequest::RemoveWorkerDpRank(worker_id, dp_rank) => {
            shard.remove_worker_dp_rank(worker_id, dp_rank).await;
        }
        ShardRequest::Shutdown => {
            shard.shutdown();
        }

        // ── request-response ────────────────────────────────────────────
        ShardRequest::FindMatchesFromAnchor {
            req_id,
            anchor,
            suffix,
        } => {
            let result = shard
                .find_matches_from_anchor(anchor, suffix)
                .await
                .map(WireOverlapScores::from)
                .map_err(|e| e.to_string());
            let resp = ShardResponse::FindMatchesFromAnchor { req_id, result };
            send_response(&resp, &writer).await;
        }
        ShardRequest::DumpEvents { req_id } => {
            let result = shard.dump_events().await.map_err(|e| e.to_string());
            let resp = ShardResponse::DumpEvents { req_id, result };
            send_response(&resp, &writer).await;
        }
        ShardRequest::ShardSizes { req_id } => {
            let snapshot = shard.shard_sizes().await;
            let resp = ShardResponse::ShardSizes { req_id, snapshot };
            send_response(&resp, &writer).await;
        }
        ShardRequest::Flush { req_id } => {
            let queue_depth = shard.flush().await;
            let resp = ShardResponse::Flush {
                req_id,
                queue_depth,
            };
            send_response(&resp, &writer).await;
        }
    }
}

async fn send_response(resp: &ShardResponse, writer: &LockedWriter) {
    let mut guard = writer.lock().await;
    if let Err(e) = write_msg(&mut *guard, resp).await {
        warn!(error = %e, "shard server: failed to send response");
    }
}

// ---------------------------------------------------------------------------
// Socket cleanup guard
// ---------------------------------------------------------------------------

struct SocketPathGuard(PathBuf);

impl Drop for SocketPathGuard {
    fn drop(&mut self) {
        let _ = std::fs::remove_file(&self.0);
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::indexer::concurrent_radix_tree_compressed::ConcurrentRadixTreeCompressed;
    use crate::indexer::{ShardSizeSnapshot, ThreadPoolIndexer};
    use crate::protocols::{KvCacheEvent, KvCacheEventData, RouterEvent};
    use crate::shard_router::wire::{ShardRequest, ShardResponse, read_msg, write_msg};
    use tempfile::TempDir;
    use tokio::net::UnixStream;

    type TestShard = ThreadPoolIndexer<ConcurrentRadixTreeCompressed>;

    fn make_shard() -> TestShard {
        ThreadPoolIndexer::new(ConcurrentRadixTreeCompressed::new(), 2, 32)
    }

    fn cleared_event(worker_id: u64) -> RouterEvent {
        RouterEvent::new(
            worker_id,
            KvCacheEvent {
                event_id: 0,
                data: KvCacheEventData::Cleared,
                dp_rank: 0,
            },
        )
    }

    async fn connect_raw(
        path: &PathBuf,
    ) -> (
        tokio::io::ReadHalf<UnixStream>,
        tokio::io::WriteHalf<UnixStream>,
    ) {
        let stream = UnixStream::connect(path).await.expect("connect failed");
        tokio::io::split(stream)
    }

    /// Server binds successfully and socket file is created.
    #[tokio::test]
    async fn server_binds_creates_socket() {
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("bind_test.sock");
        let _server = RawUdsShardServer::bind(path.clone(), make_shard())
            .await
            .unwrap();
        assert!(path.exists(), "socket file should exist after bind");
    }

    /// Socket file is removed when the server is dropped.
    #[tokio::test]
    async fn server_drop_removes_socket() {
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("drop_test.sock");
        {
            let _server = RawUdsShardServer::bind(path.clone(), make_shard())
                .await
                .unwrap();
        }
        assert!(!path.exists(), "socket file should be removed after drop");
    }

    /// `Flush` RPC returns a valid response.
    #[tokio::test]
    async fn flush_rpc_round_trip() {
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("flush.sock");
        let _server = RawUdsShardServer::bind(path.clone(), make_shard())
            .await
            .unwrap();

        let (mut rx, mut tx) = connect_raw(&path).await;
        write_msg(&mut tx, &ShardRequest::Flush { req_id: 7 })
            .await
            .unwrap();
        let resp: ShardResponse = read_msg(&mut rx).await.unwrap().unwrap();
        assert_eq!(resp.req_id(), 7);
        assert!(matches!(resp, ShardResponse::Flush { .. }));
    }

    /// `DumpEvents` on a fresh shard returns an empty events vec.
    #[tokio::test]
    async fn dump_events_empty() {
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("dump.sock");
        let _server = RawUdsShardServer::bind(path.clone(), make_shard())
            .await
            .unwrap();

        let (mut rx, mut tx) = connect_raw(&path).await;
        write_msg(&mut tx, &ShardRequest::DumpEvents { req_id: 1 })
            .await
            .unwrap();
        let resp: ShardResponse = read_msg(&mut rx).await.unwrap().unwrap();
        assert!(matches!(
            &resp,
            ShardResponse::DumpEvents { result: Ok(events), .. } if events.is_empty()
        ));
    }

    /// `ShardSizes` RPC returns a zero snapshot for a fresh shard.
    #[tokio::test]
    async fn shard_sizes_fresh() {
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("sizes.sock");
        let _server = RawUdsShardServer::bind(path.clone(), make_shard())
            .await
            .unwrap();

        let (mut rx, mut tx) = connect_raw(&path).await;
        write_msg(&mut tx, &ShardRequest::ShardSizes { req_id: 5 })
            .await
            .unwrap();
        let resp: ShardResponse = read_msg(&mut rx).await.unwrap().unwrap();
        assert!(matches!(
            resp,
            ShardResponse::ShardSizes {
                req_id: 5,
                snapshot: ShardSizeSnapshot {
                    worker_count: 0,
                    ..
                }
            }
        ));
    }

    /// Fire-and-forget `ApplyEvent` followed by `Flush` completes without error.
    #[tokio::test]
    async fn apply_event_then_flush() {
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("apply.sock");
        let _server = RawUdsShardServer::bind(path.clone(), make_shard())
            .await
            .unwrap();

        let (mut rx, mut tx) = connect_raw(&path).await;
        write_msg(&mut tx, &ShardRequest::ApplyEvent(cleared_event(42)))
            .await
            .unwrap();
        write_msg(&mut tx, &ShardRequest::Flush { req_id: 10 })
            .await
            .unwrap();
        let resp: ShardResponse = read_msg(&mut rx).await.unwrap().unwrap();
        assert_eq!(resp.req_id(), 10);
    }
}
