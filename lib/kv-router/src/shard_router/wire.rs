// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Wire types and frame codec for the raw-UDS shard protocol.
//!
//! ## Frame format
//!
//! Each message on the wire is:
//!
//! ```text
//! ┌──────────────────┬──────────────────────────┐
//! │  4-byte LE u32   │  N bytes (rmp-serde body) │
//! │  (body length)   │                           │
//! └──────────────────┴──────────────────────────┘
//! ```
//!
//! The body is serialised with [`rmp_serde`] (MessagePack).  Both
//! [`ShardRequest`] and [`ShardResponse`] are self-describing enums,
//! so a single connection may carry interleaved request/response pairs
//! without needing a separate framing layer.
//!
//! ## Request/response correlation
//!
//! Fire-and-forget messages (`ApplyEvent`, `EnqueueAnchor`, `RemoveWorker`,
//! `RemoveWorkerDpRank`) carry no `req_id`.  Read RPCs carry a `req_id: u32`
//! that is echoed back verbatim in the matching [`ShardResponse`].  The
//! client maintains a `DashMap<u32, oneshot::Sender<_>>` to dispatch
//! incoming responses.

use serde::{Deserialize, Serialize};
use tokio::io::{AsyncRead, AsyncReadExt, AsyncWrite, AsyncWriteExt};

use crate::indexer::{AnchorRef, AnchorTask, ShardSizeSnapshot};
use crate::protocols::{
    DpRank, LocalBlockHash, OverlapScores, RouterEvent, WorkerId, WorkerWithDpRank,
};

// ---------------------------------------------------------------------------
// Request envelope
// ---------------------------------------------------------------------------

/// Messages sent from a [`RawUdsShardClient`] to a [`RawUdsShardServer`].
///
/// Variants without `req_id` are fire-and-forget; the server processes them
/// asynchronously and sends no reply.  Variants with `req_id` are RPCs; the
/// server echoes the id in the matching [`ShardResponse`].
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ShardRequest {
    // ── fire-and-forget ──────────────────────────────────────────────────
    /// Apply a KV cache event to the shard.
    ApplyEvent(RouterEvent),
    /// Install an anchor before a dependent suffix event.
    EnqueueAnchor {
        worker: WorkerWithDpRank,
        anchor: AnchorTask,
    },
    /// Remove all state for the given worker.
    RemoveWorker(WorkerId),
    /// Remove state for a specific `(worker_id, dp_rank)` pair.
    RemoveWorkerDpRank(WorkerId, DpRank),
    /// Ask the server to shut down gracefully.
    Shutdown,

    // ── request-response ────────────────────────────────────────────────
    /// Find block matches starting from `anchor` through `suffix` blocks.
    FindMatchesFromAnchor {
        req_id: u32,
        anchor: AnchorRef,
        suffix: Vec<LocalBlockHash>,
    },
    /// Dump all stored events (for recovery / state transfer).
    DumpEvents { req_id: u32 },
    /// Return the current size snapshot for the shard.
    ShardSizes { req_id: u32 },
    /// Flush all pending writes; returns the queue depth at call time.
    Flush { req_id: u32 },
}

impl ShardRequest {
    /// True for fire-and-forget variants (no reply expected).
    pub fn is_fire_and_forget(&self) -> bool {
        matches!(
            self,
            ShardRequest::ApplyEvent(_)
                | ShardRequest::EnqueueAnchor { .. }
                | ShardRequest::RemoveWorker(_)
                | ShardRequest::RemoveWorkerDpRank(_, _)
                | ShardRequest::Shutdown
        )
    }
}

// ---------------------------------------------------------------------------
// Response envelope
// ---------------------------------------------------------------------------

/// Messages sent from a [`RawUdsShardServer`] back to a [`RawUdsShardClient`].
///
/// Each variant echoes the `req_id` from the originating [`ShardRequest`].
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ShardResponse {
    FindMatchesFromAnchor {
        req_id: u32,
        /// `Ok` contains the serialisable form of [`OverlapScores`].
        result: Result<WireOverlapScores, String>,
    },
    DumpEvents {
        req_id: u32,
        result: Result<Vec<RouterEvent>, String>,
    },
    ShardSizes {
        req_id: u32,
        snapshot: ShardSizeSnapshot,
    },
    Flush {
        req_id: u32,
        /// Queue depth at the moment the flush completed.
        queue_depth: usize,
    },
}

impl ShardResponse {
    /// The correlation id for this response.
    pub fn req_id(&self) -> u32 {
        match self {
            ShardResponse::FindMatchesFromAnchor { req_id, .. } => *req_id,
            ShardResponse::DumpEvents { req_id, .. } => *req_id,
            ShardResponse::ShardSizes { req_id, .. } => *req_id,
            ShardResponse::Flush { req_id, .. } => *req_id,
        }
    }
}

// ---------------------------------------------------------------------------
// Wire-safe OverlapScores
// ---------------------------------------------------------------------------

/// Wire-safe form of [`OverlapScores`].
///
/// [`OverlapScores`] uses `FxHashMap<WorkerWithDpRank, u32>` whose struct
/// keys are not valid JSON map keys.  For msgpack the issue is that
/// `FxHashMap` serialises as a map whose keys are structs — this round-trips
/// fine with `rmp_serde` but we keep the explicit vec-of-pairs form here for
/// forward compatibility with other transports and for easier debugging.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct WireOverlapScores {
    pub scores: Vec<(WorkerWithDpRank, u32)>,
    pub frequencies: Vec<usize>,
}

impl From<OverlapScores> for WireOverlapScores {
    fn from(s: OverlapScores) -> Self {
        Self {
            scores: s.scores.into_iter().collect(),
            frequencies: s.frequencies,
        }
    }
}

impl From<WireOverlapScores> for OverlapScores {
    fn from(w: WireOverlapScores) -> Self {
        Self {
            scores: w.scores.into_iter().collect(),
            frequencies: w.frequencies,
        }
    }
}

// ---------------------------------------------------------------------------
// Frame codec
// ---------------------------------------------------------------------------

/// Maximum frame body size (64 MiB) — protects against corrupt length fields.
pub const MAX_FRAME_BYTES: u32 = 64 * 1024 * 1024;

/// Encode `msg` into a length-prefixed msgpack frame.
///
/// The returned `Vec<u8>` is: `[len_le: 4 bytes][body: len bytes]`.
pub fn encode_msg<T: Serialize>(msg: &T) -> anyhow::Result<Vec<u8>> {
    let body = rmp_serde::to_vec_named(msg)?;
    let len = u32::try_from(body.len())
        .map_err(|_| anyhow::anyhow!("frame body too large: {} bytes", body.len()))?;
    let mut frame = Vec::with_capacity(4 + body.len());
    frame.extend_from_slice(&len.to_le_bytes());
    frame.extend_from_slice(&body);
    Ok(frame)
}

/// Read and decode one length-prefixed msgpack frame from `reader`.
///
/// Returns `None` if the connection was closed cleanly (zero-byte read on the
/// length header).  Returns `Err` on truncated frames, oversized frames, or
/// deserialisation failures.
pub async fn read_msg<T, R>(reader: &mut R) -> anyhow::Result<Option<T>>
where
    T: for<'de> Deserialize<'de>,
    R: AsyncRead + Unpin,
{
    // Read the 4-byte length header.  A zero-byte read means the peer closed
    // the connection cleanly (Ok(None)).  A 1-3 byte read is a truncated header
    // and must be treated as an error, not a clean close.
    let mut len_buf = [0u8; 4];
    let n = match reader.read(&mut len_buf[..1]).await {
        Ok(0) => return Ok(None), // clean EOF before any header byte
        Ok(n) => n,
        Err(e) => return Err(e.into()),
    };
    if n < 1 {
        return Ok(None);
    }
    // Read the remaining 3 bytes; any short read here is a truncated frame.
    match reader.read_exact(&mut len_buf[1..]).await {
        Ok(_) => {}
        Err(e) => {
            return Err(anyhow::anyhow!(
                "truncated frame header ({} of 4 bytes read): {}",
                1,
                e
            ));
        }
    }
    let body_len = u32::from_le_bytes(len_buf);
    if body_len > MAX_FRAME_BYTES {
        anyhow::bail!(
            "incoming frame too large: {} bytes (max {})",
            body_len,
            MAX_FRAME_BYTES
        );
    }
    let mut body = vec![0u8; body_len as usize];
    reader.read_exact(&mut body).await?;
    let msg = rmp_serde::from_slice(&body)?;
    Ok(Some(msg))
}

/// Write one length-prefixed msgpack frame to `writer`.
pub async fn write_msg<T, W>(writer: &mut W, msg: &T) -> anyhow::Result<()>
where
    T: Serialize,
    W: AsyncWrite + Unpin,
{
    let frame = encode_msg(msg)?;
    writer.write_all(&frame).await?;
    Ok(())
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::protocols::{
        ExternalSequenceBlockHash, KvCacheEvent, KvCacheEventData, WorkerWithDpRank,
    };
    use tokio::io::BufReader;

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

    /// Round-trip a `ShardRequest::ApplyEvent` through encode/decode.
    #[test]
    fn encode_decode_fire_and_forget() {
        let req = ShardRequest::ApplyEvent(cleared_event(1));
        let frame = encode_msg(&req).unwrap();
        // verify 4-byte length prefix + body round-trips
        let body = &frame[4..];
        let decoded: ShardRequest = rmp_serde::from_slice(body).unwrap();
        assert!(matches!(decoded, ShardRequest::ApplyEvent(_)));
    }

    /// Round-trip a `ShardResponse::Flush` through encode/decode.
    #[test]
    fn encode_decode_flush_response() {
        let resp = ShardResponse::Flush {
            req_id: 42,
            queue_depth: 7,
        };
        let frame = encode_msg(&resp).unwrap();
        let body = &frame[4..];
        let decoded: ShardResponse = rmp_serde::from_slice(body).unwrap();
        assert_eq!(decoded.req_id(), 42);
    }

    /// `read_msg` returns `None` on clean EOF.
    #[tokio::test]
    async fn read_msg_eof_returns_none() {
        let empty: &[u8] = &[];
        let mut reader = BufReader::new(empty);
        let result: anyhow::Result<Option<ShardRequest>> = read_msg(&mut reader).await;
        assert!(result.unwrap().is_none());
    }

    /// `read_msg` rejects oversized frames.
    #[tokio::test]
    async fn read_msg_rejects_oversized_frame() {
        let huge_len = (MAX_FRAME_BYTES + 1).to_le_bytes();
        let data: &[u8] = &huge_len;
        let mut reader = BufReader::new(data);
        let result: anyhow::Result<Option<ShardRequest>> = read_msg(&mut reader).await;
        assert!(result.is_err());
    }

    /// End-to-end async write + read round-trip via a duplex pipe.
    #[tokio::test]
    async fn write_read_round_trip() {
        let (client, server) = tokio::io::duplex(4096);
        let (client_read, mut client_write) = tokio::io::split(client);
        let (mut server_read, _server_write) = tokio::io::split(server);

        let req = ShardRequest::ShardSizes { req_id: 99 };
        write_msg(&mut client_write, &req).await.unwrap();
        drop(client_write);

        let decoded: ShardRequest = read_msg(&mut server_read).await.unwrap().unwrap();
        assert!(matches!(decoded, ShardRequest::ShardSizes { req_id: 99 }));

        let _ = client_read; // suppress unused warning
    }

    /// `is_fire_and_forget` correctly identifies non-RPC variants.
    #[test]
    fn is_fire_and_forget_classification() {
        let faf_cases = [
            ShardRequest::ApplyEvent(cleared_event(1)),
            ShardRequest::RemoveWorker(1),
            ShardRequest::RemoveWorkerDpRank(1, 0),
            ShardRequest::Shutdown,
        ];
        for req in &faf_cases {
            assert!(
                req.is_fire_and_forget(),
                "{req:?} should be fire-and-forget"
            );
        }

        let rpc_cases = [
            ShardRequest::FindMatchesFromAnchor {
                req_id: 1,
                anchor: AnchorRef {
                    anchor_id: ExternalSequenceBlockHash(0),
                    anchor_local_hash: LocalBlockHash(0),
                    anchor_depth: 0,
                },
                suffix: vec![],
            },
            ShardRequest::DumpEvents { req_id: 2 },
            ShardRequest::ShardSizes { req_id: 3 },
            ShardRequest::Flush { req_id: 4 },
        ];
        for req in &rpc_cases {
            assert!(!req.is_fire_and_forget(), "{req:?} should be RPC");
        }
    }

    /// `WireOverlapScores` round-trips through `OverlapScores` losslessly.
    #[test]
    fn wire_overlap_scores_round_trip() {
        let worker = WorkerWithDpRank {
            worker_id: 3,
            dp_rank: 1,
        };
        let mut original = OverlapScores::default();
        original.scores.insert(worker, 5);
        original.frequencies = vec![1, 2, 3];

        let wire = WireOverlapScores::from(original.clone());
        let back = OverlapScores::from(wire);

        assert_eq!(back.scores.get(&worker), Some(&5));
        assert_eq!(back.frequencies, vec![1, 2, 3]);
    }
}
