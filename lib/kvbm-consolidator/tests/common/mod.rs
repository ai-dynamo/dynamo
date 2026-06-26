// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// Suppress dead-code warnings: items here are shared across multiple test binaries; each binary
// only uses a subset, so clippy sees "unused" per-binary.
#![allow(dead_code)]

//! Shared integration-test harness for kvbm-consolidator.
//!
//! Topology:
//!   Test PUB  bind tcp://127.0.0.1:A  →  Consolidator connect (zmq_in = A)
//!   Consolidator bind tcp://127.0.0.1:B  →  Test SUB connect (zmq_out = B)
//!
//! The PUB socket is built via `kvbm_consolidator::zmq_util::bind_pub_socket` (test side
//! binds because it plays the role of vLLM).  The SUB socket likewise reuses the crate's
//! `connect_sub_socket`.

use std::sync::OnceLock;
use std::time::Duration;

use futures::StreamExt as _;
use kvbm_consolidator::wire::vllm_in::RawKvEvent;
use kvbm_consolidator::zmq_util::{SharedPubSocket, bind_pub_socket, connect_sub_socket};
use serde::{Deserialize, Serialize};
use tokio::sync::mpsc;
use tracing_subscriber::EnvFilter;

// ─── Wire-serialization helper ──────────────────────────────────────────────

/// Tuple-struct matching the publisher's array encoding:
/// `[ts, events, dp_rank]`. Used by the test PUB side to inject batches.
#[derive(Debug, Serialize)]
pub struct TestBatch(pub f64, pub Vec<RawKvEvent>, pub Option<i32>);

impl TestBatch {
    pub fn encode(&self) -> Vec<u8> {
        rmp_serde::to_vec(self).expect("TestBatch serialize")
    }
}

// ─── Egress mirror types (Deserialize only — src/ has Serialize-only) ───────

/// Mirror of `router_out::EventBatch` for test decoding.
/// Encoded as `[ts, events, dp_rank]` (array / tuple struct on the wire).
#[derive(Debug, Deserialize)]
pub struct EventBatchMirror(pub f64, pub Vec<EventMirror>, pub Option<i32>);

/// Mirror of `router_out::Event` — same msgpack shape, owns a Deserialize impl.
#[derive(Debug, Deserialize, Serialize)]
#[serde(tag = "type")]
pub enum EventMirror {
    BlockStored {
        block_hashes: Vec<u64>,
        #[serde(default)]
        parent_block_hash: Option<u64>,
        token_ids: Vec<i32>,
        block_size: i32,
        #[serde(default)]
        lora_name: Option<String>,
        #[serde(default)]
        medium: Option<String>,
    },
    BlockRemoved {
        block_hashes: Vec<u64>,
        #[serde(default)]
        medium: Option<String>,
    },
    AllBlocksCleared {},
}

// ─── Port allocation ─────────────────────────────────────────────────────────

pub fn pick_port() -> u16 {
    portpicker::pick_unused_port().expect("no free port")
}

pub fn make_endpoint(port: u16) -> String {
    format!("tcp://127.0.0.1:{port}")
}

// ─── Tracing init ────────────────────────────────────────────────────────────

static TRACING: OnceLock<()> = OnceLock::new();

pub fn init_tracing() {
    TRACING.get_or_init(|| {
        let _ = tracing_subscriber::fmt()
            .with_env_filter(
                EnvFilter::try_from_default_env()
                    .unwrap_or_else(|_| EnvFilter::new("kvbm_consolidator=debug")),
            )
            .with_test_writer()
            .try_init();
    });
}

// ─── ZmqPubHandle ────────────────────────────────────────────────────────────

pub struct ZmqPubHandle {
    socket: SharedPubSocket,
    pub endpoint: String,
}

impl ZmqPubHandle {
    /// Bind a PUB socket on a free port; simulates the vLLM publisher.
    pub async fn spawn() -> Self {
        let port = pick_port();
        let endpoint = make_endpoint(port);
        let socket = bind_pub_socket(&endpoint)
            .await
            .expect("bind_pub_socket failed");
        Self { socket, endpoint }
    }

    /// Encode `batch` as msgpack and send as 3-frame multipart `[b"", seq, payload]`.
    pub async fn send_batch(&self, batch: &TestBatch) -> anyhow::Result<()> {
        let payload = batch.encode();
        self.send_frames(vec![vec![], vec![0u8; 8], payload]).await
    }

    /// Send arbitrary raw frames (for malformed-frame tests).
    pub async fn send_frames(&self, frames: Vec<Vec<u8>>) -> anyhow::Result<()> {
        kvbm_consolidator::zmq_util::send_multipart(&self.socket, frames).await
    }
}

// ─── ZmqSubHandle ────────────────────────────────────────────────────────────

pub struct ZmqSubHandle {
    pub rx: mpsc::Receiver<(u64, EventBatchMirror)>,
}

impl ZmqSubHandle {
    /// Connect a SUB socket; spawn a background task that decodes 3-frame batches
    /// and forwards `(seq_u64, EventBatchMirror)` to the returned channel.
    pub async fn spawn(endpoint: &str) -> anyhow::Result<Self> {
        let mut socket = connect_sub_socket(endpoint, None).await?;
        let (tx, rx) = mpsc::channel::<(u64, EventBatchMirror)>(1024);

        tokio::spawn(async move {
            while let Some(msg) = socket.next().await {
                let frames = match msg {
                    Ok(f) => kvbm_consolidator::zmq_util::multipart_message(f),
                    Err(_) => break,
                };
                if frames.len() < 3 {
                    continue;
                }
                let seq = {
                    let mut arr = [0u8; 8];
                    let src = &frames[1];
                    let len = src.len().min(8);
                    arr[..len].copy_from_slice(&src[..len]);
                    u64::from_be_bytes(arr)
                };
                match rmp_serde::from_slice::<EventBatchMirror>(&frames[2]) {
                    Ok(batch) => {
                        if tx.send((seq, batch)).await.is_err() {
                            break;
                        }
                    }
                    Err(e) => {
                        tracing::warn!("ZmqSubHandle: failed to decode egress batch: {e}");
                    }
                }
            }
        });

        Ok(Self { rx })
    }

    /// Collect up to `n` items within `timeout`. Returns what arrived.
    pub async fn recv_n(
        &mut self,
        n: usize,
        timeout: Duration,
    ) -> anyhow::Result<Vec<(u64, EventBatchMirror)>> {
        let mut collected = Vec::with_capacity(n);
        let deadline = tokio::time::Instant::now() + timeout;
        while collected.len() < n {
            let remaining = deadline.saturating_duration_since(tokio::time::Instant::now());
            if remaining.is_zero() {
                break;
            }
            match tokio::time::timeout(remaining, self.rx.recv()).await {
                Ok(Some(item)) => collected.push(item),
                Ok(None) => break,
                Err(_) => break,
            }
        }
        Ok(collected)
    }
}

// ─── sync_pulse ──────────────────────────────────────────────────────────────

/// Defeat the ZMQ slow-joiner problem.
///
/// Sends an AllBlocksCleared sentinel and waits until the SUB side receives *any*
/// AllBlocksCleared event. Returns once the SUB is proven to be connected and
/// receiving. Call this after wiring up PUB/Consolidator/SUB but before asserting
/// real stimuli.
pub async fn sync_pulse(
    pub_handle: &ZmqPubHandle,
    sub_handle: &mut ZmqSubHandle,
    timeout: Duration,
) -> bool {
    let sentinel = TestBatch(0.0, vec![RawKvEvent::AllBlocksCleared], None);
    // Keep sending until we see the response.
    let start = tokio::time::Instant::now();
    loop {
        if start.elapsed() >= timeout {
            return false;
        }
        // Drain the channel first to discard stale messages.
        while sub_handle.rx.try_recv().is_ok() {}

        let _ = pub_handle.send_batch(&sentinel).await;

        // Wait up to 200 ms for a single AllBlocksCleared to arrive.
        let received = sub_handle
            .recv_n(1, Duration::from_millis(200))
            .await
            .unwrap_or_default();

        let has_clear = received.iter().any(|(_, batch)| {
            batch
                .1
                .iter()
                .any(|e| matches!(e, EventMirror::AllBlocksCleared {}))
        });
        if has_clear {
            return true;
        }
    }
}

// ─── wait_for ────────────────────────────────────────────────────────────────

/// Spin until `probe()` returns true or `deadline` elapses. Returns the final
/// probe result. Uses short yields — not sleep.
pub async fn wait_for(deadline: Duration, mut probe: impl FnMut() -> bool) -> bool {
    let end = tokio::time::Instant::now() + deadline;
    loop {
        if probe() {
            return true;
        }
        if tokio::time::Instant::now() >= end {
            return false;
        }
        tokio::task::yield_now().await;
    }
}
