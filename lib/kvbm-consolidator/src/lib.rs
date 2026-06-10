// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! KV-cache event consolidator.
//!
//! Consumes events from multiple sources and publishes a deduplicated, kv-router-compatible
//! stream:
//! - G1 events from vLLM / TensorRT-LLM worker processes over ZMQ SUB.
//! - G2/G3 events from in-process KVBM via [`kvbm_logical::events::EventsManager`] broadcast.
//!
//! Dedup key is [`kvbm_logical::SequenceHash`] (a 128-bit [`dynamo_tokens::PositionalLineageHash`]).
//! All sources produce the same key for the same logical block, so the first STORE
//! publishes and only the last REMOVE publishes — no external-hash translation needed
//! between the tracker and kvbm-logical's registry.
//!
//! For external string-hashed sources (vLLM / TRT-LLM), a `(EventSource, String)` map inside
//! the tracker resolves framework block hashes to the canonical `SequenceHash`.
//!
//! Egress events use kv-router's u64 wire format: the 128-bit hash is projected via
//! [`hash::router_block_hash`] / [`hash::router_parent_hash`].

pub mod egress;
pub mod hash;
pub mod ingress;
pub mod source;
pub mod tracker;
pub mod wire;
pub mod zmq_util;

#[cfg(feature = "testing")]
pub mod test_utils;

use std::pin::Pin;
use std::sync::Arc;
use std::time::Duration;

use anyhow::Result;
use futures::Stream;
use kvbm_logical::{BlockRegistry, SequenceHash, events::protocol::KvCacheEvent};
use tokio::sync::RwLock;
use tokio::task::JoinHandle;
use tokio_util::sync::CancellationToken;

pub use source::EventSource;
pub use tracker::{ConsolidatedEvent, Tracker};

/// Handle for direct KVBM callers that want to bypass the broadcast bridge.
#[derive(Clone)]
pub struct ConsolidatorHandle {
    tracker: Arc<RwLock<Tracker>>,
}

impl ConsolidatorHandle {
    /// Forward a KVBM store event directly to the tracker.
    pub async fn handle_kvbm_store(
        &self,
        seq_hash: SequenceHash,
        token_ids: Vec<u32>,
        block_size: usize,
        lora_name: Option<String>,
    ) {
        let mut tracker = self.tracker.write().await;
        tracker.handle_kvbm_store(seq_hash, token_ids, block_size, lora_name);
    }

    /// Forward a KVBM remove event directly to the tracker.
    pub async fn handle_kvbm_remove(&self, seq_hash: SequenceHash) {
        let mut tracker = self.tracker.write().await;
        tracker.handle_kvbm_remove(seq_hash);
    }

    /// Clear all blocks.
    pub async fn handle_clear_all(&self) {
        let mut tracker = self.tracker.write().await;
        tracker.handle_clear_all();
    }
}

/// Boxed stream of KV-cache events from `kvbm_logical::events::EventsManager`.
///
/// `EventsManager::subscribe()` returns `impl Stream<Item = KvCacheEvent>`, so the
/// builder accepts any `Stream` and boxes it for storage. Tests can wrap a
/// `broadcast::Receiver` with `tokio_stream::wrappers::BroadcastStream` and filter
/// `Ok` if they want broadcast semantics.
pub type KvCacheEventStream = Pin<Box<dyn Stream<Item = KvCacheEvent> + Send + 'static>>;

/// Builder for [`Consolidator`].
pub struct ConsolidatorBuilder {
    zmq_in: Option<String>,
    zmq_out: String,
    engine_source: EventSource,
    registry: Option<BlockRegistry>,
    kvbm_events: Option<KvCacheEventStream>,
    poll_interval: Duration,
}

impl ConsolidatorBuilder {
    /// Create a builder with the required egress endpoint and engine source.
    ///
    /// `engine_source` tags any ZMQ-ingress events with their framework (Vllm or Trtllm).
    pub fn new(zmq_out: impl Into<String>, engine_source: EventSource) -> Self {
        Self {
            zmq_in: None,
            zmq_out: zmq_out.into(),
            engine_source,
            registry: None,
            kvbm_events: None,
            poll_interval: Duration::from_millis(50),
        }
    }

    /// Enable the ZMQ SUB ingress. Omit to disable G1 ingress.
    pub fn zmq_in(mut self, endpoint: impl Into<String>) -> Self {
        self.zmq_in = Some(endpoint.into());
        self
    }

    /// Attach a kvbm-logical registry so the consolidator can register G2/G3 blocks
    /// against the shared presence oracle.
    pub fn registry(mut self, registry: BlockRegistry) -> Self {
        self.registry = Some(registry);
        self
    }

    /// Enable the KVBM ingress from any stream of [`KvCacheEvent`]. Pass
    /// `EventsManager::subscribe()` directly in production.
    pub fn kvbm_events<S>(mut self, stream: S) -> Self
    where
        S: Stream<Item = KvCacheEvent> + Send + 'static,
    {
        self.kvbm_events = Some(Box::pin(stream));
        self
    }

    /// Publisher drain interval. Defaults to 50ms.
    pub fn poll_interval(mut self, interval: Duration) -> Self {
        self.poll_interval = interval;
        self
    }

    /// Start all configured tasks and return a live [`Consolidator`].
    pub async fn build(self) -> Result<Consolidator> {
        let cancel = CancellationToken::new();
        let tracker = Arc::new(RwLock::new(Tracker::new(self.registry.clone())));

        let mut join_handles: Vec<JoinHandle<()>> = Vec::new();

        if let Some(endpoint) = self.zmq_in {
            join_handles.push(
                ingress::zmq_subscriber::spawn(
                    endpoint,
                    tracker.clone(),
                    self.engine_source,
                    cancel.clone(),
                )
                .await?,
            );
        }

        if let Some(stream) = self.kvbm_events {
            join_handles.push(ingress::kvbm_bridge::spawn(
                stream,
                tracker.clone(),
                cancel.clone(),
            ));
        }

        join_handles.push(
            egress::zmq_publisher::spawn(
                self.zmq_out,
                tracker.clone(),
                self.poll_interval,
                cancel.clone(),
            )
            .await?,
        );

        Ok(Consolidator {
            tracker,
            cancel,
            join_handles,
        })
    }
}

/// Live consolidator with running ingress / egress tasks.
pub struct Consolidator {
    tracker: Arc<RwLock<Tracker>>,
    cancel: CancellationToken,
    join_handles: Vec<JoinHandle<()>>,
}

impl Consolidator {
    /// Obtain a handle for direct KVBM injection (bypasses the broadcast bridge).
    pub fn handle(&self) -> ConsolidatorHandle {
        ConsolidatorHandle {
            tracker: self.tracker.clone(),
        }
    }

    /// Cancel all tasks and wait for them to finish.
    pub async fn shutdown(self) {
        self.cancel.cancel();
        for h in self.join_handles {
            let _ = h.await;
        }
    }
}
