// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Selection-core side of the benchmark: the KV-event ingress for each arm and
//! the velo relay's stream consumers.
//!
//! Both arms build the partition index the stock way
//! (`WorkerRegistry::get_or_create_indexer`) and keep a handle to it for
//! lane-depth sampling. The direct arm is the stock ZMQ ingress; the relay arm
//! takes its events from velo streams instead and needs no per-worker
//! endpoint.

use std::path::Path;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, OnceLock};
use std::time::Duration;

use async_trait::async_trait;
use bytes::Bytes;
use dynamo_kv_router::identity::RoutingPartitionId;
use dynamo_kv_router::protocols::RouterEvent;
use dynamo_kv_router::services::indexer::backend::Indexer;
use dynamo_kv_router::services::indexer::registry::WorkerRegistry;
use dynamo_kv_router::services::selection::{
    KvEventIngress, SelectionError, WorkerCatalogRecord, ZmqDirectIngress,
};

use velo::streaming::AnchorConfig;
use velo::{Handler, StreamAnchor, StreamFrame, Velo};

use crate::relay_common::{OPEN_HANDLER, OpenReply, OpenRequest};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum IngressMode {
    ZmqDirect,
    VeloRelay,
}

#[derive(Default)]
pub struct RelayStats {
    pub streams_opened: AtomicU64,
    pub records: AtomicU64,
    pub record_bytes: AtomicU64,
    pub events: AtomicU64,
    pub decode_errors: AtomicU64,
    pub apply_errors: AtomicU64,
    pub streams_ended: AtomicU64,
}

impl RelayStats {
    pub fn snapshot(&self) -> serde_json::Value {
        serde_json::json!({
            "streams_opened": self.streams_opened.load(Ordering::Relaxed),
            "records": self.records.load(Ordering::Relaxed),
            "record_bytes": self.record_bytes.load(Ordering::Relaxed),
            "events": self.events.load(Ordering::Relaxed),
            "decode_errors": self.decode_errors.load(Ordering::Relaxed),
            "apply_errors": self.apply_errors.load(Ordering::Relaxed),
            "streams_ended": self.streams_ended.load(Ordering::Relaxed),
        })
    }
}

pub struct BenchIngress {
    mode: IngressMode,
    /// The one partition's index; the benchmark registers a single model.
    index: OnceLock<Indexer>,
    pub relay: RelayStats,
}

impl BenchIngress {
    pub fn new(mode: IngressMode) -> Arc<Self> {
        Arc::new(Self {
            mode,
            index: OnceLock::new(),
            relay: RelayStats::default(),
        })
    }

    pub fn index(&self) -> Option<&Indexer> {
        self.index.get()
    }

    /// Events queued on each CRTC lane and not yet applied.
    pub fn lane_depths(&self) -> Vec<usize> {
        match self.index.get() {
            Some(Indexer::Concurrent { primary, .. }) => primary.queue_depths(),
            _ => Vec::new(),
        }
    }
}

#[async_trait]
impl KvEventIngress for BenchIngress {
    fn open(
        &self,
        registry: &WorkerRegistry,
        key: &RoutingPartitionId,
        block_size: u32,
    ) -> Indexer {
        let indexer = registry.get_or_create_indexer(key.clone(), block_size);
        if self.index.set(indexer.clone()).is_err() {
            tracing::warn!(
                ?key,
                "a second partition opened; stats and relay use the first"
            );
        }
        indexer
    }

    fn missing_metadata(&self, record: &WorkerCatalogRecord) -> Vec<String> {
        match self.mode {
            IngressMode::ZmqDirect => ZmqDirectIngress.missing_metadata(record),
            IngressMode::VeloRelay => Vec::new(),
        }
    }

    async fn reconcile(
        &self,
        registry: &WorkerRegistry,
        previous: Option<&WorkerCatalogRecord>,
        record: &WorkerCatalogRecord,
    ) -> Result<(), SelectionError> {
        match self.mode {
            IngressMode::ZmqDirect => ZmqDirectIngress.reconcile(registry, previous, record).await,
            IngressMode::VeloRelay => Ok(()),
        }
    }

    async fn detach(&self, registry: &WorkerRegistry, record: &WorkerCatalogRecord) {
        match self.mode {
            IngressMode::ZmqDirect => ZmqDirectIngress.detach(registry, record).await,
            // Workers never leave during a run.
            IngressMode::VeloRelay => {}
        }
    }
}

/// Write `velo`'s peer info so relays can find it; renamed into place so a
/// reader never sees a partial file.
pub fn write_peer_file(path: &Path, velo: &Velo) -> anyhow::Result<()> {
    let staged = path.with_extension("staged");
    std::fs::write(&staged, rmp_serde::to_vec(&velo.peer_info())?)?;
    std::fs::rename(&staged, path)?;
    Ok(())
}

/// Register the handler relays call to open one anchor per worker. Each anchor
/// gets its own consumer task, which applies records in order.
pub fn register_relay_handler(velo: &Velo, ingress: Arc<BenchIngress>) -> anyhow::Result<()> {
    let weak = velo.downgrade();
    let handler = Handler::unary_handler(OPEN_HANDLER, move |ctx| {
        let request: OpenRequest = rmp_serde::from_slice(&ctx.payload)?;
        let velo = weak
            .upgrade()
            .ok_or_else(|| anyhow::anyhow!("core is shutting down"))?;
        let indexer = ingress
            .index()
            .cloned()
            .ok_or_else(|| anyhow::anyhow!("no partition yet: register workers first"))?;
        let mut handles = Vec::with_capacity(request.worker_ids.len());
        for worker_id in request.worker_ids {
            let anchor = velo
                .anchor_manager()
                .create_anchor_with_config::<Bytes>(AnchorConfig {
                    unattached_timeout: Some(Duration::from_secs(600)),
                    heartbeat_interval: None,
                });
            handles.push(anchor.handle().as_u128());
            ingress.relay.streams_opened.fetch_add(1, Ordering::Relaxed);
            tokio::spawn(consume(
                worker_id,
                anchor,
                indexer.clone(),
                Arc::clone(&ingress),
            ));
        }
        Ok(Some(Bytes::from(rmp_serde::to_vec(&OpenReply {
            handles,
        })?)))
    })
    .build();
    velo.register_handler(handler)
}

async fn consume(
    worker_id: u64,
    mut anchor: StreamAnchor<Bytes>,
    indexer: Indexer,
    ingress: Arc<BenchIngress>,
) {
    let stats = &ingress.relay;
    while let Some(frame) = anchor.next_bytes().await {
        match frame {
            Ok(StreamFrame::Item(record)) => {
                stats.records.fetch_add(1, Ordering::Relaxed);
                stats
                    .record_bytes
                    .fetch_add(record.len() as u64, Ordering::Relaxed);
                let events: Vec<RouterEvent> = match rmp_serde::from_slice(&record) {
                    Ok(events) => events,
                    Err(error) => {
                        stats.decode_errors.fetch_add(1, Ordering::Relaxed);
                        tracing::warn!(worker_id, %error, "undecodable relay record");
                        continue;
                    }
                };
                for event in events {
                    if let Err(error) = indexer.apply_event_routed(event).await {
                        stats.apply_errors.fetch_add(1, Ordering::Relaxed);
                        tracing::warn!(worker_id, %error, "relay event not applied");
                    } else {
                        stats.events.fetch_add(1, Ordering::Relaxed);
                    }
                }
            }
            Ok(StreamFrame::Finalized | StreamFrame::Detached) => break,
            Ok(other) => {
                tracing::error!(worker_id, frame = ?other, "relay stream ended abnormally");
                break;
            }
            Err(error) => {
                tracing::error!(worker_id, %error, "relay stream failed");
                break;
            }
        }
    }
    stats.streams_ended.fetch_add(1, Ordering::Relaxed);
}
