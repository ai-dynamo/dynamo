// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! The Global Router's velo ingress: relays ask for one stream per worker, and
//! each stream's consumer applies that worker's records to the index in order.
//! A record is one event-plane payload, msgpack `Vec<RouterEvent>`, forwarded
//! unchanged by the pool relay.

use std::path::Path;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Duration;

use bytes::Bytes;
use dynamo_kv_router::protocols::RouterEvent;
use dynamo_kv_router::services::indexer::backend::Indexer;
use velo::streaming::AnchorConfig;
use velo::{Handler, StreamAnchor, StreamFrame, Velo};

use crate::relay_common::{OPEN_HANDLER, OpenReply, OpenRequest};
use crate::route::PoolMap;

#[derive(Default)]
pub struct StreamStats {
    pub streams_opened: AtomicU64,
    pub records: AtomicU64,
    pub record_bytes: AtomicU64,
    pub events: AtomicU64,
    pub decode_errors: AtomicU64,
    pub apply_errors: AtomicU64,
    pub streams_ended: AtomicU64,
}

impl StreamStats {
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

/// Write `velo`'s peer info so relays can find it; renamed into place so a
/// reader never sees a partial file.
pub fn write_peer_file(path: &Path, velo: &Velo) -> anyhow::Result<()> {
    let staged = path.with_extension("staged");
    std::fs::write(&staged, rmp_serde::to_vec(&velo.peer_info())?)?;
    std::fs::rename(&staged, path)?;
    Ok(())
}

/// Register the handler relays call to open one anchor per worker of their pool.
pub fn register_relay_handler(
    velo: &Velo,
    indexer: Indexer,
    pools: Arc<PoolMap>,
    stats: Arc<StreamStats>,
) -> anyhow::Result<()> {
    let weak = velo.downgrade();
    let handler = Handler::unary_handler(OPEN_HANDLER, move |ctx| {
        let request: OpenRequest = rmp_serde::from_slice(&ctx.payload)?;
        anyhow::ensure!(
            (request.pool as usize) < pools.pools(),
            "pool {} out of range",
            request.pool
        );
        let velo = weak
            .upgrade()
            .ok_or_else(|| anyhow::anyhow!("core is shutting down"))?;
        let mut handles = Vec::with_capacity(request.worker_ids.len());
        for worker_id in request.worker_ids {
            pools.insert(worker_id, request.pool);
            let anchor = velo
                .anchor_manager()
                .create_anchor_with_config::<Bytes>(AnchorConfig {
                    unattached_timeout: Some(Duration::from_secs(600)),
                    heartbeat_interval: None,
                });
            handles.push(anchor.handle().as_u128());
            stats.streams_opened.fetch_add(1, Ordering::Relaxed);
            tokio::spawn(consume(
                worker_id,
                anchor,
                indexer.clone(),
                Arc::clone(&stats),
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
    stats: Arc<StreamStats>,
) {
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
