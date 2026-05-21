// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! KV-index hub publisher wiring.
//!
//! When the connector is given hub details (env [`HUB_URL_ENV`]), it probes the
//! hub for the KV-indexer feature via `GET /v1/features/kv-index/config`. If the
//! feature is present and its `block_size` matches this worker's page size, the
//! connector connects a ZMQ `PUB` socket to the advertised endpoint and wires a
//! [`Publisher`] into the block-registry [`EventsManager`] so block create/remove
//! events flow to the hub's index.

use std::sync::Arc;
use std::time::Duration;

use anyhow::{Context, Result};
use bytes::Bytes;
use futures::SinkExt;
use futures::future::BoxFuture;
use kvbm_logical::pubsub::Publisher;
use serde::Deserialize;
use tmq::{Context as ZmqContext, Multipart, publish::Publish, publish::publish};
use tokio::sync::{Mutex, mpsc};

/// Environment variable carrying the KV-index hub URL (discovery base, e.g.
/// `http://hub-host:1337`). Intended for non-disagg / local use; under vLLM it
/// does not survive the EngineCore spawn. When unset, the connector falls back
/// to `disagg.hub_url` (which does survive, via `kv_connector_extra_config`).
pub const HUB_URL_ENV: &str = "DYN_KVBM_KV_INDEX_HUB_URL";

/// Subject/topic frame prepended to each published batch.
pub const SUBJECT: &str = "kvbm.kv_index";

const PROBE_TIMEOUT: Duration = Duration::from_secs(5);
const ZMQ_LINGER_MS: i32 = 0;

/// Subset of the hub's `GET /config` response we need to wire a publisher.
#[derive(Debug, Deserialize)]
struct KvIndexConfig {
    block_size: usize,
    zmq_endpoint: String,
}

/// Probes the hub for the KV-indexer feature.
///
/// Returns the advertised ZMQ endpoint when the feature is present, reachable,
/// and its `block_size` matches `page_size`. Any failure (feature absent, hub
/// unreachable, size mismatch) is logged and yields `None` — KV indexing is
/// best-effort and never blocks connector startup.
///
/// This probe runs **once**, during `initialize_async`. If the hub is not yet
/// reachable at that moment the publisher stays disabled for the life of the
/// leader (no retry / re-probe). Deployments must therefore start the hub
/// before the workers. A worker that publishes to the index but never
/// `register_instance`s with the hub also never triggers the hub's
/// `on_unregister` index sweep — its entries are reclaimed only via explicit
/// `KvCacheEvents::Remove`/`Shutdown` from the event pipeline.
pub async fn probe(hub_url: &str, page_size: usize) -> Option<String> {
    let url = format!(
        "{}/v1/features/kv-index/config",
        hub_url.trim_end_matches('/')
    );
    let client = reqwest::Client::new();
    let resp = match client.get(&url).timeout(PROBE_TIMEOUT).send().await {
        Ok(r) => r,
        Err(e) => {
            tracing::info!(hub_url, error = %e, "kv-index probe failed; publisher disabled");
            return None;
        }
    };
    if !resp.status().is_success() {
        tracing::info!(hub_url, status = %resp.status(), "kv-index feature absent; publisher disabled");
        return None;
    }
    let cfg: KvIndexConfig = match resp.json().await {
        Ok(c) => c,
        Err(e) => {
            tracing::warn!(hub_url, error = %e, "kv-index config decode failed; publisher disabled");
            return None;
        }
    };
    if cfg.block_size != page_size {
        tracing::warn!(
            hub_block_size = cfg.block_size,
            page_size,
            "kv-index block_size mismatch; publisher disabled"
        );
        return None;
    }
    if cfg.zmq_endpoint.is_empty() {
        tracing::warn!("kv-index reported empty zmq_endpoint; publisher disabled");
        return None;
    }
    tracing::info!(endpoint = %cfg.zmq_endpoint, "kv-index feature present; wiring publisher");
    Some(cfg.zmq_endpoint)
}

/// [`Publisher`] backed by a ZMQ `PUB` socket connected to the hub's `SUB`
/// ingest socket.
///
/// The [`Publisher`] contract is synchronous (`publish` returns immediately),
/// but the tmq socket send is async. A bounded mpsc channel bridges the two: a
/// background task owns the socket and drains the channel. Backpressure drops
/// the oldest pending batch rather than blocking the event pipeline — KV index
/// freshness is advisory.
pub struct ZmqHubPublisher {
    tx: mpsc::Sender<Bytes>,
}

impl ZmqHubPublisher {
    /// Connects a `PUB` socket to `endpoint` and spawns the send task.
    pub fn connect(endpoint: &str) -> Result<Self> {
        let ctx = ZmqContext::new();
        let socket = publish(&ctx)
            .set_linger(ZMQ_LINGER_MS)
            .connect(endpoint)
            .with_context(|| format!("connecting kv-index PUB socket to {endpoint}"))?;
        let socket = Arc::new(Mutex::new(socket));

        let (tx, mut rx) = mpsc::channel::<Bytes>(1024);
        tokio::spawn(async move {
            while let Some(payload) = rx.recv().await {
                if let Err(e) = send_batch(&socket, payload).await {
                    tracing::warn!(error = %e, "kv-index PUB send failed");
                }
            }
            tracing::info!("kv-index PUB send task stopped");
        });

        Ok(Self { tx })
    }
}

async fn send_batch(socket: &Arc<Mutex<Publish>>, payload: Bytes) -> Result<()> {
    let frames: Vec<Vec<u8>> = vec![SUBJECT.as_bytes().to_vec(), payload.to_vec()];
    socket.lock().await.send(Multipart::from(frames)).await?;
    Ok(())
}

impl Publisher for ZmqHubPublisher {
    fn publish(&self, _subject: &str, payload: Bytes) -> Result<()> {
        // Non-blocking hand-off. On a full channel, drop the batch (advisory
        // index) rather than stalling the broadcast subscriber.
        match self.tx.try_send(payload) {
            Ok(()) => Ok(()),
            Err(mpsc::error::TrySendError::Full(_)) => {
                tracing::warn!("kv-index publish channel full; dropping batch");
                Ok(())
            }
            Err(mpsc::error::TrySendError::Closed(_)) => {
                anyhow::bail!("kv-index publish channel closed")
            }
        }
    }

    fn flush(&self) -> BoxFuture<'static, Result<()>> {
        Box::pin(async { Ok(()) })
    }
}
