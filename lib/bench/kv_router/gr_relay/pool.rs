// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Pool side of the velo relay prototype.
//!
//! Each engine's batches reach its worker's relay task, which encodes them as
//! the engine's vLLM ZMQ publisher would, decodes and normalizes them exactly
//! as the selection core's ZMQ listener does, and sends the resulting
//! `Vec<RouterEvent>` as one record on that worker's velo stream. The core
//! therefore receives the same events as on the direct path, without token ids
//! and without decoding or hashing them itself.
//!
//! Everything runs on the relay's own runtime (threads named `gr-relay`), so
//! the cost moved off the core is visible per thread. There is no replay: a
//! stream error ends that worker's relay.

use std::path::Path;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, Instant};

use anyhow::Context as _;
use bytes::Bytes;
use dynamo_kv_router::protocols::{RouterEvent, WorkerWithDpRank};
use dynamo_kv_router::zmq_wire::{ZmqEventNormalizer, decode_event_batch};
use dynamo_mocker::common::protocols::{RawKvEvent, RawKvEventSink};
use dynamo_mocker::services::zmq_events::encode_event_batch;
use serde::Serialize;
use tokio::sync::{mpsc, oneshot};
use velo::streaming::ByteItemBuilder;
use velo::{PeerInfo, StreamAnchorHandle, StreamSender, Velo};

use crate::relay_common::{OPEN_HANDLER, OpenReply, OpenRequest, VeloOptions, build_velo};

#[derive(Default)]
pub struct RelayCounters {
    /// Engine batches handed to a relay task and not yet sent.
    pub pending_batches: AtomicU64,
    pub records: AtomicU64,
    pub record_bytes: AtomicU64,
    pub router_events: AtomicU64,
    pub send_errors: AtomicU64,
}

impl RelayCounters {
    pub fn snapshot(&self) -> serde_json::Value {
        serde_json::json!({
            "pending_batches": self.pending_batches.load(Ordering::Relaxed),
            "records": self.records.load(Ordering::Relaxed),
            "record_bytes": self.record_bytes.load(Ordering::Relaxed),
            "router_events": self.router_events.load(Ordering::Relaxed),
            "send_errors": self.send_errors.load(Ordering::Relaxed),
        })
    }
}

/// The engine-facing end of one worker's relay.
pub struct RelaySink {
    tx: mpsc::UnboundedSender<Vec<RawKvEvent>>,
    counters: Arc<RelayCounters>,
}

impl RawKvEventSink for RelaySink {
    fn publish(&self, event: RawKvEvent) -> anyhow::Result<()> {
        self.publish_batch(vec![event])
    }

    fn publish_batch(&self, events: Vec<RawKvEvent>) -> anyhow::Result<()> {
        if events.is_empty() {
            return Ok(());
        }
        self.counters
            .pending_batches
            .fetch_add(1, Ordering::Relaxed);
        self.tx
            .send(events)
            .map_err(|_| anyhow::anyhow!("relay task for this worker has stopped"))
    }
}

pub struct PoolRelay {
    /// Always `Some` until drop, which shuts it down without blocking, since the
    /// relay is owned from inside another runtime.
    runtime: Option<tokio::runtime::Runtime>,
    velo: Velo,
    workers: Vec<(u64, oneshot::Sender<StreamSender<Bytes>>)>,
    counters: Arc<RelayCounters>,
}

impl PoolRelay {
    /// Start the relay runtime and velo instance, with one sink per worker.
    /// Workers' relay tasks wait until [`Self::open`] gives them a stream.
    pub async fn start(
        options: VeloOptions,
        threads: usize,
        worker_ids: &[u64],
        block_size: u32,
    ) -> anyhow::Result<(Self, Vec<RelaySink>)> {
        let runtime = tokio::runtime::Builder::new_multi_thread()
            .worker_threads(threads)
            .thread_name("gr-relay")
            .enable_all()
            .build()?;
        let velo = runtime
            .spawn(build_velo(options, None))
            .await
            .context("relay velo build task")??;
        let counters = Arc::new(RelayCounters::default());
        let mut workers = Vec::with_capacity(worker_ids.len());
        let mut sinks = Vec::with_capacity(worker_ids.len());
        for &worker_id in worker_ids {
            let (tx, rx) = mpsc::unbounded_channel();
            let (stream_tx, stream_rx) = oneshot::channel();
            runtime.spawn(relay_worker(
                worker_id,
                block_size,
                rx,
                stream_rx,
                Arc::clone(&counters),
            ));
            workers.push((worker_id, stream_tx));
            sinks.push(RelaySink {
                tx,
                counters: Arc::clone(&counters),
            });
        }
        Ok((
            Self {
                runtime: Some(runtime),
                velo,
                workers,
                counters,
            },
            sinks,
        ))
    }

    /// Ask the core for one anchor per worker and attach every stream. Call
    /// after the workers are registered, so the core has their partition.
    pub async fn open(&mut self, peer_file: &Path, timeout: Duration) -> anyhow::Result<()> {
        let router = read_peer_file(peer_file, timeout).await?;
        let router_id = router.instance_id();
        let velo = self.velo.clone();
        let worker_ids: Vec<u64> = self.workers.iter().map(|(id, _)| *id).collect();
        let senders = self
            .runtime()
            .spawn(async move {
                velo.register_peer(router)?;
                tokio::time::timeout(timeout, velo.wait_for_handler(router_id, OPEN_HANDLER))
                    .await
                    .context("core's relay handler did not appear")??;
                let request = rmp_serde::to_vec(&OpenRequest {
                    worker_ids: worker_ids.clone(),
                })?;
                let reply = velo
                    .unary(OPEN_HANDLER)?
                    .raw_payload(Bytes::from(request))
                    .instance(router_id)
                    .send()
                    .await?;
                let reply: OpenReply = rmp_serde::from_slice(&reply)?;
                anyhow::ensure!(
                    reply.handles.len() == worker_ids.len(),
                    "core opened {} streams for {} workers",
                    reply.handles.len(),
                    worker_ids.len()
                );
                let attaches = reply.handles.iter().map(|&handle| {
                    velo.attach_anchor::<Bytes>(StreamAnchorHandle::from_u128(handle))
                });
                let senders = futures_util::future::try_join_all(attaches).await?;
                Ok::<_, anyhow::Error>(senders)
            })
            .await
            .context("relay open task")??;
        for ((_, stream_tx), sender) in self.workers.drain(..).zip(senders) {
            // A closed receiver means that worker's task already stopped.
            let _ = stream_tx.send(sender);
        }
        Ok(())
    }

    pub fn counters(&self) -> &Arc<RelayCounters> {
        &self.counters
    }

    fn runtime(&self) -> &tokio::runtime::Runtime {
        self.runtime
            .as_ref()
            .expect("relay runtime lives until drop")
    }
}

impl Drop for PoolRelay {
    fn drop(&mut self) {
        if let Some(runtime) = self.runtime.take() {
            runtime.shutdown_background();
        }
    }
}

/// Wait for the core's peer file.
pub async fn read_peer_file(path: &Path, timeout: Duration) -> anyhow::Result<PeerInfo> {
    let deadline = Instant::now() + timeout;
    loop {
        match std::fs::read(path) {
            Ok(bytes) => {
                return rmp_serde::from_slice(&bytes).with_context(|| {
                    format!("peer file {} is not velo peer info", path.display())
                });
            }
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
                anyhow::ensure!(
                    Instant::now() < deadline,
                    "no peer file at {} after {timeout:?}",
                    path.display()
                );
                tokio::time::sleep(Duration::from_millis(200)).await;
            }
            Err(error) => return Err(error.into()),
        }
    }
}

/// Convert one worker's batches and send each as one record, in order.
async fn relay_worker(
    worker_id: u64,
    block_size: u32,
    mut batches: mpsc::UnboundedReceiver<Vec<RawKvEvent>>,
    stream: oneshot::Receiver<StreamSender<Bytes>>,
    counters: Arc<RelayCounters>,
) {
    let Ok(sender) = stream.await else {
        return;
    };
    let mut normalizer = ZmqEventNormalizer::new(block_size);
    let worker = WorkerWithDpRank::new(worker_id, 0);
    let mut seq = 0u64;
    let mut events = Vec::new();
    while let Some(batch) = batches.recv().await {
        counters.pending_batches.fetch_sub(1, Ordering::Relaxed);
        if let Err(error) = convert(
            &mut normalizer,
            worker,
            seq,
            &batch,
            block_size,
            &mut events,
        ) {
            tracing::warn!(worker_id, %error, "relay conversion failed; batch dropped");
            continue;
        }
        seq += 1;
        if events.is_empty() {
            continue;
        }
        match encode_record(&events) {
            Ok(item) => {
                let bytes = item.payload().len() as u64;
                if let Err(error) = sender
                    .send_frame(item.finish().expect("record frame"))
                    .await
                {
                    counters.send_errors.fetch_add(1, Ordering::Relaxed);
                    tracing::error!(worker_id, %error, "relay stream send failed; worker relay stopped");
                    return;
                }
                counters.records.fetch_add(1, Ordering::Relaxed);
                counters.record_bytes.fetch_add(bytes, Ordering::Relaxed);
                counters
                    .router_events
                    .fetch_add(events.len() as u64, Ordering::Relaxed);
            }
            Err(error) => tracing::warn!(worker_id, %error, "relay record encode failed"),
        }
        events.clear();
    }
}

/// The router events the core's ZMQ listener would apply for `batch`.
fn convert(
    normalizer: &mut ZmqEventNormalizer,
    worker: WorkerWithDpRank,
    seq: u64,
    batch: &[RawKvEvent],
    block_size: u32,
    out: &mut Vec<RouterEvent>,
) -> anyhow::Result<()> {
    let Some(payload) = encode_event_batch(batch, block_size, worker.dp_rank)? else {
        return Ok(());
    };
    let decoded = decode_event_batch(&payload)?;
    for raw in decoded.events {
        if let Some(placement) = normalizer.normalize(raw, seq, worker) {
            out.push(
                placement
                    .into_router_event()
                    .context("local worker placement must convert to a router event")?,
            );
        }
    }
    Ok(())
}

pub fn encode_record<T: Serialize>(
    events: &[T],
) -> Result<ByteItemBuilder, rmp_serde::encode::Error> {
    let mut item = ByteItemBuilder::with_capacity(64 * events.len().max(1));
    rmp_serde::encode::write_named(item.buffer(), events)?;
    Ok(item)
}

#[cfg(test)]
mod tests {
    use super::*;
    use dynamo_kv_router::protocols::{
        ExternalSequenceBlockHash, KvCacheEvent, KvCacheEventData, KvCacheRemoveData,
        KvCacheStoreData, KvCacheStoredBlockData, LocalBlockHash, StorageTier,
    };

    fn stored(block_hashes: &[u64], parent: Option<u64>, block_size: u32) -> RawKvEvent {
        RawKvEvent {
            event: KvCacheEvent {
                event_id: 0,
                data: KvCacheEventData::Stored(KvCacheStoreData {
                    parent_hash: parent.map(ExternalSequenceBlockHash),
                    start_position: None,
                    blocks: block_hashes
                        .iter()
                        .map(|&hash| KvCacheStoredBlockData {
                            block_hash: ExternalSequenceBlockHash(hash),
                            tokens_hash: LocalBlockHash(0),
                            mm_extra_info: None,
                        })
                        .collect(),
                }),
                dp_rank: 0,
            },
            block_token_ids: Some(
                block_hashes
                    .iter()
                    .map(|&hash| (0..block_size).map(|i| (hash as u32) ^ i).collect())
                    .collect(),
            ),
            storage_tier: StorageTier::Device,
            session_id: None,
        }
    }

    fn removed(block_hashes: &[u64]) -> RawKvEvent {
        RawKvEvent {
            event: KvCacheEvent {
                event_id: 0,
                data: KvCacheEventData::Removed(KvCacheRemoveData {
                    block_hashes: block_hashes
                        .iter()
                        .map(|&hash| ExternalSequenceBlockHash(hash))
                        .collect(),
                }),
                dp_rank: 0,
            },
            block_token_ids: None,
            storage_tier: StorageTier::Device,
            session_id: None,
        }
    }

    /// A relay record decodes to exactly the events the core's ZMQ listener
    /// derives from the same batch, so both arms index the same thing.
    #[test]
    fn relay_record_matches_direct_listener_events() {
        let block_size = 4;
        let worker = WorkerWithDpRank::new(7, 0);
        let batch = vec![
            stored(&[11, 12], None, block_size),
            stored(&[13], Some(12), block_size),
            removed(&[11]),
        ];

        // Direct path: the vLLM ZMQ payload, decoded and normalized by the listener's steps.
        let payload = encode_event_batch(&batch, block_size, 0).unwrap().unwrap();
        let mut listener = ZmqEventNormalizer::new(block_size);
        let direct: Vec<RouterEvent> = decode_event_batch(&payload)
            .unwrap()
            .events
            .into_iter()
            .filter_map(|raw| listener.normalize(raw, 3, worker))
            .map(|placement| placement.into_router_event().unwrap())
            .collect();
        assert_eq!(direct.len(), 3);

        let mut relay = ZmqEventNormalizer::new(block_size);
        let mut events = Vec::new();
        convert(&mut relay, worker, 3, &batch, block_size, &mut events).unwrap();
        let record = encode_record(&events).unwrap();
        let received: Vec<RouterEvent> = rmp_serde::from_slice(record.payload()).unwrap();
        assert_eq!(received, direct);
    }
}
