// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! One pool's KV relay for the Global Router benchmark's velo arm.
//!
//! It subscribes to its pool's workers on Dynamo's event plane (the same
//! `kv-events` publishers the pool's frontend reads), and forwards each
//! envelope's payload, msgpack `Vec<RouterEvent>`, unchanged onto that worker's
//! velo stream to `gr_core_host`. One stream per worker keeps each worker's
//! events in order; velo's mux packs many streams into each write.
//!
//! Prototype limits: no gap recovery (gaps in a publisher's sequence are
//! counted and make the run invalid), and the worker set is fixed once it is
//! complete. Once a second it appends counters to `--stats-out`.

#[path = "gr_relay/common.rs"]
mod relay_common;
#[path = "gr_relay/peer_file.rs"]
mod relay_peer;

use std::io::Write as _;
use std::net::SocketAddr;
use std::path::PathBuf;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use anyhow::Context as _;
use bytes::Bytes;
use clap::Parser;
use dynamo_kv_router::protocols::{KV_EVENT_SUBJECT, RouterEvent};
use dynamo_runtime::transports::event_plane::EventSubscriber;
use dynamo_runtime::{DistributedRuntime, Runtime, Worker};
use rustc_hash::FxHashMap;
use tokio::sync::{mpsc, oneshot};
use velo::{StreamAnchorHandle, StreamSender, VeloMetrics};

use relay_common::{OPEN_HANDLER, OpenReply, OpenRequest, VeloOptions, build_velo};
use relay_peer::read_peer_file;

#[derive(Debug, Clone, Parser)]
#[command(about = "Forward one pool's event-plane KV events to the Global Router over velo")]
struct Args {
    /// The pool's Dynamo namespace; its workers serve `<namespace>.backend.generate`.
    #[arg(long)]
    namespace: String,
    /// The pool's index at the Global Router.
    #[arg(long)]
    pool: u32,
    /// Workers to wait for before opening streams.
    #[arg(long, default_value_t = 128)]
    workers: usize,
    #[arg(long)]
    peer_file: PathBuf,
    #[arg(long)]
    velo_bind: SocketAddr,
    #[arg(long, default_value_t = 4)]
    velo_lanes: u16,
    #[arg(long, default_value_t = 2)]
    velo_linger_ms: u64,
    /// Touched once every stream is open.
    #[arg(long)]
    ready_file: Option<PathBuf>,
    #[arg(long)]
    stats_out: PathBuf,
    #[arg(long, default_value_t = 900)]
    wait_secs: u64,
}

#[derive(Default)]
struct RelayStats {
    envelopes: AtomicU64,
    payload_bytes: AtomicU64,
    records: AtomicU64,
    pending: AtomicU64,
    gaps: AtomicU64,
    unknown_workers: AtomicU64,
    decode_errors: AtomicU64,
    subscriber_errors: AtomicU64,
    send_errors: AtomicU64,
}

impl RelayStats {
    fn snapshot(&self) -> serde_json::Value {
        let get = |c: &AtomicU64| c.load(Ordering::Relaxed);
        serde_json::json!({
            "envelopes": get(&self.envelopes),
            "payload_bytes": get(&self.payload_bytes),
            "records": get(&self.records),
            "pending": get(&self.pending),
            "gaps": get(&self.gaps),
            "unknown_workers": get(&self.unknown_workers),
            "decode_errors": get(&self.decode_errors),
            "subscriber_errors": get(&self.subscriber_errors),
            "send_errors": get(&self.send_errors),
        })
    }
}

struct Publisher {
    worker: u64,
    last_sequence: u64,
}

fn main() -> anyhow::Result<()> {
    let args = Args::parse();
    dynamo_runtime::logging::init();
    Worker::from_settings()?.execute(move |runtime| app(runtime, args))
}

async fn app(runtime: Runtime, args: Args) -> anyhow::Result<()> {
    let drt = DistributedRuntime::from_settings(runtime).await?;
    let endpoint = drt
        .namespace(&args.namespace)?
        .component("backend")?
        .endpoint("generate");
    // Subscribe before anything else, so no event published after a worker
    // registers is missed.
    let mut subscriber = EventSubscriber::for_endpoint_id(&drt, &endpoint.id(), KV_EVENT_SUBJECT)
        .await
        .context("event-plane subscription")?;
    let client = endpoint.client().await?;
    let deadline = Instant::now() + Duration::from_secs(args.wait_secs);
    let mut worker_ids = loop {
        let ids = client.instance_ids();
        if ids.len() >= args.workers {
            break ids;
        }
        anyhow::ensure!(
            Instant::now() < deadline,
            "{} has {} of {} workers after {}s",
            args.namespace,
            ids.len(),
            args.workers,
            args.wait_secs
        );
        tokio::time::sleep(Duration::from_millis(500)).await;
    };
    worker_ids.sort_unstable();

    let stats = Arc::new(RelayStats::default());
    let registry = prometheus::Registry::new();
    let metrics = Arc::new(VeloMetrics::register(&registry)?);
    {
        let stats = Arc::clone(&stats);
        let path = args.stats_out.clone();
        std::thread::Builder::new()
            .name("relay-stats".to_string())
            .spawn(move || {
                if let Err(error) = write_stats(&path, &stats) {
                    tracing::error!(%error, "stats writer stopped");
                }
            })?;
    }

    // One ordered queue and forwarding task per worker; each waits for its stream.
    let mut queues: FxHashMap<u64, mpsc::UnboundedSender<Bytes>> = FxHashMap::default();
    let mut stream_slots = Vec::with_capacity(worker_ids.len());
    for &worker in &worker_ids {
        let (tx, rx) = mpsc::unbounded_channel();
        let (stream_tx, stream_rx) = oneshot::channel();
        tokio::spawn(forward(worker, rx, stream_rx, Arc::clone(&stats)));
        queues.insert(worker, tx);
        stream_slots.push(stream_tx);
    }

    let velo = build_velo(
        VeloOptions {
            bind: args.velo_bind,
            lanes: args.velo_lanes,
            linger_ms: args.velo_linger_ms,
        },
        Some(metrics),
    )
    .await?;
    let core = read_peer_file(&args.peer_file, Duration::from_secs(args.wait_secs)).await?;
    let core_id = core.instance_id();
    velo.register_peer(core)?;
    tokio::time::timeout(
        Duration::from_secs(args.wait_secs),
        velo.wait_for_handler(core_id, OPEN_HANDLER),
    )
    .await
    .context("the Global Router's relay handler did not appear")??;
    let request = rmp_serde::to_vec(&OpenRequest {
        worker_ids: worker_ids.clone(),
        pool: args.pool,
    })?;
    let reply = velo
        .unary(OPEN_HANDLER)?
        .raw_payload(Bytes::from(request))
        .instance(core_id)
        .send()
        .await?;
    let reply: OpenReply = rmp_serde::from_slice(&reply)?;
    anyhow::ensure!(
        reply.handles.len() == worker_ids.len(),
        "core opened {} streams for {} workers",
        reply.handles.len(),
        worker_ids.len()
    );
    let senders = futures_util::future::try_join_all(
        reply
            .handles
            .iter()
            .map(|&handle| velo.attach_anchor::<Bytes>(StreamAnchorHandle::from_u128(handle))),
    )
    .await?;
    for (slot, sender) in stream_slots.into_iter().zip(senders) {
        let _ = slot.send(sender);
    }
    if let Some(path) = &args.ready_file {
        std::fs::write(path, format!("{}\n", worker_ids.len()))?;
    }
    tracing::info!(namespace = %args.namespace, workers = worker_ids.len(), "relay streams open");

    let mut publishers: FxHashMap<u64, Publisher> = FxHashMap::default();
    while let Some(next) = subscriber.next().await {
        let envelope = match next {
            Ok(envelope) => envelope,
            Err(error) => {
                stats.subscriber_errors.fetch_add(1, Ordering::Relaxed);
                tracing::warn!(%error, "event-plane receive failed");
                continue;
            }
        };
        stats.envelopes.fetch_add(1, Ordering::Relaxed);
        stats
            .payload_bytes
            .fetch_add(envelope.payload.len() as u64, Ordering::Relaxed);
        let worker = match publishers.get_mut(&envelope.publisher_id) {
            Some(publisher) => {
                if envelope.sequence > publisher.last_sequence + 1 {
                    stats.gaps.fetch_add(
                        envelope.sequence - publisher.last_sequence - 1,
                        Ordering::Relaxed,
                    );
                }
                publisher.last_sequence = publisher.last_sequence.max(envelope.sequence);
                publisher.worker
            }
            None => {
                // Every event from one publisher belongs to one worker.
                let worker = match rmp_serde::from_slice::<Vec<RouterEvent>>(&envelope.payload) {
                    Ok(events) => match events.first() {
                        Some(event) => event.worker_id,
                        None => continue,
                    },
                    Err(error) => {
                        stats.decode_errors.fetch_add(1, Ordering::Relaxed);
                        tracing::warn!(%error, "undecodable event-plane payload");
                        continue;
                    }
                };
                publishers.insert(
                    envelope.publisher_id,
                    Publisher {
                        worker,
                        last_sequence: envelope.sequence,
                    },
                );
                worker
            }
        };
        match queues.get(&worker) {
            Some(queue) => {
                stats.pending.fetch_add(1, Ordering::Relaxed);
                if queue.send(envelope.payload).is_err() {
                    stats.pending.fetch_sub(1, Ordering::Relaxed);
                    stats.send_errors.fetch_add(1, Ordering::Relaxed);
                }
            }
            None => {
                stats.unknown_workers.fetch_add(1, Ordering::Relaxed);
            }
        }
    }
    anyhow::bail!("event-plane subscription ended")
}

/// Send one worker's payloads, in order, once its stream is open.
async fn forward(
    worker: u64,
    mut payloads: mpsc::UnboundedReceiver<Bytes>,
    stream: oneshot::Receiver<StreamSender<Bytes>>,
    stats: Arc<RelayStats>,
) {
    let Ok(sender) = stream.await else {
        return;
    };
    while let Some(payload) = payloads.recv().await {
        stats.pending.fetch_sub(1, Ordering::Relaxed);
        if let Err(error) = sender.send_bytes(&payload).await {
            stats.send_errors.fetch_add(1, Ordering::Relaxed);
            tracing::error!(worker, %error, "relay stream send failed; worker relay stopped");
            return;
        }
        stats.records.fetch_add(1, Ordering::Relaxed);
    }
}

fn write_stats(path: &std::path::Path, stats: &RelayStats) -> anyhow::Result<()> {
    let mut file = std::io::BufWriter::new(std::fs::File::create(path)?);
    loop {
        std::thread::sleep(Duration::from_secs(1));
        let mut line = stats.snapshot();
        line["t"] = serde_json::json!(SystemTime::now().duration_since(UNIX_EPOCH)?.as_secs_f64());
        serde_json::to_writer(&mut file, &line)?;
        file.write_all(b"\n")?;
        file.flush()?;
    }
}

#[cfg(test)]
mod tests {
    use dynamo_kv_router::protocols::{
        ExternalSequenceBlockHash, KvCacheEvent, KvCacheEventData, KvCacheRemoveData,
        KvCacheStoreData, KvCacheStoredBlockData, LocalBlockHash, RouterEvent,
    };
    use dynamo_runtime::transports::event_plane::MsgpackCodec;

    /// The relay forwards event-plane payload bytes unchanged; the Global Router
    /// decodes them as `Vec<RouterEvent>`, so the publisher's encoding must round-trip.
    #[test]
    fn event_plane_payload_decodes_as_router_events() {
        let stored = RouterEvent::new(
            7,
            KvCacheEvent {
                event_id: 3,
                data: KvCacheEventData::Stored(KvCacheStoreData {
                    parent_hash: Some(ExternalSequenceBlockHash(11)),
                    start_position: None,
                    blocks: vec![KvCacheStoredBlockData {
                        block_hash: ExternalSequenceBlockHash(12),
                        tokens_hash: LocalBlockHash(99),
                        mm_extra_info: None,
                    }],
                }),
                dp_rank: 0,
            },
        );
        let removed = RouterEvent::new(
            7,
            KvCacheEvent {
                event_id: 4,
                data: KvCacheEventData::Removed(KvCacheRemoveData {
                    block_hashes: vec![ExternalSequenceBlockHash(12)],
                }),
                dp_rank: 0,
            },
        );
        let events = vec![stored, removed];
        let payload = MsgpackCodec.encode_payload(&events).unwrap();
        let decoded: Vec<RouterEvent> = rmp_serde::from_slice(&payload).unwrap();
        assert_eq!(decoded, events);
        assert_eq!(decoded[0].worker_id, 7, "the relay reads the worker from the first event");
    }
}
