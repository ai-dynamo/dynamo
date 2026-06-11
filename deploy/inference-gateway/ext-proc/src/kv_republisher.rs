// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! ZMQ KV-event republisher for router-only ("on-ramp") mode.
//!
//! In router-only mode the EPP fronts raw `vllm serve` pods that publish their
//! native KV-cache events on a ZMQ PUB socket (`--kv-events-config`). There is
//! no Dynamo event-plane bridge, so this module subscribes to each pod's PUB
//! socket directly, normalizes the native vLLM events into [`RouterEvent`]s
//! (stamped with the pod's `worker_id`), and **re-publishes** them onto the
//! embedded router's own ZMQ event plane via [`EventPublisher`].
//!
//! The embedded `KvRouter`'s existing event-plane subscriber then ingests them
//! exactly as it would Dynamo-worker events — so this needs **no changes to any
//! code under `lib/llm/src/kv_router/`**. The decode + normalization reuse the
//! public `dynamo_kv_router::zmq_wire` primitives; only the refcount dedup
//! (which is `pub(crate)` inside `lib/llm`) is reimplemented here.
//!
//! NOTE: this module is not yet wired into [`crate::epp::Router`]; the
//! router-only constructor that drives it lands in a subsequent change.

use std::collections::HashMap;
use std::collections::hash_map::Entry;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Duration;

use anyhow::Result;
use futures::StreamExt;
use tokio_util::sync::CancellationToken;

use dynamo_kv_router::protocols::{
    DpRank, ExternalSequenceBlockHash, KV_EVENT_SUBJECT, KvCacheEventData, KvCacheRemoveData,
    KvCacheStoreData, PlacementEvent, RouterEvent, StorageTier, WorkerId, WorkerWithDpRank,
};
use dynamo_kv_router::zmq_wire::{ZmqEventNormalizer, decode_event_batch};
use dynamo_runtime::component::Component;
use dynamo_runtime::discovery::EventTransportKind;
use dynamo_runtime::transports::event_plane::EventPublisher;

/// Backoff before reconnecting a dropped per-pod listener.
const RECONNECT_BACKOFF: Duration = Duration::from_secs(2);

/// vLLM publishes KV events as a 3-frame multipart message: `[topic, seq, payload]`.
const VLLM_FRAME_COUNT: usize = 3;

/// Republishes raw per-pod vLLM ZMQ KV-cache events onto the embedded router's
/// ZMQ event plane.
///
/// One [`EventPublisher`] (bound on the router's component + [`KV_EVENT_SUBJECT`])
/// is shared across all per-pod listeners; the router's event-plane subscriber
/// discovers it through the (in-process, `mem`) discovery backend.
pub struct KvRepublisher {
    publisher: Arc<EventPublisher>,
    block_size: u32,
    kv_event_port: u16,
    kv_event_topic: String,
}

impl KvRepublisher {
    /// Bind the shared event-plane publisher on `component`'s `KV_EVENT_SUBJECT`
    /// topic using the ZMQ transport. The embedded router subscribes to the
    /// same component+topic, so republished events flow straight into its index.
    pub async fn new(
        component: &Component,
        block_size: u32,
        kv_event_port: u16,
        kv_event_topic: String,
    ) -> Result<Self> {
        let publisher = EventPublisher::for_component_with_transport(
            component,
            KV_EVENT_SUBJECT,
            EventTransportKind::Zmq,
        )
        .await?;
        Ok(Self {
            publisher: Arc::new(publisher),
            block_size,
            kv_event_port,
            kv_event_topic,
        })
    }

    /// Start a per-pod listener that subscribes to `pod_ip`'s vLLM ZMQ PUB
    /// socket and republishes its KV events stamped with `worker_id`.
    ///
    /// Reconnects with backoff until the returned [`CancellationToken`] is
    /// cancelled (e.g. when the pod is removed from the reflector). The caller
    /// owns the token and should cancel it on pod deletion.
    pub fn register_worker(&self, worker_id: WorkerId, pod_ip: &str) -> CancellationToken {
        let endpoint = format!("tcp://{}:{}", pod_ip, self.kv_event_port);
        let topic = self.kv_event_topic.clone();
        let publisher = self.publisher.clone();
        let block_size = self.block_size;
        let token = CancellationToken::new();
        let listen = token.clone();

        tokio::spawn(async move {
            // Monotonic event-id source for the normalizer, shared across
            // reconnects so replayed batches keep increasing ids.
            let next_event_id = Arc::new(AtomicU64::new(0));
            tracing::info!(worker_id, endpoint = %endpoint, "Starting vLLM KV-event listener");

            while !listen.is_cancelled() {
                if let Err(error) = run_listener(
                    &endpoint,
                    &topic,
                    worker_id,
                    block_size,
                    &publisher,
                    &next_event_id,
                    &listen,
                )
                .await
                {
                    tracing::warn!(worker_id, endpoint = %endpoint, %error, "vLLM KV-event listener error; backing off");
                }
                if listen.is_cancelled() {
                    break;
                }
                tokio::select! {
                    _ = listen.cancelled() => break,
                    _ = tokio::time::sleep(RECONNECT_BACKOFF) => {}
                }
            }
            tracing::debug!(worker_id, endpoint = %endpoint, "vLLM KV-event listener stopped");
        });

        token
    }
}

/// Connect a SUB socket to a single pod's vLLM PUB endpoint and pump its KV
/// events onto the event plane until the stream ends or `cancel` fires.
async fn run_listener(
    endpoint: &str,
    topic: &str,
    worker_id: WorkerId,
    block_size: u32,
    publisher: &EventPublisher,
    next_event_id: &AtomicU64,
    cancel: &CancellationToken,
) -> Result<()> {
    use tmq::{Context, subscribe::subscribe};

    let ctx = Context::new();
    let mut socket = subscribe(&ctx)
        .connect(endpoint)?
        .subscribe(topic.as_bytes())?;

    let mut normalizer = ZmqEventNormalizer::new(block_size);
    let mut dedup = RefcountDedup::new();

    loop {
        tokio::select! {
            biased;
            _ = cancel.cancelled() => return Ok(()),
            msg = socket.next() => {
                let Some(msg) = msg else {
                    // SUB stream ended; let the caller back off and reconnect.
                    return Ok(());
                };
                let multipart = msg?;
                let frames: Vec<Vec<u8>> = multipart.into_iter().map(|f| f.to_vec()).collect();
                if frames.len() != VLLM_FRAME_COUNT {
                    tracing::warn!(
                        worker_id,
                        frame_count = frames.len(),
                        "Unexpected vLLM KV-event frame count (expected 3); skipping"
                    );
                    continue;
                }

                // frames = [topic, seq, payload]; the event-plane publisher
                // assigns its own sequence, so we only need the payload.
                let payload = &frames[VLLM_FRAME_COUNT - 1];
                let batch = match decode_event_batch(payload) {
                    Ok(batch) => batch,
                    Err(error) => {
                        tracing::warn!(worker_id, %error, "Failed to decode vLLM KvEventBatch; skipping");
                        continue;
                    }
                };

                let dp_rank: DpRank = batch.data_parallel_rank.unwrap_or(0).cast_unsigned();
                let worker = WorkerWithDpRank::new(worker_id, dp_rank);

                for raw in batch.events {
                    let event_id = next_event_id.fetch_add(1, Ordering::SeqCst);
                    let Some(placement) = normalizer.normalize(raw, event_id, worker) else {
                        continue;
                    };
                    let Some(router_event) = dedup.process(placement) else {
                        continue;
                    };
                    if let Err(error) = publisher.publish(&router_event).await {
                        tracing::warn!(worker_id, %error, "Failed to republish KV event onto event plane");
                    }
                }
            }
        }
    }
}

/// Reference-counting dedup for the per-pod ZMQ KV-event path.
///
/// vLLM can emit duplicate store/remove events; applying them straight to the
/// router index corrupts block refcounts (a remove could fire before the worker
/// actually evicts the block). This reimplements — for the router-only EPP crate —
/// the same refcount-correct dedup that `lib/llm`'s `EventDedupFilter` applies
/// on the Dynamo-worker path (which is `pub(crate)` and not reachable here).
///
/// Refcounts are tracked **per (DP rank, storage tier)** because identical block
/// hashes on different ranks/tiers are independent blocks.
struct RefcountDedup {
    per_rank_tier: HashMap<(DpRank, StorageTier), HashMap<ExternalSequenceBlockHash, usize>>,
}

impl RefcountDedup {
    fn new() -> Self {
        Self {
            per_rank_tier: HashMap::new(),
        }
    }

    /// Run a normalized placement event through dedup and convert it to the
    /// `RouterEvent` to publish, or `None` when the event is fully filtered (a
    /// duplicate remove) or carries no local-worker placement.
    fn process(&mut self, mut event: PlacementEvent) -> Option<RouterEvent> {
        let dp_rank = event.event.dp_rank;
        let tier = event.placement.tier;
        // Take the data out to dedup, then restore (mirrors lib/llm PerWorkerDedup).
        let data = std::mem::replace(&mut event.event.data, KvCacheEventData::Cleared);
        let new_data = match data {
            KvCacheEventData::Stored(store) => {
                self.track_store(dp_rank, tier, &store);
                KvCacheEventData::Stored(store)
            }
            KvCacheEventData::Removed(remove) => {
                KvCacheEventData::Removed(self.filter_remove(dp_rank, tier, remove)?)
            }
            KvCacheEventData::Cleared => {
                self.clear();
                KvCacheEventData::Cleared
            }
        };
        event.event.data = new_data;
        event.into_router_event()
    }

    fn track_store(&mut self, dp_rank: DpRank, tier: StorageTier, data: &KvCacheStoreData) {
        let refcounts = self.per_rank_tier.entry((dp_rank, tier)).or_default();
        for block in &data.blocks {
            *refcounts.entry(block.block_hash).or_insert(0) += 1;
        }
    }

    fn filter_remove(
        &mut self,
        dp_rank: DpRank,
        tier: StorageTier,
        mut data: KvCacheRemoveData,
    ) -> Option<KvCacheRemoveData> {
        let refcounts = self.per_rank_tier.entry((dp_rank, tier)).or_default();
        data.block_hashes
            .retain(|hash| match refcounts.entry(*hash) {
                Entry::Occupied(mut entry) => {
                    *entry.get_mut() -= 1;
                    if *entry.get() == 0 {
                        entry.remove();
                        true // refcount hit 0 -> pass through
                    } else {
                        false // still referenced -> filter out
                    }
                }
                // Not tracked -> pass through defensively.
                Entry::Vacant(_) => true,
            });
        if data.block_hashes.is_empty() {
            None
        } else {
            Some(data)
        }
    }

    fn clear(&mut self) {
        self.per_rank_tier.clear();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use dynamo_kv_router::protocols::{
        KvCacheEvent, KvCacheStoredBlockData, LocalBlockHash, Placement,
    };

    fn stored(hashes: &[u64]) -> KvCacheEventData {
        KvCacheEventData::Stored(KvCacheStoreData {
            parent_hash: None,
            start_position: None,
            blocks: hashes
                .iter()
                .map(|h| KvCacheStoredBlockData {
                    block_hash: ExternalSequenceBlockHash(*h),
                    tokens_hash: LocalBlockHash(*h),
                    mm_extra_info: None,
                })
                .collect(),
        })
    }

    fn removed(hashes: &[u64]) -> KvCacheEventData {
        KvCacheEventData::Removed(KvCacheRemoveData {
            block_hashes: hashes
                .iter()
                .map(|h| ExternalSequenceBlockHash(*h))
                .collect(),
        })
    }

    fn placement(data: KvCacheEventData) -> PlacementEvent {
        PlacementEvent::new(
            Placement::local_gpu(7, 0),
            KvCacheEvent {
                event_id: 0,
                data,
                dp_rank: 0,
            },
        )
    }

    /// Stores always pass through; a remove only passes once its refcount hits
    /// zero — so a block stored twice survives the first remove and is evicted
    /// only on the second. This is the refcount invariant the router index
    /// relies on; if it regresses, duplicate vLLM events corrupt routing state.
    #[test]
    fn remove_passes_only_when_refcount_reaches_zero() {
        let mut dedup = RefcountDedup::new();

        assert!(dedup.process(placement(stored(&[1, 2]))).is_some());
        assert!(dedup.process(placement(stored(&[1]))).is_some()); // block 1 stored twice

        // First remove of block 1: still referenced -> fully filtered -> None.
        assert!(dedup.process(placement(removed(&[1]))).is_none());
        // Second remove of block 1: refcount hits 0 -> passes through.
        assert!(dedup.process(placement(removed(&[1]))).is_some());
        // Block 2 was stored once -> first remove passes.
        assert!(dedup.process(placement(removed(&[2]))).is_some());
    }

    /// A remove for an untracked hash passes through defensively (we never
    /// suppress eviction of something we didn't see stored).
    #[test]
    fn untracked_remove_passes_through() {
        let mut dedup = RefcountDedup::new();
        assert!(dedup.process(placement(removed(&[99]))).is_some());
    }

    /// Cleared resets all refcounts, so a subsequent remove of a
    /// previously-stored block is treated as untracked (passes through).
    #[test]
    fn cleared_resets_refcounts() {
        let mut dedup = RefcountDedup::new();
        assert!(dedup.process(placement(stored(&[1, 1, 1]))).is_some());
        assert!(
            dedup
                .process(placement(KvCacheEventData::Cleared))
                .is_some()
        );
        // After clear, refcount for 1 is gone -> remove passes through.
        assert!(dedup.process(placement(removed(&[1]))).is_some());
    }
}
