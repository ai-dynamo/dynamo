// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::sync::Arc;

use anyhow::Result;
use tokio_util::sync::CancellationToken;

use crate::ConcurrentRadixTreeCompressed;
use crate::ThreadPoolIndexer;
use crate::indexer::{
    KvIndexer, KvIndexerInterface, KvIndexerMetrics, LowerTierIndexers, MatchDetails,
    TieredMatchDetails, query_lower_tiers,
};
use crate::protocols::{KvCacheEventData, LocalBlockHash, OverlapScores, RouterEvent, WorkerId};

/// Block-content indexer wrapping a primary device-tier backend plus a
/// per-tier registry of lower-tier indexers (host-pinned, disk, …).
///
/// Both standalone-indexer variants own a `lower_tier: LowerTierIndexers` so
/// tier-tagged events get routed alongside device events and tier-aware
/// queries can return per-tier hit counts.
#[derive(Clone)]
pub enum Indexer {
    Single {
        primary: KvIndexer,
        lower_tier: LowerTierIndexers,
    },
    Concurrent {
        primary: Arc<ThreadPoolIndexer<ConcurrentRadixTreeCompressed>>,
        lower_tier: LowerTierIndexers,
    },
}

impl Indexer {
    /// Apply an event without tier dispatch — kept for callers that have
    /// already determined this is a device-tier event. Most callers should
    /// use [`Self::apply_event_routed`].
    pub async fn apply_event(&self, event: RouterEvent) {
        match self {
            Indexer::Single { primary, .. } => primary.apply_event(event).await,
            Indexer::Concurrent { primary, .. } => primary.apply_event(event).await,
        }
    }

    /// Apply an event, routing to the device-tier primary when
    /// `event.storage_tier.is_gpu()` and to the appropriate lower-tier
    /// indexer otherwise. `Cleared` events fan out to every tier so per-tier
    /// state stays consistent with the primary.
    pub async fn apply_event_routed(&self, event: RouterEvent) {
        match self {
            Indexer::Single {
                primary,
                lower_tier,
            } => match &event.event.data {
                KvCacheEventData::Cleared => {
                    primary.apply_event(event.clone()).await;
                    for indexer in lower_tier.all() {
                        indexer.apply_event(event.clone()).await;
                    }
                }
                _ if event.storage_tier.is_gpu() => {
                    primary.apply_event(event).await;
                }
                _ => {
                    lower_tier
                        .get_or_create(event.storage_tier)
                        .apply_event(event)
                        .await;
                }
            },
            Indexer::Concurrent {
                primary,
                lower_tier,
            } => match &event.event.data {
                KvCacheEventData::Cleared => {
                    primary.apply_event(event.clone()).await;
                    for indexer in lower_tier.all() {
                        indexer.apply_event(event.clone()).await;
                    }
                }
                _ if event.storage_tier.is_gpu() => {
                    primary.apply_event(event).await;
                }
                _ => {
                    lower_tier
                        .get_or_create(event.storage_tier)
                        .apply_event(event)
                        .await;
                }
            },
        }
    }

    pub async fn remove_worker(&self, worker_id: WorkerId) {
        match self {
            Indexer::Single {
                primary,
                lower_tier,
            } => {
                for indexer in lower_tier.all() {
                    indexer.remove_worker(worker_id).await;
                }
                primary.remove_worker(worker_id).await;
            }
            Indexer::Concurrent {
                primary,
                lower_tier,
            } => {
                for indexer in lower_tier.all() {
                    indexer.remove_worker(worker_id).await;
                }
                primary.remove_worker(worker_id).await;
            }
        }
    }

    pub async fn remove_worker_dp_rank(&self, worker_id: WorkerId, dp_rank: u32) {
        match self {
            Indexer::Single {
                primary,
                lower_tier,
            } => {
                for indexer in lower_tier.all() {
                    indexer.remove_worker_dp_rank(worker_id, dp_rank).await;
                }
                primary.remove_worker_dp_rank(worker_id, dp_rank).await;
            }
            Indexer::Concurrent {
                primary,
                lower_tier,
            } => {
                for indexer in lower_tier.all() {
                    indexer.remove_worker_dp_rank(worker_id, dp_rank).await;
                }
                primary.remove_worker_dp_rank(worker_id, dp_rank).await;
            }
        }
    }

    /// Device-tier overlap scores. Existing flat-shape callers continue to use
    /// this; tier-aware callers should use [`Self::find_tiered_matches`].
    pub async fn find_matches(&self, hashes: Vec<LocalBlockHash>) -> Result<OverlapScores> {
        match self {
            Indexer::Single { primary, .. } => {
                primary.find_matches(hashes).await.map_err(Into::into)
            }
            Indexer::Concurrent { primary, .. } => {
                primary.find_matches(hashes).await.map_err(Into::into)
            }
        }
    }

    /// Device match details + per-tier hits, suitable for building the
    /// Mooncake-RFC-shape per-instance breakdown.
    pub async fn find_tiered_matches(
        &self,
        sequence: Vec<LocalBlockHash>,
    ) -> Result<TieredMatchDetails> {
        match self {
            Indexer::Single {
                primary,
                lower_tier,
            } => {
                let device = primary.find_match_details(sequence.clone()).await?;
                let lt = query_lower_tiers(lower_tier, &sequence, &device);
                Ok(TieredMatchDetails {
                    device,
                    lower_tier: lt,
                })
            }
            Indexer::Concurrent {
                primary,
                lower_tier,
            } => {
                let device: MatchDetails =
                    primary.backend().find_match_details_impl(&sequence, false);
                let lt = query_lower_tiers(lower_tier, &sequence, &device);
                Ok(TieredMatchDetails {
                    device,
                    lower_tier: lt,
                })
            }
        }
    }

    /// Dump every event tracked by this indexer in a form replayable by a
    /// peer's [`Self::apply_event_routed`].
    ///
    /// Returns the primary device-tier dump first, followed by every allocated
    /// lower-tier indexer's dump with each event's `storage_tier` retagged to
    /// the tier it lives in. The `LowerTierIndexer::dump_events` synthetic
    /// events default `storage_tier` to `Device`, so without retagging a peer
    /// would replay them into the wrong slot.
    pub async fn dump_events(&self) -> Result<Vec<RouterEvent>> {
        let (primary_events, lower_tier_entries) = match self {
            Indexer::Single {
                primary,
                lower_tier,
            } => (
                primary.dump_events().await.map_err(anyhow::Error::from)?,
                lower_tier.entries(),
            ),
            Indexer::Concurrent {
                primary,
                lower_tier,
            } => (
                primary.dump_events().await.map_err(anyhow::Error::from)?,
                lower_tier.entries(),
            ),
        };

        let mut out = primary_events;
        for (tier, indexer) in lower_tier_entries {
            let events = indexer.dump_events().await.map_err(anyhow::Error::from)?;
            for mut event in events {
                event.storage_tier = tier;
                out.push(event);
            }
        }
        Ok(out)
    }
}

#[cfg(test)]
pub(crate) mod test_util {
    use crate::protocols::{
        ExternalSequenceBlockHash, KvCacheEvent, KvCacheEventData, KvCacheStoreData,
        KvCacheStoredBlockData, LocalBlockHash, RouterEvent, StorageTier,
        compute_seq_hash_for_block,
    };

    /// Construct a STORE [`RouterEvent`] for `local_hashes` that chain off
    /// `prefix_hashes` (the parent path). The event is tagged `storage_tier`.
    /// Mirrors the helper used in the request-plane tests so behavior across
    /// the two indexer surfaces stays comparable.
    pub fn store_event(
        worker_id: u64,
        dp_rank: u32,
        event_id: u64,
        prefix_hashes: &[u64],
        local_hashes: &[u64],
        storage_tier: StorageTier,
    ) -> RouterEvent {
        let prefix_block_hashes: Vec<LocalBlockHash> =
            prefix_hashes.iter().copied().map(LocalBlockHash).collect();
        let parent_hash = compute_seq_hash_for_block(&prefix_block_hashes)
            .last()
            .copied()
            .map(ExternalSequenceBlockHash);

        let full_hashes: Vec<LocalBlockHash> = prefix_hashes
            .iter()
            .chain(local_hashes.iter())
            .copied()
            .map(LocalBlockHash)
            .collect();
        let full_sequence_hashes = compute_seq_hash_for_block(&full_hashes);
        let new_sequence_hashes = &full_sequence_hashes[prefix_hashes.len()..];
        let blocks = local_hashes
            .iter()
            .zip(new_sequence_hashes.iter())
            .map(|(&local_hash, &sequence_hash)| KvCacheStoredBlockData {
                block_hash: ExternalSequenceBlockHash(sequence_hash),
                tokens_hash: LocalBlockHash(local_hash),
                mm_extra_info: None,
            })
            .collect();

        RouterEvent::with_storage_tier(
            worker_id,
            KvCacheEvent {
                event_id,
                data: KvCacheEventData::Stored(KvCacheStoreData {
                    parent_hash,
                    start_position: None,
                    blocks,
                }),
                dp_rank,
            },
            storage_tier,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::test_util::store_event;
    use super::*;
    use crate::indexer::KvIndexerInterface;
    use crate::protocols::{LocalBlockHash, StorageTier, WorkerWithDpRank};

    /// Apply a Device store and a HostPinned store anchored on it. The tiered
    /// query must surface both tier hits, and the device-tier `find_matches`
    /// must still see the device store (i.e. dispatch routed it to the
    /// primary, not to the lower-tier slot).
    #[tokio::test]
    async fn apply_event_routed_dispatches_by_tier() {
        let indexer = create_indexer(4, 1);
        let worker = WorkerWithDpRank::new(7, 0);

        indexer
            .apply_event_routed(store_event(
                worker.worker_id,
                worker.dp_rank,
                1,
                &[],
                &[11, 12],
                StorageTier::Device,
            ))
            .await;

        indexer
            .apply_event_routed(store_event(
                worker.worker_id,
                worker.dp_rank,
                2,
                &[11, 12],
                &[13],
                StorageTier::HostPinned,
            ))
            .await;

        // Flush primary so the in-flight events are observable by the query.
        if let Indexer::Single { primary, .. } = &indexer {
            let _ = primary.flush().await;
        }
        if let Indexer::Single { lower_tier, .. } = &indexer {
            for inner in lower_tier.all() {
                let _ = inner.dump_events().await.unwrap();
            }
        }

        let sequence = vec![LocalBlockHash(11), LocalBlockHash(12), LocalBlockHash(13)];
        let tiered = indexer.find_tiered_matches(sequence).await.unwrap();

        assert_eq!(
            tiered.device.overlap_scores.scores.get(&worker).copied(),
            Some(2),
            "device should match 2 blocks for the worker"
        );
        let host_hits = tiered
            .lower_tier
            .get(&StorageTier::HostPinned)
            .expect("host-pinned tier should have been allocated");
        assert_eq!(
            host_hits.hits.get(&worker).copied(),
            Some(1),
            "host-pinned should report 1 additional matched block beyond device"
        );
    }

    /// Dump every tier's events from a populated indexer, replay through a
    /// fresh indexer with `apply_event_routed`, and assert the tiered query
    /// result is identical. This is the peer-recovery round trip: it would
    /// fail if `dump_events` skipped lower-tier events or omitted their tier
    /// tags (peer would replay HostPinned events into the device primary).
    #[tokio::test]
    async fn dump_events_round_trips_through_apply_event_routed() {
        let block_size = 4;
        let worker = WorkerWithDpRank::new(7, 0);
        let source = create_indexer(block_size, 1);

        source
            .apply_event_routed(store_event(
                worker.worker_id,
                worker.dp_rank,
                1,
                &[],
                &[11, 12],
                StorageTier::Device,
            ))
            .await;
        source
            .apply_event_routed(store_event(
                worker.worker_id,
                worker.dp_rank,
                2,
                &[11, 12],
                &[13],
                StorageTier::HostPinned,
            ))
            .await;
        source
            .apply_event_routed(store_event(
                worker.worker_id,
                worker.dp_rank,
                3,
                &[11, 12, 13],
                &[14],
                StorageTier::Disk,
            ))
            .await;

        if let Indexer::Single {
            primary,
            lower_tier,
        } = &source
        {
            let _ = primary.flush().await;
            for inner in lower_tier.all() {
                let _ = inner.dump_events().await.unwrap();
            }
        }

        let dump = source.dump_events().await.unwrap();

        // Sanity: dump must surface events from every tier we fed in.
        assert!(
            dump.iter()
                .any(|e| matches!(e.storage_tier, StorageTier::Device)),
            "dump must contain Device events"
        );
        assert!(
            dump.iter()
                .any(|e| matches!(e.storage_tier, StorageTier::HostPinned)),
            "dump must contain HostPinned events with tier retagged"
        );
        assert!(
            dump.iter()
                .any(|e| matches!(e.storage_tier, StorageTier::Disk)),
            "dump must contain Disk events with tier retagged"
        );

        let replayed = create_indexer(block_size, 1);
        for event in dump {
            replayed.apply_event_routed(event).await;
        }
        if let Indexer::Single {
            primary,
            lower_tier,
        } = &replayed
        {
            let _ = primary.flush().await;
            for inner in lower_tier.all() {
                let _ = inner.dump_events().await.unwrap();
            }
        }

        let sequence = vec![
            LocalBlockHash(11),
            LocalBlockHash(12),
            LocalBlockHash(13),
            LocalBlockHash(14),
        ];
        let tiered = replayed.find_tiered_matches(sequence).await.unwrap();

        assert_eq!(
            tiered.device.overlap_scores.scores.get(&worker).copied(),
            Some(2),
            "device should match 2 blocks after replay"
        );
        assert_eq!(
            tiered
                .lower_tier
                .get(&StorageTier::HostPinned)
                .and_then(|d| d.hits.get(&worker).copied()),
            Some(1),
            "host-pinned should report 1 additional block after replay"
        );
        assert_eq!(
            tiered
                .lower_tier
                .get(&StorageTier::Disk)
                .and_then(|d| d.hits.get(&worker).copied()),
            Some(1),
            "disk should report 1 additional block after replay"
        );
    }
}

pub fn create_indexer(block_size: u32, num_threads: usize) -> Indexer {
    if num_threads > 1 {
        Indexer::Concurrent {
            primary: Arc::new(ThreadPoolIndexer::new(
                ConcurrentRadixTreeCompressed::new(),
                num_threads,
                block_size,
            )),
            lower_tier: LowerTierIndexers::new(num_threads, block_size),
        }
    } else {
        Indexer::Single {
            primary: KvIndexer::new_with_frequency(
                CancellationToken::new(),
                None,
                block_size,
                Arc::new(KvIndexerMetrics::new_unregistered()),
                None,
            ),
            lower_tier: LowerTierIndexers::new(1, block_size),
        }
    }
}
