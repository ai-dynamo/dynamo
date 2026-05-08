// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::sync::Arc;

use dynamo_kv_router::ConcurrentRadixTreeCompressed;
use dynamo_kv_router::LocalBlockHash;
use dynamo_kv_router::indexer::{
    KvIndexer, KvIndexerInterface, KvIndexerMetrics, LowerTierContinuation, LowerTierIndexer,
    MatchDetails, ThreadPoolIndexer,
};
use dynamo_kv_router::protocols::{KvCacheEvent, KvCacheEventData, RouterEvent, StorageTier};
use rustc_hash::FxHashMap;
use tokio_util::sync::CancellationToken;

use super::{
    BLOCK_SIZE, NUM_EVENT_WORKERS, NormalizedOverlapScores, ReplayEntryKind, WorkerReplayArtifacts,
    collect_replay_entries,
};

fn additive_device_and_host_pinned_scores(
    device_details: MatchDetails,
    request: &[LocalBlockHash],
    host_pinned: &LowerTierIndexer,
) -> NormalizedOverlapScores {
    let mut scores = device_details.overlap_scores.scores;
    let mut continuations = FxHashMap::default();

    for (worker, device_blocks) in &scores {
        let Some(last_hash) = device_details.last_matched_hashes.get(worker) else {
            continue;
        };
        let start_pos = *device_blocks as usize;
        if start_pos < request.len() {
            continuations.insert(*worker, LowerTierContinuation::new(start_pos, *last_hash));
        }
    }

    if let Some(first_hash) = request.first() {
        for worker in host_pinned.root_workers(*first_hash) {
            continuations
                .entry(worker)
                .or_insert_with(|| LowerTierContinuation::from_root(0));
        }
    }

    for (worker, host_blocks) in host_pinned.query_contiguous_hits(request, &continuations) {
        if host_blocks == 0 {
            continue;
        }
        let host_blocks = u32::try_from(host_blocks).unwrap_or(u32::MAX);
        scores
            .entry(worker)
            .and_modify(|score| *score = score.saturating_add(host_blocks))
            .or_insert(host_blocks);
    }

    scores.into_iter().filter(|(_, score)| *score > 0).collect()
}

struct ReferenceTieredReplay {
    primary: KvIndexer,
    host_pinned: ThreadPoolIndexer<LowerTierIndexer>,
}

impl ReferenceTieredReplay {
    fn new() -> Self {
        Self {
            primary: KvIndexer::new(
                CancellationToken::new(),
                BLOCK_SIZE,
                Arc::new(KvIndexerMetrics::new_unregistered()),
            ),
            host_pinned: ThreadPoolIndexer::new(
                LowerTierIndexer::new(),
                NUM_EVENT_WORKERS,
                BLOCK_SIZE,
            ),
        }
    }

    async fn apply_event(&self, worker_id: u64, event: KvCacheEvent, storage_tier: StorageTier) {
        if matches!(&event.data, KvCacheEventData::Cleared) {
            self.primary
                .apply_event(RouterEvent::with_storage_tier(
                    worker_id,
                    event.clone(),
                    StorageTier::Device,
                ))
                .await;
            self.host_pinned
                .apply_event(RouterEvent::with_storage_tier(
                    worker_id,
                    event,
                    StorageTier::HostPinned,
                ))
                .await;
            return;
        }

        match storage_tier {
            StorageTier::Device => {
                self.primary
                    .apply_event(RouterEvent::with_storage_tier(
                        worker_id,
                        event,
                        storage_tier,
                    ))
                    .await;
            }
            StorageTier::HostPinned => {
                self.host_pinned
                    .apply_event(RouterEvent::with_storage_tier(
                        worker_id,
                        event,
                        storage_tier,
                    ))
                    .await;
            }
            StorageTier::Disk | StorageTier::External => {}
        }
    }

    async fn find_matches(
        &self,
        request: &[LocalBlockHash],
    ) -> anyhow::Result<NormalizedOverlapScores> {
        let details = self.primary.find_match_details(request.to_vec()).await?;
        Ok(additive_device_and_host_pinned_scores(
            details,
            request,
            self.host_pinned.backend(),
        ))
    }

    async fn flush(&self) {
        let _ = self.primary.flush().await;
        self.host_pinned.flush().await;
    }

    fn shutdown(&self) {
        self.primary.shutdown();
        self.host_pinned.shutdown();
    }
}

struct CrtcLowerTierReplay {
    primary: ThreadPoolIndexer<ConcurrentRadixTreeCompressed>,
    host_pinned: ThreadPoolIndexer<LowerTierIndexer>,
}

impl CrtcLowerTierReplay {
    fn new() -> Self {
        Self {
            primary: ThreadPoolIndexer::new_with_metrics(
                ConcurrentRadixTreeCompressed::new(),
                NUM_EVENT_WORKERS,
                BLOCK_SIZE,
                Some(Arc::new(KvIndexerMetrics::new_unregistered())),
            ),
            host_pinned: ThreadPoolIndexer::new(
                LowerTierIndexer::new(),
                NUM_EVENT_WORKERS,
                BLOCK_SIZE,
            ),
        }
    }

    async fn apply_event(&self, worker_id: u64, event: KvCacheEvent, storage_tier: StorageTier) {
        if matches!(&event.data, KvCacheEventData::Cleared) {
            self.primary
                .apply_event(RouterEvent::with_storage_tier(
                    worker_id,
                    event.clone(),
                    StorageTier::Device,
                ))
                .await;
            self.host_pinned
                .apply_event(RouterEvent::with_storage_tier(
                    worker_id,
                    event,
                    StorageTier::HostPinned,
                ))
                .await;
            return;
        }

        match storage_tier {
            StorageTier::Device => {
                self.primary
                    .apply_event(RouterEvent::with_storage_tier(
                        worker_id,
                        event,
                        storage_tier,
                    ))
                    .await;
            }
            StorageTier::HostPinned => {
                self.host_pinned
                    .apply_event(RouterEvent::with_storage_tier(
                        worker_id,
                        event,
                        storage_tier,
                    ))
                    .await;
            }
            StorageTier::Disk | StorageTier::External => {}
        }
    }

    fn find_matches(&self, request: &[LocalBlockHash]) -> NormalizedOverlapScores {
        let details = self
            .primary
            .backend()
            .find_match_details_impl(request, false);
        additive_device_and_host_pinned_scores(details, request, self.host_pinned.backend())
    }

    async fn flush(&self) {
        self.primary.flush().await;
        self.host_pinned.flush().await;
    }

    fn shutdown(&self) {
        self.primary.shutdown();
        self.host_pinned.shutdown();
    }
}

pub(crate) async fn collect_tiered_replay_scores(
    artifacts: &[WorkerReplayArtifacts],
) -> anyhow::Result<(
    Vec<NormalizedOverlapScores>,
    Vec<NormalizedOverlapScores>,
    usize,
)> {
    let reference = ReferenceTieredReplay::new();
    let crtc_lower_tier = CrtcLowerTierReplay::new();
    let entries = collect_replay_entries(artifacts);
    let mut reference_scores = Vec::new();
    let mut crtc_scores = Vec::new();
    let mut idx = 0;

    while idx < entries.len() {
        let timestamp_us = entries[idx].timestamp_us;
        while idx < entries.len() && entries[idx].timestamp_us == timestamp_us {
            match &entries[idx].kind {
                ReplayEntryKind::Request(request) => {
                    reference_scores.push(reference.find_matches(request).await?);
                    crtc_scores.push(crtc_lower_tier.find_matches(request));
                }
                ReplayEntryKind::Event {
                    event,
                    storage_tier,
                } => {
                    reference
                        .apply_event(entries[idx].worker_id, event.clone(), *storage_tier)
                        .await;
                    crtc_lower_tier
                        .apply_event(entries[idx].worker_id, event.clone(), *storage_tier)
                        .await;
                }
                ReplayEntryKind::ApproxWrite(_) => {
                    anyhow::bail!("approximate writes are not part of KV-event replay artifacts")
                }
            }
            idx += 1;
        }
        reference.flush().await;
        crtc_lower_tier.flush().await;
    }

    crtc_lower_tier.flush().await;
    let host_pinned_events = crtc_lower_tier.host_pinned.dump_events().await?.len();
    reference.shutdown();
    crtc_lower_tier.shutdown();

    Ok((reference_scores, crtc_scores, host_pinned_events))
}

pub(crate) fn score_sum(scores: &NormalizedOverlapScores) -> u64 {
    scores.values().map(|score| u64::from(*score)).sum()
}
