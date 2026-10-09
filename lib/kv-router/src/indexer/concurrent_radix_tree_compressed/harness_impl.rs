// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use super::*;
use crate::indexer::harness::HarnessBackend;

impl HarnessBackend for ConcurrentRadixTreeCompressed {
    fn harness_new() -> Self {
        Self::new()
    }

    fn harness_match_details(&self, sequence: &[LocalBlockHash]) -> Option<MatchDetails> {
        Some(self.find_match_details_impl(sequence, false))
    }

    fn harness_rank_slot(&self, rank: WorkerWithDpRank) -> Option<usize> {
        self.slot_for_test(rank).map(|slot| slot.index())
    }

    fn harness_structure_size(&self) -> Option<usize> {
        Some(self.raw_child_edge_count())
    }
}

/// CRTC that sweeps on volume as soon as any block is dead, with no gap between sweeps,
/// so the shared harness drives stores and removals against near-continuous reclamation.
pub(crate) struct EagerReclaimCrtc(ConcurrentRadixTreeCompressed);

impl SyncIndexer for EagerReclaimCrtc {
    fn worker(
        &self,
        event_receiver: flume::Receiver<WorkerTask>,
        metrics: Option<Arc<KvIndexerMetrics>>,
    ) -> anyhow::Result<()> {
        self.0.worker(event_receiver, metrics)
    }

    fn find_matches(&self, sequence: &[LocalBlockHash], early_exit: bool) -> OverlapScores {
        self.0.find_matches(sequence, early_exit)
    }

    fn try_schedule_cleanup(&self) -> bool {
        self.0.try_schedule_cleanup()
    }

    fn cancel_scheduled_cleanup(&self) {
        self.0.cancel_scheduled_cleanup();
    }

    fn run_cleanup_task(&self) {
        self.0.run_cleanup_task();
    }
}

/// Volume-triggered sweeps run by every dropped `EagerReclaimCrtc`.
pub(crate) static EAGER_VOLUME_SWEEPS: std::sync::atomic::AtomicU64 =
    std::sync::atomic::AtomicU64::new(0);

impl Drop for EagerReclaimCrtc {
    fn drop(&mut self) {
        EAGER_VOLUME_SWEEPS.fetch_add(
            self.0.probe_shape().sweeps_volume,
            std::sync::atomic::Ordering::Relaxed,
        );
    }
}

impl HarnessBackend for EagerReclaimCrtc {
    fn harness_new() -> Self {
        Self(ConcurrentRadixTreeCompressed::with_reclaim_config(
            ReclaimConfig {
                volume_sweep: true,
                dead_floor: 1,
                min_gap: std::time::Duration::ZERO,
                ..ReclaimConfig::default()
            },
        ))
    }

    fn harness_match_details(&self, sequence: &[LocalBlockHash]) -> Option<MatchDetails> {
        self.0.harness_match_details(sequence)
    }

    fn harness_rank_slot(&self, rank: WorkerWithDpRank) -> Option<usize> {
        self.0.harness_rank_slot(rank)
    }

    fn harness_structure_size(&self) -> Option<usize> {
        self.0.harness_structure_size()
    }
}

/// The eager backend really sweeps on volume under the pool differential, so its suite
/// exercises reclamation racing every store and removal.
#[test]
fn eager_backend_sweeps_on_volume_in_the_pool_differential() {
    use crate::indexer::harness::differential_pool_for_test;
    let before = EAGER_VOLUME_SWEEPS.load(std::sync::atomic::Ordering::Relaxed);
    differential_pool_for_test::<EagerReclaimCrtc>("crtc_eager_sweeps", 4);
    let after = EAGER_VOLUME_SWEEPS.load(std::sync::atomic::Ordering::Relaxed);
    assert!(after > before, "no volume-triggered sweep ran");
}
