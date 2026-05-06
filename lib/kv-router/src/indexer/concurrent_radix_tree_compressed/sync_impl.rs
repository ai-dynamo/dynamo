// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use super::*;

// ============================================================================
// SyncIndexer implementation for ConcurrentRadixTreeCompressed
// ============================================================================

impl SyncIndexer for ConcurrentRadixTreeCompressed {
    #[cfg_attr(feature = "profile", inline(never))]
    fn worker(
        &self,
        event_receiver: flume::Receiver<WorkerTask>,
        metrics: Option<Arc<KvIndexerMetrics>>,
    ) -> anyhow::Result<()> {
        let mut lookup = FxHashMap::default();
        let counters = metrics.as_ref().map(|m| m.prebind());

        while let Ok(task) = event_receiver.recv() {
            match task {
                WorkerTask::Event(event) => {
                    let kind = EventKind::of(&event.event.data);
                    let result = self.apply_event(&mut lookup, event, counters.as_ref());
                    if result.is_err() {
                        tracing::warn!("Failed to apply event: {:?}", result.as_ref().err());
                    }
                    if let Some(ref c) = counters {
                        c.inc(kind, result);
                    }
                }
                WorkerTask::EventWithAck { event, resp } => {
                    let kind = EventKind::of(&event.event.data);
                    let result = self.apply_event(&mut lookup, event, counters.as_ref());
                    let applied = result.is_ok();
                    if result.is_err() {
                        tracing::warn!("Failed to apply event: {:?}", result.as_ref().err());
                    }
                    if let Some(ref c) = counters {
                        c.inc(kind, result);
                    }
                    let _ = resp.send(applied);
                }
                WorkerTask::Anchor { worker, anchor } => {
                    if let Err(error) = self.apply_anchor(worker, anchor) {
                        tracing::warn!(?error, "Failed to apply anchor");
                    }
                }
                WorkerTask::RemoveWorker(worker_id) => {
                    self.remove_or_clear_worker_blocks(&mut lookup, worker_id, false);
                }
                WorkerTask::RemoveWorkerDpRank(worker_id, dp_rank) => {
                    self.remove_worker_dp_rank(&mut lookup, worker_id, dp_rank);
                }
                WorkerTask::CleanupStaleChildren => {
                    self.run_cleanup_task();
                }
                WorkerTask::DumpEvents(_sender) => {
                    let _ = _sender.send(Ok(Vec::new()));
                }
                WorkerTask::Stats(sender) => {
                    let stats = WorkerLookupStats::from_worker_block_counts(
                        lookup
                            .iter()
                            .map(|(worker, worker_lookup)| (*worker, worker_lookup.len())),
                    );
                    let _ = sender.send(stats);
                }
                WorkerTask::Flush(sender) => {
                    let _ = sender.send(());
                }
                WorkerTask::Terminate => {
                    break;
                }
            }
        }

        tracing::debug!("ConcurrentRadixTreeCompressed worker thread shutting down");
        Ok(())
    }

    fn find_matches(&self, sequence: &[LocalBlockHash], early_exit: bool) -> OverlapScores {
        self.find_matches_impl(sequence, early_exit)
    }

    fn apply_anchor(
        &self,
        worker: WorkerWithDpRank,
        anchor: AnchorTask,
    ) -> Result<(), KvCacheEventError> {
        let anchor_node = {
            let entry = self
                .anchor_nodes
                .entry(anchor.anchor_id)
                .or_insert_with(|| {
                    Arc::new(Node::from_anchor(
                        anchor.anchor_local_hash,
                        anchor.anchor_id,
                    ))
                });
            entry.clone()
        };

        anchor_node.promote_worker_to_full_edge(worker);
        Ok(())
    }

    fn find_matches_from_anchor(
        &self,
        anchor: AnchorRef,
        suffix: &[LocalBlockHash],
    ) -> Result<OverlapScores, KvRouterError> {
        let Some(anchor_node) = self
            .anchor_nodes
            .get(&anchor.anchor_id)
            .map(|entry| entry.clone())
        else {
            return Ok(OverlapScores::new());
        };
        let details = if suffix.len() <= MAX_NO_COPY_ANCHORED_SUFFIX_BLOCKS {
            self.find_details_from_seq(
                Some(anchor_node),
                AnchoredHashSequence {
                    head: anchor.anchor_local_hash,
                    tail: suffix,
                },
                false,
            )
        } else {
            let mut sequence = Vec::with_capacity(suffix.len() + 1);
            sequence.push(anchor.anchor_local_hash);
            sequence.extend_from_slice(suffix);
            self.find_details_from_seq(Some(anchor_node), SliceHashSequence(&sequence), false)
        };
        let mut scores = details.overlap_scores;
        let depth_adjustment = anchor.anchor_depth.saturating_sub(1) as u32;
        if depth_adjustment > 0 {
            for score in scores.scores.values_mut() {
                *score += depth_adjustment;
            }
        }
        Ok(scores)
    }

    fn try_schedule_cleanup(&self) -> bool {
        self.cleanup.try_schedule()
    }

    fn cancel_scheduled_cleanup(&self) {
        self.cleanup.cancel();
    }

    fn run_cleanup_task(&self) {
        let mut cleanup_guard = CleanupGuard::new(&self.cleanup);
        self.sweep_stale_children();
        cleanup_guard.mark_completed();
    }

    fn timing_report(&self) -> String {
        #[cfg(not(feature = "bench"))]
        {
            String::new()
        }

        #[cfg(feature = "bench")]
        {
            let node_splits = self.bench_metrics.node_splits.load(Ordering::Relaxed);
            let lookup_repair_scans = self
                .bench_metrics
                .lookup_repair_scans
                .load(Ordering::Relaxed);
            let lookup_repair_entries = self
                .bench_metrics
                .lookup_repair_entries
                .load(Ordering::Relaxed);
            format!(
                "ConcurrentRadixTreeCompressed bench metrics:\n  \
                 node splits = {node_splits}\n  \
                 lookup repair scans = {lookup_repair_scans}\n  \
                 lookup repair entries = {lookup_repair_entries}"
            )
        }
    }

    fn dump_events(&self) -> Option<Vec<RouterEvent>> {
        Some(self.dump_tree_as_events())
    }
}
