// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Mutation checks for the harness itself: a backend wrapper that silently drops every
//! `Removed` event must fail both the differential and the strict soak.

use std::sync::Arc;

use super::differential::{self, Mode};
use super::{HarnessBackend, soak};
use crate::indexer::concurrent_radix_tree_compressed::ConcurrentRadixTreeCompressed;
use crate::indexer::{KvIndexerMetrics, MatchDetails, SyncIndexer, WorkerTask};
use crate::protocols::{KvCacheEventData, LocalBlockHash, OverlapScores};

/// CRTC that acknowledges removals without applying them.
struct DropRemovals(ConcurrentRadixTreeCompressed);

fn is_removal(task: &WorkerTask) -> bool {
    match task {
        WorkerTask::Event(event) | WorkerTask::EventWithAck { event, .. } => {
            matches!(event.event.data, KvCacheEventData::Removed(_))
        }
        _ => false,
    }
}

impl SyncIndexer for DropRemovals {
    fn worker(
        &self,
        event_receiver: flume::Receiver<WorkerTask>,
        metrics: Option<Arc<KvIndexerMetrics>>,
    ) -> anyhow::Result<()> {
        let (tx, rx) = flume::unbounded();
        std::thread::scope(|scope| {
            scope.spawn(move || {
                for task in event_receiver {
                    let terminate = matches!(task, WorkerTask::Terminate);
                    if !is_removal(&task) {
                        let _ = tx.send(task);
                    } else if let WorkerTask::EventWithAck { resp, .. } = task {
                        let _ = resp.send(true);
                    }
                    if terminate {
                        break;
                    }
                }
            });
            self.0.worker(rx, metrics)
        })
    }

    fn find_matches(&self, sequence: &[LocalBlockHash], early_exit: bool) -> OverlapScores {
        self.0.find_matches(sequence, early_exit)
    }

    fn run_cleanup_task(&self) {
        self.0.run_cleanup_task();
    }
}

impl HarnessBackend for DropRemovals {
    fn harness_new() -> Self {
        Self(ConcurrentRadixTreeCompressed::new())
    }

    fn harness_match_details(&self, sequence: &[LocalBlockHash]) -> Option<MatchDetails> {
        self.0.harness_match_details(sequence)
    }
}

#[test]
fn differential_catches_dropped_removals() {
    let caught = std::panic::catch_unwind(|| {
        differential::run_suite::<DropRemovals>("drop_removals", Mode::Serial);
    });
    assert!(caught.is_err(), "the differential missed dropped removals");
}

#[test]
fn soak_catches_dropped_removals() {
    let caught = std::panic::catch_unwind(|| {
        let mut config = soak::Config::from_env(Some(false));
        config.secs = 2;
        soak::run_with::<DropRemovals>("drop_removals", config);
    });
    assert!(caught.is_err(), "the strict soak missed dropped removals");
}
