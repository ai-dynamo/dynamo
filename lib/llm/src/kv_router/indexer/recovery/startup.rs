// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Best-effort startup barrier. Count source incarnations, never retry tasks.

use std::{collections::HashMap, sync::Arc, time::Duration};

use dynamo_kv_router::protocols::WorkerWithDpRank;
use parking_lot::Mutex;
use tokio::sync::{Notify, watch};
use tokio::time::Instant;
use tokio_util::sync::CancellationToken;

use crate::discovery::{KvSourceMembershipView, KvSourceStatus};

const TARGET_PERCENT: usize = 95;
const MAX_WAIT: Duration = Duration::from_secs(600);
const SETTLE: Duration = Duration::from_millis(100);

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct Source {
    pub publisher_id: u64,
    /// None for legacy sources; Some for the versioned state-agent protocol.
    pub attachment_generation: Option<u64>,
    pub versioned: bool,
}

#[derive(Clone, Default, Debug)]
pub(crate) struct StartupRecovery(Arc<Inner>);

#[derive(Default, Debug)]
struct Inner {
    state: Mutex<State>,
    changed: Notify,
}

#[derive(Default, Debug)]
struct State {
    closed: bool,
    revision: u64,
    attempts: HashMap<WorkerWithDpRank, (Source, bool)>,
}

#[derive(Clone, Debug)]
pub(super) struct Attempt {
    gate: StartupRecovery,
    worker: WorkerWithDpRank,
    source: Source,
}

impl StartupRecovery {
    pub(super) fn register(&self, worker: WorkerWithDpRank, source: Source) -> Option<Attempt> {
        let mut state = self.0.state.lock();
        if state.closed {
            return None;
        }
        let changed = state
            .attempts
            .get(&worker)
            .is_none_or(|entry| entry.0 != source);
        if changed {
            state.attempts.insert(worker, (source, false));
            state.revision += 1;
        }
        drop(state);
        if changed {
            self.0.changed.notify_one();
        }
        Some(Attempt {
            gate: self.clone(),
            worker,
            source,
        })
    }

    fn progress(&self, view: &KvSourceMembershipView) -> (usize, usize, u64) {
        let mut state = self.0.state.lock();
        state
            .attempts
            .retain(|worker, _| view.sources.contains_key(worker));
        let mut expected = 0;
        let mut completed = 0;
        for (worker, status) in &view.sources {
            if matches!(status, KvSourceStatus::ActiveLiveOnly(_))
                || (view.kv_event_publishing_enabled(worker.worker_id) == Some(false)
                    && !matches!(status, KvSourceStatus::Suppressed))
            {
                continue;
            }
            if view.recovery_expected(worker) == Some(false)
                && !matches!(
                    status,
                    KvSourceStatus::ActiveRecoverable(_) | KvSourceStatus::Suppressed
                )
            {
                continue;
            }
            expected += 1;
            let Some((source, true)) = state.attempts.get(worker) else {
                continue;
            };
            let matches = match status {
                KvSourceStatus::ActiveRecoverable(active) => {
                    !source.versioned && active.publisher_id == source.publisher_id
                }
                KvSourceStatus::Suppressed => source.versioned,
                _ => false,
            };
            completed += usize::from(matches);
        }
        (expected, completed, state.revision)
    }

    pub(super) async fn wait(
        &self,
        mut membership: watch::Receiver<KvSourceMembershipView>,
        cancel: &CancellationToken,
    ) -> anyhow::Result<()> {
        let deadline = Instant::now() + MAX_WAIT;
        let mut stable_since = None;
        let mut log = tokio::time::interval(Duration::from_secs(5));
        loop {
            // Register before inspecting state so a completion cannot be lost.
            let changed = self.0.changed.notified();
            tokio::pin!(changed);
            changed.as_mut().enable();
            anyhow::ensure!(
                !cancel.is_cancelled(),
                "cancelled while waiting for initial KV recovery"
            );
            let (expected, completed, revision) = self.progress(&membership.borrow_and_update());
            let threshold = expected.saturating_mul(TARGET_PERCENT).div_ceil(100);
            let enough = completed >= threshold;
            let settled = if enough {
                Instant::now().duration_since(*stable_since.get_or_insert_with(Instant::now))
                    >= SETTLE
            } else {
                stable_since = None;
                false
            };
            if settled || Instant::now() >= deadline {
                let mut state = self.0.state.lock();
                anyhow::ensure!(
                    !cancel.is_cancelled(),
                    "cancelled while waiting for initial KV recovery"
                );
                let membership_changed = membership.has_changed().map_err(|_| {
                    anyhow::anyhow!("KV membership watch closed during initial recovery")
                })?;
                if Instant::now() < deadline && (state.revision != revision || membership_changed) {
                    stable_since = None;
                    continue;
                }
                state.closed = true;
                state.attempts.clear();
                drop(state);
                if settled {
                    tracing::info!(
                        expected,
                        completed,
                        target_percent = TARGET_PERCENT,
                        "Initial KV recovery attempts reached startup target"
                    );
                } else {
                    tracing::warn!(
                        expected,
                        completed,
                        "Initial KV recovery wait timed out; remaining recovery continues in background"
                    );
                }
                return Ok(());
            }
            tokio::select! {
                biased;
                _ = cancel.cancelled() => anyhow::bail!("cancelled while waiting for initial KV recovery"),
                result = membership.changed() => {
                    result.map_err(|_| anyhow::anyhow!("KV membership watch closed during initial recovery"))?;
                    stable_since = None;
                }
                _ = changed => { stable_since = None; }
                _ = tokio::time::sleep_until(deadline) => {},
                _ = tokio::time::sleep(SETTLE), if enough => {},
                _ = log.tick() => tracing::info!(expected, completed, "Waiting for initial KV recovery attempts"),
            }
        }
    }
}

impl Attempt {
    pub fn finish(&self) {
        let mut state = self.gate.0.state.lock();
        let changed = if let Some((source, completed)) = state.attempts.get_mut(&self.worker)
            && *source == self.source
            && !*completed
        {
            *completed = true;
            state.revision += 1;
            true
        } else {
            false
        };
        drop(state);
        if changed {
            self.gate.0.changed.notify_one();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::discovery::{KvEventSource, KvStateEndpointResolution};
    use dynamo_runtime::protocols::EndpointId;

    fn view(count: u64) -> KvSourceMembershipView {
        let endpoint = EndpointId {
            namespace: "test".into(),
            component: "worker".into(),
            name: "generate".into(),
        };
        KvSourceMembershipView {
            serving_endpoint: endpoint.clone(),
            endpoint_resolution: KvStateEndpointResolution::Resolved(endpoint.clone()),
            sources: (0..count)
                .map(|id| {
                    let worker = WorkerWithDpRank::new(id, 0);
                    (
                        worker,
                        KvSourceStatus::ActiveRecoverable(KvEventSource {
                            kv_state_endpoint: endpoint.clone(),
                            worker,
                            publisher_id: id,
                            recovery_target: None,
                        }),
                    )
                })
                .collect(),
            kv_event_publishing_enabled: HashMap::new(),
            kv_event_source_mode: HashMap::new(),
            recovery_expected: HashMap::new(),
        }
    }

    fn register(gate: &StartupRecovery, worker: u64, publisher: u64) -> Attempt {
        gate.register(
            WorkerWithDpRank::new(worker, 0),
            Source {
                publisher_id: publisher,
                attachment_generation: None,
                versioned: false,
            },
        )
        .unwrap()
    }

    #[tokio::test(start_paused = true)]
    async fn waits_for_single_worker_and_tolerates_five_percent_stragglers() {
        for count in [1, 20] {
            let gate = StartupRecovery::default();
            let (_tx, rx) = watch::channel(view(count));
            let cancel = CancellationToken::new();
            let wait = gate.wait(rx, &cancel);
            tokio::pin!(wait);
            assert!(futures::poll!(&mut wait).is_pending());
            let target = (count as usize * TARGET_PERCENT).div_ceil(100);
            for worker in 0..target as u64 {
                register(&gate, worker, worker).finish();
            }
            wait.await.unwrap();
            assert!(gate.0.state.lock().closed);
        }
    }

    #[tokio::test(start_paused = true)]
    async fn arrivals_join_the_wait_and_departures_stop_counting() {
        let gate = StartupRecovery::default();
        let (tx, rx) = watch::channel(view(1));
        let cancel = CancellationToken::new();
        let wait = gate.wait(rx, &cancel);
        tokio::pin!(wait);
        assert!(futures::poll!(&mut wait).is_pending());
        tx.send_replace(view(2));
        register(&gate, 0, 0).finish();
        assert!(futures::poll!(&mut wait).is_pending());
        tokio::time::advance(SETTLE * 2).await;
        assert!(futures::poll!(&mut wait).is_pending());
        tx.send_replace(view(1));
        wait.await.unwrap();
    }

    #[tokio::test(start_paused = true)]
    async fn replacements_require_a_new_attempt_and_retries_do_not_inflate_progress() {
        let gate = StartupRecovery::default();
        let old = register(&gate, 0, 0);
        let replacement = register(&gate, 0, 100);
        let mut membership = view(2);
        if let KvSourceStatus::ActiveRecoverable(source) = membership
            .sources
            .get_mut(&WorkerWithDpRank::new(0, 0))
            .unwrap()
        {
            source.publisher_id = 100;
        }
        old.finish();
        assert_eq!(gate.progress(&membership).1, 0);
        replacement.finish();
        for _ in 0..100 {
            register(&gate, 0, 100).finish();
        }
        assert_eq!(gate.progress(&membership).0, 2);
        assert_eq!(gate.progress(&membership).1, 1);
        let (_tx, rx) = watch::channel(membership);
        let cancel = CancellationToken::new();
        let wait = gate.wait(rx, &cancel);
        tokio::pin!(wait);
        assert!(futures::poll!(&mut wait).is_pending());
        register(&gate, 1, 1).finish();
        wait.await.unwrap();
    }

    #[tokio::test(start_paused = true)]
    async fn missing_expected_workers_wait_until_deadline_and_cancellation_errors() {
        for cancelled in [false, true] {
            let gate = StartupRecovery::default();
            let mut membership = view(1);
            let worker = WorkerWithDpRank::new(0, 0);
            membership.sources.insert(worker, KvSourceStatus::Missing);
            membership.recovery_expected.insert(worker, true);
            let (_tx, rx) = watch::channel(membership);
            let cancel = CancellationToken::new();
            let wait = gate.wait(rx, &cancel);
            tokio::pin!(wait);
            assert!(futures::poll!(&mut wait).is_pending());
            tokio::time::advance(MAX_WAIT - Duration::from_secs(1)).await;
            assert!(futures::poll!(&mut wait).is_pending());
            if cancelled {
                cancel.cancel();
            } else {
                tokio::time::advance(Duration::from_secs(1)).await;
            }
            assert_eq!(wait.await.is_err(), cancelled);
        }
    }

    #[tokio::test(start_paused = true)]
    async fn membership_churn_does_not_extend_the_deadline() {
        let gate = StartupRecovery::default();
        let (tx, rx) = watch::channel(view(1));
        let cancel = CancellationToken::new();
        let wait = gate.wait(rx, &cancel);
        tokio::pin!(wait);
        assert!(futures::poll!(&mut wait).is_pending());
        for _ in 0..600 {
            tx.send_replace(view(1));
            assert!(futures::poll!(&mut wait).is_pending());
            tokio::time::advance(Duration::from_secs(1)).await;
        }
        tx.send_replace(view(2));
        wait.await.unwrap();
        assert!(gate.0.state.lock().closed);
    }

    #[tokio::test(start_paused = true)]
    async fn versioned_sources_wait_for_their_own_attempt() {
        let gate = StartupRecovery::default();
        let worker = WorkerWithDpRank::new(0, 0);
        let mut membership = view(1);
        membership
            .sources
            .insert(worker, KvSourceStatus::Suppressed);
        register(&gate, 0, 0).finish(); // An old legacy completion cannot release the V2 gate.
        let (_tx, rx) = watch::channel(membership);
        let cancel = CancellationToken::new();
        let wait = gate.wait(rx, &cancel);
        tokio::pin!(wait);
        assert!(futures::poll!(&mut wait).is_pending());
        let old = gate
            .register(
                worker,
                Source {
                    publisher_id: 10,
                    attachment_generation: Some(1),
                    versioned: true,
                },
            )
            .unwrap();
        let replacement = gate
            .register(
                worker,
                Source {
                    publisher_id: 10,
                    attachment_generation: Some(2),
                    versioned: true,
                },
            )
            .unwrap();
        old.finish();
        assert!(futures::poll!(&mut wait).is_pending());
        replacement.finish();
        wait.await.unwrap();
    }

    #[tokio::test(start_paused = true)]
    async fn empty_and_live_only_membership_do_not_wait_for_nonexistent_recovery() {
        for mut membership in [view(0), view(1)] {
            for status in membership.sources.values_mut() {
                if let KvSourceStatus::ActiveRecoverable(source) = status {
                    *status = KvSourceStatus::ActiveLiveOnly(source.clone());
                }
            }
            let gate = StartupRecovery::default();
            let (_tx, rx) = watch::channel(membership);
            let cancel = CancellationToken::new();
            let started = Instant::now();
            gate.wait(rx, &cancel).await.unwrap();
            assert!(started.elapsed() < MAX_WAIT);
        }
    }
}
