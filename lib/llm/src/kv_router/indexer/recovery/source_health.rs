// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::{
    collections::{BTreeMap, HashMap, HashSet},
    time::Duration,
};

use dynamo_kv_router::protocols::{WorkerId, WorkerWithDpRank};
use dynamo_runtime::protocols::EndpointId;
use tokio::{sync::oneshot, time::Instant};
use tokio_util::sync::CancellationToken;

use crate::{
    discovery::{
        KvSourceAmbiguity, KvSourceMembershipView, KvSourceMembershipWatch, KvSourceStatus,
    },
    kv_router::KvEventSourceRequirement,
    worker_type::WorkerType,
};

const SOURCE_JOIN_GRACE: Duration = Duration::from_secs(30);

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
enum DiagnosticCode {
    PublisherDisabled,
    SourceNotObserved,
    SourceAmbiguous,
    SourceRecovered,
}

impl DiagnosticCode {
    fn as_str(self) -> &'static str {
        match self {
            Self::PublisherDisabled => "kv_event_publisher_disabled",
            Self::SourceNotObserved => "kv_event_source_not_observed",
            Self::SourceAmbiguous => "kv_event_source_ambiguous",
            Self::SourceRecovered => "kv_event_source_recovered",
        }
    }
}

#[derive(Debug, Clone)]
struct RankDiagnostic {
    code: DiagnosticCode,
    worker: WorkerWithDpRank,
    kv_event_publishing_enabled: bool,
    waited: Duration,
}

#[derive(Debug, Clone, Copy)]
struct DeadlineToken {
    worker: WorkerWithDpRank,
    epoch: u64,
    deadline: Instant,
}

#[derive(Debug, Clone, Copy)]
enum RankIssue {
    Idle,
    Pending {
        code: DiagnosticCode,
        since: Instant,
        deadline: Instant,
    },
    Warned {
        code: DiagnosticCode,
        since: Instant,
    },
}

#[derive(Debug, Clone, Copy)]
struct RankHealth {
    epoch: u64,
    lifecycle_generation: u64,
    issue: RankIssue,
}

impl RankHealth {
    fn new(epoch: u64, lifecycle_generation: u64) -> Self {
        Self {
            epoch,
            lifecycle_generation,
            issue: RankIssue::Idle,
        }
    }
}

struct SourceHealthState {
    requirement: KvEventSourceRequirement,
    ranks: HashMap<WorkerWithDpRank, RankHealth>,
    next_epoch: u64,
}

impl SourceHealthState {
    fn new(requirement: KvEventSourceRequirement) -> Self {
        Self {
            requirement,
            ranks: HashMap::new(),
            next_epoch: 1,
        }
    }

    fn reconcile(&mut self, view: &KvSourceMembershipView, now: Instant) -> Vec<RankDiagnostic> {
        if !self.requirement.requires_source() {
            self.ranks.clear();
            return Vec::new();
        }

        let present: HashSet<_> = view.sources.keys().copied().collect();
        self.ranks.retain(|worker, _| present.contains(worker));

        let mut diagnostics = Vec::new();
        for (&worker, status) in &view.sources {
            let capability = view.kv_event_publishing_enabled(worker.worker_id);
            let generation = view.lifecycle_generation(&worker).unwrap_or(0);
            self.reconcile_rank(
                worker,
                status,
                capability,
                view.observation_state.is_bound(),
                generation,
                now,
                &mut diagnostics,
            );
        }
        diagnostics
    }

    #[allow(clippy::too_many_arguments)]
    fn reconcile_rank(
        &mut self,
        worker: WorkerWithDpRank,
        status: &KvSourceStatus,
        capability: Option<bool>,
        observation_bound: bool,
        lifecycle_generation: u64,
        now: Instant,
        diagnostics: &mut Vec<RankDiagnostic>,
    ) {
        let Some(capability) = capability else {
            self.ranks.remove(&worker);
            return;
        };

        let mut state = self
            .ranks
            .remove(&worker)
            .unwrap_or_else(|| RankHealth::new(self.take_epoch(), lifecycle_generation));
        let generation_changed = state.lifecycle_generation != lifecycle_generation;
        state.lifecycle_generation = lifecycle_generation;

        if !capability {
            self.set_immediate_issue(
                worker,
                &mut state,
                DiagnosticCode::PublisherDisabled,
                false,
                now,
                diagnostics,
            );
            self.ranks.insert(worker, state);
            return;
        }

        match status {
            KvSourceStatus::ActiveRecoverable(_) | KvSourceStatus::ActiveLiveOnly(_) => {
                if let RankIssue::Warned { since, .. } = state.issue {
                    diagnostics.push(RankDiagnostic {
                        code: DiagnosticCode::SourceRecovered,
                        worker,
                        kv_event_publishing_enabled: true,
                        waited: now.saturating_duration_since(since),
                    });
                }
                self.set_issue(&mut state, RankIssue::Idle);
            }
            KvSourceStatus::Missing if observation_bound => {
                let keep_existing = matches!(
                    state.issue,
                    RankIssue::Pending {
                        code: DiagnosticCode::SourceNotObserved,
                        ..
                    } | RankIssue::Warned {
                        code: DiagnosticCode::SourceNotObserved,
                        ..
                    }
                ) && !generation_changed;
                if !keep_existing {
                    self.set_issue(
                        &mut state,
                        RankIssue::Pending {
                            code: DiagnosticCode::SourceNotObserved,
                            since: now,
                            deadline: now + SOURCE_JOIN_GRACE,
                        },
                    );
                }
            }
            KvSourceStatus::Ambiguous(KvSourceAmbiguity::EndpointMapping { .. }) => {
                self.set_immediate_issue(
                    worker,
                    &mut state,
                    DiagnosticCode::SourceAmbiguous,
                    true,
                    now,
                    diagnostics,
                );
            }
            KvSourceStatus::Ambiguous(_) if observation_bound => {
                self.set_immediate_issue(
                    worker,
                    &mut state,
                    DiagnosticCode::SourceAmbiguous,
                    true,
                    now,
                    diagnostics,
                );
            }
            KvSourceStatus::Missing | KvSourceStatus::Ambiguous(_) => {
                if matches!(state.issue, RankIssue::Pending { .. }) {
                    self.set_issue(&mut state, RankIssue::Idle);
                }
            }
        }

        self.ranks.insert(worker, state);
    }

    fn set_immediate_issue(
        &mut self,
        worker: WorkerWithDpRank,
        state: &mut RankHealth,
        code: DiagnosticCode,
        capability: bool,
        now: Instant,
        diagnostics: &mut Vec<RankDiagnostic>,
    ) {
        if matches!(state.issue, RankIssue::Warned { code: current, .. } if current == code) {
            return;
        }
        diagnostics.push(RankDiagnostic {
            code,
            worker,
            kv_event_publishing_enabled: capability,
            waited: Duration::ZERO,
        });
        self.set_issue(state, RankIssue::Warned { code, since: now });
    }

    fn set_issue(&mut self, state: &mut RankHealth, issue: RankIssue) {
        state.epoch = self.take_epoch();
        state.issue = issue;
    }

    fn take_epoch(&mut self) -> u64 {
        let epoch = self.next_epoch;
        self.next_epoch = self.next_epoch.wrapping_add(1);
        epoch
    }

    fn next_deadline(&self) -> Option<DeadlineToken> {
        self.ranks
            .iter()
            .filter_map(|(&worker, state)| {
                let RankIssue::Pending { deadline, .. } = state.issue else {
                    return None;
                };
                Some(DeadlineToken {
                    worker,
                    epoch: state.epoch,
                    deadline,
                })
            })
            .min_by_key(|token| token.deadline)
    }

    fn fire_deadline(&mut self, token: DeadlineToken, now: Instant) -> Vec<RankDiagnostic> {
        let Some(state) = self.ranks.get(&token.worker) else {
            return Vec::new();
        };
        if state.epoch != token.epoch {
            return Vec::new();
        }
        let RankIssue::Pending { deadline, .. } = state.issue else {
            return Vec::new();
        };
        if deadline != token.deadline || deadline > now {
            return Vec::new();
        }

        let due: Vec<_> = self
            .ranks
            .iter()
            .filter_map(|(&worker, state)| {
                let RankIssue::Pending {
                    code,
                    since,
                    deadline,
                } = state.issue
                else {
                    return None;
                };
                (deadline <= now).then_some((worker, code, since))
            })
            .collect();
        let mut diagnostics = Vec::with_capacity(due.len());
        for (worker, code, since) in due {
            let epoch = self.take_epoch();
            let state = self
                .ranks
                .get_mut(&worker)
                .expect("due rank came from health map");
            state.epoch = epoch;
            state.issue = RankIssue::Warned { code, since };
            diagnostics.push(RankDiagnostic {
                code,
                worker,
                kv_event_publishing_enabled: true,
                waited: now.saturating_duration_since(since),
            });
        }
        diagnostics
    }
}

#[derive(Debug)]
struct AggregatedDiagnostic {
    code: DiagnosticCode,
    worker_id: WorkerId,
    kv_event_publishing_enabled: bool,
    waited_ms: u64,
    dp_ranks: Vec<u32>,
}

fn aggregate_diagnostics(diagnostics: Vec<RankDiagnostic>) -> Vec<AggregatedDiagnostic> {
    let mut grouped = BTreeMap::<(DiagnosticCode, WorkerId, bool), (u64, Vec<u32>)>::new();
    for diagnostic in diagnostics {
        let waited_ms = diagnostic.waited.as_millis().min(u128::from(u64::MAX)) as u64;
        let entry = grouped
            .entry((
                diagnostic.code,
                diagnostic.worker.worker_id,
                diagnostic.kv_event_publishing_enabled,
            ))
            .or_default();
        entry.0 = entry.0.max(waited_ms);
        entry.1.push(diagnostic.worker.dp_rank);
    }
    grouped
        .into_iter()
        .map(
            |((code, worker_id, kv_event_publishing_enabled), (waited_ms, mut dp_ranks))| {
                dp_ranks.sort_unstable();
                AggregatedDiagnostic {
                    code,
                    worker_id,
                    kv_event_publishing_enabled,
                    waited_ms,
                    dp_ranks,
                }
            },
        )
        .collect()
}

struct DiagnosticContext<'a> {
    model: &'a str,
    worker_role: Option<WorkerType>,
    requirement: KvEventSourceRequirement,
    serving_endpoint: &'a EndpointId,
}

fn emit_diagnostics(context: &DiagnosticContext<'_>, diagnostics: Vec<RankDiagnostic>) {
    let worker_role = context
        .worker_role
        .map(|role| role.as_str())
        .unwrap_or("unknown");
    for diagnostic in aggregate_diagnostics(diagnostics) {
        let diagnostic_code = diagnostic.code.as_str();
        let requirement = context.requirement.as_str();
        let rank_count = diagnostic.dp_ranks.len();
        let dp_ranks = diagnostic
            .dp_ranks
            .iter()
            .map(u32::to_string)
            .collect::<Vec<_>>()
            .join(",");
        if diagnostic.code == DiagnosticCode::SourceRecovered {
            tracing::info!(
                diagnostic_code,
                model = context.model,
                worker_role,
                requirement,
                worker_id = diagnostic.worker_id,
                serving_endpoint = %context.serving_endpoint,
                kv_event_publishing_enabled = diagnostic.kv_event_publishing_enabled,
                waited_ms = diagnostic.waited_ms,
                rank_count,
                dp_ranks = %dp_ranks,
                "KV event source recovered"
            );
        } else {
            tracing::warn!(
                diagnostic_code,
                model = context.model,
                worker_role,
                requirement,
                worker_id = diagnostic.worker_id,
                serving_endpoint = %context.serving_endpoint,
                kv_event_publishing_enabled = diagnostic.kv_event_publishing_enabled,
                waited_ms = diagnostic.waited_ms,
                rank_count,
                dp_ranks = %dp_ranks,
                "KV event source health warning"
            );
        }
    }
}

pub(super) fn spawn(
    mut membership_watch: KvSourceMembershipWatch,
    model: String,
    worker_role: Option<WorkerType>,
    requirement: KvEventSourceRequirement,
    serving_endpoint: EndpointId,
    cancellation_token: CancellationToken,
) -> oneshot::Receiver<()> {
    let (completion_tx, completion_rx) = oneshot::channel();
    tokio::spawn(async move {
        let mut state = SourceHealthState::new(requirement);
        let context = DiagnosticContext {
            model: &model,
            worker_role,
            requirement,
            serving_endpoint: &serving_endpoint,
        };
        let initial = membership_watch.borrow_and_update().clone();
        emit_diagnostics(&context, state.reconcile(&initial, Instant::now()));

        loop {
            let Some(deadline) = state.next_deadline() else {
                tokio::select! {
                    biased;
                    _ = cancellation_token.cancelled() => break,
                    changed = membership_watch.changed() => {
                        if changed.is_err() {
                            break;
                        }
                        let view = membership_watch.borrow_and_update().clone();
                        emit_diagnostics(&context, state.reconcile(&view, Instant::now()));
                    }
                }
                continue;
            };

            tokio::select! {
                biased;
                _ = cancellation_token.cancelled() => break,
                changed = membership_watch.changed() => {
                    if changed.is_err() {
                        break;
                    }
                    let now = Instant::now();
                    let view = membership_watch.borrow_and_update().clone();
                    let mut diagnostics = state.reconcile(&view, now);
                    if let Some(current) = state.next_deadline()
                        && current.deadline <= now
                    {
                        diagnostics.extend(state.fire_deadline(current, now));
                    }
                    emit_diagnostics(&context, diagnostics);
                }
                _ = tokio::time::sleep_until(deadline.deadline) => {
                    // Re-read membership before using the wake-up token. A removal, rejoin, or
                    // source arrival invalidates the per-rank epoch and suppresses the stale alarm.
                    let now = Instant::now();
                    let view = membership_watch.borrow_and_update().clone();
                    let mut diagnostics = state.reconcile(&view, now);
                    diagnostics.extend(state.fire_deadline(deadline, now));
                    emit_diagnostics(&context, diagnostics);
                }
            }
        }
        let _ = completion_tx.send(());
    });
    completion_rx
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::discovery::{KvEventSource, KvSourceObservationState, KvStateEndpointResolution};

    fn endpoint() -> EndpointId {
        EndpointId {
            namespace: "ns".to_string(),
            component: "worker".to_string(),
            name: "generate".to_string(),
        }
    }

    fn view(
        observation_state: KvSourceObservationState,
        capabilities: impl IntoIterator<Item = (WorkerId, Option<bool>)>,
        statuses: impl IntoIterator<Item = (WorkerWithDpRank, KvSourceStatus)>,
    ) -> KvSourceMembershipView {
        let endpoint = endpoint();
        let sources: HashMap<_, _> = statuses.into_iter().collect();
        KvSourceMembershipView {
            serving_endpoint: endpoint.clone(),
            endpoint_resolution: KvStateEndpointResolution::Resolved(endpoint),
            observation_state,
            lifecycle_generations: sources.keys().map(|worker| (*worker, 0)).collect(),
            recovery_expected: HashMap::new(),
            kv_event_publishing_enabled: capabilities.into_iter().collect(),
            sources,
        }
    }

    fn active(worker: WorkerWithDpRank) -> KvSourceStatus {
        KvSourceStatus::ActiveLiveOnly(KvEventSource {
            kv_state_endpoint: endpoint(),
            worker,
            publisher_id: 17,
            recovery_target: None,
        })
    }

    fn required_state() -> SourceHealthState {
        SourceHealthState::new(KvEventSourceRequirement::CacheAwareRouting)
    }

    #[test]
    fn disabled_capability_warns_immediately_and_once() {
        let worker = WorkerWithDpRank::new(7, 0);
        let view = view(
            KvSourceObservationState::Rebinding,
            [(7, Some(false))],
            [(worker, KvSourceStatus::Missing)],
        );
        let now = Instant::now();
        let mut state = required_state();

        let diagnostics = state.reconcile(&view, now);
        assert_eq!(diagnostics.len(), 1);
        assert_eq!(diagnostics[0].code, DiagnosticCode::PublisherDisabled);
        assert!(!diagnostics[0].kv_event_publishing_enabled);
        assert!(state.reconcile(&view, now).is_empty());
    }

    #[test]
    fn missing_source_waits_full_grace_and_active_source_recovers_once() {
        let worker = WorkerWithDpRank::new(7, 0);
        let missing = view(
            KvSourceObservationState::Bound,
            [(7, Some(true))],
            [(worker, KvSourceStatus::Missing)],
        );
        let now = Instant::now();
        let mut state = required_state();

        assert!(state.reconcile(&missing, now).is_empty());
        let deadline = state.next_deadline().expect("missing source has deadline");
        assert!(
            state
                .fire_deadline(deadline, deadline.deadline - Duration::from_nanos(1))
                .is_empty()
        );
        let diagnostics = state.fire_deadline(deadline, deadline.deadline);
        assert_eq!(diagnostics.len(), 1);
        assert_eq!(diagnostics[0].code, DiagnosticCode::SourceNotObserved);
        assert_eq!(diagnostics[0].waited, SOURCE_JOIN_GRACE);

        let active = view(
            KvSourceObservationState::Bound,
            [(7, Some(true))],
            [(worker, active(worker))],
        );
        let recovered = state.reconcile(&active, deadline.deadline + Duration::from_secs(1));
        assert_eq!(recovered.len(), 1);
        assert_eq!(recovered[0].code, DiagnosticCode::SourceRecovered);
        assert!(
            state
                .reconcile(&active, deadline.deadline + Duration::from_secs(32))
                .is_empty()
        );
    }

    #[test]
    fn rebinding_resets_pending_grace_and_fences_old_deadline() {
        let worker = WorkerWithDpRank::new(7, 0);
        let bound_missing = view(
            KvSourceObservationState::Bound,
            [(7, Some(true))],
            [(worker, KvSourceStatus::Missing)],
        );
        let rebinding = view(
            KvSourceObservationState::Rebinding,
            [(7, Some(true))],
            [(worker, KvSourceStatus::Missing)],
        );
        let now = Instant::now();
        let mut state = required_state();

        state.reconcile(&bound_missing, now);
        let stale = state.next_deadline().unwrap();
        state.reconcile(&rebinding, now + Duration::from_secs(20));
        assert!(state.next_deadline().is_none());
        state.reconcile(&bound_missing, now + Duration::from_secs(25));
        let fresh = state.next_deadline().unwrap();
        assert_ne!(fresh.epoch, stale.epoch);
        assert!(
            state
                .fire_deadline(stale, now + SOURCE_JOIN_GRACE)
                .is_empty()
        );
        assert_eq!(
            state.fire_deadline(fresh, fresh.deadline)[0].code,
            DiagnosticCode::SourceNotObserved
        );
    }

    #[test]
    fn remove_and_rejoin_fences_old_deadline() {
        let worker = WorkerWithDpRank::new(7, 0);
        let missing = view(
            KvSourceObservationState::Bound,
            [(7, Some(true))],
            [(worker, KvSourceStatus::Missing)],
        );
        let removed = view(
            KvSourceObservationState::Bound,
            std::iter::empty(),
            std::iter::empty(),
        );
        let now = Instant::now();
        let mut state = required_state();

        state.reconcile(&missing, now);
        let stale = state.next_deadline().unwrap();
        state.reconcile(&removed, now + Duration::from_secs(1));
        state.reconcile(&missing, now + Duration::from_secs(2));
        let fresh = state.next_deadline().unwrap();
        assert_ne!(fresh.epoch, stale.epoch);
        assert!(state.fire_deadline(stale, stale.deadline).is_empty());
        assert_eq!(
            state.fire_deadline(fresh, fresh.deadline)[0].code,
            DiagnosticCode::SourceNotObserved
        );
    }

    #[test]
    fn ambiguity_warns_once_then_active_emits_one_recovery() {
        let worker = WorkerWithDpRank::new(7, 0);
        let ambiguity = crate::discovery::KvSourceAmbiguity::Incarnations {
            publisher_ids: vec![1, 2],
        };
        let ambiguous = view(
            KvSourceObservationState::Bound,
            [(7, Some(true))],
            [(worker, KvSourceStatus::Ambiguous(ambiguity))],
        );
        let now = Instant::now();
        let mut state = required_state();

        let diagnostics = state.reconcile(&ambiguous, now);
        assert_eq!(diagnostics[0].code, DiagnosticCode::SourceAmbiguous);
        assert!(state.reconcile(&ambiguous, now).is_empty());

        let active = view(
            KvSourceObservationState::Bound,
            [(7, Some(true))],
            [(worker, active(worker))],
        );
        assert_eq!(
            state.reconcile(&active, now + Duration::from_secs(1))[0].code,
            DiagnosticCode::SourceRecovered
        );
        assert!(
            state
                .reconcile(&active, now + Duration::from_secs(2))
                .is_empty()
        );
    }

    #[test]
    fn endpoint_mapping_ambiguity_warns_while_source_watch_is_unbound() {
        let worker = WorkerWithDpRank::new(7, 0);
        let ambiguous = view(
            KvSourceObservationState::Rebinding,
            [(7, Some(true))],
            [(
                worker,
                KvSourceStatus::Ambiguous(KvSourceAmbiguity::EndpointMapping {
                    endpoints: vec![endpoint()],
                }),
            )],
        );
        let now = Instant::now();
        let mut state = required_state();

        let diagnostics = state.reconcile(&ambiguous, now);
        assert_eq!(diagnostics.len(), 1);
        assert_eq!(diagnostics[0].code, DiagnosticCode::SourceAmbiguous);
        assert!(state.reconcile(&ambiguous, now).is_empty());
    }

    #[test]
    fn unknown_capability_and_unknown_requirement_are_silent() {
        let worker = WorkerWithDpRank::new(7, 0);
        let unknown_capability = view(
            KvSourceObservationState::Bound,
            [(7, None)],
            [(worker, KvSourceStatus::Missing)],
        );
        let now = Instant::now();
        let mut required = required_state();
        assert!(required.reconcile(&unknown_capability, now).is_empty());
        assert!(required.next_deadline().is_none());

        let disabled = view(
            KvSourceObservationState::Bound,
            [(7, Some(false))],
            [(worker, KvSourceStatus::Missing)],
        );
        let mut unknown = SourceHealthState::new(KvEventSourceRequirement::Unknown);
        assert!(unknown.reconcile(&disabled, now).is_empty());
        assert!(unknown.next_deadline().is_none());
    }

    #[test]
    fn diagnostics_aggregate_by_worker_and_code() {
        let diagnostics = aggregate_diagnostics(vec![
            RankDiagnostic {
                code: DiagnosticCode::SourceNotObserved,
                worker: WorkerWithDpRank::new(7, 3),
                kv_event_publishing_enabled: true,
                waited: Duration::from_secs(30),
            },
            RankDiagnostic {
                code: DiagnosticCode::SourceNotObserved,
                worker: WorkerWithDpRank::new(7, 1),
                kv_event_publishing_enabled: true,
                waited: Duration::from_secs(31),
            },
        ]);

        assert_eq!(diagnostics.len(), 1);
        assert_eq!(diagnostics[0].worker_id, 7);
        assert_eq!(diagnostics[0].waited_ms, 31_000);
        assert_eq!(diagnostics[0].dp_ranks, vec![1, 3]);
    }
}
