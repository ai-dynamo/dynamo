// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::{
    collections::{BTreeMap, HashMap},
    time::Duration,
};

use dynamo_kv_router::protocols::WorkerId;
use dynamo_runtime::protocols::EndpointId;
use tokio::{sync::oneshot, time::Instant};
use tokio_util::sync::CancellationToken;

use crate::{
    discovery::{KvSourceMembershipView, KvSourceMembershipWatch, KvSourceStatus},
    kv_router::KvEventSourceRequirement,
    worker_type::WorkerType,
};

const SOURCE_JOIN_GRACE: Duration = Duration::from_secs(30);

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
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
struct Diagnostic {
    code: DiagnosticCode,
    worker_id: WorkerId,
    kv_event_publishing_enabled: bool,
    waited: Duration,
    dp_ranks: Vec<u32>,
}

#[derive(Debug, Clone)]
enum WorkerIssue {
    MissingSince(Instant),
    Warned {
        code: DiagnosticCode,
        since: Instant,
        dp_ranks: Vec<u32>,
    },
}

struct SourceHealthState {
    requirement: KvEventSourceRequirement,
    issues: HashMap<WorkerId, WorkerIssue>,
}

impl SourceHealthState {
    fn new(requirement: KvEventSourceRequirement) -> Self {
        Self {
            requirement,
            issues: HashMap::new(),
        }
    }

    fn reconcile(&mut self, view: &KvSourceMembershipView, now: Instant) -> Vec<Diagnostic> {
        if !self.requirement.requires_source() {
            self.issues.clear();
            return Vec::new();
        }

        let mut workers = BTreeMap::<WorkerId, Vec<(u32, &KvSourceStatus)>>::new();
        for (worker, status) in &view.sources {
            workers
                .entry(worker.worker_id)
                .or_default()
                .push((worker.dp_rank, status));
        }
        self.issues
            .retain(|worker_id, _| workers.contains_key(worker_id));

        let mut diagnostics = Vec::new();
        for (worker_id, statuses) in workers {
            let previous = self.issues.remove(&worker_id);
            let capability = view.kv_event_publishing_enabled(worker_id);
            let (next, diagnostic) =
                reconcile_worker(worker_id, statuses, capability, previous, now);
            if let Some(next) = next {
                self.issues.insert(worker_id, next);
            }
            if let Some(diagnostic) = diagnostic {
                diagnostics.push(diagnostic);
            }
        }
        diagnostics
    }

    fn next_deadline(&self) -> Option<Instant> {
        self.issues
            .values()
            .filter_map(|issue| match issue {
                WorkerIssue::MissingSince(since) => Some(*since + SOURCE_JOIN_GRACE),
                WorkerIssue::Warned { .. } => None,
            })
            .min()
    }
}

fn reconcile_worker(
    worker_id: WorkerId,
    mut statuses: Vec<(u32, &KvSourceStatus)>,
    capability: Option<bool>,
    previous: Option<WorkerIssue>,
    now: Instant,
) -> (Option<WorkerIssue>, Option<Diagnostic>) {
    let Some(capability) = capability else {
        return (None, None);
    };

    statuses.sort_unstable_by_key(|(rank, _)| *rank);
    let all_ranks = || statuses.iter().map(|(rank, _)| *rank).collect::<Vec<_>>();

    if !capability {
        return immediate_warning(
            worker_id,
            DiagnosticCode::PublisherDisabled,
            false,
            all_ranks(),
            previous,
            now,
        );
    }

    let ambiguous_ranks = statuses
        .iter()
        .filter_map(|(rank, status)| {
            matches!(status, KvSourceStatus::Ambiguous(_)).then_some(*rank)
        })
        .collect::<Vec<_>>();
    if !ambiguous_ranks.is_empty() {
        return immediate_warning(
            worker_id,
            DiagnosticCode::SourceAmbiguous,
            true,
            ambiguous_ranks,
            previous,
            now,
        );
    }

    let missing_ranks = statuses
        .iter()
        .filter_map(|(rank, status)| matches!(status, KvSourceStatus::Missing).then_some(*rank))
        .collect::<Vec<_>>();
    if !missing_ranks.is_empty() {
        return missing_source(worker_id, missing_ranks, previous, now);
    }

    match previous {
        Some(WorkerIssue::Warned {
            since, dp_ranks, ..
        }) => (
            None,
            Some(Diagnostic {
                code: DiagnosticCode::SourceRecovered,
                worker_id,
                kv_event_publishing_enabled: true,
                waited: now.saturating_duration_since(since),
                dp_ranks,
            }),
        ),
        Some(WorkerIssue::MissingSince(_)) | None => (None, None),
    }
}

fn immediate_warning(
    worker_id: WorkerId,
    code: DiagnosticCode,
    capability: bool,
    dp_ranks: Vec<u32>,
    previous: Option<WorkerIssue>,
    now: Instant,
) -> (Option<WorkerIssue>, Option<Diagnostic>) {
    if matches!(
        previous,
        Some(WorkerIssue::Warned {
            code: current,
            ..
        }) if current == code
    ) {
        return (previous, None);
    }

    (
        Some(WorkerIssue::Warned {
            code,
            since: now,
            dp_ranks: dp_ranks.clone(),
        }),
        Some(Diagnostic {
            code,
            worker_id,
            kv_event_publishing_enabled: capability,
            waited: Duration::ZERO,
            dp_ranks,
        }),
    )
}

fn missing_source(
    worker_id: WorkerId,
    dp_ranks: Vec<u32>,
    previous: Option<WorkerIssue>,
    now: Instant,
) -> (Option<WorkerIssue>, Option<Diagnostic>) {
    match previous {
        Some(
            issue @ WorkerIssue::Warned {
                code: DiagnosticCode::SourceNotObserved,
                ..
            },
        ) => (Some(issue), None),
        Some(WorkerIssue::MissingSince(since))
            if now.saturating_duration_since(since) >= SOURCE_JOIN_GRACE =>
        {
            (
                Some(WorkerIssue::Warned {
                    code: DiagnosticCode::SourceNotObserved,
                    since,
                    dp_ranks: dp_ranks.clone(),
                }),
                Some(Diagnostic {
                    code: DiagnosticCode::SourceNotObserved,
                    worker_id,
                    kv_event_publishing_enabled: true,
                    waited: now.saturating_duration_since(since),
                    dp_ranks,
                }),
            )
        }
        Some(WorkerIssue::MissingSince(since)) => (Some(WorkerIssue::MissingSince(since)), None),
        Some(WorkerIssue::Warned { .. }) | None => (Some(WorkerIssue::MissingSince(now)), None),
    }
}

struct DiagnosticContext<'a> {
    model: &'a str,
    worker_role: Option<WorkerType>,
    requirement: KvEventSourceRequirement,
    serving_endpoint: &'a EndpointId,
}

fn emit_diagnostics(context: &DiagnosticContext<'_>, diagnostics: Vec<Diagnostic>) {
    let worker_role = context
        .worker_role
        .map(|role| role.as_str())
        .unwrap_or("unknown");
    for diagnostic in diagnostics {
        let diagnostic_code = diagnostic.code.as_str();
        let requirement = context.requirement.as_str();
        let waited_ms = diagnostic.waited.as_millis().min(u128::from(u64::MAX)) as u64;
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
                waited_ms,
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
                waited_ms,
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
            let changed = if let Some(deadline) = state.next_deadline() {
                tokio::select! {
                    biased;
                    _ = cancellation_token.cancelled() => break,
                    changed = membership_watch.changed() => Some(changed),
                    _ = tokio::time::sleep_until(deadline) => None,
                }
            } else {
                tokio::select! {
                    biased;
                    _ = cancellation_token.cancelled() => break,
                    changed = membership_watch.changed() => Some(changed),
                }
            };
            if changed.is_some_and(|result| result.is_err()) {
                break;
            }

            let view = membership_watch.borrow_and_update().clone();
            emit_diagnostics(&context, state.reconcile(&view, Instant::now()));
        }
        let _ = completion_tx.send(());
    });
    completion_rx
}

#[cfg(test)]
mod tests {
    use dynamo_kv_router::protocols::WorkerWithDpRank;

    use super::*;
    use crate::discovery::{KvEventSource, KvStateEndpointResolution};

    fn endpoint() -> EndpointId {
        EndpointId {
            namespace: "ns".to_string(),
            component: "worker".to_string(),
            name: "generate".to_string(),
        }
    }

    fn view(
        capabilities: impl IntoIterator<Item = (WorkerId, Option<bool>)>,
        statuses: impl IntoIterator<Item = (WorkerWithDpRank, KvSourceStatus)>,
    ) -> KvSourceMembershipView {
        let endpoint = endpoint();
        let sources: HashMap<_, _> = statuses.into_iter().collect();
        KvSourceMembershipView {
            serving_endpoint: endpoint.clone(),
            endpoint_resolution: KvStateEndpointResolution::Resolved(endpoint),
            lifecycle_generations: sources.keys().map(|worker| (*worker, 0)).collect(),
            recovery_expected: HashMap::new(),
            kv_event_publishing_enabled: capabilities.into_iter().collect(),
            sources,
        }
    }

    fn missing(worker_id: WorkerId, dp_rank: u32) -> (WorkerWithDpRank, KvSourceStatus) {
        (
            WorkerWithDpRank::new(worker_id, dp_rank),
            KvSourceStatus::Missing,
        )
    }

    fn active(worker_id: WorkerId, dp_rank: u32) -> (WorkerWithDpRank, KvSourceStatus) {
        let worker = WorkerWithDpRank::new(worker_id, dp_rank);
        (
            worker,
            KvSourceStatus::ActiveLiveOnly(KvEventSource {
                kv_state_endpoint: endpoint(),
                worker,
                publisher_id: 11,
                recovery_target: None,
            }),
        )
    }

    fn ambiguous(worker_id: WorkerId, dp_rank: u32) -> (WorkerWithDpRank, KvSourceStatus) {
        (
            WorkerWithDpRank::new(worker_id, dp_rank),
            KvSourceStatus::Ambiguous(crate::discovery::KvSourceAmbiguity::Incarnations {
                publisher_ids: vec![1, 2],
            }),
        )
    }

    fn required_state() -> SourceHealthState {
        SourceHealthState::new(KvEventSourceRequirement::CacheAwareRouting)
    }

    #[test]
    fn source_health_state_matrix() {
        struct Case {
            name: &'static str,
            requirement: KvEventSourceRequirement,
            capability: Option<bool>,
            statuses: Vec<(WorkerWithDpRank, KvSourceStatus)>,
            expected_code: Option<DiagnosticCode>,
            expected_ranks: Vec<u32>,
        }

        let cases = [
            Case {
                name: "disabled publisher",
                requirement: KvEventSourceRequirement::CacheAwareRouting,
                capability: Some(false),
                statuses: vec![missing(7, 3), missing(7, 1)],
                expected_code: Some(DiagnosticCode::PublisherDisabled),
                expected_ranks: vec![1, 3],
            },
            Case {
                name: "ambiguous source",
                requirement: KvEventSourceRequirement::CacheAwareRouting,
                capability: Some(true),
                statuses: vec![ambiguous(7, 0)],
                expected_code: Some(DiagnosticCode::SourceAmbiguous),
                expected_ranks: vec![0],
            },
            Case {
                name: "active idle source",
                requirement: KvEventSourceRequirement::CacheAwareRouting,
                capability: Some(true),
                statuses: vec![active(7, 0)],
                expected_code: None,
                expected_ranks: vec![],
            },
            Case {
                name: "unknown capability",
                requirement: KvEventSourceRequirement::CacheAwareRouting,
                capability: None,
                statuses: vec![missing(7, 0)],
                expected_code: None,
                expected_ranks: vec![],
            },
            Case {
                name: "source not required",
                requirement: KvEventSourceRequirement::NotRequired,
                capability: Some(false),
                statuses: vec![missing(7, 0)],
                expected_code: None,
                expected_ranks: vec![],
            },
        ];
        let now = Instant::now();

        for case in cases {
            let source_view = view([(7, case.capability)], case.statuses);
            let mut state = SourceHealthState::new(case.requirement);
            let diagnostics = state.reconcile(&source_view, now);
            assert_eq!(
                diagnostics.first().map(|diagnostic| diagnostic.code),
                case.expected_code,
                "{}",
                case.name
            );
            assert_eq!(
                diagnostics
                    .first()
                    .map(|diagnostic| diagnostic.dp_ranks.as_slice())
                    .unwrap_or_default(),
                case.expected_ranks,
                "{}",
                case.name
            );
            assert!(
                state
                    .reconcile(&source_view, now + SOURCE_JOIN_GRACE * 2)
                    .is_empty(),
                "{}",
                case.name
            );
            assert!(state.next_deadline().is_none(), "{}", case.name);
        }
    }

    #[test]
    fn missing_source_warns_at_grace_boundary_and_recovers_once() {
        let missing = view([(7, Some(true))], [missing(7, 0), missing(7, 1)]);
        let active = view([(7, Some(true))], [active(7, 0), active(7, 1)]);
        let now = Instant::now();
        let mut state = required_state();

        assert!(state.reconcile(&missing, now).is_empty());
        assert!(
            state
                .reconcile(&missing, now + SOURCE_JOIN_GRACE - Duration::from_millis(1))
                .is_empty()
        );
        let warning = state.reconcile(&missing, now + SOURCE_JOIN_GRACE);
        assert_eq!(warning.len(), 1);
        assert_eq!(warning[0].code, DiagnosticCode::SourceNotObserved);
        assert_eq!(warning[0].dp_ranks, vec![0, 1]);

        let recovered = state.reconcile(&active, now + SOURCE_JOIN_GRACE);
        assert_eq!(recovered.len(), 1);
        assert_eq!(recovered[0].code, DiagnosticCode::SourceRecovered);
        assert!(state.reconcile(&active, now + SOURCE_JOIN_GRACE).is_empty());
    }

    #[test]
    fn worker_removal_clears_pending_warning() {
        let missing = view([(7, Some(true))], [missing(7, 0)]);
        let removed = view(
            std::iter::empty::<(WorkerId, Option<bool>)>(),
            std::iter::empty::<(WorkerWithDpRank, KvSourceStatus)>(),
        );
        let now = Instant::now();
        let mut state = required_state();

        assert!(state.reconcile(&missing, now).is_empty());
        assert!(state.reconcile(&removed, now).is_empty());
        assert!(state.next_deadline().is_none());
        assert!(
            state
                .reconcile(&missing, now + SOURCE_JOIN_GRACE)
                .is_empty()
        );
    }
}
