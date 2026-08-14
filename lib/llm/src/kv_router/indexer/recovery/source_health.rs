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

    fn message(self) -> &'static str {
        match self {
            Self::PublisherDisabled => {
                "KV-event publishing is disabled for a role that requires KV events"
            }
            Self::SourceNotObserved => {
                "KV-event source was not observed after the discovery grace period"
            }
            Self::SourceAmbiguous => "KV-event source registration is ambiguous",
            Self::SourceRecovered => "KV-event source recovered",
        }
    }

    fn suggested_action(self) -> &'static str {
        match self {
            Self::PublisherDisabled => {
                "Enable backend KV-event publishing, or select a routing mode that does not require KV events"
            }
            Self::SourceNotObserved => {
                "Check publisher startup, endpoint registration, discovery, and NATS/ZMQ connectivity"
            }
            Self::SourceAmbiguous => {
                "Check for stale or duplicate publisher incarnations and conflicting KV-state endpoint mappings"
            }
            Self::SourceRecovered => "No action required",
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
struct Warning {
    code: DiagnosticCode,
    since: Instant,
    dp_ranks: Vec<u32>,
}

#[derive(Debug, Clone)]
enum WorkerIssue {
    MissingSince {
        since: Instant,
        previous_warning: Option<Warning>,
    },
    Warned(Warning),
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
                WorkerIssue::MissingSince { since, .. } => Some(*since + SOURCE_JOIN_GRACE),
                WorkerIssue::Warned(_) => None,
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

    let previous_warning = match previous {
        Some(WorkerIssue::Warned(warning)) => Some(warning),
        Some(WorkerIssue::MissingSince {
            previous_warning, ..
        }) => previous_warning,
        None => None,
    };
    match previous_warning {
        Some(warning) => (
            None,
            Some(Diagnostic {
                code: DiagnosticCode::SourceRecovered,
                worker_id,
                kv_event_publishing_enabled: true,
                waited: now.saturating_duration_since(warning.since),
                dp_ranks: warning.dp_ranks,
            }),
        ),
        None => (None, None),
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
    let previous_warning = match previous {
        Some(WorkerIssue::Warned(warning)) => Some(warning),
        Some(WorkerIssue::MissingSince {
            previous_warning, ..
        }) => previous_warning,
        None => None,
    };
    if let Some(warning) = previous_warning
        && warning.code == code
    {
        return (Some(WorkerIssue::Warned(warning)), None);
    }

    (
        Some(WorkerIssue::Warned(Warning {
            code,
            since: now,
            dp_ranks: dp_ranks.clone(),
        })),
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
        Some(WorkerIssue::Warned(warning)) if warning.code == DiagnosticCode::SourceNotObserved => {
            (Some(WorkerIssue::Warned(warning)), None)
        }
        Some(WorkerIssue::MissingSince { since, .. })
            if now.saturating_duration_since(since) >= SOURCE_JOIN_GRACE =>
        {
            (
                Some(WorkerIssue::Warned(Warning {
                    code: DiagnosticCode::SourceNotObserved,
                    since,
                    dp_ranks: dp_ranks.clone(),
                })),
                Some(Diagnostic {
                    code: DiagnosticCode::SourceNotObserved,
                    worker_id,
                    kv_event_publishing_enabled: true,
                    waited: now.saturating_duration_since(since),
                    dp_ranks,
                }),
            )
        }
        Some(issue @ WorkerIssue::MissingSince { .. }) => (Some(issue), None),
        Some(WorkerIssue::Warned(warning)) => (
            Some(WorkerIssue::MissingSince {
                since: now,
                previous_warning: Some(warning),
            }),
            None,
        ),
        None => (
            Some(WorkerIssue::MissingSince {
                since: now,
                previous_warning: None,
            }),
            None,
        ),
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
        let diagnostic_message = diagnostic.code.message();
        let suggested_action = diagnostic.code.suggested_action();
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
                suggested_action,
                "{diagnostic_message}"
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
                suggested_action,
                "{diagnostic_message}"
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
            recovery_expected: HashMap::new(),
            kv_event_publishing_enabled: capabilities.into_iter().collect(),
            kv_event_source_mode: HashMap::new(),
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
    fn ambiguous_source_recovers_after_transient_missing_state() {
        let ambiguous = view([(7, Some(true))], [ambiguous(7, 0)]);
        let missing = view([(7, Some(true))], [missing(7, 0)]);
        let active = view([(7, Some(true))], [active(7, 0)]);
        let now = Instant::now();
        let mut state = required_state();

        let warning = state.reconcile(&ambiguous, now);
        assert_eq!(warning.len(), 1);
        assert_eq!(warning[0].code, DiagnosticCode::SourceAmbiguous);

        assert!(
            state
                .reconcile(&missing, now + Duration::from_millis(1))
                .is_empty()
        );

        let recovered = state.reconcile(&active, now + Duration::from_millis(2));
        assert_eq!(recovered.len(), 1);
        assert_eq!(recovered[0].code, DiagnosticCode::SourceRecovered);
        assert_eq!(recovered[0].waited, Duration::from_millis(2));
        assert!(
            state
                .reconcile(&active, now + Duration::from_millis(3))
                .is_empty()
        );
    }

    #[test]
    fn ambiguous_source_that_stays_missing_reports_missing_source() {
        let ambiguous = view([(7, Some(true))], [ambiguous(7, 0)]);
        let missing = view([(7, Some(true))], [missing(7, 0)]);
        let active = view([(7, Some(true))], [active(7, 0)]);
        let now = Instant::now();
        let mut state = required_state();

        let ambiguity_warning = state.reconcile(&ambiguous, now);
        assert_eq!(ambiguity_warning.len(), 1);
        assert_eq!(ambiguity_warning[0].code, DiagnosticCode::SourceAmbiguous);

        let missing_since = now + Duration::from_millis(1);
        assert!(state.reconcile(&missing, missing_since).is_empty());
        let missing_warning = state.reconcile(&missing, missing_since + SOURCE_JOIN_GRACE);
        assert_eq!(missing_warning.len(), 1);
        assert_eq!(missing_warning[0].code, DiagnosticCode::SourceNotObserved);

        let recovered = state.reconcile(&active, missing_since + SOURCE_JOIN_GRACE);
        assert_eq!(recovered.len(), 1);
        assert_eq!(recovered[0].code, DiagnosticCode::SourceRecovered);
        assert_eq!(recovered[0].waited, SOURCE_JOIN_GRACE);
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
