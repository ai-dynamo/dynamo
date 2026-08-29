// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! The serving-readiness contract shared by the request plane and the KV DC Relay.
//!
//! Both consumers must answer "can this namespace serve this model" identically, so the
//! card normalization and the readiness evaluation live here as pure functions over
//! [`ReadinessUnit`]s; `Model::evaluate_namespace` and the relay topology projection are
//! adapters that build units from their own liveness sources.

use std::collections::HashSet;

use crate::model_card::ModelDeploymentCard;
use crate::worker_type::WorkerType;

/// The readiness-relevant projection of one WorkerSet: one distinct worker card shape
/// with the number of live workers behind it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ReadinessUnit {
    /// `None` is a legacy card without a declared worker type.
    pub worker_type: Option<WorkerType>,
    pub live_count: usize,
    pub needs: Vec<Vec<WorkerType>>,
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct ReadinessEval {
    pub ready: bool,
    pub has_legacy: bool,
    pub legacy_live_workers: usize,
    pub present: HashSet<WorkerType>,
    pub missing: HashSet<WorkerType>,
    pub ambiguous: HashSet<WorkerType>,
}

/// Project the topology implicit in a pre-`worker_type` prefill card into the
/// explicit contract used by current workers.
///
/// TODO(v1.5): Remove this projection together with the missing-role fallback
/// in `effective_worker_type` and the legacy readiness bypass after the v1.2
/// MDC compatibility window expires.
pub fn normalize_legacy_prefill_topology(card: &mut ModelDeploymentCard) {
    if card.worker_type.is_some() || !card.model_type.supports_prefill() {
        return;
    }

    card.worker_type = Some(WorkerType::Prefill);
    if card.needs.is_empty() {
        card.needs = vec![vec![WorkerType::Decode]];
    }
}

/// Whether a namespace's units are ready to serve traffic.
pub fn evaluate_readiness(units: &[ReadinessUnit]) -> ReadinessEval {
    let mut present: HashSet<WorkerType> = HashSet::new();
    let mut missing: HashSet<WorkerType> = HashSet::new();
    let mut has_legacy = false;
    let mut legacy_live_workers = 0usize;
    let mut has_live_worker = false;
    let mut live_units_by_type = std::collections::HashMap::new();

    // First pass: which worker types have a live worker (+ legacy detection).
    for unit in units {
        if unit.live_count > 0 {
            has_live_worker = true;
        }
        match unit.worker_type {
            Some(worker_type) => {
                if unit.live_count > 0 {
                    present.insert(worker_type);
                    *live_units_by_type.entry(worker_type).or_insert(0usize) += 1;
                }
            }
            // No declared worker_type → legacy card.
            None => {
                has_legacy = true;
                legacy_live_workers += unit.live_count;
            }
        }
    }

    // COMPAT branch: a legacy card disables strict gating; the disaggregated
    // worker types can't be reconstructed, so ready iff any worker is live.
    if has_legacy {
        return ReadinessEval {
            ready: has_live_worker,
            has_legacy,
            legacy_live_workers,
            present,
            missing,
            ambiguous: HashSet::new(),
        };
    }

    let ambiguous = live_units_by_type
        .into_iter()
        .filter_map(|(worker_type, count)| {
            (worker_type != WorkerType::Aggregated && count > 1).then_some(worker_type)
        })
        .collect::<HashSet<_>>();

    // Strict path: a registered worker type with no live worker anywhere is
    // missing; a *live* unit whose `needs` DNF is unsatisfied flags its
    // absent peers.
    for unit in units {
        let Some(worker_type) = unit.worker_type else {
            continue;
        };
        if !present.contains(&worker_type) {
            missing.insert(worker_type);
        }
        if unit.live_count == 0 || unit.needs.is_empty() {
            continue;
        }
        let satisfied = unit
            .needs
            .iter()
            .any(|alt| alt.iter().all(|t| present.contains(t)));
        if !satisfied {
            for alt in &unit.needs {
                for t in alt {
                    if !present.contains(t) {
                        missing.insert(*t);
                    }
                }
            }
        }
    }

    ReadinessEval {
        ready: has_live_worker && missing.is_empty() && ambiguous.is_empty(),
        has_legacy,
        legacy_live_workers,
        present,
        missing,
        ambiguous,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn unit(
        worker_type: Option<WorkerType>,
        live_count: usize,
        needs: Vec<Vec<WorkerType>>,
    ) -> ReadinessUnit {
        ReadinessUnit {
            worker_type,
            live_count,
            needs,
        }
    }

    #[test]
    fn evaluator_matches_aggregated_pd_epd_and_dead_worker_semantics() {
        let aggregated = evaluate_readiness(&[unit(Some(WorkerType::Aggregated), 1, vec![])]);
        assert!(aggregated.ready);
        assert!(aggregated.present.contains(&WorkerType::Aggregated));

        let pd = evaluate_readiness(&[
            unit(Some(WorkerType::Prefill), 1, vec![vec![WorkerType::Decode]]),
            unit(Some(WorkerType::Decode), 1, vec![vec![WorkerType::Prefill]]),
        ]);
        assert!(pd.ready);

        let epd = evaluate_readiness(&[
            unit(
                Some(WorkerType::Encode),
                1,
                vec![
                    vec![WorkerType::Prefill, WorkerType::Decode],
                    vec![WorkerType::Aggregated],
                ],
            ),
            unit(Some(WorkerType::Prefill), 1, vec![]),
            unit(Some(WorkerType::Decode), 1, vec![]),
        ]);
        assert!(epd.ready);

        let missing_decode = evaluate_readiness(&[
            unit(Some(WorkerType::Prefill), 1, vec![vec![WorkerType::Decode]]),
            unit(Some(WorkerType::Decode), 0, vec![vec![WorkerType::Prefill]]),
        ]);
        assert!(!missing_decode.ready);
        assert!(missing_decode.missing.contains(&WorkerType::Decode));
        assert_eq!(missing_decode.missing.len(), 1);
    }

    #[test]
    fn evaluator_matches_legacy_fallback_for_legacy_only_and_mixed_inputs() {
        for units in [
            vec![unit(None, 1, vec![])],
            vec![
                unit(None, 1, vec![]),
                unit(Some(WorkerType::Decode), 0, vec![vec![WorkerType::Prefill]]),
            ],
        ] {
            let evaluation = evaluate_readiness(&units);
            assert!(evaluation.ready);
            assert!(evaluation.has_legacy);
            assert!(evaluation.missing.is_empty());
        }
        let unavailable = evaluate_readiness(&[unit(None, 0, vec![])]);
        assert!(!unavailable.ready);
        assert!(unavailable.has_legacy);
    }

    #[test]
    fn evaluator_ignores_needs_of_dead_workers_but_reports_their_declared_role() {
        let evaluation = evaluate_readiness(&[
            unit(Some(WorkerType::Aggregated), 1, vec![]),
            unit(
                Some(WorkerType::Encode),
                0,
                vec![vec![WorkerType::Prefill, WorkerType::Decode]],
            ),
        ]);
        assert!(!evaluation.ready);
        assert_eq!(evaluation.missing, HashSet::from([WorkerType::Encode]),);
    }

    #[test]
    fn duplicate_live_units_of_one_role_are_ambiguous_and_not_ready() {
        let evaluation = evaluate_readiness(&[
            unit(Some(WorkerType::Prefill), 1, vec![vec![WorkerType::Decode]]),
            unit(Some(WorkerType::Prefill), 1, vec![vec![WorkerType::Decode]]),
            unit(Some(WorkerType::Decode), 1, vec![vec![WorkerType::Prefill]]),
        ]);
        assert!(!evaluation.ready);
        assert_eq!(evaluation.ambiguous, HashSet::from([WorkerType::Prefill]));

        let scale_out = evaluate_readiness(&[
            unit(Some(WorkerType::Aggregated), 1, vec![]),
            unit(Some(WorkerType::Aggregated), 1, vec![]),
        ]);
        assert!(scale_out.ready);
        assert!(scale_out.ambiguous.is_empty());
    }

    #[test]
    fn normalization_projects_legacy_prefill_cards() {
        use crate::model_card::ModelDeploymentCard;
        use crate::model_type::ModelType;

        let mut legacy_prefill = ModelDeploymentCard::with_name_only("m");
        legacy_prefill.model_type = ModelType::Prefill;
        normalize_legacy_prefill_topology(&mut legacy_prefill);
        assert_eq!(legacy_prefill.worker_type, Some(WorkerType::Prefill));
        assert_eq!(legacy_prefill.needs, vec![vec![WorkerType::Decode]]);

        let mut plain_legacy = ModelDeploymentCard::with_name_only("m");
        normalize_legacy_prefill_topology(&mut plain_legacy);
        assert_eq!(plain_legacy.worker_type, None);
    }

    #[test]
    fn empty_topology_is_not_ready() {
        let evaluation = evaluate_readiness(&[]);
        assert!(!evaluation.ready);
        assert!(!evaluation.has_legacy);
        assert!(evaluation.present.is_empty());
        assert!(evaluation.missing.is_empty());
    }
}
