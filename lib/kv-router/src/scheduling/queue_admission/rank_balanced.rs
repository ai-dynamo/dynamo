// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::collections::{BTreeSet, VecDeque};

use rustc_hash::FxHashSet;
use serde::Deserialize;

use super::{
    AdmissionAction, AdmissionCohort, AdmissionDecision, AdmissionEvent, AdmissionId,
    AdmissionPolicyConfig, AdmissionRequest, PolicyClassAdmissionPolicy, WorkerEligibility,
    WorkerPlacement,
};
use crate::protocols::{WorkerId, WorkerWithDpRank};

pub const RANK_BALANCED_COHORT_POLICY_TYPE: &str = "rank_balanced_cohort";
pub const RANK_BALANCED_COHORT_BYPASS_POLICY_CLASS: &str = "prefill-rank-balanced-passthrough";

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct RankBalancedCohortOptions {
    cohort_size: u32,
}

struct PendingAdmission {
    id: AdmissionId,
    eligibility: WorkerEligibility,
}

/// Forms a full-DP population and assigns its ranks in one scheduler-owned
/// transaction.
///
/// Unpinned requests remain deferred until a single worker can accept one
/// member on every rank in `[0, cohort_size)`. Requests whose exact placement
/// is already owned by the caller or conversation-affinity layer bypass this
/// policy; combining them into a synthetic full-rank population would either
/// violate affinity or recreate the missing-rank deadlock this policy fixes.
/// No timeout is used to infer a population boundary.
pub struct RankBalancedCohortAdmissionPolicy {
    cohort_size: u32,
    namespace: u64,
    next_cohort: u64,
    pending: VecDeque<PendingAdmission>,
}

impl RankBalancedCohortAdmissionPolicy {
    pub fn from_config(config: &AdmissionPolicyConfig, namespace: u64) -> Result<Self, String> {
        if config.policy_type() != RANK_BALANCED_COHORT_POLICY_TYPE {
            return Err(format!(
                "expected admission policy type {RANK_BALANCED_COHORT_POLICY_TYPE:?}, got {:?}",
                config.policy_type()
            ));
        }
        let options: RankBalancedCohortOptions =
            serde_yaml::from_value(serde_yaml::Value::Mapping(config.options().clone()))
                .map_err(|error| format!("invalid rank-balanced cohort policy options: {error}"))?;
        Self::new(options.cohort_size, namespace)
    }

    pub fn new(cohort_size: u32, namespace: u64) -> Result<Self, String> {
        if cohort_size <= 1 {
            return Err("rank-balanced cohort size must be greater than one".to_string());
        }
        Ok(Self {
            cohort_size,
            namespace,
            next_cohort: 0,
            pending: VecDeque::new(),
        })
    }

    fn remove_pending(&mut self, id: AdmissionId) {
        self.pending.retain(|pending| pending.id != id);
    }

    fn make_ready(&mut self) -> Vec<AdmissionAction> {
        let mut actions = Vec::new();
        while let Some((worker_id, matching)) = self.find_cohort() {
            let cohort_id = format!(
                "dynamo-router-rank-wave:{:016x}:{:016x}",
                self.namespace, self.next_cohort
            );
            self.next_cohort = self.next_cohort.wrapping_add(1);

            let selected_ids: FxHashSet<_> = matching
                .iter()
                .map(|(request_index, _)| self.pending[*request_index].id)
                .collect();
            for (request_index, rank) in matching {
                let id = self.pending[request_index].id;
                let cohort = AdmissionCohort::new(cohort_id.clone(), self.cohort_size, rank)
                    .expect("rank-balanced policy constructs a valid cohort");
                actions.push(AdmissionAction::MakeReady {
                    id,
                    placement: WorkerPlacement::Exact(WorkerWithDpRank::new(worker_id, rank)),
                    cohort: Some(cohort),
                });
            }
            self.pending
                .retain(|pending| !selected_ids.contains(&pending.id));
        }
        actions
    }

    /// Returns `(worker_id, [(pending_index, rank)])` for the earliest prefix
    /// that has a perfect request-to-rank matching.
    fn find_cohort(&self) -> Option<(WorkerId, Vec<(usize, u32)>)> {
        if self.pending.len() < self.cohort_size as usize {
            return None;
        }

        let snapshots = self
            .pending
            .iter()
            .map(|pending| pending.eligibility.snapshot())
            .collect::<Vec<_>>();
        let candidate_workers = snapshots
            .iter()
            .flat_map(|snapshot| snapshot.structural_workers())
            .filter(|worker| worker.dp_rank < self.cohort_size)
            .map(|worker| worker.worker_id)
            .collect::<BTreeSet<_>>();

        for worker_id in candidate_workers {
            let allowed = snapshots
                .iter()
                .map(|snapshot| {
                    (0..self.cohort_size)
                        .map(|rank| {
                            snapshot.structurally_allows(WorkerWithDpRank::new(worker_id, rank))
                        })
                        .collect::<Vec<_>>()
                })
                .collect::<Vec<_>>();
            let mut rank_to_request = vec![None; self.cohort_size as usize];
            for request_index in 0..allowed.len() {
                let mut seen_ranks = vec![false; self.cohort_size as usize];
                if augment_matching(
                    request_index,
                    &allowed,
                    &mut seen_ranks,
                    &mut rank_to_request,
                ) && rank_to_request.iter().all(Option::is_some)
                {
                    let matching = rank_to_request
                        .into_iter()
                        .enumerate()
                        .map(|(rank, request_index)| {
                            (
                                request_index.expect("perfect matching has every request"),
                                rank as u32,
                            )
                        })
                        .collect();
                    return Some((worker_id, matching));
                }
            }
        }
        None
    }
}

fn augment_matching(
    request_index: usize,
    allowed: &[Vec<bool>],
    seen_ranks: &mut [bool],
    rank_to_request: &mut [Option<usize>],
) -> bool {
    for rank in 0..rank_to_request.len() {
        if !allowed[request_index][rank] || seen_ranks[rank] {
            continue;
        }
        seen_ranks[rank] = true;
        let can_claim = match rank_to_request[rank] {
            None => true,
            Some(previous) => augment_matching(previous, allowed, seen_ranks, rank_to_request),
        };
        if can_claim {
            rank_to_request[rank] = Some(request_index);
            return true;
        }
    }
    false
}

impl PolicyClassAdmissionPolicy for RankBalancedCohortAdmissionPolicy {
    fn admit(&mut self, request: AdmissionRequest<'_>) -> AdmissionDecision {
        if request.pinned_worker().is_some() {
            return AdmissionDecision::Bypass;
        }
        self.pending.push_back(PendingAdmission {
            id: request.id(),
            eligibility: request.worker_eligibility().clone(),
        });
        AdmissionDecision::Defer
    }

    fn on_event(&mut self, event: AdmissionEvent) -> Vec<AdmissionAction> {
        if let AdmissionEvent::Aborted { id } = event {
            self.remove_pending(id);
        }
        self.make_ready()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::scheduling::queue_admission::{AdmissionRequest, WorkerEligibilitySnapshot};

    fn eligibility(workers: Vec<WorkerWithDpRank>) -> WorkerEligibility {
        WorkerEligibility::new(move || WorkerEligibilitySnapshot::new(workers.clone()))
    }

    fn admit(
        policy: &mut RankBalancedCohortAdmissionPolicy,
        id: u64,
        workers: Vec<WorkerWithDpRank>,
    ) {
        assert_eq!(
            policy.admit(AdmissionRequest::new(
                AdmissionId::new(id),
                None,
                1,
                eligibility(workers),
            )),
            AdmissionDecision::Defer
        );
    }

    fn full_worker(worker_id: WorkerId) -> Vec<WorkerWithDpRank> {
        (0..4)
            .map(|rank| WorkerWithDpRank::new(worker_id, rank))
            .collect()
    }

    #[test]
    fn assigns_one_request_to_every_rank_in_one_cohort() {
        let mut policy = RankBalancedCohortAdmissionPolicy::new(4, 7).unwrap();
        for id in 0..4 {
            admit(&mut policy, id, full_worker(11));
        }
        let actions = policy.on_event(AdmissionEvent::Reconcile);
        assert_eq!(actions.len(), 4);
        let mut ranks = Vec::new();
        let mut ids = FxHashSet::default();
        for action in actions {
            let AdmissionAction::MakeReady {
                placement, cohort, ..
            } = action;
            let WorkerPlacement::Exact(worker) = placement else {
                panic!("rank-balanced action must use exact placement");
            };
            let cohort = cohort.expect("rank-balanced action must carry a cohort");
            ranks.push(worker.dp_rank);
            ids.insert(cohort.id().to_string());
            assert_eq!(cohort.index(), worker.dp_rank);
            assert_eq!(cohort.size(), 4);
        }
        ranks.sort_unstable();
        assert_eq!(ranks, vec![0, 1, 2, 3]);
        assert_eq!(ids.len(), 1);
    }

    #[test]
    fn exact_affinity_pin_bypasses_cohort_formation() {
        let mut policy = RankBalancedCohortAdmissionPolicy::new(4, 7).unwrap();
        let pinned = WorkerWithDpRank::new(11, 0);
        let request = AdmissionRequest::new(
            AdmissionId::new(0),
            Some("conversation"),
            1,
            eligibility(vec![pinned]),
        )
        .with_pinned_worker(Some(pinned));
        assert_eq!(policy.admit(request), AdmissionDecision::Bypass);
        assert!(policy.pending.is_empty());
    }

    #[test]
    fn never_combines_ranks_from_different_workers() {
        let mut policy = RankBalancedCohortAdmissionPolicy::new(4, 7).unwrap();
        for (id, rank) in [0, 1].into_iter().enumerate() {
            admit(
                &mut policy,
                id as u64,
                vec![WorkerWithDpRank::new(11, rank)],
            );
        }
        for (id, rank) in [2, 3].into_iter().enumerate() {
            admit(
                &mut policy,
                (id + 2) as u64,
                vec![WorkerWithDpRank::new(12, rank)],
            );
        }
        assert!(policy.on_event(AdmissionEvent::Reconcile).is_empty());
    }

    #[test]
    fn aborted_deferred_request_is_removed() {
        let mut policy = RankBalancedCohortAdmissionPolicy::new(4, 7).unwrap();
        admit(&mut policy, 0, full_worker(11));
        assert!(
            policy
                .on_event(AdmissionEvent::Aborted {
                    id: AdmissionId::new(0),
                })
                .is_empty()
        );
        assert!(policy.pending.is_empty());
    }

    #[test]
    fn validates_cohort_size() {
        assert!(RankBalancedCohortAdmissionPolicy::new(1, 7).is_err());
    }
}
