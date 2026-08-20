// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::collections::{BTreeSet, HashMap, VecDeque};

use rustc_hash::FxHashSet;
use serde::Deserialize;

use super::{
    AdmissionAction, AdmissionCohort, AdmissionDecision, AdmissionEvent, AdmissionId,
    AdmissionPolicyConfig, AdmissionPopulationClose, AdmissionRequest, PolicyClassAdmissionPolicy,
    WorkerEligibility, WorkerPlacement,
};
use crate::protocols::{WorkerId, WorkerWithDpRank};

pub const RANK_BALANCED_COHORT_POLICY_TYPE: &str = "rank_balanced_cohort";
pub const RANK_BALANCED_COHORT_BYPASS_POLICY_CLASS: &str = "prefill-rank-balanced-passthrough";
const MAX_ACTIVE_POPULATIONS: usize = 4096;
const COMPLETED_POPULATION_RETENTION: usize = 4096;

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct RankBalancedCohortOptions {
    cohort_size: u32,
    #[serde(default)]
    explicit_population_only: bool,
}

struct PendingAdmission {
    id: AdmissionId,
    eligibility: WorkerEligibility,
    pinned_worker: Option<WorkerWithDpRank>,
}

#[derive(Default)]
struct PendingPopulation {
    final_count: Option<u64>,
    seen_indices: BTreeSet<u64>,
    pending: VecDeque<PendingAdmission>,
}

impl PendingPopulation {
    fn is_complete(&self) -> bool {
        let Some(final_count) = self.final_count else {
            return false;
        };
        self.seen_indices.len() == final_count as usize
            && self.seen_indices.iter().copied().eq(0..final_count)
    }
}

/// Forms a full-DP population and assigns its ranks in one scheduler-owned
/// transaction.
///
/// Eligible requests remain deferred until a single worker can accept one
/// member on every rank in `[0, cohort_size)`. In explicit-population mode,
/// exact affinity placements may participate without being changed; a closed
/// residual tail preserves its exact placement and carries no cohort metadata.
/// Unscoped exact placements bypass for compatibility. No timeout is used to
/// infer a population boundary.
pub struct RankBalancedCohortAdmissionPolicy {
    cohort_size: u32,
    namespace: u64,
    next_cohort: u64,
    explicit_population_only: bool,
    unscoped_pending: VecDeque<PendingAdmission>,
    populations: HashMap<String, PendingPopulation>,
    completed_populations: HashMap<String, u64>,
    completed_population_order: VecDeque<String>,
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
        Self::new_with_population_mode(
            options.cohort_size,
            namespace,
            options.explicit_population_only,
        )
    }

    pub fn new(cohort_size: u32, namespace: u64) -> Result<Self, String> {
        Self::new_with_population_mode(cohort_size, namespace, false)
    }

    pub fn new_with_population_mode(
        cohort_size: u32,
        namespace: u64,
        explicit_population_only: bool,
    ) -> Result<Self, String> {
        if cohort_size <= 1 {
            return Err("rank-balanced cohort size must be greater than one".to_string());
        }
        Ok(Self {
            cohort_size,
            namespace,
            next_cohort: 0,
            explicit_population_only,
            unscoped_pending: VecDeque::new(),
            populations: HashMap::new(),
            completed_populations: HashMap::new(),
            completed_population_order: VecDeque::new(),
        })
    }

    fn remove_pending(&mut self, id: AdmissionId) {
        self.unscoped_pending.retain(|pending| pending.id != id);
        for population in self.populations.values_mut() {
            population.pending.retain(|pending| pending.id != id);
        }
    }

    fn make_ready(&mut self) -> Vec<AdmissionAction> {
        let mut actions = make_ready_cohorts(
            &mut self.unscoped_pending,
            self.cohort_size,
            self.namespace,
            &mut self.next_cohort,
        );
        let mut population_ids = self.populations.keys().cloned().collect::<Vec<_>>();
        population_ids.sort_unstable();
        let mut completed = Vec::new();
        for population_id in population_ids {
            let population = self
                .populations
                .get_mut(&population_id)
                .expect("population key came from the same map");
            actions.extend(make_ready_cohorts(
                &mut population.pending,
                self.cohort_size,
                self.namespace,
                &mut self.next_cohort,
            ));
            if population.is_complete() {
                actions.extend(population.pending.drain(..).map(|pending| {
                    AdmissionAction::MakeReady {
                        id: pending.id,
                        placement: pending
                            .pinned_worker
                            .map_or(WorkerPlacement::Any, WorkerPlacement::Exact),
                        cohort: None,
                    }
                }));
                completed.push((
                    population_id,
                    population
                        .final_count
                        .expect("complete population has a final count"),
                ));
            }
        }
        for (population_id, final_count) in completed {
            self.populations.remove(&population_id);
            self.remember_completed_population(population_id, final_count);
        }
        actions
    }

    fn remember_completed_population(&mut self, id: String, final_count: u64) {
        self.completed_populations.insert(id.clone(), final_count);
        self.completed_population_order.push_back(id);
        while self.completed_population_order.len() > COMPLETED_POPULATION_RETENTION {
            if let Some(expired) = self.completed_population_order.pop_front() {
                self.completed_populations.remove(&expired);
            }
        }
    }
}

fn make_ready_cohorts(
    pending: &mut VecDeque<PendingAdmission>,
    cohort_size: u32,
    namespace: u64,
    next_cohort: &mut u64,
) -> Vec<AdmissionAction> {
    let mut actions = Vec::new();
    while let Some((worker_id, matching)) = find_cohort(pending, cohort_size) {
        let cohort_id = format!(
            "dynamo-router-rank-wave:{:016x}:{:016x}",
            namespace, *next_cohort
        );
        *next_cohort = (*next_cohort).wrapping_add(1);

        let selected_ids: FxHashSet<_> = matching
            .iter()
            .map(|(request_index, _)| pending[*request_index].id)
            .collect();
        for (request_index, rank) in matching {
            let id = pending[request_index].id;
            let cohort = AdmissionCohort::new(cohort_id.clone(), cohort_size, rank)
                .expect("rank-balanced policy constructs a valid cohort");
            actions.push(AdmissionAction::MakeReady {
                id,
                placement: WorkerPlacement::Exact(WorkerWithDpRank::new(worker_id, rank)),
                cohort: Some(cohort),
            });
        }
        pending.retain(|pending| !selected_ids.contains(&pending.id));
    }
    actions
}

/// Returns `(worker_id, [(pending_index, rank)])` for the earliest prefix
/// that has a perfect request-to-rank matching.
fn find_cohort(
    pending: &VecDeque<PendingAdmission>,
    cohort_size: u32,
) -> Option<(WorkerId, Vec<(usize, u32)>)> {
    if pending.len() < cohort_size as usize {
        return None;
    }

    let snapshots = pending
        .iter()
        .map(|pending| pending.eligibility.snapshot())
        .collect::<Vec<_>>();
    let candidate_workers = snapshots
        .iter()
        .flat_map(|snapshot| snapshot.structural_workers())
        .filter(|worker| worker.dp_rank < cohort_size)
        .map(|worker| worker.worker_id)
        .collect::<BTreeSet<_>>();

    for worker_id in candidate_workers {
        let allowed = snapshots
            .iter()
            .map(|snapshot| {
                (0..cohort_size)
                    .map(|rank| {
                        snapshot.structurally_allows(WorkerWithDpRank::new(worker_id, rank))
                    })
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        let mut rank_to_request = vec![None; cohort_size as usize];
        for request_index in 0..allowed.len() {
            let mut seen_ranks = vec![false; cohort_size as usize];
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
        let Some(population) = request.population().cloned() else {
            if self.explicit_population_only || request.pinned_worker().is_some() {
                return AdmissionDecision::Bypass;
            }
            self.unscoped_pending.push_back(PendingAdmission {
                id: request.id(),
                eligibility: request.worker_eligibility().clone(),
                pinned_worker: None,
            });
            return AdmissionDecision::Defer;
        };

        if self.completed_populations.contains_key(population.id()) {
            return AdmissionDecision::Reject(format!(
                "admission population {:?} is already complete",
                population.id()
            ));
        }
        if !self.populations.contains_key(population.id())
            && self.populations.len() >= MAX_ACTIVE_POPULATIONS
        {
            return AdmissionDecision::Reject(format!(
                "rank-balanced admission reached its limit of {MAX_ACTIVE_POPULATIONS} active populations"
            ));
        }

        let state = self
            .populations
            .entry(population.id().to_string())
            .or_default();
        if state
            .final_count
            .is_some_and(|final_count| population.index() >= final_count)
        {
            return AdmissionDecision::Reject(format!(
                "admission population {:?} index {} is outside its closed range",
                population.id(),
                population.index()
            ));
        }
        if !state.seen_indices.insert(population.index()) {
            return AdmissionDecision::Reject(format!(
                "admission population {:?} contains duplicate index {}",
                population.id(),
                population.index()
            ));
        }
        state.pending.push_back(PendingAdmission {
            id: request.id(),
            eligibility: request.worker_eligibility().clone(),
            pinned_worker: request.pinned_worker(),
        });
        AdmissionDecision::Defer
    }

    fn close_population(
        &mut self,
        close: AdmissionPopulationClose,
    ) -> Result<Vec<AdmissionAction>, String> {
        if let Some(existing) = self.completed_populations.get(close.id()) {
            return if *existing == close.final_count() {
                Ok(Vec::new())
            } else {
                Err(format!(
                    "admission population {:?} was already closed at {existing}, not {}",
                    close.id(),
                    close.final_count()
                ))
            };
        }
        if !self.populations.contains_key(close.id())
            && self.populations.len() >= MAX_ACTIVE_POPULATIONS
        {
            return Err(format!(
                "rank-balanced admission reached its limit of {MAX_ACTIVE_POPULATIONS} active populations"
            ));
        }
        let state = self.populations.entry(close.id().to_string()).or_default();
        if let Some(existing) = state.final_count {
            if existing != close.final_count() {
                return Err(format!(
                    "admission population {:?} was already closed at {existing}, not {}",
                    close.id(),
                    close.final_count()
                ));
            }
            return Ok(self.make_ready());
        }
        if let Some(index) = state
            .seen_indices
            .iter()
            .copied()
            .find(|index| *index >= close.final_count())
        {
            return Err(format!(
                "admission population {:?} already observed index {index} outside close count {}",
                close.id(),
                close.final_count()
            ));
        }
        state.final_count = Some(close.final_count());
        Ok(self.make_ready())
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
    use crate::scheduling::queue_admission::{
        AdmissionPopulationMember, AdmissionRequest, WorkerEligibilitySnapshot,
    };

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

    fn admit_population_member(
        policy: &mut RankBalancedCohortAdmissionPolicy,
        admission_id: u64,
        population_id: &str,
        population_index: u64,
        workers: Vec<WorkerWithDpRank>,
        pinned_worker: Option<WorkerWithDpRank>,
    ) -> AdmissionDecision {
        policy.admit(
            AdmissionRequest::new(
                AdmissionId::new(admission_id),
                None,
                1,
                eligibility(workers),
            )
            .with_pinned_worker(pinned_worker)
            .with_population(Some(
                AdmissionPopulationMember::new(population_id.to_string(), population_index)
                    .unwrap(),
            )),
        )
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
        assert!(policy.unscoped_pending.is_empty());
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
        assert!(policy.unscoped_pending.is_empty());
    }

    #[test]
    fn validates_cohort_size() {
        assert!(RankBalancedCohortAdmissionPolicy::new(1, 7).is_err());
    }

    #[test]
    fn explicit_mode_bypasses_unscoped_requests() {
        let mut policy =
            RankBalancedCohortAdmissionPolicy::new_with_population_mode(4, 7, true).unwrap();
        let request =
            AdmissionRequest::new(AdmissionId::new(0), None, 1, eligibility(full_worker(11)));

        assert_eq!(policy.admit(request), AdmissionDecision::Bypass);
        assert!(policy.unscoped_pending.is_empty());
    }

    #[test]
    fn close_releases_terminal_populations_smaller_than_cohort() {
        for final_count in 1..4 {
            let mut policy =
                RankBalancedCohortAdmissionPolicy::new_with_population_mode(4, 7, true).unwrap();
            for index in 0..final_count {
                assert_eq!(
                    admit_population_member(
                        &mut policy,
                        index,
                        "tail",
                        index,
                        full_worker(11),
                        None,
                    ),
                    AdmissionDecision::Defer
                );
                assert!(policy.on_event(AdmissionEvent::Reconcile).is_empty());
            }

            let actions = policy
                .close_population(
                    AdmissionPopulationClose::new("tail".to_string(), final_count).unwrap(),
                )
                .unwrap();
            assert_eq!(actions.len(), final_count as usize);
            assert!(actions.into_iter().all(|action| matches!(
                action,
                AdmissionAction::MakeReady {
                    placement: WorkerPlacement::Any,
                    cohort: None,
                    ..
                }
            )));
        }
    }

    #[test]
    fn complete_cohorts_release_before_close_and_close_releases_only_tail() {
        let mut policy =
            RankBalancedCohortAdmissionPolicy::new_with_population_mode(4, 7, true).unwrap();
        let mut cohort_actions = Vec::new();
        for index in 0..5 {
            assert_eq!(
                admit_population_member(&mut policy, index, "five", index, full_worker(11), None,),
                AdmissionDecision::Defer
            );
            cohort_actions.extend(policy.on_event(AdmissionEvent::Reconcile));
        }
        assert_eq!(cohort_actions.len(), 4);
        assert!(cohort_actions.iter().all(|action| matches!(
            action,
            AdmissionAction::MakeReady {
                cohort: Some(_),
                ..
            }
        )));

        let tail = policy
            .close_population(AdmissionPopulationClose::new("five".to_string(), 5).unwrap())
            .unwrap();
        assert!(matches!(
            tail.as_slice(),
            [AdmissionAction::MakeReady {
                placement: WorkerPlacement::Any,
                cohort: None,
                ..
            }]
        ));
    }

    #[test]
    fn close_can_arrive_before_out_of_order_members_without_early_flush() {
        let mut policy =
            RankBalancedCohortAdmissionPolicy::new_with_population_mode(4, 7, true).unwrap();
        assert!(
            policy
                .close_population(AdmissionPopulationClose::new("race".to_string(), 3).unwrap())
                .unwrap()
                .is_empty()
        );
        for (admission_id, index) in [(0, 2), (1, 0)] {
            assert_eq!(
                admit_population_member(
                    &mut policy,
                    admission_id,
                    "race",
                    index,
                    full_worker(11),
                    None,
                ),
                AdmissionDecision::Defer
            );
            assert!(policy.on_event(AdmissionEvent::Reconcile).is_empty());
        }
        assert_eq!(
            admit_population_member(&mut policy, 2, "race", 1, full_worker(11), None),
            AdmissionDecision::Defer
        );
        let actions = policy.on_event(AdmissionEvent::Reconcile);
        assert_eq!(actions.len(), 3);
        assert!(
            actions
                .iter()
                .all(|action| matches!(action, AdmissionAction::MakeReady { cohort: None, .. }))
        );
    }

    #[test]
    fn populations_are_isolated_and_never_form_cross_population_cohorts() {
        let mut policy =
            RankBalancedCohortAdmissionPolicy::new_with_population_mode(4, 7, true).unwrap();
        for (admission_id, population_id, index) in
            [(0, "a", 0), (1, "b", 0), (2, "a", 1), (3, "b", 1)]
        {
            assert_eq!(
                admit_population_member(
                    &mut policy,
                    admission_id,
                    population_id,
                    index,
                    full_worker(11),
                    None,
                ),
                AdmissionDecision::Defer
            );
        }
        assert!(policy.on_event(AdmissionEvent::Reconcile).is_empty());
        assert_eq!(
            policy
                .close_population(AdmissionPopulationClose::new("a".to_string(), 2).unwrap())
                .unwrap()
                .len(),
            2
        );
        assert_eq!(policy.populations["b"].pending.len(), 2);
    }

    #[test]
    fn explicit_pinned_members_form_only_a_matching_full_rank_cohort() {
        let mut policy =
            RankBalancedCohortAdmissionPolicy::new_with_population_mode(4, 7, true).unwrap();
        for rank in [2, 0, 3, 1] {
            let worker = WorkerWithDpRank::new(11, rank);
            assert_eq!(
                admit_population_member(
                    &mut policy,
                    rank as u64,
                    "pinned",
                    rank as u64,
                    vec![worker],
                    Some(worker),
                ),
                AdmissionDecision::Defer
            );
        }
        let actions = policy.on_event(AdmissionEvent::Reconcile);
        assert_eq!(actions.len(), 4);
        let mut ranks = actions
            .into_iter()
            .map(|action| {
                let AdmissionAction::MakeReady {
                    placement: WorkerPlacement::Exact(worker),
                    cohort: Some(cohort),
                    ..
                } = action
                else {
                    panic!("pinned population member lost its full-rank cohort")
                };
                assert_eq!(worker.dp_rank, cohort.index());
                worker.dp_rank
            })
            .collect::<Vec<_>>();
        ranks.sort_unstable();
        assert_eq!(ranks, vec![0, 1, 2, 3]);
    }

    #[test]
    fn duplicate_member_and_contradictory_close_are_rejected() {
        let mut policy =
            RankBalancedCohortAdmissionPolicy::new_with_population_mode(4, 7, true).unwrap();
        assert_eq!(
            admit_population_member(&mut policy, 0, "bad", 0, full_worker(11), None),
            AdmissionDecision::Defer
        );
        assert!(matches!(
            admit_population_member(&mut policy, 1, "bad", 0, full_worker(11), None),
            AdmissionDecision::Reject(message) if message.contains("duplicate index")
        ));
        policy
            .close_population(AdmissionPopulationClose::new("bad".to_string(), 1).unwrap())
            .unwrap();
        assert!(
            policy
                .close_population(AdmissionPopulationClose::new("bad".to_string(), 2).unwrap())
                .unwrap_err()
                .contains("already closed")
        );
    }

    #[test]
    fn completed_population_close_is_idempotent_and_late_members_fail_closed() {
        let mut policy =
            RankBalancedCohortAdmissionPolicy::new_with_population_mode(4, 7, true).unwrap();
        assert_eq!(
            admit_population_member(&mut policy, 0, "done", 0, full_worker(11), None),
            AdmissionDecision::Defer
        );
        let close = AdmissionPopulationClose::new("done".to_string(), 1).unwrap();
        assert_eq!(policy.close_population(close.clone()).unwrap().len(), 1);
        assert!(policy.close_population(close).unwrap().is_empty());
        assert!(matches!(
            admit_population_member(&mut policy, 1, "done", 0, full_worker(11), None),
            AdmissionDecision::Reject(message) if message.contains("already complete")
        ));
        assert!(
            policy
                .close_population(AdmissionPopulationClose::new("done".to_string(), 2).unwrap())
                .unwrap_err()
                .contains("already closed")
        );
    }

    #[test]
    fn aborted_member_counts_as_observed_but_is_not_released() {
        let mut policy =
            RankBalancedCohortAdmissionPolicy::new_with_population_mode(4, 7, true).unwrap();
        for index in 0..3 {
            assert_eq!(
                admit_population_member(&mut policy, index, "cancel", index, full_worker(11), None,),
                AdmissionDecision::Defer
            );
        }
        assert!(
            policy
                .on_event(AdmissionEvent::Aborted {
                    id: AdmissionId::new(1),
                })
                .is_empty()
        );
        let actions = policy
            .close_population(AdmissionPopulationClose::new("cancel".to_string(), 3).unwrap())
            .unwrap();
        assert_eq!(actions.len(), 2);
        assert!(actions.iter().all(|action| match action {
            AdmissionAction::MakeReady { id, cohort, .. } => {
                *id != AdmissionId::new(1) && cohort.is_none()
            }
        }));
    }
}
