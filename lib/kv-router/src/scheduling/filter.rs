// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::collections::{HashMap, HashSet};

use super::types::KvSchedulerError;
use crate::protocols::{DpRank, RoutingConstraints, WorkerConfigLike, WorkerId, WorkerWithDpRank};

#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum WorkerEligibilityError {
    #[error("worker {worker_id} is not in allowed worker set")]
    WorkerNotAllowed { worker_id: WorkerId },

    #[error("worker {worker_id} is unavailable")]
    WorkerUnavailable { worker_id: WorkerId },

    #[error("worker {worker_id} dp_rank {dp_rank} is outside [{start}, {end})")]
    DpRankUnavailable {
        worker_id: WorkerId,
        dp_rank: DpRank,
        start: DpRank,
        end: DpRank,
    },

    #[error("worker {worker_id} is overloaded")]
    WorkerOverloaded { worker_id: WorkerId },

    #[error("worker {worker_id} is fenced (dead/deregistered)")]
    WorkerFenced { worker_id: WorkerId },

    #[error("worker {worker_id} does not satisfy routing constraints")]
    RoutingConstraintsUnsatisfied { worker_id: WorkerId },
}

#[derive(Clone, Copy)]
pub struct RoutingEligibility<'a> {
    allowed_worker_ids: Option<&'a HashSet<WorkerId>>,
    overloaded_worker_ids: Option<&'a HashSet<WorkerId>>,
    // DIS-2404: workers fenced on death/deregistration. Unlike `overloaded`
    // (a transient, affinity-ignorable condition), a fenced worker is never
    // eligible on any path — availability overrides cache affinity.
    fenced_worker_ids: Option<&'a HashSet<WorkerId>>,
    pinned_worker: Option<WorkerWithDpRank>,
    routing_constraints: &'a RoutingConstraints,
}

impl<'a> RoutingEligibility<'a> {
    #[inline]
    pub fn new(
        allowed_worker_ids: Option<&'a HashSet<WorkerId>>,
        overloaded_worker_ids: Option<&'a HashSet<WorkerId>>,
        pinned_worker: Option<WorkerWithDpRank>,
        routing_constraints: &'a RoutingConstraints,
    ) -> Self {
        Self {
            allowed_worker_ids,
            overloaded_worker_ids,
            fenced_worker_ids: None,
            pinned_worker,
            routing_constraints,
        }
    }

    /// Attach the set of fenced (dead/deregistered) workers. A fenced worker is
    /// rejected on every eligibility path, including the affinity/pinned path
    /// that otherwise ignores transient overload.
    #[inline]
    pub fn with_fenced_workers(mut self, fenced_worker_ids: Option<&'a HashSet<WorkerId>>) -> Self {
        self.fenced_worker_ids = fenced_worker_ids;
        self
    }

    #[inline]
    pub fn is_worker_fenced(&self, worker_id: WorkerId) -> bool {
        self.fenced_worker_ids
            .is_some_and(|worker_ids| worker_ids.contains(&worker_id))
    }

    #[inline]
    pub fn pinned_worker(&self) -> Option<WorkerWithDpRank> {
        self.pinned_worker
    }

    #[inline]
    pub fn caller_allows_worker_id(&self, worker_id: WorkerId) -> bool {
        self.allowed_worker_ids
            .is_none_or(|worker_ids| worker_ids.contains(&worker_id))
    }

    #[inline]
    pub fn is_worker_overloaded(&self, worker_id: WorkerId) -> bool {
        self.overloaded_worker_ids
            .is_some_and(|worker_ids| worker_ids.contains(&worker_id))
    }

    #[inline]
    pub fn allows_worker_id(&self, worker_id: WorkerId) -> bool {
        self.caller_allows_worker_id(worker_id)
            && !self.is_worker_fenced(worker_id)
            && !self.is_worker_overloaded(worker_id)
    }

    #[inline]
    pub fn allows_worker_ignoring_overload<C: WorkerConfigLike>(
        &self,
        worker_id: WorkerId,
        config: &C,
    ) -> bool {
        self.caller_allows_worker_id(worker_id)
            && !self.is_worker_fenced(worker_id)
            && self
                .routing_constraints
                .is_compatible_with_worker_taints(config.taints())
    }

    #[inline]
    pub fn allows_worker<C: WorkerConfigLike>(&self, worker_id: WorkerId, config: &C) -> bool {
        self.allows_worker_id(worker_id)
            && self
                .routing_constraints
                .is_compatible_with_worker_taints(config.taints())
    }

    #[inline]
    pub fn has_eligible_worker<'w, C, I>(&self, workers: I) -> bool
    where
        C: WorkerConfigLike + 'w,
        I: IntoIterator<Item = (WorkerId, &'w C)>,
    {
        for (worker_id, config) in workers {
            if !self.allows_worker_id(worker_id) {
                continue;
            }

            if self.allows_worker(worker_id, config) {
                return true;
            }
        }

        false
    }

    #[inline]
    pub fn has_eligible_worker_ignoring_overload<'w, C, I>(&self, workers: I) -> bool
    where
        C: WorkerConfigLike + 'w,
        I: IntoIterator<Item = (WorkerId, &'w C)>,
    {
        for (worker_id, config) in workers {
            if self.allows_worker_ignoring_overload(worker_id, config) {
                return true;
            }
        }

        false
    }

    #[inline]
    pub fn validate_worker_rank<'w, C: WorkerConfigLike>(
        &self,
        workers: &'w HashMap<WorkerId, C>,
        worker: WorkerWithDpRank,
    ) -> Result<&'w C, WorkerEligibilityError> {
        if !self.caller_allows_worker_id(worker.worker_id) {
            return Err(WorkerEligibilityError::WorkerNotAllowed {
                worker_id: worker.worker_id,
            });
        }

        // A fenced worker is dead — reject before taint/overload checks so it is
        // never selected, even on a cache-hit affinity pin.
        if self.is_worker_fenced(worker.worker_id) {
            return Err(WorkerEligibilityError::WorkerFenced {
                worker_id: worker.worker_id,
            });
        }

        let config = worker_config_for_rank(workers, worker)?;
        if !self
            .routing_constraints
            .is_compatible_with_worker_taints(config.taints())
        {
            return Err(WorkerEligibilityError::RoutingConstraintsUnsatisfied {
                worker_id: worker.worker_id,
            });
        }

        if self.is_worker_overloaded(worker.worker_id) {
            return Err(WorkerEligibilityError::WorkerOverloaded {
                worker_id: worker.worker_id,
            });
        }

        Ok(config)
    }

    pub fn any_eligible_worker_rank<C, F>(
        &self,
        workers: &HashMap<WorkerId, C>,
        mut predicate: F,
    ) -> bool
    where
        C: WorkerConfigLike,
        F: FnMut(WorkerWithDpRank, &C) -> bool,
    {
        if let Some(worker) = self.pinned_worker {
            let Ok(config) = self.validate_worker_rank(workers, worker) else {
                return false;
            };
            return predicate(worker, config);
        }

        for (&worker_id, config) in workers {
            if !self.allows_worker(worker_id, config) {
                continue;
            }

            let dp_start = config.data_parallel_start_rank();
            let dp_end = dp_start + config.data_parallel_size();
            for dp_rank in dp_start..dp_end {
                if predicate(WorkerWithDpRank::new(worker_id, dp_rank), config) {
                    return true;
                }
            }
        }

        false
    }

    pub fn for_each_eligible_worker_rank<C, F>(&self, workers: &HashMap<WorkerId, C>, mut visit: F)
    where
        C: WorkerConfigLike,
        F: FnMut(WorkerWithDpRank, &C),
    {
        self.any_eligible_worker_rank(workers, |worker, config| {
            visit(worker, config);
            false
        });
    }

    #[inline]
    pub(crate) fn validate_pinned_worker_allowed(&self) -> Result<(), KvSchedulerError> {
        let Some(pinned_worker) = self.pinned_worker else {
            return Ok(());
        };

        if self.caller_allows_worker_id(pinned_worker.worker_id) {
            return Ok(());
        }

        Err(KvSchedulerError::PinnedWorkerNotAllowed {
            worker_id: pinned_worker.worker_id,
        })
    }
}

fn worker_config_for_rank<C: WorkerConfigLike>(
    workers: &HashMap<WorkerId, C>,
    worker: WorkerWithDpRank,
) -> Result<&C, WorkerEligibilityError> {
    let Some(config) = workers.get(&worker.worker_id) else {
        return Err(WorkerEligibilityError::WorkerUnavailable {
            worker_id: worker.worker_id,
        });
    };
    let dp_start_rank = config.data_parallel_start_rank();
    let dp_end_rank = dp_start_rank + config.data_parallel_size();
    if !(dp_start_rank..dp_end_rank).contains(&worker.dp_rank) {
        return Err(WorkerEligibilityError::DpRankUnavailable {
            worker_id: worker.worker_id,
            dp_rank: worker.dp_rank,
            start: dp_start_rank,
            end: dp_end_rank,
        });
    }

    Ok(config)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[derive(Clone)]
    struct TestWorkerConfig {
        dp_start: DpRank,
        dp_size: DpRank,
        taints: HashSet<String>,
    }

    impl Default for TestWorkerConfig {
        fn default() -> Self {
            Self {
                dp_start: 0,
                dp_size: 1,
                taints: HashSet::new(),
            }
        }
    }

    impl WorkerConfigLike for TestWorkerConfig {
        fn data_parallel_start_rank(&self) -> u32 {
            self.dp_start
        }

        fn data_parallel_size(&self) -> u32 {
            self.dp_size
        }

        fn max_num_batched_tokens(&self) -> Option<u64> {
            None
        }

        fn total_kv_blocks(&self) -> Option<u64> {
            None
        }

        fn taints(&self) -> &HashSet<String> {
            &self.taints
        }
    }

    fn workers() -> HashMap<WorkerId, TestWorkerConfig> {
        HashMap::from([(
            7,
            TestWorkerConfig {
                dp_start: 2,
                dp_size: 3,
                taints: HashSet::from(["zone-a".to_string()]),
            },
        )])
    }

    #[test]
    fn routing_eligibility_accepts_allowed_rank_matching_constraints() {
        let workers = workers();
        let allowed = HashSet::from([7]);
        let constraints = RoutingConstraints {
            required_taints: HashSet::from(["zone-a".to_string()]),
            preferred_taints: HashMap::new(),
        };
        let eligibility = RoutingEligibility::new(Some(&allowed), None, None, &constraints);

        let result = eligibility.validate_worker_rank(&workers, WorkerWithDpRank::new(7, 3));

        assert!(result.is_ok());
    }

    #[test]
    fn routing_eligibility_rejects_disallowed_worker() {
        let workers = workers();
        let allowed = HashSet::from([8]);
        let constraints = RoutingConstraints::default();
        let eligibility = RoutingEligibility::new(Some(&allowed), None, None, &constraints);

        let result = eligibility.validate_worker_rank(&workers, WorkerWithDpRank::new(7, 3));

        assert_eq!(
            result.err(),
            Some(WorkerEligibilityError::WorkerNotAllowed { worker_id: 7 })
        );
    }

    #[test]
    fn routing_eligibility_rejects_overloaded_worker() {
        let workers = workers();
        let overloaded = HashSet::from([7]);
        let constraints = RoutingConstraints::default();
        let eligibility = RoutingEligibility::new(None, Some(&overloaded), None, &constraints);

        let result = eligibility.validate_worker_rank(&workers, WorkerWithDpRank::new(7, 3));

        assert_eq!(
            result.err(),
            Some(WorkerEligibilityError::WorkerOverloaded { worker_id: 7 })
        );
    }

    // DIS-2404: a fenced (dead/deregistered) worker must be rejected on rank
    // validation even though it is allowed, in range, taint-compatible, and not
    // overloaded — availability overrides cache affinity.
    #[test]
    fn routing_eligibility_rejects_fenced_worker() {
        let workers = workers();
        let fenced = HashSet::from([7]);
        let constraints = RoutingConstraints::default();
        let eligibility = RoutingEligibility::new(None, None, None, &constraints)
            .with_fenced_workers(Some(&fenced));

        let result = eligibility.validate_worker_rank(&workers, WorkerWithDpRank::new(7, 3));

        assert_eq!(
            result.err(),
            Some(WorkerEligibilityError::WorkerFenced { worker_id: 7 })
        );
    }

    // DIS-2404: unlike transient overload (which the affinity/pinned pre-validation
    // path deliberately ignores), a fence must NOT be ignored — a dead worker is
    // never eligible, even for a cache-hit affinity pin.
    #[test]
    fn fenced_worker_not_ignored_by_affinity_path() {
        let workers = workers();
        let config = workers.get(&7).unwrap();
        let fenced = HashSet::from([7]);
        let constraints = RoutingConstraints::default();

        let unfenced = RoutingEligibility::new(None, None, None, &constraints);
        assert!(
            unfenced.allows_worker_ignoring_overload(7, config),
            "worker 7 is eligible when not fenced"
        );

        let eligibility = RoutingEligibility::new(None, None, None, &constraints)
            .with_fenced_workers(Some(&fenced));
        assert!(
            !eligibility.allows_worker_ignoring_overload(7, config),
            "fenced worker 7 must be rejected even on the affinity-ignoring-overload path"
        );
    }

    #[test]
    fn routing_eligibility_prefers_allow_list_error_before_overload() {
        let workers = workers();
        let allowed = HashSet::from([8]);
        let overloaded = HashSet::from([7]);
        let constraints = RoutingConstraints::default();
        let eligibility =
            RoutingEligibility::new(Some(&allowed), Some(&overloaded), None, &constraints);

        let result = eligibility.validate_worker_rank(&workers, WorkerWithDpRank::new(7, 3));

        assert_eq!(
            result.err(),
            Some(WorkerEligibilityError::WorkerNotAllowed { worker_id: 7 })
        );
    }

    #[test]
    fn routing_eligibility_rejects_out_of_range_dp_rank() {
        let workers = workers();
        let constraints = RoutingConstraints::default();
        let eligibility = RoutingEligibility::new(None, None, None, &constraints);

        let result = eligibility.validate_worker_rank(&workers, WorkerWithDpRank::new(7, 5));

        assert_eq!(
            result.err(),
            Some(WorkerEligibilityError::DpRankUnavailable {
                worker_id: 7,
                dp_rank: 5,
                start: 2,
                end: 5,
            })
        );
    }

    #[test]
    fn routing_eligibility_rejects_unsatisfied_required_taints() {
        let workers = workers();
        let constraints = RoutingConstraints {
            required_taints: HashSet::from(["zone-b".to_string()]),
            preferred_taints: HashMap::new(),
        };
        let eligibility = RoutingEligibility::new(None, None, None, &constraints);

        let result = eligibility.validate_worker_rank(&workers, WorkerWithDpRank::new(7, 3));

        assert_eq!(
            result.err(),
            Some(WorkerEligibilityError::RoutingConstraintsUnsatisfied { worker_id: 7 })
        );
    }

    #[test]
    fn routing_eligibility_applies_allowed_overloaded_and_taints() {
        let allowed_worker_ids = HashSet::from([1, 2]);
        let overloaded_worker_ids = HashSet::from([2]);
        let routing_constraints = RoutingConstraints {
            required_taints: HashSet::from(["mdc-a".to_string()]),
            preferred_taints: HashMap::new(),
        };
        let eligibility = RoutingEligibility::new(
            Some(&allowed_worker_ids),
            Some(&overloaded_worker_ids),
            None,
            &routing_constraints,
        );

        let compatible = TestWorkerConfig {
            taints: HashSet::from(["mdc-a".to_string()]),
            ..Default::default()
        };
        let incompatible = TestWorkerConfig {
            taints: HashSet::from(["mdc-b".to_string()]),
            ..Default::default()
        };

        assert!(eligibility.allows_worker(1, &compatible));
        assert!(!eligibility.allows_worker(2, &compatible));
        assert!(!eligibility.allows_worker(3, &compatible));
        assert!(!eligibility.allows_worker(1, &incompatible));
        assert!(eligibility.has_eligible_worker([(1, &compatible), (2, &compatible)]));
        assert!(!eligibility.has_eligible_worker([(2, &compatible)]));
        assert!(eligibility.has_eligible_worker_ignoring_overload([(2, &compatible)]));
        assert!(
            eligibility.has_eligible_worker_ignoring_overload([(1, &compatible), (2, &compatible)])
        );
    }

    #[test]
    fn routing_eligibility_expands_all_eligible_dp_ranks() {
        let workers = HashMap::from([
            (
                7,
                TestWorkerConfig {
                    dp_start: 2,
                    dp_size: 3,
                    taints: HashSet::from(["zone-a".to_string()]),
                },
            ),
            (
                8,
                TestWorkerConfig {
                    dp_start: 0,
                    dp_size: 2,
                    taints: HashSet::from(["zone-b".to_string()]),
                },
            ),
        ]);
        let allowed = HashSet::from([7]);
        let constraints = RoutingConstraints {
            required_taints: HashSet::from(["zone-a".to_string()]),
            preferred_taints: HashMap::new(),
        };
        let eligibility = RoutingEligibility::new(Some(&allowed), None, None, &constraints);
        let mut ranks = Vec::new();

        eligibility.for_each_eligible_worker_rank(&workers, |worker, _| ranks.push(worker));

        assert_eq!(
            ranks,
            vec![
                WorkerWithDpRank::new(7, 2),
                WorkerWithDpRank::new(7, 3),
                WorkerWithDpRank::new(7, 4),
            ]
        );
    }

    #[test]
    fn routing_eligibility_rank_expansion_skips_overloaded_workers() {
        let workers = HashMap::from([
            (
                7,
                TestWorkerConfig {
                    dp_start: 2,
                    dp_size: 2,
                    taints: HashSet::from(["zone-a".to_string()]),
                },
            ),
            (
                8,
                TestWorkerConfig {
                    dp_start: 4,
                    dp_size: 2,
                    taints: HashSet::from(["zone-a".to_string()]),
                },
            ),
        ]);
        let overloaded = HashSet::from([7]);
        let constraints = RoutingConstraints {
            required_taints: HashSet::from(["zone-a".to_string()]),
            preferred_taints: HashMap::new(),
        };
        let eligibility = RoutingEligibility::new(None, Some(&overloaded), None, &constraints);
        let mut ranks = Vec::new();

        eligibility.for_each_eligible_worker_rank(&workers, |worker, _| ranks.push(worker));
        ranks.sort_by_key(|worker| (worker.worker_id, worker.dp_rank));

        assert_eq!(
            ranks,
            vec![WorkerWithDpRank::new(8, 4), WorkerWithDpRank::new(8, 5)]
        );
    }

    #[test]
    fn routing_eligibility_pinned_expansion_yields_exact_rank() {
        let workers = workers();
        let constraints = RoutingConstraints::default();
        let eligibility =
            RoutingEligibility::new(None, None, Some(WorkerWithDpRank::new(7, 3)), &constraints);
        let mut ranks = Vec::new();

        eligibility.for_each_eligible_worker_rank(&workers, |worker, _| ranks.push(worker));

        assert_eq!(ranks, vec![WorkerWithDpRank::new(7, 3)]);
    }

    #[test]
    fn routing_eligibility_pinned_expansion_rejects_bad_rank() {
        let workers = workers();
        let constraints = RoutingConstraints::default();
        let eligibility =
            RoutingEligibility::new(None, None, Some(WorkerWithDpRank::new(7, 5)), &constraints);
        let mut ranks = Vec::new();

        eligibility.for_each_eligible_worker_rank(&workers, |worker, _| ranks.push(worker));

        assert!(ranks.is_empty());
    }
}
