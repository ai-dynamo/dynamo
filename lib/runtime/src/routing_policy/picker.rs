// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::sync::atomic::{AtomicU64, Ordering};

use dynamo_routing_algorithms::{self as algorithms, DeviceAwareContext, Samples, ThreadRandom};

use super::{AdmissionKind, CandidateView, RouteContext, RouteDecision, RoutePolicy};

/// Dispatches a routing mode to its algorithm in `dynamo-routing-algorithms` and decides
/// whether the pick is admitted against occupancy. It owns the only per-router state, the
/// round-robin cursor.
#[derive(Debug)]
pub(crate) struct RoutePicker {
    policy: RoutePolicy,
    round_robin_cursor: AtomicU64,
}

impl RoutePicker {
    pub(crate) const fn new(policy: RoutePolicy) -> Self {
        Self {
            policy,
            round_robin_cursor: AtomicU64::new(0),
        }
    }

    pub(crate) const fn policy(&self) -> RoutePolicy {
        self.policy
    }

    #[inline(always)]
    pub(crate) fn peek(
        &self,
        candidates: CandidateView<'_>,
        context: RouteContext,
        load: impl Fn(u64) -> u64,
    ) -> Option<RouteDecision> {
        self.choose_with_samples(candidates, context, &load, false, &mut ThreadRandom)
    }

    #[inline(always)]
    pub(crate) fn select(
        &self,
        candidates: CandidateView<'_>,
        context: RouteContext,
        load: impl Fn(u64) -> u64,
    ) -> Option<RouteDecision> {
        self.choose_with_samples(candidates, context, &load, true, &mut ThreadRandom)
    }

    #[inline(always)]
    fn choose_with_samples(
        &self,
        candidates: CandidateView<'_>,
        context: RouteContext,
        load: &impl Fn(u64) -> u64,
        commit: bool,
        samples: &mut impl Samples,
    ) -> Option<RouteDecision> {
        let len = candidates.len();
        if len == 0 {
            return None;
        }
        let load_at = |index: usize| load(candidates.target(index).worker_id);

        let (index, admission) = match self.policy {
            RoutePolicy::RoundRobin => {
                let cursor = if commit {
                    self.round_robin_cursor.fetch_add(1, Ordering::Relaxed)
                } else {
                    self.round_robin_cursor.load(Ordering::Relaxed)
                };
                (algorithms::round_robin(cursor, len)?, AdmissionKind::None)
            }
            RoutePolicy::Random => (algorithms::random(len, samples)?, AdmissionKind::None),
            RoutePolicy::PowerOfTwoChoices => (
                algorithms::power_of_two(len, load_at, samples)?,
                AdmissionKind::Occupancy,
            ),
            RoutePolicy::LeastLoaded => (
                algorithms::least_loaded(len, load_at, samples)?,
                AdmissionKind::Occupancy,
            ),
            RoutePolicy::DeviceAwareWeighted => {
                let CandidateView::DeviceAware(device_candidates) = candidates else {
                    return None;
                };
                let choice = algorithms::device_aware(
                    len,
                    |index| device_candidates[index].device,
                    |index| device_candidates[index].cache_hits,
                    load_at,
                    DeviceAwareContext {
                        required_cache_hits: context.required_cache_hits,
                        non_cpu_to_cpu_ratio: context.non_cpu_to_cpu_ratio,
                    },
                    samples,
                )?;
                // A worker that already holds every embedding the request needs does no
                // encode work for it, so the pick is not counted as load.
                let admission = if choice.full_cache_hit {
                    AdmissionKind::None
                } else {
                    AdmissionKind::Occupancy
                };
                (choice.index, admission)
            }
        };
        Some(RouteDecision {
            target: candidates.target(index),
            admission,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::super::{RouteCandidate, RouteDevice, RouteTarget};
    use super::*;

    struct ScriptedSamples {
        values: Vec<usize>,
        consumed: usize,
    }

    impl Samples for ScriptedSamples {
        fn index(&mut self, upper: usize) -> usize {
            let value = self.values[self.consumed];
            self.consumed += 1;
            value % upper
        }
    }

    #[test]
    fn round_robin_peek_does_not_advance_selection() {
        let picker = RoutePicker::new(RoutePolicy::RoundRobin);
        let candidates = CandidateView::Workers(&[10, 20, 30, 40]);
        for _ in 0..16 {
            assert_eq!(
                picker
                    .peek(candidates, RouteContext::default(), |_| 0)
                    .unwrap()
                    .target
                    .worker_id,
                10
            );
        }
        let selected = (0..4)
            .map(|_| {
                picker
                    .select(candidates, RouteContext::default(), |_| 0)
                    .unwrap()
                    .target
                    .worker_id
            })
            .collect::<Vec<_>>();
        assert_eq!(selected, [10, 20, 30, 40]);
    }

    #[test]
    fn load_based_modes_admit_and_read_loads_by_worker_id() {
        let candidates = CandidateView::Workers(&[10, 20, 30, 40]);
        let load = |worker_id: u64| if worker_id == 20 { 5 } else { 1 };
        let picker = RoutePicker::new(RoutePolicy::PowerOfTwoChoices);
        let mut samples = ScriptedSamples {
            values: vec![1, 1],
            consumed: 0,
        };
        let decision = picker
            .choose_with_samples(
                candidates,
                RouteContext::default(),
                &load,
                true,
                &mut samples,
            )
            .unwrap();
        assert_eq!(decision.target.worker_id, 40);
        assert_eq!(decision.admission, AdmissionKind::Occupancy);

        let picker = RoutePicker::new(RoutePolicy::Random);
        let decision = picker
            .peek(candidates, RouteContext::default(), load)
            .unwrap();
        assert_eq!(decision.admission, AdmissionKind::None);
    }

    #[test]
    fn device_aware_full_cache_hits_skip_admission() {
        let picker = RoutePicker::new(RoutePolicy::DeviceAwareWeighted);
        let candidates = [
            RouteCandidate {
                target: RouteTarget::worker(10),
                device: RouteDevice::Cpu,
                cache_hits: 0,
            },
            RouteCandidate {
                target: RouteTarget::worker(20),
                device: RouteDevice::Accelerator,
                cache_hits: 2,
            },
            RouteCandidate {
                target: RouteTarget::worker(30),
                device: RouteDevice::Accelerator,
                cache_hits: 1,
            },
        ];
        let decision = picker
            .peek(
                CandidateView::DeviceAware(&candidates),
                RouteContext {
                    required_cache_hits: 2,
                    non_cpu_to_cpu_ratio: 8,
                },
                |worker| worker,
            )
            .unwrap();
        assert_eq!(decision.target.worker_id, 20);
        assert_eq!(decision.admission, AdmissionKind::None);
    }
}
