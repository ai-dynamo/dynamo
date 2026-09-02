// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::sync::atomic::{AtomicU64, Ordering};

use super::{
    AdmissionKind, CandidateView, RouteCandidate, RouteContext, RouteDecision, RouteDevice,
    RouteLoad, RoutePolicy, RouteTarget,
};

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
        load: impl Fn(u64) -> RouteLoad,
    ) -> Option<RouteDecision> {
        if self.policy == RoutePolicy::Random {
            return random_decision(candidates);
        }
        let mut samples = RandomSamples;
        self.choose_with_samples(candidates, context, &load, false, &mut samples)
    }

    #[inline(always)]
    pub(crate) fn select(
        &self,
        candidates: CandidateView<'_>,
        context: RouteContext,
        load: impl Fn(u64) -> RouteLoad,
    ) -> Option<RouteDecision> {
        if self.policy == RoutePolicy::Random {
            return random_decision(candidates);
        }
        let mut samples = RandomSamples;
        self.choose_with_samples(candidates, context, &load, true, &mut samples)
    }

    #[inline(always)]
    fn choose_with_samples(
        &self,
        candidates: CandidateView<'_>,
        context: RouteContext,
        load: &impl Fn(u64) -> RouteLoad,
        commit: bool,
        samples: &mut impl SampleSource,
    ) -> Option<RouteDecision> {
        if candidates.is_empty() {
            return None;
        }

        match self.policy {
            RoutePolicy::RoundRobin => {
                let cursor = if commit {
                    self.round_robin_cursor.fetch_add(1, Ordering::Relaxed)
                } else {
                    self.round_robin_cursor.load(Ordering::Relaxed)
                };
                Some(RouteDecision {
                    target: candidates.target(cursor as usize % candidates.len()),
                    admission: AdmissionKind::None,
                })
            }
            RoutePolicy::Random => Some(RouteDecision {
                target: candidates.target(samples.index(candidates.len())),
                admission: AdmissionKind::None,
            }),
            RoutePolicy::PowerOfTwoChoices => {
                let first = samples.index(candidates.len());
                if candidates.len() == 1 {
                    return Some(RouteDecision {
                        target: candidates.target(first),
                        admission: AdmissionKind::Occupancy,
                    });
                }
                let second_offset = 1 + samples.index(candidates.len() - 1);
                let second = (first + second_offset) % candidates.len();
                let first_target = candidates.target(first);
                let second_target = candidates.target(second);
                let target = if load(first_target.worker_id).requests
                    <= load(second_target.worker_id).requests
                {
                    first_target
                } else {
                    second_target
                };
                Some(RouteDecision {
                    target,
                    admission: AdmissionKind::Occupancy,
                })
            }
            RoutePolicy::LeastLoaded => {
                lowest_load(candidates, load, samples).map(|target| RouteDecision {
                    target,
                    admission: AdmissionKind::Occupancy,
                })
            }
            RoutePolicy::DeviceAwareWeighted => {
                let CandidateView::DeviceAware(candidates) = candidates else {
                    return None;
                };
                device_aware(candidates, context, load, samples)
            }
        }
    }
}

trait SampleSource {
    fn index(&mut self, upper: usize) -> usize;
}

struct RandomSamples;

impl SampleSource for RandomSamples {
    #[inline(always)]
    fn index(&mut self, upper: usize) -> usize {
        fastrand::usize(..upper)
    }
}

#[inline(always)]
fn random_decision(candidates: CandidateView<'_>) -> Option<RouteDecision> {
    let upper = candidates.len();
    if upper == 0 {
        return None;
    }
    let index = fastrand::usize(..upper);
    let target = match candidates {
        CandidateView::Workers(workers) => RouteTarget::worker(workers[index]),
        CandidateView::DeviceAware(candidates) => candidates[index].target,
    };
    Some(RouteDecision {
        target,
        admission: AdmissionKind::None,
    })
}

#[inline(always)]
fn lowest_load(
    candidates: CandidateView<'_>,
    load: &impl Fn(u64) -> RouteLoad,
    samples: &mut impl SampleSource,
) -> Option<RouteTarget> {
    let mut best: Option<(RouteTarget, RouteLoad)> = None;
    let mut ties = 0usize;
    for index in 0..candidates.len() {
        let target = candidates.target(index);
        let target_load = load(target.worker_id);
        match best {
            Some((_, best_load)) if target_load > best_load => {}
            Some((_, best_load)) if target_load == best_load => {
                ties += 1;
                if samples.index(ties) == 0 {
                    best = Some((target, target_load));
                }
            }
            _ => {
                best = Some((target, target_load));
                ties = 1;
            }
        }
    }
    best.map(|(target, _)| target)
}

#[derive(Default)]
struct DeviceGroup {
    count: u64,
    total_load: u64,
    best: Option<RouteTarget>,
    best_load: u64,
    best_ties: usize,
}

impl DeviceGroup {
    #[inline(always)]
    fn consider(&mut self, target: RouteTarget, target_load: u64, samples: &mut impl SampleSource) {
        self.count += 1;
        self.total_load = self.total_load.saturating_add(target_load);
        if self.best.is_none() || target_load < self.best_load {
            self.best = Some(target);
            self.best_load = target_load;
            self.best_ties = 1;
        } else if target_load == self.best_load {
            self.best_ties += 1;
            if samples.index(self.best_ties) == 0 {
                self.best = Some(target);
            }
        }
    }
}

#[inline(always)]
fn device_aware(
    candidates: &[RouteCandidate],
    context: RouteContext,
    load: &impl Fn(u64) -> RouteLoad,
    samples: &mut impl SampleSource,
) -> Option<RouteDecision> {
    let mut cpu = DeviceGroup {
        best_load: u64::MAX,
        ..Default::default()
    };
    let mut accelerator = DeviceGroup {
        best_load: u64::MAX,
        ..Default::default()
    };
    let mut full_cache = DeviceGroup {
        best_load: u64::MAX,
        ..Default::default()
    };

    for candidate in candidates {
        // Device-aware balancing budgets CPU work against accelerator work by
        // request count, so it reads the request-count component only.
        let target_load = load(candidate.target.worker_id).requests;
        match candidate.device {
            RouteDevice::Cpu => cpu.consider(candidate.target, target_load, samples),
            RouteDevice::Accelerator => {
                accelerator.consider(candidate.target, target_load, samples)
            }
        }
        if context.required_cache_hits > 0 && candidate.cache_hits >= context.required_cache_hits {
            full_cache.consider(candidate.target, target_load, samples);
        }
    }

    if let Some(target) = full_cache.best {
        return Some(RouteDecision {
            target,
            admission: AdmissionKind::None,
        });
    }

    let target = match (cpu.best, accelerator.best) {
        (None, None) => return None,
        (Some(target), None) => target,
        (None, Some(target)) => target,
        (Some(cpu_target), Some(accelerator_target)) => {
            let ratio = context.non_cpu_to_cpu_ratio.max(1) as u64;
            let allowed_cpu = accelerator.total_load.saturating_mul(cpu.count)
                / ratio.saturating_mul(accelerator.count);
            if cpu.total_load < allowed_cpu {
                cpu_target
            } else {
                accelerator_target
            }
        }
    };

    Some(RouteDecision {
        target,
        admission: AdmissionKind::Occupancy,
    })
}

#[cfg(test)]
mod tests {
    use std::cell::Cell;

    use super::*;

    struct ScriptedSamples {
        values: Vec<usize>,
        consumed: usize,
    }

    impl ScriptedSamples {
        fn new(values: impl Into<Vec<usize>>) -> Self {
            Self {
                values: values.into(),
                consumed: 0,
            }
        }
    }

    impl SampleSource for ScriptedSamples {
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
                    .peek(candidates, RouteContext::default(), |_| RouteLoad::default(
                    ))
                    .unwrap()
                    .target
                    .worker_id,
                10
            );
        }
        let selected = (0..4)
            .map(|_| {
                picker
                    .select(candidates, RouteContext::default(), |_| {
                        RouteLoad::default()
                    })
                    .unwrap()
                    .target
                    .worker_id
            })
            .collect::<Vec<_>>();
        assert_eq!(selected, [10, 20, 30, 40]);
    }

    #[test]
    fn random_peek_uses_one_independent_sample() {
        let picker = RoutePicker::new(RoutePolicy::Random);
        let candidates = CandidateView::Workers(&[10, 20, 30, 40]);
        let mut samples = ScriptedSamples::new(vec![2]);
        let decision = picker
            .choose_with_samples(
                candidates,
                RouteContext::default(),
                &|_| RouteLoad::default(),
                false,
                &mut samples,
            )
            .unwrap();
        assert_eq!(decision.target.worker_id, 30);
        assert_eq!(samples.consumed, 1);
    }

    #[test]
    fn p2c_uses_exactly_two_samples_and_two_load_reads() {
        let picker = RoutePicker::new(RoutePolicy::PowerOfTwoChoices);
        let candidates = CandidateView::Workers(&[10, 20, 30, 40]);
        let mut samples = ScriptedSamples::new(vec![1, 1]);
        let reads = Cell::new(0);
        let decision = picker
            .choose_with_samples(
                candidates,
                RouteContext::default(),
                &|worker| {
                    reads.set(reads.get() + 1);
                    RouteLoad::requests(if worker == 20 { 5 } else { 1 })
                },
                true,
                &mut samples,
            )
            .unwrap();
        assert_eq!(decision.target.worker_id, 40);
        assert_eq!(samples.consumed, 2);
        assert_eq!(reads.get(), 2);
    }

    #[test]
    fn least_loaded_ties_follow_scripted_reservoir_samples() {
        let picker = RoutePicker::new(RoutePolicy::LeastLoaded);
        let candidates = CandidateView::Workers(&[10, 20, 30]);
        let mut samples = ScriptedSamples::new(vec![1, 0]);
        let decision = picker
            .choose_with_samples(
                candidates,
                RouteContext::default(),
                &|_| RouteLoad::requests(3),
                true,
                &mut samples,
            )
            .unwrap();
        assert_eq!(decision.target.worker_id, 30);
        assert_eq!(samples.consumed, 2);
    }

    #[test]
    fn least_loaded_orders_by_queued_tokens_then_request_count() {
        let picker = RoutePicker::new(RoutePolicy::LeastLoaded);
        let candidates = CandidateView::Workers(&[10, 20, 30]);
        let mut samples = ScriptedSamples::new(vec![0, 0, 0]);
        let decision = picker
            .choose_with_samples(
                candidates,
                RouteContext::default(),
                &|worker| match worker {
                    // Fewest requests, but by far the most queued prompt tokens.
                    10 => RouteLoad {
                        queued_tokens: 4096,
                        requests: 1,
                    },
                    20 => RouteLoad {
                        queued_tokens: 64,
                        requests: 3,
                    },
                    _ => RouteLoad {
                        queued_tokens: 64,
                        requests: 9,
                    },
                },
                true,
                &mut samples,
            )
            .unwrap();
        assert_eq!(
            decision.target.worker_id, 20,
            "queued tokens decide, and the request count only breaks the 20/30 tie"
        );
    }

    #[test]
    fn p2c_ignores_queued_tokens() {
        let picker = RoutePicker::new(RoutePolicy::PowerOfTwoChoices);
        let candidates = CandidateView::Workers(&[10, 20]);
        let mut samples = ScriptedSamples::new(vec![0, 0]);
        let decision = picker
            .choose_with_samples(
                candidates,
                RouteContext::default(),
                &|worker| {
                    if worker == 10 {
                        RouteLoad {
                            queued_tokens: 4096,
                            requests: 1,
                        }
                    } else {
                        RouteLoad {
                            queued_tokens: 0,
                            requests: 2,
                        }
                    }
                },
                true,
                &mut samples,
            )
            .unwrap();
        assert_eq!(
            decision.target.worker_id, 10,
            "power-of-two-choices stays ordered by in-flight request count"
        );
    }

    #[test]
    fn device_aware_scans_each_candidate_once_and_full_hits_skip_admission() {
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
        let reads = Cell::new(0);
        let decision = picker
            .peek(
                CandidateView::DeviceAware(&candidates),
                RouteContext {
                    required_cache_hits: 2,
                    non_cpu_to_cpu_ratio: 8,
                },
                |worker| {
                    reads.set(reads.get() + 1);
                    RouteLoad::requests(worker)
                },
            )
            .unwrap();
        assert_eq!(decision.target.worker_id, 20);
        assert_eq!(decision.admission, AdmissionKind::None);
        assert_eq!(reads.get(), candidates.len());
    }
}
