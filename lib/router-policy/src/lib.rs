// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Dependency-neutral worker-picking policies.
//!
//! Hosts retain ownership of discovery, eligibility, load, and admission. This crate owns only
//! the allocation-free picking algorithms and their policy-local state.

use std::sync::atomic::{AtomicU64, Ordering};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RoutingPolicy {
    RoundRobin,
    Random,
    PowerOfTwoChoices,
    LeastLoaded,
    DeviceAwareWeighted,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum RouteDevice {
    Cpu,
    #[default]
    Accelerator,
}

#[derive(Clone, Copy, Debug)]
pub struct RouteContext {
    pub required_cache_hits: usize,
    pub non_cpu_to_cpu_ratio: usize,
}

impl Default for RouteContext {
    fn default() -> Self {
        Self {
            required_cache_hits: 0,
            non_cpu_to_cpu_ratio: 8,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AdmissionKind {
    None,
    Occupancy,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PolicyDecision {
    pub index: usize,
    pub admission: AdmissionKind,
}

/// Host-owned, borrowed candidate table consumed by a picker.
///
/// Static policies only call [`Self::len`]. Load-aware policies additionally call
/// [`Self::load`], and the device-aware policy reads the device and cache columns. A host can
/// therefore wrap its native discovery snapshot without constructing policy-owned rows.
pub trait RouteCandidates {
    fn len(&self) -> usize;

    fn is_empty(&self) -> bool {
        self.len() == 0
    }

    fn load(&self, index: usize) -> u64;

    fn device(&self, _index: usize) -> RouteDevice {
        RouteDevice::Accelerator
    }

    fn cache_hits(&self, _index: usize) -> usize {
        0
    }
}

#[derive(Debug)]
pub struct RoutePicker {
    policy: RoutingPolicy,
    round_robin_cursor: AtomicU64,
}

impl RoutePicker {
    pub const fn new(policy: RoutingPolicy) -> Self {
        Self {
            policy,
            round_robin_cursor: AtomicU64::new(0),
        }
    }

    pub const fn policy(&self) -> RoutingPolicy {
        self.policy
    }

    #[inline(always)]
    pub fn peek<C: RouteCandidates + ?Sized>(
        &self,
        candidates: &C,
        context: RouteContext,
    ) -> Option<PolicyDecision> {
        if self.policy == RoutingPolicy::Random {
            return random_decision(candidates);
        }
        let mut samples = RandomSamples;
        self.choose_with_samples(candidates, context, false, &mut samples)
    }

    #[inline(always)]
    pub fn select<C: RouteCandidates + ?Sized>(
        &self,
        candidates: &C,
        context: RouteContext,
    ) -> Option<PolicyDecision> {
        if self.policy == RoutingPolicy::Random {
            return random_decision(candidates);
        }
        let mut samples = RandomSamples;
        self.choose_with_samples(candidates, context, true, &mut samples)
    }

    #[inline(always)]
    fn choose_with_samples<C: RouteCandidates + ?Sized>(
        &self,
        candidates: &C,
        context: RouteContext,
        commit: bool,
        samples: &mut impl SampleSource,
    ) -> Option<PolicyDecision> {
        if candidates.is_empty() {
            return None;
        }

        let (index, admission) = match self.policy {
            RoutingPolicy::RoundRobin => {
                let cursor = if commit {
                    self.round_robin_cursor.fetch_add(1, Ordering::Relaxed)
                } else {
                    self.round_robin_cursor.load(Ordering::Relaxed)
                };
                (cursor as usize % candidates.len(), AdmissionKind::None)
            }
            RoutingPolicy::Random => (samples.index(candidates.len()), AdmissionKind::None),
            RoutingPolicy::PowerOfTwoChoices => {
                let first = samples.index(candidates.len());
                if candidates.len() == 1 {
                    return Some(PolicyDecision {
                        index: first,
                        admission: AdmissionKind::Occupancy,
                    });
                }
                let second_offset = 1 + samples.index(candidates.len() - 1);
                let second = (first + second_offset) % candidates.len();
                let index = if candidates.load(first) <= candidates.load(second) {
                    first
                } else {
                    second
                };
                (index, AdmissionKind::Occupancy)
            }
            RoutingPolicy::LeastLoaded => {
                (lowest_load(candidates, samples)?, AdmissionKind::Occupancy)
            }
            RoutingPolicy::DeviceAwareWeighted => {
                return device_aware(candidates, context, samples);
            }
        };
        Some(PolicyDecision { index, admission })
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
fn random_decision<C: RouteCandidates + ?Sized>(candidates: &C) -> Option<PolicyDecision> {
    (!candidates.is_empty()).then(|| PolicyDecision {
        index: fastrand::usize(..candidates.len()),
        admission: AdmissionKind::None,
    })
}

#[inline(always)]
fn lowest_load<C: RouteCandidates + ?Sized>(
    candidates: &C,
    samples: &mut impl SampleSource,
) -> Option<usize> {
    let mut best = None;
    let mut best_load = u64::MAX;
    let mut ties = 0usize;
    for index in 0..candidates.len() {
        let load = candidates.load(index);
        if load < best_load {
            best = Some(index);
            best_load = load;
            ties = 1;
        } else if load == best_load {
            ties += 1;
            if samples.index(ties) == 0 {
                best = Some(index);
            }
        }
    }
    best
}

#[derive(Default)]
struct DeviceGroup {
    count: u64,
    total_load: u64,
    best: Option<usize>,
    best_load: u64,
    best_ties: usize,
}

impl DeviceGroup {
    #[inline(always)]
    fn consider(&mut self, index: usize, load: u64, samples: &mut impl SampleSource) {
        self.count += 1;
        self.total_load = self.total_load.saturating_add(load);
        if self.best.is_none() || load < self.best_load {
            self.best = Some(index);
            self.best_load = load;
            self.best_ties = 1;
        } else if load == self.best_load {
            self.best_ties += 1;
            if samples.index(self.best_ties) == 0 {
                self.best = Some(index);
            }
        }
    }
}

#[inline(always)]
fn device_aware<C: RouteCandidates + ?Sized>(
    candidates: &C,
    context: RouteContext,
    samples: &mut impl SampleSource,
) -> Option<PolicyDecision> {
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

    for index in 0..candidates.len() {
        let load = candidates.load(index);
        match candidates.device(index) {
            RouteDevice::Cpu => cpu.consider(index, load, samples),
            RouteDevice::Accelerator => accelerator.consider(index, load, samples),
        }
        if context.required_cache_hits > 0
            && candidates.cache_hits(index) >= context.required_cache_hits
        {
            full_cache.consider(index, load, samples);
        }
    }

    if let Some(index) = full_cache.best {
        return Some(PolicyDecision {
            index,
            admission: AdmissionKind::None,
        });
    }

    let index = match (cpu.best, accelerator.best) {
        (None, None) => return None,
        (Some(index), None) | (None, Some(index)) => index,
        (Some(cpu_index), Some(accelerator_index)) => {
            let ratio = context.non_cpu_to_cpu_ratio.max(1) as u64;
            let allowed_cpu = accelerator.total_load.saturating_mul(cpu.count)
                / ratio.saturating_mul(accelerator.count);
            if cpu.total_load < allowed_cpu {
                cpu_index
            } else {
                accelerator_index
            }
        }
    };
    Some(PolicyDecision {
        index,
        admission: AdmissionKind::Occupancy,
    })
}

#[cfg(test)]
mod tests {
    use std::cell::Cell;

    use super::*;

    struct Rows<'a> {
        loads: &'a [u64],
    }

    impl RouteCandidates for Rows<'_> {
        fn len(&self) -> usize {
            self.loads.len()
        }

        fn load(&self, index: usize) -> u64 {
            self.loads[index]
        }
    }

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
    fn round_robin_peek_does_not_advance_committed_cursor() {
        let picker = RoutePicker::new(RoutingPolicy::RoundRobin);
        let rows = Rows { loads: &[0, 0] };
        assert_eq!(
            picker.peek(&rows, RouteContext::default()).unwrap().index,
            0
        );
        assert_eq!(
            picker.peek(&rows, RouteContext::default()).unwrap().index,
            0
        );
        assert_eq!(
            picker.select(&rows, RouteContext::default()).unwrap().index,
            0
        );
        assert_eq!(
            picker.select(&rows, RouteContext::default()).unwrap().index,
            1
        );
    }

    #[test]
    fn least_loaded_uses_host_loads() {
        let picker = RoutePicker::new(RoutingPolicy::LeastLoaded);
        let rows = Rows { loads: &[7, 2, 9] };
        let decision = picker.select(&rows, RouteContext::default()).unwrap();
        assert_eq!(decision.index, 1);
        assert_eq!(decision.admission, AdmissionKind::Occupancy);
    }

    #[test]
    fn p2c_uses_two_samples_and_two_load_reads() {
        struct CountingRows<'a> {
            loads: &'a [u64],
            reads: Cell<usize>,
        }

        impl RouteCandidates for CountingRows<'_> {
            fn len(&self) -> usize {
                self.loads.len()
            }

            fn load(&self, index: usize) -> u64 {
                self.reads.set(self.reads.get() + 1);
                self.loads[index]
            }
        }

        let picker = RoutePicker::new(RoutingPolicy::PowerOfTwoChoices);
        let rows = CountingRows {
            loads: &[0, 5, 0, 1],
            reads: Cell::new(0),
        };
        let mut samples = ScriptedSamples::new(vec![1, 1]);
        let decision = picker
            .choose_with_samples(&rows, RouteContext::default(), true, &mut samples)
            .unwrap();
        assert_eq!(decision.index, 3);
        assert_eq!(samples.consumed, 2);
        assert_eq!(rows.reads.get(), 2);
    }

    #[test]
    fn full_device_cache_hit_skips_occupancy_admission() {
        struct DeviceRows {
            loads: [u64; 3],
            devices: [RouteDevice; 3],
            cache_hits: [usize; 3],
        }

        impl RouteCandidates for DeviceRows {
            fn len(&self) -> usize {
                self.loads.len()
            }

            fn load(&self, index: usize) -> u64 {
                self.loads[index]
            }

            fn device(&self, index: usize) -> RouteDevice {
                self.devices[index]
            }

            fn cache_hits(&self, index: usize) -> usize {
                self.cache_hits[index]
            }
        }

        let picker = RoutePicker::new(RoutingPolicy::DeviceAwareWeighted);
        let rows = DeviceRows {
            loads: [0, 20, 1],
            devices: [
                RouteDevice::Cpu,
                RouteDevice::Accelerator,
                RouteDevice::Accelerator,
            ],
            cache_hits: [0, 2, 1],
        };
        let decision = picker
            .peek(
                &rows,
                RouteContext {
                    required_cache_hits: 2,
                    non_cpu_to_cpu_ratio: 8,
                },
            )
            .unwrap();
        assert_eq!(decision.index, 1);
        assert_eq!(decision.admission, AdmissionKind::None);
    }
}
