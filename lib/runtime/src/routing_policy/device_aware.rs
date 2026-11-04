// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Device-aware weighted selection for heterogeneous workers. It stays in the runtime until it
//! becomes a worker-selection plugin.

use super::{AdmissionKind, RouteCandidate, RouteContext, RouteDecision, RouteDevice, RouteTarget};

/// Pick a worker from `candidates`, reading each one's load once.
///
/// A worker holding every embedding the request needs wins outright, least loaded first,
/// and books no occupancy. Otherwise it keeps the least-loaded CPU worker and the
/// least-loaded accelerator worker, and sends work to the CPU worker while the CPU group's
/// total load stays under its share, weighted by `non_cpu_to_cpu_ratio`.
pub(crate) fn select(
    candidates: &[RouteCandidate],
    context: RouteContext,
    load: impl Fn(u64) -> u64,
) -> Option<RouteDecision> {
    device_aware(candidates, context, &load, &mut RandomSamples)
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
    load: &dyn Fn(u64) -> u64,
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
        let target_load = load(candidate.target.worker_id);
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

    #[test]
    fn device_aware_scans_each_candidate_once_and_full_hits_skip_admission() {
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
        let decision = select(
            &candidates,
            RouteContext {
                required_cache_hits: 2,
                non_cpu_to_cpu_ratio: 8,
            },
            |worker| {
                reads.set(reads.get() + 1);
                worker
            },
        )
        .unwrap();
        assert_eq!(decision.target.worker_id, 20);
        assert_eq!(decision.admission, AdmissionKind::None);
        assert_eq!(reads.get(), candidates.len());
    }
}
