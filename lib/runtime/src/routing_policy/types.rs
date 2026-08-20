// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct RouteTarget {
    pub worker_id: u64,
    pub dp_rank: Option<u32>,
}

impl RouteTarget {
    pub const fn worker(worker_id: u64) -> Self {
        Self {
            worker_id,
            dp_rank: None,
        }
    }

    pub const fn new(worker_id: u64, dp_rank: Option<u32>) -> Self {
        Self { worker_id, dp_rank }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum RoutePolicy {
    RoundRobin,
    Random,
    PowerOfTwoChoices,
    LeastLoaded,
    DeviceAwareWeighted,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) enum RouteDevice {
    Cpu,
    #[default]
    Accelerator,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct RouteCandidate {
    pub(crate) target: RouteTarget,
    pub(crate) device: RouteDevice,
    pub(crate) cache_hits: usize,
}

impl RouteCandidate {
    pub(crate) const fn worker(worker_id: u64) -> Self {
        Self {
            target: RouteTarget::worker(worker_id),
            device: RouteDevice::Accelerator,
            cache_hits: 0,
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub(crate) enum CandidateView<'a> {
    Workers(&'a [u64]),
    DeviceAware(&'a [RouteCandidate]),
}

impl CandidateView<'_> {
    #[inline(always)]
    pub(super) fn len(&self) -> usize {
        match self {
            Self::Workers(workers) => workers.len(),
            Self::DeviceAware(candidates) => candidates.len(),
        }
    }

    #[inline(always)]
    pub(super) fn is_empty(&self) -> bool {
        self.len() == 0
    }

    #[inline(always)]
    pub(super) fn target(&self, index: usize) -> RouteTarget {
        match self {
            Self::Workers(workers) => RouteTarget::worker(workers[index]),
            Self::DeviceAware(candidates) => candidates[index].target,
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub(crate) struct RouteContext {
    pub(crate) required_cache_hits: usize,
    pub(crate) non_cpu_to_cpu_ratio: usize,
}

impl Default for RouteContext {
    fn default() -> Self {
        Self {
            required_cache_hits: 0,
            non_cpu_to_cpu_ratio: DEFAULT_NON_CPU_TO_CPU_RATIO,
        }
    }
}

pub(crate) const DEFAULT_NON_CPU_TO_CPU_RATIO: usize = 8;

/// Resolves configured, environment, and default ratios in precedence order.
pub(crate) fn resolve_non_cpu_to_cpu_ratio(configured: Option<usize>, env: Option<&str>) -> usize {
    configured
        .filter(|value| *value >= 1)
        .or_else(|| {
            env.and_then(|value| value.parse::<usize>().ok())
                .filter(|value| *value >= 1)
        })
        .unwrap_or(DEFAULT_NON_CPU_TO_CPU_RATIO)
}

pub(crate) fn non_cpu_to_cpu_ratio(configured: Option<usize>) -> usize {
    use crate::config::environment_names::router::DYN_ENCODER_CUDA_TO_CPU_RATIO;
    let env = std::env::var(DYN_ENCODER_CUDA_TO_CPU_RATIO).ok();
    resolve_non_cpu_to_cpu_ratio(configured, env.as_deref())
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum AdmissionKind {
    None,
    Occupancy,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct RouteDecision {
    pub(crate) target: RouteTarget,
    pub(crate) admission: AdmissionKind,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn configured_ratio_wins_over_environment() {
        assert_eq!(resolve_non_cpu_to_cpu_ratio(Some(3), Some("16")), 3);
    }

    #[test]
    fn environment_applies_when_nothing_is_configured() {
        assert_eq!(resolve_non_cpu_to_cpu_ratio(None, Some("16")), 16);
    }

    #[test]
    fn default_applies_when_neither_source_gives_a_ratio() {
        assert_eq!(
            resolve_non_cpu_to_cpu_ratio(None, None),
            DEFAULT_NON_CPU_TO_CPU_RATIO
        );
    }

    #[test]
    fn unusable_environment_values_fall_back_to_the_default() {
        for value in ["", "0", "-1", "eight", "8.5"] {
            assert_eq!(
                resolve_non_cpu_to_cpu_ratio(None, Some(value)),
                DEFAULT_NON_CPU_TO_CPU_RATIO,
                "value {value:?}"
            );
        }
    }

    #[test]
    fn a_configured_zero_falls_through_to_the_environment() {
        assert_eq!(resolve_non_cpu_to_cpu_ratio(Some(0), Some("16")), 16);
        assert_eq!(
            resolve_non_cpu_to_cpu_ratio(Some(0), None),
            DEFAULT_NON_CPU_TO_CPU_RATIO
        );
    }

    #[test]
    fn the_route_context_default_uses_the_default_ratio() {
        assert_eq!(
            RouteContext::default().non_cpu_to_cpu_ratio,
            DEFAULT_NON_CPU_TO_CPU_RATIO
        );
    }
}
