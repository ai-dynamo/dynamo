// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
/// Worker-level routing target with an optional data-parallel rank.
pub struct RouteTarget {
    /// Runtime worker instance ID.
    pub worker_id: u64,
    /// Data-parallel rank when routing is rank-specific.
    pub dp_rank: Option<u32>,
}

impl RouteTarget {
    /// Construct a worker-level target without a data-parallel rank.
    pub const fn worker(worker_id: u64) -> Self {
        Self {
            worker_id,
            dp_rank: None,
        }
    }

    /// Construct a target with an optional data-parallel rank.
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
/// Device family consumed by device-aware worker selection.
pub enum RouteDevice {
    /// CPU worker.
    Cpu,
    #[default]
    /// Accelerator worker, including unknown device metadata for compatibility.
    Accelerator,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
/// One candidate for device-aware routing.
pub struct RouteCandidate {
    pub(crate) target: RouteTarget,
    pub(crate) device: RouteDevice,
    pub(crate) cache_hits: usize,
}

impl RouteCandidate {
    /// Construct one device-aware routing candidate.
    pub const fn new(target: RouteTarget, device: RouteDevice, cache_hits: usize) -> Self {
        Self {
            target,
            device,
            cache_hits,
        }
    }

    /// Return the candidate target.
    pub const fn target(self) -> RouteTarget {
        self.target
    }

    /// Return the candidate device class.
    pub const fn device(self) -> RouteDevice {
        self.device
    }

    /// Return request-specific multimodal cache hits for this candidate.
    pub const fn cache_hits(self) -> usize {
        self.cache_hits
    }

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
/// Request-level context for device-aware routing.
pub struct RouteContext {
    pub(crate) required_cache_hits: usize,
    pub(crate) non_cpu_to_cpu_ratio: usize,
}

impl RouteContext {
    /// Construct device-aware request context.
    pub const fn new(required_cache_hits: usize, non_cpu_to_cpu_ratio: usize) -> Self {
        Self {
            required_cache_hits,
            non_cpu_to_cpu_ratio,
        }
    }

    /// Return the hit count required for a complete request-cache hit.
    pub const fn required_cache_hits(self) -> usize {
        self.required_cache_hits
    }

    /// Return the accelerator-to-CPU weighting ratio.
    pub const fn non_cpu_to_cpu_ratio(self) -> usize {
        self.non_cpu_to_cpu_ratio
    }
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
pub(crate) enum AdmissionKind {
    None,
    Occupancy,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct RouteDecision {
    pub(crate) target: RouteTarget,
    pub(crate) admission: AdmissionKind,
}
