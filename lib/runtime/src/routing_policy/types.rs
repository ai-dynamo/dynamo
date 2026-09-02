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

/// Per-worker load used to order routing candidates.
///
/// Field order is the comparison order: derived `Ord` compares `queued_tokens`
/// first and uses `requests` only to break ties. When token-aware occupancy is
/// disabled `queued_tokens` is always zero, so the ordering degenerates to the
/// in-flight request count.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, PartialOrd, Ord)]
pub(crate) struct RouteLoad {
    /// Prompt tokens of the requests currently charged to this worker.
    pub(crate) queued_tokens: u64,
    /// In-flight requests currently charged to this worker.
    pub(crate) requests: u64,
}

impl RouteLoad {
    /// A load carrying only a request count, as used by the request-count-ordered policies.
    pub(crate) const fn requests(requests: u64) -> Self {
        Self {
            queued_tokens: 0,
            requests,
        }
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
