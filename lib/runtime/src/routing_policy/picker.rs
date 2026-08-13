// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use dynamo_router_policy::RoutePicker as PolicyPicker;

use super::{CandidateView, RouteContext, RouteDecision, RoutePolicy};

struct RuntimeCandidates<'a, F> {
    candidates: CandidateView<'a>,
    load: &'a F,
}

impl<F> dynamo_router_policy::RouteCandidates for RuntimeCandidates<'_, F>
where
    F: Fn(u64) -> u64,
{
    #[inline(always)]
    fn len(&self) -> usize {
        self.candidates.len()
    }

    #[inline(always)]
    fn load(&self, index: usize) -> u64 {
        (self.load)(self.candidates.target(index).worker_id)
    }

    #[inline(always)]
    fn device(&self, index: usize) -> dynamo_router_policy::RouteDevice {
        match self.candidates {
            CandidateView::Workers(_) => dynamo_router_policy::RouteDevice::Accelerator,
            CandidateView::DeviceAware(candidates) => candidates[index].device,
        }
    }

    #[inline(always)]
    fn cache_hits(&self, index: usize) -> usize {
        match self.candidates {
            CandidateView::Workers(_) => 0,
            CandidateView::DeviceAware(candidates) => candidates[index].cache_hits,
        }
    }
}

#[derive(Debug)]
pub(crate) struct RoutePicker {
    inner: PolicyPicker,
}

impl RoutePicker {
    pub(crate) const fn new(policy: RoutePolicy) -> Self {
        Self {
            inner: PolicyPicker::new(policy),
        }
    }

    pub(crate) const fn policy(&self) -> RoutePolicy {
        self.inner.policy()
    }

    #[inline(always)]
    pub(crate) fn peek(
        &self,
        candidates: CandidateView<'_>,
        context: RouteContext,
        load: impl Fn(u64) -> u64,
    ) -> Option<RouteDecision> {
        let rows = RuntimeCandidates {
            candidates,
            load: &load,
        };
        self.inner
            .peek(&rows, context)
            .map(|decision| RouteDecision {
                target: candidates.target(decision.index),
                admission: decision.admission,
            })
    }

    #[inline(always)]
    pub(crate) fn select(
        &self,
        candidates: CandidateView<'_>,
        context: RouteContext,
        load: impl Fn(u64) -> u64,
    ) -> Option<RouteDecision> {
        let rows = RuntimeCandidates {
            candidates,
            load: &load,
        };
        self.inner
            .select(&rows, context)
            .map(|decision| RouteDecision {
                target: candidates.target(decision.index),
                admission: decision.admission,
            })
    }
}
