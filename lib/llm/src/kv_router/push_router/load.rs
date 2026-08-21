// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::sync::Arc;

use anyhow::{Context, Result};
use dynamo_runtime::pipeline::{
    OccupancyPolicy, OccupancyReservation, PushRouter, RoutingOccupancyState,
};
use dynamo_runtime::protocols::annotated::Annotated;

use crate::{preprocessor::PreprocessedRequest, protocols::common::llm_backend::LLMEngineOutput};

use super::BuiltinRoutingPolicy;

/// O(1) active-request accounting behind builtin [`WorkerInputs::LOAD`].
///
/// RoutingHost owns policy selection and the request guard owns the returned
/// reservation. PushRouter supplies only discovery eligibility and transport.
pub(crate) struct RoutingLoadState {
    occupancy: Arc<RoutingOccupancyState>,
    policy: OccupancyPolicy,
}

impl RoutingLoadState {
    pub(crate) fn new(
        router: &PushRouter<PreprocessedRequest, Annotated<LLMEngineOutput>>,
        policy: BuiltinRoutingPolicy,
    ) -> Result<Self> {
        let policy = match policy {
            BuiltinRoutingPolicy::PowerOfTwoChoices => OccupancyPolicy::PowerOfTwoChoices,
            BuiltinRoutingPolicy::LeastLoaded => OccupancyPolicy::LeastLoaded,
            BuiltinRoutingPolicy::RoundRobin | BuiltinRoutingPolicy::Random => {
                anyhow::bail!("{policy:?} does not consume LOAD")
            }
        };
        let occupancy = router
            .routing_occupancy_state()
            .context("load-aware router has no occupancy capability")?;
        Ok(Self { occupancy, policy })
    }

    pub(crate) fn select_and_reserve(
        &self,
        router: &PushRouter<PreprocessedRequest, Annotated<LLMEngineOutput>>,
        pinned_worker: Option<u64>,
    ) -> Result<RoutingLoadSelection> {
        if let Some(worker_id) = pinned_worker {
            router.ensure_routable(worker_id)?;
            let reservation = self.occupancy.reserve(worker_id);
            return Ok(RoutingLoadSelection {
                worker_id,
                candidate_count: 1,
                load: reservation.load(),
                reservation,
            });
        }

        let candidates = router.selectable_worker_ids()?;
        let selection = self
            .occupancy
            .select_and_reserve(self.policy, &candidates)
            .context("load-aware routing had no selectable worker")?;
        let worker_id = selection.worker_id();
        let candidate_count = selection.candidate_count();
        let load = selection.load();
        Ok(RoutingLoadSelection {
            worker_id,
            candidate_count,
            load,
            reservation: selection.into_reservation(),
        })
    }

    pub(crate) fn peek(
        &self,
        router: &PushRouter<PreprocessedRequest, Annotated<LLMEngineOutput>>,
    ) -> Option<u64> {
        let candidates = router.selectable_worker_ids().ok()?;
        self.occupancy.peek_policy(self.policy, &candidates)
    }
}

pub(crate) struct RoutingLoadSelection {
    pub(crate) worker_id: u64,
    pub(crate) candidate_count: usize,
    pub(crate) load: u64,
    pub(crate) reservation: OccupancyReservation,
}

pub(crate) type RoutingLoadReservation = OccupancyReservation;
