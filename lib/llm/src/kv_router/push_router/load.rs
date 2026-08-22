// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::sync::Arc;

use anyhow::{Context, Result};
use dynamo_kv_router::selector::{WorkerSelectionInput, WorkerSelector};
use dynamo_runtime::pipeline::{OccupancyReservation, PushRouter, RoutingOccupancyState};
use dynamo_runtime::protocols::annotated::Annotated;

use crate::{
    local_model::runtime_config::ModelRuntimeConfig, preprocessor::PreprocessedRequest,
    protocols::common::llm_backend::LLMEngineOutput,
};

use super::builtin::BuiltinWorkerSelector;

/// O(1) active-request accounting behind builtin `WorkerInputs::LOAD`.
///
/// RoutingHost owns policy selection and the request guard owns the returned
/// reservation. PushRouter supplies only discovery eligibility and transport.
pub(crate) struct RoutingLoadState {
    occupancy: Arc<RoutingOccupancyState>,
}

impl RoutingLoadState {
    pub(crate) fn new(
        router: &PushRouter<PreprocessedRequest, Annotated<LLMEngineOutput>>,
    ) -> Result<Self> {
        let occupancy = router
            .routing_occupancy_state()
            .context("load-aware router has no occupancy capability")?;
        Ok(Self { occupancy })
    }

    pub(crate) fn select_and_reserve(
        &self,
        router: &PushRouter<PreprocessedRequest, Annotated<LLMEngineOutput>>,
        selector: &BuiltinWorkerSelector,
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
            .select_and_reserve_with(&candidates, |load| {
                selector
                    .select_worker(WorkerSelectionInput::<ModelRuntimeConfig>::hosted(
                        &candidates,
                        Some(load),
                    ))
                    .map(|selection| selection.worker.worker_id)
            })?;
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
        selector: &BuiltinWorkerSelector,
    ) -> Option<u64> {
        let candidates = router.selectable_worker_ids().ok()?;
        let load = |worker_id| self.occupancy.load(worker_id);
        selector
            .peek_worker(WorkerSelectionInput::hosted(&candidates, Some(&load)))
            .ok()
    }
}

pub(crate) struct RoutingLoadSelection {
    pub(crate) worker_id: u64,
    pub(crate) candidate_count: usize,
    pub(crate) load: u64,
    pub(crate) reservation: OccupancyReservation,
}

pub(crate) type RoutingLoadReservation = OccupancyReservation;
