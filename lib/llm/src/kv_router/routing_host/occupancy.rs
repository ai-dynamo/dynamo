// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::sync::Arc;

use anyhow::{Context, Result};
use dynamo_kv_router::selector::{HostedSelectionRequest, HostedWorkerPicker};
use dynamo_runtime::pipeline::{OccupancyReservation, PushRouter, RoutingOccupancyState};
use dynamo_runtime::protocols::annotated::Annotated;

use crate::{preprocessor::PreprocessedRequest, protocols::common::llm_backend::LLMEngineOutput};

pub(crate) struct HostedOccupancy {
    state: Arc<RoutingOccupancyState>,
}

impl HostedOccupancy {
    pub(crate) fn new(
        router: &PushRouter<PreprocessedRequest, Annotated<LLMEngineOutput>>,
    ) -> Result<Self> {
        let state = router
            .routing_occupancy_state()
            .context("occupancy-aware router has no occupancy capability")?;
        Ok(Self { state })
    }

    pub(crate) fn select_and_reserve(
        &self,
        router: &PushRouter<PreprocessedRequest, Annotated<LLMEngineOutput>>,
        picker: &HostedWorkerPicker,
        request: &HostedSelectionRequest,
        pinned_worker: Option<u64>,
    ) -> Result<HostedOccupancySelection> {
        if let Some(worker_id) = pinned_worker {
            router.ensure_routable(worker_id)?;
            let reservation = self.state.reserve(worker_id);
            return Ok(HostedOccupancySelection {
                worker_id,
                candidate_count: 1,
                occupancy: reservation.load(),
                reservation,
            });
        }

        let candidates = router.selectable_worker_ids()?;
        let selection = self
            .state
            .select_and_reserve_with(&candidates, |occupancy| {
                picker
                    .select(&candidates, Some(occupancy), request)
                    .map(|selection| selection.worker.worker_id)
            })?;
        let result = HostedOccupancySelection {
            worker_id: selection.worker_id(),
            candidate_count: selection.candidate_count(),
            occupancy: selection.load(),
            reservation: selection.into_reservation(),
        };
        Ok(result)
    }

    pub(crate) fn peek(
        &self,
        router: &PushRouter<PreprocessedRequest, Annotated<LLMEngineOutput>>,
        picker: &HostedWorkerPicker,
    ) -> Option<u64> {
        let candidates = router.selectable_worker_ids().ok()?;
        let occupancy = |worker_id| self.state.load(worker_id);
        picker
            .select(
                &candidates,
                Some(&occupancy),
                &HostedSelectionRequest::new(0, 0).with_preview(true),
            )
            .ok()
            .map(|selection| selection.worker.worker_id)
    }
}

pub(crate) struct HostedOccupancySelection {
    pub(crate) worker_id: u64,
    pub(crate) candidate_count: usize,
    pub(crate) occupancy: u64,
    pub(crate) reservation: OccupancyReservation,
}
