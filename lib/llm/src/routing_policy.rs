// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::{
    collections::{HashMap, HashSet},
    sync::Arc,
};

use dynamo_runtime::pipeline::Error;
use dynamo_scheduler::host::{RoutingConstraints, SchedulingRequest};

use crate::{discovery::RuntimeConfigWatch, preprocessor::PreprocessedRequest};
use dynamo_model_card::local_model::runtime_config::ModelRuntimeConfig;

pub(crate) type AttemptLease = Box<dyn Send + Sync>;

pub(crate) struct CandidateSelection {
    pub worker: u64,
    pub fallback: HashSet<u64>,
    pub lease: Option<AttemptLease>,
}

pub(crate) trait HostedCandidatePolicy: Send + Sync {
    /// `Ok(None)` defers to normal selection; policy rejections return `Err`, never fallback.
    fn select(
        &self,
        request: &PreprocessedRequest,
        pin: Option<u64>,
        live: &dyn Fn() -> Vec<u64>,
        free: &dyn Fn() -> Vec<u64>,
    ) -> Result<Option<CandidateSelection>, Error>;

    fn check_selected(
        &self,
        request: &PreprocessedRequest,
        worker: u64,
        pin: Option<u64>,
        live: &dyn Fn() -> Vec<u64>,
    ) -> Result<(), Error>;
}

pub(crate) trait KvAdmissionPolicy: Send + Sync {
    fn restrict(
        &self,
        request: &PreprocessedRequest,
        workers: &HashMap<u64, ModelRuntimeConfig>,
        live: &HashSet<u64>,
        allowed: Option<HashSet<u64>>,
        constraints: &RoutingConstraints,
        pins: &[u64],
    ) -> Result<Option<HashSet<u64>>, Error>;

    fn check_current(
        &self,
        request: &PreprocessedRequest,
        worker: u64,
        workers: &HashMap<u64, ModelRuntimeConfig>,
        live: &HashSet<u64>,
        allowed: Option<HashSet<u64>>,
        constraints: &RoutingConstraints,
    ) -> Result<(), Error>;

    /// `None` leaves availability unrestricted; `Some(empty)` excludes every worker.
    fn refresh_admission(
        &self,
        request: &SchedulingRequest,
        available: Option<Arc<HashSet<u64>>>,
        workers: &RuntimeConfigWatch,
    ) -> Option<Arc<HashSet<u64>>>;
}

#[derive(Clone, Copy)]
pub(crate) enum ExecutionStage {
    Prefill,
    Encoder,
}

#[derive(Clone, Copy)]
pub(crate) struct ExecutionRequirements {
    pub allows_replay: bool,
    pub policy_required: bool,
    pub accounting_attribution: dynamo_scheduler::host::AccountingAttribution,
}

pub(crate) fn execution_requirements(request: &PreprocessedRequest) -> ExecutionRequirements {
    crate::lora_routing::execution_requirements(request)
}

pub(crate) fn check_candidate_policy(
    request: &PreprocessedRequest,
    installed: bool,
) -> Result<(), Error> {
    if execution_requirements(request).policy_required {
        crate::lora_routing::check_candidate_policy(request, installed)?;
    }
    Ok(())
}

pub(crate) fn check_stage(
    request: &PreprocessedRequest,
    stage: ExecutionStage,
) -> Result<(), Error> {
    crate::lora_routing::check_stage(request, stage)
}
