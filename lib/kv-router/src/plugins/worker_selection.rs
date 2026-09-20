// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Public filter, scorer, and picker contracts and their input signals.

mod config;

pub use super::registry::{
    WorkerSelectionPolicyParameters, WorkerSelectionPolicyProvider,
    WorkerSelectionPolicyProviderError, WorkerSelectionPolicyRegistryError,
};
pub use crate::scheduling::selector::WorkerSelectionPolicy;
pub use crate::scheduling::{
    SessionContext, WorkerSelectionInputTrigger, WorkerSelectionPolicyError,
};
pub(crate) use config::RawWorkerSelectionConfig;
pub use config::{WorkerSelectionConfig, WorkerSelectionInstance};

use std::ops::BitOr;
use std::sync::Arc;

use crate::protocols::{WorkerAffinityTarget, WorkerWithDpRank};
use crate::scheduling::SchedulingRequest;
use crate::scheduling::selector::LogitWeights;
use crate::{KvRouterConfig, RoutingPartitionRef, WorkerType};

/// Factory that creates one worker-selection policy per routing partition.
pub type WorkerSelectionPolicyFactory = Arc<
    dyn for<'a> Fn(&KvRouterConfig, WorkerType, RoutingPartitionRef<'a>) -> WorkerSelectionPolicy
        + Send
        + Sync,
>;

/// Request-level values available to custom filters, scorers, and pickers.
pub struct WorkerSelectionContext<'a> {
    pub(crate) request: &'a SchedulingRequest,
    pub(crate) request_id: &'a str,
    pub(crate) request_blocks: u64,
    pub(crate) block_size: u32,
    pub(crate) track_prefill_tokens: bool,
    pub(crate) weights: LogitWeights,
    pub(crate) router_temperature_override: Option<f64>,
}

/// One eligible worker and the optional inputs requested by a filter or scorer.
pub struct WorkerCandidate {
    pub(crate) worker: WorkerWithDpRank,
    pub(crate) inputs: WorkerInputs,
    pub(crate) cache: WorkerCacheInput,
    pub(crate) load: WorkerLoadInput,
    pub(crate) preferred_taint_multiplier: Option<f64>,
}

/// One eligible worker and its total cost after all scorers run.
#[derive(Clone, Copy)]
pub struct ScoredWorkerCandidate {
    pub(crate) worker: WorkerWithDpRank,
    pub(crate) cost: f64,
    pub(crate) preferred_taint_multiplier: Option<f64>,
}

/// Optional worker-signal groups requested by scorers and pickers.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct WorkerInputs(u8);

impl WorkerInputs {
    /// Request no optional worker inputs.
    pub const NONE: Self = Self(0);
    /// Request KV-cache overlap inputs.
    pub const CACHE: Self = Self(1 << 0);
    /// Request active-load inputs.
    pub const LOAD: Self = Self(1 << 1);
    /// Request preferred-taint routing metadata.
    pub const PREFERRED_TAINT: Self = Self(1 << 2);
    /// Request host-owned active-request counts.
    pub const OCCUPANCY: Self = Self(1 << 5);
    pub(crate) const ALL: Self = Self(Self::CACHE.0 | Self::LOAD.0 | Self::PREFERRED_TAINT.0);

    pub const fn contains(self, other: Self) -> bool {
        self.0 & other.0 == other.0
    }

    pub(crate) fn without(self, other: Self) -> Self {
        Self(self.0 & !other.0)
    }
}

impl BitOr for WorkerInputs {
    type Output = Self;

    fn bitor(self, rhs: Self) -> Self::Output {
        Self(self.0 | rhs.0)
    }
}

/// KV-cache overlap values for one worker.
#[derive(Clone, Copy, Default)]
pub struct WorkerCacheInput {
    pub(crate) effective_overlap_blocks: f64,
    pub(crate) device_overlap_blocks: f64,
    pub(crate) host_overlap_blocks: f64,
    pub(crate) disk_overlap_blocks: f64,
    pub(crate) shared_beyond_device_blocks: u32,
}

/// Active-load values for one worker.
#[derive(Clone, Copy, Default)]
pub struct WorkerLoadInput {
    pub(crate) raw_prefill_blocks: f64,
    pub(crate) active_prefill_tokens: usize,
    pub(crate) decode_cost_blocks: f64,
    pub(crate) active_requests: usize,
}

/// Borrowed, index-aligned view of one custom picker's requested worker inputs.
#[derive(Clone, Copy)]
pub struct WorkerInputView<'a> {
    pub(crate) candidates: &'a [ScoredWorkerCandidate],
    pub(crate) cache: Option<&'a [WorkerCacheInput]>,
    pub(crate) load: Option<&'a [WorkerLoadInput]>,
}

/// Adds one finite cost contribution to each eligible worker.
pub trait WorkerScorer: Send {
    /// Declare the worker-signal groups needed by this scorer.
    fn required_worker_inputs(&self) -> WorkerInputs {
        WorkerInputs::NONE
    }

    /// Return one finite, lower-is-better cost contribution for an eligible worker row.
    fn score(
        &mut self,
        context: &WorkerSelectionContext<'_>,
        candidate: &WorkerCandidate,
    ) -> Result<f64, WorkerSelectionPolicyError>;
}

/// Filters run in declaration order for each candidate. Callback order across different
/// candidates and scorers is unspecified; implementations must not depend on filters and scorers
/// being interleaved.
pub trait WorkerFilter: Send {
    /// Declare the worker-signal groups needed by this filter.
    fn required_worker_inputs(&self) -> WorkerInputs {
        WorkerInputs::NONE
    }

    /// Return `true` to keep an eligible worker in the policy candidate set.
    fn keep(
        &mut self,
        context: &WorkerSelectionContext<'_>,
        candidate: &WorkerCandidate,
    ) -> Result<bool, WorkerSelectionPolicyError>;
}

/// Selects one row after all filters and scorers run.
pub trait WorkerPicker: Send {
    /// Declare the optional worker-signal columns needed by this picker.
    fn required_worker_inputs(&self) -> WorkerInputs {
        WorkerInputs::NONE
    }

    /// Return one row index from the host-owned eligible candidate table. Row order is
    /// unspecified; inspect candidate data instead of relying on a stable position.
    fn pick(
        &mut self,
        context: &WorkerSelectionContext<'_>,
        input: WorkerInputView<'_>,
    ) -> Result<usize, WorkerSelectionPolicyError>;
}

impl WorkerSelectionContext<'_> {
    /// Return the incoming prompt size in KV blocks.
    pub fn request_blocks(&self) -> u64 {
        self.request_blocks
    }

    /// Return the number of tokens in one KV block.
    pub fn block_size(&self) -> u32 {
        self.block_size
    }

    /// Return whether this request contributes to prefill-load tracking.
    pub fn tracks_prefill_tokens(&self) -> bool {
        self.track_prefill_tokens
    }

    /// Return the session metadata available to worker selection.
    pub fn session_context(&self) -> Option<&SessionContext> {
        self.request.session_context.as_ref()
    }

    /// Return the session-affinity target resolved by the request host.
    ///
    /// The default selector treats an eligible target as exclusive. Custom policies receive it as
    /// advisory context; it may be absent from their candidate set when unavailable or filtered.
    pub fn affinity_target(&self) -> Option<WorkerAffinityTarget> {
        self.request.affinity_target
    }

    /// Return the expected output length, if the request supplies one.
    pub fn expected_output_tokens(&self) -> Option<u32> {
        self.request.expected_output_tokens
    }

    /// Return the request's scheduler priority boost.
    pub fn priority_jump(&self) -> f64 {
        self.request.priority_jump
    }

    /// Return the request's strict integer priority.
    pub fn strict_priority(&self) -> u32 {
        self.request.strict_priority
    }

    /// Return the request-level router temperature override, if present.
    pub fn router_temperature_override(&self) -> Option<f64> {
        self.router_temperature_override
    }
}

impl WorkerCandidate {
    /// Return this candidate's worker ID and data-parallel rank.
    pub fn worker(&self) -> WorkerWithDpRank {
        self.worker
    }

    /// Return KV-cache inputs when the component requested [`WorkerInputs::CACHE`].
    pub fn cache(&self) -> Option<&WorkerCacheInput> {
        self.inputs
            .contains(WorkerInputs::CACHE)
            .then_some(&self.cache)
    }

    /// Return active-load inputs when the component requested [`WorkerInputs::LOAD`].
    pub fn load(&self) -> Option<&WorkerLoadInput> {
        self.inputs
            .contains(WorkerInputs::LOAD)
            .then_some(&self.load)
    }

    /// Return the optional cost multiplier from preferred routing constraints when the component
    /// requested [`WorkerInputs::PREFERRED_TAINT`].
    ///
    /// Required routing constraints are enforced by host eligibility. This preferred value is
    /// ordinary candidate metadata and is only materialized for components that declare the
    /// capability.
    pub fn preferred_taint_multiplier(&self) -> Option<f64> {
        self.preferred_taint_multiplier
    }

    pub(crate) fn with_inputs_from(&self, additional: &Self, inputs: WorkerInputs) -> Self {
        debug_assert_eq!(self.worker, additional.worker);
        Self {
            worker: self.worker,
            inputs,
            cache: if inputs.contains(WorkerInputs::CACHE) {
                if self.inputs.contains(WorkerInputs::CACHE) {
                    self.cache
                } else {
                    additional.cache
                }
            } else {
                WorkerCacheInput::default()
            },
            load: if inputs.contains(WorkerInputs::LOAD) {
                if self.inputs.contains(WorkerInputs::LOAD) {
                    self.load
                } else {
                    additional.load
                }
            } else {
                WorkerLoadInput::default()
            },
            preferred_taint_multiplier: self
                .preferred_taint_multiplier
                .or(additional.preferred_taint_multiplier),
        }
    }
}

impl ScoredWorkerCandidate {
    /// Return this candidate's worker ID and data-parallel rank.
    pub fn worker(&self) -> WorkerWithDpRank {
        self.worker
    }

    /// Return the sum of all scorer contributions for this candidate.
    pub fn cost(&self) -> f64 {
        self.cost
    }

    /// Return the optional cost multiplier from preferred routing constraints when the picker
    /// requested [`WorkerInputs::PREFERRED_TAINT`].
    pub fn preferred_taint_multiplier(&self) -> Option<f64> {
        self.preferred_taint_multiplier
    }
}

impl WorkerCacheInput {
    /// Return device-resident prefix overlap in KV blocks.
    pub fn device_overlap_blocks(&self) -> f64 {
        self.device_overlap_blocks
    }

    /// Return host-pinned prefix overlap in KV blocks.
    pub fn host_overlap_blocks(&self) -> f64 {
        self.host_overlap_blocks
    }

    /// Return disk prefix overlap in KV blocks.
    pub fn disk_overlap_blocks(&self) -> f64 {
        self.disk_overlap_blocks
    }

    /// Return shared-cache hits beyond the device-resident prefix.
    pub fn shared_beyond_device_blocks(&self) -> u32 {
        self.shared_beyond_device_blocks
    }
}

impl WorkerLoadInput {
    /// Return the tokens active in this worker's prefill stage.
    pub fn active_prefill_tokens(&self) -> usize {
        self.active_prefill_tokens
    }

    /// Return the projected active decode footprint in KV blocks.
    pub fn decode_cost_blocks(&self) -> f64 {
        self.decode_cost_blocks
    }

    /// Return this worker's active request count.
    pub fn active_requests(&self) -> usize {
        self.active_requests
    }
}

impl<'a> WorkerInputView<'a> {
    /// Return the eligible candidates and their total costs.
    pub fn candidates(self) -> &'a [ScoredWorkerCandidate] {
        self.candidates
    }

    /// Return index-aligned KV-cache inputs when the picker requested them.
    pub fn cache(self) -> Option<&'a [WorkerCacheInput]> {
        self.cache
    }

    /// Return index-aligned active-load inputs when the picker requested them.
    pub fn load(self) -> Option<&'a [WorkerLoadInput]> {
        self.load
    }
}
