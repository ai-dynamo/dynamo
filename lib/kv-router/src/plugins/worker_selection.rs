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

use rustc_hash::FxHashMap;

use crate::protocols::{WorkerAffinityTarget, WorkerId, WorkerWithDpRank};
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
    pub(crate) occupancy: WorkerOccupancyInput,
    pub(crate) device_aware: WorkerDeviceAwareInput,
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
    /// Request device class and request-specific multimodal-cache hits.
    pub const DEVICE_AWARE: Self = Self(1 << 6);

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

/// Reservation-aware in-flight request count for one worker/rank candidate.
#[derive(Clone, Copy, Default)]
pub struct WorkerOccupancyInput {
    pub(crate) active_requests: u64,
}

/// Device family used by device-aware routing.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum WorkerDevice {
    /// Host CPU.
    Cpu,
    #[default]
    /// Any non-CPU accelerator, or a worker whose device metadata is absent.
    Accelerator,
}

/// Request-specific device-aware inputs for one worker/rank candidate.
#[derive(Clone, Copy, Default)]
pub struct WorkerDeviceAwareInput {
    pub(crate) device: WorkerDevice,
    pub(crate) cache_hits: usize,
}

/// Owned request-plane signals supplied only when a policy requests
/// [`WorkerInputs::DEVICE_AWARE`].
#[derive(Clone, Default)]
pub struct DeviceAwareRequestInputs {
    workers: FxHashMap<WorkerId, WorkerDeviceAwareInput>,
    required_cache_hits: usize,
    non_cpu_to_cpu_ratio: usize,
}

impl DeviceAwareRequestInputs {
    /// Construct request-level device-aware signals keyed by worker ID.
    pub fn new(
        workers: impl IntoIterator<Item = (WorkerId, WorkerDeviceAwareInput)>,
        required_cache_hits: usize,
        non_cpu_to_cpu_ratio: usize,
    ) -> Self {
        Self {
            workers: workers.into_iter().collect(),
            required_cache_hits,
            non_cpu_to_cpu_ratio: non_cpu_to_cpu_ratio.max(1),
        }
    }

    pub(crate) fn worker(&self, worker_id: WorkerId) -> WorkerDeviceAwareInput {
        self.workers.get(&worker_id).copied().unwrap_or_default()
    }

    /// Number of distinct request cache keys required for a complete hit.
    pub fn required_cache_hits(&self) -> usize {
        self.required_cache_hits
    }

    /// Relative accelerator-to-CPU weighting used by the shared picker.
    pub fn non_cpu_to_cpu_ratio(&self) -> usize {
        self.non_cpu_to_cpu_ratio
    }
}

/// Borrowed, index-aligned view of one custom picker's requested worker inputs.
#[derive(Clone, Copy)]
pub struct WorkerInputView<'a> {
    pub(crate) candidates: &'a [ScoredWorkerCandidate],
    pub(crate) cache: Option<&'a [WorkerCacheInput]>,
    pub(crate) load: Option<&'a [WorkerLoadInput]>,
    pub(crate) occupancy: Option<&'a [WorkerOccupancyInput]>,
    pub(crate) device_aware: Option<&'a [WorkerDeviceAwareInput]>,
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

    /// Whether the selected request contributes to reservation-aware occupancy.
    ///
    /// Most policies admit every selected request. Device-aware routing overrides this for a
    /// complete multimodal-cache hit, matching the legacy top-level mode.
    fn occupancy_admission(
        &mut self,
        _context: &WorkerSelectionContext<'_>,
        _input: WorkerInputView<'_>,
        _selected_row: usize,
    ) -> bool {
        true
    }

    /// Whether an eligible affinity target must exclusively constrain selection.
    fn uses_exclusive_affinity_target(&self) -> bool {
        false
    }

    /// Whether requests without an explicit or bound-affinity target must be rejected.
    fn requires_exact_target(&self) -> bool {
        false
    }

    /// Whether this picker resolves a worker-only exact target to a DP rank.
    ///
    /// The host preserves its legacy requirement for an explicit rank unless the selected picker
    /// opts in. First-party non-KV adapters opt in because their source modes select workers before
    /// resolving a configured worker/rank row.
    fn resolves_worker_only_target(&self) -> bool {
        false
    }

    /// Whether this picker supports requests carrying a LoRA adapter name.
    ///
    /// Custom policies retain the configured KV scheduler's existing LoRA behavior by default.
    fn supports_lora(&self) -> bool {
        true
    }
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

    /// Return the exact request or bound-affinity target, if one exists.
    ///
    /// An explicit request pin wins. A worker-only affinity target remains worker-only so the
    /// policy can apply its documented DP-rank resolver.
    pub fn exact_target(&self) -> Option<WorkerAffinityTarget> {
        self.request
            .pinned_worker
            .map(WorkerAffinityTarget::from)
            .or(self.request.affinity_target)
    }

    /// Return request-plane device-aware context when the picker requested it.
    pub fn device_aware(&self) -> Option<&DeviceAwareRequestInputs> {
        self.request.device_aware_inputs.as_ref()
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

    /// Return reservation-aware occupancy when the component requested
    /// [`WorkerInputs::OCCUPANCY`].
    pub fn occupancy(&self) -> Option<&WorkerOccupancyInput> {
        self.inputs
            .contains(WorkerInputs::OCCUPANCY)
            .then_some(&self.occupancy)
    }

    /// Return device-aware inputs when the component requested
    /// [`WorkerInputs::DEVICE_AWARE`].
    pub fn device_aware(&self) -> Option<&WorkerDeviceAwareInput> {
        self.inputs
            .contains(WorkerInputs::DEVICE_AWARE)
            .then_some(&self.device_aware)
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
            occupancy: if inputs.contains(WorkerInputs::OCCUPANCY) {
                if self.inputs.contains(WorkerInputs::OCCUPANCY) {
                    self.occupancy
                } else {
                    additional.occupancy
                }
            } else {
                WorkerOccupancyInput::default()
            },
            device_aware: if inputs.contains(WorkerInputs::DEVICE_AWARE) {
                if self.inputs.contains(WorkerInputs::DEVICE_AWARE) {
                    self.device_aware
                } else {
                    additional.device_aware
                }
            } else {
                WorkerDeviceAwareInput::default()
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

impl WorkerOccupancyInput {
    /// Return the atomically reserved in-flight request count for this worker/rank.
    pub fn active_requests(&self) -> u64 {
        self.active_requests
    }
}

impl WorkerDeviceAwareInput {
    /// Construct device and request-specific cache-hit input for one worker.
    pub fn new(device: WorkerDevice, cache_hits: usize) -> Self {
        Self { device, cache_hits }
    }

    /// Return the worker's discovered device class.
    pub fn device(&self) -> WorkerDevice {
        self.device
    }

    /// Return this worker's hit count for the request's cache keys.
    pub fn cache_hits(&self) -> usize {
        self.cache_hits
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

    /// Return index-aligned reservation-aware occupancy when requested.
    pub fn occupancy(self) -> Option<&'a [WorkerOccupancyInput]> {
        self.occupancy
    }

    /// Return index-aligned device-aware inputs when requested.
    pub fn device_aware(self) -> Option<&'a [WorkerDeviceAwareInput]> {
        self.device_aware
    }
}
