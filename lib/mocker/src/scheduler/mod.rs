// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Engine-specific scheduling implementations.

mod kv_event_sink;
mod live_boundary;
#[path = "sglang/mod.rs"]
pub mod sglang;
mod source_holds;
pub mod vllm;

pub use crate::common::protocols::ForwardPassSnapshot;
use crate::common::protocols::{DirectRequest, OutputSignal};
use dynamo_kv_router::protocols::RouterEvent;
pub(crate) use kv_event_sink::{CapturedRouterEventBuffer, capture_router_event_sink};
pub(crate) use live_boundary::{
    LiveBoundaryCore, LivePassExecution, LiveSchedulerState, spawn_live_scheduler,
};
use rustc_hash::FxHashSet;
pub(crate) use source_holds::{
    ActiveHandoffRequests, DestinationHolds, PendingDestinations, RemovedSource, SourceCompletion,
    SourceHolds,
};
pub use source_holds::{
    SchedulerCommand, SchedulerCommandEffects, SchedulerCommandResult, SchedulerLifecycleEvent,
};
use tokio::sync::{mpsc, oneshot};
use uuid::Uuid;

#[cfg(feature = "kvbm-offload")]
pub(crate) struct OffloadTickEffects {
    pub kv_events: Vec<RouterEvent>,
    pub lifecycle_events: Vec<SchedulerLifecycleEvent>,
}

/// Welford's online algorithm for count / sum / population-variance.
///
/// Mirrors the Python `WelfordAccumulator` in `forward_pass_metrics.py`.
#[derive(Default)]
pub(crate) struct WelfordAcc {
    pub(crate) count: u32,
    pub(crate) sum: f64,
    mean: f64,
    m2: f64,
}

impl WelfordAcc {
    pub(crate) fn add(&mut self, v: f64) {
        self.count += 1;
        self.sum += v;
        let delta = v - self.mean;
        self.mean += delta / self.count as f64;
        let delta2 = v - self.mean;
        self.m2 += delta * delta2;
    }

    pub(crate) fn variance(&self) -> f64 {
        if self.count == 0 {
            return 0.0;
        }
        self.m2 / self.count as f64
    }
}

/// Build a [`ForwardPassSnapshot`] from engine-agnostic iterators.
///
/// Each engine (vLLM, SGLang) calls this with its own iterators, avoiding
/// duplicated variance/accumulation logic.
///
/// - `scheduled_prefills`: `(prompt_len, prefix_tokens, tokens_computed)` per request
/// - `scheduled_decodes`: `sequence_len` per request
/// - `queued_prefills`: `prompt_len` per waiting prefill request
/// - `queued_decodes`: `kv_tokens` per preempted decode request
pub(crate) fn build_fpm_snapshot(
    scheduled_prefills: impl Iterator<Item = (u64, u64, u64)>,
    scheduled_decodes: impl Iterator<Item = u64>,
    queued_prefills: impl Iterator<Item = u64>,
    queued_decodes: impl Iterator<Item = u64>,
    wall_time_secs: f64,
) -> ForwardPassSnapshot {
    let mut prefill_acc = WelfordAcc::default();
    let mut decode_acc = WelfordAcc::default();
    let mut sum_prefill_tokens: u64 = 0;
    let mut sum_prefill_kv_tokens: u64 = 0;

    for (prompt_len, prefix_tokens, tokens_computed) in scheduled_prefills {
        sum_prefill_tokens += tokens_computed;
        sum_prefill_kv_tokens += prefix_tokens;
        prefill_acc.add(prompt_len as f64);
    }

    for sequence_len in scheduled_decodes {
        decode_acc.add(sequence_len as f64);
    }

    let mut queued_prefill_acc = WelfordAcc::default();
    let mut queued_decode_acc = WelfordAcc::default();

    for prompt_len in queued_prefills {
        queued_prefill_acc.add(prompt_len as f64);
    }

    for kv_tokens in queued_decodes {
        queued_decode_acc.add(kv_tokens as f64);
    }

    ForwardPassSnapshot {
        num_prefill_requests: prefill_acc.count,
        sum_prefill_tokens,
        var_prefill_length: prefill_acc.variance(),
        sum_prefill_kv_tokens,
        num_decode_requests: decode_acc.count,
        sum_decode_kv_tokens: decode_acc.sum as u64,
        var_decode_kv_tokens: decode_acc.variance(),
        num_queued_prefill: queued_prefill_acc.count,
        sum_queued_prefill_tokens: queued_prefill_acc.sum as u64,
        var_queued_prefill_length: queued_prefill_acc.variance(),
        num_queued_decode: queued_decode_acc.count,
        sum_queued_decode_kv_tokens: queued_decode_acc.sum as u64,
        var_queued_decode_kv_tokens: queued_decode_acc.variance(),
        wall_time_secs,
        ..Default::default()
    }
}

/// Recompute (visible output tokens, request-forwards) for debug validation
/// and the cold live-output suppression path. Ordinary pass execution carries
/// these counters directly from decode.
pub(crate) fn accept_length_sample(output_signals: &[OutputSignal]) -> (usize, usize) {
    let visible_tokens = output_signals
        .iter()
        .filter(|signal| !signal.rejected && signal.token_id.is_some())
        .count();
    if visible_tokens == 0 {
        return (0, 0);
    }

    let request_forwards = output_signals
        .iter()
        .filter(|signal| !signal.rejected && signal.token_id.is_some())
        .map(|signal| signal.uuid)
        .collect::<FxHashSet<_>>()
        .len();
    (visible_tokens, request_forwards)
}

#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub(crate) struct AcceptLengthSample {
    pub(crate) output_tokens: usize,
    pub(crate) decode_forwards: usize,
}

impl AcceptLengthSample {
    pub(crate) fn record_forward(&mut self, output_tokens: usize) {
        if output_tokens == 0 {
            return;
        }
        self.output_tokens += output_tokens;
        self.decode_forwards += 1;
    }
}

pub(crate) use sglang::SglangCore;
pub use sglang::SglangScheduler;
pub(crate) use vllm::VllmCore;
pub use vllm::{MockerMetrics, Scheduler};

#[derive(Debug, Clone)]
pub(crate) struct AdmissionEvent {
    pub(crate) uuid: Uuid,
    pub(crate) reused_input_tokens: usize,
}

/// Read-only scheduler facts safe to expose at an external placement boundary.
///
/// Token counters contain only tokens already materialized in each request's
/// current sequence (prompt plus generated tokens visible to the scheduler).
/// They deliberately exclude maximum/recorded output lengths and all future
/// timing. KV counters are physical G1 blocks; active and inactive cached
/// blocks both occupy capacity.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct SchedulerObservation {
    pub(crate) queued_requests: usize,
    pub(crate) running_requests: usize,
    pub(crate) queued_tokens: usize,
    pub(crate) running_tokens: usize,
    /// `None` means the scheduler has no configured finite sequence cap.
    pub(crate) max_num_seqs: Option<usize>,
    /// Cumulative rank-local scheduler preemptions committed by this time.
    pub(crate) preemption_count: u64,
    pub(crate) kv_capacity_blocks: usize,
    pub(crate) kv_occupied_blocks: usize,
    pub(crate) kv_free_blocks: usize,
}

#[derive(Debug, Clone)]
pub(crate) struct EnginePassResult {
    pub(crate) end_ms: f64,
    /// Rank-local token completion boundary before non-model wakeups (such as
    /// KVBM stall-advance) can extend `end_ms`.
    pub(crate) token_completion_ms: f64,
    pub(crate) completed_requests: usize,
    pub(crate) output_signals: Vec<OutputSignal>,
    pub(crate) admissions: Vec<AdmissionEvent>,
    pub(crate) lifecycle_events: Vec<SchedulerLifecycleEvent>,
    pub(crate) mocker_metrics: MockerMetrics,
    /// Controls when replay/live schedulers should expose this pass's buffered
    /// KV events to the real router or publisher sink.
    pub(crate) router_event_visibility: RouterEventVisibility,
    /// Router-visible KV events emitted during this pass.
    pub(crate) kv_events: Vec<RouterEvent>,
    /// Forward pass metrics snapshot for this iteration.
    pub(crate) fpm: Option<ForwardPassSnapshot>,
    /// Visible output tokens emitted by this pass for accept-length accounting.
    pub(crate) accept_length_output_tokens: usize,
    /// Number of request decode forwards that emitted those visible tokens.
    pub(crate) accept_length_decode_forwards: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum RouterEventVisibility {
    /// Expose buffered KV events when the pass starts, before the modeled sleep.
    PassStart,
    /// Expose buffered KV events when the pass finishes, before output flush.
    PassEnd,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum AdmissionStage {
    Materialized,
    PendingDestinationHead,
    FreshKv,
}

#[derive(Debug, Clone, Copy)]
pub(crate) struct AdmissionInvariant {
    pending_destination: bool,
}

impl AdmissionInvariant {
    pub(crate) fn new(pending_destination: bool) -> Self {
        Self {
            pending_destination,
        }
    }

    pub(crate) fn stage_for(self, materialized: bool) -> AdmissionStage {
        if materialized {
            AdmissionStage::Materialized
        } else if self.pending_destination {
            AdmissionStage::PendingDestinationHead
        } else {
            AdmissionStage::FreshKv
        }
    }
}

#[allow(clippy::large_enum_variant)]
pub(crate) enum EngineCore {
    Vllm(VllmCore),
    Sglang(SglangCore),
}

impl EngineCore {
    pub(crate) fn receive(&mut self, request: DirectRequest) -> Uuid {
        match self {
            Self::Vllm(core) => core.receive(request),
            Self::Sglang(core) => core.receive(request),
        }
    }

    pub(crate) fn is_empty(&self) -> bool {
        match self {
            Self::Vllm(core) => core.is_empty(),
            Self::Sglang(core) => core.is_empty(),
        }
    }

    #[allow(dead_code)]
    pub(crate) fn is_drained(&self) -> bool {
        match self {
            Self::Vllm(core) => core.is_drained(),
            Self::Sglang(core) => core.is_drained(),
        }
    }

    #[allow(dead_code)]
    pub(crate) fn apply_command(
        &mut self,
        command: SchedulerCommand,
    ) -> anyhow::Result<SchedulerCommandResult> {
        match self {
            Self::Vllm(core) => core.apply_command(command),
            Self::Sglang(core) => core.apply_command(command),
        }
    }

    pub(crate) fn apply_command_effects(
        &mut self,
        command: SchedulerCommand,
        allow_destination_admission: bool,
    ) -> anyhow::Result<SchedulerCommandEffects> {
        match self {
            Self::Vllm(core) => core.apply_command_effects(command, allow_destination_admission),
            Self::Sglang(core) => core.apply_command_effects(command, allow_destination_admission),
        }
    }

    pub(crate) fn retry_pending_destinations(&mut self) -> Vec<SchedulerLifecycleEvent> {
        match self {
            Self::Vllm(core) => core.retry_pending_destinations(),
            Self::Sglang(core) => core.retry_pending_destinations(),
        }
    }

    pub(crate) fn drain_kv_events(&self) -> Vec<dynamo_kv_router::protocols::RouterEvent> {
        match self {
            Self::Vllm(core) => core.drain_kv_events(),
            Self::Sglang(core) => core.drain_kv_events(),
        }
    }

    pub(crate) fn num_requests(&self) -> usize {
        match self {
            Self::Vllm(core) => core.num_requests(),
            Self::Sglang(core) => core.num_requests(),
        }
    }

    /// Return scheduler-safe placement facts when the backend implements the
    /// observation contract. The P0 contract is implemented for the shared
    /// vLLM/TRT-LLM core; unsupported backends remain explicit rather than
    /// synthesizing zero-valued load or KV state.
    pub(crate) fn scheduler_observation(&self) -> Option<SchedulerObservation> {
        match self {
            Self::Vllm(core) => Some(core.scheduler_observation()),
            Self::Sglang(_) => None,
        }
    }

    pub(crate) fn native_prefix_overlap_tokens(&self, tokens: &[u32]) -> Option<usize> {
        match self {
            Self::Vllm(core) => core.native_prefix_overlap_tokens(tokens),
            Self::Sglang(_) => None,
        }
    }

    pub(crate) fn native_prefix_snapshot(&self) -> Option<crate::kv_manager::NativePrefixSnapshot> {
        match self {
            Self::Vllm(core) => core.native_prefix_snapshot(),
            Self::Sglang(_) => None,
        }
    }

    pub(crate) fn pass_start_observation(&self) -> Option<SchedulerObservation> {
        match self {
            Self::Vllm(core) => core.pass_start_observation(),
            Self::Sglang(_) => None,
        }
    }

    pub(crate) fn try_execute_pass(&mut self, now_ms: f64) -> anyhow::Result<EnginePassResult> {
        match self {
            Self::Vllm(core) => core.try_execute_pass(now_ms),
            Self::Sglang(core) => core.try_execute_pass(now_ms),
        }
    }

    #[cfg(test)]
    pub(crate) fn execute_pass_unrecorded(&mut self, now_ms: f64) -> EnginePassResult {
        self.try_execute_pass(now_ms)
            .expect("engine scheduler pass failed")
    }

    #[cfg(feature = "kvbm-offload")]
    pub(crate) fn tick_offload_only(&mut self, now_ms: f64) -> OffloadTickEffects {
        match self {
            Self::Vllm(core) => core.tick_offload_only(now_ms),
            Self::Sglang(_) => OffloadTickEffects {
                kv_events: Vec::new(),
                lifecycle_events: Vec::new(),
            },
        }
    }

    #[cfg(feature = "kvbm-offload")]
    pub(crate) fn tick_offload_transport_only(&mut self, now_ms: f64) -> OffloadTickEffects {
        match self {
            Self::Vllm(core) => core.tick_offload_transport_only(now_ms),
            Self::Sglang(_) => OffloadTickEffects {
                kv_events: Vec::new(),
                lifecycle_events: Vec::new(),
            },
        }
    }

    #[cfg(feature = "kvbm-offload")]
    pub(crate) fn earliest_offload_deadline(&self) -> Option<f64> {
        match self {
            Self::Vllm(core) => core.earliest_offload_deadline(),
            Self::Sglang(_) => None,
        }
    }
}

pub struct SchedulerCommandEnvelope {
    pub command: SchedulerCommand,
    pub reply: oneshot::Sender<anyhow::Result<SchedulerCommandEffects>>,
}

/// Output channel used by a live scheduler.
///
#[derive(Clone)]
pub(crate) enum SchedulerOutputSender {
    Unbounded(mpsc::UnboundedSender<Vec<OutputSignal>>),
}

impl SchedulerOutputSender {
    pub(crate) async fn send(&self, signals: Vec<OutputSignal>) -> Result<(), Vec<OutputSignal>> {
        match self {
            Self::Unbounded(tx) => tx.send(signals).map_err(|error| error.0),
        }
    }
}

impl From<mpsc::UnboundedSender<Vec<OutputSignal>>> for SchedulerOutputSender {
    fn from(tx: mpsc::UnboundedSender<Vec<OutputSignal>>) -> Self {
        Self::Unbounded(tx)
    }
}

#[derive(Debug)]
pub(crate) enum LiveEngineEvent {
    Admissions(Vec<AdmissionEvent>),
    Outputs(Vec<OutputSignal>),
}

#[derive(Clone)]
pub(crate) enum SchedulerEventSender {
    Outputs(SchedulerOutputSender),
    Ordered {
        tx: mpsc::Sender<LiveEngineEvent>,
        forward_admissions: bool,
    },
}

pub(crate) enum SchedulerEventSendError {
    OutputClosed(Vec<OutputSignal>),
    OrderedLaneClosed,
}

impl SchedulerEventSender {
    pub(crate) async fn send_admissions(
        &self,
        admissions: &[AdmissionEvent],
    ) -> Result<(), SchedulerEventSendError> {
        if admissions.is_empty() {
            return Ok(());
        }
        match self {
            Self::Outputs(_) => {
                // Legacy output-only consumers do not have an admission event sink.
                Ok(())
            }
            Self::Ordered {
                forward_admissions: false,
                ..
            } => Ok(()),
            Self::Ordered { tx, .. } => tx
                .send(LiveEngineEvent::Admissions(admissions.to_vec()))
                .await
                .map_err(|_| SchedulerEventSendError::OrderedLaneClosed),
        }
    }

    pub(crate) async fn send_outputs(
        &self,
        signals: Vec<OutputSignal>,
    ) -> Result<(), SchedulerEventSendError> {
        match self {
            Self::Outputs(tx) => tx
                .send(signals)
                .await
                .map_err(SchedulerEventSendError::OutputClosed),
            Self::Ordered { tx, .. } => tx
                .send(LiveEngineEvent::Outputs(signals))
                .await
                .map_err(|_| SchedulerEventSendError::OrderedLaneClosed),
        }
    }
}

impl From<SchedulerOutputSender> for SchedulerEventSender {
    fn from(tx: SchedulerOutputSender) -> Self {
        Self::Outputs(tx)
    }
}

pub struct SchedulerCancellationEnvelope {
    pub request_id: Uuid,
    pub discard_pending_output: bool,
    pub reply: oneshot::Sender<anyhow::Result<SchedulerCommandEffects>>,
}

impl From<SchedulerCancellationEnvelope> for SchedulerCommandEnvelope {
    fn from(cancellation: SchedulerCancellationEnvelope) -> Self {
        Self {
            command: SchedulerCommand::CancelRequest {
                request_id: cancellation.request_id,
            },
            reply: cancellation.reply,
        }
    }
}

/// Engine-agnostic scheduler interface.
///
/// Both vLLM and SGLang schedulers implement this trait so that the engine
/// wrapper (`MockEngine`) can work with either backend through the same API.
pub trait SchedulerHandle: Send + Sync {
    /// Send a request to the scheduler's waiting queue.
    fn receive(&self, request: DirectRequest);

    /// Get a clone of the request sender channel for direct sending.
    fn request_sender(&self) -> mpsc::UnboundedSender<DirectRequest>;

    /// Get a watch receiver for scheduler metrics (active decode blocks, etc.).
    fn metrics_receiver(&self) -> tokio::sync::watch::Receiver<MockerMetrics>;

    /// Bounded ordered channel for request and disaggregated lifecycle commands.
    fn command_sender(&self) -> mpsc::Sender<SchedulerCommandEnvelope>;

    /// Bounded cancellation channel observed even while a modeled pass is running.
    ///
    /// Cancellation removes scheduler state and can suppress pending output immediately. During a
    /// modeled pass, published running/waiting metrics refresh at the next pass boundary; exact
    /// mid-pass metrics would require incremental per-request residency accounting.
    fn cancellation_sender(&self) -> mpsc::Sender<SchedulerCancellationEnvelope>;

    /// Take the single lifecycle-event stream owned by this DP-rank scheduler.
    fn take_lifecycle_receiver(&mut self) -> Option<mpsc::Receiver<SchedulerLifecycleEvent>>;
}

pub(crate) fn handoff_channel_capacity(args: &crate::common::protocols::MockEngineArgs) -> usize {
    args.effective_handoff_capacity()
        .checked_mul(2)
        .expect("mocker handoff channel capacity overflow")
}

/// Attach a [`crate::kvbm_offload::MockOffloadEngine`] driven by
/// wall-clock `now_ms` supplied by live replay. Returns `Ok(None)` unless
/// `num_g2_blocks` explicitly opts into G2 and `kv_bytes_per_token` supplies
/// the simulated block size.
#[cfg(feature = "kvbm-offload")]
pub async fn init_kvbm_live(
    args: &crate::common::protocols::MockEngineArgs,
    kv_manager: &mut crate::kv_manager::G1Manager,
) -> anyhow::Result<Option<std::sync::Arc<std::sync::Mutex<crate::kvbm_offload::MockOffloadEngine>>>>
{
    use crate::kvbm_offload::{KvbmDriveMode, KvbmOffloadConfig};
    let Some(config) = KvbmOffloadConfig::from_args(args)? else {
        return Ok(None);
    };
    let engine =
        std::thread::spawn(move || build_owned_offload_engine(config, KvbmDriveMode::Live))
            .join()
            .map_err(|_| anyhow::anyhow!("kvbm-offload live init thread panicked"))??;
    Ok(Some(kv_manager.attach_new_offload_engine(engine)))
}

/// Attach a [`crate::kvbm_offload::MockOffloadEngine`] driven by
/// virtual `now_ms` supplied by offline replay. Offline construction enables
/// an explicit completion/settlement boundary; live construction retains its
/// eager best-effort behavior.
#[cfg(feature = "kvbm-offload")]
pub fn init_kvbm_offline(
    args: &crate::common::protocols::MockEngineArgs,
    kv_manager: &mut crate::kv_manager::G1Manager,
) -> anyhow::Result<Option<std::sync::Arc<std::sync::Mutex<crate::kvbm_offload::MockOffloadEngine>>>>
{
    use crate::kvbm_offload::{KvbmDriveMode, KvbmOffloadConfig};
    let Some(config) = KvbmOffloadConfig::from_args(args)? else {
        return Ok(None);
    };
    tracing::debug!(
        num_g2_blocks = config.num_g2_blocks,
        num_g3_blocks = config.num_g3_blocks,
        g4_enabled = config.enable_g4_storage,
        offload_batch_size = config.offload_batch_size,
        bw_g1_to_g2_gbps = config.bandwidth_g1_to_g2_gbps,
        bw_g2_to_g1_gbps = config.bandwidth_g2_to_g1_gbps,
        bw_g2_to_g3_gbps = config.bandwidth_g2_to_g3_gbps,
        bw_g3_to_g2_gbps = config.bandwidth_g3_to_g2_gbps,
        bw_g2_to_g4_gbps = config.bandwidth_g2_to_g4_gbps,
        bw_g4_to_g2_gbps = config.bandwidth_g4_to_g2_gbps,
        "kvbm-offload: init_kvbm_offline attaching engine"
    );
    let engine = build_owned_offload_engine(config, KvbmDriveMode::OfflineDeterministic)?;
    Ok(Some(kv_manager.attach_new_offload_engine(engine)))
}

/// Build an offload engine with its private runtime attached.
///
/// kvbm-engine uses background pipeline/session tasks even though the mocker
/// scheduler is synchronous. Keeping the runtime inside the engine lets each
/// scheduler pass explicitly pump those tasks after transfer completions.
#[cfg(feature = "kvbm-offload")]
fn build_owned_offload_engine(
    config: crate::kvbm_offload::KvbmOffloadConfig,
    drive_mode: crate::kvbm_offload::KvbmDriveMode,
) -> anyhow::Result<crate::kvbm_offload::MockOffloadEngine> {
    let rt = tokio::runtime::Builder::new_multi_thread()
        .worker_threads(1)
        .enable_all()
        .build()?;
    let mut engine = rt.block_on(crate::kvbm_offload::MockOffloadEngine::new_with_drive_mode(
        config, drive_mode,
    ))?;
    engine.attach_runtime(rt);
    Ok(engine)
}

/// Shared test utilities for scheduler stress tests.
#[cfg(test)]
pub(crate) mod test_utils;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::common::handoff::HandoffId;
    use crate::common::protocols::{EngineType, MockEngineArgs, WorkerType};

    fn core(engine_type: EngineType, worker_type: WorkerType, blocks: usize) -> EngineCore {
        let args = MockEngineArgs::builder()
            .engine_type(engine_type)
            .block_size(4)
            .num_gpu_blocks(blocks)
            .max_num_batched_tokens(Some(16))
            .max_num_seqs(Some(1))
            .enable_prefix_caching(true)
            .worker_type(worker_type)
            .speedup_ratio(0.0)
            .build()
            .unwrap();
        match engine_type {
            EngineType::Vllm | EngineType::Trtllm => EngineCore::Vllm(VllmCore::new(args)),
            EngineType::Sglang => EngineCore::Sglang(SglangCore::new(args)),
        }
    }

    fn request(uuid: Uuid, tokens: Vec<u32>) -> DirectRequest {
        DirectRequest {
            tokens,
            max_output_tokens: 2,
            uuid: Some(uuid),
            dp_rank: 0,
            arrival_timestamp_ms: None,
            ..Default::default()
        }
    }

    fn destination_reservation_attempts(core: &EngineCore) -> usize {
        match core {
            EngineCore::Vllm(core) => core.destination_reservation_attempts(),
            EngineCore::Sglang(core) => core.destination_reservation_attempts(),
        }
    }

    fn request_metrics(core: &EngineCore) -> MockerMetrics {
        match core {
            EngineCore::Vllm(core) => core.mocker_metrics(),
            EngineCore::Sglang(core) => core.mocker_metrics(),
        }
    }

    #[test]
    fn request_cancellation_removes_waiting_and_running_requests_for_each_engine() {
        for (case, engine_type) in [EngineType::Vllm, EngineType::Sglang]
            .into_iter()
            .enumerate()
        {
            let mut core = core(engine_type, WorkerType::Aggregated, 16);
            let waiting_id = Uuid::from_u128(20_000 + case as u128);
            core.receive(request(waiting_id, (0..4).collect()));
            assert_eq!(request_metrics(&core).waiting_requests, 1);
            assert_eq!(
                core.apply_command(SchedulerCommand::CancelRequest {
                    request_id: waiting_id,
                })
                .unwrap(),
                SchedulerCommandResult::Applied
            );
            assert_eq!(
                core.apply_command(SchedulerCommand::CancelRequest {
                    request_id: waiting_id,
                })
                .unwrap(),
                SchedulerCommandResult::Noop
            );
            assert_eq!(core.num_requests(), 0);

            let running_id = Uuid::from_u128(20_100 + case as u128);
            let mut running_request = request(running_id, (100..108).collect());
            running_request.max_output_tokens = 32;
            core.receive(running_request);
            core.execute_pass_unrecorded(0.0);
            assert_eq!(request_metrics(&core).running_requests, 1);
            let active_blocks_before_cancel = request_metrics(&core).active_decode_blocks;
            assert!(
                active_blocks_before_cancel > 0,
                "{engine_type:?} running request should own KV blocks"
            );
            assert_eq!(
                core.apply_command(SchedulerCommand::CancelRequest {
                    request_id: running_id,
                })
                .unwrap(),
                SchedulerCommandResult::Applied
            );
            assert_eq!(core.num_requests(), 0);
            let active_blocks_after_cancel = request_metrics(&core).active_decode_blocks;
            assert!(
                active_blocks_after_cancel < active_blocks_before_cancel,
                "{engine_type:?} cancellation should release request-owned KV blocks"
            );
            if engine_type == EngineType::Vllm {
                assert_eq!(active_blocks_after_cancel, 0);
            }
        }
    }
    #[test]
    fn welford_acc_empty() {
        let acc = WelfordAcc::default();
        assert_eq!(acc.count, 0);
        assert_eq!(acc.sum, 0.0);
        assert_eq!(acc.variance(), 0.0);
    }

    #[test]
    fn accept_length_ignores_terminal_signals_without_tokens() {
        let token_uuid = Uuid::from_u128(1);
        let signals = [
            OutputSignal {
                uuid: Uuid::from_u128(2),
                token_id: None,
                completed: true,
                rejected: false,
                cached_tokens: None,
                handoff_delay_ms: None,
            },
            OutputSignal {
                uuid: token_uuid,
                token_id: Some(7),
                completed: false,
                rejected: false,
                cached_tokens: None,
                handoff_delay_ms: None,
            },
            OutputSignal {
                uuid: token_uuid,
                token_id: Some(8),
                completed: true,
                rejected: false,
                cached_tokens: None,
                handoff_delay_ms: None,
            },
            OutputSignal {
                uuid: Uuid::from_u128(3),
                token_id: Some(9),
                completed: true,
                rejected: true,
                cached_tokens: None,
                handoff_delay_ms: None,
            },
        ];

        assert_eq!(accept_length_sample(&signals), (2, 1));
    }

    #[test]
    fn accept_length_counters_count_tokens_and_forwards() {
        let mut sample = AcceptLengthSample::default();
        sample.record_forward(3);
        sample.record_forward(0);
        sample.record_forward(1);

        assert_eq!(sample.output_tokens, 4);
        assert_eq!(sample.decode_forwards, 2);
    }

    #[test]
    fn welford_acc_single_value() {
        let mut acc = WelfordAcc::default();
        acc.add(42.0);
        assert_eq!(acc.count, 1);
        assert_eq!(acc.sum, 42.0);
        assert_eq!(acc.variance(), 0.0);
    }

    #[test]
    fn welford_acc_population_variance() {
        let mut acc = WelfordAcc::default();
        // Values: 2, 4, 4, 4, 5, 5, 7, 9
        // Mean = 5, Population variance = 4.0
        for v in [2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0] {
            acc.add(v);
        }
        assert_eq!(acc.count, 8);
        assert_eq!(acc.sum, 40.0);
        assert!((acc.variance() - 4.0).abs() < 1e-10);
    }

    #[test]
    fn welford_acc_matches_python() {
        // Reproduce the Python WelfordAccumulator behavior:
        // values = [100, 200, 300], mean = 200,
        // population variance = ((100-200)^2 + (200-200)^2 + (300-200)^2) / 3
        //                     = (10000 + 0 + 10000) / 3 = 6666.666...
        let mut acc = WelfordAcc::default();
        acc.add(100.0);
        acc.add(200.0);
        acc.add(300.0);
        assert_eq!(acc.count, 3);
        assert_eq!(acc.sum, 600.0);
        let expected = 20000.0 / 3.0;
        assert!(
            (acc.variance() - expected).abs() < 1e-10,
            "expected {expected}, got {}",
            acc.variance()
        );
    }

    #[test]
    fn unavailable_destination_keeps_source_held_until_both_owners_are_cancelled() {
        for (case, engine_type) in [EngineType::Vllm, EngineType::Sglang]
            .into_iter()
            .enumerate()
        {
            let mut source = core(engine_type, WorkerType::Prefill, 8);
            let mut destination = core(engine_type, WorkerType::Decode, 2);
            let held_handoff = HandoffId::from(Uuid::from_u128(30_000 + case as u128));
            let capacity_handoff = HandoffId::from(Uuid::from_u128(30_100 + case as u128));
            let request_id = Uuid::from_u128(30_200 + case as u128);

            assert!(matches!(
                destination
                    .apply_command(SchedulerCommand::ReserveDestination {
                        handoff_id: capacity_handoff,
                        request: request(
                            Uuid::from_u128(30_300 + case as u128),
                            (100..108).collect(),
                        ),
                    })
                    .unwrap(),
                SchedulerCommandResult::DestinationAccepted { .. }
            ));
            source
                .apply_command(SchedulerCommand::SubmitHandoffPrefill {
                    handoff_id: held_handoff,
                    request: request(request_id, (0..8).collect()),
                })
                .unwrap();
            let mut now_ms = 0.0;
            for _ in 0..8 {
                let pass = source.execute_pass_unrecorded(now_ms);
                now_ms = pass.end_ms;
                if source.is_empty() {
                    break;
                }
            }
            assert!(source.is_empty());
            assert!(!source.is_drained());

            assert_eq!(
                destination
                    .apply_command(SchedulerCommand::ReserveDestination {
                        handoff_id: held_handoff,
                        request: request(request_id, (0..4).collect()),
                    })
                    .unwrap(),
                SchedulerCommandResult::DestinationAccepted { request_id }
            );
            assert_eq!(
                destination
                    .apply_command(SchedulerCommand::CancelDestination {
                        handoff_id: held_handoff,
                    })
                    .unwrap(),
                SchedulerCommandResult::Applied
            );
            assert_eq!(
                destination
                    .apply_command(SchedulerCommand::CancelDestination {
                        handoff_id: capacity_handoff,
                    })
                    .unwrap(),
                SchedulerCommandResult::Applied
            );
            assert_eq!(
                source
                    .apply_command(SchedulerCommand::CancelSource {
                        handoff_id: held_handoff,
                    })
                    .unwrap(),
                SchedulerCommandResult::Applied
            );
            assert!(source.is_empty());
            assert!(source.is_drained());
            assert!(destination.is_empty());
            assert!(destination.is_drained());
        }
    }

    #[test]
    fn destination_cancellation_retries_the_blocked_fifo_head() {
        for (case, engine_type) in [EngineType::Vllm, EngineType::Sglang]
            .into_iter()
            .enumerate()
        {
            let mut destination = core(engine_type, WorkerType::Decode, 2);
            let first_handoff = HandoffId::from(Uuid::from_u128(35_000 + case as u128));
            let second_handoff = HandoffId::from(Uuid::from_u128(35_100 + case as u128));
            let second_request = Uuid::from_u128(35_200 + case as u128);

            let first = destination
                .apply_command_effects(
                    SchedulerCommand::ReserveDestination {
                        handoff_id: first_handoff,
                        request: request(
                            Uuid::from_u128(35_300 + case as u128),
                            (100..108).collect(),
                        ),
                    },
                    true,
                )
                .unwrap();
            assert!(matches!(
                first.lifecycle_events.as_slice(),
                [SchedulerLifecycleEvent::DestinationReserved {
                    handoff_id,
                    ..
                }] if *handoff_id == first_handoff
            ));

            let second = destination
                .apply_command_effects(
                    SchedulerCommand::ReserveDestination {
                        handoff_id: second_handoff,
                        request: request(second_request, (200..204).collect()),
                    },
                    true,
                )
                .unwrap();
            assert!(second.lifecycle_events.is_empty());

            let canceled = destination
                .apply_command_effects(
                    SchedulerCommand::CancelDestination {
                        handoff_id: first_handoff,
                    },
                    true,
                )
                .unwrap();
            assert_eq!(canceled.result, SchedulerCommandResult::Applied);
            assert!(matches!(
                canceled.lifecycle_events.as_slice(),
                [SchedulerLifecycleEvent::DestinationReserved {
                    handoff_id,
                    request_id,
                    ..
                }] if *handoff_id == second_handoff && *request_id == second_request
            ));
        }
    }

    #[test]
    fn blocked_destination_head_prevents_fresh_kv_admission_without_spinning() {
        for (case, engine_type) in [EngineType::Vllm, EngineType::Sglang]
            .into_iter()
            .enumerate()
        {
            let mut destination = core(engine_type, WorkerType::Decode, 4);
            let owner_handoff = HandoffId::from(Uuid::from_u128(36_000 + case as u128));
            let blocked_handoff = HandoffId::from(Uuid::from_u128(36_100 + case as u128));
            let owner_request = Uuid::from_u128(36_200 + case as u128);
            let fresh_request = Uuid::from_u128(36_400 + case as u128);

            let owner = destination
                .apply_command_effects(
                    SchedulerCommand::ReserveDestination {
                        handoff_id: owner_handoff,
                        request: request(owner_request, (100..108).collect()),
                    },
                    true,
                )
                .unwrap();
            assert_eq!(owner.lifecycle_events.len(), 1);
            let occupied_before = match &destination {
                EngineCore::Vllm(core) => core.mocker_metrics().active_decode_blocks,
                EngineCore::Sglang(core) => core.mocker_metrics().active_decode_blocks,
            };
            assert!(occupied_before > 0);

            let blocked = destination
                .apply_command_effects(
                    SchedulerCommand::ReserveDestination {
                        handoff_id: blocked_handoff,
                        request: request(
                            Uuid::from_u128(36_300 + case as u128),
                            (200..212).collect(),
                        ),
                    },
                    true,
                )
                .unwrap();
            assert!(blocked.lifecycle_events.is_empty());
            assert!(destination.is_empty());
            assert!(!destination.is_drained());
            let pending_only = destination.execute_pass_unrecorded(0.0);
            assert_eq!(pending_only.end_ms, 0.0);
            assert!(pending_only.admissions.is_empty());
            assert!(pending_only.output_signals.is_empty());

            destination.receive(request(fresh_request, (300..304).collect()));
            let pass = destination.execute_pass_unrecorded(0.0);
            assert!(pass.admissions.is_empty());
            assert!(pass.output_signals.is_empty());
            assert_eq!(pass.end_ms, 0.0);
            assert_eq!(destination.num_requests(), 1);
            let occupied_after = match &destination {
                EngineCore::Vllm(core) => core.mocker_metrics().active_decode_blocks,
                EngineCore::Sglang(core) => core.mocker_metrics().active_decode_blocks,
            };
            assert_eq!(occupied_after, occupied_before);

            assert_eq!(
                destination
                    .apply_command(SchedulerCommand::ActivateDestination {
                        handoff_id: owner_handoff,
                    })
                    .unwrap(),
                SchedulerCommandResult::Applied
            );
            let materialized = destination.execute_pass_unrecorded(0.0);
            assert!(
                materialized
                    .admissions
                    .iter()
                    .any(|admission| admission.uuid == owner_request)
            );
            assert!(
                materialized
                    .admissions
                    .iter()
                    .all(|admission| admission.uuid != fresh_request)
            );

            assert_eq!(
                destination
                    .apply_command(SchedulerCommand::CancelDestination {
                        handoff_id: blocked_handoff,
                    })
                    .unwrap(),
                SchedulerCommandResult::Applied
            );
            assert_eq!(
                destination
                    .apply_command(SchedulerCommand::CancelDestination {
                        handoff_id: owner_handoff,
                    })
                    .unwrap(),
                SchedulerCommandResult::Applied
            );
            let fresh = destination.execute_pass_unrecorded(1.0);
            assert!(
                fresh
                    .admissions
                    .iter()
                    .any(|admission| admission.uuid == fresh_request)
            );
        }
    }

    #[test]
    fn unchanged_capacity_generation_does_not_reprobe_pending_destination() {
        for (case, engine_type) in [EngineType::Vllm, EngineType::Sglang]
            .into_iter()
            .enumerate()
        {
            let mut destination = core(engine_type, WorkerType::Decode, 2);
            let owner_handoff = HandoffId::from(Uuid::from_u128(36_500 + case as u128));
            let pending_handoff = HandoffId::from(Uuid::from_u128(36_600 + case as u128));
            let pending_request = Uuid::from_u128(36_700 + case as u128);

            let owner = destination
                .apply_command_effects(
                    SchedulerCommand::ReserveDestination {
                        handoff_id: owner_handoff,
                        request: request(Uuid::from_u128(36_800 + case as u128), (0..8).collect()),
                    },
                    true,
                )
                .unwrap();
            assert_eq!(owner.lifecycle_events.len(), 1);
            let pending = destination
                .apply_command_effects(
                    SchedulerCommand::ReserveDestination {
                        handoff_id: pending_handoff,
                        request: request(pending_request, (100..104).collect()),
                    },
                    true,
                )
                .unwrap();
            assert!(pending.lifecycle_events.is_empty());

            let attempts_after_initial_failure = destination_reservation_attempts(&destination);
            for _ in 0..3 {
                assert!(destination.retry_pending_destinations().is_empty());
            }
            assert_eq!(
                destination_reservation_attempts(&destination),
                attempts_after_initial_failure
            );

            let cancellation = destination
                .apply_command_effects(
                    SchedulerCommand::CancelDestination {
                        handoff_id: owner_handoff,
                    },
                    true,
                )
                .unwrap();
            assert!(matches!(
                cancellation.lifecycle_events.as_slice(),
                [SchedulerLifecycleEvent::DestinationReserved {
                    handoff_id,
                    request_id,
                    ..
                }] if *handoff_id == pending_handoff && *request_id == pending_request
            ));
            assert_eq!(
                destination_reservation_attempts(&destination),
                attempts_after_initial_failure + 1
            );
            assert_eq!(
                destination
                    .apply_command(SchedulerCommand::CancelDestination {
                        handoff_id: pending_handoff,
                    })
                    .unwrap(),
                SchedulerCommandResult::Applied
            );
        }
    }

    #[test]
    fn vllm_prebuilt_waiting_request_runs_before_blocked_pending_destination() {
        let mut destination = core(EngineType::Vllm, WorkerType::Decode, 4);
        let ready_handoff = HandoffId::from(Uuid::from_u128(36_900));
        let pending_handoff = HandoffId::from(Uuid::from_u128(36_901));
        let ready_request = Uuid::from_u128(36_902);
        let pending_request = Uuid::from_u128(36_903);
        let fresh_request = Uuid::from_u128(36_904);

        destination.receive(request(fresh_request, (200..204).collect()));

        assert_eq!(
            destination
                .apply_command(SchedulerCommand::ReserveDestination {
                    handoff_id: ready_handoff,
                    request: request(ready_request, (0..8).collect()),
                })
                .unwrap(),
            SchedulerCommandResult::DestinationAccepted {
                request_id: ready_request
            }
        );
        assert_eq!(
            destination
                .apply_command(SchedulerCommand::ActivateDestination {
                    handoff_id: ready_handoff,
                })
                .unwrap(),
            SchedulerCommandResult::Applied
        );
        let pending = destination
            .apply_command_effects(
                SchedulerCommand::ReserveDestination {
                    handoff_id: pending_handoff,
                    request: request(pending_request, (100..112).collect()),
                },
                true,
            )
            .unwrap();
        assert!(pending.lifecycle_events.is_empty());

        let first_pass = destination.execute_pass_unrecorded(0.0);
        assert!(
            first_pass
                .admissions
                .iter()
                .any(|admission| admission.uuid == ready_request)
        );
        assert!(
            first_pass
                .admissions
                .iter()
                .all(|admission| admission.uuid != fresh_request)
        );

        let mut reservation_events = Vec::new();
        for now_ms in 1..=4 {
            destination.execute_pass_unrecorded(f64::from(now_ms));
            reservation_events.extend(destination.retry_pending_destinations());
            if !reservation_events.is_empty() {
                break;
            }
        }
        assert!(matches!(
            reservation_events.as_slice(),
            [SchedulerLifecycleEvent::DestinationReserved {
                handoff_id,
                request_id,
                ..
            }] if *handoff_id == pending_handoff && *request_id == pending_request
        ));
        assert_eq!(
            destination
                .apply_command(SchedulerCommand::CancelDestination {
                    handoff_id: pending_handoff,
                })
                .unwrap(),
            SchedulerCommandResult::Applied
        );
    }

    #[test]
    fn activated_waiting_destination_cancels_without_consuming_the_running_slot() {
        for (case, engine_type) in [EngineType::Vllm, EngineType::Sglang]
            .into_iter()
            .enumerate()
        {
            let mut destination = core(engine_type, WorkerType::Decode, 16);
            let handoff_id = HandoffId::from(Uuid::from_u128(38_000 + case as u128));
            let request_id = Uuid::from_u128(38_100 + case as u128);
            let reserved = destination
                .apply_command_effects(
                    SchedulerCommand::ReserveDestination {
                        handoff_id,
                        request: request(request_id, (0..8).collect()),
                    },
                    true,
                )
                .unwrap();
            assert_eq!(reserved.lifecycle_events.len(), 1);
            let reserved_occupancy = match &destination {
                EngineCore::Vllm(core) => core.mocker_metrics().active_decode_blocks,
                EngineCore::Sglang(core) => core.mocker_metrics().active_decode_blocks,
            };
            assert!(reserved_occupancy > 0);

            destination.receive(DirectRequest {
                tokens: (100..108).collect(),
                max_output_tokens: 8,
                uuid: Some(Uuid::from_u128(38_200 + case as u128)),
                ..Default::default()
            });
            let pass = destination.execute_pass_unrecorded(0.0);
            assert_eq!(pass.admissions.len(), 1);
            let before_activation = match &destination {
                EngineCore::Vllm(core) => core.mocker_metrics().active_decode_blocks,
                EngineCore::Sglang(core) => core.mocker_metrics().active_decode_blocks,
            };
            assert_eq!(
                destination
                    .apply_command(SchedulerCommand::ActivateDestination { handoff_id })
                    .unwrap(),
                SchedulerCommandResult::Applied
            );
            let activated_occupancy = match &destination {
                EngineCore::Vllm(core) => core.mocker_metrics().active_decode_blocks,
                EngineCore::Sglang(core) => core.mocker_metrics().active_decode_blocks,
            };
            assert_eq!(activated_occupancy, before_activation);
            assert_eq!(
                destination
                    .apply_command(SchedulerCommand::CancelDestination { handoff_id })
                    .unwrap(),
                SchedulerCommandResult::Applied
            );
            assert_eq!(
                destination
                    .apply_command(SchedulerCommand::CancelDestination { handoff_id })
                    .unwrap(),
                SchedulerCommandResult::Noop
            );
            assert_eq!(destination.num_requests(), 1);
        }
    }

    #[test]
    fn preterminal_source_cancel_removes_scheduled_request_once() {
        for (case, engine_type) in [EngineType::Vllm, EngineType::Sglang]
            .into_iter()
            .enumerate()
        {
            let mut source = core(engine_type, WorkerType::Prefill, 8);
            let handoff_id = HandoffId::from(Uuid::from_u128(40_000 + case as u128));
            let request_id = Uuid::from_u128(40_100 + case as u128);
            source
                .apply_command(SchedulerCommand::SubmitHandoffPrefill {
                    handoff_id,
                    request: request(request_id, (0..8).collect()),
                })
                .unwrap();
            assert_eq!(source.num_requests(), 1);

            assert_eq!(
                source
                    .apply_command(SchedulerCommand::CancelSource { handoff_id })
                    .unwrap(),
                SchedulerCommandResult::Applied
            );
            assert!(source.is_empty());
            assert!(source.is_drained());
            assert_eq!(
                source
                    .apply_command(SchedulerCommand::CancelSource { handoff_id })
                    .unwrap(),
                SchedulerCommandResult::Noop
            );
        }
    }
}

#[cfg(all(test, feature = "kvbm-offload"))]
mod offload_init_tests {
    use super::{init_kvbm_live, init_kvbm_offline};
    use crate::common::protocols::{KvEventPublishers, MockEngineArgs};
    use crate::kv_manager::G1Manager;
    use crate::kvbm_offload::KvbmDriveMode;

    fn make_kv_manager() -> G1Manager {
        G1Manager::new_with_event_sink(8, 4, KvEventPublishers::default(), 0)
    }

    fn args_with_g2_and_bpt(bpt: usize) -> MockEngineArgs {
        MockEngineArgs::builder()
            .num_gpu_blocks(8)
            .num_g2_blocks(Some(8))
            .block_size(4)
            .kv_bytes_per_token(Some(bpt))
            .build()
            .unwrap()
            .normalized()
            .unwrap()
    }

    #[tokio::test]
    async fn init_kvbm_live_attaches_engine_when_g2_and_bpt_set() {
        let args = args_with_g2_and_bpt(131_072);
        let mut kv = make_kv_manager();
        assert!(!kv.has_offload_engine());
        let engine = init_kvbm_live(&args, &mut kv)
            .await
            .expect("init must succeed")
            .expect("engine built with G2 and bpt present");
        assert!(kv.has_offload_engine());
        // Returned Arc shares the same engine as the one on kv_manager;
        // earliest_offload_deadline reflects an idle engine.
        assert!(engine.lock().unwrap().earliest_pending_deadline().is_none());
        assert_eq!(engine.lock().unwrap().drive_mode(), KvbmDriveMode::Live);
        assert!(kv.earliest_offload_deadline().is_none());
    }

    #[tokio::test]
    async fn init_kvbm_live_returns_none_without_g2_blocks() {
        let args = MockEngineArgs::builder()
            .num_gpu_blocks(8)
            .block_size(4)
            .kv_bytes_per_token(Some(131_072))
            .build()
            .unwrap()
            .normalized()
            .unwrap();
        assert!(args.num_g2_blocks.is_none());
        let mut kv = make_kv_manager();
        let result = init_kvbm_live(&args, &mut kv)
            .await
            .expect("init must succeed");
        assert!(result.is_none());
        assert!(!kv.has_offload_engine());
    }

    #[tokio::test]
    async fn init_kvbm_live_returns_none_without_bpt() {
        let args = MockEngineArgs::default();
        assert!(args.kv_bytes_per_token.is_none());
        let mut kv = make_kv_manager();
        let result = init_kvbm_live(&args, &mut kv)
            .await
            .expect("init must succeed");
        assert!(result.is_none());
        assert!(!kv.has_offload_engine());
    }

    #[test]
    fn init_kvbm_offline_attaches_engine_and_keeps_runtime_alive() {
        // Sync entry: no ambient tokio runtime. init_kvbm_offline owns
        // its own runtime and moves it onto the engine via
        // attach_runtime. After init returns, the engine (and its
        // runtime) must still be usable — `tick` is a sync call that
        // internally depends on the worker thread continuing to drain
        // kvbm-engine's background tasks.
        let args = args_with_g2_and_bpt(131_072);
        let mut kv = make_kv_manager();
        let engine = init_kvbm_offline(&args, &mut kv)
            .expect("offline init must succeed")
            .expect("engine built with G2 and bpt present");
        assert!(kv.has_offload_engine());
        // Engine is still callable post-init — no runtime-dropped hang.
        engine.lock().unwrap().tick(100.0);
        assert_eq!(
            engine.lock().unwrap().drive_mode(),
            KvbmDriveMode::OfflineDeterministic
        );
        assert!(kv.earliest_offload_deadline().is_none());
    }

    #[test]
    fn init_kvbm_offline_returns_none_without_g2_blocks() {
        let args = MockEngineArgs::builder()
            .num_gpu_blocks(8)
            .block_size(4)
            .kv_bytes_per_token(Some(131_072))
            .build()
            .unwrap()
            .normalized()
            .unwrap();
        assert!(args.num_g2_blocks.is_none());
        let mut kv = make_kv_manager();
        let result = init_kvbm_offline(&args, &mut kv).expect("init must succeed");
        assert!(result.is_none());
        assert!(!kv.has_offload_engine());
    }

    #[test]
    fn init_kvbm_offline_returns_none_without_bpt() {
        let args = MockEngineArgs::default();
        assert!(args.kv_bytes_per_token.is_none());
        let mut kv = make_kv_manager();
        let result = init_kvbm_offline(&args, &mut kv).expect("init must succeed");
        assert!(result.is_none());
        assert!(!kv.has_offload_engine());
    }
}
