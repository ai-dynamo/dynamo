// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Benchmark-only monolithic Dynamo KV-router implementation of AIPerf's
//! Steppable ABI.
//!
//! Configuration crosses the boundary only once, when a replay is created.
//! Request, event, and measurement records use the ABI crate's fixed-layout
//! data-plane records.

use aiperf_steppable_abi::{
    ByteSliceV1, CreateRequestV1, DirectRequestSliceV1, DirectRequestV1, EngineEventSliceV1,
    EngineEventV1, PluginDescriptorV1, PluginVTableV1, REQUEST_FACT_FLAG_ADMISSION,
    REQUEST_FACT_FLAG_LATENCIES, REQUEST_FACT_FLAG_OUTPUT_LENGTH, REQUEST_FLAG_UUID,
    ReplayHandleV1, ReplayStateV1, RequestFactSliceV1, RequestFactV1, RequestIdMutSliceV1,
    RequestIdSliceV1, RequestIdV1, SLA_FLAG_E2E, SLA_FLAG_ITL, SLA_FLAG_TTFT, SlaThresholdsV1,
    StatusV1, StepRequestV1, StepResultV1, U32SliceV1,
};
use aisimulate_core::replay::loadgen::{DynPlacement, SteppableAgg, SteppableReplay};
use aisimulate_core::replay::{
    DirectRequest, ReplayEngineConfig, ReplayEngineFactory, ReplayTerminalStatus, SlaThresholds,
};
use dynamo_mocker::placement::{
    KvReplayMetadata, KvRouterPlacement, MockEngineArgs, MockEngineArgsBuilder,
    RouterEventObservation,
};
use serde::{Deserialize, Serialize};
use std::ffi::c_char;
use uuid::Uuid;

/// Topology built by the backend for one steppable replay.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum BackendTopology {
    /// One or more aggregate workers using the selected placement policy.
    #[default]
    Aggregated,
}

/// Explicit location and creation inputs for a dynamic placement provider.
///
/// The backend never searches the environment, working directory, or plugin
/// registry for a placement provider. Selecting one always requires this
/// complete, provider-owned configuration in the create payload.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DynamicPlacementLocator {
    /// Shared library containing the V1 placement provider.
    pub library_path: std::path::PathBuf,
    /// Deterministic selector seed supplied to the provider.
    #[serde(default)]
    pub selector_seed: [u8; 32],
    /// Namespace identifying the provider's opaque options format.
    #[serde(default)]
    pub options_namespace: Vec<u8>,
    /// Provider-defined options in `options_namespace`.
    #[serde(default)]
    pub provider_options: Vec<u8>,
    /// Host-selected bounds for provider batch results.
    #[serde(default)]
    pub limits: DynamicPlacementLimits,
}

/// JSON representation of the output limits negotiated with a placement provider.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct DynamicPlacementLimits {
    /// Maximum mutations permitted in one provider batch.
    pub max_mutations: u64,
    /// Maximum admission decisions returned in one provider result.
    pub max_admission_results: u64,
    /// Maximum released placements returned in one provider result.
    pub max_released: u64,
    /// Maximum diagnostic bytes returned in one provider result.
    pub max_diagnostic_bytes: u64,
}

impl Default for DynamicPlacementLimits {
    fn default() -> Self {
        // The neutral adapter calls the V1 provider once per replay mutation,
        // so a one-record result bound is enough for the built-in composition.
        Self {
            max_mutations: 1,
            max_admission_results: 1,
            max_released: 1,
            max_diagnostic_bytes: 0,
        }
    }
}

/// Provider-owned configuration encoded in `CreateRequestV1::provider_payload`.
///
/// The aggregate fields deliberately match AISimulate's outer dynamic plugin.
/// `dynamic_placement` is not loaded here: its selector seed is consumed by
/// the statically linked Dynamo router so an oracle run accepts the same
/// aggregate payload without another FFI routing boundary.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct BackendConfig {
    /// Version of this provider configuration. Omitted payloads are V1 for
    /// compatibility with AISimulate's current outer-plugin configuration.
    #[serde(default = "default_backend_config_version")]
    pub version: u32,
    /// Backend topology selected before the replay is constructed.
    pub topology: BackendTopology,
    /// Scheduler configuration applied to the selected topology.
    pub engine: ReplayEngineConfig,
    /// Aggregate worker count.
    pub workers: usize,
    /// Prefill worker count for a disaggregated replay.
    pub prefill_workers: usize,
    /// Decode worker count for a disaggregated replay.
    pub decode_workers: usize,
    /// The outer plugin's placement locator; this monolithic provider retains
    /// its selector seed but does not dynamically load its library.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub dynamic_placement: Option<DynamicPlacementLocator>,
}

impl Default for BackendConfig {
    fn default() -> Self {
        Self {
            version: default_backend_config_version(),
            topology: BackendTopology::Aggregated,
            engine: ReplayEngineConfig::default(),
            workers: 1,
            prefill_workers: 1,
            decode_workers: 1,
            dynamic_placement: None,
        }
    }
}

const fn default_backend_config_version() -> u32 {
    1
}

impl BackendConfig {
    /// Minimal bounded aggregate configuration for a one-worker oracle run.
    #[must_use]
    pub fn one_worker() -> Self {
        Self::default()
    }
}

const MAX_PROVIDER_PAYLOAD_BYTES: u64 = 1024 * 1024;

type RouterPlacement = DynPlacement<RouterEventObservation, KvReplayMetadata>;

fn router_args(engine: &ReplayEngineConfig) -> anyhow::Result<MockEngineArgs> {
    MockEngineArgsBuilder::default()
        .block_size(engine.rank.block_size)
        .num_gpu_blocks(engine.rank.num_gpu_blocks)
        .enable_prefix_caching(engine.rank.enable_prefix_caching)
        .max_num_batched_tokens(Some(engine.rank.max_num_batched_tokens))
        .max_num_seqs(Some(engine.rank.max_num_seqs))
        .dp_size(engine.dp_size)
        .build()
        .map_err(Into::into)
}

const PROVIDER_ID: &[u8] = b"dynamo.kv-router.monolithic\0";

struct BackendReplay {
    engine: Box<dyn SteppableReplay>,
    last_error: String,
}

fn allocated_bytes(value: String) -> ByteSliceV1 {
    let bytes = value.into_bytes().into_boxed_slice();
    let len = bytes.len() as u64;
    let data = Box::into_raw(bytes).cast::<u8>();
    ByteSliceV1 { data, len }
}

unsafe fn backend_mut(handle: ReplayHandleV1) -> Result<&'static mut BackendReplay, StatusV1> {
    if handle.0.is_null() {
        return Err(StatusV1::INVALID_ARGUMENT);
    }
    // Safety: every non-null handle comes from `create`, and ownership stays
    // with the caller until `destroy` consumes it.
    Ok(unsafe { &mut *handle.0.cast::<BackendReplay>() })
}

unsafe fn borrowed_tokens(slice: U32SliceV1) -> Result<&'static [u32], StatusV1> {
    if slice.len > usize::MAX as u64 || (slice.data.is_null() && slice.len != 0) {
        return Err(StatusV1::INVALID_ARGUMENT);
    }
    if slice.len == 0 {
        return Ok(&[]);
    }
    // Safety: a non-empty ABI input slice is borrowed and valid for the call.
    Ok(unsafe { std::slice::from_raw_parts(slice.data, slice.len as usize) })
}

unsafe fn direct_request(request: DirectRequestV1) -> Result<DirectRequest, StatusV1> {
    if request.struct_size as usize != std::mem::size_of::<DirectRequestV1>()
        || request.max_output_tokens > usize::MAX as u64
    {
        return Err(StatusV1::INVALID_ARGUMENT);
    }
    // Safety: input slices are caller-owned and valid for this FFI call.
    let tokens = unsafe { borrowed_tokens(request.tokens) }?.to_vec();
    // Safety: input slices are caller-owned and valid for this FFI call.
    let output_token_ids = unsafe { borrowed_tokens(request.output_token_ids) }?.to_vec();
    Ok(DirectRequest {
        tokens,
        max_output_tokens: request.max_output_tokens as usize,
        output_token_ids: (!output_token_ids.is_empty()).then_some(output_token_ids),
        uuid: (request.flags & REQUEST_FLAG_UUID != 0).then_some(Uuid::from_bytes(request.uuid)),
        dp_rank: request.dp_rank,
        preferred_dp_rank: None,
        preferred_prefill_dp_rank: None,
        arrival_timestamp_ms: None,
        priority: request.priority,
        strict_priority: request.strict_priority,
        policy_class: None,
        replay_context: None,
    })
}

unsafe extern "C" fn create(
    request: CreateRequestV1,
    handle: *mut ReplayHandleV1,
    error: *mut ByteSliceV1,
) -> StatusV1 {
    if handle.is_null() || error.is_null() {
        return StatusV1::INVALID_ARGUMENT;
    }
    // Safety: pointers were validated above and outputs are written once.
    unsafe {
        *handle = ReplayHandleV1(std::ptr::null_mut());
        *error = ByteSliceV1::EMPTY;
    }
    if request.provider_payload.len > MAX_PROVIDER_PAYLOAD_BYTES
        || request.provider_payload.len > usize::MAX as u64
        || (request.provider_payload.data.is_null() && request.provider_payload.len != 0)
    {
        return StatusV1::INVALID_ARGUMENT;
    }
    // Safety: the ABI guarantees borrowed input bytes remain valid for this
    // call; null is accepted only for an empty slice as checked above.
    let payload = if request.provider_payload.len == 0 {
        &[]
    } else {
        // Safety: a non-empty ABI input slice is valid for this call.
        unsafe {
            std::slice::from_raw_parts(
                request.provider_payload.data.cast::<u8>(),
                request.provider_payload.len as usize,
            )
        }
    };
    let config: BackendConfig = match serde_json::from_slice(payload) {
        Ok(config) => config,
        Err(parse_error) => {
            // Safety: validated non-null output pointer.
            unsafe { *error = allocated_bytes(parse_error.to_string()) };
            return StatusV1::REJECTED;
        }
    };
    let created: anyhow::Result<Box<dyn SteppableReplay>> = (|| {
        anyhow::ensure!(
            config.version == default_backend_config_version(),
            "unsupported Dynamo steppable BackendConfig version {}",
            config.version
        );
        anyhow::ensure!(
            matches!(config.topology, BackendTopology::Aggregated),
            "the monolithic Dynamo provider supports only aggregated topology"
        );
        anyhow::ensure!(
            config.workers > 0,
            "aggregate worker count must be positive"
        );
        let selector_seed = config.dynamic_placement.as_ref().map_or(0, |locator| {
            u64::from_le_bytes(
                locator.selector_seed[..8]
                    .try_into()
                    .expect("fixed-size seed"),
            )
        });
        let router_args = router_args(&config.engine)?;
        let worker_count = config.workers;
        let engine = SteppableAgg::<RouterPlacement, RouterEventObservation, KvReplayMetadata>::with_placement(
            config.engine,
            &ReplayEngineFactory::new(),
            worker_count,
            move |dp_size, topology| {
                anyhow::ensure!(
                    topology.len() == worker_count,
                    "runtime published {} topology entries for {worker_count} worker(s) at dp_size {dp_size}",
                    topology.len()
                );
                anyhow::ensure!(
                    dp_size == router_args.dp_size.max(1),
                    "runtime DP size {dp_size} disagrees with Dynamo engine DP size {}",
                    router_args.dp_size.max(1)
                );
                let placement = KvRouterPlacement::new(
                    &router_args,
                    None,
                    None,
                    topology.len(),
                    Some(selector_seed),
                )?;
                Ok(Box::new(placement) as RouterPlacement)
            },
        )?;
        Ok(Box::new(engine) as Box<dyn SteppableReplay>)
    })();
    match created {
        Ok(engine) => {
            let replay = Box::new(BackendReplay {
                engine,
                last_error: String::new(),
            });
            // Safety: validated non-null output pointer.
            unsafe { *handle = ReplayHandleV1(Box::into_raw(replay).cast()) };
            StatusV1::OK
        }
        Err(build_error) => {
            // Safety: validated non-null output pointer.
            unsafe { *error = allocated_bytes(build_error.to_string()) };
            StatusV1::REJECTED
        }
    }
}

unsafe extern "C" fn submit(
    handle: ReplayHandleV1,
    request: DirectRequestV1,
    request_id: *mut RequestIdV1,
) -> StatusV1 {
    if request_id.is_null() {
        return StatusV1::INVALID_ARGUMENT;
    }
    // Safety: request slices are valid for this call as required by the ABI.
    let request = match unsafe { direct_request(request) } {
        Ok(request) => request,
        Err(status) => return status,
    };
    // Safety: the handle is only dereferenced after null validation.
    let replay = match unsafe { backend_mut(handle) } {
        Ok(replay) => replay,
        Err(status) => return status,
    };
    match replay.engine.submit(request) {
        Ok(uuid) => {
            // Safety: validated non-null output pointer.
            unsafe { *request_id = *uuid.as_bytes() };
            StatusV1::OK
        }
        Err(error) => {
            replay.last_error = error.to_string();
            StatusV1::REJECTED
        }
    }
}

unsafe extern "C" fn submit_batch(
    handle: ReplayHandleV1,
    requests: DirectRequestSliceV1,
    request_ids: RequestIdMutSliceV1,
) -> StatusV1 {
    if requests.len > usize::MAX as u64
        || request_ids.len != requests.len
        || request_ids.len > usize::MAX as u64
        || (requests.data.is_null() && requests.len != 0)
        || (request_ids.data.is_null() && request_ids.len != 0)
    {
        return StatusV1::INVALID_ARGUMENT;
    }
    let requests = if requests.len == 0 {
        &[]
    } else {
        // Safety: non-empty input batch is valid for this FFI call.
        unsafe { std::slice::from_raw_parts(requests.data, requests.len as usize) }
    };
    let converted = match requests
        .iter()
        .map(|request| {
            // Safety: each record's borrowed fields are valid for this call.
            unsafe { direct_request(*request) }
        })
        .collect::<Result<Vec<_>, _>>()
    {
        Ok(requests) => requests,
        Err(status) => return status,
    };
    let output = if request_ids.len == 0 {
        &mut []
    } else {
        // Safety: caller supplies a writable output batch matching input size.
        unsafe { std::slice::from_raw_parts_mut(request_ids.data, request_ids.len as usize) }
    };
    let replay = match unsafe { backend_mut(handle) } {
        Ok(replay) => replay,
        Err(status) => return status,
    };
    for (request, output) in converted.into_iter().zip(output.iter_mut()) {
        match replay.engine.submit(request) {
            Ok(uuid) => *output = *uuid.as_bytes(),
            Err(error) => {
                replay.last_error = error.to_string();
                return StatusV1::REJECTED;
            }
        }
    }
    StatusV1::OK
}

unsafe extern "C" fn cancel(
    handle: ReplayHandleV1,
    request_id: *const RequestIdV1,
    event: *mut EngineEventV1,
    canceled: *mut u8,
) -> StatusV1 {
    if request_id.is_null() || event.is_null() || canceled.is_null() {
        return StatusV1::INVALID_ARGUMENT;
    }
    // Safety: validated non-null input pointer.
    let request_id = Uuid::from_bytes(unsafe { *request_id });
    let replay = match unsafe { backend_mut(handle) } {
        Ok(replay) => replay,
        Err(status) => return status,
    };
    match replay.engine.cancel(request_id) {
        Ok(Some(terminal)) => {
            let terminal_status = match terminal.terminal_status {
                Some(ReplayTerminalStatus::Completed) => 1,
                Some(ReplayTerminalStatus::Rejected) => 2,
                Some(ReplayTerminalStatus::Canceled) => 3,
                Some(ReplayTerminalStatus::Failed) => 4,
                None => 0,
            };
            // Safety: validated output pointers.
            unsafe {
                *event = EngineEventV1 {
                    request_id: *terminal.uuid.as_bytes(),
                    flags: 1 << 1,
                    token_id: terminal.token_id.unwrap_or_default(),
                    terminal_status,
                    reserved: 0,
                };
                *canceled = 1;
            }
            StatusV1::OK
        }
        Ok(None) => {
            // Safety: validated output pointer.
            unsafe { *canceled = 0 };
            StatusV1::OK
        }
        Err(error) => {
            replay.last_error = error.to_string();
            StatusV1::REJECTED
        }
    }
}

unsafe extern "C" fn cancel_batch(
    handle: ReplayHandleV1,
    request_ids: RequestIdSliceV1,
    events: *mut EngineEventSliceV1,
) -> StatusV1 {
    if events.is_null()
        || request_ids.len > usize::MAX as u64
        || (request_ids.data.is_null() && request_ids.len != 0)
    {
        return StatusV1::INVALID_ARGUMENT;
    }
    let request_ids = if request_ids.len == 0 {
        &[]
    } else {
        // Safety: non-empty input batch is valid for this FFI call.
        unsafe { std::slice::from_raw_parts(request_ids.data, request_ids.len as usize) }
    };
    let replay = match unsafe { backend_mut(handle) } {
        Ok(replay) => replay,
        Err(status) => return status,
    };
    let mut terminals = Vec::new();
    for request_id in request_ids {
        match replay.engine.cancel(Uuid::from_bytes(*request_id)) {
            Ok(Some(terminal)) => {
                let terminal_status = match terminal.terminal_status {
                    Some(ReplayTerminalStatus::Completed) => 1,
                    Some(ReplayTerminalStatus::Rejected) => 2,
                    Some(ReplayTerminalStatus::Canceled) => 3,
                    Some(ReplayTerminalStatus::Failed) => 4,
                    None => 0,
                };
                terminals.push(EngineEventV1 {
                    request_id: *terminal.uuid.as_bytes(),
                    flags: 1 << 1,
                    token_id: terminal.token_id.unwrap_or_default(),
                    terminal_status,
                    reserved: 0,
                });
            }
            Ok(None) => {}
            Err(error) => {
                replay.last_error = error.to_string();
                return StatusV1::REJECTED;
            }
        }
    }
    let terminals = terminals.into_boxed_slice();
    let len = terminals.len() as u64;
    let data = Box::into_raw(terminals).cast::<EngineEventV1>();
    // Safety: validated non-null output pointer.
    unsafe { *events = EngineEventSliceV1 { data, len } };
    StatusV1::OK
}

unsafe extern "C" fn step(
    handle: ReplayHandleV1,
    request: StepRequestV1,
    result: *mut StepResultV1,
) -> StatusV1 {
    if result.is_null()
        || request.struct_size as usize != std::mem::size_of::<StepRequestV1>()
        || request.until_ms.is_nan()
    {
        return StatusV1::INVALID_ARGUMENT;
    }
    // Safety: the handle is only dereferenced after null validation.
    let replay = match unsafe { backend_mut(handle) } {
        Ok(replay) => replay,
        Err(status) => return status,
    };
    let outcome = match replay.engine.step_until(request.until_ms) {
        Ok(outcome) => outcome,
        Err(error) => {
            replay.last_error = error.to_string();
            return StatusV1::REJECTED;
        }
    };
    let request_facts = outcome
        .events
        .iter()
        .filter_map(|event| {
            let mut flags = 0;
            let mut reused_input_tokens = 0;
            let mut admission_ms = 0.0;
            if let Some((at_ms, reused)) = replay.engine.request_admission(event.uuid) {
                flags |= REQUEST_FACT_FLAG_ADMISSION;
                admission_ms = at_ms;
                reused_input_tokens = reused as u64;
            }
            let mut ttft_ms = 0.0;
            let mut mean_itl_ms = 0.0;
            if let Some((ttft, mean_itl)) = replay.engine.request_latencies(event.uuid) {
                flags |= REQUEST_FACT_FLAG_LATENCIES;
                ttft_ms = ttft;
                mean_itl_ms = mean_itl;
            }
            let mut output_length = 0;
            if let Some(length) = replay.engine.actual_output_length(event.uuid) {
                flags |= REQUEST_FACT_FLAG_OUTPUT_LENGTH;
                output_length = length as u64;
            }
            (flags != 0).then_some(RequestFactV1 {
                request_id: *event.uuid.as_bytes(),
                flags,
                reserved: 0,
                reused_input_tokens,
                output_length,
                admission_ms,
                ttft_ms,
                mean_itl_ms,
            })
        })
        .collect::<Vec<_>>()
        .into_boxed_slice();
    let request_facts_len = request_facts.len() as u64;
    let request_facts_data = Box::into_raw(request_facts).cast::<RequestFactV1>();
    let events = outcome
        .events
        .into_iter()
        .map(|event| {
            let mut flags = 0;
            if event.emitted_token {
                flags |= 1;
            }
            if event.terminal_status.is_some() {
                flags |= 1 << 1;
            }
            let terminal_status = match event.terminal_status {
                Some(ReplayTerminalStatus::Completed) => 1,
                Some(ReplayTerminalStatus::Rejected) => 2,
                Some(ReplayTerminalStatus::Canceled) => 3,
                Some(ReplayTerminalStatus::Failed) => 4,
                None => 0,
            };
            EngineEventV1 {
                request_id: *event.uuid.as_bytes(),
                flags,
                token_id: event.token_id.unwrap_or_default(),
                terminal_status,
                reserved: 0,
            }
        })
        .collect::<Vec<_>>()
        .into_boxed_slice();
    let events_len = events.len() as u64;
    let events_data = Box::into_raw(events).cast::<EngineEventV1>();
    let next_event_ms = replay.engine.next_event_ms().unwrap_or(f64::NAN);
    // Safety: validated non-null output pointer. Event ownership transfers to
    // the host, which must call `release_events` exactly once.
    unsafe {
        *result = StepResultV1 {
            struct_size: std::mem::size_of::<StepResultV1>() as u32,
            flags: 0,
            end_ms: outcome.end_ms,
            next_event_ms,
            in_flight: replay.engine.in_flight() as u64,
            is_idle: u8::from(replay.engine.is_idle()),
            reserved: [0; 7],
            events: EngineEventSliceV1 {
                data: events_data,
                len: events_len,
            },
            request_facts: RequestFactSliceV1 {
                data: request_facts_data,
                len: request_facts_len,
            },
        };
    }
    StatusV1::OK
}

unsafe extern "C" fn take_report(
    handle: ReplayHandleV1,
    wall_ms: f64,
    report: *mut ByteSliceV1,
) -> StatusV1 {
    if report.is_null() || !wall_ms.is_finite() {
        return StatusV1::INVALID_ARGUMENT;
    }
    let replay = match unsafe { backend_mut(handle) } {
        Ok(replay) => replay,
        Err(status) => return status,
    };
    let report_value = match replay.engine.take_report(wall_ms) {
        Ok(report_value) => report_value,
        Err(error) => {
            replay.last_error = error.to_string();
            return StatusV1::REJECTED;
        }
    };
    let encoded = match serde_json::to_string(&report_value) {
        Ok(encoded) => encoded,
        Err(error) => {
            replay.last_error = error.to_string();
            return StatusV1::INTERNAL;
        }
    };
    // Safety: validated non-null output pointer.
    unsafe { *report = allocated_bytes(encoded) };
    StatusV1::OK
}

unsafe extern "C" fn release_bytes(bytes: ByteSliceV1) {
    if bytes.data.is_null() {
        return;
    }
    if bytes.len > usize::MAX as u64 {
        return;
    }
    // Safety: every non-empty output byte slice is allocated by
    // `allocated_bytes` as an exact-length boxed slice.
    unsafe {
        drop(Box::from_raw(std::ptr::slice_from_raw_parts_mut(
            bytes.data.cast_mut(),
            bytes.len as usize,
        )));
    }
}
unsafe extern "C" fn release_events(events: EngineEventSliceV1) {
    if events.data.is_null() {
        return;
    }
    if events.len > usize::MAX as u64 {
        return;
    }
    // Safety: `step` allocates exact-length boxed slices and transfers one
    // release obligation to the host.
    unsafe {
        drop(Box::from_raw(std::ptr::slice_from_raw_parts_mut(
            events.data.cast_mut(),
            events.len as usize,
        )));
    }
}
unsafe extern "C" fn release_request_facts(facts: RequestFactSliceV1) {
    if facts.data.is_null() || facts.len > usize::MAX as u64 {
        return;
    }
    // Safety: `step` allocates exact-length boxed slices and transfers one
    // release obligation to the host.
    unsafe {
        drop(Box::from_raw(std::ptr::slice_from_raw_parts_mut(
            facts.data.cast_mut(),
            facts.len as usize,
        )));
    }
}

unsafe extern "C" fn state(handle: ReplayHandleV1, state: *mut ReplayStateV1) -> StatusV1 {
    if state.is_null() {
        return StatusV1::INVALID_ARGUMENT;
    }
    // Safety: the non-null handle was created by this plugin and remains live
    // until the caller invokes `destroy`.
    let replay = match unsafe { backend_mut(handle) } {
        Ok(replay) => replay,
        Err(status) => return status,
    };
    let next_event_ms = replay.engine.next_event_ms().unwrap_or(f64::NAN);
    // Safety: validated non-null output pointer.
    unsafe {
        *state = ReplayStateV1 {
            now_ms: replay.engine.now_ms(),
            next_event_ms,
            in_flight: replay.engine.in_flight() as u64,
            is_idle: u8::from(replay.engine.is_idle()),
            reserved: [0; 7],
        };
    }
    StatusV1::OK
}

unsafe extern "C" fn advance_now_ms(handle: ReplayHandleV1, now_ms: f64) -> StatusV1 {
    if !now_ms.is_finite() {
        return StatusV1::INVALID_ARGUMENT;
    }
    // Safety: the handle is only dereferenced after null validation in
    // `backend_mut`.
    let replay = match unsafe { backend_mut(handle) } {
        Ok(replay) => replay,
        Err(status) => return status,
    };
    replay.engine.advance_now_ms(now_ms);
    StatusV1::OK
}

unsafe extern "C" fn set_capture_per_request(handle: ReplayHandleV1, capture: u8) -> StatusV1 {
    if capture > 1 {
        return StatusV1::INVALID_ARGUMENT;
    }
    let replay = match unsafe { backend_mut(handle) } {
        Ok(replay) => replay,
        Err(status) => return status,
    };
    replay.engine.set_capture_per_request(capture != 0);
    StatusV1::OK
}

unsafe extern "C" fn set_sla_thresholds(
    handle: ReplayHandleV1,
    thresholds: SlaThresholdsV1,
) -> StatusV1 {
    let selected = |flag, value| (thresholds.flags & flag != 0).then_some(value);
    let sla = SlaThresholds {
        ttft_ms: selected(SLA_FLAG_TTFT, thresholds.ttft_ms),
        itl_ms: selected(SLA_FLAG_ITL, thresholds.itl_ms),
        e2e_ms: selected(SLA_FLAG_E2E, thresholds.e2e_ms),
    };
    if ((sla.ttft_ms.is_some() || sla.itl_ms.is_some()) && sla.e2e_ms.is_some())
        || [sla.ttft_ms, sla.itl_ms, sla.e2e_ms]
            .into_iter()
            .flatten()
            .any(|value| !value.is_finite() || value <= 0.0)
    {
        return StatusV1::INVALID_ARGUMENT;
    }
    let replay = match unsafe { backend_mut(handle) } {
        Ok(replay) => replay,
        Err(status) => return status,
    };
    replay.engine.set_sla_thresholds(sla);
    StatusV1::OK
}

unsafe extern "C" fn last_error(handle: ReplayHandleV1, error: *mut ByteSliceV1) -> StatusV1 {
    if error.is_null() {
        return StatusV1::INVALID_ARGUMENT;
    }
    let replay = match unsafe { backend_mut(handle) } {
        Ok(replay) => replay,
        Err(status) => return status,
    };
    // Safety: validated non-null output pointer.
    unsafe {
        *error = if replay.last_error.is_empty() {
            ByteSliceV1::EMPTY
        } else {
            allocated_bytes(replay.last_error.clone())
        };
    }
    StatusV1::OK
}

unsafe extern "C" fn destroy(handle: ReplayHandleV1) {
    if handle.0.is_null() {
        return;
    }
    // Safety: caller transfers the unique handle returned by `create`.
    unsafe { drop(Box::from_raw(handle.0.cast::<BackendReplay>())) };
}

static VTABLE: PluginVTableV1 = PluginVTableV1 {
    struct_size: std::mem::size_of::<PluginVTableV1>() as u32,
    flags: 0,
    create: Some(create),
    submit: Some(submit),
    submit_batch: Some(submit_batch),
    cancel: Some(cancel),
    cancel_batch: Some(cancel_batch),
    step: Some(step),
    take_report: Some(take_report),
    release_bytes: Some(release_bytes),
    release_events: Some(release_events),
    release_request_facts: Some(release_request_facts),
    state: Some(state),
    advance_now_ms: Some(advance_now_ms),
    set_capture_per_request: Some(set_capture_per_request),
    set_sla_thresholds: Some(set_sla_thresholds),
    last_error: Some(last_error),
    destroy: Some(destroy),
};

static DESCRIPTOR: PluginDescriptorV1 = PluginDescriptorV1 {
    abi_major: 1,
    abi_minor: 0,
    struct_size: std::mem::size_of::<PluginDescriptorV1>() as u32,
    flags: 0,
    capabilities: 0,
    provider_id: PROVIDER_ID.as_ptr().cast::<c_char>(),
    vtable: &VTABLE,
};

/// Returns the static V1 plugin descriptor for dynamic loading.
#[unsafe(no_mangle)]
pub extern "C" fn aiperf_steppable_plugin_v1() -> *const PluginDescriptorV1 {
    &DESCRIPTOR
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn router_args_project_the_aggregate_engine_capacity() {
        let mut engine = ReplayEngineConfig {
            dp_size: 2,
            ..Default::default()
        };
        engine.rank.num_gpu_blocks = 29;
        engine.rank.max_num_batched_tokens = 37;
        engine.rank.max_num_seqs = 7;
        let args = router_args(&engine).expect("aggregate engine projects to Dynamo args");

        assert_eq!(args.dp_size, 2);
        assert_eq!(args.num_gpu_blocks, 29);
        assert_eq!(args.max_num_batched_tokens, Some(37));
        assert_eq!(args.max_num_seqs, Some(7));
    }
}
