// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Benchmark-only monolithic Dynamo KV-router implementation of AIPerf's
//! Steppable ABI.
//!
//! Configuration crosses the boundary only once, when a replay is created.
//! Request, event, and measurement records use the ABI crate's fixed-layout
//! data-plane records.

use aiperf_steppable_abi::{
    ByteSliceV1, CAPABILITY_COMPACT_BUFFER_LEASES_V1, CAPABILITY_COMPACT_REQUEST_V1,
    CompactRequestV1, CreateRequestV1, DirectRequestSliceV1, DirectRequestV1, EngineEventSliceV1,
    EngineEventV1, HashBufferIdV1, HashBufferLeaseCallbacksV1, HashBufferRangeV1,
    MAX_REPLAY_CONTEXT_METADATA_BYTES_V1, PluginDescriptorV1, PluginVTableV1,
    REPLAY_CONTEXT_FLAG_METADATA, REPLAY_CONTEXT_FLAG_SESSION_ID, REPLAY_CONTEXT_FLAG_TURN_INDEX,
    REQUEST_FACT_FLAG_ADMISSION, REQUEST_FACT_FLAG_LATENCIES, REQUEST_FACT_FLAG_OUTPUT_LENGTH,
    REQUEST_FLAG_ARRIVAL_TIMESTAMP, REQUEST_FLAG_OUTPUT_TOKEN_IDS, REQUEST_FLAG_POLICY_CLASS,
    REQUEST_FLAG_PREFERRED_DP_RANK, REQUEST_FLAG_PREFERRED_PREFILL_DP_RANK,
    REQUEST_FLAG_REPLAY_CONTEXT, REQUEST_FLAG_UUID, ReplayHandleV1, ReplayStateV1,
    RequestFactSliceV1, RequestFactV1, RequestIdMutSliceV1, RequestIdSliceV1, RequestIdV1,
    SLA_FLAG_E2E, SLA_FLAG_ITL, SLA_FLAG_TTFT, SlaThresholdsV1, StatusV1, StepRequestV1,
    StepResultV1, U32SliceV1,
};
use aisimulate_core::replay::loadgen::{
    CompactDirectRequest, CompactHashIdsLease, DynPlacement, SteppableAgg, SteppableReplay,
};
use aisimulate_core::replay::{
    DirectRequest, PlacementBatchError, ReplayEngineConfig, ReplayEngineFactory,
    ReplayPromptTokenSource, ReplayRequestContext, ReplayTerminalStatus, SlaThresholds,
};
use dynamo_mocker::placement::{
    KvReplayMetadata, KvRouterConfig, KvRouterPlacement, MockEngineArgs, MockEngineArgsBuilder,
    RouterEventObservation,
};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use std::ffi::c_char;
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use uuid::Uuid;

#[cfg(test)]
static FORCE_FFI_PANIC: AtomicBool = AtomicBool::new(false);

#[cfg(test)]
static FORCE_BATCH_PANIC_AFTER_MUTATION: AtomicBool = AtomicBool::new(false);

#[cfg(test)]
fn panic_when_test_requested() {
    if FORCE_FFI_PANIC.swap(false, Ordering::SeqCst) {
        panic!("test-only FFI boundary panic");
    }
}

#[cfg(not(test))]
fn panic_when_test_requested() {}

#[cfg(test)]
fn panic_after_batch_mutation_when_test_requested() {
    if FORCE_BATCH_PANIC_AFTER_MUTATION.swap(false, Ordering::SeqCst) {
        panic!("test-only panic after batch replay mutation");
    }
}

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

#[derive(Debug, Clone, Copy)]
struct RegisteredHashBuffer {
    data: *const u32,
    len: usize,
}

#[derive(Debug)]
struct HostHashIdsLease {
    data: *const u32,
    len: usize,
    buffer_id: HashBufferIdV1,
    callbacks: HashBufferLeaseCallbacksV1,
    accepted: AtomicBool,
}

impl CompactHashIdsLease for HostHashIdsLease {
    fn hash_ids(&self) -> &[u32] {
        let Some(len) = checked_slice_len::<u32>(self.len as u64) else {
            return &[];
        };
        // Safety: registration validated this non-null, aligned buffer and the
        // host keeps it immutable and live until `Drop` invokes its release
        // callback. The selected range was bounds-checked before construction.
        unsafe { std::slice::from_raw_parts(self.data, len) }
    }

    fn on_accepted(&self) {
        self.accepted.store(true, Ordering::Release);
    }
}

impl Drop for HostHashIdsLease {
    fn drop(&mut self) {
        if !self.accepted.swap(false, Ordering::AcqRel) {
            return;
        }
        release_hash_buffer(self.callbacks, self.buffer_id);
    }
}

fn release_hash_buffer(callbacks: HashBufferLeaseCallbacksV1, buffer_id: HashBufferIdV1) {
    let release = callbacks
        .release_hash_buffer
        .expect("validated compact hash-buffer release callback");
    // Safety: callback validity is established by lease-aware creation. The
    // host contract requires this synchronous callback not to panic or re-enter
    // the provider.
    unsafe { release(callbacks.context, buffer_id) };
}

struct BackendReplay {
    engine: Box<dyn SteppableReplay>,
    poisoned: bool,
    /// Explicit UUIDs accepted in the current report epoch. Keeping this in
    /// lock-step with the engine lets `submit_batch` reject duplicate input
    /// before it mutates the first request in the batch.
    submitted_ids: HashSet<Uuid>,
    lease_callbacks: Option<HashBufferLeaseCallbacksV1>,
    registered_hash_buffers: HashMap<HashBufferIdV1, RegisteredHashBuffer>,
    next_hash_buffer_id: u64,
    last_error: String,
}

fn build_engine(config: &BackendConfig) -> anyhow::Result<Box<dyn SteppableReplay>> {
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
    let engine =
        SteppableAgg::<RouterPlacement, RouterEventObservation, KvReplayMetadata>::with_placement(
            config.engine.clone(),
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
                    Some(KvRouterConfig {
                        router_queue_threshold: Some(0.0),
                        ..KvRouterConfig::default()
                    }),
                    None,
                    topology.len(),
                    Some(selector_seed),
                )?;
                Ok(Box::new(placement) as RouterPlacement)
            },
        )?;
    Ok(Box::new(engine) as Box<dyn SteppableReplay>)
}

fn allocated_bytes(value: String) -> ByteSliceV1 {
    let bytes = value.into_bytes().into_boxed_slice();
    let len = bytes.len() as u64;
    let data = Box::into_raw(bytes).cast::<u8>();
    ByteSliceV1 { data, len }
}

unsafe fn backend_mut_even_if_poisoned(
    handle: ReplayHandleV1,
) -> Result<&'static mut BackendReplay, StatusV1> {
    if handle.0.is_null() {
        return Err(StatusV1::INVALID_ARGUMENT);
    }
    // Safety: every non-null handle comes from `create`, and ownership stays
    // with the caller until `destroy` consumes it.
    Ok(unsafe { &mut *handle.0.cast::<BackendReplay>() })
}

unsafe fn backend_mut(handle: ReplayHandleV1) -> Result<&'static mut BackendReplay, StatusV1> {
    let replay = unsafe { backend_mut_even_if_poisoned(handle) }?;
    if replay.poisoned {
        return Err(StatusV1::INTERNAL);
    }
    Ok(replay)
}

unsafe fn borrowed_tokens(slice: U32SliceV1) -> Result<&'static [u32], StatusV1> {
    let Some(len) = checked_slice_len::<u32>(slice.len) else {
        return Err(StatusV1::INVALID_ARGUMENT);
    };
    if (slice.data.is_null() && len != 0)
        || (!slice.data.is_null()
            && !(slice.data as usize).is_multiple_of(std::mem::align_of::<u32>()))
    {
        return Err(StatusV1::INVALID_ARGUMENT);
    }
    if len == 0 {
        return Ok(&[]);
    }
    // Safety: a non-empty ABI input slice is borrowed and valid for the call.
    Ok(unsafe { std::slice::from_raw_parts(slice.data, len) })
}

fn checked_slice_len<T>(len: u64) -> Option<usize> {
    usize::try_from(len)
        .ok()
        .filter(|&len| len <= isize::MAX as usize / std::mem::size_of::<T>())
}

fn batch_error_status(error: &PlacementBatchError) -> StatusV1 {
    if error.is_poisoned() {
        StatusV1::INTERNAL
    } else {
        StatusV1::REJECTED
    }
}

unsafe fn borrowed_bytes(slice: ByteSliceV1) -> Result<&'static [u8], StatusV1> {
    let Some(len) = checked_slice_len::<u8>(slice.len) else {
        return Err(StatusV1::INVALID_ARGUMENT);
    };
    if slice.data.is_null() && len != 0 {
        return Err(StatusV1::INVALID_ARGUMENT);
    }
    if len == 0 {
        return Ok(&[]);
    }
    Ok(unsafe { std::slice::from_raw_parts(slice.data, len) })
}

unsafe fn utf8(slice: ByteSliceV1) -> Result<String, StatusV1> {
    String::from_utf8(unsafe { borrowed_bytes(slice) }?.to_vec())
        .map_err(|_| StatusV1::INVALID_ARGUMENT)
}

unsafe fn replay_context(
    request: DirectRequestV1,
) -> Result<Option<ReplayRequestContext>, StatusV1> {
    if request.flags & REQUEST_FLAG_REPLAY_CONTEXT == 0 {
        return Ok(None);
    }
    let context = request.replay_context;
    if context.struct_size as usize != std::mem::size_of_val(&context) {
        return Err(StatusV1::INVALID_ARGUMENT);
    }
    let authored_id = unsafe { utf8(context.authored_id) }?;
    if authored_id.is_empty() {
        return Err(StatusV1::INVALID_ARGUMENT);
    }
    let session_id = (context.flags & REPLAY_CONTEXT_FLAG_SESSION_ID != 0)
        .then(|| unsafe { utf8(context.session_id) })
        .transpose()?;
    let metadata = if context.flags & REPLAY_CONTEXT_FLAG_METADATA != 0 {
        // `ReplayContextV1::metadata` is opaque application-owned bytes, not
        // JSON. AISimulate retains context metadata as `serde_json::Value`, so
        // lower each byte to its exact unsigned value rather than interpreting
        // or rejecting caller data. An array cannot collide with the core's
        // object-only routing controls and round-trips every byte, including
        // invalid UTF-8.
        let metadata = unsafe { borrowed_bytes(context.metadata) }?;
        if metadata.len() > MAX_REPLAY_CONTEXT_METADATA_BYTES_V1 {
            return Err(StatusV1::INVALID_ARGUMENT);
        }
        serde_json::Value::Array(
            metadata
                .iter()
                .copied()
                .map(serde_json::Value::from)
                .collect(),
        )
    } else {
        serde_json::Value::Null
    };
    let prompt_token_source = match context.prompt_token_source {
        0 => ReplayPromptTokenSource::Materialized,
        1 => ReplayPromptTokenSource::LengthOnlySynthetic,
        _ => return Err(StatusV1::INVALID_ARGUMENT),
    };
    Ok(Some(ReplayRequestContext {
        authored_id,
        session_id,
        turn_index: (context.flags & REPLAY_CONTEXT_FLAG_TURN_INDEX != 0)
            .then_some(context.turn_index as usize),
        metadata,
        prompt_token_source,
    }))
}

unsafe fn direct_request(request: DirectRequestV1) -> Result<DirectRequest, StatusV1> {
    if request.struct_size as usize != std::mem::size_of::<DirectRequestV1>()
        || request.max_output_tokens > usize::MAX as u64
    {
        return Err(StatusV1::INVALID_ARGUMENT);
    }
    // Safety: input slices are caller-owned and valid for this FFI call.
    let tokens = unsafe { borrowed_tokens(request.tokens) }?.to_vec();
    let output_token_ids = (request.flags & REQUEST_FLAG_OUTPUT_TOKEN_IDS != 0)
        .then(|| unsafe { borrowed_tokens(request.output_token_ids) }.map(ToOwned::to_owned))
        .transpose()?;
    let policy_class = (request.flags & REQUEST_FLAG_POLICY_CLASS != 0)
        .then(|| unsafe { utf8(request.policy_class) })
        .transpose()?;
    if request.flags & REQUEST_FLAG_ARRIVAL_TIMESTAMP != 0
        && !request.arrival_timestamp_ms.is_finite()
    {
        return Err(StatusV1::INVALID_ARGUMENT);
    }
    Ok(DirectRequest {
        tokens,
        max_output_tokens: request.max_output_tokens as usize,
        output_token_ids,
        // Always give the core a concrete ID. Besides making an ABI submit's
        // returned ID stable for rollback, this lets a batch preflight replay
        // exactly the same KV-router key space as the live engine.
        uuid: (request.flags & REQUEST_FLAG_UUID != 0)
            .then_some(Uuid::from_bytes(request.uuid))
            .or_else(|| Some(Uuid::new_v4())),
        dp_rank: request.dp_rank,
        preferred_dp_rank: (request.flags & REQUEST_FLAG_PREFERRED_DP_RANK != 0)
            .then_some(request.preferred_dp_rank),
        preferred_prefill_dp_rank: (request.flags & REQUEST_FLAG_PREFERRED_PREFILL_DP_RANK != 0)
            .then_some(request.preferred_prefill_dp_rank),
        arrival_timestamp_ms: (request.flags & REQUEST_FLAG_ARRIVAL_TIMESTAMP != 0)
            .then_some(request.arrival_timestamp_ms),
        priority: request.priority,
        strict_priority: request.strict_priority,
        policy_class,
        replay_context: unsafe { replay_context(request) }?,
    })
}

unsafe fn compact_request_parts(
    request: CompactRequestV1,
) -> Result<(DirectRequest, usize, usize), StatusV1> {
    if request.struct_size as usize != std::mem::size_of::<CompactRequestV1>()
        || request.flags != 0
        || request.reserved != 0
        || request.input_token_count > usize::MAX as u64
        || request.trace_block_size == 0
    {
        return Err(StatusV1::INVALID_ARGUMENT);
    }
    let direct = unsafe { direct_request(request.request) }?;
    if !direct.tokens.is_empty() {
        return Err(StatusV1::INVALID_ARGUMENT);
    }
    Ok((
        direct,
        request.input_token_count as usize,
        request.trace_block_size as usize,
    ))
}

unsafe fn owned_compact_request(
    request: CompactRequestV1,
) -> Result<CompactDirectRequest, StatusV1> {
    let (direct, input_token_count, trace_block_size) = unsafe { compact_request_parts(request) }?;
    let hash_ids = unsafe { borrowed_tokens(request.hash_ids) }?.to_vec();
    Ok(CompactDirectRequest::owned(
        direct,
        input_token_count,
        trace_block_size,
        hash_ids,
    ))
}

unsafe fn create_impl(
    request: CreateRequestV1,
    lease_callbacks: Option<HashBufferLeaseCallbacksV1>,
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
    let Some(payload_len) = checked_slice_len::<u8>(request.provider_payload.len) else {
        return StatusV1::INVALID_ARGUMENT;
    };
    if request.provider_payload.len > MAX_PROVIDER_PAYLOAD_BYTES
        || (request.provider_payload.data.is_null() && payload_len != 0)
    {
        return StatusV1::INVALID_ARGUMENT;
    }
    // Safety: the ABI guarantees borrowed input bytes remain valid for this
    // call; null is accepted only for an empty slice as checked above.
    let payload = if payload_len == 0 {
        &[]
    } else {
        // Safety: a non-empty ABI input slice is valid for this call.
        unsafe {
            std::slice::from_raw_parts(request.provider_payload.data.cast::<u8>(), payload_len)
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
    let created = build_engine(&config);
    match created {
        Ok(engine) => {
            let replay = Box::new(BackendReplay {
                engine,
                poisoned: false,
                submitted_ids: HashSet::new(),
                lease_callbacks,
                registered_hash_buffers: HashMap::new(),
                next_hash_buffer_id: 1,
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

unsafe fn submit_impl(
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
    match replay.engine.submit(request.clone()) {
        Ok(uuid) => {
            replay.submitted_ids.insert(uuid);
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

unsafe fn submit_compact_impl(
    handle: ReplayHandleV1,
    request: CompactRequestV1,
    request_id: *mut RequestIdV1,
) -> StatusV1 {
    if request_id.is_null() {
        return StatusV1::INVALID_ARGUMENT;
    }
    let request = match unsafe { owned_compact_request(request) } {
        Ok(request) => request,
        Err(status) => return status,
    };
    let replay = match unsafe { backend_mut(handle) } {
        Ok(replay) => replay,
        Err(status) => return status,
    };
    match replay.engine.submit_compact(request) {
        Ok(uuid) => {
            replay.submitted_ids.insert(uuid);
            unsafe { *request_id = *uuid.as_bytes() };
            StatusV1::OK
        }
        Err(error) => {
            replay.last_error = error.to_string();
            StatusV1::REJECTED
        }
    }
}

unsafe fn register_hash_buffer_impl(
    handle: ReplayHandleV1,
    hash_ids: U32SliceV1,
    buffer_id: *mut HashBufferIdV1,
) -> StatusV1 {
    let Some(hash_id_count) = checked_slice_len::<u32>(hash_ids.len) else {
        return StatusV1::INVALID_ARGUMENT;
    };
    if buffer_id.is_null()
        || hash_ids.len == 0
        || hash_ids.data.is_null()
        || !(hash_ids.data as usize).is_multiple_of(std::mem::align_of::<u32>())
    {
        return StatusV1::INVALID_ARGUMENT;
    }
    let replay = match unsafe { backend_mut(handle) } {
        Ok(replay) => replay,
        Err(status) => return status,
    };
    if replay.lease_callbacks.is_none() {
        return StatusV1::UNSUPPORTED;
    }
    if replay.next_hash_buffer_id == HashBufferIdV1::INVALID.0 {
        replay.last_error = "compact hash-buffer identifiers are exhausted".to_owned();
        return StatusV1::INTERNAL;
    }
    let id = HashBufferIdV1(replay.next_hash_buffer_id);
    replay.next_hash_buffer_id = replay.next_hash_buffer_id.wrapping_add(1);
    let replaced = replay.registered_hash_buffers.insert(
        id,
        RegisteredHashBuffer {
            data: hash_ids.data,
            len: hash_id_count,
        },
    );
    if replaced.is_some() {
        replay.last_error = "compact hash-buffer identifier was reused while live".to_owned();
        return StatusV1::INTERNAL;
    }
    // Safety: validated non-null output pointer.
    unsafe { *buffer_id = id };
    StatusV1::OK
}

unsafe fn submit_compact_hash_buffer_range_impl(
    handle: ReplayHandleV1,
    request: CompactRequestV1,
    range: HashBufferRangeV1,
    request_id: *mut RequestIdV1,
) -> StatusV1 {
    if request_id.is_null()
        || !range.is_valid()
        || range.offset > usize::MAX as u64
        || range.len > usize::MAX as u64
        || request.hash_ids.len != 0
        || !request.hash_ids.data.is_null()
    {
        return StatusV1::INVALID_ARGUMENT;
    }
    let (direct, input_token_count, trace_block_size) =
        match unsafe { compact_request_parts(request) } {
            Ok(parts) => parts,
            Err(status) => return status,
        };
    let expected_hash_ids = input_token_count.div_ceil(trace_block_size);
    if range.len as usize != expected_hash_ids {
        return StatusV1::INVALID_ARGUMENT;
    }
    let replay = match unsafe { backend_mut(handle) } {
        Ok(replay) => replay,
        Err(status) => return status,
    };
    let Some(callbacks) = replay.lease_callbacks else {
        return StatusV1::UNSUPPORTED;
    };
    let Some(buffer) = replay
        .registered_hash_buffers
        .get(&range.buffer_id)
        .copied()
    else {
        return StatusV1::INVALID_ARGUMENT;
    };
    let start = range.offset as usize;
    let Some(end) = start.checked_add(range.len as usize) else {
        return StatusV1::INVALID_ARGUMENT;
    };
    if end > buffer.len {
        return StatusV1::INVALID_ARGUMENT;
    }
    let data = if range.len == 0 {
        std::ptr::null()
    } else {
        // Safety: the validated non-empty registered buffer covers `start`.
        unsafe { buffer.data.add(start) }
    };
    // The core lease API uses `Arc` for local lifecycle ownership. This host
    // buffer pointer deliberately does not cross threads, so it must not gain
    // an unsound `Send` or `Sync` implementation merely to satisfy Clippy.
    #[allow(clippy::arc_with_non_send_sync)]
    let lease: Arc<dyn CompactHashIdsLease> = Arc::new(HostHashIdsLease {
        data,
        len: range.len as usize,
        buffer_id: range.buffer_id,
        callbacks,
        accepted: AtomicBool::new(false),
    });
    let request = CompactDirectRequest::leased(direct, input_token_count, trace_block_size, lease);
    match replay.engine.submit_compact(request) {
        Ok(uuid) => {
            let removed = replay.registered_hash_buffers.remove(&range.buffer_id);
            debug_assert!(
                removed.is_some(),
                "validated registered compact hash buffer disappeared"
            );
            replay.submitted_ids.insert(uuid);
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

unsafe fn submit_batch_impl(
    handle: ReplayHandleV1,
    requests: DirectRequestSliceV1,
    request_ids: RequestIdMutSliceV1,
) -> StatusV1 {
    let Some(request_count) = checked_slice_len::<DirectRequestV1>(requests.len) else {
        return StatusV1::INVALID_ARGUMENT;
    };
    let Some(request_id_count) = checked_slice_len::<RequestIdV1>(request_ids.len) else {
        return StatusV1::INVALID_ARGUMENT;
    };
    if request_ids.len != requests.len
        || (requests.data.is_null() && requests.len != 0)
        || (requests.len != 0
            && !(requests.data as usize).is_multiple_of(std::mem::align_of::<DirectRequestV1>()))
        || (request_ids.data.is_null() && request_ids.len != 0)
    {
        return StatusV1::INVALID_ARGUMENT;
    }
    let requests = if requests.len == 0 {
        &[]
    } else {
        // Safety: non-empty input batch is valid for this FFI call.
        unsafe { std::slice::from_raw_parts(requests.data, request_count) }
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
        unsafe { std::slice::from_raw_parts_mut(request_ids.data, request_id_count) }
    };
    output.fill([0; 16]);
    let replay = match unsafe { backend_mut(handle) } {
        Ok(replay) => replay,
        Err(status) => return status,
    };
    let mut batch_ids = HashSet::with_capacity(converted.len());
    for request in &converted {
        if let Some(uuid) = request.uuid
            && (!batch_ids.insert(uuid) || replay.submitted_ids.contains(&uuid))
        {
            replay.last_error = format!("steppable replay request {uuid} is already retained");
            return StatusV1::REJECTED;
        }
    }
    match replay.engine.submit_batch(converted) {
        Ok(uuids) => {
            #[cfg(test)]
            panic_after_batch_mutation_when_test_requested();
            debug_assert_eq!(uuids.len(), output.len());
            for (uuid, request_id) in uuids.into_iter().zip(output.iter_mut()) {
                replay.submitted_ids.insert(uuid);
                *request_id = *uuid.as_bytes();
            }
            StatusV1::OK
        }
        Err(error) => {
            replay.last_error = error.to_string();
            batch_error_status(&error)
        }
    }
}

unsafe fn cancel_impl(
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

unsafe fn cancel_batch_impl(
    handle: ReplayHandleV1,
    request_ids: RequestIdSliceV1,
    events: *mut EngineEventSliceV1,
) -> StatusV1 {
    let Some(request_id_count) = checked_slice_len::<RequestIdV1>(request_ids.len) else {
        return StatusV1::INVALID_ARGUMENT;
    };
    if events.is_null() || (request_ids.data.is_null() && request_ids.len != 0) {
        return StatusV1::INVALID_ARGUMENT;
    }
    let request_ids = if request_ids.len == 0 {
        &[]
    } else {
        // Safety: non-empty input batch is valid for this FFI call.
        unsafe { std::slice::from_raw_parts(request_ids.data, request_id_count) }
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

unsafe fn step_impl(
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

unsafe fn take_report_impl(
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
    replay.submitted_ids.clear();
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

unsafe fn release_bytes_impl(bytes: ByteSliceV1) {
    if bytes.data.is_null() {
        return;
    }
    let Some(len) = checked_slice_len::<u8>(bytes.len) else {
        return;
    };
    // Safety: every non-empty output byte slice is allocated by
    // `allocated_bytes` as an exact-length boxed slice.
    unsafe {
        drop(Box::from_raw(std::ptr::slice_from_raw_parts_mut(
            bytes.data.cast_mut(),
            len,
        )));
    }
}
unsafe fn release_events_impl(events: EngineEventSliceV1) {
    if events.data.is_null() {
        return;
    }
    let Some(len) = checked_slice_len::<EngineEventV1>(events.len) else {
        return;
    };
    // Safety: `step` allocates exact-length boxed slices and transfers one
    // release obligation to the host.
    unsafe {
        drop(Box::from_raw(std::ptr::slice_from_raw_parts_mut(
            events.data.cast_mut(),
            len,
        )));
    }
}
unsafe fn release_request_facts_impl(facts: RequestFactSliceV1) {
    if facts.data.is_null() {
        return;
    }
    let Some(len) = checked_slice_len::<RequestFactV1>(facts.len) else {
        return;
    };
    // Safety: `step` allocates exact-length boxed slices and transfers one
    // release obligation to the host.
    unsafe {
        drop(Box::from_raw(std::ptr::slice_from_raw_parts_mut(
            facts.data.cast_mut(),
            len,
        )));
    }
}

unsafe fn state_impl(handle: ReplayHandleV1, state: *mut ReplayStateV1) -> StatusV1 {
    panic_when_test_requested();
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

unsafe fn advance_now_ms_impl(handle: ReplayHandleV1, now_ms: f64) -> StatusV1 {
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

unsafe fn set_capture_per_request_impl(handle: ReplayHandleV1, capture: u8) -> StatusV1 {
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

unsafe fn set_sla_thresholds_impl(handle: ReplayHandleV1, thresholds: SlaThresholdsV1) -> StatusV1 {
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

unsafe fn last_error_impl(handle: ReplayHandleV1, error: *mut ByteSliceV1) -> StatusV1 {
    if error.is_null() {
        return StatusV1::INVALID_ARGUMENT;
    }
    let replay = match unsafe { backend_mut_even_if_poisoned(handle) } {
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

unsafe fn destroy_impl(handle: ReplayHandleV1) {
    if handle.0.is_null() {
        return;
    }
    // Safety: caller transfers the unique handle returned by `create`.
    unsafe { drop(Box::from_raw(handle.0.cast::<BackendReplay>())) };
}

fn panic_status(handle: ReplayHandleV1) -> StatusV1 {
    // Error recording is best-effort: the only hard FFI guarantee here is
    // containment. Keep an error in a valid replay when possible, but never
    // risk a second unwind while handling the first one.
    let _ = catch_unwind(AssertUnwindSafe(|| {
        if let Ok(replay) = unsafe { backend_mut_even_if_poisoned(handle) } {
            replay.last_error = "panic contained at Dynamo steppable ABI boundary".to_owned();
        }
    }));
    StatusV1::INTERNAL
}

fn batch_panic_status(handle: ReplayHandleV1) -> StatusV1 {
    // A batch unwind can follow a stateful placement or replay mutation. The
    // outer provider therefore owns a fail-stop latch independent of whether
    // the inner replay had an opportunity to return its poisoned error type.
    let _ = catch_unwind(AssertUnwindSafe(|| {
        if let Ok(replay) = unsafe { backend_mut_even_if_poisoned(handle) } {
            replay.poisoned = true;
            replay.last_error =
                "panic contained after Dynamo steppable batch mutation; replay is poisoned"
                    .to_owned();
        }
    }));
    StatusV1::INTERNAL
}

fn catch_status(operation: impl FnOnce() -> StatusV1, handle: ReplayHandleV1) -> StatusV1 {
    match catch_unwind(AssertUnwindSafe(operation)) {
        Ok(status) => status,
        Err(_) => panic_status(handle),
    }
}

unsafe extern "C" fn create(
    request: CreateRequestV1,
    handle: *mut ReplayHandleV1,
    error: *mut ByteSliceV1,
) -> StatusV1 {
    if !handle.is_null() {
        unsafe { *handle = ReplayHandleV1(std::ptr::null_mut()) };
    }
    if !error.is_null() {
        unsafe { *error = ByteSliceV1::EMPTY };
    }
    match catch_unwind(AssertUnwindSafe(|| unsafe {
        create_impl(request, None, handle, error)
    })) {
        Ok(status) => status,
        Err(_) => StatusV1::INTERNAL,
    }
}

unsafe extern "C" fn create_with_hash_buffer_leases(
    request: CreateRequestV1,
    callbacks: HashBufferLeaseCallbacksV1,
    handle: *mut ReplayHandleV1,
    error: *mut ByteSliceV1,
) -> StatusV1 {
    if !handle.is_null() {
        unsafe { *handle = ReplayHandleV1(std::ptr::null_mut()) };
    }
    if !error.is_null() {
        unsafe { *error = ByteSliceV1::EMPTY };
    }
    if !callbacks.has_release_callback() {
        return StatusV1::INVALID_ARGUMENT;
    }
    match catch_unwind(AssertUnwindSafe(|| unsafe {
        create_impl(request, Some(callbacks), handle, error)
    })) {
        Ok(status) => status,
        Err(_) => StatusV1::INTERNAL,
    }
}

unsafe extern "C" fn submit(
    handle: ReplayHandleV1,
    request: DirectRequestV1,
    request_id: *mut RequestIdV1,
) -> StatusV1 {
    if !request_id.is_null() {
        unsafe { *request_id = [0; 16] };
    }
    catch_status(
        || unsafe { submit_impl(handle, request, request_id) },
        handle,
    )
}

unsafe extern "C" fn submit_compact(
    handle: ReplayHandleV1,
    request: CompactRequestV1,
    request_id: *mut RequestIdV1,
) -> StatusV1 {
    if !request_id.is_null() {
        unsafe { *request_id = [0; 16] };
    }
    catch_status(
        || unsafe { submit_compact_impl(handle, request, request_id) },
        handle,
    )
}

unsafe extern "C" fn register_hash_buffer(
    handle: ReplayHandleV1,
    hash_ids: U32SliceV1,
    buffer_id: *mut HashBufferIdV1,
) -> StatusV1 {
    if !buffer_id.is_null() {
        unsafe { *buffer_id = HashBufferIdV1::INVALID };
    }
    catch_status(
        || unsafe { register_hash_buffer_impl(handle, hash_ids, buffer_id) },
        handle,
    )
}

unsafe extern "C" fn submit_compact_hash_buffer_range(
    handle: ReplayHandleV1,
    request: CompactRequestV1,
    range: HashBufferRangeV1,
    request_id: *mut RequestIdV1,
) -> StatusV1 {
    if !request_id.is_null() {
        unsafe { *request_id = [0; 16] };
    }
    catch_status(
        || unsafe { submit_compact_hash_buffer_range_impl(handle, request, range, request_id) },
        handle,
    )
}

unsafe extern "C" fn submit_batch(
    handle: ReplayHandleV1,
    requests: DirectRequestSliceV1,
    request_ids: RequestIdMutSliceV1,
) -> StatusV1 {
    if checked_slice_len::<DirectRequestV1>(requests.len).is_none() {
        return StatusV1::INVALID_ARGUMENT;
    }
    if request_ids.len <= usize::MAX as u64 && (!request_ids.data.is_null() || request_ids.len == 0)
    {
        let Some(len) = checked_slice_len::<RequestIdV1>(request_ids.len) else {
            return StatusV1::INVALID_ARGUMENT;
        };
        let output = if request_ids.len == 0 {
            &mut []
        } else {
            unsafe { std::slice::from_raw_parts_mut(request_ids.data, len) }
        };
        output.fill([0; 16]);
    }
    match catch_unwind(AssertUnwindSafe(|| unsafe {
        submit_batch_impl(handle, requests, request_ids)
    })) {
        Ok(status) => status,
        Err(_) => batch_panic_status(handle),
    }
}

unsafe extern "C" fn cancel(
    handle: ReplayHandleV1,
    request_id: *const RequestIdV1,
    event: *mut EngineEventV1,
    canceled: *mut u8,
) -> StatusV1 {
    if !event.is_null() {
        unsafe { *event = std::mem::zeroed() };
    }
    if !canceled.is_null() {
        unsafe { *canceled = 0 };
    }
    catch_status(
        || unsafe { cancel_impl(handle, request_id, event, canceled) },
        handle,
    )
}

unsafe extern "C" fn cancel_batch(
    handle: ReplayHandleV1,
    request_ids: RequestIdSliceV1,
    events: *mut EngineEventSliceV1,
) -> StatusV1 {
    if !events.is_null() {
        unsafe {
            *events = EngineEventSliceV1 {
                data: std::ptr::null(),
                len: 0,
            };
        }
    }
    catch_status(
        || unsafe { cancel_batch_impl(handle, request_ids, events) },
        handle,
    )
}

unsafe extern "C" fn step(
    handle: ReplayHandleV1,
    request: StepRequestV1,
    result: *mut StepResultV1,
) -> StatusV1 {
    if !result.is_null() {
        unsafe { *result = StepResultV1::EMPTY };
    }
    catch_status(|| unsafe { step_impl(handle, request, result) }, handle)
}

unsafe extern "C" fn take_report(
    handle: ReplayHandleV1,
    wall_ms: f64,
    report: *mut ByteSliceV1,
) -> StatusV1 {
    if !report.is_null() {
        unsafe { *report = ByteSliceV1::EMPTY };
    }
    catch_status(
        || unsafe { take_report_impl(handle, wall_ms, report) },
        handle,
    )
}

unsafe extern "C" fn release_bytes(bytes: ByteSliceV1) {
    let _ = catch_unwind(AssertUnwindSafe(|| unsafe { release_bytes_impl(bytes) }));
}

unsafe extern "C" fn release_events(events: EngineEventSliceV1) {
    let _ = catch_unwind(AssertUnwindSafe(|| unsafe { release_events_impl(events) }));
}

unsafe extern "C" fn release_request_facts(facts: RequestFactSliceV1) {
    let _ = catch_unwind(AssertUnwindSafe(|| unsafe {
        release_request_facts_impl(facts)
    }));
}

unsafe extern "C" fn state(handle: ReplayHandleV1, state: *mut ReplayStateV1) -> StatusV1 {
    if !state.is_null() {
        unsafe { *state = ReplayStateV1::EMPTY };
    }
    catch_status(|| unsafe { state_impl(handle, state) }, handle)
}

unsafe extern "C" fn advance_now_ms(handle: ReplayHandleV1, now_ms: f64) -> StatusV1 {
    catch_status(|| unsafe { advance_now_ms_impl(handle, now_ms) }, handle)
}

unsafe extern "C" fn set_capture_per_request(handle: ReplayHandleV1, capture: u8) -> StatusV1 {
    catch_status(
        || unsafe { set_capture_per_request_impl(handle, capture) },
        handle,
    )
}

unsafe extern "C" fn set_sla_thresholds(
    handle: ReplayHandleV1,
    thresholds: SlaThresholdsV1,
) -> StatusV1 {
    catch_status(
        || unsafe { set_sla_thresholds_impl(handle, thresholds) },
        handle,
    )
}

unsafe extern "C" fn last_error(handle: ReplayHandleV1, error: *mut ByteSliceV1) -> StatusV1 {
    if !error.is_null() {
        unsafe { *error = ByteSliceV1::EMPTY };
    }
    catch_status(|| unsafe { last_error_impl(handle, error) }, handle)
}

unsafe extern "C" fn destroy(handle: ReplayHandleV1) {
    let _ = catch_unwind(AssertUnwindSafe(|| unsafe { destroy_impl(handle) }));
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
    submit_compact: Some(submit_compact),
    create_with_hash_buffer_leases: Some(create_with_hash_buffer_leases),
    register_hash_buffer: Some(register_hash_buffer),
    submit_compact_hash_buffer_range: Some(submit_compact_hash_buffer_range),
};

static DESCRIPTOR: PluginDescriptorV1 = PluginDescriptorV1 {
    abi_major: 1,
    abi_minor: 0,
    struct_size: std::mem::size_of::<PluginDescriptorV1>() as u32,
    flags: 0,
    capabilities: CAPABILITY_COMPACT_REQUEST_V1 | CAPABILITY_COMPACT_BUFFER_LEASES_V1,
    provider_id: PROVIDER_ID.as_ptr().cast::<c_char>(),
    vtable: &VTABLE,
};

/// Returns the static V1 plugin descriptor for dynamic loading.
#[unsafe(no_mangle)]
pub extern "C" fn aiperf_steppable_plugin_v1() -> *const PluginDescriptorV1 {
    catch_unwind(AssertUnwindSafe(|| {
        &DESCRIPTOR as *const PluginDescriptorV1
    }))
    .unwrap_or(std::ptr::null())
}

#[cfg(test)]
mod tests {
    use super::*;
    use aiperf_steppable_abi::{
        REPLAY_CONTEXT_FLAG_METADATA, REPLAY_CONTEXT_FLAG_SESSION_ID,
        REPLAY_CONTEXT_FLAG_TURN_INDEX, REQUEST_FLAG_ARRIVAL_TIMESTAMP,
        REQUEST_FLAG_OUTPUT_TOKEN_IDS, REQUEST_FLAG_POLICY_CLASS, REQUEST_FLAG_PREFERRED_DP_RANK,
        REQUEST_FLAG_PREFERRED_PREFILL_DP_RANK, REQUEST_FLAG_REPLAY_CONTEXT,
    };

    #[test]
    fn poisoned_batch_errors_map_to_internal_without_changing_rejections() {
        assert_eq!(
            batch_error_status(&aisimulate_core::replay::PlacementBatchError::poisoned(
                anyhow::anyhow!("poisoned"),
            )),
            StatusV1::INTERNAL
        );
        assert_eq!(
            batch_error_status(&aisimulate_core::replay::PlacementBatchError::unchanged(
                anyhow::anyhow!("rejected"),
            )),
            StatusV1::REJECTED
        );
    }

    #[test]
    fn checked_slice_len_rejects_lengths_that_exceed_isize_byte_bound() {
        let u32_limit = (isize::MAX as usize / std::mem::size_of::<u32>()) as u64;
        let request_limit = (isize::MAX as usize / std::mem::size_of::<DirectRequestV1>()) as u64;
        let request_id_limit = (isize::MAX as usize / std::mem::size_of::<RequestIdV1>()) as u64;

        assert_eq!(
            checked_slice_len::<u32>(u32_limit),
            Some(u32_limit as usize)
        );
        assert_eq!(checked_slice_len::<u32>(u32_limit + 1), None);
        assert_eq!(
            checked_slice_len::<DirectRequestV1>(request_limit),
            Some(request_limit as usize)
        );
        assert_eq!(
            checked_slice_len::<DirectRequestV1>(request_limit + 1),
            None
        );
        assert_eq!(
            checked_slice_len::<RequestIdV1>(request_id_limit),
            Some(request_id_limit as usize)
        );
        assert_eq!(checked_slice_len::<RequestIdV1>(request_id_limit + 1), None);
    }

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

    #[test]
    fn direct_request_preserves_all_present_abi_fields_including_empty_plan() {
        let output = [];
        let policy = b"priority";
        let authored = b"trace-17";
        let session = b"session-2";
        let metadata = br#"{"source":"abi"}"#;
        let request = DirectRequestV1 {
            struct_size: std::mem::size_of::<DirectRequestV1>() as u32,
            flags: REQUEST_FLAG_OUTPUT_TOKEN_IDS
                | REQUEST_FLAG_PREFERRED_DP_RANK
                | REQUEST_FLAG_PREFERRED_PREFILL_DP_RANK
                | REQUEST_FLAG_ARRIVAL_TIMESTAMP
                | REQUEST_FLAG_POLICY_CLASS
                | REQUEST_FLAG_REPLAY_CONTEXT,
            tokens: U32SliceV1::EMPTY,
            output_token_ids: U32SliceV1 {
                data: output.as_ptr(),
                len: 0,
            },
            max_output_tokens: 9,
            uuid: [8; 16],
            dp_rank: 1,
            preferred_dp_rank: 2,
            preferred_prefill_dp_rank: 3,
            arrival_timestamp_ms: 4.5,
            priority: 7,
            strict_priority: 11,
            policy_class: ByteSliceV1 {
                data: policy.as_ptr(),
                len: policy.len() as u64,
            },
            replay_context: aiperf_steppable_abi::ReplayContextV1 {
                struct_size: std::mem::size_of::<aiperf_steppable_abi::ReplayContextV1>() as u32,
                flags: REPLAY_CONTEXT_FLAG_SESSION_ID
                    | REPLAY_CONTEXT_FLAG_TURN_INDEX
                    | REPLAY_CONTEXT_FLAG_METADATA,
                authored_id: ByteSliceV1 {
                    data: authored.as_ptr(),
                    len: authored.len() as u64,
                },
                session_id: ByteSliceV1 {
                    data: session.as_ptr(),
                    len: session.len() as u64,
                },
                metadata: ByteSliceV1 {
                    data: metadata.as_ptr(),
                    len: metadata.len() as u64,
                },
                turn_index: 6,
                prompt_token_source: 0,
                reserved: 0,
            },
        };
        let converted = unsafe { direct_request(request) }.expect("valid ABI request");
        assert_eq!(converted.output_token_ids.as_deref(), Some(&[][..]));
        assert_eq!(converted.preferred_dp_rank, Some(2));
        assert_eq!(converted.preferred_prefill_dp_rank, Some(3));
        assert_eq!(converted.arrival_timestamp_ms, Some(4.5));
        assert_eq!(converted.policy_class.as_deref(), Some("priority"));
        let context = converted.replay_context.expect("context");
        assert_eq!(context.authored_id, "trace-17");
        assert_eq!(context.session_id.as_deref(), Some("session-2"));
        assert_eq!(context.turn_index, Some(6));
        assert_eq!(
            context.metadata,
            serde_json::Value::Array(
                metadata
                    .iter()
                    .copied()
                    .map(serde_json::Value::from)
                    .collect()
            )
        );
    }

    #[test]
    fn replay_context_accepts_and_preserves_non_json_metadata_bytes() {
        let authored = b"opaque-metadata";
        let metadata = [0xff_u8];
        let request = DirectRequestV1 {
            struct_size: std::mem::size_of::<DirectRequestV1>() as u32,
            flags: REQUEST_FLAG_REPLAY_CONTEXT,
            tokens: U32SliceV1::EMPTY,
            output_token_ids: U32SliceV1::EMPTY,
            max_output_tokens: 1,
            uuid: [0; 16],
            dp_rank: 0,
            preferred_dp_rank: 0,
            preferred_prefill_dp_rank: 0,
            arrival_timestamp_ms: 0.0,
            priority: 0,
            strict_priority: 0,
            policy_class: ByteSliceV1::EMPTY,
            replay_context: aiperf_steppable_abi::ReplayContextV1 {
                struct_size: std::mem::size_of::<aiperf_steppable_abi::ReplayContextV1>() as u32,
                flags: REPLAY_CONTEXT_FLAG_METADATA,
                authored_id: ByteSliceV1 {
                    data: authored.as_ptr(),
                    len: authored.len() as u64,
                },
                session_id: ByteSliceV1::EMPTY,
                metadata: ByteSliceV1 {
                    data: metadata.as_ptr(),
                    len: metadata.len() as u64,
                },
                turn_index: 0,
                prompt_token_source: 0,
                reserved: 0,
            },
        };

        let context = unsafe { direct_request(request) }
            .expect("opaque metadata must not be JSON-validated")
            .replay_context
            .expect("context");
        assert_eq!(context.metadata, serde_json::json!([255]));
    }

    #[test]
    fn panic_in_ffi_callback_becomes_internal_with_empty_output() {
        let config = BackendConfig::one_worker();
        let replay = Box::new(BackendReplay {
            engine: build_engine(&config).expect("test engine"),
            poisoned: false,
            submitted_ids: HashSet::new(),
            lease_callbacks: None,
            registered_hash_buffers: HashMap::new(),
            next_hash_buffer_id: 1,
            last_error: String::new(),
        });
        let handle = ReplayHandleV1(Box::into_raw(replay).cast());
        FORCE_FFI_PANIC.store(true, Ordering::SeqCst);
        let mut output = ReplayStateV1 {
            now_ms: 1.0,
            next_event_ms: 1.0,
            in_flight: 1,
            is_idle: 1,
            reserved: [1; 7],
        };

        assert_eq!(unsafe { state(handle, &mut output) }, StatusV1::INTERNAL);
        assert_eq!(output.now_ms, 0.0);
        assert_eq!(output.in_flight, 0);
        assert_eq!(output.is_idle, ReplayStateV1::EMPTY.is_idle);

        unsafe { destroy(handle) };
    }

    #[test]
    fn contained_batch_panic_poison_rejects_every_later_operation() {
        let config = BackendConfig::one_worker();
        let replay = Box::new(BackendReplay {
            engine: build_engine(&config).expect("test engine"),
            poisoned: false,
            submitted_ids: HashSet::new(),
            lease_callbacks: None,
            registered_hash_buffers: HashMap::new(),
            next_hash_buffer_id: 1,
            last_error: String::new(),
        });
        let handle = ReplayHandleV1(Box::into_raw(replay).cast());
        let tokens = [1_u32];
        let direct = DirectRequestV1 {
            struct_size: std::mem::size_of::<DirectRequestV1>() as u32,
            flags: REQUEST_FLAG_UUID,
            tokens: U32SliceV1 {
                data: tokens.as_ptr(),
                len: 1,
            },
            output_token_ids: U32SliceV1::EMPTY,
            max_output_tokens: 1,
            uuid: [61; 16],
            dp_rank: 0,
            preferred_dp_rank: 0,
            preferred_prefill_dp_rank: 0,
            arrival_timestamp_ms: 0.0,
            priority: 0,
            strict_priority: 0,
            policy_class: ByteSliceV1::EMPTY,
            replay_context: aiperf_steppable_abi::ReplayContextV1::EMPTY,
        };
        let requests = [direct];
        let mut request_ids = [[99; 16]];
        FORCE_BATCH_PANIC_AFTER_MUTATION.store(true, Ordering::SeqCst);

        assert_eq!(
            unsafe {
                submit_batch(
                    handle,
                    DirectRequestSliceV1 {
                        data: requests.as_ptr(),
                        len: 1,
                    },
                    RequestIdMutSliceV1 {
                        data: request_ids.as_mut_ptr(),
                        len: 1,
                    },
                )
            },
            StatusV1::INTERNAL
        );
        assert_eq!(request_ids, [[0; 16]]);
        assert_eq!(
            unsafe { backend_mut_even_if_poisoned(handle) }
                .expect("handle remains owned by test")
                .engine
                .in_flight(),
            1,
            "the injected unwind follows a real replay mutation"
        );

        let mut request_id = [0; 16];
        assert_eq!(
            unsafe { submit(handle, direct, &raw mut request_id) },
            StatusV1::INTERNAL
        );
        let compact_hashes = [7_u32];
        let compact = CompactRequestV1 {
            struct_size: std::mem::size_of::<CompactRequestV1>() as u32,
            flags: 0,
            input_token_count: 1,
            trace_block_size: 1,
            reserved: 0,
            hash_ids: U32SliceV1 {
                data: compact_hashes.as_ptr(),
                len: 1,
            },
            request: DirectRequestV1 {
                tokens: U32SliceV1::EMPTY,
                uuid: [62; 16],
                ..direct
            },
        };
        assert_eq!(
            unsafe { submit_compact(handle, compact, &raw mut request_id) },
            StatusV1::INTERNAL
        );
        assert_eq!(
            unsafe {
                submit_batch(
                    handle,
                    DirectRequestSliceV1 {
                        data: requests.as_ptr(),
                        len: 1,
                    },
                    RequestIdMutSliceV1 {
                        data: request_ids.as_mut_ptr(),
                        len: 1,
                    },
                )
            },
            StatusV1::INTERNAL
        );

        let mut event = unsafe { std::mem::zeroed() };
        let mut canceled = 0;
        assert_eq!(
            unsafe { cancel(handle, &direct.uuid, &raw mut event, &raw mut canceled) },
            StatusV1::INTERNAL
        );
        let mut events = EngineEventSliceV1 {
            data: std::ptr::null(),
            len: 0,
        };
        assert_eq!(
            unsafe {
                cancel_batch(
                    handle,
                    RequestIdSliceV1 {
                        data: std::ptr::null(),
                        len: 0,
                    },
                    &raw mut events,
                )
            },
            StatusV1::INTERNAL
        );
        let mut step_result = StepResultV1::EMPTY;
        assert_eq!(
            unsafe {
                step(
                    handle,
                    StepRequestV1 {
                        struct_size: std::mem::size_of::<StepRequestV1>() as u32,
                        flags: 0,
                        until_ms: f64::INFINITY,
                    },
                    &raw mut step_result,
                )
            },
            StatusV1::INTERNAL
        );
        let mut report = ByteSliceV1::EMPTY;
        assert_eq!(
            unsafe { take_report(handle, 1.0, &raw mut report) },
            StatusV1::INTERNAL
        );
        let mut replay_state = ReplayStateV1::EMPTY;
        assert_eq!(
            unsafe { state(handle, &raw mut replay_state) },
            StatusV1::INTERNAL
        );
        assert_eq!(unsafe { advance_now_ms(handle, 1.0) }, StatusV1::INTERNAL);
        assert_eq!(
            unsafe { set_capture_per_request(handle, 1) },
            StatusV1::INTERNAL
        );
        assert_eq!(
            unsafe { set_sla_thresholds(handle, SlaThresholdsV1::EMPTY) },
            StatusV1::INTERNAL
        );

        let mut buffer_id = HashBufferIdV1::INVALID;
        assert_eq!(
            unsafe {
                register_hash_buffer(
                    handle,
                    U32SliceV1 {
                        data: compact_hashes.as_ptr(),
                        len: 1,
                    },
                    &raw mut buffer_id,
                )
            },
            StatusV1::INTERNAL
        );
        let leased_compact = CompactRequestV1 {
            hash_ids: U32SliceV1::EMPTY,
            ..compact
        };
        assert_eq!(
            unsafe {
                submit_compact_hash_buffer_range(
                    handle,
                    leased_compact,
                    HashBufferRangeV1 {
                        buffer_id: HashBufferIdV1(1),
                        offset: 0,
                        len: 1,
                    },
                    &raw mut request_id,
                )
            },
            StatusV1::INTERNAL
        );

        unsafe { destroy(handle) };
    }
}
