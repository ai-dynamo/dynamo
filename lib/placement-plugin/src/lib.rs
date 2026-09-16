// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Dynamo's dynamically loaded KV-router placement provider.
//!
//! This first ABI checkpoint owns a configured [`KvRouterPlacement`] behind
//! an opaque V1 handle. It deliberately accepts only an empty provider-options
//! payload: decoding Dynamo's full router configuration belongs in a later,
//! separately versioned provider-options contract.

use std::collections::HashSet;
use std::ffi::CStr;
use std::panic::{AssertUnwindSafe, catch_unwind};

use aisimulate_core::replay::loadgen::ReplayRequestHashes;
use aisimulate_core::replay::{
    DirectRequest, Placement, PlacementCacheSample, PlacementDecision, PlacementPolicy,
    ReplayAdmissionMetadata, ReplayRequestContext, WorkerTopology,
};
use aisimulate_placement_abi::{
    AdmissionDecisionV1, ByteSliceV1, CAPABILITY_LOSSLESS_KV_EVENTS_V1, KvEventKindV1,
    KvEventSliceV1, KvEventV1, PlacementAdmissionV1, PlacementBatchResultV1,
    PlacementCacheSampleV1, PlacementCreateRequestV1, PlacementDiagnosticSliceV1,
    PlacementHandleV1, PlacementMetadataV1, PlacementMutationKindV1, PlacementMutationSliceV1,
    PlacementResultSliceV1, PlacementResultV1, PlacementSliceV1, PlacementV1, PluginDescriptorV1,
    PluginVTableV1, PromptIdentityV1, StatusV1, WorkerTopologyV1, validate_create_request_v1,
    validate_kv_event_batch_v1, validate_mutation_batch_v1,
};
use dynamo_kv_router::protocols::{
    ExternalSequenceBlockHash, KvCacheEvent, KvCacheEventData, KvCacheRemoveData, KvCacheStoreData,
    KvCacheStoredBlockData, LocalBlockHash, RouterEvent, StorageTier,
};
use dynamo_mocker::placement::{
    KvReplayMetadata, KvRouterConfig, KvRouterPlacement, MockEngineArgs,
};
use uuid::Uuid;

const PROVIDER_ID: &CStr = c"dynamo.kv-router";
const DIAGNOSTIC_LIMIT: usize = 256;

/// A resolved scheduler topology that is retained with the opaque provider
/// handle. Admission support will use it to translate Dynamo router indexes
/// back to the host's stable worker and scheduler identities.
#[derive(Debug)]
struct WorkerRoute {
    worker_id: u64,
    router_worker_id: usize,
    scheduler_ids: Vec<u64>,
}

/// Plugin-private state for one configured ABI placement policy.
struct ConfiguredPlacement {
    placement: KvRouterPlacement,
    topology: Vec<WorkerRoute>,
    max_diagnostic_bytes: usize,
    max_mutations: usize,
    max_admission_results: usize,
    max_released: usize,
    last_now_ms: f64,
    last_error: String,
}

static VTABLE: PluginVTableV1 = PluginVTableV1 {
    struct_size: std::mem::size_of::<PluginVTableV1>() as u32,
    flags: 0,
    create: Some(create),
    apply_batch: Some(apply_batch),
    release_results: Some(release_results),
    release_bytes: Some(release_bytes),
    last_error: Some(last_error),
    destroy: Some(destroy),
    apply_kv_events: Some(apply_kv_events),
};

static DESCRIPTOR: PluginDescriptorV1 = PluginDescriptorV1 {
    abi_major: PluginDescriptorV1::ABI_MAJOR,
    abi_minor: PluginDescriptorV1::ABI_MINOR,
    struct_size: std::mem::size_of::<PluginDescriptorV1>() as u32,
    flags: 0,
    capabilities: CAPABILITY_LOSSLESS_KV_EVENTS_V1,
    provider_id: PROVIDER_ID.as_ptr(),
    vtable: &VTABLE,
};

/// Returns Dynamo's immutable placement-plugin V1 descriptor.
///
/// The fixed symbol name is the entry point resolved by an AISimulate plugin
/// loader. The Rust-callable form is intentionally the same ABI so the
/// integration test exercises the exact descriptor a loader receives.
#[unsafe(no_mangle)]
pub extern "C" fn aisimulate_placement_plugin_v1() -> *const PluginDescriptorV1 {
    &raw const DESCRIPTOR
}

unsafe extern "C" fn create(
    request: PlacementCreateRequestV1,
    out_handle: *mut PlacementHandleV1,
    out_error: *mut ByteSliceV1,
) -> StatusV1 {
    if out_handle.is_null() || out_error.is_null() {
        return StatusV1::INVALID_ARGUMENT;
    }
    // Safety: both pointers were checked for null and are caller-owned output records.
    unsafe {
        *out_handle = PlacementHandleV1(std::ptr::null_mut());
        *out_error = ByteSliceV1::EMPTY;
    }

    match catch_unwind(AssertUnwindSafe(|| create_impl(request))) {
        Ok(Ok(placement)) => {
            // Safety: `out_handle` was validated above.
            unsafe { *out_handle = PlacementHandleV1(Box::into_raw(Box::new(placement)).cast()) };
            StatusV1::OK
        }
        Ok(Err(message)) => {
            // Safety: `out_error` was validated above.
            unsafe { *out_error = owned_bytes(&message, DIAGNOSTIC_LIMIT) };
            StatusV1::REJECTED
        }
        Err(_) => {
            // Safety: `out_error` was validated above.
            unsafe {
                *out_error = owned_bytes(
                    "dynamo placement provider panicked while creating an instance",
                    DIAGNOSTIC_LIMIT,
                );
            }
            StatusV1::INTERNAL
        }
    }
}

fn create_impl(request: PlacementCreateRequestV1) -> Result<ConfiguredPlacement, String> {
    validate_create_request_v1(&request)
        .map_err(|error| format!("invalid Dynamo placement create request: {error:?}"))?;
    validate_empty_options(request.options_namespace, request.provider_options)?;
    let (max_running_requests, total_kv_blocks) = decode_capacity_profile(&request.capacities)?;
    let topology = decode_topology(request.workers, request.capacities)?;
    if topology.is_empty() {
        return Err("Dynamo placement requires at least one worker".to_owned());
    }

    let selector_seed = u64::from_le_bytes(
        request.selector_seed[..8]
            .try_into()
            .expect("fixed-size seed"),
    );
    let engine_args = MockEngineArgs {
        max_num_seqs: Some(max_running_requests),
        num_gpu_blocks: total_kv_blocks,
        ..MockEngineArgs::default()
    };
    let router_config = KvRouterConfig {
        router_queue_threshold: Some(0.0),
        ..KvRouterConfig::default()
    };
    let placement = KvRouterPlacement::new(
        &engine_args,
        Some(router_config),
        None,
        topology.len(),
        Some(selector_seed),
    )
    .map_err(|error| format!("Dynamo KV Router placement initialization failed: {error}"))?;

    Ok(ConfiguredPlacement {
        placement,
        topology,
        max_diagnostic_bytes: usize::try_from(request.limits.max_diagnostic_bytes)
            .unwrap_or(DIAGNOSTIC_LIMIT),
        max_mutations: usize::try_from(request.limits.max_mutations)
            .map_err(|_| "Dynamo placement mutation limit exceeds platform bounds")?,
        max_admission_results: usize::try_from(request.limits.max_admission_results)
            .map_err(|_| "Dynamo placement admission-result limit exceeds platform bounds")?,
        max_released: usize::try_from(request.limits.max_released)
            .map_err(|_| "Dynamo placement released-result limit exceeds platform bounds")?,
        last_now_ms: 0.0,
        last_error: String::new(),
    })
}

fn validate_empty_options(namespace: ByteSliceV1, options: ByteSliceV1) -> Result<(), String> {
    // Safety: the neutral ABI validation checked that non-empty input slices
    // have non-null pointers; their borrowed lifetime is this create call.
    let namespace = borrowed_bytes(&namespace);
    // Safety: same as `namespace` above.
    let options = borrowed_bytes(&options);
    if namespace.is_empty() && options.is_empty() {
        return Ok(());
    }
    Err("Dynamo placement provider V1 accepts only empty provider options".to_owned())
}

fn decode_topology(
    workers: aisimulate_placement_abi::WorkerTopologySliceV1,
    capacities: aisimulate_placement_abi::WorkerCapacitySliceV1,
) -> Result<Vec<WorkerRoute>, String> {
    // Safety: the neutral ABI validation checked the outer borrowed slices.
    let workers = unsafe { std::slice::from_raw_parts(workers.data, workers.len as usize) };
    // Safety: the neutral ABI validation checked the outer borrowed slices.
    let capacities =
        unsafe { std::slice::from_raw_parts(capacities.data, capacities.len as usize) };
    if workers.len() != capacities.len() {
        return Err("Dynamo placement requires one capacity record per worker".to_owned());
    }

    let capacity_workers: HashSet<u64> = capacities
        .iter()
        .map(|capacity| capacity.worker_id)
        .collect();
    if capacity_workers.len() != capacities.len() {
        return Err("Dynamo placement topology has duplicate capacity worker IDs".to_owned());
    }

    let mut worker_ids = HashSet::with_capacity(workers.len());
    workers
        .iter()
        .enumerate()
        .map(|(router_worker_id, worker)| decode_worker(worker, router_worker_id))
        .map(|worker| {
            let worker = worker?;
            if !worker_ids.insert(worker.worker_id) {
                return Err("Dynamo placement topology has duplicate worker IDs".to_owned());
            }
            if !capacity_workers.contains(&worker.worker_id) {
                return Err("Dynamo placement capacity record does not match topology".to_owned());
            }
            Ok(worker)
        })
        .collect()
}

fn decode_worker(
    worker: &WorkerTopologyV1,
    router_worker_id: usize,
) -> Result<WorkerRoute, String> {
    if worker.scheduler_ids.len != 1 || worker.scheduler_ids.data.is_null() {
        return Err(
            "Dynamo placement requires exactly one scheduler per worker until DP mapping is implemented"
                .to_owned(),
        );
    }
    let scheduler_count = usize::try_from(worker.scheduler_ids.len)
        .map_err(|_| "Dynamo placement scheduler count exceeds platform bounds".to_owned())?;
    // Safety: a non-empty scheduler slice is required by the V1 caller
    // contract. This provider only copies it while the create call borrows it.
    let scheduler_ids =
        unsafe { std::slice::from_raw_parts(worker.scheduler_ids.data, scheduler_count).to_vec() };
    if scheduler_ids.iter().collect::<HashSet<_>>().len() != scheduler_ids.len() {
        return Err("Dynamo placement topology has duplicate scheduler IDs".to_owned());
    }
    Ok(WorkerRoute {
        worker_id: worker.worker_id,
        router_worker_id,
        scheduler_ids,
    })
}

fn decode_capacity_profile(
    capacities: &aisimulate_placement_abi::WorkerCapacitySliceV1,
) -> Result<(usize, usize), String> {
    // Safety: create-request validation established a bounded, readable slice.
    let capacities =
        unsafe { std::slice::from_raw_parts(capacities.data, capacities.len as usize) };
    let Some(first) = capacities.first() else {
        return Err("Dynamo placement requires at least one capacity record".to_owned());
    };
    if first.available_kv_blocks != first.total_kv_blocks {
        return Err(
            "Dynamo placement does not support partially available KV-block capacity".to_owned(),
        );
    }
    let max_running_requests = usize::try_from(first.max_running_requests).map_err(|_| {
        "Dynamo placement running-request capacity exceeds platform bounds".to_owned()
    })?;
    let total_kv_blocks = usize::try_from(first.total_kv_blocks)
        .map_err(|_| "Dynamo placement KV-block capacity exceeds platform bounds".to_owned())?;
    if max_running_requests == 0 {
        return Err("Dynamo placement requires a positive running-request capacity".to_owned());
    }
    if total_kv_blocks == 0 {
        return Err("Dynamo placement requires a positive KV-block capacity".to_owned());
    }
    if capacities.iter().skip(1).any(|capacity| {
        capacity.max_running_requests != first.max_running_requests
            || capacity.total_kv_blocks != first.total_kv_blocks
            || capacity.available_kv_blocks != first.available_kv_blocks
    }) {
        return Err(
            "Dynamo placement requires homogeneous worker capacities until per-worker capacity mapping is implemented"
                .to_owned(),
        );
    }
    Ok((max_running_requests, total_kv_blocks))
}

unsafe extern "C" fn apply_batch(
    handle: PlacementHandleV1,
    batch: PlacementMutationSliceV1,
    out_result: *mut PlacementBatchResultV1,
) -> StatusV1 {
    if handle.0.is_null() || out_result.is_null() {
        return StatusV1::INVALID_ARGUMENT;
    }
    // Safety: `out_result` was validated above.
    unsafe { *out_result = empty_batch_result() };
    match catch_unwind(AssertUnwindSafe(|| {
        // Safety: a non-null handle can only be created by this V1 table and
        // remains exclusively owned by the caller until `destroy`.
        let placement = unsafe { &mut *handle.0.cast::<ConfiguredPlacement>() };
        // Safety: `out_result` was validated and initialized above.
        apply_batch_impl(placement, batch, unsafe { &mut *out_result })
    })) {
        Ok(status) => status,
        Err(_) => {
            // Safety: the handle remains provider-owned even when a policy
            // callback panics; do not let an unwind cross the ABI boundary.
            let placement = unsafe { &mut *handle.0.cast::<ConfiguredPlacement>() };
            placement.last_error = bounded_message(
                "dynamo placement provider panicked while applying mutations",
                placement.max_diagnostic_bytes,
            );
            StatusV1::INTERNAL
        }
    }
}

unsafe extern "C" fn apply_kv_events(
    handle: PlacementHandleV1,
    events: KvEventSliceV1,
    now_ms: f64,
    out_result: *mut PlacementBatchResultV1,
) -> StatusV1 {
    if handle.0.is_null() || out_result.is_null() {
        return StatusV1::INVALID_ARGUMENT;
    }
    // Safety: `out_result` was validated above.
    unsafe { *out_result = empty_batch_result() };
    match catch_unwind(AssertUnwindSafe(|| {
        // Safety: a non-null handle can only be created by this V1 table and
        // remains exclusively owned by the caller until `destroy`.
        let placement = unsafe { &mut *handle.0.cast::<ConfiguredPlacement>() };
        // Safety: `out_result` was validated and initialized above.
        apply_kv_events_impl(placement, events, now_ms, unsafe { &mut *out_result })
    })) {
        Ok(status) => status,
        Err(_) => {
            // Safety: the handle remains provider-owned even when a policy
            // callback panics; do not let an unwind cross the ABI boundary.
            let placement = unsafe { &mut *handle.0.cast::<ConfiguredPlacement>() };
            placement.last_error = bounded_message(
                "dynamo placement provider panicked while applying KV events",
                placement.max_diagnostic_bytes,
            );
            StatusV1::INTERNAL
        }
    }
}

fn apply_kv_events_impl(
    placement: &mut ConfiguredPlacement,
    events: KvEventSliceV1,
    now_ms: f64,
    out_result: &mut PlacementBatchResultV1,
) -> StatusV1 {
    // Safety: the ABI validator verifies the bounded outer and nested slices
    // before this provider reads any packet or payload.
    if let Err(status) = unsafe { validate_kv_event_batch_v1(events) } {
        placement.last_error = "invalid Dynamo placement KV event batch".to_owned();
        return status;
    }
    if !now_ms.is_finite() || now_ms < placement.last_now_ms {
        placement.last_error =
            "Dynamo placement KV event time must be finite and monotonic".to_owned();
        return StatusV1::INVALID_ARGUMENT;
    }
    let Ok(event_count) = usize::try_from(events.len) else {
        placement.last_error = "Dynamo placement KV event count exceeds platform bounds".to_owned();
        return StatusV1::INVALID_ARGUMENT;
    };
    if event_count > placement.max_mutations {
        placement.last_error =
            "Dynamo placement KV event count exceeds the negotiated limit".to_owned();
        return StatusV1::INVALID_ARGUMENT;
    }
    if event_count == 0 {
        placement.last_now_ms = now_ms;
        placement.last_error.clear();
        out_result.struct_size = std::mem::size_of::<PlacementBatchResultV1>() as u32;
        out_result.pending_count =
            PlacementPolicy::<DirectRequest>::pending_count(&placement.placement) as u64;
        return StatusV1::OK;
    }
    // Safety: `validate_kv_event_batch_v1` established a readable bounded slice.
    let events = unsafe { std::slice::from_raw_parts(events.data, event_count) };
    let router_events = match decode_kv_events(placement, events) {
        Ok(events) => events,
        Err(error) => {
            placement.last_error = bounded_message(&error.message, placement.max_diagnostic_bytes);
            return error.status;
        }
    };
    let released = match placement.placement.observe_router_events(router_events) {
        Ok(released) => released,
        Err(error) => {
            placement.last_error = bounded_message(
                &format!("Dynamo placement KV observation failed: {error}"),
                placement.max_diagnostic_bytes,
            );
            return StatusV1::REJECTED;
        }
    };
    let mut released_records = Vec::new();
    if let Err(error) = append_placements(placement, released, &mut released_records) {
        placement.last_error = bounded_message(&error.message, placement.max_diagnostic_bytes);
        return error.status;
    }
    if released_records.len() > placement.max_released {
        placement.last_error =
            "Dynamo placement policy exceeded negotiated released-result bounds".to_owned();
        out_result.applied_mutations = event_count as u64;
        out_result.pending_count =
            PlacementPolicy::<DirectRequest>::pending_count(&placement.placement) as u64;
        return StatusV1::REJECTED;
    }
    placement.last_now_ms = now_ms;
    placement.last_error.clear();
    *out_result = PlacementBatchResultV1 {
        struct_size: std::mem::size_of::<PlacementBatchResultV1>() as u32,
        flags: 0,
        applied_mutations: event_count as u64,
        pending_count: PlacementPolicy::<DirectRequest>::pending_count(&placement.placement) as u64,
        admission_results: PlacementResultSliceV1 {
            data: std::ptr::null(),
            len: 0,
        },
        released: owned_placements(released_records),
        diagnostics: PlacementDiagnosticSliceV1 {
            data: std::ptr::null(),
            len: 0,
        },
    };
    StatusV1::OK
}

fn decode_kv_events(
    configured: &ConfiguredPlacement,
    events: &[KvEventV1],
) -> Result<Vec<RouterEvent>, ApplyError> {
    events
        .iter()
        .map(|event| decode_kv_event(configured, event))
        .collect()
}

fn decode_kv_event(
    configured: &ConfiguredPlacement,
    event: &KvEventV1,
) -> Result<RouterEvent, ApplyError> {
    let route = configured
        .topology
        .iter()
        .find(|route| route.worker_id == event.worker_id)
        .ok_or_else(|| {
            ApplyError::rejected(format!(
                "Dynamo placement does not know KV event worker {}",
                event.worker_id
            ))
        })?;
    let data = match event.kind {
        KvEventKindV1::STORED => {
            let start_position = event
                .start_position()
                .map(u32::try_from)
                .transpose()
                .map_err(|_| {
                    ApplyError::rejected(
                        "Dynamo placement KV event start position exceeds router bounds",
                    )
                })?;
            let blocks = event
                .stored_blocks()
                .ok_or_else(|| ApplyError::rejected("invalid Dynamo placement stored KV packet"))?
                .iter()
                .map(|block| KvCacheStoredBlockData {
                    block_hash: ExternalSequenceBlockHash(block.sequence_hash),
                    tokens_hash: LocalBlockHash(block.token_hash),
                    mm_extra_info: None,
                })
                .collect();
            KvCacheEventData::Stored(KvCacheStoreData {
                parent_hash: event.parent_hash().map(ExternalSequenceBlockHash),
                start_position,
                blocks,
            })
        }
        KvEventKindV1::REMOVED => KvCacheEventData::Removed(KvCacheRemoveData {
            block_hashes: event
                .removed_hashes()
                .ok_or_else(|| ApplyError::rejected("invalid Dynamo placement removed KV packet"))?
                .iter()
                .copied()
                .map(ExternalSequenceBlockHash)
                .collect(),
        }),
        _ => {
            return Err(ApplyError::rejected(
                "invalid Dynamo placement KV event kind",
            ));
        }
    };
    Ok(RouterEvent::with_storage_tier(
        route.router_worker_id as u64,
        KvCacheEvent {
            event_id: event.event_id,
            data,
            dp_rank: event.dp_rank,
        },
        StorageTier::Device,
    ))
}

fn apply_batch_impl(
    placement: &mut ConfiguredPlacement,
    batch: PlacementMutationSliceV1,
    out_result: &mut PlacementBatchResultV1,
) -> StatusV1 {
    // Safety: the ABI validator verifies bounded length, non-null input when
    // non-empty, and finite mutation times before this provider reads it.
    if let Err(status) = unsafe { validate_mutation_batch_v1(batch) } {
        placement.last_error = "invalid Dynamo placement mutation batch".to_owned();
        return status;
    }
    if batch.len == 0 {
        return StatusV1::OK;
    }
    let Ok(batch_len) = usize::try_from(batch.len) else {
        placement.last_error = "Dynamo placement mutation count exceeds platform bounds".to_owned();
        return StatusV1::INVALID_ARGUMENT;
    };
    if batch_len > placement.max_mutations {
        placement.last_error =
            "Dynamo placement mutation count exceeds the negotiated limit".to_owned();
        return StatusV1::INVALID_ARGUMENT;
    }
    // Safety: batch validation above established a readable non-empty slice.
    let mutations = unsafe { std::slice::from_raw_parts(batch.data, batch_len) };
    if mutations
        .iter()
        .any(|mutation| mutation.now_ms < placement.last_now_ms)
        || mutations
            .windows(2)
            .any(|pair| pair[1].now_ms < pair[0].now_ms)
    {
        placement.last_error = "Dynamo placement mutation times must be monotonic".to_owned();
        return StatusV1::INVALID_ARGUMENT;
    }

    let mut admissions = Vec::new();
    let mut released = Vec::new();
    for (index, mutation) in mutations.iter().enumerate() {
        match apply_mutation(placement, mutation, &mut admissions, &mut released) {
            Ok(()) => placement.last_now_ms = mutation.now_ms,
            Err(error) => {
                placement.last_error =
                    bounded_message(&error.message, placement.max_diagnostic_bytes);
                // Output ownership is transferred only for a fully successful batch.
                // The committed prefix remains observable without allocating
                // slices a host might not release after a non-OK status.
                out_result.applied_mutations = index as u64;
                out_result.pending_count =
                    PlacementPolicy::<DirectRequest>::pending_count(&placement.placement) as u64;
                return error.status;
            }
        }
    }
    if admissions.len() > placement.max_admission_results || released.len() > placement.max_released
    {
        placement.last_error =
            "Dynamo placement policy exceeded negotiated result bounds".to_owned();
        // The router has already applied every mutation. Preserve that fact
        // for a host that will not receive owned result slices on failure.
        out_result.applied_mutations = batch_len as u64;
        out_result.pending_count =
            PlacementPolicy::<DirectRequest>::pending_count(&placement.placement) as u64;
        return StatusV1::REJECTED;
    }
    placement.last_error.clear();
    *out_result = PlacementBatchResultV1 {
        struct_size: std::mem::size_of::<PlacementBatchResultV1>() as u32,
        flags: 0,
        applied_mutations: batch_len as u64,
        pending_count: PlacementPolicy::<DirectRequest>::pending_count(&placement.placement) as u64,
        admission_results: owned_admission_results(admissions),
        released: owned_placements(released),
        diagnostics: PlacementDiagnosticSliceV1 {
            data: std::ptr::null(),
            len: 0,
        },
    };
    StatusV1::OK
}

struct ApplyError {
    status: StatusV1,
    message: String,
}

impl ApplyError {
    fn rejected(message: impl Into<String>) -> Self {
        Self {
            status: StatusV1::REJECTED,
            message: message.into(),
        }
    }

    fn unsupported(kind: u32) -> Self {
        Self {
            status: StatusV1::UNSUPPORTED,
            message: format!(
                "Dynamo placement provider does not support observation mutation kind {kind}"
            ),
        }
    }
}

fn apply_mutation(
    configured: &mut ConfiguredPlacement,
    mutation: &aisimulate_placement_abi::PlacementMutationV1,
    admissions: &mut Vec<PlacementResultV1>,
    released: &mut Vec<PlacementV1>,
) -> Result<(), ApplyError> {
    match mutation.kind {
        PlacementMutationKindV1::ADMIT => {
            // Safety: `validate_mutation_batch_v1` validated the active admission DTO.
            let admission = unsafe { mutation.payload.admission };
            let (request, metadata, session_id) = decode_admission(admission)?;
            let effects = PlacementPolicy::<DirectRequest>::place(
                &mut configured.placement,
                &request,
                metadata,
                session_id,
                mutation.now_ms,
            )
            .map_err(|error| {
                ApplyError::rejected(format!("Dynamo placement admission failed: {error}"))
            })?;
            let (decision, result_placement) = match effects.decision {
                PlacementDecision::Immediate(placement) => (
                    AdmissionDecisionV1::IMMEDIATE,
                    convert_placement(configured, placement)?,
                ),
                PlacementDecision::Queued => (
                    AdmissionDecisionV1::QUEUED,
                    PlacementV1 {
                        request_id: admission.request_id,
                        worker_id: 0,
                        scheduler_id: 0,
                        reported_overlap_tokens: 0,
                        cache_sample: PlacementCacheSampleV1 {
                            flags: 0,
                            overlap_blocks: 0,
                            best_available_overlap_blocks: 0,
                            isl_blocks: 0,
                        },
                        placement_replica_id: 0,
                        reserved: 0,
                    },
                ),
            };
            admissions.push(PlacementResultV1 {
                request_id: admission.request_id,
                decision,
                reserved: [0; 4],
                placement: result_placement,
            });
            append_placements(configured, effects.released, released)?;
        }
        PlacementMutationKindV1::CANCEL_PENDING => {
            // Safety: this active union member is selected by the mutation kind.
            let lifecycle = unsafe { mutation.payload.request_lifecycle };
            let _ = PlacementPolicy::<DirectRequest>::cancel_pending(
                &mut configured.placement,
                Uuid::from_bytes(lifecycle.request_id),
            );
        }
        PlacementMutationKindV1::REQUEST_TERMINAL => {
            // Safety: this active union member is selected by the mutation kind.
            let lifecycle = unsafe { mutation.payload.request_lifecycle };
            let effects = PlacementPolicy::<DirectRequest>::request_terminal(
                &mut configured.placement,
                Uuid::from_bytes(lifecycle.request_id),
                mutation.now_ms,
            )
            .map_err(|error| {
                ApplyError::rejected(format!("Dynamo placement terminal update failed: {error}"))
            })?;
            append_placements(configured, effects, released)?;
        }
        PlacementMutationKindV1::PREFILL_COMPLETED => {
            // Safety: this active union member is selected by the mutation kind.
            let lifecycle = unsafe { mutation.payload.request_lifecycle };
            let effects = PlacementPolicy::<DirectRequest>::prefill_completed(
                &mut configured.placement,
                Uuid::from_bytes(lifecycle.request_id),
                mutation.now_ms,
            )
            .map_err(|error| {
                ApplyError::rejected(format!("Dynamo placement prefill update failed: {error}"))
            })?;
            append_placements(configured, effects, released)?;
        }
        PlacementMutationKindV1::WORKER_READY => {
            // Safety: this active union member is selected by the mutation kind.
            let worker = unsafe { mutation.payload.worker };
            let route = ready_route(configured, worker)?;
            let effects = PlacementPolicy::<DirectRequest>::worker_ready(
                &mut configured.placement,
                route,
                mutation.now_ms,
            )
            .map_err(|error| {
                ApplyError::rejected(format!(
                    "Dynamo placement worker-ready update failed: {error}"
                ))
            })?;
            append_placements(configured, effects, released)?;
        }
        PlacementMutationKindV1::WORKER_DRAINING | PlacementMutationKindV1::WORKER_REMOVED => {
            // Safety: this active union member is selected by the mutation kind.
            let worker = unsafe { mutation.payload.worker };
            let route = existing_route(configured, worker)?;
            let effects = if mutation.kind == PlacementMutationKindV1::WORKER_DRAINING {
                PlacementPolicy::<DirectRequest>::worker_draining(
                    &mut configured.placement,
                    route,
                    mutation.now_ms,
                )
            } else {
                PlacementPolicy::<DirectRequest>::worker_removed(
                    &mut configured.placement,
                    route,
                    mutation.now_ms,
                )
            }
            .map_err(|error| {
                ApplyError::rejected(format!("Dynamo placement worker update failed: {error}"))
            })?;
            append_placements(configured, effects, released)?;
        }
        PlacementMutationKindV1::TOPOLOGY_SETTLED => {
            let effects = PlacementPolicy::<DirectRequest>::topology_settled(
                &mut configured.placement,
                mutation.now_ms,
            )
            .map_err(|error| {
                ApplyError::rejected(format!(
                    "Dynamo placement topology settlement failed: {error}"
                ))
            })?;
            append_placements(configured, effects, released)?;
        }
        kind => return Err(ApplyError::unsupported(kind.0)),
    }
    Ok(())
}

fn decode_admission(
    admission: PlacementAdmissionV1,
) -> Result<(DirectRequest, KvReplayMetadata, Option<String>), ApplyError> {
    let prompt_tokens = usize::try_from(admission.prompt_tokens).map_err(|_| {
        ApplyError::rejected("Dynamo placement prompt length exceeds platform bounds")
    })?;
    let max_output_tokens = usize::try_from(admission.max_output_tokens).map_err(|_| {
        ApplyError::rejected("Dynamo placement output length exceeds platform bounds")
    })?;
    let identity = admission.prompt_identity;
    let has_tokens = identity.flags & PromptIdentityV1::MATERIALIZED_TOKEN_IDS_PRESENT != 0;
    let has_local_hashes = identity.flags & PromptIdentityV1::LOCAL_BLOCK_HASHES_PRESENT != 0;
    let has_sequence_hashes = identity.flags & PromptIdentityV1::SEQUENCE_BLOCK_HASHES_PRESENT != 0;
    if has_local_hashes != has_sequence_hashes {
        return Err(ApplyError::rejected(
            "Dynamo KV placement requires local and sequence replay hashes together",
        ));
    }
    if !has_tokens && !has_local_hashes {
        return Err(ApplyError::rejected(
            "Dynamo KV placement requires materialized prompt tokens or paired replay hashes",
        ));
    }
    let tokens = if has_tokens {
        // Safety: ABI batch validation bounded this readable slice and checked
        // that its length equals `prompt_tokens`.
        unsafe { copy_abi_slice(identity.materialized_token_ids.data, prompt_tokens) }
    } else {
        // Replay hashes are the cache identity in this branch. The existing
        // policy still needs a length-bearing request view for scheduling;
        // these placeholders are never used for cache matching.
        vec![0; prompt_tokens]
    };
    let hashes = if has_local_hashes {
        // Safety: ABI batch validation checked both paired hash slices and
        // their equal bounded lengths.
        let local_block_hashes = unsafe {
            copy_abi_slice(
                identity.local_block_hashes.data,
                identity.local_block_hashes.len as usize,
            )
        };
        let sequence_hashes = unsafe {
            copy_abi_slice(
                identity.sequence_block_hashes.data,
                identity.sequence_block_hashes.len as usize,
            )
        };
        Some(ReplayRequestHashes {
            local_block_hashes,
            sequence_hashes,
        })
    } else {
        None
    };
    let replay_context = decode_metadata(admission.metadata)?;
    let session_id = replay_context
        .as_ref()
        .and_then(|context| context.session_id.clone());
    Ok((
        DirectRequest {
            tokens,
            max_output_tokens,
            uuid: Some(Uuid::from_bytes(admission.request_id)),
            priority: admission.priority,
            replay_context,
            ..DirectRequest::default()
        },
        KvReplayMetadata::from_hashes(hashes),
        session_id,
    ))
}

fn decode_metadata(
    metadata: PlacementMetadataV1,
) -> Result<Option<ReplayRequestContext>, ApplyError> {
    if metadata.format == aisimulate_placement_abi::AdmissionMetadataFormatV1::NONE {
        return Ok(None);
    }
    // Safety: ABI batch validation checked this non-empty JSON byte slice.
    let bytes =
        unsafe { std::slice::from_raw_parts(metadata.bytes.data, metadata.bytes.len as usize) };
    serde_json::from_slice(bytes).map(Some).map_err(|error| {
        ApplyError::rejected(format!(
            "Dynamo placement metadata is not a replay context: {error}"
        ))
    })
}

fn ready_route(
    configured: &mut ConfiguredPlacement,
    worker: WorkerTopologyV1,
) -> Result<WorkerTopology, ApplyError> {
    let decoded =
        decode_worker(&worker, configured.topology.len()).map_err(ApplyError::rejected)?;
    let router_worker_id = match configured
        .topology
        .iter()
        .position(|route| route.worker_id == decoded.worker_id)
    {
        Some(index) => {
            if configured.topology[index].scheduler_ids != decoded.scheduler_ids {
                return Err(ApplyError::rejected(
                    "Dynamo placement worker-ready scheduler topology changed",
                ));
            }
            configured.topology[index].router_worker_id
        }
        None => {
            let router_worker_id = configured.topology.len();
            configured.topology.push(WorkerRoute {
                router_worker_id,
                ..decoded
            });
            router_worker_id
        }
    };
    route_topology(configured, router_worker_id)
}

fn existing_route(
    configured: &ConfiguredPlacement,
    worker: WorkerTopologyV1,
) -> Result<WorkerTopology, ApplyError> {
    let decoded = decode_worker(&worker, 0).map_err(ApplyError::rejected)?;
    let worker_id = decoded.worker_id;
    let route = configured
        .topology
        .iter()
        .find(|route| route.worker_id == worker_id)
        .ok_or_else(|| {
            ApplyError::rejected(format!("Dynamo placement does not know worker {worker_id}"))
        })?;
    if route.scheduler_ids != decoded.scheduler_ids {
        return Err(ApplyError::rejected(
            "Dynamo placement worker lifecycle scheduler topology changed",
        ));
    }
    route_topology(configured, route.router_worker_id)
}

fn route_topology(
    configured: &ConfiguredPlacement,
    router_worker_id: usize,
) -> Result<WorkerTopology, ApplyError> {
    let route = configured
        .topology
        .iter()
        .find(|route| route.router_worker_id == router_worker_id)
        .ok_or_else(|| ApplyError::rejected("Dynamo placement lost its worker route"))?;
    let scheduler_ids = route
        .scheduler_ids
        .iter()
        .copied()
        .map(usize::try_from)
        .collect::<Result<Vec<_>, _>>()
        .map_err(|_| {
            ApplyError::rejected("Dynamo placement scheduler ID exceeds platform bounds")
        })?;
    Ok(WorkerTopology {
        worker_id: router_worker_id,
        scheduler_ids,
    })
}

fn append_placements(
    configured: &ConfiguredPlacement,
    values: Vec<Placement>,
    out: &mut Vec<PlacementV1>,
) -> Result<(), ApplyError> {
    for value in values {
        out.push(convert_placement(configured, value)?);
    }
    Ok(())
}

fn convert_placement(
    configured: &ConfiguredPlacement,
    value: Placement,
) -> Result<PlacementV1, ApplyError> {
    let route = configured
        .topology
        .iter()
        .find(|route| route.router_worker_id == value.scheduler_id)
        .ok_or_else(|| {
            ApplyError::rejected("Dynamo placement selected an unknown router worker")
        })?;
    let scheduler_id = *route
        .scheduler_ids
        .first()
        .ok_or_else(|| ApplyError::rejected("Dynamo placement worker has no scheduler"))?;
    let cache_sample = match value.cache_sample {
        Some(PlacementCacheSample {
            overlap_blocks,
            best_available_overlap_blocks,
            isl_blocks,
        }) => PlacementCacheSampleV1 {
            flags: PlacementCacheSampleV1::PRESENT,
            overlap_blocks,
            best_available_overlap_blocks,
            isl_blocks,
        },
        None => PlacementCacheSampleV1 {
            flags: 0,
            overlap_blocks: 0,
            best_available_overlap_blocks: 0,
            isl_blocks: 0,
        },
    };
    Ok(PlacementV1 {
        request_id: value.request_id.into_bytes(),
        worker_id: route.worker_id,
        scheduler_id,
        reported_overlap_tokens: u64::try_from(value.reported_overlap_tokens).map_err(|_| {
            ApplyError::rejected("Dynamo placement overlap count exceeds ABI bounds")
        })?,
        cache_sample,
        placement_replica_id: value
            .placement_replica_id
            .map(u64::try_from)
            .transpose()
            .map_err(|_| ApplyError::rejected("Dynamo placement replica ID exceeds ABI bounds"))?
            .unwrap_or(0),
        reserved: 0,
    })
}

fn owned_admission_results(values: Vec<PlacementResultV1>) -> PlacementResultSliceV1 {
    owned_slice(values, |data, len| PlacementResultSliceV1 { data, len })
}

fn owned_placements(values: Vec<PlacementV1>) -> PlacementSliceV1 {
    owned_slice(values, |data, len| PlacementSliceV1 { data, len })
}

fn owned_slice<T, S>(values: Vec<T>, make_slice: impl FnOnce(*const T, u64) -> S) -> S {
    if values.is_empty() {
        return make_slice(std::ptr::null(), 0);
    }
    let values = values.into_boxed_slice();
    let len = values.len() as u64;
    let data = values.as_ptr();
    let _ = Box::into_raw(values);
    make_slice(data, len)
}

/// Copies a validated ABI input slice while preserving canonical empty slices,
/// whose pointer is allowed to be null by the V1 contract.
///
/// # Safety
/// Callers must have validated that every non-empty range is readable.
unsafe fn copy_abi_slice<T: Copy>(data: *const T, len: usize) -> Vec<T> {
    if len == 0 {
        return Vec::new();
    }
    // Safety: documented by this helper's caller contract.
    unsafe { std::slice::from_raw_parts(data, len).to_vec() }
}

fn bounded_message(message: &str, maximum_bytes: usize) -> String {
    let maximum_bytes = maximum_bytes.min(DIAGNOSTIC_LIMIT);
    let mut end = message.len().min(maximum_bytes);
    while !message.is_char_boundary(end) {
        end -= 1;
    }
    message[..end].to_owned()
}

unsafe extern "C" fn release_results(result: PlacementBatchResultV1) {
    // Safety: successful batches allocate these independent boxed slices and
    // transfer one matching release obligation through this callback.
    unsafe {
        release_slice(result.admission_results.data, result.admission_results.len);
        release_slice(result.released.data, result.released.len);
        release_diagnostics(result.diagnostics);
    }
}

unsafe fn release_slice<T>(data: *const T, len: u64) {
    if data.is_null() {
        return;
    }
    let Ok(len) = usize::try_from(len) else {
        return;
    };
    // Safety: callers return only slices allocated by `owned_slice`.
    unsafe {
        drop(Box::from_raw(std::ptr::slice_from_raw_parts_mut(
            data.cast_mut(),
            len,
        )))
    };
}

unsafe fn release_diagnostics(diagnostics: PlacementDiagnosticSliceV1) {
    if diagnostics.data.is_null() {
        return;
    }
    let Ok(len) = usize::try_from(diagnostics.len) else {
        return;
    };
    // Safety: diagnostics follow the same output ownership contract, with an
    // additional owned byte allocation per record.
    let diagnostics = unsafe {
        Box::from_raw(std::ptr::slice_from_raw_parts_mut(
            diagnostics.data.cast_mut(),
            len,
        ))
    };
    for diagnostic in diagnostics.iter() {
        // Safety: every diagnostic message is allocated by this provider.
        unsafe { release_bytes(diagnostic.message) };
    }
}

unsafe extern "C" fn release_bytes(bytes: ByteSliceV1) {
    if bytes.data.is_null() {
        return;
    }
    let Ok(len) = usize::try_from(bytes.len) else {
        return;
    };
    // Safety: every non-empty byte slice this provider returns comes from
    // `Box<[u8]>` with exactly this length and is released once through this table.
    unsafe {
        drop(Box::from_raw(std::ptr::slice_from_raw_parts_mut(
            bytes.data.cast_mut(),
            len,
        )))
    };
}

unsafe extern "C" fn last_error(
    handle: PlacementHandleV1,
    out_error: *mut ByteSliceV1,
) -> StatusV1 {
    if handle.0.is_null() || out_error.is_null() {
        return StatusV1::INVALID_ARGUMENT;
    }
    // Safety: this is a live handle from this provider's create operation.
    let placement = unsafe { &mut *handle.0.cast::<ConfiguredPlacement>() };
    // Safety: `out_error` was validated above.
    unsafe { *out_error = owned_bytes(&placement.last_error, placement.max_diagnostic_bytes) };
    StatusV1::OK
}

unsafe extern "C" fn destroy(handle: PlacementHandleV1) {
    if !handle.0.is_null() {
        // Safety: the handle was allocated by `create` and ownership returns
        // exactly once through the matching vtable's destroy operation.
        unsafe { drop(Box::from_raw(handle.0.cast::<ConfiguredPlacement>())) };
    }
}

fn empty_batch_result() -> PlacementBatchResultV1 {
    PlacementBatchResultV1 {
        struct_size: std::mem::size_of::<PlacementBatchResultV1>() as u32,
        flags: 0,
        applied_mutations: 0,
        pending_count: 0,
        admission_results: PlacementResultSliceV1 {
            data: std::ptr::null(),
            len: 0,
        },
        released: PlacementSliceV1 {
            data: std::ptr::null(),
            len: 0,
        },
        diagnostics: PlacementDiagnosticSliceV1 {
            data: std::ptr::null(),
            len: 0,
        },
    }
}

fn borrowed_bytes(bytes: &ByteSliceV1) -> &[u8] {
    if bytes.len == 0 {
        return &[];
    }
    // Safety: V1 creation validation requires a non-null pointer for every
    // non-empty byte slice, and the bytes are borrowed for this call.
    unsafe { std::slice::from_raw_parts(bytes.data, bytes.len as usize) }
}

fn owned_bytes(message: &str, maximum_bytes: usize) -> ByteSliceV1 {
    if maximum_bytes == 0 || message.is_empty() {
        return ByteSliceV1::EMPTY;
    }
    let message = owned_boxed_bytes(message, maximum_bytes);
    let bytes = ByteSliceV1 {
        data: message.as_ptr(),
        len: message.len() as u64,
    };
    let _ = Box::into_raw(message);
    bytes
}

fn owned_boxed_bytes(message: &str, maximum_bytes: usize) -> Box<[u8]> {
    let maximum_bytes = maximum_bytes.min(DIAGNOSTIC_LIMIT);
    let mut end = message.len().min(maximum_bytes);
    while !message.is_char_boundary(end) {
        end -= 1;
    }
    message.as_bytes()[..end].to_vec().into_boxed_slice()
}
