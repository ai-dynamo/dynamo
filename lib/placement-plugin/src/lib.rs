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

use aisimulate_placement_abi::{
    ByteSliceV1, MAX_CREATE_WORKERS_V1, PlacementBatchResultV1, PlacementCreateRequestV1,
    PlacementDiagnosticSliceV1, PlacementHandleV1, PlacementMutationSliceV1,
    PlacementResultSliceV1, PlacementSliceV1, PluginDescriptorV1, PluginVTableV1, StatusV1,
    WorkerTopologyV1, validate_create_request_v1, validate_mutation_batch_v1,
};
use dynamo_mocker::placement::{KvRouterPlacement, MockEngineArgs};

const PROVIDER_ID: &CStr = c"dynamo.kv-router";
const DIAGNOSTIC_LIMIT: usize = 256;

/// A resolved scheduler topology that is retained with the opaque provider
/// handle. Admission support will use it to translate Dynamo router indexes
/// back to the host's stable worker and scheduler identities.
#[derive(Debug)]
struct WorkerRoute {
    worker_id: u64,
    _scheduler_ids: Vec<u64>,
}

/// Plugin-private state for one configured ABI placement policy.
struct ConfiguredPlacement {
    _placement: KvRouterPlacement,
    _topology: Vec<WorkerRoute>,
    max_diagnostic_bytes: usize,
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
};

static DESCRIPTOR: PluginDescriptorV1 = PluginDescriptorV1 {
    abi_major: PluginDescriptorV1::ABI_MAJOR,
    abi_minor: 0,
    struct_size: std::mem::size_of::<PluginDescriptorV1>() as u32,
    flags: 0,
    capabilities: 0,
    provider_id: PROVIDER_ID.as_ptr(),
    vtable: &VTABLE,
};

/// Returns Dynamo's immutable placement-plugin V1 descriptor.
///
/// The fixed symbol name is the entry point resolved by an AISimulate plugin
/// loader. The Rust-callable form is intentionally the same ABI so the
/// integration test exercises the exact descriptor a loader receives.
#[unsafe(no_mangle)]
pub extern "C" fn dynamo_placement_plugin_entry_v1() -> *const PluginDescriptorV1 {
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
    let topology = decode_topology(request.workers, request.capacities)?;
    if topology.is_empty() {
        return Err("Dynamo placement requires at least one worker".to_owned());
    }

    let selector_seed = u64::from_le_bytes(
        request.selector_seed[..8]
            .try_into()
            .expect("fixed-size seed"),
    );
    let placement = KvRouterPlacement::new(
        &MockEngineArgs::default(),
        None,
        None,
        topology.len(),
        Some(selector_seed),
    )
    .map_err(|error| format!("Dynamo KV Router placement initialization failed: {error}"))?;

    Ok(ConfiguredPlacement {
        _placement: placement,
        _topology: topology,
        max_diagnostic_bytes: usize::try_from(request.limits.max_diagnostic_bytes)
            .unwrap_or(DIAGNOSTIC_LIMIT),
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
        .map(decode_worker)
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

fn decode_worker(worker: &WorkerTopologyV1) -> Result<WorkerRoute, String> {
    if worker.scheduler_ids.len == 0 || worker.scheduler_ids.data.is_null() {
        return Err("Dynamo placement requires at least one scheduler per worker".to_owned());
    }
    if worker.scheduler_ids.len > MAX_CREATE_WORKERS_V1 {
        return Err("Dynamo placement scheduler count exceeds the V1 bound".to_owned());
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
        _scheduler_ids: scheduler_ids,
    })
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
        apply_batch_impl(placement, batch)
    })) {
        Ok(status) => status,
        Err(_) => StatusV1::INTERNAL,
    }
}

fn apply_batch_impl(
    placement: &mut ConfiguredPlacement,
    batch: PlacementMutationSliceV1,
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
    // Safety: batch validation above established a readable non-empty slice.
    let mutation = unsafe { &*batch.data };
    let message = format!(
        "Dynamo placement provider V1 does not yet support mutation kind {}",
        mutation.kind.0
    );
    placement.last_error = message.clone();
    StatusV1::UNSUPPORTED
}

unsafe extern "C" fn release_results(_result: PlacementBatchResultV1) {}

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
