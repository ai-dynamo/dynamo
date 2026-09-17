// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use aiperf_steppable_abi::{
    ByteSliceV1, CAPABILITY_COMPACT_BUFFER_LEASES_V1, CompactBufferLeaseVTableTailV1,
    CompactRequestV1, CreateRequestV1, DirectRequestSliceV1, DirectRequestV1, EngineEventV1,
    HashBufferIdV1, HashBufferLeaseCallbacksV1, HashBufferRangeV1, PluginDescriptorV1,
    PluginVTableV1, PluginVTableV1Prefix, REPLAY_CONTEXT_FLAG_METADATA,
    REQUEST_FLAG_OUTPUT_TOKEN_IDS, REQUEST_FLAG_REPLAY_CONTEXT, REQUEST_FLAG_UUID, ReplayHandleV1,
    RequestIdMutSliceV1, StatusV1, StepRequestV1, StepResultV1, U32SliceV1, validate_descriptor_v1,
};
use std::ffi::{CStr, CString, c_char, c_int, c_void};
use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};

fn create_replay_with(
    config: dynamo_steppable_provider::BackendConfig,
) -> (
    &'static aiperf_steppable_abi::PluginVTableV1Prefix,
    ReplayHandleV1,
) {
    let descriptor = dynamo_steppable_provider::aiperf_steppable_plugin_v1();
    let table = unsafe { &*validate_descriptor_v1(descriptor).expect("complete V1 descriptor") };
    let payload = serde_json::to_vec(&config).expect("serializable aggregate configuration");
    let mut handle = ReplayHandleV1(std::ptr::null_mut());
    let mut error = ByteSliceV1::EMPTY;
    assert_eq!(
        unsafe {
            table.create.expect("create")(
                CreateRequestV1 {
                    struct_size: std::mem::size_of::<CreateRequestV1>() as u32,
                    flags: 0,
                    provider_payload: ByteSliceV1 {
                        data: payload.as_ptr(),
                        len: payload.len() as u64,
                    },
                },
                &mut handle,
                &mut error,
            )
        },
        StatusV1::OK
    );
    assert!(error.data.is_null());
    (table, handle)
}

fn create_replay() -> (
    &'static aiperf_steppable_abi::PluginVTableV1Prefix,
    ReplayHandleV1,
) {
    create_replay_with(dynamo_steppable_provider::BackendConfig::one_worker())
}

fn request(prompt: &[u32], id: [u8; 16]) -> DirectRequestV1 {
    DirectRequestV1 {
        struct_size: std::mem::size_of::<DirectRequestV1>() as u32,
        flags: REQUEST_FLAG_UUID,
        tokens: U32SliceV1 {
            data: prompt.as_ptr(),
            len: prompt.len() as u64,
        },
        output_token_ids: U32SliceV1::EMPTY,
        max_output_tokens: 1,
        uuid: id,
        dp_rank: 0,
        preferred_dp_rank: 0,
        preferred_prefill_dp_rank: 0,
        arrival_timestamp_ms: 0.0,
        priority: 0,
        strict_priority: 0,
        policy_class: ByteSliceV1::EMPTY,
        replay_context: aiperf_steppable_abi::ReplayContextV1::EMPTY,
    }
}

fn step_to_terminal(
    table: &aiperf_steppable_abi::PluginVTableV1Prefix,
    handle: ReplayHandleV1,
    id: [u8; 16],
) -> bool {
    for until_ms in [0.0, 1.0, 10.0, 100.0, 1_000.0] {
        let mut result = StepResultV1::EMPTY;
        assert_eq!(
            unsafe {
                table.step.expect("step")(
                    handle,
                    StepRequestV1 {
                        struct_size: std::mem::size_of::<StepRequestV1>() as u32,
                        flags: 0,
                        until_ms,
                    },
                    &mut result,
                )
            },
            StatusV1::OK
        );
        let completed = if result.events.len == 0 {
            false
        } else {
            unsafe { std::slice::from_raw_parts(result.events.data, result.events.len as usize) }
                .iter()
                .any(|event| {
                    event.request_id == id
                        && event.flags & aiperf_steppable_abi::ENGINE_EVENT_FLAG_TERMINAL != 0
                })
        };
        unsafe {
            table.release_events.expect("release events")(result.events);
            table.release_request_facts.expect("release facts")(result.request_facts);
        }
        if completed {
            return true;
        }
    }
    false
}

#[derive(Default)]
struct ReleaseLog {
    count: AtomicUsize,
    last_buffer_id: AtomicU64,
}

unsafe extern "C" fn record_hash_buffer_release(context: *mut c_void, buffer_id: HashBufferIdV1) {
    if context.is_null() {
        return;
    }
    // Safety: lease-aware test creation supplies a live `ReleaseLog` for the
    // whole replay lifetime. The callback only updates atomics and never
    // re-enters the provider.
    let log = unsafe { &*context.cast::<ReleaseLog>() };
    log.last_buffer_id.store(buffer_id.0, Ordering::SeqCst);
    log.count.fetch_add(1, Ordering::SeqCst);
}

fn lease_tail() -> (
    &'static PluginVTableV1Prefix,
    CompactBufferLeaseVTableTailV1,
) {
    let descriptor = dynamo_steppable_provider::aiperf_steppable_plugin_v1();
    let table = unsafe { &*validate_descriptor_v1(descriptor).expect("complete V1 descriptor") };
    let descriptor = unsafe { &*descriptor };
    assert_ne!(
        descriptor.capabilities & CAPABILITY_COMPACT_BUFFER_LEASES_V1,
        0
    );
    let tail = unsafe {
        PluginVTableV1::compact_buffer_leases(table, descriptor.capabilities)
            .expect("complete compact hash-buffer lease tail")
    };
    (table, tail)
}

fn create_leased_replay(
    log: &ReleaseLog,
) -> (
    &'static PluginVTableV1Prefix,
    CompactBufferLeaseVTableTailV1,
    ReplayHandleV1,
) {
    let (table, tail) = lease_tail();
    let payload = serde_json::to_vec(&dynamo_steppable_provider::BackendConfig::one_worker())
        .expect("serializable aggregate configuration");
    let mut handle = ReplayHandleV1(std::ptr::null_mut());
    let mut error = ByteSliceV1::EMPTY;
    assert_eq!(
        unsafe {
            tail.create_with_hash_buffer_leases
                .expect("lease-aware create")(
                CreateRequestV1 {
                    struct_size: std::mem::size_of::<CreateRequestV1>() as u32,
                    flags: 0,
                    provider_payload: ByteSliceV1 {
                        data: payload.as_ptr(),
                        len: payload.len() as u64,
                    },
                },
                HashBufferLeaseCallbacksV1 {
                    struct_size: std::mem::size_of::<HashBufferLeaseCallbacksV1>() as u32,
                    flags: 0,
                    context: std::ptr::from_ref(log).cast_mut().cast(),
                    release_hash_buffer: Some(record_hash_buffer_release),
                },
                &mut handle,
                &mut error,
            )
        },
        StatusV1::OK
    );
    assert!(error.data.is_null());
    (table, tail, handle)
}

fn register_hash_buffer(
    tail: CompactBufferLeaseVTableTailV1,
    handle: ReplayHandleV1,
    hash_ids: &[u32],
) -> HashBufferIdV1 {
    let mut buffer_id = HashBufferIdV1::INVALID;
    assert_eq!(
        unsafe {
            tail.register_hash_buffer.expect("register hash buffer")(
                handle,
                U32SliceV1 {
                    data: hash_ids.as_ptr(),
                    len: hash_ids.len() as u64,
                },
                &mut buffer_id,
            )
        },
        StatusV1::OK
    );
    assert!(buffer_id.is_valid());
    buffer_id
}

fn compact_request(id: [u8; 16], hash_ids: U32SliceV1) -> CompactRequestV1 {
    CompactRequestV1 {
        struct_size: std::mem::size_of::<CompactRequestV1>() as u32,
        flags: 0,
        input_token_count: 8,
        trace_block_size: 4,
        reserved: 0,
        hash_ids,
        request: request(&[], id),
    }
}

fn submit_hash_buffer_range(
    tail: CompactBufferLeaseVTableTailV1,
    handle: ReplayHandleV1,
    buffer_id: HashBufferIdV1,
    id: [u8; 16],
) -> StatusV1 {
    let mut request_id = [0; 16];
    let status = unsafe {
        tail.submit_compact_hash_buffer_range
            .expect("submit compact hash-buffer range")(
            handle,
            compact_request(id, U32SliceV1::EMPTY),
            HashBufferRangeV1 {
                buffer_id,
                offset: 0,
                len: 2,
            },
            &mut request_id,
        )
    };
    if status == StatusV1::OK {
        assert_eq!(request_id, id);
    } else {
        assert_eq!(request_id, [0; 16]);
    }
    status
}

fn submit_direct_batch(
    table: &PluginVTableV1Prefix,
    handle: ReplayHandleV1,
    requests: &[DirectRequestV1],
) -> (StatusV1, Vec<[u8; 16]>) {
    let mut request_ids = vec![[99; 16]; requests.len()];
    let status = unsafe {
        table.submit_batch.expect("submit batch")(
            handle,
            DirectRequestSliceV1 {
                data: requests.as_ptr(),
                len: requests.len() as u64,
            },
            RequestIdMutSliceV1 {
                data: request_ids.as_mut_ptr(),
                len: request_ids.len() as u64,
            },
        )
    };
    (status, request_ids)
}

#[test]
fn descriptor_creates_and_completes_one_routed_request() {
    let descriptor = dynamo_steppable_provider::aiperf_steppable_plugin_v1();
    let table = unsafe { &*validate_descriptor_v1(descriptor).expect("complete V1 descriptor") };

    let payload = serde_json::to_vec(&dynamo_steppable_provider::BackendConfig::one_worker())
        .expect("serializable aggregate configuration");
    let request = CreateRequestV1 {
        struct_size: std::mem::size_of::<CreateRequestV1>() as u32,
        flags: 0,
        provider_payload: ByteSliceV1 {
            data: payload.as_ptr(),
            len: payload.len() as u64,
        },
    };
    let mut handle = ReplayHandleV1(std::ptr::null_mut());
    let mut error = ByteSliceV1::EMPTY;
    assert_eq!(
        unsafe { table.create.expect("create")(request, &mut handle, &mut error) }.0,
        0
    );
    assert!(error.data.is_null());

    let prompt = [7_u32, 11, 13, 17];
    let mut request_id = [0_u8; 16];
    assert_eq!(
        unsafe {
            table.submit.expect("submit")(
                handle,
                DirectRequestV1 {
                    struct_size: std::mem::size_of::<DirectRequestV1>() as u32,
                    flags: 0,
                    tokens: U32SliceV1 {
                        data: prompt.as_ptr(),
                        len: prompt.len() as u64,
                    },
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
                    replay_context: aiperf_steppable_abi::ReplayContextV1::EMPTY,
                },
                &mut request_id,
            )
        }
        .0,
        0
    );

    let mut completed = false;
    let mut saw_facts = false;
    for until_ms in [0.0, 1.0, 10.0, 100.0, 1_000.0] {
        let mut result = StepResultV1::EMPTY;
        assert_eq!(
            unsafe {
                table.step.expect("step")(
                    handle,
                    StepRequestV1 {
                        struct_size: std::mem::size_of::<StepRequestV1>() as u32,
                        flags: 0,
                        until_ms,
                    },
                    &mut result,
                )
            }
            .0,
            0
        );
        let events = if result.events.len == 0 {
            &[]
        } else {
            unsafe { std::slice::from_raw_parts(result.events.data, result.events.len as usize) }
        };
        completed |= events.iter().any(|event: &EngineEventV1| {
            event.request_id == request_id
                && event.flags & aiperf_steppable_abi::ENGINE_EVENT_FLAG_TERMINAL != 0
                && event.terminal_status == aiperf_steppable_abi::TERMINAL_STATUS_COMPLETED
        });
        let facts = if result.request_facts.len == 0 {
            &[]
        } else {
            unsafe {
                std::slice::from_raw_parts(
                    result.request_facts.data,
                    result.request_facts.len as usize,
                )
            }
        };
        saw_facts |= facts
            .iter()
            .any(|fact| fact.request_id == request_id && fact.flags != 0);
        unsafe {
            table.release_events.expect("release events")(result.events);
            table.release_request_facts.expect("release facts")(result.request_facts);
        }
        if completed {
            break;
        }
    }
    assert!(completed, "the routed request must complete");
    assert!(
        saw_facts,
        "the routed request must publish measurement facts"
    );
    unsafe { table.destroy.expect("destroy")(handle) };

    let descriptor = unsafe { &*descriptor };
    assert_eq!(descriptor.abi_major, PluginDescriptorV1::ABI_MAJOR);
    assert_eq!(
        unsafe { CStr::from_ptr(descriptor.provider_id) }.to_bytes(),
        b"dynamo.kv-router.monolithic"
    );
}

#[test]
fn leased_compact_buffer_releases_once_at_terminal_completion() {
    let log = ReleaseLog::default();
    let (table, tail, handle) = create_leased_replay(&log);
    let hash_ids = [101_u32, 102];
    let buffer_id = register_hash_buffer(tail, handle, &hash_ids);
    let id = [51; 16];

    assert_eq!(
        submit_hash_buffer_range(tail, handle, buffer_id, id),
        StatusV1::OK
    );
    assert_eq!(log.count.load(Ordering::SeqCst), 0);
    assert!(step_to_terminal(table, handle, id));
    assert_eq!(log.count.load(Ordering::SeqCst), 1);
    assert_eq!(log.last_buffer_id.load(Ordering::SeqCst), buffer_id.0);
    assert_eq!(
        submit_hash_buffer_range(tail, handle, buffer_id, [60; 16]),
        StatusV1::INVALID_ARGUMENT
    );
    assert_eq!(log.count.load(Ordering::SeqCst), 1);

    unsafe { table.destroy.expect("destroy")(handle) };
    assert_eq!(log.count.load(Ordering::SeqCst), 1);
}

#[test]
fn leased_compact_buffer_releases_once_when_canceled() {
    let log = ReleaseLog::default();
    let (table, tail, handle) = create_leased_replay(&log);
    let hash_ids = [201_u32, 202];
    let buffer_id = register_hash_buffer(tail, handle, &hash_ids);
    let id = [52; 16];
    assert_eq!(
        submit_hash_buffer_range(tail, handle, buffer_id, id),
        StatusV1::OK
    );

    let mut event: EngineEventV1 = unsafe { std::mem::zeroed() };
    let mut canceled = 0;
    assert_eq!(
        unsafe { table.cancel.expect("cancel")(handle, &id, &mut event, &mut canceled) },
        StatusV1::OK
    );
    assert_eq!(canceled, 1);
    assert_eq!(log.count.load(Ordering::SeqCst), 1);
    assert_eq!(log.last_buffer_id.load(Ordering::SeqCst), buffer_id.0);
    assert_eq!(
        submit_hash_buffer_range(tail, handle, buffer_id, [61; 16]),
        StatusV1::INVALID_ARGUMENT
    );
    assert_eq!(log.count.load(Ordering::SeqCst), 1);

    unsafe { table.destroy.expect("destroy")(handle) };
    assert_eq!(log.count.load(Ordering::SeqCst), 1);
}

#[test]
fn submit_batch_remains_available_after_a_leased_request_reaches_terminal_state() {
    let log = ReleaseLog::default();
    let (table, tail, handle) = create_leased_replay(&log);
    let hash_ids = [211_u32, 212];
    let buffer_id = register_hash_buffer(tail, handle, &hash_ids);
    let leased_id = [62; 16];
    assert_eq!(
        submit_hash_buffer_range(tail, handle, buffer_id, leased_id),
        StatusV1::OK
    );
    assert!(step_to_terminal(table, handle, leased_id));
    assert_eq!(log.count.load(Ordering::SeqCst), 1);

    let batch_id = [63; 16];
    let (status, request_ids) = submit_direct_batch(table, handle, &[request(&[1, 2], batch_id)]);
    assert_eq!(status, StatusV1::OK);
    assert_eq!(request_ids, vec![batch_id]);

    unsafe { table.destroy.expect("destroy")(handle) };
}

#[test]
fn submit_batch_remains_atomic_after_a_leased_request_is_canceled() {
    let log = ReleaseLog::default();
    let (table, tail, handle) = create_leased_replay(&log);
    let hash_ids = [221_u32, 222];
    let buffer_id = register_hash_buffer(tail, handle, &hash_ids);
    let leased_id = [64; 16];
    assert_eq!(
        submit_hash_buffer_range(tail, handle, buffer_id, leased_id),
        StatusV1::OK
    );
    let mut event: EngineEventV1 = unsafe { std::mem::zeroed() };
    let mut canceled = 0;
    assert_eq!(
        unsafe { table.cancel.expect("cancel")(handle, &leased_id, &mut event, &mut canceled) },
        StatusV1::OK
    );
    assert_eq!(canceled, 1);
    assert_eq!(log.count.load(Ordering::SeqCst), 1);

    let candidate = [65; 16];
    let (status, request_ids) = submit_direct_batch(
        table,
        handle,
        &[request(&[3, 4], candidate), request(&[3, 4], candidate)],
    );
    assert_eq!(status, StatusV1::REJECTED);
    assert_eq!(request_ids, vec![[0; 16]; 2]);

    let (status, request_ids) = submit_direct_batch(table, handle, &[request(&[3, 4], candidate)]);
    assert_eq!(status, StatusV1::OK);
    assert_eq!(request_ids, vec![candidate]);
    assert_eq!(log.count.load(Ordering::SeqCst), 1);
    assert_eq!(
        submit_hash_buffer_range(tail, handle, buffer_id, [66; 16]),
        StatusV1::INVALID_ARGUMENT
    );

    unsafe { table.destroy.expect("destroy")(handle) };
}

#[test]
fn leased_replay_batch_preflights_a_later_router_rejection_without_commitment() {
    let log = ReleaseLog::default();
    let (table, tail, handle) = create_leased_replay(&log);
    let hash_ids = [231_u32, 232];
    let buffer_id = register_hash_buffer(tail, handle, &hash_ids);
    let leased_id = [67; 16];
    assert_eq!(
        submit_hash_buffer_range(tail, handle, buffer_id, leased_id),
        StatusV1::OK
    );
    let mut event: EngineEventV1 = unsafe { std::mem::zeroed() };
    let mut canceled = 0;
    assert_eq!(
        unsafe { table.cancel.expect("cancel")(handle, &leased_id, &mut event, &mut canceled) },
        StatusV1::OK
    );
    assert_eq!(log.count.load(Ordering::SeqCst), 1);

    let authored_id = b"synthetic-prompt";
    let mut rejected = request(&[7, 8], [69; 16]);
    rejected.flags |= REQUEST_FLAG_REPLAY_CONTEXT;
    rejected.replay_context = aiperf_steppable_abi::ReplayContextV1 {
        struct_size: std::mem::size_of::<aiperf_steppable_abi::ReplayContextV1>() as u32,
        flags: 0,
        authored_id: ByteSliceV1 {
            data: authored_id.as_ptr(),
            len: authored_id.len() as u64,
        },
        session_id: ByteSliceV1::EMPTY,
        metadata: ByteSliceV1::EMPTY,
        turn_index: 0,
        prompt_token_source: 1,
        reserved: 0,
    };
    let accepted = request(&[7, 8], [68; 16]);
    let (status, ids) = submit_direct_batch(table, handle, &[accepted, rejected]);
    assert_eq!(status, StatusV1::REJECTED);
    assert_eq!(ids, vec![[0; 16]; 2]);

    let (status, ids) = submit_direct_batch(table, handle, &[accepted]);
    assert_eq!(status, StatusV1::OK);
    assert_eq!(ids, vec![[68; 16]]);
    assert_eq!(log.count.load(Ordering::SeqCst), 1);

    unsafe { table.destroy.expect("destroy")(handle) };
}

#[test]
fn destroy_releases_each_remaining_compact_buffer_once() {
    let log = ReleaseLog::default();
    let (table, tail, handle) = create_leased_replay(&log);
    let hash_ids = [301_u32, 302];
    let buffer_id = register_hash_buffer(tail, handle, &hash_ids);
    assert_eq!(
        submit_hash_buffer_range(tail, handle, buffer_id, [53; 16]),
        StatusV1::OK
    );
    assert_eq!(log.count.load(Ordering::SeqCst), 0);

    unsafe { table.destroy.expect("destroy")(handle) };
    assert_eq!(log.count.load(Ordering::SeqCst), 1);
    assert_eq!(log.last_buffer_id.load(Ordering::SeqCst), buffer_id.0);
}

#[test]
fn leased_compact_range_validation_fails_closed() {
    let log = ReleaseLog::default();
    let (table, tail, handle) = create_leased_replay(&log);
    let hash_ids = [401_u32, 402];
    let buffer_id = register_hash_buffer(tail, handle, &hash_ids);
    let mut request_id = [99; 16];

    assert_eq!(
        unsafe {
            tail.submit_compact_hash_buffer_range
                .expect("submit compact hash-buffer range")(
                handle,
                compact_request([54; 16], U32SliceV1::EMPTY),
                HashBufferRangeV1 {
                    buffer_id,
                    offset: 1,
                    len: 2,
                },
                &mut request_id,
            )
        },
        StatusV1::INVALID_ARGUMENT
    );
    assert_eq!(request_id, [0; 16]);
    assert_eq!(log.count.load(Ordering::SeqCst), 0);

    request_id = [99; 16];
    assert_eq!(
        unsafe {
            tail.submit_compact_hash_buffer_range
                .expect("submit compact hash-buffer range")(
                handle,
                compact_request(
                    [55; 16],
                    U32SliceV1 {
                        data: hash_ids.as_ptr(),
                        len: 0,
                    },
                ),
                HashBufferRangeV1 {
                    buffer_id,
                    offset: 0,
                    len: 2,
                },
                &mut request_id,
            )
        },
        StatusV1::INVALID_ARGUMENT
    );
    assert_eq!(request_id, [0; 16]);

    request_id = [99; 16];
    assert_eq!(
        unsafe {
            tail.submit_compact_hash_buffer_range
                .expect("submit compact hash-buffer range")(
                handle,
                compact_request([55; 16], U32SliceV1::EMPTY),
                HashBufferRangeV1 {
                    buffer_id: HashBufferIdV1(buffer_id.0 + 1),
                    offset: 0,
                    len: 2,
                },
                &mut request_id,
            )
        },
        StatusV1::INVALID_ARGUMENT
    );
    assert_eq!(request_id, [0; 16]);

    assert_eq!(
        submit_hash_buffer_range(tail, handle, buffer_id, [57; 16]),
        StatusV1::OK
    );

    unsafe { table.destroy.expect("destroy")(handle) };
    assert_eq!(log.count.load(Ordering::SeqCst), 1);
}

#[test]
fn rejected_leased_submission_keeps_the_registration_without_releasing_it() {
    let log = ReleaseLog::default();
    let (table, tail, handle) = create_leased_replay(&log);
    let hash_ids = [451_u32, 452];
    let first_buffer_id = register_hash_buffer(tail, handle, &hash_ids);
    let request_id = [58; 16];
    assert_eq!(
        submit_hash_buffer_range(tail, handle, first_buffer_id, request_id),
        StatusV1::OK
    );

    let rejected_buffer_id = register_hash_buffer(tail, handle, &hash_ids);
    assert_eq!(
        submit_hash_buffer_range(tail, handle, rejected_buffer_id, request_id),
        StatusV1::REJECTED
    );
    assert_eq!(log.count.load(Ordering::SeqCst), 0);

    assert_eq!(
        submit_hash_buffer_range(tail, handle, rejected_buffer_id, [59; 16]),
        StatusV1::OK
    );
    unsafe { table.destroy.expect("destroy")(handle) };
    assert_eq!(log.count.load(Ordering::SeqCst), 2);
}

#[test]
fn lease_registration_rejects_empty_and_misaligned_buffers() {
    let log = ReleaseLog::default();
    let (table, tail, handle) = create_leased_replay(&log);
    let mut buffer_id = HashBufferIdV1::INVALID;
    assert_eq!(
        unsafe {
            tail.register_hash_buffer.expect("register hash buffer")(
                handle,
                U32SliceV1::EMPTY,
                &mut buffer_id,
            )
        },
        StatusV1::INVALID_ARGUMENT
    );
    let bytes = [0_u8; 12];
    assert_eq!(
        unsafe {
            tail.register_hash_buffer.expect("register hash buffer")(
                handle,
                U32SliceV1 {
                    data: bytes.as_ptr().add(1).cast(),
                    len: 1,
                },
                &mut buffer_id,
            )
        },
        StatusV1::INVALID_ARGUMENT
    );
    unsafe { table.destroy.expect("destroy")(handle) };
    assert_eq!(log.count.load(Ordering::SeqCst), 0);
}

#[test]
fn copied_compact_submission_remains_available() {
    let (table, handle) = create_replay();
    let hash_ids = [501_u32, 502];
    let id = [56; 16];
    let submit = unsafe { PluginVTableV1::compact_submit(table) }
        .expect("legacy copied compact submission tail");
    let mut request_id = [0; 16];
    assert_eq!(
        unsafe {
            submit(
                handle,
                compact_request(
                    id,
                    U32SliceV1 {
                        data: hash_ids.as_ptr(),
                        len: hash_ids.len() as u64,
                    },
                ),
                &mut request_id,
            )
        },
        StatusV1::OK
    );
    assert_eq!(request_id, id);
    assert!(step_to_terminal(table, handle, id));
    unsafe { table.destroy.expect("destroy")(handle) };
}

#[test]
fn ffi_rejects_slice_lengths_over_the_isize_byte_bound() {
    let (table, handle) = create_replay();
    let u32_limit = (isize::MAX as usize / std::mem::size_of::<u32>()) as u64;
    let mut request_id = [0_u8; 16];
    let mut oversized = request(&[], [60; 16]);
    oversized.tokens = U32SliceV1 {
        data: std::ptr::NonNull::<u32>::dangling().as_ptr(),
        len: u32_limit + 1,
    };
    assert_eq!(
        unsafe { table.submit.expect("submit")(handle, oversized, &mut request_id,) },
        StatusV1::INVALID_ARGUMENT
    );

    let request_value = request(&[], [61; 16]);
    let request_limit = (isize::MAX as usize / std::mem::size_of::<DirectRequestV1>()) as u64;
    assert_eq!(
        unsafe {
            table.submit_batch.expect("submit batch")(
                handle,
                DirectRequestSliceV1 {
                    data: &raw const request_value,
                    len: request_limit + 1,
                },
                RequestIdMutSliceV1 {
                    data: std::ptr::NonNull::<aiperf_steppable_abi::RequestIdV1>::dangling()
                        .as_ptr(),
                    len: request_limit + 1,
                },
            )
        },
        StatusV1::INVALID_ARGUMENT
    );

    let request_id_limit =
        (isize::MAX as usize / std::mem::size_of::<aiperf_steppable_abi::RequestIdV1>()) as u64;
    assert_eq!(
        unsafe {
            table.submit_batch.expect("submit batch")(
                handle,
                DirectRequestSliceV1::EMPTY,
                RequestIdMutSliceV1 {
                    data: std::ptr::NonNull::<aiperf_steppable_abi::RequestIdV1>::dangling()
                        .as_ptr(),
                    len: request_id_limit + 1,
                },
            )
        },
        StatusV1::INVALID_ARGUMENT
    );

    unsafe { table.destroy.expect("destroy")(handle) };
}

#[test]
fn create_rejects_an_unsupported_backend_config_version() {
    let descriptor = dynamo_steppable_provider::aiperf_steppable_plugin_v1();
    let table = unsafe { &*validate_descriptor_v1(descriptor).expect("complete V1 descriptor") };
    let mut config = dynamo_steppable_provider::BackendConfig::one_worker();
    config.version = 2;
    let payload = serde_json::to_vec(&config).expect("serializable aggregate configuration");
    let mut handle = ReplayHandleV1(std::ptr::null_mut());
    let mut error = ByteSliceV1::EMPTY;

    assert_eq!(
        unsafe {
            table.create.expect("create")(
                CreateRequestV1 {
                    struct_size: std::mem::size_of::<CreateRequestV1>() as u32,
                    flags: 0,
                    provider_payload: ByteSliceV1 {
                        data: payload.as_ptr(),
                        len: payload.len() as u64,
                    },
                },
                &mut handle,
                &mut error,
            )
        }
        .0,
        aiperf_steppable_abi::StatusV1::REJECTED.0
    );
    assert!(handle.0.is_null());
    assert!(!error.data.is_null());
    unsafe { table.release_bytes.expect("release error")(error) };
}

#[test]
fn aggregate_outer_config_uses_static_router_without_loading_nested_provider() {
    let descriptor = dynamo_steppable_provider::aiperf_steppable_plugin_v1();
    let table = unsafe { &*validate_descriptor_v1(descriptor).expect("complete V1 descriptor") };
    let mut config = dynamo_steppable_provider::BackendConfig::one_worker();
    config.dynamic_placement = Some(dynamo_steppable_provider::DynamicPlacementLocator {
        library_path: PathBuf::from("/this/path/must/not/be/opened.so"),
        selector_seed: [9; 32],
        options_namespace: b"dynamo".to_vec(),
        provider_options: vec![1, 2, 3],
        limits: dynamo_steppable_provider::DynamicPlacementLimits::default(),
    });
    let payload = serde_json::to_vec(&config).expect("outer aggregate payload");
    let mut handle = ReplayHandleV1(std::ptr::null_mut());
    let mut error = ByteSliceV1::EMPTY;

    assert_eq!(
        unsafe {
            table.create.expect("create")(
                CreateRequestV1 {
                    struct_size: std::mem::size_of::<CreateRequestV1>() as u32,
                    flags: 0,
                    provider_payload: ByteSliceV1 {
                        data: payload.as_ptr(),
                        len: payload.len() as u64,
                    },
                },
                &mut handle,
                &mut error,
            )
        }
        .0,
        0
    );
    assert!(error.data.is_null());
    unsafe { table.destroy.expect("destroy")(handle) };
}

#[test]
fn invalid_direct_request_returns_invalid_argument_without_an_id() {
    let (table, handle) = create_replay();
    let prompt = [1_u32];
    let mut malformed = request(&prompt, [40; 16]);
    malformed.struct_size = 0;
    let mut returned = [99; 16];
    assert_eq!(
        unsafe { table.submit.expect("submit")(handle, malformed, &mut returned) },
        StatusV1::INVALID_ARGUMENT
    );
    assert_eq!(returned, [0; 16]);
    unsafe { table.destroy.expect("destroy")(handle) };
}

#[test]
fn opaque_non_json_replay_metadata_is_accepted_by_the_abi_table() {
    let (table, handle) = create_replay();
    let prompt = [9_u32];
    let authored_id = b"opaque-1";
    let metadata = [0xff_u8];
    let mut opaque = request(&prompt, [45; 16]);
    opaque.flags |= REQUEST_FLAG_REPLAY_CONTEXT;
    opaque.replay_context = aiperf_steppable_abi::ReplayContextV1 {
        struct_size: std::mem::size_of::<aiperf_steppable_abi::ReplayContextV1>() as u32,
        flags: REPLAY_CONTEXT_FLAG_METADATA,
        authored_id: ByteSliceV1 {
            data: authored_id.as_ptr(),
            len: authored_id.len() as u64,
        },
        session_id: ByteSliceV1::EMPTY,
        metadata: ByteSliceV1 {
            data: metadata.as_ptr(),
            len: metadata.len() as u64,
        },
        turn_index: 0,
        prompt_token_source: 0,
        reserved: 0,
    };
    let mut returned = [0; 16];
    assert_eq!(
        unsafe { table.submit.expect("submit")(handle, opaque, &mut returned) },
        StatusV1::OK
    );
    assert_eq!(returned, [45; 16]);
    unsafe { table.destroy.expect("destroy")(handle) };
}

#[test]
fn replay_metadata_limit_rejects_before_request_commitment() {
    let (table, handle) = create_replay();
    let prompt = [10_u32];
    let authored_id = b"metadata-boundary";
    let at_limit = vec![0xff_u8; aiperf_steppable_abi::MAX_REPLAY_CONTEXT_METADATA_BYTES_V1];
    let over_limit = vec![0xff_u8; at_limit.len() + 1];
    let context = |metadata: &[u8]| aiperf_steppable_abi::ReplayContextV1 {
        struct_size: std::mem::size_of::<aiperf_steppable_abi::ReplayContextV1>() as u32,
        flags: REPLAY_CONTEXT_FLAG_METADATA,
        authored_id: ByteSliceV1 {
            data: authored_id.as_ptr(),
            len: authored_id.len() as u64,
        },
        session_id: ByteSliceV1::EMPTY,
        metadata: ByteSliceV1 {
            data: metadata.as_ptr(),
            len: metadata.len() as u64,
        },
        turn_index: 0,
        prompt_token_source: 0,
        reserved: 0,
    };

    let mut at_limit_request = request(&prompt, [46; 16]);
    at_limit_request.flags |= REQUEST_FLAG_REPLAY_CONTEXT;
    at_limit_request.replay_context = context(&at_limit);
    let mut returned = [0; 16];
    assert_eq!(
        unsafe { table.submit.expect("at-limit submit")(handle, at_limit_request, &mut returned) },
        StatusV1::OK
    );
    assert_eq!(returned, [46; 16]);

    let mut oversized_request = request(&prompt, [47; 16]);
    oversized_request.flags |= REQUEST_FLAG_REPLAY_CONTEXT;
    oversized_request.replay_context = context(&over_limit);
    returned = [99; 16];
    assert_eq!(
        unsafe {
            table.submit.expect("oversized submit")(handle, oversized_request, &mut returned)
        },
        StatusV1::INVALID_ARGUMENT
    );
    assert_eq!(returned, [0; 16]);

    assert_eq!(
        unsafe {
            table.submit.expect("submit after oversized rejection")(
                handle,
                request(&prompt, [47; 16]),
                &mut returned,
            )
        },
        StatusV1::OK
    );
    assert_eq!(returned, [47; 16]);
    unsafe { table.destroy.expect("destroy")(handle) };
}

#[test]
fn provider_cdylib_loads_through_the_fixed_steppable_entrypoint() {
    let test_binary = std::env::current_exe().expect("test binary path");
    let cdylib = test_binary
        .parent()
        .expect("test binary directory")
        .join(format!(
            "{}dynamo_steppable_provider{}",
            std::env::consts::DLL_PREFIX,
            std::env::consts::DLL_SUFFIX
        ));
    let cdylib = CString::new(cdylib.to_string_lossy().as_bytes()).expect("cdylib path");
    unsafe {
        let library = dlopen(cdylib.as_ptr(), RTLD_NOW);
        assert!(
            !library.is_null(),
            "cdylib must load: {}",
            dlerror_message()
        );
        let entry = dlsym(library, c"aiperf_steppable_plugin_v1".as_ptr());
        assert!(!entry.is_null(), "cdylib must export the V1 entrypoint");
        let entry: unsafe extern "C" fn() -> *const PluginDescriptorV1 = std::mem::transmute(entry);
        assert!(validate_descriptor_v1(entry()).is_ok());
        assert_eq!(dlclose(library), 0);
    }
}

#[test]
fn empty_exact_output_plan_is_authoritative() {
    let (table, handle) = create_replay();
    let prompt = [1_u32, 2, 3];
    let empty_plan = [];
    let id = [41; 16];
    let mut request = request(&prompt, id);
    request.flags |= REQUEST_FLAG_OUTPUT_TOKEN_IDS;
    request.max_output_tokens = 99;
    request.output_token_ids = U32SliceV1 {
        data: empty_plan.as_ptr(),
        len: 0,
    };
    let mut returned = [0; 16];
    assert_eq!(
        unsafe { table.submit.expect("submit")(handle, request, &mut returned) },
        StatusV1::OK
    );
    assert_eq!(returned, id);
    assert!(step_to_terminal(table, handle, id));
    unsafe { table.destroy.expect("destroy")(handle) };
}

#[test]
fn cancellation_and_report_use_the_abi_owned_outputs() {
    let (table, handle) = create_replay();
    let prompt = [5_u32, 6];
    let id = [42; 16];
    let mut returned = [0; 16];
    assert_eq!(
        unsafe { table.submit.expect("submit")(handle, request(&prompt, id), &mut returned) },
        StatusV1::OK
    );
    let mut event: EngineEventV1 = unsafe { std::mem::zeroed() };
    let mut canceled = 0;
    assert_eq!(
        unsafe { table.cancel.expect("cancel")(handle, &returned, &mut event, &mut canceled) },
        StatusV1::OK
    );
    assert_eq!(canceled, 1);
    assert_eq!(event.request_id, id);
    assert_eq!(
        event.terminal_status,
        aiperf_steppable_abi::TERMINAL_STATUS_CANCELED
    );

    let mut report = ByteSliceV1::EMPTY;
    assert_eq!(
        unsafe { table.take_report.expect("report")(handle, 1.0, &mut report) },
        StatusV1::OK
    );
    assert!(!report.data.is_null());
    unsafe {
        table.release_bytes.expect("release report")(report);
        table.destroy.expect("destroy")(handle);
    }
}

#[test]
fn rejected_later_duplicate_batch_has_no_prior_commitment() {
    let (table, handle) = create_replay();
    let prompt = [7_u32, 8];
    let existing = [43; 16];
    let candidate = [44; 16];
    let mut returned = [0; 16];
    assert_eq!(
        unsafe { table.submit.expect("submit")(handle, request(&prompt, existing), &mut returned) },
        StatusV1::OK
    );

    let batch = [request(&prompt, candidate), request(&prompt, existing)];
    let mut output = [[99; 16]; 2];
    assert_eq!(
        unsafe {
            table.submit_batch.expect("submit batch")(
                handle,
                DirectRequestSliceV1 {
                    data: batch.as_ptr(),
                    len: batch.len() as u64,
                },
                RequestIdMutSliceV1 {
                    data: output.as_mut_ptr(),
                    len: output.len() as u64,
                },
            )
        },
        StatusV1::REJECTED
    );
    assert_eq!(output, [[0; 16]; 2]);

    assert_eq!(
        unsafe {
            table.submit.expect("submit candidate after rejection")(
                handle,
                request(&prompt, candidate),
                &mut returned,
            )
        },
        StatusV1::OK
    );
    assert_eq!(returned, candidate);
    unsafe { table.destroy.expect("destroy")(handle) };
}

const RTLD_NOW: c_int = 2;

unsafe extern "C" {
    fn dlopen(filename: *const c_char, flags: c_int) -> *mut c_void;
    fn dlsym(handle: *mut c_void, symbol: *const c_char) -> *mut c_void;
    fn dlclose(handle: *mut c_void) -> c_int;
    fn dlerror() -> *const c_char;
}

fn dlerror_message() -> String {
    unsafe {
        let error = dlerror();
        if error.is_null() {
            "unknown dynamic-loader error".to_owned()
        } else {
            CStr::from_ptr(error).to_string_lossy().into_owned()
        }
    }
}
