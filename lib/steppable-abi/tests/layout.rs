// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::ffi::c_void;

use aiperf_steppable_abi::{
    ByteSliceV1, CAPABILITY_COMPACT_BUFFER_LEASES_V1, CompactRequestV1, CreateRequestV1,
    DirectRequestSliceV1, DirectRequestV1, EngineEventSliceV1, HashBufferIdV1,
    HashBufferLeaseCallbacksV1, HashBufferRangeV1, PluginDescriptorV1, PluginVTableV1,
    PluginVTableV1Prefix, ReplayHandleV1, ReplayStateV1, RequestFactSliceV1, RequestIdMutSliceV1,
    RequestIdV1, SlaThresholdsV1, StatusV1, StepRequestV1, StepResultV1, U32SliceV1,
    validate_descriptor_v1,
};

unsafe extern "C" fn release_hash_buffer(_: *mut c_void, _: HashBufferIdV1) {}

unsafe extern "C" fn create_with_hash_buffer_leases(
    _: CreateRequestV1,
    _: HashBufferLeaseCallbacksV1,
    _: *mut ReplayHandleV1,
    _: *mut ByteSliceV1,
) -> StatusV1 {
    StatusV1::OK
}

unsafe extern "C" fn register_hash_buffer(
    _: ReplayHandleV1,
    _: U32SliceV1,
    _: *mut HashBufferIdV1,
) -> StatusV1 {
    StatusV1::OK
}

unsafe extern "C" fn submit_compact_hash_buffer_range(
    _: ReplayHandleV1,
    _: CompactRequestV1,
    _: HashBufferRangeV1,
    _: *mut RequestIdV1,
) -> StatusV1 {
    StatusV1::OK
}

unsafe extern "C" fn prefix_create(
    _: CreateRequestV1,
    _: *mut ReplayHandleV1,
    _: *mut ByteSliceV1,
) -> StatusV1 {
    StatusV1::OK
}

unsafe extern "C" fn prefix_submit(
    _: ReplayHandleV1,
    _: DirectRequestV1,
    _: *mut RequestIdV1,
) -> StatusV1 {
    StatusV1::OK
}

unsafe extern "C" fn prefix_submit_batch(
    _: ReplayHandleV1,
    _: DirectRequestSliceV1,
    _: RequestIdMutSliceV1,
) -> StatusV1 {
    StatusV1::OK
}

unsafe extern "C" fn prefix_cancel(
    _: ReplayHandleV1,
    _: *const RequestIdV1,
    _: *mut aiperf_steppable_abi::EngineEventV1,
    _: *mut u8,
) -> StatusV1 {
    StatusV1::OK
}

unsafe extern "C" fn prefix_cancel_batch(
    _: ReplayHandleV1,
    _: aiperf_steppable_abi::RequestIdSliceV1,
    _: *mut EngineEventSliceV1,
) -> StatusV1 {
    StatusV1::OK
}

unsafe extern "C" fn prefix_step(
    _: ReplayHandleV1,
    _: StepRequestV1,
    _: *mut StepResultV1,
) -> StatusV1 {
    StatusV1::OK
}

unsafe extern "C" fn prefix_take_report(
    _: ReplayHandleV1,
    _: f64,
    _: *mut ByteSliceV1,
) -> StatusV1 {
    StatusV1::OK
}

unsafe extern "C" fn prefix_release_bytes(_: ByteSliceV1) {}

unsafe extern "C" fn prefix_release_events(_: EngineEventSliceV1) {}

unsafe extern "C" fn prefix_release_request_facts(_: RequestFactSliceV1) {}

unsafe extern "C" fn prefix_state(_: ReplayHandleV1, _: *mut ReplayStateV1) -> StatusV1 {
    StatusV1::OK
}

unsafe extern "C" fn prefix_advance_now(_: ReplayHandleV1, _: f64) -> StatusV1 {
    StatusV1::OK
}

unsafe extern "C" fn prefix_set_capture(_: ReplayHandleV1, _: u8) -> StatusV1 {
    StatusV1::OK
}

unsafe extern "C" fn prefix_set_sla(_: ReplayHandleV1, _: SlaThresholdsV1) -> StatusV1 {
    StatusV1::OK
}

unsafe extern "C" fn prefix_last_error(_: ReplayHandleV1, _: *mut ByteSliceV1) -> StatusV1 {
    StatusV1::OK
}

unsafe extern "C" fn prefix_destroy(_: ReplayHandleV1) {}

#[test]
fn compact_buffer_lease_records_validate_ids_ranges_and_callbacks() {
    let valid = HashBufferRangeV1 {
        buffer_id: HashBufferIdV1(7),
        offset: 3,
        len: 4,
    };
    assert!(valid.buffer_id.is_valid());
    assert!(valid.is_valid());
    assert!(
        !HashBufferRangeV1 {
            buffer_id: HashBufferIdV1::INVALID,
            offset: 0,
            len: 4,
        }
        .is_valid()
    );
    assert!(
        !HashBufferRangeV1 {
            buffer_id: HashBufferIdV1(7),
            offset: u64::MAX,
            len: 1,
        }
        .is_valid()
    );
    assert!(
        HashBufferLeaseCallbacksV1 {
            struct_size: std::mem::size_of::<HashBufferLeaseCallbacksV1>() as u32,
            flags: 0,
            context: std::ptr::null_mut(),
            release_hash_buffer: Some(release_hash_buffer),
        }
        .has_release_callback()
    );
}

#[test]
fn compact_buffer_lease_tail_is_complete_and_capability_gated() {
    let table = PluginVTableV1 {
        struct_size: std::mem::size_of::<PluginVTableV1>() as u32,
        flags: 0,
        create: None,
        submit: None,
        submit_batch: None,
        cancel: None,
        cancel_batch: None,
        step: None,
        take_report: None,
        release_bytes: None,
        release_events: None,
        release_request_facts: None,
        state: None,
        advance_now_ms: None,
        set_capture_per_request: None,
        set_sla_thresholds: None,
        last_error: None,
        destroy: None,
        submit_compact: None,
        create_with_hash_buffer_leases: Some(create_with_hash_buffer_leases),
        register_hash_buffer: Some(register_hash_buffer),
        submit_compact_hash_buffer_range: Some(submit_compact_hash_buffer_range),
    };
    assert_eq!(
        PluginVTableV1::COMPACT_SUBMIT_SIZE,
        std::mem::offset_of!(PluginVTableV1, create_with_hash_buffer_leases)
    );
    assert_eq!(
        PluginVTableV1::COMPACT_BUFFER_LEASES_SIZE,
        std::mem::size_of::<PluginVTableV1>()
    );
    let prefix = (&raw const table).cast();
    assert!(
        unsafe {
            PluginVTableV1::compact_buffer_leases(prefix, CAPABILITY_COMPACT_BUFFER_LEASES_V1)
        }
        .is_some()
    );
    assert!(unsafe { PluginVTableV1::compact_buffer_leases(prefix, 0) }.is_none());
}

#[test]
fn legacy_v1_prefix_allocation_is_not_read_as_a_newer_tail() {
    // Allocate only the legacy prefix. A host must not inspect optional fields
    // after the allocation merely because a capability bit was supplied.
    let prefix: Box<PluginVTableV1Prefix> = Box::new(PluginVTableV1Prefix {
        struct_size: std::mem::size_of::<PluginVTableV1Prefix>() as u32,
        flags: 0,
        create: Some(prefix_create),
        submit: Some(prefix_submit),
        submit_batch: Some(prefix_submit_batch),
        cancel: Some(prefix_cancel),
        cancel_batch: Some(prefix_cancel_batch),
        step: Some(prefix_step),
        take_report: Some(prefix_take_report),
        release_bytes: Some(prefix_release_bytes),
        release_events: Some(prefix_release_events),
        release_request_facts: Some(prefix_release_request_facts),
        state: Some(prefix_state),
        advance_now_ms: Some(prefix_advance_now),
        set_capture_per_request: Some(prefix_set_capture),
        set_sla_thresholds: Some(prefix_set_sla),
        last_error: Some(prefix_last_error),
        destroy: Some(prefix_destroy),
    });
    let raw = Box::into_raw(prefix);

    let provider_id = c"layout-test";
    let descriptor = Box::new(PluginDescriptorV1 {
        abi_major: 1,
        abi_minor: 0,
        struct_size: std::mem::size_of::<PluginDescriptorV1>() as u32,
        flags: 0,
        capabilities: 0,
        provider_id: provider_id.as_ptr(),
        vtable: raw.cast(),
    });
    let descriptor_raw = Box::into_raw(descriptor);

    assert_eq!(
        unsafe { validate_descriptor_v1(descriptor_raw) },
        Ok(raw.cast_const())
    );

    assert!(
        unsafe { PluginVTableV1::compact_buffer_leases(raw, CAPABILITY_COMPACT_BUFFER_LEASES_V1) }
            .is_none()
    );
    assert!(unsafe { PluginVTableV1::compact_submit(raw) }.is_none());

    unsafe {
        drop(Box::from_raw(descriptor_raw));
        drop(Box::from_raw(raw));
    }
}
