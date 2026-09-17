// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::ffi::c_void;

use aiperf_steppable_abi::{
    ByteSliceV1, CAPABILITY_COMPACT_BUFFER_LEASES_V1, CompactRequestV1, CreateRequestV1,
    HashBufferIdV1, HashBufferLeaseCallbacksV1, HashBufferRangeV1, PluginVTableV1,
    PluginVTableV1Prefix, ReplayHandleV1, RequestIdV1, StatusV1, U32SliceV1,
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
    let mut prefix: Box<PluginVTableV1Prefix> = Box::new(unsafe { std::mem::zeroed() });
    prefix.struct_size = std::mem::size_of::<PluginVTableV1Prefix>() as u32;
    let raw = Box::into_raw(prefix);

    assert!(
        unsafe { PluginVTableV1::compact_buffer_leases(raw, CAPABILITY_COMPACT_BUFFER_LEASES_V1) }
            .is_none()
    );
    assert!(unsafe { PluginVTableV1::compact_submit(raw) }.is_none());

    unsafe { drop(Box::from_raw(raw)) };
}
