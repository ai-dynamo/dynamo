// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::ffi::c_void;

use aisimulate_placement_abi::{
    ByteSliceV1, PlacementCreateRequestV1, PlacementHandleV1, PlacementLimitsV1,
    PluginDescriptorV1, SchedulerIdSliceV1, StatusV1, WorkerCapacitySliceV1, WorkerCapacityV1,
    WorkerTopologySliceV1, WorkerTopologyV1, validate_descriptor_v1,
};

#[test]
fn exported_descriptor_is_a_complete_v1_plugin() {
    let descriptor = dynamo_placement_plugin::dynamo_placement_plugin_entry_v1();

    assert!(!descriptor.is_null());
    assert!(unsafe { validate_descriptor_v1(descriptor) }.is_ok());

    // Safety: the provider's descriptor is immutable process-lifetime data.
    let descriptor = unsafe { &*descriptor };
    assert_eq!(descriptor.abi_major, PluginDescriptorV1::ABI_MAJOR);
    assert!(!descriptor.provider_id.cast::<c_void>().is_null());
}

#[test]
fn provider_creates_and_destroys_a_narrow_validated_kv_router_instance() {
    let scheduler_ids = [19_u64];
    let workers = [WorkerTopologyV1 {
        worker_id: 7,
        scheduler_ids: SchedulerIdSliceV1 {
            data: scheduler_ids.as_ptr(),
            len: scheduler_ids.len() as u64,
        },
    }];
    let capacities = [WorkerCapacityV1 {
        worker_id: 7,
        total_kv_blocks: 100,
        available_kv_blocks: 100,
        max_running_requests: 1,
        flags: 0,
        reserved: 0,
    }];
    let request = PlacementCreateRequestV1 {
        struct_size: std::mem::size_of::<PlacementCreateRequestV1>() as u32,
        payload_version: 1,
        flags: 0,
        reserved: 0,
        selector_seed: [0; 32],
        workers: WorkerTopologySliceV1 {
            data: workers.as_ptr(),
            len: workers.len() as u64,
        },
        capacities: WorkerCapacitySliceV1 {
            data: capacities.as_ptr(),
            len: capacities.len() as u64,
        },
        options_namespace: ByteSliceV1::EMPTY,
        provider_options: ByteSliceV1::EMPTY,
        limits: PlacementLimitsV1 {
            max_mutations: 1,
            max_admission_results: 1,
            max_released: 1,
            max_diagnostic_bytes: 256,
        },
    };
    let descriptor = dynamo_placement_plugin::dynamo_placement_plugin_entry_v1();
    // Safety: the immutable provider descriptor has a validated V1 operation table.
    let table = unsafe { &*validate_descriptor_v1(descriptor).expect("valid descriptor") };
    let mut handle = PlacementHandleV1(std::ptr::null_mut());
    let mut error = ByteSliceV1::EMPTY;

    let status = unsafe {
        table.create.expect("required create operation")(request, &mut handle, &mut error)
    };

    assert_eq!(status, StatusV1::OK);
    assert!(!handle.0.is_null());
    assert!(error.data.is_null());
    // Safety: the handle was created by this exact V1 operation table.
    unsafe { table.destroy.expect("required destroy operation")(handle) };
}
