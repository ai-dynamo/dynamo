// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use aiperf_steppable_abi::{
    ByteSliceV1, CreateRequestV1, DirectRequestV1, EngineEventV1, PluginDescriptorV1,
    ReplayHandleV1, StepRequestV1, StepResultV1, U32SliceV1, validate_descriptor_v1,
};
use std::ffi::CStr;
use std::path::PathBuf;

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
        unsafe {
            table.release_events.expect("release events")(result.events);
            table.release_request_facts.expect("release facts")(result.request_facts);
        }
        if completed {
            break;
        }
    }
    assert!(completed, "the routed request must complete");
    unsafe { table.destroy.expect("destroy")(handle) };

    let descriptor = unsafe { &*descriptor };
    assert_eq!(descriptor.abi_major, PluginDescriptorV1::ABI_MAJOR);
    assert_eq!(
        unsafe { CStr::from_ptr(descriptor.provider_id) }.to_bytes(),
        b"dynamo.kv-router.monolithic"
    );
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
