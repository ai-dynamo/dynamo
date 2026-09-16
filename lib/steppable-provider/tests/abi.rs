// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use aiperf_steppable_abi::{
    ByteSliceV1, CreateRequestV1, DirectRequestSliceV1, DirectRequestV1, EngineEventV1,
    PluginDescriptorV1, REQUEST_FLAG_OUTPUT_TOKEN_IDS, REQUEST_FLAG_UUID, ReplayHandleV1,
    RequestIdMutSliceV1, StatusV1, StepRequestV1, StepResultV1, U32SliceV1, validate_descriptor_v1,
};
use std::ffi::{CStr, CString, c_char, c_int, c_void};
use std::path::PathBuf;

fn create_replay_with(
    config: dynamo_steppable_provider::BackendConfig,
) -> (
    &'static aiperf_steppable_abi::PluginVTableV1,
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
    &'static aiperf_steppable_abi::PluginVTableV1,
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
    table: &aiperf_steppable_abi::PluginVTableV1,
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

#[test]
fn contended_admission_matches_the_dynamic_kv_router_provider() {
    use aisimulate_placement_abi::{
        AdmissionDecisionV1, ByteSliceV1 as PlacementBytes, PlacementAdmissionV1,
        PlacementBatchResultV1, PlacementCreateRequestV1, PlacementHandleV1, PlacementLimitsV1,
        PlacementMetadataV1, PlacementMutationKindV1, PlacementMutationPayloadV1,
        PlacementMutationSliceV1, PlacementMutationV1, PromptIdentityV1, SchedulerIdSliceV1,
        WorkerCapacitySliceV1, WorkerCapacityV1, WorkerTopologySliceV1, WorkerTopologyV1,
        validate_descriptor_v1 as validate_placement_descriptor,
    };

    let scheduler_ids = [0_u64];
    let workers = [WorkerTopologyV1 {
        worker_id: 0,
        scheduler_ids: SchedulerIdSliceV1 {
            data: scheduler_ids.as_ptr(),
            len: 1,
        },
    }];
    let capacities = [WorkerCapacityV1 {
        worker_id: 0,
        total_kv_blocks: 1_000,
        available_kv_blocks: 1_000,
        max_running_requests: 1,
        flags: 0,
        reserved: 0,
    }];
    let descriptor = dynamo_placement_plugin::aisimulate_placement_plugin_v1();
    let placement = unsafe {
        &*validate_placement_descriptor(descriptor).expect("dynamic placement descriptor")
    };
    let mut dynamic_handle = PlacementHandleV1(std::ptr::null_mut());
    let mut dynamic_error = PlacementBytes::EMPTY;
    assert!(unsafe {
        placement.create.expect("placement create")(
            PlacementCreateRequestV1 {
                struct_size: std::mem::size_of::<PlacementCreateRequestV1>() as u32,
                payload_version: 1,
                flags: 0,
                reserved: 0,
                selector_seed: [0; 32],
                workers: WorkerTopologySliceV1 {
                    data: workers.as_ptr(),
                    len: 1,
                },
                capacities: WorkerCapacitySliceV1 {
                    data: capacities.as_ptr(),
                    len: 1,
                },
                options_namespace: PlacementBytes::EMPTY,
                provider_options: PlacementBytes::EMPTY,
                limits: PlacementLimitsV1 {
                    max_mutations: 1,
                    max_admission_results: 1,
                    max_released: 1,
                    max_diagnostic_bytes: 0,
                },
            },
            &mut dynamic_handle,
            &mut dynamic_error,
        )
        .0 == 0
    });
    let tokens = [1_u32, 2, 3, 4];
    let admission = |id, now_ms| PlacementMutationV1 {
        struct_size: std::mem::size_of::<PlacementMutationV1>() as u32,
        kind: PlacementMutationKindV1::ADMIT,
        flags: 0,
        sequence: 0,
        now_ms,
        payload: PlacementMutationPayloadV1 {
            admission: PlacementAdmissionV1 {
                request_id: id,
                flags: 0,
                priority: 0,
                prompt_tokens: tokens.len() as u64,
                max_output_tokens: 1,
                prompt_identity: PromptIdentityV1 {
                    flags: PromptIdentityV1::MATERIALIZED_TOKEN_IDS_PRESENT,
                    reserved: 0,
                    materialized_token_ids: aisimulate_placement_abi::TokenIdSliceV1 {
                        data: tokens.as_ptr(),
                        len: tokens.len() as u64,
                    },
                    ..PromptIdentityV1::OMITTED
                },
                metadata: PlacementMetadataV1::EMPTY,
                session_id: PlacementBytes::EMPTY,
            },
        },
    };
    for (id, expect) in [
        ([51; 16], AdmissionDecisionV1::IMMEDIATE),
        ([52; 16], AdmissionDecisionV1::QUEUED),
    ] {
        let mutation = admission(id, 0.0);
        let mut result: PlacementBatchResultV1 = unsafe { std::mem::zeroed() };
        assert!(unsafe {
            placement.apply_batch.expect("placement apply")(
                dynamic_handle,
                PlacementMutationSliceV1 {
                    data: &mutation,
                    len: 1,
                },
                &mut result,
            )
            .0 == 0
        });
        assert_eq!(unsafe { &*result.admission_results.data }.decision, expect);
        unsafe { placement.release_results.expect("placement release")(result) };
    }

    let mut config = dynamo_steppable_provider::BackendConfig::one_worker();
    config.engine.rank.max_num_seqs = 1;
    let (table, handle) = create_replay_with(config);
    for id in [[51; 16], [52; 16]] {
        let mut returned = [0; 16];
        assert_eq!(
            unsafe {
                table.submit.expect("monolithic submit")(
                    handle,
                    request(&tokens, id),
                    &mut returned,
                )
            },
            StatusV1::OK
        );
    }
    let mut state = aiperf_steppable_abi::ReplayStateV1::EMPTY;
    assert_eq!(
        unsafe { table.state.expect("state")(handle, &mut state) },
        StatusV1::OK
    );
    assert_eq!(
        state.in_flight, 2,
        "the second contended request remains router-owned"
    );
    unsafe {
        table.destroy.expect("destroy")(handle);
        placement.destroy.expect("placement destroy")(dynamic_handle);
    }
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
