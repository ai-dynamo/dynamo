// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::ffi::{CStr, CString, c_char, c_int, c_void};

use aisimulate_placement_abi::{
    AdmissionDecisionV1, BlockHashSliceV1, ByteSliceV1, CAPABILITY_LOSSLESS_KV_EVENTS_V1,
    EngineObservationV1, KvEventSliceV1, KvEventV1, KvStorageTierV1, KvStoredBlockV1,
    PlacementAdmissionV1, PlacementBatchResultV1, PlacementCreateRequestV1, PlacementHandleV1,
    PlacementLimitsV1, PlacementMetadataV1, PlacementMutationKindV1, PlacementMutationPayloadV1,
    PlacementMutationSliceV1, PlacementMutationV1, PluginDescriptorV1, PromptIdentityV1,
    SchedulerIdSliceV1, StatusV1, WorkerCapacitySliceV1, WorkerCapacityV1, WorkerTopologySliceV1,
    WorkerTopologyV1, validate_descriptor_v1,
};

#[test]
fn exported_descriptor_is_a_complete_v1_plugin() {
    let descriptor = dynamo_placement_plugin::aisimulate_placement_plugin_v1();

    assert!(!descriptor.is_null());
    assert!(unsafe { validate_descriptor_v1(descriptor) }.is_ok());

    // Safety: the provider's descriptor is immutable process-lifetime data.
    let descriptor = unsafe { &*descriptor };
    assert_eq!(descriptor.abi_major, PluginDescriptorV1::ABI_MAJOR);
    assert_ne!(
        descriptor.capabilities & CAPABILITY_LOSSLESS_KV_EVENTS_V1,
        0,
        "Dynamo's KV-aware provider must advertise the lossless event callback"
    );
    assert!(!descriptor.provider_id.cast::<c_void>().is_null());
    // Safety: descriptor validation above establishes the complete immutable table.
    assert!(unsafe { &*descriptor.vtable }.supports_lossless_kv_events());
}

#[test]
fn provider_cdylib_loads_through_the_fixed_aisimulate_entrypoint() {
    let test_binary = std::env::current_exe().expect("test binary path");
    let cdylib = test_binary
        .parent()
        .expect("test binary directory")
        .join(format!(
            "{}dynamo_placement_plugin{}",
            std::env::consts::DLL_PREFIX,
            std::env::consts::DLL_SUFFIX
        ));

    let cdylib = CString::new(cdylib.to_string_lossy().as_bytes()).expect("cdylib path");
    // Safety: the test owns the loaded handle and uses the fixed C ABI symbol
    // resolved by `DynamicPlacementPlugin` on this Unix target.
    unsafe {
        let library = dlopen(cdylib.as_ptr(), RTLD_NOW);
        assert!(
            !library.is_null(),
            "Dynamo cdylib must load: {}",
            dlerror_message()
        );
        let entry = dlsym(library, c"aisimulate_placement_plugin_v1".as_ptr());
        assert!(
            !entry.is_null(),
            "Dynamo cdylib must export AISimulate's fixed placement entrypoint"
        );
        let entry: unsafe extern "C" fn() -> *const PluginDescriptorV1 = std::mem::transmute(entry);
        assert!(validate_descriptor_v1(entry()).is_ok());
        assert_eq!(dlclose(library), 0);
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
    // Safety: `dlerror` returns a process-lifetime message pointer or null.
    unsafe {
        let error = dlerror();
        if error.is_null() {
            "unknown dynamic-loader error".to_owned()
        } else {
            CStr::from_ptr(error).to_string_lossy().into_owned()
        }
    }
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
    let descriptor = dynamo_placement_plugin::aisimulate_placement_plugin_v1();
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

#[test]
fn provider_applies_one_lossless_kv_event_to_the_router() {
    let (table, handle) = create_provider();
    let blocks = [KvStoredBlockV1 {
        sequence_hash: 101,
        token_hash: 202,
    }];
    let event = KvEventV1::stored(7, 0, KvStorageTierV1::DEVICE, 1, None, None, &blocks);
    let mut result = empty_result();

    let status = unsafe {
        table.apply_kv_events.expect("lossless KV callback")(
            handle,
            KvEventSliceV1 {
                data: &event,
                len: 1,
            },
            0.0,
            &mut result,
        )
    };

    assert_eq!(status, StatusV1::OK);
    assert_eq!(result.applied_mutations, 1);
    assert_eq!(result.admission_results.len, 0);
    unsafe { table.release_results.expect("release results")(result) };

    let local_hashes = [202_u64];
    let sequence_hashes = [101_u64];
    let admission = PlacementMutationV1 {
        struct_size: std::mem::size_of::<PlacementMutationV1>() as u32,
        kind: PlacementMutationKindV1::ADMIT,
        flags: 0,
        sequence: 2,
        now_ms: 1.0,
        payload: PlacementMutationPayloadV1 {
            admission: PlacementAdmissionV1 {
                request_id: [9; 16],
                flags: 0,
                priority: 0,
                prompt_tokens: 16,
                max_output_tokens: 1,
                prompt_identity: PromptIdentityV1 {
                    flags: PromptIdentityV1::LOCAL_BLOCK_HASHES_PRESENT
                        | PromptIdentityV1::SEQUENCE_BLOCK_HASHES_PRESENT,
                    reserved: 0,
                    materialized_token_ids: Default::default(),
                    local_block_hashes: BlockHashSliceV1 {
                        data: local_hashes.as_ptr(),
                        len: local_hashes.len() as u64,
                    },
                    sequence_block_hashes: BlockHashSliceV1 {
                        data: sequence_hashes.as_ptr(),
                        len: sequence_hashes.len() as u64,
                    },
                },
                metadata: PlacementMetadataV1::EMPTY,
                session_id: ByteSliceV1::EMPTY,
            },
        },
    };
    let admission_result = apply_one(&table, handle, &admission);
    let placement = unsafe { &*admission_result.admission_results.data }.placement;
    assert_eq!(placement.cache_sample.overlap_blocks, 1);
    unsafe {
        table.release_results.expect("release results")(admission_result);
        table.destroy.expect("destroy")(handle);
    }
}

#[test]
fn provider_reports_kv_event_prefix_when_later_event_has_unsupported_dp_rank() {
    let (table, handle) = create_provider_with_limits(PlacementLimitsV1 {
        max_mutations: 2,
        max_admission_results: 2,
        max_released: 2,
        max_diagnostic_bytes: 256,
    });
    let blocks = [KvStoredBlockV1 {
        sequence_hash: 101,
        token_hash: 202,
    }];
    let events = [
        KvEventV1::stored(7, 0, KvStorageTierV1::DEVICE, 1, None, None, &blocks),
        KvEventV1::stored(7, 1, KvStorageTierV1::DEVICE, 2, None, None, &blocks),
    ];
    let mut result = empty_result();

    let status = unsafe {
        table.apply_kv_events.expect("lossless KV callback")(
            handle,
            KvEventSliceV1 {
                data: events.as_ptr(),
                len: events.len() as u64,
            },
            0.0,
            &mut result,
        )
    };

    assert_eq!(status, StatusV1::REJECTED);
    assert_eq!(result.applied_mutations, 1);
    assert_eq!(result.pending_count, 0);

    let local_hashes = [202_u64];
    let sequence_hashes = [101_u64];
    let admission = admission_with_replay_hashes([29; 16], 1.0, &local_hashes, &sequence_hashes);
    let admission_result = apply_one(&table, handle, &admission);
    let placement = unsafe { &*admission_result.admission_results.data }.placement;
    assert_eq!(placement.cache_sample.overlap_blocks, 1);
    unsafe {
        table.release_results.expect("release results")(admission_result);
        table.destroy.expect("destroy")(handle);
    }
}

#[test]
fn provider_admits_a_materialized_prompt_and_returns_host_topology_ids() {
    let (table, handle) = create_provider();
    let tokens = [11_u32, 12, 13, 14];
    let mutation = PlacementMutationV1 {
        struct_size: std::mem::size_of::<PlacementMutationV1>() as u32,
        kind: PlacementMutationKindV1::ADMIT,
        flags: 0,
        sequence: 1,
        now_ms: 0.0,
        payload: PlacementMutationPayloadV1 {
            admission: PlacementAdmissionV1 {
                request_id: [1; 16],
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
                session_id: ByteSliceV1::EMPTY,
            },
        },
    };
    let mut result = empty_result();

    let status = unsafe {
        table.apply_batch.expect("required apply operation")(
            handle,
            PlacementMutationSliceV1 {
                data: &mutation,
                len: 1,
            },
            &mut result,
        )
    };

    assert_eq!(status, StatusV1::OK);
    assert_eq!(result.applied_mutations, 1);
    assert_eq!(result.pending_count, 0);
    assert_eq!(result.admission_results.len, 1);
    let admission = unsafe { &*result.admission_results.data };
    assert_eq!(admission.request_id, [1; 16]);
    assert_eq!(admission.decision, AdmissionDecisionV1::IMMEDIATE);
    assert_eq!(admission.placement.worker_id, 7);
    assert_eq!(admission.placement.scheduler_id, 19);

    unsafe {
        table.release_results.expect("required release operation")(result);
        table.destroy.expect("required destroy operation")(handle);
    }
}

#[test]
fn provider_queues_when_busy_then_releases_after_request_terminal() {
    let (table, handle) = create_provider();
    let first = admission_mutation([2; 16], 0.0);
    let first_result = apply_one(&table, handle, &first);
    assert_eq!(first_result.pending_count, 0);
    unsafe { table.release_results.expect("required release operation")(first_result) };

    let second = admission_mutation([3; 16], 1.0);
    let second_result = apply_one(&table, handle, &second);
    assert_eq!(second_result.pending_count, 1);
    let queued = unsafe { &*second_result.admission_results.data };
    assert_eq!(queued.decision, AdmissionDecisionV1::QUEUED);
    unsafe { table.release_results.expect("required release operation")(second_result) };

    let terminal = lifecycle_mutation(PlacementMutationKindV1::REQUEST_TERMINAL, [2; 16], 2.0);
    let released_result = apply_one(&table, handle, &terminal);
    assert_eq!(released_result.pending_count, 0);
    assert_eq!(released_result.released.len, 1);
    let released = unsafe { &*released_result.released.data };
    assert_eq!(released.request_id, [3; 16]);
    assert_eq!(released.worker_id, 7);
    assert_eq!(released.scheduler_id, 19);

    unsafe {
        table.release_results.expect("required release operation")(released_result);
        table.destroy.expect("required destroy operation")(handle);
    }
}

#[test]
fn provider_cancels_a_pending_request_without_releasing_it_later() {
    let (table, handle) = create_provider();
    let first = admission_mutation([4; 16], 0.0);
    let second = admission_mutation([5; 16], 1.0);
    unsafe {
        table.release_results.expect("required release operation")(apply_one(
            &table, handle, &first,
        ))
    };
    unsafe {
        table.release_results.expect("required release operation")(apply_one(
            &table, handle, &second,
        ))
    };

    let cancel = lifecycle_mutation(PlacementMutationKindV1::CANCEL_PENDING, [5; 16], 2.0);
    let cancelled = apply_one(&table, handle, &cancel);
    assert_eq!(cancelled.pending_count, 0);
    unsafe { table.release_results.expect("required release operation")(cancelled) };

    let terminal = lifecycle_mutation(PlacementMutationKindV1::REQUEST_TERMINAL, [4; 16], 3.0);
    let terminal_result = apply_one(&table, handle, &terminal);
    assert_eq!(terminal_result.released.len, 0);
    unsafe {
        table.release_results.expect("required release operation")(terminal_result);
        table.destroy.expect("required destroy operation")(handle);
    }
}

#[test]
fn provider_releases_pending_request_after_prefill_completion() {
    let (table, handle) = create_provider();
    let first = admission_mutation([6; 16], 0.0);
    let second = admission_mutation([7; 16], 1.0);
    unsafe {
        table.release_results.expect("required release operation")(apply_one(
            &table, handle, &first,
        ))
    };
    unsafe {
        table.release_results.expect("required release operation")(apply_one(
            &table, handle, &second,
        ))
    };

    let prefill = lifecycle_mutation(PlacementMutationKindV1::PREFILL_COMPLETED, [6; 16], 2.0);
    let result = apply_one(&table, handle, &prefill);
    assert_eq!(result.pending_count, 0);
    assert_eq!(result.released.len, 1);
    assert_eq!(unsafe { &*result.released.data }.request_id, [7; 16]);
    unsafe {
        table.release_results.expect("required release operation")(result);
        table.destroy.expect("required destroy operation")(handle);
    }
}

#[test]
fn provider_settles_topology_with_stable_worker_and_scheduler_ids() {
    let (table, handle) = create_provider();
    let first = admission_mutation([8; 16], 0.0);
    let second = admission_mutation([9; 16], 1.0);
    unsafe {
        table.release_results.expect("required release operation")(apply_one(
            &table, handle, &first,
        ))
    };
    unsafe {
        table.release_results.expect("required release operation")(apply_one(
            &table, handle, &second,
        ))
    };

    let draining = worker_mutation(PlacementMutationKindV1::WORKER_DRAINING, 7, 19, 2.0);
    let ready = worker_mutation(PlacementMutationKindV1::WORKER_READY, 8, 29, 3.0);
    unsafe {
        table.release_results.expect("required release operation")(apply_one(
            &table, handle, &draining,
        ))
    };
    unsafe {
        table.release_results.expect("required release operation")(apply_one(
            &table, handle, &ready,
        ))
    };

    let settled = PlacementMutationV1::topology_settled(4.0);
    let result = apply_one(&table, handle, &settled);
    assert_eq!(result.pending_count, 0);
    assert_eq!(result.released.len, 1);
    let released = unsafe { &*result.released.data };
    assert_eq!(released.request_id, [9; 16]);
    assert_eq!(released.worker_id, 8);
    assert_eq!(released.scheduler_id, 29);
    unsafe {
        table.release_results.expect("required release operation")(result);
        table.destroy.expect("required destroy operation")(handle);
    }
}

#[test]
fn provider_accepts_paired_hash_identity_with_typed_replay_metadata() {
    let (table, handle) = create_provider();
    let local_hashes = [101_u64];
    let sequence_hashes = [202_u64];
    let metadata = br#"{"authored_id":"request","session_id":"session","metadata":null,"prompt_token_source":"materialized"}"#;
    let mutation = PlacementMutationV1 {
        struct_size: std::mem::size_of::<PlacementMutationV1>() as u32,
        kind: PlacementMutationKindV1::ADMIT,
        flags: 0,
        sequence: 1,
        now_ms: 0.0,
        payload: PlacementMutationPayloadV1 {
            admission: PlacementAdmissionV1 {
                request_id: [10; 16],
                flags: 0,
                priority: 0,
                prompt_tokens: 4,
                max_output_tokens: 1,
                prompt_identity: PromptIdentityV1 {
                    flags: PromptIdentityV1::LOCAL_BLOCK_HASHES_PRESENT
                        | PromptIdentityV1::SEQUENCE_BLOCK_HASHES_PRESENT,
                    reserved: 0,
                    materialized_token_ids: Default::default(),
                    local_block_hashes: BlockHashSliceV1 {
                        data: local_hashes.as_ptr(),
                        len: local_hashes.len() as u64,
                    },
                    sequence_block_hashes: BlockHashSliceV1 {
                        data: sequence_hashes.as_ptr(),
                        len: sequence_hashes.len() as u64,
                    },
                },
                metadata: PlacementMetadataV1 {
                    format: aisimulate_placement_abi::AdmissionMetadataFormatV1::JSON_UTF8,
                    flags: 0,
                    bytes: ByteSliceV1 {
                        data: metadata.as_ptr(),
                        len: metadata.len() as u64,
                    },
                },
                session_id: ByteSliceV1::EMPTY,
            },
        },
    };
    let result = apply_one(&table, handle, &mutation);
    assert_eq!(result.admission_results.len, 1);
    assert_eq!(
        unsafe { &*result.admission_results.data }.decision,
        AdmissionDecisionV1::IMMEDIATE
    );
    unsafe {
        table.release_results.expect("required release operation")(result);
        table.destroy.expect("required destroy operation")(handle);
    }
}

#[test]
fn provider_rejects_unsupported_observations_with_a_bounded_diagnostic() {
    let (table, handle) = create_provider();
    let mutation = unsupported_engine_observation(0.0);
    let mut result = empty_result();
    let status = unsafe {
        table.apply_batch.expect("required apply operation")(
            handle,
            PlacementMutationSliceV1 {
                data: &mutation,
                len: 1,
            },
            &mut result,
        )
    };
    assert_eq!(status, StatusV1::UNSUPPORTED);
    assert_eq!(result.applied_mutations, 0);
    let mut error = ByteSliceV1::EMPTY;
    assert_eq!(
        unsafe { table.last_error.expect("required last error operation")(handle, &mut error) },
        StatusV1::OK
    );
    assert!(error.len > 0);
    assert!(error.len <= 256);
    unsafe {
        table
            .release_bytes
            .expect("required byte release operation")(error);
        table.destroy.expect("required destroy operation")(handle);
    }
}

#[test]
fn provider_reports_the_committed_prefix_before_an_unsupported_observation() {
    let (table, handle) = create_provider();
    let admission = admission_mutation([11; 16], 0.0);
    let observation = unsupported_engine_observation(1.0);
    let mutations = [admission, observation];
    let mut result = empty_result();
    let status = unsafe {
        table.apply_batch.expect("required apply operation")(
            handle,
            PlacementMutationSliceV1 {
                data: mutations.as_ptr(),
                len: mutations.len() as u64,
            },
            &mut result,
        )
    };
    assert_eq!(status, StatusV1::UNSUPPORTED);
    assert_eq!(result.applied_mutations, 1);
    assert_eq!(result.pending_count, 0);
    unsafe { table.destroy.expect("required destroy operation")(handle) };
}

#[test]
fn provider_reports_admission_overflow_as_a_committed_batch() {
    let (table, handle) = create_provider_with_limits(PlacementLimitsV1 {
        max_mutations: 2,
        max_admission_results: 1,
        max_released: 2,
        max_diagnostic_bytes: 256,
    });
    let mutations = [
        admission_mutation([12; 16], 0.0),
        admission_mutation([13; 16], 1.0),
    ];
    let mut result = empty_result();

    let status = unsafe {
        table.apply_batch.expect("required apply operation")(
            handle,
            PlacementMutationSliceV1 {
                data: mutations.as_ptr(),
                len: mutations.len() as u64,
            },
            &mut result,
        )
    };

    assert_eq!(status, StatusV1::REJECTED);
    assert_eq!(result.applied_mutations, 2);
    assert_eq!(result.pending_count, 1);
    unsafe { table.destroy.expect("required destroy operation")(handle) };
}

#[test]
fn provider_reports_release_overflow_as_a_committed_batch() {
    let (table, handle) = create_provider_with_capacity_and_limits(
        2,
        PlacementLimitsV1 {
            max_mutations: 4,
            max_admission_results: 4,
            max_released: 1,
            max_diagnostic_bytes: 256,
        },
    );
    let admissions = [
        admission_mutation([14; 16], 0.0),
        admission_mutation([15; 16], 1.0),
        admission_mutation([16; 16], 2.0),
        admission_mutation([17; 16], 3.0),
    ];
    let mut admission_result = empty_result();
    assert_eq!(
        unsafe {
            table.apply_batch.expect("required apply operation")(
                handle,
                PlacementMutationSliceV1 {
                    data: admissions.as_ptr(),
                    len: admissions.len() as u64,
                },
                &mut admission_result,
            )
        },
        StatusV1::OK
    );
    unsafe { table.release_results.expect("required release operation")(admission_result) };

    let terminals = [
        lifecycle_mutation(PlacementMutationKindV1::REQUEST_TERMINAL, [14; 16], 4.0),
        lifecycle_mutation(PlacementMutationKindV1::REQUEST_TERMINAL, [15; 16], 5.0),
    ];
    let mut result = empty_result();
    let status = unsafe {
        table.apply_batch.expect("required apply operation")(
            handle,
            PlacementMutationSliceV1 {
                data: terminals.as_ptr(),
                len: terminals.len() as u64,
            },
            &mut result,
        )
    };

    assert_eq!(status, StatusV1::REJECTED);
    assert_eq!(result.applied_mutations, 2);
    assert_eq!(result.pending_count, 1);
    unsafe { table.destroy.expect("required destroy operation")(handle) };
}

#[test]
fn provider_rejects_heterogeneous_or_partially_available_capacity() {
    let scheduler_ids = [19_u64, 29];
    let workers = [
        worker_topology(7, &scheduler_ids[..1]),
        worker_topology(8, &scheduler_ids[1..]),
    ];
    let heterogeneous = [
        worker_capacity(7, 100, 100, 1),
        worker_capacity(8, 200, 200, 1),
    ];
    let partially_available = [worker_capacity(7, 100, 75, 1)];
    let one_worker = [worker_topology(7, &scheduler_ids[..1])];

    assert_create_rejected(&workers, &heterogeneous);
    assert_create_rejected(&one_worker, &partially_available);
}

#[test]
fn provider_rejects_multiple_scheduler_ids_per_worker() {
    let scheduler_ids = [19_u64, 20];
    let workers = [worker_topology(7, &scheduler_ids)];
    let capacities = [worker_capacity(7, 100, 100, 1)];

    assert_create_rejected(&workers, &capacities);
}

#[test]
fn provider_rejects_one_sided_replay_hash_flags_even_with_tokens() {
    let (table, handle) = create_provider();
    let tokens = [11_u32, 12, 13, 14];
    let hashes = [101_u64];

    for flags in [
        PromptIdentityV1::LOCAL_BLOCK_HASHES_PRESENT,
        PromptIdentityV1::SEQUENCE_BLOCK_HASHES_PRESENT,
    ] {
        let mutation = PlacementMutationV1 {
            struct_size: std::mem::size_of::<PlacementMutationV1>() as u32,
            kind: PlacementMutationKindV1::ADMIT,
            flags: 0,
            sequence: 1,
            now_ms: 0.0,
            payload: PlacementMutationPayloadV1 {
                admission: PlacementAdmissionV1 {
                    request_id: [18; 16],
                    flags: 0,
                    priority: 0,
                    prompt_tokens: tokens.len() as u64,
                    max_output_tokens: 1,
                    prompt_identity: PromptIdentityV1 {
                        flags: PromptIdentityV1::MATERIALIZED_TOKEN_IDS_PRESENT | flags,
                        reserved: 0,
                        materialized_token_ids: aisimulate_placement_abi::TokenIdSliceV1 {
                            data: tokens.as_ptr(),
                            len: tokens.len() as u64,
                        },
                        local_block_hashes: BlockHashSliceV1 {
                            data: hashes.as_ptr(),
                            len: hashes.len() as u64,
                        },
                        sequence_block_hashes: BlockHashSliceV1 {
                            data: hashes.as_ptr(),
                            len: hashes.len() as u64,
                        },
                    },
                    metadata: PlacementMetadataV1::EMPTY,
                    session_id: ByteSliceV1::EMPTY,
                },
            },
        };
        let mut result = empty_result();

        let status = unsafe {
            table.apply_batch.expect("required apply operation")(
                handle,
                PlacementMutationSliceV1 {
                    data: &mutation,
                    len: 1,
                },
                &mut result,
            )
        };

        assert_eq!(status, StatusV1::INVALID_ARGUMENT);
        assert_eq!(result.applied_mutations, 0);
    }

    unsafe { table.destroy.expect("required destroy operation")(handle) };
}

fn create_provider() -> (aisimulate_placement_abi::PluginVTableV1, PlacementHandleV1) {
    let scheduler_ids = [19_u64];
    let workers = [worker_topology(7, &scheduler_ids)];
    let capacities = [worker_capacity(7, 100, 100, 1)];
    create_provider_with_config(
        &workers,
        &capacities,
        PlacementLimitsV1 {
            max_mutations: 8,
            max_admission_results: 8,
            max_released: 8,
            max_diagnostic_bytes: 256,
        },
    )
}

fn create_provider_with_limits(
    limits: PlacementLimitsV1,
) -> (aisimulate_placement_abi::PluginVTableV1, PlacementHandleV1) {
    create_provider_with_capacity_and_limits(1, limits)
}

fn create_provider_with_capacity_and_limits(
    max_running_requests: u64,
    limits: PlacementLimitsV1,
) -> (aisimulate_placement_abi::PluginVTableV1, PlacementHandleV1) {
    let scheduler_ids = [19_u64];
    let workers = [worker_topology(7, &scheduler_ids)];
    let capacities = [worker_capacity(7, 100, 100, max_running_requests)];
    create_provider_with_config(&workers, &capacities, limits)
}

fn assert_create_rejected(workers: &[WorkerTopologyV1], capacities: &[WorkerCapacityV1]) {
    let descriptor = dynamo_placement_plugin::aisimulate_placement_plugin_v1();
    let table = unsafe { *validate_descriptor_v1(descriptor).expect("valid descriptor") };
    let mut handle = PlacementHandleV1(std::ptr::null_mut());
    let mut error = ByteSliceV1::EMPTY;
    let request = create_request(workers, capacities, default_limits());

    let status = unsafe {
        table.create.expect("required create operation")(request, &mut handle, &mut error)
    };

    assert_eq!(status, StatusV1::REJECTED);
    assert!(handle.0.is_null());
    assert!(error.len > 0);
    unsafe {
        table
            .release_bytes
            .expect("required byte release operation")(error)
    };
}

fn create_provider_with_config(
    workers: &[WorkerTopologyV1],
    capacities: &[WorkerCapacityV1],
    limits: PlacementLimitsV1,
) -> (aisimulate_placement_abi::PluginVTableV1, PlacementHandleV1) {
    let descriptor = dynamo_placement_plugin::aisimulate_placement_plugin_v1();
    let table = unsafe { *validate_descriptor_v1(descriptor).expect("valid descriptor") };
    let mut handle = PlacementHandleV1(std::ptr::null_mut());
    let mut error = ByteSliceV1::EMPTY;
    let request = create_request(workers, capacities, limits);

    let status = unsafe {
        table.create.expect("required create operation")(request, &mut handle, &mut error)
    };
    assert_eq!(status, StatusV1::OK);
    assert!(!handle.0.is_null());
    assert!(error.data.is_null());
    (table, handle)
}

fn create_request(
    workers: &[WorkerTopologyV1],
    capacities: &[WorkerCapacityV1],
    limits: PlacementLimitsV1,
) -> PlacementCreateRequestV1 {
    PlacementCreateRequestV1 {
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
        limits,
    }
}

fn default_limits() -> PlacementLimitsV1 {
    PlacementLimitsV1 {
        max_mutations: 8,
        max_admission_results: 8,
        max_released: 8,
        max_diagnostic_bytes: 256,
    }
}

fn worker_topology(worker_id: u64, scheduler_ids: &[u64]) -> WorkerTopologyV1 {
    WorkerTopologyV1 {
        worker_id,
        scheduler_ids: SchedulerIdSliceV1 {
            data: scheduler_ids.as_ptr(),
            len: scheduler_ids.len() as u64,
        },
    }
}

fn worker_capacity(
    worker_id: u64,
    total_kv_blocks: u64,
    available_kv_blocks: u64,
    max_running_requests: u64,
) -> WorkerCapacityV1 {
    WorkerCapacityV1 {
        worker_id,
        total_kv_blocks,
        available_kv_blocks,
        max_running_requests,
        flags: 0,
        reserved: 0,
    }
}

fn empty_result() -> PlacementBatchResultV1 {
    PlacementBatchResultV1 {
        struct_size: 0,
        flags: 0,
        applied_mutations: 0,
        pending_count: 0,
        admission_results: aisimulate_placement_abi::PlacementResultSliceV1 {
            data: std::ptr::null(),
            len: 0,
        },
        released: aisimulate_placement_abi::PlacementSliceV1 {
            data: std::ptr::null(),
            len: 0,
        },
        diagnostics: aisimulate_placement_abi::PlacementDiagnosticSliceV1 {
            data: std::ptr::null(),
            len: 0,
        },
    }
}

fn apply_one(
    table: &aisimulate_placement_abi::PluginVTableV1,
    handle: PlacementHandleV1,
    mutation: &PlacementMutationV1,
) -> PlacementBatchResultV1 {
    let mut result = empty_result();
    let status = unsafe {
        table.apply_batch.expect("required apply operation")(
            handle,
            PlacementMutationSliceV1 {
                data: mutation,
                len: 1,
            },
            &mut result,
        )
    };
    assert_eq!(status, StatusV1::OK);
    assert_eq!(result.applied_mutations, 1);
    result
}

fn admission_mutation(request_id: [u8; 16], now_ms: f64) -> PlacementMutationV1 {
    static TOKENS: [u32; 4] = [11, 12, 13, 14];
    PlacementMutationV1 {
        struct_size: std::mem::size_of::<PlacementMutationV1>() as u32,
        kind: PlacementMutationKindV1::ADMIT,
        flags: 0,
        sequence: 1,
        now_ms,
        payload: PlacementMutationPayloadV1 {
            admission: PlacementAdmissionV1 {
                request_id,
                flags: 0,
                priority: 0,
                prompt_tokens: TOKENS.len() as u64,
                max_output_tokens: 1,
                prompt_identity: PromptIdentityV1 {
                    flags: PromptIdentityV1::MATERIALIZED_TOKEN_IDS_PRESENT,
                    reserved: 0,
                    materialized_token_ids: aisimulate_placement_abi::TokenIdSliceV1 {
                        data: TOKENS.as_ptr(),
                        len: TOKENS.len() as u64,
                    },
                    ..PromptIdentityV1::OMITTED
                },
                metadata: PlacementMetadataV1::EMPTY,
                session_id: ByteSliceV1::EMPTY,
            },
        },
    }
}

fn admission_with_replay_hashes(
    request_id: [u8; 16],
    now_ms: f64,
    local_hashes: &[u64],
    sequence_hashes: &[u64],
) -> PlacementMutationV1 {
    PlacementMutationV1 {
        struct_size: std::mem::size_of::<PlacementMutationV1>() as u32,
        kind: PlacementMutationKindV1::ADMIT,
        flags: 0,
        sequence: 1,
        now_ms,
        payload: PlacementMutationPayloadV1 {
            admission: PlacementAdmissionV1 {
                request_id,
                flags: 0,
                priority: 0,
                prompt_tokens: 16,
                max_output_tokens: 1,
                prompt_identity: PromptIdentityV1 {
                    flags: PromptIdentityV1::LOCAL_BLOCK_HASHES_PRESENT
                        | PromptIdentityV1::SEQUENCE_BLOCK_HASHES_PRESENT,
                    reserved: 0,
                    materialized_token_ids: Default::default(),
                    local_block_hashes: BlockHashSliceV1 {
                        data: local_hashes.as_ptr(),
                        len: local_hashes.len() as u64,
                    },
                    sequence_block_hashes: BlockHashSliceV1 {
                        data: sequence_hashes.as_ptr(),
                        len: sequence_hashes.len() as u64,
                    },
                },
                metadata: PlacementMetadataV1::EMPTY,
                session_id: ByteSliceV1::EMPTY,
            },
        },
    }
}

fn lifecycle_mutation(
    kind: PlacementMutationKindV1,
    request_id: [u8; 16],
    now_ms: f64,
) -> PlacementMutationV1 {
    PlacementMutationV1 {
        struct_size: std::mem::size_of::<PlacementMutationV1>() as u32,
        kind,
        flags: 0,
        sequence: 1,
        now_ms,
        payload: PlacementMutationPayloadV1 {
            request_lifecycle: aisimulate_placement_abi::RequestLifecycleV1 {
                request_id,
                flags: 0,
                reserved: 0,
            },
        },
    }
}

fn worker_mutation(
    kind: PlacementMutationKindV1,
    worker_id: u64,
    scheduler_id: u64,
    now_ms: f64,
) -> PlacementMutationV1 {
    let scheduler_ids = Box::leak(Box::new([scheduler_id]));
    PlacementMutationV1 {
        struct_size: std::mem::size_of::<PlacementMutationV1>() as u32,
        kind,
        flags: 0,
        sequence: 1,
        now_ms,
        payload: PlacementMutationPayloadV1 {
            worker: WorkerTopologyV1 {
                worker_id,
                scheduler_ids: SchedulerIdSliceV1 {
                    data: scheduler_ids.as_ptr(),
                    len: scheduler_ids.len() as u64,
                },
            },
        },
    }
}

fn unsupported_engine_observation(now_ms: f64) -> PlacementMutationV1 {
    PlacementMutationV1 {
        struct_size: std::mem::size_of::<PlacementMutationV1>() as u32,
        kind: PlacementMutationKindV1::OBSERVE_ENGINE,
        flags: 0,
        sequence: 1,
        now_ms,
        payload: PlacementMutationPayloadV1 {
            engine: EngineObservationV1 {
                worker_id: 7,
                scheduler_id: 19,
                request_id: [0; 16],
                event_kind: 1,
                flags: 0,
                value: 0,
            },
        },
    }
}
