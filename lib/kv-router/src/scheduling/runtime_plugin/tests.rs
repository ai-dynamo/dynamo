// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::process::Command;

use crate::{
    protocols::RoutingConstraints,
    scheduling::{OverlapSignals, ScheduleMode, selector::WorkerSelector},
    sequences::WorkerLoadProjection,
    test_utils::SimpleWorkerConfig,
};

use super::*;

unsafe extern "C" fn create(
    _config: ByteSliceV1,
    _router_role: u32,
    _state_out: *mut *mut c_void,
    _error_out: *mut WorkerSelectorErrorBufferV1,
) -> i32 {
    STATUS_OK
}

unsafe extern "C" fn select(
    _state: *mut c_void,
    _input: *const WorkerSelectorInputV1,
    _candidate_index_out: *mut usize,
    _error_out: *mut WorkerSelectorErrorBufferV1,
) -> i32 {
    STATUS_OK
}

unsafe extern "C" fn destroy(_state: *mut c_void) {}

fn descriptor() -> WorkerSelectorPluginV1 {
    WorkerSelectorPluginV1 {
        abi_version: ABI_VERSION_V1,
        struct_size: size_of::<WorkerSelectorPluginV1>() as u32,
        create: Some(create),
        select: Some(select),
        destroy: Some(destroy),
    }
}

#[test]
fn descriptor_validation_rejects_incompatible_plugins() {
    assert!(unsafe { validate_descriptor(std::ptr::null()) }.is_err());

    let mut invalid = descriptor();
    invalid.abi_version += 1;
    assert!(unsafe { validate_descriptor(&invalid) }.is_err());

    let mut invalid = descriptor();
    invalid.struct_size = size_of::<WorkerSelectorPluginHeaderV1>() as u32;
    assert!(unsafe { validate_descriptor(&invalid) }.is_err());

    let mut invalid = descriptor();
    invalid.create = None;
    assert!(unsafe { validate_descriptor(&invalid) }.is_err());

    let mut invalid = descriptor();
    invalid.select = None;
    assert!(unsafe { validate_descriptor(&invalid) }.is_err());

    let mut invalid = descriptor();
    invalid.destroy = None;
    assert!(unsafe { validate_descriptor(&invalid) }.is_err());
}

#[cfg(target_os = "linux")]
#[test]
fn eager_binding_rejects_unresolved_symbols() {
    let target = tempfile::tempdir().unwrap();
    let source = target.path().join("unresolved.c");
    let library = target.path().join("libunresolved.so");
    std::fs::write(
        &source,
        r#"
            extern void dynamo_intentionally_missing(void);
            __attribute__((visibility("default")))
            void dynamo_worker_selector_plugin_v1(void) {
                dynamo_intentionally_missing();
            }
        "#,
    )
    .unwrap();
    let output = Command::new("cc")
        .args(["-shared", "-fPIC", "-Wl,--allow-shlib-undefined"])
        .arg(&source)
        .arg("-o")
        .arg(&library)
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "test plugin build failed:\n{}\n{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );

    let error = unsafe { load_plugin(&library) }.err().unwrap();
    assert!(format!("{error:#}").contains("dynamo_intentionally_missing"));
}

#[test]
fn loads_real_plugin_and_delegates_to_default() {
    let manifest =
        Path::new(env!("CARGO_MANIFEST_DIR")).join("examples/custom-worker-selector/Cargo.toml");
    let target = tempfile::tempdir().unwrap();
    let output = Command::new(env!("CARGO"))
        .args(["build", "--offline", "--manifest-path"])
        .arg(&manifest)
        .arg("--target-dir")
        .arg(target.path())
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "plugin build failed:\n{}\n{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    let library = target.path().join("debug").join(format!(
        "{}custom_worker_selector{}",
        std::env::consts::DLL_PREFIX,
        std::env::consts::DLL_SUFFIX
    ));
    let plugin = |config: &[u8]| RuntimePluginConfig {
        path: library.clone(),
        config: config.to_vec(),
    };

    let workers = HashMap::from([
        (10, SimpleWorkerConfig::default()),
        (20, SimpleWorkerConfig::default()),
    ]);
    let warm_worker = WorkerWithDpRank::new(20, 0);
    let mut request = SchedulingRequest {
        mode: ScheduleMode::QueryOnly {
            request_id: Some("request-1".into()),
        },
        token_seq: None,
        isl_tokens: 65,
        lora_name: None,
        expected_output_tokens: Some(32),
        pinned_worker: None,
        allowed_worker_ids: None,
        routing_constraints: RoutingConstraints::default(),
        router_config_override: None,
        track_prefill_tokens: true,
        priority_jump: 0.0,
        strict_priority: 0,
        policy_class: None,
        session_id: Some("session-1".into()),
        overlap: OverlapSignals::default(),
        shared_cache_hits: None,
        worker_loads: Default::default(),
        resp_tx: None,
    };
    request
        .overlap
        .effective_cached_tokens
        .insert(warm_worker, 64);
    request
        .overlap
        .effective_overlap_blocks
        .insert(warm_worker, 4.0);

    let selector =
        unsafe { plugin(b"").load(RuntimePluginRouterRole::Decode, KvRouterConfig::default()) }
            .unwrap();
    assert_eq!(
        selector
            .select_worker(&workers, &request, request.eligibility(), 16)
            .unwrap(),
        warm_worker
    );

    request.worker_loads.insert(
        warm_worker,
        WorkerLoadProjection {
            active_decode_blocks: 2,
            ..Default::default()
        },
    );
    let default_config = KvRouterConfig {
        overlap_score_credit: 0.0,
        ..Default::default()
    };
    let expected = DefaultWorkerSelector::new(Some(default_config.clone()), "decode")
        .select_worker(&workers, &request, request.eligibility(), 16)
        .unwrap()
        .worker;
    let selector =
        unsafe { plugin(b"use-default").load(RuntimePluginRouterRole::Decode, default_config) }
            .unwrap();
    assert_eq!(
        selector
            .select_worker(&workers, &request, request.eligibility(), 16)
            .unwrap(),
        expected
    );
}
