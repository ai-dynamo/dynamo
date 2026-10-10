// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use super::*;
use std::process::{Command, Stdio};
use std::time::{Duration, Instant};

fn run_child(scenario: &str) {
    let mut child = Command::new(std::env::current_exe().unwrap())
        .args(["--exact", "tests::process_lifecycle_child", "--nocapture"])
        .env("DYNAMO_C_API_TEST_SCENARIO", scenario)
        .env("RUST_LOG", "error")
        .env("DYN_DISCOVERY_BACKEND", "mem")
        .env("DYN_REQUEST_PLANE", "tcp")
        .env("DYN_RESPONSE_PLANE", "tcp")
        .env("DYN_EVENT_PLANE", "zmq")
        .env("DYN_RUNTIME_NUM_WORKER_THREADS", "2")
        .env_remove("NATS_SERVER")
        .env_remove("DYN_ZMQ_BROKER_URL")
        .env_remove("DYN_ZMQ_BROKER_ENABLED")
        .env_remove("DYN_KV_EVENTS_ZMQ_ENDPOINT")
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .unwrap();
    let deadline = Instant::now() + Duration::from_secs(60);
    loop {
        if child.try_wait().unwrap().is_some() {
            let output = child.wait_with_output().unwrap();
            assert!(
                output.status.success(),
                "{scenario}: {}\n{}\n{}",
                output.status,
                String::from_utf8_lossy(&output.stdout),
                String::from_utf8_lossy(&output.stderr),
            );
            return;
        }
        if Instant::now() >= deadline {
            child.kill().unwrap();
            let output = child.wait_with_output().unwrap();
            panic!(
                "{scenario} did not finish within 60 seconds:\n{}\n{}",
                String::from_utf8_lossy(&output.stdout),
                String::from_utf8_lossy(&output.stderr),
            );
        }
        std::thread::sleep(Duration::from_millis(10));
    }
}

#[test]
fn init_shutdown_reinit_in_subprocess() {
    run_child("initialized");
}

#[test]
fn shutdown_interrupts_discovery_in_subprocess() {
    run_child("waiting");
}

fn init(endpoint: &CStr, block_size: u32) -> DynamoLlmResult {
    unsafe {
        dynamo_llm_init(
            c"test".as_ptr(),
            c"backend".as_ptr(),
            endpoint.as_ptr(),
            block_size,
        )
    }
}

fn assert_publishing_returns_err() {
    let block_ids = [1];
    let tokens = [1; 4];
    let token_counts = [tokens.len()];
    assert_eq!(
        dynamo_kv_event_publish_removed(1, block_ids.as_ptr(), block_ids.len()),
        DynamoLlmResult::ERR
    );
    assert_eq!(
        unsafe {
            dynamo_kv_event_publish_stored(
                2,
                tokens.as_ptr(),
                token_counts.as_ptr(),
                block_ids.as_ptr(),
                block_ids.len(),
                std::ptr::null(),
                std::ptr::null(),
            )
        },
        DynamoLlmResult::ERR
    );
}

#[test]
fn process_lifecycle_child() {
    let Ok(scenario) = std::env::var("DYNAMO_C_API_TEST_SCENARIO") else {
        return;
    };
    assert_publishing_returns_err();
    assert_eq!(dynamo_llm_shutdown(), DynamoLlmResult::ERR);
    let initializing = std::thread::spawn(|| init(c"generate", 4));
    let deadline = Instant::now() + Duration::from_secs(20);
    let drt = loop {
        if let Some(drt) = DRT.get() {
            break drt;
        }
        assert!(
            !initializing.is_finished(),
            "initialization failed before discovery"
        );
        assert!(
            Instant::now() < deadline,
            "initialization did not reach discovery"
        );
        std::thread::sleep(Duration::from_millis(10));
    };

    match scenario.as_str() {
        "initialized" => {
            let mut card = dynamo_llm::model_card::ModelDeploymentCard::default();
            card.display_name = "test-model".to_string();
            let spec = dynamo_runtime::discovery::DiscoverySpec::from_model(
                "test".to_string(),
                "backend".to_string(),
                "generate".to_string(),
                &card,
            )
            .unwrap();
            drt.runtime()
                .secondary()
                .block_on(drt.discovery().register(spec))
                .unwrap();
            assert_eq!(initializing.join().unwrap(), DynamoLlmResult::OK);
            assert!(KV_PUB.get().is_some());
            assert_eq!(init(c"generate", 4), DynamoLlmResult::OK);
            assert_eq!(init(c"different", 4), DynamoLlmResult::ERR);
            assert_eq!(init(c"generate", 8), DynamoLlmResult::ERR);
            assert_eq!(dynamo_llm_shutdown(), DynamoLlmResult::OK);
        }
        "waiting" => {
            assert_eq!(dynamo_llm_shutdown(), DynamoLlmResult::OK);
            assert_eq!(initializing.join().unwrap(), DynamoLlmResult::ERR);
            assert!(KV_PUB.get().is_none());
        }
        _ => panic!("unknown scenario {scenario}"),
    }

    let shutdown_deadline = Instant::now() + Duration::from_secs(10);
    while !drt.primary_token().is_cancelled() {
        assert!(
            Instant::now() < shutdown_deadline,
            "runtime shutdown did not complete"
        );
        std::thread::sleep(Duration::from_millis(10));
    }
    assert_eq!(dynamo_llm_shutdown(), DynamoLlmResult::OK);
    assert_eq!(init(c"generate", 4), DynamoLlmResult::ERR);
    assert_eq!(init(c"different", 8), DynamoLlmResult::ERR);
    assert_publishing_returns_err();
}

#[test]
fn invalid_config_is_rejected_before_runtime_creation() {
    for namespace in [
        std::ptr::null(),
        c"".as_ptr(),
        c"  ".as_ptr(),
        c"\xff".as_ptr(),
    ] {
        assert_eq!(
            unsafe { dynamo_llm_init(namespace, std::ptr::null(), c"generate".as_ptr(), 4) },
            DynamoLlmResult::ERR
        );
    }
    for endpoint in [
        std::ptr::null(),
        c"".as_ptr(),
        c"  ".as_ptr(),
        c"\xff".as_ptr(),
    ] {
        assert_eq!(
            unsafe { dynamo_llm_init(c"test".as_ptr(), std::ptr::null(), endpoint, 4) },
            DynamoLlmResult::ERR
        );
    }
    assert_eq!(init(c"generate", 0), DynamoLlmResult::ERR);
    assert!(WK.get().is_none());
    assert!(DRT.get().is_none());
    assert!(KV_PUB.get().is_none());
}
