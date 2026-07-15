// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use super::*;

#[tokio::test]
async fn sidecar_passes_conformance() {
    let handle = spawn_fake_engine(FakeConfig::default());
    dynamo_backend_common::testing::run_conformance(|| {
        engine_for(&handle, DisaggregationMode::Aggregated)
    })
    .await
    .expect("vllm sidecar must satisfy conformance");
}

#[tokio::test]
async fn start_advertises_discovered_metadata() {
    let handle = spawn_fake_engine(FakeConfig::default());
    let engine = engine_for(&handle, DisaggregationMode::Aggregated);

    let cfg = engine.start(0).await.expect("start");
    assert_eq!(cfg.model, "fake-model");
    assert_eq!(cfg.served_model_name.as_deref(), Some("fake-served"));
    let llm = cfg.llm.as_ref().expect("LLM registration");
    assert_eq!(llm.context_length, Some(4096));
    assert_eq!(llm.kv_cache_block_size, Some(16));
    assert_eq!(llm.total_kv_blocks, Some(1000));
    assert_eq!(llm.max_num_seqs, Some(256));
    assert_eq!(llm.max_num_batched_tokens, Some(8192));
    assert!(llm.bootstrap_host.is_none());
    assert_eq!(
        cfg.runtime_data.get("vllm_inference_v1_generate"),
        Some(&serde_json::Value::Bool(true))
    );

    engine.cleanup().await.unwrap();
}

#[tokio::test]
async fn generate_capability_is_role_aware() {
    for (mode, expected) in [
        (DisaggregationMode::Aggregated, true),
        (DisaggregationMode::Decode, true),
        (DisaggregationMode::Prefill, false),
    ] {
        let handle = spawn_fake_engine(FakeConfig::default());
        let engine = engine_for(&handle, mode);
        let cfg = engine.start(0).await.expect("start");
        assert_eq!(
            cfg.runtime_data
                .get("vllm_inference_v1_generate")
                .and_then(serde_json::Value::as_bool),
            expected.then_some(true),
            "unexpected Generate capability for {mode:?}"
        );
        engine.cleanup().await.unwrap();
    }
}

#[tokio::test]
async fn double_start_is_rejected() {
    let handle = spawn_fake_engine(FakeConfig::default());
    let engine = engine_for(&handle, DisaggregationMode::Aggregated);
    engine.start(0).await.expect("first start");

    let error = engine.start(0).await.expect_err("second start must fail");
    assert_eq!(
        error.error_type(),
        ErrorType::Backend(BackendError::EngineShutdown)
    );
    engine.cleanup().await.unwrap();
}

#[tokio::test]
async fn watch_reports_engine_shutdown_when_health_endpoint_disappears() {
    let mut handle = spawn_fake_engine(FakeConfig::default());
    let engine = engine_for(&handle, DisaggregationMode::Aggregated);
    engine.start(0).await.expect("start");

    handle.shutdown();

    let error = tokio::time::timeout(Duration::from_secs(3), engine.watch())
        .await
        .expect("watch should notice fake engine shutdown")
        .expect_err("engine shutdown should fail watch");
    assert_eq!(
        error.error_type(),
        ErrorType::Backend(BackendError::EngineShutdown)
    );
    engine.cleanup().await.unwrap();
}

#[tokio::test]
async fn generate_before_start_errors() {
    let handle = spawn_fake_engine(FakeConfig::default());
    let engine = engine_for(&handle, DisaggregationMode::Aggregated);

    let result = engine
        .generate(request(Some(2)), gen_ctx(fresh_ctx()))
        .await;
    let Err(error) = result else {
        panic!("generate before start must fail");
    };
    assert_eq!(
        error.error_type(),
        ErrorType::Backend(BackendError::EngineShutdown)
    );
}

#[tokio::test]
async fn abort_sends_abort_rpc() {
    let handle = spawn_fake_engine(FakeConfig::default());
    let engine = engine_for(&handle, DisaggregationMode::Aggregated);
    engine.start(0).await.unwrap();

    engine.abort(fresh_ctx()).await;
    assert_eq!(handle.abort_count.load(Ordering::SeqCst), 1);
    engine.cleanup().await.unwrap();
}

#[tokio::test]
async fn abort_before_start_is_noop() {
    let handle = spawn_fake_engine(FakeConfig::default());
    let engine = engine_for(&handle, DisaggregationMode::Aggregated);

    engine.abort(fresh_ctx()).await;
    assert_eq!(handle.abort_count.load(Ordering::SeqCst), 0);
}

#[tokio::test]
async fn lora_lifecycle_proxies_to_vllm_grpc() {
    let handle = spawn_fake_engine(FakeConfig {
        supports_lora: true,
        max_loras: 4,
        ..FakeConfig::default()
    });
    let engine = engine_for(&handle, DisaggregationMode::Aggregated);
    let config = engine.start(0).await.unwrap();
    let llm = config.llm.as_ref().unwrap();
    assert!(llm.supports_lora);
    assert_eq!(llm.max_loras, Some(4));

    let adapter = LoraAdapter {
        id: 17,
        name: "adapter-a".to_string(),
        path: "/tmp/adapter-a".to_string(),
    };
    assert_eq!(engine.load_lora(adapter.clone()).await.unwrap(), adapter);
    assert_eq!(engine.list_loras().await.unwrap(), vec![adapter.clone()]);
    assert_eq!(engine.unload_lora("adapter-a").await.unwrap(), adapter);
    assert!(engine.list_loras().await.unwrap().is_empty());
    engine.cleanup().await.unwrap();
}

#[tokio::test]
async fn lora_load_rejects_mismatched_engine_identity() {
    let handle = spawn_fake_engine(FakeConfig {
        supports_lora: true,
        max_loras: 4,
        mismatch_lora_load_reply: true,
        ..FakeConfig::default()
    });
    let engine = engine_for(&handle, DisaggregationMode::Aggregated);
    engine.start(0).await.unwrap();

    let error = engine
        .load_lora(LoraAdapter {
            id: 17,
            name: "adapter-a".to_string(),
            path: "/tmp/adapter-a".to_string(),
        })
        .await
        .unwrap_err();
    assert_eq!(
        error.error_type(),
        ErrorType::Backend(BackendError::EngineShutdown)
    );
}

#[tokio::test]
async fn unary_control_calls_have_a_response_deadline() {
    let handle = spawn_fake_engine(FakeConfig {
        supports_lora: true,
        max_loras: 4,
        hang_list_loras: true,
        ..FakeConfig::default()
    });
    let engine = VllmSidecarEngine::new(
        handle.endpoint.clone(),
        TransportConfig {
            connect_timeout: Duration::from_millis(50),
            ..test_transport()
        },
        DisaggregationMode::Aggregated,
    );
    engine.start(0).await.unwrap();

    let error = tokio::time::timeout(Duration::from_secs(1), engine.list_loras())
        .await
        .expect("control call exceeded transport deadline")
        .unwrap_err();
    assert_eq!(
        error.error_type(),
        ErrorType::Backend(BackendError::Cancelled)
    );
}

#[tokio::test]
async fn drain_reports_completion() {
    let handle = spawn_fake_engine(FakeConfig::default());
    let engine = engine_for(&handle, DisaggregationMode::Aggregated);
    engine.start(0).await.unwrap();

    assert!(engine.drain_engine().await.expect("drain must succeed"));
    engine.cleanup().await.unwrap();
}

#[tokio::test]
async fn decode_disables_the_generation_canary() {
    let handle = spawn_fake_engine(FakeConfig {
        role: FakeRole::Decode,
        ..FakeConfig::default()
    });
    let engine = engine_for(&handle, DisaggregationMode::Decode);
    assert!(engine.health_check_payload().await.unwrap().is_none());
}
