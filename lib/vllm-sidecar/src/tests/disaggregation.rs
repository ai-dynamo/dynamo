// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use super::*;

#[tokio::test]
async fn prefill_emits_terminal_with_disaggregated_params() {
    let handle = spawn_fake_engine(FakeConfig {
        role: pb::EngineRole::Prefill,
        ..FakeConfig::default()
    });
    let engine = engine_for(&handle, DisaggregationMode::Prefill);
    engine.start(0).await.unwrap();

    let chunks = collect_ok(
        engine
            .generate(request(Some(5)), gen_ctx(fresh_ctx()))
            .await
            .expect("stream"),
    )
    .await;
    assert_eq!(chunks.len(), 1);
    let params = chunks[0]
        .disaggregated_params
        .as_ref()
        .expect("prefill terminal must carry disaggregated_params");
    assert_eq!(params["remote_engine_id"], "engine-fake");
    assert_eq!(params["remote_block_ids"], serde_json::json!([1, 2, 3]));
    engine.cleanup().await.unwrap();
}

#[tokio::test]
async fn decode_rejects_request_without_prefill_result() {
    let handle = spawn_fake_engine(FakeConfig {
        role: pb::EngineRole::Decode,
        ..FakeConfig::default()
    });
    let engine = engine_for(&handle, DisaggregationMode::Decode);
    engine.start(0).await.unwrap();

    let result = engine
        .generate(request(Some(2)), gen_ctx(fresh_ctx()))
        .await;
    let Err(error) = result else {
        panic!("decode without prefill_result must fail");
    };
    assert_eq!(
        error.error_type(),
        ErrorType::Backend(BackendError::InvalidArgument)
    );
    engine.cleanup().await.unwrap();
}

#[tokio::test]
async fn decode_forwards_prefill_result_as_kv_transfer_params() {
    let handle = spawn_fake_engine(FakeConfig {
        role: pb::EngineRole::Decode,
        ..FakeConfig::default()
    });
    let engine = engine_for(&handle, DisaggregationMode::Decode);
    engine.start(0).await.unwrap();

    let handoff = serde_json::json!({
        "remote_engine_id": "engine-fake",
        "remote_port": 20097,
        "remote_block_ids": [4, 5, 6],
    });
    let _ = collect_ok(
        engine
            .generate(
                request_with_prefill_result(handoff.clone()),
                gen_ctx(fresh_ctx()),
            )
            .await
            .expect("stream"),
    )
    .await;

    let captured = handle
        .last_kv_transfer_params
        .lock()
        .unwrap()
        .clone()
        .expect("decode request must carry kv_transfer_params");
    let captured = prost_struct_to_json(&captured);
    assert_eq!(captured, handoff);
    engine.cleanup().await.unwrap();
}

#[tokio::test]
async fn generate_forwards_router_forced_dp_rank() {
    let handle = spawn_fake_engine(FakeConfig::default());
    let engine = engine_for(&handle, DisaggregationMode::Aggregated);
    engine.start(0).await.unwrap();

    let mut input = request(Some(4));
    input.routing = Some(RoutingHints {
        dp_rank: Some(3),
        ..Default::default()
    });
    let _ = collect_ok(engine.generate(input, gen_ctx(fresh_ctx())).await.unwrap()).await;
    assert_eq!(*handle.last_dp_rank.lock().unwrap(), Some(3));
    engine.cleanup().await.unwrap();
}

#[tokio::test]
async fn generate_without_routing_leaves_dp_rank_unset() {
    let handle = spawn_fake_engine(FakeConfig::default());
    let engine = engine_for(&handle, DisaggregationMode::Aggregated);
    engine.start(0).await.unwrap();

    let _ = collect_ok(
        engine
            .generate(request(Some(4)), gen_ctx(fresh_ctx()))
            .await
            .unwrap(),
    )
    .await;
    assert_eq!(*handle.last_dp_rank.lock().unwrap(), None);
    engine.cleanup().await.unwrap();
}
