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
    assert_eq!(params["transfer_backend"], "NixlConnector");
    assert_eq!(
        params["attributes_struct"]["remote_block_ids"],
        serde_json::json!([1, 2, 3])
    );
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
async fn decode_lifts_prefill_result_onto_kv_session() {
    let handle = spawn_fake_engine(FakeConfig {
        role: pb::EngineRole::Decode,
        ..FakeConfig::default()
    });
    let engine = engine_for(&handle, DisaggregationMode::Decode);
    engine.start(0).await.unwrap();

    let handoff = serde_json::json!({
        "session_id": "sess-xyz",
        "transfer_backend": "NixlConnector",
        "dp_rank": 2,
        "attributes_struct": { "remote_engine_id": "engine-fake", "remote_port": 20097 },
    });
    let _ = collect_ok(
        engine
            .generate(request_with_prefill_result(handoff), gen_ctx(fresh_ctx()))
            .await
            .expect("stream"),
    )
    .await;

    let captured = handle
        .last_kv_session
        .lock()
        .unwrap()
        .clone()
        .expect("decode request must carry a kv_session");
    assert_eq!(captured.session_id, "sess-xyz");
    assert_eq!(captured.transfer_backend, "NixlConnector");
    assert_eq!(captured.dp_rank, 2);
    let attributes = prost_struct_to_json(captured.attributes_struct.as_ref().unwrap());
    assert_eq!(attributes["remote_port"], serde_json::json!(20097));
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
