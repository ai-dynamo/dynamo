// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use super::*;

#[tokio::test]
async fn prefill_emits_terminal_with_disaggregated_params() {
    let handle = spawn_fake_engine(FakeConfig {
        role: FakeRole::Prefill,
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
async fn prefill_carries_prompt_logprobs_in_the_decode_handoff() {
    let handle = spawn_fake_engine(FakeConfig {
        role: FakeRole::Prefill,
        non_finite_prompt_logprobs: true,
        ..FakeConfig::default()
    });
    let engine = engine_for(&handle, DisaggregationMode::Prefill);
    engine.start(0).await.unwrap();
    let mut input = request(Some(5));
    input.output_options.prompt_logprobs = Some(1);

    let chunks = collect_ok(
        engine
            .generate(input, gen_ctx(fresh_ctx()))
            .await
            .expect("stream"),
    )
    .await;
    assert_eq!(chunks.len(), 1);
    assert_eq!(
        chunks[0]
            .disaggregated_params
            .as_ref()
            .and_then(|value| value.get("prompt_logprobs")),
        Some(&serde_json::json!([
            null,
            {
                "11": {"logprob": -1e30, "rank": 1},
                "21": {"logprob": f64::from(-0.3_f32), "rank": 2}
            },
            {
                "12": {"logprob": f64::from(-0.2_f32), "rank": 1},
                "22": {"logprob": -1e30, "rank": 2}
            }
        ]))
    );
    engine.cleanup().await.unwrap();
}

#[tokio::test]
async fn decode_rejects_request_without_prefill_result() {
    let handle = spawn_fake_engine(FakeConfig {
        role: FakeRole::Decode,
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
        role: FakeRole::Decode,
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
async fn decode_emits_handed_off_prompt_logprobs_without_recomputing_them() {
    let handle = spawn_fake_engine(FakeConfig {
        role: FakeRole::Decode,
        tokens: 1,
        ..FakeConfig::default()
    });
    let engine = engine_for(&handle, DisaggregationMode::Decode);
    engine.start(0).await.unwrap();
    let prompt_logprobs = serde_json::json!([
        null,
        {"99": {"logprob": -0.9, "rank": 4}}
    ]);
    let mut input = request_with_prefill_result(serde_json::json!({
        "remote_engine_id": "engine-fake",
        "prompt_logprobs": prompt_logprobs,
    }));
    input.output_options.prompt_logprobs = Some(1);

    let chunks = collect_ok(
        engine
            .generate(input, gen_ctx(fresh_ctx()))
            .await
            .expect("stream"),
    )
    .await;
    assert_eq!(
        chunks[0]
            .engine_data
            .as_ref()
            .and_then(|value| value.get("prompt_logprobs")),
        Some(&prompt_logprobs)
    );
    let captured = handle
        .last_kv_transfer_params
        .lock()
        .unwrap()
        .clone()
        .expect("decode request must carry kv_transfer_params");
    assert_eq!(
        prost_struct_to_json(&captured),
        serde_json::json!({"remote_engine_id": "engine-fake"})
    );
    engine.cleanup().await.unwrap();
}

#[tokio::test]
async fn decode_rejects_missing_requested_prompt_logprobs_handoff() {
    let handle = spawn_fake_engine(FakeConfig {
        role: FakeRole::Decode,
        ..FakeConfig::default()
    });
    let engine = engine_for(&handle, DisaggregationMode::Decode);
    engine.start(0).await.unwrap();
    let mut input = request_with_prefill_result(serde_json::json!({
        "remote_engine_id": "engine-fake",
    }));
    input.output_options.prompt_logprobs = Some(1);

    let result = engine.generate(input, gen_ctx(fresh_ctx())).await;
    let Err(error) = result else {
        panic!("decode must reject a missing requested prompt-logprob handoff");
    };
    assert_eq!(
        error.error_type(),
        ErrorType::Backend(BackendError::InvalidArgument)
    );
    engine.cleanup().await.unwrap();
}
