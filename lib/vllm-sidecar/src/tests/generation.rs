// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use super::*;

#[tokio::test]
async fn generate_streams_tokens_then_stop_terminal() {
    let handle = spawn_fake_engine(FakeConfig {
        tokens: 4,
        ..FakeConfig::default()
    });
    let engine = engine_for(&handle, DisaggregationMode::Aggregated);
    engine.start(0).await.unwrap();

    let stream = engine
        .generate(request(Some(64)), gen_ctx(fresh_ctx()))
        .await
        .expect("stream");
    let chunks = collect_ok(stream).await;
    let token_chunks = &chunks[..chunks.len() - 1];
    assert_eq!(
        token_chunks
            .iter()
            .map(|chunk| chunk.token_ids.len())
            .sum::<usize>(),
        4
    );
    assert!(matches!(
        chunks.last().unwrap().finish_reason,
        Some(FinishReason::Stop)
    ));
    engine.cleanup().await.unwrap();
}

#[tokio::test]
async fn generate_observes_midstream_cancellation() {
    let handle = spawn_fake_engine(FakeConfig {
        tokens: 50,
        ..FakeConfig::default()
    });
    let engine = engine_for(&handle, DisaggregationMode::Aggregated);
    engine.start(0).await.unwrap();

    let ctx = fresh_ctx();
    let mut stream = engine
        .generate(request(Some(10_000)), gen_ctx(ctx.clone()))
        .await
        .expect("stream");
    assert!(
        stream
            .next()
            .await
            .unwrap()
            .unwrap()
            .finish_reason
            .is_none()
    );
    ctx.stop_generating();
    let rest: Vec<_> = stream.collect().await;
    assert!(matches!(
        rest.last().unwrap().as_ref().unwrap().finish_reason,
        Some(FinishReason::Cancelled)
    ));
    engine.cleanup().await.unwrap();
}

#[tokio::test]
async fn generate_cancelled_before_first_poll_yields_cancelled() {
    let handle = spawn_fake_engine(FakeConfig::default());
    let engine = engine_for(&handle, DisaggregationMode::Aggregated);
    engine.start(0).await.unwrap();

    let ctx = fresh_ctx();
    ctx.stop_generating();
    let chunks = collect_ok(
        engine
            .generate(request(Some(10_000)), gen_ctx(ctx))
            .await
            .expect("stream"),
    )
    .await;
    assert_eq!(chunks.len(), 1);
    assert!(matches!(
        chunks[0].finish_reason,
        Some(FinishReason::Cancelled)
    ));
    engine.cleanup().await.unwrap();
}

#[tokio::test]
async fn generate_forwards_lora_routing_hint() {
    let handle = spawn_fake_engine(FakeConfig {
        supports_lora: true,
        max_loras: 4,
        ..FakeConfig::default()
    });
    let engine = engine_for(&handle, DisaggregationMode::Aggregated);
    engine.start(0).await.unwrap();

    let mut input = request(Some(1));
    input.routing = Some(RoutingHints {
        lora_name: Some("adapter-a".to_string()),
        ..Default::default()
    });
    let _ = collect_ok(engine.generate(input, gen_ctx(fresh_ctx())).await.unwrap()).await;
    assert_eq!(
        handle.last_lora_name.lock().unwrap().as_deref(),
        Some("adapter-a")
    );
    engine.cleanup().await.unwrap();
}

#[test]
fn request_sampling_preserves_presence_and_seed_values() {
    let base = build_generate_request(&request(Some(8)), "req-defaults", false).unwrap();
    let base_sampling = base.sampling.expect("sampling");
    assert_eq!(base_sampling.temperature, None);
    assert_eq!(base_sampling.seed, None);

    let mut input = request(Some(8));
    input.sampling_options.temperature = Some(0.0);
    input.sampling_options.seed = Some(-7);
    input.sampling_options.min_p = Some(0.1);
    input.sampling_options.repetition_penalty = Some(1.2);
    input.stop_conditions.min_tokens = Some(2);
    let sampling = build_generate_request(&input, "req-explicit", false)
        .unwrap()
        .sampling
        .expect("sampling");
    assert_eq!(sampling.temperature, Some(0.0));
    assert_eq!(sampling.seed, Some(-7));
    assert_eq!(sampling.min_tokens, Some(2));
}

#[test]
fn request_rejects_unsupported_sampling_and_output_options() {
    let mut input = request(Some(8));
    input.sampling_options.n = Some(2);
    assert!(build_generate_request(&input, "req-n", false).is_err());

    let mut input = request(Some(8));
    input.sampling_options.use_beam_search = Some(true);
    assert!(build_generate_request(&input, "req-beam", false).is_err());

    let mut input = request(Some(8));
    input.sampling_options.guided_decoding = Some(Default::default());
    assert!(build_generate_request(&input, "req-guided", false).is_err());

    let mut input = request(Some(8));
    input.output_options.logprobs = Some(1);
    assert!(build_generate_request(&input, "req-logprobs", false).is_err());
}
