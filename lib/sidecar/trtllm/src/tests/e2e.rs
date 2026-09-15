// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Ignored by default: these drive a live TensorRT-LLM OpenEngine server.

use super::*;

/// Drives a real disaggregated prefill -> decode handoff against two live
/// OpenEngine servers, each backed by a TensorRT-LLM engine with a KV cache
/// transceiver. Ignored by default; run explicitly with both endpoints:
///
/// ```text
/// TRTLLM_E2E_PREFILL_ENDPOINT=http://127.0.0.1:50051 \
/// TRTLLM_E2E_DECODE_ENDPOINT=http://127.0.0.1:50052 \
///   cargo test -p dynamo-trtllm-sidecar e2e_real_disagg -- --ignored --nocapture
/// ```
#[tokio::test]
#[ignore = "requires two live TensorRT-LLM OpenEngine servers with a KV transceiver"]
async fn e2e_real_disagg_handoff() {
    let prefill_endpoint = std::env::var("TRTLLM_E2E_PREFILL_ENDPOINT")
        .expect("set TRTLLM_E2E_PREFILL_ENDPOINT, e.g. http://127.0.0.1:50051");
    let decode_endpoint = std::env::var("TRTLLM_E2E_DECODE_ENDPOINT")
        .expect("set TRTLLM_E2E_DECODE_ENDPOINT, e.g. http://127.0.0.1:50052");
    let model = std::env::var("TRTLLM_E2E_MODEL")
        .unwrap_or_else(|_| "TinyLlama/TinyLlama-1.1B-Chat-v1.0".to_string());

    // `start` resolves the context length from `Control.GetModelInfo`.
    let configured = |source: String| ConfiguredModel {
        source,
        context_length: None,
    };
    let prefill = TrtllmSidecarEngine::new(
        GrpcEndpoint::parse(&prefill_endpoint, "--grpc-endpoint").expect("valid endpoint"),
        transport(1),
        configured(model.clone()),
        DisaggregationMode::Prefill,
    );
    let decode = TrtllmSidecarEngine::new(
        GrpcEndpoint::parse(&decode_endpoint, "--grpc-endpoint").expect("valid endpoint"),
        transport(1),
        configured(model.clone()),
        DisaggregationMode::Decode,
    );
    prefill.start(0).await.expect("start prefill worker");
    decode.start(0).await.expect("start decode worker");

    // "<s> Hello, my name is" in the Llama tokenizer.
    let base = || {
        PreprocessedRequest::builder()
            .model(model.clone())
            .token_ids(vec![1, 15043, 29892, 590, 1024, 338])
            .stop_conditions(StopConditions {
                max_tokens: Some(16),
                ..Default::default()
            })
            .sampling_options(SamplingOptions {
                temperature: Some(0.0),
                ..Default::default()
            })
            .output_options(OutputOptions::default())
            .build()
            .expect("request")
    };

    // Phase 1: prefill.
    let prefill_outputs = collect(&prefill, base()).await;
    eprintln!(
        "[e2e-disagg] prefill produced {} chunk(s)",
        prefill_outputs.len()
    );
    assert!(
        prefill_outputs
            .iter()
            .all(|output| output.token_ids.is_empty()),
        "the prefill worker must not stream tokens"
    );
    let handoff = prefill_outputs
        .iter()
        .find_map(|output| output.disaggregated_params.clone())
        .expect("prefill terminal carries the KV handoff");
    eprintln!("[e2e-disagg] handoff = {handoff}");

    // Phase 2: decode replays the handoff.
    let mut decode_request = base();
    decode_request.prefill_result = Some(dynamo_backend_common::PrefillResult {
        disaggregated_params: handoff,
        prompt_tokens_details: None,
    });
    let decode_outputs = collect(&decode, decode_request).await;

    let mut generated = Vec::new();
    let mut terminal = None;
    for output in &decode_outputs {
        generated.extend(output.token_ids.iter().copied());
        if output.finish_reason.is_some() {
            terminal = Some(output);
        }
    }
    eprintln!("[e2e-disagg] decode generated token IDs: {generated:?}");
    let terminal = terminal.expect("a terminal output carrying a finish_reason");
    eprintln!("[e2e-disagg] finish_reason = {:?}", terminal.finish_reason);
    let usage = terminal.completion_usage.as_ref().expect("terminal usage");
    eprintln!(
        "[e2e-disagg] usage: prompt={}, completion={}",
        usage.prompt_tokens, usage.completion_tokens
    );

    assert!(
        !generated.is_empty(),
        "the decode worker must generate tokens from the prefill handoff"
    );
}

/// Drives the real `TrtllmSidecarEngine` against a live OpenEngine gRPC server
/// (a real TensorRT-LLM engine on a GPU). Ignored by default; run explicitly with
/// a reachable endpoint:
///
/// ```text
/// TRTLLM_E2E_ENDPOINT=http://127.0.0.1:50051 \
///   cargo test -p dynamo-trtllm-sidecar e2e_real_openengine -- --ignored --nocapture
/// ```
#[tokio::test]
#[ignore = "requires a live TensorRT-LLM OpenEngine server; set TRTLLM_E2E_ENDPOINT"]
async fn e2e_real_openengine_server() {
    let endpoint = std::env::var("TRTLLM_E2E_ENDPOINT")
        .expect("set TRTLLM_E2E_ENDPOINT, e.g. http://127.0.0.1:50051");
    let model = std::env::var("TRTLLM_E2E_MODEL")
        .unwrap_or_else(|_| "TinyLlama/TinyLlama-1.1B-Chat-v1.0".to_string());

    let engine = TrtllmSidecarEngine::new(
        GrpcEndpoint::parse(&endpoint, "--grpc-endpoint").expect("valid endpoint"),
        transport(2),
        // `start` resolves the context length from `Control.GetModelInfo`.
        ConfiguredModel {
            source: model.clone(),
            context_length: None,
        },
        AGG,
    );
    let config = engine
        .start(0)
        .await
        .expect("start against the live server");
    eprintln!("[e2e] connected; registered model = {}", config.model);

    // "<s> Hello, my name is" in the Llama tokenizer; any valid token IDs work.
    let request = PreprocessedRequest::builder()
        .model(model)
        .token_ids(vec![1, 15043, 29892, 590, 1024, 338])
        .stop_conditions(StopConditions {
            max_tokens: Some(16),
            ..Default::default()
        })
        .sampling_options(SamplingOptions {
            temperature: Some(0.0),
            ..Default::default()
        })
        .output_options(OutputOptions {
            logprobs: Some(1),
            ..Default::default()
        })
        .build()
        .expect("request");

    let outputs = collect(&engine, request).await;
    eprintln!("[e2e] received {} stream item(s)", outputs.len());

    let mut generated = Vec::new();
    let mut terminal = None;
    let mut saw_logprobs = false;
    for output in &outputs {
        generated.extend(output.token_ids.iter().copied());
        saw_logprobs |= output.log_probs.is_some();
        if output.finish_reason.is_some() {
            terminal = Some(output);
        }
    }
    eprintln!("[e2e] generated token IDs: {generated:?}");

    let terminal = terminal.expect("a terminal output carrying a finish_reason");
    eprintln!("[e2e] finish_reason = {:?}", terminal.finish_reason);
    let usage = terminal.completion_usage.as_ref().expect("terminal usage");
    eprintln!(
        "[e2e] usage: prompt={}, completion={}",
        usage.prompt_tokens, usage.completion_tokens
    );

    assert!(
        !generated.is_empty(),
        "expected at least one generated token"
    );
    assert_eq!(
        usage.prompt_tokens, 6,
        "prompt token count should echo the input"
    );
    assert!(
        usage.completion_tokens > 0,
        "expected nonzero completion usage"
    );
    assert!(
        saw_logprobs,
        "logprobs were requested but none were surfaced"
    );
}
