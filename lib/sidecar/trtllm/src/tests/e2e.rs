// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Ignored by default: these drive a live TensorRT-LLM OpenEngine server.

use std::collections::HashMap;

use dynamo_backend_common::MultimodalData;

use super::*;

/// Drives a real disaggregated prefill -> decode handoff against two live
/// OpenEngine servers, each backed by a TensorRT-LLM engine with a KV cache
/// transceiver. Ignored by default; run explicitly with both endpoints:
///
/// ```text
/// TRTLLM_E2E_PREFILL_ENDPOINT=http://127.0.0.1:30001 \
/// TRTLLM_E2E_DECODE_ENDPOINT=http://127.0.0.1:30011 \
///   cargo test -p dynamo-trtllm-sidecar e2e_real_disagg -- --ignored --nocapture
/// ```
#[tokio::test]
#[ignore = "requires two live TensorRT-LLM OpenEngine servers with a KV transceiver"]
async fn e2e_real_disagg_handoff() {
    let prefill_endpoint = std::env::var("TRTLLM_E2E_PREFILL_ENDPOINT")
        .expect("set TRTLLM_E2E_PREFILL_ENDPOINT, e.g. http://127.0.0.1:30001");
    let decode_endpoint = std::env::var("TRTLLM_E2E_DECODE_ENDPOINT")
        .expect("set TRTLLM_E2E_DECODE_ENDPOINT, e.g. http://127.0.0.1:30011");
    let model = std::env::var("TRTLLM_E2E_MODEL")
        .unwrap_or_else(|_| "TinyLlama/TinyLlama-1.1B-Chat-v1.0".to_string());

    // `start` resolves the context length from `Control.GetModelInfo`.
    let configured = |source: String| ConfiguredModel {
        source,
        context_length: None,
        ..Default::default()
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

/// Both servers may share one sufficiently large GPU for local validation.
///
/// ```text
/// TRTLLM_E2E_PREFILL_ENDPOINT=http://127.0.0.1:50051 \
/// TRTLLM_E2E_DECODE_ENDPOINT=http://127.0.0.1:50052 \
/// TRTLLM_E2E_MODEL=Qwen/Qwen3-VL-2B-Instruct \
///   cargo test -p dynamo-trtllm-sidecar e2e_real_openengine_image_disagg -- --ignored --nocapture
/// ```
#[tokio::test]
#[ignore = "requires two live image-capable OpenEngine servers with a KV transceiver"]
async fn e2e_real_openengine_image_disagg() {
    let prefill_endpoint = std::env::var("TRTLLM_E2E_PREFILL_ENDPOINT")
        .expect("set TRTLLM_E2E_PREFILL_ENDPOINT, e.g. http://127.0.0.1:50051");
    let decode_endpoint = std::env::var("TRTLLM_E2E_DECODE_ENDPOINT")
        .expect("set TRTLLM_E2E_DECODE_ENDPOINT, e.g. http://127.0.0.1:50052");
    let model = std::env::var("TRTLLM_E2E_MODEL")
        .unwrap_or_else(|_| "Qwen/Qwen3-VL-2B-Instruct".to_string());
    let configured = |source: String| ConfiguredModel {
        source,
        context_length: None,
        ..Default::default()
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
    prefill.start(0).await.expect("start image prefill worker");
    decode.start(0).await.expect("start image decode worker");

    let token_ids = vec![
        151644, 872, 198, 151652, 151655, 151653, 675, 279, 7987, 504, 2115, 311, 1290, 13, 151645,
        198, 151644, 77091, 198,
    ];
    let image_request = || {
        let mut request = PreprocessedRequest::builder()
            .model(model.clone())
            .token_ids(token_ids.clone())
            .stop_conditions(StopConditions::default())
            .sampling_options(SamplingOptions {
                temperature: Some(0.0),
                ..Default::default()
            })
            .output_options(OutputOptions::default())
            .build()
            .expect("request");
        add_images(
            &mut request,
            [concat!(
                "data:image/png;base64,",
                "iVBORw0KGgoAAAANSUhEUgAAABwAAAAcCAIAAAD9b0jDAAAAK0lEQVR4nGP8z0AsYGQgVi0T0WaSAEYNHTV01NBRQ0cNHTV01NBRQwefoQBvcAI3xbX7+QAAAABJRU5ErkJggg=="
            )],
        );
        request
    };

    let prefill_outputs = collect(&prefill, image_request()).await;
    assert!(
        prefill_outputs
            .iter()
            .all(|output| output.token_ids.is_empty()),
        "the image prefill worker must not stream tokens"
    );
    let handoff = prefill_outputs
        .iter()
        .find_map(|output| output.disaggregated_params.clone())
        .expect("image prefill terminal carries the KV handoff");

    let mut decode_request = image_request();
    decode_request.prefill_result = Some(dynamo_backend_common::PrefillResult {
        disaggregated_params: handoff,
        prompt_tokens_details: None,
    });
    let decode_outputs = collect(&decode, decode_request).await;
    let generated = decode_outputs
        .iter()
        .flat_map(|output| output.token_ids.iter().copied())
        .collect::<Vec<_>>();
    let terminal = decode_outputs
        .iter()
        .find(|output| output.finish_reason.is_some())
        .expect("image decode terminal");
    let usage = terminal.completion_usage.as_ref().expect("terminal usage");

    assert!(
        !generated.is_empty(),
        "the image decode worker must generate"
    );
    assert!(
        usage.prompt_tokens > token_ids.len() as u32,
        "the disaggregated image prompt must report its expanded token count"
    );
}

/// Drives the real `TrtllmSidecarEngine` against a live OpenEngine gRPC server
/// (a real TensorRT-LLM engine on a GPU). Ignored by default; run explicitly with
/// a reachable endpoint:
///
/// ```text
/// TRTLLM_E2E_ENDPOINT=http://127.0.0.1:30001 \
///   cargo test -p dynamo-trtllm-sidecar e2e_real_openengine -- --ignored --nocapture
/// ```
#[tokio::test]
#[ignore = "requires a live TensorRT-LLM OpenEngine server; set TRTLLM_E2E_ENDPOINT"]
async fn e2e_real_openengine_server() {
    let endpoint = std::env::var("TRTLLM_E2E_ENDPOINT")
        .expect("set TRTLLM_E2E_ENDPOINT, e.g. http://127.0.0.1:30001");
    let model = std::env::var("TRTLLM_E2E_MODEL")
        .unwrap_or_else(|_| "TinyLlama/TinyLlama-1.1B-Chat-v1.0".to_string());

    let engine = TrtllmSidecarEngine::new(
        GrpcEndpoint::parse(&endpoint, "--grpc-endpoint").expect("valid endpoint"),
        transport(2),
        // `start` resolves the context length from `Control.GetModelInfo`.
        ConfiguredModel {
            source: model.clone(),
            context_length: None,
            ..Default::default()
        },
        AGG,
    );
    let config = engine
        .start(0)
        .await
        .expect("start against the live server");
    eprintln!("[e2e] connected; registered model = {}", config.model);

    // "<s> Hello, my name is" in the Llama tokenizer; any valid token IDs work.
    let mut request = PreprocessedRequest::builder()
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

    // The router supplies logical rank zero, including for anonymous calls.
    // Run this with native conversation affinity enabled to verify its stricter
    // request contract, and with a legacy server to verify rank-zero fallback.
    request.routing = Some(dynamo_backend_common::engine::RoutingHints {
        dp_rank: Some(0),
        ..Default::default()
    });
    let outputs = collect(&engine, request.clone()).await;
    eprintln!("[e2e] received {} stream item(s)", outputs.len());
    request.agent_context =
        Some(serde_json::from_value(json!({"session_id": "openengine-e2e-session"})).unwrap());
    for _ in 0..2 {
        let session_outputs = collect(&engine, request.clone()).await;
        assert!(
            session_outputs
                .iter()
                .any(|output| output.finish_reason.is_some())
        );
        assert!(
            session_outputs
                .iter()
                .any(|output| !output.token_ids.is_empty())
        );
    }

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

/// The fixed token IDs are Qwen3-VL's chat-template rendering of one image
/// marker and a short question. The tiny PNG is split red/blue so a manual run
/// produces an obvious answer.
///
/// ```text
/// TRTLLM_E2E_ENDPOINT=http://127.0.0.1:50051 \
/// TRTLLM_E2E_MODEL=Qwen/Qwen3-VL-2B-Instruct \
///   cargo test -p dynamo-trtllm-sidecar e2e_real_openengine_image -- --ignored --nocapture
/// ```
#[tokio::test]
#[ignore = "requires a live image-capable TensorRT-LLM OpenEngine server"]
async fn e2e_real_openengine_image() {
    let endpoint = std::env::var("TRTLLM_E2E_ENDPOINT")
        .expect("set TRTLLM_E2E_ENDPOINT, e.g. http://127.0.0.1:50051");
    let model = std::env::var("TRTLLM_E2E_MODEL")
        .unwrap_or_else(|_| "Qwen/Qwen3-VL-2B-Instruct".to_string());

    let engine = TrtllmSidecarEngine::new(
        GrpcEndpoint::parse(&endpoint, "--grpc-endpoint").expect("valid endpoint"),
        transport(1),
        ConfiguredModel {
            source: model.clone(),
            context_length: None,
            ..Default::default()
        },
        AGG,
    );
    engine
        .start(0)
        .await
        .expect("start against the live image-capable server");

    let token_ids = vec![
        151644, 872, 198, 151652, 151655, 151653, 675, 279, 7987, 504, 2115, 311, 1290, 13, 151645,
        198, 151644, 77091, 198,
    ];
    let mut request = PreprocessedRequest::builder()
        .model(model)
        .token_ids(token_ids.clone())
        // Leave max_tokens omitted: only TensorRT-LLM can derive the output
        // budget after expanding the image placeholder.
        .stop_conditions(StopConditions::default())
        .sampling_options(SamplingOptions {
            temperature: Some(0.0),
            ..Default::default()
        })
        .output_options(OutputOptions::default())
        .build()
        .expect("request");
    add_images(
        &mut request,
        [concat!(
            "data:image/png;base64,",
            "iVBORw0KGgoAAAANSUhEUgAAABwAAAAcCAIAAAD9b0jDAAAAK0lEQVR4nGP8z0AsYGQgVi0T0WaSAEYNHTV01NBRQ0cNHTV01NBRQwefoQBvcAI3xbX7+QAAAABJRU5ErkJggg=="
        )],
    );

    let routing = std::env::var_os("TRTLLM_E2E_IMAGE_ROUTING").is_some();
    if routing {
        let mut control = crate::proto::control_client::ControlClient::connect(endpoint.clone())
            .await
            .expect("connect routing capability client");
        let info = control
            .get_model_info(pb::GetModelInfoRequest {
                model: request.model.clone(),
            })
            .await
            .expect("read live routing capability")
            .into_inner();
        assert!(info.extra.as_ref().is_some_and(|extra| matches!(
            extra.fields.get("trtllm_image_routing_version").and_then(|value| value.kind.as_ref()),
            Some(prost_types::value::Kind::NumberValue(version)) if *version == 1.0
        )), "the official runtime must include TRT-LLM #21125 image routing support");
    }
    let mut requests = vec![(request.clone(), "0123456789abcdef")];
    if routing {
        requests.push((request.clone(), "0123456789abcdef"));
        requests.push((request.clone(), "fedcba9876543210"));
        request.multi_modal_data = Some(HashMap::from([(
            "image_url".into(),
            vec![MultimodalData::RawUrl(
                std::env::var("TRTLLM_E2E_CHANGED_IMAGE")
                    .expect("set TRTLLM_E2E_CHANGED_IMAGE for the changed-content check"),
            )],
        )]));
        requests.push((request, "0123456789abcdef"));
    }
    let mut results = Vec::new();
    for (mut request, hash) in requests {
        if routing {
            let count: usize = std::env::var("TRTLLM_E2E_IMAGE_TOKEN_COUNT")
                .expect("set TRTLLM_E2E_IMAGE_TOKEN_COUNT")
                .parse()
                .expect("image token count");
            let pad = dynamo_kv_router::protocols::pad_value_for_mm_hash(
                u64::from_str_radix(hash, 16).unwrap(),
            );
            let expanded: Vec<u32> = token_ids
                .iter()
                .flat_map(|&id| {
                    if id == 151655 {
                        vec![pad; count]
                    } else {
                        vec![id]
                    }
                })
                .collect();
            request.mm_routing_info =
                Some(dynamo_llm::protocols::common::preprocessor::MmRoutingInfo {
                    expanded_prompt_len: expanded.len() as _,
                    routing_token_ids: expanded,
                    ..Default::default()
                });
            request.extra_args = Some(json!({"mm_hashes": [hash]}));
            request.stop_conditions.max_tokens = Some(64);
        }
        let outputs = collect(&engine, request).await;
        let generated = outputs
            .iter()
            .flat_map(|output| output.token_ids.iter().copied())
            .collect::<Vec<_>>();
        let terminal = outputs
            .iter()
            .find(|output| output.finish_reason.is_some())
            .expect("terminal output");
        let usage = terminal.completion_usage.as_ref().expect("terminal usage");
        assert!(!generated.is_empty(), "expected image-conditioned output");
        assert!(
            usage.prompt_tokens > token_ids.len() as u32,
            "the engine must report the expanded image prompt"
        );
        assert!(
            usage.completion_tokens > 0,
            "expected completion token usage"
        );
        results.push(json!({
            "token_ids": generated,
            "prompt_tokens": usage.prompt_tokens,
            "completion_tokens": usage.completion_tokens,
            "cached_tokens": usage.prompt_tokens_details.as_ref().and_then(|details| details.cached_tokens),
        }));
    }
    if routing {
        let cached: Vec<u64> = results
            .iter()
            .map(|result| {
                result["cached_tokens"]
                    .as_u64()
                    .expect("authoritative cache usage")
            })
            .collect();
        assert!(
            cached[1] > cached[0],
            "repeated image must reuse its KV prefix"
        );
        assert!(
            cached[2] < cached[1],
            "a distinct UUID must isolate native cache identity"
        );
        assert!(
            cached[3] < cached[1],
            "changed content must not reuse the first image's KV prefix"
        );
    }
    if let Ok(path) = std::env::var("TRTLLM_E2E_RESULT") {
        let output = if routing {
            json!({"cases": results})
        } else {
            results.remove(0)
        };
        std::fs::write(
            path,
            serde_json::to_vec(&output).expect("serialize image result"),
        )
        .expect("write image result");
    }
}

#[test]
#[ignore = "requires captured native image-routing KV events"]
fn e2e_real_image_routing_event_conversion() {
    use dynamo_kv_router::zmq_wire::{RawKvEvent, decode_event_batch};
    let path = std::env::var("TRTLLM_E2E_ROUTING_EVENTS").expect("set TRTLLM_E2E_ROUTING_EVENTS");
    let payloads: Vec<Vec<u8>> = serde_json::from_slice(&std::fs::read(path).unwrap()).unwrap();
    let pads = [0x0123456789abcdef, 0xfedcba9876543210]
        .map(dynamo_kv_router::protocols::pad_value_for_mm_hash);
    let mut seen = [false; 2];
    for payload in payloads {
        let batch = decode_event_batch(&payload).expect("decode native event batch");
        for event in batch.events {
            if let RawKvEvent::BlockStored { token_ids, .. } = event {
                for (found, pad) in seen.iter_mut().zip(pads) {
                    *found |= token_ids.contains(&pad);
                }
            }
        }
    }
    assert!(
        seen.into_iter().all(|found| found),
        "native events for both UUID hints must reach Dynamo as canonical routing pads"
    );
}
