// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::time::Duration;

use dynamo_backend_common::{
    DisaggregationMode, FinishReason, GenerateContext, GuidedDecodingOptions, LLMEngine,
    PrefillResult, PreprocessedRequest, testing::mock_context,
};
use dynamo_sidecar_testkit::{bounded, fixtures};
use dynamo_vllm_sidecar::VllmSidecarEngine;
use futures::StreamExt;

fn required(name: &str) -> String {
    std::env::var(name).unwrap_or_else(|_| panic!("native test requires {name}"))
}

async fn engine(endpoint: &str, mode: DisaggregationMode) -> VllmSidecarEngine {
    let argv = vec![
        "native-test".to_owned(),
        "--grpc-endpoint".to_owned(),
        required(endpoint),
        "--grpc-connections".to_owned(),
        "1".to_owned(),
        "--grpc-startup-deadline-secs".to_owned(),
        "10".to_owned(),
        "--disaggregation-mode".to_owned(),
        mode.to_string(),
    ];
    let engine = tokio::task::spawn_blocking(move || VllmSidecarEngine::from_args(Some(argv)))
        .await
        .unwrap()
        .unwrap()
        .0;
    bounded("native sidecar startup", engine.start(0))
        .await
        .unwrap();
    engine
}

fn request(max_tokens: u32) -> PreprocessedRequest {
    let mut request =
        fixtures::request(&required("SIDECAR_NATIVE_MODEL"), vec![11; 128], max_tokens);
    request.stop_conditions.ignore_eos = Some(true);
    request.sampling_options.temperature = Some(0.0);
    request
}

async fn metrics() -> String {
    reqwest::Client::builder()
        .timeout(Duration::from_secs(2))
        .build()
        .unwrap()
        .get(required("SIDECAR_NATIVE_METRICS"))
        .send()
        .await
        .unwrap()
        .error_for_status()
        .unwrap()
        .text()
        .await
        .unwrap()
}

fn metric(body: &str, name: &str) -> f64 {
    let values: Vec<f64> = body
        .lines()
        .filter(|line| {
            line.strip_prefix(name)
                .is_some_and(|tail| tail.starts_with('{') || tail.starts_with(' '))
        })
        .map(|line| line.rsplit_once(' ').unwrap().1.parse().unwrap())
        .collect();
    assert!(!values.is_empty(), "missing native metric {name}: {body}");
    values.iter().sum()
}

async fn scheduler(active: bool) {
    let deadline = tokio::time::Instant::now() + Duration::from_secs(10);
    loop {
        let body = metrics().await;
        let running = metric(&body, "vllm:num_requests_running");
        let waiting = metric(&body, "vllm:num_requests_waiting");
        if (active && running > 0.0) || (!active && running == 0.0 && waiting == 0.0) {
            return;
        }
        assert!(
            tokio::time::Instant::now() < deadline,
            "scheduler active={active}: {body}"
        );
        tokio::time::sleep(Duration::from_millis(20)).await;
    }
}

async fn recovery(engine: &VllmSidecarEngine) {
    let outputs = fixtures::collect(
        engine,
        request(4),
        GenerateContext::new(mock_context(), None),
    )
    .await;
    let values: Vec<_> = outputs
        .iter()
        .flat_map(|o| o.as_ref().unwrap().token_ids.iter().copied())
        .collect();
    assert_eq!(values.len(), 4);
    dynamo_sidecar_testkit::assert::terminal(outputs, &values, 128, FinishReason::Length);
}

#[tokio::test]
async fn vllm_native_logprobs_and_structured_output_are_compatible() {
    tokio::time::timeout(Duration::from_secs(90), async {
        let engine = engine("SIDECAR_NATIVE_GRPC", DisaggregationMode::Aggregated).await;
        let mut logprob_request = request(8);
        logprob_request.output_options.logprobs = Some(2);
        logprob_request.output_options.prompt_logprobs = Some(2);
        let outputs = fixtures::collect(
            &engine,
            logprob_request,
            GenerateContext::new(mock_context(), None),
        )
        .await;
        let chunks: Vec<_> = outputs.iter().map(|item| item.as_ref().unwrap()).collect();
        assert!(chunks.iter().filter(|o| !o.token_ids.is_empty()).count() > 1);
        let values: Vec<_> = chunks
            .iter()
            .flat_map(|o| o.token_ids.iter().copied())
            .collect();
        assert_eq!(values.len(), 8);
        for chunk in &chunks {
            let selected = chunk.log_probs.as_ref().expect("requested output logprobs");
            let candidates = chunk.top_logprobs.as_ref().expect("requested top logprobs");
            assert_eq!(selected.len(), chunk.token_ids.len());
            assert_eq!(candidates.len(), chunk.token_ids.len());
            for ((token, logprob), row) in chunk.token_ids.iter().zip(selected).zip(candidates) {
                assert_eq!(row.len(), 3, "selected token plus two native candidates");
                assert_eq!(row[0].token_id, *token);
                assert_eq!(row[0].logprob, *logprob);
                assert_eq!(row[0].rank, 1);
                assert_eq!(row[1].token_id, *token);
                assert_eq!(row[1].logprob, *logprob);
                assert_eq!(row[1].rank, 1);
                assert_eq!(row[2].rank, 2);
                assert_ne!(row[1].token_id, row[2].token_id);
                for entry in row {
                    assert!(entry.rank > 0);
                    assert!(entry.logprob.is_finite());
                    assert!(entry.logprob > -9999.0 && entry.logprob <= 0.0);
                }
            }
            if chunk.finish_reason.is_none() {
                assert!(chunk.engine_data.is_none());
            }
        }
        let prompt = chunks.last().unwrap().engine_data.as_ref().unwrap()["prompt_logprobs"]
            .as_array()
            .expect("requested terminal prompt logprobs");
        assert_eq!(prompt.len(), 128);
        assert!(prompt[0].is_null());
        for position in &prompt[1..] {
            let entries = position.as_object().unwrap();
            assert!((2..=3).contains(&entries.len()));
            assert!(entries.contains_key("11"), "prompt token association");
            for rank in [1, 2] {
                assert!(
                    entries
                        .values()
                        .any(|entry| entry["rank"].as_u64() == Some(rank))
                );
            }
            for (token, entry) in entries {
                token.parse::<u32>().expect("candidate token ID");
                assert!(entry["rank"].as_u64().is_some_and(|rank| rank > 0));
                let logprob = entry["logprob"].as_f64().unwrap();
                assert!(logprob.is_finite() && logprob > -9999.0 && logprob <= 0.0);
            }
        }
        dynamo_sidecar_testkit::assert::terminal(outputs, &values, 128, FinishReason::Length);

        let mut structured_request = request(64);
        structured_request.token_ids =
            serde_json::from_str(&required("SIDECAR_NATIVE_STRUCTURED_PROMPT")).unwrap();
        let prompt_tokens = u32::try_from(structured_request.token_ids.len()).unwrap();
        assert!(prompt_tokens > 0);
        structured_request.stop_conditions.ignore_eos = Some(false);
        structured_request.sampling_options.guided_decoding = Some(GuidedDecodingOptions {
            json: Some(serde_json::json!({
                "type": "object",
                "properties": {"ok": {"type": "boolean", "const": true}},
                "required": ["ok"],
                "additionalProperties": false
            })),
            ..Default::default()
        });
        let outputs = tokio::time::timeout(Duration::from_secs(60), async {
            engine
                .generate(
                    structured_request,
                    GenerateContext::new(mock_context(), None),
                )
                .await
                .unwrap()
                .collect::<Vec<_>>()
                .await
        })
        .await
        .expect("native structured-output compilation and generation");
        let mut text = String::new();
        let mut values = Vec::new();
        for output in &outputs {
            let output = output.as_ref().unwrap();
            text.push_str(output.text.as_deref().expect("native decoded text"));
            values.extend_from_slice(&output.token_ids);
        }
        assert!(!values.is_empty());
        let value: serde_json::Value = serde_json::from_str(&text)
            .unwrap_or_else(|error| panic!("native structured output {text:?}: {error}"));
        assert_eq!(value, serde_json::json!({"ok": true}));
        dynamo_sidecar_testkit::assert::terminal(
            outputs,
            &values,
            prompt_tokens,
            FinishReason::Stop,
        );
        scheduler(false).await;
        engine.cleanup().await.unwrap();
    })
    .await
    .expect("native compatibility scenario timed out");
}

#[tokio::test]
async fn vllm_cancellation_and_consumer_drop_release_native_work() {
    tokio::time::timeout(Duration::from_secs(90), async {
        let engine = engine("SIDECAR_NATIVE_GRPC", DisaggregationMode::Aggregated).await;
        recovery(&engine).await;
        for explicit in [true, false] {
            scheduler(false).await;
            let context = mock_context();
            let mut stream = engine
                .generate(request(4096), GenerateContext::new(context.clone(), None))
                .await
                .unwrap();
            let first = bounded("first native output", stream.next())
                .await
                .unwrap()
                .unwrap();
            assert!(!first.token_ids.is_empty());
            assert_eq!(first.finish_reason, None);
            scheduler(true).await;
            if explicit {
                context.stop_generating();
                let terminal = bounded("native cancellation", stream.next())
                    .await
                    .unwrap()
                    .unwrap();
                assert_eq!(terminal.finish_reason, Some(FinishReason::Cancelled));
                assert!(
                    bounded("native cancellation EOF", stream.next())
                        .await
                        .is_none()
                );
            }
            drop(stream);
            scheduler(false).await;
            recovery(&engine).await;
        }
        engine.cleanup().await.unwrap();
    })
    .await
    .expect("native cancellation scenario timed out");
}

#[tokio::test]
async fn vllm_handoff_transfers_native_kv() {
    tokio::time::timeout(Duration::from_secs(90), async {
        let prefill = engine("SIDECAR_NATIVE_PREFILL_GRPC", DisaggregationMode::Prefill).await;
        let decode = engine("SIDECAR_NATIVE_GRPC", DisaggregationMode::Decode).await;
        let outputs = fixtures::collect(
            &prefill,
            request(8),
            GenerateContext::new(mock_context(), None),
        )
        .await;
        let outputs: Vec<_> = outputs.into_iter().collect::<Result<_, _>>().unwrap();
        assert!(outputs.iter().all(|o| o.token_ids.is_empty()));
        let last = outputs.last().unwrap();
        assert_eq!(last.finish_reason, Some(FinishReason::Length));
        let handoff = last
            .disaggregated_params
            .clone()
            .expect("native prefill handoff");
        assert!(
            handoff["remote_engine_id"]
                .as_str()
                .is_some_and(|id| !id.is_empty())
        );
        assert!(
            handoff["remote_block_ids"]
                .as_array()
                .is_some_and(|ids| !ids.is_empty())
        );
        let mut request = request(8);
        request.prefill_result = Some(PrefillResult {
            disaggregated_params: handoff,
            prompt_tokens_details: None,
        });
        let outputs =
            fixtures::collect(&decode, request, GenerateContext::new(mock_context(), None)).await;
        let values: Vec<_> = outputs
            .iter()
            .flat_map(|o| o.as_ref().unwrap().token_ids.iter().copied())
            .collect();
        assert_eq!(values.len(), 8);
        dynamo_sidecar_testkit::assert::terminal(outputs, &values, 128, FinishReason::Length);
        let telemetry = std::fs::read_to_string(required("SIDECAR_NATIVE_TRANSFER_PROBE"))
            .expect("native NIXL telemetry must be recorded");
        let bytes: u64 = telemetry
            .lines()
            .map(|line| {
                serde_json::from_str::<serde_json::Value>(line).unwrap()["bytes"]
                    .as_u64()
                    .unwrap()
            })
            .sum();
        assert!(
            bytes > 0,
            "successful stream alone does not establish KV transfer"
        );
        scheduler(false).await;
        prefill.cleanup().await.unwrap();
        decode.cleanup().await.unwrap();
    })
    .await
    .expect("native handoff scenario timed out");
}
