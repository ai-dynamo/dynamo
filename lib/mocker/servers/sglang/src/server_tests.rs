// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use super::*;
use dynamo_mocker::common::protocols::{EngineType, MockerConfig};
use dynamo_sglang_sidecar::proto::generate_stream_item::Item;
use dynamo_sglang_sidecar::proto::sglang_service_server::SglangService;
use futures::StreamExt;
use serde_json::json;

fn engine_args() -> MockerConfig {
    MockerConfig::from_value(serde_json::json!({
        "engine": {
            "backend": EngineType::Sglang,
            "block_size": 4,
            "num_gpu_blocks": 128,
            "max_num_seqs": 8,
            "max_num_batched_tokens": 64,
            "speedup_ratio": 0.0
        }
    }))
    .unwrap()
}

fn int64_one(value: i64) -> Option<pb::Int64OrList> {
    Some(pb::Int64OrList {
        value: Some(pb::int64_or_list::Value::One(value)),
    })
}

fn request(request_id: &str) -> pb::GenerateRequest {
    pb::GenerateRequest {
        input_ids: Some(pb::TokenIdsOrList {
            value: Some(pb::token_ids_or_list::Value::One(pb::TokenIds {
                ids: vec![1, 2, 3],
            })),
        }),
        sampling_params: Some(pb::SamplingParamsOrList {
            value: Some(pb::sampling_params_or_list::Value::One(
                pb::SamplingParams {
                    max_new_tokens: Some(2),
                    n: Some(1),
                    ..Default::default()
                },
            )),
        }),
        stream: Some(true),
        return_logprob: Some(pb::BoolOrList {
            value: Some(pb::bool_or_list::Value::One(true)),
        }),
        top_logprobs_num: int64_one(2),
        logprob_start_len: int64_one(0),
        rid: Some(pb::StringOrList {
            value: Some(pb::string_or_list::Value::One(request_id.to_string())),
        }),
        ..Default::default()
    }
}

fn sampling(request: &mut pb::GenerateRequest) -> &mut pb::SamplingParams {
    match request
        .sampling_params
        .as_mut()
        .and_then(|params| params.value.as_mut())
    {
        Some(pb::sampling_params_or_list::Value::One(params)) => params,
        _ => unreachable!(),
    }
}

fn frame(item: pb::GenerateStreamItem) -> pb::GenerateResponse {
    match item.item {
        Some(Item::Frame(frame)) => frame,
        other => panic!("expected a frame: {other:?}"),
    }
}

fn meta(frame: &pb::GenerateResponse) -> &pb::GenerateMetaInfo {
    frame.meta_info.as_ref().unwrap()
}

async fn server_info(service: &SglangMockerService) -> pb::GetServerInfoResponse {
    service
        .get_server_info(Request::new(pb::GetServerInfoRequest {}))
        .await
        .unwrap()
        .into_inner()
}

#[tokio::test]
async fn service_rejects_normalized_multi_rank_ais_args() {
    let args = MockerConfig::from_value(json!({
        "engine": {
            "backend": "sglang",
            "timing_model": {
                "type": "external",
                "provider": "ais",
                "config": {
                    "model": "model",
                    "system": "h200_sxm",
                    "backend": "sglang",
                    "worker_type": "aggregated",
                    "attention_dp": 2
                }
            }
        }
    }))
    .unwrap();
    assert_eq!(args.dp_size, 2);

    let error = SglangMockerService::new(MockerServerConfig::default(), args)
        .err()
        .expect("normalized attention DP must be rejected before engine initialization");
    assert_eq!(error.to_string(), "Mocker dp_size must be 1");
}

#[tokio::test]
async fn generate_rejects_invalid_requests() {
    let service = SglangMockerService::new(MockerServerConfig::default(), engine_args()).unwrap();
    let mut negative = request("negative");
    negative.input_ids = Some(pb::TokenIdsOrList {
        value: Some(pb::token_ids_or_list::Value::One(pb::TokenIds {
            ids: vec![-1],
        })),
    });
    let mut bad_n = request("bad-n");
    sampling(&mut bad_n).n = Some(2);
    let mut excessive_top_logprobs = request("excessive-top-logprobs");
    excessive_top_logprobs.top_logprobs_num = int64_one(21);
    let mut batch = request("batch");
    batch.input_ids = Some(pb::TokenIdsOrList {
        value: Some(pb::token_ids_or_list::Value::Many(pb::TokenIdsList {
            items: vec![pb::TokenIds { ids: vec![1] }],
        })),
    });
    for (request, reason) in [
        (negative, "negative token ID should be rejected"),
        (bad_n, "multiple sequences should be rejected"),
        (
            excessive_top_logprobs,
            "unbounded top logprobs should be rejected",
        ),
        (batch, "batch requests are outside the Mocker contract"),
    ] {
        assert_eq!(
            service
                .generate(Request::new(request))
                .await
                .err()
                .expect(reason)
                .code(),
            tonic::Code::InvalidArgument,
            "{reason}"
        );
    }

    // Like SGLang, the context check fails the request inside its stream.
    let short_context = SglangMockerService::new(
        MockerServerConfig {
            context_length: 4,
            ..Default::default()
        },
        engine_args(),
    )
    .unwrap();
    let items = short_context
        .generate(Request::new(request("context-overflow")))
        .await
        .unwrap()
        .into_inner()
        .collect::<Vec<_>>()
        .await;
    assert_eq!(items.len(), 1);
    let Some(Item::Error(error)) = items.into_iter().next().unwrap().unwrap().item else {
        panic!("expected an in-stream error item");
    };
    let body = error.error.unwrap();
    assert_eq!(body.code, 400);
    assert!(
        body.message.contains("maximum context length of 4 tokens"),
        "{}",
        body.message
    );
    assert_eq!(short_context.active_request_count(), 0);

    let prefill_service = SglangMockerService::new(
        MockerServerConfig {
            mode: ServerMode::Prefill,
            ..Default::default()
        },
        engine_args(),
    )
    .unwrap();
    assert_eq!(
        prefill_service
            .generate(Request::new(request("missing-handoff")))
            .await
            .err()
            .expect("missing rendezvous metadata should be rejected")
            .code(),
        tonic::Code::FailedPrecondition
    );
}

#[tokio::test]
async fn omitted_max_new_tokens_uses_native_default() {
    let service = SglangMockerService::new(MockerServerConfig::default(), engine_args()).unwrap();
    let mut omitted = request("omitted-max-new-tokens");
    let params = sampling(&mut omitted);
    params.max_new_tokens = None;
    params.ignore_eos = Some(true);
    let frames = service
        .generate(Request::new(omitted))
        .await
        .unwrap()
        .into_inner()
        .map(|item| frame(item.unwrap()))
        .collect::<Vec<_>>()
        .await;

    assert_eq!(
        frames
            .iter()
            .map(|frame| frame.output_ids.as_ref().unwrap().ids.len())
            .sum::<usize>(),
        128
    );
    let terminal = meta(frames.last().unwrap());
    assert_eq!(terminal.completion_tokens, 128);
    assert!(terminal.e2e_latency.is_some());
    assert_eq!(
        terminal.finish_reason,
        Some(pb::FinishReason {
            kind: Some(pb::finish_reason::Kind::Length(pb::FinishLength {
                length: Some(128)
            })),
        })
    );
    assert!(
        frames[..frames.len() - 1]
            .iter()
            .all(|frame| meta(frame).finish_reason.is_none())
    );
}

#[tokio::test]
async fn zero_top_logprobs_omits_top_logprob_metadata() {
    let service = SglangMockerService::new(MockerServerConfig::default(), engine_args()).unwrap();
    let mut selected_only = request("selected-only-logprobs");
    selected_only.top_logprobs_num = int64_one(0);
    let frames = service
        .generate(Request::new(selected_only))
        .await
        .unwrap()
        .into_inner()
        .map(|item| frame(item.unwrap()))
        .collect::<Vec<_>>()
        .await;

    assert_eq!(frames.len(), 2);
    for (index, frame) in frames.iter().enumerate() {
        let meta = meta(frame);
        assert_eq!(
            meta.output_token_logprobs.as_ref().unwrap().entries.len(),
            1
        );
        assert!(meta.output_top_logprobs.is_empty());
        assert!(meta.input_top_logprobs.is_empty());
        // Like SGLang's incremental stream, prompt logprobs ride the first
        // frame and later frames carry an empty list.
        let input = &meta.input_token_logprobs.as_ref().unwrap().entries;
        if index == 0 {
            assert_eq!(input.len(), 3);
            assert_eq!(input[0].logprob, None);
            assert_eq!(input[0].token_id, 1);
            assert!(input[1].logprob.is_some());
        } else {
            assert!(input.is_empty());
        }
    }
    assert!(meta(&frames[1]).finish_reason.is_some());
}

#[tokio::test]
async fn streaming_survives_a_producer_that_outruns_a_stalled_consumer() {
    let service = SglangMockerService::new(MockerServerConfig::default(), engine_args()).unwrap();
    let mut bursty = request("bursty");
    sampling(&mut bursty).max_new_tokens = Some(50);
    let mut stream = service
        .generate(Request::new(bursty))
        .await
        .unwrap()
        .into_inner();

    tokio::time::sleep(std::time::Duration::from_millis(50)).await;

    let mut responses = 0;
    while let Some(item) = stream.next().await {
        let frame = frame(item.expect("a stalled consumer must not cancel valid generation"));
        responses += 1;
        if meta(&frame).finish_reason.is_some() {
            break;
        }
    }
    assert_eq!(responses, 50);
}

#[tokio::test]
async fn missing_abort_is_idempotent() {
    let service = SglangMockerService::new(MockerServerConfig::default(), engine_args()).unwrap();
    for _ in 0..2 {
        let response = service
            .abort(Request::new(pb::AbortRequest {
                rid: "missing-or-finished".to_string(),
                abort_all: false,
            }))
            .await
            .unwrap()
            .into_inner();
        // Like SGLang, an unknown or finished rid matches nothing.
        assert!(!response.success);
    }
    for (rid, abort_all) in [("", false), ("rid", true)] {
        assert_eq!(
            service
                .abort(Request::new(pb::AbortRequest {
                    rid: rid.to_string(),
                    abort_all,
                }))
                .await
                .unwrap_err()
                .code(),
            tonic::Code::InvalidArgument
        );
    }
}

#[tokio::test]
async fn abort_ends_the_stream_with_a_cancellation_item() {
    let mut args = engine_args();
    args.speedup_ratio = 0.1;
    let service = SglangMockerService::new(MockerServerConfig::default(), args).unwrap();
    let mut long = request("aborted");
    // Long enough to outlast the abort, short enough to fit the KV cache.
    sampling(&mut long).max_new_tokens = Some(400);
    sampling(&mut long).ignore_eos = Some(true);
    let mut stream = service
        .generate(Request::new(long))
        .await
        .unwrap()
        .into_inner();
    let first = frame(stream.next().await.unwrap().unwrap());
    assert!(meta(&first).finish_reason.is_none());

    let response = service
        .abort(Request::new(pb::AbortRequest {
            rid: "aborted".to_string(),
            abort_all: false,
        }))
        .await
        .unwrap()
        .into_inner();
    assert!(response.success);
    let items = tokio::time::timeout(
        std::time::Duration::from_secs(5),
        stream.collect::<Vec<_>>(),
    )
    .await
    .expect("an aborted stream must end");
    let (last, frames) = items.split_last().unwrap();
    assert!(frames.iter().all(|item| matches!(
        item.as_ref().unwrap().item,
        Some(Item::Frame(ref frame)) if meta(frame).finish_reason.is_none()
    )));
    let Some(Item::Error(error)) = last.as_ref().unwrap().item.clone() else {
        panic!("expected a cancellation item: {last:?}");
    };
    assert_eq!(error.error.unwrap().code, 499);
}

#[tokio::test]
async fn kv_event_discovery_follows_regular_mocker_rules() {
    for (mode, prefix_caching, publishes) in [
        (ServerMode::Aggregated, true, true),
        (ServerMode::Prefill, true, true),
        (ServerMode::Decode, true, false),
        (ServerMode::Aggregated, false, false),
    ] {
        let mut args = engine_args();
        args.enable_prefix_caching = prefix_caching;
        let service = SglangMockerService::new(
            MockerServerConfig {
                mode,
                ..Default::default()
            },
            args,
        )
        .unwrap();
        let deployment = server_info(&service).await.deployment_info.unwrap();
        assert_eq!(
            (deployment.nnodes, deployment.dp_size, deployment.dp_rank),
            (1, 1, Some(0))
        );
        let sources = deployment.kv_event_sources;
        assert_eq!(sources.len(), usize::from(publishes));
        if publishes {
            let source = &sources[0];
            assert_eq!(source.dp_rank, 0);
            let port = source
                .endpoint
                .strip_prefix("tcp://127.0.0.1:")
                .unwrap()
                .parse::<u16>()
                .unwrap();
            assert!(port > 0);
            assert_eq!(source.block_size, 4);
            assert_eq!(source.topic, "");
        }
    }
}

#[tokio::test]
async fn discovery_uses_sglang_prefill_limit_for_unbounded_batches() {
    for (batch_limit, expected) in [(64, 64), (usize::MAX, 4096)] {
        let mut args = engine_args();
        args.enable_prefix_caching = false;
        args.max_num_batched_tokens = batch_limit;
        args.sglang.max_prefill_tokens = 4096;
        let service = SglangMockerService::new(MockerServerConfig::default(), args).unwrap();
        assert_eq!(
            server_info(&service)
                .await
                .deployment_info
                .unwrap()
                .max_prefill_tokens,
            expected
        );
    }
}

#[tokio::test]
async fn failed_kv_publisher_is_not_advertised() {
    let occupied = tokio::net::TcpListener::bind("0.0.0.0:0").await.unwrap();
    let mut args = engine_args();
    args.enable_prefix_caching = true;
    args.runtime.zmq_kv_events_port = Some(occupied.local_addr().unwrap().port());
    let service = SglangMockerService::new(MockerServerConfig::default(), args).unwrap();
    assert!(
        server_info(&service)
            .await
            .deployment_info
            .unwrap()
            .kv_event_sources
            .is_empty()
    );
    assert!(
        service
            .generate(Request::new(request("no-publisher")))
            .await
            .is_ok()
    );
}
