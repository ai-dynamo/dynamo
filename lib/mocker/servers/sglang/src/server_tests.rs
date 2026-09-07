// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use super::*;
use dynamo_sglang_sidecar::proto::sglang_service_server::SglangService;
use futures::StreamExt;

fn engine_args() -> MockEngineArgs {
    MockEngineArgs::builder()
        .engine_type(EngineType::Sglang)
        .block_size(4)
        .num_gpu_blocks(128)
        .max_num_seqs(Some(8))
        .max_num_batched_tokens(Some(64))
        .speedup_ratio(0.0)
        .build()
        .unwrap()
}

fn request(request_id: &str) -> pb::GenerateRequest {
    pb::GenerateRequest {
        input_ids: vec![1, 2, 3],
        sampling_params: Some(pb::SamplingParams {
            max_new_tokens: Some(2),
            n: Some(1),
            ..Default::default()
        }),
        stream: Some(true),
        return_logprob: Some(true),
        top_logprobs_num: Some(2),
        logprob_start_len: Some(0),
        rid: Some(request_id.to_string()),
        ..Default::default()
    }
}

#[tokio::test]
async fn generate_rejects_invalid_requests() {
    let service = SglangMockerService::new(MockerServerConfig::default(), engine_args()).unwrap();
    let mut negative = request("negative");
    negative.input_ids = vec![-1];
    assert_eq!(
        service
            .generate(Request::new(negative))
            .await
            .err()
            .expect("negative token ID should be rejected")
            .code(),
        tonic::Code::InvalidArgument
    );

    let mut bad_n = request("bad-n");
    bad_n.sampling_params.as_mut().unwrap().n = Some(2);
    assert_eq!(
        service
            .generate(Request::new(bad_n))
            .await
            .err()
            .expect("multiple sequences should be rejected")
            .code(),
        tonic::Code::InvalidArgument
    );

    let mut excessive_top_logprobs = request("excessive-top-logprobs");
    excessive_top_logprobs.top_logprobs_num = Some(21);
    assert_eq!(
        service
            .generate(Request::new(excessive_top_logprobs))
            .await
            .err()
            .expect("unbounded top logprobs should be rejected")
            .code(),
        tonic::Code::InvalidArgument
    );

    let short_context = SglangMockerService::new(
        MockerServerConfig {
            context_length: 4,
            ..Default::default()
        },
        engine_args(),
    )
    .unwrap();
    assert_eq!(
        short_context
            .generate(Request::new(request("context-overflow")))
            .await
            .err()
            .expect("prompt and output should exceed the advertised context")
            .code(),
        tonic::Code::InvalidArgument
    );

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
async fn zero_top_logprobs_omits_top_logprob_metadata() {
    let service = SglangMockerService::new(MockerServerConfig::default(), engine_args()).unwrap();
    let mut selected_only = request("selected-only-logprobs");
    selected_only.top_logprobs_num = Some(0);
    let mut stream = service
        .generate(Request::new(selected_only))
        .await
        .unwrap()
        .into_inner();

    while let Some(response) = stream.next().await {
        let response = response.unwrap();
        assert!(response.meta_info.contains_key("output_token_logprobs"));
        assert!(!response.meta_info.contains_key("output_top_logprobs"));
        if response.finished {
            assert!(response.meta_info.contains_key("input_token_logprobs"));
            assert!(!response.meta_info.contains_key("input_top_logprobs"));
            return;
        }
    }
    panic!("stream ended without a terminal response");
}

#[tokio::test]
async fn streaming_survives_a_producer_that_outruns_a_stalled_consumer() {
    let service = SglangMockerService::new(MockerServerConfig::default(), engine_args()).unwrap();
    let mut bursty = request("bursty");
    bursty.sampling_params.as_mut().unwrap().max_new_tokens = Some(50);
    let mut stream = service
        .generate(Request::new(bursty))
        .await
        .unwrap()
        .into_inner();

    tokio::time::sleep(std::time::Duration::from_millis(50)).await;

    let mut responses = 0;
    while let Some(response) = stream.next().await {
        let response = response.expect("a stalled consumer must not cancel valid generation");
        responses += 1;
        if response.finished {
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
        assert!(response.success);
    }
}
