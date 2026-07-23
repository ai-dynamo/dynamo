// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use super::request::*;
use super::*;
use prost_types::Struct;
use std::collections::BTreeMap;

fn request(id: &str) -> pb::GenerateRequest {
    pb::GenerateRequest {
        request_id: id.to_string(),
        prompt: Some(pb::generate_request::Prompt::TokenIds(pb::TokenIds {
            ids: vec![1, 2, 3],
        })),
        stopping: Some(pb::StoppingCriteria {
            max_new_tokens: 2,
            ..Default::default()
        }),
        response: Some(pb::ResponseOptions {
            prompt_logprobs: true,
            output_text: Some(true),
            output_token_ids: true,
            output_logprobs: true,
            output_candidates: Some(pb::CandidateTokens {
                select: Some(pb::candidate_tokens::Select::TopN(2)),
            }),
            ..Default::default()
        }),
        ..Default::default()
    }
}

#[test]
fn preparation_is_deterministic() {
    let config = MockerServerConfig::default();
    let first = PreparedRequest::new(request("stable"), &config).unwrap();
    let second = PreparedRequest::new(request("stable"), &config).unwrap();
    assert_eq!(first.uuid, second.uuid);
    assert_eq!(first.output_token(0), second.output_token(0));
    assert_eq!(first.output_token(1), second.output_token(1));
    assert_eq!(
        first.direct_request().output_token_ids,
        second.direct_request().output_token_ids
    );
}

#[test]
fn oversized_generation_is_rejected_before_token_planning() {
    let mut oversized = request("too-many-tokens");
    oversized.stopping.as_mut().unwrap().max_new_tokens = MAX_NEW_TOKENS + 1;
    let error = PreparedRequest::new(oversized, &MockerServerConfig::default()).unwrap_err();
    assert_eq!(error.code(), tonic::Code::InvalidArgument);
}

#[test]
fn minimum_tokens_must_not_exceed_the_effective_maximum() {
    let config = MockerServerConfig::default();
    let mut contradictory = request("contradictory-stopping");
    let stopping = contradictory.stopping.as_mut().unwrap();
    stopping.max_new_tokens = 1;
    stopping.min_new_tokens = 2;
    let error = PreparedRequest::new(contradictory, &config).unwrap_err();
    assert_eq!(error.code(), tonic::Code::InvalidArgument);

    let mut default_boundary = request("default-boundary");
    let stopping = default_boundary.stopping.as_mut().unwrap();
    stopping.max_new_tokens = 0;
    stopping.min_new_tokens = DEFAULT_MAX_NEW_TOKENS;
    let prepared = PreparedRequest::new(default_boundary, &config).unwrap();
    assert_eq!(prepared.max_output_tokens, DEFAULT_MAX_NEW_TOKENS as usize);

    let mut above_default = request("above-default");
    let stopping = above_default.stopping.as_mut().unwrap();
    stopping.max_new_tokens = 0;
    stopping.min_new_tokens = DEFAULT_MAX_NEW_TOKENS + 1;
    let error = PreparedRequest::new(above_default, &config).unwrap_err();
    assert_eq!(error.code(), tonic::Code::InvalidArgument);
}

#[test]
fn role_validation_rejects_missing_ambiguous_or_malformed_handoffs() {
    let prefill_config = MockerServerConfig {
        mode: ServerMode::Prefill,
        ..Default::default()
    };
    let error = PreparedRequest::new(request("missing"), &prefill_config).unwrap_err();
    assert_eq!(error.code(), tonic::Code::FailedPrecondition);

    let mut ambiguous = request("ambiguous");
    ambiguous.kv = Some(pb::KvCacheParameters {
        kv_transfer_params: Some(Struct {
            fields: BTreeMap::from([
                ("do_remote_decode".to_string(), bool_value(true)),
                ("do_remote_prefill".to_string(), bool_value(true)),
            ]),
        }),
        ..Default::default()
    });
    let error = PreparedRequest::new(ambiguous, &prefill_config).unwrap_err();
    assert_eq!(error.code(), tonic::Code::InvalidArgument);

    for field in DECODE_RENDEZVOUS_FIELDS {
        let mut contradictory = request(field);
        contradictory.kv = Some(pb::KvCacheParameters {
            kv_transfer_params: Some(Struct {
                fields: BTreeMap::from([
                    ("do_remote_decode".to_string(), bool_value(true)),
                    (field.to_string(), number_value(1.0)),
                ]),
            }),
            ..Default::default()
        });
        let error = PreparedRequest::new(contradictory, &prefill_config).unwrap_err();
        assert_eq!(error.code(), tonic::Code::InvalidArgument, "field: {field}");
    }

    let mut malformed = request("malformed");
    malformed.kv = Some(pb::KvCacheParameters {
        kv_transfer_params: Some(Struct {
            fields: BTreeMap::from([
                ("do_remote_prefill".to_string(), bool_value(true)),
                ("remote_engine_id".to_string(), number_value(1.0)),
            ]),
        }),
        ..Default::default()
    });
    let decode_config = MockerServerConfig {
        mode: ServerMode::Decode,
        ..Default::default()
    };
    let error = PreparedRequest::new(malformed, &decode_config).unwrap_err();
    assert_eq!(error.code(), tonic::Code::InvalidArgument);
}

#[test]
fn text_prompts_fail_with_an_actionable_status() {
    let mut request = request("text");
    request.prompt = Some(pb::generate_request::Prompt::Text("hello".to_string()));
    let error = PreparedRequest::new(request, &MockerServerConfig::default()).unwrap_err();
    assert_eq!(error.code(), tonic::Code::Unimplemented);
    assert!(error.message().contains("token_ids"));
}

#[test]
fn service_rejects_non_vllm_or_multi_rank_engines() {
    let sglang = MockEngineArgs::builder()
        .engine_type(EngineType::Sglang)
        .build()
        .unwrap();
    assert!(
        VllmMockerService::new(MockerServerConfig::default(), sglang)
            .err()
            .unwrap()
            .to_string()
            .contains("engine_type")
    );

    let multi_rank = MockEngineArgs::builder().dp_size(2).build().unwrap();
    assert!(
        VllmMockerService::new(MockerServerConfig::default(), multi_rank)
            .err()
            .unwrap()
            .to_string()
            .contains("dp_size")
    );

    let disaggregated = MockEngineArgs::builder()
        .worker_type(WorkerType::Prefill)
        .build()
        .unwrap();
    assert!(
        VllmMockerService::new(MockerServerConfig::default(), disaggregated)
            .err()
            .unwrap()
            .to_string()
            .contains("worker_type")
    );

    let disabled = MockerServerConfig {
        max_concurrent_requests: 0,
        ..Default::default()
    };
    assert!(
        VllmMockerService::new(disabled, MockEngineArgs::default())
            .err()
            .unwrap()
            .to_string()
            .contains("max_concurrent_requests")
    );
}

#[tokio::test]
async fn unary_generate_maps_capacity_rejection_to_resource_exhausted() {
    let args = MockEngineArgs::builder()
        .block_size(4)
        .num_gpu_blocks(1)
        .max_num_seqs(Some(8))
        .max_num_batched_tokens(Some(64))
        .speedup_ratio(0.0)
        .build()
        .unwrap();
    let service = VllmMockerService::new(MockerServerConfig::default(), args).unwrap();
    let mut oversized = request("oversized");
    oversized.prompt = Some(pb::generate_request::Prompt::TokenIds(pb::TokenIds {
        ids: vec![1, 2, 3, 4, 5],
    }));

    let error = pb::generate_server::Generate::generate(&service, Request::new(oversized))
        .await
        .unwrap_err();
    assert_eq!(error.code(), tonic::Code::ResourceExhausted);
}

#[tokio::test]
async fn concurrent_request_limit_rejects_a_stalled_stream() {
    let args = MockEngineArgs::builder()
        .block_size(4)
        .num_gpu_blocks(128)
        .max_num_seqs(Some(1))
        .speedup_ratio(0.01)
        .build()
        .unwrap();
    let service = VllmMockerService::new(
        MockerServerConfig {
            max_concurrent_requests: 2,
            ..Default::default()
        },
        args,
    )
    .unwrap();
    let mut first_request = request("stalled");
    first_request.stopping.as_mut().unwrap().max_new_tokens = 100;
    let first =
        pb::generate_server::Generate::generate_stream(&service, Request::new(first_request))
            .await
            .unwrap();

    let mut queued_request = request("queued");
    queued_request.stopping.as_mut().unwrap().max_new_tokens = 100;
    let queued =
        pb::generate_server::Generate::generate_stream(&service, Request::new(queued_request))
            .await
            .unwrap();

    let error = match pb::generate_server::Generate::generate_stream(
        &service,
        Request::new(request("rejected")),
    )
    .await
    {
        Ok(_) => panic!("third request unexpectedly exceeded the concurrency limit"),
        Err(error) => error,
    };
    assert_eq!(error.code(), tonic::Code::ResourceExhausted);
    drop(queued);
    drop(first);
}
