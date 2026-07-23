// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use super::request::*;
use super::*;
use futures::StreamExt;
use prost_types::Struct;
use std::collections::BTreeMap;

/// Engine arguments with ample capacity and instant simulated timing, for
/// tests that need requests to be admitted and run to completion quickly.
fn admitting_args() -> MockEngineArgs {
    MockEngineArgs::builder()
        .block_size(4)
        .num_gpu_blocks(4096)
        .max_num_seqs(Some(64))
        .max_num_batched_tokens(Some(1024))
        .speedup_ratio(0.0)
        .build()
        .unwrap()
}

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

#[test]
fn decode_rejects_a_handoff_missing_the_opacity_sentinel() {
    // A decode payload carrying every rendezvous field but missing the
    // non-rendezvous sentinel emulates a sidecar that failed to forward the
    // opaque handoff verbatim; the decode role must reject it.
    let prefill_config = MockerServerConfig {
        mode: ServerMode::Prefill,
        ..Default::default()
    };
    let mut prefill_request = request("sentinel-source");
    prefill_request.kv = Some(pb::KvCacheParameters {
        kv_transfer_params: Some(Struct {
            fields: BTreeMap::from([("do_remote_decode".to_string(), bool_value(true))]),
        }),
        ..Default::default()
    });
    let prepared = PreparedRequest::new(prefill_request, &prefill_config).unwrap();
    let mut handoff = prepared.handoff();
    assert!(
        handoff.fields.remove(HANDOFF_SENTINEL_FIELD).is_some(),
        "prefill handoff should stamp the opacity sentinel"
    );

    let mut decode_request = request("dropped-sentinel");
    decode_request.kv = Some(pb::KvCacheParameters {
        kv_transfer_params: Some(handoff),
        ..Default::default()
    });
    let decode_config = MockerServerConfig {
        mode: ServerMode::Decode,
        ..Default::default()
    };
    let error = PreparedRequest::new(decode_request, &decode_config).unwrap_err();
    assert_eq!(error.code(), tonic::Code::InvalidArgument);
    assert!(error.message().contains(HANDOFF_SENTINEL_FIELD));
}

#[tokio::test]
async fn unary_generate_accumulates_output_and_terminal_metadata() {
    let service = VllmMockerService::new(MockerServerConfig::default(), admitting_args()).unwrap();

    let response =
        pb::generate_server::Generate::generate(&service, Request::new(request("unary")))
            .await
            .unwrap()
            .into_inner();

    assert!(response.prompt_info.is_some());
    let outputs = response
        .outputs
        .expect("unary response accumulates a single sequence output");
    // request() sets max_new_tokens = 2 and output_token_ids = true.
    assert_eq!(outputs.num_tokens, 2);
    assert_eq!(outputs.token_ids.len(), 2);
    let finish = outputs
        .finish_info
        .expect("unary response carries terminal finish info");
    assert_eq!(
        finish.finish_reason,
        pb::finish_info::FinishReason::Length as i32
    );
    assert_eq!(finish.num_output_tokens, 2);
    assert_eq!(service.active_request_count(), 0);
}

#[tokio::test]
async fn streaming_generate_maps_capacity_rejection_to_resource_exhausted() {
    // Same undersized KV cache as the unary case, but exercised through the
    // streaming RPC the production sidecar uses: the call opens successfully and
    // the rejection arrives as a later stream item after prompt info.
    let args = MockEngineArgs::builder()
        .block_size(4)
        .num_gpu_blocks(1)
        .max_num_seqs(Some(8))
        .max_num_batched_tokens(Some(64))
        .speedup_ratio(0.0)
        .build()
        .unwrap();
    let service = VllmMockerService::new(MockerServerConfig::default(), args).unwrap();
    let mut oversized = request("oversized-stream");
    oversized.prompt = Some(pb::generate_request::Prompt::TokenIds(pb::TokenIds {
        ids: vec![1, 2, 3, 4, 5],
    }));

    let mut stream =
        pb::generate_server::Generate::generate_stream(&service, Request::new(oversized))
            .await
            .expect("streaming RPC opens before the scheduler rejects")
            .into_inner();

    let prompt = stream
        .next()
        .await
        .expect("prompt info precedes the rejection")
        .expect("prompt info is not an error");
    assert!(prompt.prompt_info.is_some());

    let status = stream
        .next()
        .await
        .expect("a rejection stream item follows prompt info")
        .expect_err("capacity rejection surfaces as a stream error");
    assert_eq!(status.code(), tonic::Code::ResourceExhausted);
}

#[tokio::test]
async fn streaming_survives_a_producer_that_outruns_a_stalled_consumer() {
    // The producer runs instantly while the consumer stalls, so the request
    // races far past LiveEngine's fixed per-request buffer. The server's pump
    // must absorb the burst instead of shedding the stream into an INTERNAL.
    let service = VllmMockerService::new(MockerServerConfig::default(), admitting_args()).unwrap();
    let mut bursty = request("bursty");
    bursty.stopping.as_mut().unwrap().max_new_tokens = 50;

    let mut stream = pb::generate_server::Generate::generate_stream(&service, Request::new(bursty))
        .await
        .unwrap()
        .into_inner();

    // Stall the consumer so the instant producer fills and overflows the fixed
    // per-request buffer before we read anything.
    tokio::time::sleep(std::time::Duration::from_millis(50)).await;

    let prompt = stream.next().await.unwrap().unwrap();
    assert!(prompt.prompt_info.is_some());

    let mut tokens = 0usize;
    let mut terminal = None;
    while let Some(item) = stream.next().await {
        let response = item.expect("a stalled consumer must not be shed into an INTERNAL error");
        if let Some(output) = response.outputs {
            tokens += output.num_tokens as usize;
            if let Some(finish) = output.finish_info {
                terminal = Some(finish);
            }
        }
    }
    assert_eq!(tokens, 50);
    let finish = terminal.expect("the stream must terminate");
    assert_eq!(
        finish.finish_reason,
        pb::finish_info::FinishReason::Length as i32
    );
    assert_eq!(finish.num_output_tokens, 50);
    assert_eq!(service.active_request_count(), 0);
}
