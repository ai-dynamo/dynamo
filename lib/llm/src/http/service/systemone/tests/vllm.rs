// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use super::*;

#[derive(Clone, Copy)]
pub(super) enum Outcome {
    Success,
    MissingCache,
    MissingCandidate,
    Cancelled,
    Error,
}

pub(super) fn terminal(ids: &[u32], tokens: usize, outcome: Outcome) -> LLMEngineOutput {
    let mut output: LLMEngineOutput = serde_json::from_value(json!({
        "token_ids": [99], "index": 0, "finish_reason": "length",
        "top_logprobs": [[
            {"rank": 2, "token_id": ids[0], "logprob": -2.0},
            {"rank": 3, "token_id": ids[1], "logprob": -3.0},
            {"rank": 1, "token_id": 99, "logprob": -0.5}
        ]],
        "completion_usage": {"prompt_tokens": tokens, "completion_tokens": 1,
            "total_tokens": tokens + 1, "prompt_tokens_details": {"cached_tokens": 0}}
    }))
    .unwrap();
    match outcome {
        Outcome::Success => {}
        Outcome::MissingCache => {
            output
                .completion_usage
                .as_mut()
                .unwrap()
                .prompt_tokens_details = None;
        }
        Outcome::MissingCandidate => {
            output.top_logprobs.as_mut().unwrap()[0].remove(0);
        }
        Outcome::Cancelled | Outcome::Error => {
            output = LLMEngineOutput {
                finish_reason: Some(match outcome {
                    Outcome::Cancelled => crate::protocols::common::FinishReason::Cancelled,
                    _ => crate::protocols::common::FinishReason::Error(
                        "private worker detail".into(),
                    ),
                }),
                ..Default::default()
            };
        }
    }
    output
}

async fn run_causal(
    dialect: dynamo_decisions::Dialect,
    outcome: Outcome,
) -> (Response, Arc<TestEngine>) {
    let engine = TestEngine::new(EngineBehavior::Vllm(outcome));
    let mut selected = selection(engine.clone());
    selected.backend = ScoringBackend::Vllm;
    selected.card.runtime_config.max_num_seqs = Some(1);
    selected.card.worker_type = Some(crate::worker_type::WorkerType::Aggregated);
    selected
        .card
        .runtime_config
        .runtime_data
        .insert("vllm_inference_v1_generate".into(), true.into());
    selected.card.runtime_config.data_parallel_size = 1;
    selected.card.runtime_config.data_parallel_start_rank = 3;
    selected
        .card
        .runtime_config
        .runtime_data
        .insert("vllm_causal_decision_v1".into(), true.into());
    let mut request = canonical_request(3).as_ref().clone();
    request.dialect = dialect;
    let branches = (0..3)
        .map(|index| {
            let mut branch = prepared_noul_branch(index);
            branch.native = Some(
                super::super::causal_scoring_request(
                    branch.prompt_ids.clone(),
                    &branch.label_ids,
                    "preflight",
                )
                .unwrap(),
            );
            branch
        })
        .collect();
    let response = dispatch(
        selected,
        branches,
        "request".into(),
        Context::new(()).context(),
        Arc::new(Semaphore::new(3))
            .acquire_many_owned(3)
            .await
            .unwrap(),
        Arc::new(super::super::super::metrics::Metrics::new()),
        Arc::new(request),
    )
    .await;
    (response, engine)
}

#[tokio::test]
async fn causal_dispatch_preserves_both_contracts_usage_and_pinned_placement() {
    for dialect in [
        dynamo_decisions::Dialect::OpenAi,
        dynamo_decisions::Dialect::Jev,
    ] {
        let (response, engine) = run_causal(dialect, Outcome::Success).await;
        assert_eq!(response.status(), StatusCode::OK);
        let body = response_body(response).await;
        assert_eq!(body["usage"]["output_tokens"], 3);
        let observations = engine.observations.lock().unwrap();
        assert_eq!(observations.len(), 3);
        for observation in observations.iter() {
            assert_eq!(
                observation.native["cache_salt"],
                observations[0].native["cache_salt"]
            );
            assert_eq!(observation.routing.expected_output_tokens, Some(1));
            if observation.index > 0 {
                assert_eq!(observation.routing.backend_instance_id, Some(41));
                assert_eq!(observation.routing.dp_rank, Some(3));
            }
        }
    }
}

#[tokio::test]
async fn malformed_causal_results_do_not_become_successful_answers() {
    for dialect in [
        dynamo_decisions::Dialect::OpenAi,
        dynamo_decisions::Dialect::Jev,
    ] {
        for outcome in [Outcome::MissingCache, Outcome::MissingCandidate] {
            let (response, engine) = run_causal(dialect, outcome).await;
            assert_eq!(response.status(), StatusCode::INTERNAL_SERVER_ERROR);
            assert!(response_body(response).await.get("answers").is_none());
            assert_eq!(engine.observations.lock().unwrap().len(), 1);
        }
    }
}

#[tokio::test]
async fn causal_worker_cancellation_and_failure_have_distinct_statuses() {
    for dialect in [
        dynamo_decisions::Dialect::OpenAi,
        dynamo_decisions::Dialect::Jev,
    ] {
        for (outcome, expected_status) in [
            (Outcome::Cancelled, StatusCode::SERVICE_UNAVAILABLE),
            (Outcome::Error, StatusCode::INTERNAL_SERVER_ERROR),
        ] {
            let (response, engine) =
                tokio::time::timeout(Duration::from_secs(2), run_causal(dialect, outcome))
                    .await
                    .expect("worker abort must not wait for stream closure");
            assert_eq!(response.status(), expected_status);
            let body = response_body(response).await;
            let error = if dialect == dynamo_decisions::Dialect::OpenAi {
                &body["error"]
            } else {
                &body
            };
            assert_eq!(error["code"], "server_error");
            assert!(body.get("answers").is_none());
            assert!(!body.to_string().contains("private worker detail"));
            let observations = engine.observations.lock().unwrap();
            assert_eq!(observations.len(), 1);
            assert!(observations[0].request_context.is_killed());
            assert!(observations[0].stream_context.is_killed());
        }
    }
}

#[tokio::test]
async fn causal_worker_failure_retains_private_detail_for_server_logging() {
    let prepared = prepared_noul_branch(0);
    let output = terminal(
        &prepared.label_ids,
        prepared.prompt_ids.len(),
        Outcome::Error,
    );
    let stream = ResponseStream::new(
        Box::pin(stream::iter([Annotated::from_data(output)])),
        Context::new(()).context(),
    );
    let result = super::super::read_vllm_branch(
        prepared,
        Arc::new(RequestTracker::new()),
        stream,
        0,
        1,
        Context::new(()).context(),
    )
    .await;
    let Err(error) = result else {
        panic!("worker failure must not produce an answer")
    };
    assert_eq!(error.detail, "private worker detail");
    assert_eq!(error.public_message, "scoring worker failed");
}
