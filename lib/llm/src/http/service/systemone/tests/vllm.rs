// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use super::*;

pub(super) fn terminal(ids: &[u32], tokens: usize) -> LLMEngineOutput {
    serde_json::from_value(json!({
        "token_ids": [99], "index": 0, "finish_reason": "length",
        "top_logprobs": [[
            {"rank": 2, "token_id": ids[0], "logprob": -2.0},
            {"rank": 3, "token_id": ids[1], "logprob": -3.0},
            {"rank": 1, "token_id": 99, "logprob": -0.5}
        ]],
        "completion_usage": {"prompt_tokens": tokens, "completion_tokens": 1,
            "total_tokens": tokens + 1, "prompt_tokens_details": {"cached_tokens": 0}}
    }))
    .unwrap()
}

async fn run_causal(
    dialect: dynamo_decisions::Dialect,
    missing_cache: bool,
) -> (Response, Arc<TestEngine>) {
    let engine = TestEngine::new(EngineBehavior::Vllm { missing_cache });
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
        let (response, engine) = run_causal(dialect, false).await;
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
async fn malformed_causal_usage_does_not_become_a_successful_answer() {
    let (response, engine) = run_causal(dynamo_decisions::Dialect::OpenAi, true).await;
    assert_eq!(response.status(), StatusCode::INTERNAL_SERVER_ERROR);
    assert!(response_body(response).await.get("answers").is_none());
    assert_eq!(engine.observations.lock().unwrap().len(), 1);
}
