// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use super::*;

pub(super) fn terminal_output(
    label_ids: &[u32],
    prompt_tokens: usize,
) -> Annotated<LLMEngineOutput> {
    Annotated::from_data(
        serde_json::from_value(json!({
            "token_ids": [99], "index": 0, "finish_reason": "length",
            "top_logprobs": [[
                {"rank": 2, "token_id": label_ids[0], "token": null, "logprob": -2.0},
                {"rank": 3, "token_id": label_ids[1], "token": null, "logprob": -3.0},
                {"rank": 1, "token_id": 99, "token": null, "logprob": -0.5}
            ]],
            "completion_usage": { "prompt_tokens": prompt_tokens, "completion_tokens": 1,
                "total_tokens": prompt_tokens + 1, "prompt_tokens_details": { "cached_tokens": 0 } }
        }))
        .unwrap(),
    )
}

#[tokio::test]
async fn causal_dispatch_preserves_both_contracts_usage_and_placement() {
    for dialect in [
        dynamo_decisions::Dialect::Jev,
        dynamo_decisions::Dialect::OpenAi,
    ] {
        let engine = TestEngine::new(EngineBehavior::Success);
        let mut selected = selection(engine.clone());
        selected.profile = super::super::DecisionExecutionProfile::VllmCausalV1;
        let branches = (0..3)
            .map(|index| {
                let mut branch = prepared_noul_branch(index);
                branch.profile = selected.profile;
                branch.native = Some(super::super::PreparedScoringRequest::Vllm(Box::new(
                    crate::protocols::systemone::vllm::causal_scoring_request(
                        branch.prompt_ids.clone(),
                        &branch.label_ids,
                        "validation",
                    )
                    .unwrap(),
                )));
                branch
            })
            .collect();
        let parent = Context::new(()).context();
        let semaphore = Arc::new(Semaphore::new(3));
        let permit = semaphore.clone().acquire_many_owned(3).await.unwrap();
        let mut request = canonical_request(3).as_ref().clone();
        request.dialect = dialect;
        let response = dispatch(
            selected,
            branches,
            "request".into(),
            parent,
            permit,
            Arc::new(super::super::Metrics::new()),
            Arc::new(request),
        )
        .await;
        assert_eq!(response.status(), StatusCode::OK);
        let body = response_body(response).await;
        assert_eq!(body["usage"]["input_tokens"], 6);
        assert_eq!(body["usage"]["output_tokens"], 3);
        assert_eq!(semaphore.available_permits(), 3);
        if dialect == dynamo_decisions::Dialect::OpenAi {
            assert_eq!(body["usage"]["total_tokens"], 9);
            assert_eq!(body["usage"]["input_tokens_details"]["cached_tokens"], 0);
            assert_eq!(body["answers"].as_array().unwrap().len(), 3);
        } else {
            assert_eq!(body["answers"].as_object().unwrap().len(), 3);
        }
        let observations = engine.observations.lock().unwrap();
        assert_eq!(observations.len(), 3);
        let salt = observations[0].native["cache_salt"].as_str().unwrap();
        for observation in observations.iter() {
            assert_eq!(observation.native["cache_salt"], salt);
            assert_eq!(observation.routing.cache_namespace.as_deref(), Some(salt));
            assert_eq!(observation.routing.expected_output_tokens, Some(1));
            assert_eq!(observation.native["sampling_params"]["max_tokens"], 1);
            if observation.index > 0 {
                assert_eq!(observation.routing.backend_instance_id, Some(41));
                assert_eq!(observation.routing.dp_rank, Some(3));
            }
        }
    }
}
