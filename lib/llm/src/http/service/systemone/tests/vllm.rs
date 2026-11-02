// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use super::*;

fn vllm_frame(request: &PreprocessedRequest) -> (Value, Annotated<LLMEngineOutput>) {
    let native = request.extra_args.as_ref().unwrap()["vllm_tito"].clone();
    let labels: Vec<u32> =
        serde_json::from_value(native["sampling_params"]["logprob_token_ids"].clone()).unwrap();
    let terminal = terminal_output(&labels, request.token_ids.len());
    (native, terminal)
}

async fn causal_selection_for_test(engine: Arc<TestEngine>) -> PreparedSelectionFixture {
    let selected = selection_for_test(engine).await;
    let mut card = selected.context.card.clone();
    card.runtime_config.max_num_seqs = Some(1);
    card.runtime_config.data_parallel_size = 1;
    card.runtime_config.data_parallel_start_rank = 3;
    let executor = token_executor(&selected);
    token_selection_for_test(
        card,
        executor.engine().clone(),
        executor.preprocessor().clone(),
        Profile::VllmCausalV1,
    )
    .await
}

fn prepared_causal_branch(index: usize) -> PreparedBranch {
    let label_ids = vec![17 + index as u32 * 100, 4 + index as u32 * 100];
    PreparedBranch {
        scoring: ScoringBranch {
            index,
            temperature: 1.0,
            prompt_len: 2,
            label_ids: label_ids.clone(),
        },
        wire: super::super::PreparedScoringRequest::vllm(vec![1, 2], &label_ids, "causal-fixture")
            .unwrap(),
    }
}

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
        let engine = TestEngine::with_frame(EngineBehavior::Success, vllm_frame);
        let selected = causal_selection_for_test(engine.clone()).await;
        let branches = (0..3).map(prepared_causal_branch).collect();
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

#[tokio::test]
async fn causal_dispatch_runs_only_one_sibling_at_a_time() {
    let engine = TestEngine::with_frame(EngineBehavior::PendingSiblings, vllm_frame);
    let parent = Context::new(()).context();
    let admission = Arc::new(Semaphore::new(5));
    let task = tokio::spawn(dispatch(
        causal_selection_for_test(engine.clone()).await,
        (0..5).map(prepared_causal_branch).collect(),
        "serial".into(),
        parent.clone(),
        admission.clone().acquire_many_owned(5).await.unwrap(),
        Arc::new(super::super::Metrics::new()),
        canonical_request(5),
    ));
    tokio::time::timeout(Duration::from_secs(2), async {
        while engine.observations.lock().unwrap().len() < 2 {
            engine.started.notified().await;
        }
    })
    .await
    .unwrap();
    tokio::time::sleep(Duration::from_millis(20)).await;
    let observed = engine.observations.lock().unwrap().len();
    parent.kill();
    let response = task.await.unwrap();
    assert_eq!(response.status().as_u16(), 499);
    assert_eq!(admission.available_permits(), 5);
    assert_eq!(
        observed, 2,
        "only the first pending sibling may reach the worker"
    );
}
