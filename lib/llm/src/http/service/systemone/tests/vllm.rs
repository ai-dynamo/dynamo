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

pub(super) fn terminal(
    ids: &[u32],
    tokens: usize,
    cached_tokens: usize,
    outcome: Outcome,
) -> LLMEngineOutput {
    let mut output: LLMEngineOutput = serde_json::from_value(json!({
        "token_ids": [99], "index": 0, "finish_reason": "length",
        "top_logprobs": [[
            {"rank": 2, "token_id": ids[0], "logprob": -2.0},
            {"rank": 3, "token_id": ids[1], "logprob": -3.0},
            {"rank": 1, "token_id": 99, "logprob": -0.5}
        ]],
        "completion_usage": {"prompt_tokens": tokens, "completion_tokens": 1,
            "total_tokens": tokens + 1, "prompt_tokens_details": {"cached_tokens": cached_tokens}}
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

pub(super) fn configure_selection(selected: &mut SystemOneExecutionSelection) {
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
}

#[tokio::test]
async fn mixed_profiles_route_the_whole_request_before_admission() {
    let service = super::super::super::service_v2::HttpService::builder()
        .build()
        .unwrap();
    let state = service.state_clone();
    let engine = TestEngine::new(EngineBehavior::Success);
    let register = |namespace: &str, backend| {
        let mut selected = selection(engine.clone());
        if backend == ScoringBackend::Vllm {
            configure_selection(&mut selected);
        } else {
            selected.card.worker_type = Some(crate::worker_type::WorkerType::Aggregated);
            selected.card.runtime_config.max_num_seqs = Some(1);
            selected.card.runtime_config.data_parallel_size = 1;
            for capability in ["sglang_generate", "sglang_systemone_serial_v1"] {
                selected
                    .card
                    .runtime_config
                    .runtime_data
                    .insert(capability.into(), true.into());
            }
        }
        assert_eq!(selected.card.systemone_backend(), Some(backend));
        let mut workers =
            crate::discovery::WorkerSet::new(namespace.into(), namespace.into(), selected.card);
        workers.generate_engine = Some(selected.engine);
        workers.systemone_preprocessor = Some(selected.preprocessor);
        let (members, receiver) = tokio::sync::watch::channel(vec![41]);
        workers.set_instance_watcher(receiver);
        assert!(
            state
                .manager()
                .add_worker_set("test-model", namespace, workers)
        );
        members
    };
    let vllm = register("vllm", ScoringBackend::Vllm);
    let sglang = register("sglang", ScoringBackend::Sglang);
    assert!(state.manager().register_alias("public-model", "test-model"));
    let admission = state.systemone_admission();
    let preflight = state.systemone_preflight_admission();
    let _permit = admission
        .clone()
        .acquire_many_owned(admission.available_permits() as u32)
        .await
        .unwrap();
    let before = (admission.available_permits(), preflight.available_permits());
    for dialect in [
        dynamo_decisions::Dialect::Jev,
        dynamo_decisions::Dialect::OpenAi,
    ] {
        for (vllm_live, sglang_live, candidates, expected) in [
            (true, false, 26, dynamo_error::http::overload_status_code()),
            (true, false, 27, StatusCode::SERVICE_UNAVAILABLE),
            (false, true, 27, dynamo_error::http::overload_status_code()),
            (false, true, 255, dynamo_error::http::overload_status_code()),
            (true, true, 27, dynamo_error::http::overload_status_code()),
            (false, false, 27, StatusCode::SERVICE_UNAVAILABLE),
        ] {
            vllm.send_replace(if vllm_live { vec![41, 42] } else { vec![] });
            sglang.send_replace(if sglang_live { vec![43] } else { vec![] });
            let criteria: serde_json::Map<String, Value> = (0..candidates)
                .map(|index| (format!("candidate-{index}"), json!("A candidate")))
                .collect();
            let mut request = dynamo_decisions::parse_request(
                &serde_json::to_vec(&json!({"model":"public-model", "state":"test",
                "questions":{
                    "a-predicate":{"type":"noul", "instructions":"urgent?"},
                    "z-choice":{"type":"choice", "criteria":criteria}
                }}))
                .unwrap(),
                dynamo_decisions::Route::SystemOne,
            )
            .unwrap();
            assert_eq!(request.questions[0].candidates.len(), 2);
            assert_eq!(request.questions[1].candidates.len(), candidates);
            request.dialect = dialect;
            if expected == dynamo_error::http::overload_status_code() {
                let selected = state
                    .manager()
                    .get_systemone_execution_selection("public-model", candidates)
                    .unwrap();
                assert_eq!(selected.canonical_model, "test-model");
                assert_eq!(
                    selected.backend,
                    if sglang_live {
                        ScoringBackend::Sglang
                    } else {
                        ScoringBackend::Vllm
                    }
                );
                assert_eq!(
                    selected.decision_capabilities().max_candidates,
                    if sglang_live { 255 } else { 26 }
                );
            }
            let response =
                super::super::handle_request(state.clone(), "mixed-profiles".into(), request).await;
            assert_eq!(
                response.status(),
                expected,
                "{dialect:?}, {candidates}, vllm={vllm_live}, sglang={sglang_live}"
            );
            assert_eq!(
                (admission.available_permits(), preflight.available_permits()),
                before
            );
            assert!(engine.observations.lock().unwrap().is_empty());
        }
    }
}

#[tokio::test]
async fn candidate_limit_is_reported_and_rejected_before_admission() {
    let service = super::super::super::service_v2::HttpService::builder()
        .build()
        .unwrap();
    let state = service.state_clone();
    let engine = TestEngine::new(EngineBehavior::Vllm(Outcome::Success));
    let mut selected = selection(engine.clone());
    configure_selection(&mut selected);
    let mut workers =
        crate::discovery::WorkerSet::new("worker".into(), "worker".into(), selected.card);
    workers.generate_engine = Some(selected.engine);
    workers.systemone_preprocessor = Some(selected.preprocessor);
    let (_members, receiver) = tokio::sync::watch::channel(vec![41]);
    workers.set_instance_watcher(receiver);
    assert!(
        state
            .manager()
            .add_worker_set("test-model", "worker", workers)
    );
    let capabilities = state
        .manager()
        .get_systemone_execution_selection("test-model", 2)
        .unwrap()
        .decision_capabilities();
    assert_eq!(capabilities.max_candidates, 26);
    let criteria: serde_json::Map<String, Value> = (0..27)
        .map(|index| (format!("candidate-{index}"), json!("A candidate")))
        .collect();
    let admission = state.systemone_admission();
    let preflight = state.systemone_preflight_admission();
    let _admission_permit = admission
        .clone()
        .acquire_many_owned(admission.available_permits() as u32)
        .await
        .unwrap();
    let _preflight_permit = preflight
        .clone()
        .acquire_many_owned(preflight.available_permits() as u32)
        .await
        .unwrap();
    let before = (admission.available_permits(), preflight.available_permits());
    for (dialect, status) in [
        (dynamo_decisions::Dialect::Jev, 422),
        (dynamo_decisions::Dialect::OpenAi, 400),
    ] {
        let mut request = dynamo_decisions::parse_request(
            &serde_json::to_vec(&json!({"model":"test-model", "state":"test",
                "questions":{"choice":{"type":"choice", "criteria":criteria}}}))
            .unwrap(),
            dynamo_decisions::Route::SystemOne,
        )
        .unwrap();
        request.dialect = dialect;
        let response =
            super::super::handle_request(state.clone(), "candidate-limit".into(), request).await;
        assert_eq!(response.status().as_u16(), status);
        assert_eq!(
            (admission.available_permits(), preflight.available_permits()),
            before
        );
        assert!(engine.observations.lock().unwrap().is_empty());
    }
}

pub(super) fn prepared_branch(index: usize) -> PreparedBranch {
    let mut branch = prepared_noul_branch(index);
    branch.prompt_ids.push(3);
    branch.native = Some(
        super::super::causal_scoring_request(&branch.prompt_ids, &branch.label_ids, "preflight")
            .unwrap(),
    );
    branch
}

async fn run_causal(
    dialect: dynamo_decisions::Dialect,
    outcome: Outcome,
) -> (Response, Arc<TestEngine>) {
    let engine = TestEngine::new(EngineBehavior::Vllm(outcome));
    let mut selected = selection(engine.clone());
    configure_selection(&mut selected);
    let mut request = canonical_request(3).as_ref().clone();
    request.dialect = dialect;
    let branches = (0..3).map(prepared_branch).collect();
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
        let expected_usage = match dialect {
            dynamo_decisions::Dialect::OpenAi => json!({
                "input_tokens": 9,
                "output_tokens": 3,
                "total_tokens": 12,
                "input_tokens_details": {"cached_tokens": 6, "cache_write_tokens": 0},
                "output_tokens_details": {"reasoning_tokens": 0}
            }),
            dynamo_decisions::Dialect::Jev => json!({"input_tokens": 9, "output_tokens": 3}),
        };
        assert_eq!(body["usage"], expected_usage);
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
        1,
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
