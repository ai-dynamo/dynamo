// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use super::*;

async fn assert_rollout_http_status(
    router: &axum::Router,
    expected: StatusCode,
    expected_code: Option<&str>,
) {
    use tower::ServiceExt;
    for (path, wire) in [
        (
            "/v1/decisions",
            json!({"model":"test-model", "input":"test", "questions":[{"name":"route", "type":"choice", "instructions":"Choose a team.", "choices":[{"value":"billing", "description":"Payments"}, {"value":"technical", "description":"Software"}]}]}),
        ),
        (
            "/v1/systemone",
            json!({"model":"test-model", "state":"test", "questions":{"route":{"type":"choice", "instructions":"Choose a team.", "criteria":{"billing":"Payments", "technical":"Software"}}}}),
        ),
    ] {
        let response = router
            .clone()
            .oneshot(
                axum::http::Request::post(path)
                    .header("content-type", "application/json")
                    .header("x-request-id", "rollout-contract")
                    .body(axum::body::Body::from(serde_json::to_vec(&wire).unwrap()))
                    .unwrap(),
            )
            .await
            .unwrap();
        let status = response.status();
        assert_eq!(response.headers()["x-dynamo-systemone-version"], "1");
        let request_id = response.headers()["x-request-id"].clone();
        assert!(!request_id.is_empty());
        assert_eq!(response.headers()["x-typesafe-request-id"], request_id);
        let body = response_body(response).await;
        assert_eq!(status, expected, "{path}: {body}");
        assert_eq!(body.get("answers").is_some(), expected == StatusCode::OK);
        if let Some(expected_code) = expected_code {
            let code = if path == "/v1/systemone" {
                &body["code"]
            } else {
                &body["error"]["code"]
            };
            assert_eq!(code, expected_code, "{path}: {body}");
        }
    }
}

#[tokio::test]
async fn decision_rollout_and_closed_publication_are_retryable_in_both_dialects() {
    use crate::discovery::{DecisionAvailability, test_support};
    let service = super::super::service_v2::HttpService::builder()
        .build()
        .unwrap();
    let state = service.state_clone();
    let (_, router) = super::super::router(state.clone(), None);
    let fixture = super::preflight::preflight_selection_for_test(None).await;
    let engine = TestEngine::with_frame(EngineBehavior::Success, |request| {
        let (wire, mut terminal) = sglang_frame(request);
        terminal
            .data
            .as_mut()
            .unwrap()
            .engine_data
            .as_mut()
            .unwrap()["sglang_response"]["meta_info"]["cached_tokens"] = json!(0);
        (wire, terminal)
    });
    let mut workers = WorkerSet::new(
        "rollout".into(),
        "fixture".into(),
        fixture.context.card.clone(),
    );
    let resources = test_support::token_resources(
        engine.clone(),
        token_executor(&fixture).preprocessor().clone(),
    );
    let (publisher, fleet) = test_support::fleet(None, vec![]);
    let (_membership, available) = tokio::sync::watch::channel(vec![41]);
    workers.set_instance_watcher(available);
    test_support::attach_surface_for_test(&mut workers, resources, fleet).await;
    assert!(
        state
            .manager()
            .add_worker_set("test-model", "rollout", workers)
    );
    let qualification = (
        fixture.context.profile(),
        fixture.context.decision_capabilities().max_candidates,
    );
    for (availability, expected) in [
        (DecisionAvailability::Unadvertised, StatusCode::BAD_REQUEST),
        (
            DecisionAvailability::Pending,
            StatusCode::SERVICE_UNAVAILABLE,
        ),
        (
            DecisionAvailability::qualified(qualification, vec![41]),
            StatusCode::OK,
        ),
        (
            DecisionAvailability::Pending,
            StatusCode::SERVICE_UNAVAILABLE,
        ),
        (
            DecisionAvailability::qualified(qualification, vec![]),
            StatusCode::SERVICE_UNAVAILABLE,
        ),
        (
            DecisionAvailability::qualified(qualification, vec![41]),
            StatusCode::OK,
        ),
    ] {
        publisher.send_replace(availability);
        let model = state.manager().get_committed_model("test-model").unwrap();
        let workers = &model.worker_sets()[0];
        assert!(workers.has_any_serving_engine());
        assert_eq!(
            workers.decision_state().is_ready(),
            expected == StatusCode::OK
        );
        assert_eq!(
            workers.decision_worker_count(),
            usize::from(expected == StatusCode::OK)
        );
        let calls_before = engine.observations.lock().unwrap().len();
        assert_rollout_http_status(&router, expected, None).await;
        assert_eq!(
            engine.observations.lock().unwrap().len(),
            calls_before + if expected == StatusCode::OK { 2 } else { 0 }
        );
    }
    drop(publisher);
    assert_rollout_http_status(&router, StatusCode::SERVICE_UNAVAILABLE, None).await;
}

#[tokio::test]
async fn persistent_native_qualification_failure_is_nonretryable_in_both_dialects() {
    let service = super::super::service_v2::HttpService::builder()
        .build()
        .unwrap();
    let state = service.state_clone();
    let (_, router) = super::super::router(state.clone(), None);
    let fixture = super::preflight::preflight_selection_for_test(None).await;
    let engine = TestEngine::new(EngineBehavior::Success);
    let mut card = fixture.context.card.clone();
    card.model_input = crate::model_type::ModelInput::Tokens;
    card.model_type = crate::model_type::ModelType::Chat;
    card.worker_type = Some(crate::worker_type::WorkerType::Aggregated);
    card.runtime_config.data_parallel_size = 1;
    let profiles = card.advertised_decision_profiles().collect::<Vec<_>>();
    for profile in profiles {
        card.runtime_config
            .runtime_data
            .remove(profile.decision_capability());
    }
    card.runtime_config.runtime_data.insert(
        "vllm_native_decision_qualification_failure_v1".into(),
        json!({"profile":"vllm_liquid_decision_v1"}),
    );
    assert_eq!(
        card.decision_qualification_failure(),
        Some(Profile::VllmLiquidV1)
    );
    let mut workers = WorkerSet::new("failed".into(), "fixture".into(), card);
    let resources = crate::discovery::test_support::token_resources(
        engine.clone(),
        token_executor(&fixture).preprocessor().clone(),
    );
    let (_membership, available) = tokio::sync::watch::channel(vec![41]);
    workers.set_instance_watcher(available);
    let _publication =
        crate::discovery::test_support::attach_card_surface_for_test(&mut workers, resources).await;
    assert!(
        state
            .manager()
            .add_worker_set("test-model", "failed", workers)
    );
    assert_rollout_http_status(
        &router,
        StatusCode::BAD_REQUEST,
        Some("model_qualification_failed"),
    )
    .await;
    assert!(engine.observations.lock().unwrap().is_empty());
    let metrics = state.metrics_clone();
    let model = state.manager().metric_model_for("test-model");
    for endpoint in [
        super::super::super::metrics::Endpoint::Decisions,
        super::super::super::metrics::Endpoint::SystemOne,
    ] {
        for (classification, expected) in [
            (super::super::super::metrics::ErrorType::Internal, 1),
            (super::super::super::metrics::ErrorType::Validation, 0),
        ] {
            assert_eq!(
                metrics.get_request_counter(
                    model,
                    &endpoint,
                    &super::super::super::metrics::RequestType::Unary,
                    &super::super::super::metrics::Status::Error,
                    &classification,
                ),
                expected,
            );
        }
    }
}

#[tokio::test]
async fn lazy_token_configuration_failure_is_terminal_from_first_http_request() {
    use crate::discovery::{DecisionResources, ScoreFactory, TokenizerLoad, test_support};
    let (_runtime, load) = test_support::routing_load_for_test().await;
    let directory = tempfile::tempdir().unwrap();
    let tokenizer = directory.path().join("tokenizer.json");
    std::fs::write(&tokenizer, b"{}").unwrap();
    let fixture = super::preflight::preflight_selection_for_test(None).await;
    let engine = TestEngine::new(EngineBehavior::Success);
    let mut card = fixture.context.card.clone();
    card.model_input = crate::model_type::ModelInput::Tokens;
    card.model_type = crate::model_type::ModelType::Chat;
    card.worker_type = Some(crate::worker_type::WorkerType::Aggregated);
    card.runtime_config.data_parallel_size = 1;
    card.runtime_config.max_num_seqs = Some(1);
    for capability in [
        crate::decision_profile::SGLANG_SYSTEMONE_SERIAL_V1,
        crate::local_model::runtime_config::SGLANG_GENERATE_CAPABILITY,
    ] {
        card.runtime_config
            .set_engine_specific(capability, true)
            .unwrap();
    }
    card.tokenizer = Some(crate::model_card::TokenizerKind::HfTokenizerJson(
        crate::common::checked_file::CheckedFile::from_disk(&tokenizer).unwrap(),
    ));
    let qualification = card.decision_qualification().unwrap();
    let mut workers = WorkerSet::new("lazy-failure".into(), "fixture".into(), card.clone());
    let resources = DecisionResources::for_worker(
        card,
        Some(ScoreFactory::new(engine.clone(), TokenizerLoad::Card)),
        Some(load),
        dynamo_runtime::pipeline::RouterMode::RoundRobin,
        workers.initialize_allocator_trim_on_teardown(),
    );
    let resources = resources.into_shared();
    let (_publication, fleet) = test_support::fleet(Some(qualification), vec![41]);
    let (_available, membership) = tokio::sync::watch::channel(vec![41]);
    workers.set_instance_watcher(membership);
    test_support::attach_surface_for_test(&mut workers, resources, fleet).await;
    let service = super::super::service_v2::HttpService::builder()
        .build()
        .unwrap();
    let state = service.state_clone();
    let (_, router) = super::super::router(state.clone(), None);
    assert!(
        state
            .manager()
            .add_worker_set("test-model", "lazy-failure", workers)
    );
    for _ in 0..3 {
        assert_rollout_http_status(&router, StatusCode::BAD_REQUEST, None).await;
        let model = state.manager().get_committed_model("test-model").unwrap();
        assert!(!model.has_decision_engine());
        assert!(!model.is_ready_to_serve());
    }
    assert!(engine.observations.lock().unwrap().is_empty());
}

#[tokio::test]
async fn eager_and_lazy_token_cell_outcomes_match_in_both_http_dialects() {
    let (_runtime, load) = crate::discovery::test_support::routing_load_for_test().await;
    use crate::discovery::{
        DecisionResources, DecisionState, ScoreFactory, TokenResources, TokenizerLoad, test_support,
    };

    #[derive(Clone, Copy, Debug)]
    enum Outcome {
        Ready,
        Retryable,
        Permanent,
    }

    for eager in [true, false] {
        for outcome in [Outcome::Ready, Outcome::Retryable, Outcome::Permanent] {
            let fixture = super::preflight::preflight_selection_for_test(None).await;
            let engine = TestEngine::with_frame(EngineBehavior::Success, |request| {
                let (wire, mut terminal) = sglang_frame(request);
                terminal
                    .data
                    .as_mut()
                    .unwrap()
                    .engine_data
                    .as_mut()
                    .unwrap()["sglang_response"]["meta_info"]["cached_tokens"] = json!(0);
                (wire, terminal)
            });
            let mut card = fixture.context.card.clone();
            card.model_input = crate::model_type::ModelInput::Tokens;
            card.model_type = crate::model_type::ModelType::Chat;
            card.worker_type = Some(crate::worker_type::WorkerType::Aggregated);
            card.runtime_config.data_parallel_size = 1;
            card.runtime_config.max_num_seqs = Some(1);
            for capability in [
                crate::decision_profile::SGLANG_SYSTEMONE_SERIAL_V1,
                crate::local_model::runtime_config::SGLANG_GENERATE_CAPABILITY,
            ] {
                card.runtime_config
                    .set_engine_specific(capability, true)
                    .unwrap();
            }
            let directory = tempfile::tempdir().unwrap();
            let tokenizer = directory.path().join("tokenizer.json");
            if matches!(outcome, Outcome::Retryable) {
                std::fs::write(&tokenizer, b"{}").unwrap();
                card.tokenizer = Some(crate::model_card::TokenizerKind::HfTokenizerJson(
                    crate::common::checked_file::CheckedFile::from_disk(&tokenizer).unwrap(),
                ));
                std::fs::remove_file(&tokenizer).unwrap();
            }
            let qualification = card.decision_qualification().unwrap();
            let mut workers =
                WorkerSet::new("cell-matrix".into(), card.mdcsum().into(), card.clone());
            let resources = DecisionResources::for_worker(
                card,
                Some(ScoreFactory::new(engine.clone(), TokenizerLoad::Card)),
                Some(load.clone()),
                dynamo_runtime::pipeline::RouterMode::RoundRobin,
                workers.initialize_allocator_trim_on_teardown(),
            )
            .into_shared()
            .unwrap();
            let seed = || {
                let result = resources.initialize_token_for_test(|| match outcome {
                    Outcome::Ready => Ok(Arc::new(TokenResources {
                        engine: engine.clone(),
                        preprocessor: token_executor(&fixture).preprocessor().clone(),
                    })),
                    Outcome::Retryable => {
                        Err(std::io::Error::other("tokenizer unavailable").into())
                    }
                    Outcome::Permanent => Err(anyhow::anyhow!("invalid tokenizer configuration")),
                });
                assert_eq!(result.is_ok(), matches!(outcome, Outcome::Ready));
            };
            if eager {
                seed();
            }
            let (_publication, fleet) = test_support::fleet(Some(qualification), vec![41]);
            let (_membership, available) = tokio::sync::watch::channel(vec![41]);
            workers.set_instance_watcher(available);
            test_support::attach_surface_for_test(&mut workers, Some(resources.clone()), fleet)
                .await;
            if !eager {
                seed();
            }
            let permanent = matches!(outcome, Outcome::Permanent);
            assert_eq!(
                matches!(workers.decision_state(), DecisionState::Unqualified),
                permanent,
                "eager={eager}, outcome={outcome:?}"
            );
            assert_eq!(workers.decision_state().is_qualified(), !permanent);
            assert_eq!(workers.decision_state().is_ready(), !permanent);
            let service = super::super::service_v2::HttpService::builder()
                .build()
                .unwrap();
            let state = service.state_clone();
            let (_, router) = super::super::router(state.clone(), None);
            assert!(
                state
                    .manager()
                    .add_worker_set("test-model", "cell-matrix", workers)
            );
            let expected = match outcome {
                Outcome::Ready => StatusCode::OK,
                Outcome::Retryable => StatusCode::SERVICE_UNAVAILABLE,
                Outcome::Permanent => StatusCode::BAD_REQUEST,
            };
            assert_rollout_http_status(&router, expected, None).await;
            assert_eq!(
                engine.observations.lock().unwrap().len(),
                if matches!(outcome, Outcome::Ready) {
                    2
                } else {
                    0
                },
                "eager={eager}, outcome={outcome:?}"
            );
        }
    }
}

#[tokio::test]
async fn prepared_token_requests_isolate_salts_and_use_bounded_metric_labels() {
    let engine = TestEngine::new(EngineBehavior::Success);
    let metrics = Arc::new(super::super::Metrics::new_with_prefix(Some(
        "prepared_token".into(),
    )));
    let registry = prometheus::Registry::new();
    metrics.register(&registry).unwrap();
    let manager = crate::discovery::ModelManager::new();
    let label = manager.metric_model_for("test-model");
    assert_ne!(label, "test-model");
    for id in ["prepared-one", "prepared-two"] {
        let questions: serde_json::Map<String, Value> = (0..3).map(|index| (
            format!("question-{index}"),
            json!({"type": "choice", "instructions": "Choose a team.", "criteria": {"billing": "Payments", "technical": "Software"}}),
        )).collect();
        let request = Arc::new(
            dynamo_decisions::parse_request(
                &serde_json::to_vec(
                    &json!({"model": "test-model", "state": "test", "questions": questions}),
                )
                .unwrap(),
                dynamo_decisions::Route::SystemOne,
            )
            .unwrap(),
        );
        let fixture = super::preflight::preflight_selection_for_test(None).await;
        let selected = token_selection_for_test(
            fixture.context.card.clone(),
            engine.clone(),
            token_executor(&fixture).preprocessor().clone(),
            Profile::SglangSerialV1,
        )
        .await;
        let prepared =
            super::super::execution::prepare(&request, selected.selection.clone(), 65536).unwrap();
        super::super::execution::observe_work(label, &prepared, &metrics);
        let response = super::super::execution::dispatch(
            prepared,
            id.into(),
            Context::new(()).context(),
            Arc::new(Semaphore::new(3))
                .acquire_many_owned(3)
                .await
                .unwrap(),
            metrics.clone(),
            label.into(),
            request,
        )
        .await;
        assert_eq!(response.status(), StatusCode::OK);
    }
    let observed = engine.observations.lock().unwrap();
    assert_eq!(observed.len(), 6);
    let salt = &observed[0].native["cache_salt"];
    assert!(
        observed[..3]
            .iter()
            .all(|branch| &branch.native["cache_salt"] == salt)
    );
    let next = &observed[3].native["cache_salt"];
    assert!(
        observed[3..]
            .iter()
            .all(|branch| &branch.native["cache_salt"] == next)
    );
    assert_ne!(salt, next);
    super::metrics::assert_bounded_labels(
        &registry,
        "prepared_token_systemone_",
        label,
        &[
            "questions",
            "input_tokens",
            "candidate_ids",
            "admitted_branches_total",
            "phase_seconds",
        ],
    );
}

#[tokio::test]
async fn empty_canonical_questions_are_rejected_before_admission() {
    let service = super::super::service_v2::HttpService::builder()
        .build()
        .unwrap();
    let state = service.state_clone();
    let admission = state.systemone_admission();
    let before = admission.available_permits();
    for dialect in [
        dynamo_decisions::Dialect::OpenAi,
        dynamo_decisions::Dialect::Jev,
    ] {
        let request = dynamo_decisions::CanonicalRequest {
            dialect,
            questions: Vec::new(),
            ..canonical_request(1).as_ref().clone()
        };
        let response = super::super::handle_request(state.clone(), "empty".into(), request).await;
        assert_eq!(
            response.status().as_u16(),
            if dialect == dynamo_decisions::Dialect::Jev {
                422
            } else {
                400
            }
        );
        let body = response_body(response).await;
        if dialect == dynamo_decisions::Dialect::Jev {
            assert!(body["detail"].is_array());
        } else {
            assert_eq!(body["error"]["type"], "invalid_request_error");
        }
        assert_eq!(admission.available_permits(), before);
    }
}

#[tokio::test]
async fn dialect_errors_preserve_request_ids_and_validation_envelopes() {
    use dynamo_decisions::{DecisionError, Dialect};
    for dialect in [Dialect::OpenAi, Dialect::Jev] {
        let mut response =
            super::super::decision_error(DecisionError::validation(dialect, "invalid questions"));
        super::super::attach_request_ids(&mut response, "fixture-request");
        assert_eq!(response.headers()["x-request-id"], "fixture-request");
        assert_eq!(response.headers()["x-dynamo-systemone-version"], "1");
        assert_eq!(
            response.headers()["x-typesafe-request-id"],
            "fixture-request"
        );
        assert_eq!(
            response.status().as_u16(),
            if dialect == Dialect::Jev { 422 } else { 400 }
        );
        let body = response_body(response).await;
        match dialect {
            Dialect::OpenAi => assert_eq!(body["error"]["type"], "invalid_request_error"),
            Dialect::Jev => assert!(body["detail"].is_array()),
        }
    }
}

#[test]
fn missing_cache_measurement_is_not_zero() {
    assert_eq!(
        super::super::measured_cached_tokens(&json!({"meta_info":{}}), 4).unwrap(),
        None
    );
    assert_eq!(
        super::super::measured_cached_tokens(&json!({"meta_info":{"cached_tokens":0}}), 4).unwrap(),
        Some(0)
    );
    assert!(
        super::super::measured_cached_tokens(&json!({"meta_info":{"cached_tokens":5}}), 4).is_err()
    );
    assert!(
        super::super::measured_cached_tokens(&json!({"meta_info":{"cached_tokens":-1}}), 4)
            .is_err()
    );
}

#[tokio::test]
async fn deadline_kills_stalled_dispatch_and_releases_local_admission() {
    let engine = TestEngine::new(EngineBehavior::PendingStream);
    let parent = Context::new(()).context();
    let admission = Arc::new(Semaphore::new(1));
    let permit = admission.clone().acquire_owned().await.unwrap();
    let task = tokio::spawn(dispatch(
        selection_for_test(engine.clone()).await,
        vec![prepared_noul_branch(0)],
        "deadline".into(),
        parent.clone(),
        permit,
        Arc::new(super::super::super::metrics::Metrics::new()),
        canonical_request(1),
    ));
    tokio::time::timeout(Duration::from_secs(2), engine.started.notified())
        .await
        .unwrap();
    let response = super::super::response::dispatch_with_deadline(
        task,
        tokio::time::Instant::now() + Duration::from_millis(20),
        parent.clone(),
        dynamo_decisions::Dialect::Jev,
    )
    .await;
    assert_eq!(response.status(), StatusCode::GATEWAY_TIMEOUT);
    assert!(parent.is_killed());
    let recovered = tokio::time::timeout(Duration::from_secs(1), admission.clone().acquire_owned())
        .await
        .unwrap()
        .unwrap();
    drop(recovered);
    assert_eq!(admission.available_permits(), 1);
    assert!(
        engine
            .observations
            .lock()
            .unwrap()
            .iter()
            .all(|o| o.request_context.is_killed())
    );
}

#[tokio::test]
async fn empty_and_closed_admission_are_not_capacity_rejections() {
    for dialect in [
        dynamo_decisions::Dialect::Jev,
        dynamo_decisions::Dialect::OpenAi,
    ] {
        let admission = Arc::new(Semaphore::new(1));
        let empty = *super::super::acquire_admission(admission.clone(), 0, 1, dialect).unwrap_err();
        assert_eq!(
            empty.status().as_u16(),
            if dialect == dynamo_decisions::Dialect::Jev {
                422
            } else {
                400
            }
        );
        assert!(
            empty
                .headers()
                .get(axum::http::header::RETRY_AFTER)
                .is_none()
        );
        admission.close();
        let closed = *super::super::acquire_admission(admission, 1, 1, dialect).unwrap_err();
        assert_eq!(closed.status(), StatusCode::SERVICE_UNAVAILABLE);
        assert!(
            closed
                .headers()
                .get(axum::http::header::RETRY_AFTER)
                .is_none()
        );
        let body = response_body(closed).await;
        assert_ne!(body["code"], "rate_limit_exceeded");
        assert_ne!(body["error"]["code"], "rate_limit_exceeded");
    }
}

#[test]
fn impossible_admission_is_not_retryable_but_busy_capacity_is() {
    let admission = Arc::new(Semaphore::new(2));
    let response =
        super::super::acquire_admission(admission.clone(), 3, 2, dynamo_decisions::Dialect::Jev)
            .unwrap_err();
    assert_eq!(response.status(), StatusCode::UNPROCESSABLE_ENTITY);
    assert!(!response.headers().contains_key("retry-after"));
    assert!(
        response
            .extensions()
            .get::<super::super::DecisionCapacityRejection>()
            .is_none()
    );
    let permit =
        super::super::acquire_admission(admission.clone(), 2, 2, dynamo_decisions::Dialect::Jev)
            .unwrap();
    let response =
        super::super::acquire_admission(admission.clone(), 1, 2, dynamo_decisions::Dialect::Jev)
            .unwrap_err();
    assert_eq!(
        response.status(),
        super::super::super::error::overload_status_code()
    );
    assert_eq!(response.headers()["retry-after"], "1");
    assert!(
        response
            .extensions()
            .get::<super::super::DecisionCapacityRejection>()
            .is_some()
    );
    drop(permit);
    assert!(
        super::super::acquire_admission(admission, 1, 2, dynamo_decisions::Dialect::Jev).is_ok()
    );
}

#[tokio::test]
async fn capacity_error_code_is_independent_of_configured_status() {
    for dialect in [
        dynamo_decisions::Dialect::OpenAi,
        dynamo_decisions::Dialect::Jev,
    ] {
        for status in [429, 503, 529] {
            let parent = Context::new(()).context();
            let response = super::super::dispatch_error(
                "capacity",
                &parent,
                super::super::DispatchError {
                    code: None,
                    status: StatusCode::from_u16(status).unwrap(),
                    public_message: "capacity exhausted",
                    detail: "private detail".into(),
                    retry_after: true,
                    is_capacity_exhausted: true,
                },
                dialect,
            );
            assert_eq!(response.status().as_u16(), status);
            assert_eq!(response.headers()["retry-after"], "1");
            assert!(
                response
                    .extensions()
                    .get::<super::super::DecisionCapacityRejection>()
                    .is_some()
            );
            let body = response_body(response).await;
            let code = if dialect == dynamo_decisions::Dialect::Jev {
                &body["code"]
            } else {
                &body["error"]["code"]
            };
            assert_eq!(code, "rate_limit_exceeded");
            assert!(!body.to_string().contains("private detail"));
        }
    }
}

#[tokio::test]
async fn unavailable_with_retry_after_does_not_claim_capacity() {
    for dialect in [
        dynamo_decisions::Dialect::OpenAi,
        dynamo_decisions::Dialect::Jev,
    ] {
        let parent = Context::new(()).context();
        let response = super::super::dispatch_error(
            "unavailable",
            &parent,
            super::super::DispatchError {
                retry_after: true,
                ..super::super::DispatchError::unavailable("private detail")
            },
            dialect,
        );
        assert_eq!(response.status(), StatusCode::SERVICE_UNAVAILABLE);
        assert_eq!(response.headers()["retry-after"], "1");
        assert!(
            response
                .extensions()
                .get::<super::super::DecisionCapacityRejection>()
                .is_none()
        );
        let body = response_body(response).await;
        let code = if dialect == dynamo_decisions::Dialect::Jev {
            &body["code"]
        } else {
            &body["error"]["code"]
        };
        assert_eq!(code, "server_error");
    }
}
