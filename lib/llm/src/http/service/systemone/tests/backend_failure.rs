// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use super::*;

struct EmptyNonFusedEngine;

#[async_trait]
impl AsyncEngine<SingleIn<PreprocessedRequest>, ManyOut<Annotated<LLMEngineOutput>>, Error>
    for EmptyNonFusedEngine
{
    async fn generate(
        &self,
        request: SingleIn<PreprocessedRequest>,
    ) -> Result<ManyOut<Annotated<LLMEngineOutput>>, Error> {
        crate::discovery::test_support::validate_fixture_dispatch(&request, 41);
        Ok(ResponseStream::new(
            Box::pin(stream::unfold((), |_| async {
                None::<(Annotated<LLMEngineOutput>, ())>
            })),
            request.context(),
        ))
    }
}

async fn assert_empty_nonfused_scoring_stream(profile: Profile) {
    for dialect in [
        dynamo_decisions::Dialect::Jev,
        dynamo_decisions::Dialect::OpenAi,
    ] {
        let base = selection_for_test(TestEngine::new(EngineBehavior::Success)).await;
        let mut card = base.context.card.clone();
        card.runtime_config.max_num_seqs = Some(1);
        card.runtime_config.data_parallel_size = 1;
        let selection = token_selection_for_test(
            card,
            Arc::new(EmptyNonFusedEngine),
            token_executor(&base).preprocessor().clone(),
            profile,
        )
        .await;
        let mut request = canonical_request(1);
        Arc::make_mut(&mut request).dialect = dialect;
        let parent = Context::new(()).context();
        let admission = Arc::new(Semaphore::new(1));
        let response = dispatch(
            selection,
            vec![prepared_noul_branch(0)],
            "empty-nonfused".into(),
            parent.clone(),
            admission.clone().acquire_owned().await.unwrap(),
            Arc::new(super::super::Metrics::new()),
            request,
        )
        .await;
        assert_eq!(response.status(), StatusCode::BAD_GATEWAY);
        let body = response_body(response).await;
        let code = body.pointer("/error/code").or_else(|| body.get("code"));
        assert_eq!(code, Some(&json!("invalid_executor_result")));
        assert!(body.get("answers").is_none());
        assert!(parent.is_killed());
        assert_eq!(admission.available_permits(), 1);
    }
}

#[tokio::test]
async fn sglang_empty_nonfused_scoring_stream_releases_admission() {
    assert_empty_nonfused_scoring_stream(Profile::SglangSerialV1).await;
}

#[tokio::test]
async fn vllm_empty_nonfused_scoring_stream_releases_admission() {
    assert_empty_nonfused_scoring_stream(Profile::VllmCausalV1).await;
}

#[test]
fn executor_errors_use_a_profile_neutral_message_and_distinct_origin_status() {
    let executor = super::super::DispatchError::invalid_executor("private token frame");
    assert_eq!(executor.status, StatusCode::BAD_GATEWAY);
    assert_eq!(executor.code, Some("invalid_executor_result"));
    assert_eq!(
        executor.public_message,
        "invalid decision executor response"
    );
    let frontend = super::super::DispatchError::malformed("private construction detail");
    assert_eq!(frontend.status, StatusCode::INTERNAL_SERVER_ERROR);
    assert!(frontend.code.is_none());
    assert_eq!(
        frontend.public_message,
        "decision request construction failed"
    );
}

#[tokio::test]
async fn token_backend_validation_errors_preserve_dialect_specific_status() {
    use dynamo_runtime::error::ErrorClass;
    for class in [ErrorClass::InvalidRequest, ErrorClass::InvalidArgument] {
        for (dialect, expected) in [
            (dynamo_decisions::Dialect::OpenAi, StatusCode::BAD_REQUEST),
            (
                dynamo_decisions::Dialect::Jev,
                StatusCode::UNPROCESSABLE_ENTITY,
            ),
        ] {
            let engine = TestEngine::new(EngineBehavior::RejectRequest {
                class,
                branch: 0,
                in_stream: false,
            });
            let mut request = canonical_request(1);
            Arc::make_mut(&mut request).dialect = dialect;
            let response = dispatch(
                selection_for_test(engine).await,
                vec![prepared_noul_branch(0)],
                "invalid".into(),
                Context::new(()).context(),
                Arc::new(Semaphore::new(1)).acquire_owned().await.unwrap(),
                Arc::new(super::super::Metrics::new()),
                request,
            )
            .await;
            assert_eq!(response.status(), expected);
            assert!(response_body(response).await.get("answers").is_none());
        }
    }
}

#[test]
fn derives_only_a_proven_single_data_parallel_rank() {
    assert_eq!(resolve_dp_rank(None, 7, 1), Some(7));
    assert_eq!(resolve_dp_rank(None, 7, 2), None);
    assert_eq!(resolve_dp_rank(Some(9), 7, 2), Some(9));
}

#[tokio::test]
async fn backend_overload_preserves_retry_status_and_releases_admission() {
    for behavior in [EngineBehavior::RejectGenerate, EngineBehavior::RejectStream] {
        let (response, parent, semaphore) =
            dispatch_request(TestEngine::new(behavior), 2, "request").await;
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
        let body = response_body(response).await;
        assert!(body.get("answers").is_none());
        assert!(!body.to_string().contains("private worker detail"));
        assert!(parent.is_killed());
        assert_eq!(semaphore.available_permits(), 2);
    }
}

#[tokio::test]
async fn deterministic_worker_rejections_match_for_first_and_pinned_branches() {
    use dynamo_runtime::error::ErrorClass;
    for (class, status) in [
        (ErrorClass::InvalidRequest, StatusCode::UNPROCESSABLE_ENTITY),
        (ErrorClass::NotFound, StatusCode::NOT_FOUND),
    ] {
        for branch in [0, 1] {
            for in_stream in [false, true] {
                let (response, parent, admission) = dispatch_request(
                    TestEngine::new(EngineBehavior::RejectRequest {
                        class,
                        branch,
                        in_stream,
                    }),
                    2,
                    "invalid-request",
                )
                .await;
                assert_eq!(response.status(), status);
                assert!(!response.headers().contains_key("retry-after"));
                let body = response_body(response).await;
                assert!(body.get("answers").is_none());
                assert!(
                    !body
                        .to_string()
                        .contains("private deterministic worker detail")
                );
                assert!(parent.is_killed());
                assert_eq!(admission.available_permits(), 2);
            }
        }
    }
}

#[tokio::test]
async fn invalid_worker_accounting_never_projects_a_normal_answer() {
    use dynamo_runtime::error::ErrorClass;
    for in_stream in [false, true] {
        let (response, parent, admission) = dispatch_request(
            TestEngine::new(EngineBehavior::RejectRequest {
                class: ErrorClass::BackendProtocol,
                branch: 0,
                in_stream,
            }),
            2,
            "invalid-accounting",
        )
        .await;
        assert_eq!(response.status(), StatusCode::BAD_GATEWAY);
        let body = response_body(response).await;
        assert_eq!(body["code"], "invalid_executor_result");
        assert!(body.get("answers").is_none());
        assert!(
            !body
                .to_string()
                .contains("private deterministic worker detail")
        );
        assert!(parent.is_killed());
        assert_eq!(admission.available_permits(), 2);
    }
}

#[tokio::test]
async fn unclassified_transport_stream_failure_is_unavailable() {
    let (response, parent, admission) = dispatch_request(
        TestEngine::new(EngineBehavior::TransportFailure),
        1,
        "transport-failure",
    )
    .await;
    assert_eq!(response.status(), StatusCode::SERVICE_UNAVAILABLE);
    let body = response_body(response).await;
    assert!(body.get("answers").is_none());
    assert!(!body.to_string().contains("private transport detail"));
    assert!(parent.is_killed());
    assert_eq!(admission.available_permits(), 1);
}

#[tokio::test]
async fn admitted_worker_removal_during_fanout_does_not_reroute_siblings() {
    let (admissions, receiver) = tokio::sync::watch::channel(vec![41, 99]);
    let engine = Arc::new(TestEngine {
        resume_sibling: None,
        behavior: EngineBehavior::RemoveWorkerDuringFanout,
        observations: Mutex::new(Vec::new()),
        started: Notify::new(),
        sibling_started: Notify::new(),
        admissions: Some(admissions),
        frame: sglang_frame,
    });
    let base = selection_for_test(engine.clone()).await;
    let card = base.context.card.clone();
    let workers = WorkerSet::new("removal".into(), card.mdcsum().into(), card);
    let resources = crate::discovery::test_support::token_resources(
        engine.clone(),
        token_executor(&base).preprocessor().clone(),
    );
    let selection = crate::discovery::test_support::forced_selection_with_admissions_for_test(
        "test-model".into(),
        Profile::SglangSerialV1,
        workers,
        resources,
        Some(receiver),
    )
    .await
    .unwrap();
    let parent = Context::new(()).context();
    let admission = Arc::new(Semaphore::new(3));
    let permit = admission.clone().acquire_many_owned(3).await.unwrap();
    let response = dispatch(
        selection,
        (0..3).map(prepared_noul_branch).collect(),
        "worker-removal".into(),
        parent.clone(),
        permit,
        Arc::new(super::super::super::metrics::Metrics::new()),
        canonical_request(3),
    )
    .await;
    assert_eq!(response.status(), StatusCode::SERVICE_UNAVAILABLE);
    assert!(response_body(response).await.get("answers").is_none());
    assert!(parent.is_killed());
    assert_eq!(admission.available_permits(), 3);
    let observations = engine.observations.lock().unwrap();
    assert_eq!(observations.len(), 2);
    assert_eq!(observations[1].routing.backend_instance_id, Some(41));
}

#[test]
fn canonical_backend_errors_preserve_client_error_classification() {
    use dynamo_runtime::error::{DynamoError, ErrorClass};
    for (class, expected) in [
        (ErrorClass::InvalidRequest, StatusCode::UNPROCESSABLE_ENTITY),
        (ErrorClass::NotFound, StatusCode::NOT_FOUND),
    ] {
        let cause = DynamoError::builder().class(class).build();
        assert_eq!(
            super::super::backend_error::classify(&cause).status,
            expected
        );
    }
    let cause = DynamoError::builder().class(ErrorClass::Internal).build();
    assert_eq!(
        super::super::backend_error::classify(&cause).status,
        StatusCode::INTERNAL_SERVER_ERROR
    );
}
