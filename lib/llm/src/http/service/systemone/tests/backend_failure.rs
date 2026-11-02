// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use super::*;

#[tokio::test]
async fn sglang_dispatch_rejects_mismatched_scoring_transport_before_starting() {
    for mismatched in [0, 1] {
        let engine = TestEngine::new(EngineBehavior::Success);
        let mut branches: Vec<_> = (0..2).map(prepared_noul_branch).collect();
        branches[mismatched].wire =
            super::super::PreparedScoringRequest::vllm(vec![1, 2], &[13, 14], "salt").unwrap();
        let parent = Context::new(()).context();
        let admission = Arc::new(Semaphore::new(2));
        let permit = admission.clone().acquire_many_owned(2).await.unwrap();
        let response = dispatch(
            selection(engine.clone()),
            branches,
            "mismatched-transport".into(),
            parent.clone(),
            permit,
            Arc::new(super::super::super::metrics::Metrics::new()),
            canonical_request(2),
        )
        .await;
        assert_eq!(response.status(), StatusCode::INTERNAL_SERVER_ERROR);
        assert!(response_body(response).await.get("answers").is_none());
        assert!(engine.observations.lock().unwrap().is_empty());
        assert!(parent.is_killed());
        assert_eq!(admission.available_permits(), 2);
    }
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
                selection(engine),
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
        behavior: EngineBehavior::RemoveWorkerDuringFanout,
        observations: Mutex::new(Vec::new()),
        started: Notify::new(),
        sibling_started: Notify::new(),
        admissions: Some(admissions),
        frame: sglang_frame,
    });
    let selection = SystemOneExecutionSelection {
        admitted_ids: Some(receiver),
        ..selection(engine.clone())
    };
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
