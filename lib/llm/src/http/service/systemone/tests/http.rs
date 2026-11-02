// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use super::*;

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
        selection(engine.clone()),
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
