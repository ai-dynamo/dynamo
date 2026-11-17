// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use super::*;
use axum::{
    Json,
    extract::{FromRequest, rejection::JsonRejection},
    response::IntoResponse,
};

// The extractor/error boundaries of the handlers at 5235cfc93d, before the
// plugin refactor. Comparing real HTTP responses catches status, media-type,
// and serde-path drift that successful-request properties cannot detect.
async fn legacy_rejection(request: Request<Body>, sglang: bool) -> Response {
    let error = if sglang {
        Json::<dynamo_sglang_generate::generate::SglangGenerateRequest>::from_request(request, &())
            .await
            .unwrap_err()
    } else {
        Json::<dynamo_vllm_generate::generate::GenerateRequest>::from_request(request, &())
            .await
            .unwrap_err()
    };
    if sglang {
        (StatusCode::BAD_REQUEST, Json(serde_json::json!({
            "object": "error", "message": error.body_text(), "type": "Bad Request", "param": null, "code": 400,
        }))).into_response()
    } else if matches!(error, JsonRejection::JsonDataError(_)) {
        (
            StatusCode::BAD_REQUEST,
            Json(serde_json::json!({
                "message": error.body_text(), "type": "Bad Request", "code": 400,
            })),
        )
            .into_response()
    } else {
        error.into_response()
    }
}

#[tokio::test]
async fn native_parse_errors_match_the_original_http_contract() {
    for is_sglang in [false, true] {
        let format: SharedFormat = if is_sglang {
            sglang()
        } else {
            Arc::new(dynamo_vllm_generate::VllmFormat)
        };
        let valid = if is_sglang {
            r#"{"input_ids":[1]}"#
        } else {
            r#"{"token_ids":[1],"sampling_params":{}}"#
        };
        let wrong_type = if is_sglang {
            r#"{"input_ids":"bad"}"#
        } else {
            r#"{"token_ids":"bad","sampling_params":{}}"#
        };
        for (content_type, body) in [
            (None, Some(valid)),
            (Some("text/plain"), Some(valid)),
            (Some("application/json"), Some("{")),
            (Some("application/json"), Some("")),
            (Some("application/json"), Some(wrong_type)),
            (Some("application/problem+json"), Some("{}")),
            (Some("application/json"), None),
            (Some("text/plain"), None),
            (None, None),
        ] {
            let request = || {
                let mut request = Request::post("/generate");
                if let Some(content_type) = content_type {
                    request = request.header("content-type", content_type);
                }
                let body = match body {
                    Some(body) => Body::from(body),
                    None => {
                        Body::from_stream(futures::stream::iter([Err::<axum::body::Bytes, _>(
                            std::io::Error::other("broken body"),
                        )]))
                    }
                };
                request.body(body).unwrap()
            };
            let expected = legacy_rejection(request(), is_sglang).await;
            let (_, router) = format_generate::router(state(), None, format.clone());
            let actual = router.oneshot(request()).await.unwrap();
            assert_eq!(
                actual.status(),
                expected.status(),
                "{is_sglang} {content_type:?} {body:?}"
            );
            assert_eq!(
                actual.headers()["content-type"],
                expected.headers()["content-type"]
            );
            let json_body = actual.headers()["content-type"] == "application/json";
            let actual = axum::body::to_bytes(actual.into_body(), usize::MAX)
                .await
                .unwrap();
            let expected = axum::body::to_bytes(expected.into_body(), usize::MAX)
                .await
                .unwrap();
            if json_body {
                assert_eq!(
                    serde_json::from_slice::<Value>(&actual).unwrap(),
                    serde_json::from_slice::<Value>(&expected).unwrap()
                );
            } else {
                assert_eq!(actual, expected);
            }
        }
        let (_, router) = format_generate::router(state(), None, format);
        let response = router
            .oneshot(
                Request::put("/generate")
                    .header("content-type", "application/json")
                    .body(Body::from(valid))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(
            response.status(),
            if is_sglang {
                StatusCode::NOT_FOUND
            } else {
                StatusCode::METHOD_NOT_ALLOWED
            }
        );
    }
}

#[tokio::test]
async fn formats_preserve_typed_validation_error_policy_before_and_after_dispatch() {
    use dynamo_runtime::error::{BackendError, ErrorType as Backend};
    for is_sglang in [false, true] {
        for kind in [
            Backend::InvalidArgument,
            Backend::Backend(BackendError::InvalidArgument),
            Backend::InvalidRequest,
        ] {
            let is_private = kind == Backend::InvalidRequest;
            for after_dispatch in [false, true] {
                let expected_message = if is_sglang {
                    if is_private {
                        if after_dispatch {
                            "Internal server error"
                        } else {
                            "internal server error"
                        }
                    } else {
                        "private worker details"
                    }
                } else {
                    "Invalid request"
                };
                for streaming in [false, true]
                    .into_iter()
                    .filter(|streaming| is_sglang || !streaming)
                {
                    let format: SharedFormat = if is_sglang {
                        sglang()
                    } else {
                        Arc::new(dynamo_vllm_generate::VllmFormat)
                    };
                    let engine: dynamo_llm::types::openai::generate::GenerateStreamingEngine =
                        if after_dispatch {
                            let error = DynamoError::builder()
                                .error_type(kind)
                                .message("private worker details")
                                .build();
                            Arc::new(ReplayEngine(vec![Annotated::from_err(error)], false))
                        } else {
                            Arc::new(FailingEngine(kind))
                        };
                    let body = if is_sglang {
                        serde_json::json!({"input_ids":[1], "stream":streaming})
                    } else {
                        serde_json::json!({"token_ids":[1],"sampling_params":{}})
                    };
                    let response = respond(state(), format, engine, body, HeaderMap::new()).await;
                    let expected_status = if streaming && after_dispatch {
                        StatusCode::OK
                    } else if is_sglang && is_private {
                        StatusCode::INTERNAL_SERVER_ERROR
                    } else {
                        StatusCode::BAD_REQUEST
                    };
                    assert_eq!(response.status(), expected_status);
                    let bytes = axum::body::to_bytes(response.into_body(), usize::MAX)
                        .await
                        .unwrap();
                    let body = String::from_utf8(bytes.to_vec()).unwrap();
                    assert!(body.contains(expected_message), "{body}");
                    if !is_sglang || is_private {
                        assert!(!body.contains("private worker details"));
                    }
                }
            }
        }
    }
}

#[tokio::test]
async fn early_vllm_failures_keep_metrics_and_readiness_precedence() {
    for (body, draining, status, error_type) in [
        (
            serde_json::json!({"token_ids":[],"sampling_params":{}}),
            false,
            StatusCode::BAD_REQUEST,
            ErrorType::Validation,
        ),
        (
            serde_json::json!({"token_ids":[1],"sampling_params":{},"stream":true}),
            false,
            StatusCode::NOT_IMPLEMENTED,
            ErrorType::NotImplemented,
        ),
        (
            serde_json::json!({"token_ids":[],"sampling_params":{}}),
            true,
            StatusCode::SERVICE_UNAVAILABLE,
            ErrorType::Unavailable,
        ),
    ] {
        let state = state();
        if draining {
            state.service_observer().start_draining();
        }
        let response = respond(
            state.clone(),
            Arc::new(dynamo_vllm_generate::VllmFormat),
            Arc::new(ReplayEngine(vec![], false)),
            body,
            HeaderMap::new(),
        )
        .await;
        assert_eq!(response.status(), status);
        let bytes = axum::body::to_bytes(response.into_body(), usize::MAX)
            .await
            .unwrap();
        let response: Value = serde_json::from_slice(&bytes).unwrap();
        if draining {
            assert_eq!(
                response,
                serde_json::json!({
                    "message": "Service is not ready", "type": "Service Unavailable", "code": 503,
                })
            );
        } else {
            assert!(response.get("error").is_some(), "{response}");
        }
        assert_eq!(
            state.metrics_clone().get_request_counter(
                state.manager().metric_model_for("test-model"),
                &Endpoint::Generate,
                &RequestType::Unary,
                &Status::Error,
                &error_type
            ),
            1
        );
    }
}
