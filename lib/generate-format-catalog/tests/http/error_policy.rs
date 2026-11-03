// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use super::*;
use dynamo_runtime::error::{ErrorReason, ErrorType as Backend};

struct ErrorEngine(Vec<Backend>, bool);

impl ErrorEngine {
    fn error(&self) -> DynamoError {
        let mut cause = None;
        for kind in self.0.iter().rev() {
            let mut error = DynamoError::builder()
                .error_type(*kind)
                .message("invalid params");
            if *kind == Backend::DeadlineExceeded {
                error = error.reason(ErrorReason::new("router.queue_deadline_exceeded").unwrap());
            }
            if let Some(cause) = cause {
                error = error.cause(cause);
            }
            cause = Some(error.build());
        }
        cause.unwrap()
    }
}

#[async_trait::async_trait]
impl AsyncEngine<Input, Output, Error> for ErrorEngine {
    async fn generate(&self, request: Input) -> Result<Output, Error> {
        let error = self.error();
        if !self.1 {
            return Err(error.into());
        }
        Ok(ResponseStream::new(
            Box::pin(futures::stream::iter([Annotated::from_err(error)])),
            request.context(),
        ))
    }
}

// Literal outcomes from the two handlers at 5235cfc93d. Each row gives
// dispatch, SGLang response-stream, and vLLM response-stream outcomes.
#[tokio::test]
async fn native_error_phase_and_nested_chain_compatibility() {
    use ErrorType::{Cancelled, Internal, Overload, Unavailable, Validation};
    for (kinds, expected) in [
        (
            vec![Backend::DeadlineExceeded],
            [
                (429, Cancelled, "request deadline exceeded"),
                (499, Cancelled, "request deadline exceeded"),
                (500, Internal, "internal server error"),
            ],
        ),
        (
            vec![Backend::WorkerOverloaded],
            [
                (503, Unavailable, "engine rejected the request"),
                (503, Overload, "Service temporarily overloaded"),
                (500, Internal, "internal server error"),
            ],
        ),
        (
            vec![Backend::WorkerUnavailable],
            [
                (503, Unavailable, "Service temporarily unavailable"),
                (503, Unavailable, "Service temporarily unavailable"),
                (503, Unavailable, "Service temporarily unavailable"),
            ],
        ),
        (
            vec![Backend::Cancelled],
            [
                (499, Cancelled, "request was cancelled"),
                (499, Cancelled, "Request cancelled"),
                (499, Cancelled, "request was cancelled"),
            ],
        ),
        (
            vec![Backend::WorkerUnavailable, Backend::Cancelled],
            [
                (499, Cancelled, "request was cancelled"),
                (503, Unavailable, "Service temporarily unavailable"),
                (499, Cancelled, "request was cancelled"),
            ],
        ),
        (
            vec![Backend::InvalidArgument, Backend::Cancelled],
            [
                (499, Cancelled, "request was cancelled"),
                (400, Validation, "invalid params"),
                (499, Cancelled, "request was cancelled"),
            ],
        ),
        (
            vec![Backend::WorkerOverloaded, Backend::DeadlineExceeded],
            [
                (429, Cancelled, "request deadline exceeded"),
                (499, Cancelled, "request deadline exceeded"),
                (500, Internal, "internal server error"),
            ],
        ),
        (
            vec![Backend::DeadlineExceeded, Backend::Cancelled],
            [
                (429, Cancelled, "request deadline exceeded"),
                (499, Cancelled, "request deadline exceeded"),
                (499, Cancelled, "request was cancelled"),
            ],
        ),
    ] {
        for is_sglang in [false, true] {
            for after_dispatch in [false, true] {
                for streaming in [false, true].into_iter().filter(|s| is_sglang || !s) {
                    let (status, metric, message) = expected[if !after_dispatch {
                        0
                    } else if is_sglang {
                        1
                    } else {
                        2
                    }]
                    .clone();
                    let state = state();
                    let format: SharedFormat = if is_sglang {
                        sglang()
                    } else {
                        Arc::new(dynamo_vllm_generate::VllmFormat)
                    };
                    let body = if is_sglang {
                        serde_json::json!({"input_ids":[1],"stream":streaming})
                    } else {
                        serde_json::json!({"token_ids":[1],"sampling_params":{}})
                    };
                    let response = respond(
                        state.clone(),
                        format,
                        Arc::new(ErrorEngine(kinds.clone(), after_dispatch)),
                        body,
                        HeaderMap::new(),
                    )
                    .await;
                    assert_eq!(
                        response.status().as_u16(),
                        if streaming && after_dispatch {
                            200
                        } else {
                            status
                        },
                        "{kinds:?}"
                    );
                    let bytes = axum::body::to_bytes(response.into_body(), usize::MAX)
                        .await
                        .unwrap();
                    let body = String::from_utf8(bytes.to_vec()).unwrap();
                    let body = if streaming && after_dispatch {
                        body.lines()
                            .find_map(|line| line.strip_prefix("data: "))
                            .unwrap()
                    } else {
                        &body
                    };
                    let body: Value = serde_json::from_str(body).unwrap();
                    let expected_body = if is_sglang {
                        serde_json::json!({"error":{"message":message}})
                    } else {
                        let kind = match status {
                            400 => "invalid_request_error",
                            429 => "deadline_exceeded",
                            499 => "request_cancelled",
                            503 => "service_unavailable",
                            _ => "internal_error",
                        };
                        serde_json::json!({"error":{"message":message,"type":kind,"code":status}})
                    };
                    assert_eq!(body, expected_body, "{kinds:?}");
                    assert_eq!(
                        state.metrics_clone().get_request_counter(
                            state.manager().metric_model_for("test-model"),
                            &Endpoint::Generate,
                            &if streaming {
                                RequestType::Stream
                            } else {
                                RequestType::Unary
                            },
                            &Status::Error,
                            &metric,
                        ),
                        1,
                        "{kinds:?}"
                    );
                    assert_eq!(
                        state
                            .metrics_clone()
                            .get_rejection_count("test-model", Endpoint::Generate),
                        u64::from(!after_dispatch && kinds == [Backend::WorkerOverloaded])
                    );
                }
            }
        }
    }
}
