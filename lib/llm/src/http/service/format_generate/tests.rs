// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use super::*;
use crate::generate_format::GenerateFormat;

struct TestFormat;
impl GenerateFormat for TestFormat {
    fn name(&self) -> &str {
        "test"
    }
    fn capability(&self) -> &'static str {
        "test"
    }
    fn open(&self, _: ClientRequest<'_>) -> Result<Opened, ClientError> {
        unreachable!()
    }
}

#[test]
fn model_aliases_are_deduplicated_before_implicit_selection() {
    let manager = crate::discovery::ModelManager::new();
    assert!(manager.register_alias("alias", "primary"));
    let models = ["primary", "alias", "other"].map(String::from).to_vec();
    assert_eq!(
        canonical_generate_models(&manager, models),
        ["other", "primary"]
    );
}

#[test]
fn stream_errors_keep_their_classification_and_shape() {
    use dynamo_runtime::error::{ErrorReason, ErrorType as DynamoErrorType};

    let typed = |error_type, message: &str| {
        DynamoError::builder()
            .error_type(error_type)
            .message(message)
            .build()
    };
    let deadline = DynamoError::builder()
        .error_type(DynamoErrorType::WorkerOverloaded)
        .message("worker overloaded")
        .cause(
            DynamoError::builder()
                .error_type(DynamoErrorType::DeadlineExceeded)
                .reason(ErrorReason::new("router.queue_deadline_exceeded").unwrap())
                .message("internal deadline detail")
                .build(),
        )
        .build();
    for (error, expected_type, expected_message) in [
        (
            typed(DynamoErrorType::InvalidArgument, "bad params"),
            ErrorType::Validation,
            "Invalid request",
        ),
        (
            typed(DynamoErrorType::Unavailable, "x"),
            ErrorType::Unavailable,
            "Service temporarily unavailable",
        ),
        (
            typed(DynamoErrorType::WorkerUnavailable, "x"),
            ErrorType::Unavailable,
            "Service temporarily unavailable",
        ),
        (
            typed(DynamoErrorType::Cancelled, "x"),
            ErrorType::Cancelled,
            "Request cancelled",
        ),
        (
            DynamoError::msg("backend failure"),
            ErrorType::Internal,
            "Internal server error",
        ),
        // Deadline wins over overload.
        (deadline, ErrorType::Cancelled, "request deadline exceeded"),
    ] {
        let (error_type, body) = stream_error(
            &(Arc::new(TestFormat) as SharedFormat),
            &axum::Error::new(error),
        );
        let body: Value = serde_json::from_str(&body).unwrap();
        assert_eq!(error_type, expected_type);
        assert_eq!(
            body,
            serde_json::json!({"error": {"message": expected_message}})
        );
    }
}
