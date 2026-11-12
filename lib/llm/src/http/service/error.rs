// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use dynamo_runtime::error::{DynamoError, ErrorType as DynamoErrorType};
use thiserror::Error;

/// Implementation of the Completion Engines served by the HTTP service should
/// map their custom errors to to this error type if they wish to return error
/// codes besides 500.
#[derive(Debug, Error)]
#[error("HTTP Error {code}: {message}")]
pub struct HttpError {
    pub code: u16,
    pub message: String,
}

/// Construct a typed invalid-argument error for validation performed at an
/// HTTP protocol adapter boundary.
pub(crate) fn invalid_argument(message: impl Into<String>) -> DynamoError {
    let message = message.into();
    DynamoError::builder()
        .error_type(DynamoErrorType::InvalidArgument)
        .message(message.clone())
        .public_message(message)
        .build()
}

#[cfg(test)]
mod tests {
    #[test]
    fn overload_status_env_name_matches_runtime_registry() {
        assert_eq!(
            dynamo_error::http::DYN_HTTP_OVERLOAD_STATUS_CODE,
            dynamo_runtime::config::environment_names::llm::DYN_HTTP_OVERLOAD_STATUS_CODE
        );
    }
}
