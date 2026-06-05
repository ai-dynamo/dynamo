// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Helpers for constructing backend-common errors with stable backend error types.

use dynamo_backend_common::{BackendError, DynamoError, ErrorType};

/// Builds an invalid-argument backend error.
pub(crate) fn invalid_arg(msg: impl Into<String>) -> DynamoError {
    DynamoError::builder()
        .error_type(ErrorType::Backend(BackendError::InvalidArgument))
        .message(msg)
        .build()
}

/// Builds a connection-failure backend error.
pub(crate) fn cannot_connect(msg: impl Into<String>) -> DynamoError {
    DynamoError::builder()
        .error_type(ErrorType::Backend(BackendError::CannotConnect))
        .message(msg)
        .build()
}

/// Builds an engine-shutdown backend error.
pub(crate) fn engine_shutdown(msg: impl Into<String>) -> DynamoError {
    DynamoError::builder()
        .error_type(ErrorType::Backend(BackendError::EngineShutdown))
        .message(msg)
        .build()
}

/// Builds a generic unknown backend error.
pub(crate) fn backend_unknown(msg: impl Into<String>) -> DynamoError {
    DynamoError::builder()
        .error_type(ErrorType::Backend(BackendError::Unknown))
        .message(msg)
        .build()
}

/// Converts a clap parser error into Dynamo's invalid-argument error shape.
pub(crate) fn clap_error(error: clap::Error) -> DynamoError {
    invalid_arg(error.to_string())
}
