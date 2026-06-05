// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Error types for the service shell. `ServiceError` converts cleanly into
//! `tonic::Status` for gRPC responses and into an axum `IntoResponse` for the
//! HTTP sidecar.

use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use thiserror::Error;
use tonic::Status;

pub type ServiceResult<T> = Result<T, ServiceError>;

#[derive(Debug, Error)]
pub enum ServiceError {
    #[error("invalid request: {0}")]
    InvalidArgument(String),

    /// Another tenant already holds the service with a different key.
    #[error("registration key conflict: {0}")]
    KeyConflict(String),

    /// Same-key registration but slot capacity is exhausted.
    #[error("no capacity available: {0}")]
    NoCapacity(String),

    /// Service is in the middle of a graceful shutdown; new registrations
    /// are not accepted.
    #[error("service is shutting down: {0}")]
    ShuttingDown(String),

    /// Stream lifecycle ended (client disconnected or server shut down).
    #[error("registration stream closed: {0}")]
    StreamClosed(String),

    /// The requested feature is reserved but not yet supported.
    #[error("not yet implemented: {0}")]
    Unimplemented(String),

    #[error("io error: {0}")]
    Io(#[from] std::io::Error),

    #[error("internal error: {0}")]
    Internal(String),
}

impl ServiceError {
    pub fn internal<E: std::fmt::Display>(err: E) -> Self {
        Self::Internal(err.to_string())
    }
}

impl From<ServiceError> for Status {
    fn from(err: ServiceError) -> Self {
        match err {
            ServiceError::InvalidArgument(msg) => Status::invalid_argument(msg),
            ServiceError::KeyConflict(msg) => Status::failed_precondition(msg),
            ServiceError::NoCapacity(msg) => Status::resource_exhausted(msg),
            ServiceError::ShuttingDown(msg) => Status::unavailable(msg),
            ServiceError::StreamClosed(msg) => Status::cancelled(msg),
            ServiceError::Unimplemented(msg) => Status::unimplemented(msg),
            ServiceError::Io(e) => Status::internal(format!("io: {e}")),
            ServiceError::Internal(msg) => Status::internal(msg),
        }
    }
}

impl IntoResponse for ServiceError {
    fn into_response(self) -> Response {
        let status = match &self {
            ServiceError::InvalidArgument(_) => StatusCode::BAD_REQUEST,
            ServiceError::KeyConflict(_) => StatusCode::CONFLICT,
            ServiceError::NoCapacity(_) => StatusCode::SERVICE_UNAVAILABLE,
            ServiceError::ShuttingDown(_) => StatusCode::SERVICE_UNAVAILABLE,
            ServiceError::StreamClosed(_) => StatusCode::GONE,
            ServiceError::Unimplemented(_) => StatusCode::NOT_IMPLEMENTED,
            ServiceError::Io(_) | ServiceError::Internal(_) => StatusCode::INTERNAL_SERVER_ERROR,
        };
        let body = serde_json::json!({ "error": self.to_string() });
        (status, axum::Json(body)).into_response()
    }
}
