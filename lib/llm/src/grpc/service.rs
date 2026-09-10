// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

pub mod kserve;
pub mod openai;
pub mod tensor;

use tonic::Status;

use crate::http::service::error::SanitizedError;
use crate::http::service::metrics::request_was_unavailable;

/// Map a dispatch error that is not a client-visible rejection onto its gRPC status.
///
/// Worker-scoped and pool-scoped unavailability both mean the request found no
/// servable worker, so both answer `UNAVAILABLE` (14) and let the client retry.
/// Anything else is a server fault and keeps `INTERNAL` (13). The unavailable
/// message is sanitized to match the HTTP frontends; only the internal arm keeps
/// the caller's context string.
pub(crate) fn dispatch_error_status(
    error: &(dyn std::error::Error + 'static),
    internal_context: &str,
) -> Status {
    if request_was_unavailable(error) {
        return Status::unavailable(SanitizedError::Unavailable.to_string());
    }
    Status::internal(format!("{internal_context}: {error}"))
}

#[cfg(test)]
mod tests {
    use super::*;
    use dynamo_runtime::error::{DynamoError, ErrorType};

    fn error(error_type: ErrorType) -> anyhow::Error {
        DynamoError::builder()
            .error_type(error_type)
            .message("boom")
            .build()
            .into()
    }

    #[test]
    fn unavailable_dispatch_errors_map_to_grpc_unavailable() {
        for error_type in [ErrorType::WorkerUnavailable, ErrorType::Unavailable] {
            let status = dispatch_error_status(error(error_type).as_ref(), "ctx");
            assert_eq!(status.code(), tonic::Code::Unavailable, "{error_type}");
            assert_eq!(status.message(), "Service temporarily unavailable");
        }
    }

    #[test]
    fn other_dispatch_errors_stay_internal_with_context() {
        let status = dispatch_error_status(error(ErrorType::Unknown).as_ref(), "ctx");
        assert_eq!(status.code(), tonic::Code::Internal);
        assert!(status.message().starts_with("ctx: "));
    }
}
