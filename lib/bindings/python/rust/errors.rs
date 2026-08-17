// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Python exception types mirroring Dynamo's [`ErrorType`] enum.
//!
//! The [`define_dynamo_exceptions!`] macro auto-generates a Python exception class
//! for each Dynamo error variant, a conversion function from Python exceptions back
//! to [`DynamoError`], and a registration function for the `_core` module.
//!
//! When new variants are added to [`ErrorType`] or [`BackendError`], add a
//! corresponding entry to the macro invocation below to keep Python exceptions
//! in sync.

use dynamo_runtime::error::{BackendError, ErrorType};
use pyo3::prelude::*;
use pyo3::types::PyModule;

// Base exception for all Dynamo errors.
pyo3::create_exception!(dynamo._core, DynamoException, pyo3::exceptions::PyException);
pyo3::create_exception!(dynamo._core, RouterQueueLimitExceeded, DynamoException);

pub fn queue_rejection_to_pyerr(rejection: dynamo_kv_router::scheduling::QueueRejection) -> PyErr {
    let error = PyErr::new::<RouterQueueLimitExceeded, _>(rejection.to_string());
    let attributes = Python::with_gil(|py| -> PyResult<()> {
        let value = error.value(py);
        value.setattr("policy_class", &rejection.policy_class)?;
        value.setattr("limit_kind", rejection.limit_kind.to_string())?;
        value.setattr("current", rejection.current)?;
        value.setattr("limit", rejection.limit)?;
        Ok(())
    });
    if let Err(error) = attributes {
        return error;
    }
    error
}

// Raised by the in-process `SelectionService` bindings for selector failures
// that are not malformed input. Instances carry `kind` (a stable,
// machine-readable category) and `status_code` (an HTTP-style status) so
// callers can branch without matching on the message string.
pyo3::create_exception!(dynamo._core, SelectionServiceError, DynamoException);

/// Defines Python exception classes for each Dynamo error type.
///
/// For each `(RustExceptionName, BackendError)` pair, the macro:
/// 1. Creates a Python exception class inheriting from `DynamoException`
/// 2. Adds it to `py_exception_to_backend_error()` for Python → `BackendError` extraction
/// 3. Adds it to `register_exceptions()` for module registration
///
/// The conversion intentionally returns a `BackendError` variant and message
/// rather than a fully constructed `DynamoError`. This lets the caller decide
/// how to wrap it — backend contexts use `ErrorType::Backend(...)`, while
/// other contexts could map to top-level `ErrorType` variants.
macro_rules! define_dynamo_exceptions {
    ( $( ($name:ident, $backend_error:expr) ),* $(,)? ) => {
        $(
            pyo3::create_exception!(dynamo._core, $name, DynamoException);
        )*

        /// Extract a [`BackendError`] variant from a Python exception if it is
        /// a known Dynamo exception.
        ///
        /// Returns `Some((BackendError, message))` if the exception is a Dynamo
        /// exception, `None` otherwise. The caller decides how to wrap the
        /// `BackendError` into an `ErrorType`.
        pub fn py_exception_to_backend_error(
            py: Python<'_>,
            err: &PyErr,
        ) -> Option<(BackendError, String)> {
            // Check specific subtypes first (most-specific match wins).
            $(
                if err.is_instance_of::<$name>(py) {
                    let message = err
                        .value(py)
                        .str()
                        .map(|s| s.to_string_lossy().into_owned())
                        .unwrap_or_default();
                    return Some(($backend_error, message));
                }
            )*

            // Fall back: check if it's a bare DynamoException (Unknown).
            if err.is_instance_of::<DynamoException>(py) {
                let message = err
                    .value(py)
                    .str()
                    .map(|s| s.to_string_lossy().into_owned())
                    .unwrap_or_default();
                return Some((BackendError::Unknown, message));
            }

            None
        }

        /// Register all Dynamo exception classes on the `_core` module.
        pub fn register_exceptions(m: &Bound<'_, PyModule>) -> PyResult<()> {
            m.add("DynamoException", m.py().get_type::<DynamoException>())?;
            m.add(
                "RouterQueueLimitExceeded",
                m.py().get_type::<RouterQueueLimitExceeded>(),
            )?;
            m.add(
                "SelectionServiceError",
                m.py().get_type::<SelectionServiceError>(),
            )?;
            $(
                m.add(stringify!($name), m.py().get_type::<$name>())?;
            )*
            Ok(())
        }
    };
}

// ---------------------------------------------------------------------------
// Exception definitions — one entry per BackendError variant.
//
// All error types are exposed to Python as exception classes. When raised by
// Python backend code, they are interpreted as Backend* errors in Rust
// (e.g., raising `InvalidArgument` in Python becomes `BackendInvalidArgument`
// on the Rust side).
//
// When a new variant is added to BackendError in error.rs, add a
// corresponding line here so that a Python exception is generated.
// ---------------------------------------------------------------------------
define_dynamo_exceptions!(
    (Unknown, BackendError::Unknown),
    (InvalidArgument, BackendError::InvalidArgument),
    (CannotConnect, BackendError::CannotConnect),
    (Disconnected, BackendError::Disconnected),
    (ConnectionTimeout, BackendError::ConnectionTimeout),
    (Cancelled, BackendError::Cancelled),
    (EngineShutdown, BackendError::EngineShutdown),
    (StreamIncomplete, BackendError::StreamIncomplete),
);

/// Read `(code, message)` off a Python exception carrying an HTTP-style
/// status. Accepts `.code` (matches [`HttpError`] in `http.rs`) or `.status`
/// (matches `dynamo.common.http.HttpStatusError`) plus `.message`.
///
/// SECURITY: `.message` is forwarded verbatim to clients on 4xx responses
/// (HTTP protocol contract). Python callers must ensure it contains no
/// internal state, file paths, traceback strings, or backend identifiers.
/// Non-4xx codes (including 5xx) are sanitized downstream — the original
/// message survives in server logs only.
/// Classify a worker-supplied HTTP-like status code into a Dynamo error type.
///
/// 503 is how a Python worker reports that it hit its own admission limit, e.g.
/// "Worker local total request limit reached (32/32)". Mapping it to
/// [`ErrorType::WorkerOverloaded`] rather than `Backend(Unknown)` puts the
/// category in the error chain, which is what
/// `dynamo_llm::http::service::metrics::request_was_rejected` matches on, so the
/// frontend applies its configured overload status instead of forwarding the
/// worker's 503.
///
/// `WorkerOverloaded` rather than `ResourceExhausted`: the worker is stating that
/// *it* is full, not that the eligible pool is exhausted, so the request stays
/// eligible for migration to another worker.
pub fn error_type_for_http_like_code(code: u16) -> ErrorType {
    match code {
        503 => ErrorType::WorkerOverloaded,
        400..=499 => ErrorType::Backend(BackendError::InvalidArgument),
        _ => ErrorType::Backend(BackendError::Unknown),
    }
}

pub fn extract_http_like_error(py: Python<'_>, err: &PyErr) -> Option<(u16, String)> {
    let value = err.value(py);
    let code = value
        .getattr("code")
        .ok()
        .and_then(|a| a.extract::<u16>().ok())
        .or_else(|| {
            value
                .getattr("status")
                .ok()
                .and_then(|a| a.extract::<u16>().ok())
        })?;
    let message = value.getattr("message").ok()?.extract::<String>().ok()?;
    Some((code, message))
}

#[cfg(test)]
mod http_like_code_tests {
    use super::error_type_for_http_like_code;
    use dynamo_runtime::error::{BackendError, ErrorType};

    #[test]
    fn worker_capacity_rejection_is_typed_not_backend_unknown() {
        // A Python worker rejecting on its own admission limit reports 503.
        // Before this mapping it fell through to Backend(Unknown), which
        // request_was_rejected does not match, so the worker's 503 reached the
        // client instead of the frontend's configured overload status.
        assert_eq!(
            error_type_for_http_like_code(503),
            ErrorType::WorkerOverloaded
        );
    }

    #[test]
    fn client_errors_stay_invalid_argument_and_other_server_errors_stay_unknown() {
        for code in [400, 404, 422, 499] {
            assert_eq!(
                error_type_for_http_like_code(code),
                ErrorType::Backend(BackendError::InvalidArgument),
                "code {code}"
            );
        }
        for code in [500, 502, 504] {
            assert_eq!(
                error_type_for_http_like_code(code),
                ErrorType::Backend(BackendError::Unknown),
                "code {code}"
            );
        }
    }
}
