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

use dynamo_runtime::error::{BackendError, DynamoError, ErrorClass};
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
/// SECURITY: `.message` is diagnostic-only. HTTP-like exceptions can wrap remote
/// fetch failures whose text includes URLs or internal connection details. The
/// semantic status is preserved, but client renderers use the class catalog
/// unless the producer raised an explicitly public Dynamo exception.
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

/// Map an explicit Python HTTP exception status to Dynamo's semantic class.
pub fn error_class_for_http_status(code: u16) -> ErrorClass {
    match code {
        400 => ErrorClass::InvalidRequest,
        401 => ErrorClass::Unauthenticated,
        403 => ErrorClass::PermissionDenied,
        404 => ErrorClass::NotFound,
        408 => ErrorClass::DeadlineExceeded,
        409 => ErrorClass::Conflict,
        413 => ErrorClass::PayloadTooLarge,
        415 => ErrorClass::UnsupportedMedia,
        431 => ErrorClass::PayloadTooLarge,
        429 => ErrorClass::RateLimited,
        499 => ErrorClass::Cancelled,
        501 => ErrorClass::NotImplemented,
        502 => ErrorClass::BackendProtocol,
        503 => ErrorClass::Unavailable,
        504 => ErrorClass::DeadlineExceeded,
        529 => ErrorClass::CapacityExhausted,
        402 | 405..=407 | 410..=412 | 414 | 416..=428 | 430 | 432..=498 => {
            ErrorClass::InvalidRequest
        }
        _ => ErrorClass::Internal,
    }
}

pub(crate) fn http_like_error_to_dynamo(py: Python<'_>, err: &PyErr) -> Option<DynamoError> {
    let (code, message) = extract_http_like_error(py, err)?;
    Some(build_http_like_error(code, message))
}

fn build_http_like_error(code: u16, message: String) -> DynamoError {
    let legacy_message = serde_json::json!({
        "message": message,
        "code": code,
    })
    .to_string();

    DynamoError::builder()
        .class(error_class_for_http_status(code))
        .diagnostic(legacy_message)
        .build()
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde::Deserialize;

    #[derive(Deserialize)]
    struct LegacyDynamoError {
        message: String,
    }

    #[derive(Deserialize)]
    struct LegacyHttpError {
        code: u16,
        message: String,
    }

    #[test]
    fn http_like_errors_support_semantic_and_legacy_readers() {
        for (code, expected_class) in [
            (409, ErrorClass::Conflict),
            (415, ErrorClass::UnsupportedMedia),
            (503, ErrorClass::Unavailable),
        ] {
            let error = build_http_like_error(code, "private backend detail".to_string());
            assert_eq!(error.class(), expected_class);
            assert_eq!(error.public_message(), None);

            let wire = serde_json::to_value(&error).expect("serialize DynamoError");
            let semantic: DynamoError =
                serde_json::from_value(wire.clone()).expect("semantic reader");
            assert_eq!(semantic.class(), expected_class);
            assert_eq!(semantic.public_message(), None);

            let legacy: LegacyDynamoError = serde_json::from_value(wire).expect("legacy reader");
            let legacy_http: LegacyHttpError =
                serde_json::from_str(&legacy.message).expect("legacy HTTP payload");
            assert_eq!(legacy_http.code, code);
            assert_eq!(legacy_http.message, "private backend detail");
        }
    }
}
