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

use dynamo_runtime::error::{BackendError, DynamoError, ErrorClass, ErrorReason, PublicDetails};
use dynamo_runtime::protocols::annotated::Annotated;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyModule};

// Base exception for all Dynamo errors.
pyo3::create_exception!(dynamo._core, DynamoException, pyo3::exceptions::PyException);
pyo3::create_exception!(dynamo._core, RouterQueueLimitExceeded, DynamoException);
pyo3::create_exception!(dynamo._core, ClassifiedError, DynamoException);

/// Construct a semantic engine error without choosing a transport status.
/// Public details must be fixed or explicitly allowlisted by the producer.
#[pyfunction]
#[pyo3(signature = (error_class, *, reason=None, diagnostic=None, public_details=None))]
pub fn classified_error(
    py: Python<'_>,
    error_class: &str,
    reason: Option<&str>,
    diagnostic: Option<String>,
    public_details: Option<&Bound<'_, PyDict>>,
) -> PyResult<Py<PyAny>> {
    let class: ErrorClass = serde_json::from_value(serde_json::json!(error_class))
        .map_err(|_| pyo3::exceptions::PyValueError::new_err("Invalid error class"))?;
    if class.as_str() != error_class || class != class.normalized() {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "Expected a canonical error class",
        ));
    }
    let mut builder = DynamoError::builder().class(class);
    if let Some(reason) = reason {
        builder = builder.reason(ErrorReason::new(reason).map_err(|_| {
            pyo3::exceptions::PyValueError::new_err("Unknown or malformed error reason")
        })?);
    }
    if let Some(diagnostic) = diagnostic {
        builder = builder.diagnostic(diagnostic);
    }
    if let Some(public_details) = public_details {
        let details: PublicDetails = pythonize::depythonize(public_details)
            .map_err(|_| pyo3::exceptions::PyValueError::new_err("Invalid public error details"))?;
        builder = match details {
            PublicDetails::Message { message } => builder.public_message(message),
            details => builder.public_details(details),
        };
    }
    let error = builder.build();
    if error.class() != class || reason.is_some_and(|reason| error.reason().as_str() != reason) {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "Error reason does not match its class",
        ));
    }
    Ok(dynamo_error_to_pyerr(error).into_value(py).into())
}

/// Preserve the canonical identity even when an anyhow context wraps the source.
pub(crate) fn pipeline_error_to_pyerr(error: anyhow::Error) -> PyErr {
    if let Some(semantic) = error
        .chain()
        .find_map(|cause| cause.downcast_ref::<DynamoError>())
    {
        return dynamo_error_to_pyerr(semantic.clone());
    }
    if let Some(rejection) = error
        .chain()
        .find_map(|cause| cause.downcast_ref::<dynamo_kv_router::scheduling::QueueRejection>())
    {
        return queue_rejection_to_pyerr(rejection.clone());
    }
    dynamo_error_to_pyerr(DynamoError::from(error.as_ref()))
}

pub(crate) fn dynamo_error_to_pyerr(error: DynamoError) -> PyErr {
    let python_error = match public_invalid_request_message(&error) {
        Some(message) => InvalidArgument::new_err(message.to_owned()),
        None => ClassifiedError::new_err(format!(
            "{} ({})",
            error.class().as_str(),
            error.reason().as_str()
        )),
    };
    attach_dynamo_error(python_error, error)
}

fn attach_dynamo_error(python_error: PyErr, error: DynamoError) -> PyErr {
    Python::with_gil(|py| -> PyResult<()> {
        let value = python_error.value(py);
        value.setattr(
            "_dynamo_error",
            serde_json::to_string(&error).map_err(crate::to_pyerr)?,
        )?;
        value.setattr("error_class", error.class().as_str())?;
        value.setattr("reason", error.reason().as_str())?;
        value.setattr(
            "diagnostic",
            error.diagnostic().map(|detail| detail.as_str()),
        )?;
        value.setattr(
            "public_details",
            pythonize::pythonize(py, &error.public_details())?,
        )?;
        Ok(())
    })
    .err()
    .unwrap_or(python_error)
}

pub(crate) fn py_exception_to_dynamo_error(py: Python<'_>, error: &PyErr) -> Option<DynamoError> {
    if !error.is_instance_of::<ClassifiedError>(py)
        && !error.is_instance_of::<InvalidArgument>(py)
        && !error.is_instance_of::<RouterQueueLimitExceeded>(py)
    {
        return None;
    }
    let payload = match error.value(py).getattr("_dynamo_error") {
        Ok(payload) => payload,
        Err(_) => return None,
    };
    Some(
        payload
            .extract::<String>()
            .ok()
            .and_then(|payload| serde_json::from_str(&payload).ok())
            .unwrap_or_else(|| {
                DynamoError::builder()
                    .class(ErrorClass::Internal)
                    .reason(
                        ErrorReason::new("runtime.invalid_error").expect("registered error reason"),
                    )
                    .diagnostic("Invalid classified Python error payload")
                    .build()
            }),
    )
}

pub fn queue_rejection_to_pyerr(rejection: dynamo_kv_router::scheduling::QueueRejection) -> PyErr {
    let semantic = DynamoError::builder()
        .class(ErrorClass::CapacityExhausted)
        .diagnostic(rejection.to_string())
        .build();
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
    attach_dynamo_error(error, semantic)
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
            m.add("ClassifiedError", m.py().get_type::<ClassifiedError>())?;
            m.add_function(wrap_pyfunction!(classified_error, m)?)?;
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
    (WorkerOverloaded, BackendError::WorkerOverloaded),
);

/// Preserve the canonical error and explicitly approved public details.
pub(crate) fn check_response_error<R>(response: Annotated<R>) -> PyResult<Annotated<R>> {
    if response.is_error() {
        let error = response.error.clone().unwrap_or_else(|| {
            DynamoError::msg(
                response
                    .comment
                    .as_ref()
                    .map(|comments| comments.join(", "))
                    .unwrap_or_else(|| "unknown error".to_owned()),
            )
        });
        return Err(dynamo_error_to_pyerr(error));
    }
    Ok(response)
}

fn public_invalid_request_message(error: &DynamoError) -> Option<&str> {
    if error.class() == ErrorClass::InvalidRequest {
        error.public_message()
    } else {
        None
    }
}

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
    fn invalid_request_conversion_uses_only_public_message() {
        let error = DynamoError::builder()
            .class(ErrorClass::InvalidRequest)
            .diagnostic("private backend detail")
            .public_message("Invalid JSON schema")
            .build();
        assert_eq!(
            public_invalid_request_message(&error),
            Some("Invalid JSON schema")
        );

        let error = DynamoError::builder()
            .class(ErrorClass::InvalidRequest)
            .diagnostic("private validation detail")
            .build();
        assert_eq!(public_invalid_request_message(&error), None);
    }

    #[test]
    fn other_error_classes_are_not_promoted_to_invalid_arguments() {
        for class in [
            ErrorClass::Internal,
            ErrorClass::Cancelled,
            ErrorClass::Unavailable,
        ] {
            let error = DynamoError::builder()
                .class(class)
                .diagnostic("private backend detail")
                .public_message("Public error message")
                .build();
            assert_eq!(public_invalid_request_message(&error), None);
        }
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

#[cfg(test)]
mod semantic_roundtrip_tests {
    use super::*;
    use dynamo_runtime::error::{ErrorReason, PublicDetails};
    use dynamo_runtime::protocols::maybe_error::MaybeError;

    fn queue_rejection() -> dynamo_kv_router::scheduling::QueueRejection {
        dynamo_kv_router::scheduling::QueueRejection {
            policy_class: "latency".into(),
            limit_kind: dynamo_kv_router::scheduling::QueueLimitKind::Requests,
            current: 2,
            limit: 1,
        }
    }

    #[test]
    fn initial_queue_rejection_retains_capacity_identity() {
        pyo3::prepare_freethreaded_python();
        let error = anyhow::Error::new(queue_rejection()).context("prefill selection failed");
        let python_error = pipeline_error_to_pyerr(error);
        let restored = crate::engine::map_python_exception(python_error);
        assert_eq!(restored.class(), ErrorClass::CapacityExhausted);
        assert_eq!(restored.reason().as_str(), "capacity.exhausted");
        assert_eq!(restored.public_details(), None);
    }

    #[test]
    fn direct_queue_exception_keeps_attributes_and_semantics() {
        pyo3::prepare_freethreaded_python();
        let python_error = queue_rejection_to_pyerr(queue_rejection());
        Python::with_gil(|py| {
            assert!(python_error.is_instance_of::<RouterQueueLimitExceeded>(py));
            assert_eq!(
                python_error
                    .value(py)
                    .getattr("policy_class")
                    .unwrap()
                    .extract::<String>()
                    .unwrap(),
                "latency"
            );
            assert_eq!(
                python_error
                    .value(py)
                    .getattr("limit_kind")
                    .unwrap()
                    .extract::<String>()
                    .unwrap(),
                "requests"
            );
            assert_eq!(
                python_error
                    .value(py)
                    .getattr("current")
                    .unwrap()
                    .extract::<usize>()
                    .unwrap(),
                2
            );
            assert_eq!(
                python_error
                    .value(py)
                    .getattr("limit")
                    .unwrap()
                    .extract::<usize>()
                    .unwrap(),
                1
            );
        });
        let restored = crate::engine::map_python_exception(python_error);
        assert_eq!(restored.class(), ErrorClass::CapacityExhausted);
        assert_eq!(restored.reason().as_str(), "capacity.exhausted");
        assert_eq!(restored.public_details(), None);
    }

    #[test]
    fn annotated_error_preserves_canonical_identity_and_public_details() {
        pyo3::prepare_freethreaded_python();
        let original = DynamoError::builder()
            .class(ErrorClass::CapacityExhausted)
            .reason(ErrorReason::new("capacity.worker_overloaded").unwrap())
            .diagnostic("private engine detail")
            .public_details(PublicDetails::SizeLimit {
                limit: 4,
                actual: Some(5),
            })
            .build();
        let response = Annotated::<String>::from_err(original.clone());
        let python_error = check_response_error(response).unwrap_err();
        let restored = crate::engine::map_python_exception(python_error);
        assert_eq!(restored.class(), original.class());
        assert_eq!(restored.reason(), original.reason());
        assert_eq!(restored.public_details(), original.public_details());
        assert_eq!(restored.diagnostic(), original.diagnostic());
    }

    #[test]
    fn all_canonical_classes_survive_the_stream_bridge() {
        pyo3::prepare_freethreaded_python();
        for class in [
            ErrorClass::InvalidRequest,
            ErrorClass::Unauthenticated,
            ErrorClass::PermissionDenied,
            ErrorClass::NotFound,
            ErrorClass::Conflict,
            ErrorClass::PayloadTooLarge,
            ErrorClass::UnsupportedMedia,
            ErrorClass::RateLimited,
            ErrorClass::CapacityExhausted,
            ErrorClass::Unavailable,
            ErrorClass::BackendProtocol,
            ErrorClass::DeadlineExceeded,
            ErrorClass::NotImplemented,
            ErrorClass::Internal,
            ErrorClass::Cancelled,
        ] {
            let original = DynamoError::builder()
                .class(class)
                .diagnostic("private secret https://internal.invalid")
                .build();
            let python_error =
                check_response_error(Annotated::<String>::from_err(original.clone())).unwrap_err();
            assert!(!python_error.to_string().contains("secret"));
            let restored = crate::engine::map_python_exception(python_error);
            assert_eq!(restored.class(), class);
            assert_eq!(restored.reason(), original.reason());
            assert_eq!(restored.public_details(), None);
            assert_eq!(restored.diagnostic(), original.diagnostic());
        }
    }

    #[test]
    fn explicitly_public_validation_retains_legacy_type_and_full_identity() {
        pyo3::prepare_freethreaded_python();
        let original = DynamoError::builder()
            .class(ErrorClass::InvalidRequest)
            .reason(ErrorReason::new("request.invalid_argument").unwrap())
            .public_message("The message role is invalid")
            .diagnostic("private request body")
            .build();
        let python_error = dynamo_error_to_pyerr(original.clone());
        Python::with_gil(|py| assert!(python_error.is_instance_of::<InvalidArgument>(py)));
        let restored = crate::engine::map_python_exception(python_error);
        assert_eq!(restored.reason(), original.reason());
        assert_eq!(restored.public_details(), original.public_details());
        assert_eq!(restored.diagnostic(), original.diagnostic());
    }

    #[test]
    fn malformed_payload_fails_closed_without_public_details() {
        pyo3::prepare_freethreaded_python();
        let python_error = ClassifiedError::new_err("private forged diagnostic");
        Python::with_gil(|py| {
            python_error
                .value(py)
                .setattr("_dynamo_error", "invalid json")
                .unwrap()
        });
        let restored = crate::engine::map_python_exception(python_error);
        assert_eq!(restored.class(), ErrorClass::Internal);
        assert_eq!(restored.reason().as_str(), "runtime.invalid_error");
        assert_eq!(restored.public_details(), None);
        assert!(!restored.diagnostic().unwrap().as_str().contains("forged"));
    }
}
