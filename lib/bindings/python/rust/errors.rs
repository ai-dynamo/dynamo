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

use dynamo_runtime::error::{
    BackendError, DynamoError, ErrorClass, PublicDetails, RetryHint, UpstreamError,
};
use dynamo_runtime::protocols::annotated::Annotated;
use pyo3::prelude::*;
use pyo3::types::{PyModule, PyString, PyTuple};

// Base exception for all Dynamo errors.
pyo3::create_exception!(dynamo._core, DynamoException, pyo3::exceptions::PyException);
pyo3::create_exception!(dynamo._core, RouterQueueLimitExceeded, DynamoException);

pub fn queue_rejection_to_pyerr(rejection: dynamo_admission::host::QueueRejection) -> PyErr {
    let error = PyErr::new::<RouterQueueLimitExceeded, _>(rejection.to_string());
    let attributes = Python::attach(|py| -> PyResult<()> {
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

// Raised by `RoutedEngine.generate` when the routed pipeline reports that no
// worker on the request's route can serve it yet (for example, prefill routing
// is still activating). Python chat processors re-raise it unchanged, and
// `map_python_exception` restores `ErrorClass::WorkerUnavailable` so the client
// gets the intended HTTP 503 instead of a generic 500.
pyo3::create_exception!(dynamo._core, WorkerUnavailable, DynamoException);
pyo3::create_exception!(dynamo._core, WorkerDraining, WorkerUnavailable);

// Raised by `RoutedEngine.generate` and its response stream for any other
// classified backend error. It subclasses `ValueError`, which the stream raised
// for these before, and keeps the backend's `DynamoError`: a chat engine that
// lets it escape answers with the backend's status, because
// `map_python_exception` restores the original error.
//
// Only a `RoutedEngine` raises it, and only a frontend chat engine holds one.
// A worker that relays another worker's stream (`Client`, `KvRouter`) must not
// forward that worker's error as its own: its caller would take a relayed
// `Disconnected` for the relay's and stop routing to it.
pyo3::create_exception!(
    dynamo._core,
    RoutedEngineError,
    pyo3::exceptions::PyValueError
);

/// Holds the JSON-serialized `DynamoError` a [`RoutedEngineError`] carries.
const ROUTED_ENGINE_ERROR_ATTR: &str = "_dynamo_error";
// Worker relays retain content without claiming the remote worker's failure identity.
const RELAY_CONTENT_ATTR: &str = "_dynamo_relay_content";

/// Mark serialized errors for full frontend restoration or content-only worker relaying.
fn attach_error(error: &DynamoError, exception: PyErr, routed_engine: bool) -> PyErr {
    if let Ok(serialized) = serde_json::to_string(error) {
        Python::attach(|py| {
            let attribute = if routed_engine {
                ROUTED_ENGINE_ERROR_ATTR
            } else {
                RELAY_CONTENT_ATTR
            };
            let _ = exception.value(py).setattr(attribute, serialized);
        });
    }
    exception
}

/// Decode a retained exception attribute through Dynamo's validated wire format.
fn error_attribute(py: Python<'_>, err: &PyErr, name: &str) -> Option<DynamoError> {
    let serialized: String = err.value(py).getattr(name).ok()?.extract().ok()?;
    serde_json::from_str(&serialized).ok()
}

/// Raise a backend's `DynamoError` in Python as a [`RoutedEngineError`] with
/// `message`, the text of the exception it replaces.
fn routed_engine_error(error: &DynamoError, message: String) -> PyErr {
    attach_error(error, RoutedEngineError::new_err(message), true)
}

/// Restore the complete error attached to a frontend routed or typed stream exception.
pub(crate) fn routed_engine_error_to_dynamo(py: Python<'_>, err: &PyErr) -> Option<DynamoError> {
    if !(err.is_instance_of::<RoutedEngineError>(py)
        || err.is_instance_of::<InvalidArgument>(py)
        || err.is_instance_of::<WorkerUnavailable>(py))
    {
        return None;
    }
    error_attribute(py, err, ROUTED_ENGINE_ERROR_ATTR)
}

/// Convert a routed-pipeline error to a Python exception, keeping the
/// `WorkerUnavailable` and `InvalidArgument` class that the generic [`crate::to_pyerr`] would drop,
/// and any other classified error as a [`RoutedEngineError`].
pub fn routed_error_to_pyerr(err: anyhow::Error) -> PyErr {
    let Some(error) = find_dynamo_error(err.as_ref()) else {
        return crate::to_pyerr(err);
    };
    let draining = is_worker_draining(err.as_ref());
    let unavailable = is_worker_unavailable(err.as_ref());
    let exception = if draining {
        WorkerDraining::new_err(err.to_string())
    } else if unavailable {
        WorkerUnavailable::new_err(err.to_string())
    } else if let Some(message) = public_invalid_argument_message(err.as_ref()) {
        InvalidArgument::new_err(message)
    } else {
        return routed_engine_error(error, err.to_string());
    };
    if unavailable
        && (error.error_type() != ErrorClass::WorkerUnavailable
            || (draining && error.retry_hint() != Some(RetryHint::Draining)))
    {
        // Preserve the established typed catch/status while keeping outer body context.
        let mut routed = DynamoError::builder()
            .class(ErrorClass::WorkerUnavailable)
            .diagnostic(err.to_string())
            .cause(error.clone());
        if draining {
            routed = routed.retry_hint(RetryHint::Draining);
        }
        attach_error(&routed.build(), exception, true)
    } else {
        attach_error(error, exception, true)
    }
}

/// Keep dispatch provenance without assigning a remote worker's identity to its relay.
pub(crate) fn dispatch_error_to_pyerr(err: anyhow::Error) -> PyErr {
    let exception = crate::to_pyerr(&err);
    if let Some(error) = find_dynamo_error(err.as_ref()) {
        return attach_error(error, exception, false);
    }
    // Untyped dispatch Display text is local diagnostic context, not received content.
    let error = DynamoError::builder()
        .upstream_error(UpstreamError::new("").mark_incomplete_metadata())
        .build();
    attach_error(&error, exception, false)
}

/// Record the exception's module and qualified name for frontend error rules.
fn qualified_type(value: &Bound<'_, PyAny>) -> String {
    let ty = value.get_type();
    let module = ty
        .getattr("__module__")
        .and_then(|value| value.extract::<String>())
        .unwrap_or_default();
    let name = ty
        .getattr("__qualname__")
        .and_then(|value| value.extract::<String>())
        .unwrap_or_else(|_| "Exception".to_string());
    format!("{module}.{name}")
}

/// Identify known media-fetch families from exception ancestry and typed input metadata.
fn is_media_fetch_error(value: &Bound<'_, PyAny>) -> bool {
    let Ok(mro) = value.get_type().getattr("__mro__") else {
        return false;
    };
    let Ok(bases) = mro.cast::<PyTuple>() else {
        return false;
    };
    let mut vllm_unprocessable = false;
    for base in bases.iter() {
        let module = base
            .getattr("__module__")
            .and_then(|value| value.extract::<String>())
            .unwrap_or_default();
        let name = base
            .getattr("__name__")
            .and_then(|value| value.extract::<String>())
            .unwrap_or_default();
        if (module == "dynamo.common.http.base" && name == "HttpError")
            || (module == "dynamo.common.http.url_validator"
                && matches!(name.as_str(), "UrlValidationError" | "MediaSizeError"))
            || (module == "requests.exceptions" && name == "RequestException")
            || (module == "aiohttp.client_exceptions" && name == "ClientError")
        {
            return true;
        }
        vllm_unprocessable |= module == "vllm.exceptions" && name == "VLLMUnprocessableEntityError";
    }
    if !vllm_unprocessable {
        return false;
    }
    let parameter = value
        .getattr("parameter")
        .and_then(|value| value.extract::<String>())
        .unwrap_or_default();
    if !matches!(parameter.as_str(), "image_url" | "video_url" | "audio_url") {
        return false;
    }
    // The typed parameter identifies media even for malformed URLs, paths or
    // inline data. Parsing a URL here would let those private values escape.
    value
        .getattr("value")
        .is_ok_and(|value| value.is_instance_of::<PyString>())
}

/// Recognize built-in programming-error families across Python subclasses.
fn is_python_programming_error(value: &Bound<'_, PyAny>) -> bool {
    value.is_instance_of::<pyo3::exceptions::PyTypeError>()
        || value.is_instance_of::<pyo3::exceptions::PyAttributeError>()
        || value.is_instance_of::<pyo3::exceptions::PyKeyError>()
}

/// Recognize Python and Dynamo transport failures before wrappers obscure their types.
fn is_python_transport_error(value: &Bound<'_, PyAny>) -> bool {
    value.is_instance_of::<pyo3::exceptions::PyConnectionError>()
        || value.is_instance_of::<pyo3::exceptions::PyTimeoutError>()
        || value.is_instance_of::<CannotConnect>()
        || value.is_instance_of::<Disconnected>()
        || value.is_instance_of::<ConnectionTimeout>()
        || value.is_instance_of::<StreamIncomplete>()
}

/// Bound the displayed exception chain used to select content, independently of privacy scans.
fn visible_python_causes(value: &Bound<'_, PyAny>) -> Result<([usize; 16], usize, bool), ()> {
    let mut current = value.clone();
    let mut nodes = [0_usize; 16];
    let mut programming = false;
    for index in 0..nodes.len() {
        let pointer = current.as_ptr() as usize;
        if nodes[..index].contains(&pointer) {
            return Err(());
        }
        nodes[index] = pointer;
        programming |= is_python_programming_error(&current);
        if let Some(cause) = current
            .getattr("__cause__")
            .ok()
            .filter(|value| !value.is_none())
        {
            current = cause;
            continue;
        }
        if current
            .getattr("__suppress_context__")
            .and_then(|value| value.extract::<bool>())
            .unwrap_or(false)
        {
            return Ok((nodes, index + 1, programming));
        }
        let Some(context) = current
            .getattr("__context__")
            .ok()
            .filter(|value| !value.is_none())
        else {
            return Ok((nodes, index + 1, programming));
        };
        current = context;
    }
    Err(())
}

/// Collected source content and privacy facts, without remote worker health ownership.
#[derive(Default)]
struct PythonSource {
    family: Option<&'static str>,
    retained_error: bool,
    public_copy: bool,
    public: Option<PublicDetails>,
    upstream: Option<UpstreamError>,
}

/// Scan both cause links for privacy while selecting relay content only from visible or copied errors.
fn inspect_python_causes(
    value: &Bound<'_, PyAny>,
    public_candidate: Option<&str>,
    received_candidate: &str,
) -> Result<PythonSource, ()> {
    const MAX_NODES: usize = 16;
    let (visible_nodes, visible_len, visible_programming) = visible_python_causes(value)?;
    let mut source = PythonSource::default();
    let mut visited = [0_usize; MAX_NODES];
    let mut active = [0_usize; MAX_NODES];
    let mut visited_len = 0;
    let mut active_len = 0;
    let mut pending = Vec::with_capacity(3 * MAX_NODES);
    pending.push((value.clone(), false));
    let candidate = public_candidate.unwrap_or(received_candidate);
    let mut programming_copy = false;
    // Privacy follows both links, including suppressed context. Shared cause/
    // context nodes are a DAG; only active ancestors are cycles. The worklist
    // cannot grow beyond the fixed node/depth budget below.
    while let Some((current, exiting)) = pending.pop() {
        let pointer = current.as_ptr() as usize;
        if exiting {
            active_len -= 1;
            continue;
        }
        if active[..active_len].contains(&pointer) {
            return Err(());
        }
        if visited[..visited_len].contains(&pointer) {
            continue;
        }
        if visited_len == MAX_NODES {
            return Err(());
        }
        visited[visited_len] = pointer;
        visited_len += 1;
        active[active_len] = pointer;
        active_len += 1;
        if is_python_programming_error(&current) {
            let text = current.str().map_err(|_| ())?;
            let text = text.to_string_lossy();
            programming_copy |= candidate == text.as_ref()
                || (text.len() > dynamo_runtime::error::Diagnostic::MAX_BYTES
                    && candidate == UpstreamError::new(text.as_ref()).message());
        }
        for attribute in [ROUTED_ENGINE_ERROR_ATTR, RELAY_CONTENT_ATTR] {
            let Ok(serialized) = current.getattr(attribute) else {
                continue;
            };
            let retained_text = current.str().map_err(|_| ())?;
            let retained_text = retained_text.to_string_lossy();
            let relay_copy = !retained_text.is_empty()
                && (candidate.contains(retained_text.as_ref())
                    || candidate == UpstreamError::new(retained_text.as_ref()).message());
            source.public_copy |= relay_copy;
            let selected = visible_nodes[..visible_len].contains(&pointer) || relay_copy;
            let remote: DynamoError = serialized
                .extract::<String>()
                .ok()
                .and_then(|value| serde_json::from_str(&value).ok())
                .ok_or(())?;
            let (public, upstream) = relay_content(&remote);
            // Hidden branches still supply privacy facts, but an unrelated
            // retained error must not replace the selected message/status.
            source.retained_error |= selected;
            if let Some(public) = public.filter(|_| selected)
                && (source.public.is_none()
                    || (!source
                        .public
                        .as_ref()
                        .is_some_and(renderable_public_details)
                        && renderable_public_details(&public)))
            {
                source.public = Some(public);
            }
            if let Some(upstream) = upstream {
                if !upstream.metadata_valid() {
                    return Err(());
                }
                match upstream.family_name() {
                    Some("media_fetch") => source.family = Some("media_fetch"),
                    Some("python_programming") if source.family != Some("media_fetch") => {
                        source.family = Some("python_programming");
                    }
                    Some("transport") if source.family.is_none() => {
                        source.family = Some("transport")
                    }
                    _ => {}
                }
                if selected {
                    let http_status = source
                        .upstream
                        .as_ref()
                        .and_then(UpstreamError::http_status_value)
                        .or(upstream.http_status_value());
                    let mut merged = source.upstream.take().unwrap_or(upstream);
                    if let Some(status) = http_status {
                        merged = merged.http_status(status);
                    }
                    source.upstream = Some(merged);
                }
            }
        }
        if is_media_fetch_error(&current) {
            source.family = Some("media_fetch");
        } else if source.family.is_none() && is_python_transport_error(&current) {
            source.family = Some("transport");
        }
        if pending.len() + 3 > 3 * MAX_NODES {
            return Err(());
        }
        pending.push((current.clone(), true));
        // Stack order keeps explicit causes ahead of implicit context.
        for attribute in ["__context__", "__cause__"] {
            if let Some(next) = current
                .getattr(attribute)
                .ok()
                .filter(|value| !value.is_none())
            {
                pending.push((next, false));
            }
        }
    }
    // A distinct typed validation message owns its explanation. Direct bugs
    // and exact copies remain private; hidden media/transport/relay facts above
    // are independent of Python's displayed exception chain.
    let authored_validation =
        value.is_instance_of::<InvalidArgument>() && public_candidate.is_some();
    if is_python_programming_error(value)
        || programming_copy
        || (source.family.is_none() && !authored_validation && visible_programming)
    {
        source.family = Some("python_programming");
    }
    Ok(source)
}

/// Recognize legacy transport reasons used when explicit HTTP-response provenance is absent.
fn is_transport_reason(reason: &str) -> bool {
    matches!(
        reason,
        "transport.cannot_connect"
            | "transport.disconnected"
            | "transport.connection_timeout"
            | "backend.cannot_connect"
            | "backend.disconnected"
            | "backend.connection_timeout"
            | "backend.stream_incomplete"
    )
}

/// Prefer public details that existing protocol renderers can turn into useful wording.
fn renderable_public_details(details: &PublicDetails) -> bool {
    match details {
        PublicDetails::Message { .. } => true,
        PublicDetails::ContextLength { limit, actual, .. } => {
            actual.is_some_and(|actual| actual >= *limit)
        }
        PublicDetails::SizeLimit { .. } | PublicDetails::RateLimit { .. } => false,
    }
}

/// Project bounded source content and privacy facts without copying remote health identity.
fn relay_content(error: &DynamoError) -> (Option<PublicDetails>, Option<UpstreamError>) {
    let mut current: Option<&(dyn std::error::Error + 'static)> = Some(error);
    let mut public = None;
    let mut upstream = None;
    let mut http_status = None;
    let mut family = None;
    let mut incomplete = false;
    for _ in 0..16 {
        let Some(node) = current else {
            if let Some(status) = http_status {
                upstream = Some(
                    upstream
                        .unwrap_or_else(|| UpstreamError::new(""))
                        .http_status(status),
                );
            }
            if let Some(family) = family {
                upstream = Some(
                    upstream
                        .unwrap_or_else(|| UpstreamError::new(""))
                        .family(family),
                );
            }
            if incomplete {
                upstream = Some(
                    upstream
                        .unwrap_or_else(|| UpstreamError::new(""))
                        .mark_incomplete_metadata(),
                );
            }
            return (public, upstream);
        };
        if let Some(error) = node.downcast_ref::<DynamoError>() {
            if error.reason().as_str() == "runtime.invalid_error" {
                return (
                    None,
                    Some(UpstreamError::new("").mark_incomplete_metadata()),
                );
            }
            if let Some(details) = error.public_details()
                && (public.is_none()
                    || (!public.as_ref().is_some_and(renderable_public_details)
                        && renderable_public_details(details)))
            {
                public = Some(details.clone());
            }
            if let Some(payload) = error.upstream_error() {
                http_status = http_status.or(payload.http_status_value());
                upstream = upstream.or_else(|| Some(payload.clone()));
                incomplete |= !payload.metadata_valid();
                match payload.family_name() {
                    Some("media_fetch") => family = Some("media_fetch"),
                    Some("python_programming") if family != Some("media_fetch") => {
                        family = Some("python_programming");
                    }
                    Some("transport") if family.is_none() => family = Some("transport"),
                    _ => {}
                }
            }
            if family.is_none()
                && is_transport_reason(error.reason().as_str())
                && error
                    .upstream_error()
                    .and_then(UpstreamError::http_status_value)
                    .is_none()
            {
                family = Some("transport");
            }
        }
        current = node.source();
    }
    (
        None,
        Some(UpstreamError::new("").mark_incomplete_metadata()),
    )
}

/// Retain received Python content and provenance while preserving the caller's local error identity.
pub(crate) fn preserve_python_error_content(
    py: Python<'_>,
    exception: &PyErr,
    mut error: DynamoError,
) -> DynamoError {
    let value = exception.value(py).as_any();
    let message = value
        .getattr("message")
        .and_then(|value| value.extract::<String>())
        .or_else(|_| {
            value
                .str()
                .map(|value| value.to_string_lossy().into_owned())
        })
        .unwrap_or_else(|_| "Python exception message unavailable".to_string());
    let captured = UpstreamError::new(message).error_type(qualified_type(value));
    let http_status = value
        .getattr("status_code")
        .ok()
        .and_then(|value| value.extract::<u16>().ok())
        .filter(|status| (400..=599).contains(status))
        .or_else(|| {
            error
                .upstream_error()
                .and_then(UpstreamError::http_status_value)
        });
    let mut upstream =
        match inspect_python_causes(value, error.public_message(), captured.message()) {
            Ok(source) if source.retained_error => {
                // The Python wrapper may have copied a private Display diagnostic
                // into its text and hence the automatic InvalidArgument public field.
                // Only retained content is eligible; absence on an older producer
                // must stay absent instead of turning that diagnostic into upstream.
                let legacy_copy =
                    source.public_copy && source.public.is_none() && source.upstream.is_none();
                if source.public_copy || error.public.is_none() {
                    error.public = source.public.map(Box::new);
                }
                let mut upstream = source.upstream;
                if legacy_copy {
                    upstream = Some(UpstreamError::new("").mark_incomplete_metadata());
                }
                if let Some(family) = source.family {
                    upstream = Some(
                        upstream
                            .unwrap_or_else(|| UpstreamError::new(""))
                            .family(family),
                    );
                }
                error.upstream_error = upstream.map(Box::new);
                return error;
            }
            Ok(source) => match source.family {
                Some(family) => captured.family(family),
                None => captured,
            },
            Err(()) => {
                error.public = None;
                UpstreamError::new("").mark_incomplete_metadata()
            }
        };
    if upstream.family_name().is_none()
        && error
            .upstream_error()
            .is_some_and(|previous| previous.family_name() == Some("http"))
    {
        upstream = upstream.family("http");
    }
    if let Some(status) = http_status {
        upstream = upstream.http_status(status);
    }
    if let Some(code) = ["code", "status", "status_code"].iter().find_map(|name| {
        value.getattr(*name).ok().and_then(|value| {
            value
                .extract::<u16>()
                .map(|value| value.to_string())
                .or_else(|_| value.extract::<String>())
                .ok()
        })
    }) {
        upstream = upstream.code(code);
    }
    error.upstream_error = Some(Box::new(upstream));
    error
}

fn is_worker_draining(error: &(dyn std::error::Error + 'static)) -> bool {
    let mut current = Some(error);
    while let Some(cause) = current {
        if cause
            .downcast_ref::<DynamoError>()
            .is_some_and(|error| error.reason().as_str() == "runtime.invalid_error")
        {
            return false;
        }
        if cause
            .downcast_ref::<DynamoError>()
            .is_some_and(|error| error.retry_hint() == Some(RetryHint::Draining))
        {
            return true;
        }
        current = cause.source();
    }
    false
}

fn find_dynamo_error<'a>(error: &'a (dyn std::error::Error + 'static)) -> Option<&'a DynamoError> {
    let mut current = Some(error);
    while let Some(cause) = current {
        if let Some(error) = cause.downcast_ref::<DynamoError>() {
            return Some(error);
        }
        current = cause.source();
    }
    None
}

fn is_worker_unavailable(error: &(dyn std::error::Error + 'static)) -> bool {
    let mut current = Some(error);
    while let Some(cause) = current {
        if cause
            .downcast_ref::<DynamoError>()
            .is_some_and(|error| error.reason().as_str() == "runtime.invalid_error")
        {
            return false;
        }
        // `class()` normalizes `WorkerUnavailable` to `Unavailable`; the raw
        // error type keeps the migratable worker-scoped distinction.
        if cause
            .downcast_ref::<DynamoError>()
            .is_some_and(|error| error.error_type() == ErrorClass::WorkerUnavailable)
        {
            return true;
        }
        current = cause.source();
    }
    false
}

pub(crate) fn worker_unavailable_to_dynamo(py: Python<'_>, err: &PyErr) -> Option<DynamoError> {
    if !err.is_instance_of::<WorkerUnavailable>(py) {
        return None;
    }
    let message = err
        .value(py)
        .str()
        .map(|s| s.to_string_lossy().into_owned())
        .unwrap_or_default();
    let mut error = DynamoError::builder()
        .class(ErrorClass::WorkerUnavailable)
        .message(message);
    if err.is_instance_of::<WorkerDraining>(py) {
        error = error.retry_hint(RetryHint::Draining);
    }
    Some(error.build())
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
            m.add("WorkerUnavailable", m.py().get_type::<WorkerUnavailable>())?;
            m.add("WorkerDraining", m.py().get_type::<WorkerDraining>())?;
            m.add("RoutedEngineError", m.py().get_type::<RoutedEngineError>())?;
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

/// Generic label for an InvalidRequest that carries no explicit
/// public message.
const GENERIC_INVALID_REQUEST_MESSAGE: &str = "Invalid request";

fn invalid_request_client_message(error: &DynamoError) -> Option<String> {
    (error.class() == ErrorClass::InvalidRequest).then(|| {
        error
            .public_message()
            .unwrap_or(GENERIC_INVALID_REQUEST_MESSAGE)
            .to_owned()
    })
}

/// Preserve validation errors across Rust-to-Python streams: a classified
/// InvalidRequest raises a typed `InvalidArgument`, carrying the producer's
/// public message or the generic label when there is none; diagnostic text must
/// never be promoted into a public `InvalidArgument` message. Any other error
/// raises a `ValueError`, which on a `RoutedEngine`'s stream (`routed_engine`)
/// is a [`RoutedEngineError`] that keeps the backend's `DynamoError`.
pub(crate) fn check_response_error<R>(
    response: Annotated<R>,
    routed_engine: bool,
) -> PyResult<Annotated<R>> {
    if !response.is_error() {
        return Ok(response);
    }
    let error = response
        .into_data()
        .err()
        .unwrap_or_else(|| DynamoError::msg("unknown error"));
    let exception = if let Some(message) = invalid_request_client_message(&error) {
        InvalidArgument::new_err(message)
    } else if routed_engine {
        return Err(routed_engine_error(&error, error.to_string()));
    } else {
        pyo3::exceptions::PyValueError::new_err(error.to_string())
    };
    Err(attach_error(&error, exception, routed_engine))
}

fn public_invalid_argument_message(err: &(dyn std::error::Error + 'static)) -> Option<String> {
    find_dynamo_error(err).and_then(invalid_request_client_message)
}

/// Test hook: raise what a response stream raises for the error event a backend
/// `HttpError(code, message)` produces: a `RoutedEngine`'s stream by default,
/// any other with `routed_engine=False`.
#[pyfunction]
#[pyo3(signature = (code, message, routed_engine=true))]
pub fn _raise_routed_http_error(code: u16, message: String, routed_engine: bool) -> PyResult<()> {
    let mut response = Annotated::<()>::from_error(String::new());
    response.error = Some(build_http_like_error(code, message));
    check_response_error(response, routed_engine).map(|_| ())
}

/// Test hook: send a bounded failure through the endpoint dispatch converter.
#[pyfunction]
#[pyo3(signature = (message, typed=true, received=true))]
pub fn _raise_dispatch_error(message: String, typed: bool, received: bool) -> PyResult<()> {
    let upstream = UpstreamError::new(message);
    let message = upstream.message().to_owned();
    let error = if typed {
        let mut builder = DynamoError::builder()
            .error_type(dynamo_runtime::error::ErrorType::CannotConnect)
            .diagnostic(message);
        if received {
            builder = builder.upstream_error(upstream);
        }
        anyhow::Error::new(builder.build())
    } else {
        anyhow::anyhow!(message)
    };
    Err(dispatch_error_to_pyerr(error))
}

/// Read `(code, message)` off a Python exception carrying an HTTP-style
/// status. Accepts `.code` (matches [`HttpError`] in `http.rs`) or `.status`
/// (matches `dynamo.common.http.HttpStatusError`), plus a string `.message`.
///
/// Message text and source provenance are captured separately from diagnostics.
/// The frontend blocklist owns disclosure, including remote-fetch failures that
/// can contain URLs or internal connection details. Legacy public-message
/// projections are retained separately for older frontend readers.
pub fn extract_http_like_error(py: Python<'_>, err: &PyErr) -> Option<(u16, String)> {
    let value = err.value(py);
    // Keep main's classification precedence; status_code is captured separately.
    let code = ["code", "status"].iter().find_map(|name| {
        value
            .getattr(*name)
            .ok()
            .and_then(|value| value.extract::<u16>().ok())
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
    Some(preserve_python_error_content(
        py,
        err,
        build_http_like_error(code, message),
    ))
}

/// Legacy public-message projections for N-2 frontend compatibility. New
/// frontends apply the blocklist to the complete captured error instead.
/// Python producers must use the same text: the Python test
/// `test_video_public_message_matches_binding_allowlist` fails if they diverge.
const PUBLIC_HTTP_MESSAGES: &[&str] = dynamo_llm::preprocessor::media::PUBLIC_MEDIA_MESSAGES;

/// Capture HTTP-shaped content without deciding the enclosing adapter's status ownership.
pub(crate) fn build_http_like_error(code: u16, message: String) -> DynamoError {
    let legacy_message = serde_json::json!({
        "message": &message,
        "code": code,
    })
    .to_string();

    let mut upstream = UpstreamError::new(&message)
        .code(code.to_string())
        .family("http");
    // Only this established HTTP-shaped conversion asserts a source HTTP
    // status. Generic received numeric codes retain their own namespace.
    if (400..=599).contains(&code) {
        upstream = upstream.http_status(code);
    }
    let mut builder = DynamoError::builder()
        .class(error_class_for_http_status(code))
        .diagnostic(legacy_message)
        .upstream_error(upstream);
    if PUBLIC_HTTP_MESSAGES.contains(&message.as_str()) {
        builder = builder.public_message(message);
    }
    builder.build()
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde::Deserialize;

    #[test]
    fn relay_content_keeps_public_details_separate_from_received_facts() {
        for class in [ErrorClass::InvalidRequest, ErrorClass::WorkerUnavailable] {
            for public in [
                PublicDetails::Message {
                    message: "Authored relay replacement".into(),
                },
                PublicDetails::ContextLength {
                    limit: 8,
                    actual: Some(10),
                    prompt_tokens: Some(10),
                },
            ] {
                let upstream = UpstreamError::new("received relay explanation")
                    .http_status(422)
                    .code("7");
                let source = DynamoError::builder()
                    .class(class)
                    .retry_hint(RetryHint::Draining)
                    .public_details(public.clone())
                    .upstream_error(upstream.clone())
                    .diagnostic("PRIVATE_RELAY_WRAPPER")
                    .build();
                let (projected_public, projected_upstream) = relay_content(&source);
                assert_eq!(projected_public.as_ref(), Some(&public));
                assert_eq!(projected_upstream.as_ref(), Some(&upstream));
            }
        }

        let context = PublicDetails::ContextLength {
            limit: 8,
            actual: Some(10),
            prompt_tokens: Some(10),
        };
        let source = DynamoError::builder()
            .public_details(context.clone())
            .build();
        let outer = DynamoError::builder()
            .public_details(PublicDetails::SizeLimit {
                limit: 8,
                actual: Some(10),
            })
            .cause(source.clone())
            .build();
        assert_eq!(relay_content(&outer).0, Some(context));
        let authored = PublicDetails::Message {
            message: "Authored outer replacement".into(),
        };
        let outer = DynamoError::builder()
            .public_details(authored.clone())
            .cause(source)
            .build();
        assert_eq!(relay_content(&outer).0, Some(authored));
    }

    #[test]
    fn relay_content_projects_nested_provenance_and_stops_at_invalid_identity() {
        let inner = DynamoError::builder()
            .error_type(dynamo_runtime::error::ErrorType::Backend(
                BackendError::Disconnected,
            ))
            .upstream_error(UpstreamError::new("remote explanation"))
            .build();
        let outer = DynamoError::builder()
            .class(ErrorClass::Internal)
            .diagnostic("PRIVATE_RELAY_WRAPPER")
            .cause(inner.clone())
            .build();
        for source in [&inner, &outer] {
            let (public, upstream) = relay_content(source);
            assert!(public.is_none());
            let upstream = upstream.unwrap();
            assert_eq!(upstream.message(), "remote explanation");
            assert_eq!(upstream.family_name(), Some("transport"));
        }

        let invalid: DynamoError = serde_json::from_value(serde_json::json!({
            "class": "InvalidRequest", "reason": "backend.unknown", "caused_by": inner,
        }))
        .unwrap();
        let (public, upstream) = relay_content(&invalid);
        assert!(public.is_none());
        let upstream = upstream.unwrap();
        assert!(!upstream.metadata_valid());
        assert!(upstream.message().is_empty());
    }

    #[test]
    fn relay_content_merges_http_facts_without_changing_transport_provenance() {
        let remote = build_http_like_error(503, "received HTTP explanation".into());
        let outer = DynamoError::builder()
            .class(ErrorClass::Internal)
            .upstream_error(UpstreamError::new("received outer explanation"))
            .cause(remote)
            .build();
        let upstream = relay_content(&outer).1.unwrap();
        assert_eq!(upstream.message(), "received outer explanation");
        assert_eq!(upstream.http_status_value(), Some(503));

        for (http_response, explicit_transport) in [(true, false), (true, true), (false, false)] {
            let mut upstream = UpstreamError::new("received HTTP explanation");
            if http_response {
                upstream = upstream.http_status(503);
            }
            if explicit_transport {
                upstream = upstream.family("transport");
            }
            let remote = DynamoError::builder()
                .error_type(dynamo_runtime::error::ErrorType::Backend(
                    BackendError::CannotConnect,
                ))
                .upstream_error(upstream)
                .build();
            let upstream = relay_content(&remote).1.unwrap();
            assert_eq!(upstream.http_status_value(), http_response.then_some(503));
            assert_eq!(
                upstream.family_name(),
                (explicit_transport || !http_response).then_some("transport")
            );
        }
    }

    #[test]
    fn worker_lifecycle_helpers_reject_invalid_outer_identity() {
        let worker = DynamoError::builder()
            .class(ErrorClass::WorkerUnavailable)
            .retry_hint(RetryHint::Draining)
            .build();
        for class in [ErrorClass::Internal, ErrorClass::WorkerUnavailable] {
            let outer = DynamoError::builder()
                .class(class)
                .cause(worker.clone())
                .build();
            assert!(is_worker_unavailable(&outer));
            assert!(is_worker_draining(&outer));
        }
        let invalid: DynamoError = serde_json::from_value(serde_json::json!({
            "class": "Internal", "reason": "request.invalid", "caused_by": worker,
        }))
        .unwrap();
        assert!(!is_worker_unavailable(&invalid));
        assert!(!is_worker_draining(&invalid));
    }

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
    fn worker_unavailable_survives_class_normalization_and_context() {
        let rejection = DynamoError::builder()
            .class(ErrorClass::WorkerUnavailable)
            .message("prefill routing is still activating")
            .build();
        // `class()` reports the normalized `Unavailable`; detection must not rely on it.
        assert_eq!(rejection.class(), ErrorClass::Unavailable);
        assert!(is_worker_unavailable(&rejection));

        let wrapped = anyhow::Error::from(rejection).context("routed pipeline failed");
        assert!(is_worker_unavailable(wrapped.as_ref()));

        let pool_unavailable = DynamoError::builder()
            .class(ErrorClass::Unavailable)
            .message("no workers")
            .build();
        assert!(!is_worker_unavailable(&pool_unavailable));
    }

    #[test]
    fn worker_draining_remains_distinct_through_error_context() {
        let rejection = DynamoError::builder()
            .class(ErrorClass::WorkerUnavailable)
            .retry_hint(RetryHint::Draining)
            .message("worker closed admission")
            .build();
        assert_eq!(rejection.class(), ErrorClass::Unavailable);
        assert!(is_worker_draining(&rejection));
        assert!(is_worker_draining(
            anyhow::Error::from(rejection)
                .context("worker RPC")
                .as_ref()
        ));
        let generic = DynamoError::builder()
            .class(ErrorClass::WorkerUnavailable)
            .build();
        assert!(!is_worker_draining(&generic));
    }

    #[test]
    fn invalid_request_uses_public_message_or_generic_fallback() {
        let error = DynamoError::builder()
            .class(ErrorClass::InvalidRequest)
            .diagnostic("private backend detail")
            .public_message("Invalid JSON schema")
            .build();
        assert_eq!(
            invalid_request_client_message(&error),
            Some("Invalid JSON schema".to_string())
        );

        let error = DynamoError::builder()
            .class(ErrorClass::InvalidRequest)
            .diagnostic("private validation detail")
            .build();
        assert_eq!(
            invalid_request_client_message(&error),
            Some(GENERIC_INVALID_REQUEST_MESSAGE.to_string())
        );
    }

    #[derive(Debug)]
    struct RouterWrapped(DynamoError);

    impl std::fmt::Display for RouterWrapped {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            write!(f, "Prefill execution failed: {}", self.0)
        }
    }

    impl std::error::Error for RouterWrapped {
        fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
            Some(&self.0)
        }
    }

    #[test]
    fn error_chain_yields_public_invalid_argument_message() {
        let worker_error = DynamoError::builder()
            .class(ErrorClass::InvalidRequest)
            .diagnostic("private backend detail")
            .public_message("use --enable-multimodal")
            .build();
        let wrapped = anyhow::Error::new(RouterWrapped(worker_error));
        assert_eq!(
            public_invalid_argument_message(wrapped.as_ref()),
            Some("use --enable-multimodal".to_string())
        );
    }

    #[test]
    fn legacy_worker_rejection_without_public_message_yields_generic_message() {
        let wire = serde_json::json!({
            "error_type": {"Backend": "InvalidArgument"},
            "message": "private backend detail",
        });
        let error: DynamoError = serde_json::from_value(wire).expect("legacy wire error");
        assert_eq!(error.class(), ErrorClass::InvalidRequest);
        assert_eq!(error.public_message(), None);

        let wrapped = anyhow::Error::new(RouterWrapped(error));
        assert_eq!(
            public_invalid_argument_message(wrapped.as_ref()),
            Some(GENERIC_INVALID_REQUEST_MESSAGE.to_string())
        );
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
            assert_eq!(invalid_request_client_message(&error), None);
        }
    }

    #[test]
    fn http_like_errors_support_semantic_and_legacy_readers() {
        for (code, expected_class) in [
            (7, ErrorClass::Internal),
            (408, ErrorClass::DeadlineExceeded),
            (409, ErrorClass::Conflict),
            (415, ErrorClass::UnsupportedMedia),
            (418, ErrorClass::InvalidRequest),
            (422, ErrorClass::InvalidRequest),
            (431, ErrorClass::PayloadTooLarge),
            (503, ErrorClass::Unavailable),
            (529, ErrorClass::CapacityExhausted),
        ] {
            let error = build_http_like_error(code, "private backend detail".to_string());
            assert_eq!(error.class(), expected_class);
            assert_eq!(error.public_message(), None);

            let wire = serde_json::to_value(&error).expect("serialize DynamoError");
            let semantic: DynamoError =
                serde_json::from_value(wire.clone()).expect("semantic reader");
            assert_eq!(semantic.class(), expected_class);
            assert_eq!(semantic.public_message(), None);
            let upstream = semantic.upstream_error().unwrap();
            assert_eq!(upstream.message(), "private backend detail");
            assert_eq!(upstream.code_value(), Some(code.to_string().as_str()));
            assert_eq!(
                upstream.http_status_value(),
                (400..=599).contains(&code).then_some(code)
            );

            let legacy: LegacyDynamoError = serde_json::from_value(wire).expect("legacy reader");
            let legacy_http: LegacyHttpError =
                serde_json::from_str(&legacy.message).expect("legacy HTTP payload");
            assert_eq!(legacy_http.code, code);
            assert_eq!(legacy_http.message, "private backend detail");
        }
    }

    #[test]
    fn allowlisted_http_message_is_public_and_a_prefix_is_not() {
        let error = build_http_like_error(415, "Unsupported video media".to_string());
        assert_eq!(error.class(), ErrorClass::UnsupportedMedia);
        assert_eq!(error.public_message(), Some("Unsupported video media"));

        let prefixed = build_http_like_error(
            415,
            "Unsupported video media https://example.com/secret".to_string(),
        );
        assert_eq!(prefixed.class(), ErrorClass::UnsupportedMedia);
        assert_eq!(prefixed.public_message(), None);
    }

    #[test]
    fn image_pixel_limit_message_requires_exact_text() {
        let message = "Image exceeds the decoded pixel limit";
        for status in [400, 413] {
            assert_eq!(
                build_http_like_error(status, message.into()).public_message(),
                Some(message)
            );
        }
        for text in [
            format!("{message} https://example.com/?secret=private"),
            "private decoder detail".to_string(),
        ] {
            assert_eq!(build_http_like_error(413, text).public_message(), None);
        }
    }
}
