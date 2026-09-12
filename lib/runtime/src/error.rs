// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Dynamo's shared semantic error contract.
//!
//! A [`DynamoError`] describes **what failed**. It deliberately does not prescribe an HTTP status, retry action, stream action, or worker-health decision. Those decisions belong to the consumer that owns the relevant local state, such as response commitment, retry budget, replayability, deadline, and worker observations.
//!
//! The component closest to the cause classifies a failure once. Callers should propagate an existing [`DynamoError`] directly; internal transports preserve its semantic identity, and protocol boundaries render it for their clients. Generic conversion from an error chain recovers a typed non-internal semantic source, but it intentionally treats an unclassified or internal source as `Internal/runtime.unclassified`. See [DEP #14354](https://github.com/ai-dynamo/dynamo/issues/14354) for the architecture and rollout rationale.
//!
//! # Error identity
//!
//! Every error has two identity fields:
//!
//! - [`ErrorClass`] is the coarse, closed category used for exhaustive consumer policy. New producers should select a canonical class such as [`ErrorClass::InvalidRequest`] or [`ErrorClass::Unavailable`]. The older transport-specific variants are retained for compatibility and normalize through [`DynamoError::class`].
//! - [`ErrorReason`] is a stable, bounded catalog key that identifies the specific cause. A reason belongs to exactly one normalized class. It is safe for low-cardinality policy and metric dimensions; arbitrary exception text and user input are not.
//!
//! The builder and deserializer validate the class/reason pair. An unknown, malformed, or mismatched identity fails closed to `Internal/runtime.invalid_error`, and its public details are discarded. Consumers should therefore use [`DynamoError::class`], [`DynamoError::reason`], and [`DynamoError::public_details`] instead of reading the public fields directly. [`DynamoError::error_type`] exists for legacy policy only.
//!
//! ## Choosing a canonical class
//!
//! | Class | Use when |
//! |---|---|
//! | [`ErrorClass::InvalidRequest`] | The caller supplied malformed input or failed request-level validation. |
//! | [`ErrorClass::Unauthenticated`] | Authentication credentials are missing or invalid. |
//! | [`ErrorClass::PermissionDenied`] | The authenticated caller is not allowed to perform the operation. |
//! | [`ErrorClass::NotFound`] | A caller-visible resource does not exist. |
//! | [`ErrorClass::Conflict`] | The request conflicts with current resource state. |
//! | [`ErrorClass::PayloadTooLarge`] | The request exceeds an advertised size limit. |
//! | [`ErrorClass::UnsupportedMedia`] | The request uses an unsupported media type. |
//! | [`ErrorClass::RateLimited`] | A caller-specific admission or rate limit was exceeded. |
//! | [`ErrorClass::CapacityExhausted`] | Dynamo, a selected worker, or the eligible worker pool lacks capacity; the reason distinguishes worker overload from pool exhaustion. |
//! | [`ErrorClass::Cancelled`] | The request was cancelled, usually because the client disconnected. |
//! | [`ErrorClass::Unavailable`] | A service, worker, pool, or Dynamo-owned dependency is unavailable. |
//! | [`ErrorClass::BackendProtocol`] | A backend response violates the expected protocol. |
//! | [`ErrorClass::DeadlineExceeded`] | A request or attempt deadline expired. |
//! | [`ErrorClass::NotImplemented`] | A valid requested capability is unsupported. |
//! | [`ErrorClass::Internal`] | A defect, invalid classification, or otherwise unclassified failure occurred. |
//!
//! Similar symptoms do not necessarily have the same class. For example, malformed client input is [`ErrorClass::InvalidRequest`], invalid operator configuration is [`ErrorClass::Internal`], caller-specific throttling is [`ErrorClass::RateLimited`], and pool-wide capacity pressure is [`ErrorClass::CapacityExhausted`].
//!
//! # Data visibility
//!
//! [`Diagnostic`] contains bounded operator-facing context. It may be used in logs and traces, but it must never be copied into a client response or metric label. Do not place secrets, credentials, prompts, generated text, or raw upstream response bodies in a diagnostic.
//!
//! [`PublicDetails`] is the only occurrence-specific channel available for client rendering, but the type does not sanitize its values. Producers must populate it only at a boundary that knows the value is safe. A [`PublicDetails::Message`] must contain fixed, cataloged, or explicitly allowlisted text; it must not contain user input, prompts, generated content, secrets, URLs, diagnostics, or raw upstream responses. Never derive public details from a diagnostic or arbitrary backend exception text.
//!
//! # Producer guidance
//!
//! 1. Classify at the boundary that understands the cause instead of relying on message parsing farther downstream.
//! 2. Select the canonical class for the failure's meaning, independent of the current transport.
//! 3. Use a registered reason that belongs to that class. Add new reasons to the catalog with class-consistency tests rather than emitting dynamic strings.
//! 4. Add a diagnostic only when it improves operator debugging, and add public details only when they are explicitly client-safe.
//! 5. Propagate an existing [`DynamoError`] directly whenever possible. If a generic wrapper is unavoidable, retain the typed error as its source and remember that generic conversion recovers only non-internal semantic classifications.
//!
//! [`DynamoError::msg`], conversion from an unclassified [`std::error::Error`], and conversion from a generic wrapper whose typed source is internal intentionally produce `Internal/runtime.unclassified`; use them as a fallback, not for failures whose meaning is known.
//!
//! ```rust,no_run
//! use dynamo_runtime::error::{DynamoError, ErrorClass, ErrorReason};
//!
//! let error = DynamoError::builder()
//!     .class(ErrorClass::InvalidRequest)
//!     .reason(ErrorReason::new("request.invalid").expect("registered reason"))
//!     .diagnostic("messages[2].role failed validation")
//!     .public_message("The message role is invalid")
//!     .build();
//!
//! assert_eq!(error.class(), ErrorClass::InvalidRequest);
//! assert_eq!(error.reason().as_str(), "request.invalid");
//! assert_eq!(error.public_message(), Some("The message role is invalid"));
//! ```
//!
//! Structured details are preferable when the protocol renderer needs machine-readable values:
//!
//! ```rust,no_run
//! use dynamo_runtime::error::{DynamoError, ErrorClass, ErrorReason, PublicDetails};
//!
//! let error = DynamoError::builder()
//!     .class(ErrorClass::PayloadTooLarge)
//!     .reason(ErrorReason::new("request.payload_too_large").expect("registered reason"))
//!     .public_details(PublicDetails::SizeLimit {
//!         limit: 1_048_576,
//!         actual: Some(1_250_000),
//!     })
//!     .build();
//! ```
//!
//! # Consumer guidance
//!
//! Consumers apply their own policy to the validated semantic identity:
//!
//! - Protocol renderers map [`DynamoError::class`] to a status or legal terminal stream event and expose only approved [`PublicDetails`].
//! - Retry and migration code owns its reason policy and combines it with locally owned budget, deadline, routing, replay, and continuation state.
//! - Worker-health code combines the reason with local worker and pool observations rather than inferring health from an HTTP status.
//! - Metrics use the normalized class and registered reason. Diagnostics, request identifiers, exception types, URLs, and user-controlled values must not become labels.
//!
//! A recovered internal attempt is not a final client failure. Terminal failure accounting belongs at the frontend rendering boundary so one propagated error is not counted by every subsystem it crosses.

use serde::{Deserialize, Serialize};
use std::fmt;
use std::sync::{Arc, LazyLock};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub enum ErrorClass {
    /// The request contains invalid input (e.g., prompt exceeds context length).
    InvalidArgument,
    /// Failed to establish a connection to a remote worker.
    CannotConnect,
    /// An established connection was lost unexpectedly.
    Disconnected,
    /// A connection or request timed out.
    ConnectionTimeout,
    /// The backend accepted the request but stopped responding (stream inactivity timeout).
    ResponseTimeout,
    /// The request was cancelled (e.g., client disconnected).
    Cancelled,
    /// The capacity constraint cannot be relieved by selecting another worker.
    /// This most commonly means the whole eligible worker pool is exhausted.
    ResourceExhausted,
    /// One selected worker is out of capacity while others may still have room.
    /// Distinct from [`Self::ResourceExhausted`] so a request whose routing
    /// constraints permit reassignment can migrate; both surface as HTTP 529.
    WorkerOverloaded,
    /// No backend worker is currently available to handle the request.
    Unavailable,
    /// One addressed worker answered that it does not serve this request's
    /// endpoint instance (stale discovery or a worker shutting down) while
    /// other workers may. Distinct from [`Self::Unavailable`] so the request
    /// can migrate; both surface as HTTP 503.
    WorkerUnavailable,
    /// Error originating from a backend engine.
    Backend(BackendError),
    /// The client request is malformed or fails request-level validation.
    InvalidRequest,
    /// Authentication credentials are missing or invalid.
    Unauthenticated,
    /// The authenticated caller is not permitted to perform the operation.
    PermissionDenied,
    /// The requested resource does not exist.
    NotFound,
    /// The request conflicts with current resource state.
    Conflict,
    /// The request body exceeds a configured size limit.
    PayloadTooLarge,
    /// The request uses an unsupported media type.
    UnsupportedMedia,
    /// The caller exceeded an admission or request-rate limit.
    RateLimited,
    /// The eligible worker pool has no capacity.
    CapacityExhausted,
    /// A backend response violates the expected protocol.
    BackendProtocol,
    /// The operation exceeded its deadline.
    DeadlineExceeded,
    /// The requested operation is not implemented.
    NotImplemented,
    /// An internal defect or unclassified failure occurred.
    Internal,
    /// Uncategorized or unknown error.
    Unknown,
}

#[derive(Deserialize)]
#[serde(untagged)]
#[allow(non_snake_case)]
enum ErrorClassWire {
    Named(String),
    Backend { Backend: BackendError },
}

impl ErrorClass {
    fn from_wire_name(name: &str) -> Self {
        match name {
            "InvalidArgument" => Self::InvalidArgument,
            "CannotConnect" => Self::CannotConnect,
            "Disconnected" => Self::Disconnected,
            "ConnectionTimeout" => Self::ConnectionTimeout,
            "ResponseTimeout" => Self::ResponseTimeout,
            "Cancelled" => Self::Cancelled,
            "ResourceExhausted" => Self::ResourceExhausted,
            "WorkerOverloaded" => Self::WorkerOverloaded,
            "Unavailable" => Self::Unavailable,
            "WorkerUnavailable" => Self::WorkerUnavailable,
            "InvalidRequest" => Self::InvalidRequest,
            "Unauthenticated" => Self::Unauthenticated,
            "PermissionDenied" => Self::PermissionDenied,
            "NotFound" => Self::NotFound,
            "Conflict" => Self::Conflict,
            "PayloadTooLarge" => Self::PayloadTooLarge,
            "UnsupportedMedia" => Self::UnsupportedMedia,
            "RateLimited" => Self::RateLimited,
            "CapacityExhausted" => Self::CapacityExhausted,
            "BackendProtocol" => Self::BackendProtocol,
            "DeadlineExceeded" => Self::DeadlineExceeded,
            "NotImplemented" => Self::NotImplemented,
            "Internal" => Self::Internal,
            _ => Self::Unknown,
        }
    }
}

impl<'de> Deserialize<'de> for ErrorClass {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        Ok(match ErrorClassWire::deserialize(deserializer)? {
            ErrorClassWire::Named(name) => Self::from_wire_name(&name),
            ErrorClassWire::Backend { Backend: error } => Self::Backend(error),
        })
    }
}

impl fmt::Display for ErrorClass {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ErrorClass::Unknown => write!(f, "Unknown"),
            ErrorClass::InvalidArgument => write!(f, "InvalidArgument"),
            ErrorClass::CannotConnect => write!(f, "CannotConnect"),
            ErrorClass::Disconnected => write!(f, "Disconnected"),
            ErrorClass::ConnectionTimeout => write!(f, "ConnectionTimeout"),
            ErrorClass::ResponseTimeout => write!(f, "ResponseTimeout"),
            ErrorClass::Cancelled => write!(f, "Cancelled"),
            ErrorClass::ResourceExhausted => write!(f, "ResourceExhausted"),
            ErrorClass::WorkerOverloaded => write!(f, "WorkerOverloaded"),
            ErrorClass::Unavailable => write!(f, "Unavailable"),
            ErrorClass::WorkerUnavailable => write!(f, "WorkerUnavailable"),
            ErrorClass::Backend(sub) => write!(f, "Backend{sub}"),
            ErrorClass::InvalidRequest => write!(f, "InvalidRequest"),
            ErrorClass::Unauthenticated => write!(f, "Unauthenticated"),
            ErrorClass::PermissionDenied => write!(f, "PermissionDenied"),
            ErrorClass::NotFound => write!(f, "NotFound"),
            ErrorClass::Conflict => write!(f, "Conflict"),
            ErrorClass::PayloadTooLarge => write!(f, "PayloadTooLarge"),
            ErrorClass::UnsupportedMedia => write!(f, "UnsupportedMedia"),
            ErrorClass::RateLimited => write!(f, "RateLimited"),
            ErrorClass::CapacityExhausted => write!(f, "CapacityExhausted"),
            ErrorClass::BackendProtocol => write!(f, "BackendProtocol"),
            ErrorClass::DeadlineExceeded => write!(f, "DeadlineExceeded"),
            ErrorClass::NotImplemented => write!(f, "NotImplemented"),
            ErrorClass::Internal => write!(f, "Internal"),
        }
    }
}

impl ErrorClass {
    /// Return the canonical semantic class for a legacy or backend-specific variant.
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Unknown => "Unknown",
            Self::InvalidArgument => "InvalidArgument",
            Self::CannotConnect => "CannotConnect",
            Self::Disconnected => "Disconnected",
            Self::ConnectionTimeout => "ConnectionTimeout",
            Self::ResponseTimeout => "ResponseTimeout",
            Self::Cancelled => "Cancelled",
            Self::ResourceExhausted => "ResourceExhausted",
            Self::WorkerOverloaded => "WorkerOverloaded",
            Self::Unavailable => "Unavailable",
            Self::WorkerUnavailable => "WorkerUnavailable",
            Self::Backend(BackendError::Unknown) => "BackendUnknown",
            Self::Backend(BackendError::InvalidArgument) => "BackendInvalidArgument",
            Self::Backend(BackendError::CannotConnect) => "BackendCannotConnect",
            Self::Backend(BackendError::Disconnected) => "BackendDisconnected",
            Self::Backend(BackendError::ConnectionTimeout) => "BackendConnectionTimeout",
            Self::Backend(BackendError::ResponseTimeout) => "BackendResponseTimeout",
            Self::Backend(BackendError::Cancelled) => "BackendCancelled",
            Self::Backend(BackendError::EngineShutdown) => "BackendEngineShutdown",
            Self::Backend(BackendError::StreamIncomplete) => "BackendStreamIncomplete",
            Self::InvalidRequest => "InvalidRequest",
            Self::Unauthenticated => "Unauthenticated",
            Self::PermissionDenied => "PermissionDenied",
            Self::NotFound => "NotFound",
            Self::Conflict => "Conflict",
            Self::PayloadTooLarge => "PayloadTooLarge",
            Self::UnsupportedMedia => "UnsupportedMedia",
            Self::RateLimited => "RateLimited",
            Self::CapacityExhausted => "CapacityExhausted",
            Self::BackendProtocol => "BackendProtocol",
            Self::DeadlineExceeded => "DeadlineExceeded",
            Self::NotImplemented => "NotImplemented",
            Self::Internal => "Internal",
        }
    }

    /// Map a legacy or backend-specific variant to its canonical semantic class.
    ///
    /// New producers should construct canonical classes directly. [`DynamoError::class`] applies this mapping after validating the class/reason identity and is the normal consumer entry point. Serialization also emits that validated canonical class, while retaining a compatible legacy representation during the migration window.
    pub fn normalized(self) -> Self {
        match self {
            Self::Unknown => Self::Internal,
            Self::InvalidArgument => Self::InvalidRequest,
            Self::CannotConnect | Self::Disconnected | Self::WorkerUnavailable => Self::Unavailable,
            Self::ConnectionTimeout | Self::ResponseTimeout => Self::DeadlineExceeded,
            Self::ResourceExhausted | Self::WorkerOverloaded => Self::CapacityExhausted,
            Self::Backend(error) => match error {
                BackendError::Unknown => Self::Internal,
                BackendError::InvalidArgument => Self::InvalidRequest,
                BackendError::CannotConnect
                | BackendError::Disconnected
                | BackendError::EngineShutdown
                | BackendError::StreamIncomplete => Self::Unavailable,
                BackendError::ConnectionTimeout | BackendError::ResponseTimeout => {
                    Self::DeadlineExceeded
                }
                BackendError::Cancelled => Self::Cancelled,
            },
            canonical @ (Self::Cancelled
            | Self::Unavailable
            | Self::InvalidRequest
            | Self::Unauthenticated
            | Self::PermissionDenied
            | Self::NotFound
            | Self::Conflict
            | Self::PayloadTooLarge
            | Self::UnsupportedMedia
            | Self::RateLimited
            | Self::CapacityExhausted
            | Self::BackendProtocol
            | Self::DeadlineExceeded
            | Self::NotImplemented
            | Self::Internal) => canonical,
        }
    }
}

/// Backward-compatible name retained while callers migrate to ErrorClass.
pub type ErrorType = ErrorClass;

/// Categorizes errors into a fixed set of standard types.
///
/// Consumers (e.g., the migration module) inspect the error type to decide
/// what action to take, rather than the error defining its own behavior.
/// Backend engine error subcategories.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum BackendError {
    /// Uncategorized or unknown backend error.
    Unknown,
    /// The request contains invalid input (e.g., prompt exceeds context length).
    InvalidArgument,
    /// Failed to establish a connection to a remote worker.
    CannotConnect,
    /// An established connection was lost unexpectedly.
    Disconnected,
    /// A connection or request timed out.
    ConnectionTimeout,
    /// The backend accepted the request but stopped responding (stream inactivity timeout).
    ResponseTimeout,
    /// The request was cancelled (e.g., client disconnected).
    Cancelled,
    /// The engine process has shut down or crashed.
    EngineShutdown,
    /// The response stream was terminated before completion (e.g., engine dropped mid-stream).
    StreamIncomplete,
}

impl fmt::Display for BackendError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            BackendError::Unknown => write!(f, "Unknown"),
            BackendError::InvalidArgument => write!(f, "InvalidArgument"),
            BackendError::CannotConnect => write!(f, "CannotConnect"),
            BackendError::Disconnected => write!(f, "Disconnected"),
            BackendError::ConnectionTimeout => write!(f, "ConnectionTimeout"),
            BackendError::ResponseTimeout => write!(f, "ResponseTimeout"),
            BackendError::Cancelled => write!(f, "Cancelled"),
            BackendError::EngineShutdown => write!(f, "EngineShutdown"),
            BackendError::StreamIncomplete => write!(f, "StreamIncomplete"),
        }
    }
}

/// Stable, bounded catalog key for a specific failure cause.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct ErrorReason(String);

impl ErrorReason {
    pub const MAX_BYTES: usize = 128;

    pub fn new(value: impl Into<String>) -> Result<Self, InvalidErrorReason> {
        let value = value.into();
        if value.is_empty() {
            return Err(InvalidErrorReason::Empty);
        }
        if value.len() > Self::MAX_BYTES {
            return Err(InvalidErrorReason::TooLong);
        }
        if !value.bytes().all(|byte| {
            byte.is_ascii_lowercase() || byte.is_ascii_digit() || b"._-".contains(&byte)
        }) {
            return Err(InvalidErrorReason::InvalidCharacter);
        }
        if Self::catalog_class(&value).is_none() {
            return Err(InvalidErrorReason::UnknownCatalogKey);
        }
        Ok(Self(value))
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }

    fn catalog_class(value: &str) -> Option<ErrorClass> {
        match value {
            "runtime.unclassified" | "runtime.invalid_error" | "runtime.internal" => {
                Some(ErrorClass::Internal)
            }
            "request.invalid_argument" | "backend.invalid_argument" | "request.invalid" => {
                Some(ErrorClass::InvalidRequest)
            }
            "transport.cannot_connect"
            | "transport.disconnected"
            | "backend.unavailable"
            | "backend.worker_unavailable"
            | "backend.cannot_connect"
            | "backend.disconnected"
            | "backend.engine_shutdown"
            | "backend.stream_incomplete" => Some(ErrorClass::Unavailable),
            "transport.connection_timeout"
            | "backend.response_timeout"
            | "backend.connection_timeout"
            | "request.deadline_exceeded" => Some(ErrorClass::DeadlineExceeded),
            "request.cancelled" | "backend.cancelled" => Some(ErrorClass::Cancelled),
            "capacity.pool_exhausted" | "capacity.worker_overloaded" | "capacity.exhausted" => {
                Some(ErrorClass::CapacityExhausted)
            }
            "backend.unknown" => Some(ErrorClass::Internal),
            "backend.protocol" => Some(ErrorClass::BackendProtocol),
            "request.unauthenticated" => Some(ErrorClass::Unauthenticated),
            "request.permission_denied" => Some(ErrorClass::PermissionDenied),
            "request.not_found" => Some(ErrorClass::NotFound),
            "request.conflict" => Some(ErrorClass::Conflict),
            "request.payload_too_large" => Some(ErrorClass::PayloadTooLarge),
            "request.unsupported_media" => Some(ErrorClass::UnsupportedMedia),
            "request.rate_limited" => Some(ErrorClass::RateLimited),
            "runtime.not_implemented" => Some(ErrorClass::NotImplemented),
            _ => None,
        }
    }

    fn from_static(value: &'static str) -> Self {
        debug_assert!(!value.is_empty() && value.len() <= Self::MAX_BYTES);
        debug_assert!(value.bytes().all(|byte| {
            byte.is_ascii_lowercase() || byte.is_ascii_digit() || b"._-".contains(&byte)
        }));
        debug_assert!(Self::catalog_class(value).is_some());
        Self(value.to_owned())
    }

    fn for_class(class: ErrorClass) -> Self {
        let value = match class {
            ErrorClass::Unknown => "runtime.unclassified",
            ErrorClass::InvalidArgument => "request.invalid_argument",
            ErrorClass::CannotConnect => "transport.cannot_connect",
            ErrorClass::Disconnected => "transport.disconnected",
            ErrorClass::ConnectionTimeout => "transport.connection_timeout",
            ErrorClass::ResponseTimeout => "backend.response_timeout",
            ErrorClass::Cancelled => "request.cancelled",
            ErrorClass::ResourceExhausted => "capacity.pool_exhausted",
            ErrorClass::WorkerOverloaded => "capacity.worker_overloaded",
            ErrorClass::Unavailable => "backend.unavailable",
            ErrorClass::WorkerUnavailable => "backend.worker_unavailable",
            ErrorClass::Backend(error) => match error {
                BackendError::Unknown => "backend.unknown",
                BackendError::InvalidArgument => "backend.invalid_argument",
                BackendError::CannotConnect => "backend.cannot_connect",
                BackendError::Disconnected => "backend.disconnected",
                BackendError::ConnectionTimeout => "backend.connection_timeout",
                BackendError::ResponseTimeout => "backend.response_timeout",
                BackendError::Cancelled => "backend.cancelled",
                BackendError::EngineShutdown => "backend.engine_shutdown",
                BackendError::StreamIncomplete => "backend.stream_incomplete",
            },
            ErrorClass::InvalidRequest => "request.invalid",
            ErrorClass::Unauthenticated => "request.unauthenticated",
            ErrorClass::PermissionDenied => "request.permission_denied",
            ErrorClass::NotFound => "request.not_found",
            ErrorClass::Conflict => "request.conflict",
            ErrorClass::PayloadTooLarge => "request.payload_too_large",
            ErrorClass::UnsupportedMedia => "request.unsupported_media",
            ErrorClass::RateLimited => "request.rate_limited",
            ErrorClass::CapacityExhausted => "capacity.exhausted",
            ErrorClass::BackendProtocol => "backend.protocol",
            ErrorClass::DeadlineExceeded => "request.deadline_exceeded",
            ErrorClass::NotImplemented => "runtime.not_implemented",
            ErrorClass::Internal => "runtime.internal",
        };
        Self::from_static(value)
    }
}

static INVALID_ERROR_REASON: LazyLock<ErrorReason> =
    LazyLock::new(|| ErrorReason::from_static("runtime.invalid_error"));

impl fmt::Display for ErrorReason {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

impl Serialize for ErrorReason {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        serializer.serialize_str(&self.0)
    }
}

impl<'de> Deserialize<'de> for ErrorReason {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        let value = String::deserialize(deserializer)?;
        Self::new(value).map_err(serde::de::Error::custom)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum InvalidErrorReason {
    Empty,
    TooLong,
    InvalidCharacter,
    UnknownCatalogKey,
}

impl fmt::Display for InvalidErrorReason {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Empty => f.write_str("error reason cannot be empty"),
            Self::TooLong => write!(f, "error reason exceeds {} bytes", ErrorReason::MAX_BYTES),
            Self::InvalidCharacter => f.write_str(
                "error reason may contain only lowercase ASCII, digits, '.', '_', or '-'",
            ),
            Self::UnknownCatalogKey => f.write_str("error reason is not registered in the catalog"),
        }
    }
}

impl std::error::Error for InvalidErrorReason {}

/// Closed set of structured, client-safe details.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum PublicDetails {
    Message {
        message: String,
    },
    SizeLimit {
        limit: u64,
        actual: Option<u64>,
    },
    ContextLength {
        limit: u64,
        actual: Option<u64>,
    },
    RateLimit {
        limit: Option<u64>,
        remaining: Option<u64>,
    },
}

impl PublicDetails {
    /// Returns the client-safe rejection message when one was explicitly captured.
    pub fn message(&self) -> Option<&str> {
        match self {
            Self::Message { message } => Some(message),
            Self::SizeLimit { .. } | Self::ContextLength { .. } | Self::RateLimit { .. } => None,
        }
    }
}

/// Bounded operator-only diagnostic text.
#[derive(Debug, Clone, Default)]
pub struct Diagnostic {
    message: String,
    source: Option<Arc<dyn std::error::Error + Send + Sync>>,
}

impl Diagnostic {
    pub const MAX_BYTES: usize = 4096;
    pub const TRUNCATION_SUFFIX: &'static str = "...[truncated]";

    pub fn new(value: impl Into<String>) -> Self {
        let mut message = value.into();
        if message.len() > Self::MAX_BYTES {
            let mut end = Self::MAX_BYTES - Self::TRUNCATION_SUFFIX.len();
            while !message.is_char_boundary(end) {
                end -= 1;
            }
            message.truncate(end);
            message.push_str(Self::TRUNCATION_SUFFIX);
        }
        Self {
            message,
            source: None,
        }
    }

    fn with_source(mut self, source: impl std::error::Error + Send + Sync + 'static) -> Self {
        self.source = Some(Arc::new(source));
        self
    }

    pub fn as_str(&self) -> &str {
        &self.message
    }

    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        self.source
            .as_deref()
            .map(|source| source as &(dyn std::error::Error + 'static))
    }
}

impl PartialEq for Diagnostic {
    fn eq(&self, other: &Self) -> bool {
        self.message == other.message
    }
}

impl Eq for Diagnostic {}

impl Serialize for Diagnostic {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        serializer.serialize_str(&self.message)
    }
}

impl<'de> Deserialize<'de> for Diagnostic {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        Ok(Self::new(String::deserialize(deserializer)?))
    }
}

// ============================================================================
// DynamoError - The Standardized Error Type
// ============================================================================

/// The standardized error type for Dynamo.
///
/// `DynamoError` is a serializable semantic error that:
/// - Carries an [`ErrorClass`] for categorization
/// - Is serializable for network transmission via `Annotated`
/// - Can be created from any [`std::error::Error`]
///
/// # Display
///
/// `Display` shows the private diagnostic when present and otherwise the reason.
///
/// ```rust,ignore
/// let err = DynamoError::msg("outer");
/// println!("{}", err); // "Internal: outer"
/// ```
#[derive(Debug, Clone)]
pub struct DynamoError {
    /// Coarse semantic category. This may retain a compatible legacy variant internally; use [`Self::class`] for validated consumer policy.
    pub class: ErrorClass,
    /// Registered, bounded cause key that must belong to the normalized class; use [`Self::reason`] for validated consumer policy.
    pub reason: ErrorReason,
    /// Optional bounded operator-only context. This is not client-safe and must not be used as a metric label.
    pub diagnostic: Option<Diagnostic>,
    /// Optional explicitly client-safe data. The producer remains responsible for ensuring every contained value is safe to expose.
    pub public: Option<PublicDetails>,
}

impl Serialize for DynamoError {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        use serde::ser::SerializeStruct;

        let public = self.public_details();
        let caused_by = self
            .diagnostic
            .as_ref()
            .and_then(Diagnostic::source)
            .and_then(|source| source.downcast_ref::<DynamoError>());
        let mut state = serializer.serialize_struct(
            "DynamoError",
            4 + usize::from(self.diagnostic.is_some())
                + usize::from(public.is_some())
                + usize::from(caused_by.is_some()),
        )?;
        state.serialize_field("error_type", &self.legacy_wire_error_type())?;
        state.serialize_field("class", &self.class())?;
        state.serialize_field("reason", self.reason())?;
        state.serialize_field("message", &self.message())?;
        if let Some(diagnostic) = &self.diagnostic {
            state.serialize_field("diagnostic", diagnostic)?;
        }
        if let Some(public) = public {
            state.serialize_field("public", public)?;
        }
        if let Some(caused_by) = caused_by {
            state.serialize_field("caused_by", caused_by)?;
        }
        state.end()
    }
}

impl<'de> Deserialize<'de> for DynamoError {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        #[derive(Deserialize)]
        struct Representation {
            #[serde(default)]
            class: Option<ErrorClass>,
            #[serde(default)]
            error_type: Option<ErrorClass>,
            #[serde(default)]
            reason: Option<String>,
            #[serde(default)]
            diagnostic: Option<Diagnostic>,
            #[serde(default)]
            message: Option<Diagnostic>,
            #[serde(default, rename = "public", alias = "public_details")]
            public: Option<PublicDetails>,
            #[serde(default)]
            caused_by: Option<Box<DynamoError>>,
        }

        let representation = Representation::deserialize(deserializer)?;
        let declared_class = representation.class;
        let legacy_error_type = representation.error_type;
        let source_class = declared_class
            .or(legacy_error_type)
            .unwrap_or(ErrorClass::Unknown);
        let canonical_class = match source_class {
            ErrorClass::Unknown => ErrorClass::Internal,
            class => class.normalized(),
        };
        let reason = match representation.reason {
            Some(reason) => ErrorReason::new(reason).ok(),
            None => Some(ErrorReason::for_class(
                legacy_error_type.unwrap_or(source_class),
            )),
        };
        let valid_reason = reason
            .as_ref()
            .and_then(|reason| ErrorReason::catalog_class(reason.as_str()))
            .is_some_and(|class| class == canonical_class);
        let stored_class = legacy_error_type
            .filter(|class| *class != ErrorClass::Unknown && class.normalized() == canonical_class)
            .unwrap_or(canonical_class);

        let (class, reason, public) = match reason {
            Some(reason) if valid_reason => (stored_class, reason, representation.public),
            _ => (
                ErrorClass::Internal,
                ErrorReason::from_static("runtime.invalid_error"),
                None,
            ),
        };

        let diagnostic = match (
            representation.diagnostic.or(representation.message),
            representation.caused_by,
        ) {
            (Some(diagnostic), Some(caused_by)) => Some(diagnostic.with_source(*caused_by)),
            (None, Some(caused_by)) => Some(Diagnostic::new("").with_source(*caused_by)),
            (diagnostic, None) => diagnostic,
        };

        Ok(Self {
            class,
            reason,
            diagnostic,
            public,
        })
    }
}

impl DynamoError {
    /// Create a builder for constructing a `DynamoError`.
    pub fn builder() -> DynamoErrorBuilder {
        DynamoErrorBuilder::default()
    }

    /// Shorthand to create an internal error with a private diagnostic.
    pub fn msg(message: impl Into<String>) -> Self {
        Self::builder().diagnostic(message).build()
    }

    /// Returns the validated legacy error type without normalization.
    ///
    /// Invalid public-field combinations fail closed so legacy policy callers
    /// cannot bypass the canonical identity check.
    pub fn error_type(&self) -> ErrorType {
        if self.has_valid_identity() {
            self.class
        } else {
            ErrorClass::Internal
        }
    }

    fn has_valid_identity(&self) -> bool {
        ErrorReason::catalog_class(self.reason.as_str())
            .is_some_and(|class| class == self.class.normalized())
    }

    /// Returns the canonical semantic error class.
    pub fn class(&self) -> ErrorClass {
        if self.has_valid_identity() {
            self.class.normalized()
        } else {
            ErrorClass::Internal
        }
    }

    fn legacy_wire_error_type(&self) -> ErrorClass {
        match self.error_type() {
            ErrorClass::Unknown
            | ErrorClass::Internal
            | ErrorClass::Unauthenticated
            | ErrorClass::PermissionDenied
            | ErrorClass::NotFound
            | ErrorClass::Conflict
            | ErrorClass::BackendProtocol
            | ErrorClass::NotImplemented => ErrorClass::Unknown,
            ErrorClass::InvalidArgument
            | ErrorClass::InvalidRequest
            | ErrorClass::PayloadTooLarge
            | ErrorClass::UnsupportedMedia => ErrorClass::InvalidArgument,
            ErrorClass::CannotConnect => ErrorClass::CannotConnect,
            ErrorClass::Disconnected => ErrorClass::Disconnected,
            ErrorClass::ConnectionTimeout => ErrorClass::ConnectionTimeout,
            ErrorClass::ResponseTimeout | ErrorClass::DeadlineExceeded => {
                ErrorClass::ResponseTimeout
            }
            ErrorClass::Cancelled => ErrorClass::Cancelled,
            ErrorClass::ResourceExhausted
            | ErrorClass::WorkerOverloaded
            | ErrorClass::CapacityExhausted
            | ErrorClass::RateLimited => ErrorClass::ResourceExhausted,
            ErrorClass::Unavailable => ErrorClass::Unavailable,
            ErrorClass::WorkerUnavailable => ErrorClass::WorkerUnavailable,
            ErrorClass::Backend(error) => ErrorClass::Backend(error),
        }
    }

    /// Returns the stable reason key.
    pub fn reason(&self) -> &ErrorReason {
        if self.has_valid_identity() {
            &self.reason
        } else {
            &INVALID_ERROR_REASON
        }
    }

    /// Returns the optional private diagnostic.
    pub fn diagnostic(&self) -> Option<&Diagnostic> {
        self.diagnostic.as_ref()
    }

    /// Returns structured client-safe details.
    pub fn public_details(&self) -> Option<&PublicDetails> {
        self.has_valid_identity()
            .then_some(self.public.as_ref())
            .flatten()
    }

    /// Returns the explicitly captured client-safe rejection message.
    pub fn public_message(&self) -> Option<&str> {
        self.public_details().and_then(PublicDetails::message)
    }

    /// Returns the legacy error message view.
    pub fn message(&self) -> &str {
        self.diagnostic
            .as_ref()
            .map(Diagnostic::as_str)
            .unwrap_or_default()
    }
}

impl fmt::Display for DynamoError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self.diagnostic() {
            Some(diagnostic) if !diagnostic.as_str().is_empty() => {
                write!(f, "{}: {}", self.class(), diagnostic.as_str())
            }
            _ => write!(f, "{}: {}", self.class(), self.reason()),
        }
    }
}

impl std::error::Error for DynamoError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        self.diagnostic.as_ref().and_then(Diagnostic::source)
    }
}

/// Convert from a reference to any `std::error::Error`.
impl<'a> From<&'a (dyn std::error::Error + 'static)> for DynamoError {
    fn from(err: &'a (dyn std::error::Error + 'static)) -> Self {
        if let Some(dynamo_err) = err.downcast_ref::<DynamoError>() {
            return dynamo_err.clone();
        }

        let diagnostic = Diagnostic::new(err.to_string());
        let Some(source) = err.source() else {
            return Self {
                class: ErrorClass::Internal,
                reason: ErrorReason::from_static("runtime.unclassified"),
                diagnostic: Some(diagnostic),
                public: None,
            };
        };

        let source = DynamoError::from(source);
        let diagnostic = diagnostic.with_source(source.clone());
        if source.has_valid_identity() && source.class() != ErrorClass::Internal {
            return Self {
                class: source.error_type(),
                reason: source.reason().clone(),
                diagnostic: Some(diagnostic),
                public: source.public_details().cloned(),
            };
        }

        Self {
            class: ErrorClass::Internal,
            reason: ErrorReason::from_static("runtime.unclassified"),
            diagnostic: Some(diagnostic),
            public: None,
        }
    }
}

/// Convert from an owned boxed `std::error::Error`.
impl From<Box<dyn std::error::Error + 'static>> for DynamoError {
    fn from(err: Box<dyn std::error::Error + 'static>) -> Self {
        match err.downcast::<DynamoError>() {
            Ok(dynamo_err) => *dynamo_err,
            Err(err) => DynamoError::from(&*err as &(dyn std::error::Error + 'static)),
        }
    }
}

// ============================================================================
// DynamoErrorBuilder
// ============================================================================

/// Builder for constructing a [`DynamoError`].
///
/// # Example
/// ```rust,ignore
/// let err = DynamoError::builder()
///     .error_type(ErrorClass::Disconnected)
///     .message("worker lost")
///     .cause(some_io_error)
///     .build();
/// ```
#[derive(Default)]
pub struct DynamoErrorBuilder {
    class: Option<ErrorClass>,
    reason: Option<ErrorReason>,
    diagnostic: Option<Diagnostic>,
    public: Option<PublicDetails>,
}

impl DynamoErrorBuilder {
    /// Set the legacy or canonical error class.
    pub fn error_type(mut self, error_type: ErrorType) -> Self {
        self.class = Some(error_type);
        self
    }

    /// Set the canonical error class.
    pub fn class(self, class: ErrorClass) -> Self {
        self.error_type(class)
    }

    /// Set the stable reason key.
    pub fn reason(mut self, reason: ErrorReason) -> Self {
        self.reason = Some(reason);
        self
    }

    /// Set the private bounded diagnostic.
    pub fn diagnostic(mut self, diagnostic: impl Into<String>) -> Self {
        self.diagnostic = Some(Diagnostic::new(diagnostic));
        self
    }

    /// Set the legacy error message view.
    pub fn message(self, message: impl Into<String>) -> Self {
        self.diagnostic(message)
    }

    /// Set a client-safe rejection message.
    pub fn public_message(mut self, message: impl Into<String>) -> Self {
        self.public = Some(PublicDetails::Message {
            message: message.into(),
        });
        self
    }

    /// Set structured client-safe details.
    pub fn public_details(mut self, public: PublicDetails) -> Self {
        self.public = Some(public);
        self
    }

    /// Preserve compatibility with existing builders while keeping native causes out of the semantic payload.
    pub fn cause(mut self, cause: impl std::error::Error + 'static) -> Self {
        let message = cause.to_string();
        let source = DynamoError::from(&cause as &(dyn std::error::Error + 'static));
        let diagnostic = self
            .diagnostic
            .take()
            .unwrap_or_else(|| Diagnostic::new(message));
        self.diagnostic = Some(diagnostic.with_source(source));
        self
    }

    /// Build the `DynamoError` and fail closed on a class/reason mismatch.
    pub fn build(self) -> DynamoError {
        let raw_class = self.class.unwrap_or(ErrorClass::Internal);
        let reason = self
            .reason
            .unwrap_or_else(|| ErrorReason::for_class(raw_class));
        let valid_reason = ErrorReason::catalog_class(reason.as_str())
            .is_some_and(|class| class == raw_class.normalized());

        if valid_reason {
            DynamoError {
                class: raw_class,
                reason,
                diagnostic: self.diagnostic,
                public: self.public,
            }
        } else {
            DynamoError {
                class: ErrorClass::Internal,
                reason: ErrorReason::from_static("runtime.invalid_error"),
                diagnostic: self.diagnostic,
                public: None,
            }
        }
    }
}

// ============================================================================
// Utility Functions
// ============================================================================

/// Check whether an error chain contains a specific set of error types
/// while not containing any of the excluded error types.
///
/// Walks the chain via `source()`, inspecting each error that can be downcast
/// to `DynamoError`. Returns `false` immediately if any error's type is in
/// `exclude_set`. Otherwise, returns `true` if at least one error's type is
/// in `match_set`. Errors that are not `DynamoError` are skipped.
pub fn match_error_chain(
    err: &(dyn std::error::Error + 'static),
    match_set: &[ErrorClass],
    exclude_set: &[ErrorClass],
) -> bool {
    let mut found = false;
    let mut current: Option<&(dyn std::error::Error + 'static)> = Some(err);

    while let Some(e) = current {
        if let Some(dynamo_err) = e.downcast_ref::<DynamoError>() {
            if exclude_set.contains(&dynamo_err.error_type()) {
                return false;
            }
            if match_set.contains(&dynamo_err.error_type()) {
                found = true;
            }
        }
        current = e.source();
    }

    found
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use std::error::Error;

    // Compile-time assertions that DynamoError is std::error::Error + Send + Sync + 'static.
    // These fail at compile time if a future change breaks these guarantees.
    const _: () = {
        fn assert_stderror<T: std::error::Error>() {}
        fn assert_send<T: Send>() {}
        fn assert_sync<T: Sync>() {}
        fn assert_static<T: 'static>() {}
        fn assert_all() {
            assert_stderror::<DynamoError>();
            assert_send::<DynamoError>();
            assert_sync::<DynamoError>();
            assert_static::<DynamoError>();
        }
    };

    #[test]
    fn test_msg_constructor() {
        let err = DynamoError::msg("something failed");
        assert_eq!(err.error_type(), ErrorClass::Internal);
        assert_eq!(err.reason().as_str(), "runtime.internal");
        assert_eq!(err.message(), "something failed");
        assert!(err.source().is_none());
    }

    #[test]
    fn cause_is_preserved_for_legacy_wire_compatibility() {
        let err = DynamoError::builder()
            .class(ErrorClass::Internal)
            .reason(ErrorReason::new("runtime.internal").unwrap())
            .diagnostic("operation failed")
            .cause(std::io::Error::other("io error"))
            .build();

        assert_eq!(err.message(), "operation failed");
        assert_eq!(err.source().unwrap().to_string(), "Internal: io error");
        let value = serde_json::to_value(&err).unwrap();
        assert_eq!(value.as_object().unwrap().len(), 6);
        assert!(value.get("caused_by").is_some());

        let roundtrip: DynamoError = serde_json::from_value(value).unwrap();
        assert_eq!(
            roundtrip.source().unwrap().to_string(),
            "Internal: io error"
        );
    }

    #[test]
    fn display_uses_diagnostic_or_reason() {
        let with_diagnostic = DynamoError::builder()
            .class(ErrorClass::Internal)
            .reason(ErrorReason::new("runtime.internal").unwrap())
            .diagnostic("operation failed")
            .build();
        let without_diagnostic = DynamoError::builder()
            .class(ErrorClass::Internal)
            .reason(ErrorReason::new("runtime.internal").unwrap())
            .build();

        assert_eq!(with_diagnostic.to_string(), "Internal: operation failed");
        assert_eq!(without_diagnostic.to_string(), "Internal: runtime.internal");
    }

    #[test]
    fn conversion_preserves_nested_semantic_source_locally() {
        #[derive(Debug)]
        struct OuterError {
            source: DynamoError,
        }

        impl fmt::Display for OuterError {
            fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                f.write_str("outer failure")
            }
        }

        impl std::error::Error for OuterError {
            fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
                Some(&self.source)
            }
        }

        let outer = OuterError {
            source: DynamoError::builder()
                .error_type(ErrorType::InvalidArgument)
                .diagnostic("invalid input")
                .build(),
        };
        let converted = DynamoError::from(&outer as &(dyn std::error::Error + 'static));

        assert!(match_error_chain(
            &converted,
            &[ErrorType::InvalidArgument],
            &[]
        ));
        let value = serde_json::to_value(converted).unwrap();
        assert!(value.get("caused_by").is_some());
    }

    #[test]
    fn test_from_boxed_std_error() {
        let std_err = std::io::Error::other("io error");
        let boxed: Box<dyn std::error::Error> = Box::new(std_err);
        let dynamo_err = DynamoError::from(boxed);

        assert_eq!(dynamo_err.class(), ErrorClass::Internal);
        assert_eq!(dynamo_err.reason().as_str(), "runtime.unclassified");
        assert_eq!(dynamo_err.message(), "io error");
    }

    #[test]
    fn test_from_boxed_takes_ownership_of_dynamo_error() {
        let inner = DynamoError::msg("original");
        let boxed: Box<dyn std::error::Error> = Box::new(inner);
        let dynamo_err = DynamoError::from(boxed);

        assert_eq!(dynamo_err.class(), ErrorClass::Internal);
        assert_eq!(dynamo_err.message(), "original");
    }

    #[test]
    fn semantic_metadata_roundtrips() {
        let err = DynamoError::builder()
            .class(ErrorClass::RateLimited)
            .reason(ErrorReason::new("request.rate_limited").unwrap())
            .diagnostic("request rate exceeded")
            .public_details(PublicDetails::RateLimit {
                limit: Some(100),
                remaining: Some(0),
            })
            .build();

        let json = serde_json::to_string(&err).unwrap();
        let deserialized: DynamoError = serde_json::from_str(&json).unwrap();

        assert_eq!(deserialized.class(), ErrorClass::RateLimited);
        assert_eq!(deserialized.reason().as_str(), "request.rate_limited");
        assert_eq!(
            deserialized.diagnostic().map(Diagnostic::as_str),
            Some("request rate exceeded")
        );
        assert_eq!(
            deserialized.public_details(),
            Some(&PublicDetails::RateLimit {
                limit: Some(100),
                remaining: Some(0),
            })
        );
    }

    #[test]
    fn semantic_schema_includes_legacy_compatibility_fields() {
        let err = DynamoError::builder()
            .class(ErrorClass::RateLimited)
            .reason(ErrorReason::new("request.rate_limited").unwrap())
            .diagnostic("request rate exceeded")
            .public_details(PublicDetails::RateLimit {
                limit: Some(100),
                remaining: Some(0),
            })
            .build();

        let value = serde_json::to_value(err).unwrap();
        let object = value.as_object().unwrap();

        assert_eq!(object.len(), 6);
        assert_eq!(value["error_type"], "ResourceExhausted");
        assert_eq!(value["message"], "request rate exceeded");
        assert_eq!(value["class"], "RateLimited");
        assert_eq!(value["reason"], "request.rate_limited");
        assert_eq!(value["diagnostic"], "request rate exceeded");
        assert!(value.get("public").is_some());
    }

    #[test]
    fn optional_semantic_fields_are_omitted() {
        let err = DynamoError::builder()
            .class(ErrorClass::InvalidRequest)
            .reason(ErrorReason::new("request.invalid").unwrap())
            .build();

        let value = serde_json::to_value(err).unwrap();
        let object = value.as_object().unwrap();

        assert_eq!(object.len(), 4);
        assert_eq!(value["error_type"], "InvalidArgument");
        assert_eq!(value["message"], "");
        assert_eq!(value["class"], "InvalidRequest");
        assert_eq!(value["reason"], "request.invalid");
        assert!(value.get("diagnostic").is_none());
        assert!(value.get("public").is_none());
    }

    #[test]
    fn unknown_catalog_reason_fails_closed() {
        let json = r#"{
            "class": "InvalidRequest",
            "reason": "request.user_supplied_metric_label",
            "diagnostic": "private details"
        }"#;
        let err: DynamoError = serde_json::from_str(json).unwrap();

        assert_eq!(err.class(), ErrorClass::Internal);
        assert_eq!(err.reason().as_str(), "runtime.invalid_error");
        assert_eq!(
            err.diagnostic().map(Diagnostic::as_str),
            Some("private details")
        );
    }

    #[test]
    fn public_fields_fail_closed_at_consumer_boundaries() {
        let err = DynamoError {
            class: ErrorClass::InvalidRequest,
            reason: ErrorReason::new("request.rate_limited").unwrap(),
            diagnostic: Some(Diagnostic::new("private details")),
            public: Some(PublicDetails::RateLimit {
                limit: Some(10),
                remaining: Some(0),
            }),
        };

        assert_eq!(err.class(), ErrorClass::Internal);
        assert_eq!(err.error_type(), ErrorClass::Internal);
        assert_eq!(err.reason().as_str(), "runtime.invalid_error");
        assert!(err.public_details().is_none());
        assert_eq!(err.to_string(), "Internal: private details");
        assert_eq!(
            serde_json::to_value(err).unwrap(),
            serde_json::json!({
                "error_type": "Unknown",
                "message": "private details",
                "class": "Internal",
                "reason": "runtime.invalid_error",
                "diagnostic": "private details"
            })
        );
    }

    #[test]
    fn diagnostic_is_bounded_at_utf8_boundary() {
        let truncation_index = Diagnostic::MAX_BYTES - Diagnostic::TRUNCATION_SUFFIX.len();
        let diagnostic = Diagnostic::new(
            "x".repeat(truncation_index - 1) + "é" + &"x".repeat(Diagnostic::MAX_BYTES),
        );

        assert!(diagnostic.as_str().len() <= Diagnostic::MAX_BYTES);
        assert!(
            diagnostic
                .as_str()
                .is_char_boundary(diagnostic.as_str().len())
        );
        assert!(diagnostic.as_str().ends_with(Diagnostic::TRUNCATION_SUFFIX));
    }

    #[test]
    fn legacy_builder_derives_semantic_defaults() {
        let legacy_type: ErrorType = ErrorType::InvalidArgument;
        let err = DynamoError::builder()
            .error_type(legacy_type)
            .message("bad request")
            .build();

        assert_eq!(err.error_type(), ErrorType::InvalidArgument);
        assert_eq!(err.class(), ErrorClass::InvalidRequest);
        assert_eq!(err.reason().as_str(), "request.invalid_argument");
        assert_eq!(err.message(), "bad request");
    }

    #[test]
    fn legacy_json_derives_semantic_defaults() {
        let json = r#"{"error_type":"InvalidArgument","message":"bad request"}"#;
        let err: DynamoError = serde_json::from_str(json).unwrap();

        assert_eq!(err.class(), ErrorClass::InvalidRequest);
        assert_eq!(err.reason().as_str(), "request.invalid_argument");
        assert_eq!(
            err.diagnostic().map(Diagnostic::as_str),
            Some("bad request")
        );
        assert!(err.public_details().is_none());
    }

    #[test]
    fn legacy_unknown_preserves_nested_classification_across_the_wire() {
        let json = r#"{
            "error_type":"Unknown",
            "message":"generate failed",
            "caused_by":{"error_type":"InvalidArgument","message":"bad request"}
        }"#;
        let err: DynamoError = serde_json::from_str(json).unwrap();

        assert_eq!(err.reason().as_str(), "runtime.unclassified");
        assert_eq!(err.class(), ErrorClass::Internal);
        assert!(match_error_chain(&err, &[ErrorClass::InvalidArgument], &[]));

        let reserialized = serde_json::to_value(&err).unwrap();
        assert_eq!(reserialized["caused_by"]["error_type"], "InvalidArgument");
    }

    #[test]
    fn backend_class_roundtrips_as_its_canonical_class() {
        let error = DynamoError::builder()
            .error_type(ErrorClass::Backend(BackendError::InvalidArgument))
            .build();

        let serialized = serde_json::to_string(&error).unwrap();
        assert!(serialized.contains("\"class\":\"InvalidRequest\""));

        let decoded: DynamoError = serde_json::from_str(&serialized).unwrap();
        assert_eq!(
            decoded.error_type(),
            ErrorClass::Backend(BackendError::InvalidArgument)
        );
        assert_eq!(decoded.class(), ErrorClass::InvalidRequest);
        assert_eq!(decoded.reason().as_str(), "backend.invalid_argument");
    }

    #[test]
    fn transport_subtype_roundtrips() {
        let error = DynamoError::builder()
            .error_type(ErrorClass::CannotConnect)
            .build();

        let decoded: DynamoError =
            serde_json::from_str(&serde_json::to_string(&error).unwrap()).unwrap();

        assert_eq!(decoded.error_type(), ErrorClass::CannotConnect);
        assert_eq!(decoded.class(), ErrorClass::Unavailable);
        assert_eq!(decoded.reason().as_str(), "transport.cannot_connect");
    }

    #[test]
    fn worker_unavailable_roundtrips_with_semantic_identity() {
        let error = DynamoError::builder()
            .error_type(ErrorClass::WorkerUnavailable)
            .build();

        let decoded: DynamoError =
            serde_json::from_str(&serde_json::to_string(&error).unwrap()).unwrap();

        assert_eq!(decoded.error_type(), ErrorClass::WorkerUnavailable);
        assert_eq!(decoded.class(), ErrorClass::Unavailable);
        assert_eq!(decoded.reason().as_str(), "backend.worker_unavailable");
    }

    #[test]
    fn response_timeout_subtypes_roundtrip() {
        for error_type in [
            ErrorClass::ResponseTimeout,
            ErrorClass::Backend(BackendError::ResponseTimeout),
        ] {
            let error = DynamoError::builder().error_type(error_type).build();
            let decoded: DynamoError =
                serde_json::from_str(&serde_json::to_string(&error).unwrap()).unwrap();

            assert_eq!(decoded.error_type(), error_type);
            assert_eq!(decoded.class(), ErrorClass::DeadlineExceeded);
        }
    }

    #[test]
    fn unknown_class_fails_closed_during_deserialization() {
        let json = r#"{"class":"FutureErrorClass","reason":"runtime.internal"}"#;
        let error: DynamoError = serde_json::from_str(json).unwrap();

        assert_eq!(error.error_type(), ErrorClass::Internal);
        assert_eq!(error.class(), ErrorClass::Internal);
        assert_eq!(error.reason().as_str(), "runtime.internal");
    }

    #[test]
    fn error_class_deserialization_preserves_backend_and_fails_closed() {
        let backend: ErrorClass = serde_json::from_str(r#"{"Backend":"InvalidArgument"}"#).unwrap();
        assert_eq!(backend, ErrorClass::Backend(BackendError::InvalidArgument));

        let unknown: ErrorClass = serde_json::from_str(r#""FutureErrorClass""#).unwrap();
        assert_eq!(unknown, ErrorClass::Unknown);
    }

    #[test]
    fn wrapping_preserves_the_semantic_head_across_the_wire() {
        #[derive(Debug)]
        struct Wrapper(DynamoError);

        impl fmt::Display for Wrapper {
            fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                f.write_str("transport wrapper")
            }
        }

        impl std::error::Error for Wrapper {
            fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
                Some(&self.0)
            }
        }

        let wrapped = Wrapper(
            DynamoError::builder()
                .class(ErrorClass::Unavailable)
                .reason(ErrorReason::new("transport.disconnected").unwrap())
                .build(),
        );
        let error = DynamoError::from(&wrapped as &(dyn std::error::Error + 'static));

        assert_eq!(error.class(), ErrorClass::Unavailable);
        assert_eq!(error.reason().as_str(), "transport.disconnected");
        let value = serde_json::to_value(&error).unwrap();
        assert!(value.get("caused_by").is_some());
        let decoded: DynamoError = serde_json::from_value(value).unwrap();
        assert_eq!(decoded.class(), ErrorClass::Unavailable);
        assert_eq!(decoded.reason().as_str(), "transport.disconnected");
    }

    #[test]
    fn malformed_semantic_reason_fails_closed() {
        let json = r#"{
            "class": "InvalidRequest",
            "reason": "INVALID REASON",
            "diagnostic": "private details",
            "public": {"type": "rate_limit", "limit": 10, "remaining": 0}
        }"#;
        let err: DynamoError = serde_json::from_str(json).unwrap();

        assert_eq!(err.error_type(), ErrorClass::Internal);
        assert_eq!(err.class(), ErrorClass::Internal);
        assert_eq!(err.reason().as_str(), "runtime.invalid_error");
        assert!(err.public_details().is_none());
    }

    #[test]
    fn test_error_type_display() {
        assert_eq!(ErrorClass::Unknown.to_string(), "Unknown");
        assert_eq!(ErrorClass::InvalidArgument.to_string(), "InvalidArgument");
        assert_eq!(ErrorClass::CannotConnect.to_string(), "CannotConnect");
        assert_eq!(ErrorClass::Disconnected.to_string(), "Disconnected");
        assert_eq!(
            ErrorClass::ConnectionTimeout.to_string(),
            "ConnectionTimeout"
        );
        assert_eq!(ErrorClass::ResponseTimeout.to_string(), "ResponseTimeout");
        assert_eq!(ErrorClass::Cancelled.to_string(), "Cancelled");
        assert_eq!(
            ErrorClass::ResourceExhausted.to_string(),
            "ResourceExhausted"
        );
        assert_eq!(ErrorClass::WorkerOverloaded.to_string(), "WorkerOverloaded");
        assert_eq!(ErrorClass::Unavailable.to_string(), "Unavailable");
        assert_eq!(
            ErrorClass::WorkerUnavailable.to_string(),
            "WorkerUnavailable"
        );
        assert_eq!(
            ErrorClass::Backend(BackendError::Unknown).to_string(),
            "BackendUnknown"
        );
        assert_eq!(
            ErrorClass::Backend(BackendError::InvalidArgument).to_string(),
            "BackendInvalidArgument"
        );
        assert_eq!(
            ErrorClass::Backend(BackendError::CannotConnect).to_string(),
            "BackendCannotConnect"
        );
        assert_eq!(
            ErrorClass::Backend(BackendError::Disconnected).to_string(),
            "BackendDisconnected"
        );
        assert_eq!(
            ErrorClass::Backend(BackendError::ConnectionTimeout).to_string(),
            "BackendConnectionTimeout"
        );
        assert_eq!(
            ErrorClass::Backend(BackendError::Cancelled).to_string(),
            "BackendCancelled"
        );
        assert_eq!(
            ErrorClass::Backend(BackendError::EngineShutdown).to_string(),
            "BackendEngineShutdown"
        );
        assert_eq!(
            ErrorClass::Backend(BackendError::StreamIncomplete).to_string(),
            "BackendStreamIncomplete"
        );
        assert_eq!(
            ErrorClass::Backend(BackendError::ResponseTimeout).to_string(),
            "BackendResponseTimeout"
        );
    }
}
