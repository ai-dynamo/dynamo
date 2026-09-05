// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Dynamo Error System
//!
//! This module provides a standardized, serializable error type for Dynamo.
//!
//! # DynamoError
//!
//! [`DynamoError`] is the standardized error type for Dynamo. It can be created
//! directly or converted from any [`std::error::Error`]:
//!
//! ```rust,ignore
//! use dynamo_runtime::error::{DynamoError, ErrorClass};
//!
//! // Simple error
//! let err = DynamoError::msg("something failed");
//!
//! // Typed error with a private diagnostic
//! let err = DynamoError::builder()
//!     .class(ErrorClass::Internal)
//!     .diagnostic("operation failed")
//!     .build();
//!
//! // Convert from any std::error::Error
//! let std_err = std::io::Error::other("io error");
//! let dynamo_err = DynamoError::from(Box::new(std_err) as Box<dyn std::error::Error>);
//! ```

use serde::{Deserialize, Serialize};
use std::fmt;
use std::sync::{Arc, LazyLock};

// ============================================================================
// ErrorClass Enum
// ============================================================================

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ErrorClass {
    /// Uncategorized or unknown error.
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
    /// The capacity constraint cannot be relieved by selecting another worker.
    /// This most commonly means the whole eligible worker pool is exhausted.
    ResourceExhausted,
    /// One selected worker is out of capacity while others may still have room.
    /// Distinct from [`Self::ResourceExhausted`] so a request whose routing
    /// constraints permit reassignment can migrate; both surface as HTTP 529.
    WorkerOverloaded,
    /// No backend worker is currently available to handle the request.
    Unavailable,
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

    pub fn normalized(self) -> Self {
        match self {
            Self::Unknown => Self::Internal,
            Self::InvalidArgument => Self::InvalidRequest,
            Self::CannotConnect | Self::Disconnected => Self::Unavailable,
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
    pub class: ErrorClass,
    pub reason: ErrorReason,
    pub diagnostic: Option<Diagnostic>,
    pub public: Option<PublicDetails>,
}

impl Serialize for DynamoError {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        use serde::ser::SerializeStruct;

        let public = self.public_details();
        let mut state = serializer.serialize_struct(
            "DynamoError",
            2 + usize::from(self.diagnostic.is_some()) + usize::from(public.is_some()),
        )?;
        state.serialize_field("class", &self.class())?;
        state.serialize_field("reason", self.reason())?;
        if let Some(diagnostic) = &self.diagnostic {
            state.serialize_field("diagnostic", diagnostic)?;
        }
        if let Some(public) = public {
            state.serialize_field("public", public)?;
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
            #[serde(alias = "error_type")]
            class: ErrorClass,
            #[serde(default)]
            reason: Option<String>,
            #[serde(default, alias = "message")]
            diagnostic: Option<Diagnostic>,
            #[serde(default, rename = "public", alias = "public_details")]
            public: Option<PublicDetails>,
        }

        let representation = Representation::deserialize(deserializer)?;
        let raw_class = representation.class;
        let reason = match representation.reason {
            Some(reason) => ErrorReason::new(reason).ok(),
            None => Some(ErrorReason::for_class(raw_class)),
        };
        let valid_reason = reason
            .as_ref()
            .and_then(|reason| ErrorReason::catalog_class(reason.as_str()))
            .is_some_and(|class| class == raw_class.normalized());

        let (class, reason, public) = match reason {
            Some(reason) if valid_reason => (raw_class, reason, representation.public),
            _ => (
                ErrorClass::Internal,
                ErrorReason::from_static("runtime.invalid_error"),
                None,
            ),
        };

        Ok(Self {
            class,
            reason,
            diagnostic: representation.diagnostic,
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
        let diagnostic = match err.source() {
            Some(source) => diagnostic.with_source(DynamoError::from(source)),
            None => diagnostic,
        };

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
    fn cause_is_local_and_does_not_expand_the_semantic_payload() {
        let err = DynamoError::builder()
            .class(ErrorClass::Internal)
            .reason(ErrorReason::new("runtime.internal").unwrap())
            .diagnostic("operation failed")
            .cause(std::io::Error::other("io error"))
            .build();

        assert_eq!(err.message(), "operation failed");
        assert_eq!(err.source().unwrap().to_string(), "Internal: io error");
        let value = serde_json::to_value(&err).unwrap();
        assert_eq!(value.as_object().unwrap().len(), 3);

        let roundtrip: DynamoError = serde_json::from_value(value).unwrap();
        assert!(roundtrip.source().is_none());
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
        assert!(value.get("caused_by").is_none());
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
    fn semantic_schema_serializes_only_four_fields() {
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

        assert_eq!(object.len(), 4);
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

        assert_eq!(object.len(), 2);
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
                "class": "Internal",
                "reason": "runtime.invalid_error",
                "diagnostic": "private details"
            })
        );
    }

    #[test]
    fn diagnostic_is_bounded_at_utf8_boundary() {
        let diagnostic = Diagnostic::new("x".repeat(Diagnostic::MAX_BYTES - 1) + "é");

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
