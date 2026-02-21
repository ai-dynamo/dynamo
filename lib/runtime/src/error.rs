// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Dynamo Error System
//!
//! This module provides a standardized error type for Dynamo with support for:
//! - Categorized error types via [`ErrorType`] enum
//! - Error chaining via the standard [`std::error::Error::source()`] method
//! - Serialization for network transmission via serde
//!
//! # DynamoError
//!
//! [`DynamoError`] is the standardized error type for Dynamo. It can be created
//! directly or converted from any [`std::error::Error`]:
//!
//! ```rust,ignore
//! use dynamo_runtime::error::{DynamoError, ErrorType};
//!
//! // Simple error
//! let err = DynamoError::msg("something failed");
//!
//! // Typed error with cause
//! let cause = std::io::Error::other("io error");
//! let err = DynamoError::new(ErrorType::Unknown, "operation failed", Some(cause));
//!
//! // Convert from any std::error::Error
//! let std_err = std::io::Error::other("io error");
//! let dynamo_err = DynamoError::from(Box::new(std_err) as Box<dyn std::error::Error>);
//! ```

use serde::{Deserialize, Serialize};
use std::fmt;

// ============================================================================
// ErrorType Enum
// ============================================================================

/// Categorizes errors into a fixed set of standard types.
///
/// Consumers (e.g., the migration module) inspect the error type to decide
/// what action to take, rather than the error defining its own behavior.
/// Backend engine error subcategories.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum BackendError {
    /// The engine process has shut down or crashed.
    EngineShutdown,
}

impl fmt::Display for BackendError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            BackendError::EngineShutdown => write!(f, "EngineShutdown"),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ErrorType {
    /// Uncategorized or unknown error.
    Unknown,
    /// Failed to establish a connection to a remote worker.
    CannotConnect,
    /// An established connection was lost unexpectedly.
    Disconnected,
    /// A connection or request timed out.
    ConnectionTimeout,
    /// Error originating from a backend engine.
    Backend(BackendError),
}

impl fmt::Display for ErrorType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ErrorType::Unknown => write!(f, "Unknown"),
            ErrorType::CannotConnect => write!(f, "CannotConnect"),
            ErrorType::Disconnected => write!(f, "Disconnected"),
            ErrorType::ConnectionTimeout => write!(f, "ConnectionTimeout"),
            ErrorType::Backend(sub) => write!(f, "Backend.{sub}"),
        }
    }
}

/// Walk an error chain and check if any `DynamoError` in it matches the given set.
///
/// Returns `true` if any error in the chain (via `source()`) can be downcast to
/// `DynamoError` and its `ErrorType` is in `match_set`, unless an error in
/// `exclude_set` is also found (in which case returns `false` immediately).
pub fn chain_contains(
    err: &(dyn std::error::Error + 'static),
    match_set: &[ErrorType],
    exclude_set: &[ErrorType],
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
// DynamoError - The Standardized Error Type
// ============================================================================

/// The standardized error type for Dynamo.
///
/// `DynamoError` is a serializable, chainable error that:
/// - Carries an [`ErrorType`] for categorization
/// - Supports error chaining via [`std::error::Error::source()`]
/// - Is serializable for network transmission via `Annotated`
/// - Can be created from any [`std::error::Error`]
///
/// # Display
///
/// `Display` shows only the current error (standard Rust convention).
/// Use `source()` to walk the cause chain:
///
/// ```rust,ignore
/// let err = DynamoError::msg("outer");
/// println!("{}", err); // "Unknown: outer"
/// ```
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DynamoError {
    error_type: ErrorType,
    message: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    caused_by: Option<Box<DynamoError>>,
}

impl DynamoError {
    /// Create a new DynamoError with the given type, message, and optional cause.
    ///
    /// If the cause is a `DynamoError`, it is preserved as-is. Otherwise, it is
    /// converted to a `DynamoError` with `ErrorType::Unknown`.
    pub fn new(
        error_type: ErrorType,
        message: impl Into<String>,
        cause: Option<impl std::error::Error + 'static>,
    ) -> Self {
        Self {
            error_type,
            message: message.into(),
            caused_by: cause
                .map(|e| Box::new(DynamoError::from(&e as &(dyn std::error::Error + 'static)))),
        }
    }

    /// Shorthand to create an `Unknown` error with just a message and no cause.
    pub fn msg(message: impl Into<String>) -> Self {
        Self {
            error_type: ErrorType::Unknown,
            message: message.into(),
            caused_by: None,
        }
    }

    /// Returns the error type.
    pub fn error_type(&self) -> ErrorType {
        self.error_type
    }

    /// Returns the error message.
    pub fn message(&self) -> &str {
        &self.message
    }
}

impl fmt::Display for DynamoError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}: {}", self.error_type, self.message)
    }
}

impl std::error::Error for DynamoError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        self.caused_by
            .as_deref()
            .map(|e| e as &(dyn std::error::Error + 'static))
    }
}

/// Convert from a reference to any `std::error::Error`.
///
/// If the error is already a `DynamoError`, it is cloned. Otherwise, it is
/// wrapped as `ErrorType::Unknown` with the display string as the message.
/// The source chain is recursively converted, preserving `DynamoError` instances.
impl<'a> From<&'a (dyn std::error::Error + 'static)> for DynamoError {
    fn from(err: &'a (dyn std::error::Error + 'static)) -> Self {
        if let Some(dynamo_err) = err.downcast_ref::<DynamoError>() {
            return dynamo_err.clone();
        }

        Self {
            error_type: ErrorType::Unknown,
            message: err.to_string(),
            caused_by: err.source().map(|s| Box::new(DynamoError::from(s))),
        }
    }
}

/// Convert from an owned boxed `std::error::Error`.
///
/// If the error is already a `DynamoError`, ownership is taken without cloning.
/// Otherwise, falls back to the reference-based conversion.
impl From<Box<dyn std::error::Error + 'static>> for DynamoError {
    fn from(err: Box<dyn std::error::Error + 'static>) -> Self {
        match err.downcast::<DynamoError>() {
            Ok(dynamo_err) => *dynamo_err,
            Err(err) => DynamoError::from(&*err as &(dyn std::error::Error + 'static)),
        }
    }
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
        assert_eq!(err.error_type(), ErrorType::Unknown);
        assert_eq!(err.message(), "something failed");
        assert!(err.source().is_none());
    }

    #[test]
    fn test_new_constructor_with_cause() {
        let cause = std::io::Error::other("io error");
        let err = DynamoError::new(ErrorType::Unknown, "operation failed", Some(cause));

        assert_eq!(err.error_type(), ErrorType::Unknown);
        assert_eq!(err.message(), "operation failed");
        assert!(err.source().is_some());
    }

    #[test]
    fn test_display_shows_only_current_error() {
        let cause = std::io::Error::other("io error");
        let err = DynamoError::new(ErrorType::Unknown, "operation failed", Some(cause));

        // Display should only show the current error, not the chain
        assert_eq!(err.to_string(), "Unknown: operation failed");
    }

    #[test]
    fn test_source_chain() {
        let cause = std::io::Error::other("io error");
        let err = DynamoError::new(ErrorType::Unknown, "operation failed", Some(cause));

        // source() should return the cause
        let source = err.source().unwrap();
        assert!(source.to_string().contains("io error"));
    }

    #[test]
    fn test_from_boxed_std_error() {
        let std_err = std::io::Error::other("io error");
        let boxed: Box<dyn std::error::Error> = Box::new(std_err);
        let dynamo_err = DynamoError::from(boxed);

        assert_eq!(dynamo_err.error_type(), ErrorType::Unknown);
        assert_eq!(dynamo_err.message(), "io error");
    }

    #[test]
    fn test_from_boxed_takes_ownership_of_dynamo_error() {
        let inner = DynamoError::msg("original");
        let boxed: Box<dyn std::error::Error> = Box::new(inner);
        let dynamo_err = DynamoError::from(boxed);

        // Should take ownership, not clone or wrap
        assert_eq!(dynamo_err.error_type(), ErrorType::Unknown);
        assert_eq!(dynamo_err.message(), "original");
    }

    #[test]
    fn test_from_boxed_with_source_chain() {
        #[derive(Debug)]
        struct OuterError {
            source: std::io::Error,
        }

        impl fmt::Display for OuterError {
            fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                write!(f, "outer error occurred")
            }
        }

        impl std::error::Error for OuterError {
            fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
                Some(&self.source)
            }
        }

        let inner = std::io::Error::new(std::io::ErrorKind::NotFound, "file not found");
        let outer = OuterError { source: inner };
        let boxed: Box<dyn std::error::Error> = Box::new(outer);
        let dynamo_err = DynamoError::from(boxed);

        assert_eq!(dynamo_err.message(), "outer error occurred");
        assert!(dynamo_err.source().is_some());

        let cause = dynamo_err.source().unwrap();
        assert!(cause.to_string().contains("file not found"));
    }

    #[test]
    fn test_serialization_roundtrip() {
        let cause = DynamoError::msg("inner cause");
        let err = DynamoError::new(ErrorType::Unknown, "outer error", Some(cause));

        let json = serde_json::to_string(&err).unwrap();
        let deserialized: DynamoError = serde_json::from_str(&json).unwrap();

        assert_eq!(deserialized.error_type(), ErrorType::Unknown);
        assert_eq!(deserialized.message(), "outer error");
        assert!(deserialized.source().is_some());

        let cause = deserialized
            .source()
            .unwrap()
            .downcast_ref::<DynamoError>()
            .unwrap();
        assert_eq!(cause.message(), "inner cause");
    }

    #[test]
    fn test_error_type_display() {
        assert_eq!(ErrorType::Unknown.to_string(), "Unknown");
    }
}
