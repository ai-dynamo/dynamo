// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! MaybeError trait for types that may contain error information.
//!
//! This module provides the `MaybeError` trait which allows types to represent
//! either successful data or error states. It integrates with the `DynamoError`
//! system to provide structured error handling.

use crate::error::DynamoError;

/// A trait for types that may contain error information.
///
/// This trait allows a type to represent either a successful value or an error state.
/// It integrates with `DynamoError` for structured error information.
///
/// # Example
///
/// ```rust,ignore
/// use dynamo_runtime::protocols::maybe_error::MaybeError;
/// use dynamo_runtime::error::DynamoError;
///
/// struct MyResponse {
///     data: Option<String>,
///     error: Option<DynamoError>,
/// }
///
/// impl MaybeError for MyResponse {
///     fn from_err(err: impl std::error::Error + 'static) -> Self {
///         MyResponse {
///             data: None,
///             error: Some(DynamoError::from(
///                 Box::new(err) as Box<dyn std::error::Error + 'static>
///             )),
///         }
///     }
///
///     fn err(&self) -> Option<DynamoError> {
///         self.error.clone()
///     }
/// }
/// ```
pub trait MaybeError {
    /// Construct an instance from an error.
    ///
    /// The error is converted to a `DynamoError` for serialization.
    fn from_err(err: impl std::error::Error + 'static) -> Self;

    /// Get the error as a `DynamoError` if this represents an error state.
    ///
    /// Returns `Some(DynamoError)` if this instance represents an error, `None` otherwise.
    fn err(&self) -> Option<DynamoError>;

    /// Check if the current instance represents a success.
    fn is_ok(&self) -> bool {
        !self.is_err()
    }

    /// Check if the current instance represents an error.
    fn is_err(&self) -> bool {
        self.err().is_some()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    struct TestError {
        error: Option<DynamoError>,
    }

    impl MaybeError for TestError {
        fn from_err(err: impl std::error::Error + 'static) -> Self {
            TestError {
                error: Some(DynamoError::from(
                    Box::new(err) as Box<dyn std::error::Error + 'static>
                )),
            }
        }

        fn err(&self) -> Option<DynamoError> {
            self.error.clone()
        }
    }

    #[test]
    fn test_maybe_error_default_implementations() {
        let dynamo_err = DynamoError::msg("Test error");
        let err = TestError::from_err(dynamo_err);
        assert!(err.err().unwrap().to_string().contains("Test error"));
        assert!(!err.is_ok());
        assert!(err.is_err());
    }

    #[test]
    fn test_from_std_error() {
        let std_err = std::io::Error::other("io failure");
        let test_err = TestError::from_err(std_err);

        assert!(test_err.is_err());
        assert!(test_err.err().unwrap().to_string().contains("io failure"));
    }

    #[test]
    fn test_not_error() {
        let test = TestError { error: None };
        assert!(test.is_ok());
        assert!(!test.is_err());
        assert!(test.err().is_none());
    }
}
