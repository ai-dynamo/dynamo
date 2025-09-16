// SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::error::Error;

pub trait MaybeError {
    /// Construct an instance from an error.
    fn from_err(err: Box<dyn Error + Send + Sync>) -> Self;

    /// Construct into an error instance.
    fn err(&self) -> Option<anyhow::Error>;

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
        message: String,
    }
    impl MaybeError for TestError {
        fn from_err(err: Box<dyn Error + Send + Sync>) -> Self {
            TestError {
                message: err.to_string(),
            }
        }
        fn err(&self) -> Option<anyhow::Error> {
            Some(anyhow::Error::msg(self.message.clone()))
        }
    }

    #[test]
    fn test_maybe_error_default_implementations() {
        let err = TestError::from_err(anyhow::Error::msg("Test error".to_string()).into());
        assert_eq!(format!("{}", err.err().unwrap()), "Test error");
        assert!(!err.is_ok());
        assert!(err.is_err());
    }
}
