// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

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
