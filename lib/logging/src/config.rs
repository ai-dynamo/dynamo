// SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

//! Configuration functions for the logging module.

/// Check if an environment variable is truthy
pub fn env_is_truthy(env: &str) -> bool {
    match std::env::var(env) {
        Ok(val) => is_truthy(val.as_str()),
        Err(_) => false,
    }
}

/// Check if a string is truthy
/// This will be used to evaluate environment variables or any other subjective
/// configuration parameters that can be set by the user that should be evaluated
/// as a boolean value.
pub fn is_truthy(val: &str) -> bool {
    matches!(val.to_lowercase().as_str(), "1" | "true" | "on" | "yes")
}

/// Check whether JSONL logging enabled
/// Set the `DYN_LOGGING_JSONL` environment variable to a [`is_truthy`] value
pub fn jsonl_logging_enabled() -> bool {
    env_is_truthy("DYN_LOGGING_JSONL")
}

/// Check whether logging with ANSI terminal escape codes and colors is disabled.
/// Set the `DYN_SDK_DISABLE_ANSI_LOGGING` environment variable to a [`is_truthy`] value
pub fn disable_ansi_logging() -> bool {
    env_is_truthy("DYN_SDK_DISABLE_ANSI_LOGGING")
}

/// Check whether to use local timezone for logging timestamps (default is UTC)
/// Set the `DYN_LOG_USE_LOCAL_TZ` environment variable to a [`is_truthy`] value
pub fn use_local_timezone() -> bool {
    env_is_truthy("DYN_LOG_USE_LOCAL_TZ")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_is_truthy() {
        assert!(is_truthy("1"));
        assert!(is_truthy("true"));
        assert!(is_truthy("TRUE"));
        assert!(is_truthy("True"));
        assert!(is_truthy("on"));
        assert!(is_truthy("ON"));
        assert!(is_truthy("yes"));
        assert!(is_truthy("YES"));

        assert!(!is_truthy("0"));
        assert!(!is_truthy("false"));
        assert!(!is_truthy("FALSE"));
        assert!(!is_truthy("off"));
        assert!(!is_truthy("OFF"));
        assert!(!is_truthy("no"));
        assert!(!is_truthy("NO"));
        assert!(!is_truthy(""));
    }
}
