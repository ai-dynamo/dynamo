// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Shared sensitivity detection for context metadata and captured HTTP headers.

/// Match caller-selected sensitive names or the frontend's Bearer value prefix.
/// Name policies may differ, but Bearer detection always applies.
pub(crate) fn is_sensitive_metadata(
    raw_key: &str,
    raw_value: &str,
    sensitive_names: &[impl AsRef<str>],
) -> bool {
    let value = raw_value.trim_start();
    sensitive_names
        .iter()
        .any(|name| raw_key.eq_ignore_ascii_case(name.as_ref()))
        || value
            .get(.."bearer ".len())
            .is_some_and(|prefix| prefix.eq_ignore_ascii_case("bearer "))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn detects_bearer_values_only() {
        for value in ["Bearer secret", " \t bEaReR secret", "bearer "] {
            assert!(is_sensitive_metadata("x-custom", value, &[] as &[&str]));
        }
        for value in [
            "",
            "bearer",
            "bearer-token",
            "Basic secret",
            "tenant",
            "éééé",
        ] {
            assert!(!is_sensitive_metadata("x-custom", value, &[] as &[&str]));
        }
    }

    #[test]
    fn respects_caller_name_policy_case_insensitively() {
        assert!(is_sensitive_metadata(
            "AUTHORIZATION",
            "Basic secret",
            &["authorization"]
        ));
        assert!(!is_sensitive_metadata(
            "authorization",
            "Basic secret",
            &[] as &[&str]
        ));
        assert!(is_sensitive_metadata(
            "X-Custom",
            "opaque",
            &[String::from("x-custom")]
        ));
    }
}
