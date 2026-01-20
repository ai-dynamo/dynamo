// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Configuration utilities and trait re-exports.
//!
//! This module provides utility functions for parsing configuration values
//! and re-exports the core configuration traits from the integrations module.

// ===== Environment Variable Utilities =====

/// Check if a string is truthy.
///
/// This will be used to evaluate environment variables or any other subjective
/// configuration parameters that can be set by the user that should be evaluated
/// as a boolean value.
///
/// Truthy values: "1", "true", "on", "yes" (case-insensitive)
///
/// Returns `false` for invalid values. Use [`parse_bool`] if you need to error on invalid values.
pub fn is_truthy(val: &str) -> bool {
    matches!(val.to_lowercase().as_str(), "1" | "true" | "on" | "yes")
}

/// Check if a string is falsey.
///
/// This will be used to evaluate environment variables or any other subjective
/// configuration parameters that can be set by the user that should be evaluated
/// as a boolean value (opposite of is_truthy).
///
/// Falsey values: "0", "false", "off", "no" (case-insensitive)
///
/// Returns `false` for invalid values. Use [`parse_bool`] if you need to error on invalid values.
pub fn is_falsey(val: &str) -> bool {
    matches!(val.to_lowercase().as_str(), "0" | "false" | "off" | "no")
}

/// Parse a string as a boolean value, returning an error if invalid.
///
/// This function strictly validates that the input is a valid boolean representation.
///
/// # Arguments
/// * `val` - The string value to parse
///
/// # Returns
/// * `Ok(true)` - For truthy values: "1", "true", "on", "yes" (case-insensitive)
/// * `Ok(false)` - For falsey values: "0", "false", "off", "no" (case-insensitive)
/// * `Err(_)` - For any other value
///
/// # Example
/// ```ignore
/// assert_eq!(parse_bool("true")?, true);
/// assert_eq!(parse_bool("0")?, false);
/// assert!(parse_bool("maybe").is_err());
/// ```
pub fn parse_bool(val: &str) -> anyhow::Result<bool> {
    if is_truthy(val) {
        Ok(true)
    } else if is_falsey(val) {
        Ok(false)
    } else {
        anyhow::bail!(
            "Invalid boolean value: '{}'. Expected one of: true/false, 1/0, on/off, yes/no",
            val
        )
    }
}

/// Check if an environment variable is truthy.
///
/// Returns `false` if the environment variable is not set or is invalid.
/// Use [`env_parse_bool`] if you need to distinguish between unset, valid, and invalid values.
pub fn env_is_truthy(env: &str) -> bool {
    match std::env::var(env) {
        Ok(val) => is_truthy(val.as_str()),
        Err(_) => false,
    }
}

/// Check if an environment variable is falsey.
///
/// Returns `false` if the environment variable is not set or is invalid.
/// Use [`env_parse_bool`] if you need to distinguish between unset, valid, and invalid values.
pub fn env_is_falsey(env: &str) -> bool {
    match std::env::var(env) {
        Ok(val) => is_falsey(val.as_str()),
        Err(_) => false,
    }
}

/// Parse an environment variable as a boolean, returning an error if invalid.
///
/// # Arguments
/// * `env` - The environment variable name
///
/// # Returns
/// * `Ok(Some(true))` - If the env var is set to a truthy value
/// * `Ok(Some(false))` - If the env var is set to a falsey value
/// * `Ok(None)` - If the env var is not set
/// * `Err(_)` - If the env var is set to an invalid value
///
/// # Example
/// ```ignore
/// match env_parse_bool("MY_FLAG")? {
///     Some(true) => println!("enabled"),
///     Some(false) => println!("disabled"),
///     None => println!("not configured"),
/// }
/// ```
pub fn env_parse_bool(env: &str) -> anyhow::Result<Option<bool>> {
    match std::env::var(env) {
        Ok(val) => parse_bool(&val).map(Some),
        Err(std::env::VarError::NotPresent) => Ok(None),
        Err(e) => anyhow::bail!("Failed to read environment variable {}: {}", env, e),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_is_truthy() {
        assert!(is_truthy("1"));
        assert!(is_truthy("true"));
        assert!(is_truthy("True"));
        assert!(is_truthy("TRUE"));
        assert!(is_truthy("on"));
        assert!(is_truthy("ON"));
        assert!(is_truthy("yes"));
        assert!(is_truthy("YES"));

        assert!(!is_truthy("0"));
        assert!(!is_truthy("false"));
        assert!(!is_truthy("off"));
        assert!(!is_truthy("no"));
        assert!(!is_truthy(""));
        assert!(!is_truthy("random"));
    }

    #[test]
    fn test_is_falsey() {
        assert!(is_falsey("0"));
        assert!(is_falsey("false"));
        assert!(is_falsey("False"));
        assert!(is_falsey("FALSE"));
        assert!(is_falsey("off"));
        assert!(is_falsey("OFF"));
        assert!(is_falsey("no"));
        assert!(is_falsey("NO"));

        assert!(!is_falsey("1"));
        assert!(!is_falsey("true"));
        assert!(!is_falsey("on"));
        assert!(!is_falsey("yes"));
        assert!(!is_falsey(""));
        assert!(!is_falsey("random"));
    }

    #[test]
    fn test_env_is_truthy_not_set() {
        // Test with a variable that definitely doesn't exist
        assert!(!env_is_truthy("DEFINITELY_NOT_SET_VAR_12345"));
    }

    #[test]
    fn test_env_is_falsey_not_set() {
        // Test with a variable that definitely doesn't exist
        assert!(!env_is_falsey("DEFINITELY_NOT_SET_VAR_12345"));
    }

    #[test]
    fn test_parse_bool() {
        // Truthy values
        assert!(parse_bool("1").unwrap());
        assert!(parse_bool("true").unwrap());
        assert!(parse_bool("TRUE").unwrap());
        assert!(parse_bool("on").unwrap());
        assert!(parse_bool("yes").unwrap());

        // Falsey values
        assert!(!parse_bool("0").unwrap());
        assert!(!parse_bool("false").unwrap());
        assert!(!parse_bool("FALSE").unwrap());
        assert!(!parse_bool("off").unwrap());
        assert!(!parse_bool("no").unwrap());

        // Invalid values
        assert!(parse_bool("").is_err());
        assert!(parse_bool("maybe").is_err());
        assert!(parse_bool("2").is_err());
        assert!(parse_bool("random").is_err());
    }

    #[test]
    fn test_env_parse_bool_not_set() {
        // Test with a variable that definitely doesn't exist
        assert_eq!(
            env_parse_bool("DEFINITELY_NOT_SET_VAR_12345").unwrap(),
            None
        );
    }
}
