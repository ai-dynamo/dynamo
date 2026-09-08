// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Shared support for lazily resolved scalar environment configuration.

use std::{env::VarError, str::FromStr};

/// Parse an environment variable or return `default` when it is unset or invalid.
#[doc(hidden)]
pub fn parse_or_default<T>(name: &str, default: T) -> T
where
    T: FromStr,
    T::Err: std::fmt::Display,
{
    match std::env::var(name) {
        Ok(value) => match value.parse() {
            Ok(parsed) => parsed,
            Err(error) => {
                tracing::warn!(
                    env_var = name,
                    value,
                    %error,
                    "invalid environment variable, using default"
                );
                default
            }
        },
        Err(VarError::NotPresent) => default,
        Err(VarError::NotUnicode(value)) => {
            tracing::warn!(
                env_var = name,
                ?value,
                "environment variable is not valid Unicode, using default"
            );
            default
        }
    }
}

/// Declare a function that parses and caches one scalar environment variable.
#[macro_export]
macro_rules! env_config {
    (
        $(#[$attr:meta])*
        $visibility:vis fn $name:ident() -> $value_type:ty =
            $env_var:path, default = $default:expr;
    ) => {
        $(#[$attr])*
        $visibility fn $name() -> $value_type {
            static VALUE: ::std::sync::OnceLock<$value_type> = ::std::sync::OnceLock::new();
            VALUE.get_or_init(|| {
                $crate::config::env_config::parse_or_default($env_var, $default)
            }).clone()
        }
    };
}

#[cfg(test)]
mod tests {
    const TEST_ENV: &str = "DYN_TEST_ENV_CONFIG_PARSE_OR_DEFAULT";
    const INVALID_TEST_ENV: &str = "DYN_TEST_ENV_CONFIG_INVALID";

    crate::env_config! {
        fn configured_value() -> usize = TEST_ENV, default = 17;
    }

    crate::env_config! {
        fn configured_string() -> String = TEST_ENV, default = "fallback".to_string();
    }

    #[test]
    fn uses_parsed_environment_value() {
        temp_env::with_var(TEST_ENV, Some("42"), || {
            assert_eq!(configured_value(), 42);
        });
    }

    #[test]
    fn invalid_value_uses_default() {
        temp_env::with_var(INVALID_TEST_ENV, Some("invalid"), || {
            assert_eq!(super::parse_or_default::<usize>(INVALID_TEST_ENV, 17), 17);
        });
    }

    #[test]
    fn supports_non_copy_values() {
        temp_env::with_var(TEST_ENV, Some("configured"), || {
            assert_eq!(configured_string(), "configured");
        });
    }
}
