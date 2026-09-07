// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Canonical truthy/falsy flag parsing for Dynamo.
//!
//! Single owner of the boolean vocabulary accepted from user-supplied
//! configuration (environment variables, HTTP headers, config values):
//! truthy = `1 | true | on | yes`, falsy = `0 | false | off | no` or empty,
//! case-insensitive, surrounding whitespace ignored.
//!
//! `dynamo_runtime::config` re-exports these helpers and is the canonical
//! import path for crates that already depend on `dynamo-runtime`. Crates that
//! cannot (e.g. `dynamo-memory`, `dynamo-kv-router`, `dynamo-mocker`) depend on
//! this crate directly. Do not hand-roll new bool parsers — a divergent
//! accepted set means `SOMEFLAG=on` works for one flag and silently not
//! another. `tests/no_bool_parse_forks.rs` greps the workspace for forks.

/// Check if a string is truthy: `1 | true | on | yes`, case-insensitive,
/// surrounding whitespace ignored. Everything else is not truthy.
pub fn is_truthy(val: &str) -> bool {
    matches!(
        val.trim().to_lowercase().as_str(),
        "1" | "true" | "on" | "yes"
    )
}

/// Check if a string is falsey: `0 | false | off | no` or empty,
/// case-insensitive, surrounding whitespace ignored.
pub fn is_falsey(val: &str) -> bool {
    matches!(
        val.trim().to_lowercase().as_str(),
        "" | "0" | "false" | "off" | "no"
    )
}

/// Parse a string as a boolean value, returning an error if it is neither
/// truthy nor falsey.
///
/// # Returns
/// * `Ok(true)` - for truthy values ([`is_truthy`])
/// * `Ok(false)` - for falsey values ([`is_falsey`]), including empty
/// * `Err(_)` - for any other value
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

/// Tri-state parse for call sites that preserve their own default unless the
/// value is a deliberate boolean choice.
///
/// Unlike [`parse_bool`], an empty (or whitespace-only) value yields `None` —
/// a variable declared without a value (common in Kubernetes manifests and
/// Docker Compose files) must not override a default — and so does any
/// unrecognized value.
///
/// # Returns
/// * `Some(true)` - for truthy values ([`is_truthy`])
/// * `Some(false)` - for `0 | false | off | no`
/// * `None` - for empty or unrecognized values
pub fn parse_bool_opt(val: &str) -> Option<bool> {
    if is_truthy(val) {
        Some(true)
    } else if val.trim().is_empty() {
        None
    } else if is_falsey(val) {
        Some(false)
    } else {
        None
    }
}

/// Check if an environment variable is set to a truthy value.
/// Unset (or non-unicode) variables are not truthy.
pub fn env_is_truthy(env: &str) -> bool {
    match std::env::var(env) {
        Ok(val) => is_truthy(val.as_str()),
        Err(_) => false,
    }
}

/// Check if an environment variable is set to a falsey value (including set
/// but empty). Unset (or non-unicode) variables are not falsey.
pub fn env_is_falsey(env: &str) -> bool {
    match std::env::var(env) {
        Ok(val) => is_falsey(val.as_str()),
        Err(_) => false,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const TRUTHY: &[&str] = &[
        "1", "true", "TRUE", "True", "on", "ON", "On", "yes", "YES", "Yes", " true ", "\tyes\n",
    ];
    const FALSEY: &[&str] = &[
        "0", "false", "FALSE", "False", "off", "OFF", "Off", "no", "NO", "No", "", "  ", " false ",
    ];
    const NEITHER: &[&str] = &[
        "2", "enabled", "disabled", "maybe", "y", "n", "t", "f", "-1",
    ];

    #[test]
    fn truthy_spellings() {
        for val in TRUTHY {
            assert!(is_truthy(val), "value={val:?}");
            assert!(!is_falsey(val), "value={val:?}");
            assert!(parse_bool(val).unwrap(), "value={val:?}");
        }
    }

    #[test]
    fn falsey_spellings() {
        for val in FALSEY {
            assert!(is_falsey(val), "value={val:?}");
            assert!(!is_truthy(val), "value={val:?}");
            assert!(!parse_bool(val).unwrap(), "value={val:?}");
        }
    }

    #[test]
    fn invalid_spellings() {
        for val in NEITHER {
            assert!(!is_truthy(val), "value={val:?}");
            assert!(!is_falsey(val), "value={val:?}");
            assert!(parse_bool(val).is_err(), "value={val:?}");
        }
    }

    #[test]
    fn parse_bool_opt_spellings() {
        for val in TRUTHY {
            assert_eq!(parse_bool_opt(val), Some(true), "value={val:?}");
        }
        for val in FALSEY {
            let expected = if val.trim().is_empty() {
                None // declared-but-empty is not a deliberate choice
            } else {
                Some(false)
            };
            assert_eq!(parse_bool_opt(val), expected, "value={val:?}");
        }
        for val in NEITHER {
            assert_eq!(parse_bool_opt(val), None, "value={val:?}");
        }
    }

    #[test]
    fn env_helpers() {
        // Each test uses its own variable name: tests run concurrently and
        // the process environment is shared.
        const UNSET: &str = "DYN_TRUTHY_TEST_UNSET";
        assert!(!env_is_truthy(UNSET));
        assert!(!env_is_falsey(UNSET));

        const SET: &str = "DYN_TRUTHY_TEST_SET";
        for (val, truthy, falsey) in [("on", true, false), ("off", false, true), ("", false, true)]
        {
            // SAFETY: single-threaded with respect to this variable name.
            unsafe { std::env::set_var(SET, val) };
            assert_eq!(env_is_truthy(SET), truthy, "value={val:?}");
            assert_eq!(env_is_falsey(SET), falsey, "value={val:?}");
        }
    }
}
