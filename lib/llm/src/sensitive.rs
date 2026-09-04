// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

const SENSITIVE_HEADER_NAMES: &[&str] = &[
    "authorization",
    "proxy-authorization",
    "cookie",
    "set-cookie",
    "x-api-key",
    "api-key",
    "x-auth-token",
    "x-access-token",
];

const CREDENTIAL_SCHEMES: &[&str] = &["basic", "bearer"];

pub(crate) fn is_sensitive_header(name: &str, value: &str) -> bool {
    SENSITIVE_HEADER_NAMES
        .iter()
        .any(|sensitive_name| name.eq_ignore_ascii_case(sensitive_name))
        || value
            .trim_start()
            .split_once(char::is_whitespace)
            .is_some_and(|(scheme, _)| {
                CREDENTIAL_SCHEMES
                    .iter()
                    .any(|credential_scheme| scheme.eq_ignore_ascii_case(credential_scheme))
            })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn identifies_sensitive_header_names_case_insensitively() {
        for name in SENSITIVE_HEADER_NAMES {
            assert!(is_sensitive_header(name, "value"));
            assert!(is_sensitive_header(&name.to_ascii_uppercase(), "value"));
        }
    }

    #[test]
    fn identifies_credential_schemes_case_insensitively_after_whitespace() {
        for scheme in ["bAsIc", "bEaReR"] {
            assert!(is_sensitive_header(
                "x-custom",
                &format!("  {scheme} secret")
            ));
        }
    }

    #[test]
    fn allows_non_sensitive_headers() {
        assert!(!is_sensitive_header("x-request-id", "request-123"));
    }
}
