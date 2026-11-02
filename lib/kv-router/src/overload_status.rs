// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! The HTTP status of an overload rejection. The frontend returns it for overload and
//! admission-control rejections, and the selection service and the endpoint picker return it
//! when a router class queue drops a request. It lives here, rather than in the frontend, so the
//! selection service can share it without depending on `dynamo-runtime`.

use std::sync::LazyLock;

/// Sets the overload status. `dynamo_runtime::config::environment_names::llm` documents the same
/// name; this crate cannot depend on `dynamo-runtime`, so it repeats it.
pub const DYN_HTTP_OVERLOAD_STATUS_CODE: &str = "DYN_HTTP_OVERLOAD_STATUS_CODE";

/// The overload status when none is configured: 529, "Site is overloaded".
pub const DEFAULT_OVERLOAD_STATUS_CODE: u16 = 529;

/// Parses a configured overload status. Any status from 200 through 999 is accepted; an
/// informational value from 100 through 199, an unparseable value, or an out-of-range value
/// falls back to 529.
pub fn parse_overload_status_code(value: Option<&str>) -> u16 {
    value
        .and_then(|value| value.trim().parse::<u16>().ok())
        .filter(|code| (200..=999).contains(code))
        .unwrap_or(DEFAULT_OVERLOAD_STATUS_CODE)
}

/// Overload / admission-control rejection status. Reads `DYN_HTTP_OVERLOAD_STATUS_CODE`
/// (default 529) on first use; cached since the environment is fixed at runtime and this is
/// on the rejection path.
pub fn overload_status_code() -> u16 {
    static CODE: LazyLock<u16> = LazyLock::new(|| {
        let value = std::env::var(DYN_HTTP_OVERLOAD_STATUS_CODE).ok();
        parse_overload_status_code(value.as_deref())
    });
    *CODE
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_configured_overload_status_code() {
        for value in [
            None,
            Some(""),
            Some("not-a-code"),
            Some("99"),
            Some("100"),
            Some("199"),
            Some("1000"),
        ] {
            assert_eq!(
                parse_overload_status_code(value),
                529,
                "expected {value:?} to fall back to 529"
            );
        }

        for value in [200_u16, 503, 529, 600, 999] {
            let configured = value.to_string();
            assert_eq!(
                parse_overload_status_code(Some(&configured)),
                value,
                "expected {value} to be preserved"
            );
        }
        assert_eq!(parse_overload_status_code(Some(" 503 ")), 503);
    }
}
