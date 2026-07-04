// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/// HTTP body size limit environment variable, in MiB.
pub const DYN_HTTP_BODY_LIMIT_MB: &str = "DYN_HTTP_BODY_LIMIT_MB";

/// Default body limit in bytes (45 MiB) to support 500k+ token payloads.
pub fn get_body_limit() -> usize {
    std::env::var(DYN_HTTP_BODY_LIMIT_MB)
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .map(|mb| mb * 1024 * 1024)
        .unwrap_or(45 * 1024 * 1024)
}

pub fn is_json_content_type(content_type: &str) -> bool {
    let media_type = content_type.split(';').next().unwrap_or_default().trim();
    let Some((media_type, subtype)) = media_type.split_once('/') else {
        return false;
    };

    media_type.eq_ignore_ascii_case("application")
        && (subtype.eq_ignore_ascii_case("json")
            || subtype
                .to_ascii_lowercase()
                .rsplit_once('+')
                .is_some_and(|(_, suffix)| suffix == "json"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn recognizes_json_media_types() {
        assert!(is_json_content_type("application/json"));
        assert!(is_json_content_type("Application/JSON; charset=utf-8"));
        assert!(is_json_content_type("application/problem+json"));
        assert!(!is_json_content_type("text/json"));
        assert!(!is_json_content_type("application/jsonp"));
    }
}
