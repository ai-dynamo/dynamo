// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use dynamo_runtime::config::environment_names::llm::DYN_HTTP_BODY_LIMIT_MB;

// Default axum max body limit without configuring is 2MB: https://docs.rs/axum/latest/axum/extract/struct.DefaultBodyLimit.html
/// Default body limit in bytes (45MB) to support 500k+ token payloads.
/// Can be configured at runtime using the DYN_HTTP_BODY_LIMIT_MB environment variable.
pub fn get_body_limit() -> usize {
    std::env::var(DYN_HTTP_BODY_LIMIT_MB)
        .ok()
        .and_then(|s| s.parse::<usize>().ok())
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
    fn test_is_json_content_type() {
        assert!(is_json_content_type("application/json"));
        assert!(is_json_content_type("application/json; charset=utf-8"));
        assert!(is_json_content_type("Application/JSON"));
        assert!(is_json_content_type("application/vnd.dynamo+json"));
        assert!(!is_json_content_type("text/plain"));
        assert!(!is_json_content_type("application/json-patch"));
        assert!(!is_json_content_type("application"));
    }
}
