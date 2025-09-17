// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
use serde::Serialize;
use std::{
    collections::HashMap,
    sync::{Mutex, OnceLock},
};
use tracing::info;

static AUDIT_ENABLED: OnceLock<bool> = OnceLock::new();
static REQ_STASH: OnceLock<Mutex<HashMap<String, String>>> = OnceLock::new();

fn stash() -> &'static Mutex<HashMap<String, String>> {
    REQ_STASH.get_or_init(|| Mutex::new(HashMap::new()))
}

pub fn audit_enabled() -> bool {
    *AUDIT_ENABLED.get_or_init(|| {
        std::env::var("DYN_AUDIT_ENABLED")
            .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
            .unwrap_or(false)
    })
}

pub fn stash_request(id: impl Into<String>, req_json: String) {
    let _ = stash().lock().unwrap().insert(id.into(), req_json);
}

/// Remove and return the stashed request for this id (if present).
pub fn take_request(id: &str) -> Option<String> {
    stash().lock().unwrap().remove(id)
}

pub fn log_stored_completion<Resp: Serialize>(
    request_id: &str,
    request_json: &str,
    response: &Resp,
) {
    let resp_json = serde_json::to_string(response).unwrap_or_else(|_| "{}".to_string());
    let ts_ms = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_millis() as u64)
        .unwrap_or_default();

    info!(
        target: "dynamo_llm::audit",
        log_type = "audit",
        schema_version = 1.0,
        ts_ms = ts_ms,
        request_id = request_id,
        request = %request_json,
        response = %resp_json,
        "Audit log for stored completion"
    );
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn test_stash_and_take() {
        let id = "test-123";
        let req_json = r#"{"test": "request"}"#.to_string();

        // Stash a request
        stash_request(id, req_json.clone());

        // Take it back
        let retrieved = take_request(id);
        assert_eq!(retrieved, Some(req_json));

        // Should be gone now
        let retrieved_again = take_request(id);
        assert_eq!(retrieved_again, None);
    }

    #[test]
    fn test_log_function_doesnt_panic() {
        let test_response = json!({
            "id": "test-123",
            "model": "test-model",
            "choices": []
        });

        // Should not panic
        log_stored_completion("req-123", r#"{"test": "request"}"#, &test_response);
    }
}
