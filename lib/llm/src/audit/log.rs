// SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use serde::Serialize;
use tracing::info;

pub fn log_stored_completion<Resp: Serialize>(
    request_id: &str,
    request_json: &str,
    response: &Resp,
) {
    let resp_json = serde_json::to_string(response).unwrap_or_else(|_| "{}".into());
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
        request = request_json,
        response = %resp_json,
        "Audit log for stored completion"
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_log_stored_completion_doesnt_panic() {
        let test_response = serde_json::json!({
            "id": "test-123",
            "content": "Hello world"
        });

        // Should not panic
        log_stored_completion("req-123", r#"{"test": true}"#, &test_response);
    }
}
