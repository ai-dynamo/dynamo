// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use serde::{Deserialize, Serialize};
use std::sync::Arc;
use xxhash_rust::xxh3::xxh3_64;

use super::{bus, config};
use crate::protocols::openai::chat_completions::{
    NvCreateChatCompletionRequest, NvCreateChatCompletionResponse,
};

const SAMPLE_BUCKETS: u64 = 10_000;

#[derive(Serialize, Deserialize, Clone)]
pub struct AuditRecord {
    pub schema_version: u32,
    pub request_id: String,
    pub requested_streaming: bool,
    pub model: String,
    /// Logical deployment that produced this record (e.g.
    /// DynamoGraphDeployment name). Set from `DYN_AUDIT_DEPLOYMENT`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub deployment: Option<String>,
    /// UTC wall-clock unix-millis stamped at `AuditHandle::emit()`.
    #[serde(default)]
    pub emitted_at_unix_ms: i64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub request: Option<Arc<NvCreateChatCompletionRequest>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub response: Option<Arc<NvCreateChatCompletionResponse>>,
}

pub struct AuditHandle {
    requested_streaming: bool,
    request_id: String,
    model: String,
    deployment: Option<String>,
    req_full: Option<Arc<NvCreateChatCompletionRequest>>,
    resp_full: Option<Arc<NvCreateChatCompletionResponse>>,
}

impl AuditHandle {
    pub fn streaming(&self) -> bool {
        self.requested_streaming
    }

    pub fn set_request(&mut self, req: Arc<NvCreateChatCompletionRequest>) {
        self.req_full = Some(req);
    }
    pub fn set_response(&mut self, resp: Arc<NvCreateChatCompletionResponse>) {
        self.resp_full = Some(resp);
    }

    #[cfg(test)]
    pub(crate) fn deployment_for_test(&self) -> Option<&str> {
        self.deployment.as_deref()
    }

    /// Emit exactly once (publishes to the bus; sinks do I/O).
    pub fn emit(self) {
        let rec = AuditRecord {
            schema_version: 1,
            request_id: self.request_id,
            requested_streaming: self.requested_streaming,
            model: self.model,
            deployment: self.deployment,
            emitted_at_unix_ms: chrono::Utc::now().timestamp_millis(),
            request: self.req_full,
            response: self.resp_full,
        };
        bus::publish(rec);
    }
}

pub fn create_handle(req: &NvCreateChatCompletionRequest, request_id: &str) -> Option<AuditHandle> {
    let policy = config::policy();
    create_handle_with_config(
        req,
        request_id,
        policy.enabled,
        policy.force_logging,
        policy.sample_rate,
        policy.deployment.clone(),
    )
}

fn create_handle_with_config(
    req: &NvCreateChatCompletionRequest,
    request_id: &str,
    enabled: bool,
    force_logging: bool,
    sample_rate: f32,
    deployment: Option<String>,
) -> Option<AuditHandle> {
    if !enabled {
        return None;
    }
    let store_flag = req.inner.store.unwrap_or(false);
    let force = force_logging || store_flag;
    // If neither force_logging nor the request `store` flag is set, this
    // request is not eligible for capture at all.
    if !force {
        return None;
    }
    // Forced requests bypass head-based sampling so debugging /
    // compliance flags always land in the audit log.
    if !force_logging && sample_rate < 1.0 && !is_sampled(request_id, sample_rate) {
        return None;
    }
    let requested_streaming = req.inner.stream.unwrap_or(false);
    let model = req.inner.model.clone();

    Some(AuditHandle {
        requested_streaming,
        request_id: request_id.to_string(),
        model,
        deployment,
        req_full: None,
        resp_full: None,
    })
}

fn is_sampled(request_id: &str, sample_rate: f32) -> bool {
    if sample_rate <= 0.0 {
        return false;
    }
    if sample_rate >= 1.0 {
        return true;
    }
    let bucket = xxh3_64(request_id.as_bytes()) % SAMPLE_BUCKETS;
    let threshold = (sample_rate as f64 * SAMPLE_BUCKETS as f64).round() as u64;
    bucket < threshold
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;

    fn create_test_request(model: &str, store: bool) -> NvCreateChatCompletionRequest {
        let json = serde_json::json!({
            "model": model,
            "messages": [{"role": "user", "content": "test"}],
            "store": store
        });
        serde_json::from_value(json).expect("Failed to create test request")
    }

    fn create_test_request_with_nvext() -> NvCreateChatCompletionRequest {
        let json = serde_json::json!({
            "model": "test-model",
            "messages": [{"role": "user", "content": "test"}],
            "store": true,
            "nvext": {
                "agent_hints": {
                    "priority": 5
                }
            }
        });
        serde_json::from_value(json).expect("Failed to create test request")
    }

    fn create_test_response(content: &str) -> NvCreateChatCompletionResponse {
        let json = serde_json::json!({
            "id": "chatcmpl-test",
            "object": "chat.completion",
            "created": 1234567890,
            "model": "test-model",
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": content
                },
                "finish_reason": "stop"
            }]
        });
        serde_json::from_value(json).expect("Failed to create test response")
    }

    /// Test that DYN_AUDIT_FORCE_LOGGING=true bypasses store=false
    /// When force logging is enabled, audit handle should be created even when store=false
    #[test]
    fn test_force_logging_bypasses_store() {
        let request = create_test_request("test-model", false);
        let handle = create_handle_with_config(&request, "test-id", true, true, 1.0, None);

        assert!(
            handle.is_some(),
            "force logging should create a handle even with store=false"
        );
    }

    #[test]
    fn audit_record_serializes_nvext_and_response_content() {
        let record = AuditRecord {
            schema_version: 1,
            request_id: "req-123".to_string(),
            requested_streaming: true,
            model: "test-model".to_string(),
            deployment: None,
            emitted_at_unix_ms: 1_700_000_000_000,
            request: Some(Arc::new(create_test_request_with_nvext())),
            response: Some(Arc::new(create_test_response("final answer"))),
        };

        let value = serde_json::to_value(record).unwrap();

        assert_eq!(value["request"]["nvext"]["agent_hints"]["priority"], 5);
        assert_eq!(
            value["response"]["choices"][0]["message"]["content"],
            "final answer"
        );
    }

    #[test]
    fn deployment_field_appears_in_serialized_record_when_set() {
        let record = AuditRecord {
            schema_version: 1,
            request_id: "req-1".to_string(),
            requested_streaming: false,
            model: "m".to_string(),
            deployment: Some("dgd-prod".to_string()),
            emitted_at_unix_ms: 0,
            request: None,
            response: None,
        };
        let value = serde_json::to_value(record).unwrap();
        assert_eq!(value["deployment"], "dgd-prod");
    }

    #[test]
    fn deployment_field_omitted_when_none() {
        let record = AuditRecord {
            schema_version: 1,
            request_id: "req-1".to_string(),
            requested_streaming: false,
            model: "m".to_string(),
            deployment: None,
            emitted_at_unix_ms: 0,
            request: None,
            response: None,
        };
        let value = serde_json::to_value(record).unwrap();
        assert!(value.get("deployment").is_none());
    }

    #[test]
    fn sample_rate_zero_skips_unforced_request() {
        let request = create_test_request("m", true);
        let handle = create_handle_with_config(&request, "req-1", true, false, 0.0, None);
        assert!(
            handle.is_none(),
            "sample_rate=0.0 should skip even when store=true (force_logging=false)"
        );
    }

    #[test]
    fn force_logging_bypasses_zero_sample_rate() {
        let request = create_test_request("m", false);
        let handle = create_handle_with_config(&request, "req-1", true, true, 0.0, None);
        assert!(
            handle.is_some(),
            "force_logging must bypass sampling regardless of sample_rate"
        );
    }

    #[test]
    fn sample_rate_one_keeps_all_requests() {
        let request = create_test_request("m", true);
        for i in 0..32 {
            let id = format!("req-{i}");
            assert!(
                create_handle_with_config(&request, &id, true, false, 1.0, None).is_some(),
                "sample_rate=1.0 must capture every request (id={id})"
            );
        }
    }

    #[test]
    fn sample_rate_partial_is_deterministic_and_in_range() {
        let request = create_test_request("m", true);
        let mut kept = 0usize;
        let total = 1000;
        for i in 0..total {
            let id = format!("req-{i:06}");
            // Same input must produce same decision twice — deterministic.
            let first = create_handle_with_config(&request, &id, true, false, 0.1, None).is_some();
            let second = create_handle_with_config(&request, &id, true, false, 0.1, None).is_some();
            assert_eq!(first, second, "sampling must be deterministic for id={id}");
            if first {
                kept += 1;
            }
        }
        // Expect ~100; allow generous slack to keep the test stable.
        assert!(
            (40..=180).contains(&kept),
            "sample_rate=0.1 over {total} ids: expected ~100 kept, got {kept}"
        );
    }

    #[test]
    fn deployment_attaches_to_record_via_emit() {
        let request = create_test_request("m", true);
        let mut handle =
            create_handle_with_config(&request, "req-x", true, false, 1.0, Some("dgd-a".into()))
                .expect("handle");
        handle.set_request(Arc::new(request));
        // We can't easily intercept bus::publish in this unit test without
        // initializing the bus, so just verify the handle remembers the
        // deployment until emit. The exhaustive end-to-end check lives in
        // the integration tests.
        assert_eq!(handle.deployment_for_test(), Some("dgd-a"));
    }
}
