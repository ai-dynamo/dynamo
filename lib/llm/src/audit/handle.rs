// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use serde::{Deserialize, Serialize};
use std::sync::Arc;
use std::time::SystemTime;

use super::{bus, config};
use crate::protocols::openai::chat_completions::{
    NvCreateChatCompletionRequest, NvCreateChatCompletionResponse,
};

/// One combined audit record per chat completion: the request, plus the response
/// when it completed (`response = None` on client cancel / timeout / aggregation
/// failure, so those stay auditable).
#[derive(Serialize, Deserialize, Clone)]
pub struct AuditRecord {
    pub schema_version: u32,
    pub request_id: String,
    pub requested_streaming: bool,
    pub model: String,
    /// Request arrival time, captured at handle creation. Used as the OTLP
    /// `LogRecord` Timestamp; `#[serde(skip)]` since it is record metadata, not
    /// audited payload, and only `OtelSink` reads it.
    #[serde(skip, default = "std::time::SystemTime::now")]
    pub event_time: SystemTime,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub request: Option<Arc<NvCreateChatCompletionRequest>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub response: Option<Arc<NvCreateChatCompletionResponse>>,
}

pub struct AuditHandle {
    requested_streaming: bool,
    request_id: String,
    model: String,
    event_time: SystemTime,
    request: Arc<NvCreateChatCompletionRequest>,
}

impl AuditHandle {
    pub fn streaming(&self) -> bool {
        self.requested_streaming
    }

    pub fn request_id(&self) -> &str {
        &self.request_id
    }

    /// Publish the combined audit record on the bus. Consumes the handle to
    /// enforce exactly one record per request. `response` is `None` on client
    /// cancel / gateway timeout / aggregation failure — the record still
    /// carries the request so those cases remain auditable.
    pub fn emit(self, response: Option<Arc<NvCreateChatCompletionResponse>>) {
        let rec = AuditRecord {
            schema_version: 1,
            request_id: self.request_id,
            requested_streaming: self.requested_streaming,
            model: self.model,
            event_time: self.event_time,
            request: Some(self.request),
            response,
        };
        bus::publish(rec);
    }
}

pub fn create_handle(req: &NvCreateChatCompletionRequest, request_id: &str) -> Option<AuditHandle> {
    let policy = config::policy();
    // `capture_enabled()` is `policy.enabled && CAPTURE_ACTIVE`: it additionally
    // requires the audit subsystem to have been initialized, so a stale handle
    // can't be created before/after the audit lifecycle.
    create_handle_with_config(req, request_id, config::capture_enabled(), policy.force_logging)
}

fn create_handle_with_config(
    req: &NvCreateChatCompletionRequest,
    request_id: &str,
    enabled: bool,
    force_logging: bool,
) -> Option<AuditHandle> {
    if !enabled {
        return None;
    }
    // If force_logging is enabled, ignore the store flag
    if !force_logging && !req.inner.store.unwrap_or(false) {
        return None;
    }
    let requested_streaming = req.inner.stream.unwrap_or(false);
    let model = req.inner.model.clone();

    Some(AuditHandle {
        requested_streaming,
        request_id: request_id.to_string(),
        model,
        // Snapshot the pristine inbound request (before the preprocessor
        // overrides stream/usage) and stamp arrival time on the producing
        // thread, so the record reflects what the client sent and when.
        event_time: SystemTime::now(),
        request: Arc::new(req.clone()),
    })
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

    struct AuditPolicyResetGuard;

    impl Drop for AuditPolicyResetGuard {
        fn drop(&mut self) {
            crate::audit::config::clear_policy_override_for_test();
        }
    }

    /// Test that DYN_AUDIT_FORCE_LOGGING=true bypasses store=false
    /// When force logging is enabled, audit handle should be created even when store=false
    #[test]
    #[serial_test::serial]
    fn test_force_logging_bypasses_store() {
        temp_env::with_vars(
            [
                ("DYN_AUDIT_SINKS", Some("stderr")),
                ("DYN_AUDIT_FORCE_LOGGING", Some("true")),
            ],
            || {
                crate::audit::config::override_policy_from_env_for_test();
                crate::audit::config::mark_capture_active();
                let _reset_guard = AuditPolicyResetGuard;

                let request = create_test_request("test-model", false);
                let handle = create_handle(&request, "test-id");

                assert!(
                    handle.is_some(),
                    "When DYN_AUDIT_FORCE_LOGGING=true, handle should be created even with store=false"
                );
            },
        );
    }

    #[test]
    fn audit_record_serializes_nvext_and_response_content() {
        let record = AuditRecord {
            schema_version: 1,
            request_id: "req-123".to_string(),
            requested_streaming: true,
            model: "test-model".to_string(),
            event_time: SystemTime::now(),
            request: Some(Arc::new(create_test_request_with_nvext())),
            response: Some(Arc::new(create_test_response("final answer"))),
        };

        let value = serde_json::to_value(record).unwrap();

        assert_eq!(value["request"]["nvext"]["agent_hints"]["priority"], 5);
        assert_eq!(
            value["response"]["choices"][0]["message"]["content"],
            "final answer"
        );
        // No event_type discriminator anymore.
        assert!(value.get("event_type").is_none());
    }

    #[test]
    fn audit_record_omits_response_when_absent() {
        // Cancel/timeout case: the record still carries the request, response
        // is omitted via skip_serializing_if.
        let record = AuditRecord {
            schema_version: 1,
            request_id: "req-456".to_string(),
            requested_streaming: false,
            model: "test-model".to_string(),
            event_time: SystemTime::now(),
            request: Some(Arc::new(create_test_request("test-model", true))),
            response: None,
        };

        let value = serde_json::to_value(&record).unwrap();
        assert!(value["request"].is_object());
        assert!(value.get("response").is_none());
    }

    /// Test-only constructor. `create_handle` gates on env vars + a cached
    /// `OnceLock` policy, which is too brittle for a focused bus-roundtrip test.
    impl AuditHandle {
        pub(crate) fn for_test(request_id: &str, model: &str, streaming: bool) -> Self {
            Self {
                requested_streaming: streaming,
                request_id: request_id.to_string(),
                model: model.to_string(),
                event_time: SystemTime::now(),
                request: Arc::new(create_test_request(model, true)),
            }
        }
    }

    #[tokio::test]
    #[serial_test::serial]
    async fn emit_publishes_one_combined_record_with_and_without_response() {
        // Exercises the contract: `emit` publishes exactly one record. With a
        // response present the record carries both request and response; with
        // `None` (cancel/timeout) it carries the request only.
        bus::init(8);
        let mut rx = bus::subscribe();

        AuditHandle::for_test("req-ok", "test-model", true)
            .emit(Some(Arc::new(create_test_response("hello"))));
        AuditHandle::for_test("req-cancel", "test-model", true).emit(None);

        let first = tokio::time::timeout(std::time::Duration::from_secs(1), rx.recv())
            .await
            .expect("first record arrives before timeout")
            .expect("first record receives ok");
        let second = tokio::time::timeout(std::time::Duration::from_secs(1), rx.recv())
            .await
            .expect("second record arrives before timeout")
            .expect("second record receives ok");

        assert_eq!(first.request_id, "req-ok");
        assert!(first.request.is_some());
        assert!(first.response.is_some());

        assert_eq!(second.request_id, "req-cancel");
        assert!(second.request.is_some());
        assert!(second.response.is_none());
    }
}
