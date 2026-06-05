// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use serde::{Deserialize, Serialize};
use std::sync::Arc;

use axum::http::HeaderMap;

use super::{bus, config};
use crate::protocols::openai::chat_completions::{
    NvCreateChatCompletionRequest, NvCreateChatCompletionResponse,
};

pub const OTEL_HTTP_HEADERS_CONTEXT_KEY: &str = "audit.otel.http.request.headers";

#[derive(Clone)]
pub struct AuditHttpRequestHeaders {
    headers: Arc<HeaderMap>,
}

impl AuditHttpRequestHeaders {
    pub fn new(headers: Arc<HeaderMap>) -> Self {
        Self { headers }
    }

    pub fn headers(&self) -> &HeaderMap {
        self.headers.as_ref()
    }
}

/// Distinguishes the two record types emitted per chat completion.
///
/// Request and response are published as separate `AuditRecord`s sharing the same
/// `request_id`. Downstream consumers correlate by `request_id`; the request emit
/// is scheduled before the worker dispatches, the response record is emitted after
/// the response stream completes successfully. On client cancel mid-stream (or
/// aggregation failure) only the request record is emitted.
#[derive(Serialize, Deserialize, Clone, Copy, Debug, Eq, PartialEq)]
#[serde(rename_all = "lowercase")]
pub enum AuditEventType {
    Request,
    Response,
}

#[derive(Serialize, Deserialize, Clone)]
pub struct AuditRecord {
    pub schema_version: u32,
    pub event_type: AuditEventType,
    pub request_id: String,
    pub requested_streaming: bool,
    pub model: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub request: Option<Arc<NvCreateChatCompletionRequest>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub response: Option<Arc<NvCreateChatCompletionResponse>>,
    #[serde(skip)]
    pub otel_http_headers: Option<Arc<AuditHttpRequestHeaders>>,
}

#[derive(Clone)]
pub struct AuditHandle {
    requested_streaming: bool,
    request_id: String,
    model: String,
    otel_http_headers: Option<Arc<AuditHttpRequestHeaders>>,
}

impl AuditHandle {
    pub fn streaming(&self) -> bool {
        self.requested_streaming
    }

    pub fn request_id(&self) -> &str {
        &self.request_id
    }

    /// Publish a `Request` event record on the audit bus. Call once after the
    /// request is captured. The preprocessor schedules this before worker
    /// dispatch so downstream observers can see hung / canceled requests that
    /// never produce a response record.
    pub fn emit_request(&self, request: Arc<NvCreateChatCompletionRequest>) {
        let rec = AuditRecord {
            schema_version: 1,
            event_type: AuditEventType::Request,
            request_id: self.request_id.clone(),
            requested_streaming: self.requested_streaming,
            model: self.model.clone(),
            request: Some(request),
            response: None,
            otel_http_headers: self.otel_http_headers.clone(),
        };
        bus::publish(rec);
    }

    /// Publish a `Response` event record on the audit bus. Consumes the handle to
    /// enforce one-response-per-request at the type level. The response record does
    /// not duplicate the request payload — downstream JOINs on `request_id`.
    pub fn emit_response(self, response: Arc<NvCreateChatCompletionResponse>) {
        let rec = AuditRecord {
            schema_version: 1,
            event_type: AuditEventType::Response,
            request_id: self.request_id,
            requested_streaming: self.requested_streaming,
            model: self.model,
            request: None,
            response: Some(response),
            otel_http_headers: None,
        };
        bus::publish(rec);
    }
}

pub fn create_handle(
    req: &NvCreateChatCompletionRequest,
    request_id: &str,
    otel_http_headers: Option<Arc<AuditHttpRequestHeaders>>,
) -> Option<AuditHandle> {
    let policy = config::policy();
    // `capture_enabled()` is `policy.enabled && CAPTURE_ACTIVE`: it additionally
    // requires the audit subsystem to have been initialized, so a stale handle
    // can't be created before/after the audit lifecycle.
    create_handle_with_config(
        req,
        request_id,
        config::capture_enabled(),
        policy.force_logging,
        otel_http_headers,
    )
}

fn create_handle_with_config(
    req: &NvCreateChatCompletionRequest,
    request_id: &str,
    enabled: bool,
    force_logging: bool,
    otel_http_headers: Option<Arc<AuditHttpRequestHeaders>>,
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
        otel_http_headers,
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

    fn create_test_request_with_agent_context() -> NvCreateChatCompletionRequest {
        let json = serde_json::json!({
            "model": "test-model",
            "messages": [{"role": "user", "content": "test"}],
            "store": true,
            "nvext": {
                "agent_context": {
                    "session_type_id": "deep_research",
                    "session_id": "run-123",
                    "trajectory_id": "run-123:researcher",
                    "parent_trajectory_id": "run-123:planner"
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
                let handle = create_handle(&request, "test-id", None);

                assert!(
                    handle.is_some(),
                    "When DYN_AUDIT_FORCE_LOGGING=true, handle should be created even with store=false"
                );
            },
        );
    }

    #[test]
    fn request_record_carries_request_only() {
        let record = AuditRecord {
            schema_version: 1,
            event_type: AuditEventType::Request,
            request_id: "req-123".to_string(),
            requested_streaming: true,
            model: "test-model".to_string(),
            request: Some(Arc::new(create_test_request_with_agent_context())),
            response: None,
            otel_http_headers: None,
        };

        let value = serde_json::to_value(&record).unwrap();
        assert_eq!(value["event_type"], "request");
        assert_eq!(value["request_id"], "req-123");
        assert_eq!(
            value["request"]["nvext"]["agent_context"]["session_id"],
            "run-123"
        );
        // Response side must be absent (skip_serializing_if).
        assert!(value.get("response").is_none());
    }

    #[test]
    fn response_record_carries_response_only() {
        let record = AuditRecord {
            schema_version: 1,
            event_type: AuditEventType::Response,
            request_id: "req-123".to_string(),
            requested_streaming: true,
            model: "test-model".to_string(),
            request: None,
            response: Some(Arc::new(create_test_response("final answer"))),
            otel_http_headers: None,
        };

        let value = serde_json::to_value(&record).unwrap();
        assert_eq!(value["event_type"], "response");
        assert_eq!(value["request_id"], "req-123");
        assert_eq!(
            value["response"]["choices"][0]["message"]["content"],
            "final answer"
        );
        // Request side must be absent (skip_serializing_if). Downstream JOINs on request_id.
        assert!(value.get("request").is_none());
    }

    /// Test-only constructor. `create_handle` gates on env vars + a cached
    /// `OnceLock` policy, which is too brittle for a focused bus-roundtrip test.
    impl AuditHandle {
        pub(crate) fn for_test(request_id: &str, model: &str, streaming: bool) -> Self {
            Self {
                requested_streaming: streaming,
                request_id: request_id.to_string(),
                model: model.to_string(),
                otel_http_headers: None,
            }
        }
    }

    #[tokio::test]
    #[serial_test::serial]
    async fn emit_request_and_response_publish_two_records_in_order() {
        // Exercises the end-to-end contract: emit_request + emit_response each
        // publish exactly one record to the audit bus, in that order, with the
        // expected `event_type` discriminant and request/response payload
        // exclusivity (request record has no response, and vice versa).
        bus::init(8);
        let mut rx = bus::subscribe();

        let request = create_test_request("test-model", true);
        let response = create_test_response("hello");

        let handle = AuditHandle::for_test("req-test-emit", "test-model", true);
        handle.emit_request(Arc::new(request));
        handle.emit_response(Arc::new(response));

        let first = tokio::time::timeout(std::time::Duration::from_secs(1), rx.recv())
            .await
            .expect("first record arrives before timeout")
            .expect("first record receives ok");
        let second = tokio::time::timeout(std::time::Duration::from_secs(1), rx.recv())
            .await
            .expect("second record arrives before timeout")
            .expect("second record receives ok");

        assert_eq!(first.event_type, AuditEventType::Request);
        assert_eq!(first.request_id, "req-test-emit");
        assert!(first.request.is_some());
        assert!(first.response.is_none());

        assert_eq!(second.event_type, AuditEventType::Response);
        assert_eq!(second.request_id, "req-test-emit");
        assert!(second.request.is_none());
        assert!(second.response.is_some());
    }
}
