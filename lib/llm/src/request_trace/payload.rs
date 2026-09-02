// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::collections::BTreeMap;
use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};

use axum::http::HeaderMap;

use crate::protocols::openai::chat_completions::{
    NvCreateChatCompletionRequest, NvCreateChatCompletionResponse,
};

/// Context key for the allowlisted headers captured at the HTTP layer.
pub const HTTP_HEADERS_CONTEXT_KEY: &str = "request_trace.http.request.headers";

/// True when payload records are being captured and a header allowlist is set.
pub(crate) fn http_header_capture_active() -> bool {
    if !super::config::capture_enabled() {
        return false;
    }
    let policy = super::policy();
    policy.emit_request_payload_records() && !policy.http_header_capture_list.is_empty()
}

/// Collect the allowlisted request headers (case-insensitive, comma-joined on
/// repeats). `None` unless payload capture is active and the allowlist is non-empty.
pub fn capture_http_headers(headers: &HeaderMap) -> Option<BTreeMap<String, String>> {
    if !http_header_capture_active() {
        return None;
    }
    capture_http_headers_with_list(headers, &super::policy().http_header_capture_list)
}

fn capture_http_headers_with_list(
    headers: &HeaderMap,
    capture_list: &[String],
) -> Option<BTreeMap<String, String>> {
    if capture_list.is_empty() {
        return None;
    }
    let mut out = BTreeMap::new();
    for name in capture_list {
        let joined = headers
            .get_all(name.as_str())
            .iter()
            .filter_map(|value| value.to_str().ok())
            .filter(|value| !value.is_empty())
            .collect::<Vec<_>>()
            .join(", ");
        if !joined.is_empty() {
            out.insert(name.clone(), joined);
        }
    }
    (!out.is_empty()).then_some(out)
}

const DROP_UNSPECIFIED: &str = "unspecified";

pub struct RequestPayloadHandle {
    requested_streaming: bool,
    request_id: String,
    model: String,
    event_time: SystemTime,
    request: Arc<NvCreateChatCompletionRequest>,
    http_request_headers: Option<Arc<BTreeMap<String, String>>>,
}

impl RequestPayloadHandle {
    pub fn streaming(&self) -> bool {
        self.requested_streaming
    }

    pub fn request_id(&self) -> &str {
        &self.request_id
    }

    /// Publish one request trace payload record. Consumes the handle to enforce
    /// exactly one payload record per request. `response` is `None` on client
    /// cancel / gateway timeout / aggregation failure, and may be a partial
    /// record when some content was delivered before a failure; the record still
    /// carries the request so those cases remain inspectable.
    ///
    /// `drop_reason` is required rather than defaulted so a new call site cannot
    /// quietly re-introduce a record that claims completeness it does not have.
    /// Pass `None` only for a faithful, fully aggregated response; otherwise pass
    /// the reason, which is published verbatim as `payload_drop_reason`.
    /// `payload_complete` is derived from the two together, never assumed.
    pub fn emit(
        self,
        response: Option<Arc<NvCreateChatCompletionResponse>>,
        drop_reason: Option<String>,
    ) {
        let payload_complete = response.is_some() && drop_reason.is_none();
        let drop_reason = match (payload_complete, drop_reason) {
            (true, reason) => reason,
            (false, Some(reason)) => Some(reason),
            (false, None) => Some(DROP_UNSPECIFIED.to_string()),
        };
        super::record::emit_request_payload(
            super::RequestTracePayload {
                request_id: self.request_id,
                endpoint: "openai.chat_completion".to_string(),
                model: self.model,
                request: Some(self.request),
                response,
                http_request_headers: self.http_request_headers,
                payload_complete,
                payload_drop_reason: drop_reason,
            },
            unix_time_ms(self.event_time),
        );
    }
}

pub fn create_handle(
    req: &NvCreateChatCompletionRequest,
    request_id: &str,
    http_request_headers: Option<Arc<BTreeMap<String, String>>>,
) -> Option<RequestPayloadHandle> {
    let policy = super::policy();
    // `capture_enabled()` is `policy.enabled && CAPTURE_ACTIVE`: it additionally
    // requires request trace initialization, so a stale payload handle cannot be
    // created before/after the request trace lifecycle.
    create_handle_with_config(
        req,
        request_id,
        super::config::capture_enabled(),
        policy.emit_request_payload_records(),
        http_request_headers,
    )
}

fn unix_time_ms(time: SystemTime) -> u64 {
    time.duration_since(UNIX_EPOCH)
        .map(|duration| duration.as_millis().min(u128::from(u64::MAX)) as u64)
        .unwrap_or(0)
}

fn create_handle_with_config(
    req: &NvCreateChatCompletionRequest,
    request_id: &str,
    enabled: bool,
    emit_request_payload: bool,
    http_request_headers: Option<Arc<BTreeMap<String, String>>>,
) -> Option<RequestPayloadHandle> {
    if !enabled || !emit_request_payload {
        return None;
    }
    let requested_streaming = req.inner.stream.unwrap_or(false);
    let model = req.inner.model.clone();

    Some(RequestPayloadHandle {
        requested_streaming,
        request_id: request_id.to_string(),
        model,
        // Snapshot the pristine inbound request (before the preprocessor
        // overrides stream/usage) and stamp arrival time on the producing
        // thread, so the record reflects what the client sent and when.
        event_time: SystemTime::now(),
        request: Arc::new(req.clone()),
        http_request_headers,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;
    use std::sync::Arc;

    fn create_test_request(model: &str, store: bool) -> NvCreateChatCompletionRequest {
        let json = serde_json::json!({
            "model": model,
            "messages": [{"role": "user", "content": "test"}],
            "store": store
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

    #[test]
    fn request_payload_records_emit_even_when_store_is_false() {
        let request = create_test_request("test-model", false);
        let handle = create_handle_with_config(&request, "test-id", true, true, None);

        assert!(
            handle.is_some(),
            "request_payload records should create a handle even with store=false"
        );
    }

    #[test]
    fn request_payload_records_disabled_skips_store_true_payloads() {
        let request = create_test_request("test-model", true);
        let handle = create_handle_with_config(&request, "test-id", true, false, None);

        assert!(
            handle.is_none(),
            "request_payload records disabled should skip payloads even with store=true"
        );
    }

    #[test]
    fn capture_http_headers_records_only_allowlisted() {
        let capture_list = vec!["x-request-id".to_string(), "nvcf-function-id".to_string()];

        let mut headers = HeaderMap::new();
        headers.insert("x-request-id", "abc-123".parse().unwrap());
        headers.insert("NVCF-Function-Id", "fn-9".parse().unwrap());
        headers.insert("authorization", "Bearer secret".parse().unwrap());

        let captured = capture_http_headers_with_list(&headers, &capture_list)
            .expect("allowlisted headers are captured");
        assert_eq!(
            captured.get("x-request-id").map(String::as_str),
            Some("abc-123")
        );
        assert_eq!(
            captured.get("nvcf-function-id").map(String::as_str),
            Some("fn-9")
        );
        assert!(
            !captured.contains_key("authorization"),
            "non-allowlisted header must never be captured"
        );
    }

    #[test]
    fn capture_http_headers_empty_list_captures_nothing() {
        let mut headers = HeaderMap::new();
        headers.insert("x-request-id", "abc-123".parse().unwrap());

        assert!(
            capture_http_headers_with_list(&headers, &[]).is_none(),
            "empty allowlist must capture nothing"
        );
    }

    #[test]
    fn capture_http_headers_joins_repeated_headers() {
        let capture_list = vec!["x-tag".to_string()];

        let mut headers = HeaderMap::new();
        headers.append("x-tag", "a".parse().unwrap());
        headers.append("x-tag", "b".parse().unwrap());

        let captured = capture_http_headers_with_list(&headers, &capture_list)
            .expect("repeated header is captured");
        assert_eq!(captured.get("x-tag").map(String::as_str), Some("a, b"));
    }

    #[test]
    fn capture_http_headers_omits_repeated_empty_values() {
        let capture_list = vec!["x-tag".to_string()];

        let mut headers = HeaderMap::new();
        headers.append("x-tag", "".parse().unwrap());
        headers.append("x-tag", "".parse().unwrap());

        assert!(
            capture_http_headers_with_list(&headers, &capture_list).is_none(),
            "repeated empty values must be omitted, not joined into \", \""
        );
    }

    #[test]
    fn capture_http_headers_skips_empty_values_when_joining() {
        let capture_list = vec!["x-tag".to_string()];

        let mut headers = HeaderMap::new();
        headers.append("x-tag", "".parse().unwrap());
        headers.append("x-tag", "tenant-a".parse().unwrap());

        let captured = capture_http_headers_with_list(&headers, &capture_list)
            .expect("non-empty value is captured");
        assert_eq!(captured.get("x-tag").map(String::as_str), Some("tenant-a"));
    }

    /// Test-only constructor. `create_handle` gates on env vars + a cached
    /// `OnceLock` policy, which is too brittle for a focused bus-roundtrip test.
    impl RequestPayloadHandle {
        pub(crate) fn for_test(request_id: &str, model: &str, streaming: bool) -> Self {
            Self {
                requested_streaming: streaming,
                request_id: request_id.to_string(),
                model: model.to_string(),
                event_time: SystemTime::now(),
                request: Arc::new(create_test_request(model, true)),
                http_request_headers: None,
            }
        }
    }

    #[tokio::test]
    #[serial_test::serial]
    async fn emit_publishes_one_combined_record_with_and_without_response() {
        // Exercises the contract: `emit` publishes exactly one request trace
        // payload record. With a response present the record carries both
        // request and response; with `None` it carries the request only.
        crate::request_trace::init_bus_for_test(8);
        let mut rx = crate::request_trace::subscribe();

        RequestPayloadHandle::for_test("payload-test-req-ok", "test-model", true)
            .emit(Some(Arc::new(create_test_response("hello"))), None);
        RequestPayloadHandle::for_test("payload-test-req-dropped", "test-model", true)
            .emit(None, Some("response_stream_dropped".to_string()));

        let mut records = HashMap::new();
        tokio::time::timeout(std::time::Duration::from_secs(5), async {
            while records.len() < 2 {
                let record = rx.recv().await.expect("record receives ok");
                if record.event_type != crate::request_trace::RequestTraceEventType::RequestPayload
                {
                    continue;
                }
                let Some(payload) = record.payload.as_ref() else {
                    continue;
                };
                if matches!(
                    payload.request_id.as_str(),
                    "payload-test-req-ok" | "payload-test-req-dropped"
                ) {
                    records.insert(payload.request_id.clone(), record);
                }
            }
        })
        .await
        .expect("expected request payload records before timeout");

        let first = records
            .remove("payload-test-req-ok")
            .expect("payload-test-req-ok record");
        let second = records
            .remove("payload-test-req-dropped")
            .expect("payload-test-req-dropped record");

        assert_eq!(
            first.event_type,
            crate::request_trace::RequestTraceEventType::RequestPayload
        );
        let first_payload = first.payload.as_ref().expect("first payload");
        assert_eq!(first_payload.request_id, "payload-test-req-ok");
        assert!(first_payload.request.is_some());
        assert!(first_payload.response.is_some());
        assert!(first_payload.payload_complete);
        assert!(first_payload.payload_drop_reason.is_none());

        assert_eq!(
            second.event_type,
            crate::request_trace::RequestTraceEventType::RequestPayload
        );
        let second_payload = second.payload.as_ref().expect("second payload");
        assert_eq!(second_payload.request_id, "payload-test-req-dropped");
        assert!(second_payload.request.is_some());
        assert!(second_payload.response.is_none());
        assert!(
            !second_payload.payload_complete,
            "a record with no response must not claim to be complete"
        );
        assert_eq!(
            second_payload.payload_drop_reason.as_deref(),
            Some("response_stream_dropped")
        );
    }

    #[tokio::test]
    #[serial_test::serial]
    async fn emit_without_response_or_reason_still_marks_the_record_incomplete() {
        // Guards a future call site that drops the response without saying why:
        // the record must be labelled, not left indistinguishable from a good one.
        crate::request_trace::init_bus_for_test(8);
        let mut rx = crate::request_trace::subscribe();

        RequestPayloadHandle::for_test("payload-test-req-unspecified", "test-model", true)
            .emit(None, None);

        let record = tokio::time::timeout(std::time::Duration::from_secs(5), async {
            loop {
                let record = rx.recv().await.expect("record receives ok");
                if record.event_type != crate::request_trace::RequestTraceEventType::RequestPayload
                {
                    continue;
                }
                let Some(payload) = record.payload.as_ref() else {
                    continue;
                };
                if payload.request_id == "payload-test-req-unspecified" {
                    return record;
                }
            }
        })
        .await
        .expect("expected request payload record before timeout");

        let payload = record.payload.as_ref().expect("payload");
        assert!(payload.response.is_none());
        assert!(
            !payload.payload_complete,
            "a record with no response must not claim to be complete"
        );
        assert_eq!(payload.payload_drop_reason.as_deref(), Some("unspecified"));
    }
}
