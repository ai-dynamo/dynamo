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
///
/// A capture entry ending in `*` matches every header name carrying that prefix, so a
/// gateway-injected family can be captured without enumerating it. Denylist entries take
/// the same syntax and subtract from whatever the capture list matched.
pub fn capture_http_headers(headers: &HeaderMap) -> Option<BTreeMap<String, String>> {
    if !http_header_capture_active() {
        return None;
    }
    let policy = super::policy();
    capture_http_headers_with_lists(
        headers,
        &policy.http_header_capture_list,
        &policy.http_header_deny_list,
    )
}

/// True when `entry` ends in `*`, which is the one form that cannot be resolved by a
/// direct name lookup.
fn is_prefix_pattern(entry: &str) -> bool {
    entry.len() > 1 && entry.ends_with('*')
}

/// Match one configured entry against a header name. Both sides are already lowercased.
fn entry_matches(entry: &str, name: &str) -> bool {
    match entry.strip_suffix('*') {
        // A zero-length prefix never matches, so an all-`*` entry cannot turn into
        // capture-all here. The deny list keeps such an entry and handles it in
        // `denied`; the capture list drops it when the list is parsed.
        Some(prefix) => !prefix.is_empty() && name.starts_with(prefix),
        None => entry == name,
    }
}

fn any_entry_matches(entries: &[String], name: &str) -> bool {
    entries.iter().any(|entry| entry_matches(entry, name))
}

/// True when every character is `*`. Only the deny list keeps such an entry, where it
/// means "deny everything"; the capture list drops it at config load.
fn is_star_sentinel(entry: &str) -> bool {
    !entry.is_empty() && entry.chars().all(|c| c == '*')
}

/// Deny entries match like capture entries, plus the all-`*` "deny everything" sentinel.
fn denied(deny_list: &[String], name: &str) -> bool {
    deny_list
        .iter()
        .any(|entry| is_star_sentinel(entry) || entry_matches(entry, name))
}

/// Comma-join the non-empty values recorded under `name`.
fn joined_values(headers: &HeaderMap, name: &str) -> Option<String> {
    let joined = headers
        .get_all(name)
        .iter()
        .filter_map(|value| value.to_str().ok())
        .filter(|value| !value.is_empty())
        .collect::<Vec<_>>()
        .join(", ");
    (!joined.is_empty()).then_some(joined)
}

fn capture_http_headers_with_lists(
    headers: &HeaderMap,
    capture_list: &[String],
    deny_list: &[String],
) -> Option<BTreeMap<String, String>> {
    if capture_list.is_empty() {
        return None;
    }
    let mut out = BTreeMap::new();
    if capture_list.iter().any(|entry| is_prefix_pattern(entry)) {
        // A prefix entry has no single name to look up, so walk the request's headers
        // once. Keys come off the request, because that is where the matched name lives.
        for name in headers.keys() {
            let name = name.as_str();
            if !any_entry_matches(capture_list, name) || denied(deny_list, name) {
                continue;
            }
            if let Some(joined) = joined_values(headers, name) {
                out.insert(name.to_string(), joined);
            }
        }
    } else {
        // Every entry is a literal name, so keep the direct per-entry lookup.
        for name in capture_list {
            if denied(deny_list, name) {
                continue;
            }
            if let Some(joined) = joined_values(headers, name.as_str()) {
                out.insert(name.clone(), joined);
            }
        }
    }
    (!out.is_empty()).then_some(out)
}

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
    /// cancel / gateway timeout / aggregation failure; the record still carries
    /// the request so those cases remain inspectable.
    pub fn emit(self, response: Option<Arc<NvCreateChatCompletionResponse>>) {
        super::record::emit_request_payload(
            super::RequestTracePayload {
                request_id: self.request_id,
                endpoint: "openai.chat_completion".to_string(),
                model: self.model,
                request: Some(self.request),
                response,
                http_request_headers: self.http_request_headers,
                payload_complete: true,
                payload_drop_reason: None,
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

        let captured = capture_http_headers_with_lists(&headers, &capture_list, &[])
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
            capture_http_headers_with_lists(&headers, &[], &[]).is_none(),
            "empty allowlist must capture nothing"
        );
    }

    #[test]
    fn capture_http_headers_joins_repeated_headers() {
        let capture_list = vec!["x-tag".to_string()];

        let mut headers = HeaderMap::new();
        headers.append("x-tag", "a".parse().unwrap());
        headers.append("x-tag", "b".parse().unwrap());

        let captured = capture_http_headers_with_lists(&headers, &capture_list, &[])
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
            capture_http_headers_with_lists(&headers, &capture_list, &[]).is_none(),
            "repeated empty values must be omitted, not joined into \", \""
        );
    }

    #[test]
    fn capture_http_headers_skips_empty_values_when_joining() {
        let capture_list = vec!["x-tag".to_string()];

        let mut headers = HeaderMap::new();
        headers.append("x-tag", "".parse().unwrap());
        headers.append("x-tag", "tenant-a".parse().unwrap());

        let captured = capture_http_headers_with_lists(&headers, &capture_list, &[])
            .expect("non-empty value is captured");
        assert_eq!(captured.get("x-tag").map(String::as_str), Some("tenant-a"));
    }

    #[test]
    fn capture_http_headers_matches_a_prefix_family() {
        let capture_list = vec!["x-someplatform-*".to_string(), "x-request-id".to_string()];

        let mut headers = HeaderMap::new();
        headers.insert("x-someplatform-tenant", "acme".parse().unwrap());
        headers.insert("x-someplatform-region", "us-east".parse().unwrap());
        headers.insert("x-request-id", "abc-123".parse().unwrap());
        headers.insert("x-unrelated", "nope".parse().unwrap());

        let captured = capture_http_headers_with_lists(&headers, &capture_list, &[])
            .expect("prefix entry captures the family");
        assert_eq!(
            captured.keys().map(String::as_str).collect::<Vec<_>>(),
            vec![
                "x-request-id",
                "x-someplatform-region",
                "x-someplatform-tenant"
            ],
            "a prefix entry captures every matching name and nothing else"
        );
        assert_eq!(
            captured.get("x-someplatform-tenant").map(String::as_str),
            Some("acme"),
            "keys are the header names off the request, not the configured entry"
        );
    }

    #[test]
    fn capture_http_headers_denylist_subtracts_from_a_prefix_match() {
        let capture_list = vec!["x-someplatform-*".to_string()];
        let deny_list = vec!["x-someplatform-internal-token".to_string()];

        let mut headers = HeaderMap::new();
        headers.insert("x-someplatform-tenant", "acme".parse().unwrap());
        headers.insert("x-someplatform-internal-token", "secret".parse().unwrap());

        let captured = capture_http_headers_with_lists(&headers, &capture_list, &deny_list)
            .expect("the rest of the family is still captured");
        assert_eq!(
            captured.keys().map(String::as_str).collect::<Vec<_>>(),
            vec!["x-someplatform-tenant"]
        );
        assert!(
            !captured.contains_key("x-someplatform-internal-token"),
            "a denied name must not reach the record"
        );
    }

    #[test]
    fn capture_http_headers_denylist_accepts_prefix_entries() {
        let capture_list = vec!["x-someplatform-*".to_string()];
        let deny_list = vec!["x-someplatform-internal-*".to_string()];

        let mut headers = HeaderMap::new();
        headers.insert("x-someplatform-tenant", "acme".parse().unwrap());
        headers.insert("x-someplatform-internal-token", "secret".parse().unwrap());
        headers.insert("x-someplatform-internal-trace", "secret".parse().unwrap());

        let captured = capture_http_headers_with_lists(&headers, &capture_list, &deny_list)
            .expect("the non-denied name is still captured");
        assert_eq!(
            captured.keys().map(String::as_str).collect::<Vec<_>>(),
            vec!["x-someplatform-tenant"]
        );
    }

    #[test]
    fn capture_http_headers_denylist_applies_to_literal_entries() {
        let capture_list = vec!["x-request-id".to_string(), "x-tenant".to_string()];
        let deny_list = vec!["x-tenant".to_string()];

        let mut headers = HeaderMap::new();
        headers.insert("x-request-id", "abc-123".parse().unwrap());
        headers.insert("x-tenant", "acme".parse().unwrap());

        let captured = capture_http_headers_with_lists(&headers, &capture_list, &deny_list)
            .expect("the allowed literal is still captured");
        assert_eq!(
            captured.keys().map(String::as_str).collect::<Vec<_>>(),
            vec!["x-request-id"],
            "the denylist subtracts on the literal path too, not just under a prefix"
        );
    }

    #[test]
    fn capture_http_headers_denylist_only_subtracts() {
        let capture_list = vec!["x-request-id".to_string()];
        let deny_list = vec!["x-tenant".to_string()];

        let mut headers = HeaderMap::new();
        headers.insert("x-request-id", "abc-123".parse().unwrap());
        headers.insert("x-tenant", "acme".parse().unwrap());

        let captured = capture_http_headers_with_lists(&headers, &capture_list, &deny_list)
            .expect("capture list still applies");
        assert_eq!(
            captured.keys().map(String::as_str).collect::<Vec<_>>(),
            vec!["x-request-id"],
            "naming an uncaptured header in the denylist changes nothing"
        );
    }

    #[test]
    fn capture_http_headers_bare_star_entry_captures_nothing() {
        // The capture-all sentinel is deferred until captured values are redacted. If a
        // bare `*` survives config parsing it must not silently become capture-all.
        let mut headers = HeaderMap::new();
        headers.insert("authorization", "Bearer secret".parse().unwrap());
        headers.insert("x-request-id", "abc-123".parse().unwrap());

        for capture_list in [vec!["*".to_string()], vec!["**".to_string()]] {
            assert!(
                capture_http_headers_with_lists(&headers, &capture_list, &[]).is_none(),
                "an all-`*` capture entry must not capture every header"
            );
        }
    }

    #[test]
    fn capture_http_headers_deny_all_sentinel_subtracts_everything() {
        // Deny-all only ever subtracts, so it is honoured rather than dropped. Failing
        // open on an explicit denial would be the dangerous direction here.
        let capture_list = vec!["x-someplatform-*".to_string(), "x-request-id".to_string()];

        let mut headers = HeaderMap::new();
        headers.insert("x-someplatform-tenant", "acme".parse().unwrap());
        headers.insert("x-request-id", "abc-123".parse().unwrap());

        for deny in [vec!["*".to_string()], vec!["**".to_string()]] {
            assert!(
                capture_http_headers_with_lists(&headers, &capture_list, &deny).is_none(),
                "an all-`*` deny entry must subtract every captured header"
            );
        }
    }

    #[test]
    fn capture_http_headers_interior_star_is_matched_literally() {
        let capture_list = vec!["x-*-id".to_string()];

        let mut headers = HeaderMap::new();
        headers.insert("x-request-id", "abc-123".parse().unwrap());

        assert!(
            capture_http_headers_with_lists(&headers, &capture_list, &[]).is_none(),
            "`*` is only a trailing prefix marker, so an interior one matches no real name"
        );
    }

    #[test]
    fn capture_http_headers_joins_repeats_matched_by_a_prefix() {
        let capture_list = vec!["x-someplatform-*".to_string()];

        let mut headers = HeaderMap::new();
        headers.append("x-someplatform-tag", "tenant-a".parse().unwrap());
        headers.append("x-someplatform-tag", "".parse().unwrap());
        headers.append("x-someplatform-tag", "tenant-b".parse().unwrap());

        let captured = capture_http_headers_with_lists(&headers, &capture_list, &[])
            .expect("prefix match captures repeats");
        assert_eq!(
            captured.get("x-someplatform-tag").map(String::as_str),
            Some("tenant-a, tenant-b"),
            "repeats join and empty values drop, same as the literal path"
        );
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
            .emit(Some(Arc::new(create_test_response("hello"))));
        RequestPayloadHandle::for_test("payload-test-req-cancel", "test-model", true).emit(None);

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
                    "payload-test-req-ok" | "payload-test-req-cancel"
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
            .remove("payload-test-req-cancel")
            .expect("payload-test-req-cancel record");

        assert_eq!(
            first.event_type,
            crate::request_trace::RequestTraceEventType::RequestPayload
        );
        let first_payload = first.payload.as_ref().expect("first payload");
        assert_eq!(first_payload.request_id, "payload-test-req-ok");
        assert!(first_payload.request.is_some());
        assert!(first_payload.response.is_some());

        assert_eq!(
            second.event_type,
            crate::request_trace::RequestTraceEventType::RequestPayload
        );
        let second_payload = second.payload.as_ref().expect("second payload");
        assert_eq!(second_payload.request_id, "payload-test-req-cancel");
        assert!(second_payload.request.is_some());
        assert!(second_payload.response.is_none());
    }
}
