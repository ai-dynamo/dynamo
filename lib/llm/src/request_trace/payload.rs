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

/// Replace media (image/video/audio) request content with text placeholders so the
/// request-trace pipeline never records raw media bytes or URLs.
///
/// Payload records carry the full inbound request; for multimodal requests that
/// includes base64 data URIs or media URLs, which must not be persisted to any sink
/// (size, and privacy — the OTLP sink forwards to a collector).
///
/// Applied where the pristine snapshot is taken, so no sink can observe media even
/// transiently, and both `create_handle` call sites are covered by construction.
///
/// Cost: pure-text requests pay only a scan (a borrow); the clone happens only when
/// at least one media part is present.
///
/// Returns `Some(redacted)` if any message carries a media part, `None` for pure text.
///
/// NOTE: BOTH `User` and `Tool` messages can carry media parts — `preprocessor.rs`
/// converts tool media parts to user parts for multimodal processing. Scanning only
/// user messages would leak media supplied in tool results.
fn redact_media(req: &NvCreateChatCompletionRequest) -> Option<NvCreateChatCompletionRequest> {
    use dynamo_protocols::types::{
        ChatCompletionRequestMessage, ChatCompletionRequestMessageContentPartText,
        ChatCompletionRequestToolMessageContent, ChatCompletionRequestToolMessageContentPart,
        ChatCompletionRequestUserMessageContent, ChatCompletionRequestUserMessageContentPart,
    };

    // `None` => text, keep as-is. `Some(kind)` => media, replace with a placeholder
    // naming `kind`. The trailing catch-all is unreachable for the enum as defined
    // today; it is kept as a fail-closed net so a newly added media variant is
    // redacted as "unknown" rather than leaked.
    fn user_kind(part: &ChatCompletionRequestUserMessageContentPart) -> Option<&'static str> {
        #[allow(unreachable_patterns)]
        match part {
            ChatCompletionRequestUserMessageContentPart::Text(_) => None,
            ChatCompletionRequestUserMessageContentPart::ImageUrl(_) => Some("image_url"),
            ChatCompletionRequestUserMessageContentPart::VideoUrl(_) => Some("video_url"),
            ChatCompletionRequestUserMessageContentPart::AudioUrl(_) => Some("audio"),
            _ => Some("unknown"),
        }
    }
    fn tool_kind(part: &ChatCompletionRequestToolMessageContentPart) -> Option<&'static str> {
        #[allow(unreachable_patterns)]
        match part {
            ChatCompletionRequestToolMessageContentPart::Text(_) => None,
            ChatCompletionRequestToolMessageContentPart::ImageUrl(_) => Some("image_url"),
            ChatCompletionRequestToolMessageContentPart::VideoUrl(_) => Some("video_url"),
            ChatCompletionRequestToolMessageContentPart::AudioUrl(_) => Some("audio"),
            _ => Some("unknown"),
        }
    }

    // ---- Phase 1: scan only, no clone. ----
    let has_media = req.inner.messages.iter().any(|msg| match msg {
        ChatCompletionRequestMessage::User(u) => match &u.content {
            ChatCompletionRequestUserMessageContent::Array(parts) => {
                parts.iter().any(|p| user_kind(p).is_some())
            }
            _ => false,
        },
        ChatCompletionRequestMessage::Tool(t) => match &t.content {
            ChatCompletionRequestToolMessageContent::Array(parts) => {
                parts.iter().any(|p| tool_kind(p).is_some())
            }
            _ => false,
        },
        _ => false,
    });
    if !has_media {
        return None;
    }

    // ---- Phase 2: media present. Clone once, rewrite in place. ----
    let mut redacted = req.clone();
    for msg in redacted.inner.messages.iter_mut() {
        match msg {
            ChatCompletionRequestMessage::User(u) => {
                if let ChatCompletionRequestUserMessageContent::Array(parts) = &mut u.content {
                    for part in parts.iter_mut() {
                        if let Some(kind) = user_kind(part) {
                            *part = ChatCompletionRequestUserMessageContentPart::Text(
                                ChatCompletionRequestMessageContentPartText {
                                    text: format!("[{kind} omitted by audit]"),
                                },
                            );
                        }
                    }
                }
            }
            ChatCompletionRequestMessage::Tool(t) => {
                if let ChatCompletionRequestToolMessageContent::Array(parts) = &mut t.content {
                    for part in parts.iter_mut() {
                        if let Some(kind) = tool_kind(part) {
                            *part = ChatCompletionRequestToolMessageContentPart::Text(
                                ChatCompletionRequestMessageContentPartText {
                                    text: format!("[{kind} omitted by audit]"),
                                },
                            );
                        }
                    }
                }
            }
            _ => {}
        }
    }
    Some(redacted)
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
        policy.redact_media,
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
    redact_media_enabled: bool,
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
        // When enabled, media is redacted BEFORE the snapshot, so no sink (file,
        // stderr, otel) can observe raw image/video/audio bytes even transiently --
        // see `redact_media`. Off by default (`DYN_REQUEST_TRACE_REDACT_MEDIA`), in
        // which case the inbound request is recorded verbatim as it was before.
        request: Arc::new(if redact_media_enabled {
            redact_media(req).unwrap_or_else(|| req.clone())
        } else {
            req.clone()
        }),
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
        let handle = create_handle_with_config(&request, "test-id", true, true, false, None);

        assert!(
            handle.is_some(),
            "request_payload records should create a handle even with store=false"
        );
    }

    #[test]
    fn request_payload_records_disabled_skips_store_true_payloads() {
        let request = create_test_request("test-model", true);
        let handle = create_handle_with_config(&request, "test-id", true, false, false, None);

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
    // -------------------------------------------------------------------------
    // Media redaction (patch F)
    // -------------------------------------------------------------------------

    fn req_from_json(v: serde_json::Value) -> NvCreateChatCompletionRequest {
        serde_json::from_value(v).expect("request should deserialize")
    }

    /// Pure text: no clone, request untouched.
    #[test]
    fn redact_media_none_for_text_only() {
        let req = req_from_json(serde_json::json!({
            "model": "m",
            "messages": [{"role": "user", "content": "hello"}]
        }));
        assert!(redact_media(&req).is_none());
    }

    /// Array of only text parts is still pure text.
    #[test]
    fn redact_media_none_for_text_parts_array() {
        let req = req_from_json(serde_json::json!({
            "model": "m",
            "messages": [{"role": "user", "content": [{"type": "text", "text": "hi"}]}]
        }));
        assert!(redact_media(&req).is_none());
    }

    /// image_url in a USER message is replaced; the text sibling survives; the
    /// ORIGINAL request is not mutated.
    #[test]
    fn redact_media_replaces_user_image_url() {
        let req = req_from_json(serde_json::json!({
            "model": "m",
            "messages": [{"role": "user", "content": [
                {"type": "text", "text": "describe"},
                {"type": "image_url", "image_url": {"url": "data:image/png;base64,SECRET"}}
            ]}]
        }));
        let out = redact_media(&req).expect("media present");
        let v = serde_json::to_value(&out).unwrap();
        let parts = &v["messages"][0]["content"];
        assert_eq!(parts[0]["text"], "describe");
        assert_eq!(parts[1]["text"], "[image_url omitted by audit]");
        assert!(!serde_json::to_string(&v).unwrap().contains("SECRET"));
        // original untouched
        let orig = serde_json::to_value(&req).unwrap();
        assert_eq!(
            orig["messages"][0]["content"][1]["image_url"]["url"],
            "data:image/png;base64,SECRET"
        );
    }

    /// TOOL messages carry media too. The reference implementation this was ported
    /// from scanned only user messages, which would leak here.
    #[test]
    fn redact_media_replaces_tool_media() {
        let req = req_from_json(serde_json::json!({
            "model": "m",
            "messages": [
                {"role": "user", "content": "go"},
                {"role": "tool", "tool_call_id": "c1", "content": [
                    {"type": "text", "text": "result"},
                    {"type": "image_url", "image_url": {"url": "data:image/png;base64,TOOLSECRET"}}
                ]}
            ]
        }));
        let out = redact_media(&req).expect("tool media should be detected");
        let s = serde_json::to_string(&serde_json::to_value(&out).unwrap()).unwrap();
        assert!(!s.contains("TOOLSECRET"), "tool media leaked: {s}");
        assert!(s.contains("[image_url omitted by audit]"));
        assert!(s.contains("result"), "text sibling must survive");
    }

    /// video_url and audio get their own placeholder kinds. Both are exercised: the
    /// `AudioUrl` variant maps to the bare kind `audio`, not `audio_url`, so a copy-paste
    /// of the video arm would not have caught a mislabelled audio placeholder.
    #[test]
    fn redact_media_labels_video_and_audio() {
        let req = req_from_json(serde_json::json!({
            "model": "m",
            "messages": [{"role": "user", "content": [
                {"type": "video_url", "video_url": {"url": "data:video/mp4;base64,VSECRET"}},
                {"type": "audio_url", "audio_url": {"url": "data:audio/wav;base64,ASECRET"}}
            ]}]
        }));
        let out = redact_media(&req).expect("media present");
        let s = serde_json::to_string(&serde_json::to_value(&out).unwrap()).unwrap();
        assert!(s.contains("[video_url omitted by audit]"), "got {s}");
        assert!(s.contains("[audio omitted by audit]"), "got {s}");
        assert!(!s.contains("VSECRET"));
        assert!(!s.contains("ASECRET"));
    }

    /// Every other test here calls `redact_media` directly. This one goes through the
    /// real ingress hook, so a future bypass at `create_handle_with_config` -- the single
    /// point where the pristine snapshot is taken -- cannot leave the helper tests green
    /// while raw media reaches the file/stderr/OTLP sinks.
    #[test]
    fn create_handle_stores_redacted_request() {
        let req = req_from_json(serde_json::json!({
            "model": "m",
            "messages": [{"role": "user", "content": [
                {"type": "image_url", "image_url": {"url": "data:image/png;base64,HOOKSECRET"}}
            ]}]
        }));

        let handle = create_handle_with_config(&req, "test-id", true, true, true, None)
            .expect("payload handle should be created when capture is enabled");
        let stored = serde_json::to_string(handle.request.as_ref()).unwrap();

        assert!(
            !stored.contains("HOOKSECRET"),
            "raw media stored by the snapshot hook: {stored}"
        );
        assert!(
            stored.contains("[image_url omitted by audit]"),
            "got {stored}"
        );
    }

    /// Redaction is OPT-IN (`DYN_REQUEST_TRACE_REDACT_MEDIA`, default off). With the
    /// flag clear the snapshot must be byte-for-byte what the client sent, so enabling
    /// request-trace payload records does not silently change record contents for
    /// deployments that were already consuming them.
    #[test]
    fn create_handle_preserves_media_when_redaction_disabled() {
        let req = req_from_json(serde_json::json!({
            "model": "m",
            "messages": [{"role": "user", "content": [
                {"type": "image_url", "image_url": {"url": "data:image/png;base64,KEEPME"}}
            ]}]
        }));

        let handle = create_handle_with_config(&req, "test-id", true, true, false, None)
            .expect("payload handle should be created when capture is enabled");
        let stored = serde_json::to_string(handle.request.as_ref()).unwrap();

        assert!(stored.contains("KEEPME"), "media was redacted: {stored}");
        assert!(
            !stored.contains("omitted by audit"),
            "placeholder leaked with redaction off: {stored}"
        );
    }

    /// Redaction ON, no media: the overwhelmingly common case. `redact_media` returns
    /// `None` here (scan only, no clone), so the hook must fall back to the request
    /// unchanged -- a regression in that fallback would corrupt or drop content on
    /// effectively all traffic, not just multimodal traffic.
    #[test]
    fn create_handle_leaves_text_only_request_intact_when_redaction_enabled() {
        let req = req_from_json(serde_json::json!({
            "model": "m",
            "messages": [{"role": "user", "content": "just text"}]
        }));

        let handle = create_handle_with_config(&req, "test-id", true, true, true, None)
            .expect("payload handle should be created when capture is enabled");
        let stored = serde_json::to_value(handle.request.as_ref()).unwrap();

        assert_eq!(stored, serde_json::to_value(&req).unwrap());
    }

    /// A `/v1/responses` request carries media as `{"type": "input_image", ...}` --
    /// a shape `redact_media` never sees directly. The HTTP handler converts
    /// `NvCreateResponse` -> `UnifiedRequest` -> `NvCreateChatCompletionRequest`
    /// (`unified.rs`, delegating to `responses::mod.rs`) BEFORE the preprocessor takes
    /// the trace snapshot, and that conversion maps `InputContent::InputImage` onto
    /// `ChatCompletionRequestUserMessageContentPart::ImageUrl`.
    ///
    /// So the Responses API is covered only *by construction*, and nothing pinned it.
    /// If a future direct Responses path skips the chat conversion, or the conversion
    /// stops producing `ImageUrl`, raw media would reach every sink and no other test
    /// in this file would notice -- they all start from a chat-shaped request.
    #[test]
    fn redact_media_covers_responses_api_input_image() {
        use crate::protocols::openai::responses::NvCreateResponse;

        let json = serde_json::json!({
            "model": "m",
            "input": [{
                "role": "user",
                "content": [
                    {"type": "input_text", "text": "describe"},
                    {"type": "input_image", "image_url": "data:image/png;base64,RSECRET"}
                ]
            }]
        });
        let responses_req: NvCreateResponse =
            serde_json::from_value(json).expect("responses request should deserialize");
        let chat: NvCreateChatCompletionRequest = responses_req
            .try_into()
            .expect("responses -> chat conversion should succeed");

        let out = redact_media(&chat).expect("input_image must be media after conversion");
        let s = serde_json::to_string(&serde_json::to_value(&out).unwrap()).unwrap();
        assert!(!s.contains("RSECRET"), "responses-api image leaked: {s}");
        assert!(s.contains("[image_url omitted by audit]"), "got {s}");
        assert!(s.contains("describe"), "text sibling must survive: {s}");
    }
}
