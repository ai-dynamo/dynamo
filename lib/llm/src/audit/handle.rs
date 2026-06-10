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
/// `request_id`. Downstream consumers correlate by `request_id`; the request record
/// is emitted before the worker dispatches, the response record is emitted after the
/// response stream completes successfully. On client cancel mid-stream (or
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

/// Replace media (image/video/audio) request content with text placeholders so the
/// audit pipeline never logs raw media bytes or URLs.
///
/// Contract: this MUST run before [`bus::publish`] so media never enters the
/// broadcast channel or any sink. It is cost-free for pure-text requests: we only
/// *scan* the message content parts (a borrow of `req`), and we clone the request
/// *only* when at least one non-text part is present. Returns:
/// - `Some(redacted)` if any user message carries a non-text content part. The
///   clone has each non-text part replaced by a `Text` part reading
///   `"[<kind> omitted by audit]"`.
/// - `None` if the request is pure text (plain-string content and/or arrays that
///   contain only `Text` parts). The caller keeps the original `Arc` untouched.
///
/// Content shapes handled (see `dynamo_protocols::types`):
/// - `ChatCompletionRequestUserMessageContent::Text(String)` — a plain string;
///   never contains media, left untouched.
/// - `ChatCompletionRequestUserMessageContent::Array(Vec<ContentPart>)` — scanned
///   part-by-part; non-`Text` parts are what we redact.
///
/// Only `User` messages carry the user-content-part enum (and therefore media);
/// System/Developer/Assistant/Tool/Function message contents are text/refusal/
/// tool-call shapes only, so they are not scanned.
fn redact_media(
    req: &NvCreateChatCompletionRequest,
) -> Option<NvCreateChatCompletionRequest> {
    use dynamo_protocols::types::{
        ChatCompletionRequestMessage, ChatCompletionRequestMessageContentPartText,
        ChatCompletionRequestUserMessageContent, ChatCompletionRequestUserMessageContentPart,
    };

    // Classify a single content part. `None` => text (keep as-is); `Some(kind)` =>
    // non-text media that must be redacted, with `kind` naming the placeholder.
    // A catch-all arm future-proofs against new (non-exhaustive) variants: any
    // unknown part is treated as media and redacted rather than leaked.
    fn media_kind(part: &ChatCompletionRequestUserMessageContentPart) -> Option<&'static str> {
        // The five arms below are exhaustive for the enum as defined in
        // dynamo-protocols at v1.2.0, so the trailing catch-all is currently
        // unreachable. We keep it (under `allow(unreachable_patterns)`) as a
        // fail-closed net: if a new media content-part variant is ever added
        // upstream, it is conservatively redacted as "unknown" rather than
        // leaked, and this stays warning-free today.
        #[allow(unreachable_patterns)]
        match part {
            ChatCompletionRequestUserMessageContentPart::Text(_) => None,
            ChatCompletionRequestUserMessageContentPart::ImageUrl(_) => Some("image_url"),
            ChatCompletionRequestUserMessageContentPart::VideoUrl(_) => Some("video_url"),
            ChatCompletionRequestUserMessageContentPart::AudioUrl(_) => Some("audio"),
            ChatCompletionRequestUserMessageContentPart::InputAudio(_) => Some("audio"),
            // Fail-closed catch-all for any future variant.
            _ => Some("unknown"),
        }
    }

    // ---- Phase 1: scan only (no clone). Borrow `req`. ----
    let has_media = req.inner.messages.iter().any(|msg| match msg {
        ChatCompletionRequestMessage::User(u) => match &u.content {
            // Plain string content can never carry media.
            ChatCompletionRequestUserMessageContent::Text(_) => false,
            ChatCompletionRequestUserMessageContent::Array(parts) => {
                parts.iter().any(|p| media_kind(p).is_some())
            }
        },
        _ => false,
    });

    if !has_media {
        // Pure-text request: zero extra cost, no clone.
        return None;
    }

    // ---- Phase 2: media present. Clone once, then rewrite in place. ----
    let mut redacted = req.clone();
    for msg in redacted.inner.messages.iter_mut() {
        if let ChatCompletionRequestMessage::User(u) = msg
            && let ChatCompletionRequestUserMessageContent::Array(parts) = &mut u.content
        {
            for part in parts.iter_mut() {
                if let Some(kind) = media_kind(part) {
                    *part = ChatCompletionRequestUserMessageContentPart::Text(
                        ChatCompletionRequestMessageContentPartText {
                            text: format!("[{kind} omitted by audit]"),
                        },
                    );
                }
            }
        }
    }
    Some(redacted)
}

impl AuditHandle {
    pub fn streaming(&self) -> bool {
        self.requested_streaming
    }

    pub fn request_id(&self) -> &str {
        &self.request_id
    }

    /// Publish a `Request` event record on the audit bus. Call once, as soon as the
    /// request is captured and before worker dispatch — this lets downstream
    /// observers see hung / canceled requests that never produce a response record.
    ///
    /// Media (image/video/audio) request content is redacted to a text placeholder
    /// here, *before* the record is constructed and published, so raw media bytes
    /// or URLs never enter the broadcast channel or any sink. Pure-text requests
    /// pay only a scan (no clone) — see [`redact_media`].
    pub fn emit_request(&self, request: Arc<NvCreateChatCompletionRequest>) {
        let request = match redact_media(&request) {
            Some(redacted) => Arc::new(redacted),
            None => request,
        };
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
    if !config::capture_enabled() {
        return None;
    }
    // If force_logging is enabled, ignore the store flag
    if !policy.force_logging && !req.inner.store.unwrap_or(false) {
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
    use temp_env::with_vars;

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

    /// Test that DYN_AUDIT_FORCE_LOGGING=true bypasses store=false
    /// When force logging is enabled, audit handle should be created even when store=false
    #[test]
    fn test_force_logging_bypasses_store() {
        with_vars(
            [
                ("DYN_AUDIT_SINKS", Some("stderr")),
                ("DYN_AUDIT_FORCE_LOGGING", Some("true")),
            ],
            || {
                // `capture_enabled()` now requires `CAPTURE_ACTIVE`; mimic the
                // audit init lifecycle (`init_from_env_with_shutdown`) instead of
                // relying on the old "uninitialized counts as enabled" semantics.
                crate::audit::config::mark_capture_active();

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

    // -------------------------------------------------------------------------
    // Media redaction tests
    // -------------------------------------------------------------------------

    /// Pure-text request (plain-string content) is left untouched: `redact_media`
    /// returns `None`, so `emit_request` keeps the original `Arc`.
    #[test]
    fn redact_media_returns_none_for_plain_string_content() {
        let request = create_test_request("test-model", true);
        assert!(redact_media(&request).is_none());
    }

    /// A content-parts array containing only `text` parts is still pure text:
    /// `redact_media` returns `None` (no clone).
    #[test]
    fn redact_media_returns_none_for_text_only_parts_array() {
        let json = serde_json::json!({
            "model": "test-model",
            "store": true,
            "messages": [{
                "role": "user",
                "content": [
                    {"type": "text", "text": "hello"},
                    {"type": "text", "text": "world"}
                ]
            }]
        });
        let request: NvCreateChatCompletionRequest =
            serde_json::from_value(json).expect("valid request");
        assert!(redact_media(&request).is_none());
    }

    /// Image content is replaced by the text placeholder; sibling text parts and
    /// the original URL are gone from the redacted clone.
    #[test]
    fn redact_media_replaces_image_url_with_placeholder() {
        let json = serde_json::json!({
            "model": "test-model",
            "store": true,
            "messages": [{
                "role": "user",
                "content": [
                    {"type": "text", "text": "describe this"},
                    {"type": "image_url", "image_url": {"url": "https://example.com/secret.png"}}
                ]
            }]
        });
        let request: NvCreateChatCompletionRequest =
            serde_json::from_value(json).expect("valid request");

        let redacted = redact_media(&request).expect("media present => Some");
        let value = serde_json::to_value(&redacted).unwrap();
        let parts = &value["messages"][0]["content"];
        assert_eq!(parts[0]["type"], "text");
        assert_eq!(parts[0]["text"], "describe this");
        assert_eq!(parts[1]["type"], "text");
        assert_eq!(parts[1]["text"], "[image_url omitted by audit]");

        // The original media URL must not survive anywhere in the payload.
        let serialized = serde_json::to_string(&value).unwrap();
        assert!(
            !serialized.contains("secret.png"),
            "redacted payload must not contain the original media URL"
        );
    }

    /// Video and audio-URL parts are both redacted with their respective kinds
    /// (`video_url` and `audio`). `audio_url` is a dynamo-local content part; the
    /// upstream `input_audio` (base64) variant maps to the same `"audio"` kind.
    #[test]
    fn redact_media_replaces_video_and_audio() {
        let json = serde_json::json!({
            "model": "test-model",
            "store": true,
            "messages": [{
                "role": "user",
                "content": [
                    {"type": "video_url", "video_url": {"url": "https://example.com/clip.mp4"}},
                    {"type": "audio_url", "audio_url": {"url": "https://example.com/clip.wav"}}
                ]
            }]
        });
        let request: NvCreateChatCompletionRequest =
            serde_json::from_value(json).expect("valid request");

        let redacted = redact_media(&request).expect("media present => Some");
        let value = serde_json::to_value(&redacted).unwrap();
        let parts = &value["messages"][0]["content"];
        assert_eq!(parts[0]["text"], "[video_url omitted by audit]");
        assert_eq!(parts[1]["text"], "[audio omitted by audit]");

        let serialized = serde_json::to_string(&value).unwrap();
        assert!(!serialized.contains("clip.mp4"));
        assert!(!serialized.contains("clip.wav"));
    }

    /// `emit_request` publishes the *redacted* request: the original media URL must
    /// never reach the bus.
    #[tokio::test]
    #[serial_test::serial]
    async fn emit_request_publishes_redacted_media() {
        bus::init(8);
        let mut rx = bus::subscribe();

        let json = serde_json::json!({
            "model": "test-model",
            "store": true,
            "messages": [{
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": "https://example.com/secret.png"}}
                ]
            }]
        });
        let request: NvCreateChatCompletionRequest =
            serde_json::from_value(json).expect("valid request");

        let handle = AuditHandle::for_test("req-redact", "test-model", false);
        handle.emit_request(Arc::new(request));

        let rec = tokio::time::timeout(std::time::Duration::from_secs(1), rx.recv())
            .await
            .expect("record arrives before timeout")
            .expect("record receives ok");

        let value = serde_json::to_value(&rec).unwrap();
        let serialized = serde_json::to_string(&value).unwrap();
        assert!(
            !serialized.contains("secret.png"),
            "media URL must not reach the audit bus"
        );
        assert_eq!(
            value["request"]["messages"][0]["content"][0]["text"],
            "[image_url omitted by audit]"
        );
    }
}
