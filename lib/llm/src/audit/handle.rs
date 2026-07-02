// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use serde::{Deserialize, Serialize};
use std::sync::Arc;

use super::{bus, config};
use crate::protocols::openai::chat_completions::{
    NvCreateChatCompletionRequest, NvCreateChatCompletionResponse,
};

#[derive(Serialize, Deserialize, Clone)]
pub struct AuditRecord {
    pub schema_version: u32,
    pub request_id: String,
    pub requested_streaming: bool,
    pub model: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub request: Option<Arc<NvCreateChatCompletionRequest>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub response: Option<Arc<NvCreateChatCompletionResponse>>,
}

pub struct AuditHandle {
    requested_streaming: bool,
    request_id: String,
    model: String,
    req_full: Option<Arc<NvCreateChatCompletionRequest>>,
    resp_full: Option<Arc<NvCreateChatCompletionResponse>>,
}

/// Replace media (image/video/audio) request content with text placeholders so the
/// audit pipeline never logs raw media bytes or URLs.
///
/// Audit records carry the full request payload ([`AuditRecord::request`]); for
/// multimodal requests that payload includes base64 data URIs or media URLs, which
/// most operators do not want persisted to audit sinks (size and privacy). This
/// redacts them at ingress, before the request is stored on the handle and later
/// published, so media never reaches any sink.
///
/// It is cost-free for pure-text requests: we only *scan* the message content parts
/// (a borrow of `req`), and clone the request *only* when at least one non-text part
/// is present. Returns:
/// - `Some(redacted)` if any user message carries a non-text content part — the clone
///   has each non-text part replaced by a `Text` part reading `"[<kind> omitted by audit]"`.
/// - `None` if the request is pure text (plain-string content and/or arrays of only
///   `Text` parts). The caller keeps the original `Arc` untouched.
///
/// Only `User` messages carry the user-content-part enum (and therefore media);
/// System/Developer/Assistant/Tool/Function message contents are text/refusal/
/// tool-call shapes only, so they are not scanned.
fn redact_media(req: &NvCreateChatCompletionRequest) -> Option<NvCreateChatCompletionRequest> {
    use dynamo_protocols::types::{
        ChatCompletionRequestMessage, ChatCompletionRequestMessageContentPartText,
        ChatCompletionRequestUserMessageContent, ChatCompletionRequestUserMessageContentPart,
    };

    // Classify a single content part. `None` => text (keep as-is); `Some(kind)` =>
    // non-text media that must be redacted, with `kind` naming the placeholder.
    fn media_kind(part: &ChatCompletionRequestUserMessageContentPart) -> Option<&'static str> {
        // The five arms below are exhaustive for the enum as currently defined, so
        // the trailing catch-all is unreachable today. We keep it (under
        // `allow(unreachable_patterns)`) as a fail-closed net: if a new media
        // content-part variant is ever added upstream, it is conservatively
        // redacted as "unknown" rather than leaked, while staying warning-free now.
        #[allow(unreachable_patterns)]
        match part {
            ChatCompletionRequestUserMessageContentPart::Text(_) => None,
            ChatCompletionRequestUserMessageContentPart::ImageUrl(_) => Some("image_url"),
            ChatCompletionRequestUserMessageContentPart::VideoUrl(_) => Some("video_url"),
            ChatCompletionRequestUserMessageContentPart::AudioUrl(_) => Some("audio"),
            ChatCompletionRequestUserMessageContentPart::InputAudio(_) => Some("audio"),
            _ => Some("unknown"),
        }
    }

    // ---- Phase 1: scan only (no clone). Borrow `req`. ----
    let has_media = req.inner.messages.iter().any(|msg| match msg {
        ChatCompletionRequestMessage::User(u) => match &u.content {
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

    /// Store the request payload for later emission. Media (image/video/audio)
    /// content is redacted to a text placeholder here, before the payload is held
    /// or published, so raw media bytes/URLs never reach any audit sink. Pure-text
    /// requests pay only a scan (no clone) — see [`redact_media`].
    pub fn set_request(&mut self, req: Arc<NvCreateChatCompletionRequest>) {
        self.req_full = Some(match redact_media(&req) {
            Some(redacted) => Arc::new(redacted),
            None => req,
        });
    }
    pub fn set_response(&mut self, resp: Arc<NvCreateChatCompletionResponse>) {
        self.resp_full = Some(resp);
    }

    /// Emit exactly once (publishes to the bus; sinks do I/O).
    pub fn emit(self) {
        let rec = AuditRecord {
            schema_version: 1,
            request_id: self.request_id,
            requested_streaming: self.requested_streaming,
            model: self.model,
            request: self.req_full,
            response: self.resp_full,
        };
        bus::publish(rec);
    }
}

pub fn create_handle(req: &NvCreateChatCompletionRequest, request_id: &str) -> Option<AuditHandle> {
    let policy = config::policy();
    create_handle_with_config(req, request_id, policy.enabled, policy.force_logging)
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
        req_full: None,
        resp_full: None,
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

    /// Test that DYN_AUDIT_FORCE_LOGGING=true bypasses store=false
    /// When force logging is enabled, audit handle should be created even when store=false
    #[test]
    fn test_force_logging_bypasses_store() {
        let request = create_test_request("test-model", false);
        let handle = create_handle_with_config(&request, "test-id", true, true);

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

    // -------------------------------------------------------------------------
    // Media redaction
    // -------------------------------------------------------------------------

    /// Pure-text request (plain-string content) is left untouched: `redact_media`
    /// returns `None`, so `set_request` keeps the original `Arc`.
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

    /// Image content is replaced by the text placeholder; sibling text parts survive
    /// and the original URL is gone from the redacted clone.
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

        // The original request still carries the URL (we did not mutate it).
        let original = serde_json::to_value(&request).unwrap();
        assert_eq!(
            original["messages"][0]["content"][1]["image_url"]["url"],
            "https://example.com/secret.png"
        );
    }

    /// End-to-end: a request set via `set_request` and emitted carries no media
    /// bytes/URLs in the published record's request payload.
    #[test]
    fn set_request_redacts_media_before_emit() {
        let json = serde_json::json!({
            "model": "test-model",
            "store": true,
            "messages": [{
                "role": "user",
                "content": [
                    {"type": "text", "text": "look"},
                    {"type": "image_url", "image_url": {"url": "data:image/png;base64,SECRET"}}
                ]
            }]
        });
        let request: NvCreateChatCompletionRequest =
            serde_json::from_value(json).expect("valid request");

        let mut handle = AuditHandle {
            requested_streaming: false,
            request_id: "req-1".to_string(),
            model: "test-model".to_string(),
            req_full: None,
            resp_full: None,
        };
        handle.set_request(Arc::new(request));

        let stored = serde_json::to_value(handle.req_full.as_ref().unwrap()).unwrap();
        let serialized = stored.to_string();
        assert!(!serialized.contains("SECRET"), "media leaked: {serialized}");
        assert_eq!(
            stored["messages"][0]["content"][1]["text"],
            "[image_url omitted by audit]"
        );
    }
}
