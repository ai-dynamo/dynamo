// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::fmt;

use dynamo_protocols::types::{
    ChatCompletionRequestAssistantMessageContent as AssistantContent,
    ChatCompletionRequestDeveloperMessageContent as DeveloperContent,
    ChatCompletionRequestSystemMessageContent as SystemContent,
    ChatCompletionRequestToolMessageContent as ToolContent,
    ChatCompletionRequestUserMessageContent as UserContent,
};
use serde::{Deserialize, de::value::MapDeserializer};

use crate::protocols::common::metadata_upload::UniqueJsonValue;

pub(super) struct UnsupportedContentPart {
    message_index: usize,
    part_index: usize,
    type_name: Option<String>,
}

impl fmt::Display for UnsupportedContentPart {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str("Unsupported content part type")?;
        if let Some(type_name) = &self.type_name {
            write!(formatter, " '{type_name}'")?;
        }
        write!(
            formatter,
            " at messages[{}].content[{}]",
            self.message_index, self.part_index
        )
    }
}

// Only diagnose requests the typed parser already rejected; never change admission.
pub(super) fn find_unsupported_content_part(body: &[u8]) -> Option<UnsupportedContentPart> {
    let UniqueJsonValue(value) = serde_json::from_slice(body).ok()?;
    for (message_index, message) in value.get("messages")?.as_array()?.iter().enumerate() {
        let Some(role) = message.get("role").and_then(|role| role.as_str()) else {
            continue;
        };
        let Some(parts) = message
            .get("content")
            .and_then(|content| content.as_array())
        else {
            continue;
        };
        for (part_index, part) in parts.iter().enumerate() {
            let Some(tag) = part.get("type").and_then(|tag| tag.as_str()) else {
                continue;
            };
            let is_unknown = match role {
                "user" => is_unknown_type(tag, UserContent::Array),
                "tool" => is_unknown_type(tag, ToolContent::Array),
                "assistant" => is_unknown_type(tag, AssistantContent::Array),
                "system" => is_unknown_type(tag, SystemContent::Array),
                "developer" => is_unknown_type(tag, DeveloperContent::Array),
                _ => false,
            };
            if is_unknown {
                return Some(UnsupportedContentPart {
                    message_index,
                    part_index,
                    type_name: reportable_tag(tag),
                });
            }
        }
    }
    None
}

fn reportable_tag(tag: &str) -> Option<String> {
    let is_safe = !tag.is_empty()
        && tag.len() <= 64
        && tag
            .bytes()
            .all(|byte| byte.is_ascii_lowercase() || byte.is_ascii_digit() || byte == b'_');
    is_safe.then(|| tag.to_owned())
}

// Array constructors infer each role's canonical element type, including upstream
// types that dynamo-protocols does not re-export. Payload errors are irrelevant:
// only the tagged enum's unknown_variant callback means the type is unsupported.
fn is_unknown_type<'de, T: Deserialize<'de>, C>(tag: &'de str, _array: fn(Vec<T>) -> C) -> bool {
    let deserializer = MapDeserializer::<_, TagError>::new([("type", tag)].into_iter());
    matches!(T::deserialize(deserializer), Err(TagError::UnknownType))
}

#[derive(Debug, thiserror::Error)]
enum TagError {
    #[error("unknown content type")]
    UnknownType,
    #[error("invalid content part")]
    Invalid,
}

impl serde::de::Error for TagError {
    fn custom<T: fmt::Display>(_message: T) -> Self {
        Self::Invalid
    }

    fn unknown_variant(_variant: &str, _expected: &'static [&'static str]) -> Self {
        Self::UnknownType
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::{Value, json};

    fn diagnostic(role: &str, part: Value) -> Option<UnsupportedContentPart> {
        find_unsupported_content_part(
            &serde_json::to_vec(&json!({"messages": [{"role": role, "content": [part]}]})).unwrap(),
        )
    }

    #[test]
    fn canonical_role_types_distinguish_unknown_tags_from_missing_payloads() {
        for (role, tags) in [
            (
                "user",
                &[
                    "text",
                    "image_url",
                    "video_url",
                    "audio_url",
                    "custom",
                    "input_audio",
                ][..],
            ),
            ("tool", &["text", "image_url", "video_url", "audio_url"][..]),
            ("assistant", &["text", "refusal"][..]),
            ("system", &["text"][..]),
            ("developer", &["text"][..]),
        ] {
            for tag in tags {
                assert!(
                    diagnostic(role, json!({"type": tag})).is_none(),
                    "{role}/{tag}"
                );
            }
            for tag in ["file", "future_media"] {
                let error = diagnostic(role, json!({"type": tag})).unwrap();
                assert_eq!(error.type_name.as_deref(), Some(tag));
            }
        }
        assert!(diagnostic("tool", json!({"type": "input_audio"})).is_some());
    }

    #[test]
    fn diagnostic_reports_first_position_without_echoing_payload_or_unsafe_tags() {
        let body = serde_json::to_vec(&json!({"messages": [
            {"role": "user", "content": "hello"},
            {"role": "user", "content": [{"type": "text", "text": "summarize"},
                {"type": "file", "file": {"file_data": "private-document"}}, {"type": "other"}]}
        ]}))
        .unwrap();
        let error = find_unsupported_content_part(&body).unwrap();
        assert_eq!((error.message_index, error.part_index), (1, 1));
        let message = error.to_string();
        assert!(message.contains("'file'"));
        assert!(message.contains("messages[1].content[1]"));
        assert!(!message.contains("private-document"));
        assert!(!message.contains("other"));
        for tag in [
            "".to_string(),
            "x".repeat(65),
            "secret\n".to_string(),
            "秘密".to_string(),
        ] {
            let error = diagnostic("user", json!({"type": tag})).unwrap();
            assert!(error.type_name.is_none());
            assert!(error.to_string().len() < 128);
            assert!(error.to_string().contains("messages[0].content[0]"));
        }
        let tag = "x".repeat(64);
        assert_eq!(
            diagnostic("user", json!({"type": tag})).unwrap().type_name,
            Some(tag)
        );
    }

    #[test]
    fn unrelated_parse_errors_and_duplicate_keys_keep_the_original_diagnostic() {
        for part in [
            json!({}),
            json!({"type": 7}),
            json!({"type": "image_url", "image_url": 7}),
        ] {
            assert!(diagnostic("user", part).is_none());
        }
        assert!(diagnostic("unknown", json!({"type": "file"})).is_none());
        assert!(diagnostic("function", json!({"type": "file"})).is_none());
        for body in [
            "null",
            "{}",
            r#"{"messages": [null, {}]}"#,
            r#"{"messages":[{"role":"user","content":[{"type":"file"}]}]"#,
            r#"{"messages":[{"role":"user","content":[{"type":"text","type":"file"}]}]}"#,
            r#"{"messages":[],"messages":[{"role":"user","content":[{"type":"file"}]}]}"#,
        ] {
            assert!(
                find_unsupported_content_part(body.as_bytes()).is_none(),
                "{body}"
            );
        }
    }
}
