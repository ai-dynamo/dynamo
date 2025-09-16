// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use crate::protocols::openai::chat_completions::NvCreateChatCompletionRequest;
use serde_json::{Value, json};
use std::time::{SystemTime, UNIX_EPOCH};
use uuid::Uuid;

fn audit_enabled() -> bool {
    std::env::var("DYN_AUDIT_ENABLED")
        .map(|s| s == "1" || s.eq_ignore_ascii_case("true"))
        .unwrap_or(false)
}

pub fn should_audit_flags(store: bool, streaming: bool) -> bool {
    !streaming && audit_enabled() && store
}

pub fn log_stored_completion(
    request_id: &str,
    req: &NvCreateChatCompletionRequest,
    response_json: Value,
) {
    let request_val = serde_json::to_value(req).unwrap_or_else(|_| json!({}));

    let ts_ms = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_millis();
    let store_id = format!("store_{}", Uuid::new_v4().simple());

    tracing::info!(
        log_type = "audit",
        schema_version = "1.0",
        ts_ms = ts_ms,
        store_id = %store_id,
        request_id = request_id,
        request = %request_val,
        response = %response_json,
        "Audit log for stored completion"
    );
}

#[cfg(test)]
mod tests {
    use super::*;
    use temp_env;

    #[test]
    fn test_should_audit_flags_logic() {
        temp_env::with_vars([("DYN_AUDIT_ENABLED", Some("true"))], || {
            assert!(should_audit_flags(true, false)); // store=true, non-streaming
            assert!(!should_audit_flags(false, false)); // store=false
            assert!(!should_audit_flags(true, true)); // streaming (not supported)
        });

        temp_env::with_vars([("DYN_AUDIT_ENABLED", Some("false"))], || {
            assert!(!should_audit_flags(true, false)); // disabled
        });

        temp_env::with_vars([("DYN_AUDIT_ENABLED", None::<&str>)], || {
            assert!(!should_audit_flags(true, false)); // missing env var
        });
    }

    #[test]
    fn test_env_var_parsing() {
        for value in ["1", "true", "TRUE"] {
            temp_env::with_vars([("DYN_AUDIT_ENABLED", Some(value))], || {
                assert!(should_audit_flags(true, false));
            });
        }

        for value in ["0", "false", ""] {
            temp_env::with_vars([("DYN_AUDIT_ENABLED", Some(value))], || {
                assert!(!should_audit_flags(true, false));
            });
        }
    }

    #[test]
    fn test_log_function_doesnt_panic() {
        use crate::protocols::openai::chat_completions::NvCreateChatCompletionRequest;
        use dynamo_async_openai::types::{
            ChatCompletionRequestMessage, ChatCompletionRequestUserMessage,
            ChatCompletionRequestUserMessageContent, CreateChatCompletionRequest,
        };
        use serde_json::json;

        temp_env::with_vars([("DYN_AUDIT_ENABLED", Some("true"))], || {
            let request = NvCreateChatCompletionRequest {
                inner: CreateChatCompletionRequest {
                    model: "test".to_string(),
                    messages: vec![ChatCompletionRequestMessage::User(
                        ChatCompletionRequestUserMessage {
                            content: ChatCompletionRequestUserMessageContent::Text(
                                "test".to_string(),
                            ),
                            name: None,
                        },
                    )],
                    store: Some(true),
                    ..Default::default()
                },
                common: Default::default(),
                nvext: None,
            };
            let response = json!({"id": "test", "choices": []});

            // Should not panic
            log_stored_completion("req-123", &request, response);
        });
    }
}
