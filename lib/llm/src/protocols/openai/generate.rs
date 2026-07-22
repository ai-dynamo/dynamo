// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Protocol types for the token-in/token-out `Generate` API
//! (`POST /inference/v1/generate`).
//!
//! These mirror vLLM's `GenerateRequest` / `GenerateResponse` wire contract
//! (`vllm/entrypoints/serve/disagg/protocol.py`). The text-only subset is
//! captured here; `sampling_params` is kept opaque (`serde_json::Value`) for
//! now — the typed sampling envelope lands in a follow-up.
//!
//! Deferred to follow-up PRs (intentionally absent here): `features`
//! (multimodal), `stream_options`, negative-`token_ids` validation-message
//! parity with vLLM, and auto-generating a `request_id` when absent.

use serde::{Deserialize, Serialize};

/// Token-in/token-out generation request.
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct GenerateRequest {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub request_id: Option<String>,

    pub token_ids: Vec<u32>,

    pub sampling_params: serde_json::Value,

    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub model: Option<String>,

    #[serde(default)]
    pub stream: bool,

    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cache_salt: Option<String>,

    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub priority: Option<i64>,

    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub kv_transfer_params: Option<serde_json::Value>,
}

/// A single choice in a `GenerateResponse`.
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct GenerateResponseChoice {
    pub index: u32,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub token_ids: Option<Vec<u32>>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub logprobs: Option<serde_json::Value>,

    pub finish_reason: Option<String>,
}

/// Token-in/token-out generation response.
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct GenerateResponse {
    pub request_id: String,

    pub choices: Vec<GenerateResponseChoice>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub prompt_logprobs: Option<serde_json::Value>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub kv_transfer_params: Option<serde_json::Value>,
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn generate_request_deserializes_from_vllm_json() {
        let raw = json!({
            "request_id": "req-123",
            "token_ids": [1, 2, 3, 4],
            "sampling_params": {"temperature": 0.7, "max_tokens": 16},
            "model": "test-model",
            "stream": false,
            "cache_salt": "salt",
            "priority": 0,
            "kv_transfer_params": null
        });
        let req: GenerateRequest = serde_json::from_value(raw).expect("deserialize");
        assert_eq!(req.request_id.as_deref(), Some("req-123"));
        assert_eq!(req.token_ids, vec![1, 2, 3, 4]);
        assert!(!req.stream);
        assert_eq!(req.model.as_deref(), Some("test-model"));
    }

    #[test]
    fn generate_request_minimal_defaults() {
        // Unknown fields are ignored, stream defaults false, optionals default None.
        let raw = json!({
            "token_ids": [5, 6],
            "sampling_params": {},
            "future_field": "ignored"
        });
        let req: GenerateRequest = serde_json::from_value(raw).expect("deserialize");
        assert_eq!(req.token_ids, vec![5, 6]);
        assert!(!req.stream);
        assert_eq!(req.request_id, None);
        assert_eq!(req.priority, None);
    }

    #[test]
    fn generate_response_round_trips_without_usage_key() {
        let resp = GenerateResponse {
            request_id: "req-123".to_string(),
            choices: vec![GenerateResponseChoice {
                index: 0,
                token_ids: Some(vec![10, 11, 12]),
                logprobs: None,
                finish_reason: Some("stop".to_string()),
            }],
            prompt_logprobs: None,
            kv_transfer_params: None,
        };

        let value = serde_json::to_value(&resp).expect("serialize");
        assert!(
            value.get("usage").is_none(),
            "GenerateResponse must not emit a `usage` key"
        );
        assert!(value.get("prompt_logprobs").is_none());
        assert!(value.get("kv_transfer_params").is_none());

        let round: GenerateResponse =
            serde_json::from_value(value).expect("round-trip deserialize");
        assert_eq!(round.request_id, "req-123");
        assert_eq!(round.choices.len(), 1);
        assert_eq!(round.choices[0].token_ids, Some(vec![10, 11, 12]));
        assert_eq!(round.choices[0].finish_reason.as_deref(), Some("stop"));
    }
}
