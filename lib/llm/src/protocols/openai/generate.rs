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

use std::collections::HashMap;

use anyhow::Result;
use futures::{Stream, StreamExt};
use serde::{Deserialize, Serialize};

use crate::protocols::Annotated;
use crate::protocols::common::llm_backend::LLMEngineOutput;

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

/// Per-index accumulation state while folding a stream of
/// [`LLMEngineOutput`] deltas into a single [`GenerateResponse`].
struct GenerateChoiceAcc {
    index: u32,
    token_ids: Vec<crate::protocols::TokenIdType>,
    finish_reason: Option<String>,
}

/// Folds a stream of [`Annotated<LLMEngineOutput>`] deltas into a single
/// [`GenerateResponse`]. Each chunk carries a delta of newly generated
/// `token_ids` keyed by `index` (default 0); the aggregator appends the
/// deltas per index and records the terminal `finish_reason`.
struct GenerateAggregator {
    request_id: String,
    choices: HashMap<u32, GenerateChoiceAcc>,
    error: Option<String>,
}

impl GenerateAggregator {
    fn new(request_id: String) -> Self {
        Self {
            request_id,
            choices: HashMap::new(),
            error: None,
        }
    }

    async fn apply(
        stream: impl Stream<Item = Annotated<LLMEngineOutput>>,
        request_id: String,
    ) -> Result<GenerateResponse> {
        let aggregator = stream
            .fold(
                GenerateAggregator::new(request_id),
                |mut agg, delta| async move {
                    let delta = match delta.ok() {
                        Ok(delta) => delta,
                        Err(error) => {
                            agg.error = Some(error);
                            return agg;
                        }
                    };

                    if agg.error.is_none()
                        && let Some(output) = delta.data
                    {
                        let index = output.index.unwrap_or(0);
                        let choice = agg.choices.entry(index).or_insert(GenerateChoiceAcc {
                            index,
                            token_ids: Vec::new(),
                            finish_reason: None,
                        });
                        // token_ids are per-chunk deltas; append (never replace).
                        choice.token_ids.extend(output.token_ids);
                        if let Some(finish_reason) = output.finish_reason {
                            choice.finish_reason = Some(finish_reason.to_string());
                        }
                    }
                    agg
                },
            )
            .await;

        if let Some(error) = aggregator.error {
            return Err(anyhow::anyhow!(error));
        }

        let mut choices: Vec<GenerateResponseChoice> = aggregator
            .choices
            .into_values()
            .map(|acc| GenerateResponseChoice {
                index: acc.index,
                token_ids: Some(acc.token_ids),
                logprobs: None,
                finish_reason: acc.finish_reason,
            })
            .collect();
        choices.sort_by_key(|c| c.index);

        Ok(GenerateResponse {
            request_id: aggregator.request_id,
            choices,
            prompt_logprobs: None,
            kv_transfer_params: None,
        })
    }
}

impl GenerateResponse {
    /// Aggregate a stream of [`Annotated<LLMEngineOutput>`] deltas into a
    /// single [`GenerateResponse`] for the non-streaming `Generate` endpoint.
    pub async fn from_annotated_stream(
        stream: impl Stream<Item = Annotated<LLMEngineOutput>>,
        request_id: String,
    ) -> Result<GenerateResponse> {
        GenerateAggregator::apply(stream, request_id).await
    }
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

    /// Fold a 4-chunk delta stream (each carrying one token_id) into a single
    /// choice: assert the deltas are appended (not duplicated), the terminal
    /// finish_reason is captured, and exactly one choice is emitted.
    #[tokio::test]
    async fn from_annotated_stream_appends_deltas() {
        let chunks = vec![
            LLMEngineOutput {
                token_ids: vec![100],
                index: Some(0),
                ..Default::default()
            },
            LLMEngineOutput {
                token_ids: vec![101],
                index: Some(0),
                ..Default::default()
            },
            LLMEngineOutput {
                token_ids: vec![102],
                index: Some(0),
                ..Default::default()
            },
            LLMEngineOutput {
                token_ids: vec![103],
                index: Some(0),
                finish_reason: Some(crate::protocols::common::FinishReason::Length),
                ..Default::default()
            },
        ];
        let stream = futures::stream::iter(chunks.into_iter().map(Annotated::from_data));

        let resp = GenerateResponse::from_annotated_stream(stream, "req-agg".to_string())
            .await
            .expect("aggregate");

        assert_eq!(resp.request_id, "req-agg");
        assert_eq!(resp.choices.len(), 1);
        let choice = &resp.choices[0];
        assert_eq!(choice.index, 0);
        assert_eq!(choice.token_ids, Some(vec![100, 101, 102, 103]));
        assert_eq!(choice.finish_reason.as_deref(), Some("length"));
    }
}
