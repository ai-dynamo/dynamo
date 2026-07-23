// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Protocol types for the `/v1/classify` endpoint (sequence-classification /
//! cross-encoder pooling models, e.g. NLI or sentiment).
//!
//! There is no OpenAI-native classification schema, so — unlike embeddings,
//! which wraps `dynamo_protocols::types::CreateEmbeddingRequest` — these types
//! are defined fully in-repo. The wire shape mirrors vLLM's `/classify`
//! endpoint (`vllm/entrypoints/pooling/classify/protocol.py`, vLLM 0.24.0):
//! request `{model, input}` → response
//! `{id, object, created, model, data:[{index, label, probs, num_classes}], usage}`.

use dynamo_runtime::protocols::annotated::AnnotationsProvider;
use serde::{Deserialize, Serialize};
use utoipa::ToSchema;
use validator::Validate;

mod aggregator;
mod nvext;

pub use nvext::{NvExt, NvExtProvider};

/// Classification input — text or pre-tokenized prompts, single or batched.
/// Mirrors the `input` field of vLLM's `ClassificationCompletionRequest`
/// (`CompletionRequestMixin`: `list[int] | list[list[int]] | str | list[str]`),
/// so upstream-tokenized clients migrate unchanged.
#[derive(ToSchema, Serialize, Deserialize, Debug, Clone)]
#[serde(untagged)]
pub enum ClassificationInput {
    /// A single text to classify.
    Single(String),
    /// A batch of texts to classify.
    Batch(Vec<String>),
    /// A single pre-tokenized prompt (token IDs).
    Tokens(Vec<u32>),
    /// A batch of pre-tokenized prompts.
    TokenBatch(Vec<Vec<u32>>),
}

/// Request for the `/classify` endpoint.
#[derive(ToSchema, Serialize, Deserialize, Validate, Debug, Clone)]
pub struct NvCreateClassifyRequest {
    /// The model to use for classification.
    pub model: String,

    /// The text (or texts) to classify.
    pub input: ClassificationInput,

    /// Whether to apply the classification pooler's activation
    /// (sigmoid/softmax). `None` uses the pooler's default. Forwarded verbatim
    /// to `PoolingParams`, mirroring vLLM's `ClassifyRequestMixin`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub use_activation: Option<bool>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub nvext: Option<NvExt>,
}

/// One classification result within a [`NvCreateClassifyResponse`].
#[derive(ToSchema, Serialize, Deserialize, Debug, Clone)]
pub struct ClassificationData {
    /// Index of this item within the request batch.
    pub index: u32,

    /// Predicted label (from the model's `id2label`), if available.
    #[serde(default)]
    pub label: Option<String>,

    /// Per-class probability vector.
    pub probs: Vec<f32>,

    /// Number of classes (== `probs.len()`).
    pub num_classes: u32,
}

/// Usage information for a classification response.
#[derive(ToSchema, Serialize, Deserialize, Debug, Clone, Default)]
pub struct ClassificationUsage {
    pub prompt_tokens: u32,
    pub total_tokens: u32,
}

/// Response for the `/classify` endpoint.
#[derive(ToSchema, Serialize, Deserialize, Validate, Debug, Clone)]
pub struct NvCreateClassifyResponse {
    pub id: String,
    pub object: String,
    pub created: u64,
    pub model: String,
    pub data: Vec<ClassificationData>,
    pub usage: ClassificationUsage,
}

impl NvCreateClassifyResponse {
    pub fn empty() -> Self {
        Self {
            id: String::new(),
            object: "list".to_string(),
            created: 0,
            model: "classify".to_string(),
            data: vec![],
            usage: ClassificationUsage::default(),
        }
    }
}

/// Implements `NvExtProvider` for `NvCreateClassifyRequest`,
/// providing access to NVIDIA-specific extensions.
impl NvExtProvider for NvCreateClassifyRequest {
    fn nvext(&self) -> Option<&NvExt> {
        self.nvext.as_ref()
    }
}

/// Implements `AnnotationsProvider` for `NvCreateClassifyRequest`,
/// enabling retrieval and management of request annotations.
impl AnnotationsProvider for NvCreateClassifyRequest {
    fn annotations(&self) -> Option<Vec<String>> {
        self.nvext
            .as_ref()
            .and_then(|nvext| nvext.annotations.clone())
    }

    fn has_annotation(&self, annotation: &str) -> bool {
        self.nvext
            .as_ref()
            .and_then(|nvext| nvext.annotations.as_ref())
            .map(|annotations| annotations.contains(&annotation.to_string()))
            .unwrap_or(false)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn single_input_round_trips() {
        let request: NvCreateClassifyRequest = serde_json::from_value(json!({
            "model": "test-model",
            "input": "hello world"
        }))
        .unwrap();
        assert!(matches!(request.input, ClassificationInput::Single(_)));
        let value = serde_json::to_value(&request).unwrap();
        assert_eq!(value["input"], "hello world");
    }

    #[test]
    fn batch_input_round_trips() {
        let request: NvCreateClassifyRequest = serde_json::from_value(json!({
            "model": "test-model",
            "input": ["a", "b"]
        }))
        .unwrap();
        match &request.input {
            ClassificationInput::Batch(v) => assert_eq!(v.len(), 2),
            _ => panic!("expected batch input"),
        }
    }

    #[test]
    fn token_inputs_parse_as_token_variants() {
        let request: NvCreateClassifyRequest = serde_json::from_value(json!({
            "model": "test-model",
            "input": [101, 2023, 102]
        }))
        .unwrap();
        assert!(matches!(request.input, ClassificationInput::Tokens(_)));

        let request: NvCreateClassifyRequest = serde_json::from_value(json!({
            "model": "test-model",
            "input": [[101, 102], [101, 103]]
        }))
        .unwrap();
        match &request.input {
            ClassificationInput::TokenBatch(v) => assert_eq!(v.len(), 2),
            other => panic!("expected token batch, got {other:?}"),
        }
    }

    #[test]
    fn use_activation_round_trips_and_defaults_to_none() {
        let request: NvCreateClassifyRequest = serde_json::from_value(json!({
            "model": "test-model",
            "input": "hello"
        }))
        .unwrap();
        assert!(request.use_activation.is_none());
        // Omitted → not serialized onto the wire to the worker.
        assert!(
            serde_json::to_value(&request)
                .unwrap()
                .get("use_activation")
                .is_none()
        );

        let request: NvCreateClassifyRequest = serde_json::from_value(json!({
            "model": "test-model",
            "input": "hello",
            "use_activation": false
        }))
        .unwrap();
        assert_eq!(request.use_activation, Some(false));
        assert_eq!(
            serde_json::to_value(&request).unwrap()["use_activation"],
            serde_json::json!(false)
        );
    }

    #[test]
    fn omitted_nvext_is_not_serialized() {
        let request: NvCreateClassifyRequest = serde_json::from_value(json!({
            "model": "test-model",
            "input": "hello"
        }))
        .unwrap();
        assert!(request.nvext.is_none());
        let value = serde_json::to_value(request).unwrap();
        assert!(value.get("nvext").is_none());
    }
}
