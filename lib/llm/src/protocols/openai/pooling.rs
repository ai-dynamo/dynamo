// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Protocol types for the `/v1/pooling` endpoint (raw pooler output from
//! pooling-runner models: token-level embeddings, per-token classification
//! logits, reward-model scores, …).
//!
//! Like `/classify`, there is no OpenAI-native schema; the wire shape mirrors
//! vLLM's `/pooling` endpoint (`vllm/entrypoints/pooling/pooling/protocol.py`):
//! request `{model, input, task?, encoding_format?, use_activation?, …}` →
//! response `{id, object, created, model,
//! data:[{index, object: "pooling", data}], usage}`.
//!
//! The completion-style request (`input`) is supported; vLLM's chat-messages
//! and IOProcessor-plugin request variants are not.
//!
//! `task` is forwarded verbatim and resolved by the vLLM engine: `None`
//! defaults to `token_embed` → `token_classify` → `plugin` (first supported),
//! and unsupported tasks are rejected engine-side — identical to bare
//! `vllm serve`.

use dynamo_runtime::protocols::annotated::AnnotationsProvider;
use serde::{Deserialize, Serialize};
use utoipa::ToSchema;
use validator::Validate;

mod aggregator;
mod nvext;

pub use nvext::{NvExt, NvExtProvider};

/// Pooling input — raw text or pre-tokenized prompts, single or batched.
/// Mirrors the `input` field of vLLM's pooling request
/// (`list[int] | list[list[int]] | str | list[str]`).
#[derive(ToSchema, Serialize, Deserialize, Debug, Clone)]
#[serde(untagged)]
pub enum PoolingInput {
    /// A single text prompt.
    Single(String),
    /// A batch of text prompts.
    Batch(Vec<String>),
    /// A single pre-tokenized prompt (token IDs).
    Tokens(Vec<u32>),
    /// A batch of pre-tokenized prompts.
    TokenBatch(Vec<Vec<u32>>),
}

fn default_encoding_format() -> String {
    "float".to_string()
}

/// Request for the `/v1/pooling` endpoint.
#[derive(ToSchema, Serialize, Deserialize, Validate, Debug, Clone)]
pub struct NvCreatePoolingRequest {
    /// The model to run the pooling pass on.
    pub model: String,

    /// The prompt(s) to pool.
    pub input: PoolingInput,

    /// vLLM pooling task (`embed`, `classify`, `token_embed`,
    /// `token_classify`, …). Forwarded verbatim; `None` lets the engine pick
    /// its default and unknown values are rejected engine-side, so new vLLM
    /// tasks work without a frontend change.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub task: Option<String>,

    /// `"float"` (default) or `"base64"`. Always serialized: the classify
    /// and pooling engines push to the same worker endpoint, and the worker
    /// dispatches on this key's presence (`ClassifyWorkerHandler.generate`),
    /// so it must appear in every pooling wire request.
    #[serde(default = "default_encoding_format")]
    pub encoding_format: String,

    /// Whether to apply the pooler's activation (sigmoid/softmax) to the
    /// output. `None` uses the pooler's default.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub use_activation: Option<bool>,

    /// Matryoshka dimensionality reduction. vLLM's `/pooling` rejects this
    /// parameter ("dimensions is currently not supported"); accepted here so
    /// the same 400 can be returned instead of silently ignoring it.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub dimensions: Option<u32>,

    /// base64 element dtype. vLLM's `EncodingRequestMixin` supports several
    /// (`float32`, `float16`, `bfloat16`, …) and packs them via `tensor2binary`,
    /// but the worker always emits little-endian `float32`. Captured so a
    /// non-default value is rejected with a 400 rather than silently returning
    /// a `float32` payload the caller would misinterpret. `None`/`"float32"`
    /// are accepted.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub embed_dtype: Option<String>,

    /// base64 byte order. vLLM's default is `"native"` (little-endian on the
    /// x86 servers this runs on), which the worker matches. Captured so a
    /// non-default (`"big"`) is rejected with a 400 instead of silently
    /// returning little-endian bytes. `None`/`"native"`/`"little"` are accepted.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub endianness: Option<String>,

    /// Truncate the tokenized prompt to this many tokens (`-1` = model max).
    /// Forwarded to the worker's tokenizer path for raw-text inputs.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub truncate_prompt_tokens: Option<i64>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub nvext: Option<NvExt>,
}

/// One pooled output within a [`NvCreatePoolingResponse`]. The payload shape
/// depends on the resolved task: a matrix for token-level tasks
/// (`token_embed` / `token_classify`), a vector for sequence-level tasks
/// (`embed` / `classify`), or a base64 string of packed floats when the
/// request asked for `encoding_format: "base64"`.
#[derive(ToSchema, Serialize, Deserialize, Debug, Clone)]
#[serde(untagged)]
pub enum PoolingOutput {
    /// Base64-encoded packed float bytes.
    Base64(String),
    /// Sequence-level pooled vector.
    Vector(Vec<f32>),
    /// Token-level output: one row per input token.
    Matrix(Vec<Vec<f32>>),
}

fn default_pooling_object() -> String {
    "pooling".to_string()
}

/// One result item within a [`NvCreatePoolingResponse`].
#[derive(ToSchema, Serialize, Deserialize, Debug, Clone)]
pub struct PoolingData {
    /// Index of this item within the request batch.
    pub index: u32,

    /// Always `"pooling"`.
    #[serde(default = "default_pooling_object")]
    pub object: String,

    /// The raw pooler output for this input.
    pub data: PoolingOutput,
}

/// Usage information for a pooling response. `completion_tokens` is always 0
/// (a pooling pass generates nothing) but is kept for parity with vLLM's
/// `UsageInfo`, which serializes it.
#[derive(ToSchema, Serialize, Deserialize, Debug, Clone, Default)]
pub struct PoolingUsage {
    pub prompt_tokens: u32,
    pub total_tokens: u32,
    #[serde(default)]
    pub completion_tokens: u32,
}

/// Response for the `/v1/pooling` endpoint.
#[derive(ToSchema, Serialize, Deserialize, Validate, Debug, Clone)]
pub struct NvCreatePoolingResponse {
    pub id: String,
    pub object: String,
    pub created: u64,
    pub model: String,
    pub data: Vec<PoolingData>,
    pub usage: PoolingUsage,
}

impl NvCreatePoolingResponse {
    pub fn empty() -> Self {
        Self {
            id: String::new(),
            object: "list".to_string(),
            created: 0,
            model: "pooling".to_string(),
            data: vec![],
            usage: PoolingUsage::default(),
        }
    }
}

/// Implements `NvExtProvider` for `NvCreatePoolingRequest`,
/// providing access to NVIDIA-specific extensions.
impl NvExtProvider for NvCreatePoolingRequest {
    fn nvext(&self) -> Option<&NvExt> {
        self.nvext.as_ref()
    }
}

/// Implements `AnnotationsProvider` for `NvCreatePoolingRequest`,
/// enabling retrieval and management of request annotations.
impl AnnotationsProvider for NvCreatePoolingRequest {
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
    fn single_input_round_trips_and_wire_carries_encoding_format() {
        let request: NvCreatePoolingRequest = serde_json::from_value(json!({
            "model": "test-model",
            "input": "hello world"
        }))
        .unwrap();
        assert!(matches!(request.input, PoolingInput::Single(_)));
        assert_eq!(request.encoding_format, "float");
        assert!(request.task.is_none());

        // The wire discriminator must survive serialization even when the
        // client omitted it (see ClassifyWorkerHandler dispatch).
        let value = serde_json::to_value(&request).unwrap();
        assert_eq!(value["encoding_format"], "float");
        assert!(value.get("task").is_none());
    }

    #[test]
    fn token_inputs_parse_as_token_variants() {
        let request: NvCreatePoolingRequest = serde_json::from_value(json!({
            "model": "test-model",
            "input": [101, 2023, 102]
        }))
        .unwrap();
        assert!(matches!(request.input, PoolingInput::Tokens(_)));

        let request: NvCreatePoolingRequest = serde_json::from_value(json!({
            "model": "test-model",
            "input": [[101, 102], [101, 103]]
        }))
        .unwrap();
        match &request.input {
            PoolingInput::TokenBatch(v) => assert_eq!(v.len(), 2),
            other => panic!("expected token batch, got {other:?}"),
        }
    }

    #[test]
    fn task_and_options_round_trip() {
        let request: NvCreatePoolingRequest = serde_json::from_value(json!({
            "model": "test-model",
            "input": ["a", "b"],
            "task": "token_classify",
            "encoding_format": "base64",
            "use_activation": false
        }))
        .unwrap();
        assert_eq!(request.task.as_deref(), Some("token_classify"));
        assert_eq!(request.encoding_format, "base64");
        assert_eq!(request.use_activation, Some(false));
    }

    #[test]
    fn embed_dtype_and_endianness_are_captured() {
        // Captured (not silently dropped) so the handler can 400 on non-default
        // values instead of returning a mislabeled base64 payload.
        let request: NvCreatePoolingRequest = serde_json::from_value(json!({
            "model": "test-model",
            "input": "hi",
            "encoding_format": "base64",
            "embed_dtype": "float16",
            "endianness": "big"
        }))
        .unwrap();
        assert_eq!(request.embed_dtype.as_deref(), Some("float16"));
        assert_eq!(request.endianness.as_deref(), Some("big"));

        // Omitted → None, and not serialized onto the worker wire.
        let request: NvCreatePoolingRequest = serde_json::from_value(json!({
            "model": "test-model",
            "input": "hi"
        }))
        .unwrap();
        assert!(request.embed_dtype.is_none() && request.endianness.is_none());
        let value = serde_json::to_value(&request).unwrap();
        assert!(value.get("embed_dtype").is_none() && value.get("endianness").is_none());
    }

    #[test]
    fn response_data_accepts_vector_matrix_and_base64() {
        let response: NvCreatePoolingResponse = serde_json::from_value(json!({
            "id": "pool-1",
            "object": "list",
            "created": 1,
            "model": "m",
            "data": [
                {"index": 0, "object": "pooling", "data": [0.1, 0.2]},
                {"index": 1, "object": "pooling", "data": [[0.1], [0.2]]},
                {"index": 2, "object": "pooling", "data": "AAAA"}
            ],
            "usage": {"prompt_tokens": 3, "total_tokens": 3}
        }))
        .unwrap();
        assert!(matches!(response.data[0].data, PoolingOutput::Vector(_)));
        assert!(matches!(response.data[1].data, PoolingOutput::Matrix(_)));
        assert!(matches!(response.data[2].data, PoolingOutput::Base64(_)));
        assert_eq!(response.usage.completion_tokens, 0);
    }
}
