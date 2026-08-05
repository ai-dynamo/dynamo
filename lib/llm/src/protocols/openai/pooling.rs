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
//! and IOProcessor-plugin request variants are not. Pooling controls such as
//! `task`, `priority`, `cache_salt`, `mm_processor_kwargs`, `use_activation`,
//! `encoding_format`, `add_special_tokens`, and `truncate_prompt_tokens` are
//! forwarded; `dimensions` is rejected with a 400.
//!
//! `task` is forwarded verbatim when set. When omitted, the worker resolves
//! the model's configured/default task using vLLM's `ModelConfig`.

use std::collections::HashMap;

use dynamo_runtime::protocols::annotated::AnnotationsProvider;
use serde::{Deserialize, Serialize};
use utoipa::ToSchema;
use validator::Validate;

mod aggregator;

use super::PromptTruncationSide;
pub use super::embeddings::{NvExt, NvExtProvider};

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

/// Client-visible pooling response representation.
#[derive(ToSchema, Serialize, Deserialize, Debug, Clone, Copy, PartialEq, Eq, Default)]
#[serde(rename_all = "snake_case")]
pub enum PoolingEncodingFormat {
    #[default]
    Float,
    Base64,
    Bytes,
    BytesOnly,
}

/// Element type used for base64 and binary pooling responses.
#[derive(ToSchema, Serialize, Deserialize, Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum PoolingEmbedDType {
    #[default]
    #[serde(rename = "float32")]
    Float32,
    #[serde(rename = "float16")]
    Float16,
    #[serde(rename = "bfloat16")]
    Bfloat16,
    #[serde(rename = "fp8_e4m3")]
    Fp8E4m3,
    #[serde(rename = "fp8_e5m2")]
    Fp8E5m2,
}

impl PoolingEmbedDType {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Float32 => "float32",
            Self::Float16 => "float16",
            Self::Bfloat16 => "bfloat16",
            Self::Fp8E4m3 => "fp8_e4m3",
            Self::Fp8E5m2 => "fp8_e5m2",
        }
    }
}

/// Byte order used for base64 and binary pooling responses.
#[derive(ToSchema, Serialize, Deserialize, Debug, Clone, Copy, PartialEq, Eq, Default)]
#[serde(rename_all = "snake_case")]
pub enum PoolingEndianness {
    #[default]
    Native,
    Big,
    Little,
}

impl PoolingEndianness {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Native => "native",
            Self::Big => "big",
            Self::Little => "little",
        }
    }
}

/// Request for the `/v1/pooling` endpoint.
#[derive(ToSchema, Serialize, Deserialize, Validate, Debug, Clone)]
pub struct NvCreatePoolingRequest {
    /// The model to run the pooling pass on.
    pub model: String,

    /// The prompt(s) to pool.
    pub input: PoolingInput,

    /// A unique identifier representing the end user.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub user: Option<String>,

    /// Optional caller-provided identifier used in the response ID. Internal
    /// engine request IDs remain server-generated.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub request_id: Option<String>,

    /// Scheduling priority. Lower values are scheduled first.
    #[serde(default)]
    pub priority: i64,

    /// Additional keyword arguments forwarded to the Hugging Face processor.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub mm_processor_kwargs: Option<HashMap<String, serde_json::Value>>,

    /// Salt applied to prefix-cache keys.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cache_salt: Option<String>,

    /// vLLM pooling task (`embed`, `classify`, `token_embed`,
    /// `token_classify`, …). Forwarded verbatim when set; `None` lets the
    /// worker resolve the model's configured/default task.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub task: Option<String>,

    /// `"float"` (default), `"base64"`, `"bytes"`, or `"bytes_only"`. Always
    /// serialized: the classify and pooling engines push to the same worker
    /// endpoint, and the worker dispatches on this key's presence.
    #[serde(default)]
    pub encoding_format: PoolingEncodingFormat,

    /// Whether to apply the pooler's activation (sigmoid/softmax) to the
    /// output. `None` uses the pooler's default.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub use_activation: Option<bool>,

    /// Whether to add the tokenizer's special tokens (BOS/CLS/SEP). `None`
    /// uses the tokenizer default (`true`). Forwarded to the worker's
    /// tokenization, mirroring vLLM's `CompletionRequestMixin.add_special_tokens`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub add_special_tokens: Option<bool>,

    /// Which side to truncate from. When omitted, the tokenizer default is
    /// used.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub truncation_side: Option<PromptTruncationSide>,

    /// Matryoshka dimensionality reduction. vLLM's `/pooling` rejects this
    /// parameter ("dimensions is currently not supported"); accepted here so
    /// the same 400 can be returned instead of silently ignoring it.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub dimensions: Option<u32>,

    /// Element dtype for base64 and binary responses.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub embed_dtype: Option<PoolingEmbedDType>,

    /// Byte order for base64 and binary responses.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub endianness: Option<PoolingEndianness>,

    /// Truncate the tokenized prompt to this many tokens (`-1` = model max).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub truncate_prompt_tokens: Option<i64>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub nvext: Option<NvExt>,
}

/// One pooled output within a [`NvCreatePoolingResponse`]. The payload shape
/// depends on the resolved task: a matrix for token-level tasks
/// (`token_embed` / `token_classify`), a vector for sequence-level tasks
/// (`embed` / `classify`), or a base64 string of packed tensor bytes for an
/// encoded response. Binary responses are decoded at the HTTP boundary.
#[derive(ToSchema, Serialize, Deserialize, Debug, Clone)]
#[serde(untagged)]
pub enum PoolingOutput {
    /// Base64-encoded packed tensor bytes.
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

    /// Tensor shape carried on Dynamo's internal wire for binary responses.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[schema(ignore)]
    pub shape: Option<Vec<u64>>,
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
        assert_eq!(request.encoding_format, PoolingEncodingFormat::Float);
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
        assert_eq!(request.encoding_format, PoolingEncodingFormat::Base64);
        assert_eq!(request.use_activation, Some(false));
    }

    #[test]
    fn binary_encoding_options_round_trip() {
        let request: NvCreatePoolingRequest = serde_json::from_value(json!({
            "model": "test-model",
            "input": "hi",
            "encoding_format": "bytes",
            "embed_dtype": "float16",
            "endianness": "big"
        }))
        .unwrap();
        assert_eq!(request.encoding_format, PoolingEncodingFormat::Bytes);
        assert_eq!(request.embed_dtype, Some(PoolingEmbedDType::Float16));
        assert_eq!(request.endianness, Some(PoolingEndianness::Big));
        let value = serde_json::to_value(&request).unwrap();
        assert_eq!(value["encoding_format"], "bytes");
        assert_eq!(value["embed_dtype"], "float16");
        assert_eq!(value["endianness"], "big");

        let request: NvCreatePoolingRequest = serde_json::from_value(json!({
            "model": "test-model",
            "input": "hi",
            "encoding_format": "bytes_only",
            "embed_dtype": "fp8_e4m3",
            "endianness": "little"
        }))
        .unwrap();
        assert_eq!(request.encoding_format, PoolingEncodingFormat::BytesOnly);
        assert_eq!(request.embed_dtype, Some(PoolingEmbedDType::Fp8E4m3));
        assert_eq!(request.endianness, Some(PoolingEndianness::Little));

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
    fn tokenization_options_round_trip() {
        let request: NvCreatePoolingRequest = serde_json::from_value(json!({
            "model": "test-model",
            "input": "hi",
            "add_special_tokens": false,
            "truncate_prompt_tokens": 64,
            "truncation_side": "left"
        }))
        .unwrap();
        assert_eq!(request.add_special_tokens, Some(false));
        assert_eq!(request.truncate_prompt_tokens, Some(64));
        assert_eq!(request.truncation_side, Some(PromptTruncationSide::Left));
    }

    #[test]
    fn pooling_request_controls_round_trip() {
        let request: NvCreatePoolingRequest = serde_json::from_value(json!({
            "model": "test-model",
            "input": "hello",
            "user": "user-1",
            "request_id": "request-1",
            "priority": -2,
            "mm_processor_kwargs": {"do_resize": false},
            "cache_salt": "salt"
        }))
        .unwrap();

        assert_eq!(request.user.as_deref(), Some("user-1"));
        assert_eq!(request.request_id.as_deref(), Some("request-1"));
        assert_eq!(request.priority, -2);
        assert_eq!(request.cache_salt.as_deref(), Some("salt"));

        let value = serde_json::to_value(request).unwrap();
        assert_eq!(value["mm_processor_kwargs"]["do_resize"], false);
        assert_eq!(value["priority"], -2);
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
