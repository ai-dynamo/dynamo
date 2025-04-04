// SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

use derive_builder::Builder;
use serde::{Deserialize, Serialize};
use serde_repr::{Deserialize_repr, Serialize_repr};

pub mod kv;
pub mod outputs;
pub mod stats;

pub use outputs::*;

#[derive(Serialize, Deserialize, Default)]
pub struct SamplingConfig {
    pub beam_width: u32,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_k: Option<u32>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_p: Option<f32>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_p_min: Option<f32>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_p_reset_ids: Option<u32>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_p_decay: Option<f32>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub seed: Option<u32>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f32>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub min_tokens: Option<u32>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub beam_search_diversity_rate: Option<f32>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub repetition_penalty: Option<f32>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub presence_penalty: Option<f32>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub frequency_penalty: Option<f32>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub length_penalty: Option<f32>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub early_stopping: Option<u32>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub no_repeat_ngram_size: Option<u32>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub num_return_sequences: Option<u32>,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct OutputConfig {
    pub return_log_probs: bool,
    pub return_context_logits: bool,
    pub return_generation_logits: bool,
    pub exclude_input_from_output: bool,
    pub return_encoder_output: bool,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct RetentionPriorityAndDuration {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub retention_priority: Option<u32>, // google.protobuf.UInt32Value
    #[serde(skip_serializing_if = "Option::is_none")]
    pub duration_ms: Option<u64>, // google.protobuf.UInt64Value
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct TokenRangeRetentionConfig {
    pub token_start: u32,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub token_end: Option<u32>, // google.protobuf.UInt32Value
    pub priority: u32,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub duration_ms: Option<u64>, // google.protobuf.UInt64Value
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct KvCacheRetentionConfig {
    pub token_range_retention_configs: Vec<TokenRangeRetentionConfig>,
    pub decode_retention_priority: u32,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub decode_duration_ms: Option<u64>, // google.protobuf.UInt64Value
}

#[derive(Serialize, Deserialize, Debug, Clone, Builder)]
pub struct Request {
    pub input_token_ids: Vec<u32>,
    pub max_tokens: u32,
    pub streaming: bool,
    // pub sampling_config: SamplingConfig,
    // pub output_config: OutputConfig,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub end_id: Option<u32>,
    // pub pad_id: Option<u32>, // google.protobuf.UInt32Value
    // pub position_ids: Vec<u32>,
    // pub bad_words: Vec<u32>,
    // pub stop_words: Vec<u32>,
    // pub embedding_bias: Vec<u8>, // bytes
    // // TODO: Add external_draft_tokens_config: ExternalDraftTokensConfig
    // // TODO: Add prompt_tuning_config: PromptTuningConfig
    // // TODO: Add lora_config: LoraConfig
    // // TODO: Add lookahead_config: LookaheadDecodingConfig
    // pub kv_cache_retention_config: KvCacheRetentionConfig,
    // pub logits_post_processor_name: String,
    // pub encoder_input_token_ids: Vec<u32>,
    // pub client_id: Option<u64>, // google.protobuf.UInt64Value
    // pub return_all_generated_tokens: bool,
    // pub priority: f32,
    // pub request_type: u32,
    // // TODO: Add context_phase_params: ContextPhaseParams
    // pub encoder_input_features: Vec<u8>,    // bytes
    // pub encoder_output_length: Option<u32>, // google.protobuf.UInt32Value
    // pub cross_attention_mask: Vec<u8>,      // bytes
    // pub num_return_sequences: u32,
    // // TODO: Add eagle_config: EagleConfig
    // pub skip_cross_attn_blocks: Vec<u8>, // bytes
}

// todo - return a Result
impl Request {
    pub fn new(input_token_ids: Vec<u32>, max_tokens: u32) -> Self {
        RequestBuilder::default()
            .input_token_ids(input_token_ids)
            .max_tokens(max_tokens)
            .streaming(true)
            .build()
            .unwrap()
    }
}

// todo convert to a TryFrom
impl From<dynamo_llm::protocols::common::llm_backend::BackendInput> for Request {
    fn from(input: dynamo_llm::protocols::common::llm_backend::BackendInput) -> Self {
        let request = RequestBuilder::default()
            .input_token_ids(input.token_ids)
            .max_tokens(input.stop_conditions.max_tokens.unwrap_or(16))
            .streaming(true)
            .end_id(input.eos_token_ids.last().cloned())
            .build()
            .unwrap();

        request
    }
}
