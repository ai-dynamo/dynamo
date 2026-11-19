// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::collections::HashMap;

use dynamo_backend_common::{EngineConfig, LlmRegistration};
use dynamo_llm::local_model::runtime_config::{
    REJECT_INCOMPLETE_NATIVE_TOOL_CALLS_CAPABILITY, REQUIRE_CLOSED_REASONING_PREFILL_CAPABILITY,
    RESPONSES_CONTEXT_OVERFLOW_EVENTS_CAPABILITY, TOKEN_BUDGET_RUNTIME_KEY, TokenBudget,
    VISIBLE_STOP_STRINGS_CAPABILITY,
};

/// Model identity and registration metadata for the TensorRT-LLM sidecar.
#[derive(Clone, Debug, Default)]
pub(crate) struct ConfiguredModel {
    /// HF repo name or local path used for tokenization and templates.
    pub source: String,
    /// Maximum sequence length (input + output), from the `--context-length`
    /// argument if it was supplied, else from a server `GetModelInfo` report,
    /// if known.
    pub context_length: Option<u32>,
    pub kv_cache_block_size: Option<u32>,
    pub total_kv_blocks: Option<u64>,
    pub max_num_seqs: Option<u64>,
    pub max_num_batched_tokens: Option<u64>,
    pub data_parallel_size: Option<u32>,
    pub data_parallel_start_rank: Option<u32>,
}

impl ConfiguredModel {
    pub(crate) fn engine_config(&self) -> EngineConfig {
        let mut runtime_data = HashMap::new();
        runtime_data.insert(
            "grpc_service".to_string(),
            serde_json::Value::String("openengine.v1.Inference".to_string()),
        );
        if let Some(combined_limit) = self.context_length {
            // TensorRT-LLM rejects prompt overflow but clamps oversized output budgets.
            let token_budget = TokenBudget {
                combined_limit,
                reject_prompt_overflow: true,
                reject_total_overflow: false,
            };
            runtime_data.insert(
                TOKEN_BUDGET_RUNTIME_KEY.to_string(),
                serde_json::to_value(token_budget).expect("token budget is serializable"),
            );
        }

        for capability in [
            VISIBLE_STOP_STRINGS_CAPABILITY,
            REQUIRE_CLOSED_REASONING_PREFILL_CAPABILITY,
            REJECT_INCOMPLETE_NATIVE_TOOL_CALLS_CAPABILITY,
            RESPONSES_CONTEXT_OVERFLOW_EVENTS_CAPABILITY,
        ] {
            runtime_data.insert(capability.to_string(), serde_json::Value::Bool(true));
        }

        EngineConfig {
            model: self.source.clone(),
            served_model_name: None,
            model_aliases: Vec::new(),
            runtime_data,
            llm: Some(LlmRegistration {
                context_length: self.context_length,
                kv_cache_block_size: self.kv_cache_block_size,
                total_kv_blocks: self.total_kv_blocks_per_rank(),
                max_num_seqs: self.max_num_seqs,
                max_num_batched_tokens: self.max_num_batched_tokens,
                data_parallel_size: self.data_parallel_size,
                data_parallel_start_rank: self.data_parallel_start_rank,
                ..Default::default()
            }),
        }
    }

    pub(crate) fn data_parallel_size(&self) -> u32 {
        self.data_parallel_size.unwrap_or(1).max(1)
    }

    fn total_kv_blocks_per_rank(&self) -> Option<u64> {
        let total = self.total_kv_blocks.filter(|value| *value > 0)?;
        let ranks = u64::from(self.data_parallel_size());
        let per_rank = total / ranks;
        (per_rank > 0).then_some(per_rank)
    }
}
