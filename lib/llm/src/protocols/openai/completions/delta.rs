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

use super::{CompletionResponse, NvCreateCompletionRequest};
use crate::protocols::common;

impl NvCreateCompletionRequest {
    // put this method on the request
    // inspect the request to extract options
    pub fn response_generator(&self) -> DeltaGenerator {
        let options = DeltaGeneratorOptions {
            enable_usage: true,
            enable_logprobs: false,
        };

        DeltaGenerator::new(self.inner.model.clone(), options)
    }
}

#[derive(Debug, Clone, Default)]
pub struct DeltaGeneratorOptions {
    pub enable_usage: bool,
    pub enable_logprobs: bool,
}

#[derive(Debug, Clone)]
pub struct DeltaGenerator {
    id: String,
    object: String,
    created: u64,
    model: String,
    system_fingerprint: Option<String>,
    usage: async_openai::types::CompletionUsage,
    options: DeltaGeneratorOptions,
}

impl DeltaGenerator {
    pub fn new(model: String, options: DeltaGeneratorOptions) -> Self {
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs();

        // Previously, our home-rolled CompletionUsage impl'd Default
        // PR !387 - https://github.com/64bit/async-openai/pull/387
        let usage = async_openai::types::CompletionUsage {
            completion_tokens: 0,
            prompt_tokens: 0,
            total_tokens: 0,
            completion_tokens_details: None,
            prompt_tokens_details: None,
        };

        Self {
            id: format!("cmpl-{}", uuid::Uuid::new_v4()),
            object: "text_completion".to_string(),
            created: now,
            model,
            system_fingerprint: None,
            usage,
            options,
        }
    }

    pub fn update_isl(&mut self, isl: u32) {
        self.usage.prompt_tokens = isl;
    }

    pub fn create_choice(
        &self,
        index: u64,
        text: Option<String>,
        finish_reason: Option<async_openai::types::CompletionFinishReason>,
    ) -> CompletionResponse {
        // todo - update for tool calling

        let mut usage = self.usage.clone();
        if self.options.enable_usage {
            usage.total_tokens = usage.prompt_tokens + usage.completion_tokens;
        }

        CompletionResponse {
            id: self.id.clone(),
            object: self.object.clone(),
            created: self.created,
            model: self.model.clone(),
            system_fingerprint: self.system_fingerprint.clone(),
            choices: vec![async_openai::types::Choice {
                text: text.unwrap_or_default(),
                index: index as u32,
                finish_reason,
                logprobs: None,
            }],
            usage: if self.options.enable_usage {
                Some(usage)
            } else {
                None
            },
        }
    }
}

impl crate::protocols::openai::DeltaGeneratorExt<CompletionResponse> for DeltaGenerator {
    fn choice_from_postprocessor(
        &mut self,
        delta: common::llm_backend::BackendOutput,
    ) -> anyhow::Result<CompletionResponse> {
        // aggregate usage
        if self.options.enable_usage {
            self.usage.completion_tokens += delta.token_ids.len() as u32;
        }

        // TODO logprobs

        let finish_reason = delta.finish_reason.map(Into::into);

        // create choice
        let index = delta.index.unwrap_or(0).into();
        let response = self.create_choice(index, delta.text.clone(), finish_reason);
        Ok(response)
    }

    fn get_isl(&self) -> Option<u32> {
        Some(self.usage.prompt_tokens)
    }
}
