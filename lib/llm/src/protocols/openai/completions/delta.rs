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

use super::{CompletionChoice, CompletionRequest, CompletionResponse};
use crate::protocols::common;
use crate::protocols::openai::CompletionUsage;

impl CompletionRequest {
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
    usage: CompletionUsage,

    options: DeltaGeneratorOptions,
}

impl DeltaGenerator {
    pub fn new(model: String, options: DeltaGeneratorOptions) -> Self {
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs();

        Self {
            id: format!("cmpl-{}", uuid::Uuid::new_v4()),
            object: "text_completion".to_string(),
            created: now,
            model,
            system_fingerprint: None,
            usage: CompletionUsage::default(),
            options,
        }
    }

    pub fn update_isl(&mut self, isl: i32) {
        self.usage.prompt_tokens = isl;
    }

    pub fn create_choice(
        &self,
        index: u64,
        text: Option<String>,
        finish_reason: Option<String>,
    ) -> CompletionResponse {
        // todo - update for tool calling

        CompletionResponse {
            id: self.id.clone(),
            object: self.object.clone(),
            created: self.created,
            model: self.model.clone(),
            system_fingerprint: self.system_fingerprint.clone(),
            choices: vec![CompletionChoice {
                text: text.unwrap_or_default(),
                index,
                finish_reason,
                logprobs: None,
            }],
            usage: if self.options.enable_usage {
                Some(self.usage.clone())
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
            self.usage.completion_tokens += delta.token_ids.len() as i32;
        }

        // todo logprobs

        let finish_reason = match delta.finish_reason {
            Some(common::FinishReason::EoS) => Some("stop".to_string()),
            Some(common::FinishReason::Stop) => Some("stop".to_string()),
            Some(common::FinishReason::Length) => Some("length".to_string()),
            Some(common::FinishReason::Cancelled) => Some("cancelled".to_string()),
            Some(common::FinishReason::Error(err_msg)) => {
                return Err(anyhow::anyhow!(err_msg));
            }
            None => None,
        };

        // create choice
        let index = 0;
        Ok(self.create_choice(index, delta.text, finish_reason))
    }

    // TODO: This is a hack. Change `prompt_tokens` to u32
    fn get_isl(&self) -> Option<u32> {
        Some(self.usage.prompt_tokens as u32)
    }
}
