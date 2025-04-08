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

use super::*;
use dynamo_llm::protocols::{
    common::{self},
    TokenIdType,
};

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct Responses {
    pub responses: Vec<Response>,
    pub shutdown: bool,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct Response {
    pub request_id: u64,
    pub client_id: Option<u64>, // Optional client ID.

    pub error_msg: Option<String>, // Error message if the request failed.
    pub output: Option<Output>,    // Output if the request succeeded.
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct Output {
    pub is_final: bool,

    pub token_ids: Vec<TokenIdType>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub cum_log_prob: Option<f64>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub log_probs: Option<Vec<f64>>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub finish_reason: Option<FinishReasonEnum>,
}

#[derive(Serialize_repr, Deserialize_repr, Debug, Clone)]
#[repr(u8)]
pub enum FinishReasonEnum {
    FinishReasonNotDone = 0,
    FinishReasonEos = 1,
    FinishReasonStop = 2,
    FinishReasonLength = 3,
}

impl From<Output> for common::llm_backend::LLMEngineOutput {
    fn from(output: Output) -> Self {
        let finish_reason = match output.finish_reason {
            Some(FinishReasonEnum::FinishReasonNotDone) => None,
            Some(FinishReasonEnum::FinishReasonEos) => Some(common::FinishReason::EoS),
            Some(FinishReasonEnum::FinishReasonStop) => Some(common::FinishReason::Stop),
            Some(FinishReasonEnum::FinishReasonLength) => Some(common::FinishReason::Length),
            None => None,
        };

        common::llm_backend::LLMEngineOutput {
            // todo - propagate mdcsum
            token_ids: output.token_ids,
            tokens: None,
            text: None,
            cum_log_probs: output.cum_log_prob,
            log_probs: None,
            finish_reason,
        }
    }
}
