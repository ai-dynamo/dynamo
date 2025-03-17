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

use super::nvext::NvExt;
use super::nvext::NvExtProvider;
use super::OpenAISamplingOptionsProvider;
use super::OpenAIStopConditionsProvider;
use dynamo_runtime::protocols::annotated::AnnotationsProvider;
use serde::{Deserialize, Serialize};
use validator::Validate;

mod aggregator;
mod delta;

pub use aggregator::DeltaAggregator;
pub use delta::DeltaGenerator;

/// A request structure for creating a chat completion, extending OpenAI's
/// `CreateChatCompletionRequest` with [`NvExt`] extensions.
///
/// # Fields
/// - `inner`: The base OpenAI chat completion request, embedded using `serde(flatten)`.
/// - `nvext`: The optional NVIDIA extension field. See [`NvExt`] for
///   more details.
#[derive(Serialize, Deserialize, Validate, Debug, Clone)]
pub struct NvCreateChatCompletionRequest {
    #[serde(flatten)]
    pub inner: async_openai::types::CreateChatCompletionRequest,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub nvext: Option<NvExt>,
}

/// A response structure for unary chat completion responses, embedding OpenAI's
/// `CreateChatCompletionResponse`.
///
/// # Fields
/// - `inner`: The base OpenAI unary chat completion response, embedded
///   using `serde(flatten)`.
#[derive(Serialize, Deserialize, Validate, Debug, Clone)]
pub struct NvCreateChatCompletionResponse {
    #[serde(flatten)]
    pub inner: async_openai::types::CreateChatCompletionResponse,
}

/// A response structure for streamed chat completions, embedding OpenAI's
/// `CreateChatCompletionStreamResponse`.
///
/// # Fields
/// - `inner`: The base OpenAI streaming chat completion response, embedded
///   using `serde(flatten)`.
#[derive(Serialize, Deserialize, Validate, Debug, Clone)]
pub struct NvCreateChatCompletionStreamResponse {
    #[serde(flatten)]
    pub inner: async_openai::types::CreateChatCompletionStreamResponse,
}

/// Implements `NvExtProvider` for `NvCreateChatCompletionRequest`,
/// providing access to NVIDIA-specific extensions.
impl NvExtProvider for NvCreateChatCompletionRequest {
    /// Returns a reference to the optional `NvExt` extension, if available.
    fn nvext(&self) -> Option<&NvExt> {
        self.nvext.as_ref()
    }

    /// Returns `None`, as raw prompt extraction is not implemented.
    fn raw_prompt(&self) -> Option<String> {
        None
    }
}

/// Implements `AnnotationsProvider` for `NvCreateChatCompletionRequest`,
/// enabling retrieval and management of request annotations.
impl AnnotationsProvider for NvCreateChatCompletionRequest {
    /// Retrieves the list of annotations from `NvExt`, if present.
    fn annotations(&self) -> Option<Vec<String>> {
        self.nvext
            .as_ref()
            .and_then(|nvext| nvext.annotations.clone())
    }

    /// Checks whether a specific annotation exists in the request.
    ///
    /// # Arguments
    /// * `annotation` - A string slice representing the annotation to check.
    ///
    /// # Returns
    /// `true` if the annotation exists, `false` otherwise.
    fn has_annotation(&self, annotation: &str) -> bool {
        self.nvext
            .as_ref()
            .and_then(|nvext| nvext.annotations.as_ref())
            .map(|annotations| annotations.contains(&annotation.to_string()))
            .unwrap_or(false)
    }
}

/// Implements `OpenAISamplingOptionsProvider` for `NvCreateChatCompletionRequest`,
/// exposing OpenAI's sampling parameters for chat completion.
impl OpenAISamplingOptionsProvider for NvCreateChatCompletionRequest {
    /// Retrieves the temperature parameter for sampling, if set.
    fn get_temperature(&self) -> Option<f32> {
        self.inner.temperature
    }

    /// Retrieves the top-p (nucleus sampling) parameter, if set.
    fn get_top_p(&self) -> Option<f32> {
        self.inner.top_p
    }

    /// Retrieves the frequency penalty parameter, if set.
    fn get_frequency_penalty(&self) -> Option<f32> {
        self.inner.frequency_penalty
    }

    /// Retrieves the presence penalty parameter, if set.
    fn get_presence_penalty(&self) -> Option<f32> {
        self.inner.presence_penalty
    }

    /// Returns a reference to the optional `NvExt` extension, if available.
    fn nvext(&self) -> Option<&NvExt> {
        self.nvext.as_ref()
    }
}

/// Implements `OpenAIStopConditionsProvider` for `NvCreateChatCompletionRequest`,
/// providing access to stop conditions that control chat completion behavior.
impl OpenAIStopConditionsProvider for NvCreateChatCompletionRequest {
    /// Retrieves the maximum number of tokens allowed in the response.
    #[allow(deprecated)]
    fn get_max_tokens(&self) -> Option<u32> {
        self.inner.max_completion_tokens.or(self.inner.max_tokens)
    }

    /// Retrieves the minimum number of tokens required in the response.
    ///
    /// # Note
    /// This method is currently a placeholder and always returns `None`
    /// since `min_tokens` is not an OpenAI-supported parameter.
    fn get_min_tokens(&self) -> Option<u32> {
        None
    }

    /// Retrieves the stop conditions that terminate the chat completion response.
    ///
    /// Converts OpenAI's `Stop` enum to a `Vec<String>`, normalizing the representation.
    ///
    /// # Returns
    /// * `Some(Vec<String>)` if stop conditions are set.
    /// * `None` if no stop conditions are defined.
    fn get_stop(&self) -> Option<Vec<String>> {
        self.inner.stop.as_ref().map(|stop| match stop {
            async_openai::types::Stop::String(s) => vec![s.clone()],
            async_openai::types::Stop::StringArray(arr) => arr.clone(),
        })
    }

    /// Returns a reference to the optional `NvExt` extension, if available.
    fn nvext(&self) -> Option<&NvExt> {
        self.nvext.as_ref()
    }
}
