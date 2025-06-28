// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

//! # Dynamo LLM Protocols
//!
//! This module contains the Dynamo-specific protocols for LLM inference.
//! These protocols are designed to work with pre-tokenized input and bypass
//! text preprocessing steps.

use super::{
    common::{SamplingOptionsProvider, StopConditionsProvider},
    openai::{
        nvext::{NvExt, NvExtProvider},
        OpenAISamplingOptionsProvider, OpenAIStopConditionsProvider,
    },
    TokenIdType,
};
use crate::protocols::common::llm_backend::{BackendOutput, PreprocessedRequest};
use crate::protocols::openai::DeltaGeneratorExt;
use async_stream::stream;
use dynamo_runtime::engine::ResponseStream;
use dynamo_runtime::pipeline::{async_trait, AsyncEngine, Error, ManyOut, Operator, SingleIn};
use dynamo_runtime::protocols::annotated::Annotated;
use dynamo_runtime::protocols::annotated::AnnotationsProvider;
use futures::StreamExt;
use serde::{Deserialize, Serialize};
use std::sync::Arc;

/// Token-based completion request - bypasses all preprocessing
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct DynamoTokenCompletionRequest {
    /// Pre-tokenized input - goes directly to backend
    pub token_ids: Vec<TokenIdType>,

    /// Model name
    pub model: String,

    /// Maximum tokens to generate
    pub max_tokens: Option<u32>,

    /// Sampling temperature
    pub temperature: Option<f32>,

    /// Top-p sampling
    pub top_p: Option<f32>,

    /// Whether to stream responses
    pub stream: Option<bool>,

    /// Stop sequences
    pub stop: Option<Vec<String>>,

    /// Frequency penalty
    pub frequency_penalty: Option<f32>,

    /// Presence penalty
    pub presence_penalty: Option<f32>,

    /// NVIDIA extensions
    #[serde(skip_serializing_if = "Option::is_none")]
    pub nvext: Option<NvExt>,
}

// Implement required traits for compatibility
impl NvExtProvider for DynamoTokenCompletionRequest {
    fn nvext(&self) -> Option<&NvExt> {
        self.nvext.as_ref()
    }

    fn raw_prompt(&self) -> Option<String> {
        None // No raw prompt for token input
    }
}

impl AnnotationsProvider for DynamoTokenCompletionRequest {
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

impl OpenAISamplingOptionsProvider for DynamoTokenCompletionRequest {
    fn get_temperature(&self) -> Option<f32> {
        self.temperature
    }
    fn get_top_p(&self) -> Option<f32> {
        self.top_p
    }
    fn get_frequency_penalty(&self) -> Option<f32> {
        self.frequency_penalty
    }
    fn get_presence_penalty(&self) -> Option<f32> {
        self.presence_penalty
    }
    fn nvext(&self) -> Option<&NvExt> {
        self.nvext.as_ref()
    }
}

impl OpenAIStopConditionsProvider for DynamoTokenCompletionRequest {
    fn get_max_tokens(&self) -> Option<u32> {
        self.max_tokens
    }
    fn get_min_tokens(&self) -> Option<u32> {
        None
    }
    fn get_stop(&self) -> Option<Vec<String>> {
        self.stop.clone()
    }
    fn nvext(&self) -> Option<&NvExt> {
        self.nvext.as_ref()
    }
}

/// Direct converter from token request to preprocessed request
impl TryFrom<DynamoTokenCompletionRequest> for PreprocessedRequest {
    type Error = anyhow::Error;

    fn try_from(request: DynamoTokenCompletionRequest) -> Result<Self, Self::Error> {
        if request.token_ids.is_empty() {
            return Err(anyhow::anyhow!("token_ids cannot be empty"));
        }

        let stop_conditions = request.extract_stop_conditions()?;
        let sampling_options = request.extract_sampling_options()?;

        let annotations = request.annotations().unwrap_or_default();

        let mut builder = PreprocessedRequest::builder();
        builder.token_ids(request.token_ids);
        builder.sampling_options(sampling_options);
        builder.stop_conditions(stop_conditions);
        builder.annotations(annotations);
        builder.mdc_sum(None);
        builder.estimated_prefix_hit_num_blocks(None);

        Ok(builder.build()?)
    }
}

// Reuse OpenAI response types - no custom response code needed
pub use crate::protocols::openai::completions::CompletionResponse;

/// Simple converter from DynamoTokenCompletionRequest to PreprocessedRequest
pub struct TokenConverter;

#[async_trait]
impl
    Operator<
        SingleIn<DynamoTokenCompletionRequest>,
        ManyOut<Annotated<CompletionResponse>>,
        SingleIn<PreprocessedRequest>,
        ManyOut<Annotated<BackendOutput>>,
    > for TokenConverter
{
    async fn generate(
        &self,
        request: SingleIn<DynamoTokenCompletionRequest>,
        next: Arc<
            dyn AsyncEngine<
                SingleIn<PreprocessedRequest>,
                ManyOut<Annotated<BackendOutput>>,
                Error,
            >,
        >,
    ) -> Result<ManyOut<Annotated<CompletionResponse>>, Error> {
        let (token_request, context) = request.into_parts();

        // Convert using existing TryFrom
        let preprocessed: PreprocessedRequest = token_request
            .try_into()
            .map_err(|e: anyhow::Error| Error::from(e))?;

        let preprocessed_request = context.map(|_| preprocessed);

        // Forward to backend
        let mut backend_stream = next.generate(preprocessed_request).await?;
        let context = backend_stream.context(); // Extract context before moving
        let mut delta_generator = crate::protocols::openai::completions::DeltaGenerator::new(
            "model".to_string(),
            Default::default(),
        );

        // Convert backend output to completion responses using async stream
        let output = stream! {
            while let Some(annotated) = backend_stream.next().await {
                match annotated.data {
                    Some(backend_output) => {
                        match delta_generator.choice_from_postprocessor(backend_output) {
                            Ok(response) => {
                                yield Annotated {
                                    data: Some(response),
                                    id: annotated.id,
                                    event: annotated.event,
                                    comment: annotated.comment,
                                };
                            }
                            Err(_) => {
                                yield Annotated {
                                    data: None,
                                    id: annotated.id,
                                    event: Some("error".to_string()),
                                    comment: Some(vec!["Conversion failed".to_string()]),
                                };
                            }
                        }
                    }
                    None => {
                        yield Annotated {
                            data: None,
                            id: annotated.id,
                            event: annotated.event,
                            comment: annotated.comment,
                        };
                    }
                }
            }
        };

        Ok(ResponseStream::new(Box::pin(output), context))
    }
}
