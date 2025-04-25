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

//! The Preprocessor consists of the following modules
//!
//! - `translation`: This module converts the allowed Ingress message types to the corresponding
//!   internal representation.
//! - `apply`: This module applies ModelConfig defaults to any empty optional fields specified
//! - `prompt`: This module applies any prompt template logic to the internal Request object.
//! - `tokenize`: This module tokenizes the formatted prompt string and returns the token ids.
//!
//! The Preprocessor will accept any IngressRequest and transform it to a BackendRequest.

pub mod prompt;
pub mod tools;

use anyhow::Result;
use futures::stream::{self, StreamExt};
use prompt::OAIPromptFormatter;
use std::{collections::HashMap, sync::Arc};
use tracing;

use crate::model_card::model::{ModelDeploymentCard, ModelInfo, TokenizerKind};
use crate::preprocessor::prompt::OAIChatLikeRequest;
use crate::tokenizers::Encoding;

use dynamo_runtime::engine::{AsyncEngine, AsyncEngineContextProvider, ResponseStream};
use dynamo_runtime::pipeline::{
    async_trait, AsyncEngineContext, Error, ManyOut, Operator, SingleIn,
};
use dynamo_runtime::protocols::annotated::{Annotated, AnnotationsProvider};

use crate::protocols::{
    common::{SamplingOptionsProvider, StopConditionsProvider},
    openai::{
        chat_completions::{NvCreateChatCompletionRequest, NvCreateChatCompletionStreamResponse},
        completions::{CompletionRequest, CompletionResponse},
        nvext::NvExtProvider,
        DeltaGeneratorExt,
    },
};
use crate::tokenizers::{traits::Tokenizer, HuggingFaceTokenizer};

use crate::preprocessor::prompt::PromptFormatter;

pub use crate::protocols::common::llm_backend::{BackendInput, BackendOutput};

pub const ANNOTATION_FORMATTED_PROMPT: &str = "formatted_prompt";
pub const ANNOTATION_TOKEN_IDS: &str = "token_ids";

pub struct OpenAIPreprocessor {
    mdcsum: String,
    formatter: Arc<dyn OAIPromptFormatter>,
    tokenizer: Arc<dyn Tokenizer>,
    model_info: Arc<dyn ModelInfo>,
}

impl OpenAIPreprocessor {
    pub async fn new(mdc: ModelDeploymentCard) -> Result<Arc<Self>> {
        let mdcsum = mdc.mdcsum();
        let formatter = PromptFormatter::from_mdc(mdc.clone()).await?;
        let PromptFormatter::OAI(formatter) = formatter;

        let tokenizer = match &mdc.tokenizer {
            Some(TokenizerKind::HfTokenizerJson(file)) => HuggingFaceTokenizer::from_file(file)?,
            Some(TokenizerKind::GGUF(tokenizer)) => {
                HuggingFaceTokenizer::from_tokenizer(*tokenizer.clone())
            }
            None => {
                anyhow::bail!(
                    "Blank ModelDeploymentCard cannot be used for pre-processing, no tokenizer"
                );
            }
        };
        let tokenizer = Arc::new(tokenizer);

        let Some(model_info) = mdc.model_info else {
            anyhow::bail!(
                "Blank ModelDeploymentCard cannot be used for pre-processing, no model_info"
            );
        };
        let model_info = model_info.get_model_info().await?;

        Ok(Arc::new(Self {
            formatter,
            tokenizer,
            model_info,
            mdcsum,
        }))
    }

    /// Encode a string to it's tokens
    pub fn tokenize(&self, s: &str) -> anyhow::Result<Encoding> {
        self.tokenizer.encode(s)
    }

    /// Translate a [`NvCreateChatCompletionRequest`] request to a common completion request.
    /// Returns both the common completion request and a hashmap of annotations.
    ///
    /// Annotations evaluated by this method include:
    /// - `formatted_prompt`
    /// - `token_ids`
    pub fn preprocess_request<
        R: OAIChatLikeRequest
            + AnnotationsProvider
            + SamplingOptionsProvider
            + StopConditionsProvider
            + NvExtProvider,
    >(
        &self,
        request: &R,
    ) -> Result<(BackendInput, HashMap<String, String>)> {
        let mut annotations = HashMap::new();
        let mut builder = BackendInput::builder();

        let use_raw_prompt = request
            .nvext()
            .is_some_and(|ext| ext.use_raw_prompt.unwrap_or(false));

        let formatted_prompt = if use_raw_prompt {
            match request.raw_prompt() {
                Some(prompt) => prompt,
                None => {
                    tracing::warn!("Raw prompt requested but not available");
                    self.formatter.render(request)?
                }
            }
        } else {
            self.formatter.render(request)?
        };

        let encoding = tokio::task::block_in_place(|| self.tokenizer.encode(&formatted_prompt))?;

        if request.has_annotation(ANNOTATION_FORMATTED_PROMPT) {
            annotations.insert(ANNOTATION_FORMATTED_PROMPT.to_string(), formatted_prompt);
        }

        if request.has_annotation(ANNOTATION_TOKEN_IDS) {
            annotations.insert(
                ANNOTATION_TOKEN_IDS.to_string(),
                serde_json::to_string(&encoding.token_ids)?,
            );
        }

        let mut stop_conditions = request.extract_stop_conditions()?;
        if let Some(stop_tokens) = &mut stop_conditions.stop_token_ids_hidden {
            for eos_token in self.model_info.eos_token_ids() {
                if !stop_tokens.contains(&eos_token) {
                    stop_tokens.push(eos_token);
                }
            }
        } else {
            stop_conditions.stop_token_ids_hidden = Some(self.model_info.eos_token_ids());
        }

        // apply ignore eos if not already set
        stop_conditions.apply_ignore_eos();

        if !stop_conditions.ignore_eos.unwrap_or(false) {
            builder.eos_token_ids(self.model_info.eos_token_ids());
        }

        builder.token_ids(encoding.token_ids);
        builder.sampling_options(request.extract_sampling_options()?);
        builder.stop_conditions(stop_conditions);
        builder.annotations(request.annotations().unwrap_or_default());
        builder.mdc_sum(Some(self.mdcsum.clone()));

        Ok((builder.build()?, annotations))
    }

    pub fn transform_postprocessor_stream<Resp: Send + Sync + 'static + std::fmt::Debug>(
        stream: ManyOut<Annotated<BackendOutput>>,
        generator: Box<dyn DeltaGeneratorExt<Resp>>,
    ) -> ManyOut<Annotated<Resp>> {
        let context = stream.context();

        struct State<Resp: Send + Sync + 'static + std::fmt::Debug> {
            response_stream: ManyOut<Annotated<BackendOutput>>,
            response_generator: Box<dyn DeltaGeneratorExt<Resp>>,
            context: Arc<dyn AsyncEngineContext>,
            cancelled: bool,
        }

        let state = State {
            response_stream: stream,
            response_generator: generator,
            context: context.clone(),
            cancelled: false,
        };

        // transform the common response stream into a chat response stream
        let stream = stream::unfold(state, |mut inner| {
            async move {
                if let Some(response) = inner.response_stream.next().await {
                    if inner.cancelled {
                        tracing::debug!(
                            request_id = inner.context.id(),
                            "Cancellation issued last message; closing stream"
                        );
                        return None;
                    }

                    tracing::trace!(
                        request_id = inner.context.id(),
                        "Processing common response: {:?}",
                        response
                    );

                    let response = response.map_data(|data| {
                        inner
                            .response_generator
                            .choice_from_postprocessor(data)
                            .inspect_err(|e| {
                                tracing::error!(
                                    request_id = inner.context.id(),
                                    "Error processing common response: {:?}",
                                    e
                                );
                                inner.cancelled = true;
                                inner.context.stop_generating();
                            })
                            .map_err(|e| e.to_string())
                    });

                    tracing::trace!(
                        request_id = inner.context.id(),
                        "OpenAI NvCreateChatCompletionStreamResponse: {:?}",
                        response
                    );

                    Some((response, inner))
                } else {
                    // stream closed with out graceful closure
                    // we did not detect an is_finished/completed message
                    // Ok(None)
                    None
                }
            }
        });

        ResponseStream::new(Box::pin(stream), context)
    }
}

// for pals, we do not want to add the generation prompt to the formatted prompt
// we also need to know if the template support this add_generation_prompt bool
// any prompt template that does not support this should return an error
// oob - we should update any prompt template that does not support this to support it

#[async_trait]
impl
    Operator<
        SingleIn<NvCreateChatCompletionRequest>,
        ManyOut<Annotated<NvCreateChatCompletionStreamResponse>>,
        SingleIn<BackendInput>,
        ManyOut<Annotated<BackendOutput>>,
    > for OpenAIPreprocessor
{
    async fn generate(
        &self,
        request: SingleIn<NvCreateChatCompletionRequest>,
        next: Arc<
            dyn AsyncEngine<SingleIn<BackendInput>, ManyOut<Annotated<BackendOutput>>, Error>,
        >,
    ) -> Result<ManyOut<Annotated<NvCreateChatCompletionStreamResponse>>, Error> {
        // unpack the request
        let (request, context) = request.into_parts();

        // create a response generator
        let response_generator = request.response_generator();
        let mut response_generator = Box::new(response_generator);

        // convert the chat completion request to a common completion request
        let (common_request, annotations) = self.preprocess_request(&request)?;

        // update isl
        response_generator.update_isl(common_request.token_ids.len() as u32);

        // repack the common completion request
        let common_request = context.map(|_| common_request);

        // create a stream of annotations this will be prepend to the response stream
        let annotations: Vec<Annotated<NvCreateChatCompletionStreamResponse>> = annotations
            .into_iter()
            .flat_map(|(k, v)| Annotated::from_annotation(k, &v))
            .collect();
        let annotations_stream = stream::iter(annotations);

        // forward the common completion request to the next operator
        let response_stream = next.generate(common_request).await?;

        // transform the postprocessor stream
        let stream = Self::transform_postprocessor_stream(response_stream, response_generator);
        let context = stream.context();

        // prepend the annotations to the response stream
        let stream = annotations_stream.chain(stream);

        // return the response stream
        Ok(ResponseStream::new(Box::pin(stream), context))
    }
}

#[async_trait]
impl
    Operator<
        SingleIn<CompletionRequest>,
        ManyOut<Annotated<CompletionResponse>>,
        SingleIn<BackendInput>,
        ManyOut<Annotated<BackendOutput>>,
    > for OpenAIPreprocessor
{
    async fn generate(
        &self,
        request: SingleIn<CompletionRequest>,
        next: Arc<
            dyn AsyncEngine<SingleIn<BackendInput>, ManyOut<Annotated<BackendOutput>>, Error>,
        >,
    ) -> Result<ManyOut<Annotated<CompletionResponse>>, Error> {
        // unpack the request
        let (request, context) = request.into_parts();

        // create a response generator
        let response_generator = request.response_generator();
        let mut response_generator = Box::new(response_generator);
        // convert the chat completion request to a common completion request
        let (common_request, annotations) = self.preprocess_request(&request)?;

        // update isl
        response_generator.update_isl(common_request.token_ids.len() as i32);

        // repack the common completion request
        let common_request = context.map(|_| common_request);

        // create a stream of annotations this will be prepend to the response stream
        let annotations: Vec<Annotated<CompletionResponse>> = annotations
            .into_iter()
            .flat_map(|(k, v)| Annotated::from_annotation(k, &v))
            .collect();
        let annotations_stream = stream::iter(annotations);

        // forward the common completion request to the next operator
        let response_stream = next.generate(common_request).await?;

        // transform the postprocessor stream
        let stream = Self::transform_postprocessor_stream(response_stream, response_generator);
        let context = stream.context();

        // prepend the annotations to the response stream
        let stream = annotations_stream.chain(stream);

        // return the response stream
        Ok(ResponseStream::new(Box::pin(stream), context))
    }
}
