// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! The Preprocessor consists of the following modules
//!
//! - `translation`: This module converts the allowed Ingress message types to the corresponding
//!   internal representation.
//! - `apply`: This module applies ModelConfig defaults to any empty optional fields specified
//! - `prompt`: This module applies any prompt template logic to the internal Request object.
//! - `tokenize`: This module tokenizes the formatted prompt string and returns the token ids.
//!
//! The Preprocessor will accept any IngressRequest and transform it to a BackendRequest.

pub mod media;
pub mod prompt;
pub mod tools;
use anyhow::Context;
use anyhow::{Result, bail};
use dynamo_async_openai::types::{
    ChatCompletionRequestMessage, ChatCompletionRequestUserMessageContent,
    ChatCompletionRequestUserMessageContentPart, ChatCompletionToolChoiceOption, EncodingFormat,
};
use futures::Stream;
use futures::stream::{self, StreamExt};
use prompt::OAIPromptFormatter;
use std::{collections::HashMap, pin::Pin, sync::Arc};
use tracing;

use crate::model_card::{ModelDeploymentCard, ModelInfo};
#[cfg(feature = "media-nixl")]
use crate::preprocessor::media::MediaLoader;
use crate::preprocessor::prompt::OAIChatLikeRequest;
use crate::protocols::common::preprocessor::{
    MultimodalData, MultimodalDataMap, PreprocessedRequestBuilder, RoutingHints,
};
use crate::tokenizers::Encoding;

use dynamo_parsers::{ReasoningParser, ReasoningParserType};
use dynamo_runtime::engine::{AsyncEngine, AsyncEngineContextProvider, ResponseStream};
use dynamo_runtime::pipeline::{
    AsyncEngineContext, Error, ManyOut, Operator, SingleIn, async_trait,
};
use dynamo_runtime::protocols::annotated::{Annotated, AnnotationsProvider};

use crate::protocols::{
    common::{OutputOptionsProvider, SamplingOptionsProvider, StopConditionsProvider},
    openai::{
        DeltaGeneratorExt,
        chat_completions::{
            NvCreateChatCompletionRequest, NvCreateChatCompletionStreamResponse, jail::JailedStream,
        },
        completions::{NvCreateCompletionRequest, NvCreateCompletionResponse},
        embeddings::{NvCreateEmbeddingRequest, NvCreateEmbeddingResponse},
        nvext::NvExtProvider,
    },
};
use crate::tokenizers::{HuggingFaceTokenizer, traits::Tokenizer};

use crate::preprocessor::prompt::{PromptFormatter, PromptInput, TextInput, TokenInput};

pub use crate::protocols::common::llm_backend::{BackendOutput, PreprocessedRequest};
pub use crate::protocols::common::preprocessor::PreprocessedEmbeddingRequest;

use crate::protocols::common::llm_backend::EmbeddingsEngineOutput;

pub const ANNOTATION_FORMATTED_PROMPT: &str = "formatted_prompt";
pub const ANNOTATION_TOKEN_IDS: &str = "token_ids";
pub const ANNOTATION_LLM_METRICS: &str = "llm_metrics";
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct LLMMetricAnnotation {
    pub input_tokens: usize,
    pub output_tokens: usize,
    pub chunk_tokens: usize,
    pub cached_tokens: Option<usize>,
}

impl LLMMetricAnnotation {
    /// Convert this metrics struct to an Annotated event
    pub fn to_annotation<T>(&self) -> Result<Annotated<T>, serde_json::Error> {
        Annotated::from_annotation(ANNOTATION_LLM_METRICS, self)
    }

    /// Extract LLM metrics from an Annotated event, if present
    pub fn from_annotation<T>(
        annotation: &Annotated<T>,
    ) -> Result<Option<LLMMetricAnnotation>, Box<dyn std::error::Error>> {
        if annotation.event.is_none() {
            return Ok(None);
        }
        if annotation.event.as_ref().unwrap() != ANNOTATION_LLM_METRICS {
            return Ok(None);
        }
        let comments = annotation
            .comment
            .as_ref()
            .ok_or("missing comments block")?;
        if comments.len() != 1 {
            return Err("malformed comments block - expected exactly 1 comment".into());
        }
        let metrics: LLMMetricAnnotation = serde_json::from_str(&comments[0])?;
        Ok(Some(metrics))
    }
}

// Reasoning State for reasoning parsing transformation step
struct ReasoningState {
    stream: Pin<Box<dyn Stream<Item = Annotated<NvCreateChatCompletionStreamResponse>> + Send>>,
    reasoning_parser: Option<Box<dyn ReasoningParser>>,
}

pub struct OpenAIPreprocessor {
    mdcsum: String,
    formatter: Arc<dyn OAIPromptFormatter>,
    tokenizer: Arc<dyn Tokenizer>,
    model_info: Arc<dyn ModelInfo>,
    lora_name: Option<String>,
    /// Per-model runtime configuration propagated to response generator (e.g., reasoning/tool parser)
    runtime_config: crate::local_model::runtime_config::ModelRuntimeConfig,
    tool_call_parser: Option<String>,
    #[cfg(feature = "media-nixl")]
    media_loader: Option<MediaLoader>,
}

impl OpenAIPreprocessor {
    pub fn new(mdc: ModelDeploymentCard) -> Result<Arc<Self>> {
        let formatter = PromptFormatter::from_mdc(&mdc)?;
        let tokenizer = mdc.tokenizer_hf()?;
        match formatter {
            PromptFormatter::OAI(formatter) => Self::new_with_parts(mdc, formatter, tokenizer),
        }
    }

    pub fn new_with_parts(
        mdc: ModelDeploymentCard,
        formatter: Arc<dyn OAIPromptFormatter>,
        hf_tokenizer: tokenizers::Tokenizer,
    ) -> Result<Arc<Self>> {
        let mdcsum = mdc.mdcsum().to_string();
        let tokenizer = Arc::new(HuggingFaceTokenizer::from_tokenizer(hf_tokenizer));
        let lora_name = mdc.lora_name.clone();
        let Some(ref model_info) = mdc.model_info else {
            anyhow::bail!(
                "Blank ModelDeploymentCard cannot be used for pre-processing, no model_info"
            );
        };
        let model_info = model_info.get_model_info()?;
        let tool_call_parser = mdc.runtime_config.tool_call_parser.clone();

        if let Some(ref lora_name) = lora_name {
            tracing::info!(model = %mdc.display_name, lora_name, "LoRA adapter detected in MDC");
        }

        // // Initialize runtime config from the ModelDeploymentCard
        let runtime_config = mdc.runtime_config.clone();

        #[cfg(feature = "media-nixl")]
        let media_loader = match mdc.media_decoder {
            Some(media_decoder) => Some(MediaLoader::new(media_decoder, mdc.media_fetcher)?),
            None => None,
        };

        Ok(Arc::new(Self {
            formatter,
            tokenizer,
            model_info,
            mdcsum,
            lora_name,
            runtime_config,
            tool_call_parser,
            #[cfg(feature = "media-nixl")]
            media_loader,
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
    pub async fn preprocess_request<
        R: OAIChatLikeRequest
            + AnnotationsProvider
            + SamplingOptionsProvider
            + StopConditionsProvider
            + OutputOptionsProvider
            + NvExtProvider,
    >(
        &self,
        request: &R,
    ) -> Result<(PreprocessedRequest, HashMap<String, String>)> {
        let mut builder = self.builder(request)?;
        let formatted_prompt = self
            .apply_template(request)
            .with_context(|| "Failed to apply prompt template")?;
        let annotations = self
            .gather_tokens(request, &mut builder, formatted_prompt)
            .with_context(|| "Failed to gather tokens")?;
        self.gather_multi_modal_data(request, &mut builder)
            .await
            .with_context(|| "Failed to gather multimodal data")?;

        Ok((builder.build()?, annotations))
    }

    pub fn builder<
        R: OAIChatLikeRequest
            + AnnotationsProvider
            + SamplingOptionsProvider
            + StopConditionsProvider
            + OutputOptionsProvider
            + NvExtProvider,
    >(
        &self,
        request: &R,
    ) -> Result<PreprocessedRequestBuilder> {
        let mut builder = PreprocessedRequest::builder();
        builder.model(request.model());

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

        builder.stop_conditions(stop_conditions);
        builder.sampling_options(request.extract_sampling_options()?);
        builder.output_options(request.extract_output_options()?);
        builder.annotations(request.annotations().unwrap_or_default());
        builder.mdc_sum(Some(self.mdcsum.clone()));
        let lora_name = self.lora_name.clone();

        // Extract routing hints from nvext if present
        if let Some(nvext) = request.nvext() {
            // Build routing hints from nvext fields
            let routing = RoutingHints {
                backend_instance_id: nvext.backend_instance_id,
                prefill_worker_id: nvext.prefill_worker_id,
                decode_worker_id: nvext.decode_worker_id,
                dp_rank: None, // dp_rank is set later in the pipeline
                enable_local_updates: nvext.enable_local_updates,
                expected_output_tokens: nvext.expected_output_tokens,
                lora_name,
            };
            builder.routing(Some(routing));
        } else if lora_name.is_some() {
            // Ensure LoRA-aware routing still gets hints even when nvext is absent.
            builder.routing(Some(RoutingHints {
                lora_name,
                ..Default::default()
            }));
        }

        Ok(builder)
    }

    pub fn apply_template<
        R: OAIChatLikeRequest
            + AnnotationsProvider
            + SamplingOptionsProvider
            + StopConditionsProvider
            + OutputOptionsProvider
            + NvExtProvider,
    >(
        &self,
        request: &R,
    ) -> Result<Option<String>> {
        if let PromptInput::Text(_) = request.prompt_input_type()
            && let Some(TextInput::Single(_)) = request.extract_text()
        {
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
            Ok(Some(formatted_prompt))
        } else {
            Ok(None)
        }
    }

    pub async fn gather_multi_modal_data<R: OAIChatLikeRequest>(
        &self,
        request: &R,
        builder: &mut PreprocessedRequestBuilder,
    ) -> Result<()> {
        let mut media_map: MultimodalDataMap = HashMap::new();
        #[cfg(feature = "media-nixl")]
        let mut fetch_tasks: Vec<(String, ChatCompletionRequestUserMessageContentPart)> =
            Vec::new();

        let Some(messages) = request.typed_messages() else {
            return Ok(());
        };
        for message in messages.iter() {
            let content_parts = match message {
                ChatCompletionRequestMessage::User(u) => match &u.content {
                    ChatCompletionRequestUserMessageContent::Array(parts) => parts,
                    _ => continue,
                },
                _ => continue,
            };
            // Iterate over content parts
            for content_part in content_parts.iter() {
                let (type_str, url) = match content_part {
                    ChatCompletionRequestUserMessageContentPart::ImageUrl(image_part) => {
                        ("image_url".to_string(), image_part.image_url.url.clone())
                    }
                    ChatCompletionRequestUserMessageContentPart::VideoUrl(video_part) => {
                        ("video_url".to_string(), video_part.video_url.url.clone())
                    }
                    ChatCompletionRequestUserMessageContentPart::AudioUrl(audio_part) => {
                        ("audio_url".to_string(), audio_part.audio_url.url.clone())
                    }
                    _ => continue,
                };

                #[cfg(feature = "media-nixl")]
                if self.media_loader.is_some() {
                    fetch_tasks.push((type_str, content_part.clone()));
                    continue;
                }

                //Fallback: ust pass the URL through
                media_map
                    .entry(type_str)
                    .or_default()
                    .push(MultimodalData::Url(url));
            }
        }

        // Execute all fetch tasks
        #[cfg(feature = "media-nixl")]
        if !fetch_tasks.is_empty() {
            let loader = self.media_loader.as_ref().unwrap();
            let media_io_kwargs = request.media_io_kwargs();
            let results = futures::future::join_all(fetch_tasks.iter().map(|(_, content_part)| {
                loader.fetch_and_decode_media_part(content_part, media_io_kwargs)
            }))
            .await;

            for ((type_str, _), result) in fetch_tasks.into_iter().zip(results.into_iter()) {
                // if one item fails, errors the whole request, other items will be cleaned up by Drop
                let rdma_descriptor = result?;
                media_map
                    .entry(type_str)
                    .or_default()
                    .push(MultimodalData::Decoded(rdma_descriptor));
            }
        }

        if !media_map.is_empty() {
            builder.multi_modal_data(Some(media_map));

            // Preserve original messages in extra_args for multimodal workers that need them
            // (e.g., TRT-LLM multimodal processor needs raw messages for proper tokenization)
            let messages_json = serde_json::to_value(request.messages())?;
            let extra_args = serde_json::json!({
                "messages": messages_json
            });
            builder.extra_args(Some(extra_args));
        }

        Ok(())
    }

    pub fn gather_tokens<
        R: OAIChatLikeRequest
            + AnnotationsProvider
            + SamplingOptionsProvider
            + StopConditionsProvider
            + OutputOptionsProvider
            + NvExtProvider,
    >(
        &self,
        request: &R,
        builder: &mut PreprocessedRequestBuilder,
        formatted_prompt: Option<String>,
    ) -> Result<HashMap<String, String>> {
        let mut annotations = HashMap::new();
        // match request type before any conversion/processing
        match request.prompt_input_type() {
            PromptInput::Tokens(_) => {
                if let Some(token_input) = request.extract_tokens() {
                    match token_input {
                        TokenInput::Single(tokens) => {
                            builder.token_ids(tokens);
                        }
                        TokenInput::Batch(token_batches) => {
                            if token_batches.len() == 1 {
                                builder.token_ids(token_batches[0].clone());
                            } else {
                                bail!(
                                    "Batch token input not supported for more than one token in requests (got {})",
                                    token_batches.len()
                                );
                            }
                        }
                    }
                }
            }
            PromptInput::Text(_) => {
                if let Some(text_input) = request.extract_text() {
                    match text_input {
                        TextInput::Single(raw_prompt) => {
                            if let Some(f) = formatted_prompt.as_ref()
                                && request.has_annotation(ANNOTATION_FORMATTED_PROMPT)
                            {
                                annotations
                                    .insert(ANNOTATION_FORMATTED_PROMPT.to_string(), f.to_string());
                            }

                            // Completions will use raw_prompt, no template
                            let prompt = formatted_prompt.unwrap_or(raw_prompt);

                            // Check if backend_instance_id is present and token_data is provided
                            let has_backend_instance_id = request
                                .nvext()
                                .and_then(|ext| ext.backend_instance_id)
                                .is_some();

                            let token_data =
                                request.nvext().and_then(|ext| ext.token_data.as_ref());

                            let (tokens_vec, skip_token_annotation) = if has_backend_instance_id {
                                if let Some(tokens) = token_data {
                                    tracing::trace!(
                                        "Using provided tokens from EPP: {} ids",
                                        tokens.len()
                                    );
                                    // need ownership for the builder, so clone.
                                    (tokens.clone(), true)
                                } else {
                                    tracing::warn!(
                                        "backend_instance_id provided but no token_data; tokenizing prompt"
                                    );
                                    let encoding = self.tokenizer.encode(&prompt)?;
                                    (encoding.token_ids().to_vec(), false)
                                }
                            } else {
                                // No backend_instance_id provided, continue the normal flow.
                                let encoding = self.tokenizer.encode(&prompt)?;
                                (encoding.token_ids().to_vec(), false)
                            };

                            if request.has_annotation(ANNOTATION_TOKEN_IDS)
                                && !skip_token_annotation
                            {
                                annotations.insert(
                                    ANNOTATION_TOKEN_IDS.to_string(),
                                    serde_json::to_string(&tokens_vec)?,
                                );
                            }

                            builder.token_ids(tokens_vec);
                        }
                        TextInput::Batch(texts) => {
                            if texts.len() == 1 {
                                let encoding = self.tokenizer.encode(&texts[0])?;
                                builder.token_ids(encoding.token_ids().to_vec());
                            } else {
                                bail!(
                                    "Batch text input not supported for more than one text in requests (got {})",
                                    texts.len()
                                );
                            }
                        }
                    }
                }
            }
        }
        Ok(annotations)
    }

    /// Preprocess an embedding request, handling both text and token ID inputs.
    ///
    /// For text inputs, tokenizes the text using the configured tokenizer.
    /// For token ID inputs, uses the provided token IDs directly and skips tokenization.
    ///
    /// Returns both the preprocessed request and a hashmap of annotations.
    pub async fn preprocess_embedding_request(
        &self,
        request: &NvCreateEmbeddingRequest,
    ) -> Result<(PreprocessedEmbeddingRequest, HashMap<String, String>)> {
        let mut annotations = HashMap::new();
        let mut builder = PreprocessedEmbeddingRequest::builder();

        let all_token_ids = match &request.inner.input {
            dynamo_async_openai::types::EmbeddingInput::String(s) => {
                let encoding = self.tokenizer.encode(s)?;
                vec![encoding.token_ids().to_vec()]
            }
            dynamo_async_openai::types::EmbeddingInput::StringArray(arr) => {
                let input_strs: Vec<String> = arr.to_vec();
                let encodings = tokio::task::spawn_blocking({
                    let tokenizer = self.tokenizer.clone();
                    let strs = input_strs.clone();
                    move || {
                        tokenizer.encode_batch(&strs.iter().map(|s| s.as_str()).collect::<Vec<_>>())
                    }
                })
                .await??;
                let token_arrays: Vec<Vec<u32>> = encodings
                    .into_iter()
                    .map(|encoding| encoding.token_ids().to_vec())
                    .collect();
                token_arrays
            }
            dynamo_async_openai::types::EmbeddingInput::IntegerArray(token_ids) => {
                vec![token_ids.clone()]
            }
            dynamo_async_openai::types::EmbeddingInput::ArrayOfIntegerArray(token_arrays) => {
                token_arrays.clone()
            }
        };

        // Handle annotations
        if request.has_annotation(ANNOTATION_TOKEN_IDS) {
            annotations.insert(
                ANNOTATION_TOKEN_IDS.to_string(),
                serde_json::to_string(&all_token_ids)?,
            );
        }

        builder.token_ids(all_token_ids);
        builder.model(request.inner.model.clone());
        builder.encoding_format(request.inner.encoding_format.as_ref().map(|f| match f {
            EncodingFormat::Float => "float".to_string(),
            EncodingFormat::Base64 => "base64".to_string(),
        }));
        builder.dimensions(request.inner.dimensions);

        builder.annotations(request.annotations().unwrap_or_default());
        builder.mdc_sum(Some(self.mdcsum.clone()));

        Ok((builder.build()?, annotations))
    }

    pub fn transform_postprocessor_stream<S, Resp>(
        stream: S,
        generator: Box<dyn DeltaGeneratorExt<Resp>>,
        context: Arc<dyn AsyncEngineContext>,
    ) -> impl Stream<Item = Annotated<Resp>> + Send
    where
        S: Stream<Item = Annotated<BackendOutput>> + Send + 'static,
        Resp: Send + Sync + 'static + std::fmt::Debug,
    {
        struct State<Resp>
        where
            Resp: Send + Sync + 'static + std::fmt::Debug,
        {
            response_stream: Pin<Box<dyn Stream<Item = Annotated<BackendOutput>> + Send>>,
            response_generator: Box<dyn DeltaGeneratorExt<Resp>>,
            context: Arc<dyn AsyncEngineContext>,
            cancelled: bool,
            cumulative_output_tokens: usize,
            finish_reason_sent: bool,
            usage_chunk_sent: bool,
            finished: bool,
        }

        let state = State {
            response_stream: Box::pin(stream),
            response_generator: generator,
            context: context.clone(),
            cancelled: false,
            cumulative_output_tokens: 0,
            finish_reason_sent: false,
            usage_chunk_sent: false,
            finished: false,
        };

        // transform the common response stream into a chat response stream

        stream::unfold(state, |mut inner| {
            async move {
                // If already finished, return None immediately
                if inner.finished {
                    return None;
                }

                if let Some(response) = inner.response_stream.next().await {
                    if inner.cancelled {
                        tracing::debug!(
                            request_id = inner.context.id(),
                            "Cancellation issued last message; closing stream"
                        );
                        inner.finished = true; // Mark as finished
                        return None;
                    }

                    tracing::trace!(
                        request_id = inner.context.id(),
                        "Processing common response: {:?}",
                        response
                    );

                    // Check if this response has a finish_reason
                    let has_finish_reason = response
                        .data
                        .as_ref()
                        .map(|d| d.finish_reason.is_some())
                        .unwrap_or(false);

                    let (chunk_tokens, isl) = if let Some(ref backend_output) = response.data {
                        let chunk_tokens = backend_output.token_ids.len();
                        inner.cumulative_output_tokens += chunk_tokens;

                        let isl = inner.response_generator.get_isl().unwrap_or(0) as usize;

                        (chunk_tokens, isl)
                    } else {
                        (0, 0)
                    };

                    let current_osl = inner.cumulative_output_tokens;

                    let mut response = response.map_data(|data| {
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

                    // Create LLM metrics annotation
                    let llm_metrics = LLMMetricAnnotation {
                        input_tokens: isl,
                        output_tokens: current_osl,
                        chunk_tokens,
                        cached_tokens: None,
                    };

                    if let Ok(metrics_annotated) = llm_metrics.to_annotation::<()>() {
                        // Only set event if not already set to avoid overriding existing events (like errors)
                        if response.event.is_none() {
                            response.event = metrics_annotated.event;
                            response.comment = metrics_annotated.comment;
                        }
                    }

                    // Mark if we've seen a finish_reason
                    if has_finish_reason {
                        inner.finish_reason_sent = true;
                    }

                    tracing::trace!(
                        request_id = inner.context.id(),
                        "OpenAI NvCreateChatCompletionStreamResponse: {:?}",
                        response
                    );

                    Some((response, inner))
                } else {
                    // Stream has ended - must set finished to true to prevent unfold from polling
                    // again. The stream is exhausted and will panic if polled after None.
                    inner.finished = true;

                    if inner.finish_reason_sent && !inner.usage_chunk_sent {
                        inner.usage_chunk_sent = true;

                        let usage_chunk = inner.response_generator.create_usage_chunk();
                        let usage = inner.response_generator.get_usage();
                        let llm_metrics = LLMMetricAnnotation {
                            input_tokens: usage.prompt_tokens as usize,
                            output_tokens: usage.completion_tokens as usize,
                            chunk_tokens: 0,
                            cached_tokens: usage
                                .prompt_tokens_details
                                .as_ref()
                                .and_then(|d| d.cached_tokens.map(|c| c as usize)),
                        };

                        // Create annotation string
                        let annotation = llm_metrics.to_annotation::<()>().unwrap_or_else(|e| {
                            tracing::warn!("Failed to serialize metrics: {}", e);
                            Annotated::<()>::from_data(())
                        });

                        // Send the usage chunk if needed
                        let data = if inner.response_generator.is_usage_enabled() {
                            Some(usage_chunk)
                        } else {
                            None
                        };

                        let annotated_usage = Annotated::<Resp> {
                            id: None,
                            data,
                            event: Some(ANNOTATION_LLM_METRICS.to_string()),
                            comment: annotation.comment,
                        };

                        tracing::trace!(
                            request_id = inner.context.id(),
                            "Sending final usage chunk for OpenAI compliance, annotated_usage: {:?}",
                            annotated_usage
                        );

                        Some((annotated_usage, inner))
                    } else {
                        // stream closed
                        None
                    }
                }
            }
        })
        .fuse()
    }

    /// Transform engine embedding output stream to OpenAI embedding response stream
    pub fn transform_embedding_postprocessor_stream<S>(
        stream: S,
        original_request: NvCreateEmbeddingRequest,
    ) -> impl Stream<Item = Annotated<NvCreateEmbeddingResponse>> + Send
    where
        S: Stream<Item = Annotated<EmbeddingsEngineOutput>> + Send + 'static,
    {
        stream.map(move |output| {
            output.map_data(|engine_output| {
                // Convert engine output to OpenAI response format
                let embeddings: Vec<dynamo_async_openai::types::Embedding> = engine_output
                    .embeddings
                    .into_iter()
                    .enumerate()
                    .map(|(index, embedding)| dynamo_async_openai::types::Embedding {
                        index: index as u32,
                        object: "embedding".to_string(),
                        embedding: embedding.into_iter().map(|f| f as f32).collect(),
                    })
                    .collect();

                let response = NvCreateEmbeddingResponse {
                    inner: dynamo_async_openai::types::CreateEmbeddingResponse {
                        object: "list".to_string(),
                        model: original_request.inner.model.clone(),
                        data: embeddings,
                        usage: dynamo_async_openai::types::EmbeddingUsage {
                            prompt_tokens: engine_output.prompt_tokens,
                            total_tokens: engine_output.total_tokens,
                        },
                    },
                };

                Ok(response)
            })
        })
    }

    /// Determine if we should apply the tool calling jail based on configuration
    /// Returns Ok(true) if jail should be applied, Ok(false) if not, or Err if invalid config
    pub fn should_apply_tool_jail(
        tool_call_parser: Option<&String>,
        tool_choice: Option<&ChatCompletionToolChoiceOption>,
        has_tools: bool,
    ) -> std::result::Result<bool, Error> {
        match (tool_call_parser, tool_choice, has_tools) {
            // tool_choice=required/named work without parser (use Immediate jail mode)
            (None, Some(ChatCompletionToolChoiceOption::Required), true) => Ok(true),
            (None, Some(ChatCompletionToolChoiceOption::Named(_)), true) => Ok(true),

            // tool_choice=auto requires a parser
            (None, Some(ChatCompletionToolChoiceOption::Auto), true) => {
                tracing::warn!(
                    "Tool choice 'auto' specified but no tool parser configured; proceeding without jailing"
                );
                Ok(false)
            }

            // Parser exists and tools might be called
            (Some(_), Some(ChatCompletionToolChoiceOption::None), _) => {
                Ok(false) // Explicitly disabled
            }
            (Some(_), Some(_), true) => Ok(true), // Any other tool_choice with tools
            (Some(_), None, true) => Ok(true),    // Default behavior when tools present

            // No tools or no parser
            _ => Ok(false),
        }
    }

    /// Apply tool calling jail to the stream if needed
    pub fn apply_tool_calling_jail<S>(
        tool_call_parser: Option<String>,
        tool_choice: Option<dynamo_async_openai::types::ChatCompletionToolChoiceOption>,
        tool_definitions: Option<Vec<dynamo_parsers::tool_calling::ToolDefinition>>,
        stream: S,
    ) -> impl Stream<Item = Annotated<NvCreateChatCompletionStreamResponse>> + Send
    where
        S: Stream<Item = Annotated<NvCreateChatCompletionStreamResponse>> + Send + 'static,
    {
        use dynamo_async_openai::types::ChatCompletionToolChoiceOption;

        let mut builder = JailedStream::builder();

        // Set tool definitions if provided
        if let Some(tool_definitions) = tool_definitions
            && !tool_definitions.is_empty()
        {
            builder = builder.tool_definitions(tool_definitions);
        }

        // Configure jail based on tool_choice
        match tool_choice {
            Some(ChatCompletionToolChoiceOption::Named(named)) => {
                // Immediate jail mode for named tool choice
                builder = builder.tool_choice_named(named.function.name.clone());
            }
            Some(ChatCompletionToolChoiceOption::Required) => {
                // Immediate jail mode for required tool choice
                builder = builder.tool_choice_required();
            }
            Some(ChatCompletionToolChoiceOption::Auto)
            | Some(ChatCompletionToolChoiceOption::None)
            | None => {
                // Traditional marker-based jail for auto/none/unspecified
                if let Some(parser) = tool_call_parser {
                    builder = builder.tool_call_parser(parser);
                }
            }
        }

        let jail = builder.build();
        jail.apply_with_finish_reason(stream)
    }

    // Motivation: Each transformation on the stream should be a separate step to allow for more flexibility
    // Earlier reasoning parser logic was nested under delta generation logic in choice_from_postprocessor
    // Since we have tool calling parsing as separate step, it makes sense to have reasoning parser as separate step as well
    pub fn parse_reasoning_content_from_stream<S>(
        stream: S,
        parser_name: String,
    ) -> impl Stream<Item = Annotated<NvCreateChatCompletionStreamResponse>> + Send
    where
        S: Stream<Item = Annotated<NvCreateChatCompletionStreamResponse>> + Send + 'static,
    {
        // Initialize reasoning parser from parser_name
        let reasoning_parser = Box::new(ReasoningParserType::get_reasoning_parser_from_name(
            parser_name.as_ref(),
        )) as Box<dyn ReasoningParser>;

        let state = ReasoningState {
            stream: Box::pin(stream),
            reasoning_parser: Some(reasoning_parser),
        };

        stream::unfold(state, |mut state| async move {
            if let Some(response) = state.stream.next().await {
                // Process the response through reasoning parser if available
                let processed_response = if let Some(ref mut parser) = state.reasoning_parser {
                    response.map_data(|mut data| {
                        // Process all choices, not just the first one
                        for choice in data.choices.iter_mut() {
                            if let Some(text) = choice.delta.content.as_ref() {
                                let parser_result =
                                    parser.parse_reasoning_streaming_incremental(text, &[]);

                                // Update this specific choice with parsed content
                                choice.delta.content = parser_result.get_some_normal_text();
                                choice.delta.reasoning_content = parser_result.get_some_reasoning();
                            }
                        }
                        Ok(data)
                    })
                } else {
                    // No reasoning parser configured, pass through unchanged
                    response
                };

                Some((processed_response, state))
            } else {
                None
            }
        })
        .fuse()
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
        SingleIn<PreprocessedRequest>,
        ManyOut<Annotated<BackendOutput>>,
    > for OpenAIPreprocessor
{
    async fn generate(
        &self,
        request: SingleIn<NvCreateChatCompletionRequest>,
        next: Arc<
            dyn AsyncEngine<SingleIn<PreprocessedRequest>, ManyOut<Annotated<BackendOutput>>, Error>,
        >,
    ) -> Result<ManyOut<Annotated<NvCreateChatCompletionStreamResponse>>, Error> {
        // unpack the request
        let (mut request, context) = request.into_parts();

        // Preserve original inbound streaming flag before any internal overrides
        let request_id = context.id().to_string();
        let original_stream_flag = request.inner.stream.unwrap_or(false);

        // Build audit handle (None if no DYN_AUDIT_SINKS)
        let mut audit_handle = crate::audit::handle::create_handle(&request, &request_id);

        if let Some(ref mut h) = audit_handle {
            h.set_request(std::sync::Arc::new(request.clone()));
        }

        // For non-streaming requests (stream=false), enable usage by default
        // This ensures compliance with OpenAI API spec where non-streaming responses
        // always include usage statistics
        request.enable_usage_for_nonstreaming(original_stream_flag);

        // Set stream=true for internal processing (after audit capture)
        request.inner.stream = Some(true);

        // create a response generator
        let response_generator = request.response_generator(context.id().to_string());

        // convert the chat completion request to a common completion request
        let (mut common_request, annotations) = self.preprocess_request(&request).await?;

        // Attach the timing tracker to the request so downstream components can record metrics
        common_request.tracker = response_generator.tracker();

        let mut response_generator = Box::new(response_generator);

        // Update ISL only for text prompts (embeddings get sequence length from tensor shape)
        if common_request.prompt_embeds.is_none() {
            let isl = common_request.token_ids.len() as u32;
            response_generator.update_isl(isl);
        }

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
        // Extract context once
        let context = response_stream.context();

        // transform the postprocessor stream (no boxing yet)
        let stream = Self::transform_postprocessor_stream(
            response_stream,
            response_generator,
            context.clone(),
        );

        // Try to parse reasoning content only if parser is configured
        let should_parse_reasoning = self.runtime_config.reasoning_parser.is_some();

        // Reasoning Content Parsing Transformation Step
        // Current Solution:
        // This step operates on Deltas created by the transform_postprocessor_stream function
        // Only access to text and not token_ids - so can not support parsing based on token_ids for now
        // Future Solution:
        // To address the limitation if needed in future: move this step before transform_postprocessor_stream and add new field of reasoning_content to the backend output
        // Use backend_output.reasoning_content field to fill out the deltas.
        let stream: Pin<Box<dyn Stream<Item = _> + Send>> = if should_parse_reasoning {
            Box::pin(Self::parse_reasoning_content_from_stream(
                stream,
                self.runtime_config.reasoning_parser.clone().unwrap(), // Safety: We already checked that parser is some, so gtg
            ))
        } else {
            Box::pin(stream)
        };

        // Check if tools are present and if we should apply jail
        let has_tools =
            request.inner.tools.is_some() && !request.inner.tools.as_ref().unwrap().is_empty();

        // Determine if we should apply jail (do this before moving request)
        let should_jail = Self::should_apply_tool_jail(
            self.tool_call_parser.as_ref(),
            request.inner.tool_choice.as_ref(),
            has_tools,
        )?;

        // Convert OpenAI tools to parser ToolDefinition format before applying jail
        let tool_definitions = request.inner.tools.as_ref().map(|tools| {
            tools
                .iter()
                .map(|tool| dynamo_parsers::tool_calling::ToolDefinition {
                    name: tool.function.name.clone(),
                    parameters: tool.function.parameters.clone(),
                })
                .collect()
        });

        // Apply jail conditionally
        let transformed_stream: Pin<Box<dyn Stream<Item = _> + Send>> = if should_jail {
            Box::pin(Self::apply_tool_calling_jail(
                self.tool_call_parser.clone(),
                request.inner.tool_choice.clone(),
                tool_definitions,
                stream,
            ))
        } else {
            Box::pin(stream)
        };

        // Step 4: Apply audit aggregation strategy
        let final_stream = if let Some(mut audit) = audit_handle {
            let (stream, agg_fut) = if audit.streaming() {
                // Streaming: apply scan (pass-through + parallel aggregation)
                crate::audit::stream::scan_aggregate_with_future(transformed_stream)
            } else {
                // Non-streaming: apply fold (collect all, then emit single chunk)
                crate::audit::stream::fold_aggregate_with_future(transformed_stream)
            };

            // Spawn audit task
            tokio::spawn(async move {
                let final_resp = agg_fut.await;
                audit.set_response(Arc::new(final_resp));
                audit.emit();
            });

            Box::pin(stream)
        } else {
            transformed_stream
        };

        // prepend the annotations to the response stream
        let stream = annotations_stream.chain(final_stream);

        // return the response stream - single boxing at the end
        Ok(ResponseStream::new(Box::pin(stream), context))
    }
}

#[async_trait]
impl
    Operator<
        SingleIn<NvCreateCompletionRequest>,
        ManyOut<Annotated<NvCreateCompletionResponse>>,
        SingleIn<PreprocessedRequest>,
        ManyOut<Annotated<BackendOutput>>,
    > for OpenAIPreprocessor
{
    async fn generate(
        &self,
        request: SingleIn<NvCreateCompletionRequest>,
        next: Arc<
            dyn AsyncEngine<SingleIn<PreprocessedRequest>, ManyOut<Annotated<BackendOutput>>, Error>,
        >,
    ) -> Result<ManyOut<Annotated<NvCreateCompletionResponse>>, Error> {
        // unpack the request
        let (mut request, context) = request.into_parts();

        // Preserve original streaming flag
        let original_stream_flag = request.inner.stream.unwrap_or(false);

        // For non-streaming requests (stream=false), enable usage by default
        // This ensures compliance with OpenAI API spec where non-streaming responses
        // always include usage statistics
        request.enable_usage_for_nonstreaming(original_stream_flag);

        request.inner.stream = Some(true);

        // create a response generator
        let response_generator = request.response_generator(context.id().to_string());
        let mut response_generator = Box::new(response_generator);
        // convert the chat completion request to a common completion request
        let mut builder = self.builder(&request)?;

        // Check if embeddings are provided - skip tokenization path
        let annotations = if let Some(ref prompt_embeds) = request.inner.prompt_embeds {
            // Skip tokenization for embeddings
            builder.token_ids(vec![]); // Empty token IDs
            builder.prompt_embeds(Some(prompt_embeds.clone()));
            // No token annotations
            HashMap::new()
        } else {
            // Normal path: tokenize the prompt
            self.gather_tokens(&request, &mut builder, None)?
        };

        // Gather multimodal data (works with both embeddings and text prompts)
        self.gather_multi_modal_data(&request, &mut builder).await?;

        let mut common_request = builder.build()?;

        // Attach the timing tracker to the request so downstream components can record metrics
        common_request.tracker = response_generator.tracker();

        // Update ISL only for text prompts (embeddings get sequence length from tensor shape)
        if common_request.prompt_embeds.is_none() {
            let isl = common_request.token_ids.len() as u32;
            response_generator.update_isl(isl);
        }

        // repack the common completion request
        let common_request = context.map(|_| common_request);

        // create a stream of annotations this will be prepend to the response stream
        let annotations: Vec<Annotated<NvCreateCompletionResponse>> = annotations
            .into_iter()
            .flat_map(|(k, v)| Annotated::from_annotation(k, &v))
            .collect();
        let annotations_stream = stream::iter(annotations);

        // forward the common completion request to the next operator
        let response_stream = next.generate(common_request).await?;

        // Extract context once
        let context = response_stream.context();

        // transform the postprocessor stream
        let stream = Self::transform_postprocessor_stream(
            response_stream,
            response_generator,
            context.clone(),
        );

        // prepend the annotations to the response stream
        let stream = annotations_stream.chain(stream);

        // return the response stream
        Ok(ResponseStream::new(Box::pin(stream), context))
    }
}

#[async_trait]
impl
    Operator<
        SingleIn<NvCreateEmbeddingRequest>,
        ManyOut<Annotated<NvCreateEmbeddingResponse>>,
        SingleIn<PreprocessedEmbeddingRequest>,
        ManyOut<Annotated<EmbeddingsEngineOutput>>,
    > for OpenAIPreprocessor
{
    async fn generate(
        &self,
        request: SingleIn<NvCreateEmbeddingRequest>,
        next: Arc<
            dyn AsyncEngine<
                    SingleIn<PreprocessedEmbeddingRequest>,
                    ManyOut<Annotated<EmbeddingsEngineOutput>>,
                    Error,
                >,
        >,
    ) -> Result<ManyOut<Annotated<NvCreateEmbeddingResponse>>, Error> {
        // Unpack request
        let (request, context) = request.into_parts();

        // Preprocess the embedding request
        let (preprocessed_request, annotations) =
            self.preprocess_embedding_request(&request).await?;

        // Forward to next stage
        let preprocessed_request = context.map(|_| preprocessed_request);
        let response_stream = next.generate(preprocessed_request).await?;

        // Extract context once
        let context = response_stream.context();

        // Transform response stream back to OpenAI format
        let stream = Self::transform_embedding_postprocessor_stream(response_stream, request);

        // Prepend annotations
        let annotations_stream = stream::iter(
            annotations
                .into_iter()
                .flat_map(|(k, v)| Annotated::from_annotation(k, &v))
                .collect::<Vec<_>>(),
        );

        let combined_stream = annotations_stream.chain(stream);
        Ok(ResponseStream::new(Box::pin(combined_stream), context))
    }
}

// Note: tests for jailing and parser detection live in `lib/llm/tests/test_jail.rs`
