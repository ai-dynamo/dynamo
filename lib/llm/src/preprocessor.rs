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
pub mod speculative_prefill;
pub mod tools;
use anyhow::Context;
use anyhow::{Result, bail};

use dynamo_protocols::types::{
    ChatCompletionRequestMessage, ChatCompletionRequestUserMessageContent,
    ChatCompletionRequestUserMessageContentPart, ChatCompletionToolChoiceOption, EncodingFormat,
};
use dynamo_runtime::error::{DynamoError, ErrorType};
use futures::Stream;
use futures::stream::{self, StreamExt};
use prompt::OAIPromptFormatter;
use std::time::{Duration, Instant};

use dynamo_runtime::dynamo_nvtx_range;
use dynamo_runtime::metrics::frontend_perf::{
    DETOKENIZE_TOKEN_COUNT, DETOKENIZE_TOTAL_US, STAGE_DURATION_SECONDS, STAGE_PREPROCESS,
    StageGuard, TEMPLATE_SECONDS, TOKENIZE_SECONDS,
};
use std::borrow::Cow;
use std::{collections::HashMap, pin::Pin, sync::Arc};
use tracing;

use crate::model_card::{ModelDeploymentCard, ModelInfo};
use crate::preprocessor::media::MediaLoader;
use crate::preprocessor::prompt::OAIChatLikeRequest;
use crate::protocols::common::preprocessor::{
    MultimodalData, MultimodalDataMap, PreprocessedRequestBuilder, RoutingHints,
};
use crate::protocols::common::timing::RequestTracker;
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
use crate::tokenizers::traits::Tokenizer;

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
    /// Prefill worker ID (for TTFT attribution in disaggregated mode)
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub prefill_worker_id: Option<u64>,
    /// Prefill worker DP rank
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub prefill_dp_rank: Option<u32>,
    /// Prefill worker type ("prefill" or "decode") for Prometheus metric labeling.
    /// Stored at routing time to avoid expensive MDC lookup when updating TTFT metrics.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub prefill_worker_type: Option<String>,
    /// Decode worker ID (for ITL attribution in disaggregated mode)
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub decode_worker_id: Option<u64>,
    /// Decode worker DP rank
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub decode_dp_rank: Option<u32>,
    /// Decode worker type ("prefill" or "decode") for Prometheus metric labeling.
    /// Stored at routing time to avoid expensive MDC lookup when updating ITL metrics.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub decode_worker_type: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tokenize_latency: Option<Duration>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub detokenize_total_latency: Option<Duration>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub detokenize_count: Option<u64>,
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
    media_loader: Option<MediaLoader>,
    /// Max context length (in tokens) this model can handle, from ModelDeploymentCard
    context_length: u32,
}

impl OpenAIPreprocessor {
    pub fn new(mdc: ModelDeploymentCard) -> Result<Arc<Self>> {
        let formatter = PromptFormatter::from_mdc(&mdc)?;
        let tokenizer = mdc.tokenizer()?;
        match formatter {
            PromptFormatter::OAI(formatter) => Self::new_with_parts(mdc, formatter, tokenizer),
        }
    }

    pub fn new_with_parts(
        mdc: ModelDeploymentCard,
        formatter: Arc<dyn OAIPromptFormatter>,
        tokenizer: crate::tokenizers::Tokenizer,
    ) -> Result<Arc<Self>> {
        let mdcsum = mdc.mdcsum().to_string();
        let tokenizer: Arc<dyn Tokenizer> = (*tokenizer).clone();
        let lora_name = mdc.lora.as_ref().map(|l| l.name.clone());
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

        let media_loader = match mdc.media_decoder {
            Some(media_decoder) => Some(MediaLoader::new(media_decoder, mdc.media_fetcher)?),
            None => None,
        };

        let context_length = mdc.context_length;

        Ok(Arc::new(Self {
            formatter,
            tokenizer,
            model_info,
            mdcsum,
            lora_name,
            runtime_config,
            tool_call_parser,
            media_loader,
            context_length,
        }))
    }
    /// Encode a string to it's tokens
    pub fn tokenize(&self, s: &str) -> anyhow::Result<Encoding> {
        self.tokenizer.encode(s)
    }

    /// Translate a [`NvCreateChatCompletionRequest`] request to a common completion request.
    /// Returns the common completion request, a hashmap of annotations, and a boolean
    /// indicating whether the rendered prompt ends with a reasoning start token (e.g.,
    /// `<think>`), meaning the model's completion will begin mid-reasoning.
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
        tracker: Option<&RequestTracker>,
    ) -> Result<(PreprocessedRequest, HashMap<String, String>, bool)> {
        let _stage_guard = StageGuard::new(STAGE_PREPROCESS, "");
        let preprocess_start = Instant::now();
        let mut builder = self.builder(request)?;

        let template_start = Instant::now();
        let formatted_prompt = {
            let _nvtx = dynamo_nvtx_range!("preprocess.template");
            self.apply_template(request)
                .with_context(|| "Failed to apply prompt template")?
        };
        TEMPLATE_SECONDS.observe(template_start.elapsed().as_secs_f64());

        // Check if the chat template injected a reasoning start token at the end
        // of the prompt (e.g., Qwen3.5 appends `<think>\n` when enable_thinking
        // is not explicitly false). If so, the model's completion starts
        // mid-reasoning and the parser should begin in reasoning mode.
        let prompt_injected_reasoning = formatted_prompt
            .as_ref()
            .is_some_and(|p| p.trim_end().ends_with("<think>"));

        let tokenize_start = Instant::now();
        let annotations = {
            let _nvtx = dynamo_nvtx_range!("preprocess.tokenize");
            self.gather_tokens(request, &mut builder, formatted_prompt.clone(), tracker)
                .with_context(|| "Failed to gather tokens")?
        };
        TOKENIZE_SECONDS.observe(tokenize_start.elapsed().as_secs_f64());

        self.gather_multi_modal_data(request, &mut builder, formatted_prompt)
            .await
            .with_context(|| "Failed to gather multimodal data")?;

        STAGE_DURATION_SECONDS
            .with_label_values(&[STAGE_PREPROCESS])
            .observe(preprocess_start.elapsed().as_secs_f64());

        Ok((builder.build()?, annotations, prompt_injected_reasoning))
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
            let hints = nvext.agent_hints.as_ref();
            builder.request_timestamp_ms(nvext.request_timestamp_ms);
            builder.agent_context(nvext.agent_context.clone());
            let routing = RoutingHints {
                backend_instance_id: nvext.backend_instance_id,
                prefill_worker_id: nvext.prefill_worker_id,
                decode_worker_id: nvext.decode_worker_id,
                dp_rank: nvext.dp_rank,
                prefill_dp_rank: nvext.prefill_dp_rank,
                expected_output_tokens: hints.and_then(|h| h.osl),
                priority_jump: hints.and_then(|h| {
                    h.priority
                        .map(|priority| priority.max(0) as f64)
                        .or(h.latency_sensitivity)
                }),
                priority: hints.and_then(|h| h.priority),
                lora_name,
                allowed_worker_ids: None,
                session_control: nvext.session_control.clone(),
            };
            builder.routing(Some(routing));
        } else if lora_name.is_some() {
            // Ensure routing hints exist when we have LoRA,
            // even when nvext is absent.
            builder.routing(Some(RoutingHints {
                lora_name,
                ..Default::default()
            }));
        }

        // Forward mm_processor_kwargs (e.g. use_audio_in_video) to the backend.
        builder.mm_processor_kwargs(request.mm_processor_kwargs().cloned());

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

    /// Replace inline `data:` URLs with empty strings in message content parts.
    /// Preserves HTTP(S) URLs, text content, and overall message structure.
    fn strip_inline_data_urls(messages: &mut serde_json::Value) {
        let Some(arr) = messages.as_array_mut() else {
            return;
        };
        for msg in arr {
            let Some(content) = msg.get_mut("content") else {
                continue;
            };
            let Some(parts) = content.as_array_mut() else {
                continue;
            };
            for part in parts {
                for key in ["image_url", "video_url", "audio_url"] {
                    if let Some(media) = part.get_mut(key)
                        && let Some(url) = media.get_mut("url")
                        && url.as_str().is_some_and(|s| s.starts_with("data:"))
                    {
                        *url = serde_json::Value::String(String::new());
                    }
                }
            }
        }
    }

    pub async fn gather_multi_modal_data<R: OAIChatLikeRequest>(
        &self,
        request: &R,
        builder: &mut PreprocessedRequestBuilder,
        formatted_prompt: Option<String>,
    ) -> Result<()> {
        let mut media_map: MultimodalDataMap = HashMap::new();
        let mut fetch_tasks: Vec<(String, &ChatCompletionRequestUserMessageContentPart)> =
            Vec::new();

        let Some(messages) = request.typed_messages() else {
            return Ok(());
        };
        let has_media_loader = self.media_loader.is_some();

        for message in messages.iter() {
            let content_parts = match message {
                ChatCompletionRequestMessage::User(u) => match &u.content {
                    ChatCompletionRequestUserMessageContent::Array(parts) => parts,
                    _ => continue,
                },
                _ => continue,
            };
            for content_part in content_parts.iter() {
                if has_media_loader {
                    let type_str = match content_part {
                        ChatCompletionRequestUserMessageContentPart::ImageUrl(_) => "image_url",
                        ChatCompletionRequestUserMessageContentPart::VideoUrl(_) => "video_url",
                        ChatCompletionRequestUserMessageContentPart::AudioUrl(_) => "audio_url",
                        _ => continue,
                    };
                    fetch_tasks.push((type_str.to_string(), content_part));
                } else {
                    let (type_str, url) = match content_part {
                        ChatCompletionRequestUserMessageContentPart::ImageUrl(p) => {
                            ("image_url", p.image_url.url.clone())
                        }
                        ChatCompletionRequestUserMessageContentPart::VideoUrl(p) => {
                            ("video_url", p.video_url.url.clone())
                        }
                        ChatCompletionRequestUserMessageContentPart::AudioUrl(p) => {
                            ("audio_url", p.audio_url.url.clone())
                        }
                        _ => continue,
                    };
                    media_map
                        .entry(type_str.to_string())
                        .or_default()
                        .push(MultimodalData::Url(url));
                }
            }
        }

        // Execute all fetch tasks
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

            // Preserve original messages and formatted prompt in extra_args for multimodal
            // workers (e.g., TRT-LLM needs messages and the template-rendered prompt with
            // <image> placeholders for embedding-path / NIXL flows).
            let messages_json = serde_json::to_value(request.messages())?;
            let mut extra_args = serde_json::json!({
                "messages": messages_json
            });

            // Strip redundant inline data: URLs only when frontend decoding is active
            // (media_loader decoded the images into RDMA descriptors). TRT-LLM and
            // other backends that pass URLs through still need the original data: URIs.
            if self.media_loader.is_some() {
                Self::strip_inline_data_urls(&mut extra_args["messages"]);
            }

            if let Some(ref prompt) = formatted_prompt {
                extra_args["formatted_prompt"] = serde_json::Value::String(prompt.clone());
            }
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
        tracker: Option<&RequestTracker>,
    ) -> Result<HashMap<String, String>> {
        let mut annotations = HashMap::new();
        let mut token_count: Option<usize> = None;
        // match request type before any conversion/processing
        match request.prompt_input_type() {
            PromptInput::Tokens(_) => {
                if let Some(token_input) = request.extract_tokens() {
                    match token_input {
                        TokenInput::Single(tokens) => {
                            token_count = Some(tokens.len());
                            builder.token_ids(tokens);
                        }
                        TokenInput::Batch(token_batches) => {
                            if token_batches.len() == 1 {
                                token_count = Some(token_batches[0].len());
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

                            // If nvext.token_data is present, use the pre-computed tokens
                            // directly and skip tokenization.  This avoids redundant
                            // tokenization when an external component (e.g. the GAIE EPP
                            // KV-router) has already tokenized the prompt.
                            // When backend_instance_id is set without token_data, warn
                            // but fall back to tokenization (backward compat for non-GAIE
                            // routers that set the header without providing tokens).
                            let has_backend_instance_id = request
                                .nvext()
                                .and_then(|ext| ext.backend_instance_id)
                                .is_some();

                            let token_data =
                                request.nvext().and_then(|ext| ext.token_data.as_ref());

                            let (tokens_vec, skip_token_annotation) = if let Some(tokens) =
                                token_data
                            {
                                tracing::info!(
                                    token_count = tokens.len(),
                                    first_tokens = ?&tokens[..std::cmp::min(5, tokens.len())],
                                    "[SIDECAR-SKIP-TOKENIZE] Found nvext.token_data — using pre-computed tokens, SKIPPING tokenization"
                                );
                                (tokens.clone(), true)
                            } else if has_backend_instance_id {
                                tracing::warn!(
                                    "backend_instance_id provided but no token_data; tokenizing prompt"
                                );
                                let encoding = self.encode_with_timing(&prompt, tracker)?;
                                (encoding.token_ids().to_vec(), false)
                            } else {
                                let encoding = self.encode_with_timing(&prompt, tracker)?;
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

                            token_count = Some(tokens_vec.len());
                            builder.token_ids(tokens_vec);
                        }
                        TextInput::Batch(texts) => {
                            if texts.len() == 1 {
                                let encoding = self.encode_with_timing(&texts[0], tracker)?;
                                let tokens = encoding.token_ids().to_vec();
                                token_count = Some(tokens.len());
                                builder.token_ids(tokens);
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

        // Validate prompt token count against model's context length
        if let Some(count) = token_count {
            Self::validate_token_count(count, self.context_length)?;
        }

        Ok(annotations)
    }

    /// Validate that the prompt token count does not consume the model's entire context length.
    /// Returns an error if the prompt leaves no room for output tokens.
    fn validate_token_count(token_count: usize, context_length: u32) -> Result<()> {
        let max_len = context_length as usize;
        // max_len == 0 means context_length was not configured (model_card.rs defaults
        // to 0 when max_position_embeddings is absent), so skip validation.
        // Use >= because context_length is the total budget (input + output): if the
        // prompt alone fills it, there is zero room for output tokens.
        if max_len > 0 && token_count >= max_len {
            return Err(DynamoError::builder()
                .error_type(ErrorType::InvalidArgument)
                .message(format!(
                    "This model's maximum context length is {} tokens. \
                     However, your messages resulted in {} tokens. \
                     Please reduce the length of the messages.",
                    max_len, token_count,
                ))
                .build()
                .into());
        }
        Ok(())
    }

    fn encode_with_timing(
        &self,
        prompt: &str,
        tracker: Option<&RequestTracker>,
    ) -> anyhow::Result<Encoding> {
        let encode_start = Instant::now();
        let prompt = if prompt.contains('\0') {
            tracing::debug!("Prompt contains null bytes; stripping to avoid tokenizer divergence");
            Cow::Owned(prompt.replace('\0', ""))
        } else {
            Cow::Borrowed(prompt)
        };
        let encoding = self.tokenizer.encode(prompt.as_ref())?;
        if let Some(t) = tracker {
            t.record_tokenize_latency(encode_start.elapsed());
        }
        Ok(encoding)
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
        let _stage_guard = StageGuard::new(STAGE_PREPROCESS, "");
        let mut annotations = HashMap::new();
        let mut builder = PreprocessedEmbeddingRequest::builder();

        let all_token_ids = match &request.inner.input {
            dynamo_protocols::types::EmbeddingInput::String(s) => {
                let encoding = self.tokenizer.encode(s)?;
                vec![encoding.token_ids().to_vec()]
            }
            dynamo_protocols::types::EmbeddingInput::StringArray(arr) => {
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
            dynamo_protocols::types::EmbeddingInput::IntegerArray(token_ids) => {
                vec![token_ids.clone()]
            }
            dynamo_protocols::types::EmbeddingInput::ArrayOfIntegerArray(token_arrays) => {
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

    pub fn postprocessor_parsing_stream<S>(
        &self,
        stream: S,
        request: &NvCreateChatCompletionRequest,
        prompt_injected_reasoning: bool,
    ) -> anyhow::Result<
        impl Stream<Item = Annotated<NvCreateChatCompletionStreamResponse>> + Send + 'static,
    >
    where
        S: Stream<Item = Annotated<NvCreateChatCompletionStreamResponse>> + Send + 'static,
    {
        // Tool-continuation turns (last message role=tool) gate the force-
        // reasoning flag off: the model produces the final user-facing answer
        // directly from the tool result and typically does not re-enter
        // reasoning, so leaving the parser in forced-reasoning mode would
        // mislabel the final answer as reasoning_content. Matches SGLang's
        // observed behavior for Kimi K2.5 tool-result follow-ups.
        let last_is_tool = matches!(
            request.inner.messages.last(),
            Some(ChatCompletionRequestMessage::Tool(_))
        );
        let prompt_injected_reasoning = prompt_injected_reasoning && !last_is_tool;

        // tool_choice=required/named forces the backend into guided decoding,
        // which constrains output to a bare JSON shape with no reasoning
        // wrapper. Running the reasoning parser on that output is both
        // pointless (nothing to extract) and actively harmful for parsers
        // that inject a `<think>` prefix unconditionally (e.g. MiniMax
        // append-think), because the prefix would contaminate the
        // tool-call JSON fed into the jail.
        let tool_choice_forces_guided_json = matches!(
            request.inner.tool_choice,
            Some(ChatCompletionToolChoiceOption::Required)
                | Some(ChatCompletionToolChoiceOption::Named(_))
        );

        // Try to parse reasoning content only if parser is configured.
        let should_parse_reasoning = self.runtime_config.reasoning_parser.is_some()
            && !Self::is_reasoning_disabled_by_request(
                self.runtime_config.reasoning_parser.as_deref(),
                request.chat_template_args.as_ref(),
            )
            && !tool_choice_forces_guided_json;

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
                prompt_injected_reasoning,
            ))
        } else {
            Box::pin(stream)
        };

        // Check if tools are present and if we should apply jail
        let has_tools = request
            .inner
            .tools
            .as_ref()
            .is_some_and(|tools| !tools.is_empty());

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

        Ok(transformed_stream)
    }

    pub fn transform_postprocessor_stream<S, Resp>(
        stream: S,
        generator: Box<dyn DeltaGeneratorExt<Resp>>,
        context: Arc<dyn AsyncEngineContext>,
        trace_tokens_enabled: bool,
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
            trace_tokens_enabled: bool,
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
            trace_tokens_enabled,
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
                        // inner.finished = true; // Mark as finished
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

                        let isl = inner.response_generator.get_isl().map(|isl| isl as usize);

                        (chunk_tokens, isl)
                    } else {
                        (0, None)
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

                    // Create LLM metrics annotation with prefill/decode worker info from tracker.
                    // Worker types are stored at routing time to avoid expensive MDC lookup.
                    let tracker = inner.response_generator.tracker();
                    let prefill_worker_id = tracker.as_ref().and_then(|t| t.prefill_worker_id());
                    let prefill_dp_rank = tracker.as_ref().and_then(|t| t.prefill_dp_rank());
                    let prefill_worker_type = tracker
                        .as_ref()
                        .and_then(|t| t.prefill_worker_type())
                        .map(String::from);
                    let decode_worker_id = tracker.as_ref().and_then(|t| t.decode_worker_id());
                    let decode_dp_rank = tracker.as_ref().and_then(|t| t.decode_dp_rank());
                    let decode_worker_type = tracker
                        .as_ref()
                        .and_then(|t| t.decode_worker_type())
                        .map(String::from);
                    let llm_metrics = LLMMetricAnnotation {
                        input_tokens: isl.unwrap_or(0),
                        output_tokens: current_osl,
                        chunk_tokens,
                        cached_tokens: None,
                        prefill_worker_id,
                        prefill_dp_rank,
                        prefill_worker_type,
                        decode_worker_id,
                        decode_dp_rank,
                        decode_worker_type,
                        tokenize_latency: tracker.as_ref().and_then(|t| t.tokenize_latency()),
                        detokenize_total_latency: tracker.as_ref().and_then(|t| t.detokenize_total_latency()),
                        detokenize_count: tracker.as_ref().map(|t| t.detokenize_count()),
                    };
                    if inner.trace_tokens_enabled {
                        crate::agents::trace::record_llm_metric_tokens(
                            tracker.as_deref(),
                            isl,
                            current_osl,
                            None,
                        );
                    }

                    // Flush per-request detokenize accumulators to global Prometheus counters
                    // (once per request instead of per-token).
                    if let Some(t) = tracker.as_ref() {
                        if let Some(total) = t.detokenize_total_latency() {
                            DETOKENIZE_TOTAL_US.inc_by(total.as_micros() as f64);
                        }
                        DETOKENIZE_TOKEN_COUNT.inc_by(t.detokenize_count() as f64);
                    }

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
                        let tracker = inner.response_generator.tracker();
                        let cached_tokens = usage
                            .prompt_tokens_details
                            .as_ref()
                            .and_then(|d| d.cached_tokens.map(|c| c as usize));
                        let prefill_worker_id =
                            tracker.as_ref().and_then(|t| t.prefill_worker_id());
                        let prefill_dp_rank = tracker.as_ref().and_then(|t| t.prefill_dp_rank());
                        let prefill_worker_type = tracker
                            .as_ref()
                            .and_then(|t| t.prefill_worker_type())
                            .map(String::from);
                        let decode_worker_id = tracker.as_ref().and_then(|t| t.decode_worker_id());
                        let decode_dp_rank = tracker.as_ref().and_then(|t| t.decode_dp_rank());
                        let decode_worker_type = tracker
                            .as_ref()
                            .and_then(|t| t.decode_worker_type())
                            .map(String::from);
                        let llm_metrics = LLMMetricAnnotation {
                            input_tokens: usage.prompt_tokens as usize,
                            output_tokens: usage.completion_tokens as usize,
                            chunk_tokens: 0,
                            cached_tokens,
                            prefill_worker_id,
                            prefill_dp_rank,
                            prefill_worker_type,
                            decode_worker_id,
                            decode_dp_rank,
                            decode_worker_type,
                            tokenize_latency: tracker.as_ref().and_then(|t| t.tokenize_latency()),
                            detokenize_total_latency: tracker
                                .as_ref()
                                .and_then(|t| t.detokenize_total_latency()),
                            detokenize_count: tracker.as_ref().map(|t| t.detokenize_count()),
                        };
                        if inner.trace_tokens_enabled {
                            crate::agents::trace::record_llm_metric_tokens(
                                tracker.as_deref(),
                                Some(usage.prompt_tokens as usize),
                                usage.completion_tokens as usize,
                                cached_tokens,
                            );
                        }

                        // Flush per-request detokenize accumulators to global Prometheus counters
                        // (once per request instead of per-token).
                        if let Some(t) = tracker.as_ref() {
                            if let Some(total) = t.detokenize_total_latency() {
                                DETOKENIZE_TOTAL_US.inc_by(total.as_micros() as f64);
                            }
                            DETOKENIZE_TOKEN_COUNT.inc_by(t.detokenize_count() as f64);
                        }

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
                            error: None,
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
                let embeddings: Vec<dynamo_protocols::types::Embedding> = engine_output
                    .embeddings
                    .into_iter()
                    .enumerate()
                    .map(|(index, embedding)| dynamo_protocols::types::Embedding {
                        index: index as u32,
                        object: "embedding".to_string(),
                        embedding: embedding.into_iter().map(|f| f as f32).collect(),
                    })
                    .collect();

                let response = NvCreateEmbeddingResponse {
                    inner: dynamo_protocols::types::CreateEmbeddingResponse {
                        object: "list".to_string(),
                        model: original_request.inner.model.clone(),
                        data: embeddings,
                        usage: dynamo_protocols::types::EmbeddingUsage {
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
        tool_choice: Option<dynamo_protocols::types::ChatCompletionToolChoiceOption>,
        tool_definitions: Option<Vec<dynamo_parsers::tool_calling::ToolDefinition>>,
        stream: S,
    ) -> impl Stream<Item = Annotated<NvCreateChatCompletionStreamResponse>> + Send
    where
        S: Stream<Item = Annotated<NvCreateChatCompletionStreamResponse>> + Send + 'static,
    {
        use dynamo_protocols::types::ChatCompletionToolChoiceOption;

        let mut builder = JailedStream::builder();

        // Set tool definitions if provided
        if let Some(tool_definitions) = tool_definitions
            && !tool_definitions.is_empty()
        {
            builder = builder.tool_definitions(tool_definitions);
        }

        // Configure jail based on tool_choice
        //
        // For tool_choice=required or named we mirror SGLang / vLLM: assume the
        // backend applied guided decoding and emit a bare JSON shape, so parse
        // via the JSON array parser (base_json_parser) rather than the model's
        // native-format parser.  If a parser is also configured we still carry
        // it so the Immediate branch can fall back to marker-based parsing for
        // backends that do not honor guided decoding (e.g. XML-native models
        // like qwen3_coder — see regression test_tool_choice_required_with_
        // qwen3_coder_parser).
        match tool_choice {
            Some(ChatCompletionToolChoiceOption::Named(named)) => {
                builder = builder
                    .tool_choice_named(named.function.name.clone())
                    .named_tool_filter(named.function.name.clone());
                if let Some(parser) = tool_call_parser {
                    builder = builder.tool_call_parser(parser);
                }
            }
            Some(ChatCompletionToolChoiceOption::Required) => {
                builder = builder.tool_choice_required();
                if let Some(parser) = tool_call_parser {
                    builder = builder.tool_call_parser(parser);
                }
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

    /// Check if reasoning parsing should be disabled based on per-request parameters.
    /// For kimi_k25: disabled when chat_template_args contains "thinking": false.
    /// For nemotron_nano: disabled when chat_template_args contains "enable_thinking": false
    ///   or "force_nonempty_content": true.
    /// For deepseek_r1 / deepseek_v4: disabled when chat_template_args contains
    ///   "thinking": false or "thinking_mode": "chat" — matches the V4 formatter's
    ///   `resolve_thinking_mode` convention, so the parser and the prompt stay in sync.
    fn is_reasoning_disabled_by_request(
        reasoning_parser: Option<&str>,
        chat_template_args: Option<&std::collections::HashMap<String, serde_json::Value>>,
    ) -> bool {
        match reasoning_parser {
            Some("kimi_k25") => {
                if let Some(args) = chat_template_args
                    && let Some(thinking) = args.get("thinking")
                {
                    return thinking == &serde_json::Value::Bool(false);
                }
                false
            }
            Some("nemotron_nano") | Some("nemotron3") => {
                if let Some(args) = chat_template_args {
                    if let Some(enable_thinking) = args.get("enable_thinking")
                        && enable_thinking == &serde_json::Value::Bool(false)
                    {
                        return true;
                    }
                    if let Some(force_nonempty) = args.get("force_nonempty_content")
                        && force_nonempty == &serde_json::Value::Bool(true)
                    {
                        return true;
                    }
                }
                false
            }
            Some("deepseek_r1") | Some("deepseek_v4") | Some("deepseek-v4")
            | Some("deepseekv4") => {
                if let Some(enabled) =
                    crate::preprocessor::prompt::thinking_bool_from_args(chat_template_args)
                {
                    return !enabled;
                }
                if let Some(args) = chat_template_args
                    && let Some(mode) = args.get("thinking_mode").and_then(|v| v.as_str())
                {
                    return mode == "chat";
                }
                false
            }
            _ => false,
        }
    }

    // Motivation: Each transformation on the stream should be a separate step to allow for more flexibility
    // Earlier reasoning parser logic was nested under delta generation logic in choice_from_postprocessor
    // Since we have tool calling parsing as separate step, it makes sense to have reasoning parser as separate step as well
    /// Apply reasoning parsing to the output stream, splitting content into
    /// `reasoning_content` and normal `content` based on think tags.
    ///
    /// When `prompt_injected_reasoning` is `true`, the parser starts in reasoning
    /// mode immediately — use this when the chat template already appended the
    /// reasoning start token (e.g., `<think>`) to the prompt, so the model's
    /// completion begins with thinking content without an explicit start tag.
    pub fn parse_reasoning_content_from_stream<S>(
        stream: S,
        parser_name: String,
        prompt_injected_reasoning: bool,
    ) -> impl Stream<Item = Annotated<NvCreateChatCompletionStreamResponse>> + Send
    where
        S: Stream<Item = Annotated<NvCreateChatCompletionStreamResponse>> + Send + 'static,
    {
        // Initialize reasoning parser from parser_name
        let mut reasoning_parser = Box::new(ReasoningParserType::get_reasoning_parser_from_name(
            parser_name.as_ref(),
        )) as Box<dyn ReasoningParser>;

        if prompt_injected_reasoning {
            reasoning_parser.set_in_reasoning(true);
        }

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
                        for choice in data.inner.choices.iter_mut() {
                            // Reasoning parsing only applies to text content
                            if let Some(
                                dynamo_protocols::types::ChatCompletionMessageContent::Text(text),
                            ) = choice.delta.content.as_ref()
                            {
                                let parser_result =
                                    parser.parse_reasoning_streaming_incremental(text, &[]);

                                // Update this specific choice with parsed content
                                choice.delta.content = parser_result.get_some_normal_text().map(
                                    dynamo_protocols::types::ChatCompletionMessageContent::Text,
                                );
                                choice.delta.reasoning_content = parser_result.get_some_reasoning();
                            }
                            // For multimodal content, pass through unchanged
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
        let tracker = response_generator.tracker();

        // convert the chat completion request to a common completion request
        let (mut common_request, annotations, prompt_injected_reasoning) = self
            .preprocess_request(&request, tracker.as_deref())
            .await?;
        tracing::trace!(request = ?common_request, prompt_injected_reasoning, "Pre-processed request");
        let trace_state = if crate::agents::trace::is_enabled() {
            common_request.agent_context.clone().map(|agent_context| {
                let request_model = common_request.model.clone();
                let request_tracker = tracker.clone();
                let x_request_id = dynamo_runtime::logging::get_distributed_tracing_context()
                    .and_then(|context| context.x_request_id)
                    .or_else(|| {
                        context
                            .get::<String>(crate::agents::trace::X_REQUEST_ID_CONTEXT_KEY)
                            .ok()
                            .map(|value| value.as_ref().clone())
                    });
                (agent_context, request_model, request_tracker, x_request_id)
            })
        } else {
            None
        };
        let trace_tokens_enabled = trace_state.is_some();

        // Attach the timing tracker to the request so downstream components can record metrics
        common_request.tracker = tracker;

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

        // transform the postprocessor stream (no boxing yet) - detokenize
        let stream = Self::transform_postprocessor_stream(
            response_stream,
            response_generator,
            context.clone(),
            trace_tokens_enabled,
        );

        let transformed_stream =
            self.postprocessor_parsing_stream(stream, &request, prompt_injected_reasoning)?;

        // Apply audit aggregation strategy.
        // The audit branch already returns Pin<Box<...>> from scan/fold_aggregate_with_future,
        // while the non-audit branch boxes the impl Stream from postprocessor_parsing_stream.
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

            stream
        } else {
            Box::pin(transformed_stream)
        };

        // Step 5: Speculative next-turn prefill
        let final_stream = speculative_prefill::maybe_wrap_stream(
            final_stream,
            &request,
            &next,
            &self.formatter,
            &self.tokenizer,
        );

        let final_stream = if let Some((
            agent_context,
            request_model,
            request_tracker,
            x_request_id,
        )) = trace_state
        {
            let (stream, done_fut) = crate::telemetry::stream::notify_on_completion(final_stream);
            tokio::spawn(async move {
                done_fut.await;
                if request_tracker.is_none() {
                    tracing::warn!(
                        request_id,
                        "agent_context present but request tracker is missing; emitting partial trace"
                    );
                }
                let metrics = crate::agents::trace::request_metrics(
                    request_id,
                    x_request_id,
                    request_model,
                    request_tracker.as_deref(),
                );
                crate::agents::trace::emit_request_end(agent_context, metrics);
            });
            stream
        } else {
            final_stream
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
        let _stage_guard = StageGuard::new(STAGE_PREPROCESS, "");

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
        let tracker = response_generator.tracker();
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
            self.gather_tokens(&request, &mut builder, None, tracker.as_deref())?
        };

        // Gather multimodal data (works with both embeddings and text prompts)
        self.gather_multi_modal_data(&request, &mut builder, None)
            .await?;

        let mut common_request = builder.build()?;

        // Attach the timing tracker to the request so downstream components can record metrics
        common_request.tracker = tracker;

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

        // End preprocess stage before handing off to downstream (route/dispatch).
        drop(_stage_guard);

        // forward the common completion request to the next operator
        let response_stream = next.generate(common_request).await?;

        // Extract context once
        let context = response_stream.context();

        // transform the postprocessor stream
        let stream = Self::transform_postprocessor_stream(
            response_stream,
            response_generator,
            context.clone(),
            false,
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

#[cfg(test)]
mod strip_tests {
    use super::OpenAIPreprocessor;

    #[test]
    fn test_strip_inline_data_urls_replaces_data_urls() {
        let mut messages = serde_json::json!([{
            "role": "user",
            "content": [
                {"type": "text", "text": "What is this?"},
                {"type": "image_url", "image_url": {"url": "data:image/png;base64,iVBOR...longdata..."}},
                {"type": "image_url", "image_url": {"url": "https://example.com/img.png"}}
            ]
        }]);
        OpenAIPreprocessor::strip_inline_data_urls(&mut messages);
        let parts = messages[0]["content"].as_array().unwrap();
        assert_eq!(parts[0]["text"], "What is this?");
        assert_eq!(parts[1]["image_url"]["url"], "");
        assert_eq!(parts[2]["image_url"]["url"], "https://example.com/img.png");
    }

    #[test]
    fn test_strip_inline_data_urls_handles_video_audio() {
        let mut messages = serde_json::json!([{
            "role": "user",
            "content": [
                {"type": "video_url", "video_url": {"url": "data:video/mp4;base64,AAAA..."}},
                {"type": "audio_url", "audio_url": {"url": "https://example.com/audio.wav"}}
            ]
        }]);
        OpenAIPreprocessor::strip_inline_data_urls(&mut messages);
        let parts = messages[0]["content"].as_array().unwrap();
        assert_eq!(parts[0]["video_url"]["url"], "");
        assert_eq!(
            parts[1]["audio_url"]["url"],
            "https://example.com/audio.wav"
        );
    }

    #[test]
    fn test_strip_inline_data_urls_preserves_text_only() {
        let mut messages = serde_json::json!([{
            "role": "user",
            "content": "plain text message"
        }]);
        let original = messages.clone();
        OpenAIPreprocessor::strip_inline_data_urls(&mut messages);
        assert_eq!(messages, original);
    }

    #[test]
    fn test_strip_inline_data_urls_empty_messages() {
        let mut messages = serde_json::json!([]);
        OpenAIPreprocessor::strip_inline_data_urls(&mut messages);
        assert_eq!(messages, serde_json::json!([]));
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_is_reasoning_disabled_by_request() {
        let thinking_true = {
            let mut m = std::collections::HashMap::new();
            m.insert("thinking".to_string(), serde_json::Value::Bool(true));
            m
        };
        let thinking_false = {
            let mut m = std::collections::HashMap::new();
            m.insert("thinking".to_string(), serde_json::Value::Bool(false));
            m
        };
        let enable_thinking_true = {
            let mut m = std::collections::HashMap::new();
            m.insert("enable_thinking".to_string(), serde_json::Value::Bool(true));
            m
        };
        let enable_thinking_false = {
            let mut m = std::collections::HashMap::new();
            m.insert(
                "enable_thinking".to_string(),
                serde_json::Value::Bool(false),
            );
            m
        };
        let thinking_mode_chat = {
            let mut m = std::collections::HashMap::new();
            m.insert(
                "thinking_mode".to_string(),
                serde_json::Value::String("chat".to_string()),
            );
            m
        };
        let thinking_mode_thinking = {
            let mut m = std::collections::HashMap::new();
            m.insert(
                "thinking_mode".to_string(),
                serde_json::Value::String("thinking".to_string()),
            );
            m
        };
        let empty_args = std::collections::HashMap::new();

        // (parser, args, expected_disabled, description)
        let cases = [
            (
                Some("kimi_k25"),
                Some(&thinking_false),
                true,
                "kimi_k25 + thinking=false → disabled",
            ),
            (
                Some("kimi_k25"),
                Some(&thinking_true),
                false,
                "kimi_k25 + thinking=true → enabled",
            ),
            (
                Some("kimi_k25"),
                None,
                false,
                "kimi_k25 + no args → enabled",
            ),
            (
                Some("kimi_k25"),
                Some(&empty_args),
                false,
                "kimi_k25 + empty args → enabled",
            ),
            // deepseek_r1 uses "thinking" bool or "thinking_mode" string
            (
                Some("deepseek_r1"),
                Some(&thinking_false),
                true,
                "deepseek_r1 + thinking=false → disabled",
            ),
            (
                Some("deepseek_r1"),
                Some(&thinking_true),
                false,
                "deepseek_r1 + thinking=true → enabled",
            ),
            (
                Some("deepseek_r1"),
                Some(&thinking_mode_chat),
                true,
                "deepseek_r1 + thinking_mode=chat → disabled",
            ),
            (
                Some("deepseek_r1"),
                Some(&thinking_mode_thinking),
                false,
                "deepseek_r1 + thinking_mode=thinking → enabled",
            ),
            (
                Some("deepseek_r1"),
                None,
                false,
                "deepseek_r1 + no args → enabled",
            ),
            (
                Some("deepseek_r1"),
                Some(&empty_args),
                false,
                "deepseek_r1 + empty args → enabled",
            ),
            (
                Some("basic"),
                Some(&thinking_false),
                false,
                "basic → never disabled",
            ),
            (
                None,
                Some(&thinking_false),
                false,
                "no parser → never disabled",
            ),
            // nemotron_nano uses "enable_thinking" key
            (
                Some("nemotron_nano"),
                Some(&enable_thinking_false),
                true,
                "nemotron_nano + enable_thinking=false → disabled",
            ),
            (
                Some("nemotron_nano"),
                Some(&enable_thinking_true),
                false,
                "nemotron_nano + enable_thinking=true → enabled",
            ),
            (
                Some("nemotron_nano"),
                None,
                false,
                "nemotron_nano + no args → enabled",
            ),
            (
                Some("nemotron_nano"),
                Some(&empty_args),
                false,
                "nemotron_nano + empty args → enabled",
            ),
            // deepseek_v4 — same convention as deepseek_r1; verify all three aliases
            // (deepseek_v4 / deepseek-v4 / deepseekv4) plus both signal keys.
            (
                Some("deepseek_v4"),
                Some(&thinking_false),
                true,
                "deepseek_v4 + thinking=false → disabled",
            ),
            (
                Some("deepseek_v4"),
                Some(&thinking_true),
                false,
                "deepseek_v4 + thinking=true → enabled",
            ),
            (
                Some("deepseek_v4"),
                Some(&thinking_mode_chat),
                true,
                "deepseek_v4 + thinking_mode=chat → disabled",
            ),
            (
                Some("deepseek_v4"),
                Some(&thinking_mode_thinking),
                false,
                "deepseek_v4 + thinking_mode=thinking → enabled",
            ),
            (
                Some("deepseek_v4"),
                None,
                false,
                "deepseek_v4 + no args → enabled",
            ),
            (
                Some("deepseek-v4"),
                Some(&thinking_false),
                true,
                "deepseek-v4 (hyphen alias) + thinking=false → disabled",
            ),
            (
                Some("deepseekv4"),
                Some(&thinking_mode_chat),
                true,
                "deepseekv4 (joined alias) + thinking_mode=chat → disabled",
            ),
            (
                Some("deepseek_v4"),
                Some(&enable_thinking_false),
                true,
                "deepseek_v4 + enable_thinking=false → disabled (vLLM alias)",
            ),
            (
                Some("deepseek_v4"),
                Some(&enable_thinking_true),
                false,
                "deepseek_v4 + enable_thinking=true → enabled (vLLM alias)",
            ),
        ];

        for (parser, args, expected, desc) in cases {
            assert_eq!(
                OpenAIPreprocessor::is_reasoning_disabled_by_request(parser, args),
                expected,
                "FAILED: {desc}",
            );
        }
    }
}
