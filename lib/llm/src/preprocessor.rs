// SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

pub mod prompt;
pub mod tools;

use anyhow::Result;
use dynamo_async_openai::types::ChatCompletionToolChoiceOption;
use dynamo_async_openai::types::EncodingFormat;
use futures::stream::{self, StreamExt};
use prompt::OAIPromptFormatter;
use rayon::iter::{IntoParallelRefIterator, ParallelIterator};
use std::{collections::HashMap, sync::Arc};
use tracing;

use dynamo_parsers::tool_calling::{
    parsers::detect_tool_call_start, try_tool_call_parse_aggregate,
};

use crate::model_card::{ModelDeploymentCard, ModelInfo};
use crate::preprocessor::prompt::OAIChatLikeRequest;
use crate::protocols::common::preprocessor::PreprocessedRequestBuilder;
use crate::tokenizers::Encoding;

use dynamo_runtime::engine::{AsyncEngine, AsyncEngineContextProvider, ResponseStream};
use dynamo_runtime::pipeline::{
    AsyncEngineContext, Error, ManyOut, Operator, SingleIn, async_trait,
};
use dynamo_runtime::protocols::annotated::{Annotated, AnnotationsProvider};

use crate::protocols::{
    common::{OutputOptionsProvider, SamplingOptionsProvider, StopConditionsProvider},
    openai::{
        DeltaGeneratorExt,
        chat_completions::{NvCreateChatCompletionRequest, NvCreateChatCompletionStreamResponse},
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
pub const ANNOTATION_POSSIBLE_TOOL_CALL: &str = "possible_tool_call";
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct LLMMetricAnnotation {
    pub input_tokens: usize,
    pub output_tokens: usize,
    pub chunk_tokens: usize,
}

#[derive(Debug)]
pub struct JailState {
    stream: ManyOut<Annotated<NvCreateChatCompletionStreamResponse>>,
    is_jailed: bool,
    tool_call_parser: Option<String>,
    accumulated_content: HashMap<u32, String>, // choice index -> accumulated content
    last_response_metadata: Option<NvCreateChatCompletionStreamResponse>, // for response structure
    finished: bool,                            // Add this flag to track if stream is finished
}

pub fn maybe_enable_tool_call(
    parser_str: Option<&str>,
    request: &NvCreateChatCompletionRequest,
) -> bool {
    // Enable tool call if the below two conditions are satisfied
    // 1. parser_str is not None
    // 2. tool_choice is not None
    parser_str.is_some()
        && !matches!(
            request.inner.tool_choice,
            Some(ChatCompletionToolChoiceOption::None)
        )
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

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct PossibleToolCallAnnotation {
    pub possible_tokens: usize,
    pub possible_content: String,
    pub parser_used: Option<String>,
}

impl PossibleToolCallAnnotation {
    /// Convert this possible tool call annotation to an Annotated event
    pub fn to_annotation<T>(&self) -> Result<Annotated<T>, serde_json::Error> {
        Annotated::from_annotation(ANNOTATION_POSSIBLE_TOOL_CALL, self)
    }

    /// Extract possible tool call info from an Annotated event, if present
    pub fn from_annotation<T>(
        annotation: &Annotated<T>,
    ) -> Result<Option<PossibleToolCallAnnotation>, Box<dyn std::error::Error>> {
        if annotation.event.is_none() {
            return Ok(None);
        }
        if annotation.event.as_ref().unwrap() != ANNOTATION_POSSIBLE_TOOL_CALL {
            return Ok(None);
        }
        let comments = annotation
            .comment
            .as_ref()
            .ok_or("missing comments block")?;
        if comments.len() != 1 {
            return Err("malformed comments block - expected exactly 1 comment".into());
        }
        let possible_info: PossibleToolCallAnnotation = serde_json::from_str(&comments[0])?;
        Ok(Some(possible_info))
    }
}

pub struct OpenAIPreprocessor {
    mdcsum: String,
    formatter: Arc<dyn OAIPromptFormatter>,
    tokenizer: Arc<dyn Tokenizer>,
    model_info: Arc<dyn ModelInfo>,
    /// Per-model runtime configuration propagated to response generator (e.g., reasoning/tool parser)
    runtime_config: crate::local_model::runtime_config::ModelRuntimeConfig,
    tool_call_parser: Option<String>,
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
        let mdcsum = mdc.mdcsum();
        let tokenizer = Arc::new(HuggingFaceTokenizer::from_tokenizer(hf_tokenizer));
        let Some(model_info) = mdc.model_info else {
            anyhow::bail!(
                "Blank ModelDeploymentCard cannot be used for pre-processing, no model_info"
            );
        };
        let model_info = model_info.get_model_info()?;
        let tool_call_parser = mdc.runtime_config.tool_call_parser.clone();

        // // Initialize runtime config from the ModelDeploymentCard
        let runtime_config = mdc.runtime_config.clone();

        Ok(Arc::new(Self {
            formatter,
            tokenizer,
            model_info,
            mdcsum,
            runtime_config,
            tool_call_parser,
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
            + OutputOptionsProvider
            + NvExtProvider,
    >(
        &self,
        request: &R,
    ) -> Result<(PreprocessedRequest, HashMap<String, String>)> {
        let mut builder = self.builder(request)?;
        let formatted_prompt = self.apply_template(request)?;
        let annotations = self.gather_tokens(request, &mut builder, formatted_prompt)?;

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
        builder.estimated_prefix_hit_num_blocks(None);
        // Extract backend_instance_id from nvext if present
        if let Some(nvext) = request.nvext() {
            builder.backend_instance_id(nvext.backend_instance_id);
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
                                builder.batch_token_ids(Some(token_batches));
                                builder.token_ids(vec![]);
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
                            let token_batches: Vec<Vec<u32>> = texts
                                .par_iter()
                                .map(|text| {
                                    self.tokenizer
                                        .encode(text)
                                        .map(|encoded| encoded.token_ids().to_vec())
                                })
                                .collect::<Result<Vec<_>>>()?;
                            builder.batch_token_ids(Some(token_batches));
                            builder.token_ids(vec![]);
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
            cumulative_output_tokens: usize,
            finish_reason_sent: bool,
            usage_chunk_sent: bool,
            finished: bool, // Add this flag to track if stream is finished
        }

        let state = State {
            response_stream: stream,
            response_generator: generator,
            context: context.clone(),
            cancelled: false,
            cumulative_output_tokens: 0,
            finish_reason_sent: false,
            usage_chunk_sent: false,
            finished: false, // Initialize as not finished
        };

        // transform the common response stream into a chat response stream
        let stream = stream::unfold(state, |mut inner| {
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
                    // Stream has ended - check if we need to send a usage chunk
                    if inner.response_generator.is_usage_enabled()
                        && inner.finish_reason_sent
                        && !inner.usage_chunk_sent
                        && !inner.finished
                    {
                        inner.usage_chunk_sent = true;

                        // Create the final usage chunk
                        let usage_chunk = inner.response_generator.create_usage_chunk();
                        let annotated_usage = Annotated::<Resp> {
                            id: None,
                            data: Some(usage_chunk),
                            event: Some(ANNOTATION_LLM_METRICS.to_string()),
                            comment: None,
                        };

                        tracing::trace!(
                            request_id = inner.context.id(),
                            "Sending final usage chunk for OpenAI compliance"
                        );

                        Some((annotated_usage, inner))
                    } else {
                        // stream closed
                        inner.finished = true; // Mark as finished
                        None
                    }
                }
            }
        });

        ResponseStream::new(Box::pin(stream), context)
    }

    /// Transform engine embedding output stream to OpenAI embedding response stream
    pub fn transform_embedding_postprocessor_stream(
        stream: ManyOut<Annotated<EmbeddingsEngineOutput>>,
        original_request: NvCreateEmbeddingRequest,
    ) -> ManyOut<Annotated<NvCreateEmbeddingResponse>> {
        let context = stream.context();

        let transformed_stream = stream.map(move |output| {
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
        });

        ResponseStream::new(Box::pin(transformed_stream), context)
    }

    /// Apply tool calling jail to the stream using the preprocessor's tool call parser
    pub async fn apply_tool_calling_jail_with_parser(
        &self,
        stream: ManyOut<Annotated<NvCreateChatCompletionStreamResponse>>,
    ) -> ManyOut<Annotated<NvCreateChatCompletionStreamResponse>> {
        apply_tool_calling_jail_internal(stream, self.tool_call_parser.clone()).await
    }
}

/// Apply tool calling jail to the stream - stops/jails the stream under certain conditions
/// When jailed, the stream will be unjailed when the input stream ends
pub async fn apply_tool_calling_jail_internal(
    stream: ManyOut<Annotated<NvCreateChatCompletionStreamResponse>>,
    tool_call_parser: Option<String>,
) -> ManyOut<Annotated<NvCreateChatCompletionStreamResponse>> {
    let context = stream.context();

    let jail_state = JailState {
        stream,
        is_jailed: false,
        tool_call_parser,
        accumulated_content: HashMap::new(),
        last_response_metadata: None,
        finished: false,
    };

    // Transform the stream using unfold to maintain state
    // Input: ManyOut<Annotated<NvCreateChatCompletionStreamResponse>>
    // Returns None if the stream is finished
    // Returns Some((Annotated<NvCreateChatCompletionStreamResponse>, JailState)) if the stream is not finished
    // End output: ManyOut<Annotated<NvCreateChatCompletionStreamResponse>>
    let jailed_stream = stream::unfold(jail_state, |mut state| async move {
        // If already finished, return None immediately
        if state.finished {
            return None;
        }

        if let Some(response) = state.stream.next().await {
            // Check if we should jail the stream
            if !state.is_jailed {
                // Handle the case where response.data is Option<T>
                if let Some(ref chat_response) = response.data {
                    // Store metadata for potential tool call parsing later
                    state.last_response_metadata = Some(chat_response.clone());

                    // Extract text content from the response
                    if let Some(choice) = chat_response.choices.first()
                        && let Some(ref content) = choice.delta.content
                    {
                        // Check for tool call start
                        match detect_tool_call_start(content, state.tool_call_parser.as_deref()) {
                            Ok(should_jail) => {
                                if should_jail {
                                    tracing::debug!("Tool call detected, jailing stream");
                                    state.is_jailed = true;

                                    // Start accumulating content for this choice
                                    state
                                        .accumulated_content
                                        .insert(choice.index, content.clone());

                                    // Create possible tool call annotation with token information
                                    let possible_annotation = PossibleToolCallAnnotation {
                                        possible_tokens: 1, // This chunk contains tokens being processed
                                        possible_content: content.clone(),
                                        parser_used: state.tool_call_parser.clone(),
                                    };

                                    // Create annotated response instead of empty response
                                    let mut annotated_response = response.clone();
                                    if let Ok(possible_annotated) =
                                        possible_annotation
                                            .to_annotation::<NvCreateChatCompletionStreamResponse>()
                                    {
                                        // Set annotation event and comment
                                        annotated_response.event = possible_annotated.event;
                                        annotated_response.comment = possible_annotated.comment;
                                    }

                                    // Modify the response to have empty content but keep metadata
                                    annotated_response =
                                        annotated_response.map_data(|mut chat_response| {
                                            // Clear the content but keep choice structure for ITL measurement
                                            for choice in &mut chat_response.choices {
                                                choice.delta.content = Some(String::new()); // Empty content
                                            }
                                            Ok(chat_response)
                                        });

                                    return Some((annotated_response, state));
                                }
                            }
                            Err(e) => {
                                tracing::warn!("Error detecting tool call start: {}", e);
                            }
                        }
                    }
                }
            } else if state.is_jailed {
                // If already jailed, continue to jail but with annotations and accumulate content
                if let Some(ref chat_response) = response.data {
                    // Extract content for annotation and accumulation
                    for choice in &chat_response.choices {
                        if let Some(ref content) = choice.delta.content
                            && !content.is_empty()
                        {
                            // Accumulate content for this choice
                            state
                                .accumulated_content
                                .entry(choice.index)
                                .or_default()
                                .push_str(content);

                            // Create possible tool call annotation
                            let possible_annotation = PossibleToolCallAnnotation {
                                possible_tokens: 1,
                                possible_content: content.clone(),
                                parser_used: state.tool_call_parser.clone(),
                            };

                            // Create annotated response
                            let mut annotated_response = response.clone();
                            if let Ok(possible_annotated) = possible_annotation
                                .to_annotation::<NvCreateChatCompletionStreamResponse>(
                            ) {
                                annotated_response.event = possible_annotated.event;
                                annotated_response.comment = possible_annotated.comment;
                            }

                            // Clear content but keep structure
                            annotated_response =
                                annotated_response.map_data(|mut chat_response| {
                                    for choice in &mut chat_response.choices {
                                        choice.delta.content = Some(String::new());
                                    }
                                    Ok(chat_response)
                                });

                            return Some((annotated_response, state));
                        }
                    }
                }
            }

            // If not jailed or jailing condition not met, return the response as-is
            Some((response, state))
        } else {
            // Stream ended - if we were jailed, we should unjail now and parse tool calls
            if state.is_jailed {
                tracing::debug!("Stream ended, unjailing and parsing accumulated content");
                state.is_jailed = false;

                // Parse accumulated content for tool calls
                if !state.accumulated_content.is_empty()
                    && let Some(base_response) = state.last_response_metadata.take()
                {
                    // Try to parse tool calls from accumulated content for each choice
                    let mut final_response = base_response.clone();

                    for (choice_index, accumulated_text) in &state.accumulated_content {
                        if let Ok((tool_calls, normal_text)) = try_tool_call_parse_aggregate(
                            accumulated_text,
                            state.tool_call_parser.as_deref(),
                        )
                        .await
                        {
                            // Found tool calls, create a final response with them
                            tracing::debug!(
                                "Parsed {} tool calls from accumulated content",
                                tool_calls.len()
                            );
                            for tool_call in &tool_calls {
                                tracing::debug!(
                                    tool_call_id = %tool_call.id,
                                    function_name = %tool_call.function.name,
                                    arguments = %tool_call.function.arguments,
                                    "Parsed structured tool call from accumulated content in jail"
                                );
                            }

                            // Convert ChatCompletionMessageToolCall to ChatCompletionMessageToolCallChunk for streaming
                            let tool_call_chunks: Vec<
                                dynamo_async_openai::types::ChatCompletionMessageToolCallChunk,
                            > = tool_calls
                                .into_iter()
                                .enumerate()
                                .map(|(idx, tool_call)| {
                                    dynamo_async_openai::types::ChatCompletionMessageToolCallChunk {
                                        index: idx as u32,
                                        id: Some(tool_call.id),
                                        r#type: Some(tool_call.r#type),
                                        function: Some(
                                            dynamo_async_openai::types::FunctionCallStream {
                                                name: Some(tool_call.function.name),
                                                arguments: Some(tool_call.function.arguments),
                                            },
                                        ),
                                    }
                                })
                                .collect();

                            // Create a choice with tool calls
                            #[allow(deprecated)]
                            let final_choice = dynamo_async_openai::types::ChatChoiceStream {
                                index: *choice_index,
                                delta:
                                    dynamo_async_openai::types::ChatCompletionStreamResponseDelta {
                                        role: Some(dynamo_async_openai::types::Role::Assistant),
                                        content: normal_text.filter(|t| !t.is_empty()),
                                        tool_calls: Some(tool_call_chunks.clone()),
                                        function_call: None,
                                        refusal: None,
                                        reasoning_content: None,
                                    },
                                finish_reason: Some(
                                    dynamo_async_openai::types::FinishReason::ToolCalls,
                                ),
                                logprobs: None,
                            };

                            // Update the response choices
                            final_response.choices = vec![final_choice];

                            // Create final annotated response
                            let final_annotated = Annotated {
                                data: Some(final_response),
                                id: None,
                                event: None,
                                comment: None,
                            };

                            state.finished = true; // Mark as finished before returning
                            return Some((final_annotated, state));
                        }
                    }
                }
            }
            state.finished = true; // Mark as finished
            None
        }
    });

    // Jailed Stream contains empty content chunks with annotation event "possible_tool_call" whenever the stream is jailed
    // This is a bad UX for the user, as they have to see a lot of empty content chunks
    // Filter out the empty content chunks with annotation event "possible_tool_call"
    let filtered_stream = jailed_stream.filter(|annotated| {
        let keep = annotated.event.as_deref() != Some(ANNOTATION_POSSIBLE_TOOL_CALL);
        async move { keep }
    });

    ResponseStream::new(Box::pin(filtered_stream), context)
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
        let (request, context) = request.into_parts();

        // create a response generator
        let response_generator = request.response_generator(context.id().to_string());
        let mut response_generator = Box::new(response_generator);

        // set the runtime configuration
        response_generator.set_reasoning_parser(self.runtime_config.clone());
        let enable_tool_calling =
            maybe_enable_tool_call(self.tool_call_parser.as_deref(), &request);
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

        // Apply tool calling jail to the stream if tool call parser is present
        let stream = if enable_tool_calling {
            self.apply_tool_calling_jail_with_parser(stream).await
        } else {
            stream
        };

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
        let (request, context) = request.into_parts();

        // create a response generator
        let response_generator = request.response_generator(context.id().to_string());
        let mut response_generator = Box::new(response_generator);
        // convert the chat completion request to a common completion request
        let mut builder = self.builder(&request)?;
        let annotations = self.gather_tokens(&request, &mut builder, None)?;
        let common_request = builder.build()?;

        // update isl
        response_generator.update_isl(common_request.token_ids.len() as u32);

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

        // Transform response stream back to OpenAI format
        let stream = Self::transform_embedding_postprocessor_stream(response_stream, request);
        let context = stream.context();

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
