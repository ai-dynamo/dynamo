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
use dynamo_async_openai::types::EncodingFormat;
use futures::stream::{self, StreamExt};
use prompt::OAIPromptFormatter;
use rayon::iter::{IntoParallelRefIterator, ParallelIterator};
use std::{collections::HashMap, sync::Arc};
use tracing;

use dynamo_parsers::tool_calling::try_tool_call_parse_aggregate;

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

        Ok(Arc::new(Self {
            formatter,
            tokenizer,
            model_info,
            mdcsum,
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
        }

        let state = State {
            response_stream: stream,
            response_generator: generator,
            context: context.clone(),
            cancelled: false,
            cumulative_output_tokens: 0,
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
    pub fn apply_tool_calling_jail_with_parser(
        &self,
        stream: ManyOut<Annotated<NvCreateChatCompletionStreamResponse>>,
    ) -> ManyOut<Annotated<NvCreateChatCompletionStreamResponse>> {
        apply_tool_calling_jail_internal(stream, self.tool_call_parser.clone())
    }
}

/// Detect if the given text chunk indicates the start of a tool call
/// Checks for tool call start patterns based on the parser type
pub fn detect_tool_call_start(chunk: &str, parser_str: Option<&str>) -> anyhow::Result<bool> {
    let parser_name = parser_str.unwrap_or("default");
    
    // Check for common tool call start patterns based on parser type
    match parser_name {
        "nemotron_deci" | "default" => {
            // Check for <TOOLCALL> pattern
            Ok(chunk.contains("<TOOLCALL>"))
        }
        "hermes" => {
            // Check for <tool_call> pattern
            Ok(chunk.contains("<tool_call>"))
        }
        "phi4" => {
            // Check for functools[ pattern
            Ok(chunk.contains("functools["))
        }
        "mistral" | "llama3_json" => {
            // Check for various JSON array patterns or python tag
            Ok(chunk.contains("[{") || 
               chunk.contains("<|python_tag|>") || 
               chunk.contains("[TOOL_CALLS]"))
        }
        "pythonic" => {
            // Check for function call pattern like [function_name(
            Ok(chunk.contains("[") && chunk.contains("("))
        }
        "harmony" => {
            // Check for harmony-specific patterns
            Ok(chunk.contains("<|channel|>") && chunk.contains("functions."))
        }
        "deepseek_v3_1" => {
            // Check for deepseek patterns
            Ok(chunk.contains("｜tool▁calls▁begin｜") || chunk.contains("｜tool▁call▁begin｜"))
        }
        _ => {
            // For unknown parsers, check for common patterns
            Ok(chunk.contains("<TOOLCALL>") || 
               chunk.contains("<tool_call>") || 
               chunk.contains("functools[") ||
               chunk.contains("[{") ||
               chunk.contains("<|python_tag|>"))
        }
    }
}

/// Apply tool calling jail to the stream - stops/jails the stream under certain conditions
/// When jailed, the stream will be unjailed when the input stream ends
fn apply_tool_calling_jail_internal(
    stream: ManyOut<Annotated<NvCreateChatCompletionStreamResponse>>,
    tool_call_parser: Option<String>,
) -> ManyOut<Annotated<NvCreateChatCompletionStreamResponse>> {
    let context = stream.context();
    
    struct JailState {
        stream: ManyOut<Annotated<NvCreateChatCompletionStreamResponse>>,
        is_jailed: bool,
        tool_call_parser: Option<String>,
        accumulated_content: HashMap<u32, String>, // choice index -> accumulated content
        last_response_metadata: Option<NvCreateChatCompletionStreamResponse>, // for response structure
    }

    let jail_state = JailState {
        stream,
        is_jailed: false,
        tool_call_parser,
        accumulated_content: HashMap::new(),
        last_response_metadata: None,
    };

    // Transform the stream using unfold to maintain state
    let jailed_stream = stream::unfold(jail_state, |mut state| async move {
        if let Some(response) = state.stream.next().await {
            // Check if we should jail the stream
            if !state.is_jailed {
                // Handle the case where response.data is Option<T>
                if let Some(ref chat_response) = response.data {
                    // Store metadata for potential tool call parsing later
                    state.last_response_metadata = Some(chat_response.clone());
                    
                    // Extract text content from the response
                    if let Some(choice) = chat_response.choices.first() {
                        if let Some(ref content) = choice.delta.content {
                            // Check for tool call start
                            match detect_tool_call_start(
                                content,
                                state.tool_call_parser.as_deref(),
                            ) {
                                Ok(should_jail) => {
                                    if should_jail {
                                        tracing::debug!("Tool call detected, jailing stream");
                                        state.is_jailed = true;
                                        
                                        // Start accumulating content for this choice
                                        state.accumulated_content.insert(choice.index, content.clone());
                                        
                                        // Create possible tool call annotation with token information
                                        let possible_annotation = PossibleToolCallAnnotation {
                                            possible_tokens: 1, // This chunk contains tokens being processed
                                            possible_content: content.clone(),
                                            parser_used: state.tool_call_parser.clone(),
                                        };
                                        
                                        // Create annotated response instead of empty response
                                        let mut annotated_response = response.clone();
                                        if let Ok(possible_annotated) = possible_annotation.to_annotation::<NvCreateChatCompletionStreamResponse>() {
                                            // Set annotation event and comment
                                            annotated_response.event = possible_annotated.event;
                                            annotated_response.comment = possible_annotated.comment;
                                        }
                                        
                                        // Modify the response to have empty content but keep metadata
                                        annotated_response = annotated_response.map_data(|mut chat_response| {
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
                }
            } else if state.is_jailed {
                // If already jailed, continue to jail but with annotations and accumulate content
                if let Some(ref chat_response) = response.data {
                    // Extract content for annotation and accumulation
                    for choice in &chat_response.choices {
                        if let Some(ref content) = choice.delta.content {
                            if !content.is_empty() {
                                // Accumulate content for this choice
                                state.accumulated_content
                                    .entry(choice.index)
                                    .or_insert_with(String::new)
                                    .push_str(content);
                                
                                // Create possible tool call annotation
                                let possible_annotation = PossibleToolCallAnnotation {
                                    possible_tokens: 1,
                                    possible_content: content.clone(),
                                    parser_used: state.tool_call_parser.clone(),
                                };
                                
                                // Create annotated response
                                let mut annotated_response = response.clone();
                                if let Ok(possible_annotated) = possible_annotation.to_annotation::<NvCreateChatCompletionStreamResponse>() {
                                    annotated_response.event = possible_annotated.event;
                                    annotated_response.comment = possible_annotated.comment;
                                }
                                
                                // Clear content but keep structure
                                annotated_response = annotated_response.map_data(|mut chat_response| {
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
            }

            // If not jailed or jailing condition not met, return the response as-is
            Some((response, state))
        } else {
            // Stream ended - if we were jailed, we should unjail now and parse tool calls
            if state.is_jailed {
                tracing::debug!("Stream ended, unjailing and parsing accumulated content");
                state.is_jailed = false;
                
                // Parse accumulated content for tool calls
                if !state.accumulated_content.is_empty() {
                    if let Some(base_response) = state.last_response_metadata.take() {
                        // Try to parse tool calls from accumulated content for each choice
                        let mut final_response = base_response.clone();
                        
                        for (choice_index, accumulated_text) in &state.accumulated_content {
                            if let Ok((tool_calls, normal_text)) = try_tool_call_parse_aggregate(
                                accumulated_text,
                                state.tool_call_parser.as_deref(),
                            ) {
                                if !tool_calls.is_empty() {
                                    // Found tool calls, create a final response with them
                                    tracing::debug!("Parsed {} tool calls from accumulated content", tool_calls.len());
                                    
                                    for tool_call in &tool_calls {
                                        tracing::debug!(
                                            tool_call_id = %tool_call.id,
                                            function_name = %tool_call.function.name,
                                            arguments = %tool_call.function.arguments,
                                            "Parsed structured tool call from accumulated content in jail"
                                        );
                                    }
                                    
                                    // Convert ChatCompletionMessageToolCall to ChatCompletionMessageToolCallChunk for streaming
                                    let tool_call_chunks: Vec<dynamo_async_openai::types::ChatCompletionMessageToolCallChunk> = tool_calls
                                        .into_iter()
                                        .enumerate()
                                        .map(|(idx, tool_call)| dynamo_async_openai::types::ChatCompletionMessageToolCallChunk {
                                            index: idx as u32,
                                            id: Some(tool_call.id),
                                            r#type: Some(tool_call.r#type),
                                            function: Some(dynamo_async_openai::types::FunctionCallStream {
                                                name: Some(tool_call.function.name),
                                                arguments: Some(tool_call.function.arguments),
                                            }),
                                        })
                                        .collect();
                                    
                                    // Create a choice with tool calls
                                    #[allow(deprecated)]
                                    let final_choice = dynamo_async_openai::types::ChatChoiceStream {
                                        index: *choice_index,
                                        delta: dynamo_async_openai::types::ChatCompletionStreamResponseDelta {
                                            role: Some(dynamo_async_openai::types::Role::Assistant),
                                            content: if let Some(text) = normal_text.filter(|t| !t.is_empty()) {
                                                Some(text)
                                            } else {
                                                None
                                            },
                                            tool_calls: Some(tool_call_chunks),
                                            function_call: None,
                                            refusal: None,
                                            reasoning_content: None,
                                        },
                                        finish_reason: Some(dynamo_async_openai::types::FinishReason::ToolCalls),
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
                                    
                                    return Some((final_annotated, state));
                                }
                            }
                        }
                    }
                }
            }
            None
        }
    });

    ResponseStream::new(Box::pin(jailed_stream), context)
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

        let stream = self.apply_tool_calling_jail_with_parser(stream);
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

#[allow(deprecated)]
#[cfg(test)]
mod tests {
    use super::*;
    use futures::stream::{self, StreamExt};
    use dynamo_async_openai::types::{
        ChatChoiceStream, ChatCompletionStreamResponseDelta, Role, FinishReason as OAIFinishReason
    };
    use dynamo_runtime::protocols::annotated::Annotated;
    use dynamo_runtime::pipeline::ResponseStream;
    use std::sync::Arc;

    // Helper function to create a mock chat response chunk
    fn create_mock_response_chunk(content: String, index: u32) -> Annotated<NvCreateChatCompletionStreamResponse> {
        let choice = ChatChoiceStream {
            index,
            delta: ChatCompletionStreamResponseDelta {
                role: Some(Role::Assistant),
                content: Some(content),
                tool_calls: None,
                function_call: None,
                refusal: None,
                reasoning_content: None,
            },
            finish_reason: None,
            logprobs: None,
        };

        let response = NvCreateChatCompletionStreamResponse {
            id: "test-id".to_string(),
            choices: vec![choice],
            created: 1234567890,
            model: "test-model".to_string(),
            system_fingerprint: Some("test-fingerprint".to_string()),
            object: "chat.completion.chunk".to_string(),
            usage: None,
            service_tier: None,
        };

        Annotated {
            data: Some(response),
            id: None,
            event: None,
            comment: None,
        }
    }

    // Helper function to create a final response chunk with finish reason
    fn create_final_response_chunk(index: u32) -> Annotated<NvCreateChatCompletionStreamResponse> {
        let choice = ChatChoiceStream {
            index,
            delta: ChatCompletionStreamResponseDelta {
                role: None,
                content: None,
                tool_calls: None,
                function_call: None,
                refusal: None,
                reasoning_content: None,
            },
            finish_reason: Some(OAIFinishReason::Stop),
            logprobs: None,
        };

        let response = NvCreateChatCompletionStreamResponse {
            id: "test-id".to_string(),
            choices: vec![choice],
            created: 1234567890,
            model: "test-model".to_string(),
            system_fingerprint: Some("test-fingerprint".to_string()),
            object: "chat.completion.chunk".to_string(),
            usage: None,
            service_tier: None,
        };

        Annotated {
            data: Some(response),
            id: None,
            event: None,
            comment: None,
        }
    }

    // Mock async engine context for testing
    #[derive(Debug)]
    struct MockAsyncEngineContext {
        id: String,
        stopped: std::sync::atomic::AtomicBool,
    }

    impl MockAsyncEngineContext {
        fn new(id: String) -> Self {
            Self {
                id,
                stopped: std::sync::atomic::AtomicBool::new(false),
            }
        }
    }

    #[async_trait]
    impl dynamo_runtime::pipeline::AsyncEngineContext for MockAsyncEngineContext {
        fn id(&self) -> &str {
            &self.id
        }

        fn stop(&self) {
            self.stopped.store(true, std::sync::atomic::Ordering::Relaxed);
        }

        fn stop_generating(&self) {
            self.stopped.store(true, std::sync::atomic::Ordering::Relaxed);
        }

        fn kill(&self) {
            self.stopped.store(true, std::sync::atomic::Ordering::Relaxed);
        }

        fn is_stopped(&self) -> bool {
            self.stopped.load(std::sync::atomic::Ordering::Relaxed)
        }

        fn is_killed(&self) -> bool {
            self.stopped.load(std::sync::atomic::Ordering::Relaxed)
        }

        async fn stopped(&self) {
            // No-op for testing
        }

        async fn killed(&self) {
            // No-op for testing
        }

        fn link_child(&self, _: Arc<dyn dynamo_runtime::pipeline::AsyncEngineContext>) {
            // No-op for testing
        }
    }

    #[tokio::test]
    async fn test_apply_tool_calling_jail_internal_with_tool_call_detection() {
        // Create a stream with tool call content that SHOULD trigger jailing
        let mock_context = Arc::new(MockAsyncEngineContext::new("test-request-id".to_string()));
        
        // Create chunks that represent a tool call being generated
        let chunks = vec![
            create_mock_response_chunk("<TOOLCALL>".to_string(), 0),
            create_mock_response_chunk("[{\"name\": \"get_weather\", ".to_string(), 0),
            create_mock_response_chunk("\"arguments\": {\"location\": \"San Francisco\"}}]".to_string(), 0),
            create_mock_response_chunk("</TOOLCALL>".to_string(), 0),
        ];

        let input_stream = stream::iter(chunks);
        let response_stream = ResponseStream::new(Box::pin(input_stream), mock_context.clone());

        // Apply the jail with nemotron_deci parser - should trigger jailing on first chunk
        let jailed_stream = apply_tool_calling_jail_internal(response_stream, Some("nemotron_deci".to_string()));

        // Collect all results
        let results: Vec<_> = jailed_stream.collect().await;

        // Verify that jailing was triggered
        assert!(!results.is_empty(), "Should have some results");

        // Find the result that triggered jailing (first chunk with <TOOLCALL>)
        let first_result = &results[0];
        if let Some(ref response_data) = first_result.data {
            // First chunk should trigger jailing - content should be emptied
            assert!(
                response_data.choices[0].delta.content.as_ref().map_or(true, |c| c.is_empty()),
                "First chunk should have empty content after jailing"
            );
            // Should have annotation event indicating possible tool call
            assert!(first_result.event.is_some(), "First chunk should have annotation event");
            assert_eq!(first_result.event.as_deref(), Some(ANNOTATION_POSSIBLE_TOOL_CALL));
        }

        // Subsequent chunks while jailed should also have empty content but with annotations
        for (i, result) in results.iter().enumerate().skip(1) {
            if let Some(ref response_data) = result.data {
                // While jailed, all chunks should have empty content
                if response_data.choices[0].delta.content.is_some() {
                    assert!(
                        response_data.choices[0].delta.content.as_ref().unwrap().is_empty(),
                        "Chunk {} should have empty content while jailed", i
                    );
                }
                // Should have annotation events for content accumulated during jailing
                if response_data.choices[0].delta.content.is_some() {
                    assert!(result.event.is_some(), "Jailed chunk {} should have annotation event", i);
                }
            }
        }

        // The last result might be the parsed tool call result when stream ends and unjails
        if let Some(last_result) = results.last() {
            if let Some(ref response_data) = last_result.data {
                // Check if tool calls were parsed and included after unjailing
                if let Some(ref tool_calls) = response_data.choices[0].delta.tool_calls {
                    assert!(!tool_calls.is_empty(), "Should have parsed tool calls");
                    assert_eq!(tool_calls[0].function.as_ref().unwrap().name.as_ref().unwrap(), "get_weather");
                }
            }
        }
    }

    #[tokio::test]
    async fn test_apply_tool_calling_jail_internal_no_tool_calls() {
        // Create a stream with regular content that should NOT trigger jailing
        let mock_context = Arc::new(MockAsyncEngineContext::new("test-request-id-2".to_string()));
        
        let chunks = vec![
            create_mock_response_chunk("Hello, ".to_string(), 0),
            create_mock_response_chunk("how can I ".to_string(), 0),
            create_mock_response_chunk("help you today?".to_string(), 0),
            create_final_response_chunk(0),
        ];

        let input_stream = stream::iter(chunks);
        let response_stream = ResponseStream::new(Box::pin(input_stream), mock_context.clone());

        // Apply the jail with nemotron_deci parser - regular text should NOT be jailed
        let jailed_stream = apply_tool_calling_jail_internal(response_stream, Some("nemotron_deci".to_string()));

        // Collect all results
        let results: Vec<_> = jailed_stream.collect().await;

        // Should have results and they should NOT be jailed (content should be preserved)
        assert!(!results.is_empty(), "Should have results");
        assert_eq!(results.len(), 4, "Should have all 4 chunks");

        // Verify that content is NOT jailed - first few chunks should have their original content
        for (i, result) in results.iter().take(3).enumerate() {
            if let Some(ref response_data) = result.data {
                let expected_content = match i {
                    0 => "Hello, ",
                    1 => "how can I ",
                    2 => "help you today?",
                    _ => unreachable!(),
                };
                assert_eq!(
                    response_data.choices[0].delta.content.as_deref(),
                    Some(expected_content),
                    "Chunk {} should have original content, not be jailed",
                    i
                );
                // Should NOT have annotation events for regular content
                assert!(result.event.is_none(), "Regular content should not have annotation events");
            }
        }

        // Last chunk should be the final response with finish reason
        if let Some(last_result) = results.last() {
            if let Some(ref response_data) = last_result.data {
                assert_eq!(response_data.choices[0].finish_reason, Some(OAIFinishReason::Stop));
            }
        }
    }

    #[tokio::test]
    async fn test_apply_tool_calling_jail_internal_with_empty_stream() {
        let mock_context = Arc::new(MockAsyncEngineContext::new("test-request-id-3".to_string()));
        
        let chunks: Vec<Annotated<NvCreateChatCompletionStreamResponse>> = vec![];
        let input_stream = stream::iter(chunks);
        let response_stream = ResponseStream::new(Box::pin(input_stream), mock_context.clone());

        let jailed_stream = apply_tool_calling_jail_internal(response_stream, None);
        let results: Vec<_> = jailed_stream.collect().await;

        assert!(results.is_empty(), "Empty stream should produce no results");
    }

    #[tokio::test]
    async fn test_apply_tool_calling_jail_internal_with_different_parsers() {
        let mock_context = Arc::new(MockAsyncEngineContext::new("test-request-id-4".to_string()));
        
        // Test with hermes parser format
        let chunks = vec![
            create_mock_response_chunk("<tool_call>".to_string(), 0),
            create_mock_response_chunk("{\"name\": \"get_weather\", \"arguments\": {\"location\": \"Tokyo\"}}".to_string(), 0),
            create_mock_response_chunk("</tool_call>".to_string(), 0),
        ];

        let input_stream = stream::iter(chunks);
        let response_stream = ResponseStream::new(Box::pin(input_stream), mock_context.clone());

        let jailed_stream = apply_tool_calling_jail_internal(response_stream, Some("hermes".to_string()));
        let results: Vec<_> = jailed_stream.collect().await;

        assert!(!results.is_empty(), "Should have results for hermes parser");
    }

    #[tokio::test]
    async fn test_detect_tool_call_start_different_parsers() {
        // Test nemotron_deci parser
        assert!(detect_tool_call_start("<TOOLCALL>", Some("nemotron_deci")).unwrap());
        assert!(!detect_tool_call_start("Hello world", Some("nemotron_deci")).unwrap());
        assert!(!detect_tool_call_start("<tool_call>", Some("nemotron_deci")).unwrap()); // Wrong format

        // Test hermes parser
        assert!(detect_tool_call_start("<tool_call>", Some("hermes")).unwrap());
        assert!(!detect_tool_call_start("Hello world", Some("hermes")).unwrap());
        assert!(!detect_tool_call_start("<TOOLCALL>", Some("hermes")).unwrap()); // Wrong format

        // Test phi4 parser
        assert!(detect_tool_call_start("functools[", Some("phi4")).unwrap());
        assert!(!detect_tool_call_start("Hello world", Some("phi4")).unwrap());

        // Test mistral parser
        assert!(detect_tool_call_start("[{", Some("mistral")).unwrap());
        assert!(detect_tool_call_start("<|python_tag|>", Some("mistral")).unwrap());
        assert!(detect_tool_call_start("[TOOL_CALLS]", Some("mistral")).unwrap());
        assert!(!detect_tool_call_start("Hello world", Some("mistral")).unwrap());

        // Test default parser (should behave like nemotron_deci)
        assert!(detect_tool_call_start("<TOOLCALL>", None).unwrap());
        assert!(!detect_tool_call_start("Hello world", None).unwrap());
    }

    #[tokio::test]
    async fn test_apply_tool_calling_jail_internal_hermes_parser() {
        // Test with hermes parser format
        let mock_context = Arc::new(MockAsyncEngineContext::new("test-request-id-hermes".to_string()));
        
        let chunks = vec![
            create_mock_response_chunk("I'll help you with that. ".to_string(), 0),
            create_mock_response_chunk("<tool_call>".to_string(), 0), // This should trigger jailing
            create_mock_response_chunk("{\"name\": \"get_weather\", \"arguments\": {\"location\": \"Tokyo\"}}".to_string(), 0),
            create_mock_response_chunk("</tool_call>".to_string(), 0),
        ];

        let input_stream = stream::iter(chunks);
        let response_stream = ResponseStream::new(Box::pin(input_stream), mock_context.clone());

        let jailed_stream = apply_tool_calling_jail_internal(response_stream, Some("hermes".to_string()));
        let results: Vec<_> = jailed_stream.collect().await;

        assert!(!results.is_empty(), "Should have results for hermes parser");

        // First chunk should pass through normally (no tool call pattern)
        if let Some(ref first_result) = results.first() {
            if let Some(ref response_data) = first_result.data {
                assert_eq!(
                    response_data.choices[0].delta.content.as_deref(),
                    Some("I'll help you with that. "),
                    "First chunk should pass through normally"
                );
                assert!(first_result.event.is_none(), "First chunk should not have annotation");
            }
        }

        // Second chunk should trigger jailing
        if results.len() > 1 {
            let second_result = &results[1];
            if let Some(ref response_data) = second_result.data {
                assert!(
                    response_data.choices[0].delta.content.as_ref().map_or(true, |c| c.is_empty()),
                    "Second chunk should be jailed (empty content)"
                );
                assert!(second_result.event.is_some(), "Second chunk should have annotation event");
            }
        }
    }

    #[tokio::test]
    async fn test_possible_tool_call_annotation_serialization() {
        let annotation = PossibleToolCallAnnotation {
            possible_tokens: 5,
            possible_content: "test content".to_string(),
            parser_used: Some("nemotron_deci".to_string()),
        };

        let annotated_result = annotation.to_annotation::<NvCreateChatCompletionStreamResponse>();
        assert!(annotated_result.is_ok(), "Should be able to create annotation");

        let annotated = annotated_result.unwrap();
        assert_eq!(annotated.event, Some(ANNOTATION_POSSIBLE_TOOL_CALL.to_string()));
        assert!(annotated.comment.is_some(), "Should have comment");

        // Test deserialization
        let parsed_annotation = PossibleToolCallAnnotation::from_annotation(&annotated);
        assert!(parsed_annotation.is_ok(), "Should be able to parse annotation");

        let parsed = parsed_annotation.unwrap();
        assert!(parsed.is_some(), "Should have parsed annotation");

        let parsed = parsed.unwrap();
        assert_eq!(parsed.possible_tokens, 5);
        assert_eq!(parsed.possible_content, "test content");
        assert_eq!(parsed.parser_used, Some("nemotron_deci".to_string()));
    }
}
