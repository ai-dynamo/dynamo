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

//! Engine Protocols
//! ================
//!
//! This module contains the protocols in public API for the LLM Engine and AsyncEngine facades.
//!
//! The core components are the `CompletionRequest` and `StreamingCompletionResponse` objects.
//!
//! The `StreamingCompletionResponse` objects are the outputs of the LLM Engine; however, we
//! need some additional information to propagate intermediate results for improved observability.
//! The metadata is transferred via the other arms of the `StreamingResponse` enum.
//!
use std::collections::HashMap;
use std::time::SystemTime;

use anyhow::Result;
use derive_builder::Builder;
use serde::ser::SerializeStruct;
use serde::{Deserialize, Deserializer, Serialize, Serializer};

use super::TokenIdType;

pub mod llm_backend;
pub mod postprocessor;
pub mod preprocessor;

/// SamplingOptionsProvider is a trait that allows the caller to extract the sampling options from
/// the object that implements it. This will mutate the object.
pub trait SamplingOptionsProvider {
    fn extract_sampling_options(&self) -> Result<SamplingOptions>;
}

pub trait StopConditionsProvider {
    fn extract_stop_conditions(&self) -> Result<StopConditions>;
}

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq, Eq)]
pub enum FinishReason {
    #[serde(rename = "eos")]
    EoS,

    #[serde(rename = "length")]
    Length,

    #[serde(rename = "stop")]
    Stop,

    #[serde(rename = "error")]
    Error(String),

    #[serde(rename = "cancelled")]
    Cancelled,
}

impl std::fmt::Display for FinishReason {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            FinishReason::EoS => write!(f, "eos"),
            FinishReason::Length => write!(f, "length"),
            FinishReason::Stop => write!(f, "stop"),
            FinishReason::Error(msg) => write!(f, "error: {}", msg),
            FinishReason::Cancelled => write!(f, "cancelled"),
        }
    }
}

impl std::str::FromStr for FinishReason {
    type Err = anyhow::Error;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "eos" => Ok(FinishReason::EoS),
            "length" => Ok(FinishReason::Length),
            "stop" => Ok(FinishReason::Stop),
            "cancelled" => Ok(FinishReason::Cancelled),
            s if s.starts_with("error: ") => Ok(FinishReason::Error(s[7..].to_string())),
            _ => Err(anyhow::anyhow!("Invalid FinishReason variant: '{}'", s)),
        }
    }
}

/// LLM Inference Engines can accept a variety of input types. Not all Engines will support all
/// input types. For example, the trtllm::AsyncEngine only supports `PromptType::Tokens` as an
/// input type. The higher-level `Backend` class is a general wrapper around Engines that will
/// enable many of the input options that require pre/postprocessing.
#[derive(Serialize, Deserialize, Debug, Clone, Eq, PartialEq)]
pub enum PromptType {
    /// If allowed, this input type allowed the caller to pass a list of token_ids directly to the
    /// inference engine. This is an advanced feature that requires the caller to handle all of the
    /// necessary prompt formatting and tokenization.
    #[serde(rename = "token_ids")]
    TokenIds(Vec<TokenIdType>),

    /// If allowed, the raw text will be tokenized and converted to token_ids without any additional
    /// preprocessing. This is an advanced features that requires the caller to correctly format the
    /// prompt as defined by the model.
    #[serde(rename = "raw")]
    Raw(String),

    /// If allowed, the `CompletionContext` will be preprocessed server-side. If the `Model` trait
    /// `requires_prompt_template` returns true then the `CompletionContext` will be used to
    /// to render the formatted prompt from the template. `Completion` is the preferred `PromptType`
    /// for single turn completions.
    #[serde(rename = "completion")]
    Completion(CompletionContext),

    /// If allowed, the `ChatContext` will be preprocessed server-side. Most chat models will have
    /// a predefined prompt format/structure. If the `Model` trait `requires_prompt_template` returns
    /// true then the `ChatContext` will be used to to render the formatted prompt from the template.
    /// `ChatCompletion` is the preferred `PromptType` for multi-turn completions.
    #[serde(rename = "chat_completion")]
    ChatCompletion(ChatContext),

    /// If allowed, then `Model::requires_prompt_template()` must also return true. The `serde_json::Value`
    /// will be passed directly the prompt template. This allows for a complete generic data model and
    /// prompt template to be passed to be defined and used by the server.
    #[serde(rename = "custom_json")]
    CustomJson(serde_json::Value),
}

/// TensorRT LLM does not perform preprocessing or postprocessing. The input_ids / token_ids
/// are expected to be preprocessed by the client. The client is responsible for constructing
/// the model specific prompt template and applying the tokenizer.
///
/// TensorRT LLM will perform some server side postprocessing to ensure that generation is
/// efficiently stopped. See `StopConditions` below.
#[derive(Serialize, Deserialize, Debug, Clone, Builder)]
pub struct CompletionRequest {
    /// Type of prompt
    pub prompt: PromptType,

    /// StopConditions are conditions that the inference engine will use to stop generation.
    pub stop_conditions: StopConditions,

    /// SamplingOptions directs the inference engine to use sampling instead of greedy decoding.
    /// More documentation on how and on the order in which sampling options are applied
    /// are needed.
    pub sampling_options: SamplingOptions,

    /// The computed checksum of the Model Deployment Card (MDC).
    #[builder(default)]
    pub mdc_sum: Option<String>,

    /// User requested annotations for the request
    #[builder(default)]
    pub annotations: Option<Vec<String>>,
}

impl CompletionRequest {
    pub fn builder() -> CompletionRequestBuilder {
        CompletionRequestBuilder::default()
    }
}

#[derive(Serialize, Deserialize, Debug, Clone, Eq, PartialEq)]
/// Defines the prompt template and system prompt for a completion request.
/// If the model does not support prompt templates, the system_prompt will be ignored.
pub struct CompletionContext {
    /// Prompt sent by the user
    pub prompt: String,

    /// Optional system_prompt for models that support prompt templates with system_prompts.
    pub system_prompt: Option<String>,
}

/// ChatTurn is a struct that contains the user and assistant messages in a chat.
#[derive(Serialize, Deserialize, Debug, Clone, Eq, PartialEq)]
pub struct ChatTurn {
    /// The user message
    pub user: String,

    /// The assistant response
    pub assistant: String,
}

/// ChatContext is a struct that contains the role and context of a chat message
/// along with a flattened CompletionContext.
#[derive(Serialize, Deserialize, Debug, Clone, Eq, PartialEq)]
pub struct ChatContext {
    /// CompletionContext for this chat turn
    #[serde(flatten)]
    pub completion: CompletionContext,

    /// The history/context of the user and assistant messages in the chat context
    pub context: Vec<ChatTurn>,
}

/// TensorRT LLM server-side stop conditions. These options allow for the server to evaluate
/// the generated sequence and stop generation if the sequence meets a stop condition.
#[derive(Serialize, Deserialize, Debug, Clone, Default)]
pub struct StopConditions {
    /// The maximum number of tokens to generate
    pub max_tokens: Option<u32>,

    /// List of strings that stop the generation when they are generated.
    /// The returned output will not contain the stop strings.
    pub stop: Option<Vec<String>>,

    /// List of tokens that stop the generation when they are
    /// generated. The returned output will NOT contain the stop tokens.
    pub stop_token_ids_hidden: Option<Vec<TokenIdType>>,

    /// The minimum number of tokens to generate
    /// To ignore_eos, set min_tokens to max_tokens
    pub min_tokens: Option<u32>,

    /// Whether to ignore the EOS token and continue generating
    /// tokens after the EOS token is generated.
    // TODO(ignore_eos) - improve this my masking the EOS token with logit bias
    pub ignore_eos: Option<bool>,
}

impl StopConditions {
    pub fn apply_ignore_eos(&mut self) {
        if self.ignore_eos.unwrap_or(false) {
            self.min_tokens = self.max_tokens;
            self.stop = None;
            self.stop_token_ids_hidden = None;
        }
    }
}

/// Temperature range for sampling.
pub const TEMPERATURE_RANGE: (f32, f32) = (0.0, 1.0);

/// Top P range for sampling.
pub const TOP_P_RANGE: (f32, f32) = (0.0, 1.0);

/// Frequency Penalty range for sampling.
pub const FREQUENCY_PENALTY_RANGE: (f32, f32) = (-1.0, 1.0);

/// Collection of options that control the sampling behavior of the inference engine.
#[derive(Serialize, Deserialize, Debug, Clone, Default)]
pub struct SamplingOptions {
    /// Number of output sequences to return for the given prompt
    pub n: Option<i32>,

    /// Number of output sequences that are generated from the prompt.
    /// From these `best_of` sequences, the top `n` sequences are returned.
    /// `best_of` must be greater than or equal to `n`. This is treated as
    /// the beam width when `use_beam_search` is True. By default, `best_of`
    /// is set to `n`.
    pub best_of: Option<i32>,

    /// Float that penalizes new tokens based on whether they
    /// appear in the generated text so far. Values > 0 encourage the model
    /// to use new tokens, while values < 0 encourage the model to repeat
    /// tokens.
    pub presence_penalty: Option<f32>,

    /// Float that penalizes new tokens based on their
    /// frequency in the generated text so far. Values > 0 encourage the
    /// model to use new tokens, while values < 0 encourage the model to
    /// repeat tokens.
    pub frequency_penalty: Option<f32>,

    /// Float that penalizes new tokens based on whether
    /// they appear in the prompt and the generated text so far. Values > 1
    /// encourage the model to use new tokens, while values < 1 encourage
    /// the model to repeat tokens.
    pub repetition_penalty: Option<f32>,

    /// Float that controls the randomness of the sampling. Lower
    /// values make the model more deterministic, while higher values make
    /// the model more random. Zero means greedy sampling.
    pub temperature: Option<f32>,

    /// Float that controls the cumulative probability of the top tokens
    /// to consider. Must be in (0, 1]. Set to 1 to consider all tokens.
    pub top_p: Option<f32>,

    /// Integer that controls the number of top tokens to consider. Set
    /// to -1 to consider all tokens.
    pub top_k: Option<i32>,

    /// Float that represents the minimum probability for a token to be
    /// considered, relative to the probability of the most likely token.
    /// Must be in [0, 1]. Set to 0 to disable this.
    pub min_p: Option<f32>,

    /// Whether to use beam search instead of sampling.
    pub use_beam_search: Option<bool>,

    /// Float that penalizes sequences based on their length.
    /// Used in beam search.
    pub length_penalty: Option<f32>,

    /// The seed to use when sampling
    pub seed: Option<i64>,
}

impl SamplingOptions {
    pub fn force_greedy(&mut self) {
        self.presence_penalty = None;
        self.frequency_penalty = None;
        self.repetition_penalty = None;
        self.temperature = None;
        self.top_p = None;
        self.top_k = None;
        self.min_p = None;
    }
}

/// Collection of options that control what information the inference engine returns in the response.
#[derive(Serialize, Deserialize, Debug, Clone, Default)]
pub struct OutputOptions {
    /// Number of log probabilities to return per output token.
    /// Note that the implementation follows the OpenAI API: The return
    /// result includes the log probabilities on the `logprobs` most likely
    /// tokens, as well the chosen tokens. The API will always return the
    /// log probability of the sampled token, so there  may be up to
    /// `logprobs+1` elements in the response
    pub logprobs: Option<u32>,

    /// Number of log probabilities to return per prompt token.
    pub prompt_logprobs: Option<u32>,

    /// Whether to skip special tokens in the output.
    /// spaces_between_special_tokens: Whether to add spaces between special
    /// tokens in the output.  Defaults to True.
    pub skip_special_tokens: Option<bool>,

    /// If true, the Context object will contain the prompt that was pass to
    /// the tokenizer. This is useful for inspecting the behavior of prompt
    /// templates that are applied during the backend preprocessing.
    pub formatted_prompt: Option<bool>,
}

// Struct for log probability information
#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct ChatCompletionLogprobs {
    /// A list of message content tokens with log probability information.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub content: Option<Vec<ChatCompletionTokenLogprob>>,

    /// A list of message refusal tokens with log probability information.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub refusal: Option<Vec<ChatCompletionTokenLogprob>>,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct ChatCompletionTokenLogprob {
    /// The token.
    pub token: String,

    /// The log probability of this token, if it is within the top 20 most likely tokens.
    /// Otherwise, the value `-9999.0` signifies that the token is very unlikely.
    pub logprob: f64,

    /// A list of integers representing the UTF-8 bytes representation of the token.
    /// Useful in instances where characters are represented by multiple tokens and their
    /// byte representations must be combined to generate the correct text representation.
    /// Can be `None` if there is no bytes representation for the token.
    pub bytes: Option<Vec<u8>>,

    /// List of the most likely tokens and their log probability, at this token position.
    /// In rare cases, there may be fewer than the requested number of `top_logprobs` returned.
    pub top_logprobs: Vec<TopLogprob>,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct TopLogprob {
    /// The token.
    pub token: String,

    /// The log probability of this token.
    pub logprob: f64,

    /// A list of integers representing the UTF-8 bytes representation of the token.
    /// Can be `None` if there is no bytes representation for the token.
    pub bytes: Option<Vec<u8>>,
}

// /// UserData is a struct that contains user-defined data that can be passed to the inference engine.
// /// This information will be use to annotate the distributed traces for improved observability.
// #[derive(Serialize, Deserialize, Debug, Clone, Default)]
// pub struct UserData {
//     /// Apply server-side prompt template to the request
//     pub request_uuid: Option<uuid::Uuid>,
// }

/// StreamingResponse is the primary response object for the LLM Engine. The response stream
/// can emit three different types of messages. The Initialize and Finalize messages are optional
/// and primarily used over disaggreated transports to move states from the server to the client.
#[derive(Serialize, Deserialize, Debug)]
pub enum StreamingResponse {
    /// Initialize transports a Prologue object which communication the LLM Engine Context
    Initialize(Option<Prologue>),

    /// Step is the primary data in the response stream. It contains the StreamingCompletionResponse
    Step(Box<StreamingCompletionResponse>),

    /// Finalize is an optional final message in the response stream. It contains the Epilogue object which
    /// is used to communicate extra information about the completion and the engine statistics.
    Finalize(Option<Epilogue>),
}

// TODO(ryan) - this should be part of the internal api as it is not deserializble
//              the public API should drop the Option<Arc<Stats>> in favor of Option<Stats>
//              the two variants both serialize to the same json; however, the internal version
//              can not be deserialized directly.
//              we use the internal one on the server side to avoid the cost of cloning the Stats
//              object; however, client side, we should always fully materialize the Stats object.
//
// TODO(ryan) - update this object to use an enum where we have the current definition be the
//              StepResponse arm; then we will add the following arms:
//              - Initialize(Prologue)
//              - Step()
//              - Finalize(Epilogue)

/// This is the first message that will be emitted by an Engine Response Stream
/// It indicates that the request has been preprocessed and queued for execution on the backend.
#[derive(Serialize, Deserialize, Debug)]
pub struct Prologue {
    /// If the request was preprocessed with a prompt template, this will contain the formatted prompt
    pub formatted_prompt: Option<String>,

    /// If the request did not contain TokenIds, this will contain the token_ids that were generated
    /// from tokenizing the prompt.
    pub input_token_ids: Option<Vec<TokenIdType>>,
}

/// This is the final message that will be emitted by a Engine Response Stream when it
/// finishes without error. In some cases, the engine may emit an error which will indicate
/// the end of the steam. Another case in which an Finalize(Epilogue) will not be emitted is
/// if the response handler has stalled and too many responses
#[derive(Serialize, Deserialize, Debug)]
pub struct Epilogue {}

#[derive(Debug)]
pub struct StreamingCompletionResponse {
    pub delta: Delta,
    pub logprobs: Option<ChatCompletionLogprobs>,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub enum StreamState {
    Active,
    Finished(FinishReason),
}

#[derive(Serialize, Deserialize, Debug, Clone)]
#[serde(rename_all = "snake_case")]
pub enum Logits {
    All(Vec<f32>),
    Sparse(Vec<(u32, f32)>),
}

#[derive(Serialize, Deserialize, Debug, Clone)]
#[serde(rename_all = "snake_case")]
pub enum LogProbs {
    Normalized(Logits),
    Raw(Logits),
}

/// At each SequencePosition we hold position specific data
pub struct SequencePositionData {
    pub token_id: TokenIdType,

    /// The log probability of the token
    pub logprobs: Option<LogProbs>,
}

// todo(ryan) - we need to create a DeltaBuilder which is a mutable object that can be passed
// around from the low-level compute engine to the high-level api. The DeltaBuilder will allow
// us to construct the Delta object at multiple layers in the streaming response path.
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub struct Delta {
    pub is_complete: bool,

    pub finish_reason: Option<FinishReason>,

    // new token_ids
    pub token_ids: Option<Vec<u32>>,

    // tokens
    pub tokens: Option<Vec<String>>,

    // decoded text
    pub text: Option<String>,

    // current sequence length
    // when stream, we expect this to increase by 1 on each response
    pub sequence_length: Option<usize>,

    // if the number of slots for a given request is greater than 1
    // this indicates the index of the slot for the response
    pub index: Option<usize>,

    /// cumulative log probabilities
    pub cum_log_probs: Option<f64>,

    /// error message from engine
    /// if this is set, is_complete should also be true
    pub err_msg: Option<String>,

    /// usage info
    pub usage: Option<Usage>,
}

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub struct Usage {
    pub input_tokens_count: usize,
    pub output_tokens_count: usize,
}

// todo(ryan) - we need to update this object to make it more generic
// we need to define a set of generic stats traits that allow those stats to be None
// then back them by a concrete implementation like a TrtllmStats object
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub struct Stats {
    /// Time since the last Epoch/Forward Pass in microseconds (us).
    /// This is measured and recorded by the Response Router rather then the
    /// Inference Engine. Note, when evaluating the responses, if the this
    /// values is greater then the stream's measured value, then there was a gap
    /// between forward passes. In normal operation, the value of this field should
    /// be less than the recorded value on the response stream.
    pub time_since_last_forward_pass_us: Option<u64>,

    pub request_active_count: u32,

    pub request_context_count: u32,

    pub request_generation_count: u32,

    pub request_scheduled_count: u32,

    pub request_max_count: u32,

    pub kv_free_cache_blocks: u64,

    pub kv_max_cache_blocks: u64,

    pub kv_used_cache_blocks: u64,

    pub kv_tokens_per_cache_block: u64,

    pub runtime_cpu_memory_usage: u64,

    pub runtime_gpu_memory_usage: u64,

    pub runtime_pinned_memory_usage: u64,

    pub iteration_counter: u64,

    pub microbatch_id: u64,

    pub total_context_tokens: u32,

    pub timestamp: String,
}

impl Serialize for StreamingCompletionResponse {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        let mut state = serializer.serialize_struct("StreamingCompletionResponse", 2)?;

        // Serialize `delta` field
        state.serialize_field("delta", &self.delta)?;

        state.end()
    }
}

impl<'de> Deserialize<'de> for StreamingCompletionResponse {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        // Create a temporary struct for deserialization
        #[derive(Deserialize)]
        struct TempResponse {
            delta: Delta,
            logprobs: Option<ChatCompletionLogprobs>,
        }

        let TempResponse { delta, logprobs } = TempResponse::deserialize(deserializer)?;

        Ok(StreamingCompletionResponse { delta, logprobs })
    }
}

#[derive(Serialize, Deserialize, Debug)]
pub struct ScatterData<T> {
    pub x: Vec<T>,
    pub y: Vec<T>,
}

#[derive(Serialize, Deserialize, Debug)]
pub struct Trace {
    pub time_to_first_token: u64,
    pub token_to_token: Vec<u64>,
    pub start: SystemTime,
    pub complete: SystemTime,
    pub initial_tokens: u32,
    pub max_tokens: u32,
    pub t2ft_iteration_count: u64,
    pub t2t_iteration_count: Vec<u64>,
}

#[derive(Serialize, Deserialize, Debug)]
pub struct PerformanceModel {
    // linear regression parameters fitting t2ft vs. initial tokens
    pub t2ft_intercept: f64,
    pub t2ft_slope: f64,

    // linear regression parameters fitting t2tl vs. initial tokens
    pub t2tl_intercept: f64,
    pub t2tl_slope: f64,

    // r2 values from the regression
    pub t2ft_fit_r2: f64,
    pub t2tl_fit_r2: f64,
}

#[derive(Serialize, Deserialize, Debug)]
pub struct CalibrationResults {
    pub effective_flops: f64,
    pub effective_memory_bandwidth: f64,
    pub max_q: u32,
    pub performance_model: PerformanceModel,
    pub traces: Vec<Trace>,
    pub t2ft_scatter_data: ScatterData<f64>,
    pub t2tl_scatter_data: ScatterData<f64>,
}

#[derive(Serialize, Deserialize, Debug)]
pub struct LoadgenResults {
    pub stats_by_iteration: HashMap<u64, Stats>,
    pub traces: Vec<Trace>,
}

impl CompletionContext {
    /// Create a new CompletionContext
    pub fn new(prompt: String, system_prompt: Option<String>) -> Self {
        Self {
            prompt,
            system_prompt,
        }
    }

    /// Create a new CompletionContext with only a prompt
    pub fn from_prompt(prompt: String) -> Self {
        Self {
            prompt,
            system_prompt: None,
        }
    }

    /// Create a new CompletionContext with a prompt and system prompt
    pub fn with_system_prompt(prompt: String, system_prompt: String) -> Self {
        Self {
            prompt,
            system_prompt: Some(system_prompt),
        }
    }
}

// todo(ryan) - create a builder for chat context
impl From<CompletionContext> for PromptType {
    fn from(context: CompletionContext) -> Self {
        PromptType::Completion(context)
    }
}

#[cfg(test)]
mod tests {
    use serde_json;

    use super::*;

    #[test]
    fn test_completion_context_new() {
        let prompt = "Hello, world!".to_string();
        let system_prompt = Some("This is a system prompt.".to_string());
        let context = CompletionContext::new(prompt.clone(), system_prompt.clone());

        assert_eq!(context.prompt, prompt);
        assert_eq!(context.system_prompt, system_prompt);
    }

    #[test]
    fn test_completion_context_from_prompt() {
        let prompt = "Hello, world!".to_string();
        let context = CompletionContext::from_prompt(prompt.clone());

        assert_eq!(context.prompt, prompt);
        assert_eq!(context.system_prompt, None);
    }

    #[test]
    fn test_completion_context_with_system_prompt() {
        let prompt = "Hello, world!".to_string();
        let system_prompt = "This is a system prompt.".to_string();
        let context = CompletionContext::with_system_prompt(prompt.clone(), system_prompt.clone());

        assert_eq!(context.prompt, prompt);
        assert_eq!(context.system_prompt, Some(system_prompt));
    }

    #[test]
    fn test_completion_context_into_prompt_type() {
        let prompt = "Hello, world!".to_string();
        let system_prompt = "This is a system prompt.".to_string();
        let context = CompletionContext::with_system_prompt(prompt.clone(), system_prompt.clone());
        let prompt_type: PromptType = context.into();

        if let PromptType::Completion(completion_context) = prompt_type {
            assert_eq!(completion_context.prompt, prompt);
            assert_eq!(completion_context.system_prompt, Some(system_prompt));
        } else {
            panic!("Expected a Completion variant");
        }
    }

    // #[test]
    // fn test_serialize_with_stats() {
    //     let response = StreamingCompletionResponse {
    //         delta: Delta {
    //             is_complete: true,
    //             finish_reason: Some(FinishReason::Length),
    //             token_ids: Some(vec![101, 102, 103]),
    //             tokens: Some(vec!["token1".to_string(), "token2".to_string()]),
    //             text: Some("example text".to_string()),
    //             sequence_length: Some(3),
    //             index: Some(0),
    //             cum_log_probs: Some(-0.5),
    //             err_msg: None,
    //             usage: None,
    //         },
    //         logprobs: None,
    //     };

    //     // Serialize the response
    //     let serialized = serde_json::to_string(&response).expect("Failed to serialize");

    //     // Expected JSON string (simplified)
    //     let expected = r#"{
    //         "delta": {
    //             "is_complete": true,
    //             "finish_reason": "length",
    //             "token_ids": [101, 102, 103],
    //             "tokens": ["token1", "token2"],
    //             "text": "example text",
    //             "sequence_length": 3,
    //             "index": 0,
    //             "cum_log_probs": -0.5,
    //             "err_msg": null,
    //             "usage": null
    //         },
    //         "stats": {
    //             "time_since_last_forward_pass_us": 1000,
    //             "request_active_count": 2,
    //             "request_context_count": 1,
    //             "request_generation_count": 3,
    //             "request_scheduled_count": 1,
    //             "request_max_count": 10,
    //             "kv_free_cache_blocks": 500,
    //             "kv_max_cache_blocks": 1000,
    //             "kv_used_cache_blocks": 500,
    //             "kv_tokens_per_cache_block": 10,
    //             "runtime_cpu_memory_usage": 5000,
    //             "runtime_gpu_memory_usage": 2000,
    //             "runtime_pinned_memory_usage": 1000,
    //             "iteration_counter": 5,
    //             "microbatch_id": 12345,
    //             "total_context_tokens": 256,
    //             "timestamp": "2024-01-01T00:00:00Z"
    //         }
    //     }"#;

    //     assert_eq!(
    //         serde_json::from_str::<serde_json::Value>(&serialized).unwrap(),
    //         serde_json::from_str::<serde_json::Value>(expected).unwrap()
    //     );
    // }

    #[test]
    fn test_serialize_without_stats() {
        let response = StreamingCompletionResponse {
            delta: Delta {
                is_complete: false,
                finish_reason: None,
                token_ids: None,
                tokens: None,
                text: None,
                sequence_length: None,
                index: None,
                cum_log_probs: None,
                err_msg: None,
                usage: None,
            },
            logprobs: None,
        };

        // Serialize the response
        let serialized = serde_json::to_string(&response).expect("Failed to serialize");

        // Expected JSON string
        let expected = r#"{
            "delta": {
                "is_complete": false,
                "finish_reason": null,
                "token_ids": null,
                "tokens": null,
                "text": null,
                "sequence_length": null,
                "index": null,
                "cum_log_probs": null,
                "err_msg": null,
                "usage": null
            }
        }"#;

        assert_eq!(
            serde_json::from_str::<serde_json::Value>(&serialized).unwrap(),
            serde_json::from_str::<serde_json::Value>(expected).unwrap()
        );
    }

    // #[test]
    // fn test_deserialize_with_stats() {
    //     let json_data = r#"{
    //         "delta": {
    //             "is_complete": true,
    //             "finish_reason": "length",
    //             "token_ids": [101, 102, 103],
    //             "tokens": ["token1", "token2"],
    //             "text": "example text",
    //             "sequence_length": 3,
    //             "index": 0,
    //             "cum_log_probs": -0.5,
    //             "err_msg": null,
    //             "usage": null
    //         },
    //         "stats": {
    //             "time_since_last_forward_pass_us": 1000,
    //             "request_active_count": 2,
    //             "request_context_count": 1,
    //             "request_generation_count": 3,
    //             "request_scheduled_count": 1,
    //             "request_max_count": 10,
    //             "kv_free_cache_blocks": 500,
    //             "kv_max_cache_blocks": 1000,
    //             "kv_used_cache_blocks": 500,
    //             "kv_tokens_per_cache_block": 10,
    //             "runtime_cpu_memory_usage": 5000,
    //             "runtime_gpu_memory_usage": 2000,
    //             "runtime_pinned_memory_usage": 1000,
    //             "iteration_counter": 5,
    //             "microbatch_id": 12345,
    //             "total_context_tokens": 256,
    //             "timestamp": "2024-01-01T00:00:00Z"
    //         }
    //     }"#;

    //     // Deserialize the JSON string
    //     let deserialized: StreamingCompletionResponse =
    //         serde_json::from_str(json_data).expect("Failed to deserialize");

    //     // Expected response object
    //     let expected = StreamingCompletionResponse {
    //         delta: Delta {
    //             is_complete: true,
    //             finish_reason: Some(FinishReason::Length),
    //             token_ids: Some(vec![101, 102, 103]),
    //             tokens: Some(vec!["token1".to_string(), "token2".to_string()]),
    //             text: Some("example text".to_string()),
    //             sequence_length: Some(3),
    //             index: Some(0),
    //             cum_log_probs: Some(-0.5),
    //             err_msg: None,
    //             usage: None,
    //         },
    //         logprobs: None,
    //     };

    //     // This is wieldy but we can no longer do assert_eq!(deserialized, expected);
    //     // because the struct no longer has the PartialEq trait
    //     assert_eq!(deserialized.delta.is_complete, expected.delta.is_complete);
    //     assert_eq!(
    //         deserialized.delta.finish_reason,
    //         expected.delta.finish_reason
    //     );
    //     assert_eq!(deserialized.delta.token_ids, expected.delta.token_ids);
    //     assert_eq!(deserialized.delta.tokens, expected.delta.tokens);
    //     assert_eq!(deserialized.delta.text, expected.delta.text);
    //     assert_eq!(
    //         deserialized.delta.sequence_length,
    //         expected.delta.sequence_length
    //     );
    //     assert_eq!(deserialized.delta.index, expected.delta.index);
    //     assert_eq!(
    //         deserialized.delta.cum_log_probs,
    //         expected.delta.cum_log_probs
    //     );
    //     assert_eq!(deserialized.delta.err_msg, expected.delta.err_msg);
    //     assert_eq!(deserialized.delta.usage, expected.delta.usage);

    //     assert_eq!(
    //         deserialized_stats.time_since_last_forward_pass_us,
    //         expected_stats.time_since_last_forward_pass_us
    //     );
    //     assert_eq!(
    //         deserialized_stats.request_active_count,
    //         expected_stats.request_active_count
    //     );
    //     assert_eq!(
    //         deserialized_stats.request_context_count,
    //         expected_stats.request_context_count
    //     );
    //     assert_eq!(
    //         deserialized_stats.request_generation_count,
    //         expected_stats.request_generation_count
    //     );
    //     assert_eq!(
    //         deserialized_stats.request_scheduled_count,
    //         expected_stats.request_scheduled_count
    //     );
    //     assert_eq!(
    //         deserialized_stats.request_max_count,
    //         expected_stats.request_max_count
    //     );
    //     assert_eq!(
    //         deserialized_stats.kv_free_cache_blocks,
    //         expected_stats.kv_free_cache_blocks
    //     );
    //     assert_eq!(
    //         deserialized_stats.kv_max_cache_blocks,
    //         expected_stats.kv_max_cache_blocks
    //     );
    //     assert_eq!(
    //         deserialized_stats.kv_used_cache_blocks,
    //         expected_stats.kv_used_cache_blocks
    //     );
    //     assert_eq!(
    //         deserialized_stats.kv_tokens_per_cache_block,
    //         expected_stats.kv_tokens_per_cache_block
    //     );
    //     assert_eq!(
    //         deserialized_stats.runtime_cpu_memory_usage,
    //         expected_stats.runtime_cpu_memory_usage
    //     );
    //     assert_eq!(
    //         deserialized_stats.runtime_gpu_memory_usage,
    //         expected_stats.runtime_gpu_memory_usage
    //     );
    //     assert_eq!(
    //         deserialized_stats.runtime_pinned_memory_usage,
    //         expected_stats.runtime_pinned_memory_usage
    //     );
    //     assert_eq!(
    //         deserialized_stats.iteration_counter,
    //         expected_stats.iteration_counter
    //     );
    //     assert_eq!(
    //         deserialized_stats.microbatch_id,
    //         expected_stats.microbatch_id
    //     );
    //     assert_eq!(
    //         deserialized_stats.total_context_tokens,
    //         expected_stats.total_context_tokens
    //     );
    //     assert_eq!(deserialized_stats.timestamp, expected_stats.timestamp);
    // }

    #[test]
    fn test_deserialize_without_stats() {
        let json_data = r#"{
            "delta": {
                "is_complete": false,
                "finish_reason": null,
                "token_ids": null,
                "tokens": null,
                "text": null,
                "sequence_length": null,
                "index": null,
                "cum_log_probs": null,
                "err_msg": null,
                "usage": null
            }
        }"#;

        // Deserialize the JSON string
        let deserialized: StreamingCompletionResponse =
            serde_json::from_str(json_data).expect("Failed to deserialize");

        // Expected response object
        let expected = StreamingCompletionResponse {
            delta: Delta {
                is_complete: false,
                finish_reason: None,
                token_ids: None,
                tokens: None,
                text: None,
                sequence_length: None,
                index: None,
                cum_log_probs: None,
                err_msg: None,
                usage: None,
            },
            logprobs: None,
        };

        // This is wieldy but we can no longer do assert_eq!(deserialized, expected);
        // because the struct no longer has the PartialEq trait
        assert_eq!(deserialized.delta.is_complete, expected.delta.is_complete);
        assert_eq!(
            deserialized.delta.finish_reason,
            expected.delta.finish_reason
        );
        assert_eq!(deserialized.delta.token_ids, expected.delta.token_ids);
        assert_eq!(deserialized.delta.tokens, expected.delta.tokens);
        assert_eq!(deserialized.delta.text, expected.delta.text);
        assert_eq!(
            deserialized.delta.sequence_length,
            expected.delta.sequence_length
        );
        assert_eq!(deserialized.delta.index, expected.delta.index);
        assert_eq!(
            deserialized.delta.cum_log_probs,
            expected.delta.cum_log_probs
        );
        assert_eq!(deserialized.delta.err_msg, expected.delta.err_msg);
        assert_eq!(deserialized.delta.usage, expected.delta.usage);
    }

    #[test]
    fn test_serialize_delta_and_none_stats() {
        let response = StreamingCompletionResponse {
            delta: Delta {
                is_complete: true,
                finish_reason: Some(FinishReason::Length),
                token_ids: Some(vec![101, 102, 103]),
                tokens: Some(vec!["token1".to_string(), "token2".to_string()]),
                text: Some("example text".to_string()),
                sequence_length: Some(3),
                index: Some(0),
                cum_log_probs: Some(-0.5),
                err_msg: None,
                usage: None,
            },
            logprobs: None,
        };

        // Serialize the response
        let serialized = serde_json::to_string(&response).expect("Failed to serialize");

        // Expected JSON string where stats is null
        let expected_json = r#"{
            "delta": {
                "is_complete": true,
                "finish_reason": "length",
                "token_ids": [101, 102, 103],
                "tokens": ["token1", "token2"],
                "text": "example text",
                "sequence_length": 3,
                "index": 0,
                "cum_log_probs": -0.5,
                "err_msg": null,
                "usage": null
            }
        }"#;

        // Parse both the serialized response and the expected JSON as serde_json::Value for easy comparison
        assert_eq!(
            serde_json::from_str::<serde_json::Value>(&serialized).unwrap(),
            serde_json::from_str::<serde_json::Value>(expected_json).unwrap()
        );
    }
}
