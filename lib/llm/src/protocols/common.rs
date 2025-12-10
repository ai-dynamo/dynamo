// SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

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

use anyhow::Result;
use derive_builder::Builder;
use serde::{Deserialize, Serialize};

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

pub trait OutputOptionsProvider {
    fn extract_output_options(&self) -> Result<OutputOptions>;
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

    #[serde(rename = "content_filter")]
    ContentFilter,
}

impl std::fmt::Display for FinishReason {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            FinishReason::EoS => write!(f, "eos"),
            FinishReason::Length => write!(f, "length"),
            FinishReason::Stop => write!(f, "stop"),
            FinishReason::Error(msg) => write!(f, "error: {}", msg),
            FinishReason::Cancelled => write!(f, "cancelled"),
            FinishReason::ContentFilter => write!(f, "content_filter"),
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

impl From<FinishReason> for dynamo_async_openai::types::CompletionFinishReason {
    fn from(reason: FinishReason) -> Self {
        match reason {
            FinishReason::EoS | FinishReason::Stop | FinishReason::Cancelled => {
                dynamo_async_openai::types::CompletionFinishReason::Stop
            }
            FinishReason::ContentFilter => {
                dynamo_async_openai::types::CompletionFinishReason::ContentFilter
            }
            FinishReason::Length => dynamo_async_openai::types::CompletionFinishReason::Length,
            FinishReason::Error(_) => dynamo_async_openai::types::CompletionFinishReason::Stop,
        }
    }
}

impl From<dynamo_async_openai::types::CompletionFinishReason> for FinishReason {
    fn from(reason: dynamo_async_openai::types::CompletionFinishReason) -> Self {
        match reason {
            dynamo_async_openai::types::CompletionFinishReason::Stop => FinishReason::Stop,
            dynamo_async_openai::types::CompletionFinishReason::Length => FinishReason::Length,
            dynamo_async_openai::types::CompletionFinishReason::ContentFilter => {
                FinishReason::ContentFilter
            }
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

    #[builder(default)]
    pub output_options: OutputOptions,

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

    /// Maximum number of thinking tokens allowed
    /// NOTE: Currently a passthrough - no enforcement logic implemented
    pub max_thinking_tokens: Option<u32>,
}

impl StopConditions {
    pub fn apply_ignore_eos(&mut self) {
        if self.ignore_eos.unwrap_or(false) {
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
    pub n: Option<u8>,

    /// Number of output sequences that are generated from the prompt.
    /// From these `best_of` sequences, the top `n` sequences are returned.
    /// `best_of` must be greater than or equal to `n`. This is treated as
    /// the beam width when `use_beam_search` is True. By default, `best_of`
    /// is set to `n`.
    pub best_of: Option<u8>,

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

    /// Whether to include the stop string in the output.
    pub include_stop_str_in_output: Option<bool>,

    /// Guided Decoding Options
    pub guided_decoding: Option<GuidedDecodingOptions>,
}

/// Guided Decoding Options
///
/// Only one of `json`, `regex`, `choice`, or `grammar` should be set.
#[derive(Serialize, Deserialize, Debug, Clone, Default)]
pub struct GuidedDecodingOptions {
    /// If specified, the output will follow the JSON schema. Can be a string, an object, or null.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub json: Option<serde_json::Value>,

    /// If specified, the output will follow the regex pattern. Can be a string or null.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub regex: Option<String>,

    /// If specified, the output will be exactly one of the choices.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub choice: Option<Vec<String>>,

    /// If specified, the output will follow the context-free grammar. Can be a string or null.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub grammar: Option<String>,

    /// If specified, the backend to use for guided decoding, can be backends like xgrammar or custom guided decoding backend
    #[serde(skip_serializing_if = "Option::is_none")]
    pub backend: Option<String>,

    /// If specified, whitespace pattern to use for guided decoding. Can be a string or null.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub whitespace_pattern: Option<String>,
}

impl GuidedDecodingOptions {
    /// Construct without validation
    pub fn new(
        json: Option<serde_json::Value>,
        regex: Option<String>,
        choice: Option<Vec<String>>,
        grammar: Option<String>,
        backend: Option<String>,
        whitespace_pattern: Option<String>,
    ) -> Self {
        Self {
            json,
            regex,
            choice,
            grammar,
            backend,
            whitespace_pattern,
        }
    }

    /// Construct and validate (fallible)
    pub fn validated(
        json: Option<serde_json::Value>,
        regex: Option<String>,
        choice: Option<Vec<String>>,
        grammar: Option<String>,
        backend: Option<String>,
        whitespace_pattern: Option<String>,
    ) -> Result<Self> {
        let instance = Self::new(json, regex, choice, grammar, backend, whitespace_pattern);
        instance.validate()?;
        Ok(instance)
    }

    /// Construct only if one field is Some (fallible)
    pub fn from_optional(
        json: Option<serde_json::Value>,
        regex: Option<String>,
        choice: Option<Vec<String>>,
        grammar: Option<String>,
        backend: Option<String>,
        whitespace_pattern: Option<String>,
    ) -> Result<Option<Self>> {
        let is_empty_choice = choice.as_ref().is_none_or(|v| v.is_empty());
        if json.is_none()
            && regex.is_none()
            && is_empty_choice
            && grammar.is_none()
            && whitespace_pattern.is_none()
        {
            return Ok(None);
        }
        let instance = Self::validated(json, regex, choice, grammar, backend, whitespace_pattern)?;
        Ok(Some(instance))
    }

    /// Validate that only one guided decoding option is set
    pub fn validate(&self) -> Result<()> {
        let count = [
            self.json.is_some(),
            self.regex.is_some(),
            self.choice.as_ref().is_some_and(|v| !v.is_empty()),
            self.grammar.is_some(),
            self.whitespace_pattern.is_some(),
        ]
        .iter()
        .filter(|&&v| v)
        .count();

        if count > 1 {
            Err(anyhow::anyhow!(
                "Only one of json, regex, choice, or grammar can be set, but multiple are specified: {:?}",
                self
            ))
        } else {
            Ok(())
        }
    }
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

#[derive(Debug)]
pub struct StreamingCompletionResponse {
    pub delta: Delta,
    pub logprobs: Option<ChatCompletionLogprobs>,
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

    #[test]
    fn test_guided_decoding_options_new_and_exclusive() {
        // Only JSON set
        let json_val = serde_json::json!({"type": "object"});
        let backend = Some("xgrammar".to_string());
        let opts = GuidedDecodingOptions::validated(
            Some(json_val.clone()),
            None,
            None,
            None,
            backend.clone(),
            None,
        );
        assert!(opts.is_ok());
        let opts = opts.unwrap();
        assert_eq!(opts.json, Some(json_val));
        assert!(opts.regex.is_none());
        assert!(opts.choice.is_none());
        assert!(opts.grammar.is_none());
        assert_eq!(opts.backend, backend);
        assert!(opts.whitespace_pattern.is_none());

        // Only regex set
        let regex = Some(r"\d+".to_string());
        let opts = GuidedDecodingOptions::validated(None, regex.clone(), None, None, None, None);
        assert!(opts.is_ok());
        let opts = opts.unwrap();
        assert_eq!(opts.regex, regex);
        assert!(opts.json.is_none());
        assert!(opts.choice.is_none());
        assert!(opts.grammar.is_none());
        assert!(opts.whitespace_pattern.is_none());

        // Only choice set
        let choice = Some(vec!["A".to_string(), "B".to_string()]);
        let opts = GuidedDecodingOptions::validated(None, None, choice.clone(), None, None, None);
        assert!(opts.is_ok());
        let opts = opts.unwrap();
        assert_eq!(opts.choice, choice);
        assert!(opts.json.is_none());
        assert!(opts.regex.is_none());
        assert!(opts.grammar.is_none());
        assert!(opts.whitespace_pattern.is_none());

        // Only grammar set
        let grammar = Some("root ::= 'yes' | 'no'".to_string());
        let opts = GuidedDecodingOptions::validated(None, None, None, grammar.clone(), None, None);
        assert!(opts.is_ok());
        let opts = opts.unwrap();
        assert_eq!(opts.grammar, grammar);
        assert!(opts.json.is_none());
        assert!(opts.regex.is_none());
        assert!(opts.choice.is_none());
        assert!(opts.whitespace_pattern.is_none());

        // Only whitespace_pattern set
        let whitespace_pattern = Some(r"\s+".to_string());
        let opts = GuidedDecodingOptions::validated(
            None,
            None,
            None,
            None,
            None,
            whitespace_pattern.clone(),
        );
        assert!(opts.is_ok());
        let opts = opts.unwrap();
        assert_eq!(opts.whitespace_pattern, whitespace_pattern);
        assert!(opts.json.is_none());
        assert!(opts.regex.is_none());
        assert!(opts.choice.is_none());
        assert!(opts.grammar.is_none());

        // Multiple fields set (should error)
        let opts = GuidedDecodingOptions::validated(
            Some(serde_json::json!({})),
            Some(r"\d+".to_string()),
            None,
            None,
            None,
            None,
        );
        assert!(opts.is_err());

        let opts = GuidedDecodingOptions::validated(
            None,
            Some(r"\d+".to_string()),
            Some(vec!["A".to_string()]),
            None,
            None,
            None,
        );
        assert!(opts.is_err());

        let opts = GuidedDecodingOptions::validated(
            Some(serde_json::json!({})),
            None,
            Some(vec!["A".to_string()]),
            Some("root ::= 'yes'".to_string()),
            None,
            None,
        );
        assert!(opts.is_err());

        // All fields None (should be ok, but not useful)
        let opts = GuidedDecodingOptions::validated(None, None, None, None, None, None);
        assert!(opts.is_ok());
    }

    #[test]
    fn test_guided_decoding_options_from_optional() {
        // All None returns Ok(None)
        let opts = GuidedDecodingOptions::from_optional(None, None, None, None, None, None);
        assert!(opts.is_ok());
        assert!(opts.unwrap().is_none());

        // Only one set returns Ok(Some)
        let regex = Some(r"\w+".to_string());
        let opts =
            GuidedDecodingOptions::from_optional(None, regex.clone(), None, None, None, None);
        assert!(opts.is_ok());
        let val = opts.unwrap();
        assert!(val.is_some());
        let val = val.unwrap();
        assert_eq!(val.regex, regex);

        // Multiple set returns Err
        let opts = GuidedDecodingOptions::from_optional(
            Some(serde_json::json!({})),
            Some(r"\d+".to_string()),
            None,
            None,
            None,
            None,
        );
        assert!(opts.is_err());

        // Choice set but empty vector should not count as set
        let opts = GuidedDecodingOptions::from_optional(None, None, Some(vec![]), None, None, None);
        assert!(opts.is_ok());
        let val = opts.unwrap();
        assert!(val.is_none());

        // Choice set with non-empty vector
        let opts = GuidedDecodingOptions::from_optional(
            None,
            None,
            Some(vec!["A".to_string()]),
            None,
            None,
            None,
        );
        assert!(opts.is_ok());
        let val = opts.unwrap();
        assert!(val.is_some());
        let val = val.unwrap();
        assert_eq!(val.choice, Some(vec!["A".to_string()]));
    }
}
