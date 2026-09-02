// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! POC adapter from Switchyard's provider-neutral request IR into Dynamo preprocessing.
//!
//! This module deliberately stops at the frontend. Only [`PreprocessedRequest`](crate::preprocessor::PreprocessedRequest)
//! crosses the frontend/worker boundary.

use std::collections::HashMap;

use anyhow::{Context, Result, bail};
use dynamo_renderer::{OAIChatLikeRequest, TextInput};
use dynamo_runtime::protocols::annotated::AnnotationsProvider;
use minijinja::value::Value as TemplateValue;
use serde_json::{Map, Value, json};
use switchyard_protocol::llm::{ContentBlock, LlmRequest, Message, Role, ToolChoice, ToolResult};

use crate::preprocessor::prompt::MediaRequestExt;
use crate::protocols::common::{
    GuidedDecodingOptions, OutputOptions, OutputOptionsProvider, SamplingOptions,
    SamplingOptionsProvider, StopConditions, StopConditionsProvider,
    extensions::{NvExt, NvExtProvider},
};
use crate::protocols::openai::common_ext::{CommonExt, CommonExtProvider};
use crate::protocols::openai::validate::{
    BEST_OF_RANGE, FREQUENCY_PENALTY_RANGE, MIN_P_RANGE, N_RANGE, PRESENCE_PENALTY_RANGE,
    TEMPERATURE_RANGE, validate_range, validate_top_p,
};

/// Dynamo-only controls that do not belong in Switchyard's semantic prompt IR.
///
/// Common fields in [`LlmRequest`] supply defaults. A field set here overrides
/// the corresponding IR field.
#[derive(Debug, Clone, Default)]
pub struct DynamoExecutionOptions {
    pub stop_conditions: StopConditions,
    pub sampling_options: SamplingOptions,
    pub output_options: OutputOptions,
    pub nvext: Option<NvExt>,
    pub annotations: Vec<String>,
    pub common: CommonExt,
    pub chat_template_args: Option<HashMap<String, Value>>,
    pub mm_processor_kwargs: Option<Value>,
    pub media_io_kwargs: Option<Value>,
    pub raw_prompt: Option<String>,
}

/// Borrowed renderer and preprocessor view over [`LlmRequest`].
///
/// The view owns only the JSON values required by the current template ABI. It
/// does not construct or serialize an `NvCreateChatCompletionRequest`.
pub(crate) struct LlmRequestView<'a> {
    request: &'a LlmRequest,
    execution: &'a DynamoExecutionOptions,
    messages: Vec<Value>,
    tools: Option<Value>,
    tool_choice: Option<Value>,
}

impl<'a> LlmRequestView<'a> {
    pub(crate) fn try_new(
        request: &'a LlmRequest,
        execution: &'a DynamoExecutionOptions,
    ) -> Result<Self> {
        request
            .model
            .as_deref()
            .filter(|model| !model.is_empty())
            .context("Switchyard LlmRequest.model is required")?;

        Ok(Self {
            request,
            execution,
            messages: render_messages(request)?,
            tools: render_tools(request),
            tool_choice: request.tool_choice.as_ref().map(render_tool_choice),
        })
    }
}

impl OAIChatLikeRequest for LlmRequestView<'_> {
    fn model(&self) -> String {
        self.request.model.clone().unwrap_or_default()
    }

    fn messages(&self) -> TemplateValue {
        TemplateValue::from_serialize(&self.messages)
    }

    fn tools(&self) -> Option<TemplateValue> {
        self.tools.as_ref().map(TemplateValue::from_serialize)
    }

    fn tool_choice(&self) -> Option<TemplateValue> {
        self.tool_choice.as_ref().map(TemplateValue::from_serialize)
    }

    fn response_format(&self) -> Option<TemplateValue> {
        self.request
            .output
            .response_format
            .as_ref()
            .map(TemplateValue::from_serialize)
    }

    fn reasoning_effort(&self) -> Option<TemplateValue> {
        self.request
            .reasoning
            .effort
            .as_ref()
            .map(TemplateValue::from_serialize)
    }

    fn should_add_generation_prompt(&self) -> bool {
        self.execution.common.continue_final_message != Some(true)
            && self.execution.common.add_generation_prompt.unwrap_or(true)
    }

    fn extract_text(&self) -> Option<TextInput> {
        Some(TextInput::Single(String::new()))
    }

    fn chat_template_args(&self) -> Option<&HashMap<String, Value>> {
        self.execution.chat_template_args.as_ref()
    }

    fn mm_processor_kwargs(&self) -> Option<&Value> {
        self.execution.mm_processor_kwargs.as_ref()
    }
}

impl MediaRequestExt for LlmRequestView<'_> {
    fn media_io_kwargs(&self) -> Option<&Value> {
        self.execution.media_io_kwargs.as_ref()
    }
}

impl AnnotationsProvider for LlmRequestView<'_> {
    fn annotations(&self) -> Option<Vec<String>> {
        if self.execution.annotations.is_empty() {
            self.execution
                .nvext
                .as_ref()
                .and_then(|nvext| nvext.annotations.clone())
        } else {
            Some(self.execution.annotations.clone())
        }
    }
}

impl SamplingOptionsProvider for LlmRequestView<'_> {
    fn extract_sampling_options(&self) -> Result<SamplingOptions> {
        let mut options = self.execution.sampling_options.clone();

        if options.temperature.is_none() {
            options.temperature = self
                .request
                .sampling
                .temperature
                .map(|value| checked_f32("temperature", value))
                .transpose()?;
        }
        if options.top_p.is_none() {
            options.top_p = self
                .request
                .sampling
                .top_p
                .map(|value| checked_f32("top_p", value))
                .transpose()?;
        }
        if options.top_k.is_none() {
            options.top_k = match self.execution.common.top_k {
                Some(value) => Some(value),
                None => self
                    .request
                    .sampling
                    .top_k
                    .map(|value| {
                        i32::try_from(value)
                            .with_context(|| format!("top_k {value} does not fit in i32"))
                    })
                    .transpose()?,
            };
        }
        if options.top_k == Some(0) {
            options.top_k = Some(-1);
        }
        if options.min_p.is_none() {
            options.min_p = self.execution.common.min_p;
        }
        if options.repetition_penalty.is_none() {
            options.repetition_penalty = self.execution.common.repetition_penalty;
        }
        if options.include_stop_str_in_output.is_none() {
            options.include_stop_str_in_output = self.execution.common.include_stop_str_in_output;
        }

        options.temperature = validate_range(options.temperature, &TEMPERATURE_RANGE)
            .context("invalid Switchyard temperature")?;
        validate_top_p(options.top_p).context("invalid Switchyard top_p")?;
        options.frequency_penalty =
            validate_range(options.frequency_penalty, &FREQUENCY_PENALTY_RANGE)
                .context("invalid frequency_penalty")?;
        options.presence_penalty =
            validate_range(options.presence_penalty, &PRESENCE_PENALTY_RANGE)
                .context("invalid presence_penalty")?;
        options.min_p = validate_range(options.min_p, &MIN_P_RANGE).context("invalid min_p")?;
        options.n = validate_range(options.n, &N_RANGE).context("invalid n")?;
        options.best_of =
            validate_range(options.best_of, &BEST_OF_RANGE).context("invalid best_of")?;

        if self
            .execution
            .nvext
            .as_ref()
            .is_some_and(|nvext| nvext.greed_sampling.unwrap_or(false))
        {
            options.temperature = None;
            options.top_p = None;
        }

        if options.guided_decoding.is_none() {
            let guided_json = self.execution.common.guided_json.clone().or_else(|| {
                guided_json_from_response_format(self.request.output.response_format.as_ref())
            });
            options.guided_decoding = GuidedDecodingOptions::from_optional(
                guided_json,
                self.execution.common.guided_regex.clone(),
                self.execution.common.guided_choice.clone(),
                self.execution.common.guided_grammar.clone(),
                self.execution.common.guided_decoding_backend.clone(),
                self.execution.common.guided_whitespace_pattern.clone(),
                None,
            )?;
        }

        Ok(options)
    }
}

impl StopConditionsProvider for LlmRequestView<'_> {
    fn extract_stop_conditions(&self) -> Result<StopConditions> {
        let mut conditions = self.execution.stop_conditions.clone();
        if conditions.max_tokens.is_none() {
            conditions.max_tokens = self
                .request
                .output
                .max_output_tokens
                .map(|value| {
                    u32::try_from(value)
                        .with_context(|| format!("max_output_tokens {value} does not fit in u32"))
                })
                .transpose()?;
        }
        if conditions.min_tokens.is_none() {
            conditions.min_tokens = self.execution.common.min_tokens;
        }
        if conditions.ignore_eos.is_none() {
            conditions.ignore_eos = self.execution.common.ignore_eos;
        }
        if conditions.max_thinking_tokens.is_none() {
            conditions.max_thinking_tokens = self
                .execution
                .nvext
                .as_ref()
                .and_then(|nvext| nvext.max_thinking_tokens);
        }
        if conditions.stop.as_ref().is_some_and(|stop| stop.len() > 4) {
            bail!("stop conditions must be less than 4");
        }
        if conditions
            .stop_token_ids
            .as_ref()
            .is_some_and(|stop| stop.len() > 4)
        {
            bail!("stop token IDs must be less than 4");
        }
        Ok(conditions)
    }
}

impl OutputOptionsProvider for LlmRequestView<'_> {
    fn extract_output_options(&self) -> Result<OutputOptions> {
        let mut options = self.execution.output_options.clone();
        if options.prompt_logprobs.is_none() {
            options.prompt_logprobs = self.execution.common.prompt_logprobs;
        }
        if options.skip_special_tokens.is_none() {
            options.skip_special_tokens = self.execution.common.skip_special_tokens;
        }
        Ok(options)
    }
}

impl NvExtProvider for LlmRequestView<'_> {
    fn nvext(&self) -> Option<&NvExt> {
        self.execution.nvext.as_ref()
    }

    fn raw_prompt(&self) -> Option<String> {
        self.execution.raw_prompt.clone()
    }
}

impl CommonExtProvider for LlmRequestView<'_> {
    fn common_ext(&self) -> Option<&CommonExt> {
        Some(&self.execution.common)
    }

    fn get_guided_json(&self) -> Option<Value> {
        self.execution.common.guided_json.clone()
    }

    fn get_guided_regex(&self) -> Option<String> {
        self.execution.common.guided_regex.clone()
    }

    fn get_guided_grammar(&self) -> Option<String> {
        self.execution.common.guided_grammar.clone()
    }

    fn get_guided_choice(&self) -> Option<Vec<String>> {
        self.execution.common.guided_choice.clone()
    }

    fn get_guided_decoding_backend(&self) -> Option<String> {
        self.execution.common.guided_decoding_backend.clone()
    }

    fn get_guided_whitespace_pattern(&self) -> Option<String> {
        self.execution.common.guided_whitespace_pattern.clone()
    }

    fn get_top_k(&self) -> Option<i32> {
        self.execution.common.top_k
    }

    fn get_min_p(&self) -> Option<f32> {
        self.execution.common.min_p
    }

    fn get_repetition_penalty(&self) -> Option<f32> {
        self.execution.common.repetition_penalty
    }

    fn get_include_stop_str_in_output(&self) -> Option<bool> {
        self.execution.common.include_stop_str_in_output
    }

    fn get_skip_special_tokens(&self) -> Option<bool> {
        self.execution.common.skip_special_tokens
    }

    fn get_prompt_logprobs_count(&self) -> Option<u32> {
        self.execution.common.prompt_logprobs
    }
}

fn checked_f32(name: &str, value: f64) -> Result<f32> {
    let converted = value as f32;
    if !value.is_finite() || !converted.is_finite() {
        bail!("{name} must be a finite f32, got {value}");
    }
    Ok(converted)
}

fn guided_json_from_response_format(format: Option<&Value>) -> Option<Value> {
    let format = format?.as_object()?;
    match format.get("type").and_then(Value::as_str) {
        Some("json_object") => Some(json!({ "type": "object" })),
        Some("json_schema") => format
            .get("json_schema")
            .and_then(Value::as_object)
            .and_then(|schema| schema.get("schema"))
            .cloned(),
        _ => None,
    }
}

fn render_messages(request: &LlmRequest) -> Result<Vec<Value>> {
    let mut rendered = Vec::with_capacity(request.instructions.len() + request.messages.len());

    for instruction in &request.instructions {
        rendered.push(json!({
            "role": role_name(instruction.role),
            "content": plain_text(&instruction.content, "instruction")?,
        }));
    }
    for message in &request.messages {
        render_message(message, &mut rendered)?;
    }

    Ok(rendered)
}

fn render_message(message: &Message, rendered: &mut Vec<Value>) -> Result<()> {
    if message.role == Role::Assistant {
        rendered.push(render_assistant_message(message)?);
        return Ok(());
    }

    let tool_results = message
        .content
        .iter()
        .filter_map(|block| match block {
            ContentBlock::ToolResult(result) => Some(result),
            _ => None,
        })
        .collect::<Vec<_>>();
    if !tool_results.is_empty() {
        if tool_results.len() != message.content.len() {
            bail!("a Switchyard message cannot mix tool results with other content in this POC");
        }
        for result in tool_results {
            rendered.push(render_tool_result(result)?);
        }
        return Ok(());
    }

    if message.role == Role::Tool {
        bail!("a Switchyard tool message must contain a ToolResult block");
    }

    rendered.push(json!({
        "role": role_name(message.role),
        "content": plain_text(&message.content, "message")?,
    }));
    Ok(())
}

fn render_assistant_message(message: &Message) -> Result<Value> {
    let mut text = String::new();
    let mut refusal = String::new();
    let mut pending_reasoning = String::new();
    let mut reasoning_segments = Vec::new();
    let mut tool_calls = Vec::new();
    let mut has_reasoning = false;

    for block in &message.content {
        match block {
            ContentBlock::Text { text: value } => text.push_str(value),
            ContentBlock::Refusal { text: value } => refusal.push_str(value),
            ContentBlock::Reasoning { text: value, .. } => {
                if !pending_reasoning.is_empty() {
                    pending_reasoning.push('\n');
                }
                pending_reasoning.push_str(value);
                has_reasoning = true;
            }
            ContentBlock::ToolCall(call) => {
                reasoning_segments.push(std::mem::take(&mut pending_reasoning));
                tool_calls.push(json!({
                    "id": call.id,
                    "type": "function",
                    "function": {
                        "name": call.name,
                        "arguments": serde_json::to_string(&call.arguments)?,
                    },
                }));
            }
            ContentBlock::ToolResult(_) => {
                bail!("an assistant message cannot contain a ToolResult block");
            }
            ContentBlock::Image { .. }
            | ContentBlock::Audio { .. }
            | ContentBlock::Video { .. }
            | ContentBlock::File { .. } => {
                bail!("multimodal Switchyard lowering is not implemented in this POC");
            }
            ContentBlock::Unknown { provider, .. } => {
                bail!("cannot render an unknown {provider} content block");
            }
        }
    }

    let mut output = Map::new();
    output.insert("role".to_string(), Value::String("assistant".to_string()));
    output.insert(
        "content".to_string(),
        if text.is_empty() {
            Value::Null
        } else {
            Value::String(text)
        },
    );
    if !refusal.is_empty() {
        output.insert("refusal".to_string(), Value::String(refusal));
    }
    if !tool_calls.is_empty() {
        reasoning_segments.push(pending_reasoning);
        output.insert("tool_calls".to_string(), Value::Array(tool_calls));
        if has_reasoning {
            output.insert(
                "reasoning_content".to_string(),
                Value::Array(reasoning_segments.into_iter().map(Value::String).collect()),
            );
        }
    } else if has_reasoning {
        output.insert(
            "reasoning_content".to_string(),
            Value::String(pending_reasoning),
        );
    }

    Ok(Value::Object(output))
}

fn render_tool_result(result: &ToolResult) -> Result<Value> {
    Ok(json!({
        "role": "tool",
        "tool_call_id": result.tool_call_id,
        "content": plain_text(&result.content, "tool result")?,
    }))
}

fn plain_text(blocks: &[ContentBlock], context: &str) -> Result<String> {
    let mut output = String::new();
    for block in blocks {
        match block {
            ContentBlock::Text { text } | ContentBlock::Refusal { text } => output.push_str(text),
            ContentBlock::Unknown { provider, .. } => {
                bail!("cannot render an unknown {provider} block in {context}");
            }
            ContentBlock::Image { .. }
            | ContentBlock::Audio { .. }
            | ContentBlock::Video { .. }
            | ContentBlock::File { .. } => {
                bail!("multimodal Switchyard lowering is not implemented in this POC");
            }
            ContentBlock::Reasoning { .. }
            | ContentBlock::ToolCall(_)
            | ContentBlock::ToolResult(_) => {
                bail!("unsupported Switchyard block in {context}: {block:?}");
            }
        }
    }
    Ok(output)
}

fn role_name(role: Role) -> &'static str {
    match role {
        Role::System => "system",
        Role::Developer => "developer",
        Role::User => "user",
        Role::Assistant => "assistant",
        Role::Tool => "tool",
    }
}

fn render_tools(request: &LlmRequest) -> Option<Value> {
    (!request.tools.is_empty()).then(|| {
        Value::Array(
            request
                .tools
                .iter()
                .map(|tool| {
                    let mut function = Map::new();
                    function.insert("name".to_string(), Value::String(tool.name.clone()));
                    if let Some(description) = &tool.description {
                        function.insert(
                            "description".to_string(),
                            Value::String(description.clone()),
                        );
                    }
                    function.insert("parameters".to_string(), tool.parameters.clone());
                    if let Some(strict) = tool.strict {
                        function.insert("strict".to_string(), Value::Bool(strict));
                    }
                    json!({ "type": "function", "function": function })
                })
                .collect(),
        )
    })
}

fn render_tool_choice(choice: &ToolChoice) -> Value {
    match choice {
        ToolChoice::Auto => Value::String("auto".to_string()),
        ToolChoice::Required => Value::String("required".to_string()),
        ToolChoice::None => Value::String("none".to_string()),
        ToolChoice::Tool { name } => json!({
            "type": "function",
            "function": { "name": name },
        }),
        ToolChoice::Raw(value) => value.clone(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use crate::model_card::ModelDeploymentCard;
    use crate::preprocessor::OpenAIPreprocessor;
    use crate::protocols::anthropic::AnthropicCreateMessageRequest;
    use crate::protocols::openai::chat_completions::NvCreateChatCompletionRequest;
    use crate::protocols::openai::responses::NvCreateResponse;
    use switchyard_protocol::llm::{ReasoningParams, ToolCall};
    use switchyard_translation::{
        PreservationPolicy, TranslationEngine, TranslationPolicy, WireFormat,
    };

    const MODEL_PATH: &str = "tests/data/sample-models/mock-llama-3.1-8b-instruct";

    fn preprocessor() -> std::sync::Arc<OpenAIPreprocessor> {
        let mut mdc = ModelDeploymentCard::load_from_disk(MODEL_PATH, None).unwrap();
        mdc.set_name("test-model");
        OpenAIPreprocessor::new(mdc).unwrap()
    }

    fn decode_request(format: WireFormat, body: &Value) -> LlmRequest {
        let policy = TranslationPolicy {
            preservation: PreservationPolicy::Disabled,
            ..TranslationPolicy::default()
        };
        let decoded = TranslationEngine::default()
            .decode_request(format, body, &policy)
            .unwrap();
        assert!(
            decoded.diagnostics.is_empty(),
            "unexpected {format} diagnostics: {:?}",
            decoded.diagnostics
        );
        assert!(decoded.request.preservation.requests.is_empty());
        decoded.request
    }

    fn legacy_chat_request(format: WireFormat, body: Value) -> NvCreateChatCompletionRequest {
        match format {
            WireFormat::OpenAiChat => serde_json::from_value(body).unwrap(),
            WireFormat::AnthropicMessages => {
                let request: AnthropicCreateMessageRequest = serde_json::from_value(body).unwrap();
                request.try_into().unwrap()
            }
            WireFormat::OpenAiResponses => {
                let request: NvCreateResponse = serde_json::from_value(body).unwrap();
                request.try_into().unwrap()
            }
        }
    }

    #[tokio::test]
    async fn direct_lowering_matches_current_path_for_three_api_shapes() {
        let cases = [
            (
                WireFormat::OpenAiChat,
                json!({
                    "model": "test-model",
                    "messages": [
                        { "role": "system", "content": "You are concise." },
                        { "role": "user", "content": "Hello" }
                    ],
                    "max_completion_tokens": 17,
                    "temperature": 0.2,
                    "top_p": 0.8
                }),
            ),
            (
                WireFormat::OpenAiResponses,
                json!({
                    "model": "test-model",
                    "instructions": "You are concise.",
                    "input": "Hello",
                    "max_output_tokens": 17,
                    "temperature": 0.2,
                    "top_p": 0.8
                }),
            ),
            (
                WireFormat::AnthropicMessages,
                json!({
                    "model": "test-model",
                    "system": "You are concise.",
                    "messages": [{ "role": "user", "content": "Hello" }],
                    "max_tokens": 17,
                    "temperature": 0.2,
                    "top_p": 0.8
                }),
            ),
        ];
        let preprocessor = preprocessor();

        for (format, body) in cases {
            let legacy_request = legacy_chat_request(format, body.clone());
            let llm_request = decode_request(format, &body);
            let execution = DynamoExecutionOptions::default();

            let legacy = preprocessor
                .preprocess_request(&legacy_request, None)
                .await
                .unwrap()
                .0;
            let direct = preprocessor
                .preprocess_llm_request(&llm_request, &execution, None)
                .await
                .unwrap()
                .0;

            assert_eq!(
                serde_json::to_value(direct).unwrap(),
                serde_json::to_value(legacy).unwrap(),
                "direct preprocessing diverged for {format}"
            );
        }
    }

    #[tokio::test]
    async fn typed_execution_options_fill_fields_missing_from_llm_request() {
        let body = json!({
            "model": "test-model",
            "messages": [{ "role": "user", "content": "Hello" }],
            "max_completion_tokens": 17,
            "temperature": 0.2
        });
        let request = decode_request(WireFormat::OpenAiChat, &body);
        let mut execution = DynamoExecutionOptions::default();
        execution.stop_conditions.stop = Some(vec!["DONE".to_string()]);
        execution.sampling_options.presence_penalty = Some(0.5);
        execution.sampling_options.seed = Some(42);
        execution.output_options.logprobs = Some(5);
        execution.common.min_tokens = Some(2);
        execution.common.min_p = Some(0.1);
        execution.common.prompt_logprobs = Some(3);
        execution.common.skip_special_tokens = Some(false);
        execution.annotations = vec!["token_ids".to_string()];

        let preprocessed = preprocessor()
            .preprocess_llm_request(&request, &execution, None)
            .await
            .unwrap()
            .0;

        assert_eq!(preprocessed.stop_conditions.max_tokens, Some(17));
        assert_eq!(preprocessed.stop_conditions.min_tokens, Some(2));
        assert_eq!(
            preprocessed.stop_conditions.stop,
            Some(vec!["DONE".to_string()])
        );
        assert_eq!(preprocessed.sampling_options.temperature, Some(0.2));
        assert_eq!(preprocessed.sampling_options.min_p, Some(0.1));
        assert_eq!(preprocessed.sampling_options.presence_penalty, Some(0.5));
        assert_eq!(preprocessed.sampling_options.seed, Some(42));
        assert_eq!(preprocessed.output_options.logprobs, Some(5));
        assert_eq!(preprocessed.output_options.prompt_logprobs, Some(3));
        assert_eq!(preprocessed.output_options.skip_special_tokens, Some(false));
        assert_eq!(preprocessed.annotations, vec!["token_ids".to_string()]);
    }

    #[test]
    fn preserves_interleaved_reasoning_and_tool_calls() {
        let request = LlmRequest {
            model: Some("test-model".to_string()),
            messages: vec![Message {
                role: Role::Assistant,
                content: vec![
                    ContentBlock::Reasoning {
                        text: "first".to_string(),
                        signature: Some("sig-1".to_string()),
                    },
                    ContentBlock::ToolCall(ToolCall {
                        id: "call-1".to_string(),
                        name: "one".to_string(),
                        arguments: json!({ "x": 1 }),
                    }),
                    ContentBlock::Reasoning {
                        text: "second".to_string(),
                        signature: Some("sig-2".to_string()),
                    },
                    ContentBlock::ToolCall(ToolCall {
                        id: "call-2".to_string(),
                        name: "two".to_string(),
                        arguments: json!({ "y": 2 }),
                    }),
                    ContentBlock::Reasoning {
                        text: "third".to_string(),
                        signature: None,
                    },
                ],
            }],
            reasoning: ReasoningParams::default(),
            ..LlmRequest::default()
        };
        let execution = DynamoExecutionOptions::default();
        let view = LlmRequestView::try_new(&request, &execution).unwrap();

        assert_eq!(
            view.messages[0]["reasoning_content"],
            json!(["first", "second", "third"])
        );
        assert_eq!(view.messages[0]["tool_calls"][0]["id"], "call-1");
        assert_eq!(view.messages[0]["tool_calls"][1]["id"], "call-2");
    }

    #[test]
    fn rejects_multimodal_content_instead_of_silently_dropping_it() {
        let request = LlmRequest {
            model: Some("test-model".to_string()),
            messages: vec![Message {
                role: Role::User,
                content: vec![ContentBlock::Image {
                    source: switchyard_protocol::llm::ImageSource::Url {
                        url: "https://example.com/image.png".to_string(),
                        detail: None,
                    },
                }],
            }],
            ..LlmRequest::default()
        };
        let execution = DynamoExecutionOptions::default();

        let error = LlmRequestView::try_new(&request, &execution)
            .err()
            .expect("multimodal input must fail closed");
        assert!(
            error
                .to_string()
                .contains("multimodal Switchyard lowering is not implemented")
        );
    }
}
