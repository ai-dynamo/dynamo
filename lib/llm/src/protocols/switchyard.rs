// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! POC adapter from Switchyard's provider-neutral request IR into Dynamo preprocessing.
//!
//! This module deliberately stops at the frontend. Only [`PreprocessedRequest`](crate::preprocessor::PreprocessedRequest)
//! crosses the frontend/worker boundary.

use std::collections::HashMap;

use anyhow::{Context, Result, bail};
use dynamo_protocols::types::{
    ChatCompletionRequestMessage, ChatCompletionRequestToolMessageContent,
    ChatCompletionRequestToolMessageContentPart, ChatCompletionRequestUserMessageContent,
    ChatCompletionRequestUserMessageContentPart,
};
use dynamo_renderer::{OAIChatLikeRequest, TextInput};
use dynamo_runtime::protocols::annotated::AnnotationsProvider;
use minijinja::value::Value as TemplateValue;
use serde_json::{Map, Value, json};
use switchyard_protocol::llm::{
    ContentBlock, ImageSource, InstructionBlock, LlmRequest, MediaSource, Message, Role,
    ToolChoice, ToolResult,
};
use switchyard_translation::{
    LossyConversionPolicy, PreservationPolicy, TranslationEngine, TranslationPolicy,
    UnknownFieldPolicy, WireFormat,
};

use crate::preprocessor::prompt::MediaRequestExt;
use crate::protocols::common::{
    GuidedDecodingOptions, OutputOptions, OutputOptionsProvider, SamplingOptions,
    SamplingOptionsProvider, StopConditions, StopConditionsProvider,
    extensions::{NvExt, NvExtProvider},
};
use crate::protocols::openai::chat_completions::NvCreateChatCompletionRequest;
use crate::protocols::openai::common_ext::{CommonExt, CommonExtProvider};
use crate::protocols::openai::validate::{
    BEST_OF_RANGE, FREQUENCY_PENALTY_RANGE, MIN_P_RANGE, N_RANGE, PRESENCE_PENALTY_RANGE,
    TEMPERATURE_RANGE, validate_range, validate_top_p,
};

/// Pipeline context key for a provider-neutral request that replaces Chat as
/// the prompt/preprocessing IR.
pub const LLM_REQUEST_CONTEXT_KEY: &str = "dynamo.llm_request";

/// Provider-neutral semantic request plus Dynamo execution controls.
#[derive(Debug, Clone)]
pub struct DynamoLlmRequest {
    pub request: LlmRequest,
    pub execution: DynamoExecutionOptions,
}

/// Decode one supported public wire shape into the request IR used by Dynamo
/// preprocessing. Full-body preservation is disabled on the serving path.
pub fn decode_wire_request(
    format: WireFormat,
    mut body: Value,
    execution: DynamoExecutionOptions,
) -> Result<DynamoLlmRequest> {
    normalize_wire_request(format, &mut body);
    let policy = TranslationPolicy {
        // Dynamo extensions are extracted into `DynamoExecutionOptions`; keep
        // accepting them without retaining a second copy in the semantic IR.
        unknown_field_policy: UnknownFieldPolicy::Preserve,
        lossy_conversion_policy: LossyConversionPolicy::Reject,
        preservation: PreservationPolicy::Disabled,
        ..TranslationPolicy::default()
    };
    let decoded = TranslationEngine::default()
        .decode_request(format, &body, &policy)
        .with_context(|| format!("failed to decode {format} request into LlmRequest"))?;
    if !decoded.diagnostics.is_empty() {
        bail!(
            "{format} request produced lossy Switchyard diagnostics: {:?}",
            decoded.diagnostics
        );
    }
    let mut request = decoded.request;
    if format == WireFormat::OpenAiResponses {
        normalize_responses_prompt_contract(&mut request)?;
    }
    normalize_dynamo_media_blocks(&mut request)?;
    Ok(DynamoLlmRequest { request, execution })
}

fn normalize_wire_request(format: WireFormat, body: &mut Value) {
    if format != WireFormat::OpenAiResponses {
        return;
    }
    let Some(items) = body.get_mut("input").and_then(Value::as_array_mut) else {
        return;
    };
    for item in items {
        let Some(item) = item.as_object_mut() else {
            continue;
        };
        if !item.contains_key("type") && item.contains_key("role") && item.contains_key("content") {
            item.insert("type".to_string(), Value::String("message".to_string()));
        }
    }
}

/// Preserve the template-facing behavior of Dynamo's existing Responses path
/// before the provider-neutral request enters generic preprocessing.
fn normalize_responses_prompt_contract(request: &mut LlmRequest) -> Result<()> {
    for message in &mut request.messages {
        if matches!(message.role, Role::System | Role::Developer) {
            message.role = Role::System;
        }
        if message.role == Role::Assistant {
            for block in &mut message.content {
                if let ContentBlock::Refusal { text } = block {
                    *block = ContentBlock::Text {
                        text: std::mem::take(text),
                    };
                }
            }
        }
    }

    let leading_system_count = request
        .messages
        .iter()
        .take_while(|message| message.role == Role::System)
        .count();
    if request.instructions.is_empty() && leading_system_count == 0 {
        return Ok(());
    }

    let mut segments = Vec::with_capacity(request.instructions.len() + leading_system_count);
    for instruction in std::mem::take(&mut request.instructions) {
        segments.push(plain_text(&instruction.content, "Responses instruction")?);
    }
    for message in request.messages.drain(..leading_system_count) {
        segments.push(plain_text(
            &message.content,
            "Responses leading system message",
        )?);
    }
    request.instructions.push(InstructionBlock {
        role: Role::System,
        content: vec![ContentBlock::Text {
            text: segments.join("\n\n"),
        }],
    });
    Ok(())
}

/// Capture Dynamo execution controls from a Chat request without using Chat as
/// the prompt IR. This is also the compatibility source for fields Switchyard
/// 0.2 does not model yet.
pub fn execution_from_chat(
    request: &NvCreateChatCompletionRequest,
) -> Result<DynamoExecutionOptions> {
    Ok(DynamoExecutionOptions {
        stop_conditions: request.extract_stop_conditions()?,
        sampling_options: request.extract_sampling_options()?,
        output_options: request.extract_output_options()?,
        nvext: request.nvext.clone(),
        annotations: request.annotations().unwrap_or_default(),
        common: request.common.clone(),
        chat_template_args: request.chat_template_args.clone(),
        mm_processor_kwargs: request.inner.mm_processor_kwargs.clone(),
        media_io_kwargs: request.media_io_kwargs.clone(),
        image_uuids: chat_image_uuids(&request.inner.messages),
        raw_prompt: request.raw_prompt(),
        unsupported_fields: request.unsupported_fields.clone(),
    })
}

fn chat_image_uuids(messages: &[ChatCompletionRequestMessage]) -> Vec<Option<String>> {
    let mut uuids = Vec::new();
    for message in messages {
        match message {
            ChatCompletionRequestMessage::User(user) => {
                if let ChatCompletionRequestUserMessageContent::Array(parts) = &user.content {
                    uuids.extend(parts.iter().filter_map(|part| match part {
                        ChatCompletionRequestUserMessageContentPart::ImageUrl(image) => {
                            Some(image.uuid.clone())
                        }
                        _ => None,
                    }));
                }
            }
            ChatCompletionRequestMessage::Tool(tool) => {
                if let ChatCompletionRequestToolMessageContent::Array(parts) = &tool.content {
                    uuids.extend(parts.iter().filter_map(|part| match part {
                        ChatCompletionRequestToolMessageContentPart::ImageUrl(image) => {
                            Some(image.uuid.clone())
                        }
                        _ => None,
                    }));
                }
            }
            _ => {}
        }
    }
    uuids
}

fn normalize_dynamo_media_blocks(request: &mut LlmRequest) -> Result<()> {
    for instruction in &mut request.instructions {
        normalize_media_blocks(&mut instruction.content)?;
    }
    for message in &mut request.messages {
        normalize_media_blocks(&mut message.content)?;
        for block in &mut message.content {
            if let ContentBlock::ToolResult(result) = block {
                normalize_media_blocks(&mut result.content)?;
            }
        }
    }
    Ok(())
}

fn normalize_media_blocks(blocks: &mut [ContentBlock]) -> Result<()> {
    for block in blocks {
        let ContentBlock::Unknown { raw, .. } = block else {
            continue;
        };
        let Some(object) = raw.as_object() else {
            continue;
        };
        let Some(kind) = object.get("type").and_then(Value::as_str) else {
            continue;
        };
        let (block_kind, field) = match kind {
            "video_url" => ("video", "video_url"),
            "audio_url" => ("audio", "audio_url"),
            _ => continue,
        };
        let payload = object
            .get(field)
            .with_context(|| format!("{kind} content block is missing `{field}`"))?;
        let (url, media_type) = match payload {
            Value::String(url) => (url.clone(), None),
            Value::Object(payload) => (
                payload
                    .get("url")
                    .and_then(Value::as_str)
                    .with_context(|| format!("{kind}.{field} is missing `url`"))?
                    .to_string(),
                payload
                    .get("media_type")
                    .and_then(Value::as_str)
                    .map(str::to_string),
            ),
            _ => bail!("{kind}.{field} must be a URL string or object"),
        };
        let source = MediaSource::Url { url, media_type };
        *block = match block_kind {
            "video" => ContentBlock::Video { source },
            "audio" => ContentBlock::Audio { source },
            _ => unreachable!(),
        };
    }
    Ok(())
}

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
    /// vLLM image-cache identities in image-block traversal order.
    /// Switchyard 0.2 does not yet model these provider extensions.
    pub image_uuids: Vec<Option<String>>,
    pub raw_prompt: Option<String>,
    /// Accepted backend passthrough controls not modeled by the provider-neutral IR.
    pub unsupported_fields: HashMap<String, Value>,
}

/// Borrowed renderer and preprocessor view over [`LlmRequest`].
///
/// The view owns only the JSON values required by the current template ABI. It
/// does not construct or serialize an `NvCreateChatCompletionRequest`.
pub(crate) struct LlmRequestView<'a> {
    request: &'a LlmRequest,
    execution: &'a DynamoExecutionOptions,
    messages: Vec<Value>,
    media_parts: Vec<ChatCompletionRequestUserMessageContentPart>,
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

        let (messages, media_parts) = render_messages(request, execution)?;
        Ok(Self {
            request,
            execution,
            messages,
            media_parts,
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

    fn canonical_media_parts(&self) -> Option<&[ChatCompletionRequestUserMessageContentPart]> {
        Some(&self.media_parts)
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

    fn unsupported_fields(&self) -> Option<&HashMap<String, Value>> {
        Some(&self.execution.unsupported_fields)
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

fn render_messages(
    request: &LlmRequest,
    execution: &DynamoExecutionOptions,
) -> Result<(Vec<Value>, Vec<ChatCompletionRequestUserMessageContentPart>)> {
    let mut rendered = Vec::with_capacity(request.instructions.len() + request.messages.len());
    let mut media_parts = Vec::new();
    let mut image_uuid_index = 0;

    for instruction in &request.instructions {
        rendered.push(json!({
            "role": role_name(instruction.role),
            "content": plain_text(&instruction.content, "instruction")?,
        }));
    }
    for message in &request.messages {
        render_message(
            message,
            &mut rendered,
            &mut media_parts,
            &execution.image_uuids,
            &mut image_uuid_index,
        )?;
    }
    if image_uuid_index < execution.image_uuids.len() {
        bail!(
            "received {} image UUIDs for {image_uuid_index} Switchyard image blocks",
            execution.image_uuids.len()
        );
    }

    Ok((rendered, media_parts))
}

fn render_message(
    message: &Message,
    rendered: &mut Vec<Value>,
    media_parts: &mut Vec<ChatCompletionRequestUserMessageContentPart>,
    image_uuids: &[Option<String>],
    image_uuid_index: &mut usize,
) -> Result<()> {
    if message.role == Role::Assistant {
        rendered.push(render_assistant_message(message)?);
        return Ok(());
    }

    if message
        .content
        .iter()
        .any(|block| matches!(block, ContentBlock::ToolResult(_)))
    {
        let mut segment_start = 0;
        for (index, block) in message.content.iter().enumerate() {
            let ContentBlock::ToolResult(result) = block else {
                continue;
            };
            if segment_start < index {
                render_non_assistant_message(
                    message.role,
                    &message.content[segment_start..index],
                    rendered,
                    media_parts,
                    image_uuids,
                    image_uuid_index,
                )?;
            }
            rendered.push(render_tool_result(
                result,
                media_parts,
                image_uuids,
                image_uuid_index,
            )?);
            segment_start = index + 1;
        }
        if segment_start < message.content.len() {
            render_non_assistant_message(
                message.role,
                &message.content[segment_start..],
                rendered,
                media_parts,
                image_uuids,
                image_uuid_index,
            )?;
        }
        return Ok(());
    }

    render_non_assistant_message(
        message.role,
        &message.content,
        rendered,
        media_parts,
        image_uuids,
        image_uuid_index,
    )
}

fn render_non_assistant_message(
    role: Role,
    content_blocks: &[ContentBlock],
    rendered: &mut Vec<Value>,
    media_parts: &mut Vec<ChatCompletionRequestUserMessageContentPart>,
    image_uuids: &[Option<String>],
    image_uuid_index: &mut usize,
) -> Result<()> {
    if role == Role::Tool {
        bail!("a Switchyard tool message must contain a ToolResult block");
    }

    let content = if role == Role::User {
        render_content(
            content_blocks,
            "message",
            media_parts,
            image_uuids,
            image_uuid_index,
        )?
    } else {
        Value::String(plain_text(content_blocks, "message")?)
    };
    rendered.push(json!({ "role": role_name(role), "content": content }));
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
                bail!("assistant multimodal content is not supported by Dynamo preprocessing");
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

fn render_tool_result(
    result: &ToolResult,
    media_parts: &mut Vec<ChatCompletionRequestUserMessageContentPart>,
    image_uuids: &[Option<String>],
    image_uuid_index: &mut usize,
) -> Result<Value> {
    Ok(json!({
        "role": "tool",
        "tool_call_id": result.tool_call_id,
        "content": render_content(
            &result.content,
            "tool result",
            media_parts,
            image_uuids,
            image_uuid_index,
        )?,
    }))
}

fn render_content(
    blocks: &[ContentBlock],
    context: &str,
    media_parts: &mut Vec<ChatCompletionRequestUserMessageContentPart>,
    image_uuids: &[Option<String>],
    image_uuid_index: &mut usize,
) -> Result<Value> {
    let has_media = blocks.iter().any(|block| {
        matches!(
            block,
            ContentBlock::Image { .. } | ContentBlock::Audio { .. } | ContentBlock::Video { .. }
        )
    });
    if !has_media {
        return Ok(Value::String(plain_text(blocks, context)?));
    }

    let mut rendered = Vec::with_capacity(blocks.len());
    for block in blocks {
        let part = match block {
            ContentBlock::Text { text } | ContentBlock::Refusal { text } => {
                json!({ "type": "text", "text": text })
            }
            ContentBlock::Image { source } => {
                let uuid = image_uuids.get(*image_uuid_index).cloned().unwrap_or(None);
                *image_uuid_index += 1;
                render_image_part(source, uuid)?
            }
            ContentBlock::Audio { source } => render_media_part("audio_url", source)?,
            ContentBlock::Video { source } => render_media_part("video_url", source)?,
            ContentBlock::File { .. } => {
                bail!("file content is not supported by Dynamo preprocessing in {context}")
            }
            ContentBlock::Unknown { provider, .. } => {
                bail!("cannot render an unknown {provider} block in {context}")
            }
            ContentBlock::Reasoning { .. }
            | ContentBlock::ToolCall(_)
            | ContentBlock::ToolResult(_) => {
                bail!("unsupported Switchyard block in {context}: {block:?}")
            }
        };
        if !matches!(
            block,
            ContentBlock::Text { .. } | ContentBlock::Refusal { .. }
        ) {
            media_parts.push(serde_json::from_value(part.clone()).with_context(|| {
                format!("invalid canonical multimodal content part in {context}")
            })?);
        }
        rendered.push(part);
    }
    Ok(Value::Array(rendered))
}

fn render_image_part(source: &ImageSource, uuid: Option<String>) -> Result<Value> {
    let (url, detail) = image_url(source)?;
    let mut part = json!({
        "type": "image_url",
        "image_url": { "url": url, "detail": detail },
    });
    if let Some(uuid) = uuid {
        part["uuid"] = Value::String(uuid);
    }
    Ok(part)
}

fn render_media_part(kind: &'static str, source: &MediaSource) -> Result<Value> {
    let url = media_url(source, kind)?;
    let mut part = Map::new();
    part.insert("type".to_string(), Value::String(kind.to_string()));
    let payload = if kind == "video_url" {
        // Match Dynamo's typed Chat-compatible template projection. Detail is
        // optional on the wire but serializes as null in the existing ABI.
        json!({ "url": url, "detail": null })
    } else {
        json!({ "url": url })
    };
    part.insert(kind.to_string(), payload);
    Ok(Value::Object(part))
}

fn image_url(source: &ImageSource) -> Result<(String, Option<String>)> {
    match source {
        ImageSource::Url { url, detail } => Ok((url.clone(), detail.clone())),
        ImageSource::Base64 { media_type, data } => {
            Ok((inline_data_url(media_type.as_deref(), data, "image")?, None))
        }
        ImageSource::Raw(raw) => {
            let raw = raw
                .as_object()
                .context("raw Switchyard image source must be an object")?;
            match raw.get("type").and_then(Value::as_str) {
                Some("base64") => Ok((
                    inline_data_url(
                        raw.get("media_type").and_then(Value::as_str),
                        raw.get("data").and_then(Value::as_str).unwrap_or_default(),
                        "image",
                    )?,
                    None,
                )),
                Some("url") => Ok((
                    raw.get("url")
                        .and_then(Value::as_str)
                        .context("raw Switchyard image URL is missing `url`")?
                        .to_string(),
                    raw.get("detail")
                        .and_then(Value::as_str)
                        .map(str::to_string),
                )),
                other => bail!("unsupported raw Switchyard image source type {other:?}"),
            }
        }
    }
}

fn media_url(source: &MediaSource, kind: &str) -> Result<String> {
    match source {
        MediaSource::Url { url, .. } => Ok(url.clone()),
        MediaSource::Base64 { media_type, data } => {
            inline_data_url(media_type.as_deref(), data, kind)
        }
        MediaSource::Raw(_) => bail!("unsupported raw Switchyard {kind} source"),
    }
}

fn inline_data_url(media_type: Option<&str>, data: &str, kind: &str) -> Result<String> {
    let media_type = media_type.context(format!(
        "base64 Switchyard {kind} content requires a MIME type"
    ))?;
    if data.is_empty() {
        bail!("base64 Switchyard {kind} content requires non-empty data");
    }
    Ok(format!("data:{media_type};base64,{data}"))
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
                bail!("multimodal content is not supported for {context}");
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
    use crate::preprocessor::{BackendOutput, OpenAIPreprocessor, PreprocessedRequest};
    use crate::protocols::Annotated;
    use crate::protocols::anthropic::AnthropicCreateMessageRequest;
    use crate::protocols::openai::chat_completions::NvCreateChatCompletionRequest;
    use crate::protocols::openai::responses::NvCreateResponse;
    use dynamo_runtime::pipeline::{
        AsyncEngine, AsyncEngineContextProvider, Error, ManyOut, Operator, ResponseStream,
        SingleIn, async_trait,
    };
    use futures::stream;
    use std::sync::{Arc, Mutex};
    use switchyard_protocol::llm::{ReasoningParams, ToolCall};
    use switchyard_translation::WireFormat;

    const MODEL_PATH: &str = "tests/data/sample-models/mock-llama-3.1-8b-instruct";

    fn preprocessor() -> std::sync::Arc<OpenAIPreprocessor> {
        let mut mdc = ModelDeploymentCard::load_from_disk(MODEL_PATH, None).unwrap();
        mdc.set_name("test-model");
        OpenAIPreprocessor::new(mdc).unwrap()
    }

    fn decode_request(format: WireFormat, body: &Value) -> LlmRequest {
        decode_wire_request(format, body.clone(), DynamoExecutionOptions::default())
            .unwrap()
            .request
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

    async fn assert_responses_direct_lowering_matches_legacy(body: Value) -> LlmRequest {
        let legacy_request = legacy_chat_request(WireFormat::OpenAiResponses, body.clone());
        let canonical = decode_wire_request(
            WireFormat::OpenAiResponses,
            body,
            execution_from_chat(&legacy_request).unwrap(),
        )
        .unwrap();
        let preprocessor = preprocessor();

        let legacy = preprocessor
            .preprocess_request(&legacy_request, None)
            .await
            .unwrap()
            .0;
        let direct = preprocessor
            .preprocess_llm_request(&canonical.request, &canonical.execution, None)
            .await
            .unwrap()
            .0;

        assert_eq!(
            serde_json::to_value(direct).unwrap(),
            serde_json::to_value(legacy).unwrap(),
            "direct Responses preprocessing diverged from the compatibility path"
        );
        canonical.request
    }

    #[tokio::test]
    async fn direct_multimodal_lowering_matches_chat_compatibility_path() {
        let cases = [
            (
                WireFormat::OpenAiChat,
                json!({
                    "model": "test-model",
                    "messages": [{
                        "role": "user",
                        "content": [
                            { "type": "text", "text": "inspect" },
                            {
                                "type": "image_url",
                                "image_url": { "url": "https://example.com/image.png", "detail": "high" },
                                "uuid": "image-cache-key"
                            },
                            { "type": "video_url", "video_url": { "url": "https://example.com/video.mp4" } },
                            { "type": "audio_url", "audio_url": { "url": "https://example.com/audio.wav" } }
                        ]
                    }],
                    "max_completion_tokens": 17
                }),
            ),
            (
                WireFormat::OpenAiResponses,
                json!({
                    "model": "test-model",
                    "input": [{
                        "role": "user",
                        "content": [
                            { "type": "input_text", "text": "inspect" },
                            { "type": "input_image", "image_url": "https://example.com/image.png", "detail": "high" }
                        ]
                    }],
                    "max_output_tokens": 17
                }),
            ),
            (
                WireFormat::AnthropicMessages,
                json!({
                    "model": "test-model",
                    "messages": [{
                        "role": "user",
                        "content": [
                            { "type": "text", "text": "inspect" },
                            {
                                "type": "image",
                                "source": {
                                    "type": "base64",
                                    "media_type": "image/png",
                                    "data": "aGVsbG8="
                                }
                            }
                        ]
                    }],
                    "max_tokens": 17
                }),
            ),
        ];
        let preprocessor = preprocessor();

        for (format, body) in cases {
            let legacy_request = legacy_chat_request(format, body.clone());
            let canonical = decode_wire_request(
                format,
                body.clone(),
                execution_from_chat(&legacy_request).unwrap(),
            )
            .unwrap();

            let legacy = preprocessor
                .preprocess_request(&legacy_request, None)
                .await
                .unwrap()
                .0;
            let direct = preprocessor
                .preprocess_llm_request(&canonical.request, &canonical.execution, None)
                .await
                .unwrap()
                .0;

            assert_eq!(
                serde_json::to_value(direct).unwrap(),
                serde_json::to_value(legacy).unwrap(),
                "direct multimodal preprocessing diverged for {format}"
            );
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
    async fn responses_instructions_and_developer_lowering_matches_current_path() {
        let canonical = assert_responses_direct_lowering_matches_legacy(json!({
            "model": "test-model",
            "instructions": "You are a coding agent.",
            "input": [
                {
                    "type": "message",
                    "role": "developer",
                    "content": [{ "type": "input_text", "text": "Follow safety guidelines." }]
                },
                {
                    "type": "message",
                    "role": "user",
                    "content": [{ "type": "input_text", "text": "What is 2+2?" }]
                }
            ],
            "max_output_tokens": 17
        }))
        .await;

        assert_eq!(canonical.instructions.len(), 1);
        assert_eq!(
            plain_text(&canonical.instructions[0].content, "test").unwrap(),
            "You are a coding agent.\n\nFollow safety guidelines."
        );
        assert_eq!(canonical.messages.len(), 1);
        assert_eq!(canonical.messages[0].role, Role::User);
    }

    #[tokio::test]
    async fn responses_refusal_lowering_matches_current_path() {
        let canonical = assert_responses_direct_lowering_matches_legacy(json!({
            "model": "test-model",
            "input": [
                {
                    "type": "message",
                    "role": "user",
                    "content": [{ "type": "input_text", "text": "try again" }]
                },
                {
                    "type": "message",
                    "role": "assistant",
                    "content": [{ "type": "refusal", "refusal": "I cannot help with that." }]
                },
                {
                    "type": "message",
                    "role": "user",
                    "content": [{ "type": "input_text", "text": "ok different question" }]
                }
            ],
            "max_output_tokens": 17
        }))
        .await;

        assert!(matches!(
            canonical.messages[1].content.as_slice(),
            [ContentBlock::Text { text }] if text == "I cannot help with that."
        ));
    }

    struct CaptureBackend {
        request: Arc<Mutex<Option<PreprocessedRequest>>>,
    }

    #[async_trait]
    impl AsyncEngine<SingleIn<PreprocessedRequest>, ManyOut<Annotated<BackendOutput>>, Error>
        for CaptureBackend
    {
        async fn generate(
            &self,
            request: SingleIn<PreprocessedRequest>,
        ) -> Result<ManyOut<Annotated<BackendOutput>>, Error> {
            let (request, context) = request.transfer(());
            *self.request.lock().unwrap() = Some(request);
            Ok(ResponseStream::new(
                Box::pin(stream::empty()),
                context.context(),
            ))
        }
    }

    #[tokio::test]
    async fn chat_operator_preprocesses_the_attached_llm_request() {
        let chat_request: NvCreateChatCompletionRequest = serde_json::from_value(json!({
            "model": "test-model",
            "messages": [{ "role": "user", "content": "legacy chat prompt" }],
            "max_completion_tokens": 17
        }))
        .unwrap();
        let canonical = decode_wire_request(
            WireFormat::OpenAiResponses,
            json!({
                "model": "test-model",
                "input": "canonical responses prompt",
                "max_output_tokens": 17
            }),
            execution_from_chat(&chat_request).unwrap(),
        )
        .unwrap();
        let expected = preprocessor()
            .preprocess_llm_request(&canonical.request, &canonical.execution, None)
            .await
            .unwrap()
            .0;

        let captured = Arc::new(Mutex::new(None));
        let next: Arc<
            dyn AsyncEngine<SingleIn<PreprocessedRequest>, ManyOut<Annotated<BackendOutput>>, Error>,
        > = Arc::new(CaptureBackend {
            request: captured.clone(),
        });
        let mut request = SingleIn::new(chat_request);
        request.insert(LLM_REQUEST_CONTEXT_KEY, canonical);

        let _stream = Operator::generate(preprocessor().as_ref(), request, next)
            .await
            .unwrap();
        let actual = captured.lock().unwrap().take().unwrap();

        assert_eq!(actual.token_ids, expected.token_ids);
        assert_eq!(actual.extra_args, expected.extra_args);
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
    fn projects_multimodal_content_without_a_chat_request() {
        let request = LlmRequest {
            model: Some("test-model".to_string()),
            messages: vec![Message {
                role: Role::User,
                content: vec![
                    ContentBlock::Text {
                        text: "inspect".to_string(),
                    },
                    ContentBlock::Image {
                        source: ImageSource::Url {
                            url: "https://example.com/image.png".to_string(),
                            detail: Some("high".to_string()),
                        },
                    },
                    ContentBlock::Video {
                        source: MediaSource::Url {
                            url: "https://example.com/video.mp4".to_string(),
                            media_type: Some("video/mp4".to_string()),
                        },
                    },
                    ContentBlock::Audio {
                        source: MediaSource::Base64 {
                            media_type: Some("audio/wav".to_string()),
                            data: "AAAA".to_string(),
                        },
                    },
                ],
            }],
            ..LlmRequest::default()
        };
        let execution = DynamoExecutionOptions {
            image_uuids: vec![Some("image-cache-key".to_string())],
            ..Default::default()
        };

        let view = LlmRequestView::try_new(&request, &execution).unwrap();

        assert_eq!(
            view.messages[0]["content"][1],
            json!({
                "type": "image_url",
                "image_url": {
                    "url": "https://example.com/image.png",
                    "detail": "high"
                },
                "uuid": "image-cache-key"
            })
        );
        assert_eq!(view.media_parts.len(), 3);
        assert_eq!(
            serde_json::to_value(&view.media_parts[1]).unwrap(),
            json!({
                "type": "video_url",
                "video_url": {
                    "url": "https://example.com/video.mp4",
                    "detail": null
                }
            })
        );
        assert_eq!(
            serde_json::to_value(&view.media_parts[2]).unwrap(),
            json!({
                "type": "audio_url",
                "audio_url": { "url": "data:audio/wav;base64,AAAA" }
            })
        );
    }
}
