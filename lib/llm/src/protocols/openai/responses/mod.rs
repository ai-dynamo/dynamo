// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

pub mod stream_converter;

use std::collections::HashMap;

use dynamo_protocols::types::responses::{
    AssistantRole, FunctionCallOutput, FunctionToolCall, IncludeEnum, InputContent, InputItem,
    InputOutputMessageContent, InputParam, InputRole, InputTokenDetails, Instructions, Item,
    MessageItem, OutputItem, OutputMessage, OutputMessageContent, OutputStatus, OutputTextContent,
    OutputTokenDetails, Reasoning, ReasoningItem, Response, ResponseTextParam, ResponseUsage,
    Role as ResponseRole, ServiceTier, Status, SummaryPart, SummaryTextContent,
    TextResponseFormatConfiguration, Tool, ToolChoiceOptions, ToolChoiceParam, Truncation,
};
use dynamo_protocols::types::{
    ChatCompletionMessageToolCall, ChatCompletionNamedToolChoice,
    ChatCompletionRequestAssistantMessage, ChatCompletionRequestAssistantMessageContent,
    ChatCompletionRequestMessage, ChatCompletionRequestMessageContentPartImage,
    ChatCompletionRequestMessageContentPartText, ChatCompletionRequestSystemMessage,
    ChatCompletionRequestSystemMessageContent, ChatCompletionRequestToolMessage,
    ChatCompletionRequestToolMessageContent, ChatCompletionRequestUserMessage,
    ChatCompletionRequestUserMessageContent, ChatCompletionRequestUserMessageContentPart,
    ChatCompletionTool, ChatCompletionToolChoiceOption, ChatCompletionToolType,
    CreateChatCompletionRequest, FunctionName, FunctionObject, FunctionType,
    ImageDetail as ChatImageDetail, ImageUrl, ReasoningContent, ResponseFormat,
    ServiceTier as ChatServiceTier,
};
use dynamo_runtime::protocols::annotated::AnnotationsProvider;
use serde::{Deserialize, Serialize};
use utoipa::ToSchema;
use uuid::Uuid;
use validator::Validate;

use super::chat_completions::{NvCreateChatCompletionRequest, NvCreateChatCompletionResponse};
use super::nvext::{NvExt, NvExtProvider};
use super::{OpenAISamplingOptionsProvider, OpenAIStopConditionsProvider};

#[derive(ToSchema, Serialize, Deserialize, Validate, Debug, Clone)]
pub struct NvCreateResponse {
    /// Flattened CreateResponse fields (model, input, temperature, etc.).
    ///
    /// `CreateResponse` and its `input` chain (`InputParam`, `InputItem`,
    /// `Item`, `MessageItem`, `OutputMessage`, `OutputMessageContent`,
    /// `OutputTextContent`) are Dynamo-owned in `dynamo-protocols`. They mirror
    /// upstream async-openai but accept the relaxed shapes real clients emit
    /// (optional `id` / `status` on assistant messages, optional `annotations`
    /// on `output_text` parts). See `dynamo_protocols::types::responses` for
    /// the full rationale.
    #[serde(flatten)]
    #[schema(value_type = Object)]
    pub inner: dynamo_protocols::types::responses::CreateResponse,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub nvext: Option<NvExt>,
}

#[derive(ToSchema, Serialize, Deserialize, Validate, Debug, Clone)]
pub struct NvResponse {
    /// Flattened Response fields (includes upstream + extended spec fields).
    #[serde(flatten)]
    #[schema(value_type = Object)]
    pub inner: dynamo_protocols::types::responses::Response,

    /// NVIDIA extension field for response metadata (worker IDs, etc.)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub nvext: Option<serde_json::Value>,
}

/// Implements `NvExtProvider` for `NvCreateResponse`,
/// providing access to NVIDIA-specific extensions.
impl NvExtProvider for NvCreateResponse {
    fn nvext(&self) -> Option<&NvExt> {
        self.nvext.as_ref()
    }

    fn raw_prompt(&self) -> Option<String> {
        None
    }
}

/// Implements `AnnotationsProvider` for `NvCreateResponse`,
/// enabling retrieval and management of request annotations.
impl AnnotationsProvider for NvCreateResponse {
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

impl OpenAISamplingOptionsProvider for NvCreateResponse {
    fn get_temperature(&self) -> Option<f32> {
        self.inner.temperature
    }

    fn get_top_p(&self) -> Option<f32> {
        self.inner.top_p
    }

    fn get_frequency_penalty(&self) -> Option<f32> {
        None
    }

    fn get_presence_penalty(&self) -> Option<f32> {
        None
    }

    fn nvext(&self) -> Option<&NvExt> {
        self.nvext.as_ref()
    }

    fn get_seed(&self) -> Option<i64> {
        None
    }

    fn get_n(&self) -> Option<u8> {
        None
    }

    fn get_best_of(&self) -> Option<u8> {
        None
    }
}

impl OpenAIStopConditionsProvider for NvCreateResponse {
    #[allow(deprecated)]
    fn get_max_tokens(&self) -> Option<u32> {
        self.inner.max_output_tokens
    }

    fn get_min_tokens(&self) -> Option<u32> {
        None
    }

    fn get_stop(&self) -> Option<Vec<String>> {
        None
    }

    fn nvext(&self) -> Option<&NvExt> {
        self.nvext.as_ref()
    }
}

// ---------------------------------------------------------------------------
// Responses API -> Chat Completions conversion
// ---------------------------------------------------------------------------

/// Convert a Responses API ImageDetail to the Chat Completions ImageDetail.
/// The responses module re-exports an `ImageDetail` from the upstream async-openai
/// crate which is distinct from `dynamo_protocols::types::ImageDetail` (chat).
/// We bridge via serde to avoid direct cross-crate type dependencies.
fn convert_image_detail_str(detail: &impl serde::Serialize) -> ChatImageDetail {
    match serde_json::to_value(detail)
        .ok()
        .and_then(|v| v.as_str().map(String::from))
        .as_deref()
    {
        Some("low") => ChatImageDetail::Low,
        Some("high") => ChatImageDetail::High,
        _ => ChatImageDetail::Auto,
    }
}

/// Convert a slice of InputContent to ChatCompletionRequestUserMessageContent.
fn convert_input_content_to_user_content(
    content: &[InputContent],
) -> Result<ChatCompletionRequestUserMessageContent, anyhow::Error> {
    // If there's a single InputText, treat as simple text
    if content.len() == 1
        && let InputContent::InputText(t) = &content[0]
    {
        return Ok(ChatCompletionRequestUserMessageContent::Text(
            t.text.clone(),
        ));
    }

    let mut chat_parts = Vec::with_capacity(content.len());
    for part in content {
        match part {
            InputContent::InputText(t) => {
                chat_parts.push(ChatCompletionRequestUserMessageContentPart::Text(
                    ChatCompletionRequestMessageContentPartText {
                        text: t.text.clone(),
                    },
                ));
            }
            InputContent::InputImage(img) => {
                if img.file_id.is_some() && img.image_url.is_none() {
                    return Err(anyhow::anyhow!(
                        "Image input by file_id is not yet supported"
                    ));
                }
                let url_str = img
                    .image_url
                    .as_deref()
                    .ok_or_else(|| anyhow::anyhow!("input_image requires image_url"))?;
                let url = url::Url::parse(url_str)
                    .map_err(|e| anyhow::anyhow!("Invalid image URL '{}': {}", url_str, e))?;
                chat_parts.push(ChatCompletionRequestUserMessageContentPart::ImageUrl(
                    ChatCompletionRequestMessageContentPartImage {
                        image_url: ImageUrl {
                            url,
                            detail: Some(convert_image_detail_str(&img.detail)),
                            uuid: None,
                        },
                    },
                ));
            }
            // TODO: handle InputVideo / InputAudio when upstream adds them
            InputContent::InputFile(_) => {
                return Err(anyhow::anyhow!("File input content is not yet supported"));
            }
        }
    }
    Ok(ChatCompletionRequestUserMessageContent::Array(chat_parts))
}

/// Convert a slice of InputContent to a plain text string (for system/developer/assistant messages).
fn convert_input_content_to_text(content: &[InputContent]) -> String {
    content
        .iter()
        .filter_map(|p| match p {
            InputContent::InputText(t) => Some(t.text.as_str()),
            _ => None,
        })
        .collect::<Vec<_>>()
        .join("")
}

/// Accumulator for consecutive assistant-side items (OutputMessage, FunctionCall,
/// Reasoning, assistant EasyMessage). Chat Completions represents an assistant
/// turn as a single message carrying `content`, `tool_calls`, and
/// `reasoning_content`, so we coalesce adjacent assistant-side Responses input
/// items before emitting.
///
/// `touched` records whether any assistant-side item was seen in the current
/// run. Without it, a standalone assistant message with empty text (or only
/// Refusal content parts that this converter currently strips) would be lost
/// entirely — breaking turn boundaries the model relies on.
#[derive(Default)]
struct PendingAssistant {
    content: Option<String>,
    reasoning_content: Option<String>,
    tool_calls: Vec<ChatCompletionMessageToolCall>,
    touched: bool,
}

impl PendingAssistant {
    fn push_text(&mut self, text: &str) {
        self.touched = true;
        if text.is_empty() {
            return;
        }
        match self.content.as_mut() {
            Some(existing) => existing.push_str(text),
            None => self.content = Some(text.to_string()),
        }
    }

    /// Route prior-turn reasoning summary text into the pending assistant's
    /// `reasoning_content`. Codex and the Agents SDK round-trip `Item::Reasoning`
    /// mid-turn so the model can see its own chain-of-thought as input context.
    fn push_reasoning(&mut self, text: &str) {
        self.touched = true;
        if text.is_empty() {
            return;
        }
        match self.reasoning_content.as_mut() {
            Some(existing) => existing.push_str(text),
            None => self.reasoning_content = Some(text.to_string()),
        }
    }

    fn push_tool_call(&mut self, call: ChatCompletionMessageToolCall) {
        self.touched = true;
        self.tool_calls.push(call);
    }

    fn flush_into(self, out: &mut Vec<ChatCompletionRequestMessage>) {
        if !self.touched {
            return;
        }
        // Content rules:
        //   - real text pushed → emit Some(Text(text))
        //   - pure tool-call turn (no text, has tool_calls) → emit None, matching
        //     Chat Completions spec's nullable-content contract and the converter's
        //     behavior before the coalescing refactor.
        //   - turn-boundary preservation (assistant item seen but no text, no
        //     tool_calls) → emit Some(Text("")) so adjacent user turns aren't
        //     silently merged.
        let content = if self.content.is_some() || self.tool_calls.is_empty() {
            Some(
                self.content
                    .map(ChatCompletionRequestAssistantMessageContent::Text)
                    .unwrap_or_else(|| {
                        ChatCompletionRequestAssistantMessageContent::Text(String::new())
                    }),
            )
        } else {
            None
        };
        out.push(ChatCompletionRequestMessage::Assistant(
            ChatCompletionRequestAssistantMessage {
                content,
                reasoning_content: self.reasoning_content.map(ReasoningContent::Text),
                refusal: None,
                name: None,
                audio: None,
                tool_calls: if self.tool_calls.is_empty() {
                    None
                } else {
                    Some(self.tool_calls)
                },
                #[allow(deprecated)]
                function_call: None,
            },
        ));
    }
}

/// Convert InputParam::Items to a Vec of ChatCompletionRequestMessages.
fn convert_input_items_to_messages(
    items: &[InputItem],
) -> Result<Vec<ChatCompletionRequestMessage>, anyhow::Error> {
    let mut messages = Vec::with_capacity(items.len());
    let mut pending = PendingAssistant::default();

    for item in items {
        match item {
            InputItem::Item(inner_item) => match inner_item {
                Item::Message(msg_item) => match msg_item {
                    MessageItem::Input(msg) => {
                        std::mem::take(&mut pending).flush_into(&mut messages);
                        let chat_msg = match msg.role {
                            InputRole::System | InputRole::Developer => {
                                let text = convert_input_content_to_text(&msg.content);
                                ChatCompletionRequestMessage::System(
                                    ChatCompletionRequestSystemMessage {
                                        content: ChatCompletionRequestSystemMessageContent::Text(
                                            text,
                                        ),
                                        name: None,
                                    },
                                )
                            }
                            InputRole::User => {
                                let content = convert_input_content_to_user_content(&msg.content)?;
                                ChatCompletionRequestMessage::User(
                                    ChatCompletionRequestUserMessage {
                                        content,
                                        name: None,
                                    },
                                )
                            }
                        };
                        messages.push(chat_msg);
                    }
                    MessageItem::Output(out_msg) => {
                        let text = out_msg
                            .content
                            .iter()
                            .filter_map(|c| match c {
                                InputOutputMessageContent::OutputText(t) => Some(t.text.as_str()),
                                _ => None,
                            })
                            .collect::<Vec<_>>()
                            .join("");
                        pending.push_text(&text);
                    }
                },
                Item::FunctionCall(fc) => {
                    pending.push_tool_call(ChatCompletionMessageToolCall {
                        id: fc.call_id.clone(),
                        r#type: FunctionType::Function,
                        function: dynamo_protocols::types::FunctionCall {
                            name: fc.name.clone(),
                            arguments: fc.arguments.clone(),
                        },
                    });
                }
                Item::FunctionCallOutput(fco) => {
                    std::mem::take(&mut pending).flush_into(&mut messages);
                    let output_text = match &fco.output {
                        FunctionCallOutput::Text(text) => text.clone(),
                        FunctionCallOutput::Content(parts) => convert_input_content_to_text(parts),
                    };
                    messages.push(ChatCompletionRequestMessage::Tool(
                        ChatCompletionRequestToolMessage {
                            content: ChatCompletionRequestToolMessageContent::Text(output_text),
                            tool_call_id: fco.call_id.clone(),
                        },
                    ));
                }
                Item::Reasoning(r) => {
                    let text = r
                        .summary
                        .iter()
                        .map(|SummaryPart::SummaryText(t)| t.text.as_str())
                        .collect::<Vec<_>>()
                        .join("");
                    pending.push_reasoning(&text);
                }
                other => {
                    // Unknown / unsupported variants (ComputerCall, WebSearchCall,
                    // tool-output items other than FunctionCallOutput, etc.). We do
                    // not have a faithful Chat Completions mapping, but silently
                    // consuming them without flushing would let a following
                    // FunctionCall coalesce with tool_calls from a different
                    // semantic turn. Flush first, then skip.
                    tracing::debug!(
                        "Skipping unsupported input item type during conversion: {:?}",
                        std::mem::discriminant(other)
                    );
                    std::mem::take(&mut pending).flush_into(&mut messages);
                }
            },
            InputItem::EasyMessage(easy) => {
                let content_text = match &easy.content {
                    dynamo_protocols::types::responses::EasyInputContent::Text(text) => {
                        text.clone()
                    }
                    dynamo_protocols::types::responses::EasyInputContent::ContentList(parts) => {
                        convert_input_content_to_text(parts)
                    }
                };
                match easy.role {
                    ResponseRole::System | ResponseRole::Developer => {
                        std::mem::take(&mut pending).flush_into(&mut messages);
                        messages.push(ChatCompletionRequestMessage::System(
                            ChatCompletionRequestSystemMessage {
                                content: ChatCompletionRequestSystemMessageContent::Text(
                                    content_text,
                                ),
                                name: None,
                            },
                        ));
                    }
                    ResponseRole::User => {
                        std::mem::take(&mut pending).flush_into(&mut messages);
                        messages.push(ChatCompletionRequestMessage::User(
                            ChatCompletionRequestUserMessage {
                                content: ChatCompletionRequestUserMessageContent::Text(
                                    content_text,
                                ),
                                name: None,
                            },
                        ));
                    }
                    ResponseRole::Assistant => {
                        pending.push_text(&content_text);
                    }
                }
            }
            InputItem::ItemReference(_) => {
                // Skip item references
            }
        }
    }

    pending.flush_into(&mut messages);

    Ok(messages)
}

/// Convert Responses API Tool to ChatCompletionTool.
fn convert_tools(tools: &[Tool]) -> Vec<ChatCompletionTool> {
    tools
        .iter()
        .filter_map(|tool| match tool {
            Tool::Function(f) => Some(ChatCompletionTool {
                r#type: ChatCompletionToolType::Function,
                function: FunctionObject {
                    name: f.name.clone(),
                    description: f.description.clone(),
                    parameters: f.parameters.clone(),
                    strict: f.strict,
                },
            }),
            _ => None, // Only function tools are forwarded to chat completions
        })
        .collect()
}

/// Convert Responses API ToolChoiceParam to ChatCompletionToolChoiceOption.
fn convert_tool_choice(tc: &ToolChoiceParam) -> ChatCompletionToolChoiceOption {
    match tc {
        ToolChoiceParam::Mode(mode) => match mode {
            ToolChoiceOptions::None => ChatCompletionToolChoiceOption::None,
            ToolChoiceOptions::Auto => ChatCompletionToolChoiceOption::Auto,
            ToolChoiceOptions::Required => ChatCompletionToolChoiceOption::Required,
        },
        ToolChoiceParam::Function(f) => {
            ChatCompletionToolChoiceOption::Named(ChatCompletionNamedToolChoice {
                r#type: ChatCompletionToolType::Function,
                function: FunctionName {
                    name: f.name.clone(),
                },
            })
        }
        ToolChoiceParam::Hosted(_) => {
            // Hosted tools are not forwarded to chat completions
            ChatCompletionToolChoiceOption::Auto
        }
        _ => {
            // Other tool choice types (AllowedTools, Mcp, Custom, etc.) default to auto
            ChatCompletionToolChoiceOption::Auto
        }
    }
}

/// Convert Responses API `text.format` to Chat Completions `response_format`.
fn convert_text_format(text: &ResponseTextParam) -> Option<ResponseFormat> {
    match &text.format {
        TextResponseFormatConfiguration::Text => None,
        TextResponseFormatConfiguration::JsonObject => Some(ResponseFormat::JsonObject),
        TextResponseFormatConfiguration::JsonSchema(s) => Some(ResponseFormat::JsonSchema {
            json_schema: s.clone(),
        }),
    }
}

/// Convert Responses API `ServiceTier` to Chat Completions `ServiceTier`.
/// These are structurally identical enums in different modules.
fn convert_service_tier(tier: &ServiceTier) -> ChatServiceTier {
    match tier {
        ServiceTier::Auto => ChatServiceTier::Auto,
        ServiceTier::Default => ChatServiceTier::Default,
        ServiceTier::Flex => ChatServiceTier::Flex,
        ServiceTier::Scale => ChatServiceTier::Scale,
        ServiceTier::Priority => ChatServiceTier::Priority,
    }
}

impl TryFrom<NvCreateResponse> for NvCreateChatCompletionRequest {
    type Error = anyhow::Error;

    fn try_from(resp: NvCreateResponse) -> Result<Self, Self::Error> {
        let mut messages = Vec::new();

        // Prepend instructions as system message if present
        if let Some(instructions) = &resp.inner.instructions {
            messages.push(ChatCompletionRequestMessage::System(
                ChatCompletionRequestSystemMessage {
                    content: ChatCompletionRequestSystemMessageContent::Text(instructions.clone()),
                    name: None,
                },
            ));
        }

        // Convert input to messages
        match &resp.inner.input {
            InputParam::Text(text) => {
                messages.push(ChatCompletionRequestMessage::User(
                    ChatCompletionRequestUserMessage {
                        content: ChatCompletionRequestUserMessageContent::Text(text.clone()),
                        name: None,
                    },
                ));
            }
            InputParam::Items(items) => {
                let item_messages = convert_input_items_to_messages(items)?;
                messages.extend(item_messages);
            }
        }

        let top_logprobs = convert_top_logprobs(resp.inner.top_logprobs);

        // Convert tools if present
        let tools = resp
            .inner
            .tools
            .as_ref()
            .map(|t| convert_tools(t))
            .filter(|t: &Vec<_>| !t.is_empty());

        // Convert tool_choice if present
        let tool_choice = resp.inner.tool_choice.as_ref().map(convert_tool_choice);

        // Determine stream setting: respect caller's preference, default to true for aggregation
        let stream = resp.inner.stream.or(Some(true));

        // Map reasoning.effort to reasoning_effort
        let reasoning_effort = resp.inner.reasoning.as_ref().and_then(|r| r.effort.clone());

        // Map text.format to response_format
        let response_format = resp.inner.text.as_ref().and_then(convert_text_format);

        // Map service_tier
        let service_tier = resp.inner.service_tier.as_ref().map(convert_service_tier);

        Ok(NvCreateChatCompletionRequest {
            inner: CreateChatCompletionRequest {
                messages,
                model: resp.inner.model.unwrap_or_default(),
                temperature: resp.inner.temperature,
                top_p: resp.inner.top_p,
                max_completion_tokens: resp.inner.max_output_tokens,
                store: resp.inner.store,
                parallel_tool_calls: resp.inner.parallel_tool_calls,
                top_logprobs,
                metadata: resp
                    .inner
                    .metadata
                    .map(|m| serde_json::to_value(m).unwrap_or_default()),
                stream,
                tools,
                tool_choice,
                reasoning_effort,
                response_format,
                service_tier,
                ..Default::default()
            },
            common: Default::default(),
            nvext: resp.nvext,
            chat_template_args: None,
            media_io_kwargs: None,
            unsupported_fields: Default::default(),
        })
    }
}

fn convert_top_logprobs(input: Option<u8>) -> Option<u8> {
    input.map(|x| x.min(20))
}

/// Parse `<tool_call>` blocks from model text output.
/// Returns a list of (name, arguments_json) tuples.
/// Returns an empty vec immediately if no `<tool_call>` tag is present.
fn parse_tool_call_text(text: &str) -> Vec<(String, String)> {
    if !text.contains("<tool_call>") {
        return Vec::new();
    }
    let mut results = Vec::new();
    let mut search_start = 0;
    while let Some(start) = text[search_start..].find("<tool_call>") {
        let abs_start = search_start + start + "<tool_call>".len();
        if let Some(end) = text[abs_start..].find("</tool_call>") {
            let block = text[abs_start..abs_start + end].trim();
            if let Ok(parsed) = serde_json::from_str::<serde_json::Value>(block) {
                let name = parsed
                    .get("name")
                    .and_then(|v| v.as_str())
                    .unwrap_or("")
                    .to_string();
                let arguments = if let Some(args) = parsed.get("arguments") {
                    if args.is_string() {
                        args.as_str().unwrap_or("{}").to_string()
                    } else {
                        serde_json::to_string(args).unwrap_or_else(|_| "{}".to_string())
                    }
                } else {
                    "{}".to_string()
                };
                if !name.is_empty() {
                    results.push((name, arguments));
                }
            }
            search_start = abs_start + end + "</tool_call>".len();
        } else {
            break;
        }
    }
    results
}

/// Strip `<tool_call>...</tool_call>` blocks and any `<think>...</think>` blocks from text.
/// Returns the original string (no allocation) if no tags are present.
fn strip_tool_call_text(text: &str) -> std::borrow::Cow<'_, str> {
    let has_tool = text.contains("<tool_call>");
    let has_think = text.contains("<think>");
    if !has_tool && !has_think {
        return std::borrow::Cow::Borrowed(text);
    }

    fn strip_tag(input: &mut String, open: &str, close: &str) {
        while let Some(start) = input.find(open) {
            if let Some(end_offset) = input[start..].find(close) {
                input.replace_range(start..start + end_offset + close.len(), "");
            } else {
                input.truncate(start);
                break;
            }
        }
    }

    let mut result = text.to_string();
    if has_tool {
        strip_tag(&mut result, "<tool_call>", "</tool_call>");
    }
    if has_think {
        strip_tag(&mut result, "<think>", "</think>");
    }
    std::borrow::Cow::Owned(result)
}

// ---------------------------------------------------------------------------
// Chat Completions -> Responses API response conversion
// ---------------------------------------------------------------------------

/// Request parameters to echo back in Response objects.
/// Extracted from the incoming CreateResponse request so that
/// response objects reflect actual request values.
#[derive(Clone, Debug, Default)]
pub struct ResponseParams {
    pub model: Option<String>,
    pub temperature: Option<f32>,
    pub top_p: Option<f32>,
    pub max_output_tokens: Option<u32>,
    pub parallel_tool_calls: Option<bool>,
    pub store: Option<bool>,
    pub tools: Option<Vec<Tool>>,
    pub tool_choice: Option<ToolChoiceParam>,
    pub instructions: Option<String>,
    pub reasoning: Option<Reasoning>,
    pub text: Option<ResponseTextParam>,
    pub service_tier: Option<ServiceTier>,
    pub include: Option<Vec<IncludeEnum>>,
    pub truncation: Option<Truncation>,
}

/// Normalize tools so that `FunctionTool.strict` is always set.
/// The upstream type uses `skip_serializing_if = "Option::is_none"` on `strict`,
/// so `None` causes the field to be omitted during JSON serialization.
/// Schema validators (Zod, etc.) expect `strict` to always be present.
/// OpenAI defaults `strict` to `true`.
pub(super) fn normalize_tools(tools: Vec<Tool>) -> Vec<Tool> {
    tools
        .into_iter()
        .map(|tool| match tool {
            Tool::Function(mut ft) => {
                if ft.strict.is_none() {
                    ft.strict = Some(true);
                }
                Tool::Function(ft)
            }
            other => other,
        })
        .collect()
}

/// Build an assistant text message output item.
fn make_text_message(id: String, text: String) -> OutputItem {
    OutputItem::Message(OutputMessage {
        id,
        role: AssistantRole::Assistant,
        status: OutputStatus::Completed,
        phase: None,
        content: vec![OutputMessageContent::OutputText(OutputTextContent {
            text,
            annotations: vec![],
            logprobs: Some(vec![]),
        })],
    })
}

/// Build a function call output item with generated IDs.
fn make_function_call(name: String, arguments: String) -> OutputItem {
    OutputItem::FunctionCall(FunctionToolCall {
        arguments,
        call_id: format!("call_{}", Uuid::new_v4().simple()),
        namespace: None,
        name,
        id: Some(format!("fc_{}", Uuid::new_v4().simple())),
        status: Some(OutputStatus::Completed),
    })
}

/// Convert a ChatCompletion response into a Responses API response object,
/// echoing back the actual request parameters from `params`.
pub fn chat_completion_to_response(
    nv_resp: NvCreateChatCompletionResponse,
    params: &ResponseParams,
    api_context: Option<&crate::protocols::unified::ResponsesContext>,
) -> Result<NvResponse, anyhow::Error> {
    let nvext = nv_resp.nvext.clone();
    let chat_resp = nv_resp.inner;
    let message_id = format!("msg_{}", Uuid::new_v4().simple());
    let response_id = format!("resp_{}", Uuid::new_v4().simple());

    let choice = chat_resp.choices.into_iter().next();
    let mut output = Vec::new();

    if let Some(choice) = choice {
        // Handle structured tool calls
        if let Some(tool_calls) = choice.message.tool_calls {
            for tc in &tool_calls {
                output.push(OutputItem::FunctionCall(FunctionToolCall {
                    arguments: tc.function.arguments.clone(),
                    call_id: tc.id.clone(),
                    namespace: None,
                    name: tc.function.name.clone(),
                    id: Some(format!("fc_{}", Uuid::new_v4().simple())),
                    status: Some(OutputStatus::Completed),
                }));
            }
        }

        // Map reasoning_content to a Reasoning output item
        if let Some(reasoning_text) = choice.message.reasoning_content
            && !reasoning_text.is_empty()
        {
            output.push(OutputItem::Reasoning(ReasoningItem {
                id: format!("rs_{}", Uuid::new_v4().simple()),
                summary: vec![SummaryPart::SummaryText(SummaryTextContent {
                    text: reasoning_text,
                })],
                content: None,
                encrypted_content: None,
                status: Some(OutputStatus::Completed),
            }));
        }

        // Handle text content -- also parse <tool_call> blocks from models
        // that emit tool calls as text (e.g. Qwen3)
        let content_text = match choice.message.content {
            Some(dynamo_protocols::types::ChatCompletionMessageContent::Text(text)) => Some(text),
            Some(dynamo_protocols::types::ChatCompletionMessageContent::Parts(_)) => {
                tracing::warn!(
                    "Multimodal content in responses API not yet supported, using placeholder"
                );
                Some("[multimodal content]".to_string())
            }
            None => None,
        };
        if let Some(content_text) = content_text
            && !content_text.is_empty()
        {
            let parsed_calls = parse_tool_call_text(&content_text);
            if !parsed_calls.is_empty() {
                for (name, arguments) in parsed_calls {
                    output.push(make_function_call(name, arguments));
                }
                let remaining = strip_tool_call_text(&content_text);
                if !remaining.trim().is_empty() {
                    output.push(make_text_message(
                        message_id.clone(),
                        remaining.into_owned(),
                    ));
                }
            } else {
                output.push(make_text_message(message_id.clone(), content_text));
            }
        }

        if output.is_empty() {
            output.push(make_text_message(message_id, String::new()));
        }
    } else {
        tracing::warn!("No choices in chat completion response, using empty content");
        output.push(make_text_message(message_id, String::new()));
    }

    // Apply `include` filtering: strip logprobs from output text unless
    // the caller explicitly requested them via `message.output_text.logprobs`.
    let keep_logprobs = params
        .include
        .as_ref()
        .is_some_and(|inc| inc.contains(&IncludeEnum::MessageOutputTextLogprobs));
    if !keep_logprobs {
        for item in &mut output {
            if let OutputItem::Message(msg) = item {
                for content in &mut msg.content {
                    if let OutputMessageContent::OutputText(text) = content {
                        text.logprobs = None;
                    }
                }
            }
        }
    }

    let created_at = chat_resp.created as u64;
    let response = Response {
        id: response_id,
        object: "response".to_string(),
        created_at,
        completed_at: Some(created_at),
        model: if chat_resp.model == "unknown" {
            params.model.clone().unwrap_or(chat_resp.model)
        } else {
            chat_resp.model
        },
        status: Status::Completed,
        output,
        // Spec-required defaults (OpenResponses requires these as non-null)
        background: Some(false),
        metadata: Some(HashMap::new()),
        parallel_tool_calls: params.parallel_tool_calls.or(Some(true)),
        temperature: params.temperature.or(Some(1.0)),
        text: Some(params.text.clone().unwrap_or(ResponseTextParam {
            format: TextResponseFormatConfiguration::Text,
            verbosity: None,
        })),
        tool_choice: params
            .tool_choice
            .clone()
            .or(Some(ToolChoiceParam::Mode(ToolChoiceOptions::Auto))),
        tools: Some(
            params
                .tools
                .clone()
                .map(normalize_tools)
                .unwrap_or_default(),
        ),
        top_p: params.top_p.or(Some(1.0)),
        truncation: Some(params.truncation.unwrap_or(Truncation::Disabled)),
        // Nullable but required to be present (null is valid)
        billing: None,
        conversation: None,
        error: None,
        incomplete_details: None,
        instructions: params.instructions.clone().map(Instructions::Text),
        max_output_tokens: params.max_output_tokens,
        previous_response_id: api_context.and_then(|ctx| ctx.previous_response_id.clone()),
        prompt: None,
        prompt_cache_key: None,
        prompt_cache_retention: None,
        reasoning: params.reasoning.clone(),
        safety_identifier: None,
        service_tier: Some(params.service_tier.unwrap_or(ServiceTier::Auto)),
        top_logprobs: Some(0),
        usage: chat_resp.usage.map(|u| ResponseUsage {
            input_tokens: u.prompt_tokens,
            input_tokens_details: InputTokenDetails {
                cached_tokens: u
                    .prompt_tokens_details
                    .map(|d| d.cached_tokens.unwrap_or(0))
                    .unwrap_or(0),
            },
            output_tokens: u.completion_tokens,
            output_tokens_details: OutputTokenDetails {
                reasoning_tokens: u
                    .completion_tokens_details
                    .map(|d| d.reasoning_tokens.unwrap_or(0))
                    .unwrap_or(0),
            },
            total_tokens: u.total_tokens,
        }),
    };

    Ok(NvResponse {
        inner: response,
        nvext,
    })
}

#[cfg(test)]
mod tests {
    use dynamo_protocols::types::responses::{
        CreateResponse, FunctionCallOutput, FunctionCallOutputItemParam, FunctionTool,
        FunctionToolCall, InputContent, InputImageContent, InputItem, InputMessage,
        InputOutputMessage, InputOutputMessageContent, InputOutputTextContent, InputParam,
        InputRole, InputTextContent, Item, MessageItem, Tool,
    };
    use dynamo_protocols::types::{
        ChatCompletionRequestMessage, ChatCompletionRequestUserMessageContent,
    };

    use super::*;
    use crate::types::openai::chat_completions::NvCreateChatCompletionResponse;

    fn make_response_with_input(text: &str) -> NvCreateResponse {
        NvCreateResponse {
            inner: CreateResponse {
                input: InputParam::Text(text.into()),
                model: Some("test-model".into()),
                max_output_tokens: Some(1024),
                temperature: Some(0.5),
                top_p: Some(0.9),
                top_logprobs: Some(15),
                ..Default::default()
            },
            nvext: Some(NvExt {
                annotations: Some(vec!["debug".into(), "trace".into()]),
                ..Default::default()
            }),
        }
    }

    #[test]
    fn test_annotations_trait_behavior() {
        let req = make_response_with_input("hello");
        assert_eq!(
            req.annotations(),
            Some(vec!["debug".to_string(), "trace".to_string()])
        );
        assert!(req.has_annotation("debug"));
        assert!(req.has_annotation("trace"));
        assert!(!req.has_annotation("missing"));
    }

    #[test]
    fn test_openai_sampling_trait_behavior() {
        let req = make_response_with_input("hello");
        assert_eq!(req.get_temperature(), Some(0.5));
        assert_eq!(req.get_top_p(), Some(0.9));
        assert_eq!(req.get_frequency_penalty(), None);
        assert_eq!(req.get_presence_penalty(), None);
    }

    #[test]
    fn test_openai_stop_conditions_trait_behavior() {
        let req = make_response_with_input("hello");
        assert_eq!(req.get_max_tokens(), Some(1024));
        assert_eq!(req.get_min_tokens(), None);
        assert_eq!(req.get_stop(), None);
    }

    #[test]
    fn test_into_nvcreate_chat_completion_request() {
        let nv_req: NvCreateChatCompletionRequest =
            make_response_with_input("hi there").try_into().unwrap();

        assert_eq!(nv_req.inner.model, "test-model");
        assert_eq!(nv_req.inner.temperature, Some(0.5));
        assert_eq!(nv_req.inner.top_p, Some(0.9));
        assert_eq!(nv_req.inner.max_completion_tokens, Some(1024));
        assert_eq!(nv_req.inner.top_logprobs, Some(15));
        assert_eq!(nv_req.inner.stream, Some(true));

        let messages = &nv_req.inner.messages;
        assert_eq!(messages.len(), 1);
        match &messages[0] {
            ChatCompletionRequestMessage::User(user_msg) => match &user_msg.content {
                ChatCompletionRequestUserMessageContent::Text(t) => {
                    assert_eq!(t, "hi there");
                }
                _ => panic!("unexpected user content type"),
            },
            _ => panic!("expected user message"),
        }
    }

    #[test]
    fn test_store_mapped_to_chat_completion_request() {
        let mut req = make_response_with_input("audit me");
        req.inner.store = Some(true);

        let nv_req: NvCreateChatCompletionRequest = req.try_into().unwrap();
        assert_eq!(nv_req.inner.store, Some(true));
    }

    #[test]
    fn test_instructions_prepended_as_system_message() {
        let req = NvCreateResponse {
            inner: CreateResponse {
                input: InputParam::Text("hello".into()),
                model: Some("test-model".into()),
                instructions: Some("You are a helpful assistant.".into()),
                ..Default::default()
            },
            nvext: None,
        };

        let chat_req: NvCreateChatCompletionRequest = req.try_into().unwrap();
        let messages = &chat_req.inner.messages;
        assert_eq!(messages.len(), 2);

        match &messages[0] {
            ChatCompletionRequestMessage::System(sys) => match &sys.content {
                ChatCompletionRequestSystemMessageContent::Text(t) => {
                    assert_eq!(t, "You are a helpful assistant.");
                }
                _ => panic!("expected text content"),
            },
            _ => panic!("expected system message first"),
        }
    }

    #[test]
    fn test_input_items_multi_turn() {
        let req = NvCreateResponse {
            inner: CreateResponse {
                input: InputParam::Items(vec![
                    InputItem::Item(Item::Message(MessageItem::Input(InputMessage {
                        content: vec![InputContent::InputText(InputTextContent {
                            text: "Be concise.".into(),
                        })],
                        role: InputRole::System,
                        status: None,
                    }))),
                    InputItem::Item(Item::Message(MessageItem::Input(InputMessage {
                        content: vec![InputContent::InputText(InputTextContent {
                            text: "What is 2+2?".into(),
                        })],
                        role: InputRole::User,
                        status: None,
                    }))),
                    InputItem::Item(Item::Message(MessageItem::Output(InputOutputMessage {
                        id: Some("msg_1".into()),
                        role: AssistantRole::Assistant,
                        status: Some(OutputStatus::Completed),
                        phase: None,
                        content: vec![InputOutputMessageContent::OutputText(
                            InputOutputTextContent {
                                text: "4".into(),
                                annotations: vec![],
                                logprobs: None,
                            },
                        )],
                    }))),
                    InputItem::Item(Item::Message(MessageItem::Input(InputMessage {
                        content: vec![InputContent::InputText(InputTextContent {
                            text: "And 3+3?".into(),
                        })],
                        role: InputRole::User,
                        status: None,
                    }))),
                ]),
                model: Some("test-model".into()),
                ..Default::default()
            },
            nvext: None,
        };

        let chat_req: NvCreateChatCompletionRequest = req.try_into().unwrap();
        let messages = &chat_req.inner.messages;
        assert_eq!(messages.len(), 4);
        assert!(matches!(
            messages[0],
            ChatCompletionRequestMessage::System(_)
        ));
        assert!(matches!(messages[1], ChatCompletionRequestMessage::User(_)));
        assert!(matches!(
            messages[2],
            ChatCompletionRequestMessage::Assistant(_)
        ));
        assert!(matches!(messages[3], ChatCompletionRequestMessage::User(_)));
    }

    #[test]
    fn test_input_items_with_image() {
        let req = NvCreateResponse {
            inner: CreateResponse {
                input: InputParam::Items(vec![InputItem::Item(Item::Message(MessageItem::Input(
                    InputMessage {
                        content: vec![
                            InputContent::InputText(InputTextContent {
                                text: "What is in this image?".into(),
                            }),
                            InputContent::InputImage(InputImageContent {
                                detail: Default::default(), // ImageDetail::Auto
                                file_id: None,
                                image_url: Some("https://example.com/cat.jpg".into()),
                            }),
                        ],
                        role: InputRole::User,
                        status: None,
                    },
                )))]),
                model: Some("test-model".into()),
                ..Default::default()
            },
            nvext: None,
        };

        let chat_req: NvCreateChatCompletionRequest = req.try_into().unwrap();
        let messages = &chat_req.inner.messages;
        assert_eq!(messages.len(), 1);
        match &messages[0] {
            ChatCompletionRequestMessage::User(u) => match &u.content {
                ChatCompletionRequestUserMessageContent::Array(parts) => {
                    assert_eq!(parts.len(), 2);
                }
                _ => panic!("expected array content"),
            },
            _ => panic!("expected user message"),
        }
    }

    #[test]
    fn test_function_call_input_items() {
        let req = NvCreateResponse {
            inner: CreateResponse {
                input: InputParam::Items(vec![
                    InputItem::Item(Item::Message(MessageItem::Input(InputMessage {
                        content: vec![InputContent::InputText(InputTextContent {
                            text: "What's the weather?".into(),
                        })],
                        role: InputRole::User,
                        status: None,
                    }))),
                    InputItem::Item(Item::FunctionCall(FunctionToolCall {
                        arguments: r#"{"location":"SF"}"#.into(),
                        call_id: "call_123".into(),
                        namespace: None,
                        name: "get_weather".into(),
                        id: None,
                        status: None,
                    })),
                    InputItem::Item(Item::FunctionCallOutput(FunctionCallOutputItemParam {
                        call_id: "call_123".into(),
                        output: FunctionCallOutput::Text(r#"{"temp":"72F"}"#.into()),
                        id: None,
                        status: None,
                    })),
                ]),
                model: Some("test-model".into()),
                ..Default::default()
            },
            nvext: None,
        };

        let chat_req: NvCreateChatCompletionRequest = req.try_into().unwrap();
        let messages = &chat_req.inner.messages;
        assert_eq!(messages.len(), 3);
        assert!(matches!(messages[0], ChatCompletionRequestMessage::User(_)));
        assert!(matches!(
            messages[1],
            ChatCompletionRequestMessage::Assistant(_)
        ));
        assert!(matches!(messages[2], ChatCompletionRequestMessage::Tool(_)));
    }

    #[test]
    fn test_function_call_with_interstitial_assistant_message_is_coalesced() {
        // Regression: prior turn was `function_call` + assistant text + `function_call_output`.
        // The converter must emit a SINGLE assistant chat message carrying both `content`
        // and `tool_calls`, otherwise chat templates that require a tool message to
        // immediately follow its assistant tool_call (e.g. MiniMax) will reject the input.
        let req = NvCreateResponse {
            inner: CreateResponse {
                input: InputParam::Items(vec![
                    InputItem::Item(Item::Message(MessageItem::Input(InputMessage {
                        content: vec![InputContent::InputText(InputTextContent {
                            text: "What's the weather?".into(),
                        })],
                        role: InputRole::User,
                        status: None,
                    }))),
                    InputItem::Item(Item::FunctionCall(FunctionToolCall {
                        arguments: r#"{"location":"SF"}"#.into(),
                        call_id: "call_123".into(),
                        namespace: None,
                        name: "get_weather".into(),
                        id: None,
                        status: None,
                    })),
                    InputItem::Item(Item::Message(MessageItem::Output(InputOutputMessage {
                        id: Some("msg_interstitial".into()),
                        role: AssistantRole::Assistant,
                        status: Some(OutputStatus::Completed),
                        phase: None,
                        content: vec![InputOutputMessageContent::OutputText(
                            InputOutputTextContent {
                                text: "\n\n".into(),
                                annotations: vec![],
                                logprobs: None,
                            },
                        )],
                    }))),
                    InputItem::Item(Item::FunctionCallOutput(FunctionCallOutputItemParam {
                        call_id: "call_123".into(),
                        output: FunctionCallOutput::Text(r#"{"temp":"72F"}"#.into()),
                        id: None,
                        status: None,
                    })),
                ]),
                model: Some("test-model".into()),
                ..Default::default()
            },
            nvext: None,
        };

        let chat_req: NvCreateChatCompletionRequest = req.try_into().unwrap();
        let messages = &chat_req.inner.messages;
        assert_eq!(
            messages.len(),
            3,
            "expected coalesced [user, assistant, tool]"
        );
        assert!(matches!(messages[0], ChatCompletionRequestMessage::User(_)));
        match &messages[1] {
            ChatCompletionRequestMessage::Assistant(a) => {
                let tool_calls = a.tool_calls.as_ref().expect("tool_calls must be present");
                assert_eq!(tool_calls.len(), 1);
                assert_eq!(tool_calls[0].id, "call_123");
                assert_eq!(tool_calls[0].function.name, "get_weather");
                match a
                    .content
                    .as_ref()
                    .expect("content must carry interstitial text")
                {
                    ChatCompletionRequestAssistantMessageContent::Text(t) => {
                        assert_eq!(t, "\n\n");
                    }
                    _ => panic!("expected text content"),
                }
            }
            _ => panic!("expected a single merged assistant message at index 1"),
        }
        assert!(matches!(messages[2], ChatCompletionRequestMessage::Tool(_)));
    }

    #[test]
    fn test_easy_message_assistant_coalesced_with_adjacent_function_call() {
        // The same coalescing rule applies to EasyInputMessage shape (string content,
        // role=assistant, no `type:"message"` discriminator).
        use dynamo_protocols::types::responses::{
            EasyInputContent, EasyInputMessage, Role as ResponseRole,
        };

        let req = NvCreateResponse {
            inner: CreateResponse {
                input: InputParam::Items(vec![
                    InputItem::EasyMessage(EasyInputMessage {
                        role: ResponseRole::User,
                        content: EasyInputContent::Text("x".into()),
                        ..Default::default()
                    }),
                    InputItem::Item(Item::FunctionCall(FunctionToolCall {
                        arguments: "{}".into(),
                        call_id: "c".into(),
                        namespace: None,
                        name: "f".into(),
                        id: None,
                        status: None,
                    })),
                    InputItem::EasyMessage(EasyInputMessage {
                        role: ResponseRole::Assistant,
                        content: EasyInputContent::Text("".into()),
                        ..Default::default()
                    }),
                    InputItem::Item(Item::FunctionCallOutput(FunctionCallOutputItemParam {
                        call_id: "c".into(),
                        output: FunctionCallOutput::Text("x".into()),
                        id: None,
                        status: None,
                    })),
                ]),
                model: Some("test-model".into()),
                ..Default::default()
            },
            nvext: None,
        };

        let chat_req: NvCreateChatCompletionRequest = req.try_into().unwrap();
        let messages = &chat_req.inner.messages;
        assert_eq!(messages.len(), 3);
        match &messages[1] {
            ChatCompletionRequestMessage::Assistant(a) => {
                assert!(a.tool_calls.is_some());
                assert_eq!(a.tool_calls.as_ref().unwrap().len(), 1);
            }
            _ => panic!("expected merged assistant message"),
        }
        assert!(matches!(messages[2], ChatCompletionRequestMessage::Tool(_)));
    }

    #[test]
    fn test_standalone_assistant_message_with_empty_content_preserves_turn() {
        // A prior assistant turn that produced no text (empty content or
        // refusal-only parts the converter strips) must still emit an assistant
        // message. Otherwise adjacent user turns get silently merged, which
        // breaks strict-alternation chat templates and distorts the context
        // the model sees.
        let req = NvCreateResponse {
            inner: CreateResponse {
                input: InputParam::Items(vec![
                    InputItem::Item(Item::Message(MessageItem::Input(InputMessage {
                        content: vec![InputContent::InputText(InputTextContent {
                            text: "first question".into(),
                        })],
                        role: InputRole::User,
                        status: None,
                    }))),
                    InputItem::Item(Item::Message(MessageItem::Output(InputOutputMessage {
                        id: None,
                        role: AssistantRole::Assistant,
                        status: None,
                        phase: None,
                        content: vec![InputOutputMessageContent::OutputText(
                            InputOutputTextContent {
                                text: "".into(),
                                annotations: vec![],
                                logprobs: None,
                            },
                        )],
                    }))),
                    InputItem::Item(Item::Message(MessageItem::Input(InputMessage {
                        content: vec![InputContent::InputText(InputTextContent {
                            text: "second question".into(),
                        })],
                        role: InputRole::User,
                        status: None,
                    }))),
                ]),
                model: Some("test-model".into()),
                ..Default::default()
            },
            nvext: None,
        };

        let chat_req: NvCreateChatCompletionRequest = req.try_into().unwrap();
        let messages = &chat_req.inner.messages;
        assert_eq!(
            messages.len(),
            3,
            "empty assistant turn must not be silently dropped"
        );
        assert!(matches!(messages[0], ChatCompletionRequestMessage::User(_)));
        match &messages[1] {
            ChatCompletionRequestMessage::Assistant(a) => {
                assert!(a.tool_calls.is_none());
                match a.content.as_ref().expect("empty turn still emits content") {
                    ChatCompletionRequestAssistantMessageContent::Text(t) => {
                        assert_eq!(t, "");
                    }
                    _ => panic!("expected text content"),
                }
            }
            _ => panic!("expected assistant turn boundary preserved"),
        }
        assert!(matches!(messages[2], ChatCompletionRequestMessage::User(_)));
    }

    #[test]
    fn test_easy_assistant_message_with_empty_content_preserves_turn() {
        // Same turn-boundary preservation applies to EasyInputMessage shape.
        use dynamo_protocols::types::responses::{
            EasyInputContent, EasyInputMessage, Role as ResponseRole,
        };

        let req = NvCreateResponse {
            inner: CreateResponse {
                input: InputParam::Items(vec![
                    InputItem::EasyMessage(EasyInputMessage {
                        role: ResponseRole::User,
                        content: EasyInputContent::Text("first".into()),
                        ..Default::default()
                    }),
                    InputItem::EasyMessage(EasyInputMessage {
                        role: ResponseRole::Assistant,
                        content: EasyInputContent::Text("".into()),
                        ..Default::default()
                    }),
                    InputItem::EasyMessage(EasyInputMessage {
                        role: ResponseRole::User,
                        content: EasyInputContent::Text("second".into()),
                        ..Default::default()
                    }),
                ]),
                model: Some("test-model".into()),
                ..Default::default()
            },
            nvext: None,
        };

        let chat_req: NvCreateChatCompletionRequest = req.try_into().unwrap();
        let messages = &chat_req.inner.messages;
        assert_eq!(messages.len(), 3);
        assert!(matches!(
            messages[1],
            ChatCompletionRequestMessage::Assistant(_)
        ));
    }

    #[test]
    fn test_pure_function_call_turn_emits_null_content() {
        // Chat Completions spec allows `content: null` on assistant messages
        // that carry only `tool_calls`. Some Jinja templates gate on
        // `{% if message.content is not none %}`; we must not emit
        // `content: ""` for pure-tool-call turns. Turn-boundary cases (empty
        // OutputMessage with no tool_calls) still emit `Some(Text(""))`.
        let req = NvCreateResponse {
            inner: CreateResponse {
                input: InputParam::Items(vec![
                    InputItem::Item(Item::Message(MessageItem::Input(InputMessage {
                        content: vec![InputContent::InputText(InputTextContent {
                            text: "hi".into(),
                        })],
                        role: InputRole::User,
                        status: None,
                    }))),
                    InputItem::Item(Item::FunctionCall(FunctionToolCall {
                        arguments: "{}".into(),
                        call_id: "c".into(),
                        namespace: None,
                        name: "f".into(),
                        id: None,
                        status: None,
                    })),
                    InputItem::Item(Item::FunctionCallOutput(FunctionCallOutputItemParam {
                        call_id: "c".into(),
                        output: FunctionCallOutput::Text("ok".into()),
                        id: None,
                        status: None,
                    })),
                ]),
                model: Some("test-model".into()),
                ..Default::default()
            },
            nvext: None,
        };

        let chat_req: NvCreateChatCompletionRequest = req.try_into().unwrap();
        let messages = &chat_req.inner.messages;
        assert_eq!(messages.len(), 3);
        match &messages[1] {
            ChatCompletionRequestMessage::Assistant(a) => {
                assert!(
                    a.content.is_none(),
                    "pure tool-call turn must have content: null, got {:?}",
                    a.content
                );
                assert!(a.tool_calls.is_some());
            }
            _ => panic!("expected assistant message"),
        }
    }

    #[test]
    fn test_reasoning_item_routed_into_reasoning_content() {
        // Regression: Codex / Agents SDK round-trip Item::Reasoning mid-turn.
        // The converter must route the reasoning summary into the coalesced
        // assistant message's `reasoning_content`, not silently drop it.
        use dynamo_protocols::types::responses::{ReasoningItem, SummaryPart, SummaryTextContent};

        let req = NvCreateResponse {
            inner: CreateResponse {
                input: InputParam::Items(vec![
                    InputItem::Item(Item::Message(MessageItem::Input(InputMessage {
                        content: vec![InputContent::InputText(InputTextContent {
                            text: "solve".into(),
                        })],
                        role: InputRole::User,
                        status: None,
                    }))),
                    InputItem::Item(Item::Reasoning(ReasoningItem {
                        id: "rs_1".into(),
                        summary: vec![SummaryPart::SummaryText(SummaryTextContent {
                            text: "thinking step 1".into(),
                        })],
                        content: None,
                        encrypted_content: None,
                        status: None,
                    })),
                    InputItem::Item(Item::FunctionCall(FunctionToolCall {
                        arguments: "{}".into(),
                        call_id: "c".into(),
                        namespace: None,
                        name: "f".into(),
                        id: None,
                        status: None,
                    })),
                    InputItem::Item(Item::FunctionCallOutput(FunctionCallOutputItemParam {
                        call_id: "c".into(),
                        output: FunctionCallOutput::Text("ok".into()),
                        id: None,
                        status: None,
                    })),
                ]),
                model: Some("test-model".into()),
                ..Default::default()
            },
            nvext: None,
        };

        let chat_req: NvCreateChatCompletionRequest = req.try_into().unwrap();
        let messages = &chat_req.inner.messages;
        assert_eq!(messages.len(), 3);
        match &messages[1] {
            ChatCompletionRequestMessage::Assistant(a) => {
                match a
                    .reasoning_content
                    .as_ref()
                    .expect("reasoning must be preserved")
                {
                    ReasoningContent::Text(t) => assert_eq!(t, "thinking step 1"),
                    _ => panic!("expected Text reasoning content"),
                }
                assert!(a.tool_calls.is_some());
            }
            _ => panic!("expected assistant message with reasoning + tool_calls"),
        }
    }

    #[test]
    fn test_unsupported_item_variant_flushes_pending() {
        // Sequence: function_call → (an unsupported tool-output variant) →
        // function_call → function_call_output. Without a flush on the
        // catch-all, the two FunctionCalls would coalesce into a single
        // assistant `tool_calls` list despite being different semantic turns.
        use dynamo_protocols::types::responses::{
            ComputerCallOutputItemParam, ComputerScreenshotImage, ComputerScreenshotImageType,
        };

        let req = NvCreateResponse {
            inner: CreateResponse {
                input: InputParam::Items(vec![
                    InputItem::Item(Item::Message(MessageItem::Input(InputMessage {
                        content: vec![InputContent::InputText(InputTextContent {
                            text: "go".into(),
                        })],
                        role: InputRole::User,
                        status: None,
                    }))),
                    InputItem::Item(Item::FunctionCall(FunctionToolCall {
                        arguments: "{}".into(),
                        call_id: "c1".into(),
                        namespace: None,
                        name: "f".into(),
                        id: None,
                        status: None,
                    })),
                    InputItem::Item(Item::ComputerCallOutput(ComputerCallOutputItemParam {
                        call_id: "cc1".into(),
                        output: ComputerScreenshotImage {
                            r#type: ComputerScreenshotImageType::ComputerScreenshot,
                            image_url: None,
                            file_id: None,
                        },
                        acknowledged_safety_checks: None,
                        id: None,
                        status: None,
                    })),
                    InputItem::Item(Item::FunctionCall(FunctionToolCall {
                        arguments: "{}".into(),
                        call_id: "c2".into(),
                        namespace: None,
                        name: "f".into(),
                        id: None,
                        status: None,
                    })),
                    InputItem::Item(Item::FunctionCallOutput(FunctionCallOutputItemParam {
                        call_id: "c2".into(),
                        output: FunctionCallOutput::Text("ok".into()),
                        id: None,
                        status: None,
                    })),
                ]),
                model: Some("test-model".into()),
                ..Default::default()
            },
            nvext: None,
        };

        let chat_req: NvCreateChatCompletionRequest = req.try_into().unwrap();
        let messages = &chat_req.inner.messages;
        // Expected: User, Assistant(tc=[c1]), Assistant(tc=[c2]), Tool(c2)
        // Without the catch-all flush, we'd get Assistant(tc=[c1,c2]) instead.
        assert!(messages.len() >= 4, "catch-all must flush pending");
        let tc_msgs: Vec<_> = messages
            .iter()
            .filter_map(|m| match m {
                ChatCompletionRequestMessage::Assistant(a) => a.tool_calls.as_ref(),
                _ => None,
            })
            .collect();
        assert_eq!(
            tc_msgs.len(),
            2,
            "two tool-call turns must not coalesce across unsupported variant"
        );
        assert_eq!(tc_msgs[0].len(), 1);
        assert_eq!(tc_msgs[0][0].id, "c1");
        assert_eq!(tc_msgs[1].len(), 1);
        assert_eq!(tc_msgs[1][0].id, "c2");
    }

    #[test]
    fn test_tools_conversion() {
        let req = NvCreateResponse {
            inner: CreateResponse {
                input: InputParam::Text("hello".into()),
                model: Some("test-model".into()),
                tools: Some(vec![Tool::Function(FunctionTool {
                    name: "get_weather".into(),
                    parameters: Some(serde_json::json!({
                        "type": "object",
                        "properties": {
                            "location": {"type": "string"}
                        },
                        "required": ["location"]
                    })),
                    strict: Some(true),
                    description: Some("Get weather info".into()),
                    defer_loading: None,
                })]),
                ..Default::default()
            },
            nvext: None,
        };

        let chat_req: NvCreateChatCompletionRequest = req.try_into().unwrap();
        assert!(chat_req.inner.tools.is_some());
        let tools = chat_req.inner.tools.unwrap();
        assert_eq!(tools.len(), 1);
        assert_eq!(tools[0].function.name, "get_weather");
    }

    #[allow(deprecated)]
    #[test]
    fn test_into_nvresponse_from_chat_response() {
        let now = 1_726_000_000;
        let chat_resp = NvCreateChatCompletionResponse {
            inner: dynamo_protocols::types::CreateChatCompletionResponse {
                id: "chatcmpl-xyz".into(),
                choices: vec![dynamo_protocols::types::ChatChoice {
                    index: 0,
                    message: dynamo_protocols::types::ChatCompletionResponseMessage {
                        content: Some(dynamo_protocols::types::ChatCompletionMessageContent::Text(
                            "This is a reply".to_string(),
                        )),
                        refusal: None,
                        tool_calls: None,
                        role: dynamo_protocols::types::Role::Assistant,
                        function_call: None,
                        audio: None,
                        reasoning_content: None,
                    },
                    finish_reason: None,
                    stop_reason: None,
                    logprobs: None,
                }],
                created: now,
                model: "llama-3.1-8b-instruct".into(),
                service_tier: None,
                system_fingerprint: None,
                object: "chat.completion".to_string(),
                usage: None,
            },
            nvext: None,
        };

        let wrapped =
            chat_completion_to_response(chat_resp, &ResponseParams::default(), None).unwrap();

        assert_eq!(wrapped.inner.model, "llama-3.1-8b-instruct");
        assert_eq!(wrapped.inner.status, Status::Completed);
        assert_eq!(wrapped.inner.object, "response");
        assert!(wrapped.inner.id.starts_with("resp_"));

        let msg = match &wrapped.inner.output[0] {
            OutputItem::Message(m) => m,
            _ => panic!("Expected Message variant"),
        };
        assert_eq!(msg.role, AssistantRole::Assistant);

        match &msg.content[0] {
            OutputMessageContent::OutputText(txt) => {
                assert_eq!(txt.text, "This is a reply");
            }
            _ => panic!("Expected OutputText content"),
        }
    }

    #[allow(deprecated)]
    #[test]
    fn test_response_with_tool_calls() {
        let now = 1_726_000_000;
        let chat_resp = NvCreateChatCompletionResponse {
            inner: dynamo_protocols::types::CreateChatCompletionResponse {
                id: "chatcmpl-xyz".into(),
                choices: vec![dynamo_protocols::types::ChatChoice {
                    index: 0,
                    message: dynamo_protocols::types::ChatCompletionResponseMessage {
                        content: None,
                        refusal: None,
                        tool_calls: Some(vec![ChatCompletionMessageToolCall {
                            id: "call_abc".into(),
                            r#type: FunctionType::Function,
                            function: dynamo_protocols::types::FunctionCall {
                                name: "get_weather".into(),
                                arguments: r#"{"location":"SF"}"#.into(),
                            },
                        }]),
                        role: dynamo_protocols::types::Role::Assistant,
                        function_call: None,
                        audio: None,
                        reasoning_content: None,
                    },
                    finish_reason: None,
                    stop_reason: None,
                    logprobs: None,
                }],
                created: now,
                model: "test-model".into(),
                service_tier: None,
                system_fingerprint: None,
                object: "chat.completion".to_string(),
                usage: None,
            },
            nvext: None,
        };

        let wrapped =
            chat_completion_to_response(chat_resp, &ResponseParams::default(), None).unwrap();
        assert_eq!(wrapped.inner.output.len(), 1);
        match &wrapped.inner.output[0] {
            OutputItem::FunctionCall(fc) => {
                assert_eq!(fc.call_id, "call_abc");
                assert_eq!(fc.name, "get_weather");
            }
            _ => panic!("Expected FunctionCall output"),
        }
    }

    #[test]
    fn test_convert_top_logprobs_clamped() {
        assert_eq!(convert_top_logprobs(Some(5)), Some(5));
        assert_eq!(convert_top_logprobs(Some(21)), Some(20));
        assert_eq!(convert_top_logprobs(Some(255)), Some(20));
        assert_eq!(convert_top_logprobs(None), None);
    }

    #[test]
    fn test_parse_tool_call_text() {
        // Standard Qwen3 format
        let text = r#"<think>
Let me check the weather.
</think>

<tool_call>
{"name": "get_weather", "arguments": {"location": "San Francisco"}}
</tool_call>"#;
        let calls = parse_tool_call_text(text);
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].0, "get_weather");
        let args: serde_json::Value = serde_json::from_str(&calls[0].1).unwrap();
        assert_eq!(args["location"], "San Francisco");
    }

    #[test]
    fn test_parse_tool_call_text_multiple() {
        let text = r#"<tool_call>
{"name": "func_a", "arguments": {"x": 1}}
</tool_call>
<tool_call>
{"name": "func_b", "arguments": {"y": 2}}
</tool_call>"#;
        let calls = parse_tool_call_text(text);
        assert_eq!(calls.len(), 2);
        assert_eq!(calls[0].0, "func_a");
        assert_eq!(calls[1].0, "func_b");
    }

    #[test]
    fn test_parse_tool_call_text_no_calls() {
        let text = "Just a regular message with no tool calls.";
        let calls = parse_tool_call_text(text);
        assert!(calls.is_empty());
    }

    #[test]
    fn test_strip_tool_call_text() {
        let text = r#"<think>
thinking
</think>

<tool_call>
{"name": "f", "arguments": {}}
</tool_call>"#;
        let stripped = strip_tool_call_text(text);
        assert!(!stripped.contains("<tool_call>"));
        assert!(!stripped.contains("<think>"));
    }

    // ── PR1: reasoning / text.format / service_tier pass-through tests ──

    #[test]
    fn test_reasoning_effort_mapped_to_chat_completion() {
        use dynamo_protocols::types::ReasoningEffort;
        use dynamo_protocols::types::responses::Reasoning;

        let mut req = make_response_with_input("think hard");
        req.inner.reasoning = Some(Reasoning {
            effort: Some(ReasoningEffort::Medium),
            ..Default::default()
        });

        let chat: NvCreateChatCompletionRequest = req.try_into().unwrap();
        assert_eq!(chat.inner.reasoning_effort, Some(ReasoningEffort::Medium));
    }

    #[test]
    fn test_reasoning_none_leaves_chat_field_none() {
        let req = make_response_with_input("no reasoning");
        let chat: NvCreateChatCompletionRequest = req.try_into().unwrap();
        assert_eq!(chat.inner.reasoning_effort, None);
    }

    #[test]
    fn test_text_format_json_object_mapped() {
        use dynamo_protocols::types::ResponseFormat;
        use dynamo_protocols::types::responses::{
            ResponseTextParam, TextResponseFormatConfiguration,
        };

        let mut req = make_response_with_input("give json");
        req.inner.text = Some(ResponseTextParam {
            format: TextResponseFormatConfiguration::JsonObject,
            verbosity: None,
        });

        let chat: NvCreateChatCompletionRequest = req.try_into().unwrap();
        assert_eq!(chat.inner.response_format, Some(ResponseFormat::JsonObject));
    }

    #[test]
    fn test_text_format_json_schema_mapped() {
        use dynamo_protocols::types::responses::{
            ResponseTextParam, TextResponseFormatConfiguration,
        };
        use dynamo_protocols::types::{ResponseFormat, ResponseFormatJsonSchema};

        let schema = ResponseFormatJsonSchema {
            name: "city".into(),
            description: None,
            schema: Some(serde_json::json!({"type": "object"})),
            strict: Some(true),
        };
        let mut req = make_response_with_input("structured");
        req.inner.text = Some(ResponseTextParam {
            format: TextResponseFormatConfiguration::JsonSchema(schema.clone()),
            verbosity: None,
        });

        let chat: NvCreateChatCompletionRequest = req.try_into().unwrap();
        assert_eq!(
            chat.inner.response_format,
            Some(ResponseFormat::JsonSchema {
                json_schema: schema
            })
        );
    }

    #[test]
    fn test_text_format_plain_text_leaves_response_format_none() {
        use dynamo_protocols::types::responses::{
            ResponseTextParam, TextResponseFormatConfiguration,
        };

        let mut req = make_response_with_input("plain");
        req.inner.text = Some(ResponseTextParam {
            format: TextResponseFormatConfiguration::Text,
            verbosity: None,
        });

        let chat: NvCreateChatCompletionRequest = req.try_into().unwrap();
        assert_eq!(chat.inner.response_format, None);
    }

    #[test]
    fn test_service_tier_mapped_to_chat_completion() {
        use dynamo_protocols::types::ServiceTier as ChatServiceTier;
        use dynamo_protocols::types::responses::ServiceTier as RespServiceTier;

        let mut req = make_response_with_input("priority");
        req.inner.service_tier = Some(RespServiceTier::Priority);

        let chat: NvCreateChatCompletionRequest = req.try_into().unwrap();
        assert_eq!(chat.inner.service_tier, Some(ChatServiceTier::Priority));
    }

    #[test]
    fn test_parallel_tool_calls_mapped_to_chat_completion() {
        let mut req = make_response_with_input("parallel tools off");
        req.inner.parallel_tool_calls = Some(false);

        let chat: NvCreateChatCompletionRequest = req.try_into().unwrap();
        assert_eq!(chat.inner.parallel_tool_calls, Some(false));
    }

    #[test]
    fn test_response_echoes_reasoning() {
        use dynamo_protocols::types::ReasoningEffort;
        use dynamo_protocols::types::responses::Reasoning;

        let params = ResponseParams {
            reasoning: Some(Reasoning {
                effort: Some(ReasoningEffort::High),
                ..Default::default()
            }),
            ..Default::default()
        };

        let chat_resp = NvCreateChatCompletionResponse {
            inner: dynamo_protocols::types::CreateChatCompletionResponse {
                choices: vec![],
                created: 0,
                id: "test".into(),
                model: "m".into(),
                service_tier: None,
                system_fingerprint: None,
                object: "chat.completion".into(),
                usage: None,
            },
            nvext: None,
        };

        let resp = chat_completion_to_response(chat_resp, &params, None).unwrap();
        let reasoning = resp.inner.reasoning.unwrap();
        assert_eq!(reasoning.effort, Some(ReasoningEffort::High));
    }

    #[test]
    fn test_response_echoes_text_format() {
        use dynamo_protocols::types::responses::{
            ResponseTextParam, TextResponseFormatConfiguration,
        };

        let params = ResponseParams {
            text: Some(ResponseTextParam {
                format: TextResponseFormatConfiguration::JsonObject,
                verbosity: None,
            }),
            ..Default::default()
        };

        let chat_resp = NvCreateChatCompletionResponse {
            inner: dynamo_protocols::types::CreateChatCompletionResponse {
                choices: vec![],
                created: 0,
                id: "test".into(),
                model: "m".into(),
                service_tier: None,
                system_fingerprint: None,
                object: "chat.completion".into(),
                usage: None,
            },
            nvext: None,
        };

        let resp = chat_completion_to_response(chat_resp, &params, None).unwrap();
        let text = resp.inner.text.unwrap();
        assert_eq!(text.format, TextResponseFormatConfiguration::JsonObject);
    }

    #[test]
    fn test_response_echoes_service_tier() {
        use dynamo_protocols::types::responses::ServiceTier;

        let params = ResponseParams {
            service_tier: Some(ServiceTier::Flex),
            ..Default::default()
        };

        let chat_resp = NvCreateChatCompletionResponse {
            inner: dynamo_protocols::types::CreateChatCompletionResponse {
                choices: vec![],
                created: 0,
                id: "test".into(),
                model: "m".into(),
                service_tier: None,
                system_fingerprint: None,
                object: "chat.completion".into(),
                usage: None,
            },
            nvext: None,
        };

        let resp = chat_completion_to_response(chat_resp, &params, None).unwrap();
        assert_eq!(resp.inner.service_tier, Some(ServiceTier::Flex));
    }

    #[test]
    fn test_response_echoes_parallel_tool_calls() {
        let params = ResponseParams {
            parallel_tool_calls: Some(false),
            ..Default::default()
        };

        let chat_resp = NvCreateChatCompletionResponse {
            inner: dynamo_protocols::types::CreateChatCompletionResponse {
                choices: vec![],
                created: 0,
                id: "test".into(),
                model: "m".into(),
                service_tier: None,
                system_fingerprint: None,
                object: "chat.completion".into(),
                usage: None,
            },
            nvext: None,
        };

        let resp = chat_completion_to_response(chat_resp, &params, None).unwrap();
        assert_eq!(resp.inner.parallel_tool_calls, Some(false));
    }

    #[test]
    fn test_bare_assistant_output_message_deserializes_via_owned_types() {
        // Regression: upstream async-openai's OutputMessage required `id` and
        // `status`. Dynamo-owned types make them optional so real-world client
        // shapes (no id/status, no annotations) round-trip successfully.
        let json = serde_json::json!({
            "role": "assistant",
            "content": [{"type": "output_text", "text": "Hello!"}],
            "type": "message"
        });

        let item: InputItem =
            serde_json::from_value(json).expect("relaxed deserialize should succeed");
        match item {
            InputItem::Item(Item::Message(MessageItem::Output(msg))) => {
                assert_eq!(msg.role, AssistantRole::Assistant);
                assert!(msg.id.is_none());
                assert!(msg.status.is_none());
            }
            other => panic!("Expected Item::Message(Output), got {:?}", other),
        }
    }

    #[test]
    fn test_nvcreate_response_accepts_bare_assistant_messages() {
        // End-to-end: a real Codex-style payload with an interstitial assistant
        // text item (no id/status/annotations) deserializes into NvCreateResponse
        // via the standard derive on our Dynamo-owned CreateResponse chain.
        let body = serde_json::json!({
            "model": "m",
            "input": [
                {"type": "message", "role": "user", "content": [
                    {"type": "input_text", "text": "hi"}
                ]},
                {"type": "function_call", "call_id": "c", "name": "f", "arguments": "{}"},
                {"type": "message", "role": "assistant", "content": [
                    {"type": "output_text", "text": "\n\n\n"}
                ]},
                {"type": "function_call_output", "call_id": "c", "output": "x"}
            ]
        });

        let req: NvCreateResponse =
            serde_json::from_value(body).expect("relaxed deserialize should succeed");
        let items = match &req.inner.input {
            InputParam::Items(items) => items,
            _ => panic!("expected Items input"),
        };
        assert_eq!(items.len(), 4);
        match &items[2] {
            InputItem::Item(Item::Message(MessageItem::Output(out))) => {
                assert_eq!(out.role, AssistantRole::Assistant);
            }
            other => panic!("expected MessageItem::Output, got {:?}", other),
        }
    }

    #[test]
    fn test_output_message_with_id_and_status_still_works() {
        use dynamo_protocols::types::responses::{InputItem, Item, MessageItem, OutputStatus};

        let json = serde_json::json!({
            "role": "assistant",
            "id": "msg_abc123",
            "status": "completed",
            "content": [{"type": "output_text", "text": "Hello!", "annotations": []}],
            "type": "message"
        });

        let item: InputItem = serde_json::from_value(json).unwrap();
        match item {
            InputItem::Item(Item::Message(MessageItem::Output(msg))) => {
                assert_eq!(msg.id.as_deref(), Some("msg_abc123"));
                assert_eq!(msg.status, Some(OutputStatus::Completed));
            }
            other => panic!("Expected Item::Message(Output), got {:?}", other),
        }
    }

    // ── PR2: include filtering + truncation echo-back tests ──

    fn make_chat_resp_with_text(text: &str) -> NvCreateChatCompletionResponse {
        use dynamo_protocols::types::{
            ChatChoice, ChatCompletionMessageContent, ChatCompletionResponseMessage, FinishReason,
        };
        NvCreateChatCompletionResponse {
            inner: dynamo_protocols::types::CreateChatCompletionResponse {
                choices: vec![ChatChoice {
                    index: 0,
                    #[allow(deprecated)]
                    message: ChatCompletionResponseMessage {
                        content: Some(ChatCompletionMessageContent::Text(text.into())),
                        role: dynamo_protocols::types::Role::Assistant,
                        tool_calls: None,
                        refusal: None,
                        reasoning_content: None,
                        function_call: None,
                        audio: None,
                    },
                    finish_reason: Some(FinishReason::Stop),
                    stop_reason: None,
                    logprobs: None,
                }],
                created: 0,
                id: "test".into(),
                model: "m".into(),
                service_tier: None,
                system_fingerprint: None,
                object: "chat.completion".into(),
                usage: None,
            },
            nvext: None,
        }
    }

    #[test]
    fn test_include_logprobs_stripped_by_default() {
        let chat_resp = make_chat_resp_with_text("hello");
        let params = ResponseParams::default();
        let resp = chat_completion_to_response(chat_resp, &params, None).unwrap();

        for item in &resp.inner.output {
            if let OutputItem::Message(msg) = item {
                for content in &msg.content {
                    if let OutputMessageContent::OutputText(t) = content {
                        assert!(
                            t.logprobs.is_none(),
                            "logprobs should be stripped by default"
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn test_include_logprobs_kept_when_requested() {
        use dynamo_protocols::types::responses::IncludeEnum;

        let chat_resp = make_chat_resp_with_text("hello");
        let params = ResponseParams {
            include: Some(vec![IncludeEnum::MessageOutputTextLogprobs]),
            ..Default::default()
        };
        let resp = chat_completion_to_response(chat_resp, &params, None).unwrap();

        let mut found_text = false;
        for item in &resp.inner.output {
            if let OutputItem::Message(msg) = item {
                for content in &msg.content {
                    if let OutputMessageContent::OutputText(t) = content {
                        found_text = true;
                        assert!(
                            t.logprobs.is_some(),
                            "logprobs should be preserved when included"
                        );
                    }
                }
            }
        }
        assert!(found_text, "Expected text output");
    }

    #[test]
    fn test_truncation_auto_echoed_back() {
        use dynamo_protocols::types::responses::Truncation;

        let chat_resp = make_chat_resp_with_text("hello");
        let params = ResponseParams {
            truncation: Some(Truncation::Auto),
            ..Default::default()
        };
        let resp = chat_completion_to_response(chat_resp, &params, None).unwrap();
        assert_eq!(resp.inner.truncation, Some(Truncation::Auto));
    }

    #[test]
    fn test_truncation_defaults_to_disabled() {
        let chat_resp = make_chat_resp_with_text("hello");
        let params = ResponseParams::default();
        let resp = chat_completion_to_response(chat_resp, &params, None).unwrap();
        assert_eq!(resp.inner.truncation, Some(Truncation::Disabled));
    }

    /// Validate the JSON wire shape of NvResponse.
    ///
    /// The migration to upstream async-openai v0.34 removed fields that were
    /// incorrectly present on our old local Response type (they belong on the
    /// request, not the response, per the OpenAI Responses API spec).
    #[test]
    fn test_response_wire_format_shape() {
        let chat_resp = make_chat_resp_with_text("hello");
        let params = ResponseParams::default();
        let resp = chat_completion_to_response(chat_resp, &params, None).unwrap();
        let json = serde_json::to_value(&resp).unwrap();

        // Fields that were on our old local type but are NOT in the OpenAI
        // Responses API spec -- they are request-level, not response-level.
        assert!(json.get("frequency_penalty").is_none());
        assert!(json.get("presence_penalty").is_none());
        assert!(json.get("store").is_none());
        assert!(json.get("max_tool_calls").is_none());

        // Fields that should be present with expected values
        assert_eq!(json["object"], "response");
        assert_eq!(json["status"], "completed");
        assert_eq!(json["metadata"], serde_json::json!({}));
        assert!(json["output"].is_array());
        assert!(json["output"][0].get("id").is_some());
        assert!(json["output"][0].get("status").is_some());

        // Optional fields with None should be omitted (upstream uses skip_serializing_if)
        assert!(json.get("error").is_none());
        assert!(json.get("incomplete_details").is_none());
        assert!(json.get("billing").is_none());
        assert!(json.get("conversation").is_none());
        assert!(json.get("safety_identifier").is_none());

        // nvext should be omitted when None
        assert!(json.get("nvext").is_none());
    }
}
