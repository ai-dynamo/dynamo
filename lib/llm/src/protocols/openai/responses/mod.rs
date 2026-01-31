// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

pub mod stream_converter;

use dynamo_async_openai::types::responses::{
    AssistantRole, FunctionCallOutput, FunctionToolCall, InputContent, InputItem, InputParam,
    InputRole, Item, MessageItem, OutputItem, OutputMessage, OutputMessageContent, OutputStatus,
    OutputTextContent, Response, ResponseTextParam, Role as ResponseRole, ServiceTier, Status,
    TextResponseFormatConfiguration, Tool, ToolChoiceOptions, ToolChoiceParam, Truncation,
};
use dynamo_async_openai::types::{
    ChatCompletionMessageToolCall, ChatCompletionNamedToolChoice, ChatCompletionRequestAssistantMessage,
    ChatCompletionRequestAssistantMessageContent, ChatCompletionRequestMessage,
    ChatCompletionRequestMessageContentPartImage, ChatCompletionRequestMessageContentPartText,
    ChatCompletionRequestMessageContentPartVideo, ChatCompletionRequestSystemMessage,
    ChatCompletionRequestSystemMessageContent, ChatCompletionRequestToolMessage,
    ChatCompletionRequestToolMessageContent, ChatCompletionRequestUserMessage,
    ChatCompletionRequestUserMessageContent, ChatCompletionRequestUserMessageContentPart,
    ChatCompletionTool, ChatCompletionToolChoiceOption, ChatCompletionToolType,
    CreateChatCompletionRequest, FunctionName, FunctionObject, ImageDetail as ChatImageDetail,
    ImageUrl, VideoUrl,
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
    /// Flattened CreateResponse fields (model, input, temperature, etc.)
    #[serde(flatten)]
    pub inner: dynamo_async_openai::types::responses::CreateResponse,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub nvext: Option<NvExt>,
}

#[derive(ToSchema, Serialize, Deserialize, Validate, Debug, Clone)]
pub struct NvResponse {
    /// Flattened Response fields.
    #[serde(flatten)]
    pub inner: dynamo_async_openai::types::responses::Response,

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
fn convert_image_detail(
    detail: &dynamo_async_openai::types::responses::ImageDetail,
) -> ChatImageDetail {
    match detail {
        dynamo_async_openai::types::responses::ImageDetail::Auto => ChatImageDetail::Auto,
        dynamo_async_openai::types::responses::ImageDetail::Low => ChatImageDetail::Low,
        dynamo_async_openai::types::responses::ImageDetail::High => ChatImageDetail::High,
    }
}

/// Convert a slice of InputContent to ChatCompletionRequestUserMessageContent.
fn convert_input_content_to_user_content(
    content: &[InputContent],
) -> Result<ChatCompletionRequestUserMessageContent, anyhow::Error> {
    // If there's a single InputText, treat as simple text
    if content.len() == 1 {
        if let InputContent::InputText(t) = &content[0] {
            return Ok(ChatCompletionRequestUserMessageContent::Text(t.text.clone()));
        }
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
                let url_str = img.image_url.as_deref().unwrap_or_default();
                let url = url::Url::parse(url_str).map_err(|e| {
                    anyhow::anyhow!("Invalid image URL '{}': {}", url_str, e)
                })?;
                chat_parts.push(ChatCompletionRequestUserMessageContentPart::ImageUrl(
                    ChatCompletionRequestMessageContentPartImage {
                        image_url: ImageUrl {
                            url,
                            detail: Some(convert_image_detail(&img.detail)),
                            uuid: None,
                        },
                    },
                ));
            }
            InputContent::InputVideo(vid) => {
                let url = url::Url::parse(&vid.video).map_err(|e| {
                    anyhow::anyhow!("Invalid video URL '{}': {}", vid.video, e)
                })?;
                chat_parts.push(ChatCompletionRequestUserMessageContentPart::VideoUrl(
                    ChatCompletionRequestMessageContentPartVideo {
                        video_url: VideoUrl {
                            url,
                            detail: None,
                            uuid: None,
                        },
                    },
                ));
            }
            InputContent::InputAudio(_) => {
                return Err(anyhow::anyhow!(
                    "Audio input content is not yet supported"
                ));
            }
            InputContent::InputFile(_) => {
                return Err(anyhow::anyhow!(
                    "File input content is not yet supported"
                ));
            }
        }
    }
    Ok(ChatCompletionRequestUserMessageContent::Array(chat_parts))
}

/// Convert a slice of InputContent to a plain text string (for system/developer messages).
fn convert_input_content_to_text(content: &[InputContent]) -> String {
    // Concatenate all text parts; non-text parts are skipped.
    content
        .iter()
        .filter_map(|p| match p {
            InputContent::InputText(t) => Some(t.text.as_str()),
            _ => None,
        })
        .collect::<Vec<_>>()
        .join("")
}

/// Convert InputParam::Items to a Vec of ChatCompletionRequestMessages.
fn convert_input_items_to_messages(
    items: &[InputItem],
) -> Result<Vec<ChatCompletionRequestMessage>, anyhow::Error> {
    let mut messages = Vec::with_capacity(items.len());

    for item in items {
        match item {
            InputItem::Item(inner_item) => match inner_item {
                Item::Message(msg_item) => match msg_item {
                    MessageItem::Input(msg) => {
                        let chat_msg = match msg.role {
                            InputRole::System | InputRole::Developer => {
                                let text = convert_input_content_to_text(&msg.content);
                                ChatCompletionRequestMessage::System(
                                    ChatCompletionRequestSystemMessage {
                                        content:
                                            ChatCompletionRequestSystemMessageContent::Text(text),
                                        name: None,
                                    },
                                )
                            }
                            InputRole::User => {
                                let content =
                                    convert_input_content_to_user_content(&msg.content)?;
                                ChatCompletionRequestMessage::User(
                                    ChatCompletionRequestUserMessage { content, name: None },
                                )
                            }
                        };
                        messages.push(chat_msg);
                    }
                    MessageItem::Output(out_msg) => {
                        // Previous assistant output message -> assistant message
                        let text = out_msg
                            .content
                            .iter()
                            .filter_map(|c| match c {
                                OutputMessageContent::OutputText(t) => Some(t.text.as_str()),
                                _ => None,
                            })
                            .collect::<Vec<_>>()
                            .join("");
                        messages.push(ChatCompletionRequestMessage::Assistant(
                            ChatCompletionRequestAssistantMessage {
                                content: Some(
                                    ChatCompletionRequestAssistantMessageContent::Text(text),
                                ),
                                refusal: None,
                                name: None,
                                audio: None,
                                tool_calls: None,
                                #[allow(deprecated)]
                                function_call: None,
                            },
                        ));
                    }
                },
                Item::FunctionCall(fc) => {
                    // A function call from a previous assistant turn -> assistant message with tool_calls
                    messages.push(ChatCompletionRequestMessage::Assistant(
                        ChatCompletionRequestAssistantMessage {
                            content: None,
                            refusal: None,
                            name: None,
                            audio: None,
                            tool_calls: Some(vec![ChatCompletionMessageToolCall {
                                id: fc.call_id.clone(),
                                r#type: ChatCompletionToolType::Function,
                                function: dynamo_async_openai::types::FunctionCall {
                                    name: fc.name.clone(),
                                    arguments: fc.arguments.clone(),
                                },
                            }]),
                            #[allow(deprecated)]
                            function_call: None,
                        },
                    ));
                }
                Item::FunctionCallOutput(fco) => {
                    // The output of a function call -> tool message
                    let output_text = match &fco.output {
                        FunctionCallOutput::Text(text) => text.clone(),
                        FunctionCallOutput::Content(parts) => {
                            convert_input_content_to_text(parts)
                        }
                    };
                    messages.push(ChatCompletionRequestMessage::Tool(
                        ChatCompletionRequestToolMessage {
                            content: ChatCompletionRequestToolMessageContent::Text(output_text),
                            tool_call_id: fco.call_id.clone(),
                        },
                    ));
                }
                _ => {
                    // Skip other item types (file search, computer call, etc.)
                }
            },
            InputItem::EasyMessage(easy) => {
                // Handle easy input messages based on role
                let content_text = match &easy.content {
                    dynamo_async_openai::types::responses::EasyInputContent::Text(text) => {
                        text.clone()
                    }
                    dynamo_async_openai::types::responses::EasyInputContent::ContentList(parts) => {
                        convert_input_content_to_text(parts)
                    }
                };
                let chat_msg = match easy.role {
                    ResponseRole::System | ResponseRole::Developer => {
                        ChatCompletionRequestMessage::System(
                            ChatCompletionRequestSystemMessage {
                                content: ChatCompletionRequestSystemMessageContent::Text(
                                    content_text,
                                ),
                                name: None,
                            },
                        )
                    }
                    ResponseRole::User => ChatCompletionRequestMessage::User(
                        ChatCompletionRequestUserMessage {
                            content: ChatCompletionRequestUserMessageContent::Text(content_text),
                            name: None,
                        },
                    ),
                    ResponseRole::Assistant => ChatCompletionRequestMessage::Assistant(
                        ChatCompletionRequestAssistantMessage {
                            content: Some(
                                ChatCompletionRequestAssistantMessageContent::Text(content_text),
                            ),
                            refusal: None,
                            name: None,
                            audio: None,
                            tool_calls: None,
                            #[allow(deprecated)]
                            function_call: None,
                        },
                    ),
                };
                messages.push(chat_msg);
            }
            InputItem::ItemReference(_) => {
                // Skip item references
            }
        }
    }

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

impl TryFrom<NvCreateResponse> for NvCreateChatCompletionRequest {
    type Error = anyhow::Error;

    fn try_from(resp: NvCreateResponse) -> Result<Self, Self::Error> {
        let mut messages = Vec::new();

        // Prepend instructions as system message if present
        if let Some(instructions) = &resp.inner.instructions {
            messages.push(ChatCompletionRequestMessage::System(
                ChatCompletionRequestSystemMessage {
                    content: ChatCompletionRequestSystemMessageContent::Text(
                        instructions.clone(),
                    ),
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

        Ok(NvCreateChatCompletionRequest {
            inner: CreateChatCompletionRequest {
                messages,
                model: resp.inner.model.unwrap_or_default(),
                temperature: resp.inner.temperature,
                top_p: resp.inner.top_p,
                max_completion_tokens: resp.inner.max_output_tokens,
                top_logprobs,
                metadata: resp.inner.metadata,
                stream,
                tools,
                tool_choice,
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

/// Build an assistant text message output item.
fn make_text_message(id: String, text: String) -> OutputItem {
    OutputItem::Message(OutputMessage {
        id,
        role: AssistantRole::Assistant,
        status: OutputStatus::Completed,
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
        name,
        id: Some(format!("fc_{}", Uuid::new_v4().simple())),
        status: Some(OutputStatus::Completed),
    })
}

impl TryFrom<NvCreateChatCompletionResponse> for NvResponse {
    type Error = anyhow::Error;

    fn try_from(nv_resp: NvCreateChatCompletionResponse) -> Result<Self, Self::Error> {
        let chat_resp = nv_resp;
        let nvext = chat_resp.nvext.clone();
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
                        name: tc.function.name.clone(),
                        id: Some(format!("fc_{}", Uuid::new_v4().simple())),
                        status: Some(OutputStatus::Completed),
                    }));
                }
            }

            // Handle text content -- also parse <tool_call> blocks from models
            // that emit tool calls as text (e.g. Qwen3)
            if let Some(content_text) = choice.message.content {
                if !content_text.is_empty() {
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
            }

            if output.is_empty() {
                output.push(make_text_message(message_id, String::new()));
            }
        } else {
            tracing::warn!("No choices in chat completion response, using empty content");
            output.push(make_text_message(message_id, String::new()));
        }

        let created_at = chat_resp.created as u64;
        let response = Response {
            id: response_id,
            object: "response".to_string(),
            created_at,
            completed_at: Some(created_at),
            model: chat_resp.model,
            status: Status::Completed,
            output,
            // Spec-required defaults (OpenResponses requires these as non-null)
            background: Some(false),
            frequency_penalty: Some(0.0),
            metadata: Some(serde_json::Value::Object(Default::default())),
            parallel_tool_calls: Some(true),
            presence_penalty: Some(0.0),
            store: Some(true),
            temperature: Some(1.0),
            text: Some(ResponseTextParam {
                format: TextResponseFormatConfiguration::Text,
                verbosity: None,
            }),
            tool_choice: Some(ToolChoiceParam::Mode(ToolChoiceOptions::Auto)),
            tools: Some(vec![]),
            top_p: Some(1.0),
            truncation: Some(Truncation::Disabled),
            // Nullable but required to be present (null is valid)
            billing: None,
            conversation: None,
            error: None,
            incomplete_details: None,
            instructions: None,
            max_output_tokens: None,
            max_tool_calls: None,
            previous_response_id: None,
            prompt: None,
            prompt_cache_key: None,
            prompt_cache_retention: None,
            reasoning: None,
            safety_identifier: None,
            service_tier: Some(ServiceTier::Auto),
            top_logprobs: Some(0),
            usage: None,
        };

        Ok(NvResponse {
            inner: response,
            nvext,
        })
    }
}

#[cfg(test)]
mod tests {
    use dynamo_async_openai::types::responses::{
        CreateResponse, FunctionCallOutput, FunctionCallOutputItemParam, FunctionTool,
        FunctionToolCall, ImageDetail, InputContent, InputImageContent, InputItem, InputMessage,
        InputParam, InputRole, InputTextContent, Item, MessageItem, Tool,
    };
    use dynamo_async_openai::types::{
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
                    InputItem::Item(Item::Message(MessageItem::Output(OutputMessage {
                        id: "msg_1".into(),
                        role: AssistantRole::Assistant,
                        status: OutputStatus::Completed,
                        content: vec![OutputMessageContent::OutputText(OutputTextContent {
                            text: "4".into(),
                            annotations: vec![],
                            logprobs: None,
                        })],
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
        assert!(matches!(messages[0], ChatCompletionRequestMessage::System(_)));
        assert!(matches!(messages[1], ChatCompletionRequestMessage::User(_)));
        assert!(matches!(messages[2], ChatCompletionRequestMessage::Assistant(_)));
        assert!(matches!(messages[3], ChatCompletionRequestMessage::User(_)));
    }

    #[test]
    fn test_input_items_with_image() {
        let req = NvCreateResponse {
            inner: CreateResponse {
                input: InputParam::Items(vec![InputItem::Item(Item::Message(
                    MessageItem::Input(InputMessage {
                        content: vec![
                            InputContent::InputText(InputTextContent {
                                text: "What is in this image?".into(),
                            }),
                            InputContent::InputImage(InputImageContent {
                                detail: ImageDetail::Auto,
                                file_id: None,
                                image_url: Some("https://example.com/cat.jpg".into()),
                            }),
                        ],
                        role: InputRole::User,
                        status: None,
                    }),
                ))]),
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
        assert!(matches!(messages[1], ChatCompletionRequestMessage::Assistant(_)));
        assert!(matches!(messages[2], ChatCompletionRequestMessage::Tool(_)));
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
            id: "chatcmpl-xyz".into(),
            choices: vec![dynamo_async_openai::types::ChatChoice {
                index: 0,
                message: dynamo_async_openai::types::ChatCompletionResponseMessage {
                    content: Some("This is a reply".into()),
                    refusal: None,
                    tool_calls: None,
                    role: dynamo_async_openai::types::Role::Assistant,
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
            nvext: None,
        };

        let wrapped: NvResponse = chat_resp.try_into().unwrap();

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
            id: "chatcmpl-xyz".into(),
            choices: vec![dynamo_async_openai::types::ChatChoice {
                index: 0,
                message: dynamo_async_openai::types::ChatCompletionResponseMessage {
                    content: None,
                    refusal: None,
                    tool_calls: Some(vec![ChatCompletionMessageToolCall {
                        id: "call_abc".into(),
                        r#type: ChatCompletionToolType::Function,
                        function: dynamo_async_openai::types::FunctionCall {
                            name: "get_weather".into(),
                            arguments: r#"{"location":"SF"}"#.into(),
                        },
                    }]),
                    role: dynamo_async_openai::types::Role::Assistant,
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
            nvext: None,
        };

        let wrapped: NvResponse = chat_resp.try_into().unwrap();
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
}
