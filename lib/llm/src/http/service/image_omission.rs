// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use dynamo_protocols::types::responses::{
    EasyInputContent, FunctionCallOutput, InputContent, InputItem, InputParam, InputTextContent,
    Item, MessageItem,
};
use dynamo_protocols::types::{
    ChatCompletionRequestMessage, ChatCompletionRequestMessageContentPartText,
    ChatCompletionRequestToolMessageContent, ChatCompletionRequestToolMessageContentPart,
    ChatCompletionRequestUserMessageContent, ChatCompletionRequestUserMessageContentPart,
};

pub(super) const OMITTED_IMAGE: &str = "[image omitted]";

pub(super) type ChatEngineSelection = (
    crate::types::openai::chat_completions::OpenAIChatCompletionsStreamingEngine,
    crate::protocols::openai::ParsingOptions,
    bool,
);

pub(super) fn select_chat_engine(
    state: &super::service_v2::State,
    model: &str,
) -> Result<ChatEngineSelection, crate::discovery::ModelManagerError> {
    state
        .manager()
        .get_committed_model(model)
        .ok_or_else(|| crate::discovery::ModelManagerError::ModelNotFound(model.to_owned()))?
        .get_chat_engine_with_image_policy()
}

pub(super) fn select_chat_engine_for_capability(
    state: &super::service_v2::State,
    model: &str,
    capability: &str,
) -> Result<Option<ChatEngineSelection>, crate::discovery::ModelManagerError> {
    state
        .manager()
        .get_committed_model(model)
        .ok_or_else(|| crate::discovery::ModelManagerError::ModelNotFound(model.to_owned()))?
        .get_chat_engine_with_image_policy_for_capability(Some(capability))
}

pub(super) fn omit_chat_images(messages: &mut [ChatCompletionRequestMessage]) {
    for message in messages {
        match message {
            ChatCompletionRequestMessage::User(message) => {
                if let ChatCompletionRequestUserMessageContent::Array(parts) = &mut message.content
                {
                    for part in parts {
                        if matches!(
                            part,
                            ChatCompletionRequestUserMessageContentPart::ImageUrl(_)
                        ) {
                            *part = ChatCompletionRequestUserMessageContentPart::Text(
                                ChatCompletionRequestMessageContentPartText {
                                    text: OMITTED_IMAGE.into(),
                                    prompt_cache_breakpoint: None,
                                },
                            );
                        }
                    }
                }
            }
            ChatCompletionRequestMessage::Tool(message) => {
                if let ChatCompletionRequestToolMessageContent::Array(parts) = &mut message.content
                {
                    for part in parts {
                        if matches!(
                            part,
                            ChatCompletionRequestToolMessageContentPart::ImageUrl(_)
                        ) {
                            *part = ChatCompletionRequestToolMessageContentPart::Text(
                                ChatCompletionRequestMessageContentPartText {
                                    text: OMITTED_IMAGE.into(),
                                    prompt_cache_breakpoint: None,
                                },
                            );
                        }
                    }
                }
            }
            _ => {}
        }
    }
}

pub(super) fn omit_response_images(input: &mut InputParam) {
    let InputParam::Items(items) = input else {
        return;
    };
    for item in items {
        let parts = match item {
            InputItem::EasyMessage(message) => match &mut message.content {
                EasyInputContent::ContentList(parts) => parts,
                EasyInputContent::Text(_) => continue,
            },
            InputItem::Item(Item::Message(MessageItem::Input(message))) => &mut message.content,
            InputItem::Item(Item::FunctionCallOutput(output)) => match &mut output.output {
                FunctionCallOutput::Content(parts) => parts,
                FunctionCallOutput::Text(_) => continue,
            },
            _ => continue,
        };
        for part in parts {
            if matches!(part, InputContent::InputImage(_)) {
                *part = InputContent::InputText(InputTextContent {
                    text: OMITTED_IMAGE.into(),
                    prompt_cache_breakpoint: None,
                });
            }
        }
    }
}
