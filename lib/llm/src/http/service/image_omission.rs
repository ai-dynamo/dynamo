// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use crate::protocols::anthropic::types::{
    AnthropicContentBlock, AnthropicMessage, AnthropicMessageContent, AnthropicRole,
    ToolResultContent, ToolResultContentBlock,
};
use dynamo_protocols::types::responses::{
    EasyInputContent, FunctionCallOutput, InputContent, InputItem, InputParam, InputRole,
    InputTextContent, Item, MessageItem, Role,
};
use dynamo_protocols::types::{
    ChatCompletionRequestMessage, ChatCompletionRequestMessageContentPartText,
    ChatCompletionRequestToolMessageContent, ChatCompletionRequestToolMessageContentPart,
    ChatCompletionRequestUserMessageContent, ChatCompletionRequestUserMessageContentPart,
};

pub(super) const OMITTED_IMAGE: &str = "[image omitted]";

pub(super) fn log_image_omission(model: &str, replaced_images: usize) {
    if replaced_images > 0 {
        tracing::debug!(
            model,
            replaced_images,
            "Replaced images for text-only deployment"
        );
    }
}

pub(super) fn omit_chat_images(messages: &mut [ChatCompletionRequestMessage]) -> usize {
    let mut replaced_images = 0;
    for message in messages {
        match message {
            ChatCompletionRequestMessage::User(message) => {
                if let ChatCompletionRequestUserMessageContent::Array(parts) = &mut message.content
                {
                    for part in parts {
                        if matches!(
                            part,
                            ChatCompletionRequestUserMessageContentPart::ImageUrl(image)
                                if (image.image_url.is_some() || image.uuid.is_some())
                                    && image.uuid.as_deref().is_none_or(|uuid| !uuid.is_empty())
                        ) {
                            replaced_images += 1;
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
                            ChatCompletionRequestToolMessageContentPart::ImageUrl(image)
                                if (image.image_url.is_some() || image.uuid.is_some())
                                    && image.uuid.as_deref().is_none_or(|uuid| !uuid.is_empty())
                        ) {
                            replaced_images += 1;
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
    replaced_images
}

/// Rewrites only parts the converter keeps as images: user messages and
/// function-call outputs. Other roles are reduced to text for every deployment.
pub(super) fn omit_response_images(input: &mut InputParam) -> usize {
    let InputParam::Items(items) = input else {
        return 0;
    };
    let mut replaced_images = 0;
    for item in items {
        let parts = match item {
            InputItem::EasyMessage(message) if matches!(message.role, Role::User) => {
                match &mut message.content {
                    EasyInputContent::ContentList(parts) => parts,
                    EasyInputContent::Text(_) => continue,
                }
            }
            InputItem::Item(Item::Message(MessageItem::Input(message)))
                if matches!(message.role, InputRole::User) =>
            {
                &mut message.content
            }
            InputItem::Item(Item::FunctionCallOutput(output)) => match &mut output.output {
                FunctionCallOutput::Content(parts) => parts,
                FunctionCallOutput::Text(_) => continue,
            },
            _ => continue,
        };
        for part in parts {
            let InputContent::InputImage(image) = part else {
                continue;
            };
            let valid_source = match image.image_url.as_deref() {
                Some(url) => url::Url::parse(url).is_ok(),
                None => image
                    .file_id
                    .as_deref()
                    .is_some_and(|id| !id.trim().is_empty()),
            };
            if valid_source {
                replaced_images += 1;
                *part = InputContent::InputText(InputTextContent {
                    text: OMITTED_IMAGE.into(),
                    prompt_cache_breakpoint: None,
                });
            }
        }
    }
    replaced_images
}

/// Only user turns keep images through conversion.
pub(super) fn omit_anthropic_images(messages: &mut [AnthropicMessage]) -> usize {
    let mut replaced_images = 0;
    for message in messages {
        if !matches!(message.role, AnthropicRole::User) {
            continue;
        }
        let AnthropicMessageContent::Blocks { content } = &mut message.content else {
            continue;
        };
        for block in content {
            match block {
                AnthropicContentBlock::Image { .. } => {
                    *block = AnthropicContentBlock::Text {
                        text: OMITTED_IMAGE.into(),
                        citations: None,
                        cache_control: None,
                    };
                    replaced_images += 1;
                }
                AnthropicContentBlock::ToolResult {
                    content: Some(ToolResultContent::Blocks(blocks)),
                    ..
                } => {
                    for block in blocks {
                        if matches!(block, ToolResultContentBlock::Image { .. }) {
                            *block = ToolResultContentBlock::Text {
                                text: OMITTED_IMAGE.into(),
                            };
                            replaced_images += 1;
                        }
                    }
                }
                _ => {}
            }
        }
    }
    replaced_images
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn source_less_chat_images_reach_validation_unchanged() {
        let mut messages: Vec<ChatCompletionRequestMessage> = serde_json::from_value(json!([
            {"role":"user","content":[{"type":"image_url"}]},
            {"role":"tool","tool_call_id":"call_1","content":[{"type":"image_url"}]},
            {"role":"user","content":[{"type":"image_url","uuid":""}]},
            {"role":"tool","tool_call_id":"call_2","content":[{"type":"image_url","uuid":"","image_url":{"url":"https://example.com/image.png"}}]}
        ]))
        .unwrap();
        let original = serde_json::to_value(&messages).unwrap();
        assert_eq!(omit_chat_images(&mut messages), 0);
        assert_eq!(serde_json::to_value(&messages).unwrap(), original);
    }

    #[test]
    fn only_roles_that_keep_images_are_rewritten() {
        let image = json!({"type":"input_image","image_url":"https://example.com/a.png"});
        let mut input: InputParam = serde_json::from_value(json!([
            {"role":"system","content":[image]},
            {"role":"developer","content":[image]},
            {"role":"assistant","content":[image]},
            {"type":"message","role":"system","content":[image]},
            {"role":"user","content":[image]},
            {"type":"message","role":"user","content":[image]},
            {"type":"function_call_output","call_id":"call_1","output":[image]}
        ]))
        .unwrap();
        assert_eq!(omit_response_images(&mut input), 3);
        let InputParam::Items(items) = &input else {
            unreachable!()
        };
        for (index, item) in items.iter().enumerate() {
            let rewritten = serde_json::to_string(item).unwrap().contains(OMITTED_IMAGE);
            assert_eq!(rewritten, index >= 4, "item {index}");
        }

        let image =
            json!({"type":"image","source":{"type":"url","url":"https://example.com/a.png"}});
        let mut messages: Vec<AnthropicMessage> = serde_json::from_value(json!([
            {"role":"assistant","content":[image]},
            {"role":"user","content":[image]}
        ]))
        .unwrap();
        assert_eq!(omit_anthropic_images(&mut messages), 1);
        assert!(matches!(
            &messages[0].content,
            AnthropicMessageContent::Blocks { content }
                if matches!(content[0], AnthropicContentBlock::Image { .. })
        ));
    }
}
