// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use dynamo_async_openai::types::{ChatChoiceStream, ChatCompletionStreamResponseDelta, Role};
use dynamo_llm::preprocessor::OpenAIPreprocessor;
use dynamo_llm::protocols::openai::chat_completions::NvCreateChatCompletionStreamResponse;
use dynamo_runtime::protocols::annotated::Annotated;
use futures::{StreamExt, stream};

#[cfg(test)]
mod tests {
    use super::*;

    /// Helper function to create a mock chat response chunk
    fn create_mock_response_chunk(
        content: String,
        reasoning_content: Option<String>,
    ) -> Annotated<NvCreateChatCompletionStreamResponse> {
        #[allow(deprecated)]
        let choice = ChatChoiceStream {
            index: 0,
            delta: ChatCompletionStreamResponseDelta {
                role: Some(Role::Assistant),
                content: Some(content),
                tool_calls: None,
                function_call: None,
                refusal: None,
                reasoning_content,
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
            id: Some("test-id".to_string()),
            data: Some(response),
            event: None,
            comment: None,
        }
    }

    #[tokio::test]
    async fn test_reasoning_parser_passthrough_when_no_parser_configured() {
        // Create a mock runtime config without reasoning parser
        let runtime_config = dynamo_llm::local_model::runtime_config::ModelRuntimeConfig {
            reasoning_parser: None,
            ..Default::default()
        };

        // Create test input stream with some content
        let input_chunks = vec![
            create_mock_response_chunk("Hello ".to_string(), None),
            create_mock_response_chunk("world!".to_string(), None),
        ];
        let input_stream = stream::iter(input_chunks.clone());

        // Apply the reasoning parser transformation
        let output_stream =
            OpenAIPreprocessor::parse_reasoning_content_from_stream(input_stream, runtime_config);

        // Pin the stream and collect all output chunks
        let mut output_stream = std::pin::pin!(output_stream);
        let mut output_chunks = Vec::new();
        while let Some(chunk) = output_stream.next().await {
            output_chunks.push(chunk);
        }

        // Verify that the output is identical to input when no parser is configured
        assert_eq!(output_chunks.len(), input_chunks.len());

        for (input, output) in input_chunks.iter().zip(output_chunks.iter()) {
            assert_eq!(
                input.data.as_ref().unwrap().choices[0].delta.content,
                output.data.as_ref().unwrap().choices[0].delta.content
            );
            assert_eq!(
                input.data.as_ref().unwrap().choices[0]
                    .delta
                    .reasoning_content,
                output.data.as_ref().unwrap().choices[0]
                    .delta
                    .reasoning_content
            );
        }
    }

    #[tokio::test]
    async fn test_reasoning_parser_with_basic_parser() {
        // Create a mock runtime config with basic reasoning parser
        let runtime_config = dynamo_llm::local_model::runtime_config::ModelRuntimeConfig {
            reasoning_parser: Some("basic".to_string()),
            ..Default::default()
        };

        // Create test input stream with reasoning content
        let input_chunks = vec![create_mock_response_chunk(
            "<think>This is reasoning content</think> Here's my answer.".to_string(),
            None,
        )];
        let input_stream = stream::iter(input_chunks);

        // Apply the reasoning parser transformation
        let output_stream =
            OpenAIPreprocessor::parse_reasoning_content_from_stream(input_stream, runtime_config);

        // Pin the stream and collect all output chunks
        let mut output_stream = std::pin::pin!(output_stream);
        let mut output_chunks = Vec::new();
        while let Some(chunk) = output_stream.next().await {
            output_chunks.push(chunk);
        }

        // Verify that reasoning content was parsed correctly
        assert_eq!(output_chunks.len(), 1);
        let output_choice = &output_chunks[0].data.as_ref().unwrap().choices[0];

        // The basic parser should extract reasoning content from <think> tags
        // and separate it from normal content
        assert!(
            output_choice.delta.reasoning_content.is_some(),
            "Reasoning content should be extracted"
        );
        assert!(
            output_choice.delta.content.is_some(),
            "Normal content should be present"
        );

        let reasoning_content = output_choice.delta.reasoning_content.as_ref().unwrap();
        let normal_content = output_choice.delta.content.as_ref().unwrap();

        // Verify the actual content was parsed correctly
        assert_eq!(
            reasoning_content, "This is reasoning content",
            "Reasoning content should be extracted without tags"
        );
        assert_eq!(
            normal_content, " Here's my answer.",
            "Normal content should be the text outside reasoning tags"
        );
    }

    #[tokio::test]
    async fn test_reasoning_parser_with_only_reasoning_content() {
        // Create a mock runtime config with basic reasoning parser
        let runtime_config = dynamo_llm::local_model::runtime_config::ModelRuntimeConfig {
            reasoning_parser: Some("basic".to_string()),
            ..Default::default()
        };

        // Create test input stream with only reasoning content
        let input_chunks = vec![create_mock_response_chunk(
            "<think>Only reasoning here</think>".to_string(),
            None,
        )];
        let input_stream = stream::iter(input_chunks);

        // Apply the reasoning parser transformation
        let output_stream =
            OpenAIPreprocessor::parse_reasoning_content_from_stream(input_stream, runtime_config);

        // Pin the stream and collect all output chunks
        let mut output_stream = std::pin::pin!(output_stream);
        let mut output_chunks = Vec::new();
        while let Some(chunk) = output_stream.next().await {
            output_chunks.push(chunk);
        }

        // Verify that only reasoning content was parsed
        assert_eq!(output_chunks.len(), 1);
        let output_choice = &output_chunks[0].data.as_ref().unwrap().choices[0];

        assert!(
            output_choice.delta.reasoning_content.is_some(),
            "Reasoning content should be extracted"
        );
        let reasoning_content = output_choice.delta.reasoning_content.as_ref().unwrap();
        assert_eq!(
            reasoning_content, "Only reasoning here",
            "Reasoning content should be extracted without tags"
        );

        // Normal content should be None or empty when there's no text outside reasoning tags
        if let Some(normal_content) = &output_choice.delta.content {
            assert!(
                normal_content.is_empty(),
                "Normal content should be empty when there's no text outside reasoning tags"
            );
        }
    }

    #[tokio::test]
    async fn test_reasoning_parser_with_only_normal_content() {
        // Create a mock runtime config with basic reasoning parser
        let runtime_config = dynamo_llm::local_model::runtime_config::ModelRuntimeConfig {
            reasoning_parser: Some("basic".to_string()),
            ..Default::default()
        };

        // Create test input stream with only normal content (no reasoning tags)
        let input_chunks = vec![create_mock_response_chunk(
            "Just normal text without reasoning tags.".to_string(),
            None,
        )];
        let input_stream = stream::iter(input_chunks);

        // Apply the reasoning parser transformation
        let output_stream =
            OpenAIPreprocessor::parse_reasoning_content_from_stream(input_stream, runtime_config);

        // Pin the stream and collect all output chunks
        let mut output_stream = std::pin::pin!(output_stream);
        let mut output_chunks = Vec::new();
        while let Some(chunk) = output_stream.next().await {
            output_chunks.push(chunk);
        }

        // Verify that only normal content is present
        assert_eq!(output_chunks.len(), 1);
        let output_choice = &output_chunks[0].data.as_ref().unwrap().choices[0];

        assert!(
            output_choice.delta.content.is_some(),
            "Normal content should be present"
        );
        let normal_content = output_choice.delta.content.as_ref().unwrap();
        assert_eq!(
            normal_content, "Just normal text without reasoning tags.",
            "Normal content should be preserved"
        );

        // Reasoning content should be None when there are no reasoning tags
        if let Some(reasoning_content) = &output_choice.delta.reasoning_content {
            assert!(
                reasoning_content.is_empty(),
                "Reasoning content should be empty when there are no reasoning tags"
            );
        }
    }

    #[tokio::test]
    async fn test_reasoning_parser_with_invalid_parser_name() {
        // Create a mock runtime config with invalid reasoning parser
        let runtime_config = dynamo_llm::local_model::runtime_config::ModelRuntimeConfig {
            reasoning_parser: Some("invalid_parser_name".to_string()),
            ..Default::default()
        };

        // Create test input stream
        let input_chunks = vec![create_mock_response_chunk("Hello world!".to_string(), None)];
        let input_stream = stream::iter(input_chunks.clone());

        // Apply the reasoning parser transformation
        let output_stream =
            OpenAIPreprocessor::parse_reasoning_content_from_stream(input_stream, runtime_config);

        // Pin the stream and collect all output chunks
        let mut output_stream = std::pin::pin!(output_stream);
        let mut output_chunks = Vec::new();
        while let Some(chunk) = output_stream.next().await {
            output_chunks.push(chunk);
        }

        // Verify that invalid parser name results in passthrough behavior
        assert_eq!(output_chunks.len(), input_chunks.len());

        for (input, output) in input_chunks.iter().zip(output_chunks.iter()) {
            assert_eq!(
                input.data.as_ref().unwrap().choices[0].delta.content,
                output.data.as_ref().unwrap().choices[0].delta.content
            );
        }
    }
}
