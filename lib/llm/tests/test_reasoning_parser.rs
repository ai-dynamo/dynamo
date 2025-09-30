// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use dynamo_async_openai::types::{ChatChoiceStream, ChatCompletionStreamResponseDelta, Role};
use dynamo_llm::preprocessor::OpenAIPreprocessor;
use dynamo_llm::protocols::openai::chat_completions::NvCreateChatCompletionStreamResponse;
use dynamo_runtime::protocols::annotated::Annotated;
use futures::{StreamExt, stream};

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

#[cfg(test)]
mod tests {
    use super::*;

    /// Helper function to assert choice content and reasoning content
    fn assert_choice(
        choice: &ChatChoiceStream,
        expected_content: Option<&str>,
        expected_reasoning_content: Option<&str>,
    ) {
        match expected_content {
            Some(expected) => {
                assert_eq!(
                    choice.delta.content.as_deref(),
                    Some(expected),
                    "Content mismatch"
                );
            }
            None => {
                assert!(
                    choice.delta.content.is_none()
                        || choice.delta.content.as_ref().unwrap().is_empty(),
                    "Expected content to be None or empty, got: {:?}",
                    choice.delta.content
                );
            }
        }

        match expected_reasoning_content {
            Some(expected) => {
                assert_eq!(
                    choice.delta.reasoning_content.as_deref(),
                    Some(expected),
                    "Reasoning content mismatch"
                );
            }
            None => {
                assert!(
                    choice.delta.reasoning_content.is_none(),
                    "Expected reasoning content to be None, got: {:?}",
                    choice.delta.reasoning_content
                );
            }
        }
    }

    #[tokio::test]
    async fn test_reasoning_parser_with_basic_parser() {
        // Basic Parser test <think> </think> tags
        // <think> This is reasoning content </think> Here's my answer.
        // content: Here's my answer.
        // reasoning_content: This is reasoning content

        // Create a mock runtime config with basic reasoning parser
        let runtime_config = dynamo_llm::local_model::runtime_config::ModelRuntimeConfig {
            reasoning_parser: Some("basic".to_string()),
            ..Default::default()
        };

        // Create test input stream with reasoning content
        let input_chunks = vec![
            create_mock_response_chunk("<think>This".to_string(), None),
            create_mock_response_chunk(" is reasoning content".to_string(), None),
            create_mock_response_chunk("</think> Here's my answer.".to_string(), None),
        ];
        let input_stream = stream::iter(input_chunks);

        // Apply the reasoning parser transformation
        let output_stream = OpenAIPreprocessor::parse_reasoning_content_from_stream(
            input_stream,
            runtime_config.reasoning_parser,
        );

        // Pin the stream and collect all output chunks
        let mut output_stream = std::pin::pin!(output_stream);
        let mut output_chunks = Vec::new();
        while let Some(chunk) = output_stream.next().await {
            output_chunks.push(chunk);
        }

        // Verify that reasoning content was parsed correctly
        assert_eq!(output_chunks.len(), 3);

        // Chunk 0: "<think>This"
        let output_choice_0 = &output_chunks[0].data.as_ref().unwrap().choices[0];
        assert_choice(output_choice_0, None, Some("This"));

        // Chunk 1: " is reasoning content"
        let output_choice_1 = &output_chunks[1].data.as_ref().unwrap().choices[0];
        assert_choice(output_choice_1, None, Some(" is reasoning content"));

        // Chunk 2: "</think> Here's my answer."
        let output_choice_2 = &output_chunks[2].data.as_ref().unwrap().choices[0];
        assert_choice(output_choice_2, Some(" Here's my answer."), None);
    }

    #[tokio::test]
    async fn test_reasoning_parser_with_only_reasoning_content() {
        // Create a mock runtime config with basic reasoning parser
        let runtime_config = dynamo_llm::local_model::runtime_config::ModelRuntimeConfig {
            reasoning_parser: Some("basic".to_string()),
            ..Default::default()
        };

        // Create test input stream with only reasoning content
        let input_chunks = vec![
            create_mock_response_chunk("<think>Only".to_string(), None),
            create_mock_response_chunk(" reasoning".to_string(), None),
            create_mock_response_chunk(" here</think>".to_string(), None),
        ];
        let input_stream = stream::iter(input_chunks);

        // Apply the reasoning parser transformation
        let output_stream = OpenAIPreprocessor::parse_reasoning_content_from_stream(
            input_stream,
            runtime_config.reasoning_parser,
        );

        // Pin the stream and collect all output chunks
        let mut output_stream = std::pin::pin!(output_stream);
        let mut output_chunks = Vec::new();
        while let Some(chunk) = output_stream.next().await {
            output_chunks.push(chunk);
        }

        // Verify that reasoning content was parsed correctly across three chunks
        assert_eq!(output_chunks.len(), 3);

        // Chunk 0: "<think>Only"
        let output_choice_0 = &output_chunks[0].data.as_ref().unwrap().choices[0];
        assert_choice(output_choice_0, None, Some("Only"));

        // Chunk 1: " reasoning"
        let output_choice_1 = &output_chunks[1].data.as_ref().unwrap().choices[0];
        assert_choice(output_choice_1, None, Some(" reasoning"));

        // Chunk 2: " here</think>"
        let output_choice_2 = &output_chunks[2].data.as_ref().unwrap().choices[0];
        assert_choice(output_choice_2, None, Some(" here"));
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
        let output_stream = OpenAIPreprocessor::parse_reasoning_content_from_stream(
            input_stream,
            runtime_config.reasoning_parser,
        );

        // Pin the stream and collect all output chunks
        let mut output_stream = std::pin::pin!(output_stream);
        let mut output_chunks = Vec::new();
        while let Some(chunk) = output_stream.next().await {
            output_chunks.push(chunk);
        }

        // Verify that only normal content is present
        assert_eq!(output_chunks.len(), 1);
        let output_choice = &output_chunks[0].data.as_ref().unwrap().choices[0];
        assert_choice(
            output_choice,
            Some("Just normal text without reasoning tags."),
            None,
        );
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
        let output_stream = OpenAIPreprocessor::parse_reasoning_content_from_stream(
            input_stream,
            runtime_config.reasoning_parser,
        );

        // Pin the stream and collect all output chunks
        let mut output_stream = std::pin::pin!(output_stream);
        let mut output_chunks = Vec::new();
        while let Some(chunk) = output_stream.next().await {
            output_chunks.push(chunk);
        }

        // Verify that invalid parser name results in passthrough behavior
        assert_eq!(output_chunks.len(), input_chunks.len());

        for (input, output) in input_chunks.iter().zip(output_chunks.iter()) {
            let input_choice = &input.data.as_ref().unwrap().choices[0];
            let output_choice = &output.data.as_ref().unwrap().choices[0];
            assert_choice(
                output_choice,
                input_choice.delta.content.as_deref(),
                input_choice.delta.reasoning_content.as_deref(),
            );
        }
    }

    #[tokio::test]
    async fn test_reasoning_parser_with_mistral_parser() {
        // Create a mock runtime config with mistral reasoning parser
        let runtime_config = dynamo_llm::local_model::runtime_config::ModelRuntimeConfig {
            reasoning_parser: Some("mistral".to_string()),
            ..Default::default()
        };

        // Create test input stream with Mistral-style reasoning tags
        let input_chunks = vec![create_mock_response_chunk(
            "Let me think. [THINK]This is Mistral reasoning[/THINK] Here's my answer.".to_string(),
            None,
        )];
        let input_stream = stream::iter(input_chunks);

        // Apply the reasoning parser transformation
        let output_stream = OpenAIPreprocessor::parse_reasoning_content_from_stream(
            input_stream,
            runtime_config.reasoning_parser,
        );

        // Pin the stream and collect all output chunks
        let mut output_stream = std::pin::pin!(output_stream);
        let mut output_chunks = Vec::new();
        while let Some(chunk) = output_stream.next().await {
            output_chunks.push(chunk);
        }

        // Verify that Mistral-style reasoning is parsed correctly
        assert_eq!(output_chunks.len(), 1);
        let output_choice = &output_chunks[0].data.as_ref().unwrap().choices[0];

        assert!(
            output_choice.delta.reasoning_content.is_some(),
            "Should extract Mistral reasoning content"
        );
        assert!(
            output_choice.delta.content.is_some(),
            "Should have normal content"
        );

        let reasoning_content = output_choice.delta.reasoning_content.as_ref().unwrap();
        let normal_content = output_choice.delta.content.as_ref().unwrap();

        // Verify the content was parsed with Mistral tags
        assert!(
            reasoning_content.contains("Mistral reasoning"),
            "Should contain Mistral reasoning content"
        );
        assert!(
            normal_content.contains("Let me think") || normal_content.contains("Here's my answer"),
            "Should contain normal content"
        );
    }

    #[tokio::test]
    #[ignore]
    async fn test_reasoning_parser_with_gpt_oss_parser() {
        // Create a mock runtime config with gpt-oss reasoning parser
        let runtime_config = dynamo_llm::local_model::runtime_config::ModelRuntimeConfig {
            reasoning_parser: Some("gpt_oss".to_string()),
            ..Default::default()
        };

        // Create test input stream with GPT-OSS style reasoning
        // Note: GPT-OSS parser may have different tag formats or behavior
        let input_chunks = vec![
            create_mock_response_chunk("I need to think about this carefully. <thinking>This is GPT-OSS reasoning content that might use different markers or patterns.</thinking> Based on my analysis, here's the answer.".to_string(), None),
        ];
        let input_stream = stream::iter(input_chunks);

        // Apply the reasoning parser transformation
        let output_stream = OpenAIPreprocessor::parse_reasoning_content_from_stream(
            input_stream,
            runtime_config.reasoning_parser,
        );

        // Pin the stream and collect all output chunks
        let mut output_stream = std::pin::pin!(output_stream);
        let mut output_chunks = Vec::new();
        while let Some(chunk) = output_stream.next().await {
            output_chunks.push(chunk);
        }

        // Verify that GPT-OSS reasoning is parsed correctly
        assert_eq!(output_chunks.len(), 1);
        let output_choice = &output_chunks[0].data.as_ref().unwrap().choices[0];

        // GPT-OSS parser should extract reasoning content
        assert!(
            output_choice.delta.reasoning_content.is_some(),
            "Should extract GPT-OSS reasoning content"
        );
        assert!(
            output_choice.delta.content.is_some(),
            "Should have normal content"
        );

        let reasoning_content = output_choice.delta.reasoning_content.as_ref().unwrap();
        let normal_content = output_choice.delta.content.as_ref().unwrap();

        // Verify the content was parsed correctly
        assert!(
            !reasoning_content.is_empty(),
            "GPT-OSS reasoning content should not be empty"
        );
        assert!(
            !normal_content.is_empty(),
            "Normal content should not be empty"
        );

        // Log the results for debugging
        println!("GPT-OSS Normal content: {:?}", normal_content);
        println!("GPT-OSS Reasoning content: {:?}", reasoning_content);
    }

    #[tokio::test]
    async fn test_reasoning_parser_with_kimi_parser() {
        // Create a mock runtime config with Kimi reasoning parser
        let runtime_config = dynamo_llm::local_model::runtime_config::ModelRuntimeConfig {
            reasoning_parser: Some("kimi".to_string()),
            ..Default::default()
        };

        // Create test input stream with Kimi-style reasoning tags
        let input_chunks = vec![
            create_mock_response_chunk("Let me analyze this. ◁think▷This is Kimi reasoning content◁/think▷ Here's my conclusion.".to_string(), None),
        ];
        let input_stream = stream::iter(input_chunks);

        // Apply the reasoning parser transformation
        let output_stream = OpenAIPreprocessor::parse_reasoning_content_from_stream(
            input_stream,
            runtime_config.reasoning_parser,
        );

        // Pin the stream and collect all output chunks
        let mut output_stream = std::pin::pin!(output_stream);
        let mut output_chunks = Vec::new();
        while let Some(chunk) = output_stream.next().await {
            output_chunks.push(chunk);
        }

        // Verify that Kimi-style reasoning is parsed correctly
        assert_eq!(output_chunks.len(), 1);
        let output_choice = &output_chunks[0].data.as_ref().unwrap().choices[0];

        assert!(
            output_choice.delta.reasoning_content.is_some(),
            "Should extract Kimi reasoning content"
        );
        assert!(
            output_choice.delta.content.is_some(),
            "Should have normal content"
        );

        let reasoning_content = output_choice.delta.reasoning_content.as_ref().unwrap();
        let normal_content = output_choice.delta.content.as_ref().unwrap();

        // Verify the content was parsed with Kimi tags
        assert!(
            reasoning_content.contains("Kimi reasoning"),
            "Should contain Kimi reasoning content"
        );
        assert!(
            normal_content.contains("Let me analyze")
                || normal_content.contains("Here's my conclusion"),
            "Should contain normal content"
        );
    }
}
