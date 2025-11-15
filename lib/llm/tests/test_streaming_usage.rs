// SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use async_trait::async_trait;
use dynamo_async_openai::types::{
    ChatCompletionRequestMessage, ChatCompletionRequestUserMessage,
    ChatCompletionRequestUserMessageContent, ChatCompletionStreamOptions,
    CreateChatCompletionRequest,
};
use dynamo_async_openai::types::{
    CompletionUsage as AoaiCompletionUsage, CreateCompletionRequestArgs, Prompt,
    PromptTokensDetails,
};
use dynamo_llm::preprocessor::OpenAIPreprocessor;
use dynamo_llm::protocols::common::llm_backend::{BackendOutput, FinishReason};
use dynamo_llm::protocols::openai::ParsingOptions;
use dynamo_llm::protocols::openai::chat_completions::{
    NvCreateChatCompletionRequest, aggregator::ChatCompletionAggregator,
};
use dynamo_llm::protocols::openai::completions::NvCreateCompletionRequest;
use dynamo_runtime::engine::{AsyncEngineContext, AsyncEngineStream};
use dynamo_runtime::protocols::annotated::Annotated;
use futures::StreamExt;
use futures::stream;
use std::pin::Pin;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

// Mock context for testing
#[derive(Debug)]
struct MockContext {
    id: String,
    stopped: AtomicBool,
    killed: AtomicBool,
}

impl MockContext {
    fn new() -> Self {
        Self {
            id: "test-request-123".to_string(),
            stopped: AtomicBool::new(false),
            killed: AtomicBool::new(false),
        }
    }
}

#[async_trait]
impl AsyncEngineContext for MockContext {
    fn id(&self) -> &str {
        &self.id
    }

    fn stop_generating(&self) {
        self.stopped.store(true, Ordering::SeqCst);
    }

    fn is_stopped(&self) -> bool {
        self.stopped.load(Ordering::SeqCst)
    }

    fn is_killed(&self) -> bool {
        self.killed.load(Ordering::SeqCst)
    }

    async fn stopped(&self) {
        // No-op for testing
    }

    async fn killed(&self) {
        // No-op for testing
    }

    fn stop(&self) {
        self.stopped.store(true, Ordering::SeqCst);
    }

    fn kill(&self) {
        self.killed.store(true, Ordering::SeqCst);
    }

    fn link_child(&self, _: Arc<dyn AsyncEngineContext>) {
        // No-op for testing
    }
}

/// Creates a mock stream of BackendOutput messages simulating a typical LLM response
fn create_mock_backend_stream(
    ctx: Arc<dyn AsyncEngineContext>,
) -> Pin<Box<dyn AsyncEngineStream<Annotated<BackendOutput>>>> {
    let outputs = build_backend_outputs_with_cached_tokens(None);

    let stream = stream::iter(outputs.into_iter().map(Annotated::from_data));

    use dynamo_runtime::engine::ResponseStream;
    ResponseStream::new(Box::pin(stream), ctx)
}

/// Build three backend outputs: "Hello", " world", "!" with optional cached_tokens on the final chunk
fn build_backend_outputs_with_cached_tokens(cached_tokens: Option<u32>) -> Vec<BackendOutput> {
    vec![
        BackendOutput {
            token_ids: vec![15339],
            tokens: vec![Some("Hello".to_string())],
            text: Some("Hello".to_string()),
            cum_log_probs: None,
            log_probs: None,
            top_logprobs: None,
            finish_reason: None,
            index: Some(0),
            completion_usage: None,
        },
        BackendOutput {
            token_ids: vec![1917],
            tokens: vec![Some(" world".to_string())],
            text: Some(" world".to_string()),
            cum_log_probs: None,
            log_probs: None,
            top_logprobs: None,
            finish_reason: None,
            index: Some(0),
            completion_usage: None,
        },
        BackendOutput {
            token_ids: vec![0],
            tokens: vec![Some("!".to_string())],
            text: Some("!".to_string()),
            cum_log_probs: None,
            log_probs: None,
            top_logprobs: None,
            finish_reason: Some(FinishReason::Stop),
            index: Some(0),
            completion_usage: cached_tokens.map(|ct| AoaiCompletionUsage {
                prompt_tokens: 0,
                completion_tokens: 0,
                total_tokens: 0,
                prompt_tokens_details: Some(PromptTokensDetails {
                    audio_tokens: None,
                    cached_tokens: Some(ct),
                }),
                completion_tokens_details: None,
            }),
        },
    ]
}

/// Create a backend stream from standard outputs with optional cached_tokens in the final chunk
fn create_backend_stream_with_cached_tokens(
    ctx: Arc<dyn AsyncEngineContext>,
    cached_tokens: Option<u32>,
) -> Pin<Box<dyn AsyncEngineStream<Annotated<BackendOutput>>>> {
    let outputs = build_backend_outputs_with_cached_tokens(cached_tokens);
    let stream = stream::iter(outputs.into_iter().map(Annotated::from_data));
    use dynamo_runtime::engine::ResponseStream;
    ResponseStream::new(Box::pin(stream), ctx)
}

/// Helper to create a chat completion request with optional stream_options
fn create_chat_request(include_usage: Option<bool>) -> NvCreateChatCompletionRequest {
    let messages = vec![ChatCompletionRequestMessage::User(
        ChatCompletionRequestUserMessage {
            content: ChatCompletionRequestUserMessageContent::Text("Hello".to_string()),
            name: None,
        },
    )];

    let stream_options = include_usage.map(|include| ChatCompletionStreamOptions {
        include_usage: include,
    });

    let inner = CreateChatCompletionRequest {
        model: "test-model".to_string(),
        messages,
        stream: Some(true),
        stream_options,
        ..Default::default()
    };

    NvCreateChatCompletionRequest {
        inner,
        common: Default::default(),
        nvext: None,
        chat_template_args: None,
        unsupported_fields: Default::default(),
    }
}

#[tokio::test]
async fn test_streaming_without_usage() {
    // Create request without stream_options (usage should not be included)
    let request = create_chat_request(None);
    let request_id = "test-123".to_string();
    let response_generator = Box::new(request.response_generator(request_id));

    // Create mock backend stream
    let ctx = Arc::new(MockContext::new());
    let backend_stream = create_mock_backend_stream(ctx.clone());

    // Transform the stream
    let transformed_stream = OpenAIPreprocessor::transform_postprocessor_stream(
        backend_stream,
        response_generator,
        ctx.clone(),
    );

    // Collect all chunks
    let chunks: Vec<_> = transformed_stream.collect().await;

    // Verify we got exactly 3 chunks (no extra usage chunk)
    assert_eq!(chunks.len(), 3, "Should have exactly 3 content chunks");

    // Verify all chunks have usage: None
    for (i, chunk) in chunks.iter().enumerate() {
        if let Some(response) = &chunk.data {
            assert!(
                response.usage.is_none(),
                "Chunk {} should have usage: None when stream_options not set",
                i
            );
            assert!(
                !response.choices.is_empty(),
                "Chunk {} should have choices",
                i
            );
        }
    }
}

#[tokio::test]
async fn test_streaming_with_usage_compliance() {
    // Create request with stream_options.include_usage = true
    let request = create_chat_request(Some(true));
    let request_id = "test-456".to_string();
    let response_generator = Box::new(request.response_generator(request_id));

    // Create mock backend stream
    let ctx = Arc::new(MockContext::new());
    let backend_stream = create_mock_backend_stream(ctx.clone());

    // Transform the stream
    let transformed_stream = OpenAIPreprocessor::transform_postprocessor_stream(
        backend_stream,
        response_generator,
        ctx.clone(),
    );

    // Collect all chunks
    let chunks: Vec<_> = transformed_stream.collect().await;

    // Verify we got 4 chunks (3 content + 1 usage)
    assert_eq!(
        chunks.len(),
        4,
        "Should have 3 content chunks + 1 usage chunk"
    );

    // Verify first 3 chunks have usage: None and non-empty choices
    for (i, chunk) in chunks.iter().take(3).enumerate() {
        if let Some(response) = &chunk.data {
            assert!(
                response.usage.is_none(),
                "Content chunk {} should have usage: None",
                i
            );
            assert!(
                !response.choices.is_empty(),
                "Content chunk {} should have choices",
                i
            );
        }
    }

    // Verify the final chunk is the usage-only chunk
    if let Some(final_response) = &chunks[3].data {
        assert!(
            final_response.choices.is_empty(),
            "Final usage chunk should have empty choices array"
        );
        assert!(
            final_response.usage.is_some(),
            "Final usage chunk should have usage statistics"
        );

        let usage = final_response.usage.as_ref().unwrap();
        assert_eq!(
            usage.completion_tokens, 3,
            "Should have 3 completion tokens"
        );
        assert_eq!(
            usage.prompt_tokens, 0,
            "Should have 0 prompt tokens (not set in test)"
        );
        assert_eq!(
            usage.total_tokens, 3,
            "Total tokens should be prompt + completion"
        );
    } else {
        panic!("Final chunk should be a valid response");
    }
}

#[tokio::test]
async fn test_streaming_with_usage_false() {
    // Create request with stream_options.include_usage = false (explicitly disabled)
    let request = create_chat_request(Some(false));
    let request_id = "test-789".to_string();
    let response_generator = Box::new(request.response_generator(request_id));

    // Create mock backend stream
    let ctx = Arc::new(MockContext::new());
    let backend_stream = create_mock_backend_stream(ctx.clone());

    // Transform the stream
    let transformed_stream = OpenAIPreprocessor::transform_postprocessor_stream(
        backend_stream,
        response_generator,
        ctx.clone(),
    );

    // Collect all chunks
    let chunks: Vec<_> = transformed_stream.collect().await;

    // Verify we got exactly 3 chunks (no extra usage chunk when explicitly false)
    assert_eq!(
        chunks.len(),
        3,
        "Should have exactly 3 content chunks when include_usage is false"
    );

    // Verify all chunks have usage: None
    for (i, chunk) in chunks.iter().enumerate() {
        if let Some(response) = &chunk.data {
            assert!(
                response.usage.is_none(),
                "Chunk {} should have usage: None when include_usage is false",
                i
            );
        }
    }
}

/// Helper to create a completion request with optional stream_options
fn create_cmpl_request(include_usage: Option<bool>, stream: bool) -> NvCreateCompletionRequest {
    let inner = {
        let mut builder = CreateCompletionRequestArgs::default();
        builder
            .model("test-model")
            .prompt(Prompt::String("Hello".to_string()))
            .stream(stream);
        if let Some(include) = include_usage {
            builder.stream_options(dynamo_async_openai::types::ChatCompletionStreamOptions {
                include_usage: include,
            });
        }
        builder.build().unwrap()
    };

    NvCreateCompletionRequest {
        inner,
        common: Default::default(),
        nvext: None,
        metadata: None,
        unsupported_fields: Default::default(),
    }
}

/// Helper to create a non-streaming chat completion request
fn create_nonstreaming_chat_request() -> NvCreateChatCompletionRequest {
    let messages = vec![ChatCompletionRequestMessage::User(
        ChatCompletionRequestUserMessage {
            content: ChatCompletionRequestUserMessageContent::Text("Hello".to_string()),
            name: None,
        },
    )];

    let inner = CreateChatCompletionRequest {
        model: "test-model".to_string(),
        messages,
        stream: Some(false),
        stream_options: None,
        ..Default::default()
    };

    NvCreateChatCompletionRequest {
        inner,
        common: Default::default(),
        nvext: None,
        chat_template_args: None,
        unsupported_fields: Default::default(),
    }
}

#[tokio::test]
async fn test_nonstreaming_has_usage_field() {
    let mut request = create_nonstreaming_chat_request();
    assert_eq!(
        request.inner.stream,
        Some(false),
        "Request should be non-streaming"
    );
    assert!(
        request.inner.stream_options.is_none(),
        "stream_options should not be set initially"
    );

    // Simulate what the preprocessor does for non-streaming requests
    let original_stream_flag = request.inner.stream.unwrap_or(false);

    // Enable usage for non-streaming requests
    request.enable_usage_for_nonstreaming(original_stream_flag);

    let request_id = "test-nonstream-123".to_string();
    let response_generator = Box::new(request.response_generator(request_id));

    // Create mock backend stream
    let ctx = Arc::new(MockContext::new());
    let backend_stream = create_mock_backend_stream(ctx.clone());

    // Transform the stream (this generates streaming chunks)
    let transformed_stream = OpenAIPreprocessor::transform_postprocessor_stream(
        backend_stream,
        response_generator,
        ctx.clone(),
    );

    // Aggregate the streaming chunks into a single non-streaming response
    // This simulates what the HTTP service does for non-streaming requests
    let result = dynamo_async_openai::types::CreateChatCompletionResponse::from_annotated_stream(
        transformed_stream,
        ParsingOptions::default(),
    )
    .await;

    assert!(result.is_ok(), "Aggregation should succeed");
    let response = result.unwrap();

    assert!(
        response.usage.is_some(),
        "Non-streaming chat completion response MUST have a usage field populated. \
         This is required for OpenAI API compliance."
    );

    let usage = response.usage.unwrap();

    // Verify usage contains valid token counts
    // In our mock, we generated 3 tokens (from the 3 backend outputs)
    assert_eq!(
        usage.completion_tokens, 3,
        "Completion tokens should match the number of tokens generated"
    );

    assert!(
        usage.total_tokens > 0,
        "Total tokens should be greater than 0"
    );

    assert_eq!(
        usage.total_tokens,
        usage.prompt_tokens + usage.completion_tokens,
        "Total tokens should equal prompt_tokens + completion_tokens"
    );
}

#[tokio::test]
async fn test_cmpl_streaming_with_usage_true_no_backend_usage() {
    // Completions: stream=true, include_usage=true, but backend does not send completion_usage
    let request = create_cmpl_request(Some(true), true);
    let request_id = "cmpl-usage-none-1".to_string();
    let response_generator = Box::new(request.response_generator(request_id));

    // Mock backend stream (no completion_usage in any chunk)
    let ctx = Arc::new(MockContext::new());
    let backend_stream = create_mock_backend_stream(ctx.clone());

    // Transform
    let transformed_stream = OpenAIPreprocessor::transform_postprocessor_stream(
        backend_stream,
        response_generator,
        ctx.clone(),
    );

    let chunks: Vec<_> = transformed_stream.collect().await;
    // Expect 3 content chunks + 1 usage-only chunk
    assert_eq!(chunks.len(), 4, "Should have 3 content + 1 usage chunk");

    // First 3 chunks: usage must be None
    for (i, chunk) in chunks.iter().take(3).enumerate() {
        if let Some(resp) = &chunk.data {
            assert!(
                resp.inner.usage.is_none(),
                "Content chunk {} should have usage: None",
                i
            );
            assert!(
                !resp.inner.choices.is_empty(),
                "Content chunk {} should have choices",
                i
            );
        }
    }

    // Final usage chunk: usage present with counts; prompt_tokens_details None (no backend usage)
    if let Some(final_resp) = &chunks[3].data {
        assert!(
            final_resp.inner.choices.is_empty(),
            "Usage-only chunk must have empty choices"
        );
        let usage = final_resp
            .inner
            .usage
            .as_ref()
            .expect("Usage must be present");
        assert_eq!(
            usage.completion_tokens, 3,
            "Aggregated completion tokens should be 3"
        );
        assert!(
            usage.prompt_tokens_details.is_none(),
            "prompt_tokens_details should be None when backend does not send usage"
        );
    } else {
        panic!("Final chunk should be present");
    }
}

#[tokio::test]
async fn test_cmpl_streaming_with_cached_tokens_propagation() {
    // Completions: include_usage=true, backend provides cached_tokens -> must propagate
    let request = create_cmpl_request(Some(true), true);
    let request_id = "cmpl-usage-cached-1".to_string();
    let mut response_generator = Box::new(request.response_generator(request_id));

    // Build a backend stream where the final chunk carries completion_usage with cached_tokens
    let ctx = Arc::new(MockContext::new());
    let backend_stream = create_backend_stream_with_cached_tokens(ctx.clone(), Some(7));

    // Align ISL so total usage gets computed correctly
    response_generator.update_isl(0);

    let transformed_stream = OpenAIPreprocessor::transform_postprocessor_stream(
        backend_stream,
        response_generator,
        ctx.clone(),
    );
    let chunks: Vec<_> = transformed_stream.collect().await;

    // Expect 4 chunks total
    assert_eq!(chunks.len(), 4, "Should have 3 content + 1 usage chunk");

    // Final usage chunk should include cached_tokens propagated
    if let Some(final_resp) = &chunks[3].data {
        let usage = final_resp
            .inner
            .usage
            .as_ref()
            .expect("Usage must be present on final chunk");
        let cached = usage
            .prompt_tokens_details
            .as_ref()
            .and_then(|d| d.cached_tokens);
        assert_eq!(
            cached,
            Some(7),
            "cached_tokens must propagate to final usage chunk"
        );
    } else {
        panic!("Final chunk should be present");
    }
}

#[tokio::test]
async fn test_chat_streaming_with_cached_tokens_propagation() {
    // Chat Completions: include_usage=true, backend provides cached_tokens -> must propagate
    let request = create_chat_request(Some(true));
    let request_id = "chat-usage-cached-1".to_string();
    let mut response_generator = Box::new(request.response_generator(request_id));

    let ctx = Arc::new(MockContext::new());
    let backend_stream = create_backend_stream_with_cached_tokens(ctx.clone(), Some(5));

    // Align ISL if needed
    response_generator.update_isl(0);

    let transformed_stream = OpenAIPreprocessor::transform_postprocessor_stream(
        backend_stream,
        response_generator,
        ctx.clone(),
    );
    let chunks: Vec<_> = transformed_stream.collect().await;

    assert_eq!(chunks.len(), 4, "Should have 3 content + 1 usage chunk");
    if let Some(final_resp) = &chunks[3].data {
        let usage = final_resp.usage.as_ref().expect("Usage must be present");
        let cached = usage
            .prompt_tokens_details
            .as_ref()
            .and_then(|d| d.cached_tokens);
        assert_eq!(
            cached,
            Some(5),
            "cached_tokens must propagate for chat completions"
        );
    } else {
        panic!("Final chunk should be present");
    }
}

#[tokio::test]
async fn test_cmpl_nonstreaming_has_usage_and_cached_tokens() {
    // Non-streaming completions must include usage in final aggregated response and propagate cached_tokens
    let mut request = create_cmpl_request(None, false);
    // Simulate preprocessor behavior for non-streaming
    let original_stream_flag = request.inner.stream.unwrap_or(false);
    request.enable_usage_for_nonstreaming(original_stream_flag);

    let request_id = "cmpl-nonstream-usage".to_string();
    let response_generator = Box::new(request.response_generator(request_id));

    // Mock backend stream with 3 chunks, last carries completion_usage with cached_tokens
    let ctx = Arc::new(MockContext::new());
    let backend_stream = create_backend_stream_with_cached_tokens(ctx.clone(), Some(9));

    // Transform to OpenAI completion stream
    let transformed_stream = OpenAIPreprocessor::transform_postprocessor_stream(
        backend_stream,
        response_generator,
        ctx.clone(),
    );

    // Aggregate into a single non-streaming response
    let parsing = ParsingOptions::default();
    let result =
        dynamo_llm::protocols::openai::completions::NvCreateCompletionResponse::from_annotated_stream(
            transformed_stream,
            parsing,
        )
        .await;
    assert!(result.is_ok(), "Aggregation should succeed");
    let resp = result.unwrap();
    let usage = resp
        .inner
        .usage
        .expect("usage must be present for non-streaming");
    assert_eq!(
        usage.completion_tokens, 3,
        "completion_tokens must aggregate"
    );
    let cached = usage.prompt_tokens_details.and_then(|d| d.cached_tokens);
    assert_eq!(
        cached,
        Some(9),
        "cached_tokens must propagate to non-streaming response"
    );
}
