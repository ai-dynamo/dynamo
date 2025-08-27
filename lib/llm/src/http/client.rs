// SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! HTTP clients for streaming LLM responses with performance recording
//!
//! This module provides HTTP clients that leverage async-openai with BYOT (Bring Your Own Types)
//! feature to work with OpenAI-compatible APIs. The clients support recording streaming responses
//! for performance analysis.

use std::pin::Pin;
use std::sync::{Arc, Mutex};
use std::task::{Context, Poll};
use std::time::Instant;

use async_trait::async_trait;
use derive_getters::Dissolve;
use dynamo_async_openai::{Client, config::OpenAIConfig, error::OpenAIError};
use futures::Stream;
use serde_json::Value;
use tokio_util::sync::CancellationToken;
use tracing;
use uuid::Uuid;

// Import our existing recording infrastructure
use crate::protocols::Annotated;
use crate::protocols::openai::chat_completions::{
    NvCreateChatCompletionRequest, NvCreateChatCompletionStreamResponse,
};
use dynamo_runtime::engine::{
    AsyncEngineContext, AsyncEngineContextProvider, AsyncEngineStream, Data, DataStream,
};

/// Configuration for HTTP clients
#[derive(Clone, Default)]
pub struct HttpClientConfig {
    /// OpenAI API configuration
    pub openai_config: OpenAIConfig,
    /// Whether to enable detailed logging
    pub verbose: bool,
}

/// Error types for HTTP clients
#[derive(Debug, thiserror::Error)]
pub enum HttpClientError {
    #[error("OpenAI API error: {0}")]
    OpenAI(#[from] OpenAIError),
    #[error("Request timeout")]
    Timeout,
    #[error("Request cancelled")]
    Cancelled,
    #[error("Invalid request: {0}")]
    InvalidRequest(String),
}

/// Context for HTTP client requests that supports cancellation
/// This bridges AsyncEngineContext and reqwest cancellation
#[derive(Clone)]
pub struct HttpRequestContext {
    /// Unique request identifier
    id: String,
    /// Tokio cancellation token for reqwest integration
    cancel_token: CancellationToken,
    /// When this context was created
    created_at: Instant,
    /// Whether the request has been stopped
    stopped: Arc<std::sync::atomic::AtomicBool>,
    /// Child contexts to be stopped if this is stopped
    child_context: Arc<Mutex<Vec<Arc<dyn AsyncEngineContext>>>>,
}

impl HttpRequestContext {
    /// Create a new HTTP request context
    pub fn new() -> Self {
        Self {
            id: Uuid::new_v4().to_string(),
            cancel_token: CancellationToken::new(),
            created_at: Instant::now(),
            stopped: Arc::new(std::sync::atomic::AtomicBool::new(false)),
            child_context: Arc::new(Mutex::new(Vec::new())),
        }
    }

    /// Create a new context with a specific ID
    pub fn with_id(id: String) -> Self {
        Self {
            id,
            cancel_token: CancellationToken::new(),
            created_at: Instant::now(),
            stopped: Arc::new(std::sync::atomic::AtomicBool::new(false)),
            child_context: Arc::new(Mutex::new(Vec::new())),
        }
    }

    /// Create a child context from this parent context
    /// The child will be cancelled when the parent is cancelled
    pub fn child(&self) -> Self {
        Self {
            id: Uuid::new_v4().to_string(),
            cancel_token: self.cancel_token.child_token(),
            created_at: Instant::now(),
            stopped: Arc::new(std::sync::atomic::AtomicBool::new(false)),
            child_context: Arc::new(Mutex::new(Vec::new())),
        }
    }

    /// Create a child context with a specific ID
    pub fn child_with_id(&self, id: String) -> Self {
        Self {
            id,
            cancel_token: self.cancel_token.child_token(),
            created_at: Instant::now(),
            stopped: Arc::new(std::sync::atomic::AtomicBool::new(false)),
            child_context: Arc::new(Mutex::new(Vec::new())),
        }
    }

    /// Get the cancellation token for use with reqwest
    pub fn cancellation_token(&self) -> CancellationToken {
        self.cancel_token.clone()
    }

    /// Get the elapsed time since context creation
    pub fn elapsed(&self) -> std::time::Duration {
        self.created_at.elapsed()
    }
}

impl Default for HttpRequestContext {
    fn default() -> Self {
        Self::new()
    }
}

impl std::fmt::Debug for HttpRequestContext {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("HttpRequestContext")
            .field("id", &self.id)
            .field("created_at", &self.created_at)
            .field("is_stopped", &self.is_stopped())
            .field("is_killed", &self.is_killed())
            .field("is_cancelled", &self.cancel_token.is_cancelled())
            .finish()
    }
}

#[async_trait]
impl AsyncEngineContext for HttpRequestContext {
    fn id(&self) -> &str {
        &self.id
    }

    fn stop(&self) {
        // Clone child Arcs to avoid deadlock if parent is accidentally linked under child
        let children = self
            .child_context
            .lock()
            .expect("Failed to lock child context")
            .iter()
            .cloned()
            .collect::<Vec<_>>();
        for child in children {
            child.stop();
        }

        self.stopped
            .store(true, std::sync::atomic::Ordering::Release);
        self.cancel_token.cancel();
    }

    fn stop_generating(&self) {
        // Clone child Arcs to avoid deadlock if parent is accidentally linked under child
        let children = self
            .child_context
            .lock()
            .expect("Failed to lock child context")
            .iter()
            .cloned()
            .collect::<Vec<_>>();
        for child in children {
            child.stop_generating();
        }

        // For HTTP clients, stop_generating is the same as stop
        self.stopped
            .store(true, std::sync::atomic::Ordering::Release);
        self.cancel_token.cancel();
    }

    fn kill(&self) {
        // Clone child Arcs to avoid deadlock if parent is accidentally linked under child
        let children = self
            .child_context
            .lock()
            .expect("Failed to lock child context")
            .iter()
            .cloned()
            .collect::<Vec<_>>();
        for child in children {
            child.kill();
        }

        self.stopped
            .store(true, std::sync::atomic::Ordering::Release);
        self.cancel_token.cancel();
    }

    fn is_stopped(&self) -> bool {
        self.stopped.load(std::sync::atomic::Ordering::Acquire)
    }

    fn is_killed(&self) -> bool {
        self.stopped.load(std::sync::atomic::Ordering::Acquire)
    }

    async fn stopped(&self) {
        self.cancel_token.cancelled().await;
    }

    async fn killed(&self) {
        // For HTTP clients, killed is the same as stopped
        self.cancel_token.cancelled().await;
    }

    fn link_child(&self, child: Arc<dyn AsyncEngineContext>) {
        self.child_context
            .lock()
            .expect("Failed to lock child context")
            .push(child);
    }
}

/// Base HTTP client with common functionality
pub struct BaseHttpClient {
    /// async-openai client
    client: Client<OpenAIConfig>,
    /// Client configuration
    config: HttpClientConfig,
    /// Root context for this client
    root_context: HttpRequestContext,
}

impl BaseHttpClient {
    /// Create a new base HTTP client
    pub fn new(config: HttpClientConfig) -> Self {
        let client = Client::with_config(config.openai_config.clone());
        Self {
            client,
            config,
            root_context: HttpRequestContext::new(),
        }
    }

    /// Get a reference to the underlying async-openai client
    pub fn client(&self) -> &Client<OpenAIConfig> {
        &self.client
    }

    /// Create a new request context as a child of the root context
    pub fn create_context(&self) -> HttpRequestContext {
        self.root_context.child()
    }

    /// Create a new request context with a specific ID as a child of the root context
    pub fn create_context_with_id(&self, id: String) -> HttpRequestContext {
        self.root_context.child_with_id(id)
    }

    /// Get the root context for this client
    pub fn root_context(&self) -> &HttpRequestContext {
        &self.root_context
    }

    /// Check if verbose logging is enabled
    pub fn is_verbose(&self) -> bool {
        self.config.verbose
    }
}

/// Type alias for NV chat response stream
pub type NvChatResponseStream =
    DataStream<Result<Annotated<NvCreateChatCompletionStreamResponse>, OpenAIError>>;

/// Type alias for generic BYOT response stream
pub type ByotResponseStream = DataStream<Result<Value, OpenAIError>>;

/// Type alias for pure OpenAI chat response stream
pub type OpenAIChatResponseStream =
    DataStream<Result<dynamo_async_openai::types::CreateChatCompletionStreamResponse, OpenAIError>>;

/// A wrapped HTTP response stream that combines a stream with its context
/// This provides a unified interface for HTTP client responses
#[derive(Dissolve)]
pub struct HttpResponseStream<T> {
    /// The underlying stream of responses
    pub stream: DataStream<T>,
    /// The context for this request
    pub context: Arc<dyn AsyncEngineContext>,
}

impl<T> HttpResponseStream<T> {
    /// Create a new HttpResponseStream
    pub fn new(stream: DataStream<T>, context: Arc<dyn AsyncEngineContext>) -> Self {
        Self { stream, context }
    }
}

impl<T: Data> Stream for HttpResponseStream<T> {
    type Item = T;

    fn poll_next(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        Pin::new(&mut self.stream).poll_next(cx)
    }
}

impl<T: Data> AsyncEngineContextProvider for HttpResponseStream<T> {
    fn context(&self) -> Arc<dyn AsyncEngineContext> {
        self.context.clone()
    }
}

impl<T: Data> HttpResponseStream<T> {
    /// Convert this HttpResponseStream to a Pin<Box<dyn AsyncEngineStream<T>>>
    /// This requires the stream to be Send + Sync, which may not be true for all streams
    pub fn into_async_engine_stream(self) -> Pin<Box<dyn AsyncEngineStream<T>>>
    where
        T: 'static,
    {
        // This will only work if the underlying stream is actually Send + Sync
        // For now, we create a wrapper that assumes this
        Box::pin(AsyncEngineStreamWrapper {
            stream: self.stream,
            context: self.context,
        })
    }
}

/// A wrapper that implements AsyncEngineStream for streams that are Send + Sync
struct AsyncEngineStreamWrapper<T> {
    stream: DataStream<T>,
    context: Arc<dyn AsyncEngineContext>,
}

impl<T: Data> Stream for AsyncEngineStreamWrapper<T> {
    type Item = T;

    fn poll_next(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        Pin::new(&mut self.stream).poll_next(cx)
    }
}

impl<T: Data> AsyncEngineContextProvider for AsyncEngineStreamWrapper<T> {
    fn context(&self) -> Arc<dyn AsyncEngineContext> {
        self.context.clone()
    }
}

impl<T: Data> AsyncEngineStream<T> for AsyncEngineStreamWrapper<T> {}

impl<T> std::fmt::Debug for AsyncEngineStreamWrapper<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("AsyncEngineStreamWrapper")
            .field("context", &self.context)
            .finish()
    }
}

impl<T: Data> std::fmt::Debug for HttpResponseStream<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("HttpResponseStream")
            .field("context", &self.context)
            .finish()
    }
}

/// Type alias for HttpResponseStream with NV chat completion responses
pub type NvHttpResponseStream =
    HttpResponseStream<Result<Annotated<NvCreateChatCompletionStreamResponse>, OpenAIError>>;

/// Type alias for HttpResponseStream with BYOT responses
pub type ByotHttpResponseStream = HttpResponseStream<Result<Value, OpenAIError>>;

/// Type alias for HttpResponseStream with pure OpenAI responses
pub type OpenAIHttpResponseStream = HttpResponseStream<
    Result<dynamo_async_openai::types::CreateChatCompletionStreamResponse, OpenAIError>,
>;

/// Pure OpenAI client using standard async-openai types
pub struct PureOpenAIClient {
    base: BaseHttpClient,
}

impl PureOpenAIClient {
    /// Create a new pure OpenAI client
    pub fn new(config: HttpClientConfig) -> Self {
        Self {
            base: BaseHttpClient::new(config),
        }
    }

    /// Create streaming chat completions using standard OpenAI types
    /// Uses a client-managed context
    pub async fn chat_stream(
        &self,
        request: dynamo_async_openai::types::CreateChatCompletionRequest,
    ) -> Result<OpenAIHttpResponseStream, HttpClientError> {
        let ctx = self.base.create_context();
        self.chat_stream_with_context(request, ctx).await
    }

    /// Create streaming chat completions with a custom context
    pub async fn chat_stream_with_context(
        &self,
        request: dynamo_async_openai::types::CreateChatCompletionRequest,
        context: HttpRequestContext,
    ) -> Result<OpenAIHttpResponseStream, HttpClientError> {
        let ctx_arc: Arc<dyn AsyncEngineContext> = Arc::new(context.clone());

        if !request.stream.unwrap_or(false) {
            return Err(HttpClientError::InvalidRequest(
                "chat_stream requires the request to have 'stream': true".to_string(),
            ));
        }

        if self.base.is_verbose() {
            tracing::info!(
                "Starting pure OpenAI chat stream for request {}",
                context.id()
            );
        }

        // Create the stream with cancellation support
        let stream = self
            .base
            .client()
            .chat()
            .create_stream(request)
            .await
            .map_err(HttpClientError::OpenAI)?;

        // TODO: In Phase 3, we'll add cancellation integration with reqwest
        // For now, return the stream as-is
        Ok(HttpResponseStream::new(stream, ctx_arc))
    }
}

/// NV Custom client using NvCreateChatCompletionRequest with Annotated responses
pub struct NvCustomClient {
    base: BaseHttpClient,
}

impl NvCustomClient {
    /// Create a new NV custom client
    pub fn new(config: HttpClientConfig) -> Self {
        Self {
            base: BaseHttpClient::new(config),
        }
    }

    /// Create streaming chat completions using NV custom types
    /// Uses a client-managed context
    pub async fn chat_stream(
        &self,
        request: NvCreateChatCompletionRequest,
    ) -> Result<NvHttpResponseStream, HttpClientError> {
        let ctx = self.base.create_context();
        self.chat_stream_with_context(request, ctx).await
    }

    /// Create streaming chat completions with a custom context
    pub async fn chat_stream_with_context(
        &self,
        request: NvCreateChatCompletionRequest,
        context: HttpRequestContext,
    ) -> Result<NvHttpResponseStream, HttpClientError> {
        let ctx_arc: Arc<dyn AsyncEngineContext> = Arc::new(context.clone());

        if !request.inner.stream.unwrap_or(false) {
            return Err(HttpClientError::InvalidRequest(
                "chat_stream requires the request to have 'stream': true".to_string(),
            ));
        }

        if self.base.is_verbose() {
            tracing::info!(
                "Starting NV custom chat stream for request {}",
                context.id()
            );
        }

        // Use BYOT feature to send NvCreateChatCompletionRequest
        // The stream type is explicitly specified to deserialize directly into Annotated<NvCreateChatCompletionStreamResponse>
        let stream = self
            .base
            .client()
            .chat()
            .create_stream_byot(request)
            .await
            .map_err(HttpClientError::OpenAI)?;

        Ok(HttpResponseStream::new(stream, ctx_arc))
    }
}

/// Generic BYOT client using serde_json::Value for maximum flexibility
pub struct GenericBYOTClient {
    base: BaseHttpClient,
}

impl GenericBYOTClient {
    /// Create a new generic BYOT client
    pub fn new(config: HttpClientConfig) -> Self {
        Self {
            base: BaseHttpClient::new(config),
        }
    }

    /// Create streaming chat completions using arbitrary JSON values
    /// Uses a client-managed context
    pub async fn chat_stream(
        &self,
        request: Value,
    ) -> Result<ByotHttpResponseStream, HttpClientError> {
        let ctx = self.base.create_context();
        self.chat_stream_with_context(request, ctx).await
    }

    /// Create streaming chat completions with a custom context
    pub async fn chat_stream_with_context(
        &self,
        request: Value,
        context: HttpRequestContext,
    ) -> Result<ByotHttpResponseStream, HttpClientError> {
        let ctx_arc: Arc<dyn AsyncEngineContext> = Arc::new(context.clone());

        if self.base.is_verbose() {
            tracing::info!(
                "Starting generic BYOT chat stream for request {}",
                context.id()
            );
        }

        // Validate that the request has stream: true
        if let Some(stream_val) = request.get("stream") {
            if !stream_val.as_bool().unwrap_or(false) {
                return Err(HttpClientError::InvalidRequest(
                    "Request must have 'stream': true for streaming".to_string(),
                ));
            }
        } else {
            return Err(HttpClientError::InvalidRequest(
                "Request must include 'stream' field".to_string(),
            ));
        }

        // Use BYOT feature with raw JSON
        // The stream type is explicitly specified to deserialize directly into serde_json::Value
        let stream = self
            .base
            .client()
            .chat()
            .create_stream_byot(request)
            .await
            .map_err(HttpClientError::OpenAI)?;

        Ok(HttpResponseStream::new(stream, ctx_arc))
    }
}

// TODO: Implement recording integration in Phase 3:
// - Recording wrapper functions
// - Capacity hints from request parameters
// - Integration with existing recording infrastructure

#[cfg(test)]
mod tests {
    use super::*;
    use tokio::time::{Duration, sleep};

    #[tokio::test]
    async fn test_http_request_context_creation() {
        let ctx = HttpRequestContext::new();
        assert!(!ctx.id().is_empty());
        assert!(!ctx.is_stopped());
        assert!(!ctx.is_killed());
    }

    #[tokio::test]
    async fn test_http_request_context_child() {
        let parent = HttpRequestContext::new();
        let child = parent.child();

        // Child should have different ID
        assert_ne!(parent.id(), child.id());

        // Child should not be stopped initially
        assert!(!child.is_stopped());

        // When parent is stopped, child should be cancelled via token
        parent.stop();
        assert!(parent.is_stopped());
        assert!(child.cancellation_token().is_cancelled());
    }

    #[tokio::test]
    async fn test_http_request_context_child_with_id() {
        let parent = HttpRequestContext::new();
        let child_id = "test-child";
        let child = parent.child_with_id(child_id.to_string());

        assert_eq!(child.id(), child_id);
        assert!(!child.is_stopped());

        // Test hierarchical cancellation
        parent.stop();
        assert!(child.cancellation_token().is_cancelled());
    }

    #[tokio::test]
    async fn test_http_request_context_cancellation() {
        let ctx = HttpRequestContext::new();
        let cancel_token = ctx.cancellation_token();

        // Test stop functionality
        assert!(!ctx.is_stopped());
        ctx.stop();
        assert!(ctx.is_stopped());
        assert!(cancel_token.is_cancelled());
    }

    #[tokio::test]
    async fn test_http_request_context_kill() {
        let ctx = HttpRequestContext::new();

        // Test kill functionality
        assert!(!ctx.is_killed());
        ctx.kill();
        assert!(ctx.is_killed());
        assert!(ctx.is_stopped());
    }

    #[tokio::test]
    async fn test_http_request_context_async_cancellation() {
        let ctx = HttpRequestContext::new();

        // Test async cancellation
        let ctx_clone = ctx.clone();
        let task = tokio::spawn(async move {
            ctx_clone.stopped().await;
        });

        // Give a moment for the task to start waiting
        sleep(Duration::from_millis(10)).await;

        // Cancel the context
        ctx.stop();

        // The task should complete
        task.await.unwrap();
    }

    #[test]
    fn test_base_http_client_creation() {
        let config = HttpClientConfig::default();
        let client = BaseHttpClient::new(config);
        assert!(!client.is_verbose());

        // Test that client has a root context
        assert!(!client.root_context().id().is_empty());
    }

    #[test]
    fn test_base_http_client_context_creation() {
        let config = HttpClientConfig::default();
        let client = BaseHttpClient::new(config);

        // Test creating child contexts
        let ctx1 = client.create_context();
        let ctx2 = client.create_context();

        // Should have different IDs
        assert_ne!(ctx1.id(), ctx2.id());

        // Should be children of root context
        client.root_context().stop();
        assert!(ctx1.cancellation_token().is_cancelled());
        assert!(ctx2.cancellation_token().is_cancelled());
    }

    #[test]
    fn test_base_http_client_context_with_id() {
        let config = HttpClientConfig::default();
        let client = BaseHttpClient::new(config);

        let custom_id = "custom-request-id";
        let ctx = client.create_context_with_id(custom_id.to_string());

        assert_eq!(ctx.id(), custom_id);

        // Should still be child of root
        client.root_context().stop();
        assert!(ctx.cancellation_token().is_cancelled());
    }

    #[test]
    fn test_http_client_config_defaults() {
        let config = HttpClientConfig::default();
        assert!(!config.verbose);
    }

    #[test]
    fn test_pure_openai_client_creation() {
        let config = HttpClientConfig::default();
        let _client = PureOpenAIClient::new(config);
        // If we get here, creation succeeded
    }

    #[test]
    fn test_nv_custom_client_creation() {
        let config = HttpClientConfig::default();
        let _client = NvCustomClient::new(config);
        // If we get here, creation succeeded
    }

    #[test]
    fn test_generic_byot_client_creation() {
        let config = HttpClientConfig::default();
        let _client = GenericBYOTClient::new(config);
        // If we get here, creation succeeded
    }
}
