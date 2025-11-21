// SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Clean, builder-based handler API for active message patterns.
//!
//! This module provides a unified builder API with configurable dispatch modes using GATs
//! to avoid async_trait overhead.
//!
//! ## Handler Types
//!
//! ### Active Message Handlers
//! - **`am_handler()`** - Sync AM handler: `Fn(Context) -> anyhow::Result<()>`
//! - **`am_handler_async()`** - Async AM handler: `Fn(Context) -> Future<anyhow::Result<()>>`
//!   - Return value is for **internal error handling only** (not sent to sender)
//!   - Handler can send 0 or N new active messages as needed
//!
//! ### Request-Response Handlers
//! - **`unary_handler()`** - Sync unary: `Fn(Context) -> UnifiedResponse`
//! - **`unary_handler_async()`** - Async unary: `Fn(Context) -> Future<UnifiedResponse>`
//!   - Return value **IS sent back to sender** as response
//!   - `Ok(None)` = ACK, `Ok(Some(bytes))` = Response, `Err(...)` = Error
//!
//! ### Typed Request-Response Handlers
//! - **`typed_unary()`** - Sync typed: `Fn(TypedContext<I>) -> anyhow::Result<O>`
//! - **`typed_unary_async()`** - Async typed: `Fn(TypedContext<I>) -> Future<anyhow::Result<O>>`
//!   - Automatic JSON serialization/deserialization
//!   - Return value **IS sent back to sender** (auto-serialized)
//!   - `Ok(output)` = Response, `Err(...)` = Error
//!
//! ## Builder API
//!
//! All handlers support configurable dispatch modes via builder pattern:
//!
//! ```ignore
//! // Default spawn mode (async execution)
//! let handler = am_handler("logger", |ctx| {
//!     println!("Log: {}", String::from_utf8_lossy(&ctx.payload));
//!     Ok(())
//! }).build();
//!
//! // Inline mode (minimal latency, use only for fast handlers)
//! let handler = typed_unary("ping", |ctx| {
//!     Ok(PingResponse { timestamp: now() })
//! }).inline().build();
//!
//! // Async handlers work the same way
//! let handler = typed_unary_async("fetch_data", |ctx| async move {
//!     let data = database.query(&ctx.input.query).await?;
//!     Ok(DataResponse { data })
//! }).spawn().build();
//! ```
//!
//! ## Context Objects
//!
//! All context objects include:
//! - **`message_id: MessageId`** - Unique, compact identifier for this message (Base58 encoded)
//! - **`nova: Arc<Nova>`** - The top-level Nova API for sending messages, querying handlers, etc.
//!
//! ### Context Types
//! - **`Context`** - For AM and unary handlers (has `message_id`, `payload`, `nova`)
//! - **`TypedContext<I>`** - For typed handlers (has `message_id`, `input`, `nova`)

mod manager;
pub(crate) use manager::HandlerManager;

pub(crate) mod events;
pub(crate) use events::NovaEvents;

use anyhow::Result;
use bytes::Bytes;
use futures::future::{BoxFuture, Ready, ready};
use std::future::Future;
use std::marker::PhantomData;
use std::pin::Pin;
use std::sync::Arc;
use tracing::{debug, error};

use super::server::dispatcher::{
    ActiveMessageDispatcher, ActiveMessageHandler, ControlMessage, HandlerContext,
    InlineDispatcher, SpawnedDispatcher,
};
use crate::am::common::events::{EventType, Outcome, encode_event_header};
use crate::am::common::messages::ResponseType;
use crate::am::common::responses::{ResponseId, encode_response_header};
use derive_getters::Dissolve;
use dynamo_identity::WorkerId;
use dynamo_nova_backend::{MessageType, NovaBackend};
use tokio_util::task::TaskTracker;

// ============================================================================
// Opaque Handles
// ============================================================================

pub struct NovaHandler {
    dispatcher: Arc<dyn ActiveMessageDispatcher>,
}

impl NovaHandler {
    pub fn name(&self) -> &str {
        self.dispatcher.as_ref().name()
    }

    /// Create a synchronous active message handler
    ///
    /// **Handler signature:** `Fn(Context) -> anyhow::Result<()>`
    ///
    /// The return value is for internal error handling only and is NOT sent to the sender.
    pub fn am_handler<F>(
        name: impl Into<String>,
        f: F,
    ) -> AmHandlerBuilder<SyncExecutor<F, Context, ()>>
    where
        F: Fn(Context) -> Result<()> + Send + Sync + 'static,
    {
        am_handler(name, f)
    }

    /// Create an asynchronous active message handler
    ///
    /// **Handler signature:** `Fn(Context) -> impl Future<Output = anyhow::Result<()>>`
    ///
    /// The return value is for internal error handling only and is NOT sent to the sender.
    pub fn am_handler_async<F, Fut>(
        name: impl Into<String>,
        f: F,
    ) -> AmHandlerBuilder<AsyncExecutor<F, Context, ()>>
    where
        F: Fn(Context) -> Fut + Send + Sync + 'static,
        Fut: Future<Output = Result<()>> + Send + 'static,
    {
        am_handler_async(name, f)
    }

    /// Create a synchronous unary (request-response) handler
    ///
    /// **Handler signature:** `Fn(Context) -> anyhow::Result<Option<Bytes>>`
    ///
    /// The return value IS sent back to the sender.
    pub fn unary_handler<F>(
        name: impl Into<String>,
        f: F,
    ) -> UnaryHandlerBuilder<SyncExecutor<F, Context, Option<Bytes>>>
    where
        F: Fn(Context) -> UnifiedResponse + Send + Sync + 'static,
    {
        unary_handler(name, f)
    }

    /// Create an asynchronous unary (request-response) handler
    ///
    /// **Handler signature:** `Fn(Context) -> impl Future<Output = anyhow::Result<Option<Bytes>>>`
    ///
    /// The return value IS sent back to the sender.
    pub fn unary_handler_async<F, Fut>(
        name: impl Into<String>,
        f: F,
    ) -> UnaryHandlerBuilder<AsyncExecutor<F, Context, Option<Bytes>>>
    where
        F: Fn(Context) -> Fut + Send + Sync + 'static,
        Fut: Future<Output = UnifiedResponse> + Send + 'static,
    {
        unary_handler_async(name, f)
    }

    /// Create a synchronous typed unary handler with automatic serialization
    ///
    /// **Handler signature:** `Fn(TypedContext<I>) -> anyhow::Result<O>`
    ///
    /// Input is automatically deserialized from JSON.
    /// Output is automatically serialized to JSON.
    pub fn typed_unary<I, O, F>(
        name: impl Into<String>,
        f: F,
    ) -> TypedUnaryHandlerBuilder<SyncExecutor<F, TypedContext<I>, O>, I, O>
    where
        I: serde::de::DeserializeOwned + Send + Sync + 'static,
        O: serde::Serialize + Send + Sync + 'static,
        F: Fn(TypedContext<I>) -> Result<O> + Send + Sync + 'static,
    {
        typed_unary(name, f)
    }

    /// Create an asynchronous typed unary handler with automatic serialization
    ///
    /// **Handler signature:** `Fn(TypedContext<I>) -> impl Future<Output = anyhow::Result<O>>`
    ///
    /// Input is automatically deserialized from JSON.
    /// Output is automatically serialized to JSON.
    pub fn typed_unary_async<I, O, F, Fut>(
        name: impl Into<String>,
        f: F,
    ) -> TypedUnaryHandlerBuilder<AsyncExecutor<F, TypedContext<I>, O>, I, O>
    where
        I: serde::de::DeserializeOwned + Send + Sync + 'static,
        O: serde::Serialize + Send + Sync + 'static,
        F: Fn(TypedContext<I>) -> Fut + Send + Sync + 'static,
        Fut: Future<Output = Result<O>> + Send + 'static,
    {
        typed_unary_async(name, f)
    }
}

// ============================================================================
// Type Definitions
// ============================================================================

/// Unified response type for request-response handlers.
///
/// - `Ok(None)` = ACK (success, no payload)
/// - `Ok(Some(bytes))` = Response (success with payload)
/// - `Err(error)` = NACK/Error
pub type UnifiedResponse = Result<Option<Bytes>>;

/// Dispatch mode for handlers
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DispatchMode {
    /// Execute handler inline on dispatcher task (minimal latency)
    Inline,
    /// Spawn handler on separate task (default, safer)
    Spawn,
}

// ============================================================================
// Context Objects
// ============================================================================

/// Context passed to active message handlers
#[derive(Clone, Dissolve)]
pub struct Context {
    /// Unique identifier for this message (compact, human-readable)
    pub message_id: crate::am::common::MessageId,
    /// The message payload
    pub payload: Bytes,
    /// Optional user headers (for tracing, metadata, etc.)
    pub headers: Option<std::collections::HashMap<String, String>>,
    /// The Nova active message system API
    pub nova: Arc<crate::am::Nova>,
}

/// Context passed to typed handlers (already deserialized input)
#[derive(Clone, Dissolve)]
pub struct TypedContext<I> {
    /// Unique identifier for this message (compact, human-readable)
    pub message_id: crate::am::common::MessageId,
    /// The deserialized input
    pub input: I,
    /// Optional user headers (for tracing, metadata, etc.)
    pub headers: Option<std::collections::HashMap<String, String>>,
    /// The Nova active message system API
    pub nova: Arc<crate::am::Nova>,
}

// ============================================================================
// Core HandlerExecutor Trait (GAT-based, avoids async_trait)
// ============================================================================

/// Core trait for handler execution with GAT to support both sync and async
/// WITHOUT using async_trait (avoids boxing overhead for sync handlers)
pub trait HandlerExecutor<C, T>: Send + Sync {
    /// The future type returned by execute
    /// For sync: Ready<Result<T>>
    /// For async: BoxFuture<'a, Result<T>>
    type Future<'a>: Future<Output = Result<T>> + Send + 'a
    where
        Self: 'a,
        C: 'a,
        T: 'a;

    /// Execute the handler with the given context
    fn execute<'a>(&'a self, ctx: C) -> Self::Future<'a>
    where
        C: 'a;

    /// Whether this handler executes asynchronously (for future optimizations)
    fn is_async(&self) -> bool;
}

// ============================================================================
// Sync Executor Implementation
// ============================================================================

/// Wraps a sync closure: Fn(C) -> Result<T>
pub struct SyncExecutor<F, C, T> {
    f: F,
    _phantom: PhantomData<fn(C) -> T>,
}

impl<F, C, T> SyncExecutor<F, C, T> {
    fn new(f: F) -> Self {
        Self {
            f,
            _phantom: PhantomData,
        }
    }
}

impl<F, C, T> HandlerExecutor<C, T> for SyncExecutor<F, C, T>
where
    F: Fn(C) -> Result<T> + Send + Sync,
    C: Send + 'static,
    T: Send + 'static,
{
    type Future<'a>
        = Ready<Result<T>>
    where
        Self: 'a,
        C: 'a,
        T: 'a;

    fn execute<'a>(&'a self, ctx: C) -> Self::Future<'a>
    where
        C: 'a,
    {
        ready((self.f)(ctx))
    }

    fn is_async(&self) -> bool {
        false
    }
}

// ============================================================================
// Async Executor Implementation
// ============================================================================

/// Wraps an async closure: Fn(C) -> impl Future<Output = Result<T>>
pub struct AsyncExecutor<F, C, T> {
    f: F,
    _phantom: PhantomData<fn(C) -> T>,
}

impl<F, C, T> AsyncExecutor<F, C, T> {
    fn new(f: F) -> Self {
        Self {
            f,
            _phantom: PhantomData,
        }
    }
}

impl<F, Fut, C, T> HandlerExecutor<C, T> for AsyncExecutor<F, C, T>
where
    F: Fn(C) -> Fut + Send + Sync,
    Fut: Future<Output = Result<T>> + Send + 'static,
    C: Send + 'static,
    T: Send + 'static,
{
    type Future<'a>
        = BoxFuture<'a, Result<T>>
    where
        Self: 'a,
        C: 'a,
        T: 'a;

    fn execute<'a>(&'a self, ctx: C) -> Self::Future<'a>
    where
        C: 'a,
    {
        Box::pin((self.f)(ctx))
    }

    fn is_async(&self) -> bool {
        true
    }
}

// ============================================================================
// Adapter: HandlerExecutor -> ActiveMessageHandler (bridges to existing infra)
// ============================================================================

/// Adapter for active message handlers
struct AmExecutorAdapter<E> {
    executor: Arc<E>,
    name: String,
}

impl<E> AmExecutorAdapter<E> {
    fn new(executor: E, name: String) -> Self {
        Self {
            executor: Arc::new(executor),
            name,
        }
    }
}

impl<E> ActiveMessageHandler for AmExecutorAdapter<E>
where
    E: HandlerExecutor<Context, ()> + 'static,
{
    fn handle(&self, ctx: HandlerContext) -> Pin<Box<dyn Future<Output = ()> + Send + 'static>> {
        // Create user-facing simplified context
        let am_ctx = Context {
            message_id: crate::am::common::MessageId::new(ctx.message_id),
            payload: ctx.payload,
            headers: ctx.headers.clone(),
            nova: ctx.system.clone(),
        };

        let executor = self.executor.clone();
        let name = self.name.clone();

        Box::pin(async move {
            if let Err(e) = executor.execute(am_ctx).await {
                error!("AM handler '{}' failed: {}", name, e);
            }
        })
    }

    fn name(&self) -> &str {
        &self.name
    }
}

/// Adapter for unary (request-response) handlers
struct UnaryExecutorAdapter<E> {
    executor: Arc<E>,
    name: String,
}

impl<E> UnaryExecutorAdapter<E> {
    fn new(executor: E, name: String) -> Self {
        Self {
            executor: Arc::new(executor),
            name,
        }
    }
}

impl<E> ActiveMessageHandler for UnaryExecutorAdapter<E>
where
    E: HandlerExecutor<Context, Option<Bytes>> + 'static,
{
    fn handle(&self, ctx: HandlerContext) -> Pin<Box<dyn Future<Output = ()> + Send + 'static>> {
        // Create user-facing simplified context
        let unary_ctx = Context {
            message_id: crate::am::common::MessageId::new(ctx.message_id),
            payload: ctx.payload,
            headers: ctx.headers.clone(),
            nova: ctx.system.clone(),
        };

        let executor = self.executor.clone();
        // Keep internal plumbing for response handling
        let backend = ctx.system.backend().clone();
        let response_id = ctx.message_id;
        let response_type = ctx.response_type;
        let headers = ctx.headers.clone();

        Box::pin(async move {
            let result = executor.execute(unary_ctx).await;

            // Send appropriate response based on response type
            let send_result = match (response_type, result) {
                // AckNack path (am_sync)
                (ResponseType::AckNack, Ok(None)) => send_ack(backend, response_id).await,
                (ResponseType::AckNack, Ok(Some(_))) => {
                    // AckNack handlers shouldn't return payloads, just send ack
                    send_ack(backend, response_id).await
                }
                (ResponseType::AckNack, Err(err)) => {
                    let error_msg = err.to_string();
                    send_nack(backend, response_id, error_msg).await
                }
                // Unary path
                (ResponseType::Unary, Ok(None)) => {
                    send_response_ok(backend, response_id, headers.clone()).await
                }
                (ResponseType::Unary, Ok(Some(bytes))) => {
                    send_response(backend, response_id, headers.clone(), bytes).await
                }
                (ResponseType::Unary, Err(err)) => {
                    let error_msg = err.to_string();
                    send_response_error(backend, response_id, headers.clone(), error_msg).await
                }
                // FireAndForget shouldn't call unary handlers
                (ResponseType::FireAndForget, _) => {
                    error!("FireAndForget message incorrectly routed to unary handler");
                    Ok(())
                }
            };

            if let Err(e) = send_result {
                debug!("Failed to send response: {}", e);
            }
        })
    }

    fn name(&self) -> &str {
        &self.name
    }
}

/// Adapter for typed unary handlers with automatic serialization
struct TypedUnaryExecutorAdapter<E, I, O> {
    executor: Arc<E>,
    name: String,
    _phantom: PhantomData<fn(I) -> O>,
}

impl<E, I, O> TypedUnaryExecutorAdapter<E, I, O> {
    fn new(executor: E, name: String) -> Self {
        Self {
            executor: Arc::new(executor),
            name,
            _phantom: PhantomData,
        }
    }
}

impl<E, I, O> ActiveMessageHandler for TypedUnaryExecutorAdapter<E, I, O>
where
    E: HandlerExecutor<TypedContext<I>, O> + 'static,
    I: serde::de::DeserializeOwned + Send + Sync + 'static,
    O: serde::Serialize + Send + Sync + 'static,
{
    fn handle(&self, ctx: HandlerContext) -> Pin<Box<dyn Future<Output = ()> + Send + 'static>> {
        let payload = ctx.payload;
        let system = ctx.system.clone();
        let msg_id = crate::am::common::MessageId::new(ctx.message_id);
        let headers = ctx.headers.clone();
        // Keep internal plumbing for response handling
        let backend = ctx.system.backend().clone();
        let response_id = ctx.message_id;
        let response_type = ctx.response_type;
        let executor = self.executor.clone();

        Box::pin(async move {
            // Deserialize input
            let input: I = match if payload.is_empty() {
                serde_json::from_slice(b"null")
            } else {
                serde_json::from_slice(&payload)
            } {
                Ok(input) => input,
                Err(e) => {
                    let error_msg = format!("Failed to deserialize input: {}", e);
                    let send_result = match response_type {
                        ResponseType::AckNack => send_nack(backend, response_id, error_msg).await,
                        ResponseType::Unary => {
                            send_response_error(backend, response_id, headers.clone(), error_msg)
                                .await
                        }
                        ResponseType::FireAndForget => Ok(()),
                    };
                    if let Err(send_err) = send_result {
                        debug!("Failed to send deserialization error: {}", send_err);
                    }
                    return;
                }
            };

            // Create user-facing simplified context
            let typed_ctx = TypedContext {
                message_id: msg_id,
                input,
                headers: headers.clone(),
                nova: system,
            };

            // Execute handler
            let result = executor.execute(typed_ctx).await;

            // Send appropriate response based on response type
            let send_result = match (response_type, result) {
                // AckNack path - typed handlers always return data, send ack for success
                (ResponseType::AckNack, Ok(_output)) => send_ack(backend, response_id).await,
                (ResponseType::AckNack, Err(err)) => {
                    let error_msg = err.to_string();
                    send_nack(backend, response_id, error_msg).await
                }
                // Unary path - serialize and send output
                (ResponseType::Unary, Ok(output)) => match serde_json::to_vec(&output) {
                    Ok(response_bytes) => {
                        let bytes = Bytes::from(response_bytes);
                        send_response(backend, response_id, headers.clone(), bytes).await
                    }
                    Err(e) => {
                        let error_msg = format!("Failed to serialize output: {}", e);
                        send_response_error(backend, response_id, headers.clone(), error_msg).await
                    }
                },
                (ResponseType::Unary, Err(err)) => {
                    let error_msg = err.to_string();
                    send_response_error(backend, response_id, headers.clone(), error_msg).await
                }
                // FireAndForget shouldn't call typed unary handlers
                (ResponseType::FireAndForget, _) => {
                    error!("FireAndForget message incorrectly routed to typed unary handler");
                    Ok(())
                }
            };

            if let Err(e) = send_result {
                debug!("Failed to send response: {}", e);
            }
        })
    }

    fn name(&self) -> &str {
        &self.name
    }
}

// ============================================================================
// Helper Functions for Sending Responses
// ============================================================================

// Static error handlers (created once, reused for all messages)
struct AckErrorHandler;
impl dynamo_nova_backend::TransportErrorHandler for AckErrorHandler {
    fn on_error(&self, _header: Bytes, _payload: Bytes, error: String) {
        error!("Failed to send ACK: {}", error);
    }
}

struct NackErrorHandler;
impl dynamo_nova_backend::TransportErrorHandler for NackErrorHandler {
    fn on_error(&self, _header: Bytes, _payload: Bytes, error: String) {
        error!("Failed to send NACK: {}", error);
    }
}

struct ResponseErrorHandler;
impl dynamo_nova_backend::TransportErrorHandler for ResponseErrorHandler {
    fn on_error(&self, _header: Bytes, _payload: Bytes, error: String) {
        error!("Failed to send response: {}", error);
    }
}

// Lazy static error handlers
static ACK_ERROR_HANDLER: std::sync::OnceLock<Arc<dyn dynamo_nova_backend::TransportErrorHandler>> =
    std::sync::OnceLock::new();
static NACK_ERROR_HANDLER: std::sync::OnceLock<
    Arc<dyn dynamo_nova_backend::TransportErrorHandler>,
> = std::sync::OnceLock::new();
static RESPONSE_ERROR_HANDLER: std::sync::OnceLock<
    Arc<dyn dynamo_nova_backend::TransportErrorHandler>,
> = std::sync::OnceLock::new();

#[inline(always)]
fn get_ack_error_handler() -> Arc<dyn dynamo_nova_backend::TransportErrorHandler> {
    ACK_ERROR_HANDLER
        .get_or_init(|| Arc::new(AckErrorHandler))
        .clone()
}

#[inline(always)]
fn get_nack_error_handler() -> Arc<dyn dynamo_nova_backend::TransportErrorHandler> {
    NACK_ERROR_HANDLER
        .get_or_init(|| Arc::new(NackErrorHandler))
        .clone()
}

#[inline(always)]
fn get_response_error_handler() -> Arc<dyn dynamo_nova_backend::TransportErrorHandler> {
    RESPONSE_ERROR_HANDLER
        .get_or_init(|| Arc::new(ResponseErrorHandler))
        .clone()
}

// --- AckNack path (am_sync) - uses Event channel with encode_event_header ---

/// Send an ACK response for am_sync (uses Event channel)
async fn send_ack(backend: Arc<NovaBackend>, response_id: ResponseId) -> Result<()> {
    let header = encode_event_header(EventType::Ack(response_id, Outcome::Ok));

    backend.send_message_to_worker(
        WorkerId::from_u64(response_id.worker_id()),
        header.to_vec(),
        vec![],
        MessageType::Ack,
        get_ack_error_handler(),
    )?;

    Ok(())
}

/// Send a NACK response for am_sync (uses Event channel)
async fn send_nack(
    backend: Arc<NovaBackend>,
    response_id: ResponseId,
    error_message: String,
) -> Result<()> {
    let header = encode_event_header(EventType::Ack(response_id, Outcome::Error));
    let payload = Bytes::from(error_message.into_bytes());

    backend.send_message_to_worker(
        WorkerId::from_u64(response_id.worker_id()),
        header.to_vec(),
        payload.to_vec(),
        MessageType::Ack,
        get_nack_error_handler(),
    )?;

    Ok(())
}

// --- Unary path (unary/typed_unary) - uses Response channel with encode_response_header ---

/// Send an OK response with empty payload for unary (uses Response channel)
async fn send_response_ok(
    backend: Arc<NovaBackend>,
    response_id: ResponseId,
    headers: Option<std::collections::HashMap<String, String>>,
) -> Result<()> {
    let header = encode_response_header(response_id, headers)
        .map_err(|e| anyhow::anyhow!("Failed to encode response header: {}", e))?;

    backend.send_message_to_worker(
        WorkerId::from_u64(response_id.worker_id()),
        header.to_vec(),
        vec![],
        MessageType::Response,
        get_response_error_handler(),
    )?;

    Ok(())
}

/// Send a response with payload for unary (uses Response channel)
async fn send_response(
    backend: Arc<NovaBackend>,
    response_id: ResponseId,
    headers: Option<std::collections::HashMap<String, String>>,
    payload: Bytes,
) -> Result<()> {
    let header = encode_response_header(response_id, headers)
        .map_err(|e| anyhow::anyhow!("Failed to encode response header: {}", e))?;

    backend.send_message_to_worker(
        WorkerId::from_u64(response_id.worker_id()),
        header.to_vec(),
        payload.to_vec(),
        MessageType::Response,
        get_response_error_handler(),
    )?;

    Ok(())
}

/// Send an error response for unary (uses Response channel)
async fn send_response_error(
    backend: Arc<NovaBackend>,
    response_id: ResponseId,
    headers: Option<std::collections::HashMap<String, String>>,
    error_message: String,
) -> Result<()> {
    let header = encode_response_header(response_id, headers)
        .map_err(|e| anyhow::anyhow!("Failed to encode response header: {}", e))?;
    let payload = Bytes::from(error_message.into_bytes());

    backend.send_message_to_worker(
        WorkerId::from_u64(response_id.worker_id()),
        header.to_vec(),
        payload.to_vec(),
        MessageType::Response,
        get_response_error_handler(),
    )?;

    Ok(())
}

// ============================================================================
// Builder Structs
// ============================================================================

/// Builder for active message handlers
pub struct AmHandlerBuilder<E> {
    executor: E,
    name: String,
    dispatch_mode: DispatchMode,
}

impl<E> AmHandlerBuilder<E>
where
    E: HandlerExecutor<Context, ()> + 'static,
{
    fn new(executor: E, name: String) -> Self {
        Self {
            executor,
            name,
            dispatch_mode: DispatchMode::Spawn, // Default
        }
    }

    /// Set dispatch mode to inline (minimal latency)
    pub fn inline(mut self) -> Self {
        self.dispatch_mode = DispatchMode::Inline;
        self
    }

    /// Set dispatch mode to spawn (async execution)
    pub fn spawn(mut self) -> Self {
        self.dispatch_mode = DispatchMode::Spawn;
        self
    }

    /// Build the handler and return a dispatcher
    pub fn build(self) -> NovaHandler {
        let adapter = AmExecutorAdapter::new(self.executor, self.name);

        let dispatcher: Arc<dyn ActiveMessageDispatcher> = match self.dispatch_mode {
            DispatchMode::Inline => Arc::new(InlineDispatcher::new(adapter)),
            DispatchMode::Spawn => {
                let task_tracker = TaskTracker::new();
                Arc::new(SpawnedDispatcher::new(adapter, task_tracker))
            }
        };

        NovaHandler { dispatcher }
    }
}

/// Builder for unary (request-response) handlers
pub struct UnaryHandlerBuilder<E> {
    executor: E,
    name: String,
    dispatch_mode: DispatchMode,
}

impl<E> UnaryHandlerBuilder<E>
where
    E: HandlerExecutor<Context, Option<Bytes>> + 'static,
{
    fn new(executor: E, name: String) -> Self {
        Self {
            executor,
            name,
            dispatch_mode: DispatchMode::Spawn, // Default
        }
    }

    /// Set dispatch mode to inline (minimal latency)
    pub fn inline(mut self) -> Self {
        self.dispatch_mode = DispatchMode::Inline;
        self
    }

    /// Set dispatch mode to spawn (async execution)
    pub fn spawn(mut self) -> Self {
        self.dispatch_mode = DispatchMode::Spawn;
        self
    }

    /// Build the handler and return a dispatcher
    pub fn build(self) -> NovaHandler {
        let adapter = UnaryExecutorAdapter::new(self.executor, self.name);

        let dispatcher: Arc<dyn ActiveMessageDispatcher> = match self.dispatch_mode {
            DispatchMode::Inline => Arc::new(InlineDispatcher::new(adapter)),
            DispatchMode::Spawn => {
                let task_tracker = TaskTracker::new();
                Arc::new(SpawnedDispatcher::new(adapter, task_tracker))
            }
        };

        NovaHandler { dispatcher }
    }
}

/// Builder for typed unary handlers
pub struct TypedUnaryHandlerBuilder<E, I, O> {
    executor: E,
    name: String,
    dispatch_mode: DispatchMode,
    _phantom: PhantomData<fn(I) -> O>,
}

impl<E, I, O> TypedUnaryHandlerBuilder<E, I, O>
where
    E: HandlerExecutor<TypedContext<I>, O> + 'static,
    I: serde::de::DeserializeOwned + Send + Sync + 'static,
    O: serde::Serialize + Send + Sync + 'static,
{
    fn new(executor: E, name: String) -> Self {
        Self {
            executor,
            name,
            dispatch_mode: DispatchMode::Spawn, // Default
            _phantom: PhantomData,
        }
    }

    /// Set dispatch mode to inline (minimal latency)
    pub fn inline(mut self) -> Self {
        self.dispatch_mode = DispatchMode::Inline;
        self
    }

    /// Set dispatch mode to spawn (async execution)
    pub fn spawn(mut self) -> Self {
        self.dispatch_mode = DispatchMode::Spawn;
        self
    }

    /// Build the handler and return a dispatcher
    pub fn build(self) -> NovaHandler {
        let adapter = TypedUnaryExecutorAdapter::new(self.executor, self.name);

        let dispatcher: Arc<dyn ActiveMessageDispatcher> = match self.dispatch_mode {
            DispatchMode::Inline => Arc::new(InlineDispatcher::new(adapter)),
            DispatchMode::Spawn => {
                let task_tracker = TaskTracker::new();
                Arc::new(SpawnedDispatcher::new(adapter, task_tracker))
            }
        };

        NovaHandler { dispatcher }
    }
}

// ============================================================================
// Entry Point Functions
// ============================================================================

/// Create a synchronous active message handler (internal)
fn am_handler<F>(name: impl Into<String>, f: F) -> AmHandlerBuilder<SyncExecutor<F, Context, ()>>
where
    F: Fn(Context) -> Result<()> + Send + Sync + 'static,
{
    let name = name.into();
    let executor = SyncExecutor::new(f);
    AmHandlerBuilder::new(executor, name)
}

/// Create an asynchronous active message handler (internal)
fn am_handler_async<F, Fut>(
    name: impl Into<String>,
    f: F,
) -> AmHandlerBuilder<AsyncExecutor<F, Context, ()>>
where
    F: Fn(Context) -> Fut + Send + Sync + 'static,
    Fut: Future<Output = Result<()>> + Send + 'static,
{
    let name = name.into();
    let executor = AsyncExecutor::new(f);
    AmHandlerBuilder::new(executor, name)
}

/// Create a synchronous unary (request-response) handler (internal)
fn unary_handler<F>(
    name: impl Into<String>,
    f: F,
) -> UnaryHandlerBuilder<SyncExecutor<F, Context, Option<Bytes>>>
where
    F: Fn(Context) -> UnifiedResponse + Send + Sync + 'static,
{
    let name = name.into();
    let executor = SyncExecutor::new(f);
    UnaryHandlerBuilder::new(executor, name)
}

/// Create an asynchronous unary (request-response) handler (internal)
fn unary_handler_async<F, Fut>(
    name: impl Into<String>,
    f: F,
) -> UnaryHandlerBuilder<AsyncExecutor<F, Context, Option<Bytes>>>
where
    F: Fn(Context) -> Fut + Send + Sync + 'static,
    Fut: Future<Output = UnifiedResponse> + Send + 'static,
{
    let name = name.into();
    let executor = AsyncExecutor::new(f);
    UnaryHandlerBuilder::new(executor, name)
}

/// Create a synchronous typed unary handler with automatic serialization (internal)
fn typed_unary<I, O, F>(
    name: impl Into<String>,
    f: F,
) -> TypedUnaryHandlerBuilder<SyncExecutor<F, TypedContext<I>, O>, I, O>
where
    I: serde::de::DeserializeOwned + Send + Sync + 'static,
    O: serde::Serialize + Send + Sync + 'static,
    F: Fn(TypedContext<I>) -> Result<O> + Send + Sync + 'static,
{
    let name = name.into();
    let executor = SyncExecutor::new(f);
    TypedUnaryHandlerBuilder::new(executor, name)
}

/// Create an asynchronous typed unary handler with automatic serialization (internal)
fn typed_unary_async<I, O, F, Fut>(
    name: impl Into<String>,
    f: F,
) -> TypedUnaryHandlerBuilder<AsyncExecutor<F, TypedContext<I>, O>, I, O>
where
    I: serde::de::DeserializeOwned + Send + Sync + 'static,
    O: serde::Serialize + Send + Sync + 'static,
    F: Fn(TypedContext<I>) -> Fut + Send + Sync + 'static,
    Fut: Future<Output = Result<O>> + Send + 'static,
{
    let name = name.into();
    let executor = AsyncExecutor::new(f);
    TypedUnaryHandlerBuilder::new(executor, name)
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use serde::{Deserialize, Serialize};

    // ============================================================================
    // Test Data Structures
    // ============================================================================

    #[derive(Serialize, Deserialize, Debug, Clone)]
    struct CalcRequest {
        a: f64,
        b: f64,
        operation: String,
    }

    #[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
    struct CalcResponse {
        result: f64,
    }

    #[derive(Serialize, Deserialize, Debug, Clone)]
    struct PingRequest {
        message: String,
    }

    #[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
    struct PingResponse {
        echo: String,
    }

    // ============================================================================
    // Test Builder API
    // ============================================================================

    #[test]
    fn test_am_handler_builder() {
        // Sync AM handler with default spawn mode
        let handler = am_handler("test_am", |_ctx| Ok(())).build();
        assert_eq!(handler.name(), "test_am");

        // Sync AM handler with explicit inline mode
        let handler = am_handler("test_am_inline", |_ctx| Ok(())).inline().build();
        assert_eq!(handler.name(), "test_am_inline");

        // Sync AM handler with explicit spawn mode
        let handler = am_handler("test_am_spawn", |_ctx| Ok(())).spawn().build();
        assert_eq!(handler.name(), "test_am_spawn");
    }

    #[test]
    fn test_am_handler_async_builder() {
        // Async AM handler with default spawn mode
        let handler = am_handler_async("test_am_async", |_ctx| async move { Ok(()) }).build();
        assert_eq!(handler.name(), "test_am_async");

        // Async AM handler with inline mode
        let handler = am_handler_async("test_am_async_inline", |_ctx| async move { Ok(()) })
            .inline()
            .build();
        assert_eq!(handler.name(), "test_am_async_inline");
    }

    #[test]
    fn test_unary_handler_builder() {
        // Sync unary handler
        let handler = unary_handler("test_unary", |_ctx| Ok(None)).build();
        assert_eq!(handler.name(), "test_unary");

        // With inline mode
        let handler = unary_handler("test_unary_inline", |_ctx| Ok(None))
            .inline()
            .build();
        assert_eq!(handler.name(), "test_unary_inline");
    }

    #[test]
    fn test_unary_handler_async_builder() {
        // Async unary handler
        let handler =
            unary_handler_async("test_unary_async", |_ctx| async move { Ok(None) }).build();
        assert_eq!(handler.name(), "test_unary_async");
    }

    #[test]
    fn test_typed_unary_builder() {
        // Sync typed handler
        let handler = typed_unary("test_typed", |ctx: TypedContext<PingRequest>| {
            Ok(PingResponse {
                echo: ctx.input.message,
            })
        })
        .build();
        assert_eq!(handler.name(), "test_typed");

        // With inline mode
        let handler = typed_unary("test_typed_inline", |ctx: TypedContext<PingRequest>| {
            Ok(PingResponse {
                echo: ctx.input.message,
            })
        })
        .inline()
        .build();
        assert_eq!(handler.name(), "test_typed_inline");
    }

    #[test]
    fn test_typed_unary_async_builder() {
        // Async typed handler
        let handler = typed_unary_async(
            "test_typed_async",
            |ctx: TypedContext<PingRequest>| async move {
                Ok(PingResponse {
                    echo: ctx.input.message,
                })
            },
        )
        .build();
        assert_eq!(handler.name(), "test_typed_async");
    }

    #[test]
    fn test_typed_unary_calculator() {
        // Calculator handler
        let handler = typed_unary("calculator", |ctx: TypedContext<CalcRequest>| {
            let req = ctx.input;
            let result = match req.operation.as_str() {
                "add" => req.a + req.b,
                "subtract" => req.a - req.b,
                "multiply" => req.a * req.b,
                "divide" => {
                    if req.b == 0.0 {
                        return Err(anyhow::anyhow!("Division by zero"));
                    }
                    req.a / req.b
                }
                _ => return Err(anyhow::anyhow!("Unknown operation: {}", req.operation)),
            };
            Ok(CalcResponse { result })
        })
        .build();

        assert_eq!(handler.name(), "calculator");
    }

    #[test]
    fn test_dispatch_modes() {
        // Default is spawn
        let handler = am_handler("default", |_ctx| Ok(())).build();
        assert_eq!(handler.name(), "default");

        // Explicit inline
        let handler = am_handler("inline", |_ctx| Ok(())).inline().build();
        assert_eq!(handler.name(), "inline");

        // Explicit spawn
        let handler = am_handler("spawn", |_ctx| Ok(())).spawn().build();
        assert_eq!(handler.name(), "spawn");
    }
}
