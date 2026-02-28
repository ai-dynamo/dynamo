// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Clean, builder-based handler API for active message patterns.
//!
//! ## Handler Types
//!
//! ### Active Message Handlers
//! - **`am_handler()`** - Sync AM handler: `Fn(Context) -> anyhow::Result<()>`
//! - **`am_handler_async()`** - Async AM handler: `Fn(Context) -> Future<anyhow::Result<()>>`
//!
//! ### Request-Response Handlers
//! - **`unary_handler()`** - Sync unary: `Fn(Context) -> UnifiedResponse`
//! - **`unary_handler_async()`** - Async unary: `Fn(Context) -> Future<UnifiedResponse>`
//!
//! ### Typed Request-Response Handlers
//! - **`typed_unary()`** - Sync typed: `Fn(TypedContext<I>) -> anyhow::Result<O>`
//! - **`typed_unary_async()`** - Async typed: `Fn(TypedContext<I>) -> Future<anyhow::Result<O>>`
//!
//! ## Context Objects
//!
//! All context objects include:
//! - **`message_id: MessageId`** - Unique, compact identifier for this message
//! - **`msg: Arc<Messenger>`** - The messenger API for sending messages, querying handlers, etc.

mod manager;
pub(crate) use manager::HandlerManager;

use anyhow::Result;
use bytes::Bytes;
use futures::future::{BoxFuture, Ready, ready};
use std::future::Future;
use std::marker::PhantomData;
use std::pin::Pin;
use std::sync::Arc;
use tracing::{debug, error};
use velo_common::WorkerId;

use crate::common::events::{EventType, Outcome, encode_event_header};
use crate::common::messages::ResponseType;
use crate::common::responses::{ResponseId, encode_response_header};
use crate::server::dispatcher::{
    ActiveMessageDispatcher, ActiveMessageHandler, ControlMessage, HandlerContext,
    InlineDispatcher, SpawnedDispatcher,
};
use derive_getters::Dissolve;
use tokio_util::task::TaskTracker;
use velo_backend::{MessageType, VeloBackend};

// ============================================================================
// Opaque Handles
// ============================================================================

pub struct Handler {
    pub(crate) dispatcher: Arc<dyn ActiveMessageDispatcher>,
}

impl Handler {
    pub fn name(&self) -> &str {
        self.dispatcher.as_ref().name()
    }

    /// Create a synchronous active message handler
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
    pub message_id: crate::common::MessageId,
    /// The message payload
    pub payload: Bytes,
    /// Optional user headers (for tracing, metadata, etc.)
    pub headers: Option<std::collections::HashMap<String, String>>,
    /// The messenger API
    pub msg: Arc<crate::Messenger>,
}

/// Context passed to typed handlers (already deserialized input)
#[derive(Clone, Dissolve)]
pub struct TypedContext<I> {
    /// Unique identifier for this message (compact, human-readable)
    pub message_id: crate::common::MessageId,
    /// The deserialized input
    pub input: I,
    /// Optional user headers (for tracing, metadata, etc.)
    pub headers: Option<std::collections::HashMap<String, String>>,
    /// The messenger API
    pub msg: Arc<crate::Messenger>,
}

// ============================================================================
// Core HandlerExecutor Trait (GAT-based, avoids async_trait)
// ============================================================================

/// Core trait for handler execution with GAT to support both sync and async
pub trait HandlerExecutor<C, T>: Send + Sync {
    type Future<'a>: Future<Output = Result<T>> + Send + 'a
    where
        Self: 'a,
        C: 'a,
        T: 'a;

    fn execute<'a>(&'a self, ctx: C) -> Self::Future<'a>
    where
        C: 'a;

    fn is_async(&self) -> bool;
}

// ============================================================================
// Sync Executor Implementation
// ============================================================================

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
// Adapter: HandlerExecutor -> ActiveMessageHandler
// ============================================================================

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
        let am_ctx = Context {
            message_id: crate::common::MessageId::new(ctx.message_id),
            payload: ctx.payload,
            headers: ctx.headers.clone(),
            msg: ctx.system.clone(),
        };

        let executor = self.executor.clone();
        let name = self.name.clone();

        let backend = ctx.system.backend().clone();
        let response_id = ctx.message_id;
        let response_type = ctx.response_type;

        Box::pin(async move {
            let result = executor.execute(am_ctx).await;

            match response_type {
                ResponseType::FireAndForget => {
                    if let Err(e) = result {
                        error!("AM handler '{}' failed: {}", name, e);
                    }
                }
                ResponseType::AckNack => {
                    let send_result = match result {
                        Ok(()) => send_ack(backend, response_id).await,
                        Err(err) => {
                            error!("AM handler '{}' failed: {}", name, err);
                            send_nack(backend, response_id, err.to_string()).await
                        }
                    };
                    if let Err(e) = send_result {
                        debug!("Failed to send ACK/NACK response: {}", e);
                    }
                }
                ResponseType::Unary => {
                    error!("Unary message incorrectly routed to AM handler '{}'", name);
                    if let Err(e) = result {
                        error!("AM handler '{}' failed: {}", name, e);
                    }
                }
            }
        })
    }

    fn name(&self) -> &str {
        &self.name
    }
}

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
        let unary_ctx = Context {
            message_id: crate::common::MessageId::new(ctx.message_id),
            payload: ctx.payload,
            headers: ctx.headers.clone(),
            msg: ctx.system.clone(),
        };

        let executor = self.executor.clone();
        let backend = ctx.system.backend().clone();
        let response_id = ctx.message_id;
        let response_type = ctx.response_type;
        let headers = ctx.headers.clone();

        Box::pin(async move {
            let result = executor.execute(unary_ctx).await;

            let send_result = match (response_type, result) {
                (ResponseType::AckNack, Ok(None)) => send_ack(backend, response_id).await,
                (ResponseType::AckNack, Ok(Some(_))) => send_ack(backend, response_id).await,
                (ResponseType::AckNack, Err(err)) => {
                    let error_msg = err.to_string();
                    send_nack(backend, response_id, error_msg).await
                }
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
        let msg_id = crate::common::MessageId::new(ctx.message_id);
        let headers = ctx.headers.clone();
        let backend = ctx.system.backend().clone();
        let response_id = ctx.message_id;
        let response_type = ctx.response_type;
        let executor = self.executor.clone();

        Box::pin(async move {
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

            let typed_ctx = TypedContext {
                message_id: msg_id,
                input,
                headers: headers.clone(),
                msg: system,
            };

            let result = executor.execute(typed_ctx).await;

            let send_result = match (response_type, result) {
                (ResponseType::AckNack, Ok(_output)) => send_ack(backend, response_id).await,
                (ResponseType::AckNack, Err(err)) => {
                    let error_msg = err.to_string();
                    send_nack(backend, response_id, error_msg).await
                }
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

struct AckErrorHandler;
impl velo_backend::TransportErrorHandler for AckErrorHandler {
    fn on_error(&self, _header: Bytes, _payload: Bytes, error: String) {
        error!("Failed to send ACK: {}", error);
    }
}

struct NackErrorHandler;
impl velo_backend::TransportErrorHandler for NackErrorHandler {
    fn on_error(&self, _header: Bytes, _payload: Bytes, error: String) {
        error!("Failed to send NACK: {}", error);
    }
}

struct ResponseErrorHandler;
impl velo_backend::TransportErrorHandler for ResponseErrorHandler {
    fn on_error(&self, _header: Bytes, _payload: Bytes, error: String) {
        error!("Failed to send response: {}", error);
    }
}

static ACK_ERROR_HANDLER: std::sync::OnceLock<Arc<dyn velo_backend::TransportErrorHandler>> =
    std::sync::OnceLock::new();
static NACK_ERROR_HANDLER: std::sync::OnceLock<Arc<dyn velo_backend::TransportErrorHandler>> =
    std::sync::OnceLock::new();
static RESPONSE_ERROR_HANDLER: std::sync::OnceLock<Arc<dyn velo_backend::TransportErrorHandler>> =
    std::sync::OnceLock::new();

#[inline(always)]
fn get_ack_error_handler() -> Arc<dyn velo_backend::TransportErrorHandler> {
    ACK_ERROR_HANDLER
        .get_or_init(|| Arc::new(AckErrorHandler))
        .clone()
}

#[inline(always)]
fn get_nack_error_handler() -> Arc<dyn velo_backend::TransportErrorHandler> {
    NACK_ERROR_HANDLER
        .get_or_init(|| Arc::new(NackErrorHandler))
        .clone()
}

#[inline(always)]
fn get_response_error_handler() -> Arc<dyn velo_backend::TransportErrorHandler> {
    RESPONSE_ERROR_HANDLER
        .get_or_init(|| Arc::new(ResponseErrorHandler))
        .clone()
}

async fn send_ack(backend: Arc<VeloBackend>, response_id: ResponseId) -> Result<()> {
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

async fn send_nack(
    backend: Arc<VeloBackend>,
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

async fn send_response_ok(
    backend: Arc<VeloBackend>,
    response_id: ResponseId,
    headers: Option<std::collections::HashMap<String, String>>,
) -> Result<()> {
    let header = encode_response_header(response_id, Outcome::Ok, headers)
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

async fn send_response(
    backend: Arc<VeloBackend>,
    response_id: ResponseId,
    headers: Option<std::collections::HashMap<String, String>>,
    payload: Bytes,
) -> Result<()> {
    let header = encode_response_header(response_id, Outcome::Ok, headers)
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

async fn send_response_error(
    backend: Arc<VeloBackend>,
    response_id: ResponseId,
    headers: Option<std::collections::HashMap<String, String>>,
    error_message: String,
) -> Result<()> {
    let header = encode_response_header(response_id, Outcome::Error, headers)
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
            dispatch_mode: DispatchMode::Spawn,
        }
    }

    pub fn inline(mut self) -> Self {
        self.dispatch_mode = DispatchMode::Inline;
        self
    }

    pub fn spawn(mut self) -> Self {
        self.dispatch_mode = DispatchMode::Spawn;
        self
    }

    pub fn build(self) -> Handler {
        let adapter = AmExecutorAdapter::new(self.executor, self.name);

        let dispatcher: Arc<dyn ActiveMessageDispatcher> = match self.dispatch_mode {
            DispatchMode::Inline => Arc::new(InlineDispatcher::new(adapter)),
            DispatchMode::Spawn => {
                let task_tracker = TaskTracker::new();
                Arc::new(SpawnedDispatcher::new(adapter, task_tracker))
            }
        };

        Handler { dispatcher }
    }
}

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
            dispatch_mode: DispatchMode::Spawn,
        }
    }

    pub fn inline(mut self) -> Self {
        self.dispatch_mode = DispatchMode::Inline;
        self
    }

    pub fn spawn(mut self) -> Self {
        self.dispatch_mode = DispatchMode::Spawn;
        self
    }

    pub fn build(self) -> Handler {
        let adapter = UnaryExecutorAdapter::new(self.executor, self.name);

        let dispatcher: Arc<dyn ActiveMessageDispatcher> = match self.dispatch_mode {
            DispatchMode::Inline => Arc::new(InlineDispatcher::new(adapter)),
            DispatchMode::Spawn => {
                let task_tracker = TaskTracker::new();
                Arc::new(SpawnedDispatcher::new(adapter, task_tracker))
            }
        };

        Handler { dispatcher }
    }
}

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
            dispatch_mode: DispatchMode::Spawn,
            _phantom: PhantomData,
        }
    }

    pub fn inline(mut self) -> Self {
        self.dispatch_mode = DispatchMode::Inline;
        self
    }

    pub fn spawn(mut self) -> Self {
        self.dispatch_mode = DispatchMode::Spawn;
        self
    }

    pub fn build(self) -> Handler {
        let adapter = TypedUnaryExecutorAdapter::new(self.executor, self.name);

        let dispatcher: Arc<dyn ActiveMessageDispatcher> = match self.dispatch_mode {
            DispatchMode::Inline => Arc::new(InlineDispatcher::new(adapter)),
            DispatchMode::Spawn => {
                let task_tracker = TaskTracker::new();
                Arc::new(SpawnedDispatcher::new(adapter, task_tracker))
            }
        };

        Handler { dispatcher }
    }
}

// ============================================================================
// Entry Point Functions
// ============================================================================

fn am_handler<F>(name: impl Into<String>, f: F) -> AmHandlerBuilder<SyncExecutor<F, Context, ()>>
where
    F: Fn(Context) -> Result<()> + Send + Sync + 'static,
{
    let name = name.into();
    let executor = SyncExecutor::new(f);
    AmHandlerBuilder::new(executor, name)
}

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

    #[test]
    fn test_am_handler_builder() {
        let handler = am_handler("test_am", |_ctx| Ok(())).build();
        assert_eq!(handler.name(), "test_am");

        let handler = am_handler("test_am_inline", |_ctx| Ok(())).inline().build();
        assert_eq!(handler.name(), "test_am_inline");

        let handler = am_handler("test_am_spawn", |_ctx| Ok(())).spawn().build();
        assert_eq!(handler.name(), "test_am_spawn");
    }

    #[test]
    fn test_am_handler_async_builder() {
        let handler = am_handler_async("test_am_async", |_ctx| async move { Ok(()) }).build();
        assert_eq!(handler.name(), "test_am_async");

        let handler = am_handler_async("test_am_async_inline", |_ctx| async move { Ok(()) })
            .inline()
            .build();
        assert_eq!(handler.name(), "test_am_async_inline");
    }

    #[test]
    fn test_unary_handler_builder() {
        let handler = unary_handler("test_unary", |_ctx| Ok(None)).build();
        assert_eq!(handler.name(), "test_unary");

        let handler = unary_handler("test_unary_inline", |_ctx| Ok(None))
            .inline()
            .build();
        assert_eq!(handler.name(), "test_unary_inline");
    }

    #[test]
    fn test_unary_handler_async_builder() {
        let handler =
            unary_handler_async("test_unary_async", |_ctx| async move { Ok(None) }).build();
        assert_eq!(handler.name(), "test_unary_async");
    }

    #[test]
    fn test_typed_unary_builder() {
        let handler = typed_unary("test_typed", |ctx: TypedContext<PingRequest>| {
            Ok(PingResponse {
                echo: ctx.input.message,
            })
        })
        .build();
        assert_eq!(handler.name(), "test_typed");

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
        let handler = am_handler("default", |_ctx| Ok(())).build();
        assert_eq!(handler.name(), "default");

        let handler = am_handler("inline", |_ctx| Ok(())).inline().build();
        assert_eq!(handler.name(), "inline");

        let handler = am_handler("spawn", |_ctx| Ok(())).spawn().build();
        assert_eq!(handler.name(), "spawn");
    }
}
