// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Convenience builders for trait-object clients (`&dyn ActiveMessageClient`).
//!
//! These wrappers delegate to [`MessageBuilder`] while keeping the ergonomic
//! `client.am_*()` APIs.

use std::collections::HashMap;
use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;
use std::task::{Context, Poll};

use anyhow::{Result, anyhow};
use bytes::Bytes;
use serde::Serialize;
use serde::de::DeserializeOwned;

use super::ActiveMessageClient;
use crate::am::{
    InstanceId,
    common::{ActiveMessage, MessageMetadata},
};
use dynamo_identity::WorkerId;

/// Fire-and-forget builder.
pub struct AmSendBuilder {
    inner: MessageBuilder,
}

impl AmSendBuilder {
    pub(crate) fn new(client: Arc<ActiveMessageClient>, handler: &str) -> Result<Self> {
        Ok(Self {
            inner: MessageBuilder::new(client, handler)?,
        })
    }

    pub fn payload<T: Serialize>(mut self, data: T) -> Result<Self> {
        self.inner = self.inner.payload(data)?;
        Ok(self)
    }

    pub fn raw_payload(mut self, data: Bytes) -> Self {
        self.inner = self.inner.raw_payload(data);
        self
    }

    pub fn instance(mut self, instance_id: InstanceId) -> Self {
        self.inner = self.inner.instance(instance_id);
        self
    }

    pub fn worker(mut self, worker_id: WorkerId) -> Self {
        self.inner = self.inner.worker(worker_id);
        self
    }

    pub fn headers(mut self, headers: HashMap<String, String>) -> Self {
        self.inner = self.inner.headers(headers);
        self
    }

    pub fn send(self) -> impl Future<Output = Result<()>> {
        self.inner.fire()
    }

    pub fn send_to(self, target: InstanceId) -> impl Future<Output = Result<()>> {
        self.inner.instance(target).fire()
    }
}

/// Builder for request/response flows that expect an acknowledgement only.
pub struct AmSyncBuilder {
    inner: MessageBuilder,
}

impl AmSyncBuilder {
    pub(crate) fn new(client: Arc<ActiveMessageClient>, handler: &str) -> Result<Self> {
        Ok(Self {
            inner: MessageBuilder::new(client, handler)?,
        })
    }

    pub fn payload<T: Serialize>(mut self, data: T) -> Result<Self> {
        self.inner = self.inner.payload(data)?;
        Ok(self)
    }

    pub fn raw_payload(mut self, data: Bytes) -> Self {
        self.inner = self.inner.raw_payload(data);
        self
    }

    pub fn instance(mut self, instance_id: InstanceId) -> Self {
        self.inner = self.inner.instance(instance_id);
        self
    }

    pub fn worker(mut self, worker_id: WorkerId) -> Self {
        self.inner = self.inner.worker(worker_id);
        self
    }

    pub fn headers(mut self, headers: HashMap<String, String>) -> Self {
        self.inner = self.inner.headers(headers);
        self
    }

    pub fn send(self) -> SyncResult {
        self.inner.sync()
    }

    pub fn send_to(self, target: InstanceId) -> SyncResult {
        self.inner.instance(target).sync()
    }
}

/// Builder for unary handlers returning raw bytes.
pub struct UnaryBuilder {
    inner: MessageBuilder,
}

impl UnaryBuilder {
    pub(crate) fn new(client: Arc<ActiveMessageClient>, handler: &str) -> Result<Self> {
        Ok(Self {
            inner: MessageBuilder::new(client, handler)?,
        })
    }

    pub fn payload<T: Serialize>(mut self, data: T) -> Result<Self> {
        self.inner = self.inner.payload(data)?;
        Ok(self)
    }

    pub fn raw_payload(mut self, data: Bytes) -> Self {
        self.inner = self.inner.raw_payload(data);
        self
    }

    pub fn instance(mut self, instance_id: InstanceId) -> Self {
        self.inner = self.inner.instance(instance_id);
        self
    }

    pub fn worker(mut self, worker_id: WorkerId) -> Self {
        self.inner = self.inner.worker(worker_id);
        self
    }

    pub fn headers(mut self, headers: HashMap<String, String>) -> Self {
        self.inner = self.inner.headers(headers);
        self
    }

    pub fn send(self) -> UnaryResult {
        self.inner.unary()
    }

    pub fn send_to(self, target: InstanceId) -> UnaryResult {
        self.inner.instance(target).unary()
    }
}

/// Builder for typed unary handlers.
pub struct TypedUnaryBuilder<R> {
    inner: MessageBuilder,
    _marker: std::marker::PhantomData<R>,
}

impl<R> TypedUnaryBuilder<R>
where
    R: DeserializeOwned + Send + 'static,
{
    pub(crate) fn new(client: Arc<ActiveMessageClient>, handler: &str) -> Result<Self> {
        Ok(Self {
            inner: MessageBuilder::new(client, handler)?,
            _marker: std::marker::PhantomData,
        })
    }

    pub fn payload<T: Serialize>(mut self, data: T) -> Result<Self> {
        self.inner = self.inner.payload(data)?;
        Ok(self)
    }

    pub fn raw_payload(mut self, data: Bytes) -> Self {
        self.inner = self.inner.raw_payload(data);
        self
    }

    pub fn instance(mut self, instance_id: InstanceId) -> Self {
        self.inner = self.inner.instance(instance_id);
        self
    }

    pub fn worker(mut self, worker_id: WorkerId) -> Self {
        self.inner = self.inner.worker(worker_id);
        self
    }

    pub fn headers(mut self, headers: HashMap<String, String>) -> Self {
        self.inner = self.inner.headers(headers);
        self
    }

    pub fn send(self) -> TypedUnaryResult<R> {
        self.inner.typed()
    }

    pub fn send_to(self, target: InstanceId) -> TypedUnaryResult<R> {
        self.inner.instance(target).typed()
    }
}

/// Error type for target resolution in message builders.
#[derive(Debug)]
enum ResolveError {
    /// Peer not found in cache - discovery needed
    UnresolvedPeer,
    /// Other validation or configuration errors
    Other(anyhow::Error),
}

/// Message type for metadata creation
#[derive(Debug, Clone, Copy)]
enum MessageType {
    Sync,
    Unary,
}

impl From<ResolveError> for anyhow::Error {
    fn from(err: ResolveError) -> Self {
        match err {
            ResolveError::UnresolvedPeer => anyhow!("Peer not found"),
            ResolveError::Other(e) => e,
        }
    }
}

/// Result wrapper for sync operations (acknowledgment only)
pub struct SyncResult {
    awaiter: Option<crate::am::common::responses::ResponseAwaiter>,
}

impl SyncResult {
    fn new(awaiter: crate::am::common::responses::ResponseAwaiter) -> Self {
        Self {
            awaiter: Some(awaiter),
        }
    }
}

impl Future for SyncResult {
    type Output = Result<()>;

    fn poll(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Self::Output> {
        let awaiter = self
            .awaiter
            .as_mut()
            .expect("SyncResult polled after completion");

        match awaiter.poll_recv(cx) {
            Poll::Ready(result) => {
                self.awaiter = None;
                Poll::Ready(match result {
                    Ok(None) | Ok(Some(_)) => Ok(()),
                    Err(err) => Err(anyhow!(err)),
                })
            }
            Poll::Pending => Poll::Pending,
        }
    }
}

/// Result wrapper for unary operations returning raw bytes
pub struct UnaryResult {
    awaiter: Option<crate::am::common::responses::ResponseAwaiter>,
}

impl UnaryResult {
    fn new(awaiter: crate::am::common::responses::ResponseAwaiter) -> Self {
        Self {
            awaiter: Some(awaiter),
        }
    }
}

impl Future for UnaryResult {
    type Output = Result<Bytes>;

    fn poll(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Self::Output> {
        let awaiter = self
            .awaiter
            .as_mut()
            .expect("UnaryResult polled after completion");

        match awaiter.poll_recv(cx) {
            Poll::Ready(result) => {
                self.awaiter = None;
                Poll::Ready(match result {
                    Ok(Some(bytes)) => Ok(bytes),
                    Ok(None) => Ok(Bytes::new()),
                    Err(err) => Err(anyhow!(err)),
                })
            }
            Poll::Pending => Poll::Pending,
        }
    }
}

/// Result wrapper for typed unary operations with deserialization
pub struct TypedUnaryResult<R> {
    awaiter: Option<crate::am::common::responses::ResponseAwaiter>,
    _marker: std::marker::PhantomData<R>,
}

impl<R> TypedUnaryResult<R> {
    fn new(awaiter: crate::am::common::responses::ResponseAwaiter) -> Self {
        Self {
            awaiter: Some(awaiter),
            _marker: std::marker::PhantomData,
        }
    }
}

impl<R: DeserializeOwned> Future for TypedUnaryResult<R> {
    type Output = Result<R>;

    fn poll(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Self::Output> {
        // SAFETY: TypedUnaryResult is Unpin because ResponseAwaiter and PhantomData are both Unpin
        let this = unsafe { self.get_unchecked_mut() };
        let awaiter = this
            .awaiter
            .as_mut()
            .expect("TypedUnaryResult polled after completion");

        match awaiter.poll_recv(cx) {
            Poll::Ready(result) => {
                this.awaiter = None;
                Poll::Ready(match result {
                    Ok(Some(bytes)) => match serde_json::from_slice(&bytes) {
                        Ok(value) => Ok(value),
                        Err(e) => Err(anyhow!("Failed to deserialize response: {}", e)),
                    },
                    Ok(None) => Err(anyhow!("Expected response data, got empty")),
                    Err(err) => Err(anyhow!(err)),
                })
            }
            Poll::Pending => Poll::Pending,
        }
    }
}

/// Minimal message builder supporting fire-and-forget and unary-style sends.
pub struct MessageBuilder {
    client: Arc<ActiveMessageClient>,
    handler: String,
    payload: Option<Bytes>,
    target_instance: Option<InstanceId>,
    target_worker: Option<WorkerId>,
    headers: Option<HashMap<String, String>>,
}

impl MessageBuilder {
    pub fn new(client: Arc<ActiveMessageClient>, handler: &str) -> Result<Self> {
        validate_handler_name(handler)?;
        Ok(Self::new_unchecked(client, handler))
    }

    pub fn new_unchecked(client: Arc<ActiveMessageClient>, handler: &str) -> Self {
        Self {
            client,
            handler: handler.to_string(),
            payload: None,
            target_instance: None,
            target_worker: None,
            headers: None,
        }
    }

    pub fn payload<T: Serialize>(mut self, data: T) -> Result<Self> {
        let bytes =
            serde_json::to_vec(&data).map_err(|e| anyhow!("failed to serialize payload: {}", e))?;
        self.payload = Some(Bytes::from(bytes));
        Ok(self)
    }

    pub fn raw_payload(mut self, data: Bytes) -> Self {
        self.payload = Some(data);
        self
    }

    pub fn instance(mut self, instance_id: InstanceId) -> Self {
        self.target_instance = Some(instance_id);
        self
    }

    pub fn worker(mut self, worker_id: WorkerId) -> Self {
        self.target_worker = Some(worker_id);
        self
    }

    pub fn headers(mut self, headers: HashMap<String, String>) -> Self {
        self.headers = Some(headers);
        self
    }

    fn resolve_target(&self) -> Result<InstanceId, ResolveError> {
        match (self.target_instance, self.target_worker) {
            (Some(instance), None) => Ok(instance),
            (None, Some(worker)) => self
                .client
                .backend
                .try_translate_worker_id(worker)
                .map_err(|_| ResolveError::UnresolvedPeer),
            (Some(_), Some(_)) => Err(ResolveError::Other(anyhow!(
                "Cannot set both .instance() and .worker() - they are mutually exclusive"
            ))),
            (None, None) => Err(ResolveError::Other(anyhow!(
                "Target not set. Call .instance() or .worker() before sending"
            ))),
        }
    }

    fn create_metadata(
        &self,
        response_id: crate::am::common::responses::ResponseId,
        message_type: MessageType,
    ) -> MessageMetadata {
        match message_type {
            MessageType::Sync => {
                MessageMetadata::new_sync(response_id, self.handler.clone(), self.headers.clone())
            }
            MessageType::Unary => {
                MessageMetadata::new_unary(response_id, self.handler.clone(), self.headers.clone())
            }
        }
    }

    fn spawn_handshake_task(
        &self,
        target: InstanceId,
        response_id: crate::am::common::responses::ResponseId,
        message_type: MessageType,
    ) {
        let client = self.client.clone();
        let handler = self.handler.clone();
        let payload = self.payload.clone();
        let headers = self.headers.clone();
        let response_manager = client.response_manager.clone();

        tokio::spawn(async move {
            // Ensure peer ready (handshake)
            if let Err(e) = client.ensure_peer_ready(target, &handler).await {
                tracing::error!(
                    target: "dynamo_nova::client",
                    error = %e,
                    "Failed to prepare peer in slow path"
                );
                let _ = response_manager
                    .complete_outcome(response_id, Err(format!("Handshake failed: {}", e)));
                return;
            }

            // Send message
            let metadata = match message_type {
                MessageType::Sync => MessageMetadata::new_sync(response_id, handler, headers),
                MessageType::Unary => MessageMetadata::new_unary(response_id, handler, headers),
            };

            let message = ActiveMessage {
                metadata,
                payload: payload.unwrap_or_default(),
            };

            if let Err(e) = client.send_message(target, message) {
                tracing::error!(
                    target: "dynamo_nova::client",
                    error = %e,
                    "Failed to send message in slow path"
                );
                let _ = response_manager
                    .complete_outcome(response_id, Err(format!("Send failed: {}", e)));
            }
        });
    }

    fn spawn_discovery_task(
        &self,
        worker_id: dynamo_identity::WorkerId,
        response_id: crate::am::common::responses::ResponseId,
        message_type: MessageType,
    ) {
        let client = self.client.clone();
        let handler = self.handler.clone();
        let payload = self.payload.clone();
        let headers = self.headers.clone();
        let response_manager = client.response_manager.clone();

        tokio::spawn(async move {
            // Resolve via discovery
            let target = match client.resolve_peer_via_discovery(worker_id).await {
                Ok(instance_id) => instance_id,
                Err(e) => {
                    tracing::error!(
                        target: "dynamo_nova::client",
                        error = %e,
                        worker_id = %worker_id,
                        "Discovery failed"
                    );
                    let _ = response_manager
                        .complete_outcome(response_id, Err(format!("Discovery failed: {}", e)));
                    return;
                }
            };

            // Ensure peer ready (handshake)
            if let Err(e) = client.ensure_peer_ready(target, &handler).await {
                tracing::error!(
                    target: "dynamo_nova::client",
                    error = %e,
                    "Failed to prepare peer after discovery"
                );
                let _ = response_manager
                    .complete_outcome(response_id, Err(format!("Handshake failed: {}", e)));
                return;
            }

            // Send message
            let metadata = match message_type {
                MessageType::Sync => MessageMetadata::new_sync(response_id, handler, headers),
                MessageType::Unary => MessageMetadata::new_unary(response_id, handler, headers),
            };

            let message = ActiveMessage {
                metadata,
                payload: payload.unwrap_or_default(),
            };

            if let Err(e) = client.send_message(target, message) {
                tracing::error!(
                    target: "dynamo_nova::client",
                    error = %e,
                    "Failed to send message after discovery"
                );
                let _ = response_manager
                    .complete_outcome(response_id, Err(format!("Send failed: {}", e)));
            }
        });
    }

    pub async fn fire(self) -> Result<()> {
        let target = self.resolve_target()?;

        // Fast path: if we can send directly, do so immediately
        if self.client.can_send_directly(target, &self.handler) {
            let outcome = self.client.register_outcome()?;
            let message = ActiveMessage {
                metadata: MessageMetadata::new_fire(
                    outcome.response_id(),
                    self.handler,
                    self.headers,
                ),
                payload: self.payload.unwrap_or_default(),
            };
            return self.client.send_message(target, message);
        }

        // Slow path: spawn task to handle discovery, then send
        // For fire-and-forget, we don't wait for the task to complete
        // perhaps we dissolve this  or into_parts() this, unless we can move self into the task as say `this`
        let client = self.client.clone();
        let handler = self.handler.clone();
        let payload = self.payload.clone();
        let headers = self.headers.clone();

        tokio::spawn(async move {
            match client.ensure_peer_ready(target, &handler).await {
                Ok(_) => match client.register_outcome() {
                    Ok(outcome) => {
                        let message = ActiveMessage {
                            metadata: MessageMetadata::new_fire(
                                outcome.response_id(),
                                handler,
                                headers,
                            ),
                            payload: payload.unwrap_or_default(),
                        };
                        if let Err(e) = client.send_message(target, message) {
                            tracing::error!(
                                target: "dynamo_nova::client",
                                error = %e,
                                "Failed to send fire-and-forget message"
                            );
                        }
                    }
                    Err(e) => {
                        tracing::error!(
                            target: "dynamo_nova::client",
                            error = %e,
                            "Failed to register outcome for fire-and-forget"
                        );
                    }
                },
                Err(e) => {
                    tracing::error!(
                        target: "dynamo_nova::client",
                        error = %e,
                        handler = %handler,
                        target = %target,
                        "Failed to prepare peer for message"
                    );
                }
            }
        });

        Ok(())
    }

    pub fn sync(self) -> SyncResult {
        let target_result = self.resolve_target();
        let worker_id = self.target_worker;

        let awaiter = match self.client.register_outcome() {
            Ok(awaiter) => awaiter,
            Err(e) => {
                tracing::error!(target: "dynamo_nova::client", error = %e, "Failed to register outcome");
                panic!("Failed to register outcome: {}", e);
            }
        };

        let response_id = awaiter.response_id();

        match target_result {
            Ok(target) if self.client.can_send_directly(target, &self.handler) => {
                // Fast path: send immediately
                let message = ActiveMessage {
                    metadata: self.create_metadata(response_id, MessageType::Sync),
                    payload: self.payload.unwrap_or_default(),
                };

                if let Err(e) = self.client.send_message(target, message) {
                    tracing::error!(target: "dynamo_nova::client", error = %e, "Failed to send message in fast path");
                }

                SyncResult::new(awaiter)
            }
            Ok(target) => {
                // Slow path: handshake needed
                self.spawn_handshake_task(target, response_id, MessageType::Sync);
                SyncResult::new(awaiter)
            }
            Err(ResolveError::UnresolvedPeer) => {
                // Slow path: discovery + handshake needed
                let Some(worker_id) = worker_id else {
                    tracing::error!(target: "dynamo_nova::client", "UnresolvedPeer but no worker_id set");
                    return SyncResult::new(awaiter);
                };

                self.spawn_discovery_task(worker_id, response_id, MessageType::Sync);
                SyncResult::new(awaiter)
            }
            Err(ResolveError::Other(e)) => {
                // Validation error - complete immediately with error
                tracing::error!(target: "dynamo_nova::client", error = %e, "Target resolution failed");
                let _ = self
                    .client
                    .response_manager
                    .complete_outcome(response_id, Err(format!("Resolution failed: {}", e)));
                SyncResult::new(awaiter)
            }
        }
    }

    pub fn unary(self) -> UnaryResult {
        let target_result = self.resolve_target();
        let worker_id = self.target_worker;

        let awaiter = match self.client.register_outcome() {
            Ok(awaiter) => awaiter,
            Err(e) => {
                tracing::error!(target: "dynamo_nova::client", error = %e, "Failed to register outcome");
                panic!("Failed to register outcome: {}", e);
            }
        };

        let response_id = awaiter.response_id();

        match target_result {
            Ok(target) if self.client.can_send_directly(target, &self.handler) => {
                // Fast path: send immediately
                let message = ActiveMessage {
                    metadata: self.create_metadata(response_id, MessageType::Unary),
                    payload: self.payload.unwrap_or_default(),
                };

                if let Err(e) = self.client.send_message(target, message) {
                    tracing::error!(target: "dynamo_nova::client", error = %e, "Failed to send message in fast path");
                }

                UnaryResult::new(awaiter)
            }
            Ok(target) => {
                // Slow path: handshake needed
                self.spawn_handshake_task(target, response_id, MessageType::Unary);
                UnaryResult::new(awaiter)
            }
            Err(ResolveError::UnresolvedPeer) => {
                // Slow path: discovery + handshake needed
                let Some(worker_id) = worker_id else {
                    tracing::error!(target: "dynamo_nova::client", "UnresolvedPeer but no worker_id set");
                    return UnaryResult::new(awaiter);
                };

                self.spawn_discovery_task(worker_id, response_id, MessageType::Unary);
                UnaryResult::new(awaiter)
            }
            Err(ResolveError::Other(e)) => {
                // Validation error - complete immediately with error
                tracing::error!(target: "dynamo_nova::client", error = %e, "Target resolution failed");
                let _ = self
                    .client
                    .response_manager
                    .complete_outcome(response_id, Err(format!("Resolution failed: {}", e)));
                UnaryResult::new(awaiter)
            }
        }
    }

    pub fn typed<R>(self) -> TypedUnaryResult<R>
    where
        R: DeserializeOwned + Send + 'static,
    {
        let target_result = self.resolve_target();
        let worker_id = self.target_worker;

        let awaiter = match self.client.register_outcome() {
            Ok(awaiter) => awaiter,
            Err(e) => {
                tracing::error!(target: "dynamo_nova::client", error = %e, "Failed to register outcome");
                panic!("Failed to register outcome: {}", e);
            }
        };

        let response_id = awaiter.response_id();

        match target_result {
            Ok(target) if self.client.can_send_directly(target, &self.handler) => {
                // Fast path: send immediately
                let message = ActiveMessage {
                    metadata: self.create_metadata(response_id, MessageType::Unary),
                    payload: self.payload.unwrap_or_default(),
                };

                if let Err(e) = self.client.send_message(target, message) {
                    tracing::error!(target: "dynamo_nova::client", error = %e, "Failed to send message in fast path");
                }

                TypedUnaryResult::new(awaiter)
            }
            Ok(target) => {
                // Slow path: handshake needed
                self.spawn_handshake_task(target, response_id, MessageType::Unary);
                TypedUnaryResult::new(awaiter)
            }
            Err(ResolveError::UnresolvedPeer) => {
                // Slow path: discovery + handshake needed
                let Some(worker_id) = worker_id else {
                    tracing::error!(target: "dynamo_nova::client", "UnresolvedPeer but no worker_id set");
                    return TypedUnaryResult::new(awaiter);
                };

                self.spawn_discovery_task(worker_id, response_id, MessageType::Unary);
                TypedUnaryResult::new(awaiter)
            }
            Err(ResolveError::Other(e)) => {
                // Validation error - complete immediately with error
                tracing::error!(target: "dynamo_nova::client", error = %e, "Target resolution failed");
                let _ = self
                    .client
                    .response_manager
                    .complete_outcome(response_id, Err(format!("Resolution failed: {}", e)));
                TypedUnaryResult::new(awaiter)
            }
        }
    }
}

pub(crate) fn validate_handler_name(handler: &str) -> Result<()> {
    if handler.starts_with('_') {
        anyhow::bail!(
            "Cannot directly call system handler '{}'. Use client convenience methods instead: health_check(), ensure_bidirectional_connection(), list_handlers(), await_handler()",
            handler
        );
    }
    Ok(())
}
