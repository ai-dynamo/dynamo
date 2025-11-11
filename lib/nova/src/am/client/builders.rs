// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Convenience builders for trait-object clients (`&dyn ActiveMessageClient`).
//!
//! These wrappers delegate to [`MessageBuilder`] while keeping the ergonomic
//! `client.am_*()` APIs.

use std::future::Future;
use std::sync::Arc;

use anyhow::{Result, anyhow};
use bytes::Bytes;
use serde::de::DeserializeOwned;
use serde::{Deserialize, Serialize};

use super::ActiveMessageClient;
use crate::am::{
    InstanceId,
    common::{ActiveMessage, MessageMetadata},
};

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

    pub fn send(self) -> impl Future<Output = Result<()>> {
        self.inner.sync()
    }

    pub fn send_to(self, target: InstanceId) -> impl Future<Output = Result<()>> {
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

    pub fn send(self) -> impl Future<Output = Result<Bytes>> {
        self.inner.unary()
    }

    pub fn send_to(self, target: InstanceId) -> impl Future<Output = Result<Bytes>> {
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

    pub fn send(self) -> impl Future<Output = Result<R>> {
        self.inner.typed()
    }

    pub fn send_to(self, target: InstanceId) -> impl Future<Output = Result<R>> {
        self.inner.instance(target).typed()
    }
}

/// Minimal message builder supporting fire-and-forget and unary-style sends.
pub struct MessageBuilder {
    client: Arc<ActiveMessageClient>,
    handler: String,
    payload: Option<Bytes>,
    target_instance: Option<InstanceId>,
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

    fn resolve_target(&self) -> anyhow::Result<InstanceId> {
        self.target_instance.ok_or_else(|| {
            anyhow::anyhow!("target instance not set; call `instance(...)` before sending")
        })
    }

    pub async fn fire(self) -> Result<()> {
        let target = self.resolve_target()?;
        self.client
            .evaluate_handler_availability(target, &self.handler)
            .await?;
        let outcome = self.client.register_outcome().await?;
        let message = ActiveMessage {
            metadata: MessageMetadata::new_fire(outcome.response_id(), self.handler),
            payload: self.payload.unwrap_or_default(),
        };
        self.client.send_message(target, message).await
    }

    pub async fn sync(self) -> Result<()> {
        self.await_outcome(None, |outcome| match outcome {
            Ok(None) => Ok(()),
            Ok(Some(_bytes)) => Ok(()),
            Err(err) => Err(anyhow!(err)),
        })
        .await
    }

    pub async fn unary(self) -> Result<Bytes> {
        self.await_outcome(
            Some(ClientExpectation::unary_bytes()),
            |outcome| match outcome {
                Ok(Some(bytes)) => Ok(bytes),
                Ok(None) => Ok(Bytes::new()),
                Err(err) => Err(anyhow!(err)),
            },
        )
        .await
    }

    pub async fn typed<R>(self) -> Result<R>
    where
        R: DeserializeOwned + Send + 'static,
    {
        let expectation = ClientExpectation::unary_typed(std::any::type_name::<R>().to_string());
        let bytes = self
            .await_outcome(Some(expectation), |outcome| match outcome {
                Ok(Some(bytes)) => Ok(bytes),
                Ok(None) => Ok(Bytes::new()),
                Err(err) => Err(anyhow!(err)),
            })
            .await?;

        serde_json::from_slice(&bytes).map_err(|e| anyhow!("failed to deserialize response: {}", e))
    }

    async fn await_outcome<F, R>(self, expectation: Option<ClientExpectation>, map: F) -> Result<R>
    where
        F: FnOnce(Result<Option<Bytes>, String>) -> Result<R>,
    {
        let target = self.resolve_target()?;
        self.client
            .evaluate_handler_availability(target, &self.handler)
            .await?;
        let mut outcome = self.client.register_outcome().await?;
        let response_id = outcome.response_id();

        // Determine response type based on expectation
        let metadata = match expectation {
            Some(_) => MessageMetadata::new_unary(response_id, self.handler), // unary or typed unary
            None => MessageMetadata::new_sync(response_id, self.handler),     // sync
        };

        let message = ActiveMessage {
            metadata,
            payload: self.payload.unwrap_or_default(),
        };
        self.client.send_message(target, message).await?;

        let result = outcome.recv().await;
        map(result)
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub(crate) struct ClientExpectation;

impl ClientExpectation {
    pub fn active_message() -> Self {
        Self
    }

    pub fn unary_bytes() -> Self {
        Self
    }

    pub fn unary_typed(_response_type_id: String) -> Self {
        Self
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
