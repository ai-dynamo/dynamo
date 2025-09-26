// SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Message status types using typestate pattern for compile-time safety.

use anyhow::Result;
use bytes::Bytes;
use serde::de::DeserializeOwned;
use std::marker::PhantomData;
use std::time::Duration;
use tokio::sync::oneshot;
use uuid::Uuid;

/// TypeState marker: Default send-and-confirm mode
pub struct SendAndConfirm;

/// TypeState marker: Detached confirmation mode
pub struct DetachedConfirm;

/// TypeState marker: With additional response expected
pub struct WithResponse;

/// Message status object with typestate for compile-time safety
#[derive(Debug)]
pub struct MessageStatus<Mode> {
    pub(crate) message_id: Uuid,
    pub(crate) acceptance_rx: Option<oneshot::Receiver<()>>,
    pub(crate) response_rx: Option<oneshot::Receiver<Bytes>>,
    pub(crate) timeout: Duration,
    pub(crate) _mode: PhantomData<Mode>,
}

impl<Mode> MessageStatus<Mode> {
    /// Create a new message status
    pub(crate) fn new(
        message_id: Uuid,
        acceptance_rx: Option<oneshot::Receiver<()>>,
        response_rx: Option<oneshot::Receiver<Bytes>>,
        timeout: Duration,
    ) -> Self {
        Self {
            message_id,
            acceptance_rx,
            response_rx,
            timeout,
            _mode: PhantomData,
        }
    }

    /// Get the message ID
    pub fn message_id(&self) -> Uuid {
        self.message_id
    }
}

/// Methods only available for DetachedConfirm mode
impl MessageStatus<DetachedConfirm> {
    /// Wait for the handler to accept the message
    pub async fn await_accepted(self) -> Result<()> {
        let rx = self.acceptance_rx
            .ok_or_else(|| anyhow::anyhow!("No acceptance receiver configured"))?;

        tokio::time::timeout(self.timeout, rx)
            .await
            .map_err(|_| anyhow::anyhow!("Acceptance timeout after {:?}", self.timeout))?
            .map_err(|_| anyhow::anyhow!("Acceptance channel dropped"))?;

        Ok(())
    }
}

/// Methods only available for WithResponse mode
impl MessageStatus<WithResponse> {
    /// Wait for a typed response from the handler
    pub async fn await_response<T: DeserializeOwned>(self) -> Result<T> {
        let rx = self.response_rx
            .ok_or_else(|| anyhow::anyhow!("No response receiver configured"))?;

        let bytes = tokio::time::timeout(self.timeout, rx)
            .await
            .map_err(|_| anyhow::anyhow!("Response timeout after {:?}", self.timeout))?
            .map_err(|_| anyhow::anyhow!("Response channel dropped"))?;

        serde_json::from_slice(&bytes)
            .map_err(|e| anyhow::anyhow!("Failed to deserialize response: {}", e))
    }

    /// Wait for raw response bytes
    pub async fn await_response_raw(self) -> Result<Bytes> {
        let rx = self.response_rx
            .ok_or_else(|| anyhow::anyhow!("No response receiver configured"))?;

        let bytes = tokio::time::timeout(self.timeout, rx)
            .await
            .map_err(|_| anyhow::anyhow!("Response timeout after {:?}", self.timeout))?
            .map_err(|_| anyhow::anyhow!("Response channel dropped"))?;

        Ok(bytes)
    }
}

/// SendAndConfirm has no additional methods - acceptance was already awaited
impl MessageStatus<SendAndConfirm> {
    // Acceptance was already confirmed when this object was created
    // No additional methods needed
}