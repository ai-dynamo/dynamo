// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use anyhow::Result;
use async_trait::async_trait;
use bytes::Bytes;
use dashmap::DashMap;
use dynamo_nova::am::Nova;

use std::sync::Arc;

use crate::v2::InstanceId;

use super::{OnboardSessionTx, SessionId, dispatch_onboard_message, messages::OnboardMessage};

/// Transport abstraction for sending onboarding messages.
///
/// This trait allows sessions to work with different transport mechanisms:
/// - Nova (distributed): Uses Nova active messages
/// - Local (testing): Direct channel dispatch
#[async_trait]
pub trait MessageTransport: Send + Sync {
    /// Send a message to a target instance.
    async fn send(&self, target: InstanceId, message: OnboardMessage) -> Result<()>;
}

/// Nova-based transport using active messages (fire-and-forget).
pub struct NovaTransport {
    nova: Arc<Nova>,
}

impl NovaTransport {
    pub fn new(nova: Arc<Nova>) -> Self {
        Self { nova }
    }
}

#[async_trait]
impl MessageTransport for NovaTransport {
    async fn send(&self, target: InstanceId, message: OnboardMessage) -> Result<()> {
        eprintln!(
            "[TRANSPORT] Sending {:?} to instance {}",
            std::mem::discriminant(&message),
            target
        );

        let bytes = Bytes::from(serde_json::to_vec(&message)?);

        self.nova
            .am_send("kvbm.leader.onboard")?
            .raw_payload(bytes)
            .instance(target)
            .send()
            .await?;

        eprintln!("[TRANSPORT] Successfully sent to {}", target);

        Ok(())
    }
}

/// Local transport for testing or same-instance communication.
///
/// Directly dispatches messages to session channels without network overhead.
pub struct LocalTransport {
    sessions: Arc<DashMap<SessionId, OnboardSessionTx>>,
}

impl LocalTransport {
    pub fn new(sessions: Arc<DashMap<SessionId, OnboardSessionTx>>) -> Self {
        Self { sessions }
    }
}

#[async_trait]
impl MessageTransport for LocalTransport {
    async fn send(&self, _target: InstanceId, message: OnboardMessage) -> Result<()> {
        dispatch_onboard_message(&self.sessions, message).await
    }
}
