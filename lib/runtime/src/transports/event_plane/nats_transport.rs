// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! NATS transport implementation for the event plane.

use anyhow::Result;
use async_trait::async_trait;
use bytes::Bytes;
use futures::StreamExt;

use super::transport::{EventTransportRx, EventTransportTx, WireStream};
use crate::DistributedRuntime;
use crate::discovery::EventTransportKind;

pub struct NatsTransport {
    drt: DistributedRuntime,
}

impl NatsTransport {
    pub fn new(drt: DistributedRuntime) -> Self {
        Self { drt }
    }
}

#[async_trait]
impl EventTransportTx for NatsTransport {
    async fn publish(&self, subject: &str, envelope_bytes: Bytes) -> Result<()> {
        self.drt
            .kv_router_nats_publish(subject.to_string(), envelope_bytes)
            .await
    }

    fn kind(&self) -> EventTransportKind {
        EventTransportKind::Nats
    }
}

#[async_trait]
impl EventTransportRx for NatsTransport {
    async fn subscribe(&self, subject: &str) -> Result<WireStream> {
        let subscriber = self
            .drt
            .kv_router_nats_subscribe(subject.to_string())
            .await?;

        let stream = subscriber.map(|msg| Ok(msg.payload));

        Ok(Box::pin(stream))
    }

    fn kind(&self) -> EventTransportKind {
        EventTransportKind::Nats
    }
}
