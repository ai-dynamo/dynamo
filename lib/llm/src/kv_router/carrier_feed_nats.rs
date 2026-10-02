// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! NATS transport for the KVBM carrier feed.

use anyhow::Result;
use async_trait::async_trait;
use bytes::Bytes;
use dynamo_kv_router::carrier_feed_client::{
    CarrierFeedTransport, HubFeedConfig, nats_feed_subject,
};
use dynamo_runtime::DistributedRuntime;
use futures_util::{StreamExt, stream::BoxStream};

pub struct NatsFeedTransport {
    drt: DistributedRuntime,
}

impl NatsFeedTransport {
    pub fn new(drt: DistributedRuntime) -> Self {
        Self { drt }
    }
}

#[async_trait]
impl CarrierFeedTransport for NatsFeedTransport {
    fn event_plane(&self) -> &'static str {
        "nats"
    }

    async fn subscribe(&self, config: &HubFeedConfig) -> Result<BoxStream<'static, Result<Bytes>>> {
        if config.nats_subject_prefix.is_empty() {
            anyhow::bail!("hub advertised an empty NATS subject prefix");
        }
        let subscriber = self
            .drt
            .kv_router_nats_subscribe(nats_feed_subject(&config.nats_subject_prefix))
            .await?;
        Ok(subscriber.map(|message| Ok(message.payload)).boxed())
    }
}
