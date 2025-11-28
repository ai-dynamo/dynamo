// SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use anyhow::{Context, Result};
use async_trait::async_trait;
use futures::stream::StreamExt;
use futures::{Stream, TryStreamExt};
use serde::{Deserialize, Serialize};

use crate::component::Component;
use crate::traits::DistributedRuntimeProvider;
use crate::traits::events::{EventPublisher, EventSubscriber};

#[async_trait]
impl EventPublisher for Component {
    fn subject(&self) -> String {
        format!("namespace.{}.component.{}", self.namespace.name, self.name)
    }

    async fn publish(
        &self,
        event_name: impl AsRef<str> + Send + Sync,
        event: &(impl Serialize + Send + Sync),
    ) -> Result<()> {
        let bytes = serde_json::to_vec(event)?;
        self.publish_bytes(event_name, bytes).await
    }

    async fn publish_bytes(
        &self,
        event_name: impl AsRef<str> + Send + Sync,
        bytes: Vec<u8>,
    ) -> Result<()> {
        let subject = format!("{}.{}", self.subject(), event_name.as_ref());
        self.drt()
            .kv_router_nats_publish(subject, bytes.into())
            .await?;
        Ok(())
    }
}

#[async_trait]
impl EventSubscriber for Component {
    async fn subscribe(
        &self,
        event_name: impl AsRef<str> + Send + Sync,
    ) -> Result<async_nats::Subscriber> {
        let subject = format!("{}.{}", self.subject(), event_name.as_ref());
        Ok(self.drt().kv_router_nats_subscribe(subject).await?)
    }

    async fn subscribe_with_type<T: for<'de> Deserialize<'de> + Send + 'static>(
        &self,
        event_name: impl AsRef<str> + Send + Sync,
    ) -> Result<impl Stream<Item = Result<T>> + Send> {
        let subscriber = self.subscribe(event_name).await?;

        // Transform the subscriber into a stream of deserialized events
        let stream = subscriber.map(move |msg| {
            serde_json::from_slice::<T>(&msg.payload)
                .with_context(|| format!("Failed to deserialize event payload: {:?}", msg.payload))
        });

        Ok(stream)
    }
}

#[cfg(feature = "integration")]
#[cfg(test)]
mod tests {
    use crate::{DistributedRuntime, Runtime};

    use super::*;

    // todo - make a distributed runtime fixture
    // todo - two options - fully mocked or integration test
    #[tokio::test]
    async fn test_publish_and_subscribe() {
        let rt = Runtime::from_current().unwrap();
        let dtr = DistributedRuntime::from_settings(rt.clone()).await.unwrap();
        let ns = dtr.namespace("test_component".to_string()).unwrap();
        let cp = ns.component("test_component".to_string()).unwrap();

        // Create a subscriber on the component
        let mut subscriber = cp.subscribe("test_event").await.unwrap();

        // Publish a message from the component
        cp.publish("test_event", &"test_message".to_string())
            .await
            .unwrap();

        // Receive the message
        if let Some(msg) = subscriber.next().await {
            let received = String::from_utf8(msg.payload.to_vec()).unwrap();
            assert_eq!(received, "\"test_message\"");
        }

        rt.shutdown();
    }
}
