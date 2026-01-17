// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use anyhow::Context;
use anyhow::Result;
use async_trait::async_trait;
use futures::stream::StreamExt;
use futures::{Stream, TryStreamExt};
use serde::Deserialize;
use serde::Serialize;

use crate::component::Namespace;
use crate::metrics::{MetricsHierarchy, MetricsRegistry};
use crate::traits::DistributedRuntimeProvider;
use crate::traits::events::{EventPublisher, EventSubscriber};

#[async_trait]
impl EventPublisher for Namespace {
    fn subject(&self) -> String {
        format!("namespace.{}", self.name)
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
impl EventSubscriber for Namespace {
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

impl MetricsHierarchy for Namespace {
    fn basename(&self) -> String {
        self.name.clone()
    }

    fn parent_hierarchies(&self) -> Vec<&dyn MetricsHierarchy> {
        let mut parents = vec![];

        // Walk up the namespace parent chain (grandparents to immediate parent)
        let parent_chain: Vec<&Namespace> =
            std::iter::successors(self.parent.as_deref(), |ns| ns.parent.as_deref()).collect();

        // Add DRT first (root)
        parents.push(&*self.runtime as &dyn MetricsHierarchy);

        // Then add parent namespaces in reverse order (root -> leaf)
        for parent_ns in parent_chain.iter().rev() {
            parents.push(*parent_ns as &dyn MetricsHierarchy);
        }

        parents
    }

    fn get_metrics_registry(&self) -> &MetricsRegistry {
        &self.metrics_registry
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
    async fn test_publish() {
        let rt = Runtime::from_current().unwrap();
        let dtr = DistributedRuntime::from_settings(rt.clone()).await.unwrap();
        let ns = dtr.namespace("test_namespace_publish".to_string()).unwrap();
        ns.publish("test_event", &"test".to_string()).await.unwrap();
        rt.shutdown();
    }

    #[tokio::test]
    async fn test_subscribe() {
        let rt = Runtime::from_current().unwrap();
        let dtr = DistributedRuntime::from_settings(rt.clone()).await.unwrap();
        let ns = dtr
            .namespace("test_namespace_subscribe".to_string())
            .unwrap();

        // Create a subscriber
        let mut subscriber = ns.subscribe("test_event").await.unwrap();

        // Publish a message
        ns.publish("test_event", &"test_message".to_string())
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
