// SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

use anyhow::Context;
use async_trait::async_trait;
use futures::stream::StreamExt;
use futures::{Stream, TryStreamExt};

use super::*;

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
        Ok(self
            .drt()
            .nats_client()
            .client()
            .publish(subject, bytes.into())
            .await?)
    }
}

#[async_trait]
impl EventSubscriber for Namespace {
    async fn subscribe(
        &self,
        event_name: impl AsRef<str> + Send + Sync,
    ) -> Result<async_nats::Subscriber> {
        let subject = format!("{}.{}", self.subject(), event_name.as_ref());
        Ok(self.drt().nats_client().client().subscribe(subject).await?)
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
    use super::*;

    // todo - make a distributed runtime fixture
    // todo - two options - fully mocked or integration test
    #[tokio::test]
    async fn test_publish() {
        let rt = Runtime::from_current().unwrap();
        let dtr = DistributedRuntime::from_settings(rt.clone()).await.unwrap();
        let ns = dtr.namespace("test".to_string()).unwrap();
        ns.publish("test", &"test".to_string()).await.unwrap();
        rt.shutdown();
    }

    #[tokio::test]
    async fn test_subscribe() {
        let rt = Runtime::from_current().unwrap();
        let dtr = DistributedRuntime::from_settings(rt.clone()).await.unwrap();
        let ns = dtr.namespace("test".to_string()).unwrap();

        // Create a subscriber
        let mut subscriber = ns.subscribe("test").await.unwrap();

        // Publish a message
        ns.publish("test", &"test_message".to_string())
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
