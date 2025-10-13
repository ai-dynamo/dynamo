// SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use std::fmt::Debug;

use crate::Result;

// #[async_trait]
// pub trait Publisher: Debug + Clone + Send + Sync {
//     async fn publish(&self, event: &(impl Serialize + Send + Sync)) -> Result<()>;
// }

/// An [EventPublisher] is an object that can publish events.
///
/// Each implementation of [EventPublisher] will define the root subject.
#[async_trait]
pub trait EventPublisher {
    /// The base subject used for this implementation of the [EventPublisher].
    fn subject(&self) -> String;

    /// Publish a single event to the event plane. The `event_name` will be `.` concatenated with the
    /// base subject provided by the implementation.
    async fn publish(
        &self,
        event_name: impl AsRef<str> + Send + Sync,
        event: &(impl Serialize + Send + Sync),
    ) -> Result<()>;

    /// Publish a single event as bytes to the event plane. The `event_name` will be `.` concatenated with the
    /// base subject provided by the implementation.
    async fn publish_bytes(
        &self,
        event_name: impl AsRef<str> + Send + Sync,
        bytes: Vec<u8>,
    ) -> Result<()>;

    // /// Create a new publisher for the given event name. The `event_name` will be `.` concatenated with the
    // /// base subject provided by the implementation.
    // fn publisher(&self, event_name: impl AsRef<str>) -> impl Publisher;

    // /// Create a new publisher for the given event name. The `event_name` will be `.` concatenated with the
    // fn publisher(&self, event_name: impl AsRef<str>) -> Result<Publisher>;
    // fn publisher_bytes(&self, event_name: impl AsRef<str>) -> &PublisherBytes;
}

/// An [EventSubscriber] is an object that can subscribe to events.
///
/// This trait provides methods to subscribe to events published on specific subjects.
#[async_trait]
pub trait EventSubscriber {
    /// Subscribe to events with the given event name.
    /// The `event_name` will be `.` concatenated with the base subject provided by the implementation.
    /// Returns a subscriber that can be used to receive events.
    async fn subscribe(
        &self,
        event_name: impl AsRef<str> + Send + Sync,
    ) -> Result<async_nats::Subscriber>;

    /// Subscribe to events with the given event name and deserialize them to the specified type.
    /// This is a convenience method that combines subscribe and deserialization.
    async fn subscribe_with_type<T: for<'de> Deserialize<'de> + Send + 'static>(
        &self,
        event_name: impl AsRef<str> + Send + Sync,
    ) -> Result<impl futures::Stream<Item = Result<T>> + Send>;
}
