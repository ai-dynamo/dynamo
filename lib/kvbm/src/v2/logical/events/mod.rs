// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

pub mod batcher;
pub mod manager;
pub mod policy;
pub mod protocol;
pub mod publisher;

#[cfg(test)]
mod tests;

pub use batcher::{BatchingConfig, EventBatcher};
pub use manager::{EventsManager, EventsManagerBuilder, EventsManagerSettings};
pub use policy::{AllEventsPolicy, EventEmissionPolicy, PowerOfTwoPolicy};
pub use protocol::{InstanceId, KvCacheEvent, KvCacheEvents, KvbmCacheEvents};
pub use publisher::{KvbmCacheEventsPublisher, KvbmCacheEventsPublisherBuilder};
