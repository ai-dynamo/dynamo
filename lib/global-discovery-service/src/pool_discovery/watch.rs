// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! The list-and-watch contract of `dynamo_runtime::discovery::Discovery`,
//! made generic so pools and router replicas share it. Its events and
//! snapshots are the ones in `dynamo_global_common`.

use std::pin::Pin;

use async_trait::async_trait;
use futures::Stream;
use tokio_util::sync::CancellationToken;

pub use dynamo_global_common::{Completeness, Snapshot, WatchEvent};

use super::DiscoveryError;

pub type WatchStream<T, K> =
    Pin<Box<dyn Stream<Item = Result<WatchEvent<T, K>, DiscoveryError>> + Send>>;

/// Read side. Method names follow `dynamo_runtime::discovery::Discovery`.
#[async_trait]
pub trait Directory<T, K>: Send + Sync {
    async fn list(&self) -> Result<Snapshot<T>, DiscoveryError>;

    async fn list_and_watch(
        &self,
        cancel_token: Option<CancellationToken>,
    ) -> Result<WatchStream<T, K>, DiscoveryError>;
}
