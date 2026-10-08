// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! The list-and-watch contract of `dynamo_runtime::discovery::Discovery`,
//! made generic so pools and router replicas share it.
//!
//! A watch starts with one `Added` per current item, then one `Resync` with the
//! same set, then changes. A `Resync` replaces all earlier state.

use std::pin::Pin;

use async_trait::async_trait;
use futures_util::Stream;
use tokio_util::sync::CancellationToken;

use super::PoolDiscoveryError;

#[derive(Clone, Debug)]
pub enum WatchEvent<T, K> {
    /// New or changed item. Replaces any earlier item with the same key.
    Added(T),
    Removed(K),
    Resync(Snapshot<T>),
}

#[derive(Clone, Debug)]
pub struct Snapshot<T> {
    pub items: Vec<T>,
    pub completeness: Completeness,
}

/// Lets a new router replica tell an exact "ready" from a best guess.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Completeness {
    /// Read from a shared store: every existing item is listed.
    Exact,
    /// Built from what this replica has heard so far, settled by a `SettlePolicy`.
    Estimated,
}

pub type WatchStream<T, K> =
    Pin<Box<dyn Stream<Item = Result<WatchEvent<T, K>, PoolDiscoveryError>> + Send>>;

/// Read side. Method names follow `dynamo_runtime::discovery::Discovery`.
#[async_trait]
pub trait Directory<T, K>: Send + Sync {
    async fn check_connection(&self) -> Result<(), PoolDiscoveryError>;

    async fn list(&self) -> Result<Snapshot<T>, PoolDiscoveryError>;

    async fn list_and_watch(
        &self,
        cancel_token: Option<CancellationToken>,
    ) -> Result<WatchStream<T, K>, PoolDiscoveryError>;
}
