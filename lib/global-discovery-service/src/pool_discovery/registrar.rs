// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use async_trait::async_trait;
pub use dynamo_global_common::{Lease, PoolRegistration};

use super::DiscoveryError;

/// The write side. A PoolRelay registers its pool on a timer, and each call
/// is also the pool's heartbeat: registering an unchanged pool again only
/// renews it. The [`Lease`] says when to register again, and how long the
/// pool stays listed after its last registration, measured on the backend's
/// clock. PoolRelays get router replicas from a [`super::ReplicaDirectory`],
/// not from this reply.
///
/// V1 assumes a trusted network, where any caller may register any pool, so
/// `register` does not take the caller.
#[async_trait]
pub trait PoolRegistrar: Send + Sync {
    async fn register(&self, registration: PoolRegistration) -> Result<Lease, DiscoveryError>;
}
