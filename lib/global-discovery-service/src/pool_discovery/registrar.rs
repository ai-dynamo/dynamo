// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use async_trait::async_trait;
use dynamo_global_common::Lease;

use super::PoolDiscoveryError;
use super::record::PoolRecord;

/// Reply to `register`. A PoolRelay registers again every heartbeat
/// interval, and the record expires a lease duration after the last
/// registration the store saw, measured on the store's clock. It does not
/// list router replicas: Relays get those from a [`super::ReplicaDirectory`],
/// served as `GET /routers`.
#[derive(Clone, Debug)]
pub struct Registration {
    pub lease: Lease,
}

/// Write side behind `POST /pools`. A PoolRelay calls `register` on a timer,
/// and each call is also the pool's heartbeat: registering an unchanged
/// record again only renews it. The memory store keeps records in this
/// process; the MCS store has no registrar.
///
/// V1 assumes a trusted network, where any caller may register any pool, so
/// `register` does not take the caller.
#[async_trait]
pub trait PoolRegistrar: Send + Sync {
    async fn register(&self, record: PoolRecord) -> Result<Registration, PoolDiscoveryError>;
}
