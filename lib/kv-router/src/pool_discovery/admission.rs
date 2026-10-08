// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use async_trait::async_trait;

use super::PoolDiscoveryError;
use super::identity::VerifiedRelayIdentity;
use super::record::PoolRecord;
use crate::global_view::PoolKey;

/// Decides whether an authenticated Relay may stream state for a pool.
/// Usually a `PoolDirectory` lookup plus a `relay_identity` check.
/// The runtime keys the stream by its `PlaneLease`, not by the claimed key.
#[async_trait]
pub trait PoolAdmission: Send + Sync {
    async fn admit(
        &self,
        key: &PoolKey,
        caller: &VerifiedRelayIdentity,
    ) -> Result<PoolRecord, PoolDiscoveryError>;
}
