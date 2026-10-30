// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use async_trait::async_trait;
use dynamo_global_common::PoolId;

/// Scope is supplied by the deployment, never inferred from client headers.
/// All replicas sharing bindings must use the same scope and policy namespace.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct AffinityKey {
    pub scope: String,
    pub policy: String,
    pub model: String,
    pub session: String,
}

/// A selected pool, not proof of successful upstream dispatch. Times are Unix seconds.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct AffinityBinding {
    pub pool: PoolId,
    /// Unique per mutation, including expiration and recreation. Not a global lock.
    pub revision: String,
    pub updated_at: u64,
    pub expires_at: u64,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum WriteOutcome {
    Applied,
    Conflict,
}

#[derive(Debug, thiserror::Error)]
pub enum AffinityError {
    #[error("affinity storage unavailable: {0}")]
    Backend(String),
    #[error("invalid affinity record: {0}")]
    InvalidRecord(String),
    #[error("invalid affinity configuration: {0}")]
    Configuration(String),
}

/// Persistence contract for affinity records. Implementations return expired
/// records too: the tracker needs their revision to safely replace them.
///
/// Conditional mutation is atomic within the backend's consistency domain.
/// For DynamoDB MREC this is the contacted regional replica, not all regions.
/// A transport error may occur after a write has committed; callers must not
/// roll it back or assume that an error means no write occurred.
#[async_trait]
pub trait AffinityStore: Send + Sync {
    async fn read(&self, key: &AffinityKey) -> Result<Option<AffinityBinding>, AffinityError>;

    /// `None` requires an absent item; `Some` requires that exact revision.
    /// A store must not silently convert an expired item into an absent item.
    async fn compare_exchange(
        &self,
        key: &AffinityKey,
        expected_revision: Option<&str>,
        binding: &AffinityBinding,
    ) -> Result<WriteOutcome, AffinityError>;
}
