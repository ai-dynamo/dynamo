// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::sync::Arc;

use async_trait::async_trait;
use dynamo_global_common::PoolId;

/// Scope is supplied by the deployment, never inferred from client headers.
/// All replicas sharing bindings must use the same scope and policy namespace.
///
/// `tenant` is an opaque digest of the caller's credential (see
/// [`tenant_id`](super::tenant_id)), so two tenants that reuse one session ID
/// never share a binding. It is not an admission decision: the pool still
/// authorizes the request.
///
/// `scope` and `policy` are fixed for a selector's lifetime, so they are
/// shared rather than copied into every request's key.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct AffinityKey {
    pub scope: Arc<str>,
    pub tenant: String,
    pub policy: Arc<str>,
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

/// What a store must find in place for a conditional write to apply.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub enum Expected {
    /// No record at all, so only the first writer creates one.
    #[default]
    Absent,
    /// Exactly this revision, so the writer that read it wins.
    Revision(String),
    /// Whatever is there. Repairs a record no reader in this deployment can
    /// decode, which no other expectation can replace: its revision cannot be
    /// read and `Absent` never matches a present item. Concurrent repairs are
    /// last-writer-wins, which is within what soft affinity already allows.
    Unreadable,
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
/// records too: the tracker needs their revision to safely replace them, and
/// they report an undecodable record as [`AffinityError::InvalidRecord`] so
/// the tracker can repair it rather than leave the session unbindable.
///
/// Conditional mutation is atomic within the backend's consistency domain.
/// For DynamoDB MREC this is the contacted regional replica, not all regions.
/// A transport error may occur after a write has committed; callers must not
/// roll it back or assume that an error means no write occurred.
///
/// This trait is internal to the Global Router. It is public only so that a
/// backend can live in its own crate, and it changes without a major version
/// while persistent affinity is being built out.
#[async_trait]
pub trait AffinityStore: Send + Sync {
    async fn read(&self, key: &AffinityKey) -> Result<Option<AffinityBinding>, AffinityError>;

    /// Applies `binding` only if the stored record still matches `expected`.
    /// A store must not silently convert an expired item into an absent item.
    async fn compare_exchange(
        &self,
        key: &AffinityKey,
        expected: &Expected,
        binding: &AffinityBinding,
    ) -> Result<WriteOutcome, AffinityError>;
}
