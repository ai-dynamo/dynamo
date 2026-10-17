// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Public post-selection KV-hint policy contract and startup configuration.

mod config;
mod registry;

pub use config::KvHintPolicyConfig;
pub(crate) use config::RawKvHintPolicyConfig;
pub(crate) use registry::{ConfiguredKvHintPolicy, KvHintPolicyRegistry};
pub use registry::{
    KvHintPolicyConstructor, KvHintPolicyConstructorError, KvHintPolicyParameters,
    KvHintPolicyRegistryError,
};

use std::error::Error;

use crate::{
    SessionContext, SessionPrefixIndexError, SessionPrefixIndexer,
    kv_hints::KvHintAction,
    protocols::{ExternalSequenceBlockHash, WorkerWithDpRank},
};

/// Request and routing state available after Dynamo has selected the target worker.
pub struct KvHintPolicyContext<'a> {
    request_id: &'a str,
    session_context: Option<&'a SessionContext>,
    selected_worker: WorkerWithDpRank,
    session_prefix_index: Option<&'a SessionPrefixIndexer>,
}

impl<'a> KvHintPolicyContext<'a> {
    #[doc(hidden)]
    pub fn new(
        request_id: &'a str,
        session_context: Option<&'a SessionContext>,
        selected_worker: WorkerWithDpRank,
        session_prefix_index: Option<&'a SessionPrefixIndexer>,
    ) -> Self {
        Self {
            request_id,
            session_context,
            selected_worker,
            session_prefix_index,
        }
    }

    /// The request identifier used by the routing and backend request lifecycle.
    pub fn request_id(&self) -> &str {
        self.request_id
    }

    /// Canonical session and agent-lifecycle metadata, when provided at ingress.
    pub fn session_context(&self) -> Option<&SessionContext> {
        self.session_context
    }

    /// The exact worker and data-parallel rank selected for this request.
    pub fn selected_worker(&self) -> WorkerWithDpRank {
        self.selected_worker
    }

    /// Find the logical node for an engine-provided sequence hash.
    pub fn get_node_from_hash(
        &self,
        block_hash: ExternalSequenceBlockHash,
    ) -> Option<crate::NodeId> {
        self.session_prefix_index?.get_node_from_hash(block_hash)
    }

    /// Read one logical node by its stable arena identifier.
    pub fn get_node(&self, node_id: crate::NodeId) -> Option<crate::LogicalNode> {
        self.session_prefix_index?.get_node(node_id)
    }

    /// Return the session's unordered worker-qualified frontier nodes.
    ///
    /// Returns `None` when session indexing is disabled and `Some(vec![])` when the session has no
    /// recorded frontier.
    pub fn get_session_frontiers(
        &self,
        session_id: &str,
    ) -> Option<Vec<(WorkerWithDpRank, crate::NodeId)>> {
        Some(self.session_prefix_index?.get_session_frontiers(session_id))
    }

    /// Return root-first block-hash paths for one session and worker.
    ///
    /// Event-derived index updates are ordered but asynchronous. This view reports committed
    /// state and does not guarantee that blocks created by the current request are present yet.
    /// Returns `Ok(None)` when session indexing is disabled.
    pub fn get_session_block_lineage(
        &self,
        session_id: &str,
        worker: WorkerWithDpRank,
        anchor_hash: Option<ExternalSequenceBlockHash>,
    ) -> Result<Option<Vec<Vec<ExternalSequenceBlockHash>>>, SessionPrefixIndexError> {
        self.session_prefix_index
            .map(|index| index.get_session_block_lineage(session_id, worker, anchor_hash))
            .transpose()
    }
}

/// Error returned by a user-provided [`KvHintPolicy`].
pub type KvHintPolicyError = dyn Error + Send + Sync + 'static;

/// User-provided policy that formulates additional KV-hint actions after worker selection.
///
/// The callback is synchronous and may run concurrently for different requests. Implementations
/// should return promptly and synchronize any mutable policy state internally.
pub trait KvHintPolicy: Send + Sync + 'static {
    fn formulate(
        &self,
        context: &KvHintPolicyContext<'_>,
    ) -> Result<Vec<KvHintAction>, Box<KvHintPolicyError>>;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn policy_context_resolves_only_the_requested_worker_lineage() {
        let index = SessionPrefixIndexer::new();
        let worker_a = WorkerWithDpRank::new(1, 0);
        let worker_b = WorkerWithDpRank::new(2, 0);
        let a = ExternalSequenceBlockHash(11);
        let ab = ExternalSequenceBlockHash(12);
        let ax = ExternalSequenceBlockHash(13);
        index
            .update_session_from_stored_blocks("session", worker_a, None, &[a, ab])
            .unwrap();
        index
            .update_session_from_stored_blocks("session", worker_b, Some(a), &[ax])
            .unwrap();

        let context = KvHintPolicyContext::new("request", None, worker_a, Some(&index));
        assert_eq!(context.request_id(), "request");
        assert_eq!(context.selected_worker(), worker_a);
        let a_node = context.get_node_from_hash(a).unwrap();
        assert_eq!(context.get_node(a_node).unwrap().block_hash(), a);
        assert_eq!(
            context
                .get_session_block_lineage("session", worker_a, None)
                .unwrap(),
            Some(vec![vec![a, ab]])
        );
        assert_eq!(
            context
                .get_session_block_lineage("session", worker_b, Some(a))
                .unwrap(),
            Some(vec![vec![a, ax]])
        );
    }
}
