// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Post-selection KV hint planning.

use std::collections::HashSet;

use dynamo_kv_router::{
    kv_hints::KvHint,
    protocols::{ExternalSequenceBlockHash, WorkerWithDpRank},
};

use crate::protocols::common::extensions::AgentContext;

/// Read-only logical paths associated with one session on the selected worker.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct SessionLineageView {
    lineages: Vec<Vec<ExternalSequenceBlockHash>>,
}

impl SessionLineageView {
    pub fn new(lineages: Vec<Vec<ExternalSequenceBlockHash>>) -> Self {
        Self { lineages }
    }

    pub fn lineages(&self) -> &[Vec<ExternalSequenceBlockHash>] {
        &self.lineages
    }

    /// Return each logical block once while preserving root-first discovery order.
    pub fn unique_block_hashes(&self) -> Vec<ExternalSequenceBlockHash> {
        let mut seen = HashSet::new();
        self.lineages
            .iter()
            .flatten()
            .copied()
            .filter(|hash| seen.insert(*hash))
            .collect()
    }

    pub fn is_empty(&self) -> bool {
        self.lineages.is_empty()
    }
}

/// Immutable inputs available after Dynamo selects a worker.
pub struct KvHintPolicyContext<'a> {
    pub agent_context: Option<&'a AgentContext>,
    pub selected_worker: WorkerWithDpRank,
    pub session_lineage: Option<&'a SessionLineageView>,
}

#[derive(Debug, thiserror::Error)]
#[error("{message}")]
pub struct KvHintPolicyError {
    message: String,
}

impl KvHintPolicyError {
    pub fn new(message: impl Into<String>) -> Self {
        Self {
            message: message.into(),
        }
    }
}

/// Produces optional KV actions without changing worker selection.
pub trait KvHintPolicy: Send + Sync {
    fn evaluate(
        &self,
        context: &KvHintPolicyContext<'_>,
    ) -> Result<Option<KvHint>, KvHintPolicyError>;
}

/// Default policy behavior: attach no KV hints.
#[derive(Debug, Default)]
pub struct NoopKvHintPolicy;

impl KvHintPolicy for NoopKvHintPolicy {
    fn evaluate(
        &self,
        _context: &KvHintPolicyContext<'_>,
    ) -> Result<Option<KvHint>, KvHintPolicyError> {
        Ok(None)
    }
}
