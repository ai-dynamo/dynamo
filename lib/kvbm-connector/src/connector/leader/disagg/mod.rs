// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Leader-facing interfaces for conditional disaggregation.
//!
//! This module is deliberately narrow. It describes the pieces of a connector
//! leader that conditional-disaggregation code needs, without putting the
//! decode/prefill state machines into the base [`ConnectorLeader`].

pub mod decode;
pub mod queue;
pub mod session;

#[cfg(any(test, feature = "testing"))]
pub mod testing;

use std::collections::HashSet;
use std::sync::Arc;

use anyhow::Result;
use kvbm_disagg_protocol::TransferParams;

use crate::BlockId;
use crate::common::RequestMetadata;
use crate::connector::leader::scheduler::{KvConnectorMetadata, SchedulerOutput};
use crate::connector::leader::{ConnectorLeader, FinishedStatus, Request};

pub use decode::{BeginOutcome, RemotePrefillCoordinator, RemotePrefillState, RemotePrefillStatus};
pub use queue::RemotePrefillQueue;
pub use session::{
    PrefillSession, PrefillSessionFactory, SessionBlocks, SessionEvent, SessionEventStream,
    hash_to_wire,
};

/// Scheduler-facing connector leader API used by wrappers/compositions.
///
/// Implemented for `Arc<ConnectorLeader>` so future wrapper types can hold a
/// base local leader behind this trait and intercept only the methods they
/// need, such as GNMT and USAA.
pub trait ConnectorLeaderApi: Send + Sync {
    fn create_slot(&self, request: Request) -> Result<()>;

    fn has_slot(&self, request_id: &str) -> bool;

    fn extend_slot_tokens(&self, request_id: &str, tokens: Vec<u32>) -> Result<()>;

    fn get_num_new_matched_tokens(
        &self,
        request_id: &str,
        num_computed_tokens: usize,
    ) -> Result<(Option<usize>, bool)>;

    fn update_state_after_alloc(
        &self,
        request_id: &str,
        block_ids: Vec<BlockId>,
        num_external_tokens: usize,
    ) -> Result<()>;

    fn build_connector_meta(&self, output: SchedulerOutput) -> Result<KvConnectorMetadata>;

    fn update_connector_output(
        &self,
        finished_sending: HashSet<String>,
        finished_recving: HashSet<String>,
    ) -> Result<()>;

    fn request_finished(&self, request_id: &str) -> FinishedStatus;
}

impl ConnectorLeaderApi for Arc<ConnectorLeader> {
    fn create_slot(&self, request: Request) -> Result<()> {
        self.as_ref().create_slot(request)
    }

    fn has_slot(&self, request_id: &str) -> bool {
        self.as_ref().has_slot(request_id)
    }

    fn extend_slot_tokens(&self, request_id: &str, tokens: Vec<u32>) -> Result<()> {
        self.as_ref().extend_slot_tokens(request_id, tokens)
    }

    fn get_num_new_matched_tokens(
        &self,
        request_id: &str,
        num_computed_tokens: usize,
    ) -> Result<(Option<usize>, bool)> {
        self.as_ref()
            .get_num_new_matched_tokens(request_id, num_computed_tokens)
    }

    fn update_state_after_alloc(
        &self,
        request_id: &str,
        block_ids: Vec<BlockId>,
        num_external_tokens: usize,
    ) -> Result<()> {
        ConnectorLeader::update_state_after_alloc(self, request_id, block_ids, num_external_tokens)
    }

    fn build_connector_meta(&self, output: SchedulerOutput) -> Result<KvConnectorMetadata> {
        self.as_ref().build_connector_meta(output)
    }

    fn update_connector_output(
        &self,
        finished_sending: HashSet<String>,
        finished_recving: HashSet<String>,
    ) -> Result<()> {
        self.as_ref()
            .update_connector_output(finished_sending, finished_recving)
    }

    fn request_finished(&self, request_id: &str) -> FinishedStatus {
        self.as_ref().request_finished(request_id)
    }
}

/// Outcome of a per-request conditional-disaggregation policy evaluation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PrefillSelection {
    Local,
    Remote,
}

/// Inputs available when deciding whether a request should prefill locally or
/// remotely.
#[derive(Debug, Clone)]
pub struct PolicyInputs {
    pub total_tokens: usize,
    pub num_computed_tokens: usize,
    pub num_connector_tokens: usize,
    pub transfer_params: Option<TransferParams>,
}

impl PolicyInputs {
    pub fn num_prefill_tokens(&self) -> usize {
        self.total_tokens
            .saturating_sub(self.num_computed_tokens)
            .saturating_sub(self.num_connector_tokens)
    }
}

/// Per-request policy trait. The default implementation is `NeverRemote`,
/// which preserves today's local connector behavior.
pub trait ConditionalDisaggPolicy: Send + Sync + std::fmt::Debug {
    fn evaluate(&self, inputs: &PolicyInputs) -> PrefillSelection;
}

#[derive(Debug, Default, Clone, Copy)]
pub struct NeverRemote;

impl ConditionalDisaggPolicy for NeverRemote {
    fn evaluate(&self, _inputs: &PolicyInputs) -> PrefillSelection {
        PrefillSelection::Local
    }
}

#[derive(Debug, Default, Clone, Copy)]
pub struct AlwaysRemote;

impl ConditionalDisaggPolicy for AlwaysRemote {
    fn evaluate(&self, _inputs: &PolicyInputs) -> PrefillSelection {
        PrefillSelection::Remote
    }
}

/// Helper for policy call sites that have raw request metadata.
pub fn parse_transfer_params(metadata: Option<&RequestMetadata>) -> Result<Option<TransferParams>> {
    metadata
        .map(RequestMetadata::disagg_transfer_params)
        .transpose()
        .map_err(Into::into)
        .map(|params| params.flatten())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn assert_leader_api<T: ConnectorLeaderApi>() {}

    #[test]
    fn arc_connector_leader_implements_api() {
        assert_leader_api::<Arc<ConnectorLeader>>();
    }

    #[test]
    fn never_remote_preserves_local_default() {
        let inputs = PolicyInputs {
            total_tokens: 128,
            num_computed_tokens: 16,
            num_connector_tokens: 32,
            transfer_params: None,
        };

        assert_eq!(NeverRemote.evaluate(&inputs), PrefillSelection::Local);
        assert_eq!(inputs.num_prefill_tokens(), 80);
    }
}
