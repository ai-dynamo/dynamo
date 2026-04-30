// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Leader-facing interfaces for conditional disaggregation.
//!
//! This module is deliberately narrow. It describes the pieces of a connector
//! leader that conditional-disaggregation code needs, without putting the
//! decode/prefill state machines into the base [`ConnectorLeader`].
//!
//! # Decode-side golden path
//!
//! 1. `get_num_new_matched_tokens` — wrapper calls inner local match. If the
//!    inner result is `(None, _)` or `(Some(0), _)`, pass through unchanged.
//! 2. Build [`PolicyInputs`] (`total`, `num_computed_tokens`,
//!    `num_connector_tokens = inner_match`) and evaluate the
//!    [`ConditionalDisaggPolicy`].
//! 3. On `Remote`: compute `((total - num_computed) / block_size) * block_size`
//!    full-block external tokens, try to reserve that many tokens from the
//!    inflight budget, and return `(None, false)` if the budget is
//!    exhausted (vLLM retries next forward pass). Otherwise insert per-
//!    request state and return `(Some(N), true)`.
//! 4. `update_state_after_alloc` — for CD-bound requests, capture the G1
//!    destination block IDs and `num_external_tokens`; do **not** call the
//!    inner USAA. The bytes will land via the `BlockSetsAdded` path.
//! 5. The prefill peer attaches via `factory.attach`, drains D's
//!    `commits()`/`availability()` to learn which blocks D will serve,
//!    calls `session.pull(...)` to RDMA-pull them, runs prefill, then
//!    `session.commit(...)` + `session.make_available(...)` to publish
//!    the output blocks back to D.
//! 6. The decode wrapper's per-request task drains P's
//!    `commits()`/`availability()` and `session.pull(...)`s the output
//!    blocks. Once both the local kick (G2→G1 for the local-match
//!    slice) and the remote pull pipeline complete, the wrapper calls
//!    `mark_onboarding_complete` and releases the inflight budget
//!    reservation.
//!
//! # No partial transfers
//!
//! GNMT returns `(Some(N), true)` only when **all** N tokens (full blocks
//! only — partial trailing tokens are excluded) will arrive via the CD
//! path. Failure of any kind is total: the wrapper releases the budget,
//! drops the per-request state, and surfaces the failure through
//! [`CdOutputSink::on_request_failed`].
//!
//! # Inflight budget
//!
//! Per-decode-leader, counted in *tokens* (not blocks, not requests).
//! Default is `usize::MAX` (unlimited) — operators opt in to admission
//! throttling via `DisaggConfig::max_inflight_remote_prefill_tokens`. The
//! unlimited path skips the atomic CAS entirely so it costs nothing when
//! disabled.
//!
//! # Role dispatch
//!
//! The wrapper reads the manager role per call (no caching) so a future
//! hub-controlled role-swap can change behavior on the next call without
//! restructuring this API. The Prefill arm is currently `todo!()` until
//! that side is wired in a follow-up.

pub mod conditional_leader;
pub mod decode;
pub mod decode_leader;
pub mod peer_resolver;
pub mod prefill_coordinator;
pub mod prefill_leader;
pub mod queue;
pub mod transport;

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

pub use conditional_leader::{ConditionalDisaggLeader, register_with_hub};
pub use decode::{BeginOutcome, RemotePrefillCoordinator, RemotePrefillState, RemotePrefillStatus};
pub use decode_leader::DecodeDisaggLeader;
pub use peer_resolver::{HubPeerResolver, PeerResolver};
pub use prefill_coordinator::PrefillCoordinator;
pub use prefill_leader::PrefillDisaggLeader;
pub use queue::{HubRemotePrefillQueue, RemotePrefillQueue};
pub use transport::{
    CdBlockTransport, CdWorkerHook, ConnectorLeaderShim, EngineCdBlockTransport, InnerLeaderShim,
    InnerLeaderWorkerHook,
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
