// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Leader-facing interfaces for conditional disaggregation.
//!
//! This module is deliberately narrow. It describes the pieces of a connector
//! leader that disaggregation code needs, without putting the
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
//! [`ConditionalDisaggLeader`] (in [`leader`]) holds the decode and
//! prefill flow leaders as `Option<Arc<...>>` and dispatches per request
//! by reading `TransferParams::remote_prefill` off the slot:
//! `Some(..)` ⇒ prefill flow; `None` ⇒ decode flow. Each instance wires
//! exactly one flow today (the kvbm-hub registers under a single role)
//! but the leader is forward-compatible with both flows simultaneously.

pub mod coordinator;
pub mod decode;
pub mod decode_leader;
pub mod leader;
pub mod lifecycle;
pub mod prefill_coordinator;
pub mod prefill_leader;
pub mod queue;

// The block-transport seam and hub peer resolver moved to the P2P feature
// module (`super::p2p`); CD builds on them. Re-exported below for path
// stability.

#[cfg(any(test, feature = "testing"))]
pub mod testing;

use std::collections::HashSet;
use std::sync::Arc;

use anyhow::Result;
use kvbm_protocols::disagg::TransferParams;

use crate::BlockId;
use crate::common::RequestMetadata;
use crate::connector::leader::scheduler::{KvConnectorMetadata, SchedulerOutput};
use crate::connector::leader::{ConnectorLeader, FinishedStatus, Request};

pub use crate::connector::leader::p2p::peer_resolver::{HubPeerResolver, PeerResolver};
pub use crate::connector::leader::p2p::transport::{
    ConnectorLeaderShim, EngineP2pBlockTransport, InnerLeaderShim, InnerLeaderWorkerHook,
    P2pBlockTransport, P2pWorkerHook,
};
pub use coordinator::{
    CdRegisterObserverFn, ConditionalDisaggCoordinator, CoordinatorParts, RemotePrefillStart,
};
pub use decode::{BeginOutcome, CdFailureSink, RemotePrefillStatus};
pub use decode_leader::{
    DecodeDisaggLeader, DecodeTierCache, HubWiring, install_tier_signal_handler,
};
pub use leader::{
    ConditionalDisaggLeader, ConditionalDisaggLeaderBuilder, build_hub_client, register_with_hub,
};
pub use lifecycle::{LIFECYCLE_WATCHDOG, LifecycleOutcome, spawn_lifecycle_watcher};
pub use prefill_leader::PrefillDisaggLeader;
pub use queue::{HubRemotePrefillQueue, RemotePrefillQueue};

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

/// Outcome of a per-request disaggregation policy evaluation.
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

/// Threshold policy: disaggregate (`Remote`) only when the *uncached* prefill
/// work meets `min_remote_prefill_tokens`, otherwise prefill locally on the
/// decode worker. The comparison is against
/// [`PolicyInputs::num_prefill_tokens`] (`total − num_computed − local
/// connector match`) — i.e. the tokens that still need a prefill forward pass
/// after local cache — not the raw prompt length.
///
/// A threshold of `0` makes every request `Remote` (equivalent to
/// [`AlwaysRemote`]); larger values keep short prompts local.
#[derive(Debug, Clone, Copy)]
pub struct ThresholdRemote {
    pub min_remote_prefill_tokens: usize,
}

impl ConditionalDisaggPolicy for ThresholdRemote {
    fn evaluate(&self, inputs: &PolicyInputs) -> PrefillSelection {
        if inputs.num_prefill_tokens() >= self.min_remote_prefill_tokens {
            PrefillSelection::Remote
        } else {
            PrefillSelection::Local
        }
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

    #[test]
    fn threshold_remote_gates_on_uncached_prefill_tokens() {
        let policy = ThresholdRemote {
            min_remote_prefill_tokens: 256,
        };
        let cold = |total: usize| PolicyInputs {
            total_tokens: total,
            num_computed_tokens: 0,
            num_connector_tokens: 0,
            transfer_params: None,
        };
        // Cold prompt of 300 uncached tokens >= 256 -> disaggregate.
        assert_eq!(policy.evaluate(&cold(300)), PrefillSelection::Remote);
        // Cold prompt of 200 uncached tokens < 256 -> local prefill.
        assert_eq!(policy.evaluate(&cold(200)), PrefillSelection::Local);
        // Boundary: exactly 256 uncached -> Remote (>=).
        assert_eq!(policy.evaluate(&cold(256)), PrefillSelection::Remote);
        // 300 total but mostly cached (100 computed + 60 matched) => 140
        // uncached < 256 -> local prefill, even though the prompt is long.
        let mostly_cached = PolicyInputs {
            total_tokens: 300,
            num_computed_tokens: 100,
            num_connector_tokens: 60,
            transfer_params: None,
        };
        assert_eq!(mostly_cached.num_prefill_tokens(), 140);
        assert_eq!(policy.evaluate(&mostly_cached), PrefillSelection::Local);
        // Threshold 0 is equivalent to AlwaysRemote.
        let always = ThresholdRemote {
            min_remote_prefill_tokens: 0,
        };
        assert_eq!(always.evaluate(&cold(1)), PrefillSelection::Remote);
    }
}
