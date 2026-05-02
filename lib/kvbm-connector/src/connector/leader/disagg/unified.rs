// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Per-request dispatching conditional-disagg leader.
//!
//! Replaces the per-instance role dispatch in
//! [`super::ConditionalDisaggLeader`] with a per-request rule:
//!
//! - Slot has `TransferParams::remote_prefill = Some(..)` → route to the
//!   wired prefill flow (this instance is the puller; the request was
//!   produced by a remote decode peer that already opened a session).
//! - Slot has no `TransferParams` → route to the wired decode flow (this
//!   instance is the initiator; the policy decides whether the request
//!   actually goes remote).
//! - Classified flow not wired → fall through to the inner leader
//!   (matches today's prefill-side `non_cd` passthrough behavior).
//!
//! The unified leader holds the existing role-specific leaders unchanged
//! ([`super::DecodeDisaggLeader`] / [`super::PrefillDisaggLeader`]) and
//! dispatches per-request.  Both can be wired simultaneously; today's
//! production wiring sets exactly one because the kvbm-hub registers
//! each instance under a single role and queue routing is single-role.
//!
//! ### Why this exists alongside [`super::ConditionalDisaggLeader`]
//!
//! The old [`super::ConditionalDisaggLeader`] is a role-dispatching
//! wrapper that holds an `Arc<dyn ConnectorLeaderApi>` chosen at init
//! time from `DisaggConfig::role`.  Its impl is a pure forward to the
//! inner; it has no per-request logic.  The unified leader replaces
//! that role wiring with TransferParams-based per-request dispatch,
//! enabling future homogeneous (single-role-per-instance lifted)
//! deployments.  The old type stays in place during validation; it
//! will be removed once the unified leader has soaked.
//!
//! ### Audit semantics during transition
//!
//! When only one flow is wired (the production case during validation),
//! audits emitted by the wrapped leader carry the same `role` tag as
//! before — `role = "decode"` on a decode-only instance, `role =
//! "prefill"` on a prefill-only instance.  When both flows are wired,
//! per-request audits inside the leaders still tag accurately because
//! each leader knows its own role; only `create_slot` /
//! `update_connector_output` / `build_connector_meta` (which precede
//! per-request classification or are per-tick) prefer the
//! decode-or-prefill flow that is wired.

use std::collections::HashSet;
use std::sync::Arc;

use anyhow::{Result, anyhow};
use kvbm_hub::{ConditionalDisaggClient, HubClient};
use velo::InstanceId;

use crate::BlockId;
use crate::connector::leader::scheduler::{KvConnectorMetadata, SchedulerOutput};
use crate::connector::leader::{FinishedStatus, Request};

use super::ConnectorLeaderApi;
use super::decode_leader::DecodeDisaggLeader;
use super::prefill_coordinator::PrefillCoordinator;
use super::prefill_leader::PrefillDisaggLeader;
use super::transport::InnerLeaderShim;
use crate::connector::leader::audit::audit_build_meta;

/// Per-request classification used by [`UnifiedDisaggLeader`] to choose
/// which wired flow handles a given API call.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum RequestRole {
    /// Route to the prefill leader.  Either the slot carries
    /// `TransferParams::remote_prefill = Some(..)` (CD-bound prefill
    /// puller) and prefill is wired, or only prefill is wired and the
    /// leader's own non-CD passthrough handles non-CD requests.
    Prefill,
    /// Route to the decode leader.  Either the slot has no
    /// `TransferParams` and decode is wired, or only decode is wired
    /// and the leader's own policy handles whatever the slot carries.
    Decode,
    /// Neither flow is wired — pure passthrough to the inner leader.
    NonCd,
}

/// Per-request-dispatching conditional-disaggregation leader.
///
/// Construct with [`UnifiedDisaggLeader::builder`].
///
/// ### Tick-level method semantics
///
/// For methods that aren't scoped to a specific request id
/// (`create_slot`, `build_connector_meta`, `update_connector_output`),
/// the unified leader calls `inner.<method>` exactly once and applies
/// each wired flow's decorator (audit emissions and, for prefill,
/// the coordinator's `observe_forward`) on top.  This is the
/// "decorator" model — flows observe a single canonical inner call
/// rather than each driving their own.
///
/// Per-request methods (`get_num_new_matched_tokens`,
/// `update_state_after_alloc`, `request_finished`) classify the
/// request via `slot_transfer_params` and dispatch to exactly the
/// wrapped flow leader for that request's role; the flow leader
/// drives its own inner call as today.
pub struct UnifiedDisaggLeader {
    inner: Arc<dyn InnerLeaderShim>,
    decode: Option<Arc<DecodeDisaggLeader>>,
    prefill: Option<Arc<PrefillDisaggLeader>>,
    /// Prefill coordinator held independently of `prefill` so the
    /// unified leader's `build_connector_meta` can invoke
    /// `observe_forward` without going through the leader (which
    /// would call `inner.build_connector_meta` again).
    prefill_coordinator: Option<Arc<dyn PrefillCoordinator>>,
    hub: Option<Arc<HubClient>>,
    client: Option<Arc<ConditionalDisaggClient>>,
    hub_velo_id: Option<InstanceId>,
}

impl std::fmt::Debug for UnifiedDisaggLeader {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("UnifiedDisaggLeader")
            .field("decode_wired", &self.decode.is_some())
            .field("prefill_wired", &self.prefill.is_some())
            .field("hub_velo_id", &self.hub_velo_id)
            .finish()
    }
}

/// Builder for [`UnifiedDisaggLeader`].  At least one flow must be set
/// before [`build`](Self::build).
pub struct UnifiedDisaggLeaderBuilder {
    inner: Arc<dyn InnerLeaderShim>,
    decode: Option<Arc<DecodeDisaggLeader>>,
    prefill: Option<Arc<PrefillDisaggLeader>>,
    prefill_coordinator: Option<Arc<dyn PrefillCoordinator>>,
    hub: Option<Arc<HubClient>>,
    client: Option<Arc<ConditionalDisaggClient>>,
    hub_velo_id: Option<InstanceId>,
}

impl UnifiedDisaggLeaderBuilder {
    pub fn with_decode(mut self, decode: Arc<DecodeDisaggLeader>) -> Self {
        self.decode = Some(decode);
        self
    }

    /// Wire the prefill flow.  Both the leader (for per-request
    /// dispatch in GNMT/USAA/request_finished) and its coordinator
    /// (for the tick-level `observe_forward` decorator on
    /// `build_connector_meta`) are required so the unified leader
    /// can run the prefill side without re-entering
    /// `PrefillDisaggLeader::build_connector_meta` (which would
    /// double-call inner).
    pub fn with_prefill(
        mut self,
        prefill: Arc<PrefillDisaggLeader>,
        coordinator: Arc<dyn PrefillCoordinator>,
    ) -> Self {
        self.prefill = Some(prefill);
        self.prefill_coordinator = Some(coordinator);
        self
    }

    pub fn with_hub(mut self, hub: Arc<HubClient>) -> Self {
        self.hub = Some(hub);
        self
    }

    pub fn with_client(mut self, client: Arc<ConditionalDisaggClient>) -> Self {
        self.client = Some(client);
        self
    }

    pub fn with_hub_velo_id(mut self, hub_velo_id: InstanceId) -> Self {
        self.hub_velo_id = Some(hub_velo_id);
        self
    }

    pub fn build(self) -> Result<Arc<UnifiedDisaggLeader>> {
        if self.decode.is_none() && self.prefill.is_none() {
            anyhow::bail!(
                "UnifiedDisaggLeader requires at least one flow to be wired \
                 (call .with_decode(..) and/or .with_prefill(.., ..))"
            );
        }
        if self.prefill.is_some() != self.prefill_coordinator.is_some() {
            anyhow::bail!(
                "with_prefill must be called with both leader and coordinator \
                 (internal invariant)"
            );
        }
        Ok(Arc::new(UnifiedDisaggLeader {
            inner: self.inner,
            decode: self.decode,
            prefill: self.prefill,
            prefill_coordinator: self.prefill_coordinator,
            hub: self.hub,
            client: self.client,
            hub_velo_id: self.hub_velo_id,
        }))
    }
}

impl UnifiedDisaggLeader {
    /// Begin a new builder.  `inner` is the shared shim used for the
    /// classification lookup (`slot_transfer_params`) and for fall-through
    /// routing when neither flow is wired for a given request role.
    pub fn builder(inner: Arc<dyn InnerLeaderShim>) -> UnifiedDisaggLeaderBuilder {
        UnifiedDisaggLeaderBuilder {
            inner,
            decode: None,
            prefill: None,
            prefill_coordinator: None,
            hub: None,
            client: None,
            hub_velo_id: None,
        }
    }

    pub fn decode_flow(&self) -> Option<&Arc<DecodeDisaggLeader>> {
        self.decode.as_ref()
    }

    pub fn prefill_flow(&self) -> Option<&Arc<PrefillDisaggLeader>> {
        self.prefill.as_ref()
    }

    pub fn hub(&self) -> Option<&Arc<HubClient>> {
        self.hub.as_ref()
    }

    pub fn client(&self) -> Option<&Arc<ConditionalDisaggClient>> {
        self.client.as_ref()
    }

    pub fn hub_velo_id(&self) -> Option<InstanceId> {
        self.hub_velo_id
    }

    /// Classify a request by reading the slot's transfer params once.
    ///
    /// Routing rule (in priority order):
    ///
    /// 1. **Both flows wired**: dispatch on `TransferParams::remote_prefill`.
    ///    Some(..) → Prefill; None → Decode.  This is the homogeneous
    ///    instance case.
    /// 2. **Only prefill wired**: route everything to Prefill.  The bare
    ///    `PrefillDisaggLeader::get_num_new_matched_tokens` already
    ///    handles non-CD requests via its `gnmt_passthrough_non_cd`
    ///    audit chain; falling through to inner here would skip those
    ///    audits and break trace-equivalence with the bare leader.
    /// 3. **Only decode wired**: route everything to Decode.  The bare
    ///    `DecodeDisaggLeader` does not consult `transfer_params`; its
    ///    policy decides local vs remote per request shape.
    /// 4. **Neither wired**: NonCd — pass through to inner.
    ///
    /// The classification is single-shot per call site and the source of
    /// truth is the slot — there is no in-leader cache.  If profiling
    /// shows this is hot, the classification can be memoized on the slot
    /// after `create_slot` populates `transfer_params`.
    fn classify(&self, request_id: &str) -> Result<RequestRole> {
        let role = match (self.decode.is_some(), self.prefill.is_some()) {
            (false, false) => RequestRole::NonCd,
            (true, false) => RequestRole::Decode,
            (false, true) => RequestRole::Prefill,
            (true, true) => {
                let params = self.inner.slot_transfer_params(request_id)?;
                let has_remote_prefill = params
                    .as_ref()
                    .and_then(|p| p.remote_prefill.as_ref())
                    .is_some();
                if has_remote_prefill {
                    RequestRole::Prefill
                } else {
                    RequestRole::Decode
                }
            }
        };
        Ok(role)
    }
}

impl ConnectorLeaderApi for UnifiedDisaggLeader {
    fn create_slot(&self, request: Request) -> Result<()> {
        // Decorator model: emit each wired flow's create_slot audit
        // (preserving its exact field shape — both flows happen to
        // emit the same fields, only `role` differs), then call
        // inner.create_slot exactly once.  When only one flow is
        // wired, the audit stream matches what the bare flow's
        // create_slot would emit, so trace-equivalence holds.
        let request_id = request.request_id.clone();
        let num_tokens = request.tokens.len();
        let has_kv_transfer = request
            .metadata
            .as_ref()
            .and_then(|m| m.kv_transfer_params.as_ref())
            .is_some();
        if self.decode.is_some() {
            crate::audit!(
                "create_slot",
                role = "decode",
                request_id = %request_id,
                num_tokens,
                has_kv_transfer
            );
        }
        if self.prefill.is_some() {
            crate::audit!(
                "create_slot",
                role = "prefill",
                request_id = %request_id,
                num_tokens,
                has_kv_transfer
            );
        }
        self.inner.create_slot(request)
    }

    fn has_slot(&self, request_id: &str) -> bool {
        // Pure forwarder — bare leaders emit no audit and only
        // delegate to inner.
        self.inner.has_slot(request_id)
    }

    fn extend_slot_tokens(&self, request_id: &str, tokens: Vec<u32>) -> Result<()> {
        // Pure forwarder — bare leaders emit no audit and only
        // delegate to inner.
        self.inner.extend_slot_tokens(request_id, tokens)
    }

    fn get_num_new_matched_tokens(
        &self,
        request_id: &str,
        num_computed_tokens: usize,
    ) -> Result<(Option<usize>, bool)> {
        match self.classify(request_id)? {
            RequestRole::Prefill => self
                .prefill
                .as_ref()
                .ok_or_else(|| anyhow!("classify=Prefill but prefill flow not wired"))?
                .get_num_new_matched_tokens(request_id, num_computed_tokens),
            RequestRole::Decode => self
                .decode
                .as_ref()
                .ok_or_else(|| anyhow!("classify=Decode but decode flow not wired"))?
                .get_num_new_matched_tokens(request_id, num_computed_tokens),
            RequestRole::NonCd => self
                .inner
                .get_num_new_matched_tokens(request_id, num_computed_tokens),
        }
    }

    fn update_state_after_alloc(
        &self,
        request_id: &str,
        block_ids: Vec<BlockId>,
        num_external_tokens: usize,
    ) -> Result<()> {
        match self.classify(request_id)? {
            RequestRole::Prefill => self
                .prefill
                .as_ref()
                .ok_or_else(|| anyhow!("classify=Prefill but prefill flow not wired"))?
                .update_state_after_alloc(request_id, block_ids, num_external_tokens),
            RequestRole::Decode => self
                .decode
                .as_ref()
                .ok_or_else(|| anyhow!("classify=Decode but decode flow not wired"))?
                .update_state_after_alloc(request_id, block_ids, num_external_tokens),
            RequestRole::NonCd => self
                .inner
                .update_state_after_alloc(request_id, block_ids, num_external_tokens),
        }
    }

    fn build_connector_meta(&self, output: SchedulerOutput) -> Result<KvConnectorMetadata> {
        // Decorator model: emit each wired flow's build_meta_entry
        // audit, call inner.build_connector_meta exactly once, then
        // run prefill's coordinator.observe_forward (the only
        // forward-pass-time hook either flow has today).  When only
        // one flow is wired the audit stream matches the bare flow's.
        if self.decode.is_some() {
            audit_build_meta("decode", &output);
        }
        if self.prefill.is_some() {
            audit_build_meta("prefill", &output);
        }
        let meta = self.inner.build_connector_meta(output)?;
        if let Some(coord) = &self.prefill_coordinator {
            // Mirror prefill_leader.rs: warn-log-and-emit-audit on
            // observe_forward error rather than propagating, because
            // observe_forward is informational and a failure here
            // shouldn't abort the scheduler tick.
            if let Err(err) = coord.observe_forward("", &meta) {
                tracing::warn!(error = %err, "observe_forward failed");
                crate::audit!(
                    "observe_forward_error",
                    role = "prefill",
                    error = %err
                );
            }
        }
        Ok(meta)
    }

    fn update_connector_output(
        &self,
        finished_sending: HashSet<String>,
        finished_recving: HashSet<String>,
    ) -> Result<()> {
        // Decorator model: emit per-request uco audits per wired
        // flow with that flow's role tag (decode also includes a
        // `cd_tracked` field sourced from its cd_request_state),
        // then call inner.update_connector_output exactly once.
        if let Some(decode) = &self.decode {
            for rid in &finished_sending {
                crate::audit!(
                    "uco_finished_sending",
                    role = "decode",
                    request_id = %rid,
                    cd_tracked = decode.has_active_cd_request(rid)
                );
            }
            for rid in &finished_recving {
                crate::audit!(
                    "uco_finished_recving",
                    role = "decode",
                    request_id = %rid,
                    cd_tracked = decode.has_active_cd_request(rid)
                );
            }
        }
        if self.prefill.is_some() {
            for rid in &finished_sending {
                crate::audit!(
                    "uco_finished_sending",
                    role = "prefill",
                    request_id = %rid
                );
            }
            for rid in &finished_recving {
                crate::audit!(
                    "uco_finished_recving",
                    role = "prefill",
                    request_id = %rid
                );
            }
        }
        self.inner
            .update_connector_output(finished_sending, finished_recving)
    }

    fn request_finished(&self, request_id: &str) -> FinishedStatus {
        // request_finished is per-request, classified the same as
        // GNMT/USAA.  May be called after slot teardown — both
        // leaders' request_finished is idempotent against unknown
        // request_ids (cd_request_state / coordinator state lookups
        // return None and noop), so a benign fallback to inner is
        // correct on classify error.
        match self.classify(request_id) {
            Ok(RequestRole::Prefill) => self
                .prefill
                .as_ref()
                .expect("classify=Prefill ⇒ flow wired")
                .request_finished(request_id),
            Ok(RequestRole::Decode) => self
                .decode
                .as_ref()
                .expect("classify=Decode ⇒ flow wired")
                .request_finished(request_id),
            Ok(RequestRole::NonCd) => self.inner.request_finished(request_id),
            Err(_) => self.inner.request_finished(request_id),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use crate::G2;
    use crate::connector::leader::disagg::testing::{MockInnerLeaderShim, MockSlot, TEST_BLOCK_SIZE};
    use kvbm_disagg_protocol::{RemotePrefillParams, TransferParams};
    use kvbm_engine::testing::managers::{TestManagerBuilder, TestRegistryBuilder};
    use kvbm_logical::manager::BlockManager;

    fn make_g2_manager() -> Arc<BlockManager<G2>> {
        let registry = TestRegistryBuilder::new().build();
        Arc::new(
            TestManagerBuilder::<G2>::new()
                .block_count(8)
                .block_size(TEST_BLOCK_SIZE)
                .registry(registry)
                .build(),
        )
    }

    fn install_slot_without_params(inner: &Arc<MockInnerLeaderShim>, request_id: &str) {
        inner.install_slot(
            request_id,
            MockSlot {
                block_size: TEST_BLOCK_SIZE,
                ..MockSlot::default()
            },
        );
    }

    fn install_slot_with_remote_prefill(inner: &Arc<MockInnerLeaderShim>, request_id: &str) {
        inner.install_slot(
            request_id,
            MockSlot {
                block_size: TEST_BLOCK_SIZE,
                transfer_params: Some(TransferParams::remote_prefill(RemotePrefillParams::new(
                    uuid::Uuid::new_v4(),
                    uuid::Uuid::new_v4().into(),
                ))),
                ..MockSlot::default()
            },
        );
    }

    /// Builder errors when no flow is wired.
    #[test]
    fn builder_requires_at_least_one_flow() {
        let inner = MockInnerLeaderShim::new(TEST_BLOCK_SIZE, make_g2_manager());
        let result = UnifiedDisaggLeader::builder(inner).build();
        let err = result.expect_err("expected builder error with no flows wired");
        assert!(
            err.to_string().contains("at least one flow"),
            "unexpected error: {err}"
        );
    }

    /// Build a leader with both flows unwired so we can exercise
    /// `classify`'s parameter-shape branches without standing up real
    /// `DecodeDisaggLeader` / `PrefillDisaggLeader` instances.  Real-flow
    /// routing (where `classify` returns `Decode` / `Prefill`) is
    /// exercised by the integration test in
    /// `tests/cd_unified_dispatch.rs`.
    fn no_flows_leader(inner: Arc<dyn InnerLeaderShim>) -> UnifiedDisaggLeader {
        UnifiedDisaggLeader {
            inner,
            decode: None,
            prefill: None,
            prefill_coordinator: None,
            hub: None,
            client: None,
            hub_velo_id: None,
        }
    }

    /// With no flows wired, `classify` returns `NonCd` regardless of
    /// transfer-params shape.  Dispatch falls through to inner.
    #[test]
    fn classify_returns_non_cd_when_no_flows_wired_no_params() {
        let inner = MockInnerLeaderShim::new(TEST_BLOCK_SIZE, make_g2_manager());
        install_slot_without_params(&inner, "req-x");
        let leader = no_flows_leader(inner);
        assert_eq!(
            leader.classify("req-x").unwrap(),
            RequestRole::NonCd,
            "no flows wired ⇒ NonCd"
        );
    }

    #[test]
    fn classify_returns_non_cd_when_no_flows_wired_with_remote_prefill() {
        let inner = MockInnerLeaderShim::new(TEST_BLOCK_SIZE, make_g2_manager());
        install_slot_with_remote_prefill(&inner, "req-x");
        let leader = no_flows_leader(inner);
        assert_eq!(
            leader.classify("req-x").unwrap(),
            RequestRole::NonCd,
            "no flows wired ⇒ NonCd even with remote_prefill params"
        );
    }

    /// With no flows wired, `classify` returns `NonCd` even for an
    /// unknown request_id — the slot lookup is short-circuited because
    /// neither flow needs the transfer-params shape (both fall back to
    /// inner regardless).  This is a deliberate change from the
    /// pre-refactor behavior that always queried the slot:
    /// `request_finished` arriving after slot teardown is now
    /// classified as `NonCd` rather than producing an error.
    #[test]
    fn classify_short_circuits_when_no_flows_wired() {
        let inner = MockInnerLeaderShim::new(TEST_BLOCK_SIZE, make_g2_manager());
        let leader = no_flows_leader(inner);
        // No slot installed — slot_transfer_params would error if
        // called, but classify must not call it because no flow needs
        // the result.
        assert_eq!(
            leader.classify("never-installed").unwrap(),
            RequestRole::NonCd
        );
    }

    // Single-flow and both-flows-wired routing matrix is exercised
    // by the integration tests:
    //   - tests/cd_unified_audit_equiv.rs           (decode-only)
    //   - tests/cd_unified_prefill_audit_equiv.rs   (prefill-only)
    //   - tests/cd_unified_both_wired.rs            (both wired)
    //   - tests/cd_unified_dispatch.rs              (cross-shape)
    // Constructing real `DecodeDisaggLeader` / `PrefillDisaggLeader`
    // instances inside this unit-test module would duplicate those
    // harnesses; the unit tests above focus narrowly on the
    // no-flows-wired short-circuit.
}
