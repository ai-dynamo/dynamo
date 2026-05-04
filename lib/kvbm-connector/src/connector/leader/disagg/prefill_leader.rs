// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Prefill-side conditional-disaggregation leader wrapper.
//!
//! `PrefillDisaggLeader` wraps a base inner [`InnerLeaderShim`]
//! and intercepts the scheduler-facing API
//! ([`ConnectorLeaderApi`]) on the prefill participant. See
//! `disagg/mod.rs` and the canonical plan
//! (`/home/ryan/.claude/plans/cd-usaa-pipeline.md` §"Phase A
//! redesign") for the full state machine.

use std::collections::HashSet;
use std::sync::Arc;

use anyhow::{Context, Result};
use kvbm_disagg_protocol::TransferParams;

use crate::BlockId;
use crate::connector::leader::scheduler::{KvConnectorMetadata, SchedulerOutput};
use crate::connector::leader::{FinishedStatus, Request};

use super::ConnectorLeaderApi;
use super::coordinator::ConditionalDisaggCoordinator;
use super::transport::{CdWorkerHook, InnerLeaderShim};

/// RAII payload installed on the prefill slot's `OnboardingState`
/// when a CD-bound request enters the wrapper with n>0 (decode
/// forwarded local-match cache that prefill must pull). Drop is
/// called when `process_finished_onboarding` takes the
/// `OnboardingState`; gives the canonical cleanup point a hook
/// to observe async-load completion. The Drop intentionally does
/// NOT call `coordinator.on_request_finished` — that would close
/// the session before the prefill side has forward-passed and
/// offloaded its net-new G2 blocks (which need to be published
/// back to decode via `session.commit` / `make_available` from
/// the offload-pipeline observer). Session lifecycle is owned
/// solely by `PrefillDisaggLeader::request_finished` (vLLM's
/// signal), which fires after offload settles.
struct PrefillCdOnboardingPayload {
    request_id: String,
    #[allow(dead_code)]
    coordinator: std::sync::Weak<ConditionalDisaggCoordinator>,
}

impl std::fmt::Debug for PrefillCdOnboardingPayload {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PrefillCdOnboardingPayload")
            .field("request_id", &self.request_id)
            .finish()
    }
}

impl crate::connector::leader::slot::CdOnboardingPayload for PrefillCdOnboardingPayload {}

impl Drop for PrefillCdOnboardingPayload {
    fn drop(&mut self) {
        // Audit-only marker: emitted when async-load (G2→G1) for
        // the pulled-from-decode prefix is complete and the slot's
        // OnboardingState has been taken by
        // `process_finished_onboarding`. Subsequent forward-pass +
        // G1→G2 offload + publish-back to decode all happen AFTER
        // this drop. Do NOT close the session here — see struct
        // doc above.
        crate::audit!(
            "prefill_cd_payload_drop",
            role = "prefill",
            request_id = %self.request_id
        );
    }
}

pub struct PrefillDisaggLeader {
    inner: Arc<dyn InnerLeaderShim>,
    coordinator: Arc<ConditionalDisaggCoordinator>,
    #[allow(dead_code)]
    worker_hook: Arc<dyn CdWorkerHook>,
}

impl std::fmt::Debug for PrefillDisaggLeader {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PrefillDisaggLeader").finish()
    }
}

impl PrefillDisaggLeader {
    pub fn from_parts(
        inner: Arc<dyn InnerLeaderShim>,
        coordinator: Arc<ConditionalDisaggCoordinator>,
        worker_hook: Arc<dyn CdWorkerHook>,
    ) -> Arc<Self> {
        Arc::new(Self {
            inner,
            coordinator,
            worker_hook,
        })
    }
}

impl ConnectorLeaderApi for PrefillDisaggLeader {
    fn create_slot(&self, request: Request) -> Result<()> {
        let request_id = request.request_id.clone();
        let num_tokens = request.tokens.len();
        let has_kv_transfer = request
            .metadata
            .as_ref()
            .and_then(|m| m.kv_transfer_params.as_ref())
            .is_some();
        crate::audit!(
            "create_slot",
            role = "prefill",
            request_id = %request_id,
            num_tokens,
            has_kv_transfer
        );
        self.inner.create_slot(request)
    }

    fn has_slot(&self, request_id: &str) -> bool {
        self.inner.has_slot(request_id)
    }

    fn extend_slot_tokens(&self, request_id: &str, tokens: Vec<u32>) -> Result<()> {
        self.inner.extend_slot_tokens(request_id, tokens)
    }

    #[tracing::instrument(level = "info", skip(self))]
    fn get_num_new_matched_tokens(
        &self,
        request_id: &str,
        num_computed_tokens: usize,
    ) -> Result<(Option<usize>, bool)> {
        crate::audit!(
            "gnmt_entry",
            role = "prefill",
            request_id,
            num_computed_tokens
        );
        // Detect CD-bound requests via transfer_params on the slot.
        let params = self
            .inner
            .slot_transfer_params(request_id)
            .with_context(|| format!("read transfer_params for {}", request_id))?;
        tracing::info!(
            has_transfer_params = params.is_some(),
            "prefill_gnmt: transfer_params lookup"
        );
        crate::audit!(
            "transfer_params_lookup",
            role = "prefill",
            request_id,
            has_transfer_params = params.is_some()
        );
        let Some(remote) = params.as_ref().and_then(remote_prefill_marker) else {
            // Non-CD request: passthrough to inner.
            tracing::info!("prefill_gnmt: non-CD request — passthrough to inner");
            crate::audit!(
                "gnmt_passthrough_non_cd",
                role = "prefill",
                request_id
            );
            let r = self
                .inner
                .get_num_new_matched_tokens(request_id, num_computed_tokens);
            audit_gnmt_exit("prefill", request_id, &r);
            return r;
        };

        tracing::info!(
            num_sequence_hashes = remote.sequence_hashes.len(),
            num_computed_tokens = remote.num_computed_tokens,
            initiator = %remote.initiator_instance_id,
            session_id = %remote.session_id,
            "prefill_gnmt: CD-bound — ensure_started"
        );
        crate::audit!(
            "cd_bound_ensure_started",
            role = "prefill",
            request_id,
            num_sequence_hashes = remote.sequence_hashes.len(),
            decode_num_computed_tokens = remote.num_computed_tokens,
            initiator = %remote.initiator_instance_id,
            session_id = %remote.session_id
        );
        // Idempotent: first call installs state and spawns
        // attach/diff/pull; subsequent calls just return the
        // cached external-token count. Pass a synchronous payload-
        // install closure so the slot's RAII payload lands BEFORE
        // run_setup is spawned (race-free) and only on the first
        // call (production slot rejects double-install).
        let inner_for_install = Arc::clone(&self.inner);
        let coord_weak_for_install = Arc::downgrade(&self.coordinator);
        let install_payload = move |rid: &str| -> Result<()> {
            let payload = Box::new(PrefillCdOnboardingPayload {
                request_id: rid.to_string(),
                coordinator: coord_weak_for_install,
            });
            inner_for_install.install_cd_onboarding_payload(rid, payload)
        };
        let n = self.coordinator.ensure_started(
            request_id,
            remote,
            num_computed_tokens,
            install_payload,
        )?;

        // On-policy invariant (mirrors `Slot::finalize_match_check` in
        // `connector/leader/slot.rs`, which encodes the same rule for
        // local matches): `async_load = true` requires
        // `matched_tokens > 0`. vLLM's scheduler asserts this at
        // `vllm/v1/core/sched/scheduler.py` (search for
        // `num_external_computed_tokens > 0` under `if load_kv_async:`).
        //
        // When `ensure_started` returns 0 — common when decode has no
        // local-match cache for prefill to onboard from G2 — there is
        // nothing to async-load. The CD setup (peer registration,
        // session attach) has already completed synchronously inside
        // `ensure_started`; the worker observer publishes blocks to
        // decode as they're produced during the upcoming forward
        // pass. Fall through to the inner gnmt so vLLM schedules
        // normal compute on the prompt.
        if n == 0 {
            tracing::info!(
                "prefill_gnmt: ensure_started returned 0 external tokens — \
                 passthrough to inner so vLLM forward-passes the prompt"
            );
            crate::audit!(
                "ensure_started_zero_passthrough",
                role = "prefill",
                request_id
            );
            let r = self
                .inner
                .get_num_new_matched_tokens(request_id, num_computed_tokens);
            audit_gnmt_exit("prefill", request_id, &r);
            return r;
        }

        tracing::info!(
            external_tokens = n,
            "prefill_gnmt: ensure_started returned (Some(N>0), true) — async onboard"
        );
        crate::audit!(
            "ensure_started_async_onboard",
            role = "prefill",
            request_id,
            external_tokens = n
        );
        // The slot RAII payload was installed inside ensure_started
        // (race-free, idempotent). Just emit the audit + return.
        crate::audit!(
            "prefill_cd_payload_installed",
            role = "prefill",
            request_id,
            external_tokens = n
        );
        let r: Result<(Option<usize>, bool)> = Ok((Some(n), true));
        audit_gnmt_exit("prefill", request_id, &r);
        r
    }

    fn update_state_after_alloc(
        &self,
        request_id: &str,
        block_ids: Vec<BlockId>,
        num_external_tokens: usize,
    ) -> Result<()> {
        // The coordinator owns USAA only when there's CD-bound work to
        // onboard (prefill-role tracked AND `bits.num_external_tokens > 0`).
        // The zero-hash case (decode forwarded no local-match hashes) leaves
        // the coordinator tracking the request for observer-side output flow
        // but with no onboarding work; in that case the inner connector owns
        // USAA, so an inner-cache-hit's `num_external_tokens` doesn't collide
        // with the coordinator's expected zero.
        let coord_owns_usaa = self.coordinator.prefill_owns_usaa(request_id);
        crate::audit!(
            "usaa_entry",
            role = "prefill",
            request_id,
            num_block_ids = block_ids.len(),
            num_external_tokens,
            coord_owns_usaa
        );
        if !coord_owns_usaa {
            // Inner owns USAA. Either non-CD, or CD-bound zero-hash
            // passthrough where the inner connector may have its own
            // local cache hit driving USAA.
            return self
                .inner
                .update_state_after_alloc(request_id, block_ids, num_external_tokens);
        }

        let r = self
            .coordinator
            .on_usaa(request_id, &block_ids, num_external_tokens);
        match &r {
            Ok(()) => crate::audit!("usaa_exit", role = "prefill", request_id, ok = true),
            Err(err) => crate::audit!(
                "usaa_error",
                role = "prefill",
                request_id,
                error = %err
            ),
        }
        r
    }

    fn build_connector_meta(&self, output: SchedulerOutput) -> Result<KvConnectorMetadata> {
        crate::connector::leader::audit::audit_build_meta("prefill", &output);
        let meta = self.inner.build_connector_meta(output)?;
        // Per-request observer install. The coordinator only
        // acts on request_ids it tracks; passing the whole meta
        // is fine.
        // TODO(A.5): once meta carries scheduled CD request_ids
        // explicitly, iterate them rather than passing meta as
        // a whole. For golden-path scaffolding we pass the meta
        // and the coordinator currently no-ops.
        if let Err(err) = self.coordinator.observe_forward("", &meta) {
            tracing::warn!(error = %err, "observe_forward failed");
            crate::audit!(
                "observe_forward_error",
                role = "prefill",
                error = %err
            );
        }
        Ok(meta)
    }

    fn update_connector_output(
        &self,
        finished_sending: HashSet<String>,
        finished_recving: HashSet<String>,
    ) -> Result<()> {
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
        self.inner
            .update_connector_output(finished_sending, finished_recving)
    }

    fn request_finished(&self, request_id: &str) -> FinishedStatus {
        crate::audit!("request_finished_entry", role = "prefill", request_id);
        let status = self.inner.request_finished(request_id);
        self.coordinator.on_request_finished(request_id);
        crate::audit!(
            "request_finished_exit",
            role = "prefill",
            request_id,
            status = ?status
        );
        status
    }
}

fn audit_gnmt_exit(
    role: &'static str,
    request_id: &str,
    result: &Result<(Option<usize>, bool)>,
) {
    match result {
        Ok((count, async_load)) => crate::audit!(
            "gnmt_exit",
            role,
            request_id,
            count = ?count,
            async_load
        ),
        Err(err) => crate::audit!(
            "gnmt_error",
            role,
            request_id,
            error = %err
        ),
    }
}

/// Extract the remote-prefill marker from parsed transfer
/// params, if present. `None` for non-CD requests or when only
/// a remote-decode marker is present.
fn remote_prefill_marker(
    params: &TransferParams,
) -> Option<&kvbm_disagg_protocol::RemotePrefillParams> {
    params.remote_prefill.as_ref()
}
