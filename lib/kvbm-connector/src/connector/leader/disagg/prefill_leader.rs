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
use super::prefill_coordinator::PrefillCoordinator;
use super::transport::{CdWorkerHook, InnerLeaderShim};

pub struct PrefillDisaggLeader {
    inner: Arc<dyn InnerLeaderShim>,
    coordinator: Arc<dyn PrefillCoordinator>,
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
        coordinator: Arc<dyn PrefillCoordinator>,
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
        // cached external-token count.
        let n = self.coordinator.ensure_started(request_id, remote)?;

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
        crate::audit!(
            "usaa_entry",
            role = "prefill",
            request_id,
            num_block_ids = block_ids.len(),
            num_external_tokens
        );
        // Always call inner first.
        self.inner
            .update_state_after_alloc(request_id, block_ids.clone(), num_external_tokens)?;

        // For CD-bound requests, hand the deltas to the
        // coordinator. Non-CD requests have no coordinator
        // state and `on_usaa` is a no-op.
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
