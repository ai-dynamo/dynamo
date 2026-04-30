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
        // Detect CD-bound requests via transfer_params on the slot.
        let params = self
            .inner
            .slot_transfer_params(request_id)
            .with_context(|| format!("read transfer_params for {}", request_id))?;
        tracing::info!(
            has_transfer_params = params.is_some(),
            "prefill_gnmt: transfer_params lookup"
        );
        let Some(remote) = params.as_ref().and_then(remote_prefill_marker) else {
            // Non-CD request: passthrough to inner.
            tracing::info!("prefill_gnmt: non-CD request — passthrough to inner");
            return self
                .inner
                .get_num_new_matched_tokens(request_id, num_computed_tokens);
        };

        tracing::info!(
            num_sequence_hashes = remote.sequence_hashes.len(),
            num_computed_tokens = remote.num_computed_tokens,
            initiator = %remote.initiator_instance_id,
            session_id = %remote.session_id,
            "prefill_gnmt: CD-bound — ensure_started"
        );
        // Idempotent: first call installs state and spawns
        // attach/diff/pull; subsequent calls just return the
        // cached external-token count.
        let n = self.coordinator.ensure_started(request_id, remote)?;
        tracing::info!(
            external_tokens = n,
            "prefill_gnmt: ensure_started returned (Some(N), true)"
        );
        Ok((Some(n), true))
    }

    fn update_state_after_alloc(
        &self,
        request_id: &str,
        block_ids: Vec<BlockId>,
        num_external_tokens: usize,
    ) -> Result<()> {
        // Always call inner first.
        self.inner
            .update_state_after_alloc(request_id, block_ids.clone(), num_external_tokens)?;

        // For CD-bound requests, hand the deltas to the
        // coordinator. Non-CD requests have no coordinator
        // state and `on_usaa` is a no-op.
        self.coordinator
            .on_usaa(request_id, &block_ids, num_external_tokens)
    }

    fn build_connector_meta(&self, output: SchedulerOutput) -> Result<KvConnectorMetadata> {
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
        }
        Ok(meta)
    }

    fn update_connector_output(
        &self,
        finished_sending: HashSet<String>,
        finished_recving: HashSet<String>,
    ) -> Result<()> {
        self.inner
            .update_connector_output(finished_sending, finished_recving)
    }

    fn request_finished(&self, request_id: &str) -> FinishedStatus {
        let status = self.inner.request_finished(request_id);
        self.coordinator.on_request_finished(request_id);
        status
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
