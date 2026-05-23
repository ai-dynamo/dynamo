// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Cross-leader block-layout compatibility check (engine-side).
//!
//! Decides whether a remote leader's per-worker metadata is compatible
//! with this leader's local metadata under the chosen
//! [`BlockLayoutMode`]. See `kvbm_common::block_layout_mode` for the
//! semantic write-up.
//!
//! The predicate itself lives in `kvbm_protocols::control::layout_compat`
//! so the hub-side startup gate and this engine-side per-import gate
//! cannot drift. This module is the engine-specific adapter:
//!
//! - [`build_layout_compat_payload`] turns a per-worker
//!   [`SerializedLayout`] into the wire-stable
//!   [`LayoutCompatPayload`].
//! - [`check_import_compat`] walks the routed local↔remote worker pairs
//!   (operational) or rank 0 (universal) and delegates each comparison
//!   to [`check_layout_compat`].
//!
//! Operational mode additionally walks `route_local_to_remote` so a
//! TP-fanned import surfaces a precise per-pair error. Universal mode
//! retains the `require_all_labeled` check on both sides per the Codex
//! review finding.

use anyhow::{Context as _, Result, anyhow, bail};
use kvbm_common::{BlockLayoutMode, KvBlockLayout};
use kvbm_physical::layout::LayoutTypeDetails;
use kvbm_physical::manager::{
    SerializedLayout, canonical_shape_from_worker, select_transfer_canonical_layout,
};
use kvbm_protocols::control::layout_compat::{LayoutCompatPayload, check_layout_compat};

use super::describe_map::to_layout_config_description;
use super::parallelism::ParallelismTemplate;
use super::state::route_local_to_remote;

/// Build the wire-stable [`LayoutCompatPayload`] for one worker.
///
/// The hub uses this at register time (built from the leader's local
/// `cached_worker_metadata[0]`); the engine uses it per-pair during
/// import-time enforcement. Both callers run the same
/// [`check_layout_compat`] predicate so the gates cannot drift.
///
/// # Errors
///
/// - Worker carries no `ParallelismDescriptor` (legacy / unstamped) —
///   universal mode requires the canonical extents.
/// - Worker exports no layouts.
/// - Worker's [`KvBlockLayout`] is `Unknown` and `mode` is
///   [`BlockLayoutMode::Universal`].
/// - `inner_dim` is not divisible by `num_heads`.
pub fn build_layout_compat_payload(
    mode: BlockLayoutMode,
    worker: &SerializedLayout,
) -> Result<LayoutCompatPayload> {
    build_layout_compat_payload_with_template(mode, worker, None)
}

/// Build the wire-stable [`LayoutCompatPayload`] using a leader-side
/// [`ParallelismTemplate`] as the source of canonical extents when the
/// per-worker `SerializedLayout` has not yet been stamped.
///
/// The connector's hub-registration path holds the raw worker metadata
/// (cached at `worker.initialize()` time, before
/// [`stamp_parallelism_descriptors`](super::parallelism::stamp_parallelism_descriptors)
/// runs) and a freshly-built `ParallelismTemplate`. Passing the template
/// lets universal mode derive the canonical aggregate even though the
/// per-worker bytes carry `parallelism = None`.
///
/// The engine-side defence-in-depth at `connect_remote` keeps the
/// no-template entrypoint via [`build_layout_compat_payload`] — that
/// path operates on already-stamped wire metadata.
pub fn build_layout_compat_payload_with_template(
    mode: BlockLayoutMode,
    worker: &SerializedLayout,
    template: Option<&ParallelismTemplate>,
) -> Result<LayoutCompatPayload> {
    let unpacked = worker.unpack()?;
    // c3: pick the transfer-canonical layout (G2 if present, else first).
    // In Universal mode the connector pins G2 to KvBlockLayout::Universal,
    // so peers must see G2's layout — not G1's operational layout — on the
    // wire.
    let canonical_layout =
        select_transfer_canonical_layout(&unpacked.layouts).ok_or_else(|| {
            anyhow!(
                "build_layout_compat_payload: worker {} exported no layouts",
                unpacked.worker_address.worker_id,
            )
        })?;

    let kv_layout = kv_block_layout_of(&canonical_layout.layout.layout_type_details);
    if mode == BlockLayoutMode::Universal && matches!(kv_layout, KvBlockLayout::Unknown) {
        bail!(
            "build_layout_compat_payload: worker {} has KvBlockLayout::Unknown; \
             universal mode requires every axis to be labeled",
            unpacked.worker_address.worker_id,
        );
    }

    // Canonical aggregate sources, in priority order:
    //   1. ParallelismTemplate when provided (the connector's pre-build
    //      register path — worker metadata is raw / unstamped).
    //   2. SerializedLayout's stamped ParallelismDescriptor (engine's
    //      per-import defence-in-depth — workers are wire-mirrored and
    //      already stamped).
    //   3. None — fall through; universal mode then relies on the
    //      hub-side predicate to reject if neither side could derive a
    //      canonical, while operational mode degrades gracefully to
    //      per-worker comparison.
    let (canonical, tp_size, pp_size) = if let Some(t) = template {
        (t.canonical_block_shape(), t.tp_size, t.pp_size)
    } else {
        match unpacked.parallelism.as_ref() {
            Some(p) => (
                Some(canonical_shape_from_worker(worker)?),
                p.tp_size,
                p.pp_size,
            ),
            None if mode == BlockLayoutMode::Universal => {
                bail!(
                    "build_layout_compat_payload: worker {} carries no \
                     ParallelismDescriptor and no ParallelismTemplate; \
                     universal mode requires global_extents",
                    unpacked.worker_address.worker_id,
                );
            }
            None => (None, 1, 1),
        }
    };

    let per_worker_config = to_layout_config_description(&canonical_layout.layout.layout_config);

    Ok(LayoutCompatPayload {
        mode,
        canonical,
        per_worker_layout: kv_layout,
        per_worker_config,
        tp_size,
        pp_size,
    })
}

/// Reject a remote leader's metadata when its layout is incompatible
/// with this leader under the chosen [`BlockLayoutMode`].
///
/// `local_template` lets the local side derive canonical extents from
/// the leader's `ParallelismTemplate` when its cached worker metadata
/// is unstamped (which is the production path — workers emit
/// `parallelism = None` and stamping happens lazily inside the leader).
/// Remote workers go through the wire's stamped `ParallelismDescriptor`.
/// Pass `None` to fall back to the metadata-only path (used by callers
/// that don't yet have a template, e.g. the operational defence-in-
/// depth gate where canonical equality is best-effort).
///
/// Operational walks `route_local_to_remote` pairs so the error names
/// the exact `(local_rank, remote_rank)` whose shapes diverged.
/// Universal compares rank-0 payloads after ensuring every worker on
/// both sides is labeled.
///
/// Empty inputs are a no-op (Ok).
pub fn check_import_compat(
    mode: BlockLayoutMode,
    local: &[SerializedLayout],
    remote: &[SerializedLayout],
    local_template: Option<&ParallelismTemplate>,
) -> Result<()> {
    if local.is_empty() || remote.is_empty() {
        return Ok(());
    }
    match mode {
        BlockLayoutMode::Operational => check_operational_pairs(local, remote, local_template),
        BlockLayoutMode::Universal => check_universal(local, remote, local_template),
    }
}

fn check_operational_pairs(
    local: &[SerializedLayout],
    remote: &[SerializedLayout],
    local_template: Option<&ParallelismTemplate>,
) -> Result<()> {
    let local_n = local.len();
    let remote_n = remote.len();
    for (local_rank, local_worker) in local.iter().enumerate() {
        let local_payload = build_layout_compat_payload_with_template(
            BlockLayoutMode::Operational,
            local_worker,
            local_template,
        )
        .with_context(|| format!("operational compat: local rank {local_rank}"))?;
        for remote_rank in route_local_to_remote(local_rank, local_n, remote_n) {
            let remote_payload =
                build_layout_compat_payload(BlockLayoutMode::Operational, &remote[remote_rank])
                    .with_context(|| format!("operational compat: remote rank {remote_rank}"))?;
            check_layout_compat(&local_payload, &remote_payload).map_err(|e| {
                anyhow!(
                    "operational compat: local rank {local_rank} \
                     incompatible with remote rank {remote_rank}: {e}"
                )
            })?;
        }
    }
    Ok(())
}

fn check_universal(
    local: &[SerializedLayout],
    remote: &[SerializedLayout],
    local_template: Option<&ParallelismTemplate>,
) -> Result<()> {
    require_all_labeled(local, "local")?;
    require_all_labeled(remote, "remote")?;
    let local_payload = build_layout_compat_payload_with_template(
        BlockLayoutMode::Universal,
        &local[0],
        local_template,
    )
    .map_err(|e| anyhow!("layout_compat (local worker 0): {e}"))?;
    let remote_payload = build_layout_compat_payload(BlockLayoutMode::Universal, &remote[0])
        .map_err(|e| anyhow!("layout_compat (remote worker 0): {e}"))?;
    check_layout_compat(&local_payload, &remote_payload)
}

fn kv_block_layout_of(details: &LayoutTypeDetails) -> KvBlockLayout {
    match details {
        LayoutTypeDetails::FullyContiguous(d) => d.kv_block_layout,
        LayoutTypeDetails::LayerSeparate(d) => d.kv_block_layout,
    }
}

fn require_all_labeled(workers: &[SerializedLayout], side: &str) -> Result<()> {
    for (rank, w) in workers.iter().enumerate() {
        let unpacked = w.unpack()?;
        if unpacked.layouts.is_empty() {
            bail!("layout_compat: {side} rank {rank} exported no layouts");
        }
        // c3: walk **every** tier, not just the transfer-canonical one.
        // The wire-reported layout (G2) is always `Universal` by
        // construction in Universal mode (mode-dominant selection), so
        // an Unknown G1 would slip through if we only checked G2 — and
        // the fused permute kernel needs G1's labeled axis order
        // (NHD vs HND) to permute correctly.
        for layout in &unpacked.layouts {
            let kv = kv_block_layout_of(&layout.layout.layout_type_details);
            if matches!(kv, KvBlockLayout::Unknown) {
                bail!(
                    "universal compat: {side} rank {rank} tier {:?} has \
                     KvBlockLayout::Unknown; universal mode requires every axis \
                     of every tier to be labeled at registration",
                    layout.logical_type
                );
            }
        }
    }
    Ok(())
}
