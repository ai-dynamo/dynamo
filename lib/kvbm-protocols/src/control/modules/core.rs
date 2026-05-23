// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! `core` module protocol: instance describe.

use std::time::SystemTime;

use serde::{Deserialize, Serialize};

use super::super::ModuleId;
use super::super::layout_compat::LayoutCompatPayload;

// ---------------------------------------------------------------------------
// describe_instance
// ---------------------------------------------------------------------------

/// Velo handler name for the always-on `describe_instance` operation.
///
/// In steady state the leader **pushes** an [`InstanceDescription`] to the hub
/// via `POST /v1/instances/{id}/describe`. This velo handler exists for the
/// hub's fallback pull (cold cache after hub restart, operator-triggered
/// re-fetch via `GET /describe?force=true` or `POST /control/core/describe_instance`).
pub const DESCRIBE_INSTANCE_HANDLER: &str = "kvbm.leader.control.describe_instance";

/// Empty request for [`DESCRIBE_INSTANCE_HANDLER`]. The target instance is
/// implied by the velo addressing.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct DescribeInstanceRequest {}

/// Structured topology snapshot of one leader instance.
///
/// Built by `InstanceLeader::describe()` on the engine side; consumed by the
/// hub's `describe_cache` and the web UI. Authoritative ownership lives at the
/// leader — the leader pushes; the hub stores.
///
/// Fields whose data isn't yet available at the moment of describe return as
/// `None`/empty (`block_size`, `parallelism`, `workers`, `tier_capacity`) — a
/// pre-stamping snapshot is still a valid `Ok(InstanceDescription)`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InstanceDescription {
    // ---------- Identity ----------
    /// Stringified `InstanceId` of this leader.
    pub instance_id: String,
    /// Worker IDs attached to this leader (snapshot of `workers[*].worker_id`).
    pub worker_ids: Vec<u64>,
    /// Stringified `InstanceId` of the hub this leader believes it registered
    /// with. `None` if not yet known (pre-register pushes shouldn't happen but
    /// this stays Optional defensively).
    pub hub_instance_id: Option<String>,

    // ---------- Top-level topology ----------
    /// Token block size — `LayoutConfig::page_size` common across layouts.
    /// `None` if heterogeneous (today: never) or pre-stamping.
    pub block_size: Option<usize>,
    /// Aggregate tp/pp/shard for the leader. Derived from per-worker
    /// [`ParallelismDescription`]; present when all workers agree.
    pub parallelism: Option<ParallelismDescription>,
    /// Per-tier capacity summary, summed across workers.
    pub tier_capacity: Vec<TierCapacity>,

    // ---------- Per-worker detail ----------
    pub workers: Vec<WorkerInfo>,

    // ---------- Capabilities + config ----------
    /// Modules enabled on the leader's control plane.
    pub modules: Vec<ModuleId>,
    /// Disaggregation role if this leader is part of a P/D split. `None` for
    /// a standalone leader.
    pub role: Option<DisaggRole>,
    /// Opaque JSON of the leader's `KvbmConfig` (set by the connector via
    /// `set_config_blob`). `None` until the connector has injected it.
    pub config: Option<serde_json::Value>,

    // ---------- Process ----------
    pub host: HostInfo,
    pub started_at: SystemTime,

    // ---------- Layout compatibility ----------
    /// Operative layout-compat payload — same shape as the one carried in
    /// `Feature::P2P` at register time. The hub's `P2pManager` stores the
    /// register-time payload as the baseline; describe-push validates
    /// `Some(payload)` against that baseline via `check_layout_compat`.
    ///
    /// `None` is the legacy / pre-stamping snapshot path and bypasses the
    /// check — matches the existing `block_size` / `parallelism` optionality
    /// contract.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub layout_compat: Option<LayoutCompatPayload>,
}

/// Per-worker detail in [`InstanceDescription`].
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorkerInfo {
    pub worker_id: u64,
    /// NIXL agent name registered on this worker. Pairs with `worker_id`
    /// inside [`WorkerAddress`] on the engine side — both are pulled from
    /// the same source. Useful for cross-referencing NIXL diagnostics.
    pub nixl_agent_name: String,
    /// This worker's place in the leader's parallelism grid.
    ///
    /// `None` when the worker's `SerializedLayout` carries no
    /// [`ParallelismDescriptor`] — i.e. the leader has no
    /// `ParallelismTemplate` configured (single-rank leader) or the
    /// stamping pass hasn't completed yet. **Never synthesise a
    /// `Some(1x1)`** for an unstamped worker — that lies about
    /// topology for a multi-worker TP leader pre-stamping. The
    /// aggregate [`InstanceDescription::parallelism`] mirrors this
    /// rule: it stays `None` if any worker reports `None`.
    pub parallelism: Option<ParallelismDescription>,
    /// Every layout this worker exposes — one entry per tier.
    pub layouts: Vec<LayoutDescription>,
}

/// One tier's layout on one worker.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LayoutDescription {
    pub tier: TierKind,
    /// Full layout config — wire mirror of
    /// `kvbm_physical::layout::config::LayoutConfig`.
    pub config: LayoutConfigDescription,
    /// Where this layout's memory lives — wire mirror of
    /// `dynamo_memory::StorageKind`. Surfaces GPU device index for `Device`,
    /// the disk handle for `Disk`, etc.
    pub location: StorageKindDescription,
    /// `LayoutTypeDetails` discriminant — `"fully_contiguous"` or
    /// `"layer_separate"`.
    pub layout_type: String,
    /// `KvBlockLayout` discriminant — e.g. `"universal_tp"`, `"universal_pp"`,
    /// `"operational_nhd"`. Per-tier.
    pub block_layout: String,
    /// Convenience: `num_layers * outer_dim * page_size * inner_dim *
    /// dtype_width_bytes`.
    pub bytes_per_block: usize,
    /// Convenience: `bytes_per_block * num_blocks`.
    pub total_bytes: usize,
}

/// Wire mirror of `dynamo_memory::StorageKind`. Same variants, snake_case
/// serde so JSON keys match the rest of the control plane.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum StorageKindDescription {
    /// System memory (malloc).
    System,
    /// CUDA pinned host memory.
    Pinned,
    /// CUDA device memory. Carries the CUDA device index.
    Device(u32),
    /// Disk-backed memory (mmap). Carries the implementation-defined handle
    /// (currently a `u64` — file fd or backing identifier, depending on the
    /// allocator).
    Disk(u64),
}

/// Wire mirror of `kvbm_physical::layout::config::LayoutConfig`.
///
/// Same field set, no cargo dep on kvbm-physical (avoids dragging NIXL/CUDA
/// into the protocol crate). Leader-side `describe()` does the 1:1 copy. The
/// engine's round-trip test catches drift if kvbm-physical adds a field.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct LayoutConfigDescription {
    pub num_blocks: usize,
    pub num_layers: usize,
    pub outer_dim: usize,
    pub page_size: usize,
    pub inner_dim: usize,
    pub alignment: usize,
    pub dtype_width_bytes: usize,
    pub num_heads: Option<usize>,
}

/// Wire mirror of `kvbm_physical::manager::metadata::ParallelismDescriptor`.
///
/// `KvDim` discriminants flatten to strings; `Range<usize>` flattens to
/// [`LayerRange`] for explicit serde compatibility.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ParallelismDescription {
    pub tp_size: usize,
    pub pp_size: usize,
    pub rank: usize,
    /// `KvDim` variant name, snake_case (e.g. `"head_count"`).
    pub shard_axis: String,
    /// Global (pre-shard) extents per axis. `Vec` of `(axis, size)` rather
    /// than a map so order is deterministic.
    pub global_extents: Vec<(String, usize)>,
    /// Layer range this worker owns. For `pp_size == 1` this is `0..num_layers`.
    pub layer_ownership: LayerRange,
}

/// Wire-serialisable half-open layer range, replacing `Range<usize>` (which
/// has no `Eq` impl and inconsistent serde shape).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct LayerRange {
    pub start: usize,
    pub end: usize,
}

impl LayerRange {
    pub fn len(&self) -> usize {
        self.end.saturating_sub(self.start)
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

/// Per-tier capacity summary, summed across workers.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TierCapacity {
    pub tier: TierKind,
    /// Summed across workers.
    pub num_blocks: usize,
    /// Per-worker bytes-per-block (identical across workers today).
    pub bytes_per_block: usize,
    /// `num_blocks * bytes_per_block`, summed across workers.
    pub total_bytes: u64,
}

/// Tier discriminant — wire alias of `kvbm_common::LogicalLayoutHandle` with
/// snake_case serde matching the rest of the control plane (cf.
/// [`super::dev::Tier`]).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TierKind {
    G1,
    G2,
    G3,
    G4,
}

/// Disaggregation role wire enum (mirrors `kvbm_config::DisaggregationRole`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DisaggRole {
    Prefill,
    Decode,
}

/// Process-level identity.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct HostInfo {
    pub hostname: String,
    pub pid: u32,
}
