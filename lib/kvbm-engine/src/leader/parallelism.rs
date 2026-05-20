// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Leader-side stamping of [`ParallelismDescriptor`] onto per-worker
//! [`SerializedLayout`] metadata.
//!
//! Workers don't intrinsically know the leader's tp/pp sizes — block-ID
//! and rank space is leader-scoped. So before forwarding a worker's
//! exported metadata to a peer leader (via `kvbm.leader.export_metadata`),
//! the leader stamps a [`ParallelismDescriptor`] onto each per-worker
//! payload describing where that worker sits in the leader's
//! parallelism grid. The peer's cross-parallelism dispatcher reads this
//! to plan transfers without inferring tp_size from
//! `Vec<SerializedLayout>.len()` or guessing the shard axis.
//!
//! AB-1a step 2: this module is the computation. Wiring into the
//! `export_metadata_callback` is a separate step.

use anyhow::{Result, bail};
use kvbm_common::{KvDim, LogicalLayoutHandle};
use kvbm_config::ParallelismMode;
use kvbm_physical::layout::LayoutConfig;
use kvbm_physical::manager::{ParallelismDescriptor, SerializedLayout};

/// Per-leader parallelism template — the knobs the leader applies when
/// stamping per-worker descriptors. PP is reserved (must be 1 for now).
#[derive(Debug, Clone)]
pub struct ParallelismTemplate {
    /// Total tensor-parallel size for this leader.
    pub tp_size: usize,
    /// Reserved — must be 1.
    pub pp_size: usize,
    /// Parallelism mode the leader configured its workers with.
    /// Used to flag whether sharding is actually in effect; the
    /// descriptor's `shard_axis` is meaningful only under
    /// [`ParallelismMode::TensorParallel`].
    pub parallelism_mode: ParallelismMode,
    /// Axis along which workers shard. Typically [`KvDim::HeadCount`].
    pub shard_axis: KvDim,
    /// Global (pre-shard) extents per axis. Empty is legal — the
    /// peer's compatibility gate will skip per-axis extent checks
    /// when extents are absent. Populating this enables strict
    /// AB-1b cross-leader gate checks.
    pub global_extents: Vec<(KvDim, usize)>,
    /// Total number of layers. Today every worker owns `0..num_layers`
    /// (PP=1). The first PP PR will replace this with per-rank ranges.
    pub num_layers: usize,
    /// dtype width in bytes (from `LayoutConfig::dtype_width_bytes`).
    /// Carried on the template so the Universal block-layout
    /// compatibility check can compare it against remote canonicals
    /// without re-reading per-worker `LayoutConfig`.
    pub dtype_width_bytes: usize,
}

impl ParallelismTemplate {
    /// Build a template from the leader's per-worker [`LayoutConfig`], the
    /// configured [`ParallelismMode`], and the worker count.
    ///
    /// Today PP is unsupported (`pp_size = 1`). The shard axis is
    /// [`KvDim::HeadCount`] under [`ParallelismMode::TensorParallel`]; under
    /// [`ParallelismMode::ReplicatedData`] no axis is actually sharded but
    /// the field is set to [`KvDim::HeadCount`] by convention — the receiver
    /// disambiguates by comparing per-worker layout extents to
    /// `global_extents`.
    pub fn from_layout_config(
        layout: &LayoutConfig,
        mode: ParallelismMode,
        num_workers: usize,
    ) -> Result<Self> {
        if num_workers == 0 {
            bail!("from_layout_config: num_workers must be > 0");
        }
        let per_worker_heads = layout.num_heads.ok_or_else(|| {
            anyhow::anyhow!(
                "from_layout_config: LayoutConfig.num_heads must be set to derive HeadCount extent"
            )
        })?;
        if per_worker_heads == 0 {
            bail!("from_layout_config: num_heads must be > 0");
        }
        // `inner_dim` is the trailing dim of the block tensor and equals
        // `num_heads * head_size` (see kvbm-physical/src/layout/mod.rs
        // `resolve_head_dims`). Per-head channel count is the ratio. Under
        // TensorParallel both `inner_dim` and `num_heads` shard by the same
        // factor, so the ratio is invariant — `head_size` is a global
        // constant either way.
        if !layout.inner_dim.is_multiple_of(per_worker_heads) {
            bail!(
                "from_layout_config: inner_dim ({}) is not divisible by num_heads ({}) \
                 — cannot derive HeadSize extent",
                layout.inner_dim,
                per_worker_heads,
            );
        }
        let head_size = layout.inner_dim / per_worker_heads;
        let global_heads = match mode {
            ParallelismMode::TensorParallel => per_worker_heads * num_workers,
            ParallelismMode::ReplicatedData => per_worker_heads,
        };
        Ok(Self {
            tp_size: num_workers,
            pp_size: 1,
            parallelism_mode: mode,
            shard_axis: KvDim::HeadCount,
            // global_extents covers axes that are model-global (same
            // across leaders running the same model). Block is
            // deliberately omitted — block-id space is per-leader and
            // two leaders can legitimately have different block counts.
            global_extents: vec![
                (KvDim::Layer, layout.num_layers),
                (KvDim::Outer, layout.outer_dim),
                (KvDim::Page, layout.page_size),
                (KvDim::HeadCount, global_heads),
                (KvDim::HeadSize, head_size),
            ],
            num_layers: layout.num_layers,
            dtype_width_bytes: layout.dtype_width_bytes,
        })
    }

    /// Build a [`CanonicalBlockShape`](kvbm_common::CanonicalBlockShape)
    /// from this template.
    ///
    /// Returns `None` if `global_extents` is missing one of the required
    /// canonical axes (Layer / Outer / Page / HeadCount / HeadSize). This
    /// is the local side of the Universal-mode block-layout
    /// compatibility check in [`SpmdParallelWorkers::connect_remote`].
    pub fn canonical_block_shape(&self) -> Option<kvbm_common::CanonicalBlockShape> {
        let extent = |axis: KvDim| -> Option<usize> {
            self.global_extents
                .iter()
                .find_map(|(d, n)| if *d == axis { Some(*n) } else { None })
        };
        Some(kvbm_common::CanonicalBlockShape {
            num_layers_total: extent(KvDim::Layer)?,
            outer_dim: extent(KvDim::Outer)?,
            page_size: extent(KvDim::Page)?,
            num_heads_total: extent(KvDim::HeadCount)?,
            head_dim: extent(KvDim::HeadSize)?,
            dtype_width_bytes: self.dtype_width_bytes,
        })
    }

    /// Build the [`ParallelismDescriptor`] for a specific rank.
    pub fn descriptor_for(&self, rank: usize) -> Result<ParallelismDescriptor> {
        if self.pp_size != 1 {
            bail!(
                "ParallelismTemplate::descriptor_for: pp_size={} not supported yet (PP is a non-goal)",
                self.pp_size
            );
        }
        let total = self.tp_size * self.pp_size;
        if rank >= total {
            bail!(
                "ParallelismTemplate::descriptor_for: rank {} out of range (tp_size={}, pp_size={})",
                rank,
                self.tp_size,
                self.pp_size
            );
        }
        Ok(ParallelismDescriptor {
            tp_size: self.tp_size,
            pp_size: self.pp_size,
            rank,
            shard_axis: self.shard_axis,
            global_extents: self.global_extents.clone(),
            layer_ownership: 0..self.num_layers,
        })
    }
}

/// Compatibility error variants surfaced by [`validate_remote_metadata`].
///
/// Each variant pinpoints exactly which gate rejected the remote
/// metadata so callers (and operators reading logs) can diagnose the
/// configuration mismatch without re-deriving the check.
#[derive(Debug, thiserror::Error, PartialEq, Eq)]
pub enum CompatError {
    /// Either side reported `tp_size == 0` or `pp_size == 0`.
    #[error(
        "compatibility: zero-sized parallelism (tp_size={tp_size}, pp_size={pp_size}, side={side})"
    )]
    ZeroSize {
        side: &'static str,
        tp_size: usize,
        pp_size: usize,
    },

    /// `pp_size != 1` on either side. PP is a non-goal for AB-1..AB-6.
    #[error(
        "compatibility: pipeline-parallel unsupported (pp_size={pp_size} on {side}); only pp_size=1 is accepted"
    )]
    PipelineParallelUnsupported { side: &'static str, pp_size: usize },

    /// The two TP sizes are neither equal nor a clean multiple of one
    /// another (e.g. local TP=3 vs remote TP=2).
    #[error(
        "compatibility: coprime tensor-parallel sizes (local_tp={local_tp}, remote_tp={remote_tp}); \
         one must divide the other"
    )]
    Coprime { local_tp: usize, remote_tp: usize },

    /// The two sides shard along different axes (e.g. local on
    /// `HeadCount`, remote on `Layer`).
    #[error("compatibility: shard axis mismatch (local={local:?}, remote={remote:?})")]
    ShardAxisMismatch { local: KvDim, remote: KvDim },

    /// Two leaders disagree on a global extent. `global_extents` is
    /// the pre-shard model-global value (e.g. total HeadCount across
    /// all ranks), so it must match on every axis including the
    /// shard axis — tp_size scales the *per-worker* value, not the
    /// global one.
    #[error("compatibility: extent mismatch on axis {axis:?} (local={local}, remote={remote})")]
    ExtentMismatch {
        axis: KvDim,
        local: usize,
        remote: usize,
    },

    /// A global axis is reported by one side but not the other.
    /// `usize::MAX` is the sentinel value used for the missing side.
    #[error("compatibility: axis {axis:?} missing on {missing_on} side (present={present})")]
    MissingExtent {
        axis: KvDim,
        missing_on: &'static str,
        present: usize,
    },

    /// Remote descriptors disagree among themselves about parallelism
    /// shape (different ranks reported different tp_size / shard_axis /
    /// global_extents). Indicates a buggy stamper on the peer.
    #[error("compatibility: remote descriptors are internally inconsistent: {reason}")]
    InconsistentRemote { reason: String },

    /// The peer reported `tp_size * pp_size = expected` but the vector
    /// of descriptors has a different length, or rank values do not
    /// form a complete `0..expected` set (duplicates, out-of-range, or
    /// missing ranks).
    #[error(
        "compatibility: incomplete remote rank metadata: {reason} \
         (expected {expected} ranks, got {actual})"
    )]
    IncompleteRemoteRanks {
        reason: String,
        expected: usize,
        actual: usize,
    },

    /// The required logical tier (typically G2) is absent from at
    /// least one remote worker's layouts list.
    #[error("compatibility: required logical tier {tier:?} missing from remote rank {rank}")]
    MissingLogicalTier {
        rank: usize,
        tier: LogicalLayoutHandle,
    },
}

/// Validate that a vector of remote per-worker [`ParallelismDescriptor`]s
/// (one per remote rank) is compatible with the local leader's
/// [`ParallelismTemplate`] for cross-parallelism transfers.
///
/// Each remote rank's layout list (`remote_tiers[i]`) is also checked
/// to contain `required_tier`. The lists must be parallel:
/// `remote_descriptors.len() == remote_tiers.len()`.
///
/// On success, returns `Ok(())`. On failure, returns the first gate
/// that rejected — gates are checked in this order:
///   1. zero-sized parallelism on either side
///   2. `pp_size != 1` on either side
///   3. internal consistency of remote descriptors
///   4. TP divisibility
///   5. shard axis agreement
///   6. non-shard extent agreement
///   7. required logical tier presence on every remote rank
pub fn validate_remote_metadata(
    local: &ParallelismTemplate,
    remote_descriptors: &[ParallelismDescriptor],
    remote_tiers: &[&[LogicalLayoutHandle]],
    required_tier: LogicalLayoutHandle,
) -> std::result::Result<(), CompatError> {
    if remote_descriptors.len() != remote_tiers.len() {
        return Err(CompatError::InconsistentRemote {
            reason: format!(
                "remote_descriptors.len() ({}) != remote_tiers.len() ({})",
                remote_descriptors.len(),
                remote_tiers.len()
            ),
        });
    }
    if remote_descriptors.is_empty() {
        return Err(CompatError::InconsistentRemote {
            reason: "remote_descriptors is empty".to_string(),
        });
    }

    // Gate 1: zero-sized parallelism
    if local.tp_size == 0 || local.pp_size == 0 {
        return Err(CompatError::ZeroSize {
            side: "local",
            tp_size: local.tp_size,
            pp_size: local.pp_size,
        });
    }
    for d in remote_descriptors {
        if d.tp_size == 0 || d.pp_size == 0 {
            return Err(CompatError::ZeroSize {
                side: "remote",
                tp_size: d.tp_size,
                pp_size: d.pp_size,
            });
        }
    }

    // Gate 2: pp_size == 1
    if local.pp_size != 1 {
        return Err(CompatError::PipelineParallelUnsupported {
            side: "local",
            pp_size: local.pp_size,
        });
    }
    for d in remote_descriptors {
        if d.pp_size != 1 {
            return Err(CompatError::PipelineParallelUnsupported {
                side: "remote",
                pp_size: d.pp_size,
            });
        }
    }

    // Gate 3: internal consistency of remote descriptors. All ranks
    // from one peer leader must agree on tp_size, shard_axis, and
    // global_extents.
    let head = &remote_descriptors[0];
    for (i, d) in remote_descriptors.iter().enumerate().skip(1) {
        if d.tp_size != head.tp_size {
            return Err(CompatError::InconsistentRemote {
                reason: format!(
                    "remote rank {} reports tp_size={} but rank 0 reports {}",
                    i, d.tp_size, head.tp_size
                ),
            });
        }
        if d.shard_axis != head.shard_axis {
            return Err(CompatError::InconsistentRemote {
                reason: format!(
                    "remote rank {} reports shard_axis={:?} but rank 0 reports {:?}",
                    i, d.shard_axis, head.shard_axis
                ),
            });
        }
        if d.global_extents != head.global_extents {
            return Err(CompatError::InconsistentRemote {
                reason: format!("remote rank {} has different global_extents than rank 0", i),
            });
        }
    }

    // Gate 3.5: rank coverage. The remote descriptor vector must
    // have exactly tp_size * pp_size entries, and the union of their
    // `rank` values must be exactly {0, 1, ..., total-1}. Anything
    // else means we received partial / duplicated / out-of-range
    // rank metadata and cannot safely plan transfers.
    let expected = head.tp_size * head.pp_size;
    let actual = remote_descriptors.len();
    if actual != expected {
        return Err(CompatError::IncompleteRemoteRanks {
            reason: format!(
                "descriptor count {} does not match tp_size * pp_size = {} * {} = {}",
                actual, head.tp_size, head.pp_size, expected
            ),
            expected,
            actual,
        });
    }
    let mut seen = vec![false; expected];
    for d in remote_descriptors {
        if d.rank >= expected {
            return Err(CompatError::IncompleteRemoteRanks {
                reason: format!(
                    "rank {} is out of range for tp_size * pp_size = {}",
                    d.rank, expected
                ),
                expected,
                actual,
            });
        }
        if seen[d.rank] {
            return Err(CompatError::IncompleteRemoteRanks {
                reason: format!("rank {} appears more than once", d.rank),
                expected,
                actual,
            });
        }
        seen[d.rank] = true;
    }
    if let Some(missing) = seen.iter().position(|present| !*present) {
        return Err(CompatError::IncompleteRemoteRanks {
            reason: format!("rank {missing} is missing from remote descriptor set"),
            expected,
            actual,
        });
    }

    // Gate 4: TP divisibility — one tp_size must divide the other.
    let local_tp = local.tp_size;
    let remote_tp = head.tp_size;
    if !local_tp.is_multiple_of(remote_tp) && !remote_tp.is_multiple_of(local_tp) {
        return Err(CompatError::Coprime {
            local_tp,
            remote_tp,
        });
    }

    // Gate 5: shard axis agreement
    if local.shard_axis != head.shard_axis {
        return Err(CompatError::ShardAxisMismatch {
            local: local.shard_axis,
            remote: head.shard_axis,
        });
    }

    // Gate 6: global extents agree on EVERY axis. `global_extents` is
    // the pre-shard model-global value, so even the shard axis must
    // match — tp_size scales the per-worker extent, not the global
    // one (e.g. global HeadCount=32 is the same whether the leader
    // runs TP=2 or TP=4; the per-worker value differs). Axes present
    // on one side but not the other are also rejected: silent
    // pass-through would let mis-stamped descriptors through.
    let local_axes: std::collections::HashSet<KvDim> =
        local.global_extents.iter().map(|(a, _)| *a).collect();
    let remote_axes: std::collections::HashSet<KvDim> =
        head.global_extents.iter().map(|(a, _)| *a).collect();
    let mut all_axes: Vec<KvDim> = local_axes.union(&remote_axes).copied().collect();
    // Deterministic iteration order for error reproducibility.
    all_axes.sort_by_key(|d| format!("{d:?}"));
    for axis in &all_axes {
        let local_val = local
            .global_extents
            .iter()
            .find(|(a, _)| a == axis)
            .map(|(_, v)| *v);
        let remote_val = head
            .global_extents
            .iter()
            .find(|(a, _)| a == axis)
            .map(|(_, v)| *v);
        match (local_val, remote_val) {
            (Some(l), Some(r)) if l != r => {
                return Err(CompatError::ExtentMismatch {
                    axis: *axis,
                    local: l,
                    remote: r,
                });
            }
            (Some(l), None) => {
                return Err(CompatError::MissingExtent {
                    axis: *axis,
                    missing_on: "remote",
                    present: l,
                });
            }
            (None, Some(r)) => {
                return Err(CompatError::MissingExtent {
                    axis: *axis,
                    missing_on: "local",
                    present: r,
                });
            }
            _ => {}
        }
    }

    // Gate 7: required logical tier present on every remote rank.
    for (i, tiers) in remote_tiers.iter().enumerate() {
        if !tiers.contains(&required_tier) {
            return Err(CompatError::MissingLogicalTier {
                rank: i,
                tier: required_tier,
            });
        }
    }

    Ok(())
}

/// Stamp a [`ParallelismDescriptor`] onto every per-worker
/// [`SerializedLayout`], producing a new vector ready for peer export.
///
/// Caller invariant: `metadata.len() == template.tp_size * template.pp_size`.
/// The function does not assume which logical-tier handles are present;
/// any handles already in the layouts list pass through untouched.
pub fn stamp_parallelism_descriptors(
    template: &ParallelismTemplate,
    metadata: Vec<SerializedLayout>,
) -> Result<Vec<SerializedLayout>> {
    let expected = template.tp_size * template.pp_size;
    if metadata.len() != expected {
        bail!(
            "stamp_parallelism_descriptors: metadata length {} does not match \
             tp_size * pp_size = {}",
            metadata.len(),
            expected
        );
    }

    let mut out = Vec::with_capacity(metadata.len());
    for (rank, layout) in metadata.into_iter().enumerate() {
        let unpacked = layout.unpack()?;
        let descriptor = template.descriptor_for(rank)?;
        let repacked = SerializedLayout::pack(
            unpacked.worker_address,
            unpacked.nixl_metadata,
            unpacked.layouts,
            Some(descriptor),
        )?;
        out.push(repacked);
    }
    Ok(out)
}

#[cfg(all(test, feature = "testing"))]
mod tests {
    use super::*;
    use kvbm_physical::manager::{LogicalLayoutDescriptor, WorkerAddress};

    fn empty_layout_for(worker_id: u64) -> SerializedLayout {
        SerializedLayout::pack(
            WorkerAddress::new(worker_id, format!("agent-{worker_id}")),
            vec![],
            Vec::<LogicalLayoutDescriptor>::new(),
            None,
        )
        .unwrap()
    }

    fn make_template(tp_size: usize) -> ParallelismTemplate {
        ParallelismTemplate {
            tp_size,
            pp_size: 1,
            parallelism_mode: ParallelismMode::TensorParallel,
            shard_axis: KvDim::HeadCount,
            global_extents: vec![(KvDim::HeadCount, 32), (KvDim::Layer, 24)],
            num_layers: 24,
            dtype_width_bytes: 2,
        }
    }

    #[test]
    fn stamps_one_descriptor_per_worker_with_correct_rank() {
        let template = make_template(4);
        let metadata = (0..4).map(empty_layout_for).collect();

        let stamped = stamp_parallelism_descriptors(&template, metadata).unwrap();

        assert_eq!(stamped.len(), 4);
        for (i, layout) in stamped.iter().enumerate() {
            let unpacked = layout.unpack().unwrap();
            let desc = unpacked.parallelism.expect("descriptor must be stamped");
            assert_eq!(desc.tp_size, 4);
            assert_eq!(desc.pp_size, 1);
            assert_eq!(desc.rank, i);
            assert_eq!(desc.shard_axis, KvDim::HeadCount);
            assert_eq!(desc.global_extents, template.global_extents);
            assert_eq!(desc.layer_ownership, 0..24);
        }
    }

    #[test]
    fn preserves_existing_layouts_and_worker_address() {
        let template = make_template(2);
        let metadata = vec![
            SerializedLayout::pack(
                WorkerAddress::new(42, "agent-42".to_string()),
                vec![1, 2, 3],
                vec![],
                None,
            )
            .unwrap(),
            empty_layout_for(7),
        ];

        let stamped = stamp_parallelism_descriptors(&template, metadata).unwrap();

        let unpacked0 = stamped[0].unpack().unwrap();
        assert_eq!(unpacked0.worker_address.worker_id, 42);
        assert_eq!(unpacked0.nixl_metadata, vec![1, 2, 3]);
        assert_eq!(unpacked0.parallelism.unwrap().rank, 0);

        let unpacked1 = stamped[1].unpack().unwrap();
        assert_eq!(unpacked1.worker_address.worker_id, 7);
        assert_eq!(unpacked1.parallelism.unwrap().rank, 1);
    }

    #[test]
    fn overwrites_any_preexisting_descriptor() {
        let template = make_template(2);
        let metadata = vec![empty_layout_for(0), empty_layout_for(1)];
        let stamped_once = stamp_parallelism_descriptors(&template, metadata).unwrap();
        let stamped_twice = stamp_parallelism_descriptors(&template, stamped_once).unwrap();

        // Re-stamping with the same template is idempotent.
        for (i, layout) in stamped_twice.iter().enumerate() {
            let unpacked = layout.unpack().unwrap();
            assert_eq!(unpacked.parallelism.unwrap().rank, i);
        }
    }

    /// Build a per-worker LayoutConfig with realistic `inner_dim =
    /// num_heads * head_size`. Returns the config plus the head_size so
    /// tests can assert the derived `HeadSize` extent.
    fn layout_with_heads(num_heads: usize, head_size: usize) -> LayoutConfig {
        LayoutConfig::builder()
            .num_blocks(16)
            .num_layers(24)
            .outer_dim(2)
            .page_size(8)
            .inner_dim(num_heads * head_size)
            .dtype_width_bytes(2)
            .num_heads(Some(num_heads))
            .build()
            .unwrap()
    }

    fn extent_for(tpl: &ParallelismTemplate, axis: KvDim) -> Option<usize> {
        tpl.global_extents
            .iter()
            .find(|(d, _)| *d == axis)
            .map(|(_, v)| *v)
    }

    #[test]
    fn from_layout_config_tp_multiplies_heads() {
        // Per-worker num_heads = 8, head_size = 64. Under TP across 4
        // workers, global HeadCount = 32 and HeadSize stays 64.
        let layout = layout_with_heads(8, 64);
        let tpl =
            ParallelismTemplate::from_layout_config(&layout, ParallelismMode::TensorParallel, 4)
                .unwrap();
        assert_eq!(tpl.tp_size, 4);
        assert_eq!(tpl.pp_size, 1);
        assert_eq!(tpl.shard_axis, KvDim::HeadCount);
        assert_eq!(
            extent_for(&tpl, KvDim::HeadCount),
            Some(32),
            "global HeadCount = per-worker num_heads * tp_size",
        );
        assert_eq!(
            extent_for(&tpl, KvDim::HeadSize),
            Some(64),
            "HeadSize = inner_dim / num_heads (head_size is global, not sharded)",
        );
        assert_eq!(tpl.num_layers, 24);
    }

    #[test]
    fn from_layout_config_replicated_keeps_heads() {
        let layout = layout_with_heads(8, 64);
        let tpl =
            ParallelismTemplate::from_layout_config(&layout, ParallelismMode::ReplicatedData, 4)
                .unwrap();
        assert_eq!(
            extent_for(&tpl, KvDim::HeadCount),
            Some(8),
            "ReplicatedData → global HeadCount == per-worker (no shard)",
        );
        assert_eq!(
            extent_for(&tpl, KvDim::HeadSize),
            Some(64),
            "HeadSize is invariant across parallelism modes",
        );
    }

    #[test]
    fn from_layout_config_rejects_indivisible_inner_dim() {
        let layout = LayoutConfig::builder()
            .num_blocks(16)
            .num_layers(24)
            .outer_dim(2)
            .page_size(8)
            .inner_dim(100)
            .dtype_width_bytes(2)
            .num_heads(Some(7))
            .build()
            .unwrap();
        let err =
            ParallelismTemplate::from_layout_config(&layout, ParallelismMode::TensorParallel, 2)
                .unwrap_err();
        assert!(err.to_string().contains("not divisible"));
    }

    #[test]
    fn from_layout_config_rejects_zero_workers() {
        let layout = layout_with_heads(8, 64);
        let err =
            ParallelismTemplate::from_layout_config(&layout, ParallelismMode::TensorParallel, 0)
                .unwrap_err();
        assert!(err.to_string().contains("num_workers"));
    }

    #[test]
    fn from_layout_config_rejects_missing_heads() {
        let layout = LayoutConfig::builder()
            .num_blocks(16)
            .num_layers(24)
            .outer_dim(2)
            .page_size(8)
            .inner_dim(64)
            .dtype_width_bytes(2)
            .build()
            .unwrap();
        let err =
            ParallelismTemplate::from_layout_config(&layout, ParallelismMode::TensorParallel, 2)
                .unwrap_err();
        assert!(err.to_string().contains("num_heads"));
    }

    #[test]
    fn rejects_length_mismatch() {
        let template = make_template(4);
        let metadata = vec![empty_layout_for(0), empty_layout_for(1)];
        let err = stamp_parallelism_descriptors(&template, metadata).unwrap_err();
        assert!(
            err.to_string().contains("tp_size * pp_size"),
            "unexpected error: {err}"
        );
    }

    // -- compatibility gate tests --

    fn local_template_tp(tp_size: usize) -> ParallelismTemplate {
        ParallelismTemplate {
            tp_size,
            pp_size: 1,
            parallelism_mode: ParallelismMode::TensorParallel,
            shard_axis: KvDim::HeadCount,
            global_extents: vec![
                (KvDim::Layer, 24),
                (KvDim::Page, 8),
                (KvDim::HeadCount, 32),
                (KvDim::HeadSize, 64),
            ],
            num_layers: 24,
            dtype_width_bytes: 2,
        }
    }

    fn remote_descriptor_tp(tp_size: usize, rank: usize) -> ParallelismDescriptor {
        ParallelismDescriptor {
            tp_size,
            pp_size: 1,
            rank,
            shard_axis: KvDim::HeadCount,
            global_extents: vec![
                (KvDim::Layer, 24),
                (KvDim::Page, 8),
                (KvDim::HeadCount, 32),
                (KvDim::HeadSize, 64),
            ],
            layer_ownership: 0..24,
        }
    }

    fn g2_tiers_for(n_ranks: usize) -> Vec<Vec<LogicalLayoutHandle>> {
        (0..n_ranks)
            .map(|_| vec![LogicalLayoutHandle::G2])
            .collect()
    }

    /// Bypass-host tier layout: [G1, G3] (no G2 — G2 allocation is
    /// skipped when DYN_KVBM_DISK_CACHE_GB is set and
    /// DYN_KVBM_CPU_CACHE_GB is unset).
    fn bypass_host_tiers_for(n_ranks: usize) -> Vec<Vec<LogicalLayoutHandle>> {
        (0..n_ranks)
            .map(|_| vec![LogicalLayoutHandle::G1, LogicalLayoutHandle::G3])
            .collect()
    }

    /// Forward-looking semantic guard: `validate_remote_metadata`
    /// itself is tier-agnostic — callers can pass `G3` and a
    /// bypass-host remote with `[G1, G3]` will be accepted at this
    /// layer. The connect_remote call site in `worker::group::spmd`
    /// still hard-codes `required_tier = G2` because the downstream
    /// RDMA pull also hard-requires G2; both must change together to
    /// enable bypass-host Universal imports end-to-end (c3 follow-on).
    /// This test pins the function's flexibility so the eventual
    /// caller change doesn't fight an unexpected gate behaviour.
    #[test]
    fn compat_accepts_bypass_host_remote_with_g3_required_tier() {
        let local = local_template_tp(2);
        let remote: Vec<_> = (0..2).map(|r| remote_descriptor_tp(2, r)).collect();
        let tiers = bypass_host_tiers_for(2);
        validate_remote_metadata(&local, &remote, &tier_refs(&tiers), LogicalLayoutHandle::G3)
            .expect("bypass-host remote (G1+G3) must be accepted when required_tier=G3");
    }

    /// Forward regression: with required_tier=G2, a bypass-host remote
    /// still rejects (because no G2). This pins the existing semantic so
    /// the fix at the call site is the only behavioural change.
    #[test]
    fn compat_rejects_bypass_host_remote_when_required_tier_is_g2() {
        let local = local_template_tp(2);
        let remote: Vec<_> = (0..2).map(|r| remote_descriptor_tp(2, r)).collect();
        let tiers = bypass_host_tiers_for(2);
        let err =
            validate_remote_metadata(&local, &remote, &tier_refs(&tiers), LogicalLayoutHandle::G2)
                .unwrap_err();
        assert!(
            matches!(
                err,
                CompatError::MissingLogicalTier {
                    tier: LogicalLayoutHandle::G2,
                    ..
                }
            ),
            "bypass-host remote must reject when required_tier=G2 (no G2 present); got: {err:?}",
        );
    }

    fn tier_refs(t: &[Vec<LogicalLayoutHandle>]) -> Vec<&[LogicalLayoutHandle]> {
        t.iter().map(|v| v.as_slice()).collect()
    }

    #[test]
    fn compat_accepts_symmetric_tp() {
        let local = local_template_tp(2);
        let remote: Vec<_> = (0..2).map(|r| remote_descriptor_tp(2, r)).collect();
        let tiers = g2_tiers_for(2);
        validate_remote_metadata(&local, &remote, &tier_refs(&tiers), LogicalLayoutHandle::G2)
            .expect("symmetric TP=2 accepted");
    }

    #[test]
    fn compat_accepts_asymmetric_tp_local_smaller() {
        let local = local_template_tp(2);
        let remote: Vec<_> = (0..4).map(|r| remote_descriptor_tp(4, r)).collect();
        let tiers = g2_tiers_for(4);
        validate_remote_metadata(&local, &remote, &tier_refs(&tiers), LogicalLayoutHandle::G2)
            .expect("TP=2 local pulling from TP=4 remote accepted");
    }

    #[test]
    fn compat_accepts_asymmetric_tp_local_larger() {
        let local = local_template_tp(4);
        let remote: Vec<_> = (0..2).map(|r| remote_descriptor_tp(2, r)).collect();
        let tiers = g2_tiers_for(2);
        validate_remote_metadata(&local, &remote, &tier_refs(&tiers), LogicalLayoutHandle::G2)
            .expect("TP=4 local pulling from TP=2 remote accepted");
    }

    #[test]
    fn compat_rejects_coprime_tp() {
        let local = local_template_tp(3);
        let remote: Vec<_> = (0..2).map(|r| remote_descriptor_tp(2, r)).collect();
        let tiers = g2_tiers_for(2);
        let err =
            validate_remote_metadata(&local, &remote, &tier_refs(&tiers), LogicalLayoutHandle::G2)
                .unwrap_err();
        assert!(matches!(
            err,
            CompatError::Coprime {
                local_tp: 3,
                remote_tp: 2
            }
        ));
    }

    #[test]
    fn compat_rejects_shard_axis_mismatch() {
        let local = local_template_tp(2);
        let mut remote: Vec<_> = (0..2).map(|r| remote_descriptor_tp(2, r)).collect();
        for d in &mut remote {
            d.shard_axis = KvDim::Layer;
        }
        let tiers = g2_tiers_for(2);
        let err =
            validate_remote_metadata(&local, &remote, &tier_refs(&tiers), LogicalLayoutHandle::G2)
                .unwrap_err();
        assert!(matches!(
            err,
            CompatError::ShardAxisMismatch {
                local: KvDim::HeadCount,
                remote: KvDim::Layer,
            }
        ));
    }

    #[test]
    fn compat_rejects_non_shard_extent_mismatch() {
        let local = local_template_tp(2);
        let mut remote: Vec<_> = (0..2).map(|r| remote_descriptor_tp(2, r)).collect();
        for d in &mut remote {
            // Change `Layer` (non-shard axis) to a different value.
            if let Some(entry) = d
                .global_extents
                .iter_mut()
                .find(|(a, _)| *a == KvDim::Layer)
            {
                entry.1 = 48;
            }
        }
        let tiers = g2_tiers_for(2);
        let err =
            validate_remote_metadata(&local, &remote, &tier_refs(&tiers), LogicalLayoutHandle::G2)
                .unwrap_err();
        assert!(matches!(
            err,
            CompatError::ExtentMismatch {
                axis: KvDim::Layer,
                local: 24,
                remote: 48
            }
        ));
    }

    #[test]
    fn compat_rejects_shard_axis_global_extent_mismatch() {
        // `global_extents` is the pre-shard model-global value, so it
        // must match on every axis — including the shard axis. The
        // per-worker (post-shard) value is `global / tp_size` and
        // legitimately differs across asymmetric-TP sides, but the
        // GLOBAL must agree between leaders running the same model.
        let mut local = local_template_tp(2);
        if let Some(e) = local
            .global_extents
            .iter_mut()
            .find(|(a, _)| *a == KvDim::HeadCount)
        {
            e.1 = 16;
        }
        // Remote keeps the canonical 32.
        let remote: Vec<_> = (0..2).map(|r| remote_descriptor_tp(2, r)).collect();
        let tiers = g2_tiers_for(2);
        let err =
            validate_remote_metadata(&local, &remote, &tier_refs(&tiers), LogicalLayoutHandle::G2)
                .unwrap_err();
        assert!(matches!(
            err,
            CompatError::ExtentMismatch {
                axis: KvDim::HeadCount,
                local: 16,
                remote: 32,
            }
        ));
    }

    #[test]
    fn compat_rejects_missing_extent_on_remote() {
        let local = local_template_tp(2);
        let mut remote: Vec<_> = (0..2).map(|r| remote_descriptor_tp(2, r)).collect();
        for d in &mut remote {
            d.global_extents.retain(|(a, _)| *a != KvDim::Layer);
        }
        let tiers = g2_tiers_for(2);
        let err =
            validate_remote_metadata(&local, &remote, &tier_refs(&tiers), LogicalLayoutHandle::G2)
                .unwrap_err();
        assert!(matches!(
            err,
            CompatError::MissingExtent {
                axis: KvDim::Layer,
                missing_on: "remote",
                ..
            }
        ));
    }

    #[test]
    fn compat_rejects_missing_extent_on_local() {
        let mut local = local_template_tp(2);
        local.global_extents.retain(|(a, _)| *a != KvDim::HeadSize);
        let remote: Vec<_> = (0..2).map(|r| remote_descriptor_tp(2, r)).collect();
        let tiers = g2_tiers_for(2);
        let err =
            validate_remote_metadata(&local, &remote, &tier_refs(&tiers), LogicalLayoutHandle::G2)
                .unwrap_err();
        assert!(matches!(
            err,
            CompatError::MissingExtent {
                axis: KvDim::HeadSize,
                missing_on: "local",
                ..
            }
        ));
    }

    #[test]
    fn compat_rejects_pp_not_one_on_remote() {
        let local = local_template_tp(2);
        let mut remote: Vec<_> = (0..2).map(|r| remote_descriptor_tp(2, r)).collect();
        for d in &mut remote {
            d.pp_size = 2;
        }
        let tiers = g2_tiers_for(2);
        let err =
            validate_remote_metadata(&local, &remote, &tier_refs(&tiers), LogicalLayoutHandle::G2)
                .unwrap_err();
        assert!(matches!(
            err,
            CompatError::PipelineParallelUnsupported {
                side: "remote",
                pp_size: 2
            }
        ));
    }

    #[test]
    fn compat_rejects_missing_logical_tier() {
        let local = local_template_tp(2);
        let remote: Vec<_> = (0..2).map(|r| remote_descriptor_tp(2, r)).collect();
        // Rank 0 has G2, rank 1 does not.
        let tiers = vec![vec![LogicalLayoutHandle::G2], vec![LogicalLayoutHandle::G1]];
        let err =
            validate_remote_metadata(&local, &remote, &tier_refs(&tiers), LogicalLayoutHandle::G2)
                .unwrap_err();
        assert!(matches!(
            err,
            CompatError::MissingLogicalTier {
                rank: 1,
                tier: LogicalLayoutHandle::G2
            }
        ));
    }

    #[test]
    fn compat_rejects_zero_tp() {
        let mut local = local_template_tp(0);
        local.tp_size = 0;
        let remote: Vec<_> = (0..2).map(|r| remote_descriptor_tp(2, r)).collect();
        let tiers = g2_tiers_for(2);
        let err =
            validate_remote_metadata(&local, &remote, &tier_refs(&tiers), LogicalLayoutHandle::G2)
                .unwrap_err();
        assert!(matches!(err, CompatError::ZeroSize { side: "local", .. }));
    }

    #[test]
    fn compat_rejects_partial_remote_rank_set() {
        // Remote descriptors claim tp_size=4 but only 2 ranks present.
        let local = local_template_tp(2);
        let remote: Vec<_> = (0..2).map(|r| remote_descriptor_tp(4, r)).collect();
        let tiers = g2_tiers_for(2);
        let err =
            validate_remote_metadata(&local, &remote, &tier_refs(&tiers), LogicalLayoutHandle::G2)
                .unwrap_err();
        assert!(matches!(
            err,
            CompatError::IncompleteRemoteRanks {
                expected: 4,
                actual: 2,
                ..
            }
        ));
    }

    #[test]
    fn compat_rejects_duplicate_remote_rank() {
        let local = local_template_tp(2);
        // Two ranks both claim rank=0.
        let remote = vec![remote_descriptor_tp(2, 0), remote_descriptor_tp(2, 0)];
        let tiers = g2_tiers_for(2);
        let err =
            validate_remote_metadata(&local, &remote, &tier_refs(&tiers), LogicalLayoutHandle::G2)
                .unwrap_err();
        let msg = match &err {
            CompatError::IncompleteRemoteRanks { reason, .. } => reason.clone(),
            other => panic!("expected IncompleteRemoteRanks, got {other:?}"),
        };
        assert!(msg.contains("more than once"), "msg: {msg}");
    }

    #[test]
    fn compat_rejects_out_of_range_remote_rank() {
        let local = local_template_tp(2);
        let remote = vec![remote_descriptor_tp(2, 0), remote_descriptor_tp(2, 7)];
        let tiers = g2_tiers_for(2);
        let err =
            validate_remote_metadata(&local, &remote, &tier_refs(&tiers), LogicalLayoutHandle::G2)
                .unwrap_err();
        let msg = match &err {
            CompatError::IncompleteRemoteRanks { reason, .. } => reason.clone(),
            other => panic!("expected IncompleteRemoteRanks, got {other:?}"),
        };
        assert!(msg.contains("out of range"), "msg: {msg}");
    }

    #[test]
    fn compat_rejects_inconsistent_remote_tp_size() {
        let local = local_template_tp(2);
        let mut remote: Vec<_> = (0..2).map(|r| remote_descriptor_tp(2, r)).collect();
        remote[1].tp_size = 4;
        let tiers = g2_tiers_for(2);
        let err =
            validate_remote_metadata(&local, &remote, &tier_refs(&tiers), LogicalLayoutHandle::G2)
                .unwrap_err();
        assert!(matches!(err, CompatError::InconsistentRemote { .. }));
    }

    #[test]
    fn rejects_pp_not_one() {
        let mut template = make_template(2);
        template.pp_size = 2;
        let metadata = vec![
            empty_layout_for(0),
            empty_layout_for(1),
            empty_layout_for(2),
            empty_layout_for(3),
        ];
        let err = stamp_parallelism_descriptors(&template, metadata).unwrap_err();
        assert!(
            err.to_string().contains("pp_size"),
            "unexpected error: {err}"
        );
    }
}
