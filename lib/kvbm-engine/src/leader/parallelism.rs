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
use kvbm_common::KvDim;
use kvbm_config::ParallelismMode;
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
}

impl ParallelismTemplate {
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
