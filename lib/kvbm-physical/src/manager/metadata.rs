// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Serialization types for exporting/importing layout metadata with NIXL integration.

use super::handle::LayoutHandle;
use crate::layout::LayoutDescriptor;
use anyhow::Result;
use bincode::{Decode, Encode};
use serde::{Deserialize, Serialize};
use std::ops::Range;

use kvbm_common::{KvDim, LogicalLayoutHandle};

/// Worker identification combining worker_id and NIXL agent name.
#[derive(Debug, Clone, Encode, Decode, PartialEq, Eq)]
pub struct WorkerAddress {
    /// Unique identifier for this worker
    pub worker_id: u64,
    /// NIXL agent name on this worker
    pub nixl_agent_name: String,
}

impl WorkerAddress {
    /// Create a new worker address.
    pub fn new(worker_id: u64, nixl_agent_name: String) -> Self {
        Self {
            worker_id,
            nixl_agent_name,
        }
    }
}

/// Layout descriptor with its assigned handle and logical type for RDMA metadata exchange.
///
/// This includes the logical layout type (G1, G2, G3, G4) so that remote instances
/// know which physical handle corresponds to which tier.
#[derive(Debug, Clone, Encode, Decode)]
pub struct LogicalLayoutDescriptor {
    /// Unique handle for this layout
    pub handle: LayoutHandle,
    /// The logical layout type (G1, G2, G3, G4)
    #[bincode(with_serde)]
    pub logical_type: LogicalLayoutHandle,
    /// Serialized layout data (uses Serde, bridged via bincode)
    #[bincode(with_serde)]
    pub layout: LayoutDescriptor,
}

impl LogicalLayoutDescriptor {
    /// Create a new layout descriptor with handle and logical type.
    pub fn new(
        handle: LayoutHandle,
        logical_type: LogicalLayoutHandle,
        layout: LayoutDescriptor,
    ) -> Self {
        Self {
            handle,
            logical_type,
            layout,
        }
    }

    /// Create a layout descriptor with G2 as the default logical type.
    ///
    /// This is provided for backwards compatibility with code that doesn't
    /// track logical types. G2 is used as the default since it's the most
    /// common tier for RDMA transfers (GPU memory for KV cache).
    ///
    /// For proper RDMA transfers between instances, use `new()` with the
    /// correct logical type from the Worker's registered handles.
    pub fn new_with_default_type(handle: LayoutHandle, layout: LayoutDescriptor) -> Self {
        Self {
            handle,
            logical_type: LogicalLayoutHandle::G2,
            layout,
        }
    }
}

/// Type alias for backwards compatibility.
pub type LocalLayoutDescriptor = LogicalLayoutDescriptor;

/// Per-worker parallelism descriptor exchanged alongside layout metadata.
///
/// Carries the information a peer leader needs to plan cross-parallelism
/// transfers (cross-TP, future cross-PP) without inferring it from
/// `Vec<SerializedLayout>.len()` or per-worker `LayoutConfig` alone.
///
/// PP fields are reserved — `pp_size` must be 1 for now;
/// `layer_ownership` must be `0..num_layers` accordingly.
#[derive(Debug, Clone, Encode, Decode, PartialEq, Eq)]
pub struct ParallelismDescriptor {
    /// Total tensor-parallel size on this worker's leader.
    pub tp_size: usize,
    /// Reserved for future pipeline-parallel support. Must be 1 today.
    pub pp_size: usize,
    /// This worker's rank within the leader's worker set: `0..tp_size * pp_size`.
    pub rank: usize,
    /// Axis along which this worker holds a shard. Typically [`KvDim::HeadCount`].
    #[bincode(with_serde)]
    pub shard_axis: KvDim,
    /// Global extents (full, un-sharded sizes) per labelled axis.
    /// Vec-of-tuples rather than a map so the wire format is deterministic
    /// and avoids requiring `Ord` on [`KvDim`].
    #[bincode(with_serde)]
    pub global_extents: Vec<(KvDim, usize)>,
    /// Layer range this worker owns. For `pp_size == 1` this is `0..num_layers`.
    #[bincode(with_serde)]
    pub layer_ownership: Range<usize>,
}

impl ParallelismDescriptor {
    /// Construct a descriptor for the single-worker, single-leader case.
    ///
    /// Used as a stub at call sites that don't yet have leader-level
    /// parallelism state plumbed through. Real descriptors come from the
    /// leader when assembling its export.
    pub fn single_worker(num_layers: usize) -> Self {
        Self {
            tp_size: 1,
            pp_size: 1,
            rank: 0,
            shard_axis: KvDim::HeadCount,
            global_extents: Vec::new(),
            layer_ownership: 0..num_layers,
        }
    }
}

/// The set of [`LogicalLayoutDescriptor`] that are RDMA enabled. This object packages the detail
/// about the layouts and the NIXL RDMA metadata required to reconstruct the layouts and access
/// the memory via NIXL RDMA.
///
/// `Decode` is implemented manually (not derived) so that the trailing
/// `parallelism` field is forward-compatible: legacy bytes encoded
/// before AB-1a (without the `parallelism` field) decode to
/// `parallelism = None` rather than failing with an EOF error. This
/// supports rolling upgrades where one side is still emitting the
/// pre-AB-1a wire shape.
#[derive(Debug, Encode)]
pub struct RdmaLayoutDescriptors {
    /// Worker identification
    pub worker_address: WorkerAddress,
    /// Exported NIXL metadata from nixl_sys::Agent::get_local_md()
    pub nixl_metadata: Vec<u8>,
    /// Serialized layouts (handle + logical type + layout data)
    pub layouts: Vec<LogicalLayoutDescriptor>,
    /// Per-worker parallelism descriptor for cross-parallelism planning.
    ///
    /// `None` covers two cases: (1) callers that don't yet have
    /// leader-level parallelism state plumbed through emit `None`; and
    /// (2) bytes encoded before AB-1a never carried this field, and
    /// the manual `unpack` synthesises `None` for them.
    pub parallelism: Option<ParallelismDescriptor>,
}

/// Managed memory metadata package for export/import.
///
/// This is the wire format for transmitting layout metadata between workers.
/// It contains everything needed to reconstruct remote layouts and load their
/// NIXL registration data.
#[derive(Clone, Serialize, Deserialize, Encode, Decode)]
#[serde(transparent)]
pub struct SerializedLayout(Vec<u8>);

impl SerializedLayout {
    /// Pack metadata into a serialized form.
    ///
    /// # Arguments
    /// * `worker_address` - Worker identification
    /// * `nixl_metadata` - NIXL metadata blob from get_local_md()
    /// * `layouts` - Vector of layouts with handles and logical types to export
    /// * `parallelism` - Optional [`ParallelismDescriptor`] for cross-parallelism
    ///   planning. `None` is the transitional default until leader-level
    ///   parallelism state is plumbed through (AB-1a follow-up).
    ///
    /// # Returns
    /// Packed metadata ready for transmission
    pub fn pack(
        worker_address: WorkerAddress,
        nixl_metadata: Vec<u8>,
        layouts: Vec<LogicalLayoutDescriptor>,
        parallelism: Option<ParallelismDescriptor>,
    ) -> Result<Self> {
        let inner = RdmaLayoutDescriptors {
            worker_address,
            nixl_metadata,
            layouts,
            parallelism,
        };
        let bytes = bincode::encode_to_vec(&inner, bincode::config::standard())
            .map_err(|e| anyhow::anyhow!("failed to encode managed memory metadata: {}", e))?;
        Ok(Self(bytes))
    }

    /// Unpack metadata from serialized form.
    ///
    /// Decodes field-by-field rather than via a derived `Decode` impl so
    /// that the trailing `parallelism` field is forward-compatible:
    /// legacy bytes encoded before AB-1a stop after `layouts`, and we
    /// synthesise `parallelism = None` for them. New bytes always carry
    /// at least the `Option` discriminant.
    ///
    /// # Returns
    /// Unpacked metadata structure
    pub fn unpack(&self) -> Result<RdmaLayoutDescriptors> {
        let cfg = bincode::config::standard();
        let bytes = &self.0[..];

        let (worker_address, c1): (WorkerAddress, usize) =
            bincode::decode_from_slice(bytes, cfg)
                .map_err(|e| anyhow::anyhow!("failed to decode worker_address: {}", e))?;
        let rest = &bytes[c1..];

        let (nixl_metadata, c2): (Vec<u8>, usize) = bincode::decode_from_slice(rest, cfg)
            .map_err(|e| anyhow::anyhow!("failed to decode nixl_metadata: {}", e))?;
        let rest = &rest[c2..];

        let (layouts, c3): (Vec<LogicalLayoutDescriptor>, usize) =
            bincode::decode_from_slice(rest, cfg)
                .map_err(|e| anyhow::anyhow!("failed to decode layouts: {}", e))?;
        let rest = &rest[c3..];

        let parallelism = if rest.is_empty() {
            // Pre-AB-1a wire shape — trailing field absent.
            None
        } else {
            let (p, _): (Option<ParallelismDescriptor>, usize) =
                bincode::decode_from_slice(rest, cfg).map_err(|e| {
                    anyhow::anyhow!("failed to decode parallelism descriptor: {}", e)
                })?;
            p
        };

        Ok(RdmaLayoutDescriptors {
            worker_address,
            nixl_metadata,
            layouts,
            parallelism,
        })
    }

    /// Get the raw bytes.
    pub fn as_bytes(&self) -> &[u8] {
        &self.0
    }

    /// Create from raw bytes.
    pub fn from_bytes(bytes: Vec<u8>) -> Self {
        Self(bytes)
    }

    /// Get the size in bytes.
    pub fn len(&self) -> usize {
        self.0.len()
    }

    /// Check if empty.
    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }
}

/// Pick the layout that peers will transfer over (the "transfer-canonical"
/// tier).
///
/// In `BlockLayoutMode::Universal` the connector emits G2 (and G3)
/// with `KvBlockLayout::Universal` regardless of G1 (c3), so
/// cross-leader compat checks must inspect G2's (or G3's) layout, not
/// G1's. In `BlockLayoutMode::Operational` every tier inherits G1's
/// layout, so picking G2 vs G3 vs first-layout is a no-op for the
/// equality check.
///
/// Fallback chain (paired with [`select_transfer_canonical_tier`]):
/// 1. G2 if present (normal mode).
/// 2. G3 if present (bypass-host mode: `DYN_KVBM_DISK_CACHE_GB` set,
///    `DYN_KVBM_CPU_CACHE_GB` unset — G2 is skipped and transfers go
///    G1↔G3 directly via GDS).
/// 3. First layout (degenerate: only G1 exists).
///
/// Returns `None` only when the layouts vec is empty.
pub fn select_transfer_canonical_layout(
    layouts: &[LogicalLayoutDescriptor],
) -> Option<&LogicalLayoutDescriptor> {
    use kvbm_common::LogicalLayoutHandle;
    layouts
        .iter()
        .find(|l| l.logical_type == LogicalLayoutHandle::G2)
        .or_else(|| {
            layouts
                .iter()
                .find(|l| l.logical_type == LogicalLayoutHandle::G3)
        })
        .or_else(|| layouts.first())
}

/// Pick the tier that peers will transfer over (handle-level analogue of
/// [`select_transfer_canonical_layout`]).
///
/// Same fallback chain — G2 → G3 → first — but operates on a slice of
/// [`kvbm_common::LogicalLayoutHandle`] (just the tier list, no layout
/// payload). Used by `validate_remote_metadata` to determine which
/// tier must be present on every remote rank: hard-coding G2 there
/// rejects legitimate bypass-host peers whose layout vec is
/// `[G1, G3]`. Returns `None` only when the tier list is empty.
pub fn select_transfer_canonical_tier(
    tiers: &[kvbm_common::LogicalLayoutHandle],
) -> Option<kvbm_common::LogicalLayoutHandle> {
    use kvbm_common::LogicalLayoutHandle;
    tiers
        .iter()
        .copied()
        .find(|t| *t == LogicalLayoutHandle::G2)
        .or_else(|| {
            tiers
                .iter()
                .copied()
                .find(|t| *t == LogicalLayoutHandle::G3)
        })
        .or_else(|| tiers.first().copied())
}

impl std::fmt::Debug for SerializedLayout {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SerializedLayout")
            .field("size_bytes", &self.len())
            .finish()
    }
}

#[cfg(all(test, feature = "testing-kvbm"))]
mod tests {
    use super::*;
    use crate::layout::{
        BlockFormat, FullyContiguousDetails, KvBlockLayout, LayoutConfig, LayoutDescriptor,
        LayoutTypeDetails, NixlMetadata,
    };
    use dynamo_memory::{MemoryRegion, StorageKind, nixl};
    use kvbm_common::LogicalLayoutHandle;

    fn make_test_serialized_layout() -> LayoutDescriptor {
        let config = LayoutConfig::builder()
            .num_blocks(2)
            .num_layers(2)
            .outer_dim(2)
            .page_size(4)
            .inner_dim(8)
            .dtype_width_bytes(2)
            .build()
            .unwrap();

        LayoutDescriptor {
            version: 1,
            layout_config: config,
            location: StorageKind::System,
            nixl_metadata: NixlMetadata::new("test".to_string(), nixl::MemType::Dram, 0),
            memory_descriptors: vec![MemoryRegion {
                addr: 0x1000,
                size: 4096,
            }],
            layout_type_details: LayoutTypeDetails::FullyContiguous(FullyContiguousDetails {
                block_format: BlockFormat::Operational,
                kv_block_layout: KvBlockLayout::OperationalNHD,
            }),
        }
    }

    /// Bypass-host mode (DYN_KVBM_DISK_CACHE_GB set, DYN_KVBM_CPU_CACHE_GB
    /// unset) skips G2 allocation; the layouts vec is [G1, G3] only.
    /// In Universal mode the connector pins G3 to KvBlockLayout::Universal
    /// (mode-dominant selection, c3). The transfer-canonical helper must
    /// pick G3 — picking G1 here would be the bug Codex caught at c3
    /// stop-time review, where the wire payload would falsely report
    /// G1's OperationalNHD layout instead of G3's Universal.
    #[test]
    fn select_transfer_canonical_prefers_g3_when_g2_absent() {
        let make = |kv: KvBlockLayout| LayoutDescriptor {
            version: 1,
            layout_config: LayoutConfig::builder()
                .num_blocks(2)
                .num_layers(2)
                .outer_dim(2)
                .page_size(4)
                .inner_dim(8)
                .dtype_width_bytes(2)
                .build()
                .unwrap(),
            location: StorageKind::System,
            nixl_metadata: NixlMetadata::new("test".to_string(), nixl::MemType::Dram, 0),
            memory_descriptors: vec![MemoryRegion {
                addr: 0x1000,
                size: 4096,
            }],
            layout_type_details: LayoutTypeDetails::FullyContiguous(FullyContiguousDetails {
                block_format: BlockFormat::Operational,
                kv_block_layout: kv,
            }),
        };
        let layouts = vec![
            LogicalLayoutDescriptor::new(
                LayoutHandle::new(1, 0),
                LogicalLayoutHandle::G1,
                make(KvBlockLayout::OperationalNHD),
            ),
            LogicalLayoutDescriptor::new(
                LayoutHandle::new(1, 1),
                LogicalLayoutHandle::G3,
                make(KvBlockLayout::Universal),
            ),
        ];

        let picked = select_transfer_canonical_layout(&layouts)
            .expect("non-empty layouts must yield a selection");
        assert_eq!(
            picked.logical_type,
            LogicalLayoutHandle::G3,
            "bypass-host: G2 absent so helper must fall back to G3, not G1",
        );
    }

    /// Codex stop-time review (round 3): the validate_remote_metadata
    /// site hard-coded `LogicalLayoutHandle::G2` as the required tier,
    /// which rejects bypass-host peers whose tier list is [G1, G3].
    /// The handle-level helper must mirror the layout-level fallback
    /// (G2 → G3 → first) so the call site can pass the local leader's
    /// actual transfer tier.
    #[test]
    fn select_transfer_canonical_tier_prefers_g3_when_g2_absent() {
        use kvbm_common::LogicalLayoutHandle::{G1, G2, G3};
        assert_eq!(
            select_transfer_canonical_tier(&[G1, G2, G3]),
            Some(G2),
            "G2 wins when present",
        );
        assert_eq!(
            select_transfer_canonical_tier(&[G1, G3]),
            Some(G3),
            "G3 is the bypass-host fallback when G2 is absent",
        );
        assert_eq!(
            select_transfer_canonical_tier(&[G1]),
            Some(G1),
            "single-tier degenerate falls through to first",
        );
        assert_eq!(select_transfer_canonical_tier(&[]), None);
    }

    /// Normal mode: G1+G2+G3 all present, helper picks G2.
    #[test]
    fn select_transfer_canonical_prefers_g2_when_all_present() {
        let make = |kv: KvBlockLayout| LayoutDescriptor {
            version: 1,
            layout_config: LayoutConfig::builder()
                .num_blocks(2)
                .num_layers(2)
                .outer_dim(2)
                .page_size(4)
                .inner_dim(8)
                .dtype_width_bytes(2)
                .build()
                .unwrap(),
            location: StorageKind::System,
            nixl_metadata: NixlMetadata::new("test".to_string(), nixl::MemType::Dram, 0),
            memory_descriptors: vec![MemoryRegion {
                addr: 0x1000,
                size: 4096,
            }],
            layout_type_details: LayoutTypeDetails::FullyContiguous(FullyContiguousDetails {
                block_format: BlockFormat::Operational,
                kv_block_layout: kv,
            }),
        };
        let layouts = vec![
            LogicalLayoutDescriptor::new(
                LayoutHandle::new(1, 0),
                LogicalLayoutHandle::G1,
                make(KvBlockLayout::OperationalNHD),
            ),
            LogicalLayoutDescriptor::new(
                LayoutHandle::new(1, 1),
                LogicalLayoutHandle::G2,
                make(KvBlockLayout::Universal),
            ),
            LogicalLayoutDescriptor::new(
                LayoutHandle::new(1, 2),
                LogicalLayoutHandle::G3,
                make(KvBlockLayout::Universal),
            ),
        ];

        let picked = select_transfer_canonical_layout(&layouts)
            .expect("non-empty layouts must yield a selection");
        assert_eq!(picked.logical_type, LogicalLayoutHandle::G2);
    }

    #[test]
    fn test_worker_address() {
        let addr = WorkerAddress::new(42, "test_agent".to_string());
        assert_eq!(addr.worker_id, 42);
        assert_eq!(addr.nixl_agent_name, "test_agent");
    }

    #[test]
    fn test_serialized_layout_with_handle() {
        let handle = LayoutHandle::new(1, 2);
        let layout = make_test_serialized_layout();
        let with_handle = LogicalLayoutDescriptor::new(handle, LogicalLayoutHandle::G2, layout);

        assert_eq!(with_handle.handle, handle);
        assert_eq!(with_handle.logical_type, LogicalLayoutHandle::G2);
    }

    #[test]
    fn test_metadata_pack_unpack() {
        let worker_address = WorkerAddress::new(100, "worker_100".to_string());
        let nixl_metadata = vec![1, 2, 3, 4, 5];
        let layouts = vec![LogicalLayoutDescriptor::new(
            LayoutHandle::new(100, 1),
            LogicalLayoutHandle::G2,
            make_test_serialized_layout(),
        )];

        let packed =
            SerializedLayout::pack(worker_address.clone(), nixl_metadata.clone(), layouts, None)
                .unwrap();

        assert!(!packed.is_empty());

        let unpacked = packed.unpack().unwrap();

        assert_eq!(unpacked.worker_address, worker_address);
        assert_eq!(unpacked.nixl_metadata, nixl_metadata);
        assert_eq!(unpacked.layouts.len(), 1);
        assert_eq!(unpacked.layouts[0].handle.worker_id(), 100);
        assert_eq!(unpacked.layouts[0].handle.layout_id(), 1);
        assert_eq!(unpacked.layouts[0].logical_type, LogicalLayoutHandle::G2);
    }

    #[test]
    fn test_metadata_multiple_layouts() {
        let worker_address = WorkerAddress::new(200, "worker_200".to_string());
        let nixl_metadata = vec![10, 20, 30];
        let layouts = vec![
            LogicalLayoutDescriptor::new(
                LayoutHandle::new(200, 1),
                LogicalLayoutHandle::G1,
                make_test_serialized_layout(),
            ),
            LogicalLayoutDescriptor::new(
                LayoutHandle::new(200, 2),
                LogicalLayoutHandle::G2,
                make_test_serialized_layout(),
            ),
            LogicalLayoutDescriptor::new(
                LayoutHandle::new(200, 3),
                LogicalLayoutHandle::G3,
                make_test_serialized_layout(),
            ),
        ];

        let packed =
            SerializedLayout::pack(worker_address, nixl_metadata, layouts.clone(), None).unwrap();
        let unpacked = packed.unpack().unwrap();

        assert_eq!(unpacked.layouts.len(), 3);
        let expected_logical_types = [
            LogicalLayoutHandle::G1,
            LogicalLayoutHandle::G2,
            LogicalLayoutHandle::G3,
        ];
        for (i, layout) in unpacked.layouts.iter().enumerate() {
            assert_eq!(layout.handle.worker_id(), 200);
            assert_eq!(layout.handle.layout_id(), (i + 1) as u16);
            assert_eq!(layout.logical_type, expected_logical_types[i]);
        }
    }

    #[test]
    fn test_parallelism_descriptor_roundtrip() {
        let worker_address = WorkerAddress::new(7, "tp-worker".to_string());
        let nixl_metadata = vec![9; 16];
        let layouts = vec![LogicalLayoutDescriptor::new(
            LayoutHandle::new(7, 1),
            LogicalLayoutHandle::G2,
            make_test_serialized_layout(),
        )];
        let parallelism = ParallelismDescriptor {
            tp_size: 4,
            pp_size: 1,
            rank: 2,
            shard_axis: KvDim::HeadCount,
            global_extents: vec![(KvDim::HeadCount, 32), (KvDim::Layer, 24)],
            layer_ownership: 0..24,
        };

        let packed = SerializedLayout::pack(
            worker_address,
            nixl_metadata,
            layouts,
            Some(parallelism.clone()),
        )
        .unwrap();
        let unpacked = packed.unpack().unwrap();

        assert_eq!(unpacked.parallelism.as_ref(), Some(&parallelism));
    }

    /// Legacy wire shape — pre-AB-1a bytes had only three fields and no
    /// trailing parallelism. A new decoder must read these and synthesise
    /// `parallelism = None`, not fail with an EOF error.
    #[derive(bincode::Encode)]
    struct LegacyRdmaLayoutDescriptors {
        worker_address: WorkerAddress,
        nixl_metadata: Vec<u8>,
        layouts: Vec<LogicalLayoutDescriptor>,
    }

    #[test]
    fn test_legacy_bytes_decode_to_none() {
        let worker_address = WorkerAddress::new(11, "legacy".to_string());
        let nixl_metadata = vec![0xab; 12];
        let layouts = vec![LogicalLayoutDescriptor::new(
            LayoutHandle::new(11, 1),
            LogicalLayoutHandle::G2,
            make_test_serialized_layout(),
        )];
        let legacy = LegacyRdmaLayoutDescriptors {
            worker_address: worker_address.clone(),
            nixl_metadata: nixl_metadata.clone(),
            layouts: layouts.clone(),
        };

        // Encode in the pre-AB-1a wire shape (three fields only).
        let legacy_bytes = bincode::encode_to_vec(&legacy, bincode::config::standard()).unwrap();

        // Wrap the legacy bytes in the new SerializedLayout container.
        let packed = SerializedLayout::from_bytes(legacy_bytes);
        let unpacked = packed
            .unpack()
            .expect("new decoder must read legacy bytes without error");

        assert_eq!(unpacked.worker_address, worker_address);
        assert_eq!(unpacked.nixl_metadata, nixl_metadata);
        assert_eq!(unpacked.layouts.len(), 1);
        assert!(unpacked.parallelism.is_none());
    }

    #[test]
    fn test_parallelism_descriptor_absent() {
        let worker_address = WorkerAddress::new(8, "no-parallelism".to_string());
        let packed = SerializedLayout::pack(worker_address, Vec::new(), Vec::new(), None).unwrap();
        let unpacked = packed.unpack().unwrap();
        assert!(unpacked.parallelism.is_none());
    }

    #[test]
    fn test_metadata_from_bytes() {
        let worker_address = WorkerAddress::new(42, "test".to_string());
        let nixl_metadata = vec![1, 2, 3];
        let layouts = vec![];

        let packed = SerializedLayout::pack(worker_address, nixl_metadata, layouts, None).unwrap();
        let bytes = packed.as_bytes().to_vec();

        let restored = SerializedLayout::from_bytes(bytes);
        let unpacked = restored.unpack().unwrap();

        assert_eq!(unpacked.worker_address.worker_id, 42);
    }
}
