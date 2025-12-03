// SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Typestate builder for NIXL transfers.
//!
//! This module provides a compile-time safe builder for NIXL transfers that ensures
//! all required parameters are set before execution.

use super::{PhysicalLayout, TransferContext, TransferStrategy};
use crate::block_manager::v2::physical::transfer::DescriptorHint;
use std::ops::Range;

/// Parameters for building NIXL descriptors.
///
/// Groups the common parameters needed by all descriptor building functions,
/// reducing function argument count and improving readability.
pub(crate) struct DescriptorParams<'a> {
    pub src: &'a PhysicalLayout,
    pub dst: &'a PhysicalLayout,
    pub src_block_ids: &'a [usize],
    pub dst_block_ids: &'a [usize],
    pub layers: &'a Range<usize>,
    pub hint: DescriptorHint,
}
use crate::block_manager::v2::physical::transfer::executor::contiguous_transfer::{
    build_batched_contiguous_transfer, build_multi_descriptor_transfer,
    build_per_block_offset_transfer, build_per_block_unique_target_transfer,
    build_single_descriptor_transfer,
};
use crate::block_manager::v2::physical::transfer::executor::object_transfer::get_object_device_id;
use crate::block_manager::v2::memory::StorageKind;
use crate::block_manager::v2::physical::transfer::context::TransferCompleteNotification;
use anyhow::{anyhow, Result};
use nixl_sys::{XferDescList, XferOp};
use std::marker::PhantomData;

/// Marker type for unset builder fields.
pub struct Unset;

/// Marker type for set builder fields.
pub struct Set;

/// Typestate builder for NIXL transfers.
///
/// This builder uses the typestate pattern to ensure all required parameters are set
/// at compile time. The type parameters track which fields have been set:
/// - `TSrc`: Source layout state
/// - `TDst`: Destination layout state
/// - `TSrcBlocks`: Source block IDs state
/// - `TDstBlocks`: Destination block IDs state
/// - `TStrategy`: Transfer strategy state
pub struct NixlTransferBuilder<'a, TSrc, TDst, TSrcBlocks, TDstBlocks, TStrategy> {
    src: Option<&'a PhysicalLayout>,
    dst: Option<&'a PhysicalLayout>,
    src_block_ids: Option<&'a [usize]>,
    dst_block_ids: Option<&'a [usize]>,
    strategy: Option<TransferStrategy>,
    layer_range: Option<Range<usize>>,
    write_notif: Option<uuid::Uuid>,
    descriptor_hint: DescriptorHint,
    _phantom: PhantomData<(TSrc, TDst, TSrcBlocks, TDstBlocks, TStrategy)>,
}

impl<'a> NixlTransferBuilder<'a, Unset, Unset, Unset, Unset, Unset> {
    /// Creates a new NIXL transfer builder with all fields unset.
    pub fn new() -> Self {
        Self {
            src: None,
            dst: None,
            src_block_ids: None,
            dst_block_ids: None,
            strategy: None,
            layer_range: None,
            write_notif: None,
            descriptor_hint: DescriptorHint::Auto,
            _phantom: PhantomData,
        }
    }
}

impl<'a> Default for NixlTransferBuilder<'a, Unset, Unset, Unset, Unset, Unset> {
    fn default() -> Self {
        Self::new()
    }
}

// Required field setters - these consume self and return a new builder with the field marked as Set

impl<'a, TDst, TSrcBlocks, TDstBlocks, TStrategy>
    NixlTransferBuilder<'a, Unset, TDst, TSrcBlocks, TDstBlocks, TStrategy>
{
    /// Sets the source physical layout.
    pub fn src(
        self,
        src: &'a PhysicalLayout,
    ) -> NixlTransferBuilder<'a, Set, TDst, TSrcBlocks, TDstBlocks, TStrategy> {
        NixlTransferBuilder {
            src: Some(src),
            dst: self.dst,
            src_block_ids: self.src_block_ids,
            dst_block_ids: self.dst_block_ids,
            strategy: self.strategy,
            layer_range: self.layer_range,
            write_notif: self.write_notif,
            descriptor_hint: self.descriptor_hint,
            _phantom: PhantomData,
        }
    }
}

impl<'a, TSrc, TSrcBlocks, TDstBlocks, TStrategy>
    NixlTransferBuilder<'a, TSrc, Unset, TSrcBlocks, TDstBlocks, TStrategy>
{
    /// Sets the destination physical layout.
    pub fn dst(
        self,
        dst: &'a PhysicalLayout,
    ) -> NixlTransferBuilder<'a, TSrc, Set, TSrcBlocks, TDstBlocks, TStrategy> {
        NixlTransferBuilder {
            src: self.src,
            dst: Some(dst),
            src_block_ids: self.src_block_ids,
            dst_block_ids: self.dst_block_ids,
            strategy: self.strategy,
            layer_range: self.layer_range,
            write_notif: self.write_notif,
            descriptor_hint: self.descriptor_hint,
            _phantom: PhantomData,
        }
    }
}

impl<'a, TSrc, TDst, TDstBlocks, TStrategy>
    NixlTransferBuilder<'a, TSrc, TDst, Unset, TDstBlocks, TStrategy>
{
    /// Sets the source block IDs to transfer.
    pub fn src_blocks(
        self,
        src_block_ids: &'a [usize],
    ) -> NixlTransferBuilder<'a, TSrc, TDst, Set, TDstBlocks, TStrategy> {
        NixlTransferBuilder {
            src: self.src,
            dst: self.dst,
            src_block_ids: Some(src_block_ids),
            dst_block_ids: self.dst_block_ids,
            strategy: self.strategy,
            layer_range: self.layer_range,
            write_notif: self.write_notif,
            descriptor_hint: self.descriptor_hint,
            _phantom: PhantomData,
        }
    }
}

impl<'a, TSrc, TDst, TSrcBlocks, TStrategy>
    NixlTransferBuilder<'a, TSrc, TDst, TSrcBlocks, Unset, TStrategy>
{
    /// Sets the destination block IDs to transfer.
    pub fn dst_blocks(
        self,
        dst_block_ids: &'a [usize],
    ) -> NixlTransferBuilder<'a, TSrc, TDst, TSrcBlocks, Set, TStrategy> {
        NixlTransferBuilder {
            src: self.src,
            dst: self.dst,
            src_block_ids: self.src_block_ids,
            dst_block_ids: Some(dst_block_ids),
            strategy: self.strategy,
            layer_range: self.layer_range,
            write_notif: self.write_notif,
            descriptor_hint: self.descriptor_hint,
            _phantom: PhantomData,
        }
    }
}

impl<'a, TSrc, TDst, TSrcBlocks, TDstBlocks>
    NixlTransferBuilder<'a, TSrc, TDst, TSrcBlocks, TDstBlocks, Unset>
{
    /// Sets the NIXL transfer strategy (Read or Write).
    pub fn strategy(
        self,
        strategy: TransferStrategy,
    ) -> NixlTransferBuilder<'a, TSrc, TDst, TSrcBlocks, TDstBlocks, Set> {
        NixlTransferBuilder {
            src: self.src,
            dst: self.dst,
            src_block_ids: self.src_block_ids,
            dst_block_ids: self.dst_block_ids,
            strategy: Some(strategy),
            layer_range: self.layer_range,
            write_notif: self.write_notif,
            descriptor_hint: self.descriptor_hint,
            _phantom: PhantomData,
        }
    }
}

// Optional field setters - these can be called at any point in the builder chain

impl<'a, TSrc, TDst, TSrcBlocks, TDstBlocks, TStrategy>
    NixlTransferBuilder<'a, TSrc, TDst, TSrcBlocks, TDstBlocks, TStrategy>
{
    /// Sets an optional range of layers to transfer.
    /// If not called, all layers will be transferred.
    pub fn layer_range(mut self, layer_range: Range<usize>) -> Self {
        self.layer_range = Some(layer_range);
        self
    }

    /// Sets an optional write notification UUID.
    pub fn write_notif(mut self, write_notif: uuid::Uuid) -> Self {
        self.write_notif = Some(write_notif);
        self
    }

    /// Sets a descriptor hint to optimize how blocks are mapped to descriptors.
    ///
    /// When `Auto` (default), the executor analyzes layout characteristics.
    /// Explicit hints skip detection and ensure correct descriptor pattern.
    pub fn descriptor_hint(mut self, hint: DescriptorHint) -> Self {
        self.descriptor_hint = hint;
        self
    }
}

// Execute method - only available when all required fields are Set

impl<'a> NixlTransferBuilder<'a, Set, Set, Set, Set, Set> {
    /// Executes the NIXL transfer with the configured parameters.
    ///
    /// This method is only available when all required fields have been set,
    /// enforced at compile time by the typestate pattern.
    pub(crate) fn execute(self, ctx: &TransferContext) -> Result<TransferCompleteNotification> {
        // Unwrap all required fields (safe because typestate guarantees they're set)
        let src = self.src.unwrap();
        let dst = self.dst.unwrap();
        let src_block_ids = self.src_block_ids.unwrap();
        let dst_block_ids = self.dst_block_ids.unwrap();
        let strategy = self.strategy.unwrap();
        let layer_range = self.layer_range;
        let _write_notif = self.write_notif;
        let descriptor_hint = self.descriptor_hint;

        // Validate layouts
        let src_layout = src.layout();
        let dst_layout = dst.layout();

        if src_layout.num_layers() != dst_layout.num_layers() {
            return Err(anyhow!(
                "Layouts have incompatible layer counts: src={}, dst={}",
                src_layout.num_layers(),
                dst_layout.num_layers()
            ));
        }

        if src_layout.outer_dim() != dst_layout.outer_dim() {
            return Err(anyhow!(
                "Layouts have incompatible outer dimensions: src={}, dst={}",
                src_layout.outer_dim(),
                dst_layout.outer_dim()
            ));
        }

        // Get NIXL agent
        let nixl_agent = ctx.nixl_agent();

        // Determine layer range
        let layers = layer_range.unwrap_or(0..src_layout.num_layers());

        // Determine NIXL operation type
        let xfer_op = match strategy {
            TransferStrategy::NixlRead | TransferStrategy::NixlReadFlipped => XferOp::Read,
            TransferStrategy::NixlWrite | TransferStrategy::NixlWriteFlipped => XferOp::Write,
            _ => {
                return Err(anyhow!("Invalid NIXL transfer strategy: {:?}", strategy));
            }
        };

        // For flipped operations, the actual NIXL local side will be swapped later
        // For normal operations, source must be local
        let is_flipped = matches!(
            strategy,
            TransferStrategy::NixlReadFlipped | TransferStrategy::NixlWriteFlipped
        );

        if !is_flipped {
            assert!(
                nixl_agent.name() == src.nixl_metadata().agent_name(),
                "the source must be local for non-flipped NIXL operations"
            );
        } else {
            // For flipped ops, destination is actually the local side (gets swapped)
            assert!(
                nixl_agent.name() == dst.nixl_metadata().agent_name(),
                "the destination must be local for flipped NIXL operations"
            );
        }

        // Capture NIXL metadata for both layouts
        let src_metadata = src.nixl_metadata();
        let dst_metadata = dst.nixl_metadata();

        // Build XferDescLists for source and destination
        let mut src_dl = XferDescList::new(src_metadata.mem_type())?;
        let mut dst_dl = XferDescList::new(dst_metadata.mem_type())?;

        tracing::trace!(
            "Building NIXL transfer: blocks={}, layers={}, op={:?}, src={:?}, dst={:?}, hint={:?}",
            src_block_ids.len(),
            layers.len(),
            xfer_op,
            src.location(),
            dst.location(),
            descriptor_hint
        );

        // Build descriptors based on transfer type (using hint or auto-detection)
        let params = DescriptorParams {
            src,
            dst,
            src_block_ids,
            dst_block_ids,
            layers: &layers,
            hint: descriptor_hint,
        };
        build_descriptors(&params, &mut src_dl, &mut dst_dl)?;

        tracing::trace!(
            "Built descriptors: src={:?}, dst={:?}",
            src_dl,
            dst_dl,
        );

        // Swap descriptors for flipped operations
        if matches!(
            strategy,
            TransferStrategy::NixlReadFlipped | TransferStrategy::NixlWriteFlipped
        ) {
            std::mem::swap(&mut src_dl, &mut dst_dl);
        }

        // Create transfer request
        // remote_agent should be the agent name of the REMOTE layout (not local)
        // For Read: remote is source, For Write: remote is destination
        let remote_agent_name = match xfer_op {
            XferOp::Read => src_metadata.agent_name(),  // Reading FROM source (remote)
            XferOp::Write => dst_metadata.agent_name(), // Writing TO destination (remote)
        };

        let xfer_req = nixl_agent.create_xfer_req(
            xfer_op,
            &src_dl,
            &dst_dl,
            remote_agent_name,
            None, // opt_args
        )?;

        // Post transfer request
        let still_pending = nixl_agent.post_xfer_req(&xfer_req, None)?;

        if still_pending {
            // Register for async completion via status polling
            Ok(ctx.register_nixl_status(xfer_req))
        } else {
            // Transfer completed synchronously
            Ok(TransferCompleteNotification::completed())
        }
    }

}

/// Build NIXL descriptors based on descriptor hint (or auto-detection).
///
/// Uses explicit hints when provided, otherwise auto-detects from layout characteristics.
pub(crate) fn build_descriptors(
    params: &DescriptorParams,
    src_dl: &mut XferDescList,
    dst_dl: &mut XferDescList,
) -> Result<()> {
    let DescriptorParams {
        src,
        dst,
        src_block_ids,
        dst_block_ids,
        layers,
        hint,
    } = params;

    // Derive device IDs from layouts
    let src_device_id = src.nixl_metadata().device_id();
    let dst_device_id = dst.nixl_metadata().device_id();

    // Resolve hint to concrete strategy
    let resolved_hint = match hint {
        DescriptorHint::Auto => auto_detect_hint(src, dst),
        explicit => *explicit,
    };

    tracing::debug!("Descriptor strategy: {:?} (requested: {:?})", resolved_hint, hint);

    // Check if either side involves object storage (needs per-block device IDs)
    let src_is_object = matches!(src.location(), StorageKind::Object(_));
    let dst_is_object = matches!(dst.location(), StorageKind::Object(_));
    let involves_object = src_is_object || dst_is_object;

    // Device ID closure for object storage (per-block keys) or None for others
    let get_device_id_fn: Option<&dyn Fn(&PhysicalLayout, usize, u64) -> u64> = if involves_object {
        Some(&|layout, block_id, default_id| get_object_device_id(layout, block_id, default_id))
    } else {
        None
    };

    match resolved_hint {
        DescriptorHint::Coalesced => {
            // Single descriptor covering all blocks
            build_single_descriptor_transfer(
                src, dst, src_block_ids, dst_block_ids, layers,
                src_device_id, dst_device_id, src_dl, dst_dl,
            )
        }
        DescriptorHint::PerBlockWithOffset => {
            // One descriptor per block, each at its own offset (reads)
            build_per_block_offset_transfer(
                src, dst, src_block_ids, dst_block_ids, layers,
                src_device_id, dst_device_id, src_dl, dst_dl,
                get_device_id_fn,
            )
        }
        DescriptorHint::PerBlockUniqueTarget => {
            // One descriptor per block, each to unique target (writes)
            // Object storage requires zero offset (doesn't support partial writes)
            build_per_block_unique_target_transfer(
                src, dst, src_block_ids, dst_block_ids, layers,
                src_device_id, dst_device_id, src_dl, dst_dl,
                get_device_id_fn,
                dst_is_object, // force_zero_offset for object storage writes
            )
        }
        DescriptorHint::BatchedRanges => {
            // Batched contiguous ranges
            tracing::debug!(
                "Batching optimization: {} blocks, src={:?}, dst={:?}",
                src_block_ids.len(), src.location(), dst.location()
            );
            build_batched_contiguous_transfer(
                src, dst, src_block_ids, dst_block_ids, layers,
                src_device_id, dst_device_id, src_dl, dst_dl,
            )
        }
        DescriptorHint::Auto => {
            // Fallback: multi-descriptor mode (shouldn't reach here after auto_detect)
            tracing::debug!(
                "Multi-descriptor mode: {} blocks",
                src_block_ids.len()
            );
            build_multi_descriptor_transfer(
                src, dst, src_block_ids, dst_block_ids, layers,
                &|layout, block_id, default_id| get_object_device_id(layout, block_id, default_id),
                src_device_id, dst_device_id, src_dl, dst_dl,
            )
        }
    }
}

/// Auto-detect descriptor hint from layout characteristics.
fn auto_detect_hint(src: &PhysicalLayout, dst: &PhysicalLayout) -> DescriptorHint {
    let src_is_object = matches!(src.location(), StorageKind::Object(_));
    let dst_is_object = matches!(dst.location(), StorageKind::Object(_));
    let src_is_contiguous = src.layout().is_fully_contiguous();
    let dst_is_contiguous = dst.layout().is_fully_contiguous();

    if src_is_object && dst_is_contiguous {
        // Reading from object storage (supports byte-range offsets)
        DescriptorHint::PerBlockWithOffset
    } else if dst_is_object && src_is_contiguous {
        // Writing to object storage (one-to-one mapping required)
        DescriptorHint::PerBlockUniqueTarget
    } else if src_is_contiguous && dst_is_contiguous {
        // Both contiguous: use batched optimization
        DescriptorHint::BatchedRanges
    } else {
        // Fall back to multi-descriptor
        DescriptorHint::Auto
    }
}
