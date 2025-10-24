// SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Typestate builder for NIXL transfers.
//!
//! This module provides a compile-time safe builder for NIXL transfers that ensures
//! all required parameters are set before execution.

use super::{PhysicalLayout, TransferContext, TransferStrategy};
use crate::block_manager::v2::physical::transfer::context::TransferCompleteNotification;
use anyhow::{Result, anyhow};
use nixl_sys::{XferDescList, XferOp};
use std::marker::PhantomData;
use std::ops::Range;

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

        assert!(
            nixl_agent.name() == src.nixl_metadata().agent_name(),
            "the source must be local"
        );

        // Capture NIXL metadata for both layouts
        let src_metadata = src.nixl_metadata();
        let dst_metadata = dst.nixl_metadata();

        let src_mem_type = src_metadata.mem_type();
        let dst_mem_type = dst_metadata.mem_type();

        let src_device_id = src_metadata.device_id();
        let dst_device_id = dst_metadata.device_id();

        // Build XferDescLists for source and destination
        let mut src_dl = XferDescList::new(src_mem_type)?;
        let mut dst_dl = XferDescList::new(dst_mem_type)?;

        // Add memory regions to descriptor lists
        for (&src_block_id, &dst_block_id) in src_block_ids.iter().zip(dst_block_ids.iter()) {
            for layer_id in layers.clone() {
                for outer_id in 0..src_layout.outer_dim() {
                    let src_region = src.memory_region(src_block_id, layer_id, outer_id)?;
                    let dst_region = dst.memory_region(dst_block_id, layer_id, outer_id)?;

                    if src_region.size() != dst_region.size() {
                        return Err(anyhow!(
                            "Size mismatch at block=({},{}), layer={}, outer={}: src={}, dst={}",
                            src_block_id,
                            dst_block_id,
                            layer_id,
                            outer_id,
                            src_region.size(),
                            dst_region.size()
                        ));
                    }

                    // Add to source descriptor list
                    src_dl.add_desc(src_region.addr(), src_region.size(), src_device_id)?;

                    // Add to destination descriptor list
                    dst_dl.add_desc(dst_region.addr(), dst_region.size(), dst_device_id)?;
                }
            }
        }

        // Note: Overlap detection was removed from nixl-sys 0.6.1
        // The NIXL library now handles overlap detection internally

        if matches!(
            strategy,
            TransferStrategy::NixlReadFlipped | TransferStrategy::NixlWriteFlipped
        ) {
            std::mem::swap(&mut src_dl, &mut dst_dl);
        }

        // Create transfer request
        let xfer_req = nixl_agent.create_xfer_req(
            xfer_op,
            &src_dl,
            &dst_dl,
            dst_metadata.agent_name(),
            None, // opt_args
        )?;

        // Post transfer request
        // Note: Notification handling via OptArgs can be added later if needed
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
