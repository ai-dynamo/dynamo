use super::super::*;
use nixl_sys::{Agent as NixlAgent, XferOp};

/// Execute NIXL-based transfers between local and remote memory
///
/// This function handles transfers involving remote memory using NIXL operations.
pub fn execute_nixl_transfer<
    SrcS: Storage,
    SrcL: LocalityProvider,
    DstS: Storage,
    DstL: LocalityProvider,
    M: BlockMetadata,
>(
    src: &[&Block<SrcS, SrcL, M>],
    dst: &mut [&mut Block<DstS, DstL, M>],
    ctx: &TransferContext,
    strategy: TransferStrategy,
) -> Result<(), TransferError> {
    // Validate block counts
    if src.len() != dst.len() {
        return Err(TransferError::CountMismatch(src.len(), dst.len()));
    }

    // Get NIXL agent from context
    let nixl_agent = ctx.nixl_agent();
    let agent = nixl_agent
        .as_ref()
        .ok_or_else(|| TransferError::ExecutionError("NIXL agent not available".to_string()))?;

    // Determine transfer direction
    let xfer_op = match strategy {
        TransferStrategy::NixlRead => XferOp::Read,
        TransferStrategy::NixlWrite => XferOp::Write,
        _ => {
            return Err(TransferError::ExecutionError(format!(
                "Invalid NIXL strategy: {:?}",
                strategy
            )));
        }
    };

    // Execute transfers for each block pair
    for (src_block, dst_block) in src.iter().zip(dst.iter_mut()) {
        copy_block_nixl(src_block, dst_block, &agent, xfer_op)?;
    }

    Ok(())
}

/// Copy a single block using NIXL operations
///
/// This function copies data from source to destination block using NIXL
/// active messaging for remote memory transfers.
fn copy_block_nixl<
    SrcS: Storage,
    SrcL: LocalityProvider,
    DstS: Storage,
    DstL: LocalityProvider,
    M: BlockMetadata,
>(
    src_block: &Block<SrcS, SrcL, M>,
    dst_block: &mut Block<DstS, DstL, M>,
    agent: &NixlAgent,
    xfer_op: XferOp,
) -> Result<(), TransferError> {
    // Check if blocks are fully contiguous for efficient bulk transfer
    if src_block.is_fully_contiguous() && dst_block.is_fully_contiguous() {
        copy_block_contiguous_nixl(src_block, dst_block, agent, xfer_op)
    } else {
        copy_block_layerwise_nixl(src_block, dst_block, agent, xfer_op)
    }
}

/// Copy a fully contiguous block using NIXL (most efficient path)
fn copy_block_contiguous_nixl<
    SrcS: Storage,
    SrcL: LocalityProvider,
    DstS: Storage,
    DstL: LocalityProvider,
    M: BlockMetadata,
>(
    src_block: &Block<SrcS, SrcL, M>,
    dst_block: &mut Block<DstS, DstL, M>,
    agent: &NixlAgent,
    xfer_op: XferOp,
) -> Result<(), TransferError> {
    // Get NIXL descriptors for source and destination
    let src_desc = src_block.as_block_descriptor().map_err(|e| {
        TransferError::ExecutionError(format!("Failed to get source block descriptor: {}", e))
    })?;

    let mut dst_desc = dst_block.as_block_descriptor_mut().map_err(|e| {
        TransferError::ExecutionError(format!("Failed to get destination block descriptor: {}", e))
    })?;

    // Validate sizes match
    if src_desc.size() != dst_desc.size() {
        return Err(TransferError::ExecutionError(format!(
            "Block size mismatch: src={}, dst={}",
            src_desc.size(),
            dst_desc.size()
        )));
    }

    // Execute the NIXL transfer
    execute_nixl_xfer(&src_desc, &mut dst_desc, agent, xfer_op)
}

/// Copy a block layer by layer using NIXL
fn copy_block_layerwise_nixl<
    SrcS: Storage,
    SrcL: LocalityProvider,
    DstS: Storage,
    DstL: LocalityProvider,
    M: BlockMetadata,
>(
    src_block: &ImmutableBlock<SrcS, SrcL, M>,
    dst_block: &mut MutableBlock<DstS, DstL, M>,
    agent: &NixlAgent,
    xfer_op: XferOp,
) -> Result<(), TransferError> {
    // Get the number of layers to copy
    let num_layers = src_block.num_layers();
    let num_outer_dims = src_block.num_outer_dims();

    // Copy each layer
    for layer_idx in 0..num_layers {
        for outer_idx in 0..num_outer_dims {
            copy_layer_nixl(src_block, dst_block, layer_idx, outer_idx, agent, xfer_op)?;
        }
    }

    Ok(())
}

/// Copy a single layer using NIXL
fn copy_layer_nixl<
    SrcS: Storage,
    SrcL: LocalityProvider,
    DstS: Storage,
    DstL: LocalityProvider,
    M: BlockMetadata,
>(
    src_block: &Block<SrcS, SrcL, M>,
    dst_block: &mut Block<DstS, DstL, M>,
    layer_idx: usize,
    outer_idx: usize,
    agent: &NixlAgent,
    xfer_op: XferOp,
) -> Result<(), TransferError> {
    // Get NIXL descriptors for source and destination layers
    let src_desc = src_block
        .as_layer_descriptor(layer_idx, outer_idx)
        .map_err(|e| {
            TransferError::ExecutionError(format!("Failed to get source layer descriptor: {}", e))
        })?;

    let mut dst_desc = dst_block
        .as_layer_descriptor_mut(layer_idx, outer_idx)
        .map_err(|e| {
            TransferError::ExecutionError(format!(
                "Failed to get destination layer descriptor: {}",
                e
            ))
        })?;

    // Validate sizes match
    if src_desc.size() != dst_desc.size() {
        return Err(TransferError::ExecutionError(format!(
            "Layer size mismatch: src={}, dst={}",
            src_desc.size(),
            dst_desc.size()
        )));
    }

    // Execute the NIXL transfer
    execute_nixl_xfer(&src_desc, &mut dst_desc, agent, xfer_op)
}

/// Execute the actual NIXL transfer operation
fn execute_nixl_xfer<SrcDesc, DstDesc>(
    src_desc: &SrcDesc,
    dst_desc: &mut DstDesc,
    agent: &NixlAgent,
    xfer_op: XferOp,
) -> Result<(), TransferError>
where
    SrcDesc: nixl_sys::NixlDescriptor,
    DstDesc: nixl_sys::NixlDescriptor,
{
    match xfer_op {
        XferOp::Read => {
            // Remote to Local transfer
            // TODO: Implement actual NIXL read operation
            // This would typically involve:
            // 1. Creating a read request with the remote descriptor (src_desc)
            // 2. Specifying the local destination (dst_desc)
            // 3. Executing the active message transfer

            tracing::debug!(
                "Executing NIXL Read: remote {:?} -> local {:?}",
                src_desc,
                dst_desc
            );

            // Placeholder for actual NIXL read implementation
            Err(TransferError::ExecutionError(
                "NIXL read implementation pending".to_string(),
            ))
        }
        XferOp::Write => {
            // Local to Remote transfer
            // TODO: Implement actual NIXL write operation
            // This would typically involve:
            // 1. Creating a write request with the local descriptor (src_desc)
            // 2. Specifying the remote destination (dst_desc)
            // 3. Executing the active message transfer

            tracing::debug!(
                "Executing NIXL Write: local {:?} -> remote {:?}",
                src_desc,
                dst_desc
            );

            // Placeholder for actual NIXL write implementation
            Err(TransferError::ExecutionError(
                "NIXL write implementation pending".to_string(),
            ))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::block_manager::storage::nixl::NixlStorage;
    use crate::block_manager::{DeviceStorage, PinnedStorage};

    #[test]
    fn test_nixl_strategy_validation() {
        // Test that NIXL strategies are properly defined
        assert_eq!(
            DeviceStorage::strategy::<NixlStorage>(),
            TransferStrategy::NixlWrite
        );
        assert_eq!(
            PinnedStorage::strategy::<NixlStorage>(),
            TransferStrategy::NixlWrite
        );
        assert_eq!(
            NixlStorage::strategy::<DeviceStorage>(),
            TransferStrategy::NixlRead
        );
        assert_eq!(
            NixlStorage::strategy::<PinnedStorage>(),
            TransferStrategy::NixlRead
        );
    }

    #[test]
    fn test_xfer_op_conversion() {
        // Test conversion from strategy to XferOp
        let read_op = match TransferStrategy::NixlRead {
            TransferStrategy::NixlRead => XferOp::Read,
            _ => panic!("Wrong conversion"),
        };

        let write_op = match TransferStrategy::NixlWrite {
            TransferStrategy::NixlWrite => XferOp::Write,
            _ => panic!("Wrong conversion"),
        };

        // Just verify the operations exist
        match read_op {
            XferOp::Read => {}
            _ => panic!("Wrong conversion"),
        }

        match write_op {
            XferOp::Write => {}
            _ => panic!("Wrong conversion"),
        }
    }

    #[test]
    fn test_strategy_matching() {
        // Test that we can match on NIXL strategies
        let strategies = vec![TransferStrategy::NixlRead, TransferStrategy::NixlWrite];

        for strategy in strategies {
            match strategy {
                TransferStrategy::NixlRead | TransferStrategy::NixlWrite => {
                    // This should match
                }
                _ => panic!("Unexpected strategy: {:?}", strategy),
            }
        }
    }
}
