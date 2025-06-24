use super::super::strategy::SelectStrategy;
use super::super::*;
use crate::block_manager::block::{locality, BlockDataExt};
use nixl_sys::NixlDescriptor;

/// Execute memcpy-based transfers for system memory storage types
///
/// This function handles transfers between system memory storage types using
/// standard memory copy operations. Both source and destination must be Local
/// locality blocks since memcpy requires direct memory access.
pub fn execute_memcpy_transfer<
    SrcS: Storage + NixlDescriptor,
    DstS: Storage + NixlDescriptor,
    M: BlockMetadata,
>(
    src: &[&Block<SrcS, locality::Local, M>],
    dst: &mut [&mut Block<DstS, locality::Local, M>],
    _ctx: &TransferContext,
) -> Result<(), TransferError>
where
    locality::LocalBlockData<SrcS>: BlockDataExt<SrcS>,
    locality::LocalBlockData<DstS>: BlockDataExt<DstS>,
{
    // Validate block counts
    if src.len() != dst.len() {
        return Err(TransferError::CountMismatch(src.len(), dst.len()));
    }

    // Execute transfers for each block pair
    for (src_block, dst_block) in src.iter().zip(dst.iter_mut()) {
        copy_block_memcpy(src_block, dst_block)?;
    }

    Ok(())
}

/// Copy a single block using memcpy operations
///
/// This function copies data from source to destination block by iterating through
/// all layers and performing memory copies for each layer.
fn copy_block_memcpy<
    SrcS: Storage + NixlDescriptor,
    DstS: Storage + NixlDescriptor,
    M: BlockMetadata,
>(
    src_block: &Block<SrcS, locality::Local, M>,
    dst_block: &mut Block<DstS, locality::Local, M>,
) -> Result<(), TransferError>
where
    locality::LocalBlockData<SrcS>: BlockDataExt<SrcS>,
    locality::LocalBlockData<DstS>: BlockDataExt<DstS>,
{
    // Get the number of layers to copy
    let num_layers = src_block.num_layers();
    let num_outer_dims = src_block.num_outer_dims();

    // Copy each layer
    for layer_idx in 0..num_layers {
        for outer_idx in 0..num_outer_dims {
            copy_layer_memcpy(src_block, dst_block, layer_idx, outer_idx)?;
        }
    }

    Ok(())
}

/// Copy a single layer using memcpy
///
/// This function gets memory views for the source and destination layer
/// and performs a memory copy operation.
fn copy_layer_memcpy<
    SrcS: Storage + NixlDescriptor,
    DstS: Storage + NixlDescriptor,
    M: BlockMetadata,
>(
    src_block: &Block<SrcS, locality::Local, M>,
    dst_block: &mut Block<DstS, locality::Local, M>,
    layer_idx: usize,
    outer_idx: usize,
) -> Result<(), TransferError>
where
    locality::LocalBlockData<SrcS>: BlockDataExt<SrcS>,
    locality::LocalBlockData<DstS>: BlockDataExt<DstS>,
{
    // Get layer views for source and destination
    let src_view = src_block.layer_view(layer_idx, outer_idx).map_err(|e| {
        TransferError::ExecutionError(format!("Failed to get source layer view: {}", e))
    })?;

    let mut dst_view = dst_block
        .layer_view_mut(layer_idx, outer_idx)
        .map_err(|e| {
            TransferError::ExecutionError(format!("Failed to get destination layer view: {}", e))
        })?;

    // Validate sizes match
    if src_view.size() != dst_view.size() {
        return Err(TransferError::ExecutionError(format!(
            "Layer size mismatch: src={}, dst={}",
            src_view.size(),
            dst_view.size()
        )));
    }

    // Perform the memory copy
    unsafe {
        std::ptr::copy_nonoverlapping(src_view.as_ptr(), dst_view.as_mut_ptr(), src_view.size());
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::block_manager::block::BasicMetadata;
    use crate::block_manager::{storage::SystemStorage, DiskStorage, PinnedStorage};

    #[test]
    fn test_memcpy_strategy_validation() {
        // Test that memcpy strategies are properly defined
        assert_eq!(
            <DiskStorage as SelectStrategy<DiskStorage>>::strategy(),
            TransferStrategy::Memcpy
        );
        assert_eq!(
            <DiskStorage as SelectStrategy<PinnedStorage>>::strategy(),
            TransferStrategy::Memcpy
        );
        assert_eq!(
            <PinnedStorage as SelectStrategy<DiskStorage>>::strategy(),
            TransferStrategy::Memcpy
        );
    }

    #[tokio::test]
    async fn test_count_mismatch_error() {
        // Create mock empty vectors - these should be Local blocks
        let src: Vec<&Block<SystemStorage, locality::Local, BasicMetadata>> = vec![];
        let mut dst: Vec<&mut Block<SystemStorage, locality::Local, BasicMetadata>> = vec![];

        // Create a mock transfer context (we won't actually use it)
        let rt_handle = tokio::runtime::Handle::current();
        let ctx = TransferContext::new(rt_handle);

        // Test with matching empty sizes
        let result = execute_memcpy_transfer(&src, &mut dst, &ctx);
        // We expect this to work since both are empty
        assert!(result.is_ok());
    }
}
