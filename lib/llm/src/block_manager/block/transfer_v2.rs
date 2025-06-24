mod context;
mod coordinators;
mod error;
mod executors;
mod macros;
mod strategy;

pub use context::TransferContext;
pub use coordinators::{LocalCoordinator, LogicalCoordinator, MockLogicalCoordinator};
pub use error::TransferError;
pub use executors::{TransferExecutor, UniversalCoordinator};
pub use strategy::{NixlTransferDirection, SelectStrategy, TransferStrategy};

// Re-export macros for external use
pub use crate::impl_coordinator_strategies;
pub use crate::impl_local_transfers;
pub use crate::impl_write_to_blocks;
pub use crate::impl_write_to_strategy;

use super::*;
use crate::block_manager::{storage::SystemStorage, DeviceStorage, DiskStorage, PinnedStorage};
// Comment out Nixl-related code for now
// use nixl_sys::NixlDescriptor;
use std::sync::Arc;
use tokio::sync::oneshot;

/// Core trait for transferring data between different storage/locality combinations
///
/// This trait is the foundation for all block transfers and should be implemented
/// by coordinators that understand how to move data between specific locality/storage
/// combinations.
pub trait WriteTo<
    SrcS: Storage,
    SrcL: LocalityProvider,
    DstS: Storage,
    DstL: LocalityProvider,
    M: BlockMetadata,
>
{
    /// Transfer blocks from source to destination with optional notification
    ///
    /// # Arguments
    /// * `src` - Source blocks (immutable references)
    /// * `dst` - Destination blocks (mutable references)
    /// * `notify` - Whether to provide completion notification
    /// * `ctx` - Transfer context containing streams, agents, etc.
    ///
    /// # Returns
    /// Optional receiver for transfer completion notification
    fn write_to(
        &self,
        src: &[&Block<SrcS, SrcL, M>],
        dst: &mut [&mut Block<DstS, DstL, M>],
        notify: bool,
        ctx: Arc<TransferContext>,
    ) -> Result<Option<oneshot::Receiver<()>>, TransferError>;
}

/// Locality-aware trait for Local blocks that have direct memory access
///
/// This trait provides transfer capabilities for blocks where both source and
/// destination are in Local locality, meaning we have direct access to memory
/// pointers and can use efficient local copy operations.
pub trait LocalWriteTo<SrcS: Storage, DstS: Storage, M: BlockMetadata> {
    /// Transfer local blocks using direct memory access
    fn local_write_to(
        &self,
        src: &[&Block<SrcS, locality::Local, M>],
        dst: &mut [&mut Block<DstS, locality::Local, M>],
        notify: bool,
        ctx: Arc<TransferContext>,
    ) -> Result<Option<oneshot::Receiver<()>>, TransferError>;
}

/// Locality-aware trait for cross-locality transfers involving Logical locality
///
/// This trait handles transfers where at least one side is Logical locality,
/// requiring RPC or other remote communication mechanisms.
pub trait LogicalWriteTo<
    SrcS: Storage,
    SrcL: LocalityProvider,
    DstS: Storage,
    DstL: LocalityProvider,
    M: BlockMetadata,
>
{
    /// Transfer blocks across locality boundaries using RPC/remote mechanisms
    fn logical_write_to(
        &self,
        src: &[&Block<SrcS, SrcL, M>],
        dst: &mut [&mut Block<DstS, DstL, M>],
        notify: bool,
        ctx: Arc<TransferContext>,
    ) -> Result<Option<oneshot::Receiver<()>>, TransferError>;
}

/// Extension trait for collections of blocks to support write_to operations
///
/// This trait allows calling .write_to() directly on collections like Vec<MutableBlock>
pub trait WriteToBlocks<DstS: Storage, DstL: LocalityProvider, M: BlockMetadata> {
    fn write_to_blocks(
        &self,
        dst: &mut [&mut Block<DstS, DstL, M>],
        notify: bool,
        ctx: Arc<TransferContext>,
    ) -> Result<Option<oneshot::Receiver<()>>, TransferError>;
}

/// Convenient extension trait for collections to support direct write_to operations
///
/// This provides the .write_to() method that users expect on block collections
pub trait WriteToCollection<SrcS, SrcL, M>
where
    SrcS: Storage, // Comment out NixlDescriptor requirement for now
    // SrcS: Storage + NixlDescriptor,
    SrcL: LocalityProvider,
    M: BlockMetadata,
{
    /// Write this collection to another collection of blocks
    fn write_to<DstS, DstL>(
        &self,
        dst: &mut [MutableBlock<DstS, DstL, M>],
        notify: bool,
        ctx: Arc<TransferContext>,
    ) -> Result<Option<oneshot::Receiver<()>>, TransferError>
    where
        DstS: Storage, // Comment out NixlDescriptor requirement for now
        // DstS: Storage + NixlDescriptor,
        DstL: LocalityProvider,
        SrcS: SelectStrategy<DstS>,
        SrcL: locality::LocalityProvider,
        DstL: locality::LocalityProvider;

    /// Convenience method with default notify=false and a basic context
    fn write_to_simple<DstS, DstL>(
        &self,
        dst: &mut [MutableBlock<DstS, DstL, M>],
    ) -> Result<(), TransferError>
    where
        DstS: Storage,
        DstL: LocalityProvider,
        SrcS: SelectStrategy<DstS>,
        SrcL: locality::LocalityProvider,
        DstL: locality::LocalityProvider,
    {
        // Create a basic transfer context without CUDA or NIXL specifics.
        let rt_handle = tokio::runtime::Handle::current();
        let ctx = TransferContext::new(rt_handle);

        self.write_to(dst, false, Arc::new(ctx)).map(|_| ())
    }
}

/// Implementation for Vec<MutableBlock> - the main use case
impl<SrcS, SrcL, M> WriteToCollection<SrcS, SrcL, M> for Vec<MutableBlock<SrcS, SrcL, M>>
where
    SrcS: Storage, // Comment out NixlDescriptor requirement for now
    // SrcS: Storage + NixlDescriptor,
    SrcL: LocalityProvider,
    M: BlockMetadata,
{
    fn write_to<DstS, DstL>(
        &self,
        dst: &mut [MutableBlock<DstS, DstL, M>],
        notify: bool,
        ctx: Arc<TransferContext>,
    ) -> Result<Option<oneshot::Receiver<()>>, TransferError>
    where
        DstS: Storage, // Comment out NixlDescriptor requirement for now
        // DstS: Storage + NixlDescriptor,
        DstL: LocalityProvider,
        SrcS: SelectStrategy<DstS>,
        SrcL: locality::LocalityProvider,
        DstL: locality::LocalityProvider,
    {
        // Convert collections to slice format that coordinators expect
        let src_refs = helpers::mutable_blocks_to_refs(self);
        let mut dst_refs = helpers::mutable_blocks_to_mut_refs(dst);

        // Use the UniversalCoordinator which can handle any locality combination
        let strategy = <SrcS as SelectStrategy<DstS>>::strategy();

        // Execute the transfer
        UniversalCoordinator::execute_transfer(&src_refs, &mut dst_refs, &ctx, strategy)?;

        // Handle notification
        if notify {
            let (tx, rx) = oneshot::channel();
            let _ = tx.send(());
            Ok(Some(rx))
        } else {
            Ok(None)
        }
    }

    fn write_to_simple<DstS, DstL>(
        &self,
        dst: &mut [MutableBlock<DstS, DstL, M>],
    ) -> Result<(), TransferError>
    where
        DstS: Storage,
        DstL: LocalityProvider,
        SrcS: SelectStrategy<DstS>,
        SrcL: locality::LocalityProvider,
        DstL: locality::LocalityProvider,
    {
        // Create a basic transfer context without CUDA or NIXL specifics.
        let rt_handle = tokio::runtime::Handle::current();
        let ctx = TransferContext::new(rt_handle);

        self.write_to(dst, false, Arc::new(ctx)).map(|_| ())
    }
}

/// Implementation for slice of MutableBlock
impl<SrcS, SrcL, M> WriteToCollection<SrcS, SrcL, M> for [MutableBlock<SrcS, SrcL, M>]
where
    SrcS: Storage, // Comment out NixlDescriptor requirement for now
    // SrcS: Storage + NixlDescriptor,
    SrcL: LocalityProvider,
    M: BlockMetadata,
{
    fn write_to<DstS, DstL>(
        &self,
        dst: &mut [MutableBlock<DstS, DstL, M>],
        notify: bool,
        ctx: Arc<TransferContext>,
    ) -> Result<Option<oneshot::Receiver<()>>, TransferError>
    where
        DstS: Storage, // Comment out NixlDescriptor requirement for now
        // DstS: Storage + NixlDescriptor,
        DstL: LocalityProvider,
        SrcS: SelectStrategy<DstS>,
        SrcL: locality::LocalityProvider,
        DstL: locality::LocalityProvider,
    {
        // Convert collections to slice format that coordinators expect
        let src_refs = helpers::mutable_blocks_to_refs(self);
        let mut dst_refs = helpers::mutable_blocks_to_mut_refs(dst);

        // Use the UniversalCoordinator which can handle any locality combination
        let strategy = <SrcS as SelectStrategy<DstS>>::strategy();

        // Execute the transfer
        UniversalCoordinator::execute_transfer(&src_refs, &mut dst_refs, &ctx, strategy)?;

        // Handle notification
        if notify {
            let (tx, rx) = oneshot::channel();
            let _ = tx.send(());
            Ok(Some(rx))
        } else {
            Ok(None)
        }
    }

    fn write_to_simple<DstS, DstL>(
        &self,
        dst: &mut [MutableBlock<DstS, DstL, M>],
    ) -> Result<(), TransferError>
    where
        DstS: Storage,
        DstL: LocalityProvider,
        SrcS: SelectStrategy<DstS>,
        SrcL: locality::LocalityProvider,
        DstL: locality::LocalityProvider,
    {
        // Create a basic transfer context without CUDA or NIXL specifics.
        let rt_handle = tokio::runtime::Handle::current();
        let ctx = TransferContext::new(rt_handle);

        self.write_to(dst, false, Arc::new(ctx)).map(|_| ())
    }
}

/// Transfer priority for ordering operations
#[derive(Debug, Clone, Copy)]
pub enum TransferPriority {
    Low,
    Normal,
    High,
}

/// Helper functions to convert between block collections and slice formats
pub mod helpers {
    use super::*;

    /// Convert a slice of ImmutableBlocks to a Vec<&Block> for use with WriteTo trait
    pub fn immutable_blocks_to_refs<S: Storage, L: LocalityProvider, M: BlockMetadata>(
        blocks: &[ImmutableBlock<S, L, M>],
    ) -> Vec<&Block<S, L, M>> {
        blocks.iter().map(|block| block.deref()).collect()
    }

    /// Convert a mutable slice of MutableBlocks to a Vec<&mut Block> for use with WriteTo trait
    pub fn mutable_blocks_to_mut_refs<S: Storage, L: LocalityProvider, M: BlockMetadata>(
        blocks: &mut [MutableBlock<S, L, M>],
    ) -> Vec<&mut Block<S, L, M>> {
        blocks.iter_mut().map(|block| block.deref_mut()).collect()
    }

    /// Convert a slice of MutableBlocks to a Vec<&Block> for read-only access
    pub fn mutable_blocks_to_refs<S: Storage, L: LocalityProvider, M: BlockMetadata>(
        blocks: &[MutableBlock<S, L, M>],
    ) -> Vec<&Block<S, L, M>> {
        blocks.iter().map(|block| block.deref()).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_transfer_priority_enum() {
        let _low = TransferPriority::Low;
        let _normal = TransferPriority::Normal;
        let _high = TransferPriority::High;
        // Just testing that the enum variants exist
    }

    #[test]
    fn test_universal_coordinator_exists() {
        let _coordinator = UniversalCoordinator::default();
        // Basic sanity test that the universal coordinator can be created
    }

    #[test]
    fn test_strategy_selection() {
        // Test that we can select strategies at compile time
        let _strategy = <DeviceStorage as SelectStrategy<PinnedStorage>>::strategy();
        assert_eq!(_strategy, TransferStrategy::CudaAsyncD2H);
    }
}
