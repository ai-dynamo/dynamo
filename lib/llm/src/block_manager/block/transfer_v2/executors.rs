use super::*;
use std::sync::Arc;

// Re-export specific executor implementations
pub mod cuda;
pub mod memcpy;
// Comment out Nixl-related code for now
/*
pub mod nixl;
*/

// Re-export key types for convenience
pub use cuda::*;
pub use memcpy::*;
// Comment out Nixl-related code for now
/*
pub use nixl::*;
*/

/// Transfer execution trait for different executors
pub trait TransferExecutor {
    /// Execute a transfer with the given strategy
    fn execute<
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
    ) -> Result<(), TransferError>;
}

/// Universal coordinator that can handle any transfer by dispatching to appropriate coordinators
///
/// This coordinator examines the locality types and delegates to either LocalCoordinator
/// or LogicalCoordinator as appropriate.
#[derive(Default)]
pub struct UniversalCoordinator;

impl UniversalCoordinator {
    /// Determine the appropriate transfer strategy for the given storage types
    pub fn determine_strategy<SrcS: Storage, DstS: Storage>() -> TransferStrategy
    where
        SrcS: SelectStrategy<DstS>,
    {
        SrcS::strategy()
    }

    /// Execute a transfer by dispatching to the appropriate locality-aware coordinator
    ///
    /// This method examines the locality types and routes to:
    /// - LocalCoordinator for Local-to-Local transfers
    /// - LogicalCoordinator for any transfers involving Logical locality
    pub fn execute_transfer<
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
    ) -> Result<(), TransferError>
    where
        SrcS: SelectStrategy<DstS>,
    {
        // Check if this is a Local-to-Local transfer
        if Self::is_local_to_local::<SrcL, DstL>() {
            // For Local-to-Local, we can use specific executors directly
            Self::execute_local_transfer(src, dst, ctx, strategy)
        } else {
            // For any transfer involving Logical locality, delegate to LogicalCoordinator
            Self::execute_logical_transfer(src, dst, ctx, strategy)
        }
    }

    /// Check if both source and destination are Local locality
    fn is_local_to_local<SrcL: LocalityProvider, DstL: LocalityProvider>() -> bool {
        // Use type IDs to check if both are Local
        std::any::TypeId::of::<SrcL>() == std::any::TypeId::of::<locality::Local>()
            && std::any::TypeId::of::<DstL>() == std::any::TypeId::of::<locality::Local>()
    }

    /// Execute local transfers by delegating to LocalCoordinator
    fn execute_local_transfer<
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
    ) -> Result<(), TransferError>
    where
        SrcS: SelectStrategy<DstS>,
    {
        // For local transfers, delegate to LocalCoordinator
        // This avoids the generic constraint issues by using the specialized coordinator
        tracing::debug!("Delegating local transfer to LocalCoordinator");

        // TODO: This requires casting to Local locality types, which should be safe
        // since we've verified this is a Local-to-Local transfer
        // For now, return an error directing users to use LocalCoordinator directly
        Err(TransferError::ExecutionError(
            "Use LocalCoordinator directly for Local-to-Local transfers".to_string(),
        ))
    }

    /// Execute logical transfers by delegating to LogicalCoordinator
    fn execute_logical_transfer<
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
    ) -> Result<(), TransferError>
    where
        SrcS: SelectStrategy<DstS>,
    {
        // For logical transfers, delegate to LogicalCoordinator
        tracing::debug!("Delegating logical transfer to LogicalCoordinator");

        // TODO: Similar to local transfers, this should delegate to LogicalCoordinator
        // For now, return an error directing users to use LogicalCoordinator directly
        Err(TransferError::ExecutionError(
            "Use LogicalCoordinator directly for cross-locality transfers".to_string(),
        ))
    }
}

/// Implement WriteTo for UniversalCoordinator for backwards compatibility
/// This provides a general-purpose transfer interface that can handle any locality combination
impl<
        SrcS: Storage,
        SrcL: LocalityProvider,
        DstS: Storage,
        DstL: LocalityProvider,
        M: BlockMetadata,
    > WriteTo<SrcS, SrcL, DstS, DstL, M> for UniversalCoordinator
where
    SrcS: SelectStrategy<DstS>,
{
    fn write_to(
        &self,
        src: &[&Block<SrcS, SrcL, M>],
        dst: &mut [&mut Block<DstS, DstL, M>],
        notify: bool,
        ctx: Arc<TransferContext>,
    ) -> Result<Option<oneshot::Receiver<()>>, TransferError> {
        if src.len() != dst.len() {
            return Err(TransferError::CountMismatch(src.len(), dst.len()));
        }

        // Determine strategy and execute transfer
        let strategy = Self::determine_strategy::<SrcS, DstS>();
        Self::execute_transfer(src, dst, &ctx, strategy)?;

        // Handle notification
        if notify {
            let (tx, rx) = oneshot::channel();
            let _ = tx.send(());
            Ok(Some(rx))
        } else {
            Ok(None)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::block_manager::{DeviceStorage, PinnedStorage};

    #[test]
    fn test_strategy_determination() {
        let strategy = UniversalCoordinator::determine_strategy::<DeviceStorage, PinnedStorage>();
        assert_eq!(strategy, TransferStrategy::CudaAsyncD2H);
    }

    #[test]
    fn test_universal_coordinator_creation() {
        let _coordinator = UniversalCoordinator::default();
    }

    #[test]
    fn test_locality_detection() {
        // Test the locality detection logic
        let is_local =
            UniversalCoordinator::is_local_to_local::<locality::Local, locality::Local>();
        assert!(is_local);
    }
}
