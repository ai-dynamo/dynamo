use super::*;
use crate::block_manager::block::{locality, BlockDataExt};
use nixl_sys::NixlDescriptor;
use std::any::TypeId;

/// Local transfer coordinator for blocks within the same locality
///
/// This coordinator handles transfers where both source and destination blocks
/// are in Local locality, allowing for direct memory access operations like
/// memcpy, CUDA transfers, etc.
#[derive(Default)]
pub struct LocalCoordinator;

impl LocalCoordinator {
    /// Create a new LocalCoordinator
    pub fn new() -> Self {
        Self::default()
    }
}

/// Mock logical transfer coordinator for testing transfer patterns
///
/// This coordinator handles transfers between MockLogical blocks by counting
/// bytes without actually moving data. Useful for testing transfer logic
/// without needing actual RPC infrastructure.
#[derive(Default)]
pub struct MockLogicalCoordinator;

impl MockLogicalCoordinator {
    /// Create a new MockLogicalCoordinator
    pub fn new() -> Self {
        Self::default()
    }
}

/// Logical transfer coordinator for cross-locality transfers
///
/// This coordinator handles transfers involving Logical locality blocks,
/// typically using RPC mechanisms like NIXL for remote transfers.
#[derive(Default)]
pub struct LogicalCoordinator;

impl LogicalCoordinator {
    /// Create a new LogicalCoordinator
    pub fn new() -> Self {
        Self::default()
    }
}

// ===== Local Coordinator Implementations =====

/// LocalCoordinator implements LocalWriteTo for all Local-to-Local transfers
impl<SrcS: Storage, DstS: Storage, M: BlockMetadata> LocalWriteTo<SrcS, DstS, M>
    for LocalCoordinator
where
    SrcS: SelectStrategy<DstS> + NixlDescriptor,
    DstS: NixlDescriptor,
    locality::LocalBlockData<SrcS>: BlockDataExt<SrcS>,
    locality::LocalBlockData<DstS>: BlockDataExt<DstS>,
{
    fn local_write_to(
        &self,
        src: &[&Block<SrcS, locality::Local, M>],
        dst: &mut [&mut Block<DstS, locality::Local, M>],
        notify: bool,
        ctx: Arc<TransferContext>,
    ) -> Result<Option<oneshot::Receiver<()>>, TransferError> {
        // Validate input
        if src.len() != dst.len() {
            return Err(TransferError::CountMismatch(src.len(), dst.len()));
        }

        // Determine strategy and dispatch to appropriate local executor
        let strategy = <SrcS as SelectStrategy<DstS>>::strategy();
        self.execute_local_transfer(src, dst, &ctx, strategy)?;

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

impl LocalCoordinator {
    /// Execute local transfers using the appropriate executor based on strategy
    /// This method ensures we only call locality-specific executors
    fn execute_local_transfer<SrcS: Storage, DstS: Storage, M: BlockMetadata>(
        &self,
        src: &[&Block<SrcS, locality::Local, M>],
        dst: &mut [&mut Block<DstS, locality::Local, M>],
        ctx: &TransferContext,
        strategy: TransferStrategy,
    ) -> Result<(), TransferError>
    where
        SrcS: SelectStrategy<DstS> + NixlDescriptor,
        DstS: NixlDescriptor,
        locality::LocalBlockData<SrcS>: BlockDataExt<SrcS>,
        locality::LocalBlockData<DstS>: BlockDataExt<DstS>,
    {
        match strategy {
            TransferStrategy::Memcpy => {
                // For memcpy, we can call the executor directly since both are Local
                super::executors::memcpy::execute_memcpy_transfer(src, dst, ctx)
            }
            TransferStrategy::CudaAsyncH2D
            | TransferStrategy::CudaAsyncD2H
            | TransferStrategy::CudaAsyncD2D
            | TransferStrategy::CudaBlockingH2D
            | TransferStrategy::CudaBlockingD2H => {
                // For CUDA transfers, delegate to CUDA executor
                // TODO: Implement locality-aware CUDA executor
                tracing::warn!("CUDA transfers not yet implemented for Local coordinator");
                Err(TransferError::ExecutionError(
                    "CUDA transfers not yet implemented".to_string(),
                ))
            }
            TransferStrategy::NixlRead | TransferStrategy::NixlWrite => {
                // NIXL transfers shouldn't happen between two Local blocks
                Err(TransferError::IncompatibleTypes(
                    "NIXL transfers not supported between Local blocks".to_string(),
                ))
            }
            TransferStrategy::Invalid => Err(TransferError::IncompatibleTypes(
                "Invalid transfer strategy for Local blocks".to_string(),
            )),
        }
    }
}

/// LocalCoordinator also implements the general WriteTo trait for Local-to-Local transfers
impl<SrcS: Storage, DstS: Storage, M: BlockMetadata>
    WriteTo<SrcS, locality::Local, DstS, locality::Local, M> for LocalCoordinator
where
    SrcS: SelectStrategy<DstS> + NixlDescriptor,
    DstS: NixlDescriptor,
    locality::LocalBlockData<SrcS>: BlockDataExt<SrcS>,
    locality::LocalBlockData<DstS>: BlockDataExt<DstS>,
{
    fn write_to(
        &self,
        src: &[&Block<SrcS, locality::Local, M>],
        dst: &mut [&mut Block<DstS, locality::Local, M>],
        notify: bool,
        ctx: Arc<TransferContext>,
    ) -> Result<Option<oneshot::Receiver<()>>, TransferError> {
        self.local_write_to(src, dst, notify, ctx)
    }
}

// ===== MockLogical Coordinator Implementations =====

/// MockLogicalCoordinator handles transfers between MockLogical blocks
/// It only counts bytes without moving actual data
impl<SrcS: Storage, DstS: Storage, M: BlockMetadata>
    WriteTo<SrcS, locality::MockLogical, DstS, locality::MockLogical, M> for MockLogicalCoordinator
where
    SrcS: SelectStrategy<DstS>,
{
    fn write_to(
        &self,
        src: &[&Block<SrcS, locality::MockLogical, M>],
        dst: &mut [&mut Block<DstS, locality::MockLogical, M>],
        notify: bool,
        ctx: Arc<TransferContext>,
    ) -> Result<Option<oneshot::Receiver<()>>, TransferError> {
        // Validate input
        if src.len() != dst.len() {
            return Err(TransferError::CountMismatch(src.len(), dst.len()));
        }

        // Compute total bytes to transfer (mock operation)
        let mut total_bytes = 0;
        for (src_block, _dst_block) in src.iter().zip(dst.iter()) {
            // Access the MockLogicalBlockData to compute size
            let src_data = &src_block.data;

            // Compute the total size based on block parameters
            let block_size = src_data.compute_total_size();
            total_bytes += block_size;

            tracing::debug!(
                "MockLogical transfer: block size = {} bytes ({}x{}x{}x{} elements)",
                block_size,
                src_data.num_layers(),
                src_data.outer_dim(),
                src_data.page_size(),
                src_data.inner_dim()
            );
        }

        tracing::info!(
            "MockLogical transfer completed: {} blocks, {} total bytes",
            src.len(),
            total_bytes
        );

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

// ===== Logical Coordinator Implementations =====

/// LogicalCoordinator handles transfers involving Logical localities
/// For now, this is a placeholder that delegates to UniversalCoordinator
impl<
        SrcS: Storage,
        SrcL: LocalityProvider,
        DstS: Storage,
        DstL: LocalityProvider,
        M: BlockMetadata,
    > LogicalWriteTo<SrcS, SrcL, DstS, DstL, M> for LogicalCoordinator
where
    SrcS: SelectStrategy<DstS>,
{
    fn logical_write_to(
        &self,
        src: &[&Block<SrcS, SrcL, M>],
        dst: &mut [&mut Block<DstS, DstL, M>],
        notify: bool,
        ctx: Arc<TransferContext>,
    ) -> Result<Option<oneshot::Receiver<()>>, TransferError> {
        // Validate input
        if src.len() != dst.len() {
            return Err(TransferError::CountMismatch(src.len(), dst.len()));
        }

        // For now, delegate to UniversalCoordinator
        // TODO: Implement proper logical locality handling with RPC
        let strategy = <SrcS as SelectStrategy<DstS>>::strategy();
        UniversalCoordinator::execute_transfer(src, dst, &ctx, strategy)?;

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

/// LogicalCoordinator also implements the general WriteTo trait
impl<
        SrcS: Storage,
        SrcL: LocalityProvider,
        DstS: Storage,
        DstL: LocalityProvider,
        M: BlockMetadata,
    > WriteTo<SrcS, SrcL, DstS, DstL, M> for LogicalCoordinator
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
        self.logical_write_to(src, dst, notify, ctx)
    }
}

/// Check if two locality types are compatible for transfers
pub fn check_locality_compatibility<SrcL: LocalityProvider, DstL: LocalityProvider>() -> bool {
    // For now, we support transfers within the same locality
    if std::any::TypeId::of::<SrcL>() == std::any::TypeId::of::<DstL>() {
        return true;
    }

    // Future: Add cross-locality transfer support here
    // Example: Local -> Logical via RPC, etc.

    false
}

/// Get the appropriate transfer mechanism for two locality types
pub fn get_transfer_mechanism<SrcL: LocalityProvider, DstL: LocalityProvider>(
) -> locality::TransferMechanism {
    use crate::block_manager::block::locality;

    let src_name = std::any::type_name::<SrcL>();
    let dst_name = std::any::type_name::<DstL>();

    // Same locality transfers
    if src_name == dst_name {
        if src_name.contains("Local") {
            return locality::TransferMechanism::DirectMemory;
        } else if src_name.contains("MockLogical") {
            return locality::TransferMechanism::MockCounting;
        } else if src_name.contains("Logical") {
            return locality::TransferMechanism::RemoteRpc;
        }
    }

    // Future: Cross-locality transfers
    // Local -> Logical: RemoteRpc
    // Logical -> Local: RemoteRpc
    // etc.

    locality::TransferMechanism::Unsupported
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::block_manager::{DeviceStorage, PinnedStorage};

    #[test]
    fn test_local_coordinator_creation() {
        let _coordinator = LocalCoordinator::new();
        let _coordinator_default = LocalCoordinator::default();
    }

    #[test]
    fn test_mock_logical_coordinator_creation() {
        let _coordinator = MockLogicalCoordinator::new();
        let _coordinator_default = MockLogicalCoordinator::default();
    }

    #[test]
    fn test_logical_coordinator_creation() {
        let _coordinator = LogicalCoordinator::new();
        let _coordinator_default = LogicalCoordinator::default();
    }

    #[test]
    fn test_strategy_selection_local() {
        // Test that LocalCoordinator can determine strategies
        let _strategy = <DeviceStorage as SelectStrategy<PinnedStorage>>::strategy();
        assert_eq!(_strategy, TransferStrategy::CudaAsyncD2H);
    }

    #[test]
    fn test_locality_compatibility() {
        // Test Local-Local compatibility
        let local_compat = check_locality_compatibility::<locality::Local, locality::Local>();
        assert!(local_compat);

        // Test MockLogical-MockLogical compatibility
        let mock_compat =
            check_locality_compatibility::<locality::MockLogical, locality::MockLogical>();
        assert!(mock_compat);

        // Test Local-MockLogical incompatibility
        let cross_compat = check_locality_compatibility::<locality::Local, locality::MockLogical>();
        assert!(!cross_compat);
    }

    #[test]
    fn test_transfer_mechanisms() {
        // Test Local transfer mechanism
        let local_mechanism = get_transfer_mechanism::<locality::Local, locality::Local>();
        assert_eq!(local_mechanism, locality::TransferMechanism::DirectMemory);

        // Test MockLogical transfer mechanism
        let mock_mechanism =
            get_transfer_mechanism::<locality::MockLogical, locality::MockLogical>();
        assert_eq!(mock_mechanism, locality::TransferMechanism::MockCounting);

        // Test unsupported transfer
        let unsupported_mechanism =
            get_transfer_mechanism::<locality::Local, locality::MockLogical>();
        assert_eq!(
            unsupported_mechanism,
            locality::TransferMechanism::Unsupported
        );
    }
}
