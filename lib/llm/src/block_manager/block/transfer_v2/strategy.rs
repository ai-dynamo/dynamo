use crate::block_manager::storage::nixl::NixlStorage;
use crate::block_manager::{DeviceStorage, DiskStorage, PinnedStorage};

/// Available transfer strategies for different storage combinations
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TransferStrategy {
    /// Standard memcpy for system memory transfers
    Memcpy,

    /// CUDA asynchronous host-to-device transfer
    CudaAsyncH2D,

    /// CUDA asynchronous device-to-host transfer
    CudaAsyncD2H,

    /// CUDA asynchronous device-to-device transfer
    CudaAsyncD2D,

    /// CUDA blocking host-to-device transfer
    CudaBlockingH2D,

    /// CUDA blocking device-to-host transfer
    CudaBlockingD2H,

    /// NIXL read operation (remote to local)
    NixlRead,

    /// NIXL write operation (local to remote)
    NixlWrite,

    /// Invalid/unsupported combination
    Invalid,
}

/// Trait for storage types to declare what strategy they use when transferring to another storage type
pub trait SelectStrategy<Dst> {
    fn strategy() -> TransferStrategy;
}

/// Trait for determining NIXL transfer direction
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NixlTransferDirection {
    Read,
    Write,
}

impl NixlTransferDirection {
    pub fn as_xfer_op(&self) -> nixl_sys::XferOp {
        match self {
            NixlTransferDirection::Read => nixl_sys::XferOp::Read,
            NixlTransferDirection::Write => nixl_sys::XferOp::Write,
        }
    }
}

/// Default implementations for common storage combinations

// Device -> Pinned (GPU to Host)
impl SelectStrategy<PinnedStorage> for DeviceStorage {
    fn strategy() -> TransferStrategy {
        TransferStrategy::CudaAsyncD2H
    }
}

// Pinned -> Device (Host to GPU)
impl SelectStrategy<DeviceStorage> for PinnedStorage {
    fn strategy() -> TransferStrategy {
        TransferStrategy::CudaAsyncH2D
    }
}

// Device -> Device (GPU to GPU)
impl SelectStrategy<DeviceStorage> for DeviceStorage {
    fn strategy() -> TransferStrategy {
        TransferStrategy::CudaAsyncD2D
    }
}

// System memory transfers
impl SelectStrategy<DiskStorage> for DiskStorage {
    fn strategy() -> TransferStrategy {
        TransferStrategy::Memcpy
    }
}

impl SelectStrategy<PinnedStorage> for DiskStorage {
    fn strategy() -> TransferStrategy {
        TransferStrategy::Memcpy
    }
}

impl SelectStrategy<DiskStorage> for PinnedStorage {
    fn strategy() -> TransferStrategy {
        TransferStrategy::Memcpy
    }
}

// NIXL transfers (local -> remote)
impl SelectStrategy<NixlStorage> for DeviceStorage {
    fn strategy() -> TransferStrategy {
        TransferStrategy::NixlWrite
    }
}

impl SelectStrategy<NixlStorage> for PinnedStorage {
    fn strategy() -> TransferStrategy {
        TransferStrategy::NixlWrite
    }
}

impl SelectStrategy<NixlStorage> for DiskStorage {
    fn strategy() -> TransferStrategy {
        TransferStrategy::NixlWrite
    }
}

// NIXL transfers (remote -> local)
impl SelectStrategy<DeviceStorage> for NixlStorage {
    fn strategy() -> TransferStrategy {
        TransferStrategy::NixlRead
    }
}

impl SelectStrategy<PinnedStorage> for NixlStorage {
    fn strategy() -> TransferStrategy {
        TransferStrategy::NixlRead
    }
}

impl SelectStrategy<DiskStorage> for NixlStorage {
    fn strategy() -> TransferStrategy {
        TransferStrategy::NixlRead
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cuda_strategies() {
        assert_eq!(
            <DeviceStorage as SelectStrategy<PinnedStorage>>::strategy(),
            TransferStrategy::CudaAsyncD2H
        );
        assert_eq!(
            <PinnedStorage as SelectStrategy<DeviceStorage>>::strategy(),
            TransferStrategy::CudaAsyncH2D
        );
        assert_eq!(
            <DeviceStorage as SelectStrategy<DeviceStorage>>::strategy(),
            TransferStrategy::CudaAsyncD2D
        );
    }

    #[test]
    fn test_memcpy_strategies() {
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

    #[test]
    fn test_nixl_strategies() {
        assert_eq!(
            <DeviceStorage as SelectStrategy<NixlStorage>>::strategy(),
            TransferStrategy::NixlWrite
        );
        assert_eq!(
            <NixlStorage as SelectStrategy<DeviceStorage>>::strategy(),
            TransferStrategy::NixlRead
        );
        assert_eq!(
            <PinnedStorage as SelectStrategy<NixlStorage>>::strategy(),
            TransferStrategy::NixlWrite
        );
        assert_eq!(
            <NixlStorage as SelectStrategy<PinnedStorage>>::strategy(),
            TransferStrategy::NixlRead
        );
    }
}
