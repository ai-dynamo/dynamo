// pub use context::TransferContext;
// pub use coordinators::{LocalCoordinator, LogicalCoordinator, MockLogicalCoordinator};
// pub use error::TransferError;
// pub use executors::{TransferExecutor, UniversalCoordinator};
// pub use strategy::{NixlTransferDirection, SelectStrategy, TransferStrategy};

use crate::block_manager::block::transfer_v2::TransferError;
// Re-export macros for external use
pub use crate::impl_coordinator_strategies;
pub use crate::impl_local_transfers;
pub use crate::impl_write_to_blocks;
pub use crate::impl_write_to_strategy;

use super::*;
use crate::block_manager::{
    block::locality::LocalityType, layout::GenericBlockLayout, storage::SystemStorage,
    DeviceStorage, DiskStorage, PinnedStorage,
};
// Comment out Nixl-related code for now
// use nixl_sys::NixlDescriptor;
use std::sync::Arc;
use tokio::sync::oneshot;

pub enum BlockLayerDescriptor {
    Local(LocalBlockDescriptor),
    Remote(RemoteBlockDescriptor),
    Logical(LogicalBlockDescriptor),
}

pub struct LocalBlockDescriptor {
    layout: Arc<dyn GenericBlockLayout>,
    block_idx: usize,
}

pub struct RemoteBlockDescriptor {
    layout: Arc<dyn GenericBlockLayout>,
    block_idx: usize,
}

pub struct LogicalBlockDescriptor {
    block_idx: usize,
}
