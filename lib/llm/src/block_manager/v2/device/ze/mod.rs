//! XPU (Level-Zero) backend implementation (placeholder)
//!
//! This will be fully implemented in Phase 7

use crate::block_manager::v2::device::traits::*;
use anyhow::{Result, bail};

#[derive(Debug)]
pub struct ZeContext {
    device_id: u32,
}

impl ZeContext {
    pub fn new(_device_id: u32) -> Result<Self> {
        // TODO: Implement in Phase 7
        bail!("XPU backend not yet implemented (Phase 7)")
    }
}

impl DeviceContextOps for ZeContext {
    fn device_id(&self) -> u32 {
        self.device_id
    }

    fn create_stream(&self) -> Result<Box<dyn DeviceStreamOps>> {
        bail!("Not implemented")
    }

    fn allocate_device(&self, _size: usize) -> Result<u64> {
        bail!("Not implemented")
    }

    fn free_device(&self, _ptr: u64) -> Result<()> {
        bail!("Not implemented")
    }

    fn allocate_pinned(&self, _size: usize) -> Result<u64> {
        bail!("Not implemented")
    }

    fn free_pinned(&self, _ptr: u64) -> Result<()> {
        bail!("Not implemented")
    }
}

pub fn is_available() -> bool {
    // TODO: Actually check for Level-Zero
    false
}
