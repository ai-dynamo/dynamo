// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! NCCL bootstrap for creating dedicated KVBM communicators.
//!
//! This module provides infrastructure for bootstrapping NCCL communicators
//! that are dedicated to KVBM operations, separate from other runtime comms.
//!
//! The bootstrap pattern:
//! 1. Rank 0 generates a unique NCCL ID via `ncclGetUniqueId`
//! 2. The unique ID is broadcast to all ranks (via MPI or other mechanism)
//! 3. All ranks collectively call `ncclCommInitRank` to create the communicator

use anyhow::{Context, Result};
use cudarc::nccl::sys::{
    ncclComm_t, ncclCommDestroy, ncclCommInitRankConfig, ncclConfig_t, ncclGetUniqueId,
    ncclGetVersion, ncclResult_t, ncclUniqueId,
};

/// Check NCCL result and convert to anyhow::Result
fn check_nccl_result(result: ncclResult_t, operation: &str) -> Result<()> {
    if result == ncclResult_t::ncclSuccess {
        Ok(())
    } else {
        // Provide detailed error information for debugging
        let error_name = match result {
            ncclResult_t::ncclUnhandledCudaError => "ncclUnhandledCudaError",
            ncclResult_t::ncclSystemError => "ncclSystemError",
            ncclResult_t::ncclInternalError => "ncclInternalError",
            ncclResult_t::ncclInvalidArgument => "ncclInvalidArgument",
            ncclResult_t::ncclInvalidUsage => "ncclInvalidUsage",
            ncclResult_t::ncclRemoteError => "ncclRemoteError",
            ncclResult_t::ncclInProgress => "ncclInProgress",
            _ => "Unknown",
        };
        anyhow::bail!(
            "{} failed with error: {} ({:?}). Check NCCL_DEBUG=INFO for more details.",
            operation,
            error_name,
            result
        )
    }
}

/// NCCL bootstrap for creating dedicated KVBM communicator.
///
/// This struct holds the unique ID needed to initialize an NCCL communicator
/// across multiple ranks. The typical usage pattern is:
///
/// 1. Rank 0: Call `NcclBootstrap::generate(world_size)` to create a new unique ID
/// 2. Rank 0: Serialize with `serialize()` and broadcast to other ranks
/// 3. Other ranks: Call `NcclBootstrap::deserialize(bytes)` to reconstruct
/// 4. All ranks: Call `init_communicator(rank)` collectively to create the comm
///
/// # Example
/// ```ignore
/// // On rank 0:
/// let bootstrap = NcclBootstrap::generate(world_size)?;
/// let data = bootstrap.serialize();
/// // ... broadcast data via MPI ...
///
/// // On all ranks:
/// let bootstrap = if rank == 0 {
///     bootstrap
/// } else {
///     NcclBootstrap::deserialize(&received_data)?
/// };
///
/// // All ranks call this together:
/// let comm = bootstrap.init_communicator(rank)?;
/// ```
pub struct NcclBootstrap {
    unique_id: ncclUniqueId,
    world_size: i32,
}

impl NcclBootstrap {
    /// Generate a new unique ID for NCCL communicator initialization.
    /// This should only be called on rank 0.
    ///
    /// # Arguments
    /// * `world_size` - The total number of ranks that will participate
    pub fn generate(world_size: i32) -> Result<Self> {
        let mut unique_id = ncclUniqueId { internal: [0; 128] };
        let result = unsafe { ncclGetUniqueId(&mut unique_id) };
        check_nccl_result(result, "ncclGetUniqueId")?;
        Ok(Self {
            unique_id,
            world_size,
        })
    }

    /// Serialize the bootstrap data for distribution to other ranks.
    /// Format: 4 bytes world_size (little endian) + 4 bytes padding + 128 bytes unique_id
    pub fn serialize(&self) -> Vec<u8> {
        let mut bytes = Vec::with_capacity(136);
        bytes.extend_from_slice(&self.world_size.to_le_bytes());
        bytes.extend_from_slice(&[0u8; 4]); // padding for alignment
        let internal_bytes: &[u8; 128] =
            unsafe { &*self.unique_id.internal.as_ptr().cast::<[u8; 128]>() };
        bytes.extend_from_slice(internal_bytes);
        bytes
    }

    /// Deserialize bootstrap data received from rank 0.
    ///
    /// # Arguments
    /// * `bytes` - The serialized bootstrap data (136 bytes)
    pub fn deserialize(bytes: &[u8]) -> Result<Self> {
        anyhow::ensure!(
            bytes.len() == 136,
            "Invalid bootstrap data length: expected 136, got {}",
            bytes.len()
        );

        let world_size = i32::from_le_bytes(
            bytes[0..4]
                .try_into()
                .context("Failed to parse world_size")?,
        );

        let mut unique_id = ncclUniqueId { internal: [0; 128] };
        // c_char is i8 on x86_64 but u8 on aarch64; use ptr copy to be portable
        unsafe {
            std::ptr::copy_nonoverlapping(
                bytes[8..136].as_ptr(),
                unique_id.internal.as_mut_ptr().cast::<u8>(),
                128,
            );
        }

        Ok(Self {
            unique_id,
            world_size,
        })
    }

    /// Initialize the NCCL communicator.
    ///
    /// # IMPORTANT: This is a collective operation!
    /// All ranks must call this function together with matching parameters.
    /// The function will block until all ranks have called it.
    ///
    /// # Arguments
    /// * `rank` - This rank's ID (0 to world_size-1)
    ///
    /// # Returns
    /// The raw `ncclComm_t` handle. The caller is responsible for eventually
    /// calling `ncclCommDestroy` on this handle.
    ///
    /// # Safety
    /// The returned communicator must be properly destroyed when no longer needed.
    pub fn init_communicator(&self, rank: i32) -> Result<ncclComm_t> {
        anyhow::ensure!(
            rank >= 0 && rank < self.world_size,
            "Invalid rank {}: must be in range [0, {})",
            rank,
            self.world_size
        );

        // CudaRC doesn't seem to have any nice bindings to the NCCL config.
        // We have to manually create it the same way the NCCL C++ macros do.
        let mut config: ncclConfig_t;

        // Query runtime NCCL version instead of hardcoding — avoids
        // ncclInvalidUsage when the container's NCCL version doesn't match.
        let nccl_version = {
            let mut v: std::ffi::c_int = 0;
            let result = unsafe { ncclGetVersion(&mut v) };
            check_nccl_result(result, "ncclGetVersion")?;
            tracing::debug!("NCCL runtime version: {v}");
            v as std::ffi::c_uint
        };

        let max_ctas = std::env::var("DYN_KVBM_NCCL_MAX_CTAS")
            .ok()
            .and_then(|val| val.parse::<i32>().ok())
            .unwrap_or(8);

        config = ncclConfig_t {
            size: std::mem::size_of::<ncclConfig_t>(),
            magic: 0xcafebeef, // Required Magic Number
            version: nccl_version,
            blocking: 1,
            cgaClusterSize: i32::MIN,
            minCTAs: 1,
            maxCTAs: max_ctas,
            netName: std::ptr::null_mut(),
            splitShare: i32::MIN,
            trafficClass: i32::MIN,
            commName: std::ptr::null_mut(),
            collnetEnable: 0,
            CTAPolicy: i32::MIN,
            shrinkShare: i32::MIN,
            nvlsCTAs: i32::MIN,
            nChannelsPerNetPeer: i32::MIN,
            nvlinkCentricSched: i32::MIN,
        };

        let mut comm: ncclComm_t = std::ptr::null_mut();
        tracing::debug!(
            "Calling ncclCommInitRank: rank={}, world_size={}",
            rank,
            self.world_size
        );

        let result = unsafe {
            ncclCommInitRankConfig(
                &mut comm,
                self.world_size,
                self.unique_id,
                rank,
                &mut config,
            )
        };
        check_nccl_result(
            result,
            &format!(
                "ncclCommInitRank(rank={}, world_size={})",
                rank, self.world_size
            ),
        )?;
        tracing::info!(
            "NCCL communicator initialized successfully: rank={}, world_size={}",
            rank,
            self.world_size
        );

        Ok(comm)
    }

    /// Get the world size for this bootstrap.
    pub fn world_size(&self) -> i32 {
        self.world_size
    }
}

/// RAII wrapper for ncclComm_t that destroys the communicator on drop.
pub struct NcclCommOwned {
    comm: ncclComm_t,
}

// Safety: NCCL communicators are internally thread-safe.
// NCCL serializes operations on the same communicator.
unsafe impl Send for NcclCommOwned {}
unsafe impl Sync for NcclCommOwned {}

impl NcclCommOwned {
    /// Create a new owned communicator from a raw handle.
    ///
    /// # Safety
    /// The caller must ensure that `comm` is a valid NCCL communicator
    /// that has not been destroyed and is not shared elsewhere.
    pub unsafe fn from_raw(comm: ncclComm_t) -> Self {
        Self { comm }
    }

    /// Get the raw communicator handle.
    pub fn as_raw(&self) -> ncclComm_t {
        self.comm
    }
}

impl Drop for NcclCommOwned {
    fn drop(&mut self) {
        if !self.comm.is_null() {
            let result = unsafe { ncclCommDestroy(self.comm) };
            if result != ncclResult_t::ncclSuccess {
                tracing::error!("Failed to destroy NCCL communicator: {:?}", result);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_serialize_deserialize() {
        let internal_bytes: [u8; 128] = [42u8; 128];
        let mut unique_id = ncclUniqueId { internal: [0; 128] };
        unsafe {
            std::ptr::copy_nonoverlapping(
                internal_bytes.as_ptr(),
                unique_id.internal.as_mut_ptr().cast::<u8>(),
                128,
            );
        }
        let bootstrap = NcclBootstrap {
            unique_id,
            world_size: 4,
        };

        let bytes = bootstrap.serialize();
        assert_eq!(bytes.len(), 136);

        let restored = NcclBootstrap::deserialize(&bytes).unwrap();
        assert_eq!(restored.world_size, 4);
        let restored_bytes: &[u8; 128] =
            unsafe { &*restored.unique_id.internal.as_ptr().cast::<[u8; 128]>() };
        assert_eq!(*restored_bytes, [42u8; 128]);
    }

    #[test]
    fn test_deserialize_invalid_length() {
        let bytes = vec![0u8; 100]; // Wrong length
        let result = NcclBootstrap::deserialize(&bytes);
        assert!(result.is_err());
    }
}
