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
    ncclComm_t, ncclCommDestroy, ncclCommInitRank, ncclGetUniqueId, ncclResult_t, ncclUniqueId,
};

/// Check NCCL result and convert to anyhow::Result
fn check_nccl_result(result: ncclResult_t) -> Result<()> {
    if result == ncclResult_t::ncclSuccess {
        Ok(())
    } else {
        anyhow::bail!("NCCL error: {:?}", result)
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
        check_nccl_result(result).context("ncclGetUniqueId failed")?;
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
        // 4 bytes padding for alignment
        bytes.extend_from_slice(&[0u8; 4]);
        // Cast i8 array to u8 for serialization (same binary representation)
        let internal_bytes: &[u8; 128] = unsafe { std::mem::transmute(&self.unique_id.internal) };
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
        // Cast u8 slice to i8 for the internal array (same binary representation)
        let internal_slice: &[i8] =
            unsafe { std::slice::from_raw_parts(bytes[8..136].as_ptr() as *const i8, 128) };
        unique_id.internal.copy_from_slice(internal_slice);

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

        let mut comm: ncclComm_t = std::ptr::null_mut();
        let result = unsafe { ncclCommInitRank(&mut comm, self.world_size, self.unique_id, rank) };
        check_nccl_result(result).context("ncclCommInitRank failed")?;

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
        // We can test serialization without actually calling NCCL
        let bootstrap = NcclBootstrap {
            unique_id: ncclUniqueId {
                internal: [42i8; 128],
            },
            world_size: 4,
        };

        let bytes = bootstrap.serialize();
        assert_eq!(bytes.len(), 136);

        let restored = NcclBootstrap::deserialize(&bytes).unwrap();
        assert_eq!(restored.world_size, 4);
        assert_eq!(restored.unique_id.internal, [42i8; 128]);
    }

    #[test]
    fn test_deserialize_invalid_length() {
        let bytes = vec![0u8; 100]; // Wrong length
        let result = NcclBootstrap::deserialize(&bytes);
        assert!(result.is_err());
    }
}
