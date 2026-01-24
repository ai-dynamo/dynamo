// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! NCCL bootstrap utilities for creating communicators from scratch.
//!
//! This module provides helpers for initializing NCCL communicators in standalone
//! Rust applications and tests, where no external launcher (like PyTorch) provides
//! pre-initialized communicators.
//!
//! # Two Construction Paths
//!
//! NCCL communicators can be created via two paths:
//!
//! 1. **Bootstrap (this module)**: For tests and standalone Rust applications.
//!    Rank 0 generates a unique ID, distributes it to other ranks, and all
//!    ranks collectively call `ncclCommInitRank`.
//!
//! 2. **Borrowed handles**: For production use with PyTorch, vLLM, or TensorRT-LLM.
//!    The external runtime creates the communicator, and Rust code borrows it
//!    via FFI. See [`NcclCollectives::from_borrowed`].
//!
//! # Example: Multi-process Bootstrap
//!
//! ```rust,ignore
//! use kvbm::v2::distributed::collectives::NcclBootstrap;
//!
//! // Rank 0: Generate and share the unique ID
//! if rank == 0 {
//!     let bootstrap = NcclBootstrap::generate(world_size)?;
//!     let bytes = bootstrap.serialize();
//!     // Send `bytes` to other ranks via your IPC mechanism
//! }
//!
//! // All ranks: Initialize communicator
//! let bootstrap = if rank == 0 {
//!     NcclBootstrap::generate(world_size)?
//! } else {
//!     let bytes = receive_from_rank_0();
//!     NcclBootstrap::deserialize(&bytes)?
//! };
//!
//! let comm = bootstrap.init_communicator(rank, stream)?;
//! ```

use std::mem::MaybeUninit;

use anyhow::{Context, Result};
use cudarc::driver::sys::CUstream;
use cudarc::nccl::sys::{
    ncclComm_t, ncclCommInitRank, ncclGetUniqueId, ncclResult_t, ncclUniqueId,
};

/// Bootstrap for creating NCCL communicators from scratch.
///
/// Used by tests and standalone Rust applications where NCCL communicators
/// need to be created without an external launcher.
///
/// # Workflow
///
/// 1. Rank 0 calls [`NcclBootstrap::generate`] to create the unique ID
/// 2. Rank 0 serializes via [`NcclBootstrap::serialize`] and sends to other ranks
/// 3. Other ranks deserialize via [`NcclBootstrap::deserialize`]
/// 4. All ranks collectively call [`NcclBootstrap::init_communicator`]
///
/// # Thread Safety
///
/// The bootstrap object itself is not thread-safe, but multiple threads can
/// each have their own bootstrap object with the same unique ID to initialize
/// communicators on different devices.
#[derive(Clone)]
pub struct NcclBootstrap {
    nccl_id: ncclUniqueId,
    world_size: usize,
}

impl NcclBootstrap {
    /// Generate a new bootstrap on rank 0.
    ///
    /// This creates a unique NCCL ID that must be shared with all other ranks
    /// before they can initialize their communicators.
    ///
    /// # Arguments
    /// * `world_size` - Total number of ranks in the collective group
    ///
    /// # Returns
    /// A bootstrap object that can be serialized and distributed to other ranks.
    ///
    /// # Errors
    /// Returns an error if NCCL fails to generate a unique ID.
    pub fn generate(world_size: usize) -> Result<Self> {
        let mut nccl_id = MaybeUninit::<ncclUniqueId>::uninit();

        // SAFETY: ncclGetUniqueId initializes the ncclUniqueId struct
        let result = unsafe { ncclGetUniqueId(nccl_id.as_mut_ptr()) };
        check_nccl_result(result).context("Failed to generate NCCL unique ID")?;

        // SAFETY: ncclGetUniqueId has initialized the struct
        let nccl_id = unsafe { nccl_id.assume_init() };

        Ok(Self { nccl_id, world_size })
    }

    /// Get the world size for this bootstrap.
    pub fn world_size(&self) -> usize {
        self.world_size
    }

    /// Serialize the bootstrap for transmission to other ranks.
    ///
    /// The serialized format is:
    /// - 8 bytes: world_size as little-endian u64
    /// - 128 bytes: NCCL unique ID internal data
    ///
    /// # Returns
    /// A byte vector that can be transmitted via any IPC mechanism.
    pub fn serialize(&self) -> Vec<u8> {
        let mut bytes = Vec::with_capacity(8 + 128);
        bytes.extend_from_slice(&(self.world_size as u64).to_le_bytes());
        // Convert i8 array to u8 for serialization
        for &byte in &self.nccl_id.internal {
            bytes.push(byte as u8);
        }
        bytes
    }

    /// Deserialize a bootstrap received from rank 0.
    ///
    /// # Arguments
    /// * `bytes` - Serialized bootstrap data from [`NcclBootstrap::serialize`]
    ///
    /// # Returns
    /// A bootstrap object that can be used to initialize a communicator.
    ///
    /// # Errors
    /// Returns an error if the byte array has incorrect length.
    pub fn deserialize(bytes: &[u8]) -> Result<Self> {
        if bytes.len() != 8 + 128 {
            anyhow::bail!(
                "Invalid bootstrap data length: expected {}, got {}",
                8 + 128,
                bytes.len()
            );
        }

        let world_size = u64::from_le_bytes(bytes[0..8].try_into().unwrap()) as usize;

        let mut nccl_id = ncclUniqueId {
            internal: [0i8; 128],
        };
        // Copy bytes into internal array
        for (i, &byte) in bytes[8..].iter().enumerate() {
            nccl_id.internal[i] = byte as i8;
        }

        Ok(Self { nccl_id, world_size })
    }

    /// Initialize an NCCL communicator for this rank.
    ///
    /// This is a **collective operation** - all ranks must call this method
    /// simultaneously with the same bootstrap data for initialization to succeed.
    ///
    /// # Arguments
    /// * `rank` - The rank of this worker (0 to world_size-1)
    /// * `stream` - The CUDA stream to associate with NCCL operations
    ///
    /// # Returns
    /// An NCCL communicator handle that can be used for collective operations.
    ///
    /// # Safety
    /// The returned communicator must be destroyed with `ncclCommDestroy` when
    /// no longer needed. The caller is responsible for lifetime management.
    ///
    /// # Errors
    /// Returns an error if:
    /// - `rank` is >= `world_size`
    /// - NCCL initialization fails (e.g., network issues, GPU errors)
    /// - Not all ranks call this method (will hang)
    pub fn init_communicator(&self, rank: usize, _stream: CUstream) -> Result<ncclComm_t> {
        if rank >= self.world_size {
            anyhow::bail!(
                "Rank {} is invalid for world_size {}",
                rank,
                self.world_size
            );
        }

        let mut comm = MaybeUninit::<ncclComm_t>::uninit();

        // SAFETY: ncclCommInitRank is a collective call that initializes the communicator.
        // All ranks must call this with the same nccl_id for it to complete.
        let result = unsafe {
            ncclCommInitRank(
                comm.as_mut_ptr(),
                self.world_size as i32,
                self.nccl_id,
                rank as i32,
            )
        };
        check_nccl_result(result).context("Failed to initialize NCCL communicator")?;

        // SAFETY: ncclCommInitRank has initialized the communicator
        let comm = unsafe { comm.assume_init() };

        tracing::debug!(
            rank,
            world_size = self.world_size,
            "NCCL communicator initialized"
        );

        Ok(comm)
    }
}

/// Check an NCCL result and convert to anyhow::Result.
pub(crate) fn check_nccl_result(result: ncclResult_t) -> Result<()> {
    if result == ncclResult_t::ncclSuccess {
        Ok(())
    } else {
        anyhow::bail!("NCCL error: {:?}", result)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bootstrap_serialization_roundtrip() {
        // Note: This test doesn't actually call NCCL functions,
        // it just tests the serialization logic
        let world_size = 4;

        // Create a bootstrap with a dummy ID (we can't call ncclGetUniqueId without NCCL)
        let original = NcclBootstrap {
            nccl_id: ncclUniqueId {
                internal: [42i8; 128],
            },
            world_size,
        };

        let bytes = original.serialize();
        assert_eq!(bytes.len(), 8 + 128);

        let deserialized = NcclBootstrap::deserialize(&bytes).unwrap();
        assert_eq!(deserialized.world_size, world_size);
        assert_eq!(deserialized.nccl_id.internal, original.nccl_id.internal);
    }

    #[test]
    fn test_deserialize_invalid_length() {
        let bytes = vec![0u8; 10]; // Wrong length
        let result = NcclBootstrap::deserialize(&bytes);
        assert!(result.is_err());
    }
}
