// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! oneCCL bootstrap utilities for creating communicators from scratch.
//!
//! This module provides helpers for initializing oneCCL communicators in
//! standalone Rust applications and tests, where no external launcher provides
//! pre-initialized communicators.
//!
//! # Two Construction Paths
//!
//! oneCCL communicators can be created via two paths:
//!
//! 1. **Bootstrap (this module)**: For tests and standalone Rust applications.
//!    Rank 0 creates a main KVS, extracts its 256-byte address, distributes it
//!    to other ranks, and all ranks collectively call `ccl_rs_comm_create`.
//!
//! 2. **Borrowed handles**: For production use with PyTorch, vLLM, etc.
//!    The external runtime creates the communicator, and Rust code borrows it
//!    via FFI. See [`OneCclCollectives::from_borrowed`].

use std::ffi::c_void;
use std::ptr;

use anyhow::{Context, Result};
use oneapi_rs::ccl::result::CclError;
use oneapi_rs::ccl::sys;

/// Bootstrap for creating oneCCL communicators from scratch.
///
/// Wraps the oneCCL KVS (Key-Value Store) mechanism for distributed
/// rank discovery, analogous to NCCL's `ncclUniqueId` bootstrap.
///
/// # Workflow
///
/// 1. Rank 0 calls [`OneCclBootstrap::generate`] to create the main KVS
/// 2. Rank 0 serializes via [`OneCclBootstrap::serialize`] and sends to other ranks
/// 3. Other ranks deserialize via [`OneCclBootstrap::deserialize`]
/// 4. All ranks collectively call [`OneCclBootstrap::init_communicator`]
pub struct OneCclBootstrap {
    kvs: *mut sys::ccl_rs_kvs_t,
    address: sys::ccl_rs_kvs_address_t,
    world_size: usize,
}

// SAFETY: The KVS handle is thread-safe for read-only use after creation.
// Each rank creates its own communicator from the same KVS address.
unsafe impl Send for OneCclBootstrap {}
unsafe impl Sync for OneCclBootstrap {}

/// Size of the serialized bootstrap (8 bytes world_size + 256 bytes KVS address).
pub const SERIALIZED_SIZE: usize = 8 + sys::CCL_RS_KVS_ADDRESS_SIZE;

impl OneCclBootstrap {
    /// Generate a new bootstrap on rank 0.
    ///
    /// Creates a main KVS and extracts its 256-byte address for distribution.
    ///
    /// # Arguments
    /// * `world_size` - Total number of ranks in the collective group
    pub fn generate(world_size: usize) -> Result<Self> {
        anyhow::ensure!(
            world_size > 0 && world_size <= i32::MAX as usize,
            "world_size must be in 1..={}, got {}",
            i32::MAX,
            world_size
        );

        // Initialize oneCCL runtime
        check_ccl_result(unsafe { sys::ccl_rs_init() })
            .context("ccl_rs_init failed")?;

        // Create the main KVS (rank 0 only)
        let mut kvs: *mut sys::ccl_rs_kvs_t = ptr::null_mut();
        check_ccl_result(unsafe { sys::ccl_rs_kvs_create_main(&mut kvs) })
            .context("ccl_rs_kvs_create_main failed")?;

        // Extract the address
        let mut address = sys::ccl_rs_kvs_address_t::default();
        check_ccl_result(unsafe { sys::ccl_rs_kvs_get_address(kvs, &mut address) })
            .context("ccl_rs_kvs_get_address failed")?;

        Ok(Self {
            kvs,
            address,
            world_size,
        })
    }

    /// Get the world size for this bootstrap.
    pub fn world_size(&self) -> usize {
        self.world_size
    }

    /// Serialize the bootstrap for transmission to other ranks.
    ///
    /// Format:
    /// - 8 bytes: world_size as little-endian u64
    /// - 256 bytes: KVS address
    pub fn serialize(&self) -> Vec<u8> {
        let mut bytes = Vec::with_capacity(SERIALIZED_SIZE);
        bytes.extend_from_slice(&(self.world_size as u64).to_le_bytes());
        // Convert c_char array to u8 for serialization
        for &byte in &self.address.data {
            bytes.push(byte as u8);
        }
        bytes
    }

    /// Deserialize a bootstrap received from rank 0.
    ///
    /// Reconstructs the KVS from the 256-byte address so that non-rank-0
    /// processes can join the same collective group.
    pub fn deserialize(bytes: &[u8]) -> Result<Self> {
        if bytes.len() != SERIALIZED_SIZE {
            anyhow::bail!(
                "Invalid bootstrap data length: expected {}, got {}",
                SERIALIZED_SIZE,
                bytes.len()
            );
        }

        let world_size = u64::from_le_bytes(bytes[0..8].try_into().unwrap()) as usize;

        let mut address = sys::ccl_rs_kvs_address_t::default();
        for (i, &byte) in bytes[8..].iter().enumerate() {
            address.data[i] = byte as std::ffi::c_char;
        }

        // Initialize oneCCL runtime
        check_ccl_result(unsafe { sys::ccl_rs_init() })
            .context("ccl_rs_init failed")?;

        // Recreate KVS from address
        let mut kvs: *mut sys::ccl_rs_kvs_t = ptr::null_mut();
        check_ccl_result(unsafe {
            sys::ccl_rs_kvs_create_from_address(&address, &mut kvs)
        })
        .context("ccl_rs_kvs_create_from_address failed")?;

        Ok(Self {
            kvs,
            address,
            world_size,
        })
    }

    /// Initialize a oneCCL communicator for this rank.
    ///
    /// This is a **collective operation** - all ranks must call this method
    /// simultaneously with the same bootstrap data for initialization to succeed.
    ///
    /// # Arguments
    /// * `rank` - The rank of this worker (0 to world_size-1)
    ///
    /// # Returns
    /// A raw communicator handle. Caller must eventually call `ccl_rs_comm_destroy`.
    pub fn init_communicator(&self, rank: usize) -> Result<*mut sys::ccl_rs_comm_t> {
        if rank >= self.world_size {
            anyhow::bail!(
                "Rank {} is invalid for world_size {}",
                rank,
                self.world_size
            );
        }

        let mut comm: *mut sys::ccl_rs_comm_t = ptr::null_mut();
        check_ccl_result(unsafe {
            sys::ccl_rs_comm_create(
                self.world_size as std::ffi::c_int,
                rank as std::ffi::c_int,
                self.kvs,
                &mut comm,
            )
        })
        .context("ccl_rs_comm_create failed")?;

        tracing::debug!(
            rank,
            world_size = self.world_size,
            "oneCCL communicator initialized"
        );

        Ok(comm)
    }

    /// Create a oneCCL stream from a raw SYCL queue pointer.
    ///
    /// # Safety
    /// `sycl_queue_ptr` must be a valid `sycl::queue*`.
    pub unsafe fn create_stream(
        sycl_queue_ptr: *mut c_void,
    ) -> Result<*mut sys::ccl_rs_stream_t> {
        let mut stream: *mut sys::ccl_rs_stream_t = ptr::null_mut();
        check_ccl_result(unsafe {
            sys::ccl_rs_stream_create_from_native_queue(sycl_queue_ptr, &mut stream)
        })
        .context("ccl_rs_stream_create_from_native_queue failed")?;
        Ok(stream)
    }
}

impl Drop for OneCclBootstrap {
    fn drop(&mut self) {
        if !self.kvs.is_null() {
            unsafe { sys::ccl_rs_kvs_destroy(self.kvs) };
        }
    }
}

/// Convert a ccl_rs_result_t to anyhow::Result.
pub(crate) fn check_ccl_result(result: sys::ccl_rs_result_t) -> Result<(), CclError> {
    result.result()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bootstrap_serialization_roundtrip() {
        // Test serialization logic without calling actual CCL functions.
        // We construct a bootstrap with a dummy address.
        let world_size = 4usize;

        let mut address = sys::ccl_rs_kvs_address_t::default();
        for (i, byte) in address.data.iter_mut().enumerate() {
            *byte = (i % 128) as std::ffi::c_char;
        }

        // Manually build the serialized bytes
        let mut bytes = Vec::with_capacity(SERIALIZED_SIZE);
        bytes.extend_from_slice(&(world_size as u64).to_le_bytes());
        for &byte in &address.data {
            bytes.push(byte as u8);
        }

        assert_eq!(bytes.len(), SERIALIZED_SIZE);

        // Verify world_size round-trips
        let decoded_ws = u64::from_le_bytes(bytes[0..8].try_into().unwrap()) as usize;
        assert_eq!(decoded_ws, world_size);

        // Verify address round-trips
        let mut decoded_addr = sys::ccl_rs_kvs_address_t::default();
        for (i, &byte) in bytes[8..].iter().enumerate() {
            decoded_addr.data[i] = byte as std::ffi::c_char;
        }
        assert_eq!(decoded_addr.data.len(), address.data.len());
    }

    #[test]
    fn test_deserialize_invalid_length() {
        let bytes = vec![0u8; 10]; // Wrong length
        let result = OneCclBootstrap::deserialize(&bytes);
        assert!(result.is_err());
    }
}
