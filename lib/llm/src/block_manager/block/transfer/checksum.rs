// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Block checksum computation for transfer verification.
//!
//! Provides blake3-based checksums for validating data integrity during transfers.
//! Used for G4 (remote storage) validation and general transfer verification.

use blake3::Hasher;

/// Checksum type - blake3 hash as hex string.
pub type BlockChecksum = String;

/// Compute blake3 checksum for a byte slice.
///
/// Returns the checksum as a hex string.
#[inline]
pub fn compute_checksum(data: &[u8]) -> BlockChecksum {
    blake3::hash(data).to_string()
}

/// Compute checksum for data at a raw pointer.
///
/// # Safety
/// Caller must ensure `ptr` is valid for `size` bytes.
#[inline]
pub unsafe fn compute_checksum_raw(ptr: *const u8, size: usize) -> BlockChecksum {
    let slice = unsafe { std::slice::from_raw_parts(ptr, size) };
    compute_checksum(slice)
}

/// Verify that data matches an expected checksum.
#[inline]
pub fn verify_checksum(data: &[u8], expected: &str) -> bool {
    compute_checksum(data) == expected
}

/// Compute checksum using a hasher (for incremental hashing).
///
/// Useful when checksumming data from multiple non-contiguous regions.
pub struct ChecksumBuilder {
    hasher: Hasher,
}

impl ChecksumBuilder {
    /// Create a new checksum builder.
    pub fn new() -> Self {
        Self {
            hasher: Hasher::new(),
        }
    }

    /// Add data to the checksum.
    pub fn update(&mut self, data: &[u8]) -> &mut Self {
        self.hasher.update(data);
        self
    }

    /// Add data from a raw pointer.
    ///
    /// # Safety
    /// Caller must ensure `ptr` is valid for `size` bytes.
    pub unsafe fn update_raw(&mut self, ptr: *const u8, size: usize) -> &mut Self {
        let slice = unsafe { std::slice::from_raw_parts(ptr, size) };
        self.hasher.update(slice);
        self
    }

    /// Finalize and return the checksum.
    pub fn finalize(self) -> BlockChecksum {
        self.hasher.finalize().to_string()
    }
}

impl Default for ChecksumBuilder {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_compute_checksum() {
        let data = b"hello world";
        let checksum = compute_checksum(data);

        // blake3 produces 64-char hex string
        assert_eq!(checksum.len(), 64);

        // Same data should produce same checksum
        assert_eq!(checksum, compute_checksum(data));
    }

    #[test]
    fn test_verify_checksum() {
        let data = b"test data";
        let checksum = compute_checksum(data);

        assert!(verify_checksum(data, &checksum));
        assert!(!verify_checksum(b"different data", &checksum));
    }

    #[test]
    fn test_checksum_builder() {
        let data1 = b"hello ";
        let data2 = b"world";

        // Build incrementally
        let mut builder = ChecksumBuilder::new();
        builder.update(data1);
        builder.update(data2);
        let incremental = builder.finalize();

        // Compare to full computation
        let full = compute_checksum(b"hello world");

        assert_eq!(incremental, full);
    }

    #[test]
    fn test_compute_checksum_raw() {
        let data = vec![1u8, 2, 3, 4, 5];
        let checksum1 = compute_checksum(&data);
        let checksum2 = unsafe { compute_checksum_raw(data.as_ptr(), data.len()) };

        assert_eq!(checksum1, checksum2);
    }
}
