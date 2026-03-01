// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::sync::OnceLock;

/// Whether G4 checksum validation is enabled (from env var).
static G4_CHECKSUM_ENABLED_CELL: OnceLock<bool> = OnceLock::new();

/// Check if G4 checksum validation is enabled.
pub fn g4_checksum_enabled() -> bool {
    *G4_CHECKSUM_ENABLED_CELL.get_or_init(|| {
        std::env::var("DYN_KVBM_G4_CHECKSUM_VALIDATION")
            .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
            .unwrap_or(false)
    })
}

// Re-export core checksum functions from the shared module
pub use crate::block_manager::block::transfer::checksum::{
    BlockChecksum, ChecksumBuilder, compute_checksum, compute_checksum_raw, verify_checksum,
};

/// G4 transfer checksum record for validation.
#[derive(Debug, Clone)]
pub struct G4ChecksumRecord {
    pub sequence_hash: u64,
    pub checksum: BlockChecksum,
    pub block_size: usize,
}

impl G4ChecksumRecord {
    pub fn new(sequence_hash: u64, checksum: BlockChecksum, block_size: usize) -> Self {
        Self {
            sequence_hash,
            checksum,
            block_size,
        }
    }
}

/// Compute checksums for multiple blocks in bounce buffers.
///
/// # Safety
/// Each pointer must be valid for `block_size` bytes.
pub unsafe fn compute_g4_checksums(
    bounce_ptrs: &[*const u8],
    block_size: usize,
    sequence_hashes: &[u64],
) -> Vec<G4ChecksumRecord> {
    assert_eq!(bounce_ptrs.len(), sequence_hashes.len());

    bounce_ptrs
        .iter()
        .zip(sequence_hashes.iter())
        .map(|(&ptr, &hash)| {
            let checksum = unsafe { compute_checksum_raw(ptr, block_size) };
            G4ChecksumRecord::new(hash, checksum, block_size)
        })
        .collect()
}

/// Verify checksums for blocks after G4 read.
///
/// # Safety
/// Each pointer must be valid for `block_size` bytes.
pub unsafe fn verify_g4_checksums(
    bounce_ptrs: &[*const u8],
    block_size: usize,
    expected: &[G4ChecksumRecord],
) -> Result<(), G4ChecksumError> {
    if bounce_ptrs.len() != expected.len() {
        return Err(G4ChecksumError::CountMismatch {
            actual: bounce_ptrs.len(),
            expected: expected.len(),
        });
    }

    for (idx, (&ptr, record)) in bounce_ptrs.iter().zip(expected.iter()).enumerate() {
        let actual = unsafe { compute_checksum_raw(ptr, block_size) };
        if actual != record.checksum {
            return Err(G4ChecksumError::Mismatch {
                block_index: idx,
                sequence_hash: record.sequence_hash,
                expected: record.checksum.clone(),
                actual,
            });
        }
    }

    Ok(())
}

/// Error type for G4 checksum validation failures.
#[derive(Debug, Clone)]
pub enum G4ChecksumError {
    CountMismatch { actual: usize, expected: usize },
    Mismatch {
        block_index: usize,
        sequence_hash: u64,
        expected: String,
        actual: String,
    },
}

impl std::fmt::Display for G4ChecksumError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            G4ChecksumError::CountMismatch { actual, expected } => {
                write!(
                    f,
                    "Block count mismatch: got {}, expected {}",
                    actual, expected
                )
            }
            G4ChecksumError::Mismatch {
                block_index,
                sequence_hash,
                expected,
                actual,
            } => {
                write!(
                    f,
                    "Checksum mismatch at block {} (hash {:016x}): expected {}, got {}",
                    block_index, sequence_hash, expected, actual
                )
            }
        }
    }
}

impl std::error::Error for G4ChecksumError {}
