// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Registry key types and traits.

use std::fmt::Debug;
use std::hash::Hash;

use super::storage::PositionalStorageKey;

/// Trait for registry keys.
pub trait RegistryKey: Copy + Hash + Eq + Debug + Send + Sync + 'static {
    fn to_bytes(&self) -> Vec<u8>;
    fn from_bytes(bytes: &[u8]) -> Option<Self>;
}

impl RegistryKey for u64 {
    fn to_bytes(&self) -> Vec<u8> {
        self.to_le_bytes().to_vec()
    }

    fn from_bytes(bytes: &[u8]) -> Option<Self> {
        bytes.try_into().ok().map(u64::from_le_bytes)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Key128(pub u128);

impl RegistryKey for Key128 {
    fn to_bytes(&self) -> Vec<u8> {
        self.0.to_le_bytes().to_vec()
    }

    fn from_bytes(bytes: &[u8]) -> Option<Self> {
        if bytes.len() < 16 {
            return None;
        }
        Some(Self(u128::from_le_bytes(bytes[0..16].try_into().ok()?)))
    }
}

/// Key combining worker identifier and sequence hash.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct CompositeKey {
    pub worker_id: u64,
    pub sequence_hash: u64,
}

impl RegistryKey for CompositeKey {
    fn to_bytes(&self) -> Vec<u8> {
        let mut buf = Vec::with_capacity(16);
        buf.extend_from_slice(&self.worker_id.to_le_bytes());
        buf.extend_from_slice(&self.sequence_hash.to_le_bytes());
        buf
    }

    fn from_bytes(bytes: &[u8]) -> Option<Self> {
        if bytes.len() < 16 {
            return None;
        }
        Some(Self {
            worker_id: u64::from_le_bytes(bytes[0..8].try_into().ok()?),
            sequence_hash: u64::from_le_bytes(bytes[8..16].try_into().ok()?),
        })
    }
}

/// Key with worker id, sequence hash, and position in sequence.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct PositionalKey {
    pub worker_id: u64,
    pub sequence_hash: u64,
    pub position: u32,
}

impl RegistryKey for PositionalKey {
    fn to_bytes(&self) -> Vec<u8> {
        let mut buf = Vec::with_capacity(20);
        buf.extend_from_slice(&self.worker_id.to_le_bytes());
        buf.extend_from_slice(&self.sequence_hash.to_le_bytes());
        buf.extend_from_slice(&self.position.to_le_bytes());
        buf
    }

    fn from_bytes(bytes: &[u8]) -> Option<Self> {
        if bytes.len() < 20 {
            return None;
        }
        Some(Self {
            worker_id: u64::from_le_bytes(bytes[0..8].try_into().ok()?),
            sequence_hash: u64::from_le_bytes(bytes[8..16].try_into().ok()?),
            position: u32::from_le_bytes(bytes[16..20].try_into().ok()?),
        })
    }
}

impl PositionalStorageKey for PositionalKey {
    fn position(&self) -> u64 {
        self.position as u64
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_u64_roundtrip() {
        let key: u64 = 0x123456789ABCDEF0;
        let bytes = key.to_bytes();
        assert_eq!(u64::from_bytes(&bytes), Some(key));
    }

    #[test]
    fn test_key128_roundtrip() {
        let key = Key128(0x123456789ABCDEF0_FEDCBA9876543210);
        let bytes = key.to_bytes();
        assert_eq!(Key128::from_bytes(&bytes), Some(key));
    }

    #[test]
    fn test_composite_key_roundtrip() {
        let key = CompositeKey {
            worker_id: 123,
            sequence_hash: 456,
        };
        let bytes = key.to_bytes();
        assert_eq!(CompositeKey::from_bytes(&bytes), Some(key));
    }

    #[test]
    fn test_positional_key_roundtrip() {
        let key = PositionalKey {
            worker_id: 123,
            sequence_hash: 456,
            position: 789,
        };
        let bytes = key.to_bytes();
        assert_eq!(PositionalKey::from_bytes(&bytes), Some(key));
    }
}
