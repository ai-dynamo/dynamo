// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Registry metadata types and traits.

use std::fmt::Debug;

pub trait RegistryMetadata: Clone + Debug + Default + Send + Sync + 'static {
    fn to_bytes(&self) -> Vec<u8>;
    fn from_bytes(bytes: &[u8]) -> Option<Self>;
}

#[derive(Clone, Debug, Default)]
pub struct NoMetadata;

impl PartialEq for NoMetadata {
    fn eq(&self, _other: &Self) -> bool {
        true
    }
}

impl Eq for NoMetadata {}

impl RegistryMetadata for NoMetadata {
    fn to_bytes(&self) -> Vec<u8> {
        Vec::new()
    }

    fn from_bytes(_bytes: &[u8]) -> Option<Self> {
        Some(Self)
    }
}

#[derive(Clone, Debug, Default)]
pub struct TimestampMetadata {
    pub created_at: u64,
    pub ttl_secs: Option<u32>,
}

impl RegistryMetadata for TimestampMetadata {
    fn to_bytes(&self) -> Vec<u8> {
        let mut buf = Vec::with_capacity(12);
        buf.extend_from_slice(&self.created_at.to_le_bytes());
        buf.extend_from_slice(&self.ttl_secs.unwrap_or(0).to_le_bytes());
        buf
    }

    fn from_bytes(bytes: &[u8]) -> Option<Self> {
        if bytes.len() < 12 {
            return None;
        }
        let created_at = u64::from_le_bytes(bytes[0..8].try_into().ok()?);
        let ttl_raw = u32::from_le_bytes(bytes[8..12].try_into().ok()?);
        Some(Self {
            created_at,
            ttl_secs: if ttl_raw == 0 { None } else { Some(ttl_raw) },
        })
    }
}

#[derive(Clone, Debug, Default)]
pub struct PositionMetadata {
    pub position: u32,
    pub parent_hash: Option<u64>,
    pub sequence_length: Option<u32>,
    pub created_at: u64,
}

impl RegistryMetadata for PositionMetadata {
    fn to_bytes(&self) -> Vec<u8> {
        let mut buf = Vec::with_capacity(24);
        buf.extend_from_slice(&self.position.to_le_bytes());
        buf.extend_from_slice(&self.parent_hash.unwrap_or(0).to_le_bytes());
        buf.extend_from_slice(&self.sequence_length.unwrap_or(0).to_le_bytes());
        buf.extend_from_slice(&self.created_at.to_le_bytes());
        buf
    }

    fn from_bytes(bytes: &[u8]) -> Option<Self> {
        if bytes.len() < 24 {
            return None;
        }
        let position = u32::from_le_bytes(bytes[0..4].try_into().ok()?);
        let parent_raw = u64::from_le_bytes(bytes[4..12].try_into().ok()?);
        let seq_len_raw = u32::from_le_bytes(bytes[12..16].try_into().ok()?);
        let created_at = u64::from_le_bytes(bytes[16..24].try_into().ok()?);

        Some(Self {
            position,
            parent_hash: if parent_raw == 0 { None } else { Some(parent_raw) },
            sequence_length: if seq_len_raw == 0 { None } else { Some(seq_len_raw) },
            created_at,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_no_metadata_roundtrip() {
        let meta = NoMetadata;
        let bytes = meta.to_bytes();
        assert!(bytes.is_empty());
        assert!(NoMetadata::from_bytes(&bytes).is_some());
    }

    #[test]
    fn test_timestamp_metadata_roundtrip() {
        let meta = TimestampMetadata {
            created_at: 1234567890,
            ttl_secs: Some(3600),
        };
        let bytes = meta.to_bytes();
        let decoded = TimestampMetadata::from_bytes(&bytes).unwrap();
        assert_eq!(decoded.created_at, meta.created_at);
        assert_eq!(decoded.ttl_secs, meta.ttl_secs);
    }

    #[test]
    fn test_position_metadata_roundtrip() {
        let meta = PositionMetadata {
            position: 5,
            parent_hash: Some(0xDEADBEEF),
            sequence_length: Some(10),
            created_at: 9999999999,
        };
        let bytes = meta.to_bytes();
        let decoded = PositionMetadata::from_bytes(&bytes).unwrap();
        assert_eq!(decoded.position, meta.position);
        assert_eq!(decoded.parent_hash, meta.parent_hash);
        assert_eq!(decoded.sequence_length, meta.sequence_length);
        assert_eq!(decoded.created_at, meta.created_at);
    }
}

