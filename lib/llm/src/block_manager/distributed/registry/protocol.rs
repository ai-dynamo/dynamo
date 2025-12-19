// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Wire protocol for distributed registry communication.
//!
//! All messages use a simple binary format for efficiency:
//! - Little-endian byte order
//! - No padding or alignment (packed structures)
//! - Variable-length payloads with explicit counts
//!
//! Multi-bucket support: All messages include bucket name to identify
//! which storage bucket the operation targets.
//!
//! Sequence hash support: Uses u64 for sequence hashes to match
//! the local registry format (crate::tokens::SequenceHash).

use super::types::ObjectKey;
use crate::block_manager::block::transfer::remote::RemoteKey;

/// Sequence hash type - u64 to match local registry
pub type SequenceHash = u64;

/// Bucket identifier - consistent hash of bucket name
pub type BucketId = u64;

/// Compute bucket ID from bucket name using consistent hashing
#[inline]
pub fn bucket_id_from_name(bucket_name: &str) -> BucketId {
    use std::hash::{Hash, Hasher};
    let mut hasher = std::collections::hash_map::DefaultHasher::new();
    bucket_name.hash(&mut hasher);
    hasher.finish()
}

/// Wire entry for registration messages.
///
/// Size: 16 bytes (packed)
///
/// Layout:
/// ```text
/// ┌─────────────────────────┬─────────────────────────┐
/// │    sequence_hash        │      object_key         │
/// │       8 bytes           │       8 bytes           │
/// └─────────────────────────┴─────────────────────────┘
/// ```
#[repr(C, packed)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct WireEntry {
    pub sequence_hash: u64,
    pub object_key: u64,
}

impl WireEntry {
    /// Size in bytes.
    pub const SIZE: usize = 16;

    /// Create new entry.
    pub fn new(sequence_hash: SequenceHash, object_key: ObjectKey) -> Self {
        Self {
            sequence_hash,
            object_key,
        }
    }

    /// Serialize to bytes.
    pub fn to_bytes(&self) -> [u8; Self::SIZE] {
        let mut buf = [0u8; Self::SIZE];
        buf[0..8].copy_from_slice(&self.sequence_hash.to_le_bytes());
        buf[8..16].copy_from_slice(&self.object_key.to_le_bytes());
        buf
    }

    /// Deserialize from bytes.
    pub fn from_bytes(bytes: &[u8]) -> Option<Self> {
        if bytes.len() < Self::SIZE {
            return None;
        }
        Some(Self {
            sequence_hash: u64::from_le_bytes(bytes[0..8].try_into().ok()?),
            object_key: u64::from_le_bytes(bytes[8..16].try_into().ok()?),
        })
    }
}

/// Message types for the wire protocol.
#[repr(u8)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MessageType {
    /// Payload: [bucket_id: 4][count: 2][entries: (hash, key) × count]
    Register = 1,

    /// Payload: [bucket_id: 4][count: 2][hashes × count]
    CanOffload = 2,

    /// Payload: [bucket_id: 4][count: 2][hashes × count]
    MatchSequence = 3,

    /// Payload: [count: 2][statuses × count]
    CanOffloadResponse = 5,

    /// Payload: [count: 2][entries: (hash, key) × count]
    MatchResponse = 6,

    /// Payload: bincode-serialized Vec<RemoteKey>
    MatchRemoteKeys = 7,

    /// Payload: bincode-serialized Vec<RemoteKey>
    MatchRemoteKeysResponse = 8,

    /// Payload: bincode-serialized Vec<RemoteKey>
    RegisterRemoteKeys = 9,
}

impl TryFrom<u8> for MessageType {
    type Error = ();

    fn try_from(value: u8) -> Result<Self, Self::Error> {
        match value {
            1 => Ok(Self::Register),
            2 => Ok(Self::CanOffload),
            3 => Ok(Self::MatchSequence),
            5 => Ok(Self::CanOffloadResponse),
            6 => Ok(Self::MatchResponse),
            7 => Ok(Self::MatchRemoteKeys),
            8 => Ok(Self::MatchRemoteKeysResponse),
            9 => Ok(Self::RegisterRemoteKeys),
            _ => Err(()),
        }
    }
}

// =============================================================================
// ENCODING
// =============================================================================

/// Encode a REGISTER message.
///
/// Format: `[type: 1][bucket_id: 8][count: 2][entries: count × 16]`
///
/// Size: 11 + (count × 16) bytes
///
/// # Panics
/// Panics if entries.len() exceeds u16::MAX (65535)
pub fn encode_register(bucket_id: BucketId, entries: &[(SequenceHash, ObjectKey)]) -> Vec<u8> {
    assert!(
        entries.len() <= u16::MAX as usize,
        "Too many entries: {} exceeds u16::MAX (65535)",
        entries.len()
    );
    let mut buf = Vec::with_capacity(11 + entries.len() * WireEntry::SIZE);
    buf.push(MessageType::Register as u8);
    buf.extend_from_slice(&bucket_id.to_le_bytes());
    buf.extend_from_slice(&(entries.len() as u16).to_le_bytes());
    for (hash, key) in entries {
        buf.extend_from_slice(&hash.to_le_bytes());  // 8 bytes
        buf.extend_from_slice(&key.to_le_bytes());   // 8 bytes
    }
    buf
}

/// Encode a REGISTER message using hash as key (common case).
pub fn encode_register_hash_as_key(bucket_id: BucketId, hashes: &[SequenceHash]) -> Vec<u8> {
    let entries: Vec<_> = hashes.iter().map(|&h| (h, h)).collect();
    encode_register(bucket_id, &entries)
}

/// Encode a CAN_OFFLOAD query.
///
/// Format: `[type: 1][bucket_id: 8][count: 2][hashes: count × 8]`
///
/// Size: 11 + (count × 8) bytes
///
/// # Panics
/// Panics if hashes.len() exceeds u16::MAX (65535)
pub fn encode_can_offload(bucket_id: BucketId, hashes: &[SequenceHash]) -> Vec<u8> {
    assert!(
        hashes.len() <= u16::MAX as usize,
        "Too many hashes: {} exceeds u16::MAX (65535)",
        hashes.len()
    );
    let mut buf = Vec::with_capacity(11 + hashes.len() * 8);
    buf.push(MessageType::CanOffload as u8);
    buf.extend_from_slice(&bucket_id.to_le_bytes());
    buf.extend_from_slice(&(hashes.len() as u16).to_le_bytes());
    for hash in hashes {
        buf.extend_from_slice(&hash.to_le_bytes());  // 8 bytes
    }
    buf
}

/// Encode a MATCH_SEQUENCE query.
///
/// Format: `[type: 1][bucket_id: 8][count: 2][hashes: count × 8]`
///
/// Size: 11 + (count × 8) bytes
///
/// # Panics
/// Panics if hashes.len() exceeds u16::MAX (65535)
pub fn encode_match_sequence(bucket_id: BucketId, hashes: &[SequenceHash]) -> Vec<u8> {
    assert!(
        hashes.len() <= u16::MAX as usize,
        "Too many hashes: {} exceeds u16::MAX (65535)",
        hashes.len()
    );
    let mut buf = Vec::with_capacity(11 + hashes.len() * 8);
    buf.push(MessageType::MatchSequence as u8);
    buf.extend_from_slice(&bucket_id.to_le_bytes());
    buf.extend_from_slice(&(hashes.len() as u16).to_le_bytes());
    for hash in hashes {
        buf.extend_from_slice(&hash.to_le_bytes());  // 8 bytes
    }
    buf
}

/// Status for each hash in can_offload response.
#[repr(u8)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum OffloadStatus {
    /// Hash is already stored in registry (skip offload).
    AlreadyStored = 0,
    /// Hash is leased by another worker (skip offload).
    Leased = 1,
    /// Lease granted to you (proceed with offload).
    Granted = 2,
}

impl TryFrom<u8> for OffloadStatus {
    type Error = ();

    fn try_from(value: u8) -> Result<Self, Self::Error> {
        match value {
            0 => Ok(Self::AlreadyStored),
            1 => Ok(Self::Leased),
            2 => Ok(Self::Granted),
            _ => Err(()),
        }
    }
}

/// Encode a CAN_OFFLOAD_RESPONSE with lease support.
///
/// Format: `[type: 1][count: 2][statuses: count × 1]`
///
/// Status values:
/// - 0 = AlreadyStored (hash in registry, skip)
/// - 1 = Leased (another worker has lease, skip)
/// - 2 = Granted (lease granted to you, offload it)
///
/// Size: 3 + count bytes
pub fn encode_can_offload_response_v2(statuses: &[OffloadStatus]) -> Vec<u8> {
    let mut buf = Vec::with_capacity(3 + statuses.len());
    buf.push(MessageType::CanOffloadResponse as u8);
    buf.extend_from_slice(&(statuses.len() as u16).to_le_bytes());
    for status in statuses {
        buf.push(*status as u8);
    }
    buf
}

/// Encode a CAN_OFFLOAD_RESPONSE (legacy bitmap format, no lease support).
///
/// Format: `[type: 1][count: 2][bitmap: ceil(count/8)]`
///
/// Bitmap interpretation:
/// - bit[i] = 1 → hash[i] NOT in registry → CAN offload (store it)
/// - bit[i] = 0 → hash[i] IN registry → already stored (skip)
///
/// Size: 3 + ceil(count/8) bytes
///
/// Note: This is the legacy format without lease support.
/// Use `encode_can_offload_response_v2` for lease support.
pub fn encode_can_offload_response(can_offload: &[bool]) -> Vec<u8> {
    // Convert to v2 format
    let statuses: Vec<_> = can_offload
        .iter()
        .map(|&can| {
            if can {
                OffloadStatus::Granted
            } else {
                OffloadStatus::AlreadyStored
            }
        })
        .collect();
    encode_can_offload_response_v2(&statuses)
}

/// Encode a MATCH_RESPONSE.
///
/// Format: `[type: 1][count: 2][entries: count × 16]`
///
/// Size: 3 + (count × 16) bytes
pub fn encode_match_response(entries: &[(SequenceHash, ObjectKey)]) -> Vec<u8> {
    let mut buf = Vec::with_capacity(3 + entries.len() * WireEntry::SIZE);
    buf.push(MessageType::MatchResponse as u8);
    buf.extend_from_slice(&(entries.len() as u16).to_le_bytes());
    for (hash, key) in entries {
        buf.extend_from_slice(&hash.to_le_bytes());  // 8 bytes
        buf.extend_from_slice(&key.to_le_bytes());   // 8 bytes
    }
    buf
}


pub fn encode_match_remote_keys(keys: &[RemoteKey]) -> Vec<u8> {
    let payload = bincode::serde::encode_to_vec(keys, bincode::config::standard())
        .expect("Failed to serialize RemoteKeys");
    let mut buf = Vec::with_capacity(1 + payload.len());
    buf.push(MessageType::MatchRemoteKeys as u8);
    buf.extend_from_slice(&payload);
    buf
}

pub fn encode_match_remote_keys_response(keys: &[RemoteKey]) -> Vec<u8> {
    let payload = bincode::serde::encode_to_vec(keys, bincode::config::standard())
        .expect("Failed to serialize RemoteKeys");
    let mut buf = Vec::with_capacity(1 + payload.len());
    buf.push(MessageType::MatchRemoteKeysResponse as u8);
    buf.extend_from_slice(&payload);
    buf
}

pub fn encode_register_remote_keys(keys: &[RemoteKey]) -> Vec<u8> {
    let payload = bincode::serde::encode_to_vec(keys, bincode::config::standard())
        .expect("Failed to serialize RemoteKeys");
    let mut buf = Vec::with_capacity(1 + payload.len());
    buf.push(MessageType::RegisterRemoteKeys as u8);
    buf.extend_from_slice(&payload);
    buf
}

pub fn decode_message_type(data: &[u8]) -> Option<MessageType> {
    data.first().and_then(|&b| MessageType::try_from(b).ok())
}

pub fn decode_register(data: &[u8]) -> Option<(BucketId, Vec<(SequenceHash, ObjectKey)>)> {
    // Header: 1 type + 8 bucket_id + 2 count = 11 bytes
    if data.len() < 11 || data[0] != MessageType::Register as u8 {
        return None;
    }
    let bucket_id = u64::from_le_bytes(data[1..9].try_into().ok()?);
    let count = u16::from_le_bytes(data[9..11].try_into().ok()?) as usize;
    // Entry: 8 hash + 8 key = 16 bytes
    if data.len() != 11 + count * WireEntry::SIZE {
        return None;
    }

    let mut entries = Vec::with_capacity(count);
    for i in 0..count {
        let offset = 11 + i * WireEntry::SIZE;
        let hash = u64::from_le_bytes(data[offset..offset + 8].try_into().ok()?);
        let key = u64::from_le_bytes(data[offset + 8..offset + 16].try_into().ok()?);
        entries.push((hash, key));
    }
    Some((bucket_id, entries))
}

pub fn decode_can_offload(data: &[u8]) -> Option<(BucketId, Vec<SequenceHash>)> {
    if data.len() < 11 || data[0] != MessageType::CanOffload as u8 {
        return None;
    }
    let bucket_id = u64::from_le_bytes(data[1..9].try_into().ok()?);
    let hashes = decode_hash_list(&data[9..])?;
    Some((bucket_id, hashes))
}

pub fn decode_match_sequence(data: &[u8]) -> Option<(BucketId, Vec<SequenceHash>)> {
    if data.len() < 11 || data[0] != MessageType::MatchSequence as u8 {
        return None;
    }
    let bucket_id = u64::from_le_bytes(data[1..9].try_into().ok()?);
    let hashes = decode_hash_list(&data[9..])?;
    Some((bucket_id, hashes))
}

pub fn decode_can_offload_response_v2(data: &[u8]) -> Option<Vec<OffloadStatus>> {
    if data.len() < 3 || data[0] != MessageType::CanOffloadResponse as u8 {
        return None;
    }
    let count = u16::from_le_bytes(data[1..3].try_into().ok()?) as usize;

    if data.len() == 3 + count {
        let mut result = Vec::with_capacity(count);
        for i in 0..count {
            let status = OffloadStatus::try_from(data[3 + i]).ok()?;
            result.push(status);
        }
        return Some(result);
    }

    // Legacy bitmap format for backwards compatibility
    let bitmap_bytes = (count + 7) / 8;
    if data.len() == 3 + bitmap_bytes {
        let mut result = Vec::with_capacity(count);
        for i in 0..count {
            let byte_idx = i / 8;
            let bit_idx = i % 8;
            let can_offload = (data[3 + byte_idx] >> bit_idx) & 1 == 1;
            result.push(if can_offload {
                OffloadStatus::Granted
            } else {
                OffloadStatus::AlreadyStored
            });
        }
        return Some(result);
    }

    None
}

pub fn decode_can_offload_response(data: &[u8]) -> Option<Vec<bool>> {
    let statuses = decode_can_offload_response_v2(data)?;
    Some(
        statuses
            .into_iter()
            .map(|s| s == OffloadStatus::Granted)
            .collect(),
    )
}

pub fn decode_match_response(data: &[u8]) -> Option<Vec<(SequenceHash, ObjectKey)>> {
    if data.len() < 3 || data[0] != MessageType::MatchResponse as u8 {
        return None;
    }
    let count = u16::from_le_bytes(data[1..3].try_into().ok()?) as usize;
    if data.len() != 3 + count * WireEntry::SIZE {
        return None;
    }

    let mut entries = Vec::with_capacity(count);
    for i in 0..count {
        let offset = 3 + i * WireEntry::SIZE;
        let hash = u64::from_le_bytes(data[offset..offset + 8].try_into().ok()?);
        let key = u64::from_le_bytes(data[offset + 8..offset + 16].try_into().ok()?);
        entries.push((hash, key));
    }
    Some(entries)
}

fn decode_hash_list(data: &[u8]) -> Option<Vec<SequenceHash>> {
    if data.len() < 2 {
        return None;
    }
    let count = u16::from_le_bytes(data[0..2].try_into().ok()?) as usize;
    if data.len() != 2 + count * 8 {
        return None;
    }

    let mut hashes = Vec::with_capacity(count);
    for i in 0..count {
        let offset = 2 + i * 8;
        let hash = u64::from_le_bytes(data[offset..offset + 8].try_into().ok()?);
        hashes.push(hash);
    }
    Some(hashes)
}

pub fn decode_match_remote_keys(data: &[u8]) -> Option<Vec<RemoteKey>> {
    if data.len() < 2 || data[0] != MessageType::MatchRemoteKeys as u8 {
        return None;
    }
    bincode::serde::decode_from_slice(&data[1..], bincode::config::standard())
        .map(|(keys, _)| keys)
        .ok()
}

pub fn decode_match_remote_keys_response(data: &[u8]) -> Option<Vec<RemoteKey>> {
    if data.len() < 2 || data[0] != MessageType::MatchRemoteKeysResponse as u8 {
        return None;
    }
    bincode::serde::decode_from_slice(&data[1..], bincode::config::standard())
        .map(|(keys, _)| keys)
        .ok()
}

pub fn decode_register_remote_keys(data: &[u8]) -> Option<Vec<RemoteKey>> {
    if data.len() < 2 || data[0] != MessageType::RegisterRemoteKeys as u8 {
        return None;
    }
    bincode::serde::decode_from_slice(&data[1..], bincode::config::standard())
        .map(|(keys, _)| keys)
        .ok()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_wire_entry() {
        let entry = WireEntry::new(123456789, 987654321);
        let bytes = entry.to_bytes();
        assert_eq!(bytes.len(), WireEntry::SIZE);
        assert_eq!(bytes.len(), 16);

        let decoded = WireEntry::from_bytes(&bytes).unwrap();
        let seq_hash = decoded.sequence_hash;
        let obj_key = decoded.object_key;
        assert_eq!(seq_hash, 123456789);
        assert_eq!(obj_key, 987654321);
    }

    #[test]
    fn test_bucket_id_from_name() {
        let id1 = bucket_id_from_name("kv-cache-worker-0");
        let id2 = bucket_id_from_name("kv-cache-worker-1");
        let id3 = bucket_id_from_name("kv-cache-worker-0");

        assert_ne!(id1, id2);  // Different names = different IDs
        assert_eq!(id1, id3);  // Same name = same ID
    }

    #[test]
    fn test_register_roundtrip() {
        let bucket_id = 12345u64;
        let entries = vec![(100, 100), (200, 200), (300, 300)];
        let encoded = encode_register(bucket_id, &entries);

        // header(11) = 1 type + 8 bucket_id + 2 count, entries = 3 * 16
        assert_eq!(encoded.len(), 11 + 3 * 16);
        assert_eq!(encoded[0], MessageType::Register as u8);

        let (decoded_bucket, decoded_entries) = decode_register(&encoded).unwrap();
        assert_eq!(decoded_bucket, bucket_id);
        assert_eq!(decoded_entries, entries);
    }

    #[test]
    fn test_can_offload_roundtrip() {
        let bucket_id = 67890u64;
        let hashes = vec![100, 200, 300, 400, 500];
        let encoded = encode_can_offload(bucket_id, &hashes);

        // header(11) = 1 type + 8 bucket_id + 2 count, hashes = 5 * 8
        assert_eq!(encoded.len(), 11 + 5 * 8);
        assert_eq!(encoded[0], MessageType::CanOffload as u8);

        let (decoded_bucket, decoded_hashes) = decode_can_offload(&encoded).unwrap();
        assert_eq!(decoded_bucket, bucket_id);
        assert_eq!(decoded_hashes, hashes);
    }

    #[test]
    fn test_can_offload_response_roundtrip() {
        for count in [1, 7, 8, 9, 15, 16, 17, 100] {
            let statuses: Vec<OffloadStatus> = (0..count)
                .map(|i| {
                    if i % 3 == 0 {
                        OffloadStatus::Granted
                    } else if i % 3 == 1 {
                        OffloadStatus::AlreadyStored
                    } else {
                        OffloadStatus::Leased
                    }
                })
                .collect();
            let encoded = encode_can_offload_response_v2(&statuses);

            let expected_size = 3 + count;
            assert_eq!(encoded.len(), expected_size, "count={}", count);
            assert_eq!(encoded[0], MessageType::CanOffloadResponse as u8);

            let decoded = decode_can_offload_response_v2(&encoded).unwrap();
            assert_eq!(decoded, statuses, "count={}", count);
        }
    }

    #[test]
    fn test_can_offload_response_bool_compat() {
        let can_offload: Vec<bool> = vec![true, false, true, true, false];
        let encoded = encode_can_offload_response(&can_offload);
        let decoded = decode_can_offload_response(&encoded).unwrap();
        assert_eq!(decoded, can_offload);
    }

    #[test]
    fn test_match_sequence_roundtrip() {
        let bucket_id = 11111u64;
        let hashes = vec![100, 200, 300];
        let encoded = encode_match_sequence(bucket_id, &hashes);

        assert_eq!(encoded[0], MessageType::MatchSequence as u8);

        let (decoded_bucket, decoded_hashes) = decode_match_sequence(&encoded).unwrap();
        assert_eq!(decoded_bucket, bucket_id);
        assert_eq!(decoded_hashes, hashes);
    }

    #[test]
    fn test_match_response_roundtrip() {
        let entries = vec![(100, 100), (200, 200)];
        let encoded = encode_match_response(&entries);

        assert_eq!(encoded[0], MessageType::MatchResponse as u8);

        let decoded = decode_match_response(&encoded).unwrap();
        assert_eq!(decoded, entries);
    }

    #[test]
    fn test_empty_messages() {
        let bucket_id = 0u64;

        // Empty register
        let encoded = encode_register(bucket_id, &[]);
        assert_eq!(encoded.len(), 11); // 1 type + 8 bucket_id + 2 count
        let (_, decoded) = decode_register(&encoded).unwrap();
        assert!(decoded.is_empty());

        // Empty can_offload
        let encoded = encode_can_offload(bucket_id, &[]);
        let (_, decoded) = decode_can_offload(&encoded).unwrap();
        assert!(decoded.is_empty());

        // Empty response
        let encoded = encode_can_offload_response(&[]);
        let decoded = decode_can_offload_response(&encoded).unwrap();
        assert!(decoded.is_empty());
    }

    #[test]
    fn test_message_sizes() {
        let bucket_id = 0u64;
        let hashes: Vec<_> = (0..100).collect();
        let entries: Vec<_> = hashes.iter().map(|&h| (h, h)).collect();

        // register(100) = 11 + 100*16 = 1611 bytes
        assert_eq!(encode_register(bucket_id, &entries).len(), 11 + 100 * 16);

        // can_offload(100) = 11 + 100*8 = 811 bytes
        assert_eq!(encode_can_offload(bucket_id, &hashes).len(), 11 + 100 * 8);

        // can_offload_response(100) = 3 + 100 = 103 bytes
        let statuses: Vec<OffloadStatus> = (0..100).map(|_| OffloadStatus::Granted).collect();
        assert_eq!(encode_can_offload_response_v2(&statuses).len(), 103);

        // match_response(100) = 3 + 100*16 = 1603 bytes
        assert_eq!(encode_match_response(&entries).len(), 3 + 100 * 16);
    }

    #[test]
    fn test_invalid_messages() {
        // Too short
        assert!(decode_register(&[]).is_none());
        assert!(decode_register(&[1]).is_none());
        assert!(decode_register(&[1, 0, 0, 0, 0]).is_none());

        // Wrong type
        assert!(decode_register(&[2, 0, 0, 0, 0, 0, 0]).is_none());
        assert!(decode_can_offload(&[1, 0, 0, 0, 0, 0, 0]).is_none());
    }
}

