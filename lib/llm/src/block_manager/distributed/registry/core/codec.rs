// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Codec trait and implementations.

use std::marker::PhantomData;

use super::error::{RegistryError, RegistryResult};
use super::key::RegistryKey;
use super::metadata::RegistryMetadata;
use super::value::RegistryValue;

/// Current protocol version.
pub const PROTOCOL_VERSION: u8 = 1;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum MessageType {
    Register = 1,
    CanOffload = 2,
    Match = 3,
    CanOffloadResponse = 4,
    MatchResponse = 5,
}

impl TryFrom<u8> for MessageType {
    type Error = ();

    fn try_from(value: u8) -> Result<Self, Self::Error> {
        match value {
            1 => Ok(Self::Register),
            2 => Ok(Self::CanOffload),
            3 => Ok(Self::Match),
            4 => Ok(Self::CanOffloadResponse),
            5 => Ok(Self::MatchResponse),
            _ => Err(()),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum OffloadStatus {
    Granted = 0,
    AlreadyStored = 1,
    Leased = 2,
}

impl TryFrom<u8> for OffloadStatus {
    type Error = ();

    fn try_from(value: u8) -> Result<Self, Self::Error> {
        match value {
            0 => Ok(Self::Granted),
            1 => Ok(Self::AlreadyStored),
            2 => Ok(Self::Leased),
            _ => Err(()),
        }
    }
}

#[derive(Debug, Clone)]
pub enum QueryType<K> {
    CanOffload(Vec<K>),
    Match(Vec<K>),
}

#[derive(Debug, Clone)]
pub enum ResponseType<K, V, M> {
    CanOffload(Vec<OffloadStatus>),
    Match(Vec<(K, V, M)>),
}

pub trait RegistryCodec<K, V, M>: Send + Sync
where
    K: RegistryKey,
    V: RegistryValue,
    M: RegistryMetadata,
{
    fn encode_register(&self, entries: &[(K, V, M)], buf: &mut Vec<u8>) -> RegistryResult<()>;
    fn decode_register(&self, data: &[u8]) -> Option<Vec<(K, V, M)>>;

    fn encode_query(&self, query: &QueryType<K>, buf: &mut Vec<u8>) -> RegistryResult<()>;
    fn decode_query(&self, data: &[u8]) -> Option<QueryType<K>>;

    fn encode_response(&self, response: &ResponseType<K, V, M>, buf: &mut Vec<u8>) -> RegistryResult<()>;
    fn decode_response(&self, data: &[u8]) -> Option<ResponseType<K, V, M>>;

    /// Decode with detailed error information.
    fn decode_register_result(&self, data: &[u8]) -> RegistryResult<Vec<(K, V, M)>> {
        self.decode_register(data)
            .ok_or_else(|| RegistryError::DecodeError {
                context: "register",
                expected: "valid register message".to_string(),
                got: format!("{} bytes", data.len()),
            })
    }

    /// Decode query with detailed error information.
    fn decode_query_result(&self, data: &[u8]) -> RegistryResult<QueryType<K>> {
        self.decode_query(data)
            .ok_or_else(|| RegistryError::DecodeError {
                context: "query",
                expected: "valid query message".to_string(),
                got: format!("{} bytes", data.len()),
            })
    }

    /// Decode response with detailed error information.
    fn decode_response_result(&self, data: &[u8]) -> RegistryResult<ResponseType<K, V, M>> {
        self.decode_response(data)
            .ok_or_else(|| RegistryError::DecodeError {
                context: "response",
                expected: "valid response message".to_string(),
                got: format!("{} bytes", data.len()),
            })
    }
}

/// Binary codec for fixed-size keys and values.
///
/// Wire format (versioned):
/// ```text
/// [version:1][type:1][count:4][...entries...]
/// ```
pub struct BinaryCodec<K, V, M> {
    _phantom: PhantomData<(K, V, M)>,
}

impl<K, V, M> BinaryCodec<K, V, M> {
    pub fn new() -> Self {
        Self {
            _phantom: PhantomData,
        }
    }

    /// Check protocol version in message header.
    fn check_version(data: &[u8]) -> Option<()> {
        if data.len() < 2 {
            return None;
        }
        let version = data[0];
        if version != PROTOCOL_VERSION {
            tracing::warn!(
                expected = PROTOCOL_VERSION,
                got = version,
                "Protocol version mismatch"
            );
            return None;
        }
        Some(())
    }

    /// Version byte offset (always 1 since version byte is required).
    #[inline]
    fn header_offset(_data: &[u8]) -> usize {
        1
    }
}

impl<K, V, M> Default for BinaryCodec<K, V, M> {
    fn default() -> Self {
        Self::new()
    }
}

impl<K, V, M> RegistryCodec<K, V, M> for BinaryCodec<K, V, M>
where
    K: RegistryKey,
    V: RegistryValue,
    M: RegistryMetadata,
{
    fn encode_register(&self, entries: &[(K, V, M)], buf: &mut Vec<u8>) -> RegistryResult<()> {
        if entries.len() > u32::MAX as usize {
            return Err(RegistryError::EncodeError {
                context: "entry count exceeds u32::MAX",
            });
        }
        buf.push(PROTOCOL_VERSION);
        buf.push(MessageType::Register as u8);
        buf.extend_from_slice(&(entries.len() as u32).to_le_bytes());
        for (k, v, m) in entries {
            let kb = k.to_bytes();
            let vb = v.to_bytes();
            let mb = m.to_bytes();
            if kb.len() > u16::MAX as usize {
                return Err(RegistryError::EncodeError {
                    context: "key size exceeds u16::MAX",
                });
            }
            if vb.len() > u16::MAX as usize {
                return Err(RegistryError::EncodeError {
                    context: "value size exceeds u16::MAX",
                });
            }
            if mb.len() > u16::MAX as usize {
                return Err(RegistryError::EncodeError {
                    context: "metadata size exceeds u16::MAX",
                });
            }
            buf.extend_from_slice(&(kb.len() as u16).to_le_bytes());
            buf.extend_from_slice(&kb);
            buf.extend_from_slice(&(vb.len() as u16).to_le_bytes());
            buf.extend_from_slice(&vb);
            buf.extend_from_slice(&(mb.len() as u16).to_le_bytes());
            buf.extend_from_slice(&mb);
        }
        Ok(())
    }

    fn decode_register(&self, data: &[u8]) -> Option<Vec<(K, V, M)>> {
        Self::check_version(data)?;
        let offset = Self::header_offset(data);
        let data = &data[offset..];

        if data.len() < 5 || data[0] != MessageType::Register as u8 {
            return None;
        }
        let count = u32::from_le_bytes(data[1..5].try_into().ok()?) as usize;
        let mut entries = Vec::with_capacity(count);
        let mut pos = 5;

        for _ in 0..count {
            if pos + 2 > data.len() {
                return None;
            }
            let klen = u16::from_le_bytes(data[pos..pos + 2].try_into().ok()?) as usize;
            pos += 2;
            if pos + klen > data.len() {
                return None;
            }
            let k = K::from_bytes(&data[pos..pos + klen])?;
            pos += klen;

            if pos + 2 > data.len() {
                return None;
            }
            let vlen = u16::from_le_bytes(data[pos..pos + 2].try_into().ok()?) as usize;
            pos += 2;
            if pos + vlen > data.len() {
                return None;
            }
            let v = V::from_bytes(&data[pos..pos + vlen])?;
            pos += vlen;

            if pos + 2 > data.len() {
                return None;
            }
            let mlen = u16::from_le_bytes(data[pos..pos + 2].try_into().ok()?) as usize;
            pos += 2;
            if pos + mlen > data.len() {
                return None;
            }
            let m = M::from_bytes(&data[pos..pos + mlen])?;
            pos += mlen;

            entries.push((k, v, m));
        }

        Some(entries)
    }

    fn encode_query(&self, query: &QueryType<K>, buf: &mut Vec<u8>) -> RegistryResult<()> {
        buf.push(PROTOCOL_VERSION);
        match query {
            QueryType::CanOffload(keys) | QueryType::Match(keys) => {
                if keys.len() > u32::MAX as usize {
                    return Err(RegistryError::EncodeError {
                        context: "key count exceeds u32::MAX",
                    });
                }
                let msg_type = match query {
                    QueryType::CanOffload(_) => MessageType::CanOffload,
                    QueryType::Match(_) => MessageType::Match,
                };
                buf.push(msg_type as u8);
                buf.extend_from_slice(&(keys.len() as u32).to_le_bytes());
                for k in keys {
                    let kb = k.to_bytes();
                    if kb.len() > u16::MAX as usize {
                        return Err(RegistryError::EncodeError {
                            context: "key size exceeds u16::MAX",
                        });
                    }
                    buf.extend_from_slice(&(kb.len() as u16).to_le_bytes());
                    buf.extend_from_slice(&kb);
                }
            }
        }
        Ok(())
    }

    fn decode_query(&self, data: &[u8]) -> Option<QueryType<K>> {
        Self::check_version(data)?;
        let offset = Self::header_offset(data);
        let data = &data[offset..];

        if data.len() < 5 {
            return None;
        }
        let msg_type = MessageType::try_from(data[0]).ok()?;
        let count = u32::from_le_bytes(data[1..5].try_into().ok()?) as usize;
        let mut keys = Vec::with_capacity(count);
        let mut pos = 5;

        for _ in 0..count {
            if pos + 2 > data.len() {
                return None;
            }
            let klen = u16::from_le_bytes(data[pos..pos + 2].try_into().ok()?) as usize;
            pos += 2;
            if pos + klen > data.len() {
                return None;
            }
            keys.push(K::from_bytes(&data[pos..pos + klen])?);
            pos += klen;
        }

        match msg_type {
            MessageType::CanOffload => Some(QueryType::CanOffload(keys)),
            MessageType::Match => Some(QueryType::Match(keys)),
            _ => None,
        }
    }

    fn encode_response(&self, response: &ResponseType<K, V, M>, buf: &mut Vec<u8>) -> RegistryResult<()> {
        buf.push(PROTOCOL_VERSION);
        match response {
            ResponseType::CanOffload(statuses) => {
                if statuses.len() > u32::MAX as usize {
                    return Err(RegistryError::EncodeError {
                        context: "status count exceeds u32::MAX",
                    });
                }
                buf.push(MessageType::CanOffloadResponse as u8);
                buf.extend_from_slice(&(statuses.len() as u32).to_le_bytes());
                for s in statuses {
                    buf.push(*s as u8);
                }
            }
            ResponseType::Match(entries) => {
                if entries.len() > u32::MAX as usize {
                    return Err(RegistryError::EncodeError {
                        context: "entry count exceeds u32::MAX",
                    });
                }
                buf.push(MessageType::MatchResponse as u8);
                buf.extend_from_slice(&(entries.len() as u32).to_le_bytes());
                for (k, v, m) in entries {
                    let kb = k.to_bytes();
                    let vb = v.to_bytes();
                    let mb = m.to_bytes();
                    if kb.len() > u16::MAX as usize {
                        return Err(RegistryError::EncodeError {
                            context: "key size exceeds u16::MAX",
                        });
                    }
                    if vb.len() > u16::MAX as usize {
                        return Err(RegistryError::EncodeError {
                            context: "value size exceeds u16::MAX",
                        });
                    }
                    if mb.len() > u16::MAX as usize {
                        return Err(RegistryError::EncodeError {
                            context: "metadata size exceeds u16::MAX",
                        });
                    }
                    buf.extend_from_slice(&(kb.len() as u16).to_le_bytes());
                    buf.extend_from_slice(&kb);
                    buf.extend_from_slice(&(vb.len() as u16).to_le_bytes());
                    buf.extend_from_slice(&vb);
                    buf.extend_from_slice(&(mb.len() as u16).to_le_bytes());
                    buf.extend_from_slice(&mb);
                }
            }
        }
        Ok(())
    }

    fn decode_response(&self, data: &[u8]) -> Option<ResponseType<K, V, M>> {
        Self::check_version(data)?;
        let offset = Self::header_offset(data);
        let data = &data[offset..];

        if data.len() < 5 {
            return None;
        }
        let msg_type = MessageType::try_from(data[0]).ok()?;
        let count = u32::from_le_bytes(data[1..5].try_into().ok()?) as usize;

        match msg_type {
            MessageType::CanOffloadResponse => {
                if data.len() < 5 + count {
                    return None;
                }
                let statuses: Option<Vec<_>> = data[5..5 + count]
                    .iter()
                    .map(|&b| OffloadStatus::try_from(b).ok())
                    .collect();
                Some(ResponseType::CanOffload(statuses?))
            }
            MessageType::MatchResponse => {
                let mut entries = Vec::with_capacity(count);
                let mut pos = 5;

                for _ in 0..count {
                    if pos + 2 > data.len() {
                        return None;
                    }
                    let klen = u16::from_le_bytes(data[pos..pos + 2].try_into().ok()?) as usize;
                    pos += 2;
                    if pos + klen > data.len() {
                        return None;
                    }
                    let k = K::from_bytes(&data[pos..pos + klen])?;
                    pos += klen;

                    if pos + 2 > data.len() {
                        return None;
                    }
                    let vlen = u16::from_le_bytes(data[pos..pos + 2].try_into().ok()?) as usize;
                    pos += 2;
                    if pos + vlen > data.len() {
                        return None;
                    }
                    let v = V::from_bytes(&data[pos..pos + vlen])?;
                    pos += vlen;

                    if pos + 2 > data.len() {
                        return None;
                    }
                    let mlen = u16::from_le_bytes(data[pos..pos + 2].try_into().ok()?) as usize;
                    pos += 2;
                    if pos + mlen > data.len() {
                        return None;
                    }
                    let m = M::from_bytes(&data[pos..pos + mlen])?;
                    pos += mlen;

                    entries.push((k, v, m));
                }

                Some(ResponseType::Match(entries))
            }
            _ => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::block_manager::distributed::registry::core::metadata::NoMetadata;

    #[test]
    fn test_register_roundtrip() {
        let codec: BinaryCodec<u64, u64, NoMetadata> = BinaryCodec::new();
        let entries = vec![(1u64, 100u64, NoMetadata), (2, 200, NoMetadata)];

        let mut buf = Vec::new();
        codec.encode_register(&entries, &mut buf).unwrap();

        // Check version byte is present
        assert_eq!(buf[0], PROTOCOL_VERSION);

        let decoded = codec.decode_register(&buf).unwrap();
        assert_eq!(decoded.len(), 2);
        assert_eq!(decoded[0].0, 1);
        assert_eq!(decoded[0].1, 100);
        assert_eq!(decoded[1].0, 2);
        assert_eq!(decoded[1].1, 200);
    }

    #[test]
    fn test_query_roundtrip() {
        let codec: BinaryCodec<u64, u64, NoMetadata> = BinaryCodec::new();

        let query = QueryType::CanOffload(vec![1u64, 2, 3]);
        let mut buf = Vec::new();
        codec.encode_query(&query, &mut buf).unwrap();

        // Check version byte is present
        assert_eq!(buf[0], PROTOCOL_VERSION);

        let decoded = codec.decode_query(&buf).unwrap();
        match decoded {
            QueryType::CanOffload(keys) => assert_eq!(keys, vec![1, 2, 3]),
            _ => panic!("wrong type"),
        }

        let query = QueryType::Match(vec![4u64, 5]);
        buf.clear();
        codec.encode_query(&query, &mut buf).unwrap();
        let decoded = codec.decode_query(&buf).unwrap();
        match decoded {
            QueryType::Match(keys) => assert_eq!(keys, vec![4, 5]),
            _ => panic!("wrong type"),
        }
    }

    #[test]
    fn test_response_roundtrip() {
        let codec: BinaryCodec<u64, u64, NoMetadata> = BinaryCodec::new();

        let response = ResponseType::CanOffload(vec![
            OffloadStatus::Granted,
            OffloadStatus::AlreadyStored,
            OffloadStatus::Leased,
        ]);
        let mut buf = Vec::new();
        codec.encode_response(&response, &mut buf).unwrap();

        // Check version byte is present
        assert_eq!(buf[0], PROTOCOL_VERSION);

        let decoded: ResponseType<u64, u64, NoMetadata> = codec.decode_response(&buf).unwrap();
        match decoded {
            ResponseType::CanOffload(statuses) => {
                assert_eq!(statuses.len(), 3);
                assert_eq!(statuses[0], OffloadStatus::Granted);
                assert_eq!(statuses[1], OffloadStatus::AlreadyStored);
                assert_eq!(statuses[2], OffloadStatus::Leased);
            }
            _ => panic!("wrong type"),
        }

        let response: ResponseType<u64, u64, NoMetadata> =
            ResponseType::Match(vec![(1, 100, NoMetadata), (2, 200, NoMetadata)]);
        buf.clear();
        codec.encode_response(&response, &mut buf).unwrap();
        let decoded: ResponseType<u64, u64, NoMetadata> = codec.decode_response(&buf).unwrap();
        match decoded {
            ResponseType::Match(entries) => {
                assert_eq!(entries.len(), 2);
                assert_eq!(entries[0].0, 1);
                assert_eq!(entries[0].1, 100);
            }
            _ => panic!("wrong type"),
        }
    }

    #[test]
    fn test_decode_result_error() {
        let codec: BinaryCodec<u64, u64, NoMetadata> = BinaryCodec::new();

        let result = codec.decode_register_result(&[]);
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(err.to_string().contains("register"));
    }
}
