// SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Codec Module
//!
//! Codec map structure into blobs of bytes and streams of bytes.
//!
//! In this module, we define three primary codec used to issue single, two-part or multi-part messages,
//! on a byte stream.

use bytes::Bytes;
use tokio_util::{
    bytes::{Buf, BufMut, BytesMut},
    codec::{Decoder, Encoder},
};

mod two_part;

pub use two_part::{TwoPartCodec, TwoPartMessage, TwoPartMessageType};

/// TCP request plane protocol message with endpoint routing
///
/// Wire format:
/// - endpoint_path_len: u16 (big-endian)
/// - endpoint_path: UTF-8 string
/// - payload_len: u32 (big-endian)
/// - payload: bytes
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TcpRequestMessage {
    pub endpoint_path: String,
    pub payload: Bytes,
}

impl TcpRequestMessage {
    pub fn new(endpoint_path: String, payload: Bytes) -> Self {
        Self {
            endpoint_path,
            payload,
        }
    }

    /// Encode message to bytes
    pub fn encode(&self) -> Result<Bytes, std::io::Error> {
        let endpoint_bytes = self.endpoint_path.as_bytes();
        let endpoint_len = endpoint_bytes.len();

        if endpoint_len > u16::MAX as usize {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                format!("Endpoint path too long: {} bytes", endpoint_len),
            ));
        }

        if self.payload.len() > u32::MAX as usize {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                format!("Payload too large: {} bytes", self.payload.len()),
            ));
        }

        // Use BytesMut for efficient buffer building
        let mut buf = BytesMut::with_capacity(2 + endpoint_len + 4 + self.payload.len());

        // Write endpoint path length (2 bytes)
        buf.put_u16(endpoint_len as u16);

        // Write endpoint path
        buf.put_slice(endpoint_bytes);

        // Write payload length (4 bytes)
        buf.put_u32(self.payload.len() as u32);

        // Write payload
        buf.put_slice(&self.payload);

        // Zero-copy conversion to Bytes
        Ok(buf.freeze())
    }

    /// Decode message from bytes (zero-copy when possible)
    pub fn decode(bytes: &Bytes) -> Result<Self, std::io::Error> {
        if bytes.len() < 2 {
            return Err(std::io::Error::new(
                std::io::ErrorKind::UnexpectedEof,
                "Not enough bytes for endpoint path length",
            ));
        }

        // Read endpoint path length (2 bytes)
        let endpoint_len = u16::from_be_bytes([bytes[0], bytes[1]]) as usize;
        let mut offset = 2;

        if bytes.len() < offset + endpoint_len {
            return Err(std::io::Error::new(
                std::io::ErrorKind::UnexpectedEof,
                "Not enough bytes for endpoint path",
            ));
        }

        // Read endpoint path (requires copy for UTF-8 validation)
        let endpoint_path = String::from_utf8(bytes[offset..offset + endpoint_len].to_vec())
            .map_err(|e| {
                std::io::Error::new(
                    std::io::ErrorKind::InvalidData,
                    format!("Invalid UTF-8: {}", e),
                )
            })?;
        offset += endpoint_len;

        if bytes.len() < offset + 4 {
            return Err(std::io::Error::new(
                std::io::ErrorKind::UnexpectedEof,
                "Not enough bytes for payload length",
            ));
        }

        // Read payload length (4 bytes)
        let payload_len = u32::from_be_bytes([
            bytes[offset],
            bytes[offset + 1],
            bytes[offset + 2],
            bytes[offset + 3],
        ]) as usize;
        offset += 4;

        if bytes.len() < offset + payload_len {
            return Err(std::io::Error::new(
                std::io::ErrorKind::UnexpectedEof,
                format!(
                    "Not enough bytes for payload: expected {}, got {}",
                    payload_len,
                    bytes.len() - offset
                ),
            ));
        }

        // Read payload (zero-copy slice)
        let payload = bytes.slice(offset..offset + payload_len);

        Ok(Self {
            endpoint_path,
            payload,
        })
    }
}

/// TCP response message (acknowledgment or error)
///
/// Wire format:
/// - length: u32 (big-endian)
/// - data: bytes
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TcpResponseMessage {
    pub data: Bytes,
}

impl TcpResponseMessage {
    pub fn new(data: Bytes) -> Self {
        Self { data }
    }

    pub fn empty() -> Self {
        Self { data: Bytes::new() }
    }

    /// Encode response to bytes
    pub fn encode(&self) -> Result<Bytes, std::io::Error> {
        if self.data.len() > u32::MAX as usize {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                format!("Response too large: {} bytes", self.data.len()),
            ));
        }

        // Use BytesMut for efficient buffer building
        let mut buf = BytesMut::with_capacity(4 + self.data.len());

        // Write length (4 bytes)
        buf.put_u32(self.data.len() as u32);

        // Write data
        buf.put_slice(&self.data);

        // Zero-copy conversion to Bytes
        Ok(buf.freeze())
    }

    /// Decode response from bytes (zero-copy when possible)
    pub fn decode(bytes: &Bytes) -> Result<Self, std::io::Error> {
        if bytes.len() < 4 {
            return Err(std::io::Error::new(
                std::io::ErrorKind::UnexpectedEof,
                "Not enough bytes for response length",
            ));
        }

        // Read length (4 bytes)
        let len = u32::from_be_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]) as usize;

        if bytes.len() < 4 + len {
            return Err(std::io::Error::new(
                std::io::ErrorKind::UnexpectedEof,
                format!(
                    "Not enough bytes for response: expected {}, got {}",
                    len,
                    bytes.len() - 4
                ),
            ));
        }

        // Read data (zero-copy slice)
        let data = bytes.slice(4..4 + len);

        Ok(Self { data })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tcp_request_encode_decode() {
        let msg = TcpRequestMessage::new(
            "test.endpoint".to_string(),
            Bytes::from(vec![1, 2, 3, 4, 5]),
        );

        let encoded = msg.encode().unwrap();
        let decoded = TcpRequestMessage::decode(&encoded).unwrap();

        assert_eq!(decoded, msg);
    }

    #[test]
    fn test_tcp_request_empty_payload() {
        let msg = TcpRequestMessage::new("test".to_string(), Bytes::new());

        let encoded = msg.encode().unwrap();
        let decoded = TcpRequestMessage::decode(&encoded).unwrap();

        assert_eq!(decoded, msg);
    }

    #[test]
    fn test_tcp_request_large_payload() {
        let payload = Bytes::from(vec![42u8; 1024 * 1024]); // 1MB
        let msg = TcpRequestMessage::new("large".to_string(), payload);

        let encoded = msg.encode().unwrap();
        let decoded = TcpRequestMessage::decode(&encoded).unwrap();

        assert_eq!(decoded, msg);
    }

    #[test]
    fn test_tcp_request_decode_truncated() {
        let msg = TcpRequestMessage::new("test".to_string(), Bytes::from(vec![1, 2, 3, 4, 5]));
        let encoded = msg.encode().unwrap();

        // Truncate the encoded message
        let truncated = encoded.slice(..encoded.len() - 2);
        let result = TcpRequestMessage::decode(&truncated);

        assert!(result.is_err());
    }

    #[test]
    fn test_tcp_response_encode_decode() {
        let msg = TcpResponseMessage::new(Bytes::from(vec![1, 2, 3, 4, 5]));

        let encoded = msg.encode().unwrap();
        let decoded = TcpResponseMessage::decode(&encoded).unwrap();

        assert_eq!(decoded, msg);
    }

    #[test]
    fn test_tcp_response_empty() {
        let msg = TcpResponseMessage::empty();

        let encoded = msg.encode().unwrap();
        let decoded = TcpResponseMessage::decode(&encoded).unwrap();

        assert_eq!(decoded, msg);
        assert_eq!(decoded.data.len(), 0);
    }

    #[test]
    fn test_tcp_response_decode_truncated() {
        let msg = TcpResponseMessage::new(Bytes::from(vec![1, 2, 3, 4, 5]));
        let encoded = msg.encode().unwrap();

        // Truncate the encoded message
        let truncated = encoded.slice(..3);
        let result = TcpResponseMessage::decode(&truncated);

        assert!(result.is_err());
    }

    #[test]
    fn test_tcp_request_unicode_endpoint() {
        let msg = TcpRequestMessage::new("тест.端点".to_string(), Bytes::from(vec![1, 2, 3]));

        let encoded = msg.encode().unwrap();
        let decoded = TcpRequestMessage::decode(&encoded).unwrap();

        assert_eq!(decoded, msg);
    }
}
