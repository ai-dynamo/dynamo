// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

/// TCP request plane protocol message with endpoint routing and trace context
///
/// Wire format:
/// - endpoint_path_len: u16 (big-endian)
/// - endpoint_path: UTF-8 string
/// - traceparent_len: u16 (big-endian, 0 if None)
/// - traceparent: UTF-8 string (optional)
/// - tracestate_len: u16 (big-endian, 0 if None)
/// - tracestate: UTF-8 string (optional)
/// - x_request_id_len: u16 (big-endian, 0 if None)
/// - x_request_id: UTF-8 string (optional)
/// - x_dynamo_request_id_len: u16 (big-endian, 0 if None)
/// - x_dynamo_request_id: UTF-8 string (optional)
/// - payload_len: u32 (big-endian)
/// - payload: bytes
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TcpRequestMessage {
    pub endpoint_path: String,
    /// W3C traceparent header for distributed tracing
    pub traceparent: Option<String>,
    /// W3C tracestate header for vendor-specific trace data
    pub tracestate: Option<String>,
    /// Custom request ID header
    pub x_request_id: Option<String>,
    /// Dynamo-specific request ID header
    pub x_dynamo_request_id: Option<String>,
    pub payload: Bytes,
}

impl TcpRequestMessage {
    /// Create a new TCP request message without trace context (backwards compatible)
    pub fn new(endpoint_path: String, payload: Bytes) -> Self {
        Self {
            endpoint_path,
            traceparent: None,
            tracestate: None,
            x_request_id: None,
            x_dynamo_request_id: None,
            payload,
        }
    }

    /// Create a new TCP request message with trace context
    pub fn with_trace_context(
        endpoint_path: String,
        traceparent: Option<String>,
        tracestate: Option<String>,
        x_request_id: Option<String>,
        x_dynamo_request_id: Option<String>,
        payload: Bytes,
    ) -> Self {
        Self {
            endpoint_path,
            traceparent,
            tracestate,
            x_request_id,
            x_dynamo_request_id,
            payload,
        }
    }

    /// Helper to encode an optional string field: writes length (u16) and content
    fn encode_optional_string(buf: &mut BytesMut, value: &Option<String>) {
        match value {
            Some(s) => {
                let bytes = s.as_bytes();
                buf.put_u16(bytes.len() as u16);
                buf.put_slice(bytes);
            }
            None => {
                buf.put_u16(0);
            }
        }
    }

    /// Helper to decode an optional string field: reads length (u16) and content
    fn decode_optional_string(bytes: &Bytes, offset: &mut usize) -> Result<Option<String>, std::io::Error> {
        if bytes.len() < *offset + 2 {
            return Err(std::io::Error::new(
                std::io::ErrorKind::UnexpectedEof,
                "Not enough bytes for optional string length",
            ));
        }

        let len = u16::from_be_bytes([bytes[*offset], bytes[*offset + 1]]) as usize;
        *offset += 2;

        if len == 0 {
            return Ok(None);
        }

        if bytes.len() < *offset + len {
            return Err(std::io::Error::new(
                std::io::ErrorKind::UnexpectedEof,
                "Not enough bytes for optional string content",
            ));
        }

        let s = String::from_utf8(bytes[*offset..*offset + len].to_vec()).map_err(|e| {
            std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!("Invalid UTF-8: {}", e),
            )
        })?;
        *offset += len;

        Ok(Some(s))
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

        // Calculate optional header sizes
        let traceparent_len = self.traceparent.as_ref().map_or(0, |s| s.len());
        let tracestate_len = self.tracestate.as_ref().map_or(0, |s| s.len());
        let x_request_id_len = self.x_request_id.as_ref().map_or(0, |s| s.len());
        let x_dynamo_request_id_len = self.x_dynamo_request_id.as_ref().map_or(0, |s| s.len());

        // Total size: endpoint (2 + len) + 4 optional headers (2 + len each) + payload (4 + len)
        let total_size = 2 + endpoint_len
            + 2 + traceparent_len
            + 2 + tracestate_len
            + 2 + x_request_id_len
            + 2 + x_dynamo_request_id_len
            + 4 + self.payload.len();

        // Use BytesMut for efficient buffer building
        let mut buf = BytesMut::with_capacity(total_size);

        // Write endpoint path length (2 bytes) and content
        buf.put_u16(endpoint_len as u16);
        buf.put_slice(endpoint_bytes);

        // Write optional trace headers
        Self::encode_optional_string(&mut buf, &self.traceparent);
        Self::encode_optional_string(&mut buf, &self.tracestate);
        Self::encode_optional_string(&mut buf, &self.x_request_id);
        Self::encode_optional_string(&mut buf, &self.x_dynamo_request_id);

        // Write payload length (4 bytes) and content
        buf.put_u32(self.payload.len() as u32);
        buf.put_slice(&self.payload);

        // Zero-copy conversion to Bytes
        Ok(buf.freeze())
    }

    /// Decode message from bytes (for backward compatibility, zero-copy when possible)
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

        // Read optional trace headers
        let traceparent = Self::decode_optional_string(bytes, &mut offset)?;
        let tracestate = Self::decode_optional_string(bytes, &mut offset)?;
        let x_request_id = Self::decode_optional_string(bytes, &mut offset)?;
        let x_dynamo_request_id = Self::decode_optional_string(bytes, &mut offset)?;

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
            traceparent,
            tracestate,
            x_request_id,
            x_dynamo_request_id,
            payload,
        })
    }
}

/// Codec for encoding/decoding TcpRequestMessage
/// Supports max_message_size enforcement
#[derive(Clone, Default)]
pub struct TcpRequestCodec {
    max_message_size: Option<usize>,
}

impl TcpRequestCodec {
    pub fn new(max_message_size: Option<usize>) -> Self {
        Self { max_message_size }
    }
}

impl Decoder for TcpRequestCodec {
    type Item = TcpRequestMessage;
    type Error = std::io::Error;

    fn decode(&mut self, src: &mut BytesMut) -> Result<Option<Self::Item>, Self::Error> {
        // Need at least 2 bytes for endpoint_path_len
        if src.len() < 2 {
            return Ok(None);
        }

        // Peek at endpoint path length without consuming
        let endpoint_len = u16::from_be_bytes([src[0], src[1]]) as usize;

        // Minimum header size: endpoint (2 + len) + 4 optional headers (2 each) + payload_len (4)
        // = 2 + endpoint_len + 8 + 4 = 14 + endpoint_len (without optional header contents)
        let min_header_size = 2 + endpoint_len + 8 + 4;

        if src.len() < min_header_size {
            return Ok(None);
        }

        // Peek at optional header lengths to calculate total header size
        let mut peek_offset = 2 + endpoint_len;

        // Read 4 optional header lengths
        let traceparent_len = u16::from_be_bytes([src[peek_offset], src[peek_offset + 1]]) as usize;
        peek_offset += 2;

        if src.len() < peek_offset + traceparent_len + 2 {
            return Ok(None);
        }
        peek_offset += traceparent_len;

        let tracestate_len = u16::from_be_bytes([src[peek_offset], src[peek_offset + 1]]) as usize;
        peek_offset += 2;

        if src.len() < peek_offset + tracestate_len + 2 {
            return Ok(None);
        }
        peek_offset += tracestate_len;

        let x_request_id_len = u16::from_be_bytes([src[peek_offset], src[peek_offset + 1]]) as usize;
        peek_offset += 2;

        if src.len() < peek_offset + x_request_id_len + 2 {
            return Ok(None);
        }
        peek_offset += x_request_id_len;

        let x_dynamo_request_id_len = u16::from_be_bytes([src[peek_offset], src[peek_offset + 1]]) as usize;
        peek_offset += 2;

        if src.len() < peek_offset + x_dynamo_request_id_len + 4 {
            return Ok(None);
        }
        peek_offset += x_dynamo_request_id_len;

        // Peek at payload length
        let payload_len = u32::from_be_bytes([
            src[peek_offset],
            src[peek_offset + 1],
            src[peek_offset + 2],
            src[peek_offset + 3],
        ]) as usize;

        let total_len = peek_offset + 4 + payload_len;

        // Check max message size
        if let Some(max_size) = self.max_message_size
            && total_len > max_size
        {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!(
                    "Request too large: {} bytes (max: {} bytes)",
                    total_len, max_size
                ),
            ));
        }

        // Check if we have the full message
        if src.len() < total_len {
            return Ok(None);
        }

        // We have a complete message - use the standalone decode for consistency
        let msg_bytes = src.split_to(total_len).freeze();
        TcpRequestMessage::decode(&msg_bytes).map(Some)
    }
}

impl Encoder<TcpRequestMessage> for TcpRequestCodec {
    type Error = std::io::Error;

    fn encode(&mut self, item: TcpRequestMessage, dst: &mut BytesMut) -> Result<(), Self::Error> {
        // Use the standalone encode method for consistency
        let encoded = item.encode()?;

        // Check max message size
        if let Some(max_size) = self.max_message_size
            && encoded.len() > max_size
        {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                format!(
                    "Request too large: {} bytes (max: {} bytes)",
                    encoded.len(),
                    max_size
                ),
            ));
        }

        // Reserve space and copy encoded data
        dst.reserve(encoded.len());
        dst.put_slice(&encoded);

        Ok(())
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

    /// Encode response to bytes (for backward compatibility)
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

    /// Decode response from bytes (for backward compatibility, zero-copy when possible)
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

/// Codec for encoding/decoding TcpResponseMessage
/// Supports max_message_size enforcement
#[derive(Clone, Default)]
pub struct TcpResponseCodec {
    max_message_size: Option<usize>,
}

impl TcpResponseCodec {
    pub fn new(max_message_size: Option<usize>) -> Self {
        Self { max_message_size }
    }
}

impl Decoder for TcpResponseCodec {
    type Item = TcpResponseMessage;
    type Error = std::io::Error;

    fn decode(&mut self, src: &mut BytesMut) -> Result<Option<Self::Item>, Self::Error> {
        // Need at least 4 bytes for length
        if src.len() < 4 {
            return Ok(None);
        }

        // Peek at message length without consuming
        let data_len = u32::from_be_bytes([src[0], src[1], src[2], src[3]]) as usize;
        let total_len = 4 + data_len;

        // Check max message size
        if let Some(max_size) = self.max_message_size
            && total_len > max_size
        {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!(
                    "Response too large: {} bytes (max: {} bytes)",
                    total_len, max_size
                ),
            ));
        }

        // Check if we have the full message
        if src.len() < total_len {
            return Ok(None);
        }

        // Advance past the length prefix
        src.advance(4);

        // Read data
        let data = src.split_to(data_len).freeze();

        Ok(Some(TcpResponseMessage { data }))
    }
}

impl Encoder<TcpResponseMessage> for TcpResponseCodec {
    type Error = std::io::Error;

    fn encode(&mut self, item: TcpResponseMessage, dst: &mut BytesMut) -> Result<(), Self::Error> {
        if item.data.len() > u32::MAX as usize {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                format!("Response too large: {} bytes", item.data.len()),
            ));
        }

        let total_len = 4 + item.data.len();

        // Check max message size
        if let Some(max_size) = self.max_message_size
            && total_len > max_size
        {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                format!(
                    "Response too large: {} bytes (max: {} bytes)",
                    total_len, max_size
                ),
            ));
        }

        // Reserve space
        dst.reserve(total_len);

        // Write length
        dst.put_u32(item.data.len() as u32);

        // Write data
        dst.put_slice(&item.data);

        Ok(())
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

    #[test]
    fn test_tcp_request_codec() {
        use tokio_util::codec::{Decoder, Encoder};

        let msg = TcpRequestMessage::new(
            "test.endpoint".to_string(),
            Bytes::from(vec![1, 2, 3, 4, 5]),
        );

        let mut codec = TcpRequestCodec::new(None);
        let mut buf = BytesMut::new();

        // Encode
        codec.encode(msg.clone(), &mut buf).unwrap();

        // Decode
        let decoded = codec.decode(&mut buf).unwrap().unwrap();
        assert_eq!(decoded, msg);
    }

    #[test]
    fn test_tcp_request_codec_partial() {
        use tokio_util::codec::Decoder;

        let msg = TcpRequestMessage::new(
            "test.endpoint".to_string(),
            Bytes::from(vec![1, 2, 3, 4, 5]),
        );

        let encoded = msg.encode().unwrap();
        let mut codec = TcpRequestCodec::new(None);

        // Feed partial data
        let mut buf = BytesMut::from(&encoded[..5]);
        assert!(codec.decode(&mut buf).unwrap().is_none());

        // Feed rest of data
        buf.extend_from_slice(&encoded[5..]);
        let decoded = codec.decode(&mut buf).unwrap().unwrap();
        assert_eq!(decoded, msg);
    }

    #[test]
    fn test_tcp_request_codec_max_size() {
        use tokio_util::codec::Encoder;

        let msg = TcpRequestMessage::new("test".to_string(), Bytes::from(vec![1, 2, 3, 4, 5]));

        let mut codec = TcpRequestCodec::new(Some(10)); // Too small
        let mut buf = BytesMut::new();

        let result = codec.encode(msg, &mut buf);
        assert!(result.is_err());
    }

    #[test]
    fn test_tcp_response_codec() {
        use tokio_util::codec::{Decoder, Encoder};

        let msg = TcpResponseMessage::new(Bytes::from(vec![1, 2, 3, 4, 5]));

        let mut codec = TcpResponseCodec::new(None);
        let mut buf = BytesMut::new();

        // Encode
        codec.encode(msg.clone(), &mut buf).unwrap();

        // Decode
        let decoded = codec.decode(&mut buf).unwrap().unwrap();
        assert_eq!(decoded, msg);
    }

    #[test]
    fn test_tcp_response_codec_partial() {
        use tokio_util::codec::Decoder;

        let msg = TcpResponseMessage::new(Bytes::from(vec![1, 2, 3, 4, 5]));

        let encoded = msg.encode().unwrap();
        let mut codec = TcpResponseCodec::new(None);

        // Feed partial data
        let mut buf = BytesMut::from(&encoded[..3]);
        assert!(codec.decode(&mut buf).unwrap().is_none());

        // Feed rest of data
        buf.extend_from_slice(&encoded[3..]);
        let decoded = codec.decode(&mut buf).unwrap().unwrap();
        assert_eq!(decoded, msg);
    }

    #[test]
    fn test_tcp_response_codec_max_size() {
        use tokio_util::codec::Encoder;

        let msg = TcpResponseMessage::new(Bytes::from(vec![1, 2, 3, 4, 5]));

        let mut codec = TcpResponseCodec::new(Some(5)); // Too small
        let mut buf = BytesMut::new();

        let result = codec.encode(msg, &mut buf);
        assert!(result.is_err());
    }

    /// Demonstrates how framed codec enables testability without actual TCP connections
    #[tokio::test]
    async fn test_framed_codec_integration() {
        use futures::{SinkExt, StreamExt};
        use std::io::Cursor;
        use tokio_util::codec::{FramedRead, FramedWrite};

        // Simulate a duplex connection using in-memory buffer
        let mut buffer = Vec::new();

        // Writer side: encode requests
        {
            let cursor = Cursor::new(&mut buffer);
            let mut writer = FramedWrite::new(cursor, TcpRequestCodec::new(None));

            let msg1 = TcpRequestMessage::new("endpoint1".to_string(), Bytes::from("data1"));
            let msg2 = TcpRequestMessage::new("endpoint2".to_string(), Bytes::from("data2"));

            writer.send(msg1).await.unwrap();
            writer.send(msg2).await.unwrap();
        }

        // Reader side: decode requests
        {
            let cursor = Cursor::new(&buffer[..]);
            let mut reader = FramedRead::new(cursor, TcpRequestCodec::new(None));

            let decoded1 = reader.next().await.unwrap().unwrap();
            assert_eq!(decoded1.endpoint_path, "endpoint1");
            assert_eq!(decoded1.payload, Bytes::from("data1"));

            let decoded2 = reader.next().await.unwrap().unwrap();
            assert_eq!(decoded2.endpoint_path, "endpoint2");
            assert_eq!(decoded2.payload, Bytes::from("data2"));
        }
    }

    /// Demonstrates testing partial message handling
    #[tokio::test]
    async fn test_framed_codec_partial_messages() {
        use futures::StreamExt;
        use std::io::Cursor;
        use tokio_util::codec::FramedRead;

        // Create a message and encode it
        let msg = TcpRequestMessage::new("test".to_string(), Bytes::from("hello"));
        let encoded = msg.encode().unwrap();

        // Split the encoded message into chunks
        let chunk1 = &encoded[..5];
        let chunk2 = &encoded[5..];

        // Create a buffer that simulates receiving data in chunks
        let mut full_buffer = Vec::new();
        full_buffer.extend_from_slice(chunk1);

        // Reader can't decode yet (partial data)
        {
            let cursor = Cursor::new(&full_buffer[..]);
            let _reader = FramedRead::new(cursor, TcpRequestCodec::new(None));
            // In real async, this would return Ok(None) and wait for more data
            // For Cursor, it returns None at EOF
        }

        // Add the rest of the data
        full_buffer.extend_from_slice(chunk2);

        // Now decoding succeeds
        {
            let cursor = Cursor::new(&full_buffer[..]);
            let mut reader = FramedRead::new(cursor, TcpRequestCodec::new(None));

            let decoded = reader.next().await.unwrap().unwrap();
            assert_eq!(decoded.endpoint_path, "test");
            assert_eq!(decoded.payload, Bytes::from("hello"));
        }
    }

    #[test]
    fn test_tcp_request_with_trace_context() {
        let msg = TcpRequestMessage::with_trace_context(
            "test.endpoint".to_string(),
            Some("00-4bf92f3577b34da6a3ce929d0e0e4736-00f067aa0ba902b7-01".to_string()),
            Some("congo=t61rcWkgMzE".to_string()),
            Some("req-12345".to_string()),
            Some("550e8400-e29b-41d4-a716-446655440000".to_string()),
            Bytes::from(vec![1, 2, 3, 4, 5]),
        );

        let encoded = msg.encode().unwrap();
        let decoded = TcpRequestMessage::decode(&encoded).unwrap();

        assert_eq!(decoded.endpoint_path, "test.endpoint");
        assert_eq!(
            decoded.traceparent,
            Some("00-4bf92f3577b34da6a3ce929d0e0e4736-00f067aa0ba902b7-01".to_string())
        );
        assert_eq!(decoded.tracestate, Some("congo=t61rcWkgMzE".to_string()));
        assert_eq!(decoded.x_request_id, Some("req-12345".to_string()));
        assert_eq!(
            decoded.x_dynamo_request_id,
            Some("550e8400-e29b-41d4-a716-446655440000".to_string())
        );
        assert_eq!(decoded.payload, Bytes::from(vec![1, 2, 3, 4, 5]));
    }

    #[test]
    fn test_tcp_request_with_partial_trace_context() {
        // Test with only some trace headers set
        let msg = TcpRequestMessage::with_trace_context(
            "test.endpoint".to_string(),
            Some("00-4bf92f3577b34da6a3ce929d0e0e4736-00f067aa0ba902b7-01".to_string()),
            None, // No tracestate
            Some("req-12345".to_string()),
            None, // No dynamo request id
            Bytes::from("test payload"),
        );

        let encoded = msg.encode().unwrap();
        let decoded = TcpRequestMessage::decode(&encoded).unwrap();

        assert_eq!(decoded.endpoint_path, "test.endpoint");
        assert_eq!(
            decoded.traceparent,
            Some("00-4bf92f3577b34da6a3ce929d0e0e4736-00f067aa0ba902b7-01".to_string())
        );
        assert_eq!(decoded.tracestate, None);
        assert_eq!(decoded.x_request_id, Some("req-12345".to_string()));
        assert_eq!(decoded.x_dynamo_request_id, None);
        assert_eq!(decoded.payload, Bytes::from("test payload"));
    }

    #[test]
    fn test_tcp_request_codec_with_trace_context() {
        use tokio_util::codec::{Decoder, Encoder};

        let msg = TcpRequestMessage::with_trace_context(
            "test.endpoint".to_string(),
            Some("00-4bf92f3577b34da6a3ce929d0e0e4736-00f067aa0ba902b7-01".to_string()),
            Some("congo=t61rcWkgMzE".to_string()),
            Some("req-12345".to_string()),
            Some("550e8400-e29b-41d4-a716-446655440000".to_string()),
            Bytes::from(vec![1, 2, 3, 4, 5]),
        );

        let mut codec = TcpRequestCodec::new(None);
        let mut buf = BytesMut::new();

        // Encode
        codec.encode(msg.clone(), &mut buf).unwrap();

        // Decode
        let decoded = codec.decode(&mut buf).unwrap().unwrap();
        assert_eq!(decoded, msg);
    }
}
