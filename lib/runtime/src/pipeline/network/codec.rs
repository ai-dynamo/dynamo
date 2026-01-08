// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Codec Module
//!
//! Codec map structure into blobs of bytes and streams of bytes.
//!
//! In this module, we define three primary codec used to issue single, two-part or multi-part messages,
//! on a byte stream.

use bytes::Bytes;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use tokio_util::{
    bytes::{Buf, BufMut, BytesMut},
    codec::{Decoder, Encoder},
};

mod two_part;

pub use two_part::{TwoPartCodec, TwoPartMessage, TwoPartMessageType};

/// TCP request headers including endpoint routing and OpenTelemetry tracing
/// Follows W3C Trace Context specification
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TcpRequestHeader {
    /// Endpoint path for routing
    pub endpoint_path: String,
    /// W3C traceparent: version-trace_id-parent_id-trace_flags
    #[serde(skip_serializing_if = "Option::is_none")]
    pub traceparent: Option<String>,
    /// W3C tracestate: vendor-specific trace state
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tracestate: Option<String>,
    /// Custom request ID
    #[serde(skip_serializing_if = "Option::is_none")]
    pub x_request_id: Option<String>,
}

impl TcpRequestHeader {
    pub fn new(endpoint_path: String) -> Self {
        Self {
            endpoint_path,
            traceparent: None,
            tracestate: None,
            x_request_id: None,
        }
    }

    /// Create TcpRequestHeader from endpoint path and Headers map, extracting relevant OTEL fields
    pub fn from_headers(endpoint_path: String, headers: &HashMap<String, String>) -> Self {
        Self {
            endpoint_path,
            traceparent: headers.get("traceparent").cloned(),
            tracestate: headers.get("tracestate").cloned(),
            x_request_id: headers
                .get("x-request-id")
                .cloned()
                .or_else(|| headers.get("x_request_id").cloned()),
        }
    }

    /// Encode header to bytes (JSON format with length prefix)
    ///
    /// Wire format:
    /// - header_len: u32 (big-endian)
    /// - header: JSON-encoded TcpRequestHeader
    pub fn encode(&self) -> Result<Bytes, std::io::Error> {
        let mut buf = BytesMut::new();
        self.encode_into(&mut buf)?;
        Ok(buf.freeze())
    }

    /// Encode header directly into a provided buffer (zero-copy)
    /// This is more efficient when composing multiple parts into a single buffer
    pub fn encode_into(&self, buf: &mut BytesMut) -> Result<(), std::io::Error> {
        // Serialize header to JSON
        let header_bytes = serde_json::to_vec(self).map_err(|e| {
            std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                format!("Failed to serialize header: {}", e),
            )
        })?;
        let header_len = header_bytes.len();

        if header_len > u32::MAX as usize {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                format!("Header too large: {} bytes", header_len),
            ));
        }

        // Reserve space for header length + header
        buf.reserve(4 + header_len);

        // Write header length (4 bytes)
        buf.put_u32(header_len as u32);

        // Write header (JSON)
        buf.put_slice(&header_bytes);

        Ok(())
    }

    /// Decode header from bytes (zero-copy when possible)
    /// Returns the decoded header and the number of bytes consumed
    pub fn decode(bytes: &Bytes) -> Result<(Self, usize), std::io::Error> {
        if bytes.len() < 4 {
            return Err(std::io::Error::new(
                std::io::ErrorKind::UnexpectedEof,
                "Not enough bytes for header length",
            ));
        }

        // Read header length (4 bytes)
        let header_len = u32::from_be_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]) as usize;
        let offset = 4;

        if bytes.len() < offset + header_len {
            return Err(std::io::Error::new(
                std::io::ErrorKind::UnexpectedEof,
                "Not enough bytes for header",
            ));
        }

        // Read and deserialize header
        let header_slice = &bytes[offset..offset + header_len];
        let header: TcpRequestHeader = serde_json::from_slice(header_slice).map_err(|e| {
            std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!("Failed to deserialize header: {}", e),
            )
        })?;

        Ok((header, offset + header_len))
    }
}

/// TCP request plane protocol message with headers and payload
///
/// Wire format:
/// - header_len: u32 (big-endian)
/// - header: JSON-encoded TcpRequestHeader (includes endpoint_path and OTEL headers)
/// - payload_len: u32 (big-endian)
/// - payload: bytes
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TcpRequestMessage {
    pub header: TcpRequestHeader,
    pub payload: Bytes,
}

impl TcpRequestMessage {
    pub fn new(header: TcpRequestHeader, payload: Bytes) -> Self {
        Self { header, payload }
    }

    /// Encode message to bytes
    pub fn encode(&self) -> Result<Bytes, std::io::Error> {
        if self.payload.len() > u32::MAX as usize {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                format!("Payload too large: {} bytes", self.payload.len()),
            ));
        }

        // Pre-allocate buffer for the entire message
        // Estimate: ~200 bytes for header JSON + 4 bytes header len + 4 bytes payload len + payload
        let total_size = 200 + 4 + self.payload.len();
        let mut buf = BytesMut::with_capacity(total_size);

        // Encode header directly into buffer (zero-copy)
        self.header.encode_into(&mut buf)?;

        // Write payload length (4 bytes)
        buf.put_u32(self.payload.len() as u32);

        // Write payload
        buf.put_slice(&self.payload);

        // Zero-copy conversion to Bytes
        Ok(buf.freeze())
    }

    /// Decode message from bytes (zero-copy when possible)
    pub fn decode(bytes: &Bytes) -> Result<Self, std::io::Error> {
        // Decode header
        let (header, mut offset) = TcpRequestHeader::decode(bytes)?;

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

        Ok(Self { header, payload })
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
        // Need at least 4 bytes for header_len
        if src.len() < 4 {
            return Ok(None);
        }

        // Peek at header length without consuming
        let header_len = u32::from_be_bytes([src[0], src[1], src[2], src[3]]) as usize;

        // Need at least: header_len(4) + header + payload_len(4)
        if src.len() < 4 + header_len + 4 {
            return Ok(None);
        }

        // Peek at payload length
        let payload_len_offset = 4 + header_len;
        let payload_len = u32::from_be_bytes([
            src[payload_len_offset],
            src[payload_len_offset + 1],
            src[payload_len_offset + 2],
            src[payload_len_offset + 3],
        ]) as usize;

        let total_len = 4 + header_len + 4 + payload_len;

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

        // We have a complete message, extract the bytes
        let message_bytes = src.split_to(total_len).freeze();

        // Decode header
        let (header, offset) = TcpRequestHeader::decode(&message_bytes)?;

        // Payload length already at correct offset (after header)
        // Skip the 4 bytes of payload length
        let payload_offset = offset + 4;

        // Read payload (zero-copy slice)
        let payload = message_bytes.slice(payload_offset..payload_offset + payload_len);

        Ok(Some(TcpRequestMessage { header, payload }))
    }
}

impl Encoder<TcpRequestMessage> for TcpRequestCodec {
    type Error = std::io::Error;

    fn encode(&mut self, item: TcpRequestMessage, dst: &mut BytesMut) -> Result<(), Self::Error> {
        if item.payload.len() > u32::MAX as usize {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                format!("Payload too large: {} bytes", item.payload.len()),
            ));
        }

        // Estimate header size for max message size check
        // Typical header is ~100-200 bytes depending on OTEL fields, assume 200
        let estimated_total = 200 + 4 + item.payload.len();

        // Check max message size (conservative estimate)
        if let Some(max_size) = self.max_message_size
            && estimated_total > max_size
        {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                format!(
                    "Request too large: ~{} bytes (max: {} bytes)",
                    estimated_total, max_size
                ),
            ));
        }

        // Reserve estimated space
        dst.reserve(estimated_total);

        // Encode header directly into buffer (zero-copy)
        item.header.encode_into(dst)?;

        // Write payload length
        dst.put_u32(item.payload.len() as u32);

        // Write payload
        dst.put_slice(&item.payload);

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
        let header = TcpRequestHeader::new("test.endpoint".to_string());
        let msg = TcpRequestMessage::new(header, Bytes::from(vec![1, 2, 3, 4, 5]));

        let encoded = msg.encode().unwrap();
        let decoded = TcpRequestMessage::decode(&encoded).unwrap();

        assert_eq!(decoded, msg);
    }

    #[test]
    fn test_tcp_request_empty_payload() {
        let header = TcpRequestHeader::new("test".to_string());
        let msg = TcpRequestMessage::new(header, Bytes::new());

        let encoded = msg.encode().unwrap();
        let decoded = TcpRequestMessage::decode(&encoded).unwrap();

        assert_eq!(decoded, msg);
    }

    #[test]
    fn test_tcp_request_large_payload() {
        let payload = Bytes::from(vec![42u8; 1024 * 1024]); // 1MB
        let header = TcpRequestHeader::new("large".to_string());
        let msg = TcpRequestMessage::new(header, payload);

        let encoded = msg.encode().unwrap();
        let decoded = TcpRequestMessage::decode(&encoded).unwrap();

        assert_eq!(decoded, msg);
    }

    #[test]
    fn test_tcp_request_decode_truncated() {
        let header = TcpRequestHeader::new("test".to_string());
        let msg = TcpRequestMessage::new(header, Bytes::from(vec![1, 2, 3, 4, 5]));
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
        let header = TcpRequestHeader::new("тест.端点".to_string());
        let msg = TcpRequestMessage::new(header, Bytes::from(vec![1, 2, 3]));

        let encoded = msg.encode().unwrap();
        let decoded = TcpRequestMessage::decode(&encoded).unwrap();

        assert_eq!(decoded, msg);
    }

    #[test]
    fn test_tcp_request_with_otel_headers() {
        let header = TcpRequestHeader {
            endpoint_path: "test.endpoint".to_string(),
            traceparent: Some(
                "00-4bf92f3577b34da6a3ce929d0e0e4736-00f067aa0ba902b7-01".to_string(),
            ),
            tracestate: Some("vendor=value".to_string()),
            x_request_id: Some("req-abc123".to_string()),
        };

        let msg = TcpRequestMessage::new(header.clone(), Bytes::from(vec![1, 2, 3, 4, 5]));

        let encoded = msg.encode().unwrap();
        let decoded = TcpRequestMessage::decode(&encoded).unwrap();

        assert_eq!(decoded.header.endpoint_path, msg.header.endpoint_path);
        assert_eq!(decoded.payload, msg.payload);
        assert_eq!(decoded.header.traceparent, header.traceparent);
        assert_eq!(decoded.header.tracestate, header.tracestate);
        assert_eq!(decoded.header.x_request_id, header.x_request_id);
    }

    #[test]
    fn test_tcp_request_with_empty_otel() {
        let header = TcpRequestHeader::new("test".to_string());
        let msg = TcpRequestMessage::new(header, Bytes::from(vec![1, 2, 3]));

        let encoded = msg.encode().unwrap();
        let decoded = TcpRequestMessage::decode(&encoded).unwrap();

        assert_eq!(decoded, msg);
        assert!(decoded.header.traceparent.is_none());
        assert!(decoded.header.tracestate.is_none());
        assert!(decoded.header.x_request_id.is_none());
    }

    #[test]
    fn test_tcp_request_header_encode_decode() {
        let header = TcpRequestHeader {
            endpoint_path: "test.endpoint".to_string(),
            traceparent: Some("00-trace-span-01".to_string()),
            tracestate: Some("vendor=value".to_string()),
            x_request_id: Some("req-123".to_string()),
        };

        let encoded = header.encode().unwrap();
        let (decoded, bytes_consumed) = TcpRequestHeader::decode(&encoded).unwrap();

        assert_eq!(decoded, header);
        assert_eq!(bytes_consumed, encoded.len());
    }

    #[test]
    fn test_otel_headers_from_headers() {
        use std::collections::HashMap;

        let mut headers = HashMap::new();
        headers.insert("traceparent".to_string(), "00-trace-span-01".to_string());
        headers.insert("tracestate".to_string(), "vendor=value".to_string());
        headers.insert("x-request-id".to_string(), "req-123".to_string());
        headers.insert("other-header".to_string(), "ignored".to_string());

        let header = TcpRequestHeader::from_headers("test.endpoint".to_string(), &headers);

        assert_eq!(header.endpoint_path, "test.endpoint");
        assert_eq!(header.traceparent, Some("00-trace-span-01".to_string()));
        assert_eq!(header.tracestate, Some("vendor=value".to_string()));
        assert_eq!(header.x_request_id, Some("req-123".to_string()));
    }

    #[test]
    fn test_tcp_request_header_encode_into() {
        let header = TcpRequestHeader {
            endpoint_path: "test.endpoint".to_string(),
            traceparent: Some("00-trace-01".to_string()),
            tracestate: None,
            x_request_id: Some("req-456".to_string()),
        };

        // Test encode_into
        let mut buf = BytesMut::new();
        header.encode_into(&mut buf).unwrap();
        let encoded_via_into = buf.freeze();

        // Test standalone encode
        let encoded_direct = header.encode().unwrap();

        // Both should produce identical results
        assert_eq!(encoded_via_into, encoded_direct);

        // Verify it can be decoded
        let (decoded, _) = TcpRequestHeader::decode(&encoded_via_into).unwrap();
        assert_eq!(decoded, header);
    }

    #[test]
    fn test_tcp_request_header_empty_otel() {
        let header = TcpRequestHeader::new("simple.endpoint".to_string());

        let encoded = header.encode().unwrap();
        let (decoded, bytes_consumed) = TcpRequestHeader::decode(&encoded).unwrap();

        assert_eq!(decoded.endpoint_path, "simple.endpoint");
        assert!(decoded.traceparent.is_none());
        assert!(decoded.tracestate.is_none());
        assert!(decoded.x_request_id.is_none());
        assert_eq!(bytes_consumed, encoded.len());
    }

    #[test]
    fn test_tcp_request_header_partial_otel() {
        // Test with only some OTEL fields set
        let header = TcpRequestHeader {
            endpoint_path: "api/v1/endpoint".to_string(),
            traceparent: Some("00-abc123-def456-01".to_string()),
            tracestate: None,
            x_request_id: Some("request-789".to_string()),
        };

        let encoded = header.encode().unwrap();
        let (decoded, _) = TcpRequestHeader::decode(&encoded).unwrap();

        assert_eq!(decoded, header);
        assert_eq!(decoded.traceparent, Some("00-abc123-def456-01".to_string()));
        assert_eq!(decoded.tracestate, None);
        assert_eq!(decoded.x_request_id, Some("request-789".to_string()));
    }

    #[test]
    fn test_tcp_request_header_decode_truncated() {
        let header = TcpRequestHeader::new("test".to_string());
        let encoded = header.encode().unwrap();

        // Truncate at header length field
        let truncated = encoded.slice(..2);
        let result = TcpRequestHeader::decode(&truncated);
        assert!(result.is_err());

        // Truncate in the middle of header data
        if encoded.len() > 6 {
            let truncated = encoded.slice(..6);
            let result = TcpRequestHeader::decode(&truncated);
            assert!(result.is_err());
        }
    }

    #[test]
    fn test_tcp_request_header_decode_invalid_json() {
        // Manually construct invalid JSON message
        let mut buf = BytesMut::new();
        let invalid_json = b"{invalid json}";
        buf.put_u32(invalid_json.len() as u32);
        buf.put_slice(invalid_json);

        let bytes = buf.freeze();
        let result = TcpRequestHeader::decode(&bytes);
        assert!(result.is_err());
    }

    #[test]
    fn test_tcp_request_header_unicode_endpoint() {
        let header = TcpRequestHeader {
            endpoint_path: "测试/エンドポイント/конечная точка".to_string(),
            traceparent: None,
            tracestate: None,
            x_request_id: None,
        };

        let encoded = header.encode().unwrap();
        let (decoded, _) = TcpRequestHeader::decode(&encoded).unwrap();

        assert_eq!(decoded, header);
        assert_eq!(decoded.endpoint_path, "测试/エンドポイント/конечная точка");
    }

    #[test]
    fn test_tcp_request_header_long_endpoint() {
        // Test with a very long endpoint path
        let long_endpoint = "a".repeat(1000);
        let header = TcpRequestHeader::new(long_endpoint.clone());

        let encoded = header.encode().unwrap();
        let (decoded, _) = TcpRequestHeader::decode(&encoded).unwrap();

        assert_eq!(decoded.endpoint_path, long_endpoint);
    }

    #[test]
    fn test_tcp_request_header_decode_offset() {
        let header = TcpRequestHeader {
            endpoint_path: "test".to_string(),
            traceparent: Some("00-trace-01".to_string()),
            tracestate: None,
            x_request_id: None,
        };

        let encoded = header.encode().unwrap();
        let total_len = encoded.len();

        // Decode and verify offset
        let (decoded, bytes_consumed) = TcpRequestHeader::decode(&encoded).unwrap();

        assert_eq!(decoded, header);
        assert_eq!(bytes_consumed, total_len);

        // Now test with extra data after the header
        let mut buf = BytesMut::from(&encoded[..]);
        buf.put_slice(b"extra data after header");
        let bytes_with_extra = buf.freeze();

        let (decoded2, bytes_consumed2) = TcpRequestHeader::decode(&bytes_with_extra).unwrap();
        assert_eq!(decoded2, header);
        assert_eq!(bytes_consumed2, total_len); // Should only consume header bytes
        assert!(bytes_with_extra.len() > bytes_consumed2); // Extra data still there
    }

    #[test]
    fn test_tcp_request_codec() {
        use tokio_util::codec::{Decoder, Encoder};

        let header = TcpRequestHeader::new("test.endpoint".to_string());
        let msg = TcpRequestMessage::new(header, Bytes::from(vec![1, 2, 3, 4, 5]));

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

        let header = TcpRequestHeader::new("test.endpoint".to_string());
        let msg = TcpRequestMessage::new(header, Bytes::from(vec![1, 2, 3, 4, 5]));

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

        let header = TcpRequestHeader::new("test".to_string());
        let msg = TcpRequestMessage::new(header, Bytes::from(vec![1, 2, 3, 4, 5]));

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

            let header1 = TcpRequestHeader::new("endpoint1".to_string());
            let msg1 = TcpRequestMessage::new(header1, Bytes::from("data1"));
            let header2 = TcpRequestHeader::new("endpoint2".to_string());
            let msg2 = TcpRequestMessage::new(header2, Bytes::from("data2"));

            writer.send(msg1).await.unwrap();
            writer.send(msg2).await.unwrap();
        }

        // Reader side: decode requests
        {
            let cursor = Cursor::new(&buffer[..]);
            let mut reader = FramedRead::new(cursor, TcpRequestCodec::new(None));

            let decoded1 = reader.next().await.unwrap().unwrap();
            assert_eq!(decoded1.header.endpoint_path, "endpoint1");
            assert_eq!(decoded1.payload, Bytes::from("data1"));

            let decoded2 = reader.next().await.unwrap().unwrap();
            assert_eq!(decoded2.header.endpoint_path, "endpoint2");
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
        let header = TcpRequestHeader::new("test".to_string());
        let msg = TcpRequestMessage::new(header, Bytes::from("hello"));
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
            assert_eq!(decoded.header.endpoint_path, "test");
            assert_eq!(decoded.payload, Bytes::from("hello"));
        }
    }
}
