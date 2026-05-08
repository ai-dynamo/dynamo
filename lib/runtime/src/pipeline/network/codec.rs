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
pub mod zero_copy_decoder;

pub use two_part::{TwoPartCodec, TwoPartMessage, TwoPartMessageType};
pub use zero_copy_decoder::{TcpRequestMessageZeroCopy, ZeroCopyTcpDecoder};

const TCP_REQUEST_ENDPOINT_LEN_WIDTH: usize = 2;
const TCP_REQUEST_HEADERS_LEN_WIDTH: usize = 2;
const TCP_REQUEST_PAYLOAD_LEN_WIDTH: usize = 4;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct TcpRequestWireHeader {
    endpoint_len: usize,
    headers_len: usize,
    payload_len: usize,
    header_size: usize,
    total_len: usize,
}

impl TcpRequestWireHeader {
    fn endpoint_start(&self) -> usize {
        TCP_REQUEST_ENDPOINT_LEN_WIDTH
    }

    fn endpoint_end(&self) -> usize {
        self.endpoint_start() + self.endpoint_len
    }

    fn headers_start(&self) -> usize {
        self.endpoint_end() + TCP_REQUEST_HEADERS_LEN_WIDTH
    }

    fn headers_end(&self) -> usize {
        self.headers_start() + self.headers_len
    }

    fn payload_start(&self) -> usize {
        self.header_size
    }
}

fn tcp_request_header_size(endpoint_len: usize, headers_len: usize) -> usize {
    TCP_REQUEST_ENDPOINT_LEN_WIDTH
        + endpoint_len
        + TCP_REQUEST_HEADERS_LEN_WIDTH
        + headers_len
        + TCP_REQUEST_PAYLOAD_LEN_WIDTH
}

fn tcp_request_total_len(
    endpoint_len: usize,
    headers_len: usize,
    payload_len: usize,
) -> Result<TcpRequestWireHeader, std::io::Error> {
    let header_size = tcp_request_header_size(endpoint_len, headers_len);
    let total_len = header_size.checked_add(payload_len).ok_or_else(|| {
        std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            "TCP request message length overflow",
        )
    })?;

    Ok(TcpRequestWireHeader {
        endpoint_len,
        headers_len,
        payload_len,
        header_size,
        total_len,
    })
}

fn validate_tcp_request_encode_lengths(
    endpoint_len: usize,
    headers_len: usize,
    payload_len: usize,
) -> Result<TcpRequestWireHeader, std::io::Error> {
    if endpoint_len > u16::MAX as usize {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidInput,
            format!("Endpoint path too long: {} bytes", endpoint_len),
        ));
    }

    if headers_len > u16::MAX as usize {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidInput,
            format!("Headers too large: {} bytes", headers_len),
        ));
    }

    if payload_len > u32::MAX as usize {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidInput,
            format!("Payload too large: {} bytes", payload_len),
        ));
    }

    tcp_request_total_len(endpoint_len, headers_len, payload_len)
}

fn tcp_request_endpoint_len(bytes: &[u8]) -> Result<usize, std::io::Error> {
    if bytes.len() < TCP_REQUEST_ENDPOINT_LEN_WIDTH {
        return Err(std::io::Error::new(
            std::io::ErrorKind::UnexpectedEof,
            "Not enough bytes for endpoint path length",
        ));
    }

    Ok(u16::from_be_bytes([bytes[0], bytes[1]]) as usize)
}

fn tcp_request_headers_len(bytes: &[u8], endpoint_len: usize) -> Result<usize, std::io::Error> {
    let endpoint_end = TCP_REQUEST_ENDPOINT_LEN_WIDTH + endpoint_len;
    if bytes.len() < endpoint_end {
        return Err(std::io::Error::new(
            std::io::ErrorKind::UnexpectedEof,
            "Not enough bytes for endpoint path",
        ));
    }

    if bytes.len() < endpoint_end + TCP_REQUEST_HEADERS_LEN_WIDTH {
        return Err(std::io::Error::new(
            std::io::ErrorKind::UnexpectedEof,
            "Not enough bytes for headers length",
        ));
    }

    Ok(u16::from_be_bytes([bytes[endpoint_end], bytes[endpoint_end + 1]]) as usize)
}

fn parse_tcp_request_frame_header(bytes: &[u8]) -> Result<TcpRequestWireHeader, std::io::Error> {
    let endpoint_len = tcp_request_endpoint_len(bytes)?;
    let headers_len = tcp_request_headers_len(bytes, endpoint_len)?;

    let headers_end =
        TCP_REQUEST_ENDPOINT_LEN_WIDTH + endpoint_len + TCP_REQUEST_HEADERS_LEN_WIDTH + headers_len;
    if bytes.len() < headers_end {
        return Err(std::io::Error::new(
            std::io::ErrorKind::UnexpectedEof,
            "Not enough bytes for headers",
        ));
    }

    if bytes.len() < headers_end + TCP_REQUEST_PAYLOAD_LEN_WIDTH {
        return Err(std::io::Error::new(
            std::io::ErrorKind::UnexpectedEof,
            "Not enough bytes for payload length",
        ));
    }

    let payload_len = u32::from_be_bytes([
        bytes[headers_end],
        bytes[headers_end + 1],
        bytes[headers_end + 2],
        bytes[headers_end + 3],
    ]) as usize;

    tcp_request_total_len(endpoint_len, headers_len, payload_len)
}

fn parse_tcp_request_frame(bytes: &[u8]) -> Result<TcpRequestWireHeader, std::io::Error> {
    let parsed = parse_tcp_request_frame_header(bytes)?;
    if bytes.len() < parsed.total_len {
        return Err(std::io::Error::new(
            std::io::ErrorKind::UnexpectedEof,
            format!(
                "Not enough bytes for payload: expected {}, got {}",
                parsed.payload_len,
                bytes.len().saturating_sub(parsed.payload_start())
            ),
        ));
    }

    Ok(parsed)
}

fn check_tcp_request_max_message_size(
    total_len: usize,
    max_message_size: usize,
) -> Result<(), std::io::Error> {
    if total_len > max_message_size {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            format!(
                "message too large: {} bytes (max: {} bytes)",
                total_len, max_message_size
            ),
        ));
    }

    Ok(())
}

/// TCP request plane protocol message with endpoint routing and trace headers
///
/// Wire format:
/// - endpoint_path_len: u16 (big-endian)
/// - endpoint_path: UTF-8 string
/// - headers_len: u16 (big-endian)
/// - headers: JSON-encoded HashMap<String, String>
/// - payload_len: u32 (big-endian)
/// - payload: bytes
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TcpRequestMessage {
    pub endpoint_path: String,
    pub headers: std::collections::HashMap<String, String>,
    pub payload: Bytes,
}

impl TcpRequestMessage {
    pub fn new(endpoint_path: String, payload: Bytes) -> Self {
        Self {
            endpoint_path,
            headers: std::collections::HashMap::new(),
            payload,
        }
    }

    pub fn with_headers(
        endpoint_path: String,
        headers: std::collections::HashMap<String, String>,
        payload: Bytes,
    ) -> Self {
        Self {
            endpoint_path,
            headers,
            payload,
        }
    }

    /// Encode message to bytes
    pub fn encode(&self) -> Result<Bytes, std::io::Error> {
        let endpoint_bytes = self.endpoint_path.as_bytes();
        let endpoint_len = endpoint_bytes.len();

        // Encode headers as JSON
        let headers_json = serde_json::to_vec(&self.headers).map_err(|e| {
            std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                format!("Failed to encode headers: {}", e),
            )
        })?;
        let headers_len = headers_json.len();

        let parsed =
            validate_tcp_request_encode_lengths(endpoint_len, headers_len, self.payload.len())?;

        // Use BytesMut for efficient buffer building
        let mut buf = BytesMut::with_capacity(parsed.total_len);

        // Write endpoint path length (2 bytes)
        buf.put_u16(endpoint_len as u16);

        // Write endpoint path
        buf.put_slice(endpoint_bytes);

        // Write headers length (2 bytes)
        buf.put_u16(headers_len as u16);

        // Write headers
        buf.put_slice(&headers_json);

        // Write payload length (4 bytes)
        buf.put_u32(self.payload.len() as u32);

        // Write payload
        buf.put_slice(&self.payload);

        // Zero-copy conversion to Bytes
        Ok(buf.freeze())
    }

    /// Decode message from bytes (for backward compatibility, zero-copy when possible)
    pub fn decode(bytes: &Bytes) -> Result<Self, std::io::Error> {
        let parsed = parse_tcp_request_frame(bytes)?;

        // Read endpoint path (requires copy for UTF-8 validation)
        let endpoint_path =
            String::from_utf8(bytes[parsed.endpoint_start()..parsed.endpoint_end()].to_vec())
                .map_err(|e| {
                    std::io::Error::new(
                        std::io::ErrorKind::InvalidData,
                        format!("Invalid UTF-8 in endpoint path: {}", e),
                    )
                })?;

        // Read and parse headers
        let headers: std::collections::HashMap<String, String> = serde_json::from_slice(
            &bytes[parsed.headers_start()..parsed.headers_end()],
        )
        .map_err(|e| {
            std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!("Invalid JSON in headers: {}", e),
            )
        })?;

        // Read payload (zero-copy slice)
        let payload = bytes.slice(parsed.payload_start()..parsed.total_len);

        Ok(Self {
            endpoint_path,
            headers,
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
    fn test_tcp_request_decode_invalid_endpoint_utf8() {
        let mut encoded = BytesMut::new();
        encoded.put_u16(2);
        encoded.put_slice(&[0xff, 0xff]);
        encoded.put_u16(2);
        encoded.put_slice(b"{}");
        encoded.put_u32(0);

        let result = TcpRequestMessage::decode(&encoded.freeze());

        assert!(result.is_err());
        let err = result.unwrap_err();
        assert_eq!(err.kind(), std::io::ErrorKind::InvalidData);
        assert!(err.to_string().contains("Invalid UTF-8"));
    }

    #[test]
    fn test_tcp_request_decode_invalid_headers_json() {
        let mut encoded = BytesMut::new();
        encoded.put_u16(4);
        encoded.put_slice(b"test");
        encoded.put_u16(1);
        encoded.put_slice(b"{");
        encoded.put_u32(0);

        let result = TcpRequestMessage::decode(&encoded.freeze());

        assert!(result.is_err());
        let err = result.unwrap_err();
        assert_eq!(err.kind(), std::io::ErrorKind::InvalidData);
        assert!(err.to_string().contains("Invalid JSON"));
    }

    #[test]
    fn test_tcp_request_empty_endpoint_path() {
        let msg = TcpRequestMessage::new(String::new(), Bytes::from_static(b"payload"));

        let encoded = msg.encode().unwrap();
        let decoded = TcpRequestMessage::decode(&encoded).unwrap();

        assert_eq!(decoded, msg);
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
}
