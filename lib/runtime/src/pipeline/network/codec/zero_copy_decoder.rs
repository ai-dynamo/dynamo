// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Zero-copy TCP message decoder for high-concurrency scenarios
//!
//! This decoder eliminates message reconstruction copies by:
//! 1. Reading into a reusable buffer
//! 2. Parsing headers in-place
//! 3. Splitting off exact message sizes (zero-copy via Bytes::split_to)
//! 4. Returning Arc-counted Bytes that can be cloned cheaply

use bytes::{Buf, Bytes, BytesMut};
use std::io;
use tokio::io::{AsyncRead, AsyncReadExt};

/// Maximum message size (32MB default, configurable via env)
const MAX_MESSAGE_SIZE: usize = 32 * 1024 * 1024;

fn get_max_message_size() -> usize {
    std::env::var("DYN_TCP_MAX_MESSAGE_SIZE")
        .ok()
        .and_then(|s| s.parse::<usize>().ok())
        .unwrap_or(MAX_MESSAGE_SIZE)
}

/// Zero-copy streaming decoder that reuses buffers
///
/// This decoder maintains an internal buffer and only allocates when necessary.
/// Messages are returned as Arc-counted Bytes slices, making cloning extremely cheap.
pub struct ZeroCopyTcpDecoder {
    /// Reusable read buffer - grows as needed but never shrinks
    read_buffer: BytesMut,
    /// Maximum allowed message size
    max_message_size: usize,
}

impl ZeroCopyTcpDecoder {
    /// Create a new decoder with default buffer size
    ///
    /// Initial buffer is 256KB, suitable for typical payloads
    pub fn new() -> Self {
        Self::with_capacity(262144)
    }

    /// Create a new decoder with specific initial capacity
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            read_buffer: BytesMut::with_capacity(capacity),
            max_message_size: get_max_message_size(),
        }
    }

    /// Read one complete message with ZERO copies
    ///
    /// This method:
    /// 1. Ensures headers are buffered
    /// 2. Parses headers in-place (no allocation)
    /// 3. Ensures entire message is buffered
    /// 4. Splits off exact message size (zero-copy pointer arithmetic)
    /// 5. Returns Arc-counted Bytes (cheap to clone)
    pub async fn read_message<R: AsyncRead + Unpin>(
        &mut self,
        reader: &mut R,
    ) -> io::Result<TcpRequestMessageZeroCopy> {
        // Ensure we have at least the minimum header size (2 bytes path len + some data)
        const MIN_HEADER_SIZE: usize = 6; // path_len(2) + min_path(1) + payload_len(4)

        // Fill buffer if needed
        while self.read_buffer.len() < MIN_HEADER_SIZE {
            let n = reader.read_buf(&mut self.read_buffer).await?;
            if n == 0 {
                if self.read_buffer.is_empty() {
                    return Err(io::Error::new(
                        io::ErrorKind::UnexpectedEof,
                        "connection closed",
                    ));
                } else {
                    return Err(io::Error::new(
                        io::ErrorKind::UnexpectedEof,
                        "incomplete message header",
                    ));
                }
            }
        }

        // Parse endpoint path length (first 2 bytes) - NO COPY
        let path_len = u16::from_be_bytes([self.read_buffer[0], self.read_buffer[1]]) as usize;

        // Sanity check path length
        if path_len == 0 || path_len > 1024 {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!("invalid endpoint path length: {}", path_len),
            ));
        }

        // Ensure we have path + headers_len
        let initial_header_size = 2 + path_len + 2; // path_len(2) + path + headers_len(2)
        while self.read_buffer.len() < initial_header_size {
            let n = reader.read_buf(&mut self.read_buffer).await?;
            if n == 0 {
                return Err(io::Error::new(
                    io::ErrorKind::UnexpectedEof,
                    "incomplete message header",
                ));
            }
        }

        // Parse headers length (2 bytes after path) - NO COPY
        let headers_len_offset = 2 + path_len;
        let headers_len = u16::from_be_bytes([
            self.read_buffer[headers_len_offset],
            self.read_buffer[headers_len_offset + 1],
        ]) as usize;

        // Ensure we have headers + payload length
        let full_header_size = 2 + path_len + 2 + headers_len + 4; // path_len(2) + path + headers_len(2) + headers + payload_len(4)
        while self.read_buffer.len() < full_header_size {
            let n = reader.read_buf(&mut self.read_buffer).await?;
            if n == 0 {
                return Err(io::Error::new(
                    io::ErrorKind::UnexpectedEof,
                    "incomplete message header",
                ));
            }
        }

        // Parse payload length (4 bytes after headers) - NO COPY
        let payload_len_offset = 2 + path_len + 2 + headers_len;
        let payload_len = u32::from_be_bytes([
            self.read_buffer[payload_len_offset],
            self.read_buffer[payload_len_offset + 1],
            self.read_buffer[payload_len_offset + 2],
            self.read_buffer[payload_len_offset + 3],
        ]) as usize;

        // Sanity check payload length
        if payload_len > self.max_message_size {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!(
                    "message too large: {} bytes (max: {} bytes)",
                    payload_len, self.max_message_size
                ),
            ));
        }

        // Calculate total message size
        let total_len = 2 + path_len + 2 + headers_len + 4 + payload_len;

        // Ensure entire message is buffered
        while self.read_buffer.len() < total_len {
            let n = reader.read_buf(&mut self.read_buffer).await?;
            if n == 0 {
                return Err(io::Error::new(
                    io::ErrorKind::UnexpectedEof,
                    format!(
                        "incomplete message: expected {} bytes, got {}",
                        total_len,
                        self.read_buffer.len()
                    ),
                ));
            }
        }

        // Split off exactly what we need - ZERO COPY!
        // split_to() just advances the internal pointer, doesn't allocate or copy
        let message_bytes = self.read_buffer.split_to(total_len).freeze();

        // Return zero-copy message wrapper
        Ok(TcpRequestMessageZeroCopy::new(message_bytes))
    }

    /// Get the current buffer capacity
    pub fn buffer_capacity(&self) -> usize {
        self.read_buffer.capacity()
    }

    /// Get the current buffered data size
    pub fn buffered_len(&self) -> usize {
        self.read_buffer.len()
    }
}

impl Default for ZeroCopyTcpDecoder {
    fn default() -> Self {
        Self::new()
    }
}

/// Zero-copy message representation
///
/// This struct holds an Arc-counted Bytes buffer containing the entire message.
/// All accessors return zero-copy slices or references into this buffer.
#[derive(Clone)]
pub struct TcpRequestMessageZeroCopy {
    /// Entire message as Arc-counted buffer
    /// Format: [path_len(2)][path(var)][headers_len(2)][headers(var)][payload_len(4)][payload(var)]
    raw: Bytes,
}

impl TcpRequestMessageZeroCopy {
    /// Create a new zero-copy message from raw bytes
    fn new(raw: Bytes) -> Self {
        Self { raw }
    }

    /// Get the endpoint path length
    #[inline]
    fn path_len(&self) -> usize {
        u16::from_be_bytes([self.raw[0], self.raw[1]]) as usize
    }

    /// Get endpoint path as a string slice (zero-copy)
    ///
    /// This returns a reference into the message buffer, no allocation.
    pub fn endpoint_path(&self) -> Result<&str, std::str::Utf8Error> {
        let path_len = self.path_len();
        std::str::from_utf8(&self.raw[2..2 + path_len])
    }

    /// Get endpoint path as bytes (zero-copy)
    pub fn endpoint_path_bytes(&self) -> &[u8] {
        let path_len = self.path_len();
        &self.raw[2..2 + path_len]
    }

    /// Get the headers length
    #[inline]
    fn headers_len(&self) -> usize {
        let path_len = self.path_len();
        let offset = 2 + path_len;
        u16::from_be_bytes([self.raw[offset], self.raw[offset + 1]]) as usize
    }

    /// Get headers as bytes (zero-copy)
    pub fn headers_bytes(&self) -> &[u8] {
        let path_len = self.path_len();
        let headers_len = self.headers_len();
        let headers_start = 2 + path_len + 2;
        &self.raw[headers_start..headers_start + headers_len]
    }

    /// Get headers as a HashMap (requires parsing)
    pub fn headers(&self) -> std::collections::HashMap<String, String> {
        let headers_bytes = self.headers_bytes();
        if headers_bytes.is_empty() {
            return std::collections::HashMap::new();
        }

        // Parse headers from JSON format
        serde_json::from_slice(headers_bytes).unwrap_or_default()
    }

    /// Get the payload length
    #[inline]
    fn payload_len(&self) -> usize {
        let path_len = self.path_len();
        let headers_len = self.headers_len();
        let offset = 2 + path_len + 2 + headers_len;
        u32::from_be_bytes([
            self.raw[offset],
            self.raw[offset + 1],
            self.raw[offset + 2],
            self.raw[offset + 3],
        ]) as usize
    }

    /// Get payload as zero-copy Bytes
    ///
    /// This returns an Arc-counted slice of the message buffer.
    /// Cloning the returned Bytes is extremely cheap (just Arc clone).
    pub fn payload(&self) -> Bytes {
        let path_len = self.path_len();
        let headers_len = self.headers_len();
        let payload_start = 2 + path_len + 2 + headers_len + 4;
        self.raw.slice(payload_start..) // ZERO COPY! Just Arc clone + offset
    }

    /// Get total message size in bytes
    pub fn total_size(&self) -> usize {
        self.raw.len()
    }

    /// Get the raw message bytes (for debugging)
    pub fn raw_bytes(&self) -> &Bytes {
        &self.raw
    }
}

impl std::fmt::Debug for TcpRequestMessageZeroCopy {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("TcpRequestMessageZeroCopy")
            .field("total_size", &self.total_size())
            .field("endpoint_path", &self.endpoint_path().ok())
            .field("payload_len", &self.payload_len())
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tokio::io::AsyncWriteExt;

    #[tokio::test]
    async fn test_zero_copy_decoder_basic() {
        // Create a test message
        let endpoint = "test/endpoint";
        let payload = b"Hello, World!";

        let mut message = Vec::new();
        message.extend_from_slice(&(endpoint.len() as u16).to_be_bytes());
        message.extend_from_slice(endpoint.as_bytes());
        message.extend_from_slice(&(payload.len() as u32).to_be_bytes());
        message.extend_from_slice(payload);

        // Create a mock reader
        let mut reader = &message[..];

        // Decode
        let mut decoder = ZeroCopyTcpDecoder::new();
        let msg = decoder.read_message(&mut reader).await.unwrap();

        // Verify
        assert_eq!(msg.endpoint_path().unwrap(), endpoint);
        assert_eq!(msg.payload().as_ref(), payload);
        assert_eq!(msg.total_size(), message.len());
    }

    #[tokio::test]
    async fn test_zero_copy_decoder_large_payload() {
        // Create a large payload (200KB)
        let endpoint = "large/endpoint";
        let payload = vec![0x42u8; 200 * 1024];

        let mut message = Vec::new();
        message.extend_from_slice(&(endpoint.len() as u16).to_be_bytes());
        message.extend_from_slice(endpoint.as_bytes());
        message.extend_from_slice(&(payload.len() as u32).to_be_bytes());
        message.extend_from_slice(&payload);

        let mut reader = &message[..];
        let mut decoder = ZeroCopyTcpDecoder::new();
        let msg = decoder.read_message(&mut reader).await.unwrap();

        assert_eq!(msg.endpoint_path().unwrap(), endpoint);
        assert_eq!(msg.payload().len(), payload.len());
    }
}
