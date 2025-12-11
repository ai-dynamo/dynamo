// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Zero-copy TCP framing codec for ActiveMessage transport
//!
//! Wire format (7-15 bytes overhead):
//! ```text
//! [u16 BE: schema_version][u8: frame_type][u32 BE: header_len][u32 BE: payload_len][header bytes][payload bytes]
//! ```
//!
//! The codec uses `BytesMut` for receiving and `Bytes` for output, enabling
//! zero-copy buffer slicing where header and payload share the underlying buffer.

use bytes::{Buf, Bytes, BytesMut};
use std::io::{self, IoSlice};
use tokio::io::{AsyncWrite, AsyncWriteExt};
use tokio_util::codec::Decoder;

use crate::MessageType;

/// Current schema version
const SCHEMA_VERSION_V1: u16 = 1;

/// Maximum frame size (16 MB)
const MAX_FRAME_SIZE: u32 = 16 * 1024 * 1024;

/// Minimum frame header size (version + type + 2 lengths)
const MIN_HEADER_SIZE: usize = 2 + 1 + 4 + 4; // 11 bytes

/// Zero-copy frame decoder for TCP transport
///
/// This decoder maintains state across multiple calls to support partial
/// frame reception. It decodes frames into (MessageType, header: Bytes, payload: Bytes)
/// where header and payload are zero-copy slices of the receive buffer.
#[derive(Debug, Clone)]
pub struct TcpFrameCodec {
    state: DecodeState,
}

#[derive(Debug, Clone, Copy)]
enum DecodeState {
    /// Waiting for frame header (version + type + lengths)
    AwaitingHeader,
    /// Waiting for frame data (header + payload), with known lengths
    AwaitingData {
        frame_type: MessageType,
        header_len: u32,
        payload_len: u32,
    },
}

impl TcpFrameCodec {
    /// Create a new frame codec
    pub fn new() -> Self {
        Self {
            state: DecodeState::AwaitingHeader,
        }
    }

    /// Build the frame preamble (metadata header)
    ///
    /// Returns a fixed-size preamble containing version, message type, and lengths.
    #[inline]
    pub fn build_preamble(
        msg_type: MessageType,
        header_len: u32,
        payload_len: u32,
    ) -> io::Result<[u8; MIN_HEADER_SIZE]> {
        // Validate lengths before building preamble
        Self::validate_lengths(header_len, payload_len)?;

        let mut preamble = [0u8; MIN_HEADER_SIZE];

        // Layout:
        // [0..2) = version
        // [2]    = msg_type
        // [3..7) = header_len
        // [7..11)= payload_len  (total 11 bytes)
        preamble[0..2].copy_from_slice(&SCHEMA_VERSION_V1.to_be_bytes());
        preamble[2] = msg_type.as_u8();
        preamble[3..7].copy_from_slice(&header_len.to_be_bytes());
        preamble[7..11].copy_from_slice(&payload_len.to_be_bytes());

        Ok(preamble)
    }

    /// Parse message type from a preamble
    ///
    /// Validates the schema version and extracts the message type from the preamble.
    #[inline]
    pub fn parse_message_type_from_preamble(preamble: &[u8]) -> io::Result<MessageType> {
        if preamble.len() < MIN_HEADER_SIZE {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "Preamble too short",
            ));
        }

        // Validate schema version
        let schema_version = u16::from_be_bytes([preamble[0], preamble[1]]);
        if schema_version != SCHEMA_VERSION_V1 {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!(
                    "Unsupported schema version: {} (expected {})",
                    schema_version, SCHEMA_VERSION_V1
                ),
            ));
        }

        // Extract and validate message type
        MessageType::from_u8(preamble[2]).ok_or_else(|| {
            io::Error::new(
                io::ErrorKind::InvalidData,
                format!("Invalid message type: {}", preamble[2]),
            )
        })
    }

    /// Encode and write a frame asynchronously
    #[inline]
    pub async fn encode_frame<W: AsyncWrite + Unpin>(
        writer: &mut W,
        msg_type: MessageType,
        header: &[u8],
        payload: &[u8],
    ) -> tokio::io::Result<()> {
        let preamble = Self::build_preamble(msg_type, header.len() as u32, payload.len() as u32)?;

        let bufs = [
            IoSlice::new(&preamble),
            IoSlice::new(header),
            IoSlice::new(payload),
        ];

        let total_len = preamble.len() + header.len() + payload.len();
        let written = writer.write_vectored(&bufs).await?;
        if written != total_len {
            return Err(io::Error::new(
                io::ErrorKind::WriteZero,
                format!("incomplete write: wrote {} of {} bytes", written, total_len),
            ));
        }
        Ok(())
    }

    /// Encode and write a frame synchronously
    #[inline]
    pub fn encode_frame_sync<W: std::io::Write>(
        writer: &mut W,
        msg_type: MessageType,
        header: &[u8],
        payload: &[u8],
    ) -> std::io::Result<()> {
        let preamble = Self::build_preamble(msg_type, header.len() as u32, payload.len() as u32)?;

        let bufs = [
            IoSlice::new(&preamble),
            IoSlice::new(header),
            IoSlice::new(payload),
        ];

        let total_len = preamble.len() + header.len() + payload.len();
        let written = writer.write_vectored(&bufs)?;
        if written != total_len {
            return Err(io::Error::new(
                io::ErrorKind::WriteZero,
                format!("incomplete write: wrote {} of {} bytes", written, total_len),
            ));
        }
        Ok(())
    }

    /// Validate that lengths are reasonable
    fn validate_lengths(header_len: u32, payload_len: u32) -> io::Result<()> {
        let total_len = header_len
            .checked_add(payload_len)
            .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidData, "Frame size overflow"))?;

        if total_len > MAX_FRAME_SIZE {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!(
                    "Frame size {} exceeds maximum {}",
                    total_len, MAX_FRAME_SIZE
                ),
            ));
        }

        Ok(())
    }
}

impl Default for TcpFrameCodec {
    fn default() -> Self {
        Self::new()
    }
}

impl Decoder for TcpFrameCodec {
    type Item = (MessageType, Bytes, Bytes);
    type Error = io::Error;

    fn decode(&mut self, src: &mut BytesMut) -> Result<Option<Self::Item>, Self::Error> {
        loop {
            match self.state {
                DecodeState::AwaitingHeader => {
                    // Need at least MIN_HEADER_SIZE bytes
                    if src.len() < MIN_HEADER_SIZE {
                        return Ok(None);
                    }

                    // Parse header without consuming bytes yet
                    let schema_version = u16::from_be_bytes([src[0], src[1]]);
                    let frame_type_byte = src[2];
                    let header_len = u32::from_be_bytes([src[3], src[4], src[5], src[6]]);
                    let payload_len = u32::from_be_bytes([src[7], src[8], src[9], src[10]]);

                    // Validate schema version
                    if schema_version != SCHEMA_VERSION_V1 {
                        return Err(io::Error::new(
                            io::ErrorKind::InvalidData,
                            format!(
                                "Unsupported schema version: {} (expected {})",
                                schema_version, SCHEMA_VERSION_V1
                            ),
                        ));
                    }

                    // Parse frame type
                    let frame_type = MessageType::from_u8(frame_type_byte).ok_or_else(|| {
                        io::Error::new(
                            io::ErrorKind::InvalidData,
                            format!("Invalid frame type: {}", frame_type_byte),
                        )
                    })?;

                    // Validate lengths before allocating/waiting
                    Self::validate_lengths(header_len, payload_len)?;

                    // Advance buffer past header
                    src.advance(MIN_HEADER_SIZE);

                    // Transition to data state
                    self.state = DecodeState::AwaitingData {
                        frame_type,
                        header_len,
                        payload_len,
                    };
                }

                DecodeState::AwaitingData {
                    frame_type,
                    header_len,
                    payload_len,
                    ..
                } => {
                    let total_data_len = (header_len + payload_len) as usize;

                    // Wait for full data
                    if src.len() < total_data_len {
                        return Ok(None);
                    }

                    // Zero-copy: split buffer into header and payload slices
                    let header = src.split_to(header_len as usize).freeze();
                    let payload = src.split_to(payload_len as usize).freeze();

                    // Reset state for next frame
                    self.state = DecodeState::AwaitingHeader;

                    return Ok(Some((frame_type, header, payload)));
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Test helper to encode a frame into a Vec<u8> for verification (async)
    async fn encode_frame_to_bytes(
        msg_type: MessageType,
        header: &[u8],
        payload: &[u8],
    ) -> io::Result<Vec<u8>> {
        let mut buf = Vec::new();
        TcpFrameCodec::encode_frame(&mut buf, msg_type, header, payload).await?;
        Ok(buf)
    }

    /// Test helper to encode a frame into a Vec<u8> for verification (sync)
    fn encode_frame_to_bytes_sync(
        msg_type: MessageType,
        header: &[u8],
        payload: &[u8],
    ) -> io::Result<Vec<u8>> {
        let mut buf = Vec::new();
        TcpFrameCodec::encode_frame_sync(&mut buf, msg_type, header, payload)?;
        Ok(buf)
    }

    /// Helper to create raw frames with arbitrary parameters for negative testing.
    ///
    /// This function bypasses normal validation and encoding logic to create
    /// intentionally invalid frames (wrong schema version, oversized frames, etc.)
    /// for testing error handling paths. Use `encode_frame_to_bytes()` for
    /// testing valid frame construction.
    fn create_unsafe_frame(
        schema_version: u16,
        frame_type: MessageType,
        header: &[u8],
        payload: &[u8],
    ) -> BytesMut {
        let mut buf = BytesMut::new();
        buf.extend_from_slice(&schema_version.to_be_bytes());
        buf.extend_from_slice(&[frame_type.as_u8()]);
        buf.extend_from_slice(&(header.len() as u32).to_be_bytes());
        buf.extend_from_slice(&(payload.len() as u32).to_be_bytes());
        buf.extend_from_slice(header);
        buf.extend_from_slice(payload);
        buf
    }

    #[test]
    fn test_decode_message_frame() {
        let mut codec = TcpFrameCodec::new();
        let header = b"test-header";
        let payload = b"test-payload-data";

        let framed = encode_frame_to_bytes_sync(MessageType::Message, header, payload).unwrap();
        let mut buf = BytesMut::from(&framed[..]);

        let result = codec.decode(&mut buf).unwrap();
        assert!(result.is_some());

        let (msg_type, decoded_header, decoded_payload) = result.unwrap();
        assert_eq!(msg_type, MessageType::Message);
        assert_eq!(decoded_header, Bytes::from(header.as_ref()));
        assert_eq!(decoded_payload, Bytes::from(payload.as_ref()));
    }

    #[test]
    fn test_decode_all_frame_types() {
        let frame_types = [
            MessageType::Message,
            MessageType::Response,
            MessageType::Ack,
            MessageType::Event,
        ];

        for frame_type in &frame_types {
            let mut codec = TcpFrameCodec::new();
            let header = b"header";
            let payload = b"payload";

            let framed = encode_frame_to_bytes_sync(*frame_type, header, payload).unwrap();
            let mut buf = BytesMut::from(&framed[..]);

            let result = codec.decode(&mut buf).unwrap();
            assert!(result.is_some());

            let (decoded_type, _, _) = result.unwrap();
            assert_eq!(decoded_type, *frame_type);
        }
    }

    #[test]
    fn test_decode_empty_payload() {
        let mut codec = TcpFrameCodec::new();
        let header = b"ack-header";
        let payload = b"";

        let framed = encode_frame_to_bytes_sync(MessageType::Ack, header, payload).unwrap();
        let mut buf = BytesMut::from(&framed[..]);

        let result = codec.decode(&mut buf).unwrap();
        assert!(result.is_some());

        let (msg_type, decoded_header, decoded_payload) = result.unwrap();
        assert_eq!(msg_type, MessageType::Ack);
        assert_eq!(&decoded_header[..], header);
        assert_eq!(decoded_payload.len(), 0);
    }

    #[test]
    fn test_decode_partial_frame() {
        let mut codec = TcpFrameCodec::new();
        let header = b"test-header";
        let payload = b"test-payload";

        let full_frame = encode_frame_to_bytes_sync(MessageType::Message, header, payload).unwrap();

        // Send only first 5 bytes (partial header)
        let mut buf = BytesMut::from(&full_frame[..5]);
        let result = codec.decode(&mut buf).unwrap();
        assert!(result.is_none()); // Not enough data

        // Send rest of header
        buf.extend_from_slice(&full_frame[5..MIN_HEADER_SIZE]);
        let result = codec.decode(&mut buf).unwrap();
        assert!(result.is_none()); // Header parsed, but data not yet available

        // Send complete data
        buf.extend_from_slice(&full_frame[MIN_HEADER_SIZE..]);
        let result = codec.decode(&mut buf).unwrap();
        assert!(result.is_some());

        let (msg_type, decoded_header, decoded_payload) = result.unwrap();
        assert_eq!(msg_type, MessageType::Message);
        assert_eq!(&decoded_header[..], header);
        assert_eq!(&decoded_payload[..], payload);
    }

    #[test]
    fn test_decode_invalid_schema_version() {
        let mut codec = TcpFrameCodec::new();
        let header = b"header";
        let payload = b"payload";

        let mut buf = create_unsafe_frame(999, MessageType::Message, header, payload);

        let result = codec.decode(&mut buf);
        assert!(result.is_err());
        assert!(
            result
                .unwrap_err()
                .to_string()
                .contains("Unsupported schema version")
        );
    }

    #[test]
    fn test_decode_invalid_frame_type() {
        let mut codec = TcpFrameCodec::new();
        let mut buf = BytesMut::new();

        // Create frame with invalid type byte (255)
        buf.extend_from_slice(&SCHEMA_VERSION_V1.to_be_bytes());
        buf.extend_from_slice(&[255u8]); // Invalid frame type
        buf.extend_from_slice(&10u32.to_be_bytes()); // header len
        buf.extend_from_slice(&10u32.to_be_bytes()); // payload len

        let result = codec.decode(&mut buf);
        assert!(result.is_err());
        assert!(
            result
                .unwrap_err()
                .to_string()
                .contains("Invalid frame type")
        );
    }

    #[test]
    fn test_decode_frame_too_large() {
        let mut codec = TcpFrameCodec::new();
        let mut buf = BytesMut::new();

        // Create frame that exceeds MAX_FRAME_SIZE
        buf.extend_from_slice(&SCHEMA_VERSION_V1.to_be_bytes());
        buf.extend_from_slice(&[MessageType::Message.as_u8()]);
        buf.extend_from_slice(&(MAX_FRAME_SIZE / 2 + 1).to_be_bytes());
        buf.extend_from_slice(&(MAX_FRAME_SIZE / 2 + 1).to_be_bytes());

        let result = codec.decode(&mut buf);
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("exceeds maximum"));
    }

    #[test]
    fn test_decode_multiple_frames() {
        let mut codec = TcpFrameCodec::new();
        let mut buf = BytesMut::new();

        // Add two frames to buffer
        let frame1 =
            encode_frame_to_bytes_sync(MessageType::Message, b"header1", b"payload1").unwrap();
        let frame2 =
            encode_frame_to_bytes_sync(MessageType::Response, b"header2", b"payload2").unwrap();
        buf.extend_from_slice(&frame1);
        buf.extend_from_slice(&frame2);

        // Decode first frame
        let result = codec.decode(&mut buf).unwrap();
        assert!(result.is_some());
        let (msg_type, header, payload) = result.unwrap();
        assert_eq!(msg_type, MessageType::Message);
        assert_eq!(&header[..], b"header1");
        assert_eq!(&payload[..], b"payload1");

        // Decode second frame
        let result = codec.decode(&mut buf).unwrap();
        assert!(result.is_some());
        let (msg_type, header, payload) = result.unwrap();
        assert_eq!(msg_type, MessageType::Response);
        assert_eq!(&header[..], b"header2");
        assert_eq!(&payload[..], b"payload2");

        // No more frames
        assert!(buf.is_empty());
    }

    #[test]
    fn test_zero_copy_bytes_share_buffer() {
        let mut codec = TcpFrameCodec::new();
        let header = b"shared-header";
        let payload = b"shared-payload";

        let framed = encode_frame_to_bytes_sync(MessageType::Message, header, payload).unwrap();
        let mut buf = BytesMut::from(&framed[..]);

        let result = codec.decode(&mut buf).unwrap().unwrap();
        let (_, decoded_header, decoded_payload) = result;

        // Verify the slices contain correct data
        assert_eq!(&decoded_header[..], header);
        assert_eq!(&decoded_payload[..], payload);

        // Clone should be cheap (just RC increment)
        let header_clone = decoded_header.clone();
        let payload_clone = decoded_payload.clone();

        assert_eq!(decoded_header, header_clone);
        assert_eq!(decoded_payload, payload_clone);
    }

    #[test]
    fn test_encode_frame() {
        let header = b"test-header";
        let payload = b"test-payload";

        let framed = encode_frame_to_bytes_sync(MessageType::Message, header, payload).unwrap();

        // Verify frame structure
        assert_eq!(framed.len(), MIN_HEADER_SIZE + header.len() + payload.len());

        // Verify header fields
        assert_eq!(
            u16::from_be_bytes([framed[0], framed[1]]),
            SCHEMA_VERSION_V1
        );
        assert_eq!(framed[2], MessageType::Message.as_u8());
        assert_eq!(
            u32::from_be_bytes([framed[3], framed[4], framed[5], framed[6]]),
            header.len() as u32
        );
        assert_eq!(
            u32::from_be_bytes([framed[7], framed[8], framed[9], framed[10]]),
            payload.len() as u32
        );

        // Verify data
        assert_eq!(
            &framed[MIN_HEADER_SIZE..MIN_HEADER_SIZE + header.len()],
            header
        );
        assert_eq!(&framed[MIN_HEADER_SIZE + header.len()..], payload);
    }

    #[test]
    fn test_encode_all_message_types() {
        let header = b"header";
        let payload = b"payload";

        for msg_type in &[
            MessageType::Message,
            MessageType::Response,
            MessageType::Ack,
            MessageType::Event,
        ] {
            let framed = encode_frame_to_bytes_sync(*msg_type, header, payload).unwrap();
            assert_eq!(framed[2], msg_type.as_u8());
        }
    }

    #[test]
    fn test_encode_empty_payload() {
        let header = b"ack-header";
        let payload = b"";

        let framed = encode_frame_to_bytes_sync(MessageType::Ack, header, payload).unwrap();

        assert_eq!(framed.len(), MIN_HEADER_SIZE + header.len());
        assert_eq!(
            u32::from_be_bytes([framed[7], framed[8], framed[9], framed[10]]),
            0
        );
    }

    #[test]
    fn test_encode_frame_too_large() {
        let header = vec![0u8; (MAX_FRAME_SIZE / 2 + 1) as usize];
        let payload = vec![0u8; (MAX_FRAME_SIZE / 2 + 1) as usize];

        let result = encode_frame_to_bytes_sync(MessageType::Message, &header, &payload);
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("exceeds maximum"));
    }

    #[test]
    fn test_round_trip_encode_decode() {
        let mut codec = TcpFrameCodec::new();
        let header = b"round-trip-header";
        let payload = b"round-trip-payload-data";

        // Encode
        let framed = encode_frame_to_bytes_sync(MessageType::Response, header, payload).unwrap();

        // Decode
        let mut buf = BytesMut::from(&framed[..]);
        let result = codec.decode(&mut buf).unwrap();
        assert!(result.is_some());

        let (msg_type, decoded_header, decoded_payload) = result.unwrap();
        assert_eq!(msg_type, MessageType::Response);
        assert_eq!(&decoded_header[..], header);
        assert_eq!(&decoded_payload[..], payload);
    }

    #[test]
    fn test_round_trip_all_types() {
        let types = [
            MessageType::Message,
            MessageType::Response,
            MessageType::Ack,
            MessageType::Event,
        ];

        for msg_type in &types {
            let mut codec = TcpFrameCodec::new();
            let header = b"header";
            let payload = b"payload";

            let framed = encode_frame_to_bytes_sync(*msg_type, header, payload).unwrap();

            let mut buf = BytesMut::from(&framed[..]);
            let result = codec.decode(&mut buf).unwrap().unwrap();

            assert_eq!(result.0, *msg_type);
            assert_eq!(&result.1[..], header);
            assert_eq!(&result.2[..], payload);
        }
    }

    #[test]
    fn test_encode_frame_sync() {
        let header = b"sync-header";
        let payload = b"sync-payload";

        let framed = encode_frame_to_bytes_sync(MessageType::Message, header, payload).unwrap();

        // Verify frame structure
        assert_eq!(framed.len(), MIN_HEADER_SIZE + header.len() + payload.len());

        // Verify preamble fields
        assert_eq!(
            u16::from_be_bytes([framed[0], framed[1]]),
            SCHEMA_VERSION_V1
        );
        assert_eq!(framed[2], MessageType::Message.as_u8());
        assert_eq!(
            u32::from_be_bytes([framed[3], framed[4], framed[5], framed[6]]),
            header.len() as u32
        );
        assert_eq!(
            u32::from_be_bytes([framed[7], framed[8], framed[9], framed[10]]),
            payload.len() as u32
        );

        // Verify data
        assert_eq!(
            &framed[MIN_HEADER_SIZE..MIN_HEADER_SIZE + header.len()],
            header
        );
        assert_eq!(&framed[MIN_HEADER_SIZE + header.len()..], payload);
    }

    #[test]
    fn test_sync_async_produce_same_output() {
        let header = b"test-header";
        let payload = b"test-payload";

        // Encode with sync version
        let sync_framed =
            encode_frame_to_bytes_sync(MessageType::Response, header, payload).unwrap();

        // Encode with async version (using tokio runtime)
        let async_framed = tokio::runtime::Runtime::new()
            .unwrap()
            .block_on(encode_frame_to_bytes(
                MessageType::Response,
                header,
                payload,
            ))
            .unwrap();

        // Both should produce identical output
        assert_eq!(sync_framed, async_framed);
    }
}
