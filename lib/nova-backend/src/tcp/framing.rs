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

use bytes::{Buf, BufMut, Bytes, BytesMut};
use std::io;
use tokio_util::codec::{Decoder, Encoder};

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

    /// Encode a frame into a Bytes buffer (zero-copy)
    ///
    /// This method pre-encodes a complete frame with the wire format:
    /// [schema_version][msg_type][header_len][payload_len][header][payload]
    ///
    /// # Arguments
    /// * `msg_type` - The message type (Message, Response, Ack, Event)
    /// * `header` - Header bytes (ownership transferred)
    /// * `payload` - Payload bytes (ownership transferred)
    ///
    /// # Returns
    /// Pre-encoded frame as Bytes ready for sending
    #[inline]
    pub fn encode_frame(msg_type: MessageType, header: Bytes, payload: Bytes) -> io::Result<Bytes> {
        let header_len = header.len() as u32;
        let payload_len = payload.len() as u32;

        // Validate lengths before allocation
        Self::validate_lengths(header_len, payload_len)?;

        // Pre-allocate exact capacity
        let capacity = MIN_HEADER_SIZE + header_len as usize + payload_len as usize;
        let mut buf = BytesMut::with_capacity(capacity);

        // Write frame header
        buf.put_u16(SCHEMA_VERSION_V1);
        buf.put_u8(msg_type.as_u8());
        buf.put_u32(header_len);
        buf.put_u32(payload_len);

        // Append header and payload (zero-copy via put)
        buf.put(header);
        buf.put(payload);

        // Convert to immutable Bytes (zero-cost)
        Ok(buf.freeze())
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

impl Encoder<Bytes> for TcpFrameCodec {
    type Error = io::Error;

    /// Encode pre-framed bytes into the output buffer
    ///
    /// This encoder assumes the input `Bytes` is already framed via `encode_frame()`.
    /// It simply appends the pre-framed bytes to the output buffer for zero-copy sending.
    #[inline]
    fn encode(&mut self, item: Bytes, dst: &mut BytesMut) -> Result<(), Self::Error> {
        dst.extend_from_slice(&item);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Helper to create raw frames with arbitrary parameters for negative testing.
    ///
    /// This function bypasses normal validation and encoding logic to create
    /// intentionally invalid frames (wrong schema version, oversized frames, etc.)
    /// for testing error handling paths. Use `TcpFrameCodec::encode_frame()` for
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
        let header = Bytes::from(&b"test-header"[..]);
        let payload = Bytes::from(&b"test-payload-data"[..]);

        let framed =
            TcpFrameCodec::encode_frame(MessageType::Message, header.clone(), payload.clone())
                .unwrap();
        let mut buf = BytesMut::from(&framed[..]);

        let result = codec.decode(&mut buf).unwrap();
        assert!(result.is_some());

        let (msg_type, decoded_header, decoded_payload) = result.unwrap();
        assert_eq!(msg_type, MessageType::Message);
        assert_eq!(decoded_header, header);
        assert_eq!(decoded_payload, payload);
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
            let header = Bytes::from(&b"header"[..]);
            let payload = Bytes::from(&b"payload"[..]);

            let framed = TcpFrameCodec::encode_frame(*frame_type, header, payload).unwrap();
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
        let header = Bytes::from(&b"ack-header"[..]);
        let payload = Bytes::new();

        let framed =
            TcpFrameCodec::encode_frame(MessageType::Ack, header.clone(), payload.clone()).unwrap();
        let mut buf = BytesMut::from(&framed[..]);

        let result = codec.decode(&mut buf).unwrap();
        assert!(result.is_some());

        let (msg_type, decoded_header, decoded_payload) = result.unwrap();
        assert_eq!(msg_type, MessageType::Ack);
        assert_eq!(decoded_header, header);
        assert_eq!(decoded_payload.len(), 0);
    }

    #[test]
    fn test_decode_partial_frame() {
        let mut codec = TcpFrameCodec::new();
        let header = Bytes::from(&b"test-header"[..]);
        let payload = Bytes::from(&b"test-payload"[..]);

        let full_frame =
            TcpFrameCodec::encode_frame(MessageType::Message, header.clone(), payload.clone())
                .unwrap();

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
        assert_eq!(decoded_header, header);
        assert_eq!(decoded_payload, payload);
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
        let frame1 = TcpFrameCodec::encode_frame(
            MessageType::Message,
            Bytes::from(&b"header1"[..]),
            Bytes::from(&b"payload1"[..]),
        )
        .unwrap();
        let frame2 = TcpFrameCodec::encode_frame(
            MessageType::Response,
            Bytes::from(&b"header2"[..]),
            Bytes::from(&b"payload2"[..]),
        )
        .unwrap();
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
        let header = Bytes::from(&b"shared-header"[..]);
        let payload = Bytes::from(&b"shared-payload"[..]);

        let framed =
            TcpFrameCodec::encode_frame(MessageType::Message, header.clone(), payload.clone())
                .unwrap();
        let mut buf = BytesMut::from(&framed[..]);

        let result = codec.decode(&mut buf).unwrap().unwrap();
        let (_, decoded_header, decoded_payload) = result;

        // Verify the slices contain correct data
        assert_eq!(decoded_header, header);
        assert_eq!(decoded_payload, payload);

        // Clone should be cheap (just RC increment)
        let header_clone = decoded_header.clone();
        let payload_clone = decoded_payload.clone();

        assert_eq!(decoded_header, header_clone);
        assert_eq!(decoded_payload, payload_clone);
    }

    #[test]
    fn test_encode_frame() {
        let header = Bytes::from(&b"test-header"[..]);
        let payload = Bytes::from(&b"test-payload"[..]);

        let framed =
            TcpFrameCodec::encode_frame(MessageType::Message, header.clone(), payload.clone())
                .unwrap();

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
            &header[..]
        );
        assert_eq!(&framed[MIN_HEADER_SIZE + header.len()..], &payload[..]);
    }

    #[test]
    fn test_encode_all_message_types() {
        let header = Bytes::from(&b"header"[..]);
        let payload = Bytes::from(&b"payload"[..]);

        for msg_type in &[
            MessageType::Message,
            MessageType::Response,
            MessageType::Ack,
            MessageType::Event,
        ] {
            let framed =
                TcpFrameCodec::encode_frame(*msg_type, header.clone(), payload.clone()).unwrap();
            assert_eq!(framed[2], msg_type.as_u8());
        }
    }

    #[test]
    fn test_encode_empty_payload() {
        let header = Bytes::from(&b"ack-header"[..]);
        let payload = Bytes::new();

        let framed =
            TcpFrameCodec::encode_frame(MessageType::Ack, header.clone(), payload.clone()).unwrap();

        assert_eq!(framed.len(), MIN_HEADER_SIZE + header.len());
        assert_eq!(
            u32::from_be_bytes([framed[7], framed[8], framed[9], framed[10]]),
            0
        );
    }

    #[test]
    fn test_encode_frame_too_large() {
        let header = Bytes::from(vec![0u8; (MAX_FRAME_SIZE / 2 + 1) as usize]);
        let payload = Bytes::from(vec![0u8; (MAX_FRAME_SIZE / 2 + 1) as usize]);

        let result = TcpFrameCodec::encode_frame(MessageType::Message, header, payload);
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("exceeds maximum"));
    }

    #[test]
    fn test_round_trip_encode_decode() {
        let mut codec = TcpFrameCodec::new();
        let header = Bytes::from(&b"round-trip-header"[..]);
        let payload = Bytes::from(&b"round-trip-payload-data"[..]);

        // Encode
        let framed =
            TcpFrameCodec::encode_frame(MessageType::Response, header.clone(), payload.clone())
                .unwrap();

        // Decode
        let mut buf = BytesMut::from(&framed[..]);
        let result = codec.decode(&mut buf).unwrap();
        assert!(result.is_some());

        let (msg_type, decoded_header, decoded_payload) = result.unwrap();
        assert_eq!(msg_type, MessageType::Response);
        assert_eq!(decoded_header, header);
        assert_eq!(decoded_payload, payload);
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
            let header = Bytes::from(&b"header"[..]);
            let payload = Bytes::from(&b"payload"[..]);

            let framed =
                TcpFrameCodec::encode_frame(*msg_type, header.clone(), payload.clone()).unwrap();

            let mut buf = BytesMut::from(&framed[..]);
            let result = codec.decode(&mut buf).unwrap().unwrap();

            assert_eq!(result.0, *msg_type);
            assert_eq!(result.1, header);
            assert_eq!(result.2, payload);
        }
    }
}
