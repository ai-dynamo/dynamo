// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Binary frame format for event transport
//!
//! - **Fixed 5-byte header**
//! - **Versioned**: Protocol evolution support
//! - **Payload length**: Enables proper frame boundary detection

use bytes::{Buf, BufMut, Bytes, BytesMut};
use thiserror::Error;

/// Frame protocol version
pub const FRAME_VERSION: u8 = 1;

/// Fixed header size in bytes
pub const FRAME_HEADER_SIZE: usize = 5;

/// Frame encoding/decoding errors
#[derive(Debug, Error)]
pub enum FrameError {
    #[error("Incomplete frame header: expected {FRAME_HEADER_SIZE} bytes, got {0} bytes")]
    IncompleteHeader(usize),

    #[error("Incomplete frame payload: expected {expected} bytes, got {available} bytes")]
    IncompletePayload { expected: usize, available: usize },

    #[error("Unsupported protocol version: {0} (expected {FRAME_VERSION})")]
    UnsupportedVersion(u8),

    #[error("Frame too large: {0} bytes exceeds maximum")]
    FrameTooLarge(usize),
}

/// Frame header (5 bytes fixed)
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FrameHeader {
    /// Protocol version (currently 1)
    pub version: u8,
    /// Payload length in bytes
    pub payload_len: u32,
}

impl FrameHeader {
    /// Encode header to bytes
    pub fn encode(&self, buf: &mut BytesMut) {
        buf.put_u8(self.version);
        buf.put_u32(self.payload_len);
    }

    /// Decode header from bytes
    pub fn decode(buf: &mut impl Buf) -> Result<Self, FrameError> {
        if buf.remaining() < FRAME_HEADER_SIZE {
            return Err(FrameError::IncompleteHeader(buf.remaining()));
        }

        let version = buf.get_u8();
        if version != FRAME_VERSION {
            return Err(FrameError::UnsupportedVersion(version));
        }

        let payload_len = buf.get_u32();

        Ok(FrameHeader {
            version,
            payload_len,
        })
    }

    /// Get total frame size (header + payload)
    pub fn frame_size(&self) -> usize {
        FRAME_HEADER_SIZE + self.payload_len as usize
    }
}

/// Complete frame (header + payload)
#[derive(Debug, Clone)]
pub struct Frame {
    pub header: FrameHeader,
    pub payload: Bytes,
}

impl Frame {
    pub fn new(payload: Bytes) -> Self {
        Self {
            header: FrameHeader {
                version: FRAME_VERSION,
                payload_len: payload.len() as u32,
            },
            payload,
        }
    }

    /// Encode frame to wire format
    pub fn encode(&self) -> Bytes {
        let mut buf = BytesMut::with_capacity(self.header.frame_size());
        self.header.encode(&mut buf);
        buf.put(self.payload.clone());
        buf.freeze()
    }

    /// Decode frame from wire format
    pub fn decode(mut buf: impl Buf) -> Result<Self, FrameError> {
        let header = FrameHeader::decode(&mut buf)?;

        let payload_len = header.payload_len as usize;
        if buf.remaining() < payload_len {
            return Err(FrameError::IncompletePayload {
                expected: payload_len,
                available: buf.remaining(),
            });
        }

        let payload = buf.copy_to_bytes(payload_len);

        Ok(Frame { header, payload })
    }

    pub fn size(&self) -> usize {
        self.header.frame_size()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_frame_header_encode_decode() {
        let header = FrameHeader {
            version: FRAME_VERSION,
            payload_len: 1024,
        };

        let mut buf = BytesMut::new();
        header.encode(&mut buf);

        assert_eq!(buf.len(), FRAME_HEADER_SIZE);

        let decoded = FrameHeader::decode(&mut buf).unwrap();
        assert_eq!(decoded.version, header.version);
        assert_eq!(decoded.payload_len, header.payload_len);
    }

    #[test]
    fn test_frame_encode_decode_roundtrip() {
        let payload = Bytes::from("hello world");
        let frame = Frame::new(payload.clone());

        let encoded = frame.encode();
        let decoded = Frame::decode(encoded).unwrap();

        assert_eq!(decoded.header.version, FRAME_VERSION);
        assert_eq!(decoded.payload, payload);
    }

    #[test]
    fn test_frame_error_incomplete_header() {
        let buf = Bytes::from(vec![1, 2, 3]); // Only 3 bytes
        let result = Frame::decode(buf);
        assert!(matches!(result, Err(FrameError::IncompleteHeader(3))));
    }

    #[test]
    fn test_frame_error_incomplete_payload() {
        let mut buf = BytesMut::new();
        let header = FrameHeader {
            version: FRAME_VERSION,
            payload_len: 1000, // Claims 1000 bytes
        };
        header.encode(&mut buf);
        buf.put_slice(b"short"); // Only 5 bytes provided

        let result = Frame::decode(buf.freeze());
        assert!(matches!(
            result,
            Err(FrameError::IncompletePayload {
                expected: 1000,
                available: 5
            })
        ));
    }

    #[test]
    fn test_frame_error_unsupported_version() {
        let mut buf = BytesMut::new();
        buf.put_u8(99); // Invalid version
        buf.put_u32(0); // payload_len

        let result = FrameHeader::decode(&mut buf);
        assert!(matches!(result, Err(FrameError::UnsupportedVersion(99))));
    }

    #[test]
    fn test_zero_length_payload() {
        let payload = Bytes::new();
        let frame = Frame::new(payload.clone());

        let encoded = frame.encode();
        assert_eq!(encoded.len(), FRAME_HEADER_SIZE);

        let decoded = Frame::decode(encoded).unwrap();
        assert_eq!(decoded.payload.len(), 0);
    }
}
