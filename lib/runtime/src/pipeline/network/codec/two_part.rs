// SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use bytes::{Buf, BufMut, Bytes, BytesMut};
use tokio_util::codec::{Decoder, Encoder};
use xxhash_rust::xxh3::xxh3_64;

use crate::pipeline::error::TwoPartCodecError;

#[derive(Clone, Default)]
pub struct TwoPartCodec {
    max_message_size: Option<usize>,
}

impl TwoPartCodec {
    pub fn new(max_message_size: Option<usize>) -> Self {
        TwoPartCodec { max_message_size }
    }

    /// Encodes a `TwoPartMessage` into `Bytes`, enforcing `max_message_size`.
    pub fn encode_message(&self, msg: TwoPartMessage) -> Result<Bytes, TwoPartCodecError> {
        let mut buf = BytesMut::new();
        let mut codec = self.clone();
        codec.encode(msg, &mut buf)?;
        Ok(buf.freeze())
    }

    /// Decodes a `TwoPartMessage` from `Bytes`, enforcing `max_message_size`.
    pub fn decode_message(&self, data: Bytes) -> Result<TwoPartMessage, TwoPartCodecError> {
        let mut buf = BytesMut::from(&data[..]);
        let mut codec = self.clone();
        match codec.decode(&mut buf)? {
            Some(msg) => Ok(msg),
            None => Err(TwoPartCodecError::InvalidMessage(
                "No message decoded".to_string(),
            )),
        }
    }
}

impl Decoder for TwoPartCodec {
    type Item = TwoPartMessage;
    type Error = TwoPartCodecError;

    fn decode(&mut self, src: &mut BytesMut) -> Result<Option<Self::Item>, Self::Error> {
        // Need at least 24 bytes (header_len, body_len, checksum)
        if src.len() < 24 {
            return Ok(None);
        }

        // Use a cursor to read lengths and checksum without modifying the buffer
        let mut cursor = &src[..];

        let header_len = cursor.get_u64() as usize;
        let body_len = cursor.get_u64() as usize;
        let _checksum = cursor.get_u64();

        let total_len = 24 + header_len + body_len;

        // Check if total_len exceeds max_message_size
        if let Some(max_size) = self.max_message_size
            && total_len > max_size
        {
            return Err(TwoPartCodecError::MessageTooLarge(total_len, max_size));
        }

        // Check if enough data is available
        if src.len() < total_len {
            return Ok(None);
        }

        // Advance the buffer past the lengths and checksum
        src.advance(24);

        #[cfg(debug_assertions)]
        {
            // If the server sent a dummy checksum, skip it.
            if _checksum != 0 {
                let bytes_to_hash =
                    header_len
                        .checked_add(body_len)
                        .ok_or(TwoPartCodecError::InvalidMessage(
                            "Message exceeds max allowed length.".to_string(),
                        ))?;

                let data_to_hash = &src[..bytes_to_hash];

                let computed_checksum = xxh3_64(data_to_hash);

                // Compare checksums
                if _checksum != computed_checksum {
                    return Err(TwoPartCodecError::ChecksumMismatch);
                }
            }
        }

        // Read header and body data
        let header = src.split_to(header_len).freeze();
        let data = src.split_to(body_len).freeze();

        Ok(Some(TwoPartMessage { header, data }))
    }
}

impl Encoder<TwoPartMessage> for TwoPartCodec {
    type Error = TwoPartCodecError;

    fn encode(&mut self, item: TwoPartMessage, dst: &mut BytesMut) -> Result<(), Self::Error> {
        let header_len = item.header.len();
        let body_len = item.data.len();

        let total_len = 24 + header_len + body_len; // 24 bytes for lengths and checksum

        // Check if total_len exceeds max_message_size
        if let Some(max_size) = self.max_message_size
            && total_len > max_size
        {
            return Err(TwoPartCodecError::MessageTooLarge(total_len, max_size));
        }

        dst.put_u64(header_len as u64);
        dst.put_u64(body_len as u64);

        // Only compute the checksum in debug mode.
        // If we're in release mode, put a dummy value.
        #[cfg(debug_assertions)]
        {
            // Compute checksum of the data
            let mut data_to_hash = BytesMut::with_capacity(header_len + body_len);
            data_to_hash.extend_from_slice(&item.header);
            data_to_hash.extend_from_slice(&item.data);
            let checksum = xxh3_64(&data_to_hash);

            dst.put_u64(checksum);
        }
        #[cfg(not(debug_assertions))]
        {
            dst.put_u64(0);
        }

        // Write header and body
        dst.put_slice(&item.header);
        dst.put_slice(&item.data);

        Ok(())
    }
}

pub enum TwoPartMessageType {
    HeaderOnly(Bytes),
    DataOnly(Bytes),
    HeaderAndData(Bytes, Bytes),
    Empty,
}

#[derive(Clone, Debug)]
pub struct TwoPartMessage {
    pub header: Bytes,
    pub data: Bytes,
}

impl TwoPartMessage {
    pub fn new(header: Bytes, data: Bytes) -> Self {
        TwoPartMessage { header, data }
    }

    pub fn from_header(header: Bytes) -> Self {
        TwoPartMessage {
            header,
            data: Bytes::new(),
        }
    }

    pub fn from_data(data: Bytes) -> Self {
        TwoPartMessage {
            header: Bytes::new(),
            data,
        }
    }

    pub fn from_parts(header: Bytes, data: Bytes) -> Self {
        TwoPartMessage { header, data }
    }

    pub fn parts(&self) -> (&Bytes, &Bytes) {
        (&self.header, &self.data)
    }

    pub fn optional_parts(&self) -> (Option<&Bytes>, Option<&Bytes>) {
        (self.header(), self.data())
    }

    pub fn into_parts(self) -> (Bytes, Bytes) {
        (self.header, self.data)
    }

    pub fn header(&self) -> Option<&Bytes> {
        if self.header.is_empty() {
            None
        } else {
            Some(&self.header)
        }
    }

    pub fn data(&self) -> Option<&Bytes> {
        if self.data.is_empty() {
            None
        } else {
            Some(&self.data)
        }
    }

    pub fn into_message_type(self) -> TwoPartMessageType {
        if self.header.is_empty() && self.data.is_empty() {
            TwoPartMessageType::Empty
        } else if self.header.is_empty() {
            TwoPartMessageType::DataOnly(self.data)
        } else if self.data.is_empty() {
            TwoPartMessageType::HeaderOnly(self.header)
        } else {
            TwoPartMessageType::HeaderAndData(self.header, self.data)
        }
    }
}

#[cfg(test)]
mod tests {
    use std::io::Cursor;
    use std::pin::Pin;
    use std::task::{Context, Poll};

    use bytes::{Bytes, BytesMut};
    use futures::StreamExt;
    use tokio::io::AsyncRead;
    use tokio::io::ReadBuf;
    use tokio_util::codec::{Decoder, FramedRead};

    use super::*;

    /// Test encoding and decoding of a message with both header and data.
    #[test]
    fn test_message_with_header_and_data() {
        // Create a message with both header and data.
        let header_data = Bytes::from("header data");
        let data = Bytes::from("body data");
        let message = TwoPartMessage::from_parts(header_data.clone(), data.clone());

        let codec = TwoPartCodec::new(None);

        // Encode the message.
        let encoded = codec.encode_message(message).unwrap();

        // Decode the message.
        let decoded = codec.decode_message(encoded).unwrap();

        // Verify the decoded message.
        assert_eq!(decoded.header, header_data);
        assert_eq!(decoded.data, data);
    }

    /// Test encoding and decoding of a message with only header.
    #[test]
    fn test_message_with_only_header() {
        let header_data = Bytes::from("header only");
        let message = TwoPartMessage::from_header(header_data.clone());

        let codec = TwoPartCodec::new(None);

        // Encode the message.
        let encoded = codec.encode_message(message).unwrap();

        // Decode the message.
        let decoded = codec.decode_message(encoded).unwrap();

        // Verify the decoded message.
        assert_eq!(decoded.header, header_data);
        assert!(decoded.data.is_empty());
    }

    /// Test encoding and decoding of a message with only data.
    #[test]
    fn test_message_with_only_data() {
        let data = Bytes::from("data only");
        let message = TwoPartMessage::from_data(data.clone());

        let codec = TwoPartCodec::new(None);

        // Encode the message.
        let encoded = codec.encode_message(message).unwrap();

        // Decode the message.
        let decoded = codec.decode_message(encoded).unwrap();

        // Verify the decoded message.
        assert!(decoded.header.is_empty());
        assert_eq!(decoded.data, data);
    }

    /// Test encoding and decoding of an empty message.
    #[test]
    fn test_empty_message() {
        let message = TwoPartMessage::from_parts(Bytes::new(), Bytes::new());

        let codec = TwoPartCodec::new(None);

        // Encode the message.
        let encoded = codec.encode_message(message).unwrap();

        // Decode the message.
        let decoded = codec.decode_message(encoded).unwrap();

        // Verify the decoded message.
        assert!(decoded.header.is_empty());
        assert!(decoded.data.is_empty());
    }

    /// Test encoding and decoding of a message under max_message_size.
    #[test]
    fn test_message_under_max_size() {
        let max_size = 1024; // Set max_message_size to 1024 bytes

        // Create a message smaller than max_size
        let header_data = Bytes::from(vec![b'h'; 100]);
        let body_data = Bytes::from(vec![b'd'; 200]);
        let message = TwoPartMessage::from_parts(header_data.clone(), body_data.clone());

        let codec = TwoPartCodec::new(Some(max_size));

        // Encode the message
        let encoded = codec.encode_message(message.clone()).unwrap();

        // Decode the message
        let decoded = codec.decode_message(encoded).unwrap();

        // Verify the decoded message
        assert_eq!(decoded.header, header_data);
        assert_eq!(decoded.data, body_data);
    }

    /// Test encoding and decoding of a message exactly at max_message_size.
    #[test]
    fn test_message_exactly_at_max_size() {
        let max_size = 1024; // Set max_message_size to 1024 bytes

        // Calculate the sizes
        let lengths_size = 24; // 8 bytes for header_len, 8 bytes for body_len, 8 bytes for checksum
        let data_size = max_size - lengths_size; // Total data size to reach max_size

        // Split data_size between header and body
        let header_size = data_size / 2;
        let body_size = data_size - header_size;

        // Create header and body data
        let header_data = Bytes::from(vec![b'h'; header_size]);
        let body_data = Bytes::from(vec![b'd'; body_size]);

        let message = TwoPartMessage::from_parts(header_data.clone(), body_data.clone());

        let codec = TwoPartCodec::new(Some(max_size));

        // Encode the message
        let encoded = codec.encode_message(message.clone()).unwrap();

        // The length of encoded should be exactly max_size
        assert_eq!(encoded.len(), max_size);

        // Decode the message
        let decoded = codec.decode_message(encoded).unwrap();

        // Verify the decoded message
        assert_eq!(decoded.header, header_data);
        assert_eq!(decoded.data, body_data);
    }

    /// Test encoding of a message over max_message_size.
    #[test]
    fn test_message_over_max_size() {
        let max_size = 1024; // Set max_message_size to 1024 bytes

        // Create a message larger than max_size
        let data_size = max_size - 24 + 1; // Exceed max_size by 1 byte
        let header_size = data_size / 2;
        let body_size = data_size - header_size;

        let header_data = Bytes::from(vec![b'h'; header_size]);
        let body_data = Bytes::from(vec![b'd'; body_size]);

        let message = TwoPartMessage::from_parts(header_data, body_data);

        let codec = TwoPartCodec::new(Some(max_size));

        // Attempt to encode the message
        let result = codec.encode_message(message);

        // Expect an error
        assert!(result.is_err());

        // Verify the error is MessageTooLarge
        if let Err(TwoPartCodecError::MessageTooLarge(size, max)) = result {
            assert_eq!(size, data_size + 24); // Total size including lengths and checksum
            assert_eq!(max, max_size);
        } else {
            panic!("Expected MessageTooLarge error");
        }
    }

    /// Test decoding of a message over max_message_size.
    #[test]
    fn test_decoding_message_over_max_size() {
        let max_size = 1024; // Set max_message_size to 1024 bytes

        // Create a message larger than max_size
        let data_size = max_size - 24 + 1; // Exceed max_size by 1 byte
        let header_size = data_size / 2;
        let body_size = data_size - header_size;

        let header_data = Bytes::from(vec![b'h'; header_size]);
        let body_data = Bytes::from(vec![b'd'; body_size]);

        let message = TwoPartMessage::from_parts(header_data.clone(), body_data.clone());

        let codec = TwoPartCodec::new(None); // No size limit during encoding

        // Encode the message
        let encoded = codec.encode_message(message).unwrap();

        let codec_with_limit = TwoPartCodec::new(Some(max_size));

        // Attempt to decode the message with max_message_size limit
        let result = codec_with_limit.decode_message(encoded);

        // Expect an error
        assert!(result.is_err());

        // Verify the error is MessageTooLarge
        if let Err(TwoPartCodecError::MessageTooLarge(size, max)) = result {
            assert_eq!(size, data_size + 24); // Total size including lengths and checksum
            assert_eq!(max, max_size);
        } else {
            panic!("Expected MessageTooLarge error");
        }
    }

    /// Test decoding of a message with checksum mismatch.
    #[test]
    // Checksum only computed in debug mode, so only test in debug mode.
    #[cfg(debug_assertions)]
    fn test_checksum_mismatch() {
        // Create a message
        let header_data = Bytes::from("header data");
        let data = Bytes::from("body data");
        let message = TwoPartMessage::from_parts(header_data.clone(), data.clone());

        let codec = TwoPartCodec::new(None);

        // Encode the message
        let encoded = codec.encode_message(message).unwrap();

        // Corrupt the data to cause checksum mismatch
        let mut encoded = BytesMut::from(encoded);
        let len = encoded.len();
        encoded[len - 1] ^= 0xFF; // Flip the last byte

        // Attempt to decode
        let result = codec.decode_message(encoded.into());

        // Expect an error
        assert!(result.is_err());

        // Verify the error is ChecksumMismatch
        if let Err(TwoPartCodecError::ChecksumMismatch) = result {
            // Test passed
        } else {
            panic!("Expected ChecksumMismatch error");
        }
    }

    /// Test partial data arrival and ensure decoder waits for full message.
    #[test]
    fn test_partial_data() {
        let header_data = Bytes::from("header data");
        let data = Bytes::from("body data");
        let message = TwoPartMessage::from_parts(header_data.clone(), data.clone());

        let codec = TwoPartCodec::new(None);

        // Encode the message
        let encoded = codec.encode_message(message).unwrap();

        // Simulate partial data arrival
        let partial_len = encoded.len() - 5;
        let partial_encoded = encoded.slice(0..partial_len);

        // Attempt to decode
        let result = codec.decode_message(partial_encoded);

        // Should return InvalidMessage error
        assert!(result.is_err());

        if let Err(TwoPartCodecError::InvalidMessage(_)) = result {
            // Test passed
        } else {
            panic!("Expected InvalidMessage error");
        }
    }

    /// Test multiple messages concatenated in the same buffer.
    #[test]
    fn test_multiple_messages_in_buffer() {
        let header_data1 = Bytes::from("header1");
        let data1 = Bytes::from("data1");
        let message1 = TwoPartMessage::from_parts(header_data1.clone(), data1.clone());

        let header_data2 = Bytes::from("header2");
        let data2 = Bytes::from("data2");
        let message2 = TwoPartMessage::from_parts(header_data2.clone(), data2.clone());

        let codec = TwoPartCodec::new(None);

        // Encode messages
        let encoded1 = codec.encode_message(message1).unwrap();
        let encoded2 = codec.encode_message(message2).unwrap();

        // Concatenate messages into one buffer
        let mut combined = BytesMut::new();
        combined.extend_from_slice(&encoded1);
        combined.extend_from_slice(&encoded2);

        // Decode messages
        let mut decode_buf = combined;
        let mut codec = codec.clone();

        let decoded_msg1 = codec.decode(&mut decode_buf).unwrap().unwrap();
        let decoded_msg2 = codec.decode(&mut decode_buf).unwrap().unwrap();

        // Verify messages
        assert_eq!(decoded_msg1.header, header_data1);
        assert_eq!(decoded_msg1.data, data1);

        assert_eq!(decoded_msg2.header, header_data2);
        assert_eq!(decoded_msg2.data, data2);
    }

    /// Test simulating reading from a byte stream like a TCP socket.
    #[tokio::test]
    async fn test_streaming_read() {
        // Create messages
        let header_data = Bytes::from("header data");
        let data = Bytes::from("body data");
        let message = TwoPartMessage::from_parts(header_data.clone(), data.clone());

        let codec = TwoPartCodec::new(None);

        // Encode the message
        let encoded = codec.encode_message(message.clone()).unwrap();

        // Simulate reading from a TCP socket
        // We'll use a Cursor over the encoded data to simulate an AsyncRead
        let reader = Cursor::new(encoded.clone());

        // Wrap the reader with the codec
        let mut framed_read = FramedRead::new(reader, codec.clone());

        // Read the message
        if let Some(Ok(decoded_message)) = framed_read.next().await {
            // Verify the decoded message
            assert_eq!(decoded_message.header, header_data);
            assert_eq!(decoded_message.data, data);
        } else {
            panic!("Failed to decode message from stream");
        }
    }

    /// Test simulating partial reads from a TCP socket
    #[tokio::test]
    async fn test_streaming_partial_reads() {
        // Create messages
        let header_data = Bytes::from("header data");
        let data = Bytes::from("body data");
        let message = TwoPartMessage::from_parts(header_data.clone(), data.clone());

        let codec = TwoPartCodec::new(None);

        // Encode the message
        let encoded = codec.encode_message(message.clone()).unwrap();

        // Simulate partial reads
        // We'll create a custom AsyncRead that returns data in small chunks
        struct ChunkedReader {
            data: Bytes,
            pos: usize,
            chunk_size: usize,
        }

        impl AsyncRead for ChunkedReader {
            fn poll_read(
                mut self: Pin<&mut Self>,
                _cx: &mut Context<'_>,
                buf: &mut ReadBuf<'_>,
            ) -> Poll<std::io::Result<()>> {
                if self.pos >= self.data.len() {
                    return Poll::Ready(Ok(()));
                }

                let end = std::cmp::min(self.pos + self.chunk_size, self.data.len());
                let bytes_to_read = &self.data[self.pos..end];
                buf.put_slice(bytes_to_read);
                self.pos = end;

                // if self.pos >= self.data.len() {
                //     Poll::Ready(Ok(()))
                // } else {
                //     Poll::Ready(Ok(()))
                // }
                Poll::Ready(Ok(()))
            }
        }

        let reader = ChunkedReader {
            data: encoded.clone(),
            pos: 0,
            chunk_size: 5, // Read in chunks of 5 bytes
        };

        let mut framed_read = FramedRead::new(reader, codec.clone());

        // Read the message
        if let Some(Ok(decoded_message)) = framed_read.next().await {
            // Verify the decoded message
            assert_eq!(decoded_message.header, header_data);
            assert_eq!(decoded_message.data, data);
        } else {
            panic!("Failed to decode message from stream");
        }
    }

    /// Test handling of corrupted data in a stream
    #[tokio::test]
    // Checksum only computed in debug mode, so only test in debug mode.
    #[cfg(debug_assertions)]
    async fn test_streaming_corrupted_data() {
        // Create messages
        let header_data = Bytes::from("header data");
        let data = Bytes::from("body data");
        let message = TwoPartMessage::from_parts(header_data.clone(), data.clone());

        let codec = TwoPartCodec::new(None);

        // Encode the message
        let encoded = codec.encode_message(message.clone()).unwrap();

        // Corrupt the data
        let mut encoded = BytesMut::from(encoded);
        encoded[30] ^= 0xFF; // Flip a byte in the data

        // Simulate reading from a TCP socket
        let reader = Cursor::new(encoded.clone());

        let mut framed_read = FramedRead::new(reader, codec.clone());

        // Read the message
        if let Some(result) = framed_read.next().await {
            assert!(result.is_err());

            // Verify the error is ChecksumMismatch
            if let Err(TwoPartCodecError::ChecksumMismatch) = result {
                // Test passed
            } else {
                panic!("Expected ChecksumMismatch error");
            }
        } else {
            panic!("Failed to read message from stream");
        }
    }

    /// Test handling of empty streams
    #[tokio::test]
    async fn test_empty_stream() {
        let codec = TwoPartCodec::new(None);

        // Empty reader
        let reader = Cursor::new(Vec::new());

        let mut framed_read = FramedRead::new(reader, codec.clone());

        // Try to read from empty stream
        if let Some(result) = framed_read.next().await {
            panic!("Expected no messages, but got {:?}", result);
        } else {
            // Test passed
        }
    }

    /// Test decoding of multiple messages from a stream
    #[tokio::test]
    async fn test_streaming_multiple_messages() {
        let header_data1 = Bytes::from("header1");
        let data1 = Bytes::from("data1");
        let message1 = TwoPartMessage::from_parts(header_data1.clone(), data1.clone());

        let header_data2 = Bytes::from("header2");
        let data2 = Bytes::from("data2");
        let message2 = TwoPartMessage::from_parts(header_data2.clone(), data2.clone());

        let codec = TwoPartCodec::new(None);

        // Encode messages
        let encoded1 = codec.encode_message(message1.clone()).unwrap();
        let encoded2 = codec.encode_message(message2.clone()).unwrap();

        // Concatenate messages into one buffer
        let mut combined = BytesMut::new();
        combined.extend_from_slice(&encoded1);
        combined.extend_from_slice(&encoded2);

        // Simulate reading from a TCP socket
        let reader = Cursor::new(combined.freeze());

        let mut framed_read = FramedRead::new(reader, codec.clone());

        // Read first message
        if let Some(Ok(decoded_message)) = framed_read.next().await {
            assert_eq!(decoded_message.header, header_data1);
            assert_eq!(decoded_message.data, data1);
        } else {
            panic!("Failed to decode first message from stream");
        }

        // Read second message
        if let Some(Ok(decoded_message)) = framed_read.next().await {
            assert_eq!(decoded_message.header, header_data2);
            assert_eq!(decoded_message.data, data2);
        } else {
            panic!("Failed to decode second message from stream");
        }

        // Ensure no more messages
        if let Some(result) = framed_read.next().await {
            panic!("Expected no more messages, but got {:?}", result);
        }
    }

    /// Test encoding and decoding without max_message_size.
    #[test]
    fn test_message_without_max_size() {
        // Create a large message
        let header_data = Bytes::from(vec![b'h'; 1024 * 1024]); // 1 MB
        let body_data = Bytes::from(vec![b'd'; 1024 * 1024]); // 1 MB

        let message = TwoPartMessage::from_parts(header_data.clone(), body_data.clone());

        let codec = TwoPartCodec::new(None);

        // Encode the message without max_message_size
        let encoded = codec.encode_message(message).unwrap();

        // Decode the message without max_message_size
        let decoded = codec.decode_message(encoded).unwrap();

        // Verify the decoded message
        assert_eq!(decoded.header, header_data);
        assert_eq!(decoded.data, body_data);
    }
}
