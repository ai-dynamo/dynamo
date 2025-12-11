// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! # Dynamo Active Message Common

use bytes::{Buf, BufMut, Bytes, BytesMut};
use derive_builder::Builder;
use std::collections::HashMap;
use thiserror::Error;

use super::responses::ResponseId;

const CURRENT_SCHEMA_VERSION: u8 = 1;

#[derive(Debug, Clone)]
pub(crate) struct ActiveMessage {
    pub metadata: MessageMetadata,
    pub payload: Bytes,
}

impl ActiveMessage {
    pub(crate) fn encode(
        self,
    ) -> Result<(Bytes, Bytes, dynamo_nova_backend::MessageType), EncodeError> {
        encode_active_message(self)
    }
}

#[derive(Debug, Clone, Builder)]
#[builder(setter(into))]
pub(crate) struct MessageMetadata {
    #[builder(default = "CURRENT_SCHEMA_VERSION")]
    pub schema_version: u8,
    pub response_type: ResponseType,
    pub response_id: ResponseId,
    pub handler_name: String,
    #[builder(default)]
    pub headers: Option<HashMap<String, String>>,
}

impl MessageMetadata {
    pub(crate) fn new_fire(
        response_id: ResponseId,
        handler_name: String,
        headers: Option<HashMap<String, String>>,
    ) -> Self {
        Self {
            schema_version: CURRENT_SCHEMA_VERSION,
            response_type: ResponseType::FireAndForget,
            response_id,
            handler_name,
            headers,
        }
    }

    pub(crate) fn new_sync(
        response_id: ResponseId,
        handler_name: String,
        headers: Option<HashMap<String, String>>,
    ) -> Self {
        Self {
            schema_version: CURRENT_SCHEMA_VERSION,
            response_type: ResponseType::AckNack,
            response_id,
            handler_name,
            headers,
        }
    }

    pub(crate) fn new_unary(
        response_id: ResponseId,
        handler_name: String,
        headers: Option<HashMap<String, String>>,
    ) -> Self {
        Self {
            schema_version: CURRENT_SCHEMA_VERSION,
            response_type: ResponseType::Unary,
            response_id,
            handler_name,
            headers,
        }
    }
}

#[repr(u8)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum ResponseType {
    /// Indicates an am_send or event_trigger messsage
    /// These types of messages do not expect a response from the remote instance; however,
    /// they do expect a response from the local instance when the message is successfully
    /// sent. This allows for the awaiter to know that the message was successfully sent
    /// or that an error occurred.
    FireAndForget = 0,
    /// Indicates an am_sync message
    /// These types of messages expect a response from the remote instance; or if the transport
    /// has a problem, a local sender side error could also trigger an error response.
    /// This allows for the awaiter to know that the message was sent and processed successfully,
    /// or that an error occurred either locally or remotely.
    AckNack = 1,
    /// Indicates a unary message
    /// These types of messages expect a response from the remote instance; however,
    /// they do not expect a response from the local instance when the message is successfully
    /// sent. This allows for the awaiter to know that the message was successfully sent
    /// and completed, or that an error occurred either locally or remotely.
    Unary = 2,
}

impl TryFrom<u8> for ResponseType {
    type Error = DecodeError;

    fn try_from(value: u8) -> Result<Self, Self::Error> {
        match value {
            0 => Ok(ResponseType::FireAndForget),
            1 => Ok(ResponseType::AckNack),
            2 => Ok(ResponseType::Unary),
            _ => Err(DecodeError::InvalidResponseType(value)),
        }
    }
}

impl ResponseType {
    /// Convert ResponseType to MessageType for routing
    pub(crate) fn to_message_type(self) -> dynamo_nova_backend::MessageType {
        // All active messages are requests, so they all map to MessageType::Message
        // The response will come back as MessageType::Response separately
        dynamo_nova_backend::MessageType::Message
    }
}

#[derive(Debug, Error)]
pub(crate) enum DecodeError {
    #[error("Header too short: expected at least 20 bytes")]
    HeaderTooShort,

    #[error("Invalid handler name length")]
    InvalidHandlerNameLength,

    #[error("Invalid UTF-8 in handler name")]
    InvalidUtf8,

    #[error("Invalid response type: {0}")]
    InvalidResponseType(u8),

    #[error("Invalid headers length")]
    InvalidHeadersLength,

    #[error("Failed to deserialize headers: {0}")]
    HeaderDeserializationError(#[from] rmp_serde::decode::Error),
}

#[derive(Debug, Error)]
pub(crate) enum EncodeError {
    #[error("Handler name too long: {0} bytes exceeds maximum of 65535")]
    HandlerNameTooLong(usize),

    #[error("Header value too large: key '{0}' has value of {1} bytes, max is 1024")]
    HeaderValueTooLarge(String, usize),

    #[error("Total headers too large: {0} bytes exceeds maximum of 16384")]
    TotalHeadersTooLarge(usize),

    #[error("Failed to serialize headers: {0}")]
    HeaderSerializationError(#[from] rmp_serde::encode::Error),
}

pub(crate) fn encode_active_message(
    message: ActiveMessage,
) -> Result<(Bytes, Bytes, dynamo_nova_backend::MessageType), EncodeError> {
    let handler_name_len = message.metadata.handler_name.len();

    // Validate handler name length fits in u16
    if handler_name_len > u16::MAX as usize {
        return Err(EncodeError::HandlerNameTooLong(handler_name_len));
    }

    // Validate and encode headers if present
    let headers_bytes = if let Some(ref headers) = message.metadata.headers {
        // Validate per-header size (1KB max per value)
        for (key, value) in headers.iter() {
            if value.len() > 1024 {
                return Err(EncodeError::HeaderValueTooLarge(key.clone(), value.len()));
            }
        }

        // Serialize headers to MessagePack
        let msgpack_bytes = rmp_serde::to_vec(headers)?;

        // Validate total size (16KB max)
        if msgpack_bytes.len() > 16384 {
            return Err(EncodeError::TotalHeadersTooLarge(msgpack_bytes.len()));
        }

        Some(msgpack_bytes)
    } else {
        None
    };

    // Calculate total header size
    let headers_len = headers_bytes.as_ref().map(|b| b.len()).unwrap_or(0);
    let header_size = 20 + handler_name_len + 2 + headers_len; // +2 for headers_len field
    let mut header = BytesMut::with_capacity(header_size);

    // Encode fixed fields
    header.put_u8(message.metadata.schema_version);
    header.put_u8(message.metadata.response_type as u8);
    header.put_u128_le(message.metadata.response_id.as_u128());
    header.put_u16_le(handler_name_len as u16);
    header.put_slice(message.metadata.handler_name.as_bytes());

    // Encode headers length and bytes (last in header)
    header.put_u16_le(headers_len as u16);
    if let Some(bytes) = headers_bytes {
        header.put_slice(&bytes);
    }

    let message_type = message.metadata.response_type.to_message_type();
    Ok((header.freeze(), message.payload, message_type))
}

pub(crate) fn decode_active_message(
    header: Bytes,
    payload: Bytes,
) -> Result<ActiveMessage, DecodeError> {
    let mut header = header;

    // Validate minimum size (now 22 bytes: original 20 + 2 for headers_len)
    if header.len() < 22 {
        return Err(DecodeError::HeaderTooShort);
    }

    let schema_version = header.get_u8();
    let response_type_raw = header.get_u8();
    let response_id = ResponseId::from_u128(header.get_u128_le());
    let handler_name_len = header.get_u16_le() as usize;

    // Validate handler name length
    if header.remaining() < handler_name_len + 2 {
        // +2 for headers_len field
        return Err(DecodeError::InvalidHandlerNameLength);
    }

    let handler_name_bytes = header.copy_to_bytes(handler_name_len);
    let handler_name =
        String::from_utf8(handler_name_bytes.to_vec()).map_err(|_| DecodeError::InvalidUtf8)?;

    let response_type = ResponseType::try_from(response_type_raw)?;

    // Decode headers (optional, last field in header)
    let headers_len = header.get_u16_le() as usize;
    let headers = if headers_len > 0 {
        // Validate headers length
        if header.remaining() < headers_len {
            return Err(DecodeError::InvalidHeadersLength);
        }

        let headers_bytes = header.copy_to_bytes(headers_len);
        let headers_map: HashMap<String, String> = rmp_serde::from_slice(&headers_bytes)?;
        Some(headers_map)
    } else {
        None
    };

    Ok(ActiveMessage {
        metadata: MessageMetadata {
            schema_version,
            response_type,
            response_id,
            handler_name,
            headers,
        },
        payload,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_handler_name_at_u16_max_succeeds() {
        // Create a handler name with exactly u16::MAX bytes (65,535 bytes)
        let handler_name = "a".repeat(u16::MAX as usize);
        let response_id = ResponseId::from_u128(12345);

        let message = ActiveMessage {
            metadata: MessageMetadata::new_unary(response_id, handler_name.clone(), None),
            payload: Bytes::from_static(b"test payload"),
        };

        // Should encode successfully
        let result = message.encode();
        assert!(
            result.is_ok(),
            "Handler name at u16::MAX should encode successfully"
        );

        let (header, payload, _) = result.unwrap();

        // Decode and verify
        let decoded = decode_active_message(header, payload).unwrap();
        assert_eq!(decoded.metadata.handler_name, handler_name);
    }

    #[test]
    fn test_handler_name_exceeds_u16_max_fails() {
        // Create a handler name with u16::MAX + 1 bytes (65,536 bytes)
        let handler_name = "a".repeat(u16::MAX as usize + 1);
        let response_id = ResponseId::from_u128(12345);

        let message = ActiveMessage {
            metadata: MessageMetadata::new_unary(response_id, handler_name.clone(), None),
            payload: Bytes::from_static(b"test payload"),
        };

        // Should fail to encode
        let result = message.encode();
        assert!(
            result.is_err(),
            "Handler name exceeding u16::MAX should fail to encode"
        );

        match result {
            Err(EncodeError::HandlerNameTooLong(len)) => {
                assert_eq!(len, u16::MAX as usize + 1);
            }
            _ => panic!("Expected HandlerNameTooLong error"),
        }
    }

    #[test]
    fn test_handler_name_way_too_long_fails() {
        // Create a handler name that's way too long (1 MB)
        let handler_name = "a".repeat(1024 * 1024);
        let response_id = ResponseId::from_u128(12345);

        let message = ActiveMessage {
            metadata: MessageMetadata::new_unary(response_id, handler_name, None),
            payload: Bytes::from_static(b"test payload"),
        };

        // Should fail to encode
        let result = message.encode();
        assert!(
            result.is_err(),
            "Very large handler name should fail to encode"
        );
    }

    #[test]
    fn test_normal_handler_name_succeeds() {
        // Test a normal-sized handler name
        let handler_name = "my_handler".to_string();
        let response_id = ResponseId::from_u128(12345);

        let message = ActiveMessage {
            metadata: MessageMetadata::new_unary(response_id, handler_name.clone(), None),
            payload: Bytes::from_static(b"test payload"),
        };

        // Should encode and decode successfully
        let (header, payload, _) = message.encode().unwrap();
        let decoded = decode_active_message(header, payload).unwrap();
        assert_eq!(decoded.metadata.handler_name, handler_name);
    }

    // ============================================================================
    // Headers Tests
    // ============================================================================

    #[test]
    fn test_headers_encode_decode_round_trip() {
        // Test encoding and decoding with headers
        let handler_name = "test_handler".to_string();
        let response_id = ResponseId::from_u128(12345);
        let mut headers = HashMap::new();
        headers.insert("trace-id".to_string(), "abc123".to_string());
        headers.insert("span-id".to_string(), "def456".to_string());

        let message = ActiveMessage {
            metadata: MessageMetadata::new_unary(
                response_id,
                handler_name.clone(),
                Some(headers.clone()),
            ),
            payload: Bytes::from_static(b"test payload"),
        };

        // Encode
        let (header, payload, _) = message.encode().unwrap();

        // Decode
        let decoded = decode_active_message(header, payload).unwrap();

        // Verify
        assert_eq!(decoded.metadata.handler_name, handler_name);
        assert_eq!(
            decoded.metadata.response_id.as_u128(),
            response_id.as_u128()
        );
        assert!(decoded.metadata.headers.is_some());
        let decoded_headers = decoded.metadata.headers.unwrap();
        assert_eq!(decoded_headers.len(), 2);
        assert_eq!(decoded_headers.get("trace-id").unwrap(), "abc123");
        assert_eq!(decoded_headers.get("span-id").unwrap(), "def456");
    }

    #[test]
    fn test_headers_none_encodes_with_zero_length() {
        // Test that None headers encodes with headers_len=0
        let handler_name = "test_handler".to_string();
        let response_id = ResponseId::from_u128(12345);

        let message = ActiveMessage {
            metadata: MessageMetadata::new_unary(response_id, handler_name.clone(), None),
            payload: Bytes::from_static(b"test payload"),
        };

        // Encode
        let (header, payload, _) = message.encode().unwrap();

        // The header should be: 1 + 1 + 16 + 2 + handler_name.len() + 2 (for headers_len=0)
        let expected_len = 1 + 1 + 16 + 2 + handler_name.len() + 2;
        assert_eq!(header.len(), expected_len);

        // Decode
        let decoded = decode_active_message(header, payload).unwrap();
        assert!(decoded.metadata.headers.is_none());
    }

    #[test]
    fn test_headers_empty_map_encodes_successfully() {
        // Test that empty HashMap encodes (but should be minimal)
        let handler_name = "test_handler".to_string();
        let response_id = ResponseId::from_u128(12345);
        let headers = HashMap::new();

        let message = ActiveMessage {
            metadata: MessageMetadata::new_unary(response_id, handler_name.clone(), Some(headers)),
            payload: Bytes::from_static(b"test payload"),
        };

        // Encode and decode
        let (header, payload, _) = message.encode().unwrap();
        let decoded = decode_active_message(header, payload).unwrap();

        assert!(decoded.metadata.headers.is_some());
        assert_eq!(decoded.metadata.headers.unwrap().len(), 0);
    }

    #[test]
    fn test_headers_per_value_size_limit() {
        // Test that header values exceeding 1KB are rejected
        let handler_name = "test_handler".to_string();
        let response_id = ResponseId::from_u128(12345);
        let mut headers = HashMap::new();

        // Create a value that's exactly 1KB (should succeed)
        let value_1kb = "a".repeat(1024);
        headers.insert("large-header".to_string(), value_1kb);

        let message = ActiveMessage {
            metadata: MessageMetadata::new_unary(response_id, handler_name.clone(), Some(headers)),
            payload: Bytes::from_static(b"test payload"),
        };

        // Should succeed at exactly 1KB
        let result = message.encode();
        assert!(result.is_ok(), "1KB value should encode successfully");
    }

    #[test]
    fn test_headers_per_value_size_exceeds_limit() {
        // Test that header values exceeding 1KB are rejected
        let handler_name = "test_handler".to_string();
        let response_id = ResponseId::from_u128(12345);
        let mut headers = HashMap::new();

        // Create a value that's 1KB + 1 byte (should fail)
        let value_too_large = "a".repeat(1025);
        headers.insert("large-header".to_string(), value_too_large);

        let message = ActiveMessage {
            metadata: MessageMetadata::new_unary(response_id, handler_name.clone(), Some(headers)),
            payload: Bytes::from_static(b"test payload"),
        };

        // Should fail
        let result = message.encode();
        assert!(result.is_err(), "1KB+1 value should fail to encode");

        match result {
            Err(EncodeError::HeaderValueTooLarge(key, size)) => {
                assert_eq!(key, "large-header");
                assert_eq!(size, 1025);
            }
            _ => panic!("Expected HeaderValueTooLarge error"),
        }
    }

    #[test]
    fn test_headers_total_size_limit() {
        // Test that total headers exceeding 16KB are rejected
        let handler_name = "test_handler".to_string();
        let response_id = ResponseId::from_u128(12345);
        let mut headers = HashMap::new();

        // Create many headers that together exceed 16KB when serialized
        // Each value is 500 bytes, which is under per-header limit
        // But we'll add enough to exceed 16KB total
        for i in 0..40 {
            let key = format!("header-{}", i);
            let value = "x".repeat(500);
            headers.insert(key, value);
        }

        let message = ActiveMessage {
            metadata: MessageMetadata::new_unary(response_id, handler_name.clone(), Some(headers)),
            payload: Bytes::from_static(b"test payload"),
        };

        // Should fail due to total size
        let result = message.encode();
        assert!(result.is_err(), "Total size exceeding 16KB should fail");

        match result {
            Err(EncodeError::TotalHeadersTooLarge(size)) => {
                assert!(size > 16384, "Size should exceed 16KB");
            }
            _ => panic!("Expected TotalHeadersTooLarge error"),
        }
    }

    #[test]
    fn test_headers_with_special_characters() {
        // Test headers with special characters, unicode, etc.
        let handler_name = "test_handler".to_string();
        let response_id = ResponseId::from_u128(12345);
        let mut headers = HashMap::new();
        headers.insert("emoji".to_string(), "🚀🎉".to_string());
        headers.insert("unicode".to_string(), "你好世界".to_string());
        headers.insert("special".to_string(), "a\nb\tc\"d'e".to_string());

        let message = ActiveMessage {
            metadata: MessageMetadata::new_unary(
                response_id,
                handler_name.clone(),
                Some(headers.clone()),
            ),
            payload: Bytes::from_static(b"test payload"),
        };

        // Encode and decode
        let (header, payload, _) = message.encode().unwrap();
        let decoded = decode_active_message(header, payload).unwrap();

        // Verify special characters preserved
        let decoded_headers = decoded.metadata.headers.unwrap();
        assert_eq!(decoded_headers.get("emoji").unwrap(), "🚀🎉");
        assert_eq!(decoded_headers.get("unicode").unwrap(), "你好世界");
        assert_eq!(decoded_headers.get("special").unwrap(), "a\nb\tc\"d'e");
    }

    #[test]
    fn test_headers_with_many_entries() {
        // Test with many header entries (but within size limits)
        let handler_name = "test_handler".to_string();
        let response_id = ResponseId::from_u128(12345);
        let mut headers = HashMap::new();

        // Add 100 small headers
        for i in 0..100 {
            headers.insert(format!("key-{}", i), format!("value-{}", i));
        }

        let message = ActiveMessage {
            metadata: MessageMetadata::new_unary(
                response_id,
                handler_name.clone(),
                Some(headers.clone()),
            ),
            payload: Bytes::from_static(b"test payload"),
        };

        // Encode and decode
        let (header, payload, _) = message.encode().unwrap();
        let decoded = decode_active_message(header, payload).unwrap();

        // Verify all headers present
        let decoded_headers = decoded.metadata.headers.unwrap();
        assert_eq!(decoded_headers.len(), 100);
        assert_eq!(decoded_headers.get("key-42").unwrap(), "value-42");
    }

    #[test]
    fn test_headers_all_response_types() {
        // Test headers work with all response types
        let handler_name = "test_handler".to_string();
        let response_id = ResponseId::from_u128(12345);
        let mut headers = HashMap::new();
        headers.insert("test".to_string(), "value".to_string());

        // FireAndForget
        let msg_fire = ActiveMessage {
            metadata: MessageMetadata::new_fire(
                response_id,
                handler_name.clone(),
                Some(headers.clone()),
            ),
            payload: Bytes::from_static(b"test"),
        };
        let (h, p, _) = msg_fire.encode().unwrap();
        let decoded = decode_active_message(h, p).unwrap();
        assert_eq!(decoded.metadata.response_type, ResponseType::FireAndForget);
        assert!(decoded.metadata.headers.is_some());

        // AckNack
        let msg_sync = ActiveMessage {
            metadata: MessageMetadata::new_sync(
                response_id,
                handler_name.clone(),
                Some(headers.clone()),
            ),
            payload: Bytes::from_static(b"test"),
        };
        let (h, p, _) = msg_sync.encode().unwrap();
        let decoded = decode_active_message(h, p).unwrap();
        assert_eq!(decoded.metadata.response_type, ResponseType::AckNack);
        assert!(decoded.metadata.headers.is_some());

        // Unary
        let msg_unary = ActiveMessage {
            metadata: MessageMetadata::new_unary(
                response_id,
                handler_name.clone(),
                Some(headers.clone()),
            ),
            payload: Bytes::from_static(b"test"),
        };
        let (h, p, _) = msg_unary.encode().unwrap();
        let decoded = decode_active_message(h, p).unwrap();
        assert_eq!(decoded.metadata.response_type, ResponseType::Unary);
        assert!(decoded.metadata.headers.is_some());
    }
}
