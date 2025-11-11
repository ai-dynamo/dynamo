// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! # Dynamo Active Message Common

use bytes::{Buf, BufMut, Bytes, BytesMut};
use thiserror::Error;

use super::responses::ResponseId;

const CURRENT_SCHEMA_VERSION: u8 = 1;

#[derive(Debug, Clone)]
pub(crate) struct ActiveMessage {
    pub metadata: MessageMetadata,
    pub payload: Bytes,
}

impl ActiveMessage {
    pub(crate) fn encode(self) -> (Bytes, Bytes, dynamo_nova_backend::MessageType) {
        encode_active_message(self)
    }
}

#[derive(Debug, Clone)]
pub(crate) struct MessageMetadata {
    pub schema_version: u8,
    pub response_type: ResponseType,
    pub response_id: ResponseId,
    pub handler_name: String,
}

impl MessageMetadata {
    pub(crate) fn new_fire(response_id: ResponseId, handler_name: String) -> Self {
        Self {
            schema_version: CURRENT_SCHEMA_VERSION,
            response_type: ResponseType::FireAndForget,
            response_id,
            handler_name,
        }
    }

    pub(crate) fn new_sync(response_id: ResponseId, handler_name: String) -> Self {
        Self {
            schema_version: CURRENT_SCHEMA_VERSION,
            response_type: ResponseType::AckNack,
            response_id,
            handler_name,
        }
    }

    pub(crate) fn new_unary(response_id: ResponseId, handler_name: String) -> Self {
        Self {
            schema_version: CURRENT_SCHEMA_VERSION,
            response_type: ResponseType::Unary,
            response_id,
            handler_name,
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
}

pub(crate) fn encode_active_message(
    message: ActiveMessage,
) -> (Bytes, Bytes, dynamo_nova_backend::MessageType) {
    let header_size = 20 + message.metadata.handler_name.len();
    let mut header = BytesMut::with_capacity(header_size);

    header.put_u8(message.metadata.schema_version);
    header.put_u8(message.metadata.response_type as u8);
    header.put_u128_le(message.metadata.response_id.as_u128());
    header.put_u16_le(message.metadata.handler_name.len() as u16);
    header.put_slice(message.metadata.handler_name.as_bytes());

    let message_type = message.metadata.response_type.to_message_type();
    (header.freeze(), message.payload, message_type)
}

pub(crate) fn decode_active_message(
    header: Bytes,
    payload: Bytes,
) -> Result<ActiveMessage, DecodeError> {
    let mut header = header;

    // Validate minimum size
    if header.len() < 20 {
        return Err(DecodeError::HeaderTooShort);
    }

    let schema_version = header.get_u8();
    let response_type_raw = header.get_u8();
    let response_id = ResponseId::from_u128(header.get_u128_le());
    let handler_name_len = header.get_u16_le() as usize;

    // Validate handler name length
    if header.remaining() < handler_name_len {
        return Err(DecodeError::InvalidHandlerNameLength);
    }

    let handler_name_bytes = header.copy_to_bytes(handler_name_len);
    let handler_name =
        String::from_utf8(handler_name_bytes.to_vec()).map_err(|_| DecodeError::InvalidUtf8)?;

    let response_type = ResponseType::try_from(response_type_raw)?;

    Ok(ActiveMessage {
        metadata: MessageMetadata {
            schema_version,
            response_type,
            response_id,
            handler_name,
        },
        payload,
    })
}
