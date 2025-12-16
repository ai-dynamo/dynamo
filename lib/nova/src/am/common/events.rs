// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use bytes::{Buf, BufMut, Bytes, BytesMut};
use std::mem::size_of;

use super::responses::ResponseId;

use crate::events::EventHandle;

pub(crate) enum Outcome {
    Ok,
    Error,
}

pub(crate) enum EventType {
    Ack(ResponseId, Outcome),
    Event(EventHandle, Outcome),
}

#[inline]
pub(crate) fn encode_event_header(event_type: EventType) -> Bytes {
    // Encode using two bits (always include the u128 identifier):
    // - Bit 0 (LSB): 0 = Ack, 1 = Event
    // - Bit 1: 0 = Ok, 1 = Error
    //
    // Errors carry the ID so downstream can route the failure to the correct waiter.
    match event_type {
        EventType::Ack(response_id, outcome) => {
            let id = response_id.as_u128().to_le_bytes();
            let type_byte = match outcome {
                Outcome::Error => 0b10, // Ack + Error (bit 0=0 for Ack, bit 1=1 for Error)
                Outcome::Ok => 0b00,    // Ack + Ok (bit 0=0 for Ack, bit 1=0 for Ok)
            };
            let mut bytes = BytesMut::with_capacity(1 + size_of::<u128>());
            bytes.put_u8(type_byte);
            bytes.extend_from_slice(&id);
            bytes.freeze()
        }
        EventType::Event(event_handle, outcome) => {
            let id = event_handle.as_u128().to_le_bytes();
            let type_byte = match outcome {
                Outcome::Error => 0b11, // Event + Error (bit 0=1 for Event, bit 1=1 for Error)
                Outcome::Ok => 0b01,    // Event + Ok (bit 0=1 for Event, bit 1=0 for Ok)
            };
            let mut bytes = BytesMut::with_capacity(1 + size_of::<u128>());
            bytes.put_u8(type_byte);
            bytes.extend_from_slice(&id);
            bytes.freeze()
        }
    }
}

pub(crate) fn decode_event_header(header: Bytes) -> Option<EventType> {
    if header.is_empty() {
        return None;
    }

    let mut header = header;
    let type_byte = header.get_u8();

    // Decode bits:
    // - Bit 0 (LSB): 0 = Ack, 1 = Event
    // - Bit 1: 0 = Ok, 1 = Error
    let is_event = (type_byte & 0b01) != 0;
    let is_error = (type_byte & 0b10) != 0;

    // All variants carry an ID; header must include u128.
    if header.len() < size_of::<u128>() {
        return None;
    }

    let mut bytes_array = [0u8; 16];
    header.copy_to_slice(&mut bytes_array);
    let value = u128::from_le_bytes(bytes_array);

    match (is_event, is_error) {
        (true, true) => Some(EventType::Event(
            EventHandle::from_raw(value),
            Outcome::Error,
        )),
        (true, false) => Some(EventType::Event(EventHandle::from_raw(value), Outcome::Ok)),
        (false, true) => Some(EventType::Ack(ResponseId::from_u128(value), Outcome::Error)),
        (false, false) => Some(EventType::Ack(ResponseId::from_u128(value), Outcome::Ok)),
    }
}

#[cfg(test)]
mod tests {
    use uuid::Uuid;

    use super::*;
    use crate::events::EventHandle;

    #[test]
    fn encode_decode_ack_ok_round_trip() {
        // Create a ResponseId from a UUID
        let uuid = Uuid::new_v4();
        let response_id = ResponseId::from_u128(uuid.as_u128());

        // Encode Ack + Ok
        let encoded = encode_event_header(EventType::Ack(response_id, Outcome::Ok));

        // Verify length (1 byte type + 16 bytes u128)
        assert_eq!(encoded.len(), 1 + size_of::<u128>());
        assert_eq!(encoded[0], 0b00); // Ack + Ok

        // Decode
        let decoded = decode_event_header(encoded).expect("should decode successfully");

        // Verify round-trip
        match decoded {
            EventType::Ack(decoded_id, outcome) => {
                assert!(matches!(outcome, Outcome::Ok));
                assert_eq!(decoded_id.as_u128(), response_id.as_u128());
            }
            _ => panic!("Expected Ack variant"),
        }
    }

    #[test]
    fn encode_decode_ack_error_round_trip() {
        // Create a ResponseId (value doesn't matter for error case)
        let response_id = ResponseId::from_u128(Uuid::new_v4().as_u128());

        // Encode Ack + Error
        let encoded = encode_event_header(EventType::Ack(response_id, Outcome::Error));

        // Verify length (1 byte type + 16 bytes u128)
        assert_eq!(encoded.len(), 1 + size_of::<u128>());
        assert_eq!(encoded[0], 0b10); // Ack + Error

        // Decode
        let decoded = decode_event_header(encoded).expect("should decode successfully");

        // Verify round-trip
        match decoded {
            EventType::Ack(decoded_id, outcome) => {
                assert!(matches!(outcome, Outcome::Error));
                assert_eq!(decoded_id.as_u128(), response_id.as_u128());
            }
            _ => panic!("Expected Ack variant"),
        }
    }

    #[test]
    fn encode_decode_event_ok_round_trip() {
        // Create an EventHandle
        let event_handle = EventHandle::new(42, 100, 5).expect("should create EventHandle");

        // Encode Event + Ok
        let encoded = encode_event_header(EventType::Event(event_handle, Outcome::Ok));

        // Verify length (1 byte type + 16 bytes u128)
        assert_eq!(encoded.len(), 1 + size_of::<u128>());
        assert_eq!(encoded[0], 0b01); // Event + Ok

        // Decode
        let decoded = decode_event_header(encoded).expect("should decode successfully");

        // Verify round-trip
        match decoded {
            EventType::Event(decoded_handle, outcome) => {
                assert!(matches!(outcome, Outcome::Ok));
                assert_eq!(decoded_handle.as_u128(), event_handle.as_u128());
                assert_eq!(decoded_handle.owner_worker().as_u64(), 42);
                assert_eq!(decoded_handle.local_index(), 100);
                assert_eq!(decoded_handle.generation(), 5);
            }
            _ => panic!("Expected Event variant"),
        }
    }

    #[test]
    fn encode_decode_event_error_round_trip() {
        // Create an EventHandle (value doesn't matter for error case)
        let event_handle = EventHandle::new(42, 100, 5).expect("should create EventHandle");

        // Encode Event + Error
        let encoded = encode_event_header(EventType::Event(event_handle, Outcome::Error));

        // Verify length (1 byte type + 16 bytes u128)
        assert_eq!(encoded.len(), 1 + size_of::<u128>());
        assert_eq!(encoded[0], 0b11); // Event + Error

        // Decode
        let decoded = decode_event_header(encoded).expect("should decode successfully");

        // Verify round-trip
        match decoded {
            EventType::Event(decoded_handle, outcome) => {
                assert!(matches!(outcome, Outcome::Error));
                assert_eq!(decoded_handle.as_u128(), event_handle.as_u128());
            }
            _ => panic!("Expected Event variant"),
        }
    }

    #[test]
    fn decode_invalid_length() {
        // Missing u128 bytes should fail for all variants
        let short_bytes = Bytes::from_static(&[0b00]); // Ack + Ok but no u128
        assert!(decode_event_header(short_bytes).is_none());

        let short_error = Bytes::from_static(&[0b11]); // Event + Error but no u128
        assert!(decode_event_header(short_error).is_none());

        let empty_bytes = Bytes::new();
        assert!(decode_event_header(empty_bytes).is_none());
    }

    #[test]
    fn decode_all_valid_type_bytes() {
        // Test all 4 valid combinations (0b00, 0b01, 0b10, 0b11)
        // 0b00: Ack + Ok
        let mut bytes = BytesMut::with_capacity(1 + size_of::<u128>());
        bytes.put_u8(0b00);
        bytes.extend_from_slice(&[0u8; 16]);
        let decoded = decode_event_header(bytes.freeze()).expect("0b00 should decode");
        assert!(matches!(decoded, EventType::Ack(_, Outcome::Ok)));

        // 0b01: Event + Ok
        bytes = BytesMut::with_capacity(1 + size_of::<u128>());
        bytes.put_u8(0b01);
        bytes.extend_from_slice(&[0u8; 16]);
        let decoded = decode_event_header(bytes.freeze()).expect("0b01 should decode");
        assert!(matches!(decoded, EventType::Event(_, Outcome::Ok)));

        // 0b10: Ack + Error
        bytes = BytesMut::with_capacity(1 + size_of::<u128>());
        bytes.put_u8(0b10);
        bytes.extend_from_slice(&[0u8; 16]);
        let decoded = decode_event_header(bytes.freeze()).expect("0b10 should decode");
        assert!(matches!(decoded, EventType::Ack(_, Outcome::Error)));

        // 0b11: Event + Error
        bytes = BytesMut::with_capacity(1 + size_of::<u128>());
        bytes.put_u8(0b11);
        bytes.extend_from_slice(&[0u8; 16]);
        let decoded = decode_event_header(bytes.freeze()).expect("0b11 should decode");
        assert!(matches!(decoded, EventType::Event(_, Outcome::Error)));
    }

    #[test]
    fn encode_decode_multiple_ack_ok_values() {
        // Test with different UUID values for Ack + Ok
        let test_uuids = vec![
            Uuid::nil(),
            Uuid::new_v4(),
            Uuid::from_u128(0x1234_5678_9ABC_DEF0_1234_5678_9ABC_DEF0),
            Uuid::from_u128(u128::MAX),
        ];

        for uuid in test_uuids {
            let response_id = ResponseId::from_u128(uuid.as_u128());
            let encoded = encode_event_header(EventType::Ack(response_id, Outcome::Ok));
            let decoded = decode_event_header(encoded).expect("should decode");

            match decoded {
                EventType::Ack(decoded_id, outcome) => {
                    assert!(matches!(outcome, Outcome::Ok));
                    assert_eq!(decoded_id.as_u128(), uuid.as_u128());
                }
                _ => panic!("Expected Ack variant"),
            }
        }
    }

    #[test]
    fn encode_decode_multiple_event_ok_values() {
        // Test with different EventHandle values for Event + Ok
        let test_handles = vec![
            EventHandle::new(0, 0, 0).expect("should create"),
            EventHandle::new(42, 100, 5).expect("should create"),
            EventHandle::new(u64::MAX, u32::MAX, u32::MAX).expect("should create"),
            EventHandle::new(12345, 67890, 999).expect("should create"),
        ];

        for handle in test_handles {
            let encoded = encode_event_header(EventType::Event(handle, Outcome::Ok));
            let decoded = decode_event_header(encoded).expect("should decode");

            match decoded {
                EventType::Event(decoded_handle, outcome) => {
                    assert!(matches!(outcome, Outcome::Ok));
                    assert_eq!(decoded_handle.as_u128(), handle.as_u128());
                    assert_eq!(decoded_handle.owner_worker(), handle.owner_worker());
                    assert_eq!(decoded_handle.local_index(), handle.local_index());
                    assert_eq!(decoded_handle.generation(), handle.generation());
                }
                _ => panic!("Expected Event variant"),
            }
        }
    }

    #[test]
    fn encode_ack_ok_has_correct_format() {
        let uuid = Uuid::new_v4();
        let response_id = ResponseId::from_u128(uuid.as_u128());
        let encoded = encode_event_header(EventType::Ack(response_id, Outcome::Ok));

        // First byte should be 0b00 (Ack + Ok)
        assert_eq!(encoded[0], 0b00);

        // Remaining bytes should be the u128 in little-endian
        let mut bytes_array = [0u8; 16];
        bytes_array.copy_from_slice(&encoded[1..]);
        let decoded_value = u128::from_le_bytes(bytes_array);
        assert_eq!(decoded_value, uuid.as_u128());
    }

    #[test]
    fn encode_event_ok_has_correct_format() {
        let handle = EventHandle::new(42, 100, 5).expect("should create");
        let encoded = encode_event_header(EventType::Event(handle, Outcome::Ok));

        // First byte should be 0b01 (Event + Ok)
        assert_eq!(encoded[0], 0b01);

        // Remaining bytes should be the u128 in little-endian
        let mut bytes_array = [0u8; 16];
        bytes_array.copy_from_slice(&encoded[1..]);
        let decoded_value = u128::from_le_bytes(bytes_array);
        assert_eq!(decoded_value, handle.as_u128());
    }

    #[test]
    fn encode_error_has_correct_format() {
        let response_id = ResponseId::from_u128(Uuid::new_v4().as_u128());
        let ack_error = encode_event_header(EventType::Ack(response_id, Outcome::Error));

        // Should be 1 byte type + 16-byte id with value 0b10 in first byte
        assert_eq!(ack_error.len(), 1 + size_of::<u128>());
        assert_eq!(ack_error[0], 0b10);

        let handle = EventHandle::new(42, 100, 5).expect("should create");
        let event_error = encode_event_header(EventType::Event(handle, Outcome::Error));

        // Should be 1 byte type + 16-byte id with value 0b11 in first byte
        assert_eq!(event_error.len(), 1 + size_of::<u128>());
        assert_eq!(event_error[0], 0b11);
    }
}
