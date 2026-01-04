// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Request decoding for push handler.
//!
//! Extracts control messages and typed request bodies from incoming payloads.

use bytes::Bytes;
use serde::Deserialize;

use crate::pipeline::network::codec::{TwoPartCodec, TwoPartMessageType};
use crate::pipeline::network::RequestControlMessage;
use crate::pipeline::PipelineError;

/// A decoded request containing the control message and typed payload.
#[derive(Debug)]
pub struct DecodedRequest<T> {
    pub control_msg: RequestControlMessage,
    pub request: T,
}

/// Decode a raw payload into a control message and typed request.
///
/// # Errors
///
/// Returns `PipelineError::DeserializationError` if:
/// - The payload is not a valid two-part message with header and data
/// - The header cannot be deserialized as `RequestControlMessage`
/// - The data cannot be deserialized as type `T`
pub fn decode_payload<T>(payload: Bytes) -> Result<DecodedRequest<T>, PipelineError>
where
    T: for<'de> Deserialize<'de>,
{
    let msg = TwoPartCodec::default()
        .decode_message(payload)?
        .into_message_type();

    match msg {
        TwoPartMessageType::HeaderAndData(header, data) => {
            tracing::trace!(
                "received two part message with ctrl: {} bytes, data: {} bytes",
                header.len(),
                data.len()
            );

            let control_msg: RequestControlMessage =
                serde_json::from_slice(&header).map_err(|err| {
                    let json_str = String::from_utf8_lossy(&header);
                    PipelineError::DeserializationError(format!(
                        "Failed deserializing to RequestControlMessage. err={err}, json_str={json_str}"
                    ))
                })?;

            let request: T = serde_json::from_slice(&data)?;

            Ok(DecodedRequest {
                control_msg,
                request,
            })
        }
        _ => Err(PipelineError::Generic(String::from(
            "Unexpected message from work queue; unable extract a TwoPartMessage with a header and data",
        ))),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pipeline::network::codec::TwoPartCodec;
    use crate::pipeline::network::{ConnectionInfo, RequestType, ResponseType};

    #[derive(Debug, Deserialize, PartialEq)]
    struct TestRequest {
        prompt: String,
    }

    /// Helper to build a valid two-part encoded message
    fn build_two_part_message(header: &str, data: &str) -> Bytes {
        use crate::pipeline::network::codec::TwoPartMessage;
        let message = TwoPartMessage::from_parts(
            Bytes::from(header.to_string()),
            Bytes::from(data.to_string()),
        );
        TwoPartCodec::default().encode_message(message).unwrap()
    }

    fn sample_control_message_json() -> String {
        serde_json::to_string(&RequestControlMessage {
            id: "test-123".to_string(),
            request_type: RequestType::SingleIn,
            response_type: ResponseType::ManyOut,
            connection_info: ConnectionInfo {
                transport: "tcp".to_string(),
                info: r#"{"addr":"127.0.0.1:8080"}"#.to_string(),
            },
        })
        .unwrap()
    }

    #[test]
    fn test_decode_valid_payload() {
        let control = sample_control_message_json();
        let data = r#"{"prompt":"hello world"}"#;
        let payload = build_two_part_message(&control, data);

        let result: DecodedRequest<TestRequest> = decode_payload(payload).unwrap();

        assert_eq!(result.control_msg.id, "test-123");
        assert_eq!(result.request.prompt, "hello world");
    }

    #[test]
    fn test_decode_invalid_header_returns_deserialization_error() {
        let payload = build_two_part_message("not valid json", r#"{"prompt":"hi"}"#);

        let result: Result<DecodedRequest<TestRequest>, _> = decode_payload(payload);

        assert!(matches!(
            result,
            Err(PipelineError::DeserializationError(_))
        ));
    }

    #[test]
    fn test_decode_invalid_body_returns_error() {
        let control = sample_control_message_json();
        let payload = build_two_part_message(&control, "not valid json");

        let result: Result<DecodedRequest<TestRequest>, _> = decode_payload(payload);

        // assert on the error from `serde_json::from_slice(&data)?`
        assert!(matches!(
            result,
            Err(PipelineError::SerdeJsonError(_))
        ));
    }

    #[test]
    fn test_decode_mismatched_body_type_returns_error() {
        let control = sample_control_message_json();
        // Valid JSON but wrong shape for TestRequest
        let payload = build_two_part_message(&control, r#"{"wrong_field": 123}"#);

        let result: Result<DecodedRequest<TestRequest>, _> = decode_payload(payload);
    
        // assert on the error from `serde_json::from_slice(&data)?`
        assert!(matches!(
            result,
            Err(PipelineError::SerdeJsonError(_))
        ));
    }
}

