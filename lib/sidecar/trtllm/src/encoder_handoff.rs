// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Opaque native encoder handoffs carried through Dynamo requests and responses.

use base64::Engine as _;
use base64::engine::general_purpose::STANDARD as BASE64;
use dynamo_backend_common::DynamoError;

use crate::{client, proto as pb};

pub(crate) const NIXL_ENCODER_BACKEND: &str = "trtllm-nixl-encoder-v1";

const MAX_ENCODER_HANDOFF_BYTES: usize = 64 * 1024;
const MAX_ENCODED_HANDOFF_BYTES: usize = MAX_ENCODER_HANDOFF_BYTES.div_ceil(3) * 4;

fn validate_encoder_handoff(handoff: &pb::EncoderHandoffRef) -> Result<(), DynamoError> {
    if handoff.transfer_backend != NIXL_ENCODER_BACKEND
        || handoff.opaque_state.is_empty()
        || handoff.opaque_state.len() > MAX_ENCODER_HANDOFF_BYTES
    {
        return Err(client::invalid_argument(
            "unsupported or oversized encoder handoff reference",
        ));
    }
    Ok(())
}

pub(crate) fn from_result(
    result: &serde_json::Value,
) -> Result<pb::EncoderHandoffRef, DynamoError> {
    let object = result
        .as_object()
        .filter(|fields| fields.len() == 2)
        .ok_or_else(|| client::invalid_argument("invalid encoder_result fields"))?;
    let backend = object
        .get("transfer_backend")
        .and_then(serde_json::Value::as_str)
        .filter(|backend| *backend == NIXL_ENCODER_BACKEND)
        .ok_or_else(|| client::invalid_argument("unsupported encoder_result transfer_backend"))?;
    let state = object
        .get("opaque_state")
        .ok_or_else(|| client::invalid_argument("missing encoder_result opaque_state"))?;
    let opaque_state = match state {
        serde_json::Value::String(encoded) if encoded.len() <= MAX_ENCODED_HANDOFF_BYTES => BASE64
            .decode(encoded)
            .map_err(|_| client::invalid_argument("invalid encoder_result base64"))?,
        // Pre-release EPD compatibility; remove after all supported workers emit base64.
        serde_json::Value::Array(bytes) if bytes.len() <= MAX_ENCODER_HANDOFF_BYTES => {
            serde::Deserialize::deserialize(state)
                .map_err(|_| client::invalid_argument("invalid encoder_result bytes"))?
        }
        _ => {
            return Err(client::invalid_argument(
                "invalid or oversized encoder_result",
            ));
        }
    };
    let reference = pb::EncoderHandoffRef {
        transfer_backend: backend.to_owned(),
        opaque_state,
    };
    validate_encoder_handoff(&reference)?;
    Ok(reference)
}

pub(crate) fn to_result(
    handoff: pb::EncoderHandoffRef,
) -> Result<serde_json::Map<String, serde_json::Value>, DynamoError> {
    validate_encoder_handoff(&handoff)?;
    let mut result = serde_json::Map::new();
    result.insert(
        "transfer_backend".into(),
        serde_json::json!(handoff.transfer_backend),
    );
    result.insert(
        "opaque_state".into(),
        serde_json::Value::String(BASE64.encode(&handoff.opaque_state)),
    );
    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn base64_is_bounded_and_accepts_legacy_arrays() {
        let state = vec![255_u8; MAX_ENCODER_HANDOFF_BYTES];
        let encoded = BASE64.encode(&state);
        assert_eq!(encoded.len(), MAX_ENCODED_HANDOFF_BYTES);
        let result = json!({"transfer_backend": NIXL_ENCODER_BACKEND, "opaque_state": encoded});
        assert_eq!(from_result(&result).unwrap().opaque_state, state);
        let written = to_result(pb::EncoderHandoffRef {
            transfer_backend: NIXL_ENCODER_BACKEND.into(),
            opaque_state: state,
        })
        .unwrap();
        assert_eq!(serde_json::Value::Object(written), result);
        let legacy = json!({"transfer_backend": NIXL_ENCODER_BACKEND, "opaque_state": [0, 1, 255]});
        assert_eq!(from_result(&legacy).unwrap().opaque_state, [0, 1, 255]);
        for invalid in [
            json!(""),
            json!("not base64!?"),
            json!("A".repeat(MAX_ENCODED_HANDOFF_BYTES + 1)),
            json!(BASE64.encode(vec![0_u8; MAX_ENCODER_HANDOFF_BYTES + 1])),
            json!([256]),
            json!([-1]),
            json!([1.5]),
        ] {
            let result = json!({"transfer_backend": NIXL_ENCODER_BACKEND, "opaque_state": invalid});
            assert!(
                from_result(&result).is_err(),
                "accepted malformed or oversized handoff"
            );
        }
    }
}
