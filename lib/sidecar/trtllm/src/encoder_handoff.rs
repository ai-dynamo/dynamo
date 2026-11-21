// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Opaque native encoder handoffs carried through Dynamo requests and responses.

use base64::Engine as _;
use base64::engine::general_purpose::STANDARD as BASE64;
use dynamo_backend_common::DynamoError;

use crate::{client, proto as pb};

const MAX_ENCODER_HANDOFF_BYTES: usize = 64 * 1024;
const MAX_ENCODED_HANDOFF_BYTES: usize = MAX_ENCODER_HANDOFF_BYTES.div_ceil(3) * 4;

fn validate_encoder_handoff(handoff: &pb::EncodedMediaRef) -> Result<(), DynamoError> {
    if handoff.opaque_state.is_empty() || handoff.opaque_state.len() > MAX_ENCODER_HANDOFF_BYTES {
        return Err(client::invalid_argument(
            "empty or oversized encoder handoff reference",
        ));
    }
    Ok(())
}

pub(crate) fn from_result(result: &serde_json::Value) -> Result<pb::EncodedMediaRef, DynamoError> {
    let object = result
        .as_object()
        .filter(|fields| fields.len() == 1)
        .ok_or_else(|| client::invalid_argument("invalid encoder_result fields"))?;
    let state = object
        .get("opaque_state")
        .ok_or_else(|| client::invalid_argument("missing encoder_result opaque_state"))?;
    let opaque_state = match state {
        serde_json::Value::String(encoded) if encoded.len() <= MAX_ENCODED_HANDOFF_BYTES => BASE64
            .decode(encoded)
            .map_err(|_| client::invalid_argument("invalid encoder_result base64"))?,
        _ => {
            return Err(client::invalid_argument(
                "invalid or oversized encoder_result",
            ));
        }
    };
    let reference = pb::EncodedMediaRef { opaque_state };
    validate_encoder_handoff(&reference)?;
    Ok(reference)
}

pub(crate) fn to_result(
    handoff: pb::EncodedMediaRef,
) -> Result<serde_json::Map<String, serde_json::Value>, DynamoError> {
    validate_encoder_handoff(&handoff)?;
    let mut result = serde_json::Map::new();
    result.insert(
        "opaque_state".into(),
        serde_json::Value::String(BASE64.encode(&handoff.opaque_state)),
    );
    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn arbitrary_native_reference_round_trips_without_interpretation() {
        let native = pb::EncodedMediaRef {
            opaque_state: vec![0, 255, 19, 0],
        };
        let wire = to_result(native.clone()).unwrap();
        assert_eq!(
            from_result(&serde_json::Value::Object(wire)).unwrap(),
            native
        );
    }

    #[test]
    fn invalid_or_oversized_carriers_are_rejected() {
        for state in [
            serde_json::json!(""),
            serde_json::json!("!"),
            serde_json::json!([1, 2]),
            serde_json::json!(BASE64.encode(vec![0; MAX_ENCODER_HANDOFF_BYTES + 1])),
        ] {
            assert!(from_result(&serde_json::json!({"opaque_state":state})).is_err());
        }
    }
}
