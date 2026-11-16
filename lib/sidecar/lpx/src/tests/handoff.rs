// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! The vLLM handoff codec against the golden vectors the engine tests too.

use dynamo_sidecar_common::struct_to_json;
use prost_types::value::Kind;

use super::*;
use crate::handoff::decode_session;
use crate::pb;

fn encode(disaggregated_params: Value) -> Result<pb::KvSessionRef, DynamoError> {
    decode_session(&prefill_result(disaggregated_params, None))
}

/// The attributes as the decode engine decodes them back to JSON.
fn attributes(session: &pb::KvSessionRef) -> Value {
    struct_to_json(
        session.attributes_struct.clone().expect("attributes"),
        "test",
        "attributes",
    )
    .expect("decodes")
}

fn wire(session: &pb::KvSessionRef, key: &str) -> Kind {
    session
        .attributes_struct
        .as_ref()
        .expect("attributes")
        .fields
        .get(key)
        .and_then(|value| value.kind.clone())
        .unwrap_or_else(|| panic!("no {key} on the wire"))
}

fn assert_rejected(disaggregated_params: Value, expect: &str) {
    let error = encode(disaggregated_params).expect_err("the handoff must be rejected");
    assert_backend(&error, BackendError::InvalidArgument);
    assert!(
        error.to_string().contains(expect),
        "error {error} should mention {expect:?}"
    );
}

/// G1: the Python vLLM worker's wrapped handoff becomes a NIXL session named
/// by vLLM's request ID. The attributes decode back to exactly vLLM's object;
/// JSON number equality also tells integers from floats, and every integer
/// that fits a double must arrive as an integer, or the engine's typed parse
/// (`remote_port: i32`, `remote_block_ids: Vec<Vec<i32>>`) fails. Decoding
/// cannot tell an unset value from null, so null is checked on the wire.
#[test]
fn g1_the_wrapped_shape_is_unwrapped_verbatim() {
    let session = encode(wrapped(vllm_params())).expect("G1 is accepted");
    assert_eq!(session.session_id, "request");
    assert_eq!(session.transfer_backend, "nixl");
    assert!(session.endpoints.is_empty());
    assert_eq!(session.dp_rank, 0);
    assert_eq!(attributes(&session), vllm_params());
    assert!(matches!(
        wire(&session, "remote_blocks_expiry_time"),
        Kind::NullValue(_)
    ));
}

/// `kv.proto`: an integer a double cannot hold exactly (beyond +/-2^53)
/// travels as a decimal string instead of failing the request, at any depth;
/// floats and exact integers keep their value.
#[test]
fn integers_beyond_two_to_the_53_become_strings() {
    for (key, sent, received) in [
        // G3: a fractional expiry stays a float.
        (
            "remote_blocks_expiry_time",
            json!(1_760_000_000.25),
            json!(1_760_000_000.25),
        ),
        // G4: 2^53 is still exact; one more is not.
        (
            "big",
            json!(9_007_199_254_740_992_u64),
            json!(9_007_199_254_740_992_u64),
        ),
        (
            "big",
            json!(9_007_199_254_740_993_u64),
            json!("9007199254740993"),
        ),
        (
            "remote_block_ids",
            json!([[3, 9_007_199_254_740_993_u64]]),
            json!([[3, "9007199254740993"]]),
        ),
        (
            "nested",
            json!({"low": -9_007_199_254_740_993_i64, "ok": -9_007_199_254_740_992_i64}),
            json!({"low": "-9007199254740993", "ok": -9_007_199_254_740_992_i64}),
        ),
    ] {
        let mut params = vllm_params();
        params[key] = sent;
        let attributes = attributes(&encode(wrapped(params)).expect("accepted"));
        assert_eq!(attributes[key], received, "{key}");
    }
}

/// G7: anything without a vLLM request ID is not a vLLM NIXL handoff.
#[test]
fn g7_other_handoffs_are_rejected() {
    for params in [
        json!({"session_id": "12345", "transfer_backend": "NIXL"}),
        json!({"remote_request_id": ""}),
        json!({"remote_request_id": 7}),
    ] {
        assert_rejected(params, "is not a vLLM NIXL handoff");
    }
    assert_rejected(json!("handoff"), "must be a JSON object");
}
