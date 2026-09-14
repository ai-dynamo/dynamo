// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

use super::*;
use crate::{LLMEngineOutput, chunk};
use dynamo_runtime::protocols::annotated::Annotated;
use serde_json::json;

fn assert_same_request(bytes: &[u8]) {
    let normalize = |result: Result<PreprocessedRequest, rmp_serde::decode::Error>| {
        result
            .map(|request| serde_json::to_value(request).unwrap())
            .map_err(|error| error.to_string())
    };
    assert_eq!(
        normalize(decode(bytes)),
        normalize(rmp_serde::from_slice(bytes))
    );
}

fn request_with_array(array: &[u8]) -> Vec<u8> {
    let mut bytes = vec![0x82]; // A map with model and token_ids.
    bytes.extend(rmp_serde::to_vec("model").unwrap());
    bytes.extend(rmp_serde::to_vec(&"m".repeat(4096)).unwrap());
    bytes.extend(rmp_serde::to_vec("token_ids").unwrap());
    bytes.extend_from_slice(array);
    bytes
}

#[test]
fn integer_encodings_and_truncation_match_serde() {
    let mut values = vec![
        vec![0xc0],
        vec![0xc2],
        vec![0xc3],
        vec![0xff], // nil, booleans, -1
        vec![0xca, 0, 0, 0, 0],
        vec![0xa1, b'x'], // float, string
        vec![0xd0, 1],
        vec![0xd0, 0x80],
        vec![0xd1, 0, 1],
        vec![0xd1, 0x80, 0],
        vec![0xd2, 0, 0, 0, 1],
        vec![0xd2, 0x80, 0, 0, 0],
    ];
    for value in [
        0,
        1,
        127,
        128,
        255,
        256,
        16383,
        16384,
        65535,
        65536,
        u32::MAX as u64,
        u32::MAX as u64 + 1,
        u64::MAX,
    ] {
        values.push(rmp_serde::to_vec(&value).unwrap());
        for marker in [0xcf, 0xd3] {
            values.push([&[marker], value.to_be_bytes().as_slice()].concat());
        }
    }
    for value in values {
        let bytes = [&[0x91], value.as_slice()].concat();
        for end in 0..=bytes.len() {
            let encoded = &bytes[..end];
            let mut input = encoded;
            assert_eq!(
                array(&mut input),
                rmp_serde::from_slice::<Vec<u32>>(encoded).ok()
            );
            assert_same_request(&request_with_array(encoded));
        }
    }
    for count in [0, 1, 15, 16, 65535, 65536, 262145, 775168] {
        let ids: Vec<u32> = (0..count).map(|n| n % 150_000).collect();
        let encoded = rmp_serde::to_vec(&ids).unwrap();
        assert_eq!(array(&mut encoded.as_slice()), Some(ids));
        let request = request_with_array(&encoded);
        assert!(try_request(&request).is_some());
        assert_same_request(&request);
    }
    for encoded in [
        vec![0xdd, 255, 255, 255, 255],
        vec![0xdc, 0, 2, 1],
        vec![0x92, 1],
    ] {
        assert!(array(&mut encoded.as_slice()).is_none());
        assert_same_request(&request_with_array(&encoded));
    }
}

#[test]
fn request_fields_and_fallbacks_match_serde() {
    let value = json!({
        "model": "m".repeat(4096),
        "token_ids": [1, 128, 65536, u32::MAX],
        "eos_token_ids": [42],
        "stop_conditions": {"max_tokens": 57496, "ignore_eos": true},
        "sampling_options": {"temperature": 0.75},
        "annotations": ["test"],
        "encoder_result": {"nested": [1, {"value": "kept"}]},
        "unknown_before_tokens": {"nested": [true, null, [2, 3]]},
    });
    let named = rmp_serde::to_vec_named(&value).unwrap();
    assert!(try_request(&named).is_some());
    assert_same_request(&named);
    // The same map is valid with fixmap, map16, and map32 headers.
    let fields = named[0] & 0x0f;
    assert_eq!(named[0] & 0xf0, 0x80);
    for header in [vec![0xde, 0, fields], vec![0xdf, 0, 0, 0, fields]] {
        let bytes = [header.as_slice(), &named[1..]].concat();
        assert!(try_request(&bytes).is_some());
        assert_same_request(&bytes);
    }

    let request: PreprocessedRequest = rmp_serde::from_slice(&named).unwrap();
    let positional = rmp_serde::to_vec(&request).unwrap();
    assert!(try_request(&positional).is_none());
    assert_same_request(&positional);

    let small = rmp_serde::to_vec_named(&json!({"token_ids": [1, 2]})).unwrap();
    assert!(try_request(&small).is_none());
    assert_same_request(&small);

    // Serde also supports bytes for a sequence; keep that fallback.
    assert_same_request(&request_with_array(&[0xc4, 2, 1, 2]));
    let mut duplicate = request_with_array(&[0x91, 1]);
    duplicate[0] = 0x83;
    duplicate.extend_from_slice(b"\xa9token_ids\x90");
    assert!(try_request(&duplicate).is_none());
    assert_same_request(&duplicate);

    let mut invalid_metadata = request_with_array(&[0x91, 1]);
    invalid_metadata[0] = 0x83;
    invalid_metadata.extend(rmp_serde::to_vec("stop_conditions").unwrap());
    invalid_metadata.push(0xc2);
    assert!(try_request(&invalid_metadata).is_none());
    assert_same_request(&invalid_metadata);

    for bad in [
        json!({"model": "m".repeat(4096)}),
        json!({"model": "m".repeat(4096), "token_ids": [1], "stop_conditions": false}),
        json!({"model": "m".repeat(4096), "token_ids": [1], "encoder_result": []}),
    ] {
        assert_same_request(&rmp_serde::to_vec_named(&bad).unwrap());
    }
    // Test incomplete metadata before and after the token field.
    for end in [0, 1, 2, 4096, named.len() - 1] {
        assert_same_request(&named[..end]);
    }
}

#[tokio::test]
async fn adapter_preserves_codecs_errors_and_response_frames() {
    let request =
        rmp_serde::from_slice::<PreprocessedRequest>(&request_with_array(&[0x91, 42])).unwrap();
    for codec in [
        RequestPlanePayloadCodec::Msgpack,
        RequestPlanePayloadCodec::Json,
    ] {
        let bytes = bytes::Bytes::from(codec.encode(&request).unwrap());
        let actual = TokenPayloadAdapter
            .decode_request(codec, bytes.clone())
            .await
            .unwrap();
        let expected: PreprocessedRequest = SerdeIngressPayloadAdapter
            .decode_request(codec, bytes)
            .await
            .unwrap();
        assert_eq!(
            serde_json::to_value(actual).unwrap(),
            serde_json::to_value(expected).unwrap()
        );

        let invalid = bytes::Bytes::from_static(b"invalid");
        let expected: Result<PreprocessedRequest, _> = SerdeIngressPayloadAdapter
            .decode_request(codec, invalid.clone())
            .await;
        let actual = TokenPayloadAdapter.decode_request(codec, invalid).await;
        assert_eq!(
            actual.unwrap_err().to_string(),
            expected.unwrap_err().to_string()
        );

        let responses = [
            Some(Annotated::from_data(chunk::token(42))),
            Some(Annotated::from_data(LLMEngineOutput::length())),
            Some(Annotated::<LLMEngineOutput>::from_error("test error")),
            None,
        ];
        for response in responses {
            for complete in [false, true] {
                let actual = TokenPayloadAdapter
                    .encode_response(codec, response.clone(), complete)
                    .await
                    .unwrap();
                let expected = SerdeIngressPayloadAdapter
                    .encode_response(codec, response.clone(), complete)
                    .await
                    .unwrap();
                assert_eq!(actual.bytes, expected.bytes);
                assert_eq!(actual.is_error, expected.is_error);
                assert_eq!(actual.stop_stream, expected.stop_stream);
            }
        }
    }
}
