// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use prost::Message;

#[test]
fn generate_request_keeps_its_released_wire_tags() {
    let request = dynamo_sglang_sidecar::proto::GenerateRequest {
        input_ids: vec![1, 2],
        rid: Some("rid".to_string()),
        ..Default::default()
    };
    // input_ids is packed field 1; rid is optional field 7.
    assert_eq!(
        request.encode_to_vec(),
        [0x0a, 0x02, 0x01, 0x02, 0x3a, 0x03, b'r', b'i', b'd']
    );
}

#[test]
fn release_0522_fields_keep_their_wire_tags() {
    let request = dynamo_sglang_sidecar::proto::GenerateRequest {
        return_text: Some(true),
        ..Default::default()
    };
    // return_text is optional field 18.
    assert_eq!(request.encode_to_vec(), [0x90, 0x01, 0x01]);

    let response = dynamo_sglang_sidecar::proto::GenerateResponse {
        text: Some("ok".to_string()),
        ..Default::default()
    };
    // text is optional field 4.
    assert_eq!(response.encode_to_vec(), [0x22, 0x02, b'o', b'k']);

    let chunk = dynamo_sglang_sidecar::proto::OpenAiStreamChunk {
        status_code: Some(404),
        ..Default::default()
    };
    // status_code is optional field 3.
    assert_eq!(chunk.encode_to_vec(), [0x18, 0x94, 0x03]);
}
