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
fn generate_request_keeps_pending_media_wire_tags() {
    let request = dynamo_sglang_sidecar::proto::GenerateRequest {
        image_data: vec!["i".to_string()],
        video_data: vec!["v".to_string()],
        audio_data: vec!["a".to_string()],
        mm_hashes: vec!["h".to_string()],
        ..Default::default()
    };
    // Repeated string fields 18 to 21 come from unmerged SGLang changes; check
    // them again against the merged upstream file.
    assert_eq!(
        request.encode_to_vec(),
        [
            0x92, 0x01, 0x01, b'i', 0x9a, 0x01, 0x01, b'v', 0xa2, 0x01, 0x01, b'a', 0xaa, 0x01,
            0x01, b'h'
        ]
    );
}
