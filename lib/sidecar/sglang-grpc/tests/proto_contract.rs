// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use prost::Message;
use sha2::{Digest, Sha256};

const LOCAL_PROTO_SHA256: &str = "f3d5bf6c18dd95248c311f1368a77631862d9c9f0febe748d19964b7e1154f07";

#[test]
fn vendored_proto_source_matches_the_pinned_contract() {
    let digest = Sha256::digest(include_bytes!("../proto/sglang.proto"));
    let actual = digest
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect::<String>();
    assert_eq!(actual, LOCAL_PROTO_SHA256);
}

#[test]
fn generate_request_keeps_its_released_wire_tags() {
    let request = dynamo_sglang_grpc::GenerateRequest {
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
