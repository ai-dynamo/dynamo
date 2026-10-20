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
fn engine_state_snapshot_has_stable_wire_tags() {
    let snapshot = dynamo_sglang_sidecar::proto::EngineStateSnapshot {
        instance_id: 7,
        revision: 9,
        healthy: true,
        is_pause: true,
        ..Default::default()
    };
    assert_eq!(
        snapshot.encode_to_vec(),
        [0x08, 0x07, 0x10, 0x09, 0x18, 0x01, 0x20, 0x01]
    );
}
