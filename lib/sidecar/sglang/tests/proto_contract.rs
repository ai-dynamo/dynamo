// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use dynamo_sglang_sidecar::proto as pb;
use prost::Message;

#[test]
fn generate_request_keeps_its_wire_tags() {
    let request = pb::GenerateRequest {
        rid: Some(pb::StringOrList {
            value: Some(pb::string_or_list::Value::One("rid".to_string())),
        }),
        input_ids: Some(pb::TokenIdsOrList {
            value: Some(pb::token_ids_or_list::Value::One(pb::TokenIds {
                ids: vec![1, 2],
            })),
        }),
        stream: Some(true),
        ..Default::default()
    };
    // rid is field 1 (StringOrList.one = 1); input_ids is field 3
    // (TokenIdsOrList.one = 1, packed TokenIds.ids = 1); stream is field 4.
    assert_eq!(
        request.encode_to_vec(),
        [
            0x0a, 0x05, 0x0a, 0x03, b'r', b'i', b'd', //
            0x1a, 0x06, 0x0a, 0x04, 0x0a, 0x02, 0x01, 0x02, //
            0x20, 0x01,
        ]
    );
}

#[test]
fn stream_items_keep_their_wire_tags() {
    let frame = pb::GenerateStreamItem {
        item: Some(pb::generate_stream_item::Item::Frame(
            pb::GenerateResponse {
                output_ids: Some(pb::TokenIds { ids: vec![7] }),
                ..Default::default()
            },
        )),
    };
    // frame is field 1; GenerateResponse.output_ids is field 3.
    assert_eq!(
        frame.encode_to_vec(),
        [0x0a, 0x05, 0x1a, 0x03, 0x0a, 0x01, 0x07]
    );

    let error = pb::GenerateStreamItem {
        item: Some(pb::generate_stream_item::Item::Error(
            pb::GenerateStreamError {
                error: Some(pb::ErrorBody {
                    message: "m".into(),
                    code: 499,
                }),
                index: None,
            },
        )),
    };
    // error is field 2; ErrorBody.message 1, code 2.
    assert_eq!(
        error.encode_to_vec(),
        [0x12, 0x08, 0x0a, 0x06, 0x0a, 0x01, b'm', 0x10, 0xf3, 0x03]
    );
}
