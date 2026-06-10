// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Serde roundtrip tests for `UniversalBlock` and `RequestMmObjectInfo`
//! across JSON and msgpack (rmp-serde).

use dynamo_kv_hashing::{Request, RequestMmObjectInfo, Token, UniversalBlock};

fn sample_blocks() -> Vec<UniversalBlock> {
    let tokens: Vec<Token> = (0..16).collect();
    let mm = vec![RequestMmObjectInfo {
        mm_hash: 0x1234_5678_9ABC_DEF0,
        offset: 4,
        length: 6,
    }];
    let r = Request::builder()
        .tokens(tokens)
        .lora_name(Some("lora-x".to_string()))
        .salt(Some("salt".to_string()))
        .mm_info(mm)
        .build()
        .unwrap();
    r.into_blocks(4).unwrap()
}

#[test]
fn universal_block_json_roundtrip() {
    let blocks = sample_blocks();
    let s = serde_json::to_string(&blocks).expect("json encode");
    let back: Vec<UniversalBlock> = serde_json::from_str(&s).expect("json decode");
    assert_eq!(blocks, back);
}

#[test]
fn universal_block_msgpack_roundtrip() {
    let blocks = sample_blocks();
    let bytes = rmp_serde::to_vec(&blocks).expect("msgpack encode");
    let back: Vec<UniversalBlock> = rmp_serde::from_slice(&bytes).expect("msgpack decode");
    assert_eq!(blocks, back);
}

#[test]
fn request_mm_object_info_roundtrips() {
    let mm = vec![
        RequestMmObjectInfo {
            mm_hash: 0xAA,
            offset: 0,
            length: 3,
        },
        RequestMmObjectInfo {
            mm_hash: 0xBB,
            offset: 7,
            length: 5,
        },
    ];
    let s = serde_json::to_string(&mm).unwrap();
    let back: Vec<RequestMmObjectInfo> = serde_json::from_str(&s).unwrap();
    assert_eq!(mm, back);

    let bytes = rmp_serde::to_vec(&mm).unwrap();
    let back: Vec<RequestMmObjectInfo> = rmp_serde::from_slice(&bytes).unwrap();
    assert_eq!(mm, back);
}
