// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! The wire contract.

use prost::Message;

use super::*;

/// `proto/testdata/get_snapshot_response.txtpb` as `protoc --encode` writes it.
const GOLDEN: &[u8] = include_bytes!("../../../proto/testdata/get_snapshot_response.binpb");

#[test]
fn generated_types_match_the_independent_golden_encoding() {
    assert_eq!(response().encode_to_vec(), GOLDEN);
    assert_eq!(
        GetSnapshotResponse::decode(GOLDEN).expect("the golden encoding decodes"),
        response()
    );
}
