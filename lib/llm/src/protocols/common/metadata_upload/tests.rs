// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use super::*;
use serde_json::json;

fn presigned() -> Value {
    json!({
        "kind": "generation_artifact", "contents": ["moe_routes"], "mode": "object_store",
        "destination": {
            "kind": "presigned_http_put",
            "url": "https://storage.example/a%2Fb?X-Amz-Signature=secret&x=2&x=1",
            "required_headers": {"Content-Type": "application/octet-stream", "x-amz-security-token": "private-token"},
            "expires_at": "2030-01-01T00:00:00Z", "max_bytes": 67108864,
            "object_id": "object-42", "verification_ref": "private-verifier"
        }
    })
}

#[test]
fn preserves_exact_presigned_capability_and_redacts_debug() {
    let input = presigned();
    let upload: MetadataUpload = serde_json::from_value(input.clone()).unwrap();
    assert_eq!(serde_json::to_value(&upload).unwrap(), input);
    let debug = format!("{upload:?}");
    for secret in [
        "storage.example",
        "secret",
        "private-token",
        "private-verifier",
    ] {
        assert!(!debug.contains(secret));
    }
}

#[test]
fn legacy_round_trip_preserves_main_shape_and_optional_fallback() {
    for input in [
        json!({"url": "s3://bucket/prefix"}),
        json!({"url": "s3://bucket/prefix", "fallback_url": "s3://other/prefix"}),
    ] {
        let upload: MetadataUpload = serde_json::from_value(input.clone()).unwrap();
        assert_eq!(serde_json::to_value(&upload).unwrap(), input);
        assert!(!format!("{upload:?}").contains("bucket"));
        assert!(!upload.is_artifact());
    }
}

#[test]
fn rejects_invalid_presigned_controls_without_leaking_values() {
    for (field, invalid) in [
        ("url", json!("ftp://storage.example/secret")),
        ("url", json!("https://user:secret@storage.example/object")),
        ("url", json!("https://storage.example/object#secret")),
        ("max_bytes", json!(0)),
        ("max_bytes", json!("secret")),
        ("expires_at", json!("secret")),
        ("object_id", json!("")),
        ("verification_ref", json!(" ")),
        (
            "required_headers",
            json!({"X-Token": "secret", "x-token": "secret"}),
        ),
        (
            "required_headers",
            json!({"x-token": "secret\r\nInjected: true"}),
        ),
        ("extra", json!("secret")),
    ] {
        let mut input = presigned();
        input["destination"][field] = invalid;
        let error = serde_json::from_value::<MetadataUpload>(input).unwrap_err();
        assert!(!error.to_string().contains("secret"));
    }
}

#[test]
fn rejects_traversal_and_invalid_mooncake_retention() {
    for key in [
        "../data",
        "/data",
        "a/../b",
        "a\\b",
        "s3://bucket/key",
        "a//b",
    ] {
        let input = json!({"kind": "generation_artifact", "contents": [], "mode": "object_store", "destination": {"kind": "managed_store", "profile": "test", "object_key": key}});
        assert!(serde_json::from_value::<MetadataUpload>(input).is_err());
    }
    let input = json!({"kind": "generation_artifact", "contents": [], "mode": "relay_store", "destination": {"kind": "mooncake_store", "profile": "test", "consumer_group": "trainers", "retention_seconds": 0}});
    assert!(serde_json::from_value::<MetadataUpload>(input).is_err());
}
