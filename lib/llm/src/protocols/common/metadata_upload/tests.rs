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
    let raw = serde_json::to_string(&input).unwrap();
    let upload: MetadataUpload = serde_json::from_str(&raw).unwrap();
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
fn legacy_shape_errors_name_safe_categories_without_values() {
    for (input, category) in [
        (
            json!({"url": "s3://private-bucket", "unknown": "private-value"}),
            "unknown legacy metadata_upload field",
        ),
        (
            json!({"fallback_url": "s3://private-bucket"}),
            "missing legacy metadata_upload url",
        ),
    ] {
        let error = serde_json::from_value::<MetadataUpload>(input).unwrap_err();
        let message = error.to_string();
        assert!(message.contains(category), "{message}");
        assert!(!message.contains("private-bucket"));
        assert!(!message.contains("private-value"));
    }
}

#[test]
fn rejects_duplicate_fields_in_raw_json() {
    for input in [
        r#"{"url":"s3://a","url":"s3://b"}"#,
        r#"{"url":"s3://a","u\u0072l":"s3://b"}"#,
        r#"{"url":"s3://a","fallback_url":"s3://b","fallback_url":"s3://c"}"#,
        r#"{"kind":"generation_artifact","kind":"generation_artifact","contents":[],"mode":"inline_base64"}"#,
        r#"{"kind":"generation_artifact","contents":[],"mode":"inline_base64","mode":"inline_base64"}"#,
        r#"{"kind":"generation_artifact","contents":[],"mode":"object_store","destination":{"kind":"managed_store","profile":"test","object_key":"a"},"destination":{"kind":"managed_store","profile":"test","object_key":"b"}}"#,
        r#"{"kind":"generation_artifact","contents":[],"mode":"object_store","destination":{"kind":"managed_store","profile":"test","object_key":"a","object_key":"b"}}"#,
        r#"{"kind":"generation_artifact","contents":[],"mode":"object_store","destination":{"kind":"presigned_http_put","url":"https://example.com/a","url":"https://example.com/b","required_headers":{},"expires_at":"2030-01-01T00:00:00Z","max_bytes":100,"object_id":"object"}}"#,
        r#"{"kind":"generation_artifact","contents":[],"mode":"object_store","destination":{"kind":"presigned_http_put","url":"https://example.com/a","required_headers":{"X-Token":"a","X-Token":"b"},"expires_at":"2030-01-01T00:00:00Z","max_bytes":100,"object_id":"object"}}"#,
    ] {
        let error = serde_json::from_str::<MetadataUpload>(input).unwrap_err();
        let message = error.to_string();
        assert!(message.contains("duplicate"), "{input}: {error}");
        assert!(!message.contains("s3://") && !message.contains("X-Token"));
    }
}

#[test]
fn rejects_duplicate_upload_fields_in_raw_openai_requests() {
    use crate::protocols::openai::chat_completions::NvCreateChatCompletionRequest;
    use crate::protocols::openai::completions::NvCreateCompletionRequest;

    let chat = r#"{"model":"test","messages":[{"role":"user","content":"hello"}],"nvext":{"metadata_upload":{"url":"s3://a","url":"s3://b"}}}"#;
    let error = serde_json::from_str::<NvCreateChatCompletionRequest>(chat).unwrap_err();
    assert!(
        error
            .to_string()
            .contains("duplicate metadata_upload field")
    );

    let completion = r#"{"model":"test","prompt":"hello","nvext":{"metadata_upload":{"kind":"generation_artifact","contents":[],"mode":"inline_base64","mode":"inline_base64"}}}"#;
    let error = serde_json::from_str::<NvCreateCompletionRequest>(completion).unwrap_err();
    assert!(
        error
            .to_string()
            .contains("duplicate metadata_upload field")
    );

    let valid_chat = r#"{"model":"test","messages":[{"role":"user","content":"hello"}],"nvext":{"metadata_upload":{"url":"s3://a"}}}"#;
    assert!(serde_json::from_str::<NvCreateChatCompletionRequest>(valid_chat).is_ok());
    let valid_completion = r#"{"model":"test","prompt":"hello","nvext":{"metadata_upload":{"kind":"generation_artifact","contents":[],"mode":"inline_base64"}}}"#;
    assert!(serde_json::from_str::<NvCreateCompletionRequest>(valid_completion).is_ok());
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

#[test]
fn managed_store_chunk_prefix_round_trips_and_rejects_traversal() {
    let input = json!({
        "kind": "generation_artifact", "contents": ["hidden_states"], "mode": "object_store",
        "destination": {
            "kind": "managed_store", "profile": "test", "object_key": "runs/r1/sample-7",
            "chunk_prefix": "runs/r1/chunks"
        },
        "content_options": {"hidden_states": {"chunk_tokens": 256}}
    });
    let upload: MetadataUpload = serde_json::from_value(input.clone()).unwrap();
    assert_eq!(serde_json::to_value(&upload).unwrap(), input);

    for prefix in [
        "",
        "../chunks",
        "/chunks",
        "a//b",
        "a/./b",
        "s3://bucket/chunks",
    ] {
        let input = json!({
            "kind": "generation_artifact", "contents": [], "mode": "object_store",
            "destination": {
                "kind": "managed_store", "profile": "test", "object_key": "a",
                "chunk_prefix": prefix
            }
        });
        assert!(
            serde_json::from_value::<MetadataUpload>(input).is_err(),
            "{prefix}"
        );
    }
}

fn inline_with_options(options: Value) -> Value {
    json!({
        "kind": "generation_artifact",
        "contents": ["hidden_states", "selected_logprobs"],
        "mode": "inline_base64",
        "content_options": options,
    })
}

#[test]
fn content_options_round_trip_for_requested_contents() {
    let input = inline_with_options(json!({"hidden_states": {"token_start": 12}}));
    let upload: MetadataUpload = serde_json::from_value(input.clone()).unwrap();
    assert_eq!(serde_json::to_value(&upload).unwrap(), input);

    let without = json!({
        "kind": "generation_artifact", "contents": ["hidden_states"], "mode": "inline_base64"
    });
    let upload: MetadataUpload = serde_json::from_value(without.clone()).unwrap();
    assert_eq!(serde_json::to_value(&upload).unwrap(), without);
}

#[test]
fn content_options_reject_unrequested_contents_non_objects_and_oversize() {
    for options in [
        json!({"moe_routes": {"token_start": 1}}),
        json!({"hidden_states": 12}),
        json!({"hidden_states": [12]}),
        json!({"hidden_states": {"pad": "x".repeat(5000)}}),
        json!([{"hidden_states": {}}]),
    ] {
        let result = serde_json::from_value::<MetadataUpload>(inline_with_options(options));
        assert!(result.is_err());
    }
}

#[test]
fn nixl_read_requires_its_destination_and_a_bounded_lease() {
    let input = json!({
        "kind": "generation_artifact", "contents": ["hidden_states"], "mode": "nixl_read",
        "destination": {"kind": "nixl_read", "lease_seconds": 120}
    });
    let upload: MetadataUpload = serde_json::from_value(input.clone()).unwrap();
    assert_eq!(serde_json::to_value(&upload).unwrap(), input);

    for (mode, destination) in [
        (
            "nixl_read",
            json!({"kind": "nixl_read", "lease_seconds": 0}),
        ),
        (
            "nixl_read",
            json!({"kind": "nixl_read", "lease_seconds": 3601}),
        ),
        (
            "nixl_read",
            json!({"kind": "nixl_read", "lease_seconds": 60, "host": "x"}),
        ),
        (
            "nixl_read",
            json!({"kind": "mooncake_store", "profile": "p", "consumer_group": "g", "retention_seconds": 1}),
        ),
        (
            "object_store",
            json!({"kind": "nixl_read", "lease_seconds": 60}),
        ),
    ] {
        let value = json!({
            "kind": "generation_artifact", "contents": ["hidden_states"], "mode": mode,
            "destination": destination
        });
        assert!(serde_json::from_value::<MetadataUpload>(value).is_err());
    }
    let missing = json!({
        "kind": "generation_artifact", "contents": ["hidden_states"], "mode": "nixl_read"
    });
    assert!(serde_json::from_value::<MetadataUpload>(missing).is_err());
}
