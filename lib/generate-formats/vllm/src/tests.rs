// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use super::*;

fn routing_metadata(
    kv_cache_block_size: u32,
    tower_connector_lora_enabled: bool,
    lora_name: Option<&str>,
) -> Worker<'_> {
    Worker {
        model: "test-model",
        kv_cache_block_size,
        tower_connector_lora_enabled,
        lora_name,
    }
}

fn project(raw: &Value, worker: &Worker<'_>) -> PreprocessedRequest {
    preprocessed_from_generate(
        serde_json::from_value(raw.clone()).unwrap(),
        worker.model,
        None,
        "resolved-request",
        worker,
    )
    .unwrap()
}

#[test]
fn model_alias_is_canonicalized_only_when_the_client_supplies_a_model() {
    for model in [None, Some("public-alias")] {
        let mut body = json!({"token_ids":[1], "sampling_params":{}});
        if let Some(model) = model {
            body["model"] = json!(model);
        }
        let request = project(&body, &routing_metadata(16, false, None));
        let envelope = &request.extra_args.unwrap()["vllm_tito"];
        assert_eq!(
            envelope.get("model"),
            model.map(|_| json!("test-model")).as_ref()
        );
    }
}

#[test]
fn engine_fields_reach_envelope_with_resolved_id_and_cache_namespace() {
    let raw = serde_json::json!({
        "request_id": "req-forward",
        "token_ids": [1, 2],
        "sampling_params": {
            "max_tokens": 8,
            "future_sampling_field": {"nested": true}
        },
        "model": "test-model",
        "stream": true,
        "stream_options": {"include_usage": true},
        "cache_salt": "tenant-a",
        "features": {"future_feature": [1, 2, 3]},
        "priority": 7,
        "kv_transfer_params": {"remote": "worker-a"},
        "future_top_level_field": {"anything": "works"}
    });
    let preprocessed = project(&raw, &routing_metadata(16, false, None));
    assert_eq!(preprocessed.stop_conditions.max_tokens, Some(8));
    assert_eq!(preprocessed.stop_conditions.min_tokens, None);
    assert_eq!(
        preprocessed
            .routing
            .as_ref()
            .unwrap()
            .expected_output_tokens,
        Some(8)
    );
    assert_eq!(
        preprocessed.routing.as_ref().unwrap().priority,
        Some(-7),
        "vLLM lower-is-higher priority must be inverted for Dynamo routing"
    );
    assert_eq!(
        preprocessed.routing.as_ref().unwrap().priority_jump,
        Some(0.0)
    );
    assert_eq!(
        preprocessed
            .routing
            .as_ref()
            .unwrap()
            .cache_namespace
            .as_deref(),
        Some("tenant-a")
    );
    let envelope = preprocessed
        .extra_args
        .as_ref()
        .unwrap()
        .get("vllm_tito")
        .expect("vllm_tito envelope");

    let mut expected_envelope = raw;
    expected_envelope["request_id"] = serde_json::json!("resolved-request");
    let expected_token_ids = expected_envelope
        .as_object_mut()
        .and_then(|object| object.remove("token_ids"))
        .expect("token_ids in client request");
    assert_eq!(preprocessed.token_ids.as_slice(), &[1, 2]);
    assert_eq!(expected_token_ids, serde_json::json!([1, 2]));
    assert_eq!(envelope, &expected_envelope);
    assert!(envelope.get("token_ids").is_none());
}

#[test]
fn multimodal_routing_matches_worker_events_and_preserves_execution_payload() {
    let hash_a = "a".repeat(64);
    let hash_b = "b".repeat(64);
    let raw = serde_json::json!({
        "token_ids": [10, 11, 12, 12, 12, 15, 16, 12, 12, 19],
        "sampling_params": {},
        "features": {
            "mm_hashes": {"image": [hash_a, hash_b]},
            "mm_placeholders": {"image": [
                {"offset": 2, "length": 3},
                {"offset": 7, "length": 2}
            ]},
            "kwargs_data": {"image": ["b3BhcXVlLWE=", "b3BhcXVlLWI="]}
        }
    });
    let preprocessed = project(&raw, &routing_metadata(4, false, None));

    let hash_a = dynamo_kv_router::protocols::hash_mm_identifier(&preprocessed_mm_identifier(
        "image",
        b"opaque-a",
    ))
    .expect("content-derived hash");
    let hash_b = dynamo_kv_router::protocols::hash_mm_identifier(&preprocessed_mm_identifier(
        "image",
        b"opaque-b",
    ))
    .expect("content-derived hash");
    let pad_a = dynamo_kv_router::protocols::pad_value_for_mm_hash(hash_a);
    let pad_b = dynamo_kv_router::protocols::pad_value_for_mm_hash(hash_b);
    let mm = preprocessed
        .mm_routing_info
        .as_ref()
        .expect("multimodal routing projection");
    assert_eq!(
        mm.routing_token_ids,
        vec![10, 11, pad_a, pad_a, pad_a, 15, 16, pad_b, pad_b, 19, 0, 0]
    );
    assert!(mm.block_mm_infos.is_empty());
    assert_eq!(mm.expanded_prompt_len, 10);

    assert_eq!(
        preprocessed.token_ids.as_slice(),
        &[10, 11, 12, 12, 12, 15, 16, 12, 12, 19]
    );
    let envelope = preprocessed
        .extra_args
        .as_ref()
        .unwrap()
        .get("vllm_tito")
        .expect("vllm_tito envelope");
    assert_eq!(envelope["features"], raw["features"]);
    assert_eq!(
        preprocessed
            .extra_args
            .as_ref()
            .unwrap()
            .get("dynamo_mm_routing_hashes"),
        Some(&serde_json::json!([
            dynamo_llm::protocols::common::preprocessed_mm_routing_hash("image", b"opaque-a"),
            dynamo_llm::protocols::common::preprocessed_mm_routing_hash("image", b"opaque-b")
        ]))
    );

    // A frontend-approved, marker-form hash must produce the same KV hash
    // on the request and event paths.
    let mm_identifier = "1234567890abcdef".repeat(4);
    let request: GenerateRequest = serde_json::from_value(serde_json::json!({
        "token_ids": [10, 99, 99, 20],
        "sampling_params": {},
        "features": {
            "mm_hashes": {"image": [mm_identifier.clone()]},
            "mm_placeholders": {"image": [{"offset": 1, "length": 2}]},
            "kwargs_data": {"image": ["cm91dGluZy1h"]}
        }
    }))
    .expect("deserialize request");
    let preprocessed = preprocessed_from_generate(
        request,
        "test-model",
        None,
        "resolved-request",
        &routing_metadata(4, false, None),
    )
    .expect("build request");
    let routing = preprocessed
        .mm_routing_info
        .as_ref()
        .expect("base request keeps exact MM routing");
    let request_hashes = dynamo_kv_router::protocols::compute_block_hash_for_seq(
        &routing.routing_token_ids,
        4,
        dynamo_kv_router::protocols::BlockHashOptions::default(),
    );
    let marked_identifier = preprocessed
        .extra_args
        .as_ref()
        .unwrap()
        .get("dynamo_mm_routing_hashes")
        .and_then(serde_json::Value::as_array)
        .and_then(|hashes| hashes.first())
        .and_then(serde_json::Value::as_str)
        .expect("frontend-approved marked MM hash")
        .to_string();
    let event_mm_info =
        dynamo_kv_router::zmq_wire::extra_keys_to_block_mm_infos(Some(vec![Some(vec![
            dynamo_kv_router::zmq_wire::ExtraKeyItem::HashWithUnsignedOffset((
                marked_identifier,
                1,
            )),
        ])]))
        .expect("parse canonical offset-bearing MM extra key")
        .into_iter()
        .next()
        .flatten()
        .expect("block MM metadata");
    let event_block = dynamo_kv_router::zmq_wire::create_stored_block_from_parts(
        4,
        7,
        &[10, 99, 99, 20],
        dynamo_kv_router::zmq_wire::StoredBlockOptions {
            mm_extra_info: Some(event_mm_info),
            image_token_id: Some(99),
            ..Default::default()
        },
    );

    assert_eq!(request_hashes[0], event_block.tokens_hash);
}

#[test]
fn sparse_and_ambiguous_multimodal_layouts_are_handled() {
    let mm_identifier = "opaque-renderer-image-0";
    let request: GenerateRequest = serde_json::from_value(serde_json::json!({
        "token_ids": [10, 99, 42, 99, 20],
        "sampling_params": {},
        "features": {
            "mm_hashes": {"image": [mm_identifier]},
            "mm_placeholders": {"image": [{
                "offset": 1,
                "length": 3,
                "is_embed": [true, false, true]
            }]},
            "kwargs_data": {"image": ["c3BhcnNl"]}
        }
    }))
    .expect("deserialize request");
    let routing = generate_mm_routing_info(&request, 5)
        .expect("valid sparse MM routing metadata")
        .expect("MM routing projection");
    let mm_hash = dynamo_kv_router::protocols::hash_mm_identifier(&preprocessed_mm_identifier(
        "image", b"sparse",
    ))
    .expect("content-derived identifier");
    let pad = dynamo_kv_router::protocols::pad_value_for_mm_hash(mm_hash);
    assert_eq!(routing.info.routing_token_ids, vec![10, pad, 42, pad, 20]);

    for (name, token_ids, features, expected_error) in [
        (
            "missing sparse mask",
            serde_json::json!([10, 99, 42, 99, 20]),
            serde_json::json!({
                "mm_hashes": {"image": ["image-0"]},
                "mm_placeholders": {"image": [{"offset": 1, "length": 3}]},
                "kwargs_data": {"image": ["c3BhcnNl"]}
            }),
            "mixed multimodal placeholder spans require is_embed",
        ),
        (
            "undeclared image token",
            serde_json::json!([99, 10, 99, 99, 20]),
            serde_json::json!({
                "mm_hashes": {"image": ["image-0"]},
                "mm_placeholders": {"image": [{"offset": 2, "length": 2}]},
                "kwargs_data": {"image": ["c3BhcnNl"]}
            }),
            "image tokens must be covered by multimodal placeholder ranges",
        ),
        (
            "ambiguous sparse object order",
            serde_json::json!([10, 99, 42, 99, 20, 99, 30]),
            serde_json::json!({
                "mm_hashes": {"image": ["image-a", "image-b"]},
                "mm_placeholders": {"image": [
                    {"offset": 1, "length": 3, "is_embed": [true, false, true]},
                    {"offset": 5, "length": 1}
                ]},
                "kwargs_data": {"image": ["c3BhcnNlLWE=", "c3BhcnNlLWI="]}
            }),
            "sparse multimodal layout cannot be normalized exactly by worker events",
        ),
    ] {
        let request: GenerateRequest = serde_json::from_value(serde_json::json!({
            "token_ids": token_ids,
            "sampling_params": {},
            "features": features,
        }))
        .expect("deserialize request");

        assert_eq!(
            generate_mm_routing_info(&request, 7)
                .expect_err("ambiguous layout must disable exact routing"),
            expected_error,
            "{name}"
        );
    }
}

#[test]
fn invalid_or_unsupported_multimodal_metadata_falls_back_safely() {
    for (name, token_ids, features) in [
        (
            "unrelated features",
            serde_json::json!([1, 2, 3, 4]),
            serde_json::json!({"future_feature": [1, 2, 3]}),
        ),
        (
            "malformed hashes",
            serde_json::json!([1, 2, 3, 4]),
            serde_json::json!({"mm_hashes": ["not-an-object"]}),
        ),
        (
            "empty identifier",
            serde_json::json!([1, 2, 3, 4]),
            serde_json::json!({
                "mm_hashes": {"image": [""]},
                "mm_placeholders": {"image": [{"offset": 1, "length": 2}]},
                "kwargs_data": {"image": ["opaque"]}
            }),
        ),
        (
            "overlapping placeholders",
            serde_json::json!([9, 9, 9, 4]),
            serde_json::json!({
                "mm_hashes": {"image": ["a".repeat(64), "b".repeat(64)]},
                "mm_placeholders": {"image": [
                    {"offset": 0, "length": 2},
                    {"offset": 1, "length": 2}
                ]}
            }),
        ),
        (
            "unsupported modality",
            serde_json::json!([1, 2, 3, 4]),
            serde_json::json!({
                "mm_hashes": {"audio": ["audio-0"]},
                "mm_placeholders": {"audio": [{"offset": 1, "length": 2}]}
            }),
        ),
        (
            "unpaired image metadata",
            serde_json::json!([1, 2, 3, 4]),
            serde_json::json!({
                "mm_hashes": {"image": ["image-0"]},
                "mm_placeholders": {}
            }),
        ),
        (
            "unequal item counts",
            serde_json::json!([1, 2, 3, 4]),
            serde_json::json!({
                "mm_hashes": {"image": ["image-0", "image-1"]},
                "mm_placeholders": {"image": [{"offset": 1, "length": 2}]}
            }),
        ),
        (
            "out-of-order placeholders",
            serde_json::json!([99, 99, 3, 99]),
            serde_json::json!({
                "mm_hashes": {"image": ["late", "early"]},
                "mm_placeholders": {"image": [
                    {"offset": 3, "length": 1},
                    {"offset": 0, "length": 2}
                ]}
            }),
        ),
    ] {
        let raw = serde_json::json!({
            "token_ids": token_ids,
            "sampling_params": {},
            "features": features,
        });
        let request: GenerateRequest =
            serde_json::from_value(raw.clone()).expect("deserialize request");
        let expected_token_ids = request.token_ids.clone();

        let preprocessed = preprocessed_from_generate(
            request,
            "test-model",
            None,
            "resolved-request",
            &routing_metadata(4, false, None),
        )
        .expect("invalid routing metadata must not reject execution");

        assert!(preprocessed.mm_routing_info.is_none(), "{name}");
        assert_eq!(
            preprocessed.token_ids.as_slice(),
            expected_token_ids,
            "{name}"
        );
        let envelope = preprocessed
            .extra_args
            .as_ref()
            .unwrap()
            .get("vllm_tito")
            .expect("vllm_tito envelope");
        assert_eq!(envelope["features"], raw["features"], "{name}");
        assert!(
            preprocessed
                .extra_args
                .as_ref()
                .unwrap()
                .get("dynamo_mm_routing_hashes")
                .is_none(),
            "{name}"
        );
    }

    let raw = serde_json::json!({
        "token_ids": [10, 99, 99, 20],
        "sampling_params": {},
        "features": {
            "mm_hashes": {"image": ["image-0"]},
            "mm_placeholders": {"image": [{"offset": 1, "length": 2}]},
            "kwargs_data": {"image": ["YmFzZS1pbWFnZQ=="]}
        }
    });
    let base_request: GenerateRequest =
        serde_json::from_value(raw.clone()).expect("deserialize base request");
    let base = preprocessed_from_generate(
        base_request,
        "test-model",
        None,
        "resolved-request",
        &routing_metadata(4, true, None),
    )
    .expect("build base request");
    assert!(
        base.mm_routing_info.is_some(),
        "the worker setting alone does not activate adapter-scoped MM identity"
    );

    for tower_connector_lora_enabled in [false, true] {
        let adapter_request: GenerateRequest =
            serde_json::from_value(raw.clone()).expect("deserialize adapter request");
        let adapter = preprocessed_from_generate(
            adapter_request,
            "adapter-a",
            None,
            "resolved-request",
            &routing_metadata(4, tower_connector_lora_enabled, Some("adapter-a")),
        )
        .expect("build adapter request");
        assert!(adapter.mm_routing_info.is_none());
        assert_eq!(
            adapter.routing.as_ref().unwrap().lora_name.as_deref(),
            Some("adapter-a")
        );
        assert!(
            adapter
                .extra_args
                .as_ref()
                .unwrap()
                .get("dynamo_mm_routing_hashes")
                .is_none()
        );
        let envelope = adapter
            .extra_args
            .as_ref()
            .unwrap()
            .get("vllm_tito")
            .expect("vllm_tito envelope");
        assert_eq!(envelope["features"], raw["features"]);
    }
}

#[test]
fn routing_preserves_bounds_rank_and_priority() {
    for (sampling, max, min) in [
        (json!({}), None, None),
        (json!({"min_tokens": 0}), None, Some(0)),
        (json!({"max_tokens": 8, "min_tokens": 2}), Some(8), Some(2)),
    ] {
        for (priority, expected) in [(7, -7), (-7, 7), (i32::MIN, i32::MAX)] {
            let request = serde_json::from_value(json!({
                "token_ids": [1, 2], "sampling_params": sampling, "priority": priority,
            }))
            .unwrap();
            let request = preprocessed_from_generate(
                request,
                "test-model",
                Some(3),
                "req",
                &routing_metadata(16, false, None),
            )
            .unwrap();
            assert_eq!(request.stop_conditions.max_tokens, max);
            assert_eq!(request.stop_conditions.min_tokens, min);
            let routing = request.routing.unwrap();
            assert_eq!(routing.expected_output_tokens, max);
            assert_eq!(routing.dp_rank, Some(3));
            assert_eq!(routing.priority, Some(expected));
            assert_eq!(routing.priority_jump, Some(expected.max(0) as f64));
        }
    }
}

#[test]
fn generate_projects_training_text_controls() {
    let request: GenerateRequest = serde_json::from_value(serde_json::json!({
        "token_ids": [1, 2],
        "sampling_params": {
            "temperature": 0.25,
            "top_p": 0.9,
            "top_k": -1,
            "min_p": 0.05,
            "seed": 23,
            "max_tokens": 8,
            "min_tokens": 2,
            "presence_penalty": 0.1,
            "frequency_penalty": 0.2,
            "repetition_penalty": 1.1,
            "stop_token_ids": [7, 8],
            "ignore_eos": true,
            "logprobs": 1,
            "prompt_logprobs": 1,
            "skip_reading_prefix_cache": false,
            "skip_special_tokens": false
        },
        "kv_transfer_params": {
            "connector_data": {"block_ids": [1, 2]}
        },
        "model": "test-model"
    }))
    .expect("deserialize request");

    let preprocessed = preprocessed_from_generate(
        request,
        "test-model",
        None,
        "resolved-request",
        &routing_metadata(16, false, None),
    )
    .expect("build request");

    assert_eq!(preprocessed.sampling_options.temperature, Some(0.25));
    assert_eq!(preprocessed.sampling_options.top_p, Some(0.9));
    assert_eq!(preprocessed.sampling_options.top_k, Some(-1));
    assert_eq!(preprocessed.sampling_options.min_p, Some(0.05));
    assert_eq!(preprocessed.sampling_options.seed, Some(23));
    assert_eq!(preprocessed.sampling_options.presence_penalty, Some(0.1));
    assert_eq!(preprocessed.sampling_options.frequency_penalty, Some(0.2));
    assert_eq!(preprocessed.sampling_options.repetition_penalty, Some(1.1));
    assert_eq!(preprocessed.stop_conditions.max_tokens, Some(8));
    assert_eq!(preprocessed.stop_conditions.min_tokens, Some(2));
    assert_eq!(
        preprocessed.stop_conditions.stop_token_ids_hidden,
        Some(vec![7, 8])
    );
    assert_eq!(preprocessed.stop_conditions.ignore_eos, Some(true));
    assert_eq!(preprocessed.output_options.logprobs, Some(1));
    assert_eq!(preprocessed.output_options.prompt_logprobs, Some(1));
    assert_eq!(
        preprocessed
            .extra_args
            .as_ref()
            .unwrap()
            .get("skip_reading_prefix_cache"),
        Some(&serde_json::Value::Bool(false))
    );
    assert_eq!(
        preprocessed
            .extra_args
            .as_ref()
            .unwrap()
            .get("kv_transfer_params"),
        Some(&serde_json::json!({"connector_data": {"block_ids": [1, 2]}}))
    );
    assert_eq!(
        preprocessed
            .extra_args
            .as_ref()
            .unwrap()
            .get("vllm_tito")
            .and_then(|tito| tito.get("sampling_params"))
            .and_then(|sampling| sampling.get("top_k")),
        Some(&serde_json::json!(-1))
    );
    assert_eq!(preprocessed.output_options.skip_special_tokens, Some(false));
}
