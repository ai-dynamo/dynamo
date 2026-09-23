// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

macro_rules! sidecar_vllm_tests {
    (ranks) => {
        use super::local_data_parallel_range;

        // Regression: ambiguous, out-of-bounds, or overflowing ownership can advertise
        // unreachable DP ranks; validate the metadata before worker registration.
        sidecar_test! {
            lane: pre_merge;
            #[test]
            fn local_dp_ownership_requires_a_valid_unambiguous_range() {
                for (global, start, local, expected) in [(2, 0, 0, 0..2), (8, 0, 4, 0..4), (8, 4, 4, 4..8)] {
                    assert_eq!(
                        local_data_parallel_range(global, start, local).unwrap(),
                        expected
                    );
                }
                for (global, start, local) in [
                    (0, 0, 0),
                    (8, 4, 0),
                    (8, 0, 9),
                    (8, 4, 5),
                    (u32::MAX, u32::MAX - 1, 4),
                ] {
                    assert!(
                        local_data_parallel_range(global, start, local).is_err(),
                        "invalid range: {start} + {local} of {global}"
                    );
                }
            }
        }
    };
    (json) => {
        use serde_json::json;

        use super::{Kind, json_to_struct, prost_types, struct_to_json};

        sidecar_test! {
            lane: pre_merge;
            #[test]
            fn nested_payload_round_trips_without_shape_changes() {
                let payload = json!({
                    "string": "value",
                    "bool": true,
                    "number": 42,
                    "null": null,
                    "list": [1, "two", false, {"nested": 3.5}],
                });
                let encoded = json_to_struct(payload.clone()).expect("encode");
                assert_eq!(struct_to_json(encoded).expect("decode"), payload);
            }
        }

        sidecar_test! {
            lane: pre_merge;
            #[test]
            fn rejects_non_objects_and_inexact_integers() {
                assert!(json_to_struct(json!([1, 2])).is_err());
                assert!(json_to_struct(json!({"value": 9_007_199_254_740_993_u64})).is_err());
            }
        }

        sidecar_test! {
            lane: pre_merge;
            #[test]
            fn integer_boundaries_and_negative_inexact_values_are_checked() {
                for number in [-(1_i64 << 53), -1, 0, 1, 1_i64 << 53] {
                    let value = json!({"nested": [number, {"fraction": -3.25}]});
                    assert_eq!(
                        struct_to_json(json_to_struct(value.clone()).unwrap()).unwrap(),
                        value
                    );
                }
                for number in [i64::MIN, -(1_i64 << 53) - 1, (1_i64 << 53) + 1, i64::MAX] {
                    let error = json_to_struct(json!({"nested": [number]})).unwrap_err();
                    assert_eq!(
                        error.error_type(),
                        dynamo_backend_common::ErrorType::Backend(
                            dynamo_backend_common::BackendError::InvalidArgument
                        )
                    );
                    assert!(error.to_string().contains("cannot be represented exactly"));
                }
            }
        }

        sidecar_test! {
            lane: pre_merge;
            #[test]
            fn incoming_nonfinite_numbers_fail_inside_nested_payloads() {
                for value in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
                    let wire = prost_types::Struct {
                        fields: [(
                            "nested".into(),
                            prost_types::Value {
                                kind: Some(Kind::ListValue(prost_types::ListValue {
                                    values: vec![prost_types::Value {
                                        kind: Some(Kind::NumberValue(value)),
                                    }],
                                })),
                            },
                        )]
                        .into_iter()
                        .collect(),
                    };
                    let error = struct_to_json(wire).unwrap_err();
                    assert_eq!(
                        error.error_type(),
                        dynamo_backend_common::ErrorType::Backend(dynamo_backend_common::BackendError::Unknown)
                    );
                    assert!(error.to_string().contains("NaN or infinity"));
                }
            }
        }
    };
    (lora) => {
        use super::*;
        use dynamo_backend_common::{BackendError, ErrorType};
        use serde_json::json;

        sidecar_test! {
            lane: pre_merge;
            #[test]
            fn lora_load_payload_validates_names_and_source_schemes_without_io() {
                for uri in [
                    "file:///models/adapter",
                    "hf://org/adapter",
                    "s3://bucket/adapter",
                ] {
                    let update =
                        parse_load_lora(&json!({"lora_name": "adapter-a", "source": {"uri": uri}})).unwrap();
                    assert_eq!(
                        update,
                        LoadLoraUpdate {
                            name: "adapter-a".into(),
                            uri: uri.into()
                        }
                    );
                }
                for body in [
                    json!(null),
                    json!({}),
                    json!({"lora_name": " "}),
                    json!({"lora_name": "adapter", "source": {"uri": ""}}),
                    json!({"lora_name": "adapter", "source": {"uri": "https://example.com/adapter"}}),
                    json!({"lora_name": "adapter", "source": {"uri": "file://host/adapter"}}),
                    json!({"lora_name": "adapter", "source": {"uri": "file:///adapter?query=1"}}),
                    json!({"lora_name": "adapter", "source": {"uri": "file:///adapter#fragment"}}),
                ] {
                    let error = parse_load_lora(&body).unwrap_err();
                    assert_eq!(
                        error.error_type(),
                        ErrorType::Backend(BackendError::InvalidArgument)
                    );
                }
            }
        }

        sidecar_test! {
            lane: pre_merge;
            #[test]
            fn native_lora_inventory_requires_unique_usable_identity() {
                let adapter = |name: &str, id| pb::LoraAdapter {
                    lora_name: name.into(),
                    lora_id: id,
                    source_path: "/models/adapter".into(),
                };
                let sorted = validate_inventory(vec![adapter("b", 2), adapter("a", 1)]).unwrap();
                assert_eq!(
                    sorted
                        .iter()
                        .map(|item| item.lora_name.as_str())
                        .collect::<Vec<_>>(),
                    ["a", "b"]
                );
                for inventory in [
                    vec![adapter(" ", 1)],
                    vec![adapter("a", 0)],
                    vec![adapter("a", -1)],
                    vec![pb::LoraAdapter {
                        source_path: String::new(),
                        ..adapter("a", 1)
                    }],
                    vec![adapter("a", 1), adapter("a", 2)],
                    vec![adapter("a", 1), adapter("b", 1)],
                ] {
                    let error = validate_inventory(inventory).unwrap_err();
                    assert_eq!(
                        error.error_type(),
                        ErrorType::Backend(BackendError::Unknown)
                    );
                }
            }
        }
    };
    (candidates) => {
        use super::{pb, top_n_candidates};

        sidecar_test! {
            lane: pre_merge;
            #[test]
            fn full_vocabulary_logprobs_select_all_candidates() {
                let candidates = top_n_candidates(u32::MAX).expect("map full vocabulary");
                assert_eq!(
                    candidates.select,
                    Some(pb::candidate_tokens::Select::All(true))
                );
            }
        }
    };
    (model) => {
        use super::*;
        use crate::unit_vllm_fixtures::{model_info, server_info};
        use serde_json::json;




        sidecar_test! {
            lane: pre_merge;
            #[test]
            fn engine_config_advertises_supported_capabilities() {
                let model = DiscoveredModel::from_proto(model_info(), server_info()).expect("valid discovery");
                assert!(
                    !model
                        .engine_config()
                        .runtime_data
                        .contains_key("vllm_inference_v1_generate")
                );
                assert_eq!(
                    model
                        .engine_config()
                        .runtime_data
                        .get(dynamo_llm::lora::LORA_REQUIRES_REGISTRATION),
                    Some(&json!(true))
                );
            }
        }

        sidecar_test! {
            lane: pre_merge;
            #[test]
            fn rl_worker_metadata_identifies_zero_parallelism_dimensions() {
                for (dimension, expected) in [
                    ("tensor", "tensor-parallel size of zero"),
                    ("pipeline", "pipeline-parallel size of zero"),
                ] {
                    let mut server = server_info();
                    let parallelism = server.parallelism.as_mut().expect("parallelism metadata");
                    match dimension {
                        "tensor" => parallelism.tensor_parallel_size = 0,
                        "pipeline" => parallelism.pipeline_parallel_size = 0,
                        _ => unreachable!(),
                    }
                    let model = DiscoveredModel::from_proto(model_info(), server).expect("valid discovery");
                    let error = model.rl_worker_metadata(None, None).unwrap_err();
                    assert!(error.to_string().contains(expected));
                }
            }
        }

        sidecar_test! {
            lane: pre_merge;
            #[test]
            fn discovery_rejects_zero_data_parallelism() {
                let mut server = server_info();
                server
                    .parallelism
                    .as_mut()
                    .expect("parallelism metadata")
                    .data_parallel_size = 0;

                let error = DiscoveredModel::from_proto(model_info(), server)
                    .expect_err("zero data parallelism must fail discovery");

                assert!(error.to_string().contains("data-parallel size of zero"));
            }
        }

        sidecar_test! {
            lane: pre_merge;
            #[test]
            fn startup_compatibility_rejects_parallelism_change() {
                let bootstrap = DiscoveredModel::from_proto(model_info(), server_info())
                    .expect("valid bootstrap discovery");

                for dimension in ["tensor", "pipeline", "local_dp"] {
                    let mut changed_server = server_info();
                    let parallelism = changed_server
                        .parallelism
                        .as_mut()
                        .expect("parallelism metadata");
                    match dimension {
                        "tensor" => parallelism.tensor_parallel_size += 1,
                        "pipeline" => parallelism.pipeline_parallel_size += 1,
                        "local_dp" => parallelism.data_parallel_size_local = 1,
                        _ => unreachable!(),
                    }
                    let observed = DiscoveredModel::from_proto(model_info(), changed_server)
                        .expect("valid startup discovery");

                    assert!(
                        bootstrap.ensure_startup_compatible(&observed).is_err(),
                        "{dimension} parallelism change should be rejected"
                    );
                }
            }
        }

        sidecar_test! {
            lane: pre_merge;
            #[test]
            fn discovery_rejects_incompatible_model_metadata() {
                let mut unsupported_api = server_info();
                unsupported_api.api_version = "unsupported".to_string();

                let mut missing_served_name = model_info();
                missing_served_name.served_model_name.clear();

                let mut unsupported_input = model_info();
                unsupported_input.supports_token_ids_input = false;

                for (case, model, server) in [
                    ("unsupported API", model_info(), unsupported_api),
                    ("missing served name", missing_served_name, server_info()),
                    ("unsupported input", unsupported_input, server_info()),
                ] {
                    assert!(
                        DiscoveredModel::from_proto(model, server).is_err(),
                        "{case} metadata should be rejected"
                    );
                }
            }
        }

        sidecar_test! {
            lane: pre_merge;
            #[test]
            fn discovery_rejects_nonzero_dp_start_without_local_size() {
                let mut server = server_info();
                let parallelism = server.parallelism.as_mut().unwrap();
                parallelism.data_parallel_size = 8;
                parallelism.data_parallel_rank = 4;
                assert!(DiscoveredModel::from_proto(model_info(), server).is_err());
            }
        }

        sidecar_test! {
            lane: pre_merge;
            #[test]
            fn engine_config_handles_zero_and_inexact_aggregate_kv_capacity() {
                for (aggregate_blocks, expected_per_rank_blocks) in [(0, None), (4097, Some(2048))] {
                    let mut server = server_info();
                    server.total_kv_blocks = aggregate_blocks;

                    let model =
                        DiscoveredModel::from_proto(model_info(), server).expect("valid discovery metadata");
                    let registration = model.engine_config().llm.expect("LLM registration");

                    assert_eq!(
                        registration.total_kv_blocks, expected_per_rank_blocks,
                        "aggregate blocks {aggregate_blocks}"
                    );
                }
            }
        }

        sidecar_test! {
            lane: pre_merge;
            #[test]
            fn discovered_aliases_lora_and_legacy_parallelism_are_preserved() {
                let model = DiscoveredModel::from_proto(model_info(), server_info()).unwrap();
                let config = model.engine_config();
                assert_eq!(config.model_aliases, vec!["model-alias"]);
                for name in ["model-source", "served-model", "model-alias"] {
                    assert!(model.is_base_model_name(name));
                }
                assert!(!model.is_base_model_name("other-adapter"));
                assert!(model.supports_lora());
                let llm = config.llm.unwrap();
                assert_eq!(llm.max_gpu_lora_count, Some(4));
                let mut info = model_info();
                info.supports_lora = false;
                let server = pb::ServerInfo {
                    api_version: "vllm".into(),
                    ..Default::default()
                };
                let model = DiscoveredModel::from_proto(info, server).unwrap();
                let llm = model.engine_config().llm.unwrap();
                assert_eq!(llm.max_gpu_lora_count, None);
                assert_eq!(
                    (llm.data_parallel_size, llm.data_parallel_start_rank),
                    (None, None)
                );
                assert!(!model.supports_lora());
            }
        }

        sidecar_test! {
            lane: pre_merge;
            #[test]
            fn incompatible_bootstrap_identity_and_capabilities_fail_closed() {
                let baseline = DiscoveredModel::from_proto(model_info(), server_info()).unwrap();
                let model_changes: &[fn(&mut pb::ModelInfo)] = &[
                    |m| m.model_id = "changed".into(),
                    |m| m.served_model_name = "changed".into(),
                    |m| m.served_model_aliases.push("changed".into()),
                    |m| m.reasoning_parser = "changed".into(),
                    |m| m.tool_call_parser = "changed".into(),
                    |m| m.supports_lora = false,
                ];
                for change in model_changes {
                    let mut info = model_info();
                    change(&mut info);
                    let observed = DiscoveredModel::from_proto(info, server_info()).unwrap();
                    assert!(
                        baseline
                            .ensure_startup_compatible(&observed)
                            .unwrap_err()
                            .to_string()
                            .contains("identity changed")
                    );
                }
                let mut server = server_info();
                server.rl_capabilities = None;
                let observed = DiscoveredModel::from_proto(model_info(), server).unwrap();
                assert!(
                    baseline
                        .ensure_startup_compatible(&observed)
                        .unwrap_err()
                        .to_string()
                        .contains("RL capabilities changed")
                );
            }
        }

        sidecar_test! {
            lane: pre_merge;
            #[test]
            fn absent_effective_attention_block_size_uses_legacy_fallback() {
                for (effective, physical, expected) in [
                    (None, 16, Some(16)),
                    (Some(0), 16, Some(16)),
                    (None, 0, None),
                ] {
                    let mut server = server_info();
                    server.effective_attention_block_size = effective;
                    server.kv_block_size = physical;
                    let model = DiscoveredModel::from_proto(model_info(), server).unwrap();
                    assert_eq!(
                        model.engine_config().llm.unwrap().kv_cache_block_size,
                        expected
                    );
                }
            }
        }

        sidecar_test! {
            lane: pre_merge;
            #[test]
            fn unrepresentable_effective_block_size_is_rejected_without_truncation() {
                let mut server = server_info();
                server.effective_attention_block_size = Some(u64::from(u32::MAX) + 1);
                let error = DiscoveredModel::from_proto(model_info(), server).unwrap_err();
                assert_eq!(
                    error.error_type(),
                    dynamo_backend_common::ErrorType::Backend(dynamo_backend_common::BackendError::Unknown)
                );
                assert!(error.to_string().contains("effective attention block size"));
            }
        }

        sidecar_test! {
            lane: pre_merge;
            #[test]
            fn missing_model_identity_is_rejected() {
                assert!(DiscoveredModel::from_proto({ let mut info = model_info(); info.model_id.clear(); info }, server_info()).is_err());
            }
        }
    };
    (worker) => {
        use super::*;
        use crate::unit_vllm_fixtures::{model_info, server_info};





        use super::unit_support::worker::{args, worker};

        sidecar_test! {
            lane: pre_merge;
            #[test]
            fn worker_omits_parsers_and_preserves_encode_options() {
                for mode in ["aggregated", "prefill", "decode", "encode"] {
                    let (_, config) = worker(mode);
                    assert!(config.tool_call_parser.is_none());
                    assert!(config.reasoning_parser.is_none());
                    if mode == "encode" {
                        assert_eq!(config.namespace, "test-namespace");
                        assert_eq!(config.component, "encode");
                        assert_eq!(config.endpoint, "tokens");
                        assert_eq!(config.custom_jinja_template.as_deref(), Some(std::path::Path::new("local-template.jinja")));
                        assert_eq!(config.model_name, "model-source");
                        assert_eq!(config.served_model_name.as_deref(), Some("served-model"));
                        assert!(config.enable_kv_routing);
                        assert_eq!(config.disaggregation_mode.as_str(), "encode");
                    }
                }
            }
        }

        sidecar_test! {
            lane: pre_merge;
            #[test]
            fn unsupported_parsers_and_encode_models_are_rejected() {
                for flag in ["--dyn-tool-call-parser", "--dyn-reasoning-parser"] {
                    let error = VllmSidecarEngine::from_args(Some(vec![
                        "sidecar".into(),
                        "--grpc-endpoint".into(),
                        "127.0.0.1:12345".into(),
                        flag.into(),
                        "parser".into(),
                    ]))
                    .err()
                    .expect("unsupported parser");
                    assert_eq!(
                        error.error_type(),
                        dynamo_backend_common::ErrorType::Backend(
                            dynamo_backend_common::BackendError::InvalidArgument
                        )
                    );
                    assert!(error.to_string().contains("does not preserve"));
                }
                let model = DiscoveredModel::from_proto(model_info(), server_info()).unwrap();
                let error = VllmSidecarEngine::from_discovered(args("encode"), model)
                    .err()
                    .expect("encode requires media");
                assert!(error.to_string().contains("requires a multimodal engine"));
            }
        }

        sidecar_test! {
            lane: pre_merge;
            #[tokio::test]
            async fn draft_updates_require_both_native_capabilities() {
                for flags in [
                    None,
                    Some((false, false)),
                    Some((false, true)),
                    Some((true, false)),
                    Some((true, true)),
                ] {
                    let mut server = server_info();
                    match flags {
                        None => server.rl_capabilities = None,
                        Some((transfer, draft)) => {
                            let capabilities = server.rl_capabilities.as_mut().unwrap();
                            capabilities.weight_transfer_enabled = transfer;
                            capabilities.draft_weight_updates_enabled = draft;
                        }
                    }
                    let model = DiscoveredModel::from_proto(model_info(), server).unwrap();
                    let (engine, _) = VllmSidecarEngine::from_discovered(args("aggregated"), model).unwrap();
                    let supported = flags == Some((true, true));
                    assert_eq!(
                        engine
                            .supported_updates()
                            .await
                            .unwrap()
                            .iter()
                            .any(|name| name == "start_draft_weight_update"),
                        supported,
                        "capabilities: {flags:?}"
                    );
                    assert!(engine.client.get().is_none());
                    if !supported {
                        assert_eq!(
                            engine
                                .engine_update("start_draft_weight_update".into(), serde_json::json!({}))
                                .await
                                .unwrap(),
                            serde_json::json!({
                                "status": "error",
                                "message": "unsupported engine update: start_draft_weight_update"
                            }),
                            "capabilities: {flags:?}"
                        );
                        assert!(engine.client.get().is_none());
                    }
                }
            }
        }
    };
    (requests) => {
        use super::*;
        use crate::unit_fixtures::minimal_request;
        use crate::unit_vllm_fixtures::*;
        use dynamo_backend_common::engine::RoutingHints;
        use dynamo_backend_common::{
            BackendError, ErrorType, OutputOptions, SamplingOptions, StopConditions,
        };
        use serde_json::json;

        fn assert_invalid(error: DynamoError) {
            assert_eq!(
                error.error_type(),
                ErrorType::Backend(BackendError::InvalidArgument)
            );
        }

        sidecar_test! {
            lane: pre_merge;
            #[test]
            fn compatibility_envelope_preserves_typed_controls() {
                for mode in [DisaggregationMode::Aggregated, DisaggregationMode::Decode] {
                    let request = PreprocessedRequest::builder()
                        .model("served-model".to_string())
                        .token_ids(vec![11, 22, 33])
                        .stop_conditions(StopConditions {
                            max_tokens: Some(8),
                            min_tokens: Some(2),
                            ignore_eos: Some(true),
                            ..Default::default()
                        })
                        .sampling_options(SamplingOptions {
                            n: Some(1),
                            ..Default::default()
                        })
                        .output_options(OutputOptions::default())
                        .prefill_result(if mode.is_decode() {
                            decode_request().prefill_result
                        } else {
                            None
                        })
                        .extra_args(Some(json!({
                            "vllm_tito": {
                                "sampling_params": {
                                    "max_tokens": 8,
                                    "min_tokens": 2,
                                    "ignore_eos": true,
                                    "logprobs": 2,
                                    "prompt_logprobs": 3,
                                    "skip_special_tokens": false
                                }
                            }
                        })))
                        .build()
                        .expect("v1.4 request");
                    let wire = build_generate_request(request, "legacy".to_string(), mode)
                        .expect("legacy typed controls should be preserved");
                    let stopping = wire.stopping.expect("stopping");
                    assert_eq!(stopping.max_new_tokens, 8);
                    assert_eq!(stopping.min_new_tokens, 2);
                    assert!(stopping.ignore_eos);
                    let response = wire.response.expect("response");
                    assert!(response.output_logprobs);
                    assert_eq!(
                        response.output_candidates.and_then(|tokens| tokens.select),
                        Some(pb::candidate_tokens::Select::TopN(2))
                    );
                    assert!(response.prompt_logprobs);
                    assert_eq!(
                        response.prompt_candidates.and_then(|tokens| tokens.select),
                        Some(pb::candidate_tokens::Select::TopN(3))
                    );
                    assert_eq!(response.skip_special_tokens, Some(false));
                }
            }
        }

        sidecar_test! {
            lane: pre_merge;
            #[test]
            fn native_sampling_is_rejected_instead_of_silently_discarded() {
                for mode in [DisaggregationMode::Aggregated, DisaggregationMode::Decode] {
                    let mut request = request();
                    request.extra_args = Some(json!({
                        "vllm_tito": {"sampling_params": {"temperature": 0.0}}
                    }));
                    let error = build_generate_request(request, "native".to_string(), mode)
                        .expect_err("released protocol cannot preserve native sampling semantics");
                    assert_eq!(
                        error.error_type(),
                        ErrorType::Backend(BackendError::InvalidArgument)
                    );
                    assert!(
                        error
                            .to_string()
                            .contains("sampling_params.temperature is not supported")
                    );
                }
            }
        }

        sidecar_test! {
            lane: pre_merge;
            #[test]
            fn prefill_uses_canonical_controls_without_decode_sampling_json() {
                let mut request = request();
                request.extra_args = Some(json!({
                    "vllm_tito": {"sampling_params": {"skip_special_tokens": false, "max_tokens": 100}}
                }));
                let wire = build_generate_request(request, "prefill".to_string(), DisaggregationMode::Prefill)
                    .expect("prefill does not require native decode sampling");
                let stopping = wire.stopping.expect("stopping");
                assert_eq!(stopping.max_new_tokens, 1);
                assert_eq!(stopping.min_new_tokens, 1);
                assert_eq!(wire.response.unwrap().skip_special_tokens, Some(false));
            }
        }

        sidecar_test! {
            lane: pre_merge;
            #[test]
            fn released_envelope_hydrates_kv_transfer_with_canonical_precedence() {
                let mut legacy = request();
                legacy.extra_args = Some(json!({
                    "vllm_tito": {
                        "sampling_params": {},
                        "kv_transfer_params": {"source": "legacy"}
                    }
                }));
                let legacy = normalize_response_options(legacy).expect("normalize legacy KV transfer");
                assert_eq!(
                    legacy.extra_args.as_ref().unwrap()["kv_transfer_params"],
                    json!({"source": "legacy"})
                );

                let mut canonical = request();
                canonical.extra_args = Some(json!({
                    "kv_transfer_params": {"source": "canonical"},
                    "vllm_tito": {
                        "sampling_params": {},
                        "kv_transfer_params": {"source": "legacy"}
                    }
                }));
                let canonical = normalize_response_options(canonical).expect("normalize canonical KV transfer");
                assert_eq!(
                    canonical.extra_args.as_ref().unwrap()["kv_transfer_params"],
                    json!({"source": "canonical"})
                );
            }
        }

        sidecar_test! {
            lane: pre_merge;
            #[test]
            fn unsafe_media_uuids_are_rejected() {
                for uuid in [
                    "/tmp/escape",
                    "../escape",
                    "nested/item",
                    "nested\\item",
                    ".",
                    "..",
                    "nul\0item",
                ] {
                    let mut request = epd_image_request();
                    request
                        .multi_modal_uuids
                        .as_mut()
                        .and_then(|by_modality| by_modality.get_mut("image_url"))
                        .expect("image UUIDs")[0] = Some(uuid.to_string());
                    let error = build_generate_request(
                        request,
                        "unsafe-media-uuid".to_string(),
                        DisaggregationMode::Encode,
                    )
                    .expect_err("unsafe UUID must be rejected");
                    assert!(error.to_string().contains("safe identifier"), "uuid={uuid}");
                }
            }
        }

        sidecar_test! {
            lane: pre_merge;
            #[test]
            fn encode_requests_reject_non_image_media() {
                let mut request = epd_image_request();
                request.multi_modal_data.as_mut().unwrap().insert(
                    "audio_url".to_string(),
                    vec![MultimodalData::RawUrl(
                        "https://example.com/sample.wav".to_string(),
                    )],
                );

                let error = build_generate_request(
                    request,
                    "encode-audio".to_string(),
                    DisaggregationMode::Encode,
                )
                .expect_err("Encode must remain image-only");
                assert!(error.to_string().contains("image media only"));
            }
        }

        sidecar_test! {
            lane: pre_merge;
            #[test]
            fn absent_and_explicit_zero_controls_preserve_native_sentinels() {
                for explicit in [false, true] {
                    let mut request = crate::unit_fixtures::minimal_request();
                    if explicit {
                        request.sampling_options.temperature = Some(0.0);
                        request.sampling_options.top_p = Some(0.0);
                        request.sampling_options.min_p = Some(0.0);
                        request.sampling_options.seed = Some(0);
                        request.sampling_options.presence_penalty = Some(0.0);
                        request.sampling_options.frequency_penalty = Some(0.0);
                        request.sampling_options.repetition_penalty = Some(0.0);
                        request.sampling_options.include_stop_str_in_output = Some(false);
                        request.stop_conditions.max_tokens = Some(0);
                        request.stop_conditions.min_tokens = Some(0);
                        request.stop_conditions.ignore_eos = Some(false);
                        request.output_options.logprobs = Some(0);
                        request.output_options.prompt_logprobs = Some(0);
                        request.output_options.skip_special_tokens = Some(false);
                    }
                    let wire = build_generate_request(
                        request,
                        "zero-controls".into(),
                        DisaggregationMode::Aggregated,
                    )
                    .unwrap();
                    assert_eq!(wire.request_id, "zero-controls");
                    assert_eq!(
                        wire.prompt,
                        Some(pb::generate_request::Prompt::TokenIds(pb::TokenIds {
                            ids: vec![11, 22, 33]
                        }))
                    );
                    assert_eq!(wire.temperature, explicit.then_some(0.0));
                    assert_eq!(
                        wire.sampling,
                        Some(pb::RandomSampling {
                            num_sequences: 1,
                            top_k: 0,
                            top_p: 0.0,
                            min_p: 0.0,
                            seed: explicit.then_some(0)
                        })
                    );
                    assert_eq!(
                        wire.decoding,
                        Some(pb::DecodingParameters {
                            presence_penalty: 0.0,
                            frequency_penalty: 0.0,
                            repetition_penalty: 0.0,
                            ..Default::default()
                        })
                    );
                    assert_eq!(wire.stopping, Some(pb::StoppingCriteria::default()));
                    let response = wire.response.unwrap();
                    assert_eq!(response.output_logprobs, explicit);
                    assert_eq!(response.prompt_logprobs, explicit);
                    assert_eq!(response.prompt_token_ids, explicit);
                    assert_eq!(response.skip_special_tokens, explicit.then_some(false));
                    assert_eq!(
                        response.output_candidates.and_then(|value| value.select),
                        explicit.then_some(pb::candidate_tokens::Select::TopN(0))
                    );
                    assert_eq!(
                        response.prompt_candidates.and_then(|value| value.select),
                        explicit.then_some(pb::candidate_tokens::Select::TopN(0))
                    );
                    assert!(wire.kv.unwrap().cache_salt.is_empty());
                }
            }
        }

        sidecar_test! {
            lane: pre_merge;
            #[test]
            fn native_envelope_controls_are_validated() {
                type RequestMutation = fn(&mut PreprocessedRequest);
                let cases: &[(&str, RequestMutation)] = &[
                    ("extra_args must be", |r| r.extra_args = Some(json!([]))),
                    ("extra_args.unknown", |r| {
                        r.extra_args = Some(json!({"unknown": true}))
                    }),
                    ("must be a boolean", |r| {
                        r.extra_args = Some(json!({"bypass_prefix_cache": 0}))
                    }),
                    ("does not match", |r| {
                        r.extra_args = Some(json!({"nvext": {"cache_salt": "other"}}))
                    }),
                    ("token_in must be true", |r| {
                        r.extra_args = Some(json!({"nvext": {"token_in": false}}))
                    }),
                ];
                assert!(
                    build_generate_request(
                        request(),
                        "supported".into(),
                        DisaggregationMode::Aggregated
                    )
                    .is_ok()
                );
                for (expected, change) in cases {
                    let mut request = request();
                    change(&mut request);
                    let error =
                        build_generate_request(request, "rejected".into(), DisaggregationMode::Aggregated)
                            .expect_err(expected);
                    assert_eq!(
                        error.error_type(),
                        ErrorType::Backend(BackendError::InvalidArgument),
                        "{expected}"
                    );
                    assert!(error.to_string().contains(expected), "{expected}: {error}");
                }
            }
        }

        sidecar_test! {
            lane: pre_merge;
            #[test]
            fn canonical_cache_identity_and_bypass_alias_precedence_are_preserved() {
                for (extra, expected) in [
                    (
                        json!({"bypass_prefix_cache": false, "skip_reading_prefix_cache": true}),
                        false,
                    ),
                    (json!({"skip_reading_prefix_cache": true}), true),
                    (json!({}), false),
                ] {
                    let mut request = request();
                    request.extra_args = Some(extra);
                    let wire = build_generate_request(request, "cache".into(), DisaggregationMode::Aggregated)
                        .unwrap();
                    let kv = wire.kv.unwrap();
                    assert_eq!(kv.cache_salt, "dynamo-cache-salt:cache-salt");
                    assert_eq!(kv.bypass_prefix_cache, expected);
                }
                let mut prefixed = request();
                prefixed.routing.as_mut().unwrap().cache_namespace = Some("dynamo-cache-salt:caller".into());
                prefixed.extra_args = None;
                let wire = build_generate_request(prefixed, "prefixed".into(), DisaggregationMode::Aggregated)
                    .unwrap();
                assert_eq!(
                    wire.kv.unwrap().cache_salt,
                    "dynamo-cache-salt:dynamo-cache-salt:caller"
                );
                let mut no_namespace = request();
                no_namespace.routing = None;
                no_namespace.extra_args = None;
                let wire = build_generate_request(
                    no_namespace,
                    "no-cache-identity".into(),
                    DisaggregationMode::Aggregated,
                )
                .unwrap();
                assert!(wire.kv.unwrap().cache_salt.is_empty());
            }
        }

        sidecar_test! {
            lane: pre_merge;
            #[test]
            fn native_handoff_ports_and_opaque_payloads_are_preserved() {
                for (port, expected) in [
                    (json!(5600), json!("5600")),
                    (json!(5600.0), json!("5600")),
                    (json!("5600"), json!("5600")),
                ] {
                    let mut request = decode_request();
                    let handoff =
                        json!({"remote_port": port, "opaque": {"flags": [true, null, {"ids": [1, 2]}]}});
                    request
                        .prefill_result
                        .as_mut()
                        .unwrap()
                        .disaggregated_params = handoff.clone();
                    let mut expected_handoff = handoff;
                    expected_handoff["remote_port"] = expected;
                    for _ in 0..2 {
                        let wire = build_generate_request(
                            request.clone(),
                            "decode".into(),
                            DisaggregationMode::Decode,
                        )
                        .unwrap();
                        assert_eq!(
                            struct_to_json(wire.kv.unwrap().kv_transfer_params.unwrap()).unwrap(),
                            expected_handoff
                        );
                    }
                }
            }
        }

        sidecar_test! {
            lane: pre_merge;
            #[test]
            fn vllm_extensions_preserve_native_fields() {
                let sent = build_generate_request(
                    request(),
                    "native-fields".into(),
                    DisaggregationMode::Aggregated,
                )
                .unwrap();
                assert_eq!(sent.priority, 0);
                let sampling = sent.sampling.as_ref().unwrap();
                assert_eq!(sampling.seed, Some(123));
                let decoding = sent.decoding.as_ref().unwrap();
                assert!(matches!(
                    decoding.structured_output,
                    Some(pb::decoding_parameters::StructuredOutput::Json(_))
                ));
                let stopping = sent.stopping.as_ref().unwrap();
                assert!(stopping.include_stop_strings);
                let kv = sent.kv.as_ref().unwrap();
                assert!(kv.bypass_prefix_cache);
                assert_eq!(kv.cache_salt, "dynamo-cache-salt:cache-salt");
                assert_eq!(
                    struct_to_json(kv.kv_transfer_params.clone().unwrap()).unwrap(),
                    json!({"connector_data": {"values": [1, true, null]}})
                );
            }
        }

        sidecar_test! {
            lane: pre_merge;
            #[test]
            fn encode_ignores_routing_rank() {
                let mut request = request();
                let routing = request.routing.as_mut().unwrap();
                routing.dp_rank = Some(5);
                routing.prefill_dp_rank = Some(3);
                assert_eq!(
                    data_parallel_rank(&request, DisaggregationMode::Encode),
                    None
                );
            }
        }

        sidecar_test! {
            lane: pre_merge;
            #[test]
            fn special_token_policy_is_forwarded() {
                let mut request = minimal_request();
                request.output_options.skip_special_tokens = Some(false);
                let wire = build_generate_request(
                    request, "special-tokens".into(), DisaggregationMode::Aggregated,
                ).unwrap();
                assert_eq!(wire.response.unwrap().skip_special_tokens, Some(false));
            }
        }

        sidecar_test! {
            lane: pre_merge;
            #[test]
            fn canonical_priority_preserves_native_ordering() {
                for (priority, expected) in [(-7, 7), (7, -7), (i32::MIN, i32::MAX)] {
                    let mut request = minimal_request();
                    request.routing = Some(RoutingHints {
                        priority: Some(priority),
                        ..Default::default()
                    });
                    let wire = build_generate_request(
                        request, "priority".into(), DisaggregationMode::Aggregated,
                    ).unwrap();
                    assert_eq!(wire.priority, expected);
                }
            }
        }

        sidecar_test! {
            lane: pre_merge;
            #[test]
            fn prefill_limits_generation() {
                let mut request = minimal_request();
                request.stop_conditions.max_tokens = Some(100);
                request.stop_conditions.min_tokens = Some(4);
                let wire = build_generate_request(
                    request, "prefill".into(), DisaggregationMode::Prefill,
                ).unwrap();
                let stopping = wire.stopping.unwrap();
                assert_eq!(stopping.max_new_tokens, 1);
                assert_eq!(stopping.min_new_tokens, 1);
            }
        }

        sidecar_test! {
            lane: pre_merge;
            #[test]
            fn absent_and_explicit_zero_options_remain_distinct() {
                for explicit in [false, true] {
                    let mut request = minimal_request();
                    request.sampling_options.temperature = explicit.then_some(0.0);
                    request.output_options.logprobs = explicit.then_some(0);
                    request.output_options.prompt_logprobs = explicit.then_some(0);
                    let wire = build_generate_request(
                        request, "zero".into(), DisaggregationMode::Aggregated,
                    ).unwrap();
                    assert_eq!(wire.temperature, explicit.then_some(0.0));
                    let response = wire.response.unwrap();
                    assert_eq!(response.output_logprobs, explicit);
                    assert_eq!(response.prompt_logprobs, explicit);
                }
            }
        }

        sidecar_test! {
            lane: pre_merge;
            #[test]
            fn invalid_canonical_controls_are_rejected_before_submission() {
                type Mutation = fn(&mut PreprocessedRequest);
                let cases: &[(&str, Mutation)] = &[
                    ("token_ids", |r| r.token_ids = Arc::new(Vec::new())),
                    ("n must be 1", |r| r.sampling_options.n = Some(2)),
                    ("prompt embeddings", |r| r.prompt_embeds = Some("encoded".into())),
                    ("best_of", |r| r.sampling_options.best_of = Some(2)),
                    ("beam search", |r| r.sampling_options.use_beam_search = Some(true)),
                    ("length_penalty", |r| r.sampling_options.length_penalty = Some(0.5)),
                    ("top_k", |r| r.sampling_options.top_k = Some(-2)),
                    ("visible stop", |r| r.stop_conditions.stop_token_ids_visible = Some(vec![42])),
                    ("max_thinking_tokens", |r| r.stop_conditions.max_thinking_tokens = Some(5)),
                    ("multimodal features", |r| r.mm_processor_kwargs = Some(json!({}))),
                    ("without multi_modal_data", |r| {
                        r.multi_modal_uuids = Some(std::collections::HashMap::from([(
                            "image_url".into(), vec![Some("image-a".into())],
                        )]));
                    }),
                ];
                build_generate_request(
                    minimal_request(), "supported".into(), DisaggregationMode::Aggregated,
                ).unwrap();
                for &(expected, mutate) in cases {
                    let mut request = minimal_request();
                    mutate(&mut request);
                    let error = build_generate_request(
                        request, "invalid".into(), DisaggregationMode::Aggregated,
                    ).expect_err("invalid canonical control");
                    assert!(error.to_string().contains(expected), "{expected}: {error}");
                    assert_invalid(error);
                }
            }
        }

        sidecar_test! {
            lane: pre_merge;
            #[test]
            fn guide_variants_preserve_type_and_payload() {
                use pb::decoding_parameters::StructuredOutput;
                for (guide, expected) in [
                    (
                        GuidedDecodingOptions {
                            json: Some(json!({"type": "object", "properties": {"x": {"type": "integer"}}, "required": ["x"]})),
                            ..Default::default()
                        },
                        StructuredOutput::Json(
                            r#"{"type":"object","properties":{"x":{"type":"integer"}},"required":["x"]}"#.into(),
                        ),
                    ),
                    (
                        GuidedDecodingOptions { regex: Some("[a-z]+".into()), ..Default::default() },
                        StructuredOutput::Regex("[a-z]+".into()),
                    ),
                    (
                        GuidedDecodingOptions { grammar: Some("root ::= 'yes'".into()), ..Default::default() },
                        StructuredOutput::Grammar("root ::= 'yes'".into()),
                    ),
                    (
                        GuidedDecodingOptions { choice: Some(vec!["yes".into(), "no".into()]), ..Default::default() },
                        StructuredOutput::Choice(pb::decoding_parameters::StringChoices { choices: vec!["yes".into(), "no".into()] }),
                    ),
                    (
                        GuidedDecodingOptions { structural_tag: Some(json!("<answer>")), ..Default::default() },
                        StructuredOutput::StructuralTag("<answer>".into()),
                    ),
                    (
                        GuidedDecodingOptions { structural_tag: Some(json!({"tag": "answer"})), ..Default::default() },
                        StructuredOutput::StructuralTag(r#"{"tag":"answer"}"#.into()),
                    ),
                ] {
                    let mut request = minimal_request();
                    request.sampling_options.guided_decoding = Some(guide);
                    let wire = build_generate_request(
                        request, "guide".into(), DisaggregationMode::Aggregated,
                    ).unwrap();
                    assert_eq!(wire.decoding.unwrap().structured_output, Some(expected));
                }
            }
        }

        sidecar_test! {
            lane: pre_merge;
            #[test]
            fn guide_modifiers_are_rejected() {
                for backend in [true, false] {
                    let mut request = minimal_request();
                    request.sampling_options.guided_decoding = Some(if backend {
                        GuidedDecodingOptions { backend: Some("xgrammar".into()), ..Default::default() }
                    } else {
                        GuidedDecodingOptions { whitespace_pattern: Some(" *".into()), ..Default::default() }
                    });
                    let error = build_generate_request(
                        request, "guide-modifier".into(), DisaggregationMode::Aggregated,
                    ).unwrap_err();
                    assert_invalid(error);
                }
            }
        }

        sidecar_test! {
            lane: pre_merge;
            #[test]
            fn conflicting_guides_are_rejected() {
                let mut request = minimal_request();
                request.sampling_options.guided_decoding = Some(GuidedDecodingOptions {
                    json: Some(json!({})),
                    regex: Some(".*".into()),
                    ..Default::default()
                });
                assert_invalid(build_generate_request(
                    request, "conflicting-guides".into(), DisaggregationMode::Aggregated,
                ).unwrap_err());
            }
        }

        sidecar_test! {
            lane: pre_merge;
            #[test]
            fn stopping_tokens_are_merged_and_deduplicated() {
                let mut request = minimal_request();
                request.stop_conditions.stop_token_ids = Some(vec![3, 2, 3]);
                request.stop_conditions.stop_token_ids_hidden = Some(vec![2, 4]);
                let wire = build_generate_request(
                    request, "stops".into(), DisaggregationMode::Aggregated,
                ).unwrap();
                assert_eq!(wire.stopping.unwrap().stop_token_ids, [2, 3, 4]);
            }
        }

        sidecar_test! {
            lane: pre_merge;
            #[test]
            fn top_k_preserves_native_sentinels_and_explicit_limits() {
                for (top_k, expected) in [
                    (None, 0), (Some(-1), 0), (Some(0), 0), (Some(7), 7), (Some(i32::MAX), i32::MAX as u32),
                ] {
                    let mut request = minimal_request();
                    request.sampling_options.top_k = top_k;
                    let wire = build_generate_request(
                        request, "top-k".into(), DisaggregationMode::Aggregated,
                    ).unwrap();
                    assert_eq!(wire.sampling.unwrap().top_k, expected);
                }
            }
        }

        sidecar_test! {
            lane: pre_merge;
            #[test]
            fn canonical_cache_controls_are_preserved() {
                let mut identity = minimal_request();
                identity.routing = Some(RoutingHints {
                    cache_namespace: Some("cache-salt".into()), ..Default::default()
                });
                let wire = build_generate_request(
                    identity, "cache-identity".into(), DisaggregationMode::Aggregated,
                ).unwrap();
                assert_eq!(wire.kv.unwrap().cache_salt, "dynamo-cache-salt:cache-salt");
                for bypass in [false, true] {
                    let mut request = minimal_request();
                    request.extra_args = Some(json!({"bypass_prefix_cache": bypass}));
                    let wire = build_generate_request(
                        request, "cache-bypass".into(), DisaggregationMode::Aggregated,
                    ).unwrap();
                    assert_eq!(wire.kv.unwrap().bypass_prefix_cache, bypass);
                }
            }
        }

        sidecar_test! {
            lane: pre_merge;
            #[test]
            fn decode_requires_and_preserves_valid_handoff_metadata() {
                let mut request = minimal_request();
                request.prefill_result = Some(PrefillResult {
                    disaggregated_params: json!({"remote_engine_id": "prefill-0", "remote_host": "127.0.0.1", "remote_port": 20097, "remote_block_ids": [7, 8]}),
                    prompt_tokens_details: None,
                });
                let wire = build_generate_request(
                    request, "decode".into(), DisaggregationMode::Decode,
                ).unwrap();
                assert_eq!(
                    struct_to_json(wire.kv.unwrap().kv_transfer_params.unwrap()).unwrap(),
                    json!({"remote_engine_id": "prefill-0", "remote_host": "127.0.0.1", "remote_port": "20097", "remote_block_ids": [7, 8]}),
                );
                for value in [None, Some(json!([])), Some(json!("invalid"))] {
                    let mut request = minimal_request();
                    request.prefill_result = value.map(|disaggregated_params| PrefillResult {
                        disaggregated_params, prompt_tokens_details: None,
                    });
                    assert_invalid(build_generate_request(
                        request, "bad-handoff".into(), DisaggregationMode::Decode,
                    ).unwrap_err());
                }
            }
        }

        sidecar_test! {
            lane: pre_merge;
            #[test]
            fn canonical_sampling_and_stopping_fields_are_preserved() {
                let mut request = minimal_request();
                request.sampling_options = SamplingOptions {
                    temperature: Some(0.2),
                    top_p: Some(0.9),
                    top_k: Some(4),
                    min_p: Some(0.1),
                    presence_penalty: Some(0.3),
                    frequency_penalty: Some(0.4),
                    repetition_penalty: Some(1.1),
                    ..Default::default()
                };
                request.stop_conditions = StopConditions {
                    max_tokens: Some(1),
                    min_tokens: Some(1),
                    stop: Some(vec!["done".into()]),
                    ignore_eos: Some(true),
                    ..Default::default()
                };
                let wire = build_generate_request(
                    request, "canonical-fields".into(), DisaggregationMode::Aggregated,
                ).unwrap();
                assert_eq!(wire.request_id, "canonical-fields");
                assert_eq!(wire.prompt, Some(pb::generate_request::Prompt::TokenIds(pb::TokenIds {
                    ids: vec![11, 22, 33],
                })));
                assert_eq!(wire.temperature, Some(0.2));
                let sampling = wire.sampling.unwrap();
                assert_eq!((sampling.top_k, sampling.top_p, sampling.min_p), (4, 0.9, 0.1));
                let decoding = wire.decoding.unwrap();
                assert_eq!(
                    (decoding.presence_penalty, decoding.frequency_penalty, decoding.repetition_penalty),
                    (0.3, 0.4, 1.1),
                );
                let stopping = wire.stopping.unwrap();
                assert_eq!((stopping.max_new_tokens, stopping.min_new_tokens), (1, 1));
                assert_eq!(stopping.stop_strings, ["done"]);
                assert!(stopping.ignore_eos);
            }
        }
    };
    (responses) => {
        use super::*;
        use crate::unit_fixtures::minimal_request;
        use crate::unit_vllm_fixtures::*;
        use dynamo_backend_common::{FinishReason, StopReason};
        use serde_json::json;

        sidecar_test! {
            lane: pre_merge;
            #[test]
            fn encode_response_enforces_terminal_contract() {
                let request = epd_image_request();
                let ec_transfer_params = || json_to_struct(encoder_handoff()).expect("encoder handoff");

                let mut length = encode_response(Some(ec_transfer_params()));
                length
                    .outputs
                    .as_mut()
                    .and_then(|output| output.finish_info.as_mut())
                    .expect("finish info")
                    .finish_reason = pb::finish_info::FinishReason::Length as i32;
                let error = ResponseState::new(&request, DisaggregationMode::Encode)
                    .convert(length)
                    .expect_err("Length must not become a successful encoder handoff");
                assert!(error.to_string().contains("invalid finish reason"));

                let mut token_producing = encode_response(Some(ec_transfer_params()));
                let output = token_producing.outputs.as_mut().expect("sequence output");
                output.text = "unexpected".to_string();
                output.num_tokens = 1;
                output.token_ids = vec![42];
                output
                    .finish_info
                    .as_mut()
                    .expect("finish info")
                    .num_output_tokens = 1;
                let error = ResponseState::new(&request, DisaggregationMode::Encode)
                    .convert(token_producing)
                    .expect_err("Encode must remain tokenless");
                assert!(error.to_string().contains("produced output tokens"));

                let mut cancelled = encode_response(None);
                cancelled
                    .outputs
                    .as_mut()
                    .and_then(|output| output.finish_info.as_mut())
                    .expect("finish info")
                    .finish_reason = pb::finish_info::FinishReason::Aborted as i32;
                let terminal = ResponseState::new(&request, DisaggregationMode::Encode)
                    .convert(cancelled)
                    .expect("cancelled response")
                    .expect("cancelled terminal");
                assert_eq!(terminal.finish_reason, Some(FinishReason::Cancelled));
                assert!(terminal.encoder_result.is_none());
            }
        }

        sidecar_test! {
            lane: pre_merge;
            #[test]
            fn early_prompt_frames_retain_exact_native_metadata() {
                for (has_prompt_logprobs, has_output) in [(false, false), (true, false), (true, true)] {
                    let mut request = request();
                    request.output_options.prompt_logprobs = has_prompt_logprobs.then_some(1);
                    let mut state = ResponseState::new(&request, DisaggregationMode::Aggregated);
                    let prompt = pb::PromptInfo {
                        num_prompt_tokens: 3,
                        token_ids: vec![11, 22, 33],
                        logprobs: vec![f32::NEG_INFINITY, -0.25, -0.5],
                        ranks: vec![0, 1, 2],
                        candidate_tokens: vec![
                            pb::CandidateTokenInfo::default(),
                            pb::CandidateTokenInfo {
                                tokens: vec![pb::candidate_token_info::TokenInfo {
                                    id: 23,
                                    logprob: -0.75,
                                    rank: 2,
                                }],
                            },
                            pb::CandidateTokenInfo::default(),
                        ],
                    };
                    let first = state
                        .convert(pb::GenerateResponse {
                            prompt_info: Some(prompt),
                            outputs: has_output
                                .then(|| sequence_response(false, true, None).outputs.unwrap()),
                        })
                        .unwrap();
                    if has_output {
                        let first = first.unwrap();
                        assert!(first.finish_reason.is_none());
                        assert!(first.engine_data.is_none());
                    } else {
                        assert!(first.is_none());
                    }
                    let mut response = sequence_response(true, true, None);
                    response
                        .outputs
                        .as_mut()
                        .unwrap()
                        .finish_info
                        .as_mut()
                        .unwrap()
                        .num_output_tokens = 1 + u32::from(has_output);
                    let result = state.convert(response).unwrap().unwrap();
                    assert!(result.finish_reason.is_some());
                    assert_eq!(result.engine_data, has_prompt_logprobs.then(|| json!({"prompt_logprobs": [null, {"22": {"logprob": -0.25, "rank": 1}, "23": {"logprob": -0.75, "rank": 2}}, {"33": {"logprob": -0.5, "rank": 2}}]})));
                }
            }
        }

        sidecar_test! {
            lane: pre_merge;
            #[test]
            fn negative_infinity_logprobs_are_normalized() {
                let request = request();
                let mut state = ResponseState::new(&request, DisaggregationMode::Aggregated);
                let mut response = sequence_response(true, true, None);
                response.prompt_info = Some(pb::PromptInfo {
                    num_prompt_tokens: 3,
                    token_ids: vec![11, 22, 33],
                    logprobs: vec![0.0, f32::NEG_INFINITY, -0.3],
                    ranks: vec![0, 1, 2],
                    candidate_tokens: vec![
                        pb::CandidateTokenInfo::default(),
                        pb::CandidateTokenInfo {
                            tokens: vec![pb::candidate_token_info::TokenInfo {
                                id: 23,
                                logprob: f32::NEG_INFINITY,
                                rank: 2,
                            }],
                        },
                        pb::CandidateTokenInfo::default(),
                    ],
                });
                let output = response.outputs.as_mut().unwrap();
                output.logprobs[0] = f32::NEG_INFINITY;
                output.candidate_tokens[0].tokens[0].logprob = f32::NEG_INFINITY;

                let mapped = state
                    .convert(response)
                    .expect("convert response")
                    .expect("terminal output");
                assert_eq!(mapped.log_probs.as_deref(), Some(&[-9999.0][..]));
                assert!(
                    mapped.top_logprobs.as_ref().unwrap()[0]
                        .iter()
                        .all(|entry| entry.logprob == -9999.0)
                );
                let prompt = &mapped.engine_data.as_ref().unwrap()["prompt_logprobs"][1];
                assert_eq!(prompt["22"]["logprob"], json!(-9999.0));
                assert_eq!(prompt["23"]["logprob"], json!(-9999.0));
            }
        }

        sidecar_test! {
            lane: pre_merge;
            #[test]
            fn missing_outputs_and_buffered_text_preserve_native_semantics() {
                let mut request = request();
                request.output_options = Default::default();
                let mut state = ResponseState::new(&request, DisaggregationMode::Aggregated);
                assert!(
                    state
                        .convert(pb::GenerateResponse::default())
                        .unwrap()
                        .is_none()
                );
                let first = state
                    .convert(sequence_response(false, false, None))
                    .unwrap()
                    .unwrap();
                assert_eq!(first.text.as_deref(), Some(" token"));
                let mut response = sequence_response(true, false, None);
                let output = response.outputs.as_mut().unwrap();
                output.token_ids = vec![43, 44];
                output.num_tokens = 2;
                output.text = String::new();
                output.finish_info.as_mut().unwrap().num_output_tokens = 3;
                let terminal = state.convert(response).unwrap().unwrap();
                assert_eq!(terminal.text.as_deref(), Some(""));
            }
        }

        sidecar_test! {
            lane: pre_merge;
            #[test]
            fn malformed_responses_return_typed_protocol_errors() {
                type ResponseMutation = fn(&mut pb::GenerateResponse);
                let cases: &[(&str, ResponseMutation)] = &[
                    ("sequence index", |r| r.outputs.as_mut().unwrap().index = 1),
                    ("num_tokens", |r| r.outputs.as_mut().unwrap().num_tokens = 2),
                    ("terminal num_output_tokens", |r| {
                        r.outputs
                            .as_mut()
                            .unwrap()
                            .finish_info
                            .as_mut()
                            .unwrap()
                            .num_output_tokens = 2
                    }),
                    ("unknown finish reason", |r| {
                        r.outputs
                            .as_mut()
                            .unwrap()
                            .finish_info
                            .as_mut()
                            .unwrap()
                            .finish_reason = 42
                    }),
                    ("NOT_FINISHED", |r| {
                        r.outputs
                            .as_mut()
                            .unwrap()
                            .finish_info
                            .as_mut()
                            .unwrap()
                            .finish_reason = pb::finish_info::FinishReason::NotFinished as i32
                    }),
                    ("output logprob array lengths", |r| {
                        r.outputs.as_mut().unwrap().logprobs.clear()
                    }),
                    ("output logprob array lengths", |r| {
                        r.outputs.as_mut().unwrap().ranks.clear()
                    }),
                    ("output logprob array lengths", |r| {
                        r.outputs.as_mut().unwrap().candidate_tokens.clear()
                    }),
                    ("prompt token count", |r| {
                        r.prompt_info = Some(pb::PromptInfo {
                            num_prompt_tokens: 4,
                            ..Default::default()
                        })
                    }),
                    ("prompt logprob array lengths", |r| {
                        r.prompt_info = Some(pb::PromptInfo {
                            num_prompt_tokens: 3,
                            token_ids: vec![11, 22, 33],
                            logprobs: vec![0.0; 3],
                            ranks: vec![0; 3],
                            candidate_tokens: vec![],
                        })
                    }),
                ];
                for (message, mutate) in cases {
                    let mut response = sequence_response(true, true, None);
                    mutate(&mut response);
                    let error = ResponseState::new(&request(), DisaggregationMode::Aggregated)
                        .convert(response)
                        .expect_err(message);
                    assert_eq!(
                        error.error_type(),
                        dynamo_backend_common::ErrorType::Backend(
                            dynamo_backend_common::BackendError::Unknown
                        )
                    );
                    assert!(error.to_string().contains(message), "{message}: {error}");
                }
            }
        }

        sidecar_test! {
            lane: pre_merge;
            #[test]
            fn output_logprobs_preserve_native_ranks_and_token_metadata() {
                let mut request = request();
                request.output_options.logprobs = Some(2);
                let mut state = ResponseState::new(&request, DisaggregationMode::Aggregated);
                let mut response = sequence_response(true, true, None);
                let output = response.outputs.as_mut().unwrap();
                output.token_ids = vec![42, 50];
                output.num_tokens = 2;
                output.logprobs = vec![-0.25, -0.75];
                output.ranks = vec![1, 3];
                output.candidate_tokens.push(pb::CandidateTokenInfo {
                    tokens: vec![pb::candidate_token_info::TokenInfo {
                        id: 51,
                        logprob: -0.5,
                        rank: 1,
                    }],
                });
                output.finish_info.as_mut().unwrap().num_output_tokens = 2;
                let result = state.convert(response).unwrap().unwrap();
                assert_eq!(
                    result
                        .top_logprobs
                        .unwrap()
                        .into_iter()
                        .map(|row| {
                            row.into_iter()
                                .map(|entry| {
                                    assert!(entry.token.is_none());
                                    assert!(entry.bytes.is_none());
                                    (entry.token_id, entry.rank, entry.logprob)
                                })
                                .collect::<Vec<_>>()
                        })
                        .collect::<Vec<_>>(),
                    vec![
                        vec![(42, 1, -0.25), (43, 2, -0.5)],
                        vec![(50, 3, -0.75), (51, 1, -0.5)],
                    ]
                );
            }
        }

        sidecar_test! {
            lane: pre_merge;
            #[test]
            fn nonfinite_and_underflowing_logprobs_keep_associations() {
                for value in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY, -10000.0] {
                    let mut response = sequence_response(true, true, None);
                    let output = response.outputs.as_mut().unwrap();
                    output.logprobs[0] = value;
                    output.candidate_tokens[0].tokens[0].logprob = value;
                    let result = ResponseState::new(&request(), DisaggregationMode::Aggregated)
                        .convert(response)
                        .unwrap()
                        .unwrap();
                    assert_eq!(result.log_probs, Some(vec![-9999.0]));
                    let top = &result.top_logprobs.unwrap()[0];
                    assert_eq!(
                        top.iter()
                            .map(|t| (t.token_id, t.rank, t.logprob))
                            .collect::<Vec<_>>(),
                        vec![(42, 1, -9999.0), (43, 2, -9999.0)]
                    );
                }
            }
        }

        sidecar_test! {
            lane: pre_merge;
            #[test]
            fn prefill_handoff_is_required_only_for_successful_native_terminals() {
                for reason in [
                    pb::finish_info::FinishReason::Stop,
                    pb::finish_info::FinishReason::Length,
                    pb::finish_info::FinishReason::Aborted,
                ] {
                    for include_handoff in [false, true] {
                        let handoff = json!({"remote_port": 5600, "nested": {"ids": [1, 2], "ok": true}});
                        let mut response = sequence_response(
                            true,
                            false,
                            include_handoff.then(|| json_to_struct(handoff.clone()).unwrap()),
                        );
                        response
                            .outputs
                            .as_mut()
                            .unwrap()
                            .finish_info
                            .as_mut()
                            .unwrap()
                            .finish_reason = reason as i32;
                        let result =
                            ResponseState::new(&request(), DisaggregationMode::Prefill).convert(response);
                        if reason != pb::finish_info::FinishReason::Aborted && !include_handoff {
                            let error = result.unwrap_err();
                            assert!(error.to_string().contains("missing kv_transfer_params"));
                            continue;
                        }
                        let terminal = result.unwrap().unwrap();
                        if reason == pb::finish_info::FinishReason::Aborted {
                            assert_eq!(terminal.finish_reason, Some(FinishReason::Cancelled));
                            assert!(terminal.disaggregated_params.is_none());
                            if !include_handoff {
                                assert!(terminal.token_ids.is_empty());
                                assert!(terminal.text.is_none());
                                let usage = terminal.completion_usage.unwrap();
                                assert_eq!(
                                    (usage.prompt_tokens, usage.completion_tokens, usage.total_tokens),
                                    (3, 0, 3)
                                );
                            }
                        } else {
                            assert_eq!(terminal.disaggregated_params, Some(handoff));
                        }
                    }
                }
            }
        }

        sidecar_test! {
            lane: pre_merge;
            #[test]
            fn empty_chunks_and_delta_tokens_preserve_terminal_usage() {
                let request = minimal_request();
                let mut state = ResponseState::new(&request, DisaggregationMode::Aggregated);
                assert!(state.convert(pb::GenerateResponse {
                    outputs: Some(pb::SequenceOutput::default()),
                    ..Default::default()
                }).unwrap().is_none());
                let first = state.convert(sequence_response(false, false, None)).unwrap().unwrap();
                assert_eq!(first.token_ids, vec![42]);
                assert!(first.completion_usage.is_none());
                let mut response = sequence_response(true, false, None);
                let output = response.outputs.as_mut().unwrap();
                output.token_ids = vec![43, 44];
                output.num_tokens = 2;
                output.finish_info.as_mut().unwrap().num_output_tokens = 3;
                let terminal = state.convert(response).unwrap().unwrap();
                assert_eq!(terminal.token_ids, vec![43, 44]);
                assert_eq!(terminal.finish_reason, Some(FinishReason::Stop));
                assert!(terminal.disaggregated_params.is_none());
                let usage = terminal.completion_usage.unwrap();
                assert_eq!(
                    (usage.prompt_tokens, usage.completion_tokens, usage.total_tokens),
                    (3, 3, 6)
                );

                let decode = ResponseState::new(&request, DisaggregationMode::Decode)
                    .convert(sequence_response(true, false, None))
                    .unwrap()
                    .unwrap();
                assert!(decode.disaggregated_params.is_none());
            }
        }

        sidecar_test! {
            lane: pre_merge;
            #[test]
            fn output_logprobs_preserve_opt_in_alignment_and_values() {
                for requested in [None, Some(0), Some(2)] {
                    let mut request = minimal_request();
                    request.output_options.logprobs = requested;
                    let mut state = ResponseState::new(&request, DisaggregationMode::Aggregated);
                    let first = state.convert(sequence_response(false, true, None)).unwrap().unwrap();
                    assert_eq!(first.token_ids, vec![42]);
                    assert_eq!(first.log_probs, requested.map(|_| vec![-0.25]));
                    let mut response = sequence_response(true, true, None);
                    let output = response.outputs.as_mut().unwrap();
                    output.token_ids = vec![42, 50];
                    output.num_tokens = 2;
                    output.logprobs = vec![-0.25, -0.75];
                    output.ranks = vec![1, 1];
                    output.candidate_tokens.push(pb::CandidateTokenInfo {
                        tokens: vec![pb::candidate_token_info::TokenInfo {
                            id: 51,
                            logprob: -0.5,
                            rank: 2,
                        }],
                    });
                    output.finish_info.as_mut().unwrap().num_output_tokens = 3;
                    let result = state.convert(response).unwrap().unwrap();
                    assert_eq!(result.log_probs, requested.map(|_| vec![-0.25, -0.75]));
                    let expected = (requested == Some(2)).then_some(vec![
                        vec![(42, -0.25), (43, -0.5)],
                        vec![(50, -0.75), (51, -0.5)],
                    ]);
                    assert_eq!(
                        result.top_logprobs.map(|rows| rows
                            .into_iter()
                            .map(|row| {
                                row.into_iter()
                                    .map(|entry| (entry.token_id, entry.logprob))
                                    .collect::<Vec<_>>()
                            })
                            .collect::<Vec<_>>()),
                        expected
                    );
                }
            }
        }

        sidecar_test! {
            lane: pre_merge;
            #[test]
            fn user_stops_are_preserved_and_system_stops_are_hidden() {
                for (stop, explicit, expected) in [
                    (
                        pb::finish_info::StopReason::StopString("done".into()),
                        vec![],
                        Some(StopReason::String("done".into())),
                    ),
                    (pb::finish_info::StopReason::StopTokenId(42), vec![42], Some(StopReason::Int(42))),
                    (pb::finish_info::StopReason::StopTokenId(2), vec![], None),
                    (pb::finish_info::StopReason::EosTokenId(2), vec![], None),
                    (pb::finish_info::StopReason::EosTokenId(2), vec![2], Some(StopReason::Int(2))),
                    (pb::finish_info::StopReason::StopTokenId(2), vec![2], Some(StopReason::Int(2))),
                ] {
                    let mut request = minimal_request();
                    request.stop_conditions.stop_token_ids = Some(explicit);
                    request.stop_conditions.stop_token_ids_hidden = Some(vec![2]);
                    let mut response = sequence_response(true, false, None);
                    response.outputs.as_mut().unwrap().finish_info.as_mut().unwrap().stop_reason = Some(stop);
                    let result = ResponseState::new(&request, DisaggregationMode::Aggregated)
                        .convert(response)
                        .unwrap()
                        .unwrap();
                    assert_eq!(result.stop_reason, expected);
                }
            }
        }

        sidecar_test! {
            lane: pre_merge;
            #[test]
            fn prefill_terminal_has_zero_completion_usage() {
                for reason in [
                    FinishReason::Stop,
                    FinishReason::Length,
                    FinishReason::Cancelled,
                ] {
                    let mut response = terminal_response(reason.clone());
                    response.outputs.as_mut().unwrap().finish_info.as_mut().unwrap().kv_transfer_params = Some(
                        json_to_struct(json!({"remote_port": 5600, "nested": {"ids": [1, 2], "ok": true}})).unwrap()
                    );
                    let terminal = ResponseState::new(&minimal_request(), DisaggregationMode::Prefill)
                        .convert(response)
                        .unwrap()
                        .unwrap();
                    assert!(terminal.token_ids.is_empty());
                    assert!(terminal.text.is_none());
                    let usage = terminal.completion_usage.unwrap();
                    assert_eq!(
                        (usage.prompt_tokens, usage.completion_tokens, usage.total_tokens),
                        (3, 0, 3)
                    );
                    assert_eq!(terminal.finish_reason, Some(reason.clone()));
                    if reason == FinishReason::Cancelled {
                        assert!(terminal.disaggregated_params.is_none());
                    }
                }
            }
        }
    };
}
