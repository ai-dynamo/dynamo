// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use super::*;
use crate::local_model::runtime_config::{
    StructuralTagConfig, StructuralTagSchemaMode, StructuralTagScope,
};

#[test]
fn architectural_context_prefers_config_then_text_config_then_tokenizer() -> anyhow::Result<()> {
    let dir = tempfile::tempdir()?;

    std::fs::write(
        dir.path().join("config.json"),
        r#"{"max_position_embeddings": 4096}"#,
    )?;
    std::fs::write(
        dir.path().join("tokenizer_config.json"),
        r#"{"model_max_length": 131072}"#,
    )?;
    assert_eq!(
        architectural_max_context_length_from_repo(dir.path())?,
        Some(4096)
    );

    std::fs::write(
        dir.path().join("config.json"),
        r#"{"text_config": {"max_position_embeddings": 8192}}"#,
    )?;
    assert_eq!(
        architectural_max_context_length_from_repo(dir.path())?,
        Some(8192)
    );

    std::fs::write(dir.path().join("config.json"), r#"{"model_type": "mock"}"#)?;
    assert_eq!(
        architectural_max_context_length_from_repo(dir.path())?,
        Some(131072)
    );

    std::fs::write(dir.path().join("config.json"), r#"{"text_config": {}}"#)?;
    assert_eq!(
        architectural_max_context_length_from_repo(dir.path())?,
        Some(131072)
    );

    std::fs::remove_file(dir.path().join("config.json"))?;
    assert_eq!(
        architectural_max_context_length_from_repo(dir.path())?,
        Some(131072)
    );

    Ok(())
}

#[test]
fn architectural_context_errors_on_malformed_present_fields() -> anyhow::Result<()> {
    let dir = tempfile::tempdir()?;
    std::fs::write(
        dir.path().join("tokenizer_config.json"),
        r#"{"model_max_length": 131072}"#,
    )?;

    std::fs::write(
        dir.path().join("config.json"),
        r#"{"max_position_embeddings": "not-a-number"}"#,
    )?;
    let err = architectural_max_context_length_from_repo(dir.path()).unwrap_err();
    assert!(
        err.to_string()
            .contains("Failed to deserialize max_position_embeddings"),
        "{err:?}"
    );

    std::fs::write(
        dir.path().join("config.json"),
        r#"{"text_config": {"max_position_embeddings": "not-a-number"}}"#,
    )?;
    let err = architectural_max_context_length_from_repo(dir.path()).unwrap_err();
    assert!(
        err.to_string()
            .contains("Failed to deserialize text_config.max_position_embeddings"),
        "{err:?}"
    );

    // An explicit null is a present-but-malformed value: it must error, not
    // silently fall through to text_config/tokenizer (regression guard for
    // the Option<Value> projection collapsing null into None).
    std::fs::write(
        dir.path().join("config.json"),
        r#"{"max_position_embeddings": null}"#,
    )?;
    let err = architectural_max_context_length_from_repo(dir.path()).unwrap_err();
    assert!(
        err.to_string()
            .contains("Failed to deserialize max_position_embeddings"),
        "{err:?}"
    );

    std::fs::write(
        dir.path().join("config.json"),
        r#"{"text_config": {"max_position_embeddings": null}}"#,
    )?;
    let err = architectural_max_context_length_from_repo(dir.path()).unwrap_err();
    assert!(
        err.to_string()
            .contains("Failed to deserialize text_config.max_position_embeddings"),
        "{err:?}"
    );

    Ok(())
}

#[test]
fn architectural_context_tolerates_non_finite_literals() -> anyhow::Result<()> {
    // Some HF configs (e.g. Nemotron-H) serialize fields such as
    // `time_step_limit` with the bare literal `Infinity`, which is valid
    // JSON5 but not strict JSON. Deriving max_position_embeddings must not
    // choke on such literals in fields we never read.
    let dir = tempfile::tempdir()?;
    std::fs::write(
        dir.path().join("config.json"),
        r#"{"max_position_embeddings": 262144, "time_step_limit": [0.0, Infinity]}"#,
    )?;
    assert_eq!(
        architectural_max_context_length_from_repo(dir.path())?,
        Some(262144)
    );

    Ok(())
}

#[test]
fn effective_context_prefers_runtime_then_architecture_then_unknown() {
    let mut card = ModelDeploymentCard::with_name_only("model");
    assert_eq!(card.effective_context_length(), 0);

    card.architectural_max_context_length = Some(32_768);
    assert_eq!(card.effective_context_length(), 32_768);

    card.runtime_config.context_length = Some(8_192);
    assert_eq!(card.effective_context_length(), 8_192);

    card.runtime_config.context_length = Some(0);
    assert_eq!(card.effective_context_length(), 0);
}

#[test]
fn prefill_load_model_wire_aliases_preserve_card_and_mdcsum() {
    use crate::entrypoint::RouterConfig;
    use dynamo_kv_router::scheduling::config::RouterPrefillLoadModel;
    use dynamo_runtime::pipeline::RouterMode;

    for mode in [RouterMode::KV, RouterMode::RoundRobin] {
        let mut card = ModelDeploymentCard::with_name_only("wire-compat-model");
        let mut router = RouterConfig {
            router_mode: mode,
            ..Default::default()
        };
        router.kv_router_config.router_prefill_load_model = RouterPrefillLoadModel::Ais;
        card.router_config = Some(router);
        let emitted: serde_json::Value = serde_json::from_str(&card.to_json().unwrap()).unwrap();
        assert_eq!(
            emitted["router_config"]["kv_router_config"]["router_prefill_load_model"],
            "aic"
        );

        for spelling in ["aic", "ais"] {
            let mut input = emitted.clone();
            input["router_config"]["kv_router_config"]["router_prefill_load_model"] =
                serde_json::json!(spelling);
            let parsed = ModelDeploymentCard::load_from_json_str(&input.to_string()).unwrap();
            assert_eq!(
                parsed
                    .router_config
                    .as_ref()
                    .unwrap()
                    .kv_router_config
                    .router_prefill_load_model,
                RouterPrefillLoadModel::Ais
            );
            assert_eq!(parsed.mdcsum(), card.mdcsum());
            assert_eq!(
                serde_json::from_str::<serde_json::Value>(&parsed.to_json().unwrap()).unwrap(),
                emitted
            );
        }
    }

    let card = ModelDeploymentCard::with_name_only("wire-compat-default");
    let parsed = ModelDeploymentCard::load_from_json_str(&card.to_json().unwrap()).unwrap();
    assert!(parsed.router_config.is_none());
    assert_eq!(parsed.mdcsum(), card.mdcsum());
}

#[test]
fn context_length_wire_compatibility() {
    let card = ModelDeploymentCard::with_name_only("model");
    let mut legacy_value = serde_json::to_value(&card).unwrap();
    legacy_value["context_length"] = serde_json::json!(32_768);

    let mut parsed: ModelDeploymentCard = serde_json::from_value(legacy_value).unwrap();
    assert_eq!(parsed.effective_context_length(), 32_768);

    parsed.runtime_config.context_length = Some(8_192);
    let wire_value: serde_json::Value = serde_json::from_str(&parsed.to_json().unwrap()).unwrap();
    assert_eq!(wire_value["context_length"], 8_192);
}

#[test]
fn structural_tag_wire_includes_fields_for_older_frontends() {
    let mut card = ModelDeploymentCard::with_name_only("model");
    card.runtime_config.structural_tag = Some(StructuralTagConfig {
        scope: StructuralTagScope::Always,
        schema: StructuralTagSchemaMode::Strict,
        ..Default::default()
    });

    let wire: serde_json::Value = serde_json::from_str(&card.to_json().unwrap()).unwrap();
    let runtime = &wire["runtime_config"];
    assert_eq!(runtime["structural_tag_mode"], "on");
    assert_eq!(runtime["structural_tag_scope"], "always");
    assert_eq!(runtime["structural_tag_schema"], "strict");
    assert_eq!(runtime["structural_tag"]["scope"], "always");
    assert_eq!(runtime["structural_tag"]["schema"], "strict");
}

#[test]
fn tensor_config_serializes_at_card_top_level() {
    let mut card = ModelDeploymentCard::with_name_only("tensor");
    card.tensor_model_config = Some(TensorModelConfig {
        name: "tensor".to_string(),
        ..Default::default()
    });

    let value = serde_json::to_value(&card).unwrap();
    assert_eq!(value["tensor_model_config"]["name"], "tensor");
    assert!(value["runtime_config"].get("tensor_model_config").is_none());

    let parsed: ModelDeploymentCard = serde_json::from_value(value).unwrap();
    assert_eq!(
        parsed
            .tensor_model_config
            .as_ref()
            .map(|config| config.name.as_str()),
        Some("tensor")
    );
}

#[test]
fn runtime_taints_do_not_change_mdcsum() {
    let mut fast = ModelDeploymentCard::with_name_only("model");
    fast.runtime_config.taints = std::collections::HashSet::from(["fast".to_string()]);

    let mut slow = ModelDeploymentCard::with_name_only("model");
    slow.runtime_config.taints = std::collections::HashSet::from(["slow".to_string()]);

    assert_eq!(fast.mdcsum(), slow.mdcsum());
}

#[test]
fn runtime_kv_event_capability_does_not_change_mdcsum() {
    let mut disabled = ModelDeploymentCard::with_name_only("model");
    disabled.runtime_config.kv_event_publishing_enabled = Some(false);

    let mut enabled = ModelDeploymentCard::with_name_only("model");
    enabled.runtime_config.kv_event_publishing_enabled = Some(true);

    assert_eq!(disabled.mdcsum(), enabled.mdcsum());
}

#[test]
fn tower_connector_lora_runtime_flag_isolates_worker_sets() {
    use crate::local_model::runtime_config::VLLM_ENABLE_TOWER_CONNECTOR_LORA_RUNTIME_KEY;

    let missing = ModelDeploymentCard::with_name_only("model");
    let mut disabled = ModelDeploymentCard::with_name_only("model");
    disabled.runtime_config.runtime_data.insert(
        VLLM_ENABLE_TOWER_CONNECTOR_LORA_RUNTIME_KEY.to_string(),
        false.into(),
    );
    let mut enabled = ModelDeploymentCard::with_name_only("model");
    enabled.runtime_config.runtime_data.insert(
        VLLM_ENABLE_TOWER_CONNECTOR_LORA_RUNTIME_KEY.to_string(),
        true.into(),
    );

    assert_eq!(missing.mdcsum(), disabled.mdcsum());
    assert_ne!(missing.mdcsum(), enabled.mdcsum());
}

#[test]
fn vllm_generate_capability_isolates_worker_sets() {
    use crate::local_model::runtime_config::VLLM_INFERENCE_V1_GENERATE_CAPABILITY;

    let missing = ModelDeploymentCard::with_name_only("model");
    let mut disabled = ModelDeploymentCard::with_name_only("model");
    disabled.runtime_config.runtime_data.insert(
        VLLM_INFERENCE_V1_GENERATE_CAPABILITY.to_string(),
        false.into(),
    );
    let mut enabled = ModelDeploymentCard::with_name_only("model");
    enabled.runtime_config.runtime_data.insert(
        VLLM_INFERENCE_V1_GENERATE_CAPABILITY.to_string(),
        true.into(),
    );

    assert_eq!(missing.mdcsum(), disabled.mdcsum());
    assert_ne!(missing.mdcsum(), enabled.mdcsum());
}

#[test]
fn native_decision_profiles_require_unique_eligible_revision_bound_cards() {
    use crate::worker_type::WorkerType;
    fn card(profile: DecisionExecutionProfile) -> ModelDeploymentCard {
        let mut card = ModelDeploymentCard::with_name_only("native-model");
        card.model_input = if matches!(
            profile.kind(),
            crate::decision_profile::DecisionProfileKind::Native(
                crate::decision_profile::NativeExecutorKind::Head
            )
        ) {
            ModelInput::Text
        } else {
            ModelInput::Tokens
        };
        card.model_type = if matches!(
            profile.kind(),
            crate::decision_profile::DecisionProfileKind::Native(
                crate::decision_profile::NativeExecutorKind::Head
            )
        ) {
            ModelType::Classify | ModelType::Pooling
        } else {
            ModelType::Chat
        };
        card.worker_type = Some(WorkerType::Aggregated);
        card.runtime_config.data_parallel_size = 1;
        card.runtime_config
            .runtime_data
            .insert(profile.decision_capability().into(), true.into());
        card.runtime_config.runtime_data.insert(
            crate::decision_profile::VLLM_NATIVE_DECISION_MAX_CANDIDATES.into(),
            profile.limits().1.into(),
        );
        card
    }
    for profile in [
        DecisionExecutionProfile::VllmD3V1,
        DecisionExecutionProfile::VllmClefFlashV1,
        DecisionExecutionProfile::VllmClefOmniV1,
        DecisionExecutionProfile::VllmLiquidV1,
        DecisionExecutionProfile::VllmDiffusionV1,
    ] {
        let enabled = card(profile);
        assert_eq!(enabled.decision_execution_profile(), Some(profile));
        let mut public_migration = card(profile);
        public_migration.migration_limit = 1;
        assert_eq!(public_migration.decision_execution_profile(), Some(profile));
        let mut limited = enabled.clone();
        limited.runtime_config.runtime_data.insert(
            crate::decision_profile::VLLM_NATIVE_DECISION_MAX_CANDIDATES.into(),
            3.into(),
        );
        assert_eq!(limited.decision_max_candidates(profile), Some(3));
        assert_eq!(limited.mdcsum(), enabled.mdcsum());
        let mut missing_limit = enabled.clone();
        missing_limit
            .runtime_config
            .runtime_data
            .remove(crate::decision_profile::VLLM_NATIVE_DECISION_MAX_CANDIDATES);
        assert!(missing_limit.decision_execution_profile().is_none());
        for value in [
            serde_json::Value::Null,
            true.into(),
            "3".into(),
            3.5.into(),
            1.into(),
            256.into(),
        ] {
            let mut invalid = enabled.clone();
            invalid.runtime_config.runtime_data.insert(
                crate::decision_profile::VLLM_NATIVE_DECISION_MAX_CANDIDATES.into(),
                value,
            );
            assert!(invalid.decision_execution_profile().is_none());
        }
        let mut missing = card(profile);
        missing
            .runtime_config
            .runtime_data
            .remove(profile.decision_capability());
        let mut disabled = card(profile);
        disabled
            .runtime_config
            .runtime_data
            .insert(profile.decision_capability().into(), false.into());
        assert_eq!(missing.mdcsum(), disabled.mdcsum());
        assert_eq!(enabled.mdcsum(), missing.mdcsum());
        let mut ambiguous = card(profile);
        let other = if profile == DecisionExecutionProfile::VllmLiquidV1 {
            DecisionExecutionProfile::VllmDiffusionV1
        } else {
            DecisionExecutionProfile::VllmLiquidV1
        };
        ambiguous
            .runtime_config
            .runtime_data
            .insert(other.decision_capability().into(), true.into());
        assert_eq!(ambiguous.decision_execution_profile(), None);
        let mut mixed = card(profile);
        mixed.runtime_config.runtime_data.insert(
            crate::decision_profile::VLLM_CAUSAL_DECISION_V1.into(),
            true.into(),
        );
        assert_eq!(mixed.decision_execution_profile(), None);
        assert_eq!(mixed.mdcsum(), missing.mdcsum());
        for mutate in [
            (|c: &mut ModelDeploymentCard| c.worker_type = Some(WorkerType::Decode))
                as fn(&mut ModelDeploymentCard),
            |c| c.runtime_config.data_parallel_size = 2,
            |c| c.needs = vec![vec![WorkerType::Prefill]],
            |c| {
                c.runtime_config
                    .runtime_data
                    .insert("spec_decode".into(), true.into());
            },
        ] {
            let mut invalid = card(profile);
            mutate(&mut invalid);
            assert_eq!(invalid.decision_execution_profile(), None);
        }
    }
}

#[test]
fn vllm_decision_contract_requires_eligible_aggregated_workers() {
    fn card(configure: impl FnOnce(&mut ModelDeploymentCard)) -> ModelDeploymentCard {
        let mut card = ModelDeploymentCard::with_name_only("model");
        card.model_input = ModelInput::Tokens;
        card.model_type = ModelType::Chat;
        card.worker_type = Some(crate::worker_type::WorkerType::Aggregated);
        card.runtime_config.max_num_seqs = Some(1);
        card.runtime_config
            .runtime_data
            .insert("vllm_inference_v1_generate".into(), true.into());
        card.runtime_config
            .runtime_data
            .insert("vllm_causal_decision_v1".into(), true.into());
        configure(&mut card);
        card
    }

    let enabled = card(|_| {});
    assert_eq!(
        enabled.decision_execution_profile(),
        Some(crate::decision_profile::DecisionExecutionProfile::VllmCausalV1),
    );
    let disabled = card(|card| {
        card.runtime_config
            .runtime_data
            .remove("vllm_causal_decision_v1");
    });
    let explicit_false = card(|card| {
        card.runtime_config
            .runtime_data
            .insert("vllm_causal_decision_v1".into(), false.into());
    });
    assert_eq!(disabled.mdcsum(), explicit_false.mdcsum());
    assert_eq!(enabled.mdcsum(), disabled.mdcsum());

    assert!(
        card(|card| card.migration_limit = 1)
            .decision_execution_profile()
            .is_some()
    );

    for model_type in [
        ModelType::Completions,
        ModelType::Completions | ModelType::Embedding,
    ] {
        assert!(
            card(|card| card.model_type = model_type)
                .decision_execution_profile()
                .is_some()
        );
    }
    let unsupported: [fn(&mut ModelDeploymentCard); 9] = [
        |card| card.model_type = ModelType::Embedding,
        |card| card.model_type = ModelType::Classify | ModelType::Pooling,
        |card| card.runtime_config.max_num_seqs = None,
        |card| card.runtime_config.max_num_seqs = Some(16),
        |card| card.runtime_config.data_parallel_size = 2,
        |card| {
            card.runtime_config
                .runtime_data
                .remove("vllm_inference_v1_generate");
        },
        |card| card.worker_type = Some(crate::worker_type::WorkerType::Decode),
        |card| card.needs = vec![vec![crate::worker_type::WorkerType::Prefill]],
        |card| {
            card.runtime_config
                .runtime_data
                .insert("spec_decode".into(), serde_json::json!({}));
        },
    ];
    for configure in unsupported {
        assert_eq!(card(configure).decision_execution_profile(), None);
    }
}

#[test]
fn systemone_serial_contract_does_not_partition_ordinary_worker_sets() {
    use crate::decision_profile::SGLANG_SYSTEMONE_SERIAL_V1;
    use crate::local_model::runtime_config::SGLANG_GENERATE_CAPABILITY;

    fn card(configure: impl FnOnce(&mut ModelDeploymentCard)) -> ModelDeploymentCard {
        let mut card = ModelDeploymentCard::with_name_only("model");
        card.model_input = ModelInput::Tokens;
        card.model_type = ModelType::Chat;
        card.worker_type = Some(crate::worker_type::WorkerType::Aggregated);
        card.runtime_config.max_num_seqs = Some(1);
        card.runtime_config
            .runtime_data
            .insert(SGLANG_GENERATE_CAPABILITY.to_string(), true.into());
        card.runtime_config
            .runtime_data
            .insert(SGLANG_SYSTEMONE_SERIAL_V1.to_string(), true.into());
        configure(&mut card);
        card
    }

    let enabled = card(|_| {});
    assert_eq!(
        enabled.decision_execution_profile(),
        Some(DecisionExecutionProfile::SglangSerialV1)
    );
    assert_eq!(
        card(|card| card.migration_limit = 1).decision_execution_profile(),
        Some(DecisionExecutionProfile::SglangSerialV1)
    );
    let unsupported: [fn(&mut ModelDeploymentCard); 10] = [
        |card| {
            card.runtime_config
                .runtime_data
                .remove(SGLANG_SYSTEMONE_SERIAL_V1);
        },
        |card| card.runtime_config.max_num_seqs = None,
        |card| card.runtime_config.max_num_seqs = Some(2),
        |card| card.runtime_config.data_parallel_size = 0,
        |card| card.runtime_config.data_parallel_size = 2,
        |card| {
            card.runtime_config
                .runtime_data
                .remove(SGLANG_GENERATE_CAPABILITY);
        },
        |card| card.worker_type = Some(crate::worker_type::WorkerType::Prefill),
        |card| card.needs = vec![vec![crate::worker_type::WorkerType::Prefill]],
        |card| {
            card.lora = Some(LoraInfo {
                name: "adapter".to_string(),
                max_gpu_lora_count: None,
            });
        },
        |card| {
            card.runtime_config
                .runtime_data
                .insert("spec_decode".to_string(), serde_json::json!({}));
        },
    ];
    for configure in unsupported {
        let unsupported = card(configure);
        assert_ne!(
            unsupported.decision_execution_profile(),
            Some(DecisionExecutionProfile::SglangSerialV1)
        );
        let feature_off = card(|card| {
            configure(card);
            card.runtime_config
                .runtime_data
                .remove(SGLANG_SYSTEMONE_SERIAL_V1);
        });
        assert_eq!(feature_off.mdcsum(), unsupported.mdcsum());
    }
}

#[test]
fn qwen_video_processor_contract_stays_out_of_the_checksum() {
    use crate::local_model::runtime_config::VLLM_QWEN_VIDEO_PROCESSOR_CONTRACT_RUNTIME_KEY;

    // `mdcsum()` caches via `OnceLock`, so each case uses a fresh card.

    fn card_with_contract(contract: serde_json::Value) -> ModelDeploymentCard {
        let mut card = ModelDeploymentCard::with_name_only("model");
        card.runtime_config.runtime_data.insert(
            VLLM_QWEN_VIDEO_PROCESSOR_CONTRACT_RUNTIME_KEY.to_string(),
            contract,
        );
        card
    }

    fn legacy_ceil() -> serde_json::Value {
        serde_json::json!({
            "placeholder_target": "bare_video_token",
            "resize_mode": "legacy_ceil",
        })
    }

    let withheld = ModelDeploymentCard::with_name_only("model");
    assert_eq!(
        withheld.mdcsum(),
        card_with_contract(legacy_ceil()).mdcsum(),
        "a worker that predates the contract must still join the WorkerSet of one that publishes it"
    );

    assert_eq!(
        card_with_contract(legacy_ceil()).mdcsum(),
        card_with_contract(serde_json::json!({
            "placeholder_target": "bare_video_token",
            "resize_mode": "round_ties_even",
        }))
        .mdcsum(),
        "two workers publishing different contracts must still serve as one group"
    );

    // Negative control: a deployment that never carries this key is
    // unaffected, so its cache directory and reported checksum do not move.
    let bare = ModelDeploymentCard::with_name_only("model");
    let mut unrelated = ModelDeploymentCard::with_name_only("model");
    unrelated
        .runtime_config
        .runtime_data
        .insert("some_unrelated_runtime_key".to_string(), true.into());
    assert_eq!(bare.mdcsum(), unrelated.mdcsum());
}

#[test]
fn video_processor_runtime_contract_checksum_boundaries() {
    use crate::local_model::runtime_config::{
        SGLANG_QWEN_VIDEO_PROCESSOR_CONTRACT_RUNTIME_KEY,
        VLLM_NEMOTRON_VIDEO_PROCESSOR_CONTRACT_RUNTIME_KEY,
        VLLM_QWEN_VIDEO_PROCESSOR_CONTRACT_RUNTIME_KEY,
    };

    fn card_with_contract(key: &str, value: serde_json::Value) -> ModelDeploymentCard {
        let mut card = ModelDeploymentCard::with_name_only("model");
        card.runtime_config
            .runtime_data
            .insert(key.to_string(), value);
        card
    }

    let missing = ModelDeploymentCard::with_name_only("model");
    let qwen = card_with_contract(
        VLLM_QWEN_VIDEO_PROCESSOR_CONTRACT_RUNTIME_KEY,
        serde_json::json!({
            "placeholder_target": "bare_video_token",
            "resize_mode": "round_ties_even"
        }),
    );
    let same_qwen = card_with_contract(
        VLLM_QWEN_VIDEO_PROCESSOR_CONTRACT_RUNTIME_KEY,
        serde_json::json!({
            "resize_mode": "round_ties_even",
            "placeholder_target": "bare_video_token"
        }),
    );
    let different_qwen = card_with_contract(
        VLLM_QWEN_VIDEO_PROCESSOR_CONTRACT_RUNTIME_KEY,
        serde_json::json!({
            "placeholder_target": "vision_wrapped_video_token",
            "resize_mode": "round_ties_even"
        }),
    );
    let sglang_qwen = card_with_contract(
        SGLANG_QWEN_VIDEO_PROCESSOR_CONTRACT_RUNTIME_KEY,
        serde_json::json!({
            "placeholder_target": "bare_video_token",
            "resize_mode": "legacy_ceil",
            "sglang_preprocess": {
                "image_factor": 28,
                "video_min_pixels": 100352,
                "video_max_pixels": 602112,
                "video_total_pixels": 90316800,
                "frame_factor": 2,
                "fps": 2.0,
                "min_frames": 4,
                "max_frames": 768
            }
        }),
    );
    let nemotron = card_with_contract(
        VLLM_NEMOTRON_VIDEO_PROCESSOR_CONTRACT_RUNTIME_KEY,
        serde_json::json!({
            "video_pruning_rate": 0.5,
            "sampling": {"max_frames": 64, "fps": 2.0, "frames": [{"z": 1, "a": 2}, {"y": [3, 1, 2]}]},
            "enabled": true,
            "note": null,
        }),
    );
    let same_nemotron = card_with_contract(
        VLLM_NEMOTRON_VIDEO_PROCESSOR_CONTRACT_RUNTIME_KEY,
        serde_json::json!({
            "note": null,
            "enabled": true,
            "sampling": {"frames": [{"a": 2, "z": 1}, {"y": [3, 1, 2]}], "fps": 2.0, "max_frames": 64},
            "video_pruning_rate": 0.5,
        }),
    );
    let different_nemotron = card_with_contract(
        VLLM_NEMOTRON_VIDEO_PROCESSOR_CONTRACT_RUNTIME_KEY,
        serde_json::json!({
            "video_pruning_rate": 0.5,
            "sampling": {"max_frames": 64, "fps": 2.0, "frames": [{"z": 1, "a": 3}, {"y": [3, 1, 2]}]},
            "enabled": true,
            "note": null,
        }),
    );

    let unrelated = card_with_contract("unrelated_runtime_metadata", serde_json::json!(true));

    assert_eq!(missing.mdcsum(), unrelated.mdcsum());
    assert_eq!(missing.mdcsum(), qwen.mdcsum());
    assert_eq!(missing.mdcsum(), sglang_qwen.mdcsum());
    assert_eq!(qwen.mdcsum(), sglang_qwen.mdcsum());
    assert_eq!(qwen.mdcsum(), same_qwen.mdcsum());
    assert_eq!(qwen.mdcsum(), different_qwen.mdcsum());
    assert_ne!(missing.mdcsum(), nemotron.mdcsum());
    assert_ne!(nemotron.mdcsum(), different_nemotron.mdcsum());
    assert_ne!(qwen.mdcsum(), nemotron.mdcsum());
    assert_eq!(nemotron.mdcsum(), same_nemotron.mdcsum());
    // Pins the separator, length framing, and canonical bytes of the contract.
    assert_eq!(
        nemotron.mdcsum(),
        "88381ff5b5f10389bacc63e1e2bac947a76ffc5b32c4864b7499f0d10928a96e"
    );
}
