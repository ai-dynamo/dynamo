// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use super::*;
use crate::unit_fixtures::{model_info, request, server_info};
use clap::Parser;

fn args(mode: &str) -> Args {
    Args::try_parse_from([
        "sidecar",
        "--grpc-endpoint",
        "127.0.0.1:12345",
        "--namespace",
        "test-namespace",
        "--component",
        "configured-component",
        "--endpoint",
        "tokens",
        "--disaggregation-mode",
        mode,
        "--custom-jinja-template",
        "local-template.jinja",
    ])
    .unwrap()
}

#[test]
fn worker_config_preserves_options_and_discovers_model_identity() {
    for (mode, component) in [
        ("aggregated", "configured-component"),
        ("prefill", "prefill"),
        ("decode", "backend"),
        ("encode", "encode"),
    ] {
        let mut info = model_info();
        info.supports_multimodal = true;
        let model = DiscoveredModel::from_proto(info, server_info()).unwrap();
        let (_, config) = VllmSidecarEngine::from_discovered(args(mode), model).unwrap();
        assert_eq!(config.namespace, "test-namespace");
        assert_eq!(config.component, component);
        assert_eq!(config.endpoint, "tokens");
        assert_eq!(
            config.custom_jinja_template.as_deref(),
            Some(std::path::Path::new("local-template.jinja"))
        );
        assert_eq!(config.model_name, "model-source");
        assert_eq!(config.served_model_name.as_deref(), Some("served-model"));
        assert!(config.tool_call_parser.is_none());
        assert!(config.reasoning_parser.is_none());
        assert!(config.enable_kv_routing);
        assert_eq!(
            config.disaggregation_mode.as_str(),
            if mode == "aggregated" { "agg" } else { mode }
        );
    }
}

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

#[tokio::test]
async fn unstarted_generation_fails_and_cleanup_is_idempotent() {
    let model = DiscoveredModel::from_proto(model_info(), server_info()).unwrap();
    let (engine, _) = VllmSidecarEngine::from_discovered(args("aggregated"), model).unwrap();
    let context = || GenerateContext::new(dynamo_backend_common::testing::mock_context(), None);
    let error = engine
        .generate(request(), context())
        .await
        .err()
        .expect("unstarted engine");
    assert_eq!(
        error.error_type(),
        dynamo_backend_common::ErrorType::Backend(
            dynamo_backend_common::BackendError::EngineShutdown
        )
    );
    engine.cleanup().await.unwrap();
    engine.cleanup().await.unwrap();
    assert!(engine.cancel.is_cancelled());
}
