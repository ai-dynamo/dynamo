// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use super::adapter;
use crate::unit_fixtures::minimal_request;
use dynamo_backend_common::{BackendError, ErrorType, GenerateContext, LLMEngine, WorkerConfig};

pub(super) fn assert_worker_options(config: &WorkerConfig, mode: &str, component: &str) {
    assert_eq!(config.namespace, "test-namespace");
    assert_eq!(config.component, component);
    assert_eq!(config.endpoint, "tokens");
    assert_eq!(
        config.custom_jinja_template.as_deref(),
        Some(std::path::Path::new("local-template.jinja"))
    );
    assert_eq!(config.model_name, "model-source");
    assert_eq!(config.served_model_name.as_deref(), Some("served-model"));
    assert!(config.enable_kv_routing);
    assert_eq!(
        config.disaggregation_mode.as_str(),
        if mode == "aggregated" { "agg" } else { mode }
    );
}

sidecar_test! {
    lane: pre_merge;
    #[test]
    fn worker_options_and_model_identity_are_preserved() {
        for (mode, component) in [
            ("aggregated", "configured-component"),
            ("prefill", "prefill"),
            ("decode", "backend"),
        ] {
            let (_, config) = adapter::worker(mode);
            assert_worker_options(&config, mode, component);
        }
    }
}

sidecar_test! {
    lane: pre_merge;
    #[tokio::test]
    async fn unstarted_generation_fails_and_cleanup_is_idempotent() {
        let (engine, _) = adapter::worker("aggregated");
        let context = GenerateContext::new(dynamo_backend_common::testing::mock_context(), None);
        let error = engine
            .generate(minimal_request(), context)
            .await
            .err()
            .expect("unstarted engine");
        assert_eq!(error.error_type(), ErrorType::Backend(BackendError::EngineShutdown));
        engine.cleanup().await.unwrap();
        engine.cleanup().await.unwrap();
        assert!(adapter::is_cancelled(&engine));
    }
}
