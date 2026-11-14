// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use super::*;

#[tokio::test]
async fn native_responses_record_cached_token_metrics() {
    struct MetricsEngine(Arc<prometheus::Registry>, LLMEngineOutput);

    #[async_trait::async_trait]
    impl AsyncEngine<Input, Output, Error> for MetricsEngine {
        async fn generate(&self, request: Input) -> Result<Output, Error> {
            let families = self.0.gather();
            for (name, expected) in [
                ("dynamo_frontend_awaiting_first_token_requests", 1.0),
                ("dynamo_frontend_inflight_input_tokens", 3.0),
            ] {
                let family = families
                    .iter()
                    .find(|family| family.name() == name)
                    .unwrap();
                let value: f64 = family
                    .get_metric()
                    .iter()
                    .map(|metric| metric.get_gauge().value())
                    .sum();
                assert_eq!(value, expected, "{name} before engine output");
            }
            ReplayEngine(vec![Annotated::from_data(self.1.clone())], false)
                .generate(request)
                .await
        }
    }
    for stream in [false, true] {
        let state = state();
        let registry = Arc::new(prometheus::Registry::new());
        state.metrics_clone().register(&registry).unwrap();
        let output = dynamo_sglang_generate::inspect(
            serde_json::json!({"output_ids": [7], "meta_info": {
                "finish_reason": {"type": "length"}, "prompt_tokens": 3,
                "completion_tokens": 1, "cached_tokens": 2,
            }}),
            false,
        )
        .unwrap()
        .output;
        let response = respond(
            state,
            sglang(),
            Arc::new(MetricsEngine(registry.clone(), output)),
            serde_json::json!({"input_ids": [1, 2, 3], "stream": stream}),
            HeaderMap::new(),
        )
        .await;
        assert_eq!(response.status(), StatusCode::OK);
        axum::body::to_bytes(response.into_body(), usize::MAX)
            .await
            .unwrap();
        let families = registry.gather();
        let cached = families
            .iter()
            .find(|family| family.name() == "dynamo_frontend_cached_tokens")
            .unwrap();
        let histogram = cached.get_metric()[0].get_histogram();
        assert_eq!(histogram.get_sample_count(), 1);
        assert_eq!(histogram.get_sample_sum(), 2.0);
        for (name, expected) in [
            ("dynamo_frontend_input_tokens_total", 3.0),
            ("dynamo_frontend_output_tokens_total", 1.0),
        ] {
            let family = families
                .iter()
                .find(|family| family.name() == name)
                .unwrap();
            assert_eq!(
                family.get_metric()[0].get_counter().value(),
                expected,
                "{name}"
            );
        }
        for name in [
            "dynamo_frontend_awaiting_first_token_requests",
            "dynamo_frontend_inflight_input_tokens",
        ] {
            let family = families
                .iter()
                .find(|family| family.name() == name)
                .unwrap();
            assert!(
                family
                    .get_metric()
                    .iter()
                    .all(|metric| metric.get_gauge().value() == 0.0),
                "{name} after completion"
            );
        }
    }
}
