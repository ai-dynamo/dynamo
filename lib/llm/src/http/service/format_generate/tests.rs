// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use super::*;

#[test]
fn model_aliases_are_deduplicated_before_implicit_selection() {
    let manager = crate::discovery::ModelManager::new();
    assert!(manager.register_alias("alias", "primary"));
    let models = ["primary", "alias", "other"].map(String::from).to_vec();
    assert_eq!(
        canonical_generate_models(&manager, models),
        ["other", "primary"]
    );
}

#[test]
fn forwarded_router_metadata_reaches_the_tracker_and_leaves_the_payload() {
    let state = crate::http::service::service_v2::HttpService::builder()
        .build()
        .unwrap()
        .state_clone();
    let tracker = Arc::new(RequestTracker::new());
    let mut collector =
        GenerateMetricCollector::new(state.metrics_clone(), "model", tracker.clone(), 1);
    let mut output = Annotated::from_data(LLMEngineOutput {
        routing_data: Some(
            serde_json::from_value(serde_json::json!({
                "timing": {"request_received_ms": 0, "ttft_ms": 42.0},
                "worker_id": {"decode_worker_id": 7},
                "token_ids": [11],
            }))
            .unwrap(),
        ),
        ..Default::default()
    });
    collector.observe(&mut output);
    assert!(output.data.unwrap().routing_data.is_none());
    assert_eq!(tracker.get_timing_info().ttft_ms, Some(42.0));
    assert_eq!(tracker.get_worker_info().unwrap().decode_worker_id, Some(7));
    assert_eq!(tracker.query_token_ids(), Some([11].as_slice()));
}
