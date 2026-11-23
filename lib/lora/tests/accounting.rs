// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use dynamo_lora::LoadEstimator;
use dynamo_router_protocol::worker::WorkerWithDpRank;
use dynamo_slot_tracker::host::{ActiveSequenceEvent, ActiveSequenceEventData};
use std::{sync::Arc, time::Duration};

fn lora_event(
    request_id: &str,
    lora_name: Option<&str>,
    data: ActiveSequenceEventData,
) -> ActiveSequenceEvent {
    ActiveSequenceEvent {
        request_id: request_id.to_string(),
        worker: WorkerWithDpRank::new(1, 0),
        data,
        router_id: 7,
        lora_name: lora_name.map(str::to_string),
    }
}

fn add_request() -> ActiveSequenceEventData {
    ActiveSequenceEventData::AddRequest {
        token_sequence: None,
        track_prefill_tokens: false,
        expected_output_tokens: None,
        prefill_load_hint: None,
    }
}

#[tokio::test(start_paused = true)]
async fn excluded_sequence_events_do_not_allocate_or_publish_adapter_metrics() {
    use dynamo_lora::{
        LoraAllocationConfig, LoraController, LoraInfo, LoraRoutingTable, LoraStateTracker,
    };
    use prometheus::core::Collector;

    let legacy = "dyn-lora-legacy-alias";
    let estimator = Arc::new(LoadEstimator::new());
    for event in [
        lora_event("runtime", None, add_request()),
        lora_event("legacy", Some(legacy), add_request()),
        lora_event("runtime", None, ActiveSequenceEventData::Free),
        lora_event("runtime", None, add_request()),
        lora_event("runtime", None, ActiveSequenceEventData::Free),
    ] {
        estimator.handle_event(event);
    }
    assert_eq!(
        estimator.get_inflight_counts(),
        std::collections::HashMap::from([(legacy.to_string(), 1)])
    );
    assert_eq!(
        estimator.get_raw_arrival_counts(),
        std::collections::HashMap::from([(legacy.to_string(), 1)])
    );

    let tracker = LoraStateTracker::new();
    tracker.handle_mdc_addition(
        WorkerWithDpRank::new(1, 0),
        &LoraInfo {
            name: legacy.into(),
            max_gpu_lora_count: Some(1),
        },
    );
    let routes = LoraRoutingTable::new();
    let metrics = dynamo_lora::metrics::ControllerMetrics::default();
    let cancel = tokio_util::sync::CancellationToken::new();
    let controller = LoraController::start_for_endpoint_with_metrics(
        dynamo_runtime::protocols::EndpointId::from("test.accounting.generate"),
        LoraAllocationConfig::default(),
        routes.clone(),
        tracker,
        estimator.clone(),
        cancel.clone(),
        metrics.clone(),
    );
    tokio::time::timeout(Duration::from_secs(1), async {
        while routes.is_empty() {
            tokio::time::sleep(Duration::from_millis(1)).await;
        }
    })
    .await
    .expect("controller must allocate the legacy adapter");
    cancel.cancel();
    controller.await.unwrap();
    assert_eq!(routes.list_loras(), vec![legacy.to_string()]);
    assert_eq!(routes.get_config(legacy).unwrap().replica_factor, 1);
    for gauge in [
        &metrics.replica_factor,
        &metrics.is_active,
        &metrics.raw_arrival_count,
        &metrics.estimated_load,
        &metrics.active_requests,
    ] {
        let names = gauge
            .collect()
            .iter()
            .flat_map(|family| family.get_metric())
            .flat_map(|metric| metric.get_label())
            .filter(|label| label.name() == "lora")
            .map(|label| label.value().to_string())
            .collect::<Vec<_>>();
        assert_eq!(names, vec![legacy.to_string()]);
    }
    estimator.handle_event(lora_event(
        "legacy",
        Some(legacy),
        ActiveSequenceEventData::Free,
    ));
    assert!(estimator.get_inflight_counts().is_empty());
}
