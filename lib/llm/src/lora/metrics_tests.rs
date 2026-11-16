// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::{sync::Arc, time::Duration};

use dynamo_router_protocol::worker::WorkerWithDpRank;
use dynamo_runtime::protocols::EndpointId;

use super::*;
use crate::http::service::metrics::{LORA_ACTIVE_REQUESTS_GAUGE, LORA_REPLICA_FACTOR_GAUGE};

#[tokio::test(start_paused = true)]
async fn controller_publishes_into_the_frontend_registry_for_its_endpoint() {
    let endpoint = EndpointId {
        namespace: "lora-metrics-test".into(),
        component: "worker".into(),
        name: uuid::Uuid::new_v4().to_string(),
    };
    let tracker = LoraStateTracker::new();
    tracker.handle_mdc_addition(
        WorkerWithDpRank::new(1, 0),
        &LoraInfo {
            name: "adapter".into(),
            max_gpu_lora_count: Some(4),
        },
    );
    let estimator = Arc::new(LoadEstimator::new());
    estimator.increment_load("adapter");
    let metrics = controller_metrics();
    let cancel = tokio_util::sync::CancellationToken::new();
    let handle = LoraController::start_for_endpoint_with_metrics(
        endpoint.clone(),
        LoraAllocationConfig::default(),
        LoraRoutingTable::new(),
        tracker,
        estimator,
        cancel.clone(),
        metrics.clone(),
    );
    let endpoint = endpoint.to_string();
    let labels = [endpoint.as_str(), "adapter"];
    tokio::time::timeout(Duration::from_secs(1), async {
        while LORA_REPLICA_FACTOR_GAUGE.with_label_values(&labels).get() == 0 {
            tokio::time::sleep(Duration::from_millis(1)).await;
        }
    })
    .await
    .expect("controller must publish into the frontend's registered gauge");
    assert_eq!(
        LORA_ACTIVE_REQUESTS_GAUGE.with_label_values(&labels).get(),
        1
    );
    assert_eq!(metrics.replica_factor.with_label_values(&labels).get(), 1);
    cancel.cancel();
    handle.await.unwrap();
    for gauge in [
        &metrics.replica_factor,
        &metrics.is_active,
        &metrics.raw_arrival_count,
        &metrics.estimated_load,
        &metrics.active_requests,
    ] {
        gauge.remove_label_values(&labels).unwrap();
    }
    for gauge in [
        &metrics.churn_loads,
        &metrics.churn_unloads,
        &metrics.overflow_count,
    ] {
        let _ = gauge.remove_label_values(&[endpoint.as_str()]);
    }
}
