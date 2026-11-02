// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use super::*;

#[tokio::test]
async fn text_routes_retain_graph_and_preserve_declared_surfaces() {
    use dynamo_runtime::{Runtime, distributed::DistributedConfig};

    let runtime = Runtime::from_current().unwrap();
    let drt = DistributedRuntime::new(runtime, DistributedConfig::process_local())
        .await
        .unwrap();
    let manager = Arc::new(ModelManager::new());
    let router_config = RouterConfig {
        load_threshold_config: crate::discovery::LoadThresholdConfig {
            active_decode_blocks_threshold: Some(0.8),
            ..Default::default()
        },
        ..Default::default()
    };
    let watcher = Arc::new(ModelWatcher::new(
        drt,
        manager.clone(),
        router_config.clone(),
        0,
        None,
        None,
        None,
        Arc::new(Metrics::new_with_prefix(Some(
            "watcher_mixed_text_test".to_string(),
        ))),
    ));
    let mcid = ModelCardInstanceId {
        namespace: "mixed-text-ns".to_string(),
        component: "workers".to_string(),
        endpoint: "generate".to_string(),
        instance_id: 1,
        model_suffix: None,
    };
    let mut card = ModelDeploymentCard::with_name_only("mixed-text-model");
    card.model_input = ModelInput::Text;
    card.model_type = ModelType::Chat | ModelType::Classify | ModelType::Pooling;
    card.worker_type = Some(WorkerType::Aggregated);

    let endpoint_id = model_card_endpoint_id(&mcid);
    let key = GroupKey {
        model_name: card.name().to_string(),
        worker_set_key: worker_set_key(&endpoint_id, card.model_type, card.worker_type),
    };
    let desired = DesiredInstance {
        key: mcid.to_path(),
        mcid,
        endpoint_id,
        mdc_checksum: card.mdcsum().to_string(),
        projection_fingerprint: lora_projection_fingerprint(&card).unwrap(),
        video_contract: qwen_video_contract_digest(&card),
        card,
        group_key: key.clone(),
    };
    let spec = GroupSpec {
        key,
        mdc_checksum: desired.mdc_checksum.clone(),
        fingerprint: desired.mdc_checksum.clone(),
        generation: 1,
        representative: desired.clone(),
        video_contract: desired.video_contract.clone(),
    };
    let (_admission_tx, admission_rx) = tokio::sync::watch::channel(vec![1]);
    let prepared = watcher
        .prepare_worker_set(&spec, admission_rx, CancellationToken::new())
        .await
        .unwrap();
    let worker_set = prepared.worker_set.as_ref().unwrap();
    let load_context = worker_set
        .load_context()
        .expect("text routing must retain its routing load context");
    assert_eq!(load_context.source(), RouterLoadSource::Aggregated);
    assert!(load_context.monitor().is_some());
    assert_eq!(load_context.client().endpoint.id(), desired.endpoint_id);
    watcher
        .commit_group(&spec, prepared, &[desired], &[])
        .unwrap();

    let model = manager.get_model("mixed-text-model").unwrap();
    assert!(model.has_chat_engine());
    assert!(model.has_classify_engine());
    assert!(model.has_pooling_engine());
    assert_eq!(
        model
            .load_threshold_config(None)
            .unwrap()
            .active_decode_blocks_threshold,
        Some(0.8)
    );
}
