// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

mod support;

use dynamo_kv_router::plugins::prelude::*;
use dynamo_scheduler::host::WorkerSelector;
use dynamo_scheduler::test_utils::SimpleWorkerConfig;
use std::collections::{BTreeMap, HashMap};
use std::sync::Arc;

fn policy(role: WorkerType) -> WorkerSelectionPolicy {
    let mut registry = dynamo_router_policies::default_registry();
    dynamo_router_policies::register(&mut registry).unwrap();
    let file = tempfile::NamedTempFile::new().unwrap();
    std::fs::write(file.path(), "worker_selection:\n  aggregated: dynamo-manual-weighted\n  prefill: dynamo-manual-weighted\n  decode: dynamo-manual-weighted\n").unwrap();
    let config = KvRouterConfig {
        router_policy_config: Some(file.path().display().to_string()),
        ..Default::default()
    };
    registry.resolve(&config).unwrap().unwrap()(
        &SelectionConfig::from(&config),
        role,
        RoutingPartitionRef::new("model", "default"),
    )
}

fn runtime_policy(role: WorkerType) -> WorkerSelectionPolicy {
    policy(role).with_runtime_config(
        serde_json::Map::from_iter([("exclusive_affinity".into(), false.into())]),
        Arc::new(move |parameters| {
            Ok(policy(role)
                .with_exclusive_affinity(parameters["exclusive_affinity"].as_bool().unwrap()))
        }),
    )
}

#[test]
fn runtime_updates_preserve_worker_weights_and_revision() {
    let mut policy = runtime_policy(WorkerType::Aggregated);
    let (workers, mut request) = support::fixture(2, 32);
    let initial = policy.control_worker_weights(&[0, 1], None).unwrap();
    let configured = policy
        .control_worker_weights(
            &[0, 1],
            Some(WorkerWeightUpdate::new(
                initial.revision(),
                BTreeMap::from([(0, Some(0.0)), (1, Some(0.25))]),
            )),
        )
        .unwrap();
    for update in [
        RuntimePolicyUpdate::patch(
            0,
            serde_json::Map::from_iter([("exclusive_affinity".into(), false.into())]),
        ),
        RuntimePolicyUpdate::reset(1),
    ] {
        policy.update_runtime_config(update).unwrap();
        let current = policy.control_worker_weights(&[0, 1], None).unwrap();
        assert_eq!(current.weights(), configured.weights());
        assert_eq!(current.revision(), configured.revision());
        assert!(matches!(
            policy.control_worker_weights(
                &[0, 1],
                Some(WorkerWeightUpdate::new(
                    initial.revision(),
                    BTreeMap::from([(0, None)]),
                )),
            ),
            Err(RuntimePolicyError::Conflict { .. })
        ));
        request.pinned_worker = None;
        assert_eq!(
            policy
                .select_worker(support::selection_input(&workers, &request, 16))
                .unwrap()
                .worker
                .worker_id,
            1
        );
        request.pinned_worker = Some(WorkerWithDpRank::new(0, 0));
        assert!(matches!(
            policy.select_worker(support::selection_input(&workers, &request, 16)),
            Err(dynamo_scheduler::KvSchedulerError::AllEligibleWorkersFiltered)
        ));
    }
}

#[test]
fn runtime_updates_cannot_change_weight_control() {
    struct UnweightedPicker;
    impl WorkerPicker for UnweightedPicker {
        fn pick(
            &self,
            _: &WorkerSelectionContext<'_>,
            _: WorkerInputView<'_>,
        ) -> Result<usize, WorkerSelectionPolicyError> {
            Ok(0)
        }
    }
    let build = |weighted| {
        if weighted {
            policy(WorkerType::Aggregated)
        } else {
            WorkerSelectionPolicy::new(
                Default::default(),
                "aggregated",
                vec![],
                Box::new(UnweightedPicker),
            )
        }
    };
    for weighted in [false, true] {
        let mut policy = build(weighted).with_runtime_config(
            serde_json::Map::from_iter([("weighted".into(), weighted.into())]),
            Arc::new(move |parameters| Ok(build(parameters["weighted"].as_bool().unwrap()))),
        );
        let runtime_before = policy.runtime_config().unwrap();
        let before = policy
            .control_worker_weights(&[1], None)
            .ok()
            .map(|config| {
                policy
                    .control_worker_weights(
                        &[1],
                        Some(WorkerWeightUpdate::new(
                            config.revision(),
                            BTreeMap::from([(1, Some(0.0))]),
                        )),
                    )
                    .unwrap()
            });
        assert!(matches!(
            policy.update_runtime_config(RuntimePolicyUpdate::patch(
                0,
                serde_json::Map::from_iter([("weighted".into(), (!weighted).into())]),
            )),
            Err(RuntimePolicyError::InvalidParameters(_))
        ));
        assert_eq!(policy.runtime_config().unwrap(), runtime_before);
        match before {
            Some(before) => {
                let after = policy.control_worker_weights(&[1], None).unwrap();
                assert_eq!(after.weights(), before.weights());
                assert_eq!(after.revision(), before.revision());
            }
            None => assert!(matches!(
                policy.control_worker_weights(&[1], None),
                Err(RuntimePolicyError::Unsupported)
            )),
        }
    }
}

#[test]
fn live_weights_set_worker_shares_independently_of_dp_ranks() {
    fastrand::seed(24);
    let mut policy = policy(WorkerType::Aggregated);
    let workers = HashMap::from([
        (1, SimpleWorkerConfig::default()),
        (
            2,
            SimpleWorkerConfig {
                data_parallel_size: 4,
                ..Default::default()
            },
        ),
    ]);
    let (_, mut request) = support::fixture(3, 32);
    let config = policy.control_worker_weights(&[1, 2], None).unwrap();
    assert!(config.weights().values().all(|&weight| weight == 1.0));
    let fraction = |policy: &WorkerSelectionPolicy| {
        let selected = (0..16_000)
            .filter(|_| {
                policy
                    .select_worker(support::selection_input(&workers, &request, 16))
                    .unwrap()
                    .worker
                    .worker_id
                    == 2
            })
            .count();
        selected as f64 / 16_000.0
    };
    assert!((fraction(&policy) - 0.5).abs() < 0.02);
    let config = policy
        .control_worker_weights(
            &[1, 2],
            Some(WorkerWeightUpdate::new(
                config.revision(),
                BTreeMap::from([(2, Some(4.0))]),
            )),
        )
        .unwrap();
    assert!((fraction(&policy) - 0.8).abs() < 0.02);
    policy
        .control_worker_weights(
            &[1, 2],
            Some(WorkerWeightUpdate::new(
                config.revision(),
                BTreeMap::from([(2, Some(0.0))]),
            )),
        )
        .unwrap();
    for _ in 0..8 {
        assert_eq!(
            policy
                .select_worker(support::selection_input(&workers, &request, 16))
                .unwrap()
                .worker
                .worker_id,
            1
        );
    }
    request.pinned_worker = Some(WorkerWithDpRank::new(2, 0));
    let error = policy
        .select_worker(support::selection_input(&workers, &request, 16))
        .unwrap_err();
    assert!(matches!(
        error,
        dynamo_scheduler::KvSchedulerError::AllEligibleWorkersFiltered
    ));
}

#[test]
fn weight_state_is_local_to_each_role_policy() {
    let mut aggregated = policy(WorkerType::Aggregated);
    let mut prefill = policy(WorkerType::Prefill);
    let mut decode = policy(WorkerType::Decode);
    let config = prefill.control_worker_weights(&[1], None).unwrap();
    prefill
        .control_worker_weights(
            &[1],
            Some(WorkerWeightUpdate::new(
                config.revision(),
                BTreeMap::from([(1, Some(0.0))]),
            )),
        )
        .unwrap();
    assert_eq!(
        aggregated
            .control_worker_weights(&[1], None)
            .unwrap()
            .weights()[&1],
        1.0
    );
    assert_eq!(
        decode.control_worker_weights(&[1], None).unwrap().weights()[&1],
        1.0
    );
}

#[tokio::test]
async fn zero_weight_workers_do_not_bypass_queue_backpressure() {
    use dynamo_kv_router::services::selection::local_selection_core;
    use dynamo_scheduler::host::{
        SelectAndReserveRequest, SelectRequest, SelectionCacheConfig, WorkerRequest,
    };
    let core = local_selection_core(
        KvRouterConfig {
            use_kv_events: false,
            router_queue_threshold: Some(0.0),
            ..Default::default()
        },
        1,
        Default::default(),
        SelectionCacheConfig::default(),
        Arc::new(|_, role, _| runtime_policy(role)),
    )
    .unwrap();
    for worker_id in [1, 2] {
        core.upsert_worker(WorkerRequest {
            worker_id,
            model_name: "model".into(),
            endpoint: Some(format!("http://worker-{worker_id}:8000")),
            block_size: Some(4),
            max_num_batched_tokens: Some(8),
            ..Default::default()
        })
        .await
        .unwrap();
    }
    let partition = core
        .partition(&RoutingPartitionId::new("model", "default"))
        .unwrap();
    let control = partition.scheduler().policy_control();
    let initial = control.worker_weights(None).await.unwrap();
    let changed = control
        .worker_weights(Some(WorkerWeightUpdate::new(
            initial.revision(),
            BTreeMap::from([(2, Some(0.0))]),
        )))
        .await
        .unwrap();
    let reserve = serde_json::from_value::<SelectAndReserveRequest>(serde_json::json!({
        "model_name":"model", "token_ids":[1,2,3,4], "selection_id":"busy"
    }))
    .unwrap();
    assert_eq!(core.select_and_reserve(reserve).await.unwrap().worker_id, 1);
    let request = serde_json::from_value::<SelectRequest>(serde_json::json!({
        "model_name":"model", "token_ids":[5,6,7,8]
    }))
    .unwrap();
    let pending = core.select(request);
    tokio::pin!(pending);
    tokio::time::timeout(std::time::Duration::from_secs(2), async {
        tokio::select! {
            result = &mut pending => panic!("disabled worker bypassed queue capacity: {result:?}"),
            _ = async {
                while partition.scheduler().pending_count() != 1 {
                    tokio::task::yield_now().await;
                }
            } => {}
        }
    })
    .await
    .unwrap();
    let current = tokio::time::timeout(std::time::Duration::from_secs(2), async {
        control.update(RuntimePolicyUpdate::reset(0)).await.unwrap();
        control.worker_weights(None).await.unwrap()
    })
    .await
    .unwrap();
    assert_eq!(current.weights(), changed.weights());
    assert_eq!(current.revision(), changed.revision());
    assert_eq!(partition.scheduler().pending_count(), 1);
    control
        .worker_weights(Some(WorkerWeightUpdate::new(
            changed.revision(),
            BTreeMap::from([(1, Some(0.0)), (2, None)]),
        )))
        .await
        .unwrap();
    assert_eq!(
        tokio::time::timeout(std::time::Duration::from_secs(2), pending)
            .await
            .unwrap()
            .unwrap()
            .worker_id,
        2
    );
    core.shutdown();
}
