// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use crate::decision_profile::{DecisionExecutionProfile as Profile, SGLANG_SYSTEMONE_SERIAL_V1};

use super::tests::{FakeHost, finish_build, group_key, instance};
use super::*;
use std::sync::atomic::Ordering;

fn decision_instance(
    id: u64,
    qualification: Option<super::super::test_support::DecisionQualification>,
) -> DesiredInstance {
    let mut desired = instance(id, "same");
    let mut card = ModelDeploymentCard::with_name_only("model");
    card.source_path = Some("same".into());
    card.model_input = crate::model_type::ModelInput::Text;
    card.model_type = crate::model_type::ModelType::Chat;
    card.worker_type = Some(crate::worker_type::WorkerType::Aggregated);
    if let Some((profile, capacity)) = qualification {
        card.runtime_config
            .runtime_data
            .insert(profile.decision_capability().into(), true.into());
        card.runtime_config.runtime_data.insert(
            crate::decision_profile::VLLM_NATIVE_DECISION_MAX_CANDIDATES.into(),
            capacity.into(),
        );
    }
    desired.mdc_checksum = card.mdcsum().to_owned();
    desired.card = card;
    desired
}

#[tokio::test(start_paused = true)]
async fn decision_only_updates_preserve_materialization_backoff() {
    use crate::decision_profile::DecisionExecutionProfile as Profile;
    for blocked in [false, true] {
        let (host, mut starts) = FakeHost::new();
        let mut controller = ModelDiscoveryController::new(host.clone());
        let first = decision_instance(1, Some((Profile::VllmLiquidV1, 26)));
        if blocked {
            host.commit_failures.store(1, Ordering::SeqCst);
        } else {
            host.failures.store(1, Ordering::SeqCst);
        }
        controller.apply_added(first.clone());
        controller.start_queued_builds();
        let fleet = starts.recv().await.unwrap().decision_fleet;
        host.release.add_permits(1);
        finish_build(&mut controller).await;
        let group = &controller.groups[&group_key()];
        let deadline = match &group.status {
            GroupStatus::Retrying { deadline, .. } if !blocked => *deadline,
            GroupStatus::Blocked { deadline, .. } if blocked => *deadline,
            _ => panic!("fixture did not enter ordinary retry backoff"),
        };
        let attempt = group.retry_attempt;
        let revision = controller.revision;
        let mut updated = first;
        updated
            .card
            .runtime_config
            .runtime_data
            .remove(Profile::VllmLiquidV1.decision_capability());
        controller.apply_added(updated.clone());
        assert!(controller.revision > revision);
        assert_eq!(
            controller.desired[&updated.key]
                .card
                .decision_qualification(),
            None
        );
        assert_eq!(fleet.test_qualification(), None);
        let group = &controller.groups[&group_key()];
        assert_eq!(group.retry_attempt, attempt);
        assert!(
            matches!(&group.status, GroupStatus::Retrying { deadline: current, .. } | GroupStatus::Blocked { deadline: current, .. } if *current == deadline)
        );
        controller.start_queued_builds();
        assert!(starts.try_recv().is_err());
        assert_eq!(host.starts.load(Ordering::SeqCst), 1);
        updated.video_contract = Some("new ordinary video contract".into());
        controller.apply_added(updated);
        assert!(matches!(
            controller.groups[&group_key()].status,
            GroupStatus::Queued { .. }
        ));
        assert_eq!(controller.groups[&group_key()].retry_attempt, 0);
    }
}

#[test]
fn decision_cohort_distinguishes_disagreement_from_no_valid_advertisers() {
    use crate::decision_profile::DecisionExecutionProfile as Profile;
    let head = |id, capacity| {
        let mut desired = decision_instance(id, Some((Profile::VllmClefFlashV1, capacity)));
        desired.card.model_type =
            crate::model_type::ModelType::Classify | crate::model_type::ModelType::Pooling;
        desired.mdc_checksum = desired.card.mdcsum().to_owned();
        desired
    };
    let valid = head(1, 3);
    let matching = head(2, 3);
    let conflicting = head(2, 4);
    let invalid = head(2, 0);
    let mut ordinary = matching.clone();
    ordinary
        .card
        .runtime_config
        .runtime_data
        .remove(Profile::VllmClefFlashV1.decision_capability());
    ordinary
        .card
        .runtime_config
        .runtime_data
        .remove(crate::decision_profile::VLLM_NATIVE_DECISION_MAX_CANDIDATES);
    ordinary.mdc_checksum = ordinary.card.mdcsum().to_owned();
    assert_eq!(valid.mdc_checksum, ordinary.mdc_checksum);
    assert_eq!(valid.mdc_checksum, conflicting.mdc_checksum);
    let mut failed = ordinary.clone();
    failed.card.runtime_config.runtime_data.insert(
        crate::decision_profile::VLLM_NATIVE_DECISION_QUALIFICATION_FAILURE_V1.into(),
        serde_json::json!({"profile":"vllm_clef_flash_joint_decision_v1"}),
    );
    assert_eq!(failed.card.mdcsum(), valid.card.mdcsum());
    for members in [vec![failed.clone()], vec![valid.clone(), failed]] {
        assert!(matches!(
            cohort_decision_availability(&members),
            DecisionAvailability::Failed
        ));
    }
    assert!(valid.card.decision_qualification().is_some());
    assert!(invalid.card.decision_qualification().is_none());
    assert!(matches!(
        cohort_decision_availability(&[]),
        DecisionAvailability::Unadvertised
    ));
    for members in [
        vec![ordinary.clone()],
        vec![invalid.clone()],
        vec![ordinary.clone(), invalid],
    ] {
        assert!(matches!(
            cohort_decision_availability(&members),
            DecisionAvailability::Unadvertised
        ));
    }
    for members in [
        vec![valid.clone(), ordinary],
        vec![valid.clone(), conflicting],
    ] {
        assert!(matches!(
            cohort_decision_availability(&members),
            DecisionAvailability::Pending
        ));
    }
    let DecisionAvailability::Qualified(admission) =
        cohort_decision_availability(&[valid, matching])
    else {
        panic!("uniform valid head capacities must qualify");
    };
    assert_eq!(admission.qualification, (Profile::VllmClefFlashV1, 3));
    assert_eq!(admission.ids, vec![1, 2]);
}

#[tokio::test]
async fn decision_only_capability_revokes_a_stalled_dispatch_before_ordinary_admission() {
    if crate::test_utils::run_isolated(
        concat!(
            module_path!(),
            "::decision_only_capability_revokes_a_stalled_dispatch_before_ordinary_admission"
        ),
        &[("DYN_ROUTER_MIN_INITIAL_WORKERS", "0")],
    ) {
        return;
    }
    use crate::decision_profile::DecisionExecutionProfile as Profile;
    use dynamo_runtime::{
        DistributedRuntime, Runtime,
        distributed::DistributedConfig,
        pipeline::{RouterMode, SingleIn, network::egress::push_router::PushRouter},
    };
    let runtime = Runtime::from_current().unwrap();
    let drt = DistributedRuntime::new(runtime.clone(), DistributedConfig::process_local())
        .await
        .unwrap();
    let endpoint = drt
        .namespace("namespace")
        .unwrap()
        .component("worker")
        .unwrap()
        .endpoint("generate");
    let discovered = endpoint.client().await.unwrap();
    endpoint.register_endpoint_instance().await.unwrap();
    let worker_id = discovered.wait_for_instances().await.unwrap()[0].id();
    let (host, mut starts) = FakeHost::new();
    let mut controller = ModelDiscoveryController::new(host.clone());
    controller.apply_added(decision_instance(
        worker_id,
        Some((Profile::VllmLiquidV1, 26)),
    ));
    controller.start_queued_builds();
    let fleet = starts.recv().await.unwrap().decision_fleet;
    host.release.add_permits(1);
    finish_build(&mut controller).await;
    let ordinary = discovered.with_admitted_instances(host.admissions.lock().unwrap()[0].clone());
    let context = crate::discovery::test_support::context_for_test(
        fleet.clone(),
        host.admissions.lock().unwrap()[0].clone(),
    );
    let router =
        PushRouter::<serde_json::Value, crate::protocols::Annotated<serde_json::Value>>::from_client_with_options(
            ordinary.clone(), RouterMode::Direct,
            dynamo_runtime::pipeline::PushRouterOptions { fault_detection: false, dispatch: Some(crate::discovery::DecisionDispatch::for_client(&ordinary, crate::discovery::GuardPolicy::Required).await.unwrap()), ..Default::default() },
        )
        .await
        .unwrap();
    assert!(ordinary.is_instance_live(worker_id));
    context.validate_dispatch(Some(worker_id)).unwrap();
    let mut request = SingleIn::new(serde_json::json!({}));
    context.attach_dispatch_guard(&mut request);
    let (resume, stalled) = tokio::sync::oneshot::channel();
    let pending = tokio::spawn(async move {
        stalled.await.unwrap();
        router.direct(request, worker_id).await.err().map(|error| {
            dynamo_runtime::error::match_error_chain(
                error.as_ref(),
                &[dynamo_runtime::error::ErrorType::Unavailable],
                &[],
            )
        })
    });
    tokio::task::yield_now().await;
    controller.apply_added(decision_instance(worker_id + 1, None));
    assert!(
        ordinary.is_instance_live(worker_id),
        "ordinary serving must survive the pending rebuild"
    );
    assert!(
        context.validate_dispatch(Some(worker_id)).is_err(),
        "decision revocation must not wait for client snapshot propagation"
    );
    resume.send(()).unwrap();
    let error = pending
        .await
        .unwrap()
        .expect("stale dispatch must fail before transport");
    assert!(
        error,
        "stale decision dispatch must preserve typed Unavailable"
    );
    assert_eq!(host.removed_groups.load(Ordering::SeqCst), 0);
    runtime.shutdown();
}

#[tokio::test]
async fn requalified_fleet_follows_retained_admission_when_replacement_is_blocked() {
    if crate::test_utils::run_isolated(
        concat!(
            module_path!(),
            "::requalified_fleet_follows_retained_admission_when_replacement_is_blocked"
        ),
        &[("DYN_ROUTER_MIN_INITIAL_WORKERS", "0")],
    ) {
        return;
    }
    use crate::decision_profile::DecisionExecutionProfile as Profile;
    use dynamo_runtime::{DistributedRuntime, Runtime, distributed::DistributedConfig};
    let runtime = Runtime::from_current().unwrap();
    let drt = DistributedRuntime::new(runtime.clone(), DistributedConfig::process_local())
        .await
        .unwrap();
    let endpoint = drt
        .namespace("blocked-ready")
        .unwrap()
        .component("workers")
        .unwrap()
        .endpoint("generate");
    let discovered = endpoint.client().await.unwrap();
    endpoint.register_endpoint_instance().await.unwrap();
    let worker = discovered.wait_for_instances().await.unwrap()[0].id();
    let peer = worker.checked_add(1).unwrap();
    let (host, mut starts) = FakeHost::new();
    let mut controller = ModelDiscoveryController::new(host.clone());
    let qualified = |id| decision_instance(id, Some((Profile::VllmLiquidV1, 26)));
    controller.apply_added(qualified(worker));
    controller.start_queued_builds();
    let fleet = starts.recv().await.unwrap().decision_fleet;
    host.release.add_permits(1);
    finish_build(&mut controller).await;
    let ordinary = host.admissions.lock().unwrap()[0].clone();
    let stale = super::super::test_support::context_for_test(fleet.clone(), ordinary.clone());
    assert_eq!(fleet.test_ids(), *ordinary.borrow());
    controller.apply_added(decision_instance(peer, None));
    assert!(matches!(
        *fleet.state.borrow(),
        DecisionAvailability::Pending
    ));
    assert_eq!(*ordinary.borrow(), vec![worker, peer]);
    assert!(stale.validate_dispatch(Some(worker)).is_err());
    host.replace_failures.store(1, Ordering::SeqCst);
    controller.apply_added(qualified(peer));
    assert!(matches!(
        controller.groups[&group_key()].status,
        GroupStatus::BlockedReady { .. }
    ));
    assert_eq!(
        fleet.test_ids(),
        *ordinary.borrow(),
        "requalified retained workers must not depend on an old empty fleet"
    );
    assert_eq!(fleet.test_ids(), vec![worker, peer]);
    assert_eq!(host.starts.load(Ordering::SeqCst), 1);
    assert_eq!(host.members(&group_key()).len(), 2);
    let client = discovered
        .with_admitted_instances_and_cancellation(ordinary.clone(), runtime.child_token());
    client.wait_for_instances().await.unwrap();
    assert_eq!(client.instance_ids_avail(), vec![worker]);
    let fresh = super::super::test_support::context_for_test(fleet.clone(), ordinary.clone());
    super::super::test_support::assert_native_dispatch(&client, &fresh).await;
    let mut failed = decision_instance(peer, None);
    failed.card.runtime_config.runtime_data.insert(
        crate::decision_profile::VLLM_NATIVE_DECISION_QUALIFICATION_FAILURE_V1.into(),
        serde_json::json!({"profile":"vllm_liquid_decision_v1"}),
    );
    controller.apply_added(failed);
    assert!(matches!(
        *fleet.state.borrow(),
        DecisionAvailability::Failed
    ));
    assert_eq!(*ordinary.borrow(), vec![worker, peer]);
    assert!(fresh.validate_dispatch(Some(worker)).is_err());
    controller.apply_added(qualified(peer));
    assert_eq!(fleet.test_ids(), *ordinary.borrow());
    let recovered = super::super::test_support::context_for_test(fleet, ordinary);
    super::super::test_support::assert_native_dispatch(&client, &recovered).await;
    assert!(stale.validate_dispatch(Some(worker)).is_err());
    runtime.shutdown();
}

#[tokio::test]
async fn systemone_serial_fleet_admits_unsafe_peers_only_for_ordinary_serving() {
    use crate::local_model::runtime_config::{SGLANG_GENERATE_CAPABILITY, SPEC_DECODE_RUNTIME_KEY};
    fn serial_instance(id: u64, speculative: bool) -> DesiredInstance {
        let mut desired = decision_instance(id, None);
        desired.card.model_input = crate::model_type::ModelInput::Tokens;
        desired.card.runtime_config.max_num_seqs = Some(1);
        for capability in [SGLANG_GENERATE_CAPABILITY, SGLANG_SYSTEMONE_SERIAL_V1] {
            desired
                .card
                .runtime_config
                .runtime_data
                .insert(capability.into(), true.into());
        }
        if speculative {
            desired
                .card
                .runtime_config
                .runtime_data
                .insert(SPEC_DECODE_RUNTIME_KEY.into(), serde_json::json!({}));
        }
        desired.mdc_checksum = desired.card.mdcsum().to_owned();
        desired
    }
    let (host, mut starts) = FakeHost::new();
    let mut controller = ModelDiscoveryController::new(host.clone());
    let safe = serial_instance(1, false);
    let unsafe_peer = serial_instance(2, true);
    assert_eq!(safe.mdc_checksum, unsafe_peer.mdc_checksum);
    controller.apply_added(safe);
    controller.start_queued_builds();
    let fleet = starts.recv().await.unwrap().decision_fleet;
    host.release.add_permits(1);
    finish_build(&mut controller).await;
    let ordinary = host.admissions.lock().unwrap()[0].clone();
    assert_eq!(
        fleet.test_qualification(),
        Some((Profile::SglangSerialV1, 255))
    );
    assert_eq!(fleet.test_ids(), vec![1]);
    controller.apply_added(unsafe_peer);
    assert_eq!(*ordinary.borrow(), vec![1, 2]);
    assert_eq!(host.members(&group_key()).len(), 2);
    assert_eq!(fleet.test_qualification(), None);
    assert!(fleet.test_ids().is_empty());
    controller.apply_added(serial_instance(2, false));
    assert_eq!(*ordinary.borrow(), vec![1, 2]);
    assert_eq!(
        fleet.test_qualification(),
        Some((Profile::SglangSerialV1, 255))
    );
    assert_eq!(fleet.test_ids(), vec![1, 2]);
    controller.start_queued_builds();
    assert_eq!(controller.active_builds, 0);
    assert_eq!(host.starts.load(Ordering::SeqCst), 1);
    assert_eq!(host.prepared_replacements.load(Ordering::SeqCst), 0);
}

#[test]
fn decision_admission_snapshot_preserves_noop_identity_and_fences_aba_and_close() {
    use crate::decision_profile::DecisionExecutionProfile as Profile;
    let group = DesiredGroup::new();
    let qualification = (Profile::VllmLiquidV1, 26);
    crate::discovery::test_support::publish(
        &group.decision_admission_tx,
        Some(qualification),
        vec![2, 1, 2],
    );
    let fleet = DecisionFleet {
        state: group.decision_admission_tx.subscribe(),
    };
    let original = fleet.current().unwrap().admission().unwrap().clone();
    crate::discovery::test_support::publish(
        &group.decision_admission_tx,
        Some(qualification),
        vec![1, 2],
    );
    assert!(
        Arc::ptr_eq(&original, fleet.current().unwrap().admission().unwrap()),
        "nonsemantic ID order must not revoke requests"
    );
    let (_ordinary, ids) = watch::channel(vec![1, 2]);
    let context = crate::discovery::test_support::context_for_test(fleet.clone(), ids);
    context.validate_dispatch(Some(1)).unwrap();
    crate::discovery::test_support::publish(
        &group.decision_admission_tx,
        Some(qualification),
        vec![1, 2, 3],
    );
    context.validate_dispatch(Some(1)).unwrap();
    crate::discovery::test_support::publish(
        &group.decision_admission_tx,
        Some(qualification),
        vec![1],
    );
    context.validate_dispatch(Some(1)).unwrap();
    assert!(context.validate_dispatch(Some(2)).is_err());
    crate::discovery::test_support::publish(
        &group.decision_admission_tx,
        Some(qualification),
        vec![2],
    );
    assert!(context.validate_dispatch(Some(1)).is_err());
    context.validate_dispatch(Some(2)).unwrap();
    crate::discovery::test_support::publish(
        &group.decision_admission_tx,
        Some(qualification),
        vec![],
    );
    assert!(context.validate_dispatch(None).is_err());
    crate::discovery::test_support::publish(
        &group.decision_admission_tx,
        Some(qualification),
        vec![1, 2],
    );
    assert!(
        context.validate_dispatch(Some(1)).is_err(),
        "empty membership withdrawal must revoke the old epoch"
    );
    let (_ordinary, ids) = watch::channel(vec![1, 2]);
    let context = crate::discovery::test_support::context_for_test(fleet.clone(), ids);
    DecisionAvailability::publish(&group.decision_admission_tx, DecisionAvailability::Pending);
    crate::discovery::test_support::publish(
        &group.decision_admission_tx,
        Some(qualification),
        vec![1, 2],
    );
    assert!(
        context.validate_dispatch(Some(1)).is_err(),
        "same-tuple recovery after disagreement is a distinct admission epoch"
    );
    assert!(!Arc::ptr_eq(
        &original,
        fleet.current().unwrap().admission().unwrap()
    ));
    drop(group);
    assert!(
        fleet.current().is_none(),
        "closed admission publication must fail closed"
    );
}

#[tokio::test]
async fn decision_only_capability_refresh_does_not_rebuild_ordinary_pipeline() {
    use crate::decision_profile::DecisionExecutionProfile as Profile;
    let qualification = (Profile::VllmLiquidV1, 26);
    let (host, mut starts) = FakeHost::new();
    let mut controller = ModelDiscoveryController::new(host.clone());
    controller.apply_added(decision_instance(1, None));
    controller.start_queued_builds();
    let fleet = starts.recv().await.unwrap().decision_fleet;
    host.release.add_permits(1);
    finish_build(&mut controller).await;
    let ordinary = host.admissions.lock().unwrap()[0].clone();
    controller.apply_added(decision_instance(1, Some(qualification)));
    controller.start_queued_builds();
    assert_eq!(controller.active_builds, 0);
    assert_eq!(host.starts.load(Ordering::SeqCst), 1);
    assert_eq!(host.prepared_replacements.load(Ordering::SeqCst), 0);
    assert_eq!(*ordinary.borrow(), vec![1]);
    assert_eq!(fleet.test_qualification(), Some(qualification));
    assert_eq!(fleet.test_ids(), vec![1]);
}

#[tokio::test(start_paused = true)]
async fn decision_only_capability_changes_refresh_live_admission_without_rebuilding() {
    use crate::decision_profile::DecisionExecutionProfile as Profile;
    let qualification = (Profile::VllmLiquidV1, 26);
    let (host, mut starts) = FakeHost::new();
    let mut controller = ModelDiscoveryController::new(host.clone());
    let ordinary = decision_instance(1, None);
    let qualified = decision_instance(2, Some(qualification));
    assert_eq!(ordinary.mdc_checksum, qualified.mdc_checksum);
    controller.apply_added(ordinary.clone());
    controller.apply_added(qualified.clone());
    controller.start_queued_builds();
    let mixed_spec = starts.recv().await.unwrap();
    let fleet = mixed_spec.decision_fleet;
    assert_eq!(fleet.test_qualification(), None);
    host.release.add_permits(1);
    finish_build(&mut controller).await;
    let ordinary_admission = host.admissions.lock().unwrap()[0].clone();
    assert_eq!(*ordinary_admission.borrow(), vec![1, 2]);
    assert!(fleet.test_ids().is_empty());
    assert_eq!(
        host.members(&group_key()),
        BTreeSet::from([ordinary.key.clone(), qualified.key.clone()])
    );

    // Optional qualification refreshes live gates, never ordinary pipelines.
    assert!(controller.apply_added(decision_instance(1, Some(qualification))));
    assert!(!controller.apply_added(decision_instance(1, Some(qualification))));
    assert_eq!(*ordinary_admission.borrow(), vec![1, 2]);
    assert_eq!(fleet.test_qualification(), Some(qualification));
    assert_eq!(fleet.test_ids(), vec![1, 2]);

    // Capacity disagreement revokes decisions synchronously, not ordinary workers.
    controller.apply_added(decision_instance(1, Some((Profile::VllmLiquidV1, 3))));
    assert_eq!(*ordinary_admission.borrow(), vec![1, 2]);
    assert_eq!(fleet.test_qualification(), None);
    assert!(fleet.test_ids().is_empty());
    controller.apply_added(decision_instance(2, Some((Profile::VllmLiquidV1, 3))));
    assert_eq!(fleet.test_qualification(), Some((Profile::VllmLiquidV1, 3)));
    assert_eq!(fleet.test_ids(), vec![1, 2]);

    controller.apply_added(decision_instance(1, Some((Profile::VllmDiffusionV1, 3))));
    assert_eq!(*ordinary_admission.borrow(), vec![1, 2]);
    assert_eq!(fleet.test_qualification(), None);
    assert!(fleet.test_ids().is_empty());
    controller.apply_added(decision_instance(2, Some((Profile::VllmDiffusionV1, 3))));
    assert_eq!(
        fleet.test_qualification(),
        Some((Profile::VllmDiffusionV1, 3))
    );
    assert_eq!(fleet.test_ids(), vec![1, 2]);
    controller.start_queued_builds();
    assert_eq!(controller.active_builds, 0);
    assert_eq!(host.starts.load(Ordering::SeqCst), 1);
    assert_eq!(host.prepared_replacements.load(Ordering::SeqCst), 0);
    controller.apply_added(instance(99, "different-model-revision"));
    assert_eq!(*ordinary_admission.borrow(), vec![1, 2]);
    assert_eq!(
        host.inventory
            .snapshot()
            .pop()
            .unwrap()
            .1
            .checksum_mismatches,
        HashSet::from([99])
    );
    assert_eq!(host.removed_groups.load(Ordering::SeqCst), 0);
}
