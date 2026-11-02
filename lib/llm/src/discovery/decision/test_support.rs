// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use super::*;
use tokio::sync::watch;

pub(crate) use super::super::decision_dispatch::validate_fixture_dispatch;
pub(crate) use super::super::decision_resources::token_error_is_retryable;
pub(crate) use super::DecisionQualification;
pub(crate) use super::dispatch_tests::assert_native_dispatch;

pub(crate) async fn routing_load_for_test() -> (
    dynamo_runtime::Runtime,
    Arc<crate::kv_router::RoutingLoadContext>,
) {
    let runtime = dynamo_runtime::Runtime::from_current().unwrap();
    let distributed = dynamo_runtime::DistributedRuntime::new(
        runtime.clone(),
        dynamo_runtime::distributed::DistributedConfig::process_local(),
    )
    .await
    .unwrap();
    let client = distributed
        .namespace("decision-resource-fixture")
        .unwrap()
        .component("worker")
        .unwrap()
        .endpoint("generate")
        .client()
        .await
        .unwrap();
    let load = crate::kv_router::RoutingLoadContext::start(
        client,
        crate::kv_router::RouterLoadSource::Aggregated,
        crate::discovery::LoadThresholdHandle::new(Default::default()),
        &runtime.primary_token(),
        None,
    )
    .await
    .unwrap();
    (runtime, load)
}

pub(crate) struct PreparedSelectionFixture {
    pub(crate) selection: SystemOneExecutionSelection,
    _workers: WorkerSet,
    _publication: watch::Sender<DecisionAvailability>,
    _membership: watch::Sender<Vec<u64>>,
}

impl std::ops::Deref for PreparedSelectionFixture {
    type Target = SystemOneExecutionSelection;
    fn deref(&self) -> &Self::Target {
        &self.selection
    }
}

pub(crate) async fn attach_surface_for_test(
    workers: &mut WorkerSet,
    resources: Option<Arc<DecisionResources>>,
    fleet: DecisionFleet,
) {
    if let Some(resources) = &resources {
        super::super::worker_set::prepare_teardown_for_test(
            workers,
            resources.teardown_guard_for_test(),
        );
    }
    if workers.lifecycle_cancellation().is_none() {
        workers.set_lifecycle_cancellation(tokio_util::sync::CancellationToken::new());
    }
    if workers.topology_target().is_none() {
        let runtime = dynamo_runtime::Runtime::from_current().unwrap();
        let distributed = dynamo_runtime::DistributedRuntime::new(
            runtime,
            dynamo_runtime::distributed::DistributedConfig::process_local(),
        )
        .await
        .unwrap();
        let endpoint = distributed
            .namespace("decision-fixture")
            .unwrap()
            .component("worker")
            .unwrap()
            .endpoint("generate");
        workers.set_topology_target(super::super::CommittedWorkerSetTarget {
            endpoint,
            group: "decision-fixture".into(),
            generation: 1,
            card: Arc::new(workers.card().clone()),
            admitted_ids: super::super::worker_set::instance_watcher_for_test(workers).clone(),
        });
    }
    workers.decision = resources.map(|resources| DecisionSurface { resources, fleet });
}

pub(crate) async fn attach_card_surface_for_test(
    workers: &mut WorkerSet,
    resources: Option<Arc<DecisionResources>>,
) -> watch::Sender<DecisionAvailability> {
    let ids = super::super::worker_set::instance_watcher_for_test(workers)
        .borrow()
        .clone();
    let (publication, fleet) = fleet(workers.card().decision_qualification(), ids);
    if workers.card().decision_qualification_failure().is_some() {
        DecisionAvailability::publish(&publication, DecisionAvailability::Failed);
    }
    attach_surface_for_test(workers, resources, fleet).await;
    publication
}

pub(crate) fn fleet(
    qualification: Option<DecisionQualification>,
    ids: Vec<u64>,
) -> (watch::Sender<DecisionAvailability>, DecisionFleet) {
    let initial = qualification.map_or(DecisionAvailability::Unadvertised, |qualification| {
        DecisionAvailability::qualified(qualification, ids)
    });
    let (sender, state) = watch::channel(initial);
    (sender, DecisionFleet { state })
}

pub(crate) fn publish(
    sender: &watch::Sender<DecisionAvailability>,
    qualification: Option<DecisionQualification>,
    ids: Vec<u64>,
) {
    DecisionAvailability::publish(
        sender,
        qualification.map_or(DecisionAvailability::Unadvertised, |qualification| {
            DecisionAvailability::qualified(qualification, ids)
        }),
    );
}

pub(crate) fn context_for_test(
    fleet: DecisionFleet,
    admitted_ids: watch::Receiver<Vec<u64>>,
) -> DecisionContext {
    let admission = fleet.current().unwrap().admission().unwrap().clone();
    let (profile, max_candidates) = admission.qualification;
    DecisionContext {
        canonical_model: "model".into(),
        card: ModelDeploymentCard::with_name_only("model"),
        fence: Arc::new(DispatchFence {
            admitted_ids,
            cancellation: tokio_util::sync::CancellationToken::new(),
            fleet,
            admission,
        }),
        profile,
        max_candidates,
    }
}

pub(crate) async fn prepared_selection_for_test(
    canonical_model: String,
    workers: &WorkerSet,
) -> Option<SystemOneExecutionSelection> {
    prepare_plan_for_test(SystemOneExecutionPlan::from_worker_set(
        canonical_model,
        workers,
    )?)
    .await
    .ok()
}

async fn prepare_plan_for_test(
    plan: SystemOneExecutionPlan,
) -> Result<SystemOneExecutionSelection, SystemOneSelectionError> {
    plan.prepare_native()
        .await
        .map_err(|_| SystemOneSelectionError::Unavailable)?;
    plan.prepare().map_err(|cause| {
        cause
            .downcast::<SystemOneSelectionError>()
            .unwrap_or(SystemOneSelectionError::Unavailable)
    })
}

impl DecisionFleet {
    pub(crate) fn test_qualification(&self) -> Option<DecisionQualification> {
        self.current()?.admission().map(|state| state.qualification)
    }
    pub(crate) fn test_ids(&self) -> Vec<u64> {
        self.current()
            .and_then(|state| state.admission().map(|state| state.ids.clone()))
            .unwrap_or_default()
    }
}

pub(crate) fn token_resources(
    engine: GenerateStreamingEngine,
    preprocessor: Arc<OpenAIPreprocessor>,
) -> Option<Arc<DecisionResources>> {
    let mut resources = DecisionResources::default();
    resources.insert_token(TokenResources {
        engine,
        preprocessor,
    });
    resources.into_shared()
}

pub(crate) fn native_resources(engine: DecisionEngine) -> Option<Arc<DecisionResources>> {
    let mut resources = DecisionResources::default();
    match engine {
        DecisionEngine::Head(engine) => resources.insert_head(engine),
        DecisionEngine::Read(engine) => resources.insert_read(engine),
        DecisionEngine::Token(_) => {
            unreachable!("token fixtures construct their resource pair directly")
        }
    }
    resources.into_shared()
}

pub(crate) async fn forced_selection_for_test(
    canonical_model: String,
    profile: DecisionExecutionProfile,
    workers: WorkerSet,
    resources: Option<Arc<DecisionResources>>,
) -> Option<PreparedSelectionFixture> {
    forced_selection_with_admissions_for_test(canonical_model, profile, workers, resources, None)
        .await
}

// Forced profiles exercise negative/transport cases, not deployment eligibility.
pub(crate) async fn forced_selection_with_admissions_for_test(
    canonical_model: String,
    profile: DecisionExecutionProfile,
    mut workers: WorkerSet,
    resources: Option<Arc<DecisionResources>>,
    admissions: Option<watch::Receiver<Vec<u64>>>,
) -> Option<PreparedSelectionFixture> {
    let capacity = workers.card().decision_max_candidates(profile)?;
    let (membership, available) = watch::channel(vec![41]);
    workers.set_instance_watcher(available);
    let (publication, fleet) = fleet(Some((profile, capacity)), vec![41]);
    attach_surface_for_test(&mut workers, resources, fleet).await;
    if let Some(admissions) = admissions {
        let mut target = workers.topology_target().unwrap().clone();
        target.admitted_ids = admissions;
        workers.set_topology_target(target);
    }
    let selection = prepared_selection_for_test(canonical_model, &workers).await?;
    Some(PreparedSelectionFixture {
        selection,
        _workers: workers,
        _publication: publication,
        _membership: membership,
    })
}

impl super::super::Model {
    pub(crate) async fn prepared_selection_for_test(
        &self,
    ) -> Result<SystemOneExecutionSelection, SystemOneSelectionError> {
        prepare_plan_for_test(self.get_systemone_execution_plan()?).await
    }
}

impl super::super::ModelManager {
    pub(crate) async fn prepared_selection_for_test(
        &self,
        model: &str,
    ) -> Result<SystemOneExecutionSelection, SystemOneSelectionError> {
        prepare_plan_for_test(self.get_systemone_execution_plan(model)?).await
    }
}
