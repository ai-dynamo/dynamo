// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::{
    collections::{BTreeMap, BTreeSet, HashMap, HashSet},
    panic::AssertUnwindSafe,
    sync::Arc,
    time::Duration,
};

use async_trait::async_trait;
use dynamo_runtime::{
    discovery::{
        DiscoveryEvent, DiscoveryInstance, DiscoveryInstanceId, DiscoveryStream,
        ModelCardInstanceId,
    },
    protocols::EndpointId,
};
use futures::{FutureExt, StreamExt};
use tokio::{sync::watch, task::JoinSet, time::Instant};
use tokio_util::sync::CancellationToken;

use crate::{model_card::ModelDeploymentCard, namespace::NamespaceFilter};

const DEFAULT_MAX_CONCURRENT_BUILDS: usize = 8;
const RECONCILIATION_INTERVAL: Duration = Duration::from_secs(30);

#[derive(Clone, Debug, Eq, Hash, PartialEq, Ord, PartialOrd)]
pub(crate) struct GroupKey {
    pub(crate) model_name: String,
    pub(crate) worker_set_key: String,
}

impl GroupKey {
    pub(crate) fn id(&self) -> String {
        serde_json::to_string(&(&self.model_name, &self.worker_set_key))
            .expect("serializing discovery group keys cannot fail")
    }
}

#[derive(Clone, Debug)]
pub(crate) struct DesiredInstance {
    pub(crate) key: String,
    pub(crate) mcid: ModelCardInstanceId,
    pub(crate) endpoint_id: EndpointId,
    pub(crate) card: ModelDeploymentCard,
    pub(crate) group_key: GroupKey,
    pub(crate) fingerprint: String,
    pub(crate) projection_fingerprint: String,
}

impl DesiredInstance {
    fn materializes_worker_set(&self) -> bool {
        self.mcid.model_suffix.is_none()
    }
}

#[derive(Clone, Debug)]
pub(crate) struct GroupSpec {
    pub(crate) key: GroupKey,
    pub(crate) fingerprint: String,
    pub(crate) generation: u64,
    pub(crate) representative: DesiredInstance,
}

#[async_trait]
pub(crate) trait ControllerHost: Send + Sync + 'static {
    type Prepared: Send + 'static;

    fn normalize(
        &self,
        instance: DiscoveryInstance,
        namespace_filter: &NamespaceFilter,
    ) -> anyhow::Result<Option<DesiredInstance>>;

    async fn prepare(
        &self,
        spec: GroupSpec,
        admitted_ids: watch::Receiver<Vec<u64>>,
        cancellation: CancellationToken,
    ) -> anyhow::Result<Self::Prepared>;

    fn commit_group(
        &self,
        spec: &GroupSpec,
        prepared: Self::Prepared,
        members: &[DesiredInstance],
        adapters: &[DesiredInstance],
    ) -> anyhow::Result<()>;

    fn replace_group(
        &self,
        key: &GroupKey,
        members: &[DesiredInstance],
        adapters: &[DesiredInstance],
    ) -> anyhow::Result<()>;

    fn remove_group(&self, key: &GroupKey);

    fn discard_prepared(&self, prepared: Self::Prepared);

    async fn list_instances(&self) -> anyhow::Result<Vec<DiscoveryInstance>>;
}

#[derive(Clone)]
enum GroupStatus {
    Idle,
    Queued {
        fingerprint: String,
    },
    Building {
        fingerprint: String,
        generation: u64,
        cancellation: CancellationToken,
    },
    Ready {
        fingerprint: String,
        committed_members: BTreeSet<String>,
    },
    Retrying {
        fingerprint: String,
        deadline: Instant,
    },
    Conflict,
    Blocked {
        fingerprint: String,
        deadline: Instant,
    },
    BlockedReady {
        fingerprint: String,
        committed_members: BTreeSet<String>,
        deadline: Instant,
    },
}

struct DesiredGroup {
    generation: u64,
    retry_attempt: u32,
    cohorts: HashMap<String, BTreeSet<String>>,
    admission_tx: watch::Sender<Vec<u64>>,
    status: GroupStatus,
    /// The size of each cohort at the last reported conflict. A disagreement
    /// between workers persists until an operator resolves it, so the refusal
    /// is reported when the cohorts change rather than on every reconciliation
    /// pass. Sizes, not member lists, keep the retained state and the log line
    /// bounded while a conflicting cohort grows.
    reported_conflict: Option<BTreeMap<String, usize>>,
}

impl DesiredGroup {
    fn new() -> Self {
        let (admission_tx, _) = watch::channel(Vec::new());
        Self {
            generation: 0,
            retry_attempt: 0,
            cohorts: HashMap::new(),
            admission_tx,
            status: GroupStatus::Idle,
            reported_conflict: None,
        }
    }

    /// Swap in a fresh admission channel and hand back the outgoing one.
    ///
    /// The client updater loop exits when `admitted_ids.changed()` returns an
    /// error, so dropping the returned sender is what stops a client built for
    /// the previous cohort. Holding it until then still allows a final empty
    /// send to withdraw the instances it is serving.
    fn retire_admissions(&mut self) -> watch::Sender<Vec<u64>> {
        let (admission_tx, _) = watch::channel(Vec::new());
        std::mem::replace(&mut self.admission_tx, admission_tx)
    }

    fn insert(&mut self, instance: &DesiredInstance) {
        self.cohorts
            .entry(instance.fingerprint.clone())
            .or_default()
            .insert(instance.key.clone());
    }

    fn remove(&mut self, instance: &DesiredInstance) {
        let Some(cohort) = self.cohorts.get_mut(&instance.fingerprint) else {
            return;
        };
        cohort.remove(&instance.key);
        if cohort.is_empty() {
            self.cohorts.remove(&instance.fingerprint);
        }
    }
}

enum BuildOutcome<P> {
    Prepared(P),
    Failed(anyhow::Error),
    Cancelled,
}

struct BuildResult<P> {
    spec: GroupSpec,
    outcome: BuildOutcome<P>,
}

struct ReconciliationResult {
    revision: u64,
    instances: anyhow::Result<Vec<DiscoveryInstance>>,
}

pub(crate) struct ModelDiscoveryController<H: ControllerHost> {
    host: Arc<H>,
    desired: HashMap<String, DesiredInstance>,
    groups: HashMap<GroupKey, DesiredGroup>,
    revision: u64,
    instance_revisions: HashMap<String, u64>,
    builds: JoinSet<BuildResult<H::Prepared>>,
    reconciliations: JoinSet<ReconciliationResult>,
    active_builds: usize,
    max_concurrent_builds: usize,
    next_build_generation: u64,
}

impl<H: ControllerHost> ModelDiscoveryController<H> {
    pub(crate) fn new(host: Arc<H>) -> Self {
        Self::with_max_concurrent_builds(host, DEFAULT_MAX_CONCURRENT_BUILDS)
    }

    fn with_max_concurrent_builds(host: Arc<H>, max_concurrent_builds: usize) -> Self {
        Self {
            host,
            desired: HashMap::new(),
            groups: HashMap::new(),
            revision: 0,
            instance_revisions: HashMap::new(),
            builds: JoinSet::new(),
            reconciliations: JoinSet::new(),
            active_builds: 0,
            max_concurrent_builds: max_concurrent_builds.max(1),
            next_build_generation: 1,
        }
    }

    pub(crate) async fn run(
        mut self,
        mut discovery_stream: DiscoveryStream,
        namespace_filter: NamespaceFilter,
    ) {
        let mut reconciliation_interval = tokio::time::interval_at(
            Instant::now() + RECONCILIATION_INTERVAL,
            RECONCILIATION_INTERVAL,
        );
        reconciliation_interval.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);
        loop {
            self.start_queued_builds();
            let retry_deadline = self.next_retry_deadline();

            tokio::select! {
                event = discovery_stream.next() => {
                    let Some(event) = event else {
                        tracing::warn!(
                            "Model discovery stream ended; retaining committed serving state"
                        );
                        break;
                    };
                    match event {
                        Ok(event) => self.apply_event(event, &namespace_filter),
                        Err(error) => tracing::error!(%error, "Error in model discovery stream"),
                    }
                }
                result = self.builds.join_next(), if !self.builds.is_empty() => {
                    self.active_builds = self.active_builds.saturating_sub(1);
                    match result {
                        Some(Ok(result)) => self.apply_build_result(result),
                        Some(Err(error)) => tracing::error!(%error, "Model materialization task failed"),
                        None => {}
                    }
                }
                _ = reconciliation_interval.tick(), if self.reconciliations.is_empty() => {
                    self.start_reconciliation();
                }
                result = self.reconciliations.join_next(), if !self.reconciliations.is_empty() => {
                    match result {
                        Some(Ok(result)) => self.apply_reconciliation(result, &namespace_filter),
                        Some(Err(error)) => tracing::error!(%error, "Model reconciliation task failed"),
                        None => {}
                    }
                }
                _ = wait_for_deadline(retry_deadline), if retry_deadline.is_some() => {
                    self.release_due_retries();
                }
            }
        }

        self.shutdown_builds().await;
    }

    fn apply_event(&mut self, event: DiscoveryEvent, namespace_filter: &NamespaceFilter) {
        match event {
            DiscoveryEvent::Added(instance) => {
                match self.host.normalize(instance, namespace_filter) {
                    Ok(Some(instance)) => self.apply_added(instance),
                    Ok(None) => false,
                    Err(error) => {
                        tracing::error!(
                            error = format!("{error:#}"),
                            "Rejected model discovery update; preserving last valid desired state"
                        );
                        false
                    }
                }
            }
            DiscoveryEvent::ModelTaintsUpdated(update) => {
                tracing::debug!(
                    instance_id = update.id.instance_id,
                    "Ignoring model taint update in structural model discovery"
                );
                false
            }
            DiscoveryEvent::Removed(DiscoveryInstanceId::Model(mcid)) => {
                self.apply_removed(&mcid.to_path())
            }
            DiscoveryEvent::Removed(_) => {
                tracing::error!("Unexpected non-model removal in model discovery stream");
                false
            }
        };
    }

    fn apply_added(&mut self, instance: DesiredInstance) -> bool {
        if let Some(existing) = self.desired.get(&instance.key) {
            if existing.fingerprint == instance.fingerprint
                && existing.projection_fingerprint == instance.projection_fingerprint
            {
                return false;
            }
            if existing.materializes_worker_set()
                && (existing.group_key != instance.group_key
                    || existing.fingerprint != instance.fingerprint)
            {
                tracing::error!(
                    instance = instance.key,
                    existing_group = %existing.group_key.id(),
                    candidate_group = %instance.group_key.id(),
                    "Rejected an in-place materialization change; worker instance paths identify immutable incarnations"
                );
                return false;
            }
        }

        let group_key = instance.group_key.clone();
        let endpoint_id = instance.endpoint_id.clone();
        let instance_id = instance.mcid.instance_id;
        let instance_key = instance.key.clone();
        let materializes_worker_set = instance.materializes_worker_set();
        if materializes_worker_set {
            self.groups
                .entry(group_key.clone())
                .or_insert_with(DesiredGroup::new)
                .insert(&instance);
        }
        self.desired.insert(instance.key.clone(), instance);
        self.record_mutation(instance_key);

        if materializes_worker_set {
            self.reconcile_group(&group_key, true);
        } else {
            for key in self.materialization_groups_for(&endpoint_id, instance_id) {
                self.reconcile_group(&key, true);
            }
        }
        true
    }

    fn apply_removed(&mut self, instance_key: &str) -> bool {
        let removed = self.desired.remove(instance_key);
        self.record_mutation(instance_key.to_string());
        let Some(instance) = removed else {
            return false;
        };
        let affected_groups = if instance.materializes_worker_set() {
            vec![instance.group_key.clone()]
        } else {
            self.materialization_groups_for(&instance.endpoint_id, instance.mcid.instance_id)
        };
        if instance.materializes_worker_set()
            && let Some(group) = self.groups.get_mut(&instance.group_key)
        {
            group.remove(&instance);
        }
        for key in affected_groups {
            self.reconcile_group(&key, true);
        }
        true
    }

    fn reconcile_group(&mut self, key: &GroupKey, desired_changed: bool) {
        let Some(mut group) = self.groups.remove(key) else {
            return;
        };
        let old_status = std::mem::replace(&mut group.status, GroupStatus::Idle);

        if group.cohorts.is_empty() {
            group.admission_tx.send_replace(Vec::new());
            cancel_build(&old_status);
            if status_has_commit(&old_status) {
                self.host.remove_group(key);
            }
            return;
        }

        if group.cohorts.len() > 1 {
            report_conflicting_cohorts(key, &mut group);
            group.admission_tx.send_replace(Vec::new());
            cancel_build(&old_status);
            if status_has_commit(&old_status) {
                self.host.remove_group(key);
            }
            // A client built for the withdrawn cohort can outlive
            // `remove_group`, because a published catalog snapshot holds an
            // `Arc<WorkerSet>` past the removal. Retiring the channel after the
            // withdrawing send closes that client's receiver, so it stops at
            // its last admitted set instead of later observing the instance IDs
            // of whichever cohort resolves the conflict and dispatching a
            // pipeline built for one deployment card at workers advertising
            // another.
            if status_fingerprint(&old_status).is_some() {
                drop(group.retire_admissions());
            }
            if !matches!(old_status, GroupStatus::Conflict) {
                group.generation = group.generation.wrapping_add(1);
                group.retry_attempt = 0;
            }
            group.status = GroupStatus::Conflict;
            self.groups.insert(key.clone(), group);
            return;
        }

        group.reported_conflict = None;
        let (fingerprint, member_keys) = group
            .cohorts
            .iter()
            .next()
            .map(|(fingerprint, members)| (fingerprint.clone(), members.clone()))
            .expect("non-empty group has one cohort");
        let members = self.members(&member_keys);
        let admitted = admitted_ids(&members);

        if !matches!(
            &old_status,
            GroupStatus::Ready { .. } | GroupStatus::BlockedReady { .. }
        ) {
            group.admission_tx.send_replace(admitted);
        }

        group.status = match old_status {
            GroupStatus::Ready {
                fingerprint: ready_fingerprint,
                committed_members,
            }
            | GroupStatus::BlockedReady {
                fingerprint: ready_fingerprint,
                committed_members,
                ..
            } if ready_fingerprint == fingerprint => {
                let current_members = member_keys;
                let old_admitted = group.admission_tx.borrow().clone();
                let new_admitted = admitted_ids(&members);
                let new_admitted_set = new_admitted.iter().copied().collect::<HashSet<_>>();
                group.admission_tx.send_replace(
                    old_admitted
                        .iter()
                        .copied()
                        .filter(|id| new_admitted_set.contains(id))
                        .collect(),
                );
                let adapters = self.adapters_for_members(&current_members);
                match self.host.replace_group(key, &members, &adapters) {
                    Ok(()) => {
                        group.admission_tx.send_replace(new_admitted);
                        group.retry_attempt = 0;
                        GroupStatus::Ready {
                            fingerprint,
                            committed_members: current_members,
                        }
                    }
                    Err(error) if current_members == committed_members => {
                        group.admission_tx.send_replace(old_admitted);
                        group.retry_attempt = group.retry_attempt.saturating_add(1);
                        let delay = retry_delay(group.retry_attempt);
                        tracing::warn!(
                            group = %key.id(),
                            error = format!("{error:#}"),
                            retry_ms = delay.as_millis(),
                            "Discovery-group replacement blocked; retaining the last safe commit"
                        );
                        GroupStatus::BlockedReady {
                            fingerprint,
                            committed_members,
                            deadline: Instant::now() + delay,
                        }
                    }
                    Err(error) => {
                        group.admission_tx.send_replace(Vec::new());
                        self.host.remove_group(key);
                        group.generation = group.generation.wrapping_add(1);
                        group.retry_attempt = group.retry_attempt.saturating_add(1);
                        tracing::warn!(
                            group = %key.id(),
                            error = format!("{error:#}"),
                            "Discovery-group membership replacement failed; withdrawing stale commit"
                        );
                        GroupStatus::Blocked {
                            fingerprint,
                            deadline: Instant::now() + retry_delay(group.retry_attempt),
                        }
                    }
                }
            }
            GroupStatus::Building {
                fingerprint: building_fingerprint,
                generation,
                cancellation,
            } if building_fingerprint == fingerprint => GroupStatus::Building {
                fingerprint,
                generation,
                cancellation,
            },
            GroupStatus::Queued {
                fingerprint: queued_fingerprint,
            } if queued_fingerprint == fingerprint => GroupStatus::Queued { fingerprint },
            GroupStatus::Retrying {
                fingerprint: retry_fingerprint,
                deadline,
            } if retry_fingerprint == fingerprint && !desired_changed => GroupStatus::Retrying {
                fingerprint,
                deadline,
            },
            GroupStatus::Blocked {
                fingerprint: blocked_fingerprint,
                deadline,
            } if blocked_fingerprint == fingerprint && !desired_changed => GroupStatus::Blocked {
                fingerprint,
                deadline,
            },
            previous => {
                cancel_build(&previous);
                if status_has_commit(&previous) {
                    group.admission_tx.send_replace(Vec::new());
                    self.host.remove_group(key);
                    group.admission_tx.send_replace(admitted_ids(&members));
                }
                group.generation = group.generation.wrapping_add(1);
                group.retry_attempt = 0;
                GroupStatus::Queued { fingerprint }
            }
        };
        self.groups.insert(key.clone(), group);
    }

    fn members(&self, member_keys: &BTreeSet<String>) -> Vec<DesiredInstance> {
        member_keys
            .iter()
            .filter_map(|key| self.desired.get(key).cloned())
            .collect()
    }

    fn adapters_for_members(&self, member_keys: &BTreeSet<String>) -> Vec<DesiredInstance> {
        let physical_members = member_keys
            .iter()
            .filter_map(|key| self.desired.get(key))
            .map(|member| (member.endpoint_id.clone(), member.mcid.instance_id))
            .collect::<HashSet<_>>();
        let mut adapters = self
            .desired
            .values()
            .filter(|instance| {
                !instance.materializes_worker_set()
                    && physical_members
                        .contains(&(instance.endpoint_id.clone(), instance.mcid.instance_id))
            })
            .cloned()
            .collect::<Vec<_>>();
        adapters.sort_by(|left, right| left.key.cmp(&right.key));
        adapters
    }

    fn materialization_groups_for(
        &self,
        endpoint_id: &EndpointId,
        instance_id: u64,
    ) -> Vec<GroupKey> {
        self.desired
            .values()
            .filter(|instance| {
                instance.materializes_worker_set()
                    && &instance.endpoint_id == endpoint_id
                    && instance.mcid.instance_id == instance_id
            })
            .map(|instance| instance.group_key.clone())
            .collect::<HashSet<_>>()
            .into_iter()
            .collect()
    }

    fn record_mutation(&mut self, instance_key: String) {
        self.revision = self.revision.wrapping_add(1);
        self.instance_revisions.insert(instance_key, self.revision);
    }

    fn start_queued_builds(&mut self) {
        if self.active_builds >= self.max_concurrent_builds {
            return;
        }
        let mut queued = self
            .groups
            .iter()
            .filter_map(|(key, group)| {
                matches!(group.status, GroupStatus::Queued { .. }).then_some(key.clone())
            })
            .collect::<Vec<_>>();
        queued.sort();

        for key in queued {
            if self.active_builds >= self.max_concurrent_builds {
                break;
            }
            let Some(group) = self.groups.get_mut(&key) else {
                continue;
            };
            let GroupStatus::Queued { fingerprint } = &group.status else {
                continue;
            };
            let fingerprint = fingerprint.clone();
            let Some(member_key) = group
                .cohorts
                .get(&fingerprint)
                .and_then(|members| members.first())
            else {
                continue;
            };
            let Some(representative) = self.desired.get(member_key).cloned() else {
                continue;
            };

            let cancellation = CancellationToken::new();
            let generation = self.next_build_generation;
            self.next_build_generation = self.next_build_generation.wrapping_add(1).max(1);
            let spec = GroupSpec {
                key: key.clone(),
                fingerprint: fingerprint.clone(),
                generation,
                representative,
            };
            group.status = GroupStatus::Building {
                fingerprint,
                generation,
                cancellation: cancellation.clone(),
            };

            let host = self.host.clone();
            let admitted_ids = group.admission_tx.subscribe();
            let task_spec = spec.clone();
            self.builds.spawn(async move {
                let future = AssertUnwindSafe(async {
                    tokio::select! {
                        biased;
                        _ = cancellation.cancelled() => BuildOutcome::Cancelled,
                        result = host.prepare(task_spec.clone(), admitted_ids, cancellation.clone()) => {
                            match result {
                                Ok(prepared) => BuildOutcome::Prepared(prepared),
                                Err(error) => BuildOutcome::Failed(error),
                            }
                        }
                    }
                });
                let outcome = match future.catch_unwind().await {
                    Ok(outcome) => outcome,
                    Err(_) => BuildOutcome::Failed(anyhow::anyhow!(
                        "model materialization panicked"
                    )),
                };
                BuildResult {
                    spec: task_spec,
                    outcome,
                }
            });
            self.active_builds += 1;
        }
    }

    fn apply_build_result(&mut self, result: BuildResult<H::Prepared>) {
        let Some(mut group) = self.groups.remove(&result.spec.key) else {
            if let BuildOutcome::Prepared(prepared) = result.outcome {
                self.host.discard_prepared(prepared);
            }
            return;
        };
        let is_current = matches!(
            &group.status,
            GroupStatus::Building {
                fingerprint,
                generation,
                ..
            } if fingerprint == &result.spec.fingerprint && *generation == result.spec.generation
        ) && group.cohorts.len() == 1
            && group.cohorts.contains_key(&result.spec.fingerprint);
        if !is_current {
            if let BuildOutcome::Prepared(prepared) = result.outcome {
                self.host.discard_prepared(prepared);
            }
            self.groups.insert(result.spec.key, group);
            return;
        }

        match result.outcome {
            BuildOutcome::Prepared(prepared) => {
                let member_keys = group
                    .cohorts
                    .get(&result.spec.fingerprint)
                    .cloned()
                    .unwrap_or_default();
                let members = self.members(&member_keys);
                let adapters = self.adapters_for_members(&member_keys);
                group.admission_tx.send_replace(admitted_ids(&members));
                match self
                    .host
                    .commit_group(&result.spec, prepared, &members, &adapters)
                {
                    Ok(()) => {
                        group.retry_attempt = 0;
                        group.status = GroupStatus::Ready {
                            fingerprint: result.spec.fingerprint,
                            committed_members: member_keys,
                        };
                    }
                    Err(error) => {
                        group.admission_tx.send_replace(Vec::new());
                        tracing::warn!(
                            group = %result.spec.key.id(),
                            error = format!("{error:#}"),
                            "Model materialization is blocked at commit"
                        );
                        group.retry_attempt = group.retry_attempt.saturating_add(1);
                        group.status = GroupStatus::Blocked {
                            fingerprint: result.spec.fingerprint,
                            deadline: Instant::now() + retry_delay(group.retry_attempt),
                        };
                    }
                }
            }
            BuildOutcome::Failed(error) => {
                group.retry_attempt = group.retry_attempt.saturating_add(1);
                let delay = retry_delay(group.retry_attempt);
                tracing::warn!(
                    group = %result.spec.key.id(),
                    attempt = group.retry_attempt,
                    retry_ms = delay.as_millis(),
                    error = format!("{error:#}"),
                    "Model materialization failed; scheduling retry"
                );
                group.status = GroupStatus::Retrying {
                    fingerprint: result.spec.fingerprint,
                    deadline: Instant::now() + delay,
                };
            }
            BuildOutcome::Cancelled => {
                group.status = GroupStatus::Queued {
                    fingerprint: result.spec.fingerprint,
                };
            }
        }
        self.groups.insert(result.spec.key, group);
    }

    fn next_retry_deadline(&self) -> Option<Instant> {
        self.groups
            .values()
            .filter_map(|group| match group.status {
                GroupStatus::Retrying { deadline, .. }
                | GroupStatus::Blocked { deadline, .. }
                | GroupStatus::BlockedReady { deadline, .. } => Some(deadline),
                _ => None,
            })
            .min()
    }

    /// Move groups whose retry deadline has passed back into the run loop.
    ///
    /// A released retry reconciles rather than requeueing the fingerprint it
    /// stored when it failed. Cohort membership can have moved under a group
    /// while it waited out its backoff, and the released attempt has to match
    /// the cohorts the group holds now — which may be none, one, or a
    /// disagreement that withdraws the model.
    fn release_due_retries(&mut self) {
        let now = Instant::now();
        let mut retained_retries = Vec::new();
        let mut released_retries = Vec::new();
        for (key, group) in &mut self.groups {
            let (fingerprint, deadline) = match &group.status {
                GroupStatus::Retrying {
                    fingerprint,
                    deadline,
                }
                | GroupStatus::Blocked {
                    fingerprint,
                    deadline,
                } => (fingerprint, deadline),
                GroupStatus::BlockedReady {
                    fingerprint,
                    committed_members,
                    deadline,
                } if *deadline <= now => {
                    group.status = GroupStatus::Ready {
                        fingerprint: fingerprint.clone(),
                        committed_members: committed_members.clone(),
                    };
                    retained_retries.push(key.clone());
                    continue;
                }
                _ => continue,
            };
            if *deadline <= now {
                group.status = GroupStatus::Queued {
                    fingerprint: fingerprint.clone(),
                };
                released_retries.push(key.clone());
            }
        }
        for key in retained_retries.into_iter().chain(released_retries) {
            self.reconcile_group(&key, false);
        }
    }

    async fn shutdown_builds(&mut self) {
        for group in self.groups.values_mut() {
            group.admission_tx.send_replace(Vec::new());
            cancel_build(&group.status);
        }
        self.builds.abort_all();
        while self.builds.join_next().await.is_some() {}
        self.reconciliations.abort_all();
        while self.reconciliations.join_next().await.is_some() {}
    }

    fn start_reconciliation(&mut self) {
        let host = self.host.clone();
        let revision = self.revision;
        self.reconciliations.spawn(async move {
            ReconciliationResult {
                revision,
                instances: host.list_instances().await,
            }
        });
    }

    fn apply_reconciliation(
        &mut self,
        result: ReconciliationResult,
        namespace_filter: &NamespaceFilter,
    ) {
        let instances = match result.instances {
            Ok(instances) => instances,
            Err(error) => {
                tracing::warn!(error = format!("{error:#}"), "Model reconciliation failed");
                return;
            }
        };
        let mut observed = HashSet::new();
        let mut normalized = Vec::new();
        for instance in instances {
            let DiscoveryInstanceId::Model(mcid) = instance.id() else {
                continue;
            };
            let key = mcid.to_path();
            observed.insert(key.clone());
            if self
                .instance_revisions
                .get(&key)
                .is_some_and(|revision| *revision > result.revision)
            {
                continue;
            }
            match self.host.normalize(instance, namespace_filter) {
                Ok(Some(instance)) => normalized.push(instance),
                Ok(None) => {}
                Err(error) => tracing::warn!(
                    instance = key,
                    error = format!("{error:#}"),
                    "Rejected model from reconciliation snapshot"
                ),
            }
        }

        for instance in normalized {
            self.apply_added(instance);
        }
        let removals = self
            .desired
            .keys()
            .filter(|key| {
                !observed.contains(*key)
                    && self
                        .instance_revisions
                        .get(*key)
                        .is_none_or(|revision| *revision <= result.revision)
            })
            .cloned()
            .collect::<Vec<_>>();
        for key in removals {
            self.apply_removed(&key);
        }
        self.instance_revisions.retain(|key, revision| {
            self.desired.contains_key(key) || observed.contains(key) || *revision > result.revision
        });
    }
}

fn admitted_ids(members: &[DesiredInstance]) -> Vec<u64> {
    let mut ids = members
        .iter()
        .map(|member| member.mcid.instance_id)
        .collect::<Vec<_>>();
    ids.sort_unstable();
    ids.dedup();
    ids
}

/// How many conflicting cohorts one refusal log line names. The rest are
/// covered by the cohort and worker counts, which keeps the line's size
/// independent of how many workers are misconfigured.
const CONFLICTING_COHORT_SAMPLE: usize = 3;

/// Report, at most once per distinct set of cohort sizes, that a worker set
/// disagrees about its model and is therefore serving nothing.
fn report_conflicting_cohorts(key: &GroupKey, group: &mut DesiredGroup) {
    let cohort_sizes = group
        .cohorts
        .iter()
        .map(|(fingerprint, members)| (fingerprint.clone(), members.len()))
        .collect::<BTreeMap<_, _>>();
    if group.reported_conflict.as_ref() == Some(&cohort_sizes) {
        return;
    }
    let conflicting_workers = cohort_sizes.values().sum::<usize>();
    let cohort_sample = cohort_sizes
        .iter()
        .take(CONFLICTING_COHORT_SAMPLE)
        .map(|(fingerprint, size)| {
            let representative = group
                .cohorts
                .get(fingerprint)
                .and_then(|members| members.first())
                .map(String::as_str)
                .unwrap_or("unknown");
            format!("{fingerprint} ({size} workers, e.g. {representative})")
        })
        .collect::<Vec<_>>()
        .join("; ");
    tracing::error!(
        model_name = %key.model_name,
        worker_set = %key.worker_set_key,
        conflicting_cohorts = cohort_sizes.len(),
        conflicting_workers,
        cohort_sample = %cohort_sample,
        "Workers in this worker set published model deployment cards with different checksums. \
         The model is withdrawn and no worker in this set receives traffic until they agree. \
         Restate the configuration so every worker in this worker set advertises the same card, \
         then drain the workers of this worker set that still disagree."
    );
    group.reported_conflict = Some(cohort_sizes);
}

/// The fingerprint this group is working towards, committed or not.
///
/// A group that carries one has handed its admission channel to a build or a
/// commit, which is what tells the caller to retire that channel when the
/// cohort behind it is withdrawn.
fn status_fingerprint(status: &GroupStatus) -> Option<&str> {
    match status {
        GroupStatus::Queued { fingerprint }
        | GroupStatus::Building { fingerprint, .. }
        | GroupStatus::Ready { fingerprint, .. }
        | GroupStatus::Retrying { fingerprint, .. }
        | GroupStatus::Blocked { fingerprint, .. }
        | GroupStatus::BlockedReady { fingerprint, .. } => Some(fingerprint),
        GroupStatus::Idle | GroupStatus::Conflict => None,
    }
}

fn cancel_build(status: &GroupStatus) {
    if let GroupStatus::Building { cancellation, .. } = status {
        cancellation.cancel();
    }
}

fn status_has_commit(status: &GroupStatus) -> bool {
    matches!(
        status,
        GroupStatus::Ready { .. } | GroupStatus::BlockedReady { .. }
    )
}

async fn wait_for_deadline(deadline: Option<Instant>) {
    match deadline {
        Some(deadline) => tokio::time::sleep_until(deadline).await,
        None => std::future::pending().await,
    }
}

fn retry_delay(attempt: u32) -> Duration {
    let base_seconds = match attempt {
        0 | 1 => 1,
        2 => 2,
        3 => 4,
        4 => 8,
        5 => 16,
        _ => 30,
    };
    Duration::from_secs(base_seconds)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::{
        Mutex,
        atomic::{AtomicUsize, Ordering},
    };
    use tokio::sync::{Semaphore, mpsc};

    struct Prepared(u64);

    struct FakeHost {
        starts: AtomicUsize,
        failures: AtomicUsize,
        commit_failures: AtomicUsize,
        replace_failures: AtomicUsize,
        start_tx: mpsc::UnboundedSender<GroupSpec>,
        release: Semaphore,
        committed: Mutex<HashMap<String, BTreeSet<String>>>,
        adapters: Mutex<HashMap<String, BTreeSet<String>>>,
        adapter_projections: Mutex<HashMap<String, HashMap<String, String>>>,
        replacements: AtomicUsize,
        removed_groups: AtomicUsize,
        discarded: AtomicUsize,
    }

    impl FakeHost {
        fn new() -> (Arc<Self>, mpsc::UnboundedReceiver<GroupSpec>) {
            let (start_tx, start_rx) = mpsc::unbounded_channel();
            (
                Arc::new(Self {
                    starts: AtomicUsize::new(0),
                    failures: AtomicUsize::new(0),
                    commit_failures: AtomicUsize::new(0),
                    replace_failures: AtomicUsize::new(0),
                    start_tx,
                    release: Semaphore::new(0),
                    committed: Mutex::new(HashMap::new()),
                    adapters: Mutex::new(HashMap::new()),
                    adapter_projections: Mutex::new(HashMap::new()),
                    replacements: AtomicUsize::new(0),
                    removed_groups: AtomicUsize::new(0),
                    discarded: AtomicUsize::new(0),
                }),
                start_rx,
            )
        }

        fn members(&self, key: &GroupKey) -> BTreeSet<String> {
            self.committed
                .lock()
                .unwrap()
                .get(&key.id())
                .cloned()
                .unwrap_or_default()
        }

        fn adapters(&self, key: &GroupKey) -> BTreeSet<String> {
            self.adapters
                .lock()
                .unwrap()
                .get(&key.id())
                .cloned()
                .unwrap_or_default()
        }

        fn adapter_projection(&self, key: &GroupKey, adapter_key: &str) -> Option<String> {
            self.adapter_projections
                .lock()
                .unwrap()
                .get(&key.id())
                .and_then(|adapters| adapters.get(adapter_key))
                .cloned()
        }

        fn store_adapters(&self, key: &GroupKey, adapters: &[DesiredInstance]) {
            self.adapters.lock().unwrap().insert(
                key.id(),
                adapters.iter().map(|adapter| adapter.key.clone()).collect(),
            );
            self.adapter_projections.lock().unwrap().insert(
                key.id(),
                adapters
                    .iter()
                    .map(|adapter| (adapter.key.clone(), adapter.projection_fingerprint.clone()))
                    .collect(),
            );
        }
    }

    #[async_trait]
    impl ControllerHost for FakeHost {
        type Prepared = Prepared;

        fn normalize(
            &self,
            instance: DiscoveryInstance,
            namespace_filter: &NamespaceFilter,
        ) -> anyhow::Result<Option<DesiredInstance>> {
            let DiscoveryInstance::Model {
                namespace,
                component,
                endpoint,
                instance_id,
                card_json,
                model_suffix,
            } = instance
            else {
                return Ok(None);
            };
            if !namespace_filter.matches(&namespace) {
                return Ok(None);
            }
            let card: ModelDeploymentCard = serde_json::from_value(card_json)?;
            let mcid = ModelCardInstanceId {
                namespace: namespace.clone(),
                component: component.clone(),
                endpoint: endpoint.clone(),
                instance_id,
                model_suffix,
            };
            Ok(Some(DesiredInstance {
                key: mcid.to_path(),
                mcid,
                endpoint_id: EndpointId {
                    namespace,
                    component,
                    name: endpoint,
                },
                group_key: group_key(),
                card,
                fingerprint: "spec".to_string(),
                projection_fingerprint: "projection".to_string(),
            }))
        }

        async fn prepare(
            &self,
            spec: GroupSpec,
            _admitted_ids: watch::Receiver<Vec<u64>>,
            _cancellation: CancellationToken,
        ) -> anyhow::Result<Self::Prepared> {
            let build = self.starts.fetch_add(1, Ordering::SeqCst) as u64;
            self.start_tx.send(spec).unwrap();
            self.release.acquire().await.unwrap().forget();
            if self
                .failures
                .fetch_update(Ordering::SeqCst, Ordering::SeqCst, |remaining| {
                    remaining.checked_sub(1)
                })
                .is_ok()
            {
                anyhow::bail!("injected materialization failure");
            }
            Ok(Prepared(build))
        }

        fn commit_group(
            &self,
            spec: &GroupSpec,
            prepared: Self::Prepared,
            members: &[DesiredInstance],
            adapters: &[DesiredInstance],
        ) -> anyhow::Result<()> {
            let Prepared(_build) = prepared;
            if self
                .commit_failures
                .fetch_update(Ordering::SeqCst, Ordering::SeqCst, |remaining| {
                    remaining.checked_sub(1)
                })
                .is_ok()
            {
                anyhow::bail!("injected commit conflict");
            }
            self.committed.lock().unwrap().insert(
                spec.key.id(),
                members.iter().map(|member| member.key.clone()).collect(),
            );
            self.store_adapters(&spec.key, adapters);
            Ok(())
        }

        fn replace_group(
            &self,
            key: &GroupKey,
            members: &[DesiredInstance],
            adapters: &[DesiredInstance],
        ) -> anyhow::Result<()> {
            if self
                .replace_failures
                .fetch_update(Ordering::SeqCst, Ordering::SeqCst, |remaining| {
                    remaining.checked_sub(1)
                })
                .is_ok()
            {
                anyhow::bail!("injected replacement conflict");
            }
            self.replacements.fetch_add(1, Ordering::SeqCst);
            self.committed.lock().unwrap().insert(
                key.id(),
                members.iter().map(|member| member.key.clone()).collect(),
            );
            self.store_adapters(key, adapters);
            Ok(())
        }

        fn remove_group(&self, key: &GroupKey) {
            self.committed.lock().unwrap().remove(&key.id());
            self.adapters.lock().unwrap().remove(&key.id());
            self.adapter_projections.lock().unwrap().remove(&key.id());
            self.removed_groups.fetch_add(1, Ordering::SeqCst);
        }

        fn discard_prepared(&self, prepared: Self::Prepared) {
            let Prepared(_build) = prepared;
            self.discarded.fetch_add(1, Ordering::SeqCst);
        }

        async fn list_instances(&self) -> anyhow::Result<Vec<DiscoveryInstance>> {
            Ok(Vec::new())
        }
    }

    fn group_key() -> GroupKey {
        GroupKey {
            model_name: "model".to_string(),
            worker_set_key: "group".to_string(),
        }
    }

    fn instance(id: u64, fingerprint: &str) -> DesiredInstance {
        let mcid = ModelCardInstanceId {
            namespace: "namespace".to_string(),
            component: "worker".to_string(),
            endpoint: "generate".to_string(),
            instance_id: id,
            model_suffix: None,
        };
        DesiredInstance {
            key: mcid.to_path(),
            mcid,
            endpoint_id: EndpointId {
                namespace: "namespace".to_string(),
                component: "worker".to_string(),
                name: "generate".to_string(),
            },
            card: ModelDeploymentCard::with_name_only("model"),
            group_key: group_key(),
            fingerprint: fingerprint.to_string(),
            projection_fingerprint: fingerprint.to_string(),
        }
    }

    fn discovery_instance(instance: &DesiredInstance) -> DiscoveryInstance {
        DiscoveryInstance::Model {
            namespace: instance.mcid.namespace.clone(),
            component: instance.mcid.component.clone(),
            endpoint: instance.mcid.endpoint.clone(),
            instance_id: instance.mcid.instance_id,
            card_json: serde_json::to_value(&instance.card).unwrap(),
            model_suffix: instance.mcid.model_suffix.clone(),
        }
    }

    async fn finish_build(controller: &mut ModelDiscoveryController<FakeHost>) {
        let result = controller.builds.join_next().await.unwrap().unwrap();
        controller.active_builds -= 1;
        controller.apply_build_result(result);
    }

    /// Run the controller's build side to a standstill. A cohort change can
    /// leave a cancelled build and its replacement outstanding together, so
    /// join every build and restart queued work after each join.
    async fn drain_builds(host: &FakeHost, controller: &mut ModelDiscoveryController<FakeHost>) {
        while !controller.builds.is_empty() {
            host.release.add_permits(1);
            finish_build(controller).await;
            controller.start_queued_builds();
        }
    }

    /// Register two workers of one worker set the way the run loop observes
    /// them: `ModelDiscoveryController::run` starts queued builds at the top of
    /// every iteration and applies one discovery event per iteration, so the
    /// second worker is always seen after the first worker's build has started.
    async fn register_in_arrival_order(
        first: &DesiredInstance,
        second: &DesiredInstance,
    ) -> (
        Arc<FakeHost>,
        ModelDiscoveryController<FakeHost>,
        mpsc::UnboundedReceiver<GroupSpec>,
    ) {
        let (host, starts) = FakeHost::new();
        let mut controller = ModelDiscoveryController::new(host.clone());

        controller.apply_added(first.clone());
        controller.start_queued_builds();
        controller.apply_added(second.clone());
        controller.start_queued_builds();
        drain_builds(&host, &mut controller).await;

        (host, controller, starts)
    }

    #[tokio::test]
    async fn membership_churn_keeps_one_build_and_commits_latest_members() {
        let (host, mut starts) = FakeHost::new();
        let mut controller = ModelDiscoveryController::new(host.clone());
        let first = instance(1, "same");
        let second = instance(2, "same");

        controller.apply_added(first.clone());
        controller.apply_added(second.clone());
        controller.start_queued_builds();
        starts.recv().await.unwrap();

        controller.apply_removed(&first.key);
        host.release.add_permits(1);
        finish_build(&mut controller).await;

        assert_eq!(host.starts.load(Ordering::SeqCst), 1);
        assert_eq!(
            host.members(&group_key()),
            BTreeSet::from([second.key.clone()])
        );

        let third = instance(3, "same");
        controller.apply_added(third.clone());
        assert_eq!(
            host.members(&group_key()),
            BTreeSet::from([second.key.clone(), third.key.clone()])
        );
        controller.apply_removed(&third.key);
        assert_eq!(host.members(&group_key()), BTreeSet::from([second.key]));
        assert_eq!(host.starts.load(Ordering::SeqCst), 1);
    }

    #[tokio::test]
    async fn duplicate_and_in_place_mutation_preserve_first_valid_incarnation() {
        let (host, mut starts) = FakeHost::new();
        let mut controller = ModelDiscoveryController::new(host.clone());
        let first = instance(1, "first-spec");
        let mutation = instance(1, "different-spec");

        assert!(controller.apply_added(first.clone()));
        assert!(!controller.apply_added(first.clone()));
        assert!(!controller.apply_added(mutation));
        controller.start_queued_builds();
        starts.recv().await.unwrap();
        host.release.add_permits(1);
        finish_build(&mut controller).await;

        assert_eq!(host.members(&group_key()), BTreeSet::from([first.key]));
        assert_eq!(host.starts.load(Ordering::SeqCst), 1);
    }

    fn admitted(controller: &ModelDiscoveryController<FakeHost>) -> Vec<u64> {
        controller
            .groups
            .get(&group_key())
            .expect("the group is still tracked")
            .admission_tx
            .borrow()
            .clone()
    }

    #[tokio::test]
    async fn conflict_fails_ready_group_closed_and_recovers_after_clear() {
        let (host, mut starts) = FakeHost::new();
        let mut controller = ModelDiscoveryController::new(host.clone());
        let compatible = instance(1, "first-spec");
        controller.apply_added(compatible.clone());
        controller.start_queued_builds();
        starts.recv().await.unwrap();
        host.release.add_permits(1);
        finish_build(&mut controller).await;
        assert!(!host.members(&group_key()).is_empty());

        let conflicting = instance(2, "second-spec");
        controller.apply_added(conflicting.clone());
        assert!(host.members(&group_key()).is_empty());
        assert_eq!(host.removed_groups.load(Ordering::SeqCst), 1);
        assert_eq!(admitted(&controller), Vec::<u64>::new());

        controller.apply_removed(&conflicting.key);
        controller.start_queued_builds();
        starts.recv().await.unwrap();
        host.release.add_permits(1);
        finish_build(&mut controller).await;
        assert_eq!(host.members(&group_key()), BTreeSet::from([compatible.key]));
    }

    #[tokio::test]
    async fn a_conflict_retires_the_admission_channel_of_the_withdrawn_cohort() {
        let (host, mut starts) = FakeHost::new();
        let mut controller = ModelDiscoveryController::new(host.clone());
        let committed = instance(1, "first-spec");
        controller.apply_added(committed.clone());
        controller.start_queued_builds();
        starts.recv().await.unwrap();
        host.release.add_permits(1);
        finish_build(&mut controller).await;

        // A client built for the withdrawn cohort keeps watching the channel it
        // was handed. A catalog snapshot can hold it alive past `remove_group`,
        // so it must never observe the instance IDs of whichever cohort
        // resolves the conflict.
        let retired = controller
            .groups
            .get(&group_key())
            .expect("the group is still tracked")
            .admission_tx
            .subscribe();

        let conflicting = instance(2, "second-spec");
        controller.apply_added(conflicting.clone());
        assert_eq!(*retired.borrow(), Vec::<u64>::new());
        // The updater loop exits on this error, which is how the retired client
        // stops instead of following the survivor.
        assert!(retired.has_changed().is_err());

        // The survivor is served on a fresh channel once the disagreement
        // clears, and the retired receiver never sees it.
        controller.apply_removed(&conflicting.key);
        controller.start_queued_builds();
        starts.recv().await.unwrap();
        host.release.add_permits(1);
        finish_build(&mut controller).await;
        assert_eq!(admitted(&controller), vec![committed.mcid.instance_id]);
        assert_eq!(*retired.borrow(), Vec::<u64>::new());
    }

    #[tokio::test]
    async fn conflict_during_build_cancels_without_publishing_either_cohort() {
        let (host, mut starts) = FakeHost::new();
        let mut controller = ModelDiscoveryController::new(host.clone());
        let first = instance(1, "first-spec");
        let conflicting = instance(2, "second-spec");
        controller.apply_added(first.clone());
        controller.start_queued_builds();
        starts.recv().await.unwrap();

        controller.apply_added(conflicting.clone());
        finish_build(&mut controller).await;
        assert!(host.members(&group_key()).is_empty());

        controller.apply_removed(&conflicting.key);
        controller.start_queued_builds();
        starts.recv().await.unwrap();
        host.release.add_permits(1);
        finish_build(&mut controller).await;
        assert_eq!(host.members(&group_key()), BTreeSet::from([first.key]));
    }

    #[tokio::test]
    async fn simultaneous_mixed_checksums_register_no_model_until_they_agree() {
        let self_hosted = instance(1, "self-hosted-spec");
        let fallback = instance(2, "fallback-spec");
        let (host, mut controller, _starts) =
            register_in_arrival_order(&self_hosted, &fallback).await;

        // Two workers of one worker set that disagree about the deployment card
        // serve nothing. Admitting either cohort would decide, from one
        // frontend's partial view of a rollout, which materialization the whole
        // load balancer serves.
        assert!(host.members(&group_key()).is_empty());
        assert_eq!(admitted(&controller), Vec::<u64>::new());

        // The disagreement is not fatal to the worker set: whichever cohort is
        // left once the other drains takes the model.
        let starts_before = host.starts.load(Ordering::SeqCst);
        controller.apply_removed(&fallback.key);
        controller.start_queued_builds();
        drain_builds(&host, &mut controller).await;
        assert_eq!(
            host.members(&group_key()),
            BTreeSet::from([self_hosted.key])
        );
        assert_eq!(host.starts.load(Ordering::SeqCst), starts_before + 1);
    }

    #[tokio::test]
    async fn further_conflicting_arrivals_do_not_republish_the_worker_set() {
        let (host, mut starts) = FakeHost::new();
        let mut controller = ModelDiscoveryController::new(host.clone());
        let committed = instance(1, "first-spec");
        controller.apply_added(committed.clone());
        controller.start_queued_builds();
        starts.recv().await.unwrap();
        host.release.add_permits(1);
        finish_build(&mut controller).await;

        controller.apply_added(instance(2, "second-spec"));
        let removed_after_first_conflict = host.removed_groups.load(Ordering::SeqCst);
        let replacements_after_first_conflict = host.replacements.load(Ordering::SeqCst);
        let starts_after_first_conflict = host.starts.load(Ordering::SeqCst);

        // Every later arrival of a disagreeing worker reconciles the group
        // again. None of them may reach the replacement path, which clones the
        // member cards and republishes the catalog and the LoRA projections
        // under the reservation lock.
        for id in 3..8 {
            controller.apply_added(instance(id, "second-spec"));
            controller.start_queued_builds();
        }
        assert!(controller.builds.is_empty());
        assert_eq!(
            host.removed_groups.load(Ordering::SeqCst),
            removed_after_first_conflict
        );
        assert_eq!(
            host.replacements.load(Ordering::SeqCst),
            replacements_after_first_conflict
        );
        assert_eq!(
            host.starts.load(Ordering::SeqCst),
            starts_after_first_conflict
        );
    }

    #[tokio::test]
    async fn final_removal_cancels_build_without_late_publication() {
        let (host, mut starts) = FakeHost::new();
        let mut controller = ModelDiscoveryController::new(host.clone());
        let only = instance(1, "spec");
        controller.apply_added(only.clone());
        controller.start_queued_builds();
        starts.recv().await.unwrap();

        controller.apply_removed(&only.key);
        finish_build(&mut controller).await;

        assert!(host.members(&group_key()).is_empty());
        assert_eq!(host.removed_groups.load(Ordering::SeqCst), 0);
    }

    #[tokio::test]
    async fn recreated_group_rejects_prepared_result_from_prior_lifetime() {
        let (host, mut starts) = FakeHost::new();
        let mut controller = ModelDiscoveryController::new(host.clone());
        let first = instance(1, "same");

        controller.apply_added(first.clone());
        controller.start_queued_builds();
        let first_spec = starts.recv().await.unwrap();
        host.release.add_permits(1);
        let stale = controller.builds.join_next().await.unwrap().unwrap();
        controller.active_builds -= 1;

        controller.apply_removed(&first.key);
        controller.apply_added(first.clone());
        controller.start_queued_builds();
        let replacement_spec = starts.recv().await.unwrap();
        assert_ne!(first_spec.generation, replacement_spec.generation);

        controller.apply_build_result(stale);
        assert!(host.members(&group_key()).is_empty());
        assert_eq!(host.discarded.load(Ordering::SeqCst), 1);

        host.release.add_permits(1);
        finish_build(&mut controller).await;
        assert_eq!(host.members(&group_key()), BTreeSet::from([first.key]));
    }

    #[tokio::test]
    async fn adapter_cards_neither_start_nor_keep_worker_sets_alive() {
        let (host, mut starts) = FakeHost::new();
        let mut controller = ModelDiscoveryController::new(host.clone());
        let mut adapter = instance(1, "adapter-spec");
        adapter.mcid.model_suffix = Some("adapter".to_string());
        adapter.key = adapter.mcid.to_path();

        controller.apply_added(adapter.clone());
        controller.start_queued_builds();
        assert!(starts.try_recv().is_err());

        let base = instance(1, "base-spec");
        controller.apply_added(base.clone());
        controller.start_queued_builds();
        starts.recv().await.unwrap();
        host.release.add_permits(1);
        finish_build(&mut controller).await;
        assert_eq!(
            host.members(&group_key()),
            BTreeSet::from([base.key.clone()])
        );
        assert_eq!(
            host.adapters(&group_key()),
            BTreeSet::from([adapter.key.clone()])
        );
        controller.apply_removed(&adapter.key);
        assert!(host.adapters(&group_key()).is_empty());
        controller.apply_added(adapter.clone());
        assert_eq!(
            host.adapters(&group_key()),
            BTreeSet::from([adapter.key.clone()])
        );
        assert_eq!(host.starts.load(Ordering::SeqCst), 1);

        let mut updated_adapter = adapter.clone();
        updated_adapter.projection_fingerprint = "updated-projection".to_string();
        assert!(controller.apply_added(updated_adapter));
        assert_eq!(
            host.adapter_projection(&group_key(), &adapter.key),
            Some("updated-projection".to_string())
        );
        assert_eq!(host.starts.load(Ordering::SeqCst), 1);

        controller.apply_removed(&base.key);
        assert!(host.members(&group_key()).is_empty());
        assert!(host.adapters(&group_key()).is_empty());
        assert!(controller.desired.contains_key(&adapter.key));
        assert_eq!(host.removed_groups.load(Ordering::SeqCst), 1);
    }

    #[tokio::test(start_paused = true)]
    async fn failed_adapter_replacement_retains_safe_commit_and_retries_projection() {
        let (host, mut starts) = FakeHost::new();
        let mut controller = ModelDiscoveryController::new(host.clone());
        let base = instance(1, "base-spec");
        let mut adapter = instance(1, "adapter-spec");
        adapter.mcid.model_suffix = Some("adapter".to_string());
        adapter.key = adapter.mcid.to_path();

        controller.apply_added(adapter.clone());
        controller.apply_added(base);
        controller.start_queued_builds();
        starts.recv().await.unwrap();
        host.release.add_permits(1);
        finish_build(&mut controller).await;
        assert_eq!(
            host.adapter_projection(&group_key(), &adapter.key),
            Some("adapter-spec".to_string())
        );

        host.replace_failures.store(1, Ordering::SeqCst);
        let mut updated = adapter.clone();
        updated.projection_fingerprint = "updated".to_string();
        controller.apply_added(updated);
        assert_eq!(
            host.adapter_projection(&group_key(), &adapter.key),
            Some("adapter-spec".to_string())
        );
        assert_eq!(host.members(&group_key()).len(), 1);

        tokio::time::advance(Duration::from_secs(1)).await;
        controller.release_due_retries();
        assert_eq!(
            host.adapter_projection(&group_key(), &adapter.key),
            Some("updated".to_string())
        );
        assert_eq!(host.starts.load(Ordering::SeqCst), 1);
    }

    #[tokio::test(start_paused = true)]
    async fn desired_change_retries_a_failed_build_immediately() {
        let (host, mut starts) = FakeHost::new();
        host.failures.store(1, Ordering::SeqCst);
        let mut controller = ModelDiscoveryController::new(host.clone());
        let desired = instance(1, "spec");
        controller.apply_added(desired.clone());
        controller.start_queued_builds();
        starts.recv().await.unwrap();
        host.release.add_permits(1);
        finish_build(&mut controller).await;
        assert!(host.members(&group_key()).is_empty());

        let joined = instance(2, "spec");
        controller.apply_added(joined.clone());
        controller.start_queued_builds();
        starts.recv().await.unwrap();
        host.release.add_permits(1);
        finish_build(&mut controller).await;

        assert_eq!(
            host.members(&group_key()),
            BTreeSet::from([desired.key, joined.key])
        );
        assert_eq!(host.starts.load(Ordering::SeqCst), 2);
    }

    #[tokio::test(start_paused = true)]
    async fn failed_build_remains_unpublished_until_its_retry_succeeds() {
        let (host, mut starts) = FakeHost::new();
        host.failures.store(1, Ordering::SeqCst);
        let mut controller = ModelDiscoveryController::new(host.clone());
        let desired = instance(1, "spec");
        controller.apply_added(desired.clone());
        controller.start_queued_builds();
        starts.recv().await.unwrap();
        host.release.add_permits(1);
        finish_build(&mut controller).await;
        assert!(host.members(&group_key()).is_empty());

        tokio::time::advance(Duration::from_millis(999)).await;
        controller.release_due_retries();
        controller.start_queued_builds();
        assert!(starts.try_recv().is_err());

        tokio::time::advance(Duration::from_millis(1)).await;
        controller.release_due_retries();
        controller.start_queued_builds();
        starts.recv().await.unwrap();
        host.release.add_permits(1);
        finish_build(&mut controller).await;
        assert_eq!(host.members(&group_key()), BTreeSet::from([desired.key]));
    }

    #[tokio::test(start_paused = true)]
    async fn blocked_group_retries_on_its_deadline_not_unrelated_churn() {
        let (host, mut starts) = FakeHost::new();
        host.commit_failures.store(1, Ordering::SeqCst);
        let mut controller = ModelDiscoveryController::new(host.clone());
        let desired = instance(1, "spec");
        controller.apply_added(desired.clone());
        controller.start_queued_builds();
        starts.recv().await.unwrap();
        host.release.add_permits(1);
        finish_build(&mut controller).await;

        let mut unrelated_adapter = instance(99, "spec");
        unrelated_adapter.mcid.model_suffix = Some("unrelated-adapter".to_string());
        unrelated_adapter.key = unrelated_adapter.mcid.to_path();
        controller.apply_added(unrelated_adapter);
        controller.start_queued_builds();
        assert!(starts.try_recv().is_err());

        tokio::time::advance(Duration::from_secs(1)).await;
        controller.release_due_retries();
        controller.start_queued_builds();
        starts.recv().await.unwrap();
        host.release.add_permits(1);
        finish_build(&mut controller).await;
        assert_eq!(host.members(&group_key()), BTreeSet::from([desired.key]));
    }

    #[tokio::test]
    async fn reconciliation_repairs_missed_state_without_undoing_newer_events() {
        let (host, _starts) = FakeHost::new();
        let mut controller = ModelDiscoveryController::new(host);
        let first = instance(1, "spec");
        let second = instance(2, "spec");

        controller.apply_added(first.clone());
        let snapshot_revision = controller.revision;
        controller.apply_removed(&first.key);
        controller.apply_added(second.clone());
        controller.apply_reconciliation(
            ReconciliationResult {
                revision: snapshot_revision,
                instances: Ok(vec![discovery_instance(&first)]),
            },
            &NamespaceFilter::Global,
        );

        assert!(!controller.desired.contains_key(&first.key));
        assert!(controller.desired.contains_key(&second.key));

        let repair_revision = controller.revision;
        controller.apply_reconciliation(
            ReconciliationResult {
                revision: repair_revision,
                instances: Ok(vec![discovery_instance(&first)]),
            },
            &NamespaceFilter::Global,
        );
        assert!(controller.desired.contains_key(&first.key));
        assert!(!controller.desired.contains_key(&second.key));
    }

    #[test]
    fn retry_delay_follows_the_capped_schedule() {
        let delays = (1..=7).map(retry_delay).collect::<Vec<_>>();
        assert_eq!(delays, [1, 2, 4, 8, 16, 30, 30].map(Duration::from_secs));
    }
}
