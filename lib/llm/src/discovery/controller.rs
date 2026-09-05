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
    /// The elected winner and the size of each refused cohort at the last
    /// logged election. A disagreement between workers persists until an
    /// operator resolves it, so the refusal is reported when the election
    /// changes rather than on every reconciliation pass. Sizes, not member
    /// lists, keep the retained state and the log line bounded while a
    /// conflicting cohort grows.
    reported_election: Option<(String, BTreeMap<String, usize>)>,
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
            reported_election: None,
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

        let fingerprint = elect_cohort(&group.cohorts).clone();
        let member_keys = group
            .cohorts
            .get(&fingerprint)
            .cloned()
            .expect("the elected cohort is one of this group's cohorts");
        report_refused_cohorts(key, &mut group, &fingerprint);
        let members = self.members(&member_keys);
        let admitted = admitted_ids(&members);

        // An election that changes cohort retires the outgoing cohort's
        // admission channel rather than reusing it. A client built for that
        // cohort can outlive `remove_group`, because a published catalog
        // snapshot holds an `Arc<WorkerSet>` past the removal; keeping the
        // sender would let it observe the successor's instance IDs and
        // dispatch a pipeline built for one deployment card at workers
        // advertising another. Dropping the sender instead closes the
        // receiver, and the client stops at its last admitted set.
        let retired_admissions = status_fingerprint(&old_status)
            .is_some_and(|previous| previous != fingerprint)
            .then(|| group.retire_admissions());

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
                    // Withdraw admissions on whichever channel the outgoing
                    // cohort's clients are watching, then publish the
                    // successor's instances on the new one.
                    retired_admissions
                        .as_ref()
                        .unwrap_or(&group.admission_tx)
                        .send_replace(Vec::new());
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
        ) && group.cohorts.contains_key(&result.spec.fingerprint);
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
    /// A released retry re-runs the election rather than requeueing the
    /// fingerprint it stored when it failed. Cohort membership can have moved
    /// under a group while it waited out its backoff, and the released attempt
    /// has to be the one the current cohort set elects.
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

/// Pick the one cohort of a group whose workers are allowed to serve the model.
///
/// A cohort holds the workers that agree about a model's deployment card, and
/// only one may serve: mixing cohorts could route a model's traffic to a
/// different materialization. The winner is a function of the observed cohort
/// set alone — largest cohort, ties broken by the lexicographically smallest
/// fingerprint — and nothing local to one frontend enters into it. Two
/// frontends that observe the same workers therefore admit the same cohort
/// however they each arrived at their current state, so a load balancer cannot
/// spread one logical model across two materializations.
///
/// The elected cohort can be one that fails to materialize. It stays elected
/// while it keeps failing, retrying under the controller's backoff, because
/// demoting it would make the winner depend on this frontend's own build
/// history. Restoring the model then needs the mismatch resolved at the
/// workers, which is what the refusal log asks for.
fn elect_cohort(cohorts: &HashMap<String, BTreeSet<String>>) -> &String {
    cohorts
        .iter()
        .max_by(|(left_fingerprint, left), (right_fingerprint, right)| {
            left.len()
                .cmp(&right.len())
                .then_with(|| right_fingerprint.cmp(left_fingerprint))
        })
        .map(|(fingerprint, _)| fingerprint)
        .expect("a non-empty group has at least one cohort")
}

/// How many refused cohorts one refusal log line names. The rest are covered by
/// the cohort and worker counts, which keeps the line's size independent of how
/// many workers are misconfigured.
const REFUSED_COHORT_SAMPLE: usize = 3;

fn report_refused_cohorts(key: &GroupKey, group: &mut DesiredGroup, elected: &str) {
    if group.cohorts.len() < 2 {
        group.reported_election = None;
        return;
    }
    let refused = group
        .cohorts
        .iter()
        .filter(|(fingerprint, _)| fingerprint.as_str() != elected)
        .map(|(fingerprint, members)| (fingerprint.clone(), members.len()))
        .collect::<BTreeMap<_, _>>();
    let election = (elected.to_string(), refused);
    if group.reported_election.as_ref() == Some(&election) {
        return;
    }
    let (_, refused) = &election;
    let refused_workers = refused.values().sum::<usize>();
    let refused_sample = refused
        .iter()
        .take(REFUSED_COHORT_SAMPLE)
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
        elected_checksum = %elected,
        refused_cohorts = refused.len(),
        refused_workers,
        refused_sample = %refused_sample,
        "Workers in this worker set published model deployment cards with different checksums. \
         Serving the elected cohort only; the refused workers receive no traffic. \
         Restate the configuration so every worker in this worker set advertises the same card, \
         then drain the refused workers of this worker set."
    );
    group.reported_election = Some(election);
}

/// The fingerprint this group currently has a commit on, if any.
///
/// A commit means the host is serving that cohort, so withdrawing it has to go
/// through `remove_group`. It carries no weight in the election: the winner is
/// derived from the observed cohort set alone.
fn committed_fingerprint(status: &GroupStatus) -> Option<&str> {
    match status {
        GroupStatus::Ready { fingerprint, .. } | GroupStatus::BlockedReady { fingerprint, .. } => {
            Some(fingerprint)
        }
        GroupStatus::Idle
        | GroupStatus::Queued { .. }
        | GroupStatus::Building { .. }
        | GroupStatus::Retrying { .. }
        | GroupStatus::Blocked { .. } => None,
    }
}

/// The fingerprint this group is working towards, committed or not.
///
/// A status carrying a different fingerprint than the current election is one
/// this reconciliation supersedes, which is what tells the caller to retire the
/// outgoing cohort's admission channel.
fn status_fingerprint(status: &GroupStatus) -> Option<&str> {
    match status {
        GroupStatus::Queued { fingerprint }
        | GroupStatus::Building { fingerprint, .. }
        | GroupStatus::Ready { fingerprint, .. }
        | GroupStatus::Retrying { fingerprint, .. }
        | GroupStatus::Blocked { fingerprint, .. }
        | GroupStatus::BlockedReady { fingerprint, .. } => Some(fingerprint),
        GroupStatus::Idle => None,
    }
}

fn cancel_build(status: &GroupStatus) {
    if let GroupStatus::Building { cancellation, .. } = status {
        cancellation.cancel();
    }
}

fn status_has_commit(status: &GroupStatus) -> bool {
    committed_fingerprint(status).is_some()
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

    /// Run the controller's build side to a standstill. An election change can
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
    async fn commit_does_not_pin_a_cohort_the_election_no_longer_favors() {
        let (host, mut starts) = FakeHost::new();
        let mut controller = ModelDiscoveryController::new(host.clone());
        // The committed cohort loses the election twice over: it is smaller
        // than the newcomer cohort and its checksum also loses the lexical
        // tie-break. A frontend that let its own commit win would keep serving
        // `z-spec` here, and a second frontend that restarted into the same
        // worker set would elect `a-spec` — one logical model, two
        // materializations.
        let committed = instance(1, "z-spec");
        controller.apply_added(committed.clone());
        controller.start_queued_builds();
        starts.recv().await.unwrap();
        host.release.add_permits(1);
        finish_build(&mut controller).await;
        assert_eq!(
            host.members(&group_key()),
            BTreeSet::from([committed.key.clone()])
        );

        let newcomer_first = instance(2, "a-spec");
        let newcomer_second = instance(3, "a-spec");
        controller.apply_added(newcomer_first.clone());
        controller.apply_added(newcomer_second.clone());
        controller.start_queued_builds();
        drain_builds(&host, &mut controller).await;

        // The larger cohort is promoted, and the withdrawn one is gone rather
        // than merged into it.
        assert_eq!(
            host.members(&group_key()),
            BTreeSet::from([newcomer_first.key.clone(), newcomer_second.key.clone()])
        );
        assert_eq!(
            admitted(&controller),
            vec![
                newcomer_first.mcid.instance_id,
                newcomer_second.mcid.instance_id
            ]
        );
        assert_eq!(host.removed_groups.load(Ordering::SeqCst), 1);
    }

    #[tokio::test]
    async fn an_election_that_switches_cohorts_retires_the_old_admission_channel() {
        let (host, mut starts) = FakeHost::new();
        let mut controller = ModelDiscoveryController::new(host.clone());
        let committed = instance(1, "z-spec");
        controller.apply_added(committed.clone());
        controller.start_queued_builds();
        starts.recv().await.unwrap();
        host.release.add_permits(1);
        finish_build(&mut controller).await;

        // A client built for the committed cohort keeps watching the channel it
        // was handed. A catalog snapshot can hold it alive past `remove_group`,
        // so it must never observe the successor cohort's instance IDs.
        let retired = controller
            .groups
            .get(&group_key())
            .expect("the group is still tracked")
            .admission_tx
            .subscribe();

        let newcomer_first = instance(2, "a-spec");
        let newcomer_second = instance(3, "a-spec");
        controller.apply_added(newcomer_first.clone());
        controller.apply_added(newcomer_second.clone());
        controller.start_queued_builds();
        drain_builds(&host, &mut controller).await;

        assert_eq!(
            admitted(&controller),
            vec![
                newcomer_first.mcid.instance_id,
                newcomer_second.mcid.instance_id
            ]
        );
        assert_eq!(*retired.borrow(), Vec::<u64>::new());
        // The updater loop exits on this error, which is how the retired client
        // stops instead of following the successor.
        assert!(retired.has_changed().is_err());
    }

    #[tokio::test]
    async fn conflict_during_build_commits_the_elected_cohort_alone() {
        let (host, mut starts) = FakeHost::new();
        let mut controller = ModelDiscoveryController::new(host.clone());
        // The two cohorts are the same size and `first-spec` wins the
        // tie-break, so the in-flight build survives the conflicting arrival.
        let elected = instance(1, "first-spec");
        let conflicting = instance(2, "second-spec");
        controller.apply_added(elected.clone());
        controller.start_queued_builds();
        starts.recv().await.unwrap();

        controller.apply_added(conflicting.clone());
        host.release.add_permits(1);
        finish_build(&mut controller).await;
        assert_eq!(
            host.members(&group_key()),
            BTreeSet::from([elected.key.clone()])
        );
        assert_eq!(admitted(&controller), vec![elected.mcid.instance_id]);
        assert_eq!(host.removed_groups.load(Ordering::SeqCst), 0);

        controller.apply_removed(&conflicting.key);
        assert_eq!(host.members(&group_key()), BTreeSet::from([elected.key]));
        assert_eq!(host.starts.load(Ordering::SeqCst), 1);
    }

    #[tokio::test]
    async fn simultaneous_mixed_checksums_elect_one_cohort_and_register_the_model() {
        let self_hosted = instance(1, "self-hosted-spec");
        let fallback = instance(2, "fallback-spec");
        let (host, mut controller, _starts) =
            register_in_arrival_order(&self_hosted, &fallback).await;

        // Neither cohort has committed anything, so the election is decided by the
        // cohort set alone: equal sizes, and `fallback-spec` sorts first.
        assert_eq!(
            host.members(&group_key()),
            BTreeSet::from([fallback.key.clone()])
        );
        assert_eq!(host.removed_groups.load(Ordering::SeqCst), 0);
        // Negative control: merging both cohorts would satisfy every assertion
        // above while routing one model's traffic across two materializations.
        assert!(!host.members(&group_key()).contains(&self_hosted.key));
        assert_eq!(admitted(&controller), vec![fallback.mcid.instance_id]);

        // The refused cohort is refused, not discarded: it is promoted and built
        // once the elected cohort leaves.
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
    async fn cohort_election_does_not_depend_on_worker_arrival_order() {
        // Two frontend replicas can observe the same workers in either order.
        // With no commit anywhere, both must elect the same cohort.
        let self_hosted = instance(1, "self-hosted-spec");
        let fallback = instance(2, "fallback-spec");

        let (self_hosted_first, _, _self_hosted_starts) =
            register_in_arrival_order(&self_hosted, &fallback).await;
        let (fallback_first, _, _fallback_starts) =
            register_in_arrival_order(&fallback, &self_hosted).await;

        assert_eq!(
            self_hosted_first.members(&group_key()),
            fallback_first.members(&group_key())
        );
        assert_eq!(
            self_hosted_first.members(&group_key()),
            BTreeSet::from([fallback.key])
        );
    }

    #[tokio::test]
    async fn a_failing_cohort_holds_the_election_until_a_larger_cohort_arrives() {
        let (host, mut starts) = FakeHost::new();
        let mut controller = ModelDiscoveryController::new(host.clone());
        // `failing` sorts first, so it wins every tie-break and can only lose the
        // election by losing to a larger cohort. Its first two builds fail.
        host.failures.store(2, Ordering::SeqCst);
        let failing = instance(1, "a-failing-spec");
        let healthy = instance(2, "b-healthy-spec");
        let healthy_peer = instance(3, "b-healthy-spec");

        controller.apply_added(failing.clone());
        controller.start_queued_builds();
        starts.recv().await.unwrap();
        host.release.add_permits(1);
        finish_build(&mut controller).await;

        // A second cohort of one loses the tie-break, so the failing cohort is
        // rebuilt — and fails again.
        controller.apply_added(healthy.clone());
        controller.start_queued_builds();
        starts.recv().await.unwrap();
        host.release.add_permits(1);
        finish_build(&mut controller).await;
        assert!(host.members(&group_key()).is_empty());

        // The healthy cohort now outnumbers the failing one and takes the model.
        controller.apply_added(healthy_peer.clone());
        controller.start_queued_builds();
        drain_builds(&host, &mut controller).await;
        assert_eq!(
            host.members(&group_key()),
            BTreeSet::from([healthy.key, healthy_peer.key])
        );
        assert!(!host.members(&group_key()).contains(&failing.key));
        assert_eq!(host.failures.load(Ordering::SeqCst), 0);
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
