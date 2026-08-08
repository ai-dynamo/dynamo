// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::{
    collections::{BTreeSet, HashMap, HashSet},
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

    fn add_group_member(&self, key: &GroupKey, member: &DesiredInstance) -> anyhow::Result<()>;

    fn remove_group_member(&self, key: &GroupKey, instance_key: &str) -> anyhow::Result<()>;

    fn sync_group_adapters(
        &self,
        key: &GroupKey,
        adapters: &[DesiredInstance],
    ) -> anyhow::Result<()>;

    fn remove_group(&self, key: &GroupKey);

    fn discard_prepared(&self, prepared: Self::Prepared);

    async fn list_instances(&self) -> anyhow::Result<Vec<DiscoveryInstance>>;

    fn project_lora(
        &self,
        endpoint_id: &EndpointId,
        reset_worker_ids: &HashSet<u64>,
        desired: &[(ModelCardInstanceId, ModelDeploymentCard)],
    );
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
}

struct DesiredGroup {
    generation: u64,
    retry_attempt: u32,
    cohorts: HashMap<String, BTreeSet<String>>,
    admission_tx: watch::Sender<Vec<u64>>,
    status: GroupStatus,
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
        }
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
    endpoint_workers: HashMap<EndpointId, HashSet<u64>>,
    revision: u64,
    instance_revisions: HashMap<String, u64>,
    builds: JoinSet<BuildResult<H::Prepared>>,
    reconciliations: JoinSet<ReconciliationResult>,
    active_builds: usize,
    max_concurrent_builds: usize,
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
            endpoint_workers: HashMap::new(),
            revision: 0,
            instance_revisions: HashMap::new(),
            builds: JoinSet::new(),
            reconciliations: JoinSet::new(),
            active_builds: 0,
            max_concurrent_builds: max_concurrent_builds.max(1),
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
            if existing.group_key == instance.group_key
                && existing.fingerprint == instance.fingerprint
            {
                return false;
            }
            tracing::error!(
                instance = instance.key,
                existing_group = %existing.group_key.id(),
                candidate_group = %instance.group_key.id(),
                "Rejected an in-place model-card mutation; worker instance paths identify immutable incarnations"
            );
            return false;
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
        self.project_lora_for(HashSet::from([endpoint_id]));
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
        self.project_lora_for(HashSet::from([instance.endpoint_id]));
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
            if matches!(old_status, GroupStatus::Ready { .. }) {
                self.host.remove_group(key);
            }
            return;
        }

        if group.cohorts.len() > 1 {
            group.admission_tx.send_replace(Vec::new());
            cancel_build(&old_status);
            if matches!(old_status, GroupStatus::Ready { .. }) {
                self.host.remove_group(key);
            }
            if !matches!(old_status, GroupStatus::Conflict) {
                group.generation = group.generation.wrapping_add(1);
                group.retry_attempt = 0;
            }
            group.status = GroupStatus::Conflict;
            self.groups.insert(key.clone(), group);
            return;
        }

        let (fingerprint, member_keys) = group
            .cohorts
            .iter()
            .next()
            .map(|(fingerprint, members)| (fingerprint.clone(), members.clone()))
            .expect("non-empty group has one cohort");
        let members = self.members(&member_keys);
        let admitted = admitted_ids(&members);
        group.admission_tx.send_replace(admitted);

        group.status = match old_status {
            GroupStatus::Ready {
                fingerprint: ready_fingerprint,
                committed_members,
            } if ready_fingerprint == fingerprint => {
                let current_members = member_keys;
                let mut sync_failed = false;
                for removed in committed_members.difference(&current_members) {
                    if let Err(error) = self.host.remove_group_member(key, removed) {
                        tracing::error!(
                            group = %key.id(),
                            instance = removed,
                            error = format!("{error:#}"),
                            "Failed to remove a committed discovery-group member"
                        );
                        sync_failed = true;
                    }
                }
                for added in current_members.difference(&committed_members) {
                    let Some(member) = self.desired.get(added) else {
                        continue;
                    };
                    if let Err(error) = self.host.add_group_member(key, member) {
                        tracing::error!(
                            group = %key.id(),
                            instance = added,
                            error = format!("{error:#}"),
                            "Failed to add a committed discovery-group member"
                        );
                        sync_failed = true;
                    }
                }
                if !sync_failed {
                    let adapters = self.adapters_for_members(&current_members);
                    if let Err(error) = self.host.sync_group_adapters(key, &adapters) {
                        tracing::warn!(
                            group = %key.id(),
                            error = format!("{error:#}"),
                            "Failed to synchronize committed LoRA adapter surfaces"
                        );
                        sync_failed = true;
                    }
                }
                if sync_failed {
                    group.admission_tx.send_replace(Vec::new());
                    self.host.remove_group(key);
                    group.generation = group.generation.wrapping_add(1);
                    group.retry_attempt = group.retry_attempt.saturating_add(1);
                    GroupStatus::Blocked {
                        fingerprint,
                        deadline: Instant::now() + retry_delay(group.retry_attempt),
                    }
                } else {
                    GroupStatus::Ready {
                        fingerprint,
                        committed_members: current_members,
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
                if matches!(previous, GroupStatus::Ready { .. }) {
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

    fn project_lora_for(&mut self, endpoint_ids: HashSet<EndpointId>) {
        for endpoint_id in endpoint_ids {
            let committed_bases = self
                .desired
                .values()
                .filter(|instance| {
                    instance.materializes_worker_set()
                        && instance.endpoint_id == endpoint_id
                        && self.groups.get(&instance.group_key).is_some_and(|group| {
                            matches!(
                                &group.status,
                                GroupStatus::Ready { committed_members, .. }
                                    if committed_members.contains(&instance.key)
                            )
                        })
                })
                .cloned()
                .collect::<Vec<_>>();
            let committed_workers = committed_bases
                .iter()
                .map(|instance| instance.mcid.instance_id)
                .collect::<HashSet<_>>();
            let desired = self
                .desired
                .values()
                .filter(|instance| {
                    instance.endpoint_id == endpoint_id
                        && committed_workers.contains(&instance.mcid.instance_id)
                })
                .map(|instance| (instance.mcid.clone(), instance.card.clone()))
                .collect::<Vec<_>>();
            let current_workers = desired
                .iter()
                .map(|(mcid, _)| mcid.instance_id)
                .collect::<HashSet<_>>();
            let mut reset_workers = self
                .endpoint_workers
                .remove(&endpoint_id)
                .unwrap_or_default();
            reset_workers.extend(current_workers.iter().copied());
            self.host
                .project_lora(&endpoint_id, &reset_workers, &desired);
            if !current_workers.is_empty() {
                self.endpoint_workers.insert(endpoint_id, current_workers);
            }
        }
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
            let spec = GroupSpec {
                key: key.clone(),
                fingerprint: fingerprint.clone(),
                generation: group.generation,
                representative,
            };
            group.status = GroupStatus::Building {
                fingerprint,
                generation: group.generation,
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
        let affected_endpoints = group
            .cohorts
            .values()
            .flatten()
            .filter_map(|key| self.desired.get(key))
            .map(|instance| instance.endpoint_id.clone())
            .collect::<HashSet<_>>();
        self.groups.insert(result.spec.key, group);
        self.project_lora_for(affected_endpoints);
    }

    fn next_retry_deadline(&self) -> Option<Instant> {
        self.groups
            .values()
            .filter_map(|group| match group.status {
                GroupStatus::Retrying { deadline, .. } | GroupStatus::Blocked { deadline, .. } => {
                    Some(deadline)
                }
                _ => None,
            })
            .min()
    }

    fn release_due_retries(&mut self) {
        let now = Instant::now();
        for group in self.groups.values_mut() {
            let (fingerprint, deadline) = match &group.status {
                GroupStatus::Retrying {
                    fingerprint,
                    deadline,
                }
                | GroupStatus::Blocked {
                    fingerprint,
                    deadline,
                } => (fingerprint, deadline),
                _ => continue,
            };
            if *deadline <= now {
                group.status = GroupStatus::Queued {
                    fingerprint: fingerprint.clone(),
                };
            }
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

fn cancel_build(status: &GroupStatus) {
    if let GroupStatus::Building { cancellation, .. } = status {
        cancellation.cancel();
    }
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
        start_tx: mpsc::UnboundedSender<GroupSpec>,
        release: Semaphore,
        committed: Mutex<HashMap<String, BTreeSet<String>>>,
        adapters: Mutex<HashMap<String, BTreeSet<String>>>,
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
                    start_tx,
                    release: Semaphore::new(0),
                    committed: Mutex::new(HashMap::new()),
                    adapters: Mutex::new(HashMap::new()),
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
            self.adapters.lock().unwrap().insert(
                spec.key.id(),
                adapters.iter().map(|adapter| adapter.key.clone()).collect(),
            );
            Ok(())
        }

        fn add_group_member(&self, key: &GroupKey, member: &DesiredInstance) -> anyhow::Result<()> {
            self.committed
                .lock()
                .unwrap()
                .get_mut(&key.id())
                .unwrap()
                .insert(member.key.clone());
            Ok(())
        }

        fn remove_group_member(&self, key: &GroupKey, instance_key: &str) -> anyhow::Result<()> {
            self.committed
                .lock()
                .unwrap()
                .get_mut(&key.id())
                .unwrap()
                .remove(instance_key);
            Ok(())
        }

        fn sync_group_adapters(
            &self,
            key: &GroupKey,
            adapters: &[DesiredInstance],
        ) -> anyhow::Result<()> {
            self.adapters.lock().unwrap().insert(
                key.id(),
                adapters.iter().map(|adapter| adapter.key.clone()).collect(),
            );
            Ok(())
        }

        fn remove_group(&self, key: &GroupKey) {
            self.committed.lock().unwrap().remove(&key.id());
            self.adapters.lock().unwrap().remove(&key.id());
            self.removed_groups.fetch_add(1, Ordering::SeqCst);
        }

        fn discard_prepared(&self, prepared: Self::Prepared) {
            let Prepared(_build) = prepared;
            self.discarded.fetch_add(1, Ordering::SeqCst);
        }

        async fn list_instances(&self) -> anyhow::Result<Vec<DiscoveryInstance>> {
            Ok(Vec::new())
        }

        fn project_lora(
            &self,
            _endpoint_id: &EndpointId,
            _reset_worker_ids: &HashSet<u64>,
            _desired: &[(ModelCardInstanceId, ModelDeploymentCard)],
        ) {
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

        controller.apply_removed(&conflicting.key);
        controller.start_queued_builds();
        starts.recv().await.unwrap();
        host.release.add_permits(1);
        finish_build(&mut controller).await;
        assert_eq!(host.members(&group_key()), BTreeSet::from([compatible.key]));
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

        controller.apply_removed(&base.key);
        assert!(host.members(&group_key()).is_empty());
        assert!(host.adapters(&group_key()).is_empty());
        assert!(controller.desired.contains_key(&adapter.key));
        assert_eq!(host.removed_groups.load(Ordering::SeqCst), 1);
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
