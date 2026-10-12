// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::{
    collections::{BTreeSet, HashMap, HashSet, VecDeque},
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
    pub(crate) mdc_checksum: String,
    pub(crate) projection_fingerprint: String,
    /// Digest of this worker's Qwen video prompt-expansion contract.
    pub(crate) video_contract: Option<String>,
}

impl DesiredInstance {
    fn materializes_worker_set(&self) -> bool {
        self.mcid.model_suffix.is_none()
    }
}

/// Returns the cohort's shared contract, if every member publishes one.
///
/// A WorkerSet has one video-routing processor, so exact video routing is
/// enabled only when all members agree.
fn cohort_video_contract(members: &[DesiredInstance]) -> Option<String> {
    let mut members = members.iter();
    let agreed = members.next()?.video_contract.clone()?;
    members
        .all(|member| member.video_contract.as_deref() == Some(agreed.as_str()))
        .then_some(agreed)
}

/// Identifies a cohort's active video contract without changing its MDC checksum.
fn cohort_fingerprint(mdc_checksum: &str, video_contract: Option<&str>) -> String {
    match video_contract {
        Some(contract) => format!("{mdc_checksum}\0video_contract\0{contract}"),
        None => mdc_checksum.to_string(),
    }
}

/// Splits a fingerprint back into the MDC checksum and video contract it carries.
fn fingerprint_parts(fingerprint: &str) -> (&str, Option<&str>) {
    match fingerprint.split_once("\0video_contract\0") {
        Some((mdc_checksum, contract)) => (mdc_checksum, Some(contract)),
        None => (fingerprint, None),
    }
}

#[derive(Clone, Debug)]
pub(crate) struct GroupSpec {
    pub(crate) key: GroupKey,
    pub(crate) mdc_checksum: String,
    pub(crate) fingerprint: String,
    pub(crate) generation: u64,
    pub(crate) representative: DesiredInstance,
    /// Contract shared by the cohort, if any.
    pub(crate) video_contract: Option<String>,
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

    fn replace_prepared_group(
        &self,
        spec: &GroupSpec,
        prepared: Self::Prepared,
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
        mdc_checksum: String,
        committed_members: Option<BTreeSet<String>>,
    },
    Building {
        mdc_checksum: String,
        generation: u64,
        cancellation: CancellationToken,
        committed_members: Option<BTreeSet<String>>,
    },
    Ready {
        mdc_checksum: String,
        committed_members: BTreeSet<String>,
    },
    Retrying {
        mdc_checksum: String,
        deadline: Instant,
        committed_members: Option<BTreeSet<String>>,
    },
    Blocked {
        mdc_checksum: String,
        deadline: Instant,
        committed_members: Option<BTreeSet<String>>,
    },
    BlockedReady {
        mdc_checksum: String,
        committed_members: BTreeSet<String>,
        deadline: Instant,
    },
}

/// The committed pipeline that keeps serving while its replacement is prepared.
/// Retired once the replacement commits or the commit is withdrawn.
struct RetainedPipeline {
    admission_tx: watch::Sender<Vec<u64>>,
    /// Contract this pipeline expands video prompts with. Held here because the
    /// group's status carries the *replacement's* fingerprint while a rebuild is
    /// pending, so it is the only record of what the retained workers must publish.
    /// `None` means it was built without exact video routing.
    contract: Option<String>,
}

struct DesiredGroup {
    retry_attempt: u32,
    cohorts: HashMap<String, BTreeSet<String>>,
    cohort_order: VecDeque<String>,
    reported_rejections: HashSet<String>,
    admission_tx: watch::Sender<Vec<u64>>,
    retained: Option<RetainedPipeline>,
    status: GroupStatus,
}

impl DesiredGroup {
    fn new() -> Self {
        let (admission_tx, _) = watch::channel(Vec::new());
        Self {
            retry_attempt: 0,
            cohorts: HashMap::new(),
            cohort_order: VecDeque::new(),
            reported_rejections: HashSet::new(),
            admission_tx,
            retained: None,
            status: GroupStatus::Idle,
        }
    }

    fn insert(&mut self, instance: &DesiredInstance) {
        if !self.cohorts.contains_key(&instance.mdc_checksum) {
            self.cohort_order.push_back(instance.mdc_checksum.clone());
        }
        self.cohorts
            .entry(instance.mdc_checksum.clone())
            .or_default()
            .insert(instance.key.clone());
    }

    fn remove(&mut self, instance: &DesiredInstance) {
        let Some(cohort) = self.cohorts.get_mut(&instance.mdc_checksum) else {
            return;
        };
        cohort.remove(&instance.key);
        if cohort.is_empty() {
            self.cohorts.remove(&instance.mdc_checksum);
            self.cohort_order
                .retain(|checksum| checksum != &instance.mdc_checksum);
            self.reported_rejections.remove(&instance.mdc_checksum);
        }
    }

    fn selected_checksum(&self) -> Option<&str> {
        self.cohort_order.front().map(String::as_str)
    }

    fn report_rejections(&mut self, key: &GroupKey) {
        let Some(incumbent) = self.cohort_order.front() else {
            return;
        };
        for checksum in self.cohort_order.iter().skip(1) {
            if !self.reported_rejections.insert(checksum.clone()) {
                continue;
            }
            tracing::error!(
                model = %key.model_name,
                worker_set = %key.worker_set_key,
                incumbent_checksum = %incumbent,
                rejected_checksum = %checksum,
                instance = ?self.cohorts.get(checksum).and_then(|members| members.first()),
                "Rejected incompatible workers; the first accepted configuration retains the WorkerSet"
            );
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
    /// The revision at the last `DiscoveryEvent::Resync`. A periodic list that started before
    /// it can carry an instance the resync removed, so its result is discarded.
    resync_revision: u64,
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
            resync_revision: 0,
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
            DiscoveryEvent::Resync(instances) => {
                // A resync supersedes every list that started before it, including one that
                // started at the current revision: the events it missed never reached the
                // controller, so nothing advanced the revision on their behalf.
                self.revision = self.revision.wrapping_add(1);
                self.resync_revision = self.revision;
                self.apply_reconciliation(
                    ReconciliationResult {
                        revision: self.revision,
                        instances: Ok(instances),
                    },
                    namespace_filter,
                );
                false
            }
        };
    }

    fn apply_added(&mut self, instance: DesiredInstance) -> bool {
        if let Some(existing) = self.desired.get(&instance.key) {
            if existing.mdc_checksum == instance.mdc_checksum
                && existing.projection_fingerprint == instance.projection_fingerprint
                && existing.video_contract == instance.video_contract
            {
                // A repeated registration still confirms liveness after an older snapshot.
                self.record_mutation(instance.key.clone());
                return false;
            }
            if existing.materializes_worker_set()
                && (existing.group_key != instance.group_key
                    || existing.mdc_checksum != instance.mdc_checksum)
            {
                tracing::error!(
                    instance = instance.key,
                    existing_group = %existing.group_key.id(),
                    candidate_group = %instance.group_key.id(),
                    "Rejected an in-place materialization change; worker instance paths identify immutable incarnations"
                );
                self.record_mutation(instance.key.clone());
                return false;
            }
        }

        let group_key = instance.group_key.clone();
        let endpoint_id = instance.endpoint_id.clone();
        let instance_id = instance.mcid.instance_id;
        let instance_key = instance.key.clone();
        let materializes_worker_set = instance.materializes_worker_set();
        let mut affects_selected = false;
        if materializes_worker_set {
            let group = self
                .groups
                .entry(group_key.clone())
                .or_insert_with(DesiredGroup::new);
            group.insert(&instance);
            affects_selected = group.selected_checksum() == Some(instance.mdc_checksum.as_str());
            group.report_rejections(&group_key);
        }
        self.desired.insert(instance.key.clone(), instance);
        self.record_mutation(instance_key);

        if materializes_worker_set {
            if affects_selected {
                self.reconcile_group(&group_key, true);
            }
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
            self.groups
                .get(&instance.group_key)
                .filter(|group| group.selected_checksum() == Some(instance.mdc_checksum.as_str()))
                .map(|_| vec![instance.group_key.clone()])
                .unwrap_or_default()
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
            if let Some(retained) = group.retained.take() {
                retained.admission_tx.send_replace(Vec::new());
            }
            cancel_build(&old_status);
            if status_has_commit(&old_status) {
                self.host.remove_group(key);
            }
            return;
        }

        let mdc_checksum = group
            .selected_checksum()
            .expect("non-empty group has a cohort")
            .to_string();
        let member_keys = group.cohorts[&mdc_checksum].clone();
        let members = self.members(&member_keys);
        let fingerprint =
            cohort_fingerprint(&mdc_checksum, cohort_video_contract(&members).as_deref());
        let fingerprint_changed =
            status_checksum(&old_status).is_some_and(|previous| previous != fingerprint);
        let mut retained_commit = status_committed_members(&old_status).cloned();
        // The contract the currently committed pipeline serves with. While a rebuild
        // is pending, `old_status` names the replacement, so the retained pipeline is
        // the authority on what its own workers must publish.
        let committed_contract = match &group.retained {
            Some(retained) => retained.contract.clone(),
            None => fingerprint_parts(status_checksum(&old_status).unwrap_or_default())
                .1
                .map(str::to_string),
        };
        if fingerprint_changed {
            retained_commit = retained_commit.and_then(|committed_members| {
                let (previous_checksum, _) = fingerprint_parts(status_checksum(&old_status)?);
                if previous_checksum != mdc_checksum {
                    return None;
                }
                surviving_commit(&members, &committed_members, committed_contract.as_deref())
            });
            cancel_build(&old_status);
            if retained_commit.is_some() {
                // The retained pipeline keeps its own sender, or its client leaves the
                // update loop and freezes on the worker list it last saw. Retained
                // pipeline clients must never observe the successor's IDs, so the
                // successor builds against a fresh channel.
                let (successor_tx, _) = watch::channel(Vec::new());
                let previous_tx = std::mem::replace(&mut group.admission_tx, successor_tx);
                group.retained.get_or_insert_with(|| RetainedPipeline {
                    admission_tx: previous_tx,
                    contract: committed_contract,
                });
            } else {
                group.admission_tx.send_replace(Vec::new());
                if let Some(retained) = group.retained.take() {
                    retained.admission_tx.send_replace(Vec::new());
                }
                if status_has_commit(&old_status) {
                    self.host.remove_group(key);
                }
                // Retained pipeline clients must never observe the successor's IDs.
                let (admission_tx, _) = watch::channel(Vec::new());
                group.admission_tx = admission_tx;
            }
            group.reported_rejections.clear();
            group.retry_attempt = 0;
        } else if group.retained.is_some() {
            // A rebuild is already pending, so the fingerprint tracks the replacement
            // and can stay put while the retained workers change underneath it. Its
            // contract is checked again here, or a worker that republished a different
            // one keeps receiving requests expanded with the contract it dropped.
            let had_commit = retained_commit.is_some();
            retained_commit = retained_commit.and_then(|committed_members| {
                surviving_commit(&members, &committed_members, committed_contract.as_deref())
            });
            if had_commit && retained_commit.is_none() {
                self.host.remove_group(key);
            }
        }
        // Follow the retained pipeline's membership until it is retired, so a worker
        // that leaves during a slow or failing rebuild stops receiving requests.
        if let Some(retained_members) = &retained_commit {
            if let Some(retained) = &group.retained {
                let still_serving = members
                    .iter()
                    .filter(|member| retained_members.contains(&member.key))
                    .cloned()
                    .collect::<Vec<_>>();
                retained
                    .admission_tx
                    .send_replace(admitted_ids(&still_serving));
            }
        } else if let Some(retained) = group.retained.take() {
            retained.admission_tx.send_replace(Vec::new());
        }
        group.report_rejections(key);
        let admitted = admitted_ids(&members);
        if !matches!(
            &old_status,
            GroupStatus::Ready { .. } | GroupStatus::BlockedReady { .. }
        ) || (fingerprint_changed && retained_commit.is_some())
        {
            group.admission_tx.send_replace(admitted);
        }

        group.status = match old_status {
            GroupStatus::Ready {
                mdc_checksum: ready_checksum,
                committed_members,
            }
            | GroupStatus::BlockedReady {
                mdc_checksum: ready_checksum,
                committed_members,
                ..
            } if ready_checksum == fingerprint => {
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
                            mdc_checksum: fingerprint,
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
                            mdc_checksum: fingerprint,
                            committed_members,
                            deadline: Instant::now() + delay,
                        }
                    }
                    Err(error) => {
                        group.admission_tx.send_replace(Vec::new());
                        self.host.remove_group(key);
                        group.retry_attempt = group.retry_attempt.saturating_add(1);
                        tracing::warn!(
                            group = %key.id(),
                            error = format!("{error:#}"),
                            "Discovery-group membership replacement failed; withdrawing stale commit"
                        );
                        GroupStatus::Blocked {
                            mdc_checksum: fingerprint,
                            deadline: Instant::now() + retry_delay(group.retry_attempt),
                            committed_members: None,
                        }
                    }
                }
            }
            // These arms hold a replacement whose fingerprint has not moved. They carry
            // `retained_commit` rather than the commit the status arrived with, so a
            // commit withdrawn above is not claimed again by the pending rebuild.
            GroupStatus::Building {
                mdc_checksum: building_checksum,
                generation,
                cancellation,
                ..
            } if building_checksum == fingerprint => GroupStatus::Building {
                mdc_checksum: fingerprint,
                generation,
                cancellation,
                committed_members: retained_commit,
            },
            GroupStatus::Queued {
                mdc_checksum: queued_checksum,
                ..
            } if queued_checksum == fingerprint => GroupStatus::Queued {
                mdc_checksum: fingerprint,
                committed_members: retained_commit,
            },
            GroupStatus::Retrying {
                mdc_checksum: retry_checksum,
                deadline,
                ..
            } if retry_checksum == fingerprint && !desired_changed => GroupStatus::Retrying {
                mdc_checksum: fingerprint,
                deadline,
                committed_members: retained_commit,
            },
            GroupStatus::Blocked {
                mdc_checksum: blocked_checksum,
                deadline,
                ..
            } if blocked_checksum == fingerprint && !desired_changed => GroupStatus::Blocked {
                mdc_checksum: fingerprint,
                deadline,
                committed_members: retained_commit,
            },
            previous => {
                cancel_build(&previous);
                group.admission_tx.send_replace(admitted_ids(&members));
                group.retry_attempt = 0;
                GroupStatus::Queued {
                    mdc_checksum: fingerprint,
                    committed_members: retained_commit,
                }
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
                    && self.groups.get(&instance.group_key).is_some_and(|group| {
                        group.selected_checksum() == Some(instance.mdc_checksum.as_str())
                    })
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
            let GroupStatus::Queued { .. } = &group.status else {
                continue;
            };
            let committed_members = match &group.status {
                GroupStatus::Queued {
                    committed_members, ..
                } => committed_members.clone(),
                _ => unreachable!("queued status was checked above"),
            };
            let Some(mdc_checksum) = group.selected_checksum().map(str::to_string) else {
                continue;
            };
            let Some(member_keys) = group.cohorts.get(&mdc_checksum) else {
                continue;
            };
            let members = member_keys
                .iter()
                .filter_map(|member_key| self.desired.get(member_key).cloned())
                .collect::<Vec<_>>();
            let Some(representative) = members.first().cloned() else {
                continue;
            };
            let video_contract = cohort_video_contract(&members);
            let fingerprint = cohort_fingerprint(&mdc_checksum, video_contract.as_deref());

            let cancellation = CancellationToken::new();
            let generation = self.next_build_generation;
            self.next_build_generation = self.next_build_generation.wrapping_add(1).max(1);
            let spec = GroupSpec {
                key: key.clone(),
                mdc_checksum: mdc_checksum.clone(),
                fingerprint: fingerprint.clone(),
                generation,
                representative,
                video_contract,
            };
            group.status = GroupStatus::Building {
                mdc_checksum: fingerprint,
                generation,
                cancellation: cancellation.clone(),
                committed_members,
            };

            let host = self.host.clone();
            let admitted_ids = group.admission_tx.subscribe();
            let task_spec = spec.clone();
            self.builds.spawn(async move {
                let future = AssertUnwindSafe(async {
                    tokio::select! {
                        biased;
                        _ = cancellation.cancelled() => BuildOutcome::Cancelled,
                        result = host.prepare(
                            task_spec.clone(),
                            admitted_ids,
                            cancellation.clone(),
                        ) => {
                            match result {
                                Ok(prepared) => BuildOutcome::Prepared(prepared),
                                Err(error) => BuildOutcome::Failed(error),
                            }
                        }
                    }
                });
                let outcome = match future.catch_unwind().await {
                    Ok(outcome) => outcome,
                    Err(_) => {
                        BuildOutcome::Failed(anyhow::anyhow!("model materialization panicked"))
                    }
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
                mdc_checksum,
                generation,
                ..
            } if mdc_checksum == &result.spec.fingerprint && *generation == result.spec.generation
        ) && group.selected_checksum() == Some(result.spec.mdc_checksum.as_str())
            && group.cohorts.contains_key(&result.spec.mdc_checksum)
            && group
                .cohorts
                .get(&result.spec.mdc_checksum)
                .is_some_and(|member_keys| {
                    cohort_fingerprint(
                        &result.spec.mdc_checksum,
                        cohort_video_contract(&self.members(member_keys)).as_deref(),
                    ) == result.spec.fingerprint
                });
        let committed_members = match &group.status {
            GroupStatus::Building {
                committed_members, ..
            } => committed_members.clone(),
            _ => None,
        };
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
                    .get(&result.spec.mdc_checksum)
                    .cloned()
                    .unwrap_or_default();
                let members = self.members(&member_keys);
                let adapters = self.adapters_for_members(&member_keys);
                group.admission_tx.send_replace(admitted_ids(&members));
                let commit_result = if committed_members.is_some() {
                    self.host
                        .replace_prepared_group(&result.spec, prepared, &members, &adapters)
                } else {
                    self.host
                        .commit_group(&result.spec, prepared, &members, &adapters)
                };
                match commit_result {
                    Ok(()) => {
                        // The replacement has taken over, so retire the sender the
                        // superseded pipeline was watching.
                        group.retained = None;
                        group.retry_attempt = 0;
                        group.status = GroupStatus::Ready {
                            mdc_checksum: result.spec.fingerprint,
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
                            mdc_checksum: result.spec.fingerprint,
                            deadline: Instant::now() + retry_delay(group.retry_attempt),
                            committed_members,
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
                    mdc_checksum: result.spec.fingerprint,
                    deadline: Instant::now() + delay,
                    committed_members,
                };
            }
            BuildOutcome::Cancelled => {
                group.status = GroupStatus::Queued {
                    mdc_checksum: result.spec.fingerprint,
                    committed_members,
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

    fn release_due_retries(&mut self) {
        let now = Instant::now();
        let mut retained_retries = Vec::new();
        for (key, group) in &mut self.groups {
            let (mdc_checksum, deadline, committed_members) = match &group.status {
                GroupStatus::Retrying {
                    mdc_checksum,
                    deadline,
                    committed_members,
                }
                | GroupStatus::Blocked {
                    mdc_checksum,
                    deadline,
                    committed_members,
                } => (mdc_checksum, deadline, committed_members.clone()),
                GroupStatus::BlockedReady {
                    mdc_checksum,
                    committed_members,
                    deadline,
                } if *deadline <= now => {
                    group.status = GroupStatus::Ready {
                        mdc_checksum: mdc_checksum.clone(),
                        committed_members: committed_members.clone(),
                    };
                    retained_retries.push(key.clone());
                    continue;
                }
                _ => continue,
            };
            if *deadline <= now {
                group.status = GroupStatus::Queued {
                    mdc_checksum: mdc_checksum.clone(),
                    committed_members,
                };
            }
        }
        for key in retained_retries {
            self.reconcile_group(&key, false);
        }
    }

    async fn shutdown_builds(&mut self) {
        for group in self.groups.values_mut() {
            group.admission_tx.send_replace(Vec::new());
            if let Some(retained) = group.retained.take() {
                retained.admission_tx.send_replace(Vec::new());
            }
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
        if result.revision < self.resync_revision {
            tracing::debug!(
                "Discarding a model reconciliation snapshot that predates a discovery resync"
            );
            return;
        }
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

/// The committed workers that may keep serving: those still present, provided they
/// all still publish `contract`. `None` withdraws the commit.
fn surviving_commit(
    members: &[DesiredInstance],
    committed_members: &BTreeSet<String>,
    contract: Option<&str>,
) -> Option<BTreeSet<String>> {
    // Keep serving the committed workers that are still here rather than demanding
    // the whole membership, so dropping the last worker of a rolling upgrade does
    // not withdraw a healthy group.
    let surviving = members
        .iter()
        .filter(|member| committed_members.contains(&member.key))
        .collect::<Vec<_>>();
    if surviving.is_empty() {
        return None;
    }
    // The committed pipeline expands video prompts with the contract it was built
    // from, so every worker it still serves must publish that same contract; one
    // that republished a different contract under the same ID would otherwise keep
    // receiving requests against the old one.
    if let Some(contract) = contract
        && !surviving
            .iter()
            .all(|member| member.video_contract.as_deref() == Some(contract))
    {
        return None;
    }
    Some(
        surviving
            .into_iter()
            .map(|member| member.key.clone())
            .collect(),
    )
}

fn status_checksum(status: &GroupStatus) -> Option<&str> {
    match status {
        GroupStatus::Idle => None,
        GroupStatus::Queued { mdc_checksum, .. }
        | GroupStatus::Building { mdc_checksum, .. }
        | GroupStatus::Ready { mdc_checksum, .. }
        | GroupStatus::Retrying { mdc_checksum, .. }
        | GroupStatus::Blocked { mdc_checksum, .. }
        | GroupStatus::BlockedReady { mdc_checksum, .. } => Some(mdc_checksum),
    }
}

fn status_has_commit(status: &GroupStatus) -> bool {
    status_committed_members(status).is_some()
}

fn status_committed_members(status: &GroupStatus) -> Option<&BTreeSet<String>> {
    match status {
        GroupStatus::Ready {
            committed_members, ..
        }
        | GroupStatus::BlockedReady {
            committed_members, ..
        } => Some(committed_members),
        GroupStatus::Queued {
            committed_members, ..
        }
        | GroupStatus::Building {
            committed_members, ..
        }
        | GroupStatus::Retrying {
            committed_members, ..
        }
        | GroupStatus::Blocked {
            committed_members, ..
        } => committed_members.as_ref(),
        GroupStatus::Idle => None,
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

    use crate::local_model::runtime_config::VLLM_QWEN_VIDEO_PROCESSOR_CONTRACT_RUNTIME_KEY;

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
        admissions: Mutex<Vec<watch::Receiver<Vec<u64>>>>,
        removed_groups: AtomicUsize,
        discarded: AtomicUsize,
        prepared_replacements: AtomicUsize,
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
                    admissions: Mutex::new(Vec::new()),
                    removed_groups: AtomicUsize::new(0),
                    discarded: AtomicUsize::new(0),
                    prepared_replacements: AtomicUsize::new(0),
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
                mdc_checksum: card.mdcsum().to_string(),
                projection_fingerprint: card.source_path.clone().unwrap(),
                video_contract: crate::discovery::watcher::qwen_video_contract_digest(&card),
                card,
            }))
        }

        async fn prepare(
            &self,
            spec: GroupSpec,
            admitted_ids: watch::Receiver<Vec<u64>>,
            _cancellation: CancellationToken,
        ) -> anyhow::Result<Self::Prepared> {
            let build = self.starts.fetch_add(1, Ordering::SeqCst) as u64;
            self.admissions.lock().unwrap().push(admitted_ids);
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
            // `ModelManager::commit_discovery_group` refuses a group ID it already holds.
            anyhow::ensure!(
                !self.committed.lock().unwrap().contains_key(&spec.key.id()),
                "discovery group {:?} is already committed",
                spec.key.id()
            );
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
            // `ModelManager::replace_discovery_group` requires a committed group ID.
            anyhow::ensure!(
                self.committed.lock().unwrap().contains_key(&key.id()),
                "committed discovery group {:?} not found",
                key.id()
            );
            self.committed.lock().unwrap().insert(
                key.id(),
                members.iter().map(|member| member.key.clone()).collect(),
            );
            self.store_adapters(key, adapters);
            Ok(())
        }

        fn replace_prepared_group(
            &self,
            spec: &GroupSpec,
            prepared: Self::Prepared,
            members: &[DesiredInstance],
            adapters: &[DesiredInstance],
        ) -> anyhow::Result<()> {
            let Prepared(_build) = prepared;
            self.prepared_replacements.fetch_add(1, Ordering::SeqCst);
            self.replace_group(&spec.key, members, adapters)
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

    fn instance(id: u64, mdc_checksum: &str) -> DesiredInstance {
        build_instance(id, mdc_checksum, None)
    }

    fn build_instance(
        id: u64,
        mdc_checksum: &str,
        video_contract: Option<serde_json::Value>,
    ) -> DesiredInstance {
        let mut card = ModelDeploymentCard::with_name_only("model");
        card.source_path = Some(mdc_checksum.to_string());
        if let Some(contract) = video_contract {
            card.runtime_config.runtime_data.insert(
                VLLM_QWEN_VIDEO_PROCESSOR_CONTRACT_RUNTIME_KEY.to_string(),
                contract,
            );
        }
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
            group_key: group_key(),
            mdc_checksum: card.mdcsum().to_string(),
            projection_fingerprint: mdc_checksum.to_string(),
            video_contract: crate::discovery::watcher::qwen_video_contract_digest(&card),
            card,
        }
    }

    /// Builds a worker that publishes a Qwen video prompt-expansion contract.
    ///
    /// The contract stays out of `mdcsum()`, so workers that differ only in
    /// `resize_mode` share a cohort and disagree only on the contract.
    fn instance_with_contract(id: u64, mdc_checksum: &str, resize_mode: &str) -> DesiredInstance {
        let instance = build_instance(
            id,
            mdc_checksum,
            Some(serde_json::json!({
                "placeholder_target": "bare_video_token",
                "resize_mode": resize_mode,
            })),
        );
        assert!(instance.video_contract.is_some());
        assert_eq!(
            instance.mdc_checksum,
            self::instance(id, mdc_checksum).mdc_checksum
        );
        instance
    }

    fn contract_digest(resize_mode: &str) -> String {
        instance_with_contract(0, "spec", resize_mode)
            .video_contract
            .unwrap()
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
    async fn contract_agreement_drives_the_fingerprint_and_survives_a_rolling_upgrade() {
        let (host, mut starts) = FakeHost::new();
        let mut controller = ModelDiscoveryController::new(host.clone());
        // A legacy worker that predates the contract, alongside one that publishes it.
        let legacy = instance(1, "same");
        let upgraded = instance_with_contract(2, "same", "round_ties_even");
        assert_eq!(legacy.mdc_checksum, upgraded.mdc_checksum);

        controller.apply_added(legacy.clone());
        controller.apply_added(upgraded.clone());
        controller.start_queued_builds();
        let mixed_spec = starts.recv().await.unwrap();
        // The cohort disagrees, so exact video routing stays off.
        assert_eq!(mixed_spec.video_contract, None);
        assert_eq!(mixed_spec.fingerprint, mixed_spec.mdc_checksum);
        host.release.add_permits(1);
        finish_build(&mut controller).await;
        assert_eq!(
            host.members(&group_key()),
            BTreeSet::from([legacy.key.clone(), upgraded.key.clone()])
        );
        let committed_admissions = host.admissions.lock().unwrap()[0].clone();
        assert_eq!(*committed_admissions.borrow(), vec![1, 2]);

        // Removing the last legacy worker changes membership and contract agreement
        // at once. Text service must keep running while the replacement is prepared.
        controller.apply_removed(&legacy.key);
        assert_eq!(
            host.members(&group_key()),
            BTreeSet::from([legacy.key.clone(), upgraded.key.clone()]),
            "the committed group must survive the upgrade's last legacy worker leaving"
        );
        assert_eq!(host.removed_groups.load(Ordering::SeqCst), 0);
        // The retained pipeline keeps its channel and stops admitting the worker
        // that left, so it never routes to a worker that is gone.
        assert_eq!(*committed_admissions.borrow(), vec![2]);
        assert!(committed_admissions.has_changed().is_ok());

        controller.start_queued_builds();
        let upgraded_spec = starts.recv().await.unwrap();
        assert_eq!(
            upgraded_spec.video_contract,
            Some(contract_digest("round_ties_even"))
        );
        assert_eq!(
            upgraded_spec.fingerprint,
            cohort_fingerprint(
                &upgraded_spec.mdc_checksum,
                Some(&contract_digest("round_ties_even"))
            )
        );
        host.release.add_permits(1);
        finish_build(&mut controller).await;

        // A retained commit is replaced in place, never committed a second time.
        assert_eq!(host.prepared_replacements.load(Ordering::SeqCst), 1);
        assert_eq!(host.members(&group_key()), BTreeSet::from([upgraded.key]));
        assert_eq!(host.removed_groups.load(Ordering::SeqCst), 0);
    }

    #[tokio::test]
    async fn republished_contract_withdraws_the_commit_it_no_longer_matches() {
        let (host, mut starts) = FakeHost::new();
        let mut controller = ModelDiscoveryController::new(host.clone());
        let worker = instance_with_contract(1, "same", "round_ties_even");
        let peer = instance_with_contract(2, "same", "round_ties_even");

        controller.apply_added(worker.clone());
        controller.apply_added(peer.clone());
        controller.start_queued_builds();
        let spec = starts.recv().await.unwrap();
        assert_eq!(
            spec.video_contract,
            Some(contract_digest("round_ties_even"))
        );
        host.release.add_permits(1);
        finish_build(&mut controller).await;
        assert_eq!(host.members(&group_key()).len(), 2);

        // The same instance ID republishes a different contract. The committed
        // pipeline expands video prompts with the old one, so it cannot be kept.
        let republished = instance_with_contract(1, "same", "legacy_ceil");
        assert_eq!(republished.key, worker.key);
        assert_ne!(republished.video_contract, worker.video_contract);
        assert!(controller.apply_added(republished));

        assert!(
            host.members(&group_key()).is_empty(),
            "a worker serving a contract it no longer publishes must lose its commit"
        );
        assert_eq!(host.removed_groups.load(Ordering::SeqCst), 1);

        // The cohort now disagrees, so the rebuild turns exact video routing off.
        controller.start_queued_builds();
        let rebuilt = starts.recv().await.unwrap();
        assert_eq!(rebuilt.video_contract, None);
        host.release.add_permits(1);
        finish_build(&mut controller).await;
        assert_eq!(host.prepared_replacements.load(Ordering::SeqCst), 0);
        assert_eq!(host.members(&group_key()).len(), 2);
    }

    #[tokio::test]
    async fn a_republished_contract_during_a_pending_rebuild_withdraws_the_retained_commit() {
        let (host, mut starts) = FakeHost::new();
        let mut controller = ModelDiscoveryController::new(host.clone());
        let first = instance_with_contract(1, "same", "round_ties_even");
        let second = instance_with_contract(2, "same", "round_ties_even");

        controller.apply_added(first.clone());
        controller.apply_added(second.clone());
        controller.start_queued_builds();
        starts.recv().await.unwrap();
        host.release.add_permits(1);
        finish_build(&mut controller).await;
        let committed_admissions = host.admissions.lock().unwrap()[0].clone();
        assert_eq!(*committed_admissions.borrow(), vec![1, 2]);

        // A worker without a contract joins, so the cohort stops agreeing and a
        // replacement with exact video routing off is queued. The commit built from
        // `round_ties_even` keeps serving while that replacement is prepared.
        let legacy = instance(3, "same");
        controller.apply_added(legacy.clone());
        assert_eq!(host.members(&group_key()).len(), 2);
        assert_eq!(*committed_admissions.borrow(), vec![1, 2]);
        assert_eq!(host.removed_groups.load(Ordering::SeqCst), 0);

        // Before that rebuild completes, a retained worker republishes a different
        // contract. The cohort still has no common contract, so the replacement's
        // fingerprint does not move -- but the retained pipeline still expands video
        // prompts with `round_ties_even`, which worker 1 no longer publishes.
        let republished = instance_with_contract(1, "same", "legacy_ceil");
        assert_eq!(republished.key, first.key);
        assert!(controller.apply_added(republished));

        assert!(
            host.members(&group_key()).is_empty(),
            "a retained pipeline must not keep serving a worker that republished a different contract"
        );
        assert_eq!(host.removed_groups.load(Ordering::SeqCst), 1);
        assert!(committed_admissions.borrow().is_empty());
    }

    #[tokio::test(start_paused = true)]
    async fn a_republished_contract_during_a_failed_rebuild_withdraws_the_retained_commit() {
        let (host, mut starts) = FakeHost::new();
        let mut controller = ModelDiscoveryController::new(host.clone());
        let first = instance_with_contract(1, "same", "round_ties_even");
        let second = instance_with_contract(2, "same", "round_ties_even");

        controller.apply_added(first.clone());
        controller.apply_added(second.clone());
        controller.start_queued_builds();
        starts.recv().await.unwrap();
        host.release.add_permits(1);
        finish_build(&mut controller).await;
        let committed_admissions = host.admissions.lock().unwrap()[0].clone();
        assert_eq!(*committed_admissions.borrow(), vec![1, 2]);

        // A worker without a contract joins and the replacement build fails, so the
        // commit built from `round_ties_even` keeps serving across the retry.
        host.failures.store(1, Ordering::SeqCst);
        controller.apply_added(instance(3, "same"));
        controller.start_queued_builds();
        starts.recv().await.unwrap();
        host.release.add_permits(1);
        finish_build(&mut controller).await;
        assert_eq!(host.members(&group_key()).len(), 2);
        assert_eq!(host.removed_groups.load(Ordering::SeqCst), 0);
        assert_eq!(*committed_admissions.borrow(), vec![1, 2]);

        // A retained worker republishes a different contract while the failed rebuild
        // waits to retry. The retained pipeline still expands video prompts with
        // `round_ties_even`, so it cannot keep serving that worker.
        let republished = instance_with_contract(1, "same", "legacy_ceil");
        assert_eq!(republished.key, first.key);
        assert!(controller.apply_added(republished));

        assert!(
            host.members(&group_key()).is_empty(),
            "a failed rebuild must not keep a commit whose worker republished a different contract"
        );
        assert_eq!(host.removed_groups.load(Ordering::SeqCst), 1);
        assert!(committed_admissions.borrow().is_empty());
    }

    #[tokio::test(start_paused = true)]
    async fn a_retained_pipeline_keeps_receiving_worker_updates_while_rebuilding() {
        let (host, mut starts) = FakeHost::new();
        let mut controller = ModelDiscoveryController::new(host.clone());
        let first = instance_with_contract(1, "same", "round_ties_even");
        let second = instance_with_contract(2, "same", "round_ties_even");
        let legacy = instance(3, "same");

        controller.apply_added(first.clone());
        controller.apply_added(second.clone());
        controller.apply_added(legacy.clone());
        controller.start_queued_builds();
        starts.recv().await.unwrap();
        host.release.add_permits(1);
        finish_build(&mut controller).await;
        let committed_admissions = host.admissions.lock().unwrap()[0].clone();
        assert_eq!(*committed_admissions.borrow(), vec![1, 2, 3]);

        // Dropping the legacy worker retains the commit and starts a rebuild that fails.
        host.failures.store(1, Ordering::SeqCst);
        controller.apply_removed(&legacy.key);
        assert_eq!(*committed_admissions.borrow(), vec![1, 2]);
        controller.start_queued_builds();
        starts.recv().await.unwrap();
        host.release.add_permits(1);
        finish_build(&mut controller).await;
        assert_eq!(host.members(&group_key()).len(), 3);

        // A worker leaving during the failed rebuild must still reach the retained
        // pipeline, or it keeps routing to a worker that is gone.
        controller.apply_removed(&second.key);
        assert_eq!(*committed_admissions.borrow(), vec![1]);
        assert!(committed_admissions.has_changed().is_ok());

        controller.start_queued_builds();
        starts.recv().await.unwrap();
        host.release.add_permits(1);
        finish_build(&mut controller).await;
        assert_eq!(host.members(&group_key()), BTreeSet::from([first.key]));
        assert_eq!(host.prepared_replacements.load(Ordering::SeqCst), 1);
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
    async fn competing_cohorts_preserve_serving_until_complete_incumbent_drain() {
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
        controller.apply_added(instance(3, "second-spec"));
        controller.apply_added(instance(4, "third-spec"));
        let old_admissions = host.admissions.lock().unwrap()[0].clone();
        assert_eq!(
            host.members(&group_key()),
            BTreeSet::from([compatible.key.clone()])
        );
        assert_eq!(*old_admissions.borrow(), vec![1]);

        controller.apply_removed(&compatible.key);
        assert!(host.members(&group_key()).is_empty());
        assert!(old_admissions.borrow().is_empty());
        assert!(old_admissions.has_changed().is_err());
        controller.start_queued_builds();
        starts.recv().await.unwrap();
        host.release.add_permits(1);
        finish_build(&mut controller).await;
        assert_eq!(
            host.members(&group_key()),
            BTreeSet::from([conflicting.key, instance(3, "second-spec").key])
        );
        assert_eq!(*host.admissions.lock().unwrap()[1].borrow(), vec![2, 3]);
        assert!(old_admissions.borrow().is_empty());
    }

    #[tokio::test]
    async fn first_cohort_reserves_queued_and_active_construction() {
        let (host, mut starts) = FakeHost::new();
        let mut controller = ModelDiscoveryController::new(host.clone());
        let first = instance(1, "first-spec");
        let conflicting = instance(2, "second-spec");
        controller.apply_added(first.clone());
        controller.apply_added(conflicting.clone());
        controller.start_queued_builds();
        starts.recv().await.unwrap();

        controller.apply_added(instance(3, "second-spec"));
        controller.apply_added(instance(4, "third-spec"));
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

    #[tokio::test]
    async fn snapshots_preserve_priority_and_recreated_cohorts_join_the_end() {
        let (host, mut starts) = FakeHost::new();
        let mut controller = ModelDiscoveryController::new(host.clone());
        let first = instance(1, "first");
        let second = instance(2, "second");
        let third = instance(3, "third");
        for member in [&first, &second, &third] {
            controller.apply_added(member.clone());
        }
        controller.apply_removed(&second.key);
        controller.apply_added(second.clone());
        controller.apply_reconciliation(
            ReconciliationResult {
                revision: controller.revision,
                instances: Ok([&second, &third, &first].map(discovery_instance).to_vec()),
            },
            &NamespaceFilter::Global,
        );

        // Snapshot order cannot promote a newcomer; disappearance resets priority.
        for incumbent in [&first, &third, &second] {
            controller.start_queued_builds();
            starts.recv().await.unwrap();
            host.release.add_permits(1);
            finish_build(&mut controller).await;
            assert_eq!(
                host.members(&group_key()),
                BTreeSet::from([incumbent.key.clone()])
            );
            controller.apply_removed(&incumbent.key);
        }
        assert!(host.members(&group_key()).is_empty());
    }

    #[tokio::test(start_paused = true)]
    async fn rejected_workers_and_adapters_cannot_reset_retry_or_change_publication() {
        for fail_at_commit in [false, true] {
            let (host, mut starts) = FakeHost::new();
            if fail_at_commit {
                host.commit_failures.store(1, Ordering::SeqCst);
            } else {
                host.failures.store(1, Ordering::SeqCst);
            }
            let mut controller = ModelDiscoveryController::new(host.clone());
            let incumbent = instance(1, "incumbent");
            let rejected = instance(2, "rejected");
            let mut adapter = instance(2, "adapter");
            adapter.mcid.model_suffix = Some("adapter".to_string());
            adapter.key = adapter.mcid.to_path();
            controller.apply_added(incumbent.clone());
            controller.start_queued_builds();
            starts.recv().await.unwrap();
            host.release.add_permits(1);
            finish_build(&mut controller).await;
            assert!(host.members(&group_key()).is_empty());

            tokio::time::advance(Duration::from_millis(500)).await;
            controller.apply_added(rejected.clone());
            controller.apply_added(adapter.clone());
            controller.apply_removed(&rejected.key);
            controller.apply_added(rejected);
            adapter.projection_fingerprint = "updated".to_string();
            controller.apply_added(adapter.clone());
            let mut unrelated_adapter = instance(99, "adapter");
            unrelated_adapter.mcid.model_suffix = Some("unrelated-adapter".to_string());
            unrelated_adapter.key = unrelated_adapter.mcid.to_path();
            controller.apply_added(unrelated_adapter);
            controller.start_queued_builds();
            assert!(starts.try_recv().is_err());

            tokio::time::advance(Duration::from_millis(499)).await;
            controller.release_due_retries();
            controller.start_queued_builds();
            assert!(starts.try_recv().is_err());
            assert!(host.members(&group_key()).is_empty());

            tokio::time::advance(Duration::from_millis(1)).await;
            controller.release_due_retries();
            controller.start_queued_builds();
            starts.recv().await.unwrap();
            host.release.add_permits(1);
            finish_build(&mut controller).await;
            assert_eq!(
                host.members(&group_key()),
                BTreeSet::from([incumbent.key.clone()])
            );
            assert!(host.adapters(&group_key()).is_empty());

            let mut admissions = host.admissions.lock().unwrap().last().unwrap().clone();
            admissions.borrow_and_update();
            // A publication attempt here would fail and withdraw the safe membership.
            host.replace_failures.store(1, Ordering::SeqCst);
            controller.apply_removed(&adapter.key);
            controller.apply_added(adapter);
            controller.apply_added(instance(3, "another-rejected"));
            controller.apply_removed(&instance(3, "another-rejected").key);
            assert_eq!(host.members(&group_key()), BTreeSet::from([incumbent.key]));
            assert!(!admissions.has_changed().unwrap());
            assert!(host.adapters(&group_key()).is_empty());
        }
    }

    #[tokio::test]
    async fn a_resync_replaces_the_desired_instances() {
        let (host, mut starts) = FakeHost::new();
        let mut controller = ModelDiscoveryController::new(host.clone());
        let first = instance(1, "spec");
        let second = instance(2, "spec");

        controller.apply_added(first.clone());
        controller.start_queued_builds();
        starts.recv().await.unwrap();
        host.release.add_permits(1);
        finish_build(&mut controller).await;
        assert_eq!(
            host.members(&group_key()),
            BTreeSet::from([first.key.clone()])
        );

        controller.apply_event(
            DiscoveryEvent::Resync(vec![discovery_instance(&second)]),
            &NamespaceFilter::Global,
        );
        assert_eq!(host.members(&group_key()), BTreeSet::from([second.key]));
    }

    #[tokio::test]
    async fn a_list_that_predates_a_resync_cannot_revive_a_removed_instance() {
        let (host, _starts) = FakeHost::new();
        let mut controller = ModelDiscoveryController::new(host);
        let first = instance(1, "spec");

        controller.apply_added(first.clone());
        // The periodic list starts here and captures `first` before its removal.
        let list_revision = controller.revision;
        controller.apply_removed(&first.key);
        controller.apply_event(DiscoveryEvent::Resync(Vec::new()), &NamespaceFilter::Global);
        controller.apply_reconciliation(
            ReconciliationResult {
                revision: list_revision,
                instances: Ok(vec![discovery_instance(&first)]),
            },
            &NamespaceFilter::Global,
        );
        assert!(controller.desired.is_empty());
    }

    #[tokio::test]
    async fn a_list_that_predates_a_resync_is_discarded_when_no_event_advanced_the_revision() {
        let (host, _starts) = FakeHost::new();
        let mut controller = ModelDiscoveryController::new(host);
        let first = instance(1, "spec");

        // The periodic list starts here and captures `first`. Its registration and its
        // removal are then both lost to a backend buffer overflow, so no event reaches
        // the controller to advance the revision before the resync that follows.
        let list_revision = controller.revision;
        controller.apply_event(DiscoveryEvent::Resync(Vec::new()), &NamespaceFilter::Global);
        controller.apply_reconciliation(
            ReconciliationResult {
                revision: list_revision,
                instances: Ok(vec![discovery_instance(&first)]),
            },
            &NamespaceFilter::Global,
        );
        assert!(controller.desired.is_empty());
    }

    #[tokio::test]
    async fn reconciliation_repairs_missed_state_without_undoing_newer_events() {
        let (host, mut starts) = FakeHost::new();
        let mut controller = ModelDiscoveryController::new(host.clone());
        let first = instance(1, "spec");
        let second = instance(2, "spec");

        controller.apply_added(first.clone());
        let duplicate_revision = controller.revision;
        controller.apply_added(first.clone());
        controller.apply_reconciliation(
            ReconciliationResult {
                revision: duplicate_revision,
                instances: Ok(Vec::new()),
            },
            &NamespaceFilter::Global,
        );
        controller.start_queued_builds();
        starts.recv().await.unwrap();
        host.release.add_permits(1);
        finish_build(&mut controller).await;
        assert_eq!(
            host.members(&group_key()),
            BTreeSet::from([first.key.clone()])
        );

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

        controller.start_queued_builds();
        starts.recv().await.unwrap();
        host.release.add_permits(1);
        finish_build(&mut controller).await;
        assert_eq!(
            host.members(&group_key()),
            BTreeSet::from([second.key.clone()])
        );

        let repair_revision = controller.revision;
        controller.apply_reconciliation(
            ReconciliationResult {
                revision: repair_revision,
                instances: Ok(vec![discovery_instance(&first)]),
            },
            &NamespaceFilter::Global,
        );
        assert_eq!(host.members(&group_key()), BTreeSet::from([first.key]));
    }
}
