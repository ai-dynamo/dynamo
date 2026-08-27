// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

use dashmap::DashMap;
use dynamo_kv_router::protocols::WorkerWithDpRank;
use dynamo_kv_router::sequences::SchedulerLoadSnapshot;
use tokio::sync::{Notify, mpsc};
use tokio_util::sync::CancellationToken;

use dynamo_runtime::component::Client;
use dynamo_runtime::engine::EngineContextGuard;
use dynamo_runtime::pipeline::WorkerLoadMonitor;

use crate::discovery::{KvWorkerMonitor, LoadThresholdHandle};
use crate::kv_router::KvRouter;
use crate::local_model::runtime_config::ModelRuntimeConfig;
use crate::protocols::common::timing::{WORKER_TYPE_DECODE, WORKER_TYPE_PREFILL};
use crate::worker_type::WorkerType;

/// Endpoint role whose scheduler and remote metrics feed one load context.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RouterLoadSource {
    Decode,
    Aggregated,
    Prefill,
    Encode,
}

impl RouterLoadSource {
    pub(crate) const fn metric_label(self) -> &'static str {
        match self {
            Self::Decode | Self::Aggregated => WORKER_TYPE_DECODE,
            Self::Prefill => WORKER_TYPE_PREFILL,
            Self::Encode => WorkerType::Encode.as_str(),
        }
    }

    pub const fn from_worker_type(worker_type: WorkerType) -> Self {
        match worker_type {
            WorkerType::Decode => Self::Decode,
            WorkerType::Aggregated => Self::Aggregated,
            WorkerType::Prefill => Self::Prefill,
            WorkerType::Encode => Self::Encode,
        }
    }

    pub(crate) fn from_worker_role_or_metric(
        worker_role: Option<WorkerType>,
        metric_worker_type: &'static str,
    ) -> Self {
        worker_role.map(Self::from_worker_type).unwrap_or_else(|| {
            if metric_worker_type == WORKER_TYPE_PREFILL {
                Self::Prefill
            } else {
                Self::Aggregated
            }
        })
    }

    pub(crate) const fn monitors_sequence_load(self) -> bool {
        !matches!(self, Self::Encode)
    }
}

const SCHEDULER_LOAD_CHANNEL_CAPACITY: usize = 256;

#[derive(Debug)]
enum SchedulerLoadCommand {
    Single(SchedulerLoadSnapshot),
    Batch(Vec<SchedulerLoadSnapshot>),
}

impl SchedulerLoadCommand {
    fn into_snapshots(self) -> Vec<SchedulerLoadSnapshot> {
        match self {
            Self::Single(snapshot) => vec![snapshot],
            Self::Batch(snapshots) => snapshots,
        }
    }
}

struct SchedulerLoadShared {
    overflow: DashMap<WorkerWithDpRank, SchedulerLoadSnapshot>,
    overflow_wake: Notify,
    coalesced_commands: AtomicU64,
    unexpected_closed: AtomicU64,
}

impl SchedulerLoadShared {
    fn new() -> Self {
        Self {
            overflow: DashMap::new(),
            overflow_wake: Notify::new(),
            coalesced_commands: AtomicU64::new(0),
            unexpected_closed: AtomicU64::new(0),
        }
    }

    fn coalesce(&self, command: SchedulerLoadCommand) {
        for snapshot in command.into_snapshots() {
            self.overflow.insert(snapshot.worker, snapshot);
        }
        self.overflow_wake.notify_one();

        let count = self.coalesced_commands.fetch_add(1, Ordering::Relaxed) + 1;
        if count.is_power_of_two() {
            tracing::warn!(
                coalesced_commands = count,
                "scheduler-load channel saturated; coalescing latest worker snapshots"
            );
        }
    }

    fn record_unexpected_closed(&self) {
        let count = self.unexpected_closed.fetch_add(1, Ordering::Relaxed) + 1;
        if count.is_power_of_two() {
            tracing::error!(
                closed_publications = count,
                "scheduler-load channel closed before load-context cancellation"
            );
        }
    }

    fn drain_overflow(&self, snapshots: &mut Vec<SchedulerLoadSnapshot>) {
        let workers = self
            .overflow
            .iter()
            .map(|entry| *entry.key())
            .collect::<Vec<_>>();
        for worker in workers {
            if let Some((_, snapshot)) = self.overflow.remove(&worker) {
                snapshots.push(snapshot);
            }
        }
    }
}

/// Nonblocking scheduler-load publication handle owned by one routing load context.
#[derive(Clone)]
pub struct SchedulerLoadSender {
    tx: Option<mpsc::Sender<SchedulerLoadCommand>>,
    shared: Arc<SchedulerLoadShared>,
    source: RouterLoadSource,
    cancellation_token: CancellationToken,
}

impl SchedulerLoadSender {
    pub(crate) fn disabled(
        source: RouterLoadSource,
        cancellation_token: CancellationToken,
    ) -> Self {
        Self {
            tx: None,
            shared: Arc::new(SchedulerLoadShared::new()),
            source,
            cancellation_token,
        }
    }

    pub(crate) const fn metric_label(&self) -> &'static str {
        self.source.metric_label()
    }

    pub fn publish(&self, snapshot: SchedulerLoadSnapshot) {
        self.try_publish(SchedulerLoadCommand::Single(snapshot));
    }

    pub fn publish_batch(&self, snapshots: Vec<SchedulerLoadSnapshot>) {
        if snapshots.is_empty() {
            return;
        }
        self.try_publish(SchedulerLoadCommand::Batch(snapshots));
    }

    pub fn is_enabled(&self) -> bool {
        self.tx.is_some()
    }

    fn try_publish(&self, command: SchedulerLoadCommand) {
        let Some(tx) = &self.tx else {
            return;
        };
        match tx.try_send(command) {
            Ok(()) => {}
            Err(mpsc::error::TrySendError::Full(command)) => self.shared.coalesce(command),
            Err(mpsc::error::TrySendError::Closed(_)) => {
                if !self.cancellation_token.is_cancelled() {
                    self.shared.record_unexpected_closed();
                }
            }
        }
    }
}

pub(crate) struct SchedulerLoadReceiver {
    rx: mpsc::Receiver<SchedulerLoadCommand>,
    shared: Arc<SchedulerLoadShared>,
}

impl SchedulerLoadReceiver {
    pub(crate) async fn recv(&mut self) -> Option<Vec<SchedulerLoadSnapshot>> {
        loop {
            tokio::select! {
                command = self.rx.recv() => {
                    let mut snapshots = command?.into_snapshots();
                    self.shared.drain_overflow(&mut snapshots);
                    return Some(snapshots);
                }
                _ = self.shared.overflow_wake.notified() => {
                    let mut snapshots = Vec::new();
                    self.shared.drain_overflow(&mut snapshots);
                    if !snapshots.is_empty() {
                        return Some(snapshots);
                    }
                }
            }
        }
    }
}

pub(crate) fn scheduler_load_channel(
    source: RouterLoadSource,
    cancellation_token: CancellationToken,
) -> (SchedulerLoadSender, SchedulerLoadReceiver) {
    scheduler_load_channel_with_capacity(
        source,
        cancellation_token,
        SCHEDULER_LOAD_CHANNEL_CAPACITY,
    )
}

fn scheduler_load_channel_with_capacity(
    source: RouterLoadSource,
    cancellation_token: CancellationToken,
    capacity: usize,
) -> (SchedulerLoadSender, SchedulerLoadReceiver) {
    let (tx, rx) = mpsc::channel(capacity);
    let shared = Arc::new(SchedulerLoadShared::new());
    (
        SchedulerLoadSender {
            tx: Some(tx),
            shared: shared.clone(),
            source,
            cancellation_token,
        },
        SchedulerLoadReceiver { rx, shared },
    )
}

/// Owns the load lifecycle for one typed routing endpoint.
///
/// Every selection and dispatch plane receives a clone of this context's
/// single endpoint [`Client`]. Decode, aggregated, and prefill contexts
/// are intentionally independent.
pub struct RoutingLoadContext {
    client: Client,
    source: RouterLoadSource,
    scheduler_load: SchedulerLoadSender,
    thresholds: LoadThresholdHandle,
    cancellation_token: CancellationToken,
    monitor: Option<KvWorkerMonitor>,
    _task_guard: Option<EngineContextGuard>,
}

impl RoutingLoadContext {
    pub async fn start(
        client: Client,
        source: RouterLoadSource,
        thresholds: LoadThresholdHandle,
        parent_token: &CancellationToken,
        task_guard: Option<EngineContextGuard>,
    ) -> anyhow::Result<Arc<Self>> {
        let cancellation_token = parent_token.child_token();
        let (scheduler_load, monitor) = if source.monitors_sequence_load() {
            let (scheduler_load, scheduler_load_rx) =
                scheduler_load_channel(source, cancellation_token.child_token());
            let monitor = KvWorkerMonitor::new(
                client.clone(),
                source,
                scheduler_load_rx,
                thresholds.clone(),
                cancellation_token.child_token(),
                task_guard.clone(),
            );
            monitor.start_monitoring().await?;
            (scheduler_load, Some(monitor))
        } else {
            (
                SchedulerLoadSender::disabled(source, cancellation_token.child_token()),
                None,
            )
        };

        Ok(Arc::new(Self {
            client,
            source,
            scheduler_load,
            thresholds,
            cancellation_token,
            monitor,
            _task_guard: task_guard,
        }))
    }

    pub fn client(&self) -> &Client {
        &self.client
    }

    pub fn source(&self) -> RouterLoadSource {
        self.source
    }

    pub fn scheduler_load_sender(&self) -> SchedulerLoadSender {
        self.scheduler_load.clone()
    }

    pub fn load_thresholds(&self) -> LoadThresholdHandle {
        self.thresholds.clone()
    }

    pub fn cancellation_token(&self) -> CancellationToken {
        self.cancellation_token.child_token()
    }

    pub fn monitor(&self) -> Option<&KvWorkerMonitor> {
        self.monitor.as_ref()
    }
}

impl Drop for RoutingLoadContext {
    fn drop(&mut self) {
        self.cancellation_token.cancel();
    }
}

/// Standalone KV selection surface plus the context that owns its load tasks.
pub struct ManagedKvRouter<Sel = dynamo_kv_router::selector::DefaultWorkerSelector>
where
    Sel: dynamo_kv_router::selector::WorkerSelector<ModelRuntimeConfig>,
{
    load_context: Arc<RoutingLoadContext>,
    router: Arc<KvRouter<Sel>>,
}

impl<Sel> Clone for ManagedKvRouter<Sel>
where
    Sel: dynamo_kv_router::selector::WorkerSelector<ModelRuntimeConfig>,
{
    fn clone(&self) -> Self {
        Self {
            load_context: self.load_context.clone(),
            router: self.router.clone(),
        }
    }
}

impl<Sel> std::ops::Deref for ManagedKvRouter<Sel>
where
    Sel: dynamo_kv_router::selector::WorkerSelector<ModelRuntimeConfig>,
{
    type Target = KvRouter<Sel>;

    fn deref(&self) -> &Self::Target {
        &self.router
    }
}

impl<Sel> ManagedKvRouter<Sel>
where
    Sel: dynamo_kv_router::selector::WorkerSelector<ModelRuntimeConfig>,
{
    pub fn new(load_context: Arc<RoutingLoadContext>, router: Arc<KvRouter<Sel>>) -> Self {
        Self {
            load_context,
            router,
        }
    }

    pub fn load_context(&self) -> &Arc<RoutingLoadContext> {
        &self.load_context
    }

    pub fn router(&self) -> &Arc<KvRouter<Sel>> {
        &self.router
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use dynamo_runtime::{DistributedRuntime, Runtime, distributed::DistributedConfig};

    fn snapshot(worker_id: u64, active_decode_blocks: u64) -> SchedulerLoadSnapshot {
        SchedulerLoadSnapshot {
            worker: WorkerWithDpRank::new(worker_id, 0),
            active_decode_blocks,
            active_prefill_tokens: 0,
        }
    }

    #[tokio::test]
    async fn saturated_channel_coalesces_batch_and_later_absolute_state_converges() {
        let token = CancellationToken::new();
        let (sender, mut receiver) =
            scheduler_load_channel_with_capacity(RouterLoadSource::Decode, token, 1);

        sender.publish(snapshot(1, 90));
        sender.publish_batch(vec![snapshot(1, 80), snapshot(2, 70)]);

        let expected = vec![snapshot(1, 90), snapshot(1, 80), snapshot(2, 70)];
        let received = tokio::time::timeout(std::time::Duration::from_secs(1), async {
            let mut received = Vec::new();
            while !expected.iter().all(|snapshot| received.contains(snapshot)) {
                received.extend(receiver.recv().await.unwrap());
            }
            received
        })
        .await
        .expect("queued and coalesced scheduler snapshots were not received");
        for snapshot in expected {
            assert!(received.contains(&snapshot));
        }

        sender.publish(snapshot(1, 0));
        assert_eq!(receiver.recv().await.unwrap(), vec![snapshot(1, 0)]);
    }

    #[tokio::test]
    async fn encode_context_retains_client_with_sequence_load_monitoring_disabled() {
        let runtime = Runtime::from_current().unwrap();
        let distributed =
            DistributedRuntime::new(runtime.clone(), DistributedConfig::process_local())
                .await
                .unwrap();
        let endpoint = distributed
            .namespace("encode-routing-load".to_string())
            .unwrap()
            .component("workers".to_string())
            .unwrap()
            .endpoint("generate".to_string());
        let client = endpoint.client().await.unwrap();
        let parent_token = distributed.child_token();

        let load_context = RoutingLoadContext::start(
            client.clone(),
            RouterLoadSource::Encode,
            LoadThresholdHandle::new(Default::default()),
            &parent_token,
            None,
        )
        .await
        .unwrap();

        assert_eq!(load_context.source(), RouterLoadSource::Encode);
        assert_eq!(load_context.client().endpoint.id(), client.endpoint.id());
        assert!(load_context.monitor().is_none());
        assert!(!load_context.scheduler_load_sender().is_enabled());
        load_context
            .scheduler_load_sender()
            .publish(snapshot(1, 100));
        assert_eq!(client.overloaded_instance_ids(), None);

        drop(load_context);
        assert!(!parent_token.is_cancelled());
        runtime.shutdown();
    }
}
