// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::{
    sync::{Arc, OnceLock},
    thread::JoinHandle,
};

use tokio::sync::oneshot;

use super::pruning::{ApproximateTtlTask, PendingTtlStore, PreparedTtlTask, WorkerPruneManager};
use super::{KvRouterError, WorkerTask};
use crate::protocols::{WorkerId, WorkerWithDpRank};

pub(super) enum OrderedTask {
    Task(WorkerTask),
    Ttl(ApproximateTtlTask),
    Removal {
        task: WorkerTask,
        manager: WorkerPruneManager,
    },
}

enum Source {
    Plain(flume::Receiver<WorkerTask>),
    Retained(flume::Receiver<OrderedTask>),
}

enum Pending {
    Store(PendingTtlStore, oneshot::Receiver<bool>),
    WorkerRemoval {
        manager: WorkerPruneManager,
        worker_id: WorkerId,
        result: oneshot::Receiver<()>,
        response: oneshot::Sender<()>,
    },
    RankRemoval {
        manager: WorkerPruneManager,
        worker: WorkerWithDpRank,
    },
}

/// A blocking worker receiver that preserves the public [`WorkerTask`] protocol.
///
/// Retention bookkeeping runs between tasks on the mutation worker: successful
/// stores register TTL before their callers are acknowledged, and expiry checks
/// run immediately before removal. A backend must finish each returned task
/// before calling [`recv`](Self::recv) again. Use this receiver in
/// [`SyncIndexer::worker_with_retention`](super::SyncIndexer::worker_with_retention)
/// to avoid the compatibility dispatch thread used by that method's default.
/// Normal event-driven receivers can be converted with [`From`].
pub struct WorkerTaskReceiver {
    source: Source,
    pending: Option<Pending>,
    // Only the default adapter needs backend barriers and an independent stop
    // signal. Builtin workers execute the returned task themselves.
    legacy_backend: Option<flume::Sender<WorkerTask>>,
    legacy_queue: Arc<OnceLock<flume::Sender<WorkerTask>>>,
    stop: Option<flume::Receiver<()>>,
}

impl From<flume::Receiver<WorkerTask>> for WorkerTaskReceiver {
    fn from(receiver: flume::Receiver<WorkerTask>) -> Self {
        Self {
            source: Source::Plain(receiver),
            pending: None,
            legacy_backend: None,
            legacy_queue: Arc::default(),
            stop: None,
        }
    }
}

impl WorkerTaskReceiver {
    pub(super) fn retained(receiver: flume::Receiver<OrderedTask>) -> Self {
        Self {
            source: Source::Retained(receiver),
            pending: None,
            legacy_backend: None,
            legacy_queue: Arc::default(),
            stop: None,
        }
    }

    pub(super) fn legacy_queue(&self) -> Arc<OnceLock<flume::Sender<WorkerTask>>> {
        Arc::clone(&self.legacy_queue)
    }

    fn receive_ack<T>(&self, mut result: oneshot::Receiver<T>) -> Result<T, KvRouterError> {
        if self.legacy_backend.is_some() {
            result
                .blocking_recv()
                .map_err(|_| KvRouterError::IndexerDroppedRequest)
        } else {
            // The inline worker has already finished the previous task. An
            // absent acknowledgement is a broken contract, not a reason to wait
            // for the same worker to make progress while it is inside recv.
            result
                .try_recv()
                .map_err(|_| KvRouterError::IndexerDroppedRequest)
        }
    }

    fn legacy_barrier(&self) -> Result<(), KvRouterError> {
        if let Some(backend) = &self.legacy_backend {
            let (response, result) = oneshot::channel();
            backend
                .send(WorkerTask::Flush(response))
                .map_err(|_| KvRouterError::IndexerOffline)?;
            result
                .blocking_recv()
                .map_err(|_| KvRouterError::IndexerDroppedRequest)?;
        }
        Ok(())
    }

    fn complete_pending(&mut self) -> Result<(), KvRouterError> {
        match self.pending.take() {
            Some(Pending::Store(completion, result)) => {
                completion.complete(self.receive_ack(result))?
            }
            Some(Pending::WorkerRemoval {
                manager,
                worker_id,
                result,
                response,
            }) => {
                self.receive_ack(result)?;
                manager.remove_worker(worker_id);
                let _ = response.send(());
            }
            Some(Pending::RankRemoval { manager, worker }) => {
                self.legacy_barrier()?;
                manager.remove_worker_dp_rank(worker);
            }
            None => {}
        }
        Ok(())
    }

    /// Wait for the next task, completing retention bookkeeping for the previous
    /// task first. Call this only from the backend's blocking mutation loop.
    /// Disconnection or a missing required acknowledgement stops the receiver.
    pub fn recv(&mut self) -> Result<WorkerTask, KvRouterError> {
        self.complete_pending()?;
        loop {
            let message = match &self.source {
                Source::Plain(receiver) => {
                    return receiver.recv().map_err(|_| KvRouterError::IndexerOffline);
                }
                Source::Retained(receiver) => {
                    if let Some(stop) = &self.stop {
                        flume::Selector::new()
                            .recv(receiver, |result| result.ok())
                            .recv(stop, |_| None)
                            .wait()
                            .ok_or(KvRouterError::IndexerOffline)?
                    } else {
                        receiver.recv().map_err(|_| KvRouterError::IndexerOffline)?
                    }
                }
            };
            match message {
                OrderedTask::Task(task) => return Ok(task),
                OrderedTask::Ttl(task) => match task.prepare() {
                    PreparedTtlTask::Store(event, completion) => {
                        let (response, result) = oneshot::channel();
                        self.pending = Some(Pending::Store(completion, result));
                        return Ok(WorkerTask::EventWithAck {
                            event,
                            resp: response,
                        });
                    }
                    PreparedTtlTask::Remove(task) => {
                        self.legacy_barrier()?;
                        if let Some(event) = task.into_remove_event() {
                            return Ok(WorkerTask::Event(event));
                        }
                    }
                },
                OrderedTask::Removal { task, manager } => match task {
                    WorkerTask::RemoveWorker {
                        worker_id,
                        sweep_tree,
                        resp,
                    } => {
                        let (response, result) = oneshot::channel();
                        self.pending = Some(Pending::WorkerRemoval {
                            manager,
                            worker_id,
                            result,
                            response: resp,
                        });
                        return Ok(WorkerTask::RemoveWorker {
                            worker_id,
                            sweep_tree,
                            resp: response,
                        });
                    }
                    WorkerTask::RemoveWorkerDpRank {
                        worker_id,
                        dp_rank,
                        sweep_tree,
                    } => {
                        self.pending = Some(Pending::RankRemoval {
                            manager,
                            worker: WorkerWithDpRank::new(worker_id, dp_rank),
                        });
                        return Ok(WorkerTask::RemoveWorkerDpRank {
                            worker_id,
                            dp_rank,
                            sweep_tree,
                        });
                    }
                    _ => unreachable!("only removals carry TTL cleanup"),
                },
            }
        }
    }

    pub(super) fn run_legacy(
        mut self,
        worker: impl FnOnce(flume::Receiver<WorkerTask>) -> anyhow::Result<()>,
    ) -> anyhow::Result<()> {
        if let Source::Plain(receiver) = &self.source {
            // A caller can explicitly pass a converted ordinary receiver to the
            // provided trait method. No retention coordination is needed, and
            // an idle plain recv has no adapter stop channel to wake on return.
            return worker(receiver.clone());
        }
        let (backend, receiver) = flume::unbounded();
        let (stop, stopped) = flume::bounded(1);
        self.legacy_backend = Some(backend.clone());
        let _ = self.legacy_queue.set(backend.clone());
        self.stop = Some(stopped);
        let thread = std::thread::spawn(move || {
            let _terminate = TerminateBackendOnDrop(backend.clone());
            while let Ok(task) = self.recv() {
                let terminate = matches!(task, WorkerTask::Terminate);
                if backend.send(task).is_err() || terminate {
                    break;
                }
            }
        });
        // Joining from Drop also covers a backend panic. Its receiver and any
        // in-flight ack sender unwind before this guard stops the adapter.
        let _bridge = LegacyBridge {
            stop,
            thread: Some(thread),
        };
        worker(receiver)
    }
}

impl Drop for WorkerTaskReceiver {
    fn drop(&mut self) {
        // A backend may acknowledge its last mutation and return without another
        // recv. Settle ready acknowledgements, but never wait in Drop: the sender
        // might still be held by the exiting worker or another task.
        match self.pending.take() {
            Some(Pending::Store(completion, mut result)) => {
                let _ = completion.complete(
                    result
                        .try_recv()
                        .map_err(|_| KvRouterError::IndexerDroppedRequest),
                );
            }
            Some(Pending::WorkerRemoval {
                manager,
                worker_id,
                mut result,
                response,
            }) => {
                if result.try_recv().is_ok() {
                    manager.remove_worker(worker_id);
                    let _ = response.send(());
                }
            }
            // Rank removal has no public acknowledgement. If a worker exits
            // before the next recv, completion is unproven; leave its metadata
            // intact. The disconnected lane cannot accept restored coverage.
            Some(Pending::RankRemoval { .. }) | None => {}
        }
    }
}

struct LegacyBridge {
    stop: flume::Sender<()>,
    thread: Option<JoinHandle<()>>,
}

impl Drop for LegacyBridge {
    fn drop(&mut self) {
        let _ = self.stop.send(());
        if let Some(thread) = self.thread.take() {
            let _ = thread.join();
        }
    }
}

struct TerminateBackendOnDrop(flume::Sender<WorkerTask>);

impl Drop for TerminateBackendOnDrop {
    fn drop(&mut self) {
        let _ = self.0.send(WorkerTask::Terminate);
    }
}

#[cfg(test)]
mod tests {
    use super::super::pruning::{BlockEntry, PruneConfig};
    use super::*;
    use crate::protocols::ExternalSequenceBlockHash;
    use crate::test_utils::make_store_event;

    fn block(worker: WorkerWithDpRank) -> BlockEntry {
        BlockEntry {
            key: ExternalSequenceBlockHash(1),
            worker,
            seq_position: 0,
        }
    }

    #[test]
    fn legacy_adapter_returns_with_an_open_plain_input_channel() {
        let (sender, tasks) = flume::unbounded();
        let (done, result) = flume::bounded(1);
        let thread = std::thread::spawn(move || {
            WorkerTaskReceiver::from(tasks)
                .run_legacy(|_| Ok(()))
                .unwrap();
            let _ = done.send(());
        });
        result
            .recv_timeout(std::time::Duration::from_secs(2))
            .unwrap();
        thread.join().unwrap();
        assert!(sender.is_disconnected());
    }

    #[tokio::test]
    async fn dropping_ttl_receiver_does_not_wait_for_an_outstanding_store_ack() {
        let manager = WorkerPruneManager::new(PruneConfig::default());
        manager.shutdown();
        let worker = WorkerWithDpRank::new(7, 0);
        let (sender, tasks) = flume::unbounded();
        let (response, caller) = oneshot::channel();
        sender
            .send(OrderedTask::Ttl(ApproximateTtlTask::store(
                manager.clone(),
                make_store_event(7, &[1]),
                vec![block(worker)],
                response,
            )))
            .unwrap();
        let mut receiver = WorkerTaskReceiver::retained(tasks);
        let WorkerTask::EventWithAck { resp, .. } = receiver.recv().unwrap() else {
            panic!("expected store");
        };
        let (done, dropped) = oneshot::channel();
        let thread = std::thread::spawn(move || {
            drop(receiver);
            let _ = done.send(());
        });
        tokio::time::timeout(std::time::Duration::from_secs(2), dropped)
            .await
            .unwrap()
            .unwrap();
        thread.join().unwrap();
        assert!(caller.await.is_err());
        assert!(
            resp.send(true).is_err(),
            "late ack must not outlive receiver ownership"
        );
        assert!(
            manager
                .drain_due_and_pending(
                    tokio::time::Instant::now() + std::time::Duration::from_secs(300)
                )
                .is_empty()
        );
    }

    #[rstest::rstest]
    #[case::acknowledged(true)]
    #[case::unacknowledged(false)]
    #[tokio::test]
    async fn dropping_ttl_receiver_settles_only_acknowledged_worker_removal(
        #[case] acknowledged: bool,
    ) {
        let manager = WorkerPruneManager::new(PruneConfig::default());
        manager.shutdown();
        let worker = WorkerWithDpRank::new(7, 0);
        manager.insert_block_entries(vec![block(worker)]);
        let (sender, tasks) = flume::unbounded();
        let (response, caller) = oneshot::channel();
        sender
            .send(OrderedTask::Removal {
                task: WorkerTask::RemoveWorker {
                    worker_id: 7,
                    sweep_tree: true,
                    resp: response,
                },
                manager: manager.clone(),
            })
            .unwrap();
        let mut receiver = WorkerTaskReceiver::retained(tasks);
        let WorkerTask::RemoveWorker { resp, .. } = receiver.recv().unwrap() else {
            panic!("expected worker removal");
        };
        if acknowledged {
            resp.send(()).unwrap();
        } else {
            drop(resp);
        }
        drop(receiver);
        assert_eq!(caller.await.is_ok(), acknowledged);
        assert_eq!(
            manager
                .drain_due_and_pending(
                    tokio::time::Instant::now() + std::time::Duration::from_secs(300)
                )
                .len(),
            usize::from(!acknowledged)
        );
    }

    #[tokio::test]
    async fn dropping_ttl_receiver_does_not_assume_rank_removal_completed() {
        let manager = WorkerPruneManager::new(PruneConfig::default());
        manager.shutdown();
        let worker = WorkerWithDpRank::new(7, 0);
        manager.insert_block_entries(vec![block(worker)]);
        let (sender, tasks) = flume::unbounded();
        sender
            .send(OrderedTask::Removal {
                task: WorkerTask::RemoveWorkerDpRank {
                    worker_id: 7,
                    dp_rank: 0,
                    sweep_tree: true,
                },
                manager: manager.clone(),
            })
            .unwrap();
        let mut receiver = WorkerTaskReceiver::retained(tasks);
        assert!(matches!(
            receiver.recv().unwrap(),
            WorkerTask::RemoveWorkerDpRank { .. }
        ));
        drop(receiver);
        assert_eq!(
            manager
                .drain_due_and_pending(
                    tokio::time::Instant::now() + std::time::Duration::from_secs(300)
                )
                .len(),
            1
        );
        assert!(
            sender
                .send(OrderedTask::Task(WorkerTask::Event(make_store_event(
                    7,
                    &[1]
                ))))
                .is_err(),
            "retired lane cannot accept restored coverage"
        );
    }
}
