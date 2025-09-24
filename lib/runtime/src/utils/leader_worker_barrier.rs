// SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use crate::{
    DistributedRuntime,
    transports::etcd::{Client, WatchEvent},
};
use serde::{Serialize, de::DeserializeOwned};

use std::collections::{HashMap, HashSet};
use std::marker::PhantomData;
use std::time::{Duration, Instant};

fn barrier_key(id: &str, suffix: &str) -> String {
    format!("barrier/{}/{}", id, suffix)
}

const BARRIER_DATA: &str = "data";
const BARRIER_WORKER: &str = "worker";
const BARRIER_COMPLETE: &str = "complete";
const BARRIER_ABORT: &str = "abort";

/// Watches for a specific number of items to appear under a key prefix
async fn wait_for_key_count<T: DeserializeOwned>(
    client: &Client,
    key: String,
    expected_count: usize,
    timeout: Option<Duration>,
) -> Result<HashMap<String, T>, LeaderWorkerBarrierError> {
    let (_key, _watcher, mut rx) = client
        .kv_get_and_watch_prefix(&key)
        .await
        .map_err(LeaderWorkerBarrierError::EtcdError)?
        .dissolve();

    let mut data = HashMap::new();
    let start = Instant::now();
    let timeout = timeout.unwrap_or(Duration::MAX);

    loop {
        let elapsed = start.elapsed();
        if elapsed > timeout {
            return Err(LeaderWorkerBarrierError::Timeout);
        }

        let remaining_time = timeout.saturating_sub(elapsed);

        tokio::select! {
            Some(watch_event) = rx.recv() => {
                handle_watch_event(watch_event, &mut data)?;
            }
            _ = tokio::time::sleep(remaining_time) => {
                // Timeout occurred, continue to check count
            }
        }

        if data.len() == expected_count {
            return Ok(data);
        }
    }
}

/// Handles a single watch event by updating the data map
fn handle_watch_event<T: DeserializeOwned>(
    event: WatchEvent,
    data: &mut HashMap<String, T>,
) -> Result<(), LeaderWorkerBarrierError> {
    match event {
        WatchEvent::Put(kv) => {
            let key = kv.key_str().unwrap().to_string();
            let value =
                serde_json::from_slice(kv.value()).map_err(LeaderWorkerBarrierError::SerdeError)?;
            data.insert(key, value);
        }
        WatchEvent::Delete(kv) => {
            let key = kv.key_str().unwrap();
            data.remove(key);
        }
    }
    Ok(())
}

/// Creates a key-value pair in etcd, returning a specific error if the key already exists
async fn create_barrier_key<T: Serialize>(
    client: &Client,
    key: &str,
    data: T,
    lease_id: Option<i64>,
) -> Result<(), LeaderWorkerBarrierError> {
    let serialized_data =
        serde_json::to_vec(&data).map_err(LeaderWorkerBarrierError::SerdeError)?;

    // TODO: This can fail for many reasons, the most common of which is that the key already exists.
    // Currently, the ETCD client returns a very generic error, so we can't distinguish between the them.
    // For now, just assume it's because the key already exists.
    client
        .kv_create(key, serialized_data, lease_id)
        .await
        .map_err(|_| LeaderWorkerBarrierError::IdNotUnique)?;

    Ok(())
}

/// Waits for a single key to appear (used for completion/abort signals)
async fn wait_for_signal<T: DeserializeOwned>(
    client: &Client,
    key: String,
) -> Result<T, LeaderWorkerBarrierError> {
    let data = wait_for_key_count::<T>(client, key, 1, None).await?;
    Ok(data.into_values().next().unwrap())
}

#[derive(Debug)]
pub enum LeaderWorkerBarrierError {
    EtcdClientNotFound,
    IdNotUnique,
    EtcdError(anyhow::Error),
    SerdeError(serde_json::Error),
    Timeout,
    Aborted,
    AlreadyCompleted,
}

/// A barrier for a leader to wait for a specific number of workers to join.
pub struct LeaderBarrier<LeaderData, WorkerData> {
    barrier_id: String,
    num_workers: usize,
    timeout: Option<Duration>,
    marker: PhantomData<(LeaderData, WorkerData)>,
}

impl<LeaderData: Serialize + DeserializeOwned, WorkerData: Serialize + DeserializeOwned>
    LeaderBarrier<LeaderData, WorkerData>
{
    pub fn new(barrier_id: String, num_workers: usize, timeout: Option<Duration>) -> Self {
        Self {
            barrier_id,
            num_workers,
            timeout,
            marker: PhantomData,
        }
    }

    /// Synchronize the leader with the workers.
    ///
    /// The leader will publish the barrier data, and the workers will wait for the barrier data to appear.
    /// The leader will then signal completion or abort, and the workers will wait for the signal to appear.
    pub async fn sync(
        self,
        rt: &DistributedRuntime,
        data: &LeaderData,
    ) -> anyhow::Result<HashMap<String, WorkerData>, LeaderWorkerBarrierError> {
        let etcd_client = rt
            .etcd_client()
            .ok_or(LeaderWorkerBarrierError::EtcdClientNotFound)?;

        let lease_id = etcd_client.lease_id();

        // Publish barrier data
        self.publish_barrier_data(&etcd_client, data, lease_id)
            .await?;

        // Wait for workers to join
        let worker_result = self.wait_for_workers(&etcd_client).await;

        // Signal completion or abort
        self.signal_completion(&etcd_client, &worker_result, lease_id)
            .await?;

        worker_result.map(|r| {
            r.into_iter()
                .map(|(k, v)| (k.split("/").last().unwrap().to_string(), v))
                .collect()
        })
    }

    async fn publish_barrier_data(
        &self,
        client: &Client,
        data: &LeaderData,
        lease_id: i64,
    ) -> Result<(), LeaderWorkerBarrierError> {
        let key = barrier_key(&self.barrier_id, BARRIER_DATA);
        create_barrier_key(client, &key, data, Some(lease_id)).await
    }

    async fn wait_for_workers(
        &self,
        client: &Client,
    ) -> Result<HashMap<String, WorkerData>, LeaderWorkerBarrierError> {
        let key = barrier_key(&self.barrier_id, BARRIER_WORKER);
        let workers = wait_for_key_count(client, key, self.num_workers, self.timeout).await?;
        Ok(workers)
    }

    async fn signal_completion(
        &self,
        client: &Client,
        worker_result: &Result<HashMap<String, WorkerData>, LeaderWorkerBarrierError>,
        lease_id: i64,
    ) -> Result<(), LeaderWorkerBarrierError> {
        if let Ok(worker_result) = worker_result {
            let key = barrier_key(&self.barrier_id, BARRIER_COMPLETE);

            let workers = worker_result.keys().collect::<HashSet<_>>();

            create_barrier_key(client, &key, workers, Some(lease_id)).await?;
        } else {
            let key = barrier_key(&self.barrier_id, BARRIER_ABORT);
            create_barrier_key(client, &key, (), Some(lease_id)).await?;
        }

        Ok(())
    }
}

// A barrier to synchronize a worker with a leader.
pub struct WorkerBarrier<LeaderData, WorkerData> {
    barrier_id: String,
    worker_id: String,
    marker: PhantomData<(LeaderData, WorkerData)>,
}

impl<LeaderData: Serialize + DeserializeOwned, WorkerData: Serialize + DeserializeOwned>
    WorkerBarrier<LeaderData, WorkerData>
{
    pub fn new(barrier_id: String, worker_id: String) -> Self {
        Self {
            barrier_id,
            worker_id,
            marker: PhantomData,
        }
    }

    /// Synchronize the worker with the leader.
    ///
    /// The worker will wait for the barrier data to appear, and then register as a worker.
    /// The worker will then wait for the completion or abort signal to appear.
    ///
    /// If the leader signals completion, the worker will return the barrier data.
    /// If the leader signals abort, the worker will return an error.
    pub async fn sync(
        self,
        rt: &DistributedRuntime,
        data: &WorkerData,
    ) -> anyhow::Result<LeaderData, LeaderWorkerBarrierError> {
        let etcd_client = rt
            .etcd_client()
            .ok_or(LeaderWorkerBarrierError::EtcdClientNotFound)?;

        let lease_id = etcd_client.lease_id();

        // Get barrier data while watching for abort signal
        let barrier_data = self.get_barrier_data(&etcd_client).await?;

        // Register as a worker
        let worker_key = self.register_worker(&etcd_client, data, lease_id).await?;

        // Wait for completion or abort signal
        self.wait_for_completion(&etcd_client, worker_key).await?;

        Ok(barrier_data)
    }

    async fn get_barrier_data(
        &self,
        client: &Client,
    ) -> Result<LeaderData, LeaderWorkerBarrierError> {
        let data_key = barrier_key(&self.barrier_id, BARRIER_DATA);
        let abort_key = barrier_key(&self.barrier_id, BARRIER_ABORT);

        tokio::select! {
            result = wait_for_key_count::<LeaderData>(client, data_key, 1, None) => {
                result?.into_values().next()
                    .ok_or(LeaderWorkerBarrierError::EtcdError(anyhow::anyhow!("No data found")))
            }
            _ = wait_for_signal::<()>(client, abort_key) => {
                Err(LeaderWorkerBarrierError::Aborted)
            }
        }
    }

    async fn register_worker(
        &self,
        client: &Client,
        data: &WorkerData,
        lease_id: i64,
    ) -> Result<String, LeaderWorkerBarrierError> {
        let key = barrier_key(
            &self.barrier_id,
            &format!("{}/{}", BARRIER_WORKER, self.worker_id),
        );
        create_barrier_key(client, &key, data, Some(lease_id)).await?;
        Ok(key)
    }

    async fn wait_for_completion(
        &self,
        client: &Client,
        worker_key: String,
    ) -> Result<(), LeaderWorkerBarrierError> {
        let complete_key = barrier_key(&self.barrier_id, BARRIER_COMPLETE);
        let abort_key = barrier_key(&self.barrier_id, BARRIER_ABORT);

        tokio::select! {
            Ok(workers) = wait_for_signal::<HashSet<String>>(client, complete_key) => {
                if workers.contains(&worker_key) {
                    Ok(())
                } else {
                    Err(LeaderWorkerBarrierError::AlreadyCompleted)
                }
            },
            _ = wait_for_signal::<()>(client, abort_key) => Err(LeaderWorkerBarrierError::Aborted),
        }
    }
}

#[cfg(feature = "testing-etcd")]
#[cfg(test)]
mod tests {
    use super::*;

    use crate::Runtime;
    use tokio::task::JoinHandle;

    use std::sync::atomic::{AtomicU64, Ordering};

    fn unique_id() -> String {
        static COUNTER: AtomicU64 = AtomicU64::new(0);
        let id = COUNTER.fetch_add(1, Ordering::Relaxed);

        format!("test_{}", id)
    }

    #[tokio::test]
    async fn test_no_etcd() {
        let rt = Runtime::from_current().unwrap();
        let drt = DistributedRuntime::from_settings_without_discovery(rt.clone())
            .await
            .unwrap();

        assert!(drt.etcd_client().is_none());

        let barrier = LeaderBarrier::<String, String>::new("test".to_string(), 2, None);
        let worker = WorkerBarrier::<String, String>::new("test".to_string(), "worker".to_string());

        assert!(matches!(
            barrier.sync(&drt, &"test".to_string()).await,
            Err(LeaderWorkerBarrierError::EtcdClientNotFound)
        ));
        assert!(matches!(
            worker.sync(&drt, &"test".to_string()).await,
            Err(LeaderWorkerBarrierError::EtcdClientNotFound)
        ));
    }

    #[tokio::test]
    async fn test_simple() {
        let rt = Runtime::from_current().unwrap();
        let drt = DistributedRuntime::from_settings(rt.clone()).await.unwrap();

        let id = unique_id();

        let leader = LeaderBarrier::<String, String>::new(id.clone(), 1, None);
        let worker = WorkerBarrier::<String, String>::new(id.clone(), "worker".to_string());

        let drt_clone = drt.clone();
        let leader_join: JoinHandle<Result<(), LeaderWorkerBarrierError>> =
            tokio::spawn(async move {
                let worker_data = leader.sync(&drt_clone, &"test_data".to_string()).await?;
                assert_eq!(worker_data.len(), 1);
                assert_eq!(
                    worker_data.get("worker").unwrap(),
                    &"test_worker".to_string()
                );
                Ok(())
            });

        let worker_join: JoinHandle<Result<(), LeaderWorkerBarrierError>> =
            tokio::spawn(async move {
                let res = worker.sync(&drt, &"test_worker".to_string()).await?;
                assert_eq!(res, "test_data".to_string());

                Ok(())
            });

        let (leader_res, worker_res) = tokio::join!(leader_join, worker_join);

        assert!(matches!(leader_res, Ok(Ok(_))));
        assert!(matches!(worker_res, Ok(Ok(_))));
    }

    #[tokio::test]
    async fn test_duplicate_leader() {
        let rt = Runtime::from_current().unwrap();
        let drt = DistributedRuntime::from_settings(rt.clone()).await.unwrap();

        let id = unique_id();

        let leader1 = LeaderBarrier::<String, String>::new(id.clone(), 1, None);
        let leader2 = LeaderBarrier::<String, String>::new(id.clone(), 1, None);

        let worker = WorkerBarrier::<String, String>::new(id.clone(), "worker".to_string());

        let drt_clone = drt.clone();
        let leader1_join: JoinHandle<Result<(), LeaderWorkerBarrierError>> =
            tokio::spawn(async move {
                let worker_data = leader1.sync(&drt_clone, &"test_data".to_string()).await?;
                assert_eq!(worker_data.len(), 1);
                assert_eq!(
                    worker_data.get("worker").unwrap(),
                    &"test_worker".to_string()
                );

                // Now, try to sync leader 2.
                let leader2_res = leader2.sync(&drt_clone, &"test_data2".to_string()).await;

                // Leader 2 should fail because the barrier ID is the same as leader 1.
                assert!(matches!(
                    leader2_res,
                    Err(LeaderWorkerBarrierError::IdNotUnique)
                ));

                Ok(())
            });

        let worker_join: JoinHandle<Result<(), LeaderWorkerBarrierError>> =
            tokio::spawn(async move {
                let res = worker.sync(&drt, &"test_worker".to_string()).await?;
                assert_eq!(res, "test_data".to_string());

                Ok(())
            });

        let (leader1_res, worker_res) = tokio::join!(leader1_join, worker_join);

        assert!(matches!(leader1_res, Ok(Ok(_))));
        assert!(matches!(worker_res, Ok(Ok(_))));
    }

    #[tokio::test]
    async fn test_duplicate_worker() {
        let rt = Runtime::from_current().unwrap();
        let drt = DistributedRuntime::from_settings(rt.clone()).await.unwrap();

        let id = unique_id();

        let leader = LeaderBarrier::<String, String>::new(id.clone(), 1, None);
        let worker1 = WorkerBarrier::<String, String>::new(id.clone(), "worker".to_string());
        let worker2 = WorkerBarrier::<String, String>::new(id.clone(), "worker".to_string());

        let drt_clone = drt.clone();
        let leader_join: JoinHandle<Result<(), LeaderWorkerBarrierError>> =
            tokio::spawn(async move {
                let worker_data = leader.sync(&drt_clone, &"test_data".to_string()).await?;
                assert_eq!(worker_data.len(), 1);
                assert_eq!(
                    worker_data.get("worker").unwrap(),
                    &"test_worker_1".to_string()
                );

                Ok(())
            });

        let worker_join: JoinHandle<Result<(), LeaderWorkerBarrierError>> =
            tokio::spawn(async move {
                let leader_data = worker1.sync(&drt, &"test_worker_1".to_string()).await?;
                assert_eq!(leader_data, "test_data".to_string());

                let worker2_res = worker2.sync(&drt, &"test_worker_2".to_string()).await;

                assert!(matches!(
                    worker2_res,
                    Err(LeaderWorkerBarrierError::IdNotUnique)
                ));

                Ok(())
            });

        let (leader_res, worker_res) = tokio::join!(leader_join, worker_join);

        assert!(matches!(leader_res, Ok(Ok(_))));
        assert!(matches!(worker_res, Ok(Ok(_))));
    }

    #[tokio::test]
    async fn test_timeout() {
        let rt = Runtime::from_current().unwrap();
        let drt = DistributedRuntime::from_settings(rt.clone()).await.unwrap();

        let id = unique_id();

        let leader = LeaderBarrier::<(), ()>::new(id.clone(), 2, Some(Duration::from_millis(100)));
        let worker1 = WorkerBarrier::<(), ()>::new(id.clone(), "worker1".to_string());
        let worker2 = WorkerBarrier::<(), ()>::new(id.clone(), "worker2".to_string());

        let drt_clone = drt.clone();
        let leader_join: JoinHandle<Result<(), LeaderWorkerBarrierError>> =
            tokio::spawn(async move {
                let res = leader.sync(&drt_clone, &()).await;
                assert!(matches!(res, Err(LeaderWorkerBarrierError::Timeout)));

                Ok(())
            });

        let drt_clone = drt.clone();
        let worker1_join: JoinHandle<Result<(), LeaderWorkerBarrierError>> =
            tokio::spawn(async move {
                let res = worker1.sync(&drt_clone, &()).await;
                assert!(matches!(res, Err(LeaderWorkerBarrierError::Aborted)));

                Ok(())
            });

        let worker2_join: JoinHandle<Result<(), LeaderWorkerBarrierError>> =
            tokio::spawn(async move {
                tokio::time::sleep(Duration::from_millis(200)).await;
                let res = worker2.sync(&drt, &()).await;
                assert!(matches!(res, Err(LeaderWorkerBarrierError::Aborted)));

                Ok(())
            });

        let (leader_res, worker1_res, worker2_res) =
            tokio::join!(leader_join, worker1_join, worker2_join);

        assert!(matches!(leader_res, Ok(Ok(_))));
        assert!(matches!(worker1_res, Ok(Ok(_))));
        assert!(matches!(worker2_res, Ok(Ok(_))));
    }

    #[tokio::test]
    async fn test_serde_error() {
        let rt = Runtime::from_current().unwrap();
        let drt = DistributedRuntime::from_settings(rt.clone()).await.unwrap();

        let id = unique_id();

        // Get the leader to send a (), when the worker expects a String.
        let leader =
            LeaderBarrier::<(), String>::new(id.clone(), 1, Some(Duration::from_millis(100)));
        let worker1 = WorkerBarrier::<String, String>::new(id.clone(), "worker1".to_string());

        let drt_clone = drt.clone();
        let leader_join: JoinHandle<Result<(), LeaderWorkerBarrierError>> =
            tokio::spawn(async move {
                assert!(matches!(
                    leader.sync(&drt_clone, &()).await,
                    Err(LeaderWorkerBarrierError::Timeout)
                ));
                Ok(())
            });

        let worker_join: JoinHandle<Result<(), LeaderWorkerBarrierError>> =
            tokio::spawn(async move {
                assert!(matches!(
                    worker1.sync(&drt, &"test_worker".to_string()).await,
                    Err(LeaderWorkerBarrierError::SerdeError(_))
                ));

                Ok(())
            });

        let (leader_res, worker_res) = tokio::join!(leader_join, worker_join);

        assert!(matches!(leader_res, Ok(Ok(_))));
        assert!(matches!(worker_res, Ok(Ok(_))));
    }

    #[tokio::test]
    async fn test_too_many_workers() {
        let rt = Runtime::from_current().unwrap();
        let drt = DistributedRuntime::from_settings(rt.clone()).await.unwrap();

        let id = unique_id();

        let leader = LeaderBarrier::<(), ()>::new(id.clone(), 1, None);
        let worker1 = WorkerBarrier::<(), ()>::new(id.clone(), "worker1".to_string());
        let worker2 = WorkerBarrier::<(), ()>::new(id.clone(), "worker2".to_string());

        let drt_clone = drt.clone();
        let leader_join: JoinHandle<Result<(), LeaderWorkerBarrierError>> =
            tokio::spawn(async move {
                leader.sync(&drt_clone, &()).await?;
                Ok(())
            });

        let worker_join: JoinHandle<Result<(), LeaderWorkerBarrierError>> =
            tokio::spawn(async move {
                let drt_clone = drt.clone();
                let worker1_join = tokio::spawn(async move { worker1.sync(&drt_clone, &()).await });

                let worker2_join = tokio::spawn(async move { worker2.sync(&drt, &()).await });

                let (worker1_res, worker2_res) = tokio::join!(worker1_join, worker2_join);

                let mut num_successes = 0;
                for worker_res in [worker1_res, worker2_res] {
                    if let Ok(Ok(_)) = worker_res {
                        num_successes += 1;
                    } else if let Ok(Err(LeaderWorkerBarrierError::AlreadyCompleted)) = worker_res {
                    } else {
                        panic!();
                    }
                }

                assert_eq!(num_successes, 1);
                Ok(())
            });

        let (leader_res, worker_res) = tokio::join!(leader_join, worker_join);

        assert!(matches!(leader_res, Ok(Ok(_))));
        assert!(matches!(worker_res, Ok(Ok(_))));
    }
}
