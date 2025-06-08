// SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

use crate::{
    transports::etcd::{Client, WatchEvent},
    DistributedRuntime,
};
use serde::{de::DeserializeOwned, Serialize};

use std::collections::HashMap;
use std::marker::PhantomData;
use std::time::{Duration, Instant};

fn barrier_key(id: &str, suffix: &str) -> String {
    format!("barrier/{}/{}", id, suffix)
}

const BARRIER_DATA: &str = "data";
const BARRIER_WORKER: &str = "worker";
const BARRIER_COMPLETE: &str = "complete";
const BARRIER_ABORT: &str = "abort";

async fn etcd_key_counter<T: DeserializeOwned>(
    client: &Client,
    key: String,
    num_items: usize,
    timeout: Option<Duration>,
) -> anyhow::Result<HashMap<String, T>, LeaderWorkerBarrierError> {
    let (_key, _watcher, mut rx) = client
        .kv_get_and_watch_prefix(&key)
        .await
        .map_err(LeaderWorkerBarrierError::EtcdError)?
        .dissolve();
    let timeout = timeout.unwrap_or(Duration::MAX);

    let start = Instant::now();

    let mut data = HashMap::new();

    loop {
        let elapsed = start.elapsed();
        if elapsed > timeout {
            return Err(LeaderWorkerBarrierError::Timeout);
        }

        let remaining_time = timeout - elapsed;

        tokio::select! {
            Some(watch_event) = rx.recv() => {
                match watch_event {
                    WatchEvent::Put(kv) => {
                        data.insert(kv.key_str().unwrap().to_string(), serde_json::from_slice::<T>(kv.value()).map_err(LeaderWorkerBarrierError::SerdeError)?);
                    }
                    WatchEvent::Delete(kv) => {
                        data.remove(kv.key_str().unwrap());
                    }
                }
            }
            _ = tokio::time::sleep(remaining_time) => {}
        }

        if data.len() == num_items {
            return Ok(data);
        }
    }
}

async fn etcd_kv_create<T: Serialize>(
    client: &Client,
    key: String,
    data: T,
    lease_id: Option<i64>,
    e: LeaderWorkerBarrierError,
) -> Result<(), LeaderWorkerBarrierError> {
    client
        .kv_create(
            key,
            serde_json::to_vec(&data).map_err(LeaderWorkerBarrierError::SerdeError)?,
            lease_id,
        )
        .await
        .map_err(|_| e)?;

    Ok(())
}

#[derive(Debug)]
pub enum LeaderWorkerBarrierError {
    EtcdClientNotFound,
    BarrierIdNotUnique,
    BarrierWorkerIdNotUnique,
    EtcdError(anyhow::Error),
    SerdeError(serde_json::Error),
    Timeout,
    Aborted,
}

pub struct LeaderBarrier<T> {
    barrier_id: String,
    num_workers: usize,
    timeout: Option<Duration>,
    marker: PhantomData<T>,
}

impl<T: Serialize + DeserializeOwned> LeaderBarrier<T> {
    pub fn new(barrier_id: String, num_workers: usize, timeout: Option<Duration>) -> Self {
        Self {
            barrier_id,
            num_workers,
            timeout,
            marker: PhantomData,
        }
    }

    pub async fn sync(
        self,
        rt: &DistributedRuntime,
        data: &T,
    ) -> anyhow::Result<(), LeaderWorkerBarrierError> {
        let etcd_client = rt
            .etcd_client()
            .ok_or(LeaderWorkerBarrierError::EtcdClientNotFound)?;

        let lease_id = etcd_client.lease_id();

        // Publish our barrier data.
        let barrier_data_key = barrier_key(&self.barrier_id, BARRIER_DATA);
        etcd_kv_create(
            &etcd_client,
            barrier_data_key,
            data,
            Some(lease_id),
            LeaderWorkerBarrierError::BarrierIdNotUnique,
        )
        .await?;

        // Create a watcher for workers.
        let barrier_worker_key = barrier_key(&self.barrier_id, BARRIER_WORKER);
        let worker_watcher = etcd_key_counter::<()>(
            &etcd_client,
            barrier_worker_key,
            self.num_workers,
            self.timeout,
        );

        // Wait for all workers to join, or for timeout.
        let result = worker_watcher.await;

        // If there was an error, abort.
        let suffix = if result.is_err() {
            BARRIER_ABORT
        } else {
            BARRIER_COMPLETE
        };

        let key = barrier_key(&self.barrier_id, suffix);

        // Publish the completion or abort signal.
        etcd_kv_create(
            &etcd_client,
            key,
            (),
            Some(lease_id),
            LeaderWorkerBarrierError::BarrierIdNotUnique,
        )
        .await?;

        result.map(|_| ())
    }
}

pub struct WorkerBarrier<T> {
    barrier_id: String,
    worker_id: String,
    marker: PhantomData<T>,
}

impl<T: Serialize + DeserializeOwned> WorkerBarrier<T> {
    pub fn new(barrier_id: String, worker_id: String) -> Self {
        Self {
            barrier_id,
            worker_id,
            marker: PhantomData,
        }
    }

    pub async fn sync(
        self,
        rt: &DistributedRuntime,
    ) -> anyhow::Result<T, LeaderWorkerBarrierError> {
        let etcd_client = rt
            .etcd_client()
            .ok_or(LeaderWorkerBarrierError::EtcdClientNotFound)?;

        let lease_id = etcd_client.lease_id();

        let barrier_abort_key = barrier_key(&self.barrier_id, BARRIER_ABORT);
        let barrier_complete_key = barrier_key(&self.barrier_id, BARRIER_COMPLETE);
        let barrier_data_key = barrier_key(&self.barrier_id, BARRIER_DATA);
        let barrier_worker_key = barrier_key(
            &self.barrier_id,
            &format!("{}/{}", BARRIER_WORKER, self.worker_id),
        );

        // First, try to get the barrier data.
        let barrier_data = tokio::select! {
            res = etcd_key_counter::<T>(&etcd_client, barrier_data_key, 1, None) => {
                if let Ok(data) = res {
                    data.into_values().next().unwrap()
                } else {
                    return Err(res.err().unwrap())
                }
            }
            _ = etcd_key_counter::<()>(&etcd_client, barrier_abort_key.clone(), 1, None) => {
                return Err(LeaderWorkerBarrierError::Aborted)
            }
        };

        // Register our worker.
        etcd_kv_create(
            &etcd_client,
            barrier_worker_key,
            (),
            Some(lease_id),
            LeaderWorkerBarrierError::BarrierWorkerIdNotUnique,
        )
        .await?;

        let complete_watcher = etcd_key_counter::<()>(&etcd_client, barrier_complete_key, 1, None);
        let abort_watcher = etcd_key_counter::<()>(&etcd_client, barrier_abort_key, 1, None);

        tokio::select! {
            _ = complete_watcher => {
                Ok(barrier_data)
            }
            _ = abort_watcher => {
                Err(LeaderWorkerBarrierError::Aborted)
            }
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

        let barrier = LeaderBarrier::new("test".to_string(), 2, None);
        let worker = WorkerBarrier::<()>::new("test".to_string(), "worker".to_string());

        assert!(matches!(
            barrier.sync(&drt, &"test".to_string()).await,
            Err(LeaderWorkerBarrierError::EtcdClientNotFound)
        ));
        assert!(matches!(
            worker.sync(&drt).await,
            Err(LeaderWorkerBarrierError::EtcdClientNotFound)
        ));
    }

    #[tokio::test]
    async fn test_simple() {
        let rt = Runtime::from_current().unwrap();
        let drt = DistributedRuntime::from_settings(rt.clone()).await.unwrap();

        let id = unique_id();

        let leader = LeaderBarrier::new(id.clone(), 1, None);
        let worker = WorkerBarrier::<String>::new(id.clone(), "worker".to_string());

        let drt_clone = drt.clone();
        let leader_join: JoinHandle<Result<(), LeaderWorkerBarrierError>> =
            tokio::spawn(async move {
                leader.sync(&drt_clone, &"test_data".to_string()).await?;
                Ok(())
            });

        let worker_join: JoinHandle<Result<(), LeaderWorkerBarrierError>> =
            tokio::spawn(async move {
                let res = worker.sync(&drt).await?;
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

        let leader1 = LeaderBarrier::new(id.clone(), 1, None);
        let leader2 = LeaderBarrier::new(id.clone(), 1, None);

        let worker = WorkerBarrier::<String>::new(id.clone(), "worker".to_string());

        let drt_clone = drt.clone();
        let leader1_join: JoinHandle<Result<(), LeaderWorkerBarrierError>> =
            tokio::spawn(async move {
                leader1.sync(&drt_clone, &"test_data".to_string()).await?;

                // Now, try to sync leader 2.
                let leader2_res = leader2.sync(&drt_clone, &"test_data2".to_string()).await;

                // Leader 2 should fail because the barrier ID is the same as leader 1.
                assert!(matches!(
                    leader2_res,
                    Err(LeaderWorkerBarrierError::BarrierIdNotUnique)
                ));

                Ok(())
            });

        let worker_join: JoinHandle<Result<(), LeaderWorkerBarrierError>> =
            tokio::spawn(async move {
                let res = worker.sync(&drt).await?;
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

        let leader = LeaderBarrier::new(id.clone(), 1, None);
        let worker1 = WorkerBarrier::<String>::new(id.clone(), "worker".to_string());
        let worker2 = WorkerBarrier::<String>::new(id.clone(), "worker".to_string());

        let drt_clone = drt.clone();
        let leader_join: JoinHandle<Result<(), LeaderWorkerBarrierError>> =
            tokio::spawn(async move {
                leader.sync(&drt_clone, &"test_data".to_string()).await?;
                Ok(())
            });

        let worker_join: JoinHandle<Result<(), LeaderWorkerBarrierError>> =
            tokio::spawn(async move {
                worker1.sync(&drt).await?;

                let worker2_res = worker2.sync(&drt).await;

                assert!(matches!(
                    worker2_res,
                    Err(LeaderWorkerBarrierError::BarrierWorkerIdNotUnique)
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

        let leader = LeaderBarrier::new(id.clone(), 2, Some(Duration::from_millis(100)));
        let worker1 = WorkerBarrier::<()>::new(id.clone(), "worker1".to_string());
        let worker2 = WorkerBarrier::<()>::new(id.clone(), "worker2".to_string());

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
                let res = worker1.sync(&drt_clone).await;
                assert!(matches!(res, Err(LeaderWorkerBarrierError::Aborted)));

                Ok(())
            });

        let worker2_join: JoinHandle<Result<(), LeaderWorkerBarrierError>> =
            tokio::spawn(async move {
                tokio::time::sleep(Duration::from_millis(200)).await;
                let res = worker2.sync(&drt).await;
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
        let leader = LeaderBarrier::new(id.clone(), 1, Some(Duration::from_millis(100)));
        let worker1 = WorkerBarrier::<String>::new(id.clone(), "worker1".to_string());

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
                    worker1.sync(&drt).await,
                    Err(LeaderWorkerBarrierError::SerdeError(_))
                ));

                Ok(())
            });

        let (leader_res, worker_res) = tokio::join!(leader_join, worker_join);

        assert!(matches!(leader_res, Ok(Ok(_))));
        assert!(matches!(worker_res, Ok(Ok(_))));
    }
}
