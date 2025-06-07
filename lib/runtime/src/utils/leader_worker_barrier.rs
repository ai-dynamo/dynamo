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

async fn etcd_key_counter<T: DeserializeOwned>(
    client: &Client,
    key: String,
    num_items: usize,
    timeout: Duration,
) -> anyhow::Result<HashMap<String, T>, LeaderWorkerBarrierError> {
    let (_key, _watcher, mut rx) = client
        .kv_get_and_watch_prefix(&key)
        .await
        .map_err(LeaderWorkerBarrierError::EtcdError)?
        .dissolve();

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

pub enum LeaderWorkerBarrierError {
    EtcdClientNotFound,
    BarrierIdNotUnique,
    EtcdError(anyhow::Error),
    SerdeError(serde_json::Error),
    Timeout,
}

pub struct LeaderBarrier<T> {
    barrier_id: String,
    num_workers: usize,
    timeout: Duration,
    marker: PhantomData<T>,
}

impl<T: Serialize + DeserializeOwned> LeaderBarrier<T> {
    pub fn new(barrier_id: String, num_workers: usize, timeout: Duration) -> Self {
        Self {
            barrier_id,
            num_workers,
            timeout,
            marker: PhantomData,
        }
    }

    // Wait at the barrier,
    pub async fn wait(
        self,
        rt: &DistributedRuntime,
        data: &T,
    ) -> anyhow::Result<(), LeaderWorkerBarrierError> {
        let etcd_client = rt
            .etcd_client()
            .ok_or(LeaderWorkerBarrierError::EtcdClientNotFound)?;

        let lease_id = etcd_client.lease_id();

        // Publish our barrier data.
        let barrier_data_key = barrier_key(&self.barrier_id, "data");
        etcd_client
            .kv_create(
                barrier_data_key,
                serde_json::to_vec(data).map_err(LeaderWorkerBarrierError::SerdeError)?,
                Some(lease_id),
            )
            .await
            .map_err(|_| LeaderWorkerBarrierError::BarrierIdNotUnique)?;

        // Create a watcher for workers.
        let barrier_worker_key = barrier_key(&self.barrier_id, "worker");
        let worker_watcher = etcd_key_counter::<()>(
            &etcd_client,
            barrier_worker_key,
            self.num_workers,
            self.timeout,
        );

        // Wait for all workers to join.
        worker_watcher.await?;

        // Publish completion.
        let barrier_complete_key = barrier_key(&self.barrier_id, "complete");
        etcd_client
            .kv_create(
                barrier_complete_key,
                serde_json::to_vec(&()).map_err(LeaderWorkerBarrierError::SerdeError)?,
                Some(lease_id),
            )
            .await
            .map_err(|_| LeaderWorkerBarrierError::BarrierIdNotUnique)?;

        Ok(())
    }
}

pub struct WorkerBarrier<T> {
    barrier_id: String,
    worker_id: String,
    timeout: Duration,
    marker: PhantomData<T>,
}

impl<T: Serialize + DeserializeOwned> WorkerBarrier<T> {
    pub fn new(barrier_id: String, worker_id: String, timeout: Duration) -> Self {
        Self {
            barrier_id,
            worker_id,
            timeout,
            marker: PhantomData,
        }
    }

    pub async fn wait(
        self,
        rt: &DistributedRuntime,
    ) -> anyhow::Result<T, LeaderWorkerBarrierError> {
        let etcd_client = rt
            .etcd_client()
            .ok_or(LeaderWorkerBarrierError::EtcdClientNotFound)?;

        let lease_id = etcd_client.lease_id();

        let start = Instant::now();

        let barrier_data_key = barrier_key(&self.barrier_id, "data");
        let barrier_data = etcd_key_counter::<T>(&etcd_client, barrier_data_key, 1, self.timeout)
            .await?
            .into_values()
            .next()
            .unwrap();

        // Register our worker.
        let barrier_worker_key =
            barrier_key(&self.barrier_id, &format!("worker/{}", self.worker_id));
        etcd_client
            .kv_create(
                barrier_worker_key,
                serde_json::to_vec(&()).map_err(LeaderWorkerBarrierError::SerdeError)?,
                Some(lease_id),
            )
            .await
            .map_err(|_| LeaderWorkerBarrierError::BarrierIdNotUnique)?;

        let barrier_complete_key = barrier_key(&self.barrier_id, "complete");
        etcd_key_counter::<()>(
            &etcd_client,
            barrier_complete_key,
            1,
            self.timeout - start.elapsed(),
        )
        .await?;

        Ok(barrier_data)
    }
}

pub enum LeaderWorkerBarrierType {
    Leader,
    Worker,
}
