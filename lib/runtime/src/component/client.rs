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

use crate::pipeline::{
    network::egress::push::{AddressedPushRouter, AddressedRequest, PushRouter},
    AsyncEngine, Data, ManyOut, SingleIn,
};
use rand::Rng;
use std::collections::HashMap;
use std::sync::{
    atomic::{AtomicU64, Ordering},
    Arc,
};
use tokio::{net::unix::pipe::Receiver, sync::Mutex};

use crate::{pipeline::async_trait, transports::etcd::WatchEvent, Error};

use super::*;

/// Each state will be have a nonce associated with it
/// The state will be emitted in a watch channel, so we can observe the
/// critical state transitions.
enum MapState {
    /// The map is empty; value = nonce
    Empty(u64),

    /// The map is not-empty; values are (nonce, count)
    NonEmpty(u64, u64),

    /// The watcher has finished, no more events will be emitted
    Finished,
}

enum EndpointEvent {
    Put(String, i64),
    Delete(String),
}

#[derive(Default, Debug, Clone, Copy)]
pub enum RouterMode {
    #[default]
    Random,
    RoundRobin,
    //KV,
    //
    // Always and only go to the given endpoint ID.
    // TODO: Is this useful?
    Direct(i64),
}

#[derive(Clone)]
pub struct Client<T: Data, U: Data> {
    endpoint: Endpoint,
    router: PushRouter<T, U>,
    counter: Arc<AtomicU64>,
    endpoints: EndpointSource,
    router_mode: RouterMode,
}

#[derive(Clone, Debug)]
enum EndpointSource {
    Static,
    Dynamic(tokio::sync::watch::Receiver<Vec<i64>>),
}

impl<T, U> Client<T, U>
where
    T: Data + Serialize,
    U: Data + for<'de> Deserialize<'de>,
{
    // Client will only talk to a single static endpoint
    pub(crate) async fn new_static(endpoint: Endpoint) -> Result<Self> {
        Ok(Client {
            router: router(&endpoint).await?,
            endpoint,
            counter: Arc::new(AtomicU64::new(0)),
            endpoints: EndpointSource::Static,
            router_mode: Default::default(),
        })
    }

    // Client with auto-discover endpoints using etcd
    pub(crate) async fn new_dynamic(endpoint: Endpoint) -> Result<Self> {
        // create live endpoint watcher
        let Some(etcd_client) = &endpoint.component.drt.etcd_client else {
            anyhow::bail!("Attempt to create a dynamic client on a static endpoint");
        };
        let prefix_watcher = etcd_client
            .kv_get_and_watch_prefix(endpoint.etcd_path())
            .await?;

        let (prefix, _watcher, mut kv_event_rx) = prefix_watcher.dissolve();

        let (watch_tx, watch_rx) = tokio::sync::watch::channel(vec![]);

        let secondary = endpoint.component.drt.runtime.secondary().clone();

        // this task should be included in the registry
        // currently this is created once per client, but this object/task should only be instantiated
        // once per worker/instance
        secondary.spawn(async move {
            tracing::debug!("Starting endpoint watcher for prefix: {}", prefix);
            let mut map = HashMap::new();

            loop {
                let kv_event = tokio::select! {
                    _ = watch_tx.closed() => {
                        tracing::debug!("all watchers have closed; shutting down endpoint watcher for prefix: {}", prefix);
                        break;
                    }
                    kv_event = kv_event_rx.recv() => {
                        match kv_event {
                            Some(kv_event) => kv_event,
                            None => {
                                tracing::debug!("watch stream has closed; shutting down endpoint watcher for prefix: {}", prefix);
                                break;
                            }
                        }
                    }
                };

                match kv_event {
                    WatchEvent::Put(kv) => {
                        let key = String::from_utf8(kv.key().to_vec());
                        let val = serde_json::from_slice::<ComponentEndpointInfo>(kv.value());
                        if let (Ok(key), Ok(val)) = (key, val) {
                            map.insert(key.clone(), val.lease_id);
                        } else {
                            tracing::error!("Unable to parse put endpoint event; shutting down endpoint watcher for prefix: {}", prefix);
                            break;
                        }
                    }
                    WatchEvent::Delete(kv) => {
                        match String::from_utf8(kv.key().to_vec()) {
                            Ok(key) => { map.remove(&key); }
                            Err(_) => {
                                tracing::error!("Unable to parse delete endpoint event; shutting down endpoint watcher for prefix: {}", prefix);
                                break;
                            }
                        }
                    }
                }

                let endpoint_ids: Vec<i64> = map.values().cloned().collect();

                if watch_tx.send(endpoint_ids).is_err() {
                    tracing::debug!("Unable to send watch updates; shutting down endpoint watcher for prefix: {}", prefix);
                    break;
                }

            }

            tracing::debug!("Completed endpoint watcher for prefix: {}", prefix);
            let _ = watch_tx.send(vec![]);
        });

        Ok(Client {
            router: router(&endpoint).await?,
            endpoint,
            counter: Arc::new(AtomicU64::new(0)),
            endpoints: EndpointSource::Dynamic(watch_rx),
            router_mode: Default::default(),
        })
    }

    /// String identifying `<namespace>/<component>/<endpoint>`
    pub fn path(&self) -> String {
        self.endpoint.path()
    }

    /// String identifying `<namespace>/component/<component>/<endpoint>`
    pub fn etcd_path(&self) -> String {
        self.endpoint.etcd_path()
    }

    pub fn endpoint_ids(&self) -> Vec<i64> {
        match &self.endpoints {
            EndpointSource::Static => vec![0],
            EndpointSource::Dynamic(watch_rx) => watch_rx.borrow().clone(),
        }
    }

    pub fn set_router_mode(&mut self, mode: RouterMode) {
        self.router_mode = mode
    }

    /// Wait for at least one [`Endpoint`] to be available
    pub async fn wait_for_endpoints(&self) -> Result<()> {
        if let EndpointSource::Dynamic(mut rx) = self.endpoints.clone() {
            // wait for there to be 1 or more endpoints
            loop {
                if rx.borrow_and_update().is_empty() {
                    rx.changed().await?;
                } else {
                    break;
                }
            }
        }
        Ok(())
    }

    /// Is this component know at startup and not discovered via etcd?
    pub fn is_static(&self) -> bool {
        matches!(self.endpoints, EndpointSource::Static)
    }

    /// Issue a request to the next available endpoint in a round-robin fashion
    pub async fn round_robin(&self, request: SingleIn<T>) -> Result<ManyOut<U>> {
        let counter = self.counter.fetch_add(1, Ordering::Relaxed);

        let endpoint_id = {
            let endpoints = self.endpoint_ids();
            let count = endpoints.len();
            if count == 0 {
                return Err(error!(
                    "no endpoints found for endpoint {:?}",
                    self.endpoint.etcd_path()
                ));
            }
            let offset = counter % count as u64;
            endpoints[offset as usize]
        };
        tracing::trace!("round robin router selected {endpoint_id}");

        let subject = self.endpoint.subject_to(endpoint_id);
        let request = request.map(|req| AddressedRequest::new(req, subject));

        self.router.generate(request).await
    }

    /// Issue a request to a random endpoint
    pub async fn random(&self, request: SingleIn<T>) -> Result<ManyOut<U>> {
        let endpoint_id = {
            let endpoints = self.endpoint_ids();
            let count = endpoints.len();
            if count == 0 {
                return Err(error!(
                    "no endpoints found for endpoint {:?}",
                    self.endpoint.etcd_path()
                ));
            }
            let counter = rand::rng().random::<u64>();
            let offset = counter % count as u64;
            endpoints[offset as usize]
        };
        tracing::trace!("random router selected {endpoint_id}");

        let subject = self.endpoint.subject_to(endpoint_id);
        let request = request.map(|req| AddressedRequest::new(req, subject));

        self.router.generate(request).await
    }

    /// Issue a request to a specific endpoint
    pub async fn direct(&self, request: SingleIn<T>, endpoint_id: i64) -> Result<ManyOut<U>> {
        let found = {
            let endpoints = self.endpoint_ids();
            endpoints.contains(&endpoint_id)
        };

        if !found {
            return Err(error!(
                "endpoint_id={} not found for endpoint {:?}",
                endpoint_id,
                self.endpoint.etcd_path()
            ));
        }

        let subject = self.endpoint.subject_to(endpoint_id);
        let request = request.map(|req| AddressedRequest::new(req, subject));

        self.router.generate(request).await
    }

    pub async fn r#static(&self, request: SingleIn<T>) -> Result<ManyOut<U>> {
        let subject = self.endpoint.subject();
        tracing::debug!("static got subject: {subject}");
        let request = request.map(|req| AddressedRequest::new(req, subject));
        tracing::debug!("router generate");
        self.router.generate(request).await
    }
}

async fn router(endpoint: &Endpoint) -> Result<Arc<AddressedPushRouter>> {
    AddressedPushRouter::new(
        endpoint.component.drt.nats_client.client().clone(),
        endpoint.component.drt.tcp_server().await?,
    )
}

#[async_trait]
impl<T, U> AsyncEngine<SingleIn<T>, ManyOut<U>, Error> for Client<T, U>
where
    T: Data + Serialize,
    U: Data + for<'de> Deserialize<'de>,
{
    async fn generate(&self, request: SingleIn<T>) -> Result<ManyOut<U>, Error> {
        match &self.endpoints {
            EndpointSource::Static => self.r#static(request).await,
            EndpointSource::Dynamic(_) => match self.router_mode {
                RouterMode::Random => self.random(request).await,
                RouterMode::RoundRobin => self.round_robin(request).await,
                RouterMode::Direct(endpoint_id) => self.direct(request, endpoint_id).await,
            },
        }
    }
}
