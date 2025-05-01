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
    AddressedPushRouter, AddressedRequest, AsyncEngine, Data, ManyOut, PushRouter, RouterMode,
    SingleIn,
};
use rand::Rng;
use std::collections::HashMap;
use std::sync::{
    atomic::{AtomicU64, Ordering},
    Arc,
};
use tokio::{net::unix::pipe::Receiver, sync::Mutex};

use crate::{pipeline::async_trait, transports::etcd::WatchEvent};

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

#[derive(Clone, Debug)]
pub struct Client {
    // This is me
    pub endpoint: Endpoint,
    // These are the remotes I know about
    pub endpoints: EndpointSource,
}

#[derive(Clone, Debug)]
pub enum EndpointSource {
    Static,
    Dynamic(tokio::sync::watch::Receiver<Vec<ComponentEndpointInfo>>),
}

impl Client {
    // Client will only talk to a single static endpoint
    pub(crate) async fn new_static(endpoint: Endpoint) -> Result<Self> {
        Ok(Client {
            endpoint,
            endpoints: EndpointSource::Static,
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
                            map.insert(key.clone(), val);
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

                let endpoints: Vec<ComponentEndpointInfo> = map.values().cloned().collect();

                if watch_tx.send(endpoints).is_err() {
                    tracing::debug!("Unable to send watch updates; shutting down endpoint watcher for prefix: {}", prefix);
                    break;
                }

            }

            tracing::debug!("Completed endpoint watcher for prefix: {}", prefix);
            let _ = watch_tx.send(vec![]);
        });

        Ok(Client {
            endpoint,
            endpoints: EndpointSource::Dynamic(watch_rx),
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

    pub fn endpoints(&self) -> Vec<ComponentEndpointInfo> {
        match &self.endpoints {
            EndpointSource::Static => vec![],
            EndpointSource::Dynamic(watch_rx) => watch_rx.borrow().clone(),
        }
    }

    pub fn endpoint_ids(&self) -> Vec<i64> {
        self.endpoints().into_iter().map(|ep| ep.id()).collect()
    }

    /// Wait for at least one [`Endpoint`] to be available
    pub async fn wait_for_endpoints(&self) -> Result<Vec<ComponentEndpointInfo>> {
        let mut endpoints: Vec<ComponentEndpointInfo> = vec![];
        if let EndpointSource::Dynamic(mut rx) = self.endpoints.clone() {
            // wait for there to be 1 or more endpoints
            loop {
                endpoints = rx.borrow_and_update().to_vec();
                if endpoints.is_empty() {
                    rx.changed().await?;
                } else {
                    break;
                }
            }
        }
        Ok(endpoints)
    }

    /// Is this component know at startup and not discovered via etcd?
    pub fn is_static(&self) -> bool {
        matches!(self.endpoints, EndpointSource::Static)
    }
}
