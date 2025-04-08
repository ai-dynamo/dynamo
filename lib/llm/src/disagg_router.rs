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

use serde::{Deserialize, Serialize};
use std::sync::{Arc, Mutex};
use tokio::sync::watch;
use tracing;

use dynamo_runtime::transports::etcd::WatchEvent;
use dynamo_runtime::DistributedRuntime;

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct DisaggRouterConf {
    pub max_local_prefill_length: i32,
}

impl Default for DisaggRouterConf {
    fn default() -> Self {
        Self {
            max_local_prefill_length: 1000,
        }
    }
}

impl DisaggRouterConf {
    pub async fn from_etcd_with_watcher(
        drt: Arc<DistributedRuntime>,
        model_name: &str,
    ) -> anyhow::Result<(Self, watch::Receiver<Self>)> {
        let etcd_key = format!("public/components/disagg_router/models/chat/{}", model_name);

        // Get the initial value if it exists
        let Some(etcd_client) = drt.etcd_client() else {
            anyhow::bail!("Static components don't have an etcd client");
        };
        let initial_config = match etcd_client.kv_get_prefix(&etcd_key).await {
            Ok(kvs) => {
                if let Some(kv) = kvs.first() {
                    match serde_json::from_slice::<DisaggRouterConf>(kv.value()) {
                        Ok(config) => {
                            tracing::debug!(
                                "Found initial config for key {}: {:?}",
                                etcd_key,
                                config
                            );
                            config
                        }
                        Err(e) => {
                            tracing::warn!(
                                "Failed to parse initial config for key {}: {}",
                                etcd_key,
                                e
                            );
                            DisaggRouterConf::default()
                        }
                    }
                } else {
                    tracing::debug!(
                        "No initial config found for key {}, using default",
                        etcd_key
                    );
                    DisaggRouterConf::default()
                }
            }
            Err(e) => {
                tracing::warn!("Error fetching initial config for key {}: {}", etcd_key, e);
                DisaggRouterConf::default()
            }
        };

        // Create watch channel for config updates
        let (watch_tx, watch_rx) = watch::channel(initial_config.clone());

        // Set up the watcher after getting the initial value
        let prefix_watcher = etcd_client.kv_get_and_watch_prefix(&etcd_key).await?;
        let (key, _watcher, mut kv_event_rx) = prefix_watcher.dissolve();

        // Spawn background task to watch for config changes
        drt.runtime().secondary().spawn(async move {
            tracing::info!("Starting config watcher for disagg router key: {}", key);

            loop {
                let kv_event = tokio::select! {
                    _ = watch_tx.closed() => {
                        tracing::debug!("All watchers have closed; shutting down config watcher for key: {}", key);
                        break;
                    }
                    kv_event = kv_event_rx.recv() => {
                        match kv_event {
                            Some(kv_event) => kv_event,
                            None => {
                                tracing::debug!("Watch stream has closed; shutting down config watcher for key: {}", key);
                                break;
                            }
                        }
                    }
                };

                tracing::debug!("Received watch event for key {}", key);

                match kv_event {
                    WatchEvent::Put(kv) => {
                        let val = serde_json::from_slice::<DisaggRouterConf>(kv.value());
                        if let Ok(config) = val {
                            tracing::info!("Config updated for key {}: {:?}", key, config);
                            // Broadcast the update
                            if watch_tx.send(config).is_err() {
                                tracing::debug!("Unable to send watch updates; shutting down config watcher for key: {}", key);
                                break;
                            }
                        } else {
                            tracing::error!("Unable to parse router config for key {}", key);
                            break;
                        }
                    }
                    WatchEvent::Delete(_) => {
                        tracing::warn!("Config key was deleted: {}", key);
                        // Reset to default values
                        if watch_tx.send(DisaggRouterConf::default()).is_err() {
                            tracing::debug!("Unable to send watch updates; shutting down config watcher for key: {}", key);
                            break;
                        }
                    }
                }
            }

            tracing::debug!("Completed config watcher for key: {}", key);
        });

        Ok((initial_config, watch_rx))
    }
}

#[derive(Clone)]
pub struct DisaggregatedRouter {
    max_local_prefill_length: Arc<Mutex<i32>>,
    model_name: String,
    config_watcher: Option<watch::Receiver<DisaggRouterConf>>,
}

impl DisaggregatedRouter {
    pub fn new(max_local_prefill_length: i32, model_name: String) -> Self {
        DisaggregatedRouter {
            max_local_prefill_length: Arc::new(Mutex::new(max_local_prefill_length)),
            model_name,
            config_watcher: None,
        }
    }

    pub async fn new_with_etcd_and_default(
        drt: Arc<DistributedRuntime>,
        model_name: String,
        default_max_local_prefill_length: i32,
    ) -> anyhow::Result<Self> {
        let (mut config, watcher) =
            DisaggRouterConf::from_etcd_with_watcher(drt, &model_name).await?;

        // Use the provided default if no etcd value was found (when config is the default value)
        if config.max_local_prefill_length == DisaggRouterConf::default().max_local_prefill_length {
            config.max_local_prefill_length = default_max_local_prefill_length;
        }

        let router = Self {
            max_local_prefill_length: Arc::new(Mutex::new(config.max_local_prefill_length)),
            model_name: model_name.clone(),
            config_watcher: Some(watcher),
        };

        // Start background task to watch for config updates
        router.start_config_watcher();

        Ok(router)
    }

    fn start_config_watcher(&self) {
        if let Some(watcher) = self.config_watcher.clone() {
            let mut watcher = watcher;
            // Create a clone for the task
            let model_name = self.model_name.clone();
            let max_local_prefill_length = self.max_local_prefill_length.clone();

            tokio::spawn(async move {
                tracing::info!("Starting config update watcher for model: {}", model_name);

                while watcher.changed().await.is_ok() {
                    let config = watcher.borrow().clone();
                    let new_value = config.max_local_prefill_length;

                    // Update the value using the mutex
                    let mut current_value = max_local_prefill_length.lock().unwrap();
                    let old_value = *current_value;
                    if old_value != new_value {
                        *current_value = new_value;
                        tracing::info!(
                            "Applied config update for model {}: max_local_prefill_length changed from {} to {}",
                            model_name,
                            old_value,
                            new_value
                        );
                    }
                }

                tracing::debug!("Config watcher closed for model: {}", model_name);
            });
        }
    }

    pub fn check_for_updates(&self) {
        if let Some(watcher) = &self.config_watcher {
            if watcher.has_changed().unwrap_or(false) {
                let config = watcher.borrow().clone();
                let new_value = config.max_local_prefill_length;

                // Update the value using the mutex
                let mut current_value = self.max_local_prefill_length.lock().unwrap();
                let old_value = *current_value;
                if old_value != new_value {
                    *current_value = new_value;
                    tracing::info!(
                        "Applied config update for model {}: max_local_prefill_length changed from {} to {}",
                        self.model_name,
                        old_value,
                        new_value
                    );
                }
            }
        }
    }

    pub fn prefill_remote(&self, prefill_length: i32, prefix_hit_length: i32) -> bool {
        // Check for updates before making the decision
        self.check_for_updates();

        // Get the current value from the mutex
        let max_local_prefill_length = *self.max_local_prefill_length.lock().unwrap();

        // schedule the request purely based on the prefill length
        // TODO: apply math models and compare local vs remote prefill TTFT
        prefill_length - prefix_hit_length > max_local_prefill_length
    }

    pub fn update_value(&self, max_local_prefill_length: i32) {
        let mut current = self.max_local_prefill_length.lock().unwrap();
        *current = max_local_prefill_length;
    }

    pub fn get_model_name(&self) -> &str {
        &self.model_name
    }
}
