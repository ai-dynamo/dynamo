// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::collections::HashSet;
use std::time::Duration;

use dynamo_runtime::component::Endpoint;
use tokio::sync::watch;

/// An engine owns only its prefix; topology and unrelated taints are preserved.
#[derive(Clone)]
pub struct ModelTaints {
    pub prefix: &'static str,
    pub values: watch::Receiver<HashSet<String>>,
}

impl ModelTaints {
    pub(crate) async fn publish(
        &mut self,
        endpoint: &Endpoint,
        initial: &HashSet<String>,
    ) -> anyhow::Result<()> {
        self.values
            .has_changed()
            .map_err(|_| anyhow::anyhow!("engine taint observer stopped"))?;
        let owned = self.values.borrow_and_update().clone();
        anyhow::ensure!(
            !self.prefix.is_empty(),
            "engine taint prefix must not be empty"
        );
        anyhow::ensure!(
            owned.iter().all(|taint| taint.starts_with(self.prefix)),
            "engine taint outside its owned prefix"
        );
        tokio::time::timeout(Duration::from_secs(5), async {
            let mut taints: HashSet<String> = initial
                .iter()
                .filter(|taint| {
                    !taint.starts_with(self.prefix) && !taint.starts_with("dynamo.topology/")
                })
                .cloned()
                .collect();
            taints.extend(owned);
            dynamo_llm::local_model::update_model_taints(endpoint, taints).await
        })
        .await
        .map_err(|_| anyhow::anyhow!("engine taint publication timed out; outcome unknown"))?
    }

    pub(crate) async fn follow(
        mut self,
        endpoint: &Endpoint,
        initial: HashSet<String>,
    ) -> anyhow::Result<()> {
        loop {
            self.values
                .changed()
                .await
                .map_err(|_| anyhow::anyhow!("engine taint observer stopped"))?;
            self.publish(endpoint, &initial).await?;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use dynamo_runtime::{
        DistributedRuntime, Runtime,
        discovery::{DiscoveryInstance, DiscoveryQuery, DiscoverySpec},
        distributed::DistributedConfig,
    };

    #[tokio::test]
    async fn version_updates_preserve_other_taints_and_reject_foreign_prefixes() {
        let runtime = Runtime::from_current().unwrap();
        let drt = DistributedRuntime::new(runtime.clone(), DistributedConfig::process_local())
            .await
            .unwrap();
        let endpoint = drt
            .namespace("policy_taints")
            .unwrap()
            .component("backend")
            .unwrap()
            .endpoint("generate");
        let id = endpoint.id();
        drt.discovery().register(DiscoverySpec::Model {
            namespace: id.namespace, component: id.component, endpoint: id.name,
            card_json: serde_json::json!({"display_name": "mock", "runtime_config": {
                "taints": ["capacity/fast", "dynamo.policy/version=stale", "dynamo.topology/zone=west"],
                "topology_domains": {"zone": "west"}
            }}), model_suffix: None,
        }).await.unwrap();
        let initial = HashSet::from(["capacity/fast".into(), "dynamo.topology/zone=west".into()]);
        let (tx, rx) = watch::channel(HashSet::new());
        let mut source = ModelTaints {
            prefix: "dynamo.policy/",
            values: rx,
        };
        tx.send_replace(HashSet::from(["foreign/value".into()]));
        assert!(source.publish(&endpoint, &initial).await.is_err());
        tx.send_replace(HashSet::new());
        source.publish(&endpoint, &initial).await.unwrap();
        let follower_endpoint = endpoint.clone();
        let follower = tokio_util::task::AbortOnDropHandle::new(tokio::spawn(async move {
            source.follow(&follower_endpoint, initial).await
        }));
        for version in [Some("1"), Some("2"), None] {
            let taints: HashSet<String> = version
                .map(|v| format!("dynamo.policy/version=run:{v}"))
                .into_iter()
                .collect();
            tx.send_replace(taints.clone());
            let mut expected = taints;
            expected.extend(["capacity/fast".into(), "dynamo.topology/zone=west".into()]);
            tokio::time::timeout(Duration::from_secs(2), async {
                loop {
                    let id = endpoint.id();
                    let cards = drt
                        .discovery()
                        .list(DiscoveryQuery::EndpointModels {
                            namespace: id.namespace,
                            component: id.component,
                            endpoint: id.name,
                        })
                        .await
                        .unwrap();
                    let [DiscoveryInstance::Model { card_json, .. }] = cards.as_slice() else {
                        panic!("one model required")
                    };
                    let actual: HashSet<String> = card_json["runtime_config"]["taints"]
                        .as_array()
                        .unwrap()
                        .iter()
                        .map(|v| v.as_str().unwrap().to_owned())
                        .collect();
                    if actual == expected {
                        break;
                    }
                    tokio::task::yield_now().await;
                }
            })
            .await
            .expect("taint follower did not publish the new version");
        }
        drop(tx);
        assert!(
            tokio::time::timeout(Duration::from_secs(2), follower)
                .await
                .unwrap()
                .unwrap()
                .unwrap_err()
                .to_string()
                .contains("observer stopped")
        );
        runtime.shutdown();
    }
}
