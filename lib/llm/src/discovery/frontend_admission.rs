// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Publishes which workers this frontend serves, so a rolling update can wait
//! for every frontend to serve a replacement worker generation before it
//! retires the previous one.
//!
//! The frontend registers discovery event sources on [`FRONTEND_ADMISSION_TOPIC`]:
//! a capability record announcing the protocol, and one record per worker
//! namespace listing the model-card keys it serves there (see
//! [`ServingAdmissions`]). Under Kubernetes discovery these records land in the
//! frontend pod's `DynamoWorkerMetadata`, where the Dynamo operator reads them.
//! The operator matches the topic and record fields defined here.

use std::collections::BTreeMap;
use std::sync::Arc;
use std::time::Duration;

use dynamo_runtime::discovery::{Discovery, DiscoveryInstance, DiscoverySpec, EventScope};
use tokio::sync::watch;
use tokio_util::sync::CancellationToken;

use super::controller::ServingAdmissions;

pub(crate) const FRONTEND_ADMISSION_TOPIC: &str = "frontend-model-admission";
/// Record format version. The operator ignores versions it does not know.
const FRONTEND_ADMISSION_PROTOCOL: &str = "v1";
const RETRY_INTERVAL: Duration = Duration::from_secs(1);
const MIN_PUBLISH_INTERVAL: Duration = Duration::from_millis(200);
/// Kubernetes discovery stores publisher IDs as JSON numbers.
const JSON_SAFE_ID_MASK: u64 = (1 << 53) - 1;

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
enum RecordKey {
    Capability,
    Namespace(String),
}

#[derive(Clone, Debug, PartialEq)]
struct Record {
    scope: EventScope,
    metadata: serde_json::Value,
}

fn desired_records(
    frontend_namespace: &str,
    admissions: &ServingAdmissions,
) -> BTreeMap<RecordKey, Record> {
    let mut records = BTreeMap::new();
    records.insert(
        RecordKey::Capability,
        Record {
            scope: EventScope::Namespace {
                name: frontend_namespace.to_string(),
            },
            metadata: serde_json::json!({
                "protocol": FRONTEND_ADMISSION_PROTOCOL,
                "capability": true,
            }),
        },
    );
    for (namespace, members) in admissions {
        records.insert(
            RecordKey::Namespace(namespace.clone()),
            Record {
                scope: EventScope::Namespace {
                    name: namespace.clone(),
                },
                metadata: serde_json::json!({
                    "protocol": FRONTEND_ADMISSION_PROTOCOL,
                    "members": members,
                }),
            },
        );
    }
    records
}

struct AdmissionPublisher {
    discovery: Arc<dyn Discovery>,
    frontend_namespace: String,
    published: BTreeMap<RecordKey, (Record, DiscoveryInstance)>,
}

impl AdmissionPublisher {
    fn new(discovery: Arc<dyn Discovery>, frontend_namespace: String) -> Self {
        Self {
            discovery,
            frontend_namespace,
            published: BTreeMap::new(),
        }
    }

    /// Converges the published records on `admissions` and reports whether
    /// every record now matches.
    ///
    /// A changed record is withdrawn before its replacement is registered: a
    /// reader may briefly see fewer admissions, which only delays a rollout,
    /// but never members this frontend no longer serves.
    async fn reconcile(&mut self, admissions: &ServingAdmissions) -> bool {
        let desired = desired_records(&self.frontend_namespace, admissions);
        let mut converged = true;

        let stale = self
            .published
            .iter()
            .filter(|(key, (record, _))| desired.get(*key) != Some(record))
            .map(|(key, _)| key.clone())
            .collect::<Vec<_>>();
        for key in stale {
            let Some((record, instance)) = self.published.remove(&key) else {
                continue;
            };
            if let Err(error) = self.discovery.unregister(instance.clone()).await {
                tracing::warn!(
                    error = format!("{error:#}"),
                    record = ?key,
                    "Failed to withdraw a frontend admission record; retrying"
                );
                self.published.insert(key, (record, instance));
                converged = false;
            }
        }

        for (key, record) in desired {
            if self.published.contains_key(&key) {
                continue;
            }
            let spec = DiscoverySpec::EventSource {
                scope: record.scope.clone(),
                topic: FRONTEND_ADMISSION_TOPIC.to_string(),
                publisher_id: rand::random::<u64>() & JSON_SAFE_ID_MASK,
                metadata: record.metadata.clone(),
            };
            match self.discovery.register(spec).await {
                Ok(instance) => {
                    tracing::debug!(record = ?key, "Published frontend admission record");
                    self.published.insert(key, (record, instance));
                }
                Err(error) => {
                    tracing::warn!(
                        error = format!("{error:#}"),
                        record = ?key,
                        "Failed to publish a frontend admission record; retrying"
                    );
                    converged = false;
                }
            }
        }
        converged
    }

    async fn withdraw_all(&mut self) {
        for (key, (_, instance)) in std::mem::take(&mut self.published) {
            if let Err(error) = self.discovery.unregister(instance).await {
                tracing::debug!(
                    error = format!("{error:#}"),
                    record = ?key,
                    "Failed to withdraw a frontend admission record during shutdown"
                );
            }
        }
    }
}

/// Publishes `admissions` until `cancellation` fires, then withdraws them.
///
/// Each pass runs to completion so the publisher always knows which records it
/// registered; cancellation is observed between passes. Passes start at most
/// every [`MIN_PUBLISH_INTERVAL`] and publish the latest admissions, so a burst
/// of worker churn costs a bounded number of discovery writes.
pub(crate) async fn run_admission_publisher(
    discovery: Arc<dyn Discovery>,
    frontend_namespace: String,
    mut admissions: watch::Receiver<ServingAdmissions>,
    cancellation: CancellationToken,
) {
    let mut publisher = AdmissionPublisher::new(discovery, frontend_namespace);
    loop {
        let pass_started = tokio::time::Instant::now();
        let current = admissions.borrow_and_update().clone();
        let converged = publisher.reconcile(&current).await;
        tokio::select! {
            _ = cancellation.cancelled() => break,
            changed = admissions.changed() => {
                if changed.is_err() {
                    break;
                }
            }
            _ = tokio::time::sleep(RETRY_INTERVAL), if !converged => {}
        }
        tokio::select! {
            _ = cancellation.cancelled() => break,
            _ = tokio::time::sleep_until(pass_started + MIN_PUBLISH_INTERVAL) => {}
        }
    }
    publisher.withdraw_all().await;
}

#[cfg(test)]
mod tests {
    use std::collections::{BTreeSet, HashMap};

    use dynamo_runtime::discovery::{DiscoveryQuery, EventSourceQuery, KVStoreDiscovery};
    use dynamo_runtime::storage::kv;

    use super::*;

    async fn published_records(discovery: &dyn Discovery) -> HashMap<String, serde_json::Value> {
        discovery
            .list(DiscoveryQuery::EventSources(EventSourceQuery::all()))
            .await
            .unwrap()
            .into_iter()
            .filter_map(|instance| match instance {
                DiscoveryInstance::EventSource {
                    scope,
                    topic,
                    metadata,
                    ..
                } if topic == FRONTEND_ADMISSION_TOPIC => {
                    Some((scope.namespace().to_string(), metadata))
                }
                _ => None,
            })
            .collect()
    }

    async fn wait_for_records(
        discovery: &dyn Discovery,
        expected: HashMap<String, serde_json::Value>,
    ) {
        tokio::time::timeout(Duration::from_secs(10), async {
            while published_records(discovery).await != expected {
                tokio::time::sleep(Duration::from_millis(10)).await;
            }
        })
        .await
        .unwrap_or_else(|_| panic!("expected admission records {expected:?}"));
    }

    #[tokio::test]
    async fn publisher_tracks_admissions_and_withdraws_on_shutdown() {
        let discovery: Arc<dyn Discovery> = Arc::new(KVStoreDiscovery::new(
            kv::Manager::memory(),
            CancellationToken::new(),
        ));
        let (admissions_tx, admissions_rx) = watch::channel(ServingAdmissions::new());
        let cancellation = CancellationToken::new();
        let publisher = tokio::spawn(run_admission_publisher(
            Arc::clone(&discovery),
            "frontend".to_string(),
            admissions_rx,
            cancellation.clone(),
        ));
        let capability = (
            "frontend".to_string(),
            serde_json::json!({"protocol": "v1", "capability": true}),
        );

        // A frontend that serves nothing still announces the protocol.
        wait_for_records(discovery.as_ref(), HashMap::from([capability.clone()])).await;

        // Members follow the admissions.
        let member = "graph/backend/generate/1".to_string();
        admissions_tx.send_replace(ServingAdmissions::from([(
            "graph".to_string(),
            BTreeSet::from([member.clone()]),
        )]));
        wait_for_records(
            discovery.as_ref(),
            HashMap::from([
                capability.clone(),
                (
                    "graph".to_string(),
                    serde_json::json!({"protocol": "v1", "members": [member]}),
                ),
            ]),
        )
        .await;

        // A namespace the frontend no longer serves disappears.
        admissions_tx.send_replace(ServingAdmissions::new());
        wait_for_records(discovery.as_ref(), HashMap::from([capability])).await;

        cancellation.cancel();
        publisher.await.unwrap();
        assert!(published_records(discovery.as_ref()).await.is_empty());
    }
}
