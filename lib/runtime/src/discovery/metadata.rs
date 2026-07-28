// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use anyhow::Result;
use serde::Deserialize as _;
use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use super::{
    DiscoveryInstance, DiscoveryInstanceId, DiscoveryQuery, validate_event_source_reregistration,
};

/// Deserializes a JSON `null` or missing field as `T::default()`.
///
/// Kubernetes Server-Side Apply with `schema = "disabled"` can write an empty
/// object `{}` as `null` for nested free-form fields. Without this helper, the
/// daemon fails to deserialize the `DynamoWorkerMetadata` CR, and the worker is
/// excluded from the `MetadataSnapshot` (i.e. invisible to service discovery),
/// causing `KubeDiscoveryClient::list` to return 0 instances and all inference
/// requests to 404. One concrete example is vLLM elastic EP scaling:
/// `scale_elastic_ep` reinitializes event plane sockets, which triggers
/// `unregister_event_channel()`, leaving `event_channels` as an empty map `{}`.
/// SSA then writes it back as `null`, breaking deserialization until this helper
/// treats `null` as an empty map. The issue applies to any event plane
/// implementation, not only a specific transport.
fn deserialize_null_default<'de, D, T>(deserializer: D) -> Result<T, D::Error>
where
    D: serde::Deserializer<'de>,
    T: Default + serde::Deserialize<'de>,
{
    Ok(Option::<T>::deserialize(deserializer)?.unwrap_or_default())
}

/// Metadata stored on each pod and exposed via HTTP endpoint
/// This struct holds all discovery registrations for this pod instance
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct DiscoveryMetadata {
    /// Registered endpoint instances (key: path string from EndpointInstanceId::to_path())
    #[serde(default, deserialize_with = "deserialize_null_default")]
    endpoints: HashMap<String, DiscoveryInstance>,
    /// Registered model card instances (key: path string from ModelCardInstanceId::to_path())
    #[serde(default, deserialize_with = "deserialize_null_default")]
    model_cards: HashMap<String, DiscoveryInstance>,
    /// Registered event channel instances (key: path string from EventChannelInstanceId::to_path())
    #[serde(default, deserialize_with = "deserialize_null_default")]
    event_channels: HashMap<String, DiscoveryInstance>,
    /// Registered event source instances (key: path string from EventSourceInstanceId::to_path())
    #[serde(default, deserialize_with = "deserialize_null_default")]
    event_sources: HashMap<String, DiscoveryInstance>,
}

impl DiscoveryMetadata {
    /// Create a new empty metadata store
    pub fn new() -> Self {
        Self {
            endpoints: HashMap::new(),
            model_cards: HashMap::new(),
            event_channels: HashMap::new(),
            event_sources: HashMap::new(),
        }
    }

    /// Register an endpoint instance
    pub fn register_endpoint(&mut self, instance: DiscoveryInstance) -> Result<()> {
        match instance.id() {
            DiscoveryInstanceId::Endpoint(key) => {
                self.endpoints.insert(key.to_path(), instance);
                Ok(())
            }
            DiscoveryInstanceId::Model(_) => {
                anyhow::bail!("Cannot register non-endpoint instance as endpoint")
            }
            DiscoveryInstanceId::EventChannel(_) => {
                anyhow::bail!("Cannot register EventChannel instance as endpoint")
            }
            DiscoveryInstanceId::EventSource(_) => {
                anyhow::bail!("Cannot register EventSource instance as endpoint")
            }
        }
    }

    /// Register a model card instance
    pub fn register_model_card(&mut self, instance: DiscoveryInstance) -> Result<()> {
        match instance.id() {
            DiscoveryInstanceId::Model(key) => {
                self.model_cards.insert(key.to_path(), instance);
                Ok(())
            }
            DiscoveryInstanceId::Endpoint(_) => {
                anyhow::bail!("Cannot register non-model-card instance as model card")
            }
            DiscoveryInstanceId::EventChannel(_) => {
                anyhow::bail!("Cannot register EventChannel instance as model card")
            }
            DiscoveryInstanceId::EventSource(_) => {
                anyhow::bail!("Cannot register EventSource instance as model card")
            }
        }
    }

    /// Unregister an endpoint instance
    pub fn unregister_endpoint(&mut self, instance: &DiscoveryInstance) -> Result<()> {
        match instance.id() {
            DiscoveryInstanceId::Endpoint(key) => {
                self.endpoints.remove(&key.to_path());
                Ok(())
            }
            DiscoveryInstanceId::Model(_) => {
                anyhow::bail!("Cannot unregister non-endpoint instance as endpoint")
            }
            DiscoveryInstanceId::EventChannel(_) => {
                anyhow::bail!("Cannot unregister EventChannel instance as endpoint")
            }
            DiscoveryInstanceId::EventSource(_) => {
                anyhow::bail!("Cannot unregister EventSource instance as endpoint")
            }
        }
    }

    /// Unregister a model card instance
    pub fn unregister_model_card(&mut self, instance: &DiscoveryInstance) -> Result<()> {
        match instance.id() {
            DiscoveryInstanceId::Model(key) => {
                self.model_cards.remove(&key.to_path());
                Ok(())
            }
            DiscoveryInstanceId::Endpoint(_) => {
                anyhow::bail!("Cannot unregister non-model-card instance as model card")
            }
            DiscoveryInstanceId::EventChannel(_) => {
                anyhow::bail!("Cannot unregister EventChannel instance as model card")
            }
            DiscoveryInstanceId::EventSource(_) => {
                anyhow::bail!("Cannot unregister EventSource instance as model card")
            }
        }
    }

    /// Register an event channel instance
    pub fn register_event_channel(&mut self, instance: DiscoveryInstance) -> Result<()> {
        match instance.id() {
            DiscoveryInstanceId::EventChannel(key) => {
                self.event_channels.insert(key.to_path(), instance);
                Ok(())
            }
            DiscoveryInstanceId::Endpoint(_) => {
                anyhow::bail!("Cannot register Endpoint instance as event channel")
            }
            DiscoveryInstanceId::Model(_) => {
                anyhow::bail!("Cannot register Model instance as event channel")
            }
            DiscoveryInstanceId::EventSource(_) => {
                anyhow::bail!("Cannot register EventSource instance as event channel")
            }
        }
    }

    /// Unregister an event channel instance
    pub fn unregister_event_channel(&mut self, instance: &DiscoveryInstance) -> Result<()> {
        match instance.id() {
            DiscoveryInstanceId::EventChannel(key) => {
                self.event_channels.remove(&key.to_path());
                Ok(())
            }
            DiscoveryInstanceId::Endpoint(_) => {
                anyhow::bail!("Cannot unregister Endpoint instance as event channel")
            }
            DiscoveryInstanceId::Model(_) => {
                anyhow::bail!("Cannot unregister Model instance as event channel")
            }
            DiscoveryInstanceId::EventSource(_) => {
                anyhow::bail!("Cannot unregister EventSource instance as event channel")
            }
        }
    }

    /// Register a semantic event source instance.
    pub fn register_event_source(&mut self, instance: DiscoveryInstance) -> Result<()> {
        match instance.id() {
            DiscoveryInstanceId::EventSource(key) => {
                let path = key.to_path();
                match self.event_sources.entry(path) {
                    std::collections::hash_map::Entry::Vacant(entry) => {
                        entry.insert(instance);
                        Ok(())
                    }
                    std::collections::hash_map::Entry::Occupied(entry) => {
                        validate_event_source_reregistration(entry.get(), &instance)
                    }
                }
            }
            DiscoveryInstanceId::Endpoint(_) => {
                anyhow::bail!("Cannot register Endpoint instance as event source")
            }
            DiscoveryInstanceId::Model(_) => {
                anyhow::bail!("Cannot register Model instance as event source")
            }
            DiscoveryInstanceId::EventChannel(_) => {
                anyhow::bail!("Cannot register EventChannel instance as event source")
            }
        }
    }

    /// Unregister one exact semantic event source incarnation.
    pub fn unregister_event_source(&mut self, instance: &DiscoveryInstance) -> Result<()> {
        match instance.id() {
            DiscoveryInstanceId::EventSource(key) => {
                self.event_sources.remove(&key.to_path());
                Ok(())
            }
            DiscoveryInstanceId::Endpoint(_) => {
                anyhow::bail!("Cannot unregister Endpoint instance as event source")
            }
            DiscoveryInstanceId::Model(_) => {
                anyhow::bail!("Cannot unregister Model instance as event source")
            }
            DiscoveryInstanceId::EventChannel(_) => {
                anyhow::bail!("Cannot unregister EventChannel instance as event source")
            }
        }
    }

    /// Get all registered endpoints
    pub fn get_all_endpoints(&self) -> Vec<DiscoveryInstance> {
        self.endpoints.values().cloned().collect()
    }

    /// Get all registered model cards
    pub fn get_all_model_cards(&self) -> Vec<DiscoveryInstance> {
        self.model_cards.values().cloned().collect()
    }

    /// Get all registered event channels
    pub fn get_all_event_channels(&self) -> Vec<DiscoveryInstance> {
        self.event_channels.values().cloned().collect()
    }

    /// Get all registered semantic event sources.
    pub fn get_all_event_sources(&self) -> Vec<DiscoveryInstance> {
        self.event_sources.values().cloned().collect()
    }

    /// Get all registered instances (endpoints, model cards, and event channels)
    pub fn get_all(&self) -> Vec<DiscoveryInstance> {
        self.endpoints
            .values()
            .chain(self.model_cards.values())
            .chain(self.event_channels.values())
            .chain(self.event_sources.values())
            .cloned()
            .collect()
    }

    /// Filter this metadata by query
    pub fn filter(&self, query: &DiscoveryQuery) -> Vec<DiscoveryInstance> {
        let all_instances = match query {
            DiscoveryQuery::AllEndpoints
            | DiscoveryQuery::NamespacedEndpoints { .. }
            | DiscoveryQuery::ComponentEndpoints { .. }
            | DiscoveryQuery::Endpoint { .. } => self.get_all_endpoints(),

            DiscoveryQuery::AllModels
            | DiscoveryQuery::NamespacedModels { .. }
            | DiscoveryQuery::ComponentModels { .. }
            | DiscoveryQuery::EndpointModels { .. } => self.get_all_model_cards(),

            // EventChannel queries now return actual event channels
            DiscoveryQuery::EventChannels(_) => self.get_all_event_channels(),
            DiscoveryQuery::EventSources(_) => self.get_all_event_sources(),
        };

        filter_instances(all_instances, query)
    }
}

impl Default for DiscoveryMetadata {
    fn default() -> Self {
        Self::new()
    }
}

/// Filter instances by query predicate
fn filter_instances(
    instances: Vec<DiscoveryInstance>,
    query: &DiscoveryQuery,
) -> Vec<DiscoveryInstance> {
    match query {
        DiscoveryQuery::AllEndpoints | DiscoveryQuery::AllModels => instances,

        DiscoveryQuery::NamespacedEndpoints { namespace } => instances
            .into_iter()
            .filter(|inst| match inst {
                DiscoveryInstance::Endpoint(i) => &i.namespace == namespace,
                _ => false,
            })
            .collect(),

        DiscoveryQuery::ComponentEndpoints {
            namespace,
            component,
        } => instances
            .into_iter()
            .filter(|inst| match inst {
                DiscoveryInstance::Endpoint(i) => {
                    &i.namespace == namespace && &i.component == component
                }
                _ => false,
            })
            .collect(),

        DiscoveryQuery::Endpoint {
            namespace,
            component,
            endpoint,
        } => instances
            .into_iter()
            .filter(|inst| match inst {
                DiscoveryInstance::Endpoint(i) => {
                    &i.namespace == namespace
                        && &i.component == component
                        && &i.endpoint == endpoint
                }
                _ => false,
            })
            .collect(),

        DiscoveryQuery::NamespacedModels { namespace } => instances
            .into_iter()
            .filter(|inst| match inst {
                DiscoveryInstance::Model { namespace: ns, .. } => ns == namespace,
                _ => false,
            })
            .collect(),

        DiscoveryQuery::ComponentModels {
            namespace,
            component,
        } => instances
            .into_iter()
            .filter(|inst| match inst {
                DiscoveryInstance::Model {
                    namespace: ns,
                    component: comp,
                    ..
                } => ns == namespace && comp == component,
                _ => false,
            })
            .collect(),

        DiscoveryQuery::EndpointModels {
            namespace,
            component,
            endpoint,
        } => instances
            .into_iter()
            .filter(|inst| match inst {
                DiscoveryInstance::Model {
                    namespace: ns,
                    component: comp,
                    endpoint: ep,
                    ..
                } => ns == namespace && comp == component && ep == endpoint,
                _ => false,
            })
            .collect(),

        // EventChannel queries - unified filtering with optional scope filters
        DiscoveryQuery::EventChannels(query) => instances
            .into_iter()
            .filter(|inst| match inst {
                DiscoveryInstance::EventChannel {
                    scope, topic: t, ..
                } => {
                    query
                        .scope
                        .as_ref()
                        .is_none_or(|expected| expected == scope)
                        && query.topic.as_ref().is_none_or(|qt| qt == t)
                }
                _ => false,
            })
            .collect(),

        DiscoveryQuery::EventSources(query) => instances
            .into_iter()
            .filter(|inst| match inst {
                DiscoveryInstance::EventSource {
                    scope, topic: t, ..
                } => {
                    query
                        .scope
                        .as_ref()
                        .is_none_or(|expected| expected == scope)
                        && query.topic.as_ref().is_none_or(|qt| qt == t)
                }
                _ => false,
            })
            .collect(),
    }
}

/// Snapshot of all discovered instances and their metadata
#[derive(Clone, Debug)]
pub struct MetadataSnapshot {
    /// Map of instance_id -> metadata
    pub instances: HashMap<u64, Arc<DiscoveryMetadata>>,
    /// Map of instance_id -> CR generation for change detection
    /// Keys match `instances` keys exactly - only ready pods with CRs are included
    pub generations: HashMap<u64, i64>,
    /// Sequence number for debugging
    pub sequence: u64,
    /// Timestamp for observability
    pub timestamp: std::time::Instant,
}

impl MetadataSnapshot {
    pub fn empty() -> Self {
        Self {
            instances: HashMap::new(),
            generations: HashMap::new(),
            sequence: 0,
            timestamp: std::time::Instant::now(),
        }
    }

    /// Compare with previous snapshot and return true if changed.
    /// Logs diagnostic info about what changed.
    /// This is done on the basis of the generation of the DynamoWorkerMetadata CRs that are owned by ready workers
    pub fn has_changes_from(&self, prev: &MetadataSnapshot) -> bool {
        if self.generations == prev.generations {
            tracing::trace!(
                "Snapshot (seq={}): no changes, {} instances",
                self.sequence,
                self.instances.len()
            );
            return false;
        }

        // Compute diff for logging
        let curr_ids: HashSet<u64> = self.generations.keys().copied().collect();
        let prev_ids: HashSet<u64> = prev.generations.keys().copied().collect();

        let added: Vec<_> = curr_ids
            .difference(&prev_ids)
            .map(|id| format!("{:x}", id))
            .collect();
        let removed: Vec<_> = prev_ids
            .difference(&curr_ids)
            .map(|id| format!("{:x}", id))
            .collect();
        let updated: Vec<_> = self
            .generations
            .iter()
            .filter(|(k, v)| prev.generations.get(*k).is_some_and(|pv| pv != *v))
            .map(|(k, _)| format!("{:x}", k))
            .collect();

        tracing::info!(
            "Snapshot (seq={}): {} instances, added={:?}, removed={:?}, updated={:?}",
            self.sequence,
            self.instances.len(),
            added,
            removed,
            updated
        );

        true
    }

    /// Filter all instances in the snapshot by query
    pub fn filter(&self, query: &DiscoveryQuery) -> Vec<DiscoveryInstance> {
        self.instances
            .values()
            .flat_map(|metadata| metadata.filter(query))
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::component::{Instance, TransportType};
    use crate::discovery::{EventChannelQuery, EventSourceQuery};

    #[test]
    fn test_metadata_serde() {
        let mut metadata = DiscoveryMetadata::new();

        // Add an endpoint
        let instance = DiscoveryInstance::Endpoint(Instance {
            namespace: "test".to_string(),
            component: "comp1".to_string(),
            endpoint: "ep1".to_string(),
            instance_id: 123,
            transport: TransportType::Nats("nats://localhost:4222".to_string()),
            device_type: None,
        });

        metadata.register_endpoint(instance).unwrap();

        // Serialize
        let json = serde_json::to_string(&metadata).unwrap();

        // Deserialize
        let deserialized: DiscoveryMetadata = serde_json::from_str(&json).unwrap();

        assert_eq!(deserialized.endpoints.len(), 1);
        assert_eq!(deserialized.model_cards.len(), 0);
    }

    #[tokio::test]
    async fn test_metadata_accessors() {
        let mut metadata = DiscoveryMetadata::new();

        // Register endpoints
        for i in 0..3 {
            let instance = DiscoveryInstance::Endpoint(Instance {
                namespace: "test".to_string(),
                component: "comp1".to_string(),
                endpoint: format!("ep{}", i),
                instance_id: i,
                transport: TransportType::Nats("nats://localhost:4222".to_string()),
                device_type: None,
            });
            metadata.register_endpoint(instance).unwrap();
        }

        // Register model cards
        for i in 0..2 {
            let instance = DiscoveryInstance::Model {
                namespace: "test".to_string(),
                component: "comp1".to_string(),
                endpoint: format!("ep{}", i),
                instance_id: i,
                card_json: serde_json::json!({"model": "test"}),
                model_suffix: None,
            };
            metadata.register_model_card(instance).unwrap();
        }

        assert_eq!(metadata.get_all_endpoints().len(), 3);
        assert_eq!(metadata.get_all_model_cards().len(), 2);
        assert_eq!(metadata.get_all().len(), 5);
    }

    #[test]
    fn event_source_registration_filters_and_removes_exact_incarnation() {
        use crate::discovery::EventScope;
        use crate::protocols::EndpointId;

        let mut metadata = DiscoveryMetadata::new();
        let endpoint = EndpointId {
            namespace: "test".to_string(),
            component: "worker".to_string(),
            name: "decode".to_string(),
        };
        let source = |publisher_id| DiscoveryInstance::EventSource {
            scope: EventScope::Endpoint {
                endpoint: endpoint.clone(),
            },
            topic: "kv-events".to_string(),
            publisher_id,
            metadata: serde_json::json!({"worker_id": 7, "dp_rank": 0}),
        };

        let old = source(100);
        let current = source(205);
        metadata.register_event_source(old.clone()).unwrap();
        metadata.register_event_source(old.clone()).unwrap();
        let mut conflicting = old.clone();
        let DiscoveryInstance::EventSource {
            metadata: descriptor,
            ..
        } = &mut conflicting
        else {
            unreachable!()
        };
        *descriptor = serde_json::json!({"worker_id": 7, "dp_rank": 0, "changed": true});
        assert!(
            metadata
                .register_event_source(conflicting)
                .unwrap_err()
                .to_string()
                .contains("cannot change its descriptor")
        );
        metadata.register_event_source(current.clone()).unwrap();

        let query =
            DiscoveryQuery::EventSources(EventSourceQuery::endpoint_topic(endpoint, "kv-events"));
        assert_eq!(metadata.filter(&query).len(), 2);

        metadata.unregister_event_source(&old).unwrap();
        assert_eq!(metadata.filter(&query), vec![current]);
    }

    #[test]
    fn event_channel_queries_match_exact_scope() {
        use crate::discovery::{EventScope, EventTransport};
        use crate::protocols::EndpointId;

        let mut metadata = DiscoveryMetadata::new();
        let component_scope = EventScope::Component {
            namespace: "test".to_string(),
            component: "worker".to_string(),
        };
        let endpoint_a = EndpointId {
            namespace: "test".to_string(),
            component: "worker".to_string(),
            name: "a".to_string(),
        };
        let endpoint_b = EndpointId {
            name: "b".to_string(),
            ..endpoint_a.clone()
        };

        for (instance_id, scope) in [
            (1, component_scope),
            (
                2,
                EventScope::Endpoint {
                    endpoint: endpoint_a.clone(),
                },
            ),
            (
                3,
                EventScope::Endpoint {
                    endpoint: endpoint_b.clone(),
                },
            ),
        ] {
            metadata
                .register_event_channel(DiscoveryInstance::EventChannel {
                    scope,
                    topic: "kv-events".to_string(),
                    instance_id,
                    transport: EventTransport::zmq(format!("tcp://localhost:{instance_id}")),
                })
                .unwrap();
        }

        assert_eq!(metadata.get_all_event_channels().len(), 3);
        assert_eq!(metadata.get_all().len(), 3);
        assert_eq!(
            metadata
                .filter(&DiscoveryQuery::EventChannels(EventChannelQuery::all()))
                .len(),
            3
        );

        let component = metadata.filter(&DiscoveryQuery::EventChannels(EventChannelQuery::topic(
            "test",
            "worker",
            "kv-events",
        )));
        assert_eq!(component.len(), 1);

        let endpoint_a_instances = metadata.filter(&DiscoveryQuery::EventChannels(
            EventChannelQuery::endpoint_topic(endpoint_a, "kv-events"),
        ));
        assert_eq!(endpoint_a_instances.len(), 1);
        assert_eq!(endpoint_a_instances[0].instance_id(), 2);

        let endpoint_b_instances = metadata.filter(&DiscoveryQuery::EventChannels(
            EventChannelQuery::endpoint_topic(endpoint_b, "kv-events"),
        ));
        assert_eq!(endpoint_b_instances.len(), 1);
        assert_eq!(endpoint_b_instances[0].instance_id(), 3);

        metadata
            .unregister_event_channel(&endpoint_a_instances[0])
            .unwrap();
        assert_eq!(metadata.get_all_event_channels().len(), 2);
        assert!(
            metadata
                .filter(&DiscoveryQuery::EventChannels(
                    EventChannelQuery::component("other", "worker")
                ))
                .is_empty()
        );
    }

    #[tokio::test]
    async fn test_mixed_instances() {
        use crate::discovery::{EventScope, EventTransport};

        let mut metadata = DiscoveryMetadata::new();

        // Register one of each type
        let endpoint = DiscoveryInstance::Endpoint(Instance {
            namespace: "test".to_string(),
            component: "comp1".to_string(),
            endpoint: "ep1".to_string(),
            instance_id: 1,
            transport: TransportType::Nats("nats://localhost:4222".to_string()),
            device_type: None,
        });
        metadata.register_endpoint(endpoint).unwrap();

        let model = DiscoveryInstance::Model {
            namespace: "test".to_string(),
            component: "comp1".to_string(),
            endpoint: "ep1".to_string(),
            instance_id: 2,
            card_json: serde_json::json!({"model": "test"}),
            model_suffix: None,
        };
        metadata.register_model_card(model).unwrap();

        let event_channel = DiscoveryInstance::EventChannel {
            scope: EventScope::Component {
                namespace: "test".to_string(),
                component: "comp1".to_string(),
            },
            topic: "test-topic".to_string(),
            instance_id: 3,
            transport: EventTransport::zmq("tcp://localhost:5000"),
        };
        metadata.register_event_channel(event_channel).unwrap();

        // Verify get_all returns all three
        assert_eq!(metadata.get_all().len(), 3);
        assert_eq!(metadata.get_all_endpoints().len(), 1);
        assert_eq!(metadata.get_all_model_cards().len(), 1);
        assert_eq!(metadata.get_all_event_channels().len(), 1);
    }
}
