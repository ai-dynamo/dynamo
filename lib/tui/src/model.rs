// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Data model for the Dynamo TUI.
//!
//! These types represent the hierarchical view of a Dynamo deployment:
//! Namespace → Component → Endpoint, plus NATS stats and Prometheus metrics.

use std::collections::BTreeMap;

/// Health status of a component or endpoint.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HealthStatus {
    Ready,
    #[allow(dead_code)]
    Provisioning,
    Offline,
}

impl HealthStatus {
    pub fn symbol(&self) -> &'static str {
        match self {
            HealthStatus::Ready => "●",
            HealthStatus::Provisioning => "◐",
            HealthStatus::Offline => "○",
        }
    }
}

/// A single endpoint within a component.
#[derive(Debug, Clone)]
pub struct Endpoint {
    pub name: String,
    pub instance_count: usize,
    pub status: HealthStatus,
}

/// A component within a namespace.
#[derive(Debug, Clone)]
pub struct Component {
    pub name: String,
    pub endpoints: Vec<Endpoint>,
    pub instance_count: usize,
    pub status: HealthStatus,
    pub models: Vec<String>,
}

/// A namespace in the Dynamo deployment.
#[derive(Debug, Clone)]
pub struct Namespace {
    pub name: String,
    pub components: Vec<Component>,
}

/// NATS connection and message statistics.
#[derive(Debug, Clone, Default)]
pub struct NatsStats {
    pub connected: bool,
    pub server_id: String,
    pub msgs_in: u64,
    pub msgs_out: u64,
    pub bytes_in: u64,
    pub bytes_out: u64,
    pub streams: Vec<StreamInfo>,
}

/// Info about a NATS JetStream stream.
#[derive(Debug, Clone)]
pub struct StreamInfo {
    pub name: String,
    pub consumer_count: usize,
    pub message_count: u64,
}

/// Prometheus metrics scraped from the frontend.
#[derive(Debug, Clone, Default)]
pub struct PrometheusMetrics {
    pub available: bool,
    pub ttft_p50_ms: Option<f64>,
    pub ttft_p99_ms: Option<f64>,
    pub tpot_p50_ms: Option<f64>,
    pub tpot_p99_ms: Option<f64>,
    pub requests_inflight: Option<u64>,
    pub requests_queued: Option<u64>,
    pub tokens_per_sec: Option<f64>,
}

/// Build a hierarchical tree of namespaces from flat ETCD key-value entries.
///
/// Each entry is a tuple of `(namespace, component, endpoint, instance_id)`.
/// Model entries are tuples of `(namespace, component, model_name)`.
pub fn build_tree(
    endpoint_entries: &[(String, String, String, u64)],
    model_entries: &[(String, String, String)],
) -> Vec<Namespace> {
    // namespace -> component -> endpoint -> set of instance_ids
    let mut tree: BTreeMap<String, BTreeMap<String, BTreeMap<String, Vec<u64>>>> = BTreeMap::new();
    // namespace -> component -> vec of model names
    let mut models: BTreeMap<String, BTreeMap<String, Vec<String>>> = BTreeMap::new();

    for (ns, comp, ep, instance_id) in endpoint_entries {
        tree.entry(ns.clone())
            .or_default()
            .entry(comp.clone())
            .or_default()
            .entry(ep.clone())
            .or_default()
            .push(*instance_id);
    }

    for (ns, comp, model_name) in model_entries {
        models
            .entry(ns.clone())
            .or_default()
            .entry(comp.clone())
            .or_default()
            .push(model_name.clone());
    }

    tree.into_iter()
        .map(|(ns_name, components)| {
            let comps: Vec<Component> = components
                .into_iter()
                .map(|(comp_name, endpoints)| {
                    let total_instances: usize = endpoints.values().map(|ids| ids.len()).sum();
                    let eps: Vec<Endpoint> = endpoints
                        .into_iter()
                        .map(|(ep_name, ids)| {
                            let count = ids.len();
                            Endpoint {
                                name: ep_name,
                                instance_count: count,
                                status: if count > 0 {
                                    HealthStatus::Ready
                                } else {
                                    HealthStatus::Offline
                                },
                            }
                        })
                        .collect();

                    let comp_models = models
                        .get(&ns_name)
                        .and_then(|m| m.get(&comp_name))
                        .cloned()
                        .unwrap_or_default();

                    let status = if total_instances > 0 {
                        HealthStatus::Ready
                    } else {
                        HealthStatus::Offline
                    };

                    Component {
                        name: comp_name,
                        endpoints: eps,
                        instance_count: total_instances,
                        status,
                        models: comp_models,
                    }
                })
                .collect();

            Namespace {
                name: ns_name,
                components: comps,
            }
        })
        .collect()
}

/// Parse an ETCD key into (namespace, component, endpoint, instance_id).
///
/// Expected format: `v1/instances/{namespace}/{component}/{endpoint}/{instance_id_hex}`
pub fn parse_endpoint_key(key: &str) -> Option<(String, String, String, u64)> {
    let stripped = key.strip_prefix("v1/instances/")?;
    let parts: Vec<&str> = stripped.splitn(4, '/').collect();
    if parts.len() != 4 {
        return None;
    }
    let instance_id = u64::from_str_radix(parts[3], 16).ok()?;
    Some((
        parts[0].to_string(),
        parts[1].to_string(),
        parts[2].to_string(),
        instance_id,
    ))
}

/// Parse an ETCD model card key into (namespace, component, model_name).
///
/// Expected format: `v1/mdc/{namespace}/{component}/{endpoint}/{instance_id_hex}[/{suffix}]`
pub fn parse_model_key(key: &str) -> Option<(String, String, String)> {
    let stripped = key.strip_prefix("v1/mdc/")?;
    let parts: Vec<&str> = stripped.splitn(5, '/').collect();
    if parts.len() < 4 {
        return None;
    }
    // Model name is derived from the component
    Some((
        parts[0].to_string(),
        parts[1].to_string(),
        parts[2].to_string(),
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_endpoint_key() {
        let key = "v1/instances/dynamo/backend/generate/694d98147d54be25";
        let result = parse_endpoint_key(key).unwrap();
        assert_eq!(result.0, "dynamo");
        assert_eq!(result.1, "backend");
        assert_eq!(result.2, "generate");
        assert_eq!(result.3, 0x694d98147d54be25);
    }

    #[test]
    fn test_parse_endpoint_key_invalid() {
        assert!(parse_endpoint_key("v1/instances/only/two").is_none());
        assert!(parse_endpoint_key("wrong/prefix/a/b/c/1").is_none());
        assert!(parse_endpoint_key("v1/instances/a/b/c/not_hex").is_none());
    }

    #[test]
    fn test_parse_model_key() {
        let key = "v1/mdc/dynamo/backend/generate/abc123";
        let result = parse_model_key(key).unwrap();
        assert_eq!(result.0, "dynamo");
        assert_eq!(result.1, "backend");
        assert_eq!(result.2, "generate");
    }

    #[test]
    fn test_build_tree_basic() {
        let entries = vec![
            ("ns-a".into(), "frontend".into(), "http".into(), 1u64),
            ("ns-a".into(), "frontend".into(), "http".into(), 2u64),
            ("ns-a".into(), "router".into(), "grpc".into(), 3u64),
            ("ns-b".into(), "worker".into(), "engine".into(), 4u64),
        ];
        let models = vec![];

        let tree = build_tree(&entries, &models);

        assert_eq!(tree.len(), 2); // ns-a, ns-b
        assert_eq!(tree[0].name, "ns-a");
        assert_eq!(tree[0].components.len(), 2); // frontend, router
        assert_eq!(tree[1].name, "ns-b");
        assert_eq!(tree[1].components.len(), 1); // worker

        // frontend has 1 endpoint (http) with 2 instances
        let frontend = &tree[0].components[0];
        assert_eq!(frontend.name, "frontend");
        assert_eq!(frontend.endpoints.len(), 1);
        assert_eq!(frontend.endpoints[0].name, "http");
        assert_eq!(frontend.endpoints[0].instance_count, 2);
        assert_eq!(frontend.instance_count, 2);
        assert_eq!(frontend.status, HealthStatus::Ready);

        // router has 1 endpoint (grpc) with 1 instance
        let router = &tree[0].components[1];
        assert_eq!(router.name, "router");
        assert_eq!(router.endpoints.len(), 1);
        assert_eq!(router.endpoints[0].instance_count, 1);
    }

    #[test]
    fn test_build_tree_empty() {
        let tree = build_tree(&[], &[]);
        assert!(tree.is_empty());
    }

    #[test]
    fn test_build_tree_with_models() {
        let entries = vec![("ns".into(), "backend".into(), "generate".into(), 1u64)];
        let models = vec![
            ("ns".into(), "backend".into(), "llama-7b".into()),
            ("ns".into(), "backend".into(), "mistral-7b".into()),
        ];

        let tree = build_tree(&entries, &models);
        assert_eq!(tree[0].components[0].models.len(), 2);
        assert_eq!(tree[0].components[0].models[0], "llama-7b");
    }

    #[test]
    fn test_health_status_display() {
        assert_eq!(HealthStatus::Ready.symbol(), "●");
        assert_eq!(HealthStatus::Provisioning.symbol(), "◐");
        assert_eq!(HealthStatus::Offline.symbol(), "○");
    }

    #[test]
    fn test_parse_model_key_with_suffix() {
        let key = "v1/mdc/dynamo/backend/generate/abc123/lora-adapter";
        let result = parse_model_key(key).unwrap();
        assert_eq!(result.0, "dynamo");
        assert_eq!(result.1, "backend");
        assert_eq!(result.2, "generate");
    }

    #[test]
    fn test_parse_model_key_invalid() {
        // Too few parts
        assert!(parse_model_key("v1/mdc/only/two").is_none());
        // Wrong prefix
        assert!(parse_model_key("v1/instances/a/b/c/d").is_none());
        // No prefix at all
        assert!(parse_model_key("random/key").is_none());
    }

    #[test]
    fn test_parse_endpoint_key_empty_string() {
        assert!(parse_endpoint_key("").is_none());
    }

    #[test]
    fn test_parse_model_key_empty_string() {
        assert!(parse_model_key("").is_none());
    }

    #[test]
    fn test_parse_endpoint_key_max_hex() {
        let key = "v1/instances/ns/comp/ep/ffffffffffffffff";
        let result = parse_endpoint_key(key).unwrap();
        assert_eq!(result.3, u64::MAX);
    }

    #[test]
    fn test_parse_endpoint_key_zero_hex() {
        let key = "v1/instances/ns/comp/ep/0000000000000000";
        let result = parse_endpoint_key(key).unwrap();
        assert_eq!(result.3, 0);
    }

    #[test]
    fn test_build_tree_single_namespace() {
        let entries = vec![("ns".into(), "backend".into(), "generate".into(), 1u64)];
        let tree = build_tree(&entries, &[]);
        assert_eq!(tree.len(), 1);
        assert_eq!(tree[0].name, "ns");
        assert_eq!(tree[0].components.len(), 1);
        assert_eq!(tree[0].components[0].endpoints.len(), 1);
        assert_eq!(tree[0].components[0].endpoints[0].instance_count, 1);
    }

    #[test]
    fn test_build_tree_duplicate_instance_ids() {
        // Same instance ID added twice should result in 2 entries
        // (build_tree doesn't deduplicate — that's the caller's job)
        let entries = vec![
            ("ns".into(), "comp".into(), "ep".into(), 1u64),
            ("ns".into(), "comp".into(), "ep".into(), 1u64),
        ];
        let tree = build_tree(&entries, &[]);
        assert_eq!(tree[0].components[0].endpoints[0].instance_count, 2);
    }

    #[test]
    fn test_build_tree_multiple_endpoints_same_component() {
        let entries = vec![
            ("ns".into(), "backend".into(), "generate".into(), 1u64),
            ("ns".into(), "backend".into(), "health".into(), 2u64),
            ("ns".into(), "backend".into(), "clear_kv".into(), 3u64),
        ];
        let tree = build_tree(&entries, &[]);
        assert_eq!(tree[0].components[0].endpoints.len(), 3);
        assert_eq!(tree[0].components[0].instance_count, 3);
    }

    #[test]
    fn test_build_tree_models_without_matching_endpoints() {
        // Models for a component that has no endpoints
        let entries = vec![];
        let models = vec![("ns".into(), "backend".into(), "llama-7b".into())];
        // Models are only attached to existing components in the tree
        // Since there are no endpoint entries, the tree is empty
        let tree = build_tree(&entries, &models);
        assert!(tree.is_empty());
    }

    #[test]
    fn test_build_tree_sorted_by_name() {
        // BTreeMap ensures alphabetical ordering
        let entries = vec![
            ("zebra".into(), "comp".into(), "ep".into(), 1u64),
            ("alpha".into(), "comp".into(), "ep".into(), 2u64),
            ("middle".into(), "comp".into(), "ep".into(), 3u64),
        ];
        let tree = build_tree(&entries, &[]);
        assert_eq!(tree[0].name, "alpha");
        assert_eq!(tree[1].name, "middle");
        assert_eq!(tree[2].name, "zebra");
    }

    #[test]
    fn test_build_tree_component_status_from_instances() {
        let entries = vec![("ns".into(), "backend".into(), "ep".into(), 1u64)];
        let tree = build_tree(&entries, &[]);
        assert_eq!(tree[0].components[0].status, HealthStatus::Ready);
    }

    #[test]
    fn test_endpoint_status_ready_when_has_instances() {
        let entries = vec![("ns".into(), "comp".into(), "ep".into(), 1u64)];
        let tree = build_tree(&entries, &[]);
        assert_eq!(
            tree[0].components[0].endpoints[0].status,
            HealthStatus::Ready
        );
        assert_eq!(tree[0].components[0].endpoints[0].instance_count, 1);
    }
}
