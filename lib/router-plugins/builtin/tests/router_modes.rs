// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

mod support;

use std::collections::HashSet;

use dynamo_kv_router::plugins::RouterPluginRegistry;
use dynamo_kv_router::plugins::worker_selection::WorkerSelectionPolicy;
use dynamo_kv_router::{KvRouterConfig, RoutingPartitionRef, WorkerSelector, WorkerType};
use dynamo_router_policies::router_modes::{LEAST_LOADED, RANDOM, ROUND_ROBIN};
use dynamo_router_policies::{default_registry, register, with_defaults};
use support::*;

const MODES: [&str; 3] = [RANDOM, ROUND_ROBIN, LEAST_LOADED];

fn policy(policy_type: &str) -> WorkerSelectionPolicy {
    let factory = default_registry().resolve_type(policy_type).unwrap();
    factory(
        &KvRouterConfig::default(),
        WorkerType::Decode,
        RoutingPartitionRef::new("model", "group"),
    )
}

#[test]
fn routing_modes_resolve_by_name_and_run_without_kv_state() {
    for policy_type in MODES {
        assert!(
            policy(policy_type).into_hosted_picker().is_ok(),
            "{policy_type} should need no KV state"
        );
    }
}

/// A host installs `with_defaults` of whatever catalog it was given, so `--router-mode`
/// resolves even when that catalog was built without `default_registry`, and a catalog that
/// already holds the modes keeps them.
#[test]
fn every_installed_catalog_resolves_the_routing_modes() {
    let mut custom_only = RouterPluginRegistry::default();
    register(&mut custom_only).unwrap();
    let mut from_default = default_registry();
    register(&mut from_default).unwrap();

    for registry in [
        with_defaults(RouterPluginRegistry::default()),
        with_defaults(custom_only),
        with_defaults(from_default),
    ] {
        for policy_type in MODES {
            assert!(
                registry.resolve_type(policy_type).is_ok(),
                "{policy_type} should resolve"
            );
        }
    }
}

/// A policy file can name a routing mode on the KV path too. It then selects over the
/// scheduler's candidate rows, so every pick must still be an eligible worker.
#[test]
fn routing_modes_select_eligible_workers_on_the_kv_path() {
    let (workers, request) = fixture(4, 64);
    for policy_type in MODES {
        let policy = policy(policy_type);
        let mut seen = HashSet::new();
        for _ in 0..64 {
            let selected = policy
                .select_worker(selection_input(&workers, &request, 16))
                .unwrap_or_else(|error| panic!("{policy_type}: {error}"));
            assert!(
                workers.contains_key(&selected.worker.worker_id),
                "{policy_type} picked an unknown worker"
            );
            seen.insert(selected.worker.worker_id);
        }
        if policy_type == ROUND_ROBIN {
            assert_eq!(
                seen.len(),
                workers.len(),
                "round-robin reaches every worker"
            );
        }
    }
}
