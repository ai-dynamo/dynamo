// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Runtime parameter control for the KV router selectors of one offline replay.
//!
//! A host that changes router policy parameters during a replay (for example, a planner that
//! retunes the router at a scaling tick) creates a [`ReplayRouterControls`] registry and runs the
//! replay with it as [`ReplayScope::router_controls`](crate::replay::ReplayScope). Offline
//! KV-router compositions constructed in that scope register each placement's selector under its
//! worker type. The host then applies desired parameters with [`apply_router_parameters`] between
//! replay events, so an update takes effect at the current virtual time.

use std::collections::BTreeMap;
use std::sync::Arc;

use anyhow::{Context, Result, anyhow, bail};
use dynamo_custom_policy_builtin::DefaultWorkerSelectorControl;
use dynamo_kv_router::WorkerType;
use dynamo_kv_router::plugins::worker_selection::{RuntimePolicyConfig, RuntimePolicyUpdate};
use parking_lot::Mutex;

/// Desired router policy parameters keyed by worker type (`aggregated`, `prefill` or `decode`),
/// then by policy parameter name (the `PATCH /router_policy` keys).
pub type ReplayRouterParameters = BTreeMap<String, BTreeMap<String, f64>>;

/// Shared registry of the runtime controls a replay's KV router selectors expose.
///
/// Clones share one registry. Controls appear in registration order: one `aggregated` control
/// for an aggregated replay, or one `prefill` and one `decode` control for a disaggregated one.
#[derive(Clone, Debug, Default)]
pub struct ReplayRouterControls {
    controls: Arc<Mutex<Vec<(WorkerType, DefaultWorkerSelectorControl)>>>,
}

impl ReplayRouterControls {
    pub fn new() -> Self {
        Self::default()
    }

    pub(crate) fn register(&self, worker_type: WorkerType, control: DefaultWorkerSelectorControl) {
        self.controls.lock().push((worker_type, control));
    }

    /// Worker types of the registered controls, in registration order.
    fn worker_types(&self) -> Vec<WorkerType> {
        self.controls
            .lock()
            .iter()
            .map(|(worker_type, _)| *worker_type)
            .collect()
    }

    /// Current revision and resolved parameters of every registered control, in registration
    /// order.
    pub fn runtime_configs(&self) -> Result<Vec<(WorkerType, RuntimePolicyConfig)>> {
        self.controls()
            .into_iter()
            .map(|(worker_type, control)| {
                control
                    .runtime_config()
                    .map(|config| (worker_type, config))
                    .with_context(|| format!("reading the {worker_type} replay router policy"))
            })
            .collect()
    }

    fn controls(&self) -> Vec<(WorkerType, DefaultWorkerSelectorControl)> {
        self.controls.lock().clone()
    }

    fn controls_for(&self, worker_type: WorkerType) -> Vec<DefaultWorkerSelectorControl> {
        self.controls
            .lock()
            .iter()
            .filter(|(registered, _)| *registered == worker_type)
            .map(|(_, control)| control.clone())
            .collect()
    }
}

/// Apply desired router policy parameters to the registered replay router controls.
///
/// For each worker type, every registered control of that type is compared with the desired
/// values as `f64`. A control whose parameters already match is left unchanged, since the
/// runtime policy rejects an empty patch; otherwise it receives one revision-checked patch of
/// the differing keys. Desired maps may be partial: omitted keys keep their current values.
///
/// Returns the number of controls updated. Fails on a worker type other than `aggregated`,
/// `prefill` or `decode`, on a worker type with no registered control, on a non-finite value,
/// and on an update the policy rejects (unknown key, invalid value, or a change that needs
/// worker inputs the selector was not built with). Worker types are applied in key order and a
/// failure leaves earlier updates in place, so callers should treat any error as fatal for the
/// replay.
pub fn apply_router_parameters(
    registry: &ReplayRouterControls,
    desired: &ReplayRouterParameters,
) -> Result<usize> {
    let mut updated = 0;
    for (worker_type_name, parameters) in desired {
        let worker_type = parse_router_worker_type(worker_type_name)?;
        let controls = registry.controls_for(worker_type);
        if controls.is_empty() {
            bail!(
                "router_parameters names worker type {worker_type_name:?}, but this replay has no \
                 {worker_type_name} KV router; registered worker types: {:?}",
                registry
                    .worker_types()
                    .iter()
                    .map(WorkerType::as_str)
                    .collect::<Vec<_>>()
            );
        }
        for control in controls {
            let current = control
                .runtime_config()
                .with_context(|| format!("reading the {worker_type} replay router policy"))?;
            let diff = parameter_diff(current.parameters(), parameters)
                .with_context(|| format!("router_parameters for {worker_type}"))?;
            if diff.is_empty() {
                continue;
            }
            let patch = serde_json::Value::Object(diff.clone());
            control
                .update_runtime_config(RuntimePolicyUpdate::patch(current.revision(), diff))
                .with_context(|| {
                    format!("updating the {worker_type} replay router policy with {patch}")
                })?;
            updated += 1;
        }
    }
    Ok(updated)
}

/// Keys of `desired` whose values differ from `current`, compared as `f64`. Keys missing from
/// `current` are kept so the policy reports them as unknown.
fn parameter_diff(
    current: &serde_json::Map<String, serde_json::Value>,
    desired: &BTreeMap<String, f64>,
) -> Result<serde_json::Map<String, serde_json::Value>> {
    let mut diff = serde_json::Map::new();
    for (name, &value) in desired {
        // serde_json maps a non-finite f64 to null, which a patch reads as "reset to startup".
        let number = serde_json::Number::from_f64(value)
            .ok_or_else(|| anyhow!("{name} must be a finite number, got {value}"))?;
        if current.get(name).and_then(serde_json::Value::as_f64) != Some(value) {
            diff.insert(name.clone(), serde_json::Value::Number(number));
        }
    }
    Ok(diff)
}

/// Parse a worker type that owns a replay router: `aggregated`, `prefill` or `decode`.
pub fn parse_router_worker_type(name: &str) -> Result<WorkerType> {
    match name {
        "aggregated" => Ok(WorkerType::Aggregated),
        "prefill" => Ok(WorkerType::Prefill),
        "decode" => Ok(WorkerType::Decode),
        _ => bail!(
            "unknown router worker type {name:?}; expected \"aggregated\", \"prefill\" or \"decode\""
        ),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use dynamo_custom_policy_builtin::DefaultWorkerSelector;
    use dynamo_kv_router::KvRouterConfig;
    use dynamo_kv_router::plugins::worker_selection::RuntimePolicyError;

    fn registry_with(
        controls: &[(WorkerType, KvRouterConfig)],
    ) -> (ReplayRouterControls, Vec<DefaultWorkerSelector>) {
        let registry = ReplayRouterControls::new();
        let selectors = controls
            .iter()
            .map(|(worker_type, config)| {
                let selector = DefaultWorkerSelector::new(Some(config.clone()), "replay");
                registry.register(*worker_type, selector.runtime_control());
                selector
            })
            .collect();
        (registry, selectors)
    }

    fn desired(entries: &[(&str, &[(&str, f64)])]) -> ReplayRouterParameters {
        entries
            .iter()
            .map(|(worker_type, parameters)| {
                (
                    (*worker_type).to_owned(),
                    parameters
                        .iter()
                        .map(|(name, value)| ((*name).to_owned(), *value))
                        .collect(),
                )
            })
            .collect()
    }

    fn revisions(registry: &ReplayRouterControls) -> Vec<(WorkerType, u64)> {
        registry
            .runtime_configs()
            .unwrap()
            .into_iter()
            .map(|(worker_type, config)| (worker_type, config.revision()))
            .collect()
    }

    #[test]
    fn startup_map_is_a_no_op_and_only_changed_keys_are_patched() {
        let config = KvRouterConfig {
            overlap_score_credit: 2.0,
            router_temperature: 0.5,
            ..Default::default()
        };
        let (registry, _selectors) = registry_with(&[
            (WorkerType::Prefill, config.clone()),
            (WorkerType::Decode, config),
        ]);
        // The startup map is what each registered control reports before any update.
        let startup: ReplayRouterParameters = registry
            .runtime_configs()
            .unwrap()
            .into_iter()
            .map(|(worker_type, config)| {
                let parameters = config.parameters().iter();
                let parameters =
                    parameters.map(|(name, value)| (name.clone(), value.as_f64().unwrap()));
                (worker_type.as_str().to_owned(), parameters.collect())
            })
            .collect();

        assert_eq!(apply_router_parameters(&registry, &startup).unwrap(), 0);
        assert_eq!(
            apply_router_parameters(&registry, &Default::default()).unwrap(),
            0
        );
        assert_eq!(
            revisions(&registry),
            [(WorkerType::Prefill, 0), (WorkerType::Decode, 0)]
        );

        // Integer-valued floats equal their JSON numbers; only the changed key is patched.
        let mut changed = startup.clone();
        changed
            .get_mut("prefill")
            .unwrap()
            .insert("router_temperature".to_owned(), 1.0);
        assert_eq!(apply_router_parameters(&registry, &changed).unwrap(), 1);
        let configs = registry.runtime_configs().unwrap();
        assert_eq!(configs[0].1.revision(), 1);
        assert_eq!(configs[0].1.parameters()["router_temperature"], 1.0);
        assert_eq!(configs[0].1.parameters()["overlap_score_credit"], 2.0);
        assert_eq!(configs[1].1.revision(), 0);

        // Level-triggered: re-applying the same desired map changes nothing.
        assert_eq!(apply_router_parameters(&registry, &changed).unwrap(), 0);
        assert_eq!(
            revisions(&registry),
            [(WorkerType::Prefill, 1), (WorkerType::Decode, 0)]
        );
    }

    #[test]
    fn invalid_parameters_are_errors_and_leave_the_revision_unchanged() {
        let (registry, _selectors) = registry_with(&[
            (
                WorkerType::Prefill,
                KvRouterConfig {
                    overlap_score_credit: 0.0,
                    host_cache_hit_weight: 0.0,
                    disk_cache_hit_weight: 0.0,
                    ..Default::default()
                },
            ),
            (WorkerType::Aggregated, KvRouterConfig::default()),
        ]);
        let temperature = "router_temperature";
        for (worker_type, name, value, expected) in [
            ("aggregated", temperature, -1.0, temperature),
            ("aggregated", temperature, f64::NAN, temperature),
            ("aggregated", temperature, f64::INFINITY, temperature),
            ("aggregated", "not_a_parameter", 1.0, "not_a_parameter"),
            // A startup credit of zero builds no cache scorer, so enabling credit is rejected.
            (
                "prefill",
                "overlap_score_credit",
                1.0,
                "overlap_score_credit",
            ),
            // Worker type names are strict and must name a registered KV router.
            ("bogus", temperature, 1.0, "bogus"),
            ("encode", temperature, 1.0, "encode"),
            ("Aggregated", temperature, 1.0, "Aggregated"),
            ("DECODE", temperature, 1.0, "DECODE"),
            (" decode", temperature, 1.0, " decode"),
            ("decode", temperature, 1.0, "no decode KV router"),
        ] {
            let error =
                apply_router_parameters(&registry, &desired(&[(worker_type, &[(name, value)])]))
                    .unwrap_err();
            assert!(
                format!("{error:#}").contains(expected),
                "{worker_type:?}.{name}={value}: {error:#}"
            );
            assert_eq!(
                revisions(&registry),
                [(WorkerType::Prefill, 0), (WorkerType::Aggregated, 0)]
            );
        }
        // An empty registry, as under a round-robin router, rejects any worker type.
        let error = apply_router_parameters(
            &ReplayRouterControls::new(),
            &desired(&[("aggregated", &[])]),
        )
        .unwrap_err();
        assert!(error.to_string().contains("no aggregated KV router"));
        let error = apply_router_parameters(
            &registry,
            &desired(&[("prefill", &[("overlap_score_credit", 1.0)])]),
        )
        .unwrap_err();
        assert!(
            error
                .chain()
                .any(|cause| cause.downcast_ref::<RuntimePolicyError>().is_some()),
            "{error:#}"
        );
    }
}
