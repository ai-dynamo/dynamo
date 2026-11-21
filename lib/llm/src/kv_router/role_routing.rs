// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! The routing settings each worker role runs with.
//!
//! A frontend routes two kinds of worker pool: serving sets (aggregated or decode), and the
//! prefill pool in front of a decode set. A role's settings come from the frontend's
//! [`RouterConfig`] and from the `router_config` its workers advertise on their model card.
//! The rules that combine the two live here, and so does the one function that turns a role's
//! router mode into the policy its [`RoutingHost`](super::RoutingHost) runs.

use std::borrow::Cow;
use std::time::Duration;

use dynamo_kv_router::config::{DYN_ROUTER_WORKER_SELECTION_POLICY, KvRouterConfig};
use dynamo_kv_router::plugins::worker_selection::{WorkerInputs, WorkerSelectionPolicyFactory};
use dynamo_kv_router::plugins::{RouterPluginRegistry, WorkerSelectionPolicyRegistryError};
use dynamo_kv_router::scheduling::PolicyProfile;
use dynamo_router_policies::router_modes;
use dynamo_router_protocol::identity::RoutingPartitionRef;
use dynamo_runtime::pipeline::RouterMode;
use dynamo_worker_selection::host::{HostedWorkerPicker, SelectionConfig};

use super::embedded::embedded_partition_key;
use super::plugins::router_plugin_registry;
use super::routing_host::{HostedPolicy, RoutingPolicy};
use super::{PreparedSelectionPolicy, SelectionPolicySource};
use crate::entrypoint::{RouterConfig, RouterModeSetting};
use crate::session_affinity::SessionAffinityMode;
use crate::worker_type::WorkerType;

/// Resolve the routing settings of a serving worker set (aggregated or decode).
///
/// A set that advertises a `router_config` replaces the frontend's configuration whole; it
/// does not merge with it. Two settings stay the frontend's: its worker-selection policy
/// overrides, which are process-local and never travel on a card, and the session-affinity
/// mode.
pub(crate) fn serving_set_routing<'a>(
    advertised: Option<&'a RouterConfig>,
    frontend: &'a RouterConfig,
) -> Cow<'a, RouterConfig> {
    let Some(advertised) = advertised else {
        return Cow::Borrowed(frontend);
    };

    let mut routing = advertised.clone();
    routing.kv_router_config.router_prefill_policy =
        frontend.kv_router_config.router_prefill_policy.clone();
    routing.kv_router_config.router_decode_policy =
        frontend.kv_router_config.router_decode_policy.clone();
    routing.session_affinity_mode = frontend.session_affinity_mode;
    Cow::Owned(routing)
}

/// What the prefill hop inherits from the decode set it fronts.
#[derive(Debug)]
pub(crate) struct DecodeRouting {
    pub(crate) router_mode: RouterModeSetting,
    /// The decode set's KV configuration as handed to the prefill router. `None` leaves an
    /// inheriting prefill KV router on its defaults.
    pub(crate) kv_router_config: Option<KvRouterConfig>,
    pub(crate) session_affinity_ttl: Option<Duration>,
    pub(crate) session_affinity_mode: SessionAffinityMode,
}

/// The routing settings the prefill hop runs with.
#[derive(Debug)]
pub(crate) struct PrefillRouting {
    pub(crate) router_mode: RouterModeSetting,
    pub(crate) kv_router_config: Option<KvRouterConfig>,
    pub(crate) session_affinity_ttl: Option<Duration>,
    pub(crate) session_affinity_mode: SessionAffinityMode,
    /// Whether the prefill workers advertised their own `router_config`.
    pub(crate) is_advertised: bool,
}

/// Resolve the routing settings of the prefill hop.
///
/// Prefill workers that advertise a `router_config` govern the hop. Its mode, KV configuration
/// and session-affinity TTL are taken whole, so an advertisement with no TTL means no affinity,
/// not the decode set's TTL. Workers that advertise nothing inherit the decode set's settings.
/// That is what lets a deployment run KV-routed prefill in front of round-robin decode. The
/// session-affinity mode is always the decode set's, which is the frontend's.
///
/// Whichever KV configuration the hop gets, it tracks neither active nor output blocks: prefill
/// routing is prompt-side, and crediting decode blocks here would double-count load. The policy
/// file's `router` settings are finalized first, because the KV router applies them again and
/// would turn both back on. An unreadable policy file is the error.
pub(crate) fn prefill_routing(
    advertised: Option<RouterConfig>,
    decode: DecodeRouting,
) -> Result<PrefillRouting, String> {
    let Some(advertised) = advertised else {
        return Ok(PrefillRouting {
            router_mode: decode.router_mode,
            kv_router_config: decode
                .kv_router_config
                .map(without_block_tracking)
                .transpose()?,
            session_affinity_ttl: decode.session_affinity_ttl,
            session_affinity_mode: decode.session_affinity_mode,
            is_advertised: false,
        });
    };

    Ok(PrefillRouting {
        router_mode: advertised.router_mode,
        kv_router_config: Some(without_block_tracking(advertised.kv_router_config)?),
        session_affinity_ttl: advertised
            .session_affinity_ttl_secs
            .map(Duration::from_secs),
        session_affinity_mode: decode.session_affinity_mode,
        is_advertised: true,
    })
}

fn without_block_tracking(mut config: KvRouterConfig) -> Result<KvRouterConfig, String> {
    config.finalize_policy_config()?;
    config.router_track_active_blocks = false;
    config.router_track_output_blocks = false;
    Ok(config)
}

/// What a role's routing policy is resolved from, beside its router mode.
pub struct RoleInputs<'a> {
    pub role: WorkerType,
    /// The role's KV configuration.
    pub config: &'a KvRouterConfig,
    /// The model the role serves. The policy is built for this model's partition, so the
    /// caller gives the role's [`KvRouter`](super::KvRouter) the same name.
    pub model_name: Option<&'a str>,
    /// Where `kv` gets the role's worker-selection policy: its entry in the policy file, else
    /// the host's default.
    pub selection: SelectionPolicySource,
    /// The caller installs a request classifier on the role's `KvRouter`.
    pub has_request_classifier: bool,
    /// LoRA serving is on.
    pub lora_enabled: bool,
    /// The caller books workers through the role's `KvRouter` itself, as the Runtime EPP's
    /// prefill reservations do.
    pub reserves_workers: bool,
    /// A busy threshold is set for the role.
    pub busy_thresholds: bool,
    /// The card the role is built from advertises routing taints, topology domains
    /// included. Other workers of the set are not read.
    pub workers_have_taints: bool,
    /// The decode set this prefill hop fronts runs conditional disaggregation. A serving
    /// set's own setting is read from `config`.
    pub fronts_conditional_disagg: bool,
}

/// What a role's router mode resolves to.
pub enum RolePolicy {
    /// KV routing. The role needs its own [`KvRouter`](super::KvRouter): the caller builds it
    /// and routes with [`RoutingPolicy::Kv`].
    Kv {
        /// The role's worker-selection policy, already built for the model's partition: the
        /// caller builds the `KvRouter` for the same model name.
        selection: SelectionPolicySource,
        /// The policy's name in logs: its type, or `default`.
        policy: String,
        /// What the role uses that only the KV path provides, when it uses any.
        kv_needed_by: Option<KvPathReason>,
    },
    /// A policy that needs no KV state.
    Ready(RoutingPolicy),
}

impl RolePolicy {
    /// The policy's name in the activation logs.
    pub(crate) fn name(&self) -> &str {
        match self {
            Self::Kv { policy, .. } => policy,
            Self::Ready(RoutingPolicy::Hosted(hosted)) => hosted.label(),
            Self::Ready(policy) => policy.transport_mode().telemetry_label(),
        }
    }

    pub(crate) fn has_kv_router(&self) -> bool {
        matches!(self, Self::Kv { .. })
    }

    pub(crate) fn kv_needed_by(&self) -> Option<KvPathReason> {
        match self {
            Self::Kv { kv_needed_by, .. } => *kv_needed_by,
            Self::Ready(_) => None,
        }
    }
}

/// Something a role uses that only the KV path provides. A role that uses one keeps its
/// `KvRouter` even when its policy is only a picker: without it the setting would be dropped
/// silently, or the role would fail to build.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum KvPathReason {
    /// Policy classes for the model, or a queue threshold: requests queue in the scheduler.
    PolicyClasses,
    /// A busy threshold. Only the scheduler's bookings give the prefill-token thresholds,
    /// and the booked half of the decode-block threshold, a load to judge.
    BusyThresholds,
    /// A request classifier, which the `KvRouter` runs.
    RequestClassifier,
    /// Hints listed for the role, which run when the scheduler books a request.
    Hints,
    /// Conditional disaggregation, which previews the decode route and probes prefill load
    /// through the two roles' `KvRouter`s.
    ConditionalDisagg,
    /// LoRA serving. The KV path filters workers by adapter for every policy and role. The
    /// hosted path does not on the prefill hop, with session affinity, or for a picker that
    /// reads active requests.
    Lora,
    /// Workers that advertise routing taints. Only the scheduler filters workers by the
    /// taints a request requires, as topology-aware KV transfer does.
    WorkerTaints,
    /// Router replica sync under a picker that reads active requests. Only the scheduler's
    /// counts include the requests other frontends sent.
    ReplicaSync,
    /// A prefill load model, which predicts load for the scheduler.
    PrefillLoadModel,
    /// Reservations a caller books itself. Without a `KvRouter` a reservation is only a
    /// preview: it books nothing and ignores the caller's worker subset.
    Reservations,
}

impl std::fmt::Display for KvPathReason {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(match self {
            Self::PolicyClasses => "policy classes or a queue threshold",
            Self::BusyThresholds => "a busy threshold",
            Self::RequestClassifier => "a request classifier",
            Self::Hints => "hints listed for the role",
            Self::ConditionalDisagg => "conditional disaggregation",
            Self::Lora => "LoRA serving",
            Self::WorkerTaints => "workers that advertise routing taints",
            Self::ReplicaSync => "router replica sync with a picker that reads active requests",
            Self::PrefillLoadModel => "a prefill load model",
            Self::Reservations => "worker reservations",
        })
    }
}

/// The one list of what keeps a role on the KV path. `config` has the policy file's router
/// settings applied, and `policy_inputs` are the inputs the role's policy reads.
fn kv_path_reason(
    role: &RoleInputs<'_>,
    config: &KvRouterConfig,
    policy_inputs: WorkerInputs,
) -> anyhow::Result<Option<KvPathReason>> {
    // The profile of a model with no policy classes and no queue threshold: nothing queues.
    let unqueued = PolicyProfile::synthetic(None, config.router_queue_policy);
    Ok(if config.policy_profile(role.model_name)? != unqueued {
        Some(KvPathReason::PolicyClasses)
    } else if role.busy_thresholds {
        Some(KvPathReason::BusyThresholds)
    } else if role.has_request_classifier || config.request_classifier_config()?.is_some() {
        Some(KvPathReason::RequestClassifier)
    } else if lists_hints(config, role.role)? {
        Some(KvPathReason::Hints)
    } else if config.conditional_disagg_enabled || role.fronts_conditional_disagg {
        Some(KvPathReason::ConditionalDisagg)
    } else if role.lora_enabled {
        Some(KvPathReason::Lora)
    } else if role.workers_have_taints {
        Some(KvPathReason::WorkerTaints)
    } else if config.router_replica_sync && policy_inputs.contains(WorkerInputs::ACTIVE_REQUESTS) {
        Some(KvPathReason::ReplicaSync)
    } else if config.router_prefill_load_model.is_enabled() {
        Some(KvPathReason::PrefillLoadModel)
    } else if role.reserves_workers {
        Some(KvPathReason::Reservations)
    } else {
        None
    })
}

/// Whether the role's `worker_selection` entry lists hints. Hints the role runs by default
/// are passive and do not count.
fn lists_hints(config: &KvRouterConfig, role: WorkerType) -> anyhow::Result<bool> {
    Ok(config
        .worker_selection_config()?
        .and_then(|worker_selection| worker_selection.hints(role))
        .is_some_and(|entries| !entries.is_empty()))
}

/// The policy type `config` selects for the role, or `None` for the default policy. A
/// deprecated `instances` entry is selected by its name and resolves to its type.
fn selected_policy_type(
    config: &KvRouterConfig,
    role: WorkerType,
) -> anyhow::Result<Option<String>> {
    let Some(selected) = config.selected_worker_selection_policy_instance_for(role)? else {
        return Ok(None);
    };
    Ok(Some(
        config
            .worker_selection_config()?
            .and_then(|worker_selection| worker_selection.instance(&selected))
            .map_or(selected.as_str(), |instance| instance.policy_type())
            .to_owned(),
    ))
}

/// Resolve the policy one role runs.
///
/// Under `kv` the policy is the role's own: its entry in the policy file, else the default.
/// A policy type the mode names is every role's policy. Either way the role's path follows
/// from the policy: one that is only a picker runs hosted, as round-robin does, unless the
/// role uses something only the KV path provides ([`KvPathReason`]); any other policy is
/// the role's worker-selection policy on the KV path. The other router modes name their
/// policy themselves and never take the KV path.
pub fn role_policy(mode: &RouterModeSetting, role: &RoleInputs<'_>) -> anyhow::Result<RolePolicy> {
    role_policy_in(&router_plugin_registry(), mode, role)
}

fn role_policy_in(
    registry: &RouterPluginRegistry,
    mode: &RouterModeSetting,
    role: &RoleInputs<'_>,
) -> anyhow::Result<RolePolicy> {
    let named_type = match mode {
        RouterModeSetting::Mode(RouterMode::KV) => None,
        RouterModeSetting::Mode(mode) => return fixed_mode_policy(*mode).map(RolePolicy::Ready),
        RouterModeSetting::PolicyType(policy_type) => Some(policy_type.as_str()),
    };
    let worker_type = role.role;
    // The KV router applies the policy file's router settings before it builds the policy.
    let mut config = role.config.clone();
    config.apply_policy_config().map_err(anyhow::Error::msg)?;

    let (policy_type, source) = match named_type {
        None => (
            selected_policy_type(&config, worker_type)?,
            role.selection.clone(),
        ),
        Some(policy_type) => (
            Some(policy_type.to_owned()),
            SelectionPolicySource::Prepared(PreparedSelectionPolicy::prepare(
                registry,
                resolve_policy_type(registry, policy_type)?,
                Some(policy_type),
                &config,
                worker_type,
                role.model_name,
            )?),
        ),
    };
    // Built once for the model's partition. The KV path gets this instance back, so finding
    // the role's path costs no second factory call.
    let prepared = source.prepare(&config, worker_type, worker_type.as_str(), role.model_name)?;
    let policy = policy_type.as_deref().unwrap_or("default");

    let kv_needed_by = kv_path_reason(role, &config, prepared.inputs())?;
    // Once the instance is taken, this builds further ones.
    let factory = prepared.factory.clone();
    let prepared = match kv_needed_by {
        Some(_) => prepared,
        None => match prepared.into_hosted_picker() {
            Ok(picker) => {
                return hosted_policy(policy, picker, factory, &config, role)
                    .map(|hosted| RolePolicy::Ready(RoutingPolicy::Hosted(hosted)));
            }
            Err(prepared) => prepared,
        },
    };
    Ok(RolePolicy::Kv {
        selection: SelectionPolicySource::Prepared(prepared),
        policy: policy.to_owned(),
        kv_needed_by,
    })
}

/// The hosted policy of a role whose policy is only a picker. The KV path would have
/// filtered workers by the taints a request requires, so this one fails such a request.
fn hosted_policy(
    policy_type: &str,
    picker: HostedWorkerPicker,
    factory: WorkerSelectionPolicyFactory,
    config: &KvRouterConfig,
    role: &RoleInputs<'_>,
) -> anyhow::Result<HostedPolicy> {
    // A type that a router mode also names opens the transport in that mode, which names
    // it in the transport's own logs.
    if let Some(mode) = HOSTED_MODES
        .into_iter()
        .find(|&mode| routing_mode_policy_type(mode) == Some(policy_type))
    {
        return HostedPolicy::for_mode(mode).map(HostedPolicy::rejecting_required_taints);
    }
    let fresh_picker = {
        let policy_type = policy_type.to_owned();
        let selection = SelectionConfig::from(config);
        let worker_type = role.role;
        let partition = embedded_partition_key(role.model_name);
        move || {
            factory(&selection, worker_type, partition.as_ref())
                .into_hosted_picker()
                .map_err(|_| anyhow::anyhow!("{policy_type} policy cannot run without KV routing"))
        }
    };
    Ok(HostedPolicy::for_type(policy_type, picker, fresh_picker).rejecting_required_taints())
}

/// The factory of the policy type the router mode names. A value that is not a registered
/// type may be a mistyped router mode, so the error lists both.
fn resolve_policy_type(
    registry: &RouterPluginRegistry,
    policy_type: &str,
) -> anyhow::Result<WorkerSelectionPolicyFactory> {
    registry
        .resolve_type(policy_type)
        .map_err(|error| match error {
            WorkerSelectionPolicyRegistryError::UnknownType { name, available } => {
                let modes = RouterModeSetting::MODES.map(|mode| mode.telemetry_label());
                anyhow::anyhow!(
                    "--router-mode {name:?} is neither a router mode ({modes}) nor a registered \
                     worker-selection policy type ({available})",
                    modes = modes.join(", "),
                )
            }
            error => error.into(),
        })
}

/// The policy of a router mode that names it itself. `kv` names none: the role's policy
/// comes from [`role_policy`].
fn fixed_mode_policy(mode: RouterMode) -> anyhow::Result<RoutingPolicy> {
    Ok(match mode {
        RouterMode::KV => anyhow::bail!("KV routing runs the role's worker-selection policy"),
        RouterMode::Direct => RoutingPolicy::Direct,
        RouterMode::DeviceAwareWeighted => RoutingPolicy::DeviceAwareWeighted,
        // `hosted_picker` returns the removed-mode error for `PowerOfTwoChoices`.
        RouterMode::RoundRobin
        | RouterMode::Random
        | RouterMode::LeastLoaded
        | RouterMode::PowerOfTwoChoices => RoutingPolicy::Hosted(HostedPolicy::for_mode(mode)?),
    })
}

/// Fail at startup when the router mode names a policy type that cannot run: one the linked
/// registry lacks, or one given beside another selection of the roles' policies.
pub(crate) fn check_policy_type_mode(router_config: &RouterConfig) -> anyhow::Result<()> {
    let Some(policy_type) = router_config.router_mode.policy_type() else {
        return Ok(());
    };
    resolve_policy_type(&router_plugin_registry(), policy_type)?;
    let config = &router_config.kv_router_config;
    anyhow::ensure!(
        config.router_policy_config.is_none(),
        "--router-mode {policy_type} and --router-policy-config are mutually exclusive: the \
         policy file selects each role's policy. Remove --router-mode, or name the type under \
         worker_selection in the file"
    );
    if let Some(other) = config.selected_worker_selection_policy_instance()? {
        anyhow::bail!(
            "--router-mode {policy_type} selects every role's worker-selection policy, but \
             {other} is selected too; remove --router-prefill-policy, --router-decode-policy \
             and {DYN_ROUTER_WORKER_SELECTION_POLICY}"
        );
    }
    Ok(())
}

/// The policy of a mode that needs no [`KvRouter`](super::KvRouter).
#[cfg(test)]
pub(crate) fn non_kv_policy(mode: RouterMode) -> RoutingPolicy {
    fixed_mode_policy(mode).expect("mode must resolve to a policy")
}

/// The router modes that run a registered policy type as a hosted picker.
const HOSTED_MODES: [RouterMode; 3] = [
    RouterMode::RoundRobin,
    RouterMode::Random,
    RouterMode::LeastLoaded,
];

/// The plugin type a routing mode runs as a hosted picker, if it runs as one.
fn routing_mode_policy_type(mode: RouterMode) -> Option<&'static str> {
    match mode {
        RouterMode::RoundRobin => Some(router_modes::ROUND_ROBIN),
        RouterMode::Random => Some(router_modes::RANDOM),
        RouterMode::LeastLoaded => Some(router_modes::LEAST_LOADED),
        _ => None,
    }
}

/// Resolve a routing mode to its policy by type name in the linked plugin registry, the same way
/// a policy file names a type, and take out its picker.
pub(crate) fn hosted_picker(mode: RouterMode) -> anyhow::Result<HostedWorkerPicker> {
    let policy_type = match (mode, routing_mode_policy_type(mode)) {
        (_, Some(policy_type)) => policy_type,
        // TODO(v1.9): Remove with the `PowerOfTwoChoices` wire value.
        (RouterMode::PowerOfTwoChoices, None) => anyhow::bail!(
            "the power-of-two router mode was removed in Dynamo v1.7; use another router mode"
        ),
        (mode, None) => anyhow::bail!("{mode:?} routing is not a routing-mode policy"),
    };
    let factory = router_plugin_registry().resolve_type(policy_type)?;
    // Routing-mode policies read neither the role nor the partition.
    let policy = factory(
        &SelectionConfig::default(),
        WorkerType::Aggregated,
        RoutingPartitionRef::new("", ""),
    );
    policy
        .into_hosted_picker()
        .map_err(|_| anyhow::anyhow!("{policy_type} policy cannot run without KV routing"))
}

/// Every worker role a frontend routes.
pub(crate) const EVERY_ROLE: [WorkerType; 4] = [
    WorkerType::Aggregated,
    WorkerType::Prefill,
    WorkerType::Decode,
    WorkerType::Encode,
];

/// Reject a non-KV routing mode for `roles` when one of them lists `hints` in its `worker_selection`
/// entry. Hints run at booking, which only the KV path does, so another mode would drop them
/// silently. Default hints are passive and never require the KV path.
pub(crate) fn check_hint_policies_route_on_kv_path(
    router_mode: RouterMode,
    config: &KvRouterConfig,
    roles: &[WorkerType],
) -> anyhow::Result<()> {
    if router_mode.is_kv_routing() {
        return Ok(());
    }
    let Some(worker_selection) = config.worker_selection_config()? else {
        return Ok(());
    };
    let Some(role) = roles.iter().find(|&&role| {
        worker_selection
            .hints(role)
            .is_some_and(|entries| !entries.is_empty())
    }) else {
        return Ok(());
    };
    let role = role.as_str();
    let fix = match routing_mode_policy_type(router_mode) {
        Some(policy_type) => format!(
            "run --router-mode kv and name {policy_type:?} under worker_selection.{role} in the router policy file, or remove worker_selection.{role}.hints"
        ),
        None => format!("run --router-mode kv, or remove worker_selection.{role}.hints"),
    };
    anyhow::bail!(
        "the hints in worker_selection.{role} run only on the KV path, but --router-mode {mode} routes {role} workers without it; {fix}",
        mode = router_mode.telemetry_label(),
    )
}

/// Log the settings a serving worker set routes with, each time its pipeline is built.
///
/// These are the resolved values, not what a card or a flag asked for: an advertisement
/// replaces the frontend's configuration whole, so a worker set that names a mode without
/// restating its tuning runs on defaults, and this line is what shows it. `policy` is the
/// set's resolved policy, which only a set that routes tokenized requests has.
pub(crate) fn log_serving_set_routing(
    model_name: &str,
    group: &str,
    worker_type: WorkerType,
    advertised: bool,
    routing: &RouterConfig,
    policy: Option<&RolePolicy>,
) {
    let kv_router = policy.map_or(
        routing.router_mode.fixed_mode().is_kv_routing(),
        RolePolicy::has_kv_router,
    );
    tracing::info!(
        model_name,
        group = %group,
        ?worker_type,
        router_mode = ?routing.router_mode,
        policy = policy.map(RolePolicy::name),
        kv_router = policy.map(RolePolicy::has_kv_router),
        kv_router_needed_by = policy
            .and_then(RolePolicy::kv_needed_by)
            .map(tracing::field::display),
        advertised,
        session_affinity_ttl = ?routing.session_affinity_ttl_secs.map(Duration::from_secs),
        session_affinity_mode = ?routing.session_affinity_mode,
        busy_thresholds = ?routing.load_threshold_config,
        kv_tuning = ?kv_router.then_some(&routing.kv_router_config),
        "Activating router"
    );
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;
    use std::sync::atomic::{AtomicUsize, Ordering};

    use dynamo_kv_router::plugins::prelude::{
        WorkerInputView, WorkerInputs, WorkerPicker, WorkerSelectionContext, WorkerSelectionPolicy,
        WorkerSelectionPolicyError,
    };
    use dynamo_scheduler::host::RouterPrefillLoadModel;
    use dynamo_worker_selection::host::HostedSelectionRequest;

    use super::*;

    fn decode(router_mode: RouterMode, session_affinity_ttl: Option<Duration>) -> DecodeRouting {
        DecodeRouting {
            router_mode: router_mode.into(),
            kv_router_config: None,
            session_affinity_ttl,
            session_affinity_mode: SessionAffinityMode::Hard,
        }
    }

    #[test]
    fn every_router_mode_resolves_to_its_policy_and_its_own_transport_mode() {
        use dynamo_kv_router::plugins::worker_selection::WorkerInputs;

        // The transport mode keeps each mode's log label and gives least-loaded its occupancy
        // state, so a hosted policy must open the transport in the mode it came from.
        for (mode, picker_inputs) in [
            (RouterMode::RoundRobin, Some(WorkerInputs::NONE)),
            (RouterMode::Random, Some(WorkerInputs::NONE)),
            (RouterMode::LeastLoaded, Some(WorkerInputs::ACTIVE_REQUESTS)),
            (RouterMode::Direct, None),
            (RouterMode::DeviceAwareWeighted, None),
        ] {
            let policy = non_kv_policy(mode);
            assert_eq!(policy.transport_mode(), mode);
            match (mode, &policy) {
                (RouterMode::Direct, RoutingPolicy::Direct)
                | (RouterMode::DeviceAwareWeighted, RoutingPolicy::DeviceAwareWeighted) => {}
                (_, RoutingPolicy::Hosted(hosted)) => {
                    assert_eq!(Some(hosted.picker().inputs()), picker_inputs, "{mode:?}");
                    // A router mode never filtered workers by taint.
                    assert!(!hosted.rejects_required_taints(), "{mode:?}");
                }
                _ => panic!("{mode:?} resolved to the wrong policy"),
            }
        }

        let Err(error) = fixed_mode_policy(RouterMode::PowerOfTwoChoices) else {
            panic!("the removed power-of-two mode must not resolve");
        };
        assert!(error.to_string().contains("was removed"), "{error}");
    }

    /// The picker of the test policy types. It picks the last worker, which neither a
    /// round-robin nor a least-loaded picker does first, and runs hosted when it declares no
    /// inputs.
    struct LastWorkerPicker(WorkerInputs);

    impl WorkerPicker for LastWorkerPicker {
        fn required_worker_inputs(&self) -> WorkerInputs {
            self.0
        }

        fn pick(
            &self,
            _context: &WorkerSelectionContext<'_>,
            input: WorkerInputView<'_>,
        ) -> Result<usize, WorkerSelectionPolicyError> {
            Ok(input.len() - 1)
        }
    }

    const HOSTED_TYPE: &str = "test-last-worker";
    const KV_TYPE: &str = "test-last-worker-by-load";

    /// The registry of a stock frontend (the router modes plus the policies Dynamo ships) with
    /// the two test types, and the number of policies the test types have built.
    fn registry() -> (RouterPluginRegistry, Arc<AtomicUsize>) {
        let mut registry = dynamo_router_policies::default_registry();
        dynamo_router_policies::register(&mut registry).unwrap();
        let built = Arc::new(AtomicUsize::new(0));
        for (policy_type, inputs) in [
            (HOSTED_TYPE, WorkerInputs::NONE),
            (KV_TYPE, WorkerInputs::LOAD),
        ] {
            let built = Arc::clone(&built);
            registry
                .register_worker_selection(
                    policy_type,
                    Arc::new(move |_| {
                        let built = Arc::clone(&built);
                        Ok(Arc::new(move |config: &SelectionConfig, role, _| {
                            built.fetch_add(1, Ordering::SeqCst);
                            WorkerSelectionPolicy::new(
                                config.clone(),
                                role.as_str(),
                                Vec::new(),
                                Box::new(LastWorkerPicker(inputs)),
                            )
                        }))
                    }),
                )
                .unwrap();
        }
        (registry, built)
    }

    /// A role of the model `model` that uses nothing beside its policy.
    fn role(role: WorkerType, config: &KvRouterConfig) -> RoleInputs<'_> {
        RoleInputs {
            role,
            config,
            model_name: Some("model"),
            selection: SelectionPolicySource::Registry,
            has_request_classifier: false,
            lora_enabled: false,
            reserves_workers: false,
            busy_thresholds: false,
            workers_have_taints: false,
            fronts_conditional_disagg: false,
        }
    }

    fn policy_type_policy(
        registry: &RouterPluginRegistry,
        policy_type: &str,
        config: &KvRouterConfig,
    ) -> anyhow::Result<RolePolicy> {
        role_policy_in(
            registry,
            &RouterModeSetting::PolicyType(policy_type.to_owned()),
            &role(WorkerType::Decode, config),
        )
    }

    /// A policy file and the configuration that names it.
    fn policy_file(yaml: &str) -> (tempfile::NamedTempFile, KvRouterConfig) {
        let file = tempfile::NamedTempFile::new().unwrap();
        std::fs::write(file.path(), yaml).unwrap();
        let config = KvRouterConfig {
            router_policy_config: Some(file.path().display().to_string()),
            ..Default::default()
        };
        (file, config)
    }

    /// The policy `kv` resolves for one role of a frontend that resolved the file's
    /// selections at startup, as the Python frontend does.
    fn file_role_policy(
        registry: &RouterPluginRegistry,
        worker_type: WorkerType,
        config: &KvRouterConfig,
    ) -> RolePolicy {
        let selection = SelectionPolicySource::Factory(registry.resolve(config).unwrap().unwrap());
        role_policy_in(
            registry,
            &RouterMode::KV.into(),
            &RoleInputs {
                selection,
                ..role(worker_type, config)
            },
        )
        .unwrap()
    }

    fn pick_of_three(picker: &HostedWorkerPicker) -> u64 {
        picker
            .select(&[1, 2, 3], None, &HostedSelectionRequest::new(0))
            .unwrap()
    }

    #[test]
    fn a_file_role_runs_hosted_when_its_policy_is_only_a_picker() {
        let (registry, built) = registry();
        let (_file, config) = policy_file(&format!(
            "worker_selection:\n  aggregated: {HOSTED_TYPE}\n  prefill: random\n  decode: dynamo-default-cost-fn\n"
        ));

        // The path is each role's own: a cost function keeps its KV router beside two
        // pickers that run without one.
        let RolePolicy::Kv {
            policy,
            kv_needed_by,
            ..
        } = file_role_policy(&registry, WorkerType::Decode, &config)
        else {
            panic!("a cost function must route on the KV path");
        };
        assert_eq!(
            (policy.as_str(), kv_needed_by),
            ("dynamo-default-cost-fn", None)
        );

        // A type that a router mode also names opens the transport in that mode, so the
        // transport's logs name `random` and not a stand-in.
        let RolePolicy::Ready(policy) = file_role_policy(&registry, WorkerType::Prefill, &config)
        else {
            panic!("random must run hosted");
        };
        assert_eq!(policy.transport_mode(), RouterMode::Random);
        // On the KV path the role filtered workers by required taints. Hosted, it fails a
        // request that requires any.
        assert!(
            matches!(&policy, RoutingPolicy::Hosted(hosted) if hosted.rejects_required_taints())
        );

        // Any other picker runs under its own name, built once, and its second picker (LoRA
        // filtering) is its own too.
        let RolePolicy::Ready(RoutingPolicy::Hosted(hosted)) =
            file_role_policy(&registry, WorkerType::Aggregated, &config)
        else {
            panic!("{HOSTED_TYPE} must run hosted");
        };
        assert_eq!(hosted.label(), HOSTED_TYPE);
        assert!(hosted.rejects_required_taints());
        assert_eq!(pick_of_three(hosted.picker()), 3);
        assert_eq!(built.load(Ordering::SeqCst), 1);
        assert_eq!(pick_of_three(&hosted.fresh_picker().unwrap()), 3);

        // The same role keeps its KV router when it uses something only the KV path has,
        // and that router's partition gets the instance that was probed.
        built.store(0, Ordering::SeqCst);
        let config = KvRouterConfig {
            router_queue_threshold: Some(0.5),
            ..config
        };
        let RolePolicy::Kv {
            selection,
            policy,
            kv_needed_by,
        } = file_role_policy(&registry, WorkerType::Aggregated, &config)
        else {
            panic!("a queue threshold needs the KV path");
        };
        assert_eq!(
            (policy.as_str(), kv_needed_by),
            (HOSTED_TYPE, Some(KvPathReason::PolicyClasses))
        );
        let factory = selection
            .resolve(&config, WorkerType::Aggregated, "aggregated")
            .unwrap();
        factory(
            &SelectionConfig::from(&config),
            WorkerType::Aggregated,
            embedded_partition_key(Some("model")).as_ref(),
        );
        assert_eq!(built.load(Ordering::SeqCst), 1);
    }

    #[test]
    fn each_use_of_the_kv_path_keeps_a_picker_only_role_on_it() {
        let reason =
            |role: &RoleInputs<'_>, inputs| kv_path_reason(role, role.config, inputs).unwrap();
        let none = WorkerInputs::NONE;
        let plain = KvRouterConfig::default();
        assert_eq!(reason(&role(WorkerType::Decode, &plain), none), None);

        let (_file, classes) =
            policy_file("policy_classes:\n  - name: standard\n    quantum: 100\n");
        let (_file, other_model) = policy_file(
            "models:\n  other:\n    policy_classes:\n      - name: standard\n        quantum: 100\n",
        );
        let (_file, classifier) = policy_file("request_classifier:\n  type: some_classifier\n");
        let (_file, hints) = policy_file(
            "worker_selection:\n  decode:\n    type: round-robin\n    hints:\n      - type: some_hint\n  prefill:\n    type: round-robin\n    hints: []\n",
        );
        let with = |change: fn(&mut KvRouterConfig)| {
            let mut config = KvRouterConfig::default();
            change(&mut config);
            config
        };
        let queue_threshold = with(|config| config.router_queue_threshold = Some(0.5));
        let conditional_disagg = with(|config| config.conditional_disagg_enabled = true);
        let replica_sync = with(|config| config.router_replica_sync = true);
        let prefill_load_model =
            with(|config| config.router_prefill_load_model = RouterPrefillLoadModel::Ais);

        for (config, expected) in [
            (&classes, KvPathReason::PolicyClasses),
            (&queue_threshold, KvPathReason::PolicyClasses),
            (&classifier, KvPathReason::RequestClassifier),
            (&hints, KvPathReason::Hints),
            (&conditional_disagg, KvPathReason::ConditionalDisagg),
            (&prefill_load_model, KvPathReason::PrefillLoadModel),
        ] {
            assert_eq!(
                reason(&role(WorkerType::Decode, config), none),
                Some(expected)
            );
        }
        // What the host tells the resolver: a classifier it installs, LoRA serving, a caller
        // that reserves workers, a busy threshold, workers with taints, and, for a prefill
        // hop whose workers advertise their own configuration, the decode set's conditional
        // disaggregation.
        let decode = || role(WorkerType::Decode, &plain);
        let prefill = || role(WorkerType::Prefill, &plain);
        let host_inputs = [
            (
                RoleInputs {
                    has_request_classifier: true,
                    ..decode()
                },
                KvPathReason::RequestClassifier,
            ),
            (
                RoleInputs {
                    lora_enabled: true,
                    ..decode()
                },
                KvPathReason::Lora,
            ),
            (
                RoleInputs {
                    reserves_workers: true,
                    ..prefill()
                },
                KvPathReason::Reservations,
            ),
            (
                RoleInputs {
                    busy_thresholds: true,
                    ..decode()
                },
                KvPathReason::BusyThresholds,
            ),
            (
                RoleInputs {
                    workers_have_taints: true,
                    ..decode()
                },
                KvPathReason::WorkerTaints,
            ),
            (
                RoleInputs {
                    fronts_conditional_disagg: true,
                    ..prefill()
                },
                KvPathReason::ConditionalDisagg,
            ),
        ];
        assert_eq!(
            host_inputs
                .iter()
                .map(|(role, _)| reason(role, none))
                .collect::<Vec<_>>(),
            host_inputs
                .iter()
                .map(|(_, expected)| Some(*expected))
                .collect::<Vec<_>>()
        );

        // Each of these is the role's own. Another model's classes, a role that lists no
        // hints, and replica sync under a picker that counts no requests leave it hosted.
        assert_eq!(reason(&role(WorkerType::Decode, &other_model), none), None);
        assert_eq!(reason(&role(WorkerType::Prefill, &hints), none), None);
        assert_eq!(reason(&role(WorkerType::Decode, &replica_sync), none), None);
        assert_eq!(
            reason(
                &role(WorkerType::Decode, &replica_sync),
                WorkerInputs::ACTIVE_REQUESTS
            ),
            Some(KvPathReason::ReplicaSync)
        );
    }

    #[test]
    fn a_policy_type_runs_hosted_when_it_is_only_a_picker_and_on_the_kv_path_otherwise() {
        let (registry, built) = registry();
        let config = KvRouterConfig::default();

        // A picker-only type is named by its type, not by the mode its transport opens in,
        // and both of its pickers are the type's own.
        let Ok(RolePolicy::Ready(policy)) = policy_type_policy(&registry, HOSTED_TYPE, &config)
        else {
            panic!("{HOSTED_TYPE} must run hosted");
        };
        assert_eq!(policy.transport_mode(), RouterMode::RoundRobin);
        assert_eq!(policy.name(), HOSTED_TYPE);
        let RoutingPolicy::Hosted(hosted) = &policy else {
            panic!("{HOSTED_TYPE} must be a hosted policy");
        };
        assert_eq!(hosted.label(), HOSTED_TYPE);
        assert_eq!(pick_of_three(hosted.picker()), 3);
        assert_eq!(built.load(Ordering::SeqCst), 1);
        // LoRA filtering picks with a second picker.
        assert_eq!(pick_of_three(&hosted.fresh_picker().unwrap()), 3);
        assert_eq!(built.load(Ordering::SeqCst), 2);

        // A picker that reads active requests needs the transport's occupancy state.
        // `least-loaded` is the one shipped; the frontend passes it as a router mode.
        let Ok(RolePolicy::Ready(policy)) = policy_type_policy(&registry, "least-loaded", &config)
        else {
            panic!("least-loaded must run hosted");
        };
        assert_eq!(policy.transport_mode(), RouterMode::LeastLoaded);

        // Any other type is built once for the model's partition, and the KV router that
        // serves that partition gets the same instance.
        built.store(0, Ordering::SeqCst);
        let Ok(RolePolicy::Kv { selection, .. }) = policy_type_policy(&registry, KV_TYPE, &config)
        else {
            panic!("{KV_TYPE} must route on the KV path");
        };
        let factory = selection
            .resolve(&config, WorkerType::Decode, "decode")
            .unwrap();
        factory(
            &SelectionConfig::from(&config),
            WorkerType::Decode,
            embedded_partition_key(Some("model")).as_ref(),
        );
        assert_eq!(built.load(Ordering::SeqCst), 1);

        // An indexer needs a policy that reads the cache. It is an error, not dropped.
        let config = KvRouterConfig {
            serve_indexer: true,
            ..Default::default()
        };
        let Err(error) = policy_type_policy(&registry, HOSTED_TYPE, &config) else {
            panic!("serve_indexer must not be dropped");
        };
        assert!(error.to_string().contains("serve_indexer"), "{error}");
    }

    #[test]
    fn a_policy_type_runs_the_hints_registered_for_it() {
        let (registry, _) = registry();
        let hints = |policy_type: &str| {
            let Ok(RolePolicy::Kv {
                selection: SelectionPolicySource::Prepared(prepared),
                ..
            }) = policy_type_policy(&registry, policy_type, &KvRouterConfig::default())
            else {
                panic!("{policy_type} must route on the KV path");
            };
            prepared
                .hint_policies()
                .names(WorkerType::Decode)
                .into_iter()
                .map(str::to_owned)
                .collect::<Vec<_>>()
        };
        // `kv_fetch` is on by default under `kvcr` only, and no policy file names the type.
        assert_eq!(hints("kvcr"), ["kv_fetch"]);
        assert!(hints("dynamo-two-tier-cost-fn").is_empty());
    }

    #[test]
    fn a_policy_type_mode_is_checked_at_startup() {
        let named = |policy_type: &str| {
            RouterConfig::new(
                RouterModeSetting::PolicyType(policy_type.to_owned()),
                KvRouterConfig::default(),
            )
        };
        assert!(check_policy_type_mode(&named("round-robin")).is_ok());

        // A mistyped router mode reads as a policy type, so the error lists both.
        let error = check_policy_type_mode(&named("KV"))
            .unwrap_err()
            .to_string();
        assert!(
            error.contains("device-aware-weighted") && error.contains("round-robin)"),
            "{error}"
        );

        let mut with_file = named("round-robin");
        with_file.kv_router_config.router_policy_config = Some("policy.yaml".to_owned());
        let error = check_policy_type_mode(&with_file).unwrap_err();
        assert!(error.to_string().contains("mutually exclusive"), "{error}");

        let mut with_override = named("round-robin");
        with_override.kv_router_config.router_decode_policy = Some("random".to_owned());
        let error = check_policy_type_mode(&with_override).unwrap_err();
        assert!(error.to_string().contains("is selected too"), "{error}");
    }

    #[test]
    fn worker_router_config_preserves_frontend_policy_selections() {
        let mut frontend = RouterConfig {
            session_affinity_mode: SessionAffinityMode::Soft,
            ..Default::default()
        };
        frontend.kv_router_config.router_prefill_policy = Some("frontend-prefill".to_string());
        frontend.kv_router_config.router_decode_policy = Some("frontend-decode".to_string());

        let mut worker = RouterConfig::default();
        worker.kv_router_config.router_temperature = 0.75;
        worker.kv_router_config.router_policy_config = Some("worker-policy.yaml".to_string());

        let effective = serving_set_routing(Some(&worker), &frontend);
        assert_eq!(effective.kv_router_config.router_temperature, 0.75);
        assert_eq!(
            effective.kv_router_config.router_policy_config.as_deref(),
            Some("worker-policy.yaml")
        );
        assert_eq!(
            effective.kv_router_config.router_prefill_policy.as_deref(),
            Some("frontend-prefill")
        );
        assert_eq!(
            effective.kv_router_config.router_decode_policy.as_deref(),
            Some("frontend-decode")
        );
        assert_eq!(effective.session_affinity_mode, SessionAffinityMode::Soft);
        assert!(worker.kv_router_config.router_prefill_policy.is_none());
        assert!(worker.kv_router_config.router_decode_policy.is_none());
    }

    #[test]
    fn silent_prefill_pool_inherits_the_decode_set() {
        let mut decode = decode(RouterMode::RoundRobin, None);
        decode.kv_router_config = Some(KvRouterConfig {
            router_temperature: 0.5,
            router_track_active_blocks: true,
            ..Default::default()
        });

        let resolved = prefill_routing(None, decode).unwrap();

        assert_eq!(resolved.router_mode, RouterMode::RoundRobin);
        let kv_router_config = resolved.kv_router_config.expect("decode KV config");
        assert_eq!(kv_router_config.router_temperature, 0.5);
        assert!(!kv_router_config.router_track_active_blocks);
        assert!(!resolved.is_advertised);
    }

    #[test]
    fn session_affinity_ttl_is_taken_whole_or_not_at_all() {
        // An advertisement replaces the decode set's configuration rather than merging, so a
        // card that advertises without a TTL means "no affinity", not "inherit the
        // frontend's". A card advertising nothing means inherit.
        let decode_ttl = Some(Duration::from_secs(300));
        let advertised = RouterConfig::new(RouterMode::KV, KvRouterConfig::default());
        assert_eq!(
            prefill_routing(Some(advertised), decode(RouterMode::KV, decode_ttl))
                .unwrap()
                .session_affinity_ttl,
            None,
        );
        assert_eq!(
            prefill_routing(None, decode(RouterMode::KV, decode_ttl))
                .unwrap()
                .session_affinity_ttl,
            decode_ttl,
        );
    }

    #[test]
    fn advertised_prefill_kv_config_never_tracks_active_blocks() {
        let mut advertised = RouterConfig::new(RouterMode::KV, KvRouterConfig::default());
        advertised.kv_router_config.router_track_active_blocks = true;
        advertised.kv_router_config.router_temperature = 0.5;

        let resolved =
            prefill_routing(Some(advertised), decode(RouterMode::RoundRobin, None)).unwrap();

        assert_eq!(resolved.router_mode, RouterMode::KV);
        let kv_router_config = resolved.kv_router_config.expect("advertised KV config");
        assert!(!kv_router_config.router_track_active_blocks);
        assert_eq!(kv_router_config.router_temperature, 0.5);
    }

    #[test]
    fn prefill_hop_never_tracks_output_blocks() {
        let by_flag = KvRouterConfig {
            router_track_output_blocks: true,
            ..Default::default()
        };
        let policy_file = tempfile::NamedTempFile::new().unwrap();
        std::fs::write(
            policy_file.path(),
            "router:\n  router_track_output_blocks: true\n",
        )
        .unwrap();
        let mut by_policy_file = KvRouterConfig {
            router_policy_config: Some(policy_file.path().display().to_string()),
            ..Default::default()
        };
        by_policy_file.apply_policy_config().unwrap();
        // A silent pool gets the frontend's config with active blocks already off.
        let handed_down = |frontend: &KvRouterConfig| KvRouterConfig {
            router_track_active_blocks: false,
            ..frontend.clone()
        };
        for (case, advertised, inherited) in [
            ("inherited, set by flag", None, Some(handed_down(&by_flag))),
            (
                "inherited, set in the policy file",
                None,
                Some(handed_down(&by_policy_file)),
            ),
            (
                "advertised",
                Some(RouterConfig::new(RouterMode::KV, by_flag)),
                None,
            ),
        ] {
            let mut decode = decode(RouterMode::KV, None);
            decode.kv_router_config = inherited;
            let mut resolved = prefill_routing(advertised, decode)
                .unwrap()
                .kv_router_config
                .expect("KV config");
            // The KV router applies the policy file again, then validates. An
            // invalid config keeps the hop from ever activating.
            resolved.apply_policy_config().unwrap();
            assert_eq!(resolved.validate(), Ok(()), "{case}");
            assert!(!resolved.router_track_active_blocks, "{case}");
            assert!(!resolved.router_track_output_blocks, "{case}");
        }
    }
}
