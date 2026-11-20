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

use dynamo_kv_router::config::KvRouterConfig;
use dynamo_router_policies::router_modes;
use dynamo_router_protocol::identity::RoutingPartitionRef;
use dynamo_runtime::pipeline::RouterMode;
use dynamo_worker_selection::host::{HostedWorkerPicker, SelectionConfig};

use super::routing_host::{HostedPolicy, RoutingPolicy};
use crate::entrypoint::RouterConfig;
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
    pub(crate) router_mode: RouterMode,
    /// The decode set's KV configuration as handed to the prefill router. `None` leaves an
    /// inheriting prefill KV router on its defaults.
    pub(crate) kv_router_config: Option<KvRouterConfig>,
    pub(crate) session_affinity_ttl: Option<Duration>,
    pub(crate) session_affinity_mode: SessionAffinityMode,
}

/// The routing settings the prefill hop runs with.
#[derive(Debug)]
pub(crate) struct PrefillRouting {
    pub(crate) router_mode: RouterMode,
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
/// An advertised KV configuration has `router_track_active_blocks` turned off: prefill routing
/// is prompt-side, and crediting decode blocks here would double-count load. The caller turns
/// it off in the configuration a silent prefill pool inherits.
pub(crate) fn prefill_routing(
    advertised: Option<RouterConfig>,
    decode: DecodeRouting,
) -> PrefillRouting {
    let Some(advertised) = advertised else {
        return PrefillRouting {
            router_mode: decode.router_mode,
            kv_router_config: decode.kv_router_config,
            session_affinity_ttl: decode.session_affinity_ttl,
            session_affinity_mode: decode.session_affinity_mode,
            is_advertised: false,
        };
    };

    let mut kv_router_config = advertised.kv_router_config;
    kv_router_config.router_track_active_blocks = false;
    PrefillRouting {
        router_mode: advertised.router_mode,
        kv_router_config: Some(kv_router_config),
        session_affinity_ttl: advertised
            .session_affinity_ttl_secs
            .map(Duration::from_secs),
        session_affinity_mode: decode.session_affinity_mode,
        is_advertised: true,
    }
}

/// What a role's router mode resolves to.
pub enum RolePolicy {
    /// KV routing. The role needs its own [`KvRouter`](super::KvRouter): the caller builds it
    /// and routes with [`RoutingPolicy::Kv`].
    Kv,
    /// A policy that needs no KV state.
    Ready(RoutingPolicy),
}

/// Resolve the policy a router mode selects for one role.
pub fn role_policy(mode: RouterMode) -> anyhow::Result<RolePolicy> {
    Ok(match mode {
        RouterMode::KV => RolePolicy::Kv,
        RouterMode::Direct => RolePolicy::Ready(RoutingPolicy::Direct),
        RouterMode::DeviceAwareWeighted => RolePolicy::Ready(RoutingPolicy::DeviceAwareWeighted),
        // `hosted_picker` returns the removed-mode error for `PowerOfTwoChoices`.
        RouterMode::RoundRobin
        | RouterMode::Random
        | RouterMode::LeastLoaded
        | RouterMode::PowerOfTwoChoices => {
            RolePolicy::Ready(RoutingPolicy::Hosted(HostedPolicy::for_mode(mode)?))
        }
    })
}

/// The policy of a mode that needs no [`KvRouter`](super::KvRouter).
#[cfg(test)]
pub(crate) fn non_kv_policy(mode: RouterMode) -> RoutingPolicy {
    match role_policy(mode).expect("mode must resolve to a policy") {
        RolePolicy::Ready(policy) => policy,
        RolePolicy::Kv => panic!("KV routing needs a KvRouter"),
    }
}

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
    let factory = crate::kv_router::plugins::router_plugin_registry().resolve_type(policy_type)?;
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
/// restating its tuning runs on defaults, and this line is what shows it.
pub(crate) fn log_serving_set_routing(
    model_name: &str,
    group: &str,
    worker_type: WorkerType,
    advertised: bool,
    routing: &RouterConfig,
) {
    tracing::info!(
        model_name,
        group = %group,
        ?worker_type,
        router_mode = ?routing.router_mode,
        advertised,
        session_affinity_ttl = ?routing.session_affinity_ttl_secs.map(Duration::from_secs),
        session_affinity_mode = ?routing.session_affinity_mode,
        busy_thresholds = ?routing.load_threshold_config,
        kv_tuning = ?routing
            .router_mode
            .is_kv_routing()
            .then_some(&routing.kv_router_config),
        "Activating router"
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    fn decode(router_mode: RouterMode, session_affinity_ttl: Option<Duration>) -> DecodeRouting {
        DecodeRouting {
            router_mode,
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
                    assert_eq!(Some(hosted.picker().inputs()), picker_inputs, "{mode:?}")
                }
                _ => panic!("{mode:?} resolved to the wrong policy"),
            }
        }

        assert!(matches!(role_policy(RouterMode::KV), Ok(RolePolicy::Kv)));
        let Err(error) = role_policy(RouterMode::PowerOfTwoChoices) else {
            panic!("the removed power-of-two mode must not resolve");
        };
        assert!(error.to_string().contains("was removed"), "{error}");
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
    fn silent_prefill_pool_inherits_the_decode_set_unchanged() {
        let mut decode = decode(RouterMode::RoundRobin, None);
        decode.kv_router_config = Some(KvRouterConfig {
            router_temperature: 0.5,
            router_track_active_blocks: true,
            ..Default::default()
        });

        let resolved = prefill_routing(None, decode);

        assert_eq!(resolved.router_mode, RouterMode::RoundRobin);
        let kv_router_config = resolved.kv_router_config.expect("decode KV config");
        assert_eq!(kv_router_config.router_temperature, 0.5);
        assert!(kv_router_config.router_track_active_blocks);
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
                .session_affinity_ttl,
            None,
        );
        assert_eq!(
            prefill_routing(None, decode(RouterMode::KV, decode_ttl)).session_affinity_ttl,
            decode_ttl,
        );
    }

    #[test]
    fn advertised_prefill_kv_config_never_tracks_active_blocks() {
        let mut advertised = RouterConfig::new(RouterMode::KV, KvRouterConfig::default());
        advertised.kv_router_config.router_track_active_blocks = true;
        advertised.kv_router_config.router_temperature = 0.5;

        let resolved = prefill_routing(Some(advertised), decode(RouterMode::RoundRobin, None));

        assert_eq!(resolved.router_mode, RouterMode::KV);
        let kv_router_config = resolved.kv_router_config.expect("advertised KV config");
        assert!(!kv_router_config.router_track_active_blocks);
        assert_eq!(kv_router_config.router_temperature, 0.5);
    }
}
