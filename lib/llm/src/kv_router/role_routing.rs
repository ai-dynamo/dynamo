// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! The routing settings each worker role runs with.
//!
//! A frontend routes two kinds of worker pool: serving sets (aggregated or decode), and the
//! prefill pool in front of a decode set. A role's settings come from the frontend's
//! [`RouterConfig`], from the frontend's policy file, and from the `router_config` its workers
//! advertise on their model card. The rules that combine them live here.
//!
//! Advertising a `router_config` is deprecated. A frontend with a policy file ignores it (see
//! [`advertised_routing`]), so the `advertised` inputs below are set on a frontend without one,
//! and for text-input workers.

use std::borrow::Cow;
use std::time::Duration;

use dynamo_kv_router::config::KvRouterConfig;
use dynamo_kv_router::plugins::worker_selection::{SessionAffinitySetting, SessionAffinitySpec};
use dynamo_kv_router::scheduling::RouterPolicyConfigError;
use dynamo_runtime::pipeline::RouterMode;

use crate::discovery::LoadThresholdConfig;
use crate::entrypoint::RouterConfig;
use crate::model_card::ModelDeploymentCard;
use crate::model_type::ModelInput;
use crate::session_affinity::SessionAffinityMode;
use crate::worker_type::WorkerType;

/// What a host does with the `router_config` a model card advertises.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum AdvertisedRouting {
    /// The host routes the card's role from its own configuration.
    Ignored,
    /// The card routes its worker set. `replaceable` is false while a policy file cannot
    /// express what the card asks for.
    Honored { replaceable: bool },
}

/// What `host` does with the `router_config` on `card`; `None` when the card advertises nothing.
///
/// A host with a policy file routes every role from its own configuration and ignores the
/// card. The Python frontend accepts a file only in KV mode; an embedder that sets one beside
/// another mode ignores cards as well.
///
/// A text-input card is still honored there. A policy file routes on the KV path, which needs
/// frontend tokenization, so until a file can pick the path per role the card's mode is the
/// only way such a worker set is served.
pub(crate) fn advertised_routing(
    card: &ModelDeploymentCard,
    host: Option<&KvRouterConfig>,
) -> Option<AdvertisedRouting> {
    let advertised = card.router_config.as_ref()?;
    let text_input = card.model_input == ModelInput::Text;
    if !text_input && host.is_some_and(|host| host.router_policy_config.is_some()) {
        return Some(AdvertisedRouting::Ignored);
    }
    // A policy file cannot select these two modes yet.
    let replaceable = !text_input
        && !matches!(
            advertised.router_mode,
            RouterMode::Direct | RouterMode::DeviceAwareWeighted
        );
    Some(AdvertisedRouting::Honored { replaceable })
}

/// Warn that a worker set's cards advertise a `router_config`, and say what the host does with
/// it. Callers log this once per worker set, not per card.
pub(crate) fn warn_advertised_routing(
    model_name: &str,
    worker_set: &str,
    advertised: AdvertisedRouting,
) {
    // The honored wording holds for a worker that admission then rejects, too.
    let consequence = match advertised {
        AdvertisedRouting::Ignored => {
            "is ignored because this router has a policy file (--router-policy-config): \
             configure the role under worker_selection there. Router flags on workers are \
             deprecated and will be removed in v1.9"
        }
        AdvertisedRouting::Honored { replaceable: true } => {
            "is deprecated and will be removed in v1.9. This router still routes a worker set \
             by what its admitted workers advertise. Remove the router flags from the workers \
             and configure the role under worker_selection in the frontend's \
             --router-policy-config"
        }
        AdvertisedRouting::Honored { replaceable: false } => {
            "is deprecated and will be removed in v1.9. This router still routes a worker set \
             by what its admitted workers advertise. A policy file cannot replace this \
             advertisement yet (router mode direct or device-aware-weighted, or text-input \
             workers): until it can, keep the worker flags, or set --router-mode on a frontend \
             without a policy file"
        }
    };
    tracing::warn!(
        model_name,
        %worker_set,
        "Worker model cards advertise router_config, which {consequence}"
    );
}

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

/// Where a role's session affinity was configured.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum SessionAffinitySource {
    /// The role's `session_affinity` in the frontend's policy file.
    PolicyFile,
    /// The `router_config` the role's workers advertise.
    WorkerCard,
    /// The frontend's `--router-session-affinity-*` flags.
    FrontendFlags,
    /// The decode set's card or flags, inherited by a prefill pool that configures nothing.
    DecodeSet,
}

/// A role's session affinity; `spec` is `None` when the role has none.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct RoleSessionAffinity {
    pub(crate) spec: Option<SessionAffinitySpec>,
    pub(crate) source: SessionAffinitySource,
}

impl RoleSessionAffinity {
    fn from_policy_file(setting: SessionAffinitySetting) -> Self {
        Self {
            spec: match setting {
                SessionAffinitySetting::Enabled(spec) => Some(spec),
                SessionAffinitySetting::Disabled => None,
            },
            source: SessionAffinitySource::PolicyFile,
        }
    }

    fn from_ttl(
        ttl: Option<Duration>,
        mode: SessionAffinityMode,
        source: SessionAffinitySource,
    ) -> Self {
        Self {
            spec: ttl.map(|ttl| SessionAffinitySpec {
                ttl,
                mode,
                max_entries: None,
            }),
            source,
        }
    }
}

/// A role's `session_affinity` in the frontend's policy file, if the file sets it.
///
/// Pass the frontend's own configuration. A worker set that advertises carries its workers'
/// policy-file path, which names a file this process does not read.
pub(crate) fn policy_file_session_affinity(
    frontend: &RouterConfig,
    worker_type: WorkerType,
) -> Result<Option<SessionAffinitySetting>, RouterPolicyConfigError> {
    Ok(frontend
        .kv_router_config
        .worker_selection_config()?
        .and_then(|config| config.session_affinity(worker_type)))
}

/// Resolve the session affinity of a serving worker set (aggregated or decode).
///
/// The role's entry in the frontend's policy file wins, `false` included. Without one, the
/// set's advertised TTL is taken whole (none advertised means no affinity) with the frontend's
/// mode, and a set that advertises nothing uses the frontend's flags.
pub(crate) fn serving_set_session_affinity(
    advertised: Option<&RouterConfig>,
    frontend: &RouterConfig,
    worker_type: WorkerType,
) -> Result<RoleSessionAffinity, RouterPolicyConfigError> {
    if let Some(setting) = policy_file_session_affinity(frontend, worker_type)? {
        return Ok(RoleSessionAffinity::from_policy_file(setting));
    }
    let (ttl_secs, source) = match advertised {
        Some(advertised) => (
            advertised.session_affinity_ttl_secs,
            SessionAffinitySource::WorkerCard,
        ),
        None => (
            frontend.session_affinity_ttl_secs,
            SessionAffinitySource::FrontendFlags,
        ),
    };
    Ok(RoleSessionAffinity::from_ttl(
        ttl_secs.map(Duration::from_secs),
        frontend.session_affinity_mode,
        source,
    ))
}

/// The session-affinity TTL in seconds and the mode that a prefill pool inherits from the
/// decode set it fronts, when the pool configures nothing itself.
///
/// `decode_set` is the set's [`serving_set_routing`] result, which holds its card or flag
/// value. The set's resolved [`RoleSessionAffinity`] must not be used here: it can come from
/// the policy file's `decode` entry, which applies to the decode role alone.
pub(crate) fn inherited_prefill_session_affinity(
    decode_set: &RouterConfig,
) -> (Option<u64>, SessionAffinityMode) {
    (
        decode_set.session_affinity_ttl_secs,
        decode_set.session_affinity_mode,
    )
}

/// Where a role's busy thresholds were configured.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum BusyThresholdSource {
    /// The role's `busy_threshold` in the frontend's policy file.
    PolicyFile,
    /// The `router_config` the serving set's workers advertise.
    WorkerCard,
    /// The frontend's `--active-*-threshold` flags.
    FrontendFlags,
    /// The decode set's card or flags, inherited by its prefill pool.
    DecodeSet,
}

/// The busy thresholds a role's workers start with.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct RoleBusyThresholds {
    pub(crate) thresholds: LoadThresholdConfig,
    pub(crate) source: BusyThresholdSource,
}

/// Resolve the busy thresholds a role's workers start with.
///
/// The role's `busy_threshold` in the frontend's policy file is taken whole: a limit it omits
/// is not checked, whatever a card or a flag says. A role without the key keeps `inherited`,
/// the thresholds of the serving set's [`serving_set_routing`] result, and reports
/// `inherited_from`. For the prefill pool that is the decode set's result, never the file's
/// `decode` entry: a role's entry in the file applies to that role alone. A card's thresholds
/// are part of its `router_config`, so they reach `inherited` only where the card is honored
/// (see [`advertised_routing`]).
///
/// Pass the frontend's own configuration, as for [`policy_file_session_affinity`].
pub(crate) fn role_busy_thresholds(
    frontend: &RouterConfig,
    worker_type: WorkerType,
    inherited: &LoadThresholdConfig,
    inherited_from: BusyThresholdSource,
) -> Result<RoleBusyThresholds, RouterPolicyConfigError> {
    let policy_file = frontend
        .kv_router_config
        .worker_selection_config()?
        .and_then(|config| config.busy_threshold(worker_type));
    Ok(match policy_file {
        Some(spec) => RoleBusyThresholds {
            thresholds: LoadThresholdConfig {
                active_decode_blocks_threshold: spec.active_decode_blocks,
                active_prefill_tokens_threshold: None,
                active_prefill_tokens_threshold_frac: spec.active_prefill_tokens_frac,
            },
            source: BusyThresholdSource::PolicyFile,
        },
        None => RoleBusyThresholds {
            thresholds: inherited.clone(),
            source: inherited_from,
        },
    })
}

/// What the prefill hop inherits from the decode set it fronts.
#[derive(Debug)]
pub(crate) struct DecodeRouting {
    pub(crate) router_mode: RouterMode,
    /// The decode set's KV configuration as handed to the prefill router. `None` leaves an
    /// inheriting prefill KV router on its defaults.
    pub(crate) kv_router_config: Option<KvRouterConfig>,
    /// The decode set's card or flag TTL and mode. Never its policy-file `session_affinity`:
    /// a role's entry in the file applies to that role alone.
    pub(crate) session_affinity_ttl: Option<Duration>,
    pub(crate) session_affinity_mode: SessionAffinityMode,
}

/// The routing settings the prefill hop runs with.
#[derive(Debug)]
pub(crate) struct PrefillRouting {
    pub(crate) router_mode: RouterMode,
    pub(crate) kv_router_config: Option<KvRouterConfig>,
    pub(crate) session_affinity: RoleSessionAffinity,
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
/// `policy_file` is the `prefill` entry's `session_affinity` in the frontend's policy file.
/// When the file sets it, it replaces the session affinity of either case above.
///
/// Whichever KV configuration the hop gets, it tracks neither active nor output blocks: prefill
/// routing is prompt-side, and crediting decode blocks here would double-count load. The policy
/// file's `router` settings are finalized first, because the KV router applies them again and
/// would turn both back on. An unreadable policy file is the error.
pub(crate) fn prefill_routing(
    advertised: Option<RouterConfig>,
    decode: DecodeRouting,
    policy_file: Option<SessionAffinitySetting>,
) -> Result<PrefillRouting, String> {
    let policy_file = policy_file.map(RoleSessionAffinity::from_policy_file);
    let Some(advertised) = advertised else {
        return Ok(PrefillRouting {
            router_mode: decode.router_mode,
            kv_router_config: decode
                .kv_router_config
                .map(without_block_tracking)
                .transpose()?,
            session_affinity: policy_file.unwrap_or_else(|| {
                RoleSessionAffinity::from_ttl(
                    decode.session_affinity_ttl,
                    decode.session_affinity_mode,
                    SessionAffinitySource::DecodeSet,
                )
            }),
            is_advertised: false,
        });
    };

    Ok(PrefillRouting {
        router_mode: advertised.router_mode,
        kv_router_config: Some(without_block_tracking(advertised.kv_router_config)?),
        session_affinity: policy_file.unwrap_or_else(|| {
            RoleSessionAffinity::from_ttl(
                advertised
                    .session_affinity_ttl_secs
                    .map(Duration::from_secs),
                decode.session_affinity_mode,
                SessionAffinitySource::WorkerCard,
            )
        }),
        is_advertised: true,
    })
}

fn without_block_tracking(mut config: KvRouterConfig) -> Result<KvRouterConfig, String> {
    config.finalize_policy_config()?;
    config.router_track_active_blocks = false;
    config.router_track_output_blocks = false;
    Ok(config)
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
    session_affinity: &RoleSessionAffinity,
    busy_thresholds: &RoleBusyThresholds,
) {
    tracing::info!(
        model_name,
        group = %group,
        ?worker_type,
        router_mode = ?routing.router_mode,
        advertised,
        session_affinity = ?session_affinity.spec,
        session_affinity_source = ?session_affinity.source,
        busy_thresholds = ?busy_thresholds.thresholds,
        busy_thresholds_source = ?busy_thresholds.source,
        kv_tuning = ?routing
            .router_mode
            .is_kv_routing()
            .then_some(&routing.kv_router_config),
        "Activating router"
    );
}

#[cfg(test)]
mod tests {
    use std::num::NonZeroUsize;

    use dynamo_kv_router::scheduling::RouterPolicyConfig;

    use super::*;

    /// A frontend whose policy file holds `yaml`.
    fn frontend_with_policy_file(yaml: &str) -> RouterConfig {
        let frontend = RouterConfig {
            kv_router_config: KvRouterConfig {
                router_policy_config: Some("frontend-policy.yaml".to_string()),
                ..Default::default()
            },
            ..Default::default()
        };
        frontend
            .kv_router_config
            .policy_config_cache
            .set(RouterPolicyConfig::from_yaml(yaml).unwrap())
            .expect("empty cache");
        frontend
    }

    fn spec(ttl_secs: u64, mode: SessionAffinityMode) -> SessionAffinitySpec {
        SessionAffinitySpec {
            ttl: Duration::from_secs(ttl_secs),
            mode,
            max_entries: None,
        }
    }

    fn decode(router_mode: RouterMode, session_affinity_ttl: Option<Duration>) -> DecodeRouting {
        DecodeRouting {
            router_mode,
            kv_router_config: None,
            session_affinity_ttl,
            session_affinity_mode: SessionAffinityMode::Hard,
        }
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

        let resolved = prefill_routing(None, decode, None).unwrap();

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
            prefill_routing(Some(advertised), decode(RouterMode::KV, decode_ttl), None)
                .unwrap()
                .session_affinity,
            RoleSessionAffinity {
                spec: None,
                source: SessionAffinitySource::WorkerCard,
            },
        );
        assert_eq!(
            prefill_routing(None, decode(RouterMode::KV, decode_ttl), None)
                .unwrap()
                .session_affinity,
            RoleSessionAffinity {
                spec: Some(spec(300, SessionAffinityMode::Hard)),
                source: SessionAffinitySource::DecodeSet,
            },
        );
    }

    #[test]
    fn serving_set_affinity_is_the_policy_file_then_the_card_then_the_flags() {
        let flags = |mut frontend: RouterConfig| {
            frontend.session_affinity_ttl_secs = Some(300);
            frontend.session_affinity_mode = SessionAffinityMode::Soft;
            frontend
        };
        // A card carries its workers' policy-file path. Reading it would fail: no such file.
        let card = RouterConfig {
            session_affinity_ttl_secs: Some(900),
            kv_router_config: KvRouterConfig {
                router_policy_config: Some("worker-policy.yaml".to_string()),
                ..Default::default()
            },
            ..Default::default()
        };
        let resolve = |card: Option<&RouterConfig>, frontend: &RouterConfig, worker_type| {
            serving_set_session_affinity(card, frontend, worker_type).unwrap()
        };

        let frontend = flags(frontend_with_policy_file(
            r#"
worker_selection:
  decode: {type: alpha, session_affinity: {ttl_secs: 60}}
  aggregated: {type: alpha, session_affinity: false}
"#,
        ));
        // The file's entry is taken whole: its mode, not the flag's.
        assert_eq!(
            resolve(Some(&card), &frontend, WorkerType::Decode),
            RoleSessionAffinity {
                spec: Some(spec(60, SessionAffinityMode::Hard)),
                source: SessionAffinitySource::PolicyFile,
            }
        );
        // `false` turns the role off although the flag and the card set a TTL.
        assert_eq!(
            resolve(Some(&card), &frontend, WorkerType::Aggregated),
            RoleSessionAffinity {
                spec: None,
                source: SessionAffinitySource::PolicyFile,
            }
        );

        // A role the file says nothing about keeps the card and flag rules.
        let frontend = flags(frontend_with_policy_file(
            "worker_selection:\n  prefill: {type: alpha, session_affinity: true}\n",
        ));
        assert_eq!(
            resolve(Some(&card), &frontend, WorkerType::Decode),
            RoleSessionAffinity {
                spec: Some(spec(900, SessionAffinityMode::Soft)),
                source: SessionAffinitySource::WorkerCard,
            }
        );
        assert_eq!(
            resolve(
                Some(&RouterConfig::default()),
                &frontend,
                WorkerType::Decode
            )
            .spec,
            None
        );
        assert_eq!(
            resolve(None, &frontend, WorkerType::Decode),
            RoleSessionAffinity {
                spec: Some(spec(300, SessionAffinityMode::Soft)),
                source: SessionAffinitySource::FrontendFlags,
            }
        );
    }

    #[test]
    fn prefill_affinity_is_the_policy_file_before_the_card_or_the_decode_set() {
        let policy_file = SessionAffinitySetting::Enabled(SessionAffinitySpec {
            ttl: Duration::from_secs(60),
            mode: SessionAffinityMode::Soft,
            max_entries: NonZeroUsize::new(8),
        });
        let from_file = RoleSessionAffinity::from_policy_file(policy_file);
        let decode_ttl = Some(Duration::from_secs(300));
        let mut advertised = RouterConfig::new(RouterMode::KV, KvRouterConfig::default());
        advertised.session_affinity_ttl_secs = Some(900);

        assert_eq!(
            prefill_routing(
                Some(advertised),
                decode(RouterMode::KV, decode_ttl),
                Some(policy_file)
            )
            .unwrap()
            .session_affinity,
            from_file,
        );
        assert_eq!(
            prefill_routing(None, decode(RouterMode::KV, decode_ttl), Some(policy_file))
                .unwrap()
                .session_affinity,
            from_file,
        );
        assert_eq!(
            prefill_routing(
                None,
                decode(RouterMode::KV, decode_ttl),
                Some(SessionAffinitySetting::Disabled)
            )
            .unwrap()
            .session_affinity
            .spec,
            None,
        );
    }

    /// The file's `decode.session_affinity` is the decode role's alone. A prefill pool that
    /// configures nothing inherits the decode set's card or flag TTL, which this file leaves
    /// unset.
    #[test]
    fn silent_prefill_pool_does_not_inherit_the_policy_files_decode_affinity() {
        let frontend = frontend_with_policy_file(
            "worker_selection:\n  decode: {type: alpha, session_affinity: true}\n",
        );
        let decode_set = serving_set_routing(None, &frontend);
        assert!(
            serving_set_session_affinity(None, &frontend, WorkerType::Decode)
                .unwrap()
                .spec
                .is_some()
        );

        let (inherited_ttl_secs, inherited_mode) = inherited_prefill_session_affinity(&decode_set);
        assert_eq!(inherited_ttl_secs, None);

        let resolved = prefill_routing(
            None,
            DecodeRouting {
                router_mode: decode_set.router_mode,
                kv_router_config: None,
                session_affinity_ttl: inherited_ttl_secs.map(Duration::from_secs),
                session_affinity_mode: inherited_mode,
            },
            policy_file_session_affinity(&frontend, WorkerType::Prefill).unwrap(),
        )
        .unwrap();

        assert_eq!(
            resolved.session_affinity,
            RoleSessionAffinity {
                spec: None,
                source: SessionAffinitySource::DecodeSet,
            }
        );
    }

    #[test]
    fn role_busy_thresholds_are_the_policy_file_taken_whole_else_the_card_or_flags() {
        let card_or_flags = LoadThresholdConfig {
            active_decode_blocks_threshold: Some(0.9),
            active_prefill_tokens_threshold: Some(1000),
            active_prefill_tokens_threshold_frac: Some(2.0),
        };
        let frontend = frontend_with_policy_file(
            r#"
worker_selection:
  decode: {type: alpha, busy_threshold: {active_decode_blocks: 0.5}}
  aggregated: {type: alpha, busy_threshold: {}}
  prefill: alpha
"#,
        );
        let resolve = |frontend: &RouterConfig, worker_type| {
            let resolved = role_busy_thresholds(
                frontend,
                worker_type,
                &card_or_flags,
                BusyThresholdSource::WorkerCard,
            )
            .unwrap();
            (resolved.thresholds, resolved.source)
        };

        // Taken whole: the limits the entry omits are unset, not the card's or the flags'.
        assert_eq!(
            resolve(&frontend, WorkerType::Decode),
            (
                LoadThresholdConfig {
                    active_decode_blocks_threshold: Some(0.5),
                    ..Default::default()
                },
                BusyThresholdSource::PolicyFile
            )
        );
        assert_eq!(
            resolve(&frontend, WorkerType::Aggregated),
            (
                LoadThresholdConfig::default(),
                BusyThresholdSource::PolicyFile
            )
        );
        // The prefill pool has no key, so it does not pick up the file's decode entry.
        let inherited = (card_or_flags.clone(), BusyThresholdSource::WorkerCard);
        assert_eq!(resolve(&frontend, WorkerType::Prefill), inherited);
        assert_eq!(
            resolve(&RouterConfig::default(), WorkerType::Decode),
            inherited
        );
    }

    #[test]
    fn advertised_prefill_kv_config_never_tracks_active_blocks() {
        let mut advertised = RouterConfig::new(RouterMode::KV, KvRouterConfig::default());
        advertised.kv_router_config.router_track_active_blocks = true;
        advertised.kv_router_config.router_temperature = 0.5;

        let resolved =
            prefill_routing(Some(advertised), decode(RouterMode::RoundRobin, None), None).unwrap();

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
            let mut resolved = prefill_routing(advertised, decode, None)
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
