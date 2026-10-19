// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Common CLI flags every Rust backend needs.
//!
//! Flag names match the Python runtime's `WorkerConfig` knobs so operators see
//! the same CLI surface across Rust and Python backends. Engines extend this
//! with their own `clap` `Args` using `#[command(flatten)]`.

use std::path::PathBuf;

use clap::Args;
use dynamo_llm::discovery::LoadThresholdConfig;
use dynamo_llm::entrypoint::RouterConfig;
use dynamo_llm::kv_router::scheduling::config::{
    KvRouterConfig, try_kv_router_config_from_dynamo_env,
};
use dynamo_llm::session_affinity::MAX_SESSION_AFFINITY_TTL_SECS;
use dynamo_runtime::pipeline::RouterMode;

use crate::disagg::DisaggregationMode;
use crate::{BackendError, DynamoError, ErrorType};

/// CLI flags that every Rust backend needs. Flatten these into your engine's
/// `Args` struct:
///
/// ```ignore
/// #[derive(clap::Parser)]
/// struct EngineArgs {
///     #[clap(flatten)]
///     common: CommonArgs,
///
///     #[arg(long, default_value = "sample-model")]
///     model_name: String,
/// }
/// ```
#[derive(Args, Clone, Debug)]
pub struct CommonArgs {
    #[command(flatten)]
    pub runtime: crate::RuntimeConfig,

    /// Dynamo namespace for discovery routing. `DYN_NAMESPACE_WORKER_SUFFIX`
    /// is appended as `-{suffix}` unless it is empty or already present.
    #[arg(
        long,
        default_value = "dynamo",
        env = "DYN_NAMESPACE",
        value_parser = parse_worker_namespace
    )]
    pub namespace: String,

    /// Component name within the namespace.
    #[arg(long, default_value = "backend", env = "DYN_COMPONENT")]
    pub component: String,

    /// Endpoint name exposed by this worker.
    #[arg(long, default_value = "generate", env = "DYN_ENDPOINT")]
    pub endpoint: String,

    /// Comma-separated list of supported endpoint types
    /// (`chat`, `completions`, `embeddings`, ...).
    #[arg(long, default_value = "chat,completions", env = "DYN_ENDPOINT_TYPES")]
    pub endpoint_types: String,

    /// Optional path to a custom Jinja chat template, used instead of the
    /// template shipped with the model.
    #[arg(long, env = "DYN_CUSTOM_JINJA_TEMPLATE")]
    pub custom_jinja_template: Option<PathBuf>,

    /// Dynamo frontend tool-call parser name for this model.
    #[arg(long = "dyn-tool-call-parser", env = "DYN_TOOL_CALL_PARSER")]
    pub dyn_tool_call_parser: Option<String>,

    /// Dynamo frontend reasoning parser name for this model.
    #[arg(long = "dyn-reasoning-parser", env = "DYN_REASONING_PARSER")]
    pub dyn_reasoning_parser: Option<String>,

    /// Exclude tools from the chat template when tool_choice is none.
    #[arg(
        long = "exclude-tools-when-tool-choice-none",
        env = "DYN_EXCLUDE_TOOLS_WHEN_TOOL_CHOICE_NONE",
        default_value_t = true,
        action = clap::ArgAction::Set
    )]
    pub exclude_tools_when_tool_choice_none: bool,

    /// Disaggregation role: `agg` (default), `prefill`, `decode`, or `encode`.
    /// Prefill workers register with `ModelType::empty()` and
    /// `WorkerType::Prefill` regardless of `endpoint_types`; decode and encode
    /// workers do not advertise a local KV indexer. Encode workers register as
    /// `WorkerType::Encode` and are not exposed on the public chat/completions
    /// surface.
    #[arg(
        long,
        value_enum,
        default_value_t = DisaggregationMode::Aggregated,
        env = "DYN_DISAGGREGATION_MODE",
    )]
    pub disaggregation_mode: DisaggregationMode,

    /// Declare an upstream Encode peer in this worker's topology `needs`.
    /// Meaningful only on `--disaggregation-mode agg` and `prefill`.
    /// Setting it on `decode` or `encode` is rejected at startup.
    ///
    /// Scope: Rust backends consuming `CommonArgs` via clap. Python
    /// backends populate this from their own runtime config -- the Python
    /// shim does not read this env var.
    #[arg(long, default_value_t = false, env = "DYN_ROUTE_TO_ENCODER")]
    pub route_to_encoder: bool,

    /// Publish this worker's engine control/update routes on the RL request-plane endpoint.
    #[arg(long, default_value_t = false, env = "DYN_ENABLE_RL")]
    pub enable_rl: bool,

    /// Routing mode the frontend uses for this worker set: `round-robin`,
    /// `random`, `power-of-two`, `kv`, `direct`, `least-loaded`, or
    /// `device-aware-weighted`. Setting it advertises a router configuration
    /// on the model card that replaces the frontend's for this set, which is
    /// how prefill and decode sets route differently; the three flags below
    /// qualify it and are ignored without it. Same names and environment
    /// variables as the Python workers. Omit it to inherit the frontend's
    /// routing.
    #[arg(long, env = "DYN_ROUTER_MODE", value_parser = parse_router_mode)]
    pub router_mode: Option<RouterMode>,

    /// Session-affinity idle TTL in seconds for the advertised worker set.
    #[arg(
        long,
        env = "DYN_ROUTER_SESSION_AFFINITY_TTL_SECS",
        value_parser = clap::value_parser!(u64).range(1..=MAX_SESSION_AFFINITY_TTL_SECS)
    )]
    pub router_session_affinity_ttl_secs: Option<u64>,

    /// Busy-worker rejection for the advertised set: KV block utilization
    /// (0.0 to 1.0) above which a decode worker is skipped.
    #[arg(long, env = "DYN_ACTIVE_DECODE_BLOCKS_THRESHOLD")]
    pub active_decode_blocks_threshold: Option<f64>,

    /// Busy-worker rejection for the advertised set: active prefill tokens
    /// above which a prefill worker is skipped.
    #[arg(long, env = "DYN_ACTIVE_PREFILL_TOKENS_THRESHOLD")]
    pub active_prefill_tokens_threshold: Option<u64>,

    /// Busy-worker rejection for the advertised set: active prefill tokens as
    /// a fraction of `max_num_batched_tokens` above which a prefill worker is
    /// skipped.
    #[arg(long, env = "DYN_ACTIVE_PREFILL_TOKENS_THRESHOLD_FRAC")]
    pub active_prefill_tokens_threshold_frac: Option<f64>,
}

impl CommonArgs {
    /// The router configuration this worker advertises on its model card, or
    /// `None` without `--router-mode`. An advertised `kv` set takes its KV
    /// tuning from the `DYN_ROUTER_*` environment, the variables the Python
    /// worker flags default from; there are no per-setting flags here.
    pub fn router_config(&self) -> Result<Option<RouterConfig>, DynamoError> {
        let Some(router_mode) = self.router_mode else {
            return Ok(None);
        };
        let invalid = |message: String| {
            DynamoError::builder()
                .error_type(ErrorType::Backend(BackendError::InvalidArgument))
                .message(message)
                .build()
        };
        let load_threshold_config = LoadThresholdConfig {
            active_decode_blocks_threshold: self.active_decode_blocks_threshold,
            active_prefill_tokens_threshold: self.active_prefill_tokens_threshold,
            active_prefill_tokens_threshold_frac: self.active_prefill_tokens_threshold_frac,
        };
        load_threshold_config.validate().map_err(invalid)?;
        let kv_router_config = if router_mode.is_kv_routing() {
            try_kv_router_config_from_dynamo_env().map_err(invalid)?
        } else {
            KvRouterConfig::default()
        };
        let mut config = RouterConfig::new(router_mode, kv_router_config)
            .with_load_threshold_config(load_threshold_config);
        config.session_affinity_ttl_secs = self.router_session_affinity_ttl_secs;
        Ok(Some(config))
    }
}

/// The spellings the Python workers' `--router-mode` accepts.
fn parse_router_mode(raw: &str) -> Result<RouterMode, String> {
    Ok(match raw {
        "round-robin" => RouterMode::RoundRobin,
        "random" => RouterMode::Random,
        "power-of-two" => RouterMode::PowerOfTwoChoices,
        "kv" => RouterMode::KV,
        "direct" => RouterMode::Direct,
        "least-loaded" => RouterMode::LeastLoaded,
        "device-aware-weighted" => RouterMode::DeviceAwareWeighted,
        other => {
            return Err(format!(
                "unknown router mode {other:?}; expected one of round-robin, random, \
                 power-of-two, kv, direct, least-loaded, device-aware-weighted"
            ));
        }
    })
}

fn parse_worker_namespace(namespace: &str) -> Result<String, std::convert::Infallible> {
    let Ok(suffix) = std::env::var("DYN_NAMESPACE_WORKER_SUFFIX") else {
        return Ok(namespace.to_owned());
    };
    if suffix.is_empty() {
        return Ok(namespace.to_owned());
    }
    let suffix = format!("-{suffix}");
    if namespace.ends_with(&suffix) {
        return Ok(namespace.to_owned());
    }
    Ok(format!("{namespace}{suffix}"))
}

#[cfg(test)]
mod tests {
    use clap::Parser;
    use dynamo_runtime::pipeline::RouterMode;

    use super::{CommonArgs, RouterConfig};

    #[derive(Parser)]
    struct TestArgs {
        #[command(flatten)]
        common: CommonArgs,
    }

    /// Every advertisement flag reads a `DYN_*` variable, so the ambient
    /// environment must not leak into these tests.
    const ROUTER_ENV: [&str; 8] = [
        "DYN_ROUTER_MODE",
        "DYN_ROUTER_SESSION_AFFINITY_TTL_SECS",
        "DYN_ACTIVE_DECODE_BLOCKS_THRESHOLD",
        "DYN_ACTIVE_PREFILL_TOKENS_THRESHOLD",
        "DYN_ACTIVE_PREFILL_TOKENS_THRESHOLD_FRAC",
        "DYN_ROUTER_USE_KV_EVENTS",
        "DYN_USE_KV_EVENTS",
        "DYN_ROUTER_KV_OVERLAP_SCORE_CREDIT",
    ];

    fn with_router_env<R>(set: &[(&str, &str)], f: impl FnOnce() -> R) -> R {
        let vars = ROUTER_ENV.map(|key| {
            let value = set.iter().find(|(name, _)| *name == key).map(|(_, v)| *v);
            (key, value)
        });
        temp_env::with_vars(vars, f)
    }

    fn parse(argv: &[&str]) -> Result<TestArgs, clap::Error> {
        let mut full = vec!["test"];
        full.extend_from_slice(argv);
        TestArgs::try_parse_from(full)
    }

    fn router_config(argv: &[&str]) -> Option<RouterConfig> {
        parse(argv).unwrap().common.router_config().unwrap()
    }

    #[test]
    #[serial_test::serial]
    fn router_advertisement_follows_router_mode() {
        with_router_env(&[], || {
            assert!(router_config(&[]).is_none());
            // The qualifying flags alone advertise nothing.
            assert!(router_config(&["--router-session-affinity-ttl-secs", "300"]).is_none());

            let config = router_config(&[
                "--router-mode",
                "kv",
                "--router-session-affinity-ttl-secs",
                "300",
                "--active-prefill-tokens-threshold-frac",
                "0.8",
            ])
            .expect("advertised");
            assert_eq!(config.router_mode, RouterMode::KV);
            assert_eq!(config.session_affinity_ttl_secs, Some(300));
            assert_eq!(
                config
                    .load_threshold_config
                    .active_prefill_tokens_threshold_frac,
                Some(0.8)
            );
        });
        with_router_env(
            &[
                ("DYN_ROUTER_MODE", "kv"),
                ("DYN_ROUTER_USE_KV_EVENTS", "false"),
                ("DYN_ROUTER_KV_OVERLAP_SCORE_CREDIT", "2.5"),
            ],
            || {
                let config = router_config(&[]).expect("advertised from the environment");
                assert_eq!(config.router_mode, RouterMode::KV);
                assert!(!config.kv_router_config.use_kv_events);
                assert_eq!(config.kv_router_config.overlap_score_credit, 2.5);

                // Only KV routing reads the tuning.
                let config = router_config(&["--router-mode", "least-loaded"]).expect("advertised");
                assert!(config.kv_router_config.use_kv_events);
            },
        );
    }

    #[test]
    #[serial_test::serial]
    fn router_advertisement_rejects_invalid_values() {
        with_router_env(&[], || {
            for rejected in [
                &["--router-mode", "sticky"][..],
                &["--router-session-affinity-ttl-secs", "0"][..],
                &["--router-session-affinity-ttl-secs", "31536001"][..],
            ] {
                assert!(parse(rejected).is_err(), "{rejected:?} must not parse");
            }

            let args = parse(&[
                "--router-mode",
                "kv",
                "--active-decode-blocks-threshold",
                "1.5",
            ])
            .unwrap();
            let error = args.common.router_config().unwrap_err();
            assert!(
                error.to_string().contains("active_decode_blocks_threshold"),
                "{error}"
            );
        });
    }

    #[test]
    #[serial_test::serial]
    fn runtime_environment_values_remain_compatible() {
        temp_env::with_vars(
            [
                ("DYN_DISCOVERY_BACKEND", Some("mem")),
                ("DYN_REQUEST_PLANE", Some("TCP")),
                ("DYN_RESPONSE_PLANE", None),
                ("DYN_EVENT_PLANE", Some("")),
            ],
            || {
                let args = TestArgs::try_parse_from(["test"]).unwrap();
                let config = args.common.runtime.to_distributed_config().unwrap();
                assert_eq!(
                    config.request_plane,
                    dynamo_runtime::distributed::RequestPlaneMode::Tcp
                );
                assert_eq!(
                    config.event_transport_kind,
                    dynamo_runtime::discovery::EventTransportKind::Zmq
                );
            },
        );
    }

    #[test]
    #[serial_test::serial]
    fn worker_suffix_is_applied_once_to_namespace() {
        for (namespace, suffix, argv, expected) in [
            (Some("dynamo"), Some("qa1"), &["test"][..], "dynamo-qa1"),
            (Some("dynamo-qa1"), Some("qa1"), &["test"][..], "dynamo-qa1"),
            (Some("dynamo"), Some(""), &["test"][..], "dynamo"),
            (Some("dynamo"), None, &["test"][..], "dynamo"),
            (None, Some("qa1"), &["test"][..], "dynamo-qa1"),
            (
                Some("ignored"),
                Some("qa1"),
                &["test", "--namespace", "cli"][..],
                "cli-qa1",
            ),
        ] {
            temp_env::with_vars(
                [
                    ("DYN_NAMESPACE", namespace),
                    ("DYN_NAMESPACE_WORKER_SUFFIX", suffix),
                ],
                || {
                    let args = TestArgs::try_parse_from(argv).unwrap();
                    assert_eq!(args.common.namespace, expected);
                },
            );
        }
    }
}
