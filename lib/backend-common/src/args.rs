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

    /// Advertise a routing mode for this worker set. The frontend then routes
    /// the set by this worker's configuration instead of its own, same as for
    /// the Python workers; omit it to inherit the frontend's routing. The
    /// flags below qualify it and are ignored without it.
    #[arg(long, env = "DYN_ROUTER_MODE", value_enum)]
    pub router_mode: Option<RouterModeArg>,

    /// Session-affinity idle TTL in seconds for the advertised worker set.
    #[arg(
        long,
        env = "DYN_ROUTER_SESSION_AFFINITY_TTL_SECS",
        value_parser = clap::value_parser!(u64).range(1..=MAX_SESSION_AFFINITY_TTL_SECS)
    )]
    pub router_session_affinity_ttl_secs: Option<u64>,

    /// Skip a decode worker above this KV block utilization (0.0 to 1.0);
    /// `None` disables the check.
    #[arg(long, env = "DYN_ACTIVE_DECODE_BLOCKS_THRESHOLD")]
    pub active_decode_blocks_threshold: Option<Nullable<f64>>,

    /// Skip a prefill worker above this many active prefill tokens; `None`
    /// disables the check.
    #[arg(long, env = "DYN_ACTIVE_PREFILL_TOKENS_THRESHOLD")]
    pub active_prefill_tokens_threshold: Option<Nullable<u64>>,

    /// Skip a prefill worker above this fraction of `max_num_batched_tokens`;
    /// `None` disables the check.
    #[arg(long, env = "DYN_ACTIVE_PREFILL_TOKENS_THRESHOLD_FRAC")]
    pub active_prefill_tokens_threshold_frac: Option<Nullable<f64>>,
}

/// A numeric flag value that may also be `None` or empty, meaning unset, as
/// the Python workers' `nullable_float` and `nullable_int` accept.
#[derive(Clone, Copy, Debug)]
pub struct Nullable<T>(pub Option<T>);

impl<T: std::str::FromStr> std::str::FromStr for Nullable<T>
where
    T::Err: std::fmt::Display,
{
    type Err = String;

    fn from_str(raw: &str) -> Result<Self, String> {
        match raw {
            "" | "None" => Ok(Self(None)),
            value => value
                .parse()
                .map(|value| Self(Some(value)))
                .map_err(|error: T::Err| error.to_string()),
        }
    }
}

impl CommonArgs {
    /// The router configuration this worker advertises on its model card, or
    /// `None` without `--router-mode`. An advertised `kv` set takes its KV
    /// tuning from the `DYN_ROUTER_*` environment, the variables the Python
    /// worker flags default from; there are no per-setting flags here.
    pub fn router_config(&self) -> Result<Option<RouterConfig>, DynamoError> {
        let Some(router_mode) = self.router_mode.map(RouterMode::from) else {
            return Ok(None);
        };
        let invalid = |message: String| {
            DynamoError::builder()
                .error_type(ErrorType::Backend(BackendError::InvalidArgument))
                .message(message)
                .build()
        };
        let load_threshold_config = LoadThresholdConfig {
            active_decode_blocks_threshold: self.active_decode_blocks_threshold.and_then(|v| v.0),
            active_prefill_tokens_threshold: self.active_prefill_tokens_threshold.and_then(|v| v.0),
            active_prefill_tokens_threshold_frac: self
                .active_prefill_tokens_threshold_frac
                .and_then(|v| v.0),
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

/// `--router-mode` values, spelled as the Python workers spell them.
#[derive(Clone, Copy, Debug, clap::ValueEnum)]
pub enum RouterModeArg {
    RoundRobin,
    Random,
    #[value(name = "power-of-two")]
    PowerOfTwoChoices,
    Kv,
    Direct,
    LeastLoaded,
    DeviceAwareWeighted,
}

impl From<RouterModeArg> for RouterMode {
    fn from(mode: RouterModeArg) -> Self {
        match mode {
            RouterModeArg::RoundRobin => Self::RoundRobin,
            RouterModeArg::Random => Self::Random,
            RouterModeArg::PowerOfTwoChoices => Self::PowerOfTwoChoices,
            RouterModeArg::Kv => Self::KV,
            RouterModeArg::Direct => Self::Direct,
            RouterModeArg::LeastLoaded => Self::LeastLoaded,
            RouterModeArg::DeviceAwareWeighted => Self::DeviceAwareWeighted,
        }
    }
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

    /// The advertisement flags read `DYN_*` variables, so the ambient
    /// environment must not leak into these tests.
    fn with_router_env<R>(set: &[(&str, &str)], f: impl FnOnce() -> R) -> R {
        let vars = [
            "DYN_ROUTER_MODE",
            "DYN_ROUTER_SESSION_AFFINITY_TTL_SECS",
            "DYN_ACTIVE_DECODE_BLOCKS_THRESHOLD",
            "DYN_ACTIVE_PREFILL_TOKENS_THRESHOLD",
            "DYN_ACTIVE_PREFILL_TOKENS_THRESHOLD_FRAC",
            "DYN_ROUTER_USE_KV_EVENTS",
            "DYN_USE_KV_EVENTS",
        ]
        .map(|key| (key, set.iter().find(|(k, _)| *k == key).map(|(_, v)| *v)));
        temp_env::with_vars(vars, f)
    }

    fn parse(argv: &[&str]) -> Result<TestArgs, clap::Error> {
        TestArgs::try_parse_from([&["test"], argv].concat())
    }

    fn router_config(argv: &[&str]) -> Option<RouterConfig> {
        parse(argv).unwrap().common.router_config().unwrap()
    }

    #[test]
    #[serial_test::serial]
    fn router_advertisement_follows_router_mode() {
        with_router_env(&[], || {
            assert!(router_config(&[]).is_none());
            assert!(router_config(&["--router-session-affinity-ttl-secs", "300"]).is_none());

            // `None` disables a threshold, as for the Python workers.
            let config = router_config(&[
                "--router-mode",
                "kv",
                "--active-decode-blocks-threshold",
                "None",
                "--active-prefill-tokens-threshold",
                "8192",
            ])
            .expect("advertised");
            assert_eq!(
                config.load_threshold_config.active_decode_blocks_threshold,
                None
            );
            assert_eq!(
                config.load_threshold_config.active_prefill_tokens_threshold,
                Some(8192)
            );

            let config = router_config(&[
                "--router-mode",
                "kv",
                "--router-session-affinity-ttl-secs",
                "300",
            ])
            .expect("advertised");
            assert_eq!(config.router_mode, RouterMode::KV);
            assert_eq!(config.session_affinity_ttl_secs, Some(300));
            assert!(config.kv_router_config.use_kv_events);
        });
        with_router_env(
            &[
                ("DYN_ROUTER_MODE", "kv"),
                ("DYN_ROUTER_USE_KV_EVENTS", "false"),
                ("DYN_ACTIVE_DECODE_BLOCKS_THRESHOLD", "None"),
            ],
            || {
                let config = router_config(&[]).expect("advertised from the environment");
                assert_eq!(config.router_mode, RouterMode::KV);
                assert!(!config.kv_router_config.use_kv_events);
                assert_eq!(
                    config.load_threshold_config.active_decode_blocks_threshold,
                    None
                );
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
            assert!(args.common.router_config().is_err());
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
