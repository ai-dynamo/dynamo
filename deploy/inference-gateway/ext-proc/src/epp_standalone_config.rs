// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Standalone EPP mode configuration.
//!
//! `DYN_EPP_MODE=dynamo` (default) uses the Dynamo runtime. `standalone` parses
//! the selector-only config used when the EPP fronts raw OpenAI-compatible
//! workers without a Dynamo runtime.
//!
//! [`EppStandaloneConfig::from_env`] reads envs, applies defaults, and calls
//! [`EppStandaloneConfig::validate_config`] for field and cross-field checks.

use validator::Validate;
use validator::ValidationError;

use crate::vllm_render_client::parse_tokenizer_service_base_url;
use crate::worker_role::{DEFAULT_WORKER_ROLE_LABEL, WorkerRole};

const DEFAULT_KV_EVENT_PORT: u16 = 5557;
const DEFAULT_REPLICA_SYNC_PORT: u16 = 9092;
const DEFAULT_SELECTOR_THREADS: usize = 4;
const DEFAULT_TOKENIZATION_TIMEOUT_MS: u64 = 5_000;
const DEFAULT_TOKENIZER_MAX_RESPONSE_BYTES: usize = 16 * 1024 * 1024;
/// Safety ceiling on concurrent in-flight `pick()`s (tunable), NOT a throughput
/// throttle: it caps concurrent tokenizer/render calls and buffered request
/// bodies so a burst can't exhaust memory. HTTP/2 stream multiplexing means the
/// TCP-connection cap does not bound requests, so this is the actual guardrail.
const DEFAULT_MAX_INFLIGHT_REQUESTS: usize = 1024;

/// Environment variable that selects the EPP operating mode.
pub const DYN_EPP_MODE: &str = "DYN_EPP_MODE";
/// `DYN_EPP_MODE` value selecting standalone mode.
pub const STANDALONE_MODE: &str = "standalone";
/// `DYN_EPP_MODE` value selecting the Dynamo runtime.
pub const DYNAMO_RUNTIME_MODE: &str = "dynamo";

/// Mirrors `DYN_KUBE_DISCOVERY_MODE` in `dynamo_runtime::discovery`; read
/// directly here because standalone mode has no Dynamo runtime to read it for.
const DYN_KUBE_DISCOVERY_MODE: &str = "DYN_KUBE_DISCOVERY_MODE";
/// Environment variable that selects the serving topology within standalone mode.
pub const DYN_EPP_TOPOLOGY_MODE: &str = "DYN_EPP_TOPOLOGY_MODE";
/// `DYN_EPP_TOPOLOGY_MODE` value selecting a single routable worker pool.
pub const AGGREGATED_TOPOLOGY: &str = "aggregated";
/// `DYN_EPP_TOPOLOGY_MODE` value selecting role-split prefill/decode pools.
pub const DISAGGREGATED_TOPOLOGY: &str = "disaggregated";
/// Environment variable naming the pod label that carries a worker's role.
pub const DYN_EPP_WORKER_ROLE_LABEL: &str = "DYN_EPP_WORKER_ROLE_LABEL";

/// Longest Kubernetes label-key name segment (the part after any `/`).
const MAX_LABEL_NAME_LEN: usize = 63;
/// Longest Kubernetes label-key prefix (the DNS subdomain before the `/`).
const MAX_LABEL_PREFIX_LEN: usize = 253;

/// Reads an environment variable, matching the injectable getter used in tests.
type EnvGet<'a> = dyn Fn(&str) -> Option<String> + 'a;

/// Top-level EPP operating mode from `DYN_EPP_MODE`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EppMode {
    // Connects to the Dynamo runtime and constructs a KvRouter. Requires Dynamo workers
    // to be connected to the runtime. (default)
    DynamoRuntime,
    // No runtime connection. Constructs a ServiceSelector for tracking workers, kv state and selecting best worker.
    Standalone,
}

impl EppMode {
    pub fn from_env() -> anyhow::Result<Self> {
        Self::parse(&|k| std::env::var(k).ok())
    }

    fn parse(get: &EnvGet) -> anyhow::Result<Self> {
        match trimmed(get(DYN_EPP_MODE)).as_deref() {
            None | Some(DYNAMO_RUNTIME_MODE) => Ok(Self::DynamoRuntime),
            Some(STANDALONE_MODE) => Ok(Self::Standalone),
            Some(other) => anyhow::bail!(
                "{DYN_EPP_MODE} has invalid value {other:?}; \
                 expected {STANDALONE_MODE:?} or {DYNAMO_RUNTIME_MODE:?}"
            ),
        }
    }
}

/// Serving topology inside standalone mode.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EppTopologyMode {
    /// Every eligible worker is a routable decode target. Today's behavior.
    Aggregated,
    /// The pool carries both prefill and decode pods, split by the worker-role
    /// label into two catalogs with their own `SelectionService` instances.
    Disaggregated,
}

impl EppTopologyMode {
    fn parse(get: &EnvGet) -> anyhow::Result<Self> {
        match trimmed(get(DYN_EPP_TOPOLOGY_MODE)).as_deref() {
            None | Some(AGGREGATED_TOPOLOGY) => Ok(Self::Aggregated),
            Some(DISAGGREGATED_TOPOLOGY) => Ok(Self::Disaggregated),
            // Never fall back: a typo that silently selected aggregated would
            // hand prefill pods to the gateway as destinations.
            Some(other) => anyhow::bail!(
                "{DYN_EPP_TOPOLOGY_MODE} has invalid value {other:?}; \
                 expected {DISAGGREGATED_TOPOLOGY:?} or {AGGREGATED_TOPOLOGY:?}"
            ),
        }
    }

    pub fn is_disaggregated(self) -> bool {
        matches!(self, Self::Disaggregated)
    }
}

/// Wire protocol exposed by the configured tokenizer service.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TokenizerProtocol {
    VllmRender,
}

impl std::str::FromStr for TokenizerProtocol {
    type Err = anyhow::Error;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        match value {
            "vllm-render" => Ok(Self::VllmRender),
            other => anyhow::bail!(
                "DYN_EPP_TOKENIZER_PROTOCOL has invalid value {other:?}; \
                 expected \"vllm-render\""
            ),
        }
    }
}

/// Complete replica synchronization configuration. Its presence enables
/// replica-sync; its fields are validated together when parsing the environment.
#[derive(Debug, Clone, Validate)]
pub struct PeerReplicationConfig {
    /// EPP Service used for peer discovery and state synchronization.
    pub service_name: String,
    /// Local EPP Pod IP from `POD_IP` (downward API), used to exclude self.
    pub pod_ip: String,
    /// ZMQ listener and peer dial port. Every EPP selected by `service_name`
    /// must use the same port.
    #[validate(range(
        min = 1,
        message = "DYN_EPP_REPLICA_SYNC_PORT must be greater than zero"
    ))]
    pub sync_port: u16,
}

#[derive(Debug, Clone, Validate)]
pub struct EppStandaloneConfig {
    /// KV indexer thread-pool size for the in-process selector.
    #[validate(range(min = 1))]
    pub selector_threads: usize,
    /// Enables replica synchronization when set.
    #[validate(nested)]
    pub peer_replication: Option<PeerReplicationConfig>,
    /// `InferencePool` this EPP backs; its selector + target port drive discovery.
    #[validate(length(min = 1, message = "DYN_EPP_INFERENCE_POOL_NAME is required"))]
    pub inference_pool_name: String,
    /// Whether the pool holds one routable worker set or a prefill/decode split.
    pub topology_mode: EppTopologyMode,
    /// Pod label key carrying a worker's role. Read only when
    /// `topology_mode == Disaggregated`; the key's syntax is checked there by
    /// [`EppStandaloneConfig::validate_config`].
    pub worker_role_label: String,
    /// Kubernetes namespace the EPP runs in (from `POD_NAMESPACE`, downward API).
    #[validate(length(min = 1, message = "POD_NAMESPACE is required"))]
    pub namespace: String,
    /// Served/catalog model identity used to group discovered workers.
    #[validate(length(min = 1, message = "DYN_MODEL_NAME is required"))]
    pub model_name: String,
    /// Base URL of the tokenizer service.
    #[validate(length(min = 1, message = "DYN_EPP_TOKENIZER_SERVICE_URL is required"))]
    #[validate(custom(function = "validate_tokenizer_service_url"))]
    pub tokenizer_service_url: String,
    /// Protocol spoken by the configured tokenizer service.
    pub tokenizer_protocol: TokenizerProtocol,
    /// Deadline for calls to the configured tokenization provider.
    #[validate(range(min = 1, message = "DYN_EPP_TOKENIZATION_TIMEOUT_MS must be >= 1"))]
    pub tokenization_timeout_ms: u64,
    /// Maximum successful response body accepted from the tokenizer service.
    #[validate(range(min = 1, message = "DYN_EPP_TOKENIZER_MAX_RESPONSE_BYTES must be >= 1"))]
    pub tokenizer_max_response_bytes: usize,
    /// KV-cache block size; MUST equal the inference engine block size.
    #[validate(range(min = 1, message = "DYN_KV_CACHE_BLOCK_SIZE must be >= 1"))]
    pub block_size: u32,
    /// KV zmq event port.
    #[validate(range(min = 1))]
    pub kv_event_port: u16,
    /// Optional ZMQ port the selector uses for live-stream gap replay. This
    /// must match the worker's explicitly configured replay endpoint.
    #[validate(range(
        min = 1,
        message = "DYN_EPP_KV_EVENT_REPLAY_PORT must be greater than zero when set"
    ))]
    pub replay_port: Option<u16>,
    /// Optional per-worker total KV blocks.
    pub total_kv_blocks: Option<u64>,
    /// Optional per-worker max batched tokens.
    #[validate(range(
        min = 1,
        message = "DYN_EPP_MAX_NUM_BATCHED_TOKENS must be greater than zero when set"
    ))]
    pub max_num_batched_tokens: Option<u64>,
    /// Max batched tokens for prefill workers, defaulting to
    /// [`Self::max_num_batched_tokens`]. This is the denominator of the
    /// scheduler's busy test, and shipped disaggregated recipes differ between
    /// the roles by 8-16x, so sharing one value mis-sizes one role by an order
    /// of magnitude.
    #[validate(range(
        min = 1,
        message = "DYN_EPP_PREFILL_MAX_NUM_BATCHED_TOKENS must be greater than zero when set"
    ))]
    pub prefill_max_num_batched_tokens: Option<u64>,
    /// Max batched tokens for decode workers, defaulting to
    /// [`Self::max_num_batched_tokens`].
    #[validate(range(
        min = 1,
        message = "DYN_EPP_DECODE_MAX_NUM_BATCHED_TOKENS must be greater than zero when set"
    ))]
    pub decode_max_num_batched_tokens: Option<u64>,
    /// Safety ceiling on concurrent in-flight `pick()`s: bounds concurrent
    /// tokenizer/render calls and buffered bodies (a load-shed guardrail, not a
    /// throughput throttle). Excess requests are shed with a 503, not queued.
    #[validate(range(min = 1, message = "DYN_EPP_MAX_INFLIGHT_REQUESTS must be >= 1"))]
    pub max_inflight_requests: usize,
}

impl EppStandaloneConfig {
    /// Build and validate the standalone contract from the process environment.
    pub fn from_env() -> anyhow::Result<Self> {
        reject_unsupported_container_discovery(&|k| std::env::var(k).ok())?;
        let config = Self::parse(&|k| std::env::var(k).ok())?;
        config.validate_config()?;
        Ok(config)
    }

    fn parse(get: &EnvGet) -> anyhow::Result<Self> {
        let tokenizer_protocol = trimmed(get("DYN_EPP_TOKENIZER_PROTOCOL"))
            .ok_or_else(|| anyhow::anyhow!("DYN_EPP_TOKENIZER_PROTOCOL is required"))?
            .parse()?;
        let peer_service = trimmed(get("DYN_EPP_PEER_SERVICE"));
        let pod_ip = trimmed(get("POD_IP"));
        let sync_port = opt_parse::<u16>(get, "DYN_EPP_REPLICA_SYNC_PORT")?
            .unwrap_or(DEFAULT_REPLICA_SYNC_PORT);
        let peer_replication = peer_service
            .map(|service_name| -> anyhow::Result<_> {
                let pod_ip = pod_ip.ok_or_else(|| {
                    anyhow::anyhow!(
                        "invalid {STANDALONE_MODE} EPP config: DYN_EPP_PEER_SERVICE is set but POD_IP is unavailable; \
                         inject POD_IP via the downward API (fieldRef status.podIP)"
                    )
                })?;
                Ok(PeerReplicationConfig {
                    service_name,
                    pod_ip,
                    sync_port,
                })
            })
            .transpose()?;

        // Bound first: the per-role capacity vars fall back to these, so an
        // operator can set only the shared value and have both roles inherit it.
        let total_kv_blocks = opt_parse::<u64>(get, "DYN_EPP_TOTAL_KV_BLOCKS")?;
        let max_num_batched_tokens = opt_parse::<u64>(get, "DYN_EPP_MAX_NUM_BATCHED_TOKENS")?;

        Ok(Self {
            selector_threads: opt_parse::<usize>(get, "DYN_EPP_SELECTION_INDEXER_THREADS")?
                .unwrap_or(DEFAULT_SELECTOR_THREADS),
            peer_replication,
            inference_pool_name: trimmed(get("DYN_EPP_INFERENCE_POOL_NAME")).unwrap_or_default(),
            topology_mode: EppTopologyMode::parse(get)?,
            worker_role_label: trimmed(get(DYN_EPP_WORKER_ROLE_LABEL))
                .unwrap_or_else(|| DEFAULT_WORKER_ROLE_LABEL.to_string()),
            namespace: trimmed(get("POD_NAMESPACE")).unwrap_or_default(),
            model_name: trimmed(get("DYN_MODEL_NAME")).unwrap_or_default(),
            tokenizer_service_url: trimmed(get("DYN_EPP_TOKENIZER_SERVICE_URL"))
                .unwrap_or_default(),
            tokenizer_protocol,
            tokenization_timeout_ms: opt_parse::<u64>(get, "DYN_EPP_TOKENIZATION_TIMEOUT_MS")?
                .unwrap_or(DEFAULT_TOKENIZATION_TIMEOUT_MS),
            tokenizer_max_response_bytes: opt_parse::<usize>(
                get,
                "DYN_EPP_TOKENIZER_MAX_RESPONSE_BYTES",
            )?
            .unwrap_or(DEFAULT_TOKENIZER_MAX_RESPONSE_BYTES),
            block_size: opt_parse::<u32>(get, "DYN_KV_CACHE_BLOCK_SIZE")?.unwrap_or(0),
            kv_event_port: opt_parse::<u16>(get, "DYN_EPP_KV_EVENT_PORT")?
                .unwrap_or(DEFAULT_KV_EVENT_PORT),
            replay_port: opt_parse::<u16>(get, "DYN_EPP_KV_EVENT_REPLAY_PORT")?,
            total_kv_blocks,
            max_num_batched_tokens,
            prefill_max_num_batched_tokens: opt_parse::<u64>(
                get,
                "DYN_EPP_PREFILL_MAX_NUM_BATCHED_TOKENS",
            )?
            .or(max_num_batched_tokens),
            decode_max_num_batched_tokens: opt_parse::<u64>(
                get,
                "DYN_EPP_DECODE_MAX_NUM_BATCHED_TOKENS",
            )?
            .or(max_num_batched_tokens),
            max_inflight_requests: opt_parse::<usize>(get, "DYN_EPP_MAX_INFLIGHT_REQUESTS")?
                .unwrap_or(DEFAULT_MAX_INFLIGHT_REQUESTS),
        })
    }

    /// Enforce the `validator` constraints, then the cross-field rules the
    /// derive cannot express, mapping any failure to `anyhow`.
    pub fn validate_config(&self) -> anyhow::Result<()> {
        self.validate()
            .map_err(|e| anyhow::anyhow!("invalid {STANDALONE_MODE} EPP config: {e}"))?;

        if self.topology_mode.is_disaggregated() {
            // Only meaningful when the label is actually read. Note the
            // emptiness case is unreachable through `parse` — `trimmed` maps a
            // whitespace-only value to `None` and the field takes its non-empty
            // default — but a caller building the struct literally can still
            // produce it, so `validate_label_key` rejects it on its own merits.
            validate_label_key(&self.worker_role_label).map_err(|reason| {
                anyhow::anyhow!("{DYN_EPP_WORKER_ROLE_LABEL} is not a valid label key: {reason}")
            })?;

            if self.peer_replication.is_some() {
                // Both role instances would resolve the same `replica-agg`
                // named port and race to bind one ZMQ publisher. Multi-replica
                // disaggregated EPP is Milestone 8.
                anyhow::bail!(
                    "{DYN_EPP_TOPOLOGY_MODE}={DISAGGREGATED_TOPOLOGY} does not support \
                     DYN_EPP_PEER_SERVICE; multi-replica disaggregated EPP is tracked by \
                     ai-dynamo/dynamo#13418"
                );
            }
        }

        Ok(())
    }

    /// Max batched tokens to register for a worker in `role`.
    pub fn max_num_batched_tokens_for(&self, role: WorkerRole) -> Option<u64> {
        match role {
            WorkerRole::Aggregated => self.max_num_batched_tokens,
            WorkerRole::Prefill => self.prefill_max_num_batched_tokens,
            WorkerRole::Decode => self.decode_max_num_batched_tokens,
        }
    }

    /// Env var naming the effective max-batched-tokens for `role`, for error
    /// messages that must tell an operator which knob to set.
    pub fn max_num_batched_tokens_env_for(role: WorkerRole) -> &'static str {
        match role {
            WorkerRole::Aggregated => "DYN_EPP_MAX_NUM_BATCHED_TOKENS",
            WorkerRole::Prefill => "DYN_EPP_PREFILL_MAX_NUM_BATCHED_TOKENS",
            WorkerRole::Decode => "DYN_EPP_DECODE_MAX_NUM_BATCHED_TOKENS",
        }
    }

    /// Minimal aggregated config for tests in this crate.
    ///
    /// Exists so test fixtures use `..EppStandaloneConfig::for_test()` instead
    /// of exhaustive struct literals — otherwise every new field is a
    /// compile break in every test module that builds one.
    /// `max_num_batched_tokens` is set so `Selector::new` never trips its
    /// queueing fast-fail regardless of the ambient router policy.
    #[cfg(test)]
    pub(crate) fn for_test() -> Self {
        Self {
            selector_threads: 1,
            peer_replication: None,
            inference_pool_name: "test-pool".to_string(),
            topology_mode: EppTopologyMode::Aggregated,
            worker_role_label: DEFAULT_WORKER_ROLE_LABEL.to_string(),
            namespace: "test-ns".to_string(),
            model_name: "test-model".to_string(),
            tokenizer_service_url: "http://vllm-render:8000".to_string(),
            tokenizer_protocol: TokenizerProtocol::VllmRender,
            tokenization_timeout_ms: DEFAULT_TOKENIZATION_TIMEOUT_MS,
            tokenizer_max_response_bytes: DEFAULT_TOKENIZER_MAX_RESPONSE_BYTES,
            block_size: 16,
            kv_event_port: DEFAULT_KV_EVENT_PORT,
            replay_port: None,
            total_kv_blocks: None,
            max_num_batched_tokens: Some(8192),
            prefill_max_num_batched_tokens: Some(8192),
            decode_max_num_batched_tokens: Some(8192),
            max_inflight_requests: DEFAULT_MAX_INFLIGHT_REQUESTS,
        }
    }
}

/// Reject `DYN_KUBE_DISCOVERY_MODE=container` (e.g. intra-pod GMS failover)
/// in standalone mode. Deferred, not a permanent restriction — see
/// TODO(epp-standalone-container-discovery) below for what unblocks it.
///
/// Unlike `DYN_EPP_MODE=dynamo` (which already resolves per-container worker
/// identities; see `hash_container_name` / `pod_worker_ids` in `epp.rs`),
/// standalone has no Dynamo runtime worker registration to fall back on:
/// `pod_discovery.rs` selects workers purely from the K8s Pod's own aggregate
/// `Ready` condition. An intra-pod failover pod never satisfies that
/// condition in steady state — each engine container gets its own readiness
/// probe, and the standby engine is intentionally `NotReady` while armed —
/// so the whole pod, including the healthy active engine, would be silently
/// excluded from every worker index rather than just failing to fail over.
/// Reject it at startup instead of shipping that silent malfunction.
///
/// TODO(epp-standalone-container-discovery): replace `pod_discovery.rs`'s
/// pod-aggregate `pod_is_ready()` gate with a per-named-container readiness
/// check (mirroring dynamo mode's `pod_worker_ids`) so a `WorkerIndex` entry
/// is keyed on an individual container's own `Ready` status, not the pod's.
/// Once that lands, lift this rejection.
fn reject_unsupported_container_discovery(get: &EnvGet) -> anyhow::Result<()> {
    match trimmed(get(DYN_KUBE_DISCOVERY_MODE)).as_deref() {
        Some("container") => anyhow::bail!(
            "standalone EPP ({STANDALONE_MODE} mode) does not yet support \
             {DYN_KUBE_DISCOVERY_MODE}=container (e.g. intra-pod GMS failover): pod discovery \
             selects workers from the Pod's aggregate Ready condition, which a pod with an \
             intentionally-standby engine container never satisfies; use \
             {DYN_EPP_MODE}={DYNAMO_RUNTIME_MODE} instead, or disable intra-pod failover for this worker"
        ),
        _ => Ok(()),
    }
}

fn validate_tokenizer_service_url(value: &str) -> Result<(), ValidationError> {
    if value.is_empty() {
        return Ok(());
    }

    parse_tokenizer_service_base_url(value)
        .map(|_| ())
        .map_err(|_| {
            let mut error = ValidationError::new("tokenizer_service_url_invalid");
            error.message =
                Some("DYN_EPP_TOKENIZER_SERVICE_URL must be an absolute HTTP(S) URL".into());
            error
        })
}

/// Validate a Kubernetes label key, per apimachinery's qualified-name rules.
///
/// Written here because the repo has no label-key validator and `validator` is
/// pulled in derive-only, without `regex`.
///
/// ```text
/// key    := [ prefix "/" ] name        at most one '/'
/// name   := 1..=63 chars, [A-Za-z0-9]([-A-Za-z0-9_.]*[A-Za-z0-9])?   (case-sensitive)
/// prefix := 1..=253 chars, dot-joined DNS-1123 labels, each [a-z0-9]([-a-z0-9]*[a-z0-9])?
/// ```
fn validate_label_key(key: &str) -> Result<(), String> {
    let (prefix, name) = match key.split_once('/') {
        Some((prefix, name)) => (Some(prefix), name),
        None => (None, key),
    };

    if name.contains('/') {
        return Err(format!("{key:?} contains more than one '/'"));
    }
    validate_label_name(name)?;

    match prefix {
        None => Ok(()),
        Some(prefix) => validate_label_prefix(prefix),
    }
}

fn validate_label_name(name: &str) -> Result<(), String> {
    if name.is_empty() {
        return Err("the name segment is empty".to_string());
    }
    if name.len() > MAX_LABEL_NAME_LEN {
        return Err(format!(
            "the name segment is {} characters, over the {MAX_LABEL_NAME_LEN}-character limit",
            name.len()
        ));
    }
    if !bounded_by_alphanumeric(name, char::is_ascii_alphanumeric) {
        return Err(format!(
            "the name segment {name:?} must start and end with an alphanumeric"
        ));
    }
    if let Some(bad) = name
        .chars()
        .find(|c| !(c.is_ascii_alphanumeric() || matches!(c, '-' | '_' | '.')))
    {
        return Err(format!(
            "the name segment {name:?} contains {bad:?}; only alphanumerics, '-', '_' and '.' are allowed"
        ));
    }
    Ok(())
}

fn validate_label_prefix(prefix: &str) -> Result<(), String> {
    if prefix.is_empty() {
        return Err("the prefix before '/' is empty".to_string());
    }
    if prefix.len() > MAX_LABEL_PREFIX_LEN {
        return Err(format!(
            "the prefix is {} characters, over the {MAX_LABEL_PREFIX_LEN}-character limit",
            prefix.len()
        ));
    }
    for label in prefix.split('.') {
        if label.is_empty() {
            return Err(format!(
                "the prefix {prefix:?} has an empty dot-separated label"
            ));
        }
        if !bounded_by_alphanumeric(label, |c| c.is_ascii_lowercase() || c.is_ascii_digit()) {
            return Err(format!(
                "the prefix label {label:?} must start and end with a lowercase alphanumeric"
            ));
        }
        if let Some(bad) = label
            .chars()
            .find(|c| !(c.is_ascii_lowercase() || c.is_ascii_digit() || *c == '-'))
        {
            return Err(format!(
                "the prefix label {label:?} contains {bad:?}; only lowercase alphanumerics and '-' are allowed"
            ));
        }
    }
    Ok(())
}

/// Whether the first and last characters both satisfy `allowed`.
fn bounded_by_alphanumeric(segment: &str, allowed: impl Fn(&char) -> bool) -> bool {
    let mut chars = segment.chars();
    let (Some(first), last) = (chars.next(), segment.chars().next_back()) else {
        return false;
    };
    allowed(&first) && last.is_some_and(|c| allowed(&c))
}

/// Trim a raw value and treat empty as absent.
fn trimmed(v: Option<String>) -> Option<String> {
    v.and_then(|v| {
        let t = v.trim();
        if t.is_empty() {
            None
        } else {
            Some(t.to_string())
        }
    })
}

fn opt_parse<T>(get: &EnvGet, key: &str) -> anyhow::Result<Option<T>>
where
    T: std::str::FromStr,
    T::Err: std::fmt::Display,
{
    match trimmed(get(key)) {
        None => Ok(None),
        Some(raw) => raw
            .parse::<T>()
            .map(Some)
            .map_err(|e| anyhow::anyhow!("{key} has an invalid value {raw:?}: {e}")),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    /// Build an injectable env getter from key/value pairs — no process-global
    /// env mutation, so these tests are isolated and parallel-safe.
    fn getter(pairs: &[(&str, &str)]) -> impl Fn(&str) -> Option<String> {
        let map: HashMap<String, String> = pairs
            .iter()
            .map(|(k, v)| (k.to_string(), v.to_string()))
            .collect();
        move |k| map.get(k).cloned()
    }

    fn parse_mode(pairs: &[(&str, &str)]) -> anyhow::Result<EppMode> {
        EppMode::parse(&getter(pairs))
    }

    /// Mirror `from_env`: resolve (parse) then validate.
    fn parse_cfg(pairs: &[(&str, &str)]) -> anyhow::Result<EppStandaloneConfig> {
        let cfg = EppStandaloneConfig::parse(&getter(pairs))?;
        cfg.validate_config()?;
        Ok(cfg)
    }

    #[test]
    fn mode_defaults_to_dynamo_when_unset() {
        assert_eq!(parse_mode(&[]).unwrap(), EppMode::DynamoRuntime);
    }

    #[test]
    fn mode_parses_known_values() {
        assert_eq!(
            parse_mode(&[("DYN_EPP_MODE", "standalone")]).unwrap(),
            EppMode::Standalone
        );
        assert_eq!(
            parse_mode(&[(DYN_EPP_MODE, DYNAMO_RUNTIME_MODE)]).unwrap(),
            EppMode::DynamoRuntime
        );
    }

    #[test]
    fn mode_rejects_unknown_value() {
        // An unknown value must fail fast, not silently boot full-dynamo mode.
        assert!(parse_mode(&[("DYN_EPP_MODE", "nonsense-mode")]).is_err());
    }

    #[test]
    fn container_discovery_mode_rejected_for_standalone() {
        assert!(
            reject_unsupported_container_discovery(&getter(&[(
                "DYN_KUBE_DISCOVERY_MODE",
                "container"
            )]))
            .is_err(),
            "intra-pod failover's container discovery must fail fast in standalone mode, \
             not silently exclude every pod from the worker index"
        );
    }

    #[test]
    fn pod_discovery_mode_and_unset_are_fine_for_standalone() {
        assert!(reject_unsupported_container_discovery(&getter(&[])).is_ok());
        assert!(
            reject_unsupported_container_discovery(&getter(&[("DYN_KUBE_DISCOVERY_MODE", "pod")]))
                .is_ok()
        );
    }

    #[test]
    fn parses_required_and_defaults() {
        let cfg = parse_cfg(&[
            ("DYN_EPP_INFERENCE_POOL_NAME", "vllm-qwen-pool"),
            ("POD_NAMESPACE", "inference"),
            ("DYN_MODEL_NAME", "Qwen/Qwen3-0.6B"),
            ("DYN_EPP_TOKENIZER_SERVICE_URL", "http://vllm-render:8000"),
            ("DYN_EPP_TOKENIZER_PROTOCOL", "vllm-render"),
            ("DYN_KV_CACHE_BLOCK_SIZE", "16"),
        ])
        .expect("config should parse");
        assert_eq!(cfg.selector_threads, DEFAULT_SELECTOR_THREADS);
        // No peer service => single-replica (replica sync off).
        assert!(cfg.peer_replication.is_none());
        assert_eq!(cfg.inference_pool_name, "vllm-qwen-pool");
        assert_eq!(cfg.namespace, "inference");
        assert_eq!(cfg.model_name, "Qwen/Qwen3-0.6B");
        assert_eq!(cfg.tokenizer_service_url, "http://vllm-render:8000");
        assert_eq!(cfg.tokenizer_protocol, TokenizerProtocol::VllmRender);
        assert_eq!(cfg.tokenization_timeout_ms, DEFAULT_TOKENIZATION_TIMEOUT_MS);
        assert_eq!(
            cfg.tokenizer_max_response_bytes,
            DEFAULT_TOKENIZER_MAX_RESPONSE_BYTES
        );
        assert_eq!(cfg.block_size, 16);
        assert_eq!(cfg.kv_event_port, DEFAULT_KV_EVENT_PORT);
        assert!(cfg.replay_port.is_none());
        assert!(cfg.total_kv_blocks.is_none());
        assert_eq!(cfg.max_inflight_requests, DEFAULT_MAX_INFLIGHT_REQUESTS);
    }

    #[test]
    fn missing_pod_namespace_fails() {
        // POD_NAMESPACE is the single namespace source (downward API); without
        // it the EPP can't watch its pool, pods, or peers.
        assert!(
            parse_cfg(&[
                ("DYN_EPP_INFERENCE_POOL_NAME", "vllm-qwen-pool"),
                ("DYN_MODEL_NAME", "Qwen/Qwen3-0.6B"),
                ("DYN_EPP_TOKENIZER_SERVICE_URL", "http://vllm-render:8000"),
                ("DYN_EPP_TOKENIZER_PROTOCOL", "vllm-render"),
                ("DYN_KV_CACHE_BLOCK_SIZE", "16"),
            ])
            .is_err()
        );
    }

    #[test]
    fn peer_replication_config() {
        type ExtraEnv = &'static [(&'static str, &'static str)];
        type Expected = Result<(u16, usize), &'static str>;
        type Case = (&'static str, ExtraEnv, Expected);

        let required = [
            ("DYN_EPP_INFERENCE_POOL_NAME", "vllm-qwen-pool"),
            ("POD_NAMESPACE", "inference"),
            ("DYN_MODEL_NAME", "Qwen/Qwen3-0.6B"),
            ("DYN_EPP_TOKENIZER_SERVICE_URL", "http://vllm-render:8000"),
            ("DYN_EPP_TOKENIZER_PROTOCOL", "vllm-render"),
            ("DYN_KV_CACHE_BLOCK_SIZE", "16"),
        ];
        let cases: [Case; 6] = [
            (
                "default port",
                &[
                    ("DYN_EPP_PEER_SERVICE", "dynamo-epp"),
                    ("POD_IP", "10.0.0.10"),
                    ("DYN_EPP_SELECTION_INDEXER_THREADS", "8"),
                ],
                Ok((DEFAULT_REPLICA_SYNC_PORT, 8)),
            ),
            (
                "overridden port",
                &[
                    ("DYN_EPP_PEER_SERVICE", "dynamo-epp"),
                    ("POD_IP", "10.0.0.10"),
                    ("DYN_EPP_REPLICA_SYNC_PORT", "9192"),
                ],
                Ok((9192, DEFAULT_SELECTOR_THREADS)),
            ),
            (
                "missing pod ip",
                &[("DYN_EPP_PEER_SERVICE", "dynamo-epp")],
                Err("POD_IP"),
            ),
            (
                "blank pod ip",
                &[("DYN_EPP_PEER_SERVICE", "dynamo-epp"), ("POD_IP", " ")],
                Err("POD_IP"),
            ),
            (
                "zero port",
                &[
                    ("DYN_EPP_PEER_SERVICE", "dynamo-epp"),
                    ("POD_IP", "10.0.0.10"),
                    ("DYN_EPP_REPLICA_SYNC_PORT", "0"),
                ],
                Err("DYN_EPP_REPLICA_SYNC_PORT"),
            ),
            (
                "out of range port",
                &[
                    ("DYN_EPP_PEER_SERVICE", "dynamo-epp"),
                    ("POD_IP", "10.0.0.10"),
                    ("DYN_EPP_REPLICA_SYNC_PORT", "65536"),
                ],
                Err("DYN_EPP_REPLICA_SYNC_PORT"),
            ),
        ];

        for (name, extra, expected) in cases {
            let mut env = required.to_vec();
            env.extend_from_slice(extra);
            match expected {
                Ok((port, selector_threads)) => {
                    let cfg = parse_cfg(&env).unwrap_or_else(|error| panic!("{name}: {error}"));
                    let replication = cfg
                        .peer_replication
                        .as_ref()
                        .unwrap_or_else(|| panic!("{name}: replication should be enabled"));
                    assert_eq!(replication.service_name, "dynamo-epp", "{name}");
                    assert_eq!(replication.pod_ip, "10.0.0.10", "{name}");
                    assert_eq!(replication.sync_port, port, "{name}");
                    assert_eq!(cfg.selector_threads, selector_threads, "{name}");
                }
                Err(expected_error) => {
                    let error = parse_cfg(&env).expect_err(name);
                    assert!(
                        error.to_string().contains(expected_error),
                        "{name}: {error}"
                    );
                }
            }
        }
    }

    #[test]
    fn missing_inference_pool_name_fails() {
        assert!(
            parse_cfg(&[
                ("POD_NAMESPACE", "inference"),
                ("DYN_MODEL_NAME", "Qwen/Qwen3-0.6B"),
                ("DYN_EPP_TOKENIZER_SERVICE_URL", "http://vllm-render:8000"),
                ("DYN_EPP_TOKENIZER_PROTOCOL", "vllm-render"),
                ("DYN_KV_CACHE_BLOCK_SIZE", "16"),
            ])
            .is_err()
        );
    }

    #[test]
    fn zero_block_size_fails() {
        assert!(
            parse_cfg(&[
                ("DYN_EPP_INFERENCE_POOL_NAME", "vllm-qwen-pool"),
                ("POD_NAMESPACE", "inference"),
                ("DYN_MODEL_NAME", "Qwen/Qwen3-0.6B"),
                ("DYN_EPP_TOKENIZER_SERVICE_URL", "http://vllm-render:8000"),
                ("DYN_EPP_TOKENIZER_PROTOCOL", "vllm-render"),
                ("DYN_KV_CACHE_BLOCK_SIZE", "0"),
            ])
            .is_err()
        );
    }

    #[test]
    fn replay_port_is_optional_and_uses_the_kv_event_name() {
        let cfg = parse_cfg(&[
            ("DYN_EPP_INFERENCE_POOL_NAME", "vllm-qwen-pool"),
            ("POD_NAMESPACE", "inference"),
            ("DYN_MODEL_NAME", "Qwen/Qwen3-0.6B"),
            ("DYN_EPP_TOKENIZER_SERVICE_URL", "http://vllm-render:8000"),
            ("DYN_EPP_TOKENIZER_PROTOCOL", "vllm-render"),
            ("DYN_KV_CACHE_BLOCK_SIZE", "16"),
            ("DYN_EPP_KV_EVENT_REPLAY_PORT", "5558"),
        ])
        .unwrap();
        assert_eq!(cfg.replay_port, Some(5558));
    }

    #[test]
    fn zero_replay_port_fails() {
        assert!(
            parse_cfg(&[
                ("DYN_EPP_INFERENCE_POOL_NAME", "vllm-qwen-pool"),
                ("POD_NAMESPACE", "inference"),
                ("DYN_MODEL_NAME", "Qwen/Qwen3-0.6B"),
                ("DYN_EPP_TOKENIZER_SERVICE_URL", "http://vllm-render:8000"),
                ("DYN_EPP_TOKENIZER_PROTOCOL", "vllm-render"),
                ("DYN_KV_CACHE_BLOCK_SIZE", "16"),
                ("DYN_EPP_KV_EVENT_REPLAY_PORT", "0"),
            ])
            .is_err()
        );
    }

    #[test]
    fn zero_max_num_batched_tokens_fails() {
        assert!(
            parse_cfg(&[
                ("DYN_EPP_INFERENCE_POOL_NAME", "vllm-qwen-pool"),
                ("POD_NAMESPACE", "inference"),
                ("DYN_MODEL_NAME", "Qwen/Qwen3-0.6B"),
                ("DYN_EPP_TOKENIZER_SERVICE_URL", "http://vllm-render:8000"),
                ("DYN_EPP_TOKENIZER_PROTOCOL", "vllm-render"),
                ("DYN_KV_CACHE_BLOCK_SIZE", "16"),
                ("DYN_EPP_MAX_NUM_BATCHED_TOKENS", "0"),
            ])
            .is_err()
        );
    }

    #[test]
    fn max_inflight_requests_can_be_overridden() {
        let cfg = parse_cfg(&[
            ("DYN_EPP_INFERENCE_POOL_NAME", "vllm-qwen-pool"),
            ("POD_NAMESPACE", "inference"),
            ("DYN_MODEL_NAME", "Qwen/Qwen3-0.6B"),
            ("DYN_EPP_TOKENIZER_SERVICE_URL", "http://vllm-render:8000"),
            ("DYN_EPP_TOKENIZER_PROTOCOL", "vllm-render"),
            ("DYN_KV_CACHE_BLOCK_SIZE", "16"),
            ("DYN_EPP_MAX_INFLIGHT_REQUESTS", "256"),
        ])
        .unwrap();
        assert_eq!(cfg.max_inflight_requests, 256);
    }

    #[test]
    fn zero_max_inflight_requests_fails() {
        assert!(
            parse_cfg(&[
                ("DYN_EPP_INFERENCE_POOL_NAME", "vllm-qwen-pool"),
                ("POD_NAMESPACE", "inference"),
                ("DYN_MODEL_NAME", "Qwen/Qwen3-0.6B"),
                ("DYN_EPP_TOKENIZER_SERVICE_URL", "http://vllm-render:8000"),
                ("DYN_EPP_TOKENIZER_PROTOCOL", "vllm-render"),
                ("DYN_KV_CACHE_BLOCK_SIZE", "16"),
                ("DYN_EPP_MAX_INFLIGHT_REQUESTS", "0"),
            ])
            .is_err()
        );
    }

    #[test]
    fn tokenizer_service_url_is_required() {
        assert!(
            parse_cfg(&[
                ("DYN_EPP_INFERENCE_POOL_NAME", "vllm-qwen-pool"),
                ("POD_NAMESPACE", "inference"),
                ("DYN_MODEL_NAME", "Qwen/Qwen3-0.6B"),
                ("DYN_EPP_TOKENIZER_PROTOCOL", "vllm-render"),
                ("DYN_KV_CACHE_BLOCK_SIZE", "16"),
            ])
            .is_err()
        );
    }

    #[test]
    fn tokenizer_service_url_must_be_http() {
        assert!(
            parse_cfg(&[
                ("DYN_EPP_INFERENCE_POOL_NAME", "vllm-qwen-pool"),
                ("POD_NAMESPACE", "inference"),
                ("DYN_MODEL_NAME", "Qwen/Qwen3-0.6B"),
                ("DYN_EPP_TOKENIZER_SERVICE_URL", "unix:///tmp/vllm.sock"),
                ("DYN_EPP_TOKENIZER_PROTOCOL", "vllm-render"),
                ("DYN_KV_CACHE_BLOCK_SIZE", "16"),
            ])
            .is_err()
        );
    }

    #[test]
    fn tokenization_timeout_must_be_positive() {
        assert!(
            parse_cfg(&[
                ("DYN_EPP_INFERENCE_POOL_NAME", "vllm-qwen-pool"),
                ("POD_NAMESPACE", "inference"),
                ("DYN_MODEL_NAME", "Qwen/Qwen3-0.6B"),
                ("DYN_EPP_TOKENIZER_SERVICE_URL", "http://vllm-render:8000"),
                ("DYN_EPP_TOKENIZER_PROTOCOL", "vllm-render"),
                ("DYN_EPP_TOKENIZATION_TIMEOUT_MS", "0"),
                ("DYN_KV_CACHE_BLOCK_SIZE", "16"),
            ])
            .is_err()
        );
    }

    #[test]
    fn tokenizer_max_response_bytes_can_be_overridden() {
        let cfg = parse_cfg(&[
            ("DYN_EPP_INFERENCE_POOL_NAME", "vllm-qwen-pool"),
            ("POD_NAMESPACE", "inference"),
            ("DYN_MODEL_NAME", "Qwen/Qwen3-0.6B"),
            ("DYN_EPP_TOKENIZER_SERVICE_URL", "http://vllm-render:8000"),
            ("DYN_EPP_TOKENIZER_PROTOCOL", "vllm-render"),
            ("DYN_EPP_TOKENIZER_MAX_RESPONSE_BYTES", "33554432"),
            ("DYN_KV_CACHE_BLOCK_SIZE", "16"),
        ])
        .unwrap();

        assert_eq!(cfg.tokenizer_max_response_bytes, 32 * 1024 * 1024);
    }

    #[test]
    fn tokenizer_max_response_bytes_must_be_positive() {
        assert!(
            parse_cfg(&[
                ("DYN_EPP_INFERENCE_POOL_NAME", "vllm-qwen-pool"),
                ("POD_NAMESPACE", "inference"),
                ("DYN_MODEL_NAME", "Qwen/Qwen3-0.6B"),
                ("DYN_EPP_TOKENIZER_SERVICE_URL", "http://vllm-render:8000"),
                ("DYN_EPP_TOKENIZER_PROTOCOL", "vllm-render"),
                ("DYN_EPP_TOKENIZER_MAX_RESPONSE_BYTES", "0"),
                ("DYN_KV_CACHE_BLOCK_SIZE", "16"),
            ])
            .is_err()
        );
    }

    #[test]
    fn tokenizer_protocol_is_required() {
        assert!(
            parse_cfg(&[
                ("DYN_EPP_INFERENCE_POOL_NAME", "vllm-qwen-pool"),
                ("POD_NAMESPACE", "inference"),
                ("DYN_MODEL_NAME", "Qwen/Qwen3-0.6B"),
                ("DYN_EPP_TOKENIZER_SERVICE_URL", "http://vllm-render:8000"),
                ("DYN_KV_CACHE_BLOCK_SIZE", "16"),
            ])
            .is_err()
        );
    }

    #[test]
    fn unsupported_tokenizer_protocol_fails() {
        assert!(
            parse_cfg(&[
                ("DYN_EPP_INFERENCE_POOL_NAME", "vllm-qwen-pool"),
                ("POD_NAMESPACE", "inference"),
                ("DYN_MODEL_NAME", "Qwen/Qwen3-0.6B"),
                (
                    "DYN_EPP_TOKENIZER_SERVICE_URL",
                    "http://sglang-tokenizer:30000"
                ),
                ("DYN_EPP_TOKENIZER_PROTOCOL", "sglang-tokenize"),
                ("DYN_KV_CACHE_BLOCK_SIZE", "16"),
            ])
            .is_err()
        );
    }

    // --- disaggregated topology -------------------------------------------

    /// The minimum env every valid standalone config needs, so topology tests
    /// state only what they are actually about.
    const BASE_ENV: &[(&str, &str)] = &[
        ("DYN_EPP_INFERENCE_POOL_NAME", "vllm-qwen-pool"),
        ("POD_NAMESPACE", "inference"),
        ("DYN_MODEL_NAME", "Qwen/Qwen3-0.6B"),
        ("DYN_EPP_TOKENIZER_SERVICE_URL", "http://vllm-render:8000"),
        ("DYN_EPP_TOKENIZER_PROTOCOL", "vllm-render"),
        ("DYN_KV_CACHE_BLOCK_SIZE", "16"),
    ];

    fn cfg_with(extra: &[(&str, &str)]) -> anyhow::Result<EppStandaloneConfig> {
        let mut pairs = BASE_ENV.to_vec();
        pairs.extend_from_slice(extra);
        parse_cfg(&pairs)
    }

    fn disagg_with(extra: &[(&str, &str)]) -> anyhow::Result<EppStandaloneConfig> {
        let mut pairs = vec![(DYN_EPP_TOPOLOGY_MODE, DISAGGREGATED_TOPOLOGY)];
        pairs.extend_from_slice(extra);
        cfg_with(&pairs)
    }

    fn parse_topology(pairs: &[(&str, &str)]) -> anyhow::Result<EppTopologyMode> {
        EppTopologyMode::parse(&getter(pairs))
    }

    #[test]
    fn topology_defaults_to_aggregated_when_unset_or_blank() {
        assert_eq!(parse_topology(&[]).unwrap(), EppTopologyMode::Aggregated);
        assert_eq!(
            parse_topology(&[(DYN_EPP_TOPOLOGY_MODE, "   ")]).unwrap(),
            EppTopologyMode::Aggregated
        );
        assert_eq!(
            parse_topology(&[(DYN_EPP_TOPOLOGY_MODE, AGGREGATED_TOPOLOGY)]).unwrap(),
            EppTopologyMode::Aggregated
        );
    }

    #[test]
    fn topology_parses_disaggregated() {
        assert_eq!(
            parse_topology(&[(DYN_EPP_TOPOLOGY_MODE, DISAGGREGATED_TOPOLOGY)]).unwrap(),
            EppTopologyMode::Disaggregated
        );
    }

    #[test]
    fn topology_rejects_unknown_values_naming_both_alternatives() {
        // Falling back to aggregated on a typo would make every prefill pod a
        // routable gateway destination, so this must fail loudly.
        for value in ["disagg", "DISAGGREGATED", "true", "nonsense"] {
            let error = parse_topology(&[(DYN_EPP_TOPOLOGY_MODE, value)])
                .expect_err("unknown topology must not fall back")
                .to_string();
            assert!(error.contains(DISAGGREGATED_TOPOLOGY), "{error}");
            assert!(error.contains(AGGREGATED_TOPOLOGY), "{error}");
        }
    }

    #[test]
    fn worker_role_label_has_a_default() {
        let cfg = cfg_with(&[]).unwrap();
        assert_eq!(cfg.worker_role_label, DEFAULT_WORKER_ROLE_LABEL);
        assert_eq!(cfg.topology_mode, EppTopologyMode::Aggregated);
    }

    #[test]
    fn worker_role_label_is_overridable() {
        let cfg = disagg_with(&[(
            DYN_EPP_WORKER_ROLE_LABEL,
            "nvidia.com/dynamo-component-type",
        )])
        .unwrap();
        assert_eq!(cfg.worker_role_label, "nvidia.com/dynamo-component-type");
    }

    #[test]
    fn malformed_role_label_keys_fail_only_under_disaggregated() {
        let malformed = [
            ("a/b/c", "two slashes"),
            ("prefix/", "empty name segment"),
            ("/role", "empty prefix"),
            ("-lead", "name starts with a dash"),
            ("trail-", "name ends with a dash"),
            ("NVIDIA.com/role", "uppercase in the prefix"),
        ];

        for (key, why) in malformed {
            assert!(
                disagg_with(&[(DYN_EPP_WORKER_ROLE_LABEL, key)]).is_err(),
                "{key:?} should be rejected under disaggregated ({why})"
            );
            // Aggregated never reads the label, so it must not gate startup on it.
            assert!(
                cfg_with(&[(DYN_EPP_WORKER_ROLE_LABEL, key)]).is_ok(),
                "{key:?} must be ignored under aggregated ({why})"
            );
        }
    }

    #[test]
    fn role_label_name_segment_length_is_bounded() {
        let ok = "a".repeat(MAX_LABEL_NAME_LEN);
        let too_long = "a".repeat(MAX_LABEL_NAME_LEN + 1);
        assert!(disagg_with(&[(DYN_EPP_WORKER_ROLE_LABEL, &ok)]).is_ok());
        assert!(disagg_with(&[(DYN_EPP_WORKER_ROLE_LABEL, &too_long)]).is_err());
    }

    #[test]
    fn role_label_prefix_length_is_bounded() {
        let long_prefix = format!("{}.com", "a".repeat(MAX_LABEL_PREFIX_LEN));
        assert!(
            disagg_with(&[(DYN_EPP_WORKER_ROLE_LABEL, &format!("{long_prefix}/role"))]).is_err()
        );
    }

    #[test]
    fn valid_prefixed_and_bare_role_label_keys_pass() {
        for key in [
            "nvidia.com/dynamo-worker-role",
            "role",
            "a.b.c/some_name.with-punct",
            "x/y",
        ] {
            assert!(
                disagg_with(&[(DYN_EPP_WORKER_ROLE_LABEL, key)]).is_ok(),
                "{key:?} should be accepted"
            );
        }
    }

    #[test]
    fn disaggregated_rejects_peer_service() {
        // Both role instances would resolve the same `replica-agg` named port
        // and race to bind one ZMQ publisher.
        let error = disagg_with(&[
            ("DYN_EPP_PEER_SERVICE", "dynamo-epp"),
            ("POD_IP", "10.0.0.5"),
        ])
        .expect_err("multi-replica disaggregated EPP is not supported yet")
        .to_string();
        assert!(error.contains("DYN_EPP_PEER_SERVICE"), "{error}");
        assert!(error.contains("13418"), "{error}");
    }

    #[test]
    fn aggregated_still_accepts_peer_service() {
        assert!(
            cfg_with(&[
                ("DYN_EPP_PEER_SERVICE", "dynamo-epp"),
                ("POD_IP", "10.0.0.5")
            ])
            .is_ok()
        );
    }

    // --- per-role capacity -------------------------------------------------

    #[test]
    fn per_role_capacity_falls_back_to_the_shared_value() {
        let cfg = cfg_with(&[("DYN_EPP_MAX_NUM_BATCHED_TOKENS", "8192")]).unwrap();

        for role in [WorkerRole::Prefill, WorkerRole::Decode] {
            assert_eq!(cfg.max_num_batched_tokens_for(role), Some(8192));
        }
    }

    #[test]
    fn per_role_capacity_overrides_independently() {
        // Shipped disaggregated recipes differ between the roles by 8-16x, so
        // one shared value mis-sizes one role by an order of magnitude.
        let cfg = disagg_with(&[
            ("DYN_EPP_MAX_NUM_BATCHED_TOKENS", "8192"),
            ("DYN_EPP_PREFILL_MAX_NUM_BATCHED_TOKENS", "16384"),
            ("DYN_EPP_DECODE_MAX_NUM_BATCHED_TOKENS", "2048"),
        ])
        .unwrap();

        assert_eq!(
            cfg.max_num_batched_tokens_for(WorkerRole::Prefill),
            Some(16384)
        );
        assert_eq!(
            cfg.max_num_batched_tokens_for(WorkerRole::Decode),
            Some(2048)
        );
        assert_eq!(
            cfg.max_num_batched_tokens_for(WorkerRole::Aggregated),
            Some(8192)
        );
    }

    #[test]
    fn per_role_capacity_is_absent_when_nothing_is_set() {
        let cfg = cfg_with(&[]).unwrap();
        for role in [
            WorkerRole::Aggregated,
            WorkerRole::Prefill,
            WorkerRole::Decode,
        ] {
            assert!(cfg.max_num_batched_tokens_for(role).is_none());
        }
    }

    #[test]
    fn zero_per_role_max_num_batched_tokens_fails() {
        // Mirrors the existing guard on the shared var: a zero denominator
        // would reach the scheduler's busy test.
        for var in [
            "DYN_EPP_PREFILL_MAX_NUM_BATCHED_TOKENS",
            "DYN_EPP_DECODE_MAX_NUM_BATCHED_TOKENS",
        ] {
            assert!(disagg_with(&[(var, "0")]).is_err(), "{var} = 0 must fail");
        }
    }

    #[test]
    fn per_role_capacity_env_names_are_reported_for_operators() {
        assert_eq!(
            EppStandaloneConfig::max_num_batched_tokens_env_for(WorkerRole::Prefill),
            "DYN_EPP_PREFILL_MAX_NUM_BATCHED_TOKENS"
        );
        assert_eq!(
            EppStandaloneConfig::max_num_batched_tokens_env_for(WorkerRole::Decode),
            "DYN_EPP_DECODE_MAX_NUM_BATCHED_TOKENS"
        );
        assert_eq!(
            EppStandaloneConfig::max_num_batched_tokens_env_for(WorkerRole::Aggregated),
            "DYN_EPP_MAX_NUM_BATCHED_TOKENS"
        );
    }
}
