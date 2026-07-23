// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::env::{self, VarError};
use std::fmt;
use std::path::PathBuf;
use std::str::FromStr;
use std::sync::OnceLock;
use std::time::Duration;

use derive_builder::Builder;
use serde::{Deserialize, Serialize};

use crate::protocols::{
    BlockHashOptions, LocalBlockHash, complete_block_count, compute_block_hash_for_seq,
    compute_seq_hash_for_block,
};
use crate::tracking_hash::{
    TrackingHashAlgorithm, TrackingHashContext, TrackingHashScope, validate_tracking_hash_options,
};

const fn default_track_prefill_tokens() -> bool {
    true
}

pub const DYN_ROUTER_MIN_INITIAL_WORKERS: &str = "DYN_ROUTER_MIN_INITIAL_WORKERS";

pub fn min_initial_workers_from_env() -> anyhow::Result<usize> {
    match env::var(DYN_ROUTER_MIN_INITIAL_WORKERS) {
        Ok(value) => value.parse::<usize>().map_err(|error| {
            anyhow::anyhow!(
                "{DYN_ROUTER_MIN_INITIAL_WORKERS} must be a non-negative integer, got {value:?}: {error}"
            )
        }),
        Err(VarError::NotPresent) => Ok(0),
        Err(VarError::NotUnicode(_)) => {
            anyhow::bail!("{DYN_ROUTER_MIN_INITIAL_WORKERS} must be valid unicode")
        }
    }
}

const fn default_host_cache_hit_weight() -> f64 {
    0.75
}

const fn default_disk_cache_hit_weight() -> f64 {
    0.25
}

const fn default_prefill_load_scale() -> f64 {
    1.0
}

const fn default_overlap_score_credit_decay() -> f64 {
    0.0
}

pub const OVERLAP_SCORE_CREDIT_RANGE_ERROR: &str =
    "overlap_score_credit must be a finite, non-negative number";

pub fn overlap_score_credit_error_message(value: f64) -> Option<&'static str> {
    if value.is_finite() && value >= 0.0 {
        None
    } else {
        Some(OVERLAP_SCORE_CREDIT_RANGE_ERROR)
    }
}

fn validate_overlap_score_credit(value: f64) -> Result<(), String> {
    let Some(message) = overlap_score_credit_error_message(value) else {
        return Ok(());
    };
    Err(message.to_string())
}

fn validate_min(field: &str, value: f64, min: f64) -> Result<(), String> {
    if value >= min {
        return Ok(());
    }
    Err(format!("{field} must be greater than or equal to {min}"))
}

fn validate_range(field: &str, value: f64, min: f64, max: f64) -> Result<(), String> {
    if value >= min && value <= max {
        return Ok(());
    }
    Err(format!("{field} must be between {min} and {max}"))
}

pub fn apply_deprecated_overlap_score_weight_override(
    value: f64,
    overlap_score_credit: &mut f64,
    prefill_load_scale: &mut f64,
) {
    *prefill_load_scale = value;
    if value == 0.0 {
        *overlap_score_credit = 0.0;
    }
}

/// Build a [`KvRouterConfig`] from defaults and standard Dynamo environment variables.
///
/// # Panics
///
/// Panics when `DYN_ROUTER_TRACKING_HASH` is not a supported algorithm. Startup
/// paths should use [`try_kv_router_config_from_dynamo_env`] to report the error.
pub fn kv_router_config_from_dynamo_env() -> KvRouterConfig {
    try_kv_router_config_from_dynamo_env()
        .unwrap_or_else(|error| panic!("invalid Dynamo router environment configuration: {error}"))
}

/// Build a [`KvRouterConfig`] from standard Dynamo environment variables.
pub fn try_kv_router_config_from_dynamo_env() -> Result<KvRouterConfig, String> {
    let config = kv_router_config_from_lookup(|key| env::var(key).ok())?;
    log_env_config(&config);
    Ok(config)
}

fn log_env_config(config: &KvRouterConfig) {
    tracing::info!(
        overlap_score_credit = config.overlap_score_credit,
        overlap_score_credit_decay = config.overlap_score_credit_decay,
        prefill_load_scale = config.prefill_load_scale,
        router_temperature = config.router_temperature,
        use_kv_events = config.use_kv_events,
        router_replica_sync = config.router_replica_sync,
        router_track_active_blocks = config.router_track_active_blocks,
        router_track_output_blocks = config.router_track_output_blocks,
        router_assume_kv_reuse = config.router_assume_kv_reuse,
        router_track_prefill_tokens = config.router_track_prefill_tokens,
        router_tracking_hash = %config.router_tracking_hash,
        router_tracking_key_id = ?config.router_tracking_key_id,
        router_queue_threshold = ?config.router_queue_threshold,
        router_policy_config = ?config.router_policy_config,
        conditional_disagg_enabled = config.conditional_disagg_enabled,
        conditional_disagg_policy = ?config.conditional_disagg_policy,
        conditional_disagg_eff_isl_threshold = config.conditional_disagg_eff_isl_threshold,
        conditional_disagg_eff_isl_ratio_threshold = config.conditional_disagg_eff_isl_ratio_threshold,
        conditional_disagg_prefill_busy_threshold = ?config.conditional_disagg_prefill_busy_threshold,
        conditional_disagg_decode_busy_threshold = ?config.conditional_disagg_decode_busy_threshold,
        router_predicted_ttl_secs = ?config.router_predicted_ttl_secs,
        "KvRouterConfig initialized (DYN_* env overrides applied)"
    );
}

fn kv_router_config_from_lookup(
    get_env: impl Fn(&str) -> Option<String>,
) -> Result<KvRouterConfig, String> {
    fn parse_f64(get_env: &impl Fn(&str) -> Option<String>, key: &str) -> Option<f64> {
        get_env(key).and_then(|value| value.parse().ok())
    }

    fn parse_usize(get_env: &impl Fn(&str) -> Option<String>, key: &str) -> Option<usize> {
        get_env(key).and_then(|value| value.parse().ok())
    }

    fn parse_bool(get_env: &impl Fn(&str) -> Option<String>, key: &str) -> Option<bool> {
        // Empty or unrecognized values yield None so the default is preserved.
        get_env(key).and_then(|value| dynamo_truthy::parse_bool_opt(&value))
    }

    let mut config = KvRouterConfig::default();

    if let Some(value) = parse_f64(&get_env, "DYN_ROUTER_KV_OVERLAP_SCORE_CREDIT") {
        config.overlap_score_credit = value;
    }
    if let Some(value) = parse_f64(&get_env, "DYN_ROUTER_KV_OVERLAP_SCORE_CREDIT_DECAY") {
        config.overlap_score_credit_decay = value;
    }
    if let Some(value) = parse_f64(&get_env, "DYN_ROUTER_PREFILL_LOAD_SCALE") {
        config.prefill_load_scale = value;
    }
    for key in [
        "DYN_ROUTER_KV_OVERLAP_SCORE_WEIGHT",
        "DYN_OVERLAP_SCORE_WEIGHT",
    ] {
        if let Some(value) = parse_f64(&get_env, key) {
            tracing::warn!("{key} is deprecated; use DYN_ROUTER_PREFILL_LOAD_SCALE");
            apply_deprecated_overlap_score_weight_override(
                value,
                &mut config.overlap_score_credit,
                &mut config.prefill_load_scale,
            );
            break;
        }
    }
    if let Some(value) = parse_f64(&get_env, "DYN_ROUTER_TEMPERATURE") {
        config.router_temperature = value;
    }
    if let Some(value) = parse_bool(&get_env, "DYN_USE_KV_EVENTS") {
        config.use_kv_events = value;
    }
    if let Some(value) = parse_bool(&get_env, "DYN_ROUTER_REPLICA_SYNC") {
        config.router_replica_sync = value;
    }
    if let Some(value) = parse_bool(&get_env, "DYN_ROUTER_TRACK_ACTIVE_BLOCKS") {
        config.router_track_active_blocks = value;
    }
    if let Some(value) = parse_bool(&get_env, "DYN_ROUTER_TRACK_OUTPUT_BLOCKS") {
        config.router_track_output_blocks = value;
    }
    if let Some(value) = parse_bool(&get_env, "DYN_ROUTER_ASSUME_KV_REUSE") {
        config.router_assume_kv_reuse = value;
    }
    if let Some(value) = parse_bool(&get_env, "DYN_ROUTER_TRACK_PREFILL_TOKENS") {
        config.router_track_prefill_tokens = value;
    }
    if let Some(value) = get_env("DYN_ROUTER_TRACKING_HASH") {
        config.router_tracking_hash = value.parse()?;
    }
    if let Some(value) = get_env("DYN_ROUTER_TRACKING_KEY_FILE") {
        config.router_tracking_key_file = Some(value.into());
    }
    if let Some(value) = get_env("DYN_ROUTER_TRACKING_KEY_ID") {
        config.router_tracking_key_id = Some(value);
    }
    if let Some(value) = parse_f64(&get_env, "DYN_ROUTER_QUEUE_THRESHOLD") {
        config.router_queue_threshold = Some(value);
    }
    if let Some(value) = get_env("DYN_ROUTER_POLICY_CONFIG") {
        config.router_policy_config = Some(value);
    }
    if let Some(value) = parse_bool(&get_env, "DYN_ROUTER_CONDITIONAL_DISAGG") {
        config.conditional_disagg_enabled = value;
    }
    if let Some(value) = get_env("DYN_ROUTER_CONDITIONAL_DISAGG_POLICY")
        && let Ok(policy) = value.parse()
    {
        config.conditional_disagg_policy = policy;
    }
    if let Some(value) = parse_usize(&get_env, "DYN_ROUTER_CONDITIONAL_DISAGG_EFF_ISL_THRESHOLD") {
        config.conditional_disagg_eff_isl_threshold = value;
    }
    if let Some(value) = parse_f64(
        &get_env,
        "DYN_ROUTER_CONDITIONAL_DISAGG_EFF_ISL_RATIO_THRESHOLD",
    ) {
        config.conditional_disagg_eff_isl_ratio_threshold = value;
    }
    if let Some(value) = parse_f64(
        &get_env,
        "DYN_ROUTER_CONDITIONAL_DISAGG_PREFILL_BUSY_THRESHOLD",
    ) {
        config.conditional_disagg_prefill_busy_threshold = Some(value);
    }
    if let Some(value) = parse_f64(
        &get_env,
        "DYN_ROUTER_CONDITIONAL_DISAGG_DECODE_BUSY_THRESHOLD",
    ) {
        config.conditional_disagg_decode_busy_threshold = Some(value);
    }
    if let Some(value) = parse_f64(&get_env, "DYN_ROUTER_PREDICTED_TTL_SECS") {
        config.router_predicted_ttl_secs = Some(value);
    }

    Ok(config)
}

fn apply_deprecated_overlap_score_weight_override_option(
    value: f64,
    overlap_score_credit: &mut Option<f64>,
    prefill_load_scale: &mut Option<f64>,
) {
    *prefill_load_scale = Some(value);
    if value == 0.0 {
        *overlap_score_credit = Some(0.0);
    }
}

/// Type of external shared KV cache to query during routing.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum SharedCacheType {
    /// No shared cache (default).
    #[default]
    None,
    /// HiCache L3 shared cache — queries sglang workers via the request plane.
    Hicache,
}

impl fmt::Display for SharedCacheType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::None => f.write_str("none"),
            Self::Hicache => f.write_str("hicache"),
        }
    }
}

impl FromStr for SharedCacheType {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "none" => Ok(Self::None),
            "hicache" => Ok(Self::Hicache),
            _ => Err(format!(
                "unknown shared_cache_type: {s:?}, expected 'none' or 'hicache'"
            )),
        }
    }
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum RouterQueuePolicy {
    #[default]
    Fcfs,
    Lcfs,
    Wspt,
}

impl fmt::Display for RouterQueuePolicy {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Fcfs => f.write_str("fcfs"),
            Self::Lcfs => f.write_str("lcfs"),
            Self::Wspt => f.write_str("wspt"),
        }
    }
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum RouterPrefillLoadModel {
    #[default]
    None,
    Aic,
}

impl fmt::Display for RouterPrefillLoadModel {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::None => f.write_str("none"),
            Self::Aic => f.write_str("aic"),
        }
    }
}

impl FromStr for RouterPrefillLoadModel {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "none" => Ok(Self::None),
            "aic" => Ok(Self::Aic),
            _ => Err(format!(
                "unknown prefill load model: {s:?}, expected 'none' or 'aic'"
            )),
        }
    }
}

impl RouterPrefillLoadModel {
    pub fn is_enabled(self) -> bool {
        !matches!(self, Self::None)
    }
}

/// Which conditional-disagg bypass policy to run.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ConditionalDisaggPolicyKind {
    /// Bypass when effective ISL is below both the absolute and ratio thresholds.
    #[default]
    IslBounding,
    /// Bypass when the chosen prefill worker is over the prefill-busy line.
    PrefillLoad,
    /// Bypass when either `isl_bounding` or `prefill_load` would bypass.
    IslOrLoad,
}

impl fmt::Display for ConditionalDisaggPolicyKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::IslBounding => f.write_str("isl_bounding"),
            Self::PrefillLoad => f.write_str("prefill_load"),
            Self::IslOrLoad => f.write_str("isl_or_load"),
        }
    }
}

impl FromStr for ConditionalDisaggPolicyKind {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "isl_bounding" => Ok(Self::IslBounding),
            "prefill_load" => Ok(Self::PrefillLoad),
            "isl_or_load" => Ok(Self::IslOrLoad),
            _ => Err(format!(
                "unknown conditional_disagg_policy: {s:?}, expected 'isl_bounding', 'prefill_load', or 'isl_or_load'"
            )),
        }
    }
}

impl FromStr for RouterQueuePolicy {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "fcfs" => Ok(Self::Fcfs),
            "lcfs" => Ok(Self::Lcfs),
            "wspt" => Ok(Self::Wspt),
            _ => Err(format!(
                "unknown queue policy: {s:?}, expected 'fcfs', 'lcfs', or 'wspt'"
            )),
        }
    }
}

#[derive(Debug, Clone, Default, Deserialize)]
struct RouterConfigOverrideSerde {
    overlap_score_credit: Option<f64>,
    prefill_load_scale: Option<f64>,
    overlap_score_weight: Option<f64>,
    router_temperature: Option<f64>,
    assume_kv_reuse: Option<bool>,
    track_prefill_tokens: Option<bool>,
    shared_cache_multiplier: Option<f64>,
}

/// Override configuration for router settings that can be specified per-request
#[derive(Debug, Clone, Default, Builder, Serialize, Deserialize)]
#[serde(try_from = "RouterConfigOverrideSerde")]
pub struct RouterConfigOverride {
    /// Device-local prefix-overlap credit multiplier applied to the prefill
    /// load before sampling. Values must be finite and non-negative. Values above
    /// 1.0 give device overlap extra credit. Set to 0.0 to ignore prefix matching.
    #[builder(default)]
    pub overlap_score_credit: Option<f64>,

    /// Scale applied to the adjusted prefill load after device/lower-tier
    /// cache-hit credits have been subtracted.
    #[builder(default)]
    pub prefill_load_scale: Option<f64>,

    #[builder(default)]
    pub router_temperature: Option<f64>,

    #[builder(default)]
    pub assume_kv_reuse: Option<bool>,

    #[builder(default)]
    pub track_prefill_tokens: Option<bool>,

    /// Per-request override of `shared_cache_multiplier`.
    #[builder(default)]
    pub shared_cache_multiplier: Option<f64>,
}

impl RouterConfigOverride {
    pub fn validate(&self) -> Result<(), String> {
        if let Some(value) = self.overlap_score_credit {
            validate_overlap_score_credit(value)?;
        }
        if let Some(value) = self.prefill_load_scale {
            validate_min("prefill_load_scale", value, 0.0)?;
        }
        if let Some(value) = self.router_temperature {
            validate_min("router_temperature", value, 0.0)?;
        }
        if let Some(value) = self.shared_cache_multiplier {
            validate_range("shared_cache_multiplier", value, 0.0, 1.0)?;
        }
        Ok(())
    }
}

impl TryFrom<RouterConfigOverrideSerde> for RouterConfigOverride {
    type Error = String;

    fn try_from(compat: RouterConfigOverrideSerde) -> Result<Self, Self::Error> {
        let mut overlap_score_credit = compat.overlap_score_credit;
        let mut prefill_load_scale = compat.prefill_load_scale;

        if let Some(overlap_score_weight) = compat.overlap_score_weight {
            apply_deprecated_overlap_score_weight_override_option(
                overlap_score_weight,
                &mut overlap_score_credit,
                &mut prefill_load_scale,
            );
        }

        let config = Self {
            overlap_score_credit,
            prefill_load_scale,
            router_temperature: compat.router_temperature,
            assume_kv_reuse: compat.assume_kv_reuse,
            track_prefill_tokens: compat.track_prefill_tokens,
            shared_cache_multiplier: compat.shared_cache_multiplier,
        };
        config.validate()?;
        Ok(config)
    }
}

#[derive(Debug, Clone, Deserialize)]
#[serde(default, deny_unknown_fields)]
struct KvRouterConfigSerde {
    overlap_score_credit: f64,
    overlap_score_credit_decay: f64,
    prefill_load_scale: f64,
    overlap_score_weight: Option<f64>,
    host_cache_hit_weight: f64,
    disk_cache_hit_weight: f64,
    router_temperature: f64,
    use_kv_events: bool,
    router_replica_sync: bool,
    router_track_active_blocks: bool,
    router_track_output_blocks: bool,
    router_assume_kv_reuse: bool,
    router_track_prefill_tokens: bool,
    router_tracking_hash: TrackingHashAlgorithm,
    router_tracking_key_file: Option<PathBuf>,
    router_tracking_key_id: Option<String>,
    router_prefill_load_model: RouterPrefillLoadModel,
    router_ttl_secs: f64,
    router_queue_threshold: Option<f64>,
    #[serde(default)]
    router_policy_config: Option<String>,
    router_event_threads: u32,
    skip_initial_worker_wait: bool,
    router_queue_policy: RouterQueuePolicy,
    use_remote_indexer: bool,
    serve_indexer: bool,
    shared_cache_multiplier: f64,
    shared_cache_type: SharedCacheType,
    router_predicted_ttl_secs: Option<f64>,
    conditional_disagg_enabled: bool,
    conditional_disagg_policy: ConditionalDisaggPolicyKind,
    conditional_disagg_eff_isl_threshold: usize,
    conditional_disagg_eff_isl_ratio_threshold: f64,
    #[serde(default)]
    conditional_disagg_prefill_busy_threshold: Option<f64>,
    #[serde(default)]
    conditional_disagg_decode_busy_threshold: Option<f64>,
}

impl Default for KvRouterConfigSerde {
    fn default() -> Self {
        let config = KvRouterConfig::default();
        Self {
            overlap_score_credit: config.overlap_score_credit,
            overlap_score_credit_decay: config.overlap_score_credit_decay,
            prefill_load_scale: config.prefill_load_scale,
            overlap_score_weight: None,
            host_cache_hit_weight: config.host_cache_hit_weight,
            disk_cache_hit_weight: config.disk_cache_hit_weight,
            router_temperature: config.router_temperature,
            use_kv_events: config.use_kv_events,
            router_replica_sync: config.router_replica_sync,
            router_track_active_blocks: config.router_track_active_blocks,
            router_track_output_blocks: config.router_track_output_blocks,
            router_assume_kv_reuse: config.router_assume_kv_reuse,
            router_track_prefill_tokens: config.router_track_prefill_tokens,
            router_tracking_hash: config.router_tracking_hash,
            router_tracking_key_file: config.router_tracking_key_file,
            router_tracking_key_id: config.router_tracking_key_id,
            router_prefill_load_model: config.router_prefill_load_model,
            router_ttl_secs: config.router_ttl_secs,
            router_queue_threshold: config.router_queue_threshold,
            router_policy_config: config.router_policy_config,
            router_event_threads: config.router_event_threads,
            skip_initial_worker_wait: config.skip_initial_worker_wait,
            router_queue_policy: config.router_queue_policy,
            use_remote_indexer: config.use_remote_indexer,
            serve_indexer: config.serve_indexer,
            shared_cache_multiplier: config.shared_cache_multiplier,
            shared_cache_type: config.shared_cache_type,
            router_predicted_ttl_secs: config.router_predicted_ttl_secs,
            conditional_disagg_enabled: config.conditional_disagg_enabled,
            conditional_disagg_policy: config.conditional_disagg_policy,
            conditional_disagg_eff_isl_threshold: config.conditional_disagg_eff_isl_threshold,
            conditional_disagg_eff_isl_ratio_threshold: config
                .conditional_disagg_eff_isl_ratio_threshold,
            conditional_disagg_prefill_busy_threshold: config
                .conditional_disagg_prefill_busy_threshold,
            conditional_disagg_decode_busy_threshold: config
                .conditional_disagg_decode_busy_threshold,
        }
    }
}

/// KV Router configuration parameters
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(try_from = "KvRouterConfigSerde")]
pub struct KvRouterConfig {
    /// Device-local prefix-overlap credit multiplier applied to the prefill
    /// load before sampling. Values must be finite and non-negative. Values above
    /// 1.0 give device overlap extra credit. Set to 0.0 to ignore prefix matching.
    pub overlap_score_credit: f64,

    /// Decay rate for device-local overlap credit as active prefill load rises
    /// above the least-loaded eligible worker. A value of 0.0 disables decay.
    #[serde(default = "default_overlap_score_credit_decay")]
    pub overlap_score_credit_decay: f64,

    /// Scale applied after overlap/cache-hit credits reduce the prompt-side
    /// prefill load. Defaults to 1.0.
    pub prefill_load_scale: f64,

    #[serde(default = "default_host_cache_hit_weight")]
    pub host_cache_hit_weight: f64,

    #[serde(default = "default_disk_cache_hit_weight")]
    pub disk_cache_hit_weight: f64,

    pub router_temperature: f64,

    pub use_kv_events: bool,

    pub router_replica_sync: bool,

    /// Whether to track active blocks in the router (default: true)
    pub router_track_active_blocks: bool,

    /// Whether to track output blocks during generation (default: false)
    /// When enabled, the router adds placeholder blocks as tokens are generated
    /// and applies fractional decay based on progress toward agent_hints.osl.
    pub router_track_output_blocks: bool,

    /// Whether to assume KV cache reuse when tracking active blocks (default: true).
    /// When true, computes actual block hashes for sequence tracking.
    /// When false, generates random hashes (assuming no KV cache reuse).
    pub router_assume_kv_reuse: bool,

    /// Whether to include prompt-side prefill tokens in active load accounting (default: true).
    /// When false, prompt tokens are excluded from active prefill token tracking, queue pressure,
    /// and potential prefill-token load calculations.
    #[serde(default = "default_track_prefill_tokens")]
    pub router_track_prefill_tokens: bool,

    /// Hash algorithm used for router-derived active-sequence identities.
    pub router_tracking_hash: TrackingHashAlgorithm,

    /// File containing the 32-byte provider key used by keyed tracking mode.
    pub router_tracking_key_file: Option<PathBuf>,

    /// Provider-managed epoch identifier mixed into keyed tracking scope derivation.
    pub router_tracking_key_id: Option<String>,

    /// Optional model for estimating effective prompt-side prefill load over time.
    pub router_prefill_load_model: RouterPrefillLoadModel,

    /// TTL for blocks in seconds (only used when use_kv_events is false, default: 120.0)
    pub router_ttl_secs: f64,

    /// Queue threshold fraction for prefill token capacity.
    /// When set, requests are queued if all workers exceed this fraction of max_num_batched_tokens.
    /// If None, queueing is disabled and all requests go directly to ready.
    /// Disabled by default. Must be >= 0. Use 0.0 for maximum queueing sensitivity.
    pub router_queue_threshold: Option<f64>,

    /// Optional startup-only YAML policy-class configuration.
    #[serde(default)]
    pub router_policy_config: Option<String>,

    /// Run-level model selector used by offline and online replay.
    #[serde(skip)]
    #[doc(hidden)]
    pub policy_model_name: Option<String>,

    /// Parsed startup policy document. This prevents per-model file reloads.
    #[serde(skip)]
    #[doc(hidden)]
    pub policy_config_cache: OnceLock<super::policy_config::RouterPolicyConfig>,

    /// Number of KV indexer worker threads.
    /// When > 1, uses ConcurrentRadixTree with a thread pool for event-driven
    /// and approximate routing writes. Default: 4.
    pub router_event_threads: u32,

    pub skip_initial_worker_wait: bool,

    /// Scheduling policy for the router queue.
    /// "fcfs" (default): first-come first-served with priority bumps — optimizes tail TTFT.
    /// "wspt": weighted shortest processing time (Smith's rule) — optimizes average TTFT.
    pub router_queue_policy: RouterQueuePolicy,

    /// Whether to query a remote KV indexer served from the worker component
    /// instead of maintaining a local radix tree for overlap scoring.
    #[serde(default)]
    pub use_remote_indexer: bool,

    /// Whether this router should serve its local indexer from the worker component.
    #[serde(default)]
    pub serve_indexer: bool,

    /// Multiplier for shared cache hits when scoring workers (0.0 to 1.0).
    /// Blocks available in the shared cache are less valuable than device-local blocks
    /// because they need to be fetched. A value of 0.5 means each shared cache hit
    /// counts as half a device-local hit. Default: 0.0 (shared cache scoring disabled);
    /// the CLI sets this to 0.5 when shared cache is enabled.
    pub shared_cache_multiplier: f64,

    /// Type of external shared KV cache to query during routing.
    /// "none" (default): disabled. "hicache": query sglang workers for L3 cache state.
    pub shared_cache_type: SharedCacheType,

    /// TTL in seconds applied to entries in the local predict-on-route side
    /// indexer. `None` disables predict-on-route. A value requires
    /// `use_kv_events=true` and enables a secondary approximate indexer
    /// populated by routing decisions; `find_matches` queries both the
    /// event-driven primary and local side indexer and returns the per-worker
    /// maximum overlap.
    #[serde(default)]
    pub router_predicted_ttl_secs: Option<f64>,

    /// Enable conditional-disagg bypass. When true, the `PrefillRouter`
    /// may short-circuit selected requests to prefill+decode on a decode worker.
    #[serde(default)]
    pub conditional_disagg_enabled: bool,

    /// Which conditional-disagg policy to run.
    #[serde(default)]
    pub conditional_disagg_policy: ConditionalDisaggPolicyKind,

    /// `IslBoundingPolicy` absolute effective-ISL cutoff in tokens.
    #[serde(default = "default_conditional_disagg_eff_isl_threshold")]
    pub conditional_disagg_eff_isl_threshold: usize,

    /// `IslBoundingPolicy` effective-ISL/prompt-token ratio cutoff.
    #[serde(default = "default_conditional_disagg_eff_isl_ratio_threshold")]
    pub conditional_disagg_eff_isl_ratio_threshold: f64,

    /// `PrefillLoadPolicy` busy-line fraction for the chosen prefill worker.
    /// When unset, the prefill-load condition falls back to `router_queue_threshold`.
    #[serde(default)]
    pub conditional_disagg_prefill_busy_threshold: Option<f64>,

    /// Decode-busy guard fraction for the chosen decode worker. When unset,
    /// the guard is disabled.
    #[serde(default)]
    pub conditional_disagg_decode_busy_threshold: Option<f64>,
}

fn default_conditional_disagg_eff_isl_threshold() -> usize {
    crate::conditional_disagg::DEFAULT_CONDITIONAL_DISAGG_EFF_ISL_THRESHOLD
}

fn default_conditional_disagg_eff_isl_ratio_threshold() -> f64 {
    crate::conditional_disagg::DEFAULT_CONDITIONAL_DISAGG_EFF_ISL_RATIO_THRESHOLD
}

impl Default for KvRouterConfig {
    fn default() -> Self {
        Self {
            overlap_score_credit: 1.0,
            overlap_score_credit_decay: default_overlap_score_credit_decay(),
            prefill_load_scale: default_prefill_load_scale(),
            host_cache_hit_weight: default_host_cache_hit_weight(),
            disk_cache_hit_weight: default_disk_cache_hit_weight(),
            router_temperature: 0.0,
            use_kv_events: true,
            router_replica_sync: false,
            router_track_active_blocks: true,
            router_track_output_blocks: false,
            router_assume_kv_reuse: true,
            router_track_prefill_tokens: default_track_prefill_tokens(),
            router_tracking_hash: TrackingHashAlgorithm::default(),
            router_tracking_key_file: None,
            router_tracking_key_id: None,
            router_prefill_load_model: RouterPrefillLoadModel::default(),
            router_ttl_secs: 120.0,
            router_queue_threshold: None,
            router_policy_config: None,
            policy_model_name: None,
            policy_config_cache: OnceLock::new(),
            router_event_threads: 4,
            skip_initial_worker_wait: false,
            router_queue_policy: RouterQueuePolicy::default(),
            use_remote_indexer: false,
            serve_indexer: false,
            shared_cache_multiplier: 0.0,
            shared_cache_type: SharedCacheType::default(),
            router_predicted_ttl_secs: None,
            conditional_disagg_enabled: false,
            conditional_disagg_policy: ConditionalDisaggPolicyKind::default(),
            conditional_disagg_eff_isl_threshold: default_conditional_disagg_eff_isl_threshold(),
            conditional_disagg_eff_isl_ratio_threshold:
                default_conditional_disagg_eff_isl_ratio_threshold(),
            conditional_disagg_prefill_busy_threshold: None,
            conditional_disagg_decode_busy_threshold: None,
        }
    }
}

impl TryFrom<KvRouterConfigSerde> for KvRouterConfig {
    type Error = String;

    fn try_from(compat: KvRouterConfigSerde) -> Result<Self, Self::Error> {
        let mut overlap_score_credit = compat.overlap_score_credit;
        let mut prefill_load_scale = compat.prefill_load_scale;

        if let Some(overlap_score_weight) = compat.overlap_score_weight {
            apply_deprecated_overlap_score_weight_override(
                overlap_score_weight,
                &mut overlap_score_credit,
                &mut prefill_load_scale,
            );
        }

        let config = Self {
            overlap_score_credit,
            overlap_score_credit_decay: compat.overlap_score_credit_decay,
            prefill_load_scale,
            host_cache_hit_weight: compat.host_cache_hit_weight,
            disk_cache_hit_weight: compat.disk_cache_hit_weight,
            router_temperature: compat.router_temperature,
            use_kv_events: compat.use_kv_events,
            router_replica_sync: compat.router_replica_sync,
            router_track_active_blocks: compat.router_track_active_blocks,
            router_track_output_blocks: compat.router_track_output_blocks,
            router_assume_kv_reuse: compat.router_assume_kv_reuse,
            router_track_prefill_tokens: compat.router_track_prefill_tokens,
            router_tracking_hash: compat.router_tracking_hash,
            router_tracking_key_file: compat.router_tracking_key_file,
            router_tracking_key_id: compat.router_tracking_key_id,
            router_prefill_load_model: compat.router_prefill_load_model,
            router_ttl_secs: compat.router_ttl_secs,
            router_queue_threshold: compat.router_queue_threshold,
            router_policy_config: compat.router_policy_config,
            policy_model_name: None,
            policy_config_cache: OnceLock::new(),
            router_event_threads: compat.router_event_threads,
            skip_initial_worker_wait: compat.skip_initial_worker_wait,
            router_queue_policy: compat.router_queue_policy,
            use_remote_indexer: compat.use_remote_indexer,
            serve_indexer: compat.serve_indexer,
            shared_cache_multiplier: compat.shared_cache_multiplier,
            shared_cache_type: compat.shared_cache_type,
            router_predicted_ttl_secs: compat.router_predicted_ttl_secs,
            conditional_disagg_enabled: compat.conditional_disagg_enabled,
            conditional_disagg_policy: compat.conditional_disagg_policy,
            conditional_disagg_eff_isl_threshold: compat.conditional_disagg_eff_isl_threshold,
            conditional_disagg_eff_isl_ratio_threshold: compat
                .conditional_disagg_eff_isl_ratio_threshold,
            conditional_disagg_prefill_busy_threshold: compat
                .conditional_disagg_prefill_busy_threshold,
            conditional_disagg_decode_busy_threshold: compat
                .conditional_disagg_decode_busy_threshold,
        };
        config.validate()?;
        Ok(config)
    }
}

fn validate_kv_router_config(config: &KvRouterConfig) -> Result<(), String> {
    validate_tracking_hash_options(
        config.router_tracking_hash,
        config.router_tracking_key_file.is_some(),
        config.router_tracking_key_id.as_deref(),
    )?;
    if config.router_track_output_blocks && !config.router_track_active_blocks {
        return Err(
            "router_track_output_blocks requires router_track_active_blocks=true".to_string(),
        );
    }
    if config.router_prefill_load_model.is_enabled() && !config.router_track_prefill_tokens {
        return Err(
            "router_prefill_load_model requires router_track_prefill_tokens=true".to_string(),
        );
    }
    if config.use_remote_indexer && config.serve_indexer {
        return Err("use_remote_indexer and serve_indexer are mutually exclusive".to_string());
    }
    if config.serve_indexer && config.overlap_score_credit == 0.0 {
        return Err("serve_indexer requires overlap_score_credit > 0".to_string());
    }
    if config.router_predicted_ttl_secs.is_some() && !config.use_kv_events {
        return Err("router_predicted_ttl_secs requires use_kv_events=true".to_string());
    }
    if config.conditional_disagg_enabled
        && matches!(
            config.conditional_disagg_policy,
            ConditionalDisaggPolicyKind::PrefillLoad | ConditionalDisaggPolicyKind::IslOrLoad,
        )
    {
        match (
            config.conditional_disagg_prefill_busy_threshold,
            config.router_queue_threshold,
        ) {
            (Some(threshold), _) => {
                tracing::info!(
                    busy_threshold = threshold,
                    "conditional_disagg prefill-load condition using --router-conditional-disagg-prefill-busy-threshold"
                );
            }
            (None, Some(threshold)) => {
                tracing::info!(
                    inherited_threshold = threshold,
                    "conditional_disagg prefill-load condition using --router-queue-threshold because --router-conditional-disagg-prefill-busy-threshold is unset"
                );
            }
            (None, None) => {
                tracing::warn!(
                    policy = ?config.conditional_disagg_policy,
                    "conditional_disagg prefill-load condition disabled: set --router-conditional-disagg-prefill-busy-threshold or --router-queue-threshold, or use policy=isl_bounding"
                );
            }
        }
    }
    if config.conditional_disagg_enabled
        && let Some(threshold) = config.conditional_disagg_decode_busy_threshold
    {
        tracing::info!(
            decode_busy_threshold = threshold,
            "conditional_disagg decode-busy guard enabled: bypass is disabled when the selected decode worker's projected decode load exceeds this fraction of KV capacity"
        );
    }
    if let Err(error) = config.loaded_policy_config() {
        return Err(format!("router_policy_config: {error}"));
    }
    Ok(())
}

impl KvRouterConfig {
    fn loaded_policy_config(
        &self,
    ) -> Result<
        Option<&super::policy_config::RouterPolicyConfig>,
        super::policy_config::RouterPolicyConfigError,
    > {
        let Some(path) = self.router_policy_config.as_deref() else {
            return Ok(None);
        };
        if self.policy_config_cache.get().is_none() {
            let parsed = super::policy_config::RouterPolicyConfig::from_path(path)?;
            let _ = self.policy_config_cache.set(parsed);
        }
        Ok(self.policy_config_cache.get())
    }

    pub fn policy_profile(
        &self,
        model_name: Option<&str>,
    ) -> Result<super::policy_config::PolicyProfile, super::policy_config::RouterPolicyConfigError>
    {
        let Some(policy_config) = self.loaded_policy_config()? else {
            return Ok(super::policy_config::PolicyProfile::synthetic(
                self.router_queue_threshold,
                self.router_queue_policy,
            ));
        };
        Ok(policy_config.resolve_profile(
            model_name,
            self.router_queue_threshold,
            self.router_queue_policy,
        ))
    }

    pub fn with_policy_model_name(mut self, model_name: Option<String>) -> Self {
        self.policy_model_name = model_name;
        self
    }

    pub fn configured_policy_profile(
        &self,
    ) -> Result<super::policy_config::PolicyProfile, super::policy_config::RouterPolicyConfigError>
    {
        self.policy_profile(self.policy_model_name.as_deref())
    }

    pub fn validate_config(&self) -> Result<(), String> {
        self.validate()
    }

    pub fn validate(&self) -> Result<(), String> {
        validate_overlap_score_credit(self.overlap_score_credit)?;
        validate_min(
            "overlap_score_credit_decay",
            self.overlap_score_credit_decay,
            0.0,
        )?;
        validate_min("prefill_load_scale", self.prefill_load_scale, 0.0)?;
        validate_range(
            "host_cache_hit_weight",
            self.host_cache_hit_weight,
            0.0,
            1.0,
        )?;
        validate_range(
            "disk_cache_hit_weight",
            self.disk_cache_hit_weight,
            0.0,
            1.0,
        )?;
        validate_min("router_temperature", self.router_temperature, 0.0)?;
        validate_min("router_ttl_secs", self.router_ttl_secs, 0.0)?;
        if let Some(value) = self.router_queue_threshold {
            validate_min("router_queue_threshold", value, 0.0)?;
        }
        if self.router_event_threads == 0 {
            return Err("router_event_threads must be at least 1".to_string());
        }
        validate_range(
            "shared_cache_multiplier",
            self.shared_cache_multiplier,
            0.0,
            1.0,
        )?;
        if let Some(value) = self.router_predicted_ttl_secs {
            validate_min("router_predicted_ttl_secs", value, 0.0)?;
        }
        validate_range(
            "conditional_disagg_eff_isl_ratio_threshold",
            self.conditional_disagg_eff_isl_ratio_threshold,
            0.0,
            1.0,
        )?;
        if let Some(value) = self.conditional_disagg_prefill_busy_threshold {
            validate_min("conditional_disagg_prefill_busy_threshold", value, 0.0)?;
        }
        if let Some(value) = self.conditional_disagg_decode_busy_threshold {
            validate_min("conditional_disagg_decode_busy_threshold", value, 0.0)?;
        }
        validate_kv_router_config(self)
    }

    pub fn router_queue_recheck_interval(&self) -> Duration {
        const DEFAULT_RECHECK_INTERVAL: Duration = Duration::from_secs(60);
        const PREFILL_LOAD_RECHECK_INTERVAL: Duration = Duration::from_millis(100);

        if self.router_prefill_load_model.is_enabled()
            && (self.router_policy_config.is_some() || self.router_queue_threshold.is_some())
        {
            return PREFILL_LOAD_RECHECK_INTERVAL;
        }

        DEFAULT_RECHECK_INTERVAL
    }

    pub fn predict_on_route_enabled(&self) -> bool {
        self.router_predicted_ttl_secs.is_some()
    }

    pub fn queueing_enabled(
        &self,
        model_name: Option<&str>,
    ) -> Result<bool, super::policy_config::RouterPolicyConfigError> {
        Ok(self
            .policy_profile(model_name)?
            .classes()
            .iter()
            .any(super::policy_config::PolicyClassConfig::queueing_enabled))
    }

    pub fn assume_kv_reuse(&self, config_override: Option<&RouterConfigOverride>) -> bool {
        config_override
            .and_then(|cfg| cfg.assume_kv_reuse)
            .unwrap_or(self.router_assume_kv_reuse)
    }

    pub fn track_prefill_tokens(&self, config_override: Option<&RouterConfigOverride>) -> bool {
        config_override
            .and_then(|cfg| cfg.track_prefill_tokens)
            .unwrap_or(self.router_track_prefill_tokens)
    }

    /// Compute sequence hashes for active block tracking based on configuration.
    ///
    /// Returns:
    /// - `None` if `router_track_active_blocks` is false
    /// - Random hashes if `router_track_active_blocks` is true but `router_assume_kv_reuse` is false
    /// - Actual sequence hashes if both are true
    /// # Panics
    ///
    /// Panics in keyed mode because the legacy interface has no initialized
    /// [`TrackingHashContext`]. Keyed callers must use
    /// [`Self::compute_seq_hashes_for_tracking_with_context`].
    pub fn compute_seq_hashes_for_tracking(
        &self,
        tokens: &[u32],
        block_size: u32,
        config_override: Option<&RouterConfigOverride>,
        hash_options: BlockHashOptions<'_>,
        precomputed_block_hashes: Option<&[LocalBlockHash]>,
    ) -> Option<Vec<u64>> {
        assert_eq!(
            self.router_tracking_hash,
            TrackingHashAlgorithm::PublicXxh3V1,
            "compute_seq_hashes_for_tracking cannot be used with keyed tracking; initialize a TrackingHashContext and call compute_seq_hashes_for_tracking_with_context"
        );

        if !self.router_track_active_blocks {
            return None;
        }

        let num_blocks = complete_block_count(
            tokens.len(),
            block_size,
            hash_options.is_eagle.unwrap_or(false),
        );
        if num_blocks == 0 {
            return Some(Vec::new());
        }

        if self.assume_kv_reuse(config_override) {
            let block_hashes = match precomputed_block_hashes {
                Some(block_hashes) => block_hashes,
                None => {
                    let computed = compute_block_hash_for_seq(tokens, block_size, hash_options);
                    return Some(compute_seq_hash_for_block(&computed));
                }
            };
            Some(compute_seq_hash_for_block(block_hashes))
        } else {
            Some(random_sequence_hashes(num_blocks))
        }
    }

    /// Compute sequence hashes with a router-initialized tracking-hash context.
    pub fn compute_seq_hashes_for_tracking_with_context(
        &self,
        tracking_hash: &TrackingHashContext,
        scope: TrackingHashScope<'_>,
        tokens: &[u32],
        config_override: Option<&RouterConfigOverride>,
        hash_options: BlockHashOptions<'_>,
        precomputed_block_hashes: Option<&[LocalBlockHash]>,
    ) -> Option<Vec<u64>> {
        assert_eq!(
            tracking_hash.algorithm(),
            self.router_tracking_hash,
            "tracking hash context must match KvRouterConfig"
        );
        if !self.router_track_active_blocks {
            return None;
        }

        let assume_kv_reuse = self.assume_kv_reuse(config_override);
        Some(tracking_hash.compute_sequence_hashes_for_tracking(
            scope,
            tokens,
            hash_options,
            assume_kv_reuse,
            precomputed_block_hashes,
        ))
    }

    /// Check if KV event subscription should be started.
    ///
    /// Returns false if:
    /// - KV events are disabled (`use_kv_events=false`)
    /// - Overlap scoring is disabled (`overlap_score_credit=0`)
    ///
    /// When false, the router skips starting the KV event subscription entirely,
    /// avoiding the need to query workers for their local indexer state.
    pub fn should_subscribe_to_kv_events(&self) -> bool {
        self.use_kv_events && self.overlap_score_credit > 0.0
    }
}

fn random_sequence_hashes(num_blocks: usize) -> Vec<u64> {
    (0..num_blocks).map(|_| fastrand::u64(..)).collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::identity::RoutingPartitionRef;
    use crate::protocols::{BlockExtraInfo, BlockMmObjectInfo, compute_seq_hash_for_block};
    use std::collections::HashMap;

    fn test_tracking_scope(block_size: u32) -> TrackingHashScope<'static> {
        TrackingHashScope {
            partition: RoutingPartitionRef::new("model", "default"),
            block_size,
        }
    }

    fn config_from_values(values: &[(&str, &str)]) -> KvRouterConfig {
        try_config_from_values(values).unwrap()
    }

    fn try_config_from_values(values: &[(&str, &str)]) -> Result<KvRouterConfig, String> {
        let values: HashMap<&str, &str> = values.iter().copied().collect();
        kv_router_config_from_lookup(|key| values.get(key).map(|value| (*value).to_string()))
    }

    #[test]
    fn dynamo_env_config_parses_canonical_settings() {
        let config = config_from_values(&[
            ("DYN_ROUTER_KV_OVERLAP_SCORE_CREDIT", "0.25"),
            ("DYN_ROUTER_KV_OVERLAP_SCORE_CREDIT_DECAY", "0.75"),
            ("DYN_ROUTER_PREFILL_LOAD_SCALE", "2.5"),
            ("DYN_ROUTER_TEMPERATURE", "0.7"),
            ("DYN_USE_KV_EVENTS", "false"),
            ("DYN_ROUTER_REPLICA_SYNC", "yes"),
            ("DYN_ROUTER_TRACK_ACTIVE_BLOCKS", "0"),
            ("DYN_ROUTER_TRACK_OUTPUT_BLOCKS", "on"),
            ("DYN_ROUTER_ASSUME_KV_REUSE", "false"),
            ("DYN_ROUTER_TRACK_PREFILL_TOKENS", "false"),
            ("DYN_ROUTER_TRACKING_HASH", "keyed-xxh3-v1"),
            (
                "DYN_ROUTER_TRACKING_KEY_FILE",
                "/run/secrets/dynamo/tracking-key",
            ),
            ("DYN_ROUTER_TRACKING_KEY_ID", "2026-01"),
            ("DYN_ROUTER_QUEUE_THRESHOLD", "4.5"),
        ]);

        assert_eq!(config.overlap_score_credit, 0.25);
        assert_eq!(config.overlap_score_credit_decay, 0.75);
        assert_eq!(config.prefill_load_scale, 2.5);
        assert_eq!(config.router_temperature, 0.7);
        assert!(!config.use_kv_events);
        assert!(config.router_replica_sync);
        assert!(!config.router_track_active_blocks);
        assert!(config.router_track_output_blocks);
        assert!(!config.router_assume_kv_reuse);
        assert!(!config.router_track_prefill_tokens);
        assert_eq!(
            config.router_tracking_hash,
            TrackingHashAlgorithm::KeyedXxh3V1
        );
        assert_eq!(
            config.router_tracking_key_file,
            Some(PathBuf::from("/run/secrets/dynamo/tracking-key"))
        );
        assert_eq!(config.router_tracking_key_id.as_deref(), Some("2026-01"));
        assert_eq!(config.router_queue_threshold, Some(4.5));

        let predicted = config_from_values(&[("DYN_ROUTER_PREDICTED_TTL_SECS", "60")]);
        assert_eq!(predicted.router_predicted_ttl_secs, Some(60.0));
        assert!(predicted.validate_config().is_ok());
    }

    #[test]
    fn dynamo_env_config_preserves_deprecated_alias_precedence() {
        let config = config_from_values(&[
            ("DYN_ROUTER_KV_OVERLAP_SCORE_CREDIT", "0.25"),
            ("DYN_ROUTER_PREFILL_LOAD_SCALE", "2"),
            ("DYN_ROUTER_KV_OVERLAP_SCORE_WEIGHT", "3"),
            ("DYN_OVERLAP_SCORE_WEIGHT", "4"),
        ]);

        assert_eq!(config.overlap_score_credit, 0.25);
        assert_eq!(config.prefill_load_scale, 3.0);

        let disabled = config_from_values(&[
            ("DYN_ROUTER_KV_OVERLAP_SCORE_CREDIT", "0.75"),
            ("DYN_ROUTER_KV_OVERLAP_SCORE_WEIGHT", "0"),
        ]);
        assert_eq!(disabled.overlap_score_credit, 0.0);
        assert_eq!(disabled.prefill_load_scale, 0.0);
    }

    #[test]
    fn dynamo_env_config_ignores_unparseable_values_and_validates_ranges() {
        let unparseable = config_from_values(&[
            ("DYN_ROUTER_TEMPERATURE", "warm"),
            ("DYN_ROUTER_TRACK_ACTIVE_BLOCKS", "sometimes"),
        ]);
        let default = KvRouterConfig::default();
        assert_eq!(unparseable.router_temperature, default.router_temperature);
        assert_eq!(
            unparseable.router_track_active_blocks,
            default.router_track_active_blocks
        );

        let amplified = config_from_values(&[("DYN_ROUTER_KV_OVERLAP_SCORE_CREDIT", "1.5")]);
        assert!(amplified.validate_config().is_ok());

        for value in ["-0.5", "NaN", "inf"] {
            let invalid_credit =
                config_from_values(&[("DYN_ROUTER_KV_OVERLAP_SCORE_CREDIT", value)]);
            assert!(invalid_credit.validate_config().is_err());
        }

        let error = try_config_from_values(&[("DYN_ROUTER_TRACKING_HASH", "mystery")]).unwrap_err();
        assert!(error.contains("public-xxh3-v1 or keyed-xxh3-v1"));

        assert!(serde_json::to_string(&config_from_values(&[])).is_ok());
    }

    #[test]
    fn compute_seq_hashes_for_tracking_uses_mm_hashes() {
        let cfg = KvRouterConfig::default();
        let tokens = vec![1, 2, 3, 4];
        let mm_infos = vec![
            Some(BlockExtraInfo {
                mm_objects: vec![BlockMmObjectInfo {
                    mm_hash: 42,
                    offsets: vec![],
                }],
            }),
            None,
        ];

        let without_mm = cfg
            .compute_seq_hashes_for_tracking(&tokens, 2, None, BlockHashOptions::default(), None)
            .unwrap();
        let with_mm = cfg
            .compute_seq_hashes_for_tracking(
                &tokens,
                2,
                None,
                BlockHashOptions {
                    block_mm_infos: Some(&mm_infos),
                    ..Default::default()
                },
                None,
            )
            .unwrap();

        assert_ne!(without_mm, with_mm);
    }

    #[test]
    fn compute_seq_hashes_for_tracking_uses_precomputed_block_hashes() {
        let config = KvRouterConfig::default();
        let tokens: Vec<u32> = (0..8).collect();
        let precomputed = vec![LocalBlockHash(11), LocalBlockHash(29)];

        let seq_hashes = config.compute_seq_hashes_for_tracking(
            &tokens,
            4,
            None,
            BlockHashOptions::default(),
            Some(&precomputed),
        );

        assert_eq!(seq_hashes, Some(compute_seq_hash_for_block(&precomputed)));
    }

    #[test]
    fn context_aware_tracking_matches_public_legacy_api() {
        let config = KvRouterConfig::default();
        let context = TrackingHashContext::from_config(&config).unwrap();
        let tokens: Vec<u32> = (0..8).collect();

        let legacy = config.compute_seq_hashes_for_tracking(
            &tokens,
            4,
            None,
            BlockHashOptions::default(),
            None,
        );
        let context_aware = config.compute_seq_hashes_for_tracking_with_context(
            &context,
            test_tracking_scope(4),
            &tokens,
            None,
            BlockHashOptions::default(),
            None,
        );

        assert_eq!(legacy, context_aware);
    }

    #[test]
    #[should_panic(expected = "cannot be used with keyed tracking")]
    fn legacy_tracking_api_does_not_fall_back_in_keyed_mode() {
        let config = KvRouterConfig {
            router_tracking_hash: TrackingHashAlgorithm::KeyedXxh3V1,
            ..Default::default()
        };

        let _ = config.compute_seq_hashes_for_tracking(
            &[1, 2, 3, 4],
            4,
            None,
            BlockHashOptions::default(),
            None,
        );
    }

    #[test]
    fn test_kv_router_config_rejects_out_of_range_shared_cache_multiplier() {
        let too_small = KvRouterConfig {
            shared_cache_multiplier: -0.1,
            ..Default::default()
        };
        let too_large = KvRouterConfig {
            shared_cache_multiplier: 1.1,
            ..Default::default()
        };

        assert!(too_small.validate().is_err());
        assert!(too_large.validate().is_err());
    }

    #[test]
    fn test_kv_router_config_rejects_local_approx_with_predicted_ttl() {
        let config = KvRouterConfig {
            use_kv_events: false,
            router_predicted_ttl_secs: Some(5.0),
            ..Default::default()
        };

        assert!(config.validate().is_err());
    }

    #[test]
    fn test_kv_router_config_rejects_remote_approx_with_predicted_ttl() {
        let config = KvRouterConfig {
            use_kv_events: false,
            use_remote_indexer: true,
            router_predicted_ttl_secs: Some(5.0),
            ..Default::default()
        };

        assert!(config.validate().is_err());
    }

    #[test]
    fn test_kv_router_config_allows_remote_events_with_predicted_ttl() {
        let config = KvRouterConfig {
            use_kv_events: true,
            use_remote_indexer: true,
            router_predicted_ttl_secs: Some(5.0),
            ..Default::default()
        };

        assert!(config.validate().is_ok());
    }

    #[test]
    fn test_kv_router_config_allows_served_events_with_predicted_ttl() {
        let config = KvRouterConfig {
            use_kv_events: true,
            serve_indexer: true,
            router_predicted_ttl_secs: Some(5.0),
            ..Default::default()
        };

        assert!(config.validate().is_ok());
    }

    #[test]
    fn test_kv_router_config_deserializes_policy_path() {
        let policy_file = tempfile::NamedTempFile::new().unwrap();
        std::fs::write(
            policy_file.path(),
            "default_policy_family: default\nuncached_isl_buckets:\n  - min_tokens: 0\n    bucket: all\npolicy_classes:\n  - name: default\n    policy_family: default\n    cache_bucket: all\n    quantum: 1\n",
        )
        .unwrap();
        let encoded = serde_json::json!({
            "router_policy_config": policy_file.path(),
        })
        .to_string();
        let config: KvRouterConfig = serde_json::from_str(&encoded).unwrap();

        assert_eq!(
            config.router_policy_config.as_deref(),
            Some(policy_file.path().to_str().unwrap())
        );
    }

    #[test]
    fn removed_missing_isl_queue_config_is_rejected_as_unknown() {
        for value in [
            serde_json::json!(null),
            serde_json::json!([]),
            serde_json::json!([{
                "missing_cache_tokens_floor": 0,
                "max_queue_depth": 1,
            }]),
        ] {
            let encoded = serde_json::json!({
                "router_queue_by_incoming_missing_isl": value,
            })
            .to_string();
            let error = serde_json::from_str::<KvRouterConfig>(&encoded).unwrap_err();
            let message = error.to_string();
            assert!(
                message.contains("unknown field `router_queue_by_incoming_missing_isl`"),
                "{message}"
            );
        }
    }

    #[test]
    fn policy_config_is_validated_and_cached_at_startup() {
        let policy_file = tempfile::NamedTempFile::new().unwrap();
        std::fs::write(
            policy_file.path(),
            "default_policy_family: stable\nuncached_isl_buckets:\n  - min_tokens: 0\n    bucket: all\npolicy_classes:\n  - name: stable\n    policy_family: stable\n    cache_bucket: all\n    quantum: 7\n",
        )
        .unwrap();
        let config = KvRouterConfig {
            router_policy_config: Some(policy_file.path().display().to_string()),
            ..Default::default()
        };

        config.validate_config().unwrap();
        std::fs::write(policy_file.path(), "not: [valid").unwrap();

        let profile = config.policy_profile(None).unwrap();
        assert_eq!(profile.default_class().name, "stable");
        assert_eq!(profile.default_class().quantum, 7);
    }

    #[test]
    fn invalid_policy_config_fails_config_validation() {
        let policy_file = tempfile::NamedTempFile::new().unwrap();
        std::fs::write(policy_file.path(), "not: [valid").unwrap();
        let config = KvRouterConfig {
            router_policy_config: Some(policy_file.path().display().to_string()),
            ..Default::default()
        };

        let error = config.validate_config().unwrap_err();
        assert!(
            error.contains(policy_file.path().to_str().unwrap()),
            "{error}"
        );
        assert!(
            error.contains("failed to parse router policy config"),
            "{error}"
        );
    }

    #[test]
    fn policy_config_uses_fast_recheck_with_prefill_load_model() {
        let config = KvRouterConfig {
            router_prefill_load_model: RouterPrefillLoadModel::Aic,
            router_policy_config: Some("/tmp/policy.yaml".to_string()),
            router_queue_threshold: None,
            ..Default::default()
        };

        assert_eq!(
            config.router_queue_recheck_interval(),
            Duration::from_millis(100)
        );
    }

    #[test]
    fn prefill_load_model_allows_wspt_policy_classes() {
        let config = KvRouterConfig {
            router_prefill_load_model: RouterPrefillLoadModel::Aic,
            router_queue_policy: RouterQueuePolicy::Wspt,
            ..Default::default()
        };

        assert!(config.validate().is_ok());
    }

    #[test]
    fn configured_policy_profile_uses_transient_replay_model_name() {
        let path = std::env::temp_dir().join(format!(
            "dynamo-router-policy-{}.yaml",
            uuid::Uuid::new_v4()
        ));
        std::fs::write(
            &path,
            r#"
default_policy_family: root
uncached_isl_buckets:
  - min_tokens: 0
    bucket: all
policy_classes:
  - name: root
    policy_family: root
    cache_bucket: all
    quantum: 1
models:
  replay-model:
    default_policy_family: selected
    uncached_isl_buckets:
      - min_tokens: 0
        bucket: all
    policy_classes:
      - name: selected
        policy_family: selected
        cache_bucket: all
        quantum: 9
"#,
        )
        .unwrap();
        let config = KvRouterConfig {
            router_policy_config: Some(path.display().to_string()),
            ..Default::default()
        }
        .with_policy_model_name(Some("replay-model".to_string()));

        let profile = config.configured_policy_profile().unwrap();
        assert_eq!(profile.default_class().name, "selected");
        assert_eq!(profile.default_class().quantum, 9);
        assert!(
            !serde_json::to_string(&config)
                .unwrap()
                .contains("replay-model")
        );

        std::fs::remove_file(path).unwrap();
    }

    #[test]
    fn test_kv_router_config_accepts_credit_above_one() {
        let amplified = KvRouterConfig {
            overlap_score_credit: 1.1,
            ..Default::default()
        };

        assert!(amplified.validate().is_ok());
        for value in [-0.1, f64::NAN, f64::INFINITY] {
            let invalid = KvRouterConfig {
                overlap_score_credit: value,
                ..Default::default()
            };
            assert!(invalid.validate().is_err());
        }
    }

    #[test]
    fn test_kv_router_config_maps_deprecated_overlap_weight_alias_to_prefill_scale() {
        let config: KvRouterConfig =
            serde_json::from_str(r#"{"overlap_score_weight":2.5}"#).unwrap();

        assert_eq!(config.overlap_score_credit, 1.0);
        assert_eq!(config.prefill_load_scale, 2.5);
    }

    #[test]
    fn test_kv_router_config_maps_deprecated_overlap_weight_zero_to_credit_zero() {
        let config: KvRouterConfig =
            serde_json::from_str(r#"{"overlap_score_weight":0.0}"#).unwrap();

        assert_eq!(config.overlap_score_credit, 0.0);
        assert_eq!(config.prefill_load_scale, 0.0);
        assert!(!config.should_subscribe_to_kv_events());
    }

    #[test]
    fn test_kv_router_config_deprecated_overlap_weight_overrides_canonical_fields() {
        let config: KvRouterConfig = serde_json::from_str(
            r#"{"overlap_score_weight":2.5,"overlap_score_credit":0.5,"prefill_load_scale":3.0}"#,
        )
        .unwrap();

        assert_eq!(config.overlap_score_credit, 0.5);
        assert_eq!(config.prefill_load_scale, 2.5);
    }

    #[test]
    fn test_kv_router_config_deprecated_overlap_weight_zero_overrides_credit() {
        let config: KvRouterConfig = serde_json::from_str(
            r#"{"overlap_score_weight":0.0,"overlap_score_credit":0.5,"prefill_load_scale":3.0}"#,
        )
        .unwrap();

        assert_eq!(config.overlap_score_credit, 0.0);
        assert_eq!(config.prefill_load_scale, 0.0);
    }

    #[test]
    fn test_kv_router_config_deserialize_accepts_credit_above_one() {
        let amplified: KvRouterConfig =
            serde_json::from_str(r#"{"overlap_score_credit":1.5}"#).unwrap();
        let credit_error =
            serde_json::from_str::<KvRouterConfig>(r#"{"overlap_score_credit":-0.1}"#)
                .unwrap_err()
                .to_string();
        let scale_error = serde_json::from_str::<KvRouterConfig>(r#"{"prefill_load_scale":-0.1}"#)
            .unwrap_err()
            .to_string();

        assert_eq!(amplified.overlap_score_credit, 1.5);
        assert!(credit_error.contains("overlap_score_credit"));
        assert!(scale_error.contains("prefill_load_scale"));
    }

    #[test]
    fn test_router_config_override_maps_deprecated_overlap_weight_alias_to_prefill_scale() {
        let config: RouterConfigOverride =
            serde_json::from_str(r#"{"overlap_score_weight":2.5}"#).unwrap();

        assert_eq!(config.overlap_score_credit, None);
        assert_eq!(config.prefill_load_scale, Some(2.5));
    }

    #[test]
    fn test_router_config_override_maps_deprecated_overlap_weight_zero_to_credit_zero() {
        let config: RouterConfigOverride =
            serde_json::from_str(r#"{"overlap_score_weight":0.0}"#).unwrap();

        assert_eq!(config.overlap_score_credit, Some(0.0));
        assert_eq!(config.prefill_load_scale, Some(0.0));
    }

    #[test]
    fn test_router_config_override_deprecated_overlap_weight_overrides_canonical_fields() {
        let config: RouterConfigOverride = serde_json::from_str(
            r#"{"overlap_score_weight":2.0,"overlap_score_credit":0.5,"prefill_load_scale":3.0}"#,
        )
        .unwrap();

        assert_eq!(config.overlap_score_credit, Some(0.5));
        assert_eq!(config.prefill_load_scale, Some(2.0));
    }

    #[test]
    fn test_router_config_override_deprecated_overlap_weight_zero_overrides_credit() {
        let config: RouterConfigOverride = serde_json::from_str(
            r#"{"overlap_score_weight":0.0,"overlap_score_credit":0.5,"prefill_load_scale":3.0}"#,
        )
        .unwrap();

        assert_eq!(config.overlap_score_credit, Some(0.0));
        assert_eq!(config.prefill_load_scale, Some(0.0));
    }

    #[test]
    fn test_router_config_override_deserialize_accepts_credit_above_one() {
        let amplified: RouterConfigOverride =
            serde_json::from_str(r#"{"overlap_score_credit":1.5}"#).unwrap();
        let credit_error =
            serde_json::from_str::<RouterConfigOverride>(r#"{"overlap_score_credit":-0.1}"#)
                .unwrap_err()
                .to_string();
        let scale_error =
            serde_json::from_str::<RouterConfigOverride>(r#"{"prefill_load_scale":-0.1}"#)
                .unwrap_err()
                .to_string();

        assert_eq!(amplified.overlap_score_credit, Some(1.5));
        assert!(credit_error.contains("overlap_score_credit"));
        assert!(scale_error.contains("prefill_load_scale"));
    }

    #[test]
    fn test_overlap_credit_zero_skips_kv_event_subscription() {
        let config = KvRouterConfig {
            overlap_score_credit: 0.0,
            use_kv_events: true,
            ..Default::default()
        };

        assert!(!config.should_subscribe_to_kv_events());
    }

    #[test]
    fn test_router_config_override_rejects_out_of_range_shared_cache_multiplier() {
        let too_small = RouterConfigOverride {
            overlap_score_credit: None,
            prefill_load_scale: None,
            router_temperature: None,
            assume_kv_reuse: None,
            track_prefill_tokens: None,
            shared_cache_multiplier: Some(-0.1),
        };
        let too_large = RouterConfigOverride {
            overlap_score_credit: None,
            prefill_load_scale: None,
            router_temperature: None,
            assume_kv_reuse: None,
            track_prefill_tokens: None,
            shared_cache_multiplier: Some(1.1),
        };

        assert!(too_small.validate().is_err());
        assert!(too_large.validate().is_err());
    }

    #[test]
    fn test_router_config_override_accepts_credit_above_one() {
        let amplified = RouterConfigOverride {
            overlap_score_credit: Some(1.1),
            prefill_load_scale: None,
            router_temperature: None,
            assume_kv_reuse: None,
            track_prefill_tokens: None,
            shared_cache_multiplier: None,
        };

        assert!(amplified.validate().is_ok());
        for value in [-0.1, f64::NAN, f64::INFINITY] {
            let invalid = RouterConfigOverride {
                overlap_score_credit: Some(value),
                prefill_load_scale: None,
                router_temperature: None,
                assume_kv_reuse: None,
                track_prefill_tokens: None,
                shared_cache_multiplier: None,
            };
            assert!(invalid.validate().is_err());
        }
    }

    #[test]
    fn test_kv_router_config_defaults_are_disabled() {
        assert_eq!(KvRouterConfig::default().router_queue_threshold, None);
        assert_eq!(KvRouterConfig::default().shared_cache_multiplier, 0.0);
    }

    #[test]
    fn queueing_enabled_reflects_synthetic_threshold() {
        // With default config, queueing is disabled.
        assert!(!KvRouterConfig::default().queueing_enabled(None).unwrap());
        let with_threshold = KvRouterConfig {
            router_queue_threshold: Some(0.5),
            ..Default::default()
        };
        // With a threshold set, queueing is enabled
        assert!(with_threshold.queueing_enabled(None).unwrap());
    }
}
