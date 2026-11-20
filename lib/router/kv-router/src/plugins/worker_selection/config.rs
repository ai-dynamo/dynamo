// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! YAML schema for process-wide worker-selection policy selections.
//!
//! Each worker role (`aggregated`, `prefill`, `decode`, `encode`) selects its policy inline:
//!
//! ```yaml
//! worker_selection:
//!   aggregated: dynamo-two-tier-cost-fn        # a policy type with default parameters
//!   decode:
//!     type: dynamo-default-cost-fn             # a policy type with parameters
//!     parameters:
//!       router_temperature: 0.5
//!     hints:                                   # optional: the role's hints, in run order
//!       - type: my_hint                        # any linked hint policy
//!     session_affinity: true                   # optional: keep a session on one worker
//!     busy_threshold:                          # optional: when a worker stops taking requests
//!       active_decode_blocks: 0.85
//! ```
//!
//! A role whose entry lists no `hints` runs its default hints; `hints: []` runs none.
//!
//! `session_affinity` is read by the router, not by the policy: `true` is a 300 s idle TTL in
//! `hard` mode, `{ttl_secs, mode, max_entries}` tunes it, and `false` turns it off for the role.
//!
//! `busy_threshold` is read by the router too: `{active_decode_blocks, active_prefill_tokens_frac}`
//! are the role's busy-worker thresholds, and `{}` sets none.
//!
//! The older `instances:` list, which named a `{type, parameters}` pair once and referenced it
//! from each role by name, is deprecated and still accepted.

use std::collections::HashMap;
use std::num::NonZeroUsize;
use std::time::Duration;

use dynamo_session_affinity::{
    MAX_SESSION_AFFINITY_TTL_SECS, SessionAffinity, SessionAffinityMode,
};
use serde::Deserialize;

use crate::plugins::hint_policy::{HintPolicyEntry, RawHintPolicyEntry, resolve_hints};
use crate::scheduling::policy_config::RouterPolicyConfigError;
use crate::scheduling::policy_config::validate_identifier;
use dynamo_router_protocol::WorkerType;

/// Process-wide configuration for worker selection.
#[derive(Debug, Clone, PartialEq)]
pub struct WorkerSelectionConfig {
    aggregated: Option<RoleSelection>,
    prefill: Option<RoleSelection>,
    decode: Option<RoleSelection>,
    encode: Option<RoleSelection>,
    instances: HashMap<String, WorkerSelectionInstance>,
}

/// One role's entry: the policy it selects, and the settings the router owns. The router's
/// settings sit beside the policy so that two roles with the same policy share one factory.
#[derive(Debug, Clone, PartialEq)]
struct RoleSelection {
    policy: WorkerSelectionPolicySelection,
    session_affinity: Option<SessionAffinitySetting>,
    busy_threshold: Option<BusyThresholdSpec>,
}

fn policy(role: &Option<RoleSelection>) -> Option<&WorkerSelectionPolicySelection> {
    role.as_ref().map(|role| &role.policy)
}

impl WorkerSelectionConfig {
    /// The policy selected for a full-request worker pool.
    pub fn aggregated(&self) -> Option<&WorkerSelectionPolicySelection> {
        policy(&self.aggregated)
    }

    pub(crate) fn prefill(&self) -> Option<&WorkerSelectionPolicySelection> {
        policy(&self.prefill)
    }

    pub(crate) fn decode(&self) -> Option<&WorkerSelectionPolicySelection> {
        policy(&self.decode)
    }

    pub(crate) fn encode(&self) -> Option<&WorkerSelectionPolicySelection> {
        policy(&self.encode)
    }

    /// What a role's entry says about session affinity; `None` when it does not say. An
    /// `encode` entry cannot set it.
    pub fn session_affinity(&self, worker_type: WorkerType) -> Option<SessionAffinitySetting> {
        match worker_type {
            WorkerType::Aggregated => &self.aggregated,
            WorkerType::Prefill => &self.prefill,
            WorkerType::Decode => &self.decode,
            WorkerType::Encode => &self.encode,
        }
        .as_ref()
        .and_then(|role| role.session_affinity)
    }

    /// A role's `busy_threshold`; `None` when its entry does not have the key. An `encode`
    /// entry cannot set it.
    pub fn busy_threshold(&self, worker_type: WorkerType) -> Option<BusyThresholdSpec> {
        match worker_type {
            WorkerType::Aggregated => &self.aggregated,
            WorkerType::Prefill => &self.prefill,
            WorkerType::Decode => &self.decode,
            WorkerType::Encode => &self.encode,
        }
        .as_ref()
        .and_then(|role| role.busy_threshold)
    }

    /// The hints a role's entry lists, in order; `None` when it lists none, so the role runs its
    /// default hints. Only the inline mapping form lists hints.
    pub fn hints(&self, worker_type: WorkerType) -> Option<&[HintPolicyEntry]> {
        match worker_type {
            WorkerType::Aggregated => self.aggregated(),
            WorkerType::Prefill => self.prefill(),
            WorkerType::Decode => self.decode(),
            WorkerType::Encode => self.encode(),
        }
        .and_then(WorkerSelectionPolicySelection::hints)
    }

    /// Look up one named instance from the deprecated `instances:` list.
    pub fn instance(&self, name: &str) -> Option<&WorkerSelectionInstance> {
        self.instances.get(name)
    }

    /// Return configured instance names in stable order for diagnostics.
    pub fn instance_names(&self) -> Vec<String> {
        let mut names = self.instances.keys().cloned().collect::<Vec<_>>();
        names.sort_unstable();
        names
    }
}

/// A role's `session_affinity` key.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SessionAffinitySetting {
    /// `false`: the role has no session affinity.
    Disabled,
    /// `true` or a mapping.
    Enabled(SessionAffinitySpec),
}

/// How one role keeps a session on the worker it first ran on.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SessionAffinitySpec {
    /// How long an idle session stays bound.
    pub ttl: Duration,
    pub mode: SessionAffinityMode,
    /// Most sessions the role's table holds; `None` leaves the host's limit.
    pub max_entries: Option<NonZeroUsize>,
}

impl SessionAffinitySpec {
    /// The TTL of `session_affinity: true`.
    pub const DEFAULT_TTL: Duration = Duration::from_secs(300);
}

impl Default for SessionAffinitySpec {
    fn default() -> Self {
        Self {
            ttl: Self::DEFAULT_TTL,
            mode: SessionAffinityMode::Hard,
            max_entries: None,
        }
    }
}

/// A role's `busy_threshold` key: when one of the role's workers stops taking requests. An
/// omitted limit is not checked.
#[derive(Debug, Clone, Copy, Default, PartialEq, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct BusyThresholdSpec {
    /// Fraction (0.0 to 1.0) of the worker's KV blocks in use above which it is busy.
    pub active_decode_blocks: Option<f64>,
    /// Fraction of the worker's `max_num_batched_tokens` above which its active prefill
    /// tokens make it busy.
    pub active_prefill_tokens_frac: Option<f64>,
}

/// One role's worker-selection policy selection.
#[derive(Debug, Clone, PartialEq)]
pub enum WorkerSelectionPolicySelection {
    /// A bare string: the reserved `default`, a registered policy type with default parameters,
    /// or (deprecated) the name of an entry in `instances:`.
    Named(String),
    /// An inline `{type, parameters}` spec.
    Inline(WorkerSelectionInstance),
}

impl WorkerSelectionPolicySelection {
    /// Whether this selection is the reserved built-in policy.
    pub fn is_default(&self) -> bool {
        matches!(self, Self::Named(name) if name == "default")
    }

    /// A short label for diagnostics: the name for a bare string, the type for an inline spec.
    pub fn label(&self) -> &str {
        match self {
            Self::Named(name) => name,
            Self::Inline(instance) => instance.policy_type(),
        }
    }

    /// The hints an inline spec lists; `None` for a bare string, which takes no hints.
    pub(crate) fn hints(&self) -> Option<&[HintPolicyEntry]> {
        match self {
            Self::Named(_) => None,
            Self::Inline(instance) => instance.hints(),
        }
    }
}

/// One parameterized worker-selection policy: a registered type plus its YAML parameters, and the
/// hints an inline spec lists.
#[derive(Debug, Clone, PartialEq)]
pub struct WorkerSelectionInstance {
    policy_type: String,
    parameters: serde_yaml::Value,
    hints: Option<Vec<HintPolicyEntry>>,
}

impl WorkerSelectionInstance {
    /// The policy type registered by a linked policy crate.
    pub fn policy_type(&self) -> &str {
        &self.policy_type
    }

    /// YAML parameters owned and validated by the linked policy crate.
    pub fn parameters(&self) -> &serde_yaml::Value {
        &self.parameters
    }

    /// The entry's `hints`, in order; `None` when it has none, so the role runs its defaults.
    pub fn hints(&self) -> Option<&[HintPolicyEntry]> {
        self.hints.as_deref()
    }
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct RawWorkerSelectionConfig {
    aggregated: Option<serde_yaml::Value>,
    prefill: Option<serde_yaml::Value>,
    decode: Option<serde_yaml::Value>,
    encode: Option<serde_yaml::Value>,
    #[serde(default)]
    instances: Vec<RawWorkerSelectionInstance>,
}

enum RawRoleSelection {
    Named(String),
    Inline(Box<RawInlineSelection>),
}

impl RawRoleSelection {
    // Parsed per role rather than with `#[serde(untagged)]`, which discards the inner error
    // (for example an unknown field) and reports only that no variant matched.
    fn from_value(role: &str, value: serde_yaml::Value) -> Result<Self, RouterPolicyConfigError> {
        let location = format!("worker_selection.{role}");
        match value {
            serde_yaml::Value::String(name) => Ok(Self::Named(name)),
            serde_yaml::Value::Mapping(_) => serde_yaml::from_value(value)
                .map(Self::Inline)
                .map_err(|source| {
                    RouterPolicyConfigError::Validation(format!("{location}: {source}"))
                }),
            _ => Err(RouterPolicyConfigError::Validation(format!(
                "{location} must be a policy type name or a {{type, parameters}} mapping"
            ))),
        }
    }
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct RawInlineSelection {
    #[serde(rename = "type")]
    policy_type: String,
    #[serde(default = "empty_parameters")]
    parameters: serde_yaml::Value,
    #[serde(default)]
    hints: Option<Vec<RawHintPolicyEntry>>,
    #[serde(default)]
    session_affinity: Option<serde_yaml::Value>,
    #[serde(default, deserialize_with = "present")]
    busy_threshold: Option<serde_yaml::Value>,
}

/// Keeps a key with a null value (`busy_threshold:` with its limits commented out) apart from
/// an absent key, so it is rejected instead of falling back to the card or the flags.
fn present<'de, D: serde::Deserializer<'de>>(
    deserializer: D,
) -> Result<Option<serde_yaml::Value>, D::Error> {
    serde_yaml::Value::deserialize(deserializer).map(Some)
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct RawSessionAffinity {
    ttl_secs: Option<u64>,
    mode: Option<SessionAffinityMode>,
    max_entries: Option<NonZeroUsize>,
}

fn resolve_session_affinity(
    location: &str,
    value: serde_yaml::Value,
) -> Result<SessionAffinitySetting, RouterPolicyConfigError> {
    let raw: RawSessionAffinity = match value {
        serde_yaml::Value::Bool(false) => return Ok(SessionAffinitySetting::Disabled),
        serde_yaml::Value::Bool(true) => {
            return Ok(SessionAffinitySetting::Enabled(
                SessionAffinitySpec::default(),
            ));
        }
        serde_yaml::Value::Mapping(_) => serde_yaml::from_value(value).map_err(|source| {
            RouterPolicyConfigError::Validation(format!("{location}: {source}"))
        })?,
        _ => {
            return Err(RouterPolicyConfigError::Validation(format!(
                "{location} must be true, false, or a {{ttl_secs, mode, max_entries}} mapping"
            )));
        }
    };
    let defaults = SessionAffinitySpec::default();
    let ttl = raw.ttl_secs.map_or(defaults.ttl, Duration::from_secs);
    if SessionAffinity::validate_ttl(ttl).is_err() {
        return Err(RouterPolicyConfigError::Validation(format!(
            "{location}.ttl_secs must be between 1 and {MAX_SESSION_AFFINITY_TTL_SECS}"
        )));
    }
    Ok(SessionAffinitySetting::Enabled(SessionAffinitySpec {
        ttl,
        mode: raw.mode.unwrap_or(defaults.mode),
        max_entries: raw.max_entries,
    }))
}

fn resolve_busy_threshold(
    location: &str,
    value: serde_yaml::Value,
) -> Result<BusyThresholdSpec, RouterPolicyConfigError> {
    if !value.is_mapping() {
        return Err(RouterPolicyConfigError::Validation(format!(
            "{location} must be an {{active_decode_blocks, active_prefill_tokens_frac}} mapping"
        )));
    }
    let spec: BusyThresholdSpec = serde_yaml::from_value(value)
        .map_err(|source| RouterPolicyConfigError::Validation(format!("{location}: {source}")))?;
    if let Some(threshold) = spec.active_decode_blocks
        && !(0.0..=1.0).contains(&threshold)
    {
        return Err(RouterPolicyConfigError::Validation(format!(
            "{location}.active_decode_blocks must be between 0.0 and 1.0, got {threshold}"
        )));
    }
    if let Some(threshold) = spec.active_prefill_tokens_frac
        && !(threshold.is_finite() && threshold >= 0.0)
    {
        return Err(RouterPolicyConfigError::Validation(format!(
            "{location}.active_prefill_tokens_frac must be a finite value greater than or \
             equal to 0.0, got {threshold}"
        )));
    }
    Ok(spec)
}

impl RawWorkerSelectionConfig {
    pub(crate) fn resolve(self) -> Result<WorkerSelectionConfig, RouterPolicyConfigError> {
        if self.instances.is_empty()
            && self.aggregated.is_none()
            && self.prefill.is_none()
            && self.decode.is_none()
            && self.encode.is_none()
        {
            return Err(RouterPolicyConfigError::Validation(
                "worker_selection must select a policy for aggregated, prefill, decode, or encode"
                    .to_string(),
            ));
        }

        if !self.instances.is_empty() {
            tracing::warn!(
                "worker_selection.instances is deprecated for removal in v1.8; write each role's policy inline as a type name or a {{type, parameters}} mapping"
            );
        }

        let mut instances = HashMap::with_capacity(self.instances.len());
        for raw in self.instances {
            validate_identifier(&raw.name, "instance", "worker_selection")?;
            if raw.name == "default" {
                return Err(RouterPolicyConfigError::Validation(
                    "worker_selection instance name 'default' is reserved for Dynamo's built-in selector".to_string(),
                ));
            }
            let instance = resolve_instance(
                raw.policy_type,
                raw.parameters,
                None,
                &format!("worker_selection instance {:?}", raw.name),
            )?;
            if instances.insert(raw.name.clone(), instance).is_some() {
                return Err(RouterPolicyConfigError::Validation(format!(
                    "worker_selection contains duplicate instance {:?}",
                    raw.name
                )));
            }
        }

        let resolve_role = |role: &str,
                            raw: Option<serde_yaml::Value>|
         -> Result<Option<RoleSelection>, RouterPolicyConfigError> {
            let location = format!("worker_selection.{role}");
            let Some(raw) = raw
                .map(|value| RawRoleSelection::from_value(role, value))
                .transpose()?
            else {
                return Ok(None);
            };
            let inline = match raw {
                RawRoleSelection::Named(name) => {
                    validate_identifier(&name, "policy", &location)?;
                    return Ok(Some(RoleSelection {
                        policy: WorkerSelectionPolicySelection::Named(name),
                        session_affinity: None,
                        busy_threshold: None,
                    }));
                }
                RawRoleSelection::Inline(inline) => *inline,
            };
            let session_affinity = inline
                .session_affinity
                .map(|value| {
                    resolve_session_affinity(&format!("{location}.session_affinity"), value)
                })
                .transpose()?;
            if session_affinity.is_some() && role == "encode" {
                return Err(RouterPolicyConfigError::Validation(format!(
                    "{location}.session_affinity is not supported for encode"
                )));
            }
            let busy_threshold = inline
                .busy_threshold
                .map(|value| resolve_busy_threshold(&format!("{location}.busy_threshold"), value))
                .transpose()?;
            // The router monitors no load for an encode pool, so a threshold would never apply.
            if busy_threshold.is_some() && role == "encode" {
                return Err(RouterPolicyConfigError::Validation(format!(
                    "{location}.busy_threshold is not supported for encode"
                )));
            }
            let policy = if inline.policy_type == "default" {
                // The mapping form exists so `default` can carry the router's own keys.
                let no_parameters = matches!(
                    &inline.parameters,
                    serde_yaml::Value::Mapping(parameters) if parameters.is_empty()
                );
                if !no_parameters || inline.hints.is_some() {
                    return Err(RouterPolicyConfigError::Validation(format!(
                        "{location} type 'default' takes no parameters or hints"
                    )));
                }
                WorkerSelectionPolicySelection::Named(inline.policy_type)
            } else {
                let hints = inline
                    .hints
                    .map(|hints| resolve_hints(&format!("{location}.hints"), hints))
                    .transpose()?;
                WorkerSelectionPolicySelection::Inline(resolve_instance(
                    inline.policy_type,
                    inline.parameters,
                    hints,
                    &location,
                )?)
            };
            Ok(Some(RoleSelection {
                policy,
                session_affinity,
                busy_threshold,
            }))
        };

        Ok(WorkerSelectionConfig {
            aggregated: resolve_role("aggregated", self.aggregated)?,
            prefill: resolve_role("prefill", self.prefill)?,
            decode: resolve_role("decode", self.decode)?,
            encode: resolve_role("encode", self.encode)?,
            instances,
        })
    }
}

fn resolve_instance(
    policy_type: String,
    parameters: serde_yaml::Value,
    hints: Option<Vec<HintPolicyEntry>>,
    location: &str,
) -> Result<WorkerSelectionInstance, RouterPolicyConfigError> {
    validate_identifier(&policy_type, "policy type", location)?;
    if !matches!(parameters, serde_yaml::Value::Mapping(_)) {
        return Err(RouterPolicyConfigError::Validation(format!(
            "{location} parameters must be a YAML mapping"
        )));
    }
    Ok(WorkerSelectionInstance {
        policy_type,
        parameters,
        hints,
    })
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct RawWorkerSelectionInstance {
    name: String,
    #[serde(rename = "type")]
    policy_type: String,
    #[serde(default = "empty_parameters")]
    parameters: serde_yaml::Value,
}

fn empty_parameters() -> serde_yaml::Value {
    serde_yaml::Value::Mapping(Default::default())
}
