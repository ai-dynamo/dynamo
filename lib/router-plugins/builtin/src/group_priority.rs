// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Group-aware queue priority for RL rollouts, without program-level admission.
//!
//! A data group (`x-dynamo-group`) is the `x-dynamo-group-size` rollouts of one prompt that train
//! together. The group is stamped with the highest `x-dynamo-weight-version` seen when its first
//! session arrives; its age is the current version minus that stamp. Each request of a grouped
//! session gets a router queue priority from its group and is released immediately: the classifier
//! never defers requests and never sets a worker target, so worker selection stays with the
//! configured policy. Priority orders only the router queue, which holds requests only when busy
//! thresholds are configured; the engine-facing request priority is left unchanged.

use std::collections::{BTreeMap, HashMap};
use std::sync::Arc;
use std::time::{Duration, Instant};

use dynamo_kv_router::plugins::RouterPluginRegistry;
use dynamo_kv_router::plugins::request_classifier::{
    ClassifyFuture, ClassifyRequest, RequestClassifier, RequestClassifierFactory,
    RequestClassifierParameters, RequestClassifierProviderError, RequestClassifierRegistryError,
};
use dynamo_kv_router::plugins::worker_selection::WorkerCandidates;
use serde::Deserialize;

pub(crate) const POLICY_TYPE: &str = "group_priority";

const GROUP_HEADER: &str = "x-dynamo-group";
const GROUP_SIZE_HEADER: &str = "x-dynamo-group-size";
const ENV_HEADER: &str = "x-dynamo-env-s";
const VERSION_HEADER: &str = "x-dynamo-weight-version";

/// Weights of the queue priority boost, in seconds of queue precedence under FCFS.
#[derive(Clone, Debug, Default, PartialEq, Deserialize)]
#[serde(default, deny_unknown_fields)]
struct JumpWeights {
    /// Constant added to every grouped request; lets a negative weight still rank above zero.
    offset: f64,
    /// Per weight version of group age; positive serves older groups first.
    age: f64,
    /// Per unfinished session of the group; negative serves nearly done groups first.
    left: f64,
    /// Per second of `x-dynamo-env-s`; positive serves slow environments first.
    env_s: f64,
}

#[derive(Clone, Debug, PartialEq, Deserialize)]
#[serde(default, deny_unknown_fields)]
struct GroupPriorityConfig {
    /// Strict queue tiers by group age: tier = min(age, age_tiers). 0 leaves the tier unchanged.
    age_tiers: u32,
    /// Queue boost within a tier; absent leaves the request's own boost unchanged.
    jump: Option<JumpWeights>,
    /// Forget a session, and a group with no session left, after this long without a request.
    session_retention_seconds: f64,
    /// Sessions tracked at once; sessions beyond it pass through unchanged.
    max_tracked_sessions: usize,
}

impl Default for GroupPriorityConfig {
    fn default() -> Self {
        Self {
            age_tiers: 0,
            jump: None,
            session_retention_seconds: 1_800.0,
            max_tracked_sessions: 100_000,
        }
    }
}

impl GroupPriorityConfig {
    fn validate(&self) -> Result<(), &'static str> {
        if let Some(jump) = &self.jump
            && ![jump.offset, jump.age, jump.left, jump.env_s]
                .iter()
                .all(|weight| weight.is_finite())
        {
            return Err("jump weights must be finite");
        }
        match Duration::try_from_secs_f64(self.session_retention_seconds) {
            Ok(retention) if !retention.is_zero() => {}
            _ => return Err("session_retention_seconds must be positive"),
        }
        if self.max_tracked_sessions == 0 {
            return Err("max_tracked_sessions must be positive");
        }
        Ok(())
    }
}

/// The last value of header `name`, trimmed.
fn header<'a>(headers: &'a BTreeMap<String, Vec<String>>, name: &str) -> Option<&'a str> {
    headers
        .get(name)
        .and_then(|values| values.last())
        .map(|value| value.trim())
        .filter(|value| !value.is_empty())
}

struct Group {
    version: u64,
    /// Sessions of the group that have not ended; starts at the group size.
    left: usize,
    /// Tracked sessions of the group.
    sessions: usize,
    last_seen: Instant,
}

struct Session {
    group: String,
    last_seen: Instant,
}

#[derive(Default)]
struct State {
    version: u64,
    groups: HashMap<String, Group>,
    sessions: HashMap<String, Session>,
    last_sweep: Option<Instant>,
}

impl State {
    /// Record a request of `session_id` in `group`; returns the group's age and unfinished count.
    fn observe(
        &mut self,
        session_id: &str,
        group: &str,
        size: usize,
        now: Instant,
        max_sessions: usize,
    ) -> Option<(u64, usize)> {
        if let Some(session) = self.sessions.get_mut(session_id) {
            session.last_seen = now;
        } else {
            if self.sessions.len() >= max_sessions {
                return None;
            }
            let version = self.version;
            let entry = self.groups.entry(group.to_owned()).or_insert(Group {
                version,
                left: size.max(1),
                sessions: 0,
                last_seen: now,
            });
            entry.sessions += 1;
            self.sessions.insert(
                session_id.to_owned(),
                Session {
                    group: group.to_owned(),
                    last_seen: now,
                },
            );
        }
        let group_name = &self.sessions[session_id].group;
        let group = self.groups.get_mut(group_name)?;
        group.last_seen = now;
        Some((self.version.saturating_sub(group.version), group.left))
    }

    /// `session_id` ended at `now`; its group has one fewer session to wait for.
    fn end(&mut self, session_id: &str, now: Instant) {
        let Some(session) = self.sessions.remove(session_id) else {
            return;
        };
        let Some(group) = self.groups.get_mut(&session.group) else {
            return;
        };
        group.left = group.left.saturating_sub(1);
        group.sessions = group.sessions.saturating_sub(1);
        group.last_seen = now;
        if group.left == 0 {
            self.groups.remove(&session.group);
        }
    }

    /// Forget sessions idle past `retention`, then groups with no session left and idle as long.
    fn sweep(&mut self, now: Instant, retention: Duration) {
        if self
            .last_sweep
            .is_some_and(|last| now.saturating_duration_since(last) < retention / 16)
        {
            return;
        }
        self.last_sweep = Some(now);
        let expired: Vec<String> = self
            .sessions
            .iter()
            .filter(|(_, session)| now.saturating_duration_since(session.last_seen) >= retention)
            .map(|(session_id, _)| session_id.clone())
            .collect();
        for session_id in expired {
            self.end(&session_id, now);
        }
        self.groups.retain(|_, group| {
            group.sessions > 0 || now.saturating_duration_since(group.last_seen) < retention
        });
    }
}

struct GroupPriorityClassifier {
    config: GroupPriorityConfig,
    retention: Duration,
    state: State,
}

impl GroupPriorityClassifier {
    fn new(config: GroupPriorityConfig) -> Self {
        let retention = Duration::from_secs_f64(config.session_retention_seconds);
        Self {
            config,
            retention,
            state: State::default(),
        }
    }

    /// The strict tier and queue boost for one request of `session_id`, each `None` when unchanged.
    fn priority(
        &mut self,
        session_id: &str,
        session_final: bool,
        headers: &BTreeMap<String, Vec<String>>,
        now: Instant,
    ) -> (Option<u32>, Option<f64>) {
        if let Some(version) = header(headers, VERSION_HEADER).and_then(|v| v.parse().ok()) {
            self.state.version = self.state.version.max(version);
        }
        self.state.sweep(now, self.retention);
        if session_final {
            self.state.end(session_id, now);
            return (None, None);
        }
        let Some(group) = header(headers, GROUP_HEADER) else {
            return (None, None);
        };
        let size = header(headers, GROUP_SIZE_HEADER)
            .and_then(|value| value.parse().ok())
            .unwrap_or(1);
        let env_s = header(headers, ENV_HEADER)
            .and_then(|value| value.parse::<f64>().ok())
            .filter(|value| value.is_finite())
            .unwrap_or(0.0);
        let Some((age, left)) = self.state.observe(
            session_id,
            group,
            size,
            now,
            self.config.max_tracked_sessions,
        ) else {
            return (None, None);
        };
        let tier = (self.config.age_tiers > 0).then(|| {
            u32::try_from(age)
                .unwrap_or(u32::MAX)
                .min(self.config.age_tiers)
        });
        let jump = self.config.jump.as_ref().map(|jump| {
            jump.offset + jump.age * age as f64 + jump.left * left as f64 + jump.env_s * env_s
        });
        (tier, jump)
    }
}

impl RequestClassifier for GroupPriorityClassifier {
    fn classify(
        &mut self,
        mut request: ClassifyRequest,
        _workers: WorkerCandidates<'_>,
    ) -> ClassifyFuture {
        let (tier, jump) = request.session_context().map_or((None, None), |session| {
            self.priority(
                session.session_id(),
                session.session_final() == Some(true),
                session.agent_headers(),
                Instant::now(),
            )
        });
        if let Some(tier) = tier {
            request.set_strict_priority(tier);
        }
        if let Some(jump) = jump {
            request.set_priority_jump(jump);
        }
        Box::pin(async move { Ok(request) })
    }
}

fn classifier_provider(
    parameters: &RequestClassifierParameters,
) -> Result<RequestClassifierFactory, RequestClassifierProviderError> {
    let config: GroupPriorityConfig = parameters.deserialize()?;
    config.validate().map_err(|message| {
        RequestClassifierProviderError::new(format!(
            "invalid group_priority configuration: {message}"
        ))
    })?;
    Ok(Arc::new(move |_context| {
        Box::new(GroupPriorityClassifier::new(config.clone()))
    }))
}

pub(crate) fn register(
    registry: &mut RouterPluginRegistry,
) -> Result<(), RequestClassifierRegistryError> {
    registry.register_request_classifier(POLICY_TYPE, Arc::new(classifier_provider))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn config(age_tiers: u32, jump: Option<JumpWeights>) -> GroupPriorityConfig {
        GroupPriorityConfig {
            age_tiers,
            jump,
            ..Default::default()
        }
    }

    #[test]
    fn a_group_keeps_its_first_version_and_counts_down() {
        let now = Instant::now();
        let mut state = State {
            version: 3,
            ..Default::default()
        };
        assert_eq!(state.observe("a", "g", 2, now, 10), Some((0, 2)));
        state.version = 7;
        assert_eq!(state.observe("b", "g", 2, now, 10), Some((4, 2)));
        state.end("a", now);
        assert_eq!(state.observe("b", "g", 2, now, 10), Some((4, 1)));
        state.end("b", now);
        assert!(state.groups.is_empty() && state.sessions.is_empty());
        // A new session after the group finished starts a new group at the current version.
        assert_eq!(state.observe("c", "g", 2, now, 10), Some((0, 2)));
    }

    #[test]
    fn idle_sessions_and_groups_expire() {
        let now = Instant::now();
        let retention = Duration::from_secs(60);
        let mut state = State::default();
        state.observe("a", "g", 3, now, 10);
        state.sweep(now + retention, retention);
        assert!(state.sessions.is_empty());
        assert_eq!(state.groups["g"].left, 2);
        state.sweep(now + 3 * retention, retention);
        assert!(state.groups.is_empty());
    }

    #[test]
    fn sessions_past_the_limit_pass_through() {
        let now = Instant::now();
        let mut state = State::default();
        assert!(state.observe("a", "g", 2, now, 1).is_some());
        assert!(state.observe("b", "g", 2, now, 1).is_none());
        assert!(state.observe("a", "g", 2, now, 1).is_some());
    }

    fn headers(pairs: &[(&str, &str)]) -> BTreeMap<String, Vec<String>> {
        pairs
            .iter()
            .map(|(name, value)| (name.to_string(), vec![value.to_string()]))
            .collect()
    }

    #[test]
    fn scores_grouped_sessions_and_leaves_others_unchanged() {
        let jump = JumpWeights {
            offset: 100.0,
            age: 2.0,
            left: -1.0,
            env_s: 0.5,
        };
        let mut classifier = GroupPriorityClassifier::new(config(3, Some(jump)));
        let now = Instant::now();
        let first = headers(&[
            (GROUP_HEADER, "g"),
            (GROUP_SIZE_HEADER, "2"),
            (VERSION_HEADER, "5"),
            (ENV_HEADER, "4"),
        ]);
        assert_eq!(
            classifier.priority("a", false, &first, now),
            (Some(0), Some(100.0 - 2.0 + 2.0))
        );
        let later = headers(&[
            (GROUP_HEADER, "g"),
            (GROUP_SIZE_HEADER, "2"),
            (VERSION_HEADER, "10"),
        ]);
        // Age 5 is capped at the top tier; one session ended, so one is left.
        classifier.priority("a", true, &later, now);
        assert_eq!(
            classifier.priority("b", false, &later, now),
            (Some(3), Some(100.0 + 10.0 - 1.0))
        );
        let ungrouped = headers(&[(VERSION_HEADER, "10")]);
        assert_eq!(
            classifier.priority("c", false, &ungrouped, now),
            (None, None)
        );
        let mut unscored = GroupPriorityClassifier::new(config(0, None));
        assert_eq!(unscored.priority("a", false, &first, now), (None, None));
    }

    #[test]
    fn validates_parameters() {
        assert!(GroupPriorityConfig::default().validate().is_ok());
        let nan = JumpWeights {
            age: f64::NAN,
            ..Default::default()
        };
        assert!(config(0, Some(nan)).validate().is_err());
        assert!(
            GroupPriorityConfig {
                session_retention_seconds: 0.0,
                ..Default::default()
            }
            .validate()
            .is_err()
        );
        assert!(
            GroupPriorityConfig {
                max_tracked_sessions: 0,
                ..Default::default()
            }
            .validate()
            .is_err()
        );
    }

    #[test]
    fn resolves_from_router_policy_yaml() {
        let mut registry = crate::default_registry();
        crate::register(&mut registry).unwrap();
        let policy = tempfile::NamedTempFile::new().unwrap();
        std::fs::write(
            policy.path(),
            r#"
request_classifier:
  type: group_priority
  parameters:
    age_tiers: 8
    jump: { offset: 100.0, left: -1.0 }
"#,
        )
        .unwrap();
        let config = dynamo_kv_router::KvRouterConfig {
            router_policy_config: Some(policy.path().display().to_string()),
            ..Default::default()
        };
        let plugins = registry.resolve_plugins(&config).unwrap();
        // Worker selection stays with the default policy.
        assert!(!plugins.has_custom_worker_selection());
        let factory = plugins.request_classifier().unwrap();
        let context = dynamo_kv_router::plugins::request_classifier::RequestClassifierContext::new(
            16,
            Vec::new,
        );
        let _classifier = factory(context);
    }
}
