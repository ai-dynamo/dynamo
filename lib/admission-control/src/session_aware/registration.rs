// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::collections::HashMap;

use dynamo_kv_router::RouterQueuePolicy;
use dynamo_kv_router::protocols::{WorkerConfigLike, WorkerId};
use dynamo_kv_router::scheduling::{
    PolicyClassAdmissionStrategies, PolicyProfile, QueueAdmissionConfig,
};
use thiserror::Error;
use tokio::sync::watch;

use super::{
    ConfigError, SessionAwareAdmissionControl, SessionAwareAdmissionControlConfig,
    WatchWorkerCapacity,
};

#[derive(Debug, Error)]
pub enum RegistrationError {
    #[error("session-aware admission control requires FCFS for policy class {0:?}")]
    SessionAwareAdmissionControlRequiresFcfs(String),
    #[error(
        "session-aware admission control may be configured for only one policy class (found {first:?} and {second:?})"
    )]
    MultipleSessionAwareAdmissionControlClasses { first: String, second: String },
    #[error(transparent)]
    SessionAwareAdmissionControlConfig(#[from] ConfigError),
}

/// Register configured built-in strategies without replacing caller-provided ones.
pub fn register_builtin_strategies<C>(
    profile: &PolicyProfile,
    workers: watch::Receiver<HashMap<WorkerId, C>>,
    block_size: u32,
    strategies: &mut PolicyClassAdmissionStrategies,
) -> Result<(), RegistrationError>
where
    C: WorkerConfigLike + Send + Sync + 'static,
{
    let mut session_aware_class = None;
    for class in profile.classes() {
        let Some(admission) = &class.queue_admission else {
            continue;
        };
        if strategies.contains_key(&class.name) {
            continue;
        }
        let QueueAdmissionConfig::SessionAware {
            pause_threshold,
            pause_target,
            resume_timeout_seconds,
            session_retention_seconds,
            scheduler_interval_seconds,
        } = admission;
        if class.queue_policy != RouterQueuePolicy::Fcfs {
            return Err(RegistrationError::SessionAwareAdmissionControlRequiresFcfs(
                class.name.clone(),
            ));
        }
        if let Some(first) = session_aware_class.replace(class.name.clone()) {
            return Err(
                RegistrationError::MultipleSessionAwareAdmissionControlClasses {
                    first,
                    second: class.name.clone(),
                },
            );
        }
        let mut config = SessionAwareAdmissionControlConfig::default();
        if let Some(value) = *pause_threshold {
            config.pause_threshold = value;
        }
        if let Some(value) = *pause_target {
            config.pause_target = value;
        }
        if let Some(value) = *resume_timeout_seconds {
            config.resume_timeout_seconds = value;
        }
        if let Some(value) = *session_retention_seconds {
            config.session_retention_seconds = value;
        }
        if let Some(value) = *scheduler_interval_seconds {
            config.scheduler_interval_seconds = value;
        }
        let capacity = WatchWorkerCapacity::new(workers.clone(), block_size);
        strategies.insert(
            class.name.clone(),
            Box::new(SessionAwareAdmissionControl::new(capacity, config)?),
        );
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use std::time::Duration;

    use super::*;
    use dynamo_kv_router::scheduling::{
        AdmissionDecision, AdmissionRequest, PolicyClassAdmissionStrategy, RouterPolicyConfig,
        WorkerPlacement,
    };

    struct TestWorkerConfig;

    impl WorkerConfigLike for TestWorkerConfig {
        fn data_parallel_start_rank(&self) -> u32 {
            0
        }

        fn data_parallel_size(&self) -> u32 {
            1
        }

        fn max_num_batched_tokens(&self) -> Option<u64> {
            None
        }

        fn total_kv_blocks(&self) -> Option<u64> {
            Some(1)
        }
    }

    struct CustomStrategy;

    impl PolicyClassAdmissionStrategy for CustomStrategy {
        fn admit(&mut self, _request: AdmissionRequest<'_>) -> AdmissionDecision {
            AdmissionDecision::Ready(WorkerPlacement::Any)
        }
    }

    fn profile() -> PolicyProfile {
        RouterPolicyConfig::from_yaml(
            r#"
default_policy_family: standard
uncached_isl_buckets:
  - min_tokens: 0
    bucket: all
policy_classes:
  - name: agents
    policy_family: standard
    cache_bucket: all
    queue_admission:
      type: session_aware
      scheduler_interval_seconds: 3.0
    quantum: 1
"#,
        )
        .unwrap()
        .resolve_profile(None, None, RouterQueuePolicy::Fcfs)
    }

    fn workers() -> watch::Receiver<HashMap<WorkerId, TestWorkerConfig>> {
        watch::channel(HashMap::new()).1
    }

    #[test]
    fn builds_configured_session_aware() {
        let mut strategies = PolicyClassAdmissionStrategies::new();
        register_builtin_strategies(&profile(), workers(), 16, &mut strategies).unwrap();

        assert_eq!(
            strategies["agents"].reconcile_interval(),
            Some(Duration::from_secs(3))
        );
    }

    #[test]
    fn preserves_caller_provided_strategy() {
        let mut strategies = PolicyClassAdmissionStrategies::new();
        strategies.insert("agents".to_owned(), Box::new(CustomStrategy));

        register_builtin_strategies(&profile(), workers(), 16, &mut strategies).unwrap();

        assert_eq!(strategies["agents"].reconcile_interval(), None);
    }
}
