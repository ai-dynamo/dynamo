// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::collections::HashMap;

use dynamo_kv_router::RouterQueuePolicy;
use dynamo_kv_router::protocols::{WorkerConfigLike, WorkerId};
use dynamo_kv_router::scheduling::{PolicyClassAdmissionStrategies, PolicyProfile};
use thiserror::Error;
use tokio::sync::watch;

use super::{ConfigError, STRATEGY_NAME, ThunderAgent, ThunderAgentConfig, WatchWorkerCapacity};

#[derive(Debug, Error)]
pub enum RegistrationError {
    #[error("unsupported queue admission strategy {strategy:?} for policy class {policy_class:?}")]
    UnsupportedStrategy {
        strategy: String,
        policy_class: String,
    },
    #[error("ThunderAgent queue admission requires FCFS for policy class {0:?}")]
    ThunderAgentRequiresFcfs(String),
    #[error(
        "ThunderAgent queue admission may be configured for only one policy class (found {first:?} and {second:?})"
    )]
    MultipleThunderAgentClasses { first: String, second: String },
    #[error(transparent)]
    ThunderAgentConfig(#[from] ConfigError),
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
    let mut thunderagent_class = None;
    for class in profile.classes() {
        let Some(admission) = &class.queue_admission else {
            continue;
        };
        if strategies.contains_key(&class.name) {
            continue;
        }
        if admission.strategy != STRATEGY_NAME {
            return Err(RegistrationError::UnsupportedStrategy {
                strategy: admission.strategy.clone(),
                policy_class: class.name.clone(),
            });
        }
        if class.queue_policy != RouterQueuePolicy::Fcfs {
            return Err(RegistrationError::ThunderAgentRequiresFcfs(
                class.name.clone(),
            ));
        }
        if let Some(first) = thunderagent_class.replace(class.name.clone()) {
            return Err(RegistrationError::MultipleThunderAgentClasses {
                first,
                second: class.name.clone(),
            });
        }
        let config = ThunderAgentConfig::from_options(&admission.options)?;
        let capacity = WatchWorkerCapacity::new(workers.clone(), block_size);
        strategies.insert(
            class.name.clone(),
            Box::new(ThunderAgent::new(capacity, config)?),
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

    fn profile(strategy: &str) -> PolicyProfile {
        RouterPolicyConfig::from_yaml(&format!(
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
      type: {strategy}
    quantum: 1
"#
        ))
        .unwrap()
        .resolve_profile(None, None, RouterQueuePolicy::Fcfs)
    }

    fn workers() -> watch::Receiver<HashMap<WorkerId, TestWorkerConfig>> {
        watch::channel(HashMap::new()).1
    }

    #[test]
    fn builds_configured_thunderagent() {
        let mut strategies = PolicyClassAdmissionStrategies::new();
        register_builtin_strategies(&profile(STRATEGY_NAME), workers(), 16, &mut strategies)
            .unwrap();

        assert_eq!(
            strategies["agents"].reconcile_interval(),
            Some(Duration::from_secs(5))
        );
    }

    #[test]
    fn rejects_unknown_strategy() {
        let mut strategies = PolicyClassAdmissionStrategies::new();
        let error = register_builtin_strategies(&profile("custom"), workers(), 16, &mut strategies)
            .unwrap_err();

        assert!(matches!(
            error,
            RegistrationError::UnsupportedStrategy { .. }
        ));
    }

    #[test]
    fn preserves_caller_provided_strategy() {
        let mut strategies = PolicyClassAdmissionStrategies::new();
        strategies.insert("agents".to_owned(), Box::new(CustomStrategy));

        register_builtin_strategies(&profile("custom"), workers(), 16, &mut strategies).unwrap();

        assert_eq!(strategies["agents"].reconcile_interval(), None);
    }
}
