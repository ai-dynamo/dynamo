// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::collections::HashMap;

use dynamo_kv_router::RouterQueuePolicy;
use dynamo_kv_router::protocols::{WorkerConfigLike, WorkerId};
use dynamo_kv_router::scheduling::{PolicyClassAdmissionPolicies, PolicyProfile};
use thiserror::Error;
use tokio::sync::watch;

use super::{
    ConfigError, POLICY_NAME, SessionAwareAdmissionControl, SessionAwareAdmissionControlConfig,
    WatchWorkerCapacity,
};

#[derive(Debug, Error)]
pub enum RegistrationError {
    #[error("session-aware admission control requires FCFS for policy class {0:?}")]
    SessionAwareAdmissionControlRequiresFcfs(String),
    #[error(
        "session-aware admission control requires an explicit policy class or a policy family with exactly one cache bucket: {0:?}"
    )]
    SessionAwareAdmissionControlRequiresStableClass(String),
    #[error(
        "session-aware admission control may be configured for only one policy class (found {first:?} and {second:?})"
    )]
    MultipleSessionAwareAdmissionControlClasses { first: String, second: String },
    #[error("unknown admission policy type {policy_type:?} for policy class {class_name:?}")]
    UnknownAdmissionPolicyType {
        class_name: String,
        policy_type: String,
    },
    #[error(transparent)]
    SessionAwareAdmissionControlConfig(#[from] ConfigError),
}

/// Register configured built-in policies without replacing caller-provided ones.
pub fn register_builtin_policies<C>(
    profile: &PolicyProfile,
    workers: watch::Receiver<HashMap<WorkerId, C>>,
    block_size: u32,
    policies: &mut PolicyClassAdmissionPolicies,
) -> Result<(), RegistrationError>
where
    C: WorkerConfigLike + Send + Sync + 'static,
{
    let mut session_aware_class = None;
    for (class_index, class) in profile.classes().iter().enumerate() {
        let Some(admission) = &class.admission else {
            continue;
        };

        if admission.policy_type() != POLICY_NAME {
            if policies.contains_key(&class.name) {
                continue;
            }
            return Err(RegistrationError::UnknownAdmissionPolicyType {
                class_name: class.name.clone(),
                policy_type: admission.policy_type().to_owned(),
            });
        }
        if class.queue_policy != RouterQueuePolicy::Fcfs {
            return Err(RegistrationError::SessionAwareAdmissionControlRequiresFcfs(
                class.name.clone(),
            ));
        }
        let is_explicit = profile.direct_class_index(Some(&class.name)) == Some(class_index);
        let is_stable_family_class = profile.resolve_class_index(Some(&class.name), 0)
            == class_index
            && profile.resolve_class_index(Some(&class.name), usize::MAX) == class_index;
        if !is_explicit && !is_stable_family_class {
            return Err(
                RegistrationError::SessionAwareAdmissionControlRequiresStableClass(
                    class.name.clone(),
                ),
            );
        }
        if let Some(first) = session_aware_class.replace(class.name.clone()) {
            return Err(
                RegistrationError::MultipleSessionAwareAdmissionControlClasses {
                    first,
                    second: class.name.clone(),
                },
            );
        }
        let config = SessionAwareAdmissionControlConfig::from_options(admission.options())?;
        if policies.contains_key(&class.name) {
            continue;
        }
        let capacity = WatchWorkerCapacity::new(workers.clone(), block_size);
        policies.insert(
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
        AdmissionDecision, AdmissionRequest, PolicyClassAdmissionPolicy, RouterPolicyConfig,
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

    struct CustomPolicy;

    impl PolicyClassAdmissionPolicy for CustomPolicy {
        fn admit(&mut self, _request: AdmissionRequest<'_>) -> AdmissionDecision {
            AdmissionDecision::Ready(WorkerPlacement::Any)
        }
    }

    fn resolve_profile(yaml: &str) -> PolicyProfile {
        RouterPolicyConfig::from_yaml(yaml)
            .unwrap()
            .resolve_profile(None, None, RouterQueuePolicy::Fcfs)
    }

    fn profile() -> PolicyProfile {
        resolve_profile(
            r#"
default_policy_family: standard
uncached_isl_buckets:
  - min_tokens: 0
    bucket: all
policy_classes:
  - name: standard
    policy_family: standard
    cache_bucket: all
    quantum: 1
  - name: agents
    admission:
      type: session_aware
      scheduler_interval_seconds: 3.0
    quantum: 1
"#,
        )
    }

    fn workers() -> watch::Receiver<HashMap<WorkerId, TestWorkerConfig>> {
        watch::channel(HashMap::new()).1
    }

    #[test]
    fn builds_configured_session_aware() {
        let mut policies = PolicyClassAdmissionPolicies::new();
        register_builtin_policies(&profile(), workers(), 16, &mut policies).unwrap();

        assert_eq!(
            policies["agents"].reconcile_interval(),
            Some(Duration::from_secs(3))
        );
    }

    #[test]
    fn preserves_caller_provided_policy() {
        let mut policies = PolicyClassAdmissionPolicies::new();
        policies.insert("agents".to_owned(), Box::new(CustomPolicy));

        register_builtin_policies(&profile(), workers(), 16, &mut policies).unwrap();

        assert_eq!(policies["agents"].reconcile_interval(), None);
    }

    #[test]
    fn validates_config_before_honoring_caller_policy() {
        let profile = resolve_profile(
            r#"
default_policy_family: standard
uncached_isl_buckets:
  - min_tokens: 0
    bucket: all
policy_classes:
  - name: standard
    policy_family: standard
    cache_bucket: all
    quantum: 1
  - name: agents
    admission:
      type: session_aware
      scheduler_interval_seconds: 0
    quantum: 1
"#,
        );
        let mut policies = PolicyClassAdmissionPolicies::new();
        policies.insert("agents".to_owned(), Box::new(CustomPolicy));

        assert!(matches!(
            register_builtin_policies(&profile, workers(), 16, &mut policies),
            Err(RegistrationError::SessionAwareAdmissionControlConfig(_))
        ));
    }

    #[test]
    fn counts_configured_classes_before_honoring_caller_policies() {
        let profile = resolve_profile(
            r#"
default_policy_family: standard
uncached_isl_buckets:
  - min_tokens: 0
    bucket: all
policy_classes:
  - name: standard
    policy_family: standard
    cache_bucket: all
    quantum: 1
  - name: agents-a
    admission:
      type: session_aware
    quantum: 1
  - name: agents-b
    admission:
      type: session_aware
    quantum: 1
"#,
        );
        let mut policies = PolicyClassAdmissionPolicies::new();
        policies.insert("agents-a".to_owned(), Box::new(CustomPolicy));
        policies.insert("agents-b".to_owned(), Box::new(CustomPolicy));

        assert!(matches!(
            register_builtin_policies(&profile, workers(), 16, &mut policies),
            Err(RegistrationError::MultipleSessionAwareAdmissionControlClasses { .. })
        ));
    }

    #[test]
    fn rejects_session_aware_on_a_multi_bucket_family_class() {
        let profile = resolve_profile(
            r#"
default_policy_family: agents
uncached_isl_buckets:
  - min_tokens: 0
    bucket: cached
  - min_tokens: 4096
    bucket: uncached
policy_classes:
  - name: agents-cached
    policy_family: agents
    cache_bucket: cached
    admission:
      type: session_aware
    quantum: 1
  - name: agents-uncached
    policy_family: agents
    cache_bucket: uncached
    quantum: 1
"#,
        );

        assert!(matches!(
            register_builtin_policies(
                &profile,
                workers(),
                16,
                &mut PolicyClassAdmissionPolicies::new()
            ),
            Err(RegistrationError::SessionAwareAdmissionControlRequiresStableClass(_))
        ));
    }

    #[test]
    fn builds_session_aware_on_the_default_familys_only_bucket() {
        let profile = resolve_profile(
            r#"
default_policy_family: agents
uncached_isl_buckets:
  - min_tokens: 0
    bucket: all
policy_classes:
  - name: agents
    policy_family: agents
    cache_bucket: all
    admission:
      type: session_aware
    quantum: 1
"#,
        );
        let mut policies = PolicyClassAdmissionPolicies::new();

        register_builtin_policies(&profile, workers(), 16, &mut policies).unwrap();

        assert!(policies.contains_key("agents"));
    }

    #[test]
    fn rejects_unimplemented_policy_type() {
        let profile = resolve_profile(
            r#"
default_policy_family: standard
uncached_isl_buckets:
  - min_tokens: 0
    bucket: all
policy_classes:
  - name: agents
    policy_family: standard
    cache_bucket: all
    admission:
      type: misspelled
    quantum: 1
"#,
        );

        assert!(matches!(
            register_builtin_policies(
                &profile,
                workers(),
                16,
                &mut PolicyClassAdmissionPolicies::new()
            ),
            Err(RegistrationError::UnknownAdmissionPolicyType { .. })
        ));
    }

    #[test]
    fn preserves_caller_implementation_for_custom_policy_type() {
        let profile = resolve_profile(
            r#"
default_policy_family: standard
uncached_isl_buckets:
  - min_tokens: 0
    bucket: all
policy_classes:
  - name: agents
    policy_family: standard
    cache_bucket: all
    admission:
      type: custom
      custom_option: true
    quantum: 1
"#,
        );
        let mut policies = PolicyClassAdmissionPolicies::new();
        policies.insert("agents".to_owned(), Box::new(CustomPolicy));

        register_builtin_policies(&profile, workers(), 16, &mut policies).unwrap();

        assert_eq!(policies["agents"].reconcile_interval(), None);
    }
}
