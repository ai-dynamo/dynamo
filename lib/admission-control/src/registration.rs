// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::collections::HashMap;
use std::time::Duration;

use dynamo_kv_router::RouterQueuePolicy;
use dynamo_kv_router::protocols::{WorkerConfigLike, WorkerId};
use dynamo_kv_router::scheduling::{PolicyClassAdmissionStrategies, PolicyProfile};
use thiserror::Error;
use tokio::sync::watch;

use crate::{ConfigError, STRATEGY_NAME, ThunderAgent, ThunderAgentConfig, WatchWorkerCapacity};

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
    #[error("admission strategy reconcile interval must be positive")]
    ZeroReconcileInterval,
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

pub fn strategy_recheck_interval(
    strategies: &PolicyClassAdmissionStrategies,
) -> Result<Option<Duration>, RegistrationError> {
    let mut minimum = None;
    for interval in strategies
        .values()
        .filter_map(|strategy| strategy.reconcile_interval())
    {
        if interval.is_zero() {
            return Err(RegistrationError::ZeroReconcileInterval);
        }
        minimum = Some(minimum.map_or(interval, |current: Duration| current.min(interval)));
    }
    Ok(minimum)
}
