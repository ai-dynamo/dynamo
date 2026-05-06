// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::sync::OnceLock;

use dynamo_runtime::config::environment_names::llm::audit as env_audit;

const DEFAULT_CAPACITY: usize = 1024;

#[derive(Clone, Copy, Debug)]
pub struct AuditPolicy {
    pub enabled: bool,
    pub force_logging: bool,
    pub capacity: usize,
}

static POLICY: OnceLock<AuditPolicy> = OnceLock::new();

/// Audit is enabled if we have at least one sink
fn load_from_env() -> AuditPolicy {
    let capacity = std::env::var(env_audit::DYN_AUDIT_CAPACITY)
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .filter(|v| *v > 0)
        .unwrap_or(DEFAULT_CAPACITY);
    AuditPolicy {
        enabled: std::env::var(env_audit::DYN_AUDIT_SINKS).is_ok(),
        force_logging: std::env::var(env_audit::DYN_AUDIT_FORCE_LOGGING)
            .ok()
            .and_then(|v| v.parse::<bool>().ok())
            .unwrap_or(false),
        capacity,
    }
}

pub fn policy() -> AuditPolicy {
    *POLICY.get_or_init(load_from_env)
}
