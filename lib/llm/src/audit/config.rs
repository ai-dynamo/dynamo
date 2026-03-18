// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::sync::OnceLock;

#[derive(Clone, Copy)]
pub struct AuditPolicy {
    pub enabled: bool,
    pub force_logging: bool,
}

static POLICY: OnceLock<AuditPolicy> = OnceLock::new();

/// Audit is enabled if we have at least one sink
pub fn init_from_env() -> AuditPolicy {
    AuditPolicy {
        enabled: std::env::var("DYN_AUDIT_SINKS").is_ok(),
        force_logging: std::env::var("DYN_AUDIT_FORCE_LOGGING")
            .ok()
            .and_then(|v| v.parse::<bool>().ok())
            .unwrap_or(false),
    }
}

pub fn policy() -> AuditPolicy {
    *POLICY.get_or_init(init_from_env)
}
