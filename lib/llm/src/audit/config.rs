// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::sync::OnceLock;

#[derive(Clone, Copy)]
pub struct AuditPolicy {
    pub enabled: bool,
}

static POLICY: OnceLock<AuditPolicy> = OnceLock::new();

/// Audit is enabled if we have at least one sink
pub fn init_from_env() -> AuditPolicy {
    AuditPolicy {
        enabled: std::env::var("DYN_AUDIT_SINKS").is_ok(),
    }
}

pub fn policy() -> AuditPolicy {
    *POLICY.get_or_init(init_from_env)
}
