// SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::sync::OnceLock;

#[derive(Clone, Copy)]
pub struct AuditPolicy {
    pub enabled: bool,
}

static POLICY: OnceLock<AuditPolicy> = OnceLock::new();

pub fn init_from_env() -> AuditPolicy {
    let enabled = std::env::var("DYN_AUDIT_ENABLED")
        .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
        .unwrap_or(false);
    AuditPolicy { enabled }
}

pub fn policy() -> AuditPolicy {
    *POLICY.get_or_init(init_from_env)
}
