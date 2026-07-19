// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! DC-scoped KV-cache Relay and endpoint-independent CKF identity boundary.

mod actor;
mod discovery;
mod host;
mod resolution;

pub use host::{
    DEFAULT_EXPECTED_UNIQUE_BLOCKS, KvDcRelay, KvDcRelayConfig, KvDcRelayError, KvDcRelayHealth,
};

#[cfg(feature = "ckf-diagnostics")]
pub use host::{
    KvDcRelayActorStats, KvDcRelayAggregationStats, KvDcRelayCacheDomainStats,
    KvDcRelayDiagnosticSnapshot, KvDcRelayEndpointStats, KvDcRelayIdentityStats,
    KvDcRelayMemberStats, KvDcRelayMemoryStats, KvDcRelayPublicationStats, KvDcRelayRecoveryStats,
    KvDcRelayStats,
};
