// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

mod direct_zmq;
mod subscriber;
mod target;
mod worker_query;
mod worker_query_endpoint;
mod worker_query_state;
mod worker_query_transport;

pub(crate) use subscriber::{
    KvEventSubscriptionHandle, RecoverySupervisor, start_subscriber, start_target_subscriber,
};
pub(crate) use target::{IndexerRecoveryTarget, RecoveryResetReason, RecoveryTarget, SourceEpoch};
pub(crate) use worker_query::DEFAULT_RECOVERY_ATTEMPT_TIMEOUT;
pub(crate) use worker_query::TargetFaultDisposition;
#[cfg(feature = "ckf-diagnostics")]
pub(crate) use worker_query::WorkerQueryHealthSnapshot;
pub(crate) use worker_query_endpoint::start_worker_kv_query_endpoint;
