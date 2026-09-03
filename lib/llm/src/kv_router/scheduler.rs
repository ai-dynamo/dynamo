// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Scheduling types shared by the router. The router's scheduler is one
//! partition of an embedded selection service (`kv_router::embedded`); these
//! re-exports are the runtime-free scheduling vocabulary it speaks.

pub use dynamo_kv_router::scheduling::overlap_refresh::{
    NoopOverlapScoresRefresh, OverlapScoresRefresh, RefreshedOverlap,
};
pub use dynamo_kv_router::scheduling::queue::{
    SchedulerBookingCleanup, SchedulerBookingDescriptor,
};
pub use dynamo_kv_router::scheduling::{
    AdmittedSchedulingResponse, AdvisorySchedulingResponse, AttemptId, KvSchedulerError,
    LocalScheduler, NonMaxOverlapSelectionObserver, OverloadedWorkerProvider, PotentialLoad,
    ScheduleRequest, SchedulingRequest, SchedulingResponse, TierOverlapBlocks,
    WorkerAvailabilityProvider,
};
pub use dynamo_kv_router::selector::DefaultWorkerSelector;
