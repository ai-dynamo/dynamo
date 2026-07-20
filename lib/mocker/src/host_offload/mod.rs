// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Framework-native host-offload simulation primitives.
//!
//! The types in this module deliberately avoid framework and KVBM identifiers.
//! Framework adapters resolve their behavior into [`ResolvedHostOffloadPolicy`]
//! and interact with the offload core through logical block keys and opaque G1
//! locations.

mod events;
mod eviction;
mod manager;
mod policy;

pub use events::{
    G1Location, HostBlockKey, HostOffloadEvent, HostOffloadEventSink, NoopHostOffloadEventSink,
    SourceFenceReason, TransferDirection, TransferId,
};
pub use manager::{
    CompletedTransfer, G2Lookup, HostOffloadConfig, HostOffloadEffects, HostOffloadManager,
    LoadBlock, LoadScheduleOutcome, PrepareStoreOutcome, SourceFence, SourceFenceOutcome,
    StoreBlock, SubmittedStore,
};
pub use policy::{
    CapacityHandling, G2EvictionPolicy, LoadAdmissionHeadroom, LoadExecution, LoadFence,
    LookupOrder, LookupTiming, LookupTouch, PostLoadResidency, ResolvedHostOffloadPolicy,
    SchedulerRetry, StoreAdmission, StoreCadence, StoreCohort, StoreExecution, StoreFence,
    StoreScope, StoreSubmitTiming, StoreTrigger, TransferExecution,
};
