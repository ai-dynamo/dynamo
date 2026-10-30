// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Best-effort session-to-pool affinity, independently of its persistence backend.

mod memory;
mod selector;
mod store;
mod tracker;

pub use memory::InMemoryAffinityStore;
pub use selector::{AffinityConfig, AffinitySelector};
pub use store::{AffinityBinding, AffinityError, AffinityKey, AffinityStore, WriteOutcome};
pub use tracker::{AffinitySnapshot, SessionAffinity, StoreBackedSessionAffinity};
