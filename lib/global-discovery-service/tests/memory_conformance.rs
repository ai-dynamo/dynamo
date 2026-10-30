// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! The conformance cases (see `common/mod.rs`) against the memory store, with
//! Tokio's clock paused so the test moves time.

mod common;

use std::sync::Arc;

use common::{ClockHandle, conformance, lease};
use dynamo_global_discovery_service::backends::memory::MemoryPoolStore;
use dynamo_global_discovery_service::pool_discovery::{Completeness, PoolDirectory, PoolRegistrar};

/// The store, seen as the two traits the cases use. The caller keeps the
/// returned `Arc`s alive: dropping the last one stops the store's sweep.
fn store() -> (Arc<dyn PoolRegistrar>, Arc<PoolDirectory>, ClockHandle) {
    let store = Arc::new(MemoryPoolStore::new(lease()));
    (store.clone(), store, ClockHandle::Tokio)
}

macro_rules! conforms {
    ($($case:ident),+ $(,)?) => {$(
        #[tokio::test(start_paused = true)]
        async fn $case() {
            let (registrar, pools, clock) = store();
            conformance::$case(registrar, pools, clock).await;
        }
    )+};
}

conforms!(
    register_answers_the_configured_lease,
    a_registered_pool_is_listed_normalised,
    the_last_registration_wins,
    distinct_keys_are_distinct_pools,
    registering_again_only_renews,
    a_missed_lease_hides_then_removes_the_pool,
    only_the_pool_that_missed_its_lease_expires,
);

#[tokio::test(start_paused = true)]
async fn lists_are_exact_once_a_lease_has_passed() {
    let (registrar, pools, clock) = store();
    // A memory store that just started has only the registrations it got.
    conformance::lists_are_exact_once_a_lease_has_passed(
        registrar,
        pools,
        clock,
        Completeness::Estimated,
    )
    .await;
}
