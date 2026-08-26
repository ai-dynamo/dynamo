// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Process-wide runtime wiring for [`Worker`].
//!
//! These live in their own integration-test binary because `Worker` keeps the runtime, its
//! config, and the compute-pool claim in process-global `OnceCell`s that can only be
//! initialized once. A single test per process is the only way to observe first-call
//! behaviour, so this file deliberately holds one test.

use dynamo_runtime::Worker;

/// The first [`Runtime`] wrapper carries the compute pool; later ones share the tokio runtime
/// without spawning a second Rayon pool.
///
/// Both halves matter, and call order must not decide either. `DistributedRuntime::new` calls
/// `ensure_process_runtime` first to set up the pyo3 bridge, so any rule phrased as "did I just
/// create the runtime?" drops the pool on the frontend's own path; a rule that attaches one
/// unconditionally spawns a Rayon pool per `DistributedRuntime`.
#[test]
fn first_runtime_wrapper_owns_the_compute_pool() {
    // Mirror `DistributedRuntime::new`: ensure the process runtime up front, as the bridge
    // requires, and only then build the wrapper.
    let _primary = Worker::ensure_process_runtime().expect("ensure_process_runtime failed");

    let first = Worker::runtime_from_existing().expect("first runtime_from_existing failed");
    assert!(
        first.compute_pool().is_some(),
        "first wrapper should carry the config-derived compute pool even though \
         ensure_process_runtime ran first"
    );

    let second = Worker::runtime_from_existing().expect("second runtime_from_existing failed");
    assert!(
        second.compute_pool().is_none(),
        "later wrappers must reuse the runtime without spawning another Rayon pool"
    );

    assert_eq!(
        first.primary().id(),
        second.primary().id(),
        "both wrappers should be backed by the same tokio runtime"
    );
}
