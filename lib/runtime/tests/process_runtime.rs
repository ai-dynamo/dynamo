// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Process-wide runtime wiring for `Worker`.
//!
//! Its own test binary because `Worker` keeps the runtime, its config, and the compute-pool
//! claim in process-global `OnceCell`s. First-call behaviour can only be observed once per
//! process, so this file deliberately holds a single test.

use dynamo_runtime::Worker;

/// The first `Runtime` wrapper gets the compute pool; later ones share the Tokio runtime without
/// spawning a second Rayon pool.
///
/// Call order must not decide either half. `DistributedRuntime::new` calls
/// `ensure_process_runtime` first, so a rule like "did I just create the runtime?" would drop the
/// pool on the frontend's own path, while attaching one every time would spawn a Rayon pool per
/// `DistributedRuntime`.
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
