// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Test-time GPU and NIXL serialization gates.
//!
//! `cargo test -p kvbm-physical --features testing-kvbm` at the
//! default `--test-threads` (= cpu count) oversubscribes GPU memory
//! and serialises ports into NIXL/UCX init contention. These gates
//! cap the number of GPU- and NIXL-touching tests that may run
//! concurrently, regardless of the harness's `--test-threads` value.
//!
//! Two semaphores live behind `LazyLock`:
//!
//! - [`GPU_GATE`] — default permit count `2`. Serialises tests that
//!   touch CUDA `Device(_)` storage. Override via the
//!   `KVBM_TEST_GPU_PARALLELISM` env var.
//! - [`NIXL_GATE`] — default permit count `1`. UCX backend init
//!   contention is the binding constraint, so by default only one
//!   NIXL test runs at a time. Override via
//!   `KVBM_TEST_NIXL_PARALLELISM`.
//!
//! Both env vars are read once at first acquire. A non-numeric or
//! zero value falls back to the compile-time default.
//!
//! ## Macros
//!
//! Use these as the first line inside a `#[tokio::test]` body:
//!
//! - [`gpu_serial!`] — acquire one GPU permit. Drop on scope exit.
//! - [`nixl_serial!`] — acquire one NIXL permit AND one GPU permit
//!   (NIXL tests always do GPU work too in this codebase, and we
//!   want NIXL tests to count against the GPU budget too).
//! - [`storage_serial!`] — conditional GPU acquire keyed on a list
//!   of `StorageKind` values; only blocks when at least one kind is
//!   `Device(_)`. Mirrors the shape of [`skip_if_stubs_and_device!`]
//!   so parameterised rstest tests can annotate cleanly without
//!   needing a separate "device or not" branch.
//!
//! Sync `#[test]` tests in this crate are all pure-Rust validators
//! (no GPU, no NIXL), so no `gpu_serial_sync!` is provided.

use std::sync::LazyLock;

use tokio::sync::Semaphore;

const DEFAULT_GPU_PARALLELISM: usize = 2;
const DEFAULT_NIXL_PARALLELISM: usize = 1;

/// GPU concurrency gate. See module docs.
pub(crate) static GPU_GATE: LazyLock<Semaphore> = LazyLock::new(|| {
    Semaphore::new(parallelism_from_env(
        "KVBM_TEST_GPU_PARALLELISM",
        DEFAULT_GPU_PARALLELISM,
    ))
});

/// NIXL concurrency gate. See module docs.
pub(crate) static NIXL_GATE: LazyLock<Semaphore> = LazyLock::new(|| {
    Semaphore::new(parallelism_from_env(
        "KVBM_TEST_NIXL_PARALLELISM",
        DEFAULT_NIXL_PARALLELISM,
    ))
});

fn parallelism_from_env(env_var: &str, default: usize) -> usize {
    std::env::var(env_var)
        .ok()
        .and_then(|s| s.parse::<usize>().ok())
        .filter(|&n| n > 0)
        .unwrap_or(default)
}

/// Acquire one GPU permit, awaiting if the gate is full.
///
/// Returns a `'static`-bounded permit because [`GPU_GATE`] is a
/// `static`. Hold it in a `let _guard = ...;` binding until the end
/// of the test scope.
pub(crate) async fn gpu_gate() -> tokio::sync::SemaphorePermit<'static> {
    GPU_GATE
        .acquire()
        .await
        .expect("GPU_GATE semaphore was closed unexpectedly")
}

/// Acquire one NIXL permit, awaiting if the gate is full.
pub(crate) async fn nixl_gate() -> tokio::sync::SemaphorePermit<'static> {
    NIXL_GATE
        .acquire()
        .await
        .expect("NIXL_GATE semaphore was closed unexpectedly")
}

/// Acquire one GPU permit unconditionally (paired with [`storage_serial!`]).
///
/// Equivalent to `Some(gpu_gate().await)` — wrapped in `Option` so
/// callers can bind a single guard whose value is `None` for the
/// host-only case.
pub(crate) async fn gpu_gate_some() -> Option<tokio::sync::SemaphorePermit<'static>> {
    Some(gpu_gate().await)
}

/// Acquire one GPU permit, awaiting if the gate is full.
///
/// First line of a CUDA-touching `#[tokio::test]`:
/// ```ignore
/// #[tokio::test]
/// async fn my_cuda_test() -> Result<()> {
///     gpu_serial!();
///     // ... transfer through Device(0) ...
/// }
/// ```
#[allow(unused_macros)]
macro_rules! gpu_serial {
    () => {
        let _gpu_guard = $crate::transfer::tests::gate::gpu_gate().await;
    };
}

/// Acquire one NIXL permit AND one GPU permit. Use in NIXL-backed
/// tests that also touch CUDA Device storage (the common case in
/// this codebase).
///
/// ```ignore
/// #[tokio::test]
/// async fn my_nixl_test() -> Result<()> {
///     nixl_serial!();
///     // ... cross-agent transfer that touches Device(0) ...
/// }
/// ```
#[allow(unused_macros)]
macro_rules! nixl_serial {
    () => {
        let _nixl_guard = $crate::transfer::tests::gate::nixl_gate().await;
        let _gpu_guard = $crate::transfer::tests::gate::gpu_gate().await;
    };
}

/// Acquire ONLY the NIXL permit. Use for NIXL tests that don't touch
/// CUDA Device storage (e.g. Pinned ↔ Disk via the POSIX backend) —
/// holding the GPU gate would needlessly block unrelated GPU tests.
#[allow(unused_macros)]
macro_rules! nixl_only_serial {
    () => {
        let _nixl_guard = $crate::transfer::tests::gate::nixl_gate().await;
    };
}

/// Conditionally acquire a GPU permit when at least one of the
/// supplied [`StorageKind`] values is `Device(_)`.
///
/// Use in rstest-parameterised tests where some parameter
/// combinations are pure host and others reach into CUDA.
///
/// ```ignore
/// #[rstest]
/// #[tokio::test]
/// async fn test_p2p(
///     #[values(StorageKind::System, StorageKind::Device(0))] src_kind: StorageKind,
///     #[values(StorageKind::Pinned, StorageKind::Device(0))] dst_kind: StorageKind,
/// ) -> Result<()> {
///     storage_serial!(src_kind, dst_kind);
///     // ... transfer ...
/// }
/// ```
///
/// The bound guard is `Option<SemaphorePermit<'static>>` so the
/// host-only case binds `None` and is a no-op at scope exit.
#[allow(unused_macros)]
macro_rules! storage_serial {
    ($($kind:expr),+ $(,)?) => {
        let _gpu_guard = if false $(|| matches!($kind, $crate::transfer::StorageKind::Device(_)))+ {
            $crate::transfer::tests::gate::gpu_gate_some().await
        } else {
            None
        };
    };
}

// Re-export the macros so sibling modules can use them without
// wrestling with macro-namespace rules.
#[allow(unused_imports)]
pub(crate) use gpu_serial;
#[allow(unused_imports)]
pub(crate) use nixl_only_serial;
#[allow(unused_imports)]
pub(crate) use nixl_serial;
#[allow(unused_imports)]
pub(crate) use storage_serial;
