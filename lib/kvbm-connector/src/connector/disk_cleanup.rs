// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Process-global signal-driven cleanup for G3 disk-cache files.
//!
//! ### Cleanup architecture
//!
//! The primary mechanism is a *proactive unlink* in
//! `worker::init::pending`: after `TransferManager::register_layout`
//! registers the disk file with NIXL, the worker immediately removes the
//! directory entry while keeping the underlying fd open. The kernel keeps
//! the inode alive as long as the fd lives, and reclaims the disk on
//! *any* process exit — Ctrl+C, SIGKILL, vLLM IPC shutdown, segfault,
//! `os._exit`, etc.
//!
//! This module covers the narrow race window where the file exists on
//! disk but has not yet been NIXL-registered (i.e., between
//! `open(O_CREAT)` / `fallocate` and the proactive unlink). If a
//! terminating signal arrives during that window — and is actually
//! delivered to the Rust runtime — the registered tokio task unlinks
//! the file and exits with the conventional `128 + signum` exit code.
//!
//! Note: in vLLM-style deployments the EngineCore subprocess is shut
//! down via Python IPC rather than a delivered signal, so this task may
//! never fire there. The proactive unlink is what actually fixes the
//! orphan-file problem in that environment; this module is defensive.

use std::collections::HashSet;
use std::path::{Path, PathBuf};
use std::sync::{Mutex, OnceLock};

static G3_PATHS: OnceLock<Mutex<HashSet<PathBuf>>> = OnceLock::new();
static SIGNAL_TASK: OnceLock<()> = OnceLock::new();

fn registry() -> &'static Mutex<HashSet<PathBuf>> {
    G3_PATHS.get_or_init(|| Mutex::new(HashSet::new()))
}

/// Register a G3 cache file path for unlink on shutdown signals.
///
/// The first call also spawns a single tokio task that listens for
/// shutdown signals; subsequent calls just append to the registry.
/// Must be called from within a tokio runtime context.
pub fn register(path: PathBuf) {
    registry().lock().unwrap().insert(path);
    SIGNAL_TASK.get_or_init(|| {
        tokio::spawn(signal_task());
    });
}

/// Drop a path from the registry — used to keep the set bounded over the
/// lifetime of long-running processes. Safe to call for paths that were
/// never registered.
pub fn deregister(path: &Path) {
    if let Some(reg) = G3_PATHS.get() {
        reg.lock().unwrap().remove(path);
    }
}

async fn signal_task() {
    let Some((sig_name, sig_num)) = await_shutdown_signal().await else {
        return;
    };
    tracing::info!(
        signal = sig_name,
        "received shutdown signal, removing G3 cache files"
    );
    cleanup_all();

    // 128 + signal number is the conventional exit code for signal-terminated
    // processes. We exit explicitly rather than re-raising the signal so other
    // tokio signal listeners in the same process don't deadlock waiting for a
    // redelivered signal that's been swallowed by tokio's stream replacement.
    std::process::exit(128 + sig_num);
}

/// Wait for any of SIGINT/SIGTERM/SIGQUIT/SIGHUP and return its name + number.
/// Returns `None` if any signal stream fails to install.
async fn await_shutdown_signal() -> Option<(&'static str, i32)> {
    use tokio::signal::unix::{SignalKind, signal};

    let mut sigint = signal(SignalKind::interrupt())
        .map_err(|e| {
            tracing::warn!(error = %e, "G3 cleanup: failed to install SIGINT handler");
        })
        .ok()?;
    let mut sigterm = signal(SignalKind::terminate())
        .map_err(|e| {
            tracing::warn!(error = %e, "G3 cleanup: failed to install SIGTERM handler");
        })
        .ok()?;
    let mut sigquit = signal(SignalKind::quit())
        .map_err(|e| {
            tracing::warn!(error = %e, "G3 cleanup: failed to install SIGQUIT handler");
        })
        .ok()?;
    let mut sighup = signal(SignalKind::hangup())
        .map_err(|e| {
            tracing::warn!(error = %e, "G3 cleanup: failed to install SIGHUP handler");
        })
        .ok()?;

    Some(tokio::select! {
        _ = sigint.recv()  => ("SIGINT",  2i32),
        _ = sigterm.recv() => ("SIGTERM", 15i32),
        _ = sigquit.recv() => ("SIGQUIT", 3i32),
        _ = sighup.recv()  => ("SIGHUP",  1i32),
    })
}

fn cleanup_all() {
    let Some(reg) = G3_PATHS.get() else {
        return;
    };
    let paths: Vec<PathBuf> = {
        let guard = reg.lock().unwrap();
        guard.iter().cloned().collect()
    };
    for path in paths {
        match std::fs::remove_file(&path) {
            Ok(()) => tracing::info!(path = %path.display(), "removed G3 cache file"),
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => {}
            Err(e) => tracing::warn!(
                path = %path.display(),
                error = %e,
                "failed to remove G3 cache file on shutdown"
            ),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serial_test::serial;
    use std::time::Duration;
    use tempfile::TempDir;

    // The G3_PATHS registry is process-global, and `cleanup_all()` unlinks
    // every path in it. These tests share that registry, so they must run
    // serially — otherwise one test's `cleanup_all()` deletes another's
    // freshly registered file. All four are marked `#[serial]`.

    /// Insert a path into the registry without triggering the signal task —
    /// each test manages its own cleanup expectations.
    fn register_path_only(path: PathBuf) {
        registry().lock().unwrap().insert(path);
    }

    fn make_temp_file(dir: &TempDir, name: &str) -> PathBuf {
        let path = dir.path().join(name);
        std::fs::write(&path, b"placeholder").expect("write temp file");
        path
    }

    #[test]
    #[serial]
    fn cleanup_all_unlinks_registered_files() {
        let dir = TempDir::new().expect("tempdir");
        let p1 = make_temp_file(&dir, "kvbm_g3_test_a.bin");
        let p2 = make_temp_file(&dir, "kvbm_g3_test_b.bin");

        register_path_only(p1.clone());
        register_path_only(p2.clone());
        assert!(p1.exists() && p2.exists());

        cleanup_all();

        assert!(!p1.exists(), "p1 should have been removed");
        assert!(!p2.exists(), "p2 should have been removed");

        deregister(&p1);
        deregister(&p2);
    }

    #[test]
    #[serial]
    fn cleanup_all_tolerates_missing_files() {
        let dir = TempDir::new().expect("tempdir");
        let path = dir.path().join("kvbm_g3_test_already_gone.bin");
        // intentionally do not create the file
        register_path_only(path.clone());

        // Should not panic or return an error visible to the caller.
        cleanup_all();

        deregister(&path);
    }

    #[test]
    #[serial]
    fn deregister_removes_path_from_registry() {
        let dir = TempDir::new().expect("tempdir");
        let path = make_temp_file(&dir, "kvbm_g3_test_dereg.bin");
        register_path_only(path.clone());

        assert!(registry().lock().unwrap().contains(&path));
        deregister(&path);
        assert!(!registry().lock().unwrap().contains(&path));

        // File still exists because cleanup_all was never called.
        assert!(path.exists());
    }

    /// End-to-end: spawn the signal-await task, send ourselves SIGHUP,
    /// confirm the task observes it and that cleanup_all unlinks the file.
    ///
    /// Uses SIGHUP because tokio's stream replaces the default action, so
    /// raising it on ourselves is harmless — and it's far less likely than
    /// SIGTERM/SIGINT to be sent by external tooling (e.g. cargo, IDE)
    /// during the brief test window.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    #[serial]
    async fn signal_task_triggers_cleanup_on_sighup() {
        let dir = TempDir::new().expect("tempdir");
        let path = make_temp_file(&dir, "kvbm_g3_test_signal.bin");
        register_path_only(path.clone());
        assert!(path.exists());

        let waiter = tokio::spawn(async {
            tokio::time::timeout(Duration::from_secs(5), await_shutdown_signal())
                .await
                .expect("signal not received within 5s")
        });

        // Give the signal handlers a moment to install before we raise.
        tokio::time::sleep(Duration::from_millis(200)).await;

        // SAFETY: libc::raise is async-signal-safe and just marks the signal
        // pending on the current process; tokio's stream picks it up.
        let rc = unsafe { libc::raise(libc::SIGHUP) };
        assert_eq!(rc, 0, "raise(SIGHUP) failed");

        let signal_info = waiter.await.expect("waiter task panicked");
        assert_eq!(signal_info, Some(("SIGHUP", 1)));

        cleanup_all();
        assert!(
            !path.exists(),
            "file should have been removed by cleanup_all"
        );

        deregister(&path);
    }
}
