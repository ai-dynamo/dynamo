// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! File-based mutual-exclusion lock for sysprofile captures.
//!
//! Only one capture can run per host at a time. The lockfile is created
//! atomically via `O_CREAT | O_EXCL` and removed on drop (or on explicit
//! release). If a previous capture crashed, the lockfile includes a PID
//! that can be checked for staleness.

use std::fs;
use std::io::Write;
use std::path::{Path, PathBuf};

#[derive(Debug)]
pub struct CaptureLock {
    path: PathBuf,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
struct LockInfo {
    run_id: String,
    pid: u32,
    started_at: String,
}

impl CaptureLock {
    pub fn acquire(dir: &Path, run_id: &str) -> anyhow::Result<Self> {
        fs::create_dir_all(dir)?;
        let path = dir.join(".sysprofile.lock");

        if path.exists() {
            if let Ok(contents) = fs::read_to_string(&path) {
                if let Ok(info) = serde_json::from_str::<LockInfo>(&contents) {
                    if is_pid_alive(info.pid) {
                        anyhow::bail!(
                            "capture already in progress: run_id={}, pid={} (lockfile: {})",
                            info.run_id,
                            info.pid,
                            path.display()
                        );
                    }
                    tracing::warn!(
                        old_run_id = %info.run_id,
                        old_pid = info.pid,
                        "removing stale lockfile from crashed capture"
                    );
                }
            }
            fs::remove_file(&path)?;
        }

        let info = LockInfo {
            run_id: run_id.to_string(),
            pid: std::process::id(),
            started_at: chrono::Utc::now().to_rfc3339(),
        };

        let mut file = fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&path)?;
        file.write_all(serde_json::to_string_pretty(&info)?.as_bytes())?;

        tracing::info!(run_id, path = %path.display(), "capture lock acquired");
        Ok(Self { path })
    }

    pub fn release(self) -> anyhow::Result<()> {
        // Drop will also clean up, but this gives explicit error reporting
        if self.path.exists() {
            fs::remove_file(&self.path)?;
            tracing::info!(path = %self.path.display(), "capture lock released");
        }
        std::mem::forget(self); // prevent double-remove in Drop
        Ok(())
    }
}

impl Drop for CaptureLock {
    fn drop(&mut self) {
        let _ = fs::remove_file(&self.path);
    }
}

fn is_pid_alive(pid: u32) -> bool {
    #[cfg(unix)]
    {
        unsafe { libc::kill(pid as i32, 0) == 0 }
    }
    #[cfg(not(unix))]
    {
        let _ = pid;
        false
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn acquire_and_release() {
        let dir = tempfile::tempdir().unwrap();
        let lock = CaptureLock::acquire(dir.path(), "test-run").unwrap();
        assert!(dir.path().join(".sysprofile.lock").exists());
        lock.release().unwrap();
        assert!(!dir.path().join(".sysprofile.lock").exists());
    }

    #[test]
    fn double_acquire_fails() {
        let dir = tempfile::tempdir().unwrap();
        let _lock = CaptureLock::acquire(dir.path(), "run-1").unwrap();
        let err = CaptureLock::acquire(dir.path(), "run-2");
        assert!(err.is_err());
    }

    #[test]
    fn stale_lock_is_cleaned() {
        let dir = tempfile::tempdir().unwrap();
        let lock_path = dir.path().join(".sysprofile.lock");
        let info = LockInfo {
            run_id: "old-run".into(),
            pid: 999999999, // almost certainly not alive
            started_at: "2025-01-01T00:00:00Z".into(),
        };
        fs::write(&lock_path, serde_json::to_string(&info).unwrap()).unwrap();

        let lock = CaptureLock::acquire(dir.path(), "new-run").unwrap();
        assert!(lock_path.exists());
        lock.release().unwrap();
    }

    #[test]
    fn drop_cleans_up() {
        let dir = tempfile::tempdir().unwrap();
        {
            let _lock = CaptureLock::acquire(dir.path(), "test-run").unwrap();
        } // drop here
        assert!(!dir.path().join(".sysprofile.lock").exists());
    }
}
