// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! `DYN_SYSPROFILE_*` environment variable parsing and global configuration.

use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::OnceLock;

static ENABLED: AtomicBool = AtomicBool::new(false);
static CONFIG: OnceLock<SysprofileConfig> = OnceLock::new();

#[derive(Debug, Clone)]
pub struct SysprofileConfig {
    pub enabled: bool,
    pub dir: PathBuf,
    pub sampling: f64,
    pub backends: Vec<String>,
    pub run_id: String,
    pub flush_timeout_s: u64,
}

impl Default for SysprofileConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            dir: PathBuf::from("/data/sysprofile"),
            sampling: 0.10,
            backends: vec!["vllm".into()],
            run_id: String::new(),
            flush_timeout_s: 900,
        }
    }
}

impl SysprofileConfig {
    pub fn from_env() -> Self {
        let enabled = std::env::var("DYN_SYSPROFILE_ENABLE")
            .map(|v| matches!(v.to_lowercase().as_str(), "1" | "true" | "yes" | "on"))
            .unwrap_or(false);

        let dir = std::env::var("DYN_SYSPROFILE_DIR")
            .map(PathBuf::from)
            .unwrap_or_else(|_| PathBuf::from("/data/sysprofile"));

        let sampling = std::env::var("DYN_SYSPROFILE_SAMPLING")
            .ok()
            .and_then(|v| v.parse::<f64>().ok())
            .unwrap_or(0.10)
            .clamp(0.0, 1.0);

        let backends = std::env::var("DYN_SYSPROFILE_BACKENDS")
            .map(|v| v.split(',').map(|s| s.trim().to_string()).collect())
            .unwrap_or_else(|_| vec!["vllm".into()]);

        let run_id = std::env::var("DYN_SYSPROFILE_RUN_ID")
            .unwrap_or_else(|_| uuid::Uuid::new_v4().to_string());

        let flush_timeout_s = std::env::var("DYN_SYSPROFILE_FLUSH_TIMEOUT_S")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(900);

        Self {
            enabled,
            dir,
            sampling,
            backends,
            run_id,
            flush_timeout_s,
        }
    }

    pub fn run_dir(&self) -> PathBuf {
        self.dir.join(&self.run_id)
    }

    pub fn should_sample(&self, trace_id: &str) -> bool {
        if !self.enabled {
            return false;
        }
        if self.sampling >= 1.0 {
            return true;
        }
        let hash = hash_trace_id(trace_id);
        let threshold = (self.sampling * u64::MAX as f64) as u64;
        hash < threshold
    }
}

fn hash_trace_id(trace_id: &str) -> u64 {
    let mut h: u64 = 0xcbf29ce484222325;
    for b in trace_id.as_bytes() {
        h ^= *b as u64;
        h = h.wrapping_mul(0x100000001b3);
    }
    h
}

pub fn init() {
    let cfg = SysprofileConfig::from_env();
    let is_enabled = cfg.enabled;
    ENABLED.store(is_enabled, Ordering::Relaxed);
    let _ = CONFIG.set(cfg);
    if is_enabled {
        tracing::info!(
            run_id = %global_config().run_id,
            sampling = %global_config().sampling,
            dir = %global_config().dir.display(),
            "sysprofile enabled"
        );
    }
}

#[inline(always)]
pub fn enabled() -> bool {
    ENABLED.load(Ordering::Relaxed)
}

pub fn global_config() -> &'static SysprofileConfig {
    CONFIG.get_or_init(SysprofileConfig::default)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_config() {
        let cfg = SysprofileConfig::default();
        assert!(!cfg.enabled);
        assert!((cfg.sampling - 0.10).abs() < f64::EPSILON);
    }

    #[test]
    fn sampling_deterministic() {
        let cfg = SysprofileConfig {
            enabled: true,
            sampling: 0.5,
            ..Default::default()
        };
        let result1 = cfg.should_sample("abc123");
        let result2 = cfg.should_sample("abc123");
        assert_eq!(result1, result2);
    }

    #[test]
    fn sampling_zero_rejects_all() {
        let cfg = SysprofileConfig {
            enabled: true,
            sampling: 0.0,
            ..Default::default()
        };
        for i in 0..100 {
            assert!(!cfg.should_sample(&format!("trace-{i}")));
        }
    }

    #[test]
    fn sampling_one_accepts_all() {
        let cfg = SysprofileConfig {
            enabled: true,
            sampling: 1.0,
            ..Default::default()
        };
        for i in 0..100 {
            assert!(cfg.should_sample(&format!("trace-{i}")));
        }
    }
}
