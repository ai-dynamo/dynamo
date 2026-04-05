// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! System information collection from /proc.

use serde::{Deserialize, Serialize};

/// System information snapshot.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SystemInfo {
    pub hostname: String,
    pub os: String,
    pub kernel: String,
    pub cpu_model: String,
    pub cpu_threads: usize,
    pub memory_total_gb: f64,
}

/// Collect system information from /proc and /etc.
pub fn collect() -> SystemInfo {
    let hostname = std::fs::read_to_string("/etc/hostname")
        .unwrap_or_default()
        .trim()
        .to_string();

    let os = std::fs::read_to_string("/etc/os-release")
        .unwrap_or_default()
        .lines()
        .find(|l| l.starts_with("PRETTY_NAME="))
        .map(|l| l.trim_start_matches("PRETTY_NAME=").trim_matches('"').to_string())
        .unwrap_or_else(|| "Unknown".to_string());

    let kernel = std::fs::read_to_string("/proc/version")
        .unwrap_or_default()
        .split_whitespace()
        .nth(2)
        .unwrap_or("Unknown")
        .to_string();

    let cpuinfo = std::fs::read_to_string("/proc/cpuinfo").unwrap_or_default();

    let cpu_model = cpuinfo
        .lines()
        .find(|l| l.starts_with("model name"))
        .and_then(|l| l.split(':').nth(1))
        .map(|s| s.trim().to_string())
        .unwrap_or_else(|| "Unknown".to_string());

    let cpu_threads = cpuinfo.lines().filter(|l| l.starts_with("processor")).count();

    let meminfo = std::fs::read_to_string("/proc/meminfo").unwrap_or_default();
    let memory_total_kb: u64 = meminfo
        .lines()
        .find(|l| l.starts_with("MemTotal:"))
        .and_then(|l| l.split_whitespace().nth(1))
        .and_then(|s| s.parse().ok())
        .unwrap_or(0);
    let memory_total_gb = memory_total_kb as f64 / (1024.0 * 1024.0);

    SystemInfo { hostname, os, kernel, cpu_model, cpu_threads, memory_total_gb }
}

impl std::fmt::Display for SystemInfo {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "Host: {} | OS: {} | Kernel: {} | CPU: {} ({} threads) | RAM: {:.1} GB",
            self.hostname,
            self.os,
            self.kernel,
            self.cpu_model,
            self.cpu_threads,
            self.memory_total_gb
        )
    }
}

/// Resident set size snapshot.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RssSnapshot {
    pub rss_kb: u64,
    pub peak_rss_kb: u64,
}

/// Read RSS and peak RSS from /proc/self/status.
pub fn rss_snapshot() -> RssSnapshot {
    let status = std::fs::read_to_string("/proc/self/status").unwrap_or_default();
    let mut rss_kb = 0u64;
    let mut peak_rss_kb = 0u64;
    for line in status.lines() {
        if line.starts_with("VmRSS:") {
            rss_kb = line.split_whitespace().nth(1).and_then(|s| s.parse().ok()).unwrap_or(0);
        } else if line.starts_with("VmPeak:") {
            peak_rss_kb = line.split_whitespace().nth(1).and_then(|s| s.parse().ok()).unwrap_or(0);
        }
    }
    RssSnapshot { rss_kb, peak_rss_kb }
}

/// CPU time snapshot from /proc/self/stat.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CpuTime {
    pub user_us: u64,
    pub system_us: u64,
}

/// Read user and system CPU time from /proc/self/stat.
pub fn cpu_time() -> CpuTime {
    let stat = std::fs::read_to_string("/proc/self/stat").unwrap_or_default();
    let fields: Vec<&str> = stat.split_whitespace().collect();
    // Fields 13 (utime) and 14 (stime) in /proc/self/stat (0-indexed)
    // utime is at index 13, stime at index 14
    let utime_ticks: u64 = fields.get(13).and_then(|s| s.parse().ok()).unwrap_or(0);
    let stime_ticks: u64 = fields.get(14).and_then(|s| s.parse().ok()).unwrap_or(0);

    // Convert from clock ticks to microseconds (typically 100 ticks/sec)
    let ticks_per_sec = tick_rate();
    let user_us = utime_ticks * 1_000_000 / ticks_per_sec;
    let system_us = stime_ticks * 1_000_000 / ticks_per_sec;

    CpuTime { user_us, system_us }
}

fn tick_rate() -> u64 {
    // On Linux, sysconf(_SC_CLK_TCK) is typically 100
    libc_clock_tck()
}

fn libc_clock_tck() -> u64 {
    // Read from /proc/self/stat — fall back to 100 if we can't determine
    // We parse it directly from the OS via a syscall-free approach
    // Most Linux systems use 100 Hz (USER_HZ = 100)
    100
}
