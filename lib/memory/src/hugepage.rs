// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Read-only discovery of the host's hugepage state.
//!
//! Hugepage *allocation* lives in [`crate::mmap_pinned`]; this module only
//! tells callers what the kernel is willing to give them. Reads:
//!
//! - `/proc/meminfo` → default hugepage size, system-wide totals
//! - `/sys/kernel/mm/hugepages/hugepages-{kB}/{nr,free,resv,surplus}_hugepages`
//! - `/sys/devices/system/node/node{N}/hugepages/hugepages-{kB}/{nr,free,surplus}_hugepages`
//! - `/sys/kernel/mm/transparent_hugepage/enabled`
//!
//! Never errors — degraded fields surface as `0` / `Unknown` so callers can
//! still report a useful snapshot on hosts where sysfs is partial.

use std::collections::BTreeMap;
use std::fs;
use std::path::Path;

use serde::{Deserialize, Serialize};

use crate::numa::NumaNode;

/// Snapshot of the host's hugepage configuration.
///
/// Built by [`Self::discover`]. Pure inspection — no allocation, no state.
/// [`Default`] yields an empty snapshot (no pools, `Unknown` THP, `0`
/// default size) — useful for tests and for representing "not yet probed".
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct HugepageInfo {
    /// Default hugepage size in bytes (the `Hugepagesize:` field of
    /// `/proc/meminfo`). Typically 2 MiB on x86 and Grace; some kernels
    /// configure it to 1 GiB. `0` if unreadable.
    pub default_size_bytes: usize,
    /// Transparent-hugepage policy as reported by
    /// `/sys/kernel/mm/transparent_hugepage/enabled`.
    pub thp_enabled: ThpMode,
    /// System-wide pool stats, one entry per page size present in
    /// `/sys/kernel/mm/hugepages/`. Sorted by page size ascending.
    pub pools: Vec<HugepagePool>,
    /// Per-NUMA-node pool stats, parsed from
    /// `/sys/devices/system/node/node*/hugepages/`. Sorted by node id.
    pub per_node: Vec<PerNodeHugepages>,
}

/// Reservation state for one hugepage size at the system or per-node level.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct HugepagePool {
    /// Page size for this pool, in bytes (e.g. `2 * 1024 * 1024`).
    pub page_size_bytes: usize,
    /// Configured page count (`nr_hugepages`).
    pub nr_pages: u64,
    /// Pages currently unallocated and available.
    pub free_pages: u64,
    /// Pages reserved by callers via `mmap(MAP_HUGETLB)` that are not yet
    /// faulted in. Per-node sysfs does not expose this — node-level pools
    /// report `0`.
    pub resv_pages: u64,
    /// Overcommit pages allocated above `nr_pages` (governed by
    /// `nr_overcommit_hugepages`).
    pub surplus_pages: u64,
}

impl HugepagePool {
    /// Total bytes the pool is configured for (`nr_pages * page_size`),
    /// regardless of how many are currently free.
    pub fn capacity_bytes(&self) -> u64 {
        self.nr_pages.saturating_mul(self.page_size_bytes as u64)
    }

    /// Bytes currently free in this pool.
    pub fn free_bytes(&self) -> u64 {
        self.free_pages.saturating_mul(self.page_size_bytes as u64)
    }
}

/// Per-NUMA-node hugepage reservation. Mirrors what
/// `/sys/devices/system/node/node{N}/hugepages/hugepages-{kB}/` exposes.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerNodeHugepages {
    /// NUMA node these pools belong to.
    pub node: NumaNode,
    /// One entry per page size configured on this node.
    pub pools: Vec<HugepagePool>,
}

/// Transparent-hugepage policy. Mirrors the bracketed value in
/// `/sys/kernel/mm/transparent_hugepage/enabled`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
pub enum ThpMode {
    /// Kernel applies THP to any anon mapping.
    Always,
    /// Kernel applies THP only when the mapping is hinted with
    /// `madvise(MADV_HUGEPAGE)`.
    Madvise,
    /// THP disabled.
    Never,
    /// File missing or contents unparseable.
    #[default]
    Unknown,
}

impl std::fmt::Display for ThpMode {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Always => f.write_str("always"),
            Self::Madvise => f.write_str("madvise"),
            Self::Never => f.write_str("never"),
            Self::Unknown => f.write_str("unknown"),
        }
    }
}

impl HugepageInfo {
    /// Read the host's hugepage state from `/proc` and `/sys`. Never fails.
    pub fn discover() -> Self {
        let meminfo = fs::read_to_string("/proc/meminfo").unwrap_or_default();
        let default_size_bytes = parse_hugepagesize(&meminfo).unwrap_or(0);

        let thp_enabled = match fs::read_to_string("/sys/kernel/mm/transparent_hugepage/enabled") {
            Ok(s) => parse_thp_enabled(&s),
            Err(_) => ThpMode::Unknown,
        };

        let pools = read_system_hugepage_pools();
        let per_node = read_per_node_hugepages();

        Self {
            default_size_bytes,
            thp_enabled,
            pools,
            per_node,
        }
    }

    /// Total system-wide bytes configured for the given page size, or `0`
    /// if no pool exists at that size.
    pub fn capacity_bytes(&self, page_size: usize) -> u64 {
        self.pools
            .iter()
            .find(|p| p.page_size_bytes == page_size)
            .map(HugepagePool::capacity_bytes)
            .unwrap_or(0)
    }

    /// Bytes configured for the given page size on the given NUMA node.
    pub fn capacity_bytes_on_node(&self, node: NumaNode, page_size: usize) -> u64 {
        self.per_node
            .iter()
            .find(|n| n.node == node)
            .and_then(|n| n.pools.iter().find(|p| p.page_size_bytes == page_size))
            .map(HugepagePool::capacity_bytes)
            .unwrap_or(0)
    }
}

impl std::fmt::Display for HugepageInfo {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(
            f,
            "  default page size:    {}",
            if self.default_size_bytes == 0 {
                "?".to_string()
            } else {
                crate::resources::format_bytes(self.default_size_bytes as u64)
            }
        )?;
        writeln!(f, "  THP enabled:          {}", self.thp_enabled)?;
        if self.pools.is_empty() {
            writeln!(f, "  system-wide pools:    (none)")?;
        } else {
            writeln!(f, "  system-wide pools:")?;
            for p in &self.pools {
                writeln!(
                    f,
                    "    {} pages  nr={}  free={}  resv={}  surplus={}",
                    crate::resources::format_bytes(p.page_size_bytes as u64),
                    p.nr_pages,
                    p.free_pages,
                    p.resv_pages,
                    p.surplus_pages,
                )?;
            }
        }
        if !self.per_node.is_empty() {
            writeln!(f, "  per-node pools:")?;
            for n in &self.per_node {
                for p in &n.pools {
                    writeln!(
                        f,
                        "    node {}  {} pages  nr={}  free={}  surplus={}",
                        n.node.0,
                        crate::resources::format_bytes(p.page_size_bytes as u64),
                        p.nr_pages,
                        p.free_pages,
                        p.surplus_pages,
                    )?;
                }
            }
        }
        Ok(())
    }
}

/// Parse `Hugepagesize:` out of `/proc/meminfo`. Returns bytes.
fn parse_hugepagesize(meminfo: &str) -> Option<usize> {
    for line in meminfo.lines() {
        if let Some(rest) = line.strip_prefix("Hugepagesize:") {
            let mut it = rest.split_whitespace();
            let value: usize = it.next()?.parse().ok()?;
            let unit = it.next().unwrap_or("kB");
            let mult = match unit {
                "kB" | "KB" => 1024usize,
                "MB" => 1024 * 1024,
                "GB" => 1024 * 1024 * 1024,
                _ => 1024,
            };
            return Some(value.saturating_mul(mult));
        }
    }
    None
}

/// Parse `always [madvise] never` style transparent_hugepage/enabled file.
fn parse_thp_enabled(content: &str) -> ThpMode {
    let trimmed = content.trim();
    // Active mode is bracketed: "always [madvise] never"
    for token in trimmed.split_whitespace() {
        if let Some(inner) = token.strip_prefix('[').and_then(|s| s.strip_suffix(']')) {
            return match inner {
                "always" => ThpMode::Always,
                "madvise" => ThpMode::Madvise,
                "never" => ThpMode::Never,
                _ => ThpMode::Unknown,
            };
        }
    }
    ThpMode::Unknown
}

fn read_system_hugepage_pools() -> Vec<HugepagePool> {
    read_hugepage_pools_dir(Path::new("/sys/kernel/mm/hugepages"), true)
}

fn read_per_node_hugepages() -> Vec<PerNodeHugepages> {
    let mut out: BTreeMap<u32, Vec<HugepagePool>> = BTreeMap::new();
    let nodes_dir = Path::new("/sys/devices/system/node");
    let entries = match fs::read_dir(nodes_dir) {
        Ok(e) => e,
        Err(_) => return Vec::new(),
    };
    for entry in entries.flatten() {
        let path = entry.path();
        let name = match path.file_name().and_then(|n| n.to_str()) {
            Some(n) => n.to_string(),
            None => continue,
        };
        if !name.starts_with("node") {
            continue;
        }
        let node_id: u32 = match name[4..].parse() {
            Ok(v) => v,
            Err(_) => continue,
        };
        let hp_dir = path.join("hugepages");
        if !hp_dir.exists() {
            continue;
        }
        let pools = read_hugepage_pools_dir(&hp_dir, false);
        if !pools.is_empty() {
            out.insert(node_id, pools);
        }
    }
    out.into_iter()
        .map(|(node, pools)| PerNodeHugepages {
            node: NumaNode(node),
            pools,
        })
        .collect()
}

/// Walk a `hugepages/` directory (system or per-node) reading every
/// `hugepages-{kB}` subdirectory. `with_resv` controls whether to look for
/// `resv_hugepages` (system-wide has it; per-node doesn't).
fn read_hugepage_pools_dir(dir: &Path, with_resv: bool) -> Vec<HugepagePool> {
    let mut pools = Vec::new();
    let entries = match fs::read_dir(dir) {
        Ok(e) => e,
        Err(_) => return pools,
    };
    for entry in entries.flatten() {
        let path = entry.path();
        let name = match path.file_name().and_then(|n| n.to_str()) {
            Some(n) => n.to_string(),
            None => continue,
        };
        let kb_part = match name
            .strip_prefix("hugepages-")
            .and_then(|s| s.strip_suffix("kB"))
        {
            Some(p) => p,
            None => continue,
        };
        let kb: usize = match kb_part.parse() {
            Ok(v) => v,
            Err(_) => continue,
        };
        let page_size_bytes = kb.saturating_mul(1024);
        let nr_pages = read_count(&path.join("nr_hugepages"));
        let free_pages = read_count(&path.join("free_hugepages"));
        let surplus_pages = read_count(&path.join("surplus_hugepages"));
        let resv_pages = if with_resv {
            read_count(&path.join("resv_hugepages"))
        } else {
            0
        };
        pools.push(HugepagePool {
            page_size_bytes,
            nr_pages,
            free_pages,
            resv_pages,
            surplus_pages,
        });
    }
    pools.sort_by_key(|p| p.page_size_bytes);
    pools
}

fn read_count(path: &Path) -> u64 {
    fs::read_to_string(path)
        .ok()
        .and_then(|s| s.trim().parse().ok())
        .unwrap_or(0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_hugepagesize_kb() {
        let meminfo = "\
MemTotal:       131875328 kB
MemFree:         60023456 kB
Hugepagesize:       2048 kB
Hugetlb:         2097152 kB
";
        assert_eq!(parse_hugepagesize(meminfo), Some(2 * 1024 * 1024));
    }

    #[test]
    fn parse_hugepagesize_missing() {
        assert_eq!(parse_hugepagesize("MemTotal: 1 kB"), None);
    }

    #[test]
    fn parse_thp_enabled_madvise_active() {
        let s = "always [madvise] never\n";
        assert_eq!(parse_thp_enabled(s), ThpMode::Madvise);
    }

    #[test]
    fn parse_thp_enabled_always_active() {
        assert_eq!(parse_thp_enabled("[always] madvise never"), ThpMode::Always);
    }

    #[test]
    fn parse_thp_enabled_never_active() {
        assert_eq!(parse_thp_enabled("always madvise [never]"), ThpMode::Never);
    }

    #[test]
    fn parse_thp_enabled_unbracketed_is_unknown() {
        assert_eq!(parse_thp_enabled("always madvise never"), ThpMode::Unknown);
    }

    #[test]
    fn pool_capacity_math() {
        let p = HugepagePool {
            page_size_bytes: 2 * 1024 * 1024,
            nr_pages: 16,
            free_pages: 4,
            resv_pages: 0,
            surplus_pages: 0,
        };
        assert_eq!(p.capacity_bytes(), 32 * 1024 * 1024);
        assert_eq!(p.free_bytes(), 8 * 1024 * 1024);
    }

    #[test]
    fn capacity_lookups() {
        let info = HugepageInfo {
            default_size_bytes: 2 * 1024 * 1024,
            thp_enabled: ThpMode::Madvise,
            pools: vec![HugepagePool {
                page_size_bytes: 2 * 1024 * 1024,
                nr_pages: 64,
                free_pages: 64,
                resv_pages: 0,
                surplus_pages: 0,
            }],
            per_node: vec![PerNodeHugepages {
                node: NumaNode(0),
                pools: vec![HugepagePool {
                    page_size_bytes: 2 * 1024 * 1024,
                    nr_pages: 32,
                    free_pages: 32,
                    resv_pages: 0,
                    surplus_pages: 0,
                }],
            }],
        };
        assert_eq!(info.capacity_bytes(2 * 1024 * 1024), 128 * 1024 * 1024);
        assert_eq!(info.capacity_bytes(1024 * 1024 * 1024), 0);
        assert_eq!(
            info.capacity_bytes_on_node(NumaNode(0), 2 * 1024 * 1024),
            64 * 1024 * 1024
        );
        assert_eq!(info.capacity_bytes_on_node(NumaNode(1), 2 * 1024 * 1024), 0);
    }
}
