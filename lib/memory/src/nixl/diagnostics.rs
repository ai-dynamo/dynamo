// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Diagnostics for NIXL/UCX backend initialization failures.
//!
//! When the UCX backend fails to initialize (e.g. a bare `NIXL_ERR_BACKEND`),
//! the underlying cause is frequently that UCX bound the wrong RDMA NICs:
//! with `UCX_NET_DEVICES` unset or set to `all`, UCX may pick up management or
//! BlueField NICs that are not wired to the GPUs, which breaks multi-GPU /
//! multi-NIC `rc` initialization. The fix is to scope `UCX_NET_DEVICES` to the
//! per-GPU NICs (e.g. `mlx5_0:1,mlx5_1:1,...`).
//!
//! This module produces an actionable, copy/paste-ready hint to append to the
//! propagated error. Everything here is read-only inspection of sysfs and the
//! process environment — no external processes are spawned.

use std::fs;
use std::path::{Path, PathBuf};

const INFINIBAND_SYSFS: &str = "/sys/class/infiniband";
const PCI_DEVICES_SYSFS: &str = "/sys/bus/pci/devices";

/// NVIDIA PCI vendor id, used to identify GPUs in sysfs.
const NVIDIA_VENDOR_ID: &str = "0x10de";

/// An RDMA device port that is reported as `ACTIVE`.
#[derive(Debug, Clone, PartialEq, Eq)]
struct ActivePort {
    /// Device name, e.g. `mlx5_0`.
    device: String,
    /// Port number, e.g. `1`.
    port: u32,
}

impl ActivePort {
    /// Format as a `UCX_NET_DEVICES` entry, e.g. `mlx5_0:1`.
    fn ucx_entry(&self) -> String {
        format!("{}:{}", self.device, self.port)
    }
}

/// Build a diagnostic hint for a UCX backend initialization failure, reading
/// the real sysfs and process environment.
///
/// The returned string is intended to be appended to the propagated error.
pub fn ucx_backend_failure_hint() -> String {
    build_ucx_hint(
        Path::new(INFINIBAND_SYSFS),
        Path::new(PCI_DEVICES_SYSFS),
        &|name| std::env::var(name).ok(),
    )
}

/// Inner implementation with injectable sysfs roots and environment lookup so
/// it can be exercised against fixtures in tests.
fn build_ucx_hint(ib_root: &Path, pci_root: &Path, env: &dyn Fn(&str) -> Option<String>) -> String {
    let active = active_ports(ib_root);
    let gpu_affine = gpu_affine_ports(ib_root, pci_root, &active);

    let net_devices = env("UCX_NET_DEVICES");
    let net_devices_is_default = net_devices
        .as_deref()
        .map(|v| v.eq_ignore_ascii_case("all"))
        .unwrap_or(true);

    let mut out = String::new();
    out.push_str("\n\n--- UCX_NET_DEVICES diagnostics ---\n");
    out.push_str(
        "UCX/NIXL backend creation failed. A common cause is UCX binding the wrong RDMA NICs \
         (e.g. management or BlueField NICs) when UCX_NET_DEVICES is unset or set to 'all'.\n",
    );

    // Show the current UCX settings that are relevant to NIC selection.
    out.push_str("\nCurrent UCX settings:\n");
    for var in ["UCX_NET_DEVICES", "UCX_TLS", "UCX_IB_ADDR_TYPE"] {
        match env(var) {
            Some(v) => out.push_str(&format!("  {var}={v}\n")),
            None => out.push_str(&format!("  {var}=<unset>\n")),
        }
    }

    if active.is_empty() {
        out.push_str(
            "\nNo active RDMA ports were found under /sys/class/infiniband. Verify that RDMA \
             devices are injected into the container (e.g. request `rdma/ib`) and that at least \
             one port is in the ACTIVE state (`ibstat`, `ibv_devinfo`).\n",
        );
        out.push_str(doc_pointer());
        return out;
    }

    let all_list = join_entries(&active);
    out.push_str(&format!(
        "\nActive RDMA ports visible in this container:\n  {all_list}\n",
    ));

    if !gpu_affine.is_empty() && gpu_affine.len() != active.len() {
        // We could distinguish GPU-affine NICs from the rest: recommend that subset.
        let affine_list = join_entries(&gpu_affine);
        out.push_str(&format!(
            "\nRecommended (GPU-affine NICs, derived from PCIe topology):\n  \
             UCX_NET_DEVICES={affine_list}\n",
        ));
        out.push_str(
            "\nThe remaining active ports above are not paired with a GPU and are likely \
             management/BlueField NICs. Verify the pairing with `nvidia-smi topo -m`.\n",
        );
    } else {
        // Either every active NIC is GPU-affine, or affinity could not be
        // determined; fall back to the full active list and tell the user to prune.
        out.push_str(&format!(
            "\nRecommended: set UCX_NET_DEVICES to the per-GPU NICs (prune any \
             management/BlueField NICs):\n  UCX_NET_DEVICES={all_list}\n",
        ));
        out.push_str("\nUse `nvidia-smi topo -m` to confirm which NICs are paired with GPUs.\n");
    }

    if !net_devices_is_default {
        out.push_str(
            "\nNote: UCX_NET_DEVICES is already set to a specific value above; if it is wrong \
             for this node, replace it with the recommendation.\n",
        );
    }

    out.push_str(doc_pointer());
    out
}

fn doc_pointer() -> &'static str {
    "\nSee docs/kubernetes/disagg-communication-guide.md (\"UCX_NET_DEVICES\") for details.\n"
}

/// Join active ports into a comma-separated `UCX_NET_DEVICES` value.
fn join_entries(ports: &[ActivePort]) -> String {
    ports
        .iter()
        .map(ActivePort::ucx_entry)
        .collect::<Vec<_>>()
        .join(",")
}

/// Enumerate active RDMA ports under `<ib_root>/<device>/ports/<n>/state`.
///
/// A port is considered active when its `state` file contains `ACTIVE`
/// (the file looks like `4: ACTIVE`). Returned ports are sorted naturally by
/// device name (so `mlx5_2` precedes `mlx5_10`) and then by port number.
fn active_ports(ib_root: &Path) -> Vec<ActivePort> {
    let mut ports = Vec::new();

    let Ok(devices) = fs::read_dir(ib_root) else {
        return ports;
    };

    for device in devices.flatten() {
        let device_name = device.file_name().to_string_lossy().to_string();
        let ports_dir = device.path().join("ports");
        let Ok(port_entries) = fs::read_dir(&ports_dir) else {
            continue;
        };

        for port_entry in port_entries.flatten() {
            let Some(port_num) = port_entry.file_name().to_string_lossy().parse::<u32>().ok()
            else {
                continue;
            };

            let state = fs::read_to_string(port_entry.path().join("state")).unwrap_or_default();
            if state.to_ascii_uppercase().contains("ACTIVE") {
                ports.push(ActivePort {
                    device: device_name.clone(),
                    port: port_num,
                });
            }
        }
    }

    ports.sort_by(|a, b| {
        natural_key(&a.device)
            .cmp(&natural_key(&b.device))
            .then(a.port.cmp(&b.port))
    });
    ports
}

/// Natural sort key for device names: splits a trailing integer so that
/// `mlx5_2` sorts before `mlx5_10`.
fn natural_key(name: &str) -> (String, u64) {
    let split = name
        .char_indices()
        .rev()
        .take_while(|(_, c)| c.is_ascii_digit())
        .last()
        .map(|(i, _)| i);

    match split {
        Some(i) => {
            let (prefix, digits) = name.split_at(i);
            (prefix.to_string(), digits.parse().unwrap_or(0))
        }
        None => (name.to_string(), 0),
    }
}

/// Determine which active ports belong to NICs that share a PCIe switch with a
/// GPU. Returns an empty vec when affinity cannot be determined (the caller
/// then falls back to the full active list).
fn gpu_affine_ports(ib_root: &Path, pci_root: &Path, active: &[ActivePort]) -> Vec<ActivePort> {
    let gpu_paths = gpu_pci_paths(pci_root);
    if gpu_paths.is_empty() {
        return Vec::new();
    }

    active
        .iter()
        .filter(|p| {
            nic_pci_path(ib_root, &p.device)
                .map(|nic| gpu_paths.iter().any(|gpu| shares_pcie_switch(&nic, gpu)))
                .unwrap_or(false)
        })
        .cloned()
        .collect()
}

/// Resolve the canonical sysfs path of a NIC's PCI device via the
/// `<ib_root>/<device>/device` symlink.
fn nic_pci_path(ib_root: &Path, device: &str) -> Option<PathBuf> {
    fs::canonicalize(ib_root.join(device).join("device")).ok()
}

/// Enumerate canonical PCI paths of NVIDIA GPUs under `pci_root`.
fn gpu_pci_paths(pci_root: &Path) -> Vec<PathBuf> {
    let mut gpus = Vec::new();
    let Ok(entries) = fs::read_dir(pci_root) else {
        return gpus;
    };

    for entry in entries.flatten() {
        let vendor = fs::read_to_string(entry.path().join("vendor")).unwrap_or_default();
        if vendor.trim() != NVIDIA_VENDOR_ID {
            continue;
        }
        // Display controllers: PCI base class 0x03 (e.g. 0x030000 VGA, 0x030200 3D).
        let class = fs::read_to_string(entry.path().join("class")).unwrap_or_default();
        if !class.trim_start_matches("0x").starts_with("03") {
            continue;
        }
        if let Ok(path) = fs::canonicalize(entry.path()) {
            gpus.push(path);
        }
    }
    gpus
}

/// Two PCI devices share a PCIe switch when their canonical sysfs paths share
/// at least one PCI bus segment (`DDDD:BB:DD.F`) beyond the host bridge root.
/// Devices on separate root complexes (e.g. a management NIC on a different
/// socket) share none.
fn shares_pcie_switch(a: &Path, b: &Path) -> bool {
    let a_segs = pci_bdf_segments(a);
    let b_segs = pci_bdf_segments(b);

    // Count shared leading segments, excluding the final (device) segment of
    // each, so that only shared *upstream* switches/bridges count.
    let a_upstream = a_segs.len().saturating_sub(1);
    let b_upstream = b_segs.len().saturating_sub(1);
    let limit = a_upstream.min(b_upstream);

    (0..limit).take_while(|&i| a_segs[i] == b_segs[i]).count() > 0
}

/// Extract the ordered PCI bus-device-function segments (`DDDD:BB:DD.F`) from a
/// sysfs path.
fn pci_bdf_segments(path: &Path) -> Vec<String> {
    path.components()
        .filter_map(|c| c.as_os_str().to_str())
        .filter(|s| is_pci_bdf(s))
        .map(|s| s.to_string())
        .collect()
}

/// Returns true for a PCI bus-device-function token like `0000:3b:00.0`.
fn is_pci_bdf(s: &str) -> bool {
    let parts: Vec<&str> = s.split(':').collect();
    if parts.len() != 3 {
        return false;
    }
    let (domain, bus, devfn) = (parts[0], parts[1], parts[2]);
    let (dev, func) = match devfn.split_once('.') {
        Some(pair) => pair,
        None => return false,
    };
    domain.len() == 4
        && domain.chars().all(|c| c.is_ascii_hexdigit())
        && bus.len() == 2
        && bus.chars().all(|c| c.is_ascii_hexdigit())
        && dev.chars().all(|c| c.is_ascii_hexdigit())
        && !dev.is_empty()
        && func.chars().all(|c| c.is_ascii_hexdigit())
        && !func.is_empty()
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;
    use std::fs;
    use std::path::Path;

    /// Build a fake env lookup from a map.
    fn env_from(pairs: &[(&str, &str)]) -> impl Fn(&str) -> Option<String> {
        let map: HashMap<String, String> = pairs
            .iter()
            .map(|(k, v)| (k.to_string(), v.to_string()))
            .collect();
        move |name: &str| map.get(name).cloned()
    }

    /// Create `<ib_root>/<device>/ports/<port>/state` with the given state text.
    fn make_port(ib_root: &Path, device: &str, port: u32, state: &str) {
        let dir = ib_root.join(device).join("ports").join(port.to_string());
        fs::create_dir_all(&dir).unwrap();
        fs::write(dir.join("state"), state).unwrap();
    }

    /// Symlink `<ib_root>/<device>/device` -> `target`.
    fn link_nic_device(ib_root: &Path, device: &str, target: &Path) {
        let dev_dir = ib_root.join(device);
        fs::create_dir_all(&dev_dir).unwrap();
        std::os::unix::fs::symlink(target, dev_dir.join("device")).unwrap();
    }

    /// Create a fake PCI device dir at `<pci_root>/<addr>` resolving (via the
    /// canonical `pci_devices` tree) to a path containing `bdf_chain`.
    /// Returns the canonical leaf path.
    fn make_pci_device(
        root: &Path,
        pci_root: &Path,
        addr: &str,
        bdf_chain: &[&str],
        vendor: Option<&str>,
        class: Option<&str>,
    ) -> std::path::PathBuf {
        // Real hierarchy under <root>/devices/pci0000:00/<bdf_chain...>
        let mut leaf = root.join("devices").join("pci0000:00");
        for seg in bdf_chain {
            leaf = leaf.join(seg);
        }
        fs::create_dir_all(&leaf).unwrap();
        if let Some(v) = vendor {
            fs::write(leaf.join("vendor"), v).unwrap();
        }
        if let Some(c) = class {
            fs::write(leaf.join("class"), c).unwrap();
        }
        // Symlink <pci_root>/<addr> -> leaf
        fs::create_dir_all(pci_root).unwrap();
        std::os::unix::fs::symlink(&leaf, pci_root.join(addr)).unwrap();
        fs::canonicalize(pci_root.join(addr)).unwrap()
    }

    #[test]
    fn active_ports_filters_down_ports_and_sorts_naturally() {
        let tmp = tempfile::tempdir().unwrap();
        let ib = tmp.path().join("ib");
        make_port(&ib, "mlx5_10", 1, "4: ACTIVE");
        make_port(&ib, "mlx5_2", 1, "4: ACTIVE");
        make_port(&ib, "mlx5_0", 1, "4: ACTIVE");
        make_port(&ib, "mlx5_1", 1, "1: DOWN");

        let ports = active_ports(&ib);
        let entries: Vec<String> = ports.iter().map(ActivePort::ucx_entry).collect();
        // mlx5_1 is DOWN and excluded; natural order keeps 2 before 10.
        assert_eq!(entries, vec!["mlx5_0:1", "mlx5_2:1", "mlx5_10:1"]);
    }

    #[test]
    fn hint_recommends_gpu_affine_subset() {
        let tmp = tempfile::tempdir().unwrap();
        let root = tmp.path();
        let ib = root.join("ib");
        let pci = root.join("pci");

        // GPU under switch 0000:0a:00.0 -> 0000:0b:00.0 -> 0000:0c:00.0
        make_pci_device(
            root,
            &pci,
            "0000:0c:00.0",
            &[
                "0000:09:00.0",
                "0000:0a:00.0",
                "0000:0b:00.0",
                "0000:0c:00.0",
            ],
            Some(NVIDIA_VENDOR_ID),
            Some("0x030200"),
        );
        // GPU-affine NIC under the same switch 0000:0a:00.0
        let nic0 = make_pci_device(
            root,
            &pci,
            "0000:0d:00.0",
            &[
                "0000:09:00.0",
                "0000:0a:00.0",
                "0000:0b:01.0",
                "0000:0d:00.0",
            ],
            Some("0x15b3"),
            Some("0x020700"),
        );
        // Management NIC on a different root port (shares nothing past root)
        let nic1 = make_pci_device(
            root,
            &pci,
            "0000:81:00.0",
            &["0000:80:00.0", "0000:81:00.0"],
            Some("0x15b3"),
            Some("0x020700"),
        );

        link_nic_device(&ib, "mlx5_0", &nic0);
        link_nic_device(&ib, "mlx5_5", &nic1);
        make_port(&ib, "mlx5_0", 1, "4: ACTIVE");
        make_port(&ib, "mlx5_5", 1, "4: ACTIVE");

        let env = env_from(&[("UCX_NET_DEVICES", "all")]);
        let hint = build_ucx_hint(&ib, &pci, &env);

        assert!(
            hint.contains("UCX_NET_DEVICES=mlx5_0:1\n"),
            "hint was: {hint}"
        );
        assert!(hint.contains("GPU-affine"), "hint was: {hint}");
        assert!(hint.contains("management/BlueField"), "hint was: {hint}");
    }

    #[test]
    fn hint_falls_back_to_all_ports_without_affinity() {
        let tmp = tempfile::tempdir().unwrap();
        let root = tmp.path();
        let ib = root.join("ib");
        let pci = root.join("pci"); // empty -> no GPUs discoverable

        make_port(&ib, "mlx5_0", 1, "4: ACTIVE");
        make_port(&ib, "mlx5_1", 1, "4: ACTIVE");

        let env = env_from(&[]);
        let hint = build_ucx_hint(&ib, &pci, &env);

        assert!(
            hint.contains("UCX_NET_DEVICES=mlx5_0:1,mlx5_1:1"),
            "hint was: {hint}"
        );
        assert!(hint.contains("prune any"), "hint was: {hint}");
        assert!(hint.contains("UCX_NET_DEVICES=<unset>"), "hint was: {hint}");
    }

    #[test]
    fn hint_handles_no_active_ports() {
        let tmp = tempfile::tempdir().unwrap();
        let root = tmp.path();
        let ib = root.join("ib");
        fs::create_dir_all(&ib).unwrap();
        let pci = root.join("pci");

        let env = env_from(&[("UCX_NET_DEVICES", "all"), ("UCX_TLS", "rc,cuda_copy")]);
        let hint = build_ucx_hint(&ib, &pci, &env);

        assert!(hint.contains("No active RDMA ports"), "hint was: {hint}");
        assert!(hint.contains("UCX_TLS=rc,cuda_copy"), "hint was: {hint}");
    }

    #[test]
    fn is_pci_bdf_matches_real_tokens() {
        assert!(is_pci_bdf("0000:3b:00.0"));
        assert!(is_pci_bdf("0000:0a:00.0"));
        assert!(!is_pci_bdf("pci0000:00"));
        assert!(!is_pci_bdf("ports"));
        assert!(!is_pci_bdf("mlx5_0"));
    }
}
