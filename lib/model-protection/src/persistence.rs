// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::path::Path;

use rustix::process::{
    DumpableBehavior, Resource, Rlimit, dumpable_behavior, set_dumpable_behavior, setrlimit,
};

use crate::{ProtectionError, Result};

/// Makes core dumping irreversible for this process and requires effective cgroup swap to be off.
pub fn enforce_process_persistence_policy() -> Result<()> {
    setrlimit(
        Resource::Core,
        Rlimit {
            current: Some(0),
            maximum: Some(0),
        },
    )
    .map_err(|_| ProtectionError::HostPolicyInvalid)?;
    set_dumpable_behavior(DumpableBehavior::NotDumpable)
        .map_err(|_| ProtectionError::HostPolicyInvalid)?;
    if dumpable_behavior().map_err(|_| ProtectionError::HostPolicyInvalid)?
        != DumpableBehavior::NotDumpable
    {
        return Err(ProtectionError::HostPolicyInvalid);
    }

    let cgroup = std::fs::read_to_string("/proc/self/cgroup")
        .map_err(|_| ProtectionError::HostPolicyInvalid)?;
    let relative = cgroup
        .lines()
        .find_map(|line| line.strip_prefix("0::/"))
        .ok_or(ProtectionError::HostPolicyInvalid)?;
    if !relative.is_empty()
        && relative
            .split('/')
            .any(|part| matches!(part, "" | "." | ".."))
    {
        return Err(ProtectionError::HostPolicyInvalid);
    }
    let root = Path::new("/sys/fs/cgroup");
    require_swap_disabled(root, &root.join(relative))
}

fn require_swap_disabled(base: &Path, leaf: &Path) -> Result<()> {
    let current = read_number(&leaf.join("memory.swap.current"))?;
    if current != 0 {
        return Err(ProtectionError::HostPolicyInvalid);
    }

    let mut effective_limit = None;
    let mut path = leaf;
    loop {
        let maximum = match std::fs::read_to_string(path.join("memory.swap.max")) {
            Ok(maximum) => maximum,
            Err(error) if error.kind() == std::io::ErrorKind::NotFound && path != leaf => break,
            Err(_) => return Err(ProtectionError::HostPolicyInvalid),
        };
        if maximum.trim() != "max" {
            let maximum = maximum
                .trim()
                .parse::<u64>()
                .map_err(|_| ProtectionError::HostPolicyInvalid)?;
            effective_limit =
                Some(effective_limit.map_or(maximum, |limit: u64| limit.min(maximum)));
        }
        if path == base {
            break;
        }
        path = path
            .parent()
            .filter(|parent| parent.starts_with(base))
            .ok_or(ProtectionError::HostPolicyInvalid)?;
    }
    if effective_limit != Some(0) {
        return Err(ProtectionError::HostPolicyInvalid);
    }
    Ok(())
}

fn read_number(path: &Path) -> Result<u64> {
    std::fs::read_to_string(path)
        .map_err(|_| ProtectionError::HostPolicyInvalid)?
        .trim()
        .parse()
        .map_err(|_| ProtectionError::HostPolicyInvalid)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn swap_policy_uses_effective_ancestor_limit() {
        let base = tempfile::tempdir().unwrap();
        let leaf = base.path().join("parent/leaf");
        std::fs::create_dir_all(&leaf).unwrap();
        std::fs::write(leaf.join("memory.swap.current"), "0").unwrap();
        for (path, limit) in [
            (base.path(), "max"),
            (leaf.parent().unwrap(), "0"),
            (leaf.as_path(), "max"),
        ] {
            std::fs::write(path.join("memory.swap.max"), limit).unwrap();
        }
        assert!(require_swap_disabled(base.path(), &leaf).is_ok());
        std::fs::write(leaf.parent().unwrap().join("memory.swap.max"), "1024").unwrap();
        assert!(matches!(
            require_swap_disabled(base.path(), &leaf),
            Err(ProtectionError::HostPolicyInvalid)
        ));
    }
}
