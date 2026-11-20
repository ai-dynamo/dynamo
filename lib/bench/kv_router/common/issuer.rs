// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#[cfg(target_os = "linux")]
use std::collections::BTreeMap;

/// Assign a dense logical worker/source ID to one contiguous issuer shard.
pub fn contiguous_worker_issuer(
    worker_id: usize,
    logical_workers: usize,
    issuer_count: usize,
) -> usize {
    debug_assert!(issuer_count > 0);
    debug_assert!(logical_workers > 0);
    debug_assert!(worker_id < logical_workers);
    let workers_per_issuer = logical_workers.div_ceil(issuer_count).max(1);
    (worker_id / workers_per_issuer).min(issuer_count - 1)
}

/// Pin the current write-issuer thread to one CPU when the platform supports it.
pub fn pin_current_thread(cpu: Option<usize>) -> std::io::Result<()> {
    let Some(cpu) = cpu else {
        return Ok(());
    };
    pin_current_thread_to_cpus(std::slice::from_ref(&cpu))
}

/// Restrict the current benchmark thread to a set of CPUs when the platform supports it.
pub fn pin_current_thread_to_cpus(cpus: &[usize]) -> std::io::Result<()> {
    if cpus.is_empty() {
        return Ok(());
    }
    #[cfg(target_os = "linux")]
    {
        let mut set = unsafe { std::mem::zeroed::<libc::cpu_set_t>() };
        unsafe {
            libc::CPU_ZERO(&mut set);
            for &cpu in cpus {
                libc::CPU_SET(cpu, &mut set);
            }
        }
        let rc = unsafe {
            libc::sched_setaffinity(
                0,
                std::mem::size_of::<libc::cpu_set_t>(),
                &set as *const libc::cpu_set_t,
            )
        };
        if rc != 0 {
            return Err(std::io::Error::last_os_error());
        }
    }
    #[cfg(not(target_os = "linux"))]
    let _ = cpus;
    Ok(())
}

/// Parse a CPU list such as `0-3,6,8-9` into sorted, unique CPU IDs.
pub fn parse_cpu_list(value: &str) -> anyhow::Result<Vec<usize>> {
    let mut cpus = Vec::new();
    for part in value
        .split(',')
        .map(str::trim)
        .filter(|part| !part.is_empty())
    {
        let Some((start, end)) = part.split_once('-') else {
            cpus.push(part.parse()?);
            continue;
        };
        let start: usize = start.parse()?;
        let end: usize = end.parse()?;
        if start > end {
            anyhow::bail!("invalid descending CPU range {part}");
        }
        cpus.extend(start..=end);
    }
    cpus.sort_unstable();
    cpus.dedup();
    if cpus.is_empty() {
        anyhow::bail!("CPU list must not be empty");
    }
    Ok(cpus)
}

pub fn validate_cpu_partition(
    issuer_cpus: &[usize],
    query_issuer_cpu: Option<usize>,
    backend_cpus: &[usize],
) -> anyhow::Result<()> {
    if backend_cpus.is_empty() {
        if !issuer_cpus.is_empty() || query_issuer_cpu.is_some() {
            anyhow::bail!("--backend-cpus is required when generator CPUs are specified");
        }
        return Ok(());
    }
    if issuer_cpus.iter().any(|cpu| backend_cpus.contains(cpu)) {
        anyhow::bail!("an issuer CPU overlaps the backend CPU set");
    }
    if query_issuer_cpu.is_some_and(|cpu| backend_cpus.contains(&cpu)) {
        anyhow::bail!("query-issuer CPU overlaps the backend CPU set");
    }
    if query_issuer_cpu.is_some_and(|cpu| issuer_cpus.contains(&cpu)) {
        anyhow::bail!("query and event issuer CPUs must be distinct");
    }
    let mut unique_issuers = issuer_cpus.to_vec();
    unique_issuers.sort_unstable();
    unique_issuers.dedup();
    if unique_issuers.len() != issuer_cpus.len() {
        anyhow::bail!("issuer CPUs must be unique");
    }

    #[cfg(target_os = "linux")]
    {
        let allowed = allowed_cpu_set()?;
        let mut selected_cores = BTreeMap::new();
        for cpu in issuer_cpus
            .iter()
            .copied()
            .chain(query_issuer_cpu)
            .chain(backend_cpus.iter().copied())
        {
            if !allowed.contains(&cpu) {
                anyhow::bail!("CPU {cpu} is outside the process affinity mask");
            }
            if let Some(core) = physical_core(cpu)?
                && let Some(existing) = selected_cores.insert(core, cpu)
            {
                anyhow::bail!(
                    "CPUs {existing} and {cpu} are SMT siblings; select one logical CPU per physical core"
                );
            }
        }
    }
    Ok(())
}

#[cfg(target_os = "linux")]
fn allowed_cpu_set() -> anyhow::Result<Vec<usize>> {
    let mut set = unsafe { std::mem::zeroed::<libc::cpu_set_t>() };
    let rc = unsafe {
        libc::sched_getaffinity(
            0,
            std::mem::size_of::<libc::cpu_set_t>(),
            &mut set as *mut libc::cpu_set_t,
        )
    };
    if rc != 0 {
        return Err(std::io::Error::last_os_error().into());
    }
    Ok((0..libc::CPU_SETSIZE as usize)
        .filter(|&cpu| unsafe { libc::CPU_ISSET(cpu, &set) })
        .collect())
}

#[cfg(target_os = "linux")]
fn physical_core(cpu: usize) -> anyhow::Result<Option<(u32, u32)>> {
    let topology = format!("/sys/devices/system/cpu/cpu{cpu}/topology");
    let package_path = format!("{topology}/physical_package_id");
    let core_path = format!("{topology}/core_id");
    if !std::path::Path::new(&package_path).exists() || !std::path::Path::new(&core_path).exists() {
        return Ok(None);
    }
    let package = std::fs::read_to_string(package_path)?.trim().parse()?;
    let core = std::fs::read_to_string(core_path)?.trim().parse()?;
    Ok(Some((package, core)))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn contiguous_partition_keeps_each_worker_on_one_balanced_issuer() {
        let assignments = (0..10)
            .map(|worker| contiguous_worker_issuer(worker, 10, 3))
            .collect::<Vec<_>>();
        assert_eq!(assignments, vec![0, 0, 0, 0, 1, 1, 1, 1, 2, 2]);
    }
}
