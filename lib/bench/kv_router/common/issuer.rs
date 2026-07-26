// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

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
