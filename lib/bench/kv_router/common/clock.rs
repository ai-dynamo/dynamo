// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Deadline clock shared by the KV-router open-loop replay benchmarks.
//!
//! On Linux, `wait_until` sleeps to an absolute `CLOCK_MONOTONIC` deadline and
//! then spins for the final `spin_ns`. Elsewhere it falls back to a relative
//! sleep plus spin, which is not authoritative for timing results.

use std::time::Instant;

pub struct BenchmarkClock {
    epoch: Instant,
    spin_ns: u64,
    #[cfg(target_os = "linux")]
    monotonic_epoch_ns: u64,
}

impl BenchmarkClock {
    pub fn new(spin_ns: u64) -> anyhow::Result<Self> {
        #[cfg(target_os = "linux")]
        let monotonic_epoch_ns = monotonic_now_ns()?;
        Ok(Self {
            epoch: Instant::now(),
            spin_ns,
            #[cfg(target_os = "linux")]
            monotonic_epoch_ns,
        })
    }

    #[inline]
    pub fn epoch(&self) -> Instant {
        self.epoch
    }

    // `now_ns` and `wait_until` run on the issuer's spin path from other
    // crates (bench and test binaries); `#[inline]` keeps them inlinable there.
    #[inline]
    pub fn now_ns(&self) -> u64 {
        self.epoch.elapsed().as_nanos().min(u64::MAX as u128) as u64
    }

    #[inline]
    pub fn wait_until(&self, target_ns: u64) {
        let sleep_target = target_ns.saturating_sub(self.spin_ns);
        let now = self.now_ns();
        #[cfg(target_os = "linux")]
        if sleep_target > now {
            sleep_until_monotonic(self.monotonic_epoch_ns.saturating_add(sleep_target));
        }
        #[cfg(not(target_os = "linux"))]
        if sleep_target > now {
            std::thread::sleep(std::time::Duration::from_nanos(sleep_target - now));
        }
        while self.now_ns() < target_ns {
            std::hint::spin_loop();
        }
    }

    pub fn timer_name(&self) -> &'static str {
        if cfg!(target_os = "linux") {
            "clock_nanosleep_monotonic_absolute"
        } else {
            "portable_sleep_spin_non_authoritative"
        }
    }
}

#[cfg(target_os = "linux")]
fn monotonic_now_ns() -> anyhow::Result<u64> {
    let mut timestamp = libc::timespec {
        tv_sec: 0,
        tv_nsec: 0,
    };
    let rc = unsafe { libc::clock_gettime(libc::CLOCK_MONOTONIC, &mut timestamp) };
    if rc != 0 {
        return Err(std::io::Error::last_os_error().into());
    }
    Ok((timestamp.tv_sec as u64)
        .saturating_mul(1_000_000_000)
        .saturating_add(timestamp.tv_nsec as u64))
}

#[cfg(target_os = "linux")]
fn sleep_until_monotonic(target_ns: u64) {
    let request = libc::timespec {
        tv_sec: (target_ns / 1_000_000_000) as libc::time_t,
        tv_nsec: (target_ns % 1_000_000_000) as libc::c_long,
    };
    loop {
        let rc = unsafe {
            libc::clock_nanosleep(
                libc::CLOCK_MONOTONIC,
                libc::TIMER_ABSTIME,
                &request,
                std::ptr::null_mut(),
            )
        };
        if rc == 0 {
            return;
        }
        if rc != libc::EINTR {
            return;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::BenchmarkClock;

    // Only lower bounds are asserted; upper bounds are flaky on shared CI.
    #[test]
    fn wait_until_never_returns_before_target() {
        let spin_ns = 200_000;
        let clock = BenchmarkClock::new(spin_ns).unwrap();

        // Past the spin window: sleeps, then spins.
        let sleep_target = clock.now_ns() + 5_000_000;
        clock.wait_until(sleep_target);
        let after_sleep = clock.now_ns();
        assert!(after_sleep >= sleep_target);

        // Inside the spin window: spins only.
        let spin_target = clock.now_ns() + spin_ns / 2;
        clock.wait_until(spin_target);
        let after_spin = clock.now_ns();
        assert!(after_spin >= spin_target);
        assert!(after_spin >= after_sleep);

        // A target already in the past returns without waiting.
        clock.wait_until(0);
        assert!(clock.now_ns() >= after_spin);
    }
}
