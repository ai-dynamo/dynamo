// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::future::Future;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::task::{Wake, Waker};
use std::time::Duration;

use anyhow::{Context, Result, ensure};
use tokio::runtime::{EnterGuard, Handle, Runtime};
use tokio::time::Instant;

/// Runs the native scheduler on the simulator's clock, independent of host CPU time.
pub(super) struct ReplayRuntime {
    runtime: Runtime,
    epoch: Instant,
    elapsed: Duration,
}

impl ReplayRuntime {
    pub(super) fn new() -> Result<Self> {
        ensure_outside_runtime()?;
        let runtime = tokio::runtime::Builder::new_current_thread()
            .enable_time()
            .start_paused(true)
            .build()
            .context("could not create the native router runtime")?;
        let epoch = {
            let _guard = runtime.enter();
            Instant::now()
        };
        Ok(Self {
            runtime,
            epoch,
            elapsed: Duration::ZERO,
        })
    }

    pub(super) fn epoch(&self) -> Instant {
        self.epoch
    }

    #[cfg(test)]
    pub(super) fn now_ms(&self) -> f64 {
        self.elapsed.as_secs_f64() * 1000.0
    }

    pub(super) fn enter(&self) -> EnterGuard<'_> {
        self.runtime.enter()
    }

    /// Only drive operations that can finish at the current simulation timestamp.
    pub(super) fn block_on<F: Future>(&self, future: F) -> Result<F::Output> {
        ensure_outside_runtime()?;
        let (output, now) = self
            .runtime
            .block_on(async { (future.await, Instant::now()) });
        ensure!(
            now.duration_since(self.epoch) == self.elapsed,
            "native router operation advanced time outside the simulation event loop"
        );
        Ok(output)
    }

    pub(super) fn advance_to_ms(&mut self, now_ms: f64) -> Result<()> {
        ensure_outside_runtime()?;
        ensure!(
            now_ms.is_finite() && now_ms >= 0.0,
            "invalid native router time {now_ms}"
        );
        let elapsed = Duration::try_from_secs_f64(now_ms / 1000.0)
            .context("native router time exceeds the supported duration")?;
        ensure!(
            elapsed >= self.elapsed,
            "native router time moved backwards"
        );
        self.epoch
            .checked_add(elapsed)
            .context("native router time exceeds the supported clock range")?;
        let delta = elapsed - self.elapsed;
        self.runtime.block_on(tokio::time::advance(delta));
        self.elapsed = elapsed;
        Ok(())
    }
}

fn ensure_outside_runtime() -> Result<()> {
    ensure!(
        Handle::try_current().is_err(),
        "native replay must run outside an active Tokio runtime; run the complete synchronous replay on a dedicated thread"
    );
    Ok(())
}

/// Records native future notifications without creating simulator wakeups or timers.
#[derive(Default)]
pub(super) struct WakeFlag(AtomicBool);

impl WakeFlag {
    pub(super) fn take(&self) -> bool {
        self.0.swap(false, Ordering::AcqRel)
    }

    pub(super) fn waker(self: &Arc<Self>) -> Waker {
        Waker::from(Arc::clone(self))
    }
}

impl Wake for WakeFlag {
    fn wake(self: Arc<Self>) {
        self.wake_by_ref();
    }

    fn wake_by_ref(self: &Arc<Self>) {
        self.0.store(true, Ordering::Release);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn host_delay_does_not_advance_simulation_time() {
        let mut runtime = ReplayRuntime::new().unwrap();
        std::thread::sleep(Duration::from_millis(5));
        assert_eq!(
            runtime.block_on(async { Instant::now() }).unwrap(),
            runtime.epoch()
        );
        runtime.advance_to_ms(25.5).unwrap();
        assert_eq!(runtime.now_ms(), 25.5);
        assert_eq!(
            runtime.block_on(async { Instant::now() }).unwrap() - runtime.epoch(),
            Duration::from_micros(25_500)
        );
        assert!(runtime.advance_to_ms(24.0).is_err());
        assert!(runtime.advance_to_ms(f64::NAN).is_err());
    }

    #[test]
    fn nested_runtime_and_unmodeled_timer_wait_return_errors() {
        let runtime = ReplayRuntime::new().unwrap();
        let outer = tokio::runtime::Builder::new_current_thread()
            .build()
            .unwrap();
        outer.block_on(async {
            assert!(ReplayRuntime::new().is_err());
            assert!(runtime.block_on(async {}).is_err());
        });
        assert!(
            runtime
                .block_on(async { tokio::time::sleep(Duration::from_secs(1)).await })
                .is_err()
        );
    }
}
